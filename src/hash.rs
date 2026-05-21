// SPDX-License-Identifier: MIT OR Apache-2.0
//! Hash-to-field and challenge-scalar helpers for SQIsign.
//!
//! SQIsign uses SHAKE-256 throughout — for challenge derivation, for
//! deterministic-RNG seeding inside the signing pipeline, and for
//! hash-to-curve work that maps a public key + message + commitment-curve
//! `j`-invariants into a challenge scalar (see `hash_to_challenge` in the
//! reference's `src/verification/ref/lvlx/common.c`).
//!
//! This module provides the level-agnostic building blocks:
//!
//! - [`Shake256`] — incremental SHAKE-256 wrapper over `shake::Shake256`.
//! - [`hash_to_fp`] — rejection-sample SHAKE bytes into an `F_p` element.
//! - [`hash_to_fp2`] — pair of `hash_to_fp` calls into `F_{p^2}`.
//!
//! The actual `hash_to_challenge` (which needs the level's
//! `SQIsign_response_length` and an iterated mask schedule) lands alongside
//! Sign/Verify in a later session; the primitives here are what it sits on.

use shake::Shake256 as RawShake256;
use shake::digest::{ExtendableOutput, Update, XofReader};
use subtle::CtOption;

use crate::gf::fp::BaseField;
use crate::gf::fp2::Fp2;
use crate::params::Params;

/// Incremental SHAKE-256 absorber + extendable-output reader.
///
/// Wraps `sha3::Shake256` with a slightly nicer surface for the
/// `absorb` / `finalize` / `squeeze` pattern the SQIsign reference uses.
#[derive(Clone, Default)]
pub struct Shake256 {
    inner: RawShake256,
}

impl Shake256 {
    /// Fresh SHAKE-256 state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Absorb additional input.
    pub fn absorb(&mut self, data: &[u8]) {
        self.inner.update(data);
    }

    /// Finalize absorption and squeeze `out.len()` bytes.
    pub fn finalize_into(self, out: &mut [u8]) {
        let mut reader = self.inner.finalize_xof();
        reader.read(out);
    }
}

impl core::fmt::Debug for Shake256 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Shake256").finish_non_exhaustive()
    }
}

impl Shake256 {
    /// Finalize this Shake256 state into an [`Shake256Rng`] for use as a
    /// deterministic [`CryptoRng`](rand_core::CryptoRng). The resulting
    /// RNG draws bytes from the unbounded XOF squeeze stream of the
    /// absorbed state.
    ///
    /// Use case (forward-prep for the SQIsign sign flow): callers absorb
    /// `(msg, sk, rng-derived entropy)` into a fresh Shake256 and
    /// finalize into a Shake256Rng. That RNG seeds the downstream
    /// challenge-ideal construction, ensuring the sign output binds to
    /// `msg` deterministically. The dual property — same `(msg, sk,
    /// entropy)` always produces the same RNG byte stream — is verified
    /// by the determinism tests below.
    pub fn into_rng(self) -> Shake256Rng {
        Shake256Rng {
            reader: self.inner.finalize_xof(),
        }
    }
}

/// Deterministic CryptoRng backed by an absorbed Shake256 state.
///
/// Constructed via [`Shake256::into_rng`] or [`Shake256Rng::from_seed`].
/// All bytes consumed from the RNG come from the Shake256 XOF squeeze
/// stream of the absorbed input. Seeding-by-input determinism makes
/// this a useful building block for SQIsign's message-derivation step:
/// hashing `(msg, sk, rng-entropy)` into a Shake256Rng gives a chain-
/// driving RNG whose output binds to all three inputs.
pub struct Shake256Rng {
    reader: shake::Shake256Reader,
}

impl Shake256Rng {
    /// Construct a Shake256Rng directly from a byte seed. Equivalent to
    /// `Shake256::new().absorb(seed).into_rng()`.
    pub fn from_seed(seed: &[u8]) -> Self {
        let mut h = Shake256::new();
        h.absorb(seed);
        h.into_rng()
    }
}

impl core::fmt::Debug for Shake256Rng {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Don't leak XOF state; CryptoRng best practice per rand_core 0.10 docs.
        f.debug_struct("Shake256Rng").finish_non_exhaustive()
    }
}

impl rand_core::TryRng for Shake256Rng {
    type Error = core::convert::Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        let mut buf = [0u8; 4];
        self.reader.read(&mut buf);
        Ok(u32::from_le_bytes(buf))
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let mut buf = [0u8; 8];
        self.reader.read(&mut buf);
        Ok(u64::from_le_bytes(buf))
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        self.reader.read(dst);
        Ok(())
    }
}

impl rand_core::TryCryptoRng for Shake256Rng {}

/// Maximum buffer the rejection loop will allocate on the stack per attempt.
/// 80 bytes covers Level-5's 64-byte `Fp` encoding plus a generous margin
/// for masking; bigger primes can adjust if SQIsign ever lands at a
/// post-quantum-512+ tier.
const MAX_FIELD_BYTES: usize = 80;

/// Rejection-sample SHAKE-256 bytes into a single `F_p` element.
///
/// Algorithm: absorb `domain` + `input`; squeeze `F::ENCODED_BYTES`
/// little-endian bytes; mask the high bits beyond the prime's bit-length
/// (per the reference's `mask = (1 << bits) − 1` pattern); attempt to
/// decode. If non-canonical (`≥ p`), reseed with the rejected value + a
/// counter and try again. Bounded by `max_iters` tries.
///
/// Returns `None` if `max_iters` is exhausted (probability < 2^{-max_iters}
/// in practice).
pub fn hash_to_fp<F: BaseField>(domain: &[u8], input: &[u8], max_iters: u8) -> CtOption<F> {
    let n = F::ENCODED_BYTES;
    debug_assert!(n <= MAX_FIELD_BYTES);
    let bits = F::BIT_LENGTH;
    // Mask the unused high bits of the top byte so the squeezed integer
    // lives in [0, 2^bits). Acceptance probability is then `p / 2^bits` ∈
    // [1/2, 1], i.e. typically one or two iterations suffice.
    let top_byte_keep = bits % 8;
    let top_byte_mask: u8 = if top_byte_keep == 0 {
        0xff
    } else {
        // top byte has `top_byte_keep` low bits valid (since the integer
        // is little-endian, the highest byte index holds the most-significant
        // bits at position `bits − 1` and below).
        (1u8 << top_byte_keep) - 1
    };
    let mut buf = [0u8; MAX_FIELD_BYTES];
    for ctr in 0..max_iters {
        let mut h = Shake256::new();
        h.absorb(b"pq-sqisign/hash_to_fp");
        h.absorb(domain);
        h.absorb(&[ctr]);
        h.absorb(input);
        h.finalize_into(&mut buf[..n]);
        buf[n - 1] &= top_byte_mask;
        // Caller's BaseField::from_bytes_le rejects values ≥ p.
        let opt = F::from_bytes_le(&buf[..n]);
        if bool::from(opt.is_some()) {
            return opt;
        }
    }
    CtOption::new(F::zero(), subtle::Choice::from(0))
}

/// Pair of `hash_to_fp` calls — produces a uniformly-distributed
/// `F_{p^2}` element. The `re` and `im` components are independently
/// sampled by absorbing a leading component-tag byte alongside the caller's
/// `domain` so the two `hash_to_fp` derivations diverge.
pub fn hash_to_fp2<F: BaseField>(domain: &[u8], input: &[u8], max_iters: u8) -> CtOption<Fp2<F>> {
    let re_opt = hash_to_fp_with_tag::<F>(b'r', domain, input, max_iters);
    let im_opt = hash_to_fp_with_tag::<F>(b'i', domain, input, max_iters);
    let is_some = re_opt.is_some() & im_opt.is_some();
    let re = re_opt.unwrap_or(F::zero());
    let im = im_opt.unwrap_or(F::zero());
    CtOption::new(Fp2::new(re, im), is_some)
}

/// SQIsign challenge-scalar derivation — matches the upstream reference's
/// `hash_to_challenge` (`src/verification/ref/lvlx/common.c`).
///
/// Pipeline:
/// 1. Absorb `j_pk_bytes || j_com_bytes || message` into SHAKE-256.
/// 2. Squeeze `(2 · P::SECURITY_BITS) / 8` bytes — the initial scalar.
/// 3. Re-absorb that scalar and re-squeeze for `P::HASH_ITERATIONS − 2`
///    rounds; this "thickening" exists to slow down sign-and-verify-grinding
///    attacks on the challenge oracle.
/// 4. Write the final scalar bytes into `scalar_out`.
///
/// `scalar_out.len()` must be exactly `(2 · P::SECURITY_BITS) / 8`
/// (32/48/64 bytes for Levels 1/3/5).
///
/// All three SQIsign primes have `2 · SECURITY_BITS` divisible by 8 (and
/// in fact by 64), so the top-byte-mask trick the reference uses is the
/// identity here and is omitted.
pub fn hash_to_challenge_scalar<P: Params>(
    j_pk_bytes: &[u8],
    j_com_bytes: &[u8],
    message: &[u8],
    scalar_out: &mut [u8],
) {
    let scalar_len = (2 * P::SECURITY_BITS) / 8;
    debug_assert!(
        scalar_out.len() >= scalar_len,
        "scalar_out too small for level"
    );
    debug_assert!(
        j_pk_bytes.len() >= P::FP2_BYTES,
        "j_pk_bytes too small for level"
    );
    debug_assert!(
        j_com_bytes.len() >= P::FP2_BYTES,
        "j_com_bytes too small for level"
    );
    // Round 1: absorb (j_pk, j_com, message), squeeze.
    let mut h = Shake256::new();
    h.absorb(&j_pk_bytes[..P::FP2_BYTES]);
    h.absorb(&j_com_bytes[..P::FP2_BYTES]);
    h.absorb(message);
    h.finalize_into(&mut scalar_out[..scalar_len]);
    // Rounds 2..HASH_ITERATIONS: rehash the scalar.
    for _ in 2..P::HASH_ITERATIONS {
        let mut h2 = Shake256::new();
        let tmp: [u8; 80] = {
            let mut t = [0u8; 80];
            t[..scalar_len].copy_from_slice(&scalar_out[..scalar_len]);
            t
        };
        h2.absorb(&tmp[..scalar_len]);
        h2.finalize_into(&mut scalar_out[..scalar_len]);
    }
}

fn hash_to_fp_with_tag<F: BaseField>(
    component: u8,
    domain: &[u8],
    input: &[u8],
    max_iters: u8,
) -> CtOption<F> {
    let n = F::ENCODED_BYTES;
    debug_assert!(n <= MAX_FIELD_BYTES);
    let bits = F::BIT_LENGTH;
    let top_byte_keep = bits % 8;
    let top_byte_mask: u8 = if top_byte_keep == 0 {
        0xff
    } else {
        (1u8 << top_byte_keep) - 1
    };
    let mut buf = [0u8; MAX_FIELD_BYTES];
    for ctr in 0..max_iters {
        let mut h = Shake256::new();
        h.absorb(b"pq-sqisign/hash_to_fp2");
        h.absorb(&[component]);
        h.absorb(domain);
        h.absorb(&[ctr]);
        h.absorb(input);
        h.finalize_into(&mut buf[..n]);
        buf[n - 1] &= top_byte_mask;
        let opt = F::from_bytes_le(&buf[..n]);
        if bool::from(opt.is_some()) {
            return opt;
        }
    }
    CtOption::new(F::zero(), subtle::Choice::from(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf::fp::{Fp1Element, Fp3Element, Fp5Element};

    #[test]
    fn shake256_deterministic() {
        let mut a = Shake256::new();
        a.absorb(b"hello");
        let mut out_a = [0u8; 32];
        a.finalize_into(&mut out_a);
        let mut b = Shake256::new();
        b.absorb(b"hello");
        let mut out_b = [0u8; 32];
        b.finalize_into(&mut out_b);
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn shake256_different_inputs_differ() {
        let mut a = Shake256::new();
        a.absorb(b"hello");
        let mut out_a = [0u8; 32];
        a.finalize_into(&mut out_a);
        let mut b = Shake256::new();
        b.absorb(b"world");
        let mut out_b = [0u8; 32];
        b.finalize_into(&mut out_b);
        assert_ne!(out_a, out_b);
    }

    #[test]
    fn hash_to_fp_returns_some_at_each_level() {
        let a = hash_to_fp::<Fp1Element>(b"dom", b"input", 16);
        assert!(bool::from(a.is_some()));
        let b = hash_to_fp::<Fp3Element>(b"dom", b"input", 16);
        assert!(bool::from(b.is_some()));
        let c = hash_to_fp::<Fp5Element>(b"dom", b"input", 16);
        assert!(bool::from(c.is_some()));
    }

    #[test]
    fn hash_to_fp_deterministic() {
        let a = hash_to_fp::<Fp1Element>(b"dom", b"input", 16);
        let b = hash_to_fp::<Fp1Element>(b"dom", b"input", 16);
        assert_eq!(
            a.unwrap_or(<Fp1Element as BaseField>::zero()),
            b.unwrap_or(<Fp1Element as BaseField>::zero())
        );
    }

    #[test]
    fn hash_to_fp_different_inputs_differ() {
        let a = hash_to_fp::<Fp1Element>(b"dom", b"alpha", 16)
            .unwrap_or(<Fp1Element as BaseField>::zero());
        let b = hash_to_fp::<Fp1Element>(b"dom", b"beta", 16)
            .unwrap_or(<Fp1Element as BaseField>::zero());
        assert_ne!(a, b);
    }

    #[test]
    fn hash_to_fp_domain_separation() {
        // Same input, different domain → different output.
        let a = hash_to_fp::<Fp1Element>(b"dom_a", b"input", 16)
            .unwrap_or(<Fp1Element as BaseField>::zero());
        let b = hash_to_fp::<Fp1Element>(b"dom_b", b"input", 16)
            .unwrap_or(<Fp1Element as BaseField>::zero());
        assert_ne!(a, b);
    }

    #[test]
    fn hash_to_fp2_returns_some() {
        let a = hash_to_fp2::<Fp1Element>(b"dom", b"input", 16);
        assert!(bool::from(a.is_some()));
    }

    #[test]
    fn hash_to_challenge_deterministic_lvl1() {
        use crate::params::Level1;
        let j_pk = [0x11u8; 64];
        let j_com = [0x22u8; 64];
        let mut out_a = [0u8; 32];
        let mut out_b = [0u8; 32];
        hash_to_challenge_scalar::<Level1>(&j_pk, &j_com, b"msg", &mut out_a);
        hash_to_challenge_scalar::<Level1>(&j_pk, &j_com, b"msg", &mut out_b);
        assert_eq!(out_a, out_b);
        assert!(out_a.iter().any(|&b| b != 0));
    }

    #[test]
    fn hash_to_challenge_distinct_messages_lvl1() {
        use crate::params::Level1;
        let j_pk = [0x11u8; 64];
        let j_com = [0x22u8; 64];
        let mut out_a = [0u8; 32];
        let mut out_b = [0u8; 32];
        hash_to_challenge_scalar::<Level1>(&j_pk, &j_com, b"hello", &mut out_a);
        hash_to_challenge_scalar::<Level1>(&j_pk, &j_com, b"world", &mut out_b);
        assert_ne!(out_a, out_b);
    }

    #[test]
    fn hash_to_challenge_distinct_pk_curves_lvl1() {
        use crate::params::Level1;
        let j_pk_a = [0x11u8; 64];
        let mut j_pk_b = [0x11u8; 64];
        j_pk_b[0] = 0x22;
        let j_com = [0x33u8; 64];
        let mut out_a = [0u8; 32];
        let mut out_b = [0u8; 32];
        hash_to_challenge_scalar::<Level1>(&j_pk_a, &j_com, b"msg", &mut out_a);
        hash_to_challenge_scalar::<Level1>(&j_pk_b, &j_com, b"msg", &mut out_b);
        assert_ne!(out_a, out_b);
    }

    #[test]
    fn hash_to_challenge_scales_with_level() {
        use crate::params::{Level3, Level5};
        // Outputs are sized per level.
        let j_pk_lvl3 = [0u8; 96];
        let j_com_lvl3 = [0u8; 96];
        let mut out_lvl3 = [0u8; 48];
        hash_to_challenge_scalar::<Level3>(&j_pk_lvl3, &j_com_lvl3, b"m", &mut out_lvl3);
        assert!(out_lvl3.iter().any(|&b| b != 0));

        let j_pk_lvl5 = [0u8; 128];
        let j_com_lvl5 = [0u8; 128];
        let mut out_lvl5 = [0u8; 64];
        hash_to_challenge_scalar::<Level5>(&j_pk_lvl5, &j_com_lvl5, b"m", &mut out_lvl5);
        assert!(out_lvl5.iter().any(|&b| b != 0));
    }

    #[test]
    fn hash_to_fp2_components_independent() {
        // re and im should differ for the same `(domain, input)` because of
        // internal domain separation.
        let q =
            hash_to_fp2::<Fp1Element>(b"dom", b"input", 16).unwrap_or(Fp2::<Fp1Element>::zero());
        assert_ne!(q.re, q.im);
    }

    // ── S85 — Shake256Rng deterministic CryptoRng tests ──

    #[test]
    fn shake256_rng_from_seed_is_deterministic() {
        // Two RNGs from the same seed must produce identical byte
        // streams. This is the foundational guarantee — sign-flow
        // determinism depends on it.
        let mut rng_a = Shake256Rng::from_seed(b"seed-bytes-for-deterministic-test");
        let mut rng_b = Shake256Rng::from_seed(b"seed-bytes-for-deterministic-test");
        let mut buf_a = [0u8; 64];
        let mut buf_b = [0u8; 64];
        rand_core::Rng::fill_bytes(&mut rng_a, &mut buf_a);
        rand_core::Rng::fill_bytes(&mut rng_b, &mut buf_b);
        assert_eq!(buf_a, buf_b);
    }

    #[test]
    fn shake256_rng_differs_for_different_seeds() {
        // Different seeds → different byte streams.
        let mut rng_a = Shake256Rng::from_seed(b"seed-a");
        let mut rng_b = Shake256Rng::from_seed(b"seed-b");
        let mut buf_a = [0u8; 32];
        let mut buf_b = [0u8; 32];
        rand_core::Rng::fill_bytes(&mut rng_a, &mut buf_a);
        rand_core::Rng::fill_bytes(&mut rng_b, &mut buf_b);
        assert_ne!(buf_a, buf_b);
    }

    #[test]
    fn shake256_rng_via_absorb_matches_from_seed() {
        // Constructing via absorb then into_rng must match
        // from_seed for the same input (they're definitionally
        // equivalent; verifies the from_seed convenience method).
        let mut h = Shake256::new();
        h.absorb(b"the-same-seed");
        let mut rng_a = h.into_rng();
        let mut rng_b = Shake256Rng::from_seed(b"the-same-seed");
        let mut buf_a = [0u8; 32];
        let mut buf_b = [0u8; 32];
        rand_core::Rng::fill_bytes(&mut rng_a, &mut buf_a);
        rand_core::Rng::fill_bytes(&mut rng_b, &mut buf_b);
        assert_eq!(buf_a, buf_b);
    }

    #[test]
    fn shake256_rng_multi_absorb_binds_inputs() {
        // The sign-flow use case: absorb (msg, sk, entropy) in sequence,
        // then into_rng. Different msg → different downstream RNG.
        let mut a = Shake256::new();
        a.absorb(b"msg-a");
        a.absorb(b"sk-bytes");
        a.absorb(b"entropy-bytes");
        let mut rng_a = a.into_rng();

        let mut b = Shake256::new();
        b.absorb(b"msg-b"); // different message
        b.absorb(b"sk-bytes");
        b.absorb(b"entropy-bytes");
        let mut rng_b = b.into_rng();

        let mut buf_a = [0u8; 48];
        let mut buf_b = [0u8; 48];
        rand_core::Rng::fill_bytes(&mut rng_a, &mut buf_a);
        rand_core::Rng::fill_bytes(&mut rng_b, &mut buf_b);
        assert_ne!(buf_a, buf_b);
    }

    #[test]
    fn shake256_rng_next_u32_and_u64_match_fill_bytes() {
        // next_u32 reads 4 LE bytes; next_u64 reads 8 LE bytes. Verify
        // by comparing against a fresh RNG with the same seed read
        // via fill_bytes.
        let seed = b"u32-u64-consistency";
        let mut rng_a = Shake256Rng::from_seed(seed);
        let mut rng_b = Shake256Rng::from_seed(seed);
        let x = rand_core::Rng::next_u32(&mut rng_a);
        let y = rand_core::Rng::next_u64(&mut rng_a);
        let mut buf = [0u8; 12];
        rand_core::Rng::fill_bytes(&mut rng_b, &mut buf);
        let expected_x = u32::from_le_bytes(buf[..4].try_into().expect("4 bytes"));
        let expected_y = u64::from_le_bytes(buf[4..12].try_into().expect("8 bytes"));
        assert_eq!(x, expected_x);
        assert_eq!(y, expected_y);
    }

    // Compile-time proof that Shake256Rng satisfies CryptoRng.
    const _: fn() = || {
        fn requires_crypto_rng<R: rand_core::CryptoRng + ?Sized>() {}
        requires_crypto_rng::<Shake256Rng>();
    };
}
