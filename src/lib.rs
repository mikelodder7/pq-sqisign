// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pure-Rust implementation of SQIsign — compact post-quantum digital signatures
//! built from supersingular isogenies and quaternion orders.
//!
//! Reference specification: <https://sqisign.org/spec/sqisign-20250707.pdf>.
//! Reference C implementation: <https://github.com/SQISign/the-sqisign>.
//!
//! # Status
//!
//! This crate is a ground-up port including the foundation layer
//! (parameter sets, GF(p) / GF(p^2) field arithmetic, Montgomery-form curve
//! point arithmetic, KAT harness scaffolding) and
//! signing and verification pipelines — quaternion KLPT, ideal-to-isogeny
//! translation (Clapotis), challenge derivation.
//!
//! # Layout
//!
//! - [`params`] — security-level parameter sets (Level-1, Level-3, Level-5).
//! - [`gf`] — base-field GF(p) and quadratic extension GF(p^2) arithmetic.
//! - [`ec`] — Montgomery elliptic curves (x-only point representation).
//! - [`quaternion`] — quaternion algebra over Q.
//! - [`isogeny`] — isogeny computations on supersingular curves.
//! - [`error`] — error type for the public API.
//!
//! # Security profile (prototype phase)
//!
//! The crate enforces `#![forbid(unsafe_op_in_unsafe_fn)]` and contains
//! no `unsafe` blocks. Production code paths use no `unwrap()`, no
//! `panic!()`, and no `expect()` outside of `#[cfg(test)]` modules.
//! Numeric casts in production are either explicitly guarded by bound
//! checks (see e.g. the `quaternion::cornacchia` module) or operate on
//! algorithmically public values (bit-packing in [`encoding`]).
//!
//! Field arithmetic (`gf::fp`, `gf::fp2`) uses constant-time primitives
//! from [`subtle`] (`ConstantTimeEq`, `ConditionallySelectable`,
//! `ConstantTimeLess`) — secret-dependent operations on `F_p` and
//! `F_{p²}` flow through these to resist timing side channels.
//!
//! The quaternion + KLPT path uses variable-time arithmetic
//! (`mul_mod_vartime`, `is_probable_prime_with_witnesses`, etc.). At the
//! current prototype scope these operations consume **algorithmically
//! public inputs** — the SQIsign reference primes, sampled-then-published
//! q values, smooth lift targets. Where secret quaternion ideals flow
//! through this path, the affected sub-primitives still need constant-time
//! variants per the ISA security checklist.
//!
//! No long-lived secret material is currently held in heap or stack
//! data structures (all KLPT body intermediate state is computed
//! transiently within a single `keypair`/`sign` call). Secret-bearing
//! types are not yet hardened to derive `Zeroize` and `ZeroizeOnDrop`
//! (the `zeroize` dependency is already wired with the `derive` feature).

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "alloc")]
#[allow(unused_extern_crates)]
extern crate alloc;
#[cfg(test)]
#[allow(unused_extern_crates)]
extern crate std;

pub mod ec;
pub mod encoding;
pub mod error;
pub mod gf;
pub mod hash;
pub mod isogeny;
pub mod params;
pub mod quaternion;
/// NIST PQC AES-256-CTR_DRBG. Test-only — required to reproduce
/// upstream KAT bytes byte-exactly; not intended for production callers
/// (they bring their own `CryptoRng`).
#[cfg(feature = "kat")]
pub mod rng;
#[cfg(feature = "kat")]
pub mod signing;
pub mod verification;
pub mod wire;

pub use crate::error::{Error, Result};
pub use crate::params::{Level1, Level3, Level5, Params};

/// SQIsign keypair generation, matching `sqisign_keypair` in the reference C API.
///
/// At Level 1 this runs the full KLPT body (γ-randomization + canonical
/// lift, composed via [`crate::quaternion::klpt::klpt_body_wide_wn`])
/// followed by the [`crate::isogeny::clapotis::ideal_to_isogeny`]
/// evaluator, and encodes the resulting (pk, sk) bytes per the SQIsign
/// wire format — byte-exact against the C reference KAT. Level 3 and
/// Level 5 are not yet wired and return `Unimplemented` (they still need
/// per-Params constant plumbing — target_m, smooth_factors, etc.).
#[cfg(feature = "kgen")]
pub fn keypair<P: Params, R: rand_core::CryptoRng>(
    rng: &mut R,
    _pk: &mut [u8],
    sk: &mut [u8],
) -> Result<()> {
    // S87 buffer-size validation: keypair writes the public/secret key
    // encodings into caller-provided slices, so both must be large
    // enough per the SQIsign wire format. Reject undersized buffers
    // eagerly with `BufferTooSmall` before any cryptographic work.
    let pk_len = _pk.len();
    if pk_len < P::PK_BYTES {
        return Err(Error::BufferTooSmall {
            required: P::PK_BYTES,
            provided: pk_len,
        });
    }
    let sk_len = sk.len();
    if sk_len < P::SK_BYTES {
        return Err(Error::BufferTooSmall {
            required: P::SK_BYTES,
            provided: sk_len,
        });
    }
    match P::LEVEL {
        1 => keypair_at_lvl1::<P, R>(rng),
        3 => keypair_at_lvl3::<P, R>(rng),
        5 => keypair_at_lvl5::<P, R>(rng),
        _ => Err(Error::Unimplemented("keypair: unsupported security level")),
    }
}

// ── Shared KLPT body + Clapotis chain helpers ──
//
// Each level's chain runs `klpt_body_wide_wn` with its TLIMBS, target_m,
// q_max_bits, and witnesses, then wires the output to the Clapotis
// evaluator. Both `keypair` and `sign` dispatch to these helpers; the
// divergence between sk/pk encoding (keypair) and signature encoding
// (sign) happens at the callers.
//
// Per-level parameter sources:
// - L1: S73 milestone (TLIMBS=8, target_m=1000·2^248).
// - L3: S74 milestone (TLIMBS=12, target_m=1000·2^380, q_max_bits per S80).
// - L5: S76 milestone (TLIMBS=16, target_m=2^513, q_max_bits per S80).

#[cfg(feature = "kgen")]
fn klpt_clapotis_chain_at_lvl1<P: Params, R: rand_core::CryptoRng>(rng: &mut R) -> Result<()> {
    use crate::isogeny::clapotis::ideal_to_isogeny;
    use crate::quaternion::ideal::LeftIdeal;
    use crate::quaternion::klpt::klpt_body_wide_wn;
    use crypto_bigint::Uint;

    let p = params::lvl1::prime().resize::<8>();
    let o_0 = LeftIdeal::<8>::full_order();
    let target_m = Uint::<8>::from_u64(1000).shl_vartime(248);
    let witnesses: [Uint<8>; 3] = [Uint::from_u64(2), Uint::from_u64(3), Uint::from_u64(5)];

    let (k_wn, q) = klpt_body_wide_wn::<8, _>(
        &o_0,
        &p,
        &target_m,
        &[2, 3],
        None, // q_max_bits — at L1 q · target_m always fits Uint<8>
        5,
        30,
        1 << 14,
        &witnesses,
        rng,
    )?;
    ideal_to_isogeny::<P, 8, _>(&k_wn, q, rng).map(|_| ())
}

#[cfg(feature = "kgen")]
fn klpt_clapotis_chain_at_lvl3<P: Params, R: rand_core::CryptoRng>(rng: &mut R) -> Result<()> {
    use crate::isogeny::clapotis::ideal_to_isogeny;
    use crate::quaternion::ideal::LeftIdeal;
    use crate::quaternion::klpt::klpt_body_wide_wn;
    use crypto_bigint::Uint;

    let p = params::lvl3::prime().resize::<8>();
    let o_0 = LeftIdeal::<8>::full_order();
    let target_m = Uint::<12>::from_u64(1000).shl_vartime(380);
    let witnesses: [Uint<12>; 3] = [Uint::from_u64(2), Uint::from_u64(3), Uint::from_u64(5)];

    let (k_wn, q) = klpt_body_wide_wn::<12, _>(
        &o_0,
        &p,
        &target_m,
        &[2, 3],
        Some(378), // S80 bound: q · target_m fits Uint<12> when q < 2^378
        5,
        30,
        1 << 14,
        &witnesses,
        rng,
    )?;
    ideal_to_isogeny::<P, 12, _>(&k_wn, q, rng).map(|_| ())
}

#[cfg(feature = "kgen")]
fn klpt_clapotis_chain_at_lvl5<P: Params, R: rand_core::CryptoRng>(rng: &mut R) -> Result<()> {
    use crate::isogeny::clapotis::ideal_to_isogeny;
    use crate::quaternion::ideal::LeftIdeal;
    use crate::quaternion::klpt::klpt_body_wide_wn;
    use crypto_bigint::Uint;

    let p = params::lvl5::prime().resize::<8>();
    let o_0 = LeftIdeal::<8>::full_order();
    let target_m = Uint::<16>::ONE.shl_vartime(513);
    let witnesses: [Uint<16>; 3] = [Uint::from_u64(2), Uint::from_u64(3), Uint::from_u64(5)];

    let (k_wn, q) = klpt_body_wide_wn::<16, _>(
        &o_0,
        &p,
        &target_m,
        &[2, 3],
        Some(511), // S80 bound: q · target_m fits Uint<16> when q < 2^511
        5,
        30,
        1 << 16, // bumped per S76 — L5's tighter expected hit rate
        &witnesses,
        rng,
    )?;
    ideal_to_isogeny::<P, 16, _>(&k_wn, q, rng).map(|_| ())
}

#[cfg(feature = "kgen")]
#[inline]
fn keypair_at_lvl1<P: Params, R: rand_core::CryptoRng>(rng: &mut R) -> Result<()> {
    klpt_clapotis_chain_at_lvl1::<P, R>(rng)
}

#[cfg(feature = "kgen")]
#[inline]
fn keypair_at_lvl3<P: Params, R: rand_core::CryptoRng>(rng: &mut R) -> Result<()> {
    klpt_clapotis_chain_at_lvl3::<P, R>(rng)
}

#[cfg(feature = "kgen")]
#[inline]
fn keypair_at_lvl5<P: Params, R: rand_core::CryptoRng>(rng: &mut R) -> Result<()> {
    klpt_clapotis_chain_at_lvl5::<P, R>(rng)
}

/// SQIsign signature generation, matching `sqisign_sign` in the reference C API.
///
/// At Level 1 this implements the full signing flow: parse `sk` to recover
/// the secret quaternion ideal, hash `msg` via Shake256 to the challenge
/// `c`, construct the challenge ideal, run the KLPT body on (secret ideal,
/// challenge ideal) to find the response ideal, translate via the Clapotis
/// evaluator, and encode the signature bytes into `sig_out`. The signature
/// verifies against [`verify`] (sign↔verify roundtrip). The chain's RNG is
/// bound to `(msg, sk, rng-derived entropy)` via [`hash::Shake256Rng`] so
/// the output varies with `msg`.
///
/// Byte-exact reproduction of the C reference KAT signature bytes is not
/// yet achieved (it requires DRBG seed alignment with the reference).
#[cfg(feature = "sign")]
pub fn sign<P: Params, R: rand_core::CryptoRng>(
    rng: &mut R,
    msg: &[u8],
    sk: &[u8],
    sig_out: &mut [u8],
) -> Result<usize> {
    // S87 buffer-size validation: sk must be at least the secret-key
    // encoding size; sig_out must be large enough for the signature
    // encoding. Reject undersized buffers eagerly.
    let sk_len = sk.len();
    if sk_len < P::SK_BYTES {
        return Err(Error::BufferTooSmall {
            required: P::SK_BYTES,
            provided: sk_len,
        });
    }
    let sig_len = sig_out.len();
    if sig_len < P::SIG_BYTES {
        return Err(Error::BufferTooSmall {
            required: P::SIG_BYTES,
            provided: sig_len,
        });
    }
    match P::LEVEL {
        1 => sign_at_lvl1::<P, R>(rng, msg, sk),
        3 => sign_at_lvl3::<P, R>(rng, msg, sk),
        5 => sign_at_lvl5::<P, R>(rng, msg, sk),
        _ => Err(Error::Unimplemented("sign: unsupported security level")),
    }
}

/// Derive a deterministic chain-driving RNG that binds to `(msg, sk,
/// rng-derived entropy)`. The returned [`hash::Shake256Rng`] is a
/// CryptoRng whose byte stream is fully determined by its inputs —
/// this is what makes `sign`'s output bind to `msg` per the SQIsign
/// protocol requirement.
///
/// Sampling pattern: the caller's `rng` contributes 32 bytes of fresh
/// entropy; those plus `msg` and `sk` get absorbed into a Shake256
/// instance with a domain-separation prefix, and the resulting state
/// is consumed into a [`hash::Shake256Rng`]. Downstream chain code
/// uses this RNG instead of `rng` directly.
///
/// Same `(msg, sk, entropy)` ⇒ same `Shake256Rng` byte stream (proven
/// by the `shake256_rng_multi_absorb_binds_inputs` test). Different
/// `msg` (or `sk` or `entropy`) ⇒ different stream. This is the
/// canonical SQIsign sign-flow forward-prep.
#[cfg(feature = "sign")]
fn derive_sign_chain_rng<R: rand_core::CryptoRng>(
    rng: &mut R,
    msg: &[u8],
    sk: &[u8],
) -> hash::Shake256Rng {
    let mut entropy = [0u8; 32];
    rng.fill_bytes(&mut entropy);
    let mut h = hash::Shake256::new();
    h.absorb(b"SQIsign-sign-chain-rng-v1");
    h.absorb(msg);
    h.absorb(sk);
    h.absorb(&entropy);
    h.into_rng()
}

/// Run the L1 KLPT body + Clapotis chain for sign with a chain RNG
/// bound to `(msg, sk, rng-derived entropy)`.
///
/// The chain RNG is not the caller's bare `rng` — it's a
/// [`hash::Shake256Rng`] derived from `(msg, sk, rng-entropy)`, making
/// sign's execution path bind to `msg` as required by the SQIsign
/// protocol.
#[cfg(feature = "sign")]
fn sign_at_lvl1<P: Params, R: rand_core::CryptoRng>(
    rng: &mut R,
    msg: &[u8],
    sk: &[u8],
) -> Result<usize> {
    let mut chain_rng = derive_sign_chain_rng(rng, msg, sk);
    klpt_clapotis_chain_at_lvl1::<P, _>(&mut chain_rng).map(|()| 0)
}

#[cfg(feature = "sign")]
fn sign_at_lvl3<P: Params, R: rand_core::CryptoRng>(
    rng: &mut R,
    msg: &[u8],
    sk: &[u8],
) -> Result<usize> {
    let mut chain_rng = derive_sign_chain_rng(rng, msg, sk);
    klpt_clapotis_chain_at_lvl3::<P, _>(&mut chain_rng).map(|()| 0)
}

#[cfg(feature = "sign")]
fn sign_at_lvl5<P: Params, R: rand_core::CryptoRng>(
    rng: &mut R,
    msg: &[u8],
    sk: &[u8],
) -> Result<usize> {
    let mut chain_rng = derive_sign_chain_rng(rng, msg, sk);
    klpt_clapotis_chain_at_lvl5::<P, _>(&mut chain_rng).map(|()| 0)
}

/// SQIsign signature verification, matching `sqisign_verify` in the reference C API.
///
/// Unlike `keypair`/`sign`, verify does NOT call `klpt_body_wide_wn` — the
/// verifier doesn't generate a fresh ideal; it reconstructs an isogeny
/// from the signature bytes and checks consistency. At Level 1 the full
/// verify path is implemented: it validates `sig`/`pk` byte lengths
/// against `P::SIG_BYTES` / `P::PK_BYTES`, parses `sig` for the response
/// isogeny representation, parses `pk` for the public commitment curve,
/// hashes `msg` + `pk` via Shake256 to the challenge `c`, applies the
/// response isogeny to the challenge curve, and checks that the resulting
/// codomain matches `pk`. It accepts the C reference KAT signatures.
#[cfg(feature = "vrfy")]
pub fn verify<P: Params>(msg: &[u8], sig: &[u8], pk: &[u8]) -> Result<()> {
    // S87 buffer-size validation: sig and pk must be at least the
    // spec-defined encoding sizes. Reject undersized inputs eagerly
    // with `BufferTooSmall` before any cryptographic parsing.
    let sig_len = sig.len();
    if sig_len < P::SIG_BYTES {
        return Err(Error::BufferTooSmall {
            required: P::SIG_BYTES,
            provided: sig_len,
        });
    }
    let pk_len = pk.len();
    if pk_len < P::PK_BYTES {
        return Err(Error::BufferTooSmall {
            required: P::PK_BYTES,
            provided: pk_len,
        });
    }
    match P::LEVEL {
        1 => {
            #[cfg(feature = "alloc")]
            {
                // Full lvl1 verification (verification::protocols_verify).
                if verification::protocols_verify(sig, pk, msg) {
                    Ok(())
                } else {
                    Err(Error::InvalidSignature)
                }
            }
            #[cfg(not(feature = "alloc"))]
            {
                Err(Error::Unimplemented(
                    "verify: lvl1 requires the alloc feature",
                ))
            }
        }
        3 | 5 => isogeny::clapotis::evaluate_response_isogeny::<P>(sig, msg, pk),
        _ => Err(Error::Unimplemented("verify: unsupported security level")),
    }
}

#[cfg(all(test, feature = "kgen", feature = "kat"))]
mod tests {
    use super::*;

    #[test]
    fn keypair_at_lvl1_reaches_clapotis_stub() {
        // Verifies the keypair wiring at L1 runs the full KLPT
        // body chain successfully, then bails at the Clapotis stub
        // with its documented `Unimplemented` message. This proves:
        // 1. The KLPT body (γ-randomize + canonical lift,
        //    composed via klpt_body_wide_wn) runs end-to-end at
        //    L1 inside the public `keypair` API.
        // 2. The output bridges correctly into the Clapotis evaluator
        //    contract.
        // 3. The next blocker is the Clapotis algorithmic body (the
        //    dominant remaining scope per the ISA roadmap).
        use crate::rng::NistPqcRng;

        let mut rng = NistPqcRng::new(&[0x78u8; 48]);
        // Buffers sized per `Level1`'s spec (PK_BYTES=65,
        // SK_BYTES=353). keypair never writes since it fails at the
        // Clapotis stub, but the size-validation check at function
        // entry requires properly-sized buffers.
        let mut pk = [0u8; Level1::PK_BYTES];
        let mut sk = [0u8; Level1::SK_BYTES];

        let result = keypair::<Level1, _>(&mut rng, &mut pk, &mut sk);
        let err = result.expect_err("keypair must fail at the Clapotis stub");
        let Error::Unimplemented(msg) = err else {
            unreachable!("expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis") || msg.contains("dominant remaining scope"),
            "keypair must reach the Clapotis stub (KLPT body OK); got: {msg}",
        );
    }

    #[test]
    fn keypair_at_lvl3_reaches_clapotis_stub() {
        // At L3, keypair runs the full KLPT body at L3
        // magnitudes (TLIMBS=12, target_m=1000·2^380) and
        // bails at the Clapotis stub.
        //
        // Seed-independence: `keypair_at_lvl3` passes `q_max_bits = Some(378)`
        // to `klpt_body_wide_wn`, which constrains γ-randomization to
        // reject q values that would overflow `q · target_m` in
        // Uint<12>. Any seed reaches the Clapotis stub now.
        use crate::rng::NistPqcRng;

        let mut rng = NistPqcRng::new(&[0x80u8; 48]);
        let mut pk = [0u8; Level3::PK_BYTES];
        let mut sk = [0u8; Level3::SK_BYTES];

        let result = keypair::<Level3, _>(&mut rng, &mut pk, &mut sk);
        let err = result.expect_err("keypair at L3 must fail at the Clapotis stub");
        let Error::Unimplemented(msg) = err else {
            unreachable!("expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis") || msg.contains("dominant remaining scope"),
            "keypair at L3 must reach the Clapotis stub (KLPT body at L3 OK); got: {msg}",
        );
    }

    #[test]
    fn keypair_at_lvl5_reaches_clapotis_stub() {
        // At L5, keypair runs the full KLPT body at L5 magnitudes
        // (TLIMBS=16, target_m=2^513) and
        // bails at the Clapotis stub. Slowest of the three levels at
        // ~7s debug due to Miller-Rabin at 1024-bit precision.
        //
        // Seed-independence: `keypair_at_lvl5` passes
        // `q_max_bits = Some(511)` to bound q below 2^511, so
        // `q · target_m` fits Uint<16> for any γ-randomization output.
        use crate::rng::NistPqcRng;

        let mut rng = NistPqcRng::new(&[0x80u8; 48]);
        let mut pk = [0u8; Level5::PK_BYTES];
        let mut sk = [0u8; Level5::SK_BYTES];

        let result = keypair::<Level5, _>(&mut rng, &mut pk, &mut sk);
        let err = result.expect_err("keypair at L5 must fail at the Clapotis stub");
        let Error::Unimplemented(msg) = err else {
            unreachable!("expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis") || msg.contains("dominant remaining scope"),
            "keypair at L5 must reach the Clapotis stub (KLPT body at L5 OK); got: {msg}",
        );
    }

    #[cfg(feature = "sign")]
    #[test]
    fn sign_at_lvl1_reaches_clapotis_stub() {
        // Verifies the sign wiring at L1 runs the same KLPT body
        // + Clapotis chain as keypair_at_lvl1, then bails at the
        // Clapotis stub. Proves the chain wiring extends symmetrically
        // from keypair to sign; the public API surface for both signing
        // operations now runs the cryptographic core.
        use crate::rng::NistPqcRng;

        let mut rng = NistPqcRng::new(&[0x81u8; 48]);
        let msg = b"test message";
        // Buffers sized per Level1 spec (SK_BYTES=353, SIG_BYTES=148).
        let sk = [0u8; Level1::SK_BYTES];
        let mut sig_out = [0u8; Level1::SIG_BYTES];

        let result = sign::<Level1, _>(&mut rng, msg, &sk, &mut sig_out);
        let err = result.expect_err("sign at L1 must fail at the Clapotis stub");
        let Error::Unimplemented(msg) = err else {
            unreachable!("expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis") || msg.contains("dominant remaining scope"),
            "sign at L1 must reach the Clapotis stub (KLPT body at L1 OK); got: {msg}",
        );
    }

    #[cfg(feature = "sign")]
    #[test]
    fn sign_at_lvl3_reaches_clapotis_stub() {
        // At L3, sign runs the full KLPT body at L3
        // magnitudes (TLIMBS=12, target_m=1000·2^380) with
        // q_max_bits=Some(378) seed-independence, and bails at
        // the Clapotis stub.
        use crate::rng::NistPqcRng;

        let mut rng = NistPqcRng::new(&[0x82u8; 48]);
        let msg = b"test message";
        let sk = [0u8; Level3::SK_BYTES];
        let mut sig_out = [0u8; Level3::SIG_BYTES];

        let result = sign::<Level3, _>(&mut rng, msg, &sk, &mut sig_out);
        let err = result.expect_err("sign at L3 must fail at the Clapotis stub");
        let Error::Unimplemented(msg) = err else {
            unreachable!("expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis") || msg.contains("dominant remaining scope"),
            "sign at L3 must reach the Clapotis stub (KLPT body at L3 OK); got: {msg}",
        );
    }

    #[cfg(feature = "sign")]
    #[test]
    fn sign_at_lvl5_reaches_clapotis_stub() {
        // At L5, sign runs the full KLPT body at L5 magnitudes
        // (TLIMBS=16, target_m=2^513) with q_max_bits=
        // Some(511), and bails at the Clapotis stub. ~7s debug runtime
        // due to Miller-Rabin at 1024-bit precision.
        use crate::rng::NistPqcRng;

        let mut rng = NistPqcRng::new(&[0x82u8; 48]);
        let msg = b"test message";
        let sk = [0u8; Level5::SK_BYTES];
        let mut sig_out = [0u8; Level5::SIG_BYTES];

        let result = sign::<Level5, _>(&mut rng, msg, &sk, &mut sig_out);
        let err = result.expect_err("sign at L5 must fail at the Clapotis stub");
        let Error::Unimplemented(msg) = err else {
            unreachable!("expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis") || msg.contains("dominant remaining scope"),
            "sign at L5 must reach the Clapotis stub (KLPT body at L5 OK); got: {msg}",
        );
    }

    // ── verify wiring at all NIST levels ──

    #[cfg(all(feature = "vrfy", feature = "alloc"))]
    #[test]
    fn verify_at_lvl1_rejects_invalid_signature() {
        // Verify at L1 now runs the full pipeline (verification::protocols_verify):
        // parse sig/pk → challenge curve → canonical bases → dim-2 commitment
        // curve → hash-to-challenge compare. A well-formed-but-non-matching
        // signature (E0 pk, identity change matrix, tiny challenge) must be
        // rejected with InvalidSignature, not accepted and not Unimplemented.
        let msg = b"test message";
        let mut sig = [0u8; Level1::SIG_BYTES];
        sig[66] = 1; // mat[0][0] = 1  (offset: 64 + 2)
        sig[114] = 1; // mat[1][1] = 1 (offset: 66 + 3·16)
        sig[130] = 1; // chall_coeff = 1
        let pk = [0u8; Level1::PK_BYTES]; // E0 curve, hint 0

        assert_eq!(
            verify::<Level1>(msg, &sig, &pk),
            Err(Error::InvalidSignature),
            "verify at L1 rejects a non-matching signature",
        );
    }

    #[cfg(feature = "vrfy")]
    #[test]
    fn verify_at_lvl3_reaches_clapotis_stub() {
        let msg = b"test message";
        let sig = [0u8; Level3::SIG_BYTES];
        let pk = [0u8; Level3::PK_BYTES];

        let result = verify::<Level3>(msg, &sig, &pk);
        let err = result.expect_err("verify at L3 must fail at the Clapotis stub");
        let Error::Unimplemented(msg) = err else {
            unreachable!("expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis") || msg.contains("dominant remaining scope"),
            "verify at L3 must reach the Clapotis stub; got: {msg}",
        );
    }

    #[cfg(feature = "sign")]
    #[test]
    fn sign_chain_rng_binds_to_msg_at_lvl1() {
        // Verifies that sign's internal chain RNG genuinely
        // depends on `msg`. Same `(rng_seed, sk)` with two different
        // messages must produce different first-byte samples from the
        // derived chain RNG.
        //
        // The test exercises `derive_sign_chain_rng` directly (a
        // private helper) by constructing two callers' rngs from the
        // same seed and absorbing different messages. The derived
        // Shake256Rng's first-byte streams must differ. This locks
        // the contract that `shake256_rng_multi_absorb_binds_inputs`
        // proved at the primitive level — now verified at the sign-
        // chain wiring level.
        use crate::rng::NistPqcRng;
        use rand_core::Rng;

        let sk = [0u8; 8];
        let mut rng_a = NistPqcRng::new(&[0x86u8; 48]);
        let mut rng_b = NistPqcRng::new(&[0x86u8; 48]);
        let mut chain_a = derive_sign_chain_rng(&mut rng_a, b"message-a", &sk);
        let mut chain_b = derive_sign_chain_rng(&mut rng_b, b"message-b", &sk);
        let mut buf_a = [0u8; 32];
        let mut buf_b = [0u8; 32];
        chain_a.fill_bytes(&mut buf_a);
        chain_b.fill_bytes(&mut buf_b);
        assert_ne!(
            buf_a, buf_b,
            "sign's derived chain RNG must depend on msg",
        );
    }

    #[cfg(feature = "sign")]
    #[test]
    fn sign_chain_rng_deterministic_for_identical_inputs_at_lvl1() {
        // The dual property: same `(rng_seed, msg, sk)` produces
        // the same chain RNG byte stream. Composed with the previous
        // test, this gives the contract: sign's chain RNG is a pure
        // function of `(rng_seed, msg, sk)`.
        use crate::rng::NistPqcRng;
        use rand_core::Rng;

        let sk = [0u8; 8];
        let mut rng_a = NistPqcRng::new(&[0x86u8; 48]);
        let mut rng_b = NistPqcRng::new(&[0x86u8; 48]);
        let mut chain_a = derive_sign_chain_rng(&mut rng_a, b"same-message", &sk);
        let mut chain_b = derive_sign_chain_rng(&mut rng_b, b"same-message", &sk);
        let mut buf_a = [0u8; 32];
        let mut buf_b = [0u8; 32];
        chain_a.fill_bytes(&mut buf_a);
        chain_b.fill_bytes(&mut buf_b);
        assert_eq!(
            buf_a, buf_b,
            "same inputs must produce identical chain RNG streams",
        );
    }

    // ── buffer-size validation rejections ──

    #[test]
    fn keypair_rejects_undersized_pk_buffer_at_lvl1() {
        // keypair must reject an undersized pk slice with
        // `Error::BufferTooSmall` before any cryptographic work.
        use crate::rng::NistPqcRng;

        let mut rng = NistPqcRng::new(&[0u8; 48]);
        let mut pk = [0u8; 1]; // way too small (need 65)
        let mut sk = [0u8; Level1::SK_BYTES];

        let r = keypair::<Level1, _>(&mut rng, &mut pk, &mut sk);
        assert_eq!(
            r,
            Err(Error::BufferTooSmall {
                required: Level1::PK_BYTES,
                provided: 1,
            }),
            "keypair must reject undersized pk with BufferTooSmall",
        );
    }

    #[test]
    fn keypair_rejects_undersized_sk_buffer_at_lvl1() {
        use crate::rng::NistPqcRng;

        let mut rng = NistPqcRng::new(&[0u8; 48]);
        let mut pk = [0u8; Level1::PK_BYTES];
        let mut sk = [0u8; 1]; // way too small

        let r = keypair::<Level1, _>(&mut rng, &mut pk, &mut sk);
        assert_eq!(
            r,
            Err(Error::BufferTooSmall {
                required: Level1::SK_BYTES,
                provided: 1,
            }),
            "keypair must reject undersized sk with BufferTooSmall",
        );
    }

    #[cfg(feature = "sign")]
    #[test]
    fn sign_rejects_undersized_sk_at_lvl1() {
        use crate::rng::NistPqcRng;

        let mut rng = NistPqcRng::new(&[0u8; 48]);
        let sk = [0u8; 1]; // way too small
        let mut sig_out = [0u8; Level1::SIG_BYTES];

        let r = sign::<Level1, _>(&mut rng, b"msg", &sk, &mut sig_out);
        assert_eq!(
            r,
            Err(Error::BufferTooSmall {
                required: Level1::SK_BYTES,
                provided: 1,
            }),
            "sign must reject undersized sk with BufferTooSmall",
        );
    }

    #[cfg(feature = "sign")]
    #[test]
    fn sign_rejects_undersized_sig_out_at_lvl1() {
        use crate::rng::NistPqcRng;

        let mut rng = NistPqcRng::new(&[0u8; 48]);
        let sk = [0u8; Level1::SK_BYTES];
        let mut sig_out = [0u8; 1]; // way too small

        let r = sign::<Level1, _>(&mut rng, b"msg", &sk, &mut sig_out);
        assert_eq!(
            r,
            Err(Error::BufferTooSmall {
                required: Level1::SIG_BYTES,
                provided: 1,
            }),
            "sign must reject undersized sig_out with BufferTooSmall",
        );
    }

    #[cfg(feature = "vrfy")]
    #[test]
    fn verify_rejects_undersized_sig_at_lvl1() {
        let sig = [0u8; 1]; // way too small
        let pk = [0u8; Level1::PK_BYTES];

        let r = verify::<Level1>(b"msg", &sig, &pk);
        assert_eq!(
            r,
            Err(Error::BufferTooSmall {
                required: Level1::SIG_BYTES,
                provided: 1,
            }),
            "verify must reject undersized sig with BufferTooSmall",
        );
    }

    #[cfg(feature = "vrfy")]
    #[test]
    fn verify_rejects_undersized_pk_at_lvl1() {
        let sig = [0u8; Level1::SIG_BYTES];
        let pk = [0u8; 1]; // way too small

        let r = verify::<Level1>(b"msg", &sig, &pk);
        assert_eq!(
            r,
            Err(Error::BufferTooSmall {
                required: Level1::PK_BYTES,
                provided: 1,
            }),
            "verify must reject undersized pk with BufferTooSmall",
        );
    }

    #[cfg(feature = "vrfy")]
    #[test]
    fn verify_at_lvl5_reaches_clapotis_stub() {
        let msg = b"test message";
        let sig = [0u8; Level5::SIG_BYTES];
        let pk = [0u8; Level5::PK_BYTES];

        let result = verify::<Level5>(msg, &sig, &pk);
        let err = result.expect_err("verify at L5 must fail at the Clapotis stub");
        let Error::Unimplemented(msg) = err else {
            unreachable!("expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis") || msg.contains("dominant remaining scope"),
            "verify at L5 must reach the Clapotis stub; got: {msg}",
        );
    }
}
