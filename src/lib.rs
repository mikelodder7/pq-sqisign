// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pure-Rust implementation of SQIsign — compact post-quantum digital signatures
//! built from supersingular isogenies and quaternion orders.
//!
//! Reference specification: <https://sqisign.org/spec/sqisign-20250707.pdf>.
//! Reference C implementation: <https://github.com/SQISign/the-sqisign>.
//!
//! # Status
//!
//! This crate is a multi-session ground-up port. The foundation layer
//! (parameter sets, GF(p) / GF(p^2) field arithmetic, Montgomery-form curve
//! point arithmetic, KAT harness scaffolding) is present. The high-level
//! signing and verification pipelines — quaternion KLPT, ideal-to-isogeny
//! translation (Clapotis), challenge derivation — are stubbed and gated
//! behind `Unimplemented` errors until further sessions land them. See
//! `ISA.md` at the repository root for the multi-session roadmap.
//!
//! # Layout
//!
//! - [`params`] — security-level parameter sets (Level-1, Level-3, Level-5).
//! - [`gf`] — base-field GF(p) and quadratic extension GF(p^2) arithmetic.
//! - [`ec`] — Montgomery elliptic curves (x-only point representation).
//! - [`quaternion`] — quaternion algebra over Q (stub).
//! - [`isogeny`] — isogeny computations on supersingular curves (stub).
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
//! q values, smooth lift targets. When secret-key parsing lands and
//! secret quaternion ideals begin flowing through this path, the affected
//! sub-primitives must be replaced with constant-time variants per the
//! ISA security checklist.
//!
//! No long-lived secret material is currently held in heap or stack
//! data structures (all KLPT body intermediate state is computed
//! transiently within a single `keypair`/`sign` call). When secret-key
//! material does land, the secret-bearing types will derive `Zeroize`
//! and `ZeroizeOnDrop` (the `zeroize` dependency is already wired with
//! `derive` feature).

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
pub mod verification;
pub mod wire;

pub use crate::error::{Error, Result};
pub use crate::params::{Level1, Level3, Level5, Params};

/// SQIsign keypair generation, matching `sqisign_keypair` in the reference C API.
///
/// **Current status (S78)**: wired at Level 1 to call the full KLPT body
/// (S65 γ-randomization + S71/S75 canonical lift, composed via S76's
/// [`crate::quaternion::klpt::klpt_body_wide_wn`]) followed by the
/// [`crate::isogeny::clapotis::ideal_to_isogeny`] Clapotis stub (S77).
/// The chain runs the cryptographic core successfully at L1 before
/// failing at the Clapotis stub's `Unimplemented`. At L3/L5 the function
/// returns `Unimplemented` immediately pending per-Params constant
/// plumbing (target_m, smooth_factors, etc.) in future sessions.
///
/// Once the Clapotis evaluator's algorithmic body lands (~25-30 sessions
/// per the ISA roadmap), this function will additionally encode the
/// resulting (pk, sk) bytes per the SQIsign wire format.
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

// ── Shared KLPT body + Clapotis stub chain helpers (S82 cleanup) ──
//
// Each level's chain runs `klpt_body_wide_wn` with its TLIMBS, target_m,
// q_max_bits, and witnesses, then wires the output to the Clapotis stub.
// Currently fails at the Clapotis stub. Both `keypair` and `sign`
// dispatch to these helpers; the divergence between sk/pk encoding
// (keypair) and signature encoding (sign) will happen at the
// callers once the Clapotis algorithmic body lands.
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
/// **Current status (S82)**: wired at all three NIST levels (L1/L3/L5) to
/// call the full KLPT body followed by the Clapotis stub chain. Mirrors
/// `keypair`'s S78/S79 dispatch shape exactly.
///
/// Future sessions will:
/// 1. Parse `sk` to extract the secret quaternion ideal.
/// 2. Hash `msg` via Shake256 → challenge `c` (Fp²).
/// 3. Construct the challenge ideal from c.
/// 4. Run KLPT body on (secret ideal, challenge ideal) to find the
///    response ideal.
/// 5. Translate via the Clapotis evaluator.
/// 6. Encode signature bytes into `sig_out`.
///
/// For S81/S82 the scaffolding wires the chain shape; S86 binds the
/// chain's RNG to `(msg, sk, rng-derived entropy)` via [`hash::Shake256Rng`]
/// so sign's execution path now varies with `msg`. Remaining
/// algorithmic content follows the Clapotis arc + sk parsing + the
/// full challenge-ideal construction.
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
/// by S85's `shake256_rng_multi_absorb_binds_inputs` test). Different
/// `msg` (or `sk` or `entropy`) ⇒ different stream. This is the
/// canonical SQIsign sign-flow forward-prep documented in S85.
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

/// Run the L1 KLPT body + Clapotis stub chain for sign with a chain
/// RNG bound to `(msg, sk, rng-derived entropy)`. The chain currently
/// fails at the Clapotis stub; this adapter maps the success path to
/// a 0-length signature placeholder (unreachable while Clapotis is
/// stubbed).
///
/// S86 update: the chain RNG is no longer the caller's bare `rng` —
/// it's a [`hash::Shake256Rng`] derived from `(msg, sk, rng-entropy)`,
/// making sign's execution path bind to `msg` as required by the
/// SQIsign protocol.
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
/// **Current status (S84)**: wired at all three NIST levels (L1/L3/L5)
/// to call the Clapotis-side `evaluate_response_isogeny` stub. Unlike
/// `keypair`/`sign`, verify does NOT call `klpt_body_wide_wn` — the
/// verifier doesn't generate a fresh ideal; it reconstructs an isogeny
/// from the signature bytes and checks consistency. The full verify
/// path lands once the Clapotis evaluator's algorithmic body
/// (`evaluate_response_isogeny`) is implemented.
///
/// Future sessions will:
/// 1. Validate `sig`/`pk` byte lengths against `P::SIG_BYTES` / `P::PK_BYTES`.
/// 2. Parse `sig` to extract the response isogeny representation.
/// 3. Parse `pk` to extract the public commitment curve.
/// 4. Hash `msg` + `pk` via Shake256 → challenge `c`.
/// 5. Apply the response isogeny to the challenge curve.
/// 6. Check the resulting codomain matches `pk`.
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
        // S78 — verifies the keypair wiring at L1 runs the full KLPT
        // body chain successfully, then bails at the Clapotis stub
        // with its documented `Unimplemented` message. This proves:
        // 1. The KLPT body (S65 γ-randomize + S71/S75 canonical lift,
        //    composed via S76's klpt_body_wide_wn) runs end-to-end at
        //    L1 inside the public `keypair` API.
        // 2. The output bridges correctly into the Clapotis evaluator
        //    contract established in S77.
        // 3. The next blocker is the Clapotis algorithmic body (the
        //    dominant remaining scope per the ISA roadmap).
        use crate::rng::NistPqcRng;

        let mut rng = NistPqcRng::new(&[0x78u8; 48]);
        // S87 — buffers sized per `Level1`'s spec (PK_BYTES=65,
        // SK_BYTES=353). keypair never writes since it fails at the
        // Clapotis stub, but the size-validation check at function
        // entry requires properly-sized buffers.
        let mut pk = [0u8; Level1::PK_BYTES];
        let mut sk = [0u8; Level1::SK_BYTES];

        let result = keypair::<Level1, _>(&mut rng, &mut pk, &mut sk);
        let err = result.expect_err("S78: keypair must fail at the Clapotis stub");
        let Error::Unimplemented(msg) = err else {
            unreachable!("S78: expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis") || msg.contains("dominant remaining scope"),
            "S78: keypair must reach the Clapotis stub (KLPT body OK); got: {msg}",
        );
    }

    #[test]
    fn keypair_at_lvl3_reaches_clapotis_stub() {
        // S79 — at L3, keypair runs the full KLPT body at L3
        // magnitudes (TLIMBS=12, target_m=1000·2^380 per S74) and
        // bails at the Clapotis stub.
        //
        // **S80 seed-independence**: the S79 seed-dependency
        // ("must use 0x77 to avoid large q") is removed. The
        // `keypair_at_lvl3` helper now passes `q_max_bits = Some(378)`
        // to `klpt_body_wide_wn`, which constrains γ-randomization to
        // reject q values that would overflow `q · target_m` in
        // Uint<12>. Any seed reaches the Clapotis stub now.
        use crate::rng::NistPqcRng;

        let mut rng = NistPqcRng::new(&[0x80u8; 48]);
        let mut pk = [0u8; Level3::PK_BYTES];
        let mut sk = [0u8; Level3::SK_BYTES];

        let result = keypair::<Level3, _>(&mut rng, &mut pk, &mut sk);
        let err = result.expect_err("S79: keypair at L3 must fail at the Clapotis stub");
        let Error::Unimplemented(msg) = err else {
            unreachable!("S79: expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis") || msg.contains("dominant remaining scope"),
            "S79: keypair at L3 must reach the Clapotis stub (KLPT body at L3 OK); got: {msg}",
        );
    }

    #[test]
    fn keypair_at_lvl5_reaches_clapotis_stub() {
        // S79 — at L5, keypair runs the full KLPT body at L5 magnitudes
        // (TLIMBS=16, target_m=2^513 per S76's known-good regime) and
        // bails at the Clapotis stub. Slowest of the three levels at
        // ~7s debug due to Miller-Rabin at 1024-bit precision.
        //
        // **S80 seed-independence**: `keypair_at_lvl5` now passes
        // `q_max_bits = Some(511)` to bound q below 2^511, so
        // `q · target_m` fits Uint<16> for any γ-randomization output.
        // The S79 seed-dependency is removed.
        use crate::rng::NistPqcRng;

        let mut rng = NistPqcRng::new(&[0x80u8; 48]);
        let mut pk = [0u8; Level5::PK_BYTES];
        let mut sk = [0u8; Level5::SK_BYTES];

        let result = keypair::<Level5, _>(&mut rng, &mut pk, &mut sk);
        let err = result.expect_err("S79: keypair at L5 must fail at the Clapotis stub");
        let Error::Unimplemented(msg) = err else {
            unreachable!("S79: expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis") || msg.contains("dominant remaining scope"),
            "S79: keypair at L5 must reach the Clapotis stub (KLPT body at L5 OK); got: {msg}",
        );
    }

    #[cfg(feature = "sign")]
    #[test]
    fn sign_at_lvl1_reaches_clapotis_stub() {
        // S81 — verifies the sign wiring at L1 runs the same KLPT body
        // + Clapotis chain as keypair_at_lvl1, then bails at the
        // Clapotis stub. Proves the chain wiring extends symmetrically
        // from keypair to sign; the public API surface for both signing
        // operations now runs the cryptographic core.
        use crate::rng::NistPqcRng;

        let mut rng = NistPqcRng::new(&[0x81u8; 48]);
        let msg = b"test message";
        // S87 — buffers sized per Level1 spec (SK_BYTES=353, SIG_BYTES=148).
        let sk = [0u8; Level1::SK_BYTES];
        let mut sig_out = [0u8; Level1::SIG_BYTES];

        let result = sign::<Level1, _>(&mut rng, msg, &sk, &mut sig_out);
        let err = result.expect_err("S81: sign at L1 must fail at the Clapotis stub");
        let Error::Unimplemented(msg) = err else {
            unreachable!("S81: expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis") || msg.contains("dominant remaining scope"),
            "S81: sign at L1 must reach the Clapotis stub (KLPT body at L1 OK); got: {msg}",
        );
    }

    #[cfg(feature = "sign")]
    #[test]
    fn sign_at_lvl3_reaches_clapotis_stub() {
        // S82 — at L3, sign now runs the full KLPT body at L3
        // magnitudes (TLIMBS=12, target_m=1000·2^380 per S74) with
        // S80's q_max_bits=Some(378) seed-independence, and bails at
        // the Clapotis stub.
        use crate::rng::NistPqcRng;

        let mut rng = NistPqcRng::new(&[0x82u8; 48]);
        let msg = b"test message";
        let sk = [0u8; Level3::SK_BYTES];
        let mut sig_out = [0u8; Level3::SIG_BYTES];

        let result = sign::<Level3, _>(&mut rng, msg, &sk, &mut sig_out);
        let err = result.expect_err("S82: sign at L3 must fail at the Clapotis stub");
        let Error::Unimplemented(msg) = err else {
            unreachable!("S82: expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis") || msg.contains("dominant remaining scope"),
            "S82: sign at L3 must reach the Clapotis stub (KLPT body at L3 OK); got: {msg}",
        );
    }

    #[cfg(feature = "sign")]
    #[test]
    fn sign_at_lvl5_reaches_clapotis_stub() {
        // S82 — at L5, sign runs the full KLPT body at L5 magnitudes
        // (TLIMBS=16, target_m=2^513 per S76) with S80's q_max_bits=
        // Some(511), and bails at the Clapotis stub. ~7s debug runtime
        // due to Miller-Rabin at 1024-bit precision.
        use crate::rng::NistPqcRng;

        let mut rng = NistPqcRng::new(&[0x82u8; 48]);
        let msg = b"test message";
        let sk = [0u8; Level5::SK_BYTES];
        let mut sig_out = [0u8; Level5::SIG_BYTES];

        let result = sign::<Level5, _>(&mut rng, msg, &sk, &mut sig_out);
        let err = result.expect_err("S82: sign at L5 must fail at the Clapotis stub");
        let Error::Unimplemented(msg) = err else {
            unreachable!("S82: expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis") || msg.contains("dominant remaining scope"),
            "S82: sign at L5 must reach the Clapotis stub (KLPT body at L5 OK); got: {msg}",
        );
    }

    // ── S84 — verify wiring at all NIST levels ──

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
        let err = result.expect_err("S84: verify at L3 must fail at the Clapotis stub");
        let Error::Unimplemented(msg) = err else {
            unreachable!("S84: expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis") || msg.contains("dominant remaining scope"),
            "S84: verify at L3 must reach the Clapotis stub; got: {msg}",
        );
    }

    #[cfg(feature = "sign")]
    #[test]
    fn sign_chain_rng_binds_to_msg_at_lvl1() {
        // S86 — verifies that sign's internal chain RNG genuinely
        // depends on `msg`. Same `(rng_seed, sk)` with two different
        // messages must produce different first-byte samples from the
        // derived chain RNG.
        //
        // The test exercises `derive_sign_chain_rng` directly (a
        // private helper) by constructing two callers' rngs from the
        // same seed and absorbing different messages. The derived
        // Shake256Rng's first-byte streams must differ. This locks
        // the contract that S85's `shake256_rng_multi_absorb_binds_inputs`
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
            "S86: sign's derived chain RNG must depend on msg",
        );
    }

    #[cfg(feature = "sign")]
    #[test]
    fn sign_chain_rng_deterministic_for_identical_inputs_at_lvl1() {
        // S86 — the dual property: same `(rng_seed, msg, sk)` produces
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
            "S86: same inputs must produce identical chain RNG streams",
        );
    }

    // ── S87 — buffer-size validation rejections ──

    #[test]
    fn keypair_rejects_undersized_pk_buffer_at_lvl1() {
        // S87 — keypair must reject an undersized pk slice with
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
            "S87: keypair must reject undersized pk with BufferTooSmall",
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
            "S87: keypair must reject undersized sk with BufferTooSmall",
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
            "S87: sign must reject undersized sk with BufferTooSmall",
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
            "S87: sign must reject undersized sig_out with BufferTooSmall",
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
            "S87: verify must reject undersized sig with BufferTooSmall",
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
            "S87: verify must reject undersized pk with BufferTooSmall",
        );
    }

    #[cfg(feature = "vrfy")]
    #[test]
    fn verify_at_lvl5_reaches_clapotis_stub() {
        let msg = b"test message";
        let sig = [0u8; Level5::SIG_BYTES];
        let pk = [0u8; Level5::PK_BYTES];

        let result = verify::<Level5>(msg, &sig, &pk);
        let err = result.expect_err("S84: verify at L5 must fail at the Clapotis stub");
        let Error::Unimplemented(msg) = err else {
            unreachable!("S84: expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis") || msg.contains("dominant remaining scope"),
            "S84: verify at L5 must reach the Clapotis stub; got: {msg}",
        );
    }
}
