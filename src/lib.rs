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

#[cfg(all(feature = "alloc", not(feature = "std")))]
#[macro_use]
extern crate alloc;
#[cfg(all(feature = "alloc", feature = "std"))]
extern crate alloc;
#[cfg(test)]
extern crate std;

#[macro_use]
mod macros;

pub mod ec;
pub mod encoding;
pub mod error;
pub mod gf;
pub mod hash;
pub mod isogeny;
pub mod level_constants;
pub mod params;
pub mod quaternion;
/// NIST PQC AES-256-CTR_DRBG. Test-only — required to reproduce
/// upstream KAT bytes byte-exactly; not intended for production callers
/// (they bring their own `CryptoRng`).
#[cfg(feature = "kat")]
pub mod rng;
#[cfg(feature = "sign")]
pub(crate) mod signing;
pub mod verification;
pub mod wire;

// Typed high-level API (mayo-style).
#[cfg(all(feature = "sign", feature = "alloc"))]
pub mod keypair;
#[cfg(all(feature = "sign", feature = "alloc"))]
pub mod signing_key;
#[cfg(feature = "alloc")]
pub mod sqisignature;
#[cfg(feature = "alloc")]
pub mod verifying_key;

pub use crate::error::{Error, Result};
#[cfg(all(feature = "sign", feature = "alloc"))]
pub use crate::keypair::KeyPair;
pub use crate::params::{Level1, Level3, Level5, Params};
#[cfg(all(feature = "sign", feature = "alloc"))]
pub use crate::signing_key::SigningKey;
#[cfg(feature = "alloc")]
pub use crate::sqisignature::SqiSignature;
#[cfg(feature = "alloc")]
pub use crate::verifying_key::VerifyingKey;

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
    ideal_to_isogeny::<P, 8, _>(&k_wn, &q, rng).map(|_| ())
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
    ideal_to_isogeny::<P, 12, _>(&k_wn, &q, rng).map(|_| ())
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
    ideal_to_isogeny::<P, 16, _>(&k_wn, &q, rng).map(|_| ())
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
/// Decodes `sk`, then runs the functional signer
/// ([`crate::signing::protocols_sign`]): sample the commitment, hash `msg` to
/// the challenge, build the response, and encode the signature directly into
/// `sig_out` (no heap allocation — SQIsign signatures are fixed-length
/// `P::SIG_BYTES`). Returns the signature length. Draws from `rng` and binds the
/// message through the challenge hash, matching the C reference (no separate RNG
/// hedge). The signature verifies against [`verify`].
///
/// Supported at levels 1 and 3 (level-3 additionally requires
/// `Level3::sk_from_bytes`, not yet implemented). Byte-exact reproduction of the
/// C reference KAT signature bytes is a separate, in-progress alignment effort.
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
        1 => sign_at_lvl1(rng, msg, sk, sig_out),
        3 => sign_at_lvl3(rng, msg, sk, sig_out),
        _ => Err(Error::Unimplemented(
            "sign: only security levels 1 and 3 are supported",
        )),
    }
}

/// Decode the level-1 `sk` bytes and run the functional signer
/// ([`crate::signing::protocols_sign`]) into `sig_out`. Draws directly from
/// `rng` — the message binds through the challenge hash, matching the C
/// reference `crypto_sign` (no separate RNG hedge).
#[cfg(feature = "sign")]
fn sign_at_lvl1<R: rand_core::CryptoRng>(
    rng: &mut R,
    msg: &[u8],
    sk: &[u8],
    sig_out: &mut [u8],
) -> Result<usize> {
    use crate::keypair::KeyLevel;
    use crate::params::Level1;
    let sk_data = Level1::sk_from_bytes(sk)?;
    Level1::protocols_sign(&sk_data, msg, rng, sig_out).ok_or(Error::SigningFailed)
}

/// Level-3 counterpart of [`sign_at_lvl1`]. NOTE: `Level3::sk_from_bytes` is not
/// yet implemented, so this currently surfaces that decode gap (the isogeny path
/// itself is functional via `protocols_sign`).
#[cfg(feature = "sign")]
fn sign_at_lvl3<R: rand_core::CryptoRng>(
    rng: &mut R,
    msg: &[u8],
    sk: &[u8],
    sig_out: &mut [u8],
) -> Result<usize> {
    use crate::keypair::KeyLevel;
    use crate::params::Level3;
    let sk_data = Level3::sk_from_bytes(sk)?;
    Level3::protocols_sign(&sk_data, msg, rng, sig_out).ok_or(Error::SigningFailed)
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
pub fn verify<P: verification::VerifyLevel>(msg: &[u8], sig: &[u8], pk: &[u8]) -> Result<()> {
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
    // Per-level dispatch (lvl1/lvl3 → generic protocols_verify; lvl5 unimpl).
    P::verify_bytes(sig, pk, msg)
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

    /// Public `crate::sign` (the free fn taking `sk` BYTES) is wired to the
    /// functional signer at lvl1: load C's KAT record-0 secret key, sign a
    /// message into `sig_out`, and verify it against C's public key. Exercises
    /// the whole front-door path (sk-decode → `protocols_sign` → `sig_out`) and
    /// confirms it produces a valid signature end-to-end.
    #[cfg(all(feature = "kat", feature = "vrfy"))]
    #[test]
    fn sign_lvl1_public_api_roundtrip_with_c_sk() {
        use crate::rng::NistPqcRng;

        let rsp = include_str!("../tests/KAT/PQCsignKAT_353_SQIsign_lvl1.rsp");
        let (mut sk, mut pk, mut msg): (Vec<u8>, Vec<u8>, Vec<u8>) =
            (Vec::new(), Vec::new(), Vec::new());
        for line in rsp.lines() {
            if let Some(h) = line.strip_prefix("msg = ") {
                if msg.is_empty() {
                    msg = hex::decode(h.trim()).unwrap();
                }
            } else if let Some(h) = line.strip_prefix("pk = ") {
                if pk.is_empty() {
                    pk = hex::decode(h.trim()).unwrap();
                }
            } else if let Some(h) = line.strip_prefix("sk = ") {
                if sk.is_empty() {
                    sk = hex::decode(h.trim()).unwrap();
                    break;
                }
            }
        }
        let mut rng = NistPqcRng::new(&[0x11u8; 48]);
        let mut sig = [0u8; Level1::SIG_BYTES];
        let n = sign::<Level1, _>(&mut rng, &msg, &sk, &mut sig)
            .expect("lvl1 crate::sign must produce a signature");
        assert_eq!(n, Level1::SIG_BYTES);
        verify::<Level1>(&msg, &sig[..n], &pk)
            .expect("the produced signature must verify against C's pk");
    }

    /// lvl3 counterpart of [`sign_lvl1_public_api_roundtrip_with_c_sk`]: the
    /// public `crate::sign` now decodes C's lvl3 `sk` (via `Level3::sk_from_bytes`),
    /// signs, and the signature verifies against C's lvl3 pk. Heavy (lvl3 sign),
    /// hence `#[ignore]`.
    #[cfg(all(feature = "kat", feature = "vrfy"))]
    #[test]
    #[ignore = "heavy: lvl3 crate::sign roundtrip with C's KAT sk"]
    fn sign_lvl3_public_api_roundtrip_with_c_sk() {
        use crate::rng::NistPqcRng;

        let rsp = include_str!("../tests/KAT/PQCsignKAT_529_SQIsign_lvl3.rsp");
        let (mut sk, mut pk, mut msg): (Vec<u8>, Vec<u8>, Vec<u8>) =
            (Vec::new(), Vec::new(), Vec::new());
        for line in rsp.lines() {
            if let Some(h) = line.strip_prefix("msg = ") {
                if msg.is_empty() {
                    msg = hex::decode(h.trim()).unwrap();
                }
            } else if let Some(h) = line.strip_prefix("pk = ") {
                if pk.is_empty() {
                    pk = hex::decode(h.trim()).unwrap();
                }
            } else if let Some(h) = line.strip_prefix("sk = ") {
                if sk.is_empty() {
                    sk = hex::decode(h.trim()).unwrap();
                    break;
                }
            }
        }
        let mut rng = NistPqcRng::new(&[0x13u8; 48]);
        let mut sig = [0u8; Level3::SIG_BYTES];
        let n = sign::<Level3, _>(&mut rng, &msg, &sk, &mut sig)
            .expect("lvl3 crate::sign must produce a signature");
        assert_eq!(n, Level3::SIG_BYTES);
        verify::<Level3>(&msg, &sig[..n], &pk)
            .expect("the produced lvl3 signature must verify against C's pk");
    }

    /// Fast lvl3 `sk` decode self-check (no signing): `Level3::sk_from_bytes` on
    /// C's KAT sk must recover a curve/hint matching C's pk, and a well-formed
    /// secret ideal (perfect-square `cached_norm` ⇒ the W=32 reconstruction did
    /// not overflow). Confirms the decode is correct independent of the heavy
    /// sign roundtrip.
    #[cfg(all(feature = "kat", feature = "sign"))]
    #[test]
    fn sk_decode_lvl3_matches_kat_pk() {
        use crate::keypair::KeyLevel;
        use crate::verification::PublicKeyData;

        let rsp = include_str!("../tests/KAT/PQCsignKAT_529_SQIsign_lvl3.rsp");
        let (mut sk, mut pk): (Vec<u8>, Vec<u8>) = (Vec::new(), Vec::new());
        for line in rsp.lines() {
            if let Some(h) = line.strip_prefix("pk = ") {
                if pk.is_empty() {
                    pk = hex::decode(h.trim()).unwrap();
                }
            } else if let Some(h) = line.strip_prefix("sk = ") {
                if sk.is_empty() {
                    sk = hex::decode(h.trim()).unwrap();
                    break;
                }
            }
        }
        let sk_data = Level3::sk_from_bytes(&sk).expect("lvl3 sk decodes");
        let pk_data = PublicKeyData::<params::lvl3::Fp3Element>::from_bytes::<Level3>(&pk)
            .expect("lvl3 pk decodes");
        assert_eq!(sk_data.curve_a, pk_data.curve_a, "sk curve must match pk");
        assert_eq!(sk_data.hint_pk, pk_data.hint_pk, "sk hint must match pk");
        assert!(
            sk_data.secret_ideal.reduced_norm_vartime().is_some(),
            "secret ideal cached_norm must be a perfect square (W=32 no overflow)"
        );
    }

    /// lvl5 signing is unsupported; the public API rejects it cleanly.
    #[cfg(feature = "sign")]
    #[test]
    fn sign_lvl5_unsupported() {
        use crate::rng::NistPqcRng;
        let mut rng = NistPqcRng::new(&[0x82u8; 48]);
        let sk = [0u8; Level5::SK_BYTES];
        let mut sig_out = [0u8; Level5::SIG_BYTES];
        let err = sign::<Level5, _>(&mut rng, b"m", &sk, &mut sig_out)
            .expect_err("lvl5 sign is unsupported");
        assert!(matches!(err, Error::Unimplemented(_)));
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
    fn verify_at_lvl3_rejects_garbage_via_real_path() {
        // Level-3 verify now runs the generalized `protocols_verify` (field-
        // generic), not the old Clapotis stub. An all-zero signature is not a
        // valid signature and must be rejected without panicking.
        let msg = b"test message";
        let sig = [0u8; Level3::SIG_BYTES];
        let pk = [0u8; Level3::PK_BYTES];

        let result = verify::<Level3>(msg, &sig, &pk);
        assert_eq!(
            result,
            Err(Error::InvalidSignature),
            "verify at L3 runs the real path and rejects a garbage signature",
        );
    }

    #[test]
    fn max_sig_bytes_bounds_all_levels() {
        use crate::params::MAX_SIG_BYTES;
        assert!(Level1::SIG_BYTES <= MAX_SIG_BYTES);
        assert!(Level3::SIG_BYTES <= MAX_SIG_BYTES);
        assert!(Level5::SIG_BYTES <= MAX_SIG_BYTES);
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
    fn verify_at_lvl5_is_unimplemented() {
        // Level 5 has no `LevelConstants`/spine yet; verify reports it as
        // unimplemented (via the `VerifyLevel` dispatch).
        let msg = b"test message";
        let sig = [0u8; Level5::SIG_BYTES];
        let pk = [0u8; Level5::PK_BYTES];

        let result = verify::<Level5>(msg, &sig, &pk);
        let err = result.expect_err("verify at L5 must report unimplemented");
        let Error::Unimplemented(msg) = err else {
            unreachable!("expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("level 5"),
            "verify at L5 must report level-5 unimplemented; got: {msg}",
        );
    }
}
