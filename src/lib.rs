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

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "alloc")]
#[allow(unused_extern_crates)]
extern crate alloc;

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

pub use crate::error::{Error, Result};
pub use crate::params::{Level1, Level3, Level5, Params};

/// SQIsign keypair generation, matching `sqisign_keypair` in the reference C API.
///
/// This is a placeholder; full keypair generation requires the quaternion order
/// arithmetic and ideal-to-isogeny translation that land in subsequent sessions.
#[cfg(feature = "kgen")]
pub fn keypair<P: Params, R: rand_core::CryptoRng>(
    _rng: &mut R,
    _pk: &mut [u8],
    _sk: &mut [u8],
) -> Result<()> {
    Err(Error::Unimplemented("keypair: pending isogeny pipeline"))
}

/// SQIsign signature generation, matching `sqisign_sign` in the reference C API.
#[cfg(feature = "sign")]
pub fn sign<P: Params, R: rand_core::CryptoRng>(
    _rng: &mut R,
    _msg: &[u8],
    _sk: &[u8],
    _sig_out: &mut [u8],
) -> Result<usize> {
    Err(Error::Unimplemented("sign: pending KLPT + Clapotis"))
}

/// SQIsign signature verification, matching `sqisign_verify` in the reference C API.
#[cfg(feature = "vrfy")]
pub fn verify<P: Params>(_msg: &[u8], _sig: &[u8], _pk: &[u8]) -> Result<()> {
    Err(Error::Unimplemented("verify: pending isogeny pipeline"))
}
