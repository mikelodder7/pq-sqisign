// SPDX-License-Identifier: MIT OR Apache-2.0
//! Security-level parameter sets for SQIsign.
//!
//! SQIsign defines three NIST security levels. Each level fixes a base prime
//! `p ≡ 3 (mod 4)` of the form `p = c · 2^f − 1`, with the quadratic extension
//! `F_{p^2} = F_p[i]/(i^2 + 1)` serving as the field of definition for the
//! supersingular curve `E_0 : y^2 = x^3 + x`.
//!
//! | Level | Bits of p | c  | f   | pk | sk  | sig |
//! |-------|-----------|----|-----|----|-----|-----|
//! |   1   |   251     | 5  | 248 | 65 | 353 | 148 |
//! |   3   |   383     | 65 | 376 | 97 | 529 | 224 |
//! |   5   |   505     | 27 | 500 |129 | 701 | 292 |

pub mod lvl1;
pub mod lvl3;
pub mod lvl5;

pub use lvl1::Level1;
pub use lvl3::Level3;
pub use lvl5::Level5;

/// Common parameter-set surface implemented by [`Level1`], [`Level3`], [`Level5`].
///
/// A `Params` implementor names the prime, the field-element byte size, and the
/// encoded public-key / secret-key / signature byte sizes that the SQIsign wire
/// format demands at this level. The field-arithmetic and curve-arithmetic
/// modules are generic over `Params` so they don't have to be rewritten per
/// level.
pub trait Params: 'static + Copy + core::fmt::Debug {
    /// NIST security level (1, 3, or 5).
    const LEVEL: u8;
    /// Bit-length of the base prime `p` (251, 383, or 505).
    const P_BITS: usize;
    /// Cofactor `c` in `p = c · 2^f − 1`.
    const C: u64;
    /// Exponent `f` in `p = c · 2^f − 1`.
    const F: usize;
    /// Bytes needed to encode an `F_p` element (`(P_BITS + 7) / 8`).
    const FP_BYTES: usize;
    /// Bytes needed to encode an `F_{p^2}` element (`2 · FP_BYTES`).
    const FP2_BYTES: usize;
    /// Encoded public key length, in bytes.
    const PK_BYTES: usize;
    /// Encoded secret key length, in bytes.
    const SK_BYTES: usize;
    /// Encoded signature length, in bytes.
    const SIG_BYTES: usize;
    /// `SQIsign_response_length` from the spec (challenge response bit-length).
    const RESPONSE_BITS: usize;
    /// `HASH_ITERATIONS` from the spec (challenge-derivation oracle iterations).
    const HASH_ITERATIONS: usize;
    /// `SECURITY_BITS` from the spec — the NIST-level classical security
    /// parameter (128/192/256 for L1/L3/L5).
    const SECURITY_BITS: usize;
}
