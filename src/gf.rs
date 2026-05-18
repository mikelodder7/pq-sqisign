// SPDX-License-Identifier: MIT OR Apache-2.0
//! Galois-field arithmetic for SQIsign.
//!
//! [`fp`] provides `F_p` element types — wrappers over `crypto-bigint`'s
//! constant-time `ConstMontyForm` for each parameter level. [`fp2`] provides
//! the quadratic extension `F_{p^2} = F_p[i]/(i^2 + 1)` generic over the base
//! field, including the operations the higher-level isogeny code needs:
//! addition, subtraction, multiplication, squaring, inversion, Frobenius,
//! `mul_by_i`, byte encoding/decoding.
//!
//! All operations are intended to be constant-time on the secret-dependent
//! data path. Platform-specific specialization (BMI2 `mulx`, NEON wide-mul)
//! is gated by the [`arch`] submodule, which is currently a place-holder —
//! the underlying limb multiplication intrinsics live inside `crypto-bigint`.

pub mod arch;
pub mod fp;
pub mod fp2;
