// SPDX-License-Identifier: MIT OR Apache-2.0
//! Elliptic-curve arithmetic for SQIsign.
//!
//! All SQIsign curves are supersingular Montgomery curves over `F_{p^2}` of
//! the form `E_A : y^2 = x^3 + A x^2 + x`. We carry only x-coordinates of
//! points (the standard Montgomery x-only representation, projective form
//! `(X : Z)`) because every operation the scheme needs — scalar
//! multiplication, three-point ladder, j-invariant — is x-only.

pub mod couple;
pub mod jacobian;
pub mod montgomery;

pub use montgomery::{MontgomeryCurve, MontgomeryPoint};
