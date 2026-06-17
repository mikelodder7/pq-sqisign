// SPDX-License-Identifier: MIT OR Apache-2.0
//! Isogenies between supersingular Montgomery curves over `F_{p^2}`.
//!
//! SQIsign manipulates isogenies of two complementary degree shapes:
//!
//! - 2-power isogenies (smooth-degree chains over `F_{p^2}`), evaluated with
//!   x-only Velu formulas (Costello-Hisil), forming the bulk of the signing
//!   pipeline — implemented in [`two`].
//! - Higher-dimensional theta isogenies (SQIsign 2.0.1 introduces these for
//!   the *response*-side computation in the Clapotis evaluator) — implemented
//!   across the theta modules and driven by [`clapotis_spine`].
//!
//! See spec §5 (Geometric Algorithms) for formula references.

pub mod clapotis;
#[cfg(feature = "alloc")]
pub mod clapotis_spine;
pub mod endomorphism;
#[cfg(feature = "alloc")]
pub mod fixed_degree;
pub mod gluing;
pub mod splitting;
pub mod theta;
pub mod theta_chain;
pub mod theta_doubling;
pub mod theta_isogeny;
pub mod two;

#[cfg(feature = "alloc")]
pub use two::IsogenyChain2e;
pub use two::TwoIsogeny;
