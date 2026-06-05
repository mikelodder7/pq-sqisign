// SPDX-License-Identifier: MIT OR Apache-2.0
//! Quaternion algebra `B_{p,∞} = Q⟨i, j⟩` over `Q` ramified at `p` and `∞`.
//!
//! SQIsign uses this quaternion algebra and a specific maximal order `O_0`
//! to translate between the geometric world (isogenies of supersingular
//! curves over `F_{p^2}`) and the arithmetic world (left ideals of `O_0`).
//! The KLPT algorithm — implemented in a later session — replaces a given
//! left `O_0`-ideal with a Galois-equivalent one of smooth norm, enabling
//! efficient ideal-to-isogeny translation (Clapotis).
//!
//! # Conventions
//!
//! For every SQIsign prime (which satisfies `p ≡ 3 mod 4`) the standard
//! presentation of `B_{p,∞}` is
//!
//! ```text
//!     i² = −1
//!     j² = −p
//!     k = i · j = −j · i
//! ```
//!
//! and the "special" maximal order is
//!
//! ```text
//!     O_0 = ⟨ 1, i, (i + j) / 2, (1 + k) / 2 ⟩.
//! ```
//!
//! Multiplication table for `(a + b·i + c·j + d·k) × (e + f·i + g·j + h·k)`:
//!
//! ```text
//!     1 = a·e − b·f − p · (c·g + d·h)
//!     i = a·f + b·e − p · (c·h − d·g)
//!     j = a·g + c·e + (b·h − d·f)
//!     k = a·h + d·e + (b·g − c·f)
//! ```
//!
//! Reduced norm `N(q) = q · q̄ = a² + b² + p · (c² + d²)`.
//! Reduced trace `Tr(q) = q + q̄ = 2 a`.
//!
//! # Integer width
//!
//! Coefficients live in `Int<LIMBS>` (`crypto-bigint`'s signed bigint).
//! The default `LIMBS = 8` (512 bits) is comfortable for Level-1's
//! 251-bit prime plus typical KLPT lift magnitudes. Higher levels will
//! parametrise on `LIMBS` once KLPT lands.

pub mod algebra;
pub mod connecting_ideals;
pub mod cornacchia;
pub mod curves_with_endomorphism;
pub mod dpe;
pub mod extremal_orders;
pub mod hnf;
pub mod ideal;
pub mod ideal_mul;
pub mod klpt;
pub mod lattice;
pub mod lll;
pub mod norm_search;
pub mod o0_mul;
pub mod primality;
pub mod represent_integer;
pub mod sample;
pub mod short_vec;
pub mod sign_orchestration;
pub mod smooth;
pub mod sqrt_mod;

pub use algebra::{MaximalOrder, OrderBasis, Quaternion, QuaternionAlgebra, RationalQuaternion};
pub use cornacchia::{cornacchia, cornacchia_classical};
pub use hnf::hnf_4x4;
pub use ideal::LeftIdeal;
pub use ideal_mul::{ideal_multiply, ideal_right_multiply};
#[cfg(feature = "alloc")]
pub use klpt::lift_to_any_smooth_target;
pub use klpt::{lift_to_smooth_norm, principal_ideal_with_reduced_norm};
pub use lattice::{dot4, gram_matrix_4x4, norm2, size_reduce_4x4};
pub use norm_search::find_norm_witness;
pub use o0_mul::{
    multiply_o0_basis, o0_basis_to_standard_doubled, o0_conjugate, principal_left_ideal_from_o0,
    reduced_norm_o0_basis, standard_to_o0_basis,
};
pub use sample::sample_random_quaternion_o0;
pub use short_vec::{find_quaternion_in_ideal_with_norm, shortest_quaternion_in_ideal};
#[cfg(feature = "alloc")]
pub use smooth::{SmoothNumber, enumerate_smooth, next_smooth_at_least};
pub use sqrt_mod::{pow_mod, tonelli_shanks};
