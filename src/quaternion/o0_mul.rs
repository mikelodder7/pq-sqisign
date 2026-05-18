// SPDX-License-Identifier: MIT OR Apache-2.0
//! Multiplication of `O_0`-basis-coordinate elements.
//!
//! `O_0 = ⟨1, i, (i+j)/2, (1+k)/2⟩` has *fractional* basis vectors in the
//! standard `(1, i, j, k)` basis. Doing arithmetic on `O_0` elements while
//! staying in integer coordinates requires the doubling trick:
//!
//! Let `x = a·1 + b·i + c·(i+j)/2 + d·(1+k)/2` with `(a, b, c, d) ∈ Z^4`.
//! Then `2·x` has *integer* standard-basis coordinates:
//!
//! ```text
//!     2·x = (2a + d, 2b + c, c, d)
//! ```
//!
//! Given two `O_0` elements `x`, `y`, the product `x · y ∈ O_0`. Computing
//! `2x · 2y = 4(xy)` keeps us entirely in integer-standard-basis territory,
//! and the recovery to `O_0` coordinates is:
//!
//! ```text
//!     standard coords (s_0, s_1, s_2, s_3) of (xy) satisfy
//!         s_0 = a' + d'/2, s_1 = b' + c'/2, s_2 = c'/2, s_3 = d'/2,
//!     so given T = 4(xy) with coords (T_0, T_1, T_2, T_3):
//!         o_0(xy) = (T_0 − T_3) / 4
//!         o_1(xy) = (T_1 − T_2) / 4
//!         o_2(xy) = T_2 / 2
//!         o_3(xy) = T_3 / 2.
//! ```
//!
//! All four divisions are exact (valid `O_0` elements have these integer
//! coordinates; the construction guarantees it).

use crypto_bigint::{Int, Uint};

use crate::quaternion::Quaternion;
use crate::quaternion::hnf::int_div_floor;

/// Convert a quaternion with **integer** standard `(1, i, j, k)` coordinates
/// to its `O_0`-basis coordinates `(o_0, o_1, o_2, o_3)`.
///
/// Every integer-standard quaternion lies in `Z⟨1, i, j, k⟩ ⊆ O_0`, so this
/// conversion always succeeds. The formulas come from the inverse of the
/// `O_0`-basis change:
///
/// ```text
///     standard coords (qa, qb, qc, qd) of `x = a + b·i + c·(i+j)/2 + d·(1+k)/2`:
///         qa = a + d/2,  qb = b + c/2,  qc = c/2,  qd = d/2
///     hence for integer standard inputs:
///         o_3 = 2·qd,  o_2 = 2·qc,  o_1 = qb − qc,  o_0 = qa − qd.
/// ```
pub fn standard_to_o0_basis<const LIMBS: usize>(q: &Quaternion<LIMBS>) -> [Int<LIMBS>; 4] {
    let two = Int::<LIMBS>::from_i64(2);
    [
        q.a.wrapping_sub(&q.d),
        q.b.wrapping_sub(&q.c),
        two.wrapping_mul(&q.c),
        two.wrapping_mul(&q.d),
    ]
}

/// Convert from `O_0`-basis coords to *doubled* standard coords —
/// i.e., the standard `(1, i, j, k)` coords of `2·x` (which are integer
/// even when `x`'s own coords would be half-integer). This is the
/// canonical integer-arithmetic representation of an `O_0` element when
/// you need to interact with the `(1, i, j, k)` side of the algebra.
pub fn o0_basis_to_standard_doubled<const LIMBS: usize>(
    coords: &[Int<LIMBS>; 4],
) -> Quaternion<LIMBS> {
    let two = Int::<LIMBS>::from_i64(2);
    Quaternion::<LIMBS>::new(
        two.wrapping_mul(&coords[0]).wrapping_add(&coords[3]),
        two.wrapping_mul(&coords[1]).wrapping_add(&coords[2]),
        coords[2],
        coords[3],
    )
}

/// Conjugate `γ̄` of an `O_0` element expressed in `O_0`-basis coordinates.
///
/// Derivation: with `γ` having standard coords `(a + d/2, b + c/2, c/2, d/2)`,
/// `γ̄` has standard coords `(a + d/2, −b − c/2, −c/2, −d/2)`. Inverting back
/// to `O_0` coords via `o_3 = 2·qd, o_2 = 2·qc, o_1 = qb − qc, o_0 = qa − qd`:
///
/// ```text
///     (a, b, c, d) ↦ (a + d, −b, −c, −d).
/// ```
pub fn o0_conjugate<const LIMBS: usize>(coords: &[Int<LIMBS>; 4]) -> [Int<LIMBS>; 4] {
    [
        coords[0].wrapping_add(&coords[3]),
        coords[1].wrapping_neg(),
        coords[2].wrapping_neg(),
        coords[3].wrapping_neg(),
    ]
}

/// Build the principal left ideal `O_0 · γ` as a `LeftIdeal` in canonical
/// HNF form, where `γ` is given in `O_0`-basis coordinates.
///
/// Algorithm: basis vectors of `O_0 · γ` are `e_i · γ` for `i ∈ 0..4`,
/// computed via `multiply_o0_basis`, then HNF-reduced.
pub fn principal_left_ideal_from_o0<const LIMBS: usize>(
    gamma: &[Int<LIMBS>; 4],
    p: &Uint<LIMBS>,
) -> crate::quaternion::LeftIdeal<LIMBS> {
    let zero = Int::<LIMBS>::from_i64(0);
    let mut basis = [[zero; 4]; 4];
    for k in 0..4 {
        let mut e = [zero; 4];
        e[k] = Int::<LIMBS>::from_i64(1);
        basis[k] = multiply_o0_basis(&e, gamma, p);
    }
    let reduced = crate::quaternion::hnf::hnf_4x4(&basis);
    crate::quaternion::LeftIdeal::new(reduced)
}

/// Reduced norm `N_red(x) = x · x̄ ∈ Z` of an `O_0` element expressed in
/// `O_0`-basis coordinates `(a, b, c, d)` for the canonical basis
/// `(1, i, (i+j)/2, (1+k)/2)`.
///
/// Uses the `2·x` standard-basis trick to stay in integer arithmetic:
/// `N_red(2x) = (2a+d)² + (2b+c)² + p · (c² + d²)`, then `N_red(x) =
/// N_red(2x) / 4` (exact division for valid `O_0` elements).
pub fn reduced_norm_o0_basis<const LIMBS: usize>(
    coords: &[Int<LIMBS>; 4],
    p: &Uint<LIMBS>,
) -> Int<LIMBS> {
    let two = Int::<LIMBS>::from_i64(2);
    let four = Int::<LIMBS>::from_i64(4);
    let qa = two.wrapping_mul(&coords[0]).wrapping_add(&coords[3]);
    let qb = two.wrapping_mul(&coords[1]).wrapping_add(&coords[2]);
    let qc = coords[2];
    let qd = coords[3];
    let q = Quaternion::<LIMBS>::new(qa, qb, qc, qd);
    let n_two_x = q.norm(p);
    int_div_floor(&n_two_x, &four)
}

/// Multiply two `O_0` elements expressed in `O_0`-basis coordinates.
///
/// `x_o0`, `y_o0` carry the integer coordinates `(a, b, c, d)` in
/// `(1, i, (i+j)/2, (1+k)/2)` order. `p` is the level's prime
/// (`B_{p,∞}` ramifies at `p` and `∞`).
///
/// Returns the `O_0`-basis coordinates of `x · y`.
pub fn multiply_o0_basis<const LIMBS: usize>(
    x_o0: &[Int<LIMBS>; 4],
    y_o0: &[Int<LIMBS>; 4],
    p: &Uint<LIMBS>,
) -> [Int<LIMBS>; 4] {
    let two = Int::<LIMBS>::from_i64(2);
    let four = Int::<LIMBS>::from_i64(4);

    // 2·x in standard (1, i, j, k) basis = (2a + d, 2b + c, c, d).
    let qa_x = two.wrapping_mul(&x_o0[0]).wrapping_add(&x_o0[3]);
    let qb_x = two.wrapping_mul(&x_o0[1]).wrapping_add(&x_o0[2]);
    let qc_x = x_o0[2];
    let qd_x = x_o0[3];
    let x_std = Quaternion::<LIMBS>::new(qa_x, qb_x, qc_x, qd_x);

    let qa_y = two.wrapping_mul(&y_o0[0]).wrapping_add(&y_o0[3]);
    let qb_y = two.wrapping_mul(&y_o0[1]).wrapping_add(&y_o0[2]);
    let qc_y = y_o0[2];
    let qd_y = y_o0[3];
    let y_std = Quaternion::<LIMBS>::new(qa_y, qb_y, qc_y, qd_y);

    // T = (2x)(2y) = 4(xy) in standard basis.
    let t = x_std.mul(&y_std, p);

    // Recover O_0 coords of (xy) — all divisions are exact for valid inputs.
    let o0 = int_div_floor(&t.a.wrapping_sub(&t.d), &four);
    let o1 = int_div_floor(&t.b.wrapping_sub(&t.c), &four);
    let o2 = int_div_floor(&t.c, &two);
    let o3 = int_div_floor(&t.d, &two);
    [o0, o1, o2, o3]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(v: i64) -> Int<8> {
        Int::<8>::from_i64(v)
    }

    fn fake_p() -> Uint<8> {
        Uint::<8>::from_u64(7)
    }

    /// `e_0 = 1`; `1 · 1 = 1`.
    #[test]
    fn one_squared_is_one() {
        let e0 = [n(1), n(0), n(0), n(0)];
        let r = multiply_o0_basis(&e0, &e0, &fake_p());
        assert_eq!(r, e0);
    }

    /// `e_1 = i`; `i · i = −1 = −e_0`.
    #[test]
    fn i_squared_is_minus_one() {
        let e1 = [n(0), n(1), n(0), n(0)];
        let r = multiply_o0_basis(&e1, &e1, &fake_p());
        assert_eq!(r, [n(-1), n(0), n(0), n(0)]);
    }

    /// `e_0 · e_1 = i = e_1`.
    #[test]
    fn one_times_i_is_i() {
        let e0 = [n(1), n(0), n(0), n(0)];
        let e1 = [n(0), n(1), n(0), n(0)];
        let r = multiply_o0_basis(&e0, &e1, &fake_p());
        assert_eq!(r, e1);
    }

    /// `e_3 · e_3 = ((1+k)/2)² = (1 + 2k + k²)/4 = (1 + 2k − p)/4`.
    /// For `p = 7`: standard coords `((1−7)/4, 0, 0, 1/2) = (−3/2, 0, 0, 1/2)`.
    /// `O_0` coords: `o_3 = 1`, `o_2 = 0`, `o_1 = 0`, `o_0 = qa − qd = −3/2 − 1/2 = −2`.
    /// So `e_3² = (−2, 0, 0, 1)` in `O_0`-coords for `p = 7`.
    #[test]
    fn e3_squared_for_fake_p_7() {
        let e3 = [n(0), n(0), n(0), n(1)];
        let r = multiply_o0_basis(&e3, &e3, &fake_p());
        assert_eq!(r, [n(-2), n(0), n(0), n(1)]);
    }

    /// `e_2 = (i+j)/2`. `e_2² = (i + j)²/4 = (i² + 2ij + j²)/4 = (−1 + 2k − p)/4`.
    /// For `p = 7`: standard coords `(−8/4, 0, 0, 2/4) = (−2, 0, 0, 1/2)`.
    /// `O_0` coords: `o_3 = 1`, `o_2 = 0`, `o_1 = 0`, `o_0 = −2 − 1/2 = −5/2`. NOT INTEGER!
    /// So `(i+j)/2 ∉ O_0` for general `p`? Let me re-derive: `e_2²` in standard
    /// coords is `(−(1+p)/4, 0, 0, 1/2)`. For `p=7`: `(−2, 0, 0, 1/2)`. Then
    /// `o_0 = qa − qd = −2 − 1/2 = −5/2` — not integer. But `O_0` should be closed!
    ///
    /// Actually `(i+j)/2 · (i+j)/2 = (i² + ij + ji + j²)/4 = (−1 + ij − ij − p)/4
    ///   = (−1 − p)/4`. So `e_2² = (−1 − p)/4` is a pure scalar, *not* including k.
    /// For `p ≡ 3 mod 4`, `(−1 − p)/4 ∈ Z`. For p=7: `(−1 − 7)/4 = −2`. So
    /// `e_2² = −2 = −2·e_0` and `O_0` coords are `(−2, 0, 0, 0)`.
    #[test]
    fn e2_squared_for_fake_p_7() {
        let e2 = [n(0), n(0), n(1), n(0)];
        let r = multiply_o0_basis(&e2, &e2, &fake_p());
        assert_eq!(r, [n(-2), n(0), n(0), n(0)]);
    }

    /// Distributivity sanity: `(e_0 + e_1) · e_0 = e_0 · e_0 + e_1 · e_0 = 1 + i`.
    /// In `O_0` coords: `(1, 1, 0, 0)`.
    #[test]
    fn distributivity_one_plus_i() {
        let e0_plus_e1 = [n(1), n(1), n(0), n(0)];
        let e0 = [n(1), n(0), n(0), n(0)];
        let r = multiply_o0_basis(&e0_plus_e1, &e0, &fake_p());
        assert_eq!(r, e0_plus_e1);
    }

    /// Non-commutativity: `e_1 · e_2 ≠ e_2 · e_1`.
    /// `i · (i+j)/2 = (i² + ij)/2 = (−1 + k)/2`. Standard `(−1/2, 0, 0, 1/2)`.
    /// `O_0` coords: `o_3 = 1`, `o_0 = qa − qd = −1`. So `(−1, 0, 0, 1)`.
    ///
    /// `(i+j)/2 · i = (i² + ji)/2 = (−1 − k)/2`. Standard `(−1/2, 0, 0, −1/2)`.
    /// `O_0` coords: `o_3 = −1`, `o_0 = qa − qd = −1/2 − (−1/2) = 0`. So `(0, 0, 0, −1)`.
    #[test]
    fn i_times_e2_is_not_e2_times_i() {
        let e1 = [n(0), n(1), n(0), n(0)];
        let e2 = [n(0), n(0), n(1), n(0)];
        let lhs = multiply_o0_basis(&e1, &e2, &fake_p());
        let rhs = multiply_o0_basis(&e2, &e1, &fake_p());
        assert_eq!(lhs, [n(-1), n(0), n(0), n(1)]);
        assert_eq!(rhs, [n(0), n(0), n(0), n(-1)]);
        assert_ne!(lhs, rhs);
    }

    /// `e_3 = (1+k)/2`; `e_0 · e_3 = e_3`. Confirms left-identity.
    #[test]
    fn one_times_e3_is_e3() {
        let e0 = [n(1), n(0), n(0), n(0)];
        let e3 = [n(0), n(0), n(0), n(1)];
        let r = multiply_o0_basis(&e0, &e3, &fake_p());
        assert_eq!(r, e3);
    }

    /// `o0_conjugate` of `e_0` = `1` is `1`.
    #[test]
    fn conjugate_of_one_is_one() {
        let e0 = [n(1), n(0), n(0), n(0)];
        assert_eq!(o0_conjugate(&e0), e0);
    }

    /// `o0_conjugate` of `i` is `-i`.
    #[test]
    fn conjugate_of_i_is_negative_i() {
        let e1 = [n(0), n(1), n(0), n(0)];
        let conj = o0_conjugate(&e1);
        assert_eq!(conj, [n(0), n(-1), n(0), n(0)]);
    }

    /// `o0_conjugate` of `(1+k)/2` = `(1-k)/2 = 1 - (1+k)/2 = e_0 - e_3`.
    /// So `O_0`-coords `(0, 0, 0, 1)` ↦ `(1, 0, 0, -1)`.
    #[test]
    fn conjugate_of_e3() {
        let e3 = [n(0), n(0), n(0), n(1)];
        let conj = o0_conjugate(&e3);
        assert_eq!(conj, [n(1), n(0), n(0), n(-1)]);
    }

    /// Conjugation is an involution.
    #[test]
    fn conjugate_is_involution() {
        let q = [n(3), n(-5), n(7), n(-2)];
        assert_eq!(o0_conjugate(&o0_conjugate(&q)), q);
    }

    /// `γ · γ̄ = N_red(γ)` (scalar quaternion in O_0).
    #[test]
    fn gamma_times_conj_gamma_is_norm() {
        let p = fake_p();
        let gamma = [n(3), n(-2), n(1), n(0)];
        let conj = o0_conjugate(&gamma);
        let prod = multiply_o0_basis(&gamma, &conj, &p);
        // Product should be a pure scalar in O_0: only o_0 nonzero, equal to N_red(γ).
        let norm = reduced_norm_o0_basis(&gamma, &p);
        assert_eq!(prod, [norm, n(0), n(0), n(0)]);
    }

    /// `principal_left_ideal_from_o0(1)` = `O_0` (the full order).
    #[test]
    fn principal_of_one_is_full_order() {
        let p = fake_p();
        let one = [n(1), n(0), n(0), n(0)];
        let ideal = principal_left_ideal_from_o0(&one, &p);
        let full = crate::quaternion::LeftIdeal::<8>::full_order();
        assert!(ideal.equals_lattice(&full));
        assert_eq!(ideal.norm(), Uint::<8>::from_u64(1));
    }

    /// `principal_left_ideal_from_o0(e_3)` has norm `N_red(e_3)² = 4` for p=7.
    #[test]
    fn principal_norm_is_reduced_norm_squared() {
        let p = fake_p();
        let e3 = [n(0), n(0), n(0), n(1)];
        let ideal = principal_left_ideal_from_o0(&e3, &p);
        // N_red(e_3) = 2, so norm should be 4.
        assert_eq!(ideal.norm(), Uint::<8>::from_u64(4));
    }

    /// Standard `(1, 0, 0, 0) = 1` → `O_0`-coords `(1, 0, 0, 0)`.
    #[test]
    fn standard_one_to_o0_is_e0() {
        let q = Quaternion::<8>::new(n(1), n(0), n(0), n(0));
        assert_eq!(standard_to_o0_basis(&q), [n(1), n(0), n(0), n(0)]);
    }

    /// Standard `j = (0, 0, 1, 0)` → `O_0`-coords `(0, -1, 2, 0)`.
    #[test]
    fn standard_j_to_o0() {
        let q = Quaternion::<8>::new(n(0), n(0), n(1), n(0));
        assert_eq!(standard_to_o0_basis(&q), [n(0), n(-1), n(2), n(0)]);
    }

    /// Standard `k = (0, 0, 0, 1)` → `O_0`-coords `(-1, 0, 0, 2)`.
    #[test]
    fn standard_k_to_o0() {
        let q = Quaternion::<8>::new(n(0), n(0), n(0), n(1));
        assert_eq!(standard_to_o0_basis(&q), [n(-1), n(0), n(0), n(2)]);
    }

    /// Round-trip via doubling: `O_0`-coords `→ 2·x` in standard → recover
    /// `O_0`-coords of `2·x` (which is `2·` the original, but expressed in
    /// `O_0` basis as `2·(a, b, c, d)` with a wrinkle from the basis-change).
    #[test]
    fn round_trip_via_doubling() {
        // Take γ = e_3 = (0, 0, 0, 1) in O_0 coords; doubled is 2·e_3 = (1+k)
        // with standard coords (1, 0, 0, 1). Now standard_to_o0 of that
        // gives o_0=1-1=0, o_1=0-0=0, o_2=0, o_3=2. So (0, 0, 0, 2) =
        // 2·e_3 in O_0 coords. ✓
        let gamma = [n(0), n(0), n(0), n(1)];
        let doubled = o0_basis_to_standard_doubled(&gamma);
        let recovered = standard_to_o0_basis(&doubled);
        // 2 * gamma in O_0 coords.
        assert_eq!(recovered, [n(0), n(0), n(0), n(2)]);
    }

    /// `o0_basis_to_standard_doubled(1) = (2, 0, 0, 0)` (i.e., `2·1 = 2`).
    #[test]
    fn o0_one_to_doubled_standard() {
        let one = [n(1), n(0), n(0), n(0)];
        let doubled = o0_basis_to_standard_doubled(&one);
        assert_eq!(doubled, Quaternion::<8>::new(n(2), n(0), n(0), n(0)));
    }

    /// `principal_left_ideal_from_o0(2·1)` = `2·O_0` with norm `16`.
    #[test]
    fn principal_of_two_is_doubled_order() {
        let p = fake_p();
        let two = [n(2), n(0), n(0), n(0)];
        let ideal = principal_left_ideal_from_o0(&two, &p);
        let doubled = crate::quaternion::LeftIdeal::<8>::full_order().scale(2);
        assert!(ideal.equals_lattice(&doubled));
        assert_eq!(ideal.norm(), Uint::<8>::from_u64(16));
    }

    /// Associativity probe: `(e_1 · e_2) · e_0 = e_1 · (e_2 · e_0)`.
    #[test]
    fn associativity_e1_e2_e0() {
        let e0 = [n(1), n(0), n(0), n(0)];
        let e1 = [n(0), n(1), n(0), n(0)];
        let e2 = [n(0), n(0), n(1), n(0)];
        let p = fake_p();
        let lhs_inner = multiply_o0_basis(&e1, &e2, &p);
        let lhs = multiply_o0_basis(&lhs_inner, &e0, &p);
        let rhs_inner = multiply_o0_basis(&e2, &e0, &p);
        let rhs = multiply_o0_basis(&e1, &rhs_inner, &p);
        assert_eq!(lhs, rhs);
    }
}
