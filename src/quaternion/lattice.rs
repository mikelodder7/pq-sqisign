// SPDX-License-Identifier: MIT OR Apache-2.0
//! Lattice-reduction primitives for 4-D `Z`-lattices.
//!
//! The brute-force quaternion-witness search (`find_quaternion_in_ideal_with_norm`)
//! becomes infeasible at real-prime SQIsign scale. The fix is LLL basis
//! reduction; this module ships its foundational primitive — Babai-style
//! **size reduction** — and the integer-inner-product helpers LLL will
//! also consume.
//!
//! Size reduction repeatedly subtracts integer multiples of earlier basis
//! vectors from later ones to make each `b_j` (for `j > i`) approximately
//! orthogonal to `b_i`. It is *not* the full LLL — there's no Lovász swap
//! step here — but it bounds intermediate growth and produces a basis
//! whose vectors are no longer than the input's longest vector.

use crypto_bigint::Int;

use crate::quaternion::hnf::int_div_floor;

/// Integer inner product `⟨a, b⟩ = Σ aᵢ · bᵢ`.
pub fn dot4<const LIMBS: usize>(a: &[Int<LIMBS>; 4], b: &[Int<LIMBS>; 4]) -> Int<LIMBS> {
    a[0].wrapping_mul(&b[0])
        .wrapping_add(&a[1].wrapping_mul(&b[1]))
        .wrapping_add(&a[2].wrapping_mul(&b[2]))
        .wrapping_add(&a[3].wrapping_mul(&b[3]))
}

/// Squared Euclidean length `‖a‖² = ⟨a, a⟩`.
pub fn norm2<const LIMBS: usize>(a: &[Int<LIMBS>; 4]) -> Int<LIMBS> {
    dot4(a, a)
}

/// Round-to-nearest integer division: `⌊(2n + d) / (2d)⌋` for `d > 0`.
/// Returns 0 if `d == 0`.
fn round_div<const LIMBS: usize>(n: &Int<LIMBS>, d: &Int<LIMBS>) -> Int<LIMBS> {
    let zero = Int::<LIMBS>::from_i64(0);
    if *d == zero {
        return zero;
    }
    // Work with positive denominator; carry sign through numerator.
    let (d_abs, d_neg) = d.abs_sign();
    let (n_abs, n_neg) = n.abs_sign();
    let result_neg = bool::from(n_neg) ^ bool::from(d_neg);
    // q_floor = n_abs / d_abs; remainder = n_abs - q_floor * d_abs.
    // Round-to-nearest: q_round = q_floor + (1 if 2*remainder >= d_abs else 0).
    let d_int = *d_abs.as_int();
    let n_int = *n_abs.as_int();
    let q = int_div_floor(&n_int, &d_int);
    let q_d = q.wrapping_mul(&d_int);
    let remainder = n_int.wrapping_sub(&q_d);
    let two_rem = remainder.wrapping_add(&remainder);
    let one = Int::<LIMBS>::from_i64(1);
    let bumped = if two_rem >= d_int {
        q.wrapping_add(&one)
    } else {
        q
    };
    if result_neg {
        bumped.wrapping_neg()
    } else {
        bumped
    }
}

/// Gram matrix of a 4×4 integer lattice basis: `G[i][j] = ⟨bᵢ, bⱼ⟩`.
///
/// The Gram matrix is symmetric (`G[i][j] = G[j][i]`) and positive
/// semi-definite. Its determinant equals `det(B)²` where `B` is the
/// basis matrix — a load-bearing invariant LLL implementations exploit
/// for integer arithmetic.
///
/// Useful as the integer-arithmetic stand-in for the rational
/// Gram-Schmidt coefficients that classical LLL maintains.
#[allow(clippy::needless_range_loop)]
pub fn gram_matrix_4x4<const LIMBS: usize>(basis: &[[Int<LIMBS>; 4]; 4]) -> [[Int<LIMBS>; 4]; 4] {
    let zero = Int::<LIMBS>::from_i64(0);
    let mut g = [[zero; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            g[i][j] = dot4(&basis[i], &basis[j]);
        }
    }
    g
}

/// Babai-style size reduction on a 4×4 integer basis matrix.
///
/// For each pair `(i, j)` with `i < j`, computes `r = round(⟨bⱼ, bᵢ⟩ / ⟨bᵢ, bᵢ⟩)`
/// and replaces `bⱼ ← bⱼ − r·bᵢ`. This makes each later vector closer to
/// orthogonal to the earlier ones, reducing `‖bⱼ‖` without changing the
/// lattice spanned by the basis (the update is a unimodular row operation).
///
/// **Not** the full LLL — there's no Lovász swap. The result is a basis
/// whose vectors are typically much shorter than the input's, suitable as a
/// pre-step for LLL or as a stand-alone reducer for already-near-orthogonal
/// inputs.
#[allow(clippy::needless_range_loop)]
pub fn size_reduce_4x4<const LIMBS: usize>(input: &[[Int<LIMBS>; 4]; 4]) -> [[Int<LIMBS>; 4]; 4] {
    let mut basis = *input;
    for i in 0..4 {
        let denom = norm2(&basis[i]);
        let zero = Int::<LIMBS>::from_i64(0);
        if denom == zero {
            continue;
        }
        for j in (i + 1)..4 {
            let num = dot4(&basis[j], &basis[i]);
            let r = round_div(&num, &denom);
            if r == zero {
                continue;
            }
            for c in 0..4 {
                let delta = r.wrapping_mul(&basis[i][c]);
                basis[j][c] = basis[j][c].wrapping_sub(&delta);
            }
        }
    }
    basis
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::quaternion::hnf::hnf_4x4;

    fn n(v: i64) -> Int<8> {
        Int::<8>::from_i64(v)
    }

    #[test]
    fn dot_orthogonal_basis_is_zero() {
        let e1 = [n(1), n(0), n(0), n(0)];
        let e2 = [n(0), n(1), n(0), n(0)];
        assert_eq!(dot4(&e1, &e2), n(0));
    }

    #[test]
    fn norm2_of_unit_is_one() {
        let e = [n(1), n(0), n(0), n(0)];
        assert_eq!(norm2(&e), n(1));
    }

    #[test]
    fn norm2_3_4_0_0_is_25() {
        let v = [n(3), n(4), n(0), n(0)];
        assert_eq!(norm2(&v), n(25));
    }

    #[test]
    fn round_div_basic() {
        assert_eq!(round_div(&n(7), &n(3)), n(2)); // 7/3 ≈ 2.33 → 2
        assert_eq!(round_div(&n(8), &n(3)), n(3)); // 8/3 ≈ 2.67 → 3
        assert_eq!(round_div(&n(9), &n(3)), n(3)); // exact
        assert_eq!(round_div(&n(-7), &n(3)), n(-2));
        assert_eq!(round_div(&n(7), &n(-3)), n(-2));
        assert_eq!(round_div(&n(0), &n(5)), n(0));
        assert_eq!(round_div(&n(5), &n(0)), n(0));
    }

    #[test]
    fn round_div_round_to_even_half_case() {
        // 1.5 → 2 (round half up — my impl uses ≥ so rounds up).
        assert_eq!(round_div(&n(3), &n(2)), n(2));
    }

    #[test]
    fn size_reduce_identity_is_unchanged() {
        let id: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let reduced = size_reduce_4x4(&id);
        assert_eq!(reduced, id);
    }

    #[test]
    fn size_reduce_shortens_skew_vector() {
        // Initial basis includes a "skewed" later vector that should be
        // reducible against the first.
        let m: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(7), n(1), n(0), n(0)], // ⟨b₁, b₀⟩ = 7; r = round(7/1) = 7; b₁ → (0, 1, 0, 0).
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let reduced = size_reduce_4x4(&m);
        assert_eq!(reduced[1], [n(0), n(1), n(0), n(0)]);
    }

    #[test]
    fn size_reduce_preserves_lattice() {
        // Size reduction is a unimodular row op → lattice unchanged.
        // Verify via HNF equality.
        let m: [[Int<8>; 4]; 4] = [
            [n(2), n(0), n(0), n(0)],
            [n(5), n(3), n(0), n(0)],
            [n(7), n(11), n(2), n(0)],
            [n(13), n(17), n(19), n(5)],
        ];
        let reduced = size_reduce_4x4(&m);
        // Both reduce to the same HNF.
        let hnf_orig = hnf_4x4(&m);
        let hnf_reduced = hnf_4x4(&reduced);
        assert_eq!(hnf_orig, hnf_reduced);
    }

    #[test]
    fn gram_of_identity_is_identity() {
        let id: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let g = gram_matrix_4x4(&id);
        assert_eq!(g, id);
    }

    #[test]
    fn gram_of_diagonal_is_squared_diagonal() {
        let m: [[Int<8>; 4]; 4] = [
            [n(2), n(0), n(0), n(0)],
            [n(0), n(3), n(0), n(0)],
            [n(0), n(0), n(5), n(0)],
            [n(0), n(0), n(0), n(7)],
        ];
        let g = gram_matrix_4x4(&m);
        let expected: [[Int<8>; 4]; 4] = [
            [n(4), n(0), n(0), n(0)],
            [n(0), n(9), n(0), n(0)],
            [n(0), n(0), n(25), n(0)],
            [n(0), n(0), n(0), n(49)],
        ];
        assert_eq!(g, expected);
    }

    #[test]
    fn gram_is_symmetric() {
        let m: [[Int<8>; 4]; 4] = [
            [n(3), n(1), n(0), n(2)],
            [n(2), n(4), n(1), n(0)],
            [n(0), n(1), n(5), n(1)],
            [n(1), n(0), n(2), n(3)],
        ];
        let g = gram_matrix_4x4(&m);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(g[i][j], g[j][i], "G is symmetric");
            }
        }
    }

    #[test]
    fn gram_determinant_equals_basis_determinant_squared() {
        use crate::quaternion::ideal::det_4x4;
        let m: [[Int<8>; 4]; 4] = [
            [n(3), n(1), n(0), n(2)],
            [n(2), n(4), n(1), n(0)],
            [n(0), n(1), n(5), n(1)],
            [n(1), n(0), n(2), n(3)],
        ];
        let det_b = det_4x4(&m);
        let g = gram_matrix_4x4(&m);
        let det_g = det_4x4(&g);
        // det(B)² = det(G).
        let det_b_sq = det_b.wrapping_mul(&det_b);
        assert_eq!(det_g, det_b_sq);
    }

    #[test]
    fn gram_diagonal_holds_squared_norms() {
        let m: [[Int<8>; 4]; 4] = [
            [n(3), n(4), n(0), n(0)],
            [n(1), n(1), n(1), n(1)],
            [n(2), n(0), n(2), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let g = gram_matrix_4x4(&m);
        // G[i][i] = ⟨bᵢ, bᵢ⟩ = ‖bᵢ‖².
        assert_eq!(g[0][0], n(9 + 16)); // 25
        assert_eq!(g[1][1], n(4));
        assert_eq!(g[2][2], n(8));
        assert_eq!(g[3][3], n(1));
    }

    #[test]
    fn size_reduce_lowers_norm_via_norm2_comparison() {
        // Skewed input: `b₁ = (10, 1, 0, 0)` has norm² = 101.
        // After size reduction against `b₀ = (1, 0, 0, 0)`: r = round(10/1) = 10,
        // b₁ ← (0, 1, 0, 0) with norm² = 1.
        let m: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(10), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let before_b1_norm = norm2(&m[1]);
        assert_eq!(before_b1_norm, n(101));
        let reduced = size_reduce_4x4(&m);
        let after_b1_norm = norm2(&reduced[1]);
        assert_eq!(after_b1_norm, n(1));
        assert!(after_b1_norm < before_b1_norm);
    }
}
