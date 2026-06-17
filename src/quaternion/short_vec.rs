// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shortest-quaternion search in a left `O_0`-ideal.
//!
//! KLPT repeatedly asks for a non-zero quaternion `γ ∈ I` with small
//! reduced norm. The "real" answer uses LLL on the basis matrix to find a
//! short vector in `O((log p)²)` time. This module provides the
//! **brute-force prototype**: enumerate every `Z`-combination of basis
//! vectors with bounded coefficients, compute the reduced norm, return
//! the minimiser.
//!
//! Bounds: with coefficient bound `B`, this checks `(2B+1)⁴` candidates,
//! computes one `reduced_norm_o0_basis` each — fine for small fake-prime
//! tests, useless at real SQIsign scale. The LLL-based fast path lives in
//! the lattice module and shares this module's public surface.

use crypto_bigint::{Int, Uint};

use crate::quaternion::ideal::LeftIdeal;
use crate::quaternion::o0_mul::reduced_norm_o0_basis;

/// Search the left ideal `ideal` for a non-zero quaternion with **exactly**
/// the reduced norm `target_norm`, scanning `Z`-combinations of basis
/// vectors with per-coordinate bound `|n_i| ≤ search_bound`.
///
/// Returns `Some(coords_in_o0_basis)` of the first witness found, or
/// `None` if the search space contained no candidate.
///
/// KLPT's general lift consumes this primitive to find `γ ∈ I` with
/// `N_red(γ) = N(I) · T`, after which `J = I · γ̄ / N(I)` has norm `T`.
#[allow(clippy::needless_range_loop)]
pub fn find_quaternion_in_ideal_with_norm<const LIMBS: usize>(
    ideal: &LeftIdeal<LIMBS>,
    target_norm: i64,
    p: &Uint<LIMBS>,
    search_bound: i64,
) -> Option<[Int<LIMBS>; 4]> {
    let zero = Int::<LIMBS>::from_i64(0);
    let target_int = Int::<LIMBS>::from_i64(target_norm);
    let bound = search_bound;
    for n0 in -bound..=bound {
        for n1 in -bound..=bound {
            for n2 in -bound..=bound {
                for n3 in -bound..=bound {
                    if n0 == 0 && n1 == 0 && n2 == 0 && n3 == 0 {
                        continue;
                    }
                    let n0_int = Int::<LIMBS>::from_i64(n0);
                    let n1_int = Int::<LIMBS>::from_i64(n1);
                    let n2_int = Int::<LIMBS>::from_i64(n2);
                    let n3_int = Int::<LIMBS>::from_i64(n3);
                    let mut coords = [zero; 4];
                    for c in 0..4 {
                        let t0 = n0_int.wrapping_mul(&ideal.basis[0][c]);
                        let t1 = n1_int.wrapping_mul(&ideal.basis[1][c]);
                        let t2 = n2_int.wrapping_mul(&ideal.basis[2][c]);
                        let t3 = n3_int.wrapping_mul(&ideal.basis[3][c]);
                        coords[c] = t0.wrapping_add(&t1).wrapping_add(&t2).wrapping_add(&t3);
                    }
                    let norm = reduced_norm_o0_basis(&coords, p);
                    if norm == target_int {
                        return Some(coords);
                    }
                }
            }
        }
    }
    None
}

/// Search the left ideal `ideal` for a non-zero quaternion with the
/// smallest reduced norm, scanning `Z`-combinations of basis vectors with
/// per-coordinate bound `|n_i| ≤ search_bound`.
///
/// Returns `Some((coords_in_o0_basis, norm))`, or `None` if the search
/// space contained nothing but zero.
#[allow(clippy::needless_range_loop)]
pub fn shortest_quaternion_in_ideal<const LIMBS: usize>(
    ideal: &LeftIdeal<LIMBS>,
    p: &Uint<LIMBS>,
    search_bound: i64,
) -> Option<([Int<LIMBS>; 4], Int<LIMBS>)> {
    let zero = Int::<LIMBS>::from_i64(0);
    let bound = search_bound;
    let mut best: Option<([Int<LIMBS>; 4], Int<LIMBS>)> = None;
    for n0 in -bound..=bound {
        for n1 in -bound..=bound {
            for n2 in -bound..=bound {
                for n3 in -bound..=bound {
                    if n0 == 0 && n1 == 0 && n2 == 0 && n3 == 0 {
                        continue;
                    }
                    let n0_int = Int::<LIMBS>::from_i64(n0);
                    let n1_int = Int::<LIMBS>::from_i64(n1);
                    let n2_int = Int::<LIMBS>::from_i64(n2);
                    let n3_int = Int::<LIMBS>::from_i64(n3);
                    let mut coords = [zero; 4];
                    for c in 0..4 {
                        let t0 = n0_int.wrapping_mul(&ideal.basis[0][c]);
                        let t1 = n1_int.wrapping_mul(&ideal.basis[1][c]);
                        let t2 = n2_int.wrapping_mul(&ideal.basis[2][c]);
                        let t3 = n3_int.wrapping_mul(&ideal.basis[3][c]);
                        coords[c] = t0.wrapping_add(&t1).wrapping_add(&t2).wrapping_add(&t3);
                    }
                    let norm = reduced_norm_o0_basis(&coords, p);
                    let is_better = match &best {
                        None => true,
                        Some((_, best_norm)) => norm.abs() < best_norm.abs(),
                    };
                    if is_better {
                        best = Some((coords, norm));
                    }
                }
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_p() -> Uint<8> {
        Uint::<8>::from_u64(7)
    }

    fn full_order() -> LeftIdeal<8> {
        LeftIdeal::<8>::full_order()
    }

    #[test]
    fn full_order_shortest_is_unit_norm() {
        // The full order contains 1 = e_0 with N_red(1) = 1 — the smallest
        // possible non-zero reduced norm.
        let (coords, norm) = shortest_quaternion_in_ideal(&full_order(), &fake_p(), 1)
            .expect("non-zero vector exists");
        assert_eq!(norm, Int::<8>::from_i64(1));
        // The witness should be one of ±e_0 or ±e_1 (both have norm 1).
        let abs_sum: i64 = coords
            .iter()
            .map(|c| if *c == Int::<8>::from_i64(0) { 0 } else { 1 })
            .sum();
        assert_eq!(abs_sum, 1, "should be a single basis vector");
    }

    #[test]
    fn doubled_order_shortest_is_four() {
        // 2·O_0: shortest non-zero is 2·e_0 with N_red(2) = 4.
        let two_id = full_order().scale(2);
        let (_coords, norm) =
            shortest_quaternion_in_ideal(&two_id, &fake_p(), 1).expect("non-zero vector exists");
        assert_eq!(norm, Int::<8>::from_i64(4));
    }

    #[test]
    fn find_norm_one_in_full_order() {
        let r = find_quaternion_in_ideal_with_norm(&full_order(), 1, &fake_p(), 1)
            .expect("γ=1 has norm 1");
        let norm = reduced_norm_o0_basis(&r, &fake_p());
        assert_eq!(norm, Int::<8>::from_i64(1));
    }

    #[test]
    fn find_norm_two_in_full_order_p_seven() {
        // e_3 has reduced norm 2 at p=7.
        let r =
            find_quaternion_in_ideal_with_norm(&full_order(), 2, &fake_p(), 2).expect("γ exists");
        let norm = reduced_norm_o0_basis(&r, &fake_p());
        assert_eq!(norm, Int::<8>::from_i64(2));
    }

    #[test]
    fn find_norm_four_in_two_o0_p_seven() {
        // In 2·O_0, smallest non-zero is 2·1 with norm 4.
        let two_id = full_order().scale(2);
        let r =
            find_quaternion_in_ideal_with_norm(&two_id, 4, &fake_p(), 1).expect("γ=2·1 has norm 4");
        let norm = reduced_norm_o0_basis(&r, &fake_p());
        assert_eq!(norm, Int::<8>::from_i64(4));
    }

    #[test]
    fn find_norm_three_in_full_order_via_o0_basis() {
        // O_0 admits γ = 1 + (i+j)/2 with O_0-coords (1, 0, 1, 0); reduced
        // norm = a² + b² + ad + bc + (1+p)/4 · (c² + d²) = 1 + 0 + 0 + 0 + 2·1 = 3.
        // (This catches the subtlety that norm 3 IS reachable in O_0 even
        // though no *integer-standard* quaternion has reduced norm 3 at p=7.)
        let r = find_quaternion_in_ideal_with_norm(&full_order(), 3, &fake_p(), 1)
            .expect("γ = 1 + (i+j)/2 has norm 3 at p=7");
        let norm = reduced_norm_o0_basis(&r, &fake_p());
        assert_eq!(norm, Int::<8>::from_i64(3));
    }

    #[test]
    fn find_norm_too_large_returns_none_with_small_bound() {
        // For search_bound = 1 the candidate set is finite. Pick a norm
        // unreachable from {-1, 0, 1}^4 Z-combinations of O_0 basis vectors.
        // The maximum reduced norm over (n_0, n_1, n_2, n_3) ∈ {-1, 0, 1}^4
        // is bounded; norm 1000 won't appear.
        assert!(find_quaternion_in_ideal_with_norm(&full_order(), 1000, &fake_p(), 1).is_none());
    }

    #[test]
    fn reduced_norm_of_one_is_one() {
        let one = [
            Int::<8>::from_i64(1),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        ];
        let n = reduced_norm_o0_basis(&one, &fake_p());
        assert_eq!(n, Int::<8>::from_i64(1));
    }

    #[test]
    fn reduced_norm_of_e3_for_p7() {
        // e_3 = (1+k)/2, N_red(e_3) = (1+p)/4 = 2 at p=7.
        let e3 = [
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(1),
        ];
        let n = reduced_norm_o0_basis(&e3, &fake_p());
        assert_eq!(n, Int::<8>::from_i64(2));
    }

    #[test]
    fn reduced_norm_of_j_is_p() {
        // j in O_0 coords: standard (0, 0, 1, 0). To recover O_0 coords:
        // o_3 = 2·qd = 0, o_2 = 2·qc = 2, o_1 = qb - qc = -1, o_0 = qa - qd = 0.
        // So j has O_0-coords (0, -1, 2, 0). Verify N_red(j) = p = 7.
        let j_o0 = [
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(-1),
            Int::<8>::from_i64(2),
            Int::<8>::from_i64(0),
        ];
        let n = reduced_norm_o0_basis(&j_o0, &fake_p());
        assert_eq!(n, Int::<8>::from_i64(7));
    }
}
