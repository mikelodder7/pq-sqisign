// SPDX-License-Identifier: MIT OR Apache-2.0
//! Hermite Normal Form (HNF) reduction of 4×4 integer matrices.
//!
//! Every full-rank integer matrix `M ∈ Z^{4×4}` is row-equivalent (via
//! left-multiplication by a unimodular matrix `U ∈ GL_4(Z)`) to a unique
//! upper-triangular matrix `H = U · M` with:
//!
//! 1. `H[i][i] > 0` for every diagonal entry.
//! 2. `0 ≤ H[i][j] < H[i][i]` for `j > i` (off-diagonals strictly smaller
//!    than the pivot below them in the same column).
//!
//! `H` is the **Hermite Normal Form** of `M`.
//!
//! KLPT's equivalent-ideal lift, ideal multiplication, and the
//! lattice-membership test all reduce to HNF. The standard algorithm is
//! row-by-row column elimination via the extended Euclidean algorithm.
//!
//! Coefficients use `Int<LIMBS>` from `crypto-bigint` so the wider
//! quaternion arithmetic shares one integer type.

use crypto_bigint::Int;

/// HNF of an `ROWS × 4` integer matrix. The top 4 rows of the result form
/// the canonical upper-triangular basis; rows 4..ROWS are zero.
///
/// Used by `ideal_multiply` to reduce the 16-pair product matrix.
#[allow(clippy::needless_range_loop)]
pub fn hnf_rect_4cols<const ROWS: usize, const LIMBS: usize>(
    input: &[[Int<LIMBS>; 4]; ROWS],
) -> [[Int<LIMBS>; 4]; ROWS] {
    let mut m = *input;
    for col in 0..4 {
        loop {
            let mut min_idx: Option<usize> = None;
            let mut min_abs = Int::<LIMBS>::from_i64(0);
            for r in col..ROWS {
                let v = m[r][col];
                let zero = Int::<LIMBS>::from_i64(0);
                if v == zero {
                    continue;
                }
                let abs_v = v.abs();
                let abs_min = min_abs.abs();
                if min_idx.is_none() || abs_v < abs_min {
                    min_idx = Some(r);
                    min_abs = v;
                }
            }
            let Some(pivot_row) = min_idx else {
                break;
            };
            if pivot_row != col {
                m.swap(col, pivot_row);
            }
            let pivot = m[col][col];
            if bool::from(pivot.is_negative()) {
                for c in 0..4 {
                    m[col][c] = m[col][c].wrapping_neg();
                }
            }
            let pivot = m[col][col];
            let mut any_nonzero_below = false;
            for r in (col + 1)..ROWS {
                let entry = m[r][col];
                let zero = Int::<LIMBS>::from_i64(0);
                if entry == zero {
                    continue;
                }
                let q = int_div_floor(&entry, &pivot);
                for c in 0..4 {
                    let qm = q.wrapping_mul(&m[col][c]);
                    m[r][c] = m[r][c].wrapping_sub(&qm);
                }
                if m[r][col] != zero {
                    any_nonzero_below = true;
                }
            }
            if !any_nonzero_below {
                break;
            }
        }
        // Reduce rows above the pivot mod the pivot value.
        let pivot = m[col][col];
        let zero = Int::<LIMBS>::from_i64(0);
        if pivot != zero {
            for r in 0..col {
                let entry = m[r][col];
                if entry == zero {
                    continue;
                }
                let q = int_div_floor(&entry, &pivot);
                for c in 0..4 {
                    let qm = q.wrapping_mul(&m[col][c]);
                    m[r][c] = m[r][c].wrapping_sub(&qm);
                }
            }
        }
    }
    m
}

/// Compute the HNF of a 4×4 integer matrix.
///
/// **Note**: the algorithm here is the textbook row-reduction (no LLL-style
/// size-control), which means intermediate entries can grow exponentially in
/// adversarial inputs. KLPT operates on ideals whose bases come out of
/// Cornacchia-and-conjugation chains, where this is acceptable; the
/// fraction-free / Bareiss variant lands when profiling shows overflow.
#[allow(clippy::needless_range_loop)]
pub fn hnf_4x4<const LIMBS: usize>(input: &[[Int<LIMBS>; 4]; 4]) -> [[Int<LIMBS>; 4]; 4] {
    let mut m = *input;
    // Process columns left-to-right; pivot row index `i` runs alongside.
    for col in 0..4 {
        // Use rows `col..4` as the active block.
        // Step 1: bring a non-zero entry to position (col, col).
        // Repeatedly Euclidean-reduce the column entries below the pivot.
        loop {
            // Find the smallest non-zero |entry| in rows col..4 at this column.
            let mut min_idx: Option<usize> = None;
            let mut min_abs = Int::<LIMBS>::from_i64(0);
            for r in col..4 {
                let v = m[r][col];
                let zero = Int::<LIMBS>::from_i64(0);
                if v == zero {
                    continue;
                }
                let abs_v = v.abs();
                let abs_min = min_abs.abs();
                if min_idx.is_none() || abs_v < abs_min {
                    min_idx = Some(r);
                    min_abs = v;
                }
            }
            let Some(pivot_row) = min_idx else {
                // All entries in this column at rows col..4 are zero; pivot stays 0.
                break;
            };
            // Swap pivot row to position `col`.
            if pivot_row != col {
                m.swap(col, pivot_row);
            }
            let pivot = m[col][col];
            // Make the pivot positive (sign-flip the row if needed).
            if bool::from(pivot.is_negative()) {
                for c in 0..4 {
                    m[col][c] = m[col][c].wrapping_neg();
                }
            }
            let pivot = m[col][col];
            // Reduce every other row in this column by div-mod with the pivot.
            let mut any_nonzero_below = false;
            for r in (col + 1)..4 {
                let entry = m[r][col];
                let zero = Int::<LIMBS>::from_i64(0);
                if entry == zero {
                    continue;
                }
                // q = entry div pivot (toward zero); we use repeated subtraction
                // because `Int::div` is a more involved interface; pivots in this
                // context are small.
                let q = int_div_floor(&entry, &pivot);
                let q_times_pivot = q.wrapping_mul(&pivot);
                for c in 0..4 {
                    let qm = q.wrapping_mul(&m[col][c]);
                    m[r][c] = m[r][c].wrapping_sub(&qm);
                }
                let _ = q_times_pivot;
                // After this, m[r][col] = entry - q*pivot ∈ [0, pivot).
                if m[r][col] != zero {
                    any_nonzero_below = true;
                }
            }
            if !any_nonzero_below {
                break;
            }
        }
        // Step 2: reduce rows above the pivot mod the pivot value.
        let pivot = m[col][col];
        let zero = Int::<LIMBS>::from_i64(0);
        if pivot != zero {
            for r in 0..col {
                let entry = m[r][col];
                if entry == zero {
                    continue;
                }
                let q = int_div_floor(&entry, &pivot);
                for c in 0..4 {
                    let qm = q.wrapping_mul(&m[col][c]);
                    m[r][c] = m[r][c].wrapping_sub(&qm);
                }
            }
        }
    }
    m
}

/// Integer floor-division: returns `⌊a / b⌋` for non-zero `b`, `0` when
/// `b == 0`.
///
/// Strategy: normalise both operands to unsigned via `Int::abs_sign`,
/// delegate to `crypto-bigint`'s `Uint::div_rem_vartime` (Knuth Algorithm D
/// under the hood — `O(LIMBS · 64)` work regardless of the quotient
/// magnitude), then apply the floor-vs-truncate adjustment when exactly one
/// operand was negative. The `crypto-bigint 0.7.x` *signed* division
/// surface is still unstable, so we route around it via the stable
/// `Uint` surface and reinterpret the result via `Uint::as_int`.
pub fn int_div_floor<const LIMBS: usize>(a: &Int<LIMBS>, b: &Int<LIMBS>) -> Int<LIMBS> {
    use crypto_bigint::NonZero;
    let zero = Int::<LIMBS>::from_i64(0);
    let one = Int::<LIMBS>::from_i64(1);
    if *b == zero {
        return zero;
    }
    let (a_abs, a_neg) = a.abs_sign();
    let (b_abs, b_neg) = b.abs_sign();
    let result_neg = bool::from(a_neg) ^ bool::from(b_neg);
    let Some(b_nz) = Option::<NonZero<_>>::from(NonZero::new(b_abs)) else {
        return zero;
    };
    let (q, r) = a_abs.div_rem_vartime(&b_nz);
    let q_int = *q.as_int();
    let zero_u = crypto_bigint::Uint::<LIMBS>::from_u64(0);
    if result_neg {
        if r == zero_u {
            q_int.wrapping_neg()
        } else {
            q_int.wrapping_neg().wrapping_sub(&one)
        }
    } else {
        q_int
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    type I = Int<8>;

    fn n(x: i64) -> I {
        I::from_i64(x)
    }

    fn identity() -> [[I; 4]; 4] {
        [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ]
    }

    #[test]
    fn hnf_of_identity_is_identity() {
        let h = hnf_4x4(&identity());
        assert_eq!(h, identity());
    }

    #[test]
    fn hnf_of_diagonal_is_diagonal() {
        let m = [
            [n(2), n(0), n(0), n(0)],
            [n(0), n(3), n(0), n(0)],
            [n(0), n(0), n(5), n(0)],
            [n(0), n(0), n(0), n(7)],
        ];
        let h = hnf_4x4(&m);
        assert_eq!(h, m);
    }

    #[test]
    fn hnf_pivots_are_positive() {
        let m = [
            [n(-2), n(1), n(0), n(0)],
            [n(0), n(-3), n(0), n(0)],
            [n(0), n(0), n(5), n(0)],
            [n(0), n(0), n(0), n(-7)],
        ];
        let h = hnf_4x4(&m);
        for i in 0..4 {
            assert!(
                !bool::from(h[i][i].is_negative()),
                "diagonal[{i}] should be non-negative"
            );
        }
    }

    #[test]
    fn hnf_lower_triangle_is_zero() {
        // Swap a couple of rows in identity; HNF must restore upper-triangular.
        let mut m = identity();
        m.swap(0, 2);
        m.swap(1, 3);
        let h = hnf_4x4(&m);
        for r in 0..4 {
            for c in 0..r {
                assert_eq!(h[r][c], n(0), "lower triangle nonzero at ({r}, {c})");
            }
        }
    }

    #[test]
    fn int_div_floor_basic() {
        assert_eq!(int_div_floor(&n(7), &n(2)), n(3));
        assert_eq!(int_div_floor(&n(-7), &n(2)), n(-4)); // floor, not trunc
        assert_eq!(int_div_floor(&n(7), &n(-2)), n(-4));
        assert_eq!(int_div_floor(&n(-7), &n(-2)), n(3));
        assert_eq!(int_div_floor(&n(0), &n(5)), n(0));
        assert_eq!(int_div_floor(&n(5), &n(0)), n(0));
        assert_eq!(int_div_floor(&n(10), &n(5)), n(2));
        assert_eq!(int_div_floor(&n(-10), &n(5)), n(-2));
    }

    #[test]
    fn int_div_floor_handles_real_prime_scale_quotient() {
        // 2^200 / 2^100 = 2^100. The pre-Session-35 repeated-subtraction body
        // would have needed 2^100 iterations to finish (≈ heat-death of the
        // universe). The Knuth-Algorithm-D-backed body returns in microseconds.
        let num_u = crypto_bigint::Uint::<8>::ONE.shl_vartime(200);
        let den_u = crypto_bigint::Uint::<8>::ONE.shl_vartime(100);
        let expected_u = crypto_bigint::Uint::<8>::ONE.shl_vartime(100);
        let num: Int<8> = *num_u.as_int();
        let den: Int<8> = *den_u.as_int();
        let expected: Int<8> = *expected_u.as_int();
        assert_eq!(int_div_floor(&num, &den), expected);
    }

    #[test]
    fn int_div_floor_large_negative_quotient() {
        // -(2^200) / 2^100 = -(2^100). Exact division — floor adjustment does
        // not subtract one because the remainder is zero.
        let num_u = crypto_bigint::Uint::<8>::ONE.shl_vartime(200);
        let den_u = crypto_bigint::Uint::<8>::ONE.shl_vartime(100);
        let expected_u = crypto_bigint::Uint::<8>::ONE.shl_vartime(100);
        let num: Int<8> = (*num_u.as_int()).wrapping_neg();
        let den: Int<8> = *den_u.as_int();
        let expected: Int<8> = (*expected_u.as_int()).wrapping_neg();
        assert_eq!(int_div_floor(&num, &den), expected);
    }

    #[test]
    fn int_div_floor_large_negative_quotient_with_remainder() {
        // -(2^200 + 1) / 2^100 = -(2^100) − 1   (floor of −(2^100 + 2^(−100)))
        let num_u = crypto_bigint::Uint::<8>::ONE
            .shl_vartime(200)
            .wrapping_add(&crypto_bigint::Uint::<8>::ONE);
        let den_u = crypto_bigint::Uint::<8>::ONE.shl_vartime(100);
        let num: Int<8> = (*num_u.as_int()).wrapping_neg();
        let den: Int<8> = *den_u.as_int();
        let expected: Int<8> = (*crypto_bigint::Uint::<8>::ONE.shl_vartime(100).as_int())
            .wrapping_neg()
            .wrapping_sub(&Int::<8>::from_i64(1));
        assert_eq!(int_div_floor(&num, &den), expected);
    }
}
