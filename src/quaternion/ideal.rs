// SPDX-License-Identifier: MIT OR Apache-2.0
//! Left `O_0`-ideals — the arithmetic primitive KLPT operates on.
//!
//! An *integral left `O_0`-ideal* is a `Z`-submodule of `O_0` that is closed
//! under left multiplication by `O_0`. Every such ideal has rank 4 over `Z`
//! and is therefore representable by a 4×4 integer matrix `M` whose rows are
//! `Z`-coordinate vectors of a `Z`-basis of `I`, expressed in the canonical
//! `O_0`-basis `(1, i, (i + j)/2, (1 + k)/2)`.
//!
//! Conjugation acts on the `O_0` basis as the linear map
//!
//! ```text
//!     C = [ 1  0  0  1 ]
//!         [ 0 -1  0  0 ]
//!         [ 0  0 -1  0 ]
//!         [ 0  0  0 -1 ]
//! ```
//!
//! (`conj(1) = 1`, `conj(i) = −i`, `conj((i+j)/2) = −(i+j)/2`,
//!  `conj((1+k)/2) = 1 − (1+k)/2 = basis[0] − basis[3]`).
//!
//! HNF reduction, ideal multiplication, and `random_equivalent_ideal` will
//! land alongside the KLPT session — this module ships the storage type and
//! the norm / conjugation that KLPT (and the immediate next session's
//! prototype) need.

use crypto_bigint::Int;

/// A `Z`-basis matrix for a left `O_0`-ideal, expressed in the canonical
/// `O_0`-basis `(1, i, (i+j)/2, (1+k)/2)`.
///
/// `LIMBS` is the integer-coefficient width — the same `Int<LIMBS>` used by
/// [`Quaternion`](super::Quaternion).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct LeftIdeal<const LIMBS: usize> {
    /// `basis[r]` is the `r`-th `Z`-generator of the ideal, with
    /// `basis[r][c]` being its coefficient in the `c`-th `O_0`-basis vector.
    pub basis: [[Int<LIMBS>; 4]; 4],
}

impl<const LIMBS: usize> LeftIdeal<LIMBS> {
    /// Construct an ideal from its raw 4×4 basis matrix.
    #[inline]
    pub const fn new(basis: [[Int<LIMBS>; 4]; 4]) -> Self {
        Self { basis }
    }

    /// The full order `O_0` itself, represented by the identity matrix.
    pub fn full_order() -> Self {
        let z = Int::<LIMBS>::from_i64(0);
        let o = Int::<LIMBS>::from_i64(1);
        Self::new([[o, z, z, z], [z, o, z, z], [z, z, o, z], [z, z, z, o]])
    }

    /// Scale every basis vector by an integer constant `n`. The resulting
    /// ideal is `n · self` (note its norm is `n^4 · norm(self)`).
    pub fn scale(&self, n: i64) -> Self {
        let n_int = Int::<LIMBS>::from_i64(n);
        let mut out = self.basis;
        for row in out.iter_mut() {
            for cell in row.iter_mut() {
                *cell = cell.wrapping_mul(&n_int);
            }
        }
        Self::new(out)
    }

    /// Conjugate ideal: each basis row `r` becomes `r · C` where `C` is the
    /// conjugation matrix shown at the module top.
    pub fn conjugate(&self) -> Self {
        let mut out = self.basis;
        for row in out.iter_mut() {
            let r0 = row[0];
            let r1 = row[1];
            let r2 = row[2];
            let r3 = row[3];
            // r · C, with C as documented.
            row[0] = r0.wrapping_add(&r3);
            row[1] = r1.wrapping_neg();
            row[2] = r2.wrapping_neg();
            row[3] = r3.wrapping_neg();
        }
        Self::new(out)
    }

    /// Reduced norm `N(I) = |det(M)|` — the index `[O_0 : I]` for integral
    /// ideals contained in `O_0`. Returns an unsigned `crypto_bigint::Uint`.
    pub fn norm(&self) -> crypto_bigint::Uint<LIMBS> {
        let det = det_4x4(&self.basis);
        det.abs()
    }

    /// Reduce the basis matrix to Hermite Normal Form. Returns a new
    /// `LeftIdeal` representing the same lattice as `self` but in canonical
    /// upper-triangular form.
    pub fn reduced(&self) -> Self {
        Self::new(crate::quaternion::hnf::hnf_4x4(&self.basis))
    }

    /// Test whether two ideals represent the same lattice. Cheap via HNF:
    /// two integer lattices are equal iff their HNFs are bitwise-equal.
    pub fn equals_lattice(&self, other: &Self) -> bool {
        self.reduced().basis == other.reduced().basis
    }

    /// Divide every basis coordinate by the integer `divisor`. Returns
    /// `None` if the division is not exact (i.e., any cell is not an
    /// integer multiple of `divisor`).
    ///
    /// Used by KLPT's general-case lift to extract `I · γ̄ / N(I)` after
    /// constructing the unnormalised right-product.
    pub fn divide_basis_by(&self, divisor: i64) -> Option<Self> {
        if divisor == 0 {
            return None;
        }
        let div_int = Int::<LIMBS>::from_i64(divisor);
        let mut out = self.basis;
        for row in out.iter_mut() {
            for cell in row.iter_mut() {
                let q = crate::quaternion::hnf::int_div_floor(cell, &div_int);
                let recovered = q.wrapping_mul(&div_int);
                if recovered != *cell {
                    return None;
                }
                *cell = q;
            }
        }
        Some(Self::new(out))
    }

    /// Test whether the quaternion `q` (in `O_0`-basis coordinates) is an
    /// element of the lattice represented by `self`. Implementation:
    /// reduce `self` to upper-triangular HNF, then reduce `q` against the
    /// HNF basis column-by-column; `q ∈ self` iff the residual is zero.
    pub fn contains(&self, q: &[Int<LIMBS>; 4]) -> bool {
        let h = crate::quaternion::hnf::hnf_4x4(&self.basis);
        let mut r = *q;
        let zero = Int::<LIMBS>::from_i64(0);
        for c in 0..4 {
            let pivot = h[c][c];
            if pivot == zero {
                // Null pivot — column `c` is unconstrained. `q[c]` only
                // satisfies membership if it's likewise zero after the
                // upstream eliminations.
                if r[c] != zero {
                    return false;
                }
                continue;
            }
            // Need r[c] to be an integer multiple of pivot.
            let t = crate::quaternion::hnf::int_div_floor(&r[c], &pivot);
            let t_pivot = t.wrapping_mul(&pivot);
            if t_pivot != r[c] {
                return false;
            }
            for k in c..4 {
                let delta = t.wrapping_mul(&h[c][k]);
                r[k] = r[k].wrapping_sub(&delta);
            }
        }
        // Whole residual must be zero — anything left means `q ∉ self`.
        r.iter().all(|&v| v == zero)
    }
}

/// Determinant of a 4×4 integer matrix via Laplace expansion along the
/// first row. Wrapping arithmetic — the caller picks `LIMBS` wide enough
/// that the intermediate products don't overflow.
pub fn det_4x4<const LIMBS: usize>(m: &[[Int<LIMBS>; 4]; 4]) -> Int<LIMBS> {
    let mut acc = Int::<LIMBS>::from_i64(0);
    let mut sign = 1i64;
    for c in 0..4 {
        let minor = minor_3x3(m, c);
        let term = m[0][c].wrapping_mul(&det_3x3(&minor));
        if sign == 1 {
            acc = acc.wrapping_add(&term);
        } else {
            acc = acc.wrapping_sub(&term);
        }
        sign = -sign;
    }
    acc
}

fn minor_3x3<const LIMBS: usize>(
    m: &[[Int<LIMBS>; 4]; 4],
    skip_col: usize,
) -> [[Int<LIMBS>; 3]; 3] {
    let mut out = [[Int::<LIMBS>::from_i64(0); 3]; 3];
    for (row_out, row_src) in out.iter_mut().zip(m.iter().skip(1)) {
        let mut col = 0usize;
        for (c, val) in row_src.iter().enumerate() {
            if c == skip_col {
                continue;
            }
            row_out[col] = *val;
            col += 1;
        }
    }
    out
}

fn det_3x3<const LIMBS: usize>(m: &[[Int<LIMBS>; 3]; 3]) -> Int<LIMBS> {
    // Sarrus' rule.
    let a = m[0][0].wrapping_mul(&m[1][1]).wrapping_mul(&m[2][2]);
    let b = m[0][1].wrapping_mul(&m[1][2]).wrapping_mul(&m[2][0]);
    let c = m[0][2].wrapping_mul(&m[1][0]).wrapping_mul(&m[2][1]);
    let d = m[0][2].wrapping_mul(&m[1][1]).wrapping_mul(&m[2][0]);
    let e = m[0][0].wrapping_mul(&m[1][2]).wrapping_mul(&m[2][1]);
    let f = m[0][1].wrapping_mul(&m[1][0]).wrapping_mul(&m[2][2]);
    a.wrapping_add(&b)
        .wrapping_add(&c)
        .wrapping_sub(&d)
        .wrapping_sub(&e)
        .wrapping_sub(&f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::Uint;

    type Ideal = LeftIdeal<8>;

    fn z() -> Int<8> {
        Int::<8>::from_i64(0)
    }
    fn i64_int(n: i64) -> Int<8> {
        Int::<8>::from_i64(n)
    }

    #[test]
    fn full_order_norm_is_one() {
        let i = Ideal::full_order();
        assert_eq!(i.norm(), Uint::<8>::from_u64(1));
    }

    #[test]
    fn doubled_order_norm_is_sixteen() {
        // 2 · O_0 has basis matrix 2 · I_4, det = 2^4 = 16.
        let i = Ideal::full_order().scale(2);
        assert_eq!(i.norm(), Uint::<8>::from_u64(16));
    }

    #[test]
    fn det_4x4_diagonal() {
        let m = [
            [i64_int(2), z(), z(), z()],
            [z(), i64_int(3), z(), z()],
            [z(), z(), i64_int(5), z()],
            [z(), z(), z(), i64_int(7)],
        ];
        assert_eq!(det_4x4(&m), i64_int(2 * 3 * 5 * 7));
    }

    #[test]
    fn det_4x4_known_value() {
        // m = [[3,1,0,2],[2,4,1,0],[0,1,5,1],[1,0,2,3]] → det = 74
        // (computed independently via Python reference implementation).
        let m = [
            [i64_int(3), i64_int(1), z(), i64_int(2)],
            [i64_int(2), i64_int(4), i64_int(1), z()],
            [z(), i64_int(1), i64_int(5), i64_int(1)],
            [i64_int(1), z(), i64_int(2), i64_int(3)],
        ];
        assert_eq!(det_4x4(&m), i64_int(74));
    }

    #[test]
    fn conjugate_is_involution_on_full_order() {
        let i = Ideal::full_order();
        let i_cc = i.conjugate().conjugate();
        // C · C should be identity → ideal unchanged.
        assert_eq!(i_cc.basis, i.basis);
    }

    #[test]
    fn conjugate_preserves_norm_of_full_order() {
        let i = Ideal::full_order();
        assert_eq!(i.conjugate().norm(), i.norm());
    }

    #[test]
    fn conjugate_preserves_norm_of_scaled() {
        let i = Ideal::full_order().scale(3);
        assert_eq!(i.conjugate().norm(), i.norm());
    }

    #[test]
    fn scale_norm_is_power() {
        // (n · I).norm = n^4 · I.norm
        let one = Ideal::full_order();
        assert_eq!(one.scale(5).norm(), Uint::<8>::from_u64(5u64.pow(4)));
    }

    #[test]
    fn full_order_reduced_equals_itself() {
        let i = Ideal::full_order();
        assert_eq!(i.reduced().basis, i.basis);
    }

    #[test]
    fn divide_doubled_order_by_two_is_full_order() {
        let id = Ideal::full_order();
        let two_id = id.scale(2);
        let divided = two_id.divide_basis_by(2).expect("division is exact");
        assert!(divided.equals_lattice(&id));
        assert_eq!(divided.norm(), Uint::<8>::from_u64(1));
    }

    #[test]
    fn divide_by_non_divisor_returns_none() {
        let id = Ideal::full_order();
        // O_0 has identity basis; dividing by 2 is not exact (1 is not divisible by 2).
        assert!(id.divide_basis_by(2).is_none());
    }

    #[test]
    fn divide_by_zero_returns_none() {
        let id = Ideal::full_order();
        assert!(id.divide_basis_by(0).is_none());
    }

    #[test]
    fn divide_then_scale_round_trip() {
        let id = Ideal::full_order();
        let scaled = id.scale(7);
        let divided = scaled.divide_basis_by(7).expect("division is exact");
        assert!(divided.equals_lattice(&id));
    }

    #[test]
    fn divide_negative_divisor() {
        let id = Ideal::full_order();
        let scaled = id.scale(5);
        let divided = scaled.divide_basis_by(-5).expect("division is exact");
        // The lattice is the same (negating basis vectors doesn't change the lattice).
        assert!(divided.equals_lattice(&id));
    }

    #[test]
    fn full_order_contains_every_integer_quaternion() {
        let id = Ideal::full_order();
        // (1, 0, 0, 0), (0, 1, 0, 0), etc. all in O_0.
        for c in 0..4 {
            let mut q = [z(); 4];
            q[c] = i64_int(1);
            assert!(id.contains(&q), "O_0 should contain basis vector {c}");
        }
        // (3, -5, 7, -2) ∈ O_0.
        assert!(id.contains(&[i64_int(3), i64_int(-5), i64_int(7), i64_int(-2)]));
    }

    #[test]
    fn doubled_order_excludes_odd_coordinates() {
        // 2·O_0 contains only quaternions whose O_0-coords are all even.
        let two_id = Ideal::full_order().scale(2);
        // (2, 0, 0, 0) ∈ 2·O_0; (1, 0, 0, 0) ∉ 2·O_0.
        assert!(two_id.contains(&[i64_int(2), z(), z(), z()]));
        assert!(!two_id.contains(&[i64_int(1), z(), z(), z()]));
        assert!(two_id.contains(&[i64_int(4), i64_int(6), i64_int(8), i64_int(-2)]));
        assert!(!two_id.contains(&[i64_int(4), i64_int(6), i64_int(8), i64_int(-3)]));
    }

    #[test]
    fn contains_zero_is_always_true() {
        let id = Ideal::full_order();
        assert!(id.contains(&[z(), z(), z(), z()]));
        let scaled = id.scale(17);
        assert!(scaled.contains(&[z(), z(), z(), z()]));
    }

    #[test]
    fn permuted_basis_reduces_to_canonical() {
        // The identity matrix and a row-swapped identity represent the same
        // lattice; their HNFs must agree.
        let id = Ideal::full_order();
        let mut perm = id;
        perm.basis.swap(0, 2);
        perm.basis.swap(1, 3);
        assert!(id.equals_lattice(&perm));
    }

    #[test]
    fn doubled_basis_is_distinct_lattice() {
        let id = Ideal::full_order();
        let doubled = id.scale(2);
        // 2·O_0 is a strict sublattice of O_0; they must NOT be equal.
        assert!(!id.equals_lattice(&doubled));
    }

    #[test]
    fn det_3x3_known() {
        let m = [
            [i64_int(1), i64_int(2), i64_int(3)],
            [i64_int(4), i64_int(5), i64_int(6)],
            [i64_int(7), i64_int(8), i64_int(10)],
        ];
        // det = 1*(5*10 - 6*8) - 2*(4*10 - 6*7) + 3*(4*8 - 5*7) = 2 + 4 - 9 = -3
        assert_eq!(det_3x3(&m), i64_int(-3));
    }
}
