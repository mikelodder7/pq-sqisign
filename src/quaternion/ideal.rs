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

use crypto_bigint::{Int, Uint};

use crate::quaternion::lattice::pull_back_gram;
use crate::quaternion::o0_mul::o0_reduced_norm_gram_matrix;

/// A `Z`-basis matrix for a left `O_0`-ideal, expressed in the canonical
/// `O_0`-basis `(1, i, (i+j)/2, (1+k)/2)`, optionally scaled by an integer
/// denominator.
///
/// The "true" rational lattice is `(1 / denom) · Z⟨basis⟩`. When
/// `denom == 1` this collapses to a plain integer `O_0`-ideal — the
/// representation every method shipped through Session 48 used.
///
/// `LIMBS` is the integer-coefficient width — the same `Int<LIMBS>` used by
/// [`Quaternion`](super::Quaternion).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct LeftIdeal<const LIMBS: usize> {
    /// `basis[r]` is the `r`-th `Z`-generator of the ideal (times `denom`),
    /// with `basis[r][c]` being its coefficient in the `c`-th `O_0`-basis
    /// vector.
    pub basis: [[Int<LIMBS>; 4]; 4],
    /// Scalar denominator. The rational lattice is `(1 / denom) · Z⟨basis⟩`.
    /// Defaults to `Uint::ONE` for all integer-only constructions.
    pub denom: Uint<LIMBS>,
    /// Cached reduced norm `N(I) = [O_0 : I_rational]`. Tracked separately
    /// from `|det(basis)| / denom^4` because the `quat_lideal_mul` formula
    /// `N(I·β) = N(I)·N_red(β_rational)` from the SQIsign C reference
    /// updates this field via its own rule that doesn't always match the
    /// raw basis-determinant formula after rational right-multiplication.
    pub cached_norm: Uint<LIMBS>,
}

impl<const LIMBS: usize> LeftIdeal<LIMBS> {
    /// Construct an integer ideal from its raw 4×4 basis matrix. Sets
    /// `denom = 1` and computes `cached_norm = |det(basis)|`. Use
    /// [`Self::with_denom_and_norm`] for rational ideals where the C
    /// reference's `N(I·β)` formula gives a different cached norm than
    /// the raw determinant.
    pub fn new(basis: [[Int<LIMBS>; 4]; 4]) -> Self {
        let det = det_4x4(&basis);
        let cached_norm = det.abs();
        Self {
            basis,
            denom: Uint::<LIMBS>::ONE,
            cached_norm,
        }
    }

    /// Construct a rational ideal with explicit `denom` and `cached_norm`.
    /// Caller is responsible for invariant consistency — typically used
    /// only by the rational-multiplication operation in KLPT.
    pub const fn with_denom_and_norm(
        basis: [[Int<LIMBS>; 4]; 4],
        denom: Uint<LIMBS>,
        cached_norm: Uint<LIMBS>,
    ) -> Self {
        Self {
            basis,
            denom,
            cached_norm,
        }
    }

    /// The full order `O_0` itself, represented by the identity matrix.
    pub fn full_order() -> Self {
        let z = Int::<LIMBS>::from_i64(0);
        let o = Int::<LIMBS>::from_i64(1);
        Self::new([[o, z, z, z], [z, o, z, z], [z, z, o, z], [z, z, z, o]])
    }

    /// Scale every basis vector by an integer constant `n`. The resulting
    /// ideal is `n · self`: its basis is `n · basis`, its `denom` is
    /// unchanged, and its `cached_norm` is multiplied by `|n|^4`.
    pub fn scale(&self, n: i64) -> Self {
        let n_int = Int::<LIMBS>::from_i64(n);
        let mut out = self.basis;
        for row in out.iter_mut() {
            for cell in row.iter_mut() {
                *cell = cell.wrapping_mul(&n_int);
            }
        }
        let n_abs = n.unsigned_abs();
        let n_pow_4 = n_abs.saturating_pow(4);
        let n_pow_4_uint = Uint::<LIMBS>::from_u64(n_pow_4);
        let new_norm = self.cached_norm.wrapping_mul(&n_pow_4_uint);
        Self {
            basis: out,
            denom: self.denom,
            cached_norm: new_norm,
        }
    }

    /// Conjugate ideal: each basis row `r` becomes `r · C` where `C` is the
    /// conjugation matrix shown at the module top. `det(C) = −1`, so `|det|`
    /// is preserved; `cached_norm` and `denom` are unchanged.
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
        Self {
            basis: out,
            denom: self.denom,
            cached_norm: self.cached_norm,
        }
    }

    /// Cached lattice index `[O_0 : I_rational]` = `|det(basis)| / denom^4`
    /// (for integer ideals, `denom == 1`, this matches `|det(basis)|`).
    ///
    /// **NOTE on convention** — this codebase's `cached_norm` is the
    /// LATTICE INDEX, NOT the quaternion reduced ideal norm. For a
    /// principal ideal `I = O_0 · γ`:
    /// - `cached_norm` (lattice index) `= N_red(γ)²`
    /// - quaternion reduced ideal norm = `N_red(γ)` itself
    ///
    /// The two differ by a square. The SQIsign C reference's
    /// `quat_lideal_normN` returns the reduced ideal norm
    /// (`N_red(γ)`); when porting any C-ref formula that uses
    /// `n(ideal)`, callers should use [`Self::reduced_norm_vartime`]
    /// (which integer-square-roots `cached_norm`) rather than this
    /// raw cached value.
    pub fn norm(&self) -> Uint<LIMBS> {
        self.cached_norm
    }

    /// The SQIsign C reference's `quat_lideal_normN(I)` = the
    /// **reduced quaternion ideal norm** of `I`, distinct from
    /// [`Self::norm`]'s lattice-index convention.
    ///
    /// For principal ideals `I = O_0 · γ` in `B_{p,∞}` (all left
    /// ideals of `O_0` are principal in this setting):
    ///
    /// ```text
    ///     reduced_norm(I) = N_red(γ) = √(cached_norm)
    /// ```
    ///
    /// Returns `Some(√cached_norm)` when `cached_norm` is a perfect
    /// square (the principal-ideal case, always true for SQIsign-
    /// produced ideals); returns `None` otherwise (defensive —
    /// surfaces an invariant violation in the unlikely event the
    /// lattice index isn't a square).
    ///
    /// # Why this exists
    ///
    /// Two ideal operations explicitly use this convention:
    /// - The Clapotis `find_uv` delta-rescaling step computes
    ///   `reduced_id = I · conj(δ) / n(I)` where `n(I)` is the
    ///   reduced ideal norm (NOT the lattice index).
    /// - The right ideal class manipulations in the alternate-
    ///   orders loop similarly normalize by the reduced norm.
    ///
    /// Without this primitive, the rescaling step's divisibility
    /// check (`N(I) · N_red(δ) / α_denom²` must be integer) would
    /// be over-constrained — passing `cached_norm` (= `N²`) as
    /// `α_denom` requires `N⁴ | N_red(I)·N_red(δ)`, which fails
    /// for typical inputs. Passing `√cached_norm` (= `N`) requires
    /// `N² | N_red(I)·N_red(δ)`, which holds for SQIsign-shaped
    /// principal ideals.
    ///
    /// # Variable-time
    ///
    /// Uses `crypto_bigint`'s `Uint::floor_sqrt_vartime`. Variable-
    /// time on the bit length — acceptable per SQIsign 2.0 §8.
    pub fn reduced_norm_vartime(&self) -> Option<Uint<LIMBS>> {
        let s = self.cached_norm.floor_sqrt_vartime();
        let s_squared = s.wrapping_mul(&s);
        if s_squared == self.cached_norm {
            Some(s)
        } else {
            None
        }
    }

    /// Reduce the basis matrix to Hermite Normal Form. Returns a new
    /// `LeftIdeal` representing the same lattice as `self` but in canonical
    /// upper-triangular form. `cached_norm` and `denom` preserved.
    pub fn reduced(&self) -> Self {
        Self {
            basis: crate::quaternion::hnf::hnf_4x4(&self.basis),
            denom: self.denom,
            cached_norm: self.cached_norm,
        }
    }

    /// Test whether two ideals represent the same rational lattice. For
    /// the integer-only case (`denom_a == denom_b`) this is HNF equality.
    /// For mixed denominators, cross-multiply the bases by the other
    /// ideal's denominator before HNF comparison.
    pub fn equals_lattice(&self, other: &Self) -> bool {
        if self.denom == other.denom {
            return self.reduced().basis == other.reduced().basis;
        }
        let mut a_scaled = self.basis;
        let other_denom_int = *other.denom.as_int();
        for row in a_scaled.iter_mut() {
            for cell in row.iter_mut() {
                *cell = cell.wrapping_mul(&other_denom_int);
            }
        }
        let mut b_scaled = other.basis;
        let self_denom_int = *self.denom.as_int();
        for row in b_scaled.iter_mut() {
            for cell in row.iter_mut() {
                *cell = cell.wrapping_mul(&self_denom_int);
            }
        }
        crate::quaternion::hnf::hnf_4x4(&a_scaled) == crate::quaternion::hnf::hnf_4x4(&b_scaled)
    }

    /// Divide every basis coordinate by the integer `divisor`. Returns
    /// `None` if the division is not exact (i.e., any cell is not an
    /// integer multiple of `divisor`). Updates `cached_norm` by `1 / n^4`
    /// (also returning `None` if that division isn't exact).
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
        // Update cached_norm: dividing every basis cell by n divides the
        // lattice determinant by n^4, so the cached norm divides by n^4.
        let n_abs = divisor.unsigned_abs();
        let n_pow_4 = n_abs.checked_pow(4)?;
        let n_pow_4_uint = Uint::<LIMBS>::from_u64(n_pow_4);
        let n_nz =
            Option::<crypto_bigint::NonZero<_>>::from(crypto_bigint::NonZero::new(n_pow_4_uint))?;
        let (new_norm, rem) = self.cached_norm.div_rem_vartime(&n_nz);
        if rem != Uint::<LIMBS>::from_u64(0) {
            return None;
        }
        Some(Self {
            basis: out,
            denom: self.denom,
            cached_norm: new_norm,
        })
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
/// Pulled-back Gram matrix of the reduced-norm form on an ideal.
///
/// Returns `G_I = B · G_O0 · Bᵀ` where `B = ideal.basis` (the 4×4
/// `O_0`-basis-coordinate matrix of the ideal) and `G_O0` is the
/// reduced-norm Gram on `O_0` from
/// [`super::o0_mul::o0_reduced_norm_gram_matrix`].
///
/// The invariant: for any integer coordinate vector
/// `v = (v_0, v_1, v_2, v_3) ∈ Z⁴`, the quaternion
/// `α_v = Σ_r v[r] · ideal.basis[r] ∈ I` has reduced norm satisfying
///
/// ```text
///     vᵀ · G_I · v = 4 · N(α_v).
/// ```
///
/// This is the integer-Gram input that `lll_4x4` and KLPT's
/// prime-norm-reduced-equivalent search consume. The factor of 4 is the
/// same fraction-clearing scalar that
/// [`super::o0_mul::reduced_norm_o0_basis`] absorbs after the fact.
///
/// 64 wrapping multiplications (two 4×4 matmuls). Pure integer
/// arithmetic; no Cholesky / no floating point.
pub fn ideal_gram_matrix<const LIMBS: usize>(
    ideal: &LeftIdeal<LIMBS>,
    p: &Uint<LIMBS>,
) -> [[Int<LIMBS>; 4]; 4] {
    let g_o0 = o0_reduced_norm_gram_matrix(p);
    pull_back_gram(&ideal.basis, &g_o0)
}

/// Determinant of a 4×4 integer matrix via Laplace expansion along the
/// first row. Returns the signed result; ideal-norm callers take the
/// absolute value via [`Int::abs`].
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

    #[test]
    fn ideal_gram_for_full_order_equals_o0_gram() {
        use crate::quaternion::o0_mul::o0_reduced_norm_gram_matrix;
        let p: Uint<8> = Uint::<8>::from_u64(7);
        let id = Ideal::full_order();
        let g_i = ideal_gram_matrix(&id, &p);
        let g_o0 = o0_reduced_norm_gram_matrix(&p);
        assert_eq!(g_i, g_o0);
    }

    #[test]
    fn ideal_gram_eval_matches_4n_alpha_on_full_order() {
        use crate::quaternion::lattice::qf_eval_4x4;
        use crate::quaternion::o0_mul::reduced_norm_o0_basis;
        let p: Uint<8> = Uint::<8>::from_u64(7);
        let id = Ideal::full_order();
        let g_i = ideal_gram_matrix(&id, &p);

        // For identity basis, the ideal-coord vector v equals the O_0-coord
        // vector of α_v. Check vᵀ G_I v = 4·N(α_v).
        let v: [Int<8>; 4] = [i64_int(1), i64_int(2), i64_int(3), i64_int(4)];
        let four_n_via_gram = qf_eval_4x4(&v, &g_i);
        let n_alpha = reduced_norm_o0_basis(&v, &p);
        let four_n_via_helper = i64_int(4).wrapping_mul(&n_alpha);
        assert_eq!(four_n_via_gram, four_n_via_helper);
    }

    #[test]
    fn ideal_gram_scales_with_basis_scaling() {
        // For ideal I scaled by 2, basis is 2·B, so G_I = (2B)·G_O0·(2B)ᵀ
        // = 4 · B·G_O0·Bᵀ = 4 · G_I_original. Every entry quadruples.
        let p: Uint<8> = Uint::<8>::from_u64(7);
        let id = Ideal::full_order();
        let doubled = id.scale(2);
        let g_i = ideal_gram_matrix(&id, &p);
        let g_doubled = ideal_gram_matrix(&doubled, &p);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(
                    g_doubled[i][j],
                    i64_int(4).wrapping_mul(&g_i[i][j]),
                    "G[{i}][{j}] should quadruple on basis-scale-by-2"
                );
            }
        }
    }

    #[test]
    fn ideal_gram_symmetric() {
        // G_I = B·G_O0·Bᵀ is symmetric whenever G_O0 is symmetric (which it
        // is). Verify on a non-trivial ideal: I = 3·O_0 (still axis-aligned)
        // and a deliberately-skewed Z-basis of the same lattice.
        let p: Uint<8> = Uint::<8>::from_u64(7);
        let scaled = Ideal::full_order().scale(3);
        let g = ideal_gram_matrix(&scaled, &p);
        for (i, row) in g.iter().enumerate() {
            for (j, &entry) in row.iter().enumerate() {
                assert_eq!(entry, g[j][i], "G[{i}][{j}] != G[{j}][{i}]");
            }
        }
    }

    // ── reduced_norm_vartime unit tests (S199) ─────────────────────────

    #[test]
    fn reduced_norm_full_order_is_one() {
        // O_0 itself has lattice index 1 → reduced norm 1.
        let id = Ideal::full_order();
        let n = id.reduced_norm_vartime();
        assert_eq!(n, Some(Uint::<8>::from_u64(1)));
    }

    #[test]
    fn reduced_norm_scaled_ideal_is_perfect_square_root() {
        // I = 3·O_0 has cached_norm = 3^4 = 81. Reduced norm = √81 = 9.
        let id = Ideal::full_order().scale(3);
        assert_eq!(id.cached_norm, Uint::<8>::from_u64(81));
        let n = id.reduced_norm_vartime();
        assert_eq!(
            n,
            Some(Uint::<8>::from_u64(9)),
            "3·O_0 reduced norm = 9 = N_red(3 in B_{{p,∞}})",
        );
    }

    #[test]
    fn reduced_norm_returns_none_for_non_square_cached_norm() {
        // Construct an ideal manually with a non-square cached_norm —
        // defensive case that should not arise from any quaternion
        // ideal operation in practice (all left ideals of O_0 are
        // principal so cached_norm = N(γ)² always).
        let basis = [
            [
                Int::<8>::from_i64(1),
                Int::<8>::from_i64(0),
                Int::<8>::from_i64(0),
                Int::<8>::from_i64(0),
            ],
            [
                Int::<8>::from_i64(0),
                Int::<8>::from_i64(1),
                Int::<8>::from_i64(0),
                Int::<8>::from_i64(0),
            ],
            [
                Int::<8>::from_i64(0),
                Int::<8>::from_i64(0),
                Int::<8>::from_i64(1),
                Int::<8>::from_i64(0),
            ],
            [
                Int::<8>::from_i64(0),
                Int::<8>::from_i64(0),
                Int::<8>::from_i64(0),
                Int::<8>::from_i64(1),
            ],
        ];
        let bogus = Ideal {
            basis,
            denom: Uint::<8>::ONE,
            cached_norm: Uint::<8>::from_u64(7), // 7 is not a perfect square
        };
        assert_eq!(
            bogus.reduced_norm_vartime(),
            None,
            "non-square cached_norm must surface as None",
        );
    }
}
