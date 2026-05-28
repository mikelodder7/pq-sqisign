// SPDX-License-Identifier: MIT OR Apache-2.0

//! Splitting isogeny — the dual of the gluing isogeny.
//!
//! Where [`crate::isogeny::gluing`] takes an elliptic product `E_1 × E_2`
//! to a `(2, 2)`-theta-coord Abelian variety, this module's
//! [`splitting_compute`] takes a `(2, 2)`-theta-coord variety back to
//! an elliptic product. The two boundary steps + the chain interior
//! (`theta_isogeny_compute*` in [`crate::isogeny::theta_isogeny`])
//! together form the full `(2, 2)`-isogeny chain.
//!
//! # Surface
//!
//! - [`ThetaSplitting`] — the splitting state (basis-change matrix +
//!   codomain theta-null).
//! - [`SplittingError`] — failure modes (currently `NotImplemented`
//!   until S146+ tables port).
//! - [`base_change_matrix_multiplication`] — 4×4 matrix product over
//!   `Fp2<F>`. Real testable infrastructure (S145 advisor's
//!   α-with-real-infrastructure scope).
//! - [`select_base_change_matrix`] — constant-time conditional select
//!   between two basis-change matrices. CT-relevant in the splitting
//!   base path because `U_cst == 0` is secret-derived for chains
//!   produced on the signing side.
//! - [`splitting_compute`] — main entry. Currently
//!   `Err(SplittingError::NotImplemented)` pending the four constant
//!   tables (`EVEN_INDEX`, `CHI_EVAL`, `SPLITTING_TRANSFORMS`,
//!   `NORMALIZATION_TRANSFORMS`) to be ported from the C reference
//!   in a follow-up session.
//!
//! # S145 advisor scope decision
//!
//! The full `splitting_compute` body requires four constant tables
//! that aren't accessible in /tmp/. Per S145 advisor, the β path
//! (mathematically derive the tables now) is a trap: derived
//! orderings are unvalidatable against the C reference's arbitrary
//! convention, and S146's `SPLITTING_TRANSFORMS[i]` indexes by the
//! SAME `i` — a mismatch silently misaligns and surfaces only at
//! chain-integration KAT stage. Instead, ship the α-real path: real
//! testable helpers (`base_change_matrix_multiplication`,
//! `select_base_change_matrix`) as new infrastructure, with
//! `splitting_compute` returning `Err(NotImplemented)` until S146
//! ports tables from the C reference's accessible source.

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use crate::ec::couple::{CoupleCurve, CoupleMontgomeryPoint};
use crate::ec::montgomery::{MontgomeryCurve, MontgomeryPoint};
use crate::gf::fp::BaseField;
use crate::gf::fp2::Fp2;
use crate::isogeny::gluing::BasisChangeMatrix;
use crate::isogeny::theta::{AbelianVariety2D, ThetaPoint2D};

/// Splitting-isogeny per-step state.
///
/// Mirrors the C reference's `theta_splitting_t`:
/// - `m`: the basis-change matrix used to map the input variety's
///   null point to the splitting-compatible representative.
/// - `b_null`: the codomain theta-null after applying `m`. Just the
///   null point (NOT a full [`AbelianVariety2D`]), matching S142's
///   "no fake AbelianVariety2D" doctrine — downstream callers
///   materialize a full variety when they need doubling.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ThetaSplitting<F: BaseField> {
    /// Basis-change matrix. After splitting, `b_null = m · A.null_point`.
    pub(crate) m: BasisChangeMatrix<F>,
    /// Codomain theta-null point. Compatible with elliptic-product
    /// extraction by downstream `theta_product_structure_to_elliptic_product`
    /// (S147+).
    pub(crate) b_null: ThetaPoint2D<F>,
}

impl<F: BaseField> ThetaSplitting<F> {
    /// Compute the splitting isogeny from a `(2, 2)`-theta-coord
    /// variety back to an elliptic product.
    ///
    /// Method-form alias of [`splitting_compute`]. Constructor for
    /// the splitting state; currently returns
    /// `Err(SplittingError::NotImplemented)` pending S148+ tables port.
    #[allow(dead_code)]
    pub(crate) fn compute(
        domain: &AbelianVariety2D<F>,
        zero_index: Option<usize>,
        randomize: bool,
    ) -> Result<Self, SplittingError> {
        splitting_compute(domain, zero_index, randomize)
    }
}

/// Failure modes for [`splitting_compute`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SplittingError {
    /// The full body of `splitting_compute` is not yet implemented —
    /// the four constant tables (`EVEN_INDEX`, `CHI_EVAL`,
    /// `SPLITTING_TRANSFORMS`, `NORMALIZATION_TRANSFORMS`) must be
    /// ported from the C reference first (S147+). Until then,
    /// callers receive this variant.
    NotImplemented,
}

/// Failure modes for the post-splitting extraction routines
/// [`theta_product_structure_to_elliptic_product`] and
/// [`theta_point_to_montgomery_point`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ExtractionError {
    /// The input theta point does not lie on a product theta
    /// structure — i.e., the algebraic identity `x · t = y · z`
    /// fails. (Where `t` is Rust `w`.) The C reference's
    /// `is_product_theta_point` returns 0 in this case; we propagate
    /// as a typed error so callers see the failure mode.
    NotProductTheta,
    /// A required null-point coordinate (`x`, `y`, or `z`) was zero,
    /// making the elliptic-product extraction's Riemann-theta
    /// moduli formula undefined.
    ZeroNullCoordinate,
    /// One of the computed Montgomery-curve denominators
    /// (`x^4 - z^4` for `E_1` or `x^4 - y^4` for `E_2`) was zero,
    /// so the affine `A` coefficient cannot be obtained — the
    /// elliptic curve is degenerate.
    ZeroChartCoefficient,
    /// Input point P is `(0:0:0:0)` after the alt-coord fallback in
    /// [`theta_point_to_montgomery_point`] — no extraction path
    /// produces a meaningful projective output.
    AllZeroPoint,
}

/// Check whether a theta point lies on a product theta structure.
///
/// Returns `Choice::TRUE` iff `P.x · P.w == P.y · P.z` (in C ref
/// notation: `P.x · P.t == P.y · P.z`). The product structure
/// constraint is the well-known relation that identifies decomposable
/// Abelian surfaces in the (2,2)-theta model.
///
/// Reference: `theta_structure.c:71-78` (`is_product_theta_point`).
#[allow(dead_code)]
pub(crate) fn is_product_theta_point<F: BaseField>(p: &ThetaPoint2D<F>) -> Choice {
    p.x.mul(&p.w).ct_eq(&p.y.mul(&p.z))
}

/// Extract the elliptic product `E_1 × E_2` from a 2-dimensional
/// theta-coord Abelian variety that has been split back to a
/// product structure (typically the output of [`splitting_compute`]
/// applied at the end of a `(2, 2)`-isogeny chain).
///
/// # Algorithm
///
/// Mirrors `theta_isogenies.c:970-1018` verbatim:
///
/// 1. Check `is_product_theta_point(domain.theta_null)`. If false,
///    return `Err(NotProductTheta)`.
/// 2. Check `domain.theta_null.{x, y, z}` are all non-zero. Zero in
///    any of these makes the moduli formula undefined →
///    `Err(ZeroNullCoordinate)`.
/// 3. Compute the affine Montgomery coefficients via the Riemann
///    theta-to-Montgomery moduli formula:
///    ```text
///    A_2 = -2 (x^4 + y^4)   ;   C_2 = x^4 - y^4
///    A_1 = -2 (x^4 + z^4)   ;   C_1 = x^4 - z^4
///    affine a_2 = A_2 / C_2
///    affine a_1 = A_1 / C_1
///    ```
///    Where `(x, y, z, w)` = `domain.theta_null`. If `C_1` or `C_2`
///    is zero, the inversion fails →
///    `Err(ZeroChartCoefficient)`.
///
/// Returns a [`CoupleCurve<F>`] of the two elliptic curves.
///
/// Reference: `theta_isogenies.c:theta_product_structure_to_elliptic_product`.
#[allow(dead_code)]
pub(crate) fn theta_product_structure_to_elliptic_product<F: BaseField>(
    domain: &AbelianVariety2D<F>,
) -> Result<CoupleCurve<F>, ExtractionError> {
    // Step 1: identity check.
    if !bool::from(is_product_theta_point(&domain.theta_null)) {
        return Err(ExtractionError::NotProductTheta);
    }

    let nx = &domain.theta_null.x;
    let ny = &domain.theta_null.y;
    let nz = &domain.theta_null.z;

    // Step 2: non-zero null coordinate check on x, y, z.
    let any_zero = nx.is_zero() | ny.is_zero() | nz.is_zero();
    if bool::from(any_zero) {
        return Err(ExtractionError::ZeroNullCoordinate);
    }

    // Step 3: compute the moduli formula via fourth powers.
    let x_4 = nx.square().square();
    let y_4 = ny.square().square();
    let z_4 = nz.square().square();

    // E_2: A_2 = -2 (x^4 + y^4), C_2 = x^4 - y^4
    let a_2 = x_4.add(&y_4).double().negate();
    let c_2 = x_4.sub(&y_4);

    // E_1: A_1 = -2 (x^4 + z^4), C_1 = x^4 - z^4
    let a_1 = x_4.add(&z_4).double().negate();
    let c_1 = x_4.sub(&z_4);

    // Inversions: any zero C is fatal.
    let c_2_inv_opt = c_2.invert();
    let c_1_inv_opt = c_1.invert();
    if !bool::from(c_2_inv_opt.is_some() & c_1_inv_opt.is_some()) {
        return Err(ExtractionError::ZeroChartCoefficient);
    }
    let c_2_inv = c_2_inv_opt
        .into_option()
        .ok_or(ExtractionError::ZeroChartCoefficient)?;
    let c_1_inv = c_1_inv_opt
        .into_option()
        .ok_or(ExtractionError::ZeroChartCoefficient)?;

    let affine_a_2 = a_2.mul(&c_2_inv);
    let affine_a_1 = a_1.mul(&c_1_inv);

    Ok(CoupleCurve {
        e1: MontgomeryCurve::new(affine_a_1),
        e2: MontgomeryCurve::new(affine_a_2),
    })
}

/// Extract a couple-Montgomery point (X:Z) on `E_1 × E_2` from a
/// theta-coord point P on a product theta structure.
///
/// # Algorithm
///
/// Mirrors `theta_isogenies.c:1019-1057` verbatim:
///
/// 1. Check `is_product_theta_point(P)`. If false,
///    return `Err(NotProductTheta)`.
/// 2. For `P_2`: pick `(x_src, z_src) = (P.x, P.y)` unless both are
///    zero, in which case try `(P.z, P.w)`. If both are still zero,
///    return `Err(AllZeroPoint)`. Then
///    `P_2.X = nullA.y · x_src + nullA.x · z_src`,
///    `P_2.Z = -nullA.y · x_src + nullA.x · z_src`.
/// 3. For `P_1`: pick `(x_src, z_src) = (P.x, P.z)` unless both are
///    zero, in which case try `(P.y, P.w)`. Then
///    `P_1.X = nullA.z · x_src + nullA.x · z_src`,
///    `P_1.Z = -nullA.z · x_src + nullA.x · z_src`.
///
/// # Branching note
///
/// The fallback "if both zero, try alt coords" pattern is a runtime
/// branch on the value of `P` — NOT constant-time. This matches the
/// C reference, which similarly branches. Per chain-output context
/// where this routine runs after splitting, branching is acceptable;
/// the secret-sensitive paths are upstream (kernel generation,
/// response computation) and have their own CT discipline.
///
/// Reference: `theta_isogenies.c:theta_point_to_montgomery_point`.
#[allow(dead_code)]
pub(crate) fn theta_point_to_montgomery_point<F: BaseField>(
    p: &ThetaPoint2D<F>,
    domain: &AbelianVariety2D<F>,
) -> Result<CoupleMontgomeryPoint<F>, ExtractionError> {
    if !bool::from(is_product_theta_point(p)) {
        return Err(ExtractionError::NotProductTheta);
    }

    let null_x = &domain.theta_null.x;
    let null_y = &domain.theta_null.y;
    let null_z = &domain.theta_null.z;

    // Step 2: build P_2 from (P.x, P.y) or fallback (P.z, P.w).
    let (p2_xsrc, p2_zsrc) = if bool::from(p.x.is_zero()) && bool::from(p.y.is_zero()) {
        if bool::from(p.z.is_zero()) && bool::from(p.w.is_zero()) {
            return Err(ExtractionError::AllZeroPoint);
        }
        (&p.z, &p.w)
    } else {
        (&p.x, &p.y)
    };
    let p2_term_x = null_y.mul(p2_xsrc);
    let p2_term_z = null_x.mul(p2_zsrc);
    let p2_x = p2_term_x.add(&p2_term_z);
    let p2_z = p2_term_z.sub(&p2_term_x);

    // Step 3: build P_1 from (P.x, P.z) or fallback (P.y, P.w).
    let (p1_xsrc, p1_zsrc) = if bool::from(p.x.is_zero()) && bool::from(p.z.is_zero()) {
        (&p.y, &p.w)
    } else {
        (&p.x, &p.z)
    };
    let p1_term_x = null_z.mul(p1_xsrc);
    let p1_term_z = null_x.mul(p1_zsrc);
    let p1_x = p1_term_x.add(&p1_term_z);
    let p1_z = p1_term_z.sub(&p1_term_x);

    Ok(CoupleMontgomeryPoint {
        p1: MontgomeryPoint::new(p1_x, p1_z),
        p2: MontgomeryPoint::new(p2_x, p2_z),
    })
}

/// Compute the product of two `4 × 4` basis-change matrices over
/// `Fp2<F>`.
///
/// `result[i][j] = Σ_k a[i][k] · b[k][j]` (standard row-by-column
/// matrix product). 64 field multiplications + 48 additions.
///
/// Used by [`splitting_compute`] (S146+) to compose the chosen
/// `SPLITTING_TRANSFORMS[i]` with the (signing-path) randomization
/// matrix from `NORMALIZATION_TRANSFORMS`. Mirrors the C reference's
/// `base_change_matrix_multiplication`.
#[allow(dead_code)]
pub(crate) fn base_change_matrix_multiplication<F: BaseField>(
    a: &BasisChangeMatrix<F>,
    b: &BasisChangeMatrix<F>,
) -> BasisChangeMatrix<F> {
    let mut m = [[Fp2::<F>::zero(); 4]; 4];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let mut acc = Fp2::<F>::zero();
            for k in 0..4 {
                acc = acc.add(&a.m[i][k].mul(&b.m[k][j]));
            }
            *cell = acc;
        }
        let _ = i;
    }
    BasisChangeMatrix { m }
}

/// Constant-time conditional select between two basis-change matrices.
///
/// Returns `a` if `choice` is `Choice::FALSE` (0); returns `b` if
/// `Choice::TRUE` (1). The selection is entrywise constant-time via
/// [`Fp2::conditional_select`]. CT-relevant in the splitting base
/// path: when `U_cst == 0` for index `i`, the matching
/// `SPLITTING_TRANSFORMS[i]` must be selected into the running `M`
/// — and `U_cst`'s zero-position is secret-derived for chains
/// produced on the signing side (the response-side chain is keyed on
/// the secret).
///
/// Mirrors the C reference's `select_base_change_matrix`.
#[allow(dead_code)]
pub(crate) fn select_base_change_matrix<F: BaseField>(
    a: &BasisChangeMatrix<F>,
    b: &BasisChangeMatrix<F>,
    choice: Choice,
) -> BasisChangeMatrix<F> {
    let mut m = [[Fp2::<F>::zero(); 4]; 4];
    for (out_row, (a_row, b_row)) in m.iter_mut().zip(a.m.iter().zip(b.m.iter())) {
        for (cell, (a_cell, b_cell)) in out_row.iter_mut().zip(a_row.iter().zip(b_row.iter())) {
            *cell = Fp2::<F>::conditional_select(a_cell, b_cell, choice);
        }
    }
    BasisChangeMatrix { m }
}

/// Compute the splitting isogeny from a `(2, 2)`-theta-coord variety
/// back to an elliptic product.
///
/// # Current scope (S145)
///
/// Currently returns `Err(SplittingError::NotImplemented)`. The full
/// body requires four constant tables (`EVEN_INDEX[10][2]`,
/// `CHI_EVAL[6][4]`, `SPLITTING_TRANSFORMS[10]`,
/// `NORMALIZATION_TRANSFORMS[6]`) to be ported from the C
/// reference's compiled-constants header, which isn't accessible
/// from /tmp/ in this session. Per S145 advisor: deriving the
/// tables mathematically is a trap because the ORDERING must match
/// the C reference's arbitrary convention to align with the
/// SPLITTING_TRANSFORMS index lookup.
///
/// # Future scope (S146+)
///
/// 1. Port `EVEN_INDEX[10][2]` and `CHI_EVAL[6][4]` from C ref.
/// 2. Port `SPLITTING_TRANSFORMS[10]` (each a `BasisChangeMatrix<F>`).
/// 3. Port `NORMALIZATION_TRANSFORMS[6]` (signing-path randomization,
///    feature-gated under `sign`).
/// 4. Wire body: enumerate `i in 0..10`; compute `U_cst`; CT-select
///    `SPLITTING_TRANSFORMS[i]` into `M` when `U_cst == 0`; final
///    `apply_isomorphism(b_null, M, domain.null)`; return
///    `Ok(ThetaSplitting { m, b_null })`.
///
/// Reference: `theta_isogenies.c:splitting_compute`.
#[allow(dead_code)]
pub(crate) fn splitting_compute<F: BaseField>(
    _domain: &AbelianVariety2D<F>,
    _zero_index: Option<usize>,
    _randomize: bool,
) -> Result<ThetaSplitting<F>, SplittingError> {
    Err(SplittingError::NotImplemented)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf::fp::Fp1Element;

    /// Build a `BasisChangeMatrix<Fp1Element>` from a `[[u32; 4]; 4]`
    /// of small ints. Test helper for hand-constructed matrices.
    fn from_u32_grid(grid: [[u32; 4]; 4]) -> BasisChangeMatrix<Fp1Element> {
        let mut m = [[Fp2::<Fp1Element>::zero(); 4]; 4];
        for (i, row) in grid.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                let mut acc = Fp2::<Fp1Element>::zero();
                let one = Fp2::<Fp1Element>::one();
                for _ in 0..v {
                    acc = acc.add(&one);
                }
                m[i][j] = acc;
            }
        }
        BasisChangeMatrix { m }
    }

    /// Identity 4×4 matrix.
    fn identity_4x4() -> BasisChangeMatrix<Fp1Element> {
        from_u32_grid([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    }

    #[test]
    fn base_change_matrix_multiplication_identity_left_at_lvl1() {
        let i = identity_4x4();
        let a = from_u32_grid([
            [2, 3, 5, 7],
            [11, 13, 17, 19],
            [23, 29, 31, 37],
            [41, 43, 47, 53],
        ]);
        let result = base_change_matrix_multiplication(&i, &a);
        assert_eq!(result, a, "S145: I · A = A");
    }

    #[test]
    fn base_change_matrix_multiplication_identity_right_at_lvl1() {
        let i = identity_4x4();
        let a = from_u32_grid([
            [2, 3, 5, 7],
            [11, 13, 17, 19],
            [23, 29, 31, 37],
            [41, 43, 47, 53],
        ]);
        let result = base_change_matrix_multiplication(&a, &i);
        assert_eq!(result, a, "S145: A · I = A");
    }

    /// Independent oracle: hand-computed `A · B` for two small-int
    /// matrices. Expected `c[i][j] = Σ_k a[i][k] · b[k][j]`.
    ///
    /// A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    /// B = identity-with-shift = [[0,1,0,0], [0,0,1,0], [0,0,0,1], [1,0,0,0]]
    /// A · B shifts each row's columns: A[i][k] · B[k][j] = A[i][k] when (k,j) matches B's 1 entry.
    /// B[k][j] = 1 iff j = (k + 1) mod 4 — so A·B[i][j] = A[i][(j - 1) mod 4]
    /// For row 0 of A · B: (A[0][3], A[0][0], A[0][1], A[0][2]) = (4, 1, 2, 3)
    #[test]
    fn base_change_matrix_multiplication_shift_oracle_at_lvl1() {
        let a = from_u32_grid([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]);
        let b = from_u32_grid([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]);
        let expected = from_u32_grid([
            [4, 1, 2, 3],
            [8, 5, 6, 7],
            [12, 9, 10, 11],
            [16, 13, 14, 15],
        ]);
        let result = base_change_matrix_multiplication(&a, &b);
        assert_eq!(
            result, expected,
            "S145: A · B with B as column-shift matrix must shift A's columns",
        );
    }

    #[test]
    fn select_base_change_matrix_choice_false_returns_first_at_lvl1() {
        let a = from_u32_grid([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 5, 0], [0, 0, 0, 7]]);
        let b = identity_4x4();
        let result = select_base_change_matrix(&a, &b, Choice::from(0));
        assert_eq!(result, a, "S145: Choice::FALSE selects first argument");
    }

    #[test]
    fn select_base_change_matrix_choice_true_returns_second_at_lvl1() {
        let a = from_u32_grid([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 5, 0], [0, 0, 0, 7]]);
        let b = identity_4x4();
        let result = select_base_change_matrix(&a, &b, Choice::from(1));
        assert_eq!(result, b, "S145: Choice::TRUE selects second argument");
    }

    /// CT property check: selecting between A and B with Choice::TRUE
    /// then selecting back to A with Choice::FALSE must round-trip
    /// to A. Confirms the operation is self-consistent.
    #[test]
    fn select_base_change_matrix_round_trip_at_lvl1() {
        let a = from_u32_grid([
            [2, 3, 5, 7],
            [11, 13, 17, 19],
            [23, 29, 31, 37],
            [41, 43, 47, 53],
        ]);
        let b = identity_4x4();
        let intermediate = select_base_change_matrix(&a, &b, Choice::from(1));
        let recovered = select_base_change_matrix(&intermediate, &a, Choice::from(1));
        assert_eq!(
            recovered, a,
            "S145: TRUE-select B then TRUE-select A round-trips to A",
        );
    }

    #[test]
    fn splitting_compute_returns_not_implemented_at_lvl1() {
        let null = ThetaPoint2D::new(
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
        );
        let domain = AbelianVariety2D::new(null, null);
        let result = splitting_compute(&domain, None, false);
        assert_eq!(
            result,
            Err(SplittingError::NotImplemented),
            "S145: splitting_compute body pending S147 tables port",
        );
    }

    // S146 — post-splitting extraction tests.
    // is_product_theta_point + theta_product_structure_to_elliptic_product
    // + theta_point_to_montgomery_point.

    fn small_fp2(n: u32) -> Fp2<Fp1Element> {
        let mut acc = Fp2::<Fp1Element>::zero();
        let one = Fp2::<Fp1Element>::one();
        for _ in 0..n {
            acc = acc.add(&one);
        }
        acc
    }

    /// Identity check: `(x, y, z, w) = (2, 3, 6, 9)` satisfies
    /// `x·w = 18 = y·z`. is_product_theta_point must return TRUE.
    #[test]
    fn is_product_theta_point_accepts_product_point_at_lvl1() {
        let p = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(6), small_fp2(9));
        assert!(
            bool::from(is_product_theta_point(&p)),
            "S146: (2, 3, 6, 9) satisfies x·w=18=y·z; must be product",
        );
    }

    /// Reject: `(2, 3, 5, 7)` does NOT satisfy `x·w = y·z`
    /// (2·7=14 ≠ 3·5=15). is_product_theta_point must return FALSE.
    #[test]
    fn is_product_theta_point_rejects_non_product_at_lvl1() {
        let p = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        assert!(
            !bool::from(is_product_theta_point(&p)),
            "S146: (2, 3, 5, 7) violates product identity (14 ≠ 15); must be non-product",
        );
    }

    #[test]
    fn theta_product_structure_extract_rejects_non_product_at_lvl1() {
        let null = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        let domain = AbelianVariety2D::new(null, null);
        let result = theta_product_structure_to_elliptic_product(&domain);
        assert_eq!(
            result,
            Err(ExtractionError::NotProductTheta),
            "S146: non-product null must be rejected as NotProductTheta",
        );
    }

    #[test]
    fn theta_product_structure_extract_rejects_zero_null_at_lvl1() {
        // Zero null trivially satisfies x·w=0=y·z but has zero
        // coordinates, so the moduli formula is undefined.
        let zero = Fp2::<Fp1Element>::zero();
        let null = ThetaPoint2D::new(zero, zero, zero, zero);
        let domain = AbelianVariety2D::new(null, null);
        let result = theta_product_structure_to_elliptic_product(&domain);
        assert_eq!(
            result,
            Err(ExtractionError::ZeroNullCoordinate),
            "S146: all-zero null must be rejected as ZeroNullCoordinate",
        );
    }

    /// Success path: choose a product-structure null with all
    /// non-zero coordinates AND `x^4 ≠ y^4`, `x^4 ≠ z^4`.
    /// `(x, y, z, w) = (1, 2, 3, 6)` satisfies x·w = 6 = y·z; all
    /// non-zero; x^4=1, y^4=16, z^4=81, so 1-16=-15 ≠ 0 and 1-81=-80 ≠ 0.
    #[test]
    fn theta_product_structure_extract_success_at_lvl1() {
        let null = ThetaPoint2D::new(small_fp2(1), small_fp2(2), small_fp2(3), small_fp2(6));
        let domain = AbelianVariety2D::new(null, null);
        let result = theta_product_structure_to_elliptic_product(&domain);
        let cc = result.expect("S146: valid product null must extract Ok");
        // Sanity: both curves have a non-zero (well, possibly any) a.
        // The exact a value isn't easily hand-computed but we can verify
        // it's been computed (i.e., not infinity/garbage). Confirm the
        // routine populated both halves of the couple.
        let _e1 = cc.e1;
        let _e2 = cc.e2;
    }

    /// Independent oracle on the extracted `a` coefficients.
    /// For null `(1, 2, 3, 6)`:
    ///   x^4 = 1, y^4 = 16, z^4 = 81
    ///   E_2: A_2 = -2(1+16) = -34;  C_2 = 1-16 = -15;  a_2 = -34 / -15 = 34/15
    ///   E_1: A_1 = -2(1+81) = -164; C_1 = 1-81 = -80; a_1 = -164 / -80 = 41/20
    /// At L1's prime, 34/15 and 41/20 are well-defined Fp2 elements.
    /// Verify by recomputing `a · C ?= A` (the dual identity bypassing
    /// inversion).
    #[test]
    fn theta_product_structure_extract_recompute_inverts_at_lvl1() {
        let null = ThetaPoint2D::new(small_fp2(1), small_fp2(2), small_fp2(3), small_fp2(6));
        let domain = AbelianVariety2D::new(null, null);
        let cc = theta_product_structure_to_elliptic_product(&domain)
            .expect("S146: extraction must succeed");

        // For E_2: a_2 · C_2 = A_2 where C_2 = x^4 - y^4, A_2 = -2(x^4 + y^4).
        let x_4 = small_fp2(1);
        let y_4 = small_fp2(16);
        let z_4 = small_fp2(81);
        let c_2 = x_4.sub(&y_4);
        let a_2_expected = x_4.add(&y_4).double().negate();
        assert_eq!(
            cc.e2.a.mul(&c_2),
            a_2_expected,
            "S146: a_2 · C_2 must equal A_2 (round-trip via projective form)",
        );

        let c_1 = x_4.sub(&z_4);
        let a_1_expected = x_4.add(&z_4).double().negate();
        assert_eq!(
            cc.e1.a.mul(&c_1),
            a_1_expected,
            "S146: a_1 · C_1 must equal A_1 (round-trip via projective form)",
        );
    }

    #[test]
    fn theta_point_to_montgomery_rejects_non_product_at_lvl1() {
        let p = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        let null = ThetaPoint2D::new(small_fp2(1), small_fp2(2), small_fp2(3), small_fp2(6));
        let domain = AbelianVariety2D::new(null, null);
        let result = theta_point_to_montgomery_point(&p, &domain);
        assert_eq!(
            result,
            Err(ExtractionError::NotProductTheta),
            "S146: non-product P must be rejected",
        );
    }

    #[test]
    fn theta_point_to_montgomery_rejects_all_zero_with_fallback_at_lvl1() {
        // P = (0, 0, 0, 0): all-zero → fallback alt-coords also zero → AllZeroPoint.
        let zero = Fp2::<Fp1Element>::zero();
        let p = ThetaPoint2D::new(zero, zero, zero, zero);
        let null = ThetaPoint2D::new(small_fp2(1), small_fp2(2), small_fp2(3), small_fp2(6));
        let domain = AbelianVariety2D::new(null, null);
        let result = theta_point_to_montgomery_point(&p, &domain);
        assert_eq!(
            result,
            Err(ExtractionError::AllZeroPoint),
            "S146: all-zero P with all-zero fallback must be AllZeroPoint",
        );
    }

    // S168 — BasisChangeMatrix::identity + is_identity.

    #[test]
    fn basis_change_matrix_identity_constructor_matches_identity_grid_at_lvl1() {
        let via_method = BasisChangeMatrix::<Fp1Element>::identity();
        let via_grid = identity_4x4();
        assert_eq!(
            via_method, via_grid,
            "S168: identity() must equal hand-built identity grid",
        );
    }

    #[test]
    fn basis_change_matrix_identity_is_neutral_under_mul_at_lvl1() {
        let i = BasisChangeMatrix::<Fp1Element>::identity();
        let a = from_u32_grid([
            [2, 3, 5, 7],
            [11, 13, 17, 19],
            [23, 29, 31, 37],
            [41, 43, 47, 53],
        ]);
        assert_eq!(i.mul(&a), a, "S168: I · A = A");
        assert_eq!(a.mul(&i), a, "S168: A · I = A");
    }

    #[test]
    fn basis_change_matrix_is_identity_true_for_identity_at_lvl1() {
        let i = BasisChangeMatrix::<Fp1Element>::identity();
        assert!(
            bool::from(i.is_identity()),
            "S168: identity().is_identity() must be TRUE",
        );
    }

    #[test]
    fn basis_change_matrix_is_identity_false_for_nonidentity_at_lvl1() {
        // Diagonal-only-with-non-1 value
        let m = from_u32_grid([[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
        assert!(
            !bool::from(m.is_identity()),
            "S168: diag-(2,1,1,1) is NOT identity",
        );
        // Identity-shape with one off-diagonal non-zero
        let m2 = from_u32_grid([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
        assert!(
            !bool::from(m2.is_identity()),
            "S168: identity with one off-diagonal 1 is NOT identity",
        );
    }

    // S154 — BasisChangeMatrix method-form alias tests.

    #[test]
    fn basis_change_matrix_mul_method_matches_free_function_at_lvl1() {
        let a = from_u32_grid([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]);
        let b = from_u32_grid([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]);
        let via_method = a.mul(&b);
        let via_free = base_change_matrix_multiplication(&a, &b);
        assert_eq!(
            via_method, via_free,
            "S154: a.mul(&b) must match base_change_matrix_multiplication(&a, &b)",
        );
    }

    // S155 — extraction method-form alias tests.

    // S159 — ThetaSplitting::compute method-form alias test.

    #[test]
    fn theta_splitting_compute_method_matches_free_function_at_lvl1() {
        let null = ThetaPoint2D::new(
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
        );
        let domain = AbelianVariety2D::new(null, null);

        let via_method = ThetaSplitting::compute(&domain, None, false);
        let via_free = splitting_compute(&domain, None, false);
        assert_eq!(
            via_method, via_free,
            "S159: ThetaSplitting::compute must match splitting_compute free function (both return NotImplemented currently)",
        );
    }

    #[test]
    fn abelian_variety_to_elliptic_product_method_matches_free_function_at_lvl1() {
        let null = ThetaPoint2D::new(small_fp2(1), small_fp2(2), small_fp2(3), small_fp2(6));
        let domain = AbelianVariety2D::new(null, null);

        let via_method = domain.to_elliptic_product();
        let via_free = theta_product_structure_to_elliptic_product(&domain);
        assert_eq!(
            via_method, via_free,
            "S155: domain.to_elliptic_product() must match free function",
        );
    }

    #[test]
    fn theta_point_to_montgomery_method_matches_free_function_at_lvl1() {
        let p = ThetaPoint2D::new(small_fp2(1), small_fp2(2), small_fp2(3), small_fp2(6));
        let null = ThetaPoint2D::new(small_fp2(1), small_fp2(2), small_fp2(3), small_fp2(6));
        let domain = AbelianVariety2D::new(null, null);

        let via_method = p.to_montgomery_point_on(&domain);
        let via_free = theta_point_to_montgomery_point(&p, &domain);
        assert_eq!(
            via_method, via_free,
            "S155: p.to_montgomery_point_on(&domain) must match free function",
        );
    }

    #[test]
    fn extraction_methods_propagate_errors_at_lvl1() {
        // Non-product theta_null → both methods should return
        // NotProductTheta variant matching the free functions.
        let bad_null = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        let domain = AbelianVariety2D::new(bad_null, bad_null);

        assert_eq!(
            domain.to_elliptic_product(),
            Err(ExtractionError::NotProductTheta),
            "S155: method propagates NotProductTheta",
        );

        let bad_p = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        // Use a product null for the domain so only the point-side
        // check fails:
        let good_null = ThetaPoint2D::new(small_fp2(1), small_fp2(2), small_fp2(3), small_fp2(6));
        let good_domain = AbelianVariety2D::new(good_null, good_null);
        assert_eq!(
            bad_p.to_montgomery_point_on(&good_domain),
            Err(ExtractionError::NotProductTheta),
            "S155: method propagates NotProductTheta on the point side",
        );
    }

    #[test]
    fn basis_change_matrix_select_method_matches_free_function_at_lvl1() {
        let a = from_u32_grid([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 5, 0], [0, 0, 0, 7]]);
        let b = identity_4x4();
        for &(choice_val, label) in &[(0u8, "FALSE"), (1u8, "TRUE")] {
            let choice = Choice::from(choice_val);
            let via_method = a.select(&b, choice);
            let via_free = select_base_change_matrix(&a, &b, choice);
            assert_eq!(
                via_method, via_free,
                "S154: a.select(&b, Choice::{label}) must match select_base_change_matrix",
            );
        }
    }

    #[test]
    fn theta_point_to_montgomery_success_at_lvl1() {
        // P = (1, 2, 3, 6): satisfies product identity (1·6 = 6 = 2·3).
        // Same as the success null fixture.
        let p = ThetaPoint2D::new(small_fp2(1), small_fp2(2), small_fp2(3), small_fp2(6));
        let null = ThetaPoint2D::new(small_fp2(1), small_fp2(2), small_fp2(3), small_fp2(6));
        let domain = AbelianVariety2D::new(null, null);
        let result = theta_point_to_montgomery_point(&p, &domain);
        let cmp = result.expect("S146: valid product P + valid null must extract Ok");
        // Per formula:
        //   P_2.X = null.y · P.x + null.x · P.y = 2·1 + 1·2 = 4
        //   P_2.Z = null.x · P.y - null.y · P.x = 1·2 - 2·1 = 0
        //   P_1.X = null.z · P.x + null.x · P.z = 3·1 + 1·3 = 6
        //   P_1.Z = null.x · P.z - null.z · P.x = 1·3 - 3·1 = 0
        assert_eq!(cmp.p2.x, small_fp2(4), "S146: P_2.X = 4");
        assert_eq!(cmp.p2.z, Fp2::<Fp1Element>::zero(), "S146: P_2.Z = 0");
        assert_eq!(cmp.p1.x, small_fp2(6), "S146: P_1.X = 6");
        assert_eq!(cmp.p1.z, Fp2::<Fp1Element>::zero(), "S146: P_1.Z = 0");
    }
}
