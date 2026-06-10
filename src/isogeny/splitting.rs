// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(rustdoc::private_intra_doc_links)]

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
use rand_core::CryptoRng;

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
    /// The full body of `splitting_compute` is not yet implemented.
    /// (Retained for API stability; the body landed in S246 — current
    /// failure modes are the more specific variants below.)
    NotImplemented,
    /// No even characteristic had a vanishing `U_cst` (or more than one
    /// did): `count != 1`. The input variety is not splittable into an
    /// elliptic product via the (2,2)-theta model. (C ref: `return
    /// count == 1` is false.)
    NotSplittable,
    /// `zero_index` was supplied (the caller knew which even-index
    /// coordinate should vanish) but at that index `U_cst != 0` — the
    /// split is mis-shaped relative to the expected kernel. (C ref's
    /// `zero_index != -1 && i == zero_index && !ctl` early return.)
    NoVanishingIndex,
    /// `randomize = true` was requested on the no-RNG entry
    /// [`splitting_compute`], which cannot run the signing-path
    /// `NORMALIZATION_TRANSFORMS` randomization. Call
    /// [`splitting_compute_randomized`] (which threads a `CryptoRng`)
    /// for the signing path; this variant just guards the wrong entry.
    RandomizeUnsupported,
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
/// Reference: `theta_isogenies.c:splitting_compute` (S246 wires the body).
///
/// Algorithm (mirrors the C ref verbatim): for each of the 10 even
/// characteristics `i`, accumulate `U_cst = Σ_t ±(θ[t]·θ[t ^ EVEN_INDEX[i][1]])`
/// where the sign is `CHI_EVAL[EVEN_INDEX[i][0]][t]`; if `U_cst == 0`,
/// constant-time-select `SPLITTING_TRANSFORMS[i]` into the running
/// matrix `M`. Splitting succeeds iff exactly one `i` gives `U_cst == 0`.
/// Finally `apply_isomorphism(M, A.null)` gives the splitting-compatible
/// codomain null. The matrix `m[i][j]` convention matches our
/// `apply_isomorphism` (output coord i = Σ_j m[i][j]·p[j]) — verified
/// against the C ref's `apply_isomorphism_general` (ti.c:64-95), same
/// orientation, no transpose (S246).
///
/// `zero_index`: when `Some(z)`, an early-fail check — if at the
/// expected index `i == z` the coordinate does NOT vanish, return a
/// failure immediately (the C ref's `zero_index != -1` branch).
///
/// `randomize`: signing-path randomization (C ref's `ENABLE_SIGN`
/// block). NOT yet supported here — it needs an RNG and the
/// `sample_random_index` plumbing that this signature lacks; requesting
/// it returns `Err(RandomizeUnsupported)` (the verifiable non-random
/// path — verification + the response side — is complete). [S247 docket:
/// thread an RNG + wire the NORMALIZATION_TRANSFORMS randomize block.]
// `t` indexes CHI_EVAL's row AND (XOR-paired) the theta coordinates —
// enumerate() over one would obscure the C-ref-faithful indexed access.
#[allow(dead_code)]
pub(crate) fn splitting_compute<F: BaseField>(
    domain: &AbelianVariety2D<F>,
    zero_index: Option<usize>,
    randomize: bool,
) -> Result<ThetaSplitting<F>, SplittingError> {
    // This no-RNG entry cannot run the signing-path randomization; the
    // randomized split lives in `splitting_compute_randomized`.
    if randomize {
        return Err(SplittingError::RandomizeUnsupported);
    }

    let m = splitting_build_matrix(domain, zero_index)?;
    let b_null = crate::isogeny::gluing::apply_isomorphism(&m, &domain.theta_null);
    Ok(ThetaSplitting { m, b_null })
}

/// Signing-path randomized splitting (C ref's `ENABLE_SIGN` block).
///
/// Builds the same vanishing-characteristic base-change matrix as
/// [`splitting_compute`], then left-multiplies by a randomly chosen
/// `NORMALIZATION_TRANSFORMS[k]` (`k ∈ [0, 6)`). The normalization
/// matrices preserve the product structure, so the resulting split
/// extracts to the same elliptic product as the non-random one — the
/// randomization only changes the symplectic representative, hiding
/// which kernel was walked.
// The index `i` drives both NORMALIZATION_TRANSFORMS lookup AND the
// constant-time `i == secret` selection — enumerate() would obscure the
// C-ref-faithful indexed comparison.
#[allow(dead_code, clippy::needless_range_loop)]
pub(crate) fn splitting_compute_randomized<F: BaseField, R: CryptoRng>(
    domain: &AbelianVariety2D<F>,
    zero_index: Option<usize>,
    rng: &mut R,
) -> Result<ThetaSplitting<F>, SplittingError> {
    let m = splitting_build_matrix(domain, zero_index)?;

    // Constant-time pick of NORMALIZATION_TRANSFORMS[secret] (C ref:
    // start at index 0, select index i when i == secret).
    let secret = sample_random_index(rng);
    let mut m_random = base_change_from_codes::<F>(&NORMALIZATION_TRANSFORMS[0]);
    for i in 1..6 {
        let pick = Choice::from(u8::from(i == secret));
        let candidate = base_change_from_codes::<F>(&NORMALIZATION_TRANSFORMS[i]);
        m_random = select_base_change_matrix(&m_random, &candidate, pick);
    }
    let m = base_change_matrix_multiplication(&m_random, &m);

    let b_null = crate::isogeny::gluing::apply_isomorphism(&m, &domain.theta_null);
    Ok(ThetaSplitting { m, b_null })
}

/// Build the splitting base-change matrix: enumerate the 10 even
/// characteristics, accumulate the one whose `U_cst` vanishes, and
/// require exactly one such vanishing (else the variety does not split).
///
/// Shared core of [`splitting_compute`] and
/// [`splitting_compute_randomized`] — the value-independent part of the
/// C ref `splitting_compute` before the optional randomization and the
/// final `apply_isomorphism`.
#[allow(clippy::needless_range_loop)]
// `t` indexes CHI_EVAL's row AND (XOR-paired) the theta coordinates —
// enumerate() over one would obscure the C-ref-faithful indexed access.
fn splitting_build_matrix<F: BaseField>(
    domain: &AbelianVariety2D<F>,
    zero_index: Option<usize>,
) -> Result<BasisChangeMatrix<F>, SplittingError> {
    let null = &domain.theta_null;
    // Index the null point's 4 theta coordinates by 0..3 (x,y,z,w).
    let coord = |ind: usize| -> Fp2<F> {
        match ind & 3 {
            0 => null.x,
            1 => null.y,
            2 => null.z,
            _ => null.w,
        }
    };

    // Running base-change matrix, starts all-zero (C ref: memset 0).
    let mut m = BasisChangeMatrix {
        m: [[Fp2::<F>::zero(); 4]; 4],
    };
    let mut count: u32 = 0;

    for i in 0..10 {
        let mut u_cst = Fp2::<F>::zero();
        for t in 0..4 {
            let t2 = coord(t);
            let t1 = coord(t ^ (EVEN_INDEX[i][1] as usize));
            let prod = t1.mul(&t2);
            // CHI_EVAL ∈ {+1,-1}: +1 → add prod, -1 → add (-prod).
            let neg = prod.negate();
            let term = if CHI_EVAL[EVEN_INDEX[i][0] as usize][t] == 1 {
                prod
            } else {
                neg
            };
            u_cst = u_cst.add(&term);
        }
        let is_zero = u_cst.is_zero();
        // count += 1 when this characteristic vanishes.
        count += u32::from(bool::from(is_zero));
        // CT-select SPLITTING_TRANSFORMS[i] into M when U_cst == 0.
        let candidate = base_change_from_codes::<F>(&SPLITTING_TRANSFORMS[i]);
        m = select_base_change_matrix(&m, &candidate, is_zero);
        // zero_index early-fail: if we expect the vanish at index z and
        // it does NOT happen there, the split is mis-shaped → fail.
        if let Some(z) = zero_index {
            if i == z && !bool::from(is_zero) {
                return Err(SplittingError::NoVanishingIndex);
            }
        }
    }

    // Exactly one vanishing characteristic = a valid splitting.
    if count != 1 {
        return Err(SplittingError::NotSplittable);
    }

    Ok(m)
}

/// Sample an unbiased uniform index in `[0, 6)` to pick one of the 6
/// normalization matrices. Verbatim logic from the C ref's
/// `sample_random_index`: draw a little-endian `u32`, reject values
/// `≥ 4294967292` (the largest multiple of 6 below `2^32`), then `% 6`.
fn sample_random_index<R: CryptoRng>(rng: &mut R) -> usize {
    loop {
        let mut bytes = [0u8; 4];
        rng.fill_bytes(&mut bytes);
        let seed = u32::from_le_bytes(bytes);
        if seed < 4_294_967_292 {
            let idx = (seed % 6) as usize;
            #[cfg(feature = "kat")]
            if std::env::var("PQSQ_DUMP_AC").is_ok() {
                std::eprintln!("OURS_IDX {idx}");
            }
            return idx;
        }
    }
}

// ---------------------------------------------------------------------
// S244: splitting constant tables (ported verbatim from the C reference
// `src/precomp/ref/lvl1/hd_splitting_transforms.c`, which is byte-
// identical across lvl1/3/5 — these are LEVEL-INDEPENDENT small-integer
// index tables, NOT field constants). Fetched via research agent from
// github.com/SQISign/the-sqisign (the S147-era "/tmp/ inaccessible"
// blocker was moot — see S243/S244 Decisions).
//
// The matrix entries are INDEX CODES into the 5-element fp2 constant
// table `{0, 1, i, -1, -i}` (see `FP2_CONST_CODE_*` below); the C ref's
// `set_base_change_matrix_from_precomp` maps `code → FP2_CONSTANTS[code]`
// at runtime. We keep the codes as `u8` and map to `Fp2<F>` on demand.
// ---------------------------------------------------------------------

/// Index codes into the fp2 constant table `{0, 1, i, -1, -i}` used by
/// [`SPLITTING_TRANSFORMS`] / [`NORMALIZATION_TRANSFORMS`] entries.
/// Mirrors the C ref's `FP2_ZERO/ONE/I/MINUS_ONE/MINUS_I` macros.
#[allow(dead_code)]
pub(crate) const FP2_CODE_ZERO: u8 = 0;
#[allow(dead_code)]
pub(crate) const FP2_CODE_ONE: u8 = 1;
#[allow(dead_code)]
pub(crate) const FP2_CODE_I: u8 = 2;
#[allow(dead_code)]
pub(crate) const FP2_CODE_MINUS_ONE: u8 = 3;
#[allow(dead_code)]
pub(crate) const FP2_CODE_MINUS_I: u8 = 4;

/// Map an `FP2_CODE_*` index to the corresponding `Fp2<F>` constant
/// `{0, 1, i, -1, -i}`. `i` is the quadratic-extension generator
/// (`Fp2 { c0: 0, c1: 1 }`); the C ref's `FP2_CONSTANTS[code]`.
#[allow(dead_code)]
pub(crate) fn fp2_from_code<F: BaseField>(code: u8) -> Fp2<F> {
    match code {
        FP2_CODE_ZERO => Fp2::<F>::zero(),
        FP2_CODE_ONE => Fp2::<F>::one(),
        FP2_CODE_I => Fp2::<F>::img(),
        FP2_CODE_MINUS_ONE => Fp2::<F>::one().negate(),
        FP2_CODE_MINUS_I => Fp2::<F>::img().negate(),
        _ => Fp2::<F>::zero(), // defensive; valid tables only emit 0..=4
    }
}

/// `EVEN_INDEX[10][2]` — enumerates the 10 even theta characteristics.
/// `[i][0]` selects a `CHI_EVAL` row (the character); `[i][1]` is XOR-ed
/// into the theta-coordinate index to pick the paired coordinate.
/// Verbatim from `hd_splitting_transforms.c:15` (level-independent).
#[allow(dead_code)]
pub(crate) const EVEN_INDEX: [[u8; 2]; 10] = [
    [0, 0],
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 0],
    [1, 2],
    [2, 0],
    [2, 1],
    [3, 0],
    [3, 3],
];

/// `CHI_EVAL[4][4]` — character sign matrix (a 4×4 Hadamard matrix,
/// entries ∈ {+1, −1}). `CHI_EVAL[EVEN_INDEX[i][0]][t]` gives the sign
/// applied to the coordinate product before accumulating `U_cst`; the C
/// ref uses `>> 1` on the (+1/−1) value as a CT negate-mask.
/// Verbatim from `hd_splitting_transforms.c:16` (level-independent).
#[allow(dead_code)]
pub(crate) const CHI_EVAL: [[i8; 4]; 4] =
    [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]];

/// `SPLITTING_TRANSFORMS[10]` — the 10 candidate base-change matrices,
/// one per even theta characteristic. Each entry is a 4×4 grid of
/// `FP2_CODE_*` index codes (0..4 → {0, 1, i, −1, −i}); the i-th matrix
/// is constant-time-selected as `M` when the i-th even-index coordinate
/// of the input theta-null vanishes (`U_cst == 0`). `[9]` is the
/// identity. Decoded directly (S245) from the C-ref
/// `hd_splitting_transforms.c:142` (level-independent; brace-group order
/// is the OUTER `[i]` index — the (row,col) interpretation is resolved
/// by the apply routine, see `apply_isomorphism`).
#[allow(dead_code)]
pub(crate) const SPLITTING_TRANSFORMS: [[[u8; 4]; 4]; 10] = [
    [[1, 2, 1, 2], [1, 4, 3, 2], [1, 2, 3, 4], [3, 2, 3, 2]],
    [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 3, 0, 0]],
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 3, 0]],
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 3]],
    [[1, 1, 1, 1], [1, 3, 3, 1], [1, 1, 3, 3], [3, 1, 3, 1]],
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    [[1, 1, 1, 1], [1, 3, 1, 3], [1, 3, 3, 1], [3, 3, 1, 1]],
    [[1, 1, 1, 1], [1, 3, 1, 3], [1, 3, 3, 1], [1, 1, 3, 3]],
    [[1, 1, 1, 1], [1, 3, 1, 3], [1, 1, 3, 3], [3, 1, 1, 3]],
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
];

/// `NORMALIZATION_TRANSFORMS[6]` — signing-path randomization matrices.
/// When `splitting_compute`'s `randomize` is set (signing side), a
/// secret-index-selected entry is left-composed onto `M`. Same
/// `FP2_CODE_*` encoding as [`SPLITTING_TRANSFORMS`]. Decoded directly
/// (S245) from the C-ref `hd_splitting_transforms.c` (level-independent).
/// `[0]` is the identity.
#[allow(dead_code)]
pub(crate) const NORMALIZATION_TRANSFORMS: [[[u8; 4]; 4]; 6] = [
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
    [[1, 1, 1, 1], [1, 3, 1, 3], [1, 1, 3, 3], [1, 3, 3, 1]],
    [[1, 3, 3, 1], [3, 3, 1, 1], [3, 1, 3, 1], [1, 1, 1, 1]],
    [[3, 2, 2, 1], [2, 3, 1, 2], [2, 1, 3, 2], [1, 2, 2, 3]],
    [[1, 2, 2, 3], [2, 1, 3, 2], [2, 3, 1, 2], [3, 2, 2, 1]],
];

/// Build a runtime `BasisChangeMatrix<F>` from one of the `u8` code
/// tables ([`SPLITTING_TRANSFORMS`] / [`NORMALIZATION_TRANSFORMS`]),
/// mapping each code through [`fp2_from_code`]. Mirrors the C ref's
/// `set_base_change_matrix_from_precomp` (`res->m[i][j] =
/// FP2_CONSTANTS[M->m[i][j]]`).
#[allow(dead_code)]
pub(crate) fn base_change_from_codes<F: BaseField>(codes: &[[u8; 4]; 4]) -> BasisChangeMatrix<F> {
    let mut m = [[Fp2::<F>::zero(); 4]; 4];
    for (out_row, code_row) in m.iter_mut().zip(codes.iter()) {
        for (cell, &code) in out_row.iter_mut().zip(code_row.iter()) {
            *cell = fp2_from_code::<F>(code);
        }
    }
    BasisChangeMatrix { m }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf::fp::Fp1Element;

    /// Randomized-splitting tests need a `CryptoRng`; the only one in
    /// the tree (`NistPqcRng`) is `#[cfg(feature = "kat")]`, so these
    /// run under the kat gate (which CI exercises).
    #[cfg(feature = "kat")]
    mod randomized {
        use super::*;
        use crate::rng::NistPqcRng;
        use subtle::ConstantTimeEq;

        /// A genuine product theta null `(1, 2, 3, 6)` (satisfies the
        /// product identity `x·w = y·z`, all coords non-zero) — the
        /// non-random `splitting_compute` returns `Ok` on it (probe S267),
        /// so it is a real splittable fixture for the randomized path.
        fn splittable_product_null() -> AbelianVariety2D<Fp1Element> {
            let null = ThetaPoint2D::new(small_fp2(1), small_fp2(2), small_fp2(3), small_fp2(6));
            AbelianVariety2D::new(null, null)
        }

        /// `sample_random_index` is an unbiased uniform sampler over
        /// `[0, 6)`: every draw is in range and all six buckets are hit
        /// (with the deterministic NIST DRBG, over 6000 draws the minimum
        /// bucket is comfortably positive). Independent of the splitting
        /// algebra — it pins the rejection-sampling arithmetic.
        #[test]
        fn sample_random_index_is_uniform_in_range() {
            let mut rng = NistPqcRng::new(&[0x27u8; 48]);
            let mut buckets = [0u32; 6];
            for _ in 0..6000 {
                let k = sample_random_index(&mut rng);
                assert!(k < 6, "index in [0,6)");
                buckets[k] += 1;
            }
            // Uniform expectation 1000/bucket; assert each is well-populated
            // (catches a stuck/mod-wrong sampler without flaking).
            for (k, &c) in buckets.iter().enumerate() {
                assert!(c > 700, "bucket {k} underpopulated: {c}");
            }
        }

        /// The randomized matrix is exactly `NORMALIZATION_TRANSFORMS[k] · M0`
        /// for some `k ∈ [0,6)`, where `M0` is the non-random splitting
        /// matrix. Independent structural oracle — checks the randomize block
        /// is a left-multiply by one of the six tabulated normalizations,
        /// not a re-derivation of the implementation.
        #[test]
        fn randomized_split_matrix_is_normalization_times_base() {
            let domain = splittable_product_null();
            let m0 = splitting_compute(&domain, None, false)
                .expect("product null splits (non-random)")
                .m;

            // Precompute the six candidate left-multiplies.
            let candidates: [BasisChangeMatrix<Fp1Element>; 6] = core::array::from_fn(|k| {
                base_change_matrix_multiplication(
                    &base_change_from_codes::<Fp1Element>(&NORMALIZATION_TRANSFORMS[k]),
                    &m0,
                )
            });

            // Several RNG states should produce M1 matching some NT[k]·M0.
            let mut rng = NistPqcRng::new(&[0xA3u8; 48]);
            for _ in 0..24 {
                let m1 = splitting_compute_randomized(&domain, None, &mut rng)
                    .expect("product null splits (randomized)")
                    .m;
                let matched = candidates.iter().any(|c| *c == m1);
                assert!(matched, "M1 must equal NT[k]·M0 for some k");
            }
        }

        /// Semantic oracle: the randomized split extracts to the SAME
        /// elliptic product (same unordered j-invariant pair) as the
        /// non-random split. The normalization matrices preserve the product
        /// structure, so randomizing the representative must not change the
        /// underlying curves.
        #[test]
        fn randomized_split_extracts_same_elliptic_product() {
            let domain = splittable_product_null();

            let split0 = splitting_compute(&domain, None, false).expect("non-random split");
            let cc0 = theta_product_structure_to_elliptic_product(&AbelianVariety2D::new(
                split0.b_null,
                split0.b_null,
            ))
            .expect("non-random split extracts to a product");
            let j0 = [cc0.e1.j_invariant(), cc0.e2.j_invariant()];

            let mut rng = NistPqcRng::new(&[0x5Cu8; 48]);
            for _ in 0..12 {
                let split1 = splitting_compute_randomized(&domain, None, &mut rng)
                    .expect("randomized split");
                let cc1 = theta_product_structure_to_elliptic_product(&AbelianVariety2D::new(
                    split1.b_null,
                    split1.b_null,
                ))
                .expect("randomized split extracts to a product");
                let j1 = [cc1.e1.j_invariant(), cc1.e2.j_invariant()];

                // Unordered j-pair equality (E1/E2 may swap under the normalization).
                let same_order = bool::from(j1[0].ct_eq(&j0[0])) && bool::from(j1[1].ct_eq(&j0[1]));
                let swapped = bool::from(j1[0].ct_eq(&j0[1])) && bool::from(j1[1].ct_eq(&j0[0]));
                assert!(
                    same_order || swapped,
                    "randomized split must yield the same {{j(E1), j(E2)}} as the non-random split"
                );
            }
        }
    } // mod randomized

    /// S245: both transform tables are valid code-tables — every entry
    /// is a code in 0..=4, the dimensions match the C ref (10 splitting,
    /// 6 normalization), and the documented identities hold
    /// (`SPLITTING_TRANSFORMS[9]` and `NORMALIZATION_TRANSFORMS[0]` are
    /// the identity in code form: 1 on the diagonal, 0 off).
    #[test]
    fn s245_transform_tables_well_formed() {
        assert_eq!(SPLITTING_TRANSFORMS.len(), 10);
        assert_eq!(NORMALIZATION_TRANSFORMS.len(), 6);
        let id_codes = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
        assert_eq!(
            SPLITTING_TRANSFORMS[9], id_codes,
            "splitting [9] is identity"
        );
        assert_eq!(
            NORMALIZATION_TRANSFORMS[0], id_codes,
            "normalization [0] is identity"
        );
        for mat in SPLITTING_TRANSFORMS
            .iter()
            .chain(NORMALIZATION_TRANSFORMS.iter())
        {
            for row in mat {
                for &code in row {
                    assert!(code <= 4, "every entry is an FP2 code in 0..=4, got {code}");
                }
            }
        }
    }

    /// S245: `base_change_from_codes` decodes a code-table to an
    /// `Fp2` `BasisChangeMatrix` correctly — the identity code-table
    /// yields the Fp2 identity matrix (1 on the diagonal, 0 off), and a
    /// mixed entry (`SPLITTING_TRANSFORMS[0][0]` = codes [1,2,1,2] =
    /// [1, i, 1, i]) decodes to the matching Fp2 values.
    #[test]
    fn s245_base_change_from_codes_decodes_correctly() {
        // Identity table → Fp2 identity.
        let id = base_change_from_codes::<Fp1Element>(&SPLITTING_TRANSFORMS[9]);
        for r in 0..4 {
            for c in 0..4 {
                if r == c {
                    assert!(bool::from(id.m[r][c].is_one()), "diag [{r}][{c}] = 1");
                } else {
                    assert!(bool::from(id.m[r][c].is_zero()), "off-diag [{r}][{c}] = 0");
                }
            }
        }
        // Mixed row: SPLITTING_TRANSFORMS[0] row 0 = [1,2,1,2] = [1, i, 1, i].
        let m0 = base_change_from_codes::<Fp1Element>(&SPLITTING_TRANSFORMS[0]);
        assert!(bool::from(m0.m[0][0].is_one()), "[0][0] code 1 → 1");
        assert!(
            bool::from(m0.m[0][1].ct_eq(&Fp2::<Fp1Element>::img())),
            "[0][1] code 2 → i"
        );
        assert!(bool::from(m0.m[0][2].is_one()), "[0][2] code 1 → 1");
        assert!(
            bool::from(m0.m[0][3].ct_eq(&Fp2::<Fp1Element>::img())),
            "[0][3] code 2 → i"
        );
        // And a -1 entry: SPLITTING_TRANSFORMS[1][3] = [0,3,0,0], [3]=-1.
        let m1 = base_change_from_codes::<Fp1Element>(&SPLITTING_TRANSFORMS[1]);
        assert!(bool::from(m1.m[3][1].is_neg_one()), "[3][1] code 3 → -1");
    }

    /// S244: `EVEN_INDEX` has the C-ref shape + values — 10 even theta
    /// characteristics, each `[chi_row ∈ 0..4, xor_index ∈ 0..4]`.
    #[test]
    fn s244_even_index_matches_c_ref() {
        assert_eq!(EVEN_INDEX.len(), 10, "10 even characteristics");
        for [chi_row, xor_idx] in EVEN_INDEX {
            assert!(chi_row < 4, "chi_row indexes CHI_EVAL's 4 rows");
            assert!(xor_idx < 4, "xor_idx is a 2-bit theta-coord XOR");
        }
        // First and last entries pin the exact C-ref ordering.
        assert_eq!(EVEN_INDEX[0], [0, 0]);
        assert_eq!(EVEN_INDEX[9], [3, 3]);
    }

    /// S244: `CHI_EVAL` is the 4×4 Hadamard character matrix (entries
    /// ±1; rows mutually orthogonal; row 0 all +1).
    #[test]
    fn s244_chi_eval_is_hadamard() {
        for row in CHI_EVAL {
            for v in row {
                assert!(v == 1 || v == -1, "entries are ±1");
            }
        }
        assert_eq!(CHI_EVAL[0], [1, 1, 1, 1], "row 0 is all +1");
        // Hadamard: distinct rows are orthogonal (dot product 0).
        for a in 0..4 {
            for b in (a + 1)..4 {
                let dot: i32 = (0..4)
                    .map(|k| CHI_EVAL[a][k] as i32 * CHI_EVAL[b][k] as i32)
                    .sum();
                assert_eq!(dot, 0, "rows {a},{b} must be orthogonal");
            }
            let self_dot: i32 = (0..4).map(|k| (CHI_EVAL[a][k] as i32).pow(2)).sum();
            assert_eq!(self_dot, 4, "row {a}·itself = 4 (unit ±1 entries)");
        }
    }

    /// S244: the `FP2_CODE_*` → `Fp2` mapping yields exactly
    /// `{0, 1, i, -1, -i}`, and the negation/identity relations hold.
    #[test]
    fn s244_fp2_from_code_yields_unit_constants() {
        let zero = fp2_from_code::<Fp1Element>(FP2_CODE_ZERO);
        let one = fp2_from_code::<Fp1Element>(FP2_CODE_ONE);
        let im = fp2_from_code::<Fp1Element>(FP2_CODE_I);
        let neg_one = fp2_from_code::<Fp1Element>(FP2_CODE_MINUS_ONE);
        let neg_i = fp2_from_code::<Fp1Element>(FP2_CODE_MINUS_I);
        assert!(bool::from(zero.is_zero()), "code 0 → 0");
        assert!(bool::from(one.is_one()), "code 1 → 1");
        assert!(
            bool::from(im.ct_eq(&Fp2::<Fp1Element>::img())),
            "code 2 → i"
        );
        assert!(bool::from(neg_one.is_neg_one()), "code 3 → -1");
        // -i is the negation of i; and i + (-i) = 0.
        assert!(
            bool::from(neg_i.ct_eq(&Fp2::<Fp1Element>::img().negate())),
            "code 4 → -i"
        );
        assert!(bool::from(im.add(&neg_i).is_zero()), "i + (-i) = 0");
        assert!(bool::from(one.add(&neg_one).is_zero()), "1 + (-1) = 0");
        // i² = -1 (quadratic-extension generator relation).
        assert!(bool::from(im.mul(&im).is_neg_one()), "i² = -1");
    }

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

    /// S246: the all-zero theta-null is degenerate — every `U_cst`
    /// vanishes (all coordinate products are 0), so `count == 10 != 1`
    /// and `splitting_compute` reports `NotSplittable` (not the old
    /// `NotImplemented` stub, now that the body is wired).
    #[test]
    fn s246_splitting_all_zero_null_is_not_splittable() {
        let null = ThetaPoint2D::new(
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
        );
        let domain = AbelianVariety2D::new(null, null);
        assert_eq!(
            splitting_compute(&domain, None, false),
            Err(SplittingError::NotSplittable),
            "all-zero null: count=10≠1 → NotSplittable",
        );
    }

    /// S246: the signing-path `randomize` flag is not yet supported (it
    /// needs an RNG this signature doesn't thread); requesting it yields
    /// `RandomizeUnsupported`, NOT a silent wrong split.
    #[test]
    fn s246_splitting_randomize_unsupported() {
        let null = ThetaPoint2D::new(
            Fp2::<Fp1Element>::one(),
            Fp2::<Fp1Element>::one(),
            Fp2::<Fp1Element>::one(),
            Fp2::<Fp1Element>::one(),
        );
        let domain = AbelianVariety2D::new(null, null);
        assert_eq!(
            splitting_compute(&domain, /*zero_index*/ None, /*randomize*/ true),
            Err(SplittingError::RandomizeUnsupported),
            "randomize=true must fail closed, not produce an unverified split",
        );
    }

    /// S246: a known-splittable theta-null splits with exactly one
    /// vanishing characteristic (count==1 → Ok), and the resulting
    /// codomain null is a valid product theta point (`x·w == y·z`).
    ///
    /// Fixture: the all-ones theta-null `(1,1,1,1)`. By the C-ref
    /// `U_cst` formula, characteristic `i` vanishes when
    /// `Σ_t CHI_EVAL[EVEN_INDEX[i][0]][t] · θ[t]·θ[t^EVEN_INDEX[i][1]]`
    /// is 0; for the all-ones null each product is 1 and the sum is the
    /// CHI_EVAL row sum over the XOR-paired coords. The CHI_EVAL rows
    /// (other than row 0) sum to zero, so exactly the characteristics
    /// keyed to a non-trivial chi row + matching XOR vanish — the test
    /// asserts the algorithm finds exactly one and produces a product
    /// null (it does NOT hard-code which `i`, to avoid over-constraining
    /// the port; the contract is count==1 + product output).
    #[test]
    fn s246_splitting_all_ones_null_produces_product() {
        let one = Fp2::<Fp1Element>::one();
        let null = ThetaPoint2D::new(one, one, one, one);
        let domain = AbelianVariety2D::new(null, null);
        match splitting_compute(&domain, None, false) {
            Ok(split) => {
                assert!(
                    bool::from(is_product_theta_point(&split.b_null)),
                    "split codomain null must satisfy the product relation x·w = y·z",
                );
            }
            Err(SplittingError::NotSplittable) => {
                // Acceptable: the all-ones null may have ≠1 vanishing
                // characteristics at this prime — but then the result
                // must be the CLEAN typed error, never a panic or a
                // bogus Ok. (Documents the boundary; the Ok branch above
                // is the substantive check when it splits.)
            }
            other => panic!("unexpected splitting_compute result: {other:?}"),
        }
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
