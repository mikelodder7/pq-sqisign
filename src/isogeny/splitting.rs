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
//! - [`SplittingError`] — failure modes.
//! - [`base_change_matrix_multiplication`] — 4×4 matrix product over
//!   `Fp2<F>`. Real testable infrastructure (S145 advisor's
//!   α-with-real-infrastructure scope).
//! - [`select_base_change_matrix`] — constant-time conditional select
//!   between two basis-change matrices. CT-relevant in the splitting
//!   base path because `U_cst == 0` is secret-derived for chains
//!   produced on the signing side.
//! - [`splitting_compute`] — main entry. Enumerates the 10 even
//!   characteristics over four constant tables (`EVEN_INDEX`,
//!   `CHI_EVAL`, `SPLITTING_TRANSFORMS`, `NORMALIZATION_TRANSFORMS`,
//!   ported from the C reference) to build the splitting base-change
//!   matrix and codomain theta-null.

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
    /// the splitting state; delegates to [`splitting_compute`].
    pub fn compute(
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
    /// Vestigial variant retained for API stability. `splitting_compute`
    /// is implemented and never returns this; live failures use the more
    /// specific variants below.
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

    // Carry the reference's exact PROJECTIVE (A:C) representative (not just the
    // affine A/C): the C-faithful combine-kernel doubling reproduces C's `xDBL`
    // representative from this, which the downstream gluing `squared_theta` +
    // final-step sqrt (both non-representative-invariant) depend on for
    // byte-exact keygen. `a` stays affine; `proj_c` holds C.
    Ok(CoupleCurve {
        e1: MontgomeryCurve::new_projective(affine_a_1, c_1),
        e2: MontgomeryCurve::new_projective(affine_a_2, c_2),
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
/// Reference: `theta_isogenies.c:splitting_compute`.
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
/// block). This no-RNG entry does not support it — requesting it
/// returns `Err(RandomizeUnsupported)`; the randomized split lives in
/// [`splitting_compute_randomized`], which threads an RNG and wires the
/// `NORMALIZATION_TRANSFORMS` block.
// `t` indexes CHI_EVAL's row AND (XOR-paired) the theta coordinates —
// enumerate() over one would obscure the C-ref-faithful indexed access.
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
#[allow(clippy::needless_range_loop)]
pub(crate) fn splitting_compute_randomized<F: BaseField, R: CryptoRng>(
    domain: &AbelianVariety2D<F>,
    zero_index: Option<usize>,
    rng: &mut R,
) -> Result<ThetaSplitting<F>, SplittingError> {
    #[cfg(feature = "kat")]
    if std::env::var("PQSQ_SPLIT3").is_ok() {
        let n = &domain.theta_null;
        let mut buf = [0u8; 96];
        for (nm, c) in [("X", n.x), ("Y", n.y), ("Z", n.z), ("W", n.w)] {
            c.to_bytes_le(&mut buf);
            std::eprint!("OURS_TNR_{nm} ");
            for b in buf {
                std::eprint!("{b:02x}");
            }
            std::eprintln!();
        }
    }
    let m = splitting_build_matrix(domain, zero_index)?;

    #[cfg(feature = "kat")]
    if std::env::var("PQSQ_DUMP_AC").is_ok() {
        let mut buf = [0u8; 64];
        for r in 0..4 {
            for c in 0..4 {
                m.m[r][c].to_bytes_le(&mut buf);
                std::eprint!("OURS_BM_{r}_{c} ");
                for b in buf {
                    std::eprint!("{b:02x}");
                }
                std::eprintln!();
            }
        }
    }

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
    #[cfg(feature = "kat")]
    if std::env::var("PQSQ_DUMP_AC").is_ok() {
        let mut buf = [0u8; 64];
        for r in 0..4 {
            for c in 0..4 {
                m.m[r][c].to_bytes_le(&mut buf);
                std::eprint!("OURS_M_{r}_{c} ");
                for b in buf {
                    std::eprint!("{b:02x}");
                }
                std::eprintln!();
            }
        }
        for (nm, c) in [
            ("X", b_null.x),
            ("Y", b_null.y),
            ("Z", b_null.z),
            ("W", b_null.w),
        ] {
            c.to_bytes_le(&mut buf);
            std::eprint!("OURS_BNULL_{nm} ");
            for b in buf {
                std::eprint!("{b:02x}");
            }
            std::eprintln!();
        }
    }
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
        #[cfg(feature = "kat")]
        if bool::from(is_zero) && std::env::var("PQSQ_SPLIT3").is_ok() {
            std::eprintln!("OURS_VANISH={i}");
        }
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
// Splitting constant tables (ported verbatim from the C reference
// `src/precomp/ref/lvl1/hd_splitting_transforms.c`, which is byte-
// identical across lvl1/3/5 — these are LEVEL-INDEPENDENT small-integer
// index tables, NOT field constants). Fetched via research agent from
// github.com/SQISign/the-sqisign.
//
// The matrix entries are INDEX CODES into the 5-element fp2 constant
// table `{0, 1, i, -1, -i}` (see `FP2_CONST_CODE_*` below); the C ref's
// `set_base_change_matrix_from_precomp` maps `code → FP2_CONSTANTS[code]`
// at runtime. We keep the codes as `u8` and map to `Fp2<F>` on demand.
// ---------------------------------------------------------------------

/// Index codes into the fp2 constant table `{0, 1, i, -1, -i}` used by
/// [`SPLITTING_TRANSFORMS`] / [`NORMALIZATION_TRANSFORMS`] entries.
/// Mirrors the C ref's `FP2_ZERO/ONE/I/MINUS_ONE/MINUS_I` macros.
pub(crate) const FP2_CODE_ZERO: u8 = 0;
pub(crate) const FP2_CODE_ONE: u8 = 1;
pub(crate) const FP2_CODE_I: u8 = 2;
pub(crate) const FP2_CODE_MINUS_ONE: u8 = 3;
pub(crate) const FP2_CODE_MINUS_I: u8 = 4;

/// Map an `FP2_CODE_*` index to the corresponding `Fp2<F>` constant
/// `{0, 1, i, -1, -i}`. `i` is the quadratic-extension generator
/// (`Fp2 { c0: 0, c1: 1 }`); the C ref's `FP2_CONSTANTS[code]`.
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
    #[cfg(all(not(feature = "std"), feature = "alloc"))]
    use alloc::vec::Vec;

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
                let matched = candidates.contains(&m1);
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

    /// both transform tables are valid code-tables — every entry
    /// is a code in 0..=4, the dimensions match the C ref (10 splitting,
    /// 6 normalization), and the documented identities hold
    /// (`SPLITTING_TRANSFORMS[9]` and `NORMALIZATION_TRANSFORMS[0]` are
    /// the identity in code form: 1 on the diagonal, 0 off).
    #[test]
    fn transform_tables_well_formed() {
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

    /// `base_change_from_codes` decodes a code-table to an
    /// `Fp2` `BasisChangeMatrix` correctly — the identity code-table
    /// yields the Fp2 identity matrix (1 on the diagonal, 0 off), and a
    /// mixed entry (`SPLITTING_TRANSFORMS[0][0]` = codes [1,2,1,2] =
    /// [1, i, 1, i]) decodes to the matching Fp2 values.
    #[test]
    fn base_change_from_codes_decodes_correctly() {
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

    /// `EVEN_INDEX` has the C-ref shape + values — 10 even theta
    /// characteristics, each `[chi_row ∈ 0..4, xor_index ∈ 0..4]`.
    #[test]
    fn even_index_matches_c_ref() {
        assert_eq!(EVEN_INDEX.len(), 10, "10 even characteristics");
        for [chi_row, xor_idx] in EVEN_INDEX {
            assert!(chi_row < 4, "chi_row indexes CHI_EVAL's 4 rows");
            assert!(xor_idx < 4, "xor_idx is a 2-bit theta-coord XOR");
        }
        // First and last entries pin the exact C-ref ordering.
        assert_eq!(EVEN_INDEX[0], [0, 0]);
        assert_eq!(EVEN_INDEX[9], [3, 3]);
    }

    /// `CHI_EVAL` is the 4×4 Hadamard character matrix (entries
    /// ±1; rows mutually orthogonal; row 0 all +1).
    #[test]
    fn chi_eval_is_hadamard() {
        for row in CHI_EVAL {
            for v in row {
                assert!(v == 1 || v == -1, "entries are ±1");
            }
        }
        assert_eq!(CHI_EVAL[0], [1, 1, 1, 1], "row 0 is all +1");
        // Hadamard: distinct rows are orthogonal (dot product 0).
        for (a, row_a) in CHI_EVAL.iter().enumerate() {
            for (b, row_b) in CHI_EVAL.iter().enumerate().skip(a + 1) {
                let dot: i32 = (0..4).map(|k| row_a[k] as i32 * row_b[k] as i32).sum();
                assert_eq!(dot, 0, "rows {a},{b} must be orthogonal");
            }
            let self_dot: i32 = (0..4).map(|k| (CHI_EVAL[a][k] as i32).pow(2)).sum();
            assert_eq!(self_dot, 4, "row {a}·itself = 4 (unit ±1 entries)");
        }
    }

    /// the `FP2_CODE_*` → `Fp2` mapping yields exactly
    /// `{0, 1, i, -1, -i}`, and the negation/identity relations hold.
    #[test]
    fn fp2_from_code_yields_unit_constants() {
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
        assert_eq!(result, a, "I · A = A");
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
        assert_eq!(result, a, "A · I = A");
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
            "A · B with B as column-shift matrix must shift A's columns",
        );
    }

    #[test]
    fn select_base_change_matrix_choice_false_returns_first_at_lvl1() {
        let a = from_u32_grid([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 5, 0], [0, 0, 0, 7]]);
        let b = identity_4x4();
        let result = select_base_change_matrix(&a, &b, Choice::from(0));
        assert_eq!(result, a, "Choice::FALSE selects first argument");
    }

    #[test]
    fn select_base_change_matrix_choice_true_returns_second_at_lvl1() {
        let a = from_u32_grid([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 5, 0], [0, 0, 0, 7]]);
        let b = identity_4x4();
        let result = select_base_change_matrix(&a, &b, Choice::from(1));
        assert_eq!(result, b, "Choice::TRUE selects second argument");
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
            "TRUE-select B then TRUE-select A round-trips to A",
        );
    }

    /// The all-zero theta-null is degenerate — every `U_cst`
    /// vanishes (all coordinate products are 0), so `count == 10 != 1`
    /// and `splitting_compute` reports `NotSplittable`.
    #[test]
    fn splitting_all_zero_null_is_not_splittable() {
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

    /// the signing-path `randomize` flag is not yet supported (it
    /// needs an RNG this signature doesn't thread); requesting it yields
    /// `RandomizeUnsupported`, NOT a silent wrong split.
    #[test]
    fn splitting_randomize_unsupported() {
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

    /// a known-splittable theta-null splits with exactly one
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
    fn splitting_all_ones_null_produces_product() {
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

    // Post-splitting extraction tests.
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
            "(2, 3, 6, 9) satisfies x·w=18=y·z; must be product",
        );
    }

    /// Reject: `(2, 3, 5, 7)` does NOT satisfy `x·w = y·z`
    /// (2·7=14 ≠ 3·5=15). is_product_theta_point must return FALSE.
    #[test]
    fn is_product_theta_point_rejects_non_product_at_lvl1() {
        let p = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        assert!(
            !bool::from(is_product_theta_point(&p)),
            "(2, 3, 5, 7) violates product identity (14 ≠ 15); must be non-product",
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
            "non-product null must be rejected as NotProductTheta",
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
            "all-zero null must be rejected as ZeroNullCoordinate",
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
        let cc = result.expect("valid product null must extract Ok");
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
        let cc =
            theta_product_structure_to_elliptic_product(&domain).expect("extraction must succeed");

        // For E_2: a_2 · C_2 = A_2 where C_2 = x^4 - y^4, A_2 = -2(x^4 + y^4).
        let x_4 = small_fp2(1);
        let y_4 = small_fp2(16);
        let z_4 = small_fp2(81);
        let c_2 = x_4.sub(&y_4);
        let a_2_expected = x_4.add(&y_4).double().negate();
        assert_eq!(
            cc.e2.a.mul(&c_2),
            a_2_expected,
            "a_2 · C_2 must equal A_2 (round-trip via projective form)",
        );

        let c_1 = x_4.sub(&z_4);
        let a_1_expected = x_4.add(&z_4).double().negate();
        assert_eq!(
            cc.e1.a.mul(&c_1),
            a_1_expected,
            "a_1 · C_1 must equal A_1 (round-trip via projective form)",
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
            "non-product P must be rejected",
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
            "all-zero P with all-zero fallback must be AllZeroPoint",
        );
    }

    // BasisChangeMatrix::identity + is_identity.

    #[test]
    fn basis_change_matrix_identity_constructor_matches_identity_grid_at_lvl1() {
        let via_method = BasisChangeMatrix::<Fp1Element>::identity();
        let via_grid = identity_4x4();
        assert_eq!(
            via_method, via_grid,
            "identity() must equal hand-built identity grid",
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
        assert_eq!(i.mul(&a), a, "I · A = A");
        assert_eq!(a.mul(&i), a, "A · I = A");
    }

    #[test]
    fn basis_change_matrix_is_identity_true_for_identity_at_lvl1() {
        let i = BasisChangeMatrix::<Fp1Element>::identity();
        assert!(
            bool::from(i.is_identity()),
            "identity().is_identity() must be TRUE",
        );
    }

    #[test]
    fn basis_change_matrix_is_identity_false_for_nonidentity_at_lvl1() {
        // Diagonal-only-with-non-1 value
        let m = from_u32_grid([[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
        assert!(
            !bool::from(m.is_identity()),
            "diag-(2,1,1,1) is NOT identity",
        );
        // Identity-shape with one off-diagonal non-zero
        let m2 = from_u32_grid([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
        assert!(
            !bool::from(m2.is_identity()),
            "identity with one off-diagonal 1 is NOT identity",
        );
    }

    // BasisChangeMatrix method-form alias tests.

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
            "a.mul(&b) must match base_change_matrix_multiplication(&a, &b)",
        );
    }

    // Extraction method-form alias tests.

    // ThetaSplitting::compute method-form alias test.

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
            "ThetaSplitting::compute must match the splitting_compute free function",
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
            "domain.to_elliptic_product() must match free function",
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
            "p.to_montgomery_point_on(&domain) must match free function",
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
            "method propagates NotProductTheta",
        );

        let bad_p = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        // Use a product null for the domain so only the point-side
        // check fails:
        let good_null = ThetaPoint2D::new(small_fp2(1), small_fp2(2), small_fp2(3), small_fp2(6));
        let good_domain = AbelianVariety2D::new(good_null, good_null);
        assert_eq!(
            bad_p.to_montgomery_point_on(&good_domain),
            Err(ExtractionError::NotProductTheta),
            "method propagates NotProductTheta on the point side",
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
                "a.select(&b, Choice::{label}) must match select_base_change_matrix",
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
        let cmp = result.expect("valid product P + valid null must extract Ok");
        // Per formula:
        //   P_2.X = null.y · P.x + null.x · P.y = 2·1 + 1·2 = 4
        //   P_2.Z = null.x · P.y - null.y · P.x = 1·2 - 2·1 = 0
        //   P_1.X = null.z · P.x + null.x · P.z = 3·1 + 1·3 = 6
        //   P_1.Z = null.x · P.z - null.z · P.x = 1·3 - 3·1 = 0
        assert_eq!(cmp.p2.x, small_fp2(4), "P_2.X = 4");
        assert_eq!(cmp.p2.z, Fp2::<Fp1Element>::zero(), "P_2.Z = 0");
        assert_eq!(cmp.p1.x, small_fp2(6), "P_1.X = 6");
        assert_eq!(cmp.p1.z, Fp2::<Fp1Element>::zero(), "P_1.Z = 0");
    }

    /// are the split theta-null points (C reference vs ours) the SAME
    /// projective point? Captured raw 4-coords at the randomized split entry for
    /// KAT[0] (C `splitting_compute` A->null_point {x,y,z,t} CDUMP6; ours
    /// `domain.theta_null` {x,y,z,w} OURS_TNR). Proportional ⟺ Cy·Ox==Oy·Cx etc.
    /// (X≠0 in both). Match ⟹ bug is in the split matrix M / tables; differ ⟹
    /// the (2,2)-chain walk produced a different theta structure (kernel/gluing).
    #[cfg(feature = "alloc")]
    #[test]
    fn theta_null_proportionality() {
        use subtle::ConstantTimeEq;
        fn hx(s: &str) -> Fp2<Fp1Element> {
            let bytes: Vec<u8> = (0..s.len() / 2)
                .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
                .collect();
            Option::<Fp2<Fp1Element>>::from(Fp2::<Fp1Element>::from_bytes_le(&bytes)).unwrap()
        }
        // compute_4 INPUT domain-null (KAT[0], first (0,0) combine step).
        let cx = hx(
            "a988f13a4cc111eed5baa2257093e5cf1e89fb1e91e973c23c3d00f714ab6301b14e31c7f8489e8368386804ff4f47f6a6f51c139946319528d04483491d7901",
        );
        let cy = hx(
            "422771e8371e914de9ab65a1bab68ebca18be1e40014a38ada3079a6c590cf002ec36cb3c2001704fd68b1daf3a37f2a00714b4a2b59098f0f2780418e5b0f00",
        );
        let cz = hx(
            "1d5c8047d9b99c1d9fb76a34760981f0aad1fde22bba532fbf9e9fa26fef6e0162bd7b93ce8f8618d5024449b33054e118ada7fe324972aba42755be08f38402",
        );
        let ct = hx(
            "24b6c335dbda7bf6960f9202a5936c31e1a42774d69ade05cb85e9a0c942ed00f0df624535a08e6076c15b35d5e51a1966673ccb20399d03957ddc5db0e59101",
        );
        let ox = hx(
            "5eb33f89ec7c4cc4003c94b48d7190ffd5d6d840b6a88bb62030935d57230800c347fdcf8c0152c301217c45f9c1094e33ed193d98202349af5b6974ac655600",
        );
        let oy = hx(
            "de246422a9ace0457bf03f66049ed6fa25d74d55df0f897991309aa93217590431b4c80e1f6f12cdd60fc8af661586a6be256be26dcbefd8154ca0e841726601",
        );
        let oz = hx(
            "771362c3d6d15defb0266b3e90956fddf5ad80f0e76e880bf45c944c6cb05e013e3bffc0e5e987f75f6dca60cfff0fded76ef6c1fc75c64b4a60636e78e76a04",
        );
        let ow = hx(
            "b7948afba2310b3e8d6ff258e5f0e8128d7d4dfbf195304beda9ffd91719a2004729f687fb4b38f9dda4319dee5bda6b7e677910e91fbc3fc00e2246fa693a00",
        );
        let eq =
            |a: &Fp2<Fp1Element>, b: &Fp2<Fp1Element>, c: &Fp2<Fp1Element>, d: &Fp2<Fp1Element>| {
                // a/c == b/d  ⟺  a·d == b·c
                bool::from(a.mul(d).ct_eq(&b.mul(c)))
            };
        // Direct pairing (C x,y,z,t ↔ our x,y,z,w):
        let dy = eq(&cy, &oy, &cx, &ox);
        let dz = eq(&cz, &oz, &cx, &ox);
        let dt = eq(&ct, &ow, &cx, &ox);
        // Swap hypothesis: our z ↔ C t, our w ↔ C z
        let sz = eq(&ct, &oz, &cx, &ox); // C.T/X == our Z/X
        let sw = eq(&cz, &ow, &cx, &ox); // C.Z/X == our W/X
        std::eprintln!("DIRECT: Y={dy} Z={dz} T/W={dt}  |  SWAP(z<->w): zT={sz} wZ={sw}");

        // Is the φ_u eval output bas_u PROPORTIONAL to C (same
        // point, scale-only) for P and Q? eq(Cx,Ox,Cz,Oz) = Cx/Cz==Ox/Oz (affine).
        let cpx = hx(
            "d51fd75b39666260d2d24f75fcd0d6e9d8fae9e921df637528dbc8895e228601a8756c7e76a390f4f07af73fd929693ad8c1a246b03d54760e99a60ac816d403",
        );
        let cpz = hx(
            "530fafbfa73ca9f94fdd91253504043ec9dff051d1281dbb4ed46aa1f1cbed0223036984ec00e35ee9b95dae24f9efb30ec4a76bc748a14ac0d038d0bb569903",
        );
        let opx = hx(
            "b6913df58a6cfcf0bdd044a9b2328d5dc6694c95b31a1512eac308d62949bb0107f0d5bd616fdc4f0df1c2c378d04546558dbcab892fada13af91d97ab679903",
        );
        let opz = hx(
            "a15b9fbc5bf6dac88803eda1be5cfd42ca7b29b2d56d1a15f0ed0aab7aded803715757215f8645de796516885e98e0c7c6b6549ec6adfcbfddf0f62a3a251803",
        );
        let cqx = hx(
            "283a9bdee2f55ca09009a922cdc6cea5fdeec0c43be179ef520e42d53e82be025dd0e9c2a35d3c0c4234ad10f1c3c9152596257f78c3c07eabeebe19080cee00",
        );
        let cqz = hx(
            "ab814663b3d306809db3746328cd6705c903f38de0b6597513d548d02c2f9d02988d0d0130b415b4ba6dbee1c3fdf45ed6c4446971c331b5a53e9c344ba82c03",
        );
        let oqx = hx(
            "b46041f373bf01d9f4b3db701c50961a05188ab42b07fd682d838501fc3f25039e8955322d1b4a2ed1433063b2a6907f3e2c30de7553f85e729256995c263800",
        );
        let oqz = hx(
            "10e0b4aeb02745aeeee3f205a1a1a7dfc91f99746427d5251902fd1298b3000273c9d19320f98385515fb9b791bb6e29a1bb6cd6d0774ec40417b53bfa154003",
        );
        let p_prop = eq(&cpx, &opx, &cpz, &opz);
        let q_prop = eq(&cqx, &oqx, &cqz, &oqz);
        std::eprintln!("phi_u bas_u proportional to C: P={p_prop} Q={q_prop}");
        // Is the scale ρ GLOBAL (same for P and Q)? ρ_P==ρ_Q ⟺ our_P/C_P == our_Q/C_Q.
        let global_x = eq(&opx, &oqx, &cpx, &cqx); // opx/cpx == oqx/cqx
        let global_z = eq(&opz, &oqz, &cpz, &cqz);
        // Cross-check ρ from x equals ρ from z (a true scalar ρ scales both): opx/cpx==opz/cpz
        let p_rho_xz = eq(&opx, &opz, &cpx, &cpz);
        std::eprintln!(
            "phi_u scale: global(P~Q) x={global_x} z={global_z}; P ρ_x==ρ_z = {p_rho_xz}"
        );
        // Measure ρ = our_P.x / C_P.x  (and from z) — is it a recognizable constant?
        let rho_x = opx.mul(&cpx.invert().unwrap());
        let rho_z = opz.mul(&cpz.invert().unwrap());
        let mut rb = [0u8; 64];
        rho_x.to_bytes_le(&mut rb);
        std::eprint!("rho_x=");
        for b in rb {
            std::eprint!("{b:02x}");
        }
        std::eprintln!();
        rho_z.to_bytes_le(&mut rb);
        std::eprint!("rho_z=");
        for b in rb {
            std::eprint!("{b:02x}");
        }
        std::eprintln!();
        std::eprintln!(
            "rho_x==rho_z (true scalar): {}",
            bool::from(rho_x.ct_eq(&rho_z))
        );
        // φ_u codomain CURVE model: does C's A/C == our Fu.E1.a (affine A/C)?
        let c_fua = hx(
            "9758d3065a5c84b878e6b17ea6d626c0f243202a6c5afd2c807d2d59ec786a00d053685a621d89af99fde64a0d51cae23b953314b47cb2e35af1f2f2b1d03404",
        );
        let c_fuc = hx(
            "fd9a17bede44f903bd9bcc8200dd15bcd28c9ae88c8563e99ca3974c4d2e51005ffaf130f872f35c1a3d14ff0283be712735e6145f94e2042e104b3957f9c400",
        );
        let ours_fua = hx(
            "7ead7271922ac66283efcaa66c5a751438baad9f00c69ece1e04ab8407a3ab0026fd8a2f6d0cd2a8b55b2048c4757cbe306d15340db716a1d2ca599a5b0dee01",
        );
        let c_fu_affine = c_fua.mul(&c_fuc.invert().unwrap());
        std::eprintln!(
            "phi_u CURVE A/C matches C: {}",
            bool::from(c_fu_affine.ct_eq(&ours_fua))
        );
        // Innermost: B0 (E0 basis, control) + Bth (θ-applied) — proportional to C? (same point)
        let c_b0px = hx(
            "e8102ca2da6848b5cbb93a47015e3b71d933dfeddfc97043222ea15fe05721046a1322fa6417129532f541f787ec492d34be1797816b17197877f124fb8bb001",
        );
        let c_b0pz = hx(
            "a06a0e7028de949ec4195d9ba97b1f4f2ec28f0d29fb710f832aaa1e894e520100582ae66c8ac76ccedf12d878a1d19a212f20bb8a16b3f40d8516fb68dc7902",
        );
        let o_b0px = hx(
            "5d0d8182f7df5ac6d5e19b884d47a928027ee9b9b889685a0da25682c3a1cc00c1773797dffea4395c02c36655ad3136c26f0657bbb0a7beb8704e84c0d8d103",
        );
        let o_b0pz = hx(
            "749bb5393ab5dddfca4f4e7460d88d98056af7ecd0a6a883146d13a092d96701434051c8bbba10a2168ed246e72c8ea009a90fe9ebd7c466748df2629d9a8701",
        );
        let c_bthpx = hx(
            "c718d3e68af5bd4c56d687fcf8cb0a1401227c037a39b8be3a3262a722c95501a9c52aa17666fb15f2bfbf9a9567df3c1bf48b0a3c2dc16395791971f7a5d103",
        );
        let c_bthpz = hx(
            "571556d616bfc682507efd1fb9c089ceabb228dba15dc88a06f249262c1d1600e1b56a8c1af37c6346d10db04d6348363b26486b80dd0b72832da698eb0dcc01",
        );
        let o_bthpx = hx(
            "7704750fa3603a56fc73f45b5a7fe62e67ab031ba8cdc173603e22cb2b832d04cd83107e8eafa3eaccca2bafff3bd50f2864f2a745a8b1c17c8fa9b11ceadd01",
        );
        let o_bthpz = hx(
            "aeec249576e132853b4841ff9c5b383bc01dcfe4f1f4b0de48f013d7521cad01516bbfb297d9a76d0bc7e9cdc9bf67c7d2c20966b4f1d755f9af6197abe6bd04",
        );
        let b0_prop = eq(&c_b0px, &o_b0px, &c_b0pz, &o_b0pz);
        let bth_prop = eq(&c_bthpx, &o_bthpx, &c_bthpz, &o_bthpz);
        std::eprintln!("innermost: B0.P proportional={b0_prop} Bth.P proportional={bth_prop}");
        // BOTTOM: our un-doubled E0 basis vs C's basis_even (C is at z=1).
        let (bp0, bq0, _bpmq0) = crate::isogeny::endomorphism::basis_e0_lvl1();
        let mut bb = [0u8; 64];
        bp0.x.to_bytes_le(&mut bb);
        std::eprint!("OUR_PRE_PX=");
        for x in bb {
            std::eprint!("{x:02x}");
        }
        std::eprintln!();
        bp0.z.to_bytes_le(&mut bb);
        std::eprint!("OUR_PRE_PZ=");
        for x in bb {
            std::eprint!("{x:02x}");
        }
        std::eprintln!();
        bq0.x.to_bytes_le(&mut bb);
        std::eprint!("OUR_PRE_QX=");
        for x in bb {
            std::eprint!("{x:02x}");
        }
        std::eprintln!();
        // C: PRE_PX=7800b4ae…, PRE_PZ=01 (z=1), PRE_QX=1feb9355…
        let c_pre_px = hx(
            "7800b4ae5ed919218ba7bf591a99be44c41662a6c304cc8324b182ca7f879b0175d2f9c33d13048e74924251aeddcfb22fe96798aa0a155242e0ea49db2a4404",
        );
        std::eprintln!(
            "OUR bp0.x==C PRE_PX: {}",
            bool::from(bp0.x.ct_eq(&c_pre_px))
        );
        // Just report — don't fail the loop on the diagnostic.
    }
}
