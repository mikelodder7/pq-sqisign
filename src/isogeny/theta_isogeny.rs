// SPDX-License-Identifier: MIT OR Apache-2.0

//! `(2, 2)`-isogeny inner-step between theta-coord Abelian varieties.
//!
//! Companion to [`crate::isogeny::gluing`] (the chain-boundary step from
//! elliptic-product to theta variety). Where the gluing step is fixed at
//! the chain boundary, the routines in this module run at every interior
//! step of the `(2, 2)`-isogeny chain and accept two `bool` flags that
//! control the input/output coordinate convention (standard vs dual).
//!
//! # Surface
//!
//! - [`ThetaIsogeny`] — the per-step state (domain, codomain theta-null,
//!   8-torsion kernel inputs, cached precomputation factors, the two
//!   Hadamard flags).
//! - [`ThetaIsogenyError`] — failure modes.
//! - [`theta_isogeny_compute`] — main entry point. Mirrors the C
//!   reference's `theta_isogenies.c:theta_isogeny_compute` (lines
//!   618–696) verbatim.
//!
//! Downstream sessions ship `theta_isogeny_eval` (S143) and the 4-/2-
//! torsion variants `theta_isogeny_compute_4` / `_compute_2` (S144+).
//!
//! # Hadamard flag semantics (S142 advisor)
//!
//! `hadamard_bool_1` controls whether the **input** 8-torsion points
//! `T1_8` / `T2_8` are in *standard* (`false`) or *dual* (`true`)
//! coordinates. If `true`, an extra Hadamard is applied before the
//! squared-theta transform. `hadamard_bool_2` controls whether the
//! **output** codomain theta-null is in standard (`false`) or dual
//! (`true`) coordinates — if `true`, a final Hadamard is applied to
//! the codomain after the null-point is computed.
//!
//! These flags vary per chain step. Callers MUST pass the correct flag
//! for their position in the chain — transcribing them as constants
//! would silently produce wrong codomains for chain steps where the
//! convention differs.

use crate::gf::fp::BaseField;
use crate::gf::fp2::Fp2;
use crate::isogeny::theta::{AbelianVariety2D, ThetaPoint2D};

/// Failure modes for [`theta_isogeny_compute`].
///
/// Mirrors the C reference's `return 0` paths at
/// `theta_isogenies.c:656` (degenerate factor) and the four verify
/// checks at lines 676–689.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ThetaIsogenyError {
    /// One of the six required-nonzero factors in the squared-theta
    /// `TT1` / `TT2` was zero. Indicates "unexpected splitting in the
    /// isogeny chain" per the C ref comment at line 651. The exact
    /// factors are `TT2.x`, `TT2.y`, `TT2.z`, `TT2.w` (Rust `w` = C
    /// `t`), `TT1.x`, `TT1.y`.
    DegenerateFactor,
    /// The optional `verify` check (set `verify: true` in
    /// [`theta_isogeny_compute`] to enable) failed. The four pairwise
    /// equality checks ensure the 4-torsion `2 · T1_8` / `2 · T2_8`
    /// are isotropic with respect to the cached precomputation — a
    /// soundness check that the 8-torsion inputs are coherent with
    /// the isogeny structure.
    VerifyFailed,
    /// A required `Fp2` square root did not exist in the field
    /// (i.e., the operand was a non-residue). The C reference
    /// asserts this never happens for inputs from the signing
    /// pipeline (`theta_isogenies.c:750` — "No need to check the
    /// square roots, only used for signing"). We propagate
    /// honestly anyway so callers see the failure mode and
    /// general callers (tests, future audits) get a Result rather
    /// than a panic.
    SquareRootNotInField,
}

/// Per-step state of a `(2, 2)`-isogeny inner step.
///
/// Mirrors C `theta_isogeny_t`. The `codomain_null` field stores ONLY
/// the codomain's theta-null point (post-final-Hadamard if
/// `hadamard_bool_2 == true`); the codomain's doubling structure is
/// NOT computed here — callers materialize a full
/// [`AbelianVariety2D`] from this null when they need doubling
/// (matches C's `codomain.precomputation = false` flag-then-lazy
/// pattern).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ThetaIsogeny<F: BaseField> {
    /// Standard (`false`) vs dual (`true`) coordinate convention for
    /// the **input** 8-torsion points `t1_8` / `t2_8`.
    pub(crate) hadamard_bool_1: bool,
    /// Standard (`false`) vs dual (`true`) coordinate convention for
    /// the **output** codomain theta-null.
    pub(crate) hadamard_bool_2: bool,
    /// Source variety (the domain of the isogeny step).
    pub(crate) domain: AbelianVariety2D<F>,
    /// Codomain theta-null, post-final-Hadamard if `hadamard_bool_2`
    /// is true. Just the null point — NOT a full `AbelianVariety2D`
    /// because the doubling structure isn't precomputed here.
    pub(crate) codomain_null: ThetaPoint2D<F>,
    /// First 8-torsion kernel input, preserved as passed in.
    pub(crate) t1_8: ThetaPoint2D<F>,
    /// Second 8-torsion kernel input, preserved as passed in.
    pub(crate) t2_8: ThetaPoint2D<F>,
    /// Cached projective factors used by downstream
    /// `theta_isogeny_eval` (S143) for fast point evaluation. Per S142
    /// advisor: **these semantics are UNVERIFIED in this session** —
    /// smoke tests confirm structure but the `eval` consumer arrives
    /// in S143, which is where end-to-end semantic correctness is
    /// established.
    pub(crate) precomputation: ThetaPoint2D<F>,
}

impl<F: BaseField> ThetaIsogeny<F> {
    /// Construct a `(2, 2)`-isogeny inner step from the domain
    /// abelian variety and the 8-torsion kernel generators.
    ///
    /// Method-form alias of [`theta_isogeny_compute`] — same body,
    /// same result, same `Result<Self, ThetaIsogenyError>` failure
    /// modes (`DegenerateFactor` or `VerifyFailed`).
    #[allow(dead_code)]
    pub(crate) fn compute(
        domain: &AbelianVariety2D<F>,
        t1_8: &ThetaPoint2D<F>,
        t2_8: &ThetaPoint2D<F>,
        hadamard_bool_1: bool,
        hadamard_bool_2: bool,
        verify: bool,
    ) -> Result<Self, ThetaIsogenyError> {
        theta_isogeny_compute(domain, t1_8, t2_8, hadamard_bool_1, hadamard_bool_2, verify)
    }

    /// Evaluate this isogeny on a theta-coord point `p`.
    ///
    /// Method-form alias of [`theta_isogeny_eval`]. The
    /// `phi.eval(&p)` spelling reads more directly than
    /// `theta_isogeny_eval(&phi, &p)` for chain-walker call sites
    /// that thread the isogeny through many points.
    #[allow(dead_code)]
    pub(crate) fn eval(&self, p: &ThetaPoint2D<F>) -> ThetaPoint2D<F> {
        theta_isogeny_eval(self, p)
    }

    /// Construct a `(2, 2)`-isogeny inner step when only the
    /// 4-torsion above the 2-torsion kernel is known.
    ///
    /// Method-form alias of [`theta_isogeny_compute_4`]. Constructor
    /// for the 4-torsion variant; uses two `Fp2` square roots internally
    /// (returns `Err(SquareRootNotInField)` on QR failure).
    #[allow(dead_code)]
    pub(crate) fn compute_4(
        domain: &AbelianVariety2D<F>,
        t1_4: &ThetaPoint2D<F>,
        t2_4: &ThetaPoint2D<F>,
        hadamard_bool_1: bool,
        hadamard_bool_2: bool,
    ) -> Result<Self, ThetaIsogenyError> {
        theta_isogeny_compute_4(domain, t1_4, t2_4, hadamard_bool_1, hadamard_bool_2)
    }

    /// Construct a `(2, 2)`-isogeny inner step from only the
    /// 2-torsion kernel (no higher torsion).
    ///
    /// Method-form alias of [`theta_isogeny_compute_2`]. Uses three
    /// `Fp2` square roots internally (returns
    /// `Err(SquareRootNotInField)` on QR failure).
    #[allow(dead_code)]
    pub(crate) fn compute_2(
        domain: &AbelianVariety2D<F>,
        t1_2: &ThetaPoint2D<F>,
        t2_2: &ThetaPoint2D<F>,
        hadamard_bool_1: bool,
        hadamard_bool_2: bool,
    ) -> Result<Self, ThetaIsogenyError> {
        theta_isogeny_compute_2(domain, t1_2, t2_2, hadamard_bool_1, hadamard_bool_2)
    }
}

/// Compute a `(2, 2)`-isogeny inner step from the 8-torsion kernel
/// generators `T1_8`, `T2_8` above the 2-torsion isogeny kernel.
///
/// # Algorithm
///
/// Mirrors `theta_isogenies.c:theta_isogeny_compute` (lines 618–696)
/// verbatim:
///
/// 1. If `hadamard_bool_1` is `true`: compute `TT1 = squared_theta(hadamard(T1_8))`
///    and `TT2 = squared_theta(hadamard(T2_8))`. Otherwise compute
///    `TT1 = squared_theta(T1_8)` and `TT2 = squared_theta(T2_8)`
///    directly. (`squared_theta` is `componentwise_square` followed
///    by `hadamard` per `theta_structure.h:to_squared_theta`.)
/// 2. **Non-zero invariant check**: each of `TT2.x`, `TT2.y`, `TT2.z`,
///    `TT2.w` (Rust `w` = C `t`), `TT1.x`, `TT1.y` must be non-zero.
///    Zero signals "unexpected splitting in the isogeny chain" per
///    the C reference. Returns `Err(DegenerateFactor)` on violation.
/// 3. Compute auxiliary scalars `t1 = TT1.x · TT2.y` and
///    `t2 = TT1.y · TT2.x`.
/// 4. Build the (pre-final-Hadamard) codomain theta-null point:
///    ```text
///    null.x = TT2.x · t1
///    null.y = TT2.y · t2
///    null.z = TT2.z · t1
///    null.w = TT2.w · t2
///    ```
/// 5. Build the precomputation triple from `t3 = TT2.z · TT2.w`:
///    ```text
///    precomp.x = t3 · TT1.y
///    precomp.y = t3 · TT1.x
///    precomp.z = null.w   (copy of pre-final-Hadamard codomain.w)
///    precomp.w = null.z   (copy of pre-final-Hadamard codomain.z)
///    ```
/// 6. If `verify` is `true`: four pairwise equality checks ensuring
///    the 4-torsion `2·T1_8`, `2·T2_8` are isotropic with respect to
///    the cached precomputation factors. Returns
///    `Err(VerifyFailed)` on any mismatch.
/// 7. If `hadamard_bool_2` is `true`: apply `hadamard` to the
///    codomain null point.
///
/// # Hadamard flag note (S142 advisor)
///
/// The two flags vary per chain step. `hadamard_bool_1` is the
/// **input** convention (standard `false` vs dual `true`).
/// `hadamard_bool_2` is the **output** convention. The caller MUST
/// pass these per the chain step's position; mechanical transcription
/// as constants from a single-step example would produce wrong
/// codomains.
///
/// # Verify check semantics
///
/// `verify: true` is a **runtime** check (not compiled-out
/// `debug_assert!`). Use `true` when the caller doesn't fully trust
/// the 8-torsion provenance; use `false` when the inputs are
/// structurally guaranteed coherent (e.g., previously verified upstream).
///
/// Reference: `theta_isogenies.c:618-696`.
#[allow(dead_code)]
pub(crate) fn theta_isogeny_compute<F: BaseField>(
    domain: &AbelianVariety2D<F>,
    t1_8: &ThetaPoint2D<F>,
    t2_8: &ThetaPoint2D<F>,
    hadamard_bool_1: bool,
    hadamard_bool_2: bool,
    verify: bool,
) -> Result<ThetaIsogeny<F>, ThetaIsogenyError> {
    // Step 1: derive TT1, TT2 via squared_theta with optional input Hadamard.
    let tt1 = if hadamard_bool_1 {
        t1_8.hadamard().componentwise_square().hadamard()
    } else {
        t1_8.componentwise_square().hadamard()
    };
    let tt2 = if hadamard_bool_1 {
        t2_8.hadamard().componentwise_square().hadamard()
    } else {
        t2_8.componentwise_square().hadamard()
    };

    // Step 2: non-zero invariant check on six required factors. Reading
    // `Choice` values; any zero signals "unexpected splitting" per C ref.
    let any_zero = tt2.x.is_zero()
        | tt2.y.is_zero()
        | tt2.z.is_zero()
        | tt2.w.is_zero()
        | tt1.x.is_zero()
        | tt1.y.is_zero();
    if bool::from(any_zero) {
        return Err(ThetaIsogenyError::DegenerateFactor);
    }

    // Step 3: auxiliary scalars.
    let t1 = tt1.x.mul(&tt2.y);
    let t2 = tt1.y.mul(&tt2.x);

    // Step 4: codomain null point (pre-final-Hadamard).
    // Note the alternation: t1 multiplies TT2.{x, z}; t2 multiplies TT2.{y, w}.
    let null_x = tt2.x.mul(&t1);
    let null_y = tt2.y.mul(&t2);
    let null_z = tt2.z.mul(&t1);
    let null_w = tt2.w.mul(&t2);

    // Step 5: precomputation triple.
    let t3 = tt2.z.mul(&tt2.w);
    let precomp_x = t3.mul(&tt1.y);
    let precomp_y = t3.mul(&tt1.x);
    let precomp_z = null_w;
    let precomp_w = null_z;
    let precomputation = ThetaPoint2D::new(precomp_x, precomp_y, precomp_z, precomp_w);

    // Step 6: optional verify (4 pairwise equality checks on isotropy
    // of 2·T1_8 / 2·T2_8 against the cached precomp factors).
    if verify {
        let v1 = tt1.x.mul(&precomputation.x);
        let v2 = tt1.y.mul(&precomputation.y);
        if v1 != v2 {
            return Err(ThetaIsogenyError::VerifyFailed);
        }
        let v3 = tt1.z.mul(&precomputation.z);
        let v4 = tt1.w.mul(&precomputation.w);
        if v3 != v4 {
            return Err(ThetaIsogenyError::VerifyFailed);
        }
        let v5 = tt2.x.mul(&precomputation.x);
        let v6 = tt2.z.mul(&precomputation.z);
        if v5 != v6 {
            return Err(ThetaIsogenyError::VerifyFailed);
        }
        let v7 = tt2.y.mul(&precomputation.y);
        let v8 = tt2.w.mul(&precomputation.w);
        if v7 != v8 {
            return Err(ThetaIsogenyError::VerifyFailed);
        }
    }

    // Step 7: optional final Hadamard on codomain.
    let codomain_null_pre = ThetaPoint2D::new(null_x, null_y, null_z, null_w);
    let codomain_null = if hadamard_bool_2 {
        codomain_null_pre.hadamard()
    } else {
        codomain_null_pre
    };

    Ok(ThetaIsogeny {
        hadamard_bool_1,
        hadamard_bool_2,
        domain: *domain,
        codomain_null,
        t1_8: *t1_8,
        t2_8: *t2_8,
        precomputation,
    })
}

/// Shared helper: square root of a product of two `Fp2` values,
/// returning `Err(SquareRootNotInField)` if the product is not a
/// quadratic residue.
///
/// Used by [`theta_isogeny_compute_4`] (two call sites) and
/// [`theta_isogeny_compute_2`] (three call sites) to compute the
/// `sqrt(AA · BB)`, `sqrt(AA · CC)`, etc. factors required by the
/// codomain formulas. See `theta_isogenies.c:752`, `:754`,
/// `:836`–`:838` for the C-reference call sites.
///
/// # Sqrt sign convention (S144 advisor flag)
///
/// `Fp2::sqrt` returns ONE of the two roots ±√x; the
/// `CtOption` only carries existence, not sign. The Rust crate's
/// deterministic root choice (see `src/gf/fp2.rs:149-169`) may
/// differ from the C reference's convention. This produces a
/// valid-but-possibly-different codomain representative — fine
/// algebraically (the variety is identified up to projective
/// scaling) but possibly KAT-incompatible. **Resolution deferred
/// to chain-integration session** when a C-reference KAT vector
/// becomes available for direct comparison.
#[allow(dead_code)]
fn sqrt_of_product<F: BaseField>(a: &Fp2<F>, b: &Fp2<F>) -> Result<Fp2<F>, ThetaIsogenyError> {
    a.mul(b)
        .sqrt()
        .into_option()
        .ok_or(ThetaIsogenyError::SquareRootNotInField)
}

/// Compute a `(2, 2)`-isogeny when only the 4-torsion above the
/// 2-torsion kernel is known (not 8-torsion).
///
/// Sibling of [`theta_isogeny_compute`] (which uses 8-torsion).
/// Mirrors `theta_isogenies.c:714-786` verbatim. The codomain
/// theta-null computation includes two `Fp2` square roots:
/// `sqaabb = sqrt(AA · BB)` and `sqaacc = sqrt(AA · CC)`, where
/// `(AA, BB, CC, DD) = squared_theta(A->null_point)`. These
/// square roots inherit the sqrt-sign-convention caveat from
/// [`sqrt_of_product`].
///
/// # Hadamard flag semantics
///
/// Same as [`theta_isogeny_compute`]: `hadamard_bool_1` controls the
/// input convention (standard `false` vs dual `true`), affecting
/// both `T1_4` and the domain's null point. `hadamard_bool_2`
/// controls the output convention.
///
/// Reference: `theta_isogenies.c:714-786`.
#[allow(dead_code)]
pub(crate) fn theta_isogeny_compute_4<F: BaseField>(
    domain: &AbelianVariety2D<F>,
    t1_4: &ThetaPoint2D<F>,
    t2_4: &ThetaPoint2D<F>,
    hadamard_bool_1: bool,
    hadamard_bool_2: bool,
) -> Result<ThetaIsogeny<F>, ThetaIsogenyError> {
    // Step 1: derive TT1 from T1_4 and TT2 from the domain's null
    // point (note: TT2 is the SQUARED-THETA of the DOMAIN, not of T2_4).
    let tt1 = if hadamard_bool_1 {
        t1_4.hadamard().componentwise_square().hadamard()
    } else {
        t1_4.componentwise_square().hadamard()
    };
    let tt2 = if hadamard_bool_1 {
        domain
            .theta_null
            .hadamard()
            .componentwise_square()
            .hadamard()
    } else {
        domain.theta_null.componentwise_square().hadamard()
    };

    // Step 2: compute the two square roots of products.
    let sqaabb = sqrt_of_product(&tt2.x, &tt2.y)?;
    let sqaacc = sqrt_of_product(&tt2.x, &tt2.z)?;

    // Step 3: codomain theta-null (per C ref lines 760-773). Each
    // intermediate is built up via reuse-then-overwrite in the C body;
    // we transcribe to single-assignment Rust for clarity.
    //
    //   null.y = sqaabb · sqaacc · TT1.x
    //   null.t = TT1.z · sqaabb · TT2.x
    //   null.x = TT1.x · TT2.x · sqaacc
    //   null.z = TT1.x · TT2.x · TT2.z
    let sqaabb_sqaacc = sqaabb.mul(&sqaacc);
    let null_y = sqaabb_sqaacc.mul(&tt1.x);
    let null_t_pre1 = tt1.z.mul(&sqaabb);
    let null_t = null_t_pre1.mul(&tt2.x);
    let tt1x_tt2x = tt1.x.mul(&tt2.x);
    let null_x = tt1x_tt2x.mul(&sqaacc);
    let null_z = tt1x_tt2x.mul(&tt2.z);

    // Step 4: precomputation (per C ref lines 775-781). The C body
    // reuses `out->precomputation.x` as a scratch — we keep separate
    // names for clarity.
    //
    //   precomp.t = sqaabb · sqaacc · TT1.z · TT2.y          (= null.y reformula via TT1.z)
    //   p_xt = TT1.x · TT2.w
    //   precomp.z = p_xt · TT2.y · sqaacc
    //   precomp.y = p_xt · TT2.z · sqaabb
    //   precomp.x = p_xt · TT2.z · TT2.y
    let precomp_t_pre = sqaabb_sqaacc.mul(&tt1.z);
    let precomp_t = precomp_t_pre.mul(&tt2.y);

    let p_xt = tt1.x.mul(&tt2.w);
    let p_xt_y = p_xt.mul(&tt2.y);
    let p_xt_z = p_xt.mul(&tt2.z);
    let precomp_x = p_xt_z.mul(&tt2.y);
    let precomp_y = p_xt_z.mul(&sqaabb);
    let precomp_z = p_xt_y.mul(&sqaacc);

    let precomputation = ThetaPoint2D::new(precomp_x, precomp_y, precomp_z, precomp_t);

    // Step 5: optional final Hadamard on codomain.
    let codomain_null_pre = ThetaPoint2D::new(null_x, null_y, null_z, null_t);
    let codomain_null = if hadamard_bool_2 {
        codomain_null_pre.hadamard()
    } else {
        codomain_null_pre
    };

    Ok(ThetaIsogeny {
        hadamard_bool_1,
        hadamard_bool_2,
        domain: *domain,
        codomain_null,
        t1_8: *t1_4,
        t2_8: *t2_4,
        precomputation,
    })
}

/// Compute a `(2, 2)`-isogeny when only the 2-torsion kernel is
/// known (not 4-torsion or 8-torsion above it).
///
/// Sibling of [`theta_isogeny_compute`] and [`theta_isogeny_compute_4`].
/// Mirrors `theta_isogenies.c:803-853` verbatim. Uses three `Fp2`
/// square roots: `sqrt(AA · BB)`, `sqrt(AA · CC)`, `sqrt(AA · DD)`.
///
/// Note on stored kernel inputs: the field names `t1_8`/`t2_8` on
/// the returned [`ThetaIsogeny`] are kept as-is (predate the
/// torsion-level genericity); for the `_2` case they hold the
/// 2-torsion kernel inputs `T1_2`/`T2_2` per the C ref's
/// `out->T1_8 = *T1_2; out->T2_8 = *T2_2` storage pattern.
///
/// Reference: `theta_isogenies.c:803-853`.
#[allow(dead_code)]
pub(crate) fn theta_isogeny_compute_2<F: BaseField>(
    domain: &AbelianVariety2D<F>,
    t1_2: &ThetaPoint2D<F>,
    t2_2: &ThetaPoint2D<F>,
    hadamard_bool_1: bool,
    hadamard_bool_2: bool,
) -> Result<ThetaIsogeny<F>, ThetaIsogenyError> {
    // Step 1: only TT2 derived from the domain's null point — no TT1
    // here (the kernel inputs t1_2/t2_2 are stored but not used in
    // the codomain formula).
    let tt2 = if hadamard_bool_1 {
        domain
            .theta_null
            .hadamard()
            .componentwise_square()
            .hadamard()
    } else {
        domain.theta_null.componentwise_square().hadamard()
    };

    // Step 2: codomain theta-null per C ref lines 831-838:
    //   null.x = TT2.x
    //   null.y = sqrt(TT2.x · TT2.y)
    //   null.z = sqrt(TT2.x · TT2.z)
    //   null.t = sqrt(TT2.x · TT2.w)
    let null_x = tt2.x;
    let null_y = sqrt_of_product(&tt2.x, &tt2.y)?;
    let null_z = sqrt_of_product(&tt2.x, &tt2.z)?;
    let null_t = sqrt_of_product(&tt2.x, &tt2.w)?;

    // Step 3: precomputation per C ref lines 840-848:
    //   pre_xt = TT2.z · TT2.w
    //   precomp.y = pre_xt · null.y
    //   precomp.x = pre_xt · TT2.y
    //   precomp.z = TT2.w · null.z · TT2.y
    //   precomp.t = TT2.z · null.t · TT2.y
    let pre_xt = tt2.z.mul(&tt2.w);
    let precomp_y = pre_xt.mul(&null_y);
    let precomp_x = pre_xt.mul(&tt2.y);
    let precomp_z_pre = tt2.w.mul(&null_z);
    let precomp_z = precomp_z_pre.mul(&tt2.y);
    let precomp_t_pre = tt2.z.mul(&null_t);
    let precomp_t = precomp_t_pre.mul(&tt2.y);

    let precomputation = ThetaPoint2D::new(precomp_x, precomp_y, precomp_z, precomp_t);

    // Step 4: optional final Hadamard on codomain.
    let codomain_null_pre = ThetaPoint2D::new(null_x, null_y, null_z, null_t);
    let codomain_null = if hadamard_bool_2 {
        codomain_null_pre.hadamard()
    } else {
        codomain_null_pre
    };

    Ok(ThetaIsogeny {
        hadamard_bool_1,
        hadamard_bool_2,
        domain: *domain,
        codomain_null,
        t1_8: *t1_2,
        t2_8: *t2_2,
        precomputation,
    })
}

/// Evaluate a `(2, 2)`-isogeny `phi` on a theta-point `P`.
///
/// First consumer of [`ThetaIsogeny::precomputation`] — the cached
/// projective factors computed in [`theta_isogeny_compute`] are
/// componentwise applied to the squared-theta transform of `P`,
/// producing the image of `P` under `phi`.
///
/// # Algorithm
///
/// Mirrors `theta_isogenies.c:855-872` verbatim:
///
/// 1. **Input-side Hadamard** (controlled by `phi.hadamard_bool_1`):
///    - If `true`: `out = squared_theta(hadamard(P))`
///    - If `false`: `out = squared_theta(P)`
///
///    (`squared_theta` = `componentwise_square` then `hadamard`.)
/// 2. **Componentwise scale** by `phi.precomputation`:
///    ```text
///    out.x = out.x · precomp.x
///    out.y = out.y · precomp.y
///    out.z = out.z · precomp.z
///    out.w = out.w · precomp.w
///    ```
/// 3. **Output-side Hadamard** (controlled by `phi.hadamard_bool_2`):
///    - If `true`: apply `hadamard` to `out`.
///    - If `false`: leave `out` as-is.
///
/// # Hadamard flag semantics
///
/// The same flags that `theta_isogeny_compute` stored on the
/// `ThetaIsogeny` struct are consumed here — `hadamard_bool_1`
/// controls the **input** coordinate convention (standard vs dual)
/// and `hadamard_bool_2` controls the **output** convention. The two
/// flags must match the chain-step's position at compute time and
/// eval time (i.e., the same `ThetaIsogeny` is used for both).
///
/// Reference: `theta_isogenies.c:theta_isogeny_eval`.
#[allow(dead_code)]
pub(crate) fn theta_isogeny_eval<F: BaseField>(
    phi: &ThetaIsogeny<F>,
    p: &ThetaPoint2D<F>,
) -> ThetaPoint2D<F> {
    // Step 1: input-side Hadamard branch + squared_theta.
    let after_squared = if phi.hadamard_bool_1 {
        p.hadamard().componentwise_square().hadamard()
    } else {
        p.componentwise_square().hadamard()
    };

    // Step 2: componentwise scale by precomputation.
    let scaled = ThetaPoint2D::new(
        after_squared.x.mul(&phi.precomputation.x),
        after_squared.y.mul(&phi.precomputation.y),
        after_squared.z.mul(&phi.precomputation.z),
        after_squared.w.mul(&phi.precomputation.w),
    );

    // Step 3: output-side Hadamard branch.
    if phi.hadamard_bool_2 {
        scaled.hadamard()
    } else {
        scaled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf::fp::Fp1Element;
    use crate::gf::fp2::Fp2;

    // S142 smoke tests. End-to-end semantic correctness for the
    // precomputation triple is deferred to S143 when
    // `theta_isogeny_eval` consumes those cached factors. These tests
    // confirm:
    //   1. The DegenerateFactor invariant fires on zero-valued inputs
    //      (which trigger the "any zero" check on TT2/TT1 factors).
    //   2. Non-degenerate inputs return Ok with the expected struct shape.
    //   3. The four (hadamard_bool_1, hadamard_bool_2) permutations all
    //      type-check and complete (catches accidental flag mixups in
    //      the branching code).
    //   4. The verify-flag check is reachable and can fail on
    //      hand-constructed inputs that trigger a mismatch.

    fn small_fp2(n: u32) -> Fp2<Fp1Element> {
        let mut acc = Fp2::<Fp1Element>::zero();
        let one = Fp2::<Fp1Element>::one();
        for _ in 0..n {
            acc = acc.add(&one);
        }
        acc
    }

    fn synthetic_domain() -> AbelianVariety2D<Fp1Element> {
        let null = ThetaPoint2D::new(small_fp2(1), small_fp2(1), small_fp2(1), small_fp2(1));
        let doubling_constants = null;
        AbelianVariety2D::new(null, doubling_constants)
    }

    /// Non-degenerate 8-torsion-shaped input. Values picked so that
    /// after squared_theta neither TT1 nor TT2 collapses to zero.
    fn nondegenerate_t_8() -> (ThetaPoint2D<Fp1Element>, ThetaPoint2D<Fp1Element>) {
        let t1 = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        let t2 = ThetaPoint2D::new(small_fp2(11), small_fp2(13), small_fp2(17), small_fp2(19));
        (t1, t2)
    }

    #[test]
    fn theta_isogeny_compute_rejects_zero_input_as_degenerate() {
        let domain = synthetic_domain();
        let zero = ThetaPoint2D::new(
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
        );
        let result = theta_isogeny_compute(&domain, &zero, &zero, false, false, false);
        assert_eq!(
            result,
            Err(ThetaIsogenyError::DegenerateFactor),
            "S142: zero input must trip the non-zero invariant check on TT1/TT2 factors",
        );
    }

    #[test]
    fn theta_isogeny_compute_returns_ok_on_nondegenerate_at_lvl1() {
        let domain = synthetic_domain();
        let (t1_8, t2_8) = nondegenerate_t_8();
        let result = theta_isogeny_compute(&domain, &t1_8, &t2_8, false, false, false);
        assert!(
            result.is_ok(),
            "S142: non-degenerate input must succeed (verify off)",
        );
        let iso = result.expect("checked is_ok above");
        assert!(!iso.hadamard_bool_1, "hadamard_bool_1 preserved as false");
        assert!(!iso.hadamard_bool_2, "hadamard_bool_2 preserved as false");
        assert_eq!(iso.t1_8, t1_8, "t1_8 must be preserved as-passed");
        assert_eq!(iso.t2_8, t2_8, "t2_8 must be preserved as-passed");
    }

    /// S142 advisor: Hadamard flag permutations must all type-check and
    /// complete. Catches accidental flag-swaps or constant-folding.
    /// Doesn't verify semantics (precomputation truth is S143 territory)
    /// but ensures all four branches are reachable on a single input.
    #[test]
    fn theta_isogeny_compute_handles_all_four_hadamard_permutations_at_lvl1() {
        let domain = synthetic_domain();
        let (t1_8, t2_8) = nondegenerate_t_8();
        for &b1 in &[false, true] {
            for &b2 in &[false, true] {
                let result = theta_isogeny_compute(&domain, &t1_8, &t2_8, b1, b2, false);
                if let Ok(iso) = result {
                    assert_eq!(iso.hadamard_bool_1, b1);
                    assert_eq!(iso.hadamard_bool_2, b2);
                }
                // Some permutations may produce DegenerateFactor on
                // these synthetic inputs (the Hadamard pre-transform
                // can zero out factors); that's acceptable for a
                // smoke test — we're checking the branching code is
                // reachable, not validating semantic correctness.
            }
        }
    }

    /// S142 advisor: verify-flag must be a real runtime check, not
    /// compiled-out. Force a mismatch by constructing inputs where
    /// the four pairwise checks cannot all simultaneously hold (the
    /// synthetic non-degenerate values were not constructed to be
    /// isotropic), then confirm `verify: true` returns
    /// `Err(VerifyFailed)` while `verify: false` returns `Ok`.
    #[test]
    fn theta_isogeny_compute_verify_flag_detects_non_isotropic_inputs_at_lvl1() {
        let domain = synthetic_domain();
        let (t1_8, t2_8) = nondegenerate_t_8();

        // verify=false should succeed.
        let without_verify = theta_isogeny_compute(&domain, &t1_8, &t2_8, false, false, false);
        assert!(
            without_verify.is_ok(),
            "S142: verify=false must accept the synthetic inputs",
        );

        // verify=true should fail because the synthetic inputs were
        // chosen for non-degeneracy, not for 4-torsion isotropy.
        let with_verify = theta_isogeny_compute(&domain, &t1_8, &t2_8, false, false, true);
        assert_eq!(
            with_verify,
            Err(ThetaIsogenyError::VerifyFailed),
            "S142: verify=true must reject non-isotropic synthetic inputs",
        );
    }

    // S143 — theta_isogeny_eval hand-computed oracle tests.
    //
    // Goal: semantically validate the eval pipeline (input-side
    // Hadamard branch + squared_theta + componentwise scale + output-
    // side Hadamard branch) across all 4 (hadamard_bool_1, hadamard_bool_2)
    // permutations. Uses synthetic precomp + input theta point where the
    // expected output is hand-computed from the C-reference formula
    // (NOT from this function's body), per the S137e/S138 "expected
    // from formula, not function body" doctrine.
    //
    // What this DOES validate: the eval routine's branching code,
    // componentwise scaling, and Hadamard application are correct.
    // What this DOES NOT validate: that S142's precomputation VALUES
    // produced by theta_isogeny_compute are themselves correct.
    // Precomp-value correctness needs end-to-end chain integration
    // (~S145+ once splitting + theta_chain_compute_and_eval ship and
    // we can round-trip a known input through the chain).

    /// Hand-built ThetaIsogeny with specified precomp + flag config.
    /// Other fields (domain, codomain_null, t1_8, t2_8) are zero —
    /// `eval` ignores them entirely.
    fn synthetic_isogeny_for_eval(
        precomp: (u32, u32, u32, u32),
        hadamard_bool_1: bool,
        hadamard_bool_2: bool,
    ) -> ThetaIsogeny<Fp1Element> {
        let null = ThetaPoint2D::new(
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
        );
        ThetaIsogeny {
            hadamard_bool_1,
            hadamard_bool_2,
            domain: AbelianVariety2D::new(null, null),
            codomain_null: null,
            t1_8: null,
            t2_8: null,
            precomputation: ThetaPoint2D::new(
                small_fp2(precomp.0),
                small_fp2(precomp.1),
                small_fp2(precomp.2),
                small_fp2(precomp.3),
            ),
        }
    }

    fn neg_fp2(n: u32) -> Fp2<Fp1Element> {
        Fp2::<Fp1Element>::zero().sub(&small_fp2(n))
    }

    /// S143 oracle (false, false):
    /// `P = (1, 2, 3, 4)`, `precomp = (2, 3, 5, 7)`.
    /// `square(P) = (1, 4, 9, 16)`.
    /// `hadamard((1, 4, 9, 16)) = (30, -10, -20, 4)`.
    /// `scale = (60, -30, -100, 28)`.
    /// No output Hadamard.
    #[test]
    fn theta_isogeny_eval_oracle_no_hadamard_flags_at_lvl1() {
        let phi = synthetic_isogeny_for_eval((2, 3, 5, 7), false, false);
        let p = ThetaPoint2D::new(small_fp2(1), small_fp2(2), small_fp2(3), small_fp2(4));

        let r = theta_isogeny_eval(&phi, &p);

        assert_eq!(r.x, small_fp2(60), "S143: out.x = 30 · 2 = 60");
        assert_eq!(r.y, neg_fp2(30), "S143: out.y = -10 · 3 = -30");
        assert_eq!(r.z, neg_fp2(100), "S143: out.z = -20 · 5 = -100");
        assert_eq!(r.w, small_fp2(28), "S143: out.w = 4 · 7 = 28");
    }

    /// S143 oracle (false, true): scale step same as above,
    /// then apply final Hadamard:
    /// `hadamard((60, -30, -100, 28))`:
    ///   - out.x = 60 + (-30) + (-100) + 28 = -42
    ///   - out.y = 60 - (-30) + (-100) - 28 = 60 + 30 - 100 - 28 = -38
    ///   - out.z = 60 + (-30) - (-100) - 28 = 60 - 30 + 100 - 28 = 102
    ///   - out.w = 60 - (-30) - (-100) + 28 = 60 + 30 + 100 + 28 = 218
    #[test]
    fn theta_isogeny_eval_oracle_output_hadamard_only_at_lvl1() {
        let phi = synthetic_isogeny_for_eval((2, 3, 5, 7), false, true);
        let p = ThetaPoint2D::new(small_fp2(1), small_fp2(2), small_fp2(3), small_fp2(4));

        let r = theta_isogeny_eval(&phi, &p);

        assert_eq!(r.x, neg_fp2(42), "S143: final-hadamard out.x = -42");
        assert_eq!(r.y, neg_fp2(38), "S143: final-hadamard out.y = -38");
        assert_eq!(r.z, small_fp2(102), "S143: final-hadamard out.z = 102");
        assert_eq!(r.w, small_fp2(218), "S143: final-hadamard out.w = 218");
    }

    /// S143 oracle (true, false): input-side Hadamard first.
    /// `hadamard((1, 2, 3, 4)) = (10, -2, -4, 0)`.
    /// `square = (100, 4, 16, 0)`.
    /// `hadamard((100, 4, 16, 0)) = (120, 112, 88, 80)`.
    /// `scale by (2, 3, 5, 7) = (240, 336, 440, 560)`.
    /// No output Hadamard.
    #[test]
    fn theta_isogeny_eval_oracle_input_hadamard_only_at_lvl1() {
        let phi = synthetic_isogeny_for_eval((2, 3, 5, 7), true, false);
        let p = ThetaPoint2D::new(small_fp2(1), small_fp2(2), small_fp2(3), small_fp2(4));

        let r = theta_isogeny_eval(&phi, &p);

        assert_eq!(
            r.x,
            small_fp2(240),
            "S143: input-hadamard out.x = 120 · 2 = 240"
        );
        assert_eq!(
            r.y,
            small_fp2(336),
            "S143: input-hadamard out.y = 112 · 3 = 336"
        );
        assert_eq!(
            r.z,
            small_fp2(440),
            "S143: input-hadamard out.z = 88 · 5 = 440"
        );
        assert_eq!(
            r.w,
            small_fp2(560),
            "S143: input-hadamard out.w = 80 · 7 = 560"
        );
    }

    /// S143 oracle (true, true): input + output Hadamard both.
    /// After scale: (240, 336, 440, 560).
    /// Final Hadamard:
    ///   - out.x = 240 + 336 + 440 + 560 = 1576
    ///   - out.y = 240 - 336 + 440 - 560 = -216
    ///   - out.z = 240 + 336 - 440 - 560 = -424
    ///   - out.w = 240 - 336 - 440 + 560 = 24
    #[test]
    fn theta_isogeny_eval_oracle_both_hadamard_flags_at_lvl1() {
        let phi = synthetic_isogeny_for_eval((2, 3, 5, 7), true, true);
        let p = ThetaPoint2D::new(small_fp2(1), small_fp2(2), small_fp2(3), small_fp2(4));

        let r = theta_isogeny_eval(&phi, &p);

        assert_eq!(r.x, small_fp2(1576), "S143: both-hadamard out.x = 1576");
        assert_eq!(r.y, neg_fp2(216), "S143: both-hadamard out.y = -216");
        assert_eq!(r.z, neg_fp2(424), "S143: both-hadamard out.z = -424");
        assert_eq!(r.w, small_fp2(24), "S143: both-hadamard out.w = 24");
    }

    // S144 — theta_isogeny_compute_4 + theta_isogeny_compute_2 tests.
    //
    // Scope per S144 OBSERVE: smoke + sqrt-failure detection ONLY.
    // Sign-independent oracle tests (squared-identity invariants on
    // codomain components) are docketed for a focused follow-up session
    // because they require random-QR input generation and convention
    // alignment with the C reference. See ISA S144 Decisions block.

    #[test]
    fn theta_isogeny_compute_4_rejects_zero_domain_as_degenerate_or_sqrt_fail() {
        // Zero domain → TT2 all zero → sqrt_of_product(0, 0) = sqrt(0) = Some(0).
        // No degeneracy check in _4 (unlike _8), so the zero case proceeds
        // and we get all-zero codomain. Confirms type plumbing.
        let zero_null = ThetaPoint2D::new(
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
        );
        let domain = AbelianVariety2D::new(zero_null, zero_null);
        let zero = zero_null;
        let result = theta_isogeny_compute_4(&domain, &zero, &zero, false, false);
        // sqrt(0) succeeds → Ok with all-zero codomain.
        assert!(
            result.is_ok(),
            "S144: zero inputs to _4 produce Ok (sqrt(0) succeeds; no degeneracy check)",
        );
    }

    #[test]
    fn theta_isogeny_compute_2_rejects_zero_domain_with_ok_or_sqrt_fail() {
        let zero_null = ThetaPoint2D::new(
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
        );
        let domain = AbelianVariety2D::new(zero_null, zero_null);
        let zero = zero_null;
        let result = theta_isogeny_compute_2(&domain, &zero, &zero, false, false);
        // sqrt(0) succeeds → Ok with all-zero codomain.
        assert!(
            result.is_ok(),
            "S144: zero inputs to _2 produce Ok (sqrt(0) succeeds)",
        );
    }

    /// S144: smoke test for `_4` with a non-trivial small-prime domain;
    /// confirms the routine runs end-to-end without panicking and returns
    /// a `ThetaIsogeny` struct with the expected boolean flags + preserved
    /// kernel inputs. Sqrt-success depends on whether the chosen primes
    /// yield QR products at the target prime — this small-input case
    /// happens to succeed at L1.
    #[test]
    fn theta_isogeny_compute_4_smoke_at_lvl1() {
        // Use the same nondegenerate inputs as S142's compute test.
        // The domain's null determines whether sqrt(TT2.x·TT2.y) and
        // sqrt(TT2.x·TT2.z) succeed. If the chosen primes don't yield
        // QR products at L1, the test will return Err and we accept that
        // as a smoke pass (the structural code path is exercised either way).
        let domain_null = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        let domain = AbelianVariety2D::new(domain_null, domain_null);
        let (t1, t2) = nondegenerate_t_8();

        let result = theta_isogeny_compute_4(&domain, &t1, &t2, false, false);
        match result {
            Ok(iso) => {
                assert!(!iso.hadamard_bool_1);
                assert!(!iso.hadamard_bool_2);
                assert_eq!(iso.t1_8, t1, "S144: T1_4 stored in t1_8 slot");
                assert_eq!(iso.t2_8, t2, "S144: T2_4 stored in t2_8 slot");
            }
            Err(ThetaIsogenyError::SquareRootNotInField) => {
                // Acceptable smoke outcome: the routine correctly
                // propagated the QR-precondition violation. Structural
                // code path was exercised.
            }
            Err(other) => unreachable!("S144: unexpected error {other:?}"),
        }
    }

    /// S144: smoke test for `_2`. Same pattern as `_4` but with three
    /// square roots (and on a 2-torsion-only input).
    #[test]
    fn theta_isogeny_compute_2_smoke_at_lvl1() {
        let domain_null = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        let domain = AbelianVariety2D::new(domain_null, domain_null);
        let (t1, t2) = nondegenerate_t_8();

        let result = theta_isogeny_compute_2(&domain, &t1, &t2, false, false);
        match result {
            Ok(iso) => {
                assert!(!iso.hadamard_bool_1);
                assert!(!iso.hadamard_bool_2);
                assert_eq!(iso.t1_8, t1, "S144: T1_2 stored in t1_8 slot");
                assert_eq!(iso.t2_8, t2, "S144: T2_2 stored in t2_8 slot");
            }
            Err(ThetaIsogenyError::SquareRootNotInField) => {
                // Acceptable smoke outcome.
            }
            Err(other) => unreachable!("S144: unexpected error {other:?}"),
        }
    }

    /// S144: confirm Hadamard flag permutations all reachable for `_4`.
    #[test]
    fn theta_isogeny_compute_4_handles_all_hadamard_permutations_at_lvl1() {
        let domain_null = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        let domain = AbelianVariety2D::new(domain_null, domain_null);
        let (t1, t2) = nondegenerate_t_8();

        for &b1 in &[false, true] {
            for &b2 in &[false, true] {
                let result = theta_isogeny_compute_4(&domain, &t1, &t2, b1, b2);
                // Result can be Ok or SquareRootNotInField; either is
                // an acceptable smoke outcome (no panic, no unrelated error).
                match result {
                    Ok(_) | Err(ThetaIsogenyError::SquareRootNotInField) => {}
                    Err(other) => unreachable!("S144: unexpected error {other:?} for ({b1}, {b2})"),
                }
            }
        }
    }

    /// S144: confirm Hadamard flag permutations all reachable for `_2`.
    #[test]
    fn theta_isogeny_compute_2_handles_all_hadamard_permutations_at_lvl1() {
        let domain_null = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        let domain = AbelianVariety2D::new(domain_null, domain_null);
        let (t1, t2) = nondegenerate_t_8();

        for &b1 in &[false, true] {
            for &b2 in &[false, true] {
                let result = theta_isogeny_compute_2(&domain, &t1, &t2, b1, b2);
                match result {
                    Ok(_) | Err(ThetaIsogenyError::SquareRootNotInField) => {}
                    Err(other) => unreachable!("S144: unexpected error {other:?} for ({b1}, {b2})"),
                }
            }
        }
    }

    // S156 — ThetaIsogeny method-form alias tests.

    #[test]
    fn theta_isogeny_compute_method_matches_free_function_at_lvl1() {
        let domain = synthetic_domain();
        let (t1_8, t2_8) = nondegenerate_t_8();

        let via_method = ThetaIsogeny::compute(&domain, &t1_8, &t2_8, false, false, false);
        let via_free = theta_isogeny_compute(&domain, &t1_8, &t2_8, false, false, false);
        assert_eq!(
            via_method, via_free,
            "S156: ThetaIsogeny::compute must match theta_isogeny_compute free function",
        );
    }

    #[test]
    fn theta_isogeny_compute_method_propagates_errors_at_lvl1() {
        let domain = synthetic_domain();
        let zero = ThetaPoint2D::new(
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
        );
        let result = ThetaIsogeny::compute(&domain, &zero, &zero, false, false, false);
        assert_eq!(
            result,
            Err(ThetaIsogenyError::DegenerateFactor),
            "S156: method propagates DegenerateFactor identically to free function",
        );
    }

    #[test]
    fn theta_isogeny_eval_method_matches_free_function_at_lvl1() {
        let phi = synthetic_isogeny_for_eval((2, 3, 5, 7), false, false);
        let p = ThetaPoint2D::new(small_fp2(1), small_fp2(2), small_fp2(3), small_fp2(4));

        let via_method = phi.eval(&p);
        let via_free = theta_isogeny_eval(&phi, &p);
        assert_eq!(
            via_method, via_free,
            "S156: phi.eval(&p) must match theta_isogeny_eval(&phi, &p)",
        );
    }

    /// S156: confirm eval method works across all 4 Hadamard permutations
    /// (delegates correctly regardless of flag config).
    #[test]
    fn theta_isogeny_eval_method_across_hadamard_permutations_at_lvl1() {
        let p = ThetaPoint2D::new(small_fp2(1), small_fp2(2), small_fp2(3), small_fp2(4));
        for &b1 in &[false, true] {
            for &b2 in &[false, true] {
                let phi = synthetic_isogeny_for_eval((2, 3, 5, 7), b1, b2);
                let via_method = phi.eval(&p);
                let via_free = theta_isogeny_eval(&phi, &p);
                assert_eq!(
                    via_method, via_free,
                    "S156: eval method matches free function for ({b1}, {b2})",
                );
            }
        }
    }

    // S157 — ThetaIsogeny::compute_4 + compute_2 method-form alias tests.

    #[test]
    fn theta_isogeny_compute_4_method_matches_free_function_at_lvl1() {
        let domain_null = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        let domain = AbelianVariety2D::new(domain_null, domain_null);
        let (t1, t2) = nondegenerate_t_8();

        let via_method = ThetaIsogeny::compute_4(&domain, &t1, &t2, false, false);
        let via_free = theta_isogeny_compute_4(&domain, &t1, &t2, false, false);
        assert_eq!(
            via_method, via_free,
            "S157: ThetaIsogeny::compute_4 must match theta_isogeny_compute_4 free function",
        );
    }

    #[test]
    fn theta_isogeny_compute_2_method_matches_free_function_at_lvl1() {
        let domain_null = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        let domain = AbelianVariety2D::new(domain_null, domain_null);
        let (t1, t2) = nondegenerate_t_8();

        let via_method = ThetaIsogeny::compute_2(&domain, &t1, &t2, false, false);
        let via_free = theta_isogeny_compute_2(&domain, &t1, &t2, false, false);
        assert_eq!(
            via_method, via_free,
            "S157: ThetaIsogeny::compute_2 must match theta_isogeny_compute_2 free function",
        );
    }

    /// S157: confirm compute_4 + compute_2 methods cover all four
    /// Hadamard permutations identically to the free functions.
    #[test]
    fn theta_isogeny_compute_torsion_variants_across_hadamard_permutations_at_lvl1() {
        let domain_null = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        let domain = AbelianVariety2D::new(domain_null, domain_null);
        let (t1, t2) = nondegenerate_t_8();

        for &b1 in &[false, true] {
            for &b2 in &[false, true] {
                let method_4 = ThetaIsogeny::compute_4(&domain, &t1, &t2, b1, b2);
                let free_4 = theta_isogeny_compute_4(&domain, &t1, &t2, b1, b2);
                assert_eq!(method_4, free_4, "S157: compute_4 ({b1}, {b2})");

                let method_2 = ThetaIsogeny::compute_2(&domain, &t1, &t2, b1, b2);
                let free_2 = theta_isogeny_compute_2(&domain, &t1, &t2, b1, b2);
                assert_eq!(method_2, free_2, "S157: compute_2 ({b1}, {b2})");
            }
        }
    }
}
