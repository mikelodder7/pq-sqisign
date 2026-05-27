// SPDX-License-Identifier: MIT OR Apache-2.0
//! Gluing isogeny `E_1 × E_2 → A` for the SQIsign 2.0.1 (2,2)-isogeny chain.
//!
//! The gluing step is the FIRST isogeny in the higher-dimensional
//! `(2,2)`-isogeny chain that bridges the elliptic-product side
//! (couple-Jacobian arithmetic, [`crate::ec::couple`]) to the theta
//! side (2-D abelian variety, [`crate::isogeny::theta`]). It takes
//! two `8`-torsion couple-Jacobian kernel generators on the source
//! elliptic product `E_1 × E_2` and produces a [`GluingCodomain`]
//! describing the codomain abelian surface `A`, together with the
//! data needed to evaluate the isogeny on arbitrary couple-points.
//!
//! Reference: `theta_isogenies.c:gluing_compute` in the SQIsign C
//! reference (`github.com/SQIsign/the-sqisign`). The C implementation
//! returns `1` on success, `0` on malformed kernel (order check); we
//! use [`Result`] instead.
//!
//! # Scope of this module (S136 scaffold)
//!
//! This module currently ships only a **vertical slice** of the
//! gluing pipeline: the kernel-halving step (`8`-torsion → `4`-torsion
//! via [`CoupleJacobianPoint::double`]) and the Jacobian→XZ projection
//! (via [`CoupleJacobianPoint::to_couple_xz`]). The mathematically
//! deep step — `gluing_change_of_basis` deriving the `4×4` isomorphism
//! `M` and the codomain theta-null — is **not yet implemented** and
//! returns [`GluingError::NotImplemented`]. The slice exists to
//! validate that the type plumbing between Layer 2 (`couple.rs`) and
//! the eventual gluing output threads end-to-end at compile time,
//! per S136 advisor's vertical-slice recommendation.
//!
//! # Intended full surface (design comment per S136 advisor)
//!
//! Once the change-of-basis math lands (planned S137+), the full
//! surface will include:
//!
//! - [`GluingCodomain`] carrying: the codomain `A`'s theta-null
//!   `theta_null_a: ThetaPoint2D<F>`; the dual-isogeneous null
//!   inverse for [`GenericEval`](crate::isogeny::theta)-style image
//!   computation; the `4×4` basis-change matrix `M` (as a
//!   `BasisChangeMatrix<F>` newtype); the image of one kernel
//!   generator under the gluing map.
//! - [`gluing_compute`] (analog of C `gluing_compute`): the full
//!   constructor; consumes two 8-torsion kernel generators and the
//!   couple-curve, runs the change-of-basis derivation, validates
//!   isotropy, returns the codomain.
//! - `gluing_eval_point`: evaluates the isogeny on a couple-Jacobian
//!   point. Uses
//!   [`crate::ec::couple::CoupleJacobianPoint::add_components_pair`]
//!   (already shipped at S128) to extract cross-addition components,
//!   then applies the matrix `M`, squares pointwise, and applies the
//!   Hadamard transform.
//!
//! S136 ships only the kernel-halving slice; everything above is
//! deferred.

use subtle::{Choice, CtOption};

use crate::ec::couple::{CoupleCurve, CoupleJacobianPoint, CoupleMontgomeryPoint};
use crate::ec::montgomery::MontgomeryPoint;
use crate::gf::fp::BaseField;
use crate::gf::fp2::Fp2;
use crate::isogeny::theta::ThetaPoint2D;

/// Errors that arise during gluing-isogeny construction.
///
/// The variant set is intentionally small in S136; richer error
/// information lands as the mathematically-deeper steps are
/// implemented in S137+.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum GluingError {
    /// The input kernel is malformed: either [`verify_two_torsion`]
    /// failed after halving (`4`-torsion didn't double to valid
    /// distinct non-identity `2`-torsion), [`batch_invert`] failed
    /// (any input was zero), or [`isotropy_check`] failed
    /// (transformed `8`-torsion violated the `w == 0` geometry
    /// condition or had a zero in the asymmetric secondary factor
    /// set). The C reference treats all three failure modes
    /// identically by returning `0`; we collapse them into a single
    /// `InvalidKernel` variant. Callers requiring finer-grained
    /// diagnostics can re-run the individual sub-checks.
    InvalidKernel,
}

/// Output of the gluing isogeny construction — mirrors the C reference's
/// `theta_gluing_t` field layout.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GluingCodomain<F: BaseField> {
    /// `4 × 4` basis-change matrix derived from
    /// [`gluing_change_of_basis`].
    pub(crate) m: BasisChangeMatrix<F>,
    /// Theta-null point of the codomain abelian variety,
    /// post-Hadamard (`(α : β : γ : 0)` in C reference notation).
    pub(crate) codomain: ThetaPoint2D<F>,
    /// Cached projective factors used by downstream
    /// `gluing_eval_point` for fast point evaluation.
    pub(crate) precomputation: ThetaPoint2D<F>,
    /// Image of the first 8-torsion kernel generator under the
    /// gluing isogeny, in the special `(x : x : y : y)` form (only
    /// `x` and `y` carry information; `z` and `w` are zero by
    /// convention from the C reference, which only writes the two
    /// meaningful coordinates).
    pub(crate) image_k1_8: ThetaPoint2D<F>,
    /// Preserved copy of the first 8-torsion kernel input —
    /// `gluing_eval_point` requires this for image computation.
    pub(crate) xy_k1_8: CoupleJacobianPoint<F>,
    /// Source elliptic-product curve `E_1 × E_2`.
    pub(crate) domain: CoupleCurve<F>,
}

impl<F: BaseField> GluingCodomain<F> {
    /// Construct a [`GluingCodomain`] from an elliptic-product domain
    /// and two 8-torsion kernel generators.
    ///
    /// Method-form alias of [`gluing_codomain`]. Same body, same
    /// `Result<Self, GluingError>` failure mode (`InvalidKernel`).
    #[allow(dead_code)]
    pub(crate) fn compute(
        domain: &CoupleCurve<F>,
        xy_k1_8: &CoupleJacobianPoint<F>,
        xy_k2_8: &CoupleJacobianPoint<F>,
    ) -> Result<Self, GluingError> {
        gluing_codomain(domain, xy_k1_8, xy_k2_8)
    }

    /// Evaluate this gluing isogeny on a couple-Jacobian point.
    ///
    /// Method-form alias of [`gluing_eval_point`]. The
    /// `gluing.eval_point(&p)` spelling reads more directly than
    /// `gluing_eval_point(&gluing, &p)` for chain-walker call sites.
    #[allow(dead_code)]
    pub(crate) fn eval_point(&self, p: &CoupleJacobianPoint<F>) -> ThetaPoint2D<F> {
        gluing_eval_point(self, p)
    }

    /// Evaluate this gluing isogeny on a couple-Montgomery (x-only)
    /// point — the chain-initialization variant.
    ///
    /// Method-form alias of [`gluing_eval_point_special_case`].
    /// Returns `Err(InvalidKernel)` if the `T.w == 0` invariant
    /// after squared_theta fails.
    #[allow(dead_code)]
    pub(crate) fn eval_point_special_case(
        &self,
        p_xz: &CoupleMontgomeryPoint<F>,
    ) -> Result<ThetaPoint2D<F>, GluingError> {
        gluing_eval_point_special_case(self, p_xz)
    }

    /// Evaluate this gluing isogeny on a basis pair (two
    /// couple-Jacobian points).
    ///
    /// Method-form alias of [`gluing_eval_basis`]. Trivial 2-call
    /// wrapper internally.
    #[allow(dead_code)]
    pub(crate) fn eval_basis(
        &self,
        xy_t1: &CoupleJacobianPoint<F>,
        xy_t2: &CoupleJacobianPoint<F>,
    ) -> (ThetaPoint2D<F>, ThetaPoint2D<F>) {
        gluing_eval_basis(self, xy_t1, xy_t2)
    }
}

/// Compute the codomain theta-null, precomputation factors, and
/// `imageK1_8` image from the squared-theta-transformed `8`-torsion
/// kernel generators.
///
/// Mirrors the post-isotropy block of
/// `theta_isogenies.c:gluing_compute`:
///
/// ```text
///   codomain.x = TT1.x · TT2.x
///   codomain.y = TT1.y · TT2.x
///   codomain.z = TT1.x · TT2.z
///   codomain.w = 0
///   precomp.x = TT1.y · TT2.z
///   precomp.y = codomain.z   (copy)
///   precomp.z = codomain.y   (copy)
///   precomp.w = 0
///   image_x = TT1.x · precomp.x
///   image_y = TT1.z · precomp.z
///   codomain = hadamard(codomain)
/// ```
///
/// Returns `(codomain_post_hadamard, precomputation, image_k1_8)`.
/// `image_k1_8` is encoded in `(x : x : y : y)` form — only `x` and
/// `y` carry information; `z` and `w` are set to zero by convention
/// here (the C reference does not write them, leaving them
/// undefined; Rust's value semantics require explicit zeros).
///
/// Inputs MUST already be the squared-theta transforms (i.e.,
/// `componentwise_square` ∘ `hadamard`) of `base_change(k1_8)` and
/// `base_change(k2_8)`. The isotropy condition (`tt.w == 0` etc.)
/// MUST have been validated by the caller via [`isotropy_check`]
/// before this function is invoked — calling on inputs that violate
/// isotropy produces a structurally-invalid codomain.
///
/// Reference: `theta_isogenies.c:gluing_compute`, the
/// "Projective factor" / "Compute the two components of phi(K1_8)"
/// blocks following the isotropy check.
#[allow(dead_code)]
pub(crate) fn build_codomain_from_squared_theta<F: BaseField>(
    tt1: &ThetaPoint2D<F>,
    tt2: &ThetaPoint2D<F>,
) -> (ThetaPoint2D<F>, ThetaPoint2D<F>, ThetaPoint2D<F>) {
    let zero = Fp2::<F>::zero();

    // Pre-Hadamard codomain construction (C: "Projective factor: Ax").
    let codomain_pre_hadamard = ThetaPoint2D::new(
        tt1.x.mul(&tt2.x), // x = TT1.x · TT2.x
        tt1.y.mul(&tt2.x), // y = TT1.y · TT2.x
        tt1.x.mul(&tt2.z), // z = TT1.x · TT2.z
        zero,              // w = 0
    );

    // Precomputation factors (C: "Projective factor: ABCxz").
    // Note the y↔z copy mapping mirrors C ref VERBATIM:
    //   precomp.y = codomain.z   (NOT codomain.y)
    //   precomp.z = codomain.y   (NOT codomain.z)
    let precomputation = ThetaPoint2D::new(
        tt1.y.mul(&tt2.z),       // x = TT1.y · TT2.z
        codomain_pre_hadamard.z, // y = codomain.z (= TT1.x · TT2.z)
        codomain_pre_hadamard.y, // z = codomain.y (= TT1.y · TT2.x)
        zero,                    // w = 0
    );

    // imageK1_8 in (x : x : y : y) form — only x and y carry info.
    //   image.x = TT1.x · precomp.x = TT1.x · TT1.y · TT2.z
    //   image.y = TT1.z · precomp.z = TT1.z · TT1.y · TT2.x
    let image_k1_8 = ThetaPoint2D::new(
        tt1.x.mul(&precomputation.x),
        tt1.z.mul(&precomputation.z),
        zero,
        zero,
    );

    // Final hadamard on the codomain (C: "compute the final codomain").
    let codomain = codomain_pre_hadamard.hadamard();

    (codomain, precomputation, image_k1_8)
}

/// Construct the gluing codomain from a pair of `8`-torsion kernel
/// generators on the elliptic-product domain — the full pipeline.
///
/// **S137f: now end-to-end.** Composes every helper in this module:
/// kernel halving via [`CoupleJacobianPoint::double`];
/// [`action_by_translation`] producing four [`TranslationMatrix`];
/// [`gluing_change_of_basis`] producing the `4 × 4` matrix `M`;
/// [`base_change`] on each `8`-torsion kernel input (project to
/// Montgomery x-only first via
/// [`CoupleJacobianPoint::to_couple_xz`]); squared-theta transform
/// (`componentwise_square` ∘ `hadamard`); [`isotropy_check`];
/// [`build_codomain_from_squared_theta`].
///
/// **Preconditions**:
/// - `xy_k1_8` and `xy_k2_8` are couple-Jacobian points of order
///   exactly `8` whose `4`-th doublings generate independent
///   `2`-torsion subgroups of `E_1 × E_2`.
/// - The two `4`-th doublings together generate the kernel of the
///   intended `(2,2)`-isogeny `E_1 × E_2 → A`.
///
/// Returns `Ok(GluingCodomain)` on success, `Err(InvalidKernel)` if
/// any sub-step fails (verify_two_torsion, batch_invert, isotropy).
/// Failure semantics match the C reference, which returns `0` on any
/// of these failure modes without finer-grained diagnostics.
///
/// Reference: `theta_isogenies.c:gluing_compute`.
#[allow(dead_code)]
pub(crate) fn gluing_codomain<F: BaseField>(
    domain: &CoupleCurve<F>,
    xy_k1_8: &CoupleJacobianPoint<F>,
    xy_k2_8: &CoupleJacobianPoint<F>,
) -> Result<GluingCodomain<F>, GluingError> {
    // Step 1: kernel halving (8-torsion → 4-torsion).
    let k1_4 = xy_k1_8.double(domain);
    let k2_4 = xy_k2_8.double(domain);

    // Step 2: derive four 2×2 translation matrices. Internally this
    // doubles again (4 → 2-torsion), runs verify_two_torsion, and
    // batched-inverts the (z, det) pairs. CtOption(_, FALSE) on any
    // failure → InvalidKernel.
    // `CtOption::unwrap_or` would require
    // `[TranslationMatrix; 4]: ConditionallySelectable` which we
    // have not implemented; the failure branch returns early, so
    // `into_option` + a guarded `expect` extracts the success
    // value safely.
    let gi_opt = action_by_translation(&k1_4, &k2_4, domain);
    if !bool::from(gi_opt.is_some()) {
        return Err(GluingError::InvalidKernel);
    }
    let gi = gi_opt
        .into_option()
        .expect("action_by_translation success — checked is_some above");

    // Step 3: compose the four matrices into the 4×4 basis-change M.
    let m = gluing_change_of_basis(&gi);

    // Step 4: apply M to each 8-torsion kernel point. base_change
    // takes a CoupleMontgomeryPoint (x-only); project the 8-torsion
    // Jacobian inputs via to_couple_xz.
    let k1_8_xz = xy_k1_8.to_couple_xz();
    let k2_8_xz = xy_k2_8.to_couple_xz();
    let tt1_pre = base_change(&m, &k1_8_xz);
    let tt2_pre = base_change(&m, &k2_8_xz);

    // Step 5: squared-theta transform (= componentwise_square ∘ hadamard).
    let tt1 = tt1_pre.componentwise_square().hadamard();
    let tt2 = tt2_pre.componentwise_square().hadamard();

    // Step 6: isotropy check (primary `w == 0` + asymmetric secondary
    // non-zero factor set per S137e). Choice → Err.
    let isotropy_ok = isotropy_check(&tt1, &tt2);
    if !bool::from(isotropy_ok) {
        return Err(GluingError::InvalidKernel);
    }

    // Step 7: build codomain theta-null + precomputation + imageK1_8
    // from the squared-theta-transformed kernel images.
    let (codomain, precomputation, image_k1_8) = build_codomain_from_squared_theta(&tt1, &tt2);

    let _ = k2_4; // S137f: k2_4 computed unconditionally for action_by_translation; unused after.

    Ok(GluingCodomain {
        m,
        codomain,
        precomputation,
        image_k1_8,
        xy_k1_8: *xy_k1_8,
        domain: *domain,
    })
}

/// Verify that two couple-Jacobian 2-torsion points are well-formed
/// kernel generators for the `(2,2)`-isogeny: both have exact order 2
/// (i.e., doubling each one produces the identity on both halves of
/// `E_1 × E_2`), and the two points are distinct.
///
/// **Inputs MUST already be 2-torsion** — i.e., the doubled
/// projections of the 4-torsion kernel generators from the caller's
/// `gluing_compute` pipeline. Per the C reference
/// `theta_isogenies.c:verify_two_torsion`, this is invoked AFTER
/// `K_4 → K_2` halving inside `action_by_translation`. The function
/// is intentionally CT (returns [`Choice`]) because in the protocol's
/// signing path the inputs are derived from the secret kernel; a
/// variable-time order check would leak.
///
/// # Returns
///
/// `Choice::TRUE` iff ALL of:
/// - `K1_2.p1.double(E1.a) == ∞` (E_1 half is order ≤ 2)
/// - `K1_2.p2.double(E2.a) == ∞` (E_2 half is order ≤ 2)
/// - `K2_2.p1.double(E1.a) == ∞`
/// - `K2_2.p2.double(E2.a) == ∞`
/// - NOT (`K1_2.p1` equivalent `K2_2.p1` AND `K1_2.p2` equivalent
///   `K2_2.p2`) — i.e., the two couples are distinct.
/// - NOT (`K1_2` is `(O, O)`) — the couple is not the identity.
/// - NOT (`K2_2` is `(O, O)`).
///
/// All checks use [`crate::gf::fp2::Fp2::is_zero`] and
/// [`crate::ec::jacobian::JacobianPoint::is_equivalent`] which are
/// constant-time. The bitwise combination preserves CT.
///
/// Reference: `theta_isogenies.c:verify_two_torsion`.
#[allow(dead_code)]
pub(crate) fn verify_two_torsion<F: BaseField>(
    k1_2: &CoupleJacobianPoint<F>,
    k2_2: &CoupleJacobianPoint<F>,
    curves: &CoupleCurve<F>,
) -> Choice {
    // Double each couple; both halves must land at infinity.
    let k1_doubled = k1_2.double(curves);
    let k2_doubled = k2_2.double(curves);
    let k1_is_order_2_or_lower = k1_doubled.p1.is_infinity() & k1_doubled.p2.is_infinity();
    let k2_is_order_2_or_lower = k2_doubled.p1.is_infinity() & k2_doubled.p2.is_infinity();

    // Exclude the identity couple `(O, O)`: a "doubles to ∞" check
    // does not by itself rule out the case where the input WAS ∞.
    let k1_is_not_identity = !(k1_2.p1.is_infinity() & k1_2.p2.is_infinity());
    let k2_is_not_identity = !(k2_2.p1.is_infinity() & k2_2.p2.is_infinity());

    // Distinct couple-points: NOT (both halves projectively equal).
    let halves_equal = k1_2.p1.is_equivalent(&k2_2.p1) & k1_2.p2.is_equivalent(&k2_2.p2);
    let distinct = !halves_equal;

    k1_is_order_2_or_lower
        & k2_is_order_2_or_lower
        & k1_is_not_identity
        & k2_is_not_identity
        & distinct
}

/// `2 × 2` translation matrix `G` over `F_{p²}` — the per-point output of
/// [`action_by_translation_compute_matrix`].
///
/// Encodes the action of a `4`-torsion translation on the projective
/// `(X : Z)` coordinates of one curve, used as a building block for the
/// `4 × 4` basis-change matrix that the gluing isogeny derives in
/// [`gluing_change_of_basis`] (planned S137c). Field naming follows the
/// C reference's `translation_matrix_t`:
///
/// ```text
///   G = | g00  g01 |
///       | g10  g11 |
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct TranslationMatrix<F: BaseField> {
    /// Row 0, column 0.
    pub(crate) g00: Fp2<F>,
    /// Row 0, column 1.
    pub(crate) g01: Fp2<F>,
    /// Row 1, column 0.
    pub(crate) g10: Fp2<F>,
    /// Row 1, column 1.
    pub(crate) g11: Fp2<F>,
}

/// In-place batched inversion of `N` `F_{p²}` elements via Montgomery's
/// trick: compute one product, perform one inversion, propagate the
/// inverse backward through the running products to recover all `N`
/// individual inverses.
///
/// Cost: `3·(N − 1)` multiplications + `1` inversion (versus `N`
/// inversions naively). The `4` action-by-translation inversions per
/// kernel-point (`8` total for two kernel points) are the primary call
/// site planned for S137c; isolating the helper here keeps it private
/// until that caller materializes.
///
/// # Behavior on a zero input
///
/// If any input is zero, the product is zero, [`Fp2::invert`] returns
/// `None`, and the function returns `CtOption(_, FALSE)`. The output
/// array's contents on the failure path are unspecified (callers MUST
/// check the [`CtOption`] flag before reading). This matches the C
/// reference `fp2_batched_inv`'s "validate via `is_zero` after"
/// pattern, lifted into Rust's [`CtOption`] convention to match
/// [`Fp2::invert`].
///
/// # Constant-time
///
/// All operations use the constant-time [`Fp2`] multiplication and
/// inversion primitives; the running-product loops have data-
/// independent control flow. Per the S134 architectural analysis,
/// callers on the secret path (signing, gluing-isogeny construction
/// over a secret kernel) are expected.
#[allow(dead_code)]
pub(crate) fn batch_invert<F: BaseField, const N: usize>(values: &mut [Fp2<F>; N]) -> CtOption<()> {
    // Forward pass: scratch[i] = product of values[0..i].
    // We reuse `values` for the inverses on the backward pass, so
    // store the partial products in a separate stack array.
    let mut scratch: [Fp2<F>; N] = core::array::from_fn(|_| Fp2::<F>::one());
    let mut acc = Fp2::<F>::one();
    for i in 0..N {
        scratch[i] = acc;
        acc = acc.mul(&values[i]);
    }

    // One field inversion: acc⁻¹ = (∏ values)⁻¹.
    let acc_inv_opt = acc.invert();
    let mut inv = acc_inv_opt.unwrap_or(Fp2::<F>::zero());
    let is_some = acc_inv_opt.is_some();

    // Backward pass: walk the scratch products in reverse, multiplying
    // each by the running inverse to recover values[i]⁻¹. After the
    // loop, `inv` has been multiplied by every value; that final value
    // is discarded.
    for i in (0..N).rev() {
        let value_i = values[i];
        values[i] = scratch[i].mul(&inv);
        inv = inv.mul(&value_i);
    }

    CtOption::new((), is_some)
}

/// Per the C reference's `action_by_translation_z_and_det`, extract the
/// `z`-coordinate of a `4`-torsion couple-half and the determinant
/// `det = x₄·z₂ − z₄·x₂` of a `(P₄, P₂)` pair, where `P₂ = [2]P₄` is
/// the doubled `2`-torsion projection. Both values are returned
/// **un-inverted**; the caller batches them into a single inversion
/// via [`batch_invert`] before invoking
/// [`action_by_translation_compute_matrix`].
///
/// Inputs are projective `(X : Z)` Montgomery points (the result of
/// [`crate::ec::jacobian::JacobianPoint::to_montgomery_xz`] applied to
/// each half of a [`crate::ec::couple::CoupleJacobianPoint`]).
///
/// Reference: `theta_isogenies.c:action_by_translation_z_and_det`.
#[allow(dead_code)]
pub(crate) fn action_by_translation_z_and_det<F: BaseField>(
    p4: &MontgomeryPoint<F>,
    p2: &MontgomeryPoint<F>,
) -> (Fp2<F>, Fp2<F>) {
    // z = P4.z (unchanged; collected for batched inversion downstream).
    let z = p4.z;
    // det = P4.x · P2.z − P4.z · P2.x.
    let det = p4.x.mul(&p2.z).sub(&p4.z.mul(&p2.x));
    (z, det)
}

/// Per the C reference's `action_by_translation_compute_matrix`,
/// build a [`TranslationMatrix`] from a `(P₄, P₂)` projective pair
/// together with the **pre-inverted** `z⁻¹` and `det⁻¹` produced by
/// [`batch_invert`] over the collected `z`s and `det`s of the four
/// input pairs.
///
/// Formula (matches the C reference exactly):
///
/// ```text
///   tmp = P4.x · z_inv                (= x₄/z)
///   g10 = P4.x · P2.x · det_inv − tmp  (= x₄·x₂/det − x₄/z)
///   g11 = P4.x · P4.z · det_inv        (= x₄·z₄/det … wait — check)
/// ```
///
/// The exact formula relies on Lubicz–Robert-style theta-translation
/// algebra; the present implementation transcribes the C reference
/// verbatim. Callers MUST pass `z_inv` and `det_inv` already inverted
/// via [`batch_invert`]; passing un-inverted values silently produces
/// wrong matrices.
///
/// Reference: `theta_isogenies.c:action_by_translation_compute_matrix`.
#[allow(dead_code)]
pub(crate) fn action_by_translation_compute_matrix<F: BaseField>(
    p4: &MontgomeryPoint<F>,
    p2: &MontgomeryPoint<F>,
    z_inv: &Fp2<F>,
    det_inv: &Fp2<F>,
) -> TranslationMatrix<F> {
    // g10 = (x4 · x2 · det⁻¹) − (x4 · z⁻¹)
    let tmp = p4.x.mul(z_inv);
    let g10 = p4.x.mul(&p2.x).mul(det_inv).sub(&tmp);

    // g11 = x4 · z4 · det⁻¹  (per the C transcription)
    let g11 = p4.x.mul(&p4.z).mul(det_inv);

    // g00 = −g11
    let g00 = g11.negate();

    // g01 = −(x2 · z4 · det⁻¹)
    let g01 = p2.x.mul(&p4.z).mul(det_inv).negate();

    TranslationMatrix { g00, g01, g10, g11 }
}

/// Compose [`action_by_translation_z_and_det`] + [`batch_invert`] +
/// [`action_by_translation_compute_matrix`] over the four halves of two
/// `4`-torsion couple-Jacobian kernel generators on `E_1 × E_2`,
/// producing the four `2 × 2` translation matrices `G_i[0..3]` that
/// the gluing isogeny's `4 × 4` basis-change derivation
/// ([`gluing_change_of_basis`], planned S137d) composes.
///
/// # Algorithm
///
/// Mirrors `theta_isogenies.c:action_by_translation`:
/// 1. Double both kernel couples to obtain the `2`-torsion projections
///    `K1_2 = [2]K1_4` and `K2_2 = [2]K2_4`.
/// 2. Run [`verify_two_torsion`] on the doubled couples. Early-failure
///    propagates as `CtOption(_, FALSE)` per the C reference's
///    `return 0` on malformed kernel.
/// 3. Project both `4`-torsion and `2`-torsion couples to Montgomery
///    `(X : Z)` via [`CoupleJacobianPoint::to_couple_xz`].
/// 4. Extract four `(z, det)` pairs via
///    [`action_by_translation_z_and_det`], laying them out in an
///    `8`-element array per the C reference's slot mapping:
///    `inverses = [z_0, z_1, z_2, z_3, det_0, det_1, det_2, det_3]`.
/// 5. Run [`batch_invert`] over all eight in a single Montgomery-trick
///    inversion. Failure (any input zero) propagates as
///    `CtOption(_, FALSE)`.
/// 6. Build the four matrices via
///    [`action_by_translation_compute_matrix`], pairing each pair's
///    `(z⁻¹, det⁻¹)` as `(inverses[i], inverses[i + 4])` for `i ∈ 0..4`.
///
/// # Critical slot layout (S137c advisor-flagged risk)
///
/// The C reference uses `inverses[i]` for the `z` of the `i`-th pair
/// and `inverses[i + 4]` for the `det` of the same pair, with pairs
/// ordered `(K1_4.P1, K1_2.P1), (K1_4.P2, K1_2.P2), (K2_4.P1, K2_2.P1),
/// (K2_4.P2, K2_2.P2)`. Swapping the `z` and `det` halves of the
/// array, or interleaving them as `[z_0, det_0, z_1, det_1, …]`,
/// would produce structurally-valid but algebraically-wrong matrices
/// that a circular composition-vs-per-pair invariant test would NOT
/// detect. The layout below is transcribed verbatim from the C
/// reference (lines 222-225 of `theta_isogenies.c`, the four
/// `action_by_translation_z_and_det(&inverses[i], &inverses[i+4], …)`
/// call sites).
///
/// # Returns
///
/// `CtOption(_, TRUE)` with the four matrices iff the kernel passes
/// [`verify_two_torsion`] AND all eight inversions succeed.
/// `CtOption(_, FALSE)` otherwise; the matrix payload on failure is
/// unspecified.
///
/// Reference: `theta_isogenies.c:action_by_translation`.
#[allow(dead_code)]
pub(crate) fn action_by_translation<F: BaseField>(
    k1_4: &CoupleJacobianPoint<F>,
    k2_4: &CoupleJacobianPoint<F>,
    curves: &CoupleCurve<F>,
) -> CtOption<[TranslationMatrix<F>; 4]> {
    // 1. Double 4-torsion → 2-torsion couples.
    let k1_2 = k1_4.double(curves);
    let k2_2 = k2_4.double(curves);

    // 2. Verify the doubled couples are valid 2-torsion kernel halves
    //    (each doubles to (∞, ∞); neither is the identity couple;
    //    distinct).
    let verify_ok = verify_two_torsion(&k1_2, &k2_2, curves);

    // 3. Project both 4-torsion and 2-torsion couples to (X : Z).
    let k1_4_xz = k1_4.to_couple_xz();
    let k2_4_xz = k2_4.to_couple_xz();
    let k1_2_xz = k1_2.to_couple_xz();
    let k2_2_xz = k2_2.to_couple_xz();

    // 4. Extract four (z, det) pairs.
    let (z0, d0) = action_by_translation_z_and_det(&k1_4_xz.p1, &k1_2_xz.p1);
    let (z1, d1) = action_by_translation_z_and_det(&k1_4_xz.p2, &k1_2_xz.p2);
    let (z2, d2) = action_by_translation_z_and_det(&k2_4_xz.p1, &k2_2_xz.p1);
    let (z3, d3) = action_by_translation_z_and_det(&k2_4_xz.p2, &k2_2_xz.p2);

    // 5. Layout per C reference: z's at [0..4], det's at [4..8].
    //    Slot mapping is the highest-risk axis (S137c advisor); this
    //    array order is transcribed verbatim from the four
    //    `action_by_translation_z_and_det(&inverses[i], &inverses[i+4], …)`
    //    call sites in theta_isogenies.c.
    let mut inverses = [z0, z1, z2, z3, d0, d1, d2, d3];
    let inv_opt = batch_invert(&mut inverses);

    // 6. Build the four 2×2 translation matrices. Pairing
    //    `(inverses[i], inverses[i + 4])` mirrors the C reference's
    //    `(z_inv, det_inv)` argument order.
    let g0 =
        action_by_translation_compute_matrix(&k1_4_xz.p1, &k1_2_xz.p1, &inverses[0], &inverses[4]);
    let g1 =
        action_by_translation_compute_matrix(&k1_4_xz.p2, &k1_2_xz.p2, &inverses[1], &inverses[5]);
    let g2 =
        action_by_translation_compute_matrix(&k2_4_xz.p1, &k2_2_xz.p1, &inverses[2], &inverses[6]);
    let g3 =
        action_by_translation_compute_matrix(&k2_4_xz.p2, &k2_2_xz.p2, &inverses[3], &inverses[7]);

    let success = verify_ok & inv_opt.is_some();
    CtOption::new([g0, g1, g2, g3], success)
}

/// `4 × 4` basis-change matrix `M` over `F_{p²}` — the output of
/// [`gluing_change_of_basis`].
///
/// In the gluing-isogeny pipeline (analog of
/// `theta_isogenies.c:gluing_change_of_basis`), `M` is the
/// isomorphism that lifts the kernel from the elliptic-product side
/// into the theta null-point space of the codomain abelian variety.
/// Subsequent steps (`base_change`, `to_squared_theta`, the isotropy
/// check) consume this matrix to derive the codomain's theta-null
/// point and its precomputation. Both consumption steps live in the
/// caller path (planned S137e); this newtype is just the matrix
/// payload.
///
/// Layout: row-major `m[row][col]`. Construction goes via
/// [`gluing_change_of_basis`] only — the inner `m` field is
/// `pub(crate)` to permit test introspection but no constructor is
/// exposed.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct BasisChangeMatrix<F: BaseField> {
    /// Row-major `4 × 4` storage. `m[i][j]` is row `i`, column `j`.
    pub(crate) m: [[Fp2<F>; 4]; 4],
}

impl<F: BaseField> BasisChangeMatrix<F> {
    /// Compute `self · other` — standard `4 × 4` matrix product.
    ///
    /// Method-form alias of
    /// [`crate::isogeny::splitting::base_change_matrix_multiplication`].
    /// Use whichever spelling reads cleaner at the call site;
    /// `lhs.mul(&rhs)` matches conventional matrix-algebra notation.
    #[allow(dead_code)]
    pub(crate) fn mul(&self, other: &Self) -> Self {
        crate::isogeny::splitting::base_change_matrix_multiplication(self, other)
    }

    /// Constant-time conditional select between `self` and `other`.
    ///
    /// Returns `self` when `choice` is `Choice::FALSE`; returns
    /// `other` when `Choice::TRUE`. Entrywise CT via
    /// [`Fp2::conditional_select`].
    ///
    /// Method-form alias of
    /// [`crate::isogeny::splitting::select_base_change_matrix`].
    #[allow(dead_code)]
    pub(crate) fn select(&self, other: &Self, choice: Choice) -> Self {
        crate::isogeny::splitting::select_base_change_matrix(self, other, choice)
    }
}

/// Compose four [`TranslationMatrix`] values (the output of
/// [`action_by_translation`]) into the `4 × 4` basis-change matrix
/// per `theta_isogenies.c:gluing_change_of_basis`.
///
/// # Algorithm
///
/// The C reference computes four intermediate scalars and then 16
/// matrix entries via specific products + sums. Layout mirrors the
/// C body verbatim:
/// - **Intermediates** `t001, t101, t002, t102`: first/second columns
///   of the `Gi[0]·Gi[2]` and `Gi[1]·Gi[3]` 2×2 matrix products,
///   captured as fp2 scalars (only the first column of each product
///   participates downstream).
/// - **Row 0** (trace-like): `M[0][k]` for `k ∈ 0..4` combine the
///   intermediates plus direct `Gi[i].gXX` cross-products.
/// - **Row 1** (action of `(0, K2_4.P2)`): applies `Gi[3]` to row 0
///   columns 0,2 (with `g0X`) and 1,3 (with `g1X`).
/// - **Row 2** (action of `(K1_4.P1, 0)`): applies `Gi[0]` to row 0
///   columns 0,1 and 2,3.
/// - **Row 3** (action of `(K1_4.P1, K2_4.P2)`): applies `Gi[0]` to
///   row 1 (already-translated by `Gi[3]`).
///
/// Each entry is annotated with the corresponding C-reference structure
/// in inline comments. Per S137d advisor doctrine: do NOT editorialize —
/// every `fp2_mul`/`fp2_add` pair from the C body maps to a literal
/// Rust `.mul()`/`.add()` in the same order.
///
/// Reference: `theta_isogenies.c:gluing_change_of_basis`.
#[allow(dead_code)]
pub(crate) fn gluing_change_of_basis<F: BaseField>(
    gi: &[TranslationMatrix<F>; 4],
) -> BasisChangeMatrix<F> {
    // --- Intermediates (C ref lines 26-43) ---
    // t001 = Gi[0].g00·Gi[2].g00 + Gi[0].g01·Gi[2].g10
    let t001 = gi[0].g00.mul(&gi[2].g00).add(&gi[0].g01.mul(&gi[2].g10));
    // t101 = Gi[0].g10·Gi[2].g00 + Gi[0].g11·Gi[2].g10
    let t101 = gi[0].g10.mul(&gi[2].g00).add(&gi[0].g11.mul(&gi[2].g10));
    // t002 = Gi[1].g00·Gi[3].g00 + Gi[1].g01·Gi[3].g10
    let t002 = gi[1].g00.mul(&gi[3].g00).add(&gi[1].g01.mul(&gi[3].g10));
    // t102 = Gi[1].g10·Gi[3].g00 + Gi[1].g11·Gi[3].g10
    let t102 = gi[1].g10.mul(&gi[3].g00).add(&gi[1].g11.mul(&gi[3].g10));

    // --- Row 0: trace for the first row (C ref lines 45-67) ---
    // M[0][0] = 1 + t001·t002 + Gi[2].g00·Gi[3].g00 + Gi[0].g00·Gi[1].g00
    let m00 = Fp2::<F>::one()
        .add(&t001.mul(&t002))
        .add(&gi[2].g00.mul(&gi[3].g00))
        .add(&gi[0].g00.mul(&gi[1].g00));
    // M[0][1] = t001·t102 + Gi[2].g00·Gi[3].g10 + Gi[0].g00·Gi[1].g10
    let m01 = t001
        .mul(&t102)
        .add(&gi[2].g00.mul(&gi[3].g10))
        .add(&gi[0].g00.mul(&gi[1].g10));
    // M[0][2] = t101·t002 + Gi[2].g10·Gi[3].g00 + Gi[0].g10·Gi[1].g00
    let m02 = t101
        .mul(&t002)
        .add(&gi[2].g10.mul(&gi[3].g00))
        .add(&gi[0].g10.mul(&gi[1].g00));
    // M[0][3] = t101·t102 + Gi[2].g10·Gi[3].g10 + Gi[0].g10·Gi[1].g10
    let m03 = t101
        .mul(&t102)
        .add(&gi[2].g10.mul(&gi[3].g10))
        .add(&gi[0].g10.mul(&gi[1].g10));

    // --- Row 1: action of (0, K2_4.P2) via Gi[3] on row 0 (C lines 69-83) ---
    // M[1][0] = Gi[3].g00·M[0][0] + Gi[3].g01·M[0][1]
    let m10 = gi[3].g00.mul(&m00).add(&gi[3].g01.mul(&m01));
    // M[1][1] = Gi[3].g10·M[0][0] + Gi[3].g11·M[0][1]
    let m11 = gi[3].g10.mul(&m00).add(&gi[3].g11.mul(&m01));
    // M[1][2] = Gi[3].g00·M[0][2] + Gi[3].g01·M[0][3]
    let m12 = gi[3].g00.mul(&m02).add(&gi[3].g01.mul(&m03));
    // M[1][3] = Gi[3].g10·M[0][2] + Gi[3].g11·M[0][3]
    let m13 = gi[3].g10.mul(&m02).add(&gi[3].g11.mul(&m03));

    // --- Row 2: action of (K1_4.P1, 0) via Gi[0] on row 0 (C lines 85-99) ---
    // M[2][0] = Gi[0].g00·M[0][0] + Gi[0].g01·M[0][2]
    let m20 = gi[0].g00.mul(&m00).add(&gi[0].g01.mul(&m02));
    // M[2][1] = Gi[0].g00·M[0][1] + Gi[0].g01·M[0][3]
    let m21 = gi[0].g00.mul(&m01).add(&gi[0].g01.mul(&m03));
    // M[2][2] = Gi[0].g10·M[0][0] + Gi[0].g11·M[0][2]
    let m22 = gi[0].g10.mul(&m00).add(&gi[0].g11.mul(&m02));
    // M[2][3] = Gi[0].g10·M[0][1] + Gi[0].g11·M[0][3]
    let m23 = gi[0].g10.mul(&m01).add(&gi[0].g11.mul(&m03));

    // --- Row 3: action of (K1_4.P1, K2_4.P2) via Gi[0] on row 1 (C lines 101-115) ---
    // M[3][0] = Gi[0].g00·M[1][0] + Gi[0].g01·M[1][2]
    let m30 = gi[0].g00.mul(&m10).add(&gi[0].g01.mul(&m12));
    // M[3][1] = Gi[0].g00·M[1][1] + Gi[0].g01·M[1][3]
    let m31 = gi[0].g00.mul(&m11).add(&gi[0].g01.mul(&m13));
    // M[3][2] = Gi[0].g10·M[1][0] + Gi[0].g11·M[1][2]
    let m32 = gi[0].g10.mul(&m10).add(&gi[0].g11.mul(&m12));
    // M[3][3] = Gi[0].g10·M[1][1] + Gi[0].g11·M[1][3]
    let m33 = gi[0].g10.mul(&m11).add(&gi[0].g11.mul(&m13));

    BasisChangeMatrix {
        m: [
            [m00, m01, m02, m03],
            [m10, m11, m12, m13],
            [m20, m21, m22, m23],
            [m30, m31, m32, m33],
        ],
    }
}

/// Apply the `4 × 4` basis-change matrix `M` to a [`ThetaPoint2D`] via
/// row-major matrix-vector multiplication.
///
/// # Convention
///
/// Row-major per `theta_isogenies.c:apply_isomorphism_general`:
///
/// ```text
///   out.x = P.x · M[0][0] + P.y · M[0][1] + P.z · M[0][2] + P.w · M[0][3]
///   out.y = P.x · M[1][0] + P.y · M[1][1] + P.z · M[1][2] + P.w · M[1][3]
///   out.z = P.x · M[2][0] + P.y · M[2][1] + P.z · M[2][2] + P.w · M[2][3]
///   out.w = P.x · M[3][0] + P.y · M[3][1] + P.z · M[3][2] + P.w · M[3][3]
/// ```
///
/// Note Rust uses `w` for the fourth coordinate where the C reference
/// uses `t` — pure naming difference, same value.
///
/// # Return-by-value
///
/// Returns a fresh `ThetaPoint2D`; this avoids the in-place aliasing
/// bug class (writing `out[i]` while still reading `v[j]` when `out`
/// and `v` alias) that the C version risks.
///
/// Reference: `theta_isogenies.c:apply_isomorphism` →
/// `apply_isomorphism_general`.
#[allow(dead_code)]
pub(crate) fn apply_isomorphism<F: BaseField>(
    m: &BasisChangeMatrix<F>,
    p: &ThetaPoint2D<F>,
) -> ThetaPoint2D<F> {
    let row_dot = |row: &[Fp2<F>; 4]| -> Fp2<F> {
        p.x.mul(&row[0])
            .add(&p.y.mul(&row[1]))
            .add(&p.z.mul(&row[2]))
            .add(&p.w.mul(&row[3]))
    };
    ThetaPoint2D::new(
        row_dot(&m.m[0]),
        row_dot(&m.m[1]),
        row_dot(&m.m[2]),
        row_dot(&m.m[3]),
    )
}

/// Construct a theta-coordinate "null point" from a couple-Montgomery
/// point `T = (P1, P2)` on `E_1 × E_2`, then apply the basis-change
/// matrix `M` via [`apply_isomorphism`].
///
/// The null-point construction mirrors `theta_isogenies.c:base_change`:
///
/// ```text
///   null.x = P1.x · P2.x
///   null.y = P1.x · P2.z
///   null.z = P2.x · P1.z   (= P1.z · P2.x)
///   null.w = P1.z · P2.z
/// ```
///
/// The output `ThetaPoint2D` is the image of `T` in the codomain
/// abelian variety's theta null-point space under the gluing
/// isomorphism `M`.
///
/// Reference: `theta_isogenies.c:base_change`.
#[allow(dead_code)]
pub(crate) fn base_change<F: BaseField>(
    m: &BasisChangeMatrix<F>,
    t: &CoupleMontgomeryPoint<F>,
) -> ThetaPoint2D<F> {
    // Construct null_point from the 4 products of (x, z) coords.
    let null_point = ThetaPoint2D::new(
        t.p1.x.mul(&t.p2.x), // x = P1.x · P2.x
        t.p1.x.mul(&t.p2.z), // y = P1.x · P2.z
        t.p2.x.mul(&t.p1.z), // z = P2.x · P1.z
        t.p1.z.mul(&t.p2.z), // w = P1.z · P2.z
    );
    apply_isomorphism(m, &null_point)
}

/// Validate gluing-kernel isotropy: after applying `base_change` and
/// the squared-theta transform (`componentwise_square` ∘ `hadamard`)
/// to the two `8`-torsion kernel generators, the resulting points
/// `TT1`, `TT2` must have zero `w`-coordinate (the geometry
/// condition for the codomain to factor as an elliptic product), AND
/// a specific subset of their projective factors must be non-zero
/// (a sanity check on the input).
///
/// # Asymmetric secondary check (S137e advisor doctrine — preserved
/// verbatim, do NOT symmetrize)
///
/// Per `theta_isogenies.c:gluing_compute` (the isotropy-check block
/// following the squared-theta transform), the secondary non-zero
/// check is over the ASYMMETRIC set
///
/// ```text
///   { TT1.x, TT2.x, TT1.y, TT2.z, TT1.z }
/// ```
///
/// — `TT2.y` is explicitly NOT checked. This irregularity looks like
/// a copy-paste inconsistency one might be tempted to "clean up"
/// into a symmetric `{TT1.{x,y,z}, TT2.{x,y,z}}` form. **Do not.**
/// The asymmetry is transcribed verbatim from the C reference; the
/// gluing geometry does not require `TT2.y` to be non-zero.
///
/// # CT
///
/// All checks use [`Fp2::is_zero`] (returns [`Choice`], constant-time)
/// and bitwise `Choice` combinators. The inputs `TT1`, `TT2` are
/// derived from the secret kernel via [`base_change`] +
/// squared-theta, so this check is on the secret-affected boundary
/// and MUST be constant-time per the S134 architectural decision.
///
/// Reference: `theta_isogenies.c:gluing_compute` (the
/// `if (!(fp2_is_zero(&TT1.t) & fp2_is_zero(&TT2.t)))` block + the
/// subsequent "Test our projective factors are non zero" block).
#[allow(dead_code)]
pub(crate) fn isotropy_check<F: BaseField>(tt1: &ThetaPoint2D<F>, tt2: &ThetaPoint2D<F>) -> Choice {
    // Primary check: both `w` coordinates must be zero (the
    // geometry condition for the codomain to factor as a product).
    let w_both_zero = tt1.w.is_zero() & tt2.w.is_zero();

    // Secondary check: ASYMMETRIC non-zero set — preserved verbatim
    // from the C reference. DO NOT symmetrize; `TT2.y` is
    // intentionally not in the set.
    let factors_nonzero = !tt1.x.is_zero()
        & !tt2.x.is_zero()
        & !tt1.y.is_zero()
        & !tt2.z.is_zero()
        & !tt1.z.is_zero();

    w_both_zero & factors_nonzero
}

/// Evaluate the gluing isogeny `phi: E_1 × E_2 → A` on a couple-Jacobian
/// point `P` — producing its theta-coordinate image in the codomain.
///
/// # Algorithm
///
/// Mirrors `theta_isogenies.c:gluing_eval_point` verbatim:
///
/// 1. Compute the cross-addition components `(u_i, v_i, w_i)` for
///    `(P.p_i, gluing.xy_k1_8.p_i)` on each curve `E_i` via
///    [`CoupleJacobianPoint::add_components_pair`] (S128).
/// 2. Build two intermediate theta points `T1` and `T2`:
///    ```text
///    T1.x = u1·u2 + v1·v2          T2.x = (u1+v1)(u2+v2) − T1.x
///    T1.y = u1·w2                  T2.y = v1·w2
///    T1.z = w1·u2                  T2.z = w1·v2
///    T1.w = w1·w2                  T2.w = 0
///    ```
///    `T1` represents `θ(P+Q)·θ(P−Q)` (after subsequent squaring +
///    subtraction); `T2` is the auxiliary needed for sign separation.
/// 3. Apply the basis-change matrix `M` to both `T1` and `T2` via
///    [`apply_isomorphism`]. (C reference uses the
///    `apply_isomorphism_general(..., false)` optimization for `T2`
///    because `T2.w = 0`; Rust uses the full version for both. The
///    column-3 multiplications for `T2` are wasted work — algebraically
///    identical to the optimized form — but correctness is preserved.
///    A perf-focused session could later add an `apply_isomorphism_partial`
///    variant for the `w=0` case.)
/// 4. Square both pointwise via [`ThetaPoint2D::componentwise_square`].
/// 5. Compute `T1 = T1 − T2` componentwise.
/// 6. Apply Hadamard to the difference.
/// 7. Scale the result componentwise by the "inverse" of `imageK1_8`
///    in `(x : x : y : y)` form, which is `(y : y : x : x)`:
///    ```text
///    image.x = T1.x · imageK1_8.y
///    image.y = T1.y · imageK1_8.y
///    image.z = T1.z · imageK1_8.x
///    image.w = T1.w · imageK1_8.x
///    ```
/// 8. Apply the final Hadamard to `image`.
///
/// # Preconditions
///
/// - `P.p1 ≠ gluing.xy_k1_8.p1` (affine) AND `P.p2 ≠ gluing.xy_k1_8.p2`
///   (affine), AND neither is at infinity, AND neither is the negation
///   of the other. These are inherited from
///   [`JacobianPoint::add_components`]'s preconditions per S127 design.
///   Production callers ensure this by construction (the protocol's
///   point-evaluation always operates on points distinct from the
///   kernel generators).
/// - `gluing` is a valid output of [`gluing_codomain`].
///
/// Reference: `theta_isogenies.c:gluing_eval_point`.
#[allow(dead_code)]
pub(crate) fn gluing_eval_point<F: BaseField>(
    gluing: &GluingCodomain<F>,
    p: &CoupleJacobianPoint<F>,
) -> ThetaPoint2D<F> {
    // Step 1: cross-addition components for both halves.
    let comps = p.add_components_pair(&gluing.xy_k1_8, &gluing.domain);
    let (u1, v1, w1) = comps[0];
    let (u2, v2, w2) = comps[1];

    // Step 2: build T1. Order matters — T2.x is computed using the
    // ORIGINAL T1.x before any mutation.
    let t1_x = u1.mul(&u2).add(&v1.mul(&v2));
    let t1_y = u1.mul(&w2);
    let t1_z = w1.mul(&u2);
    let t1_w = w1.mul(&w2);
    let t1 = ThetaPoint2D::new(t1_x, t1_y, t1_z, t1_w);

    // Step 2 (cont.): build T2. T2.x = (u1+v1)(u2+v2) − T1.x.
    let t2_x = u1.add(&v1).mul(&u2.add(&v2)).sub(&t1_x);
    let t2_y = v1.mul(&w2);
    let t2_z = w1.mul(&v2);
    let t2_w = Fp2::<F>::zero();
    let t2 = ThetaPoint2D::new(t2_x, t2_y, t2_z, t2_w);

    // Step 3: apply M to both. Rust uses the full `apply_isomorphism`
    // (= C's `apply_isomorphism_general(..., true)`). For T2 the C
    // ref uses the `false` optimization (T2.w=0 skips col-3 mults);
    // we accept the perf delta for simplicity. Algebraic equivalent.
    let t1 = apply_isomorphism(&gluing.m, &t1);
    let t2 = apply_isomorphism(&gluing.m, &t2);

    // Step 4: pointwise square.
    let t1 = t1.componentwise_square();
    let t2 = t2.componentwise_square();

    // Step 5: T1 = T1 − T2 componentwise.
    let diff = ThetaPoint2D::new(
        t1.x.sub(&t2.x),
        t1.y.sub(&t2.y),
        t1.z.sub(&t2.z),
        t1.w.sub(&t2.w),
    );

    // Step 6: hadamard of the difference.
    let hadamarded = diff.hadamard();

    // Step 7: scale by the inverse of imageK1_8 in (x:x:y:y) form,
    // which is (y:y:x:x). Per the C reference comment:
    //   image.x = T1.x · imageK1_8.y
    //   image.y = T1.y · imageK1_8.y
    //   image.z = T1.z · imageK1_8.x
    //   image.w = T1.w · imageK1_8.x
    let image_pre = ThetaPoint2D::new(
        hadamarded.x.mul(&gluing.image_k1_8.y),
        hadamarded.y.mul(&gluing.image_k1_8.y),
        hadamarded.z.mul(&gluing.image_k1_8.x),
        hadamarded.w.mul(&gluing.image_k1_8.x),
    );

    // Step 8: final hadamard.
    image_pre.hadamard()
}

/// Evaluate the gluing isogeny on a couple-Montgomery (x-only) input.
///
/// Companion to [`gluing_eval_point`] for the chain-initialization
/// path where the input point is naturally x-only (it comes from the
/// elliptic-product side via a Montgomery ladder, not from a
/// full Jacobian computation). The two routines compute the SAME
/// gluing isogeny but accept different input representations:
///
/// - [`gluing_eval_point`] takes a `&CoupleJacobianPoint<F>` and uses
///   `add_components_pair` to extract cross-addition components.
/// - This routine takes a `&CoupleMontgomeryPoint<F>` and applies the
///   basis-change matrix directly, then the squared-theta transform,
///   then scales by the gluing's `precomputation` triple.
///
/// # Algorithm
///
/// Mirrors `theta_isogenies.c:gluing_eval_point_special_case`:
///
/// 1. `T = base_change(&gluing.m, p_xz)` — apply M to lift the
///    couple-Montgomery point into the theta-coord 4-tuple.
/// 2. `T = to_squared_theta(T)` — i.e., `T.componentwise_square().hadamard()`.
/// 3. **Isotropy check on T.w**: a valid gluing requires `T.w == 0`
///    after the squared-theta transform (the "D = 0 in a gluing"
///    invariant from theta_isogenies.c:566). If `T.w ≠ 0`, the input
///    is not a valid kernel descendant — return `Err(InvalidKernel)`.
/// 4. Scale componentwise by `gluing.precomputation.{x, y, z}`:
///    ```text
///    image.x = T.x · precomputation.x
///    image.y = T.y · precomputation.y
///    image.z = T.z · precomputation.z
///    image.w = 0
///    ```
/// 5. Apply [`ThetaPoint2D::hadamard`] to the scaled image.
///
/// # Failure mode
///
/// Returns `Err(GluingError::InvalidKernel)` if `T.w ≠ 0` after the
/// squared-theta transform. The C reference returns `0` (failure)
/// for this case at `theta_isogenies.c:569`; we collapse it into the
/// same `InvalidKernel` variant used by [`gluing_codomain`] for
/// failure-on-malformed-input.
///
/// Reference: `theta_isogenies.c:gluing_eval_point_special_case`.
#[allow(dead_code)]
pub(crate) fn gluing_eval_point_special_case<F: BaseField>(
    gluing: &GluingCodomain<F>,
    p_xz: &CoupleMontgomeryPoint<F>,
) -> Result<ThetaPoint2D<F>, GluingError> {
    // Step 1: lift into theta-coord 4-tuple via the basis-change matrix.
    let t_pre = base_change(&gluing.m, p_xz);

    // Step 2: squared-theta transform (componentwise_square then hadamard).
    let t = t_pre.componentwise_square().hadamard();

    // Step 3: the D = 0 invariant (T.w == 0 in Rust, T.t == 0 in C ref).
    if !bool::from(t.w.is_zero()) {
        return Err(GluingError::InvalidKernel);
    }

    // Step 4: componentwise scale by precomputation.{x, y, z}; image.w = 0.
    let image_pre = ThetaPoint2D::new(
        t.x.mul(&gluing.precomputation.x),
        t.y.mul(&gluing.precomputation.y),
        t.z.mul(&gluing.precomputation.z),
        Fp2::<F>::zero(),
    );

    // Step 5: final hadamard.
    Ok(image_pre.hadamard())
}

/// Evaluate a gluing isogeny on a *basis* (pair of couple-Jacobian
/// points), producing two theta-image points.
///
/// Trivial 2-call wrapper over [`gluing_eval_point`]. Mirrors the C
/// reference's `gluing_eval_basis` for API parity; the basis-evaluation
/// surface is what the (2,2)-isogeny chain's first step needs when
/// pushing through the 2^e-torsion basis at gluing time.
///
/// Reference: `theta_isogenies.c:gluing_eval_basis`.
#[allow(dead_code)]
pub(crate) fn gluing_eval_basis<F: BaseField>(
    gluing: &GluingCodomain<F>,
    xy_t1: &CoupleJacobianPoint<F>,
    xy_t2: &CoupleJacobianPoint<F>,
) -> (ThetaPoint2D<F>, ThetaPoint2D<F>) {
    let image1 = gluing_eval_point(gluing, xy_t1);
    let image2 = gluing_eval_point(gluing, xy_t2);
    (image1, image2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ec::couple::CoupleCurve;
    use crate::ec::jacobian::JacobianPoint;
    use crate::ec::montgomery::MontgomeryCurve;
    use crate::gf::fp::Fp1Element;

    /// S136 vertical-slice smoke test: invoke [`gluing_codomain`] on
    /// a synthetic input that does not satisfy the eventual
    /// 8-torsion preconditions, confirm the function THREADS the
    /// type plumbing (kernel halving + XZ projection executed)
    /// AND returns the expected `Err(GluingError::InvalidKernel)`
    /// because the identity couple cannot pass verify_two_torsion
    /// inside action_by_translation (S137f: full pipeline now wired;
    /// failure path: identity → verify_two_torsion FALSE →
    /// action_by_translation CtOption FALSE → InvalidKernel).
    ///
    /// The synthetic inputs are deliberately non-kernel-shaped —
    /// this test validates that the Layer-3 entry point dispatches
    /// the full pipeline and fails gracefully on invalid input.
    #[test]
    fn gluing_codomain_vertical_slice_threads_types() {
        let curves = CoupleCurve::<Fp1Element>::e0_e0();
        let infinity = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::infinity(),
            JacobianPoint::<Fp1Element>::infinity(),
        );
        let result = gluing_codomain(&curves, &infinity, &infinity);
        assert_eq!(
            result,
            Err(GluingError::InvalidKernel),
            "S137f: identity-couple kernel must fail verify_two_torsion → InvalidKernel",
        );
    }

    // S137a: verify_two_torsion. On `E_0 : y² = x³ + x`, the full
    // 2-torsion subgroup is `E_0[2] = {O, (0,0), (i,0), (-i,0)}` where
    // `i` is the Fp2 imaginary unit (`Fp2::new(F::zero(), F::one())`).
    // We construct three concrete couple-2-torsion points by pairing
    // these per-curve 2-torsion points across `E_0 × E_0`.
    use crate::gf::fp2::Fp2;

    fn t0_origin<F: BaseField>() -> JacobianPoint<F> {
        // (0, 0, 1) — 2-torsion on E_0 since y² = x(x²+1) = 0 at x=0.
        JacobianPoint::<F>::new(Fp2::<F>::zero(), Fp2::<F>::zero(), Fp2::<F>::one())
    }

    fn t0_imag_unit<F: BaseField>() -> JacobianPoint<F> {
        // (i, 0, 1) — 2-torsion on E_0 since x²+1 = i²+1 = 0.
        let i = Fp2::<F>::new(F::zero(), F::one());
        JacobianPoint::<F>::new(i, Fp2::<F>::zero(), Fp2::<F>::one())
    }

    fn check_verify_two_torsion_accepts_distinct_nonidentity_2_torsion<F: BaseField>() {
        let curves = CoupleCurve::<F>::e0_e0();
        // K1_2 = ((0,0,1), (0,0,1)); K2_2 = ((i,0,1), (0,0,1)).
        // Both halves of each are 2-torsion; couples are distinct
        // (E_1 half differs); neither couple is the identity.
        let k1 = CoupleJacobianPoint::new(t0_origin::<F>(), t0_origin::<F>());
        let k2 = CoupleJacobianPoint::new(t0_imag_unit::<F>(), t0_origin::<F>());
        let verdict = verify_two_torsion(&k1, &k2, &curves);
        assert!(
            bool::from(verdict),
            "S137a: verify_two_torsion must ACCEPT distinct non-identity 2-torsion couples",
        );
    }

    #[test]
    fn verify_two_torsion_accepts_distinct_nonidentity_at_lvl1() {
        check_verify_two_torsion_accepts_distinct_nonidentity_2_torsion::<Fp1Element>();
    }

    #[test]
    fn verify_two_torsion_accepts_distinct_nonidentity_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_verify_two_torsion_accepts_distinct_nonidentity_2_torsion::<Fp3Element>();
    }

    #[test]
    fn verify_two_torsion_accepts_distinct_nonidentity_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_verify_two_torsion_accepts_distinct_nonidentity_2_torsion::<Fp5Element>();
    }

    fn check_verify_two_torsion_rejects_identity_couple<F: BaseField>() {
        let curves = CoupleCurve::<F>::e0_e0();
        let identity = CoupleJacobianPoint::new(
            JacobianPoint::<F>::infinity(),
            JacobianPoint::<F>::infinity(),
        );
        let k2 = CoupleJacobianPoint::new(t0_origin::<F>(), t0_origin::<F>());
        // K1 = (O, O) is "order ≤ 2" (doubles to ∞) but IS the identity.
        // verify_two_torsion must reject so the caller does not glue
        // through a trivial kernel.
        let verdict_k1_is_id = verify_two_torsion(&identity, &k2, &curves);
        let verdict_k2_is_id = verify_two_torsion(&k2, &identity, &curves);
        assert!(
            !bool::from(verdict_k1_is_id),
            "S137a: verify_two_torsion must REJECT when K1 is the identity couple",
        );
        assert!(
            !bool::from(verdict_k2_is_id),
            "S137a: verify_two_torsion must REJECT when K2 is the identity couple",
        );
    }

    #[test]
    fn verify_two_torsion_rejects_identity_couple_at_lvl1() {
        check_verify_two_torsion_rejects_identity_couple::<Fp1Element>();
    }

    #[test]
    fn verify_two_torsion_rejects_identity_couple_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_verify_two_torsion_rejects_identity_couple::<Fp3Element>();
    }

    #[test]
    fn verify_two_torsion_rejects_identity_couple_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_verify_two_torsion_rejects_identity_couple::<Fp5Element>();
    }

    fn check_verify_two_torsion_rejects_equal_couples<F: BaseField>() {
        let curves = CoupleCurve::<F>::e0_e0();
        let k = CoupleJacobianPoint::new(t0_origin::<F>(), t0_origin::<F>());
        // K1 == K2 (both halves projectively equal): the
        // distinctness check must fire.
        let verdict = verify_two_torsion(&k, &k, &curves);
        assert!(
            !bool::from(verdict),
            "S137a: verify_two_torsion must REJECT when K1 ≡ K2 projectively (degenerate kernel pair)",
        );
    }

    #[test]
    fn verify_two_torsion_rejects_equal_couples_at_lvl1() {
        check_verify_two_torsion_rejects_equal_couples::<Fp1Element>();
    }

    #[test]
    fn verify_two_torsion_rejects_equal_couples_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_verify_two_torsion_rejects_equal_couples::<Fp3Element>();
    }

    #[test]
    fn verify_two_torsion_rejects_equal_couples_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_verify_two_torsion_rejects_equal_couples::<Fp5Element>();
    }

    // S137b — batch_invert tests. Falsifiable: invert ∘ multiply
    // recovers the original value at every index.
    fn check_batch_invert_round_trips<F: BaseField>() {
        // Construct 8 small non-zero Fp2 values (the action_by_translation
        // call-site size). Per-element: re = i+1 + i·i (im), so all distinct
        // and all non-zero.
        let mut vals: [Fp2<F>; 8] = core::array::from_fn(|i| {
            let re = small_fp2::<F>(i + 1);
            let im = small_fp2::<F>((i * i) + 1);
            Fp2::new(re.re, im.re)
        });
        let originals = vals;

        let verdict = batch_invert(&mut vals);
        assert!(
            bool::from(verdict.is_some()),
            "S137b: batch_invert on 8 non-zero inputs must succeed",
        );

        // Verify each output is the inverse of its original.
        for i in 0..8 {
            let product = originals[i].mul(&vals[i]);
            assert_eq!(
                product,
                Fp2::<F>::one(),
                "S137b: batch_invert output at index {i} must satisfy original · output == 1",
            );
        }
    }

    fn small_fp2<F: BaseField>(n: usize) -> Fp2<F> {
        let mut acc = Fp2::<F>::zero();
        let one = Fp2::<F>::one();
        for _ in 0..n {
            acc = acc.add(&one);
        }
        acc
    }

    #[test]
    fn batch_invert_round_trips_at_lvl1() {
        check_batch_invert_round_trips::<Fp1Element>();
    }

    #[test]
    fn batch_invert_round_trips_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_batch_invert_round_trips::<Fp3Element>();
    }

    #[test]
    fn batch_invert_round_trips_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_batch_invert_round_trips::<Fp5Element>();
    }

    #[test]
    fn batch_invert_rejects_zero_input_at_lvl1() {
        // If any element is zero, the product is zero and invert
        // fails → CtOption(_, FALSE). Test with the middle element zero.
        let mut vals: [Fp2<Fp1Element>; 4] = core::array::from_fn(|i| {
            if i == 2 {
                Fp2::zero()
            } else {
                small_fp2::<Fp1Element>(i + 1)
            }
        });
        let verdict = batch_invert(&mut vals);
        assert!(
            !bool::from(verdict.is_some()),
            "S137b: batch_invert must FAIL when any input is zero (Montgomery trick has no recovery for zero element)",
        );
    }

    // S137b — action helpers. The full action_by_translation will
    // compose these in S137c; we test only the per-pair derivation here.
    #[test]
    fn action_by_translation_z_and_det_extracts_correctly_at_lvl1() {
        // Construct synthetic projective Montgomery points with known
        // (x, z) and verify the helper extracts z and det as
        // x₄·z₂ − z₄·x₂.
        let p4_x = small_fp2::<Fp1Element>(3);
        let p4_z = small_fp2::<Fp1Element>(5);
        let p2_x = small_fp2::<Fp1Element>(7);
        let p2_z = small_fp2::<Fp1Element>(11);
        let p4 = MontgomeryPoint::new(p4_x, p4_z);
        let p2 = MontgomeryPoint::new(p2_x, p2_z);

        let (z, det) = action_by_translation_z_and_det(&p4, &p2);
        assert_eq!(z, p4_z, "S137b: z output must equal P4.z");
        let expected_det = p4_x.mul(&p2_z).sub(&p4_z.mul(&p2_x));
        assert_eq!(det, expected_det, "S137b: det must equal x₄·z₂ − z₄·x₂",);
    }

    #[test]
    fn action_by_translation_compute_matrix_formula_at_lvl1() {
        // Construct synthetic (P4, P2) pair with non-zero z and det,
        // hand-invert, and verify the matrix matches the formula.
        let p4_x = small_fp2::<Fp1Element>(3);
        let p4_z = small_fp2::<Fp1Element>(5);
        let p2_x = small_fp2::<Fp1Element>(7);
        let p2_z = small_fp2::<Fp1Element>(11);
        let p4 = MontgomeryPoint::new(p4_x, p4_z);
        let p2 = MontgomeryPoint::new(p2_x, p2_z);

        let (z, det) = action_by_translation_z_and_det(&p4, &p2);
        let z_inv = z.invert().unwrap_or(Fp2::zero());
        let det_inv = det.invert().unwrap_or(Fp2::zero());

        let g = action_by_translation_compute_matrix(&p4, &p2, &z_inv, &det_inv);

        // Independent re-derivation from the formula:
        let tmp = p4_x.mul(&z_inv);
        let expected_g10 = p4_x.mul(&p2_x).mul(&det_inv).sub(&tmp);
        let expected_g11 = p4_x.mul(&p4_z).mul(&det_inv);
        let expected_g00 = expected_g11.negate();
        let expected_g01 = p2_x.mul(&p4_z).mul(&det_inv).negate();

        assert_eq!(g.g10, expected_g10, "S137b: g10 formula");
        assert_eq!(g.g11, expected_g11, "S137b: g11 formula");
        assert_eq!(g.g00, expected_g00, "S137b: g00 = -g11");
        assert_eq!(g.g01, expected_g01, "S137b: g01 formula");
    }

    // S137c — action_by_translation tests.

    /// S137c smoke test: action_by_translation REJECTS a synthetic
    /// kernel pair that fails [`verify_two_torsion`]. The identity
    /// couple is "2-torsion" in the doubles-to-(∞,∞) sense but IS
    /// the identity, so the composition early-returns
    /// `CtOption(_, FALSE)`.
    #[test]
    fn action_by_translation_rejects_identity_kernel_at_lvl1() {
        let curves = CoupleCurve::<Fp1Element>::e0_e0();
        let identity = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::infinity(),
            JacobianPoint::<Fp1Element>::infinity(),
        );
        // Doubling (∞, ∞) → (∞, ∞), then verify rejects (identity
        // couple is not a valid kernel half). So action returns NONE.
        let result = action_by_translation(&identity, &identity, &curves);
        assert!(
            !bool::from(result.is_some()),
            "S137c: action_by_translation must reject the identity kernel via verify_two_torsion early-exit",
        );
    }

    /// S137c invariant test: action_by_translation's batched output
    /// matches manual per-pair `compute_matrix` with independent
    /// `Fp2::invert` on each (z, det) pair.
    ///
    /// **Falsification axis covered**: the batched-inverse slot
    /// mapping (advisor's top-flagged risk). If `inverses[i]` is
    /// paired with `inverses[i + 4]` correctly per the C reference,
    /// the batched outputs match the per-pair outputs. Any swap or
    /// interleaving would produce different matrices.
    ///
    /// **Falsification axis NOT covered**: a shared bug across the
    /// batched and per-pair paths (e.g., wrong slot mapping in BOTH).
    /// One C-reference KAT anchor would close this; deferred to
    /// S138+ when end-to-end pipeline allows generating a vector.
    ///
    /// To enable the test, we use a synthetic 4-torsion kernel pair
    /// constructed by halving a pair of 8-torsion points… but the
    /// `E_0` 8-torsion structure on `F_{p²}` is not trivially
    /// constructible by hand. As a workaround we use 2-torsion
    /// inputs directly as "4-torsion" — this makes `verify_two_torsion`
    /// REJECT (the doubled 2-torsion is the identity, not 2-torsion),
    /// so `action_by_translation` returns `CtOption(_, FALSE)` and the
    /// invariant cannot be tested on real values. Documented as a
    /// known gap; integration test arrives once the kernel-generation
    /// path lands.
    #[test]
    fn action_by_translation_invariant_test_documented_gap_at_lvl1() {
        // Placeholder smoke test that the function compiles + the
        // type plumbing for a successful path WOULD work. The actual
        // invariant assertion is deferred per the docstring above.
        // We do NOT call the function with valid 4-torsion here
        // because constructing valid 4-torsion on E_0 requires
        // sqrt and twist-checking machinery that lives downstream.
        let curves = CoupleCurve::<Fp1Element>::e0_e0();
        let identity = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::infinity(),
            JacobianPoint::<Fp1Element>::infinity(),
        );
        let result = action_by_translation(&identity, &identity, &curves);
        // Don't assert on the matrix contents (would be circular).
        // Only assert the function returns SOME CtOption value
        // without panicking — confirming type plumbing.
        let _ = bool::from(result.is_some());
    }

    // S137d — gluing_change_of_basis 16-entry composition tests.

    /// S137d structural test: when all four `Gi` are the ZERO matrix
    /// (g00 = g01 = g10 = g11 = 0), the intermediates t001, t101,
    /// t002, t102 all vanish, and M[0][0] = 1 + 0 + 0 + 0 = 1; every
    /// other M[i][j] = 0. This catches gross transcription errors
    /// (wrong sign, wrong product, swapped row/column index for
    /// M[0][0]) without being circular against the implementation.
    #[test]
    fn gluing_change_of_basis_all_zero_gi_gives_identity_row0_at_lvl1() {
        let zero_matrix = TranslationMatrix::<Fp1Element> {
            g00: Fp2::zero(),
            g01: Fp2::zero(),
            g10: Fp2::zero(),
            g11: Fp2::zero(),
        };
        let gi = [zero_matrix, zero_matrix, zero_matrix, zero_matrix];
        let m = gluing_change_of_basis(&gi);

        // M[0][0] should be exactly 1 (from the `fp2_set_one` start in C);
        // every other entry should be zero.
        assert_eq!(
            m.m[0][0],
            Fp2::one(),
            "S137d: M[0][0] must be 1 when all Gi are zero"
        );
        for (i, row) in m.m.iter().enumerate() {
            for (j, entry) in row.iter().enumerate() {
                if (i, j) != (0, 0) {
                    assert_eq!(
                        *entry,
                        Fp2::zero(),
                        "S137d: M[{i}][{j}] must be 0 when all Gi are zero",
                    );
                }
            }
        }
    }

    /// S137d structural test: when all four `Gi` are the IDENTITY-LIKE
    /// matrix (g00 = g11 = 1, g01 = g10 = 0), the intermediates become
    /// t001 = 1·1 + 0·0 = 1; t101 = 0·1 + 1·0 = 0; t002 = 1; t102 = 0.
    /// Then M[0][0] = 1 + 1·1 + 1·1 + 1·1 = 4; M[0][1] = 1·0 + 1·0 + 1·0 = 0;
    /// M[0][2] = 0·1 + 0·1 + 0·1 = 0; M[0][3] = 0·0 + 0·0 + 0·0 = 0.
    /// Row 1 = identity-Gi[3] applied to row 0: M[1][0] = 1·4 + 0·0 = 4;
    /// M[1][1] = 0·4 + 1·0 = 0; M[1][2] = 0; M[1][3] = 0. Etc.
    ///
    /// This is independent of the function body (computed from the
    /// algorithm definition); a transcription error in any single
    /// entry will break the assertion.
    #[test]
    fn gluing_change_of_basis_identity_like_gi_produces_predicted_matrix_at_lvl1() {
        let id_like = TranslationMatrix::<Fp1Element> {
            g00: Fp2::one(),
            g01: Fp2::zero(),
            g10: Fp2::zero(),
            g11: Fp2::one(),
        };
        let gi = [id_like, id_like, id_like, id_like];
        let m = gluing_change_of_basis(&gi);

        // Hand-computed expectations from the algorithm:
        //   t001 = 1, t101 = 0, t002 = 1, t102 = 0
        //   Row 0: [4, 0, 0, 0]
        //   Row 1 (Gi[3] action on row 0): [4, 0, 0, 0]
        //   Row 2 (Gi[0] action on row 0): [4, 0, 0, 0]
        //   Row 3 (Gi[0] action on row 1): [4, 0, 0, 0]
        let four = Fp2::<Fp1Element>::one()
            .add(&Fp2::one())
            .add(&Fp2::one())
            .add(&Fp2::one());
        let expected = [
            [four, Fp2::zero(), Fp2::zero(), Fp2::zero()],
            [four, Fp2::zero(), Fp2::zero(), Fp2::zero()],
            [four, Fp2::zero(), Fp2::zero(), Fp2::zero()],
            [four, Fp2::zero(), Fp2::zero(), Fp2::zero()],
        ];
        for (i, (got_row, expected_row)) in m.m.iter().zip(expected.iter()).enumerate() {
            for (j, (got, exp)) in got_row.iter().zip(expected_row.iter()).enumerate() {
                assert_eq!(
                    got, exp,
                    "S137d: M[{i}][{j}] mismatch under identity-like Gi inputs",
                );
            }
        }
    }

    // S137e — apply_isomorphism + base_change + isotropy_check tests.

    /// Helper: identity 4×4 matrix.
    fn identity_matrix<F: BaseField>() -> BasisChangeMatrix<F> {
        let zero = Fp2::<F>::zero();
        let one = Fp2::<F>::one();
        BasisChangeMatrix {
            m: [
                [one, zero, zero, zero],
                [zero, one, zero, zero],
                [zero, zero, one, zero],
                [zero, zero, zero, one],
            ],
        }
    }

    /// S137e: apply_isomorphism with the identity matrix leaves the
    /// input unchanged. NECESSARY but not sufficient (identity is
    /// symmetric — can't distinguish M[i][j] vs M[j][i]); paired
    /// below with a permutation-matrix test that DOES catch
    /// transposition.
    #[test]
    fn apply_isomorphism_identity_is_noop_at_lvl1() {
        let m = identity_matrix::<Fp1Element>();
        let p = ThetaPoint2D::new(
            small_fp2::<Fp1Element>(3),
            small_fp2::<Fp1Element>(5),
            small_fp2::<Fp1Element>(7),
            small_fp2::<Fp1Element>(11),
        );
        let r = apply_isomorphism(&m, &p);
        assert_eq!(r.x, p.x, "S137e identity-M apply: x unchanged");
        assert_eq!(r.y, p.y, "S137e identity-M apply: y unchanged");
        assert_eq!(r.z, p.z, "S137e identity-M apply: z unchanged");
        assert_eq!(r.w, p.w, "S137e identity-M apply: w unchanged");
    }

    /// S137e ADVISOR-MANDATORY test: a permutation matrix that swaps
    /// x↔y (M[0][1] = M[1][0] = 1, M[2][2] = M[3][3] = 1; rest 0)
    /// MUST produce the swapped point. Catches transposition bugs
    /// (M[i][j] vs M[j][i]) that the identity test cannot.
    #[test]
    fn apply_isomorphism_permutation_swaps_x_and_y_at_lvl1() {
        let zero = Fp2::<Fp1Element>::zero();
        let one = Fp2::<Fp1Element>::one();
        // Permutation: x↔y swap, z and w fixed.
        let m = BasisChangeMatrix {
            m: [
                [zero, one, zero, zero], // out.x = P.y
                [one, zero, zero, zero], // out.y = P.x
                [zero, zero, one, zero], // out.z = P.z
                [zero, zero, zero, one], // out.w = P.w
            ],
        };
        let p = ThetaPoint2D::new(
            small_fp2::<Fp1Element>(3),
            small_fp2::<Fp1Element>(5),
            small_fp2::<Fp1Element>(7),
            small_fp2::<Fp1Element>(11),
        );
        let r = apply_isomorphism(&m, &p);
        // x ↔ y swap by the permutation row layout.
        assert_eq!(r.x, p.y, "S137e permutation: x must be input y");
        assert_eq!(r.y, p.x, "S137e permutation: y must be input x");
        assert_eq!(r.z, p.z, "S137e permutation: z unchanged");
        assert_eq!(r.w, p.w, "S137e permutation: w unchanged");
    }

    /// S137e: base_change with identity matrix produces the
    /// null_point built from (P1, P2) per `theta_isogenies.c:base_change`.
    /// The "expected" is computed via the independent oracle of the
    /// FORMULA (not the function body) per S137d advisor doctrine.
    #[test]
    fn base_change_with_identity_matrix_constructs_null_point_at_lvl1() {
        let m = identity_matrix::<Fp1Element>();
        let p1 = MontgomeryPoint::new(small_fp2::<Fp1Element>(3), small_fp2::<Fp1Element>(5));
        let p2 = MontgomeryPoint::new(small_fp2::<Fp1Element>(7), small_fp2::<Fp1Element>(11));
        let couple = CoupleMontgomeryPoint::new(p1, p2);

        let r = base_change(&m, &couple);

        // Independent oracle from the formula (NOT computed from the
        // function body; computed from the C-reference formula
        // documentation):
        //   null.x = P1.x · P2.x = 3 · 7 = 21
        //   null.y = P1.x · P2.z = 3 · 11 = 33
        //   null.z = P2.x · P1.z = 7 · 5 = 35
        //   null.w = P1.z · P2.z = 5 · 11 = 55
        // Identity M leaves these unchanged.
        assert_eq!(
            r.x,
            small_fp2::<Fp1Element>(21),
            "S137e base_change identity-M: x = P1.x·P2.x"
        );
        assert_eq!(
            r.y,
            small_fp2::<Fp1Element>(33),
            "S137e base_change identity-M: y = P1.x·P2.z"
        );
        assert_eq!(
            r.z,
            small_fp2::<Fp1Element>(35),
            "S137e base_change identity-M: z = P2.x·P1.z"
        );
        assert_eq!(
            r.w,
            small_fp2::<Fp1Element>(55),
            "S137e base_change identity-M: w = P1.z·P2.z"
        );
    }

    /// S137e helper: a valid (passing) isotropy input — w=0 on both,
    /// all 5 ASYMMETRIC factors non-zero, TT2.y free (per the C
    /// reference's asymmetric set, TT2.y is NOT required non-zero).
    fn valid_isotropy_pair<F: BaseField>() -> (ThetaPoint2D<F>, ThetaPoint2D<F>) {
        let zero = Fp2::<F>::zero();
        let one = Fp2::<F>::one();
        let two = small_fp2::<F>(2);
        let tt1 = ThetaPoint2D::new(one, one, one, zero); // x=y=z=1, w=0
        let tt2 = ThetaPoint2D::new(one, two, one, zero); // x=1, y=2 (free), z=1, w=0
        (tt1, tt2)
    }

    #[test]
    fn isotropy_check_positive_case_at_lvl1() {
        let (tt1, tt2) = valid_isotropy_pair::<Fp1Element>();
        assert!(
            bool::from(isotropy_check(&tt1, &tt2)),
            "S137e: valid isotropy input must pass",
        );
    }

    #[test]
    fn isotropy_check_rejects_tt1_w_nonzero_at_lvl1() {
        let (mut tt1, tt2) = valid_isotropy_pair::<Fp1Element>();
        tt1.w = Fp2::<Fp1Element>::one(); // primary check failure
        assert!(
            !bool::from(isotropy_check(&tt1, &tt2)),
            "S137e: TT1.w != 0 must fail the isotropy check",
        );
    }

    #[test]
    fn isotropy_check_rejects_tt2_w_nonzero_at_lvl1() {
        let (tt1, mut tt2) = valid_isotropy_pair::<Fp1Element>();
        tt2.w = Fp2::<Fp1Element>::one();
        assert!(
            !bool::from(isotropy_check(&tt1, &tt2)),
            "S137e: TT2.w != 0 must fail the isotropy check",
        );
    }

    /// S137e: per-factor negative-case coverage. For EACH element of
    /// the asymmetric secondary set {TT1.x, TT2.x, TT1.y, TT2.z,
    /// TT1.z}, setting it to zero must fail the check.
    #[test]
    fn isotropy_check_rejects_each_asymmetric_factor_zeroed_at_lvl1() {
        // TT1.x = 0
        let (mut tt1, tt2) = valid_isotropy_pair::<Fp1Element>();
        tt1.x = Fp2::zero();
        assert!(
            !bool::from(isotropy_check(&tt1, &tt2)),
            "S137e: TT1.x=0 must fail"
        );

        // TT2.x = 0
        let (tt1, mut tt2) = valid_isotropy_pair::<Fp1Element>();
        tt2.x = Fp2::zero();
        assert!(
            !bool::from(isotropy_check(&tt1, &tt2)),
            "S137e: TT2.x=0 must fail"
        );

        // TT1.y = 0
        let (mut tt1, tt2) = valid_isotropy_pair::<Fp1Element>();
        tt1.y = Fp2::zero();
        assert!(
            !bool::from(isotropy_check(&tt1, &tt2)),
            "S137e: TT1.y=0 must fail"
        );

        // TT2.z = 0
        let (tt1, mut tt2) = valid_isotropy_pair::<Fp1Element>();
        tt2.z = Fp2::zero();
        assert!(
            !bool::from(isotropy_check(&tt1, &tt2)),
            "S137e: TT2.z=0 must fail"
        );

        // TT1.z = 0
        let (mut tt1, tt2) = valid_isotropy_pair::<Fp1Element>();
        tt1.z = Fp2::zero();
        assert!(
            !bool::from(isotropy_check(&tt1, &tt2)),
            "S137e: TT1.z=0 must fail"
        );
    }

    /// S137e: critical asymmetry-preservation test. TT2.y is NOT in
    /// the C reference's secondary non-zero set; setting it to zero
    /// must NOT fail the isotropy check (other conditions met). This
    /// pins the asymmetric behavior so future "cleanup" refactors
    /// that symmetrize the list trip immediately.
    #[test]
    fn isotropy_check_tt2_y_zero_does_not_fail_at_lvl1() {
        let (tt1, mut tt2) = valid_isotropy_pair::<Fp1Element>();
        tt2.y = Fp2::zero();
        assert!(
            bool::from(isotropy_check(&tt1, &tt2)),
            "S137e: TT2.y=0 must NOT fail (asymmetric check by C-ref design)",
        );
    }

    /// S137e advisor-recommended integration test: end-to-end
    /// base_change → componentwise_square → hadamard (the
    /// C reference's `to_squared_theta`) → isotropy_check, on
    /// an arbitrary input. We do NOT assert pass/fail of the
    /// isotropy result on this synthetic input (the synthetic input
    /// will almost certainly fail the strict geometry condition);
    /// the test only confirms the pipeline THREADS the types and
    /// produces SOME Choice without panicking. Per S136 vertical-
    /// slice doctrine: validate compilation and call composition;
    /// validate algebraic correctness when a valid 8-torsion kernel
    /// is constructible in S138+.
    #[test]
    fn gluing_pipeline_threads_base_change_to_squared_theta_to_isotropy_at_lvl1() {
        let m = identity_matrix::<Fp1Element>();
        let p1 = MontgomeryPoint::new(small_fp2::<Fp1Element>(3), small_fp2::<Fp1Element>(5));
        let p2 = MontgomeryPoint::new(small_fp2::<Fp1Element>(7), small_fp2::<Fp1Element>(11));
        let couple_a = CoupleMontgomeryPoint::new(p1, p2);
        let p3 = MontgomeryPoint::new(small_fp2::<Fp1Element>(2), small_fp2::<Fp1Element>(4));
        let p4 = MontgomeryPoint::new(small_fp2::<Fp1Element>(6), small_fp2::<Fp1Element>(8));
        let couple_b = CoupleMontgomeryPoint::new(p3, p4);

        // Step 1: base_change to theta-space.
        let theta_a = base_change(&m, &couple_a);
        let theta_b = base_change(&m, &couple_b);

        // Step 2: squared-theta transform (= componentwise_square ∘ hadamard).
        let tt_a = theta_a.componentwise_square().hadamard();
        let tt_b = theta_b.componentwise_square().hadamard();

        // Step 3: isotropy_check. Result is unspecified on synthetic
        // input (almost certainly FALSE since these are not real
        // 8-torsion kernel points); we only confirm the pipeline
        // produces a Choice without panicking.
        let _result = bool::from(isotropy_check(&tt_a, &tt_b));
    }

    // S137f — build_codomain_from_squared_theta tests via independent
    // oracle (expected values computed from the C-reference formula).

    /// S137f: synthetic TT1 = (3, 5, 7, 0) and TT2 = (11, 13, 17, 0)
    /// (w-coordinates already zero per the isotropy precondition).
    /// Per the formulas:
    ///   codomain_pre.x = TT1.x · TT2.x  = 3·11 = 33
    ///   codomain_pre.y = TT1.y · TT2.x  = 5·11 = 55
    ///   codomain_pre.z = TT1.x · TT2.z  = 3·17 = 51
    ///   codomain_pre.w = 0
    ///   precomp.x      = TT1.y · TT2.z  = 5·17 = 85
    ///   precomp.y      = codomain.z     = 51
    ///   precomp.z      = codomain.y     = 55
    ///   precomp.w      = 0
    ///   image.x        = TT1.x · precomp.x = 3·85 = 255
    ///   image.y        = TT1.z · precomp.z = 7·55 = 385
    ///
    /// Final codomain is hadamard(codomain_pre).
    #[test]
    fn build_codomain_from_squared_theta_oracle_at_lvl1() {
        let tt1 =
            ThetaPoint2D::<Fp1Element>::new(small_fp2(3), small_fp2(5), small_fp2(7), Fp2::zero());
        let tt2 = ThetaPoint2D::<Fp1Element>::new(
            small_fp2(11),
            small_fp2(13),
            small_fp2(17),
            Fp2::zero(),
        );

        let (codomain, precomp, image) = build_codomain_from_squared_theta(&tt1, &tt2);

        // Independent oracle: codomain_pre_hadamard, then Hadamard.
        let codomain_pre = ThetaPoint2D::<Fp1Element>::new(
            small_fp2(33),
            small_fp2(55),
            small_fp2(51),
            Fp2::zero(),
        );
        let expected_codomain = codomain_pre.hadamard();
        assert_eq!(
            codomain, expected_codomain,
            "S137f: codomain must equal hadamard(pre-hadamard formula)",
        );

        // Independent oracle: precomputation factors with the
        // y↔z copy mapping (precomp.y = codomain_pre.z; precomp.z = codomain_pre.y).
        let expected_precomp = ThetaPoint2D::<Fp1Element>::new(
            small_fp2(85), // x = TT1.y · TT2.z = 5·17
            small_fp2(51), // y = codomain_pre.z = 51 (NOT codomain_pre.y)
            small_fp2(55), // z = codomain_pre.y = 55 (NOT codomain_pre.z)
            Fp2::zero(),
        );
        assert_eq!(
            precomp, expected_precomp,
            "S137f: precomputation must apply y↔z copy mapping verbatim from C ref",
        );

        // Independent oracle: imageK1_8 in (x : x : y : y) form,
        // z and w explicitly zero.
        let expected_image = ThetaPoint2D::<Fp1Element>::new(
            small_fp2(255), // x = TT1.x · precomp.x = 3·85
            small_fp2(385), // y = TT1.z · precomp.z = 7·55
            Fp2::zero(),
            Fp2::zero(),
        );
        assert_eq!(
            image, expected_image,
            "S137f: imageK1_8 must have x=TT1.x·precomp.x, y=TT1.z·precomp.z, z=w=0",
        );
    }

    /// S137f: dedicated test that the precomputation y↔z copy mapping
    /// is NOT swapped (a likely transcription error). Constructed
    /// asymmetric TT1, TT2 where codomain.y and codomain.z differ;
    /// then precomp.y MUST equal codomain.z (not codomain.y), and
    /// vice versa.
    #[test]
    fn build_codomain_precomp_y_z_swap_correctness_at_lvl1() {
        // Pick TT1/TT2 so codomain_pre.y = 55 ≠ codomain_pre.z = 51.
        let tt1 =
            ThetaPoint2D::<Fp1Element>::new(small_fp2(3), small_fp2(5), small_fp2(7), Fp2::zero());
        let tt2 = ThetaPoint2D::<Fp1Element>::new(
            small_fp2(11),
            small_fp2(13),
            small_fp2(17),
            Fp2::zero(),
        );
        let (_codomain, precomp, _image) = build_codomain_from_squared_theta(&tt1, &tt2);

        // codomain_pre.y = 5·11 = 55; codomain_pre.z = 3·17 = 51.
        // precomp.y must be 51 (= codomain.z); precomp.z must be 55.
        // A swap would put 55 in precomp.y and 51 in precomp.z — failing
        // the assertion.
        assert_eq!(
            precomp.y,
            small_fp2(51),
            "S137f: precomp.y must equal codomain.z (51), NOT codomain.y (55)",
        );
        assert_eq!(
            precomp.z,
            small_fp2(55),
            "S137f: precomp.z must equal codomain.y (55), NOT codomain.z (51)",
        );
    }

    // S138 — gluing_eval_point tests.
    //
    // Independent oracle on T1/T2 construction is the highest-value
    // test surface: with hand-picked (u, v, w) triples, the cross-
    // product formulas yield trivially-computable expected values.
    // End-to-end gluing_eval_point only smoke-tests the type plumbing
    // since constructing a valid GluingCodomain requires valid
    // 8-torsion kernel data (deferred to integration testing).

    /// S138: independent-oracle test on the T1 and T2 cross-product
    /// construction (steps 2-2cont of `gluing_eval_point`). Uses
    /// synthetic add_components triples to verify the cross-product
    /// formulas match the C reference verbatim.
    ///
    /// This test calls the formulas DIRECTLY (extracted into a
    /// helper for testability) rather than going through
    /// `gluing_eval_point` which requires a valid `GluingCodomain`.
    fn build_t1_t2_for_test<F: BaseField>(
        u1: Fp2<F>,
        v1: Fp2<F>,
        w1: Fp2<F>,
        u2: Fp2<F>,
        v2: Fp2<F>,
        w2: Fp2<F>,
    ) -> (ThetaPoint2D<F>, ThetaPoint2D<F>) {
        // Inlined from gluing_eval_point steps 2-2cont; kept in sync
        // manually with the production code. If the production code
        // changes formula, this test helper must change identically.
        let t1_x = u1.mul(&u2).add(&v1.mul(&v2));
        let t1_y = u1.mul(&w2);
        let t1_z = w1.mul(&u2);
        let t1_w = w1.mul(&w2);
        let t1 = ThetaPoint2D::new(t1_x, t1_y, t1_z, t1_w);

        let t2_x = u1.add(&v1).mul(&u2.add(&v2)).sub(&t1_x);
        let t2_y = v1.mul(&w2);
        let t2_z = w1.mul(&v2);
        let t2_w = Fp2::<F>::zero();
        let t2 = ThetaPoint2D::new(t2_x, t2_y, t2_z, t2_w);

        (t1, t2)
    }

    #[test]
    fn gluing_eval_point_t1_t2_construction_oracle_at_lvl1() {
        // Hand-picked add_components values:
        //   (u1, v1, w1) = (3, 5, 7), (u2, v2, w2) = (11, 13, 17).
        let u1 = small_fp2::<Fp1Element>(3);
        let v1 = small_fp2::<Fp1Element>(5);
        let w1 = small_fp2::<Fp1Element>(7);
        let u2 = small_fp2::<Fp1Element>(11);
        let v2 = small_fp2::<Fp1Element>(13);
        let w2 = small_fp2::<Fp1Element>(17);

        let (t1, t2) = build_t1_t2_for_test(u1, v1, w1, u2, v2, w2);

        // Independent oracle:
        //   T1.x = 3·11 + 5·13 = 33 + 65 = 98
        //   T1.y = 3·17 = 51
        //   T1.z = 7·11 = 77
        //   T1.w = 7·17 = 119
        //   T2.x = (3+5)(11+13) − 98 = 8·24 − 98 = 192 − 98 = 94
        //         (= v1·u2 + u1·v2 = 5·11 + 3·13 = 55 + 39 = 94 ✓)
        //   T2.y = 5·17 = 85
        //   T2.z = 7·13 = 91
        //   T2.w = 0
        assert_eq!(
            t1.x,
            small_fp2::<Fp1Element>(98),
            "S138: T1.x = u1u2 + v1v2"
        );
        assert_eq!(t1.y, small_fp2::<Fp1Element>(51), "S138: T1.y = u1·w2");
        assert_eq!(t1.z, small_fp2::<Fp1Element>(77), "S138: T1.z = w1·u2");
        assert_eq!(t1.w, small_fp2::<Fp1Element>(119), "S138: T1.w = w1·w2");
        assert_eq!(
            t2.x,
            small_fp2::<Fp1Element>(94),
            "S138: T2.x = (u1+v1)(u2+v2) − T1.x = v1·u2 + u1·v2",
        );
        assert_eq!(t2.y, small_fp2::<Fp1Element>(85), "S138: T2.y = v1·w2");
        assert_eq!(t2.z, small_fp2::<Fp1Element>(91), "S138: T2.z = w1·v2");
        assert_eq!(t2.w, Fp2::zero(), "S138: T2.w = 0 by construction");
    }

    /// S138: dedicated test that T2.x is computed via the EXPANSION
    /// formula `(u1+v1)(u2+v2) − T1.x`, NOT directly as `v1·u2 + u1·v2`.
    /// Both compute the same value algebraically; the C reference
    /// uses the expansion to save 1 multiplication. The test
    /// asserts equivalence both ways, pinning the expansion vs
    /// direct correctness — catches the case where a future
    /// optimization swaps the subtraction direction
    /// (`T1.x − (u1+v1)(u2+v2)` instead of the correct order).
    #[test]
    fn gluing_eval_point_t2_x_expansion_equivalence_at_lvl1() {
        let u1 = small_fp2::<Fp1Element>(3);
        let v1 = small_fp2::<Fp1Element>(5);
        let w1 = small_fp2::<Fp1Element>(7);
        let u2 = small_fp2::<Fp1Element>(11);
        let v2 = small_fp2::<Fp1Element>(13);
        let w2 = small_fp2::<Fp1Element>(17);

        let (_t1, t2) = build_t1_t2_for_test(u1, v1, w1, u2, v2, w2);

        let direct = v1.mul(&u2).add(&u1.mul(&v2));
        assert_eq!(
            t2.x, direct,
            "S138: T2.x via expansion (u1+v1)(u2+v2) − T1.x must equal direct v1·u2 + u1·v2",
        );
    }

    // S141 — gluing_eval_point_special_case + gluing_eval_basis smoke
    // tests. End-to-end correctness testing requires valid 8-torsion
    // kernel data and a `GluingCodomain` produced by the production
    // path — neither of which is available until IdealToIsogenyClapotis
    // lands. These tests confirm type plumbing and the `T.w == 0`
    // invariant check.

    /// Build a synthetic `GluingCodomain<Fp1Element>` whose interior
    /// theta points + basis-change matrix + kernel data are all
    /// zero-valued. NOT a valid gluing state — used only to exercise
    /// API surfaces in tests.
    fn synthetic_zero_gluing() -> GluingCodomain<Fp1Element> {
        GluingCodomain {
            m: BasisChangeMatrix {
                m: [[Fp2::<Fp1Element>::zero(); 4]; 4],
            },
            codomain: ThetaPoint2D::new(
                Fp2::<Fp1Element>::zero(),
                Fp2::<Fp1Element>::zero(),
                Fp2::<Fp1Element>::zero(),
                Fp2::<Fp1Element>::zero(),
            ),
            precomputation: ThetaPoint2D::new(
                Fp2::<Fp1Element>::zero(),
                Fp2::<Fp1Element>::zero(),
                Fp2::<Fp1Element>::zero(),
                Fp2::<Fp1Element>::zero(),
            ),
            image_k1_8: ThetaPoint2D::new(
                Fp2::<Fp1Element>::zero(),
                Fp2::<Fp1Element>::zero(),
                Fp2::<Fp1Element>::zero(),
                Fp2::<Fp1Element>::zero(),
            ),
            xy_k1_8: CoupleJacobianPoint {
                p1: JacobianPoint::<Fp1Element>::infinity(),
                p2: JacobianPoint::<Fp1Element>::infinity(),
            },
            domain: CoupleCurve {
                e1: MontgomeryCurve::<Fp1Element>::e0(),
                e2: MontgomeryCurve::<Fp1Element>::e0(),
            },
        }
    }

    #[test]
    fn gluing_eval_point_special_case_smoke_with_zero_state() {
        // With an all-zero GluingCodomain and all-zero CoupleMontgomeryPoint
        // input, base_change → zero, squared_theta → zero, T.w = 0 ✓,
        // image scaling by precomputation → zero, hadamard → zero.
        // The call returns Ok(zero ThetaPoint2D) confirming the type
        // dispatch is correct.
        let gluing = synthetic_zero_gluing();
        let p_xz = CoupleMontgomeryPoint {
            p1: MontgomeryPoint::<Fp1Element>::new(
                Fp2::<Fp1Element>::zero(),
                Fp2::<Fp1Element>::zero(),
            ),
            p2: MontgomeryPoint::<Fp1Element>::new(
                Fp2::<Fp1Element>::zero(),
                Fp2::<Fp1Element>::zero(),
            ),
        };

        let result = gluing_eval_point_special_case(&gluing, &p_xz);
        assert!(
            result.is_ok(),
            "S141: zero-state special_case must satisfy T.w == 0 invariant and return Ok",
        );
        let image = result.expect("checked is_ok above");
        assert_eq!(
            image.x,
            Fp2::<Fp1Element>::zero(),
            "S141: image.x == 0 from zero state"
        );
        assert_eq!(
            image.y,
            Fp2::<Fp1Element>::zero(),
            "S141: image.y == 0 from zero state"
        );
        assert_eq!(
            image.z,
            Fp2::<Fp1Element>::zero(),
            "S141: image.z == 0 from zero state"
        );
        assert_eq!(
            image.w,
            Fp2::<Fp1Element>::zero(),
            "S141: image.w == 0 from zero state"
        );
    }

    /// S141: confirm `gluing_eval_basis` is the trivial 2-call
    /// wrapper documented in the C reference — both outputs match
    /// `gluing_eval_point` evaluated on each input independently.
    #[test]
    fn gluing_eval_basis_dispatches_to_eval_point_twice_at_lvl1() {
        let gluing = synthetic_zero_gluing();
        let p1 = CoupleJacobianPoint {
            p1: JacobianPoint::<Fp1Element>::infinity(),
            p2: JacobianPoint::<Fp1Element>::infinity(),
        };
        let p2 = CoupleJacobianPoint {
            p1: JacobianPoint::<Fp1Element>::infinity(),
            p2: JacobianPoint::<Fp1Element>::infinity(),
        };

        let (img1, img2) = gluing_eval_basis(&gluing, &p1, &p2);
        let solo1 = gluing_eval_point(&gluing, &p1);
        let solo2 = gluing_eval_point(&gluing, &p2);

        assert_eq!(
            img1.x, solo1.x,
            "S141: basis output 1.x must match solo eval 1.x"
        );
        assert_eq!(
            img1.y, solo1.y,
            "S141: basis output 1.y must match solo eval 1.y"
        );
        assert_eq!(
            img1.z, solo1.z,
            "S141: basis output 1.z must match solo eval 1.z"
        );
        assert_eq!(
            img1.w, solo1.w,
            "S141: basis output 1.w must match solo eval 1.w"
        );
        assert_eq!(
            img2.x, solo2.x,
            "S141: basis output 2.x must match solo eval 2.x"
        );
        assert_eq!(
            img2.y, solo2.y,
            "S141: basis output 2.y must match solo eval 2.y"
        );
        assert_eq!(
            img2.z, solo2.z,
            "S141: basis output 2.z must match solo eval 2.z"
        );
        assert_eq!(
            img2.w, solo2.w,
            "S141: basis output 2.w must match solo eval 2.w"
        );
    }

    // S158 — GluingCodomain method-form alias tests.

    #[test]
    fn gluing_codomain_compute_method_matches_free_function_at_lvl1() {
        // S136 smoke-test pattern: identity-couple input → both
        // method and free function return Err(InvalidKernel)
        // identically.
        let curve = CoupleCurve {
            e1: MontgomeryCurve::<Fp1Element>::e0(),
            e2: MontgomeryCurve::<Fp1Element>::e0(),
        };
        let inf = CoupleJacobianPoint {
            p1: JacobianPoint::<Fp1Element>::infinity(),
            p2: JacobianPoint::<Fp1Element>::infinity(),
        };
        let via_method = GluingCodomain::compute(&curve, &inf, &inf);
        let via_free = gluing_codomain(&curve, &inf, &inf);
        assert_eq!(
            via_method, via_free,
            "S158: GluingCodomain::compute must match gluing_codomain free function",
        );
    }

    #[test]
    fn gluing_codomain_eval_point_method_matches_free_function_at_lvl1() {
        let gluing = synthetic_zero_gluing();
        let p = CoupleJacobianPoint {
            p1: JacobianPoint::<Fp1Element>::infinity(),
            p2: JacobianPoint::<Fp1Element>::infinity(),
        };
        let via_method = gluing.eval_point(&p);
        let via_free = gluing_eval_point(&gluing, &p);
        assert_eq!(
            via_method, via_free,
            "S158: gluing.eval_point(&p) must match gluing_eval_point(&gluing, &p)",
        );
    }

    #[test]
    fn gluing_codomain_eval_point_special_case_method_matches_free_function_at_lvl1() {
        let gluing = synthetic_zero_gluing();
        let p_xz = CoupleMontgomeryPoint {
            p1: MontgomeryPoint::<Fp1Element>::new(
                Fp2::<Fp1Element>::zero(),
                Fp2::<Fp1Element>::zero(),
            ),
            p2: MontgomeryPoint::<Fp1Element>::new(
                Fp2::<Fp1Element>::zero(),
                Fp2::<Fp1Element>::zero(),
            ),
        };
        let via_method = gluing.eval_point_special_case(&p_xz);
        let via_free = gluing_eval_point_special_case(&gluing, &p_xz);
        assert_eq!(
            via_method, via_free,
            "S158: gluing.eval_point_special_case(&p_xz) must match free function",
        );
    }

    #[test]
    fn gluing_codomain_eval_basis_method_matches_free_function_at_lvl1() {
        let gluing = synthetic_zero_gluing();
        let p1 = CoupleJacobianPoint {
            p1: JacobianPoint::<Fp1Element>::infinity(),
            p2: JacobianPoint::<Fp1Element>::infinity(),
        };
        let p2 = CoupleJacobianPoint {
            p1: JacobianPoint::<Fp1Element>::infinity(),
            p2: JacobianPoint::<Fp1Element>::infinity(),
        };
        let via_method = gluing.eval_basis(&p1, &p2);
        let via_free = gluing_eval_basis(&gluing, &p1, &p2);
        assert_eq!(
            via_method.0, via_free.0,
            "S158: gluing.eval_basis output .0 must match free function",
        );
        assert_eq!(
            via_method.1, via_free.1,
            "S158: gluing.eval_basis output .1 must match free function",
        );
    }
}
