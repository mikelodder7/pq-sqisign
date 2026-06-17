// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(rustdoc::private_intra_doc_links)]

//! `(2, 2)`-theta-coord point doubling using the C-reference's
//! TWO-BLOCK precomputation form.
//!
//! Mirrors `theta_structure.c:5-69` verbatim:
//!
//! - [`AbelianVarietyPrecomputed`] holds the theta-null point PLUS
//!   two cached `ThetaPoint2D<F>` blocks of 4 projective factors
//!   each (the "dual block" `{YZT0, XZT0, XYT0, XYZ0}` from the
//!   squared-theta transform of the null, and the "null block"
//!   `{yzt0, xzt0, xyt0, xyz0}` derived from the null directly).
//! - [`theta_precomputation`] is the constructor: given a theta-null,
//!   computes both blocks.
//! - [`double_point`] is the per-point doubling formula:
//!   `squared_theta → componentwise_square → multiply by dual_block
//!   → hadamard → multiply by null_block`. Five steps.
//! - [`double_iter`] iterates `double_point` `exp` times. Equivalent
//!   to scalar-multiplying by `2^exp`.
//!
//! # Coexistence with the Riemann-form doubling
//!
//! The [`crate::isogeny::theta::AbelianVariety2D::double`]
//! method uses a single-block doubling form (Riemann's duplication
//! formula with a single `doubling_constants` factor). The two forms
//! arise from different theta-coord normalization conventions. The
//! project docket noted this reconciliation as pending; this
//! session ships the C-ref-aligned TWO-BLOCK form as new
//! infrastructure for the chain-walker (`_theta_chain_compute_impl`).
//! The Riemann form coexists and is not touched.
//!
//! Per `theta_structure.h:to_squared_theta`, `squared_theta` =
//! `componentwise_square` then `hadamard`.

use crate::gf::fp::BaseField;
use crate::isogeny::theta::ThetaPoint2D;

/// Theta-coord Abelian variety with the C-reference's TWO-BLOCK
/// doubling precomputation pre-filled.
///
/// `dual_block` corresponds to `{YZT0, XZT0, XYT0, XYZ0}` (upper
/// block: derived from `squared_theta(theta_null)`).
/// `null_block` corresponds to `{yzt0, xzt0, xyt0, xyz0}` (lower
/// block: derived from `theta_null` directly).
///
/// The names `YZT0`, etc. come from the C reference's notation —
/// they're a labeling of 4 products of 3 of the 4 squared-theta
/// coordinates. The `0` suffix denotes "computed at the origin"
/// (the theta-null).
///
/// Stored as two [`ThetaPoint2D`] values for compact representation;
/// the `ThetaPoint2D` here is used as a generic 4-tuple of `Fp2`,
/// not as a theta-coord point per se.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AbelianVarietyPrecomputed<F: BaseField> {
    /// Theta-null point of the variety.
    pub(crate) theta_null: ThetaPoint2D<F>,
    /// Upper block: `(YZT0, XZT0, XYT0, XYZ0)` — products of 3
    /// squared-theta coordinates.
    pub(crate) dual_block: ThetaPoint2D<F>,
    /// Lower block: `(yzt0, xzt0, xyt0, xyz0)` — products of 3
    /// theta-null coordinates directly.
    pub(crate) null_block: ThetaPoint2D<F>,
}

impl<F: BaseField> AbelianVarietyPrecomputed<F> {
    /// Build an [`AbelianVarietyPrecomputed`] from a theta-null point.
    ///
    /// Method-form alias of [`theta_precomputation`] — same body,
    /// same result. Use whichever spelling reads cleaner at the
    /// call site.
    #[allow(dead_code)]
    pub(crate) fn new(theta_null: &ThetaPoint2D<F>) -> Self {
        theta_precomputation(theta_null)
    }

    /// Double a theta-coord point on this variety.
    ///
    /// Method-form alias of [`double_point`].
    #[allow(dead_code)]
    pub(crate) fn double_point(&self, p: &ThetaPoint2D<F>) -> ThetaPoint2D<F> {
        double_point(self, p)
    }

    /// Iterate [`Self::double_point`] `exp` times. `exp == 0`
    /// returns `p` unchanged.
    ///
    /// Method-form alias of [`double_iter`].
    #[allow(dead_code)]
    pub(crate) fn double_iter(&self, p: &ThetaPoint2D<F>, exp: usize) -> ThetaPoint2D<F> {
        double_iter(self, p, exp)
    }
}

/// Compute the TWO-BLOCK doubling precomputation for a theta-null.
///
/// Mirrors `theta_structure.c:theta_precomputation` verbatim:
///
/// ```text
/// dual = squared_theta(theta_null)   (= componentwise_square then hadamard)
/// t1 = dual.x · dual.y
/// t2 = dual.z · dual.w               (Rust w = C t)
///
/// dual_block (upper) = (YZT0, XZT0, XYT0, XYZ0):
///   YZT0 = t2 · dual.y
///   XZT0 = t2 · dual.x
///   XYT0 = t1 · dual.w
///   XYZ0 = t1 · dual.z
///
/// t1' = theta_null.x · theta_null.y
/// t2' = theta_null.z · theta_null.w
///
/// null_block (lower) = (yzt0, xzt0, xyt0, xyz0):
///   yzt0 = t2' · theta_null.y
///   xzt0 = t2' · theta_null.x
///   xyt0 = t1' · theta_null.w
///   xyz0 = t1' · theta_null.z
/// ```
///
/// Reference: `theta_structure.c:4-31`.
///
/// # See also
///
/// [`AbelianVarietyPrecomputed::new`] is the method-form alias of this
/// free function — same body, same result. Pick whichever spelling
/// reads cleaner at the call site.
#[allow(dead_code)]
pub(crate) fn theta_precomputation<F: BaseField>(
    theta_null: &ThetaPoint2D<F>,
) -> AbelianVarietyPrecomputed<F> {
    // squared_theta = componentwise_square then hadamard.
    let dual = theta_null.componentwise_square().hadamard();

    let t1 = dual.x.mul(&dual.y);
    let t2 = dual.z.mul(&dual.w);
    let dual_block = ThetaPoint2D::new(
        t2.mul(&dual.y), // YZT0
        t2.mul(&dual.x), // XZT0
        t1.mul(&dual.w), // XYT0
        t1.mul(&dual.z), // XYZ0
    );

    let t1p = theta_null.x.mul(&theta_null.y);
    let t2p = theta_null.z.mul(&theta_null.w);
    let null_block = ThetaPoint2D::new(
        t2p.mul(&theta_null.y), // yzt0
        t2p.mul(&theta_null.x), // xzt0
        t1p.mul(&theta_null.w), // xyt0
        t1p.mul(&theta_null.z), // xyz0
    );

    AbelianVarietyPrecomputed {
        theta_null: *theta_null,
        dual_block,
        null_block,
    }
}

/// Double a theta-coord point on the variety.
///
/// Mirrors `theta_structure.c:double_point` verbatim:
///
/// ```text
/// 1. out = squared_theta(in)           = componentwise_square then hadamard
/// 2. out = componentwise_square(out)
/// 3. out = componentwise mul(out, dual_block)
/// 4. out = hadamard(out)
/// 5. out = componentwise mul(out, null_block)
/// ```
///
/// The output is the theta-coord image of `2 · in` under the
/// theta-structure's doubling map.
///
/// Reference: `theta_structure.c:33-56`.
#[allow(dead_code)]
pub(crate) fn double_point<F: BaseField>(
    precomp: &AbelianVarietyPrecomputed<F>,
    p: &ThetaPoint2D<F>,
) -> ThetaPoint2D<F> {
    // Step 1: squared_theta(P).
    let step1 = p.componentwise_square().hadamard();

    // Step 2: componentwise square again.
    let step2 = step1.componentwise_square();

    // Step 3: componentwise mul by dual_block.
    let step3 = ThetaPoint2D::new(
        step2.x.mul(&precomp.dual_block.x),
        step2.y.mul(&precomp.dual_block.y),
        step2.z.mul(&precomp.dual_block.z),
        step2.w.mul(&precomp.dual_block.w),
    );

    // Step 4: hadamard.
    let step4 = step3.hadamard();

    // Step 5: componentwise mul by null_block.
    ThetaPoint2D::new(
        step4.x.mul(&precomp.null_block.x),
        step4.y.mul(&precomp.null_block.y),
        step4.z.mul(&precomp.null_block.z),
        step4.w.mul(&precomp.null_block.w),
    )
}

/// Iterate [`double_point`] `exp` times.
///
/// Equivalent to scalar-multiplying `in` by `2^exp` in the
/// theta-coord group. `exp = 0` returns `in` unchanged.
///
/// Reference: `theta_structure.c:58-69`.
#[allow(dead_code)]
pub(crate) fn double_iter<F: BaseField>(
    precomp: &AbelianVarietyPrecomputed<F>,
    p: &ThetaPoint2D<F>,
    exp: usize,
) -> ThetaPoint2D<F> {
    if exp == 0 {
        return *p;
    }
    let mut out = double_point(precomp, p);
    for _ in 1..exp {
        out = double_point(precomp, &out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf::fp::Fp1Element;
    use crate::gf::fp2::Fp2;

    fn small_fp2(n: u32) -> Fp2<Fp1Element> {
        let mut acc = Fp2::<Fp1Element>::zero();
        let one = Fp2::<Fp1Element>::one();
        for _ in 0..n {
            acc = acc.add(&one);
        }
        acc
    }

    fn neg_fp2(n: u32) -> Fp2<Fp1Element> {
        Fp2::<Fp1Element>::zero().sub(&small_fp2(n))
    }

    /// oracle: with theta_null = (2, 3, 5, 7), verify the
    /// computed dual_block and null_block match hand-computation.
    ///
    /// dual = squared_theta(2, 3, 5, 7):
    ///   square = (4, 9, 25, 49)
    ///   hadamard: x = 4+9+25+49 = 87
    ///             y = 4-9+25-49 = -29
    ///             z = 4+9-25-49 = -61
    ///             w = 4-9-25+49 = 19
    ///
    /// t1 = dual.x · dual.y = 87 · (-29) = -2523
    /// t2 = dual.z · dual.w = (-61) · 19 = -1159
    ///
    /// dual_block:
    ///   YZT0 = t2 · dual.y = (-1159) · (-29) = 33611
    ///   XZT0 = t2 · dual.x = (-1159) · 87 = -100833
    ///   XYT0 = t1 · dual.w = (-2523) · 19 = -47937
    ///   XYZ0 = t1 · dual.z = (-2523) · (-61) = 153903
    ///
    /// t1' = 2 · 3 = 6
    /// t2' = 5 · 7 = 35
    ///
    /// null_block:
    ///   yzt0 = t2' · null.y = 35 · 3 = 105
    ///   xzt0 = t2' · null.x = 35 · 2 = 70
    ///   xyt0 = t1' · null.w = 6 · 7 = 42
    ///   xyz0 = t1' · null.z = 6 · 5 = 30
    #[test]
    fn theta_precomputation_oracle_at_lvl1() {
        let theta_null = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        let precomp = theta_precomputation(&theta_null);

        assert_eq!(precomp.dual_block.x, small_fp2(33611), "YZT0 = 33611");
        assert_eq!(
            precomp.dual_block.y,
            neg_fp2(100833),
            "XZT0 = -100833"
        );
        assert_eq!(precomp.dual_block.z, neg_fp2(47937), "XYT0 = -47937");
        assert_eq!(
            precomp.dual_block.w,
            small_fp2(153903),
            "XYZ0 = 153903"
        );

        assert_eq!(precomp.null_block.x, small_fp2(105), "yzt0 = 105");
        assert_eq!(precomp.null_block.y, small_fp2(70), "xzt0 = 70");
        assert_eq!(precomp.null_block.z, small_fp2(42), "xyt0 = 42");
        assert_eq!(precomp.null_block.w, small_fp2(30), "xyz0 = 30");

        assert_eq!(
            precomp.theta_null, theta_null,
            "theta_null must be preserved on the precomputed struct",
        );
    }

    /// oracle: double_point on P = (1, 0, 0, 0) with
    /// theta_null = (2, 3, 5, 7).
    ///
    /// Step 1: squared_theta(P)
    ///   square = (1, 0, 0, 0)
    ///   hadamard = (1, 1, 1, 1)
    ///
    /// Step 2: componentwise square = (1, 1, 1, 1)
    ///
    /// Step 3: multiply by dual_block (from oracle above):
    ///   (33611, -100833, -47937, 153903)
    ///
    /// Step 4: hadamard:
    ///   x = 33611 + (-100833) + (-47937) + 153903 = 38744
    ///   y = 33611 - (-100833) + (-47937) - 153903 = 33611 + 100833 - 47937 - 153903 = -67396
    ///   z = 33611 + (-100833) - (-47937) - 153903 = -173188
    ///   w = 33611 - (-100833) - (-47937) + 153903 = 336284
    ///
    /// Step 5: multiply by null_block (105, 70, 42, 30):
    ///   x = 38744 · 105 = 4068120
    ///   y = -67396 · 70 = -4717720
    ///   z = -173188 · 42 = -7273896
    ///   w = 336284 · 30 = 10088520
    #[test]
    fn double_point_oracle_at_lvl1() {
        let theta_null = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        let precomp = theta_precomputation(&theta_null);
        let p = ThetaPoint2D::new(
            small_fp2(1),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
        );

        let r = double_point(&precomp, &p);

        assert_eq!(r.x, small_fp2(4068120), "2P.x = 4068120");
        assert_eq!(r.y, neg_fp2(4717720), "2P.y = -4717720");
        assert_eq!(r.z, neg_fp2(7273896), "2P.z = -7273896");
        assert_eq!(r.w, small_fp2(10088520), "2P.w = 10088520");
    }

    /// double_iter(0) returns input unchanged.
    #[test]
    fn double_iter_exp_zero_is_identity_at_lvl1() {
        let theta_null = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        let precomp = theta_precomputation(&theta_null);
        let p = ThetaPoint2D::new(small_fp2(11), small_fp2(13), small_fp2(17), small_fp2(19));

        let r = double_iter(&precomp, &p, 0);

        assert_eq!(r, p, "double_iter(0) = identity");
    }

    /// double_iter(1) matches single double_point.
    #[test]
    fn double_iter_exp_one_equals_double_point_at_lvl1() {
        let theta_null = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        let precomp = theta_precomputation(&theta_null);
        let p = ThetaPoint2D::new(
            small_fp2(1),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
        );

        let r_iter = double_iter(&precomp, &p, 1);
        let r_solo = double_point(&precomp, &p);

        assert_eq!(r_iter, r_solo, "double_iter(1) = double_point");
    }

    /// double_iter(2) composes — i.e., equals double_point applied
    /// twice. This is a definition test, not a semantic test (the
    /// semantic claim "double_iter(2)·P = 4·P" needs an external oracle
    /// like the C ref's KAT to validate).
    #[test]
    fn double_iter_exp_two_composes_at_lvl1() {
        let theta_null = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        let precomp = theta_precomputation(&theta_null);
        let p = ThetaPoint2D::new(small_fp2(11), small_fp2(13), small_fp2(17), small_fp2(19));

        let r_iter = double_iter(&precomp, &p, 2);
        let r_manual = double_point(&precomp, &double_point(&precomp, &p));

        assert_eq!(
            r_iter, r_manual,
            "double_iter(2) = double_point ∘ double_point"
        );
    }

    // method-form ergonomic API tests.

    #[test]
    fn new_method_matches_theta_precomputation_at_lvl1() {
        let theta_null = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        let via_method = AbelianVarietyPrecomputed::new(&theta_null);
        let via_free = theta_precomputation(&theta_null);
        assert_eq!(
            via_method, via_free,
            "AbelianVarietyPrecomputed::new must match theta_precomputation",
        );
    }

    #[test]
    fn double_point_method_matches_free_function_at_lvl1() {
        let theta_null = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        let precomp = theta_precomputation(&theta_null);
        let p = ThetaPoint2D::new(
            small_fp2(1),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
            Fp2::<Fp1Element>::zero(),
        );

        let via_method = precomp.double_point(&p);
        let via_free = double_point(&precomp, &p);
        assert_eq!(
            via_method, via_free,
            "precomp.double_point(p) must match double_point(&precomp, &p)",
        );
    }

    #[test]
    fn double_iter_method_matches_free_function_at_lvl1() {
        let theta_null = ThetaPoint2D::new(small_fp2(2), small_fp2(3), small_fp2(5), small_fp2(7));
        let precomp = theta_precomputation(&theta_null);
        let p = ThetaPoint2D::new(small_fp2(11), small_fp2(13), small_fp2(17), small_fp2(19));

        for exp in 0..4 {
            let via_method = precomp.double_iter(&p, exp);
            let via_free = double_iter(&precomp, &p, exp);
            assert_eq!(
                via_method, via_free,
                "precomp.double_iter({exp}) must match double_iter(&precomp, {exp})",
            );
        }
    }
}
