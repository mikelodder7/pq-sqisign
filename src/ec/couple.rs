// SPDX-License-Identifier: MIT OR Apache-2.0
//! Couple-pair primitives for the `(2,2)`-isogeny chain on `E_1 × E_2`.
//!
//! These types model points on the elliptic product as two independent
//! single-curve halves, one on `E_1` and one on `E_2`. They live in `src/ec/`
//! rather than `src/isogeny/` because the representation and arithmetic are
//! still curve-level objects; the higher-level `(2,2)`-isogeny code only
//! consumes them.
//!
//! This matches the C reference's couple-point layer (the role served there
//! by `theta_couple_jac_point_t` and its x-only analogue). The S121
//! architectural finding remains load-bearing here: SQIsign 2.0.1 does not
//! define theta-2D `P + Q`, so addition needed by the chain stays on the
//! elliptic-product side and is expressed as per-half Jacobian arithmetic.

use rand_core::CryptoRng;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use crate::ec::jacobian::JacobianPoint;
use crate::ec::montgomery::{MontgomeryCurve, MontgomeryPoint};
use crate::gf::fp::BaseField;
use crate::gf::fp2::Fp2;

/// Pair of Montgomery curves `(E_1, E_2)` forming an elliptic product.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CoupleCurve<F: BaseField> {
    /// Curve `E_1` of the elliptic product.
    pub e1: MontgomeryCurve<F>,
    /// Curve `E_2` of the elliptic product.
    pub e2: MontgomeryCurve<F>,
}

impl<F: BaseField> CoupleCurve<F> {
    /// Construct an elliptic-product curve pair from `(E_1, E_2)`.
    #[inline]
    pub const fn new(e1: MontgomeryCurve<F>, e2: MontgomeryCurve<F>) -> Self {
        Self { e1, e2 }
    }

    /// The elliptic-product starting point `(E_0, E_0)`.
    #[inline]
    pub fn e0_e0() -> Self {
        Self::new(MontgomeryCurve::e0(), MontgomeryCurve::e0())
    }
}

/// Pair of Jacobian points `(P_1, P_2)` on `E_1 × E_2`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CoupleJacobianPoint<F: BaseField> {
    /// Jacobian point on `E_1`.
    pub p1: JacobianPoint<F>,
    /// Jacobian point on `E_2`.
    pub p2: JacobianPoint<F>,
}

impl<F: BaseField> CoupleJacobianPoint<F> {
    /// Construct a couple-Jacobian point from its two halves.
    #[inline]
    pub const fn new(p1: JacobianPoint<F>, p2: JacobianPoint<F>) -> Self {
        Self { p1, p2 }
    }

    /// Infinity in the elliptic product `(O_{E_1}, O_{E_2})`.
    #[inline]
    pub fn infinity() -> Self {
        Self::new(JacobianPoint::infinity(), JacobianPoint::infinity())
    }

    /// `Choice::TRUE` iff **both** halves are at infinity.
    ///
    /// Returns `self.p1.is_infinity() & self.p2.is_infinity()`. Used
    /// by chain-walker tests that need to detect a torsion-walk
    /// endpoint (where iterated doubling has consumed the full
    /// 2-power) on the couple, not just one half.
    #[inline]
    pub fn is_infinity(&self) -> Choice {
        self.p1.is_infinity() & self.p2.is_infinity()
    }

    /// Componentwise negation `(-P_1, -P_2)` on the elliptic product.
    ///
    /// Wraps [`JacobianPoint::negate`] per half. The Montgomery x-only
    /// representation doesn't need a sibling because `(X : Z)` is
    /// sign-symmetric (the x-coordinate is the same for `P` and `-P`).
    #[inline]
    pub fn negate(&self) -> Self {
        Self::new(self.p1.negate(), self.p2.negate())
    }

    /// Componentwise doubling `(2·P_1, 2·P_2)`.
    #[inline]
    pub fn double(&self, curves: &CoupleCurve<F>) -> Self {
        Self::new(self.p1.double(&curves.e1.a), self.p2.double(&curves.e2.a))
    }

    /// Iterated componentwise doubling `(2^n · P_1, 2^n · P_2)`.
    pub fn double_iter(&self, n: u32, curves: &CoupleCurve<F>) -> Self {
        let mut q = *self;
        for _ in 0..n {
            q = q.double(curves);
        }
        q
    }

    /// Project both halves to Montgomery x-only coordinates.
    #[inline]
    pub fn to_couple_xz(&self) -> CoupleMontgomeryPoint<F> {
        CoupleMontgomeryPoint::new(self.p1.to_montgomery_xz(), self.p2.to_montgomery_xz())
    }

    /// Return per-half Alg 8.13 `ADDComponents` triples `[(u_1, v_1, w_1), (u_2, v_2, w_2)]`.
    ///
    /// Preconditions: `self.p1` and `q.p1` are distinct affine points on
    /// `E_1`, and `self.p2` and `q.p2` are distinct affine points on `E_2`.
    /// Each half is computed independently with its own affine Montgomery
    /// coefficient `A`; no cross-half state sharing is permitted here.
    pub fn add_components_pair(
        &self,
        q: &Self,
        curves: &CoupleCurve<F>,
    ) -> [(Fp2<F>, Fp2<F>, Fp2<F>); 2] {
        // SAFETY: S125 requires the two halves to run independent ADDComponents
        // computations, each with its own Jacobian inputs and affine `A`.
        // Do not share control decisions or algebraic intermediates across
        // halves; that optimization is intentionally out of scope here.
        let e1 = self.p1.add_components(&q.p1, &curves.e1.a);
        let e2 = self.p2.add_components(&q.p2, &curves.e2.a);
        [e1, e2]
    }

    /// Rewrite both halves of `self` in place using **independent** `λ`
    /// values per half.
    ///
    /// Calls [`JacobianPoint::randomize_in_place`] twice in sequence,
    /// once per half. Each call samples its own `λ` from `rng`, so the
    /// two halves do NOT share a multiplier — a shared `λ` across
    /// halves would preserve the cross-half ratio
    /// `(λ²·X_1) / (λ²·X_2) = X_1/X_2`, leaving some bit-pattern
    /// correlation an attacker could exploit. S125 advisor doctrine
    /// (independent per-half discrimination) applies: no shared state,
    /// not even a shared randomizer.
    ///
    /// Cost: `2 ·` the cost of [`JacobianPoint::randomize_in_place`]
    /// (2 hashes, 2 Fp2 squares, 6 Fp2 multiplications, plus rng
    /// fill of 2 · 64 bytes — the L5 entropy ceiling per half).
    pub fn randomize_in_place<R: CryptoRng>(&mut self, rng: &mut R) {
        self.p1.randomize_in_place(rng);
        self.p2.randomize_in_place(rng);
    }

    /// Consuming-self ergonomic shim over [`Self::randomize_in_place`].
    pub fn randomize_projective<R: CryptoRng>(mut self, rng: &mut R) -> Self {
        self.randomize_in_place(rng);
        self
    }
}

/// Pair of Montgomery x-only points on `E_1 × E_2`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CoupleMontgomeryPoint<F: BaseField> {
    /// Montgomery x-only point on `E_1`.
    pub p1: MontgomeryPoint<F>,
    /// Montgomery x-only point on `E_2`.
    pub p2: MontgomeryPoint<F>,
}

impl<F: BaseField> CoupleMontgomeryPoint<F> {
    /// Construct a couple x-only point from its two halves.
    #[inline]
    pub const fn new(p1: MontgomeryPoint<F>, p2: MontgomeryPoint<F>) -> Self {
        Self { p1, p2 }
    }

    /// Couple infinity in x-only Montgomery form: `(O_{E_1}, O_{E_2})`.
    #[inline]
    pub fn infinity() -> Self {
        Self::new(MontgomeryPoint::infinity(), MontgomeryPoint::infinity())
    }

    /// `Choice::TRUE` iff **both** x-only halves are at infinity.
    ///
    /// Returns `self.p1.is_infinity() & self.p2.is_infinity()`.
    /// Useful for ladder-endpoint detection (where a scalar mult
    /// `[2^k] · P` on a 2^k-torsion descendant lands at infinity)
    /// on the couple-Montgomery side of the chain.
    #[inline]
    pub fn is_infinity(&self) -> Choice {
        self.p1.is_infinity() & self.p2.is_infinity()
    }

    /// Componentwise Montgomery x-only doubling on `E_1 × E_2`.
    ///
    /// Wraps [`MontgomeryPoint::x_double`] on each half. The two
    /// `a24` parameters are the precomputed `(A + 2) / 4` for each
    /// elliptic factor, matching the established convention from
    /// [`MontgomeryPoint::x_dbl_add`] (caller-owned a24).
    ///
    /// Mirrors C reference's `double_couple_point` in `hd.c:4-9`.
    pub fn double(&self, a24_1: &Fp2<F>, a24_2: &Fp2<F>) -> Self {
        Self::new(self.p1.x_double(a24_1), self.p2.x_double(a24_2))
    }

    /// Iterated componentwise Montgomery x-only doubling.
    ///
    /// `double_iter(0)` returns `self` unchanged. `double_iter(n)` for
    /// `n > 0` applies [`Self::double`] `n` times.
    ///
    /// Mirrors C reference's `double_couple_point_iter` in `hd.c:11-25`.
    pub fn double_iter(&self, n: u32, a24_1: &Fp2<F>, a24_2: &Fp2<F>) -> Self {
        if n == 0 {
            return *self;
        }
        let mut out = self.double(a24_1, a24_2);
        for _ in 1..n {
            out = out.double(a24_1, a24_2);
        }
        out
    }

    /// Ergonomic componentwise doubling that derives `a24` from a
    /// [`CoupleCurve<F>`] internally.
    ///
    /// Equivalent to `self.double(&curves.e1.a24(), &curves.e2.a24())`
    /// — saves the caller a two-line preamble. Matches the C
    /// reference's `double_couple_point(out, in, &E1E2)` signature
    /// directly. Use [`Self::double`] when `a24` is precomputed and
    /// the chain-walker reuses it across many iterations.
    pub fn double_with_curves(&self, curves: &CoupleCurve<F>) -> Self {
        let a24_1 = curves.e1.a24();
        let a24_2 = curves.e2.a24();
        self.double(&a24_1, &a24_2)
    }

    /// Ergonomic iterated doubling that derives `a24` from a
    /// [`CoupleCurve<F>`] internally. The `a24` derivation runs once
    /// (not per iteration), so amortization vs [`Self::double_iter`]
    /// with caller-precomputed `a24` is identical for `n ≥ 1`.
    pub fn double_iter_with_curves(&self, n: u32, curves: &CoupleCurve<F>) -> Self {
        if n == 0 {
            return *self;
        }
        let a24_1 = curves.e1.a24();
        let a24_2 = curves.e2.a24();
        self.double_iter(n, &a24_1, &a24_2)
    }

    /// Componentwise constant-time Montgomery scalar ladder
    /// on `E_1 × E_2`.
    ///
    /// Returns `(k · self.p_1, k · self.p_2)` where `scalar` is
    /// interpreted big-endian (matching
    /// [`MontgomeryPoint::ladder`]'s convention). Wraps the per-half
    /// `ladder` calls into a single componentwise call.
    ///
    /// The same `scalar` is applied to both halves — this is the
    /// natural primitive for the chain-walker's kernel-descent
    /// phase, where a single secret scalar walks both elliptic
    /// factors in lockstep.
    pub fn ladder(&self, scalar: &[u8], a24_1: &Fp2<F>, a24_2: &Fp2<F>) -> Self {
        Self::new(self.p1.ladder(scalar, a24_1), self.p2.ladder(scalar, a24_2))
    }

    /// Ergonomic componentwise ladder that derives `a24` from a
    /// [`CoupleCurve<F>`] internally. The `a24` derivation runs ONCE
    /// (not per ladder iteration); the ladder body itself reuses
    /// the same `a24` for every doubling/add step, so amortization
    /// is identical to passing precomputed `a24` directly.
    pub fn ladder_with_curves(&self, scalar: &[u8], curves: &CoupleCurve<F>) -> Self {
        let a24_1 = curves.e1.a24();
        let a24_2 = curves.e2.a24();
        self.ladder(scalar, &a24_1, &a24_2)
    }
}

impl<F: BaseField> ConstantTimeEq for CoupleCurve<F> {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.e1.a.ct_eq(&other.e1.a) & self.e2.a.ct_eq(&other.e2.a)
    }
}

impl<F: BaseField> ConditionallySelectable for CoupleCurve<F> {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self::new(
            MontgomeryCurve::new(Fp2::conditional_select(&a.e1.a, &b.e1.a, choice)),
            MontgomeryCurve::new(Fp2::conditional_select(&a.e2.a, &b.e2.a, choice)),
        )
    }
}

impl<F: BaseField> ConstantTimeEq for CoupleJacobianPoint<F> {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.p1.is_equivalent(&other.p1) & self.p2.is_equivalent(&other.p2)
    }
}

impl<F: BaseField> ConditionallySelectable for CoupleJacobianPoint<F> {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self::new(
            JacobianPoint::conditional_select(&a.p1, &b.p1, choice),
            JacobianPoint::conditional_select(&a.p2, &b.p2, choice),
        )
    }
}

impl<F: BaseField> ConstantTimeEq for CoupleMontgomeryPoint<F> {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.p1.ct_eq(&other.p1) & self.p2.ct_eq(&other.p2)
    }
}

impl<F: BaseField> ConditionallySelectable for CoupleMontgomeryPoint<F> {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self::new(
            MontgomeryPoint::conditional_select(&a.p1, &b.p1, choice),
            MontgomeryPoint::conditional_select(&a.p2, &b.p2, choice),
        )
    }
}

/// 2-torsion (or general) basis on a Montgomery curve: three points
/// `(P, Q, P - Q)`.
///
/// The standard SQIsign representation of a 2-power-torsion basis
/// includes the difference `P - Q` explicitly so that x-only
/// differential ladders can be evaluated without first computing
/// the difference (Montgomery `xDBLADD` and friends consume the
/// pre-computed `x(P - Q)` as part of their input).
///
/// Mirrors C reference's `ec_basis_t` (P, Q, PmQ).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct EcBasis<F: BaseField> {
    /// First basis point `P`.
    pub p: MontgomeryPoint<F>,
    /// Second basis point `Q`.
    pub q: MontgomeryPoint<F>,
    /// Difference `P − Q` (precomputed for differential-ladder use).
    pub p_minus_q: MontgomeryPoint<F>,
}

impl<F: BaseField> EcBasis<F> {
    /// Construct a basis from its three points. Caller is responsible
    /// for ensuring `p_minus_q` is consistent with `p` and `q`.
    #[inline]
    pub const fn new(
        p: MontgomeryPoint<F>,
        q: MontgomeryPoint<F>,
        p_minus_q: MontgomeryPoint<F>,
    ) -> Self {
        Self { p, q, p_minus_q }
    }
}

/// Chain-kernel input for the `(2, 2)`-isogeny chain walker.
///
/// Holds three couple-Jacobian points `(T1, T2, T1 - T2)` on
/// `E_1 × E_2`. Each `T_i` carries an 8-torsion (or 4-, 2-, depending
/// on the variant) kernel descendant on each elliptic factor; the
/// chain walker uses them to derive the per-step kernel generators
/// via `double_iter` on the couple-Jacobian side.
///
/// Mirrors C reference's `theta_kernel_couple_points_t` (T1, T2, T1m2).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ThetaKernelCouplePoints<F: BaseField> {
    /// First kernel couple `T_1` on `E_1 × E_2`.
    pub t1: CoupleJacobianPoint<F>,
    /// Second kernel couple `T_2` on `E_1 × E_2`.
    pub t2: CoupleJacobianPoint<F>,
    /// Difference `T_1 - T_2` (precomputed).
    pub t1_minus_t2: CoupleJacobianPoint<F>,
}

impl<F: BaseField> ThetaKernelCouplePoints<F> {
    /// Construct a kernel-couple bundle from its three couple-Jacobian
    /// points. Caller is responsible for the algebraic consistency
    /// of `t1_minus_t2 = t1 - t2`.
    #[inline]
    pub const fn new(
        t1: CoupleJacobianPoint<F>,
        t2: CoupleJacobianPoint<F>,
        t1_minus_t2: CoupleJacobianPoint<F>,
    ) -> Self {
        Self {
            t1,
            t2,
            t1_minus_t2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf::fp::Fp1Element;
    use crate::hash::hash_to_fp2;

    fn small_fp2<F: BaseField>(n: u32) -> Fp2<F> {
        let mut acc = Fp2::<F>::zero();
        let one = Fp2::<F>::one();
        for _ in 0..n {
            acc = acc.add(&one);
        }
        acc
    }

    fn first_liftable_point_on_e0<F: BaseField>() -> (MontgomeryPoint<F>, JacobianPoint<F>) {
        let curve = MontgomeryCurve::<F>::e0();
        let mut found = None;

        for n in 2..=8 {
            let p = MontgomeryPoint::new(small_fp2::<F>(n), Fp2::<F>::one());
            let lift = JacobianPoint::from_montgomery_xz(&p, &curve.a);
            if bool::from(lift.is_some()) {
                found = Some((p, lift.unwrap_or(JacobianPoint::infinity())));
                break;
            }
        }

        if found.is_none() {
            for i in 0..16u8 {
                let x_opt = hash_to_fp2::<F>(b"S128-couple-lift-x-A", &[i], 16);
                if !bool::from(x_opt.is_some()) {
                    continue;
                }
                let p = MontgomeryPoint::new(x_opt.unwrap_or(Fp2::<F>::zero()), Fp2::<F>::one());
                let lift = JacobianPoint::from_montgomery_xz(&p, &curve.a);
                if bool::from(lift.is_some()) {
                    found = Some((p, lift.unwrap_or(JacobianPoint::infinity())));
                    break;
                }
            }
        }

        assert!(
            found.is_some(),
            "S128: failed to find a first deterministic liftable point on E_0",
        );
        found.unwrap_or((MontgomeryPoint::infinity(), JacobianPoint::infinity()))
    }

    fn second_liftable_point_on_e0<F: BaseField>() -> (MontgomeryPoint<F>, JacobianPoint<F>) {
        let curve = MontgomeryCurve::<F>::e0();
        let (_, first) = first_liftable_point_on_e0::<F>();
        let mut found = None;

        for n in 2..=16 {
            let p = MontgomeryPoint::new(small_fp2::<F>(n), Fp2::<F>::one());
            let lift = JacobianPoint::from_montgomery_xz(&p, &curve.a);
            if bool::from(lift.is_some()) {
                let point = lift.unwrap_or(JacobianPoint::infinity());
                if !bool::from(point.is_equivalent(&first)) {
                    found = Some((p, point));
                    break;
                }
            }
        }

        if found.is_none() {
            for i in 0..32u8 {
                let x_opt = hash_to_fp2::<F>(b"S128-couple-lift-x-B", &[i], 16);
                if !bool::from(x_opt.is_some()) {
                    continue;
                }
                let p = MontgomeryPoint::new(x_opt.unwrap_or(Fp2::<F>::zero()), Fp2::<F>::one());
                let lift = JacobianPoint::from_montgomery_xz(&p, &curve.a);
                if bool::from(lift.is_some()) {
                    let point = lift.unwrap_or(JacobianPoint::infinity());
                    if !bool::from(point.is_equivalent(&first)) {
                        found = Some((p, point));
                        break;
                    }
                }
            }
        }

        assert!(
            found.is_some(),
            "S128: failed to find a second deterministic distinct liftable point on E_0",
        );
        found.unwrap_or((MontgomeryPoint::infinity(), JacobianPoint::infinity()))
    }

    fn check_couple_double_delegates_to_per_half<F: BaseField>() {
        let curves = CoupleCurve::<F>::e0_e0();
        let (_, first) = first_liftable_point_on_e0::<F>();
        let (_, second) = second_liftable_point_on_e0::<F>();
        let couple = CoupleJacobianPoint::new(first, second);
        let doubled = couple.double(&curves);

        assert!(
            bool::from(doubled.p1.is_equivalent(&couple.p1.double(&curves.e1.a))),
            "S128: couple double must delegate to JacobianPoint::double on the E_1 half",
        );
        assert!(
            bool::from(doubled.p2.is_equivalent(&couple.p2.double(&curves.e2.a))),
            "S128: couple double must delegate to JacobianPoint::double on the E_2 half",
        );
    }

    #[test]
    fn couple_double_delegates_to_per_half_at_lvl1() {
        check_couple_double_delegates_to_per_half::<Fp1Element>();
    }

    #[test]
    fn couple_double_delegates_to_per_half_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_couple_double_delegates_to_per_half::<Fp3Element>();
    }

    #[test]
    fn couple_double_delegates_to_per_half_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_couple_double_delegates_to_per_half::<Fp5Element>();
    }

    fn check_couple_double_iter_zero_is_identity<F: BaseField>() {
        let (_, first) = first_liftable_point_on_e0::<F>();
        let (_, second) = second_liftable_point_on_e0::<F>();
        let curves = CoupleCurve::<F>::e0_e0();
        let couple = CoupleJacobianPoint::new(first, second);

        assert_eq!(
            couple.double_iter(0, &curves),
            couple,
            "S128: double_iter(0, curves) must leave the couple point unchanged pointwise",
        );
    }

    #[test]
    fn couple_double_iter_zero_is_identity_at_lvl1() {
        check_couple_double_iter_zero_is_identity::<Fp1Element>();
    }

    #[test]
    fn couple_double_iter_zero_is_identity_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_couple_double_iter_zero_is_identity::<Fp3Element>();
    }

    #[test]
    fn couple_double_iter_zero_is_identity_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_couple_double_iter_zero_is_identity::<Fp5Element>();
    }

    fn check_couple_double_iter_n_equals_n_doubles<F: BaseField>() {
        let (_, first) = first_liftable_point_on_e0::<F>();
        let (_, second) = second_liftable_point_on_e0::<F>();
        let curves = CoupleCurve::<F>::e0_e0();
        let couple = CoupleJacobianPoint::new(first, second);
        let iterated = couple.double_iter(3, &curves);
        let repeated = couple.double(&curves).double(&curves).double(&curves);

        assert!(
            bool::from(iterated.p1.is_equivalent(&repeated.p1)),
            "S128: double_iter(3, curves) must match three doubles on the E_1 half",
        );
        assert!(
            bool::from(iterated.p2.is_equivalent(&repeated.p2)),
            "S128: double_iter(3, curves) must match three doubles on the E_2 half",
        );
    }

    #[test]
    fn couple_double_iter_n_equals_n_doubles_at_lvl1() {
        check_couple_double_iter_n_equals_n_doubles::<Fp1Element>();
    }

    #[test]
    fn couple_double_iter_n_equals_n_doubles_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_couple_double_iter_n_equals_n_doubles::<Fp3Element>();
    }

    #[test]
    fn couple_double_iter_n_equals_n_doubles_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_couple_double_iter_n_equals_n_doubles::<Fp5Element>();
    }

    fn check_couple_to_xz_delegates_per_half<F: BaseField>() {
        let (_, first) = first_liftable_point_on_e0::<F>();
        let (_, second) = second_liftable_point_on_e0::<F>();
        let couple = CoupleJacobianPoint::new(first, second);
        let xz = couple.to_couple_xz();

        assert!(
            bool::from(xz.p1.ct_eq(&couple.p1.to_montgomery_xz())),
            "S128: to_couple_xz must delegate to JacobianPoint::to_montgomery_xz on the E_1 half",
        );
        assert!(
            bool::from(xz.p2.ct_eq(&couple.p2.to_montgomery_xz())),
            "S128: to_couple_xz must delegate to JacobianPoint::to_montgomery_xz on the E_2 half",
        );
    }

    #[test]
    fn couple_to_xz_delegates_per_half_at_lvl1() {
        check_couple_to_xz_delegates_per_half::<Fp1Element>();
    }

    #[test]
    fn couple_to_xz_delegates_per_half_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_couple_to_xz_delegates_per_half::<Fp3Element>();
    }

    #[test]
    fn couple_to_xz_delegates_per_half_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_couple_to_xz_delegates_per_half::<Fp5Element>();
    }

    fn check_couple_add_components_pair_matches_per_half<F: BaseField>() {
        let curves = CoupleCurve::<F>::e0_e0();
        let (_, first) = first_liftable_point_on_e0::<F>();
        let (_, second) = second_liftable_point_on_e0::<F>();
        let p = CoupleJacobianPoint::new(first, second);
        let q = CoupleJacobianPoint::new(second, first);

        assert!(
            !bool::from(p.p1.is_equivalent(&q.p1)),
            "S128: add_components_pair test must use distinct E_1 inputs",
        );
        assert!(
            !bool::from(p.p2.is_equivalent(&q.p2)),
            "S128: add_components_pair test must use distinct E_2 inputs",
        );

        let pair = p.add_components_pair(&q, &curves);
        let e1 = p.p1.add_components(&q.p1, &curves.e1.a);
        let e2 = p.p2.add_components(&q.p2, &curves.e2.a);

        assert_eq!(
            pair[0], e1,
            "S128: add_components_pair[0] must match JacobianPoint::add_components on E_1",
        );
        assert_eq!(
            pair[1], e2,
            "S128: add_components_pair[1] must match JacobianPoint::add_components on E_2",
        );
    }

    #[test]
    fn couple_add_components_pair_matches_per_half_at_lvl1() {
        check_couple_add_components_pair_matches_per_half::<Fp1Element>();
    }

    #[test]
    fn couple_add_components_pair_matches_per_half_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_couple_add_components_pair_matches_per_half::<Fp3Element>();
    }

    #[test]
    fn couple_add_components_pair_matches_per_half_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_couple_add_components_pair_matches_per_half::<Fp5Element>();
    }

    fn check_couple_infinity_doubles_to_infinity<F: BaseField>() {
        let curves = CoupleCurve::<F>::e0_e0();
        let infinity = CoupleJacobianPoint::<F>::infinity();
        let doubled = infinity.double(&curves);

        assert!(
            bool::from(doubled.p1.is_infinity()),
            "S128: doubling couple infinity must keep the E_1 half at infinity",
        );
        assert!(
            bool::from(doubled.p2.is_infinity()),
            "S128: doubling couple infinity must keep the E_2 half at infinity",
        );
        assert_eq!(
            doubled, infinity,
            "S128: doubling couple infinity must preserve the canonical pair of infinity sentinels",
        );
    }

    #[test]
    fn couple_infinity_doubles_to_infinity_at_lvl1() {
        check_couple_infinity_doubles_to_infinity::<Fp1Element>();
    }

    #[test]
    fn couple_infinity_doubles_to_infinity_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_couple_infinity_doubles_to_infinity::<Fp3Element>();
    }

    #[test]
    fn couple_infinity_doubles_to_infinity_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_couple_infinity_doubles_to_infinity::<Fp5Element>();
    }

    // S128 advisor item (3): mixed-infinity couple — `(O, P).double()` must
    // produce `(O, 2P)`. The (2,2)-isogeny gluing kernel can hit exactly
    // this boundary case during a descent step where one elliptic-product
    // component has been pushed to infinity but the other still carries a
    // non-trivial point. The componentwise-delegation claim is untested at
    // this boundary without this case, so a leak in Layer 1's edge-case
    // handling would surface only here.
    fn check_couple_double_mixed_infinity<F: BaseField>() {
        let curves = CoupleCurve::<F>::e0_e0();
        let (_, p) = first_liftable_point_on_e0::<F>();

        // (O, P) → (O, 2P)
        let left_inf = CoupleJacobianPoint::<F>::new(JacobianPoint::infinity(), p);
        let doubled_left = left_inf.double(&curves);
        assert!(
            bool::from(doubled_left.p1.is_infinity()),
            "S128: doubling (O, P) must keep the E_1 half at infinity",
        );
        let expected_2p = p.double(&curves.e2.a);
        assert!(
            bool::from(doubled_left.p2.is_equivalent(&expected_2p)),
            "S128: doubling (O, P) must produce 2P on the E_2 half",
        );

        // (P, O) → (2P, O), symmetric case
        let right_inf = CoupleJacobianPoint::<F>::new(p, JacobianPoint::infinity());
        let doubled_right = right_inf.double(&curves);
        let expected_2p_e1 = p.double(&curves.e1.a);
        assert!(
            bool::from(doubled_right.p1.is_equivalent(&expected_2p_e1)),
            "S128: doubling (P, O) must produce 2P on the E_1 half",
        );
        assert!(
            bool::from(doubled_right.p2.is_infinity()),
            "S128: doubling (P, O) must keep the E_2 half at infinity",
        );
    }

    #[test]
    fn couple_double_mixed_infinity_at_lvl1() {
        check_couple_double_mixed_infinity::<Fp1Element>();
    }

    #[test]
    fn couple_double_mixed_infinity_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_couple_double_mixed_infinity::<Fp3Element>();
    }

    #[test]
    fn couple_double_mixed_infinity_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_couple_double_mixed_infinity::<Fp5Element>();
    }

    // S133 projective coordinate randomization tests for the couple-pair.
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn fresh_rng(seed_byte: u8) -> ChaCha20Rng {
        ChaCha20Rng::from_seed([seed_byte; 32])
    }

    fn check_couple_randomize_preserves_per_half<F: BaseField>() {
        let (_, p) = first_liftable_point_on_e0::<F>();
        let couple = CoupleJacobianPoint::<F>::new(p, p);
        let mut rng = fresh_rng(0x77);
        let randomized = couple.randomize_projective(&mut rng);
        assert!(
            bool::from(randomized.p1.is_equivalent(&couple.p1)),
            "S133: couple randomize must preserve E_1-half projective equivalence",
        );
        assert!(
            bool::from(randomized.p2.is_equivalent(&couple.p2)),
            "S133: couple randomize must preserve E_2-half projective equivalence",
        );
    }

    #[test]
    fn couple_randomize_preserves_per_half_at_lvl1() {
        check_couple_randomize_preserves_per_half::<Fp1Element>();
    }

    #[test]
    fn couple_randomize_preserves_per_half_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_couple_randomize_preserves_per_half::<Fp3Element>();
    }

    #[test]
    fn couple_randomize_preserves_per_half_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_couple_randomize_preserves_per_half::<Fp5Element>();
    }

    // S133 + S125 doctrine: the two halves of a couple-randomize must
    // use INDEPENDENT λ values. If both halves shared one λ, then
    // input `(P, P)` (same Jacobian on both sides) would yield a
    // randomized `(λ²X, λ³Y, λZ)` on BOTH halves identically — i.e.,
    // ct_eq_repr(p1, p2) would still return TRUE. With independent λ
    // per half, two independent rejection samples of Fp2 collide with
    // probability ~2^-500. So if the input has p1 == p2 pointwise,
    // the output should have p1 != p2 pointwise — proving the halves
    // got different λ.
    #[test]
    fn couple_randomize_uses_independent_lambdas_at_lvl1() {
        let (_, p) = first_liftable_point_on_e0::<Fp1Element>();
        let couple = CoupleJacobianPoint::<Fp1Element>::new(p, p);
        assert!(
            bool::from(couple.p1.ct_eq_repr(&couple.p2)),
            "S133 setup: input couple must have pointwise-identical halves",
        );
        let mut rng = fresh_rng(0x88);
        let randomized = couple.randomize_projective(&mut rng);
        assert!(
            !bool::from(randomized.p1.ct_eq_repr(&randomized.p2)),
            "S133: couple randomize must use independent lambdas per half — \
             identical inputs must produce pointwise-different outputs \
             (probability of shared-lambda coincidence is ~2^-500)",
        );
        // But projective equivalence MUST still hold for each half — the
        // outputs are different bit patterns of the same affine point.
        assert!(
            bool::from(randomized.p1.is_equivalent(&couple.p1)),
            "S133: independent-lambda blinding still preserves affine E_1 point",
        );
        assert!(
            bool::from(randomized.p2.is_equivalent(&couple.p2)),
            "S133: independent-lambda blinding still preserves affine E_2 point",
        );
    }

    // S148 — EcBasis + ThetaKernelCouplePoints struct smoke tests.

    #[test]
    fn ec_basis_construction_preserves_fields_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let one = Fp2::<Fp1Element>::one();
        let two = one.add(&one);
        let three = two.add(&one);

        let p_pt = MontgomeryPoint::<Fp1Element>::new(one, two);
        let q_pt = MontgomeryPoint::<Fp1Element>::new(two, three);
        let pmq_pt = MontgomeryPoint::<Fp1Element>::new(three, one);

        let basis = EcBasis::new(p_pt, q_pt, pmq_pt);

        assert_eq!(basis.p, p_pt, "S148: EcBasis.p preserved as-passed");
        assert_eq!(basis.q, q_pt, "S148: EcBasis.q preserved as-passed");
        assert_eq!(
            basis.p_minus_q, pmq_pt,
            "S148: EcBasis.p_minus_q preserved as-passed",
        );
    }

    // S149 — CoupleMontgomeryPoint::double + double_iter tests.

    /// S149 oracle: x_double((3, 1), a24 = 1) = (64, 192) per
    /// Costello-Hisil-Smith xDBL formula.
    /// Couple: both halves at (3, 1) with same a24 → both halves output (64, 192).
    #[test]
    fn couple_montgomery_double_oracle_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let three = {
            let one = Fp2::<Fp1Element>::one();
            one.add(&one).add(&one)
        };
        let one_v = Fp2::<Fp1Element>::one();
        let a24 = one_v;

        let p = CoupleMontgomeryPoint::new(
            MontgomeryPoint::<Fp1Element>::new(three, one_v),
            MontgomeryPoint::<Fp1Element>::new(three, one_v),
        );
        let doubled = p.double(&a24, &a24);

        // Expected: (64, 192) on both halves.
        let expected_x = {
            let mut acc = Fp2::<Fp1Element>::zero();
            for _ in 0..64 {
                acc = acc.add(&one_v);
            }
            acc
        };
        let expected_z = {
            let mut acc = Fp2::<Fp1Element>::zero();
            for _ in 0..192 {
                acc = acc.add(&one_v);
            }
            acc
        };
        assert_eq!(doubled.p1.x, expected_x, "S149: doubled.p1.x = 64");
        assert_eq!(doubled.p1.z, expected_z, "S149: doubled.p1.z = 192");
        assert_eq!(doubled.p2.x, expected_x, "S149: doubled.p2.x = 64");
        assert_eq!(doubled.p2.z, expected_z, "S149: doubled.p2.z = 192");
    }

    #[test]
    fn couple_montgomery_double_iter_zero_is_identity_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let one_v = Fp2::<Fp1Element>::one();
        let two = one_v.add(&one_v);
        let p = CoupleMontgomeryPoint::new(
            MontgomeryPoint::<Fp1Element>::new(one_v, two),
            MontgomeryPoint::<Fp1Element>::new(two, one_v),
        );
        let r = p.double_iter(0, &one_v, &one_v);
        assert_eq!(r, p, "S149: double_iter(0) = identity");
    }

    // S162 — CoupleJacobianPoint::negate.

    #[test]
    fn couple_jacobian_negate_round_trip_is_identity_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let one_v = Fp2::<Fp1Element>::one();
        let two = one_v.add(&one_v);
        let three = two.add(&one_v);
        let p = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(one_v, two, one_v),
            JacobianPoint::<Fp1Element>::new(three, one_v, two),
        );
        let neg_neg = p.negate().negate();
        // Projective equivalence preserves negation round-trip.
        assert!(
            bool::from(neg_neg.p1.is_equivalent(&p.p1)),
            "S162: -(-p1) must equal p1 (projective)",
        );
        assert!(
            bool::from(neg_neg.p2.is_equivalent(&p.p2)),
            "S162: -(-p2) must equal p2 (projective)",
        );
    }

    #[test]
    fn couple_jacobian_negate_componentwise_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let one_v = Fp2::<Fp1Element>::one();
        let two = one_v.add(&one_v);
        let three = two.add(&one_v);
        let p = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(one_v, two, one_v),
            JacobianPoint::<Fp1Element>::new(three, one_v, two),
        );
        let neg = p.negate();
        assert_eq!(neg.p1, p.p1.negate(), "S162: negate p1 must match per-half");
        assert_eq!(neg.p2, p.p2.negate(), "S162: negate p2 must match per-half");
    }

    #[test]
    fn couple_jacobian_negate_of_infinity_is_infinity_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        let inf = CoupleJacobianPoint::<Fp1Element>::infinity();
        let neg_inf = inf.negate();
        assert!(
            bool::from(neg_inf.is_infinity()),
            "S162: -(O, O) must remain at infinity",
        );
    }

    // S161 — is_infinity predicates on couple types.

    #[test]
    fn couple_jacobian_is_infinity_true_for_infinity_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        let inf = CoupleJacobianPoint::<Fp1Element>::infinity();
        assert!(
            bool::from(inf.is_infinity()),
            "S161: CoupleJacobianPoint::infinity().is_infinity() must be TRUE",
        );
    }

    #[test]
    fn couple_jacobian_is_infinity_false_when_one_half_finite_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;
        let one_v = Fp2::<Fp1Element>::one();
        let finite_p = JacobianPoint::<Fp1Element>::new(one_v, one_v.add(&one_v), one_v);
        let mixed = CoupleJacobianPoint::new(JacobianPoint::infinity(), finite_p);
        assert!(
            !bool::from(mixed.is_infinity()),
            "S161: couple with one finite half is NOT infinity",
        );
        let mixed_other = CoupleJacobianPoint::new(finite_p, JacobianPoint::infinity());
        assert!(
            !bool::from(mixed_other.is_infinity()),
            "S161: couple with one finite half (other side) is NOT infinity",
        );
    }

    #[test]
    fn couple_montgomery_is_infinity_true_for_infinity_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        let inf = CoupleMontgomeryPoint::<Fp1Element>::infinity();
        assert!(
            bool::from(inf.is_infinity()),
            "S161: CoupleMontgomeryPoint::infinity().is_infinity() must be TRUE",
        );
    }

    #[test]
    fn couple_montgomery_is_infinity_false_when_one_half_finite_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;
        let one_v = Fp2::<Fp1Element>::one();
        let two = one_v.add(&one_v);
        let finite = MontgomeryPoint::<Fp1Element>::new(two, one_v);
        let mixed = CoupleMontgomeryPoint::new(MontgomeryPoint::infinity(), finite);
        assert!(
            !bool::from(mixed.is_infinity()),
            "S161: couple-Montgomery with one finite half is NOT infinity",
        );
    }

    // S160 — CoupleMontgomeryPoint::ladder + ladder_with_curves tests.

    /// S160 oracle: scalar = [0] (zero) on a couple should produce
    /// `(O, O)` componentwise. Confirms `0 · P = O` per the
    /// underlying MontgomeryPoint::ladder semantics.
    #[test]
    fn couple_montgomery_ladder_scalar_zero_yields_infinity_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let one_v = Fp2::<Fp1Element>::one();
        let two = one_v.add(&one_v);
        let three = two.add(&one_v);
        let a24 = one_v;

        let p = CoupleMontgomeryPoint::new(
            MontgomeryPoint::<Fp1Element>::new(three, one_v),
            MontgomeryPoint::<Fp1Element>::new(two, one_v),
        );
        let r = p.ladder(&[0u8], &a24, &a24);

        assert!(
            bool::from(r.p1.is_infinity()),
            "S160: 0 · P_1 must be infinity",
        );
        assert!(
            bool::from(r.p2.is_infinity()),
            "S160: 0 · P_2 must be infinity",
        );
    }

    /// S160: confirm componentwise ladder matches per-half ladder
    /// independently. Same scalar fed into both halves.
    #[test]
    fn couple_montgomery_ladder_componentwise_matches_per_half_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let one_v = Fp2::<Fp1Element>::one();
        let two = one_v.add(&one_v);
        let three = two.add(&one_v);
        let a24_1 = one_v;
        let a24_2 = two; // distinct a24s exercise the per-half independence

        let p = CoupleMontgomeryPoint::new(
            MontgomeryPoint::<Fp1Element>::new(three, one_v),
            MontgomeryPoint::<Fp1Element>::new(two, one_v),
        );
        let scalar = [0x05u8]; // arbitrary non-zero scalar
        let via_couple = p.ladder(&scalar, &a24_1, &a24_2);

        let solo_1 = p.p1.ladder(&scalar, &a24_1);
        let solo_2 = p.p2.ladder(&scalar, &a24_2);

        assert_eq!(
            via_couple.p1, solo_1,
            "S160: couple ladder.p1 must match per-half ladder on E_1",
        );
        assert_eq!(
            via_couple.p2, solo_2,
            "S160: couple ladder.p2 must match per-half ladder on E_2",
        );
    }

    /// S160: confirm `ladder_with_curves` matches the explicit-a24
    /// variant on the same input.
    #[test]
    fn couple_montgomery_ladder_with_curves_matches_explicit_at_lvl1() {
        use crate::ec::montgomery::MontgomeryCurve;
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let one_v = Fp2::<Fp1Element>::one();
        let three = one_v.add(&one_v).add(&one_v);

        let curves = CoupleCurve {
            e1: MontgomeryCurve::<Fp1Element>::e0(),
            e2: MontgomeryCurve::<Fp1Element>::e0(),
        };
        let p = CoupleMontgomeryPoint::new(
            MontgomeryPoint::<Fp1Element>::new(three, one_v),
            MontgomeryPoint::<Fp1Element>::new(three, one_v),
        );
        let scalar = [0x07u8];

        let r_wrapper = p.ladder_with_curves(&scalar, &curves);
        let r_explicit = p.ladder(&scalar, &curves.e1.a24(), &curves.e2.a24());

        assert_eq!(
            r_wrapper, r_explicit,
            "S160: ladder_with_curves must match ladder with explicit a24",
        );
    }

    /// S151: confirm `double_with_curves` produces the same output as
    /// the explicit-a24 variant. Same input fixture as S149 oracle
    /// (curve a=0, P=(3,1) on each half) — but now CoupleCurve carries
    /// a=0 and the wrapper computes a24=(0+2)/4=1/2 from it.
    ///
    /// Expected output is computed via the explicit-a24 variant with
    /// the same a24=(0+2)/4=1/2 — not a hand-derived value (the
    /// fraction 1/2 isn't trivially small_fp2).
    #[test]
    fn couple_montgomery_double_with_curves_matches_explicit_a24_at_lvl1() {
        use crate::ec::montgomery::MontgomeryCurve;
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let one_v = Fp2::<Fp1Element>::one();
        let three = one_v.add(&one_v).add(&one_v);

        let curves = CoupleCurve {
            e1: MontgomeryCurve::<Fp1Element>::e0(),
            e2: MontgomeryCurve::<Fp1Element>::e0(),
        };
        let p = CoupleMontgomeryPoint::new(
            MontgomeryPoint::<Fp1Element>::new(three, one_v),
            MontgomeryPoint::<Fp1Element>::new(three, one_v),
        );

        let r_wrapper = p.double_with_curves(&curves);
        let r_explicit = p.double(&curves.e1.a24(), &curves.e2.a24());

        assert_eq!(
            r_wrapper, r_explicit,
            "S151: double_with_curves must match double with explicit a24",
        );
    }

    #[test]
    fn couple_montgomery_double_iter_with_curves_zero_is_identity_at_lvl1() {
        use crate::ec::montgomery::MontgomeryCurve;
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let one_v = Fp2::<Fp1Element>::one();
        let two = one_v.add(&one_v);

        let curves = CoupleCurve {
            e1: MontgomeryCurve::<Fp1Element>::e0(),
            e2: MontgomeryCurve::<Fp1Element>::e0(),
        };
        let p = CoupleMontgomeryPoint::new(
            MontgomeryPoint::<Fp1Element>::new(one_v, two),
            MontgomeryPoint::<Fp1Element>::new(two, one_v),
        );
        let r = p.double_iter_with_curves(0, &curves);
        assert_eq!(r, p, "S151: double_iter_with_curves(0) = identity");
    }

    #[test]
    fn couple_montgomery_double_iter_with_curves_matches_explicit_at_lvl1() {
        use crate::ec::montgomery::MontgomeryCurve;
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let one_v = Fp2::<Fp1Element>::one();
        let three = one_v.add(&one_v).add(&one_v);

        let curves = CoupleCurve {
            e1: MontgomeryCurve::<Fp1Element>::e0(),
            e2: MontgomeryCurve::<Fp1Element>::e0(),
        };
        let p = CoupleMontgomeryPoint::new(
            MontgomeryPoint::<Fp1Element>::new(three, one_v),
            MontgomeryPoint::<Fp1Element>::new(three, one_v),
        );

        for n in 1..4 {
            let r_wrapper = p.double_iter_with_curves(n, &curves);
            let r_explicit = p.double_iter(n, &curves.e1.a24(), &curves.e2.a24());
            assert_eq!(
                r_wrapper, r_explicit,
                "S151: double_iter_with_curves({n}) must match explicit-a24 variant",
            );
        }
    }

    #[test]
    fn couple_montgomery_double_iter_one_equals_double_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let one_v = Fp2::<Fp1Element>::one();
        let two = one_v.add(&one_v);
        let three = two.add(&one_v);
        let p = CoupleMontgomeryPoint::new(
            MontgomeryPoint::<Fp1Element>::new(three, one_v),
            MontgomeryPoint::<Fp1Element>::new(two, one_v),
        );
        let r_iter = p.double_iter(1, &one_v, &one_v);
        let r_solo = p.double(&one_v, &one_v);
        assert_eq!(r_iter, r_solo, "S149: double_iter(1) = double");
    }

    #[test]
    fn theta_kernel_couple_points_construction_preserves_fields_at_lvl1() {
        use crate::ec::jacobian::JacobianPoint;
        use crate::gf::fp::Fp1Element;

        let inf = JacobianPoint::<Fp1Element>::infinity();
        let t1 = CoupleJacobianPoint { p1: inf, p2: inf };
        let t2 = CoupleJacobianPoint { p1: inf, p2: inf };
        let t1m2 = CoupleJacobianPoint { p1: inf, p2: inf };

        let kernel = ThetaKernelCouplePoints::new(t1, t2, t1m2);

        assert_eq!(kernel.t1, t1, "S148: kernel.t1 preserved");
        assert_eq!(kernel.t2, t2, "S148: kernel.t2 preserved");
        assert_eq!(
            kernel.t1_minus_t2, t1m2,
            "S148: kernel.t1_minus_t2 preserved"
        );
    }
}
