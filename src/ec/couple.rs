// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(rustdoc::private_intra_doc_links)]
//! Couple-pair primitives for the `(2,2)`-isogeny chain on `E_1 × E_2`.
//!
//! These types model points on the elliptic product as two independent
//! single-curve halves, one on `E_1` and one on `E_2`. They live in `src/ec/`
//! rather than `src/isogeny/` because the representation and arithmetic are
//! still curve-level objects; the higher-level `(2,2)`-isogeny code only
//! consumes them.
//!
//! This matches the C reference's couple-point layer (the role served there
//! by `theta_couple_jac_point_t` and its x-only analogue). The
//! architectural constraint remains load-bearing here: SQIsign 2.0.1 does not
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

    /// `Choice::TRUE` iff **both** halves are the starting curve `E_0`.
    ///
    /// Predicate companion to [`Self::e0_e0`]. Returns
    /// `self.e1.is_e0() & self.e2.is_e0()`. Useful for chain-init
    /// assertions that confirm the elliptic-product origin matches
    /// SQIsign's expected starting state.
    #[inline]
    pub fn is_e0_e0(&self) -> Choice {
        self.e1.is_e0() & self.e2.is_e0()
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

    /// **Structural-only** 2-torsion check on both halves
    /// (`Y == 0 ∧ Z ≠ 0` per
    /// [`JacobianPoint::is_two_torsion_unchecked`]).
    ///
    /// **Precondition** (caller's responsibility): both `self.p1` and
    /// `self.p2` are valid Montgomery-curve points. This predicate
    /// inherits the unchecked semantics of its per-half components —
    /// see [`JacobianPoint::is_two_torsion_unchecked`] for the full
    /// rationale + the safe-test alternative on the x-only Montgomery
    /// side.
    ///
    /// Useful for chain-walker endpoint detection inside trusted
    /// chain code where curve membership is invariant.
    #[inline]
    pub fn is_two_torsion_unchecked(&self) -> Choice {
        self.p1.is_two_torsion_unchecked() & self.p2.is_two_torsion_unchecked()
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

    /// Componentwise affine normalization on the elliptic product.
    ///
    /// Wraps [`JacobianPoint::to_affine`] per half. The Jacobian
    /// `(X : Y : Z)` representation is normalized to its `(X/Z²,
    /// Y/Z³, 1)` form (with infinity returned as the canonical
    /// `(1, 1, 0)` sentinel). Both halves are normalized
    /// independently — no shared inversion across halves (matching
    /// the per-half independent state doctrine).
    #[inline]
    pub fn to_affine(&self) -> Self {
        Self::new(self.p1.to_affine(), self.p2.to_affine())
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
        // SAFETY: The two halves must run independent ADDComponents
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
    /// correlation an attacker could exploit. The independent per-half
    /// discrimination doctrine applies: no shared state,
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

    /// `Choice::TRUE` iff **both** x-only halves are finite
    /// 2-torsion points on their respective Montgomery curves.
    ///
    /// Wraps [`MontgomeryPoint::is_two_torsion`] per half
    /// with each half's affine A coefficient (`a_1` for E_1, `a_2`
    /// for E_2). Companion to [`CoupleJacobianPoint::is_two_torsion_unchecked`]
    /// for the x-only side of the chain.
    #[inline]
    pub fn is_two_torsion(&self, a_1: &Fp2<F>, a_2: &Fp2<F>) -> Choice {
        self.p1.is_two_torsion(a_1) & self.p2.is_two_torsion(a_2)
    }

    /// Ergonomic 2-torsion predicate that derives the per-half
    /// affine A coefficients from a [`CoupleCurve<F>`] internally.
    ///
    /// Equivalent to `self.is_two_torsion(&curves.e1.a, &curves.e2.a)`
    /// but matches the [`Self::double_with_curves`] / [`Self::ladder_with_curves`]
    /// ergonomic pattern.
    #[inline]
    pub fn is_two_torsion_with_curves(&self, curves: &CoupleCurve<F>) -> Choice {
        self.is_two_torsion(&curves.e1.a, &curves.e2.a)
    }

    /// Componentwise affine normalization of both x-only halves.
    ///
    /// Wraps [`MontgomeryPoint::to_affine`] per half. Each half's
    /// `(X : Z)` projective representative is normalized to
    /// `(X/Z : 1)`. Infinity sentinels `(1 : 0)` are preserved.
    /// Companion to [`CoupleJacobianPoint::to_affine`] for the
    /// x-only side of the chain.
    #[inline]
    pub fn to_affine(&self) -> Self {
        Self::new(self.p1.to_affine(), self.p2.to_affine())
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

    /// Apply iterated doubling `2^n · ·` to all three kernel points
    /// simultaneously.
    ///
    /// Returns a new bundle `(2^n·T_1, 2^n·T_2, 2^n·(T_1 − T_2))`.
    /// Algebraically `2(T_1 − T_2) = 2T_1 − 2T_2`, so componentwise
    /// doubling preserves the difference relationship; the resulting
    /// bundle remains a valid `ThetaKernelCouplePoints` invariant.
    ///
    /// Used by the chain walker's torsion-descent phase: starting
    /// from an 8-torsion-above-kernel bundle, repeated doublings
    /// descend through 4-torsion to 2-torsion (i.e., the kernel
    /// itself) for use as input to the splitting boundary.
    pub fn double_iter(&self, n: u32, curves: &CoupleCurve<F>) -> Self {
        Self::new(
            self.t1.double_iter(n, curves),
            self.t2.double_iter(n, curves),
            self.t1_minus_t2.double_iter(n, curves),
        )
    }

    /// Project all three kernel points to Montgomery x-only form.
    ///
    /// Returns `[T_1.xz, T_2.xz, (T_1 - T_2).xz]` — a triple of
    /// [`CoupleMontgomeryPoint<F>`] in array order
    /// `(t1, t2, t1_minus_t2)`.
    ///
    /// Used at chain initialization where the gluing isogeny's
    /// x-only-input variant
    /// ([`crate::isogeny::gluing::GluingCodomain::eval_point_special_case`])
    /// expects couple-Montgomery inputs derived from the Jacobian
    /// kernel descendants.
    pub fn to_couple_xz_triple(&self) -> [CoupleMontgomeryPoint<F>; 3] {
        [
            self.t1.to_couple_xz(),
            self.t2.to_couple_xz(),
            self.t1_minus_t2.to_couple_xz(),
        ]
    }

    /// Apply projective coordinate randomization (blinding)
    /// to all three kernel points in place.
    ///
    /// Each of `t1`, `t2`, `t1_minus_t2` is blinded with its own
    /// independent random scaling — and each couple-half within
    /// those is also independently blinded per the per-half independence doctrine. The
    /// affine difference invariant `t1_minus_t2 = t1 - t2` is
    /// preserved because projective rescaling doesn't change the
    /// underlying affine point.
    ///
    /// Cost: `3 ·` the cost of
    /// [`CoupleJacobianPoint::randomize_in_place`] (each call samples
    /// its own per-half lambdas, so total entropy draw is `6 · 64`
    /// bytes at L5).
    pub fn randomize_in_place<R: CryptoRng>(&mut self, rng: &mut R) {
        self.t1.randomize_in_place(rng);
        self.t2.randomize_in_place(rng);
        self.t1_minus_t2.randomize_in_place(rng);
    }

    /// Consuming-self ergonomic shim over [`Self::randomize_in_place`].
    pub fn randomize_projective<R: CryptoRng>(mut self, rng: &mut R) -> Self {
        self.randomize_in_place(rng);
        self
    }

    /// `Choice::TRUE` iff **all three** kernel-couple fields are at
    /// infinity (i.e., the kernel has been fully consumed by
    /// torsion-descent: every doubling has driven the point to the
    /// identity).
    ///
    /// Useful for chain-walker bound checks: detecting when iterated
    /// doubling has descended past the kernel order and further
    /// doublings would be no-ops.
    pub fn is_infinity(&self) -> Choice {
        self.t1.is_infinity() & self.t2.is_infinity() & self.t1_minus_t2.is_infinity()
    }

    /// **Structural-only** 2-torsion check on all three kernel-couple
    /// fields (per
    /// [`CoupleJacobianPoint::is_two_torsion_unchecked`]).
    ///
    /// **Precondition** (caller's responsibility): every component is
    /// a valid Montgomery-curve point. This predicate inherits the
    /// unchecked semantics of its underlying
    /// [`JacobianPoint::is_two_torsion_unchecked`] calls — see that
    /// function's docs for the full rationale + the safe-test
    /// alternative.
    ///
    /// On an abelian group, the difference of two 2-torsion elements
    /// is also 2-torsion (the 2-torsion subgroup is closed under
    /// addition), so for a *valid* kernel bundle with
    /// `t1_minus_t2 = t1 - t2` the third check is redundant —
    /// but checking all three independently catches malformed
    /// bundles where the third field was constructed inconsistently.
    ///
    /// Useful for splitting-step boundary detection inside trusted
    /// chain code where curve membership is invariant.
    pub fn is_two_torsion_unchecked(&self) -> Choice {
        self.t1.is_two_torsion_unchecked()
            & self.t2.is_two_torsion_unchecked()
            & self.t1_minus_t2.is_two_torsion_unchecked()
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
            "failed to find a first deterministic liftable point on E_0",
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
            "failed to find a second deterministic distinct liftable point on E_0",
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
            "couple double must delegate to JacobianPoint::double on the E_1 half",
        );
        assert!(
            bool::from(doubled.p2.is_equivalent(&couple.p2.double(&curves.e2.a))),
            "couple double must delegate to JacobianPoint::double on the E_2 half",
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
            "double_iter(0, curves) must leave the couple point unchanged pointwise",
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
            "double_iter(3, curves) must match three doubles on the E_1 half",
        );
        assert!(
            bool::from(iterated.p2.is_equivalent(&repeated.p2)),
            "double_iter(3, curves) must match three doubles on the E_2 half",
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
            "to_couple_xz must delegate to JacobianPoint::to_montgomery_xz on the E_1 half",
        );
        assert!(
            bool::from(xz.p2.ct_eq(&couple.p2.to_montgomery_xz())),
            "to_couple_xz must delegate to JacobianPoint::to_montgomery_xz on the E_2 half",
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
            "add_components_pair test must use distinct E_1 inputs",
        );
        assert!(
            !bool::from(p.p2.is_equivalent(&q.p2)),
            "add_components_pair test must use distinct E_2 inputs",
        );

        let pair = p.add_components_pair(&q, &curves);
        let e1 = p.p1.add_components(&q.p1, &curves.e1.a);
        let e2 = p.p2.add_components(&q.p2, &curves.e2.a);

        assert_eq!(
            pair[0], e1,
            "add_components_pair[0] must match JacobianPoint::add_components on E_1",
        );
        assert_eq!(
            pair[1], e2,
            "add_components_pair[1] must match JacobianPoint::add_components on E_2",
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
            "doubling couple infinity must keep the E_1 half at infinity",
        );
        assert!(
            bool::from(doubled.p2.is_infinity()),
            "doubling couple infinity must keep the E_2 half at infinity",
        );
        assert_eq!(
            doubled, infinity,
            "doubling couple infinity must preserve the canonical pair of infinity sentinels",
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

    // Advisor item: mixed-infinity couple — `(O, P).double()` must
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
            "doubling (O, P) must keep the E_1 half at infinity",
        );
        let expected_2p = p.double(&curves.e2.a);
        assert!(
            bool::from(doubled_left.p2.is_equivalent(&expected_2p)),
            "doubling (O, P) must produce 2P on the E_2 half",
        );

        // (P, O) → (2P, O), symmetric case
        let right_inf = CoupleJacobianPoint::<F>::new(p, JacobianPoint::infinity());
        let doubled_right = right_inf.double(&curves);
        let expected_2p_e1 = p.double(&curves.e1.a);
        assert!(
            bool::from(doubled_right.p1.is_equivalent(&expected_2p_e1)),
            "doubling (P, O) must produce 2P on the E_1 half",
        );
        assert!(
            bool::from(doubled_right.p2.is_infinity()),
            "doubling (P, O) must keep the E_2 half at infinity",
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

    // Projective coordinate randomization tests for the couple-pair.
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
            "couple randomize must preserve E_1-half projective equivalence",
        );
        assert!(
            bool::from(randomized.p2.is_equivalent(&couple.p2)),
            "couple randomize must preserve E_2-half projective equivalence",
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

    // The two halves of a couple-randomize must
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
            "setup: input couple must have pointwise-identical halves",
        );
        let mut rng = fresh_rng(0x88);
        let randomized = couple.randomize_projective(&mut rng);
        assert!(
            !bool::from(randomized.p1.ct_eq_repr(&randomized.p2)),
            "couple randomize must use independent lambdas per half — \
             identical inputs must produce pointwise-different outputs \
             (probability of shared-lambda coincidence is ~2^-500)",
        );
        // But projective equivalence MUST still hold for each half — the
        // outputs are different bit patterns of the same affine point.
        assert!(
            bool::from(randomized.p1.is_equivalent(&couple.p1)),
            "independent-lambda blinding still preserves affine E_1 point",
        );
        assert!(
            bool::from(randomized.p2.is_equivalent(&couple.p2)),
            "independent-lambda blinding still preserves affine E_2 point",
        );
    }

    // EcBasis + ThetaKernelCouplePoints struct smoke tests.

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

        assert_eq!(basis.p, p_pt, "EcBasis.p preserved as-passed");
        assert_eq!(basis.q, q_pt, "EcBasis.q preserved as-passed");
        assert_eq!(
            basis.p_minus_q, pmq_pt,
            "EcBasis.p_minus_q preserved as-passed",
        );
    }

    // CoupleMontgomeryPoint::double + double_iter tests.

    /// Oracle: x_double((3, 1), a24 = 1) = (64, 192) per
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
        assert_eq!(doubled.p1.x, expected_x, "doubled.p1.x = 64");
        assert_eq!(doubled.p1.z, expected_z, "doubled.p1.z = 192");
        assert_eq!(doubled.p2.x, expected_x, "doubled.p2.x = 64");
        assert_eq!(doubled.p2.z, expected_z, "doubled.p2.z = 192");
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
        assert_eq!(r, p, "double_iter(0) = identity");
    }

    // CoupleMontgomeryPoint::to_affine.

    #[test]
    fn couple_montgomery_to_affine_componentwise_matches_per_half_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let one_v = Fp2::<Fp1Element>::one();
        let two = one_v.add(&one_v);
        let three = two.add(&one_v);

        let p = CoupleMontgomeryPoint::new(
            MontgomeryPoint::<Fp1Element>::new(two.add(&two).add(&two), two),
            MontgomeryPoint::<Fp1Element>::new(three.add(&three), two),
        );
        let affine = p.to_affine();
        assert_eq!(
            affine.p1,
            p.p1.to_affine(),
            "couple to_affine p1 must match per-half",
        );
        assert_eq!(
            affine.p2,
            p.p2.to_affine(),
            "couple to_affine p2 must match per-half",
        );
    }

    #[test]
    fn couple_montgomery_to_affine_of_infinity_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        let inf = CoupleMontgomeryPoint::<Fp1Element>::infinity();
        let affine_inf = inf.to_affine();
        assert!(
            bool::from(affine_inf.is_infinity()),
            "couple to_affine of (O, O) → (O, O)",
        );
    }

    // CoupleJacobianPoint::to_affine.

    #[test]
    fn couple_jacobian_to_affine_componentwise_matches_per_half_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let one_v = Fp2::<Fp1Element>::one();
        let two = one_v.add(&one_v);
        let three = two.add(&one_v);

        let p = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(one_v, two, three),
            JacobianPoint::<Fp1Element>::new(two, three, one_v),
        );
        let affine = p.to_affine();
        assert_eq!(
            affine.p1,
            p.p1.to_affine(),
            "to_affine p1 must match per-half",
        );
        assert_eq!(
            affine.p2,
            p.p2.to_affine(),
            "to_affine p2 must match per-half",
        );
    }

    #[test]
    fn couple_jacobian_to_affine_of_infinity_is_infinity_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        let inf = CoupleJacobianPoint::<Fp1Element>::infinity();
        let affine_inf = inf.to_affine();
        assert!(
            bool::from(affine_inf.is_infinity()),
            "to_affine of (O, O) must remain (O, O)",
        );
    }

    #[test]
    fn couple_jacobian_to_affine_idempotent_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let one_v = Fp2::<Fp1Element>::one();
        let two = one_v.add(&one_v);
        let three = two.add(&one_v);

        let p = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(one_v, two, three),
            JacobianPoint::<Fp1Element>::new(two, three, one_v),
        );
        let once = p.to_affine();
        let twice = once.to_affine();
        assert_eq!(once, twice, "to_affine is idempotent",);
    }

    // CoupleMontgomeryPoint::is_two_torsion + is_two_torsion_with_curves.

    #[test]
    fn couple_montgomery_is_two_torsion_true_when_both_halves_2t_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let zero = Fp2::<Fp1Element>::zero();
        let one_v = Fp2::<Fp1Element>::one();
        let imag = Fp2::<Fp1Element>::new(
            <Fp1Element as BaseField>::zero(),
            <Fp1Element as BaseField>::one(),
        );
        let a_e0 = zero; // E_0 has A = 0

        // (0 : 1) on E_0 for half 1, (i : 1) on E_0 for half 2.
        let p = CoupleMontgomeryPoint::new(
            MontgomeryPoint::<Fp1Element>::new(zero, one_v),
            MontgomeryPoint::<Fp1Element>::new(imag, one_v),
        );
        assert!(
            bool::from(p.is_two_torsion(&a_e0, &a_e0)),
            "couple with both halves 2-torsion → TRUE",
        );
    }

    #[test]
    fn couple_montgomery_is_two_torsion_false_when_one_half_not_2t_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let zero = Fp2::<Fp1Element>::zero();
        let one_v = Fp2::<Fp1Element>::one();
        let two_tors = MontgomeryPoint::<Fp1Element>::new(zero, one_v);
        let nontors = MontgomeryPoint::<Fp1Element>::new(one_v, one_v);
        let a_e0 = zero;

        let mixed_p1 = CoupleMontgomeryPoint::new(nontors, two_tors);
        let mixed_p2 = CoupleMontgomeryPoint::new(two_tors, nontors);
        assert!(
            !bool::from(mixed_p1.is_two_torsion(&a_e0, &a_e0)),
            "p1 non-2-torsion → FALSE",
        );
        assert!(
            !bool::from(mixed_p2.is_two_torsion(&a_e0, &a_e0)),
            "p2 non-2-torsion → FALSE",
        );
    }

    #[test]
    fn couple_montgomery_is_two_torsion_with_curves_matches_explicit_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let zero = Fp2::<Fp1Element>::zero();
        let one_v = Fp2::<Fp1Element>::one();
        let curves = CoupleCurve::<Fp1Element>::e0_e0();
        let p = CoupleMontgomeryPoint::new(
            MontgomeryPoint::<Fp1Element>::new(zero, one_v),
            MontgomeryPoint::<Fp1Element>::new(zero, one_v),
        );

        let via_wrapper = p.is_two_torsion_with_curves(&curves);
        let via_explicit = p.is_two_torsion(&curves.e1.a, &curves.e2.a);
        assert_eq!(
            bool::from(via_wrapper),
            bool::from(via_explicit),
            "with_curves variant must match explicit-A variant",
        );
        assert!(
            bool::from(via_wrapper),
            "(0:1, 0:1) on E_0 × E_0 is couple 2-torsion",
        );
    }

    // ThetaKernelCouplePoints::is_two_torsion.

    #[test]
    fn theta_kernel_is_two_torsion_true_when_all_three_2t_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let zero = Fp2::<Fp1Element>::zero();
        let one_v = Fp2::<Fp1Element>::one();
        let imag = Fp2::<Fp1Element>::new(
            <Fp1Element as BaseField>::zero(),
            <Fp1Element as BaseField>::one(),
        );

        // (0, 0, 1) on each half.
        let origin_origin = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(zero, zero, one_v),
            JacobianPoint::<Fp1Element>::new(zero, zero, one_v),
        );
        // (i, 0, 1) on each half.
        let i_i = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(imag, zero, one_v),
            JacobianPoint::<Fp1Element>::new(imag, zero, one_v),
        );
        // (-i, 0, 1) on each half.
        let neg_i_neg_i = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(imag.negate(), zero, one_v),
            JacobianPoint::<Fp1Element>::new(imag.negate(), zero, one_v),
        );

        let kernel = ThetaKernelCouplePoints::new(origin_origin, i_i, neg_i_neg_i);
        assert!(
            bool::from(kernel.is_two_torsion_unchecked()),
            "all three 2-torsion couples → kernel is_two_torsion TRUE",
        );
    }

    #[test]
    fn theta_kernel_is_two_torsion_false_when_one_field_not_2t_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let zero = Fp2::<Fp1Element>::zero();
        let one_v = Fp2::<Fp1Element>::one();

        let two_tors = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(zero, zero, one_v),
            JacobianPoint::<Fp1Element>::new(zero, zero, one_v),
        );
        let nontors = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(one_v, one_v, one_v),
            JacobianPoint::<Fp1Element>::new(one_v, one_v, one_v),
        );

        let k1 = ThetaKernelCouplePoints::new(nontors, two_tors, two_tors);
        let k2 = ThetaKernelCouplePoints::new(two_tors, nontors, two_tors);
        let k3 = ThetaKernelCouplePoints::new(two_tors, two_tors, nontors);

        for (label, k) in [
            ("t1 not 2t", k1),
            ("t2 not 2t", k2),
            ("t1_minus_t2 not 2t", k3),
        ] {
            assert!(
                !bool::from(k.is_two_torsion_unchecked()),
                "kernel with {label} must NOT be is_two_torsion",
            );
        }
    }

    #[test]
    fn theta_kernel_is_two_torsion_false_for_all_infinity_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        // All-infinity kernel: Y=1 (sentinel), Z=0 — is_two_torsion FALSE because Z=0 excludes.
        let inf = CoupleJacobianPoint::<Fp1Element>::infinity();
        let kernel = ThetaKernelCouplePoints::new(inf, inf, inf);
        assert!(
            !bool::from(kernel.is_two_torsion_unchecked()),
            "all-infinity kernel is NOT 2-torsion (Z=0 excludes)",
        );
    }

    // CoupleJacobianPoint::is_two_torsion.

    #[test]
    fn couple_jacobian_is_two_torsion_true_when_both_halves_two_torsion_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let zero = Fp2::<Fp1Element>::zero();
        let one_v = Fp2::<Fp1Element>::one();
        let imag = Fp2::<Fp1Element>::new(
            <Fp1Element as BaseField>::zero(),
            <Fp1Element as BaseField>::one(),
        );

        // (0, 0, 1) on E_0 is 2-torsion (the origin (0, 0)).
        let origin = JacobianPoint::<Fp1Element>::new(zero, zero, one_v);
        // (i, 0, 1) on E_0 is 2-torsion.
        let pos_i = JacobianPoint::<Fp1Element>::new(imag, zero, one_v);

        let couple = CoupleJacobianPoint::new(origin, pos_i);
        assert!(
            bool::from(couple.is_two_torsion_unchecked()),
            "couple with both halves 2-torsion must return TRUE",
        );
    }

    #[test]
    fn couple_jacobian_is_two_torsion_false_when_one_half_not_two_torsion_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let zero = Fp2::<Fp1Element>::zero();
        let one_v = Fp2::<Fp1Element>::one();

        let origin = JacobianPoint::<Fp1Element>::new(zero, zero, one_v);
        let nontors = JacobianPoint::<Fp1Element>::new(one_v, one_v, one_v);

        let mixed_p1 = CoupleJacobianPoint::new(nontors, origin);
        let mixed_p2 = CoupleJacobianPoint::new(origin, nontors);

        assert!(
            !bool::from(mixed_p1.is_two_torsion_unchecked()),
            "couple with p1 non-2-torsion is NOT 2-torsion",
        );
        assert!(
            !bool::from(mixed_p2.is_two_torsion_unchecked()),
            "couple with p2 non-2-torsion is NOT 2-torsion",
        );
    }

    #[test]
    fn couple_jacobian_is_two_torsion_false_for_infinity_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        // Infinity has Y=1 in canonical sentinel → is_two_torsion FALSE (Z=0 excludes).
        let inf = CoupleJacobianPoint::<Fp1Element>::infinity();
        assert!(
            !bool::from(inf.is_two_torsion_unchecked()),
            "(O, O) is NOT 2-torsion (Z=0 on both halves)",
        );
    }

    // ThetaKernelCouplePoints::is_infinity.

    #[test]
    fn theta_kernel_is_infinity_true_when_all_three_inf_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        let inf = CoupleJacobianPoint::<Fp1Element>::infinity();
        let kernel = ThetaKernelCouplePoints::new(inf, inf, inf);
        assert!(
            bool::from(kernel.is_infinity()),
            "all-infinity kernel must have is_infinity == TRUE",
        );
    }

    #[test]
    fn theta_kernel_is_infinity_false_when_one_field_finite_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;
        let one_v = Fp2::<Fp1Element>::one();
        let two = one_v.add(&one_v);
        let inf = CoupleJacobianPoint::<Fp1Element>::infinity();
        let finite = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(one_v, two, one_v),
            JacobianPoint::<Fp1Element>::new(one_v, two, one_v),
        );
        // Each of the three positions tested independently.
        let k1 = ThetaKernelCouplePoints::new(finite, inf, inf);
        let k2 = ThetaKernelCouplePoints::new(inf, finite, inf);
        let k3 = ThetaKernelCouplePoints::new(inf, inf, finite);
        for (label, k) in [
            ("t1 finite", k1),
            ("t2 finite", k2),
            ("t1_minus_t2 finite", k3),
        ] {
            assert!(
                !bool::from(k.is_infinity()),
                "kernel with {label} must NOT be is_infinity",
            );
        }
    }

    // ThetaKernelCouplePoints blinding methods.

    #[test]
    fn theta_kernel_randomize_preserves_affine_identity_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;
        use rand_chacha::ChaCha20Rng;
        use rand_core::SeedableRng;

        let one_v = Fp2::<Fp1Element>::one();
        let two = one_v.add(&one_v);
        let three = two.add(&one_v);

        let t1 = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(one_v, two, one_v),
            JacobianPoint::<Fp1Element>::new(three, one_v, two),
        );
        let t2 = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(two, three, one_v),
            JacobianPoint::<Fp1Element>::new(one_v, two, three),
        );
        let t1m2 = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(three, one_v, two),
            JacobianPoint::<Fp1Element>::new(two, three, one_v),
        );
        let kernel = ThetaKernelCouplePoints::new(t1, t2, t1m2);

        let mut rng = ChaCha20Rng::from_seed([0x42; 32]);
        let blinded = kernel.randomize_projective(&mut rng);

        // Affine identity (via is_equivalent) preserved on each field.
        assert!(
            bool::from(blinded.t1.p1.is_equivalent(&t1.p1)),
            "blinded.t1.p1 affinely equal to original",
        );
        assert!(
            bool::from(blinded.t1.p2.is_equivalent(&t1.p2)),
            "blinded.t1.p2 affinely equal to original",
        );
        assert!(
            bool::from(blinded.t2.p1.is_equivalent(&t2.p1)),
            "blinded.t2.p1 affinely equal to original",
        );
        assert!(
            bool::from(blinded.t2.p2.is_equivalent(&t2.p2)),
            "blinded.t2.p2 affinely equal to original",
        );
        assert!(
            bool::from(blinded.t1_minus_t2.p1.is_equivalent(&t1m2.p1)),
            "blinded.t1_minus_t2.p1 affinely equal to original",
        );
        assert!(
            bool::from(blinded.t1_minus_t2.p2.is_equivalent(&t1m2.p2)),
            "blinded.t1_minus_t2.p2 affinely equal to original",
        );
    }

    #[test]
    fn theta_kernel_randomize_uses_independent_lambdas_at_lvl1() {
        // With identical t1 = t2 = t1m2 inputs and a single rng,
        // randomize should produce DIFFERENT projective representations
        // for each (because each field consumes its own per-half lambdas).
        // This is the "independent per-call" doctrine in action.
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;
        use rand_chacha::ChaCha20Rng;
        use rand_core::SeedableRng;

        let one_v = Fp2::<Fp1Element>::one();
        let two = one_v.add(&one_v);
        let same = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(one_v, two, one_v),
            JacobianPoint::<Fp1Element>::new(one_v, two, one_v),
        );
        let kernel = ThetaKernelCouplePoints::new(same, same, same);

        let mut rng = ChaCha20Rng::from_seed([0x55; 32]);
        let blinded = kernel.randomize_projective(&mut rng);

        // The three blinded fields should NOT be byte-identical
        // (different lambdas → different projective coords) even
        // though their affine equivalence to `same` is preserved.
        assert!(
            blinded.t1 != blinded.t2 || blinded.t2 != blinded.t1_minus_t2,
            "blinded kernel fields must differ projectively (independent lambdas)",
        );
    }

    // ThetaKernelCouplePoints::to_couple_xz_triple.

    #[test]
    fn theta_kernel_to_couple_xz_triple_matches_per_field_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let one_v = Fp2::<Fp1Element>::one();
        let two = one_v.add(&one_v);
        let three = two.add(&one_v);

        let t1 = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(one_v, two, one_v),
            JacobianPoint::<Fp1Element>::new(three, one_v, two),
        );
        let t2 = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(two, three, one_v),
            JacobianPoint::<Fp1Element>::new(one_v, two, three),
        );
        let t1m2 = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(three, one_v, two),
            JacobianPoint::<Fp1Element>::new(two, three, one_v),
        );
        let kernel = ThetaKernelCouplePoints::new(t1, t2, t1m2);

        let triple = kernel.to_couple_xz_triple();

        assert_eq!(
            triple[0],
            t1.to_couple_xz(),
            "triple[0] = t1.to_couple_xz()"
        );
        assert_eq!(
            triple[1],
            t2.to_couple_xz(),
            "triple[1] = t2.to_couple_xz()"
        );
        assert_eq!(
            triple[2],
            t1m2.to_couple_xz(),
            "triple[2] = t1_minus_t2.to_couple_xz()"
        );
    }

    #[test]
    fn theta_kernel_to_couple_xz_triple_of_infinity_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        let inf = CoupleJacobianPoint::<Fp1Element>::infinity();
        let kernel = ThetaKernelCouplePoints::new(inf, inf, inf);
        let triple = kernel.to_couple_xz_triple();
        for (i, cmp) in triple.iter().enumerate() {
            assert!(
                bool::from(cmp.is_infinity()),
                "triple[{i}] of all-infinity kernel must be infinity",
            );
        }
    }

    // ThetaKernelCouplePoints::double_iter.

    #[test]
    fn theta_kernel_double_iter_zero_is_identity_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        let inf = CoupleJacobianPoint::<Fp1Element>::infinity();
        let kernel = ThetaKernelCouplePoints::new(inf, inf, inf);
        let curves = CoupleCurve::<Fp1Element>::e0_e0();
        let r = kernel.double_iter(0, &curves);
        assert_eq!(r.t1, inf, "double_iter(0).t1 = identity");
        assert_eq!(r.t2, inf, "double_iter(0).t2 = identity");
        assert_eq!(
            r.t1_minus_t2, inf,
            "double_iter(0).t1_minus_t2 = identity"
        );
    }

    #[test]
    fn theta_kernel_double_iter_componentwise_matches_per_field_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;

        let one_v = Fp2::<Fp1Element>::one();
        let two = one_v.add(&one_v);
        let three = two.add(&one_v);
        let curves = CoupleCurve::<Fp1Element>::e0_e0();

        let t1 = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(one_v, two, one_v),
            JacobianPoint::<Fp1Element>::new(three, one_v, two),
        );
        let t2 = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(two, three, one_v),
            JacobianPoint::<Fp1Element>::new(one_v, two, three),
        );
        let t1m2 = CoupleJacobianPoint::new(
            JacobianPoint::<Fp1Element>::new(three, one_v, two),
            JacobianPoint::<Fp1Element>::new(two, three, one_v),
        );
        let kernel = ThetaKernelCouplePoints::new(t1, t2, t1m2);

        for n in [1u32, 2, 3] {
            let r = kernel.double_iter(n, &curves);
            assert_eq!(
                r.t1,
                t1.double_iter(n, &curves),
                "t1 matches per-field"
            );
            assert_eq!(
                r.t2,
                t2.double_iter(n, &curves),
                "t2 matches per-field"
            );
            assert_eq!(
                r.t1_minus_t2,
                t1m2.double_iter(n, &curves),
                "t1_minus_t2 matches per-field",
            );
        }
    }

    // is_e0 + is_e0_e0 predicates.

    #[test]
    fn montgomery_curve_is_e0_true_for_e0_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        let e0 = MontgomeryCurve::<Fp1Element>::e0();
        assert!(
            bool::from(e0.is_e0()),
            "MontgomeryCurve::e0().is_e0() must be TRUE",
        );
    }

    #[test]
    fn montgomery_curve_is_e0_false_for_nonzero_a_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;
        let one_v = Fp2::<Fp1Element>::one();
        let curve = MontgomeryCurve::<Fp1Element>::new(one_v);
        assert!(
            !bool::from(curve.is_e0()),
            "MontgomeryCurve with A=1 must not be E_0",
        );
    }

    #[test]
    fn couple_curve_is_e0_e0_true_for_e0_e0_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        let cc = CoupleCurve::<Fp1Element>::e0_e0();
        assert!(
            bool::from(cc.is_e0_e0()),
            "e0_e0().is_e0_e0() must be TRUE",
        );
    }

    #[test]
    fn couple_curve_is_e0_e0_false_when_one_half_not_e0_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        use crate::gf::fp2::Fp2;
        let one_v = Fp2::<Fp1Element>::one();
        let mixed = CoupleCurve {
            e1: MontgomeryCurve::<Fp1Element>::e0(),
            e2: MontgomeryCurve::<Fp1Element>::new(one_v),
        };
        assert!(
            !bool::from(mixed.is_e0_e0()),
            "couple with one non-E_0 half is NOT (E_0, E_0)",
        );
    }

    // CoupleJacobianPoint::negate.

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
            "-(-p1) must equal p1 (projective)",
        );
        assert!(
            bool::from(neg_neg.p2.is_equivalent(&p.p2)),
            "-(-p2) must equal p2 (projective)",
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
        assert_eq!(neg.p1, p.p1.negate(), "negate p1 must match per-half");
        assert_eq!(neg.p2, p.p2.negate(), "negate p2 must match per-half");
    }

    #[test]
    fn couple_jacobian_negate_of_infinity_is_infinity_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        let inf = CoupleJacobianPoint::<Fp1Element>::infinity();
        let neg_inf = inf.negate();
        assert!(
            bool::from(neg_inf.is_infinity()),
            "-(O, O) must remain at infinity",
        );
    }

    // is_infinity predicates on couple types.

    #[test]
    fn couple_jacobian_is_infinity_true_for_infinity_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        let inf = CoupleJacobianPoint::<Fp1Element>::infinity();
        assert!(
            bool::from(inf.is_infinity()),
            "CoupleJacobianPoint::infinity().is_infinity() must be TRUE",
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
            "couple with one finite half is NOT infinity",
        );
        let mixed_other = CoupleJacobianPoint::new(finite_p, JacobianPoint::infinity());
        assert!(
            !bool::from(mixed_other.is_infinity()),
            "couple with one finite half (other side) is NOT infinity",
        );
    }

    #[test]
    fn couple_montgomery_is_infinity_true_for_infinity_at_lvl1() {
        use crate::gf::fp::Fp1Element;
        let inf = CoupleMontgomeryPoint::<Fp1Element>::infinity();
        assert!(
            bool::from(inf.is_infinity()),
            "CoupleMontgomeryPoint::infinity().is_infinity() must be TRUE",
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
            "couple-Montgomery with one finite half is NOT infinity",
        );
    }

    // CoupleMontgomeryPoint::ladder + ladder_with_curves tests.

    /// Oracle: scalar = [0] (zero) on a couple should produce
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
            "0 · P_1 must be infinity",
        );
        assert!(
            bool::from(r.p2.is_infinity()),
            "0 · P_2 must be infinity",
        );
    }

    /// confirm componentwise ladder matches per-half ladder
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
            "couple ladder.p1 must match per-half ladder on E_1",
        );
        assert_eq!(
            via_couple.p2, solo_2,
            "couple ladder.p2 must match per-half ladder on E_2",
        );
    }

    /// confirm `ladder_with_curves` matches the explicit-a24
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
            "ladder_with_curves must match ladder with explicit a24",
        );
    }

    /// confirm `double_with_curves` produces the same output as
    /// the explicit-a24 variant. Same input fixture as the oracle test
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
            "double_with_curves must match double with explicit a24",
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
        assert_eq!(r, p, "double_iter_with_curves(0) = identity");
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
                "double_iter_with_curves({n}) must match explicit-a24 variant",
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
        assert_eq!(r_iter, r_solo, "double_iter(1) = double");
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

        assert_eq!(kernel.t1, t1, "kernel.t1 preserved");
        assert_eq!(kernel.t2, t2, "kernel.t2 preserved");
        assert_eq!(
            kernel.t1_minus_t2, t1m2,
            "kernel.t1_minus_t2 preserved"
        );
    }
}
