// SPDX-License-Identifier: MIT OR Apache-2.0
//! Montgomery curves and x-only point arithmetic over `F_{p^2}`.
//!
//! A curve is given by its Montgomery coefficient `A ∈ F_{p^2}`; the
//! equation is `B y^2 = x^3 + A x^2 + x` with `B = 1` (every SQIsign curve
//! uses the `B = 1` form). Points are projective `(X : Z)` pairs; the
//! identity is `(1 : 0)`.
//!
//! The three primitives the higher-level isogeny code needs are:
//! - [`MontgomeryPoint::x_double`] — `xDBL`: `[2]P` from `P` using `(A+2)/4`.
//! - [`MontgomeryPoint::x_add`] — `xADD`: `P + Q` knowing `P − Q`.
//! - [`MontgomeryPoint::ladder`] — Montgomery ladder for scalar `k · P`.
//! - [`MontgomeryCurve::j_invariant`] — the curve's `j` invariant in `F_{p^2}`.
//!
//! Implementations follow Costello-Hisil, "A simple and compact algorithm
//! for SIDH with arbitrary degree isogenies" (PQC 2017), §2.

use core::marker::PhantomData;

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use crate::gf::fp::BaseField;
use crate::gf::fp2::Fp2;

/// Projective x-only Montgomery point `(X : Z)`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MontgomeryPoint<F: BaseField> {
    /// `X` coordinate.
    pub x: Fp2<F>,
    /// `Z` coordinate. The affine x-coordinate is `X / Z`.
    pub z: Fp2<F>,
    _marker: PhantomData<F>,
}

impl<F: BaseField> MontgomeryPoint<F> {
    /// Construct a projective `(X : Z)` point.
    #[inline]
    pub const fn new(x: Fp2<F>, z: Fp2<F>) -> Self {
        Self {
            x,
            z,
            _marker: PhantomData,
        }
    }

    /// Identity element `(1 : 0)`.
    #[inline]
    pub fn infinity() -> Self {
        Self::new(Fp2::one(), Fp2::zero())
    }

    /// `Choice::TRUE` iff this point is the identity.
    #[inline]
    pub fn is_infinity(&self) -> Choice {
        self.z.is_zero()
    }

    /// `Choice::TRUE` iff this is a finite 2-torsion point on the
    /// Montgomery curve `y² = x³ + Ax² + x` (affine `a`).
    ///
    /// Algebra: a point `P = (X : Z)` is 2-torsion iff `y² = 0` at
    /// `x = X/Z`. Substituting and clearing denominators yields the
    /// projective condition:
    ///
    /// ```text
    /// X · (X² + A · X · Z + Z²) == 0
    /// ```
    ///
    /// which factors into `X == 0` (the `(0, 0)` 2-torsion point) or
    /// `X² + A·X·Z + Z² == 0` (the two `(±i, 0)`-type roots on E_0,
    /// or the analogous pair on any Montgomery curve). The
    /// `Z ≠ 0` clause excludes the infinity sentinel `(1 : 0)`.
    ///
    /// Companion to [`crate::ec::jacobian::JacobianPoint::is_two_torsion`]
    /// for the x-only side of the chain layer.
    pub fn is_two_torsion(&self, a: &Fp2<F>) -> Choice {
        // Non-infinity: Z != 0.
        let is_finite = !self.z.is_zero();
        // X == 0 case.
        let x_zero = self.x.is_zero();
        // X² + A·X·Z + Z² == 0 case.
        let x_sq = self.x.square();
        let z_sq = self.z.square();
        let a_x_z = a.mul(&self.x).mul(&self.z);
        let quadratic = x_sq.add(&a_x_z).add(&z_sq);
        let quadratic_zero = quadratic.is_zero();
        is_finite & (x_zero | quadratic_zero)
    }

    /// Affine x-coordinate of a non-identity point, i.e. `X / Z`.
    pub fn affine_x(&self) -> Fp2<F> {
        let z_inv = self.z.invert();
        let zinv = z_inv.unwrap_or(Fp2::zero());
        self.x.mul(&zinv)
    }

    /// Normalize the projective `(X : Z)` representative to its
    /// affine form `(X/Z : 1)`.
    ///
    /// Companion to [`Self::affine_x`] (which returns just the
    /// affine x-coordinate as an `Fp2`). The full self-typed
    /// normalization preserves the type for callers that need a
    /// canonical projective representative downstream.
    ///
    /// Infinity `(1 : 0)` is returned unchanged — Z=0 can't be
    /// inverted, and infinity's canonical representative is itself.
    pub fn to_affine(&self) -> Self {
        let is_inf = self.z.is_zero();
        let z_inv = self.z.invert().unwrap_or(Fp2::zero());
        let affine_x = self.x.mul(&z_inv);
        let affine_point = Self::new(affine_x, Fp2::one());
        Self::conditional_select(&affine_point, &Self::infinity(), is_inf)
    }

    /// `[2] · self` on the Montgomery curve with reduced parameter
    /// `a24 = (A + 2) / 4 ∈ F_{p^2}`.
    ///
    /// Cost: 4M + 2S in `F_{p^2}` (the standard xDBL formula).
    pub fn x_double(&self, a24: &Fp2<F>) -> Self {
        let v1 = self.x.add(&self.z);
        let v1 = v1.square();
        let v2 = self.x.sub(&self.z);
        let v2 = v2.square();
        let x_out = v1.mul(&v2);
        let v1 = v1.sub(&v2);
        let v3 = a24.mul(&v1);
        let v3 = v3.add(&v2);
        let z_out = v1.mul(&v3);
        Self::new(x_out, z_out)
    }

    /// Differential addition `xADD`: returns `self + rhs` given the x-only
    /// representation of `self − rhs` (the "minus" point). Cost: 4M + 2S.
    pub fn x_add(&self, rhs: &Self, minus: &Self) -> Self {
        let v0 = self.x.add(&self.z);
        let v1 = rhs.x.sub(&rhs.z);
        let v1 = v1.mul(&v0);
        let v0 = self.x.sub(&self.z);
        let v2 = rhs.x.add(&rhs.z);
        let v2 = v2.mul(&v0);
        let sum = v1.add(&v2);
        let dif = v1.sub(&v2);
        let x = sum.square().mul(&minus.z);
        let z = dif.square().mul(&minus.x);
        Self::new(x, z)
    }

    /// Combined doubling and differential addition: `(2·P, P+Q)` from
    /// `(P, Q, P−Q)`. Saves redundant work in the Montgomery ladder.
    /// Cost: 8M + 4S.
    pub fn x_dbl_add(p: &Self, q: &Self, minus: &Self, a24: &Fp2<F>) -> (Self, Self) {
        let dbl = p.x_double(a24);
        let add = p.x_add(q, minus);
        (dbl, add)
    }

    /// Constant-time Montgomery scalar ladder: returns `k · self` where
    /// `scalar` is interpreted big-endian, ignoring `top_bit` bits above
    /// `scalar.len() * 8`. `a24 = (A + 2) / 4`.
    pub fn ladder(&self, scalar: &[u8], a24: &Fp2<F>) -> Self {
        let mut r0 = Self::infinity();
        let mut r1 = *self;
        let minus = *self;
        let n_bits = scalar.len() * 8;
        let mut i = n_bits;
        while i > 0 {
            i -= 1;
            let byte = scalar[i / 8];
            let bit = Choice::from((byte >> (i % 8)) & 1);
            let (a, b) = swap(&r0, &r1, bit);
            r0 = a;
            r1 = b;
            let (new_r0, new_r1) = Self::x_dbl_add(&r0, &r1, &minus, a24);
            r0 = new_r0;
            r1 = new_r1;
            let (a, b) = swap(&r0, &r1, bit);
            r0 = a;
            r1 = b;
        }
        r0
    }
}

fn swap<F: BaseField>(
    a: &MontgomeryPoint<F>,
    b: &MontgomeryPoint<F>,
    choice: Choice,
) -> (MontgomeryPoint<F>, MontgomeryPoint<F>) {
    let new_a = MontgomeryPoint::conditional_select(a, b, choice);
    let new_b = MontgomeryPoint::conditional_select(b, a, choice);
    (new_a, new_b)
}

impl<F: BaseField> ConstantTimeEq for MontgomeryPoint<F> {
    fn ct_eq(&self, other: &Self) -> Choice {
        // (X1 : Z1) ~ (X2 : Z2) iff X1 Z2 = X2 Z1
        let a = self.x.mul(&other.z);
        let b = other.x.mul(&self.z);
        a.ct_eq(&b)
    }
}

impl<F: BaseField> ConditionallySelectable for MontgomeryPoint<F> {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self::new(
            Fp2::conditional_select(&a.x, &b.x, choice),
            Fp2::conditional_select(&a.z, &b.z, choice),
        )
    }
}

/// A Montgomery curve `E_A : y^2 = x^3 + A x^2 + x` over `F_{p^2}`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MontgomeryCurve<F: BaseField> {
    /// Curve coefficient `A`.
    pub a: Fp2<F>,
    _marker: PhantomData<F>,
}

impl<F: BaseField> MontgomeryCurve<F> {
    /// Construct from a coefficient `A ∈ F_{p^2}`.
    #[inline]
    pub const fn new(a: Fp2<F>) -> Self {
        Self {
            a,
            _marker: PhantomData,
        }
    }

    /// The "starting curve" `E_0 : y^2 = x^3 + x` (coefficient `A = 0`).
    /// SQIsign's secret-isogeny chain begins at this curve.
    #[inline]
    pub fn e0() -> Self {
        Self::new(Fp2::zero())
    }

    /// `Choice::TRUE` iff `self` is the starting curve `E_0`
    /// (`A == 0` in affine Montgomery form).
    ///
    /// Predicate companion to [`Self::e0`]. Useful for chain-init
    /// assertions and for fast-path branches that specialize to
    /// `E_0`'s known constants (e.g., `a24 = 1/2`).
    #[inline]
    pub fn is_e0(&self) -> Choice {
        self.a.is_zero()
    }

    /// Constant `a24 = (A + 2) / 4` used by xDBL / xLAD / xDBLADD.
    pub fn a24(&self) -> Fp2<F> {
        let two = Fp2::one().double();
        let four = two.double();
        let inv_four = four.invert().unwrap_or(Fp2::zero());
        self.a.add(&two).mul(&inv_four)
    }

    /// Build the projective curve representation `(A24 : C24) = (A + 2 : 4)`
    /// — the form the isogeny pipeline operates in.
    pub fn to_a24(&self) -> CurveA24<F> {
        let two = Fp2::one().double();
        let four = two.double();
        CurveA24::new(self.a.add(&two), four)
    }

    /// `j`-invariant `j(E) = 256 (A^2 − 3)^3 / (A^2 − 4)`.
    pub fn j_invariant(&self) -> Fp2<F> {
        let a2 = self.a.square();
        let three = Fp2::one().double().add(&Fp2::one());
        let four = three.add(&Fp2::one());
        let num0 = a2.sub(&three);
        let num = num0.square().mul(&num0); // (A^2 − 3)^3
        let den = a2.sub(&four);
        let den_inv = den.invert().unwrap_or(Fp2::zero());
        // 256 · num / den
        let mut k = Fp2::<F>::one();
        for _ in 0..8 {
            k = k.double();
        }
        k.mul(&num).mul(&den_inv)
    }
}

/// Projective Montgomery curve representation `(A24 : C24)` where
/// `a24 = (A + 2)/4 = A24 / C24`.
///
/// This is the form the isogeny pipeline operates in — it avoids the per-step
/// division that the affine `A` form would require when composing many
/// 2-isogenies along a chain.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CurveA24<F: BaseField> {
    /// `A24` coordinate (numerator of `(A + 2)/4`).
    pub a24: Fp2<F>,
    /// `C24` coordinate (denominator of `(A + 2)/4`).
    pub c24: Fp2<F>,
    _marker: PhantomData<F>,
}

impl<F: BaseField> CurveA24<F> {
    /// Construct directly from projective coordinates.
    #[inline]
    pub const fn new(a24: Fp2<F>, c24: Fp2<F>) -> Self {
        Self {
            a24,
            c24,
            _marker: PhantomData,
        }
    }

    /// `E_0 : y² = x³ + x` in `(A24 : C24)` form. `A = 0` ⇒ `(2 : 4)`.
    pub fn e0() -> Self {
        let two = Fp2::one().double();
        Self::new(two, two.double())
    }

    /// Recover the affine coefficient `A = 4 A24 / C24 − 2`.
    pub fn to_affine_a(&self) -> Fp2<F> {
        let inv = self.c24.invert().unwrap_or(Fp2::zero());
        let four = Fp2::one().double().double();
        let two = Fp2::one().double();
        self.a24.mul(&four).mul(&inv).sub(&two)
    }

    /// Affine `a24 = A24 / C24`. Convenience for `MontgomeryPoint::ladder`.
    pub fn a24_affine(&self) -> Fp2<F> {
        let inv = self.c24.invert().unwrap_or(Fp2::zero());
        self.a24.mul(&inv)
    }

    /// `[2] · P` on this curve, working entirely in projective `(A24:C24)`
    /// form. Cost: 4M + 2S in `F_{p^2}`.
    pub fn x_double(&self, p: &MontgomeryPoint<F>) -> MontgomeryPoint<F> {
        let v1 = p.x.add(&p.z).square();
        let v2 = p.x.sub(&p.z).square();
        let v3 = v1.sub(&v2);
        let x_out = v1.mul(&self.c24).mul(&v2);
        // Z_out = (A24 · v3 + C24 · v2) · v3
        let z_out = self.a24.mul(&v3).add(&self.c24.mul(&v2)).mul(&v3);
        MontgomeryPoint::new(x_out, z_out)
    }

    /// Iterated doubling: `[2^n] · P`.
    pub fn x_double_n(&self, p: &MontgomeryPoint<F>, n: u32) -> MontgomeryPoint<F> {
        let mut q = *p;
        for _ in 0..n {
            q = self.x_double(&q);
        }
        q
    }
}

impl<F: BaseField> MontgomeryPoint<F> {
    /// x-only 3-point ladder — computes `P + [k] Q` given `P`, `Q`, and the
    /// x-coordinate of `P − Q`. This is the standard SIDH-style routine used
    /// to evaluate a secret-key scalar over a basis `(P, Q)` without ever
    /// computing y-coordinates. `k` is given in little-endian byte form.
    ///
    /// `a24` is `(A + 2)/4` of the curve in affine `F_{p^2}` form.
    pub fn ladder3pt(p: &Self, q: &Self, p_minus_q: &Self, k: &[u8], a24: &Fp2<F>) -> Self {
        let mut r0 = *q;
        let mut r1 = *p;
        let mut r2 = *p_minus_q;
        for &byte in k {
            for bit_idx in 0..8 {
                let bit = Choice::from((byte >> bit_idx) & 1);
                // Conditionally swap (R1, R2) using bit.
                let (a, b) = swap(&r1, &r2, bit);
                r1 = a;
                r2 = b;
                let new_r2 = r1.x_add(&r0, &r2);
                let new_r0 = r0.x_double(a24);
                r2 = new_r2;
                r0 = new_r0;
                let (a, b) = swap(&r1, &r2, bit);
                r1 = a;
                r2 = b;
            }
        }
        r1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf::fp::Fp1Element;

    type Curve = MontgomeryCurve<Fp1Element>;
    type Curve24 = CurveA24<Fp1Element>;

    #[test]
    fn e0_j_invariant_is_1728() {
        // For E_0 : y^2 = x^3 + x we have A = 0, hence
        //   j = 256 · (A^2 − 3)^3 / (A^2 − 4) = 256 · (−3)^3 / (−4)
        //     = 256 · 27 / 4 = 1728.
        let j = Curve::e0().j_invariant();
        // Build 1728 = 2^10 + 2^9 + 2^7 + 2^6 in F_{p^2} via repeated doubling.
        let one = Fp2::<Fp1Element>::one();
        let mut acc = Fp2::<Fp1Element>::zero();
        for &b in &[10u32, 9, 7, 6] {
            let mut p2 = one;
            for _ in 0..b {
                p2 = p2.double();
            }
            acc = acc.add(&p2);
        }
        assert_eq!(j, acc);
    }

    #[test]
    fn ladder3pt_at_zero_returns_p() {
        // `[0] · Q + P = P`. The 3-point ladder fed scalar zero must return P
        // unchanged (projectively).
        let curve = Curve::e0();
        let a24 = curve.a24();
        let p = MontgomeryPoint::<Fp1Element>::new(Fp2::one().double(), Fp2::one()); // x = 2
        let q = MontgomeryPoint::<Fp1Element>::new(
            Fp2::one().double().double(), // x = 4
            Fp2::one(),
        );
        let p_minus_q = MontgomeryPoint::<Fp1Element>::new(
            Fp2::one().double().double().double(), // x = 8
            Fp2::one(),
        );
        let zero_k = [0u8; 4];
        let r = MontgomeryPoint::ladder3pt(&p, &q, &p_minus_q, &zero_k, &a24);
        // r should equal p projectively: r.x * p.z == p.x * r.z
        let lhs = r.x.mul(&p.z);
        let rhs = p.x.mul(&r.z);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn curve_a24_round_trip_affine() {
        // E_0 in A24 form should recover A = 0 when converted back.
        let proj = Curve24::e0();
        let a_back = proj.to_affine_a();
        assert_eq!(a_back, Fp2::<Fp1Element>::zero());
    }

    #[test]
    fn curve_a24_double_matches_affine_double() {
        // Doubling on the projective curve must agree with doubling using the
        // affine a24 = (A+2)/4 input to MontgomeryPoint::x_double.
        let curve = Curve::e0();
        let a24_affine = curve.a24();
        let proj = curve.to_a24();
        // A generic projective point on E_0 (we don't need it to be on the curve
        // for the x-only doubling formula to give a consistent answer between
        // the two doubling routines — they must agree for ANY (X:Z)).
        let p = MontgomeryPoint::<Fp1Element>::new(
            Fp2::one().double().double(), // x = 4
            Fp2::one(),
        );
        let q_affine = p.x_double(&a24_affine);
        let q_proj = proj.x_double(&p);
        // Both should represent the same projective point; compare via cross-mul.
        let lhs = q_affine.x.mul(&q_proj.z);
        let rhs = q_proj.x.mul(&q_affine.z);
        assert_eq!(lhs, rhs);
    }

    // ── S89 — EC operations at production NIST levels ──

    /// Generic helper: at any BaseField level, E_0's j-invariant equals
    /// 1728 = 2^10 + 2^9 + 2^7 + 2^6, the well-known j-invariant of the
    /// curve `y² = x³ + x` (A = 0 Montgomery form).
    fn check_e0_j_invariant_is_1728<F: BaseField>() {
        let j = MontgomeryCurve::<F>::e0().j_invariant();
        let one = Fp2::<F>::one();
        let mut acc = Fp2::<F>::zero();
        for &b in &[10u32, 9, 7, 6] {
            let mut p2 = one;
            for _ in 0..b {
                p2 = p2.double();
            }
            acc = acc.add(&p2);
        }
        assert_eq!(j, acc, "S89: E_0 j-invariant must equal 1728");
    }

    #[test]
    fn e0_j_invariant_is_1728_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_e0_j_invariant_is_1728::<Fp3Element>();
    }

    #[test]
    fn e0_j_invariant_is_1728_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_e0_j_invariant_is_1728::<Fp5Element>();
    }

    /// Generic helper: at any BaseField level, E_0 in `CurveA24` form
    /// recovers `A = 0` when converted back to affine.
    fn check_curve_a24_round_trip_affine<F: BaseField>() {
        let proj = CurveA24::<F>::e0();
        let a_back = proj.to_affine_a();
        assert_eq!(
            a_back,
            Fp2::<F>::zero(),
            "S89: CurveA24::e0() must round-trip to affine A = 0",
        );
    }

    #[test]
    fn curve_a24_round_trip_affine_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_curve_a24_round_trip_affine::<Fp3Element>();
    }

    #[test]
    fn curve_a24_round_trip_affine_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_curve_a24_round_trip_affine::<Fp5Element>();
    }

    /// Generic helper: at any BaseField level, `CurveA24::x_double` and
    /// `MontgomeryPoint::x_double` must agree on E_0 (the existing L1
    /// test only covers Fp1Element; this generalises across levels).
    fn check_curve_a24_double_matches_affine<F: BaseField>() {
        let curve = MontgomeryCurve::<F>::e0();
        let a24_affine = curve.a24();
        let proj = curve.to_a24();
        let p = MontgomeryPoint::<F>::new(Fp2::<F>::one().double(), Fp2::<F>::one());
        let q_affine = p.x_double(&a24_affine);
        let q_proj = proj.x_double(&p);
        let lhs = q_affine.x.mul(&q_proj.z);
        let rhs = q_proj.x.mul(&q_affine.z);
        assert_eq!(
            lhs, rhs,
            "S89: CurveA24::x_double must match affine x_double on E_0",
        );
    }

    #[test]
    fn curve_a24_double_matches_affine_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_curve_a24_double_matches_affine::<Fp3Element>();
    }

    #[test]
    fn curve_a24_double_matches_affine_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_curve_a24_double_matches_affine::<Fp5Element>();
    }

    // ── S91 — ladder / ladder3pt at production NIST levels ──

    /// Generic helper: `ladder3pt(p, q, p−q, [0]·k, a24)` returns `p`
    /// unchanged (projectively). The L1 version exists as
    /// `ladder3pt_at_zero_returns_p`; this lifts it to a generic helper
    /// so L3/L5 can invoke.
    fn check_ladder3pt_at_zero_returns_p<F: BaseField>() {
        let curve = MontgomeryCurve::<F>::e0();
        let a24 = curve.a24();
        let p = MontgomeryPoint::<F>::new(Fp2::<F>::one().double(), Fp2::<F>::one()); // x = 2
        let q = MontgomeryPoint::<F>::new(
            Fp2::<F>::one().double().double(), // x = 4
            Fp2::<F>::one(),
        );
        let p_minus_q = MontgomeryPoint::<F>::new(
            Fp2::<F>::one().double().double().double(), // x = 8
            Fp2::<F>::one(),
        );
        let zero_k = [0u8; 4];
        let r = MontgomeryPoint::ladder3pt(&p, &q, &p_minus_q, &zero_k, &a24);
        // r should equal p projectively: r.x · p.z == p.x · r.z.
        let lhs = r.x.mul(&p.z);
        let rhs = p.x.mul(&r.z);
        assert_eq!(
            lhs, rhs,
            "S91: ladder3pt with scalar [0] must return p (projectively) at this level",
        );
    }

    #[test]
    fn ladder3pt_at_zero_returns_p_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_ladder3pt_at_zero_returns_p::<Fp3Element>();
    }

    #[test]
    fn ladder3pt_at_zero_returns_p_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_ladder3pt_at_zero_returns_p::<Fp5Element>();
    }

    /// Generic helper: `ladder([0], a24)` returns the point at infinity
    /// for any base point on any base-field level.
    fn check_ladder_at_zero_returns_infinity<F: BaseField>() {
        let curve = MontgomeryCurve::<F>::e0();
        let a24 = curve.a24();
        let p = MontgomeryPoint::<F>::new(Fp2::<F>::one().double(), Fp2::<F>::one()); // x = 2
        let zero_k = [0u8; 4];
        let r = p.ladder(&zero_k, &a24);
        assert!(
            bool::from(r.is_infinity()),
            "S91: ladder([0]) must return point at infinity at this level",
        );
    }

    #[test]
    fn ladder_at_zero_returns_infinity_at_lvl1() {
        check_ladder_at_zero_returns_infinity::<Fp1Element>();
    }

    #[test]
    fn ladder_at_zero_returns_infinity_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_ladder_at_zero_returns_infinity::<Fp3Element>();
    }

    #[test]
    fn ladder_at_zero_returns_infinity_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_ladder_at_zero_returns_infinity::<Fp5Element>();
    }

    /// Generic helper: `ladder([1], a24)` returns the point itself
    /// (projectively). Tests the "scalar one is identity" property of
    /// the ladder algorithm at any base-field level.
    fn check_ladder_at_one_returns_self<F: BaseField>() {
        let curve = MontgomeryCurve::<F>::e0();
        let a24 = curve.a24();
        let p = MontgomeryPoint::<F>::new(Fp2::<F>::one().double(), Fp2::<F>::one()); // x = 2
        let one_k = [1u8, 0, 0, 0]; // LE: scalar = 1
        let r = p.ladder(&one_k, &a24);
        // r should equal p projectively.
        let lhs = r.x.mul(&p.z);
        let rhs = p.x.mul(&r.z);
        assert_eq!(
            lhs, rhs,
            "S91: ladder([1]) must return self (projectively) at this level",
        );
    }

    #[test]
    fn ladder_at_one_returns_self_at_lvl1() {
        check_ladder_at_one_returns_self::<Fp1Element>();
    }

    #[test]
    fn ladder_at_one_returns_self_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_ladder_at_one_returns_self::<Fp3Element>();
    }

    #[test]
    fn ladder_at_one_returns_self_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_ladder_at_one_returns_self::<Fp5Element>();
    }

    // S176 — MontgomeryPoint::to_affine tests.

    fn check_to_affine_normalizes<F: BaseField>() {
        let one = Fp2::<F>::one();
        let two = one.add(&one);
        let three = two.add(&one);

        // (X, Z) = (6, 2) → affine x = 3 → to_affine should produce (3, 1).
        let six = two.add(&two).add(&two);
        let p = MontgomeryPoint::<F>::new(six, two);
        let affine = p.to_affine();
        assert_eq!(affine.x, three, "S176: (6, 2).to_affine().x = 3");
        assert_eq!(affine.z, one, "S176: to_affine().z = 1 for non-infinity");
    }

    fn check_to_affine_preserves_infinity<F: BaseField>() {
        let inf = MontgomeryPoint::<F>::infinity();
        let affine_inf = inf.to_affine();
        assert!(
            bool::from(affine_inf.is_infinity()),
            "S176: to_affine of infinity must remain infinity",
        );
    }

    fn check_to_affine_idempotent<F: BaseField>() {
        let one = Fp2::<F>::one();
        let two = one.add(&one);
        let p = MontgomeryPoint::<F>::new(two.add(&two).add(&two), two);
        let once = p.to_affine();
        let twice = once.to_affine();
        assert_eq!(once, twice, "S176: to_affine is idempotent");
    }

    #[test]
    fn montgomery_to_affine_normalizes_at_lvl1() {
        check_to_affine_normalizes::<Fp1Element>();
    }

    #[test]
    fn montgomery_to_affine_normalizes_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_to_affine_normalizes::<Fp3Element>();
    }

    #[test]
    fn montgomery_to_affine_normalizes_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_to_affine_normalizes::<Fp5Element>();
    }

    #[test]
    fn montgomery_to_affine_preserves_infinity_at_lvl1() {
        check_to_affine_preserves_infinity::<Fp1Element>();
    }

    #[test]
    fn montgomery_to_affine_idempotent_at_lvl1() {
        check_to_affine_idempotent::<Fp1Element>();
    }

    // S173 — MontgomeryPoint::is_two_torsion (x-only form) tests.

    fn check_is_two_torsion_xz_predicate<F: BaseField>() {
        let a_zero = Fp2::<F>::zero(); // E_0
        let zero = Fp2::<F>::zero();
        let one = Fp2::<F>::one();
        let imag = Fp2::<F>::new(F::zero(), F::one());

        // (0 : 1) on E_0 — the (0, 0) 2-torsion point.
        let origin = MontgomeryPoint::<F>::new(zero, one);
        assert!(
            bool::from(origin.is_two_torsion(&a_zero)),
            "S173: x-only (0:1) on E_0 must be 2-torsion",
        );

        // (i : 1) on E_0 — root of X² + Z² = 0 since A=0: 1·(-1) + 1 = 0.
        let pos_i = MontgomeryPoint::<F>::new(imag, one);
        assert!(
            bool::from(pos_i.is_two_torsion(&a_zero)),
            "S173: x-only (i:1) on E_0 must be 2-torsion",
        );

        // (-i : 1) on E_0 — same as (i:1) under squaring.
        let neg_i = MontgomeryPoint::<F>::new(imag.negate(), one);
        assert!(
            bool::from(neg_i.is_two_torsion(&a_zero)),
            "S173: x-only (-i:1) on E_0 must be 2-torsion",
        );

        // Infinity (1 : 0) — Z=0 excludes per predicate.
        let inf = MontgomeryPoint::<F>::infinity();
        assert!(
            !bool::from(inf.is_two_torsion(&a_zero)),
            "S173: x-only infinity is NOT 2-torsion (Z=0 excludes)",
        );

        // (1 : 1) — affine x=1, not a 2-torsion x-coord on E_0
        // (1·(1² + 0·1·1 + 1²) = 1·2 = 2 ≠ 0 at L1's prime).
        let nontors = MontgomeryPoint::<F>::new(one, one);
        assert!(
            !bool::from(nontors.is_two_torsion(&a_zero)),
            "S173: x-only (1:1) on E_0 must NOT be 2-torsion",
        );
    }

    #[test]
    fn montgomery_is_two_torsion_predicate_at_lvl1() {
        check_is_two_torsion_xz_predicate::<Fp1Element>();
    }

    #[test]
    fn montgomery_is_two_torsion_predicate_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_is_two_torsion_xz_predicate::<Fp3Element>();
    }

    #[test]
    fn montgomery_is_two_torsion_predicate_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_is_two_torsion_xz_predicate::<Fp5Element>();
    }
}
