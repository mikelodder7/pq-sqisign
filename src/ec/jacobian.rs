// SPDX-License-Identifier: MIT OR Apache-2.0
//! Jacobian-coordinate point representation on Montgomery curves over `F_{p^2}`.
//!
//! The Jacobian equivalence is `(X : Y : Z) ~ (λ²X : λ³Y : λZ)` for any
//! `λ ∈ F_q*`. See SQIsign 2.0.1 spec §8.2 for the conversion between
//! Montgomery x-only coordinates and Jacobian coordinates, and Alg 8.11
//! (DBL). ADD (8.12) and ADDComponents (8.13) are deferred to S127.
//!
//! ## Infinity sentinel
//!
//! The point at infinity uses the canonical encoding `(1, 1, 0)`. The
//! spec's `(0, 1, 0)` would algebraically collide with transient `(0, *, 0)`
//! states during the constant-time complete ADD formula's intermediate
//! computations, while `(1, 1, 0)` cannot arise as a non-infinity
//! intermediate. [`JacobianPoint::is_infinity`] tests `Z == 0` only; the
//! `X = 1`, `Y = 1` components are convention, not predicates.
//!
//! ## Equality
//!
//! Three distinct equality surfaces are exposed (semantic vs representational):
//! - [`ConstantTimeEq::ct_eq`] tests PROJECTIVE equivalence — delegates to
//!   [`JacobianPoint::is_equivalent`]. This is the **semantic** equality and
//!   the trait-contract-correct choice for a projective type. Two Jacobian
//!   triples representing the same affine point under `(X : Y : Z) ~
//!   (λ²X : λ³Y : λZ)` compare equal regardless of `λ`.
//! - [`JacobianPoint::is_equivalent`] is the same projective check exposed
//!   directly (for code that wants to be explicit about semantic vs
//!   representational comparison).
//! - [`JacobianPoint::ct_eq_repr`] / [`PartialEq`] / [`Eq`] test POINTWISE
//!   field equality of `(X, Y, Z)`. Use only when the representation itself
//!   matters (canonical-form checks, debug printing, round-trip assertions).

use core::marker::PhantomData;

use rand_core::CryptoRng;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use crate::gf::fp::BaseField;
use crate::gf::fp2::Fp2;

/// Jacobian point `(X : Y : Z)` on a Montgomery curve over `F_{p^2}`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct JacobianPoint<F: BaseField> {
    /// Projective `X` coordinate.
    pub x: Fp2<F>,
    /// Projective `Y` coordinate.
    pub y: Fp2<F>,
    /// Projective `Z` coordinate.
    pub z: Fp2<F>,
    _marker: PhantomData<F>,
}

impl<F: BaseField> JacobianPoint<F> {
    /// Construct a Jacobian point from projective `(X, Y, Z)` coordinates.
    #[inline]
    pub const fn new(x: Fp2<F>, y: Fp2<F>, z: Fp2<F>) -> Self {
        Self {
            x,
            y,
            z,
            _marker: PhantomData,
        }
    }

    /// Canonical point-at-infinity sentinel `(1, 1, 0)`.
    ///
    /// This encoding is chosen over `(0, 1, 0)` so later complete-addition
    /// formulas cannot transiently collide with the infinity sentinel while
    /// still keeping infinity detection as the simple predicate `Z == 0`.
    #[inline]
    pub fn infinity() -> Self {
        Self::new(Fp2::one(), Fp2::one(), Fp2::zero())
    }

    /// `Choice::TRUE` iff this point is at infinity, i.e. iff `Z == 0`.
    #[inline]
    pub fn is_infinity(&self) -> Choice {
        self.z.is_zero()
    }

    /// Negate this Jacobian point by sending `Y` to `-Y`.
    #[inline]
    pub fn negate(&self) -> Self {
        Self::new(self.x, self.y.negate(), self.z)
    }

    /// Convert to the affine-normalised representative `(X/Z², Y/Z³, 1)`.
    ///
    /// Infinity is returned unchanged as the canonical `(1, 1, 0)` sentinel.
    pub fn to_affine(&self) -> Self {
        let is_inf = self.is_infinity();
        let z_inv = self.z.invert().unwrap_or(Fp2::zero());
        let z_inv_sq = z_inv.square();
        let z_inv_cu = z_inv_sq.mul(&z_inv);
        let x_aff = self.x.mul(&z_inv_sq);
        let y_aff = self.y.mul(&z_inv_cu);
        let normal = Self::new(x_aff, y_aff, Fp2::one());
        Self::conditional_select(&normal, &Self::infinity(), is_inf)
    }

    /// Jacobian doubling using SQIsign spec Alg 8.11.
    ///
    /// The `a` parameter is the affine Montgomery curve coefficient
    /// `A` (the same `A` that appears in `y² = x³ + A x² + x`), **not** the
    /// reduced `(A + 2)/4` used by Montgomery `xDBL`. The spec writes this
    /// parameter as `A24` (matching the `(A24:C24)` projective coefficient
    /// notation used elsewhere in §8.2), but the Jacobian doubling formula
    /// reduces to `M = 3X² + 2·A·X·Z² + Z⁴` only when the input is the
    /// affine `A` itself — see derivation in §8.2 of the spec.
    ///
    /// The straight formula naturally maps the canonical infinity sentinel
    /// `(1, 1, 0)` back to itself, so no special-case branch is needed.
    pub fn double(&self, a: &Fp2<F>) -> Self {
        let mut t0 = self.x.square();
        let t1 = t0.double();
        t0 = t0.add(&t1);

        let mut t1 = self.z.square();
        let mut t2 = self.x.mul(a);
        t2 = t2.double();
        t2 = t1.add(&t2);
        t2 = t1.mul(&t2);
        t2 = t0.add(&t2);

        let mut z_out = self.y.mul(&self.z);
        z_out = z_out.double();

        t0 = z_out.square();
        t0 = t0.mul(a);

        t1 = self.y.square();
        t1 = t1.double();

        let mut t3 = self.x.double();
        t3 = t1.mul(&t3);

        let mut x_out = t2.square();
        x_out = x_out.sub(&t0);
        x_out = x_out.sub(&t3);
        x_out = x_out.sub(&t3);

        let mut y_out = t3.sub(&x_out);
        y_out = y_out.mul(&t2);

        t1 = t1.square();
        y_out = y_out.sub(&t1);
        y_out = y_out.sub(&t1);

        Self::new(x_out, y_out, z_out)
    }

    /// Return the SQIsign 2.0.1 Alg 8.13 `ADDComponents` triple `(u, v, w)`.
    ///
    /// Preconditions: `self` and `q` are distinct Jacobian points and do not
    /// represent affine negatives of each other. The output satisfies
    /// `x(P+Q) = (u - v) / w` and `x(P-Q) = (u + v) / w`, where `a` is the
    /// affine Montgomery coefficient `A` from `y² = x³ + A x² + x`.
    pub fn add_components(&self, q: &Self, a: &Fp2<F>) -> (Fp2<F>, Fp2<F>, Fp2<F>) {
        let z_p_sq = self.z.square();
        let z_q_sq = q.z.square();
        let t2 = self.x.mul(&z_q_sq);
        let t3 = z_p_sq.mul(&q.x);

        let mut t4 = self.y.mul(&q.z);
        t4 = t4.mul(&z_q_sq);

        let mut t5 = self.z.mul(&q.y);
        t5 = t5.mul(&z_p_sq);

        let t0 = z_p_sq.mul(&z_q_sq);
        let t6 = t4.mul(&t5);
        let t4_sq = t4.square();
        let t5_sq = t5.square();
        let t4 = t4_sq.add(&t5_sq);
        let t5 = t2.add(&t3);
        let t7 = t2.sub(&t3).square();

        let mut t1 = a.mul(&t0);
        t1 = t5.add(&t1);
        t1 = t1.mul(&t7);

        let u = t4.sub(&t1);
        let v = t6.double();
        // SPEC ERRATUM: SQIsign 2.0.1 Alg 8.13 step 23 writes `w ← t6 · t0`.
        // That is dimensionally inconsistent with the spec's own stated
        // output convention `x(P ± Q) = (u ∓ v) / w` together with Alg 8.12's
        // `Z(P + Q) = dx · Z_P · Z_Q`. The denominator must be `Z(P ± Q)^2 =
        // dx^2 · (Z_P · Z_Q)^2 = t7 · t0`.
        //
        // S130 cross-check against C reference (file
        // `src/ec/ref/lvlx/ec_jac.c`, function `jac_to_xz_add_components`,
        // lines 305-334): the C implementation computes the correct
        // algebraic formula by REUSING its `t6` variable across distinct
        // values:
        //   - L321 `fp2_mul(&t6, &t4, &t5)` → t6 = (z1z2)^3 * y1 * y2
        //     (the y-cross-product; consumed by L322 `v = 2·t6`)
        //   - L327 `fp2_add(&t6, &t3, &t3)` → t6 OVERWRITTEN to 2·z1^2·x2
        //   - L328 `fp2_sub(&t6, &t5, &t6)` → t6 OVERWRITTEN to dx = λ
        //   - L329 `fp2_sqr(&t6, &t6)` → t6 OVERWRITTEN to dx^2
        //   - L334 `fp2_mul(&add_comp->w, &t6, &t0)` → w = dx^2 · (z1z2)^2
        //
        // Our Rust uses TWO distinct variables instead of reusing: `t6`
        // at line 240 holds the y-cross-product (matching C's L321 value;
        // consumed by `v = t6.double()` at line 252, matching C's L322),
        // and `t7` at line 245 holds dx^2 (matching C's reused t6 at
        // L329). The final `w = t7 · t0` here matches C's L334
        // `add_comp->w = t6 · t0` algebraically — both compute
        // `dx^2 · (z1z2)^2`. KAT consistency is preserved: the C reference
        // (which generates KAT vectors) matches our fix.
        //
        // S130 differential-test verification (2026-05-22): temporarily
        // reverting this line to `t6 · t0` and re-running
        // `add_components_x_consistency_at_lvl1` produces a clean
        // assertion failure (algebraic mismatch on Fp2 elements, not a
        // panic or degenerate-input artifact) — confirming the fix is
        // load-bearing in our test surface and the test has genuine
        // discriminative power. Restored before final commit.
        //
        // Upstream spec-erratum filing at SQIsign/the-sqisign deferred as
        // a non-blocking follow-up (external-communication action requires
        // user approval per established policy).
        let w = t7.mul(&t0);

        (u, v, w)
    }

    /// Convert to Montgomery x-only projective coordinates `(X : Z²)`.
    #[inline]
    pub fn to_montgomery_xz(&self) -> crate::ec::montgomery::MontgomeryPoint<F> {
        crate::ec::montgomery::MontgomeryPoint::new(self.x, self.z.square())
    }

    /// Lift a Montgomery x-only point to an affine-normalised Jacobian point.
    ///
    /// For finite input `(X_M : Z_M)`, this computes `x = X_M / Z_M`,
    /// evaluates `y² = x³ + A x² + x`, and uses the principal branch returned
    /// by [`Fp2::sqrt`] to choose the sign of `Y`. This branch choice may
    /// differ from the C reference's `lift_basis` convention and should be
    /// audited when the higher-level basis code lands.
    ///
    /// If `Z_M == 0`, this succeeds and returns the Jacobian infinity
    /// sentinel `(1, 1, 0)`. If the affine curve equation is a non-square,
    /// this returns `CtOption::None`.
    pub fn from_montgomery_xz(
        p: &crate::ec::montgomery::MontgomeryPoint<F>,
        curve_a: &Fp2<F>,
    ) -> subtle::CtOption<Self> {
        let zero_point = Self::new(Fp2::zero(), Fp2::zero(), Fp2::zero());
        let is_inf = p.z.is_zero();
        let z_inv = p.z.invert().unwrap_or(Fp2::zero());
        let x_aff = p.x.mul(&z_inv);
        let x_sq = x_aff.square();
        let x_cu = x_sq.mul(&x_aff);
        let y_sq = x_cu.add(&curve_a.mul(&x_sq)).add(&x_aff);
        let y_opt = y_sq.sqrt();
        let y = y_opt.unwrap_or(Fp2::zero());
        let normal = Self::new(x_aff, y, Fp2::one());
        let success = Self::conditional_select(&normal, &Self::infinity(), is_inf);
        let is_some = is_inf | ((!is_inf) & y_opt.is_some());
        let point = Self::conditional_select(&zero_point, &success, is_some);
        subtle::CtOption::new(point, is_some)
    }

    /// `Choice::TRUE` iff `self` and `other` are projectively equivalent —
    /// i.e., they represent the same affine point under Jacobian equivalence
    /// `(X : Y : Z) ~ (λ²X : λ³Y : λZ)`. This is the semantic equality test
    /// and is what [`ConstantTimeEq`] delegates to.
    pub fn is_equivalent(&self, other: &Self) -> Choice {
        let self_z2 = self.z.square();
        let other_z2 = other.z.square();
        let self_z3 = self_z2.mul(&self.z);
        let other_z3 = other_z2.mul(&other.z);
        let x_lhs = self.x.mul(&other_z2);
        let x_rhs = other.x.mul(&self_z2);
        let y_lhs = self.y.mul(&other_z3);
        let y_rhs = other.y.mul(&self_z3);
        x_lhs.ct_eq(&x_rhs) & y_lhs.ct_eq(&y_rhs)
    }

    /// `Choice::TRUE` iff `self` and `other` share the same `(X, Y, Z)`
    /// triple componentwise. **This is NOT semantic equality** — two
    /// projectively-equivalent representations with different `λ` will
    /// return `FALSE`. Use only when the representation itself matters
    /// (debug printing, canonical-form checks, round-trip assertions).
    /// For semantic equality use [`Self::is_equivalent`] or [`ConstantTimeEq`].
    pub fn ct_eq_repr(&self, other: &Self) -> Choice {
        self.x.ct_eq(&other.x) & self.y.ct_eq(&other.y) & self.z.ct_eq(&other.z)
    }

    /// Rewrite `self` to a projectively-equivalent representative with
    /// freshly randomized `(X, Y, Z)` coordinates — in place.
    ///
    /// Samples a random non-zero `λ ∈ F_{p²}` from the supplied
    /// [`CryptoRng`] and overwrites `self` with `(λ²·X, λ³·Y, λ·Z)`.
    /// The affine point this represents is unchanged (the post-call
    /// `self.is_equivalent(&original) == Choice::TRUE`), but the
    /// bit-pattern of every coordinate is unpredictable to an
    /// attacker who does not see the rng state.
    ///
    /// # Why in-place is the primary API
    ///
    /// Blinding's security model is "the unblinded representation is
    /// gone after this call." The in-place mutation makes that the
    /// default, not an opt-in discipline the caller must remember.
    /// `JacobianPoint<F>` derives `Copy`, so a `&self → Self` return
    /// shape would silently leave the unblinded original on the stack
    /// for any subsequent code to accidentally reference. See
    /// [`Self::randomize_projective`] for the ergonomic consuming-self
    /// shim that does the same work.
    ///
    /// # Why this exists
    ///
    /// S131's dudect harness surfaced a cluster topology in
    /// `JacobianPoint::add` timing: degenerate inputs (`P = Q`,
    /// `P = -Q`, `Q = O`) execute measurably faster than non-degenerate
    /// inputs at the hardware level (cache locality and ALU
    /// zero-multiplication fast-paths). S132's source + assembly
    /// audit confirmed `add()` is structurally constant-time at both
    /// levels (7428 instructions, 0 jcc, 270 cmov) — the leak is
    /// microarchitectural, not software. Per S132 advisor:
    /// blinding inputs with projective randomization destroys the
    /// zero-correlation that produces the cluster, regardless of
    /// the underlying microarchitectural cause AND regardless of
    /// whether degenerate cases are unreachable on the
    /// secret-dependent path (a claim that historically broke
    /// curve25519 and pre-2011 OpenSSL).
    ///
    /// # Entropy budget
    ///
    /// Samples **64 bytes** (the L5 `ENCODED_BYTES` ceiling) from `rng`
    /// regardless of the parameter level. At L5 this provides 512 bits
    /// of unpredictability, matching the field size; at L1/L3 it
    /// over-supplies. Using L5's ceiling at all levels means a single
    /// fixed-size stack array works without `generic_const_exprs` and
    /// keeps the side-channel-mask-space ≥ the protocol target at every
    /// level. Cost overhead of the extra rng bytes is negligible against
    /// the field arithmetic that follows.
    ///
    /// # Cost
    ///
    /// 1 sample of 64 random bytes, 1 `hash_to_fp2` (rejection-
    /// sampling on the rare zero or hash-failure outcome), 1
    /// Fp2 square, 3 Fp2 multiplications. The hash-failure /
    /// zero-output probability is approximately `2^-256` for either
    /// branch, so the loop almost always terminates on the first
    /// iteration.
    ///
    /// # Infinity inputs
    ///
    /// Randomizing `JacobianPoint::infinity()` `(1, 1, 0)` produces
    /// `(λ², λ³, 0)` — still satisfies `is_infinity()` (which only
    /// checks `Z == 0`) but loses the canonical sentinel encoding.
    /// This is intentional and aligned with projective semantics:
    /// `(λ², λ³, 0)` is the same point at infinity. Callers that
    /// require the canonical sentinel should apply [`Self::to_affine`]
    /// before checking.
    pub fn randomize_in_place<R: CryptoRng>(&mut self, rng: &mut R) {
        let mut bytes = [0u8; 64];
        let lambda = loop {
            rng.fill_bytes(&mut bytes);
            let opt = crate::hash::hash_to_fp2::<F>(b"S133-blind", &bytes, 16);
            if bool::from(opt.is_some()) {
                let candidate = opt.unwrap_or(Fp2::<F>::zero());
                if !bool::from(candidate.is_zero()) {
                    break candidate;
                }
            }
            // Resample on hash-failure or zero. Both events have
            // probability ~2^-256 with a well-behaved CryptoRng;
            // this loop terminates on the first iteration in
            // practice.
        };
        let lambda_sq = lambda.square();
        let lambda_cu = lambda_sq.mul(&lambda);
        self.x = self.x.mul(&lambda_sq);
        self.y = self.y.mul(&lambda_cu);
        self.z = self.z.mul(&lambda);
    }

    /// Consuming-self ergonomic shim over [`Self::randomize_in_place`].
    ///
    /// Returns a projectively-equivalent representative with freshly
    /// randomized coordinates. Use the in-place variant directly when
    /// you want to enforce "the unblinded representation is gone" as
    /// a binding rewrite at the call site; use this shim when method
    /// chaining or expression-style ergonomics are preferred.
    pub fn randomize_projective<R: CryptoRng>(mut self, rng: &mut R) -> Self {
        self.randomize_in_place(rng);
        self
    }
}

impl<F: BaseField> ConstantTimeEq for JacobianPoint<F> {
    /// Projective equality — delegates to [`Self::is_equivalent`]. Two
    /// Jacobian triples that represent the same affine point return
    /// `Choice::TRUE` regardless of the specific `λ` scaling.
    fn ct_eq(&self, other: &Self) -> Choice {
        self.is_equivalent(other)
    }
}

impl<F: BaseField> ConditionallySelectable for JacobianPoint<F> {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self::new(
            Fp2::conditional_select(&a.x, &b.x, choice),
            Fp2::conditional_select(&a.y, &b.y, choice),
            Fp2::conditional_select(&a.z, &b.z, choice),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ec::montgomery::{MontgomeryCurve, MontgomeryPoint};
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

        // Try a short deterministic list of small affine x-coordinates first;
        // if they all land on the twist, fall back to hashed samples.
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
                let x_opt = hash_to_fp2::<F>(b"S126-jacobian-lift-x", &[i], 16);
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
            "S126: failed to find a deterministic liftable point on E_0",
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
                let x_opt = hash_to_fp2::<F>(b"S140-jacobian-lift-x", &[i], 16);
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
            "S140: failed to find a second deterministic distinct liftable point on E_0",
        );
        found.unwrap_or((MontgomeryPoint::infinity(), JacobianPoint::infinity()))
    }

    // S140: independent differential-oracle for add_components.
    //
    // Replaces the S127-era `add_components_x_consistency` test that
    // used the now-deleted `JacobianPoint::add` as its oracle. The new
    // oracle is the Montgomery curve's **multiplicative differential
    // identity** (Costello-Smith 2017, "Montgomery curves and their
    // arithmetic", Section 2 / Equation 8). For Montgomery
    // `y² = x³ + A x² + x` with distinct non-inverse points P, Q:
    //
    //     x(P+Q) · x(P-Q) · (x_P − x_Q)²  =  (x_P · x_Q − 1)²
    //
    // The right-hand side is **A-independent** — A appears only in the
    // additive form `x(P+Q) + x(P-Q)`, not the multiplicative form.
    //
    // Combined with the spec Alg 8.13 promise
    // `(u−v)/w = x(P+Q)` and `(u+v)/w = x(P-Q)`, we get
    // `(u² − v²) / w² = (x_P · x_Q − 1)² / (x_P − x_Q)²`, which after
    // clearing denominators yields the testable identity:
    //
    //     (u² − v²) · (x_P − x_Q)²  ==  (x_P · x_Q − 1)² · w²
    //
    // This depends ONLY on x-coordinates + w; it does not use `add()`
    // (which no longer exists per S139) and is genuinely independent
    // of `add_components`'s function body — the LHS / RHS construction
    // comes from the algorithm's *contract* (spec promise) composed
    // with curve algebra (Montgomery diff_add identity), not from the
    // function's internal variable choices.
    fn check_add_components_differential_identity<F: BaseField>() {
        let curve = MontgomeryCurve::<F>::e0();
        let (_, p) = first_liftable_point_on_e0::<F>();
        let (_, q) = second_liftable_point_on_e0::<F>();
        assert!(
            !bool::from(p.is_equivalent(&q)),
            "S140: differential-oracle test must use distinct points",
        );
        let p_neg = p.negate();
        assert!(
            !bool::from(p_neg.is_equivalent(&q)),
            "S140: differential-oracle test must use non-inverse points (Q != -P)",
        );

        let (u, v, w) = p.add_components(&q, &curve.a);

        let p_aff = p.to_affine();
        let q_aff = q.to_affine();

        // x_P - x_Q must be nonzero (we just asserted P != Q affinely
        // and P != -Q means x_P != x_Q on Montgomery curves where
        // distinct affine points share x only if they are negatives).
        let dx = p_aff.x.sub(&q_aff.x);

        // LHS: (u² − v²) · (x_P − x_Q)²
        let lhs = u.square().sub(&v.square()).mul(&dx.square());

        // RHS: (x_P · x_Q − 1)² · w²
        let xpxq_minus_one = p_aff.x.mul(&q_aff.x).sub(&Fp2::<F>::one());
        let rhs = xpxq_minus_one.square().mul(&w.square());

        assert_eq!(
            lhs, rhs,
            "S140: Montgomery diff_add identity (u² − v²)(x_P − x_Q)² = (x_P x_Q − 1)² w² must hold for add_components output",
        );
    }

    #[test]
    fn add_components_differential_identity_at_lvl1() {
        check_add_components_differential_identity::<Fp1Element>();
    }

    #[test]
    fn add_components_differential_identity_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_add_components_differential_identity::<Fp3Element>();
    }

    #[test]
    fn add_components_differential_identity_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_add_components_differential_identity::<Fp5Element>();
    }

    fn check_infinity_is_z_zero<F: BaseField>() {
        let inf = JacobianPoint::<F>::infinity();
        assert!(
            bool::from(inf.is_infinity()),
            "S126: infinity().is_infinity() must be Choice::TRUE",
        );
        assert_eq!(
            inf.x,
            Fp2::<F>::one(),
            "S126: infinity x sentinel must be 1"
        );
        assert_eq!(
            inf.y,
            Fp2::<F>::one(),
            "S126: infinity y sentinel must be 1"
        );
        assert_eq!(
            inf.z,
            Fp2::<F>::zero(),
            "S126: infinity z sentinel must be 0"
        );

        let mont_inf = MontgomeryPoint::<F>::infinity();
        let lift = JacobianPoint::from_montgomery_xz(&mont_inf, &MontgomeryCurve::<F>::e0().a);
        assert!(
            bool::from(lift.is_some()),
            "S126: lifting Montgomery infinity must succeed",
        );
        let lifted = lift.unwrap_or(JacobianPoint::new(Fp2::zero(), Fp2::zero(), Fp2::zero()));
        assert!(
            bool::from(lifted.is_infinity()),
            "S126: lifting Montgomery infinity must return Jacobian infinity",
        );
    }

    #[test]
    fn infinity_is_z_zero_at_lvl1() {
        check_infinity_is_z_zero::<Fp1Element>();
    }

    #[test]
    fn infinity_is_z_zero_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_infinity_is_z_zero::<Fp3Element>();
    }

    #[test]
    fn infinity_is_z_zero_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_infinity_is_z_zero::<Fp5Element>();
    }

    fn check_negate_negate_is_identity<F: BaseField>() {
        let p = JacobianPoint::<F>::new(small_fp2::<F>(2), small_fp2::<F>(3), small_fp2::<F>(5));
        assert_eq!(
            p.negate().negate(),
            p,
            "S126: negate(negate(P)) must equal P pointwise",
        );

        let q = JacobianPoint::<F>::new(small_fp2::<F>(7), small_fp2::<F>(11), small_fp2::<F>(13));
        let pick_p = JacobianPoint::<F>::conditional_select(&p, &q, Choice::from(0));
        let pick_q = JacobianPoint::<F>::conditional_select(&p, &q, Choice::from(1));
        assert_eq!(
            pick_p, p,
            "S126: conditional_select(_, _, FALSE) must return the first point componentwise",
        );
        assert_eq!(
            pick_q, q,
            "S126: conditional_select(_, _, TRUE) must return the second point componentwise",
        );
    }

    #[test]
    fn negate_negate_is_identity_at_lvl1() {
        check_negate_negate_is_identity::<Fp1Element>();
    }

    #[test]
    fn negate_negate_is_identity_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_negate_negate_is_identity::<Fp3Element>();
    }

    #[test]
    fn negate_negate_is_identity_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_negate_negate_is_identity::<Fp5Element>();
    }

    fn check_double_of_infinity_is_infinity<F: BaseField>() {
        let inf = JacobianPoint::<F>::infinity();
        // `a` is the affine Montgomery coefficient (spec Alg 8.11 input).
        let doubled = inf.double(&small_fp2::<F>(7));
        assert!(
            bool::from(doubled.is_infinity()),
            "S126: doubling infinity must keep Z = 0",
        );
        assert_eq!(
            doubled,
            JacobianPoint::<F>::infinity(),
            "S126: doubling the canonical infinity sentinel must round-trip exactly",
        );
    }

    #[test]
    fn double_of_infinity_is_infinity_at_lvl1() {
        check_double_of_infinity_is_infinity::<Fp1Element>();
    }

    #[test]
    fn double_of_infinity_is_infinity_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_double_of_infinity_is_infinity::<Fp3Element>();
    }

    #[test]
    fn double_of_infinity_is_infinity_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_double_of_infinity_is_infinity::<Fp5Element>();
    }

    // S129 advisor-prioritized correctness test (the only correctness-risk
    // item among the three S127-deferred items): Jacobian doubling at a
    // 2-torsion point must produce infinity. On E_0 the affine equation
    // `y² = x³ + x = x · (x² + 1)` puts the full 2-torsion subgroup
    // E_0[2] = {O, (0, 0), (i, 0), (-i, 0)} in F_{p²} where i² = -1. The
    // doubling-formula edge case at Y=0 produces a degenerate intermediate
    // in naïve transcriptions; the spec Alg 8.11 formula correctly handles
    // it via Z_{2P} = 2·Y_P·Z_P = 0 → infinity sentinel canonicalization.
    fn check_double_of_2_torsion_is_infinity<F: BaseField>() {
        let a = Fp2::<F>::zero(); // E_0's affine coefficient A = 0
        let one = Fp2::<F>::one();
        let zero = Fp2::<F>::zero();
        let imag = Fp2::<F>::new(F::zero(), F::one()); // the Fp2 imaginary unit i

        // (0, 0, 1) is always 2-torsion on E_0: y² = x·(x²+1) is zero at x=0.
        let origin = JacobianPoint::<F>::new(zero, zero, one);
        let doubled_origin = origin.double(&a);
        assert!(
            bool::from(doubled_origin.is_infinity()),
            "S129: double of (0, 0, 1) on E_0 must produce infinity",
        );

        // (i, 0, 1) is 2-torsion on E_0: x² + 1 = i² + 1 = 0.
        let pos_i = JacobianPoint::<F>::new(imag, zero, one);
        let doubled_pos_i = pos_i.double(&a);
        assert!(
            bool::from(doubled_pos_i.is_infinity()),
            "S129: double of (i, 0, 1) on E_0 must produce infinity",
        );

        // (-i, 0, 1) is 2-torsion on E_0 (symmetric case).
        let neg_i = JacobianPoint::<F>::new(imag.negate(), zero, one);
        let doubled_neg_i = neg_i.double(&a);
        assert!(
            bool::from(doubled_neg_i.is_infinity()),
            "S129: double of (-i, 0, 1) on E_0 must produce infinity",
        );
    }

    #[test]
    fn double_of_2_torsion_is_infinity_at_lvl1() {
        check_double_of_2_torsion_is_infinity::<Fp1Element>();
    }

    #[test]
    fn double_of_2_torsion_is_infinity_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_double_of_2_torsion_is_infinity::<Fp3Element>();
    }

    #[test]
    fn double_of_2_torsion_is_infinity_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_double_of_2_torsion_is_infinity::<Fp5Element>();
    }

    fn check_double_consistency_with_montgomery_xdbl<F: BaseField>() {
        let curve = MontgomeryCurve::<F>::e0();
        // Jacobian::double takes the affine `A`; Montgomery::x_double takes
        // the reduced `a24 = (A + 2)/4`. They are the same curve, expressed
        // through different parameter conventions.
        let a24 = curve.a24();
        let (p_mont, p_jac) = first_liftable_point_on_e0::<F>();
        let doubled_jac = p_jac.double(&curve.a).to_montgomery_xz();
        let doubled_mont = p_mont.x_double(&a24);
        assert!(
            bool::from(doubled_jac.ct_eq(&doubled_mont)),
            "S126: Jacobian double must agree projectively with Montgomery xDBL after x-only conversion",
        );
    }

    #[test]
    fn double_consistency_with_montgomery_xdbl_at_lvl1() {
        check_double_consistency_with_montgomery_xdbl::<Fp1Element>();
    }

    #[test]
    fn double_consistency_with_montgomery_xdbl_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_double_consistency_with_montgomery_xdbl::<Fp3Element>();
    }

    #[test]
    fn double_consistency_with_montgomery_xdbl_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_double_consistency_with_montgomery_xdbl::<Fp5Element>();
    }

    #[test]
    fn from_montgomery_xz_y_squared() {
        let curve = MontgomeryCurve::<Fp1Element>::e0();
        let (_, point) = first_liftable_point_on_e0::<Fp1Element>();
        assert_eq!(
            point.z,
            Fp2::<Fp1Element>::one(),
            "S126: lifted Jacobian point must be affine-normalised with Z = 1",
        );

        let x_sq = point.x.square();
        let lhs = point.y.square();
        let rhs = x_sq.mul(&point.x).add(&curve.a.mul(&x_sq)).add(&point.x);
        assert_eq!(
            lhs, rhs,
            "S126: lifted point must satisfy Y^2 = X^3 + A X^2 + X when Z = 1",
        );
    }

    // S126 advisor recommendation (b): the from_montgomery_xz sign convention
    // is the principal branch returned by Fp2::sqrt. Pin the convention with
    // a round-trip test: lifting a fresh point and re-projecting must yield
    // a Montgomery point whose x equals the input, and lifting twice must
    // be idempotent (no random sign flips between calls). If Fp::sqrt ever
    // changes its branch convention, this test breaks loudly — protecting
    // callers from a silent semantic shift.
    #[test]
    fn from_montgomery_xz_sign_is_deterministic_at_lvl1() {
        let curve = MontgomeryCurve::<Fp1Element>::e0();
        let (mont_in, jac_first) = first_liftable_point_on_e0::<Fp1Element>();
        // Lift again — must produce the same Y (same sqrt branch).
        let jac_again = JacobianPoint::from_montgomery_xz(&mont_in, &curve.a)
            .unwrap_or(JacobianPoint::<Fp1Element>::infinity());
        assert!(
            bool::from(jac_first.ct_eq_repr(&jac_again)),
            "S126: from_montgomery_xz must return the same sign on repeated calls",
        );
        // Round-trip: lift, then to_montgomery_xz — must equal the input
        // projectively (the x-only side is sign-invariant).
        let mont_back = jac_first.to_montgomery_xz();
        assert!(
            bool::from(mont_back.ct_eq(&mont_in)),
            "S126: lift then to_montgomery_xz must round-trip to the input x-coord",
        );
    }

    fn check_to_affine_idempotent<F: BaseField>() {
        let p = JacobianPoint::<F>::new(small_fp2::<F>(2), small_fp2::<F>(3), small_fp2::<F>(5));
        let once = p.to_affine();
        let twice = once.to_affine();
        assert_eq!(twice, once, "S126: to_affine must be idempotent");

        let lambda = small_fp2::<F>(2);
        let lambda_sq = lambda.square();
        let lambda_cu = lambda_sq.mul(&lambda);
        let scaled =
            JacobianPoint::<F>::new(p.x.mul(&lambda_sq), p.y.mul(&lambda_cu), p.z.mul(&lambda));
        // ct_eq is SEMANTIC (projective) equality: differently-scaled
        // representatives of the same affine point must compare equal.
        assert!(
            bool::from(<JacobianPoint<F> as ConstantTimeEq>::ct_eq(&p, &scaled)),
            "S126: ct_eq must be projective and accept differently-scaled representatives",
        );
        assert!(
            bool::from(p.is_equivalent(&scaled)),
            "S126: is_equivalent must accept projectively-scaled representatives",
        );
        // ct_eq_repr is REPRESENTATION equality (pointwise): differently-scaled
        // representatives must compare not-equal.
        assert!(
            !bool::from(p.ct_eq_repr(&scaled)),
            "S126: ct_eq_repr must be pointwise and reject differently-scaled representatives",
        );
    }

    #[test]
    fn to_affine_idempotent_at_lvl1() {
        check_to_affine_idempotent::<Fp1Element>();
    }

    #[test]
    fn to_affine_idempotent_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_to_affine_idempotent::<Fp3Element>();
    }

    #[test]
    fn to_affine_idempotent_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_to_affine_idempotent::<Fp5Element>();
    }

    // S133 projective coordinate randomization tests. Use a deterministic
    // ChaCha20 RNG so tests are reproducible; production callers will use
    // a CryptoRng backed by /dev/urandom or equivalent.
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn fresh_rng(seed_byte: u8) -> ChaCha20Rng {
        ChaCha20Rng::from_seed([seed_byte; 32])
    }

    fn check_randomize_preserves_is_equivalent<F: BaseField>() {
        let (_, p) = first_liftable_point_on_e0::<F>();
        let mut rng = fresh_rng(0x33);
        let randomized = p.randomize_projective(&mut rng);
        assert!(
            bool::from(randomized.is_equivalent(&p)),
            "S133: randomize_projective must preserve projective equivalence",
        );
    }

    #[test]
    fn randomize_preserves_is_equivalent_at_lvl1() {
        check_randomize_preserves_is_equivalent::<Fp1Element>();
    }

    #[test]
    fn randomize_preserves_is_equivalent_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_randomize_preserves_is_equivalent::<Fp3Element>();
    }

    #[test]
    fn randomize_preserves_is_equivalent_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_randomize_preserves_is_equivalent::<Fp5Element>();
    }

    // S133: blinding must DESTROY the canonical bit-pattern — the whole
    // point is that an attacker cannot predict (X, Y, Z) bits even given
    // the affine point. ct_eq_repr (pointwise) returning FALSE after
    // randomization is the property we want; if it returned TRUE the
    // randomization sampled λ = 1 (probability ~1/p² ≈ 2^-500), which
    // is the rejection-sample lower-bound failure we accept.
    fn check_randomize_destroys_canonical_repr<F: BaseField>() {
        let (_, p) = first_liftable_point_on_e0::<F>();
        let mut rng = fresh_rng(0x44);
        let randomized = p.randomize_projective(&mut rng);
        assert!(
            !bool::from(randomized.ct_eq_repr(&p)),
            "S133: randomize_projective must produce a different bit pattern \
             (probability of λ = 1 coincidence is ~2^-500 with a CryptoRng)",
        );
    }

    #[test]
    fn randomize_destroys_canonical_repr_at_lvl1() {
        check_randomize_destroys_canonical_repr::<Fp1Element>();
    }

    #[test]
    fn randomize_destroys_canonical_repr_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_randomize_destroys_canonical_repr::<Fp3Element>();
    }

    #[test]
    fn randomize_destroys_canonical_repr_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_randomize_destroys_canonical_repr::<Fp5Element>();
    }

    fn check_infinity_randomize_stays_infinity<F: BaseField>() {
        let mut rng = fresh_rng(0x66);
        let randomized = JacobianPoint::<F>::infinity().randomize_projective(&mut rng);
        assert!(
            bool::from(randomized.is_infinity()),
            "S133: blinding infinity must keep Z = 0 (still at infinity)",
        );
    }

    #[test]
    fn infinity_randomize_stays_infinity_at_lvl1() {
        check_infinity_randomize_stays_infinity::<Fp1Element>();
    }

    #[test]
    fn infinity_randomize_stays_infinity_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_infinity_randomize_stays_infinity::<Fp3Element>();
    }

    #[test]
    fn infinity_randomize_stays_infinity_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_infinity_randomize_stays_infinity::<Fp5Element>();
    }
}
