// SPDX-License-Identifier: MIT OR Apache-2.0
//! Concrete quaternion-algebra arithmetic and the `O_0` maximal-order shape.
//!
//! Conventions and multiplication table are documented in the parent module.

use core::marker::PhantomData;

use crypto_bigint::{Int, Uint};

use crate::params::Params;

/// Element `a + b·i + c·j + d·k` of `B_{p,∞}` with integer coefficients.
///
/// `LIMBS` is the limb-count of the underlying `Int<LIMBS>` — pick wide
/// enough that the products `a·e`, `p · c · g`, etc. don't overflow during
/// the intermediate computations.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Quaternion<const LIMBS: usize> {
    /// Coefficient of `1`.
    pub a: Int<LIMBS>,
    /// Coefficient of `i`.
    pub b: Int<LIMBS>,
    /// Coefficient of `j`.
    pub c: Int<LIMBS>,
    /// Coefficient of `k = i · j`.
    pub d: Int<LIMBS>,
}

impl<const LIMBS: usize> Quaternion<LIMBS> {
    /// Construct an element from its four integer components.
    #[inline]
    pub const fn new(a: Int<LIMBS>, b: Int<LIMBS>, c: Int<LIMBS>, d: Int<LIMBS>) -> Self {
        Self { a, b, c, d }
    }

    /// The zero element `0 + 0·i + 0·j + 0·k`.
    #[inline]
    pub fn zero() -> Self {
        let z = Int::<LIMBS>::from_i64(0);
        Self::new(z, z, z, z)
    }

    /// The unit element `1`.
    #[inline]
    pub fn one() -> Self {
        let z = Int::<LIMBS>::from_i64(0);
        Self::new(Int::<LIMBS>::from_i64(1), z, z, z)
    }

    /// The basis element `i`.
    #[inline]
    pub fn i() -> Self {
        let z = Int::<LIMBS>::from_i64(0);
        Self::new(z, Int::<LIMBS>::from_i64(1), z, z)
    }

    /// The basis element `j`.
    #[inline]
    pub fn j() -> Self {
        let z = Int::<LIMBS>::from_i64(0);
        Self::new(z, z, Int::<LIMBS>::from_i64(1), z)
    }

    /// The basis element `k = i · j`.
    #[inline]
    pub fn k() -> Self {
        let z = Int::<LIMBS>::from_i64(0);
        Self::new(z, z, z, Int::<LIMBS>::from_i64(1))
    }

    /// `Choice`-free equality test (the underlying integer comparison
    /// is variable-time, but quaternion coefficients are public-data in
    /// every SQIsign path this module touches).
    pub fn equals(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b && self.c == other.c && self.d == other.d
    }

    /// `self + rhs` (wrapping at `LIMBS` words — pick `LIMBS` wide enough).
    pub fn add(&self, rhs: &Self) -> Self {
        Self::new(
            self.a.wrapping_add(&rhs.a),
            self.b.wrapping_add(&rhs.b),
            self.c.wrapping_add(&rhs.c),
            self.d.wrapping_add(&rhs.d),
        )
    }

    /// `self − rhs`.
    pub fn sub(&self, rhs: &Self) -> Self {
        Self::new(
            self.a.wrapping_sub(&rhs.a),
            self.b.wrapping_sub(&rhs.b),
            self.c.wrapping_sub(&rhs.c),
            self.d.wrapping_sub(&rhs.d),
        )
    }

    /// `−self`.
    pub fn negate(&self) -> Self {
        Self::new(
            self.a.wrapping_neg(),
            self.b.wrapping_neg(),
            self.c.wrapping_neg(),
            self.d.wrapping_neg(),
        )
    }

    /// Quaternion conjugate `a − b·i − c·j − d·k`.
    pub fn conjugate(&self) -> Self {
        Self::new(
            self.a,
            self.b.wrapping_neg(),
            self.c.wrapping_neg(),
            self.d.wrapping_neg(),
        )
    }

    /// Reduced trace `Tr(q) = q + q̄ = 2 a`.
    pub fn trace(&self) -> Int<LIMBS> {
        self.a.wrapping_add(&self.a)
    }

    /// `self · scalar` — scalar multiplication by a small integer.
    pub fn scale(&self, k: i64) -> Self {
        let k_int = Int::<LIMBS>::from_i64(k);
        Self::new(
            self.a.wrapping_mul(&k_int),
            self.b.wrapping_mul(&k_int),
            self.c.wrapping_mul(&k_int),
            self.d.wrapping_mul(&k_int),
        )
    }

    /// Quaternion multiplication `self × rhs` using the relations
    /// `i² = −1`, `j² = −p`, `k = i j = −j i`.
    ///
    /// `p` is the level's prime, encoded as an unsigned `Uint<LIMBS>`.
    /// (Internally cast to `Int<LIMBS>` once.)
    ///
    /// **Precondition** (caller's responsibility): same as [`Self::norm`]
    /// — `p`'s top bit must be zero (`p.bits_vartime() < 64·LIMBS`).
    /// If `p`'s top bit is set, `*p.as_int()` is interpreted as a
    /// negative `Int<LIMBS>` and every product involving `p_int` flips
    /// sign. Structurally satisfied at all production SQIsign LIMBS
    /// values; debug_assert defends future callers.
    pub fn mul(&self, rhs: &Self, p: &Uint<LIMBS>) -> Self {
        debug_assert!(
            p.bits_vartime()
                < 64u32 * u32::try_from(LIMBS).expect("LIMBS fits u32 at all SQIsign levels"),
            "Quaternion::mul: p's top bit must be zero (p.bits_vartime() < 64·LIMBS); reinterpretation as Int<LIMBS> via p.as_int() would otherwise yield a negative value and sign-flip every p·c·d product",
        );
        let p_int = *p.as_int();
        // 1 = a·e − b·f − p · (c·g + d·h)
        let ae = self.a.wrapping_mul(&rhs.a);
        let bf = self.b.wrapping_mul(&rhs.b);
        let cg = self.c.wrapping_mul(&rhs.c);
        let dh = self.d.wrapping_mul(&rhs.d);
        let p_cgdh = p_int.wrapping_mul(&cg.wrapping_add(&dh));
        let out_a = ae.wrapping_sub(&bf).wrapping_sub(&p_cgdh);
        // i = a·f + b·e + p · (c·h − d·g)   (jk = p·i, kj = −p·i)
        let af = self.a.wrapping_mul(&rhs.b);
        let be = self.b.wrapping_mul(&rhs.a);
        let ch = self.c.wrapping_mul(&rhs.d);
        let dg = self.d.wrapping_mul(&rhs.c);
        let p_chdg = p_int.wrapping_mul(&ch.wrapping_sub(&dg));
        let out_b = af.wrapping_add(&be).wrapping_add(&p_chdg);
        // j = a·g + c·e + (d·f − b·h)        (ik = −j, ki = +j)
        let ag = self.a.wrapping_mul(&rhs.c);
        let ce = self.c.wrapping_mul(&rhs.a);
        let bh = self.b.wrapping_mul(&rhs.d);
        let df = self.d.wrapping_mul(&rhs.b);
        let out_c = ag.wrapping_add(&ce).wrapping_add(&df.wrapping_sub(&bh));
        // k = a·h + d·e + (b·g − c·f)
        let ah = self.a.wrapping_mul(&rhs.d);
        let de = self.d.wrapping_mul(&rhs.a);
        let bg = self.b.wrapping_mul(&rhs.c);
        let cf = self.c.wrapping_mul(&rhs.b);
        let out_d = ah.wrapping_add(&de).wrapping_add(&bg.wrapping_sub(&cf));
        Self::new(out_a, out_b, out_c, out_d)
    }

    /// Reduced norm `N(q) = a² + b² + p (c² + d²)`.
    ///
    /// **Precondition** (caller's responsibility): `p`'s top bit must be
    /// zero — i.e. `p.bits_vartime() < 64·LIMBS`. The body reinterprets
    /// `p` as `Int<LIMBS>` via `*p.as_int()`; if `p`'s top bit is set,
    /// the reinterpretation is negative and the computed reduced norm
    /// is sign-flipped. At all production SQIsign LIMBS values (L1
    /// LIMBS≥8 with p≈2^248; L3 LIMBS≥12 with p≈2^383; L5 LIMBS≥16
    /// with p≈2^505) the bound holds structurally; the debug_assert
    /// catches misuse at debug-build / test time.
    pub fn norm(&self, p: &Uint<LIMBS>) -> Int<LIMBS> {
        debug_assert!(
            p.bits_vartime()
                < 64u32 * u32::try_from(LIMBS).expect("LIMBS fits u32 at all SQIsign levels"),
            "Quaternion::norm: p's top bit must be zero (p.bits_vartime() < 64·LIMBS); reinterpretation as Int<LIMBS> via p.as_int() would otherwise yield a negative value and a sign-flipped reduced norm",
        );
        let p_int = *p.as_int();
        let a_sq = self.a.wrapping_mul(&self.a);
        let b_sq = self.b.wrapping_mul(&self.b);
        let c_sq = self.c.wrapping_mul(&self.c);
        let d_sq = self.d.wrapping_mul(&self.d);
        let cd = c_sq.wrapping_add(&d_sq);
        let p_cd = p_int.wrapping_mul(&cd);
        a_sq.wrapping_add(&b_sq).wrapping_add(&p_cd)
    }
}

/// A rational element of `B_{p,∞}` — a [`Quaternion`] over a positive
/// integer denominator.
///
/// Mirrors the SQIsign C reference's `quat_alg_elem_t` (`coord[4]` + `denom`)
/// at `src/quaternion/ref/generic/quaternion.h`. The denominator is
/// tracked separately so that operations like `quat_alg_mul` and
/// `quat_alg_normalize` can keep the rational representation in
/// reduced form.
///
/// The `Clapotis` evaluator's `find_uv` output (`beta1`, `beta2`)
/// is a `RationalQuaternion` because the LLL-reduction step + ideal
/// rescaling introduce denominators ≠ 1.
///
/// Invariant: `denom > 0`. The numerator's coefficients may be
/// negative; the denominator is always positive. The
/// [`Self::normalize`] method divides both numerator and
/// denominator by their common GCD to put the rational in
/// lowest terms.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct RationalQuaternion<const LIMBS: usize> {
    /// Numerator — a quaternion with integer coefficients (possibly negative).
    pub num: Quaternion<LIMBS>,
    /// Denominator — a positive `Uint<LIMBS>` (rational arithmetic
    /// convention: never zero, never signed).
    pub denom: Uint<LIMBS>,
}

impl<const LIMBS: usize> RationalQuaternion<LIMBS> {
    /// Construct from a numerator and a strictly positive denominator.
    /// **Panics** in debug builds if `denom == 0` (the rational is
    /// undefined). In release the zero-denom construction proceeds
    /// silently; downstream arithmetic will produce garbage.
    #[inline]
    pub fn new(num: Quaternion<LIMBS>, denom: Uint<LIMBS>) -> Self {
        debug_assert!(
            denom != Uint::<LIMBS>::from_u64(0),
            "RationalQuaternion::new: denom must be > 0",
        );
        Self { num, denom }
    }

    /// The rational integer-one element `1/1`.
    #[inline]
    pub fn one() -> Self {
        Self {
            num: Quaternion::<LIMBS>::one(),
            denom: Uint::<LIMBS>::ONE,
        }
    }

    /// Equality test on the rational representation. Two
    /// `RationalQuaternion` values are considered equal iff their
    /// stored numerators and denominators agree exactly (NOT via
    /// cross-multiplication). For cross-multiplication-based
    /// equivalence callers should call [`Self::normalize`] on both
    /// sides first.
    #[inline]
    pub fn equals_raw(&self, other: &Self) -> bool {
        self.num.equals(&other.num) && self.denom == other.denom
    }

    /// Rational-quaternion multiplication:
    /// `(a / d_a) · (b / d_b) = (a · b) / (d_a · d_b)`.
    ///
    /// Wraps [`Quaternion::mul`] for the numerator and an unsigned
    /// `wrapping_mul` for the denominator. **Does NOT normalize** —
    /// callers that need lowest-terms output must call
    /// [`Self::normalize`] explicitly. This matches the SQIsign C
    /// reference's `quat_alg_mul` (separate `quat_alg_normalize`).
    ///
    /// `p` is the level's prime. Same precondition as
    /// [`Quaternion::mul`]: `p.bits_vartime() < 64 · LIMBS`.
    pub fn mul(&self, rhs: &Self, p: &Uint<LIMBS>) -> Self {
        Self {
            num: self.num.mul(&rhs.num, p),
            denom: self.denom.wrapping_mul(&rhs.denom),
        }
    }

    /// Rational-quaternion conjugate: numerator conjugated, denominator
    /// unchanged. `conjugate(a / d) = conjugate(a) / d` because
    /// `conjugate` is a `Q`-linear involution and the denominator is
    /// a scalar.
    ///
    /// Wraps [`Quaternion::conjugate`].
    #[inline]
    pub fn conjugate(&self) -> Self {
        Self {
            num: self.num.conjugate(),
            denom: self.denom,
        }
    }

    /// Reduce the rational to lowest terms: divide numerator and
    /// denominator by `g = gcd(|num.a|, |num.b|, |num.c|, |num.d|,
    /// denom)`. Returns `self` unchanged when `g ≤ 1` (already
    /// reduced) or when the numerator is the zero quaternion (no
    /// reduction is meaningful).
    ///
    /// Mirrors the SQIsign C reference's `quat_alg_normalize`. The
    /// post-condition: `gcd(|num.a|, |num.b|, |num.c|, |num.d|,
    /// denom) == 1` (or the numerator is zero).
    ///
    /// # Variable-time
    ///
    /// Calls [`uint_gcd_vartime`](crate::quaternion::represent_integer)
    /// four times (Euclidean reduction). Variable-time on the
    /// magnitudes — acceptable per SQIsign 2.0 §8.
    pub fn normalize(&self) -> Self {
        use crate::quaternion::hnf::int_div_floor;
        use crate::quaternion::represent_integer::uint_gcd_vartime;

        let a_abs = self.num.a.abs();
        let b_abs = self.num.b.abs();
        let c_abs = self.num.c.abs();
        let d_abs = self.num.d.abs();

        // gcd over all numerator magnitudes + denominator.
        let g = uint_gcd_vartime::<LIMBS>(&a_abs, &b_abs);
        let g = uint_gcd_vartime::<LIMBS>(&g, &c_abs);
        let g = uint_gcd_vartime::<LIMBS>(&g, &d_abs);
        let g = uint_gcd_vartime::<LIMBS>(&g, &self.denom);

        // Already in lowest terms (or numerator is zero — gcd(0, denom)
        // = denom, but dividing both sides by denom collapses the
        // rational to 0 / 1; semantically a no-op for downstream
        // callers, so skip the trip through int_div_floor).
        if g <= Uint::<LIMBS>::ONE {
            return *self;
        }
        if a_abs == Uint::<LIMBS>::from_u64(0)
            && b_abs == Uint::<LIMBS>::from_u64(0)
            && c_abs == Uint::<LIMBS>::from_u64(0)
            && d_abs == Uint::<LIMBS>::from_u64(0)
        {
            return *self;
        }

        // Reinterpret g as a non-negative Int for the signed division.
        let g_int = *g.as_int();
        let new_a = int_div_floor::<LIMBS>(&self.num.a, &g_int);
        let new_b = int_div_floor::<LIMBS>(&self.num.b, &g_int);
        let new_c = int_div_floor::<LIMBS>(&self.num.c, &g_int);
        let new_d = int_div_floor::<LIMBS>(&self.num.d, &g_int);
        // Unsigned division for the denominator.
        let denom_nz = crypto_bigint::NonZero::new(g)
            .expect("uint_gcd_vartime returns 0 only if both inputs are 0; guarded above");
        let new_denom = self.denom.wrapping_div_vartime(&denom_nz);

        Self {
            num: Quaternion::<LIMBS>::new(new_a, new_b, new_c, new_d),
            denom: new_denom,
        }
    }
}

/// Marker for the quaternion algebra `B_{p,∞}` at the given parameter set.
/// Carries no data; provides level-aware helpers via the `Params` trait.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct QuaternionAlgebra<P: Params> {
    _marker: PhantomData<P>,
}

impl<P: Params> QuaternionAlgebra<P> {
    /// Construct a fresh marker.
    #[inline]
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

/// Symbolic basis vectors of the special maximal order `O_0` for `p ≡ 3 mod 4`.
///
/// `O_0 = ⟨ 1, i, (i + j) / 2, (1 + k) / 2 ⟩` over `Z`. Every element of
/// `O_0` is a `Z`-linear combination of these four. Storage as integer
/// coefficients in this basis is the cheap-and-cheerful representation
/// downstream KLPT code prefers — a 4×4 basis-change matrix into the
/// standard `(1, i, j, k)` basis lets us go to `Quaternion` when needed.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum OrderBasis {
    /// `1`
    One,
    /// `i`
    I,
    /// `(i + j) / 2`
    IPlusJOver2,
    /// `(1 + k) / 2`
    OnePlusKOver2,
}

/// Operations against the special maximal order `O_0` at the given level.
/// Concrete element representation will land alongside the KLPT session.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct MaximalOrder<P: Params> {
    _marker: PhantomData<P>,
}

impl<P: Params> MaximalOrder<P> {
    /// Construct the maximal-order marker for level `P`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// The four basis vectors of `O_0`, in canonical order.
    pub const fn basis() -> [OrderBasis; 4] {
        [
            OrderBasis::One,
            OrderBasis::I,
            OrderBasis::IPlusJOver2,
            OrderBasis::OnePlusKOver2,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── RationalQuaternion unit tests (S194) ───────────────────────────

    #[test]
    fn rational_quaternion_one_is_unit_over_unit_denom() {
        let r = RationalQuaternion::<8>::one();
        assert_eq!(r.num, Quaternion::<8>::one());
        assert_eq!(r.denom, Uint::<8>::ONE);
    }

    #[test]
    fn rational_quaternion_equals_raw_compares_components() {
        let a = RationalQuaternion::<8>::new(Quaternion::<8>::i(), Uint::<8>::from_u64(3));
        let b = RationalQuaternion::<8>::new(Quaternion::<8>::i(), Uint::<8>::from_u64(3));
        let c = RationalQuaternion::<8>::new(Quaternion::<8>::j(), Uint::<8>::from_u64(3));
        let d = RationalQuaternion::<8>::new(Quaternion::<8>::i(), Uint::<8>::from_u64(5));
        assert!(a.equals_raw(&b));
        assert!(!a.equals_raw(&c));
        assert!(!a.equals_raw(&d));
    }

    // ── RationalQuaternion arithmetic (S197) ───────────────────────────

    #[test]
    fn rational_quaternion_mul_identity_preserves_operand() {
        let p = Uint::<8>::from_u64(7);
        let x = RationalQuaternion::<8>::new(Quaternion::<8>::i(), Uint::<8>::from_u64(5));
        let one = RationalQuaternion::<8>::one();
        // 1 · x = x (numerator), denom is 1 · 5 = 5.
        let product = one.mul(&x, &p);
        assert!(product.num.equals(&x.num));
        assert_eq!(product.denom, x.denom);
    }

    #[test]
    fn rational_quaternion_mul_doubles_denom_and_uses_quaternion_mul() {
        let p = Uint::<8>::from_u64(7);
        // (i / 2) · (j / 3) = (i·j) / 6 = k / 6 (at p ≡ 3 mod 4 with k = i·j).
        let a = RationalQuaternion::<8>::new(Quaternion::<8>::i(), Uint::<8>::from_u64(2));
        let b = RationalQuaternion::<8>::new(Quaternion::<8>::j(), Uint::<8>::from_u64(3));
        let product = a.mul(&b, &p);
        assert!(product.num.equals(&Quaternion::<8>::k()));
        assert_eq!(product.denom, Uint::<8>::from_u64(6));
    }

    #[test]
    fn rational_quaternion_conjugate_negates_b_c_d_keeps_denom() {
        // (1 + i + j + k) / 7 → (1 - i - j - k) / 7.
        let num = Quaternion::<8>::new(
            Int::<8>::from_i64(1),
            Int::<8>::from_i64(1),
            Int::<8>::from_i64(1),
            Int::<8>::from_i64(1),
        );
        let r = RationalQuaternion::<8>::new(num, Uint::<8>::from_u64(7));
        let conj = r.conjugate();
        assert_eq!(conj.num.a, Int::<8>::from_i64(1));
        assert_eq!(conj.num.b, Int::<8>::from_i64(-1));
        assert_eq!(conj.num.c, Int::<8>::from_i64(-1));
        assert_eq!(conj.num.d, Int::<8>::from_i64(-1));
        assert_eq!(conj.denom, Uint::<8>::from_u64(7));
        // Involution: conj(conj(x)) == x.
        let twice = conj.conjugate();
        assert!(twice.equals_raw(&r));
    }

    #[test]
    fn rational_quaternion_normalize_divides_by_common_gcd() {
        // (2 + 2i + 0j + 0k) / 4 → (1 + i) / 2.
        let num = Quaternion::<8>::new(
            Int::<8>::from_i64(2),
            Int::<8>::from_i64(2),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        );
        let r = RationalQuaternion::<8>::new(num, Uint::<8>::from_u64(4));
        let n = r.normalize();
        assert_eq!(n.num.a, Int::<8>::from_i64(1));
        assert_eq!(n.num.b, Int::<8>::from_i64(1));
        assert_eq!(n.num.c, Int::<8>::from_i64(0));
        assert_eq!(n.num.d, Int::<8>::from_i64(0));
        assert_eq!(n.denom, Uint::<8>::from_u64(2));
    }

    #[test]
    fn rational_quaternion_normalize_idempotent_on_lowest_terms() {
        // (1 + i) / 2 already in lowest terms — no change.
        let num = Quaternion::<8>::new(
            Int::<8>::from_i64(1),
            Int::<8>::from_i64(1),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        );
        let r = RationalQuaternion::<8>::new(num, Uint::<8>::from_u64(2));
        let n = r.normalize();
        assert!(
            n.equals_raw(&r),
            "normalize must be a no-op on lowest-terms input"
        );
    }

    #[test]
    fn rational_quaternion_normalize_handles_negative_numerator_components() {
        // (-6 + 0i + 0j + 3k) / 9 → (-2 + 0i + 0j + 1k) / 3.
        // gcd(|-6|, 0, 0, |3|, 9) = gcd(6, 3, 9) = 3.
        let num = Quaternion::<8>::new(
            Int::<8>::from_i64(-6),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(3),
        );
        let r = RationalQuaternion::<8>::new(num, Uint::<8>::from_u64(9));
        let n = r.normalize();
        assert_eq!(n.num.a, Int::<8>::from_i64(-2));
        assert_eq!(n.num.b, Int::<8>::from_i64(0));
        assert_eq!(n.num.c, Int::<8>::from_i64(0));
        assert_eq!(n.num.d, Int::<8>::from_i64(1));
        assert_eq!(n.denom, Uint::<8>::from_u64(3));
    }

    #[test]
    fn rational_quaternion_normalize_preserves_zero_numerator() {
        // 0 / 5 stays 0 / 5 (no algebraic meaning to reducing further).
        let zero = Quaternion::<8>::zero();
        let r = RationalQuaternion::<8>::new(zero, Uint::<8>::from_u64(5));
        let n = r.normalize();
        assert!(
            n.equals_raw(&r),
            "normalize must preserve zero-numerator rationals as-is"
        );
    }

    /// LIMBS = 8 (512-bit signed) — plenty for Level-1's 251-bit prime plus
    /// the small integer multipliers used in these algebraic-identity tests.
    type Q = Quaternion<8>;
    type U = Uint<8>;

    /// A small prime-sized Uint for testing the `j² = −p` and norm identities
    /// without needing to compute against the real SQIsign primes.
    fn small_p() -> U {
        // Use a small "fake" prime for arithmetic-identity tests. The
        // structural identities (i² = −1, anti-commutativity, etc.) don't
        // depend on the value of p; norm and j² do.
        U::from_u64(7)
    }

    #[test]
    fn i_squared_is_minus_one() {
        let p = small_p();
        let i_sq = Q::i().mul(&Q::i(), &p);
        let minus_one = Q::one().negate();
        assert!(i_sq.equals(&minus_one));
    }

    #[test]
    fn j_squared_is_minus_p() {
        let p = small_p();
        let j_sq = Q::j().mul(&Q::j(), &p);
        // j² = -p · 1
        let minus_p = Q::new(
            p.as_int().wrapping_neg(),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        );
        assert!(j_sq.equals(&minus_p));
    }

    #[test]
    fn k_equals_ij() {
        let p = small_p();
        let ij = Q::i().mul(&Q::j(), &p);
        assert!(ij.equals(&Q::k()));
    }

    #[test]
    fn ji_equals_minus_k() {
        let p = small_p();
        let ji = Q::j().mul(&Q::i(), &p);
        let minus_k = Q::k().negate();
        assert!(ji.equals(&minus_k));
    }

    #[test]
    fn k_squared_is_minus_p() {
        let p = small_p();
        let k_sq = Q::k().mul(&Q::k(), &p);
        let minus_p = Q::new(
            p.as_int().wrapping_neg(),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        );
        assert!(k_sq.equals(&minus_p));
    }

    #[test]
    fn conjugate_is_involution() {
        let p = small_p();
        let q = Q::new(
            Int::<8>::from_i64(3),
            Int::<8>::from_i64(-5),
            Int::<8>::from_i64(7),
            Int::<8>::from_i64(-2),
        );
        assert!(q.conjugate().conjugate().equals(&q));
        let _ = p;
    }

    #[test]
    fn norm_equals_q_times_conj_a_component() {
        let p = small_p();
        let q = Q::new(
            Int::<8>::from_i64(3),
            Int::<8>::from_i64(-5),
            Int::<8>::from_i64(2),
            Int::<8>::from_i64(1),
        );
        // q · q̄ should be a pure scalar equal to N(q).
        let q_qbar = q.mul(&q.conjugate(), &p);
        let n = q.norm(&p);
        assert_eq!(q_qbar.a, n);
        assert_eq!(q_qbar.b, Int::<8>::from_i64(0));
        assert_eq!(q_qbar.c, Int::<8>::from_i64(0));
        assert_eq!(q_qbar.d, Int::<8>::from_i64(0));
    }

    #[test]
    fn norm_of_one_is_one() {
        let p = small_p();
        let n = Q::one().norm(&p);
        assert_eq!(n, Int::<8>::from_i64(1));
    }

    #[test]
    fn norm_of_i_is_one() {
        let p = small_p();
        let n = Q::i().norm(&p);
        assert_eq!(n, Int::<8>::from_i64(1));
    }

    #[test]
    fn norm_of_j_is_p() {
        let p = small_p();
        let n = Q::j().norm(&p);
        assert_eq!(n, *p.as_int());
    }

    #[test]
    fn trace_is_2a() {
        let q = Q::new(
            Int::<8>::from_i64(11),
            Int::<8>::from_i64(-3),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        );
        assert_eq!(q.trace(), Int::<8>::from_i64(22));
    }

    #[test]
    fn addition_is_commutative() {
        let p = small_p();
        let a = Q::new(
            Int::<8>::from_i64(1),
            Int::<8>::from_i64(2),
            Int::<8>::from_i64(3),
            Int::<8>::from_i64(4),
        );
        let b = Q::new(
            Int::<8>::from_i64(5),
            Int::<8>::from_i64(6),
            Int::<8>::from_i64(7),
            Int::<8>::from_i64(8),
        );
        let ab = a.add(&b);
        let ba = b.add(&a);
        assert!(ab.equals(&ba));
        let _ = p;
    }

    #[test]
    fn multiplication_is_not_commutative() {
        let p = small_p();
        let ij = Q::i().mul(&Q::j(), &p);
        let ji = Q::j().mul(&Q::i(), &p);
        assert!(!ij.equals(&ji));
    }

    #[test]
    fn order_basis_canonical_count() {
        use crate::params::Level1;
        let basis = MaximalOrder::<Level1>::basis();
        assert_eq!(basis.len(), 4);
        assert_eq!(basis[0], OrderBasis::One);
        assert_eq!(basis[3], OrderBasis::OnePlusKOver2);
    }

    #[test]
    fn scale_by_two_doubles() {
        let p = small_p();
        let q = Q::new(
            Int::<8>::from_i64(1),
            Int::<8>::from_i64(2),
            Int::<8>::from_i64(3),
            Int::<8>::from_i64(4),
        );
        let doubled = q.scale(2);
        let summed = q.add(&q);
        assert!(doubled.equals(&summed));
        let _ = p;
    }

    /// The Level-1 base prime `p = 5·2^248 − 1` widened to `Uint<8>` for
    /// quaternion arithmetic. Use this in tests that need to verify a
    /// `p`-dependent identity (`j² = −p`, `k² = −p`) at real-prime scale.
    fn level1_p() -> U {
        crate::params::lvl1::prime().resize::<8>()
    }

    #[test]
    fn i_squared_is_minus_one_at_real_lvl1_prime() {
        // i² = −1 is structural — independent of p. This test pins that the
        // wiring (real prime through `mul`) doesn't perturb p-independent
        // identities.
        let p = level1_p();
        let i_sq = Q::i().mul(&Q::i(), &p);
        let minus_one = Q::one().negate();
        assert!(i_sq.equals(&minus_one));
    }

    #[test]
    fn j_squared_is_minus_real_lvl1_prime() {
        // j² = −p. The first test where the *value* of p enters the result.
        let p = level1_p();
        let j_sq = Q::j().mul(&Q::j(), &p);
        let minus_p = Q::new(
            p.as_int().wrapping_neg(),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        );
        assert!(j_sq.equals(&minus_p));
    }

    #[test]
    fn k_equals_ij_at_real_lvl1_prime() {
        let p = level1_p();
        let ij = Q::i().mul(&Q::j(), &p);
        assert!(ij.equals(&Q::k()));
    }

    #[test]
    fn ji_equals_minus_k_at_real_lvl1_prime() {
        let p = level1_p();
        let ji = Q::j().mul(&Q::i(), &p);
        let minus_k = Q::k().negate();
        assert!(ji.equals(&minus_k));
    }

    #[test]
    fn k_squared_is_minus_real_lvl1_prime() {
        let p = level1_p();
        let k_sq = Q::k().mul(&Q::k(), &p);
        let minus_p = Q::new(
            p.as_int().wrapping_neg(),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        );
        assert!(k_sq.equals(&minus_p));
    }

    /// The Level-3 base prime `p = 65·2^376 − 1` widened to `Uint<8>`.
    fn level3_p() -> U {
        crate::params::lvl3::prime().resize::<8>()
    }

    #[test]
    fn i_squared_is_minus_one_at_real_lvl3_prime() {
        let p = level3_p();
        let i_sq = Q::i().mul(&Q::i(), &p);
        let minus_one = Q::one().negate();
        assert!(i_sq.equals(&minus_one));
    }

    #[test]
    fn j_squared_is_minus_real_lvl3_prime() {
        let p = level3_p();
        let j_sq = Q::j().mul(&Q::j(), &p);
        let minus_p = Q::new(
            p.as_int().wrapping_neg(),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        );
        assert!(j_sq.equals(&minus_p));
    }

    #[test]
    fn k_equals_ij_at_real_lvl3_prime() {
        let p = level3_p();
        let ij = Q::i().mul(&Q::j(), &p);
        assert!(ij.equals(&Q::k()));
    }

    #[test]
    fn ji_equals_minus_k_at_real_lvl3_prime() {
        let p = level3_p();
        let ji = Q::j().mul(&Q::i(), &p);
        let minus_k = Q::k().negate();
        assert!(ji.equals(&minus_k));
    }

    #[test]
    fn k_squared_is_minus_real_lvl3_prime() {
        let p = level3_p();
        let k_sq = Q::k().mul(&Q::k(), &p);
        let minus_p = Q::new(
            p.as_int().wrapping_neg(),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        );
        assert!(k_sq.equals(&minus_p));
    }

    /// The Level-5 base prime `p = 27·2^500 − 1` as `Uint<8>` (its native
    /// width — `resize::<8>()` is a structural no-op here, kept for
    /// symmetry with the L1/L3 call shape).
    fn level5_p() -> U {
        crate::params::lvl5::prime().resize::<8>()
    }

    #[test]
    fn i_squared_is_minus_one_at_real_lvl5_prime() {
        let p = level5_p();
        let i_sq = Q::i().mul(&Q::i(), &p);
        let minus_one = Q::one().negate();
        assert!(i_sq.equals(&minus_one));
    }

    #[test]
    fn j_squared_is_minus_real_lvl5_prime() {
        // Magnitude stress test: `p ~ 2^505`, `−p` lands ~6 bits below
        // `Int<8>::MIN` — the `Int<8>` sign-room check in
        // `lvl5::prime_top_bit_clear_for_int8_sign_room` proves this fits.
        let p = level5_p();
        let j_sq = Q::j().mul(&Q::j(), &p);
        let minus_p = Q::new(
            p.as_int().wrapping_neg(),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        );
        assert!(j_sq.equals(&minus_p));
    }

    #[test]
    fn k_equals_ij_at_real_lvl5_prime() {
        let p = level5_p();
        let ij = Q::i().mul(&Q::j(), &p);
        assert!(ij.equals(&Q::k()));
    }

    #[test]
    fn ji_equals_minus_k_at_real_lvl5_prime() {
        let p = level5_p();
        let ji = Q::j().mul(&Q::i(), &p);
        let minus_k = Q::k().negate();
        assert!(ji.equals(&minus_k));
    }

    #[test]
    fn k_squared_is_minus_real_lvl5_prime() {
        let p = level5_p();
        let k_sq = Q::k().mul(&Q::k(), &p);
        let minus_p = Q::new(
            p.as_int().wrapping_neg(),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        );
        assert!(k_sq.equals(&minus_p));
    }
}
