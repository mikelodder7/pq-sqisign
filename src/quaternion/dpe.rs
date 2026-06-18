// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Dpe` — double-plus-exponent floating point, a byte-exact pure-Rust port
//! of the DPE library (`dpe.h`, Pelissier/Zimmermann, v1.7) as vendored and
//! used by the SQIsign C reference's lattice reducer
//! (`src/quaternion/ref/generic/lll/l2.c` `quat_lll_core`).
//!
//! # Why this exists
//!
//! Matching the official SQIsign keygen KAT requires byte-exact
//! reproduction of the C reference's LLL reduction. `quat_lll_core` runs
//! its Gram-Schmidt / Lovász numerics in `dpe_t` (a `double` significand
//! plus a separate integer exponent), keeping only the basis transform in
//! exact integers. LLL output is non-canonical — it depends on the exact
//! sequence of float comparisons and swaps — so the secret key produced by
//! keygen depends on these float operations bit-for-bit. IEEE-754 `f64` is
//! deterministic, so a faithful transliteration of the DPE arithmetic
//! (same operations, same order) reproduces the C result exactly while
//! staying 100% pure Rust.
//!
//! # Representation
//!
//! Value = `mant · 2^exp`, with `mant` a `double` and `exp` an `int`
//! (`DPE_EXP_T = int`). After `Dpe::normalize`, `mant ∈ [1/2, 1)` for
//! nonzero finite values; zero is `mant = 0.0, exp = i32::MIN`
//! (`DPE_EXPMIN`). The C build active on x86_64 (`DPE_USE_DOUBLE`,
//! `DPE_BITSIZE = 53`) is the configuration ported here.
//!
//! # Scope (this brick)
//!
//! The float core: normalize, setters/getters, scale, add/sub/mul/div,
//! compare, round. The bignum bridge (`set_z`/`get_z`, i.e. the
//! `mpz_get_d_2exp` round-trip that feeds the Gram matrix into the GSO) and
//! the `quat_lll_core` loop itself land alongside the LLL port that
//! consumes them.

use crypto_bigint::{Int, Uint};

/// `DPE_BITSIZE` for the `DPE_USE_DOUBLE` build: the `f64` significand bits.
const BITSIZE: i32 = 53;

/// `2^s` as an exact `f64`, for the finite range used here.
/// Constructing the IEEE-754 exponent directly keeps this available in
/// `no_std` builds, where libm-backed float methods are not available.
#[inline]
fn pow2(s: i32) -> f64 {
    debug_assert!((-1074..=1023).contains(&s));
    if s >= -1022 {
        f64::from_bits(u64::try_from(s + 1023).expect("biased exponent is non-negative") << 52)
    } else {
        f64::from_bits(1u64 << u32::try_from(s + 1074).expect("subnormal shift is non-negative"))
    }
}

/// Round to nearest integer with ties away from zero, matching C `round()`.
///
/// Callers only pass values whose rounded magnitude is below `2^53`, so the
/// cast to `i64` is exact after the half-unit adjustment.
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn round_ties_away(x: f64) -> f64 {
    let adjusted = if x >= 0.0 { x + 0.5 } else { x - 0.5 };
    (adjusted as i64) as f64
}

/// `frexp`: split a finite nonzero `x` into `(m, e)` with `x = m · 2^e` and
/// `m ∈ [1/2, 1)`. Matches the C `dpe_normalize` split (both the portable
/// `frexp` path and the active x86_64 bit-twiddling path, which are
/// bit-equivalent for finite nonzero values).
fn frexp(x: f64) -> (f64, i32) {
    debug_assert!(x != 0.0 && x.is_finite());
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // Subnormal: scale up by 2^54, recurse, adjust exponent back.
        let (m, e) = frexp(x * pow2(54));
        return (m, e - 54);
    }
    // Force the biased exponent field to 1022 (0x3FE) so the significand
    // lands in [1/2, 1); the removed exponent (biased − 1022) is returned.
    let e = biased - 1022;
    let new_bits = (bits & 0x800f_ffff_ffff_ffff) | 0x3fe0_0000_0000_0000;
    (f64::from_bits(new_bits), e)
}

/// `ldexp`: `m · 2^e`, exact for finite results in the exponent range used
/// by the reducer's round/get paths. Splits large `|e|` so the intermediate
/// power of two never overflows before the (exact) power-of-two multiply.
fn ldexp(mant: f64, exp: i32) -> f64 {
    if mant == 0.0 || !mant.is_finite() {
        return mant;
    }
    let mut m = mant;
    let mut e = exp;
    while e > 1023 {
        m *= pow2(1023);
        e -= 1023;
    }
    while e < -1022 {
        m *= pow2(-1022);
        e += 1022;
    }
    m * pow2(e)
}

/// Double-plus-exponent float: value `mant · 2^exp`.
#[derive(Debug, Clone, Copy)]
pub struct Dpe {
    mant: f64,
    exp: i32,
}

impl Dpe {
    /// `dpe_set_d`: from an `f64` (exp 0 then normalize).
    pub fn from_f64(y: f64) -> Self {
        let mut x = Dpe { mant: y, exp: 0 };
        x.normalize();
        x
    }

    /// `dpe_set_ui`/`dpe_set_si`: from an integer (exp 0 then normalize).
    /// The significand may lose precision for `|y| >= 2^53` exactly as the C
    /// `(DPE_DOUBLE) y` cast does; the reducer only feeds small constants here.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_i64(y: i64) -> Self {
        let mut x = Dpe {
            mant: y as f64,
            exp: 0,
        };
        x.normalize();
        x
    }

    /// `dpe_get_d`: `ldexp(mant, exp)`.
    pub fn to_f64(&self) -> f64 {
        ldexp(self.mant, self.exp)
    }

    /// `dpe_set_z`: from an exact integer, matching GMP `mpz_get_d_2exp`.
    /// Returns `(d, e)` with `d` the significand `∈ [1/2, 1)` (sign carried)
    /// and `e` the bit length, where `d · 2^e` equals `y` **truncated toward
    /// zero to 53 significant bits**. Zero maps to `mant = 0.0, exp = 0`
    /// (GMP's convention; `set_z` does not normalize, so zero is NOT given
    /// `i32::MIN` here — matching the C). This is the single highest
    /// byte-exactness risk in the dpe port: GMP truncates, it does not round.
    #[allow(clippy::cast_precision_loss)] // t < 2^53 ⇒ exact in f64.
    pub fn from_int<const N: usize>(y: &Int<N>) -> Self {
        let abs: Uint<N> = y.abs();
        let b = abs.bits_vartime();
        if b == 0 {
            return Dpe { mant: 0.0, exp: 0 };
        }
        // Top 53 bits of |y|, truncated toward zero. For b >= 53 shift the
        // low (b-53) bits out; for b < 53 left-pad so bit 52 is the MSB.
        // Either way the result is in [2^52, 2^53), i.e. fits the low word.
        let t: u64 = if b >= 53 {
            abs.shr_vartime(b - 53).as_words()[0]
        } else {
            abs.shl_vartime(53 - b).as_words()[0]
        };
        let mant = (t as f64) * pow2(-BITSIZE);
        Dpe {
            mant: if bool::from(y.is_negative()) {
                -mant
            } else {
                mant
            },
            exp: i32::try_from(b).expect("bit length fits i32"),
        }
    }

    /// `dpe_get_z`: to the nearest integer (ties away from zero), matching
    /// the C `dpe_get_z`. `|y| < 1/2` → 0; `exp >= 53` is already integral
    /// (`mantissa · 2^53` then shift); otherwise `round(ldexp(mant, exp))`.
    #[allow(clippy::cast_possible_truncation)] // |d|,|r| < 2^53 ⇒ exact in i64.
    pub fn to_int<const N: usize>(&self) -> Int<N> {
        if self.exp < 0 {
            return Int::<N>::from_i64(0);
        }
        if self.exp >= BITSIZE {
            // Integer: the (≤53-bit) significand times 2^(exp).
            let d = self.mant * pow2(BITSIZE); // |d| < 2^53, integer-valued
            let m = d as i64;
            let shift = u32::try_from(self.exp - BITSIZE).expect("exp-53 fits u32");
            let mag = Uint::<N>::from_u64(m.unsigned_abs()).shl_vartime(shift);
            let i = *mag.as_int();
            if m < 0 { i.wrapping_neg() } else { i }
        } else {
            // 0 <= exp < 53: round to nearest integer, ties away from zero.
            let r = round_ties_away(ldexp(self.mant, self.exp));
            Int::<N>::from_i64(r as i64)
        }
    }

    /// `DPE_SIGN`: −1, 0, or +1 by significand sign.
    pub fn sign(&self) -> i32 {
        if self.mant < 0.0 {
            -1
        } else if self.mant > 0.0 {
            1
        } else {
            0
        }
    }

    /// `dpe_zero_p`.
    pub fn is_zero(&self) -> bool {
        self.mant == 0.0
    }

    /// `dpe_neg`.
    pub fn neg(&self) -> Self {
        Dpe {
            mant: -self.mant,
            exp: self.exp,
        }
    }

    /// `dpe_abs`.
    pub fn abs(&self) -> Self {
        Dpe {
            mant: if self.mant >= 0.0 {
                self.mant
            } else {
                -self.mant
            },
            exp: self.exp,
        }
    }

    /// `dpe_normalize`: significand into `[1/2, 1)`; zero → exp `i32::MIN`;
    /// NaN/Inf keep their exponent.
    fn normalize(&mut self) {
        if self.mant == 0.0 || !self.mant.is_finite() {
            if self.mant == 0.0 {
                self.exp = i32::MIN;
            }
            // NaN/Inf: leave exp unchanged.
        } else {
            let (m, e) = frexp(self.mant);
            self.mant = m;
            self.exp += e;
        }
    }

    /// `dpe_add`: `self + z`, both assumed normalized; result normalized.
    pub fn add(&self, z: &Dpe) -> Self {
        if self.exp > z.exp + BITSIZE {
            return *self;
        }
        if z.exp > self.exp + BITSIZE {
            return *z;
        }
        let d = self.exp - z.exp; // |d| <= BITSIZE
        let mut x = if d >= 0 {
            Dpe {
                mant: self.mant + z.mant * pow2(-d),
                exp: self.exp,
            }
        } else {
            Dpe {
                mant: z.mant + self.mant * pow2(d),
                exp: z.exp,
            }
        };
        x.normalize();
        x
    }

    /// `dpe_sub`: `self − z`, both assumed normalized; result normalized.
    pub fn sub(&self, z: &Dpe) -> Self {
        if self.exp > z.exp + BITSIZE {
            return *self;
        }
        if z.exp > self.exp + BITSIZE {
            return z.neg();
        }
        let d = self.exp - z.exp;
        let mut x = if d >= 0 {
            Dpe {
                mant: self.mant - z.mant * pow2(-d),
                exp: self.exp,
            }
        } else {
            Dpe {
                mant: self.mant * pow2(d) - z.mant,
                exp: z.exp,
            }
        };
        x.normalize();
        x
    }

    /// `dpe_mul`. The exponent add is `wrapping` to match the C, where a
    /// zero operand carries `exp = i32::MIN` (`DPE_EXPMIN`) and the integer
    /// sum overflows; the mantissa is then `0.0`, so `normalize` resets the
    /// exponent and the wrapped intermediate is discarded. For nonzero
    /// operands the exponents are in range and the add does not wrap.
    pub fn mul(&self, z: &Dpe) -> Self {
        let mut x = Dpe {
            mant: self.mant * z.mant,
            exp: self.exp.wrapping_add(z.exp),
        };
        x.normalize();
        x
    }

    /// `dpe_div` (`z != 0`). Exponent subtraction is `wrapping` for the same
    /// reason as [`Dpe::mul`] (a zero numerator carries `exp = i32::MIN`).
    pub fn div(&self, z: &Dpe) -> Self {
        let mut x = Dpe {
            mant: self.mant / z.mant,
            exp: self.exp.wrapping_sub(z.exp),
        };
        x.normalize();
        x
    }

    /// `dpe_cmp`: −1/0/+1. Valid because both operands are normalized
    /// (compare by sign, then exponent, then significand). Named `compare`
    /// (not `cmp`) so it is not mistaken for `Ord::cmp`, whose total-order
    /// contract this does not satisfy (NaN, distinct ±0 conventions).
    pub fn compare(&self, y: &Dpe) -> i32 {
        let sx = self.sign();
        let d = sx - y.sign();
        if d != 0 {
            d.signum()
        } else if self.exp > y.exp {
            if sx > 0 { 1 } else { -1 }
        } else if y.exp > self.exp {
            if sx > 0 { -1 } else { 1 }
        } else if self.mant < y.mant {
            -1
        } else {
            i32::from(self.mant > y.mant)
        }
    }

    /// `dpe_cmp_d`: compare against an `f64` (converted via `set_d`).
    pub fn cmp_f64(&self, d: f64) -> i32 {
        self.compare(&Dpe::from_f64(d))
    }

    /// `dpe_round`: nearest integer (still a `Dpe`); ties away from zero
    /// (matching C `round()`). `|y| < 1/2` → 0; `exp >= 53` already integral.
    pub fn round(&self) -> Self {
        if self.exp < 0 {
            Dpe::from_i64(0)
        } else if self.exp >= BITSIZE {
            *self
        } else {
            let d = ldexp(self.mant, self.exp);
            Dpe::from_f64(round_ties_away(d))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Significand normalized into [1/2, 1) for nonzero finite values.
    #[test]
    fn normalize_significand_in_half_one() {
        for &v in &[0.995_f64, 1.0, 2.0, 1024.0, 0.001, -7.5, 1e30, -1e-30] {
            let x = Dpe::from_f64(v);
            let m = x.mant.abs();
            assert!(
                (0.5..1.0).contains(&m),
                "mant {m} for v={v} must be in [1/2, 1)",
            );
            assert_eq!(
                x.to_f64(),
                v,
                "round-trip set_d/get_d must be exact for {v}"
            );
        }
    }

    /// Zero gets the minimum exponent (DPE_EXPMIN = INT_MIN).
    #[test]
    fn zero_has_expmin() {
        let z = Dpe::from_f64(0.0);
        assert_eq!(z.exp, i32::MIN);
        assert!(z.is_zero());
        assert_eq!(z.sign(), 0);
        assert_eq!(z.to_f64(), 0.0);
    }

    /// Arithmetic reproduces f64 semantics exactly for representable inputs
    /// (dpe is exact when the true result fits the 53-bit significand).
    #[test]
    fn arithmetic_matches_f64_for_representable() {
        let cases = [
            (3.0, 8.0),
            (24.0, 6.0),
            (1.5, 0.25),
            (-7.0, 2.0),
            (100.0, 7.0),
        ];
        for &(a, b) in &cases {
            let da = Dpe::from_f64(a);
            let db = Dpe::from_f64(b);
            assert_eq!(da.add(&db).to_f64(), a + b, "add {a}+{b}");
            assert_eq!(da.sub(&db).to_f64(), a - b, "sub {a}-{b}");
            assert_eq!(da.mul(&db).to_f64(), a * b, "mul {a}*{b}");
            assert_eq!(da.div(&db).to_f64(), a / b, "div {a}/{b}");
        }
    }

    /// Exponent tracking survives values far outside f64's range — the whole
    /// point of dpe (mant·2^exp with a separate integer exponent).
    #[test]
    fn exponent_tracks_beyond_f64_range() {
        // Build 2^4000 by squaring, then divide back; ratio must be exact.
        let two = Dpe::from_f64(2.0);
        let mut big = two;
        for _ in 0..12 {
            big = big.mul(&big); // 2^(2^12) = 2^4096
        }
        // big = 2^4096, well beyond f64::MAX (~2^1024); exp must record it.
        assert_eq!(big.exp, 4097); // mant=0.5, exp=4097 ⇒ 0.5·2^4097 = 2^4096
        assert!((0.5..1.0).contains(&big.mant.abs()));
        let ratio = big.div(&big);
        assert_eq!(ratio.to_f64(), 1.0, "x/x must be exactly 1");
        let half = big.div(&two);
        assert_eq!(half.exp, 4096, "2^4096 / 2 = 2^4095 ⇒ 0.5·2^4096");
    }

    /// Compare: sign, then exponent, then significand.
    #[test]
    fn compare_orders_correctly() {
        let a = Dpe::from_f64(3.0);
        let b = Dpe::from_f64(8.0);
        let na = a.neg();
        assert_eq!(a.compare(&b), -1);
        assert_eq!(b.compare(&a), 1);
        assert_eq!(a.compare(&a), 0);
        assert_eq!(na.compare(&a), -1, "negative < positive");
        assert_eq!(a.compare(&na), 1);
        assert_eq!(a.cmp_f64(2.999), 1);
        assert_eq!(a.cmp_f64(3.0), 0);
        // Across exponents.
        let big = Dpe::from_f64(1e20);
        assert_eq!(a.compare(&big), -1);
        assert_eq!(big.neg().compare(&na), -1, "−1e20 < −3");
    }

    /// Round: nearest integer, ties away from zero (C `round()` semantics).
    #[test]
    fn round_ties_away_from_zero() {
        let cases = [
            (2.5, 3.0),
            (-2.5, -3.0),
            (2.4, 2.0),
            (2.6, 3.0),
            (-0.4, 0.0),
            (0.49, 0.0),
            (123.0, 123.0),
        ];
        for &(v, want) in &cases {
            assert_eq!(Dpe::from_f64(v).round().to_f64(), want, "round({v})");
        }
    }

    /// Add/sub across large exponent gaps hit the negligible-operand
    /// shortcut (|smaller| < 1 ulp of larger) — result is the larger operand
    /// exactly, since the smaller cannot perturb the 53-bit significand.
    #[test]
    fn add_negligible_operand_shortcut() {
        // 2^100 by squaring (2^4, then ^... just multiply explicitly).
        let mut big = Dpe::from_f64(2.0);
        for _ in 0..99 {
            big = big.mul(&Dpe::from_f64(2.0)); // 2^100
        }
        let one = Dpe::from_f64(1.0);
        // exp gap is 100 > BITSIZE(53): 1.0 is negligible against 2^100.
        let sum = big.add(&one);
        assert_eq!(
            sum.mant, big.mant,
            "negligible add must return the larger significand"
        );
        assert_eq!(
            sum.exp, big.exp,
            "negligible add must return the larger exponent"
        );
        // Symmetric: small + big returns big.
        let sum2 = one.add(&big);
        assert_eq!(sum2.compare(&big), 0);
        // sub: big − negligible returns big; negligible − big returns −big.
        assert_eq!(big.sub(&one).compare(&big), 0);
        assert_eq!(one.sub(&big).compare(&big.neg()), 0);
    }

    /// `from_int` reproduces GMP `mpz_get_d_2exp`: significand in [1/2, 1),
    /// exponent = bit length, truncated toward zero to 53 bits.
    #[test]
    fn from_int_matches_mpz_get_d_2exp() {
        // 5 = 0.625 · 2^3.
        let d = Dpe::from_int::<4>(&Int::<4>::from_i64(5));
        assert_eq!((d.mant, d.exp), (0.625, 3));
        assert_eq!(d.to_f64(), 5.0);

        // −6 = −0.75 · 2^3.
        let d = Dpe::from_int::<4>(&Int::<4>::from_i64(-6));
        assert_eq!((d.mant, d.exp), (-0.75, 3));
        assert_eq!(d.to_f64(), -6.0);

        // Zero → mant 0.0, exp 0 (GMP convention; set_z does not normalize).
        let z = Dpe::from_int::<4>(&Int::<4>::from_i64(0));
        assert_eq!((z.mant, z.exp), (0.0, 0));

        // Truncation TOWARD ZERO to 53 bits: 2^60 + 1 (61 bits) drops the
        // low set bit; value becomes exactly 2^60.
        let d = Dpe::from_int::<4>(&Int::<4>::from_i64((1i64 << 60) + 1));
        assert_eq!(d.exp, 61);
        assert_eq!(d.mant, 0.5);
        assert_eq!(d.to_f64(), (1u64 << 60) as f64);
    }

    /// `to_int` (`dpe_get_z`): round-trips ≤53-bit integers exactly, rounds
    /// ties away from zero, and handles the `exp >= 53` (large integer) and
    /// `exp < 0` (→ 0) branches.
    #[test]
    fn to_int_round_trip_and_rounding() {
        for v in [
            0i64,
            1,
            -1,
            5,
            -6,
            1000,
            -1000,
            (1i64 << 52) - 1,
            -(1i64 << 52),
        ] {
            let back = Dpe::from_int::<4>(&Int::<4>::from_i64(v)).to_int::<4>();
            assert_eq!(back, Int::<4>::from_i64(v), "round-trip {v}");
        }
        // Ties away from zero (C round()).
        assert_eq!(Dpe::from_f64(2.5).to_int::<4>(), Int::<4>::from_i64(3));
        assert_eq!(Dpe::from_f64(-2.5).to_int::<4>(), Int::<4>::from_i64(-3));
        assert_eq!(Dpe::from_f64(2.4).to_int::<4>(), Int::<4>::from_i64(2));
        // |y| < 1/2 → 0.
        assert_eq!(Dpe::from_f64(0.4).to_int::<4>(), Int::<4>::from_i64(0));
        assert_eq!(Dpe::from_f64(-0.4).to_int::<4>(), Int::<4>::from_i64(0));

        // Large integer (exp >= 53 path): 2^100 round-trips exactly.
        let two = Dpe::from_f64(2.0);
        let mut p = Dpe::from_f64(1.0);
        for _ in 0..100 {
            p = p.mul(&two);
        }
        let expected = *Uint::<4>::ONE.shl_vartime(100).as_int();
        assert_eq!(p.to_int::<4>(), expected, "2^100 via exp>=53 branch");
    }
}
