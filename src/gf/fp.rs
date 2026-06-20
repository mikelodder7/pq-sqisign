// SPDX-License-Identifier: MIT OR Apache-2.0
//! `F_p` field elements per SQIsign security level.
//!
//! Each level's `F_p` type is `crypto_bigint::modular::ConstMontyForm`
//! specialised to that level's prime. The [`BaseField`] trait abstracts the
//! few operations the quadratic-extension and Montgomery-curve modules need,
//! so those modules can be written once and instantiated three times.

use crypto_bigint::Uint;
use crypto_bigint::modular::{ConstMontyForm, ConstMontyParams};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, ConstantTimeLess, CtOption};

pub use crate::params::lvl1::{Fp1Element, Lvl1Modulus};
pub use crate::params::lvl3::{Fp3Element, Lvl3Modulus};
pub use crate::params::lvl5::{Fp5Element, Lvl5Modulus};

/// Common surface of `F_p` element types — abstracted so the quadratic
/// extension `Fp2<F>` and the Montgomery curve are written once.
pub trait BaseField:
    Sized + Copy + core::fmt::Debug + Eq + Default + ConstantTimeEq + ConditionallySelectable
{
    /// Encoded byte length of one element (little-endian, fixed-width).
    const ENCODED_BYTES: usize;
    /// Bit-length of the level's prime `p` (251 / 383 / 505 for L1/L3/L5).
    /// Used by `hash::hash_to_fp` to mask squeezed bytes into `[0, 2^B)`
    /// before the rejection-sampling acceptance check.
    const BIT_LENGTH: usize;

    /// Additive identity.
    fn zero() -> Self;
    /// Multiplicative identity.
    fn one() -> Self;

    /// `Choice::from(1)` iff the value is `0`.
    fn is_zero(&self) -> Choice;

    /// `self + other`.
    fn add(&self, other: &Self) -> Self;
    /// `self - other`.
    fn sub(&self, other: &Self) -> Self;
    /// `-self`.
    fn negate(&self) -> Self;
    /// `self * other`.
    fn mul(&self, other: &Self) -> Self;
    /// `self * self`, faster than `mul(&self, &self)`.
    fn square(&self) -> Self;
    /// `self + self`.
    fn double(&self) -> Self;
    /// `self / 2`.
    fn half(&self) -> Self;
    /// Multiplicative inverse — `Some` iff `self` is non-zero.
    fn invert(&self) -> CtOption<Self>;
    /// Square root of `self` when one exists. Implementation uses the
    /// closed form for `p ≡ 3 mod 4`: `r = self^((p+1)/4)`. Returns
    /// `Some(r)` exactly when `r² = self`.
    fn sqrt(&self) -> CtOption<Self>;
    /// `Choice::TRUE` iff `self` is a quadratic residue (including `0`).
    /// Euler criterion: `self^((p−1)/2) ∈ {0, 1}`.
    fn is_square(&self) -> Choice;
    /// `self^((p−3)/4)` — used by the `Fp2` square-root routine to share
    /// one exponentiation across the inner-`sqrt` and the imaginary-coordinate
    /// division by `2x`.
    fn exp3div4(&self) -> Self;

    /// Encode the canonical residue to `out` in little-endian fixed-width form.
    /// `out.len()` must be at least [`Self::ENCODED_BYTES`].
    fn to_bytes_le(&self, out: &mut [u8]);
    /// Decode from a little-endian byte slice; non-canonical (`>= p`) returns `None`.
    fn from_bytes_le(b: &[u8]) -> CtOption<Self>;
}

/// Hot-path multiply/square, dispatched per concrete field type. Level 1
/// (4-limb, p = 5·2^248 − 1) routes to the hand-written x86_64 `mulx/adcx/adox`
/// backend in [`crate::gf::fp1_intrinsics`] when the target supports BMI2+ADX,
/// falling back to `crypto-bigint` otherwise. Levels 3/5 always use
/// `crypto-bigint`. The intrinsic result is canonicalized back to `[0, p)` so
/// it round-trips through `ConstMontyForm` exactly (proven byte-identical by
/// the `fp1_intrinsics` differential test and the keygen byte-exact KAT);
/// add/sub stay on `crypto-bigint` since multiply/square dominate field cost.
pub(crate) trait FpArith: Sized {
    /// `self * other`.
    fn fmul(&self, other: &Self) -> Self;
    /// `self * self`.
    fn fsqr(&self) -> Self;
}

impl FpArith for Fp1Element {
    #[inline]
    fn fmul(&self, other: &Self) -> Self {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "bmi2",
            target_feature = "adx"
        ))]
        {
            let a: [u64; 4] = core::array::from_fn(|i| self.as_montgomery().as_limbs()[i].0);
            let b: [u64; 4] = core::array::from_fn(|i| other.as_montgomery().as_limbs()[i].0);
            let r = unsafe { crate::gf::fp1_intrinsics::mul(&a, &b) };
            let rc = crate::gf::fp1_intrinsics::to_canonical(&r);
            Self::from_montgomery(Uint::<4>::from_words(rc))
        }
        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "bmi2",
            target_feature = "adx"
        )))]
        {
            self * other
        }
    }
    #[inline]
    fn fsqr(&self) -> Self {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "bmi2",
            target_feature = "adx"
        ))]
        {
            let a: [u64; 4] = core::array::from_fn(|i| self.as_montgomery().as_limbs()[i].0);
            let r = unsafe { crate::gf::fp1_intrinsics::square(&a) };
            let rc = crate::gf::fp1_intrinsics::to_canonical(&r);
            Self::from_montgomery(Uint::<4>::from_words(rc))
        }
        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "bmi2",
            target_feature = "adx"
        )))]
        {
            ConstMontyForm::<Lvl1Modulus, 4>::square(self)
        }
    }
}

impl FpArith for Fp3Element {
    #[inline]
    fn fmul(&self, other: &Self) -> Self {
        self * other
    }
    #[inline]
    fn fsqr(&self) -> Self {
        ConstMontyForm::<Lvl3Modulus, 6>::square(self)
    }
}

impl FpArith for Fp5Element {
    #[inline]
    fn fmul(&self, other: &Self) -> Self {
        self * other
    }
    #[inline]
    fn fsqr(&self) -> Self {
        ConstMontyForm::<Lvl5Modulus, 8>::square(self)
    }
}

macro_rules! impl_base_field {
    (
        $alias:ident,
        $modulus:ty,
        $bytes:literal,
        $limbs:literal,
        $bits:literal
    ) => {
        impl BaseField for $alias {
            const ENCODED_BYTES: usize = $bytes;
            const BIT_LENGTH: usize = $bits;

            #[inline]
            fn zero() -> Self {
                <Self>::ZERO
            }
            #[inline]
            fn one() -> Self {
                <Self>::ONE
            }
            #[inline]
            fn is_zero(&self) -> Choice {
                <Self as ConstantTimeEq>::ct_eq(self, &<Self>::ZERO)
            }
            #[inline]
            fn add(&self, other: &Self) -> Self {
                self + other
            }
            #[inline]
            fn sub(&self, other: &Self) -> Self {
                self - other
            }
            #[inline]
            fn negate(&self) -> Self {
                -*self
            }
            #[inline]
            fn mul(&self, other: &Self) -> Self {
                <$alias as FpArith>::fmul(self, other)
            }
            #[inline]
            fn square(&self) -> Self {
                <$alias as FpArith>::fsqr(self)
            }
            #[inline]
            fn double(&self) -> Self {
                ConstMontyForm::<$modulus, $limbs>::double(self)
            }
            #[inline]
            fn half(&self) -> Self {
                ConstMontyForm::<$modulus, $limbs>::div_by_2(self)
            }
            fn invert(&self) -> CtOption<Self> {
                let cb = ConstMontyForm::<$modulus, $limbs>::invert(self);
                let is_some = Choice::from(u8::from(cb.is_some()));
                let val = cb.into_option_copied().unwrap_or(<Self>::ZERO);
                CtOption::new(val, is_some)
            }
            fn sqrt(&self) -> CtOption<Self> {
                let p = <$modulus as ConstMontyParams<$limbs>>::PARAMS.modulus();
                let exp = p.as_ref().wrapping_add(&Uint::<$limbs>::ONE).shr_vartime(2);
                let r = ConstMontyForm::<$modulus, $limbs>::pow(self, &exp);
                let r_sq = ConstMontyForm::<$modulus, $limbs>::square(&r);
                let is_sqrt = <Self as ConstantTimeEq>::ct_eq(&r_sq, self);
                CtOption::new(r, is_sqrt)
            }
            fn is_square(&self) -> Choice {
                let p = <$modulus as ConstMontyParams<$limbs>>::PARAMS.modulus();
                let exp = p.as_ref().wrapping_sub(&Uint::<$limbs>::ONE).shr_vartime(1);
                let r = ConstMontyForm::<$modulus, $limbs>::pow(self, &exp);
                let one = <Self>::ONE;
                let is_one = <Self as ConstantTimeEq>::ct_eq(&r, &one);
                let is_zero = <Self as ConstantTimeEq>::ct_eq(self, &<Self>::ZERO);
                is_one | is_zero
            }
            fn exp3div4(&self) -> Self {
                let p = <$modulus as ConstMontyParams<$limbs>>::PARAMS.modulus();
                let three = Uint::<$limbs>::from_u64(3);
                let exp = p.as_ref().wrapping_sub(&three).shr_vartime(2);
                ConstMontyForm::<$modulus, $limbs>::pow(self, &exp)
            }
            fn to_bytes_le(&self, out: &mut [u8]) {
                debug_assert!(out.len() >= $bytes);
                let n: Uint<$limbs> = ConstMontyForm::<$modulus, $limbs>::retrieve(self);
                let le = n.to_le_bytes();
                out[..$bytes].copy_from_slice(&le.as_ref()[..$bytes]);
            }
            fn from_bytes_le(b: &[u8]) -> CtOption<Self> {
                if b.len() < $bytes {
                    return CtOption::new(<Self>::ZERO, Choice::from(0));
                }
                let mut buf = [0u8; $bytes];
                buf.copy_from_slice(&b[..$bytes]);
                let n = Uint::<$limbs>::from_le_slice(&buf);
                let modulus = <$modulus as ConstMontyParams<$limbs>>::PARAMS.modulus();
                let in_range = ConstantTimeLess::ct_lt(&n, modulus.as_ref());
                CtOption::new(ConstMontyForm::<$modulus, $limbs>::new(&n), in_range)
            }
        }
    };
}

impl_base_field!(Fp1Element, Lvl1Modulus, 32, 4, 251);
impl_base_field!(Fp3Element, Lvl3Modulus, 48, 6, 383);
impl_base_field!(Fp5Element, Lvl5Modulus, 64, 8, 505);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_and_one_distinct() {
        assert_ne!(
            <Fp1Element as BaseField>::zero(),
            <Fp1Element as BaseField>::one()
        );
        assert_ne!(
            <Fp3Element as BaseField>::zero(),
            <Fp3Element as BaseField>::one()
        );
        assert_ne!(
            <Fp5Element as BaseField>::zero(),
            <Fp5Element as BaseField>::one()
        );
    }

    #[test]
    fn one_squared_is_one() {
        let one = <Fp1Element as BaseField>::one();
        assert_eq!(one.square(), one);
    }

    #[test]
    fn double_equals_self_plus_self() {
        let one = <Fp1Element as BaseField>::one();
        let two_a = one.double();
        let two_b = BaseField::add(&one, &one);
        assert_eq!(two_a, two_b);
    }

    #[test]
    fn half_doubles_back() {
        let one = <Fp1Element as BaseField>::one();
        let two = one.double();
        let half = two.half();
        assert_eq!(half, one);
    }

    #[test]
    fn invert_one_is_one() {
        let one = <Fp1Element as BaseField>::one();
        let opt = BaseField::invert(&one);
        assert!(bool::from(opt.is_some()));
        let inv = opt.unwrap_or(<Fp1Element as BaseField>::zero());
        assert_eq!(inv, one);
    }

    #[test]
    fn invert_zero_is_none() {
        let z = <Fp1Element as BaseField>::zero();
        let opt = BaseField::invert(&z);
        assert!(bool::from(opt.is_none()));
    }

    #[test]
    fn round_trip_zero_bytes() {
        let z = <Fp1Element as BaseField>::zero();
        let mut bytes = [0u8; 32];
        z.to_bytes_le(&mut bytes);
        let opt = Fp1Element::from_bytes_le(&bytes);
        assert!(bool::from(opt.is_some()));
        let z2 = opt.unwrap_or(<Fp1Element as BaseField>::zero());
        assert_eq!(z, z2);
    }

    #[test]
    fn sqrt_of_one_is_one_lvl1() {
        let one = <Fp1Element as BaseField>::one();
        let opt = one.sqrt();
        assert!(bool::from(opt.is_some()));
        assert_eq!(opt.unwrap_or(<Fp1Element as BaseField>::zero()), one);
    }

    #[test]
    fn sqrt_squared_round_trips_lvl1() {
        // For each small integer x, check sqrt(x^2)^2 == x^2.
        let one = <Fp1Element as BaseField>::one();
        let mut acc = one;
        for _ in 0..16 {
            let sq = acc.square();
            let r = sq.sqrt().unwrap_or(<Fp1Element as BaseField>::zero());
            assert_eq!(r.square(), sq);
            acc = acc.add(&one);
        }
    }

    #[test]
    fn sqrt_squared_round_trips_lvl3() {
        let one = <Fp3Element as BaseField>::one();
        let mut acc = one;
        for _ in 0..16 {
            let sq = acc.square();
            let r = sq.sqrt().unwrap_or(<Fp3Element as BaseField>::zero());
            assert_eq!(r.square(), sq);
            acc = acc.add(&one);
        }
    }

    #[test]
    fn sqrt_squared_round_trips_lvl5() {
        let one = <Fp5Element as BaseField>::one();
        let mut acc = one;
        for _ in 0..16 {
            let sq = acc.square();
            let r = sq.sqrt().unwrap_or(<Fp5Element as BaseField>::zero());
            assert_eq!(r.square(), sq);
            acc = acc.add(&one);
        }
    }

    #[test]
    fn sqrt_of_zero_is_zero() {
        let z = <Fp1Element as BaseField>::zero();
        let r = z.sqrt();
        assert!(bool::from(r.is_some()));
        assert_eq!(r.unwrap_or(<Fp1Element as BaseField>::one()), z);
    }

    #[test]
    fn is_square_matches_sqrt() {
        // For every i in [0, 16): i*i is square; if i+1 happens to be non-square,
        // is_square should return FALSE and sqrt should return None for it.
        let one = <Fp1Element as BaseField>::one();
        let mut acc = one;
        let mut found_non_square = false;
        for _ in 0..32 {
            let is_sq = bool::from(acc.is_square());
            let sqrt_opt = acc.sqrt();
            let sqrt_is_some = bool::from(sqrt_opt.is_some());
            assert_eq!(is_sq, sqrt_is_some, "is_square and sqrt-is-some must agree");
            if !is_sq {
                found_non_square = true;
            }
            acc = acc.add(&one);
        }
        // p = 5*2^248 - 1: roughly half of small integers should be non-squares.
        assert!(
            found_non_square,
            "expected at least one non-square in [1, 32]"
        );
    }

    #[test]
    fn round_trip_one_bytes() {
        let o = <Fp1Element as BaseField>::one();
        let mut bytes = [0u8; 32];
        o.to_bytes_le(&mut bytes);
        let opt = Fp1Element::from_bytes_le(&bytes);
        let o2 = opt.unwrap_or(<Fp1Element as BaseField>::zero());
        assert_eq!(o, o2);
    }

    // ── Fp byte round-trip at production NIST levels ──

    #[test]
    fn round_trip_zero_bytes_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        let z = <Fp3Element as BaseField>::zero();
        let mut bytes = [0u8; 48];
        z.to_bytes_le(&mut bytes);
        let opt = Fp3Element::from_bytes_le(&bytes);
        let z2 = opt.unwrap_or(<Fp3Element as BaseField>::zero());
        assert_eq!(z, z2);
    }

    #[test]
    fn round_trip_one_bytes_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        let o = <Fp3Element as BaseField>::one();
        let mut bytes = [0u8; 48];
        o.to_bytes_le(&mut bytes);
        let opt = Fp3Element::from_bytes_le(&bytes);
        let o2 = opt.unwrap_or(<Fp3Element as BaseField>::zero());
        assert_eq!(o, o2);
    }

    #[test]
    fn round_trip_zero_bytes_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        let z = <Fp5Element as BaseField>::zero();
        let mut bytes = [0u8; 64];
        z.to_bytes_le(&mut bytes);
        let opt = Fp5Element::from_bytes_le(&bytes);
        let z2 = opt.unwrap_or(<Fp5Element as BaseField>::zero());
        assert_eq!(z, z2);
    }

    #[test]
    fn round_trip_one_bytes_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        let o = <Fp5Element as BaseField>::one();
        let mut bytes = [0u8; 64];
        o.to_bytes_le(&mut bytes);
        let opt = Fp5Element::from_bytes_le(&bytes);
        let o2 = opt.unwrap_or(<Fp5Element as BaseField>::zero());
        assert_eq!(o, o2);
    }

    #[test]
    fn round_trip_iterated_bytes_at_lvl1() {
        // Iterate over 16 small Fp values (1, 2, 3, ..., 16) at L1,
        // round-trip each, assert equality. Exercises non-trivial
        // values at production scale.
        let one = <Fp1Element as BaseField>::one();
        let mut acc = one;
        let mut bytes = [0u8; 32];
        for i in 0..16 {
            acc.to_bytes_le(&mut bytes);
            let opt = Fp1Element::from_bytes_le(&bytes);
            let acc2 = opt.unwrap_or(<Fp1Element as BaseField>::zero());
            assert_eq!(acc, acc2, "round-trip failed at iteration {i}");
            acc = acc.add(&one);
        }
    }

    #[test]
    fn round_trip_iterated_bytes_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        let one = <Fp3Element as BaseField>::one();
        let mut acc = one;
        let mut bytes = [0u8; 48];
        for i in 0..16 {
            acc.to_bytes_le(&mut bytes);
            let opt = Fp3Element::from_bytes_le(&bytes);
            let acc2 = opt.unwrap_or(<Fp3Element as BaseField>::zero());
            assert_eq!(acc, acc2, "round-trip failed at iteration {i}");
            acc = acc.add(&one);
        }
    }

    #[test]
    fn round_trip_iterated_bytes_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        let one = <Fp5Element as BaseField>::one();
        let mut acc = one;
        let mut bytes = [0u8; 64];
        for i in 0..16 {
            acc.to_bytes_le(&mut bytes);
            let opt = Fp5Element::from_bytes_le(&bytes);
            let acc2 = opt.unwrap_or(<Fp5Element as BaseField>::zero());
            assert_eq!(acc, acc2, "round-trip failed at iteration {i}");
            acc = acc.add(&one);
        }
    }

    // ── fuzz-style randomized Fp property tests at L1/L3/L5 ──

    /// Generic helper: round-trip 8 pseudo-random Fp elements through
    /// bytes. Mirrors `check_fp2_random_roundtrip_bytes` at the
    /// base-field layer.
    fn check_fp_random_roundtrip_bytes<F: BaseField>() {
        use crate::hash::hash_to_fp;
        let n = F::ENCODED_BYTES;
        let mut buf = [0u8; 64]; // big enough for L5's 64-byte Fp encoding
        for i in 0u8..8 {
            let x = hash_to_fp::<F>(b"S93-roundtrip", &[i], 16)
                .into_option()
                .unwrap_or_else(F::one);
            x.to_bytes_le(&mut buf[..n]);
            let y = F::from_bytes_le(&buf[..n]).unwrap_or_else(F::zero);
            assert_eq!(x, y, "Fp random round-trip failed at iteration {i}");
        }
    }

    #[test]
    fn fp_random_roundtrip_bytes_at_lvl1() {
        check_fp_random_roundtrip_bytes::<Fp1Element>();
    }

    #[test]
    fn fp_random_roundtrip_bytes_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_fp_random_roundtrip_bytes::<Fp3Element>();
    }

    #[test]
    fn fp_random_roundtrip_bytes_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_fp_random_roundtrip_bytes::<Fp5Element>();
    }

    /// Generic helper: for each pseudo-random Fp `x`, check `sqrt(x²)² == x²`.
    /// `x²` is always a square so sqrt must succeed; the recovered root
    /// can be ±x but squaring it recovers x² either way.
    fn check_fp_random_sqrt_squared<F: BaseField>() {
        use crate::hash::hash_to_fp;
        for i in 0u8..8 {
            let x = hash_to_fp::<F>(b"S93-sqrt-sq", &[i], 16)
                .into_option()
                .unwrap_or_else(F::one);
            let sq = x.square();
            let r = sq.sqrt().unwrap_or_else(F::zero);
            let r_sq = r.square();
            assert_eq!(
                r_sq, sq,
                "sqrt(x²)² must equal x² for random Fp x at iteration {i}",
            );
        }
    }

    #[test]
    fn fp_random_sqrt_squared_at_lvl1() {
        check_fp_random_sqrt_squared::<Fp1Element>();
    }

    #[test]
    fn fp_random_sqrt_squared_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_fp_random_sqrt_squared::<Fp3Element>();
    }

    #[test]
    fn fp_random_sqrt_squared_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_fp_random_sqrt_squared::<Fp5Element>();
    }

    /// Generic helper: `x · x.invert() == one` for pseudo-random Fp x.
    /// Skips the zero edge case (invert(0) yields None per the trait
    /// contract).
    fn check_fp_random_invert_yields_identity<F: BaseField>() {
        use crate::hash::hash_to_fp;
        for i in 0u8..8 {
            let x = hash_to_fp::<F>(b"S93-invert", &[i], 16)
                .into_option()
                .unwrap_or_else(F::one);
            if bool::from(<F as ConstantTimeEq>::ct_eq(&x, &F::zero())) {
                continue;
            }
            let inv = x.invert().unwrap_or_else(F::zero);
            let prod = x.mul(&inv);
            assert_eq!(
                prod,
                F::one(),
                "x · x.invert() must equal one for random Fp x at iteration {i}",
            );
        }
    }

    #[test]
    fn fp_random_invert_at_lvl1() {
        check_fp_random_invert_yields_identity::<Fp1Element>();
    }

    #[test]
    fn fp_random_invert_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_fp_random_invert_yields_identity::<Fp3Element>();
    }

    #[test]
    fn fp_random_invert_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_fp_random_invert_yields_identity::<Fp5Element>();
    }

    /// PK-encode byte convention: `Fp1Element::to_bytes_le` must be canonical
    /// (Montgomery→normal) little-endian, byte-IDENTICAL to the C `fp_encode`
    /// (`gf/ref/lvl1/fp_p5248_64.c`: `redc` then 32 bytes low-to-high). The
    /// input is the SAME value a standalone C `fp_encode` oracle round-tripped
    /// (out == in, decode_ok = 0xffffffff). Canonical-LE of an integer < p is
    /// unambiguous, so a correct round-trip on both sides ⇒ byte-equal output.
    /// `ONE → 01 00…` additionally guards the Montgomery convention (a leaked
    /// Montgomery value would encode `R mod p`, not 1). This is the
    /// math-oracle-invisible PK-serialization checkpoint for the keygen KAT
    /// (C `proj_to_bytes` = `fp2_encode(A·C⁻¹)` = `fp_encode(re)||fp_encode(im)`).
    #[test]
    fn fp_encode_matches_c_canonical_le() {
        // Byte i = (i·17 + 3) mod 256 for i<16, high 16 zero (< p). Matches the
        // C oracle input 031425364758697a8b9cadbecfe0f102 0…0.
        let mut input = [0u8; 32];
        for (i, b) in input.iter_mut().take(16).enumerate() {
            *b = (i * 17 + 3).to_le_bytes()[0];
        }
        let v = Fp1Element::from_bytes_le(&input)
            .into_option()
            .expect("value is canonical (< p)");
        let mut out = [0u8; 32];
        v.to_bytes_le(&mut out);
        assert_eq!(
            out, input,
            "Fp1 to_bytes_le must be canonical little-endian (== C fp_encode)",
        );

        // ONE encodes as canonical 1 = 01 00 … 00 (Montgomery-leak guard).
        let mut one_le = [0u8; 32];
        <Fp1Element as BaseField>::one().to_bytes_le(&mut one_le);
        let mut expected_one = [0u8; 32];
        expected_one[0] = 1;
        assert_eq!(
            one_le, expected_one,
            "ONE must encode as canonical 1 (no Montgomery leak)",
        );
    }
}
