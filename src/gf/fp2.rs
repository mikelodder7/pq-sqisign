// SPDX-License-Identifier: MIT OR Apache-2.0
//! Quadratic extension field `F_{p^2} = F_p[i]/(i^2 + 1)`.
//!
//! The SQIsign spec uses `i^2 = -1` (admissible because `p ≡ 3 mod 4` at
//! every level). An element `a + b·i` is stored as the pair `(re, im)`
//! where both components are base-field elements
//! (see [`crate::gf::fp::BaseField`]).
//!
//! Multiplication uses Karatsuba (`re_out = re_a · re_b − im_a · im_b`,
//! `im_out = (re_a + im_a)(re_b + im_b) − re_a·re_b − im_a·im_b`). Squaring
//! exploits the binomial identity `(a + bi)^2 = (a-b)(a+b) + 2abi`.
//!
//! Inversion, Frobenius, `mul_by_i`, `is_square`, `sqrt` all follow from the
//! base-field surface and the `i^2 = -1` identity.

use core::marker::PhantomData;

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

use super::fp::BaseField;

/// An element of `F_{p^2}` written as `re + im · i`.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Fp2<F: BaseField> {
    /// Real component.
    pub re: F,
    /// Imaginary component (coefficient of `i`).
    pub im: F,
    _marker: PhantomData<F>,
}

impl<F: BaseField> Fp2<F> {
    /// Construct from real and imaginary components.
    #[inline]
    pub const fn new(re: F, im: F) -> Self {
        Self {
            re,
            im,
            _marker: PhantomData,
        }
    }

    /// Additive identity `0 + 0i`.
    #[inline]
    pub fn zero() -> Self {
        Self::new(F::zero(), F::zero())
    }

    /// Multiplicative identity `1 + 0i`.
    #[inline]
    pub fn one() -> Self {
        Self::new(F::one(), F::zero())
    }

    /// `0 + 1i`.
    #[inline]
    pub fn img() -> Self {
        Self::new(F::zero(), F::one())
    }

    /// Choice::TRUE iff `self == 0 + 0i`.
    #[inline]
    pub fn is_zero(&self) -> Choice {
        self.re.is_zero() & self.im.is_zero()
    }

    /// `self + rhs`.
    #[inline]
    pub fn add(&self, rhs: &Self) -> Self {
        Self::new(self.re.add(&rhs.re), self.im.add(&rhs.im))
    }

    /// `self - rhs`.
    #[inline]
    pub fn sub(&self, rhs: &Self) -> Self {
        Self::new(self.re.sub(&rhs.re), self.im.sub(&rhs.im))
    }

    /// `-self`.
    #[inline]
    pub fn negate(&self) -> Self {
        Self::new(self.re.negate(), self.im.negate())
    }

    /// `self * rhs` via Karatsuba (3 base-field multiplications).
    pub fn mul(&self, rhs: &Self) -> Self {
        let t0 = self.re.mul(&rhs.re);
        let t1 = self.im.mul(&rhs.im);
        let t2 = self.re.add(&self.im);
        let t3 = rhs.re.add(&rhs.im);
        let t4 = t2.mul(&t3);
        let re = t0.sub(&t1);
        let im = t4.sub(&t0).sub(&t1);
        Self::new(re, im)
    }

    /// `self * self` via `(a+bi)^2 = (a-b)(a+b) + 2ab·i`.
    pub fn square(&self) -> Self {
        let sum = self.re.add(&self.im);
        let dif = self.re.sub(&self.im);
        let re = sum.mul(&dif);
        let twoab = self.re.mul(&self.im).double();
        Self::new(re, twoab)
    }

    /// `self + self`.
    #[inline]
    pub fn double(&self) -> Self {
        Self::new(self.re.double(), self.im.double())
    }

    /// `self / 2`.
    #[inline]
    pub fn half(&self) -> Self {
        Self::new(self.re.half(), self.im.half())
    }

    /// Multiplication by `i`: `(a + bi) · i = -b + ai`.
    #[inline]
    pub fn mul_by_i(&self) -> Self {
        Self::new(self.im.negate(), self.re)
    }

    /// Frobenius: in `F_{p^2}` with `i^2 = -1`, the Frobenius `x ↦ x^p` sends
    /// `a + bi` to `a - bi`.
    #[inline]
    pub fn frobenius(&self) -> Self {
        Self::new(self.re, self.im.negate())
    }

    /// `Choice::TRUE` iff `self` is a square in `F_{p^2}`. For `i^2 = -1`
    /// this is equivalent to `(re^2 + im^2)` being a square in `F_p`.
    pub fn is_square(&self) -> Choice {
        let norm = self.re.square().add(&self.im.square());
        norm.is_square()
    }

    /// Square root in `F_{p^2}` for `p ≡ 3 mod 4`.
    ///
    /// Algorithm (Aardal, Bernstein, Castryck et al., ePrint 2024/1563 §3.2):
    ///
    /// 1. `δ = sqrt(re^2 + im^2)` in `F_p`. If none, `self` has no sqrt.
    /// 2. If `im = 0`, restore `δ = re` to avoid the `δ = -re` degeneracy.
    /// 3. `s = δ + re`; `t = 2 s`; `w = t^((p−3)/4)`.
    /// 4. `x = s · w`; `y = w · im`; if `(2x)^2 = t` return `(x, y)`,
    ///    else return `(y, -x)` — the alternate root.
    ///
    /// Constant-time. Returns `None` exactly when `self` is not a square.
    pub fn sqrt(&self) -> CtOption<Self> {
        let inner_sqrt = self.re.square().add(&self.im.square()).sqrt();
        let inner_is_some = inner_sqrt.is_some();
        let delta0 = inner_sqrt.unwrap_or(F::zero());
        // If im == 0, set delta = re; otherwise keep delta0.
        let im_is_zero = self.im.is_zero();
        let delta = F::conditional_select(&delta0, &self.re, im_is_zero);
        let s = delta.add(&self.re);
        let t = s.double();
        let w = t.exp3div4();
        let x = s.mul(&w);
        let y = w.mul(&self.im);
        let two_x = x.double();
        let two_x_sq = two_x.square();
        // f == TRUE iff (2x)^2 == t  →  use (x, y); else use (y, -x).
        let f = two_x_sq.ct_eq(&t);
        let neg_x = x.negate();
        let out_re = F::conditional_select(&y, &x, f);
        let out_im = F::conditional_select(&neg_x, &y, f);
        CtOption::new(Self::new(out_re, out_im), inner_is_some)
    }

    /// Multiplicative inverse via `(a + bi)^{-1} = (a - bi) / (a^2 + b^2)`.
    pub fn invert(&self) -> CtOption<Self> {
        let norm = self.re.square().add(&self.im.square());
        let norm_inv = norm.invert();
        let is_some = norm_inv.is_some();
        let inv = norm_inv.unwrap_or(F::zero());
        let re = self.re.mul(&inv);
        let im = self.im.negate().mul(&inv);
        CtOption::new(Self::new(re, im), is_some)
    }

    /// Encode `(re, im)` as `re_le_bytes || im_le_bytes`.
    pub fn to_bytes_le(&self, out: &mut [u8]) {
        let n = F::ENCODED_BYTES;
        debug_assert!(out.len() >= 2 * n);
        self.re.to_bytes_le(&mut out[..n]);
        self.im.to_bytes_le(&mut out[n..2 * n]);
    }

    /// Decode `(re, im)` from `re_le_bytes || im_le_bytes`. Non-canonical
    /// components yield `None`.
    pub fn from_bytes_le(b: &[u8]) -> CtOption<Self> {
        let n = F::ENCODED_BYTES;
        if b.len() < 2 * n {
            return CtOption::new(Self::zero(), Choice::from(0));
        }
        let re_opt = F::from_bytes_le(&b[..n]);
        let im_opt = F::from_bytes_le(&b[n..2 * n]);
        let is_some = re_opt.is_some() & im_opt.is_some();
        let re = re_opt.unwrap_or(F::zero());
        let im = im_opt.unwrap_or(F::zero());
        CtOption::new(Self::new(re, im), is_some)
    }
}

impl<F: BaseField> ConstantTimeEq for Fp2<F> {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.re.ct_eq(&other.re) & self.im.ct_eq(&other.im)
    }
}

impl<F: BaseField> ConditionallySelectable for Fp2<F> {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self::new(
            F::conditional_select(&a.re, &b.re, choice),
            F::conditional_select(&a.im, &b.im, choice),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf::fp::Fp1Element;

    type F2 = Fp2<Fp1Element>;

    #[test]
    fn one_squared_is_one() {
        let one = F2::one();
        assert_eq!(one.square(), one);
    }

    #[test]
    fn i_squared_is_minus_one() {
        let i = F2::img();
        let i_sq = i.square();
        let minus_one = F2::one().negate();
        assert_eq!(i_sq, minus_one);
    }

    #[test]
    fn mul_by_i_matches_explicit_mul() {
        let x = F2::new(
            <Fp1Element as BaseField>::one().double(),
            <Fp1Element as BaseField>::one(),
        );
        let y_a = x.mul_by_i();
        let y_b = x.mul(&F2::img());
        assert_eq!(y_a, y_b);
    }

    #[test]
    fn invert_of_one_is_one() {
        let one = F2::one();
        let inv = one.invert();
        assert!(bool::from(inv.is_some()));
        let r = inv.unwrap_or(F2::zero());
        assert_eq!(r, one);
    }

    #[test]
    fn invert_zero_is_none() {
        let z = F2::zero();
        let inv = z.invert();
        assert!(bool::from(inv.is_none()));
    }

    #[test]
    fn x_times_inv_x_is_one() {
        let x = F2::new(
            <Fp1Element as BaseField>::one().double(), // re = 2
            <Fp1Element as BaseField>::one(),          // im = 1
        );
        let inv = x.invert().unwrap_or(F2::zero());
        let prod = x.mul(&inv);
        assert_eq!(prod, F2::one());
    }

    #[test]
    fn frobenius_squared_is_identity() {
        let x = F2::new(
            <Fp1Element as BaseField>::one().double(),
            <Fp1Element as BaseField>::one(),
        );
        assert_eq!(x.frobenius().frobenius(), x);
    }

    #[test]
    fn fp2_sqrt_of_one_is_one() {
        let one = F2::one();
        let r = one.sqrt().unwrap_or(F2::zero());
        assert_eq!(r.square(), one);
    }

    #[test]
    fn fp2_sqrt_squared_round_trips() {
        // For 16 distinct non-trivial F_{p^2} values, check sqrt(x^2)^2 == x^2.
        let one = <Fp1Element as BaseField>::one();
        let mut a = one;
        for _ in 0..16 {
            let mut b = one;
            for _ in 0..4 {
                let x = F2::new(a, b);
                let sq = x.square();
                let opt = sq.sqrt();
                assert!(
                    bool::from(opt.is_some()),
                    "square root of a square must exist"
                );
                let r = opt.unwrap_or(F2::zero());
                assert_eq!(r.square(), sq);
                b = b.double();
            }
            a = a.add(&one);
        }
    }

    #[test]
    fn fp2_is_square_matches_sqrt() {
        let one = <Fp1Element as BaseField>::one();
        let mut a = one;
        for _ in 0..16 {
            let x = F2::new(a, a);
            let is_sq = bool::from(x.is_square());
            let sqrt_is_some = bool::from(x.sqrt().is_some());
            assert_eq!(is_sq, sqrt_is_some);
            a = a.add(&one);
        }
    }

    #[test]
    fn fp2_pure_imaginary_sqrt() {
        // For p ≡ 3 mod 4, -1 is a non-square in F_p but a square in F_{p^2}
        // (sqrt(-1) = i). Verify.
        let minus_one = F2::one().negate();
        let r = minus_one.sqrt().unwrap_or(F2::zero());
        assert_eq!(r.square(), minus_one);
    }

    #[test]
    fn fp2_roundtrip_bytes() {
        let x = F2::new(
            <Fp1Element as BaseField>::one().double(),
            <Fp1Element as BaseField>::one(),
        );
        let n = Fp1Element::ENCODED_BYTES;
        let mut buf = [0u8; 64];
        x.to_bytes_le(&mut buf[..2 * n]);
        let y = F2::from_bytes_le(&buf[..2 * n]).unwrap_or(F2::zero());
        assert_eq!(x, y);
    }
}
