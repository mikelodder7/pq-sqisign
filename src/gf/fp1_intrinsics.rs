// SPDX-License-Identifier: MIT OR Apache-2.0
//! Hand-optimized x86_64 intrinsic backend for the Level-1 base field Fp.
//!
//! This module is a direct port of the C reference Broadwell implementation in
//! `src/gf/broadwell/lvl1/` of the SQIsign C reference. It provides raw-limb
//! arithmetic for the prime p = 5·2^248 − 1 in Montgomery form (R = 2^256).
//!
//! ## Representation
//!
//! Elements are `[u64; 4]` little-endian limbs (index 0 = least significant).
//! The representation is the same as `crypto_bigint::ConstMontyForm<Lvl1Modulus,4>`
//! internal storage, so values can be round-tripped via `as_montgomery()` /
//! `from_montgomery()` without any conversion arithmetic.
//!
//! Output ranges:
//! - `mul` / `redc`: result in [0, 6·2^248)  (partially reduced)
//! - `add` / `sub`: result in [0, 2^251)      (partially reduced)
//! - `to_canonical`: result in [0, p)          (fully reduced)
//!
//! ## Constant-time guarantee
//!
//! All conditional reductions use arithmetic masking (sign-extending borrows /
//! carries to 64-bit words) rather than branches or memory indexing on secret
//! data. This mirrors the C reference, which was written with the same constraint.
//!
//! ## Feature gate
//!
//! Every function using intrinsics is gated on
//! `#[cfg(all(target_arch = "x86_64", target_feature = "bmi2", target_feature = "adx"))]`.
//! `to_canonical` uses only wrapping arithmetic and is available unconditionally.

#![allow(unsafe_code)]

// ──────────────────────────────────────────────────────────────────────────────
// Field constants (unconditional)
// ──────────────────────────────────────────────────────────────────────────────

/// p = 5·2^248 − 1, little-endian limbs.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
pub(crate) const P_LIMBS: [u64; 4] = [
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0x04ff_ffff_ffff_ffff,
];

/// Top limb of p+1 = 5·2^248. Lower three limbs of p+1 are zero.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
const P_PLUS_1_TOP: u64 = 0x0500_0000_0000_0000; // 5 << 56

/// Top limb of −p mod 2^256, for add correction.
/// −p = [1, 0, 0, 0xFB<<56].
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
const NEG_P_TOP: u64 = 0xFB00_0000_0000_0000; // 0xFBu64 << 56

/// Low limb of (−2q mod 2^256), for sub correction.
/// gf5248_sub adds 2q by subtracting −2q = [2, 0, 0, 0xF6<<56].
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
const SUB_NEG2Q_LO: u64 = 2;
/// High limb of (−2q mod 2^256).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
const SUB_NEG2Q_HI: u64 = 0xF600_0000_0000_0000; // 0xF6u64 << 56

// ──────────────────────────────────────────────────────────────────────────────
// to_canonical — portable, safe
// ──────────────────────────────────────────────────────────────────────────────

/// Normalize [0, 6·2^248) → canonical [0, p).
///
/// Port of `inner_gf5248_normalize` from `gf5248.h`.
/// Constant-time: uses borrow-to-mask arithmetic.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
pub(crate) fn to_canonical(a: &[u64; 4]) -> [u64; 4] {
    let (d0, b0) = a[0].overflowing_sub(P_LIMBS[0]);
    let (d1, b1a) = a[1].overflowing_sub(P_LIMBS[1]);
    let (d1, b1b) = d1.overflowing_sub(b0 as u64);
    let b1 = b1a | b1b;
    let (d2, b2a) = a[2].overflowing_sub(P_LIMBS[2]);
    let (d2, b2b) = d2.overflowing_sub(b1 as u64);
    let b2 = b2a | b2b;
    let (d3, b3a) = a[3].overflowing_sub(P_LIMBS[3]);
    let (d3, b3b) = d3.overflowing_sub(b2 as u64);
    let borrow = b3a | b3b;
    let m = (borrow as u64).wrapping_neg();
    let (r0, c0) = d0.overflowing_add(P_LIMBS[0] & m);
    let (r1, c1a) = d1.overflowing_add(P_LIMBS[1] & m);
    let (r1, c1b) = r1.overflowing_add(c0 as u64);
    let c1 = c1a | c1b;
    let (r2, c2a) = d2.overflowing_add(P_LIMBS[2] & m);
    let (r2, c2b) = r2.overflowing_add(c1 as u64);
    let c2 = c2a | c2b;
    let r3 = d3.wrapping_add(P_LIMBS[3] & m).wrapping_add(c2 as u64);
    [r0, r1, r2, r3]
}

// ──────────────────────────────────────────────────────────────────────────────
// Intrinsic-backed arithmetic
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
mod intrinsics {
    use super::{NEG_P_TOP, P_PLUS_1_TOP, SUB_NEG2Q_HI, SUB_NEG2Q_LO};
    use core::arch::x86_64::{_addcarryx_u64, _mulx_u64, _subborrow_u64};

    /// Field addition: a + b mod p, result in [0, 2^251).
    /// Port of `gf5248_add` from `gf5248.h`.
    /// # Safety
    /// Requires x86_64 + BMI2 + ADX.
    pub unsafe fn add(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
        unsafe {
            let mut d0: u64 = 0;
            let mut d1: u64 = 0;
            let mut d2: u64 = 0;
            let mut d3: u64 = 0;
            let mut cc: u8 = _addcarryx_u64(0, a[0], b[0], &mut d0);
            cc = _addcarryx_u64(cc, a[1], b[1], &mut d1);
            cc = _addcarryx_u64(cc, a[2], b[2], &mut d2);
            _addcarryx_u64(cc, a[3], b[3], &mut d3);
            // First conditional subtraction of p.
            let f = d3 >> 59;
            let neg_f = (f as i64).wrapping_neg() as u64;
            cc = _addcarryx_u64(0, d0, f, &mut d0);
            cc = _addcarryx_u64(cc, d1, 0, &mut d1);
            cc = _addcarryx_u64(cc, d2, 0, &mut d2);
            _addcarryx_u64(cc, d3, NEG_P_TOP & neg_f, &mut d3);
            // Second conditional subtraction.
            let f = d3 >> 59;
            let neg_f = (f as i64).wrapping_neg() as u64;
            cc = _addcarryx_u64(0, d0, f, &mut d0);
            cc = _addcarryx_u64(cc, d1, 0, &mut d1);
            cc = _addcarryx_u64(cc, d2, 0, &mut d2);
            _addcarryx_u64(cc, d3, NEG_P_TOP & neg_f, &mut d3);
            [d0, d1, d2, d3]
        }
    }

    /// Field subtraction: a - b mod p, result in [0, 2^251).
    /// Port of `gf5248_sub` from `gf5248.h`.
    /// # Safety
    /// Requires x86_64 + BMI2 + ADX.
    pub unsafe fn sub(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
        unsafe {
            let mut d0: u64 = 0;
            let mut d1: u64 = 0;
            let mut d2: u64 = 0;
            let mut d3: u64 = 0;
            let mut cc: u8 = _subborrow_u64(0, a[0], b[0], &mut d0);
            cc = _subborrow_u64(cc, a[1], b[1], &mut d1);
            cc = _subborrow_u64(cc, a[2], b[2], &mut d2);
            cc = _subborrow_u64(cc, a[3], b[3], &mut d3);
            let mut m: u64 = 0;
            _subborrow_u64(cc, 0, 0, &mut m);
            cc = _subborrow_u64(0, d0, m & SUB_NEG2Q_LO, &mut d0);
            cc = _subborrow_u64(cc, d1, 0, &mut d1);
            cc = _subborrow_u64(cc, d2, 0, &mut d2);
            _subborrow_u64(cc, d3, SUB_NEG2Q_HI & m, &mut d3);
            let f = d3 >> 59;
            let neg_f = (f as i64).wrapping_neg() as u64;
            cc = _addcarryx_u64(0, d0, f, &mut d0);
            cc = _addcarryx_u64(cc, d1, 0, &mut d1);
            cc = _addcarryx_u64(cc, d2, 0, &mut d2);
            _addcarryx_u64(cc, d3, NEG_P_TOP & neg_f, &mut d3);
            [d0, d1, d2, d3]
        }
    }

    /// Montgomery reduction: 8-limb product -> 4-limb result in [0, 6*2^248).
    /// Port of the reduction block in `gf5248_mul` / `inner_gf5248_montgomery_reduce`.
    /// # Safety
    /// Requires x86_64 + BMI2 + ADX.
    pub unsafe fn redc(e: &[u64; 8]) -> [u64; 4] {
        unsafe {
            let f0 = e[0];
            let f1 = e[1];
            let f2 = e[2];
            let f3 = e[3].wrapping_add(e[0].wrapping_mul(5).wrapping_shl(56));

            let mut hi: u64 = 0;
            let g3_part = _mulx_u64(f0, P_PLUS_1_TOP, &mut hi);
            let g4_acc0 = hi;

            let mut hi2: u64 = 0;
            let f1_lo = _mulx_u64(f1, P_PLUS_1_TOP, &mut hi2);
            let mut g4: u64 = 0;
            let c1 = _addcarryx_u64(0, g4_acc0, f1_lo, &mut g4);
            let mut g5_acc0: u64 = 0;
            _addcarryx_u64(c1, hi2, 0, &mut g5_acc0);

            let mut hi3: u64 = 0;
            let f2_lo = _mulx_u64(f2, P_PLUS_1_TOP, &mut hi3);
            let mut g5: u64 = 0;
            let c2 = _addcarryx_u64(0, g5_acc0, f2_lo, &mut g5);
            let mut g6_acc0: u64 = 0;
            _addcarryx_u64(c2, hi3, 0, &mut g6_acc0);

            let mut hi4: u64 = 0;
            let f3_lo = _mulx_u64(f3, P_PLUS_1_TOP, &mut hi4);
            let mut g6: u64 = 0;
            let c3 = _addcarryx_u64(0, g6_acc0, f3_lo, &mut g6);
            let mut g7: u64 = 0;
            _addcarryx_u64(c3, hi4, 0, &mut g7);

            let mut g0: u64 = 0;
            let mut g1: u64 = 0;
            let mut g2: u64 = 0;
            let mut g3: u64 = 0;
            let mut g4f: u64 = 0;
            let mut g5f: u64 = 0;
            let mut g6f: u64 = 0;
            let mut g7f: u64 = 0;
            let mut bc: u8 = _subborrow_u64(0, 0, f0, &mut g0);
            bc = _subborrow_u64(bc, 0, f1, &mut g1);
            bc = _subborrow_u64(bc, 0, f2, &mut g2);
            bc = _subborrow_u64(bc, g3_part, f3, &mut g3);
            bc = _subborrow_u64(bc, g4, 0, &mut g4f);
            bc = _subborrow_u64(bc, g5, 0, &mut g5f);
            bc = _subborrow_u64(bc, g6, 0, &mut g6f);
            _subborrow_u64(bc, g7, 0, &mut g7f);

            let mut _sink: u64 = 0;
            let mut cc: u8 = _addcarryx_u64(0, g0, e[0], &mut _sink);
            cc = _addcarryx_u64(cc, g1, e[1], &mut _sink);
            cc = _addcarryx_u64(cc, g2, e[2], &mut _sink);
            cc = _addcarryx_u64(cc, g3, e[3], &mut _sink);
            let mut r0: u64 = 0;
            let mut r1: u64 = 0;
            let mut r2: u64 = 0;
            let mut r3: u64 = 0;
            cc = _addcarryx_u64(cc, g4f, e[4], &mut r0);
            cc = _addcarryx_u64(cc, g5f, e[5], &mut r1);
            cc = _addcarryx_u64(cc, g6f, e[6], &mut r2);
            _addcarryx_u64(cc, g7f, e[7], &mut r3);
            [r0, r1, r2, r3]
        }
    }

    unsafe fn mul_wide(a: &[u64; 4], b: &[u64; 4]) -> [u64; 8] {
        unsafe {
            let mut hi: u64 = 0;
            let e0 = _mulx_u64(a[0], b[0], &mut hi);
            let mut e1 = hi;
            let lo = _mulx_u64(a[1], b[1], &mut hi);
            let mut e2 = lo;
            let mut e3 = hi;
            let lo = _mulx_u64(a[2], b[2], &mut hi);
            let mut e4 = lo;
            let mut e5 = hi;
            let lo = _mulx_u64(a[3], b[3], &mut hi);
            let mut e6 = lo;
            let mut e7 = hi;

            let lo01 = _mulx_u64(a[0], b[1], &mut hi);
            let mut cc: u8 = _addcarryx_u64(0, e1, lo01, &mut e1);
            cc = _addcarryx_u64(cc, e2, hi, &mut e2);
            let lo03 = _mulx_u64(a[0], b[3], &mut hi);
            cc = _addcarryx_u64(cc, e3, lo03, &mut e3);
            cc = _addcarryx_u64(cc, e4, hi, &mut e4);
            let lo23 = _mulx_u64(a[2], b[3], &mut hi);
            cc = _addcarryx_u64(cc, e5, lo23, &mut e5);
            cc = _addcarryx_u64(cc, e6, hi, &mut e6);
            _addcarryx_u64(cc, e7, 0, &mut e7);

            let lo10 = _mulx_u64(a[1], b[0], &mut hi);
            cc = _addcarryx_u64(0, e1, lo10, &mut e1);
            cc = _addcarryx_u64(cc, e2, hi, &mut e2);
            let lo30 = _mulx_u64(a[3], b[0], &mut hi);
            cc = _addcarryx_u64(cc, e3, lo30, &mut e3);
            cc = _addcarryx_u64(cc, e4, hi, &mut e4);
            let lo32 = _mulx_u64(a[3], b[2], &mut hi);
            cc = _addcarryx_u64(cc, e5, lo32, &mut e5);
            cc = _addcarryx_u64(cc, e6, hi, &mut e6);
            _addcarryx_u64(cc, e7, 0, &mut e7);

            let lo02 = _mulx_u64(a[0], b[2], &mut hi);
            cc = _addcarryx_u64(0, e2, lo02, &mut e2);
            cc = _addcarryx_u64(cc, e3, hi, &mut e3);
            let lo13 = _mulx_u64(a[1], b[3], &mut hi);
            cc = _addcarryx_u64(cc, e4, lo13, &mut e4);
            cc = _addcarryx_u64(cc, e5, hi, &mut e5);
            cc = _addcarryx_u64(cc, e6, 0, &mut e6);
            _addcarryx_u64(cc, e7, 0, &mut e7);

            let lo20 = _mulx_u64(a[2], b[0], &mut hi);
            cc = _addcarryx_u64(0, e2, lo20, &mut e2);
            cc = _addcarryx_u64(cc, e3, hi, &mut e3);
            let lo31 = _mulx_u64(a[3], b[1], &mut hi);
            cc = _addcarryx_u64(cc, e4, lo31, &mut e4);
            cc = _addcarryx_u64(cc, e5, hi, &mut e5);
            cc = _addcarryx_u64(cc, e6, 0, &mut e6);
            _addcarryx_u64(cc, e7, 0, &mut e7);

            let lo12 = _mulx_u64(a[1], b[2], &mut hi);
            let hi12 = hi;
            let lo21 = _mulx_u64(a[2], b[1], &mut hi);
            let hi21 = hi;
            let mut lo_sum: u64 = 0;
            let mut hi_sum: u64 = 0;
            let mut carry_out: u64 = 0;
            let c12 = _addcarryx_u64(0, lo12, lo21, &mut lo_sum);
            let c_hi = _addcarryx_u64(c12, hi12, hi21, &mut hi_sum);
            _addcarryx_u64(c_hi, 0u64, 0u64, &mut carry_out);
            cc = _addcarryx_u64(0, e3, lo_sum, &mut e3);
            cc = _addcarryx_u64(cc, e4, hi_sum, &mut e4);
            cc = _addcarryx_u64(cc, e5, carry_out, &mut e5);
            cc = _addcarryx_u64(cc, e6, 0, &mut e6);
            _addcarryx_u64(cc, e7, 0, &mut e7);

            [e0, e1, e2, e3, e4, e5, e6, e7]
        }
    }

    /// Montgomery multiplication: a*b mod p, result in [0, 6*2^248).
    /// # Safety
    /// Requires x86_64 + BMI2 + ADX.
    pub unsafe fn mul(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
        unsafe {
            let e = mul_wide(a, b);
            redc(&e)
        }
    }

    /// Montgomery squaring: a^2 mod p, result in [0, 6*2^248).
    /// Delegates to mul(a,a).
    /// # Safety
    /// Requires x86_64 + BMI2 + ADX.
    pub unsafe fn square(a: &[u64; 4]) -> [u64; 4] {
        unsafe { mul(a, a) }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
pub use intrinsics::{add, mul, redc, square, sub};

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(all(
    test,
    target_arch = "x86_64",
    target_feature = "bmi2",
    target_feature = "adx"
))]
mod tests {
    use super::*;
    use crate::gf::fp::BaseField;
    use crate::params::lvl1::Fp1Element;
    use crypto_bigint::U256;
    use rand_chacha::ChaCha20Rng;
    use rand_core::{Rng, SeedableRng};

    fn limbs_to_fp(limbs: [u64; 4]) -> Fp1Element {
        Fp1Element::from_montgomery(U256::from_words(limbs))
    }

    fn fp_to_canonical(x: Fp1Element) -> [u64; 4] {
        // Compare in the SAME (Montgomery) domain the intrinsic backend uses:
        // take crypto-bigint's Montgomery-form limbs and run the identical
        // canonical reduction `to_canonical` applies to the intrinsic output.
        let m: U256 = *x.as_montgomery();
        to_canonical(&core::array::from_fn(|i| m.as_limbs()[i].0))
    }

    /// Differential correctness: 10 000 random pairs for mul/square/add/sub.
    /// Guards the byte-exact KAT achieved in session S351.
    #[test]
    fn differential_correctness_mul_square_add_sub() {
        let mut rng = ChaCha20Rng::seed_from_u64(0x1234_5678_90ab_cdef_u64);
        for iter in 0..10_000_usize {
            let mut a = [0u64; 4];
            let mut b = [0u64; 4];
            for x in a.iter_mut() {
                *x = rng.next_u64();
            }
            for x in b.iter_mut() {
                *x = rng.next_u64();
            }
            a[3] &= 0x04ff_ffff_ffff_ffff;
            b[3] &= 0x04ff_ffff_ffff_ffff;
            let fp_a = limbs_to_fp(a);
            let fp_b = limbs_to_fp(b);
            unsafe {
                let got = to_canonical(&mul(&a, &b));
                let exp = fp_to_canonical(fp_a.mul(&fp_b));
                assert_eq!(got, exp, "mul mismatch at iter {iter}: a={a:?} b={b:?}");
                let got = to_canonical(&square(&a));
                let exp = fp_to_canonical(fp_a.square());
                assert_eq!(got, exp, "square mismatch at iter {iter}: a={a:?}");
                let got = to_canonical(&add(&a, &b));
                let exp = fp_to_canonical(fp_a.add(&fp_b));
                assert_eq!(got, exp, "add mismatch at iter {iter}: a={a:?} b={b:?}");
                let got = to_canonical(&sub(&a, &b));
                let exp = fp_to_canonical(fp_a.sub(&fp_b));
                assert_eq!(got, exp, "sub mismatch at iter {iter}: a={a:?} b={b:?}");
            }
        }
    }

    /// Smoke: 1·1 = 1.
    #[test]
    fn mul_one_times_one() {
        let one = Fp1Element::one();
        let one_limbs: [u64; 4] = core::array::from_fn(|i| one.as_montgomery().as_limbs()[i].0);
        unsafe {
            let got = to_canonical(&mul(&one_limbs, &one_limbs));
            let exp = fp_to_canonical(one.mul(&one));
            assert_eq!(got, exp, "1*1 != 1");
        }
    }

    /// Smoke: a + 0 = a.
    #[test]
    fn add_zero_identity() {
        let zero = [0u64; 4];
        let a = [
            0x1234_5678_9abc_def0_u64,
            0xdead_beef_cafe_babe,
            0x0011_2233_4455_6677,
            0x0100_0000_0000_0000,
        ];
        unsafe {
            let got = to_canonical(&add(&a, &zero));
            let exp = fp_to_canonical(limbs_to_fp(a).add(&Fp1Element::zero()));
            assert_eq!(got, exp, "a + 0 != a");
        }
    }

    /// Smoke: a − a = 0.
    #[test]
    fn sub_self_is_zero() {
        let a = [
            0x1111_2222_3333_4444_u64,
            0x5555_6666_7777_8888,
            0x9999_aaaa_bbbb_cccc,
            0x0123_4567_89ab_cdef & 0x04ff_ffff_ffff_ffff,
        ];
        unsafe {
            let got = to_canonical(&sub(&a, &a));
            assert_eq!(got, [0u64; 4], "a - a != 0");
        }
    }
}
