// SPDX-License-Identifier: MIT OR Apache-2.0
//! NIST Level-1 (128-bit classical security) parameter set.
//!
//! `p = 5 · 2^248 − 1`, a 251-bit prime. The base field element fits in
//! four 64-bit limbs (`U256`); SQIsign-1 chose this prime so the 2-power
//! torsion `E_0[2^248]` is rational over `F_{p^2}`.

use crypto_bigint::modular::ConstMontyParams;
use crypto_bigint::{U256, Uint, const_monty_params};

use super::Params;

const_monty_params!(
    Lvl1Modulus,
    U256,
    "04ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    "Level-1 modulus p = 5 * 2^248 - 1"
);

/// Field element of `F_p` for Level-1 (Montgomery form).
pub type Fp1Element = crypto_bigint::modular::ConstMontyForm<Lvl1Modulus, { Lvl1Modulus::LIMBS }>;

/// Marker type implementing [`Params`] at NIST Level 1.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Level1;

impl Params for Level1 {
    type Field = Fp1Element;
    const LEVEL: u8 = 1;
    const P_BITS: usize = 251;
    const C: u64 = 5;
    const F: usize = 248;
    const FP_BYTES: usize = 32;
    const FP2_BYTES: usize = 64;
    const PK_BYTES: usize = 65;
    const SK_BYTES: usize = 353;
    const SIG_BYTES: usize = 148;
    const RESPONSE_BITS: usize = 126;
    const HASH_ITERATIONS: usize = 64;
    const SECURITY_BITS: usize = 128;
    const FINDUV_BOX_SIZE: i64 = 2;
    const NUM_ALTERNATE_EXTREMAL_ORDERS: usize = 6;
    const FINDUV_CUBE_SIZE: u64 = 624;
}

/// The Level-1 base prime `p = 5 · 2^248 − 1` as a 256-bit unsigned integer.
///
/// Use `prime().resize::<N>()` (from `crypto_bigint::Uint::resize`) to embed
/// the prime into a wider-LIMBS context — e.g. `prime().resize::<8>()` for
/// quaternion arithmetic in `Uint<8>` (512-bit signed), where the prime sits
/// alongside intermediate products of magnitude up to roughly `p²`.
pub fn prime() -> U256 {
    *Lvl1Modulus::PARAMS.modulus().as_ref()
}

/// The Level-1 secret-key isogeny degree `SEC_DEGREE = 2^512 + 75` (513-bit,
/// odd). The keygen secret ideal is sampled as an `O_0`-ideal of this norm
/// (then reduced to a prime-norm equivalent) and fed to the Clapotis
/// `ideal_to_isogeny` to obtain the public-key curve `E_A`. Equals
/// `COM_DEGREE` at this level. Source: SQIsign C reference
/// `src/precomp/ref/lvl1/torsion_constants.c` (`SEC_DEGREE`, 64-bit limbs
/// `{0x4b, 0, …, 0, 0x1}` = `2^512 + 75`). Returned at `Uint<16>` (1024-bit)
/// to match the quaternion wide-norm width used by the ideal/Clapotis path.
pub fn sec_degree() -> Uint<16> {
    Uint::<16>::ONE
        .shl_vartime(512)
        .wrapping_add(&Uint::<16>::from_u64(75))
}

/// The Level-1 commitment isogeny degree `COM_DEGREE = 2^512 + 75`. At every
/// SQIsign level `COM_DEGREE == SEC_DEGREE`; the commitment ideal (in `sign`)
/// is sampled at this norm. Source: C-ref `torsion_constants.c` (`COM_DEGREE`,
/// byte-identical to `SEC_DEGREE` at lvl1).
pub fn com_degree() -> Uint<16> {
    sec_degree()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prime_has_expected_shape() {
        // p + 1 == 5 << 248
        let p = Lvl1Modulus::PARAMS.modulus();
        let p_bytes = p.as_ref().to_be_bytes();
        // Top byte is 0x04, then 0xff repeated.
        assert_eq!(p_bytes[0], 0x04);
        for &b in &p_bytes[1..] {
            assert_eq!(b, 0xff);
        }
    }

    #[test]
    fn modulus_is_odd() {
        let p = Lvl1Modulus::PARAMS.modulus().as_ref();
        let limb0: crypto_bigint::Limb = p.as_limbs()[0];
        // Lowest bit of p is 1 (p ≡ 3 mod 4 implies p odd).
        assert_eq!(limb0.0 & 1, 1);
    }

    #[test]
    fn prime_helper_returns_canonical_p() {
        let p = prime();
        // The `prime()` helper is a stable wrapper around the Const-Monty
        // modulus surface. It must agree byte-for-byte with the hex literal
        // in the `const_monty_params!` invocation.
        let p_bytes = p.to_be_bytes();
        assert_eq!(p_bytes[0], 0x04);
        for &b in &p_bytes[1..] {
            assert_eq!(b, 0xff);
        }
        // And `p + 1 == 5 · 2^248`: every limb except limb 3 is zero, limb 3
        // carries the value `5 · 2^(248 - 192) = 5 · 2^56`.
        let p_plus_one = p.wrapping_add(&U256::ONE);
        let limbs = p_plus_one.as_limbs();
        assert_eq!(limbs[0].0, 0);
        assert_eq!(limbs[1].0, 0);
        assert_eq!(limbs[2].0, 0);
        assert_eq!(limbs[3].0, 5u64 << 56);
    }

    #[test]
    fn prime_resizes_to_uint8_zero_extending() {
        // `Uint::resize::<8>()` zero-extends to 512-bit. The lower 4 limbs
        // must equal `prime()`'s 4 limbs; the upper 4 must all be zero.
        use crypto_bigint::Uint;
        let p: U256 = prime();
        let p_wide: Uint<8> = p.resize::<8>();
        let wide_limbs = p_wide.as_limbs();
        let narrow_limbs = p.as_limbs();
        for (i, (w, n)) in wide_limbs.iter().zip(narrow_limbs.iter()).enumerate() {
            assert_eq!(w.0, n.0, "limb {i} preserved");
        }
        for (i, w) in wide_limbs.iter().enumerate().skip(4) {
            assert_eq!(w.0, 0, "upper limb {i} zero-extended");
        }
    }

    #[test]
    fn sec_and_com_degree_match_c_reference() {
        // C-ref src/precomp/ref/lvl1/torsion_constants.c: SEC_DEGREE and
        // COM_DEGREE are byte-identical, 64-bit limbs {0x4b, 0,…,0, 0x1} =
        // 2^512 + 75 (513-bit, odd).
        let sec = sec_degree();
        // Value: 2^512 + 75 exactly.
        let expected = Uint::<16>::ONE
            .shl_vartime(512)
            .wrapping_add(&Uint::<16>::from_u64(75));
        assert_eq!(sec, expected, "SEC_DEGREE must equal 2^512 + 75");
        // 513-bit and odd.
        assert_eq!(sec.bits_vartime(), 513, "SEC_DEGREE is 513 bits");
        assert_eq!(sec.as_limbs()[0].0 & 1, 1, "SEC_DEGREE is odd");
        // Low limb is 0x4b (75), limb 8 is 0x1, all others zero.
        let limbs = sec.as_limbs();
        assert_eq!(limbs[0].0, 0x4b);
        assert_eq!(limbs[8].0, 0x1);
        for (i, l) in limbs.iter().enumerate() {
            if i != 0 && i != 8 {
                assert_eq!(l.0, 0, "SEC_DEGREE limb {i} must be zero");
            }
        }
        // COM_DEGREE == SEC_DEGREE at lvl1.
        assert_eq!(com_degree(), sec, "COM_DEGREE == SEC_DEGREE at lvl1");
    }
}
