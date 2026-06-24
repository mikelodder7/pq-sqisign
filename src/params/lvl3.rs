// SPDX-License-Identifier: MIT OR Apache-2.0
//! NIST Level-3 (192-bit classical security) parameter set.
//!
//! `p = 65 · 2^376 − 1`, a 383-bit prime. Six 64-bit limbs (`U384`).

use crypto_bigint::modular::ConstMontyParams;
use crypto_bigint::{U384, Uint, const_monty_params};

use super::Params;

const_monty_params!(
    Lvl3Modulus,
    U384,
    "40ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    "Level-3 modulus p = 65 * 2^376 - 1"
);

/// Field element of `F_p` for Level-3 (Montgomery form).
pub type Fp3Element = crypto_bigint::modular::ConstMontyForm<Lvl3Modulus, { Lvl3Modulus::LIMBS }>;

/// Marker type implementing [`Params`] at NIST Level 3.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Level3;

impl Params for Level3 {
    const LEVEL: u8 = 3;
    const P_BITS: usize = 383;
    const C: u64 = 65;
    const F: usize = 376;
    const FP_BYTES: usize = 48;
    const FP2_BYTES: usize = 96;
    const PK_BYTES: usize = 97;
    const SK_BYTES: usize = 529;
    const SIG_BYTES: usize = 224;
    const RESPONSE_BITS: usize = 192;
    const HASH_ITERATIONS: usize = 256;
    const SECURITY_BITS: usize = 192;
    const FINDUV_BOX_SIZE: i64 = 3;
    // L3 NUM_ALTERNATE = 7 — confirmed via verbatim quote from
    // src/precomp/ref/lvl3/include/quaternion_data.h:4. Cross-check at
    // line 11 of the same header: CONNECTING_IDEALS[8] (= NUM + 1).
    const NUM_ALTERNATE_EXTREMAL_ORDERS: usize = 7;
    const FINDUV_CUBE_SIZE: u64 = 2400;
}

/// The Level-3 base prime `p = 65 · 2^376 − 1` as a 384-bit unsigned integer.
///
/// Use `prime().resize::<N>()` to embed in a wider-LIMBS context (e.g.
/// `prime().resize::<8>()` for quaternion arithmetic in `Uint<8>`).
pub fn prime() -> U384 {
    *Lvl3Modulus::PARAMS.modulus().as_ref()
}

/// The Level-3 secret-key isogeny degree `SEC_DEGREE = 2^768 + 183` (769-bit,
/// odd). Mirrors lvl1's `2^512 + 75`. Source: SQIsign C reference
/// `src/precomp/ref/lvl3/torsion_constants.c` — 16-bit-limb array (size 49)
/// `{0xb7, 0, …, 0, 0x1}`: low limb `0xb7 = 183`, top limb 48 contributes
/// `2^(48·16) = 2^768`. Equals `COM_DEGREE` at this level. Returned at
/// `Uint<24>` (1536-bit) to give the lvl3 quaternion wide-norm path headroom;
/// callers `resize` as needed.
pub fn sec_degree() -> Uint<24> {
    Uint::<24>::ONE
        .shl_vartime(768)
        .wrapping_add(&Uint::<24>::from_u64(183))
}

/// The Level-3 commitment isogeny degree `COM_DEGREE = 2^768 + 183`. At every
/// SQIsign level `COM_DEGREE == SEC_DEGREE`; the array in C-ref
/// `torsion_constants.c` is byte-identical to `SEC_DEGREE` at lvl3.
pub fn com_degree() -> Uint<24> {
    sec_degree()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prime_has_expected_shape() {
        let p = Lvl3Modulus::PARAMS.modulus();
        let p_bytes = p.as_ref().to_be_bytes();
        // Top byte is 0x40 (65 << 376 = 0x41 << 376; minus 1 produces 0x40_ff..ff).
        assert_eq!(p_bytes[0], 0x40);
        for &b in &p_bytes[1..] {
            assert_eq!(b, 0xff);
        }
    }

    #[test]
    fn prime_helper_returns_canonical_p() {
        // `prime()` must agree byte-for-byte with the hex literal in the
        // `const_monty_params!` invocation.
        let p = prime();
        let p_bytes = p.to_be_bytes();
        assert_eq!(p_bytes[0], 0x40);
        for &b in &p_bytes[1..] {
            assert_eq!(b, 0xff);
        }
    }

    #[test]
    fn prime_resizes_to_uint8_zero_extending() {
        // L3 prime is 6 limbs; resize to 8 limbs zero-extends the upper 2.
        use crypto_bigint::Uint;
        let p = prime();
        let p_wide: Uint<8> = p.resize::<8>();
        let wide_limbs = p_wide.as_limbs();
        let narrow_limbs = p.as_limbs();
        for (i, (w, n)) in wide_limbs.iter().zip(narrow_limbs.iter()).enumerate() {
            assert_eq!(w.0, n.0, "limb {i} preserved");
        }
        for (i, w) in wide_limbs.iter().enumerate().skip(6) {
            assert_eq!(w.0, 0, "upper limb {i} zero-extended");
        }
    }

    #[test]
    fn sec_and_com_degree_match_c_reference() {
        // C-ref src/precomp/ref/lvl3/torsion_constants.c: SEC_DEGREE and
        // COM_DEGREE are byte-identical; 16-bit-limb array {0xb7, 0,…,0, 0x1}
        // (size 49) ⇒ 183 + 2^768.
        use crypto_bigint::Uint;
        let sec = sec_degree();
        let expected = Uint::<24>::ONE
            .shl_vartime(768)
            .wrapping_add(&Uint::<24>::from_u64(183));
        assert_eq!(sec, expected, "SEC_DEGREE must equal 2^768 + 183");
        assert_eq!(sec.bits_vartime(), 769, "SEC_DEGREE is 769 bits");
        assert_eq!(sec.as_limbs()[0].0 & 1, 1, "SEC_DEGREE is odd");
        // Low 64-bit limb is 183; limb 12 (= 768/64) is 1; all others zero.
        let limbs = sec.as_limbs();
        assert_eq!(limbs[0].0, 183);
        assert_eq!(limbs[12].0, 0x1);
        for (i, l) in limbs.iter().enumerate() {
            if i != 0 && i != 12 {
                assert_eq!(l.0, 0, "SEC_DEGREE limb {i} must be zero");
            }
        }
        assert_eq!(com_degree(), sec, "COM_DEGREE == SEC_DEGREE at lvl3");
    }

    #[test]
    fn montgomery_repr_matches_c_broadwell() {
        // Foundational compatibility anchor for ALL lvl3 precomputed constants.
        //
        // The C reference stores field elements in Montgomery form with
        // R = 2^384 (its 6-limb BROADWELL representation at lvl3). Our
        // `Fp3Element = ConstMontyForm<Lvl3Modulus, 6>` also uses R = 2^(64·6) =
        // 2^384. C-ref `endomorphism_action.c` CURVES_WITH_ENDOMORPHISMS[0]
        // stores the Montgomery "1" (= R mod p) in BROADWELL limbs as
        // {0x3, 0, 0, 0, 0, 0x3d00000000000000}. If plugging those limbs in via
        // `from_montgomery` yields the canonical field element 1, the reference's
        // stored limbs ARE our internal representation and every lvl3 constant can
        // be transcribed verbatim (mirrors the lvl1 anchor {0x33,0,0,0x01<<56}).
        let c_mont_one = Fp3Element::from_montgomery(U384::from_words([
            0x3,
            0x0,
            0x0,
            0x0,
            0x0,
            0x3d00_0000_0000_0000,
        ]));
        assert_eq!(
            c_mont_one,
            Fp3Element::new(&U384::ONE),
            "C BROADWELL Montgomery-1 must equal our canonical 1 (R = 2^384 match)"
        );
    }
}
