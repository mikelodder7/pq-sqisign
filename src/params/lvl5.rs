// SPDX-License-Identifier: MIT OR Apache-2.0
//! NIST Level-5 (256-bit classical security) parameter set.
//!
//! `p = 27 · 2^500 − 1`, a 505-bit prime. Eight 64-bit limbs (`U512`).

use crypto_bigint::modular::ConstMontyParams;
use crypto_bigint::{U512, const_monty_params};

use super::Params;

const_monty_params!(
    Lvl5Modulus,
    U512,
    "01afffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    "Level-5 modulus p = 27 * 2^500 - 1"
);

/// Field element of `F_p` for Level-5 (Montgomery form).
pub type Fp5Element = crypto_bigint::modular::ConstMontyForm<Lvl5Modulus, { Lvl5Modulus::LIMBS }>;

/// Marker type implementing [`Params`] at NIST Level 5.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Level5;

impl Params for Level5 {
    type Field = Fp5Element;
    const LEVEL: u8 = 5;
    const P_BITS: usize = 505;
    const C: u64 = 27;
    const F: usize = 500;
    const FP_BYTES: usize = 64;
    const FP2_BYTES: usize = 128;
    const PK_BYTES: usize = 129;
    const SK_BYTES: usize = 701;
    const SIG_BYTES: usize = 292;
    const RESPONSE_BITS: usize = 253;
    const HASH_ITERATIONS: usize = 512;
    const SECURITY_BITS: usize = 256;
    const FINDUV_BOX_SIZE: i64 = 3;
    const NUM_ALTERNATE_EXTREMAL_ORDERS: usize = 6;
    const FINDUV_CUBE_SIZE: u64 = 2400;
}

/// The Level-5 base prime `p = 27 · 2^500 − 1` as a 512-bit unsigned integer.
///
/// L5 is the magnitude ceiling for quaternion arithmetic on `Int<8>`
/// (signed 512-bit): `p < 2^505` so `p.as_int()` is positive and `−p` is
/// `~2^505`, both ~6 bits inside the `Int<8>` envelope.
pub fn prime() -> U512 {
    *Lvl5Modulus::PARAMS.modulus().as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prime_has_expected_shape() {
        let p = Lvl5Modulus::PARAMS.modulus();
        let p_bytes = p.as_ref().to_be_bytes();
        // Top byte 0x01, next byte 0xaf, then 0xff repeated to fill 512 bits.
        assert_eq!(p_bytes[0], 0x01);
        assert_eq!(p_bytes[1], 0xaf);
        for &b in &p_bytes[2..] {
            assert_eq!(b, 0xff);
        }
    }

    #[test]
    fn prime_helper_returns_canonical_p() {
        let p = prime();
        let p_bytes = p.to_be_bytes();
        assert_eq!(p_bytes[0], 0x01);
        assert_eq!(p_bytes[1], 0xaf);
        for &b in &p_bytes[2..] {
            assert_eq!(b, 0xff);
        }
    }

    #[test]
    fn prime_top_bit_clear_for_int8_sign_room() {
        // L5 is the magnitude stress case: `Int<8>` (signed 512-bit) needs
        // the top bit free to represent `−p` without sign-bit collision.
        // `p ~ 2^505`, so bits 511..505 must be zero — pin that explicitly.
        let p = prime();
        let limbs = p.as_limbs();
        // Top limb is limbs[7] (most significant). Bits 505..511 sit in
        // bits 41..47 of that limb (since limb 7 covers bits 448..511).
        // p_top_limb = 0x01af_ffff_ffff_ffff: bits 56..62 = 0, bit 56 = 1
        // (the 0x01 nibble) — wait, the highest bit of p is at position 504.
        // Verify: limb 7 covers bits [448, 511]. Bit 504 sits at position
        // 504 − 448 = 56 within limb 7. Top byte of limb 7 (bits 56..63)
        // should be 0x01, so bits 57..63 are zero.
        let top_limb = limbs[7].0;
        assert_eq!(
            top_limb >> 57,
            0,
            "Int<8> sign-room check: bits 57..63 of top limb must be zero"
        );
        assert_eq!(
            (top_limb >> 56) & 1,
            1,
            "bit 504 of p must be set (p_top_limb top byte = 0x01)"
        );
    }
}
