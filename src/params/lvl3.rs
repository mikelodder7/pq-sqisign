// SPDX-License-Identifier: MIT OR Apache-2.0
//! NIST Level-3 (192-bit classical security) parameter set.
//!
//! `p = 65 · 2^376 − 1`, a 383-bit prime. Six 64-bit limbs (`U384`).

use crypto_bigint::modular::ConstMontyParams;
use crypto_bigint::{U384, const_monty_params};

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
}
