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
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Level5;

impl Params for Level5 {
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
}
