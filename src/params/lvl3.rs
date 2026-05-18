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
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
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
}
