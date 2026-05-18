// SPDX-License-Identifier: MIT OR Apache-2.0
//! NIST Level-1 (128-bit classical security) parameter set.
//!
//! `p = 5 · 2^248 − 1`, a 251-bit prime. The base field element fits in
//! four 64-bit limbs (`U256`); SQIsign-1 chose this prime so the 2-power
//! torsion `E_0[2^248]` is rational over `F_{p^2}`.

use crypto_bigint::modular::ConstMontyParams;
use crypto_bigint::{U256, const_monty_params};

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
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Level1;

impl Params for Level1 {
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
}
