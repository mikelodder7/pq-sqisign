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

/// The Level-1 base prime `p = 5 · 2^248 − 1` as a 256-bit unsigned integer.
///
/// Use `prime().resize::<N>()` (from `crypto_bigint::Uint::resize`) to embed
/// the prime into a wider-LIMBS context — e.g. `prime().resize::<8>()` for
/// quaternion arithmetic in `Uint<8>` (512-bit signed), where the prime sits
/// alongside intermediate products of magnitude up to roughly `p²`.
pub fn prime() -> U256 {
    *Lvl1Modulus::PARAMS.modulus().as_ref()
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
}
