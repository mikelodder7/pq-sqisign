// SPDX-License-Identifier: MIT OR Apache-2.0
//! Random sampling of `O_0` quaternions from a `CryptoRng`.
//!
//! KLPT's full algorithm relaxes the strict `target = N(I) · m²` constraint
//! of [`super::klpt::lift_to_smooth_norm`] by sampling fresh γ ∈ I (or
//! random α ∈ B*) until one with the desired norm shape lands. This
//! module provides the sampling primitives that retry loop consumes.

use crypto_bigint::Int;
use rand_core::CryptoRng;

/// Sample a uniformly random integer in `[-bound, bound]` using `rng`.
fn sample_int_in_range<R: CryptoRng>(rng: &mut R, bound: i64) -> i64 {
    debug_assert!(bound >= 0);
    if bound == 0 {
        return 0;
    }
    // Use rejection sampling on u64 to avoid modulo bias.
    #[allow(clippy::cast_sign_loss)] // bound is non-negative per debug_assert
    let bound_u = bound as u64;
    let span = bound_u.saturating_mul(2).saturating_add(1);
    let limit = u64::MAX - (u64::MAX % span);
    loop {
        let mut buf = [0u8; 8];
        rng.fill_bytes(&mut buf);
        let r = u64::from_le_bytes(buf);
        if r < limit {
            #[allow(clippy::cast_possible_wrap)] // r % span < 2·bound+1 ≤ i64::MAX
            let centred = (r % span) as i64 - bound;
            return centred;
        }
    }
}

/// Sample a random quaternion in `O_0` with each `O_0`-basis coordinate
/// drawn uniformly from `[-bound, bound]`. Bound must be non-negative.
///
/// For very small `bound` the sampled element may be `(0, 0, 0, 0)` (zero
/// quaternion); callers wanting non-zero should reject and resample.
pub fn sample_random_quaternion_o0<R: CryptoRng>(rng: &mut R, bound: i64) -> [Int<8>; 4] {
    [
        Int::<8>::from_i64(sample_int_in_range(rng, bound)),
        Int::<8>::from_i64(sample_int_in_range(rng, bound)),
        Int::<8>::from_i64(sample_int_in_range(rng, bound)),
        Int::<8>::from_i64(sample_int_in_range(rng, bound)),
    ]
}

/// Port of the C reference `ibz_rand_interval(rand, a, b)` — uniform integer
/// in `[a, b]`, works with any [`CryptoRng`].
#[cfg(feature = "alloc")]
pub(crate) fn ibz_rand_interval<const N: usize, R: CryptoRng + ?Sized>(
    rng: &mut R,
    a: &crypto_bigint::Uint<N>,
    b: &crypto_bigint::Uint<N>,
) -> crypto_bigint::Uint<N> {
    use crypto_bigint::Uint;
    let bmina = b.wrapping_sub(a);
    if bmina == Uint::<N>::ZERO {
        return *a;
    }
    let len_bits = bmina.bits_vartime();
    let len_bytes = len_bits.div_ceil(8) as usize;
    let total_bytes = N * 8;
    let mask = if (len_bits as usize) >= total_bytes * 8 {
        Uint::<N>::MAX
    } else {
        Uint::<N>::ONE
            .shl_vartime(len_bits)
            .wrapping_sub(&Uint::<N>::ONE)
    };
    let mut buf = alloc::vec::from_elem(0u8, total_bytes);
    loop {
        rng.fill_bytes(&mut buf[..len_bytes]);
        let cand = Uint::<N>::from_le_slice(&buf) & mask;
        if cand <= bmina {
            return cand.wrapping_add(a);
        }
    }
}

/// Port of the C reference `ibz_rand_interval_minm_m(rand, m)` — uniform
/// integer in `[−m, m]` via `ibz_rand_interval([0, 2m]) − m`.
#[cfg(feature = "alloc")]
pub(crate) fn ibz_rand_interval_minm_m<const N: usize, R: CryptoRng + ?Sized>(
    rng: &mut R,
    m: u32,
) -> Int<N> {
    use crypto_bigint::Uint;
    let two_m = Uint::<N>::from_u64(2 * u64::from(m));
    let r = ibz_rand_interval::<N, R>(rng, &Uint::<N>::ZERO, &two_m);
    let m_u = Uint::<N>::from_u64(u64::from(m));
    r.as_int().wrapping_sub(m_u.as_int())
}

// Sample tests use `NistPqcRng` for a deterministic, seedable CryptoRng;
// gated on `kat` so non-test builds drop the dependency.
#[cfg(all(test, feature = "kat"))]
mod tests {
    use super::*;
    use crate::rng::NistPqcRng;

    #[test]
    fn same_seed_same_output() {
        let seed = [0x42u8; 48];
        let mut rng_a = NistPqcRng::new(&seed);
        let mut rng_b = NistPqcRng::new(&seed);
        let a = sample_random_quaternion_o0(&mut rng_a, 100);
        let b = sample_random_quaternion_o0(&mut rng_b, 100);
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_diverge() {
        let mut rng_a = NistPqcRng::new(&[0x01u8; 48]);
        let mut rng_b = NistPqcRng::new(&[0x02u8; 48]);
        let a = sample_random_quaternion_o0(&mut rng_a, 100);
        let b = sample_random_quaternion_o0(&mut rng_b, 100);
        assert_ne!(a, b);
    }

    #[test]
    fn coords_within_bound() {
        let mut rng = NistPqcRng::new(&[0xabu8; 48]);
        let bound = 50;
        let bound_int = Int::<8>::from_i64(bound);
        let neg_bound_int = Int::<8>::from_i64(-bound);
        for _ in 0..32 {
            let q = sample_random_quaternion_o0(&mut rng, bound);
            for c in &q {
                // c ∈ [-bound, bound]: -bound ≤ c AND c ≤ bound.
                assert!(*c >= neg_bound_int);
                assert!(*c <= bound_int);
            }
        }
    }

    #[test]
    fn zero_bound_yields_zero() {
        let mut rng = NistPqcRng::new(&[0u8; 48]);
        let q = sample_random_quaternion_o0(&mut rng, 0);
        for c in &q {
            assert_eq!(*c, Int::<8>::from_i64(0));
        }
    }

    #[test]
    fn successive_calls_differ() {
        let mut rng = NistPqcRng::new(&[0x12u8; 48]);
        let a = sample_random_quaternion_o0(&mut rng, 1000);
        let b = sample_random_quaternion_o0(&mut rng, 1000);
        // Two consecutive draws from a CryptoRng should differ with
        // probability ~1 − 1/(2001)⁴.
        assert_ne!(a, b);
    }
}
