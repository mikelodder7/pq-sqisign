// SPDX-License-Identifier: MIT OR Apache-2.0
//! Miller-Rabin probabilistic primality test on `crypto_bigint::Uint<LIMBS>`.
//!
//! KLPT's `quat_lideal_prime_norm_reduced_equivalent` (per the SQIsign C
//! reference at `src/quaternion/ref/generic/lll/lll_applications.c`)
//! random-samples a quaternion coordinate vector `c ∈ [−k, k]^4` and
//! accepts it iff the resulting reduced norm `q(c)/N(I)` is a probable
//! prime. This module provides that acceptance primitive.
//!
//! # API shape
//!
//! [`is_probable_prime_with_witnesses`] takes the candidate `n` and a
//! slice of pre-chosen witnesses. Callers wanting **deterministic**
//! testing for small inputs supply fixed small primes (e.g. `{2, 3, 5,
//! 7, 11}` covers all `n < 3.2 × 10⁹` per Sorenson 2014). Callers
//! wanting **probabilistic** testing at real-prime scale supply
//! random witnesses drawn from a `CryptoRng` (caller's responsibility;
//! this module stays `no_std`-clean).
//!
//! For each witness `a`, the algorithm:
//!
//! 1. Write `n − 1 = d · 2^r` with `d` odd (done once, outside the
//!    per-witness loop).
//! 2. Compute `x = a^d mod n` (via [`super::sqrt_mod::pow_mod_uint`]).
//! 3. If `x ∈ {1, n − 1}`, witness `a` does not prove `n` composite.
//! 4. Otherwise, repeat `x ← x² mod n` up to `r − 1` times; if any
//!    intermediate `x` equals `n − 1`, witness `a` does not prove
//!    compositeness; otherwise it does.
//!
//! The function is variable-time in the exponent (acceptable: the
//! candidate `n` being tested is public in the protocol's threat
//! model — it is sampled, then tested, with the result observable).

use crypto_bigint::{NonZero, Uint};

use super::sqrt_mod::pow_mod_uint;

/// Odd small primes for the trial-division presieve (3..=251). 2 is omitted —
/// callers feed odd candidates, and an even composite still falls through to
/// the BPSW step, which rejects it correctly.
const SMALL_PRIMES: [u64; 53] = [
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
];

/// Fast primality test for the norm-finder hot path: a cheap small-prime
/// presieve followed by value-sized BPSW (`crypto_primes::is_prime`).
///
/// **Verdict-identical** to a full primality test (and thus RNG-stream- and
/// byte-exact-preserving):
/// - The presieve only rejects `n` divisible by a small prime — composites a
///   full test would also reject. The `n == p` guard preserves the (here
///   unreachable, since `n` is always ≫ 251) case where `n` IS a small prime.
/// - The width buckets resize `n` down to the smallest container that still
///   holds it losslessly (`n.bits() ≤ bucket·64`), and primality is a property
///   of the value, not the container — so the verdict is unchanged while the
///   modexp cost drops ~(LIMBS/bucket)².
pub fn is_prime_fast<const LIMBS: usize>(n: &Uint<LIMBS>) -> bool {
    // 1. Trial-division presieve — O(LIMBS) per small prime, ~1000× cheaper
    //    than BPSW, rejecting ~80% of composites before any modexp.
    for &p in &SMALL_PRIMES {
        let pn = NonZero::new(Uint::<LIMBS>::from_u64(p)).expect("small prime is nonzero");
        if n.rem_vartime(&pn) == Uint::<LIMBS>::ZERO {
            return *n == Uint::<LIMBS>::from_u64(p);
        }
    }
    // 2. Value-sized BPSW. Buckets cover the structural max (t < ~1280 bits);
    //    fall back to the full width if a caller ever exceeds 1440 bits.
    let bits = n.bits_vartime();
    if bits <= 480 {
        crypto_primes::is_prime(crypto_primes::Flavor::Any, &n.resize::<8>())
    } else if bits <= 960 {
        crypto_primes::is_prime(crypto_primes::Flavor::Any, &n.resize::<16>())
    } else if bits <= 1440 {
        crypto_primes::is_prime(crypto_primes::Flavor::Any, &n.resize::<24>())
    } else {
        crypto_primes::is_prime(crypto_primes::Flavor::Any, n)
    }
}

/// Test whether witness `a` proves `n` composite. Returns `true` if
/// the witness DOES NOT prove compositeness (i.e. `n` could still be
/// prime); returns `false` if `a` is a Miller-Rabin compositeness
/// witness for `n`.
///
/// Caller pre-computes `d` and `r` such that `n − 1 = d · 2^r` with
/// `d` odd, and supplies `n` wrapped as `NonZero<Uint<LIMBS>>`.
/// Assumes `n` is odd, `n > 3`, `2 ≤ a ≤ n − 2`.
fn miller_rabin_witness_consistent<const LIMBS: usize>(
    n: &Uint<LIMBS>,
    d: &Uint<LIMBS>,
    r: u32,
    a: &Uint<LIMBS>,
    n_nz: &NonZero<Uint<LIMBS>>,
) -> bool {
    let one = Uint::<LIMBS>::ONE;
    let n_minus_one = n.wrapping_sub(&one);

    let mut x = pow_mod_uint(a, d, n_nz);
    if x == one || x == n_minus_one {
        return true;
    }
    for _ in 0..r.saturating_sub(1) {
        x = x.square_mod_vartime(n_nz);
        if x == n_minus_one {
            return true;
        }
        if x == one {
            // Non-trivial sqrt of 1 — proves compositeness.
            return false;
        }
    }
    false
}

/// Miller-Rabin probable-primality test on `n` using caller-supplied
/// witnesses. Returns `true` if `n` is probably prime (no witness
/// proved compositeness); returns `false` if any witness proved
/// compositeness, or for trivially-not-prime inputs (0, 1, even
/// composites > 2).
///
/// Witnesses outside `[2, n − 2]` are silently skipped (boundary
/// witnesses give no information). With zero effective witnesses
/// (e.g. all supplied are out of range, or the slice is empty) the
/// function returns `true` on odd `n ≥ 5` — caller's responsibility
/// to supply enough witnesses.
///
/// For deterministic testing of small inputs, pass the fixed-witness
/// sets from Sorenson 2014:
/// - `n < 2_047`: `{2}`
/// - `n < 1_373_653`: `{2, 3}`
/// - `n < 3_215_031_751`: `{2, 3, 5, 7}`
/// - `n < 3.18 × 10²³`: `{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}`
///
/// At real-prime SQIsign scale, callers should sample witnesses from
/// a `CryptoRng`. The SQIsign C reference uses `primality_num_iter = 30`
/// for L1 (see `src/precomp/ref/lvl1/quaternion_data.c`).
pub fn is_probable_prime_with_witnesses<const LIMBS: usize>(
    n: &Uint<LIMBS>,
    witnesses: &[Uint<LIMBS>],
) -> bool {
    let one = Uint::<LIMBS>::ONE;
    let two = Uint::<LIMBS>::from_u64(2);
    let three = Uint::<LIMBS>::from_u64(3);

    if *n < two {
        return false;
    }
    if *n == two || *n == three {
        return true;
    }
    if n.as_limbs()[0].0 & 1 == 0 {
        return false;
    }

    let n_minus_one = n.wrapping_sub(&one);
    let mut d = n_minus_one;
    let mut r: u32 = 0;
    while d.as_limbs()[0].0 & 1 == 0 {
        d = d.shr_vartime(1);
        r += 1;
    }

    let n_nz = match Option::<NonZero<_>>::from(NonZero::new(*n)) {
        Some(nz) => nz,
        None => return false,
    };

    for a in witnesses {
        if *a < two {
            continue;
        }
        if *a >= n_minus_one {
            continue;
        }
        if !miller_rabin_witness_consistent(n, &d, r, a, &n_nz) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::Uint;

    fn u(x: u64) -> Uint<8> {
        Uint::<8>::from_u64(x)
    }

    #[test]
    fn miller_rabin_handles_edge_cases() {
        let w = [u(2)];
        assert!(!is_probable_prime_with_witnesses(&u(0), &w));
        assert!(!is_probable_prime_with_witnesses(&u(1), &w));
        assert!(is_probable_prime_with_witnesses(&u(2), &w));
        assert!(is_probable_prime_with_witnesses(&u(3), &w));
        assert!(!is_probable_prime_with_witnesses(&u(4), &w));
    }

    #[test]
    fn miller_rabin_identifies_small_primes() {
        let w = [u(2), u(3), u(5)];
        for &p in &[5u64, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53] {
            assert!(
                is_probable_prime_with_witnesses(&u(p), &w),
                "should be prime: {p}"
            );
        }
    }

    #[test]
    fn miller_rabin_rejects_small_composites() {
        let w = [u(2), u(3), u(5)];
        for &n in &[4u64, 6, 8, 9, 10, 15, 21, 25, 27, 33, 35, 39, 49, 51, 55] {
            assert!(
                !is_probable_prime_with_witnesses(&u(n), &w),
                "should be composite: {n}"
            );
        }
    }

    #[test]
    fn miller_rabin_rejects_carmichael_561() {
        // 561 = 3·11·17 — smallest Carmichael number. Fermat's Little
        // Theorem alone fails on Carmichaels; Miller-Rabin catches them.
        let w = [u(2)];
        assert!(!is_probable_prime_with_witnesses(&u(561), &w));
    }

    #[test]
    fn miller_rabin_rejects_carmichael_41041() {
        // 41041 = 7·11·13·41 — a larger Carmichael.
        let w = [u(2), u(3)];
        assert!(!is_probable_prime_with_witnesses(&u(41041), &w));
    }

    #[test]
    fn miller_rabin_handles_n_equals_5() {
        // n = 5: n - 1 = 4 = 1 · 2². So d = 1, r = 2.
        // Witness a = 2: a^d mod n = 2; 2 != 1, 2 != 4. Loop r-1 = 1
        // iteration: x = 4 = n - 1, ok. So 5 passes. Verify edge case.
        let w = [u(2), u(3)];
        assert!(is_probable_prime_with_witnesses(&u(5), &w));
    }

    #[test]
    fn miller_rabin_empty_witnesses_returns_true_on_odd_n() {
        // No witnesses → no compositeness proof → vacuously "probably prime".
        // This is the documented behaviour; caller's responsibility to
        // supply witnesses. Test pins it so a future refactor doesn't
        // silently change to "no witnesses ⇒ false".
        let w: [Uint<8>; 0] = [];
        assert!(is_probable_prime_with_witnesses(&u(9), &w));
    }

    #[test]
    fn miller_rabin_accepts_real_lvl1_prime() {
        // p_1 = 5 · 2^248 − 1, the SQIsign Level-1 prime. With small
        // witnesses {2, 3, 5, 7, 11, 13} Miller-Rabin should accept it.
        use crate::params::lvl1;
        let p_wide: Uint<8> = lvl1::prime().resize::<8>();
        let w = [u(2), u(3), u(5), u(7), u(11), u(13)];
        assert!(
            is_probable_prime_with_witnesses(&p_wide, &w),
            "Level 1 prime should test prime under deterministic witnesses"
        );
    }
}
