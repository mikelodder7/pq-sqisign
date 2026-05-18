// SPDX-License-Identifier: MIT OR Apache-2.0
//! Smooth-target enumeration for KLPT.
//!
//! KLPT's outer loop tries equivalent-ideal targets of *smooth* norm
//! `T = ∏ ℓ_i^{e_i}` where every `ℓ_i` is in a fixed set of small primes.
//! Smaller targets fail; larger targets work but burn time on isogeny
//! chain steps. This module enumerates candidates in increasing order so
//! KLPT can walk them lazily.
//!
//! For SQIsign specifically the "natural" target shape is `2^e · odd` with
//! `e` chosen to match the level's available `F_{p^2}`-rational 2-power
//! torsion (`F` in the `c · 2^F − 1` parameterisation). The odd part is
//! drawn from a small set of available isogeny degrees.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// A smooth number represented as its prime-factorisation exponents alongside
/// the bound-determined prime set the caller supplied.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SmoothNumber {
    /// `exponents[i]` is the exponent on `primes[i]` in the factorisation.
    pub exponents: [u32; 8],
    /// Concrete value of the product (for the `Vec<u128>` ordering / sorting).
    pub value: u128,
}

impl SmoothNumber {
    /// Construct from an exponents vector and primes list. Computes `value`.
    pub fn from_exponents(exponents: [u32; 8], primes: &[u64]) -> Option<Self> {
        let mut value: u128 = 1;
        for (i, &e) in exponents.iter().enumerate() {
            if e == 0 || i >= primes.len() {
                continue;
            }
            let p = primes[i] as u128;
            for _ in 0..e {
                value = value.checked_mul(p)?;
            }
        }
        Some(Self { exponents, value })
    }
}

/// Enumerate every smooth product `∏ primes[i]^{exponents[i]} ≤ bound`,
/// returning them sorted ascending by value.
///
/// `primes` should contain at most 8 distinct primes (the `exponents` array
/// is statically sized). Repeated primes / duplicates are not de-duped.
/// The value `1` (all exponents zero) is included.
///
/// Algorithm: depth-first scan over the exponent tuple, pruning whenever
/// the running product exceeds `bound`. `O(N)` work for `N` candidates.
#[cfg(feature = "alloc")]
pub fn enumerate_smooth(primes: &[u64], bound: u128) -> Vec<SmoothNumber> {
    debug_assert!(primes.len() <= 8, "at most 8 primes supported");
    let mut out = Vec::new();
    let mut exponents = [0u32; 8];
    enumerate_dfs(primes, bound, 0, 1, &mut exponents, &mut out);
    out.sort_by_key(|s| s.value);
    out
}

#[cfg(feature = "alloc")]
fn enumerate_dfs(
    primes: &[u64],
    bound: u128,
    idx: usize,
    cur_value: u128,
    exponents: &mut [u32; 8],
    out: &mut Vec<SmoothNumber>,
) {
    if idx == primes.len() {
        out.push(SmoothNumber {
            exponents: *exponents,
            value: cur_value,
        });
        return;
    }
    let p = primes[idx] as u128;
    let mut v = cur_value;
    let mut e: u32 = 0;
    loop {
        exponents[idx] = e;
        enumerate_dfs(primes, bound, idx + 1, v, exponents, out);
        // try one more multiplication
        let Some(next) = v.checked_mul(p) else { break };
        if next > bound {
            break;
        }
        v = next;
        e += 1;
    }
    exponents[idx] = 0;
}

/// Pick the smallest smooth target `T ≥ floor`. Returns `None` if no such
/// target exists within `bound`. This is the KLPT outer-loop callback —
/// "give me the next candidate to try".
#[cfg(feature = "alloc")]
pub fn next_smooth_at_least(primes: &[u64], floor: u128, bound: u128) -> Option<SmoothNumber> {
    enumerate_smooth(primes, bound)
        .into_iter()
        .find(|s| s.value >= floor)
}

#[cfg(test)]
#[cfg(feature = "alloc")]
mod tests {
    use super::*;

    #[test]
    fn powers_of_two_up_to_64() {
        let s = enumerate_smooth(&[2], 64);
        let values: Vec<u128> = s.iter().map(|x| x.value).collect();
        assert_eq!(values, vec![1, 2, 4, 8, 16, 32, 64]);
    }

    #[test]
    fn three_smooth_up_to_36() {
        // 3-smooth (Hamming) numbers ≤ 36:
        // 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36.
        let s = enumerate_smooth(&[2, 3], 36);
        let values: Vec<u128> = s.iter().map(|x| x.value).collect();
        assert_eq!(
            values,
            vec![1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
        );
    }

    #[test]
    fn five_smooth_up_to_30() {
        // Regular numbers ≤ 30:
        // 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30.
        let s = enumerate_smooth(&[2, 3, 5], 30);
        let values: Vec<u128> = s.iter().map(|x| x.value).collect();
        assert_eq!(
            values,
            vec![
                1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30
            ]
        );
    }

    #[test]
    fn empty_prime_list() {
        let s = enumerate_smooth(&[], 100);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].value, 1);
    }

    #[test]
    fn next_smooth_at_least_basic() {
        // primes {2, 3}, floor=10, bound=100 → next is 12.
        let n = next_smooth_at_least(&[2, 3], 10, 100).expect("solution exists");
        assert_eq!(n.value, 12);
    }

    #[test]
    fn next_smooth_at_least_exhausted() {
        // primes {2}, floor=1000, bound=100 → None.
        assert!(next_smooth_at_least(&[2], 1000, 100).is_none());
    }

    #[test]
    fn next_smooth_at_least_exact() {
        // primes {2, 5}, floor=20, bound=50 → 20 is itself smooth.
        let n = next_smooth_at_least(&[2, 5], 20, 50).expect("20 is smooth");
        assert_eq!(n.value, 20);
    }

    #[test]
    fn from_exponents_round_trip() {
        // 2^3 · 3^2 = 72.
        let mut e = [0u32; 8];
        e[0] = 3;
        e[1] = 2;
        let s = SmoothNumber::from_exponents(e, &[2, 3]).expect("fits");
        assert_eq!(s.value, 72);
    }

    #[test]
    fn klpt_typical_target_shape() {
        // SQIsign-ish: target = 2^e · (small odd prime). Try {2, 3, 5, 7, 11}
        // up to 2^10 — should contain values like 2^9·3 = 1536? No, that's > 1024.
        // 2^9 = 512, 2^8 · 3 = 768. Both should appear.
        let s = enumerate_smooth(&[2, 3, 5, 7, 11], 1024);
        let values: std::collections::BTreeSet<u128> = s.iter().map(|x| x.value).collect();
        assert!(values.contains(&512));
        assert!(values.contains(&768));
        assert!(values.contains(&1024));
        assert!(!values.contains(&1025)); // 1025 = 5² · 41, not in our prime set
    }
}
