// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cornacchia's algorithm — find `(x, y)` with `x² + d · y² = m`.
//!
//! KLPT's inner loop repeatedly asks "does `M = x² + d y²` have an
//! integer solution?" for various `M` along the equivalent-ideal lift.
//! Cornacchia is the standard answer.
//!
//! This module ships the trial-iteration implementation: `O(√m)` and
//! straightforward to verify by inspection. For the KLPT-production
//! path the Euclidean variant (Tonelli-Shanks for `t² ≡ −d mod m`,
//! followed by a single Euclidean reduction) replaces this in a later
//! session — the public surface stays unchanged.

use crate::quaternion::sqrt_mod::tonelli_shanks;

/// Classical Cornacchia: solve `x² + d · y² = p` where `p` is **prime**.
///
/// Algorithm (Cornacchia, 1908; see Cohen, *A Course in Computational
/// Algebraic Number Theory*, §1.5.2):
///
/// 1. Compute `t` with `t² ≡ −d (mod p)` via [`tonelli_shanks`]. If no such
///    `t` exists (i.e. `−d` is a QNR mod `p`), return `None`.
/// 2. Normalise `t` to the upper half-interval `t ≥ p / 2`.
/// 3. Run the Euclidean ladder `(p, t) → (t, p mod t) → …` until the smaller
///    remainder satisfies `r² ≤ p`.
/// 4. Check whether `(p − r²) / d` is a non-negative perfect square `s²`.
///    If yes, return `(r, s)`; else return `None`.
///
/// Runs in `O((log p)²)` versus trial's `O(√p)` — what KLPT's inner loop
/// actually needs once primes grow beyond ~2²⁰.
///
/// `p` must be an odd prime less than `2⁶³` for the `u128` intermediates to
/// stay exact. The `crypto_bigint::Int<LIMBS>` version arrives with the
/// KLPT-on-real-primes session.
pub fn cornacchia_classical(d: u64, p: u64) -> Option<(u64, u64)> {
    if p == 0 {
        return Some((0, 0));
    }
    if p == 1 {
        return None;
    }
    // d must be co-prime to p for the Tonelli-Shanks step to make sense.
    if d % p == 0 {
        // x² = p has no non-trivial solution unless p is a perfect square,
        // which is false for prime p > 1.
        return None;
    }
    // Find t with t² ≡ −d (mod p).
    let neg_d = p - (d % p);
    let mut t = tonelli_shanks(neg_d, p)?;
    if t < p / 2 {
        t = p - t;
    }
    // Euclidean ladder: (a, b) ← (p, t), then iterate (b, a mod b) until
    // b² ≤ p.
    let mut a: u64 = p;
    let mut b: u64 = t;
    while (b as u128) * (b as u128) > p as u128 {
        let r = a % b;
        a = b;
        b = r;
    }
    // Now b is the candidate x; check (p − b²) / d is a perfect square.
    let b2 = (b as u128) * (b as u128);
    let r = p as u128 - b2;
    if r % (d as u128) != 0 {
        return None;
    }
    let s2 = r / d as u128;
    if s2 > u128::from(u64::MAX) {
        return None;
    }
    #[allow(clippy::cast_possible_truncation)] // bounded by the check above
    let s2_u64 = s2 as u64;
    let s = s2_u64.isqrt();
    if (s as u128) * (s as u128) == s2 {
        return Some((b, s));
    }
    None
}

/// Solve `x² + d · y² = m` over the non-negative integers, returning the
/// lexicographically-smallest `(x, y)` pair, or `None` if no solution
/// exists.
///
/// Trivial cases:
/// - `m == 0` → `(0, 0)`.
/// - `d == 0` → `(√m, 0)` if `m` is a perfect square, else `None`.
///
/// Otherwise iterate `x` from `0` to `⌊√m⌋`, checking whether
/// `(m − x²) / d` is a non-negative perfect square. Returns the first hit.
pub fn cornacchia(d: u128, m: u128) -> Option<(u128, u128)> {
    if m == 0 {
        return Some((0, 0));
    }
    if d == 0 {
        let s = m.isqrt();
        if s.checked_mul(s)? == m {
            return Some((s, 0));
        }
        return None;
    }
    let sqrt_m = m.isqrt();
    let mut x = 0u128;
    while x <= sqrt_m {
        let x2 = x * x;
        let r = m - x2;
        if r % d == 0 {
            let y2 = r / d;
            let y = y2.isqrt();
            if y * y == y2 {
                return Some((x, y));
            }
        }
        x += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verify(d: u128, m: u128, sol: (u128, u128)) {
        let (x, y) = sol;
        assert_eq!(x * x + d * y * y, m, "x² + d·y² ≠ m");
    }

    #[test]
    fn d1_m5() {
        let s = cornacchia(1, 5).expect("solution exists");
        verify(1, 5, s);
    }

    #[test]
    fn d1_m13() {
        let s = cornacchia(1, 13).expect("solution exists");
        verify(1, 13, s);
    }

    #[test]
    fn d1_m7_unsolvable() {
        assert_eq!(cornacchia(1, 7), None);
    }

    #[test]
    fn d1_m11_unsolvable() {
        assert_eq!(cornacchia(1, 11), None);
    }

    #[test]
    fn d2_m11() {
        let s = cornacchia(2, 11).expect("solution exists");
        verify(2, 11, s);
    }

    #[test]
    fn d2_m5_unsolvable() {
        assert_eq!(cornacchia(2, 5), None);
    }

    #[test]
    fn d3_m7() {
        let s = cornacchia(3, 7).expect("solution exists");
        verify(3, 7, s);
    }

    #[test]
    fn d3_m21() {
        let s = cornacchia(3, 21).expect("solution exists");
        verify(3, 21, s);
    }

    #[test]
    fn m_zero_is_origin() {
        assert_eq!(cornacchia(1, 0), Some((0, 0)));
        assert_eq!(cornacchia(7, 0), Some((0, 0)));
    }

    #[test]
    fn d_zero_perfect_square() {
        assert_eq!(cornacchia(0, 49), Some((7, 0)));
        assert_eq!(cornacchia(0, 50), None);
    }

    #[test]
    fn d1_m_perfect_squares() {
        // m = 25 = 5² + 0 = 4² + 3² = 0² + 5². The trial iterator returns
        // (0, 5) (lexicographically smallest by x).
        let s = cornacchia(1, 25).expect("solution exists");
        verify(1, 25, s);
        assert_eq!(s, (0, 5));
    }

    #[test]
    fn d1_m1_unit() {
        assert_eq!(cornacchia(1, 1), Some((0, 1)));
    }

    #[test]
    fn d1_m2() {
        assert_eq!(cornacchia(1, 2), Some((1, 1)));
    }

    #[test]
    fn classical_d1_m13() {
        // 13 = 2² + 3² → some (x, y) with x² + y² = 13.
        let (x, y) = cornacchia_classical(1, 13).expect("13 is sum of two squares");
        assert_eq!(x * x + y * y, 13);
    }

    #[test]
    fn classical_d1_m17() {
        let (x, y) = cornacchia_classical(1, 17).expect("17 = 1 + 16");
        assert_eq!(x * x + y * y, 17);
    }

    #[test]
    fn classical_d1_m29() {
        let (x, y) = cornacchia_classical(1, 29).expect("29 = 4 + 25");
        assert_eq!(x * x + y * y, 29);
    }

    #[test]
    fn classical_d1_m37() {
        let (x, y) = cornacchia_classical(1, 37).expect("37 = 1 + 36");
        assert_eq!(x * x + y * y, 37);
    }

    #[test]
    fn classical_d1_m7_unsolvable() {
        // 7 ≡ 3 mod 4 → not a sum of two squares.
        assert_eq!(cornacchia_classical(1, 7), None);
    }

    #[test]
    fn classical_d1_m11_unsolvable() {
        assert_eq!(cornacchia_classical(1, 11), None);
    }

    #[test]
    fn classical_d2_m11() {
        let (x, y) = cornacchia_classical(2, 11).expect("11 = 9 + 2");
        assert_eq!(x * x + 2 * y * y, 11);
    }

    #[test]
    fn classical_d3_m31() {
        let (x, y) = cornacchia_classical(3, 31).expect("31 = 4 + 27");
        assert_eq!(x * x + 3 * y * y, 31);
    }

    #[test]
    fn classical_agrees_with_trial_for_primes() {
        // Both implementations should agree on whether a solution exists
        // (the specific (x, y) returned can differ in ordering but the
        // existence verdict must match).
        let cases: &[(u64, u64)] = &[
            (1, 5),
            (1, 13),
            (1, 7),
            (1, 11),
            (1, 17),
            (1, 29),
            (1, 37),
            (2, 11),
            (2, 3),
            (3, 7),
            (3, 31),
        ];
        for &(d, p) in cases {
            let trial = cornacchia(d as u128, p as u128).is_some();
            let classical = cornacchia_classical(d, p).is_some();
            assert_eq!(trial, classical, "(d={d}, p={p}) verdict mismatch");
        }
    }
}
