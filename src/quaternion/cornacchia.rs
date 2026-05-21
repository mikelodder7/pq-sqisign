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

use crate::quaternion::sqrt_mod::{tonelli_shanks, tonelli_shanks_uint};

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

/// Wide-Int variant of [`cornacchia_classical`].
///
/// Solve `x² + d · y² = p` where `p` is **prime**, at `Uint<LIMBS>` precision.
/// Mirrors [`cornacchia_classical`]'s structure: Tonelli-Shanks for the
/// modular square root + Euclidean ladder for the descent + perfect-square
/// check on `(p − b²) / d`. All steps use existing wide primitives:
/// [`super::sqrt_mod::tonelli_shanks_uint`] (S67),
/// `Uint::{rem_vartime, sub_mod, wrapping_mul, div_rem_vartime, floor_sqrt_vartime}`.
///
/// **Precision contract**: the caller's `LIMBS` MUST be large enough that
/// `p²` fits inside `Uint<LIMBS>` without overflow — i.e.,
/// `64 · LIMBS ≥ 2 · bits(p) + 1`. For the SQIsign reference primes:
/// - L1 (`p ≈ 2^248`): `LIMBS ≥ 8` (512 bits) is sufficient.
/// - L3 (`p ≈ 2^383`): `LIMBS ≥ 12` (768 bits) is sufficient.
/// - L5 (`p ≈ 2^505`): `LIMBS ≥ 16` (1024 bits) is sufficient.
///
/// The function does NOT validate this contract — under-sized `LIMBS`
/// will produce silent overflow in the `b · b` step. Match the S55-S58
/// pattern: caller widens to fit; function operates at the given width.
///
/// **Runtime**: `O((log p)²)` versus trial iteration's `O(√p)`. At L1
/// scale (`p ≈ 2^248`) trial iteration would need ≈ `2^124` steps; the
/// Euclidean ladder needs ≈ `2 · log₂(p) ≈ 500` steps. This is the
/// difference between "completes in milliseconds" and "completes after
/// the heat death of the sun".
///
/// **Use case**: this is the modular-sqrt path in `quat_represent_integer`
/// (the wide β-finder; S69+). It is also the workhorse for any wide-Int
/// sum-of-two-squares decomposition KLPT needs.
pub fn cornacchia_classical_uint<const LIMBS: usize>(
    d: &crypto_bigint::Uint<LIMBS>,
    p: &crypto_bigint::NonZero<crypto_bigint::Uint<LIMBS>>,
) -> Option<(crypto_bigint::Uint<LIMBS>, crypto_bigint::Uint<LIMBS>)> {
    let zero = crypto_bigint::Uint::<LIMBS>::from_u64(0);
    let one = crypto_bigint::Uint::<LIMBS>::ONE;

    // p == 1: trivial; no non-zero solutions.
    if *p.as_ref() == one {
        return None;
    }

    // d co-prime to p (for Tonelli step to make sense). Equivalent: d mod p ≠ 0.
    let d_mod = d.rem_vartime(p);
    if d_mod == zero {
        return None;
    }

    // t² ≡ −d (mod p). −d mod p = p − (d mod p).
    let neg_d = p.as_ref().wrapping_sub(&d_mod);
    let mut t = tonelli_shanks_uint::<LIMBS>(&neg_d, p)?;

    // Normalize: pick the t ≥ p/2 branch. `p/2` is the floor; this picks
    // the larger of the two roots so the Euclidean ladder converges in the
    // expected direction.
    let half_p = p.as_ref().shr_vartime(1);
    if t < half_p {
        t = p.as_ref().wrapping_sub(&t);
    }

    // Euclidean ladder: (a, b) ← (p, t); iterate (b, a rem b) while b² > p.
    let mut a = *p.as_ref();
    let mut b = t;
    loop {
        // b² ≤ p?  Compute b² in-place (caller's LIMBS guaranteed to fit p²
        // per the precision contract documented above).
        let b_sq = b.wrapping_mul(&b);
        if b_sq <= *p.as_ref() {
            break;
        }
        // (a, b) ← (b, a rem b).
        if b == zero {
            // Algorithm invariant says this can't happen for prime p with
            // t in range; defensive bail-out.
            return None;
        }
        let b_nz = crypto_bigint::NonZero::new(b).into_option()?;
        let r = a.rem_vartime(&b_nz);
        a = b;
        b = r;
    }

    // Candidate x = b. Check (p − b²) / d is a non-negative perfect square.
    let b_sq = b.wrapping_mul(&b);
    if b_sq > *p.as_ref() {
        return None;
    }
    let r = p.as_ref().wrapping_sub(&b_sq);

    // Use the ORIGINAL d (not d_mod), because we want exact division over
    // the integers, not modular. d_nz from the original d.
    let d_nz = crypto_bigint::NonZero::new(*d).into_option()?;
    let (s_sq, rem) = r.div_rem_vartime(&d_nz);
    if rem != zero {
        return None;
    }

    let s = s_sq.floor_sqrt_vartime();
    if s.wrapping_mul(&s) != s_sq {
        return None;
    }
    Some((b, s))
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

    // ── S68 — wide-Int classical Cornacchia (cornacchia_classical_uint) ──

    fn verify_wide<const LIMBS: usize>(
        d: &crypto_bigint::Uint<LIMBS>,
        p: &crypto_bigint::Uint<LIMBS>,
        sol: (crypto_bigint::Uint<LIMBS>, crypto_bigint::Uint<LIMBS>),
    ) {
        let (x, y) = sol;
        let x_sq = x.wrapping_mul(&x);
        let y_sq = y.wrapping_mul(&y);
        let dy_sq = d.wrapping_mul(&y_sq);
        let total = x_sq.wrapping_add(&dy_sq);
        assert_eq!(total, *p, "S68 wide Cornacchia: x² + d·y² ≠ p");
    }

    #[test]
    fn cornacchia_classical_uint_parity_with_narrow() {
        // S68 parity: for every (d, p) where narrow `cornacchia_classical`
        // returns Some/None, the wide version at `Uint<8>` must AGREE on
        // solvability, and when Some the returned (x, y) must satisfy
        // x² + d·y² = p at wide precision (may not be the SAME (x, y)
        // because the two algorithms may pick different roots).
        use crypto_bigint::{NonZero, Uint};
        let cases = [
            (1u64, 5u64),
            (1, 13),
            (1, 17),
            (1, 29),
            (1, 37),
            (2, 3),
            (2, 11),
            (3, 7),
            (3, 31),
            (1, 7),  // unsolvable
            (1, 11), // unsolvable
            (2, 5),  // unsolvable
        ];
        for &(d, p) in &cases {
            let narrow = cornacchia_classical(d, p).is_some();
            let d_w: Uint<8> = Uint::from_u64(d);
            let p_w: Uint<8> = Uint::from_u64(p);
            let p_nz: NonZero<Uint<8>> = NonZero::new(p_w).into_option().expect("p nonzero");
            let wide = cornacchia_classical_uint(&d_w, &p_nz);
            assert_eq!(
                narrow,
                wide.is_some(),
                "S68 parity verdict mismatch at (d={d}, p={p}): narrow={narrow} wide={:?}",
                wide.is_some(),
            );
            if let Some(sol) = wide {
                verify_wide(&d_w, &p_w, sol);
            }
        }
    }

    #[test]
    fn cornacchia_classical_uint_d1_p_real_lvl1_prime_minus_1_unsolvable() {
        // S68 real-prime-scale negative case: at the L1 prime
        // p = 5·2^248 − 1, p ≡ 3 (mod 4) which means -1 is a QNR mod p,
        // so x² + y² = p has no integer solution (Cornacchia returns
        // None at Tonelli-Shanks: −1 mod p is a QNR). This proves the
        // wide modular-sqrt path connects correctly at production
        // magnitude.
        use crypto_bigint::{NonZero, Uint};
        let p: Uint<8> = crate::params::lvl1::prime().resize::<8>();
        let p_nz: NonZero<Uint<8>> = NonZero::new(p).into_option().expect("p nonzero");
        let d: Uint<8> = Uint::from_u64(1);
        let r = cornacchia_classical_uint(&d, &p_nz);
        assert!(
            r.is_none(),
            "S68: p = 5·2^248 − 1 is 3 mod 4; -1 is a QNR; sum of two squares must be unsolvable, got {r:?}",
        );
    }

    #[test]
    fn cornacchia_classical_uint_d1_pseudoprime_p_eq_5_solvable() {
        // S68 minimal solvable case at wide precision: p = 5 ≡ 1 (mod 4),
        // so -1 IS a QR mod 5, Tonelli returns sqrt(-1) = sqrt(4) = ±2,
        // and the descent finds (1, 2): 1² + 1·2² = 5. Exercises the
        // FULL wide path (Tonelli iterative branch + Euclidean ladder
        // + perfect-square check).
        use crypto_bigint::{NonZero, Uint};
        let p: Uint<8> = Uint::from_u64(5);
        let p_nz: NonZero<Uint<8>> = NonZero::new(p).into_option().expect("p nonzero");
        let d: Uint<8> = Uint::from_u64(1);
        let (x, y) = cornacchia_classical_uint(&d, &p_nz).expect("5 = 1² + 2²");
        verify_wide(&d, &p, (x, y));
    }

    #[test]
    fn cornacchia_classical_uint_d_eq_zero_returns_none() {
        // d = 0: x² = p has no solution unless p is a perfect square,
        // and the function rejects d=0 cleanly at d.rem_vartime(p) == 0.
        use crypto_bigint::{NonZero, Uint};
        let p: Uint<8> = Uint::from_u64(13);
        let p_nz: NonZero<Uint<8>> = NonZero::new(p).into_option().expect("p nonzero");
        let d: Uint<8> = Uint::from_u64(0);
        assert_eq!(
            cornacchia_classical_uint(&d, &p_nz),
            None,
            "d=0 must return None (function rejects degenerate input)",
        );
    }

    #[test]
    fn cornacchia_classical_uint_p_eq_one_returns_none() {
        // p = 1: trivial; no non-zero solutions exist.
        use crypto_bigint::{NonZero, Uint};
        let p: Uint<8> = Uint::from_u64(1);
        let p_nz: NonZero<Uint<8>> = NonZero::new(p).into_option().expect("p nonzero");
        let d: Uint<8> = Uint::from_u64(1);
        assert_eq!(cornacchia_classical_uint(&d, &p_nz), None);
    }
}
