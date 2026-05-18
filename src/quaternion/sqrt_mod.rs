// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tonelli-Shanks square root modulo an odd prime.
//!
//! The classical-variant Cornacchia (and the KLPT norm-form lift) need to
//! solve `t² ≡ −d (mod m)` quickly for `m` prime. Trial-iteration scales
//! as `O(m)`; Tonelli-Shanks scales as `O((log m)² · log_2 (m − 1))`
//! amortised, with a small linear search for a quadratic non-residue.
//!
//! All arithmetic here is fixed at `u64` operands with `u128` intermediate
//! widths — i.e. the routine is exact for primes `p < 2^64`. KLPT-production
//! primes are much wider; the same algorithm composed with
//! `crypto_bigint::Uint<LIMBS>` arithmetic lands in a later session.

/// `(a · b) mod m` with `u128` intermediate, narrowing back to `u64`.
/// Safe because `% m` is `< m < 2^64`.
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn mulmod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// `base^exp mod m` for `m < 2^64`.
///
/// Uses the standard square-and-multiply ladder; intermediate products
/// promote to `u128` so the multiplication doesn't overflow.
pub fn pow_mod(mut base: u64, mut exp: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    let mut result: u64 = 1;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mulmod(result, base, m);
        }
        exp >>= 1;
        base = mulmod(base, base, m);
    }
    result
}

/// Tonelli-Shanks: return some `r` with `r² ≡ a (mod p)`, or `None` if `a`
/// is a quadratic non-residue. `p` is assumed prime and odd.
///
/// Special cases handled inline:
/// - `a ≡ 0 (mod p)` → `Some(0)`.
/// - `p == 2` → `Some(a & 1)`.
/// - `p ≡ 3 (mod 4)` → `Some(a^((p+1)/4))` (the closed form we already use
///   for `Fp::sqrt`).
///
/// Otherwise the iterative Tonelli-Shanks loop runs.
pub fn tonelli_shanks(a: u64, p: u64) -> Option<u64> {
    let a = a % p;
    if a == 0 {
        return Some(0);
    }
    if p == 2 {
        return Some(a & 1);
    }
    // Euler criterion — bail if a is not a QR.
    if pow_mod(a, (p - 1) / 2, p) != 1 {
        return None;
    }
    if p % 4 == 3 {
        return Some(pow_mod(a, (p + 1) / 4, p));
    }
    // Factor p − 1 = Q · 2^S with Q odd.
    let mut q = p - 1;
    let mut s: u32 = 0;
    while q & 1 == 0 {
        q >>= 1;
        s += 1;
    }
    // Find a quadratic non-residue z.
    let mut z: u64 = 2;
    while pow_mod(z, (p - 1) / 2, p) != p - 1 {
        z += 1;
        // The smallest QNR is at most O((log p)²) — bounded; this terminates.
    }
    let mut m: u32 = s;
    let mut c = pow_mod(z, q, p);
    let mut t = pow_mod(a, q, p);
    // `(q + 1) / 2` here is the textbook Tonelli-Shanks exponent for `R`;
    // because `q` is odd, this is `(q + 1) / 2` exact, not a `div_ceil` of
    // some larger value, but clippy can't see the algebraic context.
    #[allow(clippy::manual_div_ceil)]
    let mut r = pow_mod(a, (q + 1) / 2, p);
    loop {
        if t == 0 {
            return Some(0);
        }
        if t == 1 {
            return Some(r);
        }
        // Find least i in (0, m) with t^(2^i) ≡ 1.
        let mut i: u32 = 0;
        let mut temp = t;
        while temp != 1 && i < m {
            temp = mulmod(temp, temp, p);
            i += 1;
        }
        if i == m {
            // Shouldn't happen if a is a QR (already checked).
            return None;
        }
        // b = c^(2^(m − i − 1))
        let mut b = c;
        for _ in 0..(m - i - 1) {
            b = mulmod(b, b, p);
        }
        m = i;
        c = mulmod(b, b, p);
        t = mulmod(t, c, p);
        r = mulmod(r, b, p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Confirm `r² ≡ a (mod p)`.
    fn check(a: u64, p: u64, r: u64) {
        assert_eq!(mulmod(r, r, p), a % p, "r² ≢ a mod p");
    }

    #[test]
    fn sqrt_4_mod_7() {
        let r = tonelli_shanks(4, 7).expect("4 is QR mod 7");
        assert!(r == 2 || r == 5);
        check(4, 7, r);
    }

    #[test]
    fn sqrt_2_mod_7() {
        let r = tonelli_shanks(2, 7).expect("2 is QR mod 7 (3² = 9 ≡ 2)");
        assert!(r == 3 || r == 4);
        check(2, 7, r);
    }

    #[test]
    fn sqrt_3_mod_7_is_none() {
        assert_eq!(tonelli_shanks(3, 7), None);
    }

    #[test]
    fn sqrt_2_mod_17() {
        // p = 17 = 1 mod 16 → exercises the iterative Tonelli-Shanks branch.
        let r = tonelli_shanks(2, 17).expect("2 is QR mod 17");
        assert!(r == 6 || r == 11);
        check(2, 17, r);
    }

    #[test]
    fn sqrt_zero_is_zero() {
        assert_eq!(tonelli_shanks(0, 11), Some(0));
        assert_eq!(tonelli_shanks(0, 17), Some(0));
    }

    #[test]
    fn sqrt_one_mod_13() {
        let r = tonelli_shanks(1, 13).expect("1 is always QR");
        assert!(r == 1 || r == 12);
        check(1, 13, r);
    }

    #[test]
    fn sqrt_5_mod_29() {
        // p = 29 ≡ 1 mod 4 — iterative branch again.
        let r = tonelli_shanks(5, 29).expect("5 is QR mod 29");
        assert!(r == 11 || r == 18);
        check(5, 29, r);
    }

    #[test]
    fn sqrt_10_mod_13() {
        let r = tonelli_shanks(10, 13).expect("10 is QR mod 13");
        assert!(r == 6 || r == 7);
        check(10, 13, r);
    }

    #[test]
    fn sqrt_7_mod_23_is_none() {
        assert_eq!(tonelli_shanks(7, 23), None);
    }

    #[test]
    fn pow_mod_known_values() {
        assert_eq!(pow_mod(2, 10, 1000), 24); // 1024 mod 1000
        assert_eq!(pow_mod(3, 4, 5), 1); // 81 mod 5 = 1
        assert_eq!(pow_mod(7, 0, 13), 1);
        assert_eq!(pow_mod(0, 5, 13), 0);
    }

    #[test]
    fn pow_mod_handles_m1() {
        assert_eq!(pow_mod(123, 456, 1), 0);
    }

    #[test]
    fn fermat_little_theorem() {
        // a^(p−1) ≡ 1 (mod p) for prime p ∤ a.
        for p in [7u64, 11, 13, 17, 19, 23, 29, 31] {
            for a in [2u64, 3, 5, 6] {
                if a % p != 0 {
                    assert_eq!(pow_mod(a, p - 1, p), 1, "Fermat fail a={a} p={p}");
                }
            }
        }
    }

    #[test]
    fn sqrt_round_trip_many_primes() {
        // For each prime p ≡ 1 mod 4 (so we hit the iterative branch),
        // pick small a, compute r, check r² ≡ a.
        for p in [13u64, 17, 29, 37, 41, 53] {
            for a in 1u64..10 {
                if pow_mod(a, (p - 1) / 2, p) == 1 {
                    let r = tonelli_shanks(a, p).expect("QR");
                    check(a, p, r);
                }
            }
        }
    }
}
