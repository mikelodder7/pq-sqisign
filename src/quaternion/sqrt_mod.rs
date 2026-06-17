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

/// `base^exp mod modulus` on `crypto_bigint::Uint<LIMBS>`, the bignum
/// generalisation of [`pow_mod`].
///
/// Square-and-multiply ladder using `crypto_bigint::Uint::mul_mod_vartime`
/// / `square_mod_vartime` / `shr_vartime`. Variable-time in the exponent
/// (acceptable for Miller-Rabin witness exponents, which are public per
/// the protocol's threat model). Returns `0` when `modulus == 1` and the
/// caller's intent is unambiguous: every value is `≡ 0 (mod 1)`.
///
/// Cost: `bits(exp)` squarings + `popcount(exp)` multiplies, each
/// `O(LIMBS²)` via `mul_mod_vartime`. For `LIMBS = 8` (the quaternion
/// module's working width) and a 512-bit exponent, ~512 squarings + ~256
/// multiplies — well within a single test run.
pub fn pow_mod_uint<const LIMBS: usize>(
    base: &crypto_bigint::Uint<LIMBS>,
    exp: &crypto_bigint::Uint<LIMBS>,
    modulus: &crypto_bigint::NonZero<crypto_bigint::Uint<LIMBS>>,
) -> crypto_bigint::Uint<LIMBS> {
    let one = crypto_bigint::Uint::<LIMBS>::ONE;
    let zero = crypto_bigint::Uint::<LIMBS>::from_u64(0);
    if *modulus.as_ref() == one {
        return zero;
    }
    let mut result = one;
    let mut base_mod = base.rem_vartime(modulus);
    let mut e = *exp;
    while e != zero {
        if e.as_limbs()[0].0 & 1 == 1 {
            result = result.mul_mod_vartime(&base_mod, modulus);
        }
        e = e.shr_vartime(1);
        base_mod = base_mod.square_mod_vartime(modulus);
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

/// Wide-Int Tonelli-Shanks: returns some `r` with `r² ≡ a (mod p)`, or
/// `None` if `a` is a quadratic non-residue mod `p`. Caller must pass an
/// odd prime `p`.
///
/// Mirrors the narrow [`tonelli_shanks`] structure at `Uint<LIMBS>`
/// precision via `pow_mod_uint`, `Uint::mul_mod_vartime`,
/// `Uint::square_mod_vartime`, and `Uint::shr_vartime`.
///
/// **Fast path** — when `p ≡ 3 (mod 4)`, returns the closed form
/// `a^((p+1)/4) mod p` in one modexp. Every SQIsign reference prime
/// (`c·2^e − 1` form) hits this path: `p − 1 = c·2^e − 2 = 2·(c·2^(e−1) − 1)`
/// has `2`-adic valuation 1, so `p ≡ 3 (mod 4)`.
///
/// **Iterative path** — when `p ≡ 1 (mod 4)` (the loop case). Factors
/// `p − 1 = Q · 2^S` with `Q` odd; finds a quadratic non-residue
/// `z ∈ {2, 3, ...}` via Euler-criterion search (terminates in
/// `O((log p)²)` candidates); runs the standard Tonelli-Shanks recursion.
///
/// This is the prerequisite for the wide [`super::cornacchia::cornacchia`] port
/// (wide Cornacchia), which solves the modular-sqrt step for the
/// `quat_represent_integer` outer loop.
pub fn tonelli_shanks_uint<const LIMBS: usize>(
    a: &crypto_bigint::Uint<LIMBS>,
    p: &crypto_bigint::NonZero<crypto_bigint::Uint<LIMBS>>,
) -> Option<crypto_bigint::Uint<LIMBS>> {
    let zero = crypto_bigint::Uint::<LIMBS>::from_u64(0);
    let one = crypto_bigint::Uint::<LIMBS>::ONE;

    let a_mod = a.rem_vartime(p);
    if a_mod == zero {
        return Some(zero);
    }

    // p − 1 (p is at least 3, so this doesn't underflow).
    let p_minus_1 = p.as_ref().wrapping_sub(&one);

    // Euler criterion: a^((p−1)/2) ≡ ±1 (mod p). ≠ 1 → a is a QNR.
    let half_p_minus_1 = p_minus_1.shr_vartime(1);
    let euler = pow_mod_uint::<LIMBS>(&a_mod, &half_p_minus_1, p);
    if euler != one {
        return None;
    }

    // Fast path: p ≡ 3 (mod 4) → return a^((p+1)/4) mod p.
    // Low 2 bits of p: p.as_words()[0] & 0b11. p ≡ 3 mod 4 iff this is 3.
    let p_low = p.as_ref().as_words()[0];
    if (p_low & 0b11) == 0b11 {
        let p_plus_1 = p.as_ref().wrapping_add(&one);
        let exp = p_plus_1.shr_vartime(2);
        return Some(pow_mod_uint::<LIMBS>(&a_mod, &exp, p));
    }

    // Iterative path. Factor p − 1 = Q · 2^S with Q odd.
    let mut q = p_minus_1;
    let mut s: u32 = 0;
    while q.as_words()[0] & 1 == 0 {
        q = q.shr_vartime(1);
        s += 1;
    }

    // Find a quadratic non-residue z (z ≥ 2). The least QNR is bounded
    // by O((log p)²) under GRH; in practice ≤ a few dozen for any
    // practical prime. The unconditional bound `2 √p · log² p` is a
    // safe outer cap.
    let mut z = crypto_bigint::Uint::<LIMBS>::from_u64(2);
    loop {
        let euler_z = pow_mod_uint::<LIMBS>(&z, &half_p_minus_1, p);
        if euler_z == p_minus_1 {
            break;
        }
        z = z.wrapping_add(&one);
        // Defensive: if z overflows the modulus, p was not prime; bail.
        if z >= *p.as_ref() {
            return None;
        }
    }

    let mut m: u32 = s;
    let mut c = pow_mod_uint::<LIMBS>(&z, &q, p);
    let mut t = pow_mod_uint::<LIMBS>(&a_mod, &q, p);
    // R initial = a^((Q+1)/2) — Q is odd so Q+1 is even.
    let q_plus_1 = q.wrapping_add(&one);
    let r_exp = q_plus_1.shr_vartime(1);
    let mut r = pow_mod_uint::<LIMBS>(&a_mod, &r_exp, p);

    loop {
        if t == zero {
            return Some(zero);
        }
        if t == one {
            return Some(r);
        }
        // Least i in (0, m) with t^(2^i) ≡ 1.
        let mut i: u32 = 0;
        let mut temp = t;
        while temp != one && i < m {
            temp = temp.square_mod_vartime(p);
            i += 1;
        }
        if i == m {
            // Should not happen if a is a QR; defensive return.
            return None;
        }
        // b = c^(2^(m − i − 1))
        let mut b = c;
        for _ in 0..(m - i - 1) {
            b = b.square_mod_vartime(p);
        }
        m = i;
        c = b.square_mod_vartime(p);
        t = t.mul_mod_vartime(&c, p);
        r = r.mul_mod_vartime(&b, p);
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

    #[test]
    fn pow_mod_uint_matches_u64_pow_mod_on_small_inputs() {
        // Cross-check the bignum version against the u64 version for
        // small inputs where both are valid.
        use crypto_bigint::{NonZero, Uint};
        for (base, exp, m) in [
            (2u64, 10, 1000),
            (3, 4, 7),
            (5, 0, 13),
            (5, 1, 13),
            (7, 100, 11),
            (123, 456, 789),
        ] {
            let expected = pow_mod(base, exp, m);
            let base_u: Uint<8> = Uint::from_u64(base);
            let exp_u: Uint<8> = Uint::from_u64(exp);
            let m_u: Uint<8> = Uint::from_u64(m);
            let m_nz: NonZero<Uint<8>> = NonZero::new(m_u).into_option().expect("m nonzero");
            let actual = pow_mod_uint(&base_u, &exp_u, &m_nz);
            assert_eq!(
                actual,
                Uint::from_u64(expected),
                "mismatch at base={base} exp={exp} m={m}"
            );
        }
    }

    #[test]
    fn pow_mod_uint_modulus_one_returns_zero() {
        // Every integer is ≡ 0 (mod 1).
        use crypto_bigint::{NonZero, Uint};
        let base: Uint<8> = Uint::from_u64(42);
        let exp: Uint<8> = Uint::from_u64(13);
        let one: Uint<8> = Uint::from_u64(1);
        let one_nz: NonZero<Uint<8>> = NonZero::new(one).into_option().expect("1 nonzero");
        assert_eq!(pow_mod_uint(&base, &exp, &one_nz), Uint::from_u64(0));
    }

    #[test]
    fn pow_mod_uint_satisfies_fermats_little_theorem() {
        // For prime p and gcd(a, p) = 1: a^(p-1) ≡ 1 (mod p).
        use crypto_bigint::{NonZero, Uint};
        for p in [13u64, 17, 29, 31, 41, 43] {
            for a in 2u64..6 {
                let a_u: Uint<8> = Uint::from_u64(a);
                let p_minus_1: Uint<8> = Uint::from_u64(p - 1);
                let p_u: Uint<8> = Uint::from_u64(p);
                let p_nz: NonZero<Uint<8>> = NonZero::new(p_u).into_option().expect("p nonzero");
                let result = pow_mod_uint(&a_u, &p_minus_1, &p_nz);
                assert_eq!(
                    result,
                    Uint::from_u64(1),
                    "Fermat fail: {a}^({p} − 1) mod {p}"
                );
            }
        }
    }

    #[test]
    fn pow_mod_uint_zero_exp_returns_one() {
        // a^0 = 1 for all a (and any modulus > 1).
        use crypto_bigint::{NonZero, Uint};
        let base: Uint<8> = Uint::from_u64(7);
        let exp: Uint<8> = Uint::from_u64(0);
        let m: Uint<8> = Uint::from_u64(5);
        let m_nz: NonZero<Uint<8>> = NonZero::new(m).into_option().expect("m nonzero");
        assert_eq!(pow_mod_uint(&base, &exp, &m_nz), Uint::from_u64(1));
    }

    // ── wide-Int Tonelli-Shanks (tonelli_shanks_uint) ──

    /// Verify `r² ≡ a (mod p)` at `Uint<LIMBS>` precision.
    fn check_wide<const LIMBS: usize>(
        a: &crypto_bigint::Uint<LIMBS>,
        p: &crypto_bigint::NonZero<crypto_bigint::Uint<LIMBS>>,
        r: &crypto_bigint::Uint<LIMBS>,
    ) {
        let r_sq = r.square_mod_vartime(p);
        assert_eq!(r_sq, a.rem_vartime(p), "r² ≢ a mod p at wide precision");
    }

    #[test]
    fn tonelli_shanks_uint_parity_with_narrow_small_primes() {
        // Parity: for every small prime p where the narrow
        // `tonelli_shanks(a, p) = Some(r)`, the wide version at
        // `Uint<8>` must produce a valid square root (which may or
        // may not equal r — both ±r are valid).
        use crypto_bigint::{NonZero, Uint};
        let small_primes = [7u64, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
        let mut covered_p3 = 0;
        let mut covered_p1 = 0;
        for &p in &small_primes {
            let p_w: Uint<8> = Uint::from_u64(p);
            let p_nz: NonZero<Uint<8>> = NonZero::new(p_w).into_option().expect("p nonzero");
            for a in 1u64..10 {
                let narrow_result = tonelli_shanks(a, p);
                let a_w: Uint<8> = Uint::from_u64(a);
                let wide_result = tonelli_shanks_uint(&a_w, &p_nz);
                let parity_ok = matches!(
                    (narrow_result, wide_result),
                    (Some(_), Some(_)) | (None, None)
                );
                assert!(
                    parity_ok,
                    "parity break at p={p} a={a}: narrow={narrow_result:?} wide={wide_result:?}",
                );
                if let Some(wide_r) = wide_result {
                    check_wide(&a_w, &p_nz, &wide_r);
                    if p % 4 == 3 {
                        covered_p3 += 1;
                    } else {
                        covered_p1 += 1;
                    }
                }
            }
        }
        // Confirm coverage of both branches across the sweep.
        assert!(covered_p3 > 0, "must exercise p ≡ 3 mod 4 fast path");
        assert!(
            covered_p1 > 0,
            "must exercise p ≡ 1 mod 4 iterative path"
        );
    }

    #[test]
    fn tonelli_shanks_uint_real_lvl1_prime_p_mod_4_eq_3() {
        // Production-scale test: real L1 prime p = 5·2^248 − 1
        // hits the fast path (p ≡ 3 mod 4). Verify sqrt(4) and
        // sqrt(9) at Uint<8> precision.
        use crypto_bigint::{NonZero, Uint};
        let p: Uint<8> = crate::params::lvl1::prime().resize::<8>();
        let p_nz: NonZero<Uint<8>> = NonZero::new(p).into_option().expect("p nonzero");
        // sqrt(4) mod p — must square back to 4.
        let four: Uint<8> = Uint::from_u64(4);
        let r4 = tonelli_shanks_uint(&four, &p_nz).expect("4 is QR mod any prime");
        check_wide(&four, &p_nz, &r4);
        // sqrt(9) mod p — must square back to 9.
        let nine: Uint<8> = Uint::from_u64(9);
        let r9 = tonelli_shanks_uint(&nine, &p_nz).expect("9 is QR mod any prime");
        check_wide(&nine, &p_nz, &r9);
    }

    #[test]
    fn tonelli_shanks_uint_rejects_qnr_small_prime() {
        // 3 is a QNR mod 7. Wide path must agree with narrow path
        // (returns None).
        use crypto_bigint::{NonZero, Uint};
        let p: Uint<8> = Uint::from_u64(7);
        let p_nz: NonZero<Uint<8>> = NonZero::new(p).into_option().expect("p nonzero");
        let three: Uint<8> = Uint::from_u64(3);
        assert_eq!(tonelli_shanks_uint(&three, &p_nz), None);
    }

    #[test]
    fn tonelli_shanks_uint_zero_returns_zero() {
        // 0² ≡ 0 mod p for any p; the wide path takes the early return.
        use crypto_bigint::{NonZero, Uint};
        let p: Uint<8> = Uint::from_u64(13);
        let p_nz: NonZero<Uint<8>> = NonZero::new(p).into_option().expect("p nonzero");
        let zero: Uint<8> = Uint::from_u64(0);
        assert_eq!(
            tonelli_shanks_uint(&zero, &p_nz),
            Some(Uint::from_u64(0)),
            "sqrt(0) mod p = 0"
        );
    }

    /// Root-CHOICE oracle: pin `tonelli_shanks_uint` against GMP ground truth
    /// at SEC_DEGREE = 2^512 + 75 (the keygen secret-ideal modulus). The C
    /// `ibz_sqrt_mod_p` p%4==3 branch returns `a^((p+1)/4) mod p` — a SPECIFIC
    /// root, not `min(r, p−r)`. Since SEC_DEGREE ≡ 3 mod 8 ⇒ (2|p) = −1, that
    /// formula yields the NON-obvious roots `sqrt(4) = p−2` and `sqrt(9) = p−3`
    /// (a naive sqrt returning 2 / 3 would fail). Golden values computed by a
    /// standalone GMP oracle (`a^((p+1)/4) mod p`, the verbatim C algorithm), so
    /// this is a cross-implementation check, not a mirror of our own code. It
    /// confirms the keygen sqrt step is byte-exact with the C for p ≡ 3 mod 4.
    #[test]
    fn tonelli_root_choice_matches_gmp_at_sec_degree() {
        use crypto_bigint::{NonZero, Uint};
        let p: Uint<16> = crate::params::lvl1::sec_degree(); // 2^512 + 75, ≡ 3 mod 4
        let p_nz: NonZero<Uint<16>> = NonZero::new(p).into_option().expect("SEC_DEGREE nonzero");
        let p_minus_2 = p.wrapping_sub(&Uint::from_u64(2));
        let p_minus_3 = p.wrapping_sub(&Uint::from_u64(3));

        // GMP-confirmed roots (a^((p+1)/4) mod p).
        assert_eq!(
            tonelli_shanks_uint::<16>(&Uint::from_u64(4), &p_nz),
            Some(p_minus_2),
            "sqrt(4) mod SEC_DEGREE must be p−2 (the a^((p+1)/4) root)",
        );
        assert_eq!(
            tonelli_shanks_uint::<16>(&Uint::from_u64(9), &p_nz),
            Some(p_minus_3),
            "sqrt(9) mod SEC_DEGREE must be p−3",
        );
        // The chosen root squares back to the input.
        let four = Uint::<16>::from_u64(4);
        let r4 = tonelli_shanks_uint::<16>(&four, &p_nz).expect("4 is a QR");
        assert_eq!(
            r4.mul_mod_vartime(&r4, &p_nz),
            four,
            "root² ≡ 4 mod SEC_DEGREE"
        );
    }
}
