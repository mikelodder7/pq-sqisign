// SPDX-License-Identifier: MIT OR Apache-2.0
//! Norm-form quaternion witness search — the inner loop of KLPT's lift.
//!
//! The reduced norm of a quaternion `a + b·i + c·j + d·k ∈ B_{p,∞}` is
//!
//! ```text
//!     N(γ) = a² + b² + p · (c² + d²).
//! ```
//!
//! KLPT's body asks repeatedly: *given a target* `T`, *does there exist*
//! `γ ∈ O_0` *with* `N(γ) = T`? This module provides the prototype answer
//! for "fake prime" `p` small enough that we can brute-force search the
//! `(c, d)` plane, fixing each `(c, d)` and asking Cornacchia for `(a, b)`
//! with `a² + b² = T − p · (c² + d²)`.
//!
//! For real SQIsign primes the search space is too large to brute-force;
//! KLPT-production uses lattice closest-vector to shortcut directly to
//! good `(c, d)` candidates. The interface here matches what that path
//! will return — so plugging in the lattice-based search later is a body
//! swap, not a signature change.
//!
//! Output: a [`Quaternion<LIMBS>`] with the requested reduced norm, plus
//! a flag for whether the witness sits in the special `O_0` lattice (vs
//! merely in the broader `Z⟨1, i, j, k⟩` order).

use crate::quaternion::Quaternion;
use crate::quaternion::cornacchia::cornacchia;

/// Search for a quaternion `γ` with `N(γ) = target_norm`.
///
/// `p` is the level's prime (as a `u128` for the prototype search bound).
/// The search bound on `|c|, |d|` is `(target / p)^(1/2)`; for fake-prime
/// tests this stays well within a few thousand.
///
/// `LIMBS = 8` matches the rest of the quaternion module's working width.
/// Returns `None` if no witness exists within the search bound.
pub fn find_norm_witness(target_norm: u128, p: u128) -> Option<Quaternion<8>> {
    if target_norm == 0 {
        return Some(Quaternion::<8>::zero());
    }
    if p == 0 {
        // Without the `p · (c² + d²)` term this is pure two-square Cornacchia.
        let (a, b) = cornacchia(1, target_norm)?;
        return Some(Quaternion::<8>::new(
            into_int(a),
            into_int(b),
            into_int(0),
            into_int(0),
        ));
    }
    // The `p · (c² + d²)` contribution is bounded by `target_norm`, so
    // `c² + d²` is bounded by `target_norm / p`. Enumerate `(c, d)` with
    // increasing `c² + d²`.
    let max_cd_sq = target_norm / p;
    let max_c = max_cd_sq.isqrt();
    for c in 0..=max_c {
        let c2 = c * c;
        if c2 > max_cd_sq {
            break;
        }
        let max_d_for_c = (max_cd_sq - c2).isqrt();
        for d in 0..=max_d_for_c {
            let d2 = d * d;
            let cd_sq = c2 + d2;
            let p_cd = p.checked_mul(cd_sq)?;
            if p_cd > target_norm {
                continue;
            }
            let ab_target = target_norm - p_cd;
            // Solve a² + b² = ab_target via Cornacchia (trial).
            if let Some((a, b)) = cornacchia(1, ab_target) {
                return Some(Quaternion::<8>::new(
                    into_int(a),
                    into_int(b),
                    into_int(c),
                    into_int(d),
                ));
            }
        }
    }
    None
}

fn into_int(value: u128) -> crypto_bigint::Int<8> {
    // Truncate u128 → i64 fits for the prototype's small search range; if
    // the search bound exceeded i64::MAX we wouldn't be running this path.
    debug_assert!(value <= u128::from(i64::MAX as u64));
    #[allow(clippy::cast_possible_truncation)] // bounded by debug_assert above
    let v = value as i64;
    crypto_bigint::Int::<8>::from_i64(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_bigint::Uint;

    fn norm_matches(q: &Quaternion<8>, p: u128, expected: u128) -> bool {
        let p_u = Uint::<8>::from_u128(p);
        let n = q.norm(&p_u);
        // Convert back to u128 for comparison via the unsigned magnitude.
        let mag = n.abs();
        // Compare via byte-wise — n should be a small positive value.
        let bytes = mag.to_le_bytes();
        let mut acc: u128 = 0;
        for (i, &b) in bytes.as_ref().iter().enumerate().take(16) {
            acc |= u128::from(b) << (i * 8);
        }
        acc == expected
    }

    #[test]
    fn norm_one_returns_unit() {
        // N(1) = 1 (γ = 1 + 0i + 0j + 0k works trivially via Cornacchia(1, 1) = (0, 1)).
        let q = find_norm_witness(1, 7).expect("witness exists");
        assert!(norm_matches(&q, 7, 1));
    }

    #[test]
    fn norm_p_returns_j() {
        // N(j) = p (γ = j sets a=b=d=0, c=1).
        let p = 7u128;
        let q = find_norm_witness(p, p).expect("j is a witness");
        assert!(norm_matches(&q, p, p));
    }

    #[test]
    fn norm_two_p_plus_two() {
        // 2·7 + 2 = 16. Try p=7, target=16. Witness: a=2, b=2, c=1, d=1
        // → 4 + 4 + 7·(1 + 1) = 22 — wrong. Recompute:
        // a²+b²+p(c²+d²) = 16. Try c=d=0: a²+b² = 16 → (a,b)=(0,4).
        let q = find_norm_witness(16, 7).expect("witness exists");
        assert!(norm_matches(&q, 7, 16));
    }

    #[test]
    fn norm_seven_is_j() {
        let q = find_norm_witness(7, 7).expect("witness exists");
        // Could be (0, 0, 1, 0) (= j) or (a, b, c, d) with c²+d²=1.
        assert!(norm_matches(&q, 7, 7));
    }

    #[test]
    fn norm_eight_with_p_seven() {
        // 8 = a² + b² + 7(c² + d²). With c=d=0: 8 = 4+4. (a,b)=(2,2).
        let q = find_norm_witness(8, 7).expect("witness exists");
        assert!(norm_matches(&q, 7, 8));
    }

    #[test]
    fn norm_three_with_p_seven_unsolvable() {
        // 3 = a² + b² + 7(c²+d²). c=d=0 → a²+b² = 3 → no solution (3 ≡ 3 mod 4).
        // c²+d² > 0 → 7·(c²+d²) ≥ 7 > 3 → no solution.
        assert!(find_norm_witness(3, 7).is_none());
    }

    #[test]
    fn zero_target_is_zero_quaternion() {
        let q = find_norm_witness(0, 7).expect("zero is a witness");
        assert_eq!(q.a, crypto_bigint::Int::<8>::from_i64(0));
        assert_eq!(q.b, crypto_bigint::Int::<8>::from_i64(0));
        assert_eq!(q.c, crypto_bigint::Int::<8>::from_i64(0));
        assert_eq!(q.d, crypto_bigint::Int::<8>::from_i64(0));
    }

    #[test]
    fn p_zero_falls_back_to_two_square() {
        // p = 0 means N(γ) = a² + b² — pure Cornacchia.
        let q = find_norm_witness(13, 0).expect("13 = 4 + 9");
        let p = Uint::<8>::from_u128(0);
        let n = q.norm(&p).abs();
        let mut acc: u128 = 0;
        for (i, &b) in n.to_le_bytes().as_ref().iter().enumerate().take(16) {
            acc |= u128::from(b) << (i * 8);
        }
        assert_eq!(acc, 13);
    }
}
