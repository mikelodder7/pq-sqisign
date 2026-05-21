// SPDX-License-Identifier: MIT OR Apache-2.0
//! Wide-Int β-finder — `quat_represent_integer` port.
//!
//! Given target reduced norm `M` and prime `p`, find `β ∈ O_0` with
//! `N_red(β) = M`. The narrow `find_quaternion_in_ideal_with_norm`
//! (in [`super::short_vec`]) does brute-force enumeration bounded by
//! `i64` — fine for small inputs, infeasible at real-prime scale.
//!
//! # Algorithm — `quat_represent_integer`
//!
//! In O_0-basis coords `β = v₀·1 + v₁·i + v₂·(i+j)/2 + v₃·(1+k)/2`,
//! the reduced norm form is
//!
//! ```text
//! 4·N(β) = 4v₀² + 4v₁² + 4v₀·v₃ + 4v₁·v₂ + (1+p)·v₂² + (1+p)·v₃²
//! ```
//!
//! Completing the square on `v₀` (with linear term `4v₀·v₃`) and `v₁`
//! (with linear term `4v₁·v₂`), set `a = 2v₀ + v₃` and `b = 2v₁ + v₂`:
//!
//! ```text
//! a² = 4v₀² + 4v₀·v₃ + v₃²
//! b² = 4v₁² + 4v₁·v₂ + v₂²
//! a² + b² = 4v₀² + 4v₁² + 4v₀·v₃ + 4v₁·v₂ + v₂² + v₃²
//!         = 4N(β) − (1+p)·v₂² − (1+p)·v₃² + v₂² + v₃²
//!         = 4N(β) − p·(v₂² + v₃²)
//! ```
//!
//! So solving `N(β) = M` reduces to: pick `c = v₂`, `d = v₃`, compute
//! `T = 4M − p·(c² + d²)`, then solve `a² + b² = T` via Cornacchia. If
//! parity matches — `a ≡ d (mod 2)` and `b ≡ c (mod 2)` — recover
//! `v₀ = (a − d)/2`, `v₁ = (b − c)/2`.
//!
//! # Loop structure
//!
//! 1. Sample `(c, d)` uniformly from `[−bound, bound]²` via the existing
//!    `sample_random_quaternion_o0` sampler.
//! 2. Compute `cd_sq = c² + d²`; reject if `p · cd_sq ≥ 4M` (T would be
//!    non-positive).
//! 3. Compute `T = 4M − p · cd_sq`; reject if `T mod 4 ≠ 1` (Cornacchia
//!    with `d = 1` needs `T ≡ 1 (mod 4)` for `−1` to be a QR mod T).
//! 4. Miller-Rabin-test `T` for primality at WIDE precision (S44/S67
//!    primitives); reject composites.
//! 5. Call [`super::cornacchia::cornacchia_classical_uint`] to solve
//!    `a² + b² = T`; reject if `None`.
//! 6. Parity check: reject unless `a` ≡ `d` (mod 2) and `b` ≡ `c` (mod 2).
//! 7. Recover `v₀ = (a − d)/2`, `v₁ = (b − c)/2`. Return
//!    `[v₀, v₁, v₂ = c, v₃ = d]`.
//!
//! # Precision contract
//!
//! Caller's `LIMBS` MUST satisfy `64·LIMBS ≥ 2·bits(p) + 1` so that
//! the Cornacchia internal `b·b` doesn't overflow. See
//! [`super::cornacchia::cornacchia_classical_uint`] for the full
//! discussion. For SQIsign primes:
//! - L1 (`p ≈ 2^248`): `LIMBS ≥ 8` (512 bits).
//! - L3 (`p ≈ 2^383`): `LIMBS ≥ 12` (768 bits).
//! - L5 (`p ≈ 2^505`): `LIMBS ≥ 16` (1024 bits).
//!
//! # Performance
//!
//! Expected `O(log² M)` trials per call (random `(c, d)` ⇒ random `T`,
//! `Pr[T prime] ≈ 1/log T`, parity ≈ ¼ pass), each trial is a
//! Miller-Rabin (`witnesses.len()` modexps) plus a Cornacchia
//! (`O(log² p)`). At L1 scale roughly hundreds-of-trials per call;
//! `max_trials = 1<<14` is a conservative upper bound.

use crypto_bigint::{Int, NonZero, Uint};
use rand_core::CryptoRng;

use crate::quaternion::cornacchia::cornacchia_classical_uint;
use crate::quaternion::primality::is_probable_prime_with_witnesses;
use crate::quaternion::sample::sample_random_quaternion_o0;

/// Find `β ∈ O_0` with `N_red(β) = target_m`, at `Uint<LIMBS>` precision.
///
/// Returns `Some([v₀, v₁, v₂, v₃])` in O_0-basis coords on success, or
/// `None` if the search budget is exhausted.
///
/// See module docs for the algorithm and precision contract.
pub fn find_quaternion_in_full_order_with_norm_wide<const LIMBS: usize, R: CryptoRng>(
    target_m: &Uint<LIMBS>,
    p: &Uint<LIMBS>,
    sample_bound: i64,
    max_trials: usize,
    witnesses: &[Uint<LIMBS>],
    rng: &mut R,
) -> Option<[Int<LIMBS>; 4]> {
    let zero_u = Uint::<LIMBS>::from_u64(0);

    // 4M.
    let four_m = target_m.shl_vartime(2);

    for _ in 0..max_trials {
        // Sample (c, d) uniformly in [-bound, bound]² via the existing
        // O_0 sampler; we use coords 2 and 3 for (c = v₂, d = v₃).
        let sample = sample_random_quaternion_o0(rng, sample_bound);
        let c_n: Int<8> = sample[2];
        let d_n: Int<8> = sample[3];

        // Widen c, d to Int<LIMBS> (sign-extending) for the v₀/v₁
        // reconstruction step at the end.
        let c_w: Int<LIMBS> = c_n.resize::<LIMBS>();
        let d_w: Int<LIMBS> = d_n.resize::<LIMBS>();

        // Compute c² + d² at Uint<LIMBS>. Parity is invariant under sign,
        // so use absolute values for the unsigned multiply.
        let c_abs_n = c_n.abs();
        let d_abs_n = d_n.abs();
        let c_abs_w: Uint<LIMBS> = c_abs_n.resize::<LIMBS>();
        let d_abs_w: Uint<LIMBS> = d_abs_n.resize::<LIMBS>();
        let c_sq = c_abs_w.wrapping_mul(&c_abs_w);
        let d_sq = d_abs_w.wrapping_mul(&d_abs_w);
        let cd_sq = c_sq.wrapping_add(&d_sq);

        // p · (c² + d²).
        let p_cd_sq = p.wrapping_mul(&cd_sq);

        // T = 4M − p·(c² + d²); reject if non-positive.
        if four_m <= p_cd_sq {
            continue;
        }
        let t = four_m.wrapping_sub(&p_cd_sq);

        // T ≡ 1 (mod 4)? (Cornacchia with d=1 needs −1 a QR mod T.)
        if (t.as_words()[0] & 0b11) != 1 {
            continue;
        }

        // T probably-prime?
        if !is_probable_prime_with_witnesses::<LIMBS>(&t, witnesses) {
            continue;
        }

        // Solve a² + b² = T.
        let t_nz: NonZero<Uint<LIMBS>> = NonZero::new(t).into_option()?;
        let one_u = Uint::<LIMBS>::ONE;
        let (a, b) = cornacchia_classical_uint::<LIMBS>(&one_u, &t_nz)?;

        // Parity check: a ≡ d (mod 2) and b ≡ c (mod 2). Parity is
        // invariant under sign, so compare LSB of |c|, |d| with a, b.
        let a_lsb = a.as_words()[0] & 1;
        let b_lsb = b.as_words()[0] & 1;
        let c_lsb = c_abs_w.as_words()[0] & 1;
        let d_lsb = d_abs_w.as_words()[0] & 1;
        if a_lsb != d_lsb || b_lsb != c_lsb {
            continue;
        }

        // Convert a, b (positive Uint<LIMBS>) to Int<LIMBS>. Per the
        // precision contract, `a, b < √p < p < 2^(64·LIMBS−1)`, so the
        // high bit is zero and `Int::from_words(a.to_words())` is a
        // safe interpretation as a non-negative Int.
        let a_w: Int<LIMBS> = Int::from_words(a.to_words());
        let b_w: Int<LIMBS> = Int::from_words(b.to_words());

        // v₀ = (a − d) / 2; v₁ = (b − c) / 2. Parity guarantees these
        // are exact integers. Use Int's signed arithmetic shift right
        // (preserves sign).
        let v0_num = a_w.wrapping_sub(&d_w);
        let v1_num = b_w.wrapping_sub(&c_w);
        let v0 = v0_num.shr_vartime(1);
        let v1 = v1_num.shr_vartime(1);

        return Some([v0, v1, c_w, d_w]);
    }
    // Search budget exhausted.
    let _ = zero_u; // suppress unused (kept for readability of constants)
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify `N_red(β) = M` at the same LIMBS via `reduced_norm_o0_basis_wide`.
    fn verify_norm<const LIMBS: usize>(
        beta: &[Int<LIMBS>; 4],
        p: &Uint<LIMBS>,
        expected_m: &Uint<LIMBS>,
    ) {
        let n: Int<LIMBS> =
            crate::quaternion::o0_mul::reduced_norm_o0_basis_wide::<LIMBS, LIMBS>(beta, p);
        let n_abs = n.abs();
        assert_eq!(
            &n_abs, expected_m,
            "verify_norm: N_red(β) ≠ expected M (got {n_abs:?}, want {expected_m:?})",
        );
    }

    #[test]
    fn represent_integer_wide_small_m_at_fake_prime_finds_beta() {
        // Small-scale test: at fake prime p = 7, target M = 5.
        // A known solution is β = 2 + i (standard quaternion coords
        // (2, 1, 0, 0)): N(β) = 4 + 1 = 5. In O_0 coords, β = 2·1 +
        // 1·i + 0·(i+j)/2 + 0·(1+k)/2 = (2, 1, 0, 0).
        //
        // The wide path samples random (c, d), so the exact β
        // returned depends on the seed — but the contract is:
        // whatever β comes back must have N_red(β) = 5.
        use crate::rng::NistPqcRng;
        use crypto_bigint::Uint;
        let p: Uint<8> = Uint::from_u64(7);
        let m: Uint<8> = Uint::from_u64(5);
        // Miller-Rabin witnesses {2, 3, 5, 7, 11} suffice for primes < 3·10¹⁴.
        let witnesses: [Uint<8>; 5] = [
            Uint::from_u64(2),
            Uint::from_u64(3),
            Uint::from_u64(5),
            Uint::from_u64(7),
            Uint::from_u64(11),
        ];
        let mut rng = NistPqcRng::new(&[0x69u8; 48]);
        let beta = find_quaternion_in_full_order_with_norm_wide::<8, _>(
            &m, &p, 5, 4096, &witnesses, &mut rng,
        )
        .expect("wide β-finder must locate a β with N(β) = 5 at p = 7");
        verify_norm(&beta, &p, &m);
    }

    #[test]
    fn represent_integer_wide_returns_none_on_zero_budget() {
        // max_trials = 0 → exhausts before any sample → None.
        use crate::rng::NistPqcRng;
        use crypto_bigint::Uint;
        let p: Uint<8> = Uint::from_u64(7);
        let m: Uint<8> = Uint::from_u64(5);
        let witnesses: [Uint<8>; 2] = [Uint::from_u64(2), Uint::from_u64(3)];
        let mut rng = NistPqcRng::new(&[0x00u8; 48]);
        assert_eq!(
            find_quaternion_in_full_order_with_norm_wide::<8, _>(
                &m, &p, 5, 0, &witnesses, &mut rng,
            ),
            None,
            "zero budget must yield None",
        );
    }

    #[test]
    fn represent_integer_wide_finds_beta_for_several_small_m() {
        // Sweep representable M values at fake prime p = 7. For each,
        // the wide path must return a β whose reduced norm matches.
        // Skip M ≤ 2 because the algorithm needs `4M > p` to have any
        // valid (c, d) pair with `T = 4M − p·(c² + d²) > 0` and `T ≡ 1
        // (mod 4)`. At real-prime scale M ≫ p so this constraint is
        // never tight; it only bites at this fake-prime smoke-test
        // boundary.
        use crate::rng::NistPqcRng;
        use crypto_bigint::Uint;
        let p: Uint<8> = Uint::from_u64(7);
        let witnesses: [Uint<8>; 5] = [
            Uint::from_u64(2),
            Uint::from_u64(3),
            Uint::from_u64(5),
            Uint::from_u64(7),
            Uint::from_u64(11),
        ];
        for &m_val in &[5u64, 9, 13] {
            let m: Uint<8> = Uint::from_u64(m_val);
            let seed_byte = u8::try_from(m_val).expect("m_val ≤ 255 in this sweep");
            let mut rng = NistPqcRng::new(&[seed_byte; 48]);
            let beta = find_quaternion_in_full_order_with_norm_wide::<8, _>(
                &m, &p, 5, 8192, &witnesses, &mut rng,
            );
            assert!(
                beta.is_some(),
                "wide β-finder failed at M = {m_val}, p = 7 — representable but no witness within search budget",
            );
            verify_norm(&beta.expect("checked above"), &p, &m);
        }
    }

    #[test]
    fn represent_integer_wide_returns_none_when_m_below_4p_boundary() {
        // S69 boundary marker: M = 1 at p = 7 has 4M = 4 < p, so the
        // only candidate (c, d) = (0, 0) gives T = 4 which fails
        // `T ≡ 1 (mod 4)`. The search must exhaust and return None
        // cleanly (not panic, not loop forever). This documents the
        // algorithm's small-input boundary as an EXPLICIT contract
        // rather than a bug.
        use crate::rng::NistPqcRng;
        use crypto_bigint::Uint;
        let p: Uint<8> = Uint::from_u64(7);
        let m: Uint<8> = Uint::from_u64(1);
        let witnesses: [Uint<8>; 2] = [Uint::from_u64(2), Uint::from_u64(3)];
        let mut rng = NistPqcRng::new(&[0x01u8; 48]);
        assert_eq!(
            find_quaternion_in_full_order_with_norm_wide::<8, _>(
                &m, &p, 5, 1024, &witnesses, &mut rng,
            ),
            None,
            "M = 1 at p = 7 is below the 4M > p boundary — search must exhaust cleanly",
        );
    }
}
