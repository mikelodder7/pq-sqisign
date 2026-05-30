// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(rustdoc::private_intra_doc_links)]
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

use crate::error::{Error, Result};
use crate::quaternion::algebra::Quaternion;
use crate::quaternion::cornacchia::cornacchia_classical_uint;
use crate::quaternion::ideal::LeftIdeal;
use crate::quaternion::o0_mul::{
    left_ideal_from_element_and_integer_o0, multiply_o0_basis, reduced_norm_o0_basis,
    standard_to_o0_basis, uint_as_nonneg_int,
};
use crate::quaternion::primality::is_probable_prime_with_witnesses;
use crate::quaternion::sample::sample_random_quaternion_o0;
use crate::quaternion::sqrt_mod::tonelli_shanks_uint;

/// Variable-time Euclidean GCD on `Uint<LIMBS>`. Used to test coprimality
/// of a freshly sampled re-randomizer's norm against the target ideal norm.
///
/// **Variable-time** on both inputs. The orchestrator's call pattern
/// passes `a = N_red(gen_rerand)` (derived from a freshly sampled
/// quaternion; thus partly secret-correlated through the sampling
/// state) and `b = norm` (the public ideal norm). Per the SQIsign 2.0
/// spec §8 convention, quaternion-side variable-time arithmetic is
/// acceptable; the constant-time discipline lives at the isogeny /
/// curve layer.
pub(crate) fn uint_gcd_vartime<const LIMBS: usize>(
    a: &Uint<LIMBS>,
    b: &Uint<LIMBS>,
) -> Uint<LIMBS> {
    let zero = Uint::<LIMBS>::from_u64(0);
    let mut x = *a;
    let mut y = *b;
    while y != zero {
        let y_nz: NonZero<Uint<LIMBS>> = NonZero::new(y).expect("loop guarded by y != 0");
        let r = x.rem_vartime(&y_nz);
        x = y;
        y = r;
    }
    x
}

/// Sample a `Uint<LIMBS>` uniformly in `[0, n)` via **rejection sampling**
/// — bias-free, matching the convention of [`super::sample::sample_random_quaternion_o0`].
///
/// Algorithm: draw `LIMBS` words of raw bytes, accept iff the value is in
/// the largest multiple-of-`n` range below `2^(64·LIMBS)`; reject and
/// resample otherwise. The acceptance threshold is `2^(64·LIMBS) -
/// (2^(64·LIMBS) mod n)`. When `n` divides `2^(64·LIMBS)` (e.g. `n` is
/// a power of two at most `2^(64·LIMBS)`), the threshold equals `2^(64·LIMBS)`
/// and every draw is accepted with no rejection — the orchestrator's
/// `remain` parameter (a power of two) is the common case for this fast path.
///
/// Expected iterations: `2^(64·LIMBS) / threshold ≈ 1` at sane LIMBS
/// (the rejection probability is `(2^(64·LIMBS) mod n) / 2^(64·LIMBS)`,
/// cryptographically negligible at signing-prime scales).
///
/// Variable-time on the rejection-loop's iteration count; intended for
/// non-secret moduli. Requires `n > 0`.
///
/// Security-review S187 Finding 3: earlier versions reduced modulo `n`
/// without rejection, introducing a small (but non-zero) modulo bias.
/// This version is bias-free.
fn sample_uint_lt_vartime<const LIMBS: usize, R: CryptoRng>(
    rng: &mut R,
    n: &Uint<LIMBS>,
) -> Uint<LIMBS> {
    let n_nz: NonZero<Uint<LIMBS>> = NonZero::new(*n).expect("caller ensures n > 0");
    let zero_u = Uint::<LIMBS>::ZERO;
    let one_u = Uint::<LIMBS>::ONE;
    let max = Uint::<LIMBS>::MAX;

    // `2^(64·LIMBS) mod n = (max + 1) mod n = (max_mod_n + 1) mod n`.
    // Call this value `bias_floor`.
    //   - If `bias_floor == 0`, n divides `2^(64·LIMBS)` exactly and no
    //     rejection is needed (all draws accept).
    //   - Else the acceptance range is `[0, max - bias_floor + 1)`, i.e.
    //     accept iff `r <= max - bias_floor` (== threshold_minus_one).
    let max_mod_n = max.rem_vartime(&n_nz);
    let bias_floor = max_mod_n.wrapping_add(&one_u).rem_vartime(&n_nz);
    let all_accept = bias_floor == zero_u;
    let threshold_minus_one = max.wrapping_sub(&bias_floor);

    loop {
        let mut words = [0u64; LIMBS];
        for w in words.iter_mut() {
            let mut buf = [0u8; 8];
            rng.fill_bytes(&mut buf);
            *w = u64::from_le_bytes(buf);
        }
        let r = Uint::<LIMBS>::from_words(words);
        if all_accept || r <= threshold_minus_one {
            return r.rem_vartime(&n_nz);
        }
        // Otherwise: rejected (top bias-window draw). Loop.
    }
}

/// Sample a `Uint<LIMBS>` uniformly in `[1, n]` by sampling `[0, n)` and
/// shifting by +1. Bias inherits from `sample_uint_lt_vartime`.
fn sample_uint_in_one_to_n_vartime<const LIMBS: usize, R: CryptoRng>(
    rng: &mut R,
    n: &Uint<LIMBS>,
) -> Uint<LIMBS> {
    sample_uint_lt_vartime::<LIMBS, R>(rng, n).wrapping_add(&Uint::<LIMBS>::ONE)
}

/// Narrow an `Int<WIDE>` to `Int<8>`. Returns `None` when the high limbs
/// are not the sign-extension of the low limbs, or when bit 511 of the
/// truncated value would flip the sign relative to the original.
pub(crate) fn narrow_int_to_8<const WIDE: usize>(x: &Int<WIDE>) -> Option<Int<8>> {
    if WIDE <= 8 {
        return Some(x.resize::<8>());
    }
    let words = x.to_words();
    let sign_word = if bool::from(x.is_negative()) {
        u64::MAX
    } else {
        0u64
    };
    for w in &words[8..] {
        if *w != sign_word {
            return None;
        }
    }
    let top_bit = (words[7] >> 63) & 1;
    let sign_bit = u64::from(sign_word != 0);
    if top_bit != sign_bit {
        return None;
    }
    let mut narrow_words = [0u64; 8];
    narrow_words.copy_from_slice(&words[..8]);
    Some(Int::<8>::from_words(narrow_words))
}

/// Narrow a `Uint<WIDE>` to `Uint<8>`. Returns `None` if any high limb
/// (index ≥ 8) is non-zero.
pub(crate) fn narrow_uint_to_8<const WIDE: usize>(x: &Uint<WIDE>) -> Option<Uint<8>> {
    if WIDE <= 8 {
        return Some(x.resize::<8>());
    }
    let words = x.to_words();
    for w in &words[8..] {
        if *w != 0 {
            return None;
        }
    }
    let mut narrow_words = [0u64; 8];
    narrow_words.copy_from_slice(&words[..8]);
    Some(Uint::<8>::from_words(narrow_words))
}

/// Narrow a `LeftIdeal<WIDE>` to `LeftIdeal<8>`. Returns `None` if any
/// basis cell, `denom`, or `cached_norm` overflows the narrow type.
#[allow(clippy::needless_range_loop)]
pub(crate) fn narrow_left_ideal_to_8<const WIDE: usize>(
    wide: &LeftIdeal<WIDE>,
) -> Option<LeftIdeal<8>> {
    let mut basis = [[Int::<8>::from_i64(0); 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            basis[r][c] = narrow_int_to_8::<WIDE>(&wide.basis[r][c])?;
        }
    }
    let denom = narrow_uint_to_8::<WIDE>(&wide.denom)?;
    let cached_norm = narrow_uint_to_8::<WIDE>(&wide.cached_norm)?;
    Some(LeftIdeal::with_denom_and_norm(basis, denom, cached_norm))
}

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

/// Sample a random `O_0`-left-ideal of given norm, at `Uint<LIMBS>` precision.
///
/// **Stub — S178 ships signature only; body deferred to S179+.** This is
/// the next layer above [`find_quaternion_in_full_order_with_norm_wide`]
/// (the `quat_represent_integer` port). It mirrors the C reference
/// `quat_sampling_random_ideal_O0_given_norm` in
/// `src/quaternion/ref/generic/normeq.c:257` of the SQIsign repo.
///
/// # Algorithm (deferred body)
///
/// **Step 1 — find a generator `gen ∈ O_0` with `norm | N_red(gen)`:**
///
/// - **Fast path (`is_prime == true`):** sample a uniformly random
///   trace-zero quaternion `gen` (coord[0] = 0; coord[1..4] in
///   `[0, norm)`); compute `n = N_red(gen)`; check whether `−n` is a
///   square mod `norm` via `sqrt_mod`; on success, set `gen.coord[0]`
///   to the square root and continue. Loop until found.
/// - **General path (`is_prime == false`):** require `prime_cofactor`
///   is `Some`. Set `target = prime_cofactor · norm`; call
///   [`find_quaternion_in_full_order_with_norm_wide`] with `target`;
///   the resulting `β` has `norm | N_red(β)` by construction.
///
/// **Step 2 — re-randomize the ideal class:** sample
/// `gen_rerand` with all 4 coords uniform in `[1, norm]`; require
/// `gcd(N_red(gen_rerand), norm) == 1`; set
/// `gen ← gen · gen_rerand` (quaternion multiplication). The product
/// `gen` is still in `O_0` (closed under multiplication) but
/// re-randomized in its left-equivalence class — preventing the output
/// ideal from leaking structural information about `represent_integer`'s
/// internal sampling.
///
/// **Step 3 — build the ideal:** construct
/// `lideal = O_0 · gen + O_0 · norm` (the standard left-ideal
/// generated-by-element-and-integer construction); verify
/// `N(lideal) == norm`; return `lideal`.
///
/// # Parameters
///
/// - `norm`: target norm of the output ideal.
/// - `is_prime`: hint that `norm` is prime. The fast path is ~10× faster
///   than the general path; callers must NOT pass `true` unless `norm`
///   is genuinely prime (Miller-Rabin probable-prime is sufficient).
/// - `prime_cofactor`: required when `is_prime == false`. A prime
///   distinct from `p` of similar size, coprime to `norm`. Mirrors
///   the C reference's `prime_cofactor` parameter. Must be `None` when
///   `is_prime == true` (the fast path doesn't use it).
/// - `sample_bound`, `max_trials`, `witnesses`, `rng`: forwarded to the
///   `represent_integer` call in the general path; unused on the fast
///   path. Keeping them in the signature keeps the API stable across
///   both modes.
///
/// # Returns
///
/// - `Ok(LeftIdeal<8>)`: an `O_0`-ideal with cached norm equal to `norm`,
///   re-randomized in its left-equivalence class.
/// - `Err(Error::Unimplemented)`: stub — body deferred to a follow-up
///   session.
///
/// # Precision contract
///
/// Same as [`find_quaternion_in_full_order_with_norm_wide`]: caller's
/// `LIMBS` must satisfy `64·LIMBS ≥ 2·bits(p) + 1` for the underlying
/// Cornacchia call. Output `LeftIdeal<8>` is at narrow precision —
/// signing-flow norms fit in `Uint<8>` at L1/L3/L5; a wide-cached-norm
/// variant (`_wn` suffix) can be added later if real-prime signing
/// products exceed the `Uint<8>` ceiling.
/// Finalize a random ideal: given a wide `gen` in O_0 coords that satisfies
/// `norm | N_red(gen)`, re-randomize the left-equivalence class via
/// multiplication by a fresh `gen_rerand` (coprime to norm), build
/// `O_0 · (gen · gen_rerand) + O_0 · norm` at wide precision, then narrow
/// to `LeftIdeal<8>`. Shared between the fast and general paths so the
/// re-randomization + ideal-construction + narrowing pipeline lives in
/// exactly one place.
fn finalize_random_ideal_o0<const LIMBS: usize, R: CryptoRng>(
    gen_wide: &[Int<LIMBS>; 4],
    norm: &Uint<LIMBS>,
    p: &Uint<LIMBS>,
    max_trials: usize,
    rng: &mut R,
) -> Result<LeftIdeal<8>> {
    let one_u = Uint::<LIMBS>::ONE;

    // Step A: sample gen_rerand with all 4 O_0 coords uniform in [1, norm];
    // reject until gcd(N_red(gen_rerand), norm) == 1. The retry loop is
    // bounded by max_trials so a degenerate input (e.g. tiny norm where
    // every sample shares a factor) surfaces as Err rather than hanging.
    let gen_rerand: [Int<LIMBS>; 4] = {
        let mut found: Option<[Int<LIMBS>; 4]> = None;
        for _ in 0..max_trials {
            let r_coords = [
                sample_uint_in_one_to_n_vartime::<LIMBS, R>(rng, norm),
                sample_uint_in_one_to_n_vartime::<LIMBS, R>(rng, norm),
                sample_uint_in_one_to_n_vartime::<LIMBS, R>(rng, norm),
                sample_uint_in_one_to_n_vartime::<LIMBS, R>(rng, norm),
            ];
            // Safe-reinterpret each [1, norm] Uint as a non-negative Int.
            // Precision contract: norm < 2^(64·LIMBS − 1) (general-path
            // bound), so top bit is zero by construction. Use the
            // centralized helper for defense in depth against future
            // precision-contract changes.
            let candidate: [Int<LIMBS>; 4] = [
                uint_as_nonneg_int::<LIMBS>(&r_coords[0])
                    .expect("gen_rerand coord 0 fits non-negative Int — precision contract"),
                uint_as_nonneg_int::<LIMBS>(&r_coords[1])
                    .expect("gen_rerand coord 1 fits non-negative Int — precision contract"),
                uint_as_nonneg_int::<LIMBS>(&r_coords[2])
                    .expect("gen_rerand coord 2 fits non-negative Int — precision contract"),
                uint_as_nonneg_int::<LIMBS>(&r_coords[3])
                    .expect("gen_rerand coord 3 fits non-negative Int — precision contract"),
            ];
            let n_int = reduced_norm_o0_basis::<LIMBS>(&candidate, p);
            let n_abs = n_int.abs();
            let g = uint_gcd_vartime::<LIMBS>(&n_abs, norm);
            if g == one_u {
                found = Some(candidate);
                break;
            }
        }
        found.ok_or(Error::Internal(
            "finalize_random_ideal_o0: no coprime re-randomizer within max_trials",
        ))?
    };

    // Step B: gen ← gen · gen_rerand in O_0 basis. O_0 is closed under
    // multiplication; the product remains in O_0 with reduced norm
    // N_red(gen) · N_red(gen_rerand). Divisibility `norm | N_red(product)`
    // follows from `norm | N_red(gen)` (caller invariant).
    let gen_combined: [Int<LIMBS>; 4] = multiply_o0_basis::<LIMBS>(gen_wide, &gen_rerand, p);

    // Step C: build O_0 · gen_combined + O_0 · norm at wide precision.
    // The helper sets cached_norm = norm; correct because
    // gcd(N_red(gen_rerand), norm) == 1 guarantees the multiplication
    // preserves the norm-divisibility class of the original gen.
    let wide_ideal = left_ideal_from_element_and_integer_o0::<LIMBS>(&gen_combined, norm, p);

    // Step D: narrow to LeftIdeal<8> per the public return contract.
    narrow_left_ideal_to_8::<LIMBS>(&wide_ideal).ok_or(Error::Internal(
        "finalize_random_ideal_o0: wide LeftIdeal exceeds Uint<8> ceiling — precision contract violated",
    ))
}

/// Sample a random `O_0`-left-ideal of given norm, at `Uint<LIMBS>` precision.
///
/// Mirrors the C reference `quat_sampling_random_ideal_O0_given_norm` in
/// `src/quaternion/ref/generic/normeq.c:257` of the SQIsign repo. Two
/// paths, both yielding a `LeftIdeal<8>` with `cached_norm = norm`
/// re-randomized in its left-equivalence class:
///
/// - **Fast path (`is_prime == true`):** sample standard-basis trace-zero
///   `(0, b, c, d)` with `b, c, d ∈ [0, norm)`, compute `n_temp = b² +
///   p·(c² + d²)`, find `a` with `a² ≡ -n_temp (mod norm)` via
///   [`tonelli_shanks_uint`].
///   Reject when `-n_temp` is a quadratic non-residue (~half of attempts
///   on prime norm). Lift to O_0 coords via `standard_to_o0_basis`.
///   Requires `norm` prime; on composite `norm` the sqrt step never
///   succeeds and the loop exhausts `max_trials`.
/// - **General path (`is_prime == false`):** require `prime_cofactor` is
///   `Some`; set `target = prime_cofactor · norm`; call
///   [`find_quaternion_in_full_order_with_norm_wide`] to obtain `gen`
///   with `N_red(gen) = target`, hence `norm | N_red(gen)` by
///   construction.
///
/// Both paths converge in [`finalize_random_ideal_o0`] which
/// re-randomizes the equivalence class (sampling `gen_rerand` until
/// `gcd(N_red(gen_rerand), norm) == 1`, then `gen ← gen · gen_rerand`)
/// and builds the ideal `O_0 · gen + O_0 · norm` via
/// [`left_ideal_from_element_and_integer_o0`],
/// finally narrowing the wide `LeftIdeal<LIMBS>` to the public
/// `LeftIdeal<8>` return contract.
///
/// # Parameters
///
/// - `norm`: target reduced norm of the output ideal. Must be ≥ 2.
/// - `p`: the base prime (passed through to every reduced-norm
///   computation).
/// - `is_prime`: hint that `norm` is prime. The fast path is ~10× faster
///   than the general path; callers must NOT pass `true` unless `norm`
///   is genuinely prime.
/// - `prime_cofactor`: required when `is_prime == false`. A prime
///   distinct from `p` of similar size, coprime to `norm`. Must be
///   `None` when `is_prime == true`.
/// - `sample_bound`, `witnesses`: forwarded to the general path's wide
///   finder; unused on the fast path (the fast path doesn't need
///   Cornacchia or Miller-Rabin).
/// - `max_trials`: bounds both the gen-finding loop and the
///   re-randomizer retry loop.
///
/// # Returns
///
/// - `Ok(LeftIdeal<8>)`: an `O_0`-ideal with `cached_norm == norm` and
///   `denom == 1`.
/// - `Err(Error::Internal)`: validation failure, budget exhausted, or
///   precision-contract violation.
///
/// # Precision contract
///
/// Both paths share the general-path bound `64·LIMBS ≥ 2·bits(p) + 1`
/// (the same as [`find_quaternion_in_full_order_with_norm_wide`]).
/// The fast path's inner computation reduces `mod norm` at every
/// multiplication so intermediates stay in `[0, norm²) ⊂ [0,
/// 2^(2·bits(p)))` and fit within the general-path LIMBS budget.
///
/// (An earlier S180 formulation built the full `b² + p·(c² + d²)`
/// before reducing, which required `64·LIMBS ≥ 3·bits(p) + 2` and
/// would silently wrap at the general-path bound. The security
/// review S187 Finding 2 surfaced that as a latent corruption
/// surface; the current implementation eliminates it.)
///
/// The output's narrow `LeftIdeal<8>` is at the signing-flow norm
/// width; a wide-cached-norm variant (`_wn` suffix) can be added
/// later if real-prime signing products exceed `Uint<8>`.
#[allow(clippy::too_many_arguments)]
pub fn sampling_random_ideal_o0_given_norm_wide<const LIMBS: usize, R: CryptoRng>(
    norm: &Uint<LIMBS>,
    p: &Uint<LIMBS>,
    is_prime: bool,
    prime_cofactor: Option<&Uint<LIMBS>>,
    sample_bound: i64,
    max_trials: usize,
    witnesses: &[Uint<LIMBS>],
    rng: &mut R,
) -> Result<LeftIdeal<8>> {
    let zero_u = Uint::<LIMBS>::from_u64(0);
    let one_u = Uint::<LIMBS>::ONE;

    if *norm == zero_u || *norm == one_u {
        return Err(Error::Internal(
            "sampling_random_ideal_o0_given_norm_wide: norm must be >= 2",
        ));
    }

    // Step 1: find gen ∈ O_0 with norm | N_red(gen). Fast and general paths
    // diverge here; both converge into `finalize_random_ideal_o0` below.
    let gen_wide: [Int<LIMBS>; 4] = if is_prime {
        // Fast path: sample standard-basis (0, b, c, d) with b, c, d ∈
        // [0, norm). The reduced norm of this trace-zero quaternion is
        // n_temp = b² + p·(c² + d²). Solving N_red(gen) ≡ 0 (mod norm)
        // for the standard-basis Re-component a gives a² ≡ -n_temp (mod
        // norm), i.e. a = sqrt(-n_temp mod norm). When that sqrt exists
        // (-n_temp is a QR mod norm — ~half the time for prime norm),
        // the resulting gen = (a, b, c, d) has N_red ≡ 0 (mod norm).
        //
        // Lift standard → O_0 with the canonical map
        //   v_0 = a - d, v_1 = b - c, v_2 = 2c, v_3 = 2d
        // (every integer standard-basis element lifts to integer O_0
        // coords; no parity post-selection needed).
        //
        // Requires `norm` prime. `tonelli_shanks_uint` returns None on
        // composite-modulus / non-QR inputs; both surface as retry.
        let _ = (sample_bound, witnesses); // unused on fast path; routed to general path below.
        let norm_nz: NonZero<Uint<LIMBS>> = NonZero::new(*norm).expect("norm >= 2 above");
        let mut found: Option<[Int<LIMBS>; 4]> = None;
        for _ in 0..max_trials {
            let b_u = sample_uint_lt_vartime::<LIMBS, R>(rng, norm);
            let c_u = sample_uint_lt_vartime::<LIMBS, R>(rng, norm);
            let d_u = sample_uint_lt_vartime::<LIMBS, R>(rng, norm);

            // Compute `n_temp_mod = (b² + p·(c² + d²)) mod norm`, reducing
            // mod norm at every multiplication so intermediates stay in
            // `[0, norm²) ⊂ [0, 2^(2·bits(norm))) ⊂ [0, 2^(2·bits(p)))`.
            // This keeps the fast path within the general-path precision
            // contract `64·LIMBS ≥ 2·bits(p) + 1` — the earlier
            // formulation that built the full `b² + p·(c² + d²)` first
            // and reduced afterward required `64·LIMBS ≥ 3·bits(p) + 2`
            // and would silently `wrapping_mul`-overflow at the
            // general-path LIMBS bound. Per the security review (S187
            // Finding 2): no public API change needed once we reduce at
            // each step.
            //
            // Each of `b_u, c_u, d_u` is already in `[0, norm)` per the
            // sampler contract, so the first multiplications give values
            // < `norm²` and fit. `p_mod` is `p mod norm` (≤ norm−1).
            let b_sq_mod = b_u.wrapping_mul(&b_u).rem_vartime(&norm_nz);
            let c_sq_mod = c_u.wrapping_mul(&c_u).rem_vartime(&norm_nz);
            let d_sq_mod = d_u.wrapping_mul(&d_u).rem_vartime(&norm_nz);
            let cd_sq_mod = c_sq_mod.wrapping_add(&d_sq_mod).rem_vartime(&norm_nz);
            let p_mod = p.rem_vartime(&norm_nz);
            let p_cd_mod = p_mod.wrapping_mul(&cd_sq_mod).rem_vartime(&norm_nz);
            let n_temp_mod = b_sq_mod.wrapping_add(&p_cd_mod).rem_vartime(&norm_nz);

            // t = (-n_temp) mod norm = (norm - n_temp_mod) when n_temp_mod ≠ 0.
            let t = if n_temp_mod == zero_u {
                zero_u
            } else {
                norm.wrapping_sub(&n_temp_mod)
            };

            // sqrt(t) mod norm via Tonelli-Shanks. None on QNR — retry.
            //
            // **Composite-modulus safety**: Tonelli-Shanks is well-defined
            // ONLY on prime moduli. On composite `norm`, Euler's criterion
            // gives CRT false-positives (`a^((n−1)/2) ≡ 1 (mod n)` can hold
            // for non-square `a`), and the `n ≡ 3 (mod 4)` fast path
            // returns `a^((n+1)/4) mod n` which is NOT a square root in
            // general for composite `n`. To stay safe when a caller
            // wrongly passes `is_prime=true` on composite `norm`, we
            // re-verify `r² ≡ t (mod norm)` after the call. A failure here
            // is treated identically to QNR — `continue` the loop. If
            // every iteration fails (e.g. norm composite throughout),
            // the budget exhausts and we return `Err(Internal)` rather
            // than producing a corrupt `gen`.
            let a_u = match tonelli_shanks_uint::<LIMBS>(&t, &norm_nz) {
                Some(a) => {
                    let a_sq = a.wrapping_mul(&a);
                    let a_sq_mod = a_sq.rem_vartime(&norm_nz);
                    if a_sq_mod != t {
                        // tonelli_shanks returned a non-square — composite
                        // modulus or numerical fault. Reject and retry.
                        continue;
                    }
                    a
                }
                None => continue,
            };

            // Standard (a, b, c, d) → O_0 coords via Quaternion +
            // standard_to_o0_basis. Safe-reinterpret unsigned magnitudes
            // as non-negative Ints via the centralized helper; precision
            // contract `64·LIMBS ≥ 2·bits(p) + 1` ensures top bit is clear.
            let a_i: Int<LIMBS> = uint_as_nonneg_int::<LIMBS>(&a_u)
                .expect("fast-path a fits non-negative Int — precision contract");
            let b_i: Int<LIMBS> = uint_as_nonneg_int::<LIMBS>(&b_u)
                .expect("fast-path b fits non-negative Int — precision contract");
            let c_i: Int<LIMBS> = uint_as_nonneg_int::<LIMBS>(&c_u)
                .expect("fast-path c fits non-negative Int — precision contract");
            let d_i: Int<LIMBS> = uint_as_nonneg_int::<LIMBS>(&d_u)
                .expect("fast-path d fits non-negative Int — precision contract");
            let q_std = Quaternion::<LIMBS>::new(a_i, b_i, c_i, d_i);
            let gen_o0 = standard_to_o0_basis::<LIMBS>(&q_std);

            // Invariant verification: the algorithm guarantees `norm |
            // N_red(gen_o0)`. In debug builds verify; in release trust
            // the math.
            #[cfg(debug_assertions)]
            {
                let n_check = reduced_norm_o0_basis::<LIMBS>(&gen_o0, p).abs();
                let rem = n_check.rem_vartime(&norm_nz);
                debug_assert_eq!(
                    rem, zero_u,
                    "fast-path post-sqrt invariant violated: norm must divide N_red(gen_o0)",
                );
            }

            found = Some(gen_o0);
            break;
        }
        found.ok_or(Error::Internal(
            "sampling_random_ideal_o0_given_norm_wide: fast path exhausted max_trials without locating a QR step (norm may not be prime, or budget too small)",
        ))?
    } else {
        let cofactor = prime_cofactor.ok_or(Error::Internal(
            "sampling_random_ideal_o0_given_norm_wide: prime_cofactor required when is_prime == false",
        ))?;
        // target = prime_cofactor · norm at exact precision. checked_mul
        // surfaces overflow as Err(Internal) so the caller cannot silently
        // receive an ideal whose lattice norm diverges from the cached
        // `norm` field.
        let target = Option::<Uint<LIMBS>>::from(cofactor.checked_mul(norm)).ok_or(
            Error::Internal(
                "sampling_random_ideal_o0_given_norm_wide: prime_cofactor·norm overflows Uint<LIMBS> — caller must size LIMBS so the product fits",
            ),
        )?;
        find_quaternion_in_full_order_with_norm_wide::<LIMBS, R>(
            &target,
            p,
            sample_bound,
            max_trials,
            witnesses,
            rng,
        )
        .ok_or(Error::Internal(
            "sampling_random_ideal_o0_given_norm_wide: wide finder exhausted search budget",
        ))?
    };
    let _ = one_u; // suppress unused when both paths handle the constant inline.

    // Step 2: shared finalize — re-randomize, build O_0·gen+O_0·norm, narrow.
    finalize_random_ideal_o0::<LIMBS, R>(&gen_wide, norm, p, max_trials, rng)
}

#[cfg(all(test, feature = "kat"))]
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

    // sampling_random_ideal_o0_given_norm_wide tests:
    //   - fast path (is_prime=true) returns Err(Unimplemented) pending
    //     C-ref basis-convention study
    //   - general path (is_prime=false, cofactor required) wires through
    //     the wide finder + re-randomization + 8-row HNF and returns a
    //     LeftIdeal<8> with cached_norm equal to the input norm

    fn small_witnesses_l1() -> [Uint<8>; 5] {
        [
            Uint::from_u64(2),
            Uint::from_u64(3),
            Uint::from_u64(5),
            Uint::from_u64(7),
            Uint::from_u64(11),
        ]
    }

    #[test]
    fn sampling_random_ideal_fast_path_at_fake_prime_returns_ideal_with_correct_norm() {
        // Fast-path exercise: p = 7 (fake L1), norm = 11 (prime).
        // The Tonelli-Shanks step requires norm prime, which holds here.
        // Asserts: cached_norm == 11, denom == 1, |det(basis)| == 121 = 11².
        use crate::rng::NistPqcRng;
        let norm: Uint<8> = Uint::from_u64(11);
        let p: Uint<8> = Uint::from_u64(7);
        let witnesses = small_witnesses_l1();
        let mut rng = NistPqcRng::new(&[0x42u8; 48]);
        let ideal = sampling_random_ideal_o0_given_norm_wide::<8, _>(
            &norm, &p, true, None, 5, 4096, &witnesses, &mut rng,
        )
        .expect("fast path must produce an ideal at p=7, norm=11");
        assert_eq!(ideal.cached_norm, norm);
        assert_eq!(ideal.denom, Uint::<8>::ONE);
        let det = crate::quaternion::ideal::det_4x4::<8>(&ideal.basis);
        assert_eq!(
            det.abs(),
            Uint::<8>::from_u64(121),
            "fast-path lattice index must equal norm² (= 121 for norm=11)",
        );
    }

    #[test]
    fn sampling_random_ideal_fast_path_safe_on_composite_norm() {
        // Correctness invariant under caller misuse: when `is_prime=true`
        // is passed on a composite `norm`, the post-tonelli re-verification
        // step (`a² ≡ t (mod norm)`) prevents any CRT-false-positive sqrt
        // from producing a corrupt gen. The algorithm may STILL succeed
        // (some compositions have legitimate QRs, e.g. `n_temp ≡ 0 mod
        // norm` makes `t = 0` whose sqrt is `0` — a valid QR for any
        // modulus). In that case the returned ideal must be geometrically
        // valid: `cached_norm == norm` AND `|det(basis)| == norm²` (the
        // lattice index relationship for a left `O_0`-ideal of reduced
        // norm `norm`). If instead the budget exhausts, we get
        // `Err(Internal)`. Either outcome is acceptable; corruption is not.
        use crate::rng::NistPqcRng;
        let norm: Uint<8> = Uint::from_u64(9); // 9 = 3² is composite
        let p: Uint<8> = Uint::from_u64(7);
        let witnesses = small_witnesses_l1();
        let mut rng = NistPqcRng::new(&[0xC0u8; 48]);
        let result = sampling_random_ideal_o0_given_norm_wide::<8, _>(
            &norm, &p, true, None, 5, 4096, &witnesses, &mut rng,
        );
        if let Ok(ideal) = &result {
            assert_eq!(ideal.cached_norm, norm);
            assert_eq!(ideal.denom, Uint::<8>::ONE);
            let det = crate::quaternion::ideal::det_4x4::<8>(&ideal.basis);
            assert_eq!(
                det.abs(),
                Uint::<8>::from_u64(81),
                "composite-norm result must have valid lattice index = norm² (81 for norm=9)",
            );
        } else {
            assert!(
                matches!(result, Err(Error::Internal(_))),
                "composite norm with is_prime=true must yield Ok(valid ideal) or Err(Internal), got {result:?}",
            );
        }
    }

    #[test]
    fn sampling_random_ideal_fast_path_two_seeds_both_have_correct_norm() {
        // Two independent seeds must both produce valid ideals of the
        // requested reduced norm at fake-L1 prime. Same small-norm
        // degeneracy caveat as the general-path version: at this scale
        // the canonical HNF basis can collapse, so no basis-distinctness
        // assertion.
        use crate::rng::NistPqcRng;
        let norm: Uint<8> = Uint::from_u64(11);
        let p: Uint<8> = Uint::from_u64(7);
        let witnesses = small_witnesses_l1();
        let mut rng_a = NistPqcRng::new(&[0xA1u8; 48]);
        let mut rng_b = NistPqcRng::new(&[0xB2u8; 48]);
        let a = sampling_random_ideal_o0_given_norm_wide::<8, _>(
            &norm, &p, true, None, 5, 4096, &witnesses, &mut rng_a,
        )
        .expect("seed A must produce ideal");
        let b = sampling_random_ideal_o0_given_norm_wide::<8, _>(
            &norm, &p, true, None, 5, 4096, &witnesses, &mut rng_b,
        )
        .expect("seed B must produce ideal");
        assert_eq!(a.cached_norm, norm);
        assert_eq!(b.cached_norm, norm);
        assert_eq!(a.denom, Uint::<8>::ONE);
        assert_eq!(b.denom, Uint::<8>::ONE);
    }

    #[test]
    fn sampling_random_ideal_rejects_norm_below_two() {
        use crate::rng::NistPqcRng;
        let p: Uint<8> = Uint::from_u64(7);
        let cofactor: Uint<8> = Uint::from_u64(13);
        let witnesses = small_witnesses_l1();
        let mut rng = NistPqcRng::new(&[0x01u8; 48]);
        // norm = 0 → reject.
        let zero_norm: Uint<8> = Uint::from_u64(0);
        let r0 = sampling_random_ideal_o0_given_norm_wide::<8, _>(
            &zero_norm,
            &p,
            false,
            Some(&cofactor),
            5,
            16,
            &witnesses,
            &mut rng,
        );
        assert!(
            matches!(r0, Err(Error::Internal(_))),
            "norm = 0 must yield Err(Internal), got {r0:?}",
        );
        // norm = 1 → reject (the only ideal of norm 1 is O_0 itself; this
        // sampler is not meaningful).
        let one_norm: Uint<8> = Uint::from_u64(1);
        let r1 = sampling_random_ideal_o0_given_norm_wide::<8, _>(
            &one_norm,
            &p,
            false,
            Some(&cofactor),
            5,
            16,
            &witnesses,
            &mut rng,
        );
        assert!(
            matches!(r1, Err(Error::Internal(_))),
            "norm = 1 must yield Err(Internal), got {r1:?}",
        );
    }

    #[test]
    fn sampling_random_ideal_general_path_rejects_missing_cofactor() {
        use crate::rng::NistPqcRng;
        let norm: Uint<8> = Uint::from_u64(15);
        let p: Uint<8> = Uint::from_u64(7);
        let witnesses = small_witnesses_l1();
        let mut rng = NistPqcRng::new(&[0x55u8; 48]);
        let result = sampling_random_ideal_o0_given_norm_wide::<8, _>(
            &norm, &p, false, None, 5, 4096, &witnesses, &mut rng,
        );
        assert!(
            matches!(result, Err(Error::Internal(_))),
            "general path with prime_cofactor=None must yield Err(Internal), got {result:?}",
        );
    }

    #[test]
    fn sampling_random_ideal_general_path_at_fake_prime_returns_ideal_with_correct_norm() {
        // Real general-path exercise at p = 7, norm = 9 (composite, coprime
        // to cofactor), cofactor = 13. The wide finder must locate a gen
        // with N_red = 13·9 = 117 at p = 7 (the small-M sweep covers this
        // range), then re-randomization + ideal construction must produce
        // a LeftIdeal<8> with cached_norm = 9.
        use crate::rng::NistPqcRng;
        let norm: Uint<8> = Uint::from_u64(9);
        let p: Uint<8> = Uint::from_u64(7);
        let cofactor: Uint<8> = Uint::from_u64(13);
        let witnesses = small_witnesses_l1();
        let mut rng = NistPqcRng::new(&[0x07u8; 48]);
        let ideal = sampling_random_ideal_o0_given_norm_wide::<8, _>(
            &norm,
            &p,
            false,
            Some(&cofactor),
            5,
            16384,
            &witnesses,
            &mut rng,
        )
        .expect("general path must produce an ideal at fake L1 inputs");
        assert_eq!(
            ideal.cached_norm, norm,
            "returned LeftIdeal cached_norm must equal input norm",
        );
        assert_eq!(
            ideal.denom,
            Uint::<8>::ONE,
            "general-path output is integer (denom == 1)",
        );
        // Independent lattice-index check: for an O_0-ideal of reduced
        // norm N, [O_0 : I] = N². The cached_norm field is set by the
        // helper without geometric verification, so this assertion is
        // the real correctness probe. At norm=9, expect |det(basis)|=81.
        let det = crate::quaternion::ideal::det_4x4::<8>(&ideal.basis);
        let expected_index = Uint::<8>::from_u64(81);
        assert_eq!(
            det.abs(),
            expected_index,
            "lattice index [O_0 : I] must equal norm² (= 81 for norm=9)",
        );
    }

    #[test]
    fn sampling_random_ideal_general_path_two_seeds_both_have_correct_norm() {
        // Two independent seeds must both produce valid ideals of the
        // requested reduced norm. We do NOT assert basis distinctness:
        // at small fake-prime scale (norm = 9) the number of left ideals
        // of `O_0` of reduced norm 9 is small, and re-randomization can
        // legitimately land at the same canonical HNF lattice across
        // different seeds. The cryptographic-scale invariant
        // (re-randomization spreads across many lattices) is testable
        // only at signing-prime scales — out of reach for a unit test.
        use crate::rng::NistPqcRng;
        let norm: Uint<8> = Uint::from_u64(9);
        let p: Uint<8> = Uint::from_u64(7);
        let cofactor: Uint<8> = Uint::from_u64(13);
        let witnesses = small_witnesses_l1();
        let mut rng_a = NistPqcRng::new(&[0xA1u8; 48]);
        let mut rng_b = NistPqcRng::new(&[0xB2u8; 48]);
        let a = sampling_random_ideal_o0_given_norm_wide::<8, _>(
            &norm,
            &p,
            false,
            Some(&cofactor),
            5,
            16384,
            &witnesses,
            &mut rng_a,
        )
        .expect("seed A must produce ideal");
        let b = sampling_random_ideal_o0_given_norm_wide::<8, _>(
            &norm,
            &p,
            false,
            Some(&cofactor),
            5,
            16384,
            &witnesses,
            &mut rng_b,
        )
        .expect("seed B must produce ideal");
        assert_eq!(a.cached_norm, norm);
        assert_eq!(b.cached_norm, norm);
        assert_eq!(a.denom, Uint::<8>::ONE);
        assert_eq!(b.denom, Uint::<8>::ONE);
    }

    #[test]
    fn sampling_random_ideal_fast_path_at_real_lvl1_prime() {
        // Exercise the fast path at the real Level-1 prime
        // `p = 5·2^248 − 1` (bits(p) = 251). Per the F1 precision contract
        // surfaced by S180's Forge audit, the fast path needs
        // `64·LIMBS ≥ 3·bits(p) + 2 = 755` at L1 → `LIMBS ≥ 12` for the
        // worst case where `norm ≈ p`. We use a small prime `norm = 11`
        // here (so the inner arithmetic stays well below the LIMBS=12
        // ceiling regardless), confirming the algorithm scales from the
        // fake `p = 7` to real `p ≈ 2^248` without latent bugs.
        //
        // Output ideal must satisfy the same invariants as at fake L1:
        // `cached_norm == 11`, `denom == 1`, `|det(basis)| == 121 = 11²`
        // — the lattice-index relationship for a left `O_0`-ideal of
        // reduced norm 11.
        use crate::rng::NistPqcRng;
        let p_narrow = crate::params::lvl1::prime();
        let p: Uint<12> = p_narrow.resize::<12>();
        let norm: Uint<12> = Uint::from_u64(11);
        let witnesses: [Uint<12>; 5] = [
            Uint::from_u64(2),
            Uint::from_u64(3),
            Uint::from_u64(5),
            Uint::from_u64(7),
            Uint::from_u64(11),
        ];
        let mut rng = NistPqcRng::new(&[0xE1u8; 48]);
        let ideal = sampling_random_ideal_o0_given_norm_wide::<12, _>(
            &norm, &p, true, None, 5, 4096, &witnesses, &mut rng,
        )
        .expect("fast path must succeed at real L1 prime with norm=11 and LIMBS=12");
        assert_eq!(ideal.cached_norm, Uint::<8>::from_u64(11));
        assert_eq!(ideal.denom, Uint::<8>::ONE);
        let det = crate::quaternion::ideal::det_4x4::<8>(&ideal.basis);
        assert_eq!(
            det.abs(),
            Uint::<8>::from_u64(121),
            "real-L1 fast-path lattice index must equal norm² (= 121 for norm=11)",
        );
    }

    #[test]
    fn sampling_random_ideal_fast_path_at_real_lvl3_prime() {
        // Exercise the fast path at the real Level-3 prime
        // `p = 65·2^376 − 1` (bits(p) = 383). Per the F1 precision contract
        // surfaced by S180's Forge audit, the fast path needs
        // `64·LIMBS ≥ 3·bits(p) + 2 = 1151` at L3 → `LIMBS ≥ 18` for the
        // worst case where `norm ≈ p`. With small `norm = 11` the inner
        // arithmetic stays well below the LIMBS=18 ceiling, but we use
        // LIMBS=18 to confirm the F1-mandated bound monomorphizes
        // correctly at L3 scale.
        use crate::rng::NistPqcRng;
        let p_narrow = crate::params::lvl3::prime();
        let p: Uint<18> = p_narrow.resize::<18>();
        let norm: Uint<18> = Uint::from_u64(11);
        let witnesses: [Uint<18>; 5] = [
            Uint::from_u64(2),
            Uint::from_u64(3),
            Uint::from_u64(5),
            Uint::from_u64(7),
            Uint::from_u64(11),
        ];
        let mut rng = NistPqcRng::new(&[0xE3u8; 48]);
        let ideal = sampling_random_ideal_o0_given_norm_wide::<18, _>(
            &norm, &p, true, None, 5, 4096, &witnesses, &mut rng,
        )
        .expect("fast path must succeed at real L3 prime with norm=11 and LIMBS=18");
        assert_eq!(ideal.cached_norm, Uint::<8>::from_u64(11));
        assert_eq!(ideal.denom, Uint::<8>::ONE);
        let det = crate::quaternion::ideal::det_4x4::<8>(&ideal.basis);
        assert_eq!(
            det.abs(),
            Uint::<8>::from_u64(121),
            "real-L3 fast-path lattice index must equal norm² (= 121 for norm=11)",
        );
    }

    #[test]
    fn sampling_random_ideal_fast_path_at_real_lvl5_prime() {
        // Exercise the fast path at the real Level-5 prime
        // `p = 27·2^500 − 1` (bits(p) = 505). Per the F1 precision contract
        // surfaced by S180's Forge audit, the fast path needs
        // `64·LIMBS ≥ 3·bits(p) + 2 = 1517` at L5 → `LIMBS ≥ 24` for the
        // worst case where `norm ≈ p`. With small `norm = 11` we use
        // LIMBS=24 to confirm the F1 bound monomorphizes correctly at L5
        // scale and to catch any latent overflow before the orchestrator
        // layer goes on top.
        use crate::rng::NistPqcRng;
        let p_narrow = crate::params::lvl5::prime();
        let p: Uint<24> = p_narrow.resize::<24>();
        let norm: Uint<24> = Uint::from_u64(11);
        let witnesses: [Uint<24>; 5] = [
            Uint::from_u64(2),
            Uint::from_u64(3),
            Uint::from_u64(5),
            Uint::from_u64(7),
            Uint::from_u64(11),
        ];
        let mut rng = NistPqcRng::new(&[0xE5u8; 48]);
        let ideal = sampling_random_ideal_o0_given_norm_wide::<24, _>(
            &norm, &p, true, None, 5, 4096, &witnesses, &mut rng,
        )
        .expect("fast path must succeed at real L5 prime with norm=11 and LIMBS=24");
        assert_eq!(ideal.cached_norm, Uint::<8>::from_u64(11));
        assert_eq!(ideal.denom, Uint::<8>::ONE);
        let det = crate::quaternion::ideal::det_4x4::<8>(&ideal.basis);
        assert_eq!(
            det.abs(),
            Uint::<8>::from_u64(121),
            "real-L5 fast-path lattice index must equal norm² (= 121 for norm=11)",
        );
    }

    #[test]
    fn sampling_random_ideal_fast_path_compiles_at_l3_l5_limb_counts() {
        // Generic monomorphization smoke at LIMBS = 12 (L3) and LIMBS = 16
        // (L5). Uses the fast path with small prime norm so the actual
        // body runs end-to-end at wider LIMBS, verifying every wide-LIMBS
        // code branch compiles and dispatches correctly. Asserts the
        // returned ideal's cached_norm matches the input.
        use crate::rng::NistPqcRng;

        let norm_l3: Uint<12> = Uint::from_u64(11);
        let p_l3: Uint<12> = Uint::from_u64(7);
        let witnesses_l3: [Uint<12>; 2] = [Uint::from_u64(2), Uint::from_u64(3)];
        let mut rng_l3 = NistPqcRng::new(&[0xAAu8; 48]);
        let r3 = sampling_random_ideal_o0_given_norm_wide::<12, _>(
            &norm_l3,
            &p_l3,
            true,
            None,
            5,
            4096,
            &witnesses_l3,
            &mut rng_l3,
        )
        .expect("L3 (LIMBS=12) fast path must produce ideal");
        assert_eq!(r3.cached_norm, Uint::<8>::from_u64(11));

        let norm_l5: Uint<16> = Uint::from_u64(11);
        let p_l5: Uint<16> = Uint::from_u64(7);
        let witnesses_l5: [Uint<16>; 2] = [Uint::from_u64(2), Uint::from_u64(3)];
        let mut rng_l5 = NistPqcRng::new(&[0xBBu8; 48]);
        let r5 = sampling_random_ideal_o0_given_norm_wide::<16, _>(
            &norm_l5,
            &p_l5,
            true,
            None,
            5,
            4096,
            &witnesses_l5,
            &mut rng_l5,
        )
        .expect("L5 (LIMBS=16) fast path must produce ideal");
        assert_eq!(r5.cached_norm, Uint::<8>::from_u64(11));
    }
}
