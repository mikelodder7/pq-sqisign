// SPDX-License-Identifier: MIT OR Apache-2.0
//! KLPT (Kohel-Lauter-Petit-Tignol) equivalent-ideal lift.
//!
//! Given an integral left `O_0`-ideal `I`, KLPT produces an equivalent
//! left `O_0`-ideal `J` (same left-ideal class, i.e. `J = I · α` for some
//! `α ∈ B^*`) whose norm `N(J)` is *smooth* — a product of small primes
//! and a 2-power. Smooth norms are essential because the downstream
//! `IdealToIsogeny` translation can only walk through curves at degrees
//! that fit the available `F_{p^2}`-rational torsion.
//!
//! The high-level algorithm structure:
//!
//! 1. Pick a target smooth norm `T` (typically `T = 2^e · ℓ_1 · …`).
//! 2. Find `γ ∈ I` with reduced norm `N(γ) = N · N(I)` for some `N` co-prime
//!    to `T` and small (this is the **norm-form solve** — handled by
//!    [`super::cornacchia::cornacchia`]).
//! 3. Compute `J = I · γ̄ / N(I)` and verify `N(J) = T`.
//! 4. If verification fails or `γ` could not be found, re-randomise the
//!    intermediate steps and retry.
//!
//! Faithful KLPT is ~2000 lines of integer-norm-form solving + lattice
//! reduction in the reference. This module currently provides the typed
//! signature, the smooth-target enumerator, and ties through to Cornacchia
//! — the full algorithm body lands across subsequent sessions.

use crypto_bigint::Uint;

use crate::error::{Error, Result};
use crate::quaternion::ideal::LeftIdeal;
use crate::quaternion::ideal_mul::ideal_right_multiply;
use crate::quaternion::norm_search::find_norm_witness;
use crate::quaternion::o0_mul::{principal_left_ideal_from_o0, standard_to_o0_basis};
use crate::quaternion::short_vec::find_quaternion_in_ideal_with_norm;

/// **Principal case** of KLPT: produce a left ideal `J = O_0 · γ` with
/// `N_red(γ) = target_reduced_norm`. The resulting ideal has norm
/// `target_reduced_norm²` (because `N(O_0·γ) = N_red(γ)²` for principal
/// ideals in a maximal order).
///
/// This is the simplest KLPT case — when the input ideal is `O_0` itself,
/// every equivalent ideal is principal, and the only thing to do is find
/// a `γ` of the desired reduced norm. The general (non-principal) KLPT
/// builds on this primitive plus a randomisation step.
///
/// Returns `None` if no quaternion of the requested reduced norm exists
/// within the brute-force search bound (lifted via [`find_norm_witness`]).
pub fn principal_ideal_with_reduced_norm(
    target_reduced_norm: u128,
    p: u128,
) -> Option<LeftIdeal<8>> {
    let gamma_std = find_norm_witness(target_reduced_norm, p)?;
    let gamma_o0 = standard_to_o0_basis(&gamma_std);
    let p_uint = Uint::<8>::from_u128(p);
    Some(principal_left_ideal_from_o0(&gamma_o0, &p_uint))
}

/// Lift `ideal` to an equivalent `O_0`-ideal of norm exactly `target`.
///
/// **Uniform composition (any input ideal)**: the right-multiplication
/// norm identity `N(I·β) = N(I) · N_red(β)²` means we can hit `target`
/// whenever `target = N(I) · m²` for some integer `m`. The algorithm:
///
/// 1. Compute `n = N(I)`.
/// 2. Check `target % n == 0` and `target / n` is a perfect square `m²`.
/// 3. Find `β ∈ O_0` with `N_red(β) = m` via [`find_norm_witness`].
/// 4. Return `I · β` via [`ideal_right_multiply`].
///
/// This handles every input ideal (principal or not). The strictness on
/// `target` — divisibility by `N(I)` and the residual being a perfect
/// square — is what KLPT's full algorithm relaxes via a *randomisation
/// over equivalent γ's* step. For the prototype, the caller picks `target`
/// compatible with the constraints.
/// Sweep smooth targets in `[target_low, target_high]` (ascending),
/// returning the first equivalent ideal `lift_to_smooth_norm` produces
/// alongside the target norm that succeeded.
///
/// Composes [`super::smooth::enumerate_smooth`] over the caller's smooth
/// prime set with [`lift_to_smooth_norm`]'s strict shape check. The
/// search ends at the first success; if no smooth target in range admits
/// a witness, returns `Error::Unimplemented` with the "exhausted" message.
#[cfg(feature = "alloc")]
pub fn lift_to_any_smooth_target(
    ideal: &LeftIdeal<8>,
    primes: &[u64],
    target_low: u128,
    target_high: u128,
) -> Result<(LeftIdeal<8>, u128)> {
    let candidates = crate::quaternion::smooth::enumerate_smooth(primes, target_high);
    for s in &candidates {
        if s.value < target_low {
            continue;
        }
        if let Ok(j) = lift_to_smooth_norm(ideal, s.value) {
            return Ok((j, s.value));
        }
    }
    Err(Error::Unimplemented(
        "lift_to_any_smooth_target: no smooth target in range admits a witness",
    ))
}

/// Lift `ideal` to an equivalent `O_0`-ideal of norm exactly `target`.
///
/// Uniform composition: requires `target = N(I) · m²` for integer `m`.
/// Returns `Error::Unimplemented` for targets that don't factor; full
/// KLPT's γ-randomisation step relaxes this constraint.
pub fn lift_to_smooth_norm(ideal: &LeftIdeal<8>, target: u128) -> Result<LeftIdeal<8>> {
    // Read N(I) as u128. The norm Uint<8> is 512 bits; we require it fit u128.
    let n_uint = ideal.norm();
    let words = n_uint.as_words();
    // Uint<8> exposes 8 words of 64 bits each on 64-bit targets. First two
    // words give the low 128 bits; check the remaining are zero.
    if words[2..].iter().any(|&w| w != 0) {
        return Err(Error::Internal(
            "lift_to_smooth_norm: N(I) exceeds u128 prototype bound",
        ));
    }
    let n_i = (u128::from(words[1]) << 64) | u128::from(words[0]);
    if n_i == 0 {
        return Err(Error::Internal("lift_to_smooth_norm: zero ideal"));
    }
    if target % n_i != 0 {
        return Err(Error::Unimplemented(
            "lift_to_smooth_norm: target not divisible by N(I); full KLPT relaxes this via γ-randomisation",
        ));
    }
    let m_sq = target / n_i;
    let m = m_sq.isqrt();
    if m.checked_mul(m) != Some(m_sq) {
        return Err(Error::Unimplemented(
            "lift_to_smooth_norm: target / N(I) is not a perfect square; full KLPT relaxes this via γ-randomisation",
        ));
    }
    // Prime fixed at 7 for the prototype (the rest of the quaternion module
    // uses this fake prime); real-prime use rewires once Sign/Verify needs it.
    let p: u128 = 7;
    let p_uint = Uint::<8>::from_u128(p);
    // Find β with N_red(β) = m. Special-case m = 0 to return the zero ideal
    // (no witness needed).
    if m == 0 {
        return Ok(LeftIdeal::new(
            [[crypto_bigint::Int::<8>::from_i64(0); 4]; 4],
        ));
    }
    // Search the full order for β with N_red(β) = m. Use the O_0-aware
    // search so witnesses with fractional standard coords (e.g. (1+i+j)/2
    // and similar) are found — they're invisible to the integer-standard
    // `find_norm_witness` path.
    #[allow(clippy::cast_possible_wrap)] // m bounded for prototype (small)
    let m_i64: i64 = m
        .try_into()
        .map_err(|_| Error::Internal("lift_to_smooth_norm: m exceeds i64 prototype bound"))?;
    let full = LeftIdeal::<8>::full_order();
    let beta_o0 = find_quaternion_in_ideal_with_norm(&full, m_i64, &p_uint, 5).ok_or(
        Error::Unimplemented(
            "lift_to_smooth_norm: no β with N_red(β) = √(target/N(I)) in O_0 within bound 5",
        ),
    )?;
    Ok(ideal_right_multiply(ideal, &beta_o0, &p_uint))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lift_two_o0_to_itself_via_unit_beta() {
        // N(2·O_0) = 16, target = 16, m² = 1, m = 1, β = 1; result = 2·O_0.
        let two_id = LeftIdeal::<8>::full_order().scale(2);
        let j = lift_to_smooth_norm(&two_id, 16).expect("β=1 works");
        assert!(j.equals_lattice(&two_id));
        assert_eq!(j.norm(), Uint::<8>::from_u64(16));
    }

    #[test]
    fn lift_two_o0_target_sixty_four() {
        // N(2·O_0) = 16, target = 64, m² = 4, m = 2, β = e_3 at p=7.
        // Result has norm 16 · 4 = 64.
        let two_id = LeftIdeal::<8>::full_order().scale(2);
        let j = lift_to_smooth_norm(&two_id, 64).expect("β = e_3 works");
        assert_eq!(j.norm(), Uint::<8>::from_u64(64));
    }

    #[test]
    fn lift_target_not_divisible_by_norm() {
        // N(2·O_0) = 16; target = 5 is not divisible by 16.
        let two_id = LeftIdeal::<8>::full_order().scale(2);
        let r = lift_to_smooth_norm(&two_id, 5);
        assert!(matches!(r, Err(Error::Unimplemented(_))));
    }

    #[test]
    fn lift_full_order_perfect_square_target() {
        // target = 4 = 2²; γ with N_red = 2 is e_3 at p=7; J = O_0·e_3, norm 4.
        let id = LeftIdeal::<8>::full_order();
        let j = lift_to_smooth_norm(&id, 4).expect("perfect-square target succeeds");
        assert_eq!(j.norm(), Uint::<8>::from_u64(4));
    }

    #[test]
    fn lift_full_order_non_square_signals_unimplemented() {
        // target = 8 is not m² → principal lift can't satisfy.
        let id = LeftIdeal::<8>::full_order();
        let r = lift_to_smooth_norm(&id, 8);
        assert!(matches!(r, Err(Error::Unimplemented(_))));
    }

    #[test]
    fn lift_full_order_one_returns_full_order() {
        let id = LeftIdeal::<8>::full_order();
        let j = lift_to_smooth_norm(&id, 1).expect("γ=1 works");
        assert!(j.equals_lattice(&id));
    }

    #[test]
    fn lift_full_order_twentyfive() {
        // target = 25 = 5²; γ = 1+2i has N_red = 5 → J norm 25.
        let id = LeftIdeal::<8>::full_order();
        let j = lift_to_smooth_norm(&id, 25).expect("γ = 1+2i works");
        assert_eq!(j.norm(), Uint::<8>::from_u64(25));
    }

    #[test]
    fn principal_with_norm_one_is_full_order() {
        let j = principal_ideal_with_reduced_norm(1, 7).expect("γ=1 works");
        let full = LeftIdeal::<8>::full_order();
        assert!(j.equals_lattice(&full));
        assert_eq!(j.norm(), Uint::<8>::from_u64(1));
    }

    #[test]
    fn principal_with_norm_two_at_p_seven() {
        // γ = e_3 = (1+k)/2 has N_red = 2 at p=7.
        // O_0·γ has ideal norm = N_red(γ)² = 4.
        let j = principal_ideal_with_reduced_norm(2, 7).expect("γ exists at p=7");
        assert_eq!(j.norm(), Uint::<8>::from_u64(4));
    }

    #[test]
    fn principal_with_norm_five_at_p_seven() {
        // γ = 1 + 2i has standard N_red = 1 + 4 = 5. O_0·γ has norm 25.
        let j = principal_ideal_with_reduced_norm(5, 7).expect("γ = 1+2i works");
        assert_eq!(j.norm(), Uint::<8>::from_u64(25));
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn sweep_finds_full_order_at_target_one() {
        let id = LeftIdeal::<8>::full_order();
        let (j, t) =
            lift_to_any_smooth_target(&id, &[2, 3, 5, 7], 1, 100).expect("smooth target exists");
        assert_eq!(t, 1);
        assert!(j.equals_lattice(&id));
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn sweep_finds_four_for_doubled_order_in_lower_range() {
        let two_id = LeftIdeal::<8>::full_order().scale(2);
        // N(2·O_0) = 16. Sweep [16, 100]: smooth targets in {2,3,5,7}^* ∩ [16, 100]:
        // 16, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, ... — but
        // target must be 16·m². For target=16 → m=1 ✓.
        let (j, t) =
            lift_to_any_smooth_target(&two_id, &[2, 3, 5, 7], 16, 100).expect("target exists");
        assert_eq!(t, 16);
        assert_eq!(j.norm(), Uint::<8>::from_u64(16));
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn sweep_skips_lower_range_when_floor_too_high() {
        // For 2·O_0, min valid target is 16 (=N(I)). With target_low=100,
        // sweep should find 16·m² ≥ 100 → m=3, target=144.
        let two_id = LeftIdeal::<8>::full_order().scale(2);
        let (j, t) =
            lift_to_any_smooth_target(&two_id, &[2, 3, 5, 7], 100, 200).expect("target exists");
        assert_eq!(t, 144);
        assert_eq!(j.norm(), Uint::<8>::from_u64(144));
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn sweep_no_witness_in_range() {
        // For O_0 with smooth primes = {3}, targets are powers of 3: 1, 3, 9, 27, 81.
        // None except 1, 9, 81 are perfect squares. Restrict [2, 8] — only 3 is
        // smooth, not a square → no witness → Err.
        let id = LeftIdeal::<8>::full_order();
        let r = lift_to_any_smooth_target(&id, &[3], 2, 8);
        assert!(matches!(r, Err(Error::Unimplemented(_))));
    }

    #[test]
    fn principal_with_unrepresentable_norm_returns_none() {
        // For p=7, can we find γ with N_red(γ) = 3? Standard a²+b²+7(c²+d²)=3.
        // c=d=0: a²+b²=3 has no integer solution. c²+d²≥1: 7·1=7>3.
        // So no integer γ has norm 3. find_norm_witness returns None →
        // principal_ideal_with_reduced_norm returns None.
        assert!(principal_ideal_with_reduced_norm(3, 7).is_none());
    }
}
