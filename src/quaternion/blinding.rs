// SPDX-License-Identifier: MIT OR Apache-2.0
//! Stage-1 constant-time signing via SECRET-IDEAL BLINDING.
//!
//! ## Why blinding (not constant-time reimplementation)
//!
//! The secret-dependent signing work — `compute_response_quat_element`
//! (intersection → LLL → rejection) and the auxiliary-isogeny finder — runs on
//! lattices derived from the secret ideal. Its timing (rejection-loop iteration
//! counts, LLL step counts) depends on the secret key; `tests/ct_signing.rs`
//! measures this as an std-ratio of ~69 at baseline (a large leak).
//!
//! The C reference is itself variable-time (GMP), so a byte-exact CT path is
//! impossible; and constant-time reimplementation of LLL / rejection sampling /
//! prime search is an open research problem. **Blinding** is the tractable
//! alternative: run the *same* variable-time algorithms, but on a freshly
//! randomized class-equivalent of the secret ideal, so the observable timing
//! tracks the per-signature blinding randomness `α` rather than the key.
//!
//! ## Construction
//!
//! For secret left `O`-ideal `I` and a fresh random quaternion `α ∈ O_0`, the
//! blinded ideal is `J = I·α` (right multiplication, [`ideal_right_multiply`]),
//! with the lattice-index norm relation
//!
//! ```text
//!     N(J) = N(I) · N_red(α)².
//! ```
//!
//! `J` is a class-equivalent left `O`-ideal; the response computed against `J`
//! is unblinded by `α` to recover the response against `I`. Over random `α` the
//! sign-time distribution becomes independent of the secret key.
//!
//! ## Status / verification gates (this module is the START of Stage 1)
//!
//! - ✅ [`blind_secret_ideal`] — random equivalent ideal + unblind data `α`.
//!   Verified here by the **norm-equivalence invariant** test (checkable in
//!   isolation, no crypto assumptions).
//! - ⏳ **GATED — not yet wired into the live sign path.** The unblind of the
//!   response quaternion (and proof it preserves the sign↔verify relation) MUST
//!   be validated by `sign_verify_roundtrip` + the byte-exact keygen KAT staying
//!   green, AND by `ct_sign_blinding_eval_random_beta` showing |t| < 4.5, BEFORE
//!   `protocols_sign` is switched to the blinded path. Wiring it before those
//!   gases pass would risk the 351-session byte-exact work. The unblind math is
//!   the part that needs careful validation, not assumption.

use crypto_bigint::{Int, Uint};
use rand_core::CryptoRng;

use crate::quaternion::ideal::LeftIdeal;
use crate::quaternion::ideal_mul::ideal_right_multiply;
use crate::quaternion::o0_mul::reduced_norm_o0_basis;

/// Sample a fresh random blinding quaternion `α ∈ O_0` (small nonzero
/// `O_0`-coordinates) and return the blinded ideal `J = I·α` together with `α`
/// (the unblind data). `α` is small so `J`'s entries stay bounded; it is drawn
/// fresh per signature, which is what decorrelates timing from the key.
pub fn blind_secret_ideal<const LIMBS: usize, R: CryptoRng>(
    secret: &LeftIdeal<LIMBS>,
    p: &Uint<LIMBS>,
    rng: &mut R,
) -> (LeftIdeal<LIMBS>, [Int<LIMBS>; 4]) {
    let alpha = sample_blinding_quat::<LIMBS, R>(rng);
    let blinded = ideal_right_multiply::<LIMBS>(secret, &alpha, p);
    (blinded, alpha)
}

/// Draw a small nonzero `O_0`-coordinate quaternion for blinding. Coordinates
/// in `[-4, 4]`, retrying until nonzero. (Small keeps `N_red(α)` modest so the
/// blinded ideal's working width does not grow.)
fn sample_blinding_quat<const LIMBS: usize, R: CryptoRng>(rng: &mut R) -> [Int<LIMBS>; 4] {
    loop {
        let mut buf = [0u8; 4];
        rng.fill_bytes(&mut buf);
        let c = core::array::from_fn(|i| Int::<LIMBS>::from_i64((buf[i] % 9) as i64 - 4));
        if c != [Int::<LIMBS>::from_i64(0); 4] {
            return c;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quaternion::ideal::LeftIdeal;
    use rand_chacha::ChaCha20Rng;
    use rand_chacha::rand_core::SeedableRng;

    /// Norm-equivalence invariant. `reduced_norm_vartime` returns the *reduced*
    /// ideal norm `N_red = √cached_norm` (cached_norm is the lattice-index
    /// `|det| = N_red²`). Right-multiplication scales covolume by `N_red(α)²`, so
    /// in reduced-norm terms the relation is LINEAR: `N_red(J) = N_red(I)·N_red(α)`.
    /// This is the checkable correctness property of the blinding primitive,
    /// independent of any cryptographic unblind argument.
    #[test]
    fn blinded_ideal_satisfies_norm_invariant() {
        // L=16: N_red(α) for a full O_0 quaternion includes the p·(c²+d²) term
        // (~2^248), so N(I)·N_red(α)² needs ~744 bits — L=8 (512b) overflows
        // multiply_o0_basis. (This norm blow-up is itself the key Stage-1 finding:
        // generic right-mult blinding is NOT norm-preserving; see module docs.)
        const L: usize = 16;
        let p = crate::params::lvl1::prime().resize::<L>();
        // A concrete left ideal: O_0 scaled by 7.
        let i = LeftIdeal::<L>::full_order().scale(7);
        let n_i = i.reduced_norm_vartime().expect("norm");

        let mut rng = ChaCha20Rng::seed_from_u64(0xB11D);
        for _ in 0..32 {
            let (j, alpha) = blind_secret_ideal::<L, _>(&i, &p, &mut rng);
            let n_j = j
                .reduced_norm_vartime()
                .expect("blinded norm is a perfect square");
            let n_alpha = reduced_norm_o0_basis::<L>(&alpha, &p).abs();
            // N_red(J) == N_red(I) · N_red(α)
            let expected = n_i.wrapping_mul(&n_alpha);
            assert_eq!(
                n_j, expected,
                "N_red(J)=N_red(I)·N_red(α) failed for α={alpha:?}"
            );
        }
    }
}
