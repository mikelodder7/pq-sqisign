// SPDX-License-Identifier: MIT OR Apache-2.0
//! Clapotis evaluator — ideal-to-isogeny translation via higher-dimensional
//! theta isogenies. SQIsign 2.0.1's response-side computation.
//!
//! # The role in SQIsign's signing pipeline
//!
//! After the KLPT body (`klpt_body_wide_wn`) produces an equivalent left
//! ideal `K` of smooth norm `N(K) = q · T` where `T` is the smooth target
//! (typically a power of 2), the Clapotis evaluator translates `K` into an
//! isogeny `φ: E_0 → E_K` of degree `q · T`. The resulting isogeny is the
//! response-side computation in the SQIsign protocol.
//!
//! # Algorithm sketch (per SQIsign 2.0.1 spec §6)
//!
//! The Clapotis evaluator works in **higher-dimensional theta space** —
//! abelian varieties of dimension 2 or 4 carry theta structures from which
//! the desired isogeny on the base elliptic curve can be extracted via
//! a projection step. The construction:
//!
//! 1. Embed the ideal `K` in a higher-dimensional theta abelian variety.
//! 2. Compute the theta-coordinate evaluation of `K`'s representation.
//! 3. Project back to the elliptic-curve isogeny via the canonical map.
//!
//! The detailed algorithms (theta evaluation, gluing, projection) span
//! roughly 25-30 sessions of implementation per the ISA roadmap. This
//! module ships the scaffolding (public API + types) so downstream code
//! (Sign/Verify orchestration) can target a stable interface while the
//! inner algorithms are filled in over future sessions.
//!
//! # Output shape
//!
//! [`IdealToIsogenyResult`] bundles the data Sign needs from the
//! translation:
//! - The codomain curve `E_K = E_0 / ker(φ)` (placeholder `PhantomData<P>`
//!   for now; will be a Montgomery curve over `F_{p²}` once filled in).
//! - The kernel point or isogeny chain representation (TBD per spec).
//!
//! Sign extracts the response `σ` from this output and the upstream
//! commitment/challenge state.

use core::marker::PhantomData;

use crypto_bigint::Uint;
use rand_core::CryptoRng;

use crate::error::{Error, Result};
use crate::params::Params;
use crate::quaternion::ideal_mul::LeftIdealWideNorm;

/// Structured output of [`ideal_to_isogeny`] at security level `P`.
///
/// Placeholder for the codomain curve + isogeny representation that the
/// Clapotis evaluator returns. Filled in once the higher-dimensional theta
/// algorithms land.
#[derive(Debug, Clone)]
pub struct IdealToIsogenyResult<P: Params> {
    /// Phantom for the security level. Future fields will include the
    /// codomain Montgomery curve (`A24` projective form) and the isogeny
    /// chain or theta-coordinate data.
    pub _marker: PhantomData<P>,
}

impl<P: Params> IdealToIsogenyResult<P> {
    /// Construct an empty placeholder result.
    #[inline]
    pub const fn placeholder() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

/// Translate an equivalent left ideal `K` (output of the KLPT body) into
/// an isogeny `φ: E_0 → E_K` of degree `q · T` at security level `P`.
///
/// `klpt_output` is the `LeftIdealWideNorm<TLIMBS>` produced by
/// [`crate::quaternion::klpt::klpt_body_wide_wn`]: `K.cached_norm = q · T`
/// for some smooth target `T` and γ-randomization prime `q`.
///
/// `q` is the prime factor from γ-randomization, returned alongside `K`.
/// The downstream isogeny degree is `q · T = K.cached_norm`.
///
/// # Current status — stubbed
///
/// Returns `Error::Unimplemented("ideal_to_isogeny: ...")`. The full
/// Clapotis evaluator algorithm is the dominant remaining scope (~25-30
/// sessions per the ISA roadmap). This stub establishes the public API
/// contract so:
/// - Sign/Verify orchestration can wire against a stable signature.
/// - Tests can assert the post-KLPT-body integration shape.
/// - Future sessions can fill the body in place without touching callers.
///
/// # Inputs
///
/// - `klpt_output`: the left ideal `K` with cached norm `q · T`.
/// - `q`: the γ-randomization prime factor (used to decompose the
///   isogeny degree into `q` and the smooth `T` factor).
/// - `rng`: cryptographically secure RNG (Clapotis evaluator needs
///   randomness for the gluing step's choice of basis).
///
/// # Errors
///
/// Currently always returns `Error::Unimplemented`. Future
/// implementations may return other errors if the input ideal's norm
/// structure is incompatible with the evaluator's preconditions.
#[allow(clippy::needless_pass_by_value)] // future implementations consume the rng
pub fn ideal_to_isogeny<P: Params, const TLIMBS: usize, R: CryptoRng>(
    _klpt_output: &LeftIdealWideNorm<TLIMBS>,
    _q: Uint<8>,
    _rng: &mut R,
) -> Result<IdealToIsogenyResult<P>> {
    Err(Error::Unimplemented(
        "ideal_to_isogeny: Clapotis evaluator pending (the dominant remaining scope, ~25-30 sessions)",
    ))
}

/// Verify a SQIsign signature by evaluating the response isogeny.
///
/// Where [`ideal_to_isogeny`] is the signing-side translation (left
/// ideal → isogeny), this is the verification-side dual: given the
/// serialised signature (which encodes the response isogeny in some
/// compact form), the message, and the public key (which carries the
/// commitment / public curve), reconstruct the response isogeny,
/// apply it to the derived challenge, and check the resulting curve
/// matches the public key's expected codomain.
///
/// # Current status — stubbed
///
/// Returns `Error::Unimplemented`. The full verify path consumes the
/// same Clapotis evaluator infrastructure that [`ideal_to_isogeny`]
/// builds — once that algorithmic body lands, this function fills in
/// with the parsing + isogeny-evaluation + codomain-check sequence.
/// Currently in place so `crate::verify` can dispatch through a
/// stable contract.
///
/// # Inputs
///
/// - `sig`: serialised signature bytes (response isogeny + challenge
///   commitment per the SQIsign wire format).
/// - `msg`: signed message bytes.
/// - `pk`: public key bytes (codomain curve).
///
/// # Errors
///
/// Currently always returns `Error::Unimplemented`. Future
/// implementations will return `Error::InvalidSignature` (or similar)
/// when the response isogeny's codomain doesn't match `pk`'s curve.
pub fn evaluate_response_isogeny<P: Params>(_sig: &[u8], _msg: &[u8], _pk: &[u8]) -> Result<()> {
    Err(Error::Unimplemented(
        "evaluate_response_isogeny: Clapotis evaluator pending (the dominant remaining scope, ~25-30 sessions)",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ideal_to_isogeny_stub_returns_unimplemented_at_lvl1() {
        // S77 scaffolding test: the public API stub is in place and
        // returns the documented `Unimplemented` error. This locks
        // down the contract that downstream code (Sign/Verify
        // orchestration) can wire against.
        use crate::params::Level1;
        use crate::quaternion::ideal::LeftIdeal;
        use crate::rng::NistPqcRng;

        // Construct a placeholder KLPT output: O_0 wrapped at TLIMBS=8
        // with cached_norm = 1. The actual numerical value doesn't
        // matter for this contract test — the stub returns
        // Unimplemented regardless of input.
        let inner = LeftIdeal::<8>::full_order();
        let klpt_output: LeftIdealWideNorm<8> = LeftIdealWideNorm::from_narrow(inner);
        let q = Uint::<8>::from_u64(1);
        let mut rng = NistPqcRng::new(&[0x77u8; 48]);

        let result = ideal_to_isogeny::<Level1, 8, _>(&klpt_output, q, &mut rng);
        let err = result.expect_err("S77: stub must return Unimplemented");
        let Error::Unimplemented(msg) = err else {
            unreachable!("S77: expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis evaluator") || msg.contains("dominant remaining scope"),
            "S77: stub's error message must reference the Clapotis evaluator deferral; got: {msg}",
        );
    }

    #[test]
    fn evaluate_response_isogeny_stub_returns_unimplemented_at_lvl1() {
        // S84 contract test: the verify-side stub mirrors S77's
        // ideal_to_isogeny contract. Placeholder byte slices; the stub
        // returns Unimplemented regardless of input. Locks the contract
        // that `crate::verify` wires against.
        use crate::params::Level1;

        let result = evaluate_response_isogeny::<Level1>(&[], b"msg", &[]);
        let err = result.expect_err("S84: verify stub must return Unimplemented");
        let Error::Unimplemented(msg) = err else {
            unreachable!("S84: expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis evaluator") || msg.contains("dominant remaining scope"),
            "S84: verify stub's error must reference the Clapotis deferral; got: {msg}",
        );
    }
}
