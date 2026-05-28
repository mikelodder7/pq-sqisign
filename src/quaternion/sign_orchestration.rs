// SPDX-License-Identifier: MIT OR Apache-2.0
//! Signing-protocol orchestration layer above the KLPT body.
//!
//! Functions in this module sit between the signing-flow entry point
//! (`lib::sign`) and the primitive operations in [`super::klpt`] and
//! [`super::represent_integer`]. They convert the cryptographic state
//! (a freshly-computed quaternion response, the commit/challenge
//! lattice content, etc.) into the inputs that downstream isogeny-side
//! routines consume.
//!
//! Ships `compute_random_aux_norm_and_helpers` — the deterministic
//! orchestrator that prepares the inputs `evaluate_random_aux_isogeny_signature`
//! consumes. See the function's docs for the algorithm (transcribed from the
//! C reference `src/signature/ref/lvlx/sign.c:123-190`).

use core::marker::PhantomData;

use crypto_bigint::{Int, NonZero, Uint};
use rand_core::CryptoRng;

use crate::error::{Error, Result};
use crate::params::Params;
use crate::quaternion::hnf::int_div_floor;
use crate::quaternion::ideal::LeftIdeal;
use crate::quaternion::o0_mul::{
    left_ideal_from_element_and_integer_o0, o0_conjugate, reduced_norm_o0_basis,
};
use crate::quaternion::represent_integer::narrow_left_ideal_to_8;

/// 2-adic valuation `v_2(x)` of a `Uint<LIMBS>` — count trailing zero
/// bits across the limb array. Returns `64 · LIMBS` when `x == 0`.
///
/// **Variable-time** on the bit pattern of `x`. The orchestrator's call
/// pattern passes `x = N_red(resp_quat) / lattice_content`, derived from
/// the signing-flow secret `resp_quat`. This matches the SQIsign 2.0
/// reference's `ibz_two_adic` which is also variable-time on its input:
/// SQIsign places its constant-time discipline at the isogeny / curve
/// layer (Montgomery ladders, theta-doubling), not at the quaternion
/// layer (see spec §8 — Side-Channel Considerations). The leakage of
/// `v_2(N_red(resp_quat))` per signing iteration is considered acceptable
/// because (a) the timing channel is fragmented across many ops, (b) the
/// challenge-derivation Fiat-Shamir step blocks adaptive timing attacks,
/// (c) signatures are only released after all quaternion computation
/// completes.
fn uint_two_adic_vartime<const LIMBS: usize>(x: &Uint<LIMBS>) -> u32 {
    let words = x.as_words();
    let mut tz: u32 = 0;
    for &w in words {
        if w == 0 {
            tz += 64;
        } else {
            tz += w.trailing_zeros();
            return tz;
        }
    }
    tz
}

/// Modular inverse `a^{-1} mod m` via extended Euclidean algorithm.
/// Returns `None` when `gcd(a, m) ≠ 1` (i.e. no inverse exists) or
/// when `m ∈ {0, 1}`.
///
/// **Variable-time** on both inputs. The orchestrator's call pattern
/// passes `a = degree_odd_resp` (derived from the signing-flow secret
/// `resp_quat`) and `m = remain` (combines `degree_odd_resp`'s 2-adic
/// valuation with the public per-level `response_bits +
/// hd_extra_torsion`). Per the SQIsign 2.0 spec §8 convention,
/// quaternion-side variable-time arithmetic is acceptable; the
/// constant-time discipline lives at the isogeny / curve layer. See
/// [`uint_two_adic_vartime`] for the full rationale.
///
/// Algorithm: classical extended Euclidean tracking the bezout
/// coefficient `t` for `a` (we don't need `s` for `m`):
///   (r0, t0) = (m, 0)
///   (r1, t1) = (a mod m, 1)
///   while r1 ≠ 0: q = r0/r1; (r0, t0, r1, t1) = (r1, t1, r0 − q·r1, t0 − q·t1)
/// If `r0 == 1` at exit, `t0 mod m` is the inverse.
///
/// Bezout coefficients are tracked as `Int<LIMBS>` (signed) since `t`
/// can go negative mid-loop; `int_div_floor` is used for the signed
/// quotient (the dividend stays non-negative throughout, so floor =
/// truncating division here).
fn uint_inv_mod_vartime<const LIMBS: usize>(
    a: &Uint<LIMBS>,
    m: &Uint<LIMBS>,
) -> Option<Uint<LIMBS>> {
    let zero_u = Uint::<LIMBS>::ZERO;
    let one_u = Uint::<LIMBS>::ONE;
    if *m == zero_u || *m == one_u {
        return None;
    }
    // Precondition (Forge S184 M2/M3): the bezout coefficient `t` and the
    // intermediate `q · t1` are tracked in `Int<LIMBS>`, which has only
    // `64·LIMBS − 1` magnitude bits. The classical extended-Euclidean
    // bound `|t_i| < m` plus `|q · t1|` near `m` requires `m`'s top bit
    // to be zero (so `m_int = m.as_int()` is non-negative and the
    // intermediate product stays in range). At the orchestrator's
    // calling pattern (`m = 2^(pow_dim2_deg_resp + hd_extra_torsion)`)
    // this is structurally true at L1/L3/L5; the debug_assert defends
    // future callers against the latent contract trap.
    let limbs_u32 = u32::try_from(LIMBS).expect("LIMBS fits u32 for all SQIsign levels");
    debug_assert!(
        m.bits_vartime() + 1 < 64u32 * limbs_u32,
        "uint_inv_mod_vartime: m must fit strictly inside Int<LIMBS> magnitude (bits + 1 < 64*LIMBS)",
    );
    let m_nz = NonZero::new(*m).expect("m != 0 checked above");
    let a_mod = a.rem_vartime(&m_nz);
    if a_mod == zero_u {
        return None;
    }

    let m_int = *m.as_int();
    let a_mod_int = *a_mod.as_int();
    let zero_int = Int::<LIMBS>::from_i64(0);
    let one_int = Int::<LIMBS>::from_i64(1);

    let mut r0 = m_int;
    let mut r1 = a_mod_int;
    let mut t0 = zero_int;
    let mut t1 = one_int;

    while r1 != zero_int {
        let q = int_div_floor(&r0, &r1);
        let q_r1 = q.wrapping_mul(&r1);
        let q_t1 = q.wrapping_mul(&t1);
        let new_r = r0.wrapping_sub(&q_r1);
        let new_t = t0.wrapping_sub(&q_t1);
        r0 = r1;
        r1 = new_r;
        t0 = t1;
        t1 = new_t;
    }

    if r0 != one_int {
        return None;
    }

    // Normalize t0 into [0, m). If negative, add m.
    let normalized = if bool::from(t0.is_negative()) {
        t0.wrapping_add(&m_int)
    } else {
        t0
    };
    Some(Uint::<LIMBS>::from_words(normalized.to_words()))
}

/// Outputs of [`compute_random_aux_norm_and_helpers`].
///
/// The C reference mutates a `signature_t` struct in place and writes
/// to four out-pointers; in the Rust port we collect everything into
/// this struct so the caller can route fields into the eventual
/// `Signature` builder.
///
/// `LIMBS` is the precision tier of the wide arithmetic; the function
/// expects callers to size LIMBS for `N_red(resp_quat)` (typically the
/// general path's `64·LIMBS ≥ 2·bits(p) + 1`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuxNormHelpers<const LIMBS: usize> {
    /// `k = SQIsign_response_length − v_2(N_red(resp_quat)) − backtracking`.
    /// Returned by the C reference as the gate value: `> 0` means proceed
    /// to `evaluate_random_aux_isogeny_signature`; `== 0` means the
    /// signing loop should restart with a fresh `resp_quat`.
    ///
    /// **Caller contract**: check this field FIRST. When zero, the
    /// `random_aux_norm`, `remain`, and `degree_resp_inv` fields contain
    /// computed-but-undefined values (the algorithm still ran Steps 8-11
    /// to keep control flow uniform) — callers MUST NOT consume them.
    /// `lideal_com_resp`, `conjugated_resp_quat`, and `two_resp_length`
    /// remain meaningful at gate=0 (they're computed before Step 7's
    /// gate value is known).
    pub pow_dim2_deg_resp: u32,

    /// `random_aux_norm = 2^k − degree_odd_resp`.
    ///
    /// This is the norm at which the caller materializes the auxiliary
    /// ideal (downstream, via [`super::represent_integer::sampling_random_ideal_o0_given_norm_wide`]
    /// with `is_prime` selected per the smoothness of this value, plus
    /// `prime_cofactor` if composite).
    pub random_aux_norm: Uint<LIMBS>,

    /// `degree_resp_inv = degree_odd_resp^{-1} mod remain` where
    /// `degree_odd_resp = N_red(resp_quat) / (lattice_content · 2^{v_2})`.
    /// Caller invariant: `gcd(degree_odd_resp, remain) == 1` (the
    /// modular inverse is total over `Uint<LIMBS>` only under coprimality).
    pub degree_resp_inv: Uint<LIMBS>,

    /// `remain = 2^(k + HD_extra_torsion)`. The "remaining" 2-power
    /// budget for downstream isogeny work; combines the dim-2 step count
    /// with the SQIsign HD-extra-torsion constant.
    pub remain: Uint<LIMBS>,

    /// `lideal_com_resp = O_0 · conj(resp_quat) + O_0 · (n(commit) · degree_odd_resp)`.
    /// Built via [`super::o0_mul::left_ideal_from_element_and_integer_o0`]
    /// with `γ = conj(resp_quat)` (O_0-basis) and
    /// `n = n(lideal_commit) · degree_odd_resp`.
    pub lideal_com_resp: LeftIdeal<8>,

    /// The conjugated `resp_quat` (the C reference conjugates in place
    /// via `quat_alg_conj`). Returned here so the caller can replace its
    /// own copy if needed; the conjugation also feeds into the
    /// `lideal_com_resp` construction above.
    pub conjugated_resp_quat: [Int<LIMBS>; 4],

    /// `two_resp_length = v_2(N_red(resp_quat))` — the 2-adic valuation
    /// of the reduced norm before stripping. Stored on the signature's
    /// `two_resp_length` field in the C ref.
    pub two_resp_length: u32,
}

/// Compute the auxiliary-norm packet for a signing iteration.
///
/// Mirrors the C reference `compute_random_aux_norm_and_helpers` in
/// `src/signature/ref/lvlx/sign.c:123-190` of the SQIsign repository.
/// Despite "random" in the name, the function is **deterministic** given
/// its inputs — the randomness lives in the caller's earlier sampling
/// of `resp_quat`.
///
/// # Algorithm (12 steps, transcribed from C ref)
///
/// 1. Compute `(degree_full_resp, norm_d) = quat_alg_norm(resp_quat)`
///    as a rational. Assert `norm_d == 1` (the quaternion is integral).
/// 2. Divide `degree_full_resp` by `lattice_content` (= `n(commit) ·
///    n(secret_chall)`); the remainder MUST be zero — caller invariant.
///    Result: `degree_full_resp ← degree_full_resp / lattice_content`.
/// 3. `exp_diadic_val_full_resp = v_2(degree_full_resp)`; store on the
///    returned `two_resp_length`.
/// 4. Strip the 2-power: `degree_odd_resp = degree_full_resp / 2^{v_2}`.
///    Debug-assert `degree_odd_resp < 2^(response_bits − backtracking)`.
/// 5. Conjugate `resp_quat` in place via `o0_conjugate` (already
///    available at [`super::o0_mul::o0_conjugate`]).
/// 6. Build `lideal_com_resp = O_0 · conj(resp_quat) + O_0 ·
///    (lideal_commit_norm · degree_odd_resp)` via
///    [`super::o0_mul::left_ideal_from_element_and_integer_o0`] (the
///    S179 helper).
/// 7. `pow_dim2_deg_resp = response_bits − exp_diadic_val_full_resp −
///    backtracking`. This is the function's gate value (return value
///    in the C ref).
/// 8. `remain = 2^pow_dim2_deg_resp`.
/// 9. `random_aux_norm = remain − degree_odd_resp`.
/// 10. `remain ← remain · 2^hd_extra_torsion` (loop of `ibz_mul` by 2;
///     in Rust just a left-shift by `hd_extra_torsion`).
/// 11. `degree_resp_inv = degree_odd_resp^{-1} mod remain` via
///     extended Euclidean (`ibz_invmod` in C; we will need a small
///     `uint_inv_mod_vartime` helper).
/// 12. Return the populated `AuxNormHelpers`.
///
/// # Implementation notes
///
/// Private helpers `uint_two_adic_vartime` and `uint_inv_mod_vartime`
/// (defined at the top of this module) provide the 2-adic valuation
/// and extended-Euclidean modular inverse the algorithm needs.
/// `HD_extra_torsion` is accepted as a runtime argument; future polish
/// may promote it to a per-level `Params::HD_EXTRA_TORSION` const
/// alongside `RESPONSE_BITS`.
///
/// # Parameters
///
/// - `resp_quat`: quaternion response in O_0-basis coordinates.
/// - `lattice_content`: `n(lideal_commit) · n(lideal_secret_chall)`,
///   MUST divide `N_red(resp_quat)` exactly.
/// - `lideal_commit_norm`: `n(lideal_commit)` (the committed ideal's
///   reduced norm).
/// - `p`: base prime.
/// - `backtracking`: number of backtracking iterations consumed
///   (`sig->backtracking` in the C ref).
/// - `response_bits`: `SQIsign_response_length` per spec — per-level
///   constant (L1 = 126, L3 = 192, L5 = 253, per
///   [`crate::params::Params::RESPONSE_BITS`]).
/// - `hd_extra_torsion`: `HD_extra_torsion` per spec — extra 2-power
///   headroom for downstream isogeny work.
///
/// # Returns
///
/// - `Ok(AuxNormHelpers)`: populated with all 7 outputs.
/// - `Err(Error::Internal)`: validation failure (e.g.
///   `lattice_content ∤ N_red(resp_quat)`, response-bit budget
///   underflow, or `lideal_commit_norm · degree_odd_resp` overflowing
///   `Uint<LIMBS>`).
///
/// # Precision contract
///
/// Caller's `LIMBS` must hold `N_red(resp_quat)` at exact precision —
/// same general-path bound as [`super::represent_integer::sampling_random_ideal_o0_given_norm_wide`]
/// (`64·LIMBS ≥ 2·bits(p) + 1`). At L1 this is `LIMBS ≥ 8`.
#[allow(clippy::too_many_arguments)]
pub fn compute_random_aux_norm_and_helpers<const LIMBS: usize>(
    resp_quat: &[Int<LIMBS>; 4],
    lattice_content: &Uint<LIMBS>,
    lideal_commit_norm: &Uint<LIMBS>,
    p: &Uint<LIMBS>,
    backtracking: u32,
    response_bits: usize,
    hd_extra_torsion: u32,
) -> Result<AuxNormHelpers<LIMBS>> {
    let zero_u = Uint::<LIMBS>::ZERO;
    let one_u = Uint::<LIMBS>::ONE;

    // Step 1: degree_full_resp = N_red(resp_quat). For valid O_0 elements
    // the reduced norm is non-negative; abs is defensive. Reject zero
    // (downstream division would fail anyway).
    let nrm_int = reduced_norm_o0_basis::<LIMBS>(resp_quat, p);
    let degree_full_resp = nrm_int.abs();
    if degree_full_resp == zero_u {
        return Err(Error::Internal(
            "compute_random_aux_norm_and_helpers: N_red(resp_quat) is zero",
        ));
    }

    // Step 2: divide by lattice_content; the remainder MUST be zero — this
    // is the caller invariant (n(commit) · n(secret_chall) divides the
    // reduced norm of the response quaternion by construction of the KLPT
    // signing flow). Violation surfaces here as Err(Internal).
    let lc_nz: NonZero<Uint<LIMBS>> =
        Option::<NonZero<Uint<LIMBS>>>::from(NonZero::new(*lattice_content)).ok_or(
            Error::Internal("compute_random_aux_norm_and_helpers: lattice_content is zero"),
        )?;
    let (degree_full_resp, rem) = degree_full_resp.div_rem_vartime(&lc_nz);
    if rem != zero_u {
        return Err(Error::Internal(
            "compute_random_aux_norm_and_helpers: lattice_content does not divide N_red(resp_quat) exactly",
        ));
    }

    // Step 3: 2-adic valuation of the post-divide value.
    let two_resp_length = uint_two_adic_vartime::<LIMBS>(&degree_full_resp);

    // Step 4: strip the 2-power → degree_odd_resp.
    let degree_odd_resp = degree_full_resp.shr_vartime(two_resp_length);
    debug_assert!(
        degree_odd_resp.as_words()[0] & 1 == 1 || degree_odd_resp == zero_u,
        "degree_odd_resp must be odd after stripping the 2-power",
    );
    #[cfg(debug_assertions)]
    {
        let response_bits_u32 = u32::try_from(response_bits)
            .expect("response_bits fits in u32 for all supported levels");
        let max_bits = response_bits_u32.saturating_sub(backtracking);
        let bits_used = degree_odd_resp.bits_vartime();
        debug_assert!(
            bits_used <= max_bits,
            "degree_odd_resp ({} bits) exceeds response_bits - backtracking ({})",
            bits_used,
            max_bits,
        );
    }

    // Step 5: conjugate resp_quat in O_0 basis.
    let conjugated_resp_quat = o0_conjugate::<LIMBS>(resp_quat);

    // Step 6: build lideal_com_resp = O_0 · conj(resp_quat) + O_0 ·
    // (lideal_commit_norm · degree_odd_resp).
    //
    // Caller invariant (carried from the signing flow): lideal_commit_norm
    // divides lattice_content, so the product lideal_commit_norm ·
    // degree_odd_resp divides degree_full_resp_pre_strip = N_red(resp_quat),
    // hence also divides N_red(conj(resp_quat)) (conjugation preserves
    // reduced norm). The S179 helper's `n | N_red(γ)` precondition is
    // therefore satisfied — its debug_assert will not fire.
    let ideal_norm = Option::<Uint<LIMBS>>::from(lideal_commit_norm.checked_mul(&degree_odd_resp))
        .ok_or(Error::Internal(
            "compute_random_aux_norm_and_helpers: lideal_commit_norm * degree_odd_resp overflows Uint<LIMBS>",
        ))?;
    let wide_ideal =
        left_ideal_from_element_and_integer_o0::<LIMBS>(&conjugated_resp_quat, &ideal_norm, p);
    let lideal_com_resp = narrow_left_ideal_to_8::<LIMBS>(&wide_ideal).ok_or(
        Error::Internal(
            "compute_random_aux_norm_and_helpers: lideal_com_resp exceeds Uint<8> ceiling — precision contract violated",
        ),
    )?;

    // Step 7: pow_dim2_deg_resp = response_bits − two_resp_length − backtracking.
    // Underflow is a real error (response budget violated by caller).
    let response_bits_u32 = u32::try_from(response_bits).map_err(|_| {
        Error::Internal(
            "compute_random_aux_norm_and_helpers: response_bits exceeds u32::MAX (impossible at any SQIsign level)",
        )
    })?;
    let pow_dim2_deg_resp = response_bits_u32
        .checked_sub(two_resp_length)
        .and_then(|x| x.checked_sub(backtracking))
        .ok_or(Error::Internal(
            "compute_random_aux_norm_and_helpers: response_bits - two_resp_length - backtracking underflows",
        ))?;

    // Step 8: remain_initial = 2^pow_dim2_deg_resp.
    //
    // Precondition (Forge S184 MINOR 5): `shl_vartime` panics when its
    // shift amount is >= the type's bit width. The orchestrator's call
    // pattern keeps `pow_dim2_deg_resp + hd_extra_torsion` well below
    // `64·LIMBS` at L1/L3/L5 production sizes, but the docstring's
    // precision contract does not explicitly cover the `remain` shift;
    // the debug_assert here documents the additional bound.
    let limbs_u32 = u32::try_from(LIMBS).expect("LIMBS fits u32 for all SQIsign levels");
    debug_assert!(
        pow_dim2_deg_resp + hd_extra_torsion < 64u32 * limbs_u32,
        "compute_random_aux_norm_and_helpers: pow_dim2_deg_resp + hd_extra_torsion ({}) must be < 64*LIMBS ({})",
        pow_dim2_deg_resp + hd_extra_torsion,
        64u32 * limbs_u32,
    );
    let remain_initial = one_u.shl_vartime(pow_dim2_deg_resp);

    // Step 9: random_aux_norm = remain_initial − degree_odd_resp.
    //
    // **C-ref retry semantics (Forge S184 M1)**: `pow_dim2_deg_resp == 0`
    // is the C reference's "restart the signing loop with fresh
    // resp_quat" signal (the caller checks the gate before consuming
    // any other field). At `pow_dim2_deg_resp == 0` we have
    // `remain_initial = 1`, and `random_aux_norm = 1 - degree_odd_resp`
    // wraps to a garbage value when `degree_odd_resp > 1` — but the
    // caller will discard it. Using `wrapping_sub` (NOT a checked
    // subtraction with Err on underflow) matches C semantics: return
    // Ok with the gate value 0 so the caller can restart, instead of
    // converting an expected retry into a hard error. When
    // `pow_dim2_deg_resp > 0`, the Step 4 debug-assert
    // `degree_odd_resp < 2^(response_bits - backtracking)` ensures
    // `degree_odd_resp < 2^pow_dim2_deg_resp = remain_initial`, so the
    // subtraction is non-negative and the value is meaningful.
    let random_aux_norm = remain_initial.wrapping_sub(&degree_odd_resp);

    // Step 10: remain ← remain_initial · 2^hd_extra_torsion = 2^(pow_dim2_deg_resp + hd_extra_torsion).
    let remain = remain_initial.shl_vartime(hd_extra_torsion);

    // Step 11: degree_resp_inv = degree_odd_resp^{-1} mod remain.
    // Since `remain` is a power of two and `degree_odd_resp` is odd
    // (after stripping the 2-power in Step 4), `gcd(degree_odd_resp, remain) == 1`
    // automatically — the inverse exists. The None branch is therefore
    // a defense against degenerate inputs (e.g. degree_odd_resp = 0 from
    // an upstream bug), not the expected path.
    let degree_resp_inv = uint_inv_mod_vartime::<LIMBS>(&degree_odd_resp, &remain).ok_or(
        Error::Internal(
            "compute_random_aux_norm_and_helpers: degree_odd_resp is not coprime to remain — should be impossible after the 2-power strip",
        ),
    )?;

    Ok(AuxNormHelpers {
        pow_dim2_deg_resp,
        random_aux_norm,
        degree_resp_inv,
        remain,
        lideal_com_resp,
        conjugated_resp_quat,
        two_resp_length,
    })
}

/// Outputs of [`evaluate_random_aux_isogeny_signature`].
///
/// In the C reference (`src/signature/ref/lvlx/sign.c:192-222`) the
/// function mutates an `ec_curve_t *E_aux` and an `ec_basis_t *B_aux`
/// in place. In the Rust port we collect them into this struct so the
/// caller assigns into its own state via the returned value.
///
/// Currently a placeholder mirroring
/// [`crate::isogeny::clapotis::IdealToIsogenyResult`]: the real fields
/// (a [`crate::ec::montgomery::MontgomeryCurve`] for `E_aux` and an
/// [`crate::ec::couple::EcBasis`] for `B_aux`) land alongside the
/// Clapotis evaluator body in a future session.
#[derive(Debug, Clone)]
pub struct AuxIsogenyOutputs<P: Params> {
    /// Phantom for the security level. Future fields will include the
    /// auxiliary codomain Montgomery curve (`E_aux`) and the image
    /// basis (`B_aux = φ(B_0)`).
    pub _marker: PhantomData<P>,
}

impl<P: Params> AuxIsogenyOutputs<P> {
    /// Construct an empty placeholder.
    #[inline]
    pub const fn placeholder() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

/// Evaluate the auxiliary-isogeny step of the SQIsign signing loop.
///
/// **Stub — body deferred to a follow-up session.** Returns
/// `Err(Error::Unimplemented)` with a specific deferred-body message
/// naming the Clapotis dependency.
///
/// Mirrors the C reference `evaluate_random_aux_isogeny_signature` in
/// `src/signature/ref/lvlx/sign.c:192-222` of the SQIsign repository.
/// The function consumes the outputs of
/// [`compute_random_aux_norm_and_helpers`] (the
/// `AuxNormHelpers.random_aux_norm` and `AuxNormHelpers.lideal_com_resp`
/// fields) and materializes the auxiliary curve `E_aux` and basis
/// `B_aux` for the dim-2 challenge isogeny.
///
/// # Algorithm (deferred body, 3 steps verbatim from C ref)
///
/// 1. **Sample** a random left `O_0`-ideal `lideal_aux` of reduced norm
///    `random_aux_norm` via
///    [`super::represent_integer::sampling_random_ideal_o0_given_norm_wide`]
///    (the S179/S180 sampler). The C ref's flag argument `0` corresponds
///    to `is_prime=false` + `prime_cofactor=Some(QUAT_prime_cofactor)`
///    — the general-path branch. Failure surfaces as
///    `Err(Internal)` matching the C reference's `found == 0` retry
///    branch (caller restarts the whole signing loop).
/// 2. **Intersect** with the response-commitment ideal:
///    `lideal_aux_resp_com = lideal_com_resp ∩ lideal_aux` via
///    `quat_lideal_inter` (C ref). **No Rust equivalent exists yet** —
///    the S187 body session must add a private/`pub(crate)`
///    `lideal_intersect<LIMBS>(I, J, p) -> LeftIdeal<8>` helper. The
///    intersection is computed as an HNF of the 8×4 augmented matrix
///    `[I.basis · I.denom, J.basis · J.denom]` after cross-scaling by
///    the LCM of denominators (standard sublattice-intersection
///    algorithm for Z-lattices of full rank).
/// 3. **Materialize** the composite ideal as an isogeny via
///    [`crate::isogeny::clapotis::ideal_to_isogeny`] (the Clapotis
///    evaluator, currently stubbed pending the dominant ~25-30-session
///    higher-dimensional-theta arc). Returns the codomain curve `E_aux`
///    and the canonical-E_0-basis image `B_aux`. Both are stored on
///    the returned `AuxIsogenyOutputs<P>`.
///
/// # Retry semantics
///
/// The C reference returns `int` (1 = success, 0 = retry the whole
/// signing loop). The Rust equivalent uses `Result`:
/// - `Ok(AuxIsogenyOutputs)`: success path; caller proceeds to the
///   dim-2 challenge isogeny composition.
/// - `Err(Error::Internal)` with a `retry` marker in the message: the
///   sampler or ideal-to-isogeny step failed and the signing loop
///   should restart (matching C's `if (!ret) continue;`).
/// - `Err(Error::Unimplemented)`: stub or upstream Clapotis stub.
///
/// # Parameters
///
/// - `random_aux_norm`: from
///   `compute_random_aux_norm_and_helpers(...).random_aux_norm`.
/// - `lideal_com_resp`: from
///   `compute_random_aux_norm_and_helpers(...).lideal_com_resp`.
/// - `p`: base prime (forwarded to the sampler's `find_quaternion_in_full_order_with_norm_wide`).
/// - `prime_cofactor`: forwarded to the sampler's general path.
/// - `sample_bound`, `max_trials`, `witnesses`: forwarded to the sampler.
/// - `rng`: cryptographically secure RNG.
///
/// # Returns
///
/// - `Ok(AuxIsogenyOutputs<P>)`: populated with `E_aux` and `B_aux`.
/// - `Err(Error::Internal)`: sampler or evaluator failure (caller
///   restarts the signing loop).
/// - `Err(Error::Unimplemented)`: stub; body deferred to S187.
///
/// # Precision contract
///
/// Same as
/// [`super::represent_integer::sampling_random_ideal_o0_given_norm_wide`]:
/// `64·LIMBS ≥ 2·bits(p) + 1` for the general path that
/// `random_aux_norm` lands on (composite, with `prime_cofactor`).
#[allow(clippy::too_many_arguments)]
pub fn evaluate_random_aux_isogeny_signature<P: Params, const LIMBS: usize, R: CryptoRng>(
    random_aux_norm: &Uint<LIMBS>,
    lideal_com_resp: &LeftIdeal<8>,
    p: &Uint<LIMBS>,
    prime_cofactor: Option<&Uint<LIMBS>>,
    sample_bound: i64,
    max_trials: usize,
    witnesses: &[Uint<LIMBS>],
    rng: &mut R,
) -> Result<AuxIsogenyOutputs<P>> {
    let _ = (
        random_aux_norm,
        lideal_com_resp,
        p,
        prime_cofactor,
        sample_bound,
        max_trials,
        witnesses,
        rng,
    );
    Err(Error::Unimplemented(
        "evaluate_random_aux_isogeny_signature: body deferred to S187. \
         Algorithm transcribed in the function docstring from C ref \
         src/signature/ref/lvlx/sign.c:192-222 of github.com/SQISign/the-sqisign. \
         Step 1 (sample random ideal) wires to the S180-shipped \
         sampling_random_ideal_o0_given_norm_wide. Step 2 (ideal intersection) \
         needs a new lideal_intersect<LIMBS> helper (no Rust equivalent yet \
         for C's quat_lideal_inter). Step 3 (ideal-to-isogeny materialization) \
         dispatches into the Clapotis evaluator at \
         src/isogeny/clapotis.rs:107, currently stubbed pending the \
         ~25-30-session higher-dimensional-theta arc. AuxIsogenyOutputs<P> \
         is a placeholder mirroring IdealToIsogenyResult<P>; real fields \
         (MontgomeryCurve + EcBasis) land with the body.",
    ))
}

#[cfg(all(test, feature = "kat"))]
mod tests {
    use super::*;
    use crypto_bigint::Uint;

    // ── Helper unit tests ──────────────────────────────────────────────

    #[test]
    fn uint_two_adic_vartime_basic_cases() {
        assert_eq!(uint_two_adic_vartime::<8>(&Uint::from_u64(1)), 0);
        assert_eq!(uint_two_adic_vartime::<8>(&Uint::from_u64(2)), 1);
        assert_eq!(uint_two_adic_vartime::<8>(&Uint::from_u64(4)), 2);
        assert_eq!(uint_two_adic_vartime::<8>(&Uint::from_u64(8)), 3);
        assert_eq!(uint_two_adic_vartime::<8>(&Uint::from_u64(12)), 2); // 12 = 4·3
        assert_eq!(uint_two_adic_vartime::<8>(&Uint::from_u64(100)), 2); // 100 = 4·25
        // Zero returns the full bit-width (64·LIMBS).
        assert_eq!(uint_two_adic_vartime::<8>(&Uint::ZERO), 64 * 8);
    }

    #[test]
    fn uint_inv_mod_vartime_small_cases() {
        // 3^{-1} mod 7 = 5 (since 3·5 = 15 = 2·7 + 1).
        assert_eq!(
            uint_inv_mod_vartime::<8>(&Uint::from_u64(3), &Uint::from_u64(7)),
            Some(Uint::<8>::from_u64(5)),
        );
        // 2^{-1} mod 5 = 3 (since 2·3 = 6 = 5 + 1).
        assert_eq!(
            uint_inv_mod_vartime::<8>(&Uint::from_u64(2), &Uint::from_u64(5)),
            Some(Uint::<8>::from_u64(3)),
        );
        // 1^{-1} mod anything > 1 is 1.
        assert_eq!(
            uint_inv_mod_vartime::<8>(&Uint::from_u64(1), &Uint::from_u64(256)),
            Some(Uint::<8>::from_u64(1)),
        );
        // gcd(2, 4) = 2 ≠ 1 → None.
        assert_eq!(
            uint_inv_mod_vartime::<8>(&Uint::from_u64(2), &Uint::from_u64(4)),
            None,
        );
        // m = 1 → degenerate, None (Z/1 has no inverses).
        assert_eq!(
            uint_inv_mod_vartime::<8>(&Uint::from_u64(7), &Uint::from_u64(1)),
            None,
        );
        // a = 0 → no inverse.
        assert_eq!(
            uint_inv_mod_vartime::<8>(&Uint::ZERO, &Uint::from_u64(7)),
            None,
        );
    }

    #[test]
    fn uint_inv_mod_vartime_odd_mod_power_of_two() {
        // The orchestrator's Step 11 always inverts an odd number modulo a
        // power of two. Exercise that specific pattern here.
        // 3^{-1} mod 16: need x with 3x ≡ 1 (mod 16). 3·11 = 33 = 2·16 + 1 → 11.
        assert_eq!(
            uint_inv_mod_vartime::<8>(&Uint::from_u64(3), &Uint::from_u64(16)),
            Some(Uint::<8>::from_u64(11)),
        );
        // 5^{-1} mod 256: 5·205 = 1025 = 4·256 + 1 → 205.
        assert_eq!(
            uint_inv_mod_vartime::<8>(&Uint::from_u64(5), &Uint::from_u64(256)),
            Some(Uint::<8>::from_u64(205)),
        );
    }

    // ── Orchestrator numerical tests ───────────────────────────────────

    /// Hand-calculated case with zero 2-adic valuation.
    ///
    /// Inputs: `p = 7` (fake L1), `resp_quat = (4, 2, 0, 0)` in O_0 coords.
    /// Standard-basis lift is `(4 − 0, 2 − 0, 0, 0) = (4, 2, 0, 0)`, so
    /// `N_red = 16 + 4 + 7·0 + 7·0 = 20`. `lattice_content = 20` →
    /// `degree_full_resp / 20 = 1`. `v_2(1) = 0`, `degree_odd_resp = 1`.
    /// Conjugation: `(4, 2, 0, 0) → (4 + 0, −2, −0, −0) = (4, −2, 0, 0)`.
    /// `lideal_commit_norm = 5`, `ideal_norm = 5 · 1 = 5`,
    /// `lideal_com_resp = O · (4, −2, 0, 0) + O · 5` (cached_norm = 5).
    /// `response_bits = 12`, `backtracking = 0` →
    /// `pow_dim2_deg_resp = 12 − 0 − 0 = 12`. `remain_initial = 4096`.
    /// `random_aux_norm = 4096 − 1 = 4095`. `hd_extra_torsion = 4` →
    /// `remain = 4096 · 16 = 65536`. `degree_resp_inv = 1^{-1} mod 65536 = 1`.
    /// `two_resp_length = 0`.
    #[test]
    fn compute_random_aux_orchestrator_zero_two_adic() {
        let resp_quat = [
            Int::<8>::from_i64(4),
            Int::<8>::from_i64(2),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        ];
        let result = compute_random_aux_norm_and_helpers::<8>(
            &resp_quat,
            &Uint::<8>::from_u64(20),
            &Uint::<8>::from_u64(5),
            &Uint::<8>::from_u64(7),
            0,
            12,
            4,
        )
        .expect("zero-two-adic case must succeed");
        assert_eq!(result.pow_dim2_deg_resp, 12);
        assert_eq!(result.random_aux_norm, Uint::<8>::from_u64(4095));
        assert_eq!(result.degree_resp_inv, Uint::<8>::from_u64(1));
        assert_eq!(result.remain, Uint::<8>::from_u64(65536));
        assert_eq!(result.lideal_com_resp.cached_norm, Uint::<8>::from_u64(5));
        assert_eq!(result.lideal_com_resp.denom, Uint::<8>::ONE);
        assert_eq!(result.conjugated_resp_quat[0], Int::<8>::from_i64(4));
        assert_eq!(result.conjugated_resp_quat[1], Int::<8>::from_i64(-2));
        assert_eq!(result.conjugated_resp_quat[2], Int::<8>::from_i64(0));
        assert_eq!(result.conjugated_resp_quat[3], Int::<8>::from_i64(0));
        assert_eq!(result.two_resp_length, 0);
    }

    /// Hand-calculated case with non-trivial 2-adic split.
    ///
    /// Inputs: `p = 7`, `resp_quat = (0, 4, 0, 0)`. Standard-basis lift is
    /// `(0 − 0, 4 − 0, 0, 0) = (0, 4, 0, 0)`. `N_red = 0 + 16 + 0 + 0 = 16`.
    /// `lattice_content = 4` → `degree_full_resp = 4`. `v_2(4) = 2`,
    /// `degree_odd_resp = 1`. Conjugation: `(0, 4, 0, 0) →
    /// (0 + 0, −4, −0, −0) = (0, −4, 0, 0)`. `lideal_commit_norm = 2`,
    /// `ideal_norm = 2`. `response_bits = 8`, `backtracking = 0` →
    /// `pow_dim2_deg_resp = 8 − 2 − 0 = 6`. `remain_initial = 64`.
    /// `random_aux_norm = 64 − 1 = 63`. `hd_extra_torsion = 2` →
    /// `remain = 256`. `degree_resp_inv = 1^{-1} mod 256 = 1`.
    /// `two_resp_length = 2`.
    #[test]
    fn compute_random_aux_orchestrator_with_two_adic_split() {
        let resp_quat = [
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(4),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        ];
        let result = compute_random_aux_norm_and_helpers::<8>(
            &resp_quat,
            &Uint::<8>::from_u64(4),
            &Uint::<8>::from_u64(2),
            &Uint::<8>::from_u64(7),
            0,
            8,
            2,
        )
        .expect("two-adic split case must succeed");
        assert_eq!(result.pow_dim2_deg_resp, 6);
        assert_eq!(result.random_aux_norm, Uint::<8>::from_u64(63));
        assert_eq!(result.degree_resp_inv, Uint::<8>::from_u64(1));
        assert_eq!(result.remain, Uint::<8>::from_u64(256));
        assert_eq!(result.lideal_com_resp.cached_norm, Uint::<8>::from_u64(2));
        assert_eq!(result.two_resp_length, 2);
    }

    /// Caller invariant violation: `lattice_content` does not divide
    /// `N_red(resp_quat)`. Must surface as `Err(Internal)`, not panic
    /// or silently produce a wrong result.
    #[test]
    fn compute_random_aux_orchestrator_rejects_non_dividing_lattice_content() {
        let resp_quat = [
            Int::<8>::from_i64(4),
            Int::<8>::from_i64(2),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        ]; // N_red = 20
        let result = compute_random_aux_norm_and_helpers::<8>(
            &resp_quat,
            &Uint::<8>::from_u64(7), // 20 not divisible by 7
            &Uint::<8>::from_u64(5),
            &Uint::<8>::from_u64(7),
            0,
            12,
            4,
        );
        assert!(
            matches!(result, Err(Error::Internal(msg)) if msg.contains("does not divide")),
            "non-dividing lattice_content must Err(Internal), got {result:?}",
        );
    }

    /// Caller invariant violation: `response_bits − two_resp_length −
    /// backtracking` underflows. Reject cleanly.
    #[test]
    fn compute_random_aux_orchestrator_rejects_budget_underflow() {
        let resp_quat = [
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(4),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        ]; // N_red = 16, v_2 = 4
        let result = compute_random_aux_norm_and_helpers::<8>(
            &resp_quat,
            &Uint::<8>::from_u64(1),
            &Uint::<8>::from_u64(2),
            &Uint::<8>::from_u64(7),
            5, // backtracking + v_2(16) = 5 + 4 = 9 > response_bits = 8 → underflow
            8,
            2,
        );
        assert!(
            matches!(result, Err(Error::Internal(msg)) if msg.contains("underflow")),
            "budget underflow must Err(Internal), got {result:?}",
        );
    }

    /// Forge S184 MINOR 8 closure: exercise the orchestrator with a
    /// `degree_odd_resp` ≥ 3 so the Step 11 `degree_resp_inv` is non-
    /// trivial. Previous orchestrator tests all landed `degree_odd_resp
    /// = 1` → `inv = 1` (trivially); this test pins the full integration
    /// across the modular-inverse helper at the orchestrator level.
    ///
    /// Inputs (p = 3, the smallest prime ≡ 3 mod 4 with `(1+p)/4 = 1`
    /// integer in the O_0 norm form): `resp_quat = (1, 1, 1, 1)` in
    /// O_0-basis coords. The reduced norm is `4·N(β) = 4 + 4 + 4 + 4 +
    /// (1+3)·1 + (1+3)·1 = 24`, so `N(β) = 6`. With
    /// `lattice_content = 2` and `lideal_commit_norm = 2`:
    /// - `degree_full_resp = 6 / 2 = 3`
    /// - `v_2(3) = 0`, `degree_odd_resp = 3`
    /// - `conjugate(1,1,1,1) = (1+1, −1, −1, −1) = (2, −1, −1, −1)`
    /// - `ideal_norm = 2 · 3 = 6`
    /// - `lideal_com_resp.cached_norm = 6`
    /// - `pow_dim2_deg_resp = 4 − 0 − 0 = 4`, `remain_initial = 16`
    /// - `random_aux_norm = 16 − 3 = 13`
    /// - `remain = 16 · 1 = 16` (hd_extra_torsion = 0)
    /// - `degree_resp_inv = 3^{−1} mod 16 = 11`  ← the non-trivial probe
    #[test]
    fn compute_random_aux_orchestrator_with_nontrivial_degree_resp_inv() {
        let resp_quat = [
            Int::<8>::from_i64(1),
            Int::<8>::from_i64(1),
            Int::<8>::from_i64(1),
            Int::<8>::from_i64(1),
        ];
        let result = compute_random_aux_norm_and_helpers::<8>(
            &resp_quat,
            &Uint::<8>::from_u64(2),
            &Uint::<8>::from_u64(2),
            &Uint::<8>::from_u64(3),
            0,
            4,
            0,
        )
        .expect("non-trivial-inverse case must succeed");
        assert_eq!(result.pow_dim2_deg_resp, 4);
        assert_eq!(result.random_aux_norm, Uint::<8>::from_u64(13));
        assert_eq!(
            result.degree_resp_inv,
            Uint::<8>::from_u64(11),
            "non-trivial degree_resp_inv must equal 3^{{-1}} mod 16 = 11",
        );
        assert_eq!(result.remain, Uint::<8>::from_u64(16));
        assert_eq!(result.lideal_com_resp.cached_norm, Uint::<8>::from_u64(6));
        assert_eq!(result.lideal_com_resp.denom, Uint::<8>::ONE);
        assert_eq!(result.conjugated_resp_quat[0], Int::<8>::from_i64(2));
        assert_eq!(result.conjugated_resp_quat[1], Int::<8>::from_i64(-1));
        assert_eq!(result.conjugated_resp_quat[2], Int::<8>::from_i64(-1));
        assert_eq!(result.conjugated_resp_quat[3], Int::<8>::from_i64(-1));
        assert_eq!(result.two_resp_length, 0);
    }

    /// Forge S184 M1 regression: `pow_dim2_deg_resp == 0` is the C ref's
    /// retry signal and MUST surface as `Ok` (not `Err`), even when the
    /// `remain_initial − degree_odd_resp` step would underflow. The
    /// caller checks the gate value to decide whether to consume the
    /// other fields or restart the signing loop. Earlier S184 code
    /// converted this expected retry into a hard error; this test pins
    /// the corrected C-faithful semantics.
    ///
    /// Inputs constructed to land exactly at gate=0 with non-trivial
    /// 2-adic strip: `resp_quat = (2, 0, 0, 0)`, `N_red = 4`,
    /// `lattice_content = 2` → `degree_full_resp = 2`. `v_2(2) = 1`,
    /// `degree_odd_resp = 1`. `response_bits = 2`, `backtracking = 1`
    /// → `pow_dim2_deg_resp = 2 − 1 − 1 = 0` (the gate).
    #[test]
    fn compute_random_aux_orchestrator_returns_ok_at_gate_zero_retry_signal() {
        let resp_quat = [
            Int::<8>::from_i64(2),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        ];
        let result = compute_random_aux_norm_and_helpers::<8>(
            &resp_quat,
            &Uint::<8>::from_u64(2),
            &Uint::<8>::from_u64(1),
            &Uint::<8>::from_u64(7),
            1,
            2,
            4,
        )
        .expect("gate-zero case must return Ok with the retry signal, not Err");
        assert_eq!(
            result.pow_dim2_deg_resp, 0,
            "gate value must be 0 to signal caller-side restart",
        );
        // Fields that remain meaningful at gate=0 (computed before Step 7).
        assert_eq!(result.two_resp_length, 1);
        assert_eq!(result.lideal_com_resp.cached_norm, Uint::<8>::from_u64(1));
        // random_aux_norm, remain, degree_resp_inv are documented as
        // undefined at gate=0; not asserted here to honor the contract.
    }

    /// L3/L5 monomorphization smoke — orchestrator body must compile and
    /// execute at wider LIMBS. Use the same zero-two-adic case scaled to
    /// LIMBS=12 (L3) and LIMBS=16 (L5).
    #[test]
    fn compute_random_aux_orchestrator_monomorphizes_at_l3_l5() {
        let resp_quat_l3 = [
            Int::<12>::from_i64(4),
            Int::<12>::from_i64(2),
            Int::<12>::from_i64(0),
            Int::<12>::from_i64(0),
        ];
        let r3 = compute_random_aux_norm_and_helpers::<12>(
            &resp_quat_l3,
            &Uint::<12>::from_u64(20),
            &Uint::<12>::from_u64(5),
            &Uint::<12>::from_u64(7),
            0,
            12,
            4,
        )
        .expect("L3 monomorphization must succeed");
        assert_eq!(r3.lideal_com_resp.cached_norm, Uint::<8>::from_u64(5));

        let resp_quat_l5 = [
            Int::<16>::from_i64(4),
            Int::<16>::from_i64(2),
            Int::<16>::from_i64(0),
            Int::<16>::from_i64(0),
        ];
        let r5 = compute_random_aux_norm_and_helpers::<16>(
            &resp_quat_l5,
            &Uint::<16>::from_u64(20),
            &Uint::<16>::from_u64(5),
            &Uint::<16>::from_u64(7),
            0,
            12,
            4,
        )
        .expect("L5 monomorphization must succeed");
        assert_eq!(r5.lideal_com_resp.cached_norm, Uint::<8>::from_u64(5));
    }

    // ── evaluate_random_aux_isogeny_signature stub tests ───────────────

    /// Stub probe at LIMBS=8 (L1 width) + Params=Lvl1: signature
    /// monomorphizes, dispatches to the early-return, message names the
    /// deferred body and the Clapotis dependency.
    #[test]
    fn evaluate_random_aux_isogeny_signature_stub_returns_unimplemented_at_l1() {
        use crate::params::lvl1::Level1;
        use crate::quaternion::ideal::LeftIdeal;
        use crate::rng::NistPqcRng;
        let lideal_com_resp = LeftIdeal::<8>::full_order();
        let random_aux_norm: Uint<8> = Uint::from_u64(7);
        let p: Uint<8> = Uint::from_u64(7);
        let cofactor: Uint<8> = Uint::from_u64(13);
        let witnesses: [Uint<8>; 2] = [Uint::from_u64(2), Uint::from_u64(3)];
        let mut rng = NistPqcRng::new(&[0xE7u8; 48]);
        let result = evaluate_random_aux_isogeny_signature::<Level1, 8, _>(
            &random_aux_norm,
            &lideal_com_resp,
            &p,
            Some(&cofactor),
            5,
            16,
            &witnesses,
            &mut rng,
        );
        assert!(
            matches!(&result, Err(Error::Unimplemented(msg)) if msg.contains("body deferred to S187")),
            "stub must return Err(Unimplemented) with deferred-body marker, got {result:?}",
        );
    }

    /// Stub probe at LIMBS=12 (L3 width) + Params=Lvl3: monomorphization
    /// smoke at wider LIMBS.
    #[test]
    fn evaluate_random_aux_isogeny_signature_stub_compiles_at_l3() {
        use crate::params::lvl3::Level3;
        use crate::quaternion::ideal::LeftIdeal;
        use crate::rng::NistPqcRng;
        let lideal_com_resp = LeftIdeal::<8>::full_order();
        let mut rng = NistPqcRng::new(&[0xE3u8; 48]);
        let result = evaluate_random_aux_isogeny_signature::<Level3, 12, _>(
            &Uint::<12>::from_u64(7),
            &lideal_com_resp,
            &Uint::<12>::from_u64(7),
            Some(&Uint::<12>::from_u64(13)),
            5,
            16,
            &[Uint::<12>::from_u64(2), Uint::<12>::from_u64(3)],
            &mut rng,
        );
        assert!(
            matches!(result, Err(Error::Unimplemented(_))),
            "L3 (LIMBS=12, Lvl3) monomorphization must dispatch to the stub",
        );
    }

    /// Stub probe at LIMBS=16 (L5 width) + Params=Lvl5: monomorphization
    /// smoke at L5 width.
    #[test]
    fn evaluate_random_aux_isogeny_signature_stub_compiles_at_l5() {
        use crate::params::lvl5::Level5;
        use crate::quaternion::ideal::LeftIdeal;
        use crate::rng::NistPqcRng;
        let lideal_com_resp = LeftIdeal::<8>::full_order();
        let mut rng = NistPqcRng::new(&[0xE5u8; 48]);
        let result = evaluate_random_aux_isogeny_signature::<Level5, 16, _>(
            &Uint::<16>::from_u64(7),
            &lideal_com_resp,
            &Uint::<16>::from_u64(7),
            Some(&Uint::<16>::from_u64(13)),
            5,
            16,
            &[Uint::<16>::from_u64(2), Uint::<16>::from_u64(3)],
            &mut rng,
        );
        assert!(
            matches!(result, Err(Error::Unimplemented(_))),
            "L5 (LIMBS=16, Lvl5) monomorphization must dispatch to the stub",
        );
    }

    /// `AuxIsogenyOutputs<P>::placeholder()` constructs and the marker
    /// is the expected `PhantomData`. Mirrors the `IdealToIsogenyResult`
    /// placeholder pattern.
    #[test]
    fn aux_isogeny_outputs_placeholder_constructs_across_levels() {
        use crate::params::{lvl1::Level1, lvl3::Level3, lvl5::Level5};
        let _l1 = AuxIsogenyOutputs::<Level1>::placeholder();
        let _l3 = AuxIsogenyOutputs::<Level3>::placeholder();
        let _l5 = AuxIsogenyOutputs::<Level5>::placeholder();
    }
}
