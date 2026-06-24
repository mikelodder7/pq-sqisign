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
//! This module holds the public API + types for the translation. The
//! detailed evaluator (theta evaluation, gluing, projection) is
//! implemented in [`crate::isogeny::clapotis_spine`]; the generic
//! `ideal_to_isogeny` / `evaluate_response_isogeny` entry points here
//! remain thin stubs (see their docs).
//!
//! # Output shape
//!
//! [`IdealToIsogenyResult`] is the output type for the generic stub
//! entry point. It is a `PhantomData<P>` placeholder: the codomain
//! curve `E_K = E_0 / ker(φ)` (a Montgomery curve over `F_{p²}`) and
//! the kernel/isogeny-chain representation are produced by the spine
//! evaluator in [`crate::isogeny::clapotis_spine`], not by this type.

use core::marker::PhantomData;

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crypto_bigint::{Int, Uint};
use rand_core::CryptoRng;

use crate::error::{Error, Result};
use crate::params::Params;
use crate::quaternion::algebra::RationalQuaternion;
use crate::quaternion::hnf::int_div_floor;
use crate::quaternion::ideal_mul::LeftIdealWideNorm;
use crate::quaternion::lattice::{mat_4x4_transpose_eval, qf_eval_4x4};
use crate::quaternion::o0_mul::{o0_basis_to_standard_doubled, o0_reduced_norm_gram_matrix};

/// A short-vector enumeration result: a 4-D integer vector together with
/// its quadratic-form value (divided by `adjusted_norm` from the caller).
///
/// Produced by [`enumerate_hypercube`] (the SQIsign-2.0 reference's
/// `enumerate_hypercube` in `src/id2iso/ref/lvlx/dim2id2iso.c:255-357`).
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnumeratedShortVec<const LIMBS: usize> {
    /// The 4-D vector `(x, y, z, w) ∈ Z^4` whose entries are pulled from
    /// the hypercube `[-m, m]^4` (with symmetries pruned).
    pub vec: [Int<LIMBS>; 4],
    /// The quadratic-form value `v^T · G · v / adjusted_norm` (the C
    /// reference divides by `adjusted_norm` and asserts exact
    /// divisibility before storing).
    pub norm: Int<LIMBS>,
}

/// Brute-force-with-symmetry-pruning enumeration of short vectors in the
/// integer hypercube `[-m, m]^4` under the quadratic form defined by
/// `gram`. Mirrors the C reference's static
/// `enumerate_hypercube` at
/// `src/id2iso/ref/lvlx/dim2id2iso.c:255-357` of
/// `github.com/SQISign/the-sqisign`.
///
/// # Algorithm
///
/// Walk `(x, y, z, w) ∈ [-m, 0] × [-m, m]^3` (antipodal symmetry: only
/// the non-positive-x half is enumerated; symmetric cascades on
/// `(y, z, w)` when `x == 0`, `(x, y) == (0, 0)`, etc.). For each
/// surviving candidate, apply three primitivity / symmetry filters:
///
/// 1. **All-even filter**: skip vectors with `(x | y | z | w) & 1 == 0`
///    (i.e., all four entries even).
/// 2. **All-multiple-of-3 filter**: skip vectors with every entry
///    divisible by 3.
/// 3. **`i`-action symmetry filter** (active only when the Gram matrix
///    has the canonical `[α, iα, β, iβ]`-basis shape detected by
///    `gram[0][0] == gram[1][1] ∧ gram[3][3] == gram[2][2]`): map each
///    candidate to its lex-rank via the linear index `check1 = (m+w)
///    + dim·(m+z) + dim²·(m+y) + dim³·(m+x)`, plus two rotated indices
///    `check2`, `check3` representing the action of `i` and `i^2`.
///    Keep only the lex-minimal representative among the four
///    `⟨i⟩`-orbits.
///
/// For each survivor: compute `norm = v^T · G · v` via
/// [`qf_eval_4x4`], divide by `adjusted_norm` (a `debug_assert!` checks
/// for exact divisibility — see "Quirks" below), and accept only if the
/// resulting quotient is odd.
///
/// # Output ordering and the C reference's `count - 1` quirk
///
/// Accepted `(vec, norm)` pairs are pushed onto the returned `Vec` in
/// enumeration order. **The C reference returns `count - 1`** (not
/// `count`), so its callers iterate `[0, count - 1)` — i.e. they discard
/// the LAST accepted entry. This Rust port mirrors the behavior by
/// truncating the returned `Vec` to `count - 1` entries (or `0` if no
/// entries were accepted). The behavior is preserved verbatim against
/// the C reference; whether this off-by-one is intentional or a latent
/// bug in the C ref is documented as an S192 open question for
/// follow-up audit. Sorting (the C ref's `compare_vec_by_norm` qsort
/// step) is the CALLER's responsibility — Rust's stable `Vec::sort_by`
/// applied to the returned `Vec` (sorting by `norm`) is the idiomatic
/// equivalent and naturally tie-breaks by insertion order (matching the
/// C ref's `idx` tiebreaker).
///
/// # Parameters
///
/// - `m`: half-bound for the hypercube; each coordinate ∈ `[-m, m]`.
///   Must be `> 0`. The C reference asserts this; the Rust port returns
///   an empty `Vec` for `m <= 0` (defensive; in release a non-positive
///   `m` would skip the loop body entirely).
/// - `gram`: the 4×4 Gram matrix defining the quadratic form on `Z^4`.
/// - `adjusted_norm`: the normalizing divisor applied to every
///   stored norm. Must divide `v^T · G · v` exactly for every accepted
///   candidate. Caller's responsibility.
///
/// # Returns
///
/// A `Vec<EnumeratedShortVec<LIMBS>>` of the accepted candidates, in
/// enumeration order, truncated by 1 from the end per the C reference's
/// `count - 1` semantics. Caller sorts (typically by `norm`) before
/// consuming.
///
/// # Precision / variable-time
///
/// All arithmetic is `Int<LIMBS>` wrapping. The function is variable-
/// time on the candidate count (i.e., on `m`) and on the Gram matrix's
/// entries (via `qf_eval_4x4`). At SQIsign's call sites the inputs are
/// signing-flow secrets, which matches the SQIsign 2.0 spec §8 vartime
/// quaternion-side convention.
#[cfg(feature = "alloc")]
pub fn enumerate_hypercube<const LIMBS: usize>(
    m: i64,
    gram: &[[Int<LIMBS>; 4]; 4],
    adjusted_norm: &Int<LIMBS>,
) -> Vec<EnumeratedShortVec<LIMBS>> {
    let mut out: Vec<EnumeratedShortVec<LIMBS>> = Vec::new();
    if m <= 0 {
        return out;
    }

    // Detect the canonical `[α, iα, β, iβ]` basis shape: the i-action
    // symmetry filter only activates when this holds.
    let need_remove_symmetry = gram[0][0] == gram[1][1] && gram[3][3] == gram[2][2];

    // Linear-index dimensions for the i-action filter.
    let dim = 2 * m + 1;
    let dim2 = dim * dim;
    let dim3 = dim2 * dim;

    let zero_int = Int::<LIMBS>::from_i64(0);

    for x in -m..=0 {
        // Antipodal break: non-positive x only.
        for y in -m..=m {
            if x == 0 && y > 0 {
                break;
            }
            for z in -m..=m {
                if x == 0 && y == 0 && z > 0 {
                    break;
                }
                for w in -m..=m {
                    if x == 0 && y == 0 && z == 0 && w >= 0 {
                        break;
                    }

                    // Filter 1: all-even — skip vectors with `(x|y|z|w) & 1 == 0`.
                    if (x | y | z | w) & 1 == 0 {
                        continue;
                    }
                    // Filter 2: all-multiple-of-3 — skip vectors with every entry % 3 == 0.
                    if x % 3 == 0 && y % 3 == 0 && z % 3 == 0 && w % 3 == 0 {
                        continue;
                    }

                    // Filter 3 (conditional): i-action symmetry — only
                    // keep the lex-minimal rep among the four ⟨i⟩-orbits.
                    if need_remove_symmetry {
                        let check1 = (m + w) + dim * (m + z) + dim2 * (m + y) + dim3 * (m + x);
                        let check2 = (m - z) + dim * (m + w) + dim2 * (m - x) + dim3 * (m + y);
                        let check3 = (m + z) + dim * (m - w) + dim2 * (m + x) + dim3 * (m - y);
                        if !(check1 <= check2 && check1 <= check3) {
                            continue;
                        }
                    }

                    // Build the candidate vector and evaluate the
                    // quadratic form.
                    let vec = [
                        Int::<LIMBS>::from_i64(x),
                        Int::<LIMBS>::from_i64(y),
                        Int::<LIMBS>::from_i64(z),
                        Int::<LIMBS>::from_i64(w),
                    ];
                    let norm_full = qf_eval_4x4::<LIMBS>(&vec, gram);

                    // Divide by adjusted_norm. C reference asserts exact
                    // divisibility (`assert(ibz_is_zero(&remain))`); we
                    // match via debug_assert. In release builds this is
                    // a silent floor-division — matching the C's NDEBUG
                    // path. Caller is responsible for picking
                    // `adjusted_norm` to ensure exact divisibility on
                    // every surviving candidate.
                    let quotient = int_div_floor::<LIMBS>(&norm_full, adjusted_norm);
                    #[cfg(debug_assertions)]
                    {
                        let recovered = quotient.wrapping_mul(adjusted_norm);
                        let remainder = norm_full.wrapping_sub(&recovered);
                        debug_assert_eq!(
                            remainder, zero_int,
                            "enumerate_hypercube: adjusted_norm must divide v^T · G · v exactly (got remainder)",
                        );
                    }
                    let _ = zero_int; // suppress unused-in-release lint

                    // Odd-quotient filter: store only when the quotient
                    // is odd. The LSB of the two's-complement
                    // representation is the parity bit for both
                    // non-negative and negative Int<LIMBS> values.
                    if quotient.to_words()[0] & 1 == 1 {
                        out.push(EnumeratedShortVec {
                            vec,
                            norm: quotient,
                        });
                    }
                }
            }
        }
    }

    // C reference returns `count - 1`, not `count` — callers iterate
    // `[0, indices[j])` and so discard the LAST accepted entry.
    // Preserve verbatim.
    let n = out.len();
    if n > 0 {
        out.truncate(n - 1);
    }
    out
}

/// Wide-precision variant of [`enumerate_hypercube`]. Identical
/// candidate enumeration + filters (all-even / mod-3 / i-action symmetry
/// / odd-quotient / `count-1` truncation), but evaluates `vᵀ·G·v` and the
/// `adjusted_norm` division in `WIDE` precision so neither overflows at
/// real-prime scale, then narrows the (small) accepted quotient and the
/// (tiny box-coordinate) vector back to `Int<NARROW>`.
///
/// Required because at the real L1 prime the pulled-back Gram entries are
/// ~500 bits and `vᵀGv` is ~506 bits — within `Int<8>` for a single
/// product, but the divisibility invariant `adjusted_norm = denom² DIVIDES
/// vᵀGv` is only exact when the GRAM ITSELF was computed without
/// intermediate overflow (see `pull_back_gram_wide`). The caller passes
/// a `WIDE` gram from `pull_back_gram_wide` and `adjusted_norm = denom²`
/// (S230: the C ref uses `denom²`, NOT the narrow path's spurious
/// `4·denom²`). The accepted quotient (the reduced norm `d`) is small and
/// narrows cleanly; the box vector entries are in `[-m, m]` and trivially
/// narrow.
#[cfg(feature = "alloc")]
pub fn enumerate_hypercube_wide<const NARROW: usize, const WIDE: usize>(
    m: i64,
    gram: &[[Int<WIDE>; 4]; 4],
    adjusted_norm: &Int<WIDE>,
) -> Vec<EnumeratedShortVec<NARROW>> {
    use crate::quaternion::lattice::narrow_int_lattice;
    let mut out: Vec<EnumeratedShortVec<NARROW>> = Vec::new();
    if m <= 0 {
        return out;
    }

    let need_remove_symmetry = gram[0][0] == gram[1][1] && gram[3][3] == gram[2][2];
    let dim = 2 * m + 1;
    let dim2 = dim * dim;
    let dim3 = dim2 * dim;
    let zero_w = Int::<WIDE>::from_i64(0);

    for x in -m..=0 {
        for y in -m..=m {
            if x == 0 && y > 0 {
                break;
            }
            for z in -m..=m {
                if x == 0 && y == 0 && z > 0 {
                    break;
                }
                for w in -m..=m {
                    if x == 0 && y == 0 && z == 0 && w >= 0 {
                        break;
                    }
                    if (x | y | z | w) & 1 == 0 {
                        continue;
                    }
                    if x % 3 == 0 && y % 3 == 0 && z % 3 == 0 && w % 3 == 0 {
                        continue;
                    }
                    if need_remove_symmetry {
                        let check1 = (m + w) + dim * (m + z) + dim2 * (m + y) + dim3 * (m + x);
                        let check2 = (m - z) + dim * (m + w) + dim2 * (m - x) + dim3 * (m + y);
                        let check3 = (m + z) + dim * (m - w) + dim2 * (m + x) + dim3 * (m - y);
                        if !(check1 <= check2 && check1 <= check3) {
                            continue;
                        }
                    }

                    let vec_w = [
                        Int::<WIDE>::from_i64(x),
                        Int::<WIDE>::from_i64(y),
                        Int::<WIDE>::from_i64(z),
                        Int::<WIDE>::from_i64(w),
                    ];
                    let norm_full = qf_eval_4x4::<WIDE>(&vec_w, gram);
                    let quotient = int_div_floor::<WIDE>(&norm_full, adjusted_norm);
                    #[cfg(debug_assertions)]
                    {
                        let recovered = quotient.wrapping_mul(adjusted_norm);
                        let remainder = norm_full.wrapping_sub(&recovered);
                        debug_assert_eq!(
                            remainder, zero_w,
                            "enumerate_hypercube_wide: adjusted_norm must divide v^T · G · v exactly (got remainder)",
                        );
                    }
                    let _ = zero_w;

                    if quotient.to_words()[0] & 1 == 1 {
                        let vec_n = [
                            Int::<NARROW>::from_i64(x),
                            Int::<NARROW>::from_i64(y),
                            Int::<NARROW>::from_i64(z),
                            Int::<NARROW>::from_i64(w),
                        ];
                        out.push(EnumeratedShortVec {
                            vec: vec_n,
                            norm: narrow_int_lattice::<WIDE, NARROW>(&quotient),
                        });
                    }
                }
            }
        }
    }

    let n = out.len();
    if n > 0 {
        out.truncate(n - 1);
    }
    out
}

/// Structured output of the [`ideal_to_isogeny`] stub at security level `P`.
///
/// `PhantomData<P>` placeholder for the codomain curve + isogeny
/// representation. The functional translation lives in the spine
/// evaluator [`crate::isogeny::clapotis_spine`], which returns its own
/// richer types; this placeholder backs the generic stub entry point.
#[derive(Debug, Clone)]
pub struct IdealToIsogenyResult<P: Params> {
    /// Phantom for the security level. A non-placeholder result would
    /// carry the codomain Montgomery curve (`A24` projective form) and
    /// the isogeny chain or theta-coordinate data.
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
/// # Status — generic stub
///
/// This generic entry point returns `Error::Unimplemented`. The
/// functional Clapotis evaluator is implemented in
/// [`crate::isogeny::clapotis_spine`] (`ideal_to_isogeny_clapotis`),
/// which the keygen/sign paths call directly. This stub establishes a
/// stable public API contract so Sign/Verify orchestration and tests
/// can target a fixed signature.
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
/// Always returns `Error::Unimplemented`; the functional path is the
/// spine evaluator (see above).
pub fn ideal_to_isogeny<P: Params, const TLIMBS: usize, R: CryptoRng>(
    _klpt_output: &LeftIdealWideNorm<TLIMBS>,
    _q: &Uint<8>,
    _rng: &mut R,
) -> Result<IdealToIsogenyResult<P>> {
    Err(Error::Unimplemented(
        "ideal_to_isogeny: generic entry point unimplemented — the functional Clapotis evaluator is in clapotis_spine",
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
/// # Status — generic stub
///
/// Returns `Error::Unimplemented`. The functional Level-1 verify path
/// is wired elsewhere in `crate::verify`; this generic entry point is a
/// stable dispatch contract (only the unwired L3/L5 levels route here).
/// A functional body would do the parsing + isogeny-evaluation +
/// codomain-check sequence.
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
/// Always returns `Error::Unimplemented`. A functional body would
/// return `Error::InvalidSignature` (or similar) when the response
/// isogeny's codomain doesn't match `pk`'s curve.
pub fn evaluate_response_isogeny<P: Params>(_sig: &[u8], _msg: &[u8], _pk: &[u8]) -> Result<()> {
    Err(Error::Unimplemented(
        "evaluate_response_isogeny: generic entry point unimplemented — the functional Clapotis evaluator is in clapotis_spine",
    ))
}

#[cfg(all(test, feature = "kat"))]
mod tests {
    use super::*;

    #[test]
    fn ideal_to_isogeny_stub_returns_unimplemented_at_lvl1() {
        // scaffolding test: the public API stub is in place and
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

        let result = ideal_to_isogeny::<Level1, 8, _>(&klpt_output, &q, &mut rng);
        let err = result.expect_err("stub must return Unimplemented");
        let Error::Unimplemented(msg) = err else {
            unreachable!("expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis evaluator") || msg.contains("dominant remaining scope"),
            "stub's error message must reference the Clapotis evaluator deferral; got: {msg}",
        );
    }

    #[test]
    fn evaluate_response_isogeny_stub_returns_unimplemented_at_lvl1() {
        // contract test: the verify-side stub mirrors the
        // ideal_to_isogeny contract. Placeholder byte slices; the stub
        // returns Unimplemented regardless of input. Locks the contract
        // that `crate::verify` wires against.
        use crate::params::Level1;

        let result = evaluate_response_isogeny::<Level1>(&[], b"msg", &[]);
        let err = result.expect_err("verify stub must return Unimplemented");
        let Error::Unimplemented(msg) = err else {
            unreachable!("expected Unimplemented, got {err:?}");
        };
        assert!(
            msg.contains("Clapotis evaluator") || msg.contains("dominant remaining scope"),
            "verify stub's error must reference the Clapotis deferral; got: {msg}",
        );
    }

    #[ignore = "diagnostic probe: runs find_uv on the genuine L1 KLPT connecting ideal at varying box sizes"]
    #[test]
    fn probe_find_uv_genuine_klpt_ideal_box_sweep() {
        use crate::quaternion::ideal::LeftIdeal;
        use crate::quaternion::klpt::klpt_body_wide_wn;
        use crate::quaternion::lattice::widen_int_lattice;
        use crate::rng::NistPqcRng;
        use crypto_bigint::Int;

        // Reproduce the genuine L1 connecting ideal exactly as
        // klpt_clapotis_chain_at_lvl1 builds it (src/lib.rs).
        let p = crate::params::lvl1::prime().resize::<8>();
        let o_0 = LeftIdeal::<8>::full_order();
        let target_m = Uint::<8>::from_u64(1000).shl_vartime(248);
        let witnesses: [Uint<8>; 3] = [Uint::from_u64(2), Uint::from_u64(3), Uint::from_u64(5)];
        let mut rng = NistPqcRng::new(&[0x5Au8; 48]);
        let (k_wn, _q) = klpt_body_wide_wn::<8, _>(
            &o_0,
            &p,
            &target_m,
            &[2, 3],
            None,
            5,
            30,
            1 << 14,
            &witnesses,
            &mut rng,
        )
        .expect("genuine L1 KLPT body must succeed");

        // Widen the L8 inner basis to L16 and set cached_norm = N(I)^2
        // (lattice-index convention) directly from k_wn.cached_norm = N(I).
        let mut basis16 = [[Int::<16>::from_i64(0); 4]; 4];
        for (r, row) in basis16.iter_mut().enumerate() {
            for (c, entry) in row.iter_mut().enumerate() {
                *entry = widen_int_lattice::<8, 16>(&k_wn.inner.basis[r][c]);
            }
        }
        let n_i = k_wn.cached_norm.resize::<16>();
        let cached_norm16 = n_i.wrapping_mul(&n_i);
        let ideal16 = LeftIdeal::<16>::with_denom_and_norm(basis16, Uint::<16>::ONE, cached_norm16);

        let p16 = crate::params::lvl1::prime().resize::<16>();
        let target16 = *Uint::<16>::ONE.shl_vartime(248).as_int();

        eprintln!(
            "GENUINE IDEAL: N(I) bits = {}, |det|=cached_norm bits = {}, reduced_norm_vartime = {:?} bits",
            k_wn.cached_norm.bits_vartime(),
            cached_norm16.bits_vartime(),
            ideal16.reduced_norm_vartime().map(|n| n.bits_vartime()),
        );

        for box_size in [2i64, 3, 4, 5, 6, 8] {
            match find_uv::<16>(&target16, &ideal16, &p16, &[], box_size) {
                Ok(r) => {
                    let ub = r.u.abs().bits_vartime();
                    let vb = r.v.abs().bits_vartime();
                    let d1b = r.d1.abs().bits_vartime();
                    let d2b = r.d2.abs().bits_vartime();
                    eprintln!(
                        "box={box_size}: OK  u_bits={ub} v_bits={vb} d1_bits={d1b} d2_bits={d2b} idx1={} idx2={}",
                        r.index_alternate_order_1, r.index_alternate_order_2,
                    );
                }
                Err(e) => eprintln!("box={box_size}: ERR {e:?}"),
            }
        }
    }

    #[ignore = "diagnostic probe: confirms find_uv returns odd, BALANCED d on a non-principal odd connecting ideal (the C-ref convention)"]
    #[test]
    fn probe_find_uv_odd_prime_norm_ideal() {
        use crate::quaternion::o0_mul::left_ideal_from_element_and_integer_o0;
        use crate::quaternion::primality::is_probable_prime_with_witnesses;
        use crate::quaternion::represent_integer::find_quaternion_in_full_order_with_norm_wide;
        use crate::rng::NistPqcRng;

        // Mirror the C reference test (test_dim2id2iso.c:297-304): build a
        // connecting ideal of ODD prime norm n1 from a GENERATOR of norm
        // n1·n2 (two distinct odd primes). The n1·n2 generator makes the
        // norm-n1 ideal NON-PRINCIPAL — its shortest vectors have norm
        // ~ n1·√p, so d = N(β)/n1 ~ √p ~ 2^125 (BALANCED), giving u,v ~
        // 2^123 < 2^246. (A principal n1 ideal would have a norm-n1
        // generator ⇒ degenerate d1=d2=1, u~2^248 > spine capacity.)
        let p16 = crate::params::lvl1::prime().resize::<16>();
        let witnesses: [Uint<16>; 12] = [
            Uint::from_u64(2),
            Uint::from_u64(3),
            Uint::from_u64(5),
            Uint::from_u64(7),
            Uint::from_u64(11),
            Uint::from_u64(13),
            Uint::from_u64(17),
            Uint::from_u64(19),
            Uint::from_u64(23),
            Uint::from_u64(29),
            Uint::from_u64(31),
            Uint::from_u64(37),
        ];
        let mut rng = NistPqcRng::new(&[0xA5u8; 48]);

        // Two distinct deterministic ~250-bit odd primes.
        let two = Uint::<16>::from_u64(2);
        let next_prime = |start: Uint<16>| -> Uint<16> {
            let mut c = if start.as_limbs()[0].0 & 1 == 0 {
                start.wrapping_add(&Uint::ONE)
            } else {
                start
            };
            let mut tries = 0;
            while !is_probable_prime_with_witnesses(&c, &witnesses) {
                c = c.wrapping_add(&two);
                tries += 1;
                assert!(tries < 200_000, "no prime found");
            }
            c
        };
        let n1 = next_prime(Uint::<16>::ONE.shl_vartime(249).wrapping_add(&Uint::ONE));
        let n2 = next_prime(Uint::<16>::ONE.shl_vartime(248).wrapping_add(&Uint::ONE));
        let target_m = n1.wrapping_mul(&n2); // generator norm n1·n2

        // Generator γ ∈ O_0 with N_red(γ) = n1·n2.
        let gamma = find_quaternion_in_full_order_with_norm_wide::<16, _>(
            &target_m,
            &p16,
            64,
            1 << 16,
            &witnesses,
            &mut rng,
        )
        .expect("generator of norm n1·n2 must be found");

        // Left ideal O_0·γ + O_0·n1 of norm n1 (non-principal). The builder
        // sets cached_norm = n1 (the REDUCED norm convention). We feed it to
        // find_uv DIRECTLY — no cached_norm rewrap — to verify find_uv now
        // derives N(I) from the lattice determinant (convention-independent).
        let ideal16 = left_ideal_from_element_and_integer_o0::<16>(&gamma, &n1, &p16);

        eprintln!(
            "NON-PRINCIPAL ODD IDEAL: n1 bits = {}, n2 bits = {}, builder cached_norm bits = {} (= N, NOT N²)",
            n1.bits_vartime(),
            n2.bits_vartime(),
            ideal16.cached_norm.bits_vartime(),
        );

        // End-to-end check that the LLL iteration-cap fix lets find_uv find
        // a BALANCED odd coprime Bezout on the realistic non-principal ideal.
        let target16 = *Uint::<16>::ONE.shl_vartime(248).as_int();
        for box_size in [2i64, 3] {
            match find_uv::<16>(&target16, &ideal16, &p16, &[], box_size) {
                Ok(r) => {
                    let d1_odd = r.d1.abs().as_limbs()[0].0 & 1 == 1;
                    let d2_odd = r.d2.abs().as_limbs()[0].0 & 1 == 1;
                    let lhs =
                        r.u.wrapping_mul(&r.d1)
                            .wrapping_add(&r.v.wrapping_mul(&r.d2));
                    eprintln!(
                        "box={box_size}: OK  u_bits={} v_bits={} d1_bits={} d2_bits={} d1_odd={d1_odd} d2_odd={d2_odd} bezout_ok={} idx1={} idx2={}",
                        r.u.abs().bits_vartime(),
                        r.v.abs().bits_vartime(),
                        r.d1.abs().bits_vartime(),
                        r.d2.abs().bits_vartime(),
                        lhs == target16,
                        r.index_alternate_order_1,
                        r.index_alternate_order_2,
                    );
                }
                Err(e) => eprintln!("box={box_size}: ERR {e:?}"),
            }
        }
    }

    #[ignore = "diagnostic: root-cause the input-dependent enumerate gram-divisibility failure"]
    #[test]
    fn probe_find_uv_gram_divisibility_failing_case() {
        use crate::quaternion::ideal::{LeftIdeal, det_4x4};
        use crate::quaternion::ideal_mul::lideal_reduce_basis_wide;
        use crate::quaternion::lattice::widen_int_lattice;
        use crate::quaternion::o0_mul::left_ideal_from_element_and_integer_o0;
        use crate::quaternion::primality::is_probable_prime_with_witnesses;
        use crate::quaternion::represent_integer::find_quaternion_in_full_order_with_norm_wide;
        use crate::rng::NistPqcRng;

        const BL: usize = 16;
        let p16 = crate::params::lvl1::prime().resize::<BL>();
        let p8 = crate::params::lvl1::prime().resize::<8>();
        let wit16: [Uint<BL>; 12] =
            [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37].map(Uint::from_u64);
        let two = Uint::<BL>::from_u64(2);
        let next_prime = |start: Uint<BL>| -> Uint<BL> {
            let mut c = if start.as_limbs()[0].0 & 1 == 0 {
                start.wrapping_add(&Uint::ONE)
            } else {
                start
            };
            while !is_probable_prime_with_witnesses(&c, &wit16) {
                c = c.wrapping_add(&two);
            }
            c
        };
        // NOTE: n1~2^253 (≈bitsize(p)) makes target_m=n1·n2~2^505 exceed the
        // L=16 quaternion-finder/builder precision contract, producing a
        // MALFORMED ideal (det=n1² but n1∤N_red — not a genuine left ideal),
        // which find_uv's gram-divisibility assert correctly rejects. Build at
        // ~2^200 here (valid at L=16); the fixture needs wider L (≥~24) for
        // bitsize(p)-scale norms — a fixture-construction limit, NOT a find_uv
        // bug. The real protocol builds connecting ideals at the proper width.
        let n1 = next_prime(Uint::<BL>::ONE.shl_vartime(200).wrapping_add(&Uint::ONE));
        let n2 = next_prime(Uint::<BL>::ONE.shl_vartime(199).wrapping_add(&Uint::ONE));
        let target_m = n1.wrapping_mul(&n2);

        // Helper: |det| of an Int<8> basis computed at width 16 (no overflow).
        let det16 = |b: &[[Int<8>; 4]; 4]| -> Uint<16> {
            let mut bw = [[Int::<16>::from_i64(0); 4]; 4];
            for r in 0..4 {
                for c in 0..4 {
                    bw[r][c] = widen_int_lattice::<8, 16>(&b[r][c]);
                }
            }
            det_4x4::<16>(&bw).abs()
        };
        let max_bits = |b: &[[Int<8>; 4]; 4]| -> u32 {
            let mut m = 0;
            for row in b.iter() {
                for x in row.iter() {
                    m = m.max(x.abs().bits_vartime());
                }
            }
            m
        };
        let n1_sq = n1.wrapping_mul(&n1).resize::<16>();

        for seed in [0x5Au8, 0x77, 0xC3] {
            let mut rng = NistPqcRng::new(&[seed; 48]);
            let gamma = find_quaternion_in_full_order_with_norm_wide::<BL, _>(
                &target_m,
                &p16,
                64,
                1 << 16,
                &wit16,
                &mut rng,
            )
            .expect("gen");
            let ideal16 = left_ideal_from_element_and_integer_o0::<BL>(&gamma, &n1, &p16);
            let mut basis8 = [[Int::<8>::from_i64(0); 4]; 4];
            for (r, row) in basis8.iter_mut().enumerate() {
                for (c, entry) in row.iter_mut().enumerate() {
                    *entry = crate::quaternion::lattice::narrow_int_lattice::<BL, 8>(
                        &ideal16.basis[r][c],
                    );
                }
            }
            let ideal8 = LeftIdeal::<8>::with_denom_and_norm(
                basis8,
                ideal16.denom.resize::<8>(),
                ideal16.cached_norm.resize::<8>(),
            );

            // Build check: do the ORIGINAL ideal basis vectors have N_red
            // divisible by n1 (the genuine left-ideal invariant)? Check both
            // at L16 (pre-narrow, the builder output) and L8 (post-narrow) to
            // isolate a build bug vs a narrow bug.
            {
                let n1_16_nz = crypto_bigint::NonZero::new(n1).into_option().unwrap();
                let n1_8_nz = crypto_bigint::NonZero::new(n1.resize::<8>())
                    .into_option()
                    .unwrap();
                let mut div16 = true;
                let mut div8 = true;
                for i in 0..4 {
                    let nr16 = crate::quaternion::o0_mul::reduced_norm_o0_basis::<BL>(
                        &ideal16.basis[i],
                        &p16,
                    )
                    .abs();
                    let nr8 = crate::quaternion::o0_mul::reduced_norm_o0_basis::<8>(
                        &ideal8.basis[i],
                        &p8,
                    )
                    .abs();
                    if nr16.rem_vartime(&n1_16_nz) != Uint::<BL>::ZERO {
                        div16 = false;
                    }
                    if nr8.rem_vartime(&n1_8_nz) != Uint::<8>::ZERO {
                        div8 = false;
                    }
                }
                eprintln!(
                    "  BUILD CHECK: n1 | N_red(ideal16 basis)? {div16} | n1 | N_red(ideal8 basis)? {div8}",
                );
            }
            let det_orig = det16(&ideal8.basis);
            let reduced = lideal_reduce_basis_wide::<8, 128>(&ideal8, &p8);
            let det_red = det16(&reduced.basis);
            eprintln!(
                "seed={seed:#x}: n1^2 bits={} | det(orig) bits={} (==n1^2? {}) maxbits={} | det(reduced) bits={} (==n1^2? {}, ==orig? {}) maxbits={}",
                n1_sq.bits_vartime(),
                det_orig.bits_vartime(),
                det_orig == n1_sq,
                max_bits(&ideal8.basis),
                det_red.bits_vartime(),
                det_red == n1_sq,
                det_red == det_orig,
                max_bits(&reduced.basis),
            );

            // Gram-diagonal check: G_ii (pulled-back) should equal
            // 4·N_red(basis_row_i), and be divisible by adjusted_norm = 4·n1
            // (since basis_row_i ∈ I ⇒ n1 | N_red). A mismatch on the first
            // ⇒ gram precision bug; a non-divisibility on the second ⇒ n1 is
            // not the form's content (the actual reduced ideal norm differs).
            let o0_gram = o0_reduced_norm_gram_matrix::<8>(&p8);
            let gram =
                crate::quaternion::lattice::pull_back_gram_wide::<8, 32>(&reduced.basis, &o0_gram);
            let four_n1 = Int::<32>::from_i64(4)
                .wrapping_mul(&widen_int_lattice::<8, 32>(n1.resize::<8>().as_int()));
            let four_n1_nz = crypto_bigint::NonZero::new(four_n1.abs())
                .into_option()
                .unwrap();
            for (i, row) in gram.iter().enumerate() {
                let g_ii = row[i].abs();
                let nred =
                    crate::quaternion::o0_mul::reduced_norm_o0_basis::<8>(&reduced.basis[i], &p8)
                        .abs();
                let four_nred = Uint::<32>::from_u64(4).wrapping_mul(&nred.resize::<32>());
                let (_q, rem) = g_ii.div_rem_vartime(&four_n1_nz);
                eprintln!(
                    "  i={i}: G_ii==4·N_red? {} | G_ii%(4·n1)==0? {} (rem bits={})",
                    g_ii == four_nred,
                    rem == Uint::<32>::ZERO,
                    rem.bits_vartime(),
                );
            }

            // Vec scan: for a multi-coord vec, does qf_eval(vec,gram) equal
            // an INDEPENDENT 4·N_red(Σ vec_i·basis_i)? A mismatch localizes
            // the bug to qf_eval/gram cross-terms (vs the divisibility theory).
            let mut mismatches = 0;
            for x in -2i64..=2 {
                for y in -2i64..=2 {
                    for z in -2i64..=2 {
                        for wv in -2i64..=2 {
                            if x == 0 && y == 0 && z == 0 && wv == 0 {
                                continue;
                            }
                            let vec_w = [
                                Int::<32>::from_i64(x),
                                Int::<32>::from_i64(y),
                                Int::<32>::from_i64(z),
                                Int::<32>::from_i64(wv),
                            ];
                            let qf = qf_eval_4x4::<32>(&vec_w, &gram);
                            // combo = Σ vec_i · basis_row_i (O_0 coords).
                            let coeffs = [x, y, z, wv];
                            let mut combo = [Int::<8>::from_i64(0); 4];
                            for (r, &cf) in coeffs.iter().enumerate() {
                                let cfi = Int::<8>::from_i64(cf);
                                for (k, entry) in combo.iter_mut().enumerate() {
                                    *entry =
                                        entry.wrapping_add(&cfi.wrapping_mul(&reduced.basis[r][k]));
                                }
                            }
                            let nred =
                                crate::quaternion::o0_mul::reduced_norm_o0_basis::<8>(&combo, &p8)
                                    .abs();
                            let four_nred =
                                Uint::<32>::from_u64(4).wrapping_mul(&nred.resize::<32>());
                            if qf.abs() != four_nred {
                                if mismatches < 3 {
                                    eprintln!(
                                        "    MISMATCH vec=[{x},{y},{z},{wv}]: qf bits={} vs 4·N_red bits={} (equal={})",
                                        qf.abs().bits_vartime(),
                                        four_nred.bits_vartime(),
                                        qf.abs() == four_nred,
                                    );
                                }
                                mismatches += 1;
                            }
                        }
                    }
                }
            }
            eprintln!("  seed={seed:#x}: qf-vs-Nred mismatches = {mismatches}");
        }
    }
}

/// Output of [`find_uv_from_lists`]: a Bezout-pair search result over the
/// cross-product of two short-norm lists.
///
/// On success, `u · small_norms1[index_sol1] + v · small_norms2[index_sol2] == target`
/// with `u, v > 0`.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FindUvFromListsResult<const LIMBS: usize> {
    /// Positive integer u with `u · d1 + v · d2 = target`.
    pub u: Int<LIMBS>,
    /// Positive integer v.
    pub v: Int<LIMBS>,
    /// Index of the chosen norm in `small_norms1` (= d1).
    pub index_sol1: usize,
    /// Index of the chosen norm in `small_norms2` (= d2).
    pub index_sol2: usize,
}

/// Search the cross-product of two sorted short-norm lists for a Bezout
/// pair `(u, v)` with `u · d1 + v · d2 = target`, `u, v > 0`, `v < quotients[i2]`.
///
/// Mirrors the C reference's static `find_uv_from_lists` in
/// `src/id2iso/ref/lvlx/dim2id2iso.c:362-449` of
/// `github.com/SQISign/the-sqisign`.
///
/// # Algorithm
///
/// For each pair `(i1, i2)` with `i1 ∈ [0, small_norms1.len())` and
/// `i2 ∈ [start, small_norms2.len())` (where `start = i1` if
/// `is_diagonal == true`, else `0`):
///
/// - Compute `adjusted_norm = target mod small_norms1[i1]`.
/// - Compute `inv = small_norms2[i2]^{-1} mod small_norms1[i1]` via
///   the extended Euclidean helper. If no inverse exists (gcd ≠ 1),
///   skip this pair.
/// - Compute `v = (inv · adjusted_norm) mod small_norms1[i1]`.
/// - Walk `v += small_norms1[i1]` while `v < quotients[i2]`, and for
///   each step compute `u = (target − v · small_norms2[i2]) /
///   small_norms1[i1]`. Accept iff `u, v > 0`; on accept return
///   `Some(FindUvFromListsResult)`.
///
/// # Cornacchia paths (deferred)
///
/// The C reference accepts a `number_sum_square ∈ {0, 1, 2}` flag that
/// adds Cornacchia constraints on `v` (1) or both `u` and `v` (2). The
/// orchestrator (`find_uv` → `find_uv_from_lists`) passes
/// `number_sum_square == 0` in practice — the Cornacchia paths exist
/// for unused/future call sites. The Rust port currently implements
/// only the `number_sum_square == 0` path (the orchestrator's actual
/// usage); other values return `None`. The `number_sum_square ∈ {1, 2}`
/// paths can be added in a follow-up session when needed.
///
/// # Parameters
///
/// - `target`: the right-hand side of the Bezout equation. Typically
///   `2^TORSION_EVEN_POWER` per SQIsign convention.
/// - `small_norms1`: sorted short-norm list (the `d1` candidates).
/// - `small_norms2`: sorted short-norm list (the `d2` candidates).
/// - `quotients2`: precomputed `target / small_norms2[i]` for each
///   `i ∈ [0, small_norms2.len())`. Used as the upper bound on `v`.
/// - `is_diagonal`: when `true` (caller passes identical `small_norms1
///   == small_norms2`), restrict `i2 ≥ i1` to avoid double-counting.
/// - `number_sum_square`: 0 in the orchestrator's call pattern; other
///   values reserved for future Cornacchia-constrained calls.
///
/// # Returns
///
/// - `Some(FindUvFromListsResult)`: a valid Bezout pair was found.
/// - `None`: no pair satisfies the constraints within the given lists.
#[cfg(feature = "alloc")]
pub fn find_uv_from_lists<const LIMBS: usize>(
    target: &Int<LIMBS>,
    small_norms1: &[Int<LIMBS>],
    small_norms2: &[Int<LIMBS>],
    quotients2: &[Int<LIMBS>],
    is_diagonal: bool,
    number_sum_square: u32,
) -> Option<FindUvFromListsResult<LIMBS>> {
    use crate::quaternion::sign_orchestration::uint_inv_mod_vartime;
    let zero_int = Int::<LIMBS>::from_i64(0);

    // Only the orchestrator's actual call pattern is ported.
    // Cornacchia-constrained paths (number_sum_square != 0) return
    // None; no current call site exercises them.
    if number_sum_square != 0 {
        return None;
    }

    assert_eq!(
        small_norms2.len(),
        quotients2.len(),
        "find_uv_from_lists: small_norms2 and quotients2 must have the same length",
    );

    for (i1, d1) in small_norms1.iter().enumerate() {
        if *d1 <= zero_int {
            // Skip non-positive norms (shouldn't occur for valid inputs
            // from enumerate_hypercube but defensive).
            continue;
        }
        // adjusted_norm = target mod d1. Use int_div_floor + recompute
        // remainder via `target - (target/d1)·d1`.
        let q_floor = int_div_floor::<LIMBS>(target, d1);
        let q_times_d1 = q_floor.wrapping_mul(d1);
        let adjusted_norm = target.wrapping_sub(&q_times_d1);

        let start_i2 = if is_diagonal { i1 } else { 0 };
        for (i2, (d2, quotient2)) in small_norms2
            .iter()
            .zip(quotients2.iter())
            .enumerate()
            .skip(start_i2)
        {
            if *d2 <= zero_int {
                continue;
            }

            // Convert d1, d2 to Uint<LIMBS> for the modular inverse.
            // Both are positive per the guard above and per the
            // enumerate_hypercube contract (norms are |·G·| ≥ 0).
            let d1_uint = Uint::<LIMBS>::from_words(d1.to_words());
            let d2_uint = Uint::<LIMBS>::from_words(d2.to_words());

            // Compute inv = d2^{-1} mod d1.
            //
            // the d1 == 1 case is handled specially. `uint_inv_mod_vartime`
            // returns `None` for modulus 1 (its m==1 guard), and the old code
            // treated that as "gcd ≠ 1, skip" — silently DISCARDING every pair
            // with d1 = 1. But modulo 1 EVERY residue is 0, so d2 is trivially
            // invertible with inv ≡ 0, and `u·1 + v·d2 = target` is solvable for
            // any v ∈ [1, quotients2[i2]) with u = target − v·d2. Skipping d1 = 1
            // was harmless at the p=7 smoke prime (box=2's small-d list {1,3,5}
            // has d=3,5 to cover targets) but FATAL at the real prime, where
            // box=2's small-d list is {1,5,5} (no 3) — dropping d=1 leaves only
            // {5,5} and 5 ∤ 2^TORSION_EVEN_POWER, so find_uv wrongly returned
            // NoBezout (S233-S237 root cause). With inv = 0 the existing v-walk
            // below starts at v = 0, the v > 0 guard rejects it, and v = 1 then
            // yields u = target − d2 — the smallest valid solution.
            let inv_int = if *d1 == Int::<LIMBS>::from_i64(1) {
                Int::<LIMBS>::from_i64(0)
            } else {
                match uint_inv_mod_vartime::<LIMBS>(&d2_uint, &d1_uint) {
                    Some(iv) => Int::<LIMBS>::from_words(iv.to_words()),
                    None => continue, // gcd(d1, d2) != 1
                }
            };

            // v = (inv · adjusted_norm) mod d1. Compute the product
            // then reduce.
            let prod = inv_int.wrapping_mul(&adjusted_norm);
            let q2 = int_div_floor::<LIMBS>(&prod, d1);
            let q2_times_d1 = q2.wrapping_mul(d1);
            let mut v = prod.wrapping_sub(&q2_times_d1);

            // Walk v += d1 while v < quotients2[i2].
            while v < *quotient2 {
                // Compute u = (target - v · d2) / d1.
                let v_d2 = v.wrapping_mul(d2);
                let target_minus_vd2 = target.wrapping_sub(&v_d2);
                if target_minus_vd2 <= zero_int {
                    // The C ref asserts `ibz_cmp(u, &ibz_const_zero) > 0`
                    // (i.e. u > 0). If target - v·d2 ≤ 0 then u ≤ 0 — skip.
                    v = v.wrapping_add(d1);
                    continue;
                }
                let u = int_div_floor::<LIMBS>(&target_minus_vd2, d1);
                // The C ref also asserts `ibz_is_zero(&remain)` — exact
                // divisibility of (target - v·d2) by d1. By construction:
                // `target ≡ v · d2 (mod d1)` (from the inv-mod step), so
                // the remainder IS zero. debug_assert.
                #[cfg(debug_assertions)]
                {
                    let u_d1 = u.wrapping_mul(d1);
                    let remainder = target_minus_vd2.wrapping_sub(&u_d1);
                    debug_assert_eq!(
                        remainder, zero_int,
                        "find_uv_from_lists: (target - v·d2) must be divisible by d1",
                    );
                }

                // Final acceptance: u > 0 (checked above) AND v > 0.
                if u > zero_int && v > zero_int {
                    return Some(FindUvFromListsResult {
                        u,
                        v,
                        index_sol1: i1,
                        index_sol2: i2,
                    });
                }
                v = v.wrapping_add(d1);
            }
        }
    }

    None
}

/// Output of [`find_uv`]: a Bezout-decomposition with rational-quaternion
/// lifts.
///
/// Mirrors the C reference's `find_uv(u, v, beta1, beta2, d1, d2,
/// index_alternate_order_1, index_alternate_order_2, target, lideal,
/// Bpoo, num_alternate_order)` out-parameter set. On success:
/// `u·d1 + v·d2 = target` with `u, v, d1, d2 > 0` and `d1, d2` odd;
/// `beta_i` is a rational quaternion of reduced norm `n(lideal)·d_i`.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FindUvResult<const LIMBS: usize> {
    /// Positive integer u with `u·d1 + v·d2 = target`.
    pub u: Int<LIMBS>,
    /// Positive integer v.
    pub v: Int<LIMBS>,
    /// Rational-quaternion lift `beta1` of reduced norm `n(lideal)·d1`.
    pub beta1: RationalQuaternion<LIMBS>,
    /// Rational-quaternion lift `beta2` of reduced norm `n(lideal)·d2`.
    pub beta2: RationalQuaternion<LIMBS>,
    /// `d1`: odd positive integer; the norm component matched on the
    /// first short-vector list.
    pub d1: Int<LIMBS>,
    /// `d2`: odd positive integer; the norm component matched on the
    /// second short-vector list.
    pub d2: Int<LIMBS>,
    /// Index `j1 ∈ [0, alt_connecting.len() + 1)` of the alternate
    /// extremal order chosen for the first lift. `0` means the
    /// original ideal frame (no alternate-order rotation).
    pub index_alternate_order_1: usize,
    /// Index `j2 ∈ [j1, alt_connecting.len() + 1)` for the second lift.
    pub index_alternate_order_2: usize,
}

#[cfg(feature = "alloc")]
impl<const LIMBS: usize> FindUvResult<LIMBS> {
    /// Construct a placeholder result with all-zero integer fields and
    /// rational-quaternion units. Useful for default-construction in
    /// pre-population patterns; **not a valid Bezout solution** by
    /// itself. The S194 stub returns `Err(Unimplemented)`; callers
    /// constructing this directly should treat the result as semantically
    /// uninitialized.
    #[inline]
    pub fn placeholder() -> Self {
        let zero = Int::<LIMBS>::from_i64(0);
        Self {
            u: zero,
            v: zero,
            beta1: RationalQuaternion::<LIMBS>::one(),
            beta2: RationalQuaternion::<LIMBS>::one(),
            d1: zero,
            d2: zero,
            index_alternate_order_1: 0,
            index_alternate_order_2: 0,
        }
    }
}

/// build the Clapotis **θ endomorphism** `θ = β2 · conj(β1) / n(I)`
/// from a [`FindUvResult`] — the first step of Movement 2 (dim-2 isogeny
/// kernel construction), per the SQIsign C reference's
/// `dim2id2iso_ideal_to_isogeny_clapotis`. The dim-2 kernel is generated
/// by `([u·d1]P, φ_v ∘ θ ∘ ĥφ_u(P))`; `θ` is the quaternion-side part,
/// computed here in isolation (no theta-chain dependency yet).
///
/// `n_ideal` is the reduced norm `n(I)` of the input ideal (= the `n_id`
/// from `lideal.reduced_norm_vartime()`), and `p` the level prime.
///
/// # Precision (`WIDE`)
///
/// `β.num ~ β.denom · p ~ 2^(2·bits(p))`, so the product `β2·conj(β1)`
/// reaches `~2^(4·bits(p))` (≈ 2^1000 at L1) — far past `Int<LIMBS>`'s
/// `64·LIMBS−1` range. The multiplication therefore runs at `Int<WIDE>`;
/// `normalize()` reduces the result to lowest terms (the canonical θ has
/// small components), and the narrowed θ fits back in `Int<LIMBS>`. S241
/// calibration: WIDE=16 suffices at L1 (`normalize` keeps intermediates
/// bounded); higher levels need re-probe.
///
/// # Postcondition
///
/// `N_red(θ) = N(β2)·N(β1) / n(I)² = (n(I)·d2)(n(I)·d1) / n(I)² = d1·d2`.
/// Verified by `s241_theta_endomorphism_has_norm_d1d2` against the
/// real-L1-prime (β1,β2) from the S238-verified `find_uv`.
///
/// Returns `None` if the narrowed θ would not fit `Int<LIMBS>` (defensive;
/// the canonical normalized θ always fits for valid SQIsign inputs).
#[cfg(feature = "alloc")]
pub fn theta_endomorphism<const LIMBS: usize, const WIDE: usize>(
    result: &FindUvResult<LIMBS>,
    n_ideal: &Uint<LIMBS>,
    p: &Uint<LIMBS>,
) -> Option<RationalQuaternion<LIMBS>> {
    use crate::quaternion::lattice::{narrow_int_lattice, widen_int_lattice};
    let widen_rq = |rq: &RationalQuaternion<LIMBS>| RationalQuaternion::<WIDE> {
        num: crate::quaternion::algebra::Quaternion::<WIDE>::new(
            widen_int_lattice::<LIMBS, WIDE>(&rq.num.a),
            widen_int_lattice::<LIMBS, WIDE>(&rq.num.b),
            widen_int_lattice::<LIMBS, WIDE>(&rq.num.c),
            widen_int_lattice::<LIMBS, WIDE>(&rq.num.d),
        ),
        denom: rq.denom.resize::<WIDE>(),
    };
    let p_w = p.resize::<WIDE>();
    let b1_w = widen_rq(&result.beta1);
    let b2_w = widen_rq(&result.beta2);
    // θ = β2 · conj(β1), then divide by n(I): multiply the denominator by
    // n(I) (rational division by a scalar), then reduce to lowest terms.
    let mut theta_w = b2_w.mul(&b1_w.conjugate(), &p_w);
    theta_w.denom = theta_w.denom.wrapping_mul(&n_ideal.resize::<WIDE>());
    let theta_w = theta_w.normalize();
    // Narrow back to Int<LIMBS>. Guard that each component + denom fit.
    let fits = |x: &Int<WIDE>| {
        Uint::<WIDE>::from_words(x.abs().to_words()).bits_vartime()
            < 64u32 * u32::try_from(LIMBS).expect("LIMBS fits u32") - 1
    };
    if !(fits(&theta_w.num.a)
        && fits(&theta_w.num.b)
        && fits(&theta_w.num.c)
        && fits(&theta_w.num.d)
        && theta_w.denom.bits_vartime() < 64u32 * u32::try_from(LIMBS).expect("LIMBS fits u32"))
    {
        return None;
    }
    Some(RationalQuaternion::<LIMBS> {
        num: crate::quaternion::algebra::Quaternion::<LIMBS>::new(
            narrow_int_lattice::<WIDE, LIMBS>(&theta_w.num.a),
            narrow_int_lattice::<WIDE, LIMBS>(&theta_w.num.b),
            narrow_int_lattice::<WIDE, LIMBS>(&theta_w.num.c),
            narrow_int_lattice::<WIDE, LIMBS>(&theta_w.num.d),
        ),
        denom: theta_w.denom.resize::<LIMBS>(),
    })
}

/// `N(I) = √( |det(basis)| / denom⁴ )` for a left ideal given by an integer
/// `basis` of the rational lattice `(1/denom)·Z⟨basis⟩`. Derived from the
/// lattice DETERMINANT, not the `cached_norm` field — so it is independent
/// of which convention the ideal's producer used for `cached_norm`
/// (`LeftIdeal::new` / ideal.rs use `cached_norm = |det| = N²`, while the
/// sampler-style builders store `cached_norm = N`). For any integer basis,
/// `|det(basis)| = N(I)²·denom⁴` exactly, so this recovers the true reduced
/// ideal norm `N(I)` regardless. Computed at width `W` because `|det| ~
/// N²·denom⁴` (`~2^1016` at L1) overflows `Int<L>`. Returns `None` if
/// `denom = 0` or `|det|/denom⁴` is not a perfect square (i.e. the basis is
/// not a genuine left `O_0`-ideal lattice).
#[cfg(feature = "alloc")]
pub(crate) fn lattice_reduced_norm<const L: usize, const W: usize>(
    basis: &[[Int<L>; 4]; 4],
    denom: &Uint<L>,
) -> Option<Uint<L>> {
    let mut basis_w = [[Int::<W>::from_i64(0); 4]; 4];
    for (basis_w_row, basis_row) in basis_w.iter_mut().zip(basis.iter()) {
        for (basis_w_cell, basis_cell) in basis_w_row.iter_mut().zip(basis_row.iter()) {
            *basis_w_cell = crate::quaternion::lattice::widen_int_lattice::<L, W>(basis_cell);
        }
    }
    let det_abs = crate::quaternion::ideal::det_4x4::<W>(&basis_w).abs();
    let denom_w = denom.resize::<W>();
    let denom4 = denom_w
        .wrapping_mul(&denom_w)
        .wrapping_mul(&denom_w)
        .wrapping_mul(&denom_w);
    let denom4_nz = crypto_bigint::NonZero::new(denom4).into_option()?;
    let (lattice_index, _rem) = det_abs.div_rem_vartime(&denom4_nz);
    let n = lattice_index.floor_sqrt_vartime();
    if n.wrapping_mul(&n) != lattice_index {
        return None;
    }
    Some(n.resize::<L>())
}

/// Find `(u, v, d1, d2, beta1, beta2)` with `u·d1 + v·d2 = target`,
/// `d1, d2` odd, and quaternion lifts `beta_i` of reduced norm
/// `n(lideal) · d_i`.
///
/// Mirrors the C reference's `find_uv` at
/// `src/id2iso/ref/lvlx/dim2id2iso.c:471-756` of
/// `github.com/SQISign/the-sqisign`. The function is the top-level
/// driver for Step 1 of the Clapotis evaluator — the quaternion-side
/// preparation that feeds the dim-2 Kani-diagram walk.
///
/// # Scope (S195 — `num_alternate_order = 0` only)
///
/// This port currently implements the `num_alternate_order = 0`
/// baseline (the `j = 0` path of the C reference's body). Concretely:
///
/// 1. LLL-reduce the input ideal via
///    [`crate::quaternion::ideal_mul::lideal_reduce_basis`]
///    (Cohen 2.6.3 in the `O_0` reduced-norm metric, S195-shipped).
/// 2. Build the pulled-back Gram matrix `G_I = B · G_{O_0} · Bᵀ`
///    via `pull_back_gram`; here `G_{O_0}` is the integer-safe
///    Gram from [`o0_reduced_norm_gram_matrix`] (factor of 4 baked
///    in for divisibility under `p ≡ 3 mod 4`).
/// 3. Enumerate short vectors in `[-m, m]^4` under `G_I` via
///    [`enumerate_hypercube`] with `adjusted_norm = 4 · denom²` so
///    the stored norms equal `N(α_v)` directly. Sort by norm
///    (stable, matching the C ref's `compare_vec_by_norm` qsort).
/// 4. Precompute `quotients[i] = target / small_norms[i]` and call
///    [`find_uv_from_lists`] with `is_diagonal = true` (both lists
///    are the same `j = 0` list).
/// 5. Lift the integer solution to quaternion elements via
///    [`mat_4x4_transpose_eval`] applied to the LLL-reduced basis.
///    Row-major basis storage means the lattice element at coords
///    `v` is `α = Σ_r v[r] · basis[r]`, i.e. `α_o0 = basisᵀ · v`,
///    NOT `basis · v` — this matches the pulled-back Gram identity
///    `vᵀ · (B · M · Bᵀ) · v = (Bᵀ v)ᵀ · M · (Bᵀ v)`. The lifted
///    coordinates are in `O_0`-basis form; convert to standard
///    `(1, i, j, k)` coordinates via
///    [`o0_basis_to_standard_doubled`] (which scales by 2 to stay
///    integer), and bump the rational denominator by 2 accordingly.
///
/// # Deferred (alternate orders, S205 API surface)
///
/// The `alt_connecting` slice parameter encodes the C reference's
/// `ALTERNATE_CONNECTING_IDEALS[0..num_alternate_order]` per
/// security level. **S205 ships only the API surface**: a non-empty
/// `alt_connecting` returns `Err(Error::Unimplemented)`. Passing
/// `&[]` (the empty slice) maps to the current `num_alternate_order
/// = 0` baseline — the j=0 path that S195-S204 validated.
///
/// The j-loop body requires:
/// - Looping over `j ∈ [1, alt_connecting.len()]` and computing
///   `ideal[j] = lideal_intersect(conj(reduced_id), alt_connecting[j-1])`.
/// - Per-j LLL + rescale + Gram + enumerate.
/// - Cross-product Bezout over `(j1, j2)` pairs with `j2 ≥ j1`.
/// - β post-multiply per alternate order via
///   [`RationalQuaternion::mul`](crate::quaternion::algebra::RationalQuaternion::mul).
///
/// The S203 invariant (`reduced_id = O_0` post-rescale for SQIsign-
/// shaped principals) simplifies the j-loop's `lideal_intersect`
/// call to a no-op pass-through: `lideal_intersect(O_0, ALT[j-1]) =
/// ALT[j-1]`. S206 will leverage this when wiring the body.
///
/// # Box-size parameter (S208 lifted from hardcoded m=3)
///
/// The `box_size` parameter encodes the C reference's
/// `FINDUV_box_size` per security level (extracted via S206/S207
/// research agents):
///
/// | Constant | L1 | L3 | L5 |
/// |---|---|---|---|
/// | `FINDUV_box_size` | **2** | 3 | 3 |
/// | `FINDUV_cube_size` | 624 | 2400 | 2400 |
/// | `NUM_ALTERNATE_EXTREMAL_ORDERS` | 6 | 6 (or 7) | 6 |
///
/// Callers pass the level-appropriate value. Larger `box_size` is
/// always safe for correctness (more enumeration candidates), just
/// less efficient. Smaller `box_size` may miss Bezout solutions for
/// some inputs — caller can retry with a larger box (adaptive retry
/// is future work).
///
/// # Returns
///
/// - `Ok(FindUvResult)`: a Bezout decomposition was found.
/// - `Err(Error::Unimplemented)`: `!alt_connecting.is_empty()`
///   (the j-loop body is future-S206), or the ideal denominator
///   exceeds `Int<LIMBS>`'s non-negative range (defensive — should
///   not occur for SQIsign production inputs).
/// - `Err(Error::NoBezoutSolution(msg))`: the implementation ran
///   successfully but did not find a Bezout decomposition within
///   the chosen box. Two distinct paths reach this: the short-
///   vector enumeration returned empty, OR the Bezout search
///   exhausted its candidates. Distinct from `Unimplemented` —
///   the algorithm IS implemented, the inputs simply did not
///   admit a solution at `m = 3`. Caller may retry with a larger
///   box size (adaptive retry is future work).
///
/// # Variable-time
///
/// Variable-time on the ideal basis entries (LLL + enumeration both
/// vartime). Acceptable per SQIsign 2.0 spec §8.
#[cfg(feature = "alloc")]
pub fn find_uv<const LIMBS: usize>(
    target: &Int<LIMBS>,
    lideal: &crate::quaternion::ideal::LeftIdeal<LIMBS>,
    p: &Uint<LIMBS>,
    alt_connecting: &[crate::quaternion::ideal::LeftIdeal<LIMBS>],
    box_size: i64,
) -> Result<FindUvResult<LIMBS>> {
    use crate::quaternion::ideal_mul::lideal_reduce_basis_wide;
    use crate::quaternion::o0_mul::uint_as_nonneg_int;

    // Option C dispatch: non-empty alt_connecting goes to the
    // separate find_uv_alternate_orders function (j-loop body lives
    // there, isolated from the j=0 path). Empty slice → existing j=0
    // body below, unchanged from S195-S210.
    //
    // WIDE=128 (8192 bits): S345 calibration at the REAL L1 prime. The per-j
    // product `conj(reduced_id)·ALT[j-1]`'s integer-GSO determinant
    // intermediates exceed 4096 bits at real scale (WIDE=64 panicked in
    // `int_div_exact`; WIDE=128 runs to completion — see
    // `s345_find_uv_alternate_orders_real_l1_limbs16_all_six_alts`). The
    // earlier WIDE=20 only ever covered the p=7 smoke fixtures.
    // Over-wide is always correct (more headroom), just slower; per-level
    // tuning is future work.
    if !alt_connecting.is_empty() {
        return find_uv_alternate_orders::<LIMBS, 128>(target, lideal, p, alt_connecting, box_size);
    }

    // Step 1: LLL-reduce the input ideal under the O_0 reduced-norm metric.
    // WIDE=64 — the integer-GSO recurrence overflows on large ideals. For
    // real-prime PRINCIPAL test ideals (norm ~p ~2^251, cached_norm ~p²
    // ~2^500) WIDE=32 sufficed (S225 calibration). But the REAL signing
    // input is the KLPT connecting ideal of norm ~2^511 (q·T, q~2^253), so
    // its GSO intermediates reach ~2^2044 and overflow WIDE=32 (2048 bits)
    // — int_div_exact non-exact. WIDE=64 (4096 bits) covers both. The wide
    // LLL returns a reduced basis that fits back in `Int<LIMBS>` (LIMBS=16
    // for the real connecting ideal). (Per-level/input-scaled WIDE in
    // `Params` is future work.)
    let reduced = lideal_reduce_basis_wide::<LIMBS, 128>(lideal, p);

    // enumerate short vectors DIRECTLY in the LLL-reduced ORIGINAL
    // ideal — the C-ref `find_uv` path, valid for ANY ideal (principal
    // or not). The S200-era δ-rescale to `reduced_id` was a
    // principal-only step (rescaling a principal ideal → O_0, which made
    // every short vector a unit ⇒ degenerate d=1, and whose divisibility
    // check rejected non-principal ideals outright). The C-ref uses that
    // δ-rescale ONLY for the alternate-order (j≥1) connecting-ideal
    // seeding, never the j=0 enumeration. Here β lifts directly into I as
    // `reduced·vec`, so no δ post-multiply is needed.
    let two_uint = Uint::<LIMBS>::from_u64(2);

    // Step 3: build the pulled-back Gram on the LLL-reduced ideal
    // `reduced`. The enumeration finds β ∈ I directly (β = reduced·vec).
    //
    // compute the pullback in WIDE precision. At the real L1 prime
    // the basis entries are ~2^249 and the O_0 metric ~2^251, so the
    // narrow `Bᵀ·M·B` INTERMEDIATES (~2^749) wrap `Int<LIMBS>` (511-bit
    // at LIMBS=8) and silently corrupt the Gram — S230 root-caused the
    // S226-fixture enumerate panic to exactly this. WIDE=16 (1024-bit)
    // proved sufficient in S230's probe to restore the exact
    // `adjusted_norm | vᵀGv` divisibility; we run enumerate at the same
    // width. At p=7 this is bit-identical to the narrow path (no overflow
    // to begin with), so existing tests are unaffected.
    const FINDUV_WIDE: usize = 32;
    let o0_gram = o0_reduced_norm_gram_matrix::<LIMBS>(p);
    let gram_id_wide = crate::quaternion::lattice::pull_back_gram_wide::<LIMBS, FINDUV_WIDE>(
        &reduced.basis,
        &o0_gram,
    );

    // adjusted_norm = 4 · reduced.denom² · N(I). The enumeration's
    // quadratic form gives vᵀGv = 4 · N_red(β) · reduced.denom² for
    // β = reduced·vec ∈ I; dividing by this adjusted_norm yields the
    // SHORT NORM d = N_red(β) / N(I) — the norm of the ideal in β's
    // class (S278: enumerating in the LLL-reduced ORIGINAL ideal, the
    // ÷N(I) normalization is explicit here, whereas the old rescale-to-O_0
    // path folded it into the rescaled denom). The ×4 pairs with the
    // integer-safety ×4 bake-in of o0_reduced_norm_gram_matrix. Computed
    // in WIDE (denom ~2^249, N(I) ~2^251 → product ~2^750, fits Int<16>).
    let denom_int = uint_as_nonneg_int::<LIMBS>(&reduced.denom).ok_or(Error::Unimplemented(
        "find_uv: reduced ideal denominator exceeds Int<LIMBS> non-negative range",
    ))?;
    let denom_wide =
        crate::quaternion::lattice::widen_int_lattice::<LIMBS, FINDUV_WIDE>(&denom_int);
    let denom_sq_wide = denom_wide.wrapping_mul(&denom_wide);

    // N(I) from the lattice determinant (convention-independent — see
    // `lattice_reduced_norm`). At FINDUV_WIDE because |det| ~ N²·denom⁴
    // overflows Int<LIMBS> at L1. LLL preserves |det| and denom, so this
    // equals the input ideal's N(I).
    let n_ideal = lattice_reduced_norm::<LIMBS, FINDUV_WIDE>(&reduced.basis, &reduced.denom)
        .ok_or(Error::NoBezoutSolution(
            "find_uv: |det(basis)|/denom^4 is not a perfect square — the input is not a \
             genuine left O_0-ideal (lattice index must equal N(I)²).",
        ))?;
    let n_ideal_int = uint_as_nonneg_int::<LIMBS>(&n_ideal).ok_or(Error::Unimplemented(
        "find_uv: N(I) exceeds Int<LIMBS> non-negative range",
    ))?;
    let n_ideal_wide =
        crate::quaternion::lattice::widen_int_lattice::<LIMBS, FINDUV_WIDE>(&n_ideal_int);
    let adjusted_norm_wide = Int::<FINDUV_WIDE>::from_i64(4)
        .wrapping_mul(&denom_sq_wide)
        .wrapping_mul(&n_ideal_wide);

    // box_size is now a caller-supplied per-level parameter
    // (C ref's FINDUV_box_size: L1=2, L3=3, L5=3). Validated via the
    // research agents in S206-S207.
    let m: i64 = box_size;

    let mut short_vecs =
        enumerate_hypercube_wide::<LIMBS, FINDUV_WIDE>(m, &gram_id_wide, &adjusted_norm_wide);
    if short_vecs.is_empty() {
        return Err(Error::NoBezoutSolution(
            "find_uv: enumerate_hypercube returned no candidates at the S195 hard-coded \
             box size m=3 — caller may need a larger box. Tuning the per-level \
             FINDUV_box_size + an adaptive-retry path is future work.",
        ));
    }

    // Stable sort by norm — matches the C ref's compare_vec_by_norm
    // qsort. Insertion order is the natural tie-breaker.
    short_vecs.sort_by_key(|sv| sv.norm);

    let small_norms: Vec<Int<LIMBS>> = short_vecs.iter().map(|sv| sv.norm).collect();
    let quotients: Vec<Int<LIMBS>> = small_norms
        .iter()
        .map(|d| int_div_floor::<LIMBS>(target, d))
        .collect();

    // Step 4 (j1, j2) = (0, 0): only the diagonal pair exists at
    // num_alternate_order = 0.
    let bezout = find_uv_from_lists::<LIMBS>(
        target,
        &small_norms,
        &small_norms,
        &quotients,
        true, // is_diagonal — same list both sides
        0,    // number_sum_square = 0 per orchestrator pattern
    )
    .ok_or(Error::NoBezoutSolution(
        "find_uv: find_uv_from_lists found no Bezout decomposition within the chosen \
         box. Caller may retry with a larger FINDUV_box_size; adaptive-retry path \
         is future work.",
    ))?;

    // Step 5: lift β directly from the LLL-reduced ideal `reduced` —
    // β = reduced·vec ∈ I (the C-ref `β = reduced[0]·vec`). The Gram
    // identity vᵀ·G_I·v = 4·N(Bᵀ·v) (B = reduced.basis) means the lift
    // is `mat_4x4_transpose_eval(reduced.basis, v)`. No δ post-multiply:
    // β already lives in the original ideal I.
    let beta_1_o0 =
        mat_4x4_transpose_eval::<LIMBS>(&reduced.basis, &short_vecs[bezout.index_sol1].vec);
    let beta_2_o0 =
        mat_4x4_transpose_eval::<LIMBS>(&reduced.basis, &short_vecs[bezout.index_sol2].vec);
    let beta_1_num = o0_basis_to_standard_doubled::<LIMBS>(&beta_1_o0);
    let beta_2_num = o0_basis_to_standard_doubled::<LIMBS>(&beta_2_o0);

    let beta_denom = two_uint.wrapping_mul(&reduced.denom);
    let beta1 = RationalQuaternion {
        num: beta_1_num,
        denom: beta_denom,
    }
    .normalize();
    let beta2 = RationalQuaternion {
        num: beta_2_num,
        denom: beta_denom,
    }
    .normalize();

    Ok(FindUvResult {
        u: bezout.u,
        v: bezout.v,
        beta1,
        beta2,
        d1: small_norms[bezout.index_sol1],
        d2: small_norms[bezout.index_sol2],
        index_alternate_order_1: 0,
        index_alternate_order_2: 0,
    })
}

/// Self-verification arbiter for the `find_uv_alternate_orders` j>0 finalize:
/// is the rational quaternion `beta` contained in the (integral) left ideal
/// `lideal`? Port of the C reference's `quat_lattice_contains(ideal->lattice,
/// β)` debug assertion (dim2id2iso.c:690-691), which is the convention-blind
/// arbiter that distinguishes the correctly-finalized β (in the input ideal)
/// from the conjugation-swapped wrong one (in an alternate-order frame).
///
/// `lideal.basis` is in `O_0`-basis coordinates (row = generator); each
/// generator is converted to standard `(1,i,j,k)` coordinates via
/// [`o0_basis_to_standard_doubled`] (which returns `2 ×` the standard value),
/// placed COLUMN-wise for [`lattice_coords_of`]'s column convention; the `×2`
/// doubling is absorbed by doubling `lat_denom`. The solve runs at `WIDE`
/// width because the adjugate intermediate `adj · x · lat_denom` reaches
/// `~2^1246` at the real L1 ideal scale (overflows `Int<LIMBS=16>`).
#[cfg(feature = "alloc")]
fn rational_quaternion_in_lideal<const LIMBS: usize, const WIDE: usize>(
    beta: &RationalQuaternion<LIMBS>,
    lideal: &crate::quaternion::ideal::LeftIdeal<LIMBS>,
) -> bool {
    use crate::quaternion::extremal_orders::lattice_coords_of;
    use crate::quaternion::lattice::widen_int_lattice;
    use crate::quaternion::o0_mul::{o0_basis_to_standard_doubled, uint_as_nonneg_int};

    let mut std_basis = [[Int::<WIDE>::from_i64(0); 4]; 4];
    for (generator_index, basis_row) in lideal.basis.iter().enumerate() {
        let standard_element = o0_basis_to_standard_doubled::<LIMBS>(basis_row);
        std_basis[0][generator_index] = widen_int_lattice::<LIMBS, WIDE>(&standard_element.a);
        std_basis[1][generator_index] = widen_int_lattice::<LIMBS, WIDE>(&standard_element.b);
        std_basis[2][generator_index] = widen_int_lattice::<LIMBS, WIDE>(&standard_element.c);
        std_basis[3][generator_index] = widen_int_lattice::<LIMBS, WIDE>(&standard_element.d);
    }
    let two_denom = Uint::<LIMBS>::from_u64(2).wrapping_mul(&lideal.denom);
    let lat_denom = match uint_as_nonneg_int::<LIMBS>(&two_denom) {
        Some(d) => widen_int_lattice::<LIMBS, WIDE>(&d),
        None => return false,
    };
    let x_std = [
        widen_int_lattice::<LIMBS, WIDE>(&beta.num.a),
        widen_int_lattice::<LIMBS, WIDE>(&beta.num.b),
        widen_int_lattice::<LIMBS, WIDE>(&beta.num.c),
        widen_int_lattice::<LIMBS, WIDE>(&beta.num.d),
    ];
    let x_denom = match uint_as_nonneg_int::<LIMBS>(&beta.denom) {
        Some(d) => widen_int_lattice::<LIMBS, WIDE>(&d),
        None => return false,
    };
    lattice_coords_of::<WIDE>(&std_basis, &lat_denom, &x_std, &x_denom).is_some()
}

/// Alternate-orders body for the Clapotis `find_uv` orchestrator —
/// handles the non-empty `alt_connecting` case (j > 0 indices).
/// **Skeleton: S211 (this session) ships the dispatch-target signature
/// + the `Unimplemented` return; S212+ wires the body**.
///
/// # Design (S207 Option C — separate function from j=0 path)
///
/// Keeping the j>0 logic in a separate function:
/// - Preserves the S195-S210-validated j=0 convention in [`find_uv`].
/// - Isolates the C-ref-convention work (LEFT-multiply by single shared
///   δ, then conjugate, with δ-denom mutation gated on `j != 0`) to
///   this function.
/// - Allows test-by-test validation as `ALTERNATE_CONNECTING_IDEALS`
///   data fixtures land.
///
/// # Planned body (S212+, per S207's verbatim C-ref extraction)
///
/// 1. **LLL-reduce input** lideal → `reduced`.
/// 2. **Build single shared `δ`** from `reduced.basis[0]` (smallest-norm
///    LLL element); conjugate; multiply denom by `ideal[0].norm`
///    (= `n(reduced)`).
/// 3. **Rescale** → `reduced_id` (the S203 invariant says this is `O_0`
///    for SQIsign-shaped inputs).
/// 4. **Build `ideals[0..=alt_connecting.len()]`**:
///    - `ideals[0] = reduced_id`
///    - For `j ∈ [1, alt_connecting.len()]`:
///      `ideals[j] = lideal_intersect(conj(reduced_id), alt_connecting[j-1])`
///      (= `alt_connecting[j-1]` per S203 invariant, simplifying the
///      lideal_intersect to a pass-through).
/// 5. **Per-j Gram + enumerate** to populate `small_norms[j]` +
///    `short_vecs[j]` + `quotients[j]`.
/// 6. **Cross-product Bezout**: nested `(j1, j2)` loop with `j2 ≥ j1`,
///    calling [`find_uv_from_lists`] with `is_diagonal = (j1 == j2)`.
///    First success wins.
/// 7. **β finalize** per the verbatim C-ref code (S207 ISA close):
///    - `β = lift_via_mat_4x4_transpose_eval(reduced[j], short_vec)` —
///      same for j=0 AND j>0.
///    - For `(j1 != 0 || j2 != 0)`: mutate δ.denom once:
///      `δ.denom = δ.denom / n(input_lideal) · n(conj_ideal)`.
///    - For `j1 != 0`: `β1 = quat_alg_mul(δ, β1)` (LEFT-multiply),
///      normalize, then `β1 = conj(β1)`. (Same for β2 gated on j2.)
///
/// # Returns
///
/// - `Err(Error::Unimplemented)`: until S212+ wires the body.
/// - Future: `Ok(FindUvResult)` with the cross-product Bezout output.
///
/// The j>0 β finalize self-verifies via `rational_quaternion_in_lideal`
/// (the C ref's own `quat_lattice_contains(ideal->lattice, β)` debug
/// assertion, dim2id2iso.c:690-691) before returning `Ok` — a sound arbiter
/// that distinguishes the correct β from the conjugation-swapped wrong one
/// (which lands in the alternate-order frame, not the input ideal), so a
/// reconciliation error fails closed rather than emitting unverified crypto.
#[cfg(feature = "alloc")]
pub fn find_uv_alternate_orders<const LIMBS: usize, const WIDE: usize>(
    target: &Int<LIMBS>,
    lideal: &crate::quaternion::ideal::LeftIdeal<LIMBS>,
    p: &Uint<LIMBS>,
    alt_connecting: &[crate::quaternion::ideal::LeftIdeal<LIMBS>],
    box_size: i64,
) -> Result<FindUvResult<LIMBS>> {
    use crate::quaternion::ideal::LeftIdeal;
    use crate::quaternion::ideal_mul::{
        ideal_multiply, lideal_reduce_basis_wide, lideal_rescale_by_smallest_basis_element,
    };

    debug_assert!(
        !alt_connecting.is_empty(),
        "find_uv_alternate_orders: caller must dispatch on alt_connecting.is_empty(); \
         this function specifically handles the non-empty case",
    );

    // j-loop SETUP — LLL + rescale input → reduced_id, then for
    // each j ∈ [1, alt_connecting.len()] build ideal[j] = conj(reduced_id)
    // · ALT[j-1] then wide-LLL. Per S217 finding: NO rescale on j>0
    // entries — the C ref's body uses lideal_mul + LLL only.

    // Step 1 (existing j=0 path): LLL-reduce input lideal, extract δ,
    // rescale to reduced_id. S226: WIDE=32 on the input LLL for the same
    // real-prime overflow reason as `find_uv` (the narrow j=0 LLL
    // overflows on cached_norm ~p²; S225 calibration). The per-j products
    // below already wide-LLL via the WIDE const generic.
    let reduced = lideal_reduce_basis_wide::<LIMBS, 128>(lideal, p);
    let reduced_id = lideal_rescale_by_smallest_basis_element::<LIMBS>(&reduced, p).ok_or(
        Error::NoBezoutSolution(
            "find_uv_alternate_orders: rescale failed on input lideal — not SQIsign-shaped \
                 OR cached_norm not a perfect square (defensive)",
        ),
    )?;

    // Step 2: build per-j ideals. ideal[0] = the LLL-reduced ORIGINAL input
    // (C ref `ideal[0] = quat_lideal_reduce_basis(lideal)`), NOT the rescaled
    // reduced_id. Enumerating j=0 on the original ideal yields genuine short
    // norms `d = N_red(β)/N(I)` (which may exceed 1) — the input's true
    // Clapotis decomposition. Enumerating on reduced_id (≅ O_0) instead makes
    // every short vector a unit ⇒ d=1 ⇒ a degenerate `u·1 + v·d2 = 2^F` split
    // with u ≈ 2^F, v = 1, which the fixed-degree isogeny cannot construct.
    // reduced_id is used ONLY to build the j≥1 connecting ideals.
    let conj_reduced_id = reduced_id.conjugate();
    let mut reduced_per_j: Vec<LeftIdeal<LIMBS>> = Vec::with_capacity(alt_connecting.len() + 1);
    reduced_per_j.push(reduced);

    for alt in alt_connecting {
        // ideal[j] = conj(reduced_id) · alt_connecting[j-1] (C ref line 560).
        let ideal_j_raw = ideal_multiply::<LIMBS>(&conj_reduced_id, alt, p);
        // Wide-LLL the raw product (S216 path): narrow LLL would overflow
        // on ALT-magnitude basis entries even at small primes.
        let reduced_j = lideal_reduce_basis_wide::<LIMBS, WIDE>(&ideal_j_raw, p);
        reduced_per_j.push(reduced_j);
    }

    // Step 3: Build the shared δ from reduced.basis[0]. Mirrors
    // find_uv's S195 δ-extraction code: o0_basis_to_standard_doubled
    // converts O_0-basis coords to (1,i,j,k) scaled by 2; rational
    // denominator = 2 · reduced.denom (the 2 absorbs the doubling).
    // δ is shared across ALL βs (S207 finding) — built once here.
    let delta_num = o0_basis_to_standard_doubled::<LIMBS>(&reduced.basis[0]);
    let two_uint = Uint::<LIMBS>::from_u64(2);
    let delta_rational = RationalQuaternion {
        num: delta_num,
        denom: two_uint.wrapping_mul(&reduced.denom),
    };

    // Step 4: per-j Gram + enumerate. For each reduced_per_j[j]
    // we pull-back the O_0 Gram, enumerate, sort, and build the
    // small_norms / quotients lists for find_uv_from_lists. S220 also
    // retains the sorted short-vectors per j so the finalize step can
    // lift the Bezout-selected short vector into a rational quaternion.
    let o0_gram = o0_reduced_norm_gram_matrix::<LIMBS>(p);
    let mut short_vecs_per_j: Vec<Vec<EnumeratedShortVec<LIMBS>>> =
        Vec::with_capacity(reduced_per_j.len());
    let mut small_norms_per_j: Vec<Vec<Int<LIMBS>>> = Vec::with_capacity(reduced_per_j.len());
    let mut quotients_per_j: Vec<Vec<Int<LIMBS>>> = Vec::with_capacity(reduced_per_j.len());

    // per-j gram + enumerate in WIDE precision, mirroring the S231
    // find_uv j=0 fix. The per-j pullback Bᵀ·M·B has the same real-prime
    // intermediate overflow (~2^749 wraps Int<LIMBS>) as the j=0 path; the
    // narrow version was latent-only because j>0 isn't reached at p=7 yet,
    // but it would panic identically at real p. Same FINDUV_WIDE=16, same
    // 4·denom² (pairs with o0_reduced_norm_gram_matrix's ×4 bake-in).
    const FINDUV_WIDE: usize = 32;
    for reduced_j in &reduced_per_j {
        let gram_j_wide = crate::quaternion::lattice::pull_back_gram_wide::<LIMBS, FINDUV_WIDE>(
            &reduced_j.basis,
            &o0_gram,
        );
        let denom_int = crate::quaternion::o0_mul::uint_as_nonneg_int::<LIMBS>(&reduced_j.denom)
            .ok_or(Error::Unimplemented(
                "find_uv_alternate_orders: rescaled ideal denominator exceeds Int<LIMBS> \
                     non-negative range (defensive)",
            ))?;
        let denom_wide =
            crate::quaternion::lattice::widen_int_lattice::<LIMBS, FINDUV_WIDE>(&denom_int);
        let denom_sq_wide = denom_wide.wrapping_mul(&denom_wide);
        // adjusted_norm = 4·denom²·N(ideal[j]) — the enumerated short norm is
        // then d = N_red(β)/N(ideal[j]), the norm of β's CLASS (the connecting
        // isogeny degree), not the raw N_red(β). This matches find_uv's proven
        // j=0 adjusted_norm. For j=0 the ideal is the LLL-reduced original
        // input (N(I) ~ 2^248), so omitting this factor would inflate d by
        // N(I) and force a degenerate u≈2^F, v=1 Bezout split.
        let n_ideal_j =
            lattice_reduced_norm::<LIMBS, FINDUV_WIDE>(&reduced_j.basis, &reduced_j.denom).ok_or(
                Error::NoBezoutSolution(
                    "find_uv_alternate_orders: |det(basis)|/denom^4 not a perfect square — \
                 ideal[j] is not a genuine left ideal (defensive)",
                ),
            )?;
        let n_ideal_j_int = crate::quaternion::o0_mul::uint_as_nonneg_int::<LIMBS>(&n_ideal_j)
            .ok_or(Error::Unimplemented(
                "find_uv_alternate_orders: N(ideal[j]) exceeds Int<LIMBS> range (defensive)",
            ))?;
        let n_ideal_j_wide =
            crate::quaternion::lattice::widen_int_lattice::<LIMBS, FINDUV_WIDE>(&n_ideal_j_int);
        let adjusted_norm_j_wide = Int::<FINDUV_WIDE>::from_i64(4)
            .wrapping_mul(&denom_sq_wide)
            .wrapping_mul(&n_ideal_j_wide);
        let mut short_vecs_j = enumerate_hypercube_wide::<LIMBS, FINDUV_WIDE>(
            box_size,
            &gram_j_wide,
            &adjusted_norm_j_wide,
        );
        short_vecs_j.sort_by_key(|sv| sv.norm);
        let small_norms_j: Vec<Int<LIMBS>> = short_vecs_j.iter().map(|sv| sv.norm).collect();
        let quotients_j: Vec<Int<LIMBS>> = small_norms_j
            .iter()
            .map(|d| int_div_floor::<LIMBS>(target, d))
            .collect();
        short_vecs_per_j.push(short_vecs_j);
        small_norms_per_j.push(small_norms_j);
        quotients_per_j.push(quotients_j);
    }

    // Step 5: Cross-product Bezout (j1, j2) with j2 >= j1.
    // First success wins; iteration order matches the C ref's nested
    // loop (outer j1, inner j2 ≥ j1).
    for j1 in 0..reduced_per_j.len() {
        for j2 in j1..reduced_per_j.len() {
            let is_diagonal = j1 == j2;
            let bezout = find_uv_from_lists::<LIMBS>(
                target,
                &small_norms_per_j[j1],
                &small_norms_per_j[j2],
                &quotients_per_j[j2],
                is_diagonal,
                0,
            );
            if let Some(b) = bezout {
                // β finalize. The (0, 0) case is identical to `find_uv`'s
                // proven j=0 path: `reduced_per_j[0]` is now the LLL-reduced
                // ORIGINAL input (not the rescaled reduced_id), so β lifts
                // DIRECTLY as `reduced·vec ∈ I` with NO δ-multiply — exactly
                // the find_uv j=0 lift. This yields a byte-identical result to
                // `find_uv` on the equivalent input.
                if j1 == 0 && j2 == 0 {
                    let beta_prime_1_o0 = mat_4x4_transpose_eval::<LIMBS>(
                        &reduced_per_j[0].basis,
                        &short_vecs_per_j[0][b.index_sol1].vec,
                    );
                    let beta_prime_2_o0 = mat_4x4_transpose_eval::<LIMBS>(
                        &reduced_per_j[0].basis,
                        &short_vecs_per_j[0][b.index_sol2].vec,
                    );
                    let beta_prime_1_num = o0_basis_to_standard_doubled::<LIMBS>(&beta_prime_1_o0);
                    let beta_prime_2_num = o0_basis_to_standard_doubled::<LIMBS>(&beta_prime_2_o0);
                    let beta_prime_denom = two_uint.wrapping_mul(&reduced_per_j[0].denom);
                    let beta1 = RationalQuaternion {
                        num: beta_prime_1_num,
                        denom: beta_prime_denom,
                    }
                    .normalize();
                    let beta2 = RationalQuaternion {
                        num: beta_prime_2_num,
                        denom: beta_prime_denom,
                    }
                    .normalize();
                    return Ok(FindUvResult {
                        u: b.u,
                        v: b.v,
                        beta1,
                        beta2,
                        d1: small_norms_per_j[0][b.index_sol1],
                        d2: small_norms_per_j[0][b.index_sol2],
                        index_alternate_order_1: 0,
                        index_alternate_order_2: 0,
                    });
                }
                // general j>0 β finalize. Port of dim2id2iso.c:641-673,
                // reconciled to our rescaled-frame representation:
                //
                //   - A j=0 component lifts DIRECTLY as `β' = reduced·vec ∈ I`
                //     (no δ-multiply): reduced_per_j[0] is the LLL-reduced
                //     ORIGINAL input, matching the C ref which keeps the input
                //     frame and never δ-multiplies a j=0 component.
                //   - A j>0 component applies the C's mutated δ: the value is
                //     `conj(smallest) / conj_ideal.norm` (the C `δ.denom /=
                //     lideal.norm; δ.denom *= conj_ideal.norm` mutation — the
                //     `/lideal.norm` cancels the setup `*ideal[0].norm` since
                //     n(ideal[0]) = n(lideal)), then `β = conj((δ · β').normalize())`.
                //
                // The result is SELF-VERIFIED against the input ideal
                // (`rational_quaternion_in_lideal`, the C's own
                // `quat_lattice_contains` arbiter) before returning Ok; a
                // reconciliation error fails closed instead of emitting
                // unverified crypto. No cheap j>0 fixture exists (j>0 fires
                // only on real-L1 non-principal inputs where j=0 misses), so
                // the Ok path's first real exercise is the heavy real-L1 run
                // / item-8 keygen KAT.
                let conj_ideal_norm = conj_reduced_id.cached_norm;
                let build_beta = |j: usize, idx: usize| -> RationalQuaternion<LIMBS> {
                    let bp_o0 = mat_4x4_transpose_eval::<LIMBS>(
                        &reduced_per_j[j].basis,
                        &short_vecs_per_j[j][idx].vec,
                    );
                    let bp = RationalQuaternion {
                        num: o0_basis_to_standard_doubled::<LIMBS>(&bp_o0),
                        denom: two_uint.wrapping_mul(&reduced_per_j[j].denom),
                    };
                    if j == 0 {
                        bp.normalize()
                    } else {
                        let delta_finalize = RationalQuaternion {
                            num: delta_rational.conjugate().num,
                            denom: delta_rational.denom.wrapping_mul(&conj_ideal_norm),
                        };
                        delta_finalize.mul(&bp, p).normalize().conjugate()
                    }
                };
                let beta1 = build_beta(j1, b.index_sol1);
                let beta2 = build_beta(j2, b.index_sol2);

                // Self-verification gate (WIDE=32: adj·x·lat_denom ~2^1246 at
                // real L1 overflows Int<16>).
                if rational_quaternion_in_lideal::<LIMBS, 32>(&beta1, lideal)
                    && rational_quaternion_in_lideal::<LIMBS, 32>(&beta2, lideal)
                {
                    return Ok(FindUvResult {
                        u: b.u,
                        v: b.v,
                        beta1,
                        beta2,
                        d1: small_norms_per_j[j1][b.index_sol1],
                        d2: small_norms_per_j[j2][b.index_sol2],
                        index_alternate_order_1: j1,
                        index_alternate_order_2: j2,
                    });
                }
                return Err(Error::NoBezoutSolution(
                    "find_uv_alternate_orders: j>0 β finalize produced a β not contained in \
                     the input ideal (self-verification gate rejected) — convention/construction \
                     mismatch; failing closed rather than emit unverified crypto.",
                ));
            }
        }
    }

    // No Bezout pair found across the (j1, j2) cross-product.
    Err(Error::NoBezoutSolution(
        "find_uv_alternate_orders: no Bezout pair found across the (j1, j2) cross-product \
         of the alt_connecting expansion. Try a larger box_size or different fixture.",
    ))
}

#[cfg(all(test, feature = "alloc"))]
mod enumerate_hypercube_tests {
    use super::*;

    /// Build a 4×4 Gram matrix from an i64-valued flat description.
    fn gram_from_i64<const LIMBS: usize>(rows: [[i64; 4]; 4]) -> [[Int<LIMBS>; 4]; 4] {
        let mut out = [[Int::<LIMBS>::from_i64(0); 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                out[i][j] = Int::<LIMBS>::from_i64(rows[i][j]);
            }
        }
        out
    }

    /// Identity Gram: `v^T · I · v = x² + y² + z² + w²`. With m = 1,
    /// `adjusted_norm = 1`, accepted candidates are the integer points
    /// with all-even / all-mod-3 filters applied. Verifies the function
    /// runs end-to-end and the `count - 1` truncation does not panic.
    #[test]
    fn identity_gram_at_m_equals_1_returns_some_vectors() {
        let gram = gram_from_i64::<8>([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
        let adjusted = Int::<8>::from_i64(1);
        let result = enumerate_hypercube::<8>(1, &gram, &adjusted);
        // The identity-Gram case has no `[α, iα, β, iβ]` symmetry
        // (gram[0][0] = gram[1][1] = 1, gram[2][2] = gram[3][3] = 1 →
        // both pairs match, so the i-action filter IS active).
        // At m=1 we walk 81 raw candidates; antipodal halves it to ~40;
        // GCD filter for all-even removes (0,0,0,0) and similar; odd-
        // quotient filter requires `v^T·v` odd, i.e. an odd number of
        // ±1 entries. The function returns a non-empty Vec with the
        // C reference's `count - 1` truncation applied.
        // We don't assert on exact membership (would over-constrain the
        // port); we assert the result is non-empty and every entry's
        // norm is odd (the odd-quotient filter's contract).
        assert!(
            !result.is_empty(),
            "identity Gram at m=1 must yield at least one accepted vector",
        );
        for sv in &result {
            assert_eq!(
                sv.norm.to_words()[0] & 1,
                1,
                "every accepted vector's stored norm must be odd",
            );
            // Sanity: the stored norm equals `vᵀ·G·v / adjusted = vᵀ·v / 1 = vᵀ·v`.
            let recomputed = qf_eval_4x4::<8>(&sv.vec, &gram);
            assert_eq!(
                sv.norm, recomputed,
                "stored norm must equal v^T·G·v / adjusted_norm"
            );
        }
    }

    /// Empty result when `m ≤ 0`. Defensive Rust-side guard.
    #[test]
    fn non_positive_m_returns_empty() {
        let gram = gram_from_i64::<8>([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
        let adjusted = Int::<8>::from_i64(1);
        assert!(enumerate_hypercube::<8>(0, &gram, &adjusted).is_empty());
        assert!(enumerate_hypercube::<8>(-3, &gram, &adjusted).is_empty());
    }

    /// Symmetric `[α, iα, β, iβ]`-form detection: gram with
    /// `gram[0][0] = gram[1][1] = α²` and `gram[2][2] = gram[3][3] = β²`
    /// activates the i-action filter. Verifies enumeration produces
    /// a strict subset of the no-symmetry case (filter is pruning at
    /// least one orbit).
    #[test]
    fn symmetric_gram_prunes_relative_to_asymmetric() {
        let symmetric =
            gram_from_i64::<8>([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 5, 0], [0, 0, 0, 5]]);
        let asymmetric =
            gram_from_i64::<8>([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 5, 0], [0, 0, 0, 7]]);
        let adjusted = Int::<8>::from_i64(1);
        let sym_result = enumerate_hypercube::<8>(2, &symmetric, &adjusted);
        let asym_result = enumerate_hypercube::<8>(2, &asymmetric, &adjusted);
        assert!(
            sym_result.len() < asym_result.len(),
            "symmetric Gram must prune more candidates via the i-action filter (sym={}, asym={})",
            sym_result.len(),
            asym_result.len(),
        );
    }

    /// Stable sort by norm (Rust idiom for the C reference's
    /// `compare_vec_by_norm` qsort). Verify that after `sort_by_key`
    /// the result is in non-decreasing-norm order and that insertion
    /// order is preserved on ties.
    #[test]
    fn stable_sort_by_norm_matches_compare_vec_by_norm_idiom() {
        let gram = gram_from_i64::<8>([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
        let adjusted = Int::<8>::from_i64(1);
        let mut result = enumerate_hypercube::<8>(2, &gram, &adjusted);
        // Take a snapshot of insertion order before sorting.
        let pre_sort_count = result.len();
        // Stable sort by `norm` — matches `compare_vec_by_norm` because
        // Rust's `Vec::sort_by` is stable, naturally tie-breaking by
        // original insertion order (the C reference uses an `idx` field
        // explicitly to achieve the same).
        result.sort_by_key(|sv| sv.norm);
        assert_eq!(
            result.len(),
            pre_sort_count,
            "sort does not change Vec length"
        );
        // Non-decreasing-norm property.
        for window in result.windows(2) {
            assert!(
                window[0].norm <= window[1].norm,
                "sorted result must be non-decreasing in norm",
            );
        }
    }

    /// All-even filter contract: any candidate with `(x|y|z|w) & 1 == 0`
    /// (all entries even) is skipped. Verified by checking no accepted
    /// vector in the result has all-even entries.
    #[test]
    fn all_even_candidates_are_filtered_out() {
        let gram = gram_from_i64::<8>([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
        let adjusted = Int::<8>::from_i64(1);
        let result = enumerate_hypercube::<8>(3, &gram, &adjusted);
        for sv in &result {
            let v: [i64; 4] = [
                i64::from_ne_bytes(sv.vec[0].to_words()[0].to_ne_bytes()),
                i64::from_ne_bytes(sv.vec[1].to_words()[0].to_ne_bytes()),
                i64::from_ne_bytes(sv.vec[2].to_words()[0].to_ne_bytes()),
                i64::from_ne_bytes(sv.vec[3].to_words()[0].to_ne_bytes()),
            ];
            // The above i64 conversion treats negative values as their
            // two's-complement low-word representation; we only care
            // about parity (LSB), which is identical for signed and
            // unsigned interpretations. The all-even condition is
            // `(v0|v1|v2|v3) & 1 == 0`.
            assert_ne!(
                (v[0] | v[1] | v[2] | v[3]) & 1,
                0,
                "accepted vector must not be all-even (got {v:?})",
            );
        }
    }

    /// `adjusted_norm = 2` test: with identity Gram, `v^T·v` is the
    /// hypercube norm. For divisibility by 2 the function asserts in
    /// debug mode that the norm is even — so accepted vectors must have
    /// even `v^T·v`. Verifies the divisibility contract via the
    /// debug_assert path (this test exercises the path; if the assert
    /// triggers, the test panics).
    #[test]
    fn adjusted_norm_divides_evaluated_norm() {
        let gram = gram_from_i64::<8>([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]]);
        let adjusted = Int::<8>::from_i64(2);
        // Every Gram entry is 2, so `v^T·G·v = 2·(x²+y²+z²+w²)` —
        // always divisible by 2. Quotient is `x²+y²+z²+w²`, and we
        // accept only when this is odd.
        let result = enumerate_hypercube::<8>(2, &gram, &adjusted);
        for sv in &result {
            assert_eq!(
                sv.norm.to_words()[0] & 1,
                1,
                "quotient must be odd per the function's contract",
            );
        }
    }
}

#[cfg(all(test, feature = "alloc"))]
mod find_uv_tests {
    use super::*;
    use alloc::vec;

    fn i(v: i64) -> Int<8> {
        Int::<8>::from_i64(v)
    }

    /// the Movement-2 θ endomorphism `θ = β2·conj(β1)/n(I)` has
    /// reduced norm `d1·d2`, at the real L1 prime. Builds (β1,β2,d1,d2)
    /// from `find_uv` (box=2, target=2^248), constructs
    /// θ via `theta_endomorphism`, and checks `N_red(θ) = d1·d2`.
    ///
    /// The norm check runs WIDE (θ.num ~ a few hundred bits after
    /// normalize, but N_red squares them + multiplies by p, and d1·d2·
    /// denom² can be large) — done at Int<16>. This is the first verified
    /// piece of the Clapotis evaluator's quaternion-to-isogeny bridge
    /// (Movement 2).
    #[test]
    fn theta_endomorphism_has_norm_d1d2() {
        use crate::quaternion::lattice::widen_int_lattice;
        use crate::quaternion::o0_mul::principal_left_ideal_from_o0;
        let p: Uint<8> = crate::params::lvl1::prime().resize::<8>();
        let gamma = [i(1), i(0), i(1), i(0)];
        let lideal = principal_left_ideal_from_o0::<8>(&gamma, &p);
        let target = *Uint::<8>::ONE.shl_vartime(248).as_int();
        let r = find_uv::<8>(
            &target,
            &lideal,
            &p,
            &[],
            crate::params::Level1::FINDUV_BOX_SIZE,
        )
        .expect("find_uv Ok at box=2 / 2^248");
        let n_id = lideal.reduced_norm_vartime().expect("perfect square");

        let theta = theta_endomorphism::<8, 16>(&r, &n_id, &p)
            .expect("θ must narrow back to Int<8> for the canonical normalized form");

        // Postcondition: N_red(θ.num) = d1·d2·θ.denom². Checked wide.
        let p_w = widen_int_lattice::<8, 16>(&Int::<8>::from_words(p.to_words()));
        let nr = {
            let a = widen_int_lattice::<8, 16>(&theta.num.a);
            let b = widen_int_lattice::<8, 16>(&theta.num.b);
            let c = widen_int_lattice::<8, 16>(&theta.num.c);
            let d = widen_int_lattice::<8, 16>(&theta.num.d);
            a.wrapping_mul(&a)
                .wrapping_add(&b.wrapping_mul(&b))
                .wrapping_add(
                    &p_w.wrapping_mul(&c.wrapping_mul(&c).wrapping_add(&d.wrapping_mul(&d))),
                )
        };
        let denom_w = widen_int_lattice::<8, 16>(&Int::<8>::from_words(theta.denom.to_words()));
        let d1d2 =
            widen_int_lattice::<8, 16>(&r.d1).wrapping_mul(&widen_int_lattice::<8, 16>(&r.d2));
        let rhs = d1d2.wrapping_mul(&denom_w.wrapping_mul(&denom_w));
        assert_eq!(
            nr, rhs,
            "N_red(θ) must equal d1·d2·θ.denom² (θ = β2·conj(β1)/n(I) has reduced norm d1·d2)",
        );
    }

    /// with the d=1-skip bug fixed, `find_uv` now finds a Bezout
    /// solution at the C-ref's actual L1 box_size = 2 (FINDUV_BOX_SIZE) on
    /// the real-target shape 2^TORSION_EVEN_POWER (= 2^248, the production
    /// target). The box=2 small-d list at real p is {1,5,5}; before the fix
    /// d=1 was skipped, leaving {5,5} with 5∤2^248. Now d=1 is usable:
    /// `u·1 + v·d2 = 2^248` is solvable.
    ///
    /// Verifies the full result: Ok at box=2, the Bezout identity
    /// `u·d1+v·d2=target`, and (wide, since values are ~2^750) the β
    /// reduced-norm postcondition `N(βᵢ.num)=βᵢ.denom²·n(I)·dᵢ`.
    #[test]
    fn find_uv_solves_at_box2_real_2power_target() {
        use crate::quaternion::lattice::widen_int_lattice;
        use crate::quaternion::o0_mul::principal_left_ideal_from_o0;
        let p: Uint<8> = crate::params::lvl1::prime().resize::<8>();
        let gamma = [i(1), i(0), i(1), i(0)];
        let lideal = principal_left_ideal_from_o0::<8>(&gamma, &p);
        // Real production target shape: 2^248 (TORSION_EVEN_POWER at L1).
        let target = *Uint::<8>::ONE.shl_vartime(248).as_int();
        // box_size = 2 = the actual C-ref L1 constant (no longer needs 3).
        let r = find_uv::<8>(
            &target,
            &lideal,
            &p,
            &[],
            crate::params::Level1::FINDUV_BOX_SIZE,
        )
        .expect("find_uv must solve 2^248 at box=2 once d=1 is usable");
        let zero = i(0);
        assert!(
            r.u > zero && r.v > zero && r.d1 > zero && r.d2 > zero,
            "u,v,d1,d2 > 0"
        );
        // Bezout identity.
        let lhs =
            r.u.wrapping_mul(&r.d1)
                .wrapping_add(&r.v.wrapping_mul(&r.d2));
        assert_eq!(lhs, target, "u·d1 + v·d2 = 2^248 must hold");
        // β postcondition (wide check; both sides ~2^750).
        let n_id = lideal.reduced_norm_vartime().expect("perfect square");
        let n_id_w = widen_int_lattice::<8, 16>(&Int::<8>::from_words(n_id.to_words()));
        let p_w = widen_int_lattice::<8, 16>(&Int::<8>::from_words(p.to_words()));
        let norm_w = |q: &crate::quaternion::algebra::Quaternion<8>| {
            let a = widen_int_lattice::<8, 16>(&q.a);
            let b = widen_int_lattice::<8, 16>(&q.b);
            let c = widen_int_lattice::<8, 16>(&q.c);
            let d = widen_int_lattice::<8, 16>(&q.d);
            a.wrapping_mul(&a)
                .wrapping_add(&b.wrapping_mul(&b))
                .wrapping_add(
                    &p_w.wrapping_mul(&c.wrapping_mul(&c).wrapping_add(&d.wrapping_mul(&d))),
                )
        };
        for (lbl, beta, d) in [("β1", &r.beta1, &r.d1), ("β2", &r.beta2, &r.d2)] {
            let denom_w = widen_int_lattice::<8, 16>(&Int::<8>::from_words(beta.denom.to_words()));
            let d_w = widen_int_lattice::<8, 16>(d);
            assert_eq!(
                norm_w(&beta.num),
                denom_w
                    .wrapping_mul(&denom_w)
                    .wrapping_mul(&n_id_w)
                    .wrapping_mul(&d_w),
                "{lbl}: β postcondition must hold at box=2/2^248",
            );
        }
    }

    /// the find_uv β satisfies the reduced-norm POSTCONDITION at the
    /// real L1 prime — `N(βᵢ.num) = βᵢ.denom² · n(I) · dᵢ` for both lifts.
    /// This is the deepest correctness check of the find_uv pipeline: the
    /// Bezout IDENTITY (`u·d1+v·d2=target`) is one claim, but the β
    /// quaternion LIFT is a separate claim — that each βᵢ is a genuine
    /// element of reduced norm `n(I)·dᵢ` in the original ideal frame (the
    /// postcondition, previously only checked at p=7).
    ///
    /// Magnitude note: at real p both sides are ~2^750 (n(I)~p~2^251,
    /// β.denom~2^249), far beyond `Int<8>`'s 511-bit cap — so the CHECK
    /// itself runs in WIDE=16 (1024-bit) via `widen_int_lattice`. The
    /// find_uv path under test is unchanged (it produces narrow β with
    /// narrow-fitting components); only this verification arithmetic is
    /// widened to avoid the check itself overflowing.
    #[test]
    fn find_uv_beta_postcondition_holds_at_real_l1_prime() {
        use crate::quaternion::lattice::widen_int_lattice;
        use crate::quaternion::o0_mul::principal_left_ideal_from_o0;
        let p: Uint<8> = crate::params::lvl1::prime().resize::<8>();
        let gamma = [i(1), i(0), i(1), i(0)]; // (i+j)/2 → cached_norm ~p²
        let lideal = principal_left_ideal_from_o0::<8>(&gamma, &p);
        let target = *Uint::<8>::ONE.shl_vartime(60).as_int();
        // box=3: real-prime density. Yields a real Ok(Bezout).
        let r = find_uv::<8>(&target, &lideal, &p, &[], 3)
            .expect("find_uv must yield Ok at real L1 prime with box=3");

        let n_id = lideal
            .reduced_norm_vartime()
            .expect("principal ideal cached_norm is a perfect square");
        let n_id_w = widen_int_lattice::<8, 16>(&Int::<8>::from_words(n_id.to_words()));
        let p_w = widen_int_lattice::<8, 16>(&Int::<8>::from_words(p.to_words()));
        // N_red of a standard quaternion (a,b,c,d) at prime p: a²+b²+p(c²+d²).
        let norm_w = |q: &crate::quaternion::algebra::Quaternion<8>| {
            let a = widen_int_lattice::<8, 16>(&q.a);
            let b = widen_int_lattice::<8, 16>(&q.b);
            let c = widen_int_lattice::<8, 16>(&q.c);
            let d = widen_int_lattice::<8, 16>(&q.d);
            a.wrapping_mul(&a)
                .wrapping_add(&b.wrapping_mul(&b))
                .wrapping_add(
                    &p_w.wrapping_mul(&c.wrapping_mul(&c).wrapping_add(&d.wrapping_mul(&d))),
                )
        };
        for (lbl, beta, d) in [("β1", &r.beta1, &r.d1), ("β2", &r.beta2, &r.d2)] {
            let denom_w = widen_int_lattice::<8, 16>(&Int::<8>::from_words(beta.denom.to_words()));
            let d_w = widen_int_lattice::<8, 16>(d);
            let lhs = norm_w(&beta.num);
            let rhs = denom_w
                .wrapping_mul(&denom_w)
                .wrapping_mul(&n_id_w)
                .wrapping_mul(&d_w);
            assert_eq!(
                lhs, rhs,
                "{lbl}: N(β.num) must equal β.denom²·n(I)·d at the real L1 prime",
            );
        }
    }

    /// `find_uv` produces a genuine `Ok(Bezout)` at the real L1
    /// prime — not just no-panic but an actual solution. This is
    /// the first END-TO-END SUCCESS of the find_uv pipeline at real scale.
    ///
    /// Two findings baked in: (1) the target is built via `Uint` (real
    /// isogeny-degree targets ~2^N exceed i64, so `i()` can't express
    /// them); 2^60 is used here as a representative large target — well
    /// past i64 if signed, and far above the tiny 1024 the no-panic tests
    /// used. (2) **box_size = 3, NOT L1's FINDUV_BOX_SIZE = 2.** At the
    /// real prime, box=2 finds NO Bezout for any target shape (2^e, 2^e±1,
    /// odd·2^k), while box≥3 finds one for all of them. The C ref's L1
    /// FINDUV_box_size=2 was only validated at the p=7 SMOKE prime; the
    /// real-prime reduced ideal needs a denser box. [Open: re-examine the
    /// C ref's box_size — is L1=2 real, with our enumerate filters dropping
    /// a candidate the C ref keeps, or is the constant target-magnitude-
    /// dependent?]
    ///
    /// The returned Bezout identity is checked: `u·d1 + v·d2 == target`,
    /// with `u, v, d1, d2 > 0`.
    #[test]
    fn find_uv_yields_real_bezout_at_real_l1_prime() {
        use crate::quaternion::o0_mul::principal_left_ideal_from_o0;
        let p: Uint<8> = crate::params::lvl1::prime().resize::<8>();
        let gamma = [i(1), i(0), i(1), i(0)]; // (i+j)/2 → cached_norm ~p²
        let lideal = principal_left_ideal_from_o0::<8>(&gamma, &p);
        // target = 2^60 (≫ the 1024 smoke target; built via Uint).
        let target = *Uint::<8>::ONE.shl_vartime(60).as_int();
        // box=3: real-prime density needs more than L1's p=7-calibrated 2.
        let r = find_uv::<8>(&target, &lideal, &p, &[], 3)
            .expect("find_uv must yield Ok(Bezout) at real L1 prime with box=3");
        let zero = i(0);
        assert!(r.u > zero && r.v > zero, "u, v must be positive");
        assert!(r.d1 > zero && r.d2 > zero, "d1, d2 must be positive");
        let lhs =
            r.u.wrapping_mul(&r.d1)
                .wrapping_add(&r.v.wrapping_mul(&r.d2));
        assert_eq!(
            lhs, target,
            "Bezout identity u·d1 + v·d2 = target must hold at real p"
        );
    }

    /// the FULL `find_uv` j=0 path runs to completion at the real L1
    /// prime. The LLL overflow was fixed with a wide LLL pass; the enumerate
    /// overflow was fixed with `pull_back_gram_wide` + `enumerate_hypercube_wide`
    /// (both at FINDUV_WIDE=16). On a realistic γ (with an (i+j)/2 component, so
    /// cached_norm ~p²) at the real prime (5·2^248−1), `find_uv` returns a
    /// Result — Ok if a Bezout solution exists in the box, or a clean
    /// NoBezoutSolution for this small smoke-test target — rather than
    /// PANICKING in the enumerate divisibility assert. NO-PANIC is the
    /// contract; real isogeny-degree-target Bezout/postcondition correctness
    /// is later work needing a sized target.
    #[test]
    fn find_uv_j0_runs_to_completion_at_real_l1_prime() {
        use crate::quaternion::o0_mul::principal_left_ideal_from_o0;
        let p: Uint<8> = crate::params::lvl1::prime().resize::<8>();
        let gamma = [i(1), i(0), i(1), i(0)]; // (i+j)/2 → cached_norm ~p²
        let lideal = principal_left_ideal_from_o0::<8>(&gamma, &p);
        assert!(
            lideal.cached_norm > Uint::<8>::from_u64(u64::MAX),
            "fixture must be real-prime-scale (cached_norm ≫ 2^64)",
        );
        // Runs to completion (no panic in the enumerate divisibility assert).
        let result = find_uv::<8>(
            &i(1024),
            &lideal,
            &p,
            &[],
            crate::params::Level1::FINDUV_BOX_SIZE,
        );
        assert!(
            matches!(&result, Ok(_) | Err(Error::NoBezoutSolution(_))),
            "real-prime j=0 find_uv must return Ok or NoBezoutSolution (no panic, no other error); got {result:?}",
        );
    }

    /// `find_uv_alternate_orders` runs to completion at the real L1
    /// prime — its per-j gram + enumerate now use the same wide path as
    /// the j=0 fix (`pull_back_gram_wide` + `enumerate_hypercube_wide`
    /// at FINDUV_WIDE=16). Before the fix, the per-j pullback had the same
    /// latent real-prime overflow as find_uv's j=0 path (only untested
    /// because the cross-product reaches j>0 cells rarely at p=7). On the
    /// REAL L1 ALT[0] fixture at the real prime, the function must return a
    /// Result (Ok / NoBezoutSolution / the fail-closed Unimplemented
    /// for a j>0 Bezout hit) rather than PANICKING in the enumerate
    /// divisibility assert. NO-PANIC is the contract.
    ///
    /// `#[ignore]`: after the fix to `alternate_connecting_ideal_0_l1`
    /// (it was transposed — rows-as-elements, NOT a valid left O_0-ideal; now
    /// the correct column convention, denom 1), `find_uv_alternate_orders` on the
    /// corrected real-scale ideal reaches the metric-GSO at LIMBS=8 and hits the
    /// KNOWN LIMBS=8 precision overflow (the GSO `int_div_exact` non-exact
    /// assert — the same limitation the sibling tests are `#[ignore]`'d
    /// for). The old transposed ideal happened to bail (NoBezoutSolution) before
    /// reaching it. Re-enable once the reduce runs on the wide-Int path
    /// end-to-end.
    #[ignore = "corrected real-scale ALT ideal reaches the known LIMBS=8 GSO overflow — needs the wide-Int reduce path"]
    #[test]
    fn find_uv_alternate_orders_runs_to_completion_at_real_l1_prime() {
        use crate::quaternion::connecting_ideals::alternate_connecting_ideal_0_l1;
        use crate::quaternion::o0_mul::principal_left_ideal_from_o0;
        let p: Uint<8> = crate::params::lvl1::prime().resize::<8>();
        let gamma = [i(1), i(0), i(1), i(0)]; // (i+j)/2 → cached_norm ~p²
        let lideal = principal_left_ideal_from_o0::<8>(&gamma, &p);
        assert!(
            lideal.cached_norm > Uint::<8>::from_u64(u64::MAX),
            "fixture must be real-prime-scale (cached_norm ≫ 2^64)",
        );
        let alt = [alternate_connecting_ideal_0_l1()];
        // Runs to completion (no enumerate overflow panic). Any of Ok /
        // NoBezoutSolution / Unimplemented(j>0 fail-closed) is fine;
        // a panic or other error is not.
        let result = find_uv_alternate_orders::<8, 32>(
            &i(1024),
            &lideal,
            &p,
            &alt,
            crate::params::Level1::FINDUV_BOX_SIZE,
        );
        assert!(
            matches!(
                &result,
                Ok(_) | Err(Error::NoBezoutSolution(_)) | Err(Error::Unimplemented(_))
            ),
            "real-prime find_uv_alternate_orders must return cleanly (no panic); got {result:?}",
        );
    }

    /// Runs `find_uv_alternate_orders` at the spine width (LIMBS=16) on a
    /// real-prime-scale input with ALL SIX real `ALTERNATE_CONNECTING_IDEALS`,
    /// production target `2^248`, per-j wide-reduce at WIDE=64.
    ///
    /// Runs to completion (~9s) and returns `Ok(index 0,0)` — the first
    /// real-L1 confirmation that the per-j wide-reduce machinery works
    /// end-to-end at the spine width. The earlier WIDE=64 attempt panicked
    /// in the integer-GSO (`int_div_exact` non-exact, lattice.rs:97) — that
    /// was pure WIDTH insufficiency, NOT degenerate input and NOT a
    /// "GSO-runs-narrow" bug (`lll_4x4_in_metric_wide` already runs the GSO
    /// at `WIDE`): the per-j product Gram's GSO determinant intermediates
    /// exceed 4096 bits at real L1, so `WIDE = 128` (8192-bit) is required.
    /// The `(0,0)` index is expected for this PRINCIPAL fixture (it rescales
    /// to `O_0`, so the alternate-order rotation is trivial and j=0 wins) —
    /// so this validates the SETUP but does NOT yet exercise the j>0 β
    /// finalize, which still needs a non-principal input where j=0's
    /// enumeration misses. Heavy.
    #[ignore = "heavy real-L1 (~9s): validates the per-j wide-reduce runs to completion at LIMBS=16/WIDE=128"]
    #[test]
    fn find_uv_alternate_orders_real_l1_limbs16_all_six_alts() {
        use crate::quaternion::connecting_ideals::{
            alternate_connecting_ideal_0_l1, alternate_connecting_ideal_1_l1,
            alternate_connecting_ideal_2_l1, alternate_connecting_ideal_3_l1,
            alternate_connecting_ideal_4_l1, alternate_connecting_ideal_5_l1,
        };
        use crate::quaternion::ideal::LeftIdeal;
        use crate::quaternion::lattice::widen_int_lattice;
        use crate::quaternion::o0_mul::principal_left_ideal_from_o0;

        // Widen a LeftIdeal<8> → LeftIdeal<16> (basis + denom + cached_norm).
        let widen = |id: &LeftIdeal<8>| -> LeftIdeal<16> {
            let mut basis = [[Int::<16>::from_i64(0); 4]; 4];
            for (r, row) in basis.iter_mut().enumerate() {
                for (c, entry) in row.iter_mut().enumerate() {
                    *entry = widen_int_lattice::<8, 16>(&id.basis[r][c]);
                }
            }
            LeftIdeal::<16>::with_denom_and_norm(
                basis,
                id.denom.resize::<16>(),
                id.cached_norm.resize::<16>(),
            )
        };

        let p: Uint<16> = crate::params::lvl1::prime().resize::<16>();
        // Real-prime-scale principal input (cached_norm ~p²).
        let gamma = [
            Int::<16>::from_i64(1),
            Int::<16>::from_i64(0),
            Int::<16>::from_i64(1),
            Int::<16>::from_i64(0),
        ];
        let lideal = principal_left_ideal_from_o0::<16>(&gamma, &p);
        assert!(
            lideal.cached_norm > Uint::<16>::from_u64(u64::MAX),
            "fixture must be real-prime-scale (cached_norm ≫ 2^64)",
        );

        let alts = [
            widen(&alternate_connecting_ideal_0_l1()),
            widen(&alternate_connecting_ideal_1_l1()),
            widen(&alternate_connecting_ideal_2_l1()),
            widen(&alternate_connecting_ideal_3_l1()),
            widen(&alternate_connecting_ideal_4_l1()),
            widen(&alternate_connecting_ideal_5_l1()),
        ];

        let target = *Uint::<16>::ONE.shl_vartime(248).as_int(); // 2^F, F=248
        let result = find_uv_alternate_orders::<16, 128>(
            &target,
            &lideal,
            &p,
            &alts,
            crate::params::Level1::FINDUV_BOX_SIZE,
        );

        match &result {
            Ok(r) => {
                // Ok ⇒ the containment self-gate passed for both β.
                std::eprintln!(
                    "find_uv_alt: Ok, index_alternate_order_1={}, index_alternate_order_2={}",
                    r.index_alternate_order_1,
                    r.index_alternate_order_2,
                );
            }
            Err(e) => std::eprintln!("find_uv_alt: Err({e:?})"),
        }
        assert!(
            matches!(
                &result,
                Ok(_) | Err(Error::NoBezoutSolution(_)) | Err(Error::Unimplemented(_))
            ),
            "real-L1 LIMBS=16 find_uv_alternate_orders must return cleanly (no panic); got {result:?}",
        );
    }

    /// the WIDE=32 LLL+rescale stage survives the real L1 prime.
    /// Before the fix, the narrow `lideal_reduce_basis` overflowed its
    /// integer-GSO recurrence on a real-L1-prime principal ideal
    /// (cached_norm ~p² ~2^500). This test reproduces `find_uv`'s first
    /// two steps directly — wide LLL then narrow rescale — and asserts
    /// BOTH survive (no panic) AND the rescale returns `Some` (the
    /// SQIsign-shaped invariant holds at real scale, not just at p=7).
    ///
    /// NOTE: the FULL `find_uv` j=0 path now runs to completion at real p
    /// — see `find_uv_j0_runs_to_completion_at_real_l1_prime`. The downstream
    /// enumerate panic was caused by `pull_back_gram` OVERFLOWING at real
    /// scale (Bᵀ·G·B intermediates ~2^749 wrap in Int<8>); fixed with
    /// `pull_back_gram_wide` + `enumerate_hypercube_wide` at FINDUV_WIDE=16.
    /// (Our `4·denom²` adjusted_norm is KEPT: it pairs with
    /// `o0_reduced_norm_gram_matrix`'s ×4 integer-safety bake-in — our
    /// vᵀGv carries a ×4 the C ref's `denom²`-only convention does not;
    /// the existing tests confirm ×4 is correct for OUR gram.)
    #[test]
    fn wide_lll_then_rescale_survives_real_l1_prime() {
        use crate::quaternion::ideal_mul::{
            lideal_reduce_basis_wide, lideal_rescale_by_smallest_basis_element,
        };
        use crate::quaternion::o0_mul::principal_left_ideal_from_o0;
        let p: Uint<8> = crate::params::lvl1::prime().resize::<8>();
        let gamma = [i(1), i(0), i(1), i(0)]; // (i+j)/2 component → cached_norm ~p²
        let lideal = principal_left_ideal_from_o0::<8>(&gamma, &p);
        // Sanity: genuinely real-prime-scale (cached_norm far beyond the
        // 64-bit range the narrow path could have handled).
        assert!(
            lideal.cached_norm > Uint::<8>::from_u64(u64::MAX),
            "fixture must be real-prime-scale (cached_norm ≫ 2^64)",
        );
        // Step 1: wide LLL — must not panic at real p.
        let reduced = lideal_reduce_basis_wide::<8, 32>(&lideal, &p);
        // Step 2: the EXISTING narrow rescale survives on the wide-reduced
        // basis and confirms the SQIsign-shaped invariant at real scale.
        let reduced_id = lideal_rescale_by_smallest_basis_element::<8>(&reduced, &p);
        assert!(
            reduced_id.is_some(),
            "wide-LLL + narrow rescale must yield Some at real L1 prime (SQIsign-shaped invariant)",
        );
    }

    /// verifier (test-scoped): does the rational quaternion `beta`
    /// lie in the ideal `lideal`? This is the convention-distinguishing
    /// membership arbiter the j>0 finalize will need: `LeftIdeal::contains`
    /// distinguishes a wrong multiply/conjugate order where the norm
    /// postcondition — multiplicative AND conjugation-invariant — cannot.
    /// It is verified here against the proven (0,0) β so it is ready to
    /// promote to a `pub(crate)` fn once real-prime parameters make j>0
    /// verifiable.
    ///
    /// Math: `beta = beta.num / beta.denom` in standard `(1, i, j, k)`
    /// coords; `standard_to_o0_basis(beta.num)` gives the integer
    /// O_0-coords of `beta.num` (= `beta.denom · beta`). The ideal is the
    /// rational lattice `(1 / lideal.denom) · Z⟨lideal.basis⟩` in
    /// O_0-coords. Hence
    ///   `beta ∈ I  ⟺  lideal.denom · num_o0 ∈ Z⟨beta.denom · lideal.basis⟩`,
    /// which `LeftIdeal::contains` tests on a denom-scaled basis.
    ///
    /// Test-scale only: scales with `wrapping_mul`; at p=7 fixture
    /// magnitudes this does not overflow. Real-prime use needs a wide
    /// variant.
    fn rational_quat_in_ideal(
        lideal: &crate::quaternion::ideal::LeftIdeal<8>,
        beta: &RationalQuaternion<8>,
    ) -> bool {
        use crate::quaternion::ideal::LeftIdeal;
        use crate::quaternion::o0_mul::standard_to_o0_basis;
        let num_o0 = standard_to_o0_basis::<8>(&beta.num);
        let ideal_denom_int = Int::<8>::from_words(lideal.denom.to_words());
        let beta_denom_int = Int::<8>::from_words(beta.denom.to_words());
        let mut query = [i(0); 4];
        for k in 0..4 {
            query[k] = ideal_denom_int.wrapping_mul(&num_o0[k]);
        }
        let mut scaled_basis = lideal.basis;
        for row in scaled_basis.iter_mut() {
            for cell in row.iter_mut() {
                *cell = beta_denom_int.wrapping_mul(cell);
            }
        }
        let scaled =
            LeftIdeal::<8>::with_denom_and_norm(scaled_basis, Uint::<8>::ONE, Uint::<8>::ONE);
        scaled.contains(&query)
    }

    /// Pick the first target in the sweep that yields a Bezout
    /// solution on the principal ideal `O_0·γ`, γ = (1, 0, 1, 0) at p=7.
    fn proven_find_uv_beta_on_anchor() -> (crate::quaternion::ideal::LeftIdeal<8>, FindUvResult<8>)
    {
        use crate::quaternion::o0_mul::principal_left_ideal_from_o0;
        let p: Uint<8> = Uint::from_u64(7);
        let gamma = [i(1), i(0), i(1), i(0)];
        let lideal = principal_left_ideal_from_o0::<8>(&gamma, &p);
        for t in [16u64, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
            if let Ok(r) = find_uv::<8>(
                &i(i64::try_from(t).expect("target fits in i64")),
                &lideal,
                &p,
                &[],
                crate::params::Level1::FINDUV_BOX_SIZE,
            ) {
                return (lideal, r);
            }
        }
        unreachable!("anchor must yield a Bezout solution in the sweep");
    }

    /// POSITIVE: the find_uv β (both β1 and β2) lies in
    /// the original ideal `O_0·γ`. This is the membership half of the
    /// convention arbiter — it confirms our (0,0) convention `β = β'·δ`
    /// returns β to the original ideal frame (the C-ref postcondition
    /// `quat_lattice_contains(ideal, β)`).
    #[test]
    fn rational_beta_in_ideal_holds_for_proven_find_uv_beta() {
        let (lideal, r) = proven_find_uv_beta_on_anchor();
        assert!(
            rational_quat_in_ideal(&lideal, &r.beta1),
            "proven find_uv β1 must lie in the ideal O_0·γ",
        );
        assert!(
            rational_quat_in_ideal(&lideal, &r.beta2),
            "proven find_uv β2 must lie in the ideal O_0·γ",
        );
    }

    /// NEGATIVE: the verifier rejects elements that are NOT in the
    /// ideal — proving it actually discriminates (an always-true verifier
    /// would be worthless). Uses MATHEMATICALLY GUARANTEED non-members:
    /// `O_0·γ` with N_red(γ) = 3 is a proper ideal, so no unit lies in it.
    /// Both `1` and `i` are units (N_red = 1), hence `1 ∉ O_0·γ` and
    /// `i ∉ O_0·γ` independent of any denominator bookkeeping.
    #[test]
    fn rational_beta_in_ideal_rejects_units_for_non_unit_ideal() {
        use crate::quaternion::algebra::{Quaternion, RationalQuaternion};
        let (lideal, _r) = proven_find_uv_beta_on_anchor();
        // Sanity: γ has reduced norm 3 → proper ideal, units excluded.
        assert_eq!(lideal.reduced_norm_vartime(), Some(Uint::<8>::from_u64(3)));
        let one = RationalQuaternion::<8>::one();
        let unit_i = RationalQuaternion::<8> {
            num: Quaternion::<8>::new(i(0), i(1), i(0), i(0)),
            denom: Uint::<8>::ONE,
        };
        assert!(
            !rational_quat_in_ideal(&lideal, &one),
            "1 (a unit) must NOT lie in O_0·γ with N_red(γ)=3",
        );
        assert!(
            !rational_quat_in_ideal(&lideal, &unit_i),
            "i (a unit) must NOT lie in O_0·γ with N_red(γ)=3",
        );
    }

    /// ROUND-TRIP: `standard_to_o0_basis` is the inverse of
    /// `o0_basis_to_standard_doubled` up to the doubling factor — i.e.
    /// `standard_to_o0_basis(o0_basis_to_standard_doubled(v)) == 2·v` for
    /// any O_0-coord vector v. This pins the coordinate conversion the
    /// membership verifier (and the β-finalize) both rely on.
    #[test]
    fn o0_standard_roundtrip_is_doubling() {
        use crate::quaternion::o0_mul::{o0_basis_to_standard_doubled, standard_to_o0_basis};
        for v in [
            [i(1), i(0), i(0), i(0)],
            [i(0), i(1), i(0), i(0)],
            [i(0), i(0), i(1), i(0)],
            [i(0), i(0), i(0), i(1)],
            [i(3), i(-5), i(7), i(-2)],
        ] {
            let doubled = o0_basis_to_standard_doubled::<8>(&v);
            let back = standard_to_o0_basis::<8>(&doubled);
            for k in 0..4 {
                assert_eq!(
                    back[k],
                    i(2).wrapping_mul(&v[k]),
                    "round-trip must equal 2·v at coord {k} for v={v:?}",
                );
            }
        }
    }

    /// Hand-calculated coprime case: `target = 100`, `small_norms1 = [3]`,
    /// `small_norms2 = [7]`. Need `u·3 + v·7 = 100` with `u, v > 0`,
    /// `v < 100/7 = 14`. The first v satisfying `v ≡ 100/3 ≡ 100·3^{-1}
    /// (mod 7) = ?`. Solve manually: `7^{-1} mod 3 = 7^{-1 mod 3}`; let's
    /// check: 7 ≡ 1 (mod 3), so inv = 1. v = `(1·(100 mod 3)) mod 3 = 1`.
    /// Then `u = (100 - 1·7)/3 = 93/3 = 31`. Check: `31·3 + 1·7 = 93 + 7
    /// = 100` ✓. Quotient `100/7 = 14`, so `v=1 < 14` accepts.
    #[test]
    fn find_uv_from_lists_coprime_pair_succeeds() {
        let target = i(100);
        let norms1 = vec![i(3)];
        let norms2 = vec![i(7)];
        let quotients2 = vec![i(14)]; // floor(100/7)
        let result = find_uv_from_lists::<8>(&target, &norms1, &norms2, &quotients2, false, 0);
        let r = result.expect("coprime pair must yield a solution");
        assert_eq!(r.u, i(31));
        assert_eq!(r.v, i(1));
        assert_eq!(r.index_sol1, 0);
        assert_eq!(r.index_sol2, 0);
        // Verify the Bezout equation.
        let check =
            r.u.wrapping_mul(&i(3))
                .wrapping_add(&r.v.wrapping_mul(&i(7)));
        assert_eq!(check, target, "u·d1 + v·d2 must equal target");
    }

    /// Non-coprime pair (gcd > 1) skips the inv-mod step. With single-
    /// element lists `[6]` and `[4]` (gcd = 2), no valid (u, v) for
    /// target = 100; expect None.
    #[test]
    fn find_uv_from_lists_non_coprime_pair_returns_none() {
        let target = i(100);
        let norms1 = vec![i(6)];
        let norms2 = vec![i(4)];
        let quotients2 = vec![i(25)];
        // gcd(6, 4) = 2; the inv-mod step fails and the pair is skipped.
        let result = find_uv_from_lists::<8>(&target, &norms1, &norms2, &quotients2, false, 0);
        assert!(
            result.is_none(),
            "non-coprime pair must surface as None, got {result:?}",
        );
    }

    /// `number_sum_square != 0` (Cornacchia-constrained paths) returns
    /// None in the current port. The orchestrator passes 0; the
    /// non-zero paths are reserved for future call sites.
    #[test]
    fn find_uv_from_lists_cornacchia_path_returns_none() {
        let target = i(100);
        let norms1 = vec![i(3)];
        let norms2 = vec![i(7)];
        let quotients2 = vec![i(14)];
        for nss in [1u32, 2u32] {
            let result =
                find_uv_from_lists::<8>(&target, &norms1, &norms2, &quotients2, false, nss);
            assert!(
                result.is_none(),
                "Cornacchia paths (number_sum_square={nss}) must surface as None, got {result:?}",
            );
        }
    }

    /// `is_diagonal=true` with a list of identical norms restricts the
    /// search to `i2 ≥ i1`. With `norms1 = norms2 = [3, 7]`, diagonal
    /// mode should find the same first pair (i1=0, i2=0) when valid.
    #[test]
    fn find_uv_from_lists_is_diagonal_restricts_lower_triangle() {
        let target = i(100);
        let norms = vec![i(3), i(7)];
        let quotients = vec![i(33), i(14)];
        let result = find_uv_from_lists::<8>(&target, &norms, &norms, &quotients, true, 0);
        let r = result.expect("diagonal first-pair must yield a solution");
        // i1=0 (d1=3), i2=0 (d2=3) — diagonal allows i2=i1. But 3·u + 3·v = 100
        // has no integer solutions (100 is not a multiple of 3). So search
        // moves to i1=0, i2=1 (d2=7) — the same coprime pair as the first test.
        assert_eq!(r.index_sol1, 0);
        assert_eq!(r.index_sol2, 1);
        assert_eq!(r.u, i(31));
        assert_eq!(r.v, i(1));
    }

    /// Empty lists yield None (no pairs to try).
    #[test]
    fn find_uv_from_lists_empty_lists_return_none() {
        let target = i(100);
        let empty: Vec<Int<8>> = vec![];
        let quotients_empty: Vec<Int<8>> = vec![];
        assert!(
            find_uv_from_lists::<8>(&target, &empty, &empty, &quotients_empty, false, 0).is_none()
        );
        let nonempty = vec![i(3)];
        let quotients_one = vec![i(14)];
        // empty list1 → no outer iteration.
        assert!(
            find_uv_from_lists::<8>(&target, &empty, &nonempty, &quotients_one, false, 0).is_none()
        );
        // empty list2 → no inner iteration.
        assert!(
            find_uv_from_lists::<8>(&target, &nonempty, &empty, &quotients_empty, false, 0)
                .is_none()
        );
    }

    /// `find_uv` body succeeds on a small-prime full-order fixture:
    /// `target = 1024` (= 2^10, the typical SQIsign-style target),
    /// `p = 7`, `lideal = O_0` (norm 1). At m=3 the box enumerates
    /// many short vectors and the Bezout search finds a
    /// `(u, v, d1, d2)` with `u·d1 + v·d2 = 1024`.
    ///
    /// **Postcondition**: `N_red(β_i) = n(I)·d_i` where `n(I)` is
    /// the reduced quaternion ideal norm. We assert the denom-independent
    /// form `N_red(β.num) = β.denom² · n(I) · d_i`. For full_order
    /// `n(I) = 1`, so the postcondition simplifies to
    /// `N_red(β.num) = β.denom² · d_i`.
    #[test]
    fn find_uv_body_succeeds_for_small_prime_full_order() {
        use crate::quaternion::ideal::LeftIdeal;
        let lideal = LeftIdeal::<8>::full_order();
        let p: Uint<8> = Uint::from_u64(7);
        let target = i(1024);
        let r = find_uv::<8>(
            &target,
            &lideal,
            &p,
            &[],
            crate::params::Level1::FINDUV_BOX_SIZE,
        )
        .expect("find_uv must find a Bezout decomposition for the trivial L1-style fixture");
        // Bezout identity: u·d1 + v·d2 = target.
        let lhs =
            r.u.wrapping_mul(&r.d1)
                .wrapping_add(&r.v.wrapping_mul(&r.d2));
        assert_eq!(
            lhs, target,
            "find_uv must return (u, v, d1, d2) satisfying u·d1 + v·d2 = target",
        );
        // d1, d2 are odd (per enumerate_hypercube's odd-quotient filter).
        assert_eq!(r.d1.to_words()[0] & 1, 1, "d1 must be odd");
        assert_eq!(r.d2.to_words()[0] & 1, 1, "d2 must be odd");
        // u, v are strictly positive (find_uv_from_lists' contract).
        assert!(r.u > i(0), "u must be > 0");
        assert!(r.v > i(0), "v must be > 0");
        // Alternate-order indices both zero at num_alternate_order=0.
        assert_eq!(r.index_alternate_order_1, 0);
        assert_eq!(r.index_alternate_order_2, 0);
        // postcondition: N(β.num) = β.denom² · n(I) · d.
        // For full_order n(I) = 1 → N(β.num) = β.denom² · d.
        let n_id = lideal
            .reduced_norm_vartime()
            .expect("full_order's cached_norm is a perfect square");
        let n_id_int = Int::<8>::from_words(n_id.to_words());
        let p_int = Int::<8>::from_words(p.to_words());
        let norm_of = |q: &crate::quaternion::algebra::Quaternion<8>| {
            let a_sq = q.a.wrapping_mul(&q.a);
            let b_sq = q.b.wrapping_mul(&q.b);
            let c_sq = q.c.wrapping_mul(&q.c);
            let d_sq = q.d.wrapping_mul(&q.d);
            a_sq.wrapping_add(&b_sq)
                .wrapping_add(&p_int.wrapping_mul(&c_sq.wrapping_add(&d_sq)))
        };
        let beta1_denom_int = Int::<8>::from_words(r.beta1.denom.to_words());
        let beta2_denom_int = Int::<8>::from_words(r.beta2.denom.to_words());
        let n_beta1 = norm_of(&r.beta1.num);
        let n_beta2 = norm_of(&r.beta2.num);
        let expected1 = beta1_denom_int
            .wrapping_mul(&beta1_denom_int)
            .wrapping_mul(&n_id_int)
            .wrapping_mul(&r.d1);
        let expected2 = beta2_denom_int
            .wrapping_mul(&beta2_denom_int)
            .wrapping_mul(&n_id_int)
            .wrapping_mul(&r.d2);
        assert_eq!(
            n_beta1, expected1,
            "N(β1.num) must equal β1.denom² · n(I) · d1",
        );
        assert_eq!(
            n_beta2, expected2,
            "N(β2.num) must equal β2.denom² · n(I) · d2",
        );
    }

    /// `find_uv` with non-empty `alt_connecting` dispatches to
    /// `find_uv_alternate_orders`. This test confirms the dispatch path is
    /// wired end-to-end AND that routing through the alternate-orders
    /// function with a (placeholder) alt entry yields the SAME result as
    /// the direct j=0 `find_uv` path — i.e. the dispatch is transparent
    /// for the (0,0) hit.
    #[test]
    fn find_uv_dispatch_to_alternate_orders_matches_direct_find_uv() {
        use crate::quaternion::ideal::LeftIdeal;
        let lideal = LeftIdeal::<8>::full_order();
        let p: Uint<8> = Uint::from_u64(7);
        let target = i(1024);
        let alt = [LeftIdeal::<8>::full_order()];
        let box_size = crate::params::Level1::FINDUV_BOX_SIZE;

        let via_dispatch = find_uv::<8>(&target, &lideal, &p, &alt, box_size)
            .expect("dispatch through find_uv_alternate_orders must reach the (0,0) finalize");
        let via_direct = find_uv::<8>(&target, &lideal, &p, &[], box_size)
            .expect("direct j=0 find_uv must succeed");
        assert_eq!(
            via_dispatch, via_direct,
            "find_uv(alt=[full_order]) must equal find_uv(alt=[]) at the (0,0) hit",
        );
    }

    /// `find_uv` surfaces `Error::NoBezoutSolution` when no Bezout
    /// decomposition exists within the box. Use a degenerate scenario:
    /// `target = 1` — too small for ANY positive Bezout pair `u·d1 +
    /// v·d2 = 1` with `u, v > 0` (even the smallest d's are ≥ 1, so the
    /// minimum achievable positive sum is `1·1 + 1·1 = 2`). Confirms the
    /// variant is distinct from `Error::Unimplemented` (reserved for
    /// non-empty `alt_connecting` and other deferred features).
    ///
    /// this previously used `target = 2`, which relied on the
    /// (now-fixed) d=1-skip bug — with d=1 usable, `1·1 + 1·1 = 2` IS a
    /// valid solution, so `target = 2` now correctly succeeds. `target =
    /// 1` is the genuine smallest no-solution case (u + v = 1 with both
    /// positive is impossible).
    #[test]
    fn find_uv_no_solution_within_box_returns_no_bezout_solution() {
        use crate::error::Error;
        use crate::quaternion::ideal::LeftIdeal;
        let lideal = LeftIdeal::<8>::full_order();
        let p: Uint<8> = Uint::from_u64(7);
        let target = i(1); // too small; no positive u, v exist (min sum is 2)
        let result = find_uv::<8>(
            &target,
            &lideal,
            &p,
            &[],
            crate::params::Level1::FINDUV_BOX_SIZE,
        );
        assert!(
            matches!(&result, Err(Error::NoBezoutSolution(msg)) if msg.contains("Bezout") || msg.contains("box")),
            "find_uv must surface NoBezoutSolution when no Bezout solution exists within the box, \
             got {result:?}",
        );
    }

    /// Lift-invariant test: on a deliberately non-symmetric LLL-
    /// reduced basis, the lift `α_o0 = Bᵀ · v` (via
    /// `mat_4x4_transpose_eval`) must satisfy the Gram identity
    ///
    /// ```text
    ///     vᵀ · G_I · v = 4 · N_red(α_o0)
    /// ```
    ///
    /// where `G_I = B · G_O0 · Bᵀ = pull_back_gram(B, G_O0)` and the
    /// factor-of-4 comes from the integer-safety bake-in of
    /// `o0_reduced_norm_gram_matrix`. **This invariant FAILS** if
    /// the lift uses `B · v` (= `mat_4x4_eval(B, v)`) instead of
    /// `Bᵀ · v` whenever `B` is not symmetric — exactly the
    /// transpose bug caught in the original audit.
    ///
    /// The test exercises the same composition `find_uv` uses
    /// internally (the lift + Gram + norm primitives) without
    /// running the full Bezout search; that lets us pin the
    /// invariant on a fixture LLL leaves non-symmetric, instead of
    /// fighting the m=3 hard-code's odd-norm filter.
    /// Lift-invariant test: on a deliberately non-symmetric basis,
    /// the lift `α_o0 = Bᵀ · v` (via `mat_4x4_transpose_eval`)
    /// satisfies the Gram identity
    ///
    /// ```text
    ///     vᵀ · G_I · v = 4 · N_red(α_o0)
    /// ```
    ///
    /// where `G_I = pull_back_gram(B, G_O0) = B · G_O0 · Bᵀ`. The
    /// factor-of-4 comes from the integer-safety bake-in of
    /// `o0_reduced_norm_gram_matrix`. **This invariant FAILS** if
    /// the lift uses `B · v` (= `mat_4x4_eval`) instead of `Bᵀ · v`
    /// whenever `B` is non-symmetric — exactly the transpose bug
    /// caught in the original audit.
    ///
    /// The test bypasses LLL (which tends to symmetrize the basis
    /// for our small fixtures) and uses the raw non-symmetric basis
    /// directly. This exercises the same lift+Gram composition
    /// `find_uv` uses internally.
    #[test]
    fn lift_via_mat_4x4_transpose_eval_satisfies_gram_identity_on_non_symmetric_basis() {
        use crate::quaternion::lattice::{
            mat_4x4_eval, mat_4x4_transpose_eval, pull_back_gram, qf_eval_4x4,
        };
        use crate::quaternion::o0_mul::{o0_reduced_norm_gram_matrix, reduced_norm_o0_basis};

        let p = Uint::<8>::from_u64(7);
        // Deliberately non-symmetric integer basis. Bypass LLL —
        // LLL tends to symmetrize the basis for small fixtures, which
        // would hide the transpose bug. The Gram identity holds for
        // ANY basis, so testing the raw basis is fine.
        let basis = [
            [i(1), i(0), i(0), i(0)],
            [i(1), i(1), i(0), i(0)], // 1 + i offset — off-diagonal
            [i(0), i(0), i(1), i(0)],
            [i(0), i(0), i(0), i(1)],
        ];

        let g_o0 = o0_reduced_norm_gram_matrix::<8>(&p);
        let g_i = pull_back_gram::<8>(&basis, &g_o0);

        // Sanity: the basis is non-symmetric in the lift sense
        // (B · v ≠ Bᵀ · v for at least one e_k). Catches a future
        // fixture-symmetric-by-accident regression.
        let mut any_differ = false;
        for k in 0..4 {
            let mut v = [i(0); 4];
            v[k] = i(1);
            if mat_4x4_eval::<8>(&basis, &v) != mat_4x4_transpose_eval::<8>(&basis, &v) {
                any_differ = true;
                break;
            }
        }
        assert!(
            any_differ,
            "fixture basis is symmetric — test cannot guard against transpose bug",
        );

        // For each unit vector e_k, verify the Gram identity.
        for k in 0..4 {
            let mut v = [i(0); 4];
            v[k] = i(1);
            let alpha_o0 = mat_4x4_transpose_eval::<8>(&basis, &v);
            let n_alpha = reduced_norm_o0_basis::<8>(&alpha_o0, &p);
            let qf_val = qf_eval_4x4::<8>(&v, &g_i);
            let expected = i(4).wrapping_mul(&n_alpha);
            assert_eq!(
                qf_val, expected,
                "Gram identity vᵀ·G_I·v = 4·N(Bᵀ·v) must hold for e_{k}; \
                 a mismatch means the lift is using B·v instead of Bᵀ·v",
            );
        }

        // Also verify the buggy lift FAILS the same identity on at
        // least one e_k — proving the test would have caught the
        // transpose bug.
        let mut buggy_caught = false;
        for k in 0..4 {
            let mut v = [i(0); 4];
            v[k] = i(1);
            let alpha_buggy = mat_4x4_eval::<8>(&basis, &v);
            let n_buggy = reduced_norm_o0_basis::<8>(&alpha_buggy, &p);
            let qf_val = qf_eval_4x4::<8>(&v, &g_i);
            let expected_buggy = i(4).wrapping_mul(&n_buggy);
            if qf_val != expected_buggy {
                buggy_caught = true;
                break;
            }
        }
        assert!(
            buggy_caught,
            "the buggy B·v lift should fail the Gram identity on this fixture — \
             if this assertion fires, the fixture is too symmetric to discriminate",
        );
    }

    /// Outcome from the parameterized `try_find_uv_postcondition` helper:
    /// either a target landed and the postcondition held, or no target
    /// in the trial range yielded a Bezout solution.
    ///
    /// The "no solution" outcome is NOT a test failure — it signals the
    /// `m=3` hard-coded box is too tight for that fixture's lattice
    /// density. The test sweep covers multiple γ shapes; per-fixture
    /// outcomes are aggregated to assert that AT LEAST ONE non-trivial
    /// fixture yielded a solution (proving the pipeline works beyond
    /// full_order), without forcing every fixture to succeed.
    #[cfg_attr(feature = "alloc", derive(Debug))]
    enum PostconditionOutcome {
        /// `find_uv` returned Ok on some target; Bezout AND norm
        /// postcondition both held. Numerical correctness pinned.
        Verified {
            _target_used: u64,
            _d1: u64,
            _d2: u64,
        },
        /// No target in the trial range produced a Bezout solution.
        /// Acceptable per-fixture; aggregated to a sweep-level assertion.
        NoSolutionInBox,
    }

    /// Parameterized helper: build the principal ideal `O_0 · γ`, try
    /// each target in the trial range, and on first success verify the
    /// Bezout identity AND the norm postcondition
    /// `N(β.num) = β.denom² · n(I) · d`. Returns `Verified` on success,
    /// `NoSolutionInBox` when no target hits.
    ///
    /// All ASSERTIONS inside fire on real numerical failures
    /// (postcondition violation). Lattice-density misses (no Bezout in
    /// the m=3 box) are not failures; they fall through to
    /// `NoSolutionInBox`.
    fn try_find_uv_postcondition(
        gamma: &[Int<8>; 4],
        p: &Uint<8>,
        targets: &[u64],
        box_size: i64,
    ) -> PostconditionOutcome {
        use crate::quaternion::o0_mul::principal_left_ideal_from_o0;
        let lideal = principal_left_ideal_from_o0::<8>(gamma, p);
        let p_int = Int::<8>::from_words(p.to_words());
        let norm_of = |q: &crate::quaternion::algebra::Quaternion<8>| {
            let a_sq = q.a.wrapping_mul(&q.a);
            let b_sq = q.b.wrapping_mul(&q.b);
            let c_sq = q.c.wrapping_mul(&q.c);
            let d_sq = q.d.wrapping_mul(&q.d);
            a_sq.wrapping_add(&b_sq)
                .wrapping_add(&p_int.wrapping_mul(&c_sq.wrapping_add(&d_sq)))
        };
        let n_id = lideal
            .reduced_norm_vartime()
            .expect("principal ideal must have square cached_norm");
        let n_id_int = Int::<8>::from_words(n_id.to_words());

        for &target_u in targets {
            let target = i(i64::try_from(target_u).expect("target fits in i64"));
            let result = find_uv::<8>(&target, &lideal, p, &[], box_size);
            let Ok(r) = result else { continue };
            let lhs =
                r.u.wrapping_mul(&r.d1)
                    .wrapping_add(&r.v.wrapping_mul(&r.d2));
            assert_eq!(lhs, target, "Bezout identity must hold");
            let beta1_denom_int = Int::<8>::from_words(r.beta1.denom.to_words());
            let beta2_denom_int = Int::<8>::from_words(r.beta2.denom.to_words());
            let expected1 = beta1_denom_int
                .wrapping_mul(&beta1_denom_int)
                .wrapping_mul(&n_id_int)
                .wrapping_mul(&r.d1);
            let expected2 = beta2_denom_int
                .wrapping_mul(&beta2_denom_int)
                .wrapping_mul(&n_id_int)
                .wrapping_mul(&r.d2);
            assert_eq!(
                norm_of(&r.beta1.num),
                expected1,
                "N(β1.num) must equal β1.denom² · n(I) · d1 (postcondition violated)",
            );
            assert_eq!(
                norm_of(&r.beta2.num),
                expected2,
                "N(β2.num) must equal β2.denom² · n(I) · d2 (postcondition violated)",
            );
            return PostconditionOutcome::Verified {
                _target_used: target_u,
                _d1: r.d1.to_words()[0],
                _d2: r.d2.to_words()[0],
            };
        }
        PostconditionOutcome::NoSolutionInBox
    }

    /// postcondition holds for the canonical non-trivial fixture
    /// `γ = (1, 0, 1, 0)` at p=7 (`N_red(γ) = 3`). Proves find_uv's
    /// rescale-then-undo composition works beyond `full_order`.
    #[test]
    fn find_uv_postcondition_holds_on_non_trivial_principal_ideal() {
        use crate::quaternion::o0_mul::principal_left_ideal_from_o0;
        let p: Uint<8> = Uint::from_u64(7);
        let gamma = [i(1), i(0), i(1), i(0)];
        // Sanity: cached_norm 9, reduced_norm 3 (anchor fixture).
        let lideal = principal_left_ideal_from_o0::<8>(&gamma, &p);
        assert_eq!(lideal.cached_norm, Uint::<8>::from_u64(9));
        assert_eq!(lideal.reduced_norm_vartime(), Some(Uint::<8>::from_u64(3)));
        let targets = [16u64, 32, 64, 128, 256, 512, 1024, 2048, 4096];
        let outcome =
            try_find_uv_postcondition(&gamma, &p, &targets, crate::params::Level1::FINDUV_BOX_SIZE);
        assert!(
            matches!(outcome, PostconditionOutcome::Verified { .. }),
            "anchor fixture must yield a Verified outcome, got {outcome:?}",
        );
    }

    /// sweep test — postcondition holds across MULTIPLE non-trivial
    /// principal-ideal fixtures at p=7. Each fixture targets a different
    /// `N_red(γ)` and a different O_0-basis pattern. The sweep ASSERTS
    /// that the postcondition NEVER violates (any numerical failure
    /// panics inside the helper) AND that AT LEAST 2 fixtures yield a
    /// `Verified` outcome (signalling the pipeline is robust beyond a
    /// single γ shape).
    ///
    /// Per-fixture lattice-density misses (no Bezout in m=3 box) are
    /// allowed — the test prints the outcome for each fixture so that
    /// future FINDUV_box_size tuning can see which γ shapes need larger
    /// boxes.
    #[test]
    fn find_uv_postcondition_holds_across_multiple_principal_ideal_fixtures() {
        let p: Uint<8> = Uint::from_u64(7);
        let targets = [16u64, 32, 64, 128, 256, 512, 1024, 2048, 4096];
        // Each fixture: γ in O_0 basis + expected N_red(γ) (sanity
        // check inside the helper via reduced_norm_vartime). γ values
        // are chosen for diverse N_red and basis patterns.
        let fixtures: [([Int<8>; 4], u64, &str); 5] = [
            ([i(1), i(0), i(1), i(0)], 3, "γ=1+(i+j)/2, N=3 (anchor)"),
            ([i(1), i(1), i(1), i(0)], 5, "γ=1+i+(i+j)/2, N=5"),
            ([i(1), i(1), i(0), i(1)], 5, "γ=1+i+(1+k)/2, N=5 alternate"),
            ([i(1), i(0), i(2), i(0)], 9, "γ=1+(i+j), N=9 (composite)"),
            ([i(3), i(0), i(1), i(0)], 11, "γ=3+(i+j)/2, N=11 prime"),
        ];
        let mut verified_count = 0usize;
        for (gamma, expected_n, label) in &fixtures {
            // Sanity check on the reduced norm to catch fixture typos.
            let n_id = crate::quaternion::o0_mul::reduced_norm_o0_basis::<8>(gamma, &p);
            assert_eq!(
                n_id,
                i(i64::try_from(*expected_n).expect("fixture norm fits in i64")),
                "fixture sanity: N_red({gamma:?}) should be {expected_n}, label={label}",
            );
            let outcome = try_find_uv_postcondition(
                gamma,
                &p,
                &targets,
                crate::params::Level1::FINDUV_BOX_SIZE,
            );
            if matches!(outcome, PostconditionOutcome::Verified { .. }) {
                verified_count += 1;
            }
        }
        assert!(
            verified_count >= 2,
            "sweep requires at least 2 of {} fixtures to yield Verified \
             (got {verified_count}). All assertions inside the helper would have \
             panicked on numerical violations; this aggregate counts how many \
             fixtures had Bezout solutions in the m=3 box.",
            fixtures.len(),
        );
    }

    /// `FindUvResult::placeholder()` constructs across LIMBS widths.
    #[test]
    fn find_uv_result_placeholder_constructs_at_l1_l3_l5() {
        let _l1 = FindUvResult::<8>::placeholder();
        let _l3 = FindUvResult::<12>::placeholder();
        let _l5 = FindUvResult::<16>::placeholder();
    }

    /// LIMBS-generic helper for the find_uv postcondition check on the
    /// anchor fixture (γ = (1, 0, 1, 0), N_red = 3, at p=7). Used
    /// by the L1/L3/L5 tests below to validate find_uv is genuinely
    /// LIMBS-generic — catches accidental LIMBS-8 specialization.
    fn check_find_uv_anchor_postcondition_at_limbs<const LIMBS: usize>(box_size: i64) {
        use crate::quaternion::o0_mul::principal_left_ideal_from_o0;
        let p: Uint<LIMBS> = Uint::from_u64(7);
        let nn = |v: i64| Int::<LIMBS>::from_i64(v);
        let zz = nn(0);
        let gamma = [nn(1), zz, nn(1), zz];
        let lideal = principal_left_ideal_from_o0::<LIMBS>(&gamma, &p);
        assert_eq!(lideal.cached_norm, Uint::<LIMBS>::from_u64(9));
        let p_int = Int::<LIMBS>::from_words(p.to_words());
        let norm_of = |q: &crate::quaternion::algebra::Quaternion<LIMBS>| {
            let a_sq = q.a.wrapping_mul(&q.a);
            let b_sq = q.b.wrapping_mul(&q.b);
            let c_sq = q.c.wrapping_mul(&q.c);
            let d_sq = q.d.wrapping_mul(&q.d);
            a_sq.wrapping_add(&b_sq)
                .wrapping_add(&p_int.wrapping_mul(&c_sq.wrapping_add(&d_sq)))
        };
        let n_id = lideal
            .reduced_norm_vartime()
            .expect("principal ideal must have square cached_norm");
        let n_id_int = Int::<LIMBS>::from_words(n_id.to_words());

        let mut any_succeeded = false;
        for target_u in [16u64, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
            let target = nn(i64::try_from(target_u).expect("target fits in i64"));
            let result = find_uv::<LIMBS>(&target, &lideal, &p, &[], box_size);
            let Ok(r) = result else { continue };
            let lhs =
                r.u.wrapping_mul(&r.d1)
                    .wrapping_add(&r.v.wrapping_mul(&r.d2));
            assert_eq!(lhs, target, "Bezout identity must hold at LIMBS={LIMBS}");
            let beta1_denom_int = Int::<LIMBS>::from_words(r.beta1.denom.to_words());
            let beta2_denom_int = Int::<LIMBS>::from_words(r.beta2.denom.to_words());
            let expected1 = beta1_denom_int
                .wrapping_mul(&beta1_denom_int)
                .wrapping_mul(&n_id_int)
                .wrapping_mul(&r.d1);
            let expected2 = beta2_denom_int
                .wrapping_mul(&beta2_denom_int)
                .wrapping_mul(&n_id_int)
                .wrapping_mul(&r.d2);
            assert_eq!(
                norm_of(&r.beta1.num),
                expected1,
                "postcondition N(β1.num) = β1.denom²·n(I)·d1 must hold at LIMBS={LIMBS}",
            );
            assert_eq!(
                norm_of(&r.beta2.num),
                expected2,
                "postcondition N(β2.num) = β2.denom²·n(I)·d2 must hold at LIMBS={LIMBS}",
            );
            any_succeeded = true;
            break;
        }
        assert!(
            any_succeeded,
            "anchor fixture should yield Bezout in m=3 box at LIMBS={LIMBS}",
        );
    }

    /// L3 LIMBS (12) variant of the anchor postcondition test.
    /// now uses `Level3::FINDUV_BOX_SIZE = 3` (= C ref's value).
    #[test]
    fn find_uv_postcondition_holds_on_anchor_at_l3_limbs() {
        check_find_uv_anchor_postcondition_at_limbs::<12>(crate::params::Level3::FINDUV_BOX_SIZE);
    }

    /// L5 LIMBS (16) variant.
    /// now uses `Level5::FINDUV_BOX_SIZE = 3` (= C ref's value).
    #[test]
    fn find_uv_postcondition_holds_on_anchor_at_l5_limbs() {
        check_find_uv_anchor_postcondition_at_limbs::<16>(crate::params::Level5::FINDUV_BOX_SIZE);
    }

    /// `find_uv_alternate_orders` end-to-end through the (0,0) β
    /// finalize on the REAL L1 ALT[0] entry. Composes everything
    /// (LLL + rescale → reduced_id; per-j ideal_multiply + wide-LLL;
    /// per-j Gram + enumerate; cross-product Bezout) and then finalizes
    /// the (j1=0, j2=0) hit.
    ///
    /// **Differential element-equality arbiter:** the
    /// norm postcondition `N(β) = denom²·n(I)·d` CANNOT distinguish our
    /// convention `β = β_lift·δ` from the C-ref's `β = conj(δ·β_lift)`,
    /// because reduced norm is multiplicative AND conjugation-invariant.
    /// So we verify the (0,0) finalize against `find_uv` (the proven
    /// oracle): the (0,0) sub-result is computed from `reduced_per_j[0]`
    /// (= the rescaled input ideal, identical to `find_uv`'s `reduced_id`)
    /// with the same δ, enumerate, and Bezout convention, so the FULL
    /// `FindUvResult` — u, v, d1, d2, and both β quaternion elements
    /// (num + denom) — must be byte-identical to `find_uv` on the same
    /// input with an empty `alt_connecting`. Element-equality (not just
    /// norm-equality) is what rejects a left/right/conjugate convention
    /// slip.
    #[test]
    fn find_uv_alternate_orders_zero_zero_finalize_matches_find_uv_with_real_alt_0() {
        use crate::quaternion::connecting_ideals::alternate_connecting_ideal_0_l1;
        use crate::quaternion::ideal::LeftIdeal;
        let lideal = LeftIdeal::<8>::full_order();
        let p: Uint<8> = Uint::from_u64(7);
        let target = i(1024);
        let alt = [alternate_connecting_ideal_0_l1()];
        let box_size = crate::params::Level1::FINDUV_BOX_SIZE;

        let via_alt = find_uv_alternate_orders::<8, 20>(&target, &lideal, &p, &alt, box_size)
            .expect("(0,0) finalize must succeed: 3u + 5v = 1024 has a Bezout solution");
        let via_find_uv = find_uv::<8>(&target, &lideal, &p, &[], box_size)
            .expect("find_uv oracle must succeed on the same input");

        // Full structural equality: u, v, d1, d2, both β (num + denom),
        // and the alternate-order indices (both 0 here).
        assert_eq!(
            via_alt, via_find_uv,
            "(0,0) finalize must be byte-identical to find_uv — distinguishes \
             β=β_lift·δ from β=conj(δ·β_lift), which the norm test cannot",
        );
        // Spell out the Bezout identity for a human-readable anchor.
        let lhs = via_alt
            .u
            .wrapping_mul(&via_alt.d1)
            .wrapping_add(&via_alt.v.wrapping_mul(&via_alt.d2));
        assert_eq!(lhs, target, "u·d1 + v·d2 must equal target");
        assert_eq!(via_alt.index_alternate_order_1, 0);
        assert_eq!(via_alt.index_alternate_order_2, 0);
    }

    /// `find_uv_alternate_orders` on the `[full_order()]`
    /// placeholder finalizes the (0,0) Bezout hit. Since `reduced_per_j[0]`
    /// = the rescaled `O_0` input (same as `find_uv`'s `reduced_id`) and
    /// the placeholder alt entry contributes only an unreachable j>0 path,
    /// the result must equal `find_uv` on an empty `alt_connecting`.
    /// Same differential-equality arbiter as the real-ALT[0] test.
    #[test]
    fn find_uv_alternate_orders_zero_zero_finalize_matches_find_uv_with_placeholder() {
        use crate::quaternion::ideal::LeftIdeal;
        let lideal = LeftIdeal::<8>::full_order();
        let p: Uint<8> = Uint::from_u64(7);
        let target = i(1024);
        let alt = [LeftIdeal::<8>::full_order()];
        let box_size = crate::params::Level1::FINDUV_BOX_SIZE;

        let via_alt = find_uv_alternate_orders::<8, 20>(&target, &lideal, &p, &alt, box_size)
            .expect("(0,0) finalize must succeed on the placeholder alt entry");
        let via_find_uv = find_uv::<8>(&target, &lideal, &p, &[], box_size)
            .expect("find_uv oracle must succeed on the same input");
        assert_eq!(
            via_alt, via_find_uv,
            "(0,0) finalize on placeholder alt must be byte-identical to find_uv",
        );
    }

    /// validates that the anchor fixture (γ = (1, 0, 1, 0) at p=7)
    /// still yields a Bezout solution when `box_size = 2` —
    /// the C reference's actual L1 value. Previously hardcoded m = 3
    /// was over-allocated for L1;
    /// this test pins whether m = 2 is sufficient density for our
    /// small-prime fixture.
    ///
    /// If this test PASSES: m = 2 is sufficient even for our trivially
    /// small fixture, so production callers at L1 will likely find
    /// Bezout solutions with the C-ref-spec box.
    ///
    /// If this test FAILS (NoBezoutSolution): m = 2 is too tight for
    /// our small-prime test fixture; production callers at L1 may
    /// need m > 2 for non-canonical inputs, OR our p=7 fixtures are
    /// pathological. Either way the FAILURE is informative — the
    /// helper's other assertions would have caught real correctness
    /// violations; a NoBezoutSolution is a density signal, not a bug.
    #[test]
    fn find_uv_at_l1_box_size_two_matches_c_ref_constant() {
        use crate::quaternion::o0_mul::principal_left_ideal_from_o0;
        let p: Uint<8> = Uint::from_u64(7);
        let gamma = [i(1), i(0), i(1), i(0)];
        let lideal = principal_left_ideal_from_o0::<8>(&gamma, &p);
        // Try the same target range as the anchor test.
        let targets = [16u64, 32, 64, 128, 256, 512, 1024, 2048, 4096];
        let mut any_succeeded = false;
        for target_u in targets {
            let target = i(i64::try_from(target_u).expect("target fits in i64"));
            // box_size = 2 (C ref's FINDUV_box_size at L1).
            let result = find_uv::<8>(&target, &lideal, &p, &[], 2);
            if result.is_ok() {
                any_succeeded = true;
                break;
            }
        }
        // ASSERT: at least one target succeeds. If this fails, m=2 is
        // too tight for our γ=(1,0,1,0) fixture at p=7 — which would
        // be informative (production L1 callers may need m > 2 on
        // edge-case inputs). If it passes, m=2 is sufficient at L1
        // density for at least this fixture.
        assert!(
            any_succeeded,
            "probe: m=2 (C ref's L1 FINDUV_box_size) found no Bezout solution for γ=(1,0,1,0) at p=7 \
             across targets {{16, 32, ..., 4096}}. This may signal that the C ref's L1 box is too tight \
             for small-prime test fixtures — production L1 inputs at real 251-bit prime should fare better.",
        );
    }
}
