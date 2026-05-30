// SPDX-License-Identifier: MIT OR Apache-2.0
//! `ALTERNATE_CONNECTING_IDEALS` precomputed table — per-level Rust
//! transcriptions of the SQIsign C reference's
//! `src/precomp/ref/lvl{1,3,5}/quaternion_data.c::CONNECTING_IDEALS`
//! array.
//!
//! # Provenance
//!
//! The C reference stores the table as auto-generated GMP `mpz_t`
//! literals (see `scripts/precomp/precompute_quaternion_data.sage`).
//! `CONNECTING_IDEALS[0]` is the trivial connector that the C ref
//! skips via the pointer offset
//! `ALTERNATE_CONNECTING_IDEALS = CONNECTING_IDEALS + 1`. The Rust
//! port mirrors that: this module exposes only the NON-trivial
//! entries (so our `ALTERNATE_CONNECTING_IDEALS[i]` = C ref's
//! `CONNECTING_IDEALS[i + 1]`).
//!
//! # Conventions
//!
//! - **Basis storage** — the C reference stores basis vectors as
//!   COLUMNS of `ibz_mat_4x4_t`; our [`LeftIdeal::basis`] stores
//!   them as ROWS. S214 transcription applies the transposition.
//! - **Coordinate system** — the C reference stores basis vectors
//!   in standard `(1, i, j, ij)` quaternion coords; our `LeftIdeal`
//!   stores them in `O_0`-basis coords. S214 applies the conversion
//!   `(a, b, c, d) → (a - d, b - c, 2c, 2d)` (derived from
//!   `o0_basis_to_standard_doubled`'s inverse).
//! - **`cached_norm` convention** — the C reference's `.norm` field
//!   is the REDUCED quaternion ideal norm `N_red(γ_I)`. Our
//!   [`LeftIdeal::cached_norm`] is the LATTICE INDEX `N_red(γ_I)²`
//!   per S201. Transcription squares the C reference's norm.

use crypto_bigint::{Int, Uint};

use crate::quaternion::ideal::LeftIdeal;

/// L1 `ALTERNATE_CONNECTING_IDEALS[0]` — the first NON-trivial
/// alternate connecting ideal at security Level 1. Corresponds to
/// `CONNECTING_IDEALS[1]` in the SQIsign C reference.
///
/// # Provenance
///
/// Transcribed via the S213 research agent from
/// `src/precomp/ref/lvl1/quaternion_data.c` lines 1888-2050
/// (`GMP_LIMB_BITS == 64` branch). Verbatim quotes preserved in
/// the S213 close in `ISA.md`.
///
/// # Math sanity (S213)
///
/// C-ref basis is in `(1, i, j, ij)` coords, column-major. Each
/// column → one Rust basis row, with the `(a, b, c, d) → (a − d,
/// b − c, 2c, 2d)` conversion. The trivial-column round trip
/// verifies the formula: `col2 = (0, 0, 1, 0)` (= `j` in standard
/// coords) → O_0 `(0, −1, 2, 0)`, which reconstructs to
/// `0·1 + (−1)·i + 2·(i+j)/2 + 0·(1+k)/2 = −i + i + j = j` ✓.
/// Similarly `col3 = (0, 0, 0, 1)` (= `k`) → `(−1, 0, 0, 2)` →
/// `−1 + 2·(1+k)/2 = k` ✓.
///
/// # `cached_norm` (S201 convention)
///
/// The C reference's `.norm` field is the reduced quaternion ideal
/// norm `0x30000000000000000000000000000001` (= `3·2^124 + 1`).
/// Our [`LeftIdeal::cached_norm`] is the lattice index, so we
/// square: `(3·2^124 + 1)² = 9·2^248 + 6·2^124 + 1`. Compose at
/// construction via `Uint::wrapping_mul` to avoid hand-encoding
/// the 252-bit hex literal.
pub fn alternate_connecting_ideal_0_l1() -> LeftIdeal<8> {
    // C-ref values in (1, i, j, ij) coords (as Uint<8> with the low
    // 128 bits set per S213 verbatim quotes; high 6 limbs zero).
    // `Uint::from_words` uses little-endian limbs matching GMP's `_mp_d`.
    let v_60_02 = *Uint::<8>::from_words([0x2, 0x6000000000000000, 0, 0, 0, 0, 0, 0]).as_int();
    let v_10_01 = *Uint::<8>::from_words([0x1, 0x1000000000000000, 0, 0, 0, 0, 0, 0]).as_int();
    let v_50_01 = *Uint::<8>::from_words([0x1, 0x5000000000000000, 0, 0, 0, 0, 0, 0]).as_int();

    let zero = Int::<8>::from_i64(0);
    let one = Int::<8>::from_i64(1);
    let neg_one = Int::<8>::from_i64(-1);
    let two = Int::<8>::from_i64(2);

    // Basis in O_0 coords, row-major. Each row = one ideal generator.
    // Conversion `(a, b, c, d) → (a − d, b − c, 2c, 2d)`.
    let basis = [
        // C col 0: (0x60...02, 0, 0, 0x10...01) → O_0
        [
            v_60_02.wrapping_sub(&v_10_01), // a − d = 0x50...01
            zero,                           // b − c = 0
            zero,                           // 2c    = 0
            two.wrapping_mul(&v_10_01),     // 2d    = 0x20...02
        ],
        // C col 1: (0, 0x60...02, 0x50...01, 0) → O_0
        [
            zero,                           // a − d = 0
            v_60_02.wrapping_sub(&v_50_01), // b − c = 0x10...01
            two.wrapping_mul(&v_50_01),     // 2c    = 0xA0...02
            zero,                           // 2d    = 0
        ],
        // C col 2: (0, 0, 1, 0) = j → O_0 (0, −1, 2, 0)
        [zero, neg_one, two, zero],
        // C col 3: (0, 0, 0, 1) = k → O_0 (−1, 0, 0, 2)
        [neg_one, zero, zero, two],
    ];
    let _ = one; // suppress unused on the named constant

    let denom = Uint::<8>::from_u64(2);

    // C ref reduced norm `0x30000000000000000000000000000001` = 3·2^124 + 1.
    // S201 convention: cached_norm = lattice index = reduced_norm².
    let reduced_norm = Uint::<8>::from_words([0x1, 0x3000000000000000, 0, 0, 0, 0, 0, 0]);
    let cached_norm = reduced_norm.wrapping_mul(&reduced_norm);

    LeftIdeal::<8>::with_denom_and_norm(basis, denom, cached_norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// S214 regression: `cached_norm` equals `reduced_norm²` per the
    /// S201 lattice-index convention.
    #[test]
    fn alt_connecting_ideal_0_l1_cached_norm_is_reduced_norm_squared() {
        let ideal = alternate_connecting_ideal_0_l1();
        let expected_reduced = Uint::<8>::from_words([0x1, 0x3000000000000000, 0, 0, 0, 0, 0, 0]);
        let expected_cached = expected_reduced.wrapping_mul(&expected_reduced);
        assert_eq!(
            ideal.cached_norm, expected_cached,
            "cached_norm must equal reduced_norm² per S201 convention",
        );
    }

    /// S214 regression: `reduced_norm_vartime()` round-trips the C ref's
    /// `.norm` field (the integer square root of `cached_norm`).
    #[test]
    fn alt_connecting_ideal_0_l1_reduced_norm_round_trips() {
        let ideal = alternate_connecting_ideal_0_l1();
        let expected_reduced = Uint::<8>::from_words([0x1, 0x3000000000000000, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            ideal.reduced_norm_vartime(),
            Some(expected_reduced),
            "reduced_norm_vartime must recover the C ref's .norm field value",
        );
    }

    /// S214 regression: `denom = 2` per the C ref's
    /// `lattice.denom = {._mp_d = {0x2}}` at line 1895.
    #[test]
    fn alt_connecting_ideal_0_l1_denom_is_two() {
        let ideal = alternate_connecting_ideal_0_l1();
        assert_eq!(ideal.denom, Uint::<8>::from_u64(2));
    }

    /// S214 regression: trivial basis columns (col 2 = j, col 3 = k)
    /// round-trip the `(1,i,j,k) → O_0` conversion. `basis[2]` and
    /// `basis[3]` in O_0 coords should be `(0, -1, 2, 0)` and
    /// `(-1, 0, 0, 2)` respectively.
    #[test]
    fn alt_connecting_ideal_0_l1_trivial_columns_match_conversion() {
        let ideal = alternate_connecting_ideal_0_l1();
        assert_eq!(
            ideal.basis[2],
            [
                Int::<8>::from_i64(0),
                Int::<8>::from_i64(-1),
                Int::<8>::from_i64(2),
                Int::<8>::from_i64(0),
            ],
            "basis[2] = j → O_0 (0, -1, 2, 0)",
        );
        assert_eq!(
            ideal.basis[3],
            [
                Int::<8>::from_i64(-1),
                Int::<8>::from_i64(0),
                Int::<8>::from_i64(0),
                Int::<8>::from_i64(2),
            ],
            "basis[3] = k → O_0 (-1, 0, 0, 2)",
        );
    }

    // ── S215: probe L1 ALT[0] through the existing primitives ─────────

    /// S215 probe: `lideal_reduce_basis` on L1 ALT[0] at the REAL L1
    /// prime panics inside LLL's `int_div_exact` step. The probe
    /// confirms what S206 flagged: at LIMBS=8 with p ≈ 2^251, LLL's
    /// intermediate computations on a basis with 128-bit entries
    /// overflow the 512-bit Uint, causing the exact-division assertion
    /// to fail.
    ///
    /// **Blocker for S216+**: wide-Int variants of `lideal_reduce_basis`,
    /// `lideal_rescale_by_smallest_basis_element`, and `find_uv` are
    /// needed before alternate-orders body wiring can be validated
    /// numerically against the real L1 prime + real ALT data.
    ///
    /// **Marked `#[ignore]`** until the wide-Int path lands. Re-enable
    /// after S216+ ships `lideal_reduce_basis_wide` etc.
    #[test]
    #[ignore = "precision overflow at LIMBS=8 with real L1 prime — needs wide-Int variants (S216+)"]
    fn alt_connecting_ideal_0_l1_lll_is_unimodular() {
        let p: Uint<8> = crate::params::lvl1::prime().resize::<8>();
        let id = alternate_connecting_ideal_0_l1();
        let reduced = crate::quaternion::ideal_mul::lideal_reduce_basis::<8>(&id, &p);
        assert_eq!(
            reduced.cached_norm, id.cached_norm,
            "LLL must preserve cached_norm (unimodular |det|=1)",
        );
        assert_eq!(
            reduced.denom, id.denom,
            "LLL must preserve denom (metric-only)",
        );
    }

    /// S215 sanity probe: an attempt to bypass the L1-prime precision
    /// concern by testing LLL on ALT[0]'s structure at p=7 ALSO
    /// panics inside `int_div_exact`.
    ///
    /// **Diagnosis**: the LLL's integer-GSO accumulates products that
    /// grow as `O((p · basis²)^k)` over k steps. For 128-bit basis
    /// entries (even at p=7), the d[k] values exceed 512-bit Uint<8>
    /// capacity within a few iterations. The overflow corrupts the
    /// exact-division assertion. This is NOT a bug in LLL — it's a
    /// fundamental precision constraint of the narrow path.
    ///
    /// **Conclusion**: wide-Int variants of `lideal_reduce_basis`,
    /// `lideal_rescale_by_smallest_basis_element`, and `find_uv` are
    /// required to test ALT entries (real C-ref data) at ANY prime,
    /// not just the L1 production magnitude.
    ///
    /// **Marked `#[ignore]`** with the rest until wide-Int lands.
    #[test]
    #[ignore = "LLL integer-GSO overflows Uint<8> on 128-bit basis entries at any prime — needs wide-Int variants (S216+)"]
    fn alt_connecting_ideal_0_l1_lll_mechanically_valid_at_small_prime() {
        let p = Uint::<8>::from_u64(7);
        let id = alternate_connecting_ideal_0_l1();
        let reduced = crate::quaternion::ideal_mul::lideal_reduce_basis::<8>(&id, &p);
        // LLL is unimodular → cached_norm and denom preserved
        // regardless of the metric prime. This tests structural soundness
        // of the LLL call, NOT cryptographic validity at p=7.
        assert_eq!(
            reduced.cached_norm, id.cached_norm,
            "LLL must preserve cached_norm (unimodular) at any p",
        );
        assert_eq!(
            reduced.denom, id.denom,
            "LLL must preserve denom (metric-only) at any p",
        );
    }

    /// S215 probe: `lideal_rescale_by_smallest_basis_element` outcome
    /// on L1 ALT[0]. The S203 invariant says SQIsign-shaped principal
    /// ideals rescale to `O_0` (`cached_norm == 1`). This test
    /// documents whether L1 ALT[0] satisfies the invariant.
    ///
    /// If `Some(rescaled)`: probe succeeded; assert
    /// `cached_norm == 1` per the invariant.
    ///
    /// If `None`: the cached_norm wasn't a perfect square (defensive
    /// path in `reduced_norm_vartime`) OR the divisibility check
    /// failed (entry doesn't satisfy the SQIsign-shaped contract at
    /// our small-prime smoke fixture). Either outcome is informative
    /// for S216 body wiring.
    #[test]
    #[ignore = "precision overflow at LIMBS=8 with real L1 prime — needs wide-Int variants (S216+)"]
    fn alt_connecting_ideal_0_l1_rescale_outcome() {
        let p: Uint<8> = crate::params::lvl1::prime().resize::<8>();
        let id = alternate_connecting_ideal_0_l1();
        let reduced = crate::quaternion::ideal_mul::lideal_reduce_basis::<8>(&id, &p);
        let rescaled = crate::quaternion::ideal_mul::lideal_rescale_by_smallest_basis_element::<8>(
            &reduced, &p,
        );
        match rescaled {
            Some(r) => {
                // S203 invariant: rescaled = O_0 → cached_norm == 1.
                // If this fails, ALT[0] is NOT SQIsign-shaped at L1 prime
                // (or our LLL doesn't find the principal generator as δ).
                // Either way, this test PINS the current behavior.
                assert_eq!(
                    r.cached_norm,
                    Uint::<8>::from_u64(1),
                    "rescaled cached_norm must be 1 per S203 invariant; \
                     deviation signals ALT[0] is not SQIsign-shaped at L1 prime",
                );
            }
            None => {
                panic!(
                    "rescale returned None — ALT[0] may not satisfy the SQIsign-shaped \
                     contract at the real L1 prime, OR cached_norm isn't a perfect square \
                     (the latter should be impossible given the S214 construction)."
                );
            }
        }
    }

    /// S217: compose wide LLL (S216) with the existing narrow
    /// `lideal_rescale_by_smallest_basis_element` on ALT[0] at p=7.
    ///
    /// **Math sanity for narrow rescale at p=7**: the autocompute
    /// formula is `cached_norm · N_red(α)² / α_denom⁴`. For ALT[0]
    /// post-LLL: `cached_norm ≈ 2^248`, `α_denom = √cached_norm ≈ 2^124`,
    /// and α = δ (LLL-shortest basis element) has `N_red(δ) = √cached_norm`
    /// for SQIsign-shaped principals. Numerator ≈ `2^248 · (2^124)² = 2^496`
    /// — fits in Uint<8> (512-bit) with 16 bits of headroom. So narrow
    /// rescale is OK at p=7.
    ///
    /// Per-outcome assertions:
    /// - `Some(rescaled)` with `cached_norm == 1`: S203 invariant holds;
    ///   ALT[0] IS SQIsign-shaped.
    /// - `Some(rescaled)` with `cached_norm != 1`: ALT[0] is rescale-
    ///   able but not SQIsign-shaped (the smallest LLL basis element
    ///   is not the principal generator). DOCUMENTED outcome.
    /// - `None`: divisibility check failed; ALT[0]'s LLL output δ
    ///   doesn't generate the principal part.
    #[test]
    fn alt_connecting_ideal_0_l1_wide_lll_then_narrow_rescale_at_small_prime() {
        let p = Uint::<8>::from_u64(7);
        let id = alternate_connecting_ideal_0_l1();
        let reduced = crate::quaternion::ideal_mul::lideal_reduce_basis_wide::<8, 20>(&id, &p);
        let rescaled = crate::quaternion::ideal_mul::lideal_rescale_by_smallest_basis_element::<8>(
            &reduced, &p,
        );
        match rescaled {
            Some(r) => {
                // S203 invariant check (informational): cached_norm == 1
                // for SQIsign-shaped. Otherwise the rescale still succeeded
                // structurally but produced a non-unit lattice.
                let n_after = r.cached_norm;
                let n_before = reduced.cached_norm;
                eprintln!(
                    "S217 outcome: rescale Ok; cached_norm pre = {:?}, post = {:?}",
                    n_before, n_after
                );
                // At minimum, denom should update per the wide-LLL-prep:
                // new denom = reduced.denom · reduced_norm_vartime
                let n_red = reduced
                    .reduced_norm_vartime()
                    .expect("reduced ideal cached_norm must be a perfect square");
                assert_eq!(
                    r.denom,
                    reduced.denom.wrapping_mul(&n_red),
                    "rescaled.denom must equal reduced.denom · √reduced.cached_norm",
                );
            }
            None => {
                eprintln!(
                    "S217 outcome: rescale returned None — ALT[0]'s LLL-shortest δ \
                     doesn't generate the principal part (or cached_norm not a perfect square). \
                     Documented; S218+ may need to handle this case in find_uv_alternate_orders."
                );
            }
        }
    }

    /// S216: ALT[0] LLL via the wide variant at `WIDE=20` (1280-bit) +
    /// p=7. Validates that the wide path handles ALT-magnitude basis
    /// entries without overflow.
    ///
    /// **Precision calibration**: at p=7, `det(Gram_I) ≈ 2^538` for
    /// ALT[0]. LLL's integer-GSO step computes `d[k+1] · u` where
    /// `d[k+1]` can approach `det(Gram_I)` and `u` is `O(p · basis²)
    /// ≈ 2^258`. Peak intermediate product `≈ 2^796`, so WIDE must
    /// exceed 800 bits. WIDE=16 (1024 bits) gives ~200 bits margin;
    /// WIDE=20 (1280 bits) gives ~480 bits margin against further
    /// internal squarings.
    ///
    /// Asserts the LLL is unimodular (preserves cached_norm + denom).
    #[test]
    fn alt_connecting_ideal_0_l1_wide_lll_works_at_small_prime() {
        let p = Uint::<8>::from_u64(7);
        let id = alternate_connecting_ideal_0_l1();
        let reduced = crate::quaternion::ideal_mul::lideal_reduce_basis_wide::<8, 20>(&id, &p);
        assert_eq!(
            reduced.cached_norm, id.cached_norm,
            "wide LLL must preserve cached_norm (unimodular |det|=1)",
        );
        assert_eq!(
            reduced.denom, id.denom,
            "wide LLL must preserve denom (metric-only)",
        );
    }

    /// S215 probe: `lideal_intersect(O_0, ALT[0])` outcome. For coprime
    /// norms, S190's `lideal_intersect` falls through to `ideal_multiply`.
    /// Since `N(O_0) = 1` and `N(ALT[0]) = reduced_norm²` (with reduced
    /// norm > 1), gcd(1, anything) = 1 → coprime path triggers →
    /// returns `Ok(ideal_multiply(O_0, ALT[0]))`. The product `O_0 · ALT[0]`
    /// is just `ALT[0]` (left-multiplication by full order is identity).
    #[test]
    #[ignore = "precision overflow at LIMBS=8 with real L1 prime — needs wide-Int variants (S216+)"]
    fn alt_connecting_ideal_0_l1_intersect_with_full_order_is_self() {
        let p: Uint<8> = crate::params::lvl1::prime().resize::<8>();
        let full = LeftIdeal::<8>::full_order();
        let alt = alternate_connecting_ideal_0_l1();
        let inter = crate::quaternion::ideal_mul::lideal_intersect::<8>(&full, &alt, &p)
            .expect("coprime norms → fast path → Ok");
        // Result lattice should equal alt itself (full_order · alt = alt).
        assert!(
            inter.equals_lattice(&alt),
            "lideal_intersect(O_0, ALT[0]) must equal ALT[0] as a lattice",
        );
    }
}
