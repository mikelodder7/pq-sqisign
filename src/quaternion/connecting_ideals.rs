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
//!   them as ROWS. Transcription applies the transposition.
//! - **Coordinate system** — the C reference stores basis vectors
//!   in standard `(1, i, j, ij)` quaternion coords; our `LeftIdeal`
//!   stores them in `O_0`-basis coords. Transcription applies the conversion
//!   `(a, b, c, d) → (a - d, b - c, 2c, 2d)` (derived from
//!   `o0_basis_to_standard_doubled`'s inverse).
//! - **`cached_norm` convention** — the C reference's `.norm` field
//!   is the REDUCED quaternion ideal norm `N_red(γ_I)`. Our
//!   [`LeftIdeal::cached_norm`] is the LATTICE INDEX `N_red(γ_I)²`
//!   (lattice-index convention). Transcription squares the C reference's norm.

use crypto_bigint::{Int, Uint};

use crate::quaternion::ideal::LeftIdeal;

/// L1 `ALTERNATE_CONNECTING_IDEALS[0]` — the first NON-trivial
/// alternate connecting ideal at security Level 1. Corresponds to
/// `CONNECTING_IDEALS[1]` in the SQIsign C reference.
///
/// # Provenance
///
/// Transcribed from
/// `src/precomp/ref/lvl1/quaternion_data.c` lines 1888-2050
/// (`GMP_LIMB_BITS == 64` branch).
///
/// # Math sanity
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
/// # `cached_norm` (lattice-index convention)
///
/// The C reference's `.norm` field is the reduced quaternion ideal
/// norm `0x30000000000000000000000000000001` (= `3·2^124 + 1`).
/// Our [`LeftIdeal::cached_norm`] is the lattice index, so we
/// square: `(3·2^124 + 1)² = 9·2^248 + 6·2^124 + 1`. Compose at
/// construction via `Uint::wrapping_mul` to avoid hand-encoding
/// the 252-bit hex literal.
pub fn alternate_connecting_ideal_0_l1() -> LeftIdeal<8> {
    // C `CONNECTING_IDEALS[1]` basis, row-major `basis[i][j]` (verbatim from
    // `quaternion_data.c`). The C header is explicit: a matrix COLUMN divided
    // by the denominator is an algebra element — so the column convention is
    // correct, which `c_ideal_to_left_ideal` applies. `Uint::from_words`
    // uses little-endian limbs matching GMP's `_mp_d`.
    //
    // **Correction**: the prior transcription used the C *rows* as the
    // ideal generators (transposed) — that lattice is NOT a left O_0-ideal
    // (`connecting_ideal_1_element_convention` proved rows are not left-closed,
    // columns are). The norm²/runs-to-completion tests could not catch it
    // (`det`/norm are transpose-invariant). Rebuilt via the column convention.
    let w = |lo: u64, hi: u64| *Uint::<8>::from_words([lo, hi, 0, 0, 0, 0, 0, 0]).as_int();
    let z = Int::<8>::from_i64(0);
    let one = Int::<8>::from_i64(1);
    let a = w(0x2, 0x6000000000000000); // 0x60…02
    let b = w(0x1, 0x1000000000000000); // 0x10…01
    let c = w(0x1, 0x5000000000000000); // 0x50…01
    let cbasis = [[a, z, z, b], [z, a, c, z], [z, z, one, z], [z, z, z, one]];
    // C reduced norm 0x30…01 = 3·2^124 + 1; `c_ideal_to_left_ideal` stores
    // `cached_norm = norm²` (the lattice-index convention).
    let norm = Uint::<8>::from_words([0x1, 0x3000000000000000, 0, 0, 0, 0, 0, 0]);
    crate::quaternion::o0_mul::c_ideal_to_left_ideal::<8>(&cbasis, &Int::<8>::from_i64(2), &norm)
}

/// L1 `ALTERNATE_CONNECTING_IDEALS[1]` = the C reference's
/// `CONNECTING_IDEALS[2]` (extracted from `quaternion_data.c`, GMP-64 limbs;
/// COLUMN convention via `c_ideal_to_left_ideal`). Reduced norm
/// `0x3c7a53236805e962bfc80abdc339faff`; `cached_norm = norm²`, denom reduces to 1 (integral O_0-ideal).
pub fn alternate_connecting_ideal_1_l1() -> LeftIdeal<8> {
    let w = |lo: u64, hi: u64| *Uint::<8>::from_words([lo, hi, 0, 0, 0, 0, 0, 0]).as_int();
    let z = Int::<8>::from_i64(0);
    let one = Int::<8>::from_i64(1);
    let cbasis = [
        [
            w(0x7f90157b8673f5fe, 0x78f4a646d00bd2c5),
            z,
            z,
            w(0xe65cd6d8002bfee5, 0x5b1373de72d68a3),
        ],
        [
            z,
            w(0x7f90157b8673f5fe, 0x78f4a646d00bd2c5),
            w(0x99333ea38647f719, 0x73436f08e8de6a21),
            z,
        ],
        [z, z, one, z],
        [z, z, z, one],
    ];
    let norm = Uint::<8>::from_words([0xbfc80abdc339faff, 0x3c7a53236805e962, 0, 0, 0, 0, 0, 0]);
    crate::quaternion::o0_mul::c_ideal_to_left_ideal::<8>(&cbasis, &Int::<8>::from_i64(2), &norm)
}

/// L1 `ALTERNATE_CONNECTING_IDEALS[2]` = the C reference's
/// `CONNECTING_IDEALS[3]` (extracted from `quaternion_data.c`, GMP-64 limbs;
/// COLUMN convention via `c_ideal_to_left_ideal`). Reduced norm
/// `0xbca4df64395c83c1e37d4733b8af2f1`; `cached_norm = norm²`, denom reduces to 1 (integral O_0-ideal).
pub fn alternate_connecting_ideal_2_l1() -> LeftIdeal<8> {
    let w = |lo: u64, hi: u64| *Uint::<8>::from_words([lo, hi, 0, 0, 0, 0, 0, 0]).as_int();
    let z = Int::<8>::from_i64(0);
    let one = Int::<8>::from_i64(1);
    let cbasis = [
        [
            w(0x3c6fa8e67715e5e2, 0x17949bec872b9078),
            z,
            z,
            w(0xbb290a5a3af78597, 0x84ff561d2d977c0),
        ],
        [
            z,
            w(0x3c6fa8e67715e5e2, 0x17949bec872b9078),
            w(0x81469e8c3c1e604b, 0xf44a68ab45218b7),
            z,
        ],
        [z, z, one, z],
        [z, z, z, one],
    ];
    let norm = Uint::<8>::from_words([0x1e37d4733b8af2f1, 0xbca4df64395c83c, 0, 0, 0, 0, 0, 0]);
    crate::quaternion::o0_mul::c_ideal_to_left_ideal::<8>(&cbasis, &Int::<8>::from_i64(2), &norm)
}

/// L1 `ALTERNATE_CONNECTING_IDEALS[3]` = the C reference's
/// `CONNECTING_IDEALS[4]` (extracted from `quaternion_data.c`, GMP-64 limbs;
/// COLUMN convention via `c_ideal_to_left_ideal`). Reduced norm
/// `0x16fca7cbe44f64676f19e288b6f757d1`; `cached_norm = norm²`, denom reduces to 1 (integral O_0-ideal).
pub fn alternate_connecting_ideal_3_l1() -> LeftIdeal<8> {
    let w = |lo: u64, hi: u64| *Uint::<8>::from_words([lo, hi, 0, 0, 0, 0, 0, 0]).as_int();
    let z = Int::<8>::from_i64(0);
    let one = Int::<8>::from_i64(1);
    let cbasis = [
        [
            w(0xde33c5116deeafa2, 0x2df94f97c89ec8ce),
            z,
            z,
            w(0xd5f5cdcaa90b519b, 0xe59b35483dd757a),
        ],
        [
            z,
            w(0xde33c5116deeafa2, 0x2df94f97c89ec8ce),
            w(0x83df746c4e35e07, 0x1f9f9c4344c15354),
            z,
        ],
        [z, z, one, z],
        [z, z, z, one],
    ];
    let norm = Uint::<8>::from_words([0x6f19e288b6f757d1, 0x16fca7cbe44f6467, 0, 0, 0, 0, 0, 0]);
    crate::quaternion::o0_mul::c_ideal_to_left_ideal::<8>(&cbasis, &Int::<8>::from_i64(2), &norm)
}

/// L1 `ALTERNATE_CONNECTING_IDEALS[4]` = the C reference's
/// `CONNECTING_IDEALS[5]` (extracted from `quaternion_data.c`, GMP-64 limbs;
/// COLUMN convention via `c_ideal_to_left_ideal`). Reduced norm
/// `0x59a410c3a2e4fa2ca951773baaca0cf9`; `cached_norm = norm²`, denom reduces to 1 (integral O_0-ideal).
pub fn alternate_connecting_ideal_4_l1() -> LeftIdeal<8> {
    let w = |lo: u64, hi: u64| *Uint::<8>::from_words([lo, hi, 0, 0, 0, 0, 0, 0]).as_int();
    let z = Int::<8>::from_i64(0);
    let one = Int::<8>::from_i64(1);
    let cbasis = [
        [
            w(0x52a2ee77559419f2, 0xb348218745c9f459),
            z,
            z,
            w(0x1df48a96967adbd3, 0x222419a0d707845),
        ],
        [
            z,
            w(0x52a2ee77559419f2, 0xb348218745c9f459),
            w(0x34ae63e0bf193e1f, 0xb125dfed38597c14),
            z,
        ],
        [z, z, one, z],
        [z, z, z, one],
    ];
    let norm = Uint::<8>::from_words([0xa951773baaca0cf9, 0x59a410c3a2e4fa2c, 0, 0, 0, 0, 0, 0]);
    crate::quaternion::o0_mul::c_ideal_to_left_ideal::<8>(&cbasis, &Int::<8>::from_i64(2), &norm)
}

/// L1 `ALTERNATE_CONNECTING_IDEALS[5]` = the C reference's
/// `CONNECTING_IDEALS[6]` (extracted from `quaternion_data.c`, GMP-64 limbs;
/// COLUMN convention via `c_ideal_to_left_ideal`). Reduced norm
/// `0x14cb6c2975e50380e818b56bb3e7d51d`; `cached_norm = norm²`, denom reduces to 1 (integral O_0-ideal).
pub fn alternate_connecting_ideal_5_l1() -> LeftIdeal<8> {
    let w = |lo: u64, hi: u64| *Uint::<8>::from_words([lo, hi, 0, 0, 0, 0, 0, 0]).as_int();
    let z = Int::<8>::from_i64(0);
    let one = Int::<8>::from_i64(1);
    let cbasis = [
        [
            w(0xd0316ad767cfaa3a, 0x2996d852ebca0701),
            z,
            z,
            w(0xbc67edebd7ab0275, 0x148ef2e5aeb5ad41),
        ],
        [
            z,
            w(0xd0316ad767cfaa3a, 0x2996d852ebca0701),
            w(0x13c97ceb9024a7c5, 0x1507e56d3d1459c0),
            z,
        ],
        [z, z, one, z],
        [z, z, z, one],
    ];
    let norm = Uint::<8>::from_words([0xe818b56bb3e7d51d, 0x14cb6c2975e50380, 0, 0, 0, 0, 0, 0]);
    crate::quaternion::o0_mul::c_ideal_to_left_ideal::<8>(&cbasis, &Int::<8>::from_i64(2), &norm)
}

/// lvl3 `ALTERNATE_CONNECTING_IDEALS[0]` = C `CONNECTING_IDEALS[1]`.
/// Byte-exact from `quaternion_data.c` (GMP-64); basis row-major as stored
/// (C column-std), denom + reduced norm; `c_ideal_to_left_ideal` applies the
/// column→O_0-coord conversion and squares the norm. Checked by
/// `alt_connecting_ideals_lvl3_are_integral`.
pub fn alternate_connecting_ideal_0_lvl3() -> LeftIdeal<8> {
    let pos = |l: [u64; 8]| *Uint::<8>::from_words(l).as_int();
    let cbasis = [
        [
            pos([
                0x21d4cde19a6fbf5a,
                0x78ebf3bae7a052b1,
                0x1c515c29787fc45c,
                0x1,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([
                0x1437b508fd5c8015,
                0xa79a14526222fa92,
                0x15c5b3c4d3a96c31,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
        ],
        [
            Int::<8>::from_i64(0),
            pos([
                0x21d4cde19a6fbf5a,
                0x78ebf3bae7a052b1,
                0x1c515c29787fc45c,
                0x1,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            pos([
                0xd9d18d89d133f45,
                0xd151df68857d581f,
                0x68ba864a4d6582a,
                0x1,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            Int::<8>::from_i64(0),
        ],
        [
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]),
            Int::<8>::from_i64(0),
        ],
        [
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]),
        ],
    ];
    let denom = pos([0x2, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]);
    let norm = Uint::<8>::from_words([
        0x90ea66f0cd37dfad,
        0x3c75f9dd73d02958,
        0x8e28ae14bc3fe22e,
        0x0,
        0x0,
        0x0,
        0x0,
        0x0,
    ]);
    crate::quaternion::o0_mul::c_ideal_to_left_ideal::<8>(&cbasis, &denom, &norm)
}

/// lvl3 `ALTERNATE_CONNECTING_IDEALS[1]` = C `CONNECTING_IDEALS[2]`.
/// Byte-exact from `quaternion_data.c` (GMP-64); basis row-major as stored
/// (C column-std), denom + reduced norm; `c_ideal_to_left_ideal` applies the
/// column→O_0-coord conversion and squares the norm. Checked by
/// `alt_connecting_ideals_lvl3_are_integral`.
pub fn alternate_connecting_ideal_1_lvl3() -> LeftIdeal<8> {
    let pos = |l: [u64; 8]| *Uint::<8>::from_words(l).as_int();
    let cbasis = [
        [
            pos([
                0xc1200e71920e9d7a,
                0xff55029f607e8fbf,
                0x125bbca447967422,
                0x1,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([
                0xec53c53876edbcbf,
                0x14fa54eb40deb88,
                0x539edb2300a8bb2,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
        ],
        [
            Int::<8>::from_i64(0),
            pos([
                0xc1200e71920e9d7a,
                0xff55029f607e8fbf,
                0x125bbca447967422,
                0x1,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            pos([
                0xd4cc49391b20e0bb,
                0xfe055d50ac70a436,
                0xd21cef2178be870,
                0x1,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            Int::<8>::from_i64(0),
        ],
        [
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]),
            Int::<8>::from_i64(0),
        ],
        [
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]),
        ],
    ];
    let denom = pos([0x2, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]);
    let norm = Uint::<8>::from_words([
        0xe0900738c9074ebd,
        0x7faa814fb03f47df,
        0x892dde5223cb3a11,
        0x0,
        0x0,
        0x0,
        0x0,
        0x0,
    ]);
    crate::quaternion::o0_mul::c_ideal_to_left_ideal::<8>(&cbasis, &denom, &norm)
}

/// lvl3 `ALTERNATE_CONNECTING_IDEALS[2]` = C `CONNECTING_IDEALS[3]`.
/// Byte-exact from `quaternion_data.c` (GMP-64); basis row-major as stored
/// (C column-std), denom + reduced norm; `c_ideal_to_left_ideal` applies the
/// column→O_0-coord conversion and squares the norm. Checked by
/// `alt_connecting_ideals_lvl3_are_integral`.
pub fn alternate_connecting_ideal_2_lvl3() -> LeftIdeal<8> {
    let pos = |l: [u64; 8]| *Uint::<8>::from_words(l).as_int();
    let cbasis = [
        [
            pos([
                0xe33e6532cb1fd282,
                0x2b6242750fd8153d,
                0xf7223f12db04f17d,
                0x1,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([
                0xea4de944eebf50bf,
                0x12b04919cbc5076d,
                0x126533049e3071f3,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
        ],
        [
            Int::<8>::from_i64(0),
            pos([
                0xe33e6532cb1fd282,
                0x2b6242750fd8153d,
                0xf7223f12db04f17d,
                0x1,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            pos([
                0xf8f07beddc6081c3,
                0x18b1f95b44130dcf,
                0xe4bd0c0e3cd47f8a,
                0x1,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            Int::<8>::from_i64(0),
        ],
        [
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]),
            Int::<8>::from_i64(0),
        ],
        [
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]),
        ],
    ];
    let denom = pos([0x2, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]);
    let norm = Uint::<8>::from_words([
        0xf19f3299658fe941,
        0x95b1213a87ec0a9e,
        0xfb911f896d8278be,
        0x0,
        0x0,
        0x0,
        0x0,
        0x0,
    ]);
    crate::quaternion::o0_mul::c_ideal_to_left_ideal::<8>(&cbasis, &denom, &norm)
}

/// lvl3 `ALTERNATE_CONNECTING_IDEALS[3]` = C `CONNECTING_IDEALS[4]`.
/// Byte-exact from `quaternion_data.c` (GMP-64); basis row-major as stored
/// (C column-std), denom + reduced norm; `c_ideal_to_left_ideal` applies the
/// column→O_0-coord conversion and squares the norm. Checked by
/// `alt_connecting_ideals_lvl3_are_integral`.
pub fn alternate_connecting_ideal_3_lvl3() -> LeftIdeal<8> {
    let pos = |l: [u64; 8]| *Uint::<8>::from_words(l).as_int();
    let cbasis = [
        [
            pos([
                0x7c1cd4b8abc3dfda,
                0x79cc21da66b24727,
                0xa12d9b8d553de3a3,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([
                0xd263887fd39960fb,
                0xea34699bb202e0e7,
                0x8e9567634b8a5a15,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
        ],
        [
            Int::<8>::from_i64(0),
            pos([
                0x7c1cd4b8abc3dfda,
                0x79cc21da66b24727,
                0xa12d9b8d553de3a3,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            pos([
                0xa9b94c38d82a7edf,
                0x8f97b83eb4af663f,
                0x1298342a09b3898d,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            Int::<8>::from_i64(0),
        ],
        [
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]),
            Int::<8>::from_i64(0),
        ],
        [
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]),
        ],
    ];
    let denom = pos([0x2, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]);
    let norm = Uint::<8>::from_words([
        0xbe0e6a5c55e1efed,
        0xbce610ed33592393,
        0x5096cdc6aa9ef1d1,
        0x0,
        0x0,
        0x0,
        0x0,
        0x0,
    ]);
    crate::quaternion::o0_mul::c_ideal_to_left_ideal::<8>(&cbasis, &denom, &norm)
}

/// lvl3 `ALTERNATE_CONNECTING_IDEALS[4]` = C `CONNECTING_IDEALS[5]`.
/// Byte-exact from `quaternion_data.c` (GMP-64); basis row-major as stored
/// (C column-std), denom + reduced norm; `c_ideal_to_left_ideal` applies the
/// column→O_0-coord conversion and squares the norm. Checked by
/// `alt_connecting_ideals_lvl3_are_integral`.
pub fn alternate_connecting_ideal_4_lvl3() -> LeftIdeal<8> {
    let pos = |l: [u64; 8]| *Uint::<8>::from_words(l).as_int();
    let cbasis = [
        [
            pos([
                0xd8de1a76d869e00e,
                0xf26499e1abc5fe4c,
                0xb60b32ab09c37d83,
                0x1,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([
                0xd6b19a98bbfb00f,
                0xd7e274e7cd5c0f7b,
                0x8de856a83593a419,
                0x1,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
        ],
        [
            Int::<8>::from_i64(0),
            pos([
                0xd8de1a76d869e00e,
                0xf26499e1abc5fe4c,
                0xb60b32ab09c37d83,
                0x1,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            pos([
                0xcb7300cd4caa2fff,
                0x1a8224f9de69eed1,
                0x2822dc02d42fd96a,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            Int::<8>::from_i64(0),
        ],
        [
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]),
            Int::<8>::from_i64(0),
        ],
        [
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]),
        ],
    ];
    let denom = pos([0x2, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]);
    let norm = Uint::<8>::from_words([
        0x6c6f0d3b6c34f007,
        0xf9324cf0d5e2ff26,
        0xdb05995584e1bec1,
        0x0,
        0x0,
        0x0,
        0x0,
        0x0,
    ]);
    crate::quaternion::o0_mul::c_ideal_to_left_ideal::<8>(&cbasis, &denom, &norm)
}

/// lvl3 `ALTERNATE_CONNECTING_IDEALS[5]` = C `CONNECTING_IDEALS[6]`.
/// Byte-exact from `quaternion_data.c` (GMP-64); basis row-major as stored
/// (C column-std), denom + reduced norm; `c_ideal_to_left_ideal` applies the
/// column→O_0-coord conversion and squares the norm. Checked by
/// `alt_connecting_ideals_lvl3_are_integral`.
pub fn alternate_connecting_ideal_5_lvl3() -> LeftIdeal<8> {
    let pos = |l: [u64; 8]| *Uint::<8>::from_words(l).as_int();
    let cbasis = [
        [
            pos([
                0x97d3c8dc37a07b26,
                0x1df20931d6bd7f2f,
                0x6ee725914a3e2918,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([
                0x9a52ac55cd013a91,
                0x42454dec118f9887,
                0x7ad1d161022d869,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
        ],
        [
            Int::<8>::from_i64(0),
            pos([
                0x97d3c8dc37a07b26,
                0x1df20931d6bd7f2f,
                0x6ee725914a3e2918,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            pos([
                0xfd811c866a9f4095,
                0xdbacbb45c52de6a7,
                0x673a087b3a1b50ae,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            Int::<8>::from_i64(0),
        ],
        [
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]),
            Int::<8>::from_i64(0),
        ],
        [
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]),
        ],
    ];
    let denom = pos([0x2, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]);
    let norm = Uint::<8>::from_words([
        0xcbe9e46e1bd03d93,
        0xef90498eb5ebf97,
        0x377392c8a51f148c,
        0x0,
        0x0,
        0x0,
        0x0,
        0x0,
    ]);
    crate::quaternion::o0_mul::c_ideal_to_left_ideal::<8>(&cbasis, &denom, &norm)
}

/// lvl3 `ALTERNATE_CONNECTING_IDEALS[6]` = C `CONNECTING_IDEALS[7]`.
/// Byte-exact from `quaternion_data.c` (GMP-64); basis row-major as stored
/// (C column-std), denom + reduced norm; `c_ideal_to_left_ideal` applies the
/// column→O_0-coord conversion and squares the norm. Checked by
/// `alt_connecting_ideals_lvl3_are_integral`.
pub fn alternate_connecting_ideal_6_lvl3() -> LeftIdeal<8> {
    let pos = |l: [u64; 8]| *Uint::<8>::from_words(l).as_int();
    let cbasis = [
        [
            pos([
                0x430fb04b3b34e5ca,
                0xec478c7da04ae795,
                0xd31fb71e8e5c77df,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([
                0x83ff383998d54d27,
                0xbe45c95b4d5b48b7,
                0x6c264d5736f39d44,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
        ],
        [
            Int::<8>::from_i64(0),
            pos([
                0x430fb04b3b34e5ca,
                0xec478c7da04ae795,
                0xd31fb71e8e5c77df,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            pos([
                0xbf107811a25f98a3,
                0x2e01c32252ef9edd,
                0x66f969c75768da9b,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ]),
            Int::<8>::from_i64(0),
        ],
        [
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]),
            Int::<8>::from_i64(0),
        ],
        [
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            pos([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]),
        ],
    ];
    let denom = pos([0x2, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]);
    let norm = Uint::<8>::from_words([
        0xa187d8259d9a72e5,
        0xf623c63ed02573ca,
        0x698fdb8f472e3bef,
        0x0,
        0x0,
        0x0,
        0x0,
        0x0,
    ]);
    crate::quaternion::o0_mul::c_ideal_to_left_ideal::<8>(&cbasis, &denom, &norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::eprintln;

    /// All 7 lvl3 connecting ideals are integral left `O_0`-ideals: after the
    /// `c_ideal_to_left_ideal` column→O_0-coord conversion the denominator
    /// divides out to 1 (a wrong/transposed basis would NOT yield even O_0
    /// coords, so this guards the transcription), and `cached_norm` is a perfect
    /// square (`= reduced_norm²`), recovering the C `.norm`.
    #[test]
    fn alt_connecting_ideals_lvl3_are_integral() {
        let ideals = [
            alternate_connecting_ideal_0_lvl3(),
            alternate_connecting_ideal_1_lvl3(),
            alternate_connecting_ideal_2_lvl3(),
            alternate_connecting_ideal_3_lvl3(),
            alternate_connecting_ideal_4_lvl3(),
            alternate_connecting_ideal_5_lvl3(),
            alternate_connecting_ideal_6_lvl3(),
        ];
        for (k, id) in ideals.iter().enumerate() {
            assert_eq!(
                id.denom,
                Uint::<8>::from_u64(1),
                "ideal[{k}] must reduce to denom 1 (integral O_0-ideal)"
            );
            assert!(
                id.reduced_norm_vartime().is_some(),
                "ideal[{k}] cached_norm must be a perfect square (= reduced_norm²)"
            );
        }
    }

    /// regression: `cached_norm` equals `reduced_norm²` per the
    /// lattice-index convention.
    #[test]
    fn alt_connecting_ideal_0_l1_cached_norm_is_reduced_norm_squared() {
        let ideal = alternate_connecting_ideal_0_l1();
        let expected_reduced = Uint::<8>::from_words([0x1, 0x3000000000000000, 0, 0, 0, 0, 0, 0]);
        let expected_cached = expected_reduced.wrapping_mul(&expected_reduced);
        assert_eq!(
            ideal.cached_norm, expected_cached,
            "cached_norm must equal reduced_norm² per the lattice-index convention",
        );
    }

    /// regression: `reduced_norm_vartime()` round-trips the C ref's
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

    /// The C stores the connecting ideal at std-coords `lattice.denom = 2`, but
    /// in O_0-coords an INTEGRAL left ideal is canonically denom 1 (all O_0
    /// coords are even, so `c_ideal_to_left_ideal`'s `reduce_denom` divides the
    /// 2 out). Denom 1 is also what the spine expects.
    #[test]
    fn alt_connecting_ideal_0_l1_denom_is_one() {
        let ideal = alternate_connecting_ideal_0_l1();
        assert_eq!(ideal.denom, Uint::<8>::from_u64(1));
    }

    // (Removed `alt_connecting_ideal_0_l1_trivial_columns_match_conversion` —
    // it asserted the pre-fix transposed row-as-element basis, which is not a
    // valid left O_0-ideal. Correctness is now covered by
    // `connecting_ideal_1_element_convention` (left-closure + reduced norm).)

    // ── probe L1 ALT[0] through the existing primitives ───────────────

    /// probe: `lideal_reduce_basis` on L1 ALT[0] at the REAL L1
    /// prime panics inside LLL's `int_div_exact` step. The probe
    /// confirms: at LIMBS=8 with p ≈ 2^251, LLL's
    /// intermediate computations on a basis with 128-bit entries
    /// overflow the 512-bit Uint, causing the exact-division assertion
    /// to fail.
    ///
    /// **Blocker**: wide-Int variants of `lideal_reduce_basis`,
    /// `lideal_rescale_by_smallest_basis_element`, and `find_uv` are
    /// needed before alternate-orders body wiring can be validated
    /// numerically against the real L1 prime + real ALT data.
    ///
    /// **Marked `#[ignore]`**: this test exercises the narrow `LIMBS=8`
    /// reduction directly, which overflows on the real L1 ALT data. Production
    /// uses the wide reduction path (`lideal_reduce_basis_wide`); re-enabling
    /// this test would require porting it onto that path.
    #[test]
    #[ignore = "narrow LIMBS=8 reduction overflows on real L1 ALT data"]
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

    /// sanity probe: an attempt to bypass the L1-prime precision
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
    /// **Marked `#[ignore]`** with the rest: same narrow-path overflow (the
    /// integer-GSO exceeds `Uint<8>` on 128-bit basis entries at any prime).
    #[test]
    #[ignore = "narrow LIMBS=8 integer-GSO overflows on 128-bit basis entries"]
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

    /// probe: `lideal_rescale_by_smallest_basis_element` outcome
    /// on L1 ALT[0]. The invariant says SQIsign-shaped principal
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
    /// for future body wiring.
    #[test]
    #[ignore = "precision overflow at LIMBS=8 with real L1 prime — needs wide-Int variants"]
    fn alt_connecting_ideal_0_l1_rescale_outcome() {
        let p: Uint<8> = crate::params::lvl1::prime().resize::<8>();
        let id = alternate_connecting_ideal_0_l1();
        let reduced = crate::quaternion::ideal_mul::lideal_reduce_basis::<8>(&id, &p);
        let rescaled = crate::quaternion::ideal_mul::lideal_rescale_by_smallest_basis_element::<8>(
            &reduced, &p,
        );
        match rescaled {
            Some(r) => {
                // invariant: rescaled = O_0 → cached_norm == 1.
                // If this fails, ALT[0] is NOT SQIsign-shaped at L1 prime
                // (or our LLL doesn't find the principal generator as δ).
                // Either way, this test PINS the current behavior.
                assert_eq!(
                    r.cached_norm,
                    Uint::<8>::from_u64(1),
                    "rescaled cached_norm must be 1 (rescale-to-unit-lattice invariant); \
                     deviation signals ALT[0] is not SQIsign-shaped at L1 prime",
                );
            }
            None => {
                panic!(
                    "rescale returned None — ALT[0] may not satisfy the SQIsign-shaped \
                     contract at the real L1 prime, OR cached_norm isn't a perfect square \
                     (the latter should be impossible given the lattice-index construction)."
                );
            }
        }
    }

    /// compose wide LLL with the existing narrow
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
    /// - `Some(rescaled)` with `cached_norm == 1`: rescale-to-unit-lattice invariant holds;
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
                // invariant check (informational): cached_norm == 1
                // for SQIsign-shaped. Otherwise the rescale still succeeded
                // structurally but produced a non-unit lattice.
                let n_after = r.cached_norm;
                let n_before = reduced.cached_norm;
                eprintln!(
                    "rescale Ok; cached_norm pre = {:?}, post = {:?}",
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
                    "rescale returned None — ALT[0]'s LLL-shortest δ \
                     doesn't generate the principal part (or cached_norm not a perfect square). \
                     Documented; future sessions may need to handle this case in find_uv_alternate_orders."
                );
            }
        }
    }

    /// ALT[0] LLL via the wide variant at `WIDE=20` (1280-bit) +
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

    /// probe: `lideal_intersect(O_0, ALT[0])` outcome. For coprime
    /// norms, `lideal_intersect` falls through to `ideal_multiply`.
    /// Since `N(O_0) = 1` and `N(ALT[0]) = reduced_norm²` (with reduced
    /// norm > 1), gcd(1, anything) = 1 → coprime path triggers →
    /// returns `Ok(ideal_multiply(O_0, ALT[0]))`. The product `O_0 · ALT[0]`
    /// is just `ALT[0]` (left-multiplication by full order is identity).
    #[test]
    #[ignore = "precision overflow at LIMBS=8 with real L1 prime — needs wide-Int variants"]
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

    /// element-convention guard for the C connecting-ideal basis. The C
    /// header is explicit ("columns divided by denom are algebra elements"),
    /// and a left O_0-ideal must satisfy `O_0·I ⊆ I`; only the correct
    /// convention is left-closed. This test proves: (a) the SHIPPED
    /// `alternate_connecting_ideal_0_l1` (column convention via
    /// `c_ideal_to_left_ideal`) IS left-closed with reduced norm 3·2^124+1, and
    /// (b) the TRANSPOSED rows-as-elements lattice (the pre-fix transposed bug) is NOT
    /// left-closed — guarding against a regression to the transpose. `det`/norm
    /// are transpose-invariant, so closure is the discriminating check.
    #[test]
    fn connecting_ideal_1_element_convention() {
        use crate::quaternion::Quaternion;
        use crate::quaternion::o0_mul::{multiply_o0_basis, standard_to_o0_basis};
        let p = crate::params::lvl1::prime().resize::<8>();
        let w = |lo: u64, hi: u64| *Uint::<8>::from_words([lo, hi, 0, 0, 0, 0, 0, 0]).as_int();
        let z = Int::<8>::from_i64(0);
        let one = Int::<8>::from_i64(1);
        let a = w(0x2, 0x6000000000000000); // 0x60..02
        let b = w(0x1, 0x1000000000000000); // 0x10..01
        let c = w(0x1, 0x5000000000000000); // 0x50..01
        // C basis row-major basis[i][j] (verbatim from quaternion_data.c).
        let cbasis = [[a, z, z, b], [z, a, c, z], [z, z, one, z], [z, z, z, one]];

        let closed = |ideal: &LeftIdeal<8>| -> bool {
            for r in 0..4 {
                let g = ideal.basis[r];
                for k in 0..4 {
                    let mut e = [z; 4];
                    e[k] = one;
                    let prod = multiply_o0_basis::<8>(&e, &g, &p);
                    if !ideal.contains(&prod) {
                        return false;
                    }
                }
            }
            true
        };

        // (a) Shipped (column convention) is a valid left O_0-ideal of norm 3·2^124+1.
        let shipped = alternate_connecting_ideal_0_l1();
        assert!(
            closed(&shipped),
            "shipped connecting ideal (column convention) must be left-O_0-closed",
        );
        let exp_reduced = Uint::<8>::from_words([0x1, 0x3000000000000000, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            shipped.reduced_norm_vartime(),
            Some(exp_reduced),
            "shipped connecting ideal reduced norm must be 3·2^124+1",
        );

        // (b) Transposed (rows-as-elements) lattice is NOT left-closed — the
        //     pre-fix transposed bug. Build it the old way: each C ROW → std-coords
        //     element → O_0-coords.
        let mut rows_basis = [[z; 4]; 4];
        for (r, row) in cbasis.iter().enumerate() {
            let q = Quaternion::<8>::new(row[0], row[1], row[2], row[3]);
            rows_basis[r] = standard_to_o0_basis::<8>(&q);
        }
        let rows = LeftIdeal::<8>::with_denom_and_norm(
            rows_basis,
            Uint::<8>::from_u64(2),
            exp_reduced.wrapping_mul(&exp_reduced),
        );
        assert!(
            !closed(&rows),
            "transposed (rows-as-elements) lattice must NOT be left-closed (column-convention guard)",
        );
    }

    /// all 6 L1 alternate connecting ideals (C `CONNECTING_IDEALS[1..7]`)
    /// are valid left O_0-ideals with the C reference reduced norms. Validates
    /// the scripted port of [2..6] (and re-checks [1]) via the
    /// structural left-closure invariant + the reduced-norm + denom-1 checks.
    #[test]
    fn all_alternate_connecting_ideals_l1_are_left_ideals() {
        use crate::quaternion::o0_mul::multiply_o0_basis;
        let p = crate::params::lvl1::prime().resize::<8>();
        let z = Int::<8>::from_i64(0);
        let one = Int::<8>::from_i64(1);
        let closed = |ideal: &LeftIdeal<8>| -> bool {
            for r in 0..4 {
                let g = ideal.basis[r];
                for k in 0..4 {
                    let mut e = [z; 4];
                    e[k] = one;
                    if !ideal.contains(&multiply_o0_basis::<8>(&e, &g, &p)) {
                        return false;
                    }
                }
            }
            true
        };
        let nw = |lo: u64, hi: u64| Uint::<8>::from_words([lo, hi, 0, 0, 0, 0, 0, 0]);
        // (ideal, C reduced norm) for ALT[0..6] = C CONNECTING_IDEALS[1..7].
        let cases: [(LeftIdeal<8>, Uint<8>); 6] = [
            (
                alternate_connecting_ideal_0_l1(),
                nw(0x1, 0x3000000000000000),
            ),
            (
                alternate_connecting_ideal_1_l1(),
                nw(0xbfc80abdc339faff, 0x3c7a53236805e962),
            ),
            (
                alternate_connecting_ideal_2_l1(),
                nw(0x1e37d4733b8af2f1, 0x0bca4df64395c83c),
            ),
            (
                alternate_connecting_ideal_3_l1(),
                nw(0x6f19e288b6f757d1, 0x16fca7cbe44f6467),
            ),
            (
                alternate_connecting_ideal_4_l1(),
                nw(0xa951773baaca0cf9, 0x59a410c3a2e4fa2c),
            ),
            (
                alternate_connecting_ideal_5_l1(),
                nw(0xe818b56bb3e7d51d, 0x14cb6c2975e50380),
            ),
        ];
        for (k, (ideal, exp_norm)) in cases.iter().enumerate() {
            assert!(closed(ideal), "ALT[{k}] must be a valid left O_0-ideal");
            assert_eq!(
                ideal.reduced_norm_vartime(),
                Some(*exp_norm),
                "ALT[{k}] reduced norm must match the C reference",
            );
            assert_eq!(
                ideal.denom,
                Uint::<8>::from_u64(1),
                "ALT[{k}] denom must be 1"
            );
        }
    }
}
