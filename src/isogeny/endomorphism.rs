//! Precomputed E0 endomorphism data (the quaternion→EC-torsion bridge).
//!
//! Holds the canonical even-torsion basis on E0 and (subsequently) the action
//! matrices `ACTION_GEN2/3/4` that describe how the O0 generators act on it.
//! These are VERBATIM ports of the SQIsign reference precomputed tables
//! (`src/precomp/ref/lvl1/endomorphism_action.c`,
//! `CURVES_WITH_ENDOMORPHISMS[0]`); the action tables are meaningful only
//! relative to THIS exact basis, so neither may be regenerated independently.
//!
//! The reference stores field elements in Montgomery form with `R = 2^256`
//! (its 4-limb "BROADWELL" representation). Our `Fp1Element` is
//! `crypto_bigint::ConstMontyForm<Lvl1Modulus, 4>`, which uses the SAME
//! `R = 2^256` and limb layout — verified against the reference's stored
//! `C = 1` constant `{0x33, 0, 0, 0x0100000000000000}` (= `2^256 mod p`). So
//! the reference limbs ARE our internal representation: we plug them straight
//! in via `ConstMontyForm::from_montgomery`, no conversion.

use crate::ec::montgomery::MontgomeryPoint;
use crate::gf::fp::BaseField;
use crate::gf::fp2::Fp2;
use crate::level_constants::LevelConstants;
use crate::params::lvl1::Fp1Element;
use crate::params::lvl3::Fp3Element;
use crypto_bigint::{U256, U384, Uint};

/// The `(R, S, R−S)` x-only torsion-point triple returned by an endomorphism
/// application over the field `F`. Factored out to keep the application
/// signatures within clippy's type-complexity budget.
type EndoImageTriple<F> = (MontgomeryPoint<F>, MontgomeryPoint<F>, MontgomeryPoint<F>);

/// An `Fp` element from the reference's 4-limb Montgomery (`R = 2^256`) words.
#[inline]
fn fp_mont(limbs: [u64; 4]) -> Fp1Element {
    Fp1Element::from_montgomery(U256::from_words(limbs))
}

/// An `Fp2 = re + im·i` from two Montgomery-form limb arrays.
#[inline]
fn fp2_mont(re: [u64; 4], im: [u64; 4]) -> Fp2<Fp1Element> {
    Fp2::new(fp_mont(re), fp_mont(im))
}

/// The canonical `E0[2^248]` x-only torsion basis `(P, Q, P−Q)` at level 1.
///
/// VERBATIM from `CURVES_WITH_ENDOMORPHISMS[0].basis_even`
/// (endomorphism_action.c) — also exposed there as `BASIS_E0_PX`/`BASIS_E0_QX`
/// (e0_basis.c). All three points are affine (`z = 1`).
pub(crate) fn basis_e0_lvl1() -> (
    MontgomeryPoint<Fp1Element>,
    MontgomeryPoint<Fp1Element>,
    MontgomeryPoint<Fp1Element>,
) {
    let one = Fp2::<Fp1Element>::one();

    let px = fp2_mont(
        [
            0x052b_7956_2400_1810,
            0x8c85_0545_2654_b56d,
            0xf59a_8d87_ad37_c0da,
            0x024e_4cc2_1a23_6db3,
        ],
        [
            0xcd9d_72c0_cb90_7df8,
            0x5cc5_efa9_da1d_4a82,
            0x6a9b_bbb8_14c8_3cbd,
            0x026e_f8a8_622c_da10,
        ],
    );
    let qx = fp2_mont(
        [
            0x6ad4_3baa_b72f_065f,
            0xe7b1_cb21_0f2d_30b2,
            0xc63b_049a_9c34_05e7,
            0x04ff_74b2_ac02_49ec,
        ],
        [
            0x606e_8b20_2922_2fc7,
            0x6634_4249_82ed_efcc,
            0xe121_cbd7_b157_1ed8,
            0x04f7_61f9_6b4a_5f40,
        ],
    );
    let pmqx = fp2_mont(
        [
            0x57f1_ec00_3b5f_6e34,
            0x07b9_3675_cb70_9894,
            0x809f_9560_5ef9_5589,
            0x00c9_05fc_4929_3f44,
        ],
        [
            0x7f3c_7473_5e3a_42f3,
            0xc1b3_9b1c_2884_b023,
            0x95bb_d10f_473a_cbcf,
            0x03c4_667f_4316_e477,
        ],
    );

    (
        MontgomeryPoint::new(px, one),
        MontgomeryPoint::new(qx, one),
        MontgomeryPoint::new(pmqx, one),
    )
}

// NOTE: the lvl3 constant providers below are staged data, transcribed ahead of
// the generic-spine work that will consume them. They are exposed as `pub` so
// they read as public API (reachable) until that lift wires them internally.

/// An `Fp` element from the reference's 6-limb Montgomery (`R = 2^384`) words (lvl3).
#[inline]
fn fp_mont3(limbs: [u64; 6]) -> Fp3Element {
    Fp3Element::from_montgomery(U384::from_words(limbs))
}

/// An `Fp2 = re + im·i` (lvl3) from two Montgomery-form limb arrays.
#[inline]
fn fp2_mont3(re: [u64; 6], im: [u64; 6]) -> Fp2<Fp3Element> {
    Fp2::new(fp_mont3(re), fp_mont3(im))
}

/// The canonical `E0[2^376]` x-only torsion basis `(P, Q, P−Q)` at level 3.
///
/// VERBATIM from `CURVES_WITH_ENDOMORPHISMS[0].basis_even`
/// (`src/precomp/ref/lvl3/endomorphism_action.c`); PX/QX are also exposed as
/// `BASIS_E0_PX`/`BASIS_E0_QX` in `e0_basis.c` (cross-checked equal). All three
/// points are affine (`z = 1`). The reference stores them in 6-limb BROADWELL
/// Montgomery form (`R = 2^384`), which equals our `Fp3Element` internal
/// representation — verified by
/// `params::lvl3::tests::montgomery_repr_matches_c_broadwell` — so the limbs are
/// plugged straight in via `ConstMontyForm::from_montgomery`.
pub fn basis_e0_lvl3() -> (
    MontgomeryPoint<Fp3Element>,
    MontgomeryPoint<Fp3Element>,
    MontgomeryPoint<Fp3Element>,
) {
    let one = Fp2::<Fp3Element>::one();

    let px = fp2_mont3(
        [
            0x31c4a31adbd9a5c6,
            0xe7ad90c51d65d7b2,
            0x88ba021701e76d61,
            0x2cb3cdb2a2e90ddd,
            0xdc1b70072d06f585,
            0x16eecbda94894ad1,
        ],
        [
            0xf42096161ef8662a,
            0xcba5e8ce200d142d,
            0x2205c5d40d107d81,
            0xd00330eccc07a7e7,
            0x16d8d4adf934c3fa,
            0x6065815b3283164,
        ],
    );
    let qx = fp2_mont3(
        [
            0x6f999f727a40c5b,
            0x50a8ca71cebbf1da,
            0x65cc12a7b6e85c42,
            0x9151a12f13f8774b,
            0x8678d0d647499967,
            0x2e23bfb6dd51ff28,
        ],
        [
            0x6bcfee41588c1c62,
            0xa9249a07cd644dfe,
            0xef21e097d60b5ff8,
            0xcecabfeab509e310,
            0xf010f836ce26d4bd,
            0x2a7787556e853bb9,
        ],
    );
    let pmqx = fp2_mont3(
        [
            0x1a0f70dccedb8c78,
            0x7dec6534b94f5bd1,
            0xe508bd760193eeb6,
            0x10bf4b1c0497322f,
            0x2d7e909753d8633c,
            0x3722113986808eb1,
        ],
        [
            0x6a46366e4b4b295e,
            0xa5a183bada734009,
            0x8609a1279ac3fe52,
            0x269b74a0c7e6f7c5,
            0x9cf14a7d5c5199bd,
            0x5ce1f24843721b0,
        ],
    );

    (
        MontgomeryPoint::new(px, one),
        MontgomeryPoint::new(qx, one),
        MontgomeryPoint::new(pmqx, one),
    )
}

#[cfg(test)]
mod lvl3_basis_tests {
    use super::*;

    #[test]
    fn basis_e0_lvl3_is_well_formed() {
        let (p, q, pmq) = basis_e0_lvl3();
        let one = Fp2::<Fp3Element>::one();

        // All three points are affine (z normalized to 1) — matches the C
        // reference storing each basis point with Z = Montgomery-1.
        assert_eq!(p.z, one, "P.z must be 1");
        assert_eq!(q.z, one, "Q.z must be 1");
        assert_eq!(pmq.z, one, "(P-Q).z must be 1");

        // x-coords nonzero and pairwise distinct — catches a dropped limb,
        // a duplicated point, or two points swapped during transcription.
        assert!(!bool::from(p.x.is_zero()), "P.x nonzero");
        assert!(!bool::from(q.x.is_zero()), "Q.x nonzero");
        assert!(!bool::from(pmq.x.is_zero()), "(P-Q).x nonzero");
        assert_ne!(p.x, q.x, "P != Q");
        assert_ne!(p.x, pmq.x, "P != P-Q");
        assert_ne!(q.x, pmq.x, "Q != P-Q");

        // Independent cross-source check: PX/QX as stored in `e0_basis.c`
        // (BASIS_E0_PX/QX) — a separate C file from the `basis_even` table the
        // function transcribes — must agree, guarding against a single-limb typo
        // in either transcription.
        let e0_px = fp2_mont3(
            [
                0x31c4a31adbd9a5c6,
                0xe7ad90c51d65d7b2,
                0x88ba021701e76d61,
                0x2cb3cdb2a2e90ddd,
                0xdc1b70072d06f585,
                0x16eecbda94894ad1,
            ],
            [
                0xf42096161ef8662a,
                0xcba5e8ce200d142d,
                0x2205c5d40d107d81,
                0xd00330eccc07a7e7,
                0x16d8d4adf934c3fa,
                0x6065815b3283164,
            ],
        );
        let e0_qx = fp2_mont3(
            [
                0x6f999f727a40c5b,
                0x50a8ca71cebbf1da,
                0x65cc12a7b6e85c42,
                0x9151a12f13f8774b,
                0x8678d0d647499967,
                0x2e23bfb6dd51ff28,
            ],
            [
                0x6bcfee41588c1c62,
                0xa9249a07cd644dfe,
                0xef21e097d60b5ff8,
                0xcecabfeab509e310,
                0xf010f836ce26d4bd,
                0x2a7787556e853bb9,
            ],
        );
        assert_eq!(p.x, e0_px, "P.x matches e0_basis.c BASIS_E0_PX");
        assert_eq!(q.x, e0_qx, "Q.x matches e0_basis.c BASIS_E0_QX");
    }
}

// ---------------------------------------------------------------------------
// Endomorphism action matrices (lvl3).
//
// Same shape and meaning as the lvl1 matrices below, but the integers are
// reduced mod `2^376` (lvl3 TORSION_EVEN_POWER, F = 376) so each entry is a
// 6-limb little-endian value. VERBATIM from
// `src/precomp/ref/lvl3/endomorphism_action.c` CURVES_WITH_ENDOMORPHISMS[0]:
// the struct field order (from endomorphism_action.h) is
// `action_i, action_j, action_k, action_gen2, action_gen3, action_gen4`, i.e.
// the 64-bit GMP `_mp_d` integer arrays 1-4 (action_i), 17-20 (action_gen3),
// 21-24 (action_gen4). Offset validated against lvl1's known-good values.
// Exposed as `pub` (reachable) until the generic spine consumes them.

/// `action_i` (lvl3) = action of the quaternion `i` (== `action_gen2`).
pub const ACTION_I_LVL3: [[[u64; 6]; 2]; 2] = [
    [
        [
            0x3a84778f9c97d1,
            0x13daabd666ae39d2,
            0x5f9ff8dbb9e7f153,
            0x62b9a4f0fcb236f7,
            0xe8c5539d36945c07,
            0x9ac691f16c7631,
        ],
        [
            0x76df4a43bac61ac2,
            0xd32d1cf84a2de925,
            0xdf8bc02f1dc07867,
            0x4a9ee07d4f0cf122,
            0x357087917ce20a97,
            0x6634cc519b1749,
        ],
    ],
    [
        [
            0x9c61a4810234fb0f,
            0xe38c3a72cd584bd1,
            0xdc99f1020ea3be7b,
            0xef915d86b229f180,
            0xf66fa9d5883146c4,
            0xfc9ebd6c02a451,
        ],
        [
            0xffc57b887063682f,
            0xec2554299951c62d,
            0xa060072446180eac,
            0x9d465b0f034dc908,
            0x173aac62c96ba3f8,
            0x65396e0e9389ce,
        ],
    ],
];

/// `action_gen3` (lvl3) = action of the O0 generator `(i + j)/2`.
pub const ACTION_GEN3_LVL3: [[[u64; 6]; 2]; 2] = [
    [
        [
            0xe1f64f99ab6f83a3,
            0xec7ad9212b61c2e8,
            0xe0fdf78e75554f14,
            0x107cfb09044bb2bf,
            0x9bbe063355f7f365,
            0xf125b09c11409c,
        ],
        [
            0x127f16ca0130dc3d,
            0x2e8d3ece57d01c5c,
            0x6cab1272eb26c5ae,
            0xfeb3321b07c979c7,
            0x62c3efa2b33ec99f,
            0x4ec959777c7bbe,
        ],
    ],
    [
        [
            0x68d7ec590f9b8f83,
            0x2714909b787e8301,
            0x60f499508ea5e264,
            0xeb9a4d1b392b971d,
            0x1f24cbaadd02b9fb,
            0x910fc86afb626c,
        ],
        [
            0x1e09b06654907c5d,
            0x138526ded49e3d17,
            0x1f0208718aaab0eb,
            0xef8304f6fbb44d40,
            0x6441f9ccaa080c9a,
            0xeda4f63eebf63,
        ],
    ],
];

/// `action_gen4` (lvl3) = action of the O0 generator `(1 + k)/2`.
pub const ACTION_GEN4_LVL3: [[[u64; 6]; 2]; 2] = [
    [
        [
            0x75414cc7cecbac5a,
            0x4e827606200564a0,
            0x292d242e3ce25fda,
            0x41454a599b5d6550,
            0xa2e0d9b7bb7f3081,
            0x365b0a54c45b87,
        ],
        [
            0xfac7d5b97057947,
            0x146a1ce1812188f5,
            0x26c39d760c3c70dd,
            0xba0b51891aa57c19,
            0x3c690b13b47705ad,
            0x688e590a97fdde,
        ],
    ],
    [
        [
            0x6ea5a123443b189a,
            0x1699b8f44358c3e8,
            0xfb6b31bbf36c7f02,
            0x290f14ea45c8eea7,
            0xc64e175cd0ea9c11,
            0x896a655cf9ad0,
        ],
        [
            0x8abeb338313453a7,
            0xb17d89f9dffa9b5f,
            0xd6d2dbd1c31da025,
            0xbebab5a664a29aaf,
            0x5d1f26484480cf7e,
            0xc9a4f5ab3ba478,
        ],
    ],
];

#[cfg(test)]
mod lvl3_action_tests {
    use super::*;
    use crypto_bigint::U384;

    #[test]
    fn action_i_lvl3_squares_to_minus_identity_mod_2_376() {
        // ACTION_I is the matrix of multiplication-by-`i` on E0's 2^376-torsion
        // basis. Since i² = −1 in the quaternion algebra, ACTION_I² ≡ −I
        // (mod 2^376). This is an INDEPENDENT algebraic check on the transcribed
        // integers (not a restatement of them): one wrong limb breaks it.
        let a = [
            [
                U384::from_words(ACTION_I_LVL3[0][0]),
                U384::from_words(ACTION_I_LVL3[0][1]),
            ],
            [
                U384::from_words(ACTION_I_LVL3[1][0]),
                U384::from_words(ACTION_I_LVL3[1][1]),
            ],
        ];
        // mask = 2^376 − 1; note −1 ≡ 2^376 − 1 (mod 2^376).
        let mask = U384::ONE.shl_vartime(376).wrapping_sub(&U384::ONE);
        // (A·A)[i][k] mod 2^376. Each product is taken mod 2^384 (wrapping_mul)
        // then reduced mod 2^376 — valid because 2^376 divides 2^384.
        let entry = |i: usize, k: usize| -> U384 {
            let t0 = a[i][0].wrapping_mul(&a[0][k]);
            let t1 = a[i][1].wrapping_mul(&a[1][k]);
            t0.wrapping_add(&t1).bitand(&mask)
        };
        assert_eq!(entry(0, 0), mask, "(A^2)[0][0] == -1 mod 2^376");
        assert_eq!(entry(1, 1), mask, "(A^2)[1][1] == -1 mod 2^376");
        assert_eq!(entry(0, 1), U384::ZERO, "(A^2)[0][1] == 0");
        assert_eq!(entry(1, 0), U384::ZERO, "(A^2)[1][0] == 0");

        // gen3/gen4 sanity: distinct from each other and from i.
        assert_ne!(ACTION_GEN3_LVL3, ACTION_GEN4_LVL3, "gen3 != gen4");
        assert_ne!(ACTION_I_LVL3, ACTION_GEN3_LVL3, "i != gen3");
    }
}

// ---------------------------------------------------------------------------
// Endomorphism action matrices (lvl1).
//
// Each is a 2×2 matrix of integers describing how an O0 generator acts on the
// even-torsion basis `(P, Q)`: column `j` is the image of the `j`-th basis
// point, so the endomorphism maps `P ↦ [M00]P + [M10]Q`, `Q ↦ [M01]P + [M11]Q`.
// VERBATIM from `CURVES_WITH_ENDOMORPHISMS[0].action_*` (endomorphism_action.c);
// these are plain `ibz` integers (NOT Montgomery) reduced mod `2^TORSION_EVEN_POWER`,
// stored here as their 64-bit little-endian limbs.
//
// NOTE the reference's `action_gen2 == action_i` exactly (the O0 basis's 2nd
// generator IS `i`), which confirms our `quat_make_primitive_o0` coordinate
// ordering {1, i, (i+j)/2, (1+k)/2} aligns with the C reference `order.basis`
// columns that `action_gen2/3/4` index.
// ---------------------------------------------------------------------------

/// `action_i` = action of the quaternion `i` (== `action_gen2`).
const ACTION_I: [[[u64; 4]; 2]; 2] = [
    [
        [
            0xc5d3_bda2_1b54_56db,
            0x7475_9780_861d_dd06,
            0x7f9d_34b2_41af_33d1,
            0x00ca_b471_aa8c_7f8c,
        ],
        [
            0x7bfb_7d32_048b_7d7a,
            0xa955_9182_63d8_9bd3,
            0x76bf_6861_0344_03e1,
            0x0057_4ae3_eeb4_5cd0,
        ],
    ],
    [
        [
            0x856f_d649_3698_444f,
            0x189c_afdf_498f_41db,
            0xf7e0_0bff_e50b_cb5b,
            0x0015_35da_a88b_47f9,
        ],
        [
            0x3a2c_425d_e4ab_a925,
            0x8b8a_687f_79e2_22f9,
            0x8062_cb4d_be50_cc2e,
            0x0035_4b8e_5573_8073,
        ],
    ],
];

/// `action_gen3` = action of the O0 generator `(i + j)/2`.
const ACTION_GEN3: [[[u64; 4]; 2]; 2] = [
    [
        [
            0xfe47_49cf_b7f2_30cd,
            0xbaa3_7335_683b_db8a,
            0x8871_9dd4_74ae_ebe0,
            0x0024_2ba2_3c39_67c8,
        ],
        [
            0x6e8c_9d8a_de09_81fd,
            0x58b7_adb7_77a0_a299,
            0x1a1d_6349_7d41_13a1,
            0x00df_b772_17c5_c40b,
        ],
    ],
    [
        [
            0x523e_3a2d_d1dc_4363,
            0x376e_267e_20f1_ecad,
            0xf004_ddaa_53fc_661b,
            0x006f_d8e1_5b07_267a,
        ],
        [
            0x01b8_b630_480d_cf33,
            0x455c_8cca_97c4_2475,
            0x778e_622b_8b51_141f,
            0x00db_d45d_c3c6_9837,
        ],
    ],
];

/// `action_gen4` = action of the O0 generator `(1 + k)/2`.
const ACTION_GEN4: [[[u64; 4]; 2]; 2] = [
    [
        [
            0xd8ce_0b20_0d79_118e,
            0xf9cd_341f_7238_7b89,
            0x4827_6137_3d2a_1944,
            0x0022_2afe_3506_6ad3,
        ],
        [
            0xaae9_6f34_db42_d6bd,
            0x492f_ac8b_4274_2b3a,
            0x41c8_be28_8e5b_4605,
            0x0066_cb67_08e8_ffe7,
        ],
    ],
    [
        [
            0x4acd_8dc9_3cde_9b92,
            0xb253_93ea_378c_59f6,
            0xb325_d6f3_c63f_4da5,
            0x0024_350e_d143_d36c,
        ],
        [
            0x2731_f4df_f286_ee73,
            0x0632_cbe0_8dc7_8476,
            0xb7d8_9ec8_c2d5_e6bb,
            0x00dd_d501_caf9_952c,
        ],
    ],
];

/// Low `f`-bit mask `2^f − 1` as a `Uint<8>`.
#[inline]
fn mask_2f(f: usize) -> Uint<8> {
    Uint::<8>::MAX.wrapping_shr(u32::try_from(512 - f).expect("512 - f fits u32 for 1 <= f <= 512"))
}

/// `(a − b) mod 2^f`.
#[inline]
fn sub_mod_2f(a: &Uint<8>, b: &Uint<8>, f: usize) -> Uint<8> {
    a.wrapping_sub(b) & mask_2f(f)
}

/// Apply a 2×2 integer matrix `m` to an x-only torsion basis `(P, Q, P−Q)` of
/// order `2^f`, in place: returns `(R, S, R−S)` where `R = [m00]P + [m10]Q`,
/// `S = [m01]P + [m11]Q`, `R−S = [m00−m01]P + [m10−m11]Q` (all mod `2^f`).
///
/// Port of the C reference `matrix_application_even_basis` (id2iso.c). `a24` is
/// the affine doubling constant `(A + 2)/4`. Returns `None` if any biladder
/// fails. `m` entries are taken mod `2^f`.
pub(crate) fn matrix_application_even_basis<F: BaseField>(
    p: &MontgomeryPoint<F>,
    q: &MontgomeryPoint<F>,
    pmq: &MontgomeryPoint<F>,
    m: &[[Uint<8>; 2]; 2],
    f: usize,
    a24: &Fp2<F>,
) -> Option<(MontgomeryPoint<F>, MontgomeryPoint<F>, MontgomeryPoint<F>)> {
    use crate::ec::biscalar::ec_biscalar_mul;

    let m00 = m[0][0] & mask_2f(f);
    let m01 = m[0][1] & mask_2f(f);
    let m10 = m[1][0] & mask_2f(f);
    let m11 = m[1][1] & mask_2f(f);

    // R = [m00]P + [m10]Q
    let r = ec_biscalar_mul(&m00.to_le_bytes(), &m10.to_le_bytes(), f, p, q, pmq, a24)?;
    // S = [m01]P + [m11]Q
    let s = ec_biscalar_mul(&m01.to_le_bytes(), &m11.to_le_bytes(), f, p, q, pmq, a24)?;
    // R − S = [m00−m01]P + [m10−m11]Q
    let d_p = sub_mod_2f(&m00, &m01, f);
    let d_q = sub_mod_2f(&m10, &m11, f);
    let rmq = ec_biscalar_mul(&d_p.to_le_bytes(), &d_q.to_le_bytes(), f, p, q, pmq, a24)?;

    Some((r, s, rmq))
}

/// `(m · v) mod 2^f` for a 2×2 matrix and a 2-vector (all `Uint<8>`, mod `2^f`).
#[inline]
fn mat2x2_eval_mod_2f(m: &[[Uint<8>; 2]; 2], v: &[Uint<8>; 2], f: usize) -> [Uint<8>; 2] {
    [
        add_mod_2f(
            &mul_mod_2f(&m[0][0], &v[0], f),
            &mul_mod_2f(&m[0][1], &v[1], f),
            f,
        ),
        add_mod_2f(
            &mul_mod_2f(&m[1][0], &v[0], f),
            &mul_mod_2f(&m[1][1], &v[1], f),
            f,
        ),
    ]
}

/// Inverse of a 2×2 matrix modulo `2^f`. Returns `None` if the determinant is
/// even (not a unit mod `2^f`). `inv = det⁻¹ · [[d, −b], [−c, a]]`.
fn mat2x2_inv_mod_2f(m: &[[Uint<8>; 2]; 2], f: usize) -> Option<[[Uint<8>; 2]; 2]> {
    let det = sub_mod_2f(
        &mul_mod_2f(&m[0][0], &m[1][1], f),
        &mul_mod_2f(&m[0][1], &m[1][0], f),
        f,
    );
    let modulus = mask_2f(f).wrapping_add(&Uint::<8>::ONE); // 2^f
    let det_inv = crate::quaternion::sign_orchestration::uint_inv_mod_vartime::<8>(&det, &modulus)?;
    let neg = |x: &Uint<8>| sub_mod_2f(&Uint::<8>::ZERO, x, f);
    Some([
        [
            mul_mod_2f(&det_inv, &m[1][1], f),
            mul_mod_2f(&det_inv, &neg(&m[0][1]), f),
        ],
        [
            mul_mod_2f(&det_inv, &neg(&m[1][0]), f),
            mul_mod_2f(&det_inv, &m[0][0], f),
        ],
    ])
}

/// Build the challenge ideal whose kernel is generated by
/// `vec2[0]·B0[0] + vec2[1]·B0[1]` (B0 = canonical E0 basis of the `2^f`
/// torsion). Port of C `id2iso_kernel_dlogs_to_ideal_even` (`id2iso.c`).
///
/// Applies the endomorphisms `1` and `j + (1+k)/2` to the kernel generator to
/// form a basis, decomposes `[i]·gen` over it, and reads off a quaternion
/// `a − i + b·(j + (1+k)/2)` (denom 2) that kills the kernel; the left ideal it
/// generates of norm `2^f` is the challenge ideal. `vec2` entries are taken mod
/// `2^f`. Built at internal width 24 (the norm-`2^f` ideal's `det_4x4` reaches
/// ~`2^(4f/... )` and overflows narrow widths), returned at `LeftIdeal<16>`.
/// Returns `None` if the basis matrix is singular mod `2^f`. lvl1-pinned.
pub(crate) fn id2iso_kernel_dlogs_to_ideal_even<P: LevelConstants>(
    vec2: &[Uint<8>; 2],
    f: usize,
) -> Option<crate::quaternion::ideal::LeftIdeal<16>> {
    use crate::quaternion::Quaternion;
    use crate::quaternion::o0_mul::{c_ideal_to_left_ideal, quat_lideal_create};
    const WK: usize = 24;

    // ACTION_J = action of `j` = 2·GEN3 − I  (j = 2·(i+j)/2 − i).
    let [gen_i, gen3, gen4] = action_matrices_2f::<P>()?;
    let mut action_j = [[Uint::<8>::ZERO; 2]; 2];
    for r in 0..2 {
        for c in 0..2 {
            let two_g3 = add_mod_2f(&gen3[r][c], &gen3[r][c], f);
            action_j[r][c] = sub_mod_2f(&two_g3, &gen_i[r][c], f);
        }
    }

    // mat = [ vec2 | (j + (1+k)/2)·vec2 ] (columns), then invert mod 2^f.
    let jv = mat2x2_eval_mod_2f(&action_j, vec2, f);
    let g4v = mat2x2_eval_mod_2f(&gen4, vec2, f);
    let mat = [
        [vec2[0], add_mod_2f(&jv[0], &g4v[0], f)],
        [vec2[1], add_mod_2f(&jv[1], &g4v[1], f)],
    ];
    let inv = mat2x2_inv_mod_2f(&mat, f)?;

    // vec = inv · ([i]·vec2).
    let iv = mat2x2_eval_mod_2f(&gen_i, vec2, f);
    let vec = mat2x2_eval_mod_2f(&inv, &iv, f);

    // gen = a − i + b·(j + (1+k)/2), denom 2:
    //   coord0 = 2a + b, coord1 = −2, coord2 = 2b, coord3 = b.
    // a, b are in [0, 2^f) ⇒ widen to Int<WK> non-negative.
    let to_wk = |u: &Uint<8>| -> crypto_bigint::Int<WK> { *u.resize::<WK>().as_int() };
    let a_i = to_wk(&vec[0]);
    let b_i = to_wk(&vec[1]);
    let two = crypto_bigint::Int::<WK>::from_i64(2);
    let gen_q = Quaternion::<WK>::new(
        two.wrapping_mul(&a_i).wrapping_add(&b_i), // 2a + b
        crypto_bigint::Int::<WK>::from_i64(-2),    // −2 (the −i term, ×denom 2)
        two.wrapping_mul(&b_i),                    // 2b
        b_i,                                       // b
    );

    let two_pow = Uint::<WK>::ONE.shl_vartime(u32::try_from(f).ok()?);
    let p_wk = P::prime::<WK>();
    let (basis, denom, norm) = quat_lideal_create::<WK>(
        &gen_q,
        &crypto_bigint::Int::<WK>::from_i64(2),
        &two_pow,
        &p_wk,
    );
    // Mirror the C assert: the resulting ideal must have norm exactly 2^f.
    if norm != two_pow {
        return None;
    }
    let wide = c_ideal_to_left_ideal::<WK>(&basis, &denom, &norm);

    // Narrow to LeftIdeal<16> (norm-2^f basis entries fit; cached_norm = 2^(2f)).
    use crate::quaternion::lattice::narrow_int_lattice;
    let mut b16 = [[crypto_bigint::Int::<16>::from_i64(0); 4]; 4];
    for (row16, row_wide) in b16.iter_mut().zip(wide.basis.iter()) {
        for (e16, e_wide) in row16.iter_mut().zip(row_wide.iter()) {
            *e16 = narrow_int_lattice::<WK, 16>(e_wide);
        }
    }
    Some(
        crate::quaternion::ideal::LeftIdeal::<16>::with_denom_and_norm(
            b16,
            wide.denom.resize::<16>(),
            wide.cached_norm.resize::<16>(),
        ),
    )
}

/// Compute the challenge ideal for signing — C `compute_challenge_ideal_signature`
/// (`sign.c`). The challenge kernel is `B[0] + chall_coeff·B[1]` over the secret
/// curve's canonical basis; pulled back through the secret-key isogeny it becomes
/// `vec = sk.mat_BAcan_to_BA0_two · [1, chall_coeff]` over `E0`'s canonical
/// basis, whose `id2iso_kernel_dlogs_to_ideal_even` is the (norm-`2^f`) challenge
/// ideal. lvl1-pinned (`f = TORSION_EVEN_POWER = 248`).
pub(crate) fn compute_challenge_ideal_signature<P: LevelConstants>(
    mat_bacan_to_ba0_two: &[[Uint<8>; 2]; 2],
    chall_coeff: &Uint<8>,
    f: usize,
) -> Option<crate::quaternion::ideal::LeftIdeal<16>> {
    // vec = mat · [1, chall_coeff]  (mod 2^f).
    let vec = mat2x2_eval_mod_2f(
        mat_bacan_to_ba0_two,
        &[Uint::<8>::ONE & mask_2f(f), *chall_coeff & mask_2f(f)],
        f,
    );
    id2iso_kernel_dlogs_to_ideal_even::<P>(&vec, f)
}

/// Recover the kernel coordinates `vec2` (in the canonical `E_0[2^f]` basis) of
/// the even ideal generated by `gen` (O_0-coords) of norm `2^f`. Port of C
/// `id2iso_ideal_to_kernel_dlogs_even` (`id2iso.c`) for the case where the ideal
/// generator is known: build the matrix of `conj(gen)`'s action on the
/// `2^f`-torsion (`coeffs0·I + coeffs1·GEN2 + coeffs2·GEN3 + coeffs3·GEN4` where
/// `coeffs = conj(gen)` in O_0-coords), then read a primitive kernel column mod
/// `2^f`. Returns `[v0, v1]` with `ker = v0·B0[0] + v1·B0[1]`.
pub(crate) fn id2iso_ideal_to_kernel_dlogs_even<P: LevelConstants, const LIMBS: usize>(
    gen_q: &[crypto_bigint::Int<LIMBS>; 4],
    f: usize,
) -> [Uint<8>; 2] {
    let alpha = crate::quaternion::o0_mul::o0_conjugate::<LIMBS>(gen_q);
    let c = [
        int_to_mod_2f(&alpha[0], f),
        int_to_mod_2f(&alpha[1], f),
        int_to_mod_2f(&alpha[2], f),
        int_to_mod_2f(&alpha[3], f),
    ];
    // GEN2 == ACTION_I; `_` levels (no E0 endo data) fall back to all-zero
    // matrices — unreachable in practice (only lvl1/lvl3 sign).
    let [gen2, gen3, gen4] = action_matrices_2f::<P>().unwrap_or([[[Uint::<8>::ZERO; 2]; 2]; 3]);
    let mut mat = [[Uint::<8>::ZERO; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let mut e = if i == j { c[0] } else { Uint::<8>::ZERO };
            e = add_mod_2f(&e, &mul_mod_2f(&c[1], &gen2[i][j], f), f);
            e = add_mod_2f(&e, &mul_mod_2f(&c[2], &gen3[i][j], f), f);
            e = add_mod_2f(&e, &mul_mod_2f(&c[3], &gen4[i][j], f), f);
            mat[i][j] = e;
        }
    }
    // Pick a primitive column: [mat[0][0], mat[1][0]] unless both are even.
    let even = |x: &Uint<8>| x.to_le_bytes()[0] & 1 == 0;
    if even(&mat[0][0]) && even(&mat[1][0]) {
        [mat[0][1], mat[1][1]]
    } else {
        [mat[0][0], mat[1][0]]
    }
}

/// Build a `[[Uint<8>;2];2]` matrix from a `[[[u64;4];2];2]` limb table.
#[inline]
fn mat_from_limbs(t: &[[[u64; 4]; 2]; 2]) -> [[Uint<8>; 2]; 2] {
    [
        [
            Uint::<8>::from_words([t[0][0][0], t[0][0][1], t[0][0][2], t[0][0][3], 0, 0, 0, 0]),
            Uint::<8>::from_words([t[0][1][0], t[0][1][1], t[0][1][2], t[0][1][3], 0, 0, 0, 0]),
        ],
        [
            Uint::<8>::from_words([t[1][0][0], t[1][0][1], t[1][0][2], t[1][0][3], 0, 0, 0, 0]),
            Uint::<8>::from_words([t[1][1][0], t[1][1][1], t[1][1][2], t[1][1][3], 0, 0, 0, 0]),
        ],
    ]
}

/// Build a `[[Uint<8>;2];2]` matrix from an `N`-limb table (`N ≤ 8`), padding
/// the high words with zero. Generalizes [`mat_from_limbs`] over the field
/// width (lvl1 action matrices are 4-limb, lvl3's are 6-limb).
#[inline]
fn mat_from_limbs_n<const N: usize>(t: &[[[u64; N]; 2]; 2]) -> [[Uint<8>; 2]; 2] {
    let conv = |limbs: &[u64; N]| {
        let mut w = [0u64; 8];
        w[..N].copy_from_slice(limbs);
        Uint::<8>::from_words(w)
    };
    [
        [conv(&t[0][0]), conv(&t[0][1])],
        [conv(&t[1][0]), conv(&t[1][1])],
    ]
}

/// The per-level `E0` endomorphism action matrices `(I, GEN3, GEN4)` on the
/// `2^F`-torsion, as `[[Uint<8>;2];2]` reduced representatives. Level 1 uses the
/// 4-limb tables; level 3 the 6-limb `*_LVL3` tables. `None` for levels without
/// precomputed E0 endomorphism data.
fn action_matrices_2f<P: LevelConstants>() -> Option<[[[Uint<8>; 2]; 2]; 3]> {
    match P::LEVEL {
        1 => Some([
            mat_from_limbs(&ACTION_I),
            mat_from_limbs(&ACTION_GEN3),
            mat_from_limbs(&ACTION_GEN4),
        ]),
        3 => Some([
            mat_from_limbs_n(&ACTION_I_LVL3),
            mat_from_limbs_n(&ACTION_GEN3_LVL3),
            mat_from_limbs_n(&ACTION_GEN4_LVL3),
        ]),
        _ => None,
    }
}

/// `(a + b) mod 2^f`.
#[inline]
fn add_mod_2f(a: &Uint<8>, b: &Uint<8>, f: usize) -> Uint<8> {
    a.wrapping_add(b) & mask_2f(f)
}

/// `(a · b) mod 2^f` (operands already `< 2^f ≤ 2^256`, so the low 256 bits of
/// the product reduced mod `2^f` is exact).
#[inline]
fn mul_mod_2f(a: &Uint<8>, b: &Uint<8>, f: usize) -> Uint<8> {
    a.wrapping_mul(b) & mask_2f(f)
}

/// Reduce a signed quaternion-side `Int<L>` to `Uint<8>` modulo `2^f`
/// (two's-complement-correct: negatives map to `2^f − |x|`).
#[inline]
fn int_to_mod_2f<const L: usize>(x: &crypto_bigint::Int<L>, f: usize) -> Uint<8> {
    let mag = x.abs(); // Uint<L>
    let w = mag.to_words();
    let lo = Uint::<8>::from_words([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]) & mask_2f(f);
    if bool::from(x.is_negative()) && lo != Uint::<8>::ZERO {
        sub_mod_2f(&Uint::<8>::ZERO, &lo, f) // 2^f − |x|
    } else {
        lo
    }
}

/// Apply an O0 endomorphism `theta` (given as an integer-standard quaternion)
/// to the even-torsion basis `(P, Q, P−Q)` of order `2^f`, in place.
///
/// Port of the C reference `endomorphism_application_even_basis` (id2iso.c),
/// specialized to the standard order O0 (`index_alternate_curve = 0`):
/// decompose `theta` over O0 (`quat_make_primitive_o0` → `coeffs`, `content`),
/// build the 2×2 action matrix
/// `M = content · (coeffs0·I + coeffs1·GEN2 + coeffs2·GEN3 + coeffs3·GEN4)`
/// mod `2^f`, then apply it via [`matrix_application_even_basis`].
///
/// `GEN2 == ACTION_I` (the O0 generator at index 1 is `i`); `GEN3`/`GEN4` are
/// `(i+j)/2` and `(1+k)/2`. `a24 = (A+2)/4`. Returns `None` if a biladder
/// fails.
pub(crate) fn endomorphism_application_even_basis<P: LevelConstants, const L: usize>(
    p: &MontgomeryPoint<P::Field>,
    q: &MontgomeryPoint<P::Field>,
    pmq: &MontgomeryPoint<P::Field>,
    theta: &crate::quaternion::Quaternion<L>,
    f: usize,
    a24: &Fp2<P::Field>,
) -> Option<EndoImageTriple<P::Field>> {
    let o0 = crate::quaternion::o0_mul::standard_to_o0_basis(theta);
    endomorphism_application_o0_coords::<P, L>(p, q, pmq, &o0, f, a24)
}

/// Same as [`endomorphism_application_even_basis`] but takes the endomorphism
/// already in `O_0`-basis coordinates (the form `RepresentInteger` returns),
/// skipping the standard→O_0 conversion.
pub(crate) fn endomorphism_application_o0_coords<P: LevelConstants, const L: usize>(
    p: &MontgomeryPoint<P::Field>,
    q: &MontgomeryPoint<P::Field>,
    pmq: &MontgomeryPoint<P::Field>,
    o0_coords: &[crypto_bigint::Int<L>; 4],
    f: usize,
    a24: &Fp2<P::Field>,
) -> Option<EndoImageTriple<P::Field>> {
    let (primitive, content) =
        crate::quaternion::o0_mul::make_primitive_from_o0_coords::<L>(o0_coords);

    let c: [Uint<8>; 4] = [
        int_to_mod_2f(&primitive[0], f),
        int_to_mod_2f(&primitive[1], f),
        int_to_mod_2f(&primitive[2], f),
        int_to_mod_2f(&primitive[3], f),
    ];
    let content_u = int_to_mod_2f(&content, f);

    let curve = P::nice_curve_e0();
    let gen2 = mat_int8_to_mod_2f(&curve.action_gen2, f);
    let gen3 = mat_int8_to_mod_2f(&curve.action_gen3, f);
    let gen4 = mat_int8_to_mod_2f(&curve.action_gen4, f);

    let mut m = [[Uint::<8>::ZERO; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            // diagonal carries coeffs0 (the identity component)
            let mut e = if i == j { c[0] } else { Uint::<8>::ZERO };
            e = add_mod_2f(&e, &mul_mod_2f(&c[1], &gen2[i][j], f), f);
            e = add_mod_2f(&e, &mul_mod_2f(&c[2], &gen3[i][j], f), f);
            e = add_mod_2f(&e, &mul_mod_2f(&c[3], &gen4[i][j], f), f);
            e = mul_mod_2f(&e, &content_u, f);
            m[i][j] = e;
        }
    }

    matrix_application_even_basis(p, q, pmq, &m, f, a24)
}

/// Reduce a `[[Int<8>;2];2]` action table (entries already in `[0, 2^F)`) to
/// `[[Uint<8>;2];2]` mod `2^f`.
#[inline]
fn mat_int8_to_mod_2f(t: &[[crypto_bigint::Int<8>; 2]; 2], f: usize) -> [[Uint<8>; 2]; 2] {
    [
        [int_to_mod_2f(&t[0][0], f), int_to_mod_2f(&t[0][1], f)],
        [int_to_mod_2f(&t[1][0], f), int_to_mod_2f(&t[1][1], f)],
    ]
}

/// Apply the endomorphism `theta / theta_denom` on the alternate starting
/// curve `CURVES_WITH_ENDOMORPHISMS[index_alternate_curve]` to its even-torsion
/// basis `(P, Q, P−Q)` of order `2^f`.
///
/// Port of the C reference `endomorphism_application_even_basis`
/// (`id2iso.c:140`) in its GENERIC `index_alternate_curve` form — the
/// `n_order ≠ 0` path (`dim2id2iso.c:148`). The C body decomposes `theta` over
/// `EXTREMAL_ORDERS[index].order` (`quat_alg_make_primitive` → `coeffs` +
/// `content`, the `content` always odd), builds
/// `M = content · (coeffs0·I + coeffs1·GEN2 + coeffs2·GEN3 + coeffs3·GEN4)`
/// mod `2^f` from `CURVES_WITH_ENDOMORPHISMS[index].action_gen2/3/4`, and
/// applies it via [`matrix_application_even_basis`].
///
/// Index mapping (Rust tables are offset-by-one from the C arrays):
/// `index_alternate_curve == 0` is the standard order `O_0` — delegated to the
/// validated [`endomorphism_application_even_basis`] (which assumes an integer
/// `theta`, i.e. `theta_denom == 1`); `index_alternate_curve == k ≥ 1` uses the
/// level-selected alternate extremal order and NICE starting curve from
/// [`LevelConstants`].
///
/// Returns `None` if `theta/theta_denom ∉ EXTREMAL_ORDERS[k]` (the C `assert`
/// fails), if the index is out of range, or if a biladder fails.
///
/// Byte-exact correctness of the `k ≥ 1` matrix is NOT independently checkable
/// in-tree (no golden vectors); it is anchored here by the identity
/// endomorphism (`θ = 1 ⇒ basis fixed`) and proven end-to-end by the eventual
/// keygen KAT (item 8).
// Carries an even basis, alternate-curve index, theta quaternion/denominator, torsion exponent, and A24 constant.
#[allow(clippy::too_many_arguments)]
pub(crate) fn endomorphism_application_even_basis_indexed<P: LevelConstants>(
    p: &MontgomeryPoint<P::Field>,
    q: &MontgomeryPoint<P::Field>,
    pmq: &MontgomeryPoint<P::Field>,
    index_alternate_curve: usize,
    theta: &crate::quaternion::Quaternion<8>,
    theta_denom: &crypto_bigint::Int<8>,
    f: usize,
    a24: &Fp2<P::Field>,
) -> Option<EndoImageTriple<P::Field>> {
    if index_alternate_curve == 0 {
        // Standard O_0 path (validated): integer theta only.
        debug_assert!(
            *theta_denom == crypto_bigint::Int::<8>::ONE,
            "index 0 (O_0) path requires integer theta (denom = 1)",
        );
        return endomorphism_application_even_basis::<P, 8>(p, q, pmq, theta, f, a24);
    }

    let k = index_alternate_curve - 1;
    if k >= P::NUM_ALTERNATE_EXTREMAL_ORDERS {
        return None;
    }
    let order = P::alternate_extremal_order(k);
    let curve = P::nice_curve(k);

    // Decompose theta/theta_denom over EXTREMAL_ORDERS[k] (C quat_alg_make_primitive):
    // coeffs in the order basis (index 0 = identity component), content odd.
    let (coeffs, content) = crate::quaternion::extremal_orders::make_primitive_over_alt_order(
        &order,
        theta,
        theta_denom,
    )?;

    let c: [Uint<8>; 4] = [
        int_to_mod_2f(&coeffs[0], f),
        int_to_mod_2f(&coeffs[1], f),
        int_to_mod_2f(&coeffs[2], f),
        int_to_mod_2f(&coeffs[3], f),
    ];
    let content_u = int_to_mod_2f(&content, f);

    let gen2 = mat_int8_to_mod_2f(&curve.action_gen2, f);
    let gen3 = mat_int8_to_mod_2f(&curve.action_gen3, f);
    let gen4 = mat_int8_to_mod_2f(&curve.action_gen4, f);

    let mut m = [[Uint::<8>::ZERO; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            // diagonal carries coeffs0 (the identity component)
            let mut e = if i == j { c[0] } else { Uint::<8>::ZERO };
            e = add_mod_2f(&e, &mul_mod_2f(&c[1], &gen2[i][j], f), f);
            e = add_mod_2f(&e, &mul_mod_2f(&c[2], &gen3[i][j], f), f);
            e = add_mod_2f(&e, &mul_mod_2f(&c[3], &gen4[i][j], f), f);
            e = mul_mod_2f(&e, &content_u, f);
            m[i][j] = e;
        }
    }

    matrix_application_even_basis(p, q, pmq, &m, f, a24)
}

/// Reduce a `Uint<L>` to `Uint<8>` modulo `2^f` (low `f` bits).
#[inline]
fn uint_to_mod_2f<const L: usize>(x: &Uint<L>, f: usize) -> Uint<8> {
    let w = x.to_words();
    Uint::<8>::from_words([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]) & mask_2f(f)
}

/// Embed a `Uint<8>` value `< 2^f` as a non-negative `Int<L>`.
#[inline]
fn u256_to_int<const L: usize>(x: &Uint<8>) -> crypto_bigint::Int<L> {
    let xw = x.to_words();
    let mut words = [0u64; L];
    words[..8].copy_from_slice(&xw);
    crypto_bigint::Int::<L>::from_words(words)
}

/// Apply a RATIONAL endomorphism `(num / denom) · (1 / extra)` to the
/// even-torsion basis `(P, Q, P−Q)` of order `2^f`, in place.
///
/// The Clapotis spine scales its endomorphisms `θ` and `β1` by an
/// inverse integer mod `2^f` before applying them (the C ref multiplies
/// `θ.coord` by `invmod(d1, 2^f)` etc. and folds the rational
/// denominator implicitly). [`endomorphism_application_even_basis`]
/// takes only an integer numerator, so this wrapper folds the rational
/// denominator AND the `extra` scalar in: it converts `num` to O0 coords
/// in EXACT arithmetic (the O0 basis has `/2` terms, so reducing first
/// would corrupt them), then scales each O0 coord by
/// `s = invmod((denom · extra) mod 2^f, 2^f)` mod `2^f`. Scaling the O0
/// coords by `s` scales the resulting action matrix by `s`, i.e. the
/// images by `[s]` — exactly `num · denom⁻¹ · extra⁻¹` acting mod `2^f`.
///
/// `denom` and `extra` must be odd (invertible mod `2^f`). Returns
/// `None` if the inverse does not exist or a biladder fails.
// Carries an even basis, rational theta numerator/denominator, extra odd factor, torsion exponent, and A24 constant.
#[allow(clippy::too_many_arguments)]
pub(crate) fn endomorphism_application_rational_even_basis<P: LevelConstants, const L: usize>(
    p: &MontgomeryPoint<P::Field>,
    q: &MontgomeryPoint<P::Field>,
    pmq: &MontgomeryPoint<P::Field>,
    num: &crate::quaternion::Quaternion<L>,
    denom: &Uint<L>,
    extra: &Uint<L>,
    f: usize,
    a24: &Fp2<P::Field>,
) -> Option<EndoImageTriple<P::Field>> {
    // Split `denom` into its 2-adic part `2^t` and odd part `denom_odd`:
    // (num/denom)/extra = num / (2^t · denom_odd · extra). The odd factor
    // `denom_odd·extra` is invertible mod 2^f; the `2^t` factor is NOT, but it
    // CANCELS against the O_0 coords — find_uv's O_0→standard doubling makes
    // `denom` even while `standard_to_o0_basis(num)` carries a matching factor
    // of 2 (in the spine θ = β2·conj(β1)/n(I) ∈ O_0, so its O_0 coords are
    // integral and divisible by `2^t`). So divide the O_0 coords by `2^t`
    // exactly and scale by `(denom_odd·extra)^{-1} mod 2^f`. For ODD `denom`
    // (t = 0) this is bit-identical to inverting `denom·extra` together.
    let t = denom.trailing_zeros();
    let denom_odd = denom.wrapping_shr(t);
    let de = mul_mod_2f(&uint_to_mod_2f(&denom_odd, f), &uint_to_mod_2f(extra, f), f);
    let modulus = mask_2f(f).wrapping_add(&Uint::<8>::ONE); // 2^f
    let s = crate::quaternion::sign_orchestration::uint_inv_mod_vartime::<8>(&de, &modulus)?;

    let two_t = *Uint::<L>::ONE.shl_vartime(t).as_int(); // 2^t
    let o0 = crate::quaternion::o0_mul::standard_to_o0_basis(num);
    let scaled: [crypto_bigint::Int<L>; 4] = core::array::from_fn(|i| {
        // Exact: 2^t divides the O_0 coord (θ's 2-adic part is integral).
        let divided = crate::quaternion::hnf::int_div_floor::<L>(&o0[i], &two_t);
        debug_assert_eq!(
            divided.wrapping_mul(&two_t),
            o0[i],
            "endomorphism_application_rational_even_basis: 2^v2(denom) must divide the O_0 coord",
        );
        let cm = int_to_mod_2f(&divided, f);
        u256_to_int::<L>(&mul_mod_2f(&cm, &s, f))
    });

    endomorphism_application_o0_coords::<P, L>(p, q, pmq, &scaled, f, a24)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ec::jacobian::JacobianPoint;
    use crate::ec::montgomery::MontgomeryCurve;
    use crate::params::lvl1::Level1;
    use subtle::ConstantTimeEq;

    #[test]
    fn id2iso_kernel_dlogs_to_ideal_even_is_a_left_ideal() {
        use crate::quaternion::o0_mul::multiply_o0_basis;
        // Challenge-ideal construction for a kernel vec2 = [1, c].
        let f = 248usize;
        let vec2 = [Uint::<8>::ONE, Uint::<8>::from_u64(12345)];
        let ideal = id2iso_kernel_dlogs_to_ideal_even::<Level1>(&vec2, f)
            .expect("challenge ideal builds (norm = 2^f asserted internally)");
        // Validity by LEFT-CLOSURE (reduced_norm overflows det_4x4 at norm 2^248).
        let p16 = crate::params::lvl1::prime().resize::<16>();
        for r in 0..4 {
            let g = ideal.basis[r];
            for k in 0..4 {
                let mut e = [crypto_bigint::Int::<16>::from_i64(0); 4];
                e[k] = crypto_bigint::Int::<16>::from_i64(1);
                assert!(
                    ideal.contains(&multiply_o0_basis::<16>(&e, &g, &p16)),
                    "challenge ideal must be left-O_0-closed (gen {r}, basis elt {k})",
                );
            }
        }
        // A different challenge coefficient yields a different ideal.
        let other = id2iso_kernel_dlogs_to_ideal_even::<Level1>(
            &[Uint::<8>::ONE, Uint::<8>::from_u64(6789)],
            f,
        )
        .expect("second challenge ideal");
        assert_ne!(
            ideal.basis, other.basis,
            "distinct chall_coeff ⇒ distinct ideal"
        );
    }

    #[test]
    fn compute_challenge_ideal_signature_identity_matrix_matches_direct() {
        // With the identity change matrix, vec = [1, chall_coeff], so the
        // signing challenge ideal equals the direct id2iso construction.
        let f = 248usize;
        let identity = [
            [Uint::<8>::ONE, Uint::<8>::ZERO],
            [Uint::<8>::ZERO, Uint::<8>::ONE],
        ];
        let c = Uint::<8>::from_u64(424242);
        let via_sig =
            compute_challenge_ideal_signature::<Level1>(&identity, &c, f).expect("sig challenge");
        let direct =
            id2iso_kernel_dlogs_to_ideal_even::<Level1>(&[Uint::<8>::ONE, c], f).expect("direct");
        assert_eq!(
            via_sig.basis, direct.basis,
            "identity matrix ⇒ direct construction"
        );
        // A non-trivial matrix changes the result.
        let m = [
            [Uint::<8>::from_u8(3), Uint::<8>::from_u8(1)],
            [Uint::<8>::from_u8(2), Uint::<8>::from_u8(5)],
        ];
        let via_m =
            compute_challenge_ideal_signature::<Level1>(&m, &c, f).expect("matrix challenge");
        assert_ne!(
            via_m.basis, direct.basis,
            "non-identity matrix ⇒ different ideal"
        );
    }

    /// The ported basis points lie on E0 and each has order exactly `2^248`,
    /// and `PmQ` is a genuine difference of `P` and `Q`. This validates the
    /// Montgomery-limb port end-to-end: a transcription error would put a
    /// point off-curve or off-order.
    #[test]
    fn basis_e0_lvl1_is_on_curve_order_2_248_and_consistent() {
        let curve = MontgomeryCurve::<Fp1Element>::e0();
        let a = curve.a;
        let a24 = curve.a24();
        let (p, q, pmq) = basis_e0_lvl1();

        // (a) all three on-curve (x³ + A x² + x is a square).
        for (name, pt) in [("P", &p), ("Q", &q), ("PmQ", &pmq)] {
            assert!(
                bool::from(JacobianPoint::from_montgomery_xz(pt, &a).is_some()),
                "basis point {name} must be on E0",
            );
        }

        // (b) order exactly 2^248: [2^248]·pt = O, [2^247]·pt ≠ O.
        let order_check = |pt: &MontgomeryPoint<Fp1Element>| {
            let mut acc = *pt;
            for _ in 0..247 {
                acc = acc.x_double(&a24);
            }
            let at_247 = acc; // [2^247]·pt
            let at_248 = acc.x_double(&a24); // [2^248]·pt
            (
                !bool::from(at_247.is_infinity()),
                bool::from(at_248.is_infinity()),
            )
        };
        for (name, pt) in [("P", &p), ("Q", &q), ("PmQ", &pmq)] {
            let (nonzero_247, zero_248) = order_check(pt);
            assert!(
                nonzero_247,
                "{name}: [2^247]·pt must be ≠ O (order ≥ 2^248)"
            );
            assert!(zero_248, "{name}: [2^248]·pt must be O (order | 2^248)");
        }

        // (c) PmQ is a genuine difference: lift P, Q (independent signs) and
        // check x(PmQ) is one of x(P−Q), x(P+Q) = (u±v)/w from ADDComponents.
        let p_jac = JacobianPoint::from_montgomery_xz(&p, &a).unwrap();
        let q_jac = JacobianPoint::from_montgomery_xz(&q, &a).unwrap();
        let (u, v, w) = p_jac.add_components(&q_jac, &a);
        let w_inv = w.invert().unwrap_or(Fp2::zero());
        let diff_x = u.add(&v).mul(&w_inv); // x(P−Q)
        let sum_x = u.sub(&v).mul(&w_inv); // x(P+Q)
        assert!(
            bool::from(pmq.x.ct_eq(&diff_x)) || bool::from(pmq.x.ct_eq(&sum_x)),
            "PmQ.x must be a genuine ± difference of P and Q",
        );
    }

    /// ALIGNMENT GROUND TRUTH: the action matrix `action_i` applied to the
    /// even-torsion basis must realise the `i`-endomorphism of E0, which on
    /// `y² = x³ + x` is `ι:(x, y) ↦ (−x, √−1·y)` — i.e. `x ↦ −x`. So the
    /// matrix-transformed `(R, S)` must have x-coordinates `(−P.x, −Q.x)`.
    ///
    /// This validates the entire lower bridge stack together: the ported
    /// action table, `matrix_application_even_basis`, `ec_biscalar_mul`, and
    /// `basis_e0_lvl1` — and (with `action_gen2 == action_i`) the
    /// `make_primitive` ↔ `order.basis` coordinate alignment.
    #[test]
    fn action_i_realises_the_i_endomorphism_x_to_minus_x() {
        let curve = MontgomeryCurve::<Fp1Element>::e0();
        let a24 = curve.a24();
        let (p, q, pmq) = basis_e0_lvl1();
        let f = 248usize; // TORSION_EVEN_POWER at lvl1

        let m_i = mat_from_limbs(&ACTION_I);
        let (r, s, _rmq) =
            matrix_application_even_basis(&p, &q, &pmq, &m_i, f, &a24).expect("biladder ok");

        // ι(P).x = −P.x; compare projectively: R.x · 1 == (−P.x) · R.z.
        let neg_px = p.x.negate();
        let neg_qx = q.x.negate();
        assert!(
            bool::from(r.x.ct_eq(&neg_px.mul(&r.z))),
            "action_i(P) must have x = −P.x (the i-endomorphism x↦−x)",
        );
        assert!(
            bool::from(s.x.ct_eq(&neg_qx.mul(&s.z))),
            "action_i(Q) must have x = −Q.x",
        );
    }

    /// Full bridge through the `make_primitive` path: applying the
    /// endomorphism `θ = i` to the basis must give `x ↦ −x`. This exercises
    /// `quat_make_primitive_o0` → matrix build → `matrix_application`, i.e. the
    /// complete `endomorphism_application_even_basis`, confirming the coeff
    /// ordering routes `i` to `GEN2 = action_i`.
    #[test]
    fn endomorphism_application_of_i_is_x_to_minus_x() {
        use crate::quaternion::Quaternion;
        use crypto_bigint::Int;

        let curve = MontgomeryCurve::<Fp1Element>::e0();
        let a24 = curve.a24();
        let (p, q, pmq) = basis_e0_lvl1();
        let f = 248usize;

        // θ = i  (standard coords a=0, b=1, c=0, d=0)
        let theta_i = Quaternion::<8>::new(
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(1),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        );
        let (r, s, _rmq) =
            endomorphism_application_even_basis::<Level1, 8>(&p, &q, &pmq, &theta_i, f, &a24)
                .expect("ok");

        assert!(
            bool::from(r.x.ct_eq(&p.x.negate().mul(&r.z))),
            "endo(i)(P).x must be −P.x",
        );
        assert!(
            bool::from(s.x.ct_eq(&q.x.negate().mul(&s.z))),
            "endo(i)(Q).x must be −Q.x",
        );
    }

    /// The identity endomorphism `θ = 1` leaves the basis fixed: `endo(1)(P) =
    /// P`, `endo(1)(Q) = Q` (matrix = identity).
    #[test]
    fn endomorphism_application_of_one_is_identity() {
        use crate::quaternion::Quaternion;
        use crypto_bigint::Int;

        let curve = MontgomeryCurve::<Fp1Element>::e0();
        let a24 = curve.a24();
        let (p, q, pmq) = basis_e0_lvl1();
        let f = 248usize;

        let theta_one = Quaternion::<8>::new(
            Int::<8>::from_i64(1),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        );
        let (r, s, _rmq) =
            endomorphism_application_even_basis::<Level1, 8>(&p, &q, &pmq, &theta_one, f, &a24)
                .expect("ok");

        // x(R) == x(P), x(S) == x(Q) projectively.
        assert!(
            bool::from(r.x.ct_eq(&p.x.mul(&r.z))),
            "endo(1)(P) must equal P",
        );
        assert!(
            bool::from(s.x.ct_eq(&q.x.mul(&s.z))),
            "endo(1)(Q) must equal Q",
        );
    }

    /// GENERIC-INDEX GROUND TRUTH (item 6): on every alternate starting curve
    /// `k = 1..=6`, the identity endomorphism `θ = 1` must leave that curve's
    /// own even-torsion basis fixed. This anchors the whole `k ≥ 1` wiring of
    /// [`endomorphism_application_even_basis_indexed`] together: the alternate
    /// order decomposition (`make_primitive_over_alt_order`), the per-curve
    /// `action_gen2/3/4` tables, the matrix assembly, and
    /// `matrix_application_even_basis` — independently of any C oracle (byte
    /// exactness of a non-trivial θ defers to the keygen KAT, item 8).
    #[test]
    fn endomorphism_indexed_identity_fixes_each_alternate_basis() {
        use crate::quaternion::Quaternion;
        use crate::quaternion::curves_with_endomorphism as cwe;
        use crypto_bigint::Int;

        let f = 248usize; // TORSION_EVEN_POWER at lvl1
        let two = Fp2::<Fp1Element>::one().double();
        let four = two.double();

        let theta_one = Quaternion::<8>::new(
            Int::<8>::from_i64(1),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        );
        let denom_one = Int::<8>::ONE;

        let curves = [
            cwe::curve_with_endomorphism_0_l1(),
            cwe::curve_with_endomorphism_1_l1(),
            cwe::curve_with_endomorphism_2_l1(),
            cwe::curve_with_endomorphism_3_l1(),
            cwe::curve_with_endomorphism_4_l1(),
            cwe::curve_with_endomorphism_5_l1(),
        ];

        for (k, c) in curves.iter().enumerate() {
            let index_alternate_curve = k + 1; // C slot k+1 == Rust alt index k
            let p = MontgomeryPoint::<Fp1Element>::new(c.p_x, c.p_z);
            let q = MontgomeryPoint::<Fp1Element>::new(c.q_x, c.q_z);
            let pmq = MontgomeryPoint::<Fp1Element>::new(c.pmq_x, c.pmq_z);
            // a24 = (A + 2C)/(4C); the NICE curves have C = 1.
            let four_c_inv = four.mul(&c.curve_c).invert().unwrap_or(Fp2::zero());
            let a24 = c.curve_a.add(&two.mul(&c.curve_c)).mul(&four_c_inv);

            let (r, s, _rmq) = endomorphism_application_even_basis_indexed::<Level1>(
                &p,
                &q,
                &pmq,
                index_alternate_curve,
                &theta_one,
                &denom_one,
                f,
                &a24,
            )
            .unwrap_or_else(|| panic!("indexed endo(1) must succeed on curve k={k}"));

            assert!(
                bool::from(r.x.ct_eq(&p.x.mul(&r.z))),
                "endo(1)(P) must equal P on alternate curve k={k}",
            );
            assert!(
                bool::from(s.x.ct_eq(&q.x.mul(&s.z))),
                "endo(1)(Q) must equal Q on alternate curve k={k}",
            );
        }
    }

    /// The O_0-coords entry (the form `RepresentInteger` returns) routes `i`
    /// correctly: `i` in O_0 coords is `(0, 1, 0, 0)`, and applying it must
    /// give `x ↦ −x` — same result as the standard-coords path, confirming the
    /// shared core.
    #[test]
    fn endomorphism_application_o0_coords_of_i_is_x_to_minus_x() {
        use crypto_bigint::Int;

        let curve = MontgomeryCurve::<Fp1Element>::e0();
        let a24 = curve.a24();
        let (p, q, pmq) = basis_e0_lvl1();
        let f = 248usize;

        // i in O_0 coords (= standard_to_o0_basis of i = (0,1,0,0)).
        let i_o0 = [
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(1),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        ];
        let (r, s, _rmq) =
            endomorphism_application_o0_coords::<Level1, 8>(&p, &q, &pmq, &i_o0, f, &a24)
                .expect("ok");

        assert!(
            bool::from(r.x.ct_eq(&p.x.negate().mul(&r.z))),
            "endo_o0(i)(P).x must be −P.x",
        );
        assert!(
            bool::from(s.x.ct_eq(&q.x.negate().mul(&s.z))),
            "endo_o0(i)(Q).x must be −Q.x",
        );
    }

    /// The rational wrapper folds `s = invmod(denom·extra, 2^f)` into the
    /// endomorphism. Independent oracle: applying `θ` then `[s]` via a
    /// scalar ladder (post-multiply) must equal the fold-into-coords path
    /// — two different routes to `[s]·θ(P)`. Anchored on `θ = i`
    /// (`i`-endomorphism), `denom = 3`, `extra = 5`, so `s = invmod(15)`.
    #[test]
    fn rational_endomorphism_equals_scalar_times_plain() {
        use crate::quaternion::Quaternion;
        use crypto_bigint::{Int, Uint};

        let curve = MontgomeryCurve::<Fp1Element>::e0();
        let a24 = curve.a24();
        let (p, q, pmq) = basis_e0_lvl1();
        let f = 248usize;

        // θ = i in standard coords.
        let i_quat = Quaternion::<8>::new(
            Int::from_i64(0),
            Int::from_i64(1),
            Int::from_i64(0),
            Int::from_i64(0),
        );
        let denom = Uint::<8>::from_u64(3);
        let extra = Uint::<8>::from_u64(5);

        // Plain i-endomorphism images, then post-multiply by [s].
        let (base_r, base_s, base_rmq) =
            endomorphism_application_even_basis::<Level1, 8>(&p, &q, &pmq, &i_quat, f, &a24)
                .expect("plain i-endo");
        let f_u32 = u32::try_from(f).expect("f fits in u32");
        let modulus = (Uint::<8>::MAX >> (512 - f_u32)).wrapping_add(&Uint::<8>::ONE); // 2^f
        let de = Uint::<8>::from_u64(15) & (Uint::<8>::MAX >> (512 - f_u32));
        let s = crate::quaternion::sign_orchestration::uint_inv_mod_vartime::<8>(&de, &modulus)
            .expect("15 invertible mod 2^f");
        // ladder reads the scalar little-endian (byte[0] = LSB), like
        // ec_biscalar_mul — the doc comment saying "big-endian" is wrong.
        let s_le = s.to_le_bytes();
        let exp_r = base_r.ladder(&s_le, &a24);
        let exp_s = base_s.ladder(&s_le, &a24);
        let exp_rmq = base_rmq.ladder(&s_le, &a24);

        // Fold-into-coords path.
        let (got_r, got_s, got_rmq) = endomorphism_application_rational_even_basis::<Level1, 8>(
            &p, &q, &pmq, &i_quat, &denom, &extra, f, &a24,
        )
        .expect("rational i-endo");

        // Projective x equality: x1·z2 == x2·z1.
        let xeq = |a: &MontgomeryPoint<Fp1Element>, b: &MontgomeryPoint<Fp1Element>| {
            bool::from(a.x.mul(&b.z).ct_eq(&b.x.mul(&a.z)))
        };
        assert!(xeq(&got_r, &exp_r), "rational(P) == [s]·(i-endo P)");
        assert!(xeq(&got_s, &exp_s), "rational(Q) == [s]·(i-endo Q)");
        assert!(xeq(&got_rmq, &exp_rmq), "rational(PmQ) == [s]·(i-endo PmQ)");
    }
}
