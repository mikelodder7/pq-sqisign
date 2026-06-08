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
//! in via [`ConstMontyForm::from_montgomery`], no conversion.

use crate::ec::montgomery::MontgomeryPoint;
use crate::gf::fp2::Fp2;
use crate::params::lvl1::Fp1Element;
use crypto_bigint::U256;

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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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

/// Low `f`-bit mask `2^f − 1` as a `U256`.
#[inline]
#[allow(clippy::cast_possible_truncation)] // 1 ≤ f ≤ 256 ⇒ 256−f fits u32
fn mask_2f(f: usize) -> U256 {
    U256::MAX.wrapping_shr((256 - f) as u32)
}

/// `(a − b) mod 2^f`.
#[inline]
fn sub_mod_2f(a: &U256, b: &U256, f: usize) -> U256 {
    a.wrapping_sub(b) & mask_2f(f)
}

/// Apply a 2×2 integer matrix `m` to an x-only torsion basis `(P, Q, P−Q)` of
/// order `2^f`, in place: returns `(R, S, R−S)` where `R = [m00]P + [m10]Q`,
/// `S = [m01]P + [m11]Q`, `R−S = [m00−m01]P + [m10−m11]Q` (all mod `2^f`).
///
/// Port of the C reference `matrix_application_even_basis` (id2iso.c). `a24` is
/// the affine doubling constant `(A + 2)/4`. Returns `None` if any biladder
/// fails. `m` entries are taken mod `2^f`.
#[allow(dead_code)]
pub(crate) fn matrix_application_even_basis(
    p: &MontgomeryPoint<Fp1Element>,
    q: &MontgomeryPoint<Fp1Element>,
    pmq: &MontgomeryPoint<Fp1Element>,
    m: &[[U256; 2]; 2],
    f: usize,
    a24: &Fp2<Fp1Element>,
) -> Option<(
    MontgomeryPoint<Fp1Element>,
    MontgomeryPoint<Fp1Element>,
    MontgomeryPoint<Fp1Element>,
)> {
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

/// Build a `[[U256;2];2]` matrix from a `[[[u64;4];2];2]` limb table.
#[inline]
fn mat_from_limbs(t: &[[[u64; 4]; 2]; 2]) -> [[U256; 2]; 2] {
    [
        [U256::from_words(t[0][0]), U256::from_words(t[0][1])],
        [U256::from_words(t[1][0]), U256::from_words(t[1][1])],
    ]
}

/// `(a + b) mod 2^f`.
#[inline]
fn add_mod_2f(a: &U256, b: &U256, f: usize) -> U256 {
    a.wrapping_add(b) & mask_2f(f)
}

/// `(a · b) mod 2^f` (operands already `< 2^f ≤ 2^256`, so the low 256 bits of
/// the product reduced mod `2^f` is exact).
#[inline]
fn mul_mod_2f(a: &U256, b: &U256, f: usize) -> U256 {
    a.wrapping_mul(b) & mask_2f(f)
}

/// Reduce a signed quaternion-side `Int<L>` to `U256` modulo `2^f`
/// (two's-complement-correct: negatives map to `2^f − |x|`).
#[inline]
fn int_to_mod_2f<const L: usize>(x: &crypto_bigint::Int<L>, f: usize) -> U256 {
    let mag = x.abs(); // Uint<L>
    let w = mag.to_words();
    // low 256 bits of |x| (f ≤ 248 < 256, so this captures |x| mod 2^f)
    let lo = U256::from_words([w[0], w[1], w[2], w[3]]) & mask_2f(f);
    if bool::from(x.is_negative()) && lo != U256::ZERO {
        sub_mod_2f(&U256::ZERO, &lo, f) // 2^f − |x|
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
#[allow(dead_code)]
pub(crate) fn endomorphism_application_even_basis<const L: usize>(
    p: &MontgomeryPoint<Fp1Element>,
    q: &MontgomeryPoint<Fp1Element>,
    pmq: &MontgomeryPoint<Fp1Element>,
    theta: &crate::quaternion::Quaternion<L>,
    f: usize,
    a24: &Fp2<Fp1Element>,
) -> Option<(
    MontgomeryPoint<Fp1Element>,
    MontgomeryPoint<Fp1Element>,
    MontgomeryPoint<Fp1Element>,
)> {
    let o0 = crate::quaternion::o0_mul::standard_to_o0_basis(theta);
    endomorphism_application_o0_coords::<L>(p, q, pmq, &o0, f, a24)
}

/// Same as [`endomorphism_application_even_basis`] but takes the endomorphism
/// already in `O_0`-basis coordinates (the form `RepresentInteger` returns),
/// skipping the standard→O_0 conversion.
#[allow(dead_code)]
pub(crate) fn endomorphism_application_o0_coords<const L: usize>(
    p: &MontgomeryPoint<Fp1Element>,
    q: &MontgomeryPoint<Fp1Element>,
    pmq: &MontgomeryPoint<Fp1Element>,
    o0_coords: &[crypto_bigint::Int<L>; 4],
    f: usize,
    a24: &Fp2<Fp1Element>,
) -> Option<(
    MontgomeryPoint<Fp1Element>,
    MontgomeryPoint<Fp1Element>,
    MontgomeryPoint<Fp1Element>,
)> {
    let (primitive, content) =
        crate::quaternion::o0_mul::make_primitive_from_o0_coords::<L>(o0_coords);

    let c: [U256; 4] = [
        int_to_mod_2f(&primitive[0], f),
        int_to_mod_2f(&primitive[1], f),
        int_to_mod_2f(&primitive[2], f),
        int_to_mod_2f(&primitive[3], f),
    ];
    let content_u = int_to_mod_2f(&content, f);

    // GEN2 == ACTION_I (O0 basis element index 1 is i).
    let gen2 = mat_from_limbs(&ACTION_I);
    let gen3 = mat_from_limbs(&ACTION_GEN3);
    let gen4 = mat_from_limbs(&ACTION_GEN4);

    let mut m = [[U256::ZERO; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            // diagonal carries coeffs0 (the identity component)
            let mut e = if i == j { c[0] } else { U256::ZERO };
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
/// `[[U256;2];2]` mod `2^f`.
#[inline]
fn mat_int8_to_mod_2f(t: &[[crypto_bigint::Int<8>; 2]; 2], f: usize) -> [[U256; 2]; 2] {
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
/// Index mapping (Rust tables are offset-by-one from the C arrays, per S343):
/// `index_alternate_curve == 0` is the standard order `O_0` — delegated to the
/// validated [`endomorphism_application_even_basis`] (which assumes an integer
/// `theta`, i.e. `theta_denom == 1`); `index_alternate_curve == k ≥ 1` uses the
/// alternate extremal order `alternate_extremal_order_{k-1}_l1` and curve
/// `curve_with_endomorphism_{k-1}_l1` (both = C array slot `k`).
///
/// Returns `None` if `theta/theta_denom ∉ EXTREMAL_ORDERS[k]` (the C `assert`
/// fails), if the index is out of range, or if a biladder fails.
///
/// Byte-exact correctness of the `k ≥ 1` matrix is NOT independently checkable
/// in-tree (no golden vectors); it is anchored here by the identity
/// endomorphism (`θ = 1 ⇒ basis fixed`) and proven end-to-end by the eventual
/// keygen KAT (item 8).
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn endomorphism_application_even_basis_indexed(
    p: &MontgomeryPoint<Fp1Element>,
    q: &MontgomeryPoint<Fp1Element>,
    pmq: &MontgomeryPoint<Fp1Element>,
    index_alternate_curve: usize,
    theta: &crate::quaternion::Quaternion<8>,
    theta_denom: &crypto_bigint::Int<8>,
    f: usize,
    a24: &Fp2<Fp1Element>,
) -> Option<(
    MontgomeryPoint<Fp1Element>,
    MontgomeryPoint<Fp1Element>,
    MontgomeryPoint<Fp1Element>,
)> {
    use crate::quaternion::curves_with_endomorphism as cwe;
    use crate::quaternion::extremal_orders as eo;

    if index_alternate_curve == 0 {
        // Standard O_0 path (validated): integer theta only.
        debug_assert!(
            *theta_denom == crypto_bigint::Int::<8>::ONE,
            "index 0 (O_0) path requires integer theta (denom = 1)",
        );
        return endomorphism_application_even_basis::<8>(p, q, pmq, theta, f, a24);
    }

    let k = index_alternate_curve - 1;
    let order = match k {
        0 => eo::alternate_extremal_order_0_l1(),
        1 => eo::alternate_extremal_order_1_l1(),
        2 => eo::alternate_extremal_order_2_l1(),
        3 => eo::alternate_extremal_order_3_l1(),
        4 => eo::alternate_extremal_order_4_l1(),
        5 => eo::alternate_extremal_order_5_l1(),
        _ => return None,
    };
    let curve = match k {
        0 => cwe::curve_with_endomorphism_0_l1(),
        1 => cwe::curve_with_endomorphism_1_l1(),
        2 => cwe::curve_with_endomorphism_2_l1(),
        3 => cwe::curve_with_endomorphism_3_l1(),
        4 => cwe::curve_with_endomorphism_4_l1(),
        5 => cwe::curve_with_endomorphism_5_l1(),
        _ => return None,
    };

    // Decompose theta/theta_denom over EXTREMAL_ORDERS[k] (C quat_alg_make_primitive):
    // coeffs in the order basis (index 0 = identity component), content odd.
    let (coeffs, content) = eo::make_primitive_over_alt_order(&order, theta, theta_denom)?;

    let c: [U256; 4] = [
        int_to_mod_2f(&coeffs[0], f),
        int_to_mod_2f(&coeffs[1], f),
        int_to_mod_2f(&coeffs[2], f),
        int_to_mod_2f(&coeffs[3], f),
    ];
    let content_u = int_to_mod_2f(&content, f);

    let gen2 = mat_int8_to_mod_2f(&curve.action_gen2, f);
    let gen3 = mat_int8_to_mod_2f(&curve.action_gen3, f);
    let gen4 = mat_int8_to_mod_2f(&curve.action_gen4, f);

    let mut m = [[U256::ZERO; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            // diagonal carries coeffs0 (the identity component)
            let mut e = if i == j { c[0] } else { U256::ZERO };
            e = add_mod_2f(&e, &mul_mod_2f(&c[1], &gen2[i][j], f), f);
            e = add_mod_2f(&e, &mul_mod_2f(&c[2], &gen3[i][j], f), f);
            e = add_mod_2f(&e, &mul_mod_2f(&c[3], &gen4[i][j], f), f);
            e = mul_mod_2f(&e, &content_u, f);
            m[i][j] = e;
        }
    }

    matrix_application_even_basis(p, q, pmq, &m, f, a24)
}

/// Reduce a `Uint<L>` to `U256` modulo `2^f` (low `f` bits).
#[inline]
fn uint_to_mod_2f<const L: usize>(x: &crypto_bigint::Uint<L>, f: usize) -> U256 {
    let w = x.to_words();
    U256::from_words([w[0], w[1], w[2], w[3]]) & mask_2f(f)
}

/// Embed a `U256` value `< 2^f` (`f ≤ 248`) as a non-negative `Int<L>`.
#[inline]
fn u256_to_int<const L: usize>(x: &U256) -> crypto_bigint::Int<L> {
    let xw = x.to_words();
    let mut words = [0u64; L];
    words[..4].copy_from_slice(&xw);
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
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn endomorphism_application_rational_even_basis<const L: usize>(
    p: &MontgomeryPoint<Fp1Element>,
    q: &MontgomeryPoint<Fp1Element>,
    pmq: &MontgomeryPoint<Fp1Element>,
    num: &crate::quaternion::Quaternion<L>,
    denom: &crypto_bigint::Uint<L>,
    extra: &crypto_bigint::Uint<L>,
    f: usize,
    a24: &Fp2<Fp1Element>,
) -> Option<(
    MontgomeryPoint<Fp1Element>,
    MontgomeryPoint<Fp1Element>,
    MontgomeryPoint<Fp1Element>,
)> {
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
    let modulus = mask_2f(f).wrapping_add(&U256::ONE); // 2^f (f ≤ 248)
    let s = crate::quaternion::sign_orchestration::uint_inv_mod_vartime::<4>(&de, &modulus)?;

    let two_t = *crypto_bigint::Uint::<L>::ONE.shl_vartime(t).as_int(); // 2^t
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

    endomorphism_application_o0_coords::<L>(p, q, pmq, &scaled, f, a24)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ec::jacobian::JacobianPoint;
    use crate::ec::montgomery::MontgomeryCurve;
    use subtle::ConstantTimeEq;

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
            endomorphism_application_even_basis(&p, &q, &pmq, &theta_i, f, &a24).expect("ok");

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
            endomorphism_application_even_basis(&p, &q, &pmq, &theta_one, f, &a24).expect("ok");

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

            let (r, s, _rmq) = endomorphism_application_even_basis_indexed(
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
            endomorphism_application_o0_coords(&p, &q, &pmq, &i_o0, f, &a24).expect("ok");

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
        use crypto_bigint::{Int, U256, Uint};

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
            endomorphism_application_even_basis::<8>(&p, &q, &pmq, &i_quat, f, &a24)
                .expect("plain i-endo");
        let modulus = (U256::MAX >> (256 - f as u32)).wrapping_add(&U256::ONE); // 2^f
        let de = U256::from_u64(15) & (U256::MAX >> (256 - f as u32));
        let s = crate::quaternion::sign_orchestration::uint_inv_mod_vartime::<4>(&de, &modulus)
            .expect("15 invertible mod 2^f");
        // ladder reads the scalar little-endian (byte[0] = LSB), like
        // ec_biscalar_mul — the doc comment saying "big-endian" is wrong.
        let s_le = s.to_le_bytes();
        let exp_r = base_r.ladder(&s_le, &a24);
        let exp_s = base_s.ladder(&s_le, &a24);
        let exp_rmq = base_rmq.ladder(&s_le, &a24);

        // Fold-into-coords path.
        let (got_r, got_s, got_rmq) = endomorphism_application_rational_even_basis::<8>(
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
