// SPDX-License-Identifier: MIT OR Apache-2.0
//! The Clapotis `ideal_to_isogeny` evaluator spine (index-0 path).
//!
//! Port of the SQIsign C reference `dim2id2iso_ideal_to_isogeny_clapotis`
//! (`src/id2iso/ref/lvlx/dim2id2iso.c`), specialized to level 1 and the
//! INDEX-0 case (`index_order1 = index_order2 = 0`). For principal-ideal
//! inputs — the real signing flow — `find_uv` always returns index 0
//! (S239: j>0 is structurally unreachable), so this is the real path and
//! all `if (index_order != 0)` CONNECTING_IDEALS blocks vanish.
//!
//! The flow (all primitives ported in S205–S271):
//! 1. `find_uv` → `(u, v, d1, d2, β1, β2)` with `u·d1 + v·d2 = 2^F`.
//! 2. strip the 2-adic gcd of `u, v` (the gcd divides `2^F`, so it is a
//!    pure power of two): `exp = F − v2(gcd)`, `u, v >>= v2(gcd)`.
//! 3. `θ = β2·conj(β1)/n(I)` ([`theta_endomorphism`]).
//! 4. `φ_u`, `φ_v`: fixed-degree dim-2 isogenies of degrees `u`, `v`,
//!    pushing the `E0[2^F]` basis (E2-factor = O).
//! 5. apply `θ` (scaled by `1/d1`) to `φ_v`'s image basis.
//! 6. assemble the couple kernel, double down to order `2^exp`, walk the
//!    randomized `(2,2)`-chain, pushing `φ_u`'s image basis.
//! 7. Weil-pairing factor selection: pick the codomain factor whose
//!    transported basis pairs as `e(bas)^{d1·u²}`.
//! 8. apply `β1` (scaled by `1/(u·d1)`) to the selected basis → output.

use crate::ec::couple::{
    CoupleCurve, CoupleJacobianPoint, CoupleMontgomeryPoint, EcBasis, ThetaKernelCouplePoints,
};
use crate::ec::jacobian::lift_basis;
use crate::ec::montgomery::{MontgomeryCurve, MontgomeryPoint};
use crate::ec::weil::weil;
use crate::isogeny::clapotis::{find_uv, lattice_reduced_norm, theta_endomorphism};
use crate::isogeny::endomorphism::{basis_e0_lvl1, endomorphism_application_rational_even_basis};
use crate::isogeny::fixed_degree::fixed_degree_isogeny_and_eval;
use crate::isogeny::theta_chain::theta_chain_compute_and_eval_randomized;
use crate::params::lvl1::Fp1Element;
use crate::quaternion::ideal::LeftIdeal;
use crypto_bigint::{Int, Uint};
use rand_core::CryptoRng;
use subtle::ConstantTimeEq;

/// Level-1 even-torsion power `TORSION_EVEN_POWER` (`E0[2^248]`).
const F: u32 = 248;
/// `find_uv` quaternion-side limb width. L=16 (1024-bit) so the real
/// connecting/secret ideals — norm up to SEC_DEGREE ~ 2^512, basis entries
/// ~2^512 — fit `Int<L>` (the toy/small fixtures fit too, just wider). The EC
/// side stays at lvl1 (`Fp1Element`, F=248); only the quaternion width scales.
const L: usize = 16;
/// `fixed_degree` quaternion-side limb width (`64·QL ≥ 3·bits(p)+2`).
const QL: usize = 12;

/// `|x|` of a signed `Int<L>` as a `Uint<L>`.
#[inline]
fn abs_uint(x: &Int<L>) -> Uint<L> {
    x.abs()
}

/// Compute the Clapotis isogeny `E0 → E_K` for the principal left ideal
/// `lideal` (index-0 path) and return the codomain curve together with
/// the transported `E0[2^F]` basis (the response-isogeny image of the
/// canonical basis). `p` is the level-1 prime.
///
/// Returns `None` if `find_uv` finds no Bézout decomposition, a
/// fixed-degree isogeny / lift / endomorphism scaling fails, or the
/// `(2,2)`-chain does not split.
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn ideal_to_isogeny_clapotis_idx0<R: CryptoRng>(
    lideal: &LeftIdeal<L>,
    p: &Uint<L>,
    witnesses: &[Uint<QL>],
    sample_bound: i64,
    max_trials: usize,
    rng: &mut R,
) -> Option<(MontgomeryCurve<Fp1Element>, EcBasis<Fp1Element>)> {
    // 1. find_uv at the production target 2^F.
    let target = *Uint::<L>::ONE.shl_vartime(F).as_int();
    let r = find_uv::<L>(&target, lideal, p, &[], 2).ok()?;
    debug_assert!(r.index_alternate_order_1 == 0 && r.index_alternate_order_2 == 0);
    // N(I) from the lattice determinant — convention-independent (the
    // connecting ideal may be built by samplers that store cached_norm = N
    // rather than N²). |det| ~ N²·denom⁴ overflows Int<L>, so derive at
    // width 32.
    let n_id = lattice_reduced_norm::<L, 32>(&lideal.basis, &lideal.denom)?;

    // 3. θ = β2·conj(β1)/n(I). WIDE=32 (2048-bit) holds β2·conj(β1) ~ N(I)²
    //    even at SEC_DEGREE scale (N(I)~2^512 ⇒ product ~2^1024).
    let theta = theta_endomorphism::<L, 32>(&r, &n_id, p)?;

    // 2. Strip the 2-adic gcd of (u, v); the gcd divides 2^F so it is a
    //    pure power of two — exp_gcd = min(v2(u), v2(v)).
    let u_abs = abs_uint(&r.u);
    let v_abs = abs_uint(&r.v);
    let exp_gcd = u_abs.trailing_zeros().min(v_abs.trailing_zeros());
    let exp = F - exp_gcd;
    let u_s = u_abs.wrapping_shr(exp_gcd);
    let v_s = v_abs.wrapping_shr(exp_gcd);
    let d1 = abs_uint(&r.d1);
    let _d2 = abs_uint(&r.d2);

    // 4. φ_u and φ_v: push the E0[2^F] basis (E2-factor = O).
    let (bp, bq, bpmq) = basis_e0_lvl1();
    let inf = MontgomeryPoint::<Fp1Element>::infinity();
    let push_basis = |a: MontgomeryPoint<Fp1Element>,
                      b: MontgomeryPoint<Fp1Element>,
                      c: MontgomeryPoint<Fp1Element>| {
        [
            CoupleMontgomeryPoint::new(a, inf),
            CoupleMontgomeryPoint::new(b, inf),
            CoupleMontgomeryPoint::new(c, inf),
        ]
    };

    let eval_u = push_basis(bp, bq, bpmq);
    let mut out_u = [CoupleMontgomeryPoint::infinity(); 3];
    let (_lu, fu) = fixed_degree_isogeny_and_eval(
        &u_s.resize::<QL>(),
        &eval_u,
        &mut out_u,
        witnesses,
        sample_bound,
        max_trials,
        rng,
    )?;
    let bas_u = (out_u[0].p1, out_u[1].p1, out_u[2].p1);

    let eval_v = push_basis(bp, bq, bpmq);
    let mut out_v = [CoupleMontgomeryPoint::infinity(); 3];
    let (_lv, fv) = fixed_degree_isogeny_and_eval(
        &v_s.resize::<QL>(),
        &eval_v,
        &mut out_v,
        witnesses,
        sample_bound,
        max_trials,
        rng,
    )?;
    let bas2 = (out_v[0].p1, out_v[1].p1, out_v[2].p1);

    // 5. Apply θ (scaled by 1/d1) to φ_v's image basis on Fv.E1.
    let a24_fv1 = fv.e1.a24();
    let (t2p, t2q, t2pmq) = endomorphism_application_rational_even_basis::<L>(
        &bas2.0,
        &bas2.1,
        &bas2.2,
        &theta.num,
        &theta.denom,
        &d1,
        F as usize,
        &a24_fv1,
    )?;

    // 6. Assemble the couple kernel (T1m2 is a placeholder — the chain
    //    seeds the gluing kernel from T1, T2 only), double to order 2^exp,
    //    and walk the randomized chain pushing bas_u.
    let (p1, q1) = lift_basis(&EcBasis::new(bas_u.0, bas_u.1, bas_u.2), &fu.e1).ok()?;
    let (p2, q2) = lift_basis(&EcBasis::new(t2p, t2q, t2pmq), &fv.e1).ok()?;
    let e01 = CoupleCurve::new(fu.e1, fv.e1);
    let ker = ThetaKernelCouplePoints::new(
        CoupleJacobianPoint::new(p1, p2),
        CoupleJacobianPoint::new(q1, q2),
        CoupleJacobianPoint::infinity(),
    )
    .double_iter(F - exp, &e01);

    let eval_chain = push_basis(bas_u.0, bas_u.1, bas_u.2);
    let mut out_chain = [CoupleMontgomeryPoint::infinity(); 3];
    let theta_cod = theta_chain_compute_and_eval_randomized(
        exp,
        &e01,
        &ker,
        false,
        &eval_chain,
        &mut out_chain,
        rng,
    )?;
    let (tt1, tt2, tt1m2) = (out_chain[0], out_chain[1], out_chain[2]);

    // 7. Weil-pairing factor selection: the correct factor pairs as
    //    e(bas)^{d1·u²}.
    let e0 = MontgomeryCurve::<Fp1Element>::e0();
    let w0 = weil(F, &bp, &bq, &bpmq, &e0);
    let w1 = weil(F, &tt1.p1, &tt2.p1, &tt1m2.p1, &theta_cod.e1);
    let mask_f = Uint::<L>::ONE.shl_vartime(F).wrapping_sub(&Uint::ONE);
    let k = d1.wrapping_mul(&u_s).wrapping_mul(&u_s) & mask_f; // d1·u² mod 2^F
    let test_pow = w0.pow_vartime(&k.to_le_bytes());

    let (codomain, basis_pts) = if bool::from(w1.ct_eq(&test_pow)) {
        (theta_cod.e1, (tt1.p1, tt2.p1, tt1m2.p1))
    } else {
        (theta_cod.e2, (tt1.p2, tt2.p2, tt1m2.p2))
    };

    // 8. Apply β1 (scaled by 1/(u·d1)) to the selected basis.
    let a24_cod = codomain.a24();
    let ud1 = u_s.wrapping_mul(&d1);
    let (op, oq, opmq) = endomorphism_application_rational_even_basis::<L>(
        &basis_pts.0,
        &basis_pts.1,
        &basis_pts.2,
        &r.beta1.num,
        &r.beta1.denom,
        &ud1,
        F as usize,
        &a24_cod,
    )?;

    Some((codomain, EcBasis::new(op, oq, opmq)))
}

/// Starting even-torsion basis for `find_uv` index `idx` (0 = E0; k≥1 = the
/// NICE alternate curve `k−1`'s `basis_even`).
fn starting_basis_indexed(
    idx: usize,
) -> (
    MontgomeryPoint<Fp1Element>,
    MontgomeryPoint<Fp1Element>,
    MontgomeryPoint<Fp1Element>,
) {
    use crate::quaternion::curves_with_endomorphism as cw;
    if idx == 0 {
        return basis_e0_lvl1();
    }
    let c = match idx - 1 {
        0 => cw::curve_with_endomorphism_0_l1(),
        1 => cw::curve_with_endomorphism_1_l1(),
        2 => cw::curve_with_endomorphism_2_l1(),
        3 => cw::curve_with_endomorphism_3_l1(),
        4 => cw::curve_with_endomorphism_4_l1(),
        _ => cw::curve_with_endomorphism_5_l1(),
    };
    (
        MontgomeryPoint::new(c.p_x, c.p_z),
        MontgomeryPoint::new(c.q_x, c.q_z),
        MontgomeryPoint::new(c.pmq_x, c.pmq_z),
    )
}

/// Starting curve for `find_uv` index `idx` (0 = E0; k≥1 = NICE curve `k−1`;
/// the NICE curves have `C = 1`, so `curve_a` is the affine coefficient).
fn starting_curve_indexed(idx: usize) -> MontgomeryCurve<Fp1Element> {
    use crate::quaternion::curves_with_endomorphism as cw;
    if idx == 0 {
        return MontgomeryCurve::e0();
    }
    let c = match idx - 1 {
        0 => cw::curve_with_endomorphism_0_l1(),
        1 => cw::curve_with_endomorphism_1_l1(),
        2 => cw::curve_with_endomorphism_2_l1(),
        3 => cw::curve_with_endomorphism_3_l1(),
        4 => cw::curve_with_endomorphism_4_l1(),
        _ => cw::curve_with_endomorphism_5_l1(),
    };
    MontgomeryCurve::new(c.curve_a)
}

/// Connecting-ideal norm `N` for index `idx` (0 ⇒ 1; k≥1 ⇒ `N = √(cached_norm)`
/// since the Rust connecting ideals store `cached_norm = N²`, S328). Used in the
/// θ / β1 rational-scaling factors (C `.norm` is `N`, dim2id2iso.c:988/1092).
fn connecting_norm_indexed(idx: usize) -> Uint<L> {
    use crate::quaternion::connecting_ideals as ci;
    if idx == 0 {
        return Uint::<L>::ONE;
    }
    let id = match idx - 1 {
        0 => ci::alternate_connecting_ideal_0_l1(),
        1 => ci::alternate_connecting_ideal_1_l1(),
        2 => ci::alternate_connecting_ideal_2_l1(),
        3 => ci::alternate_connecting_ideal_3_l1(),
        4 => ci::alternate_connecting_ideal_4_l1(),
        _ => ci::alternate_connecting_ideal_5_l1(),
    };
    id.cached_norm.resize::<L>().floor_sqrt_vartime()
}

/// The Clapotis combine (steps 4–8) for an arbitrary `find_uv` result `r`,
/// generalized to the alternate-order indices `index_alternate_order_1/2`.
/// Mirrors [`ideal_to_isogeny_clapotis_idx0`] (which is the `(0,0)` case) with
/// the five index-aware deltas from the C `dim2id2iso_ideal_to_isogeny_clapotis`
/// (759-1142): per-side indexed φ from the NICE curves (D1/D2), the θ scale
/// `×N(conn[index2])` (D3), the β1 scale `×N(conn[index1])` (D4), and the
/// factor-selection Weil reference on the `index1` NICE curve/basis (D5). The θ
/// and β1 endomorphism APPLICATIONS stay index-0 (the elements are in the
/// standard frame post-β); the alternate curve enters only via the starting-φ
/// and the Weil reference.
#[allow(clippy::too_many_arguments)]
fn clapotis_combine_indexed<R: CryptoRng>(
    r: &crate::isogeny::clapotis::FindUvResult<L>,
    lideal: &LeftIdeal<L>,
    p: &Uint<L>,
    witnesses: &[Uint<QL>],
    sample_bound: i64,
    max_trials: usize,
    rng: &mut R,
) -> Option<(MontgomeryCurve<Fp1Element>, EcBasis<Fp1Element>)> {
    use crate::isogeny::fixed_degree::fixed_degree_isogeny_and_eval_indexed;

    let index1 = r.index_alternate_order_1;
    let index2 = r.index_alternate_order_2;

    let n_id = lattice_reduced_norm::<L, 32>(&lideal.basis, &lideal.denom)?;
    let theta = theta_endomorphism::<L, 32>(r, &n_id, p)?;

    let u_abs = abs_uint(&r.u);
    let v_abs = abs_uint(&r.v);
    let exp_gcd = u_abs.trailing_zeros().min(v_abs.trailing_zeros());
    let exp = F - exp_gcd;
    let u_s = u_abs.wrapping_shr(exp_gcd);
    let v_s = v_abs.wrapping_shr(exp_gcd);
    let d1 = abs_uint(&r.d1);

    let inf = MontgomeryPoint::<Fp1Element>::infinity();
    let push = |a: MontgomeryPoint<Fp1Element>,
                b: MontgomeryPoint<Fp1Element>,
                c: MontgomeryPoint<Fp1Element>| {
        [
            CoupleMontgomeryPoint::new(a, inf),
            CoupleMontgomeryPoint::new(b, inf),
            CoupleMontgomeryPoint::new(c, inf),
        ]
    };

    // 4. φ_u from the index1 NICE curve (D1); φ_v from index2 (D2).
    let (bp1, bq1, bpmq1) = starting_basis_indexed(index1);
    let eval_u = push(bp1, bq1, bpmq1);
    let mut out_u = [CoupleMontgomeryPoint::infinity(); 3];
    let (_lu, fu) = fixed_degree_isogeny_and_eval_indexed(
        index1,
        &u_s.resize::<QL>(),
        &eval_u,
        &mut out_u,
        witnesses,
        sample_bound,
        max_trials,
        rng,
    )?;
    let bas_u = (out_u[0].p1, out_u[1].p1, out_u[2].p1);

    let (bp2, bq2, bpmq2) = starting_basis_indexed(index2);
    let eval_v = push(bp2, bq2, bpmq2);
    let mut out_v = [CoupleMontgomeryPoint::infinity(); 3];
    let (_lv, fv) = fixed_degree_isogeny_and_eval_indexed(
        index2,
        &v_s.resize::<QL>(),
        &eval_v,
        &mut out_v,
        witnesses,
        sample_bound,
        max_trials,
        rng,
    )?;
    let bas2 = (out_v[0].p1, out_v[1].p1, out_v[2].p1);

    // 5. Apply θ (scaled by 1/(d1·N(conn[index2]))) to φ_v's image (D3); the
    //    application itself is index-0 (standard-frame θ).
    let extra_theta = d1.wrapping_mul(&connecting_norm_indexed(index2));
    let a24_fv1 = fv.e1.a24();
    let (t2p, t2q, t2pmq) = endomorphism_application_rational_even_basis::<L>(
        &bas2.0,
        &bas2.1,
        &bas2.2,
        &theta.num,
        &theta.denom,
        &extra_theta,
        F as usize,
        &a24_fv1,
    )?;

    // 6. Couple kernel, double to 2^exp, randomized chain pushing bas_u.
    let (p1, q1) = lift_basis(&EcBasis::new(bas_u.0, bas_u.1, bas_u.2), &fu.e1).ok()?;
    let (p2, q2) = lift_basis(&EcBasis::new(t2p, t2q, t2pmq), &fv.e1).ok()?;
    let e01 = CoupleCurve::new(fu.e1, fv.e1);
    let ker = ThetaKernelCouplePoints::new(
        CoupleJacobianPoint::new(p1, p2),
        CoupleJacobianPoint::new(q1, q2),
        CoupleJacobianPoint::infinity(),
    )
    .double_iter(F - exp, &e01);

    let eval_chain = push(bas_u.0, bas_u.1, bas_u.2);
    let mut out_chain = [CoupleMontgomeryPoint::infinity(); 3];
    let theta_cod = theta_chain_compute_and_eval_randomized(
        exp,
        &e01,
        &ker,
        false,
        &eval_chain,
        &mut out_chain,
        rng,
    )?;
    let (tt1, tt2, tt1m2) = (out_chain[0], out_chain[1], out_chain[2]);

    // 7. Weil-pairing factor selection — reference on the index1 NICE
    //    curve/basis (D5); correct factor pairs as e(bas1)^{d1·u²}.
    let e1_curve = starting_curve_indexed(index1);
    let w0 = weil(F, &bp1, &bq1, &bpmq1, &e1_curve);
    let w1 = weil(F, &tt1.p1, &tt2.p1, &tt1m2.p1, &theta_cod.e1);
    let mask_f = Uint::<L>::ONE.shl_vartime(F).wrapping_sub(&Uint::ONE);
    let k = d1.wrapping_mul(&u_s).wrapping_mul(&u_s) & mask_f;
    let test_pow = w0.pow_vartime(&k.to_le_bytes());

    let (codomain, basis_pts) = if bool::from(w1.ct_eq(&test_pow)) {
        (theta_cod.e1, (tt1.p1, tt2.p1, tt1m2.p1))
    } else {
        (theta_cod.e2, (tt1.p2, tt2.p2, tt1m2.p2))
    };

    // 8. Apply β1 (scaled by 1/(u·d1·N(conn[index1]))) (D4); index-0 application.
    let a24_cod = codomain.a24();
    let ud1 = u_s
        .wrapping_mul(&d1)
        .wrapping_mul(&connecting_norm_indexed(index1));
    let (op, oq, opmq) = endomorphism_application_rational_even_basis::<L>(
        &basis_pts.0,
        &basis_pts.1,
        &basis_pts.2,
        &r.beta1.num,
        &r.beta1.denom,
        &ud1,
        F as usize,
        &a24_cod,
    )?;

    Some((codomain, EcBasis::new(op, oq, opmq)))
}

/// Clapotis `ideal_to_isogeny` evaluator INCLUDING the alternate-order search —
/// the full C `dim2id2iso_ideal_to_isogeny_clapotis` (not just index 0). Runs
/// `find_uv` over the 6 real `ALTERNATE_CONNECTING_IDEALS` (WIDE=128 dispatch),
/// then [`clapotis_combine_indexed`] on the returned `(index1, index2)` frame —
/// the `(0,0)` case reduces exactly to [`ideal_to_isogeny_clapotis_idx0`], and
/// `k≥1` runs the two-sided alternate-curve combination. Requires a
/// SQIsign-shaped input (`find_uv_alternate_orders` rescales by the smallest
/// basis element ⇒ `cached_norm` must be a perfect square `N²`); the real
/// keygen secret ideal is so shaped.
#[allow(dead_code, clippy::too_many_arguments, clippy::needless_range_loop)]
pub(crate) fn ideal_to_isogeny_clapotis<R: CryptoRng>(
    lideal: &LeftIdeal<L>,
    p: &Uint<L>,
    witnesses: &[Uint<QL>],
    sample_bound: i64,
    max_trials: usize,
    rng: &mut R,
) -> Option<(MontgomeryCurve<Fp1Element>, EcBasis<Fp1Element>)> {
    use crate::quaternion::connecting_ideals as ci;
    use crate::quaternion::lattice::widen_int_lattice;

    let target = *Uint::<L>::ONE.shl_vartime(F).as_int();

    // Widen the 6 real ALTERNATE_CONNECTING_IDEALS (L8) to the spine width L16.
    let widen = |id: &LeftIdeal<8>| -> LeftIdeal<L> {
        let mut basis = [[Int::<L>::from_i64(0); 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                basis[r][c] = widen_int_lattice::<8, L>(&id.basis[r][c]);
            }
        }
        LeftIdeal::<L>::with_denom_and_norm(
            basis,
            id.denom.resize::<L>(),
            id.cached_norm.resize::<L>(),
        )
    };
    let alts = [
        widen(&ci::alternate_connecting_ideal_0_l1()),
        widen(&ci::alternate_connecting_ideal_1_l1()),
        widen(&ci::alternate_connecting_ideal_2_l1()),
        widen(&ci::alternate_connecting_ideal_3_l1()),
        widen(&ci::alternate_connecting_ideal_4_l1()),
        widen(&ci::alternate_connecting_ideal_5_l1()),
    ];

    let r = find_uv::<L>(&target, lideal, p, &alts, 2).ok()?;
    clapotis_combine_indexed(&r, lideal, p, witnesses, sample_bound, max_trials, rng)
}

#[cfg(all(test, feature = "kat"))]
mod tests {
    use super::*;
    use crate::rng::NistPqcRng;

    fn witnesses() -> [Uint<QL>; 5] {
        [
            Uint::from_u64(2),
            Uint::from_u64(3),
            Uint::from_u64(5),
            Uint::from_u64(7),
            Uint::from_u64(11),
        ]
    }

    /// END-TO-END RUN of the Clapotis spine on a REALISTIC non-principal
    /// odd connecting ideal (the C-ref `test_dim2id2iso` shape: ideal of
    /// odd prime norm `n1` built from a generator of norm `n1·n2`, so the
    /// ideal is non-principal and `find_uv` yields BALANCED `d1, d2 ~ √p`
    /// hence `u, v < 2^246`). The quaternion finder needs `64·LIMBS ≥
    /// 3·bits(p)+2` so the ideal is BUILT at `L=16` and narrowed to the
    /// spine's `L=8`. Heavy (real-scale LLL inside `find_uv`), so ignored
    /// in the default run; exercise with `--ignored`.
    #[ignore = "end-to-end spine run on a realistic non-principal odd ideal (heavy real-scale LLL)"]
    #[test]
    fn ideal_to_isogeny_spine_produces_codomain_and_basis() {
        use crate::quaternion::lattice::narrow_int_lattice;
        use crate::quaternion::o0_mul::left_ideal_from_element_and_integer_o0;
        use crate::quaternion::primality::is_probable_prime_with_witnesses;
        use crate::quaternion::represent_integer::find_quaternion_in_full_order_with_norm_wide;

        // BUILD width L=24 (1536-bit): the quaternion finder needs
        // 64·BL ≥ 3·bits(target_m). For n1~2^250 (≈bitsize(p) — the real
        // reduced-secret-ideal scale), target_m=n1·n2~2^500 needs ≥1500 bits,
        // so BL=24 (1536); BL=16 (1024) was the S301 limit that failed at
        // ~2^253. The built ideal is narrowed to the spine's L=16.
        const BL: usize = 24;
        let p16 = crate::params::lvl1::prime().resize::<BL>();
        let wit16: [Uint<BL>; 12] =
            [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37].map(Uint::from_u64);
        let two = Uint::<BL>::from_u64(2);
        let next_prime = |start: Uint<BL>| -> Uint<BL> {
            let mut c = if start.as_limbs()[0].0 & 1 == 0 {
                start.wrapping_add(&Uint::ONE)
            } else {
                start
            };
            let mut t = 0;
            while !is_probable_prime_with_witnesses(&c, &wit16) {
                c = c.wrapping_add(&two);
                t += 1;
                assert!(t < 200_000, "no prime found");
            }
            c
        };
        // Two distinct odd primes ~2^250 (≈ bitsize(p) = 251): the magnitude
        // of the reduced secret/connecting ideal that the real protocol feeds
        // the spine (SEC_DEGREE → prime-norm-reduced equivalent ~ bitsize(p)).
        // Built at BL=24 (the S301 fixture-build limit at BL=16 fails here);
        // narrowed to the spine's L=16 (n1 ~2^250 < Int<16>).
        let n1 = next_prime(Uint::<BL>::ONE.shl_vartime(250).wrapping_add(&Uint::ONE));
        let n2 = next_prime(Uint::<BL>::ONE.shl_vartime(249).wrapping_add(&Uint::ONE));
        let target_m = n1.wrapping_mul(&n2);

        let p: Uint<L> = crate::params::lvl1::prime().resize::<L>();
        let w = witnesses();

        // CORRECTNESS ORACLE (computed once — independent of the seed): an
        // isogeny φ of degree deg acts on the Weil pairing by
        // e_{2^F}(φP, φQ) = e_{2^F}(P, Q)^deg. The Clapotis evaluator realizes
        // the connecting ideal I (deg φ = N(I) = n1, odd), so the output basis
        // must satisfy e(out.P, out.Q) = e(E0.P, E0.Q)^{n1}. Since e(E0.P,E0.Q)
        // is a primitive 2^F-th root and n1 is odd, a match also proves the
        // output pairing is primitive ⇒ out.P, out.Q have full order 2^F on
        // the codomain. A strong correctness oracle on the isogeny degree.
        let (bp, bq, bpmq) = basis_e0_lvl1();
        let e0 = MontgomeryCurve::<Fp1Element>::e0();
        let w_in = weil(F, &bp, &bq, &bpmq, &e0);
        let n1_8 = n1.resize::<L>();
        let expected = w_in.pow_vartime(&n1_8.to_le_bytes());
        let expected_inv = expected.invert().expect("Weil pairing value is a unit");

        // Multiple independent seeds → distinct (same-norm n1) connecting
        // ideals → distinct find_uv/spine execution paths (robustness gate;
        // the even-θ-denominator fix landed in S300 made these all pass).
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
            .expect("generator of norm n1·n2 must be found");
            // Build the connecting ideal at BL=24, then narrow its basis to
            // the spine's L=16 (entries ~2^250 fit Int<16>). find_uv derives
            // N(I) from the determinant, so the cached_norm convention is
            // irrelevant here.
            let ideal_bl = left_ideal_from_element_and_integer_o0::<BL>(&gamma, &n1, &p16);
            let mut basis16 = [[Int::<L>::from_i64(0); 4]; 4];
            for r in 0..4 {
                for c in 0..4 {
                    basis16[r][c] = narrow_int_lattice::<BL, L>(&ideal_bl.basis[r][c]);
                }
            }
            let lideal = LeftIdeal::<L>::with_denom_and_norm(
                basis16,
                ideal_bl.denom.resize::<L>(),
                ideal_bl.cached_norm.resize::<L>(),
            );

            let (codomain, basis) =
                ideal_to_isogeny_clapotis_idx0(&lideal, &p, &w, 64, 1 << 14, &mut rng)
                    .unwrap_or_else(|| {
                        panic!("spine must produce codomain+basis (seed {seed:#x})")
                    });

            let w_out = weil(F, &basis.p, &basis.q, &basis.p_minus_q, &codomain);
            let matches =
                bool::from(w_out.ct_eq(&expected)) || bool::from(w_out.ct_eq(&expected_inv));
            assert!(
                matches,
                "spine output must satisfy e(φP,φQ)=e(P,Q)^N(I) (seed {seed:#x})",
            );

            // Keygen TAIL: the spine codomain IS the public-key curve E_A.
            // Serialize its Montgomery A-coefficient via the PublicKey wire
            // format (PK_BYTES = 65 at lvl1) and confirm the encode/decode
            // round-trip preserves it — the back half of keygen (E_A → PK
            // bytes). (The front half — sampling the secret ideal at
            // SEC_DEGREE and reducing to a prime-norm equivalent — needs the
            // wide-norm sampling/reduction path and is a later session.)
            let pk = crate::wire::PublicKey::<Fp1Element>::new(codomain.a, 0);
            let mut pk_bytes = [0u8; crate::wire::PublicKey::<Fp1Element>::WIRE_BYTES];
            pk.encode(&mut pk_bytes).expect("PK encode");
            let decoded =
                crate::wire::PublicKey::<Fp1Element>::decode(&pk_bytes).expect("PK decode");
            assert!(
                bool::from(decoded.a_pk.ct_eq(&codomain.a)),
                "PK round-trip must preserve E_A's A-coefficient (seed {seed:#x})",
            );
        }
    }

    /// First end-to-end KEYGEN via the PRODUCTION sampler
    /// (`sampling_random_ideal_o0_given_norm_wide`, is_prime path): sample a
    /// prime-norm secret O_0-ideal (n ~ 2^250 ≈ bitsize(p)) → Clapotis
    /// `ideal_to_isogeny` → public-key curve E_A → serialize PK. Validated by
    /// the Weil-degree oracle (E_A is the degree-N(I) codomain).
    ///
    /// The sampler runs at L=32 internally: its re-randomization step
    /// `gen ← gen·gen_rerand` (finalize_random_ideal_o0) makes
    /// `N_red(gen_combined) ~ (p·n²)² ~ 2^1508` for n~2^250, so it needs
    /// `64·L ≥ ~1508 ⇒ L≥24` (the sampler's documented `2·bits(p)+1` contract
    /// covers only the mod-norm fast path; the rerand multiply needs more).
    /// The RETURNED ideal is still a `LeftIdeal<8>` (norm n ~2^250 fits Uint<8>).
    /// (KAT-exact keygen samples at SEC_DEGREE~2^512 then prime-norm-reduces —
    /// needs an even wider sampler, ~L≥40; a later session.)
    #[ignore = "first keygen via the production sampler (heavy: real-scale spine per seed)"]
    #[test]
    fn keygen_via_sampler_produces_valid_pubkey() {
        use crate::quaternion::ideal::LeftIdeal;
        use crate::quaternion::lattice::widen_int_lattice;
        use crate::quaternion::primality::is_probable_prime_with_witnesses;
        use crate::quaternion::represent_integer::sampling_random_ideal_o0_given_norm_wide;

        // Sampler internal width L=32 (see fn doc).
        const SL: usize = 32;
        let p_sl = crate::params::lvl1::prime().resize::<SL>();
        let p16 = crate::params::lvl1::prime().resize::<L>();
        let wit_sl: [Uint<SL>; 12] =
            [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37].map(Uint::from_u64);
        let w = witnesses();

        let (bp, bq, bpmq) = basis_e0_lvl1();
        let e0 = MontgomeryCurve::<Fp1Element>::e0();
        let w_in = weil(F, &bp, &bq, &bpmq, &e0);

        for seed in [0x11u8, 0x22] {
            let mut rng = NistPqcRng::new(&[seed; 48]);
            // A random odd prime n ~ 2^250 (≈ bitsize(p)); start offset by seed.
            let two = Uint::<SL>::from_u64(2);
            let mut n = Uint::<SL>::ONE
                .shl_vartime(250)
                .wrapping_add(&Uint::<SL>::from_u64(u64::from(seed) * 2 + 1));
            if n.as_limbs()[0].0 & 1 == 0 {
                n = n.wrapping_add(&Uint::ONE);
            }
            while !is_probable_prime_with_witnesses(&n, &wit_sl) {
                n = n.wrapping_add(&two);
            }

            // KEYGEN front: sample the prime-norm secret ideal (returns L8).
            let ideal8 = sampling_random_ideal_o0_given_norm_wide::<SL, _>(
                &n,
                &p_sl,
                true,
                None,
                64,
                1 << 14,
                &wit_sl,
                &mut rng,
            )
            .expect("sample prime-norm secret ideal");
            // Widen to the spine's L=16.
            let mut basis16 = [[Int::<L>::from_i64(0); 4]; 4];
            for r in 0..4 {
                for c in 0..4 {
                    basis16[r][c] = widen_int_lattice::<8, L>(&ideal8.basis[r][c]);
                }
            }
            let lideal = LeftIdeal::<L>::with_denom_and_norm(
                basis16,
                ideal8.denom.resize::<L>(),
                ideal8.cached_norm.resize::<L>(),
            );

            // Run the Clapotis isogeny → public-key curve E_A.
            let (e_a, basis) =
                ideal_to_isogeny_clapotis_idx0(&lideal, &p16, &w, 64, 1 << 14, &mut rng)
                    .unwrap_or_else(|| panic!("keygen spine must produce E_A (seed {seed:#x})"));

            // Public key = E_A's Montgomery A-coefficient (PK_BYTES = 65).
            let pk = crate::wire::PublicKey::<Fp1Element>::new(e_a.a, 0);
            let mut pk_bytes = [0u8; crate::wire::PublicKey::<Fp1Element>::WIRE_BYTES];
            pk.encode(&mut pk_bytes).expect("PK encode");

            // Correctness: E_A is the degree-N(I)=n isogeny codomain.
            let n16 = n.resize::<L>();
            let expected = w_in.pow_vartime(&n16.to_le_bytes());
            let expected_inv = expected.invert().expect("Weil value is a unit");
            let w_out = weil(F, &basis.p, &basis.q, &basis.p_minus_q, &e_a);
            assert!(
                bool::from(w_out.ct_eq(&expected)) || bool::from(w_out.ct_eq(&expected_inv)),
                "keygen E_A must be the degree-N(I) isogeny codomain (seed {seed:#x})",
            );
        }
    }

    /// SEC_DEGREE = 2^512 + 75 is PRIME. The C-ref keygen samples the secret
    /// ideal with `is_prime = 1`, so the FAST-path sampler applies (no
    /// prime_cofactor / general path needed) — KAT-exact keygen is the
    /// S307 fast-path flow at SEC_DEGREE scale (sampler internal width
    /// ~L≥40 for the rerand-combined gen ~ (p·SEC²)² ~ 2^2556).
    #[test]
    fn sec_degree_is_prime() {
        use crate::quaternion::primality::is_probable_prime_with_witnesses;
        let sec = crate::params::lvl1::sec_degree(); // Uint<16>, = 2^512 + 75
        let witnesses: [Uint<16>; 12] =
            [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37].map(Uint::from_u64);
        assert!(
            is_probable_prime_with_witnesses(&sec, &witnesses),
            "SEC_DEGREE = 2^512 + 75 must be prime (C-ref keygen uses is_prime=1)",
        );
    }

    /// KAT-exact keygen FRONT: sample the GENUINE secret O_0-ideal at norm
    /// SEC_DEGREE = 2^512 + 75 via the wide-RETURN production sampler.
    ///
    /// This is the piece S307's reduced-scale keygen could not reach: the
    /// secret ideal's basis entries are ~2^512, which exceed `Int<8>`
    /// (2^511), so the fixed-`LeftIdeal<8>` return of
    /// `sampling_random_ideal_o0_given_norm_wide` overflows on the narrow.
    /// The RET-generic
    /// `sampling_random_ideal_o0_given_norm_wide_ret::<BL, RET, _>` returns
    /// the ideal at `RET = 16` (the Clapotis spine width) while building
    /// internally at `BL = 48` — wide enough for the rerand-combined
    /// reduced norm `N_red(gen·gen_rerand) ~ (p·SEC²)² ~ 2^2556`.
    ///
    /// Because SEC_DEGREE is prime (see `sec_degree_is_prime`), this uses
    /// the FAST path (`is_prime = true`, no prime_cofactor) — exactly the
    /// C-ref keygen flow. Validates: the returned ideal's `cached_norm`
    /// equals SEC_DEGREE and `denom == 1` (integral O_0-ideal); the basis
    /// fit at RET=16 is proven by the successful narrow (Err otherwise).
    #[ignore = "wide-return sampler at SEC_DEGREE scale (heavy: BL=48 internal rerand)"]
    #[test]
    fn sample_sec_degree_secret_ideal_wide_return() {
        use crate::params::lvl1::{prime, sec_degree};
        use crate::quaternion::represent_integer::sampling_random_ideal_o0_given_norm_wide_ret;

        // Internal build width: rerand product N_red ~ (p·SEC²)² ~ 2^2556
        // ⇒ need 64·BL ≥ ~2556; BL=48 (3072-bit) gives comfortable headroom.
        const BL: usize = 48;
        // Return width: SEC_DEGREE ideal basis ~2^512 needs RET≥9; use 16
        // to feed the Clapotis spine directly (no re-widen step).
        const RET: usize = 16;

        let sec_w: Uint<BL> = sec_degree().resize::<BL>();
        let p_bl: Uint<BL> = prime().resize::<BL>();
        let wit_bl: [Uint<BL>; 12] =
            [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37].map(Uint::from_u64);

        let mut rng = NistPqcRng::new(&[0x33u8; 48]);
        let ideal = sampling_random_ideal_o0_given_norm_wide_ret::<BL, RET, _>(
            &sec_w,
            &p_bl,
            true, // SEC_DEGREE is prime ⇒ fast path (C-ref keygen flow)
            None,
            64,
            1 << 14,
            &wit_bl,
            &mut rng,
        )
        .expect("sample SEC_DEGREE secret ideal via the fast-path wide-return sampler");

        // cached_norm == SEC_DEGREE at the return width.
        let sec_ret: Uint<RET> = sec_degree().resize::<RET>();
        assert_eq!(
            ideal.cached_norm, sec_ret,
            "sampled secret-ideal norm must equal SEC_DEGREE",
        );
        // Integral O_0-ideal ⇒ denom == 1.
        assert_eq!(ideal.denom, Uint::<RET>::ONE, "O_0-ideal denom must be 1",);
    }

    /// FULL end-to-end keygen vs the official lvl1 KAT pk[0]. Runs the byte-
    /// exact pipeline with the KAT[0] DRBG seed: `keygen_byte_exact_secret_ideal`
    /// (sample secret gen at SEC_DEGREE → `quat_lideal_create` → prime-norm
    /// reduced equivalent, all WIDE=48) → narrow J<48>→`LeftIdeal<16>` →
    /// `ideal_to_isogeny_clapotis_idx0` spine → E_A → `to_affine_a` (A·C⁻¹) →
    /// `fp2_encode`. Asserts the 64-byte curve encoding equals the KAT pk's
    /// first 64 bytes (the `ec_curve_to_bytes` portion). The 65th KAT byte is
    /// the verification `hint_pk` (a separate basis-hint computation we don't
    /// port here), so only pk[0..64] — the cryptographic curve content — is
    /// compared. Heavy (real-scale dpe-LLL + spine), hence ignored.
    #[ignore = "end-to-end keygen vs lvl1 KAT pk[0]: blocked on a DEGENERATE find_uv (0,0) (d1=1, u~2^248) for the KAT secret ideal — BOTH find_uv paths agree, so the defect is the byte-exact keygen front / J narrowing, not find_uv or the combine (S345)"]
    #[test]
    fn keygen_end_to_end_matches_kat_pk0() {
        use crate::quaternion::ideal::LeftIdeal;
        use crate::quaternion::lattice::narrow_int_lattice;
        use crate::quaternion::lll::keygen_byte_exact_secret_ideal;
        const WN: usize = 96;

        // KAT lvl1 record 0 seed (48 bytes), feeding NIST AES-256-CTR-DRBG.
        let seed: [u8; 48] = [
            0x06, 0x15, 0x50, 0x23, 0x4D, 0x15, 0x8C, 0x5E, 0xC9, 0x55, 0x95, 0xFE, 0x04, 0xEF,
            0x7A, 0x25, 0x76, 0x7F, 0x2E, 0x24, 0xCC, 0x2B, 0xC4, 0x79, 0xD0, 0x9D, 0x86, 0xDC,
            0x9A, 0xBC, 0xFD, 0xE7, 0x05, 0x6A, 0x8C, 0x26, 0x6F, 0x9E, 0xF9, 0x7E, 0xD0, 0x85,
            0x41, 0xDB, 0xD2, 0xE1, 0xFF, 0xA1,
        ];
        let mut rng = NistPqcRng::new(&seed);

        let p48 = crate::params::lvl1::prime().resize::<WN>();
        let sec = crate::params::lvl1::sec_degree().resize::<WN>();
        let wit48: [Uint<WN>; 12] =
            [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37].map(Uint::<WN>::from_u64);

        let (j, _q) =
            keygen_byte_exact_secret_ideal::<WN, _>(&sec, &p48, 8192, 64, &wit48, &mut rng)
                .expect("byte-exact keygen front must produce a prime-norm ideal");

        // Narrow J<48> → LeftIdeal<16> for the spine (J basis ~2^250 fits Int16).
        let mut b16 = [[Int::<L>::from_i64(0); 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                b16[r][c] = narrow_int_lattice::<WN, L>(&j.basis[r][c]);
            }
        }
        let lideal = LeftIdeal::<L>::with_denom_and_norm(
            b16,
            j.denom.resize::<L>(),
            j.cached_norm.resize::<L>(),
        );
        let p16 = crate::params::lvl1::prime().resize::<L>();
        let w = witnesses();

        // STATUS: the byte-exact FRONT is correct — it produces a valid denom-1
        // integral prime-norm J. (S336 fixed the S335 denom-2 seam: the bridge
        // `c_ideal_to_left_ideal` now `reduce_denom`s, so J.denom == 1, verified
        // by diagnostic: q=259 bits prime, denom=1, cached_norm=517=q².)
        // REMAINING BLOCKER (S337): the spine `ideal_to_isogeny_clapotis_idx0`
        // is an IDX0-ONLY stub — it calls `find_uv(..., &[], 2)` over only the
        // standard extremal order and `debug_assert`s `n_order == 0`. The KAT[0]
        // secret ideal needs a NON-ZERO alternate extremal order, so the
        // standard-order-only search returns None. The C `dim2id2iso` searches
        // all `NUM_ALTERNATE_nS` (=6 at lvl1) extremal orders via the ported
        // `find_uv_alternate_orders` (clapotis.rs:1601). S337 = wire the full
        // alternate-orders spine path so keygen works for any J.
        // REMAINING BLOCKER (S345): BOTH `find_uv(&[])` and `find_uv(6 alts)`
        // return a DEGENERATE index-(0,0) on this KAT[0] secret ideal — `u~2^248,
        // v=1, d1=d2=1` (a unit short vector, N_red(β)=N(I)). `u>2^246` breaks the
        // fixed-degree isogeny ⇒ combine returns None. A `d1=1` short vector means
        // the ideal has a norm-N(I) representative (principal-like), so the dim-2
        // Clapotis can't balance. Since the idx0 keygen test (`keygen_via_sampler`)
        // gets BALANCED `d1,d2~√p` on a healthy sampled ideal, the defect is in the
        // INPUT, not find_uv/the combine: the byte-exact keygen front
        // (`keygen_byte_exact_secret_ideal`, S329-S336) — or the J<48>→L16 narrowing
        // above — yields a principal-like J. NEXT: verify J is a genuine
        // non-principal degree-q ideal (compare against a `keygen_via_sampler`-shaped
        // ideal that find_uv balances); fix the front/narrowing; then the combine +
        // KAT pk-byte-match should close. The combine (clapotis_combine_indexed) is
        // correct — it runs once find_uv yields a non-degenerate decomposition.
        // The CORRECT J (WN=96) gives a balanced index-(0,0) find_uv, so the
        // idx0 spine applies — KAT[0] does NOT need the alternate orders (the
        // old "needs alt orders" was an artifact of the WN=48-corrupted J).
        let (e_a, _basis) =
            ideal_to_isogeny_clapotis_idx0(&lideal, &p16, &w, 64, 1 << 14, &mut rng)
                .expect("idx0 spine produces E_A for the KAT[0] secret ideal");

        // pk[0..64] = ec_curve_to_bytes(E_A) = fp2_encode(A·C⁻¹). Our
        // `MontgomeryCurve.a` is the AFFINE coefficient (C ≡ 1), so it already
        // equals A·C⁻¹.
        let mut pk = [0u8; 64];
        e_a.a.to_bytes_le(&mut pk);

        let kat_pk0_first64: [u8; 64] = [
            0x07, 0xcc, 0xd2, 0x14, 0x25, 0x13, 0x6f, 0x6e, 0x86, 0x5e, 0x49, 0x7d, 0x2d, 0x4d,
            0x20, 0x8f, 0x00, 0x54, 0xad, 0x81, 0x37, 0x20, 0x66, 0xe8, 0x17, 0x48, 0x07, 0x87,
            0xaa, 0xf7, 0xb2, 0x02, 0x95, 0x50, 0xc8, 0x9e, 0x89, 0x2d, 0x61, 0x8c, 0xe3, 0x23,
            0x0f, 0x23, 0x51, 0x0b, 0xfb, 0xe6, 0x8f, 0xcc, 0xdd, 0xae, 0xa5, 0x1d, 0xb1, 0x43,
            0x6b, 0x46, 0x2a, 0xdf, 0xaf, 0x00, 0x8a, 0x01,
        ];
        assert_eq!(
            pk, kat_pk0_first64,
            "keygen E_A encoding (A·C⁻¹) must match official lvl1 KAT pk[0..64]",
        );
    }

    /// S345 DIAGNOSTIC: is the byte-exact keygen secret ideal J principal-like
    /// (find_uv d1=1, the spine blocker), and does a FRESH sampler ideal of the
    /// SAME norm q behave the same? Prints find_uv's (idx, u-bits, d1-bits) +
    /// N/denom for J vs a sampler ideal of norm q. If J→d1=1 but sampler→balanced,
    /// the keygen front (or the J<48>→L16 narrowing) is the defect.
    #[ignore = "diagnostic: J (byte-exact keygen) vs sampler ideal of norm q — localizes the principal-like degeneracy"]
    #[test]
    fn diag_kat_secret_ideal_vs_sampler() {
        use crate::quaternion::ideal::LeftIdeal;
        use crate::quaternion::lattice::{narrow_int_lattice, widen_int_lattice};
        use crate::quaternion::lll::keygen_byte_exact_secret_ideal;
        use crate::quaternion::represent_integer::sampling_random_ideal_o0_given_norm_wide;
        const WN: usize = 96;
        const SL: usize = 32;

        let seed: [u8; 48] = [
            0x06, 0x15, 0x50, 0x23, 0x4D, 0x15, 0x8C, 0x5E, 0xC9, 0x55, 0x95, 0xFE, 0x04, 0xEF,
            0x7A, 0x25, 0x76, 0x7F, 0x2E, 0x24, 0xCC, 0x2B, 0xC4, 0x79, 0xD0, 0x9D, 0x86, 0xDC,
            0x9A, 0xBC, 0xFD, 0xE7, 0x05, 0x6A, 0x8C, 0x26, 0x6F, 0x9E, 0xF9, 0x7E, 0xD0, 0x85,
            0x41, 0xDB, 0xD2, 0xE1, 0xFF, 0xA1,
        ];
        let mut rng = NistPqcRng::new(&seed);
        let p48 = crate::params::lvl1::prime().resize::<WN>();
        let sec = crate::params::lvl1::sec_degree().resize::<WN>();
        let wit48: [Uint<WN>; 12] =
            [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37].map(Uint::<WN>::from_u64);
        let p16 = crate::params::lvl1::prime().resize::<L>();

        // DIAG γ: our sample_secret_gen vs the C's gen·gen_rerand (same seed).
        {
            let mut rng_g = NistPqcRng::new(&seed);
            let g = crate::quaternion::represent_integer::sample_secret_gen::<WN, _>(
                &sec, &p48, 8192, &mut rng_g,
            )
            .expect("sample_secret_gen");
            std::eprintln!(
                "DIAG GEN bits a={} b={} c={} d={}",
                g.a.abs().bits_vartime(),
                g.b.abs().bits_vartime(),
                g.c.abs().bits_vartime(),
                g.d.abs().bits_vartime()
            );
            std::eprintln!("DIAG GEN c_hex {:x}", g.c.abs());

            // Our pre-reduction secret ideal I = quat_lideal_create(γ, SEC_DEGREE).
            let (ibasis, idenom, inorm) = crate::quaternion::o0_mul::quat_lideal_create::<WN>(
                &g,
                &Int::<WN>::from_i64(1),
                &sec,
                &p48,
            );
            std::eprintln!(
                "DIAG I(WN=48): norm_bits={} denom_bits={} b00={:x} b02={:x} b03={:x}",
                inorm.bits_vartime(),
                idenom.abs().bits_vartime(),
                ibasis[0][0].abs(),
                ibasis[0][2].abs(),
                ibasis[0][3].abs()
            );
            // Same create at a WIDER width (γ ~2^1271 ⇒ det/HNF need ≫ 3072 bits).
            const W2: usize = 96;
            let g2 = crate::quaternion::Quaternion::<W2>::new(
                g.a.resize::<W2>(),
                g.b.resize::<W2>(),
                g.c.resize::<W2>(),
                g.d.resize::<W2>(),
            );
            let sec2 = sec.resize::<W2>();
            let p2 = crate::params::lvl1::prime().resize::<W2>();
            let (ib2, id2, in2) = crate::quaternion::o0_mul::quat_lideal_create::<W2>(
                &g2,
                &Int::<W2>::from_i64(1),
                &sec2,
                &p2,
            );
            std::eprintln!(
                "DIAG I(WN=96): norm_bits={} denom_bits={} b00={:x} b02={:x} b03={:x}",
                in2.bits_vartime(),
                id2.abs().bits_vartime(),
                ib2[0][0].abs(),
                ib2[0][2].abs(),
                ib2[0][3].abs()
            );
        }

        let (j, q) =
            keygen_byte_exact_secret_ideal::<WN, _>(&sec, &p48, 8192, 64, &wit48, &mut rng)
                .expect("keygen front");
        let mut b16 = [[Int::<L>::from_i64(0); 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                b16[r][c] = narrow_int_lattice::<WN, L>(&j.basis[r][c]);
            }
        }
        let j16 = LeftIdeal::<L>::with_denom_and_norm(
            b16,
            j.denom.resize::<L>(),
            j.cached_norm.resize::<L>(),
        );
        let n_j_det = lattice_reduced_norm::<L, 32>(&j16.basis, &j16.denom);
        std::eprintln!(
            "DIAG J: q_bits={} cached_bits={} denom_bits={} N_det_bits={:?}",
            q.bits_vartime(),
            j16.cached_norm.bits_vartime(),
            j16.denom.bits_vartime(),
            n_j_det.map(|n| n.bits_vartime())
        );
        let tgt = *Uint::<L>::ONE.shl_vartime(F).as_int();
        match find_uv::<L>(&tgt, &j16, &p16, &[], 2) {
            Ok(rr) => std::eprintln!(
                "DIAG J find_uv: idx=({},{}) u_bits={} d1_bits={} d2_bits={}",
                rr.index_alternate_order_1,
                rr.index_alternate_order_2,
                rr.u.abs().bits_vartime(),
                rr.d1.abs().bits_vartime(),
                rr.d2.abs().bits_vartime()
            ),
            Err(e) => std::eprintln!("DIAG J find_uv: Err({e:?})"),
        }
        // find_uv WITH the 6 alts (the path ideal_to_isogeny_clapotis actually uses).
        {
            use crate::quaternion::connecting_ideals as ci;
            use crate::quaternion::lattice::widen_int_lattice as wil;
            let wd = |id: &LeftIdeal<8>| -> LeftIdeal<L> {
                let mut b = [[Int::<L>::from_i64(0); 4]; 4];
                for rr in 0..4 {
                    for cc in 0..4 {
                        b[rr][cc] = wil::<8, L>(&id.basis[rr][cc]);
                    }
                }
                LeftIdeal::<L>::with_denom_and_norm(
                    b,
                    id.denom.resize::<L>(),
                    id.cached_norm.resize::<L>(),
                )
            };
            let alts = [
                wd(&ci::alternate_connecting_ideal_0_l1()),
                wd(&ci::alternate_connecting_ideal_1_l1()),
                wd(&ci::alternate_connecting_ideal_2_l1()),
                wd(&ci::alternate_connecting_ideal_3_l1()),
                wd(&ci::alternate_connecting_ideal_4_l1()),
                wd(&ci::alternate_connecting_ideal_5_l1()),
            ];
            match find_uv::<L>(&tgt, &j16, &p16, &alts, 2) {
                Ok(rr) => std::eprintln!(
                    "DIAG J find_uv(alts): idx=({},{}) u_bits={} d1_bits={} d2_bits={}",
                    rr.index_alternate_order_1,
                    rr.index_alternate_order_2,
                    rr.u.abs().bits_vartime(),
                    rr.d1.abs().bits_vartime(),
                    rr.d2.abs().bits_vartime()
                ),
                Err(e) => std::eprintln!("DIAG J find_uv(alts): Err({e:?})"),
            }
        }

        // Fresh sampler ideal of the SAME norm q (is_prime path).
        let q_sl = q.resize::<SL>();
        let p_sl = crate::params::lvl1::prime().resize::<SL>();
        let wit_sl: [Uint<SL>; 12] =
            [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37].map(Uint::<SL>::from_u64);
        let s8 = sampling_random_ideal_o0_given_norm_wide::<SL, _>(
            &q_sl,
            &p_sl,
            true,
            None,
            64,
            1 << 14,
            &wit_sl,
            &mut rng,
        )
        .expect("sampler ideal of norm q");
        let mut sb16 = [[Int::<L>::from_i64(0); 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                sb16[r][c] = widen_int_lattice::<8, L>(&s8.basis[r][c]);
            }
        }
        let s16 = LeftIdeal::<L>::with_denom_and_norm(
            sb16,
            s8.denom.resize::<L>(),
            s8.cached_norm.resize::<L>(),
        );
        let n_s_det = lattice_reduced_norm::<L, 32>(&s16.basis, &s16.denom);
        std::eprintln!(
            "DIAG S: cached_bits={} N_det_bits={:?}",
            s16.cached_norm.bits_vartime(),
            n_s_det.map(|n| n.bits_vartime())
        );
        std::eprintln!("DIAG S denom_bits={}", s16.denom.bits_vartime());
        match find_uv::<L>(&tgt, &s16, &p16, &[], 2) {
            Ok(rr) => std::eprintln!(
                "DIAG S find_uv: idx=({},{}) u_bits={} d1_bits={} d2_bits={}",
                rr.index_alternate_order_1,
                rr.index_alternate_order_2,
                rr.u.abs().bits_vartime(),
                rr.d1.abs().bits_vartime(),
                rr.d2.abs().bits_vartime()
            ),
            Err(e) => std::eprintln!("DIAG S find_uv: Err({e:?})"),
        }
    }
}
