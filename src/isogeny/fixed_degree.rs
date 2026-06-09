//! `fixed_degree_isogeny_and_eval` (φ) — a fixed-degree `2^length` dimension-2
//! isogeny on `E0 × E0`, computed via the Kani/Clapotis construction.
//!
//! This is the assembly capstone of the Clapotis evaluator's machine half: it
//! builds a valid theta-chain kernel from a `RepresentInteger` endomorphism and
//! runs the theta chain — the FIRST end-to-end exercise of the chain on a real
//! (valid) kernel. Port of the SQIsign C reference `_fixed_degree_isogeny_impl`
//! (`src/id2iso/ref/lvlx/dim2id2iso.c`), specialized to level 1 and the
//! standard starting curve `E0` (index 0).
//!
//! Steps: `θ = RepresentInteger(u·(2^length − u))` (O0-basis coords) → scale by
//! `u^{-1} mod 2^(length+2)` → `B0 = basis_e0` (order `2^(length+extra)`),
//! `Bθ = θ(B0)` via [`endomorphism_application_o0_coords`] → lift both to
//! Jacobian ([`lift_basis`]) → form the couple kernel `(T1, T2)` →
//! [`theta_chain_compute_and_eval`].

use crate::ec::couple::ThetaKernelCouplePoints;
use crate::ec::couple::{CoupleCurve, CoupleJacobianPoint, CoupleMontgomeryPoint, EcBasis};
use crate::ec::jacobian::lift_basis;
use crate::ec::montgomery::MontgomeryCurve;
use crate::isogeny::endomorphism::{basis_e0_lvl1, endomorphism_application_o0_coords};
use crate::isogeny::theta_chain::theta_chain_compute_and_eval;
use crate::params::lvl1::Fp1Element;
use crypto_bigint::{Int, Uint};
use rand_core::CryptoRng;

/// Quaternion-side precision for `RepresentInteger` at level 1
/// (`64·LIMBS ≥ 3·bits(p)+2 = 755` ⇒ `LIMBS ≥ 12`).
const QL: usize = 12;

/// Compute a fixed-degree `2^length` isogeny `E0 × E0 → E34` and evaluate
/// `eval_points` through it, writing images into `out_points`.
///
/// `u` must be odd with `0 < u < 2^length` and `4·u·(2^length − u) > p`
/// (so `RepresentInteger`'s `4M > p` boundary holds). `u` is a full
/// `Uint<QL>` because the Clapotis spine's `find_uv` produces Bézout
/// coefficients up to `~2^length`, far beyond `u64`. Returns
/// `Some((length, E34))` on success, or `None` if `RepresentInteger`
/// exhausts its budget, an inversion/lift fails, or the chain does not
/// produce an isogeny.
#[allow(dead_code)]
pub(crate) fn fixed_degree_isogeny_and_eval<R: CryptoRng>(
    u: &Uint<QL>,
    eval_points: &[CoupleMontgomeryPoint<Fp1Element>],
    out_points: &mut [CoupleMontgomeryPoint<Fp1Element>],
    witnesses: &[Uint<QL>],
    sample_bound: i64,
    max_trials: usize,
    rng: &mut R,
) -> Option<(u32, CoupleCurve<Fp1Element>)> {
    let length: u32 = 246; // TORSION_EVEN_POWER (248) − HD_extra_torsion (2)
    let f_basis: usize = 248; // length + HD_extra_torsion
    debug_assert!(u.as_words()[0] & 1 == 1, "u must be odd");

    // target = u · (2^length − u)
    let u12 = *u;
    let two_len = Uint::<QL>::ONE.shl_vartime(length);
    let target = u12.wrapping_mul(&two_len.wrapping_sub(&u12));

    let p = crate::params::lvl1::prime().resize::<QL>();

    // θ in O0-basis coords with N_red(θ) = target.
    let theta_o0 =
        crate::quaternion::represent_integer::find_quaternion_in_full_order_with_norm_wide::<QL, R>(
            &target,
            &p,
            sample_bound,
            max_trials,
            witnesses,
            rng,
        )?;

    // Scale θ by u^{-1} mod 2^(length+2).
    let modulus = Uint::<QL>::ONE.shl_vartime(length + 2);
    let u_inv = crate::quaternion::sign_orchestration::uint_inv_mod_vartime::<QL>(&u12, &modulus)?;
    let u_inv_i = Int::<QL>::from_words(u_inv.to_words());
    let mut theta = theta_o0;
    for c in theta.iter_mut() {
        *c = c.wrapping_mul(&u_inv_i);
    }

    // B0 = canonical even-torsion basis; Bθ = θ(B0).
    let curve = MontgomeryCurve::<Fp1Element>::e0();
    let a24 = curve.a24();
    let (bp, bq, bpmq) = basis_e0_lvl1();
    let (rp, rq, rpmq) =
        endomorphism_application_o0_coords::<QL>(&bp, &bq, &bpmq, &theta, f_basis, &a24)?;

    // Lift both x-only bases to consistent Jacobian points.
    let bas1 = EcBasis::new(bp, bq, bpmq);
    let bas2 = EcBasis::new(rp, rq, rpmq);
    let (p1, q1) = lift_basis(&bas1, &curve).ok()?;
    let (p2, q2) = lift_basis(&bas2, &curve).ok()?;

    // Couple kernel (T1, T2); the chain seeds gluing kernel from T1, T2 only,
    // so t1_minus_t2 is unused (placeholder).
    let ker = ThetaKernelCouplePoints::new(
        CoupleJacobianPoint::new(p1, p2),
        CoupleJacobianPoint::new(q1, q2),
        CoupleJacobianPoint::infinity(),
    );

    let e12 = CoupleCurve::e0_e0();
    let e34 = theta_chain_compute_and_eval(length, &e12, &ker, true, eval_points, out_points)?;
    Some((length, e34))
}

/// Indexed generalization of [`fixed_degree_isogeny_and_eval`] for the
/// `n_order ≠ 0` Clapotis path: a fixed-degree `2^length` isogeny starting
/// from the alternate NICE curve `CURVES_WITH_ENDOMORPHISMS[index]` instead of
/// `E0`. Port of the C `_fixed_degree_isogeny_impl` with `index_alternate_order`
/// (dim2id2iso.c:18-185).
///
/// `index_alternate_curve == 0` delegates to the validated O0 path
/// [`fixed_degree_isogeny_and_eval`]. For `k = index − 1 ≥ 0`:
/// `θ = represent_integer_over_alt_order(EXTREMAL_ORDERS[index], u·(2^length−u))`
/// (standard coords + denom) → scale numerator by `u^{-1} mod 2^(length+2)` →
/// `B0 = curve_with_endomorphism_{k}().basis_even`, `Bθ = θ(B0)` via the indexed
/// endomorphism application (item 6) → couple kernel on `E0_alt × E0_alt` →
/// theta chain. Returns `(length, E34)` or `None` on any sub-step failure.
///
/// NOT yet exercised end-to-end (the spine that selects a non-zero index is
/// future work); first real exercise is the item-8 keygen KAT. The k≥1 path's
/// correctness rests on the standalone-verified `represent_integer_over_alt_order`
/// (norm + membership) and `endomorphism_application_even_basis_indexed`
/// (identity-validated on all 6 curves).
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn fixed_degree_isogeny_and_eval_indexed<R: CryptoRng>(
    index_alternate_curve: usize,
    u: &Uint<QL>,
    eval_points: &[CoupleMontgomeryPoint<Fp1Element>],
    out_points: &mut [CoupleMontgomeryPoint<Fp1Element>],
    witnesses: &[Uint<QL>],
    sample_bound: i64,
    max_trials: usize,
    rng: &mut R,
) -> Option<(u32, CoupleCurve<Fp1Element>)> {
    use crate::ec::montgomery::MontgomeryPoint;
    use crate::isogeny::endomorphism::endomorphism_application_even_basis_indexed;
    use crate::quaternion::algebra::Quaternion;
    use crate::quaternion::curves_with_endomorphism as cwe_mod;
    use crate::quaternion::extremal_orders as eo;

    if index_alternate_curve == 0 {
        return fixed_degree_isogeny_and_eval(
            u,
            eval_points,
            out_points,
            witnesses,
            sample_bound,
            max_trials,
            rng,
        );
    }
    let _ = sample_bound; // alt-order represent_integer does not take a sample bound
    let k = index_alternate_curve - 1;
    let length: u32 = 246;
    let f_basis: usize = 248;
    debug_assert!(u.as_words()[0] & 1 == 1, "u must be odd");

    let u12 = *u;
    let two_len = Uint::<QL>::ONE.shl_vartime(length);
    let target = u12.wrapping_mul(&two_len.wrapping_sub(&u12));
    let p = crate::params::lvl1::prime().resize::<QL>();

    // θ over the alternate extremal order k (standard coords + denom).
    let order = match k {
        0 => eo::alternate_extremal_order_0_l1(),
        1 => eo::alternate_extremal_order_1_l1(),
        2 => eo::alternate_extremal_order_2_l1(),
        3 => eo::alternate_extremal_order_3_l1(),
        4 => eo::alternate_extremal_order_4_l1(),
        5 => eo::alternate_extremal_order_5_l1(),
        _ => return None,
    };
    let (theta_num, theta_denom) =
        crate::quaternion::represent_integer::represent_integer_over_alt_order::<QL, R>(
            &order, &target, &p, max_trials, witnesses, rng,
        )?;

    // Scale θ numerator by u^{-1} mod 2^(length+2) (denom unchanged).
    let modulus = Uint::<QL>::ONE.shl_vartime(length + 2);
    let u_inv = crate::quaternion::sign_orchestration::uint_inv_mod_vartime::<QL>(&u12, &modulus)?;
    let u_inv_i = Int::<QL>::from_words(u_inv.to_words());
    let theta_scaled = Quaternion::<QL>::new(
        theta_num.a.wrapping_mul(&u_inv_i),
        theta_num.b.wrapping_mul(&u_inv_i),
        theta_num.c.wrapping_mul(&u_inv_i),
        theta_num.d.wrapping_mul(&u_inv_i),
    );

    // E0_alt = NICE curve k (C = 1 ⇒ curve_a is affine A); B0 = its even basis.
    let cwe = match k {
        0 => cwe_mod::curve_with_endomorphism_0_l1(),
        1 => cwe_mod::curve_with_endomorphism_1_l1(),
        2 => cwe_mod::curve_with_endomorphism_2_l1(),
        3 => cwe_mod::curve_with_endomorphism_3_l1(),
        4 => cwe_mod::curve_with_endomorphism_4_l1(),
        5 => cwe_mod::curve_with_endomorphism_5_l1(),
        _ => return None,
    };
    let curve = MontgomeryCurve::<Fp1Element>::new(cwe.curve_a);
    let a24 = curve.a24();
    let bp = MontgomeryPoint::<Fp1Element>::new(cwe.p_x, cwe.p_z);
    let bq = MontgomeryPoint::<Fp1Element>::new(cwe.q_x, cwe.q_z);
    let bpmq = MontgomeryPoint::<Fp1Element>::new(cwe.pmq_x, cwe.pmq_z);

    // Bθ = θ(B0) via the indexed endomorphism application (item 6); θ narrows
    // to Int<8> (scaled coords ≤ ~2^493 < 2^511 at L1).
    let theta8 = Quaternion::<8>::new(
        theta_scaled.a.resize::<8>(),
        theta_scaled.b.resize::<8>(),
        theta_scaled.c.resize::<8>(),
        theta_scaled.d.resize::<8>(),
    );
    let (rp, rq, rpmq) = endomorphism_application_even_basis_indexed(
        &bp,
        &bq,
        &bpmq,
        index_alternate_curve,
        &theta8,
        &theta_denom.resize::<8>(),
        f_basis,
        &a24,
    )?;

    // Lift both bases, couple kernel, theta chain on E0_alt × E0_alt.
    let bas1 = EcBasis::new(bp, bq, bpmq);
    let bas2 = EcBasis::new(rp, rq, rpmq);
    let (p1, q1) = lift_basis(&bas1, &curve).ok()?;
    let (p2, q2) = lift_basis(&bas2, &curve).ok()?;
    let ker = ThetaKernelCouplePoints::new(
        CoupleJacobianPoint::new(p1, p2),
        CoupleJacobianPoint::new(q1, q2),
        CoupleJacobianPoint::infinity(),
    );
    let e12 = CoupleCurve::new(curve, curve);
    let e34 = theta_chain_compute_and_eval(length, &e12, &ker, true, eval_points, out_points)?;
    Some((length, e34))
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

    /// φ's KERNEL-CONSTRUCTION half assembles cleanly on a real input: for a
    /// large odd `u`, `RepresentInteger` finds an endomorphism of norm
    /// `u·(2^length−u)`, the endomorphism applies to the even-torsion basis,
    /// and BOTH factor bases lift to consistent Jacobian points.
    ///
    /// `u` must be large so `target ≫ p` (small `u` forces `c=d=0` in
    /// RepresentInteger, requiring `target` itself to be a sum of two squares).
    #[test]
    fn phi_kernel_construction_stages_succeed() {
        use crate::ec::couple::EcBasis;
        use crate::ec::jacobian::lift_basis;
        use crate::ec::montgomery::MontgomeryCurve;
        use crate::isogeny::endomorphism::{basis_e0_lvl1, endomorphism_application_o0_coords};

        let w = witnesses();
        let mut rng = NistPqcRng::new(&[0x5Au8; 48]);
        let length = 246u32;
        let u = (1u64 << 40) | 1;
        let u12 = Uint::<QL>::from_u64(u);
        let two_len = Uint::<QL>::ONE.shl_vartime(length);
        let target = u12.wrapping_mul(&two_len.wrapping_sub(&u12));
        let p = crate::params::lvl1::prime().resize::<QL>();

        let theta_o0 =
            crate::quaternion::represent_integer::find_quaternion_in_full_order_with_norm_wide::<
                QL,
                _,
            >(&target, &p, 64, 1 << 14, &w, &mut rng)
            .expect("RepresentInteger finds θ of norm u·(2^length−u) for large u");

        let modulus = Uint::<QL>::ONE.shl_vartime(length + 2);
        let u_inv =
            crate::quaternion::sign_orchestration::uint_inv_mod_vartime::<QL>(&u12, &modulus)
                .expect("u invertible mod 2^(length+2)");
        let u_inv_i = Int::<QL>::from_words(u_inv.to_words());
        let mut theta = theta_o0;
        for c in theta.iter_mut() {
            *c = c.wrapping_mul(&u_inv_i);
        }

        let curve = MontgomeryCurve::<Fp1Element>::e0();
        let a24 = curve.a24();
        let (bp, bq, bpmq) = basis_e0_lvl1();
        let (rp, rq, rpmq) =
            endomorphism_application_o0_coords::<QL>(&bp, &bq, &bpmq, &theta, 248, &a24)
                .expect("θ applies to the even-torsion basis");

        let bas1 = EcBasis::new(bp, bq, bpmq);
        let bas2 = EcBasis::new(rp, rq, rpmq);
        assert!(lift_basis(&bas1, &curve).is_ok(), "B0 lifts to Jacobian");
        assert!(lift_basis(&bas2, &curve).is_ok(), "θ(B0) lifts to Jacobian");
    }

    /// FULL φ end-to-end — the theta chain's first complete real isogeny.
    /// For a large odd `u`, builds the Kani kernel (order 2²⁴⁸, θ content odd,
    /// N(θ) ≡ −1 mod 2^length) and runs the chain: gluing → 245 interior steps
    /// → splitting → elliptic-product extraction, producing a well-formed
    /// `E₃ × E₄`. (The descent doubling uses the C-ref `theta_precomputation` +
    /// `double_point`; the final product-codomain skips the doubling constants
    /// it does not need.)
    #[test]
    fn fixed_degree_isogeny_produces_a_codomain() {
        let w = witnesses();
        let mut rng = NistPqcRng::new(&[0x5Au8; 48]);
        let big = 1u64 << 40;
        let mut got = None;
        for u in [big | 1, big | 3, big | 5, big | 7, big | 9, big | 11] {
            if let Some((length, e34)) = fixed_degree_isogeny_and_eval(
                &Uint::<QL>::from_u64(u),
                &[],
                &mut [],
                &w,
                64,
                1 << 14,
                &mut rng,
            ) {
                got = Some((length, e34));
                break;
            }
        }
        let (length, _e34) = got.expect("φ should produce a codomain for some large odd u");
        assert_eq!(length, 246);
    }

    /// φ with a `u` far beyond `u64` — the Clapotis spine's real regime,
    /// where `find_uv`'s Bézout `u` runs up to `~2^length`. Exercises the
    /// `Uint<QL>` generalization with `u = 2^124 + odd`; `4·u·(2^246−u)`
    /// is `≫ p`, so `RepresentInteger` has room.
    #[test]
    fn fixed_degree_isogeny_handles_large_u_beyond_u64() {
        let w = witnesses();
        let mut rng = NistPqcRng::new(&[0x5Au8; 48]);
        let base = Uint::<QL>::ONE.shl_vartime(124);
        let mut got = None;
        for odd in [1u64, 3, 5, 7, 9, 11] {
            let u = base.wrapping_add(&Uint::<QL>::from_u64(odd));
            if let Some((length, e34)) =
                fixed_degree_isogeny_and_eval(&u, &[], &mut [], &w, 64, 1 << 14, &mut rng)
            {
                got = Some((length, e34));
                break;
            }
        }
        let (length, _e34) = got.expect("φ should produce a codomain for a large (>u64) odd u");
        assert_eq!(length, 246);
    }

    /// S345: the INDEXED φ runs end-to-end from an alternate NICE curve
    /// (index 1 ⇒ k=0). Exercises the full alt-curve assembly:
    /// `represent_integer_over_alt_order` → `u^{-1}` scale → item-6 indexed
    /// endomorphism on the NICE curve's even basis → lift → theta chain on
    /// `E0_alt × E0_alt`. Contract: produces a degree-`2^246` codomain and
    /// pushes the eval points (returns `Some`). Heavy (246-step chain). Not a
    /// byte-exactness check — that is the item-8 KAT.
    #[ignore = "heavy: end-to-end indexed φ from an alternate NICE curve (S345)"]
    #[test]
    fn fixed_degree_isogeny_indexed_k1_runs_end_to_end() {
        use crate::ec::couple::CoupleMontgomeryPoint;
        use crate::ec::montgomery::MontgomeryPoint;
        use crate::quaternion::curves_with_endomorphism::curve_with_endomorphism_0_l1;

        let w = witnesses();
        let mut rng = NistPqcRng::new(&[0x77u8; 48]);

        // u large + odd so target = u·(2^246 − u) ≫ p·q/4 (representable).
        let u12 = Uint::<QL>::ONE.shl_vartime(123).wrapping_add(&Uint::ONE);

        // Eval points: the NICE curve's even basis on factor 1 (× O on factor 2).
        let cwe = curve_with_endomorphism_0_l1(); // index 1 → k = 0
        let bp = MontgomeryPoint::<Fp1Element>::new(cwe.p_x, cwe.p_z);
        let bq = MontgomeryPoint::<Fp1Element>::new(cwe.q_x, cwe.q_z);
        let bpmq = MontgomeryPoint::<Fp1Element>::new(cwe.pmq_x, cwe.pmq_z);
        let inf = MontgomeryPoint::<Fp1Element>::infinity();
        let eval = [
            CoupleMontgomeryPoint::new(bp, inf),
            CoupleMontgomeryPoint::new(bq, inf),
            CoupleMontgomeryPoint::new(bpmq, inf),
        ];
        let mut out = [CoupleMontgomeryPoint::infinity(); 3];

        let got = fixed_degree_isogeny_and_eval_indexed(
            1,
            &u12,
            &eval,
            &mut out,
            &w,
            64,
            1 << 14,
            &mut rng,
        );
        let (length, _e34) =
            got.expect("indexed φ (index 1) must produce a 2^246 codomain from the NICE curve");
        assert_eq!(length, 246);
    }
}
