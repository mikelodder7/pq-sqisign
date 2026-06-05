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
}
