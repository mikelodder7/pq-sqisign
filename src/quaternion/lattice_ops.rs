//! Quaternion-lattice operations for the signing response (`sign.c` step 3):
//! conjugate, dual, sum, and intersection of `O_0`-coordinate lattices.
//!
//! Port of the C reference `src/quaternion/ref/generic/lattice.c`
//! (`quat_lattice_*`). Unlike the C reference — which works in standard
//! `(1, i, j, k)` coordinates — these operate in the `O_0`-basis coordinates
//! that the rest of the Rust quaternion module uses ([`LeftIdeal`]'s `basis`),
//! so conjugation uses [`o0_conjugate`](crate::quaternion::o0_mul::o0_conjugate)
//! and the reduced-norm Gram form is
//! [`o0_reduced_norm_gram_matrix`](crate::quaternion::o0_mul::o0_reduced_norm_gram_matrix).
//! The dual/sum/intersection are pure `Z`-module operations and are
//! coordinate-system independent given a consistent basis.
#![allow(dead_code)]

use crate::quaternion::extremal_orders::adjugate_with_det;
use crate::quaternion::o0_mul::o0_conjugate;
use crypto_bigint::{Int, Uint};

/// Conjugate of a lattice (`O_0`-coords): conjugate each `Z`-generator. Port of
/// C `quat_lattice_conjugate_without_hnf` adapted to `O_0`-coords (the C negates
/// rows 1–3 in standard coords; here `o0_conjugate` is the coordinate-correct
/// conjugation `(a, b, c, d) ↦ (a + d, −b, −c, −d)`).
pub(crate) fn lattice_conjugate<const L: usize>(basis: &[[Int<L>; 4]; 4]) -> [[Int<L>; 4]; 4] {
    let mut out = [[Int::<L>::from_i64(0); 4]; 4];
    for (row_out, row_in) in out.iter_mut().zip(basis.iter()) {
        *row_out = o0_conjugate::<L>(row_in);
    }
    out
}

/// Dual lattice (without HNF) w.r.t. the standard coordinate dot product. Port
/// of C `quat_lattice_dual_without_hnf`: `dual.basis = denom · adj(B)ᵀ`,
/// `dual.denom = |det(B)|` (where `adj(B)/det(B) = B⁻¹`). Returns
/// `(basis, denom)`. The sign of `det` is folded into the basis so `denom`
/// stays non-negative.
pub(crate) fn lattice_dual<const L: usize>(
    basis: &[[Int<L>; 4]; 4],
    denom: &Uint<L>,
) -> ([[Int<L>; 4]; 4], Uint<L>) {
    let (adj, det) = adjugate_with_det::<L>(basis);
    let neg = bool::from(det.is_negative());
    let det_u = det.abs();
    let denom_i = *denom.as_int();
    let mut out = [[Int::<L>::from_i64(0); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            // (adjᵀ)[i][j] = adj[j][i]; scale by denom; fold det's sign in.
            let mut e = denom_i.wrapping_mul(&adj[j][i]);
            if neg {
                e = e.wrapping_neg();
            }
            out[i][j] = e;
        }
    }
    (out, det_u)
}

/// Scale every entry of a 4×4 basis by `s`.
fn scale_mat<const L: usize>(b: &[[Int<L>; 4]; 4], s: &Int<L>) -> [[Int<L>; 4]; 4] {
    let mut out = [[Int::<L>::from_i64(0); 4]; 4];
    for (row_out, row_in) in out.iter_mut().zip(b.iter()) {
        for (e_out, e_in) in row_out.iter_mut().zip(row_in.iter()) {
            *e_out = e_in.wrapping_mul(s);
        }
    }
    out
}

/// Sum of two lattices (`O_0`-coords), `L1 + L2`. Cross-scale the bases to a
/// common denominator `d1·d2`, take the row-convention HNF of all 8 generators
/// (`hnf_rect_4cols`, which is lattice-preserving — unlike `hnf_mod_core`), then
/// reduce the denominator. Returns `(basis, denom)`.
pub(crate) fn lattice_add<const L: usize>(
    b1: &[[Int<L>; 4]; 4],
    d1: &Uint<L>,
    b2: &[[Int<L>; 4]; 4],
    d2: &Uint<L>,
) -> ([[Int<L>; 4]; 4], Uint<L>) {
    use crate::quaternion::hnf::{hnf_rect_4cols, quat_lattice_reduce_denom};

    let d1_i = *d1.as_int();
    let d2_i = *d2.as_int();
    let g1 = scale_mat::<L>(b2, &d1_i); // d1 · B2
    let g2 = scale_mat::<L>(b1, &d2_i); // d2 · B1
    // Row-convention, lattice-preserving HNF of all 8 generators (the
    // modulus-based `hnf_mod_core` does NOT preserve arbitrary row-lattices —
    // see `hnf_mod_core_does_not_preserve_dual_lattice_but_hnf_4x4_does`).
    // `hnf_rect_4cols` puts the rank-4 basis in rows 0..4 (rows 4..8 → 0).
    let gens: [[Int<L>; 4]; 8] = [g1[0], g1[1], g1[2], g1[3], g2[0], g2[1], g2[2], g2[3]];
    let h = hnf_rect_4cols::<8, L>(&gens);
    let basis = [h[0], h[1], h[2], h[3]];
    let denom_prod = *d1.wrapping_mul(d2).as_int();
    let (rb, rd) = quat_lattice_reduce_denom::<L>(&basis, &denom_prod);
    (rb, rd.abs())
}

/// Intersection of two lattices (`O_0`-coords), `L1 ∩ L2 = (L1* + L2*)*`. Port
/// of C `quat_lattice_intersect` (dual–sum–dual, then HNF). Returns
/// `(basis, denom)`.
pub(crate) fn lattice_intersect<const L: usize>(
    b1: &[[Int<L>; 4]; 4],
    d1: &Uint<L>,
    b2: &[[Int<L>; 4]; 4],
    d2: &Uint<L>,
) -> ([[Int<L>; 4]; 4], Uint<L>) {
    use crate::quaternion::hnf::quat_lattice_reduce_denom;

    let (du1, dn1) = lattice_dual::<L>(b1, d1);
    let (du2, dn2) = lattice_dual::<L>(b2, d2);
    let (sum_b, sum_d) = lattice_add::<L>(&du1, &dn1, &du2, &dn2);
    let (res_b, res_d) = lattice_dual::<L>(&sum_b, &sum_d);
    // Reduce the denominator to lowest terms (HNF is optional — the C ref notes
    // "could be removed"; `equals_lattice` and downstream don't require it).
    let (rb, rd) = quat_lattice_reduce_denom::<L>(&res_b, res_d.as_int());
    (rb, rd.abs())
}

/// Sample a lattice element with reduced norm ≤ `radius`. Functional analogue
/// of C `quat_lattice_sample_from_ball` (`lat_ball.c`) — but instead of the
/// dpe-float `quat_lll_core` + bounding-parallelogram, this LLL-reduces the
/// lattice under the `O_0` reduced-norm metric (`lll_4x4_in_metric`, integer)
/// and rejection-samples small integer combinations of the reduced basis. The
/// result is a genuine lattice element inside the norm ball (sufficient for a
/// valid response quaternion); it is NOT byte-exact to C's RNG-driven sampler
/// (that, and ZK-grade uniformity, are a later refinement).
///
/// Returns `(e, denom)` with `e` in `O_0`-coords and
/// `N_red(e/denom) = qf_eval(o0_gram, e) / (4·denom²) ≤ radius`, or `None` if no
/// element is found within the search budget. lvl1-pinned via the `O_0` metric.
#[cfg(feature = "alloc")]
pub(crate) fn lattice_sample_from_ball<const L: usize, R: rand_core::CryptoRng>(
    basis: &[[Int<L>; 4]; 4],
    denom: &Uint<L>,
    radius: &Uint<L>,
    p: &Uint<L>,
    max_trials: usize,
    rng: &mut R,
) -> Option<([Int<L>; 4], Uint<L>)> {
    use crate::quaternion::lattice::{lll_4x4_in_metric, qf_eval_4x4};
    use crate::quaternion::o0_mul::o0_reduced_norm_gram_matrix;

    let metric = o0_reduced_norm_gram_matrix::<L>(p);
    let reduced = lll_4x4_in_metric::<L>(basis, &metric);

    // bound = 4·radius·denom² (the o0 Gram is 4× the reduced-norm form).
    let denom_sq = denom.wrapping_mul(denom);
    let bound = *radius
        .wrapping_mul(&denom_sq)
        .wrapping_mul(&Uint::<L>::from_u64(4))
        .as_int();

    let combo = |c: &[i64; 4]| -> [Int<L>; 4] {
        let mut e = [Int::<L>::from_i64(0); 4];
        for (r, &cr) in c.iter().enumerate() {
            if cr == 0 {
                continue;
            }
            let cs = Int::<L>::from_i64(cr);
            for t in 0..4 {
                e[t] = e[t].wrapping_add(&cs.wrapping_mul(&reduced[r][t]));
            }
        }
        e
    };
    let in_ball = |e: &[Int<L>; 4]| -> bool {
        let qf = qf_eval_4x4::<L>(e, &metric);
        !bool::from(qf.is_negative()) && qf != Int::<L>::from_i64(0) && qf.abs() <= bound.abs()
    };

    // Deterministic first pass: each reduced basis vector (shortest first).
    for r in 0..4 {
        let mut c = [0i64; 4];
        c[r] = 1;
        let e = combo(&c);
        if in_ball(&e) {
            return Some((e, *denom));
        }
    }
    // Rejection pass: random small combinations.
    let mut buf = [0u8; 4];
    for _ in 0..max_trials {
        rng.fill_bytes(&mut buf);
        let c = [
            (buf[0] % 5) as i64 - 2,
            (buf[1] % 5) as i64 - 2,
            (buf[2] % 5) as i64 - 2,
            (buf[3] % 5) as i64 - 2,
        ];
        if c == [0, 0, 0, 0] {
            continue;
        }
        let e = combo(&c);
        if in_ball(&e) {
            return Some((e, *denom));
        }
    }
    None
}

/// Compute the SQIsign response quaternion (sign step 3). Port of C
/// `compute_response_quat_element` (`sign.c:66`): intersect the challenge ideal
/// with the secret ideal, intersect that with the conjugate of the commitment
/// ideal to get the `Hom(E_chall → E_com)`-lattice, then sample a short element
/// (reduced norm ≤ `(2^response_bits − 1)·lattice_content`). Returns
/// `(resp O_0-coords, resp_denom, lattice_content)`. lvl1-pinned via the metric
/// inside `lattice_sample_from_ball`.
#[cfg(feature = "alloc")]
pub(crate) fn compute_response_quat_element<const L: usize, R: rand_core::CryptoRng>(
    secret_ideal: &crate::quaternion::ideal::LeftIdeal<L>,
    lideal_chall_two: &crate::quaternion::ideal::LeftIdeal<L>,
    lideal_commit: &crate::quaternion::ideal::LeftIdeal<L>,
    p: &Uint<L>,
    response_bits: u32,
    max_trials: usize,
    rng: &mut R,
) -> Option<([Int<L>; 4], Uint<L>, Uint<L>)> {
    use crate::quaternion::ideal_mul::lideal_intersect_lattice;

    // chall_secret = lideal_chall_two ∩ secret_ideal — the TRUE lattice
    // intersection (C `quat_lideal_inter` = dual(dual(I)+dual(J))), NOT the
    // coprime "I·J" product shortcut: the challenge and secret ideals have
    // incompatible right orders, so the product ≠ the set-intersection.
    let chall_secret = match lideal_intersect_lattice::<L, L>(lideal_chall_two, secret_ideal) {
        Ok(cs) => cs,
        Err(_) => return None,
    };
    // Hom-lattice = chall_secret ∩ conjugate(commitment).
    let conj_commit = lattice_conjugate::<L>(&lideal_commit.basis);
    let (hom_b, hom_d) = lattice_intersect::<L>(
        &chall_secret.basis,
        &chall_secret.denom,
        &conj_commit,
        &lideal_commit.denom,
    );
    // lattice_content = N(chall_secret) · N(commitment).
    let n_cs = chall_secret.reduced_norm_vartime()?;
    let n_com = lideal_commit.reduced_norm_vartime()?;
    let lattice_content = n_cs.wrapping_mul(&n_com);
    // radius = (2^response_bits − 1) · lattice_content.
    let radius = Uint::<L>::ONE
        .shl_vartime(response_bits)
        .wrapping_sub(&Uint::<L>::ONE)
        .wrapping_mul(&lattice_content);
    let (resp, resp_d) =
        lattice_sample_from_ball::<L, R>(&hom_b, &hom_d, &radius, p, max_trials, rng)?;
    Some((resp, resp_d, lattice_content))
}

/// Backtracking analysis of the response quaternion (sign step 3 tail). Port of
/// C `compute_backtracking_signature` (`sign.c`): make the response primitive in
/// `O_0`, its content's 2-adic valuation is the backtracking length, and the
/// lattice content with that 2-power removed is `remain`. Returns
/// `(backtracking, remain, primitive_resp)`.
pub(crate) fn compute_backtracking_signature<const L: usize>(
    resp_o0: &[Int<L>; 4],
    lattice_content: &Uint<L>,
) -> (u32, Uint<L>, [Int<L>; 4]) {
    use crate::quaternion::o0_mul::make_primitive_from_o0_coords;
    let (primitive, content) = make_primitive_from_o0_coords::<L>(resp_o0);
    let backtracking = content.abs().trailing_zeros();
    // remain = lattice_content / 2^backtracking (the 2-power shared with the
    // challenge, "backtracked", is removed).
    let remain = lattice_content.wrapping_shr(backtracking);
    (backtracking, remain, primitive)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quaternion::Quaternion;
    use crate::quaternion::ideal::LeftIdeal;
    use crate::quaternion::o0_mul::{multiply_o0_basis, standard_to_o0_basis};

    type L8 = LeftIdeal<8>;

    /// A small concrete left `O_0`-ideal in O_0-coords for testing: `O_0·γ + n·O_0`
    /// with γ = 1 + 2i (N_red 5), n = 5 — a genuine norm-5 left ideal.
    fn sample_ideal() -> L8 {
        let p = crate::params::lvl1::prime().resize::<8>();
        let g = Quaternion::<8>::new(
            Int::<8>::from_i64(1),
            Int::<8>::from_i64(2),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        );
        let (basis, denom, norm) = crate::quaternion::o0_mul::quat_lideal_create::<8>(
            &g,
            &Int::<8>::from_i64(1),
            &Uint::<8>::from_u64(5),
            &p,
        );
        crate::quaternion::o0_mul::c_ideal_to_left_ideal::<8>(&basis, &denom, &norm)
    }

    #[test]
    fn backtracking_extracts_two_adic_content() {
        // resp = (4, 8, 0, 0): content = gcd = 4, v2(4) = 2 ⇒ backtracking 2;
        // primitive = (1, 2, 0, 0); remain = 20 >> 2 = 5.
        let resp = [
            Int::<8>::from_i64(4),
            Int::<8>::from_i64(8),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        ];
        let (bt, remain, prim) =
            compute_backtracking_signature::<8>(&resp, &Uint::<8>::from_u64(20));
        assert_eq!(bt, 2, "backtracking = v2(content)");
        assert_eq!(remain, Uint::<8>::from_u64(5), "remain = content / 2^bt");
        assert_eq!(
            prim,
            [
                Int::<8>::from_i64(1),
                Int::<8>::from_i64(2),
                Int::<8>::from_i64(0),
                Int::<8>::from_i64(0)
            ],
            "primitive response",
        );
    }

    #[test]
    fn conjugate_is_involution_on_a_lattice() {
        let ideal = sample_ideal();
        let once = lattice_conjugate::<8>(&ideal.basis);
        let twice = lattice_conjugate::<8>(&once);
        assert_eq!(twice, ideal.basis, "conj∘conj = identity on each generator");
    }

    #[test]
    fn dual_of_dual_recovers_the_lattice() {
        let ideal = sample_ideal();
        let (d1, dn1) = lattice_dual::<8>(&ideal.basis, &ideal.denom);
        let (d2, dn2) = lattice_dual::<8>(&d1, &dn1);
        // dual(dual(L)) == L as a rational lattice (equals_lattice ignores norm).
        let recovered = LeftIdeal::<8>::with_denom_and_norm(d2, dn2, ideal.cached_norm);
        assert!(
            recovered.equals_lattice(&ideal),
            "dual∘dual must recover the original lattice",
        );
    }

    #[test]
    fn intersect_with_self_is_identity() {
        let ideal = sample_ideal();
        let (b, d) = lattice_intersect::<8>(&ideal.basis, &ideal.denom, &ideal.basis, &ideal.denom);
        let res = LeftIdeal::<8>::with_denom_and_norm(b, d, ideal.cached_norm);
        assert!(res.equals_lattice(&ideal), "L ∩ L = L");
    }

    #[test]
    fn intersect_with_superlattice_is_the_sublattice() {
        let ideal = sample_ideal();
        let o0 = LeftIdeal::<8>::full_order();
        let (b, d) = lattice_intersect::<8>(&o0.basis, &o0.denom, &ideal.basis, &ideal.denom);
        let res = LeftIdeal::<8>::with_denom_and_norm(b, d, ideal.cached_norm);
        assert!(res.equals_lattice(&ideal), "O_0 ∩ I = I for I ⊆ O_0");
    }

    /// ROOT-CAUSE RECORD: `hnf_4x4` preserves the row-lattice of the dual
    /// basis `du` (each `du` row ∈ HNF span), but `hnf_mod_core` does NOT — for
    /// any modulus — even though it works for other inputs (`prime_norm_reduce`,
    /// `add(O_0,I)`). ⇒ `lattice_add` must use an `hnf_4x4`-compatible
    /// (lattice-preserving) HNF, not `hnf_mod_core`. This is the `intersect` fix.
    #[test]
    fn hnf_mod_core_does_not_preserve_dual_lattice_but_hnf_4x4_does() {
        use crate::quaternion::hnf::{hnf_4x4, hnf_mod_core};
        use crate::quaternion::ideal::det_4x4;
        let ideal = sample_ideal();
        let (du, _dn) = lattice_dual::<8>(&ideal.basis, &ideal.denom);
        let det = det_4x4::<8>(&du).abs();
        let gens4: [[Int<8>; 4]; 4] = [du[0], du[1], du[2], du[3]];
        assert!(
            LeftIdeal::<8>::new(hnf_4x4::<8>(&du)).contains(&du[0]),
            "hnf_4x4 preserves the lattice",
        );
        assert!(
            !LeftIdeal::<8>::new(hnf_mod_core::<8>(&gens4, &det)).contains(&du[0]),
            "hnf_mod_core does NOT preserve this lattice (the lattice_add/intersect bug)",
        );
    }

    #[test]
    fn add_with_superlattice_is_the_superlattice() {
        // I ⊆ O_0 ⇒ O_0 + I = O_0.
        let ideal = sample_ideal();
        let o0 = LeftIdeal::<8>::full_order();
        let (b, d) = lattice_add::<8>(&o0.basis, &o0.denom, &ideal.basis, &ideal.denom);
        let res = LeftIdeal::<8>::with_denom_and_norm(b, d, o0.cached_norm);
        assert!(res.equals_lattice(&o0), "O_0 + I = O_0 for I ⊆ O_0");
    }

    #[cfg(all(feature = "alloc", feature = "kat"))]
    #[test]
    fn compute_response_quat_element_composes() {
        use crate::rng::NistPqcRng;
        let p = crate::params::lvl1::prime().resize::<8>();
        let o0 = LeftIdeal::<8>::full_order();
        let chall = sample_ideal(); // norm 5, ⊆ O_0
        let mut rng = NistPqcRng::new(&[0x55u8; 48]);
        // secret = O_0, commit = O_0 ⇒ chall_secret = chall, hom = chall.
        let (resp, _rd, lattice_content) =
            compute_response_quat_element::<8, _>(&o0, &chall, &o0, &p, 12, 1 << 12, &mut rng)
                .expect("response quaternion composes");
        // The full chain (ideal∩ → conjugate → lattice∩ → sample) runs and
        // yields a non-zero lattice element with a positive lattice content.
        assert!(
            lattice_content != Uint::<8>::ZERO,
            "lattice content is positive",
        );
        assert!(
            resp != [Int::<8>::from_i64(0); 4],
            "response is a non-zero lattice element",
        );
    }

    #[cfg(all(feature = "alloc", feature = "kat"))]
    #[test]
    fn sample_from_ball_returns_an_element_inside_the_ball() {
        use crate::quaternion::lattice::qf_eval_4x4;
        use crate::quaternion::o0_mul::o0_reduced_norm_gram_matrix;
        use crate::rng::NistPqcRng;
        let ideal = sample_ideal(); // norm 5
        let p = crate::params::lvl1::prime().resize::<8>();
        let radius = Uint::<8>::from_u64(1000); // generous: ≥ the lattice minimum
        let mut rng = NistPqcRng::new(&[0x33u8; 48]);
        let (e, d) = lattice_sample_from_ball::<8, _>(
            &ideal.basis,
            &ideal.denom,
            &radius,
            &p,
            1 << 12,
            &mut rng,
        )
        .expect("a lattice element within the ball exists");
        // e is a genuine lattice element.
        assert!(ideal.contains(&e), "sampled element ∈ lattice");
        // N_red(e/d) ≤ radius  ⟺  qf_eval(o0_gram, e) ≤ 4·radius·d².
        let gram = o0_reduced_norm_gram_matrix::<8>(&p);
        let qf = qf_eval_4x4::<8>(&e, &gram);
        let bound = *radius
            .wrapping_mul(&d.wrapping_mul(&d))
            .wrapping_mul(&Uint::<8>::from_u64(4))
            .as_int();
        assert!(qf != Int::<8>::from_i64(0), "non-zero element");
        assert!(qf.abs() <= bound.abs(), "reduced norm within the ball");
    }

    #[test]
    fn conjugate_preserves_norm_form_and_closure() {
        // The conjugate of a left ideal is still a full-rank lattice (its det
        // magnitude — hence index — is unchanged by per-generator conjugation,
        // since o0_conjugate is a unimodular-up-to-sign coordinate map).
        let ideal = sample_ideal();
        let conj = lattice_conjugate::<8>(&ideal.basis);
        let d_orig = crate::quaternion::ideal::det_4x4::<8>(&ideal.basis).abs();
        let d_conj = crate::quaternion::ideal::det_4x4::<8>(&conj).abs();
        assert_eq!(d_orig, d_conj, "conjugation preserves the lattice index");
        // Sanity: multiply_o0_basis still well-defined on the conjugated gens.
        let p = crate::params::lvl1::prime().resize::<8>();
        let e0 = standard_to_o0_basis::<8>(&Quaternion::<8>::new(
            Int::<8>::from_i64(1),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        ));
        let _ = multiply_o0_basis::<8>(&e0, &conj[0], &p);
    }
}
