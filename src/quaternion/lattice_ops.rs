//! Quaternion-lattice operations for the signing response (`sign.c` step 3):
//! conjugate, dual, sum, and intersection of `O_0`-coordinate lattices.
//!
//! Port of the C reference `src/quaternion/ref/generic/lattice.c`
//! (`quat_lattice_*`). Unlike the C reference — which works in standard
//! `(1, i, j, k)` coordinates — these operate in the `O_0`-basis coordinates
//! that the rest of the Rust quaternion module uses ([`crate::quaternion::ideal::LeftIdeal`]'s `basis`),
//! so conjugation uses [`crate::quaternion::o0_mul::o0_conjugate`]
//! and the reduced-norm Gram form is
//! [`crate::quaternion::o0_mul::o0_reduced_norm_gram_matrix`].
//! The dual/sum/intersection are pure `Z`-module operations and are
//! coordinate-system independent given a consistent basis.

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

    // Reduce-as-you-go: `lattice_dual` returns the RAW adjugate (entries ~‖B‖³,
    // denom = det ~‖B‖⁴). For the dual of a norm-N ideal those ~N³ entries share
    // an ~N² common factor with the denom, so reducing each dual to lowest terms
    // collapses it back to ~N-sized entries. Without this the un-reduced first
    // duals feed the sum and the SECOND dual cubes them again → the ~2^4700
    // dual-of-dual blowup that forced W=96. Reducing here keeps the working width
    // at "one dual" (~‖B‖⁴) instead of "dual-of-dual" (~‖B‖¹²). `reduce_denom`
    // is exact division by gcd ⇒ lattice-preserving (identical rational lattice).
    let reduce = |b: &[[Int<L>; 4]; 4], d: Uint<L>| -> ([[Int<L>; 4]; 4], Uint<L>) {
        let (rb, rd) = quat_lattice_reduce_denom::<L>(b, d.as_int());
        (rb, rd.abs())
    };

    let (du1, dn1) = lattice_dual::<L>(b1, d1);
    let (du1, dn1) = reduce(&du1, dn1);
    let (du2, dn2) = lattice_dual::<L>(b2, d2);
    let (du2, dn2) = reduce(&du2, dn2);
    let (sum_b, sum_d) = lattice_add::<L>(&du1, &dn1, &du2, &dn2);
    let (res_b, res_d) = lattice_dual::<L>(&sum_b, &sum_d);
    // Reduce the denominator to lowest terms (HNF is optional — the C ref notes
    // "could be removed"; `equals_lattice` and downstream don't require it).
    let (rb, rd) = reduce(&res_b, res_d);
    (rb, rd)
}

/// Convert an `O_0`-coords lattice (`basis_o0` rows are `Z`-generators in the
/// `(1, i, (i+j)/2, (1+k)/2)` basis, scaled by `denom_o0`) into C's standard
/// `(1, i, j, ij)` `quat_lattice_t` form, in Hermite Normal Form — the frame
/// the response sampler (`lat_ball.c`) works in.
///
/// Two steps, both byte-faithful to C:
/// 1. `O_0 → standard` change of basis (scaled by 2 to clear the halves in the
///    `O_0` basis): `std_gen[r] = Σ_c basis_o0[r][c]·S[c]` with rows of `S` the
///    doubled `O_0` basis vectors in standard coords —
///    `2·1, 2·i, 2·(i+j)/2, 2·(1+k)/2` = `(2,0,0,0) (0,2,0,0) (0,1,1,0)
///    (1,0,0,1)`; the denominator picks up the factor 2.
/// 2. `quat_lattice_hnf` (`lattice.c`): HNF the generators modulo `|det|`, then
///    reduce the denominator to lowest terms.
///
/// Returns C's `(basis, denom)` — `basis` column-major (columns are generators,
/// as `quat_lattice_t` stores them), `denom` dividing it. Byte-exact to C's
/// response `lattice_hom_chall_to_com` (validated by
/// `o0_lattice_to_standard_hnf_matches_c_record0`).
#[cfg(feature = "alloc")]
pub(crate) fn o0_lattice_to_standard_hnf<const N: usize>(
    basis_o0: &[[Int<N>; 4]; 4],
    denom_o0: &Uint<N>,
) -> ([[Int<N>; 4]; 4], Uint<N>) {
    use crate::quaternion::hnf::{hnf_mod_core, quat_lattice_reduce_denom};
    use crate::quaternion::ideal::det_4x4;

    // 1. O_0 → standard (×2).
    const S: [[i64; 4]; 4] = [[2, 0, 0, 0], [0, 2, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]];
    let mut std_gens = [[Int::<N>::from_i64(0); 4]; 4];
    for (r, genv) in std_gens.iter_mut().enumerate() {
        for (k, cell) in genv.iter_mut().enumerate() {
            let mut acc = Int::<N>::from_i64(0);
            for c in 0..4 {
                if S[c][k] != 0 {
                    let term = basis_o0[r][c].wrapping_mul(&Int::<N>::from_i64(S[c][k]));
                    acc = acc.wrapping_add(&term);
                }
            }
            *cell = acc;
        }
    }
    let denom_std = *denom_o0.wrapping_mul(&Uint::<N>::from_u64(2)).as_int();

    // 2. quat_lattice_hnf: HNF the generator vectors modulo |det|, reduce denom.
    let modulus = det_4x4::<N>(&std_gens).abs();
    let hnf = hnf_mod_core::<N>(&std_gens, &modulus);
    let (basis, denom) = quat_lattice_reduce_denom::<N>(&hnf, &denom_std);
    (basis, denom.abs())
}

/// Compute the SQIsign response quaternion (sign step 3). Port of C
/// `compute_response_quat_element` (`sign.c:66`): intersect the challenge ideal
/// with the secret ideal, intersect that with the conjugate of the commitment
/// ideal to get the `Hom(E_chall → E_com)`-lattice, then sample a short element
/// (reduced norm ≤ `(2^response_bits − 1)·lattice_content`). Returns
/// `(resp O_0-coords, resp_denom, lattice_content)`. Byte-exact: bridges the
/// hom-lattice into C's standard coords and samples via
/// [`crate::quaternion::lll::sample_from_ball`].
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
    // radius (C `bound`) = (2^response_bits − 1) · lattice_content.
    let radius = Uint::<L>::ONE
        .shl_vartime(response_bits)
        .wrapping_sub(&Uint::<L>::ONE)
        .wrapping_mul(&lattice_content);

    // Byte-exact response: sample in C's standard coords. Bridge the O_0
    // hom-lattice to C's standard `quat_lattice_t` (HNF), rejection-sample the
    // ball (`sample_from_ball` = `lat_ball.c`), then map the standard-coords
    // result back to O_0 coords for the downstream (backtracking / aux). Because
    // the sampled element lies in `O_0`, `standard_to_o0_basis` of its numerators
    // is exactly `2·(O_0 coords)`, so pairing it with `std_denom` (= 2) divides
    // out cleanly — matching the stand-in's `(coords, denom)` contract.
    let (std_basis, std_denom) = o0_lattice_to_standard_hnf::<L>(&hom_b, &hom_d);
    let coord_std = crate::quaternion::lll::sample_from_ball::<L, R>(
        &std_basis, &std_denom, &radius, p, max_trials, rng,
    )?;
    #[cfg(feature = "kat")]
    if std::env::var_os("PQSQ_RESP_DUMP").is_some() {
        let dby = std_denom.to_le_bytes();
        std::eprint!("OURS_XRES xden ");
        for x in &dby[..16] {
            std::eprint!("{x:02x}");
        }
        std::eprintln!();
        for (c, e) in coord_std.iter().enumerate() {
            let neg = e < &Int::<L>::from_i64(0);
            let by = e.abs_sign().0.to_le_bytes();
            std::eprint!("OURS_XRES xc{c} neg={} ", neg as u8);
            for x in &by[..64] {
                std::eprint!("{x:02x}");
            }
            std::eprintln!();
        }
    }
    // Map the standard-coords result to O_0 coords, as the *integral* element.
    // `standard_to_o0_basis(coord_std)` equals `std_denom · (true O_0 coords)`
    // (the element lies in `O_0`), so dividing by `std_denom` recovers the
    // integer O_0 coordinates the downstream consumes (it takes `resp` as the
    // response element directly; the returned denom is 1).
    let q = crate::quaternion::Quaternion::<L>::new(
        coord_std[0],
        coord_std[1],
        coord_std[2],
        coord_std[3],
    );
    let o0_scaled = crate::quaternion::o0_mul::standard_to_o0_basis::<L>(&q);
    let sd = crypto_bigint::NonZero::new(std_denom).into_option()?;
    let mut resp = [Int::<L>::from_i64(0); 4];
    for (r, num) in resp.iter_mut().zip(o0_scaled.iter()) {
        let neg = bool::from(num.is_negative());
        let (quo, _rem) = num.abs().div_rem_vartime(&sd);
        let qi = *quo.as_int();
        *r = if neg { qi.wrapping_neg() } else { qi };
    }
    Some((resp, Uint::<L>::ONE, lattice_content))
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

    /// Byte-exact reproduction of C's `quat_lattice_bound_parallelogram`
    /// (`lat_ball.c`) via the `dpe`-float `quat_lll_core`, against 5 captured
    /// lvl1-KAT dumps (boxes `[0,2,2,2] [1,2,2,2] [0,2,2,3] [1,1,2,2] [1,1,1,2]`;
    /// one `bound_parallelogram` call per signature, 100 in the lvl1 KAT, first 5
    /// pinned here).
    ///
    /// VERDICT — the float LLL matches C on both `box` AND transform `U`,
    /// byte-for-byte, on every record. This is the whole point: exact *integer*
    /// LLL reproduces the `box` but NOT `U` (record 1: 3/4 sampling-relevant rows
    /// diverge — C's float LLL picks a valid but different reduced basis), so a
    /// byte-exact response sampler requires the floating-point reducer. Column
    /// convention matches C (`quat_lll_core` tracks the basis in columns), so the
    /// dumped `U` is reproduced exactly with no transpose fixup.
    ///
    /// Recipe (mirrors `lat_ball.c` lines 18-44):
    ///   dualG, denom = adjugate_with_det(G)          // inv·det, det
    ///   quat_lll_core(dualG, U=identity)             // dpe float L² reduce
    ///   box[i]       = floor_sqrt(dualG_red[i][i]·rad / denom)
    ///   U_final      = inv(U) = adjugate(U)·det(U)   (det = ±1)
    #[cfg(feature = "kat")]
    #[test]
    fn bound_parallelogram_float_matches_c_records() {
        use crate::quaternion::lll::bound_parallelogram;

        struct Rec {
            g: [[(u8, &'static str); 4]; 4],
            u: [[(u8, &'static str); 4]; 4],
            bx: [u64; 4],
            rad: &'static str,
        }

        // Decode a little-endian hex string (C `ibz_to_digit_array` byte order)
        // into a 256-limb signed integer, applying the sign flag.
        fn dec(neg: u8, h: &str) -> Int<256> {
            let mut bytes = [0u8; 256 * 8];
            for (k, byte) in bytes.iter_mut().enumerate().take(h.len() / 2) {
                *byte = u8::from_str_radix(&h[2 * k..2 * k + 2], 16).unwrap();
            }
            let u = Uint::<256>::from_le_slice(&bytes);
            let i = *u.as_int();
            if neg == 1 { i.wrapping_neg() } else { i }
        }
        fn dec_mat(m: &[[(u8, &str); 4]; 4]) -> [[Int<256>; 4]; 4] {
            let mut out = [[Int::<256>::from_i64(0); 4]; 4];
            for (orow, mrow) in out.iter_mut().zip(m.iter()) {
                for (oe, me) in orow.iter_mut().zip(mrow.iter()) {
                    *oe = dec(me.0, me.1);
                }
            }
            out
        }

        // 5 records captured from C lvl1 KAT (PQSQ_BALL_DUMP)
        let records: &[Rec] = &[
            Rec {
                g: [
                    [
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c8ead6b090c8b7dee5fda95bbed91ab22838d531e35879f5174b3f00d04bfde489242b0074eaf383299148c4e54c9b3e7870df91dfdf309380de6eeeb84f8f60f3ef48421a0000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000090e0fe51e9a2aae8c4a510b1fe7bb9648281a17aa79367a4253cbfd0f2f71699f6a557d0bfbb563478351ce5bda0b20cb1d5493f502236faf936708668fd2ef32c3823f1b117e0184a335253074104b1f1400f90b97332e60edaac66e688cb463b9d550e0000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000743220533c23ecd9305e7db1b65658e69c7d7837115ff13010f3ea60b12495664ea60c318dd01904fc555cf52e46af8ab1becf3b0917655ae3ed55ffeb2b283195afebe30cb72da9271e996e58765d2df1919a870f0bd1d2a826a0edaf59054751f9c0140000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c8ead6b090c8b7dee5fda95bbed91ab22838d531e35879f5174b3f00d04bfde489242b0074eaf383299148c4e54c9b3e7870df91dfdf309380de6eeeb84f8f60f3ef48421a0000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "000000000000000000000000000000000000000000000000000000000000008ccddfacc3dc1326cfa1824e49a9a719638287c8eea00ecfef0c159f4edb6a99d2dd8fe9e98364316d4723b1d870daedd473192b32fc9da5f96345fe881c46fc0e329c2b354dd737c8d42e6159c6650ef535375d68e012edbd084a9e2827ef51c95d350f0000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000090e0fe51e9a2aae8c4a510b1fe7bb9648281a17aa79367a4253cbfd0f2f7161987734cd5dba2a4c809baf74d82132fce8a89b7fcd196b5a563a83abe09cb8f943d3c0a2b7d60eed47e5d84db9363c33724a3c3b0931382cb015f9f3b454a60f432c7ce0f0000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000090e0fe51e9a2aae8c4a510b1fe7bb9648281a17aa79367a4253cbfd0f2f71699f6a557d0bfbb563478351ce5bda0b20cb1d5493f502236faf936708668fd2ef32c3823f1b117e0184a335253074104b1f1400f90b97332e60edaac66e688cb463b9d550e0000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "000000000000000000000000000000000000000000000000000000000000008ccddfacc3dc1326cfa1824e49a9a719638287c8eea00ecfef0c159f4edb6a99d2dd8fe9e98364316d4723b1d870daedd473192b32fc9da5f96345fe881c46fc0e329c2b354dd737c8d42e6159c6650ef535375d68e012edbd084a9e2827ef51c95d350f0000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000028fd9b590f55d19da32413eaff57a8b2170952cb0e175f3e8a1a4f81a15c28ff44dc0d8a939f67a0f331178437f1981a6b54f4858b5e2ea7b211dbce186bb95b9c646cfb28a1168dbb37d3b16c0aaa814b3f86eac2aa800de047d7ce79645fcd8921a2100000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000c080a84bfbed2f35e806f00d28d966de6caa0cecbfa211f8ee1dba1c6aefd08dd01a56d71a705c9faf7742e89d84a6ef9ac090465f519601c7a665fabe9c5daa533d7d2330161c079df4215500363d7916b60d2eeb57c4a33c4b75856e7d2d51bd127c140000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000743220533c23ecd9305e7db1b65658e69c7d7837115ff13010f3ea60b12495664ea60c318dd01904fc555cf52e46af8ab1becf3b0917655ae3ed55ffeb2b283195afebe30cb72da9271e996e58765d2df1919a870f0bd1d2a826a0edaf59054751f9c0140000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000090e0fe51e9a2aae8c4a510b1fe7bb9648281a17aa79367a4253cbfd0f2f7161987734cd5dba2a4c809baf74d82132fce8a89b7fcd196b5a563a83abe09cb8f943d3c0a2b7d60eed47e5d84db9363c33724a3c3b0931382cb015f9f3b454a60f432c7ce0f0000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000c080a84bfbed2f35e806f00d28d966de6caa0cecbfa211f8ee1dba1c6aefd08dd01a56d71a705c9faf7742e89d84a6ef9ac090465f519601c7a665fabe9c5daa533d7d2330161c079df4215500363d7916b60d2eeb57c4a33c4b75856e7d2d51bd127c140000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "000000000000000000000000000000000000000000000000000000000000002824bce9dc72d039c2ed859093f8b529ba19259c621861e83145e605bbcaae6c10889633011e1cdd98c55128672036d6080c0c2bc81cc544ca1e31108f5143f4ac8863dc2999f7c282489492f9e6a93880ad42323401aeddb05deb22aba1aa456e42eb190000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                ],
                u: [
                    [
                        (
                            1,
                            "ecda2bf176a18779fc789c8ea08f097e425dd3e19780c2b27c0000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "7b59a95c241a2505004409a5e4a143ae3aa8f686afacc83a750000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "821942243518ac602a551829be6831926cf2c785ddfc7f2e880000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "e8a324937816e0a501fa5157d4ede3f676ab003c538662b73f0000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            1,
                            "55479d2f52e2327cd181753bde4611ce73d30a713cea6be1240000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "8c4eb56a0627a1f9717d80606c020a92eff0d9cd2fc0165e1d0000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "929318de5ae08c14993d4a3445b232f6cda6a7448e30fbcf070000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "57a1655e17cddcadcb5b0f2bc21d19ea9c87cfc413f07844290000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            1,
                            "bef961ec68e5c325f192494d83a4423ead0d88fb6e3b754a150000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "334484f9b1643c75698d0738b6921b83b85e754300bd19380d0000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "51234611937c9bc26fecd8cdf2806b6c73276f577833f258120000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "96f90a1e4537897fa18b784ff4cb1ca2f8b6ce25ed3d5e9c270000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            1,
                            "c0c1969a7ab0a6e37b222b9acb8dbd9caf7587f72fbdda6a050000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "7cb62a3046878cf4617c79f4317fb682949d67754c868702020000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "74f3e35e36002cb0d0915f2542f30f39d2a019223c2f34f40c0000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "c128b9e785af7743b79fd97d2cf03105c4d4d90b861c32cd0f0000000000000000000000000000000000000000000000",
                        ),
                    ],
                ],
                bx: [0, 2, 2, 2],
                rad: "0000000000000000000000000000000000000000000000000000000000000028e16f682fc85436a969b7d1a538ba567090332b5a8889e96e3be9e000ffc31a659f9d7e665138b2893aaa6ab55ce3139b9f030000000000000000000000000000",
            },
            Rec {
                g: [
                    [
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c8dcce04f4898bddc5c5b0251ebbaea812422415fdffe32e03996fed5eb8d258fd5a0ea9b9586d435bcc7aedc36db8eb07514b34684a5470b013e23a0e18b78670424ecc1f0200000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000646e6702fac4c5eee262d8128f5d57540921928afeff719781ccb7762f5c69ac7e2d87d45cacb6a12d66bdf6e136dcf583a8251a34252a38d809711d078c5b43382127e60f0100000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "000000000000000000000000000000000000000000000000000000000000002842a267560bc2819a8a03cc6e0b4dee2bd2cc39a8d6b46b41ebb54a95f69d2144139a226be6813d379a3f02ff153d6bd98393b46ff85e19f9ce59e0f443eea7afab3f2d0dfed14c992fd0725a6ecba1b30232b96f92bfeaf1680117d6abe58eee33071c0200000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000542e15bc7d3c8bf7778c35db2543414d44d53ad7dfe796db8c3a5b4d8de0bf75f578ceff45990eca299ef19e64ae5a903e76d1f5c5756e4acb4e51ace875498d680c909b48a39fce5d6d971ed0438d478302e46e4242cd5145ffc60df23610cfadf4bf9d0000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000646e6702fac4c5eee262d8128f5d57540921928afeff719781ccb7762f5c69ac7e2d87d45cacb6a12d66bdf6e136dcf583a8251a34252a38d809711d078c5b43382127e60f0100000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000646e6702fac4c5eee262d8128f5d57540921928afeff719781ccb7762f5c69ac7e2d87d45cacb6a12d66bdf6e136dcf583a8251a34252a38d809711d078c5b43382127e60f0100000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000d4138dabd8ce368a22fecdf048c80ba1e7fc9162c8ee1d90b4b05afd0716de7b657b3eb3fc3e7177bc6c87f544fe97d92b52444b24544f1035da379a7a6948cbef1ad72be3b2eb15fc40ec324941c3ca8de2fd428791940d90125bc6a4bd31c436754d680100000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "000000000000000000000000000000000000000000000000000000000000001421d133ab05e1404dc50166b78526f71569e61c546bdab5a0f55aa54afbce7614a9e4359ac0d15fdea9371b0a9a380187992bde1a707c0abc93303aa17c263a2b0fcce24bbc4e40bda48c540ada1f7874a61a2626d81cb21d6105b58ec48a5860ac27820000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "000000000000000000000000000000000000000000000000000000000000002842a267560bc2819a8a03cc6e0b4dee2bd2cc39a8d6b46b41ebb54a95f69d2144139a226be6813d379a3f02ff153d6bd98393b46ff85e19f9ce59e0f443eea7afab3f2d0dfed14c992fd0725a6ecba1b30232b96f92bfeaf1680117d6abe58eee33071c0200000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000d4138dabd8ce368a22fecdf048c80ba1e7fc9162c8ee1d90b4b05afd0716de7b657b3eb3fc3e7177bc6c87f544fe97d92b52444b24544f1035da379a7a6948cbef1ad72be3b2eb15fc40ec324941c3ca8de2fd428791940d90125bc6a4bd31c436754d680100000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000181f3e2b390c6876df190b82e0588f0a3fd64f0ed3c43e08a961d0ac8116cf48a560c04cd981e8ad13b2dc1078f31dde76ed314fbd0ec61e433b1c414ac65ccb397c3142c6346c202af4731ff070b81f791981c39ec0ddb20cc177864b42ed3399743f540200000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "000000000000000000000000000000000000000000000000000000000000005c8f6b2ec4c9aa9af8832984e2dad034a878df99eed505ca8941026d474cc1fe7ea2bcc7b8fa75e04daba2c9fd3626f5360c18b5eb76fbbd63b0267c2d4bd578d39ba920a66b40e06ec43bb234ff74b66fc98acac0cb795f2b2fd248bf2e8a062885b6be0000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000542e15bc7d3c8bf7778c35db2543414d44d53ad7dfe796db8c3a5b4d8de0bf75f578ceff45990eca299ef19e64ae5a903e76d1f5c5756e4acb4e51ace875498d680c909b48a39fce5d6d971ed0438d478302e46e4242cd5145ffc60df23610cfadf4bf9d0000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "000000000000000000000000000000000000000000000000000000000000001421d133ab05e1404dc50166b78526f71569e61c546bdab5a0f55aa54afbce7614a9e4359ac0d15fdea9371b0a9a380187992bde1a707c0abc93303aa17c263a2b0fcce24bbc4e40bda48c540ada1f7874a61a2626d81cb21d6105b58ec48a5860ac27820000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "000000000000000000000000000000000000000000000000000000000000005c8f6b2ec4c9aa9af8832984e2dad034a878df99eed505ca8941026d474cc1fe7ea2bcc7b8fa75e04daba2c9fd3626f5360c18b5eb76fbbd63b0267c2d4bd578d39ba920a66b40e06ec43bb234ff74b66fc98acac0cb795f2b2fd248bf2e8a062885b6be0000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000028481e2762f9a861d033e63b0d4b9989891c99639f8888f2d9f17e6e4b24b23954954c2123db00ec52159f218f2702170522d7ba74d46584a09900e9152222440f70c3b0b0a0ca3de9ae15d224961a82c6cf3c234764138199f402f34ed34a56a6b21a410000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                ],
                u: [
                    [
                        (
                            1,
                            "357337c80c64b25d8511450a5efa15383c474d1ae06c4770280000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "9939bd76c1066b4f5e2a4864c9b13bc09170f5a90bc7a20c5b0000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "34e624f9deec89c3ee01de5a7ec4c6fbc205520d4e86100d210000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "86a0d53bc6f3ee10c44111820127c5e2d3aee260a4f49b30b70000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "571edec5dbceb7e738b20fa6976f14b169a9420c7eda049a250000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "3a52cf406ce1f43f93feeed93f9260d87cdcc2c74b3cd6e8060000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "4ce3927b86de616f9833ec09bfd81167490bd882beb66b1b4a0000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "d0519761bc22b3085907c5da6ce3058857cb1db096227929700000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "2255d3656fab11ec1fdd2e637a67c8f04cff7da9401499281c0000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "cfdddcd414c29e63c0a21e011862ae5e330ea8f5caafff700a0000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "9b237d1166dc2692bc0862957aab97b260f8625c33162835400000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "3912e10391349caa13ac76abef1dc0e0735dddaac5e53dbb8c0000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            1,
                            "ea02168976e7e1b890ae346b1802a0b803b5d18a61901574230000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "c5536953a80d824954cd6f32f025c2fbe0f708878d9da73e0d0000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "4d1106ff681d144cc348a23f08fc58a1a68bf9afd407771e420000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "7c3f49e0164ca5584a8e31e455f84008c7cb7dcbbbd14c4d510000000000000000000000000000000000000000000000",
                        ),
                    ],
                ],
                bx: [1, 2, 2, 2],
                rad: "00000000000000000000000000000000000000000000000000000000000000d8294ede291c5664c8495eeaba8e1212ff1e10b2a5fcfb76044f055441cd9f9d872055f5139ffb5ba2c758fcc3e376c7467d100000000000000000000000000000",
            },
            Rec {
                g: [
                    [
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c8eca2a6f3531ce36e11aecee9be72ace26cca8ab0c93d3d7b3b5f26ed3114be1020228dba62c5f0ef3a4bbaf439f9504a863822e6464b54663a8802bd3a62072b962a4f310000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000c0db63616c5c805fddf579658bbac453e5bdd0146787cb8151115e81aa9c7ef830b9e1d5760c80da0076c9027bcdb0b83328f578609885037ecba9e5e593306ceba788441377135faa8a1c6b44c82f15bbf362f78b7892b18ea5d512775ce9bf39dc09130000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000ecbacf8b148dd893cf850f4b1d79f78c860e794d94774f3ea73ddd8cf8bee3f813f98db07b9ffb80afa41eea424e5799159aad543ee94d747f2cea43a25380e83e36c956e4a00f51a4f8a1ec0e039940e47dee0a30590428553cf61a726bfa507edf9d2d0000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c8eca2a6f3531ce36e11aecee9be72ace26cca8ab0c93d3d7b3b5f26ed3114be1020228dba62c5f0ef3a4bbaf439f9504a863822e6464b54663a8802bd3a62072b962a4f310000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000014453074eb72276c307af0b4e286087379f186b26b88b0c158c2227307411c87e91e82732e5758492608921117a0a58f08d8c1e5029606f8a52713fd1cb94e7ff713131e6b0f7b0fc3701870c70690df6c6954c907937ef5520f379b84abe9909c0602220000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000c0db63616c5c805fddf579658bbac453e5bdd0146787cb8151115e81aa9c7e401f1fa820f039beecaa7ae929bac0fd63961adb17301c8b6c28bd4aabed3224577fee7e009e4da45fcc5bc80dedeaae92c2c5f13fe009867db388fca66cd25469f7b2371a0000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000c0db63616c5c805fddf579658bbac453e5bdd0146787cb8151115e81aa9c7ef830b9e1d5760c80da0076c9027bcdb0b83328f578609885037ecba9e5e593306ceba788441377135faa8a1c6b44c82f15bbf362f78b7892b18ea5d512775ce9bf39dc09130000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000014453074eb72276c307af0b4e286087379f186b26b88b0c158c2227307411c87e91e82732e5758492608921117a0a58f08d8c1e5029606f8a52713fd1cb94e7ff713131e6b0f7b0fc3701870c70690df6c6954c907937ef5520f379b84abe9909c0602220000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000f02cfc606b9eed36a8eaf6bfddd7c17bbc215cf07ae3a7a422643159cc8730f83a7373ec21f1fe0521004ab1c46c49189c736766cd246cf1cdc4268db3101960af188a175df85a84eba2b80d6fb92f57d8d63c183d5c44c6b2f56b8cd039d44c103fce1e0000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000d4a2b10bebc8fcb6c5ccd33402c5c00006213377debdf29a044beabe0b89ddf746533aa72bc6b4824e48f567ecdcf9471c91bb92b2b1e50f4a36a5e50b1c29832223ec316ca791ab678688d99c3347a28c0702e6c4d2f9856c8b9aabfdc89ee581d8b1230000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000ecbacf8b148dd893cf850f4b1d79f78c860e794d94774f3ea73ddd8cf8bee3f813f98db07b9ffb80afa41eea424e5799159aad543ee94d747f2cea43a25380e83e36c956e4a00f51a4f8a1ec0e039940e47dee0a30590428553cf61a726bfa507edf9d2d0000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000c0db63616c5c805fddf579658bbac453e5bdd0146787cb8151115e81aa9c7e401f1fa820f039beecaa7ae929bac0fd63961adb17301c8b6c28bd4aabed3224577fee7e009e4da45fcc5bc80dedeaae92c2c5f13fe009867db388fca66cd25469f7b2371a0000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000d4a2b10bebc8fcb6c5ccd33402c5c00006213377debdf29a044beabe0b89ddf746533aa72bc6b4824e48f567ecdcf9471c91bb92b2b1e50f4a36a5e50b1c29832223ec316ca791ab678688d99c3347a28c0702e6c4d2f9856c8b9aabfdc89ee581d8b1230000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000f0ecaddfadb55e532f8f0372f372298fc03a736f5d8ad0700f6b65e551f169da1bee01a75d9fd6d4d84777c36e3fce6d7d47a2749046b828f7a8d1749da53a1c7ca82fc541fb9bec41b59886f274733e2f817071aa338dd37a2bd2e46dac801460f523380000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                ],
                u: [
                    [
                        (
                            1,
                            "32cc7c74e5afa980ce2ae7026aa873b8366480c18f5c99f02e0000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "5f084faad8b1ff2111659c7f1bc197613cb61f9319fd2fb2660000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "b997544881c455ba6b271bd9d985fa809d29552a56a50bdea10000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "3436ff8fe5349debea7fe0a0ddece0a327202292012936d1100000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            1,
                            "5b57741dcc0a78b0a7ef53593df7cabb4c94e692dd9ad063520000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "5b4d3b9af55be6873a9b78be0ae6a153ddf62c455a955bcc3a0000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "28d9390ee8ffe9480810d28f7b736bbaa843d5436bd0ea77180000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "3f2fcb029029a91da692987c2d3905de9c6190038dbdd7d84e0000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            1,
                            "8bc73d95149818f54e941ef671f4298a8938e960f6fe8b89140000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "c04121c81093ce067b93715c8a49c10eb6157e363222f3570e0000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "aebb59d6c5dc1817f3bd2693ecda133fa1b83bf3ef8f76e5370000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "fe298e268e712430309e4c1185a3d85116a4669888f74e872d0000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            1,
                            "507c24dfc4d3c4079a3fbea2f946449bb1cdcd7bfa364f19100000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "9d555ee599be6b51cc2fffc19a584feb2fd3f518475c0852150000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "ef1fbe948ccf1db190f5d188c1f1d9a4fec39ce5c93c57cc190000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "f93a3200feec5ecfa72350661077bc2edcf24ce3c75f8aa2060000000000000000000000000000000000000000000000",
                        ),
                    ],
                ],
                bx: [0, 2, 2, 3],
                rad: "00000000000000000000000000000000000000000000000000000000000000d8c11fb1aeb9307c0b042a9c7dc9e48026248939e9b0efb1dfcf7ef6c737ad3e56de77c62a08c15bc7ab9d28766546a820f7040000000000000000000000000000",
            },
            Rec {
                g: [
                    [
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000835bf72228f390099da65c4b7ef699a861d624f3047d33fd3b979d1cd86ce4ce61ced40e64b70be6962e2b7ad5c3d53a2fc74d6122e4f7844d43a4c499760c1454a7426000000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000080c1f0227648b7a19ddf6038a970e6b6b9ff6a64d77ae2fd634ac2cfaa75734e2c016bb37aa87e54609822ede9e8b77d6bc8c4c12d7b1bd0aa6a6dc40f8b48884b0089554d7ba6a2e173f57bf4521c87a7558e6a83ca2667eeb12342badeded6512e1f000000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000bccc1027512939c34644a877b5c4346d050fed897f296127ae7cc78f58dd2598828f147c117c17c130da525e169ed95f92223701da2332ae29687b35033b8df1190043f6f1e31782fdc89800d7d385816c66f7460620e3219cdb885d8a9d6cb4d0cb24000000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000835bf72228f390099da65c4b7ef699a861d624f3047d33fd3b979d1cd86ce4ce61ced40e64b70be6962e2b7ad5c3d53a2fc74d6122e4f7844d43a4c499760c1454a7426000000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "000000000000000000000000000000000000000000000000000000000000004433efd8aed6c63cb9bb57884a3bcb92faf0127680d69ed851833870a722dacf386a6d471657a3f0ffcffef0956a65a7a8867904713c0d2a340f25479da071b4f244aee4b7e33f052cdc177c779dbbbe13dce6c6b88960e0881d6d41389766f45a6007000000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000080c1f0227648b7a19ddf6038a970e6b6b9ff6a64d77ae2fd634ac2cfaa7573ce69f099f5db73ef4057100dfdee4fe3d2b586198e5b35e8cf999ddc6189bebfedc91f30b94ab30c8168209ec51be01ebb56ec876c2ea8c8dd2fd6374d04e0fe48dcb51a000000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000080c1f0227648b7a19ddf6038a970e6b6b9ff6a64d77ae2fd634ac2cfaa75734e2c016bb37aa87e54609822ede9e8b77d6bc8c4c12d7b1bd0aa6a6dc40f8b48884b0089554d7ba6a2e173f57bf4521c87a7558e6a83ca2667eeb12342badeded6512e1f000000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "000000000000000000000000000000000000000000000000000000000000004433efd8aed6c63cb9bb57884a3bcb92faf0127680d69ed851833870a722dacf386a6d471657a3f0ffcffef0956a65a7a8867904713c0d2a340f25479da071b4f244aee4b7e33f052cdc177c779dbbbe13dce6c6b88960e0881d6d41389766f45a6007000000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "000000000000000000000000000000000000000000000000000000000000006084171a48c4081fc9726789e6249093b4aa0db1ef86fa0bb98bbcec9601c3f28046c2663d344c9365d300ed2fb75ec7bf17eef315d94e3d46a1766f61c0dee91c3e781f721f2365429a1521a673e05569b26244e2994ee6e5a66d9ab1af9e55c1b21a000000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000401653e080958b7e4278062dd4320eace2e3f7e0f82a844005c42130720215d906c913d235c3f74875e33d921c494c4eb5d18f5b8ffe03d70ee3dc3c70e88548904427ea2a9332a7a983945a33c4e9bb9091a0c457b3384d0f26ec680a4a0422c9f522000000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000bccc1027512939c34644a877b5c4346d050fed897f296127ae7cc78f58dd2598828f147c117c17c130da525e169ed95f92223701da2332ae29687b35033b8df1190043f6f1e31782fdc89800d7d385816c66f7460620e3219cdb885d8a9d6cb4d0cb24000000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000080c1f0227648b7a19ddf6038a970e6b6b9ff6a64d77ae2fd634ac2cfaa7573ce69f099f5db73ef4057100dfdee4fe3d2b586198e5b35e8cf999ddc6189bebfedc91f30b94ab30c8168209ec51be01ebb56ec876c2ea8c8dd2fd6374d04e0fe48dcb51a000000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000401653e080958b7e4278062dd4320eace2e3f7e0f82a844005c42130720215d906c913d235c3f74875e33d921c494c4eb5d18f5b8ffe03d70ee3dc3c70e88548904427ea2a9332a7a983945a33c4e9bb9091a0c457b3384d0f26ec680a4a0422c9f522000000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "000000000000000000000000000000000000000000000000000000000000007875ebabf583d4b9f1e49d2413cc8d5dd73d6dd9f5f8fc1255dff7e063b397b1d1b2aa0c291b7f608bb8b963280dfced56e1a4e75045d34472a6b7530f6e44600681ac0cbfadddfce2802f8f737c5f6cb61091d1abc1c81fc9f762cb182b1f903bc335000000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                ],
                u: [
                    [
                        (
                            0,
                            "f7c1d557bf7b8685dd54c100ad1f1d6acce7e25ac2502de4040000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "c324b5c62a1b995b9e234a491272d7528d5d76b46d353d50090000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "e85f6c49256adcd079d38b9b9987f18b2fb23cc902cd20860e0000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "ed4d7644e400a4124cf69eb34dc49d8b10cdddcc08ed546b110000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "982654aa5272fb8853a874781525ac4f02a2e80dfb06c943040000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0500d57379f59decaf600a9d7c745c6ee82d3169322efe7c050000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "c1519291b1eb06970d36398a037ec781537e7db3ae489507060000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "ea4124a029321c0df4cbd806b51f0b5adce14544bca50291090000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            1,
                            "09f14854039a4108cbaff4128cfa562ef59b6c0874da98ef170000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "b86961cde13fef0433a3e42e851c4a8bfb2c9f9af8ce0c4d0b0000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "4b7a4650e1566de24968348e0829d2407ac844c836dfff4e0f0000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "f8b3ab47e3a2998b7e46a96e75dbf9fd232ec2d56b77c50a0c0000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "b23c574f6d02c7d05b64e7971236676419a537babfc944d7000000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "e9231e07156b44b52d54f9abbb95ab1e3b564e527662b50a000000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "022f607c714d0a74e3655dd655ea8ea10ccf2a4bfd66d1a4010000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "f4fd255863bae6be23c09a67836c1ee00114528dd96fa183000000000000000000000000000000000000000000000000",
                        ),
                    ],
                ],
                bx: [1, 1, 2, 2],
                rad: "000000000000000000000000000000000000000000000000000000000000007809c62199786477601be3798ff8c45b02132b2ce4d24824043774037eaf1e1e06c2d762bd8377ef88c084879704bc722846000000000000000000000000000000",
            },
            Rec {
                g: [
                    [
                        (
                            0,
                            "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000088ae00576e54141d6f8cadfe31ebdc03bd635f053c0413165995264cddae3c9bc732420e7abd71b0e88f9c950237081f0bc3601ce6f06f4a28340ed537c35dbb1db8652b2f0000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000030caa788af49c394912be6f445ccf38115b2a537316b81a6b96c229ed213f7c21ba1c6f19ad07b0d8522e1b635bd8d410cae99f894b3045fc1b199dc87572f347dfe99c0fd08647c24cb71ac23b4ac28676bf7d2164c0e4d2a4b7bf3b969dfe730aaa1230000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "000000000000000000000000000000000000000000000000000000000000007c959398e59cd93770cb16b25b0838a40f93c4e27feef2fa50976705f475a22420988b00683d490b4295a44ffff216607ae55d7d355a1e3c2379ca6eb2e2fe23900413227e813c3e0fd2b47a0e98db687464cd409e3402f952f46abd28369495a84bc71e0000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000088ae00576e54141d6f8cadfe31ebdc03bd635f053c0413165995264cddae3c9bc732420e7abd71b0e88f9c950237081f0bc3601ce6f06f4a28340ed537c35dbb1db8652b2f0000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000846a6c671a6326c88f34e94da4f7c75bf06c3b1d80110d05af6898fa0b8a5d53477b53fabe8dd8576ca4137b8dadb384b2293a830bdb514ef8a4e4725500182d2789aab754c90b0121da5126ad57bda2bbca5d90eaef4f5691fe28acc1e6d1b100b55e060000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000030caa788af49c394912be6f445ccf38115b2a537316b81a6b96c229ed213f732ec1c15c4e726878432d03b347ecf9dcc9dd68905b4f308bd604840067b7c897def1df3e6da8311702fa4420ac12548da02c2560beb6ca51a0493ab4bed584f1eb6581b010000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000030caa788af49c394912be6f445ccf38115b2a537316b81a6b96c229ed213f7c21ba1c6f19ad07b0d8522e1b635bd8d410cae99f894b3045fc1b199dc87572f347dfe99c0fd08647c24cb71ac23b4ac28676bf7d2164c0e4d2a4b7bf3b969dfe730aaa1230000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "00000000000000000000000000000000000000000000000000000000000000846a6c671a6326c88f34e94da4f7c75bf06c3b1d80110d05af6898fa0b8a5d53477b53fabe8dd8576ca4137b8dadb384b2293a830bdb514ef8a4e4725500182d2789aab754c90b0121da5126ad57bda2bbca5d90eaef4f5691fe28acc1e6d1b100b55e060000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000020aa659d3427329eee9a212ebb9a25d5d72e822633f4996940f714101455b51eecf938cd2150f179253c87f5783bd2a07d6199cdef2a06b205e4ce5e00e27cca475c45afa7be606c2fe7bf880c0aafce16bb782619fa1fadfbac74ba5afd480f77a7c61b0000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000088775fda461df2bd01c1b5620dae5db6a67cdb8bec58e12aa2b355a10d97192587e434cda472cf75b4caa66f83b31bca8420dca6d52209f27c083451500a5bf36d667197337314c5c16696085a0d97be5a11d821e035330accdfd9fa61ba9b079f3966170000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "000000000000000000000000000000000000000000000000000000000000007c959398e59cd93770cb16b25b0838a40f93c4e27feef2fa50976705f475a22420988b00683d490b4295a44ffff216607ae55d7d355a1e3c2379ca6eb2e2fe23900413227e813c3e0fd2b47a0e98db687464cd409e3402f952f46abd28369495a84bc71e0000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000030caa788af49c394912be6f445ccf38115b2a537316b81a6b96c229ed213f732ec1c15c4e726878432d03b347ecf9dcc9dd68905b4f308bd604840067b7c897def1df3e6da8311702fa4420ac12548da02c2560beb6ca51a0493ab4bed584f1eb6581b010000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000088775fda461df2bd01c1b5620dae5db6a67cdb8bec58e12aa2b355a10d97192587e434cda472cf75b4caa66f83b31bca8420dca6d52209f27c083451500a5bf36d667197337314c5c16696085a0d97be5a11d821e035330accdfd9fa61ba9b079f3966170000000000000000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0000000000000000000000000000000000000000000000000000000000000068a23f05ee7a71f86dce88983ed024ef915348a9754eeb0211b1d497b69ce6f4697f38758ee8542d7f8e040ca19af05111a291dba9198750b29af27edc48c44b33c95aa34c39516ba5599542ca742218d6f27407090e12a11b3b172d345c14bc15f41b140000000000000000000000000000000000000000000000000000000000",
                        ),
                    ],
                ],
                u: [
                    [
                        (
                            0,
                            "d6b8a2ef0c680ce8e95952ff7e7899b215c112614287bbee1a0000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "c3c8f4086de0ffd775b849e9ecb0ebb2961ddcc3968e4da3010000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "d97769b28a3d6f2c7c8aeaa9e58401e66d8781de2bd77634060000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "ab6c830448a50267e06f9527043954b3516658dfb6f39417220000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            0,
                            "75731b431c7b32d08f3f80d58e7c8d32ff9affb2220b237c0c0000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "f29793375820fd22643908ae9f08684277a9dd528afb59f7020000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "1f9a69b277b4e7feded348e65ee2920ebb25c6bbe5727e55170000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "7e20faea72098ceb04919c59998c91a90591058f570822e1070000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            1,
                            "df549f360a54d968dd69269f4e8a56e95c5f55f46a0c88a73b0000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "a579df49778c0d4a00be2e8e0d6cb767a6a64233c44df712030000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "0b75da98909b68ae1ae79f20c5e531101586049b4322359c080000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "012e7e872fb3dc3a88fda237b6332b6c6a76a54f879b9874510000000000000000000000000000000000000000000000",
                        ),
                    ],
                    [
                        (
                            1,
                            "c158510a6e7cdcd3ef7e0b1ef5d0feeb755c6568f36b76e3210000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "167dade8ea70223fb172745feeea81f43b5bd5db82d44e66070000000000000000000000000000000000000000000000",
                        ),
                        (
                            0,
                            "9a0a2d396be1003a156714c55391c94fe24b3131b17b9e4b390000000000000000000000000000000000000000000000",
                        ),
                        (
                            1,
                            "f1d1ac1f40addbdc8da54453575aec904aa4f397eae8c6640e0000000000000000000000000000000000000000000000",
                        ),
                    ],
                ],
                bx: [1, 1, 1, 2],
                rad: "00000000000000000000000000000000000000000000000000000000000000b819e1f89d55484451dcf744127d993bdaf7b9597ff8ce9d4fca4230b882520e7603f04986bc4704a7cf9fcf80c7b1f83ddb040000000000000000000000000000",
            },
        ];

        let mut failures: Vec<String> = Vec::new();
        for (idx, rec) in records.iter().enumerate() {
            let g = dec_mat(&rec.g);
            let u_expect = dec_mat(&rec.u);
            let radius = dec(0, rec.rad).abs(); // C `radius`, positive
            let box_expect: [Int<256>; 4] = rec.bx.map(|v| *Uint::<256>::from_u64(v).as_int());

            // Production wrapper — the whole `lat_ball.c` bounding-parallelogram
            // recipe over the float `quat_lll_core`. None of these records is
            // trivial (each C box has a nonzero entry), so `Some` is expected.
            let (boxes, u_final) =
                bound_parallelogram::<256>(&g, &radius).expect("bounding parallelogram");

            let box_ok = boxes == box_expect;
            let u_ok = u_final == u_expect;
            let box_dbg = boxes.map(|b| b.abs().as_words()[0]);
            eprintln!("record {idx}: box_ok={box_ok} U_exact={u_ok} box={box_dbg:?}");
            if !box_ok || !u_ok {
                failures.push(format!("record {idx}: box_ok={box_ok} U_exact={u_ok}"));
            }
        }
        // The float `quat_lll_core` reproduces C's bounding parallelogram — box
        // AND transform `U` — byte-for-byte on every record (incl. record 1,
        // where exact integer LLL diverges: it finds a valid but different
        // reduced basis). This is what makes a byte-exact response sampler
        // possible; it is the regression guard for the whole float LLL path.
        assert!(
            failures.is_empty(),
            "float quat_lll_core must reproduce C's box AND U exactly: {failures:?}"
        );
    }

    /// `o0_lattice_to_standard_hnf` must reproduce C's response
    /// `lattice_hom_chall_to_com` (`sign.c`) byte-for-byte: the O_0→standard
    /// frame bridge (denom×2) plus `quat_lattice_hnf`. Data is C's live lvl1-KAT
    /// record-0 dump (`PQSQ_RESP_DUMP`): input is our O_0 hom-lattice, expected
    /// is C's standard HNF (`latden = 2`, column-major `latb`). Confirmed once
    /// end-to-end (our O_0 hom-lattice ≡ C's lattice via a unimodular map) — this
    /// pins the exact basis the byte-exact sampler will consume.
    #[cfg(feature = "kat")]
    #[test]
    fn o0_lattice_to_standard_hnf_matches_c_record0() {
        fn dec(neg: u8, h: &str) -> Int<80> {
            let mut bytes = [0u8; 80 * 8];
            for (k, byte) in bytes.iter_mut().enumerate().take(h.len() / 2) {
                *byte = u8::from_str_radix(&h[2 * k..2 * k + 2], 16).unwrap();
            }
            let u = Uint::<80>::from_le_slice(&bytes);
            let i = *u.as_int();
            if neg == 1 { i.wrapping_neg() } else { i }
        }
        fn dec_mat(m: &[[(u8, &str); 4]; 4]) -> [[Int<80>; 4]; 4] {
            let mut out = [[Int::<80>::from_i64(0); 4]; 4];
            for (orow, mrow) in out.iter_mut().zip(m.iter()) {
                for (oe, me) in orow.iter_mut().zip(mrow.iter()) {
                    *oe = dec(me.0, me.1);
                }
            }
            out
        }

        // Record-0 O_0 hom-lattice (rows = generators), homden = 1.
        let hom_o0 = dec_mat(&[
            [
                (
                    0,
                    "00000000000000000000000000000000000000000000000000000000000000db03f212fa6635d9ca12c945ebb828b5e80e563fb3281cd9441d55b55aaef189cdcf01",
                ),
                (0, "00"),
                (0, "00"),
                (0, "00"),
            ],
            [
                (0, "00"),
                (
                    0,
                    "00000000000000000000000000000000000000000000000000000000000000db03f212fa6635d9ca12c945ebb828b5e80e563fb3281cd9441d55b55aaef189cdcf01",
                ),
                (0, "00"),
                (0, "00"),
            ],
            [
                (
                    1,
                    "2a263bde17d6b5da493b136d73e748565bf2140910040648581978d3c091d98c731f3474ad737f5328b9bfa5c8c9dd7f5cbd10b37f6e12875bd2a8461a51559ed2",
                ),
                (
                    1,
                    "74abe0c2c7be09352ca2333ca8e3ef5dc9f41c56691be0b867be5fc4e5eaf1c4643651bada6ab18dd3e93543c9c15d9f81d6a88daa12ba617b65c5c2c072452ec3",
                ),
                (0, "01"),
                (0, "00"),
            ],
            [
                (
                    1,
                    "8d541f3d3841f6cad35dccc3571c10a2360be3a996e41f479841a03b1a150e91cc0a99c52cfbebd66fe18a7529bf1ebe3497a3b744354294cf6fec87e119343c61",
                ),
                (
                    1,
                    "2a263bde17d6b5da493b136d73e748565bf2140910040648581978d3c091d95cb866ba20884869d4b2791f3999a27e8043e22f0de534a5218a9a7ef9f5e49b98b8",
                ),
                (0, "00"),
                (0, "01"),
            ],
        ]);
        let homden = Uint::<80>::from_u64(1);

        // C standard HNF (latden = 2); columns = generators.
        let c_latb = dec_mat(&[
            [
                (
                    0,
                    "00000000000000000000000000000000000000000000000000000000000000b607e425f4cd6ab29525928bd671516ad11dac7e665138b2893aaa6ab55ce3139b9f03",
                ),
                (0, "00"),
                (
                    0,
                    "acb38943d053944a6c89d92519316e53491bd6eddff7f36f4fcd0f597edc4c9c20a5bd0b7383b3eed41f0c8be0bdaed164315d00525b8d7b830519282841695efa01",
                ),
                (
                    0,
                    "e756c1858f7d136a5844677850c7dfbb92e939acd236c071cf7cbf88cbd5e3936ecef3687474dae745cf75eb1ed32c55b47d37f7c7cd2d619bca91a599afab22dd02",
                ),
            ],
            [
                (0, "00"),
                (
                    0,
                    "00000000000000000000000000000000000000000000000000000000000000b607e425f4cd6ab29525928bd671516ad11dac7e665138b2893aaa6ab55ce3139b9f03",
                ),
                (
                    0,
                    "19a93e7a7082ec95a7bb9887af3820446d16c6532dc93f8e30834077342a1c2c3e77837f18954f7a7ebe1f50dfcdae921aff2c4bfc123ec643dfdf2fdbfd883e1902",
                ),
                (
                    0,
                    "acb38943d053944a6c89d92519316e53491bd6eddff7f36f4fcd0f597edc4cfc9616b1b2bdd9dfecbf9e4c643f0c6dd096e71e4c87ce674626756dc27019dc692e02",
                ),
            ],
            [(0, "00"), (0, "00"), (0, "01"), (0, "00")],
            [(0, "00"), (0, "00"), (0, "00"), (0, "01")],
        ]);
        let c_latden = Uint::<80>::from_u64(2);

        let (basis, denom) = o0_lattice_to_standard_hnf::<80>(&hom_o0, &homden);
        assert_eq!(denom, c_latden, "standard HNF denominator must match C");
        assert_eq!(
            basis, c_latb,
            "standard HNF basis must match C's latb byte-for-byte"
        );
    }
}
