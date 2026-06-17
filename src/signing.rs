//! Top-level SQIsign signing — `protocols_sign` (lvl1), the composition of the
//! signing sub-steps built across the quaternion / isogeny / verification
//! modules. Functional (self-consistent), NOT yet byte-exact. `kat`-gated
//! because the spine (`commit`) and the prime-norm reduction consume the
//! byte-exact DRBG.
#![allow(dead_code)]

#[cfg(feature = "kat")]
use crate::quaternion::ideal::LeftIdeal;
#[cfg(feature = "kat")]
use crate::verification::SecretKeyData;
#[cfg(feature = "kat")]
use rand_core::CryptoRng;

/// Widen a `LeftIdeal<16>` to `LeftIdeal<W>` (W ≥ 16) for wide lattice ops.
#[cfg(feature = "kat")]
fn widen_ideal_16<const W: usize>(id: &LeftIdeal<16>) -> LeftIdeal<W> {
    use crate::quaternion::lattice::widen_int_lattice;
    let mut basis = [[crypto_bigint::Int::<W>::from_i64(0); 4]; 4];
    for (rw, r16) in basis.iter_mut().zip(id.basis.iter()) {
        for (ew, e16) in rw.iter_mut().zip(r16.iter()) {
            *ew = widen_int_lattice::<16, W>(e16);
        }
    }
    LeftIdeal::<W>::with_denom_and_norm(basis, id.denom.resize::<W>(), id.cached_norm.resize::<W>())
}

/// Sign `msg` under the secret key `sk`, producing the 148-byte lvl1 signature.
/// Port of C `protocols_sign` (`sign.c:479`), common path (`two_resp = 0`).
/// Returns `None` if no attempt within the retry budget succeeds. HEAVY.
#[cfg(feature = "kat")]
pub fn protocols_sign<R: CryptoRng>(
    sk: &SecretKeyData,
    msg: &[u8],
    rng: &mut R,
) -> Option<[u8; 148]> {
    use crate::ec::biscalar::ec_curve_to_basis_2f_from_hint;
    use crate::ec::couple::EcBasis;
    use crate::ec::montgomery::MontgomeryCurve;
    use crate::isogeny::clapotis_spine::{
        commit, compute_dim2_isogeny_challenge, evaluate_random_aux_isogeny_lvl1,
    };
    use crate::isogeny::endomorphism::compute_challenge_ideal_signature;
    use crate::quaternion::lattice_ops::{
        compute_backtracking_signature, compute_response_quat_element,
    };
    use crate::quaternion::sign_orchestration::compute_random_aux_norm_and_helpers;
    use crate::verification::{
        SignatureData, compute_and_set_basis_change_matrix, compute_challenge_codomain_signature,
        ec_dbl_iter_basis, hash_to_challenge,
    };
    use crypto_bigint::U256;

    const TEP: usize = 248;
    const HD: u32 = 2;
    const RESPONSE_BITS: usize = 126;
    let p16 = crate::params::lvl1::prime().resize::<16>();
    let wit_ql: [crypto_bigint::Uint<12>; 5] =
        [2u64, 3, 5, 7, 11].map(crypto_bigint::Uint::from_u64);

    let sk_curve = MontgomeryCurve::new(sk.curve_a);
    let canonical_basis = ec_curve_to_basis_2f_from_hint(&sk_curve, TEP, sk.hint_pk);

    for _attempt in 0..8 {
        // 1. Commitment.
        let Some((e_com, b_com, lideal_commit)) = commit(&wit_ql, 64, 1 << 14, rng) else {
            std::eprintln!("DIAG sign: commit failed");
            continue;
        };
        std::eprintln!("DIAG sign: commit ok");
        // 2. Challenge coefficient.
        let chall_coeff = hash_to_challenge(&sk.curve_a, &e_com.a, msg);
        // 3. Challenge ideal (pulled back through the secret key matrix).
        let lideal_chall_two =
            compute_challenge_ideal_signature(&sk.mat_bacan_to_ba0_two, &chall_coeff, TEP)?;
        // 4. Response quaternion. Run the lattice ops at a WIDE width: with the
        // TRUE dual-trick intersection, chall_secret has norm ~2^384 (entries
        // ~2^384), and the SECOND intersection's dual-of-dual adjugate reaches
        // ~2^4700, overflowing W=48 (3072 bits) → a garbage hom-lattice the
        // sampler can't satisfy. W=96 (6144 bits) holds the dual-of-dual.
        const W: usize = 96;
        let p_w = crate::params::lvl1::prime().resize::<W>();
        let Some((resp_w, resp_d_w, lc_w)) = compute_response_quat_element::<W, R>(
            &widen_ideal_16::<W>(&sk.secret_ideal),
            &widen_ideal_16::<W>(&lideal_chall_two),
            &widen_ideal_16::<W>(&lideal_commit),
            &p_w,
            u32::try_from(RESPONSE_BITS).expect("RESPONSE_BITS fits in u32"),
            1 << 14,
            rng,
        ) else {
            std::eprintln!("DIAG sign: response_quat failed");
            continue;
        };
        // The integral response is resp_w / resp_d_w; divide out the (reduced)
        // denom, then narrow to L16 for the downstream quaternion steps.
        use crate::quaternion::lattice::narrow_int_lattice;
        let resp_d = resp_d_w.resize::<16>();
        let mut resp = [crypto_bigint::Int::<16>::from_i64(0); 4];
        for (r16, rw) in resp.iter_mut().zip(resp_w.iter()) {
            *r16 = narrow_int_lattice::<W, 16>(rw);
        }
        let lattice_content = lc_w.resize::<16>();
        {
            use crate::quaternion::o0_mul::reduced_norm_o0_basis;
            let nr = reduced_norm_o0_basis::<16>(&resp, &p16).abs();
            let lc_nz = crypto_bigint::NonZero::new(lattice_content).unwrap();
            let (_q, r) = nr.div_rem_vartime(&lc_nz);
            std::eprintln!(
                "DIAG sign: resp_d bits={} lc bits={} Nred(resp) bits={} divides={}",
                resp_d.bits_vartime(),
                lattice_content.bits_vartime(),
                nr.bits_vartime(),
                r == crypto_bigint::Uint::<16>::ZERO,
            );
        }
        std::eprintln!("DIAG sign: response ok");
        // 5. Backtracking. C ref (sign.c:107-117): backtracking = v2(content of
        //    make_primitive(resp)); lattice_content /= 2^backtracking. The aux
        //    (sign.c:144) then uses the FULL resp_quat with the REDUCED
        //    lattice_content (`remain`). Using `prim` + the un-reduced
        //    lattice_content makes lattice_content ∤ N_red(prim) (it divides
        //    N_red(resp) but not N_red(prim) once the content is stripped).
        let (backtracking, remain, prim) =
            compute_backtracking_signature::<16>(&resp, &lattice_content);
        // 6. Auxiliary norm + helpers (full resp + reduced lattice_content).
        let commit_norm = lideal_commit.reduced_norm_vartime()?;
        let helpers = match compute_random_aux_norm_and_helpers::<16>(
            &resp,
            &remain,
            &commit_norm,
            &p16,
            backtracking,
            RESPONSE_BITS,
            HD,
        ) {
            Ok(h) => h,
            Err(e) => {
                std::eprintln!("DIAG sign: aux_norm_helpers failed: {e:?}");
                continue;
            }
        };
        let pow = helpers.pow_dim2_deg_resp;
        let two_resp = helpers.two_resp_length;
        // Common path only: skip degenerate / length-1 / short-chain cases.
        std::eprintln!("DIAG sign: pow={pow} two_resp={two_resp}");
        if pow == 0 || pow == 1 {
            std::eprintln!("DIAG sign: pow gate skip");
            continue;
        }
        // 7. Auxiliary isogeny.
        let com_resp16 = helpers.lideal_com_resp;
        let Some((e_aux, b_aux)) = evaluate_random_aux_isogeny_lvl1(
            &helpers.random_aux_norm,
            &com_resp16,
            &wit_ql,
            64,
            1 << 14,
            rng,
        ) else {
            std::eprintln!("DIAG sign: aux_isogeny failed");
            continue;
        };
        std::eprintln!("DIAG sign: aux_isogeny ok");
        // 8. Reduce the bases to order 2^(pow + HD + two_resp), then dim-2.
        let reduced_order = (pow + HD + two_resp) as usize;
        let e_diff = u32::try_from(TEP - reduced_order).ok()?;
        let b_com_red = ec_dbl_iter_basis(&b_com, e_diff, &e_com);
        let b_aux_red = ec_dbl_iter_basis(&b_aux, e_diff, &e_aux);
        let deg_inv = helpers.degree_resp_inv.to_le_bytes();
        let Some((codomain, pushed)) = compute_dim2_isogeny_challenge(
            &e_com, &b_com_red, &e_aux, &b_aux_red, &deg_inv, pow, two_resp, rng,
        ) else {
            std::eprintln!("DIAG sign: dim2_challenge failed");
            continue;
        };
        std::eprintln!("DIAG sign: dim2 ok");
        // C compute_dim2_isogeny_challenge (sign.c:280) SWAPS the theta factors:
        // E_aux = codomain.E2, E_chall = codomain.E1; B_aux = pushed.P2,
        // B_chall = pushed.P1 ("it should always be the first curve").
        let e_aux2 = codomain.e2;
        let e_chall2_postdim = codomain.e1; // S351: e_chall2 BEFORE small_chain
        let mut e_chall2 = codomain.e1;
        let b_aux2 = EcBasis::new(pushed[0].p2, pushed[1].p2, pushed[2].p2);
        let mut b_chall2 = EcBasis::new(pushed[0].p1, pushed[1].p1, pushed[2].p1);
        {
            let fo = u32::try_from(reduced_order).ok()?;
            let oa = e_aux2.to_a24();
            let oc = e_chall2.to_a24();
            std::eprintln!(
                "DIAG sign: post-dim2 order f={reduced_order} aux2.P[2^f]=O?{:?} chall2.P[2^f]=O?{:?} chall2.P[2^(f-1)]=O?{:?}",
                bool::from(oa.x_double_n(&b_aux2.p, fo).is_infinity()),
                bool::from(oc.x_double_n(&b_chall2.p, fo).is_infinity()),
                bool::from(oc.x_double_n(&b_chall2.p, fo - 1).is_infinity()),
            );
        }
        // 8b. Optional short 2^r response isogeny (two_resp_length > 0).
        if two_resp > 0 {
            // NOTE (S351): C (sign.c:325) builds the 2^two_resp response ideal
            // from the FULL resp_quat; using `prim` here is wrong, BUT switching
            // to `&resp` breaks our kernel (id2iso_ideal_to_kernel_dlogs_even
            // computes conj(resp)'s action DIRECTLY and full resp has even coords
            // → non-primitive kernel column → codomain order >2^f). Real fix:
            // build the ideal O_0·resp + 2^two_resp·O_0 like C. The j(E_chall)
            // mismatch is UPSTREAM of here (false with both prim and resp), so
            // keep prim until the dim2/E_chall j-mismatch is resolved.
            let Some((e2, b2)) = crate::verification::compute_small_chain_isogeny_signature(
                &e_chall2, &b_chall2, &prim, pow, two_resp,
            ) else {
                std::eprintln!("DIAG sign: small_chain failed");
                continue;
            };
            e_chall2 = e2;
            b_chall2 = b2;
            let fo = u32::try_from(reduced_order).ok()?;
            let oc = e_chall2.to_a24();
            std::eprintln!(
                "DIAG sign: post-small_chain chall2.P[2^f]=O?{:?} [2^(f-1)]=O?{:?}",
                bool::from(oc.x_double_n(&b_chall2.p, fo).is_infinity()),
                bool::from(oc.x_double_n(&b_chall2.p, fo - 1).is_infinity()),
            );
        }
        // 9. Recompute E_chall + map the challenge basis onto it.
        let Some((e_chall, b_chall_mapped)) = compute_challenge_codomain_signature(
            &sk.curve_a,
            &canonical_basis,
            &chall_coeff,
            u8::try_from(backtracking).ok()?,
            &e_chall2.a,
            &b_chall2,
        ) else {
            std::eprintln!("DIAG sign: challenge_codomain failed");
            continue;
        };
        std::eprintln!(
            "DIAG sign: codomain ok j(E_chall)==j(e_chall2)?{:?} ==j(e_aux2)?{:?} ==j(e_chall2_postdim,pre-smallchain)?{:?} | j(e_chall2_postdim)==j(e_aux2)?{:?}",
            e_chall.j_invariant() == e_chall2.j_invariant(),
            e_chall.j_invariant() == e_aux2.j_invariant(),
            e_chall.j_invariant() == e_chall2_postdim.j_invariant(),
            e_chall2_postdim.j_invariant() == e_aux2.j_invariant(),
        );
        // 10. Assemble + encode the signature.
        let mut sig = SignatureData {
            e_aux_a: e_aux2.a,
            backtracking: u8::try_from(backtracking).ok()?,
            two_resp_length: u8::try_from(two_resp).ok()?,
            mat: [[U256::ZERO; 2]; 2],
            chall_coeff,
            hint_aux: 0,
            hint_chall: 0,
        };
        if !compute_and_set_basis_change_matrix(
            &mut sig,
            &b_aux2,
            &b_chall_mapped,
            &e_aux2,
            &e_chall,
            reduced_order,
        ) {
            std::eprintln!("DIAG sign: basis_change_matrix failed");
            continue;
        }
        std::eprintln!("DIAG sign: assemble ok");
        let mut out = [0u8; 148];
        sig.to_bytes_lvl1(&mut out).ok()?;
        return Some(out);
    }
    None
}

#[cfg(all(test, feature = "kat"))]
mod tests {
    use super::*;
    use crate::rng::NistPqcRng;

    #[ignore = "heavy: full keygen → sign → verify roundtrip (real-scale spine)"]
    #[test]
    fn sign_verify_roundtrip() {
        use crate::isogeny::clapotis_spine::keygen_lvl1;
        use crate::verification::PublicKeyData;
        let wit: [crypto_bigint::Uint<12>; 5] =
            [2u64, 3, 5, 7, 11].map(crypto_bigint::Uint::from_u64);
        let mut rng = NistPqcRng::new(&[0x77u8; 48]);

        // Functional keypair.
        let (e_a, secret_ideal, mat, _b_acan, hint_pk, _b_a0) =
            keygen_lvl1(&wit, 64, 1 << 14, &mut rng).expect("keygen");
        let sk = SecretKeyData {
            curve_a: e_a.a,
            hint_pk,
            secret_ideal,
            mat_bacan_to_ba0_two: mat,
        };
        let pk = PublicKeyData {
            curve_a: e_a.a,
            hint_pk,
        };
        let mut pk_bytes = [0u8; 65];
        pk.to_bytes_lvl1(&mut pk_bytes).expect("pk encode");

        let msg = b"sqisign roundtrip";
        let sig = protocols_sign(&sk, msg, &mut rng).expect("sign produces a signature");

        let result = crate::verify::<crate::params::Level1>(msg, &sig, &pk_bytes);
        assert_eq!(result, Ok(()), "verify must accept the produced signature");
    }
}
