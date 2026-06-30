//! Top-level SQIsign signing — `protocols_sign` (lvl1), the composition of the
//! signing sub-steps built across the quaternion / isogeny / verification
//! modules. Functional (self-consistent), NOT yet byte-exact. `kat`-gated
//! because the spine (`commit`) and the prime-norm reduction consume the
//! byte-exact DRBG.

#[cfg(feature = "sign")]
use crate::quaternion::ideal::LeftIdeal;
#[cfg(feature = "sign")]
use crate::verification::SecretKeyData;
#[cfg(feature = "sign")]
use rand_core::CryptoRng;

/// Widen a `LeftIdeal<16>` to `LeftIdeal<W>` (W ≥ 16) for wide lattice ops.
#[cfg(feature = "sign")]
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

/// Sign `msg` under the secret key `sk`, producing the `P::SIG_BYTES`-byte
/// signature. Port of C `protocols_sign` (`sign.c:479`), common path
/// (`two_resp = 0`). Per-level dispatch wrapper: it pins the quaternion
/// precision `QL` and the response lattice width `W` per level, then runs the
/// generic [`protocols_sign_impl`]. Returns `None` if no attempt within the
/// retry budget succeeds. HEAVY.
#[cfg(feature = "sign")]
pub fn protocols_sign<P: crate::isogeny::fixed_degree::FixedDegreeLevel, R: CryptoRng>(
    sk: &SecretKeyData<P::Field>,
    msg: &[u8],
    rng: &mut R,
) -> Option<Vec<u8>> {
    match P::LEVEL {
        // lvl1: QL=12, W=80 (5120 bits — smallest tested width with margin).
        1 => protocols_sign_impl::<P, 12, 80, R>(sk, msg, rng),
        // lvl3: QL=18, W=160 (10240 bits) — the 2^768-scale response lattice's
        // intersection adjugate is proportionally wider; generous headroom,
        // narrowed later if the roundtrip leaves margin.
        3 => protocols_sign_impl::<P, 18, 160, R>(sk, msg, rng),
        _ => None,
    }
}

#[cfg(feature = "sign")]
fn protocols_sign_impl<
    P: crate::isogeny::fixed_degree::FixedDegreeLevel,
    const QL: usize,
    const W: usize,
    R: CryptoRng,
>(
    sk: &SecretKeyData<P::Field>,
    msg: &[u8],
    rng: &mut R,
) -> Option<Vec<u8>> {
    use crate::ec::biscalar::ec_curve_to_basis_2f_from_hint;
    use crate::ec::couple::EcBasis;
    use crate::ec::montgomery::MontgomeryCurve;
    use crate::isogeny::clapotis_spine::{
        commit, compute_dim2_isogeny_challenge, evaluate_random_aux_isogeny,
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

    let tep: usize = P::F;
    const HD: u32 = 2;
    let response_bits: usize = P::RESPONSE_BITS;
    let p16 = P::prime::<16>();
    let wit_ql: [crypto_bigint::Uint<QL>; 5] =
        [2u64, 3, 5, 7, 11].map(crypto_bigint::Uint::from_u64);

    let sk_curve = MontgomeryCurve::new(sk.curve_a);
    let canonical_basis = ec_curve_to_basis_2f_from_hint::<P>(&sk_curve, tep, sk.hint_pk);

    // Signing is rejection-sampled (commitment, response, aux-isogeny, and
    // basis-change steps each reject a fraction of attempts), so the budget needs
    // headroom: 8 occasionally exhausts for statistically unlucky messages.
    for _attempt in 0..16 {
        // 1. Commitment.
        let Some((e_com, b_com, lideal_commit)) = commit::<P, QL, _>(&wit_ql, 64, 1 << 14, rng)
        else {
            #[cfg(feature = "std")]
            eprintln!("[sign L{}] attempt {_attempt}: commit None", P::LEVEL);
            continue;
        };
        // 2. Challenge coefficient.
        let chall_coeff = hash_to_challenge::<P>(&sk.curve_a, &e_com.a, msg);
        // 3. Challenge ideal (pulled back through the secret key matrix). A
        //    failed pullback (no valid challenge ideal for this commitment)
        //    retries with a fresh commitment rather than aborting the signer.
        let Some(lideal_chall_two) =
            compute_challenge_ideal_signature::<P>(&sk.mat_bacan_to_ba0_two, &chall_coeff, tep)
        else {
            #[cfg(feature = "std")]
            eprintln!(
                "[sign L{}] attempt {_attempt}: challenge_ideal None",
                P::LEVEL
            );
            continue;
        };
        // 4. Response quaternion. The dual-of-dual intersection adjugate is wide
        // (lvl1 ~2^4700 worst case), so the lattice ops run at a widened limb
        // count `W` (pinned per level by the dispatch wrapper).
        // `lattice_intersect` reduces each intermediate dual to lowest terms
        // (lattice-preserving) to avoid further growth. The roundtrip is the
        // end-to-end correctness guard.
        let p_w = P::prime::<W>();
        let Some((resp_w, _resp_d_w, lc_w)) = compute_response_quat_element::<W, R>(
            &widen_ideal_16::<W>(&sk.secret_ideal),
            &widen_ideal_16::<W>(&lideal_chall_two),
            &widen_ideal_16::<W>(&lideal_commit),
            &p_w,
            u32::try_from(response_bits).expect("RESPONSE_BITS fits in u32"),
            1 << 14,
            rng,
        ) else {
            #[cfg(feature = "std")]
            eprintln!("[sign L{}] attempt {_attempt}: response None", P::LEVEL);
            continue;
        };
        // The integral response is resp_w / resp_d_w; divide out the (reduced)
        // denom, then narrow to L16 for the downstream quaternion steps.
        use crate::quaternion::lattice::narrow_int_lattice;
        let mut resp = [crypto_bigint::Int::<16>::from_i64(0); 4];
        for (r16, rw) in resp.iter_mut().zip(resp_w.iter()) {
            *r16 = narrow_int_lattice::<W, 16>(rw);
        }
        let lattice_content = lc_w.resize::<16>();
        // 5. Backtracking. C ref (sign.c:107-117): backtracking = v2(content of
        //    make_primitive(resp)); lattice_content /= 2^backtracking. The aux
        //    (sign.c:144) then uses the FULL resp_quat with the REDUCED
        //    lattice_content (`remain`). Using `prim` + the un-reduced
        //    lattice_content makes lattice_content ∤ N_red(prim) (it divides
        //    N_red(resp) but not N_red(prim) once the content is stripped).
        let (backtracking, remain, prim) =
            compute_backtracking_signature::<16>(&resp, &lattice_content);
        // 6. Auxiliary norm + helpers (full resp + reduced lattice_content).
        let Some(commit_norm) = lideal_commit.reduced_norm_vartime() else {
            #[cfg(feature = "std")]
            eprintln!("[sign L{}] attempt {_attempt}: commit_norm None", P::LEVEL);
            continue;
        };
        let helpers = match compute_random_aux_norm_and_helpers::<16>(
            &resp,
            &remain,
            &commit_norm,
            &p16,
            backtracking,
            response_bits,
            HD,
        ) {
            Ok(h) => h,
            Err(_e) => {
                #[cfg(feature = "std")]
                eprintln!(
                    "[sign L{}] attempt {_attempt}: aux_norm_helpers Err {_e:?}",
                    P::LEVEL
                );
                continue;
            }
        };
        let pow = helpers.pow_dim2_deg_resp;
        let two_resp = helpers.two_resp_length;
        // Common path only: skip degenerate / length-1 / short-chain cases.
        if pow == 0 || pow == 1 {
            #[cfg(feature = "std")]
            eprintln!("[sign L{}] attempt {_attempt}: pow={pow} (skip)", P::LEVEL);
            continue;
        }
        // INTERIM MITIGATION (two_resp>0 bug, still active): the short 2^r
        // response branch is partially fixed — `compute_small_chain_isogeny_signature`
        // now builds the response ideal from the primitive response and recovers a
        // canonical generator via `quat_lideal_generator_o0` (matching the C
        // reference, verified: the kernel coords are now primitive). What remains
        // is the sign↔verify kernel reconciliation for MIXED kernels: sign forms a
        // general `vec2[0]·P + vec2[1]·Q` kernel while verify selects a single
        // basis point by matrix parity, so the basis-change matrix
        // (`compute_and_set_basis_change_matrix`) must rotate the kernel onto a
        // basis vector. Until that lands, reject `two_resp>0` attempts so every
        // emitted signature takes the verified-correct common path (sound
        // rejection sampling). Remove this skip once the matrix reconciliation is
        // complete.
        if two_resp > 0 {
            #[cfg(feature = "std")]
            eprintln!(
                "[sign L{}] attempt {_attempt}: two_resp={two_resp} (skip — interim)",
                P::LEVEL
            );
            continue;
        }
        // 7. Auxiliary isogeny.
        let com_resp16 = helpers.lideal_com_resp;
        let Some((e_aux, b_aux)) = evaluate_random_aux_isogeny::<P, QL, _>(
            &helpers.random_aux_norm,
            &com_resp16,
            &wit_ql,
            64,
            1 << 14,
            rng,
        ) else {
            #[cfg(feature = "std")]
            eprintln!("[sign L{}] attempt {_attempt}: aux_isogeny None", P::LEVEL);
            continue;
        };
        // 8. Reduce the bases to order 2^(pow + HD + two_resp), then dim-2.
        let reduced_order = (pow + HD + two_resp) as usize;
        let e_diff = u32::try_from(tep - reduced_order).ok()?;
        let b_com_red = ec_dbl_iter_basis(&b_com, e_diff, &e_com);
        let b_aux_red = ec_dbl_iter_basis(&b_aux, e_diff, &e_aux);
        let deg_inv = helpers.degree_resp_inv.to_le_bytes();
        let Some((codomain, pushed)) = compute_dim2_isogeny_challenge(
            &e_com, &b_com_red, &e_aux, &b_aux_red, &deg_inv, pow, two_resp, rng,
        ) else {
            #[cfg(feature = "std")]
            eprintln!("[sign L{}] attempt {_attempt}: dim2_isogeny None", P::LEVEL);
            continue;
        };
        // C compute_dim2_isogeny_challenge (sign.c:280) SWAPS the theta factors:
        // E_aux = codomain.E2, E_chall = codomain.E1; B_aux = pushed.P2,
        // B_chall = pushed.P1 ("it should always be the first curve").
        let e_aux2 = codomain.e2;
        let mut e_chall2 = codomain.e1;
        let b_aux2 = EcBasis::new(pushed[0].p2, pushed[1].p2, pushed[2].p2);
        let mut b_chall2 = EcBasis::new(pushed[0].p1, pushed[1].p1, pushed[2].p1);
        // 8b. Optional short 2^r response isogeny (two_resp_length > 0). Pass the
        // PRIMITIVE response `prim` (C `sign.c` applies `quat_alg_make_primitive`
        // before this step): `compute_small_chain` builds the ideal
        // `O_0·prim + 2^two_resp·O_0` and recovers its canonical generator, which
        // yields the kernel the verifier reconstructs from the matrix parity.
        if two_resp > 0 {
            let Some((e2, b2)) = crate::verification::compute_small_chain_isogeny_signature::<P>(
                &e_chall2, &b_chall2, &prim, pow, two_resp,
            ) else {
                #[cfg(feature = "std")]
                eprintln!("[sign L{}] attempt {_attempt}: small_chain None", P::LEVEL);
                continue;
            };
            e_chall2 = e2;
            b_chall2 = b2;
        }
        // 9. Recompute E_chall + map the challenge basis onto it.
        let Some((e_chall, b_chall_mapped)) = compute_challenge_codomain_signature::<P>(
            &sk.curve_a,
            &canonical_basis,
            &chall_coeff,
            u8::try_from(backtracking).ok()?,
            &e_chall2.a,
            &b_chall2,
        ) else {
            #[cfg(feature = "std")]
            eprintln!(
                "[sign L{}] attempt {_attempt}: challenge_codomain None",
                P::LEVEL
            );
            continue;
        };
        // 10. Assemble + encode the signature.
        let mut sig = SignatureData {
            e_aux_a: e_aux2.a,
            backtracking: u8::try_from(backtracking).ok()?,
            two_resp_length: u8::try_from(two_resp).ok()?,
            mat: [[crypto_bigint::Uint::<8>::ZERO; 2]; 2],
            chall_coeff,
            hint_aux: 0,
            hint_chall: 0,
        };
        if !compute_and_set_basis_change_matrix::<P>(
            &mut sig,
            &b_aux2,
            &b_chall_mapped,
            &e_aux2,
            &e_chall,
            reduced_order,
        ) {
            #[cfg(feature = "std")]
            eprintln!(
                "[sign L{}] attempt {_attempt}: basis_change_matrix fail",
                P::LEVEL
            );
            continue;
        }
        #[cfg(feature = "std")]
        eprintln!(
            "[sign L{}] attempt {_attempt}: SUCCESS (pow={pow} two_resp={two_resp})",
            P::LEVEL
        );
        let mut out = alloc::vec![0u8; P::SIG_BYTES];
        sig.to_bytes::<P>(&mut out).ok()?;
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
        let sig = protocols_sign::<crate::params::Level1, _>(&sk, msg, &mut rng)
            .expect("sign produces a signature");

        let result = crate::verify::<crate::params::Level1>(msg, &sig, &pk_bytes);
        assert_eq!(result, Ok(()), "verify must accept the produced signature");
    }
}
