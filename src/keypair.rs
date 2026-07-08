// SPDX-License-Identifier: MIT OR Apache-2.0
//! [`KeyPair`] — typed SQIsign keypair generation and access.

#[cfg(feature = "kgen")]
use crate::isogeny::clapotis_spine::keygen;
use crate::params::{Params, lvl1::Level1, lvl3::Level3};
use crate::signing_key::SigningKey;
use crate::verifying_key::VerifyingKey;
use crate::{Error, Result};
#[cfg(feature = "kgen")]
use alloc::vec::Vec;

/// A SQIsign keypair (signing key + verifying key), parameterized by security level.
///
/// Construct via [`generate`](Self::generate) or
/// [`from_signing_key_bytes`](Self::from_signing_key_bytes).
pub struct KeyPair<P: Params> {
    signing_key: SigningKey<P>,
    verifying_key: VerifyingKey<P>,
}

impl<P: Params> core::fmt::Debug for KeyPair<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("KeyPair")
            .field("signing_key", &self.signing_key)
            .field("verifying_key", &self.verifying_key)
            .finish()
    }
}

impl<P: Params> KeyPair<P> {
    /// Generate a fresh keypair using the provided randomness source. Level-1
    /// and level-3 are supported; the level-specific keygen and public-key
    /// serialization are dispatched through [`KeyLevel`]. Generated keys hold a
    /// live secret — [`SigningKey::to_bytes`] is only available for keys loaded
    /// via [`from_signing_key_bytes`](Self::from_signing_key_bytes).
    #[cfg(feature = "kgen")]
    pub fn generate<R: rand_core::CryptoRng>(rng: &mut R) -> Result<Self>
    where
        P: KeyLevel,
    {
        let (sk_data, pk_bytes) = P::generate_keypair(rng)?;
        Ok(Self {
            signing_key: SigningKey::from_secret_data(sk_data),
            verifying_key: VerifyingKey::from_bytes_unchecked(&pk_bytes),
        })
    }

    /// Reconstruct a keypair from the secret-key wire-format bytes.
    ///
    /// The public key is embedded in the first `P::PK_BYTES` of the
    /// secret-key encoding and extracted automatically.
    pub fn from_signing_key_bytes(sk_bytes: &[u8]) -> Result<Self>
    where
        P: KeyLevel,
    {
        if sk_bytes.len() < P::SK_BYTES {
            return Err(Error::BufferTooSmall {
                required: P::SK_BYTES,
                provided: sk_bytes.len(),
            });
        }
        let signing_key = SigningKey::<P>::from_bytes(sk_bytes)?;
        // The SQIsign SK wire format starts with the PK bytes.
        let verifying_key = VerifyingKey::from_bytes_unchecked(&sk_bytes[..P::PK_BYTES]);
        Ok(Self {
            signing_key,
            verifying_key,
        })
    }

    /// Borrow the signing (secret) key.
    pub fn signing_key(&self) -> &SigningKey<P> {
        &self.signing_key
    }

    /// Borrow the verifying (public) key.
    pub fn verifying_key(&self) -> &VerifyingKey<P> {
        &self.verifying_key
    }

    /// Decompose the keypair into its constituent keys.
    pub fn into_parts(self) -> (SigningKey<P>, VerifyingKey<P>) {
        (self.signing_key, self.verifying_key)
    }
}

/// Per-level key operations that depend on the field type `P::Field`, dispatched
/// from the otherwise field-generic [`KeyPair`] / [`SigningKey`]. Level-1 uses the
/// byte-exact lvl1 wire paths; level-3 uses the generalized spine + generic
/// `fp2_encode`. This is the seam that keeps the key types field-agnostic.
pub trait KeyLevel: Params {
    /// Decode secret-key data from this level's wire bytes.
    fn sk_from_bytes(bytes: &[u8]) -> Result<crate::verification::SecretKeyData<Self::Field>>;

    /// Run keygen → (live secret-key data, encoded public-key bytes).
    #[cfg(feature = "kgen")]
    fn generate_keypair<R: rand_core::CryptoRng>(
        rng: &mut R,
    ) -> Result<(crate::verification::SecretKeyData<Self::Field>, Vec<u8>)>;

    /// Produce signature bytes; `None` if signing is unsupported at this level.
    #[cfg(feature = "sign")]
    fn protocols_sign<R: rand_core::CryptoRng>(
        sk: &crate::verification::SecretKeyData<Self::Field>,
        msg: &[u8],
        rng: &mut R,
    ) -> Option<Vec<u8>>;
}

impl KeyLevel for Level1 {
    fn sk_from_bytes(
        bytes: &[u8],
    ) -> Result<crate::verification::SecretKeyData<crate::params::lvl1::Fp1Element>> {
        crate::verification::SecretKeyData::from_bytes_lvl1(bytes)
    }

    #[cfg(feature = "kgen")]
    fn generate_keypair<R: rand_core::CryptoRng>(
        rng: &mut R,
    ) -> Result<(
        crate::verification::SecretKeyData<crate::params::lvl1::Fp1Element>,
        Vec<u8>,
    )> {
        use crate::isogeny::clapotis_spine::keygen_lvl1;
        use crate::verification::{PublicKeyData, SecretKeyData};

        let witnesses: [crypto_bigint::Uint<12>; 5] =
            [2u64, 3, 5, 7, 11].map(crypto_bigint::Uint::from_u64);
        let (e_a, secret_ideal, mat, _b_acan, hint_pk, _b_a0) =
            keygen_lvl1(&witnesses, 64, 1 << 14, rng)
                .ok_or(Error::Internal("keygen_lvl1 exhausted retry budget"))?;
        let sk_data = SecretKeyData {
            curve_a: e_a.a,
            hint_pk,
            secret_ideal,
            mat_bacan_to_ba0_two: mat,
        };
        let mut pk_bytes = alloc::vec![0u8; Self::PK_BYTES];
        PublicKeyData {
            curve_a: e_a.a,
            hint_pk,
        }
        .to_bytes_lvl1(&mut pk_bytes)?;
        Ok((sk_data, pk_bytes))
    }

    #[cfg(feature = "sign")]
    fn protocols_sign<R: rand_core::CryptoRng>(
        sk: &crate::verification::SecretKeyData<crate::params::lvl1::Fp1Element>,
        msg: &[u8],
        rng: &mut R,
    ) -> Option<Vec<u8>> {
        crate::signing::protocols_sign::<Level1, R>(sk, msg, rng)
    }
}

impl KeyLevel for Level3 {
    fn sk_from_bytes(
        _bytes: &[u8],
    ) -> Result<crate::verification::SecretKeyData<crate::params::lvl3::Fp3Element>> {
        Err(Error::Unimplemented(
            "signing key lvl3: byte decode not implemented",
        ))
    }

    #[cfg(feature = "kgen")]
    fn generate_keypair<R: rand_core::CryptoRng>(
        rng: &mut R,
    ) -> Result<(
        crate::verification::SecretKeyData<crate::params::lvl3::Fp3Element>,
        Vec<u8>,
    )> {
        use crate::gf::fp2::Fp2;
        use crate::verification::SecretKeyData;

        let witnesses: [crypto_bigint::Uint<18>; 5] =
            [2u64, 3, 5, 7, 11].map(crypto_bigint::Uint::from_u64);
        let (e_a, secret_ideal, mat, _b_acan, hint_pk, _b_a0) =
            keygen::<Level3, 18, R>(&witnesses, 64, 1 << 14, rng)
                .ok_or(Error::Internal("keygen_lvl3 exhausted retry budget"))?;
        let sk_data = SecretKeyData {
            curve_a: e_a.a,
            hint_pk,
            secret_ideal,
            mat_bacan_to_ba0_two: mat,
        };
        // PK wire format = fp2_encode(A) || hint_pk, generic over the field width.
        let fp2_bytes = Level3::FP2_BYTES;
        let mut pk_bytes = alloc::vec![0u8; Level3::PK_BYTES];
        Fp2::<crate::params::lvl3::Fp3Element>::to_bytes_le(&e_a.a, &mut pk_bytes[..fp2_bytes]);
        pk_bytes[fp2_bytes] = hint_pk;
        Ok((sk_data, pk_bytes))
    }

    #[cfg(feature = "sign")]
    fn protocols_sign<R: rand_core::CryptoRng>(
        sk: &crate::verification::SecretKeyData<crate::params::lvl3::Fp3Element>,
        msg: &[u8],
        rng: &mut R,
    ) -> Option<Vec<u8>> {
        crate::signing::protocols_sign::<Level3, R>(sk, msg, rng)
    }
}

#[cfg(all(test, feature = "kat"))]
mod lvl3_probe {
    use super::KeyPair;
    use crate::params::lvl3::Level3;
    use crate::rng::NistPqcRng;

    /// Probe: run the lvl3 keygen spine end-to-end (with all widths now
    /// per-level) and print the outcome. `generate_lvl3` returns
    /// `Unimplemented(PK serialization)` iff the spine COMPLETED, or
    /// `Internal(exhausted)` if the spine failed.
    #[test]
    #[ignore = "heavy probe: lvl3 keygen spine"]
    fn lvl3_keygen_spine_probe() {
        let mut rng = NistPqcRng::new(&[0x42u8; 48]);
        let result = KeyPair::<Level3>::generate(&mut rng);
        eprintln!("[lvl3-probe] generate::<Level3> => {result:?}");
    }

    /// Perf baseline: median wall-clock for keygen / sign / verify at lvl1 and
    /// lvl3. Run in RELEASE: `cargo test --release --features kat,vrfy
    /// perf_baseline -- --ignored --nocapture`.
    #[cfg(feature = "vrfy")]
    #[test]
    #[ignore = "perf baseline (run --release)"]
    fn perf_baseline() {
        use super::{KeyLevel, KeyPair};
        use crate::params::Level1;
        use crate::verification::VerifyLevel;
        use rand_chacha::ChaCha20Rng;
        use rand_chacha::rand_core::SeedableRng;
        use std::time::{Duration, Instant};

        // Under load, min-of-N (fastest iteration = least contention) is a far
        // more stable estimate of true compute time than the median.
        fn best(v: &[Duration]) -> Duration {
            *v.iter().min().expect("non-empty")
        }

        fn bench<P: KeyLevel + VerifyLevel>(tag: &str) {
            const MSG: &[u8] = b"perf baseline message";
            let seed = [0x37u8; 32];
            let mut kg = Vec::new();
            let mut kp = None;
            for _ in 0..4 {
                let mut rng = ChaCha20Rng::from_seed(seed);
                let t = Instant::now();
                let k = KeyPair::<P>::generate(&mut rng).expect("keygen");
                kg.push(t.elapsed());
                kp = Some(k);
            }
            let (sk, vk) = kp.unwrap().into_parts();
            let mut sg = Vec::new();
            let mut sig = None;
            for _ in 0..6 {
                let mut rng = ChaCha20Rng::from_seed(seed);
                let t = Instant::now();
                let s = sk.sign(MSG, &mut rng).expect("sign");
                sg.push(t.elapsed());
                sig = Some(s);
            }
            let sig = sig.unwrap();
            let mut vf = Vec::new();
            for _ in 0..8 {
                let t = Instant::now();
                vk.verify(MSG, &sig).expect("verify");
                vf.push(t.elapsed());
            }
            eprintln!(
                "[perf {tag}] keygen={:?} sign={:?} verify={:?} (min of {}/{}/{})",
                best(&kg),
                best(&sg),
                best(&vf),
                kg.len(),
                sg.len(),
                vf.len()
            );
        }
        bench::<Level1>("lvl1");
        bench::<Level3>("lvl3");
    }

    /// Width-minimization correctness oracle: keygen once, then sign+verify
    /// with many DISTINCT sign-RNG seeds. Each seed drives a different commit
    /// re-randomization (β), so the run exercises the full spread of commit-
    /// ideal entry magnitudes — the intermittent `det_4x4` HNF overflow that a
    /// too-narrow WL causes shows up as a sign failure or a verify failure on
    /// some seed. All seeds must pass. Run in RELEASE:
    /// `cargo test --release --features kat,vrfy width_stress_lvl3 -- --ignored --nocapture`.
    #[cfg(feature = "vrfy")]
    fn width_stress<P: super::KeyLevel + crate::verification::VerifyLevel>(tag: &str) {
        use super::KeyPair;
        use rand_chacha::ChaCha20Rng;
        use rand_chacha::rand_core::SeedableRng;

        const MSG: &[u8] = b"width stress message";
        const N: u64 = 16;
        let mut kg = ChaCha20Rng::seed_from_u64(0xA11CE);
        let kp = KeyPair::<P>::generate(&mut kg).expect("keygen");
        let (sk, vk) = kp.into_parts();
        let mut ok = 0u64;
        for s in 0..N {
            let mut r = ChaCha20Rng::seed_from_u64(0x5EED_0000 + s);
            match sk.sign(MSG, &mut r) {
                Ok(sig) => {
                    vk.verify(MSG, &sig).expect("verify seed");
                    ok += 1;
                }
                Err(e) => eprintln!("[width_stress {tag}] sign failed on seed {s}: {e:?}"),
            }
        }
        eprintln!("[width_stress {tag}] {ok}/{N} sign+verify OK");
        assert_eq!(ok, N, "all seeds must sign+verify");
    }

    #[cfg(feature = "vrfy")]
    #[test]
    #[ignore = "width-minimization stress (run --release)"]
    fn width_stress_lvl3() {
        width_stress::<Level3>("lvl3");
    }

    #[cfg(feature = "vrfy")]
    #[test]
    #[ignore = "width-minimization stress (run --release)"]
    fn width_stress_lvl1() {
        use crate::params::Level1;
        width_stress::<Level1>("lvl1");
    }

    /// Profiling target: sign lvl3 in a tight loop for ~30s so an external
    /// sampler (`sample <pid>`) can rank hot functions. Run in RELEASE:
    /// `cargo test --release --features kat,vrfy prof_lvl3_sign_loop -- --ignored --nocapture`.
    #[test]
    #[ignore = "profiling loop (attach sampler)"]
    fn prof_lvl3_sign_loop() {
        use super::KeyPair;
        use crate::params::Level3;
        use rand_chacha::ChaCha20Rng;
        use rand_chacha::rand_core::SeedableRng;
        use std::time::{Duration, Instant};

        const MSG: &[u8] = b"perf baseline message";
        let seed = [0x37u8; 32];
        let mut rng = ChaCha20Rng::from_seed(seed);
        let kp = KeyPair::<Level3>::generate(&mut rng).expect("keygen");
        let sk = kp.signing_key();
        eprintln!(
            "[prof] keygen done, signing loop starting (pid={})",
            std::process::id()
        );
        let start = Instant::now();
        let mut n = 0u64;
        while start.elapsed() < Duration::from_secs(30) {
            let mut r = ChaCha20Rng::from_seed(seed);
            let _ = sk.sign(MSG, &mut r).expect("sign");
            n += 1;
        }
        eprintln!("[prof] {n} signs in {:?}", start.elapsed());
    }

    /// Does the PUBLIC API `KeyPair::<Level3>::generate` reproduce the byte-exact
    /// KAT public key when seeded with the KAT record-0 DRBG seed? Decides the
    /// keygen reconciliation: if yes, the commit-based public path is already
    /// byte-exact; if no, it must adopt the `sample_secret_gen` front.
    #[test]
    #[ignore = "heavy: byte-exact check of public generate vs lvl3 KAT pk"]
    fn lvl3_generate_matches_kat_pk() {
        let seed: [u8; 48] = [
            0x06, 0x15, 0x50, 0x23, 0x4D, 0x15, 0x8C, 0x5E, 0xC9, 0x55, 0x95, 0xFE, 0x04, 0xEF,
            0x7A, 0x25, 0x76, 0x7F, 0x2E, 0x24, 0xCC, 0x2B, 0xC4, 0x79, 0xD0, 0x9D, 0x86, 0xDC,
            0x9A, 0xBC, 0xFD, 0xE7, 0x05, 0x6A, 0x8C, 0x26, 0x6F, 0x9E, 0xF9, 0x7E, 0xD0, 0x85,
            0x41, 0xDB, 0xD2, 0xE1, 0xFF, 0xA1,
        ];
        const KAT_PK0_A: [u8; 96] = [
            0xc3, 0x23, 0x77, 0xd6, 0xf6, 0xd7, 0x07, 0x29, 0x88, 0x4a, 0x7f, 0x68, 0x77, 0xef,
            0x47, 0x91, 0xe3, 0x5d, 0x21, 0xf7, 0x51, 0xa3, 0xe9, 0x6d, 0xe2, 0x3f, 0x9a, 0x7a,
            0x3c, 0x01, 0xbc, 0xd8, 0xa5, 0xf1, 0x46, 0xdc, 0x19, 0xe4, 0xe2, 0xac, 0x63, 0x00,
            0x74, 0x57, 0xf9, 0x7d, 0x8a, 0x40, 0xee, 0x84, 0xae, 0xe7, 0x56, 0x4c, 0xa9, 0xa7,
            0xfb, 0xe6, 0x20, 0x0f, 0xd3, 0xe5, 0xe5, 0x59, 0x01, 0xbf, 0xc6, 0x0e, 0xb2, 0x5c,
            0x50, 0xd3, 0x9f, 0x5c, 0x91, 0xc9, 0x65, 0x10, 0x55, 0x6b, 0xaa, 0x22, 0x02, 0x8d,
            0xf7, 0x63, 0x60, 0x84, 0x17, 0x21, 0xa6, 0x01, 0xd6, 0x5e, 0x8d, 0x0f,
        ];
        let mut rng = NistPqcRng::new(&seed);
        let kp = KeyPair::<Level3>::generate(&mut rng).expect("lvl3 keygen");
        let pk = kp.verifying_key().as_bytes();
        let matches = pk.len() >= 96 && pk[..96] == KAT_PK0_A;
        eprintln!(
            "[lvl3-gen-kat] pk[..8]={:02x?} bytes_match={matches}",
            &pk[..8]
        );
        assert!(
            matches,
            "public generate must reproduce the byte-exact lvl3 KAT pk"
        );
    }

    /// Byte-exact keygen against EVERY lvl3 KAT record, not just seed-0. This is
    /// the real C-compatibility gate: for each of the 100 NIST DRBG seeds, the
    /// public `generate` must reproduce the C reference's full public key
    /// (all `PK_BYTES`, including the hint byte). Guards the width-minimized
    /// keygen `WN` — a too-narrow width that happened to pass seed-0 but diverges
    /// on any other seed is caught here. `KAT_N` env caps the count (default all
    /// 100). Run in RELEASE (each keygen ~1.1s):
    /// `cargo test --release --features kat,vrfy lvl3_generate_matches_all_kat_pks -- --ignored --nocapture`.
    /// Byte-exact keygen vs EVERY KAT record (all `PK_BYTES`, hint included),
    /// generic over level. The C-faithful combine-kernel construction (montgomery
    /// x-only doubling + non-normalized `xDBL` + projective `(A:C)`) reproduces the
    /// reference's exact Montgomery model, so this passes 100/100. `KAT_N` caps
    /// the count. `tag` labels the run.
    fn all_kat_pks_match<P: super::KeyLevel>(rsp: &str, tag: &str) {
        let limit: usize = std::env::var("KAT_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);
        let mut seed: Option<Vec<u8>> = None;
        let mut checked = 0usize;
        let mut full_match = 0usize;
        // Byte-DIVERGED = keygen succeeded but produced a different (wrong-model)
        // pk than C — the defect the combine-kernel fix eliminates; must be 0.
        let mut model_diverged: Vec<usize> = Vec::new();
        // find_uv GAP = keygen couldn't produce a key at all for this seed. This
        // is a separate, pre-existing `find_uv_cref` coverage limitation (box
        // size / alternate orders), NOT a model bug; reported, not asserted.
        let mut finduv_gap: Vec<usize> = Vec::new();
        for line in rsp.lines() {
            if let Some(h) = line.strip_prefix("seed = ") {
                seed = Some(hex::decode(h.trim()).expect("seed hex"));
            } else if let Some(h) = line.strip_prefix("pk = ") {
                let pk_expected = hex::decode(h.trim()).expect("pk hex");
                let s = seed.take().expect("seed precedes pk in KAT record");
                let seed48: [u8; 48] = s.as_slice().try_into().expect("48-byte NIST seed");
                let mut rng = NistPqcRng::new(&seed48);
                match KeyPair::<P>::generate(&mut rng) {
                    Ok(kp) if kp.verifying_key().as_bytes() == pk_expected.as_slice() => {
                        full_match += 1;
                    }
                    Ok(_) => model_diverged.push(checked),
                    Err(_) => finduv_gap.push(checked),
                }
                checked += 1;
                if checked >= limit {
                    break;
                }
            }
        }
        eprintln!(
            "[{tag}-all-kat] {full_match}/{checked} byte-exact; model_diverged={model_diverged:?} finduv_gap={finduv_gap:?}"
        );
        // Guarantee: EVERY keygen public key is byte-identical to C's — zero
        // wrong-model divergence AND zero find_uv gaps. The alternate-order
        // find_uv (`find_uv_cref_alt`) + the index-aware combine close the last
        // gap (lvl1 record 29, index_order2 = 1), so both levels are 100/100.
        assert!(
            model_diverged.is_empty(),
            "keygen produced wrong-model pks (not byte-exact) for records {model_diverged:?}"
        );
        assert!(
            finduv_gap.is_empty(),
            "keygen find_uv failed (no isogeny) for records {finduv_gap:?}"
        );
    }

    #[test]
    #[ignore = "heavy: byte-exact keygen vs ALL 100 lvl3 KAT public keys"]
    fn lvl3_generate_matches_all_kat_pks() {
        all_kat_pks_match::<Level3>(
            include_str!("../tests/KAT/PQCsignKAT_529_SQIsign_lvl3.rsp"),
            "lvl3",
        );
    }

    #[test]
    #[ignore = "heavy: byte-exact keygen vs ALL 100 lvl1 KAT public keys"]
    fn lvl1_generate_matches_all_kat_pks() {
        use crate::params::Level1;
        all_kat_pks_match::<Level1>(
            include_str!("../tests/KAT/PQCsignKAT_353_SQIsign_lvl1.rsp"),
            "lvl1",
        );
    }

    /// Diagnostic: for every diverging lvl3 KAT record, classify the relation
    /// between our keygen's curve coefficient `A` and the C reference's `A`.
    /// Answers: is it always the same Montgomery-model transform (−A, conjugate,
    /// −conjugate), and do the j-invariants agree (same curve, just a different
    /// model)? This sizes the byte-exactness fix. Run in RELEASE:
    /// `cargo test --release --features kat,vrfy diag_lvl3_kat_model_relation -- --ignored --nocapture`.
    #[test]
    #[ignore = "diagnostic: classify keygen A-vs-C model divergence"]
    fn diag_lvl3_kat_model_relation() {
        use crate::ec::montgomery::MontgomeryCurve;
        use crate::gf::fp2::Fp2;
        use crate::params::lvl3::Fp3Element;
        use subtle::ConstantTimeEq;

        const RSP: &str = include_str!("../tests/KAT/PQCsignKAT_529_SQIsign_lvl3.rsp");
        let limit: usize = std::env::var("KAT_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);
        let decode = |bytes: &[u8]| -> Fp2<Fp3Element> {
            Fp2::<Fp3Element>::from_bytes_le(&bytes[..96]).expect("valid Fp2 A")
        };
        let mut seed: Option<Vec<u8>> = None;
        let (mut neg, mut conj, mut negconj, mut other, mut jne, mut checked) =
            (0u32, 0u32, 0u32, 0u32, 0u32, 0usize);
        for line in RSP.lines() {
            if let Some(h) = line.strip_prefix("seed = ") {
                seed = Some(hex::decode(h.trim()).expect("seed hex"));
            } else if let Some(h) = line.strip_prefix("pk = ") {
                let kat = hex::decode(h.trim()).expect("pk hex");
                let s = seed.take().expect("seed precedes pk");
                let seed48: [u8; 48] = s.as_slice().try_into().expect("48-byte seed");
                let mut rng = NistPqcRng::new(&seed48);
                let kp = KeyPair::<Level3>::generate(&mut rng).expect("keygen");
                let our = kp.verifying_key().as_bytes();
                if our[..96] != kat[..96] {
                    let a_our = decode(&our);
                    let a_kat = decode(&kat);
                    let j_our = MontgomeryCurve::<Fp3Element>::new(a_our).j_invariant();
                    let j_kat = MontgomeryCurve::<Fp3Element>::new(a_kat).j_invariant();
                    let j_eq = bool::from(j_our.ct_eq(&j_kat));
                    if !j_eq {
                        jne += 1;
                    }
                    if bool::from(a_our.ct_eq(&a_kat.negate())) {
                        neg += 1;
                    } else if bool::from(a_our.ct_eq(&a_kat.frobenius())) {
                        conj += 1;
                    } else if bool::from(a_our.ct_eq(&a_kat.frobenius().negate())) {
                        negconj += 1;
                    } else {
                        other += 1;
                        if other <= 3 {
                            eprintln!(
                                "[model] record {checked}: OTHER relation, j_eq={j_eq} our_A[..8]={:02x?} kat_A[..8]={:02x?}",
                                &our[..8],
                                &kat[..8]
                            );
                        }
                    }
                }
                checked += 1;
                if checked >= limit {
                    break;
                }
            }
        }
        eprintln!(
            "[model] diverged breakdown: -A={neg} conj={conj} -conj={negconj} other={other} | j_invariant_differs={jne}"
        );
    }

    /// One-shot: are our final-split theta null point and C's (record 3,
    /// captured via PQSQ_SPLIT3) the SAME projective point? Cross-product test
    /// X·Y', Y·X' etc. If proportional, the divergence is downstream of the
    /// null (split-matrix application); if not, our theta chain produced a
    /// genuinely different null point than C.
    #[test]
    #[ignore = "one-shot projective null-point comparison (hardcoded capture)"]
    fn diag_null_point_projective() {
        use crate::gf::fp2::Fp2;
        use crate::params::lvl3::Fp3Element;
        use subtle::ConstantTimeEq;
        let dec = |h: &str| -> Fp2<Fp3Element> {
            let bytes = hex::decode(h).expect("hex");
            Fp2::<Fp3Element>::from_bytes_le(&bytes).expect("fp2")
        };
        // record-3 final randomized-split null point, C vs ours.
        let cx = dec(
            "9b3b22a57b95bbf4f0e2793899282c257535c2697e3f96512c39bb546dc421fdd1e20311982e004cfb43bd6ca50f9f3dee52640c426c4630d7e6086f50a90c9a19fce1dc13c5d42f0429d8e9d30b8086e963d8662e4b6554a24b5ca3d83f7f0b",
        );
        let cy = dec(
            "8e5261e4f2b2695d2345650afc876d557d00c8e5a988c78add4f5b7f24d2dbb854129110bfbb056f8da91d0fa384fc08d23dc42e332e6bbddcfa55b1ffd050e0dfa7255b78728f809eab8e1129766b86f12c6d2a9dbcdca352cdfcb419157837",
        );
        let cz = dec(
            "3a8fefb33cf9d52e3df18a37d798b3513a0e027a89cacba1f4ba82b301f820bb0b1dce4765ea2d8de138949cfd788212905cb8aaf266ce24283e807573f02edf55b8d501d1c64bbed06713d30ab26cdaabc1e7cd2e23890aafba5daebcebf33e",
        );
        let ox = dec(
            "53bda7cc7ac85cd262bd56fe9e6d878904015f5298b1951819482d84d54d0dbbfb652dd37ede977c8d77ba2ee33d9225e014bb21fb6ff7811f5f75e9a365e724f47e4d1b93f369d16ae74af5072f1fd2847190407f9c81cc49a82936d2ffe230",
        );
        let oy = dec(
            "4c18ee13cd0109519b9b84fb69da20ca3761126f353e5385282d2c3b923fd935196bf5a980b2fa2e229cf17736d84b31e0b8be9b193279b45e6aef678eac1a4ddeb7e7071f27e75bbd6769bdbfe190cadf62ea17a0ef6740055fc310d74ca119",
        );
        let oz = dec(
            "b8fd4bbf1af59cf8a9498dad436005ef9b958d9b1b5125e5c7f21f8e29b814fa42725842cce4453967db2363d56093273f4d1025b51ed7675c1cbcfdf7e90b88cb2e0f38439cf51aed55bf4c691710aa1d486a1a1b324f50863b9b5f9e30fe07",
        );
        let xy = bool::from(ox.mul(&cy).ct_eq(&oy.mul(&cx)));
        let xz = bool::from(ox.mul(&cz).ct_eq(&oz.mul(&cx)));
        eprintln!("[null-proj] X:Y proportional={xy}  X:Z proportional={xz}");
        eprintln!(
            "[null-proj] => {}",
            if xy && xz {
                "SAME projective point (bug is downstream of the null)"
            } else {
                "DIFFERENT null point (theta chain diverges upstream)"
            }
        );
    }

    /// Regression for the `two_resp>0` short-response branch: signing a fixed
    /// lvl1 key over many messages must yield signatures that ALL verify. This
    /// branch (~1 in 3 messages) needed two fixes — conjugating the response
    /// before the small-chain ideal, and feeding the *primitive* response to the
    /// aux helper so `two_resp` isn't over-counted by `2·backtracking`. 40
    /// messages reliably exercise both `two_resp>0` and `backtracking>0`.
    #[test]
    #[ignore = "heavy: lvl1 sign→verify across many messages"]
    fn lvl1_sign_verify_many_messages() {
        use crate::params::Level1;
        let mut rng = NistPqcRng::new(&[0x77u8; 48]);
        let kp = KeyPair::<Level1>::generate(&mut rng).expect("lvl1 keygen");
        for i in 0u8..40 {
            let msg = [i; 4];
            let sig = kp
                .signing_key()
                .sign(&msg, &mut rng)
                .unwrap_or_else(|e| panic!("lvl1 sign failed for msg {i}: {e:?}"));
            kp.verifying_key()
                .verify(&msg, &sig)
                .unwrap_or_else(|e| panic!("lvl1 verify failed for msg {i}: {e:?}"));
        }
    }

    /// Full lvl3 end-to-end: keygen → sign → verify, the level-3 analogue of
    /// `signing::tests::sign_verify_roundtrip`. Exercises the field-generic
    /// `protocols_sign` / `protocols_verify` via the typed API. HEAVY.
    #[test]
    #[ignore = "heavy: full lvl3 keygen → sign → verify roundtrip (real-scale spine)"]
    fn sign_verify_roundtrip_lvl3() {
        use crate::params::Params;
        let mut rng = NistPqcRng::new(&[0x99u8; 48]);
        let kp = KeyPair::<Level3>::generate(&mut rng).expect("lvl3 keygen");
        let msg = b"sqisign lvl3 roundtrip";
        let sig = kp
            .signing_key()
            .sign(msg, &mut rng)
            .expect("lvl3 sign produces a signature");

        // Positive: verify accepts the genuine signature.
        kp.verifying_key()
            .verify(msg, &sig)
            .expect("lvl3 verify must accept the produced signature");

        // Negative tests — prove verify actually checks (a self-consistent
        // roundtrip alone is also satisfied by a verifier that always accepts).
        let sig_bytes = sig.as_bytes();

        // (a) Wrong message must be rejected.
        assert!(
            kp.verifying_key()
                .verify(b"a different message", &sig)
                .is_err(),
            "lvl3 verify must reject the signature under a different message",
        );

        // (b) A single flipped bit anywhere in the signature must be rejected.
        for &byte_idx in &[0usize, Level3::FP2_BYTES, Level3::SIG_BYTES - 2] {
            let mut tampered = sig_bytes.to_vec();
            tampered[byte_idx] ^= 0x01;
            let tampered_sig =
                crate::sqisignature::SqiSignature::<Level3>::from_bytes_unchecked(&tampered);
            assert!(
                kp.verifying_key().verify(msg, &tampered_sig).is_err(),
                "lvl3 verify must reject a signature with byte {byte_idx} flipped",
            );
        }
    }

    /// lvl3 analogue of `lvl1_sign_verify_many_messages`: signing a fixed lvl3
    /// key over several messages must yield signatures that ALL verify — this
    /// reliably exercises the `two_resp>0` short-response branch at level 3 (the
    /// conjugation + primitive-response fixes are field-generic). HEAVY.
    #[test]
    #[ignore = "heavy: lvl3 sign→verify across several messages (two_resp>0)"]
    fn lvl3_sign_verify_many_messages() {
        let mut rng = NistPqcRng::new(&[0x99u8; 48]);
        let kp = KeyPair::<Level3>::generate(&mut rng).expect("lvl3 keygen");
        for i in 0u8..10 {
            let msg = [i; 4];
            let sig = kp
                .signing_key()
                .sign(&msg, &mut rng)
                .unwrap_or_else(|e| panic!("lvl3 sign failed for msg {i}: {e:?}"));
            kp.verifying_key()
                .verify(&msg, &sig)
                .unwrap_or_else(|e| panic!("lvl3 verify failed for msg {i}: {e:?}"));
        }
    }
}
