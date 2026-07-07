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
