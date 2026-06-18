// SPDX-License-Identifier: MIT OR Apache-2.0
//! [`KeyPair`] — typed SQIsign keypair generation and access.

use crate::params::Params;
use crate::signing_key::SigningKey;
use crate::verifying_key::VerifyingKey;
use crate::{Error, Result};

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
    /// Generate a fresh keypair using the provided randomness source.
    ///
    /// Currently only security level 1 is supported. The generated
    /// [`SigningKey`] cannot be serialized to bytes (see
    /// [`SigningKey::to_bytes`] for details); use
    /// [`from_signing_key_bytes`](Self::from_signing_key_bytes) to create a
    /// round-trippable keypair from stored secret-key bytes.
    #[cfg(feature = "kgen")]
    pub fn generate<R: rand_core::CryptoRng>(rng: &mut R) -> Result<Self> {
        match P::LEVEL {
            1 => Self::generate_lvl1(rng),
            _ => Err(Error::Unimplemented(
                "keypair: only security level 1 supported",
            )),
        }
    }

    #[cfg(feature = "kgen")]
    fn generate_lvl1<R: rand_core::CryptoRng>(rng: &mut R) -> Result<Self> {
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

        let pk_data = PublicKeyData {
            curve_a: e_a.a,
            hint_pk,
        };
        let mut pk_bytes = alloc::vec![0u8; P::PK_BYTES];
        pk_data.to_bytes_lvl1(&mut pk_bytes)?;

        Ok(Self {
            signing_key: SigningKey::from_secret_data(sk_data),
            verifying_key: VerifyingKey::from_bytes_unchecked(&pk_bytes),
        })
    }

    /// Reconstruct a keypair from the secret-key wire-format bytes.
    ///
    /// The public key is embedded in the first `P::PK_BYTES` of the
    /// secret-key encoding and extracted automatically.
    pub fn from_signing_key_bytes(sk_bytes: &[u8]) -> Result<Self> {
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
