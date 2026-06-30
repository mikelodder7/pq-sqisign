// SPDX-License-Identifier: MIT OR Apache-2.0
//! [`SigningKey`] — typed SQIsign signing (secret) key.

use alloc::vec::Vec;

use crate::params::Params;
use crate::sqisignature::SqiSignature;
use crate::{Error, Result};
use core::marker::PhantomData;

/// A SQIsign signing (secret) key, parameterized by security level.
///
/// ## Serialization note
///
/// Keys loaded via [`from_bytes`](Self::from_bytes) can be round-tripped
/// with [`to_bytes`](Self::to_bytes). Keys generated via
/// [`KeyPair::generate`](crate::keypair::KeyPair::generate) cannot currently
/// be serialized (the keygen pipeline does not yet thread the secret generator
/// quaternion through to the output — a planned future addition).
/// [`to_bytes`](Self::to_bytes) returns
/// [`Error::SkSerializationNotSupported`] for those keys.
pub struct SigningKey<P: Params> {
    data: crate::verification::SecretKeyData<P::Field>,
    /// Present only when loaded from bytes; enables [`to_bytes`](Self::to_bytes).
    encoded: Option<Vec<u8>>,
    _params: PhantomData<P>,
}

impl<P: Params> core::fmt::Debug for SigningKey<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SigningKey")
            .field("level", &P::LEVEL)
            .field("encoded", &self.encoded.is_some())
            .finish_non_exhaustive()
    }
}

impl<P: Params> SigningKey<P> {
    /// Construct from a live [`SecretKeyData`] (generated key, no byte encoding).
    pub(crate) fn from_secret_data(data: crate::verification::SecretKeyData<P::Field>) -> Self {
        Self {
            data,
            encoded: None,
            _params: PhantomData,
        }
    }

    /// Decode a signing key from the SQIsign secret-key wire format.
    ///
    /// The decoded key supports both signing and byte round-trip via
    /// [`to_bytes`](Self::to_bytes).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self>
    where
        P: crate::keypair::KeyLevel,
    {
        if bytes.len() < P::SK_BYTES {
            return Err(Error::BufferTooSmall {
                required: P::SK_BYTES,
                provided: bytes.len(),
            });
        }
        let data = <P as crate::keypair::KeyLevel>::sk_from_bytes(&bytes[..P::SK_BYTES])?;
        Ok(Self {
            data,
            encoded: Some(bytes[..P::SK_BYTES].to_vec()),
            _params: PhantomData,
        })
    }

    /// Return the raw secret-key bytes (`P::SK_BYTES` long).
    ///
    /// Only available for keys loaded via [`from_bytes`](Self::from_bytes).
    /// Returns [`Error::SkSerializationNotSupported`] for generated keys.
    pub fn to_bytes(&self) -> Result<&[u8]> {
        self.encoded
            .as_deref()
            .ok_or(Error::SkSerializationNotSupported)
    }

    /// Sign `msg` with this key, using `rng` for commitment randomization.
    #[cfg(feature = "sign")]
    pub fn sign<R: rand_core::CryptoRng>(&self, msg: &[u8], rng: &mut R) -> Result<SqiSignature<P>>
    where
        P: crate::keypair::KeyLevel,
    {
        let sig_bytes = match <P as crate::keypair::KeyLevel>::protocols_sign(&self.data, msg, rng)
        {
            Some(b) => b,
            None if P::LEVEL == 1 => return Err(Error::SigningFailed),
            None => {
                return Err(Error::Unimplemented(
                    "sign: only security level 1 supported",
                ));
            }
        };
        Ok(SqiSignature::from_bytes_unchecked(&sig_bytes))
    }
}

// `SigningKey::from_bytes` requires `P: KeyLevel` (per-level secret-key decode),
// so its `TryFrom<&[u8]>` is written out rather than via the shared macro.
impl<P: crate::keypair::KeyLevel> TryFrom<&[u8]> for SigningKey<P> {
    type Error = Error;
    fn try_from(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes(bytes)
    }
}

#[cfg(all(test, feature = "kat"))]
mod tests {
    use crate::keypair::KeyPair;
    use crate::params::Level1;
    use crate::rng::NistPqcRng;

    #[test]
    #[ignore = "slow — run with: cargo test --features kat -- --ignored typed_api_roundtrip"]
    fn typed_api_roundtrip() {
        let seed = [0u8; 48];
        let mut rng = NistPqcRng::new(&seed);
        let kp = KeyPair::<Level1>::generate(&mut rng).expect("keygen failed");
        let msg = b"hello sqisign typed api";
        let sig = kp.signing_key().sign(msg, &mut rng).expect("sign failed");
        kp.verifying_key().verify(msg, &sig).expect("verify failed");
    }
}
