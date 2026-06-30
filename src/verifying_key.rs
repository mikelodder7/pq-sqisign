// SPDX-License-Identifier: MIT OR Apache-2.0
//! [`VerifyingKey`] — typed SQIsign verifying (public) key.

use alloc::vec::Vec;

use crate::params::Params;
use crate::sqisignature::SqiSignature;
use crate::{Error, Result};
use core::marker::PhantomData;

/// A SQIsign verifying (public) key, parameterized by security level.
///
/// Holds the public-key bytes and implements [`verify`](Self::verify).
/// Construct via `KeyPair::generate` or [`VerifyingKey::from_bytes`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyingKey<P: Params> {
    bytes: Vec<u8>,
    _params: PhantomData<P>,
}

impl<P: Params> VerifyingKey<P> {
    /// Wrap already-validated bytes (length == `P::PK_BYTES`).
    pub(crate) fn from_bytes_unchecked(bytes: &[u8]) -> Self {
        Self {
            bytes: bytes[..P::PK_BYTES].to_vec(),
            _params: PhantomData,
        }
    }

    /// Decode a verifying key from its byte representation.
    ///
    /// Validates structure (byte length, field-element canonicality).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < P::PK_BYTES {
            return Err(Error::BufferTooSmall {
                required: P::PK_BYTES,
                provided: bytes.len(),
            });
        }
        // Validate by attempting a decode.
        crate::verification::PublicKeyData::from_bytes_lvl1(&bytes[..P::PK_BYTES])?;
        Ok(Self::from_bytes_unchecked(bytes))
    }

    /// Raw public-key bytes (`P::PK_BYTES` long).
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Verify that `signature` on `msg` is valid under this key.
    ///
    /// Returns `Ok(())` on success; [`Error::InvalidSignature`] on failure.
    pub fn verify(&self, msg: &[u8], signature: &SqiSignature<P>) -> Result<()>
    where
        P: crate::verification::VerifyLevel,
    {
        P::verify_bytes(signature.as_bytes(), &self.bytes, msg)
    }
}

impl_bytes_conversions!(bytes_field: VerifyingKey<P>);

#[cfg(feature = "signature")]
impl<P: crate::verification::VerifyLevel> signature::Verifier<SqiSignature<P>> for VerifyingKey<P> {
    fn verify(
        &self,
        msg: &[u8],
        sig: &SqiSignature<P>,
    ) -> core::result::Result<(), signature::Error> {
        VerifyingKey::verify(self, msg, sig).map_err(|_| signature::Error::new())
    }
}
