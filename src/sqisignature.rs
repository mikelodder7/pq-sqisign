// SPDX-License-Identifier: MIT OR Apache-2.0
//! [`SqiSignature`] — typed SQIsign signature wrapper.

use alloc::vec::Vec;

use crate::params::Params;
use crate::{Error, Result};
use core::marker::PhantomData;

/// A SQIsign signature, parameterized by security level [`Params`].
///
/// Wraps the raw signature bytes produced by `SigningKey::sign`.
/// Use `VerifyingKey::verify` to check it against a message and public key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SqiSignature<P: Params> {
    bytes: Vec<u8>,
    _params: PhantomData<P>,
}

impl<P: Params> SqiSignature<P> {
    /// Wrap raw bytes without length-checking. Caller guarantees `bytes.len() == P::SIG_BYTES`.
    pub(crate) fn from_bytes_unchecked(bytes: &[u8]) -> Self {
        Self {
            bytes: bytes.to_vec(),
            _params: PhantomData,
        }
    }

    /// Decode a signature from its byte representation.
    ///
    /// Returns [`Error::BufferTooSmall`] if `bytes.len() < P::SIG_BYTES`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < P::SIG_BYTES {
            return Err(Error::BufferTooSmall {
                required: P::SIG_BYTES,
                provided: bytes.len(),
            });
        }
        Ok(Self::from_bytes_unchecked(&bytes[..P::SIG_BYTES]))
    }

    /// The raw signature bytes (`P::SIG_BYTES` long).
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl_bytes_conversions!(bytes_field: SqiSignature<P>);

#[cfg(feature = "signature")]
impl<P: Params> From<SqiSignature<P>> for Vec<u8> {
    fn from(sig: SqiSignature<P>) -> Vec<u8> {
        sig.bytes
    }
}

#[cfg(feature = "signature")]
impl<P: Params> TryFrom<Vec<u8>> for SqiSignature<P> {
    type Error = signature::Error;

    fn try_from(v: Vec<u8>) -> core::result::Result<Self, signature::Error> {
        if v.len() < P::SIG_BYTES {
            return Err(signature::Error::new());
        }
        Ok(Self {
            bytes: v[..P::SIG_BYTES].to_vec(),
            _params: PhantomData,
        })
    }
}

#[cfg(feature = "signature")]
impl<P: Params> signature::SignatureEncoding for SqiSignature<P> {
    type Repr = Vec<u8>;
}
