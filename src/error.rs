// SPDX-License-Identifier: MIT OR Apache-2.0
//! Error type for the public SQIsign API.

use core::fmt;

/// Result alias used throughout the crate.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors that can occur during SQIsign keypair / sign / verify operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// A provided buffer was too small for the operation.
    BufferTooSmall {
        /// Required minimum byte length.
        required: usize,
        /// Actual byte length provided by caller.
        provided: usize,
    },
    /// Encoded field element was not canonical (greater than or equal to the prime).
    NonCanonicalEncoding,
    /// Signature decoded but did not verify against the public key and message.
    InvalidSignature,
    /// Public key bytes did not decode to a valid point on the expected curve.
    InvalidPublicKey,
    /// Decoded theta-null is degenerate — the corresponding abelian variety
    /// has no well-defined doubling constants (i.e. `H(theta_null²)` has at
    /// least one zero component, so its componentwise inverse is undefined).
    InvalidThetaNull,
    /// Secret key bytes did not decode to a valid representation.
    InvalidSecretKey,
    /// Internal invariant violation — should not occur with correct inputs.
    Internal(&'static str),
    /// Path not yet implemented in this build of the crate (multi-session port in progress).
    Unimplemented(&'static str),
    /// Bezout / short-vector search exhausted its budget without finding a
    /// solution within the configured box / list size. Distinct from
    /// [`Error::Unimplemented`]: the path IS implemented, the inputs simply
    /// did not yield a solution. Emitted by the Clapotis `find_uv`
    /// orchestrator when either the short-vector enumeration returns empty
    /// OR the Bezout pair search rejects every candidate.
    NoBezoutSolution(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BufferTooSmall { required, provided } => write!(
                f,
                "buffer too small: required {required} bytes, got {provided}"
            ),
            Error::NonCanonicalEncoding => f.write_str("non-canonical field encoding"),
            Error::InvalidSignature => f.write_str("invalid signature"),
            Error::InvalidPublicKey => f.write_str("invalid public key encoding"),
            Error::InvalidThetaNull => {
                f.write_str("degenerate theta-null: doubling constants undefined")
            }
            Error::InvalidSecretKey => f.write_str("invalid secret key encoding"),
            Error::Internal(msg) => write!(f, "internal invariant violated: {msg}"),
            Error::Unimplemented(msg) => write!(f, "not yet implemented: {msg}"),
            Error::NoBezoutSolution(msg) => write!(f, "no Bezout solution found: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}
