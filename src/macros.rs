// SPDX-License-Identifier: MIT OR Apache-2.0

/// Generate byte-slice conversion impls for a [`Params`](crate::params::Params)-generic
/// typed-API type.
///
/// Two arms:
///
/// - `bytes_field` — the type stores its data as `self.bytes: Vec<u8>` directly.
///   Generates both `AsRef<[u8]>` (via `self.bytes`) and `TryFrom<&[u8]>` (via
///   `from_bytes`). Use for [`SqiSignature`](crate::sqisignature::SqiSignature)
///   and [`VerifyingKey`](crate::verifying_key::VerifyingKey).
///
/// - `no_bytes_field` — the type does not expose a direct byte slice (e.g.
///   [`SigningKey`](crate::signing_key::SigningKey) whose encoded field is
///   `Option<Vec<u8>>`). Generates only `TryFrom<&[u8]>` (via `from_bytes`).
macro_rules! impl_bytes_conversions {
    (bytes_field: $t:ty) => {
        impl<P: $crate::params::Params> AsRef<[u8]> for $t {
            #[inline]
            fn as_ref(&self) -> &[u8] {
                &self.bytes
            }
        }

        impl_bytes_conversions!(no_bytes_field: $t);
    };

    (no_bytes_field: $t:ty) => {
        impl<P: $crate::params::Params> TryFrom<&[u8]> for $t {
            type Error = $crate::Error;

            fn try_from(bytes: &[u8]) -> $crate::Result<Self> {
                Self::from_bytes(bytes)
            }
        }
    };
}
