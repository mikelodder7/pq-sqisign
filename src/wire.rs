// SPDX-License-Identifier: MIT OR Apache-2.0
//! Wire-format types for SQIsign public keys, secret keys, and signatures.
//!
//! Each spec-defined byte size factors as `2·F::ENCODED_BYTES + extra`:
//!
//! | Level | `F::ENCODED_BYTES` | `PK_BYTES` | `PK_BYTES − 2·n` |
//! |-------|--------------------|------------|------------------|
//! | L1    | 32                 | 65         | 1                |
//! | L3    | 48                 | 97         | 1                |
//! | L5    | 64                 | 129        | 1                |
//!
//! The `+1` byte is consistent across levels — modelled here as a
//! single `reserved` byte beyond the `F_{p²}` encoding of the public
//! curve's affine `A` coefficient. Its semantics are TBD per the
//! SQIsign spec (likely a compression flag or a curve-form indicator).
//! Until the spec interpretation is finalised, `reserved` is treated
//! as opaque — encoders preserve it round-trip, decoders accept any
//! byte value.
//!
//! This module ships the wire-format types and their `encode`/`decode`
//! round-trip surface. The cryptographic content (the `a_pk` element)
//! comes from the Clapotis evaluator once that algorithmic body lands.

use rand_core::CryptoRng;
use subtle::{Choice, ConstantTimeEq, CtOption};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::error::{Error, Result};
use crate::gf::fp::BaseField;
use crate::gf::fp2::Fp2;

/// SQIsign public key — the public commitment curve's affine `A`
/// coefficient (an `F_{p²}` element) plus a spec-defined reserved byte.
///
/// Wire-format size: `2·F::ENCODED_BYTES + 1` bytes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PublicKey<F: BaseField> {
    /// Affine `A` coefficient of the public Montgomery curve.
    pub a_pk: Fp2<F>,
    /// Spec-defined reserved byte. Semantics TBD per SQIsign spec
    /// study; round-tripped opaquely until clarified.
    pub reserved: u8,
}

impl<F: BaseField> PublicKey<F> {
    /// Wire-format size in bytes.
    pub const WIRE_BYTES: usize = 2 * F::ENCODED_BYTES + 1;

    /// Construct a public key from its components.
    pub const fn new(a_pk: Fp2<F>, reserved: u8) -> Self {
        Self { a_pk, reserved }
    }

    /// Encode `self` into `out`. Returns `BufferTooSmall` if
    /// `out.len() < WIRE_BYTES`. Writes exactly `WIRE_BYTES` bytes.
    pub fn encode(&self, out: &mut [u8]) -> Result<()> {
        if out.len() < Self::WIRE_BYTES {
            return Err(Error::BufferTooSmall {
                required: Self::WIRE_BYTES,
                provided: out.len(),
            });
        }
        let n = F::ENCODED_BYTES;
        self.a_pk.to_bytes_le(&mut out[..2 * n]);
        out[2 * n] = self.reserved;
        Ok(())
    }

    /// Decode a public key from `bytes`. Returns `BufferTooSmall` if
    /// `bytes.len() < WIRE_BYTES`; returns `InvalidPublicKey` if the
    /// `a_pk` component is non-canonical (`re` or `im` ≥ p).
    ///
    /// Reads exactly `WIRE_BYTES` bytes; extra bytes are ignored.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::WIRE_BYTES {
            return Err(Error::BufferTooSmall {
                required: Self::WIRE_BYTES,
                provided: bytes.len(),
            });
        }
        let n = F::ENCODED_BYTES;
        let a_pk_opt: CtOption<Fp2<F>> = Fp2::<F>::from_bytes_le(&bytes[..2 * n]);
        // CtOption → Option → Result mapping. Use `into_option` to
        // resolve the constant-time choice into a regular Option, then
        // map None to InvalidPublicKey.
        let a_pk = a_pk_opt.into_option().ok_or(Error::InvalidPublicKey)?;
        let reserved = bytes[2 * n];
        Ok(Self { a_pk, reserved })
    }
}

/// SQIsign secret key — opaque byte container of size `N` matching
/// the level's `Params::SK_BYTES`.
///
/// Unlike [`PublicKey<F>`] (which decomposes into a typed `Fp2<F>`
/// curve coefficient plus a reserved byte), the secret key is
/// treated as opaque bytes pending spec study of the internal
/// layout (quaternion / ideal / response state in the C reference).
/// This type provides the round-trip wire-format surface; the
/// internal structure is decoded only inside the sign path.
///
/// Use one of the per-level type aliases [`SecretKeyLvl1`],
/// [`SecretKeyLvl3`], [`SecretKeyLvl5`] rather than instantiating
/// the const-generic form directly.
///
/// # Security
///
/// `SecretKey<N>` derives [`Zeroize`] and [`ZeroizeOnDrop`]: the
/// stored bytes are scrubbed from memory when the struct is
/// dropped, and callers may invoke [`Zeroize::zeroize`] explicitly
/// to wipe an in-use instance. The struct is intentionally NOT
/// `Copy` — copying a secret-bearing value would defeat the
/// `ZeroizeOnDrop` guarantee. Use [`Clone`] explicitly when a
/// duplicate is required (and zeroize the clone when finished).
///
/// **`Debug` does NOT print the secret bytes** — the impl below
/// emits `SecretKey<N> { bytes: <redacted; N bytes> }`. This
/// prevents accidental leakage via `dbg!(sk)`, `tracing::debug!`,
/// `println!("{:?}", sk)`, or any other formatter that uses the
/// `Debug` trait. If you genuinely need the byte values (e.g. in
/// test fixtures), use [`Self::as_bytes`] or the public `bytes`
/// field directly.
///
/// **`PartialEq`/`Eq` is variable-time.** The derived impls
/// short-circuit at the first differing byte, leaking the position
/// of the first difference via timing. For any comparison whose
/// input may be attacker-influenced, use [`ConstantTimeEq::ct_eq`]
/// instead.
///
/// **The public `bytes` field exposes secret material** — callers
/// reading or copying it must treat the result as confidential and
/// zeroize any independent copies they create.
#[derive(Clone, Eq, PartialEq, Zeroize, ZeroizeOnDrop)]
pub struct SecretKey<const N: usize> {
    /// Opaque secret-key bytes. Internal layout decoded only by the
    /// sign path; callers should treat this field as a black box
    /// and treat any read/copy as confidential.
    pub bytes: [u8; N],
}

impl<const N: usize> core::fmt::Debug for SecretKey<N> {
    /// Redacted formatter — emits the type name and byte length but
    /// NEVER the byte values themselves. Prevents accidental leakage
    /// of secret material through log lines, panic messages,
    /// `dbg!()` output, or any other path that invokes `Debug`.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SecretKey")
            .field("bytes", &format_args!("<redacted; {} bytes>", N))
            .finish()
    }
}

impl<const N: usize> SecretKey<N> {
    /// Wire-format size in bytes (equal to the const generic `N`).
    pub const WIRE_BYTES: usize = N;

    /// Construct a secret key from its raw bytes.
    pub const fn new(bytes: [u8; N]) -> Self {
        Self { bytes }
    }

    /// Construct a secret key by filling `N` bytes from `rng`.
    ///
    /// The trait bound `R: CryptoRng` (rand_core 0.10) restricts the
    /// caller to a cryptographically-secure RNG. The result is an
    /// **opaque-bytes** SecretKey: it round-trips through encode/decode
    /// and ct_eq/zeroize, but does NOT decode to a valid SQIsign secret
    /// key under the spec-defined internal layout (which is still TBD
    /// pending KAT spec study). Use this constructor for test fixtures,
    /// fuzz-style coverage, and any path that needs a fresh SK-shaped
    /// byte container.
    ///
    /// Internally allocates an `[u8; N]` on the stack and calls
    /// `fill_bytes` (from the `rand_core::Rng` supertrait of
    /// `CryptoRng`) on it.
    pub fn random<R: CryptoRng>(rng: &mut R) -> Self {
        let mut bytes = [0u8; N];
        rng.fill_bytes(&mut bytes);
        Self { bytes }
    }

    /// Borrow the secret-key bytes.
    ///
    /// **Security caveat**: the returned slice exposes raw secret
    /// material. Callers MUST treat any copy made from this slice
    /// as confidential and zeroize it when finished. The borrow's
    /// lifetime keeps the secret in the SecretKey's owned storage
    /// (which is `ZeroizeOnDrop`), but a `.to_vec()` or
    /// `let copy = sk.as_bytes().to_vec()` creates an independent
    /// allocation NOT covered by the parent's drop-time wipe.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Encode `self` into `out`. Returns [`Error::BufferTooSmall`]
    /// if `out.len() < WIRE_BYTES`. Writes exactly `WIRE_BYTES`
    /// bytes on success.
    ///
    /// **Security caveat**: this method writes SECRET KEY bytes
    /// into the caller-provided output buffer. Callers should
    /// treat the output region as confidential and consider
    /// zeroizing it after use (e.g. via `out.zeroize()` once the
    /// caller's downstream work is complete).
    pub fn encode(&self, out: &mut [u8]) -> Result<()> {
        if out.len() < Self::WIRE_BYTES {
            return Err(Error::BufferTooSmall {
                required: Self::WIRE_BYTES,
                provided: out.len(),
            });
        }
        out[..N].copy_from_slice(&self.bytes);
        Ok(())
    }

    /// Decode a secret key from `bytes`. Returns
    /// [`Error::BufferTooSmall`] if `bytes.len() < WIRE_BYTES`.
    /// Reads exactly `WIRE_BYTES` bytes; extra bytes are ignored.
    ///
    /// Does NOT validate the internal layout — opaque byte
    /// container until spec-defined parsing lands.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::WIRE_BYTES {
            return Err(Error::BufferTooSmall {
                required: Self::WIRE_BYTES,
                provided: bytes.len(),
            });
        }
        let mut buf = [0u8; N];
        buf.copy_from_slice(&bytes[..N]);
        Ok(Self { bytes: buf })
    }
}

impl<const N: usize> ConstantTimeEq for SecretKey<N> {
    /// Constant-time byte-equality on the secret-key payload.
    ///
    /// The default `PartialEq` derived above is variable-time —
    /// it short-circuits at the first differing byte, which leaks
    /// the position of the first difference via timing. This impl
    /// uses the `subtle` crate's slice `ct_eq` to compare all
    /// `N` bytes in constant time regardless of where (or whether)
    /// they differ.
    ///
    /// Prefer this method over `==` in any path where the
    /// comparison's input may be attacker-influenced.
    fn ct_eq(&self, other: &Self) -> Choice {
        self.bytes[..].ct_eq(&other.bytes[..])
    }
}

/// SQIsign secret key at NIST Level 1 (353 bytes per
/// `params::lvl1::Level1::SK_BYTES`).
pub type SecretKeyLvl1 = SecretKey<353>;

/// SQIsign secret key at NIST Level 3 (529 bytes per
/// `params::lvl3::Level3::SK_BYTES`).
pub type SecretKeyLvl3 = SecretKey<529>;

/// SQIsign secret key at NIST Level 5 (701 bytes per
/// `params::lvl5::Level5::SK_BYTES`).
pub type SecretKeyLvl5 = SecretKey<701>;

/// SQIsign signature — opaque byte container of size `N` matching
/// the level's `Params::SIG_BYTES`.
///
/// As with [`SecretKey<N>`], the internal layout (challenge,
/// response, isogeny chain encoding) is opaque pending spec
/// study; this type provides the round-trip wire-format surface
/// only.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Signature<const N: usize> {
    /// Opaque signature bytes.
    pub bytes: [u8; N],
}

impl<const N: usize> Signature<N> {
    /// Wire-format size in bytes (equal to the const generic `N`).
    pub const WIRE_BYTES: usize = N;

    /// Construct a signature from its raw bytes.
    pub const fn new(bytes: [u8; N]) -> Self {
        Self { bytes }
    }

    /// Construct a signature by filling `N` bytes from `rng`.
    ///
    /// Test-fixture / opaque-bytes constructor — the result does NOT
    /// decode to a valid SQIsign signature under any signing key.
    /// Use [`SecretKey::random`] for the analogous SK constructor.
    pub fn random<R: CryptoRng>(rng: &mut R) -> Self {
        let mut bytes = [0u8; N];
        rng.fill_bytes(&mut bytes);
        Self { bytes }
    }

    /// Borrow the signature bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Encode `self` into `out`. Returns [`Error::BufferTooSmall`]
    /// if `out.len() < WIRE_BYTES`. Writes exactly `WIRE_BYTES`
    /// bytes on success.
    pub fn encode(&self, out: &mut [u8]) -> Result<()> {
        if out.len() < Self::WIRE_BYTES {
            return Err(Error::BufferTooSmall {
                required: Self::WIRE_BYTES,
                provided: out.len(),
            });
        }
        out[..N].copy_from_slice(&self.bytes);
        Ok(())
    }

    /// Decode a signature from `bytes`. Returns
    /// [`Error::BufferTooSmall`] if `bytes.len() < WIRE_BYTES`.
    /// Reads exactly `WIRE_BYTES` bytes; extra bytes are ignored.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::WIRE_BYTES {
            return Err(Error::BufferTooSmall {
                required: Self::WIRE_BYTES,
                provided: bytes.len(),
            });
        }
        let mut buf = [0u8; N];
        buf.copy_from_slice(&bytes[..N]);
        Ok(Self { bytes: buf })
    }
}

/// SQIsign signature at NIST Level 1 (148 bytes per
/// `params::lvl1::Level1::SIG_BYTES`).
pub type SignatureLvl1 = Signature<148>;

/// SQIsign signature at NIST Level 3 (224 bytes per
/// `params::lvl3::Level3::SIG_BYTES`).
pub type SignatureLvl3 = Signature<224>;

/// SQIsign signature at NIST Level 5 (292 bytes per
/// `params::lvl5::Level5::SIG_BYTES`).
pub type SignatureLvl5 = Signature<292>;

/// Bundled public + secret key pair for a single SQIsign instance.
///
/// Couples a typed [`PublicKey<F>`] with the matching opaque
/// [`SecretKey<SK_N>`]. The pair is treated as cryptographically
/// linked by convention — this type does NOT itself verify that
/// `pk` was actually derived from `sk` (that's the keypair
/// generation path's responsibility). Construct only via
/// [`Self::new`] from values produced by the keypair-generation
/// path (or, in test code, from independently-generated test
/// fixtures).
///
/// # Security
///
/// `KeyPair` derives [`ZeroizeOnDrop`] so the secret-key field
/// is wiped on drop. The `pk` field is annotated `#[zeroize(skip)]`
/// because public keys carry no secret. Explicit [`Zeroize::zeroize`]
/// is available via the SK field directly (`kp.sk.zeroize()`).
///
/// Use per-level type aliases [`KeyPairLvl1`], [`KeyPairLvl3`],
/// [`KeyPairLvl5`] rather than instantiating the generic form.
///
/// **`Debug` redacts the SK component** — the impl below emits the
/// public key fields normally and a `<redacted; SK_N bytes>`
/// placeholder for the SK, preventing accidental leakage of the
/// bundled secret through log/dbg/panic formatters. Inherits the
/// same redaction discipline as [`SecretKey<N>`].
#[derive(Clone, Eq, PartialEq, ZeroizeOnDrop)]
pub struct KeyPair<F: BaseField, const SK_N: usize> {
    /// Public key (typed Fp2 element + reserved byte).
    #[zeroize(skip)]
    pub pk: PublicKey<F>,
    /// Secret key (opaque bytes; wiped on drop).
    pub sk: SecretKey<SK_N>,
}

impl<F: BaseField, const SK_N: usize> core::fmt::Debug for KeyPair<F, SK_N> {
    /// Redacted formatter — emits the PublicKey component fully
    /// but suppresses the SecretKey byte values. See [`SecretKey`]'s
    /// `Debug` impl for the equivalent redaction on its own type.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("KeyPair")
            .field("pk", &self.pk)
            .field("sk", &format_args!("<redacted; {} bytes>", SK_N))
            .finish()
    }
}

impl<F: BaseField, const SK_N: usize> KeyPair<F, SK_N> {
    /// Combined wire-format size in bytes: `PublicKey::WIRE_BYTES
    /// + SecretKey::WIRE_BYTES`. Layout in the buffer is
    /// `pk_bytes || sk_bytes`.
    pub const WIRE_BYTES: usize = PublicKey::<F>::WIRE_BYTES + SecretKey::<SK_N>::WIRE_BYTES;

    /// Bundle an existing public key and secret key.
    pub const fn new(pk: PublicKey<F>, sk: SecretKey<SK_N>) -> Self {
        Self { pk, sk }
    }

    /// Borrow the public-key component.
    pub fn public_key(&self) -> &PublicKey<F> {
        &self.pk
    }

    /// Borrow the secret-key component.
    pub fn secret_key(&self) -> &SecretKey<SK_N> {
        &self.sk
    }

    /// Encode the keypair as `pk_bytes || sk_bytes` into `out`.
    ///
    /// Returns [`Error::BufferTooSmall`] if `out.len() < WIRE_BYTES`.
    /// Writes exactly `WIRE_BYTES` bytes on success.
    ///
    /// **Caller caution**: combined encoding writes the SECRET KEY
    /// into the output buffer. Callers should treat the output as
    /// confidential and consider zeroizing it after use.
    pub fn to_bytes_combined(&self, out: &mut [u8]) -> Result<()> {
        if out.len() < Self::WIRE_BYTES {
            return Err(Error::BufferTooSmall {
                required: Self::WIRE_BYTES,
                provided: out.len(),
            });
        }
        let pk_len = PublicKey::<F>::WIRE_BYTES;
        self.pk.encode(&mut out[..pk_len])?;
        self.sk.encode(&mut out[pk_len..Self::WIRE_BYTES])?;
        Ok(())
    }

    /// Decode a keypair from `pk_bytes || sk_bytes` in `bytes`.
    ///
    /// Returns [`Error::BufferTooSmall`] if input is too short;
    /// [`Error::InvalidPublicKey`] if the PK component is
    /// non-canonical.
    pub fn from_bytes_combined(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::WIRE_BYTES {
            return Err(Error::BufferTooSmall {
                required: Self::WIRE_BYTES,
                provided: bytes.len(),
            });
        }
        let pk_len = PublicKey::<F>::WIRE_BYTES;
        let pk = PublicKey::<F>::decode(&bytes[..pk_len])?;
        let sk = SecretKey::<SK_N>::decode(&bytes[pk_len..Self::WIRE_BYTES])?;
        Ok(Self { pk, sk })
    }
}

/// KeyPair bundle at NIST Level 1.
pub type KeyPairLvl1 = KeyPair<crate::gf::fp::Fp1Element, 353>;

/// KeyPair bundle at NIST Level 3.
pub type KeyPairLvl3 = KeyPair<crate::gf::fp::Fp3Element, 529>;

/// KeyPair bundle at NIST Level 5.
pub type KeyPairLvl5 = KeyPair<crate::gf::fp::Fp5Element, 701>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf::fp::Fp1Element;
    use std::format;

    /// Generic helper: encode + decode round-trips identical to the
    /// original public key.
    fn check_pk_round_trip<F: BaseField>(reserved: u8) {
        let a = Fp2::<F>::new(F::one().double(), F::one()); // (2, 1) in (re, im)
        let pk = PublicKey::<F>::new(a, reserved);
        let mut buf = [0u8; 129]; // big enough for L5's 129-byte encoding
        let n = 2 * F::ENCODED_BYTES + 1;
        pk.encode(&mut buf[..n]).expect("encode must succeed");
        let pk2 = PublicKey::<F>::decode(&buf[..n]).expect("decode must succeed");
        assert_eq!(pk, pk2, "S94: PublicKey round-trip must preserve");
    }

    #[test]
    fn pk_round_trip_at_lvl1_reserved_zero() {
        check_pk_round_trip::<Fp1Element>(0);
    }

    #[test]
    fn pk_round_trip_at_lvl1_reserved_nonzero() {
        // Reserved byte is preserved opaquely — non-zero values round-trip.
        check_pk_round_trip::<Fp1Element>(0xAB);
    }

    #[test]
    fn pk_round_trip_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_pk_round_trip::<Fp3Element>(0);
        check_pk_round_trip::<Fp3Element>(0xCD);
    }

    #[test]
    fn pk_round_trip_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_pk_round_trip::<Fp5Element>(0);
        check_pk_round_trip::<Fp5Element>(0xEF);
    }

    #[test]
    fn pk_encode_rejects_undersized_buffer() {
        let pk = PublicKey::<Fp1Element>::new(Fp2::<Fp1Element>::one(), 0);
        let mut buf = [0u8; 1];
        let r = pk.encode(&mut buf);
        assert_eq!(
            r,
            Err(Error::BufferTooSmall {
                required: PublicKey::<Fp1Element>::WIRE_BYTES,
                provided: 1,
            }),
            "S94: encode must reject undersized buffer",
        );
    }

    #[test]
    fn pk_decode_rejects_undersized_buffer() {
        let bytes = [0u8; 1];
        let r = PublicKey::<Fp1Element>::decode(&bytes);
        assert_eq!(
            r,
            Err(Error::BufferTooSmall {
                required: PublicKey::<Fp1Element>::WIRE_BYTES,
                provided: 1,
            }),
            "S94: decode must reject undersized buffer",
        );
    }

    #[test]
    fn pk_wire_bytes_matches_spec_at_all_levels() {
        // Verify WIRE_BYTES matches Params::PK_BYTES at every level.
        use crate::params::lvl3::Fp3Element;
        use crate::params::lvl5::Fp5Element;
        use crate::params::{Level1, Level3, Level5, Params};
        assert_eq!(
            PublicKey::<Fp1Element>::WIRE_BYTES,
            Level1::PK_BYTES,
            "S94: PublicKey<Fp1Element>::WIRE_BYTES must equal Level1::PK_BYTES",
        );
        assert_eq!(
            PublicKey::<Fp3Element>::WIRE_BYTES,
            Level3::PK_BYTES,
            "S94: PublicKey<Fp3Element>::WIRE_BYTES must equal Level3::PK_BYTES",
        );
        assert_eq!(
            PublicKey::<Fp5Element>::WIRE_BYTES,
            Level5::PK_BYTES,
            "S94: PublicKey<Fp5Element>::WIRE_BYTES must equal Level5::PK_BYTES",
        );
    }

    // ── S115 — SecretKey / Signature wire-format types ──

    #[test]
    fn sk_round_trip_at_lvl1() {
        let mut bytes = [0u8; 353];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i.to_le_bytes()[0].wrapping_mul(3).wrapping_add(7);
        }
        let sk = SecretKeyLvl1::new(bytes);
        let mut out = [0u8; 353];
        sk.encode(&mut out).expect("S115: encode must succeed");
        let sk2 = SecretKeyLvl1::decode(&out).expect("S115: decode must succeed");
        assert_eq!(sk, sk2, "S115: SecretKeyLvl1 round-trip must preserve");
        assert_eq!(
            sk.as_bytes(),
            &bytes,
            "S115: as_bytes must match construction"
        );
    }

    #[test]
    fn sk_round_trip_at_lvl3() {
        let mut bytes = [0u8; 529];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i.to_le_bytes()[0].wrapping_mul(5).wrapping_add(11);
        }
        let sk = SecretKeyLvl3::new(bytes);
        let mut out = [0u8; 529];
        sk.encode(&mut out).expect("S115: encode must succeed");
        let sk2 = SecretKeyLvl3::decode(&out).expect("S115: decode must succeed");
        assert_eq!(sk, sk2, "S115: SecretKeyLvl3 round-trip must preserve");
    }

    #[test]
    fn sk_round_trip_at_lvl5() {
        let mut bytes = [0u8; 701];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i.to_le_bytes()[0].wrapping_mul(7).wrapping_add(13);
        }
        let sk = SecretKeyLvl5::new(bytes);
        let mut out = [0u8; 701];
        sk.encode(&mut out).expect("S115: encode must succeed");
        let sk2 = SecretKeyLvl5::decode(&out).expect("S115: decode must succeed");
        assert_eq!(sk, sk2, "S115: SecretKeyLvl5 round-trip must preserve");
    }

    #[test]
    fn sk_encode_rejects_undersized_buffer() {
        let sk = SecretKeyLvl1::new([0u8; 353]);
        let mut tiny = [0u8; 1];
        let r = sk.encode(&mut tiny);
        assert_eq!(
            r,
            Err(Error::BufferTooSmall {
                required: SecretKeyLvl1::WIRE_BYTES,
                provided: 1,
            }),
            "S115: encode must reject undersized buffer",
        );
    }

    #[test]
    fn sk_decode_rejects_undersized_buffer() {
        let tiny = [0u8; 1];
        let r = SecretKeyLvl1::decode(&tiny);
        assert_eq!(
            r,
            Err(Error::BufferTooSmall {
                required: SecretKeyLvl1::WIRE_BYTES,
                provided: 1,
            }),
            "S115: decode must reject undersized buffer",
        );
    }

    #[test]
    fn sk_wire_bytes_matches_spec_at_all_levels() {
        use crate::params::{Level1, Level3, Level5, Params};
        assert_eq!(
            SecretKeyLvl1::WIRE_BYTES,
            Level1::SK_BYTES,
            "S115: SecretKeyLvl1::WIRE_BYTES must equal Level1::SK_BYTES",
        );
        assert_eq!(
            SecretKeyLvl3::WIRE_BYTES,
            Level3::SK_BYTES,
            "S115: SecretKeyLvl3::WIRE_BYTES must equal Level3::SK_BYTES",
        );
        assert_eq!(
            SecretKeyLvl5::WIRE_BYTES,
            Level5::SK_BYTES,
            "S115: SecretKeyLvl5::WIRE_BYTES must equal Level5::SK_BYTES",
        );
    }

    #[test]
    fn sig_round_trip_at_lvl1() {
        let mut bytes = [0u8; 148];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i.to_le_bytes()[0].wrapping_mul(2).wrapping_add(5);
        }
        let sig = SignatureLvl1::new(bytes);
        let mut out = [0u8; 148];
        sig.encode(&mut out).expect("S115: encode must succeed");
        let sig2 = SignatureLvl1::decode(&out).expect("S115: decode must succeed");
        assert_eq!(sig, sig2, "S115: SignatureLvl1 round-trip must preserve");
        assert_eq!(
            sig.as_bytes(),
            &bytes,
            "S115: as_bytes must match construction"
        );
    }

    #[test]
    fn sig_round_trip_at_lvl3() {
        let mut bytes = [0u8; 224];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i.to_le_bytes()[0].wrapping_mul(4).wrapping_add(9);
        }
        let sig = SignatureLvl3::new(bytes);
        let mut out = [0u8; 224];
        sig.encode(&mut out).expect("S115: encode must succeed");
        let sig2 = SignatureLvl3::decode(&out).expect("S115: decode must succeed");
        assert_eq!(sig, sig2, "S115: SignatureLvl3 round-trip must preserve");
    }

    #[test]
    fn sig_round_trip_at_lvl5() {
        let mut bytes = [0u8; 292];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i.to_le_bytes()[0].wrapping_mul(6).wrapping_add(15);
        }
        let sig = SignatureLvl5::new(bytes);
        let mut out = [0u8; 292];
        sig.encode(&mut out).expect("S115: encode must succeed");
        let sig2 = SignatureLvl5::decode(&out).expect("S115: decode must succeed");
        assert_eq!(sig, sig2, "S115: SignatureLvl5 round-trip must preserve");
    }

    #[test]
    fn sig_encode_rejects_undersized_buffer() {
        let sig = SignatureLvl1::new([0u8; 148]);
        let mut tiny = [0u8; 1];
        let r = sig.encode(&mut tiny);
        assert_eq!(
            r,
            Err(Error::BufferTooSmall {
                required: SignatureLvl1::WIRE_BYTES,
                provided: 1,
            }),
            "S115: encode must reject undersized buffer",
        );
    }

    #[test]
    fn sig_decode_rejects_undersized_buffer() {
        let tiny = [0u8; 1];
        let r = SignatureLvl1::decode(&tiny);
        assert_eq!(
            r,
            Err(Error::BufferTooSmall {
                required: SignatureLvl1::WIRE_BYTES,
                provided: 1,
            }),
            "S115: decode must reject undersized buffer",
        );
    }

    #[test]
    fn sig_wire_bytes_matches_spec_at_all_levels() {
        use crate::params::{Level1, Level3, Level5, Params};
        assert_eq!(
            SignatureLvl1::WIRE_BYTES,
            Level1::SIG_BYTES,
            "S115: SignatureLvl1::WIRE_BYTES must equal Level1::SIG_BYTES",
        );
        assert_eq!(
            SignatureLvl3::WIRE_BYTES,
            Level3::SIG_BYTES,
            "S115: SignatureLvl3::WIRE_BYTES must equal Level3::SIG_BYTES",
        );
        assert_eq!(
            SignatureLvl5::WIRE_BYTES,
            Level5::SIG_BYTES,
            "S115: SignatureLvl5::WIRE_BYTES must equal Level5::SIG_BYTES",
        );
    }

    // ── S116 — Zeroize + ZeroizeOnDrop for SecretKey<N> ──

    #[test]
    fn sk_zeroize_explicit_wipes_bytes_at_lvl1() {
        use zeroize::Zeroize;
        let mut bytes = [0u8; 353];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i.to_le_bytes()[0].wrapping_mul(11).wrapping_add(3);
        }
        // Verify the constructed bytes are non-zero (otherwise this test
        // would be vacuous).
        assert!(
            bytes.iter().any(|&b| b != 0),
            "S116: setup error — pseudo-random bytes are all zero",
        );
        let mut sk = SecretKeyLvl1::new(bytes);
        sk.zeroize();
        assert!(
            sk.bytes.iter().all(|&b| b == 0),
            "S116: zeroize must wipe all secret-key bytes",
        );
    }

    #[test]
    fn sk_zeroize_explicit_wipes_bytes_at_lvl3() {
        use zeroize::Zeroize;
        let mut bytes = [0u8; 529];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i.to_le_bytes()[0].wrapping_mul(13).wrapping_add(5);
        }
        let mut sk = SecretKeyLvl3::new(bytes);
        sk.zeroize();
        assert!(
            sk.bytes.iter().all(|&b| b == 0),
            "S116: zeroize must wipe all secret-key bytes at lvl3",
        );
    }

    #[test]
    fn sk_zeroize_explicit_wipes_bytes_at_lvl5() {
        use zeroize::Zeroize;
        let mut bytes = [0u8; 701];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i.to_le_bytes()[0].wrapping_mul(17).wrapping_add(7);
        }
        let mut sk = SecretKeyLvl5::new(bytes);
        sk.zeroize();
        assert!(
            sk.bytes.iter().all(|&b| b == 0),
            "S116: zeroize must wipe all secret-key bytes at lvl5",
        );
    }

    #[test]
    fn sk_clone_is_independent_for_zeroize() {
        use zeroize::Zeroize;
        let mut bytes = [0u8; 353];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i.to_le_bytes()[0].wrapping_mul(19).wrapping_add(11);
        }
        let original_bytes = bytes;
        let sk_a = SecretKeyLvl1::new(bytes);
        let mut sk_b = sk_a.clone();
        sk_b.zeroize();
        // Original is unaffected.
        assert_eq!(
            sk_a.bytes, original_bytes,
            "S116: zeroizing a clone must not affect the original",
        );
        // Clone is wiped.
        assert!(
            sk_b.bytes.iter().all(|&b| b == 0),
            "S116: the cloned-then-zeroized SK must be all-zero",
        );
    }

    #[test]
    fn sk_zeroize_on_drop_compiles() {
        // The fact that SecretKey<N> implements ZeroizeOnDrop is enforced
        // by a trait-bound assertion. We can't OBSERVE the drop-time wipe
        // (the value is gone after drop), but we can confirm the trait is
        // implemented — if it weren't, this would fail to compile.
        fn assert_zeroize_on_drop<T: ZeroizeOnDrop>() {}
        assert_zeroize_on_drop::<SecretKeyLvl1>();
        assert_zeroize_on_drop::<SecretKeyLvl3>();
        assert_zeroize_on_drop::<SecretKeyLvl5>();
    }

    // ── S117 — ConstantTimeEq for SecretKey<N> ──

    #[test]
    fn sk_ct_eq_reflexive_at_lvl1() {
        let mut bytes = [0u8; 353];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i.to_le_bytes()[0].wrapping_mul(23).wrapping_add(2);
        }
        let sk = SecretKeyLvl1::new(bytes);
        assert!(
            bool::from(sk.ct_eq(&sk)),
            "S117: ct_eq must be reflexive on SecretKey",
        );
    }

    #[test]
    fn sk_ct_eq_distinguishes_at_lvl1() {
        let mut bytes_a = [0u8; 353];
        let mut bytes_b = [0u8; 353];
        for (i, b) in bytes_a.iter_mut().enumerate() {
            *b = i.to_le_bytes()[0].wrapping_mul(29).wrapping_add(4);
        }
        for (i, b) in bytes_b.iter_mut().enumerate() {
            *b = i.to_le_bytes()[0].wrapping_mul(31).wrapping_add(6);
        }
        let sk_a = SecretKeyLvl1::new(bytes_a);
        let sk_b = SecretKeyLvl1::new(bytes_b);
        assert!(
            !bool::from(sk_a.ct_eq(&sk_b)),
            "S117: ct_eq must distinguish independent SecretKeys",
        );
    }

    #[test]
    fn sk_ct_eq_detects_single_byte_difference() {
        let mut bytes_a = [0u8; 353];
        for (i, b) in bytes_a.iter_mut().enumerate() {
            *b = i.to_le_bytes()[0].wrapping_mul(37).wrapping_add(8);
        }
        let mut bytes_b = bytes_a;
        // Perturb a single byte at the END of the buffer — a variable-time
        // implementation would short-circuit-pass on prefix equality and
        // only detect the difference at the last position. Our ct_eq must
        // detect it.
        bytes_b[352] = bytes_b[352].wrapping_add(1);
        let sk_a = SecretKeyLvl1::new(bytes_a);
        let sk_b = SecretKeyLvl1::new(bytes_b);
        assert!(
            !bool::from(sk_a.ct_eq(&sk_b)),
            "S117: ct_eq must detect a single-byte difference at the end of the buffer",
        );
    }

    #[test]
    fn sk_ct_eq_reflexive_at_lvl3() {
        let mut bytes = [0u8; 529];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i.to_le_bytes()[0].wrapping_mul(41).wrapping_add(10);
        }
        let sk = SecretKeyLvl3::new(bytes);
        assert!(
            bool::from(sk.ct_eq(&sk)),
            "S117: ct_eq must be reflexive on SecretKey at lvl3",
        );
    }

    #[test]
    fn sk_ct_eq_reflexive_at_lvl5() {
        let mut bytes = [0u8; 701];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i.to_le_bytes()[0].wrapping_mul(43).wrapping_add(12);
        }
        let sk = SecretKeyLvl5::new(bytes);
        assert!(
            bool::from(sk.ct_eq(&sk)),
            "S117: ct_eq must be reflexive on SecretKey at lvl5",
        );
    }

    // ── S118 — random<R: CryptoRng> ctor for SecretKey<N> and Signature<N> ──

    fn make_rng(domain: &[u8]) -> crate::hash::Shake256Rng {
        let mut h = crate::hash::Shake256::new();
        h.absorb(domain);
        h.into_rng()
    }

    #[test]
    fn sk_random_at_lvl1_is_non_zero() {
        let mut rng = make_rng(b"S118-sk-rand-l1");
        let sk = SecretKeyLvl1::random(&mut rng);
        // With high probability, a CryptoRng-filled SK is not all-zero.
        assert!(
            sk.bytes.iter().any(|&b| b != 0),
            "S118: random SK must be non-zero",
        );
    }

    #[test]
    fn sk_random_at_lvl3_is_non_zero() {
        let mut rng = make_rng(b"S118-sk-rand-l3");
        let sk = SecretKeyLvl3::random(&mut rng);
        assert!(
            sk.bytes.iter().any(|&b| b != 0),
            "S118: random SK at lvl3 must be non-zero",
        );
    }

    #[test]
    fn sk_random_at_lvl5_is_non_zero() {
        let mut rng = make_rng(b"S118-sk-rand-l5");
        let sk = SecretKeyLvl5::random(&mut rng);
        assert!(
            sk.bytes.iter().any(|&b| b != 0),
            "S118: random SK at lvl5 must be non-zero",
        );
    }

    #[test]
    fn sk_random_round_trips() {
        let mut rng = make_rng(b"S118-sk-rand-roundtrip");
        let sk = SecretKeyLvl1::random(&mut rng);
        let mut out = [0u8; 353];
        sk.encode(&mut out)
            .expect("S118: encode random SK must succeed");
        let sk2 = SecretKeyLvl1::decode(&out).expect("S118: decode must succeed");
        assert_eq!(
            sk, sk2,
            "S118: random SK must round-trip through encode/decode",
        );
    }

    #[test]
    fn sk_random_distinct_seeds_produce_distinct_sks() {
        let mut rng_a = make_rng(b"S118-sk-rand-distinct-A");
        let mut rng_b = make_rng(b"S118-sk-rand-distinct-B");
        let sk_a = SecretKeyLvl1::random(&mut rng_a);
        let sk_b = SecretKeyLvl1::random(&mut rng_b);
        assert!(
            !bool::from(sk_a.ct_eq(&sk_b)),
            "S118: distinct seeds must produce distinct random SKs",
        );
    }

    #[test]
    fn sig_random_at_lvl1_is_non_zero() {
        let mut rng = make_rng(b"S118-sig-rand-l1");
        let sig = SignatureLvl1::random(&mut rng);
        assert!(
            sig.bytes.iter().any(|&b| b != 0),
            "S118: random signature must be non-zero",
        );
    }

    #[test]
    fn sig_random_round_trips() {
        let mut rng = make_rng(b"S118-sig-rand-roundtrip");
        let sig = SignatureLvl3::random(&mut rng);
        let mut out = [0u8; 224];
        sig.encode(&mut out)
            .expect("S118: encode random sig must succeed");
        let sig2 = SignatureLvl3::decode(&out).expect("S118: decode must succeed");
        assert_eq!(
            sig, sig2,
            "S118: random signature must round-trip through encode/decode",
        );
    }

    // ── S119 — KeyPair bundle type ──

    fn make_test_pk<F: BaseField>(reserved: u8) -> PublicKey<F> {
        let a = Fp2::<F>::new(F::one().double(), F::one()); // (2, 1) in (re, im)
        PublicKey::<F>::new(a, reserved)
    }

    #[test]
    fn keypair_wire_bytes_matches_components_at_lvl1() {
        assert_eq!(
            KeyPairLvl1::WIRE_BYTES,
            PublicKey::<Fp1Element>::WIRE_BYTES + SecretKeyLvl1::WIRE_BYTES,
            "S119: KeyPair::WIRE_BYTES must equal PK::WIRE_BYTES + SK::WIRE_BYTES",
        );
    }

    #[test]
    fn keypair_wire_bytes_matches_components_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        assert_eq!(
            KeyPairLvl3::WIRE_BYTES,
            PublicKey::<Fp3Element>::WIRE_BYTES + SecretKeyLvl3::WIRE_BYTES,
            "S119: KeyPairLvl3::WIRE_BYTES must equal PK::WIRE_BYTES + SK::WIRE_BYTES",
        );
    }

    #[test]
    fn keypair_wire_bytes_matches_components_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        assert_eq!(
            KeyPairLvl5::WIRE_BYTES,
            PublicKey::<Fp5Element>::WIRE_BYTES + SecretKeyLvl5::WIRE_BYTES,
            "S119: KeyPairLvl5::WIRE_BYTES must equal PK::WIRE_BYTES + SK::WIRE_BYTES",
        );
    }

    #[test]
    fn keypair_combined_round_trips_at_lvl1() {
        let pk = make_test_pk::<Fp1Element>(0xAB);
        let mut rng = make_rng(b"S119-kp-rt-l1");
        let sk = SecretKeyLvl1::random(&mut rng);
        let kp = KeyPairLvl1::new(pk, sk);
        let mut buf = [0u8; KeyPairLvl1::WIRE_BYTES];
        kp.to_bytes_combined(&mut buf)
            .expect("S119: encode_combined must succeed");
        let kp2 =
            KeyPairLvl1::from_bytes_combined(&buf).expect("S119: decode_combined must succeed");
        assert_eq!(
            kp, kp2,
            "S119: KeyPair combined round-trip must preserve at lvl1",
        );
    }

    #[test]
    fn keypair_combined_round_trips_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        let pk = make_test_pk::<Fp3Element>(0xCD);
        let mut rng = make_rng(b"S119-kp-rt-l3");
        let sk = SecretKeyLvl3::random(&mut rng);
        let kp = KeyPairLvl3::new(pk, sk);
        let mut buf = [0u8; KeyPairLvl3::WIRE_BYTES];
        kp.to_bytes_combined(&mut buf)
            .expect("S119: encode_combined must succeed at lvl3");
        let kp2 = KeyPairLvl3::from_bytes_combined(&buf)
            .expect("S119: decode_combined must succeed at lvl3");
        assert_eq!(
            kp, kp2,
            "S119: KeyPair combined round-trip must preserve at lvl3",
        );
    }

    #[test]
    fn keypair_combined_round_trips_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        let pk = make_test_pk::<Fp5Element>(0xEF);
        let mut rng = make_rng(b"S119-kp-rt-l5");
        let sk = SecretKeyLvl5::random(&mut rng);
        let kp = KeyPairLvl5::new(pk, sk);
        let mut buf = [0u8; KeyPairLvl5::WIRE_BYTES];
        kp.to_bytes_combined(&mut buf)
            .expect("S119: encode_combined must succeed at lvl5");
        let kp2 = KeyPairLvl5::from_bytes_combined(&buf)
            .expect("S119: decode_combined must succeed at lvl5");
        assert_eq!(
            kp, kp2,
            "S119: KeyPair combined round-trip must preserve at lvl5",
        );
    }

    #[test]
    fn keypair_to_bytes_combined_rejects_undersized() {
        let pk = make_test_pk::<Fp1Element>(0);
        let sk = SecretKeyLvl1::new([0u8; 353]);
        let kp = KeyPairLvl1::new(pk, sk);
        let mut tiny = [0u8; 1];
        let r = kp.to_bytes_combined(&mut tiny);
        assert_eq!(
            r,
            Err(Error::BufferTooSmall {
                required: KeyPairLvl1::WIRE_BYTES,
                provided: 1,
            }),
            "S119: to_bytes_combined must reject undersized buffer",
        );
    }

    #[test]
    fn keypair_from_bytes_combined_rejects_undersized() {
        let tiny = [0u8; 1];
        let r = KeyPairLvl1::from_bytes_combined(&tiny);
        assert_eq!(
            r,
            Err(Error::BufferTooSmall {
                required: KeyPairLvl1::WIRE_BYTES,
                provided: 1,
            }),
            "S119: from_bytes_combined must reject undersized buffer",
        );
    }

    #[test]
    fn keypair_zeroize_on_drop_compiles() {
        // KeyPair must derive ZeroizeOnDrop so the SK component is wiped
        // when the bundled value is dropped.
        fn assert_zeroize_on_drop<T: ZeroizeOnDrop>() {}
        assert_zeroize_on_drop::<KeyPairLvl1>();
        assert_zeroize_on_drop::<KeyPairLvl3>();
        assert_zeroize_on_drop::<KeyPairLvl5>();
    }

    #[test]
    fn keypair_accessors_borrow_components() {
        let pk = make_test_pk::<Fp1Element>(0x42);
        let mut rng = make_rng(b"S119-kp-acc");
        let sk = SecretKeyLvl1::random(&mut rng);
        let sk_copy_bytes = sk.bytes;
        let kp = KeyPairLvl1::new(pk, sk);
        assert_eq!(kp.public_key().reserved, 0x42, "S119: pk accessor");
        assert_eq!(
            kp.secret_key().as_bytes(),
            &sk_copy_bytes,
            "S119: sk accessor",
        );
    }

    // ── S120 — security hardening: redacted Debug impls ──

    #[test]
    fn sk_debug_redacts_byte_values_at_lvl1() {
        // `format!` is the std prelude macro; tests build with std.
        // Construct an SK with a distinctive byte pattern.
        let mut bytes = [0u8; 353];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i.to_le_bytes()[0].wrapping_mul(53).wrapping_add(17);
        }
        let sk = SecretKeyLvl1::new(bytes);
        let dbg = format!("{:?}", sk);
        // The redacted Debug must say "SecretKey" and "redacted".
        assert!(
            dbg.contains("SecretKey"),
            "S120: Debug must include the type name",
        );
        assert!(
            dbg.contains("redacted"),
            "S120: Debug must include the 'redacted' marker",
        );
        // Critically: the byte values must NOT appear in the formatted output.
        // Sample a few distinctive bytes from the SK and assert each is absent
        // from the formatted string.
        for &b in &[bytes[7], bytes[100], bytes[200], bytes[350]] {
            let byte_str = format!("{}", b);
            // The string might naturally contain the substring "0", "1", etc.
            // To avoid false positives, only check distinctive (>= 10) byte
            // values, and ensure they don't appear as standalone numbers.
            if b >= 10 {
                let pattern = format!(", {}", b); // matches "[a, b, c, ...]" style
                let pattern2 = format!("[{}", b); // matches "[NNN, ..." style
                let pattern3 = format!("{},", b); // matches "..., NNN," style
                assert!(
                    !dbg.contains(&pattern) && !dbg.contains(&pattern2) && !dbg.contains(&pattern3),
                    "S120: Debug output must NOT contain SK byte value {} (formatted byte_str: {byte_str:?})",
                    b,
                );
            }
        }
    }

    #[test]
    fn sk_debug_announces_size_at_each_level() {
        // `format!` is the std prelude macro; tests build with std.
        let sk1 = SecretKeyLvl1::new([0u8; 353]);
        let sk3 = SecretKeyLvl3::new([0u8; 529]);
        let sk5 = SecretKeyLvl5::new([0u8; 701]);
        assert!(
            format!("{:?}", sk1).contains("353 bytes"),
            "S120: Debug must show byte count at lvl1",
        );
        assert!(
            format!("{:?}", sk3).contains("529 bytes"),
            "S120: Debug must show byte count at lvl3",
        );
        assert!(
            format!("{:?}", sk5).contains("701 bytes"),
            "S120: Debug must show byte count at lvl5",
        );
    }

    #[test]
    fn keypair_debug_redacts_sk_at_lvl1() {
        // `format!` is the std prelude macro; tests build with std.
        let pk = make_test_pk::<Fp1Element>(0x99);
        let mut bytes = [0u8; 353];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i.to_le_bytes()[0].wrapping_mul(59).wrapping_add(19);
        }
        let sk = SecretKeyLvl1::new(bytes);
        let kp = KeyPairLvl1::new(pk, sk);
        let dbg = format!("{:?}", kp);
        // Must include the type name and the SK redaction marker.
        assert!(
            dbg.contains("KeyPair"),
            "S120: KeyPair Debug must include the type name",
        );
        assert!(
            dbg.contains("redacted"),
            "S120: KeyPair Debug must include the SK redaction marker",
        );
        assert!(
            dbg.contains("353 bytes"),
            "S120: KeyPair Debug must include the SK byte count",
        );
        // SK bytes must not leak into the formatted output (same probe as
        // the SK-only test).
        for &b in &[bytes[7], bytes[100], bytes[200], bytes[350]] {
            if b >= 10 {
                let pattern = format!(", {}", b);
                let pattern2 = format!("[{}", b);
                let pattern3 = format!("{},", b);
                assert!(
                    !dbg.contains(&pattern) && !dbg.contains(&pattern2) && !dbg.contains(&pattern3),
                    "S120: KeyPair Debug must NOT contain SK byte value {}",
                    b,
                );
            }
        }
    }
}
