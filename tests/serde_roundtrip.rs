// SPDX-License-Identifier: MIT OR Apache-2.0
//! serde round-trip tests for the wire types (`PublicKey`, `SecretKey`,
//! `Signature`) across five formats: CBOR, postcard, JSON, TOML, and YAML.
//!
//! The serde impls use `serdect`, so human-readable formats (JSON, TOML, YAML)
//! encode the payload as a lowercase hex string while binary formats (CBOR,
//! postcard) keep the raw bytes. Both directions must reproduce the original
//! value byte-for-byte. Gated on the `serde` feature (see `Cargo.toml`'s
//! `[[test]]` entry).

use core::fmt::Debug;

use serde::Serialize;
use serde::de::DeserializeOwned;

use pq_sqisign::gf::fp::Fp1Element;
use pq_sqisign::gf::fp2::Fp2;
use pq_sqisign::wire::{PublicKey, SecretKeyLvl1, SignatureLvl1};

/// TOML's root document must be a table, so a bare hex string cannot be a
/// top-level TOML value. Wrap the payload in a single-field struct for the
/// TOML leg; every other format serializes the value directly.
#[derive(Serialize, serde::Deserialize, Clone, PartialEq, Debug)]
struct Wrap<T> {
    value: T,
}

/// Round-trip `value` through all five formats and assert each reproduces it.
fn check_roundtrips<T>(value: &T)
where
    T: Serialize + DeserializeOwned + Clone + PartialEq + Debug,
{
    // JSON (human-readable → hex).
    let j = serde_json::to_string(value).expect("json serialize");
    let back: T = serde_json::from_str(&j).expect("json deserialize");
    assert_eq!(value, &back, "json round-trip");

    // YAML (human-readable → hex).
    let y = serde_yaml::to_string(value).expect("yaml serialize");
    let back: T = serde_yaml::from_str(&y).expect("yaml deserialize");
    assert_eq!(value, &back, "yaml round-trip");

    // TOML (human-readable → hex; needs a table root, hence Wrap).
    let wrapped = Wrap {
        value: value.clone(),
    };
    let t = toml::to_string(&wrapped).expect("toml serialize");
    let back: Wrap<T> = toml::from_str(&t).expect("toml deserialize");
    assert_eq!(value, &back.value, "toml round-trip");

    // CBOR (binary → raw bytes).
    let mut cbor = Vec::new();
    ciborium::into_writer(value, &mut cbor).expect("cbor serialize");
    let back: T = ciborium::from_reader(cbor.as_slice()).expect("cbor deserialize");
    assert_eq!(value, &back, "cbor round-trip");

    // postcard (binary → raw bytes).
    let pc = postcard::to_allocvec(value).expect("postcard serialize");
    let back: T = postcard::from_bytes(&pc).expect("postcard deserialize");
    assert_eq!(value, &back, "postcard round-trip");
}

fn sample_pk() -> PublicKey<Fp1Element> {
    // `one()` is a canonical Fp2 element; the reserved byte adds variation.
    PublicKey::<Fp1Element>::new(Fp2::<Fp1Element>::one(), 0x2a)
}

fn sample_sk() -> SecretKeyLvl1 {
    let mut bytes = [0u8; 353];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = i.to_le_bytes()[0].wrapping_mul(7).wrapping_add(13);
    }
    SecretKeyLvl1::new(bytes)
}

fn sample_sig() -> SignatureLvl1 {
    let mut bytes = [0u8; 148];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = i.to_le_bytes()[0].wrapping_mul(3).wrapping_add(5);
    }
    SignatureLvl1::new(bytes)
}

#[test]
fn public_key_roundtrips_all_formats() {
    check_roundtrips(&sample_pk());
}

#[test]
fn secret_key_roundtrips_all_formats() {
    check_roundtrips(&sample_sk());
}

#[test]
fn signature_roundtrips_all_formats() {
    check_roundtrips(&sample_sig());
}

#[test]
fn human_readable_is_hex_and_binary_is_raw() {
    let sig = sample_sig();

    // JSON: a single lowercase-hex string covering every wire byte.
    let j = serde_json::to_string(&sig).expect("json serialize");
    assert!(
        j.starts_with('"') && j.ends_with('"'),
        "human-readable value must be a JSON string, got {j}",
    );
    let inner = j.trim_matches('"');
    assert!(
        inner
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "human-readable encoding must be lowercase hex",
    );
    assert_eq!(
        inner.len(),
        2 * SignatureLvl1::WIRE_BYTES,
        "hex string must cover all wire bytes",
    );

    // postcard: raw bytes (length-prefixed), NOT hex-expanded.
    let pc = postcard::to_allocvec(&sig).expect("postcard serialize");
    assert!(
        pc.len() <= SignatureLvl1::WIRE_BYTES + 3,
        "binary format must stay raw bytes (len {} for {} wire bytes)",
        pc.len(),
        SignatureLvl1::WIRE_BYTES,
    );

    // The decoded hex must equal the raw wire bytes.
    let from_hex = hex::decode(inner).expect("hex decodes");
    assert_eq!(
        from_hex,
        sig.as_bytes(),
        "hex payload must equal the raw wire bytes",
    );
}

#[test]
fn signature_from_and_to_vec_is_wire_compliant() {
    let mut raw = [0u8; 148];
    for (i, b) in raw.iter_mut().enumerate() {
        *b = i.to_le_bytes()[0].wrapping_mul(11).wrapping_add(1);
    }
    let want = raw.to_vec();

    // From<&[u8]> is the canonical conversion; to_vec yields the same bytes.
    let sig = SignatureLvl1::from(&raw[..]);
    assert_eq!(sig.to_vec(), want, "From<&[u8]> + to_vec is wire-identical");

    // Vec<u8>, &Vec<u8>, Box<[u8]> all delegate to From<&[u8]>.
    let v = want.clone();
    assert_eq!(
        SignatureLvl1::from(v.clone()).to_vec(),
        want,
        "From<Vec<u8>>"
    );
    assert_eq!(SignatureLvl1::from(&v).to_vec(), want, "From<&Vec<u8>>");
    let boxed: Box<[u8]> = v.into_boxed_slice();
    assert_eq!(SignatureLvl1::from(boxed).to_vec(), want, "From<Box<[u8]>>");
}

#[test]
fn secret_key_from_and_to_vec_is_wire_compliant() {
    let mut raw = [0u8; 353];
    for (i, b) in raw.iter_mut().enumerate() {
        *b = i.to_le_bytes()[0].wrapping_mul(5).wrapping_add(3);
    }
    let want = raw.to_vec();

    let sk = SecretKeyLvl1::from(&raw[..]);
    assert_eq!(sk.to_vec(), want, "From<&[u8]> + to_vec is wire-identical");

    let v = want.clone();
    assert_eq!(
        SecretKeyLvl1::from(v.clone()).to_vec(),
        want,
        "From<Vec<u8>>"
    );
    assert_eq!(SecretKeyLvl1::from(&v).to_vec(), want, "From<&Vec<u8>>");
    let boxed: Box<[u8]> = v.into_boxed_slice();
    assert_eq!(SecretKeyLvl1::from(boxed).to_vec(), want, "From<Box<[u8]>>");
}

#[test]
fn public_key_from_and_to_vec_is_wire_compliant() {
    let pk = sample_pk();
    let bytes = pk.to_vec();
    assert_eq!(
        bytes.len(),
        PublicKey::<Fp1Element>::WIRE_BYTES,
        "to_vec yields exactly WIRE_BYTES",
    );

    // Valid wire bytes round-trip exactly through every From variant.
    assert_eq!(
        PublicKey::<Fp1Element>::from(bytes.as_slice()),
        pk,
        "From<&[u8]>"
    );
    assert_eq!(
        PublicKey::<Fp1Element>::from(bytes.clone()),
        pk,
        "From<Vec<u8>>"
    );
    assert_eq!(PublicKey::<Fp1Element>::from(&bytes), pk, "From<&Vec<u8>>");
    let boxed: Box<[u8]> = bytes.into_boxed_slice();
    assert_eq!(PublicKey::<Fp1Element>::from(boxed), pk, "From<Box<[u8]>>");
}

#[test]
fn from_short_slice_zero_pads_opaque_types() {
    // Length-tolerant: a short input zero-pads to N for the opaque containers.
    let sig = SignatureLvl1::from(&[1u8, 2, 3][..]);
    let mut want = vec![0u8; 148];
    want[..3].copy_from_slice(&[1, 2, 3]);
    assert_eq!(sig.to_vec(), want, "short input zero-pads to WIRE_BYTES");
}
