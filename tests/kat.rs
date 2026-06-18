// SPDX-License-Identifier: MIT OR Apache-2.0
//! Known-Answer-Test harness for SQIsign Level-1, Level-3, Level-5.
//!
//! KAT response files come straight from the upstream reference repo at
//! <https://github.com/SQISign/the-sqisign> under `KAT/`. The format is the
//! NIST PQC convention: one record per test, each record an 8-line block
//! (`count = N`, `seed = …`, `mlen = …`, `msg = …`, `pk = …`, `sk = …`,
//! `smlen = …`, `sm = …`) separated by a blank line and led by a comment
//! header naming the parameter set.
//!
//! The lightweight tests validate the *parse* — every record decodes cleanly
//! and the per-level byte sizes match the `Params::PK_BYTES` / `SK_BYTES` /
//! `SIG_BYTES` constants in [`pq_sqisign::params`]. The heavier verdict tests
//! are marked `#[ignore]` (they take tens of seconds) and run on demand:
//! [`kat_lvl1_verify_only`] verifies all 100 C-generated KAT signatures.
//! Byte-exact reproduction of the KAT *signatures* by our signer additionally
//! requires aligning our DRBG to the NIST seed (see
//! [`kat_lvl1_signs_and_verifies`]).

use pq_sqisign::params::{Level1, Level3, Level5, Params};

#[derive(Debug, Clone)]
struct KatRecord {
    count: usize,
    seed: Vec<u8>,
    msg: Vec<u8>,
    pk: Vec<u8>,
    sk: Vec<u8>,
    sm: Vec<u8>,
}

fn decode_hex(s: &str) -> Vec<u8> {
    hex::decode(s.trim()).expect("KAT hex value parses")
}

fn parse_kat(text: &str) -> Vec<KatRecord> {
    let mut out = Vec::new();
    let mut cur: Option<KatRecord> = None;
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            if let Some(r) = cur.take() {
                out.push(r);
            }
            continue;
        }
        let (k, v) = match line.split_once('=') {
            Some(kv) => kv,
            None => continue,
        };
        let (k, v) = (k.trim(), v.trim());
        let rec = cur.get_or_insert(KatRecord {
            count: 0,
            seed: vec![],
            msg: vec![],
            pk: vec![],
            sk: vec![],
            sm: vec![],
        });
        match k {
            "count" => rec.count = v.parse::<usize>().expect("count parses"),
            "seed" => rec.seed = decode_hex(v),
            "mlen" => {
                let _: usize = v.parse().expect("mlen parses");
            }
            "msg" => rec.msg = decode_hex(v),
            "pk" => rec.pk = decode_hex(v),
            "sk" => rec.sk = decode_hex(v),
            "smlen" => {
                let _: usize = v.parse().expect("smlen parses");
            }
            "sm" => rec.sm = decode_hex(v),
            _ => {}
        }
    }
    if let Some(r) = cur.take() {
        out.push(r);
    }
    out
}

const KAT_LVL1: &str = include_str!("KAT/PQCsignKAT_353_SQIsign_lvl1.rsp");
const KAT_LVL3: &str = include_str!("KAT/PQCsignKAT_529_SQIsign_lvl3.rsp");
const KAT_LVL5: &str = include_str!("KAT/PQCsignKAT_701_SQIsign_lvl5.rsp");

#[test]
fn kat_lvl1_parses() {
    let records = parse_kat(KAT_LVL1);
    assert_eq!(records.len(), 100, "100 KAT records expected at Level-1");
    for r in &records {
        assert_eq!(r.seed.len(), 48, "NIST PQC seed is always 48 bytes");
        assert_eq!(r.pk.len(), Level1::PK_BYTES);
        assert_eq!(r.sk.len(), Level1::SK_BYTES);
        assert_eq!(r.sm.len(), r.msg.len() + Level1::SIG_BYTES);
    }
}

#[test]
fn kat_lvl3_parses() {
    let records = parse_kat(KAT_LVL3);
    assert_eq!(records.len(), 100, "100 KAT records expected at Level-3");
    for r in &records {
        assert_eq!(r.seed.len(), 48);
        assert_eq!(r.pk.len(), Level3::PK_BYTES);
        assert_eq!(r.sk.len(), Level3::SK_BYTES);
        assert_eq!(r.sm.len(), r.msg.len() + Level3::SIG_BYTES);
    }
}

#[test]
fn kat_lvl5_parses() {
    let records = parse_kat(KAT_LVL5);
    assert_eq!(records.len(), 100, "100 KAT records expected at Level-5");
    for r in &records {
        assert_eq!(r.seed.len(), 48);
        assert_eq!(r.pk.len(), Level5::PK_BYTES);
        assert_eq!(r.sk.len(), Level5::SK_BYTES);
        assert_eq!(r.sm.len(), r.msg.len() + Level5::SIG_BYTES);
    }
}

#[test]
fn kat_records_have_unique_seeds() {
    use std::collections::BTreeSet;
    for (lvl_name, text) in [("lvl1", KAT_LVL1), ("lvl3", KAT_LVL3), ("lvl5", KAT_LVL5)] {
        let records = parse_kat(text);
        let seeds: BTreeSet<Vec<u8>> = records.into_iter().map(|r| r.seed).collect();
        assert_eq!(seeds.len(), 100, "{lvl_name} seeds are unique");
    }
}

/// Verify every C-generated KAT signature with our verifier. `sm = sig || msg`
/// (NIST), so `sig = sm[..SIG_BYTES]`. All 100 Level-1 vectors must accept.
/// Heavy (tens of seconds), hence `#[ignore]`; run with `--ignored`.
#[test]
#[ignore = "heavy: verifies all 100 C-generated KAT signatures"]
fn kat_lvl1_verify_only() {
    let records = parse_kat(KAT_LVL1);
    for r in &records {
        let sig = &r.sm[..Level1::SIG_BYTES];
        assert!(
            pq_sqisign::verify::<Level1>(&r.msg, sig, &r.pk).is_ok(),
            "KAT verify[{}] must accept",
            r.count
        );
    }
}

/// Byte-exact KAT signing: our signer must reproduce each C-generated `sm`.
/// Still a stub — SQIsign signing is randomized, so matching the KAT bytes
/// requires seeding our DRBG from the record's NIST `seed` (not yet wired).
#[test]
#[ignore = "byte-exact KAT signing needs a seed-aligned DRBG (not yet wired)"]
fn kat_lvl1_signs_and_verifies() {
    let records = parse_kat(KAT_LVL1);
    for r in &records {
        let sig = vec![0u8; Level1::SIG_BYTES];
        // pq_sqisign::sign::<Level1, _>(&mut rng_from_seed(&r.seed), &r.msg, &r.sk, &mut sig)
        //     .expect("sign succeeds");
        // assert_eq!(sig, r.sm[..Level1::SIG_BYTES]);
        // pq_sqisign::verify::<Level1>(&r.msg, &sig, &r.pk).expect("verify succeeds");
        let _ = (r, sig);
    }
}
