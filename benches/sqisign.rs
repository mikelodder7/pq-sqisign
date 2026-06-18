// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(missing_docs)]
//! Criterion benchmarks for the SQIsign typed API.
//!
//! Run with:
//!   cargo bench --bench sqisign
//!
//! All three core operations are benchmarked at security level 1:
//!   - keygen  — `KeyPair::generate`
//!   - sign    — `SigningKey::sign` (keypair pre-generated in setup)
//!   - verify  — `VerifyingKey::verify` (keypair + signature pre-generated in setup)
//!
//! A `ChaCha20Rng` seeded with a fixed value is used throughout so results
//! are deterministic and reproducible across runs.

use criterion::{Criterion, criterion_group, criterion_main};
use pq_sqisign::{keypair::KeyPair, params::Level1};
use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};
use std::time::Duration;

const MSG: &[u8] = b"sqisign benchmark message";
const SEED: [u8; 32] = [0x42u8; 32];

fn bench_keygen(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqisign-lvl1");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(120));
    group.bench_function("keygen", |b| {
        b.iter(|| {
            let mut rng = ChaCha20Rng::from_seed(SEED);
            KeyPair::<Level1>::generate(&mut rng).expect("keygen failed")
        });
    });
    group.finish();
}

fn bench_sign(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::from_seed(SEED);
    let kp = KeyPair::<Level1>::generate(&mut rng).expect("keygen failed");
    let sk = kp.signing_key();

    let mut group = c.benchmark_group("sqisign-lvl1");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(120));
    group.bench_function("sign", |b| {
        b.iter(|| {
            let mut rng = ChaCha20Rng::from_seed(SEED);
            sk.sign(MSG, &mut rng).expect("sign failed")
        });
    });
    group.finish();
}

fn bench_verify(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::from_seed(SEED);
    let kp = KeyPair::<Level1>::generate(&mut rng).expect("keygen failed");
    let (sk, vk) = kp.into_parts();
    let sig = sk
        .sign(MSG, &mut ChaCha20Rng::from_seed(SEED))
        .expect("sign failed");

    let mut group = c.benchmark_group("sqisign-lvl1");
    group.bench_function("verify", |b| {
        b.iter(|| vk.verify(MSG, &sig).expect("verify failed"));
    });
    group.finish();
}

criterion_group!(benches, bench_keygen, bench_sign, bench_verify);
criterion_main!(benches);
