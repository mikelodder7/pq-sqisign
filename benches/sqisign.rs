// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(missing_docs)]
//! Criterion benchmarks for the SQIsign typed API.
//!
//! Run with:
//!   cargo bench --bench sqisign
//!
//! All three core operations are benchmarked at every NIST security level
//! (Level1, Level3, Level5):
//!   - keygen  — `KeyPair::generate`
//!   - sign    — `SigningKey::sign` (keypair pre-generated in setup)
//!   - verify  — `VerifyingKey::verify` (keypair + signature pre-generated in setup)
//!
//! Each level reports under its own group (`sqisign-lvl1`, `sqisign-lvl3`,
//! `sqisign-lvl5`). A `ChaCha20Rng` seeded with a fixed value is used
//! throughout so results are deterministic and reproducible across runs.

use criterion::{Criterion, criterion_group, criterion_main};
use pq_sqisign::{
    keypair::KeyPair,
    params::Params,
    params::{Level1, Level3, Level5},
};
use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};
use std::time::Duration;

const MSG: &[u8] = b"sqisign benchmark message";
const SEED: [u8; 32] = [0x42u8; 32];

/// Benchmark keygen, sign, and verify for a single security level `P`.
///
/// `level` is the short suffix used in the criterion group name, e.g. `"lvl3"`.
fn bench_level<P: Params>(c: &mut Criterion, level: &str) {
    // Pre-generate a keypair + signature once for the sign/verify benches.
    //
    // Levels 3 and 5 are defined as parameter sets but their keygen/sign paths
    // are not yet implemented in the library (they return `Error::Unimplemented`).
    // Rather than panic the whole bench run, skip any level the library cannot
    // execute — the scaffolding is ready for when those paths land.
    let mut rng = ChaCha20Rng::from_seed(SEED);
    let kp = match KeyPair::<P>::generate(&mut rng) {
        Ok(kp) => kp,
        Err(e) => {
            eprintln!("sqisign-{level}: skipped — keygen unavailable ({e})");
            return;
        }
    };
    let (sk, vk) = kp.into_parts();
    let sig = match sk.sign(MSG, &mut ChaCha20Rng::from_seed(SEED)) {
        Ok(sig) => sig,
        Err(e) => {
            eprintln!("sqisign-{level}: skipped — sign unavailable ({e})");
            return;
        }
    };

    let mut group = c.benchmark_group(format!("sqisign-{level}"));
    // keygen and sign are slow at higher levels; cap samples and extend time
    // so criterion can collect a full sample set without warnings.
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(120));

    group.bench_function("keygen", |b| {
        b.iter(|| {
            let mut rng = ChaCha20Rng::from_seed(SEED);
            KeyPair::<P>::generate(&mut rng).expect("keygen failed")
        });
    });

    group.bench_function("sign", |b| {
        b.iter(|| {
            let mut rng = ChaCha20Rng::from_seed(SEED);
            sk.sign(MSG, &mut rng).expect("sign failed")
        });
    });

    group.bench_function("verify", |b| {
        b.iter(|| vk.verify(MSG, &sig).expect("verify failed"));
    });

    group.finish();
}

fn bench_all(c: &mut Criterion) {
    bench_level::<Level1>(c, "lvl1");
    bench_level::<Level3>(c, "lvl3");
    bench_level::<Level5>(c, "lvl5");
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
