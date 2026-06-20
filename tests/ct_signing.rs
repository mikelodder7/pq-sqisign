// SPDX-License-Identifier: MIT OR Apache-2.0
//! Stage-0 constant-time measurement harness for signing.
//!
//! Goal: detect whether signing wall-time depends on the SECRET KEY. We use a
//! dudect-style fixed-vs-random methodology with Welch's two-sample t-test:
//!
//! - **FIX class**: one fixed keypair, signed repeatedly with a fixed-seed sign
//!   RNG. Signing is then deterministic for this class, so its timings form a
//!   tight cluster (measurement noise only).
//! - **RND class**: a fresh random keypair each trial, signed with the same
//!   fixed-seed sign RNG. Timing here varies with the secret key.
//!
//! If the two timing distributions differ (|t| above the dudect threshold of
//! ~4.5), signing time depends on the secret key — a timing side-channel.
//!
//! This is the verification harness for the constant-time work: the current
//! variable-time signing path is EXPECTED to leak (large |t|); blinding the
//! secret ideal should drive |t| below the threshold. Re-run after each CT
//! change to measure progress — never claim "constant-time" without this.
//!
//! Sample count via `CT_SAMPLES` env (default 40). Heavy (sign ~1.5s each), so
//! `#[ignore]`; run with:
//!   CT_SAMPLES=60 cargo test --release --features kat --test ct_signing -- --ignored --nocapture
#![allow(missing_docs)]

use pq_sqisign::keypair::KeyPair;
use pq_sqisign::params::Level1;
use rand_chacha::ChaCha20Rng;
use rand_chacha::rand_core::SeedableRng;
use std::time::Instant;

const MSG: &[u8] = b"ct-signing measurement message";
const SIGN_SEED: u64 = 0xC0FFEE_u64; // fixed sign RNG seed → isolates the key as the only variable

/// One signing measurement in nanoseconds for `kp_seed`, or `None` if this
/// seed's keypair fails to keygen or sign within the scheme's retry cap
/// (probabilistic — not every seed converges; such seeds are skipped, which is
/// secret-independent so it does not bias the CT measurement).
fn time_sign(kp_seed: u64, sign_seed: u64) -> Option<f64> {
    let mut kg_rng = ChaCha20Rng::seed_from_u64(kp_seed);
    let kp = KeyPair::<Level1>::generate(&mut kg_rng).ok()?;
    let sk = kp.signing_key();
    let mut sign_rng = ChaCha20Rng::seed_from_u64(sign_seed);
    let t = Instant::now();
    let _sig = sk.sign(MSG, &mut sign_rng).ok()?;
    Some(t.elapsed().as_secs_f64() * 1e9)
}

/// First seed ≥ `start` whose keypair signs successfully (fixed sign seed).
fn first_working_seed(start: u64) -> u64 {
    (start..start + 10_000)
        .find(|&s| time_sign(s, SIGN_SEED).is_some())
        .expect("a working keypair seed exists within range")
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}
fn var(xs: &[f64], m: f64) -> f64 {
    xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len() as f64 - 1.0)
}

#[test]
#[ignore = "heavy CT measurement: sign ~1.5s × 2·CT_SAMPLES trials"]
fn ct_sign_fixed_vs_random_key() {
    let n: usize = std::env::var("CT_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);

    // FIX class: one fixed working key, signed N times — deterministic ⇒ noise only.
    let fix_seed = first_working_seed(1);
    let fix: Vec<f64> = (0..n)
        .map(|_| time_sign(fix_seed, SIGN_SEED).expect("fixed key signs deterministically"))
        .collect();
    // RND class: N distinct working keys (scan seeds from 1000, skipping non-converging ones).
    let mut rnd: Vec<f64> = Vec::with_capacity(n);
    let mut seed = 1000u64;
    while rnd.len() < n {
        if let Some(t) = time_sign(seed, SIGN_SEED) {
            rnd.push(t);
        }
        seed += 1;
    }

    let (mf, mr) = (mean(&fix), mean(&rnd));
    let (vf, vr) = (var(&fix, mf), var(&rnd, mr));
    let t = (mf - mr) / (vf / n as f64 + vr / n as f64).sqrt();
    // Primary CT metric: with the sign-RNG fixed, FIX (one key, repeated) is the
    // noise floor; RND (many keys) is the key-induced timing spread. A constant-
    // time signer has RND_std ≈ FIX_std (ratio ≈ 1). The leak manifests as
    // VARIANCE (different keys take wildly different times), which a Welch t-test
    // on means under-detects — so the std-ratio is the verdict metric.
    let (sf, sr) = (vf.sqrt(), vr.sqrt());
    let std_ratio = sr / sf.max(1.0);

    eprintln!("\n==== CT SIGNING MEASUREMENT (n={n} per class) ====");
    eprintln!(
        "FIX (noise floor)  mean {:.1} ms  std {:.1} ms",
        mf / 1e6,
        sf / 1e6
    );
    eprintln!(
        "RND (key-induced)  mean {:.1} ms  std {:.1} ms",
        mr / 1e6,
        sr / 1e6
    );
    eprintln!(
        "std-ratio RND/FIX = {:.1}   (≈1 ⇒ constant-time; ≫1 ⇒ key-dependent leak)",
        std_ratio
    );
    eprintln!(
        "Welch t (means)   = {:.2}   (secondary; under-detects variance leaks)",
        t
    );
    eprintln!(
        "VERDICT: {}",
        if std_ratio > 3.0 {
            "LEAK — signing time depends on the secret key (expected pre-blinding)"
        } else {
            "no key-dependent timing spread detected (CT target)"
        }
    );
    // Stage 0 is measurement only — it does not assert pass/fail. It quantifies
    // the leak (baseline std-ratio) so the blinding work can be shown to shrink it.
}

/// Blinding-validation mode (random β). Blinding makes timing depend on the
/// fresh per-signature randomness β, not the key — so its guarantee is
/// DISTRIBUTIONAL: over random β, the sign-time distribution is independent of
/// the secret key. This test uses random sign RNG for both classes and asks
/// whether the key shifts the MEAN (Welch t on means is appropriate here since
/// both classes carry the same β-variance). Pre-blinding both classes have huge
/// β-variance already; post-blinding the means should match (|t| < 4.5).
///
/// NOTE: this is the test that should PASS once Stage-1 blinding lands. It is
/// the inverse of `ct_sign_fixed_vs_random_key`, which exposes the raw leak.
#[test]
#[ignore = "heavy CT blinding-validation: sign ~1.5s × 2·CT_SAMPLES trials"]
fn ct_sign_blinding_eval_random_beta() {
    let n: usize = std::env::var("CT_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let fix_seed = first_working_seed(1);

    // FIX: one fixed key, RANDOM β each trial (sign seeds 5000..).
    let mut fix: Vec<f64> = Vec::with_capacity(n);
    let mut ss = 5000u64;
    while fix.len() < n {
        if let Some(t) = time_sign(fix_seed, ss) {
            fix.push(t);
        }
        ss += 1;
    }
    // RND: random key AND random β each trial.
    let mut rnd: Vec<f64> = Vec::with_capacity(n);
    let (mut ks, mut ss2) = (1000u64, 5000u64);
    while rnd.len() < n {
        if let Some(t) = time_sign(ks, ss2) {
            rnd.push(t);
        }
        ks += 1;
        ss2 += 1;
    }
    let (mf, mr) = (mean(&fix), mean(&rnd));
    let (vf, vr) = (var(&fix, mf), var(&rnd, mr));
    let t = (mf - mr) / (vf / n as f64 + vr / n as f64).sqrt();
    eprintln!("\n==== CT BLINDING VALIDATION (random β, n={n}) ====");
    eprintln!(
        "FIX-key  mean {:.1} ms  std {:.1} ms",
        mf / 1e6,
        vf.sqrt() / 1e6
    );
    eprintln!(
        "RND-key  mean {:.1} ms  std {:.1} ms",
        mr / 1e6,
        vr.sqrt() / 1e6
    );
    eprintln!(
        "Welch t = {:.2}  (target post-blinding: |t| < 4.5 ⇒ key doesn't shift timing)",
        t
    );
}
