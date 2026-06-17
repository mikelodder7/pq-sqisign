// SPDX-License-Identifier: MIT OR Apache-2.0
//! Architecture-specific fast paths for limb-level arithmetic.
//!
//! At present the multiplication-heavy work is delegated to `crypto-bigint`,
//! which already picks the appropriate carry-aware intrinsic (`adcx/adox` on
//! x86_64 BMI2, `umulh` on aarch64) when the target supports it. This module
//! exposes a single [`Backend`] enum that names the path the build was
//! compiled for, so callers (and tests) can confirm at runtime which fast
//! path is live.
//!
//! Per-prime Montgomery reduction is not specialized here; it could move
//! into this module if the generic path becomes a bottleneck.

/// Which architecture-specific fast path the current build selected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    /// Portable path — generic 64-bit limb arithmetic via `crypto-bigint`.
    Portable,
    /// x86_64 with BMI2 (`adcx`, `adox`, `mulx`) available.
    X86_64Bmi2,
    /// aarch64 with NEON-aware wide multiplication.
    Aarch64,
}

/// The backend the crate was compiled for.
pub const BACKEND: Backend = {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        Backend::X86_64Bmi2
    }
    #[cfg(target_arch = "aarch64")]
    {
        Backend::Aarch64
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "bmi2"),
        target_arch = "aarch64"
    )))]
    {
        Backend::Portable
    }
};
