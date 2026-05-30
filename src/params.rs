// SPDX-License-Identifier: MIT OR Apache-2.0
//! Security-level parameter sets for SQIsign.
//!
//! SQIsign defines three NIST security levels. Each level fixes a base prime
//! `p ≡ 3 (mod 4)` of the form `p = c · 2^f − 1`, with the quadratic extension
//! `F_{p^2} = F_p[i]/(i^2 + 1)` serving as the field of definition for the
//! supersingular curve `E_0 : y^2 = x^3 + x`.
//!
//! | Level | Bits of p | c  | f   | pk | sk  | sig |
//! |-------|-----------|----|-----|----|-----|-----|
//! |   1   |   251     | 5  | 248 | 65 | 353 | 148 |
//! |   3   |   383     | 65 | 376 | 97 | 529 | 224 |
//! |   5   |   505     | 27 | 500 |129 | 701 | 292 |

pub mod lvl1;
pub mod lvl3;
pub mod lvl5;

pub use lvl1::Level1;
pub use lvl3::Level3;
pub use lvl5::Level5;

/// Common parameter-set surface implemented by [`Level1`], [`Level3`], [`Level5`].
///
/// A `Params` implementor names the prime, the field-element byte size, and the
/// encoded public-key / secret-key / signature byte sizes that the SQIsign wire
/// format demands at this level. The field-arithmetic and curve-arithmetic
/// modules are generic over `Params` so they don't have to be rewritten per
/// level.
pub trait Params: 'static + Copy + core::fmt::Debug {
    /// NIST security level (1, 3, or 5).
    const LEVEL: u8;
    /// Bit-length of the base prime `p` (251, 383, or 505).
    const P_BITS: usize;
    /// Cofactor `c` in `p = c · 2^f − 1`.
    const C: u64;
    /// Exponent `f` in `p = c · 2^f − 1`.
    const F: usize;
    /// Bytes needed to encode an `F_p` element (`(P_BITS + 7) / 8`).
    const FP_BYTES: usize;
    /// Bytes needed to encode an `F_{p^2}` element (`2 · FP_BYTES`).
    const FP2_BYTES: usize;
    /// Encoded public key length, in bytes.
    const PK_BYTES: usize;
    /// Encoded secret key length, in bytes.
    const SK_BYTES: usize;
    /// Encoded signature length, in bytes.
    const SIG_BYTES: usize;
    /// `SQIsign_response_length` from the spec (challenge response bit-length).
    const RESPONSE_BITS: usize;
    /// `HASH_ITERATIONS` from the spec (challenge-derivation oracle iterations).
    const HASH_ITERATIONS: usize;
    /// `SECURITY_BITS` from the spec — the NIST-level classical security
    /// parameter (128/192/256 for L1/L3/L5).
    const SECURITY_BITS: usize;
    /// `FINDUV_box_size` from the C reference — the half-side of the
    /// `[-m, m]^4` hypercube that
    /// [`enumerate_hypercube`](crate::isogeny::clapotis::enumerate_hypercube)
    /// walks when feeding the Clapotis `find_uv` orchestrator. Per-level
    /// values extracted in S206 / S209 from the SQIsign C reference's
    /// `src/precomp/ref/lvl{1,3,5}/include/quaternion_constants.h`:
    ///
    /// | Level | `FINDUV_BOX_SIZE` |
    /// |---|---|
    /// | L1 | 2 |
    /// | L3 | 3 |
    /// | L5 | 3 |
    ///
    /// Validated at L1 by the S208 probe test
    /// `find_uv_at_l1_box_size_two_matches_c_ref_constant`: m=2 finds
    /// Bezout solutions for our γ=(1,0,1,0) smoke-test fixture.
    const FINDUV_BOX_SIZE: i64;
    /// `NUM_ALTERNATE_EXTREMAL_ORDERS` from the C reference — the count
    /// of non-trivial alternate connecting ideals used by the Clapotis
    /// `find_uv` orchestrator's j-loop. The actual data array
    /// `CONNECTING_IDEALS[NUM_ALTERNATE_EXTREMAL_ORDERS + 1]` includes
    /// index `[0]` (the trivial connector that's skipped) plus the
    /// non-trivial entries (`ALTERNATE_CONNECTING_IDEALS = CONNECTING_IDEALS + 1`).
    /// Per-level values per S206 research agents:
    ///
    /// | Level | `NUM_ALTERNATE_EXTREMAL_ORDERS` |
    /// |---|---|
    /// | L1 | 6 |
    /// | L3 | 7 |
    /// | L5 | 6 |
    ///
    /// **L3 settled in S213** via verbatim quote from
    /// `src/precomp/ref/lvl3/include/quaternion_data.h:4`:
    /// `#define NUM_ALTERNATE_EXTREMAL_ORDERS 7`. Cross-check at
    /// line 11 of the same header: `extern const quat_left_ideal_t
    /// CONNECTING_IDEALS[8];` confirms the +1 invariant (Agent 1's
    /// S206 reading was correct; Agent 2 was wrong).
    const NUM_ALTERNATE_EXTREMAL_ORDERS: usize;
    /// `FINDUV_cube_size` from the C reference — the count of
    /// short-vector candidates `enumerate_hypercube` accepts before
    /// the `find_uv` orchestrator considers the box "exhausted".
    /// This is a PRE-SORT count (the C ref filters down further via
    /// the odd-quotient and i-action symmetry filters). Per-level
    /// values per S206 research agents:
    ///
    /// | Level | `FINDUV_CUBE_SIZE` |
    /// |---|---|
    /// | L1 | 624 |
    /// | L3 | 2400 |
    /// | L5 | 2400 |
    ///
    /// Not yet consumed by our Rust port — we use `Vec`'s natural
    /// growth instead of pre-allocating to `FINDUV_CUBE_SIZE` like
    /// the C ref. Future tuning may add capacity hints based on this.
    const FINDUV_CUBE_SIZE: u64;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// S209: `FINDUV_BOX_SIZE` per level matches the C reference's
    /// `src/precomp/ref/lvl{1,3,5}/include/quaternion_constants.h`
    /// (extracted via S206 research agents).
    #[test]
    fn finduv_box_size_per_level_matches_c_ref_constants() {
        assert_eq!(
            Level1::FINDUV_BOX_SIZE,
            2,
            "L1 FINDUV_box_size = 2 per C ref"
        );
        assert_eq!(
            Level3::FINDUV_BOX_SIZE,
            3,
            "L3 FINDUV_box_size = 3 per C ref"
        );
        assert_eq!(
            Level5::FINDUV_BOX_SIZE,
            3,
            "L5 FINDUV_box_size = 3 per C ref"
        );
    }

    /// S212/S213: `NUM_ALTERNATE_EXTREMAL_ORDERS` per level matches
    /// S206 research + S213 confirmation. **L3 was settled in S213**
    /// via verbatim quote from
    /// `src/precomp/ref/lvl3/include/quaternion_data.h:4`.
    #[test]
    fn num_alternate_extremal_orders_per_level_matches_c_ref_constants() {
        assert_eq!(
            Level1::NUM_ALTERNATE_EXTREMAL_ORDERS,
            6,
            "L1 NUM_ALTERNATE = 6"
        );
        assert_eq!(
            Level3::NUM_ALTERNATE_EXTREMAL_ORDERS,
            7,
            "L3 NUM_ALTERNATE = 7 — confirmed S213 (Agent 1 was right)",
        );
        assert_eq!(
            Level5::NUM_ALTERNATE_EXTREMAL_ORDERS,
            6,
            "L5 NUM_ALTERNATE = 6"
        );
    }

    /// S212: `FINDUV_CUBE_SIZE` per level matches S206 research agent
    /// values. Both agents agreed on these.
    #[test]
    fn finduv_cube_size_per_level_matches_c_ref_constants() {
        assert_eq!(Level1::FINDUV_CUBE_SIZE, 624, "L1 FINDUV_CUBE_SIZE = 624");
        assert_eq!(Level3::FINDUV_CUBE_SIZE, 2400, "L3 FINDUV_CUBE_SIZE = 2400");
        assert_eq!(Level5::FINDUV_CUBE_SIZE, 2400, "L5 FINDUV_CUBE_SIZE = 2400");
    }
}
