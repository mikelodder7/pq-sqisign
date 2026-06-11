// SPDX-License-Identifier: MIT OR Apache-2.0
//! The lvl1 ALTERNATE EXTREMAL (maximal) ORDERS `ln[1..7]` — Rust transcriptions
//! of the SQIsign C reference `quaternion_data.c::EXTREMAL_ORDERS[1..7]` (the 6
//! "NICE" alternate orders; `EXTREMAL_ORDERS[0]` is the standard order O_0).
//!
//! Each order carries: the order LATTICE (denom + 4x4 basis, std `(1,i,j,ij)`
//! coords, COLUMN convention — `basis[i][j]`, column j is an element); the
//! small-discriminant element `z` with `z^2 = -q`; the norm-`p` element `t`
//! with `t^2 = -p`, orthogonal to `z`; and `q = |z^2|`. Consumed by the
//! dim2id2iso spine's `n_order != 0` path (represent-integer on the alternate
//! order + `quat_lattice_contains`) — S340.
//!
//! Byte-exact from the C GMP-64 limbs (S340 extractor). Validated by
//! `alternate_extremal_orders_l1_satisfy_z2_q_t2_p`: `z` is trace-zero with
//! `N(z) = q*z_denom^2`, and `N(t) = p*t_denom^2`.

use crypto_bigint::{Int, Uint};

use crate::quaternion::Quaternion;

/// An alternate extremal maximal order (the C `quat_p_extremal_maximal_order_t`).
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct AltExtremalOrder {
    /// Order lattice basis, std coords, column-major `basis[i][j]`.
    pub order_basis: [[Int<8>; 4]; 4],
    /// Order lattice denominator.
    pub order_denom: Int<8>,
    /// Small-discriminant element `z` (`z^2 = -q`), std coords numerator.
    pub z: Quaternion<8>,
    /// Denominator of `z`.
    pub z_denom: Int<8>,
    /// Norm-`p` element `t` (`t^2 = -p`), std coords numerator.
    pub t: Quaternion<8>,
    /// Denominator of `t`.
    pub t_denom: Int<8>,
    /// `q = |z^2|`.
    pub q: u32,
}

/// The STANDARD order O_0 (C `EXTREMAL_ORDERS[0]` / `quat_lattice_O0_set`),
/// expressed as an [`AltExtremalOrder`] so the C-faithful `quat_represent_integer`
/// port (`represent_integer_over_alt_order`) can serve the index-0 keygen
/// fixed-degree path. `q = 1`, `z = i` (`z² = −1 = −q`), `t = j` (`t² = −p`).
/// Order lattice (std `(1,i,j,ij)` coords, COLUMN convention, denom 2) is the C
/// `quat_lattice_O0_set`: `basis[0][0]=2, basis[1][1]=2, basis[2][2]=1,
/// basis[1][2]=1, basis[3][3]=1, basis[0][3]=1` — i.e. `O_0 = ⟨1, i, (i+j)/2,
/// (1+ij)/2⟩`.
#[allow(dead_code)]
pub fn standard_order_o0_l1() -> AltExtremalOrder {
    let n = |x: i64| Int::<8>::from_i64(x);
    AltExtremalOrder {
        // basis[i][j] = coord i of generator j (column-major), over denom 2.
        order_basis: [
            [n(2), n(0), n(0), n(1)], // row 0: [0][0]=2, [0][3]=1
            [n(0), n(2), n(1), n(0)], // row 1: [1][1]=2, [1][2]=1
            [n(0), n(0), n(1), n(0)], // row 2: [2][2]=1
            [n(0), n(0), n(0), n(1)], // row 3: [3][3]=1
        ],
        order_denom: n(2),
        z: Quaternion::<8>::new(n(0), n(1), n(0), n(0)), // z = i, z² = −1 = −q
        z_denom: n(1),
        t: Quaternion::<8>::new(n(0), n(0), n(1), n(0)), // t = j, t² = −p
        t_denom: n(1),
        q: 1,
    }
}

/// Alternate extremal order 0 (C `EXTREMAL_ORDERS[1]`), q = 5.
#[allow(dead_code)]
pub fn alternate_extremal_order_0_l1() -> AltExtremalOrder {
    AltExtremalOrder {
        order_basis: [
            [
                *Uint::<8>::from_words([0x0, 0x1000000000000000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0])
                    .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([0x0, 0x800000000000000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0])
                    .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            ],
            [
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                (*Uint::<8>::from_words([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int())
                    .wrapping_neg(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                (*Uint::<8>::from_words([0x0, 0x0, 0x0, 0x80000000000000, 0x0, 0x0, 0x0, 0x0])
                    .as_int())
                .wrapping_neg(),
            ],
            [
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([0x0, 0x800000000000000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0])
                    .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            ],
            [
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                (*Uint::<8>::from_words([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int())
                    .wrapping_neg(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            ],
        ],
        order_denom: *Uint::<8>::from_words([
            0x0,
            0x1000000000000000,
            0x0,
            0x0,
            0x0,
            0x0,
            0x0,
            0x0,
        ])
        .as_int(),
        z: Quaternion::<8>::new(
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            (*Uint::<8>::from_words([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int())
                .wrapping_neg(),
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            (*Uint::<8>::from_words([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int())
                .wrapping_neg(),
        ),
        z_denom: *Uint::<8>::from_words([0x0, 0x1000000000000000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0])
            .as_int(),
        t: Quaternion::<8>::new(
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
        ),
        t_denom: *Uint::<8>::from_words([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
        q: 5,
    }
}

/// Alternate extremal order 1 (C `EXTREMAL_ORDERS[2]`), q = 17.
#[allow(dead_code)]
pub fn alternate_extremal_order_1_l1() -> AltExtremalOrder {
    AltExtremalOrder {
        order_basis: [
            [
                *Uint::<8>::from_words([
                    0xf5f27a647b8578d4,
                    0xb8746101369629b9,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([
                    0xfaf93d323dc2bc6a,
                    0x5c3a30809b4b14dc,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            ],
            [
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([
                    0x95ad2ad56fa47d47,
                    0xc89877e749be8a4b,
                    0x1,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([
                    0x3e355e2970603f47,
                    0x78dd10ae2a1bd950,
                    0x0,
                    0x280000000000000,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
            ],
            [
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([
                    0xfaf93d323dc2bc6a,
                    0x5c3a30809b4b14dc,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            ],
            [
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([0x11, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                (*Uint::<8>::from_words([
                    0xb19426e828ee3fe7,
                    0xd6de568af586d7a,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int())
                .wrapping_neg(),
            ],
        ],
        order_denom: *Uint::<8>::from_words([
            0xf5f27a647b8578d4,
            0xb8746101369629b9,
            0x0,
            0x0,
            0x0,
            0x0,
            0x0,
            0x0,
        ])
        .as_int(),
        z: Quaternion::<8>::new(
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([
                0x95ad2ad56fa47d47,
                0xc89877e749be8a4b,
                0x1,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ])
            .as_int(),
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x11, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
        ),
        z_denom: *Uint::<8>::from_words([
            0xf5f27a647b8578d4,
            0xb8746101369629b9,
            0x0,
            0x0,
            0x0,
            0x0,
            0x0,
            0x0,
        ])
        .as_int(),
        t: Quaternion::<8>::new(
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
        ),
        t_denom: *Uint::<8>::from_words([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
        q: 17,
    }
}

/// Alternate extremal order 2 (C `EXTREMAL_ORDERS[3]`), q = 37.
#[allow(dead_code)]
pub fn alternate_extremal_order_2_l1() -> AltExtremalOrder {
    AltExtremalOrder {
        order_basis: [
            [
                *Uint::<8>::from_words([
                    0x3c6fa8e67715e5e2,
                    0x17949bec872b9078,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([
                    0x1e37d4733b8af2f1,
                    0xbca4df64395c83c,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            ],
            [
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                (*Uint::<8>::from_words([
                    0xb034808274c8307a,
                    0x9ab399ac43a4e8a,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int())
                .wrapping_neg(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([
                    0x3d25ca466bc9954f,
                    0x4f5822946ed431b,
                    0xeb3e45306eb3e453,
                    0x45306eb3e45306,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
            ],
            [
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([
                    0x1e37d4733b8af2f1,
                    0xbca4df64395c83c,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            ],
            [
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([0x4, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([
                    0xbd312454ca3a0e7f,
                    0x2172f0cb4ce562,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
            ],
        ],
        order_denom: *Uint::<8>::from_words([
            0x3c6fa8e67715e5e2,
            0x17949bec872b9078,
            0x0,
            0x0,
            0x0,
            0x0,
            0x0,
            0x0,
        ])
        .as_int(),
        z: Quaternion::<8>::new(
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            (*Uint::<8>::from_words([
                0xb034808274c8307a,
                0x9ab399ac43a4e8a,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ])
            .as_int())
            .wrapping_neg(),
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x4, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
        ),
        z_denom: *Uint::<8>::from_words([
            0x3c6fa8e67715e5e2,
            0x17949bec872b9078,
            0x0,
            0x0,
            0x0,
            0x0,
            0x0,
            0x0,
        ])
        .as_int(),
        t: Quaternion::<8>::new(
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
        ),
        t_denom: *Uint::<8>::from_words([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
        q: 37,
    }
}

/// Alternate extremal order 3 (C `EXTREMAL_ORDERS[4]`), q = 41.
#[allow(dead_code)]
pub fn alternate_extremal_order_3_l1() -> AltExtremalOrder {
    AltExtremalOrder {
        order_basis: [
            [
                *Uint::<8>::from_words([
                    0xde33c5116deeafa2,
                    0x2df94f97c89ec8ce,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([
                    0x6f19e288b6f757d1,
                    0x16fca7cbe44f6467,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            ],
            [
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([
                    0xd17aa943da6bdd36,
                    0x44d44b0c564ce307,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                (*Uint::<8>::from_words([
                    0xa0a2047cc4063a03,
                    0x6cee07961df46dbc,
                    0xc7ce0c7ce0c7ce0c,
                    0x7ce0c7ce0c7ce0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int())
                .wrapping_neg(),
            ],
            [
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([
                    0x6f19e288b6f757d1,
                    0x16fca7cbe44f6467,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            ],
            [
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                (*Uint::<8>::from_words([0x8, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int())
                    .wrapping_neg(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                (*Uint::<8>::from_words([
                    0xd9f82148a1e2188f,
                    0xd6e1b21a072e79,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int())
                .wrapping_neg(),
            ],
        ],
        order_denom: *Uint::<8>::from_words([
            0xde33c5116deeafa2,
            0x2df94f97c89ec8ce,
            0x0,
            0x0,
            0x0,
            0x0,
            0x0,
            0x0,
        ])
        .as_int(),
        z: Quaternion::<8>::new(
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([
                0xd17aa943da6bdd36,
                0x44d44b0c564ce307,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ])
            .as_int(),
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            (*Uint::<8>::from_words([0x8, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int())
                .wrapping_neg(),
        ),
        z_denom: *Uint::<8>::from_words([
            0xde33c5116deeafa2,
            0x2df94f97c89ec8ce,
            0x0,
            0x0,
            0x0,
            0x0,
            0x0,
            0x0,
        ])
        .as_int(),
        t: Quaternion::<8>::new(
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
        ),
        t_denom: *Uint::<8>::from_words([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
        q: 41,
    }
}

/// Alternate extremal order 4 (C `EXTREMAL_ORDERS[5]`), q = 53.
#[allow(dead_code)]
pub fn alternate_extremal_order_4_l1() -> AltExtremalOrder {
    AltExtremalOrder {
        order_basis: [
            [
                *Uint::<8>::from_words([
                    0x380014f2025b96a4,
                    0x7bbeab7f79584e7c,
                    0x1,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([
                    0x1c000a79012dcb52,
                    0xbddf55bfbcac273e,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            ],
            [
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                (*Uint::<8>::from_words([
                    0x4ba119e7333973e3,
                    0xdbd0ee6227026ebc,
                    0x7,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int())
                .wrapping_neg(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([
                    0x9f01d923dd0ca33,
                    0x83f7e395afe92f81,
                    0xfffffffffffffffc,
                    0x27fffffffffffff,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
            ],
            [
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([
                    0x1c000a79012dcb52,
                    0xbddf55bfbcac273e,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            ],
            [
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([0x35, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([
                    0x87f571c0f93ceb73,
                    0x12fab9cbcb3c667a,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
            ],
        ],
        order_denom: *Uint::<8>::from_words([
            0x380014f2025b96a4,
            0x7bbeab7f79584e7c,
            0x1,
            0x0,
            0x0,
            0x0,
            0x0,
            0x0,
        ])
        .as_int(),
        z: Quaternion::<8>::new(
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            (*Uint::<8>::from_words([
                0x4ba119e7333973e3,
                0xdbd0ee6227026ebc,
                0x7,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ])
            .as_int())
            .wrapping_neg(),
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x35, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
        ),
        z_denom: *Uint::<8>::from_words([
            0x380014f2025b96a4,
            0x7bbeab7f79584e7c,
            0x1,
            0x0,
            0x0,
            0x0,
            0x0,
            0x0,
        ])
        .as_int(),
        t: Quaternion::<8>::new(
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
        ),
        t_denom: *Uint::<8>::from_words([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
        q: 53,
    }
}

/// Alternate extremal order 5 (C `EXTREMAL_ORDERS[6]`), q = 97.
#[allow(dead_code)]
pub fn alternate_extremal_order_5_l1() -> AltExtremalOrder {
    AltExtremalOrder {
        order_basis: [
            [
                *Uint::<8>::from_words([
                    0xe2b97b9e55af7ffa,
                    0xc227f76b578ca7af,
                    0xf,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([
                    0xf15cbdcf2ad7bffd,
                    0xe113fbb5abc653d7,
                    0x7,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            ],
            [
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                (*Uint::<8>::from_words([
                    0xa2ef1ce7f02b0d16,
                    0x66759632c56054b,
                    0x6f,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int())
                .wrapping_neg(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([
                    0x84ac06ea9d3bf0ab,
                    0xd021882bdde962e5,
                    0xffffffffffffffe2,
                    0x13ffffffffffffff,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
            ],
            [
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([
                    0xf15cbdcf2ad7bffd,
                    0xe113fbb5abc653d7,
                    0x7,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            ],
            [
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([0x308, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
                *Uint::<8>::from_words([
                    0x77013f15c4a1f37,
                    0x9281da3156007183,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                    0x0,
                ])
                .as_int(),
            ],
        ],
        order_denom: *Uint::<8>::from_words([
            0xe2b97b9e55af7ffa,
            0xc227f76b578ca7af,
            0xf,
            0x0,
            0x0,
            0x0,
            0x0,
            0x0,
        ])
        .as_int(),
        z: Quaternion::<8>::new(
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            (*Uint::<8>::from_words([
                0xa2ef1ce7f02b0d16,
                0x66759632c56054b,
                0x6f,
                0x0,
                0x0,
                0x0,
                0x0,
                0x0,
            ])
            .as_int())
            .wrapping_neg(),
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x308, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
        ),
        z_denom: *Uint::<8>::from_words([
            0xe2b97b9e55af7ffa,
            0xc227f76b578ca7af,
            0xf,
            0x0,
            0x0,
            0x0,
            0x0,
            0x0,
        ])
        .as_int(),
        t: Quaternion::<8>::new(
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
            *Uint::<8>::from_words([0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
        ),
        t_denom: *Uint::<8>::from_words([0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]).as_int(),
        q: 97,
    }
}

// ── order-basis coordinate extraction (quat_lattice_contains) ──────────────
//
// The dim2id2iso n_order≠0 path decomposes an endomorphism `θ` over an
// ALTERNATE order's basis (the C `quat_alg_make_primitive(.., &EXTREMAL_ORDERS
// [index].order)`), which calls `quat_lattice_contains` to get `θ`'s integer
// coordinates in that order's lattice. We port that here generically. The
// output coordinates are CANONICAL (the unique integer vector `c` with
// `basis · c / denom = θ`), so this matches the C bit-for-bit regardless of the
// internal adjugate algorithm (the C uses an optimized 2×2-minor expansion; a
// plain cofactor adjugate produces the same unique adjugate matrix).

/// Sign-extend an `Int<A>` into a wider `Int<B>` (`A ≤ B`). Order/element data
/// is stored at `Int<8>`; the adjugate·element·denom intermediates reach
/// ~2^620, so the coordinate solve runs at `Int<16>`.
#[inline]
fn widen_int<const A: usize, const B: usize>(x: &Int<A>) -> Int<B> {
    let mag = x.abs();
    let w = mag.to_words();
    let mut wb = [0u64; B];
    wb[..A].copy_from_slice(&w);
    let u = *Uint::<B>::from_words(wb).as_int();
    if bool::from(x.is_negative()) {
        u.wrapping_neg()
    } else {
        u
    }
}

/// Determinant of the 3×3 minor of `m` obtained by deleting row `skip_r` and
/// column `skip_c` (Sarrus' rule, wrapping arithmetic).
#[allow(clippy::needless_range_loop)]
fn minor3_det<const L: usize>(m: &[[Int<L>; 4]; 4], skip_r: usize, skip_c: usize) -> Int<L> {
    let mut a = [[Int::<L>::from_i64(0); 3]; 3];
    let mut ri = 0;
    for r in 0..4 {
        if r == skip_r {
            continue;
        }
        let mut ci = 0;
        for c in 0..4 {
            if c == skip_c {
                continue;
            }
            a[ri][ci] = m[r][c];
            ci += 1;
        }
        ri += 1;
    }
    let p = |i: usize, j: usize, k: usize| a[0][i].wrapping_mul(&a[1][j]).wrapping_mul(&a[2][k]);
    p(0, 1, 2)
        .wrapping_add(&p(1, 2, 0))
        .wrapping_add(&p(2, 0, 1))
        .wrapping_sub(&p(2, 1, 0))
        .wrapping_sub(&p(0, 2, 1))
        .wrapping_sub(&p(1, 0, 2))
}

/// Adjugate matrix and determinant of a 4×4 integer matrix: returns
/// `(adj, det)` with the invariant `adj · m = det · I`. `adj[i][j] =
/// (−1)^(i+j) · minor3_det(m, j, i)` (cofactor transpose).
#[allow(clippy::needless_range_loop)]
pub(crate) fn adjugate_with_det<const L: usize>(
    m: &[[Int<L>; 4]; 4],
) -> ([[Int<L>; 4]; 4], Int<L>) {
    let mut adj = [[Int::<L>::from_i64(0); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let cof = minor3_det(m, j, i);
            adj[i][j] = if (i + j) % 2 == 0 {
                cof
            } else {
                cof.wrapping_neg()
            };
        }
    }
    // det = Σ_c (−1)^c · m[0][c] · minor3_det(m, 0, c)
    let mut det = Int::<L>::from_i64(0);
    for c in 0..4 {
        let term = m[0][c].wrapping_mul(&minor3_det(m, 0, c));
        det = if c % 2 == 0 {
            det.wrapping_add(&term)
        } else {
            det.wrapping_sub(&term)
        };
    }
    (adj, det)
}

/// Exact signed division `num / den`, or `None` if `den ∤ num`.
fn exact_div<const L: usize>(num: &Int<L>, den: &Int<L>) -> Option<Int<L>> {
    let dnz = crypto_bigint::NonZero::new(den.abs()).into_option()?;
    let (q, r) = num.abs().div_rem_vartime(&dnz);
    if r != Uint::<L>::ZERO {
        return None;
    }
    let qi = *Uint::<L>::from_words(q.to_words()).as_int();
    let neg = bool::from(num.is_negative()) ^ bool::from(den.is_negative());
    Some(if neg { qi.wrapping_neg() } else { qi })
}

/// Coordinates of the quaternion `x = (x_std numerator) / x_denom` in the
/// lattice `(basis, lat_denom)` — the C `quat_lattice_contains` coordinate
/// output. `basis[i][j]` is coord `i` of generator `j` (column convention), so
/// `x = (1/lat_denom)·basis·coords` ⟺ `coords = lat_denom · basis⁻¹ · x`.
///
/// Returns `None` if `x` is NOT in the lattice (non-integer solution). Run at a
/// width wide enough for `adj·x·lat_denom` (≈2^620 for the lvl1 orders → use
/// `Int<16>`).
pub fn lattice_coords_of<const L: usize>(
    basis: &[[Int<L>; 4]; 4],
    lat_denom: &Int<L>,
    x_std: &[Int<L>; 4],
    x_denom: &Int<L>,
) -> Option<[Int<L>; 4]> {
    let (adj, det) = adjugate_with_det(basis);
    if det == Int::<L>::from_i64(0) {
        return None;
    }
    let work = crate::quaternion::lattice::mat_4x4_eval(&adj, x_std); // adj·x
    let prod = x_denom.wrapping_mul(&det); // x_denom·det
    let mut coords = [Int::<L>::from_i64(0); 4];
    for i in 0..4 {
        let num = work[i].wrapping_mul(lat_denom); // (adj·x)_i · lat_denom
        coords[i] = exact_div(&num, &prod)?;
    }
    Some(coords)
}

/// Coordinates of `(x / x_denom)` in an alternate extremal order's basis,
/// widened to `Int<16>` for the solve. `None` if `x ∉ order`.
#[allow(dead_code, clippy::needless_range_loop)]
pub fn alt_order_coords_of(
    order: &AltExtremalOrder,
    x: &Quaternion<8>,
    x_denom: &Int<8>,
) -> Option<[Int<16>; 4]> {
    let mut basis = [[Int::<16>::from_i64(0); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            basis[i][j] = widen_int::<8, 16>(&order.order_basis[i][j]);
        }
    }
    let lat_denom = widen_int::<8, 16>(&order.order_denom);
    let x_std = [
        widen_int::<8, 16>(&x.a),
        widen_int::<8, 16>(&x.b),
        widen_int::<8, 16>(&x.c),
        widen_int::<8, 16>(&x.d),
    ];
    let xd = widen_int::<8, 16>(x_denom);
    lattice_coords_of(&basis, &lat_denom, &x_std, &xd)
}

/// Decompose `θ` over an alternate extremal order's basis into a primitive
/// coordinate vector and its content (gcd) — the C `quat_alg_make_primitive`
/// over `EXTREMAL_ORDERS[index].order`. Returns `None` if `θ ∉ order`.
#[allow(dead_code)]
pub fn make_primitive_over_alt_order(
    order: &AltExtremalOrder,
    theta: &Quaternion<8>,
    theta_denom: &Int<8>,
) -> Option<([Int<16>; 4], Int<16>)> {
    let coords = alt_order_coords_of(order, theta, theta_denom)?;
    Some(crate::quaternion::o0_mul::make_primitive_from_o0_coords::<16>(&coords))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each alternate order's z² = −q (trace-zero, N(z)=q·z_denom²) and
    /// t² = −p (N(t)=p·t_denom²) — the defining extremal-order invariants.
    #[test]
    fn alternate_extremal_orders_l1_satisfy_z2_q_t2_p() {
        let p = crate::params::lvl1::prime().resize::<8>();
        let orders = [
            alternate_extremal_order_0_l1(),
            alternate_extremal_order_1_l1(),
            alternate_extremal_order_2_l1(),
            alternate_extremal_order_3_l1(),
            alternate_extremal_order_4_l1(),
            alternate_extremal_order_5_l1(),
        ];
        for (k, o) in orders.iter().enumerate() {
            assert_eq!(
                o.z.a,
                Int::<8>::from_i64(0),
                "z must be trace-zero (z[{k}])"
            );
            // N(z) = q · z_denom²
            let nz = o.z.norm(&p).abs();
            let qz = Uint::<8>::from_u64(u64::from(o.q))
                .wrapping_mul(&o.z_denom.abs())
                .wrapping_mul(&o.z_denom.abs());
            assert_eq!(nz, qz, "N(z) must equal q·z_denom² (z[{k}])");
            // N(t) = p · t_denom²
            let nt = o.t.norm(&p).abs();
            let pt = p
                .wrapping_mul(&o.t_denom.abs())
                .wrapping_mul(&o.t_denom.abs());
            assert_eq!(nt, pt, "N(t) must equal p·t_denom² (t[{k}])");
        }
    }

    /// adj · M = det · I for a known 4×4 matrix (the adjugate invariant the
    /// coordinate solve rests on).
    #[test]
    fn adjugate_times_matrix_is_det_identity() {
        // m = [[3,1,0,2],[2,4,1,0],[0,1,5,1],[1,0,2,3]], det = 74.
        let m: [[Int<8>; 4]; 4] = [
            [
                Int::<8>::from_i64(3),
                Int::<8>::from_i64(1),
                Int::<8>::from_i64(0),
                Int::<8>::from_i64(2),
            ],
            [
                Int::<8>::from_i64(2),
                Int::<8>::from_i64(4),
                Int::<8>::from_i64(1),
                Int::<8>::from_i64(0),
            ],
            [
                Int::<8>::from_i64(0),
                Int::<8>::from_i64(1),
                Int::<8>::from_i64(5),
                Int::<8>::from_i64(1),
            ],
            [
                Int::<8>::from_i64(1),
                Int::<8>::from_i64(0),
                Int::<8>::from_i64(2),
                Int::<8>::from_i64(3),
            ],
        ];
        let (adj, det) = adjugate_with_det(&m);
        assert_eq!(det, Int::<8>::from_i64(74), "det must be 74");
        // (adj·m)[i][j] = det·δ_ij
        for i in 0..4 {
            for j in 0..4 {
                let mut s = Int::<8>::from_i64(0);
                for k in 0..4 {
                    s = s.wrapping_add(&adj[i][k].wrapping_mul(&m[k][j]));
                }
                let want = if i == j { det } else { Int::<8>::from_i64(0) };
                assert_eq!(s, want, "adj·m must be det·I at ({i},{j})");
            }
        }
    }

    /// Every alternate order must CONTAIN its own defining elements `1`, `z`,
    /// `t` with integer coordinates that round-trip through the basis. This
    /// simultaneously validates (a) the coordinate solve and (b) the
    /// `order_basis` transpose convention — a transposed basis would generally
    /// reject `z`/`t` (mirroring the S338 connecting-ideal transpose bug).
    #[test]
    fn alternate_extremal_orders_contain_one_z_t_with_roundtrip_coords() {
        let orders = [
            alternate_extremal_order_0_l1(),
            alternate_extremal_order_1_l1(),
            alternate_extremal_order_2_l1(),
            alternate_extremal_order_3_l1(),
            alternate_extremal_order_4_l1(),
            alternate_extremal_order_5_l1(),
        ];
        let one = (
            Quaternion::<8>::new(
                Int::<8>::from_i64(1),
                Int::<8>::from_i64(0),
                Int::<8>::from_i64(0),
                Int::<8>::from_i64(0),
            ),
            Int::<8>::from_i64(1),
        );
        for (k, o) in orders.iter().enumerate() {
            // Widen the order basis once for the round-trip check.
            let mut wbasis = [[Int::<16>::from_i64(0); 4]; 4];
            for i in 0..4 {
                for j in 0..4 {
                    wbasis[i][j] = widen_int::<8, 16>(&o.order_basis[i][j]);
                }
            }
            let wdenom = widen_int::<8, 16>(&o.order_denom);

            for (label, (elem, denom)) in [
                ("1", &one),
                ("z", &(o.z, o.z_denom)),
                ("t", &(o.t, o.t_denom)),
            ] {
                let coords = alt_order_coords_of(o, elem, denom)
                    .unwrap_or_else(|| panic!("order[{k}] must contain {label}"));
                // round-trip: basis·coords / lat_denom == elem (numerator/denom).
                // Cross-multiply to avoid division: basis·coords · elem_denom
                //   == elem_numerator · lat_denom  (componentwise).
                let recon = crate::quaternion::lattice::mat_4x4_eval(&wbasis, &coords);
                let ed = widen_int::<8, 16>(denom);
                let num = [
                    widen_int::<8, 16>(&elem.a),
                    widen_int::<8, 16>(&elem.b),
                    widen_int::<8, 16>(&elem.c),
                    widen_int::<8, 16>(&elem.d),
                ];
                for i in 0..4 {
                    assert_eq!(
                        recon[i].wrapping_mul(&ed),
                        num[i].wrapping_mul(&wdenom),
                        "order[{k}] {label} round-trip failed at coord {i}",
                    );
                }
            }
        }
    }
}
