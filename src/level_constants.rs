// SPDX-License-Identifier: MIT OR Apache-2.0
//! Per-level precomputed-constant provider.
//!
//! [`LevelConstants`] gives the ideal-to-isogeny spine the right security
//! level's precomputed data — the E0 `2^F`-torsion basis, the NICE starting
//! curves with endomorphism rings, the extremal orders, and the connecting
//! ideals — generic over the base field. Level 1 dispatches to the `*_l1`
//! providers; Level 3 to the `*_lvl3` providers. This is the layer that lets
//! the (otherwise field-generic) spine be instantiated per security level.
//!
//! Level 5 is not yet provided.

use crate::ec::montgomery::MontgomeryPoint;
use crate::gf::fp::BaseField;
use crate::isogeny::endomorphism;
use crate::params::Params;
use crate::params::lvl1::{Fp1Element, Level1};
use crate::params::lvl3::{Fp3Element, Level3};
use crate::quaternion::connecting_ideals as ci;
use crate::quaternion::curves_with_endomorphism::{self as cw, CurveWithEndomorphism};
use crate::quaternion::extremal_orders::{self as eo, AltExtremalOrder};
use crate::quaternion::ideal::LeftIdeal;

/// Even-torsion basis `(P, Q, P − Q)` as x-only Montgomery points.
pub type EvenBasis<F> = (MontgomeryPoint<F>, MontgomeryPoint<F>, MontgomeryPoint<F>);

/// Provider of a security level's precomputed isogeny / quaternion constants.
pub trait LevelConstants: Params {
    /// Base field for this level (`Fp1Element` at lvl1, `Fp3Element` at lvl3).
    type Field: BaseField;

    /// E0 `2^F`-torsion basis `(P, Q, P − Q)` as x-only Montgomery points.
    fn basis_e0() -> EvenBasis<Self::Field>;

    /// The E0-with-endomorphism starting curve.
    fn nice_curve_e0() -> CurveWithEndomorphism<Self::Field>;

    /// The `idx`-th NICE alternate starting curve
    /// (`idx < NUM_ALTERNATE_EXTREMAL_ORDERS`).
    fn nice_curve(idx: usize) -> CurveWithEndomorphism<Self::Field>;

    /// The standard order `O_0`.
    fn standard_order_o0() -> AltExtremalOrder;

    /// The `idx`-th alternate extremal order.
    fn alternate_extremal_order(idx: usize) -> AltExtremalOrder;

    /// The `idx`-th alternate connecting ideal.
    fn alternate_connecting_ideal(idx: usize) -> LeftIdeal<8>;

    /// This level's base prime, zero-extended to `N` limbs (`N` must be ≥ the
    /// prime's native limb count: 4 at lvl1, 6 at lvl3). The spine works at
    /// quaternion widths wider than the field, so it requests the prime resized.
    fn prime<const N: usize>() -> crypto_bigint::Uint<N>;
}

impl LevelConstants for Level1 {
    type Field = Fp1Element;

    fn basis_e0() -> EvenBasis<Fp1Element> {
        endomorphism::basis_e0_lvl1()
    }

    fn nice_curve_e0() -> CurveWithEndomorphism<Fp1Element> {
        cw::curve_with_endomorphism_e0_l1()
    }

    fn nice_curve(idx: usize) -> CurveWithEndomorphism<Fp1Element> {
        match idx {
            0 => cw::curve_with_endomorphism_0_l1(),
            1 => cw::curve_with_endomorphism_1_l1(),
            2 => cw::curve_with_endomorphism_2_l1(),
            3 => cw::curve_with_endomorphism_3_l1(),
            4 => cw::curve_with_endomorphism_4_l1(),
            _ => cw::curve_with_endomorphism_5_l1(),
        }
    }

    fn standard_order_o0() -> AltExtremalOrder {
        eo::standard_order_o0_l1()
    }

    fn alternate_extremal_order(idx: usize) -> AltExtremalOrder {
        match idx {
            0 => eo::alternate_extremal_order_0_l1(),
            1 => eo::alternate_extremal_order_1_l1(),
            2 => eo::alternate_extremal_order_2_l1(),
            3 => eo::alternate_extremal_order_3_l1(),
            4 => eo::alternate_extremal_order_4_l1(),
            _ => eo::alternate_extremal_order_5_l1(),
        }
    }

    fn alternate_connecting_ideal(idx: usize) -> LeftIdeal<8> {
        match idx {
            0 => ci::alternate_connecting_ideal_0_l1(),
            1 => ci::alternate_connecting_ideal_1_l1(),
            2 => ci::alternate_connecting_ideal_2_l1(),
            3 => ci::alternate_connecting_ideal_3_l1(),
            4 => ci::alternate_connecting_ideal_4_l1(),
            _ => ci::alternate_connecting_ideal_5_l1(),
        }
    }

    fn prime<const N: usize>() -> crypto_bigint::Uint<N> {
        crate::params::lvl1::prime().resize::<N>()
    }
}

impl LevelConstants for Level3 {
    type Field = Fp3Element;

    fn basis_e0() -> EvenBasis<Fp3Element> {
        endomorphism::basis_e0_lvl3()
    }

    fn nice_curve_e0() -> CurveWithEndomorphism<Fp3Element> {
        cw::curve_with_endomorphism_e0_lvl3()
    }

    fn nice_curve(idx: usize) -> CurveWithEndomorphism<Fp3Element> {
        match idx {
            0 => cw::curve_with_endomorphism_0_lvl3(),
            1 => cw::curve_with_endomorphism_1_lvl3(),
            2 => cw::curve_with_endomorphism_2_lvl3(),
            3 => cw::curve_with_endomorphism_3_lvl3(),
            4 => cw::curve_with_endomorphism_4_lvl3(),
            5 => cw::curve_with_endomorphism_5_lvl3(),
            _ => cw::curve_with_endomorphism_6_lvl3(),
        }
    }

    fn standard_order_o0() -> AltExtremalOrder {
        eo::standard_order_o0_lvl3()
    }

    fn alternate_extremal_order(idx: usize) -> AltExtremalOrder {
        match idx {
            0 => eo::alternate_extremal_order_0_lvl3(),
            1 => eo::alternate_extremal_order_1_lvl3(),
            2 => eo::alternate_extremal_order_2_lvl3(),
            3 => eo::alternate_extremal_order_3_lvl3(),
            4 => eo::alternate_extremal_order_4_lvl3(),
            5 => eo::alternate_extremal_order_5_lvl3(),
            _ => eo::alternate_extremal_order_6_lvl3(),
        }
    }

    fn alternate_connecting_ideal(idx: usize) -> LeftIdeal<8> {
        match idx {
            0 => ci::alternate_connecting_ideal_0_lvl3(),
            1 => ci::alternate_connecting_ideal_1_lvl3(),
            2 => ci::alternate_connecting_ideal_2_lvl3(),
            3 => ci::alternate_connecting_ideal_3_lvl3(),
            4 => ci::alternate_connecting_ideal_4_lvl3(),
            5 => ci::alternate_connecting_ideal_5_lvl3(),
            _ => ci::alternate_connecting_ideal_6_lvl3(),
        }
    }

    fn prime<const N: usize>() -> crypto_bigint::Uint<N> {
        crate::params::lvl3::prime().resize::<N>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Level-3 provider returns lvl3-typed constants for every accessor,
    /// across the full `idx` range — this is what wires the lvl3 precomputed
    /// data into reachable use ahead of the spine lift.
    #[test]
    fn level3_provider_returns_all_lvl3_constants() {
        let (p, q, pmq) = <Level3 as LevelConstants>::basis_e0();
        let one = crate::gf::fp2::Fp2::<Fp3Element>::one();
        assert_eq!(p.z, one);
        assert_eq!(q.z, one);
        assert_eq!(pmq.z, one);

        // E0 + all 7 alternate curves, orders, and ideals resolve.
        let _e0 = <Level3 as LevelConstants>::nice_curve_e0();
        let _o0 = <Level3 as LevelConstants>::standard_order_o0();
        for idx in 0..7 {
            let _c = <Level3 as LevelConstants>::nice_curve(idx);
            let _o = <Level3 as LevelConstants>::alternate_extremal_order(idx);
            let id = <Level3 as LevelConstants>::alternate_connecting_ideal(idx);
            assert_eq!(id.denom, crypto_bigint::Uint::<8>::from_u64(1));
        }
    }
}
