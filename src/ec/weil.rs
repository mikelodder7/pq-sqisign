// SPDX-License-Identifier: MIT OR Apache-2.0
//! Weil pairing `e_{2^e}` on `E[2^e]` via biextension / cubical-torsor
//! arithmetic.
//!
//! Port of the SQIsign C reference `weil` and its cubical helpers
//! (`src/ec/ref/lvlx/biextension.c`). The Clapotis `ideal_to_isogeny`
//! spine uses this to pick which factor of the split codomain
//! `E_1 × E_2` is the image curve: it pairs a transported basis point
//! against the codomain basis and compares the pairing values.
//!
//! The cubical torsor representation gives a `2^e`-ladder for the
//! biextension monodromy `g_{P,Q}^{2^e}`. The Weil pairing combines two
//! monodromies (swapping `P`,`Q`) so the level-1 ambiguity cancels.
//!
//! `cubicalADD` is off by a factor `×4` from the true cubical
//! arithmetic (a documented quirk of the reference); this is invisible
//! to the Weil pairing because of the final ratio, so the port keeps
//! the reference formulas verbatim.
//!
//! Convention note: `weil`'s `pq` argument is `x(P ± Q)` — *either*
//! sign works. The biextension ladder seeds `(PnQ, nQ) = (pq, Q)` and
//! steps with a *fixed* differential `1/x(P)`, consistent when
//! `x(PnQ - nQ) = x(P)`; because x-coordinates are even
//! (`x(R) = x(-R)`), both `PnQ = P + Q` and `PnQ = P - Q` satisfy this.
//! The two choices yield Weil values that are inverses of each other,
//! but every downstream relation used here (antisymmetry, bilinearity,
//! and the spine's factor-selection `w1 == w0^k`) is invariant under
//! `e -> e^{-1}` provided the SAME choice is used throughout. The
//! SQIsign C reference calls `weil(..., basis.PmQ, ...)`, i.e. it
//! passes `x(P - Q)`; callers matching the reference should do the same.

#![allow(dead_code)]

use crate::ec::montgomery::{MontgomeryCurve, MontgomeryPoint};
use crate::gf::fp::BaseField;
use crate::gf::fp2::Fp2;
use subtle::ConditionallySelectable;

/// Inputs for one Weil-pairing computation, all points normalised to
/// `(X/Z : 1)` (cubical arithmetic requires normalised representatives).
struct PairingParams<F: BaseField> {
    /// Points `P`, `Q` have order `2^e`.
    e: u32,
    /// `x(P)` normalised to `(X/Z : 1)`.
    p: MontgomeryPoint<F>,
    /// `x(Q)` normalised to `(X/Z : 1)`.
    q: MontgomeryPoint<F>,
    /// `x(P ± Q)` normalised to `(X/Z : 1)` (either sign; see module note).
    pq: MontgomeryPoint<F>,
    /// `1/x(P) = P.z / P.x`.
    ix_p: Fp2<F>,
    /// `1/x(Q) = Q.z / Q.x`.
    ix_q: Fp2<F>,
    /// `((A + 2) / 4 : 1)`.
    a24: MontgomeryPoint<F>,
}

/// Cubical addition. With `ix_pq = 1/x(P - Q)` (the differential given
/// in inverted form), this is the cubical analogue of `xADD`.
/// Cost: 3M + 2S.
fn cubical_add<F: BaseField>(
    p: &MontgomeryPoint<F>,
    q: &MontgomeryPoint<F>,
    ix_pq: &Fp2<F>,
) -> MontgomeryPoint<F> {
    let t0 = p.x.add(&p.z);
    let t1 = p.x.sub(&p.z);
    let t2 = q.x.add(&q.z);
    let t3 = q.x.sub(&q.z);
    let t0 = t0.mul(&t3);
    let t1 = t1.mul(&t2);
    let t2 = t0.add(&t1);
    let t3 = t0.sub(&t1);
    let r_z = t3.square();
    let t2 = t2.square();
    let r_x = ix_pq.mul(&t2);
    MontgomeryPoint::new(r_x, r_z)
}

/// Combined cubical add and double: given cubical reps of `P`, `Q` and
/// the fixed differential `ix_pq = 1/x(P - Q)`, returns `(P + Q, [2]Q)`.
/// `a24` must be normalised as `((A+2)/4 : 1)`. Cost: 6M + 4S.
fn cubical_dbl_add<F: BaseField>(
    p: &MontgomeryPoint<F>,
    q: &MontgomeryPoint<F>,
    ix_pq: &Fp2<F>,
    a24: &MontgomeryPoint<F>,
) -> (MontgomeryPoint<F>, MontgomeryPoint<F>) {
    let t0 = p.x.add(&p.z);
    let t1 = p.x.sub(&p.z);
    let q_sum = q.x.add(&q.z); // reused as PpQ->x scratch in the C ref
    let t3 = q.x.sub(&q.z);
    let t2 = q_sum.square();
    let qq_z0 = t3.square();
    let t0 = t0.mul(&t3);
    let t1 = t1.mul(&q_sum);
    let ppq_sum = t0.add(&t1);
    let t3 = t0.sub(&t1);
    let ppq_z = t3.square();
    let ppq_x = ppq_sum.square();
    let ppq_x = ix_pq.mul(&ppq_x);
    let t3 = t2.sub(&qq_z0);
    let qq_x = t2.mul(&qq_z0);
    let t0 = t3.mul(&a24.x);
    let t0 = t0.add(&qq_z0);
    let qq_z = t0.mul(&t3);
    (
        MontgomeryPoint::new(ppq_x, ppq_z),
        MontgomeryPoint::new(qq_x, qq_z),
    )
}

/// Iterative biextension doubling: starting from `(PQ, Q)`, apply
/// `cubical_dbl_add` `e` times with the fixed differential `ix_p` to
/// obtain `(P + [2^e]Q, [2^e]Q)`.
fn biext_ladder_2e<F: BaseField>(
    e: u32,
    pq: &MontgomeryPoint<F>,
    q: &MontgomeryPoint<F>,
    ix_p: &Fp2<F>,
    a24: &MontgomeryPoint<F>,
) -> (MontgomeryPoint<F>, MontgomeryPoint<F>) {
    let mut pnq = *pq;
    let mut nq = *q;
    for _ in 0..e {
        let (new_pnq, new_nq) = cubical_dbl_add(&pnq, &nq, ix_p, a24);
        pnq = new_pnq;
        nq = new_nq;
    }
    (pnq, nq)
}

/// Monodromy ratio `X/Z` as an `(X : Z)` point (avoids a division).
/// Implicitly uses `(1, 0)` as the cubical point above `0_E`.
fn point_ratio<F: BaseField>(
    pnq: &MontgomeryPoint<F>,
    nq: &MontgomeryPoint<F>,
    p: &MontgomeryPoint<F>,
) -> MontgomeryPoint<F> {
    MontgomeryPoint::new(nq.x.mul(&p.x), pnq.x)
}

/// Cubical translation of `P` by a 2-torsion point `T`, constant-time.
///
/// - `T = (A : 0)`  → translation is `P`
/// - `T = (0 : B)`  → translation of `(X : Z)` is `(Z : X)`
/// - otherwise      → `(A·X − B·Z : B·X − A·Z)`
fn translate<F: BaseField>(p: &MontgomeryPoint<F>, t: &MontgomeryPoint<F>) -> MontgomeryPoint<F> {
    // Generic case.
    let t0 = t.x.mul(&p.x);
    let t1 = t.z.mul(&p.z);
    let mut px_new = t0.sub(&t1); // A·X − B·Z
    let t0 = t.z.mul(&p.x);
    let t1 = t.x.mul(&p.z);
    let mut pz_new = t0.sub(&t1); // B·X − A·Z

    // T = (A : 0) → return (Z : X).
    let ta_is_zero = t.x.is_zero();
    px_new = Fp2::conditional_select(&px_new, &p.z, ta_is_zero);
    pz_new = Fp2::conditional_select(&pz_new, &p.x, ta_is_zero);

    // T = (0 : B) → return (X : Z).
    let tb_is_zero = t.z.is_zero();
    px_new = Fp2::conditional_select(&px_new, &p.x, tb_is_zero);
    pz_new = Fp2::conditional_select(&pz_new, &p.z, tb_is_zero);

    MontgomeryPoint::new(px_new, pz_new)
}

/// Biextension monodromy `g_{P,Q}^{2^e}` (level 1) via the cubical
/// arithmetic of `P + [2^e]Q`. `swap_pq` swaps the roles of `P` and `Q`
/// (and the corresponding inverse x-coordinate) so a single routine
/// computes both `P + [2^e]Q` and `Q + [2^e]P`.
fn monodromy_i<F: BaseField>(data: &PairingParams<F>, swap_pq: bool) -> MontgomeryPoint<F> {
    let (p, q, ix_p) = if !swap_pq {
        (data.p, data.q, data.ix_p)
    } else {
        (data.q, data.p, data.ix_q)
    };

    let (mut pnq, nq) = biext_ladder_2e(data.e - 1, &data.pq, &q, &ix_p, &data.a24);
    pnq = translate(&pnq, &nq);
    let nq = translate(&nq, &nq);
    point_ratio(&pnq, &nq, &p)
}

/// Normalise `P`, `Q` to `(X/Z : 1)` and compute `1/x(P)`, `1/x(Q)`.
fn cubical_normalization<F: BaseField>(
    p: &MontgomeryPoint<F>,
    q: &MontgomeryPoint<F>,
) -> (Fp2<F>, Fp2<F>, MontgomeryPoint<F>, MontgomeryPoint<F>) {
    let zero = Fp2::<F>::zero();
    let px_inv = p.x.invert().unwrap_or(zero);
    let qx_inv = q.x.invert().unwrap_or(zero);
    let pz_inv = p.z.invert().unwrap_or(zero);
    let qz_inv = q.z.invert().unwrap_or(zero);

    let ix_p = p.z.mul(&px_inv); // P.z / P.x = 1/x(P)
    let ix_q = q.z.mul(&qx_inv);
    let p_n = MontgomeryPoint::new(p.x.mul(&pz_inv), Fp2::one());
    let q_n = MontgomeryPoint::new(q.x.mul(&qz_inv), Fp2::one());
    (ix_p, ix_q, p_n, q_n)
}

/// Weil pairing kernel: combine the two monodromies so the level-1
/// ambiguity cancels. Result `= (R0.z · R1.x) / (R0.x · R1.z)`.
fn weil_n<F: BaseField>(data: &PairingParams<F>) -> Fp2<F> {
    let r0 = monodromy_i(data, true);
    let r1 = monodromy_i(data, false);

    let num = r0.z.mul(&r1.x);
    let den = r0.x.mul(&r1.z);
    num.mul(&den.invert().unwrap_or(Fp2::zero()))
}

/// Compute the Weil pairing `e_{2^e}(P, Q)` on the curve `curve`.
///
/// `pq` is `x(P ± Q)` (either sign — see the module note; the C ref
/// passes `x(P - Q)`). `P`, `Q` must have order `2^e` and be independent
/// for a non-degenerate (primitive `2^e`-th root of unity) result.
/// Crashes-to-garbage (never panics) if either point is the identity,
/// mirroring the reference's "division by 0" precondition.
pub(crate) fn weil<F: BaseField>(
    e: u32,
    p: &MontgomeryPoint<F>,
    q: &MontgomeryPoint<F>,
    pq: &MontgomeryPoint<F>,
    curve: &MontgomeryCurve<F>,
) -> Fp2<F> {
    let (ix_p, ix_q, p_n, q_n) = cubical_normalization(p, q);
    let data = PairingParams {
        e,
        p: p_n,
        q: q_n,
        pq: pq.to_affine(),
        ix_p,
        ix_q,
        a24: MontgomeryPoint::new(curve.a24(), Fp2::one()),
    };
    weil_n(&data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isogeny::endomorphism::basis_e0_lvl1;
    use crate::params::lvl1::Fp1Element;
    use subtle::ConstantTimeEq;

    /// The E0 even-torsion basis has order exactly `2^248`.
    const E: u32 = 248;

    fn e0() -> MontgomeryCurve<Fp1Element> {
        MontgomeryCurve::<Fp1Element>::e0()
    }

    /// `x^(2^k)` by repeated squaring.
    fn pow2k(x: &Fp2<Fp1Element>, k: u32) -> Fp2<Fp1Element> {
        let mut r = *x;
        for _ in 0..k {
            r = r.square();
        }
        r
    }

    /// `e(P,Q)` for the canonical E0[2^248] basis is a *primitive*
    /// `2^248`-th root of unity: it vanishes after 248 squarings but
    /// not after 247. This is the non-degeneracy oracle — it fails for
    /// any pairing that collapses an independent basis.
    #[test]
    fn weil_is_primitive_2e_root_of_unity() {
        let curve = e0();
        let (p, q, pmq) = basis_e0_lvl1();
        let pq = p.x_add(&q, &pmq); // x(P + Q)
        let w = weil(E, &p, &q, &pq, &curve);

        let one = Fp2::<Fp1Element>::one();
        assert!(bool::from(pow2k(&w, E).ct_eq(&one)), "w^(2^248) == 1");
        assert!(
            !bool::from(pow2k(&w, E - 1).ct_eq(&one)),
            "w^(2^247) != 1 (primitive)"
        );
        assert!(!bool::from(w.ct_eq(&one)), "w != 1");
    }

    /// Antisymmetry: `e(Q,P) = e(P,Q)^(-1)`. Independent of the
    /// implementation's internals — a transposition identity.
    #[test]
    fn weil_is_antisymmetric() {
        let curve = e0();
        let (p, q, pmq) = basis_e0_lvl1();
        let pq = p.x_add(&q, &pmq); // x(P + Q) = x(Q + P)
        let w_pq = weil(E, &p, &q, &pq, &curve);
        let w_qp = weil(E, &q, &p, &pq, &curve);

        let inv = w_pq.invert().unwrap();
        assert!(bool::from(w_qp.ct_eq(&inv)), "e(Q,P) == e(P,Q)^-1");
    }

    /// Alternating: `e(P,P) = 1`. Differential is `x(P + P) = x(2P)`.
    #[test]
    fn weil_alternating_self_pairing_is_one() {
        let curve = e0();
        let a24 = curve.a24();
        let (p, _q, _pmq) = basis_e0_lvl1();
        let two_p = p.x_double(&a24); // x(2P)
        let w = weil(E, &p, &p, &two_p, &curve);
        assert!(
            bool::from(w.ct_eq(&Fp2::<Fp1Element>::one())),
            "e(P,P) == 1"
        );
    }

    /// Bilinearity oracle: `e([2]P, Q) = e(P,Q)^2`. The strongest
    /// independent check — a transcription bug in the cubical formulas
    /// breaks the homomorphism even when the root-of-unity order holds.
    /// All differentials are built with x-only arithmetic:
    ///   x(2P)     = xDBL(P)
    ///   x(2P - Q) = xADD(P, P-Q; diff x(Q))
    ///   x(2P + Q) = xADD(2P, Q; diff x(2P-Q))
    #[test]
    fn weil_is_bilinear_in_first_argument() {
        let curve = e0();
        let a24 = curve.a24();
        let (p, q, pmq) = basis_e0_lvl1();

        let pq = p.x_add(&q, &pmq); // x(P + Q)
        let two_p = p.x_double(&a24); // x(2P)
        let two_p_minus_q = p.x_add(&pmq, &q); // x(P + (P-Q)), diff x(Q)
        let two_p_plus_q = two_p.x_add(&q, &two_p_minus_q); // x(2P + Q), diff x(2P-Q)

        let lhs = weil(E, &two_p, &q, &two_p_plus_q, &curve);
        let rhs = weil(E, &p, &q, &pq, &curve).square();
        assert!(bool::from(lhs.ct_eq(&rhs)), "e([2]P,Q) == e(P,Q)^2");
    }
}
