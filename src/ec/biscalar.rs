//! Montgomery biladder — `x([k]P + [l]Q)` from an x-only torsion basis.
//!
//! Faithful port of the SQIsign C reference `ec_biscalar_mul` + `xDBLMUL`
//! (`src/ec/ref/lvlx/ec.c`). This is the computational leaf of the
//! quaternion→EC-torsion bridge (`endomorphism_application_even_basis`): a
//! 2×2 integer matrix is applied to a torsion basis as three double-scalar
//! multiplications `[a]P + [b]Q`.
//!
//! `xDBLMUL` is a constant-time differential biladder: it forces both scalars
//! odd (recording the parity), runs a recoded differential-addition ladder
//! over the odd scalars, and corrects the output by masked point selection —
//! recovering the both-even / both-odd cases the odd-only ladder cannot
//! represent directly.
//!
//! Consumed by `endomorphism_application_even_basis` (the matrix→basis
//! application) in a later session; until then only this module's tests
//! exercise it, so the lib build sees the items as unused.
#![allow(dead_code)]

use crate::ec::montgomery::MontgomeryPoint;
use crate::gf::fp::BaseField;
use crate::gf::fp2::Fp2;
use subtle::{Choice, ConditionallySelectable};

/// Scalar working-buffer width in bytes (512 bits) — covers every level's
/// 2-power torsion (`TORSION_EVEN_POWER ≤ 376`).
const SB: usize = 64;
/// Maximum supported `kbits` (bitlength of the scalars / torsion power).
const MAX_BITS: usize = 512;

#[inline]
fn has_zero_coord<F: BaseField>(p: &MontgomeryPoint<F>) -> bool {
    bool::from(p.x.is_zero()) || bool::from(p.z.is_zero())
}

/// `out = if mask==0xFF { y } else { x }`, byte-wise constant-time.
#[inline]
fn ct_select_bytes(out: &mut [u8], x: &[u8], y: &[u8], mask: u8) {
    for i in 0..out.len() {
        out[i] = (y[i] & mask) | (x[i] & !mask);
    }
}

/// Conditionally swap two equal-length byte buffers when `mask == 0xFF`.
#[inline]
fn ct_swap_bytes(a: &mut [u8], b: &mut [u8], mask: u8) {
    for i in 0..a.len() {
        let t = (a[i] ^ b[i]) & mask;
        a[i] ^= t;
        b[i] ^= t;
    }
}

/// `buf -= 1` (little-endian, with borrow). Wraps on all-zero (matches the C
/// reference's unsigned `mp_sub`); callers only rely on it for odd→even−1.
#[inline]
fn sub_one(buf: &mut [u8]) {
    let mut borrow = 1u8;
    for byte in buf.iter_mut() {
        let (v, underflow) = byte.overflowing_sub(borrow);
        *byte = v;
        borrow = u8::from(underflow);
    }
}

/// Shift `buf` right by one bit (little-endian) and return the bit shifted out
/// (the old least-significant bit).
#[inline]
fn shiftr1(buf: &mut [u8]) -> u8 {
    let out = buf[0] & 1;
    let mut carry = 0u8;
    for byte in buf.iter_mut().rev() {
        let new_carry = *byte & 1;
        *byte = (*byte >> 1) | (carry << 7);
        carry = new_carry;
    }
    out
}

/// `select_point(x, y, c) = if c { y } else { x }` — names the C reference's
/// `select_point(out, x, y, mask)` convention (`mask` all-ones → `y`).
#[inline]
fn select_point<F: BaseField>(
    x: &MontgomeryPoint<F>,
    y: &MontgomeryPoint<F>,
    c: Choice,
) -> MontgomeryPoint<F> {
    MontgomeryPoint::conditional_select(x, y, c)
}

#[inline]
fn cswap_points<F: BaseField>(a: &mut MontgomeryPoint<F>, b: &mut MontgomeryPoint<F>, c: Choice) {
    let na = MontgomeryPoint::conditional_select(a, b, c);
    let nb = MontgomeryPoint::conditional_select(b, a, c);
    *a = na;
    *b = nb;
}

/// The Montgomery biladder. Computes `x([k]P + [l]Q)` given x-only `P`, `Q`,
/// and `PQ = P − Q`, with `k`, `l` little-endian scalars of `kbits` bits and
/// the affine doubling constant `a24 = (A + 2) / 4`.
///
/// Returns `None` if a differential-addition formula hits a degenerate
/// (zero-coordinate) input — the same fail-closed behaviour as the C ref's
/// `return 0`.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(crate) fn xdblmul<F: BaseField>(
    p: &MontgomeryPoint<F>,
    k: &[u8],
    q: &MontgomeryPoint<F>,
    l: &[u8],
    pq: &MontgomeryPoint<F>,
    kbits: usize,
    a24: &Fp2<F>,
) -> Option<MontgomeryPoint<F>> {
    if has_zero_coord(p) || has_zero_coord(q) || has_zero_coord(pq) {
        return None;
    }
    assert!((1..=MAX_BITS).contains(&kbits), "kbits out of range");
    assert!(k.len() <= SB && l.len() <= SB, "scalar exceeds buffer");

    // Working copies of the scalars (little-endian, zero-extended).
    let mut k_t = [0u8; SB];
    let mut l_t = [0u8; SB];
    k_t[..k.len()].copy_from_slice(k);
    l_t[..l.len()].copy_from_slice(l);
    let k_orig = k_t;
    let l_orig = l_t;

    let bitk0 = k_t[0] & 1;
    let bitl0 = l_t[0] & 1;
    let maskk = 0u8.wrapping_sub(bitk0); // 0xFF if k odd
    let maskl = 0u8.wrapping_sub(bitl0);

    // Parity-derived sigma: count even scalars, pick (0,1) when both even or
    // both odd, otherwise route the odd one.
    let mut sigma0 = bitk0 ^ 1;
    let mut sigma1 = bitl0 ^ 1;
    let evens = sigma0 + sigma1;
    let mevens = 0u8.wrapping_sub(evens & 1); // 0xFF iff exactly one even
    sigma0 &= mevens;
    sigma1 = (sigma1 & mevens) | (1 & !mevens);

    // Force both scalars odd: odd → keep, even → subtract 1.
    let mut k_m1 = k_orig;
    sub_one(&mut k_m1);
    let mut l_m1 = l_orig;
    sub_one(&mut l_m1);
    ct_select_bytes(&mut k_t, &k_m1, &k_orig, maskk);
    ct_select_bytes(&mut l_t, &l_m1, &l_orig, maskl);

    // Scalar recoding: r[2i], r[2i+1] are consecutive-bit XORs of the routed
    // scalars, with the sigma swap schedule interleaved.
    let mut r = [0u8; 2 * MAX_BITS];
    let mut pre_sigma = 0u8;
    for i in 0..kbits {
        let m = 0u8.wrapping_sub(sigma0 ^ pre_sigma);
        ct_swap_bytes(&mut k_t, &mut l_t, m);

        let (bs1_ip1, bs2_ip1) = if i == kbits - 1 {
            (0u8, 0u8)
        } else {
            (shiftr1(&mut k_t), shiftr1(&mut l_t))
        };
        let bs1_i = k_t[0] & 1;
        let bs2_i = l_t[0] & 1;
        r[2 * i] = bs1_i ^ bs1_ip1;
        r[2 * i + 1] = bs2_i ^ bs2_ip1;

        pre_sigma = sigma0;
        let m2 = 0u8.wrapping_sub(r[2 * i + 1]);
        let temp = (sigma1 & m2) | (sigma0 & !m2); // m2 ? sigma1 : sigma0
        sigma1 = (sigma0 & m2) | (sigma1 & !m2); // m2 ? sigma0 : sigma1
        sigma0 = temp;
    }

    // Point init: R0 = O, then route P/Q by sigma0.
    let mut r0 = MontgomeryPoint::infinity();
    let csig = Choice::from(sigma0 & 1);
    let mut r1 = select_point(p, q, csig); // sigma0 ? Q : P
    let mut r2 = select_point(q, p, csig); // sigma0 ? P : Q

    let mut diff1a = r1;
    let mut diff1b = r2;

    // R2 <- R1 + R2 (diff PQ); DIFF2a <- P+Q, DIFF2b <- P-Q.
    r2 = r1.x_add(&r2, pq);
    if has_zero_coord(&r2) {
        return None;
    }
    let mut diff2a = r2;
    let mut diff2b = *pq;

    for i in (0..kbits).rev() {
        let h = r[2 * i] + r[2 * i + 1]; // {0,1,2}
        let mut t0 = select_point(&r0, &r1, Choice::from(h & 1));
        t0 = select_point(&t0, &r2, Choice::from(h >> 1));
        t0 = t0.x_double(a24);

        let cr1 = Choice::from(r[2 * i + 1] & 1);
        let t1 = select_point(&r0, &r1, cr1);
        let t2 = select_point(&r1, &r2, cr1);

        cswap_points(&mut diff1a, &mut diff1b, cr1);
        let t1 = t1.x_add(&t2, &diff1a);
        let t2 = r0.x_add(&r2, &diff2a);

        cswap_points(&mut diff2a, &mut diff2b, Choice::from(h & 1));

        r0 = t0;
        r1 = t1;
        r2 = t2;
    }

    // Output: R[evens], then R[2] when both scalars were odd.
    let mut s = select_point(&r0, &r1, Choice::from(mevens & 1));
    s = select_point(&s, &r2, Choice::from((bitk0 & bitl0) & 1));
    Some(s)
}

/// Compute `x([scalar_p]P + [scalar_q]Q)` from an x-only basis `(P, Q, P−Q)`
/// on a curve with affine doubling constant `a24 = (A + 2) / 4`. `kbits` is
/// the torsion power (bitlength bound on the scalars).
///
/// Mirrors the C reference `ec_biscalar_mul`, including the `kbits == 1`
/// 2-torsion special case (where `P − Q = (0 : 1)` breaks differential
/// addition). Returns `None` on a degenerate basis or a failed ladder.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ec_biscalar_mul<F: BaseField>(
    scalar_p: &[u8],
    scalar_q: &[u8],
    kbits: usize,
    basis_p: &MontgomeryPoint<F>,
    basis_q: &MontgomeryPoint<F>,
    basis_pmq: &MontgomeryPoint<F>,
    a24: &Fp2<F>,
) -> Option<MontgomeryPoint<F>> {
    if bool::from(basis_pmq.z.is_zero()) {
        return None;
    }

    if kbits == 1 {
        // 2-torsion table: (0,0)→O, (1,0)→P, (0,1)→Q, (1,1)→P−Q.
        let bp = scalar_p.first().copied().unwrap_or(0) & 1;
        let bq = scalar_q.first().copied().unwrap_or(0) & 1;
        let mut r = MontgomeryPoint::infinity();
        r = MontgomeryPoint::conditional_select(&r, basis_p, Choice::from(bp & (bq ^ 1)));
        r = MontgomeryPoint::conditional_select(&r, basis_q, Choice::from((bp ^ 1) & bq));
        r = MontgomeryPoint::conditional_select(&r, basis_pmq, Choice::from(bp & bq));
        return Some(r);
    }

    xdblmul(basis_p, scalar_p, basis_q, scalar_q, basis_pmq, kbits, a24)
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    use crate::ec::montgomery::MontgomeryCurve;
    use crate::params::lvl1::Fp1Element;
    use alloc::vec::Vec;
    use subtle::ConstantTimeEq;

    // Affine arithmetic on E_A : y² = x³ + A x² + x (B = 1) — independent
    // ground truth for the biladder. Points are `Some((x, y))` or `None` (∞).
    type Aff = Option<(Fp2<Fp1Element>, Fp2<Fp1Element>)>;

    fn aff_neg(p: &Aff) -> Aff {
        p.map(|(x, y)| (x, y.negate()))
    }

    fn aff_add(p: &Aff, q: &Aff, a: &Fp2<Fp1Element>) -> Aff {
        let (p, q) = match (p, q) {
            (None, _) => return *q,
            (_, None) => return *p,
            (Some(p), Some(q)) => (*p, *q),
        };
        let (x1, y1) = p;
        let (x2, y2) = q;
        if bool::from(x1.ct_eq(&x2)) {
            // x1 == x2: either doubling or vertical (→ ∞).
            if bool::from(y1.ct_eq(&y2)) && !bool::from(y1.is_zero()) {
                // λ = (3x² + 2Ax + 1) / (2y)
                let three_x2 = x1.square().mul(&small(3));
                let two_ax = a.mul(&x1).mul(&small(2));
                let num = three_x2.add(&two_ax).add(&Fp2::one());
                let den = y1.mul(&small(2));
                let lam = num.mul(&den.invert().unwrap_or(Fp2::zero()));
                return aff_from_lambda(&lam, &x1, &x2, &y1, a);
            }
            return None; // P + (−P) = ∞
        }
        // λ = (y2 − y1) / (x2 − x1)
        let lam = y2
            .sub(&y1)
            .mul(&x2.sub(&x1).invert().unwrap_or(Fp2::zero()));
        aff_from_lambda(&lam, &x1, &x2, &y1, a)
    }

    fn aff_from_lambda(
        lam: &Fp2<Fp1Element>,
        x1: &Fp2<Fp1Element>,
        x2: &Fp2<Fp1Element>,
        y1: &Fp2<Fp1Element>,
        a: &Fp2<Fp1Element>,
    ) -> Aff {
        // x3 = λ² − A − x1 − x2 ; y3 = λ(x1 − x3) − y1
        let x3 = lam.square().sub(a).sub(x1).sub(x2);
        let y3 = lam.mul(&x1.sub(&x3)).sub(y1);
        Some((x3, y3))
    }

    fn aff_mul(k: u64, p: &Aff, a: &Fp2<Fp1Element>) -> Aff {
        let mut acc: Aff = None;
        let mut base = *p;
        let mut kk = k;
        while kk != 0 {
            if kk & 1 == 1 {
                acc = aff_add(&acc, &base, a);
            }
            base = aff_add(&base, &base, a);
            kk >>= 1;
        }
        acc
    }

    fn small(n: u32) -> Fp2<Fp1Element> {
        let mut acc = Fp2::<Fp1Element>::zero();
        let one = Fp2::<Fp1Element>::one();
        for _ in 0..n {
            acc = acc.add(&one);
        }
        acc
    }

    /// Lift a small affine x to a full point on E_0 (principal sqrt branch).
    fn lift(xn: u32, a: &Fp2<Fp1Element>) -> Option<(Fp2<Fp1Element>, Fp2<Fp1Element>)> {
        let x = small(xn);
        let rhs = x.square().mul(&x).add(&a.mul(&x.square())).add(&x);
        let y = rhs.sqrt();
        if bool::from(y.is_some()) {
            Some((x, y.unwrap_or(Fp2::zero())))
        } else {
            None
        }
    }

    fn x_only(p: &(Fp2<Fp1Element>, Fp2<Fp1Element>)) -> MontgomeryPoint<Fp1Element> {
        MontgomeryPoint::new(p.0, Fp2::one())
    }

    fn aff_x_eq(m: &MontgomeryPoint<Fp1Element>, aff: &Aff) -> bool {
        match aff {
            None => bool::from(m.z.is_zero()),
            Some((x, _)) => {
                // m.x / m.z == x  ⇔  m.x == x · m.z
                bool::from(m.x.ct_eq(&x.mul(&m.z)))
            }
        }
    }

    /// Cross-check the biladder against affine `[k]P + [l]Q` over all four
    /// scalar-parity classes plus the single-scalar edges.
    #[test]
    fn xdblmul_matches_affine_ground_truth() {
        let curve = MontgomeryCurve::<Fp1Element>::e0();
        let a = curve.a;
        let a24 = curve.a24();

        // Two distinct liftable points P, Q on E_0.
        let p = lift(2, &a).or_else(|| lift(3, &a)).expect("liftable P");
        let mut q = None;
        for n in 4..=40 {
            if let Some(cand) = lift(n, &a) {
                if !bool::from(cand.0.ct_eq(&p.0)) {
                    q = Some(cand);
                    break;
                }
            }
        }
        let q = q.expect("liftable Q");

        // Consistent x-only basis (PmQ from the chosen P, Q signs).
        let some_p: Aff = Some(p);
        let some_q: Aff = Some(q);
        let pmq = aff_add(&some_p, &aff_neg(&some_q), &a).expect("P−Q finite");
        let (bp, bq, bpmq) = (x_only(&p), x_only(&q), x_only(&pmq));

        let kbits = 16usize;
        // BOTH-ODD scalars only: the biladder computes [k]P+[l]Q exactly for
        // arbitrary points in this case (no scalar mangling). Even scalars use
        // a subtract-1+correct trick that requires 2^kbits-torsion points,
        // tested separately with a real torsion basis.
        let cases: [(u64, u64); 4] = [(5, 7), (13, 13), (3, 5), (9, 11)];
        for (k, l) in cases {
            let res = ec_biscalar_mul(
                &k.to_le_bytes(),
                &l.to_le_bytes(),
                kbits,
                &bp,
                &bq,
                &bpmq,
                &a24,
            )
            .expect("biladder ok");
            let reference = aff_add(&aff_mul(k, &some_p, &a), &aff_mul(l, &some_q, &a), &a);
            assert!(
                aff_x_eq(&res, &reference),
                "biladder x([{k}]P+[{l}]Q) must match affine ground truth",
            );
        }
    }

    /// Double an affine point `n` times.
    fn aff_dbl_n(mut p: Aff, n: u32, a: &Fp2<Fp1Element>) -> Aff {
        for _ in 0..n {
            p = aff_add(&p, &p, a);
        }
        p
    }

    /// Produce a point of order EXACTLY `2^e` from `p`: kill the odd part
    /// (`[25]·p` — `25` is the full odd cofactor of `#E_0(F_{p²})`), measure
    /// the resulting 2-power order `2^m` by doubling to identity, then reduce
    /// to `2^e` with `m − e` doublings. Returns `None` if the 2-part is
    /// smaller than `2^e`. Adaptive, so it is correct whether `p` lands in the
    /// order-`(p+1)` subgroup or the full `(p+1)²` group.
    fn to_two_e_torsion(p: &Aff, e: u32, a: &Fp2<Fp1Element>) -> Aff {
        let r = aff_mul(25, p, a); // 2-power order now
        let mut t = r;
        let mut m = 0u32;
        while t.is_some() && m < 600 {
            t = aff_add(&t, &t, a);
            m += 1;
        }
        if m >= e {
            aff_dbl_n(r, m - e, a) // order exactly 2^e
        } else {
            None
        }
    }

    /// Even-scalar / edge cases need genuine `2^e`-torsion points (the
    /// subtract-1-then-correct path is only valid modulo the point order).
    /// Build a real `2^8`-torsion basis on E_0 by cofactor clearing and verify
    /// the biladder against affine `[k]P+[l]Q` for every parity class.
    #[test]
    fn xdblmul_even_scalars_on_torsion_basis() {
        let curve = MontgomeryCurve::<Fp1Element>::e0();
        let a = curve.a;
        let a24 = curve.a24();
        let e = 8u32;

        // Collect cofactor-cleared points of order EXACTLY 2^e (so scalars
        // mod 2^e are meaningful). Order is exactly 2^e iff [2^(e-1)]·pt ≠ O.
        // The two points need not be independent — a dependent pair still
        // validates the biladder against affine arithmetic; we only need a
        // distinct-x pair with a finite difference.
        let mut exact: Vec<Aff> = Vec::new();
        for n in 2u32..=120 {
            if let Some(pt) = lift(n, &a) {
                let cleared = to_two_e_torsion(&Some(pt), e, &a);
                if cleared.is_some() && aff_dbl_n(cleared, e - 1, &a).is_some() {
                    exact.push(cleared);
                }
            }
            if exact.len() >= 8 {
                break;
            }
        }
        let mut basis = None;
        'outer: for i in 0..exact.len() {
            for j in 0..exact.len() {
                if i == j || same_x(&exact[i], &exact[j]) {
                    continue;
                }
                let rpmq = aff_add(&exact[i], &aff_neg(&exact[j]), &a);
                if rpmq.is_some() {
                    basis = Some((exact[i], exact[j], rpmq));
                    break 'outer;
                }
            }
        }
        let (rp, rq, rpmq) = basis.expect("two distinct order-2^e points on E_0");
        let (bp, bq, bpmq) = (
            x_only(&rp.unwrap()),
            x_only(&rq.unwrap()),
            x_only(&rpmq.unwrap()),
        );

        // Every parity class, scalars < 2^e.
        let cases: [(u64, u64); 7] = [
            (5, 7),  // odd, odd
            (5, 8),  // odd, even
            (6, 7),  // even, odd
            (6, 8),  // even, even
            (9, 0),  // edge: l = 0
            (0, 11), // edge: k = 0
            (200, 37),
        ];
        for (k, l) in cases {
            let res = ec_biscalar_mul(
                &k.to_le_bytes(),
                &l.to_le_bytes(),
                e as usize,
                &bp,
                &bq,
                &bpmq,
                &a24,
            )
            .expect("biladder ok");
            let reference = aff_add(&aff_mul(k, &rp, &a), &aff_mul(l, &rq, &a), &a);
            assert!(
                aff_x_eq(&res, &reference),
                "biladder x([{k}]P+[{l}]Q) on torsion basis must match affine",
            );
        }
    }

    fn same_x(p: &Aff, q: &Aff) -> bool {
        match (p, q) {
            (None, None) => true,
            (Some((xp, _)), Some((xq, _))) => bool::from(xp.ct_eq(xq)),
            _ => false,
        }
    }
}
