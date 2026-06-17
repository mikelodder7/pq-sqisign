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

    // C `xDBLMUL` uses the `xDBL_E0` variant when `A == 0` (E0), which yields a
    // DIFFERENT projective representative (2× per step) than `xDBL_A24`. `A == 0`
    // ⟺ `a24 = (A+2)/4 = 1/2` ⟺ `2·a24 = 1`. The curve is public, so this branch
    // is not secret-dependent. Matching the variant is required for byte-exact
    // E0 biscalar outputs.
    let is_e0 = a24.double() == Fp2::<F>::one();

    for i in (0..kbits).rev() {
        let h = r[2 * i] + r[2 * i + 1]; // {0,1,2}
        let mut t0 = select_point(&r0, &r1, Choice::from(h & 1));
        t0 = select_point(&t0, &r2, Choice::from(h >> 1));
        t0 = if is_e0 {
            t0.x_double_e0()
        } else {
            t0.x_double(a24)
        };

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

/// Clear the odd cofactor from a point so it has maximal even order, then
/// double down to exact order `2^f`. Port of the C reference
/// `clear_cofactor_for_maximal_even_order` (`src/ec/ref/lvlx/basis.c`).
///
/// lvl1 specifics: `#E(F_{p²}) = (p+1)² = (5·2^248)²`, so the odd cofactor is
/// `p_cofactor_for_2f = 5` and `TORSION_EVEN_POWER = 248`. We multiply by 5
/// (leaving order dividing `2^248`) then double `248 − f` times. A
/// foundational primitive for `ec_curve_to_basis_2f[_to_hint]` (keygen
/// `hint_pk` + verify challenge/aux bases).
#[allow(dead_code)]
pub(crate) fn clear_cofactor_for_maximal_even_order<F: BaseField>(
    p: &MontgomeryPoint<F>,
    curve: &crate::ec::montgomery::MontgomeryCurve<F>,
    f: usize,
) -> MontgomeryPoint<F> {
    const TORSION_EVEN_POWER: usize = 248;
    const ODD_COFACTOR: u8 = 5; // p_cofactor_for_2f, lvl1
    debug_assert!(f <= TORSION_EVEN_POWER);
    let a24 = curve.a24();
    // [5]·P clears the odd cofactor (order now divides 2^248). C `ec_mul` uses
    // kbits = bitlength(cofactor); the exact bitlength (3 for 5) is REQUIRED so
    // the projective representative matches C — leading-zero ladder iterations
    // would change the representative and break the representative-fragile
    // `difference_point` in the verify basis recompute.
    let cof_bits = (8 - ODD_COFACTOR.leading_zeros()) as usize;
    let p_cleared = p.ladder_nbits(&[ODD_COFACTOR], cof_bits, &a24);
    // Double down to exact order 2^f.
    let n_dbl = u32::try_from(TORSION_EVEN_POWER - f).expect("f <= TORSION_EVEN_POWER ⇒ fits u32");
    curve.to_a24().x_double_n(&p_cleared, n_dbl)
}

/// Build the base-field element equal to the small non-negative integer `n`
/// (double-and-add from `one()`). Port of the C reference `fp_set_small`,
/// used by the entangled-basis NQR-factor search.
#[allow(dead_code)]
pub(crate) fn fp_small<F: BaseField>(n: u32) -> F {
    let mut acc = F::zero();
    let mut base = F::one();
    let mut k = n;
    while k > 0 {
        if k & 1 == 1 {
            acc = acc.add(&base);
        }
        base = base.double();
        k >>= 1;
    }
    acc
}

/// `y · n` for a small scalar `n` (double-and-add). Port of the C reference
/// `fp2_mul_small` used by the entangled-basis x(P) selectors.
#[allow(dead_code)]
pub(crate) fn fp2_mul_small<F: BaseField>(y: &Fp2<F>, n: u32) -> Fp2<F> {
    let mut acc = Fp2::<F>::zero();
    let mut base = *y;
    let mut k = n;
    while k > 0 {
        if k & 1 == 1 {
            acc = acc.add(&base);
        }
        base = base.double();
        k >>= 1;
    }
    acc
}

/// `Choice::TRUE` iff affine `x` is a valid x-coordinate on the (normalized,
/// `C = 1`) Montgomery curve `y² = x³ + A x² + x`. Port of C `is_on_curve`
/// (`basis.c`): returns `is_square(x³ + A·x² + x)`.
#[allow(dead_code)]
pub(crate) fn is_on_curve<F: BaseField>(x: &Fp2<F>, curve_a: &Fp2<F>) -> Choice {
    // ((x + A)·x + 1)·x = x³ + A·x² + x.
    let t0 = x.add(curve_a).mul(x).add(&Fp2::<F>::one()).mul(x);
    t0.is_square()
}

/// Find a valid x(P) of the form `n·A` for entangled-basis generation when
/// `A` is a non-quadratic-residue. Port of C `find_nA_x_coord` (`basis.c`):
/// start at `n = start`, set `x = n·A`, increment `x` by `A` until
/// `is_on_curve(x)`. Returns `(x, hint)` where `hint = n` if `n < 128`, else
/// `0` (the rare "not found in 127 tries" fallback signal). Caller must ensure
/// `A` is a NQR (C `find_nA_x_coord`).
#[allow(dead_code)]
pub(crate) fn find_na_x_coord<F: BaseField>(curve_a: &Fp2<F>, start: u8) -> (Fp2<F>, u8) {
    let mut n: u32 = u32::from(start);
    let mut x = fp2_mul_small(curve_a, n);
    let mut guard = 0u32;
    while !bool::from(is_on_curve(&x, curve_a)) {
        x = x.add(curve_a);
        n += 1;
        guard += 1;
        // ~2^16 attempts before giving up (cryptographically negligible).
        if guard > 65_535 {
            break;
        }
    }
    let hint = if n < 128 {
        u8::try_from(n).expect("n < 128 ⇒ fits u8")
    } else {
        0
    };
    (x, hint)
}

/// Find a valid x(P) of the form `−A/(1 + i·b)` for entangled-basis generation
/// when `A` is a quadratic residue. Port of C `find_nqr_factor` (`basis.c`):
/// search `b = n − 1 ≥ start − 1` for the first `b` with `1 + b²` a non-residue
/// in `Fp` (so `1 + i·b` is a non-residue in `Fp2`) AND with `−A/(1 + i·b)` on
/// the curve, equivalently `A²·(z − 1) − z²` a non-residue for `z = 1 + i·b`
/// (avoids an inversion pre-check). Returns `(x, hint)` where `hint = b` if
/// `n ≤ 128`, else `0` (the rare "not found" fallback signal). Caller must
/// ensure `A` is a QR (C `find_nqr_factor`).
#[allow(dead_code)]
pub(crate) fn find_nqr_factor<F: BaseField>(curve_a: &Fp2<F>, start: u8) -> (Fp2<F>, u8) {
    let a2 = curve_a.square();
    let mut n: u32 = u32::from(start);
    let mut guard = 0u32;
    let z = loop {
        // Advance n until 1 + b² (b = n − 1) is a non-residue in Fp.
        loop {
            let tmp = fp_small::<F>(n.wrapping_mul(n).wrapping_add(1));
            let is_sq = bool::from(tmp.is_square());
            n += 1; // tracks b = n − 1
            if !is_sq {
                break;
            }
            guard += 1;
            if guard > 65_535 {
                break;
            }
        }
        let b = fp_small::<F>(n - 1);
        let zc = Fp2::<F>::new(F::one(), b); // z = 1 + i·b
        let zm1 = Fp2::<F>::new(F::zero(), b); // z − 1 = i·b
        // A²·(z − 1) − z²  non-residue  ⇔  x = −A/z on the curve.
        let t0 = zm1.mul(&a2).sub(&zc.square());
        guard += 1;
        if !bool::from(t0.is_square()) || guard > 65_535 {
            break zc;
        }
    };
    // x = −A / z
    let inv = z.invert().into_option().unwrap_or_else(Fp2::<F>::zero);
    let x = curve_a.mul(&inv).negate();
    let hint = if n <= 128 {
        u8::try_from(n - 1).expect("n ≤ 128 ⇒ n−1 fits u8")
    } else {
        0
    };
    (x, hint)
}

/// Deterministic x-only point difference `x(P − Q)` from `x(P)`, `x(Q)`.
/// Port of the C reference `difference_point` (`src/ec/ref/lvlx/basis.c`,
/// Prop. 3 of eprint 2017/518), specialized to normalized curves (`C = 1`;
/// `curve_a` is the affine `A`). The canonical `Fp2::sqrt` sign (S347 fix)
/// makes the deterministic root choice match C. Used by `ec_curve_to_basis_2f`
/// to set `PmQ` (which fixes `Q` above `(0,0)`).
#[allow(dead_code)]
pub(crate) fn difference_point<F: BaseField>(
    p: &MontgomeryPoint<F>,
    q: &MontgomeryPoint<F>,
    curve_a: &Fp2<F>,
) -> MontgomeryPoint<F> {
    let t0 = p.x.mul(&q.x); // P.x·Q.x
    let t1 = p.z.mul(&q.z); // P.z·Q.z
    let bxx = t0.sub(&t1).square(); // C·(P.x·Q.x − P.z·Q.z)²   (C = 1)
    let pxqz = p.x.mul(&q.z);
    let pzqx = p.z.mul(&q.x);
    // C·(P.x·Q.x + P.z·Q.z)(P.x·Q.z + P.z·Q.x) + 2A·(P.x·Q.z)(P.z·Q.x)
    let bxz = t0
        .add(&t1)
        .mul(&pxqz.add(&pzqx))
        .add(&pxqz.mul(&pzqx).mul(curve_a).double());
    let bzz = pxqz.sub(&pzqx).square(); // C·(P.x·Q.z − P.z·Q.x)²   (C = 1)
    // Normalize by conj(C)²·C·conj(P.z)²·conj(Q.z)²  →  conj(P.z)²·conj(Q.z)² (C=1)
    // so the denominator is a fourth power (conj = Fp2 Frobenius).
    let zn = p.z.frobenius().square().mul(&q.z.frobenius().square());
    let bxx = bxx.mul(&zn);
    let bxz = bxz.mul(&zn);
    let bzz = bzz.mul(&zn);
    // Solve the quadratic: PQ.x = Bxz + sqrt(Bxz² − Bxx·Bzz), PQ.z = Bzz.
    let disc = bxz.square().sub(&bxx.mul(&bzz));
    let s = disc.sqrt().into_option().unwrap_or_else(Fp2::<F>::zero);
    let out = MontgomeryPoint::new(bxz.add(&s), bzz);
    #[cfg(feature = "alloc")]
    if std::env::var("PQSQ_DUMP_DP").is_ok() {
        let mut b = [0u8; 64];
        for (nm, v) in [
            ("px", &p.x),
            ("pz", &p.z),
            ("qx", &q.x),
            ("qz", &q.z),
            ("outx", &out.x),
            ("outz", &out.z),
        ] {
            v.to_bytes_le(&mut b);
            std::eprint!("DP {nm}=");
            for x in &b[..16] {
                std::eprint!("{x:02x}");
            }
            std::eprint!(" ");
        }
        std::eprintln!();
    }
    out
}

/// Entangled basis for `E0` (`A = 0`), where the QR/NQR x-selectors do not
/// apply. Port of C `ec_basis_E0_2f`: start from the precomputed full-order
/// `E0` basis points, double both down to order `2^f`, and recompute `x(P−Q)`.
/// lvl1-pinned (uses the lvl1 precomputed basis and `TORSION_EVEN_POWER = 248`).
pub(crate) fn ec_basis_e0_2f(
    f: usize,
) -> crate::ec::couple::EcBasis<crate::params::lvl1::Fp1Element> {
    use crate::isogeny::endomorphism::basis_e0_lvl1;
    const TORSION_EVEN_POWER: usize = 248;
    let (p, q, _) = basis_e0_lvl1();
    let curve = crate::ec::montgomery::MontgomeryCurve::new(Fp2::zero());
    let a24 = curve.to_a24();
    let n = u32::try_from(TORSION_EVEN_POWER - f).expect("f ≤ TORSION_EVEN_POWER");
    let p = a24.x_double_n(&p, n);
    let q = a24.x_double_n(&q, n);
    let pmq = difference_point(&p, &q, &curve.a);
    crate::ec::couple::EcBasis::new(p, q, pmq)
}

/// Build a deterministic torsion basis `E[2^f] = <P, Q>` with `Q` above
/// `(0 : 0)`, returning a compressed `u8` hint for fast recomputation. Port of
/// C `ec_curve_to_basis_2f_to_hint` (`basis.c`). The Rust `MontgomeryCurve`
/// already stores affine `A` (`C = 1`), so the C `ec_normalize_curve_and_A24`
/// step is a no-op here. lvl1-pinned via `clear_cofactor_for_maximal_even_order`
/// and `ec_basis_e0_2f`.
///
/// Hint layout: `(hint_P << 1) | hint_A`, where `hint_A` is the LSB recording
/// whether `A` is a QR (which x-selector was used) and `hint_P` is the 7-bit
/// selector hint. The stored basis relabels so that `p_minus_q` holds the point
/// above `(0 : 0)`: `q = x(P − Q_raw)`, `p_minus_q = Q_raw`.
pub(crate) fn ec_curve_to_basis_2f_to_hint(
    curve: &crate::ec::montgomery::MontgomeryCurve<crate::params::lvl1::Fp1Element>,
    f: usize,
) -> (
    crate::ec::couple::EcBasis<crate::params::lvl1::Fp1Element>,
    u8,
) {
    use crate::params::lvl1::Fp1Element;
    let a = curve.a;
    if bool::from(a.is_zero()) {
        return (ec_basis_e0_2f(f), 0);
    }
    let hint_a = bool::from(a.is_square());
    let (px, hint) = if hint_a {
        find_nqr_factor(&a, 1)
    } else {
        find_na_x_coord(&a, 1)
    };
    let one = Fp2::<Fp1Element>::one();
    let p_raw = MontgomeryPoint::new(px, one);
    let q_raw = MontgomeryPoint::new(a.add(&px).negate(), one);
    let p = clear_cofactor_for_maximal_even_order(&p_raw, curve, f);
    let q = clear_cofactor_for_maximal_even_order(&q_raw, curve, f);
    // Relabel so the stored P−Q is Q (above (0,0)): basis.q = x(P−Q).
    let r = difference_point(&p, &q, &a);
    let basis = crate::ec::couple::EcBasis::new(p, r, q);
    (basis, (hint << 1) | u8::from(hint_a))
}

/// Recompute the basis from a hint produced by [`ec_curve_to_basis_2f_to_hint`].
/// Port of C `ec_curve_to_basis_2f_from_hint` (`basis.c`). Same relabeling and
/// lvl1 pinning as the `_to_hint` variant.
pub(crate) fn ec_curve_to_basis_2f_from_hint(
    curve: &crate::ec::montgomery::MontgomeryCurve<crate::params::lvl1::Fp1Element>,
    f: usize,
    hint: u8,
) -> crate::ec::couple::EcBasis<crate::params::lvl1::Fp1Element> {
    use crate::params::lvl1::Fp1Element;
    let a = curve.a;
    if bool::from(a.is_zero()) {
        return ec_basis_e0_2f(f);
    }
    let hint_a = (hint & 1) == 1;
    let hint_p = hint >> 1;
    let px = if hint_p == 0 {
        // Rare fallback: selector found nothing in 127 tries; resume from 128.
        if hint_a {
            find_nqr_factor(&a, 128).0
        } else {
            find_na_x_coord(&a, 128).0
        }
    } else if hint_a {
        // A is a QR: x(P) = −A / (1 + i·hint_P).
        let z =
            Fp2::<Fp1Element>::new(Fp1Element::one(), fp_small::<Fp1Element>(u32::from(hint_p)));
        let inv = z.invert().into_option().unwrap_or_else(Fp2::zero);
        a.mul(&inv).negate()
    } else {
        // A is a NQR: x(P) = hint_P · A.
        fp2_mul_small(&a, u32::from(hint_p))
    };
    let one = Fp2::<Fp1Element>::one();
    let p_raw = MontgomeryPoint::new(px, one);
    let q_raw = MontgomeryPoint::new(a.add(&px).negate(), one);
    let p = clear_cofactor_for_maximal_even_order(&p_raw, curve, f);
    let q = clear_cofactor_for_maximal_even_order(&q_raw, curve, f);
    let r = difference_point(&p, &q, &a);
    crate::ec::couple::EcBasis::new(p, r, q)
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    use crate::ec::montgomery::MontgomeryCurve;
    use crate::params::lvl1::Fp1Element;
    use alloc::vec::Vec;
    use subtle::ConstantTimeEq;

    #[test]
    fn s351_difference_point_c_chall_inputs() {
        use crate::ec::montgomery::MontgomeryPoint;
        fn hx(s: &str) -> Fp2<Fp1Element> {
            let bytes: Vec<u8> = (0..s.len() / 2)
                .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
                .collect();
            Option::<Fp2<Fp1Element>>::from(Fp2::<Fp1Element>::from_bytes_le(&bytes)).unwrap()
        }
        let px = hx(
            "97b229f9754fcc53970db466a7988dc02a7b3a426bdc87a8fafd87bab6257f04ad7c7220f0c8ef9fef2ad52a97f00f5c0636a305e398c791ee6f7358edd9df01",
        );
        let pz = hx(
            "c9c8064d1e05c7d0aefd3db85abf9e34c0f0a9add3d75018e07e31e2a0118f01418e03693ee0b5f53728a6dfeecdde199a5a84a939a34c7a0e803d4bd2394601",
        );
        let qx = hx(
            "b808a0e08781251fb5379526b669c06bf4d122fb6babd0c335c63525cca71c0245242431d34f53ac0d1d8f01edc26f7531a13c2484baf5750147e15b72b8e904",
        );
        let qz = hx(
            "dbb47030a789eab9eab6f2a7a54a5de03a731a70fcb93674836b00e26c7ec501d173f4b22ee5f1483e6f0ee9dbef1747de860cf180112e5f47dfc74312c35002",
        );
        let a = hx(
            "bedbf209197818f0bb9c18010649dfdb933e635ae1f120cdf24173f3a03576029ed7bdcf70629b9390507d5bf3cef1ffecd2836f8dd526e5fe9170e787fee002",
        );
        let c_outx = hx(
            "2002d7490c9d4023caa6c55d91d84d035f5d0a4e5797ee7bb2267717ab6ec6042d939faca1344ee3ba15f91e969c4f8c24c43395254c3625e244f1e9e9881502",
        );
        let c_outz = hx(
            "89311bfe5557cd95d257efe38b8ed8c8bde3208bd600acee4c4f8a332ad98b0038846c757b817966683dfd93b4a02a1a70ee1d15b675ffae5742875d6519eb01",
        );
        let p = MontgomeryPoint::new(px, pz);
        let q = MontgomeryPoint::new(qx, qz);
        let out = difference_point(&p, &q, &a);
        let affine_match = bool::from(out.x.mul(&c_outz).ct_eq(&c_outx.mul(&out.z)));
        std::eprintln!(
            "S351 DP(C-inputs): affine_match={affine_match} full_x={} full_z={}",
            out.x == c_outx,
            out.z == c_outz
        );
        // Robustness: scale p by λ=2, q by μ=3 (same affine points), recompute.
        let two = Fp2::<Fp1Element>::one().double();
        let three = two.add(&Fp2::<Fp1Element>::one());
        let p2 = MontgomeryPoint::new(px.mul(&two), pz.mul(&two));
        let q3 = MontgomeryPoint::new(qx.mul(&three), qz.mul(&three));
        let out2 = difference_point(&p2, &q3, &a);
        let robust = bool::from(out2.x.mul(&out.z).ct_eq(&out.x.mul(&out2.z)));
        std::eprintln!("S351 DP robustness (λ=2,μ=3): affine_same_as_unscaled={robust}");
    }

    #[test]
    fn is_on_curve_distinguishes_e0_and_twist() {
        use crate::isogeny::endomorphism::basis_e0_lvl1;
        let zero = Fp2::<Fp1Element>::zero(); // E0: A = 0
        // basis_e0.P is on E0.
        let (p, _, _) = basis_e0_lvl1();
        assert!(
            bool::from(is_on_curve(&p.affine_x(), &zero)),
            "basis_e0.P.x is on E0",
        );
        // An x whose x³+x is NQR lies on the twist, not E0. Use im≠0 (every
        // element of Fp ⊂ Fp2 is a square in Fp2 for p≡3 mod 4, so real x
        // never give a NQR).
        let one = Fp2::<Fp1Element>::one();
        let i = Fp2::<Fp1Element>::img();
        let mut x = i;
        let mut g = 0;
        loop {
            let fx = x.square().mul(&x).add(&x);
            if !bool::from(fx.is_zero()) && !bool::from(fx.is_square()) {
                break;
            }
            x = x.add(&one);
            g += 1;
            assert!(g < 2000);
        }
        assert!(!bool::from(is_on_curve(&x, &zero)), "twist x is not on E0");
    }

    #[test]
    fn find_na_selects_on_curve_multiple_of_a() {
        // Pick a NQR A — must have im≠0 (real elements are all QR in Fp2).
        let one = Fp2::<Fp1Element>::one();
        let mut a = Fp2::<Fp1Element>::img(); // i
        let mut g = 0;
        while bool::from(a.is_square()) {
            a = a.add(&one);
            g += 1;
            assert!(g < 2000, "found a NQR A");
        }
        let (x, hint) = find_na_x_coord(&a, 1);
        assert!(
            bool::from(is_on_curve(&x, &a)),
            "find_nA returns an on-curve x"
        );
        assert!(hint != 0, "hint found within 127 tries");
        assert_eq!(
            x,
            fp2_mul_small(&a, u32::from(hint)),
            "x = hint·A (n·A form)",
        );
    }

    #[test]
    fn find_nqr_factor_selects_on_curve_nqr_when_a_is_qr() {
        // Build a curve whose A is a QR in Fp2: A = t².
        let t = Fp2::<Fp1Element>::new(fp_small::<Fp1Element>(2), fp_small::<Fp1Element>(1));
        let a = t.square();
        assert!(bool::from(a.is_square()), "A is a QR");
        let (x, hint) = find_nqr_factor(&a, 1);
        // C debug invariants: the selected x is on the curve and is a NQR.
        assert!(bool::from(is_on_curve(&x, &a)), "selected x is on curve");
        assert!(!bool::from(x.is_square()), "selected x is a NQR");
        assert!(hint != 0, "b found within 127 tries");
        // hint = b reconstructs x = −A/(1 + i·b).
        let z = Fp2::<Fp1Element>::new(Fp1Element::one(), fp_small::<Fp1Element>(u32::from(hint)));
        let xr = a.mul(&z.invert().into_option().unwrap()).negate();
        assert_eq!(x, xr, "hint reconstructs x = −A/(1 + i·b)");
    }

    #[test]
    fn ec_basis_e0_2f_full_order_matches_precomputed() {
        use crate::isogeny::endomorphism::basis_e0_lvl1;
        let (p, q, _) = basis_e0_lvl1();
        // f = TORSION_EVEN_POWER ⇒ zero doublings; P, Q are the precomputed pts.
        let basis = ec_basis_e0_2f(248);
        assert_eq!(basis.p.affine_x(), p.affine_x(), "E0 basis P matches");
        assert_eq!(basis.q.affine_x(), q.affine_x(), "E0 basis Q matches");
    }

    #[test]
    fn ec_curve_to_basis_2f_hint_round_trips() {
        // QR-A curve: A = t² exercises the find_nqr_factor selector path.
        let t = Fp2::<Fp1Element>::new(fp_small::<Fp1Element>(2), fp_small::<Fp1Element>(1));
        let cqr = MontgomeryCurve::new(t.square());
        let (b0, h0) = ec_curve_to_basis_2f_to_hint(&cqr, 248);
        let b0r = ec_curve_to_basis_2f_from_hint(&cqr, 248, h0);
        assert_eq!(h0 & 1, 1, "A QR ⇒ hint_A bit set");
        assert_eq!(b0.p.affine_x(), b0r.p.affine_x(), "QR P round-trips");
        assert_eq!(b0.q.affine_x(), b0r.q.affine_x(), "QR Q round-trips");
        assert_eq!(
            b0.p_minus_q.affine_x(),
            b0r.p_minus_q.affine_x(),
            "QR PmQ round-trips",
        );

        // NQR-A curve: exercises the find_na_x_coord selector path.
        let one = Fp2::<Fp1Element>::one();
        let mut a = Fp2::<Fp1Element>::img();
        let mut g = 0;
        while bool::from(a.is_square()) {
            a = a.add(&one);
            g += 1;
            assert!(g < 2000, "found a NQR A");
        }
        let cnqr = MontgomeryCurve::new(a);
        let (b1, h1) = ec_curve_to_basis_2f_to_hint(&cnqr, 248);
        let b1r = ec_curve_to_basis_2f_from_hint(&cnqr, 248, h1);
        assert_eq!(h1 & 1, 0, "A NQR ⇒ hint_A bit clear");
        assert_eq!(b1.p.affine_x(), b1r.p.affine_x(), "NQR P round-trips");
        assert_eq!(b1.q.affine_x(), b1r.q.affine_x(), "NQR Q round-trips");
        assert_eq!(
            b1.p_minus_q.affine_x(),
            b1r.p_minus_q.affine_x(),
            "NQR PmQ round-trips",
        );
    }

    #[test]
    fn difference_point_matches_e0_basis_pmq() {
        // Ground truth: on E0 the canonical even basis carries (P, Q, P−Q),
        // so difference_point(P, Q) must reproduce x(P−Q) = basis_e0.PmQ.
        use crate::isogeny::endomorphism::basis_e0_lvl1;
        let (p, q, pmq) = basis_e0_lvl1();
        let a = Fp2::<Fp1Element>::zero(); // E0: A = 0
        let computed = difference_point(&p, &q, &a);
        assert_eq!(
            computed.affine_x(),
            pmq.affine_x(),
            "difference_point reproduces basis_e0.PmQ",
        );
    }

    #[test]
    fn clear_cofactor_removes_odd_part_at_lvl1() {
        use crate::ec::montgomery::MontgomeryPoint;
        use crypto_bigint::Uint;
        let curve = MontgomeryCurve::<Fp1Element>::e0();
        let a24 = curve.a24();
        let one = Fp2::<Fp1Element>::one();
        let pow248 = Uint::<8>::ONE.shl_vartime(248).to_le_bytes();
        // Find a point on E0 (y²=x³+x) that STILL has an odd factor after the
        // even part is exhausted: [2^248]·P ≠ O ⟺ P's order has the odd 5.
        // (#E0 = (p+1)² = (5·2^248)²; a generic point has order 5·2^248.)
        let mut x = one.double();
        let mut guard = 0;
        let p = loop {
            let fx = x.square().mul(&x).add(&x); // x³ + x
            if !bool::from(fx.is_zero()) && bool::from(fx.is_square()) {
                let cand = MontgomeryPoint::new(x, one);
                if !bool::from(cand.ladder(&pow248, &a24).is_infinity()) {
                    break cand; // [2^248]P ≠ O ⇒ odd factor present to clear
                }
            }
            x = x.add(&one);
            guard += 1;
            assert!(guard < 4096, "found a point on E0 with an odd factor");
        };
        // Sanity: P itself is NOT killed by 2^248 (has the odd part).
        assert!(!bool::from(p.ladder(&pow248, &a24).is_infinity()));
        let cleared = clear_cofactor_for_maximal_even_order(&p, &curve, 248);
        // After clearing the odd cofactor, order divides 2^248 …
        assert!(
            bool::from(cleared.ladder(&pow248, &a24).is_infinity()),
            "clear_cofactor removes the odd part (order | 2^248)",
        );
        // … and the point is non-trivial (clearing didn't annihilate it).
        assert!(
            !bool::from(cleared.is_infinity()),
            "cleared point is non-trivial",
        );
    }

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
