// SPDX-License-Identifier: MIT OR Apache-2.0
//! Concrete quaternion-algebra arithmetic and the `O_0` maximal-order shape.
//!
//! Conventions and multiplication table are documented in the parent module.

use core::marker::PhantomData;

use crypto_bigint::{Int, Uint};

use crate::params::Params;

/// Element `a + b·i + c·j + d·k` of `B_{p,∞}` with integer coefficients.
///
/// `LIMBS` is the limb-count of the underlying `Int<LIMBS>` — pick wide
/// enough that the products `a·e`, `p · c · g`, etc. don't overflow during
/// the intermediate computations.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Quaternion<const LIMBS: usize> {
    /// Coefficient of `1`.
    pub a: Int<LIMBS>,
    /// Coefficient of `i`.
    pub b: Int<LIMBS>,
    /// Coefficient of `j`.
    pub c: Int<LIMBS>,
    /// Coefficient of `k = i · j`.
    pub d: Int<LIMBS>,
}

impl<const LIMBS: usize> Quaternion<LIMBS> {
    /// Construct an element from its four integer components.
    #[inline]
    pub const fn new(a: Int<LIMBS>, b: Int<LIMBS>, c: Int<LIMBS>, d: Int<LIMBS>) -> Self {
        Self { a, b, c, d }
    }

    /// The zero element `0 + 0·i + 0·j + 0·k`.
    #[inline]
    pub fn zero() -> Self {
        let z = Int::<LIMBS>::from_i64(0);
        Self::new(z, z, z, z)
    }

    /// The unit element `1`.
    #[inline]
    pub fn one() -> Self {
        let z = Int::<LIMBS>::from_i64(0);
        Self::new(Int::<LIMBS>::from_i64(1), z, z, z)
    }

    /// The basis element `i`.
    #[inline]
    pub fn i() -> Self {
        let z = Int::<LIMBS>::from_i64(0);
        Self::new(z, Int::<LIMBS>::from_i64(1), z, z)
    }

    /// The basis element `j`.
    #[inline]
    pub fn j() -> Self {
        let z = Int::<LIMBS>::from_i64(0);
        Self::new(z, z, Int::<LIMBS>::from_i64(1), z)
    }

    /// The basis element `k = i · j`.
    #[inline]
    pub fn k() -> Self {
        let z = Int::<LIMBS>::from_i64(0);
        Self::new(z, z, z, Int::<LIMBS>::from_i64(1))
    }

    /// `Choice`-free equality test (the underlying integer comparison
    /// is variable-time, but quaternion coefficients are public-data in
    /// every SQIsign path this module touches).
    pub fn equals(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b && self.c == other.c && self.d == other.d
    }

    /// `self + rhs` (wrapping at `LIMBS` words — pick `LIMBS` wide enough).
    pub fn add(&self, rhs: &Self) -> Self {
        Self::new(
            self.a.wrapping_add(&rhs.a),
            self.b.wrapping_add(&rhs.b),
            self.c.wrapping_add(&rhs.c),
            self.d.wrapping_add(&rhs.d),
        )
    }

    /// `self − rhs`.
    pub fn sub(&self, rhs: &Self) -> Self {
        Self::new(
            self.a.wrapping_sub(&rhs.a),
            self.b.wrapping_sub(&rhs.b),
            self.c.wrapping_sub(&rhs.c),
            self.d.wrapping_sub(&rhs.d),
        )
    }

    /// `−self`.
    pub fn negate(&self) -> Self {
        Self::new(
            self.a.wrapping_neg(),
            self.b.wrapping_neg(),
            self.c.wrapping_neg(),
            self.d.wrapping_neg(),
        )
    }

    /// Quaternion conjugate `a − b·i − c·j − d·k`.
    pub fn conjugate(&self) -> Self {
        Self::new(
            self.a,
            self.b.wrapping_neg(),
            self.c.wrapping_neg(),
            self.d.wrapping_neg(),
        )
    }

    /// Reduced trace `Tr(q) = q + q̄ = 2 a`.
    pub fn trace(&self) -> Int<LIMBS> {
        self.a.wrapping_add(&self.a)
    }

    /// `self · scalar` — scalar multiplication by a small integer.
    pub fn scale(&self, k: i64) -> Self {
        let k_int = Int::<LIMBS>::from_i64(k);
        Self::new(
            self.a.wrapping_mul(&k_int),
            self.b.wrapping_mul(&k_int),
            self.c.wrapping_mul(&k_int),
            self.d.wrapping_mul(&k_int),
        )
    }

    /// Quaternion multiplication `self × rhs` using the relations
    /// `i² = −1`, `j² = −p`, `k = i j = −j i`.
    ///
    /// `p` is the level's prime, encoded as an unsigned `Uint<LIMBS>`.
    /// (Internally cast to `Int<LIMBS>` once.)
    pub fn mul(&self, rhs: &Self, p: &Uint<LIMBS>) -> Self {
        let p_int = *p.as_int();
        // 1 = a·e − b·f − p · (c·g + d·h)
        let ae = self.a.wrapping_mul(&rhs.a);
        let bf = self.b.wrapping_mul(&rhs.b);
        let cg = self.c.wrapping_mul(&rhs.c);
        let dh = self.d.wrapping_mul(&rhs.d);
        let p_cgdh = p_int.wrapping_mul(&cg.wrapping_add(&dh));
        let out_a = ae.wrapping_sub(&bf).wrapping_sub(&p_cgdh);
        // i = a·f + b·e + p · (c·h − d·g)   (jk = p·i, kj = −p·i)
        let af = self.a.wrapping_mul(&rhs.b);
        let be = self.b.wrapping_mul(&rhs.a);
        let ch = self.c.wrapping_mul(&rhs.d);
        let dg = self.d.wrapping_mul(&rhs.c);
        let p_chdg = p_int.wrapping_mul(&ch.wrapping_sub(&dg));
        let out_b = af.wrapping_add(&be).wrapping_add(&p_chdg);
        // j = a·g + c·e + (d·f − b·h)        (ik = −j, ki = +j)
        let ag = self.a.wrapping_mul(&rhs.c);
        let ce = self.c.wrapping_mul(&rhs.a);
        let bh = self.b.wrapping_mul(&rhs.d);
        let df = self.d.wrapping_mul(&rhs.b);
        let out_c = ag.wrapping_add(&ce).wrapping_add(&df.wrapping_sub(&bh));
        // k = a·h + d·e + (b·g − c·f)
        let ah = self.a.wrapping_mul(&rhs.d);
        let de = self.d.wrapping_mul(&rhs.a);
        let bg = self.b.wrapping_mul(&rhs.c);
        let cf = self.c.wrapping_mul(&rhs.b);
        let out_d = ah.wrapping_add(&de).wrapping_add(&bg.wrapping_sub(&cf));
        Self::new(out_a, out_b, out_c, out_d)
    }

    /// Reduced norm `N(q) = a² + b² + p (c² + d²)`.
    pub fn norm(&self, p: &Uint<LIMBS>) -> Int<LIMBS> {
        let p_int = *p.as_int();
        let a_sq = self.a.wrapping_mul(&self.a);
        let b_sq = self.b.wrapping_mul(&self.b);
        let c_sq = self.c.wrapping_mul(&self.c);
        let d_sq = self.d.wrapping_mul(&self.d);
        let cd = c_sq.wrapping_add(&d_sq);
        let p_cd = p_int.wrapping_mul(&cd);
        a_sq.wrapping_add(&b_sq).wrapping_add(&p_cd)
    }
}

/// Marker for the quaternion algebra `B_{p,∞}` at the given parameter set.
/// Carries no data; provides level-aware helpers via the `Params` trait.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct QuaternionAlgebra<P: Params> {
    _marker: PhantomData<P>,
}

impl<P: Params> QuaternionAlgebra<P> {
    /// Construct a fresh marker.
    #[inline]
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

/// Symbolic basis vectors of the special maximal order `O_0` for `p ≡ 3 mod 4`.
///
/// `O_0 = ⟨ 1, i, (i + j) / 2, (1 + k) / 2 ⟩` over `Z`. Every element of
/// `O_0` is a `Z`-linear combination of these four. Storage as integer
/// coefficients in this basis is the cheap-and-cheerful representation
/// downstream KLPT code prefers — a 4×4 basis-change matrix into the
/// standard `(1, i, j, k)` basis lets us go to `Quaternion` when needed.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum OrderBasis {
    /// `1`
    One,
    /// `i`
    I,
    /// `(i + j) / 2`
    IPlusJOver2,
    /// `(1 + k) / 2`
    OnePlusKOver2,
}

/// Operations against the special maximal order `O_0` at the given level.
/// Concrete element representation will land alongside the KLPT session.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct MaximalOrder<P: Params> {
    _marker: PhantomData<P>,
}

impl<P: Params> MaximalOrder<P> {
    /// Construct the maximal-order marker for level `P`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// The four basis vectors of `O_0`, in canonical order.
    pub const fn basis() -> [OrderBasis; 4] {
        [
            OrderBasis::One,
            OrderBasis::I,
            OrderBasis::IPlusJOver2,
            OrderBasis::OnePlusKOver2,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LIMBS = 8 (512-bit signed) — plenty for Level-1's 251-bit prime plus
    /// the small integer multipliers used in these algebraic-identity tests.
    type Q = Quaternion<8>;
    type U = Uint<8>;

    /// A small prime-sized Uint for testing the `j² = −p` and norm identities
    /// without needing to compute against the real SQIsign primes.
    fn small_p() -> U {
        // Use a small "fake" prime for arithmetic-identity tests. The
        // structural identities (i² = −1, anti-commutativity, etc.) don't
        // depend on the value of p; norm and j² do.
        U::from_u64(7)
    }

    #[test]
    fn i_squared_is_minus_one() {
        let p = small_p();
        let i_sq = Q::i().mul(&Q::i(), &p);
        let minus_one = Q::one().negate();
        assert!(i_sq.equals(&minus_one));
    }

    #[test]
    fn j_squared_is_minus_p() {
        let p = small_p();
        let j_sq = Q::j().mul(&Q::j(), &p);
        // j² = -p · 1
        let minus_p = Q::new(
            p.as_int().wrapping_neg(),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        );
        assert!(j_sq.equals(&minus_p));
    }

    #[test]
    fn k_equals_ij() {
        let p = small_p();
        let ij = Q::i().mul(&Q::j(), &p);
        assert!(ij.equals(&Q::k()));
    }

    #[test]
    fn ji_equals_minus_k() {
        let p = small_p();
        let ji = Q::j().mul(&Q::i(), &p);
        let minus_k = Q::k().negate();
        assert!(ji.equals(&minus_k));
    }

    #[test]
    fn k_squared_is_minus_p() {
        let p = small_p();
        let k_sq = Q::k().mul(&Q::k(), &p);
        let minus_p = Q::new(
            p.as_int().wrapping_neg(),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        );
        assert!(k_sq.equals(&minus_p));
    }

    #[test]
    fn conjugate_is_involution() {
        let p = small_p();
        let q = Q::new(
            Int::<8>::from_i64(3),
            Int::<8>::from_i64(-5),
            Int::<8>::from_i64(7),
            Int::<8>::from_i64(-2),
        );
        assert!(q.conjugate().conjugate().equals(&q));
        let _ = p;
    }

    #[test]
    fn norm_equals_q_times_conj_a_component() {
        let p = small_p();
        let q = Q::new(
            Int::<8>::from_i64(3),
            Int::<8>::from_i64(-5),
            Int::<8>::from_i64(2),
            Int::<8>::from_i64(1),
        );
        // q · q̄ should be a pure scalar equal to N(q).
        let q_qbar = q.mul(&q.conjugate(), &p);
        let n = q.norm(&p);
        assert_eq!(q_qbar.a, n);
        assert_eq!(q_qbar.b, Int::<8>::from_i64(0));
        assert_eq!(q_qbar.c, Int::<8>::from_i64(0));
        assert_eq!(q_qbar.d, Int::<8>::from_i64(0));
    }

    #[test]
    fn norm_of_one_is_one() {
        let p = small_p();
        let n = Q::one().norm(&p);
        assert_eq!(n, Int::<8>::from_i64(1));
    }

    #[test]
    fn norm_of_i_is_one() {
        let p = small_p();
        let n = Q::i().norm(&p);
        assert_eq!(n, Int::<8>::from_i64(1));
    }

    #[test]
    fn norm_of_j_is_p() {
        let p = small_p();
        let n = Q::j().norm(&p);
        assert_eq!(n, *p.as_int());
    }

    #[test]
    fn trace_is_2a() {
        let q = Q::new(
            Int::<8>::from_i64(11),
            Int::<8>::from_i64(-3),
            Int::<8>::from_i64(0),
            Int::<8>::from_i64(0),
        );
        assert_eq!(q.trace(), Int::<8>::from_i64(22));
    }

    #[test]
    fn addition_is_commutative() {
        let p = small_p();
        let a = Q::new(
            Int::<8>::from_i64(1),
            Int::<8>::from_i64(2),
            Int::<8>::from_i64(3),
            Int::<8>::from_i64(4),
        );
        let b = Q::new(
            Int::<8>::from_i64(5),
            Int::<8>::from_i64(6),
            Int::<8>::from_i64(7),
            Int::<8>::from_i64(8),
        );
        let ab = a.add(&b);
        let ba = b.add(&a);
        assert!(ab.equals(&ba));
        let _ = p;
    }

    #[test]
    fn multiplication_is_not_commutative() {
        let p = small_p();
        let ij = Q::i().mul(&Q::j(), &p);
        let ji = Q::j().mul(&Q::i(), &p);
        assert!(!ij.equals(&ji));
    }

    #[test]
    fn order_basis_canonical_count() {
        use crate::params::Level1;
        let basis = MaximalOrder::<Level1>::basis();
        assert_eq!(basis.len(), 4);
        assert_eq!(basis[0], OrderBasis::One);
        assert_eq!(basis[3], OrderBasis::OnePlusKOver2);
    }

    #[test]
    fn scale_by_two_doubles() {
        let p = small_p();
        let q = Q::new(
            Int::<8>::from_i64(1),
            Int::<8>::from_i64(2),
            Int::<8>::from_i64(3),
            Int::<8>::from_i64(4),
        );
        let doubled = q.scale(2);
        let summed = q.add(&q);
        assert!(doubled.equals(&summed));
        let _ = p;
    }
}
