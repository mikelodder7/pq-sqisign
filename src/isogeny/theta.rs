// SPDX-License-Identifier: MIT OR Apache-2.0
//! Theta coordinates on 2-dimensional abelian varieties.
//!
//! Foundational data type for the Clapotis evaluator (per SQIsign
//! 2.0.1 spec §6). A 2-dim abelian variety carries a theta structure
//! whose coordinates are a 4-tuple of elements in `F_{p²}`. These
//! coordinates are what the Clapotis algorithm operates on — the
//! ideal-to-isogeny translation embeds the input left ideal as a
//! theta-coordinate point, applies the evaluator, then projects
//! back to an elliptic-curve isogeny.
//!
//! This module ships the foundational data type ([`ThetaPoint2D`])
//! and basic operations (zero, equality, conditional select). The
//! algorithmic content of the evaluator — gluing, doubling,
//! projection — lands across the upcoming Clapotis arc (~25-30
//! sessions per the ISA roadmap).
//!
//! # Type design
//!
//! [`ThetaPoint2D<F>`] is a 4-tuple `(x, y, z, w)` of `Fp2<F>` with:
//! - **Identity**: `(1, 1, 1, 1)` — the theta-coordinate origin per
//!   the affine convention. We expose it as [`ThetaPoint2D::identity`].
//! - **Zero**: `(0, 0, 0, 0)` — exposed as [`ThetaPoint2D::zero`] for
//!   "uninitialized" placeholders.
//! - **Equality**: projective in spirit, but we use exact `==` here
//!   (theta coordinates carry projective ambiguity that the higher-
//!   level algorithm normalises).
//! - **Constant-time choice**: [`ConditionallySelectable`] implemented
//!   via Fp2's componentwise CT primitives.

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use crate::error::{Error, Result};
use crate::gf::fp::BaseField;
use crate::gf::fp2::Fp2;

/// Theta-coordinate point on a 2-dim abelian variety: a 4-tuple of
/// `F_{p²}` elements. See module-level docs.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct ThetaPoint2D<F: BaseField> {
    /// First coordinate.
    pub x: Fp2<F>,
    /// Second coordinate.
    pub y: Fp2<F>,
    /// Third coordinate.
    pub z: Fp2<F>,
    /// Fourth coordinate.
    pub w: Fp2<F>,
}

impl<F: BaseField> ThetaPoint2D<F> {
    /// Construct from coordinate components.
    #[inline]
    pub const fn new(x: Fp2<F>, y: Fp2<F>, z: Fp2<F>, w: Fp2<F>) -> Self {
        Self { x, y, z, w }
    }

    /// The all-zero point `(0, 0, 0, 0)` — placeholder for
    /// uninitialised state. Not algebraically meaningful as an
    /// abelian-variety point but useful for type instantiation.
    pub fn zero() -> Self {
        Self::new(
            Fp2::<F>::zero(),
            Fp2::<F>::zero(),
            Fp2::<F>::zero(),
            Fp2::<F>::zero(),
        )
    }

    /// The theta-coordinate identity `(1, 1, 1, 1)` — the abelian-
    /// variety origin in the affine theta-coordinate convention.
    pub fn identity() -> Self {
        Self::new(
            Fp2::<F>::one(),
            Fp2::<F>::one(),
            Fp2::<F>::one(),
            Fp2::<F>::one(),
        )
    }

    /// `Choice::TRUE` iff `self` is the all-zero placeholder.
    pub fn is_zero(&self) -> Choice {
        self.x.is_zero() & self.y.is_zero() & self.z.is_zero() & self.w.is_zero()
    }

    /// Hadamard transform of `self`.
    ///
    /// On a 4-tuple `(x, y, z, w)` representing a 2-dim theta-coordinate
    /// point indexed by `(Z/2Z)²` (basis order `(0,0), (1,0), (0,1),
    /// (1,1)`), the Hadamard transform is:
    ///
    /// ```text
    /// H(x, y, z, w) = ( x + y + z + w,
    ///                  x − y + z − w,
    ///                  x + y − z − w,
    ///                  x − y − z + w )
    /// ```
    ///
    /// This is a linear transform with the well-known scaling property
    /// `H² = 4 · I`: applying it twice multiplies each coordinate by 4.
    /// The property holds for any `F_{p²}` (provided `p` is odd, which
    /// every SQIsign prime is).
    ///
    /// Foundational primitive for the theta-coordinate doubling formulas
    /// of the Clapotis evaluator (SQIsign 2.0.1 spec §6.2): doubling
    /// composes `H → square → H → multiply-by-constants`.
    ///
    /// Componentwise square: `(x, y, z, w) ↦ (x², y², z², w²)`.
    ///
    /// One `Fp2` squaring per coordinate (4 total). Building block for
    /// the theta-coordinate doubling composition
    /// `H → square → H → multiply-by-constants` per SQIsign 2.0.1
    /// spec §6.2.
    ///
    /// Squaring is cheaper than general multiplication because
    /// `Fp2::square` uses the binomial identity
    /// `(a + bi)² = (a−b)(a+b) + 2ab·i`, which has 2 base-field
    /// multiplications vs 3 for general multiplication.
    pub fn componentwise_square(&self) -> Self {
        Self::new(
            self.x.square(),
            self.y.square(),
            self.z.square(),
            self.w.square(),
        )
    }

    /// Componentwise inverse: `(x, y, z, w) ↦ (1/x, 1/y, 1/z, 1/w)`.
    ///
    /// Returns `None` if any component is zero (the abelian variety
    /// is degenerate at the corresponding coordinate). Otherwise
    /// returns `Some(inverse_tuple)`.
    ///
    /// Foundational for the constants-from-theta-null derivation:
    /// per Riemann's duplication formula on level-(2,2) theta
    /// structures (Cosset-Robert), the projective doubling constants
    /// are `1 / H(theta_null²)` — the componentwise inverse of the
    /// Hadamard transform of the squared theta-null.
    ///
    /// Cost: 4 `Fp2` inversions. Inversion is the most expensive
    /// operation in `Fp2` (involves an Fp inversion + multiplications);
    /// callers should cache inverse results where possible.
    pub fn componentwise_inverse(&self) -> Option<Self> {
        // Each component inverted independently; any zero component
        // makes the overall result undefined.
        let x_inv = self.x.invert().into_option()?;
        let y_inv = self.y.invert().into_option()?;
        let z_inv = self.z.invert().into_option()?;
        let w_inv = self.w.invert().into_option()?;
        Some(Self::new(x_inv, y_inv, z_inv, w_inv))
    }

    /// Projective equality: `Choice::TRUE` iff `other` is a non-zero
    /// scalar multiple of `self` (the two points represent the same
    /// projective theta-coordinate equivalence class).
    ///
    /// Implementation: tests all six cross-product equalities of the
    /// 2×2 minors of the 2×4 matrix `[ self; other ]`. Two 4-tuples
    /// `(a, b, c, d)` and `(a', b', c', d')` are projectively equal
    /// iff every 2×2 minor of the stacked matrix vanishes:
    ///
    /// ```text
    /// a·b' = b·a',  a·c' = c·a',  a·d' = d·a',
    /// b·c' = c·b',  b·d' = d·b',  c·d' = d·c'
    /// ```
    ///
    /// **Caveat**: this test treats the all-zero point `(0, 0, 0, 0)`
    /// as projectively equal to *any* point (all six cross-products
    /// are trivially zero), which is technically wrong since the
    /// all-zero "point" is not a valid projective representative.
    /// Callers that need to distinguish the all-zero degenerate
    /// case should pre-filter via [`Self::is_zero`].
    ///
    /// Constant-time: yes — uses `Fp2::ct_eq` and bitwise `&` only.
    ///
    /// Cost: 12 `Fp2` multiplications + 6 `Fp2::ct_eq` ops.
    pub fn project_equals(&self, other: &Self) -> Choice {
        let m_xy = self.x.mul(&other.y).ct_eq(&self.y.mul(&other.x));
        let m_xz = self.x.mul(&other.z).ct_eq(&self.z.mul(&other.x));
        let m_xw = self.x.mul(&other.w).ct_eq(&self.w.mul(&other.x));
        let m_yz = self.y.mul(&other.z).ct_eq(&self.z.mul(&other.y));
        let m_yw = self.y.mul(&other.w).ct_eq(&self.w.mul(&other.y));
        let m_zw = self.z.mul(&other.w).ct_eq(&self.w.mul(&other.z));
        m_xy & m_xz & m_xw & m_yz & m_yw & m_zw
    }

    /// Projective normalization to canonical `x = 1` form.
    ///
    /// Theta-coordinate points are inherently projective — defined up
    /// to a global non-zero scalar. This method returns the unique
    /// representative `(1, y/x, z/x, w/x)` by multiplying all
    /// components by `x⁻¹`. Required for canonical comparison and
    /// for byte-exact comparison against reference vectors (the
    /// upcoming KAT path will compare against normalised forms).
    ///
    /// Returns `None` if `x` is zero (the point cannot be normalised
    /// along the `x` axis; callers may need a different coordinate
    /// as the normalisation pivot in that case).
    ///
    /// Cost: 1 `Fp2` inversion + 3 `Fp2` multiplications.
    ///
    /// Constant-time relative to whether `x` is zero — the only
    /// branch is "Some" vs "None". For SQIsign signing paths
    /// (where `x` is derived from secret data), callers should
    /// ensure the upstream construction makes `x = 0` either
    /// impossible or constant-time-detectable before calling this.
    pub fn normalize_by_x(&self) -> Option<Self> {
        let x_inv = self.x.invert().into_option()?;
        Some(Self::new(
            Fp2::<F>::one(),
            self.y.mul(&x_inv),
            self.z.mul(&x_inv),
            self.w.mul(&x_inv),
        ))
    }

    /// Projective normalization with a caller-chosen pivot.
    ///
    /// `pivot` selects which coordinate becomes 1 after normalisation:
    /// - `0` → `x` becomes 1 (equivalent to [`Self::normalize_by_x`])
    /// - `1` → `y` becomes 1
    /// - `2` → `z` becomes 1
    /// - `3` → `w` becomes 1
    /// - any other value → `None`
    ///
    /// Returns `None` if the chosen pivot component is zero.
    ///
    /// Useful when `x` is zero but a different coordinate is not, or
    /// when the caller has algorithm-specific knowledge about which
    /// pivot makes the encoding most canonical.
    ///
    /// Cost: 1 `Fp2` inversion + 3 `Fp2` multiplications.
    ///
    /// **Side-channel note**: branching on `pivot` is timing-leaky
    /// for the *pivot value*, which is a PUBLIC parameter — the
    /// underlying field arithmetic is constant-time. Do not pass
    /// secret-derived pivots to this method.
    pub fn normalize_by_pivot(&self, pivot: u8) -> Option<Self> {
        let pivot_value = match pivot {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            3 => self.w,
            _ => return None,
        };
        let pivot_inv = pivot_value.invert().into_option()?;
        Some(Self::new(
            self.x.mul(&pivot_inv),
            self.y.mul(&pivot_inv),
            self.z.mul(&pivot_inv),
            self.w.mul(&pivot_inv),
        ))
    }

    /// Constant-time check that `self` is in canonical x-pivot-normalised
    /// form: `self.x == Fp2::<F>::one()`. Returns `Choice::TRUE` iff
    /// the point's `x` component equals the multiplicative identity.
    ///
    /// Postcondition contract: for any `p` with non-zero `x`,
    /// `p.normalize_by_x().unwrap().is_normalised_by_x()` is
    /// `Choice::TRUE`. Useful as an invariant check at boundaries
    /// (e.g. immediately after decoding from a wire format that is
    /// promised to be canonical).
    ///
    /// Cost: 1 `Fp2::ct_eq` against the precomputed `Fp2::one()`.
    /// Constant-time: yes.
    pub fn is_normalised_by_x(&self) -> Choice {
        self.x.ct_eq(&Fp2::<F>::one())
    }

    /// Constant-time check that the pivot component of `self` equals
    /// `Fp2::<F>::one()`. Mirrors [`Self::normalize_by_pivot`]'s pivot
    /// index convention:
    /// - `0` → check `self.x == 1`
    /// - `1` → check `self.y == 1`
    /// - `2` → check `self.z == 1`
    /// - `3` → check `self.w == 1`
    /// - any other value → returns `Choice::FALSE` (out-of-range pivot
    ///   cannot satisfy any invariant)
    ///
    /// Postcondition contract: for any `p` and `pivot` in `0..4`
    /// with a non-zero pivot component,
    /// `p.normalize_by_pivot(pivot).unwrap().is_normalised_by_pivot(pivot)`
    /// is `Choice::TRUE`.
    ///
    /// Cost: 1 `Fp2::ct_eq` plus the match on the public pivot value.
    /// Constant-time relative to the underlying field arithmetic; the
    /// pivot is treated as a PUBLIC parameter (same convention as
    /// [`Self::normalize_by_pivot`]).
    pub fn is_normalised_by_pivot(&self, pivot: u8) -> Choice {
        let pivot_value = match pivot {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            3 => self.w,
            _ => return Choice::from(0),
        };
        pivot_value.ct_eq(&Fp2::<F>::one())
    }

    /// Componentwise multiplication: `(x, y, z, w) ⊙ (x', y', z', w')
    /// = (x·x', y·y', z·z', w·w')`.
    ///
    /// One `Fp2` multiplication per coordinate (4 total). Building block
    /// for the theta-coordinate doubling composition (the "multiply-by-
    /// constants" final step) and for general Hadamard-product
    /// operations on theta tuples.
    pub fn componentwise_mul(&self, rhs: &Self) -> Self {
        Self::new(
            self.x.mul(&rhs.x),
            self.y.mul(&rhs.y),
            self.z.mul(&rhs.z),
            self.w.mul(&rhs.w),
        )
    }

    /// Encoded byte length of a theta-coordinate point: four `Fp2`
    /// elements, each `2 · F::ENCODED_BYTES` bytes. Layout in the
    /// buffer is `x_bytes || y_bytes || z_bytes || w_bytes` where
    /// each `_bytes` is the corresponding `Fp2` encoded as
    /// `re_le_bytes || im_le_bytes` per [`Fp2::to_bytes_le`].
    ///
    /// At the three SQIsign levels:
    /// - L1: 4 × 2 × 32 = 256 bytes
    /// - L3: 4 × 2 × 48 = 384 bytes
    /// - L5: 4 × 2 × 64 = 512 bytes
    pub const ENCODED_BYTES: usize = 8 * F::ENCODED_BYTES;

    /// Encode `self` into `out` as `x_bytes || y_bytes || z_bytes ||
    /// w_bytes`. Returns [`Error::BufferTooSmall`] if `out.len() <
    /// ENCODED_BYTES`; writes exactly `ENCODED_BYTES` bytes on
    /// success.
    ///
    /// The encoding is the *un-normalised* projective representative.
    /// Two projectively-equal points may produce different byte
    /// encodings; callers that need a canonical encoding should
    /// first apply [`Self::normalize_by_x`] (or a different
    /// pivot normalisation) before encoding.
    pub fn to_bytes_le(&self, out: &mut [u8]) -> Result<()> {
        if out.len() < Self::ENCODED_BYTES {
            return Err(Error::BufferTooSmall {
                required: Self::ENCODED_BYTES,
                provided: out.len(),
            });
        }
        let n = 2 * F::ENCODED_BYTES;
        self.x.to_bytes_le(&mut out[..n]);
        self.y.to_bytes_le(&mut out[n..2 * n]);
        self.z.to_bytes_le(&mut out[2 * n..3 * n]);
        self.w.to_bytes_le(&mut out[3 * n..4 * n]);
        Ok(())
    }

    /// Decode a theta-coordinate point from `bytes`. Returns
    /// [`Error::BufferTooSmall`] if `bytes.len() < ENCODED_BYTES`;
    /// returns [`Error::NonCanonicalEncoding`] if any of the four
    /// `Fp2` components has a non-canonical (`re` or `im` ≥ p)
    /// representation. Reads exactly `ENCODED_BYTES` bytes; extra
    /// bytes are ignored.
    ///
    /// Does NOT validate that the decoded 4-tuple represents a
    /// point on any particular abelian variety — theta coordinates
    /// are accepted as opaque algebraic data.
    pub fn from_bytes_le(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::ENCODED_BYTES {
            return Err(Error::BufferTooSmall {
                required: Self::ENCODED_BYTES,
                provided: bytes.len(),
            });
        }
        let n = 2 * F::ENCODED_BYTES;
        let x = Fp2::<F>::from_bytes_le(&bytes[..n])
            .into_option()
            .ok_or(Error::NonCanonicalEncoding)?;
        let y = Fp2::<F>::from_bytes_le(&bytes[n..2 * n])
            .into_option()
            .ok_or(Error::NonCanonicalEncoding)?;
        let z = Fp2::<F>::from_bytes_le(&bytes[2 * n..3 * n])
            .into_option()
            .ok_or(Error::NonCanonicalEncoding)?;
        let w = Fp2::<F>::from_bytes_le(&bytes[3 * n..4 * n])
            .into_option()
            .ok_or(Error::NonCanonicalEncoding)?;
        Ok(Self::new(x, y, z, w))
    }

    /// Theta-coordinate differential addition: `P + Q` given `P − Q`.
    ///
    /// Computes `P + Q` on the underlying 2-dim abelian variety from
    /// three projective theta-coordinate inputs:
    /// - `p`, `q`: the operands.
    /// - `p_minus_q`: the difference `P − Q`, supplied by the caller.
    ///   (Theta-coordinate subtraction is not derivable from `p` and
    ///   `q` alone — the level-(2,2) projective representation loses
    ///   the sign information needed to distinguish `P − Q` from
    ///   `P + Q`.)
    ///
    /// Returns `None` if any component of `H(p_minus_q)` is zero
    /// (the componentwise inversion at the heart of the formula is
    /// undefined).
    ///
    /// # Provisional formula
    ///
    /// For each index `i ∈ {0, 1, 2, 3}`:
    ///
    /// ```text
    /// (P + Q)[i] = H(p)[i] · H(q)[i] / H(p_minus_q)[i]
    /// ```
    ///
    /// where `H` is the Hadamard transform. This is the simplest
    /// plausible level-(2,2) projective differential-addition formula
    /// admitted by the Riemann bilinear relations (Mumford, *Tata
    /// Lectures on Theta II*, §6). It is symmetric in `p` and `q`
    /// (so `diff_add` is commutative) and produces `None` exactly
    /// when the inversion fails.
    ///
    /// # Spec-authority caveat
    ///
    /// SQIsign 2.0.1 §6.3 may specify a different sign or scaling
    /// convention (e.g. one involving the variety's theta-null or
    /// doubling constants explicitly). The committed formula above
    /// matches the structural identity from the public literature
    /// but **has not been verified byte-exactly against the SQIsign
    /// 2.0.1 reference C implementation**. Verification — and any
    /// necessary correction — is deferred to a future session when
    /// the spec can be studied directly. Until then, this method is
    /// safe to use for *internal-consistency* code paths but should
    /// not be relied on for KAT byte-exact comparison.
    ///
    /// # Constant-time
    ///
    /// Yes — uses only the constant-time `Fp2` primitives plus the
    /// `?` short-circuit on `componentwise_inverse`, which leaks
    /// only the public "is this input degenerate" bit.
    ///
    /// Cost: 3 Hadamard transforms (16 add/sub each = 48 total),
    /// 2 componentwise Fp2 multiplications (8 mults total), and
    /// 4 Fp2 inversions (componentwise_inverse of `H(p_minus_q)`).
    pub fn diff_add(p: &Self, q: &Self, p_minus_q: &Self) -> Option<Self> {
        let hp = p.hadamard();
        let hq = q.hadamard();
        let hpmq = p_minus_q.hadamard();
        let hpmq_inv = hpmq.componentwise_inverse()?;
        Some(hp.componentwise_mul(&hq).componentwise_mul(&hpmq_inv))
    }

    /// Theta-coordinate doubling: `P → 2P` on the 2-dim abelian variety.
    ///
    /// Composes the four primitives in spec-§6.2 order:
    ///
    /// ```text
    /// P → hadamard → componentwise_square → hadamard → componentwise_mul(constants) → 2P
    /// ```
    ///
    /// The `constants` argument is the abelian variety's "doubling
    /// constants" — a theta-coordinate tuple derived from the variety's
    /// structure. Its extraction is a separate primitive (future
    /// session); this method takes it as a parameter so the composition
    /// itself can be tested independently of the constant-derivation
    /// logic.
    ///
    /// Cost: 20 base-field multiplications (4 Fp2 squarings × 2 mults +
    /// 4 Fp2 Karatsuba mults × 3 mults) plus 16 base-field add/sub
    /// operations from the two Hadamard transforms.
    pub fn double(&self, constants: &Self) -> Self {
        self.hadamard()
            .componentwise_square()
            .hadamard()
            .componentwise_mul(constants)
    }

    /// Uses 8 `Fp2` additions/subtractions, zero multiplications.
    pub fn hadamard(&self) -> Self {
        // Compute the four sums/differences. Naming follows the
        // index pairs `(i, j)` over `(Z/2Z)²`:
        //   a0 = x + y, a1 = x − y, b0 = z + w, b1 = z − w.
        let a0 = self.x.add(&self.y);
        let a1 = self.x.sub(&self.y);
        let b0 = self.z.add(&self.w);
        let b1 = self.z.sub(&self.w);
        // Combine into the four Hadamard outputs.
        let x_h = a0.add(&b0); // (x+y) + (z+w)
        let y_h = a1.add(&b1); // (x−y) + (z−w)
        let z_h = a0.sub(&b0); // (x+y) − (z+w)
        let w_h = a1.sub(&b1); // (x−y) − (z−w)
        Self::new(x_h, y_h, z_h, w_h)
    }
}

impl<F: BaseField> ConstantTimeEq for ThetaPoint2D<F> {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.x.ct_eq(&other.x)
            & self.y.ct_eq(&other.y)
            & self.z.ct_eq(&other.z)
            & self.w.ct_eq(&other.w)
    }
}

impl<F: BaseField> ConditionallySelectable for ThetaPoint2D<F> {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self::new(
            Fp2::<F>::conditional_select(&a.x, &b.x, choice),
            Fp2::<F>::conditional_select(&a.y, &b.y, choice),
            Fp2::<F>::conditional_select(&a.z, &b.z, choice),
            Fp2::<F>::conditional_select(&a.w, &b.w, choice),
        )
    }
}

/// Aggregated parameters of a 2-dim abelian variety with its
/// theta structure: the theta-null point at the origin and the
/// doubling constants derived from it.
///
/// Where [`ThetaPoint2D`] is a generic theta-coordinate point,
/// `AbelianVariety2D` carries the variety-specific structure that
/// makes doubling (and future isogeny evaluation) well-defined. The
/// doubling constants are typically derived from the theta-null via
/// Riemann's duplication formula, but the derivation algorithm
/// depends on the SQIsign 2.0.1 spec interpretation — for now this
/// struct accepts both as inputs, so the doubling composition
/// is testable independently of the constants-extraction logic.
///
/// Future sessions add:
/// - `from_montgomery_curve(curve: &MontgomeryCurve<F>)` — derive
///   the theta-null + constants from an elliptic curve.
/// - `dual_theta_null()` and other Riemann-relation accessors.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AbelianVariety2D<F: BaseField> {
    /// Theta-null point `(θ_{00}(0), θ_{10}(0), θ_{01}(0), θ_{11}(0))`
    /// at the origin of the variety.
    pub theta_null: ThetaPoint2D<F>,
    /// Doubling constants for the theta-coordinate doubling formula.
    /// Multiplied componentwise after the second Hadamard in the
    /// `H → square → H → multiply` composition.
    pub doubling_constants: ThetaPoint2D<F>,
}

impl<F: BaseField> AbelianVariety2D<F> {
    /// Construct from explicit theta-null and doubling constants.
    pub const fn new(theta_null: ThetaPoint2D<F>, doubling_constants: ThetaPoint2D<F>) -> Self {
        Self {
            theta_null,
            doubling_constants,
        }
    }

    /// Construct from theta-null alone, deriving the doubling
    /// constants via Riemann's duplication formula in the projective
    /// convention:
    ///
    /// ```text
    /// doubling_constants = componentwise_inverse( H(theta_null²) )
    /// ```
    ///
    /// where `H(·)` is the Hadamard transform and `theta_null²` is
    /// the componentwise square. This matches Cosset-Robert's level-
    /// (2,2) theta-coordinate doubling formula up to a global scaling
    /// factor (the formula in affine coordinates includes a divide-
    /// by-4 absorbed into projective normalisation).
    ///
    /// Returns `None` if `H(theta_null²)` has a zero component — the
    /// variety is degenerate and doubling is undefined.
    ///
    /// For testing: with these derived constants,
    /// `variety.double(p)` produces the projective doubling
    /// `2P` per the standard theta-coordinate formula.
    pub fn from_theta_null(theta_null: ThetaPoint2D<F>) -> Option<Self> {
        // H(theta_null²) — the squared-dual-theta-null in spec
        // terminology. Riemann's duplication formula uses its
        // componentwise inverse as the doubling constants.
        let h_squared = theta_null.componentwise_square().hadamard();
        let doubling_constants = h_squared.componentwise_inverse()?;
        Some(Self {
            theta_null,
            doubling_constants,
        })
    }

    /// Double a theta-coordinate point on this variety: `P → 2P`.
    ///
    /// Thin wrapper around [`ThetaPoint2D::double`] that uses
    /// `self.doubling_constants` so callers don't have to thread the
    /// constants through.
    pub fn double(&self, p: &ThetaPoint2D<F>) -> ThetaPoint2D<F> {
        p.double(&self.doubling_constants)
    }

    /// Dual theta-null: `H(theta_null)`. The Hadamard transform of the
    /// theta-null at the origin gives the "dual theta-null" used in
    /// several Riemann-relation identities and in the eventual
    /// derivation of `doubling_constants` from `theta_null`.
    ///
    /// Per SQIsign 2.0.1 spec §6.2, this dual is one of the
    /// intermediate values in the constants-from-theta-null formula
    /// (Cosset-Robert / Riemann's duplication style). Shipped as a
    /// standalone accessor so the derivation formula can be composed
    /// in a future session without changing this surface.
    pub fn dual_theta_null(&self) -> ThetaPoint2D<F> {
        self.theta_null.hadamard()
    }

    /// Componentwise-squared theta-null: `(a², b², c², d²)`.
    /// Common intermediate value in theta-relation identities and
    /// in the derivation of `doubling_constants`.
    pub fn theta_null_squared(&self) -> ThetaPoint2D<F> {
        self.theta_null.componentwise_square()
    }

    /// Iterated doubling: `P → 2^n · P` on the abelian variety.
    ///
    /// Equivalent to calling [`Self::double`] `n` times. For `n = 0`
    /// returns `p` unchanged.
    ///
    /// Used by scalar-multiplication routines and by tests that
    /// exercise the doubling primitive under repetition.
    ///
    /// Cost: `n · (20 base-field mults + 16 add/sub ops)`.
    pub fn double_iterated(&self, p: &ThetaPoint2D<F>, n: u32) -> ThetaPoint2D<F> {
        let mut q = *p;
        for _ in 0..n {
            q = self.double(&q);
        }
        q
    }

    /// Encoded byte length of an abelian variety: just the theta-null
    /// (the doubling constants are re-derived from it via Riemann's
    /// duplication formula on decode, per [`Self::from_theta_null`]).
    ///
    /// Equal to [`ThetaPoint2D::ENCODED_BYTES`] = `8 · F::ENCODED_BYTES`.
    /// At the three SQIsign levels: 256 / 384 / 512 bytes.
    pub const ENCODED_BYTES: usize = ThetaPoint2D::<F>::ENCODED_BYTES;

    /// Encode `self.theta_null` into `out`. The doubling constants are
    /// NOT included — they are re-derived on decode via Riemann's
    /// duplication formula.
    ///
    /// Returns [`Error::BufferTooSmall`] if `out.len() < ENCODED_BYTES`.
    /// Writes exactly `ENCODED_BYTES` bytes on success.
    ///
    /// Round-trip correctness depends on `self.doubling_constants`
    /// having been computed via [`Self::from_theta_null`] originally
    /// — a variety constructed via [`Self::new`] with arbitrary
    /// doubling constants will NOT round-trip (decode re-derives the
    /// canonical constants, not the originals).
    pub fn to_bytes_le(&self, out: &mut [u8]) -> Result<()> {
        self.theta_null.to_bytes_le(out)
    }

    /// Decode an abelian variety from `bytes`. Reads the theta-null
    /// from the first `ENCODED_BYTES` bytes via
    /// [`ThetaPoint2D::from_bytes_le`], then re-derives the doubling
    /// constants via Riemann's duplication formula
    /// (per [`Self::from_theta_null`]).
    ///
    /// Returns [`Error::BufferTooSmall`] if input is too short,
    /// [`Error::NonCanonicalEncoding`] if any `Fp2` component is
    /// non-canonical, or [`Error::InvalidThetaNull`] if the decoded
    /// theta-null is degenerate (i.e. `H(theta_null²)` has a zero
    /// component and the doubling constants are undefined).
    pub fn from_bytes_le(bytes: &[u8]) -> Result<Self> {
        let theta_null = ThetaPoint2D::<F>::from_bytes_le(bytes)?;
        Self::from_theta_null(theta_null).ok_or(Error::InvalidThetaNull)
    }

    /// Constant-time check that `self` satisfies Riemann's duplication
    /// formula — i.e. `self.doubling_constants` is the componentwise
    /// inverse of `H(self.theta_null²)`.
    ///
    /// Returns `Choice::TRUE` iff
    /// `self.doubling_constants ⊙ H(self.theta_null²) == identity`.
    ///
    /// A variety constructed via [`Self::from_theta_null`] always
    /// satisfies this. A variety constructed via [`Self::new`] with
    /// manually-supplied doubling constants may or may not — this
    /// method lets callers verify the invariant explicitly.
    ///
    /// Use cases:
    /// - Validating a variety received over the wire (the
    ///   [`Self::from_bytes_le`] decoder already enforces this by
    ///   construction, but explicit re-verification is occasionally
    ///   useful).
    /// - Sanity checks in test code.
    /// - Debugging suspected programming errors in the Clapotis
    ///   evaluator pipeline.
    ///
    /// Cost: 4 `Fp2` squarings + 4 `Fp2` add/sub (Hadamard) + 4
    /// `Fp2` multiplications + 4 `Fp2::ct_eq` ops.
    ///
    /// Constant-time: yes — uses only `ct_eq` and bitwise `&`.
    pub fn is_consistent_with_theta_null(&self) -> Choice {
        let h_sq = self.theta_null.componentwise_square().hadamard();
        let prod = self.doubling_constants.componentwise_mul(&h_sq);
        prod.ct_eq(&ThetaPoint2D::<F>::identity())
    }

    /// Produce a canonicalised Riemann-consistent variety where
    /// `self.theta_null.x == 1`.
    ///
    /// Two algebraically-equivalent varieties — i.e. two varieties
    /// whose theta-nulls are projectively-equal — canonicalise to
    /// the same in-memory representation, and therefore to the same
    /// bytes when encoded via [`Self::to_bytes_le`].
    ///
    /// Mechanism: normalises `self.theta_null` via
    /// [`ThetaPoint2D::normalize_by_x`], then re-derives the
    /// doubling constants via Riemann's duplication formula
    /// ([`Self::from_theta_null`]). Returns `None` if either step
    /// fails:
    /// - `self.theta_null.x` is zero (cannot normalise along x).
    /// - the normalised theta-null is degenerate (its squared
    ///   Hadamard transform has a zero component).
    ///
    /// **Contract**: the result is always Riemann-consistent
    /// regardless of the input's consistency state. For an input
    /// that was already Riemann-consistent, the result represents
    /// the same algebraic variety. For a non-Riemann-consistent
    /// input, the result is the Riemann-projected canonical form
    /// of the input's theta-null.
    ///
    /// **Idempotent**: `v.canonicalise()?.canonicalise() ==
    /// v.canonicalise()?` for any input that canonicalises
    /// successfully.
    ///
    /// Cost: 1 `Fp2` inversion plus 3 `Fp2` mults (from
    /// `normalize_by_x`), then 4 `Fp2` squarings, Hadamard, and
    /// 4 `Fp2` inversions (from `from_theta_null`).
    pub fn canonicalise(&self) -> Option<Self> {
        let theta_null_n = self.theta_null.normalize_by_x()?;
        Self::from_theta_null(theta_null_n)
    }

    /// Constant-time projective equality for varieties: `Choice::TRUE`
    /// iff `self.theta_null` and `other.theta_null` are projectively
    /// equal (one is a non-zero scalar multiple of the other).
    ///
    /// **Contract for Riemann-consistent inputs**: when both `self`
    /// and `other` satisfy Riemann's duplication formula (i.e. were
    /// constructed via [`Self::from_theta_null`], [`Self::from_bytes_le`],
    /// or [`Self::canonicalise`]), projective equality of the theta-nulls
    /// is sufficient to conclude algebraic equality of the abelian
    /// varieties — the doubling constants are implied by the theta-nulls
    /// via Riemann's formula, so they cannot independently differ.
    ///
    /// **Contract for arbitrary varieties** (constructed via
    /// [`Self::new`] with manually-supplied doubling constants):
    /// this method tests only the theta-null projective equality.
    /// Two varieties may have projectively-equal theta-nulls but
    /// algebraically-different doubling behaviour. Callers requiring
    /// the stronger test should pre-filter via
    /// [`Self::is_consistent_with_theta_null`].
    ///
    /// Constant-time: yes — delegates to
    /// [`ThetaPoint2D::project_equals`].
    ///
    /// Cost: 12 `Fp2` multiplications plus 6 `Fp2::ct_eq` ops, same
    /// as `ThetaPoint2D::project_equals`.
    pub fn project_equals(&self, other: &Self) -> Choice {
        self.theta_null.project_equals(&other.theta_null)
    }

    /// One-step deterministic wire encoding: `self.canonicalise()`
    /// then [`Self::to_bytes_le`].
    ///
    /// Returns:
    /// - [`Error::InvalidThetaNull`] if `self.canonicalise()` fails
    ///   (i.e. `self.theta_null.x` is zero or the normalised theta-null
    ///   is degenerate).
    /// - [`Error::BufferTooSmall`] if `out.len() < ENCODED_BYTES`.
    /// - `Ok(())` on success, having written exactly `ENCODED_BYTES`
    ///   bytes to `out`.
    ///
    /// **Determinism contract**: two algebraically-equivalent varieties
    /// (theta-nulls related by a non-zero scalar) produce identical
    /// output bytes — the canonical-bytes guarantee from
    /// [`Self::canonicalise`] composes with the encoding.
    ///
    /// Convenience wrapper for the common pattern of "produce the
    /// deterministic wire encoding for this variety" — saves callers
    /// from intermediate `Option`-handling.
    pub fn canonicalise_to_bytes(&self, out: &mut [u8]) -> Result<()> {
        let canon = self.canonicalise().ok_or(Error::InvalidThetaNull)?;
        canon.to_bytes_le(out)
    }
}

impl<F: BaseField> ConstantTimeEq for AbelianVariety2D<F> {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.theta_null.ct_eq(&other.theta_null)
            & self.doubling_constants.ct_eq(&other.doubling_constants)
    }
}

impl<F: BaseField> ConditionallySelectable for AbelianVariety2D<F> {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self::new(
            ThetaPoint2D::<F>::conditional_select(&a.theta_null, &b.theta_null, choice),
            ThetaPoint2D::<F>::conditional_select(
                &a.doubling_constants,
                &b.doubling_constants,
                choice,
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf::fp::Fp1Element;

    fn check_zero_is_zero<F: BaseField>() {
        let z = ThetaPoint2D::<F>::zero();
        assert!(
            bool::from(z.is_zero()),
            "S95: ThetaPoint2D::zero().is_zero() must be Choice::TRUE",
        );
    }

    #[test]
    fn zero_is_zero_at_lvl1() {
        check_zero_is_zero::<Fp1Element>();
    }

    #[test]
    fn zero_is_zero_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_zero_is_zero::<Fp3Element>();
    }

    #[test]
    fn zero_is_zero_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_zero_is_zero::<Fp5Element>();
    }

    fn check_identity_is_not_zero<F: BaseField>() {
        let id = ThetaPoint2D::<F>::identity();
        assert!(
            !bool::from(id.is_zero()),
            "S95: ThetaPoint2D::identity() must not satisfy is_zero",
        );
        assert_eq!(
            id,
            ThetaPoint2D::<F>::new(
                Fp2::<F>::one(),
                Fp2::<F>::one(),
                Fp2::<F>::one(),
                Fp2::<F>::one(),
            ),
            "S95: identity = (1, 1, 1, 1)",
        );
    }

    #[test]
    fn identity_is_not_zero_at_lvl1() {
        check_identity_is_not_zero::<Fp1Element>();
    }

    #[test]
    fn identity_is_not_zero_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_identity_is_not_zero::<Fp3Element>();
    }

    #[test]
    fn identity_is_not_zero_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_identity_is_not_zero::<Fp5Element>();
    }

    fn check_ct_eq_reflexive<F: BaseField>() {
        let p = ThetaPoint2D::<F>::new(
            Fp2::<F>::one().double(),
            Fp2::<F>::one(),
            Fp2::<F>::one().double().double(),
            Fp2::<F>::one(),
        );
        assert!(
            bool::from(<ThetaPoint2D<F> as ConstantTimeEq>::ct_eq(&p, &p)),
            "S95: ct_eq must be reflexive",
        );
    }

    #[test]
    fn ct_eq_reflexive_at_lvl1() {
        check_ct_eq_reflexive::<Fp1Element>();
    }

    #[test]
    fn ct_eq_reflexive_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_ct_eq_reflexive::<Fp3Element>();
    }

    #[test]
    fn ct_eq_reflexive_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_ct_eq_reflexive::<Fp5Element>();
    }

    fn check_conditional_select<F: BaseField>() {
        let a = ThetaPoint2D::<F>::zero();
        let b = ThetaPoint2D::<F>::identity();
        let pick_a = ThetaPoint2D::<F>::conditional_select(&a, &b, Choice::from(0));
        let pick_b = ThetaPoint2D::<F>::conditional_select(&a, &b, Choice::from(1));
        assert_eq!(pick_a, a, "S95: conditional_select(_, _, FALSE) returns a");
        assert_eq!(pick_b, b, "S95: conditional_select(_, _, TRUE) returns b");
    }

    #[test]
    fn conditional_select_at_lvl1() {
        check_conditional_select::<Fp1Element>();
    }

    #[test]
    fn conditional_select_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_conditional_select::<Fp3Element>();
    }

    #[test]
    fn conditional_select_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_conditional_select::<Fp5Element>();
    }

    // ── S96 — Hadamard transform on ThetaPoint2D ──

    /// Generic helper: verify the Hadamard transform formula
    /// component-by-component on a specific input `(2, 3, 5, 7)`.
    fn check_hadamard_explicit_formula<F: BaseField>() {
        let one = Fp2::<F>::one();
        let two = one.double();
        let three = two.add(&one);
        let four = two.double();
        let five = four.add(&one);
        let six = three.double();
        let seven = six.add(&one);

        let p = ThetaPoint2D::<F>::new(two, three, five, seven);
        let h = p.hadamard();

        // Expected:
        //   x_h = 2+3+5+7 = 17.
        //   y_h = 2−3+5−7 = −3.
        //   z_h = 2+3−5−7 = −7.
        //   w_h = 2−3−5+7 = 1.
        let seventeen = {
            let mut acc = Fp2::<F>::zero();
            for _ in 0..17 {
                acc = acc.add(&one);
            }
            acc
        };
        let minus_three = three.negate();
        let minus_seven = seven.negate();
        let one_f2 = one;

        assert_eq!(h.x, seventeen, "S96: H[x] = x+y+z+w (17)");
        assert_eq!(h.y, minus_three, "S96: H[y] = x−y+z−w (−3)");
        assert_eq!(h.z, minus_seven, "S96: H[z] = x+y−z−w (−7)");
        assert_eq!(h.w, one_f2, "S96: H[w] = x−y−z+w (1)");
    }

    #[test]
    fn hadamard_explicit_formula_at_lvl1() {
        check_hadamard_explicit_formula::<Fp1Element>();
    }

    #[test]
    fn hadamard_explicit_formula_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_hadamard_explicit_formula::<Fp3Element>();
    }

    #[test]
    fn hadamard_explicit_formula_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_hadamard_explicit_formula::<Fp5Element>();
    }

    /// Generic helper: the foundational property `H² = 4·I`.
    /// Applying the Hadamard transform twice multiplies each
    /// coordinate by 4. Tested on pseudo-random Fp2 inputs (8 samples)
    /// to span the full field magnitude.
    fn check_hadamard_squared_is_4i<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..8 {
            let x = hash_to_fp2::<F>(b"S96-h-sq-x", &[i], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()));
            let y = hash_to_fp2::<F>(b"S96-h-sq-y", &[i], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()));
            let z = hash_to_fp2::<F>(b"S96-h-sq-z", &[i], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()));
            let w = hash_to_fp2::<F>(b"S96-h-sq-w", &[i], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()));
            let p = ThetaPoint2D::<F>::new(x, y, z, w);
            let h_h = p.hadamard().hadamard();
            // Expected: each coordinate multiplied by 4.
            let four_x = x.double().double();
            let four_y = y.double().double();
            let four_z = z.double().double();
            let four_w = w.double().double();
            assert_eq!(h_h.x, four_x, "S96: H²[x] = 4x at iteration {i}");
            assert_eq!(h_h.y, four_y, "S96: H²[y] = 4y at iteration {i}");
            assert_eq!(h_h.z, four_z, "S96: H²[z] = 4z at iteration {i}");
            assert_eq!(h_h.w, four_w, "S96: H²[w] = 4w at iteration {i}");
        }
    }

    #[test]
    fn hadamard_squared_is_4i_at_lvl1() {
        check_hadamard_squared_is_4i::<Fp1Element>();
    }

    #[test]
    fn hadamard_squared_is_4i_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_hadamard_squared_is_4i::<Fp3Element>();
    }

    #[test]
    fn hadamard_squared_is_4i_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_hadamard_squared_is_4i::<Fp5Element>();
    }

    /// Generic helper: the Hadamard transform of the all-zero point
    /// is the all-zero point (linear transform, trivial kernel
    /// property). And the Hadamard transform of the all-one point
    /// `(1, 1, 1, 1)` is `(4, 0, 0, 0)`.
    fn check_hadamard_special_inputs<F: BaseField>() {
        // Zero input → zero output.
        let zero = ThetaPoint2D::<F>::zero();
        let h_zero = zero.hadamard();
        assert_eq!(h_zero, ThetaPoint2D::<F>::zero(), "S96: H(0) = 0");

        // Identity (1, 1, 1, 1) → (4, 0, 0, 0).
        let id = ThetaPoint2D::<F>::identity();
        let h_id = id.hadamard();
        let four = Fp2::<F>::one().double().double();
        let zero_f2 = Fp2::<F>::zero();
        assert_eq!(h_id.x, four, "S96: H(1,1,1,1)[x] = 4");
        assert_eq!(h_id.y, zero_f2, "S96: H(1,1,1,1)[y] = 0");
        assert_eq!(h_id.z, zero_f2, "S96: H(1,1,1,1)[z] = 0");
        assert_eq!(h_id.w, zero_f2, "S96: H(1,1,1,1)[w] = 0");
    }

    #[test]
    fn hadamard_special_inputs_at_lvl1() {
        check_hadamard_special_inputs::<Fp1Element>();
    }

    #[test]
    fn hadamard_special_inputs_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_hadamard_special_inputs::<Fp3Element>();
    }

    #[test]
    fn hadamard_special_inputs_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_hadamard_special_inputs::<Fp5Element>();
    }

    // ── S97 — componentwise square + multiply on ThetaPoint2D ──

    /// Generic helper: componentwise_square on `(2, 3, 5, 7)`
    /// yields `(4, 9, 25, 49)`.
    fn check_componentwise_square_explicit<F: BaseField>() {
        let one = Fp2::<F>::one();
        let two = one.double();
        let three = two.add(&one);
        let five = three.double().sub(&one);
        let seven = five.add(&two);

        let p = ThetaPoint2D::<F>::new(two, three, five, seven);
        let p_sq = p.componentwise_square();

        let four = two.square();
        let nine = three.square();
        let twenty_five = five.square();
        let forty_nine = seven.square();

        assert_eq!(p_sq.x, four, "S97: square[x] = 4");
        assert_eq!(p_sq.y, nine, "S97: square[y] = 9");
        assert_eq!(p_sq.z, twenty_five, "S97: square[z] = 25");
        assert_eq!(p_sq.w, forty_nine, "S97: square[w] = 49");
    }

    #[test]
    fn componentwise_square_explicit_at_lvl1() {
        check_componentwise_square_explicit::<Fp1Element>();
    }

    #[test]
    fn componentwise_square_explicit_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_componentwise_square_explicit::<Fp3Element>();
    }

    #[test]
    fn componentwise_square_explicit_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_componentwise_square_explicit::<Fp5Element>();
    }

    /// Generic helper: `componentwise_square(p) == componentwise_mul(p, p)`
    /// for pseudo-random Fp2 inputs. Connects the two primitives —
    /// squaring is multiplication by self.
    fn check_square_matches_self_mul<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..8 {
            let x = hash_to_fp2::<F>(b"S97-sq-x", &[i], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()));
            let y = hash_to_fp2::<F>(b"S97-sq-y", &[i], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()));
            let z = hash_to_fp2::<F>(b"S97-sq-z", &[i], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()));
            let w = hash_to_fp2::<F>(b"S97-sq-w", &[i], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()));
            let p = ThetaPoint2D::<F>::new(x, y, z, w);
            assert_eq!(
                p.componentwise_square(),
                p.componentwise_mul(&p),
                "S97: square(p) must equal mul(p, p) at iteration {i}",
            );
        }
    }

    #[test]
    fn square_matches_self_mul_at_lvl1() {
        check_square_matches_self_mul::<Fp1Element>();
    }

    #[test]
    fn square_matches_self_mul_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_square_matches_self_mul::<Fp3Element>();
    }

    #[test]
    fn square_matches_self_mul_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_square_matches_self_mul::<Fp5Element>();
    }

    /// Generic helper: `componentwise_mul` is commutative —
    /// `p ⊙ q == q ⊙ p` on pseudo-random inputs.
    fn check_componentwise_mul_commutative<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..8 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let p = ThetaPoint2D::<F>::new(mk(b"px"), mk(b"py"), mk(b"pz"), mk(b"pw"));
            let q = ThetaPoint2D::<F>::new(mk(b"qx"), mk(b"qy"), mk(b"qz"), mk(b"qw"));
            assert_eq!(
                p.componentwise_mul(&q),
                q.componentwise_mul(&p),
                "S97: componentwise_mul must be commutative at iteration {i}",
            );
        }
    }

    #[test]
    fn componentwise_mul_commutative_at_lvl1() {
        check_componentwise_mul_commutative::<Fp1Element>();
    }

    #[test]
    fn componentwise_mul_commutative_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_componentwise_mul_commutative::<Fp3Element>();
    }

    #[test]
    fn componentwise_mul_commutative_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_componentwise_mul_commutative::<Fp5Element>();
    }

    /// Generic helper: `componentwise_mul(p, identity) == p` for any p.
    /// Identity-of-multiplication property at the theta-tuple layer.
    fn check_componentwise_mul_identity<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        let p = ThetaPoint2D::<F>::new(
            hash_to_fp2::<F>(b"id-x", &[0], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one())),
            hash_to_fp2::<F>(b"id-y", &[0], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one())),
            hash_to_fp2::<F>(b"id-z", &[0], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one())),
            hash_to_fp2::<F>(b"id-w", &[0], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one())),
        );
        let result = p.componentwise_mul(&ThetaPoint2D::<F>::identity());
        assert_eq!(
            result, p,
            "S97: componentwise_mul(p, (1,1,1,1)) must equal p",
        );
    }

    #[test]
    fn componentwise_mul_identity_at_lvl1() {
        check_componentwise_mul_identity::<Fp1Element>();
    }

    #[test]
    fn componentwise_mul_identity_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_componentwise_mul_identity::<Fp3Element>();
    }

    #[test]
    fn componentwise_mul_identity_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_componentwise_mul_identity::<Fp5Element>();
    }

    // ── S98 — double() composes H → square → H → mul ──

    /// Generic helper: `double` matches the explicit chained
    /// composition for pseudo-random p and constants. Locks the
    /// composition contract: `double(p, c) == H(square(H(p))) ⊙ c`.
    fn check_double_matches_chain<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..8 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let p = ThetaPoint2D::<F>::new(mk(b"dpx"), mk(b"dpy"), mk(b"dpz"), mk(b"dpw"));
            let c = ThetaPoint2D::<F>::new(mk(b"dcx"), mk(b"dcy"), mk(b"dcz"), mk(b"dcw"));

            // Reference: chain the four primitives explicitly.
            let chained = p
                .hadamard()
                .componentwise_square()
                .hadamard()
                .componentwise_mul(&c);
            let doubled = p.double(&c);

            assert_eq!(
                chained, doubled,
                "S98: double(p, c) must equal H(square(H(p))) ⊙ c at iteration {i}",
            );
        }
    }

    #[test]
    fn double_matches_chain_at_lvl1() {
        check_double_matches_chain::<Fp1Element>();
    }

    #[test]
    fn double_matches_chain_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_double_matches_chain::<Fp3Element>();
    }

    #[test]
    fn double_matches_chain_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_double_matches_chain::<Fp5Element>();
    }

    /// Generic helper: `double(zero, anything) = zero`. The zero point's
    /// Hadamard transform is zero (linear, sends zero to zero), square
    /// of zero is zero, second Hadamard is still zero, and componentwise-
    /// multiplying any constants by zero yields zero. End-to-end: zero
    /// is a fixed point of `double` regardless of constants.
    fn check_double_zero_is_zero<F: BaseField>() {
        let zero = ThetaPoint2D::<F>::zero();
        let arbitrary_constants = ThetaPoint2D::<F>::new(
            Fp2::<F>::one(),
            Fp2::<F>::one().double(),
            Fp2::<F>::one(),
            Fp2::<F>::one(),
        );
        let doubled = zero.double(&arbitrary_constants);
        assert_eq!(
            doubled,
            ThetaPoint2D::<F>::zero(),
            "S98: double(zero, _) must be zero (linearity end-to-end)",
        );
    }

    #[test]
    fn double_zero_is_zero_at_lvl1() {
        check_double_zero_is_zero::<Fp1Element>();
    }

    #[test]
    fn double_zero_is_zero_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_double_zero_is_zero::<Fp3Element>();
    }

    #[test]
    fn double_zero_is_zero_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_double_zero_is_zero::<Fp5Element>();
    }

    /// Generic helper: `double(p, identity_constants) == H(square(H(p)))`.
    /// Verifies that the `componentwise_mul(identity)` step is the
    /// identity operation, so `double(p, (1,1,1,1)) == H(square(H(p)))`.
    fn check_double_with_identity_constants<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        let p = ThetaPoint2D::<F>::new(
            hash_to_fp2::<F>(b"S98-id-x", &[0], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one())),
            hash_to_fp2::<F>(b"S98-id-y", &[0], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one())),
            hash_to_fp2::<F>(b"S98-id-z", &[0], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one())),
            hash_to_fp2::<F>(b"S98-id-w", &[0], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one())),
        );
        let expected = p.hadamard().componentwise_square().hadamard();
        let actual = p.double(&ThetaPoint2D::<F>::identity());
        assert_eq!(
            actual, expected,
            "S98: double(p, identity) must equal H(square(H(p)))",
        );
    }

    #[test]
    fn double_with_identity_constants_at_lvl1() {
        check_double_with_identity_constants::<Fp1Element>();
    }

    #[test]
    fn double_with_identity_constants_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_double_with_identity_constants::<Fp3Element>();
    }

    #[test]
    fn double_with_identity_constants_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_double_with_identity_constants::<Fp5Element>();
    }

    // ── S99 — AbelianVariety2D wrapper for theta-null + doubling constants ──

    /// Generic helper: `AbelianVariety2D::double(p)` matches
    /// `p.double(&variety.doubling_constants)` for pseudo-random
    /// inputs. The wrapper's `double` method is a thin pass-through;
    /// this test locks that contract.
    fn check_variety_double_matches_point_double<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..8 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let theta_null =
                ThetaPoint2D::<F>::new(mk(b"tn-x"), mk(b"tn-y"), mk(b"tn-z"), mk(b"tn-w"));
            let constants =
                ThetaPoint2D::<F>::new(mk(b"dc-x"), mk(b"dc-y"), mk(b"dc-z"), mk(b"dc-w"));
            let variety = AbelianVariety2D::<F>::new(theta_null, constants);
            let p = ThetaPoint2D::<F>::new(mk(b"p-x"), mk(b"p-y"), mk(b"p-z"), mk(b"p-w"));
            let via_variety = variety.double(&p);
            let via_direct = p.double(&constants);
            assert_eq!(
                via_variety, via_direct,
                "S99: variety.double(p) must equal p.double(&variety.doubling_constants) at iteration {i}",
            );
        }
    }

    #[test]
    fn variety_double_matches_point_double_at_lvl1() {
        check_variety_double_matches_point_double::<Fp1Element>();
    }

    #[test]
    fn variety_double_matches_point_double_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_variety_double_matches_point_double::<Fp3Element>();
    }

    #[test]
    fn variety_double_matches_point_double_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_variety_double_matches_point_double::<Fp5Element>();
    }

    /// Generic helper: the AbelianVariety2D struct round-trips its
    /// constructor inputs. `new(theta_null, constants).theta_null ==
    /// theta_null` and same for constants. Locks the struct-field
    /// contract.
    fn check_variety_round_trip<F: BaseField>() {
        let theta_null = ThetaPoint2D::<F>::new(
            Fp2::<F>::one().double(),
            Fp2::<F>::one(),
            Fp2::<F>::one(),
            Fp2::<F>::one().double(),
        );
        let constants = ThetaPoint2D::<F>::new(
            Fp2::<F>::one(),
            Fp2::<F>::one().double(),
            Fp2::<F>::one().double(),
            Fp2::<F>::one(),
        );
        let variety = AbelianVariety2D::<F>::new(theta_null, constants);
        assert_eq!(
            variety.theta_null, theta_null,
            "S99: variety.theta_null must round-trip from constructor",
        );
        assert_eq!(
            variety.doubling_constants, constants,
            "S99: variety.doubling_constants must round-trip from constructor",
        );
    }

    #[test]
    fn variety_round_trip_at_lvl1() {
        check_variety_round_trip::<Fp1Element>();
    }

    #[test]
    fn variety_round_trip_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_variety_round_trip::<Fp3Element>();
    }

    #[test]
    fn variety_round_trip_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_variety_round_trip::<Fp5Element>();
    }

    // ── S100 — dual_theta_null and theta_null_squared accessors ──

    /// Generic helper: `variety.dual_theta_null() == variety.theta_null.hadamard()`.
    /// Locks the accessor contract: the dual is computed via the
    /// theta-null's Hadamard transform.
    fn check_dual_theta_null_matches_hadamard<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..8 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let theta_null = ThetaPoint2D::<F>::new(
                mk(b"S100-tn-x"),
                mk(b"S100-tn-y"),
                mk(b"S100-tn-z"),
                mk(b"S100-tn-w"),
            );
            // Doubling constants don't affect dual_theta_null; use
            // identity for the variety constructor.
            let variety = AbelianVariety2D::<F>::new(theta_null, ThetaPoint2D::<F>::identity());
            assert_eq!(
                variety.dual_theta_null(),
                theta_null.hadamard(),
                "S100: dual_theta_null must equal theta_null.hadamard() at iteration {i}",
            );
        }
    }

    #[test]
    fn dual_theta_null_matches_hadamard_at_lvl1() {
        check_dual_theta_null_matches_hadamard::<Fp1Element>();
    }

    #[test]
    fn dual_theta_null_matches_hadamard_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_dual_theta_null_matches_hadamard::<Fp3Element>();
    }

    #[test]
    fn dual_theta_null_matches_hadamard_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_dual_theta_null_matches_hadamard::<Fp5Element>();
    }

    /// Generic helper: `variety.theta_null_squared() ==
    /// variety.theta_null.componentwise_square()`. Locks the accessor
    /// contract.
    fn check_theta_null_squared_matches_componentwise<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..8 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let theta_null = ThetaPoint2D::<F>::new(
                mk(b"S100-sq-x"),
                mk(b"S100-sq-y"),
                mk(b"S100-sq-z"),
                mk(b"S100-sq-w"),
            );
            let variety = AbelianVariety2D::<F>::new(theta_null, ThetaPoint2D::<F>::identity());
            assert_eq!(
                variety.theta_null_squared(),
                theta_null.componentwise_square(),
                "S100: theta_null_squared must equal componentwise_square at iteration {i}",
            );
        }
    }

    #[test]
    fn theta_null_squared_matches_componentwise_at_lvl1() {
        check_theta_null_squared_matches_componentwise::<Fp1Element>();
    }

    #[test]
    fn theta_null_squared_matches_componentwise_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_theta_null_squared_matches_componentwise::<Fp3Element>();
    }

    #[test]
    fn theta_null_squared_matches_componentwise_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_theta_null_squared_matches_componentwise::<Fp5Element>();
    }

    // ── S101 — componentwise_inverse + from_theta_null ──

    /// Generic helper: `p ⊙ p.componentwise_inverse() == identity`
    /// for non-degenerate pseudo-random p (no zero components).
    fn check_inverse_yields_identity<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..8 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let p = ThetaPoint2D::<F>::new(mk(b"inv-x"), mk(b"inv-y"), mk(b"inv-z"), mk(b"inv-w"));
            // hash_to_fp2 may produce zero (vanishing probability) — skip.
            if bool::from(p.is_zero())
                || bool::from(p.x.is_zero())
                || bool::from(p.y.is_zero())
                || bool::from(p.z.is_zero())
                || bool::from(p.w.is_zero())
            {
                continue;
            }
            let inv = p
                .componentwise_inverse()
                .expect("non-zero components invert");
            let prod = p.componentwise_mul(&inv);
            assert_eq!(
                prod,
                ThetaPoint2D::<F>::identity(),
                "S101: p ⊙ p⁻¹ must equal identity at iteration {i}",
            );
        }
    }

    #[test]
    fn inverse_yields_identity_at_lvl1() {
        check_inverse_yields_identity::<Fp1Element>();
    }

    #[test]
    fn inverse_yields_identity_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_inverse_yields_identity::<Fp3Element>();
    }

    #[test]
    fn inverse_yields_identity_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_inverse_yields_identity::<Fp5Element>();
    }

    /// Generic helper: a point with a zero component returns None
    /// from componentwise_inverse (no inverse exists at zero).
    fn check_inverse_rejects_zero_component<F: BaseField>() {
        // Construct (0, 1, 1, 1) — first component zero.
        let p = ThetaPoint2D::<F>::new(
            Fp2::<F>::zero(),
            Fp2::<F>::one(),
            Fp2::<F>::one(),
            Fp2::<F>::one(),
        );
        assert_eq!(
            p.componentwise_inverse(),
            None,
            "S101: componentwise_inverse must return None for a point with a zero component",
        );
    }

    #[test]
    fn inverse_rejects_zero_component_at_lvl1() {
        check_inverse_rejects_zero_component::<Fp1Element>();
    }

    #[test]
    fn inverse_rejects_zero_component_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_inverse_rejects_zero_component::<Fp3Element>();
    }

    #[test]
    fn inverse_rejects_zero_component_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_inverse_rejects_zero_component::<Fp5Element>();
    }

    /// Generic helper: `from_theta_null` derives doubling constants
    /// such that `H(theta_null²) ⊙ doubling_constants == identity`.
    /// Locks the Riemann's-duplication-formula contract.
    fn check_from_theta_null_yields_inverse_relation<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..8 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let theta_null =
                ThetaPoint2D::<F>::new(mk(b"ftn-x"), mk(b"ftn-y"), mk(b"ftn-z"), mk(b"ftn-w"));
            let variety = match AbelianVariety2D::<F>::from_theta_null(theta_null) {
                Some(v) => v,
                None => continue, // skip degenerate samples
            };
            let h_sq = theta_null.componentwise_square().hadamard();
            let identity_check = h_sq.componentwise_mul(&variety.doubling_constants);
            assert_eq!(
                identity_check,
                ThetaPoint2D::<F>::identity(),
                "S101: H(theta_null²) ⊙ doubling_constants must equal identity at iteration {i}",
            );
        }
    }

    #[test]
    fn from_theta_null_yields_inverse_relation_at_lvl1() {
        check_from_theta_null_yields_inverse_relation::<Fp1Element>();
    }

    #[test]
    fn from_theta_null_yields_inverse_relation_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_from_theta_null_yields_inverse_relation::<Fp3Element>();
    }

    #[test]
    fn from_theta_null_yields_inverse_relation_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_from_theta_null_yields_inverse_relation::<Fp5Element>();
    }

    /// Generic helper: `from_theta_null` returns None for a degenerate
    /// theta-null (one where `H(theta_null²)` has a zero component).
    /// Construct: theta_null with all components equal to one. Then
    /// theta_null² = (1, 1, 1, 1), and H(1,1,1,1) = (4, 0, 0, 0).
    /// Three zero components → None.
    fn check_from_theta_null_rejects_degenerate<F: BaseField>() {
        let degenerate = ThetaPoint2D::<F>::identity();
        let result = AbelianVariety2D::<F>::from_theta_null(degenerate);
        assert_eq!(
            result, None,
            "S101: from_theta_null((1,1,1,1)) must return None — H(1,1,1,1) = (4,0,0,0) has zero components",
        );
    }

    #[test]
    fn from_theta_null_rejects_degenerate_at_lvl1() {
        check_from_theta_null_rejects_degenerate::<Fp1Element>();
    }

    #[test]
    fn from_theta_null_rejects_degenerate_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_from_theta_null_rejects_degenerate::<Fp3Element>();
    }

    #[test]
    fn from_theta_null_rejects_degenerate_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_from_theta_null_rejects_degenerate::<Fp5Element>();
    }

    /// Generic helper: `double_iterated(p, 0)` returns `p` unchanged.
    fn check_double_iterated_zero_is_identity_op<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        let mk = |tag: &[u8]| {
            hash_to_fp2::<F>(tag, &[0], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
        };
        let theta_null = ThetaPoint2D::<F>::new(
            mk(b"di-tn-x"),
            mk(b"di-tn-y"),
            mk(b"di-tn-z"),
            mk(b"di-tn-w"),
        );
        let variety = AbelianVariety2D::<F>::from_theta_null(theta_null)
            .expect("S102: theta-null must be non-degenerate for test");
        let p = ThetaPoint2D::<F>::new(mk(b"di-p-x"), mk(b"di-p-y"), mk(b"di-p-z"), mk(b"di-p-w"));
        let q = variety.double_iterated(&p, 0);
        assert_eq!(q, p, "S102: double_iterated(p, 0) must return p unchanged");
    }

    #[test]
    fn double_iterated_zero_is_identity_op_at_lvl1() {
        check_double_iterated_zero_is_identity_op::<Fp1Element>();
    }

    #[test]
    fn double_iterated_zero_is_identity_op_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_double_iterated_zero_is_identity_op::<Fp3Element>();
    }

    #[test]
    fn double_iterated_zero_is_identity_op_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_double_iterated_zero_is_identity_op::<Fp5Element>();
    }

    /// Generic helper: `double_iterated(p, 1)` equals `variety.double(&p)`.
    fn check_double_iterated_one_matches_double<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..4 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let theta_null = ThetaPoint2D::<F>::new(
                mk(b"di1-tn-x"),
                mk(b"di1-tn-y"),
                mk(b"di1-tn-z"),
                mk(b"di1-tn-w"),
            );
            let variety = match AbelianVariety2D::<F>::from_theta_null(theta_null) {
                Some(v) => v,
                None => continue, // skip degenerate samples
            };
            let p = ThetaPoint2D::<F>::new(
                mk(b"di1-p-x"),
                mk(b"di1-p-y"),
                mk(b"di1-p-z"),
                mk(b"di1-p-w"),
            );
            let one_step = variety.double_iterated(&p, 1);
            let direct = variety.double(&p);
            assert_eq!(
                one_step, direct,
                "S102: double_iterated(p, 1) must equal variety.double(&p) at iteration {i}",
            );
        }
    }

    #[test]
    fn double_iterated_one_matches_double_at_lvl1() {
        check_double_iterated_one_matches_double::<Fp1Element>();
    }

    #[test]
    fn double_iterated_one_matches_double_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_double_iterated_one_matches_double::<Fp3Element>();
    }

    #[test]
    fn double_iterated_one_matches_double_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_double_iterated_one_matches_double::<Fp5Element>();
    }

    /// Generic helper: `double_iterated(double_iterated(p, m), n)` equals
    /// `double_iterated(p, m + n)` for several `(m, n)` pairs. Locks the
    /// composition contract: iterated doubling is additive in the exponent.
    fn check_double_iterated_additive_in_n<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..3 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let theta_null = ThetaPoint2D::<F>::new(
                mk(b"di+-tn-x"),
                mk(b"di+-tn-y"),
                mk(b"di+-tn-z"),
                mk(b"di+-tn-w"),
            );
            let variety = match AbelianVariety2D::<F>::from_theta_null(theta_null) {
                Some(v) => v,
                None => continue, // skip degenerate samples
            };
            let p = ThetaPoint2D::<F>::new(
                mk(b"di+-p-x"),
                mk(b"di+-p-y"),
                mk(b"di+-p-z"),
                mk(b"di+-p-w"),
            );
            for &(m, n) in &[(0u32, 0u32), (0, 3), (2, 0), (1, 1), (2, 3), (3, 2)] {
                let lhs = variety.double_iterated(&variety.double_iterated(&p, m), n);
                let rhs = variety.double_iterated(&p, m + n);
                assert_eq!(
                    lhs, rhs,
                    "S102: double_iterated(_, m+n) must equal composition at (m, n) = ({m}, {n}), iteration {i}",
                );
            }
        }
    }

    #[test]
    fn double_iterated_additive_in_n_at_lvl1() {
        check_double_iterated_additive_in_n::<Fp1Element>();
    }

    #[test]
    fn double_iterated_additive_in_n_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_double_iterated_additive_in_n::<Fp3Element>();
    }

    #[test]
    fn double_iterated_additive_in_n_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_double_iterated_additive_in_n::<Fp5Element>();
    }

    /// Generic helper: `ct_eq` on `AbelianVariety2D` is reflexive
    /// (returns Choice::TRUE on `(v, v)`) and distinguishes distinct
    /// varieties (returns Choice::FALSE on `(v, v')` where `v ≠ v'`).
    fn check_variety_ct_eq<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        let mk = |tag: &[u8], i: u8| {
            hash_to_fp2::<F>(tag, &[i], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
        };
        let theta_null_a = ThetaPoint2D::<F>::new(
            mk(b"va-tn-x", 0),
            mk(b"va-tn-y", 0),
            mk(b"va-tn-z", 0),
            mk(b"va-tn-w", 0),
        );
        let theta_null_b = ThetaPoint2D::<F>::new(
            mk(b"vb-tn-x", 0),
            mk(b"vb-tn-y", 0),
            mk(b"vb-tn-z", 0),
            mk(b"vb-tn-w", 0),
        );
        let v_a = AbelianVariety2D::<F>::from_theta_null(theta_null_a)
            .expect("S103: theta-null A must be non-degenerate");
        let v_b = AbelianVariety2D::<F>::from_theta_null(theta_null_b)
            .expect("S103: theta-null B must be non-degenerate");
        assert!(
            bool::from(v_a.ct_eq(&v_a)),
            "S103: ct_eq must be reflexive on AbelianVariety2D",
        );
        assert!(
            !bool::from(v_a.ct_eq(&v_b)),
            "S103: ct_eq must distinguish distinct varieties",
        );
        // A clone with one component differing in only the doubling_constants
        // must also be distinguished.
        let v_a_perturbed = AbelianVariety2D::<F>::new(v_a.theta_null, v_b.doubling_constants);
        assert!(
            !bool::from(v_a.ct_eq(&v_a_perturbed)),
            "S103: ct_eq must distinguish varieties with same theta_null but different doubling_constants",
        );
    }

    #[test]
    fn variety_ct_eq_at_lvl1() {
        check_variety_ct_eq::<Fp1Element>();
    }

    #[test]
    fn variety_ct_eq_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_variety_ct_eq::<Fp3Element>();
    }

    #[test]
    fn variety_ct_eq_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_variety_ct_eq::<Fp5Element>();
    }

    /// Generic helper: `conditional_select(a, b, choice)` returns `a`
    /// when choice is FALSE and `b` when choice is TRUE.
    fn check_variety_conditional_select<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        let mk = |tag: &[u8], i: u8| {
            hash_to_fp2::<F>(tag, &[i], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
        };
        let theta_null_a = ThetaPoint2D::<F>::new(
            mk(b"cs-a-x", 0),
            mk(b"cs-a-y", 0),
            mk(b"cs-a-z", 0),
            mk(b"cs-a-w", 0),
        );
        let theta_null_b = ThetaPoint2D::<F>::new(
            mk(b"cs-b-x", 0),
            mk(b"cs-b-y", 0),
            mk(b"cs-b-z", 0),
            mk(b"cs-b-w", 0),
        );
        let v_a = AbelianVariety2D::<F>::from_theta_null(theta_null_a)
            .expect("S103: theta-null A must be non-degenerate");
        let v_b = AbelianVariety2D::<F>::from_theta_null(theta_null_b)
            .expect("S103: theta-null B must be non-degenerate");
        let pick_a = AbelianVariety2D::<F>::conditional_select(&v_a, &v_b, Choice::from(0));
        let pick_b = AbelianVariety2D::<F>::conditional_select(&v_a, &v_b, Choice::from(1));
        assert_eq!(
            pick_a, v_a,
            "S103: conditional_select(a, b, FALSE) must return a",
        );
        assert_eq!(
            pick_b, v_b,
            "S103: conditional_select(a, b, TRUE) must return b",
        );
    }

    #[test]
    fn variety_conditional_select_at_lvl1() {
        check_variety_conditional_select::<Fp1Element>();
    }

    #[test]
    fn variety_conditional_select_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_variety_conditional_select::<Fp3Element>();
    }

    #[test]
    fn variety_conditional_select_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_variety_conditional_select::<Fp5Element>();
    }

    /// Generic helper: `conditional_select` round-trips through both
    /// choice branches with the doubling operation. Locks: doubling a
    /// CT-selected variety produces the same result as doubling the
    /// chosen variety directly.
    fn check_variety_conditional_select_commutes_with_double<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        let mk = |tag: &[u8], i: u8| {
            hash_to_fp2::<F>(tag, &[i], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
        };
        let theta_null_a = ThetaPoint2D::<F>::new(
            mk(b"cd-a-x", 0),
            mk(b"cd-a-y", 0),
            mk(b"cd-a-z", 0),
            mk(b"cd-a-w", 0),
        );
        let theta_null_b = ThetaPoint2D::<F>::new(
            mk(b"cd-b-x", 0),
            mk(b"cd-b-y", 0),
            mk(b"cd-b-z", 0),
            mk(b"cd-b-w", 0),
        );
        let v_a = AbelianVariety2D::<F>::from_theta_null(theta_null_a)
            .expect("S103: theta-null A must be non-degenerate");
        let v_b = AbelianVariety2D::<F>::from_theta_null(theta_null_b)
            .expect("S103: theta-null B must be non-degenerate");
        let p = ThetaPoint2D::<F>::new(
            mk(b"cd-p-x", 0),
            mk(b"cd-p-y", 0),
            mk(b"cd-p-z", 0),
            mk(b"cd-p-w", 0),
        );
        let picked = AbelianVariety2D::<F>::conditional_select(&v_a, &v_b, Choice::from(1));
        assert_eq!(
            picked.double(&p),
            v_b.double(&p),
            "S103: doubling on a CT-selected variety must match the chosen variety",
        );
        let picked0 = AbelianVariety2D::<F>::conditional_select(&v_a, &v_b, Choice::from(0));
        assert_eq!(
            picked0.double(&p),
            v_a.double(&p),
            "S103: doubling on a CT-selected variety must match the chosen variety (other branch)",
        );
    }

    #[test]
    fn variety_conditional_select_commutes_with_double_at_lvl1() {
        check_variety_conditional_select_commutes_with_double::<Fp1Element>();
    }

    #[test]
    fn variety_conditional_select_commutes_with_double_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_variety_conditional_select_commutes_with_double::<Fp3Element>();
    }

    #[test]
    fn variety_conditional_select_commutes_with_double_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_variety_conditional_select_commutes_with_double::<Fp5Element>();
    }

    /// Generic helper: `normalize_by_x` returns a point with `x = 1`
    /// and the remaining components scaled by `x⁻¹`.
    fn check_normalize_by_x_yields_one_in_x<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..8 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let p = ThetaPoint2D::<F>::new(mk(b"nx-x"), mk(b"nx-y"), mk(b"nx-z"), mk(b"nx-w"));
            // Skip degenerate samples where x is zero.
            if bool::from(p.x.is_zero()) {
                continue;
            }
            let normed = p.normalize_by_x().expect("S104: non-zero x must normalise");
            assert_eq!(
                normed.x,
                Fp2::<F>::one(),
                "S104: normalised x must equal one at iteration {i}",
            );
            // y/x, z/x, w/x must equal p.{y,z,w} * x_inv.
            let x_inv = p.x.invert().into_option().expect("non-zero inverts");
            assert_eq!(
                normed.y,
                p.y.mul(&x_inv),
                "S104: normalised y must equal original.y * x_inv at iteration {i}",
            );
            assert_eq!(
                normed.z,
                p.z.mul(&x_inv),
                "S104: normalised z must equal original.z * x_inv at iteration {i}",
            );
            assert_eq!(
                normed.w,
                p.w.mul(&x_inv),
                "S104: normalised w must equal original.w * x_inv at iteration {i}",
            );
        }
    }

    #[test]
    fn normalize_by_x_yields_one_in_x_at_lvl1() {
        check_normalize_by_x_yields_one_in_x::<Fp1Element>();
    }

    #[test]
    fn normalize_by_x_yields_one_in_x_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_normalize_by_x_yields_one_in_x::<Fp3Element>();
    }

    #[test]
    fn normalize_by_x_yields_one_in_x_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_normalize_by_x_yields_one_in_x::<Fp5Element>();
    }

    /// Generic helper: `normalize_by_x` is projective-invariant —
    /// `normalize_by_x(c · p) == normalize_by_x(p)` for any non-zero
    /// scalar `c` applied componentwise. Locks the canonical-form
    /// contract.
    fn check_normalize_by_x_is_projective_invariant<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..4 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let p = ThetaPoint2D::<F>::new(mk(b"pi-x"), mk(b"pi-y"), mk(b"pi-z"), mk(b"pi-w"));
            if bool::from(p.x.is_zero()) {
                continue;
            }
            // Scale by a non-zero scalar c.
            let c = mk(b"pi-c");
            if bool::from(c.is_zero()) {
                continue;
            }
            let scaled = ThetaPoint2D::<F>::new(p.x.mul(&c), p.y.mul(&c), p.z.mul(&c), p.w.mul(&c));
            let normed_p = p.normalize_by_x().expect("S104: non-zero x must normalise");
            let normed_scaled = scaled
                .normalize_by_x()
                .expect("S104: scaled non-zero x must normalise");
            assert_eq!(
                normed_p, normed_scaled,
                "S104: normalize_by_x must absorb global scaling at iteration {i}",
            );
        }
    }

    #[test]
    fn normalize_by_x_is_projective_invariant_at_lvl1() {
        check_normalize_by_x_is_projective_invariant::<Fp1Element>();
    }

    #[test]
    fn normalize_by_x_is_projective_invariant_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_normalize_by_x_is_projective_invariant::<Fp3Element>();
    }

    #[test]
    fn normalize_by_x_is_projective_invariant_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_normalize_by_x_is_projective_invariant::<Fp5Element>();
    }

    /// Generic helper: `normalize_by_x` returns `None` for a point
    /// with `x = 0` (the canonical pivot is unavailable).
    fn check_normalize_by_x_rejects_zero_x<F: BaseField>() {
        let p = ThetaPoint2D::<F>::new(
            Fp2::<F>::zero(),
            Fp2::<F>::one(),
            Fp2::<F>::one(),
            Fp2::<F>::one(),
        );
        assert_eq!(
            p.normalize_by_x(),
            None,
            "S104: normalize_by_x must return None when x is zero",
        );
    }

    #[test]
    fn normalize_by_x_rejects_zero_x_at_lvl1() {
        check_normalize_by_x_rejects_zero_x::<Fp1Element>();
    }

    #[test]
    fn normalize_by_x_rejects_zero_x_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_normalize_by_x_rejects_zero_x::<Fp3Element>();
    }

    #[test]
    fn normalize_by_x_rejects_zero_x_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_normalize_by_x_rejects_zero_x::<Fp5Element>();
    }

    /// Generic helper: `project_equals` is reflexive and absorbs
    /// projective scaling — `project_equals(p, c · p) == TRUE` for
    /// any non-zero `c`, and `project_equals(p, q) == FALSE` for
    /// independent random p, q.
    fn check_project_equals<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..6 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let p = ThetaPoint2D::<F>::new(mk(b"pe-x"), mk(b"pe-y"), mk(b"pe-z"), mk(b"pe-w"));
            // Reflexivity: p ~ p.
            assert!(
                bool::from(p.project_equals(&p)),
                "S105: project_equals must be reflexive at iteration {i}",
            );
            // Projective scaling: c · p ~ p for non-zero c.
            let c = mk(b"pe-c");
            if bool::from(c.is_zero()) {
                continue;
            }
            let scaled = ThetaPoint2D::<F>::new(p.x.mul(&c), p.y.mul(&c), p.z.mul(&c), p.w.mul(&c));
            assert!(
                bool::from(p.project_equals(&scaled)),
                "S105: project_equals must absorb non-zero scaling at iteration {i}",
            );
            // Independent point: q derived from a fresh hash; expected non-equal
            // (vanishing chance of accidental projective collision on a random
            // sample — acceptable as a sanity test).
            let q = ThetaPoint2D::<F>::new(mk(b"pe-qx"), mk(b"pe-qy"), mk(b"pe-qz"), mk(b"pe-qw"));
            // Only assert non-equality if not all components zero and not all
            // equal modulo scaling (extremely unlikely on random samples).
            if !bool::from(q.is_zero()) {
                assert!(
                    !bool::from(p.project_equals(&q)),
                    "S105: project_equals must distinguish independent random samples at iteration {i}",
                );
            }
        }
    }

    #[test]
    fn project_equals_at_lvl1() {
        check_project_equals::<Fp1Element>();
    }

    #[test]
    fn project_equals_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_project_equals::<Fp3Element>();
    }

    #[test]
    fn project_equals_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_project_equals::<Fp5Element>();
    }

    /// Generic helper: `project_equals(p, normalize_by_x(p)) == TRUE`.
    /// Locks the composition contract — normalisation preserves the
    /// projective equivalence class.
    fn check_project_equals_with_normalize_by_x<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..4 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let p = ThetaPoint2D::<F>::new(mk(b"pn-x"), mk(b"pn-y"), mk(b"pn-z"), mk(b"pn-w"));
            if bool::from(p.x.is_zero()) {
                continue;
            }
            let normed = p.normalize_by_x().expect("S105: non-zero x must normalise");
            assert!(
                bool::from(p.project_equals(&normed)),
                "S105: project_equals(p, normalize_by_x(p)) must hold at iteration {i}",
            );
        }
    }

    #[test]
    fn project_equals_with_normalize_by_x_at_lvl1() {
        check_project_equals_with_normalize_by_x::<Fp1Element>();
    }

    #[test]
    fn project_equals_with_normalize_by_x_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_project_equals_with_normalize_by_x::<Fp3Element>();
    }

    #[test]
    fn project_equals_with_normalize_by_x_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_project_equals_with_normalize_by_x::<Fp5Element>();
    }

    /// Generic helper: `to_bytes_le` then `from_bytes_le` round-trips
    /// to the original theta-coordinate point on 8 pseudo-random
    /// samples per level. Locks the encoding correctness contract.
    fn check_theta_point_round_trip<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..8 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let p = ThetaPoint2D::<F>::new(mk(b"rt-x"), mk(b"rt-y"), mk(b"rt-z"), mk(b"rt-w"));
            let mut buf = [0u8; 512]; // big enough for L5's 512-byte encoding
            let n = ThetaPoint2D::<F>::ENCODED_BYTES;
            p.to_bytes_le(&mut buf[..n])
                .expect("S106: encode must succeed");
            let p2 = ThetaPoint2D::<F>::from_bytes_le(&buf[..n])
                .expect("S106: decode must succeed on encoder output");
            assert_eq!(
                p, p2,
                "S106: ThetaPoint2D round-trip must preserve at iteration {i}",
            );
        }
    }

    #[test]
    fn theta_point_round_trip_at_lvl1() {
        check_theta_point_round_trip::<Fp1Element>();
    }

    #[test]
    fn theta_point_round_trip_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_theta_point_round_trip::<Fp3Element>();
    }

    #[test]
    fn theta_point_round_trip_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_theta_point_round_trip::<Fp5Element>();
    }

    /// Generic helper: `to_bytes_le` and `from_bytes_le` reject
    /// undersized buffers with `Error::BufferTooSmall`.
    fn check_theta_point_rejects_undersized_buffer<F: BaseField>() {
        let p = ThetaPoint2D::<F>::identity();
        let mut tiny = [0u8; 1];
        let r = p.to_bytes_le(&mut tiny);
        assert_eq!(
            r,
            Err(Error::BufferTooSmall {
                required: ThetaPoint2D::<F>::ENCODED_BYTES,
                provided: 1,
            }),
            "S106: encode must reject undersized buffer",
        );
        let r2 = ThetaPoint2D::<F>::from_bytes_le(&tiny);
        assert_eq!(
            r2,
            Err(Error::BufferTooSmall {
                required: ThetaPoint2D::<F>::ENCODED_BYTES,
                provided: 1,
            }),
            "S106: decode must reject undersized buffer",
        );
    }

    #[test]
    fn theta_point_rejects_undersized_buffer_at_lvl1() {
        check_theta_point_rejects_undersized_buffer::<Fp1Element>();
    }

    #[test]
    fn theta_point_rejects_undersized_buffer_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_theta_point_rejects_undersized_buffer::<Fp3Element>();
    }

    #[test]
    fn theta_point_rejects_undersized_buffer_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_theta_point_rejects_undersized_buffer::<Fp5Element>();
    }

    /// Generic helper: `from_bytes_le` rejects encodings where one of
    /// the four `Fp2` components is non-canonical (`re ≥ p` or
    /// `im ≥ p`). Constructed by encoding the identity point then
    /// overwriting one byte of the `x` component with `0xFF` —
    /// which makes the corresponding Fp1 component exceed the
    /// prime at every NIST level.
    fn check_theta_point_rejects_non_canonical<F: BaseField>() {
        let p = ThetaPoint2D::<F>::identity();
        let mut buf = [0u8; 512];
        let n = ThetaPoint2D::<F>::ENCODED_BYTES;
        p.to_bytes_le(&mut buf[..n])
            .expect("S106: encode must succeed");
        // Force the high byte of the x.re component to 0xFF.
        // At the canonical encoding, only the highest byte of the
        // last Fp1 element is bounded by the prime; setting it to
        // 0xFF guarantees non-canonicality (since 0xFF * 2^(8*(n-1))
        // > p for all NIST primes).
        buf[F::ENCODED_BYTES - 1] = 0xFF;
        let r = ThetaPoint2D::<F>::from_bytes_le(&buf[..n]);
        assert_eq!(
            r,
            Err(Error::NonCanonicalEncoding),
            "S106: decode must reject non-canonical Fp2 component",
        );
    }

    #[test]
    fn theta_point_rejects_non_canonical_at_lvl1() {
        check_theta_point_rejects_non_canonical::<Fp1Element>();
    }

    #[test]
    fn theta_point_rejects_non_canonical_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_theta_point_rejects_non_canonical::<Fp3Element>();
    }

    #[test]
    fn theta_point_rejects_non_canonical_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_theta_point_rejects_non_canonical::<Fp5Element>();
    }

    /// Generic helper: `ENCODED_BYTES` matches the expected formula
    /// `8 · F::ENCODED_BYTES`. Locks the sizing contract used by
    /// callers that pre-allocate decode buffers.
    fn check_theta_point_encoded_bytes_formula<F: BaseField>(expected: usize) {
        assert_eq!(
            ThetaPoint2D::<F>::ENCODED_BYTES,
            expected,
            "S106: ENCODED_BYTES must equal 8 · F::ENCODED_BYTES",
        );
        assert_eq!(
            ThetaPoint2D::<F>::ENCODED_BYTES,
            8 * F::ENCODED_BYTES,
            "S106: ENCODED_BYTES must equal 8 · F::ENCODED_BYTES (algebraic check)",
        );
    }

    #[test]
    fn theta_point_encoded_bytes_at_lvl1() {
        check_theta_point_encoded_bytes_formula::<Fp1Element>(256);
    }

    #[test]
    fn theta_point_encoded_bytes_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_theta_point_encoded_bytes_formula::<Fp3Element>(384);
    }

    #[test]
    fn theta_point_encoded_bytes_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_theta_point_encoded_bytes_formula::<Fp5Element>(512);
    }

    /// Generic helper: an `AbelianVariety2D` constructed via
    /// `from_theta_null` round-trips through `to_bytes_le` then
    /// `from_bytes_le` to a canonically-equal variety.
    fn check_variety_bytes_round_trip<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..6 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let theta_null =
                ThetaPoint2D::<F>::new(mk(b"vrt-x"), mk(b"vrt-y"), mk(b"vrt-z"), mk(b"vrt-w"));
            let variety = match AbelianVariety2D::<F>::from_theta_null(theta_null) {
                Some(v) => v,
                None => continue, // skip degenerate samples
            };
            let mut buf = [0u8; 512];
            let n = AbelianVariety2D::<F>::ENCODED_BYTES;
            variety
                .to_bytes_le(&mut buf[..n])
                .expect("S107: encode must succeed");
            let variety2 = AbelianVariety2D::<F>::from_bytes_le(&buf[..n])
                .expect("S107: decode must succeed on encoder output");
            assert_eq!(
                variety, variety2,
                "S107: AbelianVariety2D round-trip via from_theta_null must preserve at iteration {i}",
            );
        }
    }

    #[test]
    fn variety_bytes_round_trip_at_lvl1() {
        check_variety_bytes_round_trip::<Fp1Element>();
    }

    #[test]
    fn variety_bytes_round_trip_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_variety_bytes_round_trip::<Fp3Element>();
    }

    #[test]
    fn variety_bytes_round_trip_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_variety_bytes_round_trip::<Fp5Element>();
    }

    /// Generic helper: encoding the identity `(1, 1, 1, 1)` theta-null
    /// then decoding must return `Error::InvalidThetaNull` (the
    /// identity is degenerate per S101's `from_theta_null` contract).
    fn check_variety_bytes_decode_rejects_degenerate<F: BaseField>() {
        let theta_null = ThetaPoint2D::<F>::identity();
        let mut buf = [0u8; 512];
        let n = ThetaPoint2D::<F>::ENCODED_BYTES;
        theta_null
            .to_bytes_le(&mut buf[..n])
            .expect("S107: encode the identity must succeed");
        let r = AbelianVariety2D::<F>::from_bytes_le(&buf[..n]);
        assert_eq!(
            r,
            Err(Error::InvalidThetaNull),
            "S107: decode must reject the degenerate identity theta-null",
        );
    }

    #[test]
    fn variety_bytes_decode_rejects_degenerate_at_lvl1() {
        check_variety_bytes_decode_rejects_degenerate::<Fp1Element>();
    }

    #[test]
    fn variety_bytes_decode_rejects_degenerate_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_variety_bytes_decode_rejects_degenerate::<Fp3Element>();
    }

    #[test]
    fn variety_bytes_decode_rejects_degenerate_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_variety_bytes_decode_rejects_degenerate::<Fp5Element>();
    }

    /// Generic helper: encode/decode both reject undersized buffers
    /// with `Error::BufferTooSmall`.
    fn check_variety_bytes_rejects_undersized_buffer<F: BaseField>() {
        // Build a non-degenerate variety so the encode path is reached.
        use crate::hash::hash_to_fp2;
        let mk = |tag: &[u8]| {
            hash_to_fp2::<F>(tag, &[0], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
        };
        let theta_null =
            ThetaPoint2D::<F>::new(mk(b"vub-x"), mk(b"vub-y"), mk(b"vub-z"), mk(b"vub-w"));
        let variety = AbelianVariety2D::<F>::from_theta_null(theta_null)
            .expect("S107: theta-null must be non-degenerate for test");
        let mut tiny = [0u8; 1];
        let r = variety.to_bytes_le(&mut tiny);
        assert_eq!(
            r,
            Err(Error::BufferTooSmall {
                required: AbelianVariety2D::<F>::ENCODED_BYTES,
                provided: 1,
            }),
            "S107: encode must reject undersized buffer",
        );
        let r2 = AbelianVariety2D::<F>::from_bytes_le(&tiny);
        assert_eq!(
            r2,
            Err(Error::BufferTooSmall {
                required: AbelianVariety2D::<F>::ENCODED_BYTES,
                provided: 1,
            }),
            "S107: decode must reject undersized buffer",
        );
    }

    #[test]
    fn variety_bytes_rejects_undersized_buffer_at_lvl1() {
        check_variety_bytes_rejects_undersized_buffer::<Fp1Element>();
    }

    #[test]
    fn variety_bytes_rejects_undersized_buffer_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_variety_bytes_rejects_undersized_buffer::<Fp3Element>();
    }

    #[test]
    fn variety_bytes_rejects_undersized_buffer_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_variety_bytes_rejects_undersized_buffer::<Fp5Element>();
    }

    /// Generic helper: a variety constructed via `from_theta_null`
    /// always satisfies the Riemann's-duplication-formula relation.
    fn check_variety_consistent_after_from_theta_null<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..8 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let theta_null =
                ThetaPoint2D::<F>::new(mk(b"cn-x"), mk(b"cn-y"), mk(b"cn-z"), mk(b"cn-w"));
            let variety = match AbelianVariety2D::<F>::from_theta_null(theta_null) {
                Some(v) => v,
                None => continue,
            };
            assert!(
                bool::from(variety.is_consistent_with_theta_null()),
                "S108: from_theta_null must produce a Riemann-consistent variety at iteration {i}",
            );
        }
    }

    #[test]
    fn variety_consistent_after_from_theta_null_at_lvl1() {
        check_variety_consistent_after_from_theta_null::<Fp1Element>();
    }

    #[test]
    fn variety_consistent_after_from_theta_null_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_variety_consistent_after_from_theta_null::<Fp3Element>();
    }

    #[test]
    fn variety_consistent_after_from_theta_null_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_variety_consistent_after_from_theta_null::<Fp5Element>();
    }

    /// Generic helper: a variety with arbitrary (non-Riemann)
    /// doubling constants — e.g. constants equal to the theta-null
    /// itself — is detected as inconsistent.
    fn check_variety_inconsistent_with_arbitrary_constants<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        let mk = |tag: &[u8]| {
            hash_to_fp2::<F>(tag, &[0], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
        };
        let theta_null = ThetaPoint2D::<F>::new(mk(b"ic-x"), mk(b"ic-y"), mk(b"ic-z"), mk(b"ic-w"));
        // Use the theta-null itself as the (incorrect) doubling constants.
        // Almost certainly NOT the Riemann-consistent choice, so the
        // consistency check should return FALSE.
        let bogus = AbelianVariety2D::<F>::new(theta_null, theta_null);
        assert!(
            !bool::from(bogus.is_consistent_with_theta_null()),
            "S108: variety with theta_null as doubling_constants must be inconsistent",
        );
    }

    #[test]
    fn variety_inconsistent_with_arbitrary_constants_at_lvl1() {
        check_variety_inconsistent_with_arbitrary_constants::<Fp1Element>();
    }

    #[test]
    fn variety_inconsistent_with_arbitrary_constants_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_variety_inconsistent_with_arbitrary_constants::<Fp3Element>();
    }

    #[test]
    fn variety_inconsistent_with_arbitrary_constants_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_variety_inconsistent_with_arbitrary_constants::<Fp5Element>();
    }

    /// Generic helper: a variety that survives a bytes round-trip
    /// (encode then decode) is consistent with its theta-null. Locks
    /// the composition contract between byte serialization and the
    /// Riemann consistency invariant.
    fn check_variety_consistent_after_bytes_round_trip<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..4 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let theta_null =
                ThetaPoint2D::<F>::new(mk(b"cr-x"), mk(b"cr-y"), mk(b"cr-z"), mk(b"cr-w"));
            let variety = match AbelianVariety2D::<F>::from_theta_null(theta_null) {
                Some(v) => v,
                None => continue,
            };
            let mut buf = [0u8; 512];
            let n = AbelianVariety2D::<F>::ENCODED_BYTES;
            variety
                .to_bytes_le(&mut buf[..n])
                .expect("S108: encode must succeed");
            let decoded =
                AbelianVariety2D::<F>::from_bytes_le(&buf[..n]).expect("S108: decode must succeed");
            assert!(
                bool::from(decoded.is_consistent_with_theta_null()),
                "S108: decoded variety must be Riemann-consistent at iteration {i}",
            );
        }
    }

    #[test]
    fn variety_consistent_after_bytes_round_trip_at_lvl1() {
        check_variety_consistent_after_bytes_round_trip::<Fp1Element>();
    }

    #[test]
    fn variety_consistent_after_bytes_round_trip_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_variety_consistent_after_bytes_round_trip::<Fp3Element>();
    }

    #[test]
    fn variety_consistent_after_bytes_round_trip_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_variety_consistent_after_bytes_round_trip::<Fp5Element>();
    }

    /// Generic helper: `normalize_by_pivot(p, 0)` matches `normalize_by_x(p)`
    /// — backwards-compatibility with S104's single-pivot variant.
    fn check_normalize_by_pivot_zero_matches_by_x<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..6 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let p = ThetaPoint2D::<F>::new(mk(b"np-x"), mk(b"np-y"), mk(b"np-z"), mk(b"np-w"));
            if bool::from(p.x.is_zero()) {
                continue;
            }
            let by_x = p
                .normalize_by_x()
                .expect("S109: non-zero x must normalise via normalize_by_x");
            let by_pivot_0 = p
                .normalize_by_pivot(0)
                .expect("S109: non-zero x must normalise via normalize_by_pivot(0)");
            assert_eq!(
                by_x, by_pivot_0,
                "S109: normalize_by_pivot(0) must match normalize_by_x at iteration {i}",
            );
        }
    }

    #[test]
    fn normalize_by_pivot_zero_matches_by_x_at_lvl1() {
        check_normalize_by_pivot_zero_matches_by_x::<Fp1Element>();
    }

    #[test]
    fn normalize_by_pivot_zero_matches_by_x_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_normalize_by_pivot_zero_matches_by_x::<Fp3Element>();
    }

    #[test]
    fn normalize_by_pivot_zero_matches_by_x_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_normalize_by_pivot_zero_matches_by_x::<Fp5Element>();
    }

    /// Generic helper: for each pivot in `{0,1,2,3}`, the chosen
    /// component equals one after normalisation, and the result is
    /// projectively-equal to the original (via S105's `project_equals`).
    fn check_normalize_by_pivot_each_index<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..4 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let p = ThetaPoint2D::<F>::new(mk(b"np4-x"), mk(b"np4-y"), mk(b"np4-z"), mk(b"np4-w"));
            for pivot in 0u8..4 {
                let component = match pivot {
                    0 => p.x,
                    1 => p.y,
                    2 => p.z,
                    3 => p.w,
                    _ => unreachable!(),
                };
                if bool::from(component.is_zero()) {
                    continue;
                }
                let normed = p
                    .normalize_by_pivot(pivot)
                    .expect("S109: non-zero pivot must normalise");
                let chosen = match pivot {
                    0 => normed.x,
                    1 => normed.y,
                    2 => normed.z,
                    3 => normed.w,
                    _ => unreachable!(),
                };
                assert_eq!(
                    chosen,
                    Fp2::<F>::one(),
                    "S109: normalize_by_pivot({pivot}) must make the pivot component one at iteration {i}",
                );
                assert!(
                    bool::from(p.project_equals(&normed)),
                    "S109: normalize_by_pivot({pivot}) must preserve projective equivalence at iteration {i}",
                );
            }
        }
    }

    #[test]
    fn normalize_by_pivot_each_index_at_lvl1() {
        check_normalize_by_pivot_each_index::<Fp1Element>();
    }

    #[test]
    fn normalize_by_pivot_each_index_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_normalize_by_pivot_each_index::<Fp3Element>();
    }

    #[test]
    fn normalize_by_pivot_each_index_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_normalize_by_pivot_each_index::<Fp5Element>();
    }

    /// Generic helper: out-of-range pivot indices (>= 4) return None
    /// without inspecting the point. Locks the contract that the
    /// caller cannot accidentally normalise on a phantom 5th component.
    fn check_normalize_by_pivot_rejects_out_of_range<F: BaseField>() {
        let p = ThetaPoint2D::<F>::identity();
        for pivot in 4u8..=255 {
            assert_eq!(
                p.normalize_by_pivot(pivot),
                None,
                "S109: normalize_by_pivot must return None for out-of-range pivot {pivot}",
            );
        }
    }

    #[test]
    fn normalize_by_pivot_rejects_out_of_range_at_lvl1() {
        check_normalize_by_pivot_rejects_out_of_range::<Fp1Element>();
    }

    #[test]
    fn normalize_by_pivot_rejects_out_of_range_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_normalize_by_pivot_rejects_out_of_range::<Fp3Element>();
    }

    #[test]
    fn normalize_by_pivot_rejects_out_of_range_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_normalize_by_pivot_rejects_out_of_range::<Fp5Element>();
    }

    /// Generic helper: zero component at the chosen pivot returns
    /// None (the canonical pivot is unavailable along that axis).
    fn check_normalize_by_pivot_rejects_zero_component<F: BaseField>() {
        let one = Fp2::<F>::one();
        let zero = Fp2::<F>::zero();
        // For each pivot index, build a point where ONLY that component
        // is zero, and verify normalize_by_pivot(pivot) returns None.
        let cases: [(u8, ThetaPoint2D<F>); 4] = [
            (0, ThetaPoint2D::<F>::new(zero, one, one, one)),
            (1, ThetaPoint2D::<F>::new(one, zero, one, one)),
            (2, ThetaPoint2D::<F>::new(one, one, zero, one)),
            (3, ThetaPoint2D::<F>::new(one, one, one, zero)),
        ];
        for (pivot, p) in cases {
            assert_eq!(
                p.normalize_by_pivot(pivot),
                None,
                "S109: normalize_by_pivot must return None when component {pivot} is zero",
            );
        }
    }

    #[test]
    fn normalize_by_pivot_rejects_zero_component_at_lvl1() {
        check_normalize_by_pivot_rejects_zero_component::<Fp1Element>();
    }

    #[test]
    fn normalize_by_pivot_rejects_zero_component_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_normalize_by_pivot_rejects_zero_component::<Fp3Element>();
    }

    #[test]
    fn normalize_by_pivot_rejects_zero_component_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_normalize_by_pivot_rejects_zero_component::<Fp5Element>();
    }

    /// Generic helper: `is_normalised_by_x` returns TRUE for the
    /// identity point `(1, 1, 1, 1)` and for any `normalize_by_x`
    /// output, and FALSE for a point whose `x` is not one.
    fn check_is_normalised_by_x<F: BaseField>() {
        // Identity has x = 1 → TRUE.
        assert!(
            bool::from(ThetaPoint2D::<F>::identity().is_normalised_by_x()),
            "S110: identity must be normalised by x",
        );
        // Zero point has x = 0 → FALSE.
        assert!(
            !bool::from(ThetaPoint2D::<F>::zero().is_normalised_by_x()),
            "S110: zero point must NOT be normalised by x",
        );
        // Postcondition: normalize_by_x output is normalised by x.
        use crate::hash::hash_to_fp2;
        for i in 0u8..4 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let p = ThetaPoint2D::<F>::new(mk(b"nx-x"), mk(b"nx-y"), mk(b"nx-z"), mk(b"nx-w"));
            if bool::from(p.x.is_zero()) {
                continue;
            }
            let normed = p.normalize_by_x().expect("S110: non-zero x must normalise");
            assert!(
                bool::from(normed.is_normalised_by_x()),
                "S110: normalize_by_x output must be normalised by x at iteration {i}",
            );
        }
    }

    #[test]
    fn is_normalised_by_x_at_lvl1() {
        check_is_normalised_by_x::<Fp1Element>();
    }

    #[test]
    fn is_normalised_by_x_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_is_normalised_by_x::<Fp3Element>();
    }

    #[test]
    fn is_normalised_by_x_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_is_normalised_by_x::<Fp5Element>();
    }

    /// Generic helper: `is_normalised_by_pivot` matches
    /// `is_normalised_by_x` at pivot 0, returns TRUE iff the chosen
    /// pivot component equals one, and returns FALSE for out-of-range
    /// pivots.
    fn check_is_normalised_by_pivot<F: BaseField>() {
        // Pivot 0 must match is_normalised_by_x on the identity.
        let id = ThetaPoint2D::<F>::identity();
        for pivot in 0u8..4 {
            assert!(
                bool::from(id.is_normalised_by_pivot(pivot)),
                "S110: identity must be normalised by every pivot 0..4 (pivot={pivot})",
            );
        }
        // Out-of-range pivots must return FALSE even on the identity.
        for pivot in 4u8..=10 {
            assert!(
                !bool::from(id.is_normalised_by_pivot(pivot)),
                "S110: out-of-range pivot must return FALSE (pivot={pivot})",
            );
        }
        // Composition contract: normalize_by_pivot(p, k).is_normalised_by_pivot(k) == TRUE
        use crate::hash::hash_to_fp2;
        for i in 0u8..3 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let p = ThetaPoint2D::<F>::new(mk(b"np-x"), mk(b"np-y"), mk(b"np-z"), mk(b"np-w"));
            for pivot in 0u8..4 {
                let component = match pivot {
                    0 => p.x,
                    1 => p.y,
                    2 => p.z,
                    3 => p.w,
                    _ => unreachable!(),
                };
                if bool::from(component.is_zero()) {
                    continue;
                }
                let normed = p
                    .normalize_by_pivot(pivot)
                    .expect("S110: non-zero pivot component must normalise");
                assert!(
                    bool::from(normed.is_normalised_by_pivot(pivot)),
                    "S110: normalize_by_pivot output must be normalised by that pivot (pivot={pivot}, iter={i})",
                );
            }
        }
    }

    #[test]
    fn is_normalised_by_pivot_at_lvl1() {
        check_is_normalised_by_pivot::<Fp1Element>();
    }

    #[test]
    fn is_normalised_by_pivot_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_is_normalised_by_pivot::<Fp3Element>();
    }

    #[test]
    fn is_normalised_by_pivot_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_is_normalised_by_pivot::<Fp5Element>();
    }

    /// Generic helper: `is_normalised_by_pivot(p, 0)` agrees with
    /// `is_normalised_by_x(p)` on all samples — locks the
    /// backwards-compatibility contract.
    fn check_is_normalised_pivot_zero_matches_by_x<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..4 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let p = ThetaPoint2D::<F>::new(mk(b"nm-x"), mk(b"nm-y"), mk(b"nm-z"), mk(b"nm-w"));
            assert_eq!(
                bool::from(p.is_normalised_by_x()),
                bool::from(p.is_normalised_by_pivot(0)),
                "S110: is_normalised_by_pivot(0) must agree with is_normalised_by_x at iteration {i}",
            );
        }
    }

    #[test]
    fn is_normalised_pivot_zero_matches_by_x_at_lvl1() {
        check_is_normalised_pivot_zero_matches_by_x::<Fp1Element>();
    }

    #[test]
    fn is_normalised_pivot_zero_matches_by_x_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_is_normalised_pivot_zero_matches_by_x::<Fp3Element>();
    }

    #[test]
    fn is_normalised_pivot_zero_matches_by_x_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_is_normalised_pivot_zero_matches_by_x::<Fp5Element>();
    }

    /// Generic helper: `canonicalise` on a variety constructed via
    /// `from_theta_null` yields a variety whose theta-null is
    /// x-normalised (component `x` equals one) AND which is
    /// Riemann-consistent.
    fn check_variety_canonicalise_postconditions<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..6 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let theta_null =
                ThetaPoint2D::<F>::new(mk(b"cv-x"), mk(b"cv-y"), mk(b"cv-z"), mk(b"cv-w"));
            let v = match AbelianVariety2D::<F>::from_theta_null(theta_null) {
                Some(v) => v,
                None => continue,
            };
            // canonicalise needs theta_null.x != 0; from_theta_null doesn't guarantee
            // this, so skip if zero.
            if bool::from(v.theta_null.x.is_zero()) {
                continue;
            }
            let canon = v.canonicalise().expect("S111: canonicalise must succeed");
            assert!(
                bool::from(canon.theta_null.is_normalised_by_x()),
                "S111: canonicalised theta_null.x must equal one at iteration {i}",
            );
            assert!(
                bool::from(canon.is_consistent_with_theta_null()),
                "S111: canonicalised variety must be Riemann-consistent at iteration {i}",
            );
        }
    }

    #[test]
    fn variety_canonicalise_postconditions_at_lvl1() {
        check_variety_canonicalise_postconditions::<Fp1Element>();
    }

    #[test]
    fn variety_canonicalise_postconditions_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_variety_canonicalise_postconditions::<Fp3Element>();
    }

    #[test]
    fn variety_canonicalise_postconditions_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_variety_canonicalise_postconditions::<Fp5Element>();
    }

    /// Generic helper: `canonicalise` is idempotent.
    fn check_variety_canonicalise_idempotent<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..4 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let theta_null =
                ThetaPoint2D::<F>::new(mk(b"ci-x"), mk(b"ci-y"), mk(b"ci-z"), mk(b"ci-w"));
            let v = match AbelianVariety2D::<F>::from_theta_null(theta_null) {
                Some(v) => v,
                None => continue,
            };
            let canon = match v.canonicalise() {
                Some(c) => c,
                None => continue,
            };
            let canon2 = canon
                .canonicalise()
                .expect("S111: canonicalising a canonical variety must succeed");
            assert_eq!(
                canon, canon2,
                "S111: canonicalise must be idempotent at iteration {i}",
            );
        }
    }

    #[test]
    fn variety_canonicalise_idempotent_at_lvl1() {
        check_variety_canonicalise_idempotent::<Fp1Element>();
    }

    #[test]
    fn variety_canonicalise_idempotent_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_variety_canonicalise_idempotent::<Fp3Element>();
    }

    #[test]
    fn variety_canonicalise_idempotent_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_variety_canonicalise_idempotent::<Fp5Element>();
    }

    /// Generic helper: two algebraically-equivalent varieties (built
    /// from theta_null and a non-zero scalar multiple of theta_null)
    /// canonicalise to identical varieties AND identical byte
    /// encodings via S107's `to_bytes_le`. Locks the canonical-bytes
    /// contract used by deterministic wire-format consumers.
    fn check_variety_canonicalise_yields_canonical_bytes<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..4 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let theta_null =
                ThetaPoint2D::<F>::new(mk(b"cb-x"), mk(b"cb-y"), mk(b"cb-z"), mk(b"cb-w"));
            if bool::from(theta_null.x.is_zero()) {
                continue;
            }
            let v_a = match AbelianVariety2D::<F>::from_theta_null(theta_null) {
                Some(v) => v,
                None => continue,
            };
            // Build a scaled twin: theta_null_b = c * theta_null.
            let c = mk(b"cb-c");
            if bool::from(c.is_zero()) {
                continue;
            }
            let theta_null_scaled = ThetaPoint2D::<F>::new(
                theta_null.x.mul(&c),
                theta_null.y.mul(&c),
                theta_null.z.mul(&c),
                theta_null.w.mul(&c),
            );
            let v_b = match AbelianVariety2D::<F>::from_theta_null(theta_null_scaled) {
                Some(v) => v,
                None => continue,
            };
            // Canonicalise both.
            let canon_a = v_a
                .canonicalise()
                .expect("S111: canonicalise A must succeed");
            let canon_b = v_b
                .canonicalise()
                .expect("S111: canonicalise B must succeed");
            assert_eq!(
                canon_a, canon_b,
                "S111: algebraically-equivalent varieties must canonicalise to equal at iteration {i}",
            );
            // Encode both — bytes must match.
            let mut buf_a = [0u8; 512];
            let mut buf_b = [0u8; 512];
            let n = AbelianVariety2D::<F>::ENCODED_BYTES;
            canon_a
                .to_bytes_le(&mut buf_a[..n])
                .expect("S111: encode A must succeed");
            canon_b
                .to_bytes_le(&mut buf_b[..n])
                .expect("S111: encode B must succeed");
            assert_eq!(
                &buf_a[..n],
                &buf_b[..n],
                "S111: canonicalised algebraically-equivalent varieties must encode to identical bytes at iteration {i}",
            );
        }
    }

    #[test]
    fn variety_canonicalise_yields_canonical_bytes_at_lvl1() {
        check_variety_canonicalise_yields_canonical_bytes::<Fp1Element>();
    }

    #[test]
    fn variety_canonicalise_yields_canonical_bytes_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_variety_canonicalise_yields_canonical_bytes::<Fp3Element>();
    }

    #[test]
    fn variety_canonicalise_yields_canonical_bytes_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_variety_canonicalise_yields_canonical_bytes::<Fp5Element>();
    }

    /// Generic helper: `canonicalise` returns None when
    /// `self.theta_null.x` is zero (the x-pivot is unavailable).
    fn check_variety_canonicalise_rejects_zero_x<F: BaseField>() {
        let theta_null = ThetaPoint2D::<F>::new(
            Fp2::<F>::zero(),
            Fp2::<F>::one(),
            Fp2::<F>::one(),
            Fp2::<F>::one(),
        );
        // Build via `new` since from_theta_null may also reject.
        let v = AbelianVariety2D::<F>::new(theta_null, ThetaPoint2D::<F>::identity());
        assert_eq!(
            v.canonicalise(),
            None,
            "S111: canonicalise must return None when theta_null.x is zero",
        );
    }

    #[test]
    fn variety_canonicalise_rejects_zero_x_at_lvl1() {
        check_variety_canonicalise_rejects_zero_x::<Fp1Element>();
    }

    #[test]
    fn variety_canonicalise_rejects_zero_x_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_variety_canonicalise_rejects_zero_x::<Fp3Element>();
    }

    #[test]
    fn variety_canonicalise_rejects_zero_x_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_variety_canonicalise_rejects_zero_x::<Fp5Element>();
    }

    /// Generic helper: `AbelianVariety2D::project_equals` is reflexive
    /// on `from_theta_null`-built varieties.
    fn check_variety_project_equals_reflexive<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..6 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let theta_null =
                ThetaPoint2D::<F>::new(mk(b"pr-x"), mk(b"pr-y"), mk(b"pr-z"), mk(b"pr-w"));
            let v = match AbelianVariety2D::<F>::from_theta_null(theta_null) {
                Some(v) => v,
                None => continue,
            };
            assert!(
                bool::from(v.project_equals(&v)),
                "S112: AbelianVariety2D::project_equals must be reflexive at iteration {i}",
            );
        }
    }

    #[test]
    fn variety_project_equals_reflexive_at_lvl1() {
        check_variety_project_equals_reflexive::<Fp1Element>();
    }

    #[test]
    fn variety_project_equals_reflexive_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_variety_project_equals_reflexive::<Fp3Element>();
    }

    #[test]
    fn variety_project_equals_reflexive_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_variety_project_equals_reflexive::<Fp5Element>();
    }

    /// Generic helper: two varieties built from algebraically-equivalent
    /// theta-nulls (one a non-zero scalar multiple of the other) compare
    /// projective-equal even though their in-memory `==` is FALSE.
    fn check_variety_project_equals_absorbs_scaling<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..4 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let theta_null =
                ThetaPoint2D::<F>::new(mk(b"ps-x"), mk(b"ps-y"), mk(b"ps-z"), mk(b"ps-w"));
            let v_a = match AbelianVariety2D::<F>::from_theta_null(theta_null) {
                Some(v) => v,
                None => continue,
            };
            let c = mk(b"ps-c");
            if bool::from(c.is_zero()) {
                continue;
            }
            let theta_null_scaled = ThetaPoint2D::<F>::new(
                theta_null.x.mul(&c),
                theta_null.y.mul(&c),
                theta_null.z.mul(&c),
                theta_null.w.mul(&c),
            );
            let v_b = match AbelianVariety2D::<F>::from_theta_null(theta_null_scaled) {
                Some(v) => v,
                None => continue,
            };
            assert!(
                bool::from(v_a.project_equals(&v_b)),
                "S112: project_equals must absorb non-zero theta-null scaling at iteration {i}",
            );
        }
    }

    #[test]
    fn variety_project_equals_absorbs_scaling_at_lvl1() {
        check_variety_project_equals_absorbs_scaling::<Fp1Element>();
    }

    #[test]
    fn variety_project_equals_absorbs_scaling_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_variety_project_equals_absorbs_scaling::<Fp3Element>();
    }

    #[test]
    fn variety_project_equals_absorbs_scaling_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_variety_project_equals_absorbs_scaling::<Fp5Element>();
    }

    /// Generic helper: independent random varieties compare
    /// project-non-equal.
    fn check_variety_project_equals_distinguishes_independent<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..4 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let tn_a = ThetaPoint2D::<F>::new(mk(b"da-x"), mk(b"da-y"), mk(b"da-z"), mk(b"da-w"));
            let tn_b = ThetaPoint2D::<F>::new(mk(b"db-x"), mk(b"db-y"), mk(b"db-z"), mk(b"db-w"));
            let v_a = match AbelianVariety2D::<F>::from_theta_null(tn_a) {
                Some(v) => v,
                None => continue,
            };
            let v_b = match AbelianVariety2D::<F>::from_theta_null(tn_b) {
                Some(v) => v,
                None => continue,
            };
            assert!(
                !bool::from(v_a.project_equals(&v_b)),
                "S112: project_equals must distinguish independent random varieties at iteration {i}",
            );
        }
    }

    #[test]
    fn variety_project_equals_distinguishes_independent_at_lvl1() {
        check_variety_project_equals_distinguishes_independent::<Fp1Element>();
    }

    #[test]
    fn variety_project_equals_distinguishes_independent_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_variety_project_equals_distinguishes_independent::<Fp3Element>();
    }

    #[test]
    fn variety_project_equals_distinguishes_independent_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_variety_project_equals_distinguishes_independent::<Fp5Element>();
    }

    /// Generic helper: `v.canonicalise().project_equals(&v) == TRUE`
    /// — canonicalisation preserves the projective equivalence class.
    fn check_variety_project_equals_with_canonicalise<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..4 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let theta_null =
                ThetaPoint2D::<F>::new(mk(b"pc-x"), mk(b"pc-y"), mk(b"pc-z"), mk(b"pc-w"));
            let v = match AbelianVariety2D::<F>::from_theta_null(theta_null) {
                Some(v) => v,
                None => continue,
            };
            let canon = match v.canonicalise() {
                Some(c) => c,
                None => continue,
            };
            assert!(
                bool::from(v.project_equals(&canon)),
                "S112: project_equals(v, canonicalise(v)) must be TRUE at iteration {i}",
            );
        }
    }

    #[test]
    fn variety_project_equals_with_canonicalise_at_lvl1() {
        check_variety_project_equals_with_canonicalise::<Fp1Element>();
    }

    #[test]
    fn variety_project_equals_with_canonicalise_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_variety_project_equals_with_canonicalise::<Fp3Element>();
    }

    #[test]
    fn variety_project_equals_with_canonicalise_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_variety_project_equals_with_canonicalise::<Fp5Element>();
    }

    /// Generic helper: `canonicalise_to_bytes` produces identical
    /// bytes for two algebraically-equivalent varieties (theta-nulls
    /// related by a non-zero scalar), without an intermediate
    /// `canonicalise` step.
    fn check_canonicalise_to_bytes_is_deterministic<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..4 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let theta_null =
                ThetaPoint2D::<F>::new(mk(b"cb-x"), mk(b"cb-y"), mk(b"cb-z"), mk(b"cb-w"));
            if bool::from(theta_null.x.is_zero()) {
                continue;
            }
            let v_a = match AbelianVariety2D::<F>::from_theta_null(theta_null) {
                Some(v) => v,
                None => continue,
            };
            let c = mk(b"cb-c");
            if bool::from(c.is_zero()) {
                continue;
            }
            let theta_null_scaled = ThetaPoint2D::<F>::new(
                theta_null.x.mul(&c),
                theta_null.y.mul(&c),
                theta_null.z.mul(&c),
                theta_null.w.mul(&c),
            );
            let v_b = match AbelianVariety2D::<F>::from_theta_null(theta_null_scaled) {
                Some(v) => v,
                None => continue,
            };
            let mut buf_a = [0u8; 512];
            let mut buf_b = [0u8; 512];
            let n = AbelianVariety2D::<F>::ENCODED_BYTES;
            v_a.canonicalise_to_bytes(&mut buf_a[..n])
                .expect("S113: A canonicalise_to_bytes must succeed");
            v_b.canonicalise_to_bytes(&mut buf_b[..n])
                .expect("S113: B canonicalise_to_bytes must succeed");
            assert_eq!(
                &buf_a[..n],
                &buf_b[..n],
                "S113: canonicalise_to_bytes must yield identical bytes for algebraically-equivalent varieties at iteration {i}",
            );
        }
    }

    #[test]
    fn canonicalise_to_bytes_is_deterministic_at_lvl1() {
        check_canonicalise_to_bytes_is_deterministic::<Fp1Element>();
    }

    #[test]
    fn canonicalise_to_bytes_is_deterministic_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_canonicalise_to_bytes_is_deterministic::<Fp3Element>();
    }

    #[test]
    fn canonicalise_to_bytes_is_deterministic_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_canonicalise_to_bytes_is_deterministic::<Fp5Element>();
    }

    /// Generic helper: `canonicalise_to_bytes` returns
    /// `Error::InvalidThetaNull` for a variety whose `theta_null.x`
    /// is zero (no x-pivot available).
    fn check_canonicalise_to_bytes_rejects_zero_x<F: BaseField>() {
        let theta_null = ThetaPoint2D::<F>::new(
            Fp2::<F>::zero(),
            Fp2::<F>::one(),
            Fp2::<F>::one(),
            Fp2::<F>::one(),
        );
        let v = AbelianVariety2D::<F>::new(theta_null, ThetaPoint2D::<F>::identity());
        let mut buf = [0u8; 512];
        let n = AbelianVariety2D::<F>::ENCODED_BYTES;
        let r = v.canonicalise_to_bytes(&mut buf[..n]);
        assert_eq!(
            r,
            Err(Error::InvalidThetaNull),
            "S113: canonicalise_to_bytes must return InvalidThetaNull when theta_null.x is zero",
        );
    }

    #[test]
    fn canonicalise_to_bytes_rejects_zero_x_at_lvl1() {
        check_canonicalise_to_bytes_rejects_zero_x::<Fp1Element>();
    }

    #[test]
    fn canonicalise_to_bytes_rejects_zero_x_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_canonicalise_to_bytes_rejects_zero_x::<Fp3Element>();
    }

    #[test]
    fn canonicalise_to_bytes_rejects_zero_x_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_canonicalise_to_bytes_rejects_zero_x::<Fp5Element>();
    }

    /// Generic helper: `canonicalise_to_bytes` returns
    /// `Error::BufferTooSmall` for an undersized output buffer.
    fn check_canonicalise_to_bytes_rejects_undersized_buffer<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        let mk = |tag: &[u8]| {
            hash_to_fp2::<F>(tag, &[0], 16)
                .into_option()
                .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
        };
        let theta_null = ThetaPoint2D::<F>::new(mk(b"cu-x"), mk(b"cu-y"), mk(b"cu-z"), mk(b"cu-w"));
        let v = AbelianVariety2D::<F>::from_theta_null(theta_null)
            .expect("S113: theta-null must be non-degenerate for test");
        let mut tiny = [0u8; 1];
        let r = v.canonicalise_to_bytes(&mut tiny);
        assert_eq!(
            r,
            Err(Error::BufferTooSmall {
                required: AbelianVariety2D::<F>::ENCODED_BYTES,
                provided: 1,
            }),
            "S113: canonicalise_to_bytes must reject undersized buffer",
        );
    }

    #[test]
    fn canonicalise_to_bytes_rejects_undersized_buffer_at_lvl1() {
        check_canonicalise_to_bytes_rejects_undersized_buffer::<Fp1Element>();
    }

    #[test]
    fn canonicalise_to_bytes_rejects_undersized_buffer_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_canonicalise_to_bytes_rejects_undersized_buffer::<Fp3Element>();
    }

    #[test]
    fn canonicalise_to_bytes_rejects_undersized_buffer_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_canonicalise_to_bytes_rejects_undersized_buffer::<Fp5Element>();
    }

    // ── S114 — theta-coordinate differential addition (provisional) ──

    /// Generic helper: `diff_add` is symmetric in its first two
    /// arguments (commutativity of addition). For 6 random
    /// `(p, q, r)` triples per level where `H(r)` has no zero
    /// component, `diff_add(p, q, r) == diff_add(q, p, r)`.
    fn check_diff_add_commutes<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..6 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let p = ThetaPoint2D::<F>::new(
                mk(b"dac-p-x"),
                mk(b"dac-p-y"),
                mk(b"dac-p-z"),
                mk(b"dac-p-w"),
            );
            let q = ThetaPoint2D::<F>::new(
                mk(b"dac-q-x"),
                mk(b"dac-q-y"),
                mk(b"dac-q-z"),
                mk(b"dac-q-w"),
            );
            let r = ThetaPoint2D::<F>::new(
                mk(b"dac-r-x"),
                mk(b"dac-r-y"),
                mk(b"dac-r-z"),
                mk(b"dac-r-w"),
            );
            // Skip samples where H(r) has a zero component (diff_add returns None).
            let pq = ThetaPoint2D::<F>::diff_add(&p, &q, &r);
            let qp = ThetaPoint2D::<F>::diff_add(&q, &p, &r);
            // Either both succeed (and produce equal results) or both
            // fail. A mismatch in Some/None status would violate symmetry.
            assert_eq!(
                pq.is_some(),
                qp.is_some(),
                "S114: diff_add symmetry — both branches must agree on Some/None at iteration {i}",
            );
            if let (Some(pq_v), Some(qp_v)) = (pq, qp) {
                assert_eq!(
                    pq_v, qp_v,
                    "S114: diff_add must be symmetric in p, q at iteration {i}",
                );
            }
        }
    }

    #[test]
    fn diff_add_commutes_at_lvl1() {
        check_diff_add_commutes::<Fp1Element>();
    }

    #[test]
    fn diff_add_commutes_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_diff_add_commutes::<Fp3Element>();
    }

    #[test]
    fn diff_add_commutes_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_diff_add_commutes::<Fp5Element>();
    }

    /// Generic helper: `diff_add` returns None when `H(p_minus_q)`
    /// has a zero component. Construct a `p_minus_q` whose Hadamard
    /// transform has a zero in position 0 by setting
    /// `p_minus_q = (1, 1, -1, -1)` — then `H(p_minus_q)[0]
    /// = 1 + 1 + (-1) + (-1) = 0`.
    fn check_diff_add_rejects_degenerate_difference<F: BaseField>() {
        let one = Fp2::<F>::one();
        let neg_one = one.negate();
        // (1, 1, -1, -1) — Hadamard yields (0, ?, ?, ?).
        let degenerate = ThetaPoint2D::<F>::new(one, one, neg_one, neg_one);
        // Verify our construction: H(degenerate)[0] == 0.
        let h = degenerate.hadamard();
        assert!(
            bool::from(h.x.is_zero()),
            "S114: setup error — H(degenerate)[0] is not zero",
        );
        let p = ThetaPoint2D::<F>::identity();
        let q = ThetaPoint2D::<F>::identity();
        let r = ThetaPoint2D::<F>::diff_add(&p, &q, &degenerate);
        assert_eq!(
            r, None,
            "S114: diff_add must return None when H(p_minus_q) has a zero component",
        );
    }

    #[test]
    fn diff_add_rejects_degenerate_difference_at_lvl1() {
        check_diff_add_rejects_degenerate_difference::<Fp1Element>();
    }

    #[test]
    fn diff_add_rejects_degenerate_difference_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_diff_add_rejects_degenerate_difference::<Fp3Element>();
    }

    #[test]
    fn diff_add_rejects_degenerate_difference_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_diff_add_rejects_degenerate_difference::<Fp5Element>();
    }

    /// Generic helper: for inputs where the Hadamard transform of
    /// `p_minus_q` has no zero component, `diff_add` returns Some
    /// (does not unexpectedly fail). 4 random triples per level.
    fn check_diff_add_succeeds_on_non_degenerate<F: BaseField>() {
        use crate::hash::hash_to_fp2;
        for i in 0u8..4 {
            let mk = |tag: &[u8]| {
                hash_to_fp2::<F>(tag, &[i], 16)
                    .into_option()
                    .unwrap_or_else(|| Fp2::<F>::new(F::one(), F::one()))
            };
            let p = ThetaPoint2D::<F>::new(
                mk(b"das-p-x"),
                mk(b"das-p-y"),
                mk(b"das-p-z"),
                mk(b"das-p-w"),
            );
            let q = ThetaPoint2D::<F>::new(
                mk(b"das-q-x"),
                mk(b"das-q-y"),
                mk(b"das-q-z"),
                mk(b"das-q-w"),
            );
            let r = ThetaPoint2D::<F>::new(
                mk(b"das-r-x"),
                mk(b"das-r-y"),
                mk(b"das-r-z"),
                mk(b"das-r-w"),
            );
            let h_r = r.hadamard();
            // Skip samples where H(r) has a zero component — those are
            // the documented-None cases, not a failure of the success path.
            if bool::from(h_r.x.is_zero())
                || bool::from(h_r.y.is_zero())
                || bool::from(h_r.z.is_zero())
                || bool::from(h_r.w.is_zero())
            {
                continue;
            }
            let result = ThetaPoint2D::<F>::diff_add(&p, &q, &r);
            assert!(
                result.is_some(),
                "S114: diff_add must return Some when H(p_minus_q) has no zero component, iter {i}",
            );
        }
    }

    #[test]
    fn diff_add_succeeds_on_non_degenerate_at_lvl1() {
        check_diff_add_succeeds_on_non_degenerate::<Fp1Element>();
    }

    #[test]
    fn diff_add_succeeds_on_non_degenerate_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_diff_add_succeeds_on_non_degenerate::<Fp3Element>();
    }

    #[test]
    fn diff_add_succeeds_on_non_degenerate_at_lvl5() {
        use crate::params::lvl5::Fp5Element;
        check_diff_add_succeeds_on_non_degenerate::<Fp5Element>();
    }
}
