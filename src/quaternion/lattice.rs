// SPDX-License-Identifier: MIT OR Apache-2.0
//! Lattice-reduction primitives for 4-D `Z`-lattices.
//!
//! The brute-force quaternion-witness search (`find_quaternion_in_ideal_with_norm`)
//! becomes infeasible at real-prime SQIsign scale. The fix is LLL basis
//! reduction; this module ships its foundational primitive — Babai-style
//! **size reduction** — and the integer-inner-product helpers LLL will
//! also consume.
//!
//! Size reduction repeatedly subtracts integer multiples of earlier basis
//! vectors from later ones to make each `b_j` (for `j > i`) approximately
//! orthogonal to `b_i`. It is *not* the full LLL — there's no Lovász swap
//! step here — but it bounds intermediate growth and produces a basis
//! whose vectors are no longer than the input's longest vector.

use crypto_bigint::Int;

use crate::quaternion::hnf::int_div_floor;

/// Integer inner product `⟨a, b⟩ = Σ aᵢ · bᵢ`.
pub fn dot4<const LIMBS: usize>(a: &[Int<LIMBS>; 4], b: &[Int<LIMBS>; 4]) -> Int<LIMBS> {
    a[0].wrapping_mul(&b[0])
        .wrapping_add(&a[1].wrapping_mul(&b[1]))
        .wrapping_add(&a[2].wrapping_mul(&b[2]))
        .wrapping_add(&a[3].wrapping_mul(&b[3]))
}

/// Squared Euclidean length `‖a‖² = ⟨a, a⟩`.
pub fn norm2<const LIMBS: usize>(a: &[Int<LIMBS>; 4]) -> Int<LIMBS> {
    dot4(a, a)
}

/// Round-to-nearest integer division: `⌊(2n + d) / (2d)⌋` for `d > 0`.
/// Returns 0 if `d == 0`.
fn round_div<const LIMBS: usize>(n: &Int<LIMBS>, d: &Int<LIMBS>) -> Int<LIMBS> {
    let zero = Int::<LIMBS>::from_i64(0);
    if *d == zero {
        return zero;
    }
    // Work with positive denominator; carry sign through numerator.
    let (d_abs, d_neg) = d.abs_sign();
    let (n_abs, n_neg) = n.abs_sign();
    let result_neg = bool::from(n_neg) ^ bool::from(d_neg);
    // q_floor = n_abs / d_abs; remainder = n_abs - q_floor * d_abs.
    // Round-to-nearest: q_round = q_floor + (1 if 2*remainder >= d_abs else 0).
    let d_int = *d_abs.as_int();
    let n_int = *n_abs.as_int();
    let q = int_div_floor(&n_int, &d_int);
    let q_d = q.wrapping_mul(&d_int);
    let remainder = n_int.wrapping_sub(&q_d);
    let two_rem = remainder.wrapping_add(&remainder);
    let one = Int::<LIMBS>::from_i64(1);
    let bumped = if two_rem >= d_int {
        q.wrapping_add(&one)
    } else {
        q
    };
    if result_neg {
        bumped.wrapping_neg()
    } else {
        bumped
    }
}

/// Gram matrix of a 4×4 integer lattice basis: `G[i][j] = ⟨bᵢ, bⱼ⟩`.
///
/// The Gram matrix is symmetric (`G[i][j] = G[j][i]`) and positive
/// semi-definite. Its determinant equals `det(B)²` where `B` is the
/// basis matrix — a load-bearing invariant LLL implementations exploit
/// for integer arithmetic.
///
/// Useful as the integer-arithmetic stand-in for the rational
/// Gram-Schmidt coefficients that classical LLL maintains.
#[allow(clippy::needless_range_loop)]
pub fn gram_matrix_4x4<const LIMBS: usize>(basis: &[[Int<LIMBS>; 4]; 4]) -> [[Int<LIMBS>; 4]; 4] {
    let zero = Int::<LIMBS>::from_i64(0);
    let mut g = [[zero; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            g[i][j] = dot4(&basis[i], &basis[j]);
        }
    }
    g
}

/// Exact integer division `n / d` for the integer-GSO recurrence.
///
/// Caller asserts `d` divides `n`; a debug assertion checks this. Returns
/// 0 when `d == 0` (matching `int_div_floor`); the GSO caller treats a
/// zero `d[i]` as a rank-deficient basis and bails before reaching this.
fn int_div_exact<const LIMBS: usize>(n: &Int<LIMBS>, d: &Int<LIMBS>) -> Int<LIMBS> {
    let zero = Int::<LIMBS>::from_i64(0);
    if *d == zero {
        return zero;
    }
    let q = int_div_floor(n, d);
    debug_assert_eq!(
        q.wrapping_mul(d),
        *n,
        "int_div_exact called on non-exact division"
    );
    q
}

/// Cohen-style integer Gram–Schmidt orthogonalization on a 4×4 basis.
///
/// Returns `(d, lam)` where:
/// - `d[i]` = determinant of the leading `i × i` block of the Gram matrix
///   (with `d[0] = 1`). Equivalently, `d[i+1] / d[i] = ‖b*_i‖²`, so the
///   `d[]` array compactly encodes the squared Gram–Schmidt norms.
/// - `lam[i][j]` for `i > j` is the integer-scaled GS coefficient
///   `d[j+1] · µ_{i,j}` where `µ_{i,j} = ⟨b_i, b*_j⟩ / ‖b*_j‖²`.
///   On the diagonal `lam[i][i] = d[i+1]` (preserved from the recurrence).
///
/// Recurrence (Cohen Prop 2.6.7, 0-indexed form): for each `(j, i)` with
/// `i ≥ j`, starting from `u = G[i][j]`,
///
/// ```text
///     u ← (d[k+1] · u − lam[i][k] · lam[j][k]) / d[k]     for k = 0..j-1
///     lam[i][j] ← u
/// ```
///
/// Each division is exact when `basis` is a rank-4 integer lattice (Cohen
/// Prop 2.6.7). Returns `None` if some `d[i+1] == 0` (rank-deficient input);
/// the LLL caller treats this as a degenerate basis and bails.
#[allow(clippy::needless_range_loop)]
pub fn integer_gso_4x4<const LIMBS: usize>(
    basis: &[[Int<LIMBS>; 4]; 4],
) -> Option<([Int<LIMBS>; 5], [[Int<LIMBS>; 4]; 4])> {
    let gram = gram_matrix_4x4(basis);
    let zero = Int::<LIMBS>::from_i64(0);
    let one = Int::<LIMBS>::from_i64(1);
    let mut d: [Int<LIMBS>; 5] = [zero; 5];
    d[0] = one;
    let mut lam: [[Int<LIMBS>; 4]; 4] = [[zero; 4]; 4];

    for j in 0..4 {
        for i in j..4 {
            let mut u = gram[i][j];
            for k in 0..j {
                let scaled = d[k + 1].wrapping_mul(&u);
                let cross = lam[i][k].wrapping_mul(&lam[j][k]);
                let num = scaled.wrapping_sub(&cross);
                u = int_div_exact(&num, &d[k]);
            }
            lam[i][j] = u;
        }
        d[j + 1] = lam[j][j];
        if d[j + 1] == zero {
            return None;
        }
    }
    Some((d, lam))
}

/// LLL basis reduction on a 4×4 integer basis with `δ = 3/4`.
///
/// Composes [`size_reduce_4x4`] with Lovász-condition swaps until the
/// basis is LLL-reduced. The Lovász test in pure-integer Cohen form
/// (Cohen Algorithm 2.6.3): swap basis vectors `(k-1, k)` when
///
/// ```text
///     4 · d[k-1] · d[k+1]  <  3 · d[k]²  −  4 · lam[k][k-1]²
/// ```
///
/// Implementation note: the Gram–Schmidt integers are recomputed from
/// scratch after each modification (size-reduce or swap). For 4×4 this
/// is O(16) inner products per iteration — negligible — and eliminates
/// any chance of an incremental-update bug. Outer-loop iteration is
/// capped at 64 as a safety net; theoretical bound for `Int<8>` test
/// inputs is far lower.
///
/// Returns the input unchanged when the basis is rank-deficient.
#[allow(clippy::needless_range_loop)]
pub fn lll_4x4<const LIMBS: usize>(input: &[[Int<LIMBS>; 4]; 4]) -> [[Int<LIMBS>; 4]; 4] {
    let mut basis = size_reduce_4x4(input);
    let three = Int::<LIMBS>::from_i64(3);
    let four = Int::<LIMBS>::from_i64(4);
    let max_iters = 64;

    for _iter in 0..max_iters {
        let Some((d, lam)) = integer_gso_4x4(&basis) else {
            return basis;
        };
        let mut swap_k: Option<usize> = None;
        for k in 1..4 {
            let lhs = four.wrapping_mul(&d[k - 1].wrapping_mul(&d[k + 1]));
            let dk_sq = d[k].wrapping_mul(&d[k]);
            let lam_sq = lam[k][k - 1].wrapping_mul(&lam[k][k - 1]);
            let rhs = three
                .wrapping_mul(&dk_sq)
                .wrapping_sub(&four.wrapping_mul(&lam_sq));
            if lhs < rhs {
                swap_k = Some(k);
                break;
            }
        }
        match swap_k {
            None => return basis,
            Some(k) => {
                basis.swap(k - 1, k);
                basis = size_reduce_4x4(&basis);
            }
        }
    }
    basis
}

/// Widen a signed `Int<NARROW>` to `Int<WIDE>` by sign-extension.
/// Decomposes via `abs_sign`, resizes the unsigned magnitude, then
/// re-applies the original sign. Used by [`lll_4x4_in_metric_wide`] and
/// by the KLPT wide search path (`find_prime_norm_quaternion_in_ideal_wide`)
/// to route narrow-Int inputs through wide-precision arithmetic when
/// narrow would overflow.
pub(crate) fn widen_int_lattice<const NARROW: usize, const WIDE: usize>(
    x: &Int<NARROW>,
) -> Int<WIDE> {
    let (uint_n, neg) = x.abs_sign();
    let uint_w: crypto_bigint::Uint<WIDE> = uint_n.resize::<WIDE>();
    let int_w: Int<WIDE> = *uint_w.as_int();
    if bool::from(neg) {
        int_w.wrapping_neg()
    } else {
        int_w
    }
}

/// Narrow a signed `Int<WIDE>` to `Int<NARROW>` by truncation. Caller
/// asserts the magnitude fits in `NARROW · 64` bits; for valid LLL
/// outputs on bounded inputs this is true because the reduced basis
/// entries are small (size-reduction keeps them bounded). Also used by
/// the KLPT wide search to narrow the found α back to the caller's
/// `Int<NARROW>` width.
///
/// Decomposes via `abs_sign`, resizes the unsigned magnitude (truncating
/// high limbs that should already be zero), re-applies the original sign.
pub(crate) fn narrow_int_lattice<const WIDE: usize, const NARROW: usize>(
    x: &Int<WIDE>,
) -> Int<NARROW> {
    let (uint_w, neg) = x.abs_sign();
    let uint_n: crypto_bigint::Uint<NARROW> = uint_w.resize::<NARROW>();
    let int_n: Int<NARROW> = *uint_n.as_int();
    if bool::from(neg) {
        int_n.wrapping_neg()
    } else {
        int_n
    }
}

/// Wide-Int variant of [`lll_4x4_in_metric`].
///
/// Widens the narrow `Int<NARROW>` basis and metric to `Int<WIDE>` (via
/// sign-extension), runs the existing generic [`lll_4x4_in_metric`] at
/// `WIDE` precision (giving the algorithm the magnitude headroom its
/// math demands), then narrows the reduced basis back to `Int<NARROW>`.
///
/// At `NARROW = WIDE` this reduces to [`lll_4x4_in_metric`] with extra
/// widen/narrow round-trips — useful as a parity check. For
/// `WIDE > NARROW` (typically `WIDE = 2·NARROW`), this avoids the
/// wrapping-corruption that S55 confirmed at L1 large-γ scale where the
/// Lovász intermediate `4·d[k-1]·d[k+1]` can reach `2^514 > 2^511 = Int<8>::MAX`.
///
/// **Caller invariant**: the *reduced* basis entries must fit in
/// `NARROW · 64` bits. For LLL on bounded-input lattices this holds
/// because size-reduction keeps entries small; the narrow step in this
/// function will silently truncate if the invariant is violated. If
/// downstream code samples in the reduced basis and the samples
/// themselves overflow narrow, those overflows are a separate concern
/// for the *search-path* wide variant (`find_prime_norm_quaternion_in_ideal_wide`,
/// future session).
#[allow(clippy::needless_range_loop)]
pub fn lll_4x4_in_metric_wide<const NARROW: usize, const WIDE: usize>(
    basis: &[[Int<NARROW>; 4]; 4],
    metric: &[[Int<NARROW>; 4]; 4],
) -> [[Int<NARROW>; 4]; 4] {
    let zero_w = Int::<WIDE>::from_i64(0);
    let mut basis_w = [[zero_w; 4]; 4];
    let mut metric_w = [[zero_w; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            basis_w[r][c] = widen_int_lattice::<NARROW, WIDE>(&basis[r][c]);
            metric_w[r][c] = widen_int_lattice::<NARROW, WIDE>(&metric[r][c]);
        }
    }

    let reduced_w = lll_4x4_in_metric::<WIDE>(&basis_w, &metric_w);

    let zero_n = Int::<NARROW>::from_i64(0);
    let mut reduced = [[zero_n; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            reduced[r][c] = narrow_int_lattice::<WIDE, NARROW>(&reduced_w[r][c]);
        }
    }
    reduced
}

/// Pull back a 4×4 integer Gram matrix through a 4×4 basis: returns
/// `B · M · Bᵀ`. Used to compute the induced metric on a lattice when
/// `B` is the lattice's basis matrix and `M` is the ambient quadratic
/// form's Gram. Two integer 4×4 matmuls (64 wrapping multiplications),
/// pure integer arithmetic.
#[allow(clippy::needless_range_loop)]
pub fn pull_back_gram<const LIMBS: usize>(
    basis: &[[Int<LIMBS>; 4]; 4],
    metric: &[[Int<LIMBS>; 4]; 4],
) -> [[Int<LIMBS>; 4]; 4] {
    let zero = Int::<LIMBS>::from_i64(0);
    let mut bm = [[zero; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = zero;
            for k in 0..4 {
                s = s.wrapping_add(&basis[i][k].wrapping_mul(&metric[k][j]));
            }
            bm[i][j] = s;
        }
    }
    let mut out = [[zero; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = zero;
            for k in 0..4 {
                // Bᵀ[k][j] = B[j][k]
                s = s.wrapping_add(&bm[i][k].wrapping_mul(&basis[j][k]));
            }
            out[i][j] = s;
        }
    }
    out
}

/// Cohen-style integer Gram–Schmidt orthogonalization on a 4×4 *Gram
/// matrix* directly (skipping the `gram_matrix_4x4` step that
/// [`integer_gso_4x4`] performs internally). Identical semantics
/// otherwise: returns `(d, lam)` such that `d[i+1] / d[i] = ‖b*_i‖²`
/// (under the lattice's metric, whatever that is — the function is
/// metric-agnostic) and `lam[i][j] = d[j+1] · µ_{i,j}` for `i > j`.
///
/// Returns `None` for rank-deficient input. Used by
/// [`lll_4x4_in_metric`] which wants the Gram pre-pulled-back through
/// the lattice's metric, not the basis's Euclidean Gram.
#[allow(clippy::needless_range_loop)]
pub fn integer_gso_4x4_from_gram<const LIMBS: usize>(
    gram: &[[Int<LIMBS>; 4]; 4],
) -> Option<([Int<LIMBS>; 5], [[Int<LIMBS>; 4]; 4])> {
    let zero = Int::<LIMBS>::from_i64(0);
    let one = Int::<LIMBS>::from_i64(1);
    let mut d: [Int<LIMBS>; 5] = [zero; 5];
    d[0] = one;
    let mut lam: [[Int<LIMBS>; 4]; 4] = [[zero; 4]; 4];

    for j in 0..4 {
        for i in j..4 {
            let mut u = gram[i][j];
            for k in 0..j {
                let scaled = d[k + 1].wrapping_mul(&u);
                let cross = lam[i][k].wrapping_mul(&lam[j][k]);
                let num = scaled.wrapping_sub(&cross);
                u = int_div_exact(&num, &d[k]);
            }
            lam[i][j] = u;
        }
        d[j + 1] = lam[j][j];
        if d[j + 1] == zero {
            return None;
        }
    }
    Some((d, lam))
}

/// LLL basis reduction on a 4×4 integer basis under a non-trivial
/// positive-definite quadratic form `metric`, with `δ = 3/4`.
///
/// Generalises [`lll_4x4`] (which uses the Euclidean inner product) to
/// arbitrary metrics. The KLPT wiring needs this for the ideal-norm-form
/// search: the metric is the reduced-norm Gram on `O_0` (factor of 4 to
/// stay integer), pulled back through the ideal basis. Sampling in the
/// reduced-basis coordinates then biases candidates toward small
/// `N(α)`, not small Euclidean length.
///
/// Algorithm (Cohen Algorithm 2.6.3 with the Gram input swapped for
/// `pull_back_gram(basis, metric)`):
///
/// 1. Compute `gram_at_b = pull_back_gram(b, metric)`.
/// 2. Run integer GSO on `gram_at_b` directly via
///    [`integer_gso_4x4_from_gram`] to get `(d, lam)`.
/// 3. Size-reduce: for each pair `(i, j)` with `i < j`, compute
///    `r = round(lam[j][i] / d[i+1])` and update `b[j] ← b[j] − r · b[i]`.
/// 4. After size-reduction, re-pull-back the Gram and re-run GSO.
/// 5. Find the first Lovász violation
///    `4 · d[k−1] · d[k+1] < 3 · d[k]² − 4 · lam[k][k−1]²` and swap
///    `b[k−1] ↔ b[k]`. Repeat from step 1.
///
/// Outer loop capped at 64 iterations (theoretical bound for `Int<8>`
/// test inputs is much lower). Returns the input unchanged if the
/// metric or basis is rank-deficient (i.e. the GSO recurrence hits a
/// zero `d[i+1]`).
#[allow(clippy::needless_range_loop)]
pub fn lll_4x4_in_metric<const LIMBS: usize>(
    input: &[[Int<LIMBS>; 4]; 4],
    metric: &[[Int<LIMBS>; 4]; 4],
) -> [[Int<LIMBS>; 4]; 4] {
    let mut b = *input;
    let three = Int::<LIMBS>::from_i64(3);
    let four = Int::<LIMBS>::from_i64(4);
    let zero = Int::<LIMBS>::from_i64(0);
    let max_iters = 64;

    for _iter in 0..max_iters {
        // Pull back the metric through the current basis.
        let gram = pull_back_gram(&b, metric);
        let Some((d, lam)) = integer_gso_4x4_from_gram(&gram) else {
            return b;
        };

        // Size-reduce: for each (i, j) with i < j, subtract r·b[i] from b[j].
        for j in 1..4 {
            for i in (0..j).rev() {
                let r = round_div(&lam[j][i], &d[i + 1]);
                if r == zero {
                    continue;
                }
                for k in 0..4 {
                    let delta = r.wrapping_mul(&b[i][k]);
                    b[j][k] = b[j][k].wrapping_sub(&delta);
                }
            }
        }

        // Re-pull-back and re-GSO after size-reduction.
        let gram = pull_back_gram(&b, metric);
        let Some((d, lam)) = integer_gso_4x4_from_gram(&gram) else {
            return b;
        };

        // Lovász test on the size-reduced basis.
        let mut swap_k: Option<usize> = None;
        for k in 1..4 {
            let lhs = four.wrapping_mul(&d[k - 1].wrapping_mul(&d[k + 1]));
            let dk_sq = d[k].wrapping_mul(&d[k]);
            let lam_sq = lam[k][k - 1].wrapping_mul(&lam[k][k - 1]);
            let rhs = three
                .wrapping_mul(&dk_sq)
                .wrapping_sub(&four.wrapping_mul(&lam_sq));
            if lhs < rhs {
                swap_k = Some(k);
                break;
            }
        }
        match swap_k {
            None => return b,
            Some(k) => {
                b.swap(k - 1, k);
            }
        }
    }
    b
}

/// Wide-Int variant of [`qf_eval_4x4`]. Widens narrow inputs to
/// `Int<WIDE>` and evaluates `cᵀ · G · c` at wide precision, returning
/// `Int<WIDE>`.
///
/// At `NARROW = WIDE` this matches [`qf_eval_4x4`] with extra widen
/// round-trips. For `WIDE > NARROW` (typically `WIDE = 2·NARROW`),
/// the wide arithmetic avoids overflow when intermediates can reach
/// `~NARROW · 64 · 3` bits — the regime where S55 confirmed L1 large-γ
/// `reduced_norm_o0_basis` was wrapping. Combine with
/// [`lll_4x4_in_metric_wide`] and a future
/// `pull_back_gram_wide` to compose the wide search path that flips
/// the S55 `#[should_panic]` L1 large-γ test.
#[allow(clippy::needless_range_loop)]
pub fn qf_eval_4x4_wide<const NARROW: usize, const WIDE: usize>(
    c: &[Int<NARROW>; 4],
    gram: &[[Int<NARROW>; 4]; 4],
) -> Int<WIDE> {
    let zero_w = Int::<WIDE>::from_i64(0);
    let mut c_wide = [zero_w; 4];
    let mut g_wide = [[zero_w; 4]; 4];
    for i in 0..4 {
        c_wide[i] = widen_int_lattice::<NARROW, WIDE>(&c[i]);
        for j in 0..4 {
            g_wide[i][j] = widen_int_lattice::<NARROW, WIDE>(&gram[i][j]);
        }
    }
    qf_eval_4x4::<WIDE>(&c_wide, &g_wide)
}

/// Evaluate the quadratic form `Q(c) = cᵀ · G · c` on a 4-D integer
/// coordinate vector `c` against a 4×4 integer Gram matrix `G`. For
/// `G = diag(1, 1, p, p)` this returns the reduced quaternion norm
/// `a² + b² + p·(c² + d²)` directly when `c = (a, b, c, d)`. For an
/// ideal's pulled-back Gram `G_I = Bᵀ · diag(1, 1, p, p) · B` this
/// returns `denom² · N(α)` where `α ∈ I` is the quaternion at integer
/// coordinates `c` in the ideal basis — the inner-loop primitive of
/// the KLPT prime-norm-reduced-equivalent search (`quat_qf_eval` in
/// the SQIsign C reference).
///
/// 16 wrapping multiplications + 16 wrapping additions; pure integer
/// arithmetic. Symmetric Gram matrices give identical results regardless
/// of whether `G[i][j]` or `G[j][i]` is consulted, but the function
/// makes no symmetry assumption.
#[allow(clippy::needless_range_loop)]
pub fn qf_eval_4x4<const LIMBS: usize>(
    c: &[Int<LIMBS>; 4],
    gram: &[[Int<LIMBS>; 4]; 4],
) -> Int<LIMBS> {
    let mut result = Int::<LIMBS>::from_i64(0);
    for i in 0..4 {
        for j in 0..4 {
            let term = c[i].wrapping_mul(&gram[i][j]).wrapping_mul(&c[j]);
            result = result.wrapping_add(&term);
        }
    }
    result
}

/// Enumerate 2-D lattice points `(c, d) ∈ Z²` under a diagonal-Gram
/// quadratic threshold `α·c² + β·d² ≤ T`, returning the first point for
/// which `accept(&[c, d])` is true. Returns `None` if no point passes.
///
/// Order of enumeration: outer loop `c ∈ {0, 1, 2, …}` until `α·c² > T`;
/// inner loop `d ∈ {0, 1, 2, …}` until `α·c² + β·d² > T`. For each
/// `(|c|, |d|)` pair, sign variants are tried in order
/// `(+c, +d), (+c, −d), (−c, +d), (−c, −d)` with degenerate-sign
/// duplicates suppressed when `c == 0` or `d == 0`. This is **not**
/// strictly increasing-`Q` order; callers needing the smallest-`Q`
/// witness should sweep with increasing `T` ceilings.
///
/// No allocation: bignum sqrt is avoided by iterating `c, d` with
/// increment-and-compare against the wrapping product `α·c² + β·d²`.
/// Suitable for any `Int<LIMBS>` arithmetic; correctness depends on
/// `α, β, T ≥ 0` (negative threshold returns `None` immediately).
///
/// Foundation for `find_norm_witness`'s `(c, d)` sub-search where
/// `α = β = p`, threshold `= T_target_norm`, and `accept((c, d))` runs
/// Cornacchia on `T − p·(c² + d²)` and reports whether `(a, b)` is
/// solvable. Useful at prototype scale; at real-prime KLPT scale
/// (`T/p ~ 2^249`) the enumeration is infeasible regardless of how it
/// is structured — that's a complexity property of the search, not the
/// primitive.
#[allow(clippy::needless_range_loop)]
pub fn find_lattice_point_2x2_under_quad_threshold<const LIMBS: usize, F>(
    threshold: &Int<LIMBS>,
    alpha: &Int<LIMBS>,
    beta: &Int<LIMBS>,
    mut accept: F,
) -> Option<[Int<LIMBS>; 2]>
where
    F: FnMut(&[Int<LIMBS>; 2]) -> bool,
{
    let zero = Int::<LIMBS>::from_i64(0);
    let one = Int::<LIMBS>::from_i64(1);
    if *threshold < zero {
        return None;
    }

    let mut c = zero;
    loop {
        let c2 = c.wrapping_mul(&c);
        let ac2 = alpha.wrapping_mul(&c2);
        if ac2 > *threshold {
            break;
        }

        let mut d = zero;
        loop {
            let d2 = d.wrapping_mul(&d);
            let bd2 = beta.wrapping_mul(&d2);
            let q = ac2.wrapping_add(&bd2);
            if q > *threshold {
                break;
            }

            let point_pp = [c, d];
            if accept(&point_pp) {
                return Some(point_pp);
            }
            if d != zero {
                let point_pn = [c, d.wrapping_neg()];
                if accept(&point_pn) {
                    return Some(point_pn);
                }
            }
            if c != zero {
                let point_np = [c.wrapping_neg(), d];
                if accept(&point_np) {
                    return Some(point_np);
                }
                if d != zero {
                    let point_nn = [c.wrapping_neg(), d.wrapping_neg()];
                    if accept(&point_nn) {
                        return Some(point_nn);
                    }
                }
            }

            d = d.wrapping_add(&one);
        }

        c = c.wrapping_add(&one);
    }

    None
}

/// Generalized inner product `⟨u, v⟩_G = uᵀ G v` on 2-D vectors using a
/// positive-definite Gram matrix `G`. For `G = I` this matches [`dot2`];
/// for `G = diag(α, β)` it scales each axis independently
/// (`α · u₀·v₀ + β · u₁·v₁`); for off-diagonal `G` it captures the
/// cross-term contribution `(G[0][1] + G[1][0]) · u₀·v₁`.
#[allow(clippy::needless_range_loop)]
pub fn gram_dot2<const LIMBS: usize>(
    u: &[Int<LIMBS>; 2],
    v: &[Int<LIMBS>; 2],
    g: &[[Int<LIMBS>; 2]; 2],
) -> Int<LIMBS> {
    let mut result = Int::<LIMBS>::from_i64(0);
    for i in 0..2 {
        for j in 0..2 {
            let term = u[i].wrapping_mul(&g[i][j]).wrapping_mul(&v[j]);
            result = result.wrapping_add(&term);
        }
    }
    result
}

/// Quadratic-form norm-squared `Q(v) = vᵀ G v` for 2-D `v`.
pub fn gram_norm2_2<const LIMBS: usize>(
    v: &[Int<LIMBS>; 2],
    g: &[[Int<LIMBS>; 2]; 2],
) -> Int<LIMBS> {
    gram_dot2(v, v, g)
}

/// Lagrange reduction under a 2×2 positive-definite Gram matrix `G`.
///
/// Substitutes [`gram_dot2`] / [`gram_norm2_2`] for the Euclidean inner
/// product inside the Session-39 Lagrange loop. Result is the
/// `G`-reduced basis: `Q(b_0) ≤ Q(b_1)` and `2·|⟨b_0, b_1⟩_G| ≤ Q(b_0)`.
/// For `G = I` the output equals [`lll_2x2`]'s.
#[allow(clippy::needless_range_loop)]
pub fn lll_2x2_with_gram<const LIMBS: usize>(
    input: &[[Int<LIMBS>; 2]; 2],
    g: &[[Int<LIMBS>; 2]; 2],
) -> [[Int<LIMBS>; 2]; 2] {
    let mut b = *input;
    let zero = Int::<LIMBS>::from_i64(0);
    let max_iters = 64;

    for _iter in 0..max_iters {
        if gram_norm2_2(&b[1], g) < gram_norm2_2(&b[0], g) {
            b.swap(0, 1);
        }
        let denom = gram_norm2_2(&b[0], g);
        if denom == zero {
            break;
        }
        let num = gram_dot2(&b[1], &b[0], g);
        let r = round_div(&num, &denom);
        if r == zero {
            break;
        }
        for c in 0..2 {
            let delta = r.wrapping_mul(&b[0][c]);
            b[1][c] = b[1][c].wrapping_sub(&delta);
        }
    }
    b
}

/// Babai-style round-off CVP on a 2×2 lattice under a positive-definite
/// Gram matrix `G`. Minimizes `Q(v − t) = (v − t)ᵀ G (v − t)` over lattice
/// points `v ∈ Z·b_0 + Z·b_1`.
///
/// Pipeline: `lll_2x2_with_gram` to reduce the basis under `G`, then
/// round-off project the target onto each reduced basis vector using
/// `gram_dot2` / `gram_norm2_2`. For `G = I` the output equals
/// [`babai_cvp_2x2`]'s. For non-trivial `G` and non-trivial basis, the
/// QF metric biases the search toward lattice vectors that minimize
/// the weighted residual.
///
/// This is a general primitive for KLPT-style 2-D lattice searches
/// under norm-form metrics (e.g. the `(c, d)` sub-lattice with
/// `Q(c, d) = p·(c² + d²)`, when the basis is non-trivial). For
/// trivial basis `Z²` the QF still rounds per-coordinate — that case
/// needs short-vector *enumeration* (Fincke-Pohst), not CVP.
#[allow(clippy::needless_range_loop)]
pub fn babai_cvp_2x2_with_gram<const LIMBS: usize>(
    basis: &[[Int<LIMBS>; 2]; 2],
    target: &[Int<LIMBS>; 2],
    g: &[[Int<LIMBS>; 2]; 2],
) -> [Int<LIMBS>; 2] {
    let reduced = lll_2x2_with_gram(basis, g);
    let zero = Int::<LIMBS>::from_i64(0);
    let mut t_curr = *target;

    for i in (0..2).rev() {
        let denom = gram_norm2_2(&reduced[i], g);
        if denom == zero {
            continue;
        }
        let num = gram_dot2(&t_curr, &reduced[i], g);
        let c = round_div(&num, &denom);
        if c == zero {
            continue;
        }
        for col in 0..2 {
            let delta = c.wrapping_mul(&reduced[i][col]);
            t_curr[col] = t_curr[col].wrapping_sub(&delta);
        }
    }

    let mut nearest = [zero; 2];
    for i in 0..2 {
        nearest[i] = target[i].wrapping_sub(&t_curr[i]);
    }
    nearest
}

/// Integer inner product on 2-D vectors.
pub fn dot2<const LIMBS: usize>(a: &[Int<LIMBS>; 2], b: &[Int<LIMBS>; 2]) -> Int<LIMBS> {
    a[0].wrapping_mul(&b[0])
        .wrapping_add(&a[1].wrapping_mul(&b[1]))
}

/// Squared Euclidean length of a 2-D vector.
pub fn norm2_2<const LIMBS: usize>(a: &[Int<LIMBS>; 2]) -> Int<LIMBS> {
    dot2(a, a)
}

/// Lagrange / Gauss reduction on a 2×2 integer basis.
///
/// At `n = 2` the full LLL machinery collapses to a simple loop: keep
/// `b_0` the shorter vector, size-reduce `b_1` against `b_0`, repeat
/// until the size-reduction step is a no-op. The result is the
/// **Lagrange-reduced basis**: `‖b_0‖² ≤ ‖b_1‖²` and `2·|⟨b_0, b_1⟩| ≤
/// ‖b_0‖²`. This is the optimal reduction at `n = 2` — `b_0` is a
/// shortest non-zero lattice vector.
///
/// Cheaper than [`lll_4x4`] and the right tool when callers know the
/// problem is 2-D (e.g. the `(c, d)` sub-search in the norm-form quaternion
/// witness `a² + b² + p·(c² + d²) = T`). Outer-loop iteration is capped
/// at 64 as a safety net; the theoretical bound at `n = 2` with bounded
/// entries is much lower.
#[allow(clippy::needless_range_loop)]
pub fn lll_2x2<const LIMBS: usize>(input: &[[Int<LIMBS>; 2]; 2]) -> [[Int<LIMBS>; 2]; 2] {
    let mut b = *input;
    let zero = Int::<LIMBS>::from_i64(0);
    let max_iters = 64;

    for _iter in 0..max_iters {
        if norm2_2(&b[1]) < norm2_2(&b[0]) {
            b.swap(0, 1);
        }
        let denom = norm2_2(&b[0]);
        if denom == zero {
            break;
        }
        let num = dot2(&b[1], &b[0]);
        let r = round_div(&num, &denom);
        if r == zero {
            break;
        }
        for c in 0..2 {
            let delta = r.wrapping_mul(&b[0][c]);
            b[1][c] = b[1][c].wrapping_sub(&delta);
        }
    }
    b
}

/// Babai-style round-off closest-vector approximation on a 2×2 integer
/// lattice. Mirror of [`babai_cvp_4x4`] at dimension 2.
///
/// Pipeline: Gauss-reduce via [`lll_2x2`], then for `i ∈ {1, 0}` subtract
/// `round(⟨t_curr, b_i⟩ / ‖b_i‖²) · b_i` from the running target. Returns
/// `target − t_curr` (a lattice point). At `n = 2` with Lagrange
/// reduction the approximation factor is `√2`, so the result is within
/// a `√2` factor of the true closest vector.
///
/// This is the primitive the norm-form witness search will use: given a
/// `(c, d)` target derived from `T / p` and a 2-D lattice encoding the
/// quadratic-form constraint, returns a `(c, d)` candidate. Wiring lands
/// in a later session — see ISC-41.13.
#[allow(clippy::needless_range_loop)]
pub fn babai_cvp_2x2<const LIMBS: usize>(
    basis: &[[Int<LIMBS>; 2]; 2],
    target: &[Int<LIMBS>; 2],
) -> [Int<LIMBS>; 2] {
    let reduced = lll_2x2(basis);
    let zero = Int::<LIMBS>::from_i64(0);
    let mut t_curr = *target;

    for i in (0..2).rev() {
        let denom = norm2_2(&reduced[i]);
        if denom == zero {
            continue;
        }
        let num = dot2(&t_curr, &reduced[i]);
        let c = round_div(&num, &denom);
        if c == zero {
            continue;
        }
        for col in 0..2 {
            let delta = c.wrapping_mul(&reduced[i][col]);
            t_curr[col] = t_curr[col].wrapping_sub(&delta);
        }
    }

    let mut nearest = [zero; 2];
    for i in 0..2 {
        nearest[i] = target[i].wrapping_sub(&t_curr[i]);
    }
    nearest
}

/// Babai-style round-off closest-vector approximation on a 4×4 integer
/// lattice.
///
/// Given a lattice basis `B = (b_0, b_1, b_2, b_3)` and a target vector `t`,
/// returns a lattice vector `v ∈ Z·b_0 + … + Z·b_3` that approximates the
/// true closest vector to `t`. Pipeline:
///
/// 1. LLL-reduce the basis via [`lll_4x4`] so the vectors are short and
///    near-orthogonal.
/// 2. Initialise `t_curr = t`.
/// 3. For `i ∈ {3, 2, 1, 0}`: compute `c_i = round(⟨t_curr, b_i⟩ / ‖b_i‖²)`
///    and subtract `c_i · b_i` from `t_curr`.
/// 4. Return `t − t_curr`, which is a `Z`-linear combination of the basis.
///
/// This is the *round-off* variant of Babai's CVP — simpler than nearest-
/// plane (which projects onto the Gram–Schmidt vectors `b*_i` instead of
/// the LLL-reduced `b_i`). For LLL-reduced bases at `n = 4`, the
/// approximation factor is the same `2^(n/2)`; the round-off path avoids
/// the integer-GSO bookkeeping nearest-plane would demand. Upgrade to
/// nearest-plane only if a downstream caller (e.g. `find_norm_witness`)
/// shows the round-off approximation is too loose.
///
/// On rank-deficient input (zero basis vectors), the function silently
/// skips degenerate axes — the LLL pre-step's `Option`-returning
/// [`integer_gso_4x4`] caller pattern is not used here because round-off
/// only needs `‖b_i‖² ≠ 0` per individual basis vector, not full rank.
#[allow(clippy::needless_range_loop)]
pub fn babai_cvp_4x4<const LIMBS: usize>(
    basis: &[[Int<LIMBS>; 4]; 4],
    target: &[Int<LIMBS>; 4],
) -> [Int<LIMBS>; 4] {
    let reduced = lll_4x4(basis);
    let zero = Int::<LIMBS>::from_i64(0);
    let mut t_curr = *target;

    for i in (0..4).rev() {
        let denom = norm2(&reduced[i]);
        if denom == zero {
            continue;
        }
        let num = dot4(&t_curr, &reduced[i]);
        let c = round_div(&num, &denom);
        if c == zero {
            continue;
        }
        for col in 0..4 {
            let delta = c.wrapping_mul(&reduced[i][col]);
            t_curr[col] = t_curr[col].wrapping_sub(&delta);
        }
    }

    let mut nearest = [zero; 4];
    for i in 0..4 {
        nearest[i] = target[i].wrapping_sub(&t_curr[i]);
    }
    nearest
}

/// Babai-style size reduction on a 4×4 integer basis matrix.
///
/// For each pair `(i, j)` with `i < j`, computes `r = round(⟨bⱼ, bᵢ⟩ / ⟨bᵢ, bᵢ⟩)`
/// and replaces `bⱼ ← bⱼ − r·bᵢ`. This makes each later vector closer to
/// orthogonal to the earlier ones, reducing `‖bⱼ‖` without changing the
/// lattice spanned by the basis (the update is a unimodular row operation).
///
/// **Not** the full LLL — there's no Lovász swap. The result is a basis
/// whose vectors are typically much shorter than the input's, suitable as a
/// pre-step for LLL or as a stand-alone reducer for already-near-orthogonal
/// inputs.
#[allow(clippy::needless_range_loop)]
pub fn size_reduce_4x4<const LIMBS: usize>(input: &[[Int<LIMBS>; 4]; 4]) -> [[Int<LIMBS>; 4]; 4] {
    let mut basis = *input;
    for i in 0..4 {
        let denom = norm2(&basis[i]);
        let zero = Int::<LIMBS>::from_i64(0);
        if denom == zero {
            continue;
        }
        for j in (i + 1)..4 {
            let num = dot4(&basis[j], &basis[i]);
            let r = round_div(&num, &denom);
            if r == zero {
                continue;
            }
            for c in 0..4 {
                let delta = r.wrapping_mul(&basis[i][c]);
                basis[j][c] = basis[j][c].wrapping_sub(&delta);
            }
        }
    }
    basis
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::quaternion::hnf::hnf_4x4;

    fn n(v: i64) -> Int<8> {
        Int::<8>::from_i64(v)
    }

    #[test]
    fn dot_orthogonal_basis_is_zero() {
        let e1 = [n(1), n(0), n(0), n(0)];
        let e2 = [n(0), n(1), n(0), n(0)];
        assert_eq!(dot4(&e1, &e2), n(0));
    }

    #[test]
    fn norm2_of_unit_is_one() {
        let e = [n(1), n(0), n(0), n(0)];
        assert_eq!(norm2(&e), n(1));
    }

    #[test]
    fn norm2_3_4_0_0_is_25() {
        let v = [n(3), n(4), n(0), n(0)];
        assert_eq!(norm2(&v), n(25));
    }

    #[test]
    fn round_div_basic() {
        assert_eq!(round_div(&n(7), &n(3)), n(2)); // 7/3 ≈ 2.33 → 2
        assert_eq!(round_div(&n(8), &n(3)), n(3)); // 8/3 ≈ 2.67 → 3
        assert_eq!(round_div(&n(9), &n(3)), n(3)); // exact
        assert_eq!(round_div(&n(-7), &n(3)), n(-2));
        assert_eq!(round_div(&n(7), &n(-3)), n(-2));
        assert_eq!(round_div(&n(0), &n(5)), n(0));
        assert_eq!(round_div(&n(5), &n(0)), n(0));
    }

    #[test]
    fn round_div_round_to_even_half_case() {
        // 1.5 → 2 (round half up — my impl uses ≥ so rounds up).
        assert_eq!(round_div(&n(3), &n(2)), n(2));
    }

    #[test]
    fn size_reduce_identity_is_unchanged() {
        let id: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let reduced = size_reduce_4x4(&id);
        assert_eq!(reduced, id);
    }

    #[test]
    fn size_reduce_shortens_skew_vector() {
        // Initial basis includes a "skewed" later vector that should be
        // reducible against the first.
        let m: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(7), n(1), n(0), n(0)], // ⟨b₁, b₀⟩ = 7; r = round(7/1) = 7; b₁ → (0, 1, 0, 0).
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let reduced = size_reduce_4x4(&m);
        assert_eq!(reduced[1], [n(0), n(1), n(0), n(0)]);
    }

    #[test]
    fn size_reduce_preserves_lattice() {
        // Size reduction is a unimodular row op → lattice unchanged.
        // Verify via HNF equality.
        let m: [[Int<8>; 4]; 4] = [
            [n(2), n(0), n(0), n(0)],
            [n(5), n(3), n(0), n(0)],
            [n(7), n(11), n(2), n(0)],
            [n(13), n(17), n(19), n(5)],
        ];
        let reduced = size_reduce_4x4(&m);
        // Both reduce to the same HNF.
        let hnf_orig = hnf_4x4(&m);
        let hnf_reduced = hnf_4x4(&reduced);
        assert_eq!(hnf_orig, hnf_reduced);
    }

    #[test]
    fn gram_of_identity_is_identity() {
        let id: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let g = gram_matrix_4x4(&id);
        assert_eq!(g, id);
    }

    #[test]
    fn gram_of_diagonal_is_squared_diagonal() {
        let m: [[Int<8>; 4]; 4] = [
            [n(2), n(0), n(0), n(0)],
            [n(0), n(3), n(0), n(0)],
            [n(0), n(0), n(5), n(0)],
            [n(0), n(0), n(0), n(7)],
        ];
        let g = gram_matrix_4x4(&m);
        let expected: [[Int<8>; 4]; 4] = [
            [n(4), n(0), n(0), n(0)],
            [n(0), n(9), n(0), n(0)],
            [n(0), n(0), n(25), n(0)],
            [n(0), n(0), n(0), n(49)],
        ];
        assert_eq!(g, expected);
    }

    #[test]
    fn gram_is_symmetric() {
        let m: [[Int<8>; 4]; 4] = [
            [n(3), n(1), n(0), n(2)],
            [n(2), n(4), n(1), n(0)],
            [n(0), n(1), n(5), n(1)],
            [n(1), n(0), n(2), n(3)],
        ];
        let g = gram_matrix_4x4(&m);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(g[i][j], g[j][i], "G is symmetric");
            }
        }
    }

    #[test]
    fn gram_determinant_equals_basis_determinant_squared() {
        use crate::quaternion::ideal::det_4x4;
        let m: [[Int<8>; 4]; 4] = [
            [n(3), n(1), n(0), n(2)],
            [n(2), n(4), n(1), n(0)],
            [n(0), n(1), n(5), n(1)],
            [n(1), n(0), n(2), n(3)],
        ];
        let det_b = det_4x4(&m);
        let g = gram_matrix_4x4(&m);
        let det_g = det_4x4(&g);
        // det(B)² = det(G).
        let det_b_sq = det_b.wrapping_mul(&det_b);
        assert_eq!(det_g, det_b_sq);
    }

    #[test]
    fn gram_diagonal_holds_squared_norms() {
        let m: [[Int<8>; 4]; 4] = [
            [n(3), n(4), n(0), n(0)],
            [n(1), n(1), n(1), n(1)],
            [n(2), n(0), n(2), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let g = gram_matrix_4x4(&m);
        // G[i][i] = ⟨bᵢ, bᵢ⟩ = ‖bᵢ‖².
        assert_eq!(g[0][0], n(9 + 16)); // 25
        assert_eq!(g[1][1], n(4));
        assert_eq!(g[2][2], n(8));
        assert_eq!(g[3][3], n(1));
    }

    #[test]
    fn integer_gso_identity_is_canonical() {
        let id: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let (d, lam) = integer_gso_4x4(&id).expect("identity is full-rank");
        for i in 0..5 {
            assert_eq!(d[i], n(1), "d[{i}] = 1 for orthonormal basis");
        }
        for i in 1..4 {
            for j in 0..i {
                assert_eq!(
                    lam[i][j],
                    n(0),
                    "off-diagonal lam zero for orthogonal basis"
                );
            }
        }
    }

    #[test]
    fn integer_gso_d_matches_leading_gram_determinants() {
        use crate::quaternion::ideal::det_4x4;
        let m: [[Int<8>; 4]; 4] = [
            [n(3), n(1), n(0), n(2)],
            [n(2), n(4), n(1), n(0)],
            [n(0), n(1), n(5), n(1)],
            [n(1), n(0), n(2), n(3)],
        ];
        let (d, _lam) = integer_gso_4x4(&m).expect("rank-4 basis");
        let gram = gram_matrix_4x4(&m);
        assert_eq!(d[0], n(1));
        // d[1] = G[0][0]
        assert_eq!(d[1], gram[0][0]);
        // d[2] = G[0][0] * G[1][1] - G[1][0]²  (2×2 leading minor)
        let det2 = gram[0][0]
            .wrapping_mul(&gram[1][1])
            .wrapping_sub(&gram[1][0].wrapping_mul(&gram[1][0]));
        assert_eq!(d[2], det2);
        // d[4] = det of full 4×4 Gram
        assert_eq!(d[4], det_4x4(&gram));
    }

    #[test]
    fn lll_identity_is_fixpoint() {
        let id: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let reduced = lll_4x4(&id);
        assert_eq!(reduced, id);
    }

    #[test]
    fn lll_preserves_lattice() {
        use crate::quaternion::hnf::hnf_4x4;
        let m: [[Int<8>; 4]; 4] = [
            [n(7), n(2), n(0), n(1)],
            [n(13), n(5), n(2), n(0)],
            [n(0), n(11), n(7), n(3)],
            [n(4), n(0), n(9), n(13)],
        ];
        let reduced = lll_4x4(&m);
        let hnf_before = hnf_4x4(&m);
        let hnf_after = hnf_4x4(&reduced);
        assert_eq!(hnf_before, hnf_after, "LLL must preserve the Z-lattice");
    }

    #[test]
    fn lll_reduces_total_norm_on_skewed_basis() {
        let m: [[Int<8>; 4]; 4] = [
            [n(20), n(0), n(0), n(0)],
            [n(19), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let before: Int<8> = (0..4)
            .map(|i| norm2(&m[i]))
            .fold(n(0), |acc, x| acc.wrapping_add(&x));
        let reduced = lll_4x4(&m);
        let after: Int<8> = (0..4)
            .map(|i| norm2(&reduced[i]))
            .fold(n(0), |acc, x| acc.wrapping_add(&x));
        assert!(after < before, "LLL must not increase total norm²");
    }

    #[test]
    fn lll_output_satisfies_lovasz_condition() {
        // After LLL terminates, the Lovász test must hold at every k.
        let m: [[Int<8>; 4]; 4] = [
            [n(15), n(0), n(0), n(0)],
            [n(7), n(2), n(0), n(0)],
            [n(0), n(0), n(3), n(0)],
            [n(0), n(0), n(0), n(4)],
        ];
        let reduced = lll_4x4(&m);
        let (d, lam) = integer_gso_4x4(&reduced).expect("LLL output rank-4");
        let three = Int::<8>::from_i64(3);
        let four = Int::<8>::from_i64(4);
        for k in 1..4 {
            let lhs = four.wrapping_mul(&d[k - 1].wrapping_mul(&d[k + 1]));
            let dk_sq = d[k].wrapping_mul(&d[k]);
            let lam_sq = lam[k][k - 1].wrapping_mul(&lam[k][k - 1]);
            let rhs = three
                .wrapping_mul(&dk_sq)
                .wrapping_sub(&four.wrapping_mul(&lam_sq));
            assert!(
                lhs >= rhs,
                "Lovász condition violated at k={k}: lhs={lhs:?}, rhs={rhs:?}"
            );
        }
    }

    #[test]
    fn qf_eval_4x4_wide_parity_at_same_width() {
        // At WIDE = NARROW, the wide evaluation should equal the narrow.
        let c: [Int<8>; 4] = [n(3), n(4), n(0), n(0)];
        let g: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let narrow = qf_eval_4x4(&c, &g);
        let wide: Int<8> = qf_eval_4x4_wide::<8, 8>(&c, &g);
        assert_eq!(wide, narrow);
        assert_eq!(wide, n(25)); // ‖(3,4,0,0)‖² = 25
    }

    #[test]
    fn qf_eval_4x4_wide_matches_widened_narrow_on_safe_inputs() {
        // For narrow-safe inputs, qf_eval_4x4_wide<8,16>(...) should
        // equal widen_int_lattice(qf_eval_4x4<8>(...)) — i.e. computing
        // in narrow then widening matches computing in wide directly.
        let c: [Int<8>; 4] = [n(2), n(3), n(1), n(1)];
        let g: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(7), n(0)],
            [n(0), n(0), n(0), n(7)],
        ];
        let narrow = qf_eval_4x4(&c, &g);
        let wide: Int<16> = qf_eval_4x4_wide::<8, 16>(&c, &g);
        // narrow = 4 + 9 + 7·1 + 7·1 = 27. wide should equal widen(narrow).
        let narrow_widened = widen_int_lattice::<8, 16>(&narrow);
        assert_eq!(wide, narrow_widened);
        assert_eq!(narrow, n(27));
    }

    #[test]
    fn qf_eval_4x4_wide_handles_large_inputs_without_overflow() {
        // Construct an input where narrow would overflow. Use Int<8>
        // entries near the magnitude ceiling that, when multiplied, push
        // intermediates well past 2^511. WIDE=16 has 1023 bits of
        // headroom so the result computes correctly.
        //
        // c = (k, 0, 0, 0), G = diag(M, 0, 0, 0) → result = k² · M.
        // Choose k = 2^126, M = 2^260: k² · M = 2^512 (overflows narrow
        // by 1 bit; trivially fits wide).
        use crypto_bigint::Uint;
        let k_uint: Uint<8> = Uint::<8>::ONE.shl_vartime(126);
        let k_int: Int<8> = *k_uint.as_int();
        let m_uint: Uint<8> = Uint::<8>::ONE.shl_vartime(260);
        let m_int: Int<8> = *m_uint.as_int();

        let c: [Int<8>; 4] = [k_int, n(0), n(0), n(0)];
        let g: [[Int<8>; 4]; 4] = [
            [m_int, n(0), n(0), n(0)],
            [n(0), n(0), n(0), n(0)],
            [n(0), n(0), n(0), n(0)],
            [n(0), n(0), n(0), n(0)],
        ];

        let wide: Int<16> = qf_eval_4x4_wide::<8, 16>(&c, &g);
        // Expected: k² · M = 2^252 · 2^260 = 2^512. Build it via Uint<16>.
        let expected_uint: Uint<16> = Uint::<16>::ONE.shl_vartime(512);
        let expected: Int<16> = *expected_uint.as_int();
        assert_eq!(wide, expected);
    }

    #[test]
    fn lll_4x4_in_metric_wide_parity_at_same_width() {
        // At WIDE = NARROW = 8, the wide variant should produce the same
        // result as the narrow `lll_4x4_in_metric` (extra widen/narrow
        // round-trip is a no-op).
        let identity: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let metric: [[Int<8>; 4]; 4] = [
            [n(4), n(0), n(0), n(2)],
            [n(0), n(4), n(2), n(0)],
            [n(0), n(2), n(8), n(0)],
            [n(2), n(0), n(0), n(8)],
        ];
        let narrow = lll_4x4_in_metric(&identity, &metric);
        let wide_parity: [[Int<8>; 4]; 4] = lll_4x4_in_metric_wide::<8, 8>(&identity, &metric);
        assert_eq!(wide_parity, narrow);
    }

    #[test]
    fn lll_4x4_in_metric_wide_preserves_lattice() {
        // The wide variant must preserve the lattice span (HNF equality
        // before and after) — same invariant as the narrow version.
        use crate::quaternion::hnf::hnf_4x4;
        let m: [[Int<8>; 4]; 4] = [
            [n(7), n(2), n(0), n(1)],
            [n(13), n(5), n(2), n(0)],
            [n(0), n(11), n(7), n(3)],
            [n(4), n(0), n(9), n(13)],
        ];
        let metric: [[Int<8>; 4]; 4] = [
            [n(4), n(0), n(0), n(2)],
            [n(0), n(4), n(2), n(0)],
            [n(0), n(2), n(8), n(0)],
            [n(2), n(0), n(0), n(8)],
        ];
        let reduced: [[Int<8>; 4]; 4] = lll_4x4_in_metric_wide::<8, 16>(&m, &metric);
        assert_eq!(hnf_4x4(&reduced), hnf_4x4(&m));
    }

    #[test]
    fn lll_4x4_in_metric_wide_matches_narrow_on_safe_inputs() {
        // For inputs where the narrow path doesn't overflow, the wide
        // variant (after narrow round-trip) should produce the same
        // basis as the narrow path.
        let m: [[Int<8>; 4]; 4] = [
            [n(15), n(0), n(0), n(0)],
            [n(7), n(2), n(0), n(0)],
            [n(0), n(0), n(3), n(0)],
            [n(0), n(0), n(0), n(4)],
        ];
        let metric: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let narrow = lll_4x4_in_metric(&m, &metric);
        let wide: [[Int<8>; 4]; 4] = lll_4x4_in_metric_wide::<8, 16>(&m, &metric);
        assert_eq!(wide, narrow);
    }

    #[test]
    fn pull_back_gram_identity_metric_recovers_basis_gram() {
        // M = I ⇒ B · I · Bᵀ = B · Bᵀ = gram_matrix_4x4(B).
        let b: [[Int<8>; 4]; 4] = [
            [n(1), n(2), n(0), n(0)],
            [n(0), n(1), n(3), n(0)],
            [n(0), n(0), n(1), n(5)],
            [n(1), n(0), n(0), n(1)],
        ];
        let identity: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        assert_eq!(pull_back_gram(&b, &identity), gram_matrix_4x4(&b));
    }

    #[test]
    fn pull_back_gram_identity_basis_returns_metric() {
        // B = I ⇒ I · M · Iᵀ = M.
        let identity: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let m: [[Int<8>; 4]; 4] = [
            [n(4), n(0), n(0), n(2)],
            [n(0), n(4), n(2), n(0)],
            [n(0), n(2), n(8), n(0)],
            [n(2), n(0), n(0), n(8)],
        ];
        assert_eq!(pull_back_gram(&identity, &m), m);
    }

    #[test]
    fn integer_gso_from_gram_matches_from_basis_at_identity() {
        let id: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let gram_id = gram_matrix_4x4(&id);
        let (d_basis, lam_basis) = integer_gso_4x4(&id).expect("rank-4");
        let (d_gram, lam_gram) = integer_gso_4x4_from_gram(&gram_id).expect("rank-4");
        assert_eq!(d_basis, d_gram);
        assert_eq!(lam_basis, lam_gram);
    }

    #[test]
    fn lll_4x4_in_metric_identity_metric_matches_euclidean_lll() {
        // With identity metric, lll_4x4_in_metric and lll_4x4 must agree.
        let identity: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let m: [[Int<8>; 4]; 4] = [
            [n(7), n(2), n(0), n(1)],
            [n(13), n(5), n(2), n(0)],
            [n(0), n(11), n(7), n(3)],
            [n(4), n(0), n(9), n(13)],
        ];
        assert_eq!(lll_4x4_in_metric(&m, &identity), lll_4x4(&m));
    }

    #[test]
    fn lll_4x4_in_metric_identity_basis_is_fixpoint() {
        // Even under a non-trivial metric, the identity basis can't be
        // further reduced — it's already as "short" as Z⁴ allows.
        let identity: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let m: [[Int<8>; 4]; 4] = [
            [n(4), n(0), n(0), n(2)],
            [n(0), n(4), n(2), n(0)],
            [n(0), n(2), n(8), n(0)],
            [n(2), n(0), n(0), n(8)],
        ];
        let reduced = lll_4x4_in_metric(&identity, &m);
        // The reduced basis spans the same lattice but may differ from
        // identity up to permutation/sign. Verify span equality via HNF.
        use crate::quaternion::hnf::hnf_4x4;
        assert_eq!(hnf_4x4(&reduced), hnf_4x4(&identity));
    }

    #[test]
    fn lll_4x4_in_metric_preserves_lattice() {
        // For any positive-definite metric, the reduced basis spans the
        // same Z-lattice as the input. Verify via HNF equality.
        use crate::quaternion::hnf::hnf_4x4;
        let m: [[Int<8>; 4]; 4] = [
            [n(7), n(2), n(0), n(1)],
            [n(13), n(5), n(2), n(0)],
            [n(0), n(11), n(7), n(3)],
            [n(4), n(0), n(9), n(13)],
        ];
        let metric: [[Int<8>; 4]; 4] = [
            [n(4), n(0), n(0), n(2)],
            [n(0), n(4), n(2), n(0)],
            [n(0), n(2), n(8), n(0)],
            [n(2), n(0), n(0), n(8)],
        ];
        let reduced = lll_4x4_in_metric(&m, &metric);
        assert_eq!(hnf_4x4(&reduced), hnf_4x4(&m));
    }

    #[test]
    fn qf_eval_zero_vector_is_zero() {
        let c: [Int<8>; 4] = [n(0), n(0), n(0), n(0)];
        let g: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(7), n(0)],
            [n(0), n(0), n(0), n(7)],
        ];
        assert_eq!(qf_eval_4x4(&c, &g), n(0));
    }

    #[test]
    fn qf_eval_with_identity_gram_matches_norm2() {
        let c: [Int<8>; 4] = [n(3), n(4), n(0), n(0)];
        let g_id: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        // ‖c‖² = 9 + 16 = 25
        assert_eq!(qf_eval_4x4(&c, &g_id), norm2(&c));
        assert_eq!(qf_eval_4x4(&c, &g_id), n(25));
    }

    #[test]
    fn qf_eval_with_quaternion_norm_form_returns_a2_b2_p_c2_d2() {
        // c = (a, b, c, d) = (2, 3, 1, 1), G = diag(1, 1, 7, 7).
        // Q = 4 + 9 + 7·1 + 7·1 = 27.
        let c: [Int<8>; 4] = [n(2), n(3), n(1), n(1)];
        let g: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(7), n(0)],
            [n(0), n(0), n(0), n(7)],
        ];
        assert_eq!(qf_eval_4x4(&c, &g), n(27));
    }

    #[test]
    fn qf_eval_handles_off_diagonal_gram() {
        // G = [[1,1,0,0],[1,2,0,0],[0,0,0,0],[0,0,0,0]]. c = (3, 2, 0, 0).
        // c^T G c = 3·(1·3 + 1·2) + 2·(1·3 + 2·2) = 3·5 + 2·7 = 15 + 14 = 29.
        let c: [Int<8>; 4] = [n(3), n(2), n(0), n(0)];
        let g: [[Int<8>; 4]; 4] = [
            [n(1), n(1), n(0), n(0)],
            [n(1), n(2), n(0), n(0)],
            [n(0), n(0), n(0), n(0)],
            [n(0), n(0), n(0), n(0)],
        ];
        assert_eq!(qf_eval_4x4(&c, &g), n(29));
    }

    #[test]
    fn find_lattice_point_always_accept_returns_origin() {
        // threshold = 10, α = β = 1. Origin (0, 0) has Q = 0 ≤ 10.
        // First-call accept(&[0, 0]) returns true, so the enumerator
        // short-circuits and returns the origin.
        let result = find_lattice_point_2x2_under_quad_threshold(&n(10), &n(1), &n(1), |_| true);
        assert_eq!(result, Some([n(0), n(0)]));
    }

    #[test]
    fn find_lattice_point_excluding_origin_returns_nearest() {
        // threshold = 5, α = β = 1. Predicate rejects the origin.
        // Next candidate (in this enumeration order): (0, 1).
        let result = find_lattice_point_2x2_under_quad_threshold(&n(5), &n(1), &n(1), |v| {
            !(v[0] == n(0) && v[1] == n(0))
        });
        assert_eq!(result, Some([n(0), n(1)]));
    }

    #[test]
    fn find_lattice_point_no_match_returns_none() {
        // threshold = 0, α = β = 1. Only (0, 0) qualifies; predicate
        // rejects it; no other point fits. Result: None.
        let result = find_lattice_point_2x2_under_quad_threshold(&n(0), &n(1), &n(1), |v| {
            !(v[0] == n(0) && v[1] == n(0))
        });
        assert_eq!(result, None);
    }

    #[test]
    fn find_lattice_point_respects_weighted_threshold() {
        // threshold = 7, α = 7, β = 1, predicate requires v[0] != 0.
        // (0, *) all rejected; (1, 0): Q = 7 ≤ 7, accept → return (1, 0).
        let result =
            find_lattice_point_2x2_under_quad_threshold(&n(7), &n(7), &n(1), |v| v[0] != n(0));
        assert_eq!(result, Some([n(1), n(0)]));
    }

    #[test]
    fn find_lattice_point_negative_threshold_returns_none() {
        let result = find_lattice_point_2x2_under_quad_threshold(&n(-1), &n(1), &n(1), |_| true);
        assert_eq!(result, None);
    }

    #[test]
    fn gram_dot2_with_identity_matches_dot2() {
        let u: [Int<8>; 2] = [n(3), n(4)];
        let v: [Int<8>; 2] = [n(5), n(-2)];
        let g_id: [[Int<8>; 2]; 2] = [[n(1), n(0)], [n(0), n(1)]];
        assert_eq!(gram_dot2(&u, &v, &g_id), dot2(&u, &v));
        assert_eq!(gram_dot2(&u, &v, &g_id), n(7)); // 3·5 + 4·(-2) = 7
    }

    #[test]
    fn gram_dot2_diagonal_scales_per_axis() {
        let u: [Int<8>; 2] = [n(3), n(4)];
        let v: [Int<8>; 2] = [n(5), n(-2)];
        let g: [[Int<8>; 2]; 2] = [[n(2), n(0)], [n(0), n(7)]];
        // u^T G v = 2·3·5 + 7·4·(-2) = 30 − 56 = -26
        assert_eq!(gram_dot2(&u, &v, &g), n(-26));
    }

    #[test]
    fn gram_dot2_symmetric_for_symmetric_g() {
        let u: [Int<8>; 2] = [n(3), n(4)];
        let v: [Int<8>; 2] = [n(5), n(-2)];
        let g: [[Int<8>; 2]; 2] = [[n(2), n(1)], [n(1), n(3)]];
        assert_eq!(gram_dot2(&u, &v, &g), gram_dot2(&v, &u, &g));
    }

    #[test]
    fn lll_2x2_with_identity_gram_matches_lll_2x2() {
        let m: [[Int<8>; 2]; 2] = [[n(1), n(0)], [n(7), n(1)]];
        let g_id: [[Int<8>; 2]; 2] = [[n(1), n(0)], [n(0), n(1)]];
        assert_eq!(lll_2x2_with_gram(&m, &g_id), lll_2x2(&m));
    }

    #[test]
    fn babai_cvp_2x2_with_identity_gram_matches_euclidean() {
        let basis: [[Int<8>; 2]; 2] = [[n(5), n(0)], [n(1), n(7)]];
        let target = [n(12), n(8)];
        let g_id: [[Int<8>; 2]; 2] = [[n(1), n(0)], [n(0), n(1)]];
        assert_eq!(
            babai_cvp_2x2_with_gram(&basis, &target, &g_id),
            babai_cvp_2x2(&basis, &target)
        );
    }

    #[test]
    fn babai_cvp_2x2_with_gram_returns_lattice_point_on_scaled_basis() {
        // basis = 3·e0 + 5·e1 → lattice = 3Z × 5Z. Any weighted-CVP result
        // must be in this lattice (coord 0 divisible by 3, coord 1 by 5).
        use crate::quaternion::hnf::int_div_floor;
        let basis: [[Int<8>; 2]; 2] = [[n(3), n(0)], [n(0), n(5)]];
        let target = [n(10), n(13)];
        let g: [[Int<8>; 2]; 2] = [[n(2), n(0)], [n(0), n(7)]];
        let result = babai_cvp_2x2_with_gram(&basis, &target, &g);
        let q3 = int_div_floor(&result[0], &n(3));
        assert_eq!(
            result[0],
            q3.wrapping_mul(&n(3)),
            "coord 0 = {:?} not in 3Z",
            result[0]
        );
        let q5 = int_div_floor(&result[1], &n(5));
        assert_eq!(
            result[1],
            q5.wrapping_mul(&n(5)),
            "coord 1 = {:?} not in 5Z",
            result[1]
        );
    }

    #[test]
    fn dot2_orthogonal_is_zero() {
        let e0: [Int<8>; 2] = [n(1), n(0)];
        let e1: [Int<8>; 2] = [n(0), n(1)];
        assert_eq!(dot2(&e0, &e1), n(0));
    }

    #[test]
    fn norm2_2_3_4_is_25() {
        let v: [Int<8>; 2] = [n(3), n(4)];
        assert_eq!(norm2_2(&v), n(25));
    }

    #[test]
    fn lll_2x2_identity_is_fixpoint() {
        let id: [[Int<8>; 2]; 2] = [[n(1), n(0)], [n(0), n(1)]];
        let reduced = lll_2x2(&id);
        assert_eq!(reduced, id);
    }

    #[test]
    fn lll_2x2_reduces_skewed_basis() {
        // basis = [(1, 0), (7, 1)]. Size-reduce b_1 against b_0:
        //   r = round(⟨b_1, b_0⟩ / ‖b_0‖²) = round(7/1) = 7
        //   b_1 ← (7 − 7·1, 1 − 7·0) = (0, 1).
        // Result: orthogonal basis [(1, 0), (0, 1)].
        let m: [[Int<8>; 2]; 2] = [[n(1), n(0)], [n(7), n(1)]];
        let reduced = lll_2x2(&m);
        assert_eq!(reduced[0], [n(1), n(0)]);
        assert_eq!(reduced[1], [n(0), n(1)]);
    }

    #[test]
    fn lll_2x2_preserves_determinant_up_to_sign() {
        // Unimodular row ops preserve `|det|`. At 2×2,
        // `det([b_0; b_1]) = b_0[0]·b_1[1] − b_0[1]·b_1[0]`.
        let m: [[Int<8>; 2]; 2] = [[n(5), n(3)], [n(2), n(7)]];
        let det_before = m[0][0]
            .wrapping_mul(&m[1][1])
            .wrapping_sub(&m[0][1].wrapping_mul(&m[1][0]));
        let reduced = lll_2x2(&m);
        let det_after = reduced[0][0]
            .wrapping_mul(&reduced[1][1])
            .wrapping_sub(&reduced[0][1].wrapping_mul(&reduced[1][0]));
        // Either equal or negated (a swap flips the sign of the determinant).
        assert!(
            det_before == det_after || det_before == det_after.wrapping_neg(),
            "|det| must be preserved across Lagrange reduction"
        );
    }

    #[test]
    fn babai_cvp_2x2_target_on_lattice_returns_target() {
        let id: [[Int<8>; 2]; 2] = [[n(1), n(0)], [n(0), n(1)]];
        let target = [n(5), n(-3)];
        let nearest = babai_cvp_2x2(&id, &target);
        assert_eq!(nearest, target);
    }

    #[test]
    fn babai_cvp_2x2_zero_target_returns_zero() {
        let id: [[Int<8>; 2]; 2] = [[n(1), n(0)], [n(0), n(1)]];
        let target = [n(0), n(0)];
        let nearest = babai_cvp_2x2(&id, &target);
        assert_eq!(nearest, [n(0), n(0)]);
    }

    #[test]
    fn babai_cvp_2x2_scaled_identity_rounds_per_coordinate() {
        // basis = 10·identity. Target (33, −24) rounds per-axis:
        //   33 → 30 (|33−30|=3 < |33−40|=7)
        //   −24 → −20 (|−24−(−20)|=4 < |−24−(−30)|=6)
        let basis: [[Int<8>; 2]; 2] = [[n(10), n(0)], [n(0), n(10)]];
        let target = [n(33), n(-24)];
        let nearest = babai_cvp_2x2(&basis, &target);
        assert_eq!(nearest, [n(30), n(-20)]);
    }

    #[test]
    fn babai_cvp_2x2_residue_not_longer_than_target() {
        // Returning origin (a lattice point) is always feasible, so
        // ‖target − nearest‖² ≤ ‖target‖² must hold.
        let basis: [[Int<8>; 2]; 2] = [[n(5), n(0)], [n(1), n(7)]];
        let target = [n(12), n(8)];
        let target_norm = norm2_2(&target);
        let nearest = babai_cvp_2x2(&basis, &target);
        let residue: [Int<8>; 2] = [
            target[0].wrapping_sub(&nearest[0]),
            target[1].wrapping_sub(&nearest[1]),
        ];
        let residue_norm = norm2_2(&residue);
        assert!(
            residue_norm <= target_norm,
            "residue norm² {residue_norm:?} exceeds target norm² {target_norm:?}"
        );
    }

    #[test]
    fn babai_cvp_target_on_lattice_returns_target() {
        // Identity basis: every integer vector is a lattice point.
        let id: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let target = [n(3), n(5), n(-2), n(7)];
        let nearest = babai_cvp_4x4(&id, &target);
        assert_eq!(nearest, target);
    }

    #[test]
    fn babai_cvp_zero_target_returns_zero() {
        let id: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let target = [n(0), n(0), n(0), n(0)];
        let nearest = babai_cvp_4x4(&id, &target);
        assert_eq!(nearest, [n(0); 4]);
    }

    #[test]
    fn babai_cvp_scaled_identity_rounds_per_coordinate() {
        // Basis = 10·identity. Lattice points have each coord divisible by 10.
        // Target (33, 57, 71, 24) rounds to nearest multiple of 10 per axis:
        //   33 → 30 (|33−30|=3 < |33−40|=7)
        //   57 → 60 (|57−60|=3 < |57−50|=7)
        //   71 → 70 (|71−70|=1 < |71−80|=9)
        //   24 → 20 (|24−20|=4 < |24−30|=6)
        let basis: [[Int<8>; 4]; 4] = [
            [n(10), n(0), n(0), n(0)],
            [n(0), n(10), n(0), n(0)],
            [n(0), n(0), n(10), n(0)],
            [n(0), n(0), n(0), n(10)],
        ];
        let target = [n(33), n(57), n(71), n(24)];
        let nearest = babai_cvp_4x4(&basis, &target);
        assert_eq!(nearest, [n(30), n(60), n(70), n(20)]);
    }

    #[test]
    fn babai_cvp_residue_not_longer_than_target() {
        // For any basis, returning zero is always valid (zero ∈ lattice).
        // Therefore the residue `target − nearest` must be no longer than
        // `target` itself — otherwise the algorithm chose a worse lattice
        // point than the origin.
        let basis: [[Int<8>; 4]; 4] = [
            [n(5), n(0), n(0), n(0)],
            [n(1), n(7), n(0), n(0)],
            [n(0), n(2), n(11), n(0)],
            [n(0), n(0), n(3), n(13)],
        ];
        let target = [n(12), n(8), n(33), n(40)];
        let target_norm = norm2(&target);
        let nearest = babai_cvp_4x4(&basis, &target);
        let mut residue = [n(0); 4];
        for k in 0..4 {
            residue[k] = target[k].wrapping_sub(&nearest[k]);
        }
        let residue_norm = norm2(&residue);
        assert!(
            residue_norm <= target_norm,
            "residue norm² {residue_norm:?} exceeds target norm² {target_norm:?}"
        );
    }

    #[test]
    fn babai_cvp_result_in_axis_aligned_lattice() {
        // Lattice membership on a clean axis-aligned basis: every
        // coordinate of `nearest` must be a multiple of the corresponding
        // basis vector's scale. Cheap to verify, no HNF gymnastics.
        use crate::quaternion::hnf::int_div_floor;
        let basis: [[Int<8>; 4]; 4] = [
            [n(10), n(0), n(0), n(0)],
            [n(0), n(10), n(0), n(0)],
            [n(0), n(0), n(10), n(0)],
            [n(0), n(0), n(0), n(10)],
        ];
        let target = [n(73), n(57), n(-21), n(40)];
        let nearest = babai_cvp_4x4(&basis, &target);
        let ten = n(10);
        for (k, coord) in nearest.iter().enumerate() {
            let q = int_div_floor(coord, &ten);
            let rebuilt = q.wrapping_mul(&ten);
            assert_eq!(
                *coord, rebuilt,
                "coordinate {k} = {coord:?} not divisible by 10 (not in lattice)"
            );
        }
    }

    #[test]
    fn size_reduce_lowers_norm_via_norm2_comparison() {
        // Skewed input: `b₁ = (10, 1, 0, 0)` has norm² = 101.
        // After size reduction against `b₀ = (1, 0, 0, 0)`: r = round(10/1) = 10,
        // b₁ ← (0, 1, 0, 0) with norm² = 1.
        let m: [[Int<8>; 4]; 4] = [
            [n(1), n(0), n(0), n(0)],
            [n(10), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ];
        let before_b1_norm = norm2(&m[1]);
        assert_eq!(before_b1_norm, n(101));
        let reduced = size_reduce_4x4(&m);
        let after_b1_norm = norm2(&reduced[1]);
        assert_eq!(after_b1_norm, n(1));
        assert!(after_b1_norm < before_b1_norm);
    }
}
