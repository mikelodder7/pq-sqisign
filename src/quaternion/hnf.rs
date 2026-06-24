// SPDX-License-Identifier: MIT OR Apache-2.0
//! Hermite Normal Form (HNF) reduction of 4×4 integer matrices.
//!
//! Every full-rank integer matrix `M ∈ Z^{4×4}` is row-equivalent (via
//! left-multiplication by a unimodular matrix `U ∈ GL_4(Z)`) to a unique
//! upper-triangular matrix `H = U · M` with:
//!
//! 1. `H[i][i] > 0` for every diagonal entry.
//! 2. `0 ≤ H[i][j] < H[i][i]` for `j > i` (off-diagonals strictly smaller
//!    than the pivot below them in the same column).
//!
//! `H` is the **Hermite Normal Form** of `M`.
//!
//! KLPT's equivalent-ideal lift, ideal multiplication, and the
//! lattice-membership test all reduce to HNF. The standard algorithm is
//! row-by-row column elimination via the extended Euclidean algorithm.
//!
//! Coefficients use `Int<LIMBS>` from `crypto-bigint` so the wider
//! quaternion arithmetic shares one integer type.

use crypto_bigint::Int;

/// HNF of an `ROWS × 4` integer matrix. The top 4 rows of the result form
/// the canonical upper-triangular basis; rows 4..ROWS are zero.
///
/// Used by `ideal_multiply` to reduce the 16-pair product matrix.
// HNF cross-row reduction: the pivot row m[col] is read while sibling rows
// m[r] mutate (and m[col][c] feeds m[r][c]); index form mirrors the C operation
// order and avoids split-borrow scaffolding over the same matrix.
#[allow(clippy::needless_range_loop)]
pub fn hnf_rect_4cols<const ROWS: usize, const LIMBS: usize>(
    input: &[[Int<LIMBS>; 4]; ROWS],
) -> [[Int<LIMBS>; 4]; ROWS] {
    let mut m = *input;
    for col in 0..4 {
        loop {
            let mut min_idx: Option<usize> = None;
            let mut min_abs = Int::<LIMBS>::from_i64(0);
            for r in col..ROWS {
                let v = m[r][col];
                let zero = Int::<LIMBS>::from_i64(0);
                if v == zero {
                    continue;
                }
                let abs_v = v.abs();
                let abs_min = min_abs.abs();
                if min_idx.is_none() || abs_v < abs_min {
                    min_idx = Some(r);
                    min_abs = v;
                }
            }
            let Some(pivot_row) = min_idx else {
                break;
            };
            if pivot_row != col {
                m.swap(col, pivot_row);
            }
            let pivot = m[col][col];
            if bool::from(pivot.is_negative()) {
                for c in 0..4 {
                    m[col][c] = m[col][c].wrapping_neg();
                }
            }
            let pivot = m[col][col];
            let mut any_nonzero_below = false;
            for r in (col + 1)..ROWS {
                let entry = m[r][col];
                let zero = Int::<LIMBS>::from_i64(0);
                if entry == zero {
                    continue;
                }
                let q = int_div_floor(&entry, &pivot);
                for c in 0..4 {
                    let qm = q.wrapping_mul(&m[col][c]);
                    m[r][c] = m[r][c].wrapping_sub(&qm);
                }
                if m[r][col] != zero {
                    any_nonzero_below = true;
                }
            }
            if !any_nonzero_below {
                break;
            }
        }
        // Reduce rows above the pivot mod the pivot value.
        let pivot = m[col][col];
        let zero = Int::<LIMBS>::from_i64(0);
        if pivot != zero {
            for r in 0..col {
                let entry = m[r][col];
                if entry == zero {
                    continue;
                }
                let q = int_div_floor(&entry, &pivot);
                for c in 0..4 {
                    let qm = q.wrapping_mul(&m[col][c]);
                    m[r][c] = m[r][c].wrapping_sub(&qm);
                }
            }
        }
    }
    m
}

/// Compute the HNF of a 4×4 integer matrix.
///
/// **Note**: the algorithm here is the textbook row-reduction (no LLL-style
/// size-control), which means intermediate entries can grow exponentially in
/// adversarial inputs. KLPT operates on ideals whose bases come out of
/// Cornacchia-and-conjugation chains, where this is acceptable; the
/// fraction-free / Bareiss variant lands when profiling shows overflow.
// HNF cross-row reduction: the pivot row m[col] is read while sibling rows
// m[r] mutate (and m[col][c] feeds m[r][c]); index form mirrors the C operation
// order and avoids split-borrow scaffolding over the same matrix.
#[allow(clippy::needless_range_loop)]
pub fn hnf_4x4<const LIMBS: usize>(input: &[[Int<LIMBS>; 4]; 4]) -> [[Int<LIMBS>; 4]; 4] {
    let mut m = *input;
    // Process columns left-to-right; pivot row index `i` runs alongside.
    for col in 0..4 {
        // Use rows `col..4` as the active block.
        // Step 1: bring a non-zero entry to position (col, col).
        // Repeatedly Euclidean-reduce the column entries below the pivot.
        loop {
            // Find the smallest non-zero |entry| in rows col..4 at this column.
            let mut min_idx: Option<usize> = None;
            let mut min_abs = Int::<LIMBS>::from_i64(0);
            for r in col..4 {
                let v = m[r][col];
                let zero = Int::<LIMBS>::from_i64(0);
                if v == zero {
                    continue;
                }
                let abs_v = v.abs();
                let abs_min = min_abs.abs();
                if min_idx.is_none() || abs_v < abs_min {
                    min_idx = Some(r);
                    min_abs = v;
                }
            }
            let Some(pivot_row) = min_idx else {
                // All entries in this column at rows col..4 are zero; pivot stays 0.
                break;
            };
            // Swap pivot row to position `col`.
            if pivot_row != col {
                m.swap(col, pivot_row);
            }
            let pivot = m[col][col];
            // Make the pivot positive (sign-flip the row if needed).
            if bool::from(pivot.is_negative()) {
                for c in 0..4 {
                    m[col][c] = m[col][c].wrapping_neg();
                }
            }
            let pivot = m[col][col];
            // Reduce every other row in this column by div-mod with the pivot.
            let mut any_nonzero_below = false;
            for r in (col + 1)..4 {
                let entry = m[r][col];
                let zero = Int::<LIMBS>::from_i64(0);
                if entry == zero {
                    continue;
                }
                // q = entry div pivot (toward zero); we use repeated subtraction
                // because `Int::div` is a more involved interface; pivots in this
                // context are small.
                let q = int_div_floor(&entry, &pivot);
                let q_times_pivot = q.wrapping_mul(&pivot);
                for c in 0..4 {
                    let qm = q.wrapping_mul(&m[col][c]);
                    m[r][c] = m[r][c].wrapping_sub(&qm);
                }
                let _ = q_times_pivot;
                // After this, m[r][col] = entry - q*pivot ∈ [0, pivot).
                if m[r][col] != zero {
                    any_nonzero_below = true;
                }
            }
            if !any_nonzero_below {
                break;
            }
        }
        // Step 2: reduce rows above the pivot mod the pivot value.
        let pivot = m[col][col];
        let zero = Int::<LIMBS>::from_i64(0);
        if pivot != zero {
            for r in 0..col {
                let entry = m[r][col];
                if entry == zero {
                    continue;
                }
                let q = int_div_floor(&entry, &pivot);
                for c in 0..4 {
                    let qm = q.wrapping_mul(&m[col][c]);
                    m[r][c] = m[r][c].wrapping_sub(&qm);
                }
            }
        }
    }
    m
}

/// Signed extended GCD with a non-zero first coefficient, the C reference
/// `ibz_xgcd_with_u_not_0` adapted to crypto-bigint's built-in xgcd. Returns
/// `(g, u, v)` with `u·x + v·y = g`, `g = gcd(|x|,|y|) ≥ 0`, and `u ≠ 0`.
///
/// Base Bezout pair comes from `Uint::xgcd().bezout_coefficients()` on the
/// magnitudes (then sign-flipped by `sign(x)`, `sign(y)`). The `u ≠ 0`
/// safeguard matches the C: when the base gives `u = 0` (which happens iff
/// `y | x`), set `u = 1` and shift `v` by `x/y` (exact) to preserve the
/// identity. The HNF kernel relies on `u ≠ 0` so the pivot column is never
/// annihilated; the exact `(u,v)` beyond that is immaterial because the HNF
/// output is canonical (any valid Bezout pair drives it to the same form).
pub fn xgcd_with_u_not_0<const LIMBS: usize>(
    x: &Int<LIMBS>,
    y: &Int<LIMBS>,
) -> (crypto_bigint::Uint<LIMBS>, Int<LIMBS>, Int<LIMBS>) {
    use crypto_bigint::Uint;
    let zero_i = Int::<LIMBS>::from_i64(0);
    let one_i = Int::<LIMBS>::from_i64(1);
    if *x == zero_i && *y == zero_i {
        return (Uint::<LIMBS>::ONE, one_i, zero_i);
    }
    let (x_abs, x_neg) = x.abs_sign();
    let (y_abs, y_neg) = y.abs_sign();
    let out = x_abs.xgcd(&y_abs);
    let g = out.gcd;
    let (su, sv) = out.bezout_coefficients(); // su·|x| + sv·|y| = g
    let mut u = if bool::from(x_neg) {
        su.wrapping_neg()
    } else {
        su
    };
    let mut v = if bool::from(y_neg) {
        sv.wrapping_neg()
    } else {
        sv
    };
    // u·x + v·y = g. Force u ≠ 0 (the base gives u = 0 only when y | x).
    if u == zero_i {
        if *x != zero_i {
            let q = int_div_floor(x, y); // exact: y | x ⇒ floor == exact
            v = v.wrapping_sub(&q);
        }
        u = one_i;
    }
    (g, u, v)
}

/// Integer floor-division: returns `⌊a / b⌋` for non-zero `b`, `0` when
/// `b == 0`.
///
/// Strategy: normalise both operands to unsigned via `Int::abs_sign`,
/// delegate to `crypto-bigint`'s `Uint::div_rem_vartime` (Knuth Algorithm D
/// under the hood — `O(LIMBS · 64)` work regardless of the quotient
/// magnitude), then apply the floor-vs-truncate adjustment when exactly one
/// operand was negative. The `crypto-bigint 0.7.x` *signed* division
/// surface is still unstable, so we route around it via the stable
/// `Uint` surface and reinterpret the result via `Uint::as_int`.
pub fn int_div_floor<const LIMBS: usize>(a: &Int<LIMBS>, b: &Int<LIMBS>) -> Int<LIMBS> {
    use crypto_bigint::NonZero;
    let zero = Int::<LIMBS>::from_i64(0);
    let one = Int::<LIMBS>::from_i64(1);
    if *b == zero {
        return zero;
    }
    let (a_abs, a_neg) = a.abs_sign();
    let (b_abs, b_neg) = b.abs_sign();
    let result_neg = bool::from(a_neg) ^ bool::from(b_neg);
    let Some(b_nz) = Option::<NonZero<_>>::from(NonZero::new(b_abs)) else {
        return zero;
    };
    let (q, r) = a_abs.div_rem_vartime(&b_nz);
    let q_int = *q.as_int();
    let zero_u = crypto_bigint::Uint::<LIMBS>::from_u64(0);
    if result_neg {
        if r == zero_u {
            q_int.wrapping_neg()
        } else {
            q_int.wrapping_neg().wrapping_sub(&one)
        }
    } else {
        q_int
    }
}

/// Plain Euclidean residue of `a` modulo `modulus > 0`, in `[0, modulus)`
/// (= C `ibz_mod` / GMP `mpz_mod`, non-negative; `0` stays `0`). Used by the
/// HNF kernel's pivot scalar-mul reduction.
pub fn euclid_mod<const LIMBS: usize>(
    a: &Int<LIMBS>,
    modulus: &crypto_bigint::Uint<LIMBS>,
) -> Int<LIMBS> {
    use crypto_bigint::{NonZero, Uint};
    let m_nz: NonZero<Uint<LIMBS>> =
        Option::from(NonZero::new(*modulus)).expect("euclid_mod: modulus must be > 0");
    let (a_abs, a_neg) = a.abs_sign();
    let rem = a_abs.rem_vartime(&m_nz);
    let zero = Uint::<LIMBS>::from_u64(0);
    let r = if bool::from(a_neg) && rem != zero {
        modulus.wrapping_sub(&rem)
    } else {
        rem
    };
    *r.as_int()
}

/// Port of the C reference `ibz_mat_4xn_hnf_mod_core` (`hnf/hnf.c`, Cohen
/// §2.4.8): the canonical mod-`modulus` column-style Hermite Normal Form of
/// the lattice spanned by `n` COLUMN generators (each a 4-vector). Returns
/// the 4×4 HNF basis (column-major). `modulus` must be a positive multiple
/// of the lattice covolume (the C passes `|det|` for a 4-generator lattice
/// or `gcd(det1, det2)` for an 8-generator union).
///
/// Faithful to the C: outer coordinate `i = 3 → 0`; inner combine each
/// nonzero `a[j][i]` into the pivot column `a[k]` via [`xgcd_with_u_not_0`]
/// (`c = u·a[k] + v·a[j]`; `a[j] = (a[k][i]/d)·a[j] − (a[j][i]/d)·a[k]`,
/// reduced by [`centered_mod`]; `a[k] = c` reduced by `centered_mod`); pivot
/// row `w[i] = u·a[k]` reduced by [`euclid_mod`], diagonal filled with `m`
/// if zero; off-diagonals (`h > i`) floor-reduced into `[0, w[i][i])`;
/// `m ← m/d`; transpose write-back `hnf[i][j] = w[j][i]`. HNF is canonical,
/// so the exact internal `(u,v)` does not affect the output.
#[allow(
    clippy::needless_range_loop,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)] // HNF-mod core: index loops use C's −1 sentinel (i/j/k ∈ [−1,7], n ≤ 8) so
// they cannot be plain iterators, and the i32/usize index casts mirror the C `int` arithmetic.
pub fn hnf_mod_core<const LIMBS: usize>(
    generators: &[[Int<LIMBS>; 4]],
    modulus: &crypto_bigint::Uint<LIMBS>,
) -> [[Int<LIMBS>; 4]; 4] {
    use crypto_bigint::{NonZero, Uint};
    let n = generators.len();
    debug_assert!(n >= 4, "hnf_mod_core needs at least 4 generators");
    debug_assert!(n <= 8, "hnf_mod_core supports up to 8 generators");
    let zero = Int::<LIMBS>::from_i64(0);

    let mut a = [[zero; 4]; 8];
    a[..n].copy_from_slice(&generators[..n]);
    let mut w = [[zero; 4]; 4];
    let mut m = *modulus;

    // Plain linear combination ca·va + cb·vb (no reduction).
    let lincomb = |ca: &Int<LIMBS>, va: &[Int<LIMBS>; 4], cb: &Int<LIMBS>, vb: &[Int<LIMBS>; 4]| {
        let mut out = [zero; 4];
        for t in 0..4 {
            out[t] = ca
                .wrapping_mul(&va[t])
                .wrapping_add(&cb.wrapping_mul(&vb[t]));
        }
        out
    };

    let mut i: i32 = 3;
    let mut k: i32 = n as i32 - 1;
    let mut j: i32 = n as i32 - 1;

    while i != -1 {
        let iu = i as usize;
        while j != 0 {
            j -= 1;
            let ju = j as usize;
            let ku = k as usize;
            if a[ju][iu] != zero {
                let (d, u, v) = xgcd_with_u_not_0::<LIMBS>(&a[ku][iu], &a[ju][iu]);
                let d_int = *d.as_int();
                let c = lincomb(&u, &a[ku], &v, &a[ju]);
                let coeff_1 = int_div_floor(&a[ku][iu], &d_int); // exact (d | a[k][i])
                let coeff_2 = int_div_floor(&a[ju][iu], &d_int).wrapping_neg();
                let newj_raw = lincomb(&coeff_1, &a[ju], &coeff_2, &a[ku]);
                let mut newj = [zero; 4];
                let mut newk = [zero; 4];
                for t in 0..4 {
                    newj[t] = centered_mod::<LIMBS>(&newj_raw[t], &m);
                    newk[t] = centered_mod::<LIMBS>(&c[t], &m);
                }
                a[ju] = newj;
                a[ku] = newk;
            }
        }
        let ku = k as usize;
        // Pivot extraction: xgcd against m, w[i] = u·a[k] mod m.
        let (d, u, _v) = xgcd_with_u_not_0::<LIMBS>(&a[ku][iu], m.as_int());
        for t in 0..4 {
            let prod = u.wrapping_mul(&a[ku][t]);
            w[iu][t] = euclid_mod::<LIMBS>(&prod, &m);
        }
        if w[iu][iu] == zero {
            w[iu][iu] = *m.as_int();
        }
        // Off-diagonal floor-reduction into [0, w[i][i]).
        for h in (iu + 1)..4 {
            let q = int_div_floor(&w[h][iu], &w[iu][iu]).wrapping_neg();
            for t in 0..4 {
                let qm = q.wrapping_mul(&w[iu][t]);
                w[h][t] = w[h][t].wrapping_add(&qm);
            }
        }
        // m ← m / d (exact).
        let d_nz: NonZero<Uint<LIMBS>> =
            Option::from(NonZero::new(d)).expect("hnf_mod_core: d > 0");
        let (mq, mr) = m.div_rem_vartime(&d_nz);
        debug_assert!(mr == Uint::<LIMBS>::from_u64(0), "m/d must be exact");
        m = mq;
        // Advance to the next coordinate/pivot column.
        k -= 1;
        i -= 1;
        j = k;
        if i != -1 {
            let ku2 = k as usize;
            let iu2 = i as usize;
            if a[ku2][iu2] == zero {
                a[ku2][iu2] = *m.as_int();
            }
        }
    }

    // Transpose write-back: hnf[i][j] = w[j][i].
    let mut hnf = [[zero; 4]; 4];
    for jj in 0..4 {
        for ii in 0..4 {
            hnf[ii][jj] = w[jj][ii];
        }
    }
    hnf
}

/// Port of the C reference `ibz_mod_not_zero` (`hnf/hnf_internal.c`): the
/// Euclidean residue of `a` modulo `modulus > 0`, but mapping a residue of
/// `0` to `modulus` itself. Result is in `[1, modulus]`.
pub fn mod_not_zero<const LIMBS: usize>(
    a: &Int<LIMBS>,
    modulus: &crypto_bigint::Uint<LIMBS>,
) -> crypto_bigint::Uint<LIMBS> {
    use crypto_bigint::{NonZero, Uint};
    let m_nz: NonZero<Uint<LIMBS>> =
        Option::from(NonZero::new(*modulus)).expect("mod_not_zero: modulus must be > 0");
    let (a_abs, a_neg) = a.abs_sign();
    let rem_abs = a_abs.rem_vartime(&m_nz); // |a| mod m ∈ [0, m)
    let zero = Uint::<LIMBS>::from_u64(0);
    // Euclidean residue (non-negative): for a < 0 with nonzero remainder,
    // the residue is m − (|a| mod m).
    let r = if bool::from(a_neg) && rem_abs != zero {
        modulus.wrapping_sub(&rem_abs)
    } else {
        rem_abs
    };
    if r == zero { *modulus } else { r }
}

/// Port of the C reference `ibz_centered_mod` (`hnf/hnf_internal.c`): reduce
/// `a` modulo `modulus > 0` into a (positively-biased) centered range. With
/// `tmp = mod_not_zero(a, modulus) ∈ [1, modulus]` and `d = floor(modulus/2)`,
/// the result is `tmp − modulus` when `tmp > d`, else `tmp` — i.e. the
/// residue in `(floor(modulus/2) − modulus, floor(modulus/2)]` ("rather
/// positive than negative"). Used by the column-HNF kernel's combine/copy
/// steps.
pub fn centered_mod<const LIMBS: usize>(
    a: &Int<LIMBS>,
    modulus: &crypto_bigint::Uint<LIMBS>,
) -> Int<LIMBS> {
    let tmp = mod_not_zero::<LIMBS>(a, modulus); // ∈ [1, modulus]
    let d = modulus.shr_vartime(1); // floor(modulus / 2)
    if tmp > d {
        // tmp − modulus (≤ 0); both fit Int since modulus < 2^(64·LIMBS−1).
        tmp.as_int().wrapping_sub(modulus.as_int())
    } else {
        *tmp.as_int()
    }
}

/// Port of the C reference `quat_lattice_reduce_denom` (`lattice.c`): put a
/// lattice's `(basis, denom)` representation in lowest terms by dividing both
/// by `g = gcd(all 16 basis entries, denom)`. The returned denominator is
/// non-negative (the C's `ibz_abs(&reduced->denom)`); basis signs are kept on
/// the basis (the C does NOT abs the basis). Used after every
/// [`quat_lattice_add`] and after the principal-ideal construction.
pub fn quat_lattice_reduce_denom<const LIMBS: usize>(
    basis: &[[Int<LIMBS>; 4]; 4],
    denom: &Int<LIMBS>,
) -> ([[Int<LIMBS>; 4]; 4], Int<LIMBS>) {
    use crate::quaternion::represent_integer::uint_gcd_vartime;
    use crypto_bigint::{NonZero, Uint};

    // g = gcd(|basis[0][0]|, …, |basis[3][3]|, |denom|), non-negative.
    let mut g: Uint<LIMBS> = basis[0][0].abs();
    for row in basis {
        for entry in row {
            g = uint_gcd_vartime(&g, &entry.abs());
        }
    }
    g = uint_gcd_vartime(&g, &denom.abs());

    let g_nz: NonZero<Uint<LIMBS>> = Option::from(NonZero::new(g))
        .expect("quat_lattice_reduce_denom: gcd(basis, denom) must be > 0");
    let g_int = *g.as_int(); // g divides each entry ⇒ top bit clear ⇒ safe reinterpret

    // basis / g (exact signed division; int_div_floor == exact when g | entry).
    let mut out = *basis;
    for row in &mut out {
        for entry in row {
            *entry = int_div_floor(entry, &g_int);
        }
    }
    // |denom| / g, non-negative (the C abs's the result denominator).
    let (dq, _r) = denom.abs().div_rem_vartime(&g_nz);
    let denom_out = *dq.as_int();
    (out, denom_out)
}

/// Port of the C reference `quat_lattice_add` (`lattice.c`): the union (sum)
/// of two rational lattices `L1 = (1/d1)·Z⟨B1⟩` and `L2 = (1/d2)·Z⟨B2⟩`.
///
/// Faithful to the C: form the 8 column generators `[d1·B2 | d2·B1]` (the C
/// scales `lat1.denom · lat2.basis` and `lat2.denom · lat1.basis`, then
/// transposes each into a generator vector), take the modulus
/// `gcd(det(d1·B2), det(d2·B1))` (a multiple of the 8-generator integer
/// lattice's covolume, as Cohen's HNF requires), drive it through
/// [`hnf_mod_core`], set `denom = d1·d2`, and [`quat_lattice_reduce_denom`].
///
/// Because [`hnf_mod_core`] is canonical and `reduce_denom` puts the result
/// in lowest terms, the returned `(basis, denom)` is the UNIQUE reduced
/// representation of the rational lattice `L1 + L2` — independent of how `L1`
/// and `L2` were individually represented. (Same canonicity gift the HNF
/// kernel gives: byte-exactness without a byte-exact input convention.)
///
/// **Width note**: `det(d1·B2)` and `det(d2·B1)` are computed at `LIMBS`
/// precision via [`det_4x4`](crate::quaternion::ideal::det_4x4) with wrapping
/// arithmetic; the caller must pick `LIMBS` wide enough that these
/// determinants (≈ `(d·‖B‖)^4`) do not overflow. At keygen scale (ideal norm
/// ≈ `2^512`) this needs ≈ 36 limbs.
// Parallel scale of two basis matrices and column-extraction (transpose) into
// 8 generators; index form mirrors the C reference's quat_lattice_add.
#[allow(clippy::needless_range_loop)]
pub fn quat_lattice_add<const LIMBS: usize>(
    basis1: &[[Int<LIMBS>; 4]; 4],
    denom1: &Int<LIMBS>,
    basis2: &[[Int<LIMBS>; 4]; 4],
    denom2: &Int<LIMBS>,
) -> ([[Int<LIMBS>; 4]; 4], Int<LIMBS>) {
    use crate::quaternion::ideal::det_4x4;
    use crate::quaternion::represent_integer::uint_gcd_vartime;
    let zero = Int::<LIMBS>::from_i64(0);

    // scaled1 = denom1 · basis2 ; scaled2 = denom2 · basis1.
    let mut scaled1 = [[zero; 4]; 4];
    let mut scaled2 = [[zero; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            scaled1[i][j] = denom1.wrapping_mul(&basis2[i][j]);
            scaled2[i][j] = denom2.wrapping_mul(&basis1[i][j]);
        }
    }

    // Generators are the COLUMNS (the C `generators[j][i] = tmp[i][j]`):
    // first 4 from scaled1, next 4 from scaled2.
    let mut generators = [[zero; 4]; 8];
    for j in 0..4 {
        generators[j] = [scaled1[0][j], scaled1[1][j], scaled1[2][j], scaled1[3][j]];
    }
    for j in 0..4 {
        generators[j + 4] = [scaled2[0][j], scaled2[1][j], scaled2[2][j], scaled2[3][j]];
    }

    let det1 = det_4x4::<LIMBS>(&scaled1);
    let det2 = det_4x4::<LIMBS>(&scaled2);
    let modulus = uint_gcd_vartime(&det1.abs(), &det2.abs());

    let hnf = hnf_mod_core::<LIMBS>(&generators[..], &modulus);
    let denom_prod = denom1.wrapping_mul(denom2);
    quat_lattice_reduce_denom::<LIMBS>(&hnf, &denom_prod)
}

#[cfg(test)]
// Test helpers build/compare fixed matrices by [r][c] index for direct
// correspondence with the math under test; iterator rewrites add no value here.
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    type I = Int<8>;

    fn n(x: i64) -> I {
        I::from_i64(x)
    }

    fn identity() -> [[I; 4]; 4] {
        [
            [n(1), n(0), n(0), n(0)],
            [n(0), n(1), n(0), n(0)],
            [n(0), n(0), n(1), n(0)],
            [n(0), n(0), n(0), n(1)],
        ]
    }

    #[test]
    fn hnf_of_identity_is_identity() {
        let h = hnf_4x4(&identity());
        assert_eq!(h, identity());
    }

    #[test]
    fn mod_not_zero_maps_zero_to_modulus() {
        use crypto_bigint::Uint;
        let m = Uint::<8>::from_u64(5);
        assert_eq!(mod_not_zero::<8>(&n(0), &m), Uint::<8>::from_u64(5));
        assert_eq!(mod_not_zero::<8>(&n(5), &m), Uint::<8>::from_u64(5));
        assert_eq!(mod_not_zero::<8>(&n(6), &m), Uint::<8>::from_u64(1));
        assert_eq!(mod_not_zero::<8>(&n(7), &m), Uint::<8>::from_u64(2));
        assert_eq!(mod_not_zero::<8>(&n(-1), &m), Uint::<8>::from_u64(4));
        assert_eq!(mod_not_zero::<8>(&n(-5), &m), Uint::<8>::from_u64(5));
    }

    /// crypto-bigint's built-in `Uint::xgcd().bezout_coefficients()` gives a
    /// valid Bezout pair `u·x + v·y = gcd` — the HNF kernel uses this
    /// directly (per the project's crypto-bigint-only direction). The
    /// specific `(u,v)` differs from a Euclidean choice, but the HNF output
    /// is CANONICAL (unique reduced form), so `(u,v)` is internal-only and
    /// the final HNF is unaffected. Sanity-check the gcd + Bezout identity.
    #[test]
    fn crypto_bigint_xgcd_satisfies_bezout() {
        use crypto_bigint::Uint;
        let u = |x: u64| Uint::<8>::from_u64(x);
        for (x, y, g) in [(6u64, 4u64, 2u64), (15, 6, 3), (3, 5, 1), (12, 18, 6)] {
            let out = u(x).xgcd(&u(y));
            assert_eq!(out.gcd, u(g), "gcd({x},{y})");
            let (bu, bv) = out.bezout_coefficients();
            let lhs = bu
                .wrapping_mul(&n(i64::try_from(x).expect("x fits in i64")))
                .wrapping_add(&bv.wrapping_mul(&n(i64::try_from(y).expect("y fits in i64"))));
            assert_eq!(
                lhs,
                n(i64::try_from(g).expect("g fits in i64")),
                "Bezout ({x},{y})"
            );
        }
    }

    #[test]
    fn hnf_mod_core_preserves_lattice_det() {
        use crate::quaternion::ideal::det_4x4;
        use crypto_bigint::Uint;
        // 4 column generators (g_c = column c).
        let gens = [
            [n(2), n(0), n(0), n(0)],
            [n(1), n(3), n(0), n(0)],
            [n(0), n(1), n(5), n(0)],
            [n(0), n(0), n(2), n(7)],
        ];
        // Matrix whose columns are the generators; det = 2·3·5·7 = 210.
        let mut mat = [[n(0); 4]; 4];
        for col in 0..4 {
            for row in 0..4 {
                mat[row][col] = gens[col][row];
            }
        }
        let modd = det_4x4::<8>(&mat).abs();
        assert_eq!(modd, Uint::<8>::from_u64(210));

        let hnf = hnf_mod_core::<8>(&gens, &modd);
        // Lattice preserved: |det(HNF)| == covolume == mod.
        assert_eq!(
            det_4x4::<8>(&hnf).abs(),
            modd,
            "HNF must preserve the lattice covolume",
        );
        // Canonical form: strictly-positive diagonal.
        for i in 0..4 {
            assert!(
                !bool::from(hnf[i][i].is_negative()) && hnf[i][i] != n(0),
                "HNF diagonal entry {i} must be positive",
            );
        }
    }

    #[test]
    fn xgcd_with_u_not_0_bezout_and_nonzero_u() {
        use crypto_bigint::Uint;
        // (x, y) signed; assert u·x + v·y = g, g = gcd, and u ≠ 0.
        let cases: [(i64, i64, u64); 10] = [
            (6, 4, 2),
            (4, 6, 2),
            (15, 6, 3),
            (-6, 4, 2),
            (6, -4, 2),
            (-6, -4, 2),
            (8, 4, 4), // y | x ⇒ base u = 0, safeguard fires
            (4, 8, 4),
            (0, 5, 5), // x = 0
            (5, 0, 5), // y = 0
        ];
        for (x, y, g) in cases {
            let (gg, u, v) = xgcd_with_u_not_0::<8>(&n(x), &n(y));
            assert_eq!(gg, Uint::<8>::from_u64(g), "gcd({x},{y})");
            assert_ne!(u, n(0), "u must be non-zero for ({x},{y})");
            let lhs = u.wrapping_mul(&n(x)).wrapping_add(&v.wrapping_mul(&n(y)));
            assert_eq!(
                lhs,
                n(i64::try_from(g).expect("g fits in i64")),
                "u·x + v·y = g for ({x},{y})"
            );
        }
    }

    #[test]
    fn centered_mod_biases_positive() {
        use crypto_bigint::Uint;
        let m = Uint::<8>::from_u64(5); // floor(5/2) = 2
        assert_eq!(centered_mod::<8>(&n(7), &m), n(2)); // 7 mod 5 = 2 ≤ 2
        assert_eq!(centered_mod::<8>(&n(3), &m), n(-2)); // 3 > 2 ⇒ 3 − 5
        assert_eq!(centered_mod::<8>(&n(5), &m), n(0)); // 0 → 5 > 2 ⇒ 5 − 5
        assert_eq!(centered_mod::<8>(&n(-1), &m), n(-1)); // 4 > 2 ⇒ 4 − 5
        assert_eq!(centered_mod::<8>(&n(-3), &m), n(2)); // −3 mod 5 = 2 ≤ 2
    }

    #[test]
    fn hnf_of_diagonal_is_diagonal() {
        let m = [
            [n(2), n(0), n(0), n(0)],
            [n(0), n(3), n(0), n(0)],
            [n(0), n(0), n(5), n(0)],
            [n(0), n(0), n(0), n(7)],
        ];
        let h = hnf_4x4(&m);
        assert_eq!(h, m);
    }

    #[test]
    fn hnf_pivots_are_positive() {
        let m = [
            [n(-2), n(1), n(0), n(0)],
            [n(0), n(-3), n(0), n(0)],
            [n(0), n(0), n(5), n(0)],
            [n(0), n(0), n(0), n(-7)],
        ];
        let h = hnf_4x4(&m);
        for i in 0..4 {
            assert!(
                !bool::from(h[i][i].is_negative()),
                "diagonal[{i}] should be non-negative"
            );
        }
    }

    #[test]
    fn hnf_lower_triangle_is_zero() {
        // Swap a couple of rows in identity; HNF must restore upper-triangular.
        let mut m = identity();
        m.swap(0, 2);
        m.swap(1, 3);
        let h = hnf_4x4(&m);
        for r in 0..4 {
            for c in 0..r {
                assert_eq!(h[r][c], n(0), "lower triangle nonzero at ({r}, {c})");
            }
        }
    }

    #[test]
    fn int_div_floor_basic() {
        assert_eq!(int_div_floor(&n(7), &n(2)), n(3));
        assert_eq!(int_div_floor(&n(-7), &n(2)), n(-4)); // floor, not trunc
        assert_eq!(int_div_floor(&n(7), &n(-2)), n(-4));
        assert_eq!(int_div_floor(&n(-7), &n(-2)), n(3));
        assert_eq!(int_div_floor(&n(0), &n(5)), n(0));
        assert_eq!(int_div_floor(&n(5), &n(0)), n(0));
        assert_eq!(int_div_floor(&n(10), &n(5)), n(2));
        assert_eq!(int_div_floor(&n(-10), &n(5)), n(-2));
    }

    #[test]
    fn int_div_floor_handles_real_prime_scale_quotient() {
        // 2^200 / 2^100 = 2^100. The pre-Session-35 repeated-subtraction body
        // would have needed 2^100 iterations to finish (≈ heat-death of the
        // universe). The Knuth-Algorithm-D-backed body returns in microseconds.
        let num_u = crypto_bigint::Uint::<8>::ONE.shl_vartime(200);
        let den_u = crypto_bigint::Uint::<8>::ONE.shl_vartime(100);
        let expected_u = crypto_bigint::Uint::<8>::ONE.shl_vartime(100);
        let num: Int<8> = *num_u.as_int();
        let den: Int<8> = *den_u.as_int();
        let expected: Int<8> = *expected_u.as_int();
        assert_eq!(int_div_floor(&num, &den), expected);
    }

    #[test]
    fn int_div_floor_large_negative_quotient() {
        // -(2^200) / 2^100 = -(2^100). Exact division — floor adjustment does
        // not subtract one because the remainder is zero.
        let num_u = crypto_bigint::Uint::<8>::ONE.shl_vartime(200);
        let den_u = crypto_bigint::Uint::<8>::ONE.shl_vartime(100);
        let expected_u = crypto_bigint::Uint::<8>::ONE.shl_vartime(100);
        let num: Int<8> = (*num_u.as_int()).wrapping_neg();
        let den: Int<8> = *den_u.as_int();
        let expected: Int<8> = (*expected_u.as_int()).wrapping_neg();
        assert_eq!(int_div_floor(&num, &den), expected);
    }

    #[test]
    fn int_div_floor_large_negative_quotient_with_remainder() {
        // -(2^200 + 1) / 2^100 = -(2^100) − 1   (floor of −(2^100 + 2^(−100)))
        let num_u = crypto_bigint::Uint::<8>::ONE
            .shl_vartime(200)
            .wrapping_add(&crypto_bigint::Uint::<8>::ONE);
        let den_u = crypto_bigint::Uint::<8>::ONE.shl_vartime(100);
        let num: Int<8> = (*num_u.as_int()).wrapping_neg();
        let den: Int<8> = *den_u.as_int();
        let expected: Int<8> = (*crypto_bigint::Uint::<8>::ONE.shl_vartime(100).as_int())
            .wrapping_neg()
            .wrapping_sub(&Int::<8>::from_i64(1));
        assert_eq!(int_div_floor(&num, &den), expected);
    }

    fn diag(d: i64) -> [[I; 4]; 4] {
        let mut m = [[n(0); 4]; 4];
        for i in 0..4 {
            m[i][i] = n(d);
        }
        m
    }

    /// reduce_denom puts `(basis, denom)` in lowest terms and makes the
    /// denominator non-negative. `(2·I, 6)` ⇒ gcd(2,6)=2 ⇒ `(I, 3)`; a
    /// negative denom is abs'd while basis signs stay.
    #[test]
    fn reduce_denom_lowest_terms_and_positive_denom() {
        let (b, d) = quat_lattice_reduce_denom::<8>(&diag(2), &n(6));
        assert_eq!(b, identity());
        assert_eq!(d, n(3));

        // Negative denom: abs'd; basis sign preserved.
        let mut signed = diag(4);
        signed[0][1] = n(-8);
        let (b2, d2) = quat_lattice_reduce_denom::<8>(&signed, &n(-12));
        // gcd(4,8,12)=4 ⇒ basis/4, denom |−12|/4 = 3.
        assert_eq!(d2, n(3));
        assert_eq!(b2[0][0], n(1));
        assert_eq!(b2[0][1], n(-2));
    }

    /// `quat_lattice_add` — three hand-computed rational-lattice unions, each an
    /// INDEPENDENT check (the expected sum lattice is derived by hand, not from
    /// the implementation's formula).
    ///
    /// A: `2·Z⁴ + 3·Z⁴ = Z⁴` (gcd(2,3)=1). B: `Z⁴ + (1/2)·Z⁴ = (1/2)·Z⁴`.
    /// C: `6·Z⁴ + 4·Z⁴ = 2·Z⁴` (gcd(6,4)=2).
    #[test]
    fn lattice_add_hand_computed_unions() {
        let one = n(1);
        // A: (2·I, 1) + (3·I, 1) = (I, 1).
        let (ba, da) = quat_lattice_add::<8>(&diag(2), &one, &diag(3), &one);
        assert_eq!(ba, identity(), "2Z⁴ + 3Z⁴ = Z⁴");
        assert_eq!(da, n(1));

        // B: (I, 1) + (I, 2) = (I, 2)  [(1/2)Z⁴].
        let (bb, db) = quat_lattice_add::<8>(&identity(), &n(1), &identity(), &n(2));
        assert_eq!(bb, identity(), "Z⁴ + (1/2)Z⁴ = (1/2)Z⁴");
        assert_eq!(db, n(2));

        // C: (6·I, 1) + (4·I, 1) = (2·I, 1).
        let (bc, dc) = quat_lattice_add::<8>(&diag(6), &one, &diag(4), &one);
        assert_eq!(bc, diag(2), "6Z⁴ + 4Z⁴ = 2Z⁴");
        assert_eq!(dc, n(1));
    }
}
