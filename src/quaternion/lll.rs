// SPDX-License-Identifier: MIT OR Apache-2.0
//! `quat_lll_core` — a byte-faithful pure-Rust port of the SQIsign C
//! reference's 4×4 L² lattice reducer
//! (`src/quaternion/ref/generic/lll/l2.c`).
//!
//! # Why byte-faithful
//!
//! LLL output is NOT canonical — the reduced basis depends on the exact
//! sequence of float comparisons and swaps. The SQIsign keygen secret key
//! is the first prime-norm element found in a box-search over this reduced
//! basis, so reproducing the official keygen KAT requires reproducing the
//! C reference's reduced basis bit-for-bit. This port keeps the Gram /
//! basis transform in exact integers (`Int<N>`, mirroring the C's GMP
//! `ibz_t`) and runs the Gram-Schmidt / Lovász numerics in [`Dpe`] (the
//! byte-faithful `dpe_t` port), preserving the C operation order exactly.
//!
//! # Validation
//!
//! The C reference's own LLL tests assert only STRUCTURAL properties
//! (size-reduction + Lovász via an independent Gram-Schmidt verifier, plus
//! lattice-equality) — they contain no golden input→output vector pairs,
//! precisely because LLL output is non-canonical. So the tests here mirror
//! that methodology: an independent f64 Gram-Schmidt verifier checks the
//! output is LLL-reduced, and `det_4x4` confirms the lattice is preserved
//! (unimodular transform ⇒ `|det|` invariant). True byte-exactness is
//! gated by the end-to-end keygen KAT once the reducer is wired into
//! `quat_lideal_prime_norm_reduced_equivalent`.
//!
//! # Width contract
//!
//! `N` is the exact-integer width of the Gram and basis. The C uses
//! arbitrary precision (GMP); here the caller must size `N` so the Gram
//! entries and the `X · G` / `X · b` products during size reduction do not
//! overflow `Int<N>`. Toy lattices fit `Int<8>`; a SEC_DEGREE ideal's Gram
//! (~2^1278) needs a much wider `N`.

use crate::quaternion::dpe::Dpe;
use crypto_bigint::{Int, NonZero, Uint};

/// `DELTABAR` — the Lovász δ̄ threshold (`lll_internals.h`).
const DELTABAR: f64 = 0.995;
/// `ETABAR` — the size-reduction η̄ threshold (`lll_internals.h`).
const ETABAR: f64 = 0.505;

/// Lower-triangular accessor: the C `SYM(M,i,j)` macro returns the
/// lower-triangle slot — `M[j][i]` if `i < j`, else `M[i][j]`. The Gram
/// matrix is maintained as lower-triangular during the reduction (the
/// upper half is filled only at the end).
#[inline]
fn sym(i: usize, j: usize) -> (usize, usize) {
    if i < j { (j, i) } else { (i, j) }
}

/// Port of `quat_lll_core`: in-place L² reduction of the lattice `basis`
/// with Gram matrix `g`, both 4×4 over `Int<N>`. On return `basis` is
/// LLL-reduced (δ=0.995, η=0.505) and `g` is its (full, symmetric) Gram
/// matrix. `g` must enter with a correct LOWER triangle (the algorithm
/// reads/maintains the lower half and fills the upper half at the end).
#[allow(dead_code)] // consumed by the wide-input ideal reduction (next session).
#[allow(clippy::needless_range_loop)] // index-based loops mirror the C operation order.
pub fn quat_lll_core<const N: usize>(g: &mut [[Int<N>; 4]; 4], basis: &mut [[Int<N>; 4]; 4]) {
    let delta_bar = Dpe::from_f64(DELTABAR);
    let zero = Dpe::from_i64(0);
    // Float Gram-Schmidt state: r[i][j] ≈ <b_i, b_j*>, u[i][j] ≈ μ_{i,j}.
    let mut r = [[zero; 4]; 4];
    let mut u = [[zero; 4]; 4];
    let mut lovasz = [zero; 4];

    r[0][0] = Dpe::from_int::<N>(&g[0][0]);
    let mut kappa = 1usize;
    while kappa < 4 {
        // Size-reduce b_κ, recomputing the κ-th Gram-Schmidt row each pass.
        let mut done = false;
        while !done {
            for j in 0..=kappa {
                r[kappa][j] = Dpe::from_int::<N>(&g[kappa][j]);
                for k in 0..j {
                    let tmp = r[kappa][k].mul(&u[j][k]);
                    r[kappa][j] = r[kappa][j].sub(&tmp);
                }
                if j < kappa {
                    u[kappa][j] = r[kappa][j].div(&r[j][j]);
                }
            }

            done = true;
            for i in (0..kappa).rev() {
                if u[kappa][i].cmp_f64(ETABAR) > 0 || u[kappa][i].cmp_f64(-ETABAR) < 0 {
                    done = false;
                    let xf = u[kappa][i].round();
                    let x: Int<N> = xf.to_int::<N>();

                    // b_κ ← b_κ − X·b_i (all four coordinate rows).
                    for row in basis.iter_mut() {
                        let prod = x.wrapping_mul(&row[i]);
                        row[kappa] = row[kappa].wrapping_sub(&prod);
                    }

                    // <b_κ,b_κ> ← <b_κ,b_κ> − X·<b_κ,b_i> (read before the loop).
                    let prod = x.wrapping_mul(&g[kappa][i]);
                    g[kappa][kappa] = g[kappa][kappa].wrapping_sub(&prod);

                    // <b_κ,b_j> ← <b_κ,b_j> − X·<b_i,b_j> for all j. Sequential
                    // (j ascending) with LIVE reads: the j=i update to (κ,i) is
                    // intentionally seen by the j=κ read, producing the two-part
                    // diagonal formula the C comment documents.
                    for j in 0..4 {
                        let (ri, ci) = sym(i, j);
                        let prod = x.wrapping_mul(&g[ri][ci]);
                        let (rk, ck) = sym(kappa, j);
                        g[rk][ck] = g[rk][ck].wrapping_sub(&prod);
                    }

                    // Float μ-row update: u[κ][j] ← u[κ][j] − Xf·u[i][j], j<i.
                    for j in 0..i {
                        let tmp = xf.mul(&u[i][j]);
                        u[kappa][j] = u[kappa][j].sub(&tmp);
                    }
                }
            }
        }

        // Lovász: lovasz[0]=‖b_κ‖²; lovasz[i]=lovasz[i−1]−u[κ][i−1]·r[κ][i−1].
        lovasz[0] = Dpe::from_int::<N>(&g[kappa][kappa]);
        for i in 1..kappa {
            let tmp = u[kappa][i - 1].mul(&r[kappa][i - 1]);
            lovasz[i] = lovasz[i - 1].sub(&tmp);
        }
        // Find the insertion point: first swap (descending) where
        // δ̄·r[swap−1][swap−1] < lovasz[swap−1].
        let mut swap = kappa;
        while swap > 0 {
            let tmp = delta_bar.mul(&r[swap - 1][swap - 1]);
            if tmp.compare(&lovasz[swap - 1]) < 0 {
                break;
            }
            swap -= 1;
        }

        if kappa != swap {
            // Insert b_κ before b_swap: rotate columns swap..=κ of basis and
            // the lower-triangular Gram entries.
            let mut j = kappa;
            while j > swap {
                for i in 0..4 {
                    basis[i].swap(j, j - 1);
                    if i == j - 1 {
                        let t = g[i][i];
                        g[i][i] = g[j][j];
                        g[j][j] = t;
                    } else if i != j {
                        let (a, b) = sym(i, j);
                        let (c, d) = sym(i, j - 1);
                        let t = g[a][b];
                        g[a][b] = g[c][d];
                        g[c][d] = t;
                    }
                }
                j -= 1;
            }
            // Copy the κ float rows into the swap position.
            for i in 0..swap {
                u[swap][i] = u[kappa][i];
                r[swap][i] = r[kappa][i];
            }
            r[swap][swap] = lovasz[swap];
            kappa = swap;
        }

        kappa += 1;
    }

    // Fill the upper half of the (symmetric) Gram matrix.
    for i in 0..4 {
        for j in (i + 1)..4 {
            g[i][j] = g[j][i];
        }
    }
}

/// Port of `quat_lattice_gram` (`lattice.c`): the reduced-norm Gram of a
/// COLUMN-major lattice basis, `G[i][j] = 2·Σ_k w_k·b[k][i]·b[k][j]` with
/// weights `w = (1, 1, p, p)` (the form `b(u,v) = u0v0 + u1v1 + p(u2v2 +
/// u3v3)` on `(1,i,j,ij)` coords with `i²=−1, j²=−p`). The `×2` is the
/// trace pairing `2·Re(conj(x)·y)`; `lideal_reduce_basis` halves the
/// diagonal back out at the end. Full symmetric output.
#[allow(dead_code, clippy::needless_range_loop)]
pub fn lattice_gram<const N: usize>(basis: &[[Int<N>; 4]; 4], p: &Uint<N>) -> [[Int<N>; 4]; 4] {
    let p_int = *p.as_int();
    let two = Int::<N>::from_i64(2);
    let zero = Int::<N>::from_i64(0);
    let mut g = [[zero; 4]; 4];
    for i in 0..4 {
        for j in 0..=i {
            let mut acc = zero;
            for k in 0..4 {
                let mut t = basis[k][i].wrapping_mul(&basis[k][j]);
                if k >= 2 {
                    t = t.wrapping_mul(&p_int);
                }
                acc = acc.wrapping_add(&t);
            }
            g[i][j] = acc.wrapping_mul(&two);
        }
    }
    for i in 0..4 {
        for j in (i + 1)..4 {
            g[i][j] = g[j][i];
        }
    }
    g
}

/// Exact signed division `a / d` (`d > 0`). Returns `None` if the division
/// is not exact (mirrors the C `ibz_div` + `assert(remainder == 0)`).
fn int_div_exact_signed<const N: usize>(a: &Int<N>, d: &NonZero<Uint<N>>) -> Option<Int<N>> {
    let neg = bool::from(a.is_negative());
    let (q, rem) = a.abs().div_rem_vartime(d);
    if rem != Uint::<N>::from_u64(0) {
        return None;
    }
    let qi = *q.as_int();
    Some(if neg { qi.wrapping_neg() } else { qi })
}

/// Port of `quat_lideal_reduce_basis` (`lll/lll_applications.c`) with
/// `quat_lideal_class_gram` inlined: LLL-reduce a left ideal's COLUMN-major
/// `basis` (denominator `denom`, reduced norm `norm`, algebra prime `p`).
///
/// Flow, byte-faithful to the C:
/// 1. `class_gram = lattice_gram / (denom²·norm)` (exact division — the C
///    asserts the remainder is zero, which holds for a genuine ideal).
/// 2. copy the basis, run [`quat_lll_core`] under the class Gram.
/// 3. rescale the reduced Gram by `denom²`, halve the diagonal (removing the
///    `×2` from `lattice_gram`), zero the strict upper triangle.
///
/// Returns `(reduced_basis, reduced_gram)`, or `None` if the class-Gram
/// division is not exact (a non-ideal input). The reduced basis is what the
/// prime-norm box-search enumerates; feeding `quat_lll_core` the *divided*
/// class Gram (not `lattice_gram`) is required for byte-exactness, since
/// dpe rounding is not invariant under the non-power-of-two divisor.
#[allow(dead_code, clippy::needless_range_loop, clippy::type_complexity)]
pub fn lideal_reduce_basis<const N: usize>(
    basis: &[[Int<N>; 4]; 4],
    denom: &Uint<N>,
    norm: &Uint<N>,
    p: &Uint<N>,
) -> Option<([[Int<N>; 4]; 4], [[Int<N>; 4]; 4])> {
    let denom_int = *denom.as_int();
    let corrector = denom_int.wrapping_mul(&denom_int); // denom²

    // class_gram = lattice_gram / (denom²·norm), exact (lower triangle).
    let mut gram = lattice_gram::<N>(basis, p);
    let divisor_u = denom.wrapping_mul(denom).wrapping_mul(norm); // denom²·norm > 0
    let div_nz = Option::<NonZero<Uint<N>>>::from(NonZero::new(divisor_u))?;
    for i in 0..4 {
        for j in 0..=i {
            gram[i][j] = int_div_exact_signed::<N>(&gram[i][j], &div_nz)?;
        }
    }
    for i in 0..4 {
        for j in 0..i {
            gram[j][i] = gram[i][j];
        }
    }

    // Reduce (seeded by a copy of the lattice basis).
    let mut reduced = *basis;
    quat_lll_core::<N>(&mut gram, &mut reduced);

    // Rescale the reduced class Gram by denom², halve the diagonal (tdiv by
    // 2; the diagonal is strictly positive), zero the strict upper triangle.
    for i in 0..4 {
        for j in 0..4 {
            gram[i][j] = gram[i][j].wrapping_mul(&corrector);
        }
    }
    for i in 0..4 {
        let halved = gram[i][i].abs().shr_vartime(1);
        gram[i][i] = *halved.as_int();
        for j in (i + 1)..4 {
            gram[i][j] = Int::<N>::from_i64(0);
        }
    }

    Some((reduced, gram))
}

/// Port of the C `ibz_mat_4x4_eval` (`dim4.c`): row-major matrix·vector,
/// `out[i] = Σ_j mat[i][j]·vec[j]`. Computes into a fresh array, so the C
/// in-place call `ibz_mat_4x4_eval(&coord, &red, &coord)` is reproduced by
/// passing the same slice as `vec` (the result is a new array).
#[allow(dead_code, clippy::needless_range_loop)]
pub fn mat4_eval<const N: usize>(mat: &[[Int<N>; 4]; 4], vec: &[Int<N>; 4]) -> [Int<N>; 4] {
    let mut out = [Int::<N>::from_i64(0); 4];
    for i in 0..4 {
        let mut sum = Int::<N>::from_i64(0);
        for j in 0..4 {
            sum = sum.wrapping_add(&mat[i][j].wrapping_mul(&vec[j]));
        }
        out[i] = sum;
    }
    out
}

/// The prime-norm BOX-SEARCH from `quat_lideal_prime_norm_reduced_equivalent`
/// (`lll/lll_applications.c`), byte-faithful: over a reduced basis + its
/// reduce-basis Gram, sample coordinate vectors in `[−m, m]^4` (one
/// `ibz_rand_interval_minm_m` per coord, index order 0..3) until
/// `qf_eval(gram, coord) / denom²` is prime, then return that element
/// `α = reduced · coord` (in the reduced basis's element coords) and the
/// prime `q`. Bounded by `(2m+1)^4` attempts (the C `equiv_num_iter`).
///
/// This is the search core; building the prime-norm equivalent ideal `J`
/// (conjugate `α`, right-multiply `I` by `ᾱ/(denom·N(I))`) and the
/// O_0↔standard / row↔column coordinate conversions between our `LeftIdeal`
/// representation and the C's are the wiring step (next session).
///
/// Gated on `kat` because the byte-exact interval RNG it consumes lives in
/// the `kat`-gated `rng` module; it relocates to the production build when
/// keygen is wired.
#[cfg(feature = "kat")]
#[allow(dead_code, clippy::type_complexity)]
pub fn prime_norm_box_search<const N: usize, R: rand_core::CryptoRng + ?Sized>(
    reduced: &[[Int<N>; 4]; 4],
    gram: &[[Int<N>; 4]; 4],
    denom: &Uint<N>,
    equiv_bound_coeff: u32,
    primality_witnesses: &[Uint<N>],
    rng: &mut R,
) -> Option<([Int<N>; 4], Uint<N>)> {
    use crate::quaternion::lattice::qf_eval_4x4;
    use crate::quaternion::primality::is_probable_prime_with_witnesses;
    use crate::rng::ibz_rand_interval_minm_m;

    let adjusted = denom.wrapping_mul(denom); // adjusted_norm = denom²
    let adj_nz = Option::<NonZero<Uint<N>>>::from(NonZero::new(adjusted))?;
    let n = u64::from(2 * equiv_bound_coeff + 1);
    let max_iter = n.saturating_mul(n).saturating_mul(n).saturating_mul(n); // (2m+1)^4

    let mut ctr = 0u64;
    while ctr < max_iter {
        ctr += 1;
        let coord = [
            ibz_rand_interval_minm_m::<N, R>(rng, equiv_bound_coeff),
            ibz_rand_interval_minm_m::<N, R>(rng, equiv_bound_coeff),
            ibz_rand_interval_minm_m::<N, R>(rng, equiv_bound_coeff),
            ibz_rand_interval_minm_m::<N, R>(rng, equiv_bound_coeff),
        ];
        // tmp = coordᵀ·gram·coord (naive full-matrix, matching quat_qf_eval).
        let tmp = qf_eval_4x4::<N>(&coord, gram);
        // q = tmp / denom² (the C asserts exact division).
        let (q, rem) = tmp.abs().div_rem_vartime(&adj_nz);
        if rem != Uint::<N>::ZERO {
            continue;
        }
        if is_probable_prime_with_witnesses::<N>(&q, primality_witnesses) {
            let alpha = mat4_eval::<N>(reduced, &coord);
            return Some((alpha, q));
        }
    }
    None
}

/// Port of the C `quat_lideal_prime_norm_reduced_equivalent`
/// (`lll/lll_applications.c`): given a left ideal `I` in the C representation
/// (COLUMN-major standard-coords `basis`, scalar `denom`, reduced norm
/// `norm`), produce an EQUIVALENT left ideal `J = I·(ᾱ / (denom·norm))` of
/// PRIME reduced norm `q`. This is the wiring step that the module-level doc
/// flagged as the final gate on byte-exactness.
///
/// Sequence (verbatim C):
/// 1. [`lideal_reduce_basis`] → `(reduced, gram)` — the dpe-LLL reduced basis
///    and its class Gram (scaled by `denom²`, halved diagonal, upper-zeroed)
///    so `qf_eval(gram, v)/denom² = N(Σ vᵣ·reducedᵣ)/N(I)`.
/// 2. [`prime_norm_box_search`] → `(α, q)` — sample `v ∈ [−m, m]⁴` (byte-exact
///    `ibz_rand_interval_minm_m`, coords 0→3) until `q = qf_eval/denom²` is
///    prime; `α = reduced · v` in standard coords (the C `α.denom = denom`).
/// 3. Conjugate `ᾱ = (α₀, −α₁, −α₂, −α₃)`; the C then sets `α.denom = denom·norm`.
/// 4. Right-multiply: each `J` column = `Iᵢ_column · ᾱ` (the C
///    `quat_lattice_alg_elem_mul`; [`Quaternion::mul`] == `quat_alg_coord_mul`),
///    `J.denom = I.denom · α.denom = denom²·norm`, then canonical
///    [`hnf_mod_core`](crate::quaternion::hnf::hnf_mod_core) (mod `|det|`) +
///    [`quat_lattice_reduce_denom`](crate::quaternion::hnf::quat_lattice_reduce_denom).
/// 5. `J.norm = q` (the C computes `I.norm·N(α)/N(α).denom` and asserts it
///    equals the prime `q`; we take `q` directly from the search).
///
/// Returns `(J_basis, J_denom, q)`; `None` if the box search exhausts its
/// `(2m+1)⁴` budget. As with every step downstream of a canonical HNF,
/// `J`'s `(basis, denom)` is the unique reduced representation of the
/// rational lattice `I·ᾱ/c` — so the construction is byte-exact given the
/// (non-canonical, dpe-faithful) reduced basis the search consumed; true
/// byte-exactness is finally certified by the end-to-end keygen KAT.
///
/// `kat`-gated because the box search consumes the byte-exact interval RNG.
#[cfg(feature = "kat")]
#[allow(
    dead_code,
    clippy::type_complexity,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]
pub fn quat_lideal_prime_norm_reduced_equivalent<
    const N: usize,
    R: rand_core::CryptoRng + ?Sized,
>(
    basis: &[[Int<N>; 4]; 4],
    denom: &Int<N>,
    norm: &Uint<N>,
    p: &Uint<N>,
    equiv_bound_coeff: u32,
    primality_witnesses: &[Uint<N>],
    rng: &mut R,
) -> Option<([[Int<N>; 4]; 4], Int<N>, Uint<N>)> {
    use crate::quaternion::Quaternion;
    use crate::quaternion::hnf::{hnf_mod_core, quat_lattice_reduce_denom};
    use crate::quaternion::ideal::det_4x4;
    use crate::quaternion::o0_mul::uint_as_nonneg_int;

    let denom_u = denom.abs();
    let (reduced, gram) = lideal_reduce_basis::<N>(basis, &denom_u, norm, p)?;
    // KEYGEN BYTE-EXACT (C-oracle bisect): for the KAT[0] secret ideal,
    // our `reduced` shares the real- and j-coordinate entries with C's `red`
    // (e.g. red[0][0]=0x1d8003b4f6a19d62…, red[2][0]=0x1347be71…) but DIFFERS in
    // the i- and k-coordinates (a signed permutation). C `quat_lideal_reduce_basis`
    // (dpe-float `quat_lll_core`) emits a specific reduced basis; ours is a
    // sign/perm variant → a unit-equivalent α → unit-rotated secret ideal →
    // same j(E_A) but a different Montgomery MODEL (the item-8 pk mismatch).
    // FIX = match C's LLL reduced-basis sign/ordering convention exactly.
    let (alpha, q) = prime_norm_box_search::<N, R>(
        &reduced,
        &gram,
        &denom_u,
        equiv_bound_coeff,
        primality_witnesses,
        rng,
    )?;

    // ᾱ = conjugate(α) in standard (1, i, j, ij) coords.
    let alpha_bar = Quaternion::<N>::new(
        alpha[0],
        alpha[1].wrapping_neg(),
        alpha[2].wrapping_neg(),
        alpha[3].wrapping_neg(),
    );

    // Right-multiply I by ᾱ: column_j · ᾱ (== the C quat_alg_coord_mul).
    let mut prod = [[Int::<N>::from_i64(0); 4]; 4];
    for j in 0..4 {
        let col = Quaternion::<N>::new(basis[0][j], basis[1][j], basis[2][j], basis[3][j]);
        let pr = col.mul(&alpha_bar, p);
        prod[0][j] = pr.a;
        prod[1][j] = pr.b;
        prod[2][j] = pr.c;
        prod[3][j] = pr.d;
    }

    // J.denom = I.denom · α.denom = denom · (denom·norm) = denom²·norm.
    let norm_int = uint_as_nonneg_int::<N>(norm)?;
    let alpha_denom = denom.wrapping_mul(&norm_int);
    let prod_denom = denom.wrapping_mul(&alpha_denom);

    // Canonical HNF (mod = |det|, the 4-column lattice covolume) + reduce.
    let modulus = det_4x4::<N>(&prod).abs();
    let gens: [[Int<N>; 4]; 4] = [
        [prod[0][0], prod[1][0], prod[2][0], prod[3][0]],
        [prod[0][1], prod[1][1], prod[2][1], prod[3][1]],
        [prod[0][2], prod[1][2], prod[2][2], prod[3][2]],
        [prod[0][3], prod[1][3], prod[2][3], prod[3][3]],
    ];
    let hnf = hnf_mod_core::<N>(&gens, &modulus);
    let (j_basis, j_denom) = quat_lattice_reduce_denom::<N>(&hnf, &prod_denom);
    Some((j_basis, j_denom, q))
}

/// Byte-exact keygen FRONT (quaternion side, up to the spine boundary): from
/// the secret generator γ (`gen_a / gen_denom`, standard coords) and the
/// secret norm `N` (= SEC_DEGREE at keygen), build the secret ideal
/// `I = O_0·γ + N·O_0` ([`quat_lideal_create`](crate::quaternion::o0_mul::quat_lideal_create)),
/// reduce it to a PRIME-norm equivalent `J` of norm `q ~ bitsize(p)`
/// ([`quat_lideal_prime_norm_reduced_equivalent`]), and bridge `J` into the
/// spine's `LeftIdeal`
/// ([`c_ideal_to_left_ideal`](crate::quaternion::o0_mul::c_ideal_to_left_ideal)).
///
/// Equivalent left ideals share the same right order, hence the SAME isogeny
/// codomain E_A, so the dim2id2iso spine may consume the small prime-norm `J`
/// instead of the norm-`N` secret ideal. Returns `(spine_ideal, q)`; the spine
/// then maps `spine_ideal` to `E_A` = the public key. This composes the
/// validated byte-exact pieces (C-oracle create, reduced-equivalent,
/// bridge) into the full quaternion-side keygen front; the remaining wiring is
/// the byte-exact sampling of γ at SEC_DEGREE feeding this and the spine's
/// `sample_random_index`-driven index selection (later sessions).
///
/// `kat`-gated (the reduction's box search consumes the byte-exact RNG).
#[cfg(feature = "kat")]
#[allow(dead_code, clippy::too_many_arguments)]
pub fn keygen_prime_norm_left_ideal<const N: usize, R: rand_core::CryptoRng + ?Sized>(
    gen_a: &crate::quaternion::Quaternion<N>,
    gen_denom: &Int<N>,
    secret_norm: &Uint<N>,
    p: &Uint<N>,
    equiv_bound_coeff: u32,
    primality_witnesses: &[Uint<N>],
    rng: &mut R,
) -> Option<(crate::quaternion::LeftIdeal<N>, Uint<N>)> {
    use crate::quaternion::o0_mul::{c_ideal_to_left_ideal, quat_lideal_create};
    let (basis, denom, norm) = quat_lideal_create::<N>(gen_a, gen_denom, secret_norm, p);
    let (j_basis, j_denom, q) = quat_lideal_prime_norm_reduced_equivalent::<N, R>(
        &basis,
        &denom,
        &norm,
        p,
        equiv_bound_coeff,
        primality_witnesses,
        rng,
    )?;
    let spine_ideal = c_ideal_to_left_ideal::<N>(&j_basis, &j_denom, &q);
    Some((spine_ideal, q))
}

/// The COMPLETE byte-exact keygen FRONT — the C `protocols_keygen` loop body
/// up to (but not including) the deterministic isogeny call. Verbatim flow
/// (verified against `the-sqisign` `keygen.c` / `normeq.c`):
///
/// 1. [`sample_secret_gen`](crate::quaternion::represent_integer::sample_secret_gen)
///    at norm `sec_degree` — the C `quat_sampling_random_ideal_O0_given_norm`
///    is_prime path (Stage A: coord0=0, 3× `ibz_rand_interval(0, n−1)` +
///    non-square rejection + `sqrt_mod_p`; Stage B: 4× `ibz_rand_interval(1, n)`
///    + gcd rejection; `gen·gen_rerand`). Byte-exact draw order.
/// 2. [`keygen_prime_norm_left_ideal`] — `quat_lideal_create` (the secret
///    ideal `O_0·gen + sec_degree·O_0`) → `quat_lideal_prime_norm_reduced_equivalent`
///    (`equiv_bound_coeff = 64`, `primality_num_iter` = the witness count;
///    4× `ibz_rand_interval_minm_m(64)` per iteration) → bridge to `LeftIdeal`.
///
/// Returns the spine-ready prime-norm `LeftIdeal` + its prime norm q. The
/// dim2id2iso clapotis spine that follows is DETERMINISTIC (keygen draws no
/// further randomness — there is NO `sample_random_index` in keygen), so this
/// function captures the entire RNG-consuming portion of keygen in byte-exact
/// draw order. The public key is `E_A = spine(spine_ideal)`'s normalized
/// Montgomery `A/C` plus a hint byte.
///
/// `kat`-gated (consumes the byte-exact interval RNG throughout).
#[cfg(feature = "kat")]
#[allow(dead_code, clippy::too_many_arguments)]
pub fn keygen_byte_exact_secret_ideal<const N: usize, R: rand_core::CryptoRng>(
    sec_degree: &Uint<N>,
    p: &Uint<N>,
    sampler_max_trials: usize,
    equiv_bound_coeff: u32,
    primality_witnesses: &[Uint<N>],
    rng: &mut R,
) -> Option<(crate::quaternion::LeftIdeal<N>, Uint<N>)> {
    use crate::quaternion::represent_integer::sample_secret_gen;
    let secret_gen = sample_secret_gen::<N, R>(sec_degree, p, sampler_max_trials, rng)?;
    keygen_prime_norm_left_ideal::<N, R>(
        &secret_gen,
        &Int::<N>::from_i64(1),
        sec_degree,
        p,
        equiv_bound_coeff,
        primality_witnesses,
        rng,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quaternion::ideal::det_4x4;
    use crypto_bigint::Int;

    /// Build the Gram matrix `G[a][b] = b(col_a, col_b)` for the SQIsign LLL
    /// quadratic form `b(u,v) = u0·v0 + u1·v1 + q·(u2·v2 + u3·v3)`, columns
    /// of `basis` as vectors. Returns a full symmetric `Int<N>` matrix.
    fn gram_from_basis<const N: usize>(basis: &[[Int<N>; 4]; 4], q: i64) -> [[Int<N>; 4]; 4] {
        let qf = Int::<N>::from_i64(q);
        let mut g = [[Int::<N>::from_i64(0); 4]; 4];
        for a in 0..4 {
            for b in 0..4 {
                let mut acc = basis[0][a].wrapping_mul(&basis[0][b]);
                acc = acc.wrapping_add(&basis[1][a].wrapping_mul(&basis[1][b]));
                let t2 = basis[2][a].wrapping_mul(&basis[2][b]);
                let t3 = basis[3][a].wrapping_mul(&basis[3][b]);
                acc = acc.wrapping_add(&qf.wrapping_mul(&t2.wrapping_add(&t3)));
                g[a][b] = acc;
            }
        }
        g
    }

    /// Independent f64 verifier (shares no code with `quat_lll_core`):
    /// recompute the Gram-Schmidt of the basis columns under the form and
    /// check size-reduction (|μ| ≤ η) and Lovász (‖b*_i‖² ≥ (δ−μ²)‖b*_{i−1}‖²).
    /// Exact enough for the small integer test lattices used here.
    fn is_lll_reduced<const N: usize>(
        basis: &[[Int<N>; 4]; 4],
        q: f64,
        delta: f64,
        eta: f64,
    ) -> bool {
        let col = |c: usize| -> [f64; 4] {
            [0, 1, 2, 3].map(|r| {
                let v = basis[r][c];
                let neg = bool::from(v.is_negative());
                let mag = v.abs().as_words()[0] as f64;
                if neg { -mag } else { mag }
            })
        };
        let form = |a: &[f64; 4], b: &[f64; 4]| {
            a[0] * b[0] + a[1] * b[1] + q * (a[2] * b[2] + a[3] * b[3])
        };

        // Gram-Schmidt: bstar[i] = col_i − Σ_{j<i} μ_{ij} bstar[j].
        let mut bstar = [[0.0f64; 4]; 4];
        let mut mu = [[0.0f64; 4]; 4];
        let mut bnorm = [0.0f64; 4];
        for i in 0..4 {
            let ci = col(i);
            bstar[i] = ci;
            for j in 0..i {
                mu[i][j] = form(&ci, &bstar[j]) / bnorm[j];
                for k in 0..4 {
                    bstar[i][k] -= mu[i][j] * bstar[j][k];
                }
            }
            bnorm[i] = form(&bstar[i], &bstar[i]);
        }
        // Size-reduction.
        for i in 0..4 {
            for j in 0..i {
                if mu[i][j].abs() > eta + 1e-9 {
                    return false;
                }
            }
        }
        // Lovász.
        for i in 1..4 {
            if bnorm[i] < (delta - mu[i][i - 1] * mu[i][i - 1]) * bnorm[i - 1] - 1e-9 {
                return false;
            }
        }
        true
    }

    /// dpe-LLL BYTE-EXACTNESS oracle: run `quat_lll_core` on the same
    /// `(G, basis)` as the C reference `quat_lll_core` (`lll/l2.c`) and assert
    /// the reduced basis AND reduced Gram match it byte-for-byte. The golden
    /// values were produced by compiling the C `quat_lll_core` standalone
    /// (gcc + flatpak gmp.h + libgmp, `--gc-sections`) and feeding it the
    /// identical integer matrices (lattice = the cref denom60 columns, form
    /// `b(u,v)=u0v0+u1v1+103·(u2v2+u3v3)`). This is the DEEPEST keygen-KAT risk:
    /// the dpe-LLL is the ONLY non-canonical step, so its reduced basis must
    /// reproduce the C's float-driven swap/size-reduction sequence exactly.
    /// A pure cross-implementation oracle (C output vs Rust), not a mirror.
    #[test]
    fn quat_lll_core_matches_c_oracle_cref_denom60() {
        const N: usize = 8;
        let q = 103i64;
        let raw: [[i64; 4]; 4] = [[3, 1, 0, -19], [7, 0, 12, 0], [0, 0, 5, 0], [0, -6, 0, 3]];
        let mut basis = [[Int::<N>::from_i64(0); 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                basis[r][c] = Int::<N>::from_i64(raw[r][c]);
            }
        }
        let mut g = gram_from_basis::<N>(&basis, q);
        quat_lll_core::<N>(&mut g, &mut basis);

        // C `quat_lll_core` golden output on the identical (G, basis).
        let exp_basis: [[i64; 4]; 4] =
            [[3, -31, 15, -3], [7, 14, -7, 5], [0, 0, 0, 5], [0, 0, 3, 0]];
        let exp_gram: [[i64; 4]; 4] = [
            [58, 5, -4, 26],
            [5, 1157, -563, 163],
            [-4, -563, 1201, -80],
            [26, 163, -80, 2609],
        ];
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(
                    basis[r][c],
                    Int::<N>::from_i64(exp_basis[r][c]),
                    "reduced basis[{r}][{c}] must match the C dpe-LLL oracle",
                );
                assert_eq!(
                    g[r][c],
                    Int::<N>::from_i64(exp_gram[r][c]),
                    "reduced gram[{r}][{c}] must match the C dpe-LLL oracle",
                );
            }
        }
    }

    /// dpe-LLL BYTE-EXACTNESS at SCALE (the FP-rounding regime): same C
    /// `quat_lll_core` oracle, but on a lattice with entries ~2^120 so the
    /// Gram entries (~2^247) FAR exceed 2^53 — forcing the dpe 53-bit mantissa
    /// rounding the toy `cref_denom60` test never exercises. Golden values from
    /// the standalone C oracle on the IDENTICAL integer input. Confirms the
    /// Rust `Dpe` reimplementation (f64 mantissa + i32 exponent) reproduces the
    /// C `dpe_t` rounding — and that gcc-double vs LLVM-f64 do NOT diverge on
    /// the size-reduction / Lovász-swap decisions at scale. (dpe EXPONENT-path
    /// coverage, Gram > 2^1023, is the remaining full-SEC_DEGREE oracle.)
    #[test]
    fn quat_lll_core_matches_c_oracle_at_scale_mantissa_rounding() {
        const N: usize = 8;
        // Parse "±<128 hex>" → Int<8>.
        fn hx(s: &str) -> Int<8> {
            let neg = s.as_bytes()[0] == b'-';
            let u = Uint::<8>::from_be_hex(&s[1..]);
            let i = *u.as_int();
            if neg { i.wrapping_neg() } else { i }
        }
        let input: [&str; 16] = [
            "+0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000192ecb205d85207f126ae0225023544",
            "-0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000035e55beb3948aa646c676faa765ee5",
            "+0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000012df0856ee276b177e5af293d0ab16",
            "-00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000324038b15f5897adc95b7f60d2df4f",
            "-00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000241e784068b332f910688c360a9f67",
            "+000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000022c88a13fd3bc2b267fe68076a39d66",
            "-00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000156420fb0331e99d95f1ab6b73445d",
            "+000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003ebae2d8c602d8c90a13c80c5408c5",
            "+000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003dd46b8c845b13cda992009b15837d",
            "-0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000031961792a208d3a0a124b00cbe1426",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002b1304d676dd79991a4cdb13271edb0",
            "-00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000165eed39499d3c4a4cfd7da898af85",
            "-000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001f4ae27248ea4b2dd5ebbec4db38ff",
            "+000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003a6b4b674367df05d2ba8298656f36",
            "-000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003971d0d7c20c4968518182c50195c2",
            "+0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000311b67fe2a5e8f07e794ecaa1a07c09",
        ];
        let out_basis: [&str; 16] = [
            "+000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000015d07561a9f095d8cba46927a8bd65f",
            "+0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000192ecb205d85207f126ae0225023544",
            "-0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000af199d5d1fc16cc8090671c6f3ec9c6",
            "+0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000a6fcd129e1c202b3c2aca38e82fee91",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002086a28ff6b08f82d6f7df44098fdff",
            "-00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000241e784068b332f910688c360a9f67",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000e77128c7d9b47b31dcea2a0ed71774",
            "-0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000d8fcb952da1677256c1d4d8944e1fac",
            "+000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c3e53f9e252402d086d508e576f57",
            "+000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003dd46b8c845b13cda992009b15837d",
            "+0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100615c8fcf5a0ef201cfacf4db5545",
            "-000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002b1027e5d11d694631830aec150f63",
            "+000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001b2068f4fa7d93d7fccec3d38a3637",
            "-000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001f4ae27248ea4b2dd5ebbec4db38ff",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000a19a60483c5bc4d887f0b49cfcf937",
            "+0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000130e33da11e083cb0d6eac5371e208d",
        ];
        let out_gram: [&str; 16] = [
            "+00000000000000000000000000000000000000000000000000000000000000000007622d38722942ae73d3c24012a1fe54a4ef2f689eedf074bb5586ce056bb0",
            "+00000000000000000000000000000000000000000000000000000000000000000001b6f6edb2466a245e891c7fc3d6b5e5433021b0328096edb1120ed7dea041",
            "-000000000000000000000000000000000000000000000000000000000000000000014278736a3a5ef2ab2743173234657387a7dc3af435f466bd4dba65d74292",
            "-000000000000000000000000000000000000000000000000000000000000000000012b83b50e82dd06e13b3c7b9295a717214ba9a6d65bad4fd30f474057fbbb",
            "+00000000000000000000000000000000000000000000000000000000000000000001b6f6edb2466a245e891c7fc3d6b5e5433021b0328096edb1120ed7dea041",
            "+0000000000000000000000000000000000000000000000000000000000000000000a0b616635fc81219e1bfc7908c696fcfa02905d53f8e4e69023a660bfc887",
            "-0000000000000000000000000000000000000000000000000000000000000000000062df84e6bc3d4ff754d6b1a9bf1277d3b05754cf0dfa2eeb840a0643e7ec",
            "-00000000000000000000000000000000000000000000000000000000000000000000d6de1845826b80a65cc36e96b14eed1f7a4e52e22b9f94e44ed135ea6b8e",
            "-000000000000000000000000000000000000000000000000000000000000000000014278736a3a5ef2ab2743173234657387a7dc3af435f466bd4dba65d74292",
            "-0000000000000000000000000000000000000000000000000000000000000000000062df84e6bc3d4ff754d6b1a9bf1277d3b05754cf0dfa2eeb840a0643e7ec",
            "+00000000000000000000000000000000000000000000000000000000000000000108ef191b26bfa1481a66fb58383100dbd53582e12f46c60b3a2fa3fab67e5a",
            "-0000000000000000000000000000000000000000000000000000000000000000004264954bf7a70727f77f52669f0d90a0fa6eece5899af45a8eb269ee7ada52",
            "-000000000000000000000000000000000000000000000000000000000000000000012b83b50e82dd06e13b3c7b9295a717214ba9a6d65bad4fd30f474057fbbb",
            "-00000000000000000000000000000000000000000000000000000000000000000000d6de1845826b80a65cc36e96b14eed1f7a4e52e22b9f94e44ed135ea6b8e",
            "-0000000000000000000000000000000000000000000000000000000000000000004264954bf7a70727f77f52669f0d90a0fa6eece5899af45a8eb269ee7ada52",
            "+000000000000000000000000000000000000000000000000000000000000000001b9dadf81f565cbc398b915da4eedfd413e7b734a867584acbc844462a0660f",
        ];

        let mut basis = [[Int::<N>::from_i64(0); 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                basis[r][c] = hx(input[r * 4 + c]);
            }
        }
        let mut g = gram_from_basis::<N>(&basis, 103);
        quat_lll_core::<N>(&mut g, &mut basis);

        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(
                    basis[r][c],
                    hx(out_basis[r * 4 + c]),
                    "scale: reduced basis[{r}][{c}] must match the C dpe-LLL oracle",
                );
                assert_eq!(
                    g[r][c],
                    hx(out_gram[r * 4 + c]),
                    "scale: reduced gram[{r}][{c}] must match the C dpe-LLL oracle",
                );
            }
        }
    }

    /// dpe-LLL BYTE-EXACTNESS in the EXPONENT regime: same C `quat_lll_core`
    /// oracle, on a lattice with entries ~2^512 so the Gram entries (~2^1031)
    /// exceed 2^1023 — exercising the dpe i32 EXPONENT path (large `exp`
    /// fields, the >53-bit add/sub shortcut) that neither the toy nor the
    /// mantissa-scale test reaches. Golden values from the standalone C oracle
    /// on the IDENTICAL integer input, at `Int<24>` (1536-bit). Together with
    /// `quat_lll_core_matches_c_oracle_at_scale_mantissa_rounding` this fully
    /// discharges the dpe-LLL byte-exactness across both float regimes.
    #[test]
    fn quat_lll_core_matches_c_oracle_at_scale_exponent_path() {
        const N: usize = 24;
        fn hx(s: &str) -> Int<24> {
            let neg = s.as_bytes()[0] == b'-';
            let u = Uint::<24>::from_be_hex(&s[1..]);
            let i = *u.as_int();
            if neg { i.wrapping_neg() } else { i }
        }
        let input: [&str; 16] = [
            "+0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001a62c2bdf3306bf1f772f22431d345859b662f5a4236e6f4c0504e7bbbfd7a2fe381b1d623ed94e431c18a3b73e29939672c076aee4184459de10116a924651e2",
            "-00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001282a9afbe50ba6b7046ccd65959a5f8b520341da7edd59b49e08ed7415bcf41bfc7cf48f50ecca2df7dcd12782e24ea1253bbf77a5df4e488578ae19464e155",
            "+000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000032893b45a8255f2982c9a0b88b1cd81eb4164e9b15fb1c89f8a2b963845ee1944f572075f84c2083cce4c4abbebcfaa37327f0c406cee14e9ab5254178eb6bd9",
            "-000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003b119d515bb64d9b5b3b9bf2d5dda0add71a429cd38f6e1bbd80f803711147d1cfa3cce2579bf78228302355a47023a083ac779834937ea23455ec597394666",
            "-00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003f06ea550a872d6452e845d33d8472a1050976c0ab6377efec082c4f758abf13bef7dabee0eb00f535ace2eac45957cc8d3900ec585266fb3a962f79899e1dc7",
            "+0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002176b709ad997608961e20257ece72dd72415a2f4a4b85f8c0342d9b4f99d158f6fe4da0acfe10905c205c48e8d4817da78de262a75ef2cbae8fa795932043296",
            "-0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000173583639f00898accb32bc9b3d6d5fb4b6df0621a5b8671faad1292f20533867d11b1eb172c06b6594ac1ecfab4bbee9cfa8e6bca67dc7a8468649439e921b1",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003a31cac1661794555f60745cf98933f3dcfae232da6989b9711d6bdb29c618af731818b749a57ad95d01d5bab4fff8fd11ffa023aa9bc59a589a4c2568933bd6",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001d7c95eb6eebb7f2970b9b74cef0e5076c757c136fb07c4f024bdba06387783831e939e19a506ba8bf888209f8b0ee94daf68334f608127a4f8414b22b02df6e",
            "-000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011cb8796c073cf9ea5a9e68f7ad2167d9cd450429769a4d6da441512eb15e62544acb099c2a19a0073f2f3179f2abe565b27d8dc68cf1bd926c9c9becf67d070",
            "+0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002b214dad55f3983c9890d2482aad4e93bf00164295c2db7b3bcde88e5ad0a8a51dacc4a056a07615f312dba40cc0097236481af4b78da8530714998d86e44c11a",
            "-00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001a9d074ed7300510674a2aabe2ed310e54f382e02306051f51911ab6b0b09a7a6aaab79d77011062c5caff22d1fc40b5380b7a7df8db9a7c7017f3ed471b1564",
            "-00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002746299d82428997d48e98795e069556574f5dc78553deb0acd1a91c43b0be02411b545341a1eecfb05d17114781462c091f724a3f2a696afb4e4520e316d55b",
            "+0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000345aacca94bbae01d67199855561139c137bb47943887ff1a2b978b4861578c016d1a9883d7ade904fba27c0cf57aba77a7b3e6dfccc35d2105d016bc010c7b9",
            "-00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001050e75b8de134a5e6e71eb548723a244e35e3d9e26457e894a12b051bccd66fff4dd94344a1d40f95f1b25d0715790f5e104dc112b4a469e3da8300818de63b",
            "+000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000304948b86257a76f085b2efe41994db3d5f390c2a442be5821fa3f4e626ff9ce861607d962c010c3ea627d85d60558f65ddd8ad376bbc9308de9f51f1a9d79677",
        ];
        let out_basis: [&str; 16] = [
            "+000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000193a9822f74b604b406e8556cc3dab2610142c1867b8099b0bb2458e47e7bd3bc78534e1949ca81a03c9ad6a4c5fb6eac606cbab769ba4f7555b88688fde1708d",
            "+0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001a62c2bdf3306bf1f772f22431d345859b662f5a4236e6f4c0504e7bbbfd7a2fe381b1d623ed94e431c18a3b73e29939672c076aee4184459de10116a924651e2",
            "+0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006f18ce6166df33bbd586acf7d0633c53785bb36fb0825f256c99e59a30f1883c4825945279f3578c2ab548e03f74604b21e69bf1770eda1f3feb09d4c97d867cb",
            "-00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000080c53a01656fc5c73d1220a9706e8e1a1dbd87d999b2d0ff22075cd473ad74d62c930727541f266cbbf966de87812e74cca9a60a66daa7472bb9b31d362742d91",
            "+0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001d8648645cf1033250ef9bc84af62bb361f0c2c33f954e79c173aad658412567bb0ecff4beef608108c58e1a3c8eec00deba5253e1d9cc5bfae6449dfa86614cf",
            "-00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003f06ea550a872d6452e845d33d8472a1050976c0ab6377efec082c4f758abf13bef7dabee0eb00f535ace2eac45957cc8d3900ec585266fb3a962f79899e1dc7",
            "-000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000b3703119903fa03ead5aaaea79d0c7bc3eb9727970abe33c24f5185eb8c764f6cb83f947a496bb61843ce8cb91ecddd797f3b22620459b5f91ea2d07eb7fa3834",
            "+000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000123ed104595a3596ad1d631567fbf6729cdc161613e95d13da17bcafa59b087dc3dc593cf4d6afe13b315aca8db09fb1025227631ef34266da08688cb762d7332",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000bb10e54ae77e853f161b4e5541ece89cfa12bd0d846d7782807c68d78719212ed3c8947d7aed1a84b958ef25986303e7fceaa588d38f6a128ba4af35b9b0efe",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001d7c95eb6eebb7f2970b9b74cef0e5076c757c136fb07c4f024bdba06387783831e939e19a506ba8bf888209f8b0ee94daf68334f608127a4f8414b22b02df6e",
            "+0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000b44ef650a6c1e8d33135c3f4bef0d3836d07feba90cc244efaf2bd2974dac520b459a2e9bf68a0427c15c87b2c9d314dfb95cea3eb503aaa8fe94f9571f77a84",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000021ea5ed3c349eec0c95d31b3aa0207016d1b5f7c82dbb4a28b1633ec3bb653138e13e289d667547137383300ef08bee3b1db11f42aab228cce3b5315d973663f4",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000d14832d1279246a01e3010bf75a7e45bc2c56b1be34a140f5e7cf984264babdd5b65534fbd8efc09f5d10af87d6657b715bcc23bda1cc67150ebc4adcf9f25e",
            "-00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002746299d82428997d48e98795e069556574f5dc78553deb0acd1a91c43b0be02411b545341a1eecfb05d17114781462c091f724a3f2a696afb4e4520e316d55b",
            "+000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000161b6851b34c5ea8803408e63f69523d7a0910eaddd31eb073ebdf4ee79d1491eeadadc9ff212f82e5610b554359a1c7054f2abe87f15e0429f9536535d285a6e",
            "+0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000b40de8b7fd6b7b513fe1dba98daeb08b6656f10bb83f018acb77228836a6df9b463acc5d0387d5fedbdfc0f95e70e5cccf8cedb2291f6aad04acd6a3ede4448c",
        ];
        let out_gram: [&str; 16] = [
            "+000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000066007b216d2c22a51626eb735ab81602b7154d1c5b0b5e5d226c911d1f0cd5b1e8193aea77b12fa9439818e54949062446b79f91db7beaef66849fde4d64dc896aa0587d5f5ef4312db6bd8c9f9b516ea7c22d19b443aead6956cf6056068eac4870996f0078c23473abd82dffe563d197aa282abc1c81448a287e38f85e475c2",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001e16666ebc728591eb68277a5a790bd553f3c1286b10544d4c74146a24f437372a8dfc727dca6452b837cadadd51b2028e82c06353ea8ecef58f2950cf57c976d3b02fd3a02b1b00a58d3748ce43cd218c186dbee7eeb67956961229eec5f0336cf41c5f7156380458d252d84de05ce451e243330f7e71f447ca8cb081bb4d767",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000d6afcedef690b998da33a0a144b543dc1c2618f906e67912821df316302e8d11298f050f7854d2fc1d8fdcdb6e6405f1691f9bda62be2215c00b89ad14d290ae85eca8eee48b8f723191a80fab4dc0c26c27d90d62995a74b74e577d3d3fe79c7c831e3f1b462135694fef50211e2864e1733f107dedde0eb4e2da863e0bfec7",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003162bc509c66dee041733743bedf75b2801144375488421d2cea86eb3e235eb2bb47372e70dbdb8f3ff008a5f30ecc51791a73d03a4ac2f166b2f73c156f7e62add363f13e7ac91682fd8eec59650a0e9e2a53fd8206c1d3b3282c89e2262adbdb9444f2d513ce2e9c133105cb04e6a38925d9795636c660eaf6821186c7d8911",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001e16666ebc728591eb68277a5a790bd553f3c1286b10544d4c74146a24f437372a8dfc727dca6452b837cadadd51b2028e82c06353ea8ecef58f2950cf57c976d3b02fd3a02b1b00a58d3748ce43cd218c186dbee7eeb67956961229eec5f0336cf41c5f7156380458d252d84de05ce451e243330f7e71f447ca8cb081bb4d767",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006922569f71a55d534609f26f9ee30614792d0fde62dd9238eff1ae3993ca898621dd274467f5b638f090c1be1a6a94634bcbe404468247e54f4dea650dcb3e88076725239ee41cbfe6fe41706567fceef683cc44b70fcfdfb5a9be45d5601397d76fedca12bf205e6f857986b2ed0307c4997f4690325dc9422bcaf07ab25c260",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000bc19ab8190e15af08840284173a4b545af31b77dd882be91366c9e30e286185ccd2248826f1529f578a2b07b9838ebc76e12a763f600eac3e5dbf6d87e729e7935a953dcf203476cb65e82c01bb168e77c803b0a09196353851b200d5cfd3d38ba87c5d71e35b98d6450f20cde1c5eee65095938c11455ee257a89ee2dfa4334",
            "+000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007b1d470ba1848c3b92f718e1105d62a793e8a2417acc905f48cf6157456eee76ce3565bfd58aac26a3f041d40305764f54b64c5b6f564b2c76b85003b903864715eb1767cbad3f5bdf220983b261ea0f2af50ebcd649857e7103ac570f65de27274daeef4a8e2f9ec93f23fd32c7f0b289847b09d79721deeffe52d2ea791a2c",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000d6afcedef690b998da33a0a144b543dc1c2618f906e67912821df316302e8d11298f050f7854d2fc1d8fdcdb6e6405f1691f9bda62be2215c00b89ad14d290ae85eca8eee48b8f723191a80fab4dc0c26c27d90d62995a74b74e577d3d3fe79c7c831e3f1b462135694fef50211e2864e1733f107dedde0eb4e2da863e0bfec7",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000bc19ab8190e15af08840284173a4b545af31b77dd882be91366c9e30e286185ccd2248826f1529f578a2b07b9838ebc76e12a763f600eac3e5dbf6d87e729e7935a953dcf203476cb65e82c01bb168e77c803b0a09196353851b200d5cfd3d38ba87c5d71e35b98d6450f20cde1c5eee65095938c11455ee257a89ee2dfa4334",
            "+000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001a5b7a32f54639926e7efacd4d3c6eb4df0d50ee4d4b7d5cc356c5790aa16ae2bf463100619b206720dae39035e116e28a16d0751d6eeaae455011698bb275aff8225cb51b3db0f61f5b2a83dd0b4febbb3e247293d6fd6214270dd2fe572064ba6f81f11114d9bff0f2540f02ee8731fd8b580243f8bf23f6d8f334014b6974455",
            "+000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000b93350ee11cdc8fb9ee04f7eb1fd54dd7a7bed9d011d350ddb354d08c32f01ab18db6ed9b86105e8528c40750fbe230dccfc89c758103dab709a8eecde744b3bbad26ddd9b2860cdc4ad109c2890a3a6d62db728fb5073609703ee0ad54aee9b3a878214affcb53f78ab2b7578569a3faadff82487d0e37282a68abfb4553adfa5",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000003162bc509c66dee041733743bedf75b2801144375488421d2cea86eb3e235eb2bb47372e70dbdb8f3ff008a5f30ecc51791a73d03a4ac2f166b2f73c156f7e62add363f13e7ac91682fd8eec59650a0e9e2a53fd8206c1d3b3282c89e2262adbdb9444f2d513ce2e9c133105cb04e6a38925d9795636c660eaf6821186c7d8911",
            "+000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007b1d470ba1848c3b92f718e1105d62a793e8a2417acc905f48cf6157456eee76ce3565bfd58aac26a3f041d40305764f54b64c5b6f564b2c76b85003b903864715eb1767cbad3f5bdf220983b261ea0f2af50ebcd649857e7103ac570f65de27274daeef4a8e2f9ec93f23fd32c7f0b289847b09d79721deeffe52d2ea791a2c",
            "+000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000b93350ee11cdc8fb9ee04f7eb1fd54dd7a7bed9d011d350ddb354d08c32f01ab18db6ed9b86105e8528c40750fbe230dccfc89c758103dab709a8eecde744b3bbad26ddd9b2860cdc4ad109c2890a3a6d62db728fb5073609703ee0ad54aee9b3a878214affcb53f78ab2b7578569a3faadff82487d0e37282a68abfb4553adfa5",
            "+00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000243d37c59067d976d4456098e07ad504cc35596ec804163ab5717141b5978b418710f71e8497dd7fa0d867c0c0a8cfcbf4be3d66626dcf964e8c01c8b50459b388ce01232c08a0de1765e75e156805eb19485f11a72317725c7be16fcb8c95d9be738ec6d2ef3e8347417592fa71bd689301d3dd4e031056405ce2857b6254449c5",
        ];
        let mut basis = [[Int::<N>::from_i64(0); 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                basis[r][c] = hx(input[r * 4 + c]);
            }
        }
        let mut g = gram_from_basis::<N>(&basis, 103);
        quat_lll_core::<N>(&mut g, &mut basis);
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(
                    basis[r][c],
                    hx(out_basis[r * 4 + c]),
                    "exp: basis[{r}][{c}]"
                );
                assert_eq!(g[r][c], hx(out_gram[r * 4 + c]), "exp: gram[{r}][{c}]");
            }
        }
    }

    /// The hardcoded `denom=60` lattice from the C-ref test
    /// `quat_test_lll_lattice_lll` (alg prime q=103). The C HNF-normalizes
    /// first, but LLL on ANY basis of the lattice yields a reduced basis of
    /// that lattice, so the raw basis is a valid reducer input. Assert the
    /// output is LLL-reduced and spans the same lattice (|det| preserved).
    #[test]
    fn lll_reduces_cref_denom60_lattice() {
        const N: usize = 8;
        let q = 103i64;
        // basis[row][col], columns are the lattice vectors.
        let raw: [[i64; 4]; 4] = [[3, 1, 0, -19], [7, 0, 12, 0], [0, 0, 5, 0], [0, -6, 0, 3]];
        let mut basis = [[Int::<N>::from_i64(0); 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                basis[r][c] = Int::<N>::from_i64(raw[r][c]);
            }
        }
        let det_in = det_4x4::<N>(&basis).abs();
        let mut g = gram_from_basis::<N>(&basis, q);

        quat_lll_core::<N>(&mut g, &mut basis);

        let det_out = det_4x4::<N>(&basis).abs();
        assert_eq!(
            det_in, det_out,
            "LLL must preserve the lattice (|det| invariant)"
        );
        assert!(
            is_lll_reduced::<N>(&basis, q as f64, 0.99, 0.51),
            "output must be LLL-reduced under the C-ref δ=0.99, η=0.51",
        );
    }

    /// A second, independent integer lattice (identity-ish skew) at q=1
    /// (standard Euclidean form) — sanity that the reducer generalizes and
    /// preserves the lattice.
    #[test]
    fn lll_reduces_skewed_lattice_euclidean() {
        const N: usize = 8;
        let raw: [[i64; 4]; 4] = [[1, 5, 9, 13], [0, 1, 4, 7], [0, 0, 1, 2], [0, 0, 0, 1]];
        let mut basis = [[Int::<N>::from_i64(0); 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                basis[r][c] = Int::<N>::from_i64(raw[r][c]);
            }
        }
        let det_in = det_4x4::<N>(&basis).abs();
        let mut g = gram_from_basis::<N>(&basis, 1);

        quat_lll_core::<N>(&mut g, &mut basis);

        let det_out = det_4x4::<N>(&basis).abs();
        assert_eq!(det_in, det_out, "lattice preserved (|det| invariant)");
        assert!(
            is_lll_reduced::<N>(&basis, 1.0, 0.99, 0.51),
            "output must be LLL-reduced",
        );
    }

    /// An already-reduced basis stays reduced and lattice-invariant
    /// (idempotence-ish): feeding a reduced basis must not corrupt it.
    #[test]
    fn lll_on_identity_is_stable() {
        const N: usize = 8;
        let mut basis = [[Int::<N>::from_i64(0); 4]; 4];
        for i in 0..4 {
            basis[i][i] = Int::<N>::from_i64(1);
        }
        let det_in = det_4x4::<N>(&basis).abs();
        let mut g = gram_from_basis::<N>(&basis, 103);
        quat_lll_core::<N>(&mut g, &mut basis);
        let det_out = det_4x4::<N>(&basis).abs();
        assert_eq!(det_in, det_out);
        assert!(is_lll_reduced::<N>(&basis, 103.0, 0.99, 0.51));
    }

    /// `lideal_reduce_basis` builds the class Gram (= lattice_gram, since
    /// denom=norm=1) and reduces via `quat_lll_core`. The class Gram is
    /// 2× the form Gram (a power-of-two scale ⇒ identical reduced basis),
    /// so the output must be LLL-reduced under the form and preserve the
    /// lattice (|det| invariant).
    #[test]
    fn reduce_basis_trivial_denom_norm() {
        const N: usize = 8;
        let p = 103u64;
        let raw: [[i64; 4]; 4] = [[3, 1, 0, -19], [7, 0, 12, 0], [0, 0, 5, 0], [0, -6, 0, 3]];
        let mut basis = [[Int::<N>::from_i64(0); 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                basis[r][c] = Int::<N>::from_i64(raw[r][c]);
            }
        }
        let det_in = det_4x4::<N>(&basis).abs();
        let (reduced, _gram) = lideal_reduce_basis::<N>(
            &basis,
            &Uint::<N>::from_u64(1),
            &Uint::<N>::from_u64(1),
            &Uint::<N>::from_u64(p),
        )
        .expect("trivial denom/norm division is always exact");
        assert_eq!(det_4x4::<N>(&reduced).abs(), det_in, "lattice preserved");
        assert!(
            is_lll_reduced::<N>(&reduced, p as f64, 0.99, 0.51),
            "reduce_basis output must be LLL-reduced",
        );
    }

    /// Exercises the exact-division class-Gram path with norm > 1: scaling a
    /// basis by 2 scales `lattice_gram` by 4, so `norm = 4` divides it
    /// exactly, and the reduced 2·B is 2× a reduced B (still LLL-reduced,
    /// lattice preserved). Confirms the `int_div_exact_signed` path works on
    /// negative off-diagonal Gram entries.
    #[test]
    fn reduce_basis_with_norm_divisor() {
        const N: usize = 8;
        let p = 103u64;
        let raw: [[i64; 4]; 4] = [[2, 10, 18, 26], [0, 2, 8, 14], [0, 0, 2, 4], [0, 0, 0, 2]];
        let mut basis = [[Int::<N>::from_i64(0); 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                basis[r][c] = Int::<N>::from_i64(raw[r][c]);
            }
        }
        let det_in = det_4x4::<N>(&basis).abs();
        let (reduced, _gram) = lideal_reduce_basis::<N>(
            &basis,
            &Uint::<N>::from_u64(1),
            &Uint::<N>::from_u64(4),
            &Uint::<N>::from_u64(p),
        )
        .expect("lattice_gram(2·B) is divisible by norm=4");
        assert_eq!(det_4x4::<N>(&reduced).abs(), det_in, "lattice preserved");
        assert!(
            is_lll_reduced::<N>(&reduced, p as f64, 0.99, 0.51),
            "reduce_basis output must be LLL-reduced",
        );
    }

    /// The box-search core: reduce a toy lattice, then search for a
    /// prime-norm element. With denom=1 the divisor is 1, so `q = qf_eval`;
    /// over `[−64,64]^4` a prime value of the form appears within a few
    /// attempts. Asserts the returned `q` is prime and `α = reduced·coord`
    /// is non-zero. (Building the equivalent ideal `J` and the coordinate
    /// conversions are the next session.)
    #[cfg(feature = "kat")]
    #[test]
    fn prime_norm_box_search_finds_prime() {
        use crate::quaternion::primality::is_probable_prime_with_witnesses;
        use crate::rng::NistPqcRng;
        const N: usize = 8;
        let p = 103u64;
        let raw: [[i64; 4]; 4] = [[3, 1, 0, -19], [7, 0, 12, 0], [0, 0, 5, 0], [0, -6, 0, 3]];
        let mut basis = [[Int::<N>::from_i64(0); 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                basis[r][c] = Int::<N>::from_i64(raw[r][c]);
            }
        }
        let (reduced, gram) = lideal_reduce_basis::<N>(
            &basis,
            &Uint::<N>::from_u64(1),
            &Uint::<N>::from_u64(1),
            &Uint::<N>::from_u64(p),
        )
        .expect("trivial division exact");

        let witnesses: [Uint<N>; 6] = [2u64, 3, 5, 7, 11, 13].map(Uint::<N>::from_u64);
        let mut rng = NistPqcRng::new(&[0x5cu8; 48]);
        let (alpha, q) = prime_norm_box_search::<N, _>(
            &reduced,
            &gram,
            &Uint::<N>::from_u64(1),
            64,
            &witnesses,
            &mut rng,
        )
        .expect("box search must find a prime-norm element");

        assert!(
            is_probable_prime_with_witnesses::<N>(&q, &witnesses),
            "returned q must be prime",
        );
        assert!(
            alpha.iter().any(|x| *x != Int::<N>::from_i64(0)),
            "α = reduced·coord must be non-zero",
        );
    }

    /// Full byte-exact `quat_lideal_prime_norm_reduced_equivalent`: build an
    /// ideal `I` via `quat_lideal_create`, reduce → box-search →
    /// conjugate → right-multiply → `J`. INDEPENDENT invariants (mathematical
    /// facts, not the implementation formula): `N(J) = q` is prime, and the
    /// lattice index `[O_0 : J] = 4·|det(J)| / denom_J⁴ = q²` (covolume = N²,
    /// O_0 covolume 1/4) — so `J` is a well-formed prime-norm left ideal.
    ///
    /// Two ideals (per the advisor's "one passing vector is the real gap"):
    /// γ=1+3i (only the 1,i coords) and γ=1+2j (a `j`-component, exercising
    /// the `j²=−p` paths in the right-multiply quaternion product), each at a
    /// distinct DRBG seed.
    #[cfg(feature = "kat")]
    #[test]
    fn lideal_prime_norm_reduced_equivalent_produces_prime_norm_ideal() {
        use crate::quaternion::Quaternion;
        use crate::quaternion::o0_mul::quat_lideal_create;
        use crate::quaternion::primality::is_probable_prime_with_witnesses;
        use crate::rng::NistPqcRng;
        const N: usize = 8;
        let p = Uint::<N>::from_u64(7);
        let witnesses: [Uint<N>; 6] = [2u64, 3, 5, 7, 11, 13].map(Uint::<N>::from_u64);

        // (γ standard coords, N_red(γ), DRBG seed byte). γ=1+3i → 1+9=10;
        // γ=1+2j → 1 + 7·4 = 29 (the j-component drives the j²=−p product).
        let cases: [([i64; 4], u64, u8); 2] = [([1, 3, 0, 0], 10, 0x42), ([1, 0, 2, 0], 29, 0x9e)];

        for (coords, n_red, seed) in cases {
            let g = Quaternion::<N>::new(
                Int::<N>::from_i64(coords[0]),
                Int::<N>::from_i64(coords[1]),
                Int::<N>::from_i64(coords[2]),
                Int::<N>::from_i64(coords[3]),
            );
            let (basis, denom, norm) = quat_lideal_create::<N>(
                &g,
                &Int::<N>::from_i64(1),
                &Uint::<N>::from_u64(n_red),
                &p,
            );
            assert_eq!(norm, Uint::<N>::from_u64(n_red), "ideal norm = N_red(γ)");

            let mut rng = NistPqcRng::new(&[seed; 48]);
            let (j_basis, j_denom, q) = quat_lideal_prime_norm_reduced_equivalent::<N, _>(
                &basis, &denom, &norm, &p, 64, &witnesses, &mut rng,
            )
            .expect("must find a prime-norm equivalent ideal");

            // N(J) = q is prime.
            assert!(
                is_probable_prime_with_witnesses::<N>(&q, &witnesses),
                "returned norm q must be prime (γ={coords:?})",
            );
            // [O_0 : J] = 4·|det(J)| / denom_J⁴ = q² (covolume = N(J)², O_0 covol 1/4).
            // This independently re-derives q² from J's basis and the right-
            // multiply denom — agreeing with the search's q iff every step is
            // consistent (reduce, box-search, conj, right-multiply, HNF).
            let det_abs = det_4x4::<N>(&j_basis).abs();
            let scaled = det_abs.wrapping_mul(&Uint::<N>::from_u64(4));
            let d = j_denom.abs();
            let d4 = d.wrapping_mul(&d).wrapping_mul(&d).wrapping_mul(&d);
            let index = scaled
                .div_rem_vartime(&NonZero::new(d4).expect("denom_J > 0"))
                .0;
            let q_sq = q.wrapping_mul(&q);
            assert_eq!(
                index, q_sq,
                "[O_0 : J] must equal N(J)² = q² (γ={coords:?})"
            );
            // J denominator is positive and non-zero.
            assert!(
                !bool::from(j_denom.is_negative()) && j_denom != Int::<N>::from_i64(0),
                "J denominator must be positive (γ={coords:?})",
            );
        }
    }

    /// Full byte-exact keygen FRONT up to the spine boundary
    /// (`keygen_prime_norm_left_ideal`): secret γ + norm N → secret ideal →
    /// prime-norm equivalent → bridged `LeftIdeal`. INDEPENDENT invariants: the
    /// returned `q` is prime, and the bridged O_0-coords ideal has lattice
    /// index `[O_0 : I] = |det(basis)| / denom⁴ = q²` (O_0 is the identity in
    /// O_0-coords) — an independent re-derivation from the bridged basis that
    /// agrees with the search's `q` iff the whole front (create, reduce,
    /// bridge) is consistent.
    #[cfg(feature = "kat")]
    #[test]
    fn keygen_prime_norm_left_ideal_front_is_consistent() {
        use crate::quaternion::Quaternion;
        use crate::quaternion::primality::is_probable_prime_with_witnesses;
        use crate::rng::NistPqcRng;
        const N: usize = 8;
        let p = Uint::<N>::from_u64(7);
        let g = Quaternion::<N>::new(
            Int::<N>::from_i64(1),
            Int::<N>::from_i64(3),
            Int::<N>::from_i64(0),
            Int::<N>::from_i64(0),
        );
        let witnesses: [Uint<N>; 6] = [2u64, 3, 5, 7, 11, 13].map(Uint::<N>::from_u64);
        let mut rng = NistPqcRng::new(&[0x7du8; 48]);
        let (spine_ideal, q) = keygen_prime_norm_left_ideal::<N, _>(
            &g,
            &Int::<N>::from_i64(1),
            &Uint::<N>::from_u64(10),
            &p,
            64,
            &witnesses,
            &mut rng,
        )
        .expect("keygen front must produce a prime-norm spine ideal");

        assert!(
            is_probable_prime_with_witnesses::<N>(&q, &witnesses),
            "keygen front q must be prime",
        );
        // [O_0 : spine_ideal] = |det(O_0-basis)| / denom⁴ = q² (independent).
        let det_abs = det_4x4::<N>(&spine_ideal.basis).abs();
        let d = spine_ideal.denom;
        let d4 = d.wrapping_mul(&d).wrapping_mul(&d).wrapping_mul(&d);
        let index = det_abs
            .div_rem_vartime(&NonZero::new(d4).expect("denom > 0"))
            .0;
        assert_eq!(
            index,
            q.wrapping_mul(&q),
            "bridged spine ideal index must equal q²",
        );
    }

    /// The COMPLETE byte-exact keygen front including the secret SAMPLING
    /// (`keygen_byte_exact_secret_ideal`): sample γ at norm `sec_degree` via the
    /// is_prime path → secret ideal → prime-norm equivalent → bridged
    /// `LeftIdeal`. Reduced scale (sec_degree = 11, a prime ≡ 3 mod 4 so the
    /// sampler's `sqrt_mod_p` path is exercised; p = 7). INDEPENDENT invariants:
    /// q prime and `[O_0 : I] = |det|/denom⁴ = q²`. This drives the full keygen
    /// RNG-consuming path (Stage A/B sampling draws + reduction draws) in one
    /// deterministic seeded run.
    #[cfg(feature = "kat")]
    #[test]
    fn keygen_byte_exact_secret_ideal_front_runs() {
        use crate::quaternion::primality::is_probable_prime_with_witnesses;
        use crate::rng::NistPqcRng;
        const N: usize = 8;
        let p = Uint::<N>::from_u64(7);
        let sec_degree = Uint::<N>::from_u64(11); // small prime ≡ 3 mod 4
        let witnesses: [Uint<N>; 6] = [2u64, 3, 5, 7, 11, 13].map(Uint::<N>::from_u64);
        let mut rng = NistPqcRng::new(&[0xa5u8; 48]);

        let (spine_ideal, q) =
            keygen_byte_exact_secret_ideal::<N, _>(&sec_degree, &p, 4096, 64, &witnesses, &mut rng)
                .expect("byte-exact keygen front must produce a prime-norm spine ideal");

        assert!(
            is_probable_prime_with_witnesses::<N>(&q, &witnesses),
            "keygen front q must be prime",
        );
        let det_abs = det_4x4::<N>(&spine_ideal.basis).abs();
        let d = spine_ideal.denom;
        let d4 = d.wrapping_mul(&d).wrapping_mul(&d).wrapping_mul(&d);
        let index = det_abs
            .div_rem_vartime(&NonZero::new(d4).expect("denom > 0"))
            .0;
        assert_eq!(
            index,
            q.wrapping_mul(&q),
            "bridged keygen spine ideal index must equal q²",
        );
    }

    /// The byte-exact keygen front at FULL SEC_DEGREE scale (lvl1: norm
    /// 2^512+75): confirms `keygen_byte_exact_secret_ideal` runs at the real
    /// keygen width (WIDE = 48 limbs / 3072-bit — `quat_lideal_create`'s det
    /// path ≈ N⁴ ≈ 2^2052 and the reduce/box-search Gram ≈ 2^1278 both fit) and
    /// produces a prime-norm spine-ready ideal. INDEPENDENT invariants: q prime
    /// + `[O_0 : I] = |det(basis)|/denom⁴ = q²`. This is the prerequisite for
    /// the end-to-end keygen → KAT pk run. Heavy (real-scale dpe-LLL +
    /// Miller-Rabin at 3072-bit), hence ignored in the default run.
    #[cfg(feature = "kat")]
    #[ignore = "SEC_DEGREE-scale keygen front (heavy: WIDE=48 sampler + reduce)"]
    #[test]
    fn keygen_byte_exact_secret_ideal_front_at_sec_degree() {
        use crate::quaternion::ideal::det_4x4;
        use crate::quaternion::primality::is_probable_prime_with_witnesses;
        use crate::rng::NistPqcRng;
        const N: usize = 48;
        let p = crate::params::lvl1::prime().resize::<N>();
        let sec_degree = crate::params::lvl1::sec_degree().resize::<N>(); // 2^512 + 75
        let witnesses: [Uint<N>; 12] =
            [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37].map(Uint::<N>::from_u64);
        let mut rng = NistPqcRng::new(&[0x33u8; 48]);

        let (spine_ideal, q) =
            keygen_byte_exact_secret_ideal::<N, _>(&sec_degree, &p, 8192, 64, &witnesses, &mut rng)
                .expect("SEC_DEGREE keygen front must produce a prime-norm spine ideal");

        assert!(
            is_probable_prime_with_witnesses::<N>(&q, &witnesses),
            "keygen front q must be prime at SEC_DEGREE scale",
        );
        let det_abs = det_4x4::<N>(&spine_ideal.basis).abs();
        let d = spine_ideal.denom;
        let d4 = d.wrapping_mul(&d).wrapping_mul(&d).wrapping_mul(&d);
        let index = det_abs
            .div_rem_vartime(&NonZero::new(d4).expect("denom > 0"))
            .0;
        assert_eq!(
            index,
            q.wrapping_mul(&q),
            "SEC_DEGREE keygen spine ideal index must equal q²",
        );
    }

    /// BYTE-EXACT ORACLE (link 3): feed the C-byte-exact KAT[0] secret
    /// ideal basis (from `quat_lideal_create_matches_c_oracle_kat0`, denom 2,
    /// norm SEC_DEGREE) through our `lideal_reduce_basis` and assert the reduced
    /// basis `red` AND the reduce Gram match the C reference `quat_lideal_reduce_basis`
    /// byte-for-byte (captured via `the-sqisign` `lll_applications.c` CDUMP2).
    /// If this passes, the keygen pk divergence is NOT in the quaternion
    /// reduce path (create + reduce both byte-exact) — it is in the box-search
    /// α selection / RNG. If it fails, the first differing entry pins the
    /// `lideal_reduce_basis` gram/division/LLL-seed bug.
    #[test]
    fn lideal_reduce_basis_matches_c_oracle_kat0() {
        use crypto_bigint::{Int, Uint};
        const WL: usize = 32;

        fn hxw(s: &str) -> Int<WL> {
            let neg = s.as_bytes()[0] == b'-';
            let body = if neg { &s[1..] } else { s };
            let h = if body.len() % 2 == 1 {
                format!("0{body}")
            } else {
                body.to_string()
            };
            let nbytes = h.len() / 2;
            let mut buf = [0u8; WL * 8];
            for i in 0..nbytes {
                buf[WL * 8 - nbytes + i] = u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap();
            }
            let v = *Uint::<WL>::from_be_slice(&buf).as_int();
            if neg { v.wrapping_neg() } else { v }
        }
        fn hxu(s: &str) -> Uint<WL> {
            let h = if s.len() % 2 == 1 {
                format!("0{s}")
            } else {
                s.to_string()
            };
            let nbytes = h.len() / 2;
            let mut buf = [0u8; WL * 8];
            for i in 0..nbytes {
                buf[WL * 8 - nbytes + i] = u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap();
            }
            Uint::<WL>::from_be_slice(&buf)
        }

        // C secret-ideal basis (byte-exact, proven), column-major [row][col].
        let two_n = "200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000096";
        let basis: [[Int<WL>; 4]; 4] = [
            [
                hxw(two_n),
                hxw("0"),
                hxw(
                    "1687ff65ad3acae767bc282bc38e059fe5596f303032ef89615601250aa9acafea9e87b06e4844f43b0d82e58f3a85c53e01cdc756d40858f19550e9396642e80",
                ),
                hxw(
                    "166360752f138e7dfdae0eeee063281fd7095b4d0fe5fd455b238b877788863766346231bc8879c85dd1164d79c2388d452652d6c1ed925d0f58adb23577131d1",
                ),
            ],
            [
                hxw("0"),
                hxw(two_n),
                hxw(
                    "99c9f8ad0ec71820251f1111f9cd7e028f6a4b2f01a02baa4dc7478887779c899cb9dce43778637a22ee9b2863dc772bad9ad293e126da2f0a7524dca88ecec5",
                ),
                hxw(
                    "1687ff65ad3acae767bc282bc38e059fe5596f303032ef89615601250aa9acafea9e87b06e4844f43b0d82e58f3a85c53e01cdc756d40858f19550e9396642e80",
                ),
            ],
            [hxw("0"), hxw("0"), hxw("1"), hxw("0")],
            [hxw("0"), hxw("0"), hxw("0"), hxw("1")],
        ];
        let denom = hxu("2");
        let norm = hxu(
            "10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004b",
        );
        let p = crate::params::lvl1::prime().resize::<WL>();

        let (red, gram) = lideal_reduce_basis::<WL>(&basis, &denom, &norm, &p)
            .expect("reduce_basis must succeed on a genuine ideal");

        let c_red: [[Int<WL>; 4]; 4] = [
            [
                hxw(
                    "1d8003b4f6a19d62f621051e0818563242f1f179c83e104e4dbcfa742e831bdb866fe2b621b2ea4a",
                ),
                hxw(
                    "-540d2873ad84cd5d08155bf49dae271ae398b9394387af56fcb99930c49cb4c9e18de50183999527",
                ),
                hxw(
                    "-6ec32430597131530e1de08c87a4540d12c705ae9d0619f8c7343a53a85069f046576d75ded3d899",
                ),
                hxw(
                    "-5cfbdf6c0330be6593925ae7278667fad59e894f8a3c140b1a3cb069e56437624bf104a3387855b1",
                ),
            ],
            [
                hxw(
                    "-540d2873ad84cd5d08155bf49dae271ae398b9394387af56fcb99930c49cb4c9e18de50183999527",
                ),
                hxw(
                    "-1d8003b4f6a19d62f621051e0818563242f1f179c83e104e4dbcfa742e831bdb866fe2b621b2ea4a",
                ),
                hxw(
                    "5cfbdf6c0330be6593925ae7278667fad59e894f8a3c140b1a3cb069e56437624bf104a3387855b1",
                ),
                hxw(
                    "-6ec32430597131530e1de08c87a4540d12c705ae9d0619f8c7343a53a85069f046576d75ded3d899",
                ),
            ],
            [
                hxw("1347be7177d9a48df7530e4d153793c921aa3989150d2b3b7"),
                hxw("-14b3c3b640620844b0672932238c00d4c50c7ed83d554f80c"),
                hxw("2ab683b5c3ba1321118667330102ee367c8053593236525bb"),
                hxw("117f0c960028438a689adba1cb10cb5f4fcfa5fc24daba377"),
            ],
            [
                hxw("-14b3c3b640620844b0672932238c00d4c50c7ed83d554f80c"),
                hxw("-1347be7177d9a48df7530e4d153793c921aa3989150d2b3b7"),
                hxw("-117f0c960028438a689adba1cb10cb5f4fcfa5fc24daba377"),
                hxw("2ab683b5c3ba1321118667330102ee367c8053593236525bb"),
            ],
        ];
        let c_gram: [[Int<WL>; 4]; 4] = [
            [
                hxw("2ea07002adabb7888bf92fe9ad9a635c"),
                hxw("0"),
                hxw("0"),
                hxw("0"),
            ],
            [
                hxw("0"),
                hxw("2ea07002adabb7888bf92fe9ad9a635c"),
                hxw("0"),
                hxw("0"),
            ],
            [
                hxw("-28448e7f01f18ff561d65bbc8de5942c"),
                hxw("1df031a2af3b06d439d7219bc3157310"),
                hxw("7b4edc922e3bc7eaf79a64df730fc6b8"),
                hxw("0"),
            ],
            [
                hxw("1df031a2af3b06d439d7219bc3157310"),
                hxw("28448e7f01f18ff561d65bbc8de5942c"),
                hxw("0"),
                hxw("7b4edc922e3bc7eaf79a64df730fc6b8"),
            ],
        ];

        for r in 0..4 {
            for c in 0..4 {
                if red[r][c] != c_red[r][c] {
                    std::eprintln!(
                        "RED MISMATCH [{r}][{c}]:\n  ours={:x}\n  C   ={:x}",
                        red[r][c],
                        c_red[r][c]
                    );
                }
                if gram[r][c] != c_gram[r][c] {
                    std::eprintln!(
                        "GRAM MISMATCH [{r}][{c}]:\n  ours={:x}\n  C   ={:x}",
                        gram[r][c],
                        c_gram[r][c]
                    );
                }
            }
        }
        assert_eq!(gram, c_gram, "reduce Gram must match C byte-for-byte");
        assert_eq!(red, c_red, "reduced basis must match C byte-for-byte");
    }

    /// BYTE-EXACT ORACLE (link 4 — the assembly): with the C-byte-exact
    /// reduced basis `red` and the C-selected box-search coord α=[51,32,48,-23]
    /// (q=0x1879c1cc…419, ctr=116), replicate the `quat_lideal_prime_norm_reduced_equivalent`
    /// tail — α=red·coord, conjugate, right-multiply the original ideal, HNF,
    /// reduce_denom — and assert the equivalent ideal `J` matches the C reference
    /// `quat_lideal_mul` output byte-for-byte (CDUMP3). PASS ⟹ the ENTIRE
    /// quaternion keygen front (create → reduce → box-search → J assembly) is
    /// byte-exact with C, so the keygen pk[0..64] divergence is downstream in
    /// the ideal→isogeny SPINE (same j(E_A), different Montgomery model). FAIL
    /// ⟹ the first differing J entry pins a conj/mul/HNF/reduce_denom bug.
    #[test]
    fn equivalent_ideal_assembly_matches_c_oracle_kat0() {
        use crate::quaternion::Quaternion;
        use crate::quaternion::hnf::{hnf_mod_core, quat_lattice_reduce_denom};
        use crate::quaternion::ideal::det_4x4;
        use crypto_bigint::{Int, Uint};
        const WL: usize = 80;

        fn hxw(s: &str) -> Int<WL> {
            let neg = s.as_bytes()[0] == b'-';
            let body = if neg { &s[1..] } else { s };
            let h = if body.len() % 2 == 1 {
                format!("0{body}")
            } else {
                body.to_string()
            };
            let nbytes = h.len() / 2;
            let mut buf = [0u8; WL * 8];
            for i in 0..nbytes {
                buf[WL * 8 - nbytes + i] = u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap();
            }
            let v = *Uint::<WL>::from_be_slice(&buf).as_int();
            if neg { v.wrapping_neg() } else { v }
        }

        // C reduced basis `red` (byte-exact, from CDUMP2).
        let red: [[Int<WL>; 4]; 4] = [
            [
                hxw(
                    "1d8003b4f6a19d62f621051e0818563242f1f179c83e104e4dbcfa742e831bdb866fe2b621b2ea4a",
                ),
                hxw(
                    "-540d2873ad84cd5d08155bf49dae271ae398b9394387af56fcb99930c49cb4c9e18de50183999527",
                ),
                hxw(
                    "-6ec32430597131530e1de08c87a4540d12c705ae9d0619f8c7343a53a85069f046576d75ded3d899",
                ),
                hxw(
                    "-5cfbdf6c0330be6593925ae7278667fad59e894f8a3c140b1a3cb069e56437624bf104a3387855b1",
                ),
            ],
            [
                hxw(
                    "-540d2873ad84cd5d08155bf49dae271ae398b9394387af56fcb99930c49cb4c9e18de50183999527",
                ),
                hxw(
                    "-1d8003b4f6a19d62f621051e0818563242f1f179c83e104e4dbcfa742e831bdb866fe2b621b2ea4a",
                ),
                hxw(
                    "5cfbdf6c0330be6593925ae7278667fad59e894f8a3c140b1a3cb069e56437624bf104a3387855b1",
                ),
                hxw(
                    "-6ec32430597131530e1de08c87a4540d12c705ae9d0619f8c7343a53a85069f046576d75ded3d899",
                ),
            ],
            [
                hxw("1347be7177d9a48df7530e4d153793c921aa3989150d2b3b7"),
                hxw("-14b3c3b640620844b0672932238c00d4c50c7ed83d554f80c"),
                hxw("2ab683b5c3ba1321118667330102ee367c8053593236525bb"),
                hxw("117f0c960028438a689adba1cb10cb5f4fcfa5fc24daba377"),
            ],
            [
                hxw("-14b3c3b640620844b0672932238c00d4c50c7ed83d554f80c"),
                hxw("-1347be7177d9a48df7530e4d153793c921aa3989150d2b3b7"),
                hxw("-117f0c960028438a689adba1cb10cb5f4fcfa5fc24daba377"),
                hxw("2ab683b5c3ba1321118667330102ee367c8053593236525bb"),
            ],
        ];
        // Original secret ideal I (byte-exact, denom 2).
        let two_n = "200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000096";
        let basis: [[Int<WL>; 4]; 4] = [
            [
                hxw(two_n),
                hxw("0"),
                hxw(
                    "1687ff65ad3acae767bc282bc38e059fe5596f303032ef89615601250aa9acafea9e87b06e4844f43b0d82e58f3a85c53e01cdc756d40858f19550e9396642e80",
                ),
                hxw(
                    "166360752f138e7dfdae0eeee063281fd7095b4d0fe5fd455b238b877788863766346231bc8879c85dd1164d79c2388d452652d6c1ed925d0f58adb23577131d1",
                ),
            ],
            [
                hxw("0"),
                hxw(two_n),
                hxw(
                    "99c9f8ad0ec71820251f1111f9cd7e028f6a4b2f01a02baa4dc7478887779c899cb9dce43778637a22ee9b2863dc772bad9ad293e126da2f0a7524dca88ecec5",
                ),
                hxw(
                    "1687ff65ad3acae767bc282bc38e059fe5596f303032ef89615601250aa9acafea9e87b06e4844f43b0d82e58f3a85c53e01cdc756d40858f19550e9396642e80",
                ),
            ],
            [hxw("0"), hxw("0"), hxw("1"), hxw("0")],
            [hxw("0"), hxw("0"), hxw("0"), hxw("1")],
        ];
        let denom = hxw("2");
        let norm_n = {
            let v = hxw(
                "10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004b",
            );
            // norm as Uint for uint_as_nonneg_int path: rebuild via abs.
            v
        };
        let p = crate::params::lvl1::prime().resize::<WL>();

        // α = red · coord (the box-search element), coord = C's selection.
        let coord = [hxw("33"), hxw("20"), hxw("30"), hxw("-17")];
        let alpha = mat4_eval::<WL>(&red, &coord);

        // ᾱ = conj(α); J = HNF(I · ᾱ) / reduce_denom, denom = denom²·N.
        let alpha_bar = Quaternion::<WL>::new(
            alpha[0],
            alpha[1].wrapping_neg(),
            alpha[2].wrapping_neg(),
            alpha[3].wrapping_neg(),
        );
        let mut prod = [[Int::<WL>::from_i64(0); 4]; 4];
        for j in 0..4 {
            let col = Quaternion::<WL>::new(basis[0][j], basis[1][j], basis[2][j], basis[3][j]);
            let pr = col.mul(&alpha_bar, &p);
            prod[0][j] = pr.a;
            prod[1][j] = pr.b;
            prod[2][j] = pr.c;
            prod[3][j] = pr.d;
        }
        let alpha_denom = denom.wrapping_mul(&norm_n);
        let prod_denom = denom.wrapping_mul(&alpha_denom);
        let modulus = det_4x4::<WL>(&prod).abs();
        let gens: [[Int<WL>; 4]; 4] = [
            [prod[0][0], prod[1][0], prod[2][0], prod[3][0]],
            [prod[0][1], prod[1][1], prod[2][1], prod[3][1]],
            [prod[0][2], prod[1][2], prod[2][2], prod[3][2]],
            [prod[0][3], prod[1][3], prod[2][3], prod[3][3]],
        ];
        let hnf = hnf_mod_core::<WL>(&gens, &modulus);
        let (j_basis, j_denom) = quat_lattice_reduce_denom::<WL>(&hnf, &prod_denom);

        let c_j: [[Int<WL>; 4]; 4] = [
            [
                hxw("30f38398cd2922eb760a48ab7b62c632832"),
                hxw("0"),
                hxw("2f3a5e8a52b1c10381c8d7720ccaa750d6c"),
                hxw("4e85e98cabf8e644f51e68a21d6a01be23"),
            ],
            [
                hxw("0"),
                hxw("30f38398cd2922eb760a48ab7b62c632832"),
                hxw("2c0b25000269948726b86221598c2616a0f"),
                hxw("2f3a5e8a52b1c10381c8d7720ccaa750d6c"),
            ],
            [hxw("0"), hxw("0"), hxw("1"), hxw("0")],
            [hxw("0"), hxw("0"), hxw("0"), hxw("1")],
        ];
        for r in 0..4 {
            for c in 0..4 {
                if j_basis[r][c] != c_j[r][c] {
                    std::eprintln!(
                        "J MISMATCH [{r}][{c}]:\n  ours={:x}\n  C   ={:x}",
                        j_basis[r][c],
                        c_j[r][c]
                    );
                }
            }
        }
        assert_eq!(j_denom, hxw("2"), "J denom must match C (2)");
        assert_eq!(
            j_basis, c_j,
            "equivalent ideal J must match C byte-for-byte"
        );
    }
}
