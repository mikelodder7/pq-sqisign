// SPDX-License-Identifier: MIT OR Apache-2.0
//! Multiplication of `O_0`-basis-coordinate elements.
//!
//! `O_0 = ⟨1, i, (i+j)/2, (1+k)/2⟩` has *fractional* basis vectors in the
//! standard `(1, i, j, k)` basis. Doing arithmetic on `O_0` elements while
//! staying in integer coordinates requires the doubling trick:
//!
//! Let `x = a·1 + b·i + c·(i+j)/2 + d·(1+k)/2` with `(a, b, c, d) ∈ Z^4`.
//! Then `2·x` has *integer* standard-basis coordinates:
//!
//! ```text
//!     2·x = (2a + d, 2b + c, c, d)
//! ```
//!
//! Given two `O_0` elements `x`, `y`, the product `x · y ∈ O_0`. Computing
//! `2x · 2y = 4(xy)` keeps us entirely in integer-standard-basis territory,
//! and the recovery to `O_0` coordinates is:
//!
//! ```text
//!     standard coords (s_0, s_1, s_2, s_3) of (xy) satisfy
//!         s_0 = a' + d'/2, s_1 = b' + c'/2, s_2 = c'/2, s_3 = d'/2,
//!     so given T = 4(xy) with coords (T_0, T_1, T_2, T_3):
//!         o_0(xy) = (T_0 − T_3) / 4
//!         o_1(xy) = (T_1 − T_2) / 4
//!         o_2(xy) = T_2 / 2
//!         o_3(xy) = T_3 / 2.
//! ```
//!
//! All four divisions are exact (valid `O_0` elements have these integer
//! coordinates; the construction guarantees it).

use crypto_bigint::{Int, NonZero, Uint};

use crate::quaternion::Quaternion;
use crate::quaternion::hnf::int_div_floor;
use crate::quaternion::represent_integer::uint_gcd_vartime;

/// Convert a quaternion with **integer** standard `(1, i, j, k)` coordinates
/// to its `O_0`-basis coordinates `(o_0, o_1, o_2, o_3)`.
///
/// Every integer-standard quaternion lies in `Z⟨1, i, j, k⟩ ⊆ O_0`, so this
/// conversion always succeeds. The formulas come from the inverse of the
/// `O_0`-basis change:
///
/// ```text
///     standard coords (qa, qb, qc, qd) of `x = a + b·i + c·(i+j)/2 + d·(1+k)/2`:
///         qa = a + d/2,  qb = b + c/2,  qc = c/2,  qd = d/2
///     hence for integer standard inputs:
///         o_3 = 2·qd,  o_2 = 2·qc,  o_1 = qb − qc,  o_0 = qa − qd.
/// ```
pub fn standard_to_o0_basis<const LIMBS: usize>(q: &Quaternion<LIMBS>) -> [Int<LIMBS>; 4] {
    let two = Int::<LIMBS>::from_i64(2);
    [
        q.a.wrapping_sub(&q.d),
        q.b.wrapping_sub(&q.c),
        two.wrapping_mul(&q.c),
        two.wrapping_mul(&q.d),
    ]
}

/// Convert from `O_0`-basis coords to *doubled* standard coords —
/// i.e., the standard `(1, i, j, k)` coords of `2·x` (which are integer
/// even when `x`'s own coords would be half-integer). This is the
/// canonical integer-arithmetic representation of an `O_0` element when
/// you need to interact with the `(1, i, j, k)` side of the algebra.
pub fn o0_basis_to_standard_doubled<const LIMBS: usize>(
    coords: &[Int<LIMBS>; 4],
) -> Quaternion<LIMBS> {
    let two = Int::<LIMBS>::from_i64(2);
    Quaternion::<LIMBS>::new(
        two.wrapping_mul(&coords[0]).wrapping_add(&coords[3]),
        two.wrapping_mul(&coords[1]).wrapping_add(&coords[2]),
        coords[2],
        coords[3],
    )
}

/// Conjugate `γ̄` of an `O_0` element expressed in `O_0`-basis coordinates.
///
/// Derivation: with `γ` having standard coords `(a + d/2, b + c/2, c/2, d/2)`,
/// `γ̄` has standard coords `(a + d/2, −b − c/2, −c/2, −d/2)`. Inverting back
/// to `O_0` coords via `o_3 = 2·qd, o_2 = 2·qc, o_1 = qb − qc, o_0 = qa − qd`:
///
/// ```text
///     (a, b, c, d) ↦ (a + d, −b, −c, −d).
/// ```
pub fn o0_conjugate<const LIMBS: usize>(coords: &[Int<LIMBS>; 4]) -> [Int<LIMBS>; 4] {
    [
        coords[0].wrapping_add(&coords[3]),
        coords[1].wrapping_neg(),
        coords[2].wrapping_neg(),
        coords[3].wrapping_neg(),
    ]
}

/// Convert an ideal basis from our representation — `O_0`-coords,
/// ROW-major (`basis_o0[r]` = the `O_0`-coords of generator `r`) — to the
/// SQIsign C reference representation: standard `(1,i,j,k)`-coords,
/// COLUMN-major (`out[k][r]` = coordinate `k` of generator `r`), holding
/// the DOUBLED generators `2·g_r` (integer standard coords). The matching
/// denominator is `2·denom` (the doubling), so `column_r / (2·denom)` is the
/// same quaternion as our `generator_r / denom`. This is the boundary
/// adapter to the C-faithful `lattice_gram` / `quat_lll_core`, which read
/// column vectors in standard coords.
#[allow(dead_code, clippy::needless_range_loop)]
pub fn ideal_basis_o0_to_standard_col<const LIMBS: usize>(
    basis_o0: &[[Int<LIMBS>; 4]; 4],
) -> [[Int<LIMBS>; 4]; 4] {
    let mut out = [[Int::<LIMBS>::from_i64(0); 4]; 4];
    for r in 0..4 {
        let q = o0_basis_to_standard_doubled::<LIMBS>(&basis_o0[r]);
        let coords = [q.a, q.b, q.c, q.d];
        for k in 0..4 {
            out[k][r] = coords[k];
        }
    }
    out
}

/// Inverse of [`ideal_basis_o0_to_standard_col`]: convert a COLUMN-major
/// standard-coords basis holding the doubled generators (`std_col[k][r]` =
/// coord `k` of `2·g_r`) back to our ROW-major `O_0`-coords basis
/// (`out[r]` = `O_0`-coords of `g_r`). The internal `÷2` is exact because
/// `standard_to_o0_basis(2·g)` yields `2·(O_0-coords of g)`. Used to hand a
/// C-representation reduced ideal back to our (already-validated) spine.
#[allow(dead_code, clippy::needless_range_loop)]
pub fn ideal_basis_standard_col_to_o0<const LIMBS: usize>(
    std_col: &[[Int<LIMBS>; 4]; 4],
) -> [[Int<LIMBS>; 4]; 4] {
    let mut out = [[Int::<LIMBS>::from_i64(0); 4]; 4];
    for r in 0..4 {
        let q =
            Quaternion::<LIMBS>::new(std_col[0][r], std_col[1][r], std_col[2][r], std_col[3][r]);
        let doubled_o0 = standard_to_o0_basis::<LIMBS>(&q); // = 2·(O_0 coords of g_r)
        for k in 0..4 {
            debug_assert!(
                doubled_o0[k].abs().as_words()[0] & 1 == 0,
                "doubled O_0 coord must be even (exact halving)",
            );
            out[r][k] = doubled_o0[k].shr_vartime(1); // ÷2, exact (even)
        }
    }
    out
}

/// Build the COLUMN-major standard-coords basis of the principal left
/// ideal `O_0 · gen` (pre-HNF) — the first half of the C
/// `quat_lideal_create_principal` (`quat_lattice_alg_elem_mul`).
///
/// The maximal order `O_0 = ⟨1, i, (i+j)/2, (1+k)/2⟩` is stored as its
/// DOUBLED standard `(1,i,j,k)` column basis (denom 2):
/// `col0 = 2·1`, `col1 = 2·i`, `col2 = i+j`, `col3 = 1+k`. Each column
/// (an order element) is right-multiplied by `gen` via the standard
/// quaternion product, giving the `O_0·gen` columns in standard coords
/// (`out[k][j]` = coord `k` of column `j`). The denominator is unchanged
/// (`2`); the HNF reduction of `O_0·gen + norm·O_0` is the next step.
///
/// Right-multiplication by `gen` scales the lattice covolume by
/// `N(gen)²`, so `|det(out)| = |det(O_0 basis)|·N(gen)² = 4·N(gen)²`.
#[allow(dead_code, clippy::needless_range_loop)]
pub fn order_times_gen<const LIMBS: usize>(
    g: &Quaternion<LIMBS>,
    p: &Uint<LIMBS>,
) -> [[Int<LIMBS>; 4]; 4] {
    let n = |x: i64| Int::<LIMBS>::from_i64(x);
    // O_0 doubled standard column basis (denom 2).
    let order_cols = [
        Quaternion::<LIMBS>::new(n(2), n(0), n(0), n(0)), // 2·1
        Quaternion::<LIMBS>::new(n(0), n(2), n(0), n(0)), // 2·i
        Quaternion::<LIMBS>::new(n(0), n(1), n(1), n(0)), // i + j
        Quaternion::<LIMBS>::new(n(1), n(0), n(0), n(1)), // 1 + k
    ];
    let mut out = [[n(0); 4]; 4];
    for j in 0..4 {
        let prod = order_cols[j].mul(g, p); // order element · gen
        let coords = [prod.a, prod.b, prod.c, prod.d];
        for k in 0..4 {
            out[k][j] = coords[k];
        }
    }
    out
}

/// Port of the C reference `quat_lideal_create` (`ideal.c`): build the left
/// `O_0`-ideal `I = O_0·γ + N·O_0` where `γ` is the algebra element
/// `gen_a / gen_denom` (standard `(1,i,j,ij)` coords) and `N = norm_n`.
///
/// Returns `(basis, denom, ideal_norm)` in the C's representation:
/// `basis` is COLUMN-major standard-coords, the rational lattice is
/// `(1/denom)·Z⟨columns⟩`, and `ideal_norm` is the reduced quaternion ideal
/// norm `√[O_0 : I]`. When `N | N_red(γ)` and `γ` is primitive in `O_0`, the
/// ideal has reduced norm exactly `N` (the SQIsign construction).
///
/// Faithful sequence mirroring the C:
/// 1. `L1 = O_0·γ` — [`order_times_gen`] builds the doubled-`O_0` columns
///    times `gen_a` (denom `2·gen_denom`), then `quat_lattice_reduce_denom`
///    (the C `create_principal`'s reduce; the intermediate HNF is skipped
///    because `quat_lattice_add` re-HNFs and its output is canonical).
/// 2. `L2 = N·O_0` — the doubled-`O_0` standard column basis scaled by `N`,
///    denom `2` (the C `ON` lattice, deliberately un-reduced).
/// 3. `I = L1 + L2` via `quat_lattice_add`.
/// 4. norm `= √(|det(basis)| / denom⁴)` (`quat_lideal_norm` = `√[O_0 : I]`).
///
/// Because every step downstream of the rational lattices is canonical, the
/// returned `(basis, denom, norm)` is byte-exact with the C regardless of the
/// internal `O_0` representation — the same canonicity guarantee that frees
/// `quat_lattice_add` and [`hnf_mod_core`](crate::quaternion::hnf::hnf_mod_core).
///
/// **Width note**: the determinants inside `quat_lattice_add` reach
/// ≈ `(N)^4`; pick `LIMBS` wide enough (≈ 36 limbs at keygen scale).
#[allow(dead_code)]
pub fn quat_lideal_create<const LIMBS: usize>(
    gen_a: &Quaternion<LIMBS>,
    gen_denom: &Int<LIMBS>,
    norm_n: &Uint<LIMBS>,
    p: &Uint<LIMBS>,
) -> ([[Int<LIMBS>; 4]; 4], Int<LIMBS>, Uint<LIMBS>) {
    use crate::quaternion::hnf::{quat_lattice_add, quat_lattice_reduce_denom};
    use crate::quaternion::ideal::det_4x4;

    let n = |x: i64| Int::<LIMBS>::from_i64(x);
    let two = n(2);

    // L1 = O_0 · γ (principal), denom = 2·gen_denom, then reduce to lowest terms.
    let basis1 = order_times_gen::<LIMBS>(gen_a, p);
    let denom1 = two.wrapping_mul(gen_denom);
    let (basis1, denom1) = quat_lattice_reduce_denom::<LIMBS>(&basis1, &denom1);

    // L2 = N · O_0 : the doubled-O_0 standard COLUMN basis scaled by N, denom 2.
    // Columns: 2·1, 2·i, i+j, 1+k ⇒ basis[row][col].
    let n_int = uint_as_nonneg_int::<LIMBS>(norm_n)
        .expect("quat_lideal_create: norm_n.bits_vartime() must be < 64·LIMBS");
    let mut basis2 = [
        [n(2), n(0), n(0), n(1)],
        [n(0), n(2), n(1), n(0)],
        [n(0), n(0), n(1), n(0)],
        [n(0), n(0), n(0), n(1)],
    ];
    for row in basis2.iter_mut() {
        for cell in row.iter_mut() {
            *cell = cell.wrapping_mul(&n_int);
        }
    }
    let denom2 = two;

    // I = L1 + L2.
    let (basis, denom) = quat_lattice_add::<LIMBS>(&basis1, &denom1, &basis2, &denom2);

    // Ideal norm = √[O_0 : I], where [O_0 : I] = covol(I) / covol(O_0). In
    // standard (1, i, j, ij) coords the maximal order O_0 has covolume 1/4
    // (det of the doubled-O_0 basis is 4 over denom 2 ⇒ 4/2⁴ = 1/4), so the
    // index is 4·|det(basis)| / denom⁴.
    let det_abs = det_4x4::<LIMBS>(&basis).abs();
    let scaled_det = det_abs.wrapping_mul(&Uint::<LIMBS>::from_u64(4));
    let denom_abs = denom.abs();
    let d2 = denom_abs.wrapping_mul(&denom_abs);
    let d4 = d2.wrapping_mul(&d2);
    let d4_nz = NonZero::new(d4).expect("quat_lideal_create: denom > 0");
    let (index, _r) = scaled_det.div_rem_vartime(&d4_nz);
    let ideal_norm = index.floor_sqrt_vartime();
    (basis, denom, ideal_norm)
}

/// Bridge the C-faithful ideal representation — COLUMN-major standard
/// `(1, i, j, ij)` coords + scalar `denom`, the output of
/// [`quat_lideal_create`] and the S327
/// `quat_lideal_prime_norm_reduced_equivalent` — into the spine's
/// [`LeftIdeal`](crate::quaternion::LeftIdeal) (ROW-major `O_0`-coords). This
/// is the connective tissue between the byte-exact quaternion front and the
/// dim2id2iso isogeny spine that consumes a `LeftIdeal`.
///
/// Each column `c_r` (integer standard coords of the `r`-th generator) maps to
/// `standard_to_o0_basis(c_r)` — its `O_0`-coords (always integral, since
/// `c_r ∈ Z⟨1,i,j,ij⟩ ⊆ O_0`). The rational lattice `(1/denom)·Z⟨columns⟩` is
/// preserved by carrying `denom` onto `LeftIdeal.denom`. `cached_norm` is set
/// to the lattice-index convention `[O_0 : I] = N_red(I)² = norm²`.
///
/// The map is exact and lattice-preserving: `c_ideal_to_left_ideal` followed
/// by the spine sees the same rational lattice the C representation encoded.
#[allow(dead_code, clippy::needless_range_loop)]
pub fn c_ideal_to_left_ideal<const LIMBS: usize>(
    basis_col_std: &[[Int<LIMBS>; 4]; 4],
    denom: &Int<LIMBS>,
    norm: &Uint<LIMBS>,
) -> crate::quaternion::LeftIdeal<LIMBS> {
    let mut left_basis = [[Int::<LIMBS>::from_i64(0); 4]; 4];
    for r in 0..4 {
        let col = Quaternion::<LIMBS>::new(
            basis_col_std[0][r],
            basis_col_std[1][r],
            basis_col_std[2][r],
            basis_col_std[3][r],
        );
        left_basis[r] = standard_to_o0_basis::<LIMBS>(&col);
    }
    // Reduce to lowest terms so an INTEGRAL O_0-ideal comes back with denom 1
    // (the canonical form the spine expects). The C representation carries a
    // half-integer std-coords denom (e.g. denom 2 from the maximal order's own
    // denominator); in O_0-coords an integral ideal then has an all-even basis,
    // and `gcd(basis, denom)` divides the denom back out. Lattice-preserving,
    // so the equals-lattice / index invariants are unchanged.
    let (reduced_basis, reduced_denom) =
        crate::quaternion::hnf::quat_lattice_reduce_denom::<LIMBS>(&left_basis, denom);
    let cached_norm = norm.wrapping_mul(norm); // lattice index = N_red(I)²
    crate::quaternion::LeftIdeal::<LIMBS>::with_denom_and_norm(
        reduced_basis,
        reduced_denom.abs(),
        cached_norm,
    )
}

/// Decompose an integer-standard `O_0` element into its primitive part and
/// content — port of the C reference `quat_alg_make_primitive` specialized to
/// the standard order `O_0` (`EXTREMAL_ORDERS[0]`).
///
/// Given `θ` (integer standard coords, i.e. denominator 1) lying in `O_0`,
/// returns `(primitive, content)` where `content = gcd` of `θ`'s four
/// `O_0`-basis coordinates and `primitive[i] = coord[i] / content`. Thus
/// `θ = content · Λ·primitive` (`Λ` = the `O_0` basis) and `θ / content` is
/// primitive in `O_0`. `content` is non-negative; `primitive` keeps each
/// coordinate's sign.
///
/// The endomorphism-action bridge consumes `primitive` against the
/// `ACTION_GEN2/3/4` tables and asserts `content` is odd (a property of the
/// `RepresentInteger`-derived `θ` it feeds, not enforced here).
///
/// Variable-time (Euclidean gcd + division) — quaternion-side, per the
/// SQIsign 2.0 §8 vartime convention. Panics if `θ = 0` (zero content).
#[allow(dead_code)]
pub fn quat_make_primitive_o0<const LIMBS: usize>(
    theta: &Quaternion<LIMBS>,
) -> ([Int<LIMBS>; 4], Int<LIMBS>) {
    make_primitive_from_o0_coords(&standard_to_o0_basis(theta))
}

/// Factor a vector of `O_0`-basis coordinates into its primitive part and
/// content: `content = gcd(|coords|)`, `primitive[i] = coords[i] / content`
/// (exact, sign-preserved). The content-extraction half of
/// `quat_alg_make_primitive`.
///
/// Used directly when an element already arrives in `O_0` coordinates — e.g.
/// `RepresentInteger` (`find_quaternion_in_full_order_with_norm_wide`) returns
/// `O_0`-basis coords, not standard `(1, i, j, k)` coords.
///
/// Variable-time. Panics if all coordinates are zero (zero content).
#[allow(dead_code)]
pub fn make_primitive_from_o0_coords<const LIMBS: usize>(
    coords: &[Int<LIMBS>; 4],
) -> ([Int<LIMBS>; 4], Int<LIMBS>) {
    // content = gcd(|coord_0|, …, |coord_3|)
    let mut content: Uint<LIMBS> = coords[0].abs();
    for c in &coords[1..] {
        content = uint_gcd_vartime(&content, &c.abs());
    }
    let content_nz = NonZero::new(content).expect("non-zero content (θ ≠ 0)");

    // primitive[i] = coord[i] / content (exact; sign preserved)
    let mut primitive = [Int::<LIMBS>::from_i64(0); 4];
    for i in 0..4 {
        let (q, _r) = coords[i].abs().div_rem_vartime(&content_nz);
        let qi = Int::<LIMBS>::from_words(q.to_words());
        primitive[i] = if bool::from(coords[i].is_negative()) {
            qi.wrapping_neg()
        } else {
            qi
        };
    }

    (primitive, Int::<LIMBS>::from_words(content.to_words()))
}

/// Build the principal left ideal `O_0 · γ` as a `LeftIdeal` in canonical
/// HNF form, where `γ` is given in `O_0`-basis coordinates.
///
/// Algorithm: basis vectors of `O_0 · γ` are `e_i · γ` for `i ∈ 0..4`,
/// computed via `multiply_o0_basis`, then HNF-reduced.
pub fn principal_left_ideal_from_o0<const LIMBS: usize>(
    gamma: &[Int<LIMBS>; 4],
    p: &Uint<LIMBS>,
) -> crate::quaternion::LeftIdeal<LIMBS> {
    let zero = Int::<LIMBS>::from_i64(0);
    let mut basis = [[zero; 4]; 4];
    for k in 0..4 {
        let mut e = [zero; 4];
        e[k] = Int::<LIMBS>::from_i64(1);
        basis[k] = multiply_o0_basis(&e, gamma, p);
    }
    let reduced = crate::quaternion::hnf::hnf_4x4(&basis);
    crate::quaternion::LeftIdeal::new(reduced)
}

/// Build the left `O_0`-ideal `O_0 · γ + O_0 · n` from a quaternion `γ`
/// (given in `O_0`-basis coordinates) and an integer `n`. Returns the
/// ideal in canonical HNF form with `cached_norm = n`.
///
/// **Caller invariant:** `n | N_red(γ)`. When this holds, the lattice
/// `O_0 · γ + O_0 · n` has reduced norm exactly `n`. When it does not,
/// the cached norm field still records `n` but the lattice's actual
/// norm may differ from the cached value — caller is responsible for
/// asserting divisibility before calling.
///
/// Algorithm: stack the 8 generators `e_k · γ` (k ∈ 0..4) and `n · e_k`
/// (k ∈ 0..4) as an 8×4 integer matrix, then HNF-reduce with
/// `hnf_rect_4cols`. The top 4 rows of the result form the canonical
/// upper-triangular `Z`-basis of the lattice.
pub fn left_ideal_from_element_and_integer_o0<const LIMBS: usize>(
    gamma: &[Int<LIMBS>; 4],
    n: &Uint<LIMBS>,
    p: &Uint<LIMBS>,
) -> crate::quaternion::LeftIdeal<LIMBS> {
    // Debug-mode invariant check: the lattice norm equals `n` only when
    // `n` divides N_red(gamma). Violation silently corrupts every
    // downstream consumer that reads cached_norm. In release builds this
    // is the caller's responsibility per the docstring.
    #[cfg(debug_assertions)]
    {
        use crypto_bigint::NonZero;
        let n_norm_int = reduced_norm_o0_basis::<LIMBS>(gamma, p);
        let n_norm_abs = n_norm_int.abs();
        let n_nz: NonZero<Uint<LIMBS>> =
            NonZero::new(*n).expect("left_ideal_from_element_and_integer_o0: n must be non-zero");
        let rem = n_norm_abs.rem_vartime(&n_nz);
        debug_assert_eq!(
            rem,
            Uint::<LIMBS>::from_u64(0),
            "left_ideal_from_element_and_integer_o0: caller invariant n | N_red(gamma) violated",
        );
    }

    // Top-bit precondition for the `n.as_int()` reinterpretation below.
    // If `n`'s top bit is set, `*n.as_int()` is interpreted as a negative
    // `Int<LIMBS>` and the constructed `n · e_k` rows go negative,
    // corrupting the HNF. At signing-flow scale `n` (ideal norm) is well
    // below the LIMBS ceiling; the debug_assert defends future callers.
    debug_assert!(
        n.bits_vartime()
            < 64u32 * u32::try_from(LIMBS).expect("LIMBS fits u32 at all SQIsign levels"),
        "left_ideal_from_element_and_integer_o0: n's top bit must be zero (n.bits_vartime() < 64·LIMBS)",
    );

    let zero = Int::<LIMBS>::from_i64(0);
    let one_int = Int::<LIMBS>::from_i64(1);
    let n_int = *n.as_int();
    let mut rows: [[Int<LIMBS>; 4]; 8] = [[zero; 4]; 8];
    for k in 0..4 {
        let mut e = [zero; 4];
        e[k] = one_int;
        rows[k] = multiply_o0_basis(&e, gamma, p);
    }
    for k in 0..4 {
        rows[4 + k][k] = n_int;
    }
    let h = crate::quaternion::hnf::hnf_rect_4cols::<8, LIMBS>(&rows);
    let basis = [h[0], h[1], h[2], h[3]];
    crate::quaternion::LeftIdeal::with_denom_and_norm(basis, Uint::<LIMBS>::ONE, *n)
}

/// Build the 4×4 integer Gram matrix `G_O0` such that
/// `cᵀ · G_O0 · c = 4 · N(α)` for `α ∈ O_0` expressed in the canonical
/// `O_0`-basis `(1, i, (i + j) / 2, (1 + k) / 2)` with integer
/// coordinates `c = (a, b, c, d)`.
///
/// Derivation: `α` in standard `(1, i, j, k)`-basis is
/// `(a + d/2, b + c/2, c/2, d/2)`. Reduced norm is
/// `N(α) = (a + d/2)² + (b + c/2)² + p·(c/2)² + p·(d/2)²`. Multiplying
/// by 4 to clear denominators:
///
/// ```text
///     4 · N(α) = 4a² + 4b² + (1 + p)·c² + (1 + p)·d² + 4ad + 4bc.
/// ```
///
/// As a symmetric quadratic form `cᵀ G_O0 c` (each off-diagonal entry
/// contributes twice via `G[i][j] + G[j][i]`):
///
/// ```text
///     G_O0 = [[4,    0,    0,    2 ],
///             [0,    4,    2,    0 ],
///             [0,    2,  1+p,    0 ],
///             [2,    0,    0,  1+p]]
/// ```
///
/// This is the building block for `ideal_gram_matrix(ideal, p)` which
/// pulls the form back through an ideal basis: `G_I = B · G_O0 · Bᵀ`
/// so that `vᵀ · G_I · v = 4 · N(α_v)` for `α_v = Σ_r v[r] · B[r]`.
pub fn o0_reduced_norm_gram_matrix<const LIMBS: usize>(p: &Uint<LIMBS>) -> [[Int<LIMBS>; 4]; 4] {
    let zero = Int::<LIMBS>::from_i64(0);
    let two = Int::<LIMBS>::from_i64(2);
    let four = Int::<LIMBS>::from_i64(4);
    let one = Int::<LIMBS>::from_i64(1);
    // Safe-reinterpret p as a non-negative Int<LIMBS>. Precondition: p's
    // top bit is zero (structurally true at all SQIsign production LIMBS).
    let p_int = uint_as_nonneg_int::<LIMBS>(p)
        .expect("o0_reduced_norm_gram_matrix: p.bits_vartime() must be < 64·LIMBS");
    let one_plus_p = one.wrapping_add(&p_int);
    [
        [four, zero, zero, two],
        [zero, four, two, zero],
        [zero, two, one_plus_p, zero],
        [two, zero, zero, one_plus_p],
    ]
}

/// Safely reinterpret a `Uint<LIMBS>` as a non-negative `Int<LIMBS>`.
///
/// Returns `None` when `u`'s top bit (bit `64·LIMBS − 1`) is set — i.e.
/// `*u.as_int()` would interpret the value as a NEGATIVE `Int<LIMBS>`.
/// This guard prevents the silent sign-flip trap that Forge S184 M3
/// caught for `uint_inv_mod_vartime` and that S188's security review
/// surfaced at multiple sites across `algebra.rs`, `o0_mul.rs`, and
/// `represent_integer.rs`.
///
/// **Use this helper at every Uint→Int reinterpretation** rather than
/// bare `*u.as_int()`. Callers in cryptographic-hot paths can use
/// `.expect("precondition: u.bits < 64*LIMBS")` when the precondition
/// is structurally true; callers in fallible API paths can propagate
/// `None` via `?`.
pub(crate) fn uint_as_nonneg_int<const LIMBS: usize>(u: &Uint<LIMBS>) -> Option<Int<LIMBS>> {
    if LIMBS == 0 {
        // Degenerate but well-defined: zero-width Uint is zero, reinterprets
        // as zero Int. Defensive against const-generic edge cases.
        return Some(Int::<LIMBS>::from_i64(0));
    }
    let words = u.to_words();
    if (words[LIMBS - 1] >> 63) & 1 == 1 {
        return None;
    }
    Some(Int::<LIMBS>::from_words(words))
}

/// Widen a signed `Int<NARROW>` to `Int<WIDE>` by sign-extension.
/// Decomposes via `abs_sign`, resizes the unsigned magnitude, then
/// re-applies the original sign. Used by [`reduced_norm_o0_basis_wide`]
/// and other wide-Int verification paths that need to compute on
/// magnitudes that would overflow the narrow type.
fn widen_int<const NARROW: usize, const WIDE: usize>(x: &Int<NARROW>) -> Int<WIDE> {
    let (uint_n, neg) = x.abs_sign();
    let uint_w: Uint<WIDE> = uint_n.resize::<WIDE>();
    let int_w: Int<WIDE> = *uint_w.as_int();
    if bool::from(neg) {
        int_w.wrapping_neg()
    } else {
        int_w
    }
}

/// Wide-Int variant of [`reduced_norm_o0_basis`] for use as a genuinely
/// independent verification path at magnitudes where the narrow `Int<N>`
/// arithmetic would overflow.
///
/// Takes the `O_0`-basis coordinates and prime as narrow types
/// (`Int<NARROW>` / `Uint<NARROW>`), widens them to `Int<WIDE>` /
/// `Uint<WIDE>` via the private `widen_int` helper / `Uint::resize`,
/// then computes the reduced norm in `WIDE` precision. Returns
/// `Int<WIDE>`.
///
/// At `NARROW = WIDE` this reduces to [`reduced_norm_o0_basis`] with an
/// extra widen round-trip — useful as a parity check. For
/// `WIDE > NARROW` (typically `WIDE = 2·NARROW`), this is the
/// verification path KLPT tests at L1 large-γ and L3/L5 scale require.
pub fn reduced_norm_o0_basis_wide<const NARROW: usize, const WIDE: usize>(
    coords: &[Int<NARROW>; 4],
    p: &Uint<NARROW>,
) -> Int<WIDE> {
    let two = Int::<WIDE>::from_i64(2);
    let four = Int::<WIDE>::from_i64(4);
    let c0 = widen_int::<NARROW, WIDE>(&coords[0]);
    let c1 = widen_int::<NARROW, WIDE>(&coords[1]);
    let c2 = widen_int::<NARROW, WIDE>(&coords[2]);
    let c3 = widen_int::<NARROW, WIDE>(&coords[3]);
    let p_wide: Uint<WIDE> = p.resize::<WIDE>();
    // Safe-reinterpret p_wide as a non-negative Int<WIDE>. The widened
    // prime has its top bits cleared by the resize (NARROW < WIDE means
    // the high LIMBS-NARROW words are zero), so the precondition is
    // structurally satisfied when NARROW <= WIDE. The helper's check is
    // defensive against future NARROW > WIDE callers (which would be a
    // type-level shrink).
    let p_int = uint_as_nonneg_int::<WIDE>(&p_wide)
        .expect("reduced_norm_o0_basis_wide: p_wide.bits_vartime() must be < 64·WIDE");

    // 2·x in standard (1, i, j, k) basis = (2a + d, 2b + c, c, d).
    let qa = two.wrapping_mul(&c0).wrapping_add(&c3);
    let qb = two.wrapping_mul(&c1).wrapping_add(&c2);
    let qc = c2;
    let qd = c3;

    // N_red(2·x) = qa² + qb² + p · (qc² + qd²)
    let qa_sq = qa.wrapping_mul(&qa);
    let qb_sq = qb.wrapping_mul(&qb);
    let qc_sq = qc.wrapping_mul(&qc);
    let qd_sq = qd.wrapping_mul(&qd);
    let n_two_x = qa_sq
        .wrapping_add(&qb_sq)
        .wrapping_add(&p_int.wrapping_mul(&qc_sq.wrapping_add(&qd_sq)));

    // N_red(x) = N_red(2·x) / 4 (exact for valid O_0 elements).
    int_div_floor(&n_two_x, &four)
}

/// Reduced norm `N_red(x) = x · x̄ ∈ Z` of an `O_0` element expressed in
/// `O_0`-basis coordinates `(a, b, c, d)` for the canonical basis
/// `(1, i, (i+j)/2, (1+k)/2)`.
///
/// Uses the `2·x` standard-basis trick to stay in integer arithmetic:
/// `N_red(2x) = (2a+d)² + (2b+c)² + p · (c² + d²)`, then `N_red(x) =
/// N_red(2x) / 4` (exact division for valid `O_0` elements).
pub fn reduced_norm_o0_basis<const LIMBS: usize>(
    coords: &[Int<LIMBS>; 4],
    p: &Uint<LIMBS>,
) -> Int<LIMBS> {
    let two = Int::<LIMBS>::from_i64(2);
    let four = Int::<LIMBS>::from_i64(4);
    let qa = two.wrapping_mul(&coords[0]).wrapping_add(&coords[3]);
    let qb = two.wrapping_mul(&coords[1]).wrapping_add(&coords[2]);
    let qc = coords[2];
    let qd = coords[3];
    let q = Quaternion::<LIMBS>::new(qa, qb, qc, qd);
    let n_two_x = q.norm(p);
    int_div_floor(&n_two_x, &four)
}

/// Wide-Int variant of [`multiply_o0_basis`]. Widens narrow inputs to
/// `Int<WIDE>`, performs the multiplication at `WIDE` precision (where
/// intermediates like `p·c·d` can grow to `O(p³)` without overflow),
/// then narrows the result back to `Int<NARROW>`.
///
/// **Caller invariant**: the *final* product components must fit in
/// `Int<NARROW>`. For ideal-times-α products where α components are
/// bounded, the final basis entries stay bounded too; only the
/// intermediates exceed narrow precision. For inputs where the final
/// result also overflows narrow, the truncating narrow step silently
/// loses information — that case needs a wider downstream type.
pub fn multiply_o0_basis_wide<const NARROW: usize, const WIDE: usize>(
    x_o0: &[Int<NARROW>; 4],
    y_o0: &[Int<NARROW>; 4],
    p: &Uint<NARROW>,
) -> [Int<NARROW>; 4] {
    let mut x_w = [Int::<WIDE>::from_i64(0); 4];
    let mut y_w = [Int::<WIDE>::from_i64(0); 4];
    for i in 0..4 {
        x_w[i] = crate::quaternion::lattice::widen_int_lattice::<NARROW, WIDE>(&x_o0[i]);
        y_w[i] = crate::quaternion::lattice::widen_int_lattice::<NARROW, WIDE>(&y_o0[i]);
    }
    let p_w: Uint<WIDE> = p.resize::<WIDE>();
    let result_w = multiply_o0_basis::<WIDE>(&x_w, &y_w, &p_w);
    let mut result = [Int::<NARROW>::from_i64(0); 4];
    for i in 0..4 {
        result[i] = crate::quaternion::lattice::narrow_int_lattice::<WIDE, NARROW>(&result_w[i]);
    }
    result
}

/// Multiply two `O_0` elements expressed in `O_0`-basis coordinates.
///
/// `x_o0`, `y_o0` carry the integer coordinates `(a, b, c, d)` in
/// `(1, i, (i+j)/2, (1+k)/2)` order. `p` is the level's prime
/// (`B_{p,∞}` ramifies at `p` and `∞`).
///
/// Returns the `O_0`-basis coordinates of `x · y`.
pub fn multiply_o0_basis<const LIMBS: usize>(
    x_o0: &[Int<LIMBS>; 4],
    y_o0: &[Int<LIMBS>; 4],
    p: &Uint<LIMBS>,
) -> [Int<LIMBS>; 4] {
    let two = Int::<LIMBS>::from_i64(2);
    let four = Int::<LIMBS>::from_i64(4);

    // 2·x in standard (1, i, j, k) basis = (2a + d, 2b + c, c, d).
    let qa_x = two.wrapping_mul(&x_o0[0]).wrapping_add(&x_o0[3]);
    let qb_x = two.wrapping_mul(&x_o0[1]).wrapping_add(&x_o0[2]);
    let qc_x = x_o0[2];
    let qd_x = x_o0[3];
    let x_std = Quaternion::<LIMBS>::new(qa_x, qb_x, qc_x, qd_x);

    let qa_y = two.wrapping_mul(&y_o0[0]).wrapping_add(&y_o0[3]);
    let qb_y = two.wrapping_mul(&y_o0[1]).wrapping_add(&y_o0[2]);
    let qc_y = y_o0[2];
    let qd_y = y_o0[3];
    let y_std = Quaternion::<LIMBS>::new(qa_y, qb_y, qc_y, qd_y);

    // T = (2x)(2y) = 4(xy) in standard basis.
    let t = x_std.mul(&y_std, p);

    // Recover O_0 coords of (xy) — all divisions are exact for valid inputs.
    let o0 = int_div_floor(&t.a.wrapping_sub(&t.d), &four);
    let o1 = int_div_floor(&t.b.wrapping_sub(&t.c), &four);
    let o2 = int_div_floor(&t.c, &two);
    let o3 = int_div_floor(&t.d, &two);
    [o0, o1, o2, o3]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip: our O_0-row-major ideal basis → C standard-column-major
    /// (doubled) → back must be the identity. Also checks the forward map
    /// produces even doubled-O_0 coords (exact halving), and that |det| of
    /// the standard-column basis is 16× our basis's (4 generators each
    /// doubled, det scales by 2^4).
    #[test]
    fn ideal_basis_o0_standard_round_trip() {
        use crate::quaternion::ideal::det_4x4;
        let n = |x: i64| Int::<8>::from_i64(x);
        // An arbitrary integer O_0-coords basis (row-major: row r = gen r).
        let basis_o0: [[Int<8>; 4]; 4] = [
            [n(3), n(1), n(0), n(-2)],
            [n(7), n(0), n(4), n(0)],
            [n(0), n(1), n(5), n(0)],
            [n(1), n(-6), n(0), n(3)],
        ];
        let std_col = ideal_basis_o0_to_standard_col::<8>(&basis_o0);
        let back = ideal_basis_standard_col_to_o0::<8>(&std_col);
        assert_eq!(back, basis_o0, "o0 → standard-col → o0 must round-trip");

        // det relationship: std_col = M · basis_o0ᵀ where M is the
        // O_0→standard-doubled linear map, an upper-triangular matrix with
        // diagonal (2,2,1,1) ⇒ |det(M)| = 4. So |det(std_col)| = 4·|det(o0)|.
        let det_o0 = det_4x4::<8>(&basis_o0).abs();
        let det_std = det_4x4::<8>(&std_col).abs();
        let four = Uint::<8>::from_u64(4);
        assert_eq!(
            det_std,
            det_o0.wrapping_mul(&four),
            "|det(standard-col)| must be 4·|det(o0)| (det of the O_0→standard map)",
        );
    }

    /// `order_times_gen` builds the `O_0·gen` column basis; right-multiply by
    /// `gen` scales the covolume by `N(gen)²`, so `|det| = 4·N(gen)²`
    /// (|det(O_0 standard basis)| = 4).
    #[test]
    fn order_times_gen_det_is_four_n_squared() {
        use crate::quaternion::ideal::det_4x4;
        let n = |x: i64| Int::<8>::from_i64(x);
        let p = Uint::<8>::from_u64(7);
        let g = Quaternion::<8>::new(n(3), n(1), n(2), n(1));
        let basis = order_times_gen::<8>(&g, &p);
        // N(gen) = 3² + 1² + 7·(2² + 1²) = 10 + 35 = 45.
        let n_gen = g.norm(&p).abs();
        assert_eq!(n_gen, Uint::<8>::from_u64(45));
        let det = det_4x4::<8>(&basis).abs();
        let four = Uint::<8>::from_u64(4);
        let expected = four.wrapping_mul(&n_gen).wrapping_mul(&n_gen); // 4·N²
        assert_eq!(det, expected, "|det(O_0·gen)| must be 4·N(gen)²");
    }

    /// `quat_lideal_create` builds `I = O_0·γ + N·O_0`. Independent invariants
    /// (mathematical lattice facts, NOT the implementation's formula): when
    /// `N | N_red(γ)` with `γ` primitive in `O_0`, the reduced ideal norm is
    /// exactly `N`, and the lattice covolume `|det(basis)| / denom⁴ = N²`.
    #[test]
    fn quat_lideal_create_norm_and_covolume() {
        use crate::quaternion::ideal::det_4x4;
        let nn = |x: i64| Int::<8>::from_i64(x);
        let p = Uint::<8>::from_u64(7);
        // γ = 1 + 3i (standard); O_0 coords (1,3,0,0) primitive; N_red = 1+9 = 10.
        let g = Quaternion::<8>::new(nn(1), nn(3), nn(0), nn(0));
        assert_eq!(g.norm(&p).abs(), Uint::<8>::from_u64(10));

        for (n_div, expected_norm) in [(2u64, 2u64), (5, 5), (10, 10)] {
            let (basis, denom, ideal_norm) =
                quat_lideal_create::<8>(&g, &nn(1), &Uint::<8>::from_u64(n_div), &p);
            assert_eq!(
                ideal_norm,
                Uint::<8>::from_u64(expected_norm),
                "reduced ideal norm must equal N for N={n_div}",
            );
            // Index [O_0 : I] = 4·|det(basis)| / denom⁴ must equal N²
            // (the O_0 covolume in standard coords is 1/4).
            let det_abs = det_4x4::<8>(&basis).abs();
            let scaled = det_abs.wrapping_mul(&Uint::<8>::from_u64(4));
            let d = denom.abs();
            let d4 = d.wrapping_mul(&d).wrapping_mul(&d).wrapping_mul(&d);
            let index = scaled.div_rem_vartime(&NonZero::new(d4).unwrap()).0;
            let n_sq = Uint::<8>::from_u64(expected_norm * expected_norm);
            assert_eq!(index, n_sq, "[O_0 : I] must equal N² for N={n_div}");
        }
    }

    /// `c_ideal_to_left_ideal` bridges the C representation (col-major std
    /// coords + denom, from `quat_lideal_create`) into the spine's `LeftIdeal`
    /// (O_0-coords). INDEPENDENT cross-check: the bridged ideal must be the
    /// SAME lattice as the one built directly in O_0-coords by the existing
    /// `left_ideal_from_element_and_integer_o0` — two independent constructions
    /// of `O_0·γ + N·O_0` agreeing. Also: the bridged O_0-basis index
    /// `|det| / denom⁴` (O_0 is the identity in O_0-coords) equals N².
    #[test]
    fn c_ideal_to_left_ideal_matches_o0_construction() {
        use crate::quaternion::ideal::det_4x4;
        let nn = |x: i64| Int::<8>::from_i64(x);
        let p = Uint::<8>::from_u64(7);
        // γ = 1 + 3i, N_red(γ) = 10; N ∈ {2,5,10} all divide 10.
        let g = Quaternion::<8>::new(nn(1), nn(3), nn(0), nn(0));
        let g_o0 = standard_to_o0_basis::<8>(&g);
        for n_div in [2u64, 5, 10] {
            let (basis, denom, norm) =
                quat_lideal_create::<8>(&g, &nn(1), &Uint::<8>::from_u64(n_div), &p);
            let bridged = c_ideal_to_left_ideal::<8>(&basis, &denom, &norm);

            // Independent construction directly in O_0-coords.
            let direct =
                left_ideal_from_element_and_integer_o0::<8>(&g_o0, &Uint::<8>::from_u64(n_div), &p);
            assert!(
                bridged.equals_lattice(&direct),
                "bridge must yield the same lattice as the O_0 construction (N={n_div})",
            );

            // Independent index: [O_0 : I] = |det(O_0-basis)| / denom⁴ = N²
            // (O_0 is the identity in O_0-coords, so no covolume factor).
            let det_abs = det_4x4::<8>(&bridged.basis).abs();
            let d = bridged.denom;
            let d4 = d.wrapping_mul(&d).wrapping_mul(&d).wrapping_mul(&d);
            let index = det_abs
                .div_rem_vartime(&NonZero::new(d4).expect("denom > 0"))
                .0;
            assert_eq!(
                index,
                Uint::<8>::from_u64(n_div * n_div),
                "[O_0 : bridged] must equal N² (N={n_div})",
            );
        }
    }

    // ── uint_as_nonneg_int unit tests (S189) ───────────────────────────

    #[test]
    fn uint_as_nonneg_int_accepts_small_values() {
        assert_eq!(
            uint_as_nonneg_int::<8>(&Uint::from_u64(0)),
            Some(Int::<8>::from_i64(0)),
        );
        assert_eq!(
            uint_as_nonneg_int::<8>(&Uint::from_u64(1)),
            Some(Int::<8>::from_i64(1)),
        );
        assert_eq!(
            uint_as_nonneg_int::<8>(&Uint::from_u64(42)),
            Some(Int::<8>::from_i64(42)),
        );
    }

    #[test]
    fn uint_as_nonneg_int_accepts_top_bit_clear() {
        // Uint::<8>::MAX has top bit set → should reject. But a value
        // with bit (64·8 - 2) set and bit 511 clear → should accept.
        let mut words = [0u64; 8];
        words[7] = 1u64 << 62; // bit 510 set; bit 511 clear
        let u: Uint<8> = Uint::from_words(words);
        let result = uint_as_nonneg_int::<8>(&u);
        assert!(result.is_some(), "top-bit-clear value must accept");
    }

    #[test]
    fn uint_as_nonneg_int_rejects_top_bit_set() {
        // Uint::MAX has top bit set → reject.
        assert_eq!(uint_as_nonneg_int::<8>(&Uint::<8>::MAX), None);

        // Just the top bit alone → reject.
        let mut words = [0u64; 8];
        words[7] = 1u64 << 63;
        let u: Uint<8> = Uint::from_words(words);
        assert_eq!(uint_as_nonneg_int::<8>(&u), None);
    }

    #[test]
    fn uint_as_nonneg_int_works_at_l1_l3_l5_widths() {
        // Reasonable real-prime-scale values: L1 (p ≈ 2^248), L3 (p ≈ 2^383),
        // L5 (p ≈ 2^505). All have top bit clear at production LIMBS.
        let p_l1: Uint<8> = crate::params::lvl1::prime().resize::<8>();
        let p_l3: Uint<12> = crate::params::lvl3::prime().resize::<12>();
        let p_l5: Uint<16> = crate::params::lvl5::prime().resize::<16>();
        assert!(uint_as_nonneg_int::<8>(&p_l1).is_some());
        assert!(uint_as_nonneg_int::<12>(&p_l3).is_some());
        assert!(uint_as_nonneg_int::<16>(&p_l5).is_some());
    }

    fn n(v: i64) -> Int<8> {
        Int::<8>::from_i64(v)
    }

    fn fake_p() -> Uint<8> {
        Uint::<8>::from_u64(7)
    }

    /// `e_0 = 1`; `1 · 1 = 1`.
    #[test]
    fn one_squared_is_one() {
        let e0 = [n(1), n(0), n(0), n(0)];
        let r = multiply_o0_basis(&e0, &e0, &fake_p());
        assert_eq!(r, e0);
    }

    /// `e_1 = i`; `i · i = −1 = −e_0`.
    #[test]
    fn i_squared_is_minus_one() {
        let e1 = [n(0), n(1), n(0), n(0)];
        let r = multiply_o0_basis(&e1, &e1, &fake_p());
        assert_eq!(r, [n(-1), n(0), n(0), n(0)]);
    }

    /// `e_0 · e_1 = i = e_1`.
    #[test]
    fn one_times_i_is_i() {
        let e0 = [n(1), n(0), n(0), n(0)];
        let e1 = [n(0), n(1), n(0), n(0)];
        let r = multiply_o0_basis(&e0, &e1, &fake_p());
        assert_eq!(r, e1);
    }

    /// `e_3 · e_3 = ((1+k)/2)² = (1 + 2k + k²)/4 = (1 + 2k − p)/4`.
    /// For `p = 7`: standard coords `((1−7)/4, 0, 0, 1/2) = (−3/2, 0, 0, 1/2)`.
    /// `O_0` coords: `o_3 = 1`, `o_2 = 0`, `o_1 = 0`, `o_0 = qa − qd = −3/2 − 1/2 = −2`.
    /// So `e_3² = (−2, 0, 0, 1)` in `O_0`-coords for `p = 7`.
    #[test]
    fn e3_squared_for_fake_p_7() {
        let e3 = [n(0), n(0), n(0), n(1)];
        let r = multiply_o0_basis(&e3, &e3, &fake_p());
        assert_eq!(r, [n(-2), n(0), n(0), n(1)]);
    }

    /// `e_2 = (i+j)/2`. `e_2² = (i + j)²/4 = (i² + 2ij + j²)/4 = (−1 + 2k − p)/4`.
    /// For `p = 7`: standard coords `(−8/4, 0, 0, 2/4) = (−2, 0, 0, 1/2)`.
    /// `O_0` coords: `o_3 = 1`, `o_2 = 0`, `o_1 = 0`, `o_0 = −2 − 1/2 = −5/2`. NOT INTEGER!
    /// So `(i+j)/2 ∉ O_0` for general `p`? Let me re-derive: `e_2²` in standard
    /// coords is `(−(1+p)/4, 0, 0, 1/2)`. For `p=7`: `(−2, 0, 0, 1/2)`. Then
    /// `o_0 = qa − qd = −2 − 1/2 = −5/2` — not integer. But `O_0` should be closed!
    ///
    /// Actually `(i+j)/2 · (i+j)/2 = (i² + ij + ji + j²)/4 = (−1 + ij − ij − p)/4
    ///   = (−1 − p)/4`. So `e_2² = (−1 − p)/4` is a pure scalar, *not* including k.
    /// For `p ≡ 3 mod 4`, `(−1 − p)/4 ∈ Z`. For p=7: `(−1 − 7)/4 = −2`. So
    /// `e_2² = −2 = −2·e_0` and `O_0` coords are `(−2, 0, 0, 0)`.
    #[test]
    fn e2_squared_for_fake_p_7() {
        let e2 = [n(0), n(0), n(1), n(0)];
        let r = multiply_o0_basis(&e2, &e2, &fake_p());
        assert_eq!(r, [n(-2), n(0), n(0), n(0)]);
    }

    /// Distributivity sanity: `(e_0 + e_1) · e_0 = e_0 · e_0 + e_1 · e_0 = 1 + i`.
    /// In `O_0` coords: `(1, 1, 0, 0)`.
    #[test]
    fn distributivity_one_plus_i() {
        let e0_plus_e1 = [n(1), n(1), n(0), n(0)];
        let e0 = [n(1), n(0), n(0), n(0)];
        let r = multiply_o0_basis(&e0_plus_e1, &e0, &fake_p());
        assert_eq!(r, e0_plus_e1);
    }

    /// Non-commutativity: `e_1 · e_2 ≠ e_2 · e_1`.
    /// `i · (i+j)/2 = (i² + ij)/2 = (−1 + k)/2`. Standard `(−1/2, 0, 0, 1/2)`.
    /// `O_0` coords: `o_3 = 1`, `o_0 = qa − qd = −1`. So `(−1, 0, 0, 1)`.
    ///
    /// `(i+j)/2 · i = (i² + ji)/2 = (−1 − k)/2`. Standard `(−1/2, 0, 0, −1/2)`.
    /// `O_0` coords: `o_3 = −1`, `o_0 = qa − qd = −1/2 − (−1/2) = 0`. So `(0, 0, 0, −1)`.
    #[test]
    fn i_times_e2_is_not_e2_times_i() {
        let e1 = [n(0), n(1), n(0), n(0)];
        let e2 = [n(0), n(0), n(1), n(0)];
        let lhs = multiply_o0_basis(&e1, &e2, &fake_p());
        let rhs = multiply_o0_basis(&e2, &e1, &fake_p());
        assert_eq!(lhs, [n(-1), n(0), n(0), n(1)]);
        assert_eq!(rhs, [n(0), n(0), n(0), n(-1)]);
        assert_ne!(lhs, rhs);
    }

    /// `e_3 = (1+k)/2`; `e_0 · e_3 = e_3`. Confirms left-identity.
    #[test]
    fn one_times_e3_is_e3() {
        let e0 = [n(1), n(0), n(0), n(0)];
        let e3 = [n(0), n(0), n(0), n(1)];
        let r = multiply_o0_basis(&e0, &e3, &fake_p());
        assert_eq!(r, e3);
    }

    /// `o0_conjugate` of `e_0` = `1` is `1`.
    #[test]
    fn conjugate_of_one_is_one() {
        let e0 = [n(1), n(0), n(0), n(0)];
        assert_eq!(o0_conjugate(&e0), e0);
    }

    /// `o0_conjugate` of `i` is `-i`.
    #[test]
    fn conjugate_of_i_is_negative_i() {
        let e1 = [n(0), n(1), n(0), n(0)];
        let conj = o0_conjugate(&e1);
        assert_eq!(conj, [n(0), n(-1), n(0), n(0)]);
    }

    /// `o0_conjugate` of `(1+k)/2` = `(1-k)/2 = 1 - (1+k)/2 = e_0 - e_3`.
    /// So `O_0`-coords `(0, 0, 0, 1)` ↦ `(1, 0, 0, -1)`.
    #[test]
    fn conjugate_of_e3() {
        let e3 = [n(0), n(0), n(0), n(1)];
        let conj = o0_conjugate(&e3);
        assert_eq!(conj, [n(1), n(0), n(0), n(-1)]);
    }

    /// Conjugation is an involution.
    #[test]
    fn conjugate_is_involution() {
        let q = [n(3), n(-5), n(7), n(-2)];
        assert_eq!(o0_conjugate(&o0_conjugate(&q)), q);
    }

    /// `γ · γ̄ = N_red(γ)` (scalar quaternion in O_0).
    #[test]
    fn gamma_times_conj_gamma_is_norm() {
        let p = fake_p();
        let gamma = [n(3), n(-2), n(1), n(0)];
        let conj = o0_conjugate(&gamma);
        let prod = multiply_o0_basis(&gamma, &conj, &p);
        // Product should be a pure scalar in O_0: only o_0 nonzero, equal to N_red(γ).
        let norm = reduced_norm_o0_basis(&gamma, &p);
        assert_eq!(prod, [norm, n(0), n(0), n(0)]);
    }

    /// `principal_left_ideal_from_o0(1)` = `O_0` (the full order).
    #[test]
    fn principal_of_one_is_full_order() {
        let p = fake_p();
        let one = [n(1), n(0), n(0), n(0)];
        let ideal = principal_left_ideal_from_o0(&one, &p);
        let full = crate::quaternion::LeftIdeal::<8>::full_order();
        assert!(ideal.equals_lattice(&full));
        assert_eq!(ideal.norm(), Uint::<8>::from_u64(1));
    }

    /// `principal_left_ideal_from_o0(e_3)` has norm `N_red(e_3)² = 4` for p=7.
    #[test]
    fn principal_norm_is_reduced_norm_squared() {
        let p = fake_p();
        let e3 = [n(0), n(0), n(0), n(1)];
        let ideal = principal_left_ideal_from_o0(&e3, &p);
        // N_red(e_3) = 2, so norm should be 4.
        assert_eq!(ideal.norm(), Uint::<8>::from_u64(4));
    }

    /// Standard `(1, 0, 0, 0) = 1` → `O_0`-coords `(1, 0, 0, 0)`.
    #[test]
    fn standard_one_to_o0_is_e0() {
        let q = Quaternion::<8>::new(n(1), n(0), n(0), n(0));
        assert_eq!(standard_to_o0_basis(&q), [n(1), n(0), n(0), n(0)]);
    }

    /// Standard `j = (0, 0, 1, 0)` → `O_0`-coords `(0, -1, 2, 0)`.
    #[test]
    fn standard_j_to_o0() {
        let q = Quaternion::<8>::new(n(0), n(0), n(1), n(0));
        assert_eq!(standard_to_o0_basis(&q), [n(0), n(-1), n(2), n(0)]);
    }

    /// Standard `k = (0, 0, 0, 1)` → `O_0`-coords `(-1, 0, 0, 2)`.
    #[test]
    fn standard_k_to_o0() {
        let q = Quaternion::<8>::new(n(0), n(0), n(0), n(1));
        assert_eq!(standard_to_o0_basis(&q), [n(-1), n(0), n(0), n(2)]);
    }

    /// Round-trip via doubling: `O_0`-coords `→ 2·x` in standard → recover
    /// `O_0`-coords of `2·x` (which is `2·` the original, but expressed in
    /// `O_0` basis as `2·(a, b, c, d)` with a wrinkle from the basis-change).
    #[test]
    fn round_trip_via_doubling() {
        // Take γ = e_3 = (0, 0, 0, 1) in O_0 coords; doubled is 2·e_3 = (1+k)
        // with standard coords (1, 0, 0, 1). Now standard_to_o0 of that
        // gives o_0=1-1=0, o_1=0-0=0, o_2=0, o_3=2. So (0, 0, 0, 2) =
        // 2·e_3 in O_0 coords. ✓
        let gamma = [n(0), n(0), n(0), n(1)];
        let doubled = o0_basis_to_standard_doubled(&gamma);
        let recovered = standard_to_o0_basis(&doubled);
        // 2 * gamma in O_0 coords.
        assert_eq!(recovered, [n(0), n(0), n(0), n(2)]);
    }

    /// `o0_basis_to_standard_doubled(1) = (2, 0, 0, 0)` (i.e., `2·1 = 2`).
    #[test]
    fn o0_one_to_doubled_standard() {
        let one = [n(1), n(0), n(0), n(0)];
        let doubled = o0_basis_to_standard_doubled(&one);
        assert_eq!(doubled, Quaternion::<8>::new(n(2), n(0), n(0), n(0)));
    }

    /// `principal_left_ideal_from_o0(2·1)` = `2·O_0` with norm `16`.
    #[test]
    fn principal_of_two_is_doubled_order() {
        let p = fake_p();
        let two = [n(2), n(0), n(0), n(0)];
        let ideal = principal_left_ideal_from_o0(&two, &p);
        let doubled = crate::quaternion::LeftIdeal::<8>::full_order().scale(2);
        assert!(ideal.equals_lattice(&doubled));
        assert_eq!(ideal.norm(), Uint::<8>::from_u64(16));
    }

    #[test]
    fn o0_gram_has_expected_pattern_at_p_seven() {
        let p = fake_p();
        let g = o0_reduced_norm_gram_matrix(&p);
        // Diagonal entries: 4, 4, 1+p=8, 1+p=8
        assert_eq!(g[0][0], n(4));
        assert_eq!(g[1][1], n(4));
        assert_eq!(g[2][2], n(8));
        assert_eq!(g[3][3], n(8));
        // Off-diagonal cross terms: G[0][3] = G[3][0] = 2, G[1][2] = G[2][1] = 2.
        assert_eq!(g[0][3], n(2));
        assert_eq!(g[3][0], n(2));
        assert_eq!(g[1][2], n(2));
        assert_eq!(g[2][1], n(2));
        // Other entries: zero.
        for (i, j) in [
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 3),
            (2, 0),
            (2, 3),
            (3, 1),
            (3, 2),
        ] {
            assert_eq!(g[i][j], n(0), "G[{i}][{j}] should be 0");
        }
    }

    #[test]
    fn o0_gram_eval_matches_4n_alpha() {
        // For c = (1, 2, 3, 4) and p = 7, verify cᵀ G c = 4·N(α) where
        // α has O_0-coords c.
        use crate::quaternion::lattice::qf_eval_4x4;
        let p = fake_p();
        let g = o0_reduced_norm_gram_matrix(&p);
        let c: [Int<8>; 4] = [n(1), n(2), n(3), n(4)];
        let four_n_via_gram = qf_eval_4x4(&c, &g);
        let n_alpha = reduced_norm_o0_basis(&c, &p);
        let four_n_via_helper = n(4).wrapping_mul(&n_alpha);
        assert_eq!(four_n_via_gram, four_n_via_helper);
    }

    #[test]
    fn widen_int_preserves_value_for_positive_and_negative() {
        let pos = Int::<8>::from_i64(42);
        let neg = Int::<8>::from_i64(-42);
        let zero = Int::<8>::from_i64(0);

        let pos_wide: Int<16> = widen_int::<8, 16>(&pos);
        let neg_wide: Int<16> = widen_int::<8, 16>(&neg);
        let zero_wide: Int<16> = widen_int::<8, 16>(&zero);

        // Round-trip the magnitude: |pos_wide| should equal |pos| via the
        // same abs_sign decomposition.
        let (pos_w_uint, pos_w_neg) = pos_wide.abs_sign();
        assert!(!bool::from(pos_w_neg));
        assert_eq!(pos_w_uint.resize::<8>(), Uint::<8>::from_u64(42));

        let (neg_w_uint, neg_w_neg) = neg_wide.abs_sign();
        assert!(bool::from(neg_w_neg));
        assert_eq!(neg_w_uint.resize::<8>(), Uint::<8>::from_u64(42));

        assert_eq!(zero_wide, Int::<16>::from_i64(0));
    }

    #[test]
    fn reduced_norm_wide_at_same_width_matches_narrow() {
        // Parity probe: at WIDE = NARROW (no widening), the wide version
        // should agree with the narrow version on inputs that don't
        // overflow.
        let p = fake_p(); // 7
        let coords = [n(1), n(2), n(3), n(4)];
        let narrow_result = reduced_norm_o0_basis(&coords, &p);
        let wide_result: Int<8> = reduced_norm_o0_basis_wide::<8, 8>(&coords, &p);
        assert_eq!(wide_result, narrow_result);
    }

    #[test]
    fn reduced_norm_wide_8_to_16_matches_narrow_on_safe_inputs() {
        // For coords that don't overflow Int<8>, the wide-Int<16>
        // computation should give the same numeric value as the narrow
        // Int<8> path. Widen narrow → 16 and compare.
        let p = fake_p();
        let coords = [n(5), n(-3), n(2), n(-1)];
        let narrow_result = reduced_norm_o0_basis(&coords, &p);
        let wide_result: Int<16> = reduced_norm_o0_basis_wide::<8, 16>(&coords, &p);
        let narrow_widened = widen_int::<8, 16>(&narrow_result);
        assert_eq!(wide_result, narrow_widened);
    }

    /// Associativity probe: `(e_1 · e_2) · e_0 = e_1 · (e_2 · e_0)`.
    #[test]
    fn associativity_e1_e2_e0() {
        let e0 = [n(1), n(0), n(0), n(0)];
        let e1 = [n(0), n(1), n(0), n(0)];
        let e2 = [n(0), n(0), n(1), n(0)];
        let p = fake_p();
        let lhs_inner = multiply_o0_basis(&e1, &e2, &p);
        let lhs = multiply_o0_basis(&lhs_inner, &e0, &p);
        let rhs_inner = multiply_o0_basis(&e2, &e0, &p);
        let rhs = multiply_o0_basis(&e1, &rhs_inner, &p);
        assert_eq!(lhs, rhs);
    }

    /// `quat_make_primitive_o0` factors θ's O_0 coords into content·primitive.
    #[test]
    fn make_primitive_factors_content_and_primitive() {
        // θ standard coords (6, 18, 30, 42) → O_0 coords (a−d, b−c, 2c, 2d)
        //   = (6−42, 18−30, 60, 84) = (−36, −12, 60, 84); gcd = 12.
        let theta = Quaternion::new(n(6), n(18), n(30), n(42));
        let coords = standard_to_o0_basis(&theta);
        assert_eq!(coords, [n(-36), n(-12), n(60), n(84)]);

        let (primitive, content) = quat_make_primitive_o0(&theta);
        assert_eq!(content, n(12), "content = gcd(|O_0 coords|)");
        assert_eq!(
            primitive,
            [n(-3), n(-1), n(5), n(7)],
            "primitive = coords/content"
        );

        // content · primitive == coords (exact reconstruction, signs preserved)
        for i in 0..4 {
            assert_eq!(content.wrapping_mul(&primitive[i]), coords[i]);
        }
    }

    /// An already-primitive element has content 1 and is returned unchanged.
    #[test]
    fn make_primitive_on_primitive_element_is_identity() {
        // θ = 1 + i + j (standard (1,1,1,0)) → O_0 coords (1, 0, 2, 0); gcd 1.
        let theta = Quaternion::new(n(1), n(1), n(1), n(0));
        let coords = standard_to_o0_basis(&theta);
        let (primitive, content) = quat_make_primitive_o0(&theta);
        assert_eq!(content, n(1));
        assert_eq!(primitive, coords);
    }

    /// S350 BYTE-EXACT ORACLE: the C reference `quat_lideal_create` output for
    /// the FIRST keygen secret ideal under the lvl1 KAT DRBG (seed 0), captured
    /// by instrumenting `the-sqisign` `normeq.c` (CDUMP). Feeds the SAME γ and N
    /// into our `quat_lideal_create` and asserts the canonical (HNF) basis,
    /// denom, and norm match C byte-for-byte. If this fails, our HNF
    /// canonicalization diverges from C's `quat_lattice_add`/`quat_lattice_hnf`
    /// — the true keygen pk[0..64] root cause (the dpe LLL is already byte-exact,
    /// proven by `quat_lll_core_matches_c_oracle_*`). γ reaches ~1272 bits and
    /// the `quat_lattice_add` determinant ~γ⁴, so WL=96 (6144 bits) holds it.
    #[test]
    fn quat_lideal_create_matches_c_oracle_kat0() {
        use crate::quaternion::Quaternion;
        use crypto_bigint::{Int, Uint};
        const WL: usize = 96;

        // Signed hex → Int<WL> (zero-padded big-endian).
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

        // C γ (gen, standard (1,i,j,ij) coords), gen_denom = 1, N = SEC_DEGREE.
        let gamma = Quaternion::<WL>::new(
            hxw(
                "-71a7c19a1764deaa7b009c92e16b72e2c292cd8e5def16d97a2a739947f8d1c70ea1aa747c01fcbf12b053e73224299ad6bb51e1ab64b5f91b118cea21810d4a843281f094ec74e66e0dd48711efa2a7592227615cd7133dc76c331e8714f193a76185892bf13b410cb61a302f379df98f8f48ba5358de219dfc9f4e705226282828861e2f18fe18b519928a69eafd48fc8f7bae31f1143ad325b4651a51e4",
            ),
            hxw(
                "-105c3d6b64d538d5215ef67ab7d781fd0e43225857bef3fa6899ec9822a12b4d67e749226c6919061f8b82dca21b01f25a2aa8d877f857ca4b7f5cf1d1c7079bfaface99db0403393c84f717aa41dd4359efee80d9f65e9a0419dfb7c84ee81637721aaee115e802d5953c2d991fc49f953c3db95d403eeeee749ee1f8af498c83d3f00884c218ebf1f07a73619b15f76a2c9751e25a0166ca7709e35594da",
            ),
            hxw(
                "4ca03266db42fb65408e6889d7eb0a3b566c400a1f253cdfe2e9d404cf10cbb83b4a30686d142ffc7b7670576a70ba74015e89747471f2ea7d0cd1d74209f176e116c27226e33092a854c960e6fe86f52b282afe0a3725cbee2fa1ac4858b13cb8451b1e1a062eb8a4a7239594a68eb07b6312d19611db536f62de7c97c337f9",
            ),
            hxw(
                "-6381b8e97e11a9daa869e75ca9ae5c9fc1c52f6011bc411683d50120859873da778e061d87fc78a4d93d36ae790b4136d51036bba8c0a26b66b78af3b63ffaff7257eaa5947c4676a0903afd8324e818b2087fa45906f413c1a9e84943d8932f7110a13ce8b0aa3e7c2e8d0b1a1e31c415282a2eda0d87cc4602e30f3f2d0d9",
            ),
        );
        let gen_denom = Int::<WL>::from_i64(1);
        let norm_n = hxu(
            "10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000004b",
        );
        let p = crate::params::lvl1::prime().resize::<WL>();

        let (basis, denom, ideal_norm) = quat_lideal_create::<WL>(&gamma, &gen_denom, &norm_n, &p);

        // C canonical basis (column-major [row][col]), denom 2.
        let c_basis: [[Int<WL>; 4]; 4] = [
            [
                hxw(
                    "200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000096",
                ),
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
                hxw(
                    "200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000096",
                ),
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

        for r in 0..4 {
            for c in 0..4 {
                if basis[r][c] != c_basis[r][c] {
                    std::eprintln!(
                        "MISMATCH basis[{r}][{c}]:\n  ours = {:x}\n  C    = {:x}",
                        basis[r][c],
                        c_basis[r][c]
                    );
                }
            }
        }
        assert_eq!(denom, Int::<WL>::from_i64(2), "denom must match C (2)");
        assert_eq!(ideal_norm, norm_n, "ideal norm must match C (N)");
        assert_eq!(
            basis, c_basis,
            "quat_lideal_create basis must match C byte-for-byte"
        );
    }
}
