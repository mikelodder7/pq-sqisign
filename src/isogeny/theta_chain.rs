//! Theta-chain orchestration — the control-flow driver.
//!
//! This module ports the *stack machine* of the SQIsign C reference's
//! `_theta_chain_compute_impl` (`src/hd/.../theta_isogenies.c`), which walks a
//! length-`n` dimension-2 `2^n`-isogeny as a chain of 2-isogeny steps:
//!
//! 1. a **gluing** step `E₁ × E₂ → A` (the first step),
//! 2. `n − 3` (no extra torsion) or `n − 1` (extra torsion) **interior**
//!    `theta_isogeny` steps `A → A`,
//! 3. for the no-extra-torsion path, two **final** steps (`compute_4` then
//!    `compute_2`),
//! 4. a **splitting** step `A → E₃ × E₄` (the last step).
//!
//! Kernel points are gathered and pushed through each step using a
//! cost-balanced **doubling descent**: a `todo[]`/`current` stack where, at
//! each level, the number of doublings is chosen to balance "recompute later"
//! against "store the intermediate point" (the `num_dbls` rule).
//!
//! # Why a driver + visitor
//!
//! The control flow — `space`, the `todo[]` stack, the `current` pointer, the
//! `num_dbls` rule, the per-step Hadamard flags, and the kernel-push
//! bookkeeping — is **independent of the actual point values**. It depends
//! only on `(n, extra_torsion)`. That makes it both the most error-prone part
//! of the orchestration (off-by-one in the stack machine silently corrupts the
//! whole chain) and the part that can be tested *deterministically*, with no
//! field arithmetic or fixtures.
//!
//! So we separate the two concerns: [`drive_theta_chain`] owns the index logic
//! and calls a [`ChainVisitor`] for every concrete operation (descend, glue,
//! step, push, finalize, split). The point-executing visitor — which owns the
//! couple-Jacobian / theta-point stacks and threads the codomain — is wired in
//! a later session against this already-verified control flow.
//!
//! Everything here is consumed by the point-executing visitor wired in a later
//! session; until then only this module's own tests exercise it, so the lib
//! build sees the items as unused.
#![allow(dead_code)]

use crate::ec::couple::{
    CoupleCurve, CoupleJacobianPoint, CoupleMontgomeryPoint, ThetaKernelCouplePoints,
};
use crate::gf::fp::BaseField;
use crate::isogeny::gluing::{
    GluingCodomain, apply_isomorphism, gluing_codomain, gluing_eval_basis,
    gluing_eval_point_special_case,
};
use crate::isogeny::splitting::{
    splitting_compute, theta_point_to_montgomery_point, theta_product_structure_to_elliptic_product,
};
use crate::isogeny::theta::{AbelianVariety2D, ThetaPoint2D};
use crate::isogeny::theta_isogeny::{
    ThetaIsogeny, theta_isogeny_compute, theta_isogeny_compute_2, theta_isogeny_compute_4,
    theta_isogeny_eval,
};

/// Upper bound on the doubling-descent stack depth. The C reference sizes the
/// stack as `space = 1 + ⌊log₂(n−1)⌋ + 1`; for every SQIsign level (`n ≤ 248`
/// at L1, larger but well under `2^15` elsewhere) this stays `≤ 10`. `16`
/// leaves generous headroom and keeps the stack arrays fixed-size (no `alloc`).
pub(crate) const MAX_CHAIN_SPACE: usize = 16;

/// Extra-torsion offset, mirroring the C reference's `HD_extra_torsion` (the
/// only implemented values are `0` and `2`).
pub(crate) const HD_EXTRA_TORSION: u32 = 2;

/// A consumer of the theta-chain control flow. [`drive_theta_chain`] issues
/// these calls in execution order; an implementor supplies the point
/// arithmetic and threads the codomain. Fallible operations return `false` to
/// abort the chain (mirroring the C reference's early `return 0`).
///
/// The two `descend_*` methods correspond to the two kernel representations:
/// before gluing the kernel lives as couple-Jacobian points (`jacQ[]`); after
/// gluing it lives as theta points (`thetaQ[]`).
pub(crate) trait ChainVisitor {
    /// Double the couple-Jacobian kernel point in slot `from` by `num_dbls`,
    /// storing the result in slot `to` (gluing-kernel-gathering phase).
    fn descend_couple(&mut self, from: usize, to: usize, num_dbls: u32);

    /// Double the theta kernel point in slot `from` by `num_dbls`, storing the
    /// result in slot `to` (interior phase).
    fn descend_theta(&mut self, from: usize, to: usize, num_dbls: u32);

    /// Compute the gluing isogeny from the order-2 kernel in slot `at`,
    /// evaluate the transported points, and set up the first codomain's theta
    /// structure. Returns `false` on failure.
    fn glue(&mut self, at: usize) -> bool;

    /// Push the gathered gluing-kernel point in slot `j` through the gluing
    /// isogeny (couple-Jacobian → theta).
    fn push_gluing_kernel(&mut self, j: usize);

    /// Compute interior `theta_isogeny` step `i` from the order-2 kernel in
    /// slot `at`, with the given Hadamard flags, evaluate the transported
    /// points, and update the codomain. Returns `false` on failure.
    fn step(&mut self, i: u32, at: usize, hadamard_1: bool, hadamard_2: bool, verify: bool)
    -> bool;

    /// Push the theta kernel point in slot `j` through the current step.
    fn push_step_kernel(&mut self, j: usize);

    /// The penultimate `compute_4` step (no-extra-torsion path), evaluate, and
    /// update the codomain.
    fn final_4(&mut self);

    /// The ultimate `compute_2` step (no-extra-torsion path), evaluate, and
    /// update the codomain.
    fn final_2(&mut self);

    /// The final splitting step `A → E₃ × E₄`, product extraction, and final
    /// point evaluation. Returns `false` if the kernel does not split.
    fn split(&mut self, extra_torsion: bool) -> bool;
}

/// Narrow the signed stack pointer to an index. `current` is `isize` only so
/// it can hold the `-1` "stack empty" sentinel; it is always non-negative
/// wherever it indexes.
#[allow(clippy::cast_sign_loss)]
fn slot(current: isize) -> usize {
    debug_assert!(current >= 0, "stack pointer used as index while negative");
    current as usize
}

/// Stack-machine size, mirroring the C reference:
/// `int space = 1; for (i = 1; i < n; i *= 2) ++space;`
pub(crate) fn chain_space(n: u32) -> usize {
    let mut space = 1usize;
    let mut i = 1u32;
    while i < n {
        space += 1;
        i = i.saturating_mul(2);
    }
    space
}

/// Drive the theta-chain control flow for a length-`n` chain, issuing
/// operations to `visitor` in execution order. Returns `false` if any visitor
/// operation aborts (gluing/step/split failure), matching the C reference's
/// `return 0`.
///
/// Precondition: `n ≥ 3` for the no-extra-torsion path (`n ≥ 1` with extra
/// torsion); `chain_space(n) ≤ MAX_CHAIN_SPACE`.
///
/// This is a faithful port of `_theta_chain_compute_impl`'s control flow; the
/// inline comments tie each block back to the C source.
pub(crate) fn drive_theta_chain<V: ChainVisitor>(
    n: u32,
    extra_torsion: bool,
    visitor: &mut V,
) -> bool {
    let extra: u32 = if extra_torsion { HD_EXTRA_TORSION } else { 0 };
    let space = chain_space(n);
    debug_assert!(space <= MAX_CHAIN_SPACE, "chain space exceeds fixed stack");

    // `todo[c]` = remaining 2-power order of the kernel point in stack slot `c`.
    let mut todo = [0u32; MAX_CHAIN_SPACE];
    todo[0] = n.wrapping_sub(2).wrapping_add(extra); // n - 2 + extra
    let mut current: isize = 0;

    // --- Phase A: gather the gluing-isogeny kernel (couple-Jacobian) ---
    // while (todo[current] != 1) { ++current; num_dbls = balanced; descend; }
    while todo[slot(current)] != 1 {
        debug_assert!(todo[slot(current)] >= 2);
        current += 1;
        debug_assert!(slot(current) < space);
        let prev = todo[slot(current - 1)];
        // The gluing step is far more expensive than the others, so near the
        // end of the descent it is cheaper to recompute the doublings than to
        // store the intermediate point: `>= 16 ? half : all-but-one`.
        let num_dbls = if prev >= 16 { prev / 2 } else { prev - 1 };
        debug_assert!(num_dbls != 0 && num_dbls < prev);
        visitor.descend_couple(slot(current - 1), slot(current), num_dbls);
        todo[slot(current)] = prev - num_dbls;
    }

    // --- Phase B: the gluing step ---
    debug_assert!(todo[slot(current)] == 1);
    if !visitor.glue(slot(current)) {
        return false;
    }
    // push the gathered kernel points through the gluing isogeny
    for (j, order) in todo[..slot(current)].iter_mut().enumerate() {
        visitor.push_gluing_kernel(j);
        *order -= 1;
    }
    current -= 1;

    // --- Phase C: interior theta-isogeny steps ---
    // for (i = 1; current >= 0 && todo[current]; ++i)
    let mut i: u32 = 1;
    while current >= 0 && todo[slot(current)] != 0 {
        // re-descend to an order-2 kernel point for this step
        while todo[slot(current)] != 1 {
            debug_assert!(todo[slot(current)] >= 2);
            current += 1;
            debug_assert!(slot(current) < space);
            let prev = todo[slot(current - 1)];
            let num_dbls = prev / 2;
            debug_assert!(num_dbls != 0 && num_dbls < prev);
            visitor.descend_theta(slot(current - 1), slot(current), num_dbls);
            todo[slot(current)] = prev - num_dbls;
        }

        // Hadamard flags: penultimate (0,0), ultimate (1,0), else (0,1). The
        // penultimate/ultimate branches only fire on the extra-torsion path;
        // on the no-extra-torsion path the finals (Phase D) play those roles
        // and the loop never reaches i == n-2.
        let (h1, h2, verify) = if i == n - 2 {
            (false, false, true)
        } else if i == n - 1 {
            (true, false, false)
        } else {
            (false, true, true)
        };
        if !visitor.step(i, slot(current), h1, h2, verify) {
            return false;
        }

        debug_assert!(todo[slot(current)] == 1);
        for (j, order) in todo[..slot(current)].iter_mut().enumerate() {
            visitor.push_step_kernel(j);
            debug_assert!(*order != 0);
            *order -= 1;
        }
        current -= 1;
        i += 1;
    }
    debug_assert!(current == -1);

    // --- Phase D: final steps (no-extra-torsion path only) ---
    if !extra_torsion {
        if n >= 3 {
            // last interior step skipped this push (current was 0); do it now
            visitor.push_step_kernel(0);
        }
        visitor.final_4(); // penultimate
        visitor.push_step_kernel(0);
        visitor.final_2(); // ultimate
    }

    // --- Phase E: splitting ---
    visitor.split(extra_torsion)
}

/// Maximum number of points the chain transports through alongside the
/// isogeny (the C reference's `numP`). SQIsign uses a small handful.
pub(crate) const MAX_CHAIN_EVAL_POINTS: usize = 8;

/// Point-executing [`ChainVisitor`]: owns the kernel-point stacks, the running
/// codomain abelian variety, the gluing/step isogenies, and the transported
/// evaluation points. Each method is a 1:1 transcription of the corresponding
/// block of the C reference's `_theta_chain_compute_impl`; [`drive_theta_chain`]
/// supplies the (already-verified) index logic.
///
/// The kernel is seeded directly from a couple-Jacobian
/// [`ThetaKernelCouplePoints`] — our kernel type is already lifted, so the C
/// reference's `lift_basis` step happens one level up (when the kernel bundle
/// is constructed, via [`crate::ec::jacobian::lift_basis`]). The caller is
/// responsible for the sign-consistency of `t1`, `t2`, `t1_minus_t2`.
pub(crate) struct ChainExecutor<F: BaseField> {
    curves: CoupleCurve<F>,
    /// Gluing-phase kernel stacks (couple-Jacobian), slot-indexed by `current`.
    jac_q1: [CoupleJacobianPoint<F>; MAX_CHAIN_SPACE],
    jac_q2: [CoupleJacobianPoint<F>; MAX_CHAIN_SPACE],
    /// Interior-phase kernel stacks (theta), slot-indexed by `current`.
    theta_q1: [ThetaPoint2D<F>; MAX_CHAIN_SPACE],
    theta_q2: [ThetaPoint2D<F>; MAX_CHAIN_SPACE],
    /// Current codomain abelian variety (set at the gluing step).
    theta: Option<AbelianVariety2D<F>>,
    /// The gluing isogeny (first step) and the most-recent interior/final step.
    first_step: Option<GluingCodomain<F>>,
    step: Option<ThetaIsogeny<F>>,
    /// Transported points: `in_pts` are the x-only inputs, `pts` the in-flight
    /// theta images, `out_pts` the final x-only outputs.
    num_p: usize,
    in_pts: [CoupleMontgomeryPoint<F>; MAX_CHAIN_EVAL_POINTS],
    pts: [ThetaPoint2D<F>; MAX_CHAIN_EVAL_POINTS],
    out_pts: [CoupleMontgomeryPoint<F>; MAX_CHAIN_EVAL_POINTS],
    /// Output elliptic product `E₃ × E₄` (set at the splitting step).
    e34: Option<CoupleCurve<F>>,
    /// Sticky failure latch — any fallible sub-step that errors sets this so
    /// `split` returns `false` and the chain reports failure overall.
    failed: bool,
}

impl<F: BaseField> ChainExecutor<F> {
    /// Seed the executor: kernel slot 0 holds `(T₁, T₂)` directly (our kernel
    /// is already couple-Jacobian), and the transported points are copied in.
    fn new(
        e12: &CoupleCurve<F>,
        ker: &ThetaKernelCouplePoints<F>,
        eval_points: &[CoupleMontgomeryPoint<F>],
    ) -> Self {
        let inf = CoupleJacobianPoint::infinity();
        let mut jac_q1 = [inf; MAX_CHAIN_SPACE];
        let mut jac_q2 = [inf; MAX_CHAIN_SPACE];
        jac_q1[0] = ker.t1;
        jac_q2[0] = ker.t2;

        let num_p = eval_points.len();
        debug_assert!(num_p <= MAX_CHAIN_EVAL_POINTS, "too many eval points");
        let mut in_pts = [CoupleMontgomeryPoint::infinity(); MAX_CHAIN_EVAL_POINTS];
        in_pts[..num_p].copy_from_slice(eval_points);

        Self {
            curves: *e12,
            jac_q1,
            jac_q2,
            theta_q1: [ThetaPoint2D::default(); MAX_CHAIN_SPACE],
            theta_q2: [ThetaPoint2D::default(); MAX_CHAIN_SPACE],
            theta: None,
            first_step: None,
            step: None,
            num_p,
            in_pts,
            pts: [ThetaPoint2D::default(); MAX_CHAIN_EVAL_POINTS],
            out_pts: [CoupleMontgomeryPoint::infinity(); MAX_CHAIN_EVAL_POINTS],
            e34: None,
            failed: false,
        }
    }

    /// The current codomain; the driver guarantees `glue` runs before any step.
    fn variety(&self) -> &AbelianVariety2D<F> {
        self.theta
            .as_ref()
            .expect("driver invariant: glue sets the codomain before any step")
    }
}

impl<F: BaseField> ChainVisitor for ChainExecutor<F> {
    fn descend_couple(&mut self, from: usize, to: usize, num_dbls: u32) {
        self.jac_q1[to] = self.jac_q1[from].double_iter(num_dbls, &self.curves);
        self.jac_q2[to] = self.jac_q2[from].double_iter(num_dbls, &self.curves);
    }

    fn descend_theta(&mut self, from: usize, to: usize, num_dbls: u32) {
        let av = *self.variety(); // AbelianVariety2D is Copy; end the borrow
        self.theta_q1[to] = av.double_iterated(&self.theta_q1[from], num_dbls);
        self.theta_q2[to] = av.double_iterated(&self.theta_q2[from], num_dbls);
    }

    fn glue(&mut self, at: usize) -> bool {
        // gluing isogeny E₁ × E₂ → A from the order-2 kernel in slot `at`
        let gc = match gluing_codomain(&self.curves, &self.jac_q1[at], &self.jac_q2[at]) {
            Ok(gc) => gc,
            Err(_) => {
                self.failed = true;
                return false;
            }
        };
        // evaluate the transported points (x-only special-case input)
        for j in 0..self.num_p {
            match gluing_eval_point_special_case(&gc, &self.in_pts[j]) {
                Ok(image) => self.pts[j] = image,
                Err(_) => {
                    self.failed = true;
                    return false;
                }
            }
        }
        // set up the first codomain's theta structure (C-ref theta_precomputation)
        match AbelianVariety2D::from_theta_null(gc.codomain) {
            Some(av) => self.theta = Some(av),
            None => {
                self.failed = true;
                return false;
            }
        }
        self.first_step = Some(gc);
        true
    }

    fn push_gluing_kernel(&mut self, j: usize) {
        let gc = self
            .first_step
            .as_ref()
            .expect("driver invariant: glue runs before pushing the gluing kernel");
        let (t1, t2) = gluing_eval_basis(gc, &self.jac_q1[j], &self.jac_q2[j]);
        self.theta_q1[j] = t1;
        self.theta_q2[j] = t2;
    }

    fn step(&mut self, _i: u32, at: usize, h1: bool, h2: bool, verify: bool) -> bool {
        let st = {
            let av = self.variety();
            theta_isogeny_compute(av, &self.theta_q1[at], &self.theta_q2[at], h1, h2, verify)
        };
        let st = match st {
            Ok(st) => st,
            Err(_) => {
                self.failed = true;
                return false;
            }
        };
        for j in 0..self.num_p {
            self.pts[j] = theta_isogeny_eval(&st, &self.pts[j]);
        }
        match AbelianVariety2D::from_theta_null(st.codomain_null) {
            Some(av) => self.theta = Some(av),
            None => {
                self.failed = true;
                return false;
            }
        }
        self.step = Some(st);
        true
    }

    fn push_step_kernel(&mut self, j: usize) {
        if self.failed {
            return;
        }
        let st = self
            .step
            .as_ref()
            .expect("driver invariant: a step runs before pushing its kernel");
        self.theta_q1[j] = theta_isogeny_eval(st, &self.theta_q1[j]);
        self.theta_q2[j] = theta_isogeny_eval(st, &self.theta_q2[j]);
    }

    fn final_4(&mut self) {
        if self.failed {
            return;
        }
        let st = {
            let av = self.variety();
            theta_isogeny_compute_4(av, &self.theta_q1[0], &self.theta_q2[0], false, false)
        };
        let st = match st {
            Ok(st) => st,
            Err(_) => {
                self.failed = true;
                return;
            }
        };
        for j in 0..self.num_p {
            self.pts[j] = theta_isogeny_eval(&st, &self.pts[j]);
        }
        match AbelianVariety2D::from_theta_null(st.codomain_null) {
            Some(av) => self.theta = Some(av),
            None => self.failed = true,
        }
        self.step = Some(st);
    }

    fn final_2(&mut self) {
        if self.failed {
            return;
        }
        let st = {
            let av = self.variety();
            theta_isogeny_compute_2(av, &self.theta_q1[0], &self.theta_q2[0], true, false)
        };
        let st = match st {
            Ok(st) => st,
            Err(_) => {
                self.failed = true;
                return;
            }
        };
        for j in 0..self.num_p {
            self.pts[j] = theta_isogeny_eval(&st, &self.pts[j]);
        }
        match AbelianVariety2D::from_theta_null(st.codomain_null) {
            Some(av) => self.theta = Some(av),
            None => self.failed = true,
        }
        self.step = Some(st);
    }

    fn split(&mut self, extra_torsion: bool) -> bool {
        if self.failed {
            return false;
        }
        let zero_index = if extra_torsion { Some(8) } else { None };
        let last = match splitting_compute(self.variety(), zero_index, false) {
            Ok(last) => last,
            Err(_) => return false,
        };
        // The product extraction + point conversion read only the codomain
        // theta-null; the doubling constants are irrelevant here (and a split
        // product null has a zero coordinate, so `from_theta_null` would
        // reject it). Wrap the null with placeholder constants.
        let product = AbelianVariety2D::new(last.b_null, ThetaPoint2D::default());
        let e34 = match theta_product_structure_to_elliptic_product(&product) {
            Ok(e34) => e34,
            Err(_) => return false,
        };
        for j in 0..self.num_p {
            self.pts[j] = apply_isomorphism(&last.m, &self.pts[j]);
            match theta_point_to_montgomery_point(&self.pts[j], &product) {
                Ok(mp) => self.out_pts[j] = mp,
                Err(_) => return false,
            }
        }
        self.e34 = Some(e34);
        true
    }
}

/// Compute a length-`n` dimension-2 `2^n`-isogeny `E₁ × E₂ → E₃ × E₄` from the
/// couple-Jacobian kernel `ker`, transporting `eval_points` through the chain.
///
/// On success returns `Some(E₃ × E₄)` and writes the transported images into
/// `out_points` (which must have the same length as `eval_points`, both
/// `≤ MAX_CHAIN_EVAL_POINTS`). Returns `None` if any chain step fails (e.g. the
/// kernel does not generate an isogeny between elliptic products).
///
/// This wires the point-executing [`ChainExecutor`] into the verified
/// [`drive_theta_chain`] control flow, mirroring the C reference's
/// `theta_chain_compute_and_eval`.
pub(crate) fn theta_chain_compute_and_eval<F: BaseField>(
    n: u32,
    e12: &CoupleCurve<F>,
    ker: &ThetaKernelCouplePoints<F>,
    extra_torsion: bool,
    eval_points: &[CoupleMontgomeryPoint<F>],
    out_points: &mut [CoupleMontgomeryPoint<F>],
) -> Option<CoupleCurve<F>> {
    debug_assert_eq!(eval_points.len(), out_points.len());
    debug_assert!(eval_points.len() <= MAX_CHAIN_EVAL_POINTS);
    let mut exec = ChainExecutor::new(e12, ker, eval_points);
    if !drive_theta_chain(n, extra_torsion, &mut exec) {
        return None;
    }
    let num_p = exec.num_p;
    out_points[..num_p].copy_from_slice(&exec.out_pts[..num_p]);
    exec.e34
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec::Vec;

    /// Records the operation stream the driver emits, so tests can assert on
    /// the control flow without any point arithmetic.
    #[derive(Default)]
    struct RecordingVisitor {
        descend_couple: Vec<(usize, usize, u32)>,
        descend_theta: Vec<(usize, usize, u32)>,
        glue_at: Vec<usize>,
        gluing_pushes: Vec<usize>,
        steps: Vec<(u32, usize, bool, bool)>, // (i, at, h1, h2)
        step_pushes: usize,
        final_4: u32,
        final_2: u32,
        split: u32,
        split_extra: Option<bool>,
        max_current: usize,
    }

    impl ChainVisitor for RecordingVisitor {
        fn descend_couple(&mut self, from: usize, to: usize, num_dbls: u32) {
            self.max_current = self.max_current.max(to);
            self.descend_couple.push((from, to, num_dbls));
        }
        fn descend_theta(&mut self, from: usize, to: usize, num_dbls: u32) {
            self.max_current = self.max_current.max(to);
            self.descend_theta.push((from, to, num_dbls));
        }
        fn glue(&mut self, at: usize) -> bool {
            self.glue_at.push(at);
            true
        }
        fn push_gluing_kernel(&mut self, j: usize) {
            self.gluing_pushes.push(j);
        }
        fn step(&mut self, i: u32, at: usize, h1: bool, h2: bool, _verify: bool) -> bool {
            self.steps.push((i, at, h1, h2));
            true
        }
        fn push_step_kernel(&mut self, _j: usize) {
            self.step_pushes += 1;
        }
        fn final_4(&mut self) {
            self.final_4 += 1;
        }
        fn final_2(&mut self) {
            self.final_2 += 1;
        }
        fn split(&mut self, extra_torsion: bool) -> bool {
            self.split += 1;
            self.split_extra = Some(extra_torsion);
            true
        }
    }

    fn run(n: u32, extra: bool) -> RecordingVisitor {
        let mut v = RecordingVisitor::default();
        let ok = drive_theta_chain(n, extra, &mut v);
        assert!(ok, "driver should not abort with the recording visitor");
        v
    }

    #[test]
    fn chain_space_matches_c_reference_formula() {
        // space = 1 + (number of powers of two strictly less than n)
        assert_eq!(chain_space(1), 1);
        assert_eq!(chain_space(2), 2);
        assert_eq!(chain_space(3), 3);
        assert_eq!(chain_space(4), 3);
        assert_eq!(chain_space(5), 4);
        assert_eq!(chain_space(16), 5);
        assert_eq!(chain_space(17), 6);
        assert_eq!(chain_space(248), 9);
        for n in 1..=512u32 {
            assert!(chain_space(n) <= MAX_CHAIN_SPACE);
        }
    }

    /// The chain decomposes a `2^n`-isogeny into exactly `n` 2-isogeny steps:
    /// 1 gluing + interior steps + 2 finals (no extra torsion).
    #[test]
    fn no_extra_torsion_total_steps_equals_n() {
        for n in 4..=248u32 {
            let v = run(n, false);
            let total = v.glue_at.len() as u32 + v.steps.len() as u32 + v.final_4 + v.final_2;
            assert_eq!(total, n, "n={n}: total 2-isogeny steps must equal n");
            assert_eq!(v.glue_at.len(), 1, "n={n}: exactly one gluing step");
            assert_eq!(v.final_4, 1, "n={n}: exactly one compute_4 final");
            assert_eq!(v.final_2, 1, "n={n}: exactly one compute_2 final");
            assert_eq!(v.split, 1, "n={n}: exactly one splitting step");
            assert_eq!(v.split_extra, Some(false));
            // interior steps = n - 3
            assert_eq!(v.steps.len() as u32, n - 3, "n={n}: interior count");
        }
    }

    /// Extra-torsion path: finals skipped, splitting consumes the 8-torsion;
    /// total steps still `n` (1 gluing + (n-1) interior).
    #[test]
    fn extra_torsion_total_steps_equals_n() {
        for n in 4..=248u32 {
            let v = run(n, true);
            let total = v.glue_at.len() as u32 + v.steps.len() as u32 + v.final_4 + v.final_2;
            assert_eq!(total, n, "n={n}: extra-torsion total steps must equal n");
            assert_eq!(v.final_4, 0, "n={n}: no finals on the extra-torsion path");
            assert_eq!(v.final_2, 0);
            assert_eq!(v.split_extra, Some(true));
            assert_eq!(v.steps.len() as u32, n - 1, "n={n}: interior count (extra)");
        }
    }

    /// Hadamard-flag accounting: across the whole chain there is exactly one
    /// penultimate `(0,0)` and one ultimate `(1,0)`; every other interior step
    /// is `(0,1)`.
    #[test]
    fn hadamard_flags_have_one_penultimate_one_ultimate() {
        // No extra torsion: the finals are the penultimate/ultimate, so every
        // *interior* step is the (0,1) "else" branch.
        for n in 4..=248u32 {
            let v = run(n, false);
            for &(_, _, h1, h2) in &v.steps {
                assert_eq!(
                    (h1, h2),
                    (false, true),
                    "n={n}: interior step must be (0,1)"
                );
            }
        }
        // Extra torsion: exactly one (0,0) and one (1,0) among the steps.
        for n in 4..=248u32 {
            let v = run(n, true);
            let pen = v.steps.iter().filter(|&&(_, _, h1, h2)| !h1 && !h2).count();
            let ult = v.steps.iter().filter(|&&(_, _, h1, h2)| h1 && !h2).count();
            assert_eq!(pen, 1, "n={n}: exactly one penultimate (0,0)");
            assert_eq!(ult, 1, "n={n}: exactly one ultimate (1,0)");
        }
    }

    /// The doubling descents respect the stack: every target slot stays within
    /// `chain_space(n)`, and every `num_dbls` is in `(0, prev_order)`.
    #[test]
    fn descents_respect_stack_bounds() {
        for n in 4..=248u32 {
            let space = chain_space(n);
            let v = run(n, false);
            for &(from, to, num) in v.descend_couple.iter().chain(v.descend_theta.iter()) {
                assert_eq!(to, from + 1, "descent always pushes one slot");
                assert!(
                    to < space,
                    "n={n}: descent target {to} within space {space}"
                );
                assert!(num >= 1, "n={n}: num_dbls must be positive");
            }
            assert!(v.max_current < space, "n={n}: current stayed within space");
            // The gluing kernel is always gathered into a single slot: the
            // first descent reduces todo to 1, so gluing consumes that slot.
            assert_eq!(v.glue_at.len(), 1);
        }
    }

    /// Spot-check the smallest non-trivial chain end-to-end (n = 4): one
    /// gluing, one interior (0,1) step, both finals, one split.
    #[test]
    fn n4_no_extra_torsion_exact_op_stream() {
        let v = run(4, false);
        assert_eq!(v.glue_at, alloc::vec![1]);
        assert_eq!(v.steps.len(), 1);
        assert_eq!(v.steps[0].2, false); // h1
        assert_eq!(v.steps[0].3, true); // h2
        assert_eq!(v.final_4, 1);
        assert_eq!(v.final_2, 1);
        assert_eq!(v.split, 1);
    }

    /// S251 executor threading + graceful-failure smoke test. Feeds a
    /// degenerate (infinity) kernel couple on E₀ × E₀ through the real
    /// point-executing `ChainExecutor` via `theta_chain_compute_and_eval`.
    /// This drives the full pipeline — seeding, the Phase-A couple-Jacobian
    /// descent, and the gluing step — exercising every type boundary; the
    /// infinity kernel is not a valid (2ⁿ)-isogeny kernel, so
    /// `gluing_codomain` fails and the chain must return `None` cleanly
    /// (no panic). A valid-kernel semantic round-trip needs a constructed
    /// isogeny fixture and lands in a later session.
    fn check_executor_threads_and_fails_closed<F: BaseField>() {
        use crate::ec::couple::{CoupleCurve, CoupleJacobianPoint, ThetaKernelCouplePoints};

        let e12 = CoupleCurve::<F>::e0_e0();
        let inf = CoupleJacobianPoint::<F>::infinity();
        let ker = ThetaKernelCouplePoints::new(inf, inf, inf);
        let result = theta_chain_compute_and_eval::<F>(8, &e12, &ker, false, &[], &mut []);
        assert!(
            result.is_none(),
            "degenerate infinity kernel must fail closed (no split), not panic",
        );
    }

    #[test]
    fn executor_threads_and_fails_closed_at_lvl1() {
        use crate::params::lvl1::Fp1Element;
        check_executor_threads_and_fails_closed::<Fp1Element>();
    }

    #[test]
    fn executor_threads_and_fails_closed_at_lvl3() {
        use crate::params::lvl3::Fp3Element;
        check_executor_threads_and_fails_closed::<Fp3Element>();
    }
}
