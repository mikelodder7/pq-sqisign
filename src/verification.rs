//! SQIsign verification: decoded signature / public-key forms and the
//! verify orchestration helpers.
//!
//! Port of the C reference `src/verification/` (`verify.c`,
//! `encode_verification.c`). The wire formats here mirror C
//! `signature_from_bytes` / `public_key_from_bytes` exactly; the decoded
//! structs mirror `signature_t` / `public_key_t`.
//!
//! lvl1-pinned: the basis-change matrix and challenge coefficient are the
//! order scalars (`Uint<8>`), and the field element is `Fp1Element`. The
//! rest of the verify subsystem (`ec::biscalar` basis routines) is pinned the
//! same way.

use crate::error::{Error, Result};
use crate::gf::fp::BaseField;
use crate::gf::fp2::Fp2;
use crate::params::lvl1::{Fp1Element, Level1};
use crypto_bigint::{U256, Uint};
use subtle::ConstantTimeEq;

/// A Montgomery curve paired with an even-torsion basis on it, over field `F`.
type CurveBasis<F> = (
    crate::ec::montgomery::MontgomeryCurve<F>,
    crate::ec::couple::EcBasis<F>,
);
/// A pair of even-torsion bases over field `F` (challenge basis, auxiliary basis).
type BasisPair<F> = (crate::ec::couple::EcBasis<F>, crate::ec::couple::EcBasis<F>);

/// lvl1 PK wire size (bytes): `PK_BYTES = 64 (A) + 1 (hint) = 65`.
const PK_BYTES_LVL1: usize = 65;
/// lvl1 signature/field wire widths, retained for the byte-layout tests now
/// that the production encode/decode derive these from [`Params`].
#[cfg(test)]
const SIG_BYTES_LVL1: usize = 148;
/// fp2 encoding width at lvl1 (`2 × Fp1Element::ENCODED_BYTES`).
#[cfg(test)]
const FP2_BYTES_LVL1: usize = 64;
/// Matrix-entry width: `(RESPONSE_BITS + 9) / 8 = (126 + 9) / 8 = 16` at lvl1.
#[cfg(test)]
const MAT_ENTRY_BYTES_LVL1: usize = 16;
/// lvl1 secret-key wire size and field widths (C `secret_key_from_bytes`).
/// `SK_BYTES = 65 (pk) + 32 (norm) + 4·32 (gen coords) + 4·32 (matrix) = 353`.
const SK_BYTES_LVL1: usize = 353;
/// `FP_ENCODED_BYTES` / `TORSION_2POWER_BYTES` at lvl1 (both 32).
const SK_FIELD_BYTES_LVL1: usize = 32;

/// Decode a `nbytes`-byte little-endian field into `Int<W>`. When `signed`, the
/// top bit is the two's-complement sign (C `ibz_from_bytes` with `sgn=true`).
fn decode_int_le<const W: usize>(bytes: &[u8], signed: bool) -> crypto_bigint::Int<W> {
    let mut buf = [0u8; 32];
    buf[..bytes.len().min(32)].copy_from_slice(&bytes[..bytes.len().min(32)]);
    let raw = U256::from_le_slice(&buf);
    let nbytes = bytes.len();
    if signed && (bytes[nbytes - 1] >> 7) == 1 {
        // negative two's complement over `nbytes` bits: |value| = 2^(8·nbytes) − raw.
        let mag = if nbytes >= 32 {
            raw.wrapping_neg() // 2^256 − raw
        } else {
            U256::ONE
                .shl_vartime(u32::try_from(8 * nbytes).expect("8·nbytes fits u32"))
                .wrapping_sub(&raw)
        };
        mag.resize::<W>().as_int().wrapping_neg()
    } else {
        *raw.resize::<W>().as_int()
    }
}

/// Decoded secret key — C `secret_key_t` (the wire-relevant fields). Lives here
/// alongside the other decoded wire forms. lvl1-pinned.
#[derive(Clone, Debug)]
pub struct SecretKeyData<F: BaseField> {
    /// Affine Montgomery `A` of the public curve (== pk curve).
    pub curve_a: Fp2<F>,
    /// Public-key basis hint.
    pub hint_pk: u8,
    /// The secret left ideal `O_0·gen + norm·O_0`, reconstructed from the
    /// encoded generator and norm.
    pub secret_ideal: crate::quaternion::ideal::LeftIdeal<16>,
    /// Basis-change matrix `mat_BAcan_to_BA0_two`.
    pub mat_bacan_to_ba0_two: [[Uint<8>; 2]; 2],
}

impl SecretKeyData<Fp1Element> {
    /// Decode from the 353-byte lvl1 wire format — C `secret_key_from_bytes`.
    /// Layout: `pk (65) || norm (32, unsigned) || gen.coord[0..3] (4×32, signed)
    /// || mat[0][0],[0][1],[1][0],[1][1] (4×32, unsigned)`. The secret ideal is
    /// rebuilt via `quat_lideal_create(gen, denom=1, norm, O_0)` (the encoded
    /// generator's denominator is omitted — it does not change the ideal).
    pub fn from_bytes_lvl1(enc: &[u8]) -> Result<Self> {
        use crate::quaternion::Quaternion;
        use crate::quaternion::lattice::narrow_int_lattice;
        use crate::quaternion::o0_mul::{c_ideal_to_left_ideal, quat_lideal_create};
        const W: usize = 24; // create width: norm~2^250 ⇒ det_4x4 needs headroom

        if enc.len() < SK_BYTES_LVL1 {
            return Err(Error::BufferTooSmall {
                required: SK_BYTES_LVL1,
                provided: enc.len(),
            });
        }
        let pk = PublicKeyData::from_bytes_lvl1(&enc[..PK_BYTES_LVL1])?;
        let mut off = PK_BYTES_LVL1;
        let fb = SK_FIELD_BYTES_LVL1;

        let norm = decode_int_le::<W>(&enc[off..off + fb], false).abs();
        off += fb;
        let gen_q = Quaternion::<W>::new(
            decode_int_le::<W>(&enc[off..off + fb], true),
            decode_int_le::<W>(&enc[off + fb..off + 2 * fb], true),
            decode_int_le::<W>(&enc[off + 2 * fb..off + 3 * fb], true),
            decode_int_le::<W>(&enc[off + 3 * fb..off + 4 * fb], true),
        );
        off += 4 * fb;

        let read_u256 = |o: usize| -> Uint<8> {
            let mut buf = [0u8; 64];
            buf[..fb].copy_from_slice(&enc[o..o + fb]);
            Uint::<8>::from_le_slice(&buf)
        };
        let mat = [
            [read_u256(off), read_u256(off + fb)],
            [read_u256(off + 2 * fb), read_u256(off + 3 * fb)],
        ];

        // Reconstruct the secret ideal at width W, narrow to LeftIdeal<16>.
        let p_w = crate::params::lvl1::prime().resize::<W>();
        let (basis, denom, n) =
            quat_lideal_create::<W>(&gen_q, &crypto_bigint::Int::<W>::from_i64(1), &norm, &p_w);
        let wide = c_ideal_to_left_ideal::<W>(&basis, &denom, &n);
        let mut b16 = [[crypto_bigint::Int::<16>::from_i64(0); 4]; 4];
        for (row16, row_w) in b16.iter_mut().zip(wide.basis.iter()) {
            for (e16, e_w) in row16.iter_mut().zip(row_w.iter()) {
                *e16 = narrow_int_lattice::<W, 16>(e_w);
            }
        }
        let secret_ideal = crate::quaternion::ideal::LeftIdeal::<16>::with_denom_and_norm(
            b16,
            wide.denom.resize::<16>(),
            wide.cached_norm.resize::<16>(),
        );

        Ok(Self {
            curve_a: pk.curve_a,
            hint_pk: pk.hint_pk,
            secret_ideal,
            mat_bacan_to_ba0_two: mat,
        })
    }
}

/// Decoded public key — C `public_key_t`. The curve is stored by its
/// normalized affine Montgomery coefficient `A` (`C = 1`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PublicKeyData<F: BaseField> {
    /// Affine Montgomery coefficient `A` of the public curve.
    pub curve_a: Fp2<F>,
    /// Basis hint for the public curve (`ec_curve_to_basis_2f` recomputation).
    pub hint_pk: u8,
}

impl<F: BaseField> PublicKeyData<F> {
    /// Encode to the `P::PK_BYTES`-byte wire format — C `public_key_to_bytes`.
    /// Inverse of [`Self::from_bytes`]; layout `A (FP2_BYTES) || hint_pk (1)`.
    pub fn to_bytes<P: crate::params::Params<Field = F>>(&self, out: &mut [u8]) -> Result<()> {
        if out.len() < P::PK_BYTES {
            return Err(Error::BufferTooSmall {
                required: P::PK_BYTES,
                provided: out.len(),
            });
        }
        self.curve_a.to_bytes_le(&mut out[..P::FP2_BYTES]);
        out[P::FP2_BYTES] = self.hint_pk;
        Ok(())
    }

    /// Decode from the `P::PK_BYTES`-byte wire format — C `public_key_from_bytes`.
    /// Layout: `A (FP2_BYTES) || hint_pk (1)`.
    pub fn from_bytes<P: crate::params::Params<Field = F>>(enc: &[u8]) -> Result<Self> {
        if enc.len() < P::PK_BYTES {
            return Err(Error::BufferTooSmall {
                required: P::PK_BYTES,
                provided: enc.len(),
            });
        }
        let curve_a = Fp2::<F>::from_bytes_le(&enc[..P::FP2_BYTES])
            .into_option()
            .ok_or(Error::InvalidPublicKey)?;
        Ok(Self {
            curve_a,
            hint_pk: enc[P::FP2_BYTES],
        })
    }
}

impl PublicKeyData<Fp1Element> {
    /// lvl1 wire encode — thin alias for [`Self::to_bytes::<Level1>`].
    pub fn to_bytes_lvl1(&self, out: &mut [u8]) -> Result<()> {
        self.to_bytes::<Level1>(out)
    }

    /// lvl1 wire decode — thin alias for [`Self::from_bytes::<Level1>`].
    pub fn from_bytes_lvl1(enc: &[u8]) -> Result<Self> {
        Self::from_bytes::<Level1>(enc)
    }
}

/// Decoded signature — C `signature_t`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SignatureData<F: BaseField> {
    /// Montgomery `A`-coefficient of the auxiliary curve.
    pub e_aux_a: Fp2<F>,
    /// Backtracking length consumed by the response isogeny.
    pub backtracking: u8,
    /// Length of the short `2^r` response isogeny (`0` if absent).
    pub two_resp_length: u8,
    /// Basis-change matrix `mat_Bchall_can_to_B_chall` (`[[m00,m01],[m10,m11]]`).
    pub mat: [[Uint<8>; 2]; 2],
    /// Challenge coefficient (recomputed and compared during verify).
    pub chall_coeff: Uint<8>,
    /// Basis hint for the auxiliary curve.
    pub hint_aux: u8,
    /// Basis hint for the challenge curve.
    pub hint_chall: u8,
}

/// Per-level signature wire widths, derived from [`Params`]. At lvl1 these are
/// `(64, 16, 16, 148)`; at lvl3 `(96, 25, 24, 224)`.
/// `mat_entry = (RESPONSE_BITS + 9) / 8`, `chall = SECURITY_BITS / 8`.
const fn sig_wire_widths<P: crate::params::Params>() -> (usize, usize, usize, usize) {
    (
        P::FP2_BYTES,
        (P::RESPONSE_BITS + 9) / 8,
        P::SECURITY_BITS / 8,
        P::SIG_BYTES,
    )
}

impl<F: BaseField> SignatureData<F> {
    /// Encode to the `P::SIG_BYTES`-byte wire format — C `signature_to_bytes`.
    /// Inverse of [`Self::from_bytes`]; the matrix entries and challenge
    /// coefficient are written as fixed-width little-endian (C `encode_digits`),
    /// taking the low `mat_entry` / `chall` bytes.
    pub fn to_bytes<P: crate::params::Params<Field = F>>(&self, out: &mut [u8]) -> Result<()> {
        let (fp2_bytes, mat_entry, chall_bytes, sig_bytes) = sig_wire_widths::<P>();
        if out.len() < sig_bytes {
            return Err(Error::BufferTooSmall {
                required: sig_bytes,
                provided: out.len(),
            });
        }
        self.e_aux_a.to_bytes_le(&mut out[..fp2_bytes]);
        let mut off = fp2_bytes;
        out[off] = self.backtracking;
        out[off + 1] = self.two_resp_length;
        off += 2;

        let write_digits = |out: &mut [u8], o: usize, width: usize, v: &Uint<8>| {
            out[o..o + width].copy_from_slice(&v.to_le_bytes()[..width]);
        };
        write_digits(out, off, mat_entry, &self.mat[0][0]);
        write_digits(out, off + mat_entry, mat_entry, &self.mat[0][1]);
        write_digits(out, off + 2 * mat_entry, mat_entry, &self.mat[1][0]);
        write_digits(out, off + 3 * mat_entry, mat_entry, &self.mat[1][1]);
        off += 4 * mat_entry;

        write_digits(out, off, chall_bytes, &self.chall_coeff);
        off += chall_bytes;

        out[off] = self.hint_aux;
        out[off + 1] = self.hint_chall;
        Ok(())
    }

    /// Decode from the `P::SIG_BYTES`-byte wire format — C `signature_from_bytes`.
    /// Layout (lvl1): `E_aux_A (64) || backtracking (1) || two_resp_length (1) ||
    /// m00 (16) || m01 (16) || m10 (16) || m11 (16) || chall_coeff (16) ||
    /// hint_aux (1) || hint_chall (1)`; widths scale with the level.
    pub fn from_bytes<P: crate::params::Params<Field = F>>(enc: &[u8]) -> Result<Self> {
        let (fp2_bytes, mat_entry, chall_bytes, sig_bytes) = sig_wire_widths::<P>();
        if enc.len() < sig_bytes {
            return Err(Error::BufferTooSmall {
                required: sig_bytes,
                provided: enc.len(),
            });
        }
        let e_aux_a = Fp2::<F>::from_bytes_le(&enc[..fp2_bytes])
            .into_option()
            .ok_or(Error::NonCanonicalEncoding)?;
        let mut off = fp2_bytes;
        let backtracking = enc[off];
        let two_resp_length = enc[off + 1];
        off += 2;

        // decode_digits: read a fixed-width little-endian field into a Uint<8>.
        let read_u256 = |enc: &[u8], o: usize, width: usize| -> Uint<8> {
            let mut buf = [0u8; 64];
            buf[..width].copy_from_slice(&enc[o..o + width]);
            Uint::<8>::from_le_slice(&buf)
        };

        let m00 = read_u256(enc, off, mat_entry);
        let m01 = read_u256(enc, off + mat_entry, mat_entry);
        let m10 = read_u256(enc, off + 2 * mat_entry, mat_entry);
        let m11 = read_u256(enc, off + 3 * mat_entry, mat_entry);
        off += 4 * mat_entry;

        let chall_coeff = read_u256(enc, off, chall_bytes);
        off += chall_bytes;

        let hint_aux = enc[off];
        let hint_chall = enc[off + 1];

        Ok(Self {
            e_aux_a,
            backtracking,
            two_resp_length,
            mat: [[m00, m01], [m10, m11]],
            chall_coeff,
            hint_aux,
            hint_chall,
        })
    }
}

impl SignatureData<Fp1Element> {
    /// lvl1 wire encode — thin alias for [`Self::to_bytes::<Level1>`], preserved
    /// for the byte-exact lvl1 call sites and tests.
    pub fn to_bytes_lvl1(&self, out: &mut [u8]) -> Result<()> {
        self.to_bytes::<Level1>(out)
    }

    /// lvl1 wire decode — thin alias for [`Self::from_bytes::<Level1>`].
    pub fn from_bytes_lvl1(enc: &[u8]) -> Result<Self> {
        Self::from_bytes::<Level1>(enc)
    }
}

/// `x^(2^k)` by repeated squaring in `Fp2`.
fn fp2_pow2k<F: BaseField>(x: &Fp2<F>, k: u32) -> Fp2<F> {
    let mut r = *x;
    for _ in 0..k {
        r = r.square();
    }
    r
}

/// 2-adic discrete log in `μ_{2^e}`: given a primitive `2^e`-th root of unity
/// `base` and `target = base^x`, return `x mod 2^e` (little-endian `Uint<8>`).
///
/// Pohlig–Hellman over the 2-group: at step `k` the value
/// `running^(2^(e−1−k))` lies in the order-2 subgroup `{1, −1}`; it is `−1`
/// exactly when bit `k` of `x` is set. Each set bit is stripped by multiplying
/// `running` by `base^(−2^k)`. The change-of-basis matrices on `E[2^f]` are
/// assembled from four such dlogs of pairing values. `e ≤ 512`.
pub(crate) fn dlog_2f<F: BaseField>(base: &Fp2<F>, target: &Fp2<F>, e: u32) -> Uint<8> {
    let mut x = Uint::<8>::ZERO;
    let inv_base = base.invert().into_option().unwrap_or_else(Fp2::one);
    let mut inv_pow = inv_base; // base^(−2^k), starting at k = 0
    let mut running = *target; // base^(x − partial)
    for k in 0..e {
        // running^(2^(e−1−k)) ∈ {1, −1}; ≠ 1 ⇒ bit k is set.
        if !bool::from(fp2_pow2k(&running, e - 1 - k).is_one()) {
            x |= Uint::<8>::ONE.shl_vartime(k);
            running = running.mul(&inv_pow);
        }
        inv_pow = inv_pow.square();
    }
    x
}

/// Weil pairing `e_{2^f}(R, S)` from two lifted Jacobian points. Uses
/// `add_components` to recover `x(R + S)` (= `(u − v)/w`), which the cubical
/// Weil routine requires.
#[cfg(feature = "alloc")]
fn weil_jac<F: BaseField>(
    f: u32,
    r: &crate::ec::jacobian::JacobianPoint<F>,
    s: &crate::ec::jacobian::JacobianPoint<F>,
    curve: &crate::ec::montgomery::MontgomeryCurve<F>,
) -> Fp2<F> {
    use crate::ec::montgomery::MontgomeryPoint;
    let (u, v, w) = r.add_components(s, &curve.a);
    let rs_plus = MontgomeryPoint::new(u.sub(&v), w); // x(R + S) = (u − v)/w
    crate::ec::weil::weil(
        f,
        &r.to_montgomery_xz(),
        &s.to_montgomery_xz(),
        &rs_plus,
        curve,
    )
}

/// Change-of-basis matrix `M` expressing basis `b1` in terms of basis `b2` on
/// `E[2^f]` (both bases of order exactly `2^f`): `b1.P = [M00]·b2.P + [M10]·b2.Q`
/// and `b1.Q = [M01]·b2.P + [M11]·b2.Q` (all mod `2^f`). The transpose-column
/// layout matches C `change_of_basis_matrix_tate` so that
/// `matrix_application_even_basis` applied with
/// `M` reconstructs `b1` from `b2`.
///
/// Computed from Weil pairings (the matrix is pairing-independent, so this
/// matches C's Tate-based result): with `ζ = e(b2.P, b2.Q)`, for each target
/// `R`, `e(R, b2.Q) = ζ^a` and `e(R, b2.P) = ζ^(−b)`. Returns `None` if a basis
/// point fails to lift. Field-generic over the security level via
/// [`LevelConstants`]; the pairing math is identical at every level.
#[cfg(feature = "alloc")]
pub fn change_of_basis_matrix<P: crate::level_constants::LevelConstants>(
    b1: &crate::ec::couple::EcBasis<P::Field>,
    b2: &crate::ec::couple::EcBasis<P::Field>,
    curve: &crate::ec::montgomery::MontgomeryCurve<P::Field>,
    f: u32,
) -> Option<[[Uint<8>; 2]; 2]> {
    use crate::ec::jacobian::lift_basis;
    let (p2, q2) = lift_basis(b2, curve).ok()?;
    let (p1, q1) = lift_basis(b1, curve).ok()?;

    let zeta = weil_jac(f, &p2, &q2, curve);
    let mask = Uint::<8>::MAX.wrapping_shr(512 - f); // 2^f − 1
    let neg_mod = |x: Uint<8>| Uint::<8>::ZERO.wrapping_sub(&x) & mask;

    // Coordinates of R in (b2.P, b2.Q). With this implementation's Weil
    // orientation, e(R, b2.Q) = ζ^(−a) and e(R, b2.P) = ζ^b, so
    // a = −dlog(e(R,Q)) and b = dlog(e(R,P)) (mod 2^f).
    let coords = |r: &crate::ec::jacobian::JacobianPoint<P::Field>| {
        let a = neg_mod(dlog_2f(&zeta, &weil_jac(f, r, &q2, curve), f));
        let b = dlog_2f(&zeta, &weil_jac(f, r, &p2, curve), f);
        (a, b)
    };
    let (m00, m10) = coords(&p1); // b1.P = [m00]b2.P + [m10]b2.Q
    let (m01, m11) = coords(&q1); // b1.Q = [m01]b2.P + [m11]b2.Q
    Some([[m00, m01], [m10, m11]])
}

/// Sign-tail: compute the canonical-basis hints and the basis-change matrix for
/// a signature. Port of C `compute_and_set_basis_change_matrix` (`sign.c:418`).
/// Sets `sig.hint_chall`, `sig.hint_aux`, and `sig.mat` (= the matrix taking the
/// canonical challenge basis to the supplied `b_chall_2`, after re-expressing
/// the auxiliary basis in canonical form). `f` is the working order
/// `pow_dim2_deg_resp + HD_extra + two_resp_length`. Returns `false` on a lift
/// failure. lvl1-pinned (`TORSION_EVEN_POWER = 248`).
///
/// Uses the identity `change_of_basis_matrix_tate_invert(B1, B2)
/// = change_of_basis_matrix(B2, B1)` (the inverse matrix mod `2^f` equals the
/// directly-computed opposite-direction coordinates), so no explicit matrix
/// inversion is needed.
#[cfg(feature = "alloc")]
pub fn compute_and_set_basis_change_matrix<P: crate::level_constants::LevelConstants>(
    sig: &mut SignatureData<P::Field>,
    b_aux_2: &crate::ec::couple::EcBasis<P::Field>,
    b_chall_2: &crate::ec::couple::EcBasis<P::Field>,
    e_aux_2: &crate::ec::montgomery::MontgomeryCurve<P::Field>,
    e_chall: &crate::ec::montgomery::MontgomeryCurve<P::Field>,
    f: usize,
) -> bool {
    use crate::ec::biscalar::ec_curve_to_basis_2f_to_hint;
    use crate::ec::couple::EcBasis;
    use crate::isogeny::endomorphism::matrix_application_even_basis;
    let torsion_even_power = P::F;
    let e_diff = match torsion_even_power.checked_sub(f) {
        Some(d) => d,
        None => return false,
    };
    let e_diff = match u32::try_from(e_diff) {
        Ok(d) => d,
        Err(_) => return false,
    };
    let f32 = match u32::try_from(f) {
        Ok(v) => v,
        Err(_) => return false,
    };

    // Canonical full-order bases + their hints.
    let (b_can_chall, hint_chall) = ec_curve_to_basis_2f_to_hint::<P>(e_chall, torsion_even_power);
    let (b_aux_2_can, hint_aux) = ec_curve_to_basis_2f_to_hint::<P>(e_aux_2, torsion_even_power);
    // Reduce canonical bases to order 2^f to match the supplied bases.
    let b_can_chall_f = ec_dbl_iter_basis(&b_can_chall, e_diff, e_chall);
    let b_aux_2_can_f = ec_dbl_iter_basis(&b_aux_2_can, e_diff, e_aux_2);

    // M_aux = "go from B_aux_2 to B_aux_2_can" (C `change_of_basis_matrix_tate_invert`,
    // sign.c:458). With `b_aux_2 = A·b_aux_2_can`, kernel preservation requires
    // folding A⁻¹ (NOT A) into the challenge basis: verify uses canonical aux +
    // folded chall, so the chall fold must undo the aux basis change. A⁻¹ =
    // "b_aux_2_can in terms of b_aux_2" = change_of_basis_matrix(b_aux_2_can, b_aux_2).
    // (The earlier (b_aux_2, b_aux_2_can) order gave A, producing a non-isotropic
    // verify kernel even though the sign's actual kernel is isotropic.)
    let m_aux = match change_of_basis_matrix::<P>(&b_aux_2_can_f, b_aux_2, e_aux_2, f32) {
        Some(m) => m,
        None => return false,
    };
    // Apply M_aux to the supplied challenge basis (points live on E_chall).
    let a24 = e_chall.a24();
    let (cp, cq, cpmq) = match matrix_application_even_basis(
        &b_chall_2.p,
        &b_chall_2.q,
        &b_chall_2.p_minus_q,
        &m_aux,
        f,
        &a24,
    ) {
        Some(t) => t,
        None => return false,
    };
    let b_chall_2_adj = EcBasis::new(cp, cq, cpmq);

    // M_chall = canonical challenge basis → adjusted supplied basis.
    let m_chall = match change_of_basis_matrix::<P>(&b_chall_2_adj, &b_can_chall_f, e_chall, f32) {
        Some(m) => m,
        None => return false,
    };

    sig.mat = m_chall;
    sig.hint_chall = hint_chall;
    sig.hint_aux = hint_aux;
    true
}

/// Montgomery-curve isomorphism `from → to` (same j-invariant), as the linear
/// map on `(x : z)`: `x ↦ Nx·x + Nz·z`, `z ↦ D·z`. Port of C `ec_isomorphism`
/// (`isog_chains.c`) specialized to the affine `C = 1` representation. Returns
/// `(Nx, Nz, D)`, or `None` in the rare degenerate case (`Nx = 0` or `D = 0`).
pub fn ec_isomorphism<F: BaseField>(
    from_a: &Fp2<F>,
    to_a: &Fp2<F>,
) -> Option<(Fp2<F>, Fp2<F>, Fp2<F>)> {
    let three = Fp2::<F>::one().double().add(&Fp2::one());
    let nine = three.double().add(&three);
    let cube = |x: &Fp2<F>| x.square().mul(x);
    // λx = (2·toA³ − 9·toA)·(3 − fromA²);  λz = (2·fromA³ − 9·fromA)·(3 − toA²).
    let mut nx = cube(to_a)
        .double()
        .sub(&nine.mul(to_a))
        .mul(&three.sub(&from_a.square()));
    let mut d = cube(from_a)
        .double()
        .sub(&nine.mul(from_a))
        .mul(&three.sub(&to_a.square()));
    // Nz = fromA·λx − toA·λz (C = 1).
    let nz = from_a.mul(&nx).sub(&to_a.mul(&d));
    // Scale by 3·fromC·toC = 3.
    nx = nx.mul(&three);
    d = d.mul(&three);
    if bool::from(nx.is_zero()) || bool::from(d.is_zero()) {
        return None;
    }
    Some((nx, nz, d))
}

/// Apply an [`ec_isomorphism`] `(Nx, Nz, D)` to an x-only point.
pub fn apply_iso<F: BaseField>(
    p: &crate::ec::montgomery::MontgomeryPoint<F>,
    isom: &(Fp2<F>, Fp2<F>, Fp2<F>),
) -> crate::ec::montgomery::MontgomeryPoint<F> {
    let (nx, nz, d) = isom;
    crate::ec::montgomery::MontgomeryPoint::new(p.x.mul(nx).add(&p.z.mul(nz)), p.z.mul(d))
}

/// Verify the Montgomery coefficient `A` describes a valid curve: `A² − 4 ≠ 0`,
/// i.e. `A ∉ {2, −2}`. Port of C `ec_curve_verify_A` (`ec.c`).
pub fn ec_curve_verify_a<F: BaseField>(a: &Fp2<F>) -> bool {
    let two = Fp2::<F>::one().double();
    if bool::from(a.ct_eq(&two)) {
        return false;
    }
    let neg_two = two.negate();
    !bool::from(a.ct_eq(&neg_two))
}

/// Reject signatures whose basis-change matrix entries are not canonical
/// representatives mod `2^(SQIsign_response_length + HD_extra_torsion −
/// backtracking)`. Port of C `check_canonical_basis_change_matrix` (`verify.c`):
/// every entry must be strictly less than that bound (C rejects when the bound
/// is `≤` an entry). Assumes all entries are non-negative (they decode from
/// unsigned bytes). lvl1-pinned constants.
pub fn check_canonical_basis_change_matrix<P: crate::params::Params>(
    sig: &SignatureData<P::Field>,
) -> bool {
    use crate::isogeny::theta_chain::HD_EXTRA_TORSION;
    // SQIsign_response_length + HD_extra_torsion − backtracking (lvl1: 126 + 2).
    let response_length = i32::try_from(P::RESPONSE_BITS).expect("RESPONSE_BITS fits i32");
    let extra = i32::try_from(HD_EXTRA_TORSION).expect("HD_EXTRA_TORSION fits i32");
    let shift = response_length + extra - i32::from(sig.backtracking);
    if shift < 0 {
        // Bound 2^shift undefined for shift < 0 ⇒ no canonical rep exists.
        return false;
    }
    let bound = Uint::<8>::ONE.shl_vartime(u32::try_from(shift).expect("shift ≥ 0"));
    // Valid iff every matrix entry is strictly below the bound.
    sig.mat.iter().flatten().all(|e| e < &bound)
}

/// Recompute the challenge curve `E_chall` from the public key and signature.
/// Port of C `compute_challenge_verify` (`verify.c`): build the canonical basis
/// of `E_pk[2^f]` from `hint_pk`, form the kernel `P + [chall_coeff]·Q`, double
/// it `backtracking` times to order `2^(f − backtracking)`, then take the
/// codomain of that even isogeny. lvl1-pinned (`TORSION_EVEN_POWER = 248`).
#[cfg(feature = "alloc")]
pub fn compute_challenge_verify<P: crate::level_constants::LevelConstants>(
    epk: &crate::ec::montgomery::MontgomeryCurve<P::Field>,
    sig: &SignatureData<P::Field>,
    hint_pk: u8,
) -> crate::ec::montgomery::MontgomeryCurve<P::Field> {
    use crate::ec::biscalar::ec_curve_to_basis_2f_from_hint;
    use crate::ec::montgomery::{MontgomeryCurve, MontgomeryPoint};
    use crate::isogeny::two::IsogenyChain2e;
    let torsion_even_power = P::F;
    let length = torsion_even_power - usize::from(sig.backtracking);
    // Canonical basis of E_pk[2^f] from the public-key hint.
    let bas = ec_curve_to_basis_2f_from_hint::<P>(epk, torsion_even_power, hint_pk);
    // kernel = P + [chall_coeff]·Q on E_pk.
    let a24 = epk.a24();
    let kernel = MontgomeryPoint::ladder3pt(
        &bas.p,
        &bas.q,
        &bas.p_minus_q,
        &sig.chall_coeff.to_le_bytes(),
        &a24,
    );
    // Double `backtracking` times so the kernel has order 2^length.
    let a24c = epk.to_a24();
    let kernel = a24c.x_double_n(&kernel, u32::from(sig.backtracking));
    // Codomain of the even 2^length isogeny with that kernel = E_chall.
    let (chain, _) = IsogenyChain2e::new(
        a24c,
        kernel,
        u32::try_from(length).expect("length ≤ TORSION_EVEN_POWER fits u32"),
        None,
    );
    MontgomeryCurve::new(chain.codomain.to_affine_a())
}

/// Double every point of a basis `n` times — C `ec_dbl_iter_basis`. Reduces a
/// basis of order `2^k` to one of order `2^(k − n)`.
#[cfg(feature = "alloc")]
pub(crate) fn ec_dbl_iter_basis<F: BaseField>(
    bas: &crate::ec::couple::EcBasis<F>,
    n: u32,
    curve: &crate::ec::montgomery::MontgomeryCurve<F>,
) -> crate::ec::couple::EcBasis<F> {
    let a24 = curve.to_a24();
    crate::ec::couple::EcBasis::new(
        a24.x_double_n(&bas.p, n),
        a24.x_double_n(&bas.q, n),
        a24.x_double_n(&bas.p_minus_q, n),
    )
}

/// Recompute the canonical challenge and auxiliary bases for the dim-2 step.
/// Port of C `challenge_and_aux_basis_verify` (`verify.c`): from-hint bases of
/// `E_chall[2^f]` and `E_aux[2^f]`, doubled to the orders the commitment-curve
/// isogeny expects, then the signature's change matrix applied to the challenge
/// basis. Returns `(B_chall, B_aux)`, or `None` on order underflow / a failed
/// biladder. lvl1-pinned (`TORSION_EVEN_POWER = 248`, `HD_extra_torsion = 2`).
#[cfg(feature = "alloc")]
pub fn challenge_and_aux_basis_verify<P: crate::level_constants::LevelConstants>(
    e_chall: &crate::ec::montgomery::MontgomeryCurve<P::Field>,
    e_aux: &crate::ec::montgomery::MontgomeryCurve<P::Field>,
    sig: &SignatureData<P::Field>,
    pow_dim2_deg_resp: usize,
) -> Option<BasisPair<P::Field>> {
    use crate::ec::biscalar::ec_curve_to_basis_2f_from_hint;
    use crate::ec::couple::EcBasis;
    use crate::isogeny::endomorphism::matrix_application_even_basis;
    let torsion_even_power = P::F;
    const HD_EXTRA: usize = 2;
    let two_resp = usize::from(sig.two_resp_length);

    // Challenge basis: from-hint at full order, doubled down to order
    // 2^(pow_dim2_deg_resp + HD_extra + two_resp).
    let b_chall = ec_curve_to_basis_2f_from_hint::<P>(e_chall, torsion_even_power, sig.hint_chall);
    let chall_dbl = torsion_even_power
        .checked_sub(pow_dim2_deg_resp)?
        .checked_sub(HD_EXTRA)?
        .checked_sub(two_resp)?;
    let b_chall = ec_dbl_iter_basis(&b_chall, u32::try_from(chall_dbl).ok()?, e_chall);

    // Auxiliary basis: from-hint, doubled to order 2^(pow_dim2_deg_resp + HD_extra).
    let b_aux = ec_curve_to_basis_2f_from_hint::<P>(e_aux, torsion_even_power, sig.hint_aux);
    let aux_dbl = torsion_even_power
        .checked_sub(pow_dim2_deg_resp)?
        .checked_sub(HD_EXTRA)?;
    let b_aux = ec_dbl_iter_basis(&b_aux, u32::try_from(aux_dbl).ok()?, e_aux);

    // Apply the change matrix to the challenge basis (mod 2^f).
    let f = pow_dim2_deg_resp + HD_EXTRA + two_resp;
    let a24 = e_chall.a24();
    let (r, s, rms) = matrix_application_even_basis(
        &b_chall.p,
        &b_chall.q,
        &b_chall.p_minus_q,
        &sig.mat,
        f,
        &a24,
    )?;
    Some((EcBasis::new(r, s, rms), b_aux))
}

/// Apply the optional short `2^r` response isogeny (only when
/// `two_resp_length > 0`). Port of C `two_response_isogeny_verify` (`verify.c`):
/// pick the kernel generator (`B_chall.Q` if `mat[0][0]` and `mat[1][0]` are
/// both even, else `B_chall.P`), double it `pow_dim2_deg_resp + HD_extra` times
/// to order `2^two_resp_length`, take the `2^two_resp_length`-isogeny, and push
/// the challenge basis through it. Returns the updated `(E_chall, B_chall)`.
/// lvl1-pinned (`HD_extra_torsion = 2`).
#[cfg(feature = "alloc")]
pub fn two_response_isogeny_verify<F: BaseField>(
    e_chall: &crate::ec::montgomery::MontgomeryCurve<F>,
    b_chall: &crate::ec::couple::EcBasis<F>,
    sig: &SignatureData<F>,
    pow_dim2_deg_resp: usize,
) -> Option<(
    crate::ec::montgomery::MontgomeryCurve<F>,
    crate::ec::couple::EcBasis<F>,
)> {
    use crate::ec::couple::EcBasis;
    use crate::ec::montgomery::MontgomeryCurve;
    use crate::isogeny::two::IsogenyChain2e;
    const HD_EXTRA: usize = 2;

    // Kernel generator: Q if mat[0][0] and mat[1][0] are both even, else P.
    let m00_even = sig.mat[0][0].to_le_bytes()[0] & 1 == 0;
    let m10_even = sig.mat[1][0].to_le_bytes()[0] & 1 == 0;
    let ker0 = if m00_even && m10_even {
        b_chall.q
    } else {
        b_chall.p
    };

    let a24 = e_chall.to_a24();
    // Double down to a kernel of order 2^two_resp_length.
    let ker = a24.x_double_n(&ker0, u32::try_from(pow_dim2_deg_resp + HD_EXTRA).ok()?);
    let e = u32::from(sig.two_resp_length);

    // Build the 2^e-isogeny once, then push all three basis points through it.
    let (chain, _) = IsogenyChain2e::new(a24, ker, e, None);
    let push = |p: &crate::ec::montgomery::MontgomeryPoint<F>| {
        let mut q = *p;
        for step in &chain.steps {
            q = step.eval(&q);
        }
        q
    };
    let new_basis = EcBasis::new(push(&b_chall.p), push(&b_chall.q), push(&b_chall.p_minus_q));
    let e_chall_new = MontgomeryCurve::new(chain.codomain.to_affine_a());
    Some((e_chall_new, new_basis))
}

/// `true` iff both basis generators have order exactly 4 (a genuine `E[4]`
/// basis). Port of C `ec_is_basis_four_torsion` (used in the `pow == 0`
/// supersingularity check, which assumes `HD_extra_torsion == 2`).
#[cfg(feature = "alloc")]
fn is_basis_four_torsion<F: BaseField>(
    bas: &crate::ec::couple::EcBasis<F>,
    curve: &crate::ec::montgomery::MontgomeryCurve<F>,
) -> bool {
    let a24 = curve.to_a24();
    let order_4 = |p: &crate::ec::montgomery::MontgomeryPoint<F>| {
        // [4]P = O and [2]P ≠ O ⇒ order exactly 4.
        bool::from(a24.x_double_n(p, 2).is_infinity())
            && !bool::from(a24.x_double_n(p, 1).is_infinity())
    };
    order_4(&bas.p) && order_4(&bas.q)
}

/// Recover the commitment curve `E_com` from the codomain of the dim-2
/// `(2^n, 2^n)`-isogeny `E_chall × E_aux → E_com × E_aux'`, whose kernel is
/// `B_chall_can × B_aux_can`. Port of C `compute_commitment_curve_verify`
/// (`verify.c`). Returns `None` (invalid signature) if the chain's kernel does
/// not describe an isogeny between elliptic products (it must split). `E_com`
/// is always the first codomain factor. lvl1-pinned.
#[cfg(feature = "alloc")]
pub fn compute_commitment_curve_verify<F: BaseField>(
    b_chall: &crate::ec::couple::EcBasis<F>,
    b_aux: &crate::ec::couple::EcBasis<F>,
    e_chall: &crate::ec::montgomery::MontgomeryCurve<F>,
    e_aux: &crate::ec::montgomery::MontgomeryCurve<F>,
    pow_dim2_deg_resp: usize,
) -> Option<crate::ec::montgomery::MontgomeryCurve<F>> {
    use crate::ec::couple::{CoupleCurve, CoupleJacobianPoint, ThetaKernelCouplePoints};
    use crate::ec::jacobian::lift_basis;
    use crate::isogeny::theta_chain::theta_chain_compute_and_eval_verify;

    if pow_dim2_deg_resp == 0 {
        // No dim-2 computation; codomain = domain. Still require E_chall to be
        // supersingular: B_chall must be a genuine 4-torsion basis (HD_extra==2).
        if !is_basis_four_torsion(b_chall, e_chall) {
            return None;
        }
        return Some(*e_chall); // E_com = codomain.E1 = E_chall
    }

    // Lift both x-only bases to consistent Jacobian points; the chain seeds the
    // gluing kernel from T1, T2 only, so t1_minus_t2 is a placeholder.
    let (p1, q1) = lift_basis(b_chall, e_chall).ok()?;
    let (p2, q2) = lift_basis(b_aux, e_aux).ok()?;
    let ker = ThetaKernelCouplePoints::new(
        CoupleJacobianPoint::new(p1, p2),
        CoupleJacobianPoint::new(q1, q2),
        CoupleJacobianPoint::infinity(),
    );
    let e12 = CoupleCurve::new(*e_chall, *e_aux);
    let e34 = theta_chain_compute_and_eval_verify(
        u32::try_from(pow_dim2_deg_resp).ok()?,
        &e12,
        &ker,
        true,
        &[],
        &mut [],
    )?;
    // E_com is always the first factor by the (2^n,2^n)-isogeny formulae.
    Some(e34.e1)
}

/// Recompute the challenge coefficient by hashing `(j(E_pk), j(E_com), message)`.
/// Port of C `hash_to_challenge` (`common.c`). Because it hashes the
/// *j-invariants* (isomorphism invariants), the result is independent of the
/// curve *models* — so it is robust to the lift-sign choices in the dim-2 step.
///
/// The existing [`crate::hash::hash_to_challenge_scalar`] covers C's first
/// `HASH_ITERATIONS − 1` rounds (the `2·SECURITY_BITS`-wide rehash loop; the
/// per-round top-limb mask is a no-op at lvl1). This adds C's final round: one
/// more rehash squeezed to `(TORSION_EVEN_POWER − SQIsign_response_length)` bits
/// = 122 bits at lvl1, then reduced mod `2^SECURITY_BITS`. lvl1-pinned.
#[cfg(feature = "alloc")]
pub fn hash_to_challenge<P: crate::level_constants::LevelConstants>(
    pk_curve_a: &Fp2<P::Field>,
    e_com_a: &Fp2<P::Field>,
    message: &[u8],
) -> Uint<8> {
    use crate::ec::montgomery::MontgomeryCurve;
    use crate::hash::{Shake256, hash_to_challenge_scalar};
    let fp2_bytes = P::FP2_BYTES; // 64 lvl1 / 96 lvl3
    let chall_bytes = P::SECURITY_BITS / 8; // 16 lvl1 / 24 lvl3
    let scalar_bytes = 2 * P::SECURITY_BITS / 8; // 32 lvl1 / 48 lvl3
    let kept_bits = P::F - P::RESPONSE_BITS; // 122 lvl1 / 184 lvl3

    // j-invariants of the public and commitment curves, encoded as fp2 bytes.
    let mut j_pk = [0u8; 96];
    let mut j_com = [0u8; 96];
    MontgomeryCurve::new(*pk_curve_a)
        .j_invariant()
        .to_bytes_le(&mut j_pk[..fp2_bytes]);
    MontgomeryCurve::new(*e_com_a)
        .j_invariant()
        .to_bytes_le(&mut j_com[..fp2_bytes]);

    // C rounds 1 .. HASH_ITERATIONS−1 (2·SECURITY_BITS bytes wide).
    let mut scalar = [0u8; 48];
    hash_to_challenge_scalar::<P>(
        &j_pk[..fp2_bytes],
        &j_com[..fp2_bytes],
        message,
        &mut scalar[..scalar_bytes],
    );

    // C final round: rehash, squeeze `kept_bits = TORSION_EVEN_POWER −
    // response_length` low bits of a `chall_bytes`-wide little-endian value.
    let mut h = Shake256::new();
    h.absorb(&scalar[..scalar_bytes]);
    let mut chall = [0u8; 24];
    h.finalize_into(&mut chall[..chall_bytes]);
    // Mask to the low `kept_bits` bits (lvl1: 122 → low 2 bits of byte 15).
    let full = kept_bits / 8;
    let rem = kept_bits % 8;
    if full < chall_bytes {
        chall[full] &= (1u8 << rem).wrapping_sub(1);
        for b in chall.iter_mut().take(chall_bytes).skip(full + 1) {
            *b = 0;
        }
    }
    // mod 2^SECURITY_BITS is a no-op (chall is exactly that wide).
    let mut buf = [0u8; 64];
    buf[..chall_bytes].copy_from_slice(&chall[..chall_bytes]);
    Uint::<8>::from_le_slice(&buf)
}

/// Top-level SQIsign verification — C `protocols_verify` (`verify.c`). Chains
/// every verify step and returns `true` iff the recomputed challenge matches
/// the one carried by the signature. lvl1-only. `sig_bytes` / `pk_bytes` are the
/// 148-byte / 65-byte wire encodings.
#[cfg(feature = "alloc")]
pub fn protocols_verify<P: crate::level_constants::LevelConstants>(
    sig_bytes: &[u8],
    pk_bytes: &[u8],
    message: &[u8],
) -> bool {
    use crate::ec::montgomery::MontgomeryCurve;
    let sig = match SignatureData::<P::Field>::from_bytes::<P>(sig_bytes) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let pk = match PublicKeyData::<P::Field>::from_bytes::<P>(pk_bytes) {
        Ok(p) => p,
        Err(_) => return false,
    };

    // 1. Canonical basis-change matrix.
    if !check_canonical_basis_change_matrix::<P>(&sig) {
        #[cfg(feature = "std")]
        eprintln!("VERIFY FAIL: 1 canonical_basis_change_matrix");
        return false;
    }
    // 2. Length of the dim-2 isogeny; reject too-long / length-1 responses.
    let response = match i64::try_from(P::RESPONSE_BITS) {
        Ok(r) => r,
        Err(_) => return false,
    };
    let pow = response - i64::from(sig.two_resp_length) - i64::from(sig.backtracking);
    if pow < 0 || pow == 1 {
        #[cfg(feature = "std")]
        eprintln!("VERIFY FAIL: 2 pow={pow}");
        return false;
    }
    let pow = match usize::try_from(pow) {
        Ok(p) => p,
        Err(_) => return false,
    };
    // 3-4. Public and auxiliary curves must be valid Montgomery curves.
    if !ec_curve_verify_a(&pk.curve_a) || !ec_curve_verify_a(&sig.e_aux_a) {
        #[cfg(feature = "std")]
        eprintln!("VERIFY FAIL: 3-4 curve validity");
        return false;
    }
    let e_pk = MontgomeryCurve::new(pk.curve_a);
    let e_aux = MontgomeryCurve::new(sig.e_aux_a);

    // 5. Challenge curve.
    let mut e_chall = compute_challenge_verify::<P>(&e_pk, &sig, pk.hint_pk);
    // 6. Canonical challenge + auxiliary bases (matrix applied to challenge).
    let (mut b_chall, b_aux) =
        match challenge_and_aux_basis_verify::<P>(&e_chall, &e_aux, &sig, pow) {
            Some(x) => x,
            None => return false,
        };
    // 7. Optional short 2^r response isogeny.
    if sig.two_resp_length > 0 {
        match two_response_isogeny_verify(&e_chall, &b_chall, &sig, pow) {
            Some((e2, b2)) => {
                e_chall = e2;
                b_chall = b2;
            }
            None => return false,
        }
    }
    // 8. Commitment curve via the dim-2 isogeny.
    let e_com = match compute_commitment_curve_verify(&b_chall, &b_aux, &e_chall, &e_aux, pow) {
        Some(c) => c,
        None => return false,
    };
    // 9-10. Recompute the challenge and compare.
    hash_to_challenge::<P>(&pk.curve_a, &e_com.a, message) == sig.chall_coeff
}

/// Per-level dispatch for full signature verification. Bridges the runtime
/// `P::LEVEL` match in [`crate::verify`] to the compile-time `LevelConstants`
/// bound that [`protocols_verify`] requires: Level 1 and Level 3 run the generic
/// `protocols_verify`; Level 5 is not yet implemented. This is the verify-side
/// analogue of [`crate::keypair::KeyLevel`].
pub trait VerifyLevel: crate::params::Params {
    /// Verify `sig` over `msg` against public key `pk` (both wire-encoded).
    fn verify_bytes(sig: &[u8], pk: &[u8], msg: &[u8]) -> Result<()>;
}

#[cfg(feature = "alloc")]
impl VerifyLevel for Level1 {
    fn verify_bytes(sig: &[u8], pk: &[u8], msg: &[u8]) -> Result<()> {
        if protocols_verify::<Level1>(sig, pk, msg) {
            Ok(())
        } else {
            Err(Error::InvalidSignature)
        }
    }
}

#[cfg(feature = "alloc")]
impl VerifyLevel for crate::params::lvl3::Level3 {
    fn verify_bytes(sig: &[u8], pk: &[u8], msg: &[u8]) -> Result<()> {
        if protocols_verify::<crate::params::lvl3::Level3>(sig, pk, msg) {
            Ok(())
        } else {
            Err(Error::InvalidSignature)
        }
    }
}

#[cfg(feature = "alloc")]
impl VerifyLevel for crate::params::lvl5::Level5 {
    fn verify_bytes(_sig: &[u8], _pk: &[u8], _msg: &[u8]) -> Result<()> {
        Err(Error::Unimplemented("verify: level 5 not implemented"))
    }
}

// Without `alloc` the orchestration helpers are absent; every level reports
// the missing capability rather than a verification verdict.
#[cfg(not(feature = "alloc"))]
impl VerifyLevel for Level1 {
    fn verify_bytes(_sig: &[u8], _pk: &[u8], _msg: &[u8]) -> Result<()> {
        Err(Error::Unimplemented("verify: requires the alloc feature"))
    }
}

#[cfg(not(feature = "alloc"))]
impl VerifyLevel for crate::params::lvl3::Level3 {
    fn verify_bytes(_sig: &[u8], _pk: &[u8], _msg: &[u8]) -> Result<()> {
        Err(Error::Unimplemented("verify: requires the alloc feature"))
    }
}

#[cfg(not(feature = "alloc"))]
impl VerifyLevel for crate::params::lvl5::Level5 {
    fn verify_bytes(_sig: &[u8], _pk: &[u8], _msg: &[u8]) -> Result<()> {
        Err(Error::Unimplemented("verify: requires the alloc feature"))
    }
}

/// Sign — the short `2^r` response isogeny (`two_resp_length > 0`). Port of C
/// `compute_small_chain_isogeny_signature` (`sign.c:304`): the response (of
/// norm `2^two_resp`) gives a kernel via `id2iso_ideal_to_kernel_dlogs_even`;
/// build that kernel point on `B_chall_2` reduced to order `2^two_resp`, take
/// the `2^two_resp`-isogeny, and push the (original) challenge basis through it.
/// Returns the updated `(E_chall_2, B_chall_2)`. lvl1.
#[cfg(feature = "alloc")]
pub fn compute_small_chain_isogeny_signature<P: crate::level_constants::LevelConstants>(
    e_chall_2: &crate::ec::montgomery::MontgomeryCurve<P::Field>,
    b_chall_2: &crate::ec::couple::EcBasis<P::Field>,
    resp_o0: &[crypto_bigint::Int<16>; 4],
    pow_dim2: u32,
    two_resp_length: u32,
) -> Option<CurveBasis<P::Field>> {
    use crate::ec::biscalar::ec_biscalar_mul;
    use crate::ec::couple::EcBasis;
    use crate::ec::montgomery::MontgomeryCurve;
    use crate::isogeny::endomorphism::id2iso_ideal_to_kernel_dlogs_even;
    use crate::isogeny::two::IsogenyChain2e;
    const HD: u32 = 2;

    let length = usize::try_from(two_resp_length).ok()?;
    // Kernel coordinates of the response ideal (norm 2^two_resp).
    let vec2 = id2iso_ideal_to_kernel_dlogs_even::<P, 16>(resp_o0, length);
    // Reduce the challenge basis to order 2^two_resp; form the kernel point.
    let b_red = ec_dbl_iter_basis(b_chall_2, pow_dim2 + HD, e_chall_2);
    let a24 = e_chall_2.a24();
    let ker = ec_biscalar_mul(
        &vec2[0].to_le_bytes(),
        &vec2[1].to_le_bytes(),
        length,
        &b_red.p,
        &b_red.q,
        &b_red.p_minus_q,
        &a24,
    )?;
    // 2^two_resp-isogeny; push the original challenge basis through it.
    let a24c = e_chall_2.to_a24();
    let (chain, _) = IsogenyChain2e::new(a24c, ker, two_resp_length, None);
    let push = |p: &crate::ec::montgomery::MontgomeryPoint<P::Field>| {
        let mut q = *p;
        for step in &chain.steps {
            q = step.eval(&q);
        }
        q
    };
    let new_basis = EcBasis::new(
        push(&b_chall_2.p),
        push(&b_chall_2.q),
        push(&b_chall_2.p_minus_q),
    );
    Some((
        MontgomeryCurve::new(chain.codomain.to_affine_a()),
        new_basis,
    ))
}

/// Sign step 5 — recompute the challenge curve `E_chall` from the secret curve
/// and `chall_coeff`, then map the dim-2 output basis `B_chall_2` (on the
/// isomorphic `E_chall_2`) onto `E_chall` via the curve isomorphism. Port of C
/// `compute_challenge_codomain_signature` (`sign.c:362`). The `E_chall`
/// recomputation is identical to [`compute_challenge_verify`] but uses the
/// supplied canonical basis directly. Returns `(E_chall, B_chall_2_on_E_chall)`.
/// lvl1-pinned.
#[cfg(feature = "alloc")]
pub fn compute_challenge_codomain_signature<P: crate::level_constants::LevelConstants>(
    sk_curve_a: &Fp2<P::Field>,
    sk_canonical_basis: &crate::ec::couple::EcBasis<P::Field>,
    chall_coeff: &Uint<8>,
    backtracking: u8,
    e_chall_2_a: &Fp2<P::Field>,
    b_chall_2: &crate::ec::couple::EcBasis<P::Field>,
) -> Option<CurveBasis<P::Field>> {
    use crate::ec::couple::EcBasis;
    use crate::ec::montgomery::{MontgomeryCurve, MontgomeryPoint};
    use crate::isogeny::two::IsogenyChain2e;
    let torsion_even_power = P::F;

    let curve = MontgomeryCurve::new(*sk_curve_a);
    let length = torsion_even_power - usize::from(backtracking);
    // kernel = P + [chall_coeff]·Q, doubled `backtracking` times.
    let a24 = curve.a24();
    let kernel = MontgomeryPoint::ladder3pt(
        &sk_canonical_basis.p,
        &sk_canonical_basis.q,
        &sk_canonical_basis.p_minus_q,
        &chall_coeff.to_le_bytes(),
        &a24,
    );
    let a24c = curve.to_a24();
    let kernel = a24c.x_double_n(&kernel, u32::from(backtracking));
    let (chain, _) = IsogenyChain2e::new(
        a24c,
        kernel,
        u32::try_from(length).expect("length ≤ 248 fits u32"),
        None,
    );
    let e_chall = MontgomeryCurve::new(chain.codomain.to_affine_a());

    // Map B_chall_2 (on the isomorphic E_chall_2) onto E_chall.
    let isom = ec_isomorphism(e_chall_2_a, &e_chall.a)?;
    let b_mapped = EcBasis::new(
        apply_iso(&b_chall_2.p, &isom),
        apply_iso(&b_chall_2.q, &isom),
        apply_iso(&b_chall_2.p_minus_q, &isom),
    );
    Some((e_chall, b_mapped))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn compute_challenge_codomain_signature_recomputes_and_maps() {
        use crate::ec::biscalar::ec_basis_e0_2f;
        use crate::ec::montgomery::MontgomeryCurve;
        // E0 secret curve, its canonical basis; small challenge, no backtracking.
        let e0 = MontgomeryCurve::<Fp1Element>::e0();
        let basis = ec_basis_e0_2f::<Level1>(248);
        let c = Uint::<8>::from_u64(98765);
        // Learn E_chall via compute_challenge_verify (same chain, E0 basis from
        // the E0 hint branch), then map with E_chall_2 == E_chall (identity isom).
        let sig = SignatureData {
            chall_coeff: c,
            ..small_sig([[Uint::<8>::ONE; 2]; 2], 0)
        };
        let e_chall = compute_challenge_verify::<Level1>(&e0, &sig, 0);
        let (e_chall2, b_mapped) = compute_challenge_codomain_signature::<Level1>(
            &e0.a, &basis, &c, 0, &e_chall.a, &basis,
        )
        .expect("codomain + identity map");
        assert_eq!(e_chall2.a, e_chall.a, "E_chall is deterministic");
        // Identity isomorphism (E_chall → E_chall) preserves the basis x-coords.
        assert_eq!(b_mapped.p.affine_x(), basis.p.affine_x(), "P x preserved");
        assert_eq!(b_mapped.q.affine_x(), basis.q.affine_x(), "Q x preserved");
    }

    #[test]
    fn ec_isomorphism_self_is_identity() {
        use crate::ec::montgomery::MontgomeryPoint;
        // Isomorphism E → E is the identity projectively (Nz = 0, Nx = D), so a
        // point's affine x is unchanged.
        let a = Fp2::<Fp1Element>::new(
            crate::ec::biscalar::fp_small::<Fp1Element>(5),
            crate::ec::biscalar::fp_small::<Fp1Element>(3),
        );
        let isom = ec_isomorphism(&a, &a).expect("self-isom exists");
        assert!(bool::from(isom.1.is_zero()), "Nz = 0 for E → E");
        assert_eq!(isom.0, isom.2, "Nx = D for E → E");
        let p = MontgomeryPoint::<Fp1Element>::new(
            Fp2::<Fp1Element>::new(
                crate::ec::biscalar::fp_small::<Fp1Element>(7),
                Fp1Element::zero(),
            ),
            Fp2::<Fp1Element>::one(),
        );
        let q = apply_iso(&p, &isom);
        assert_eq!(q.affine_x(), p.affine_x(), "identity isom preserves x");
    }

    #[test]
    fn ec_curve_verify_a_rejects_plus_minus_two() {
        let two = Fp2::<Fp1Element>::one().double();
        assert!(!ec_curve_verify_a(&two), "A = 2 is invalid");
        assert!(!ec_curve_verify_a(&two.negate()), "A = −2 is invalid");
        // A = 0 (E0) and a generic A are valid.
        assert!(ec_curve_verify_a(&Fp2::<Fp1Element>::zero()), "A = 0 valid");
        assert!(ec_curve_verify_a(&Fp2::<Fp1Element>::img()), "A = i valid",);
    }

    #[test]
    fn signature_decode_lvl1_round_trips_fields() {
        // Hand-build a 148-byte lvl1 signature buffer with known fields.
        let mut buf = [0u8; SIG_BYTES_LVL1];
        // E_aux_A = (3 + 7·i): re = 3, im = 7 (little-endian, first byte).
        buf[0] = 3;
        buf[FP2_BYTES_LVL1 / 2] = 7;
        buf[64] = 2; // backtracking
        buf[65] = 5; // two_resp_length
        // matrix entries m00=1, m01=2, m10=3, m11=4 (LE, first byte of each).
        buf[66] = 1;
        buf[66 + MAT_ENTRY_BYTES_LVL1] = 2;
        buf[66 + 2 * MAT_ENTRY_BYTES_LVL1] = 3;
        buf[66 + 3 * MAT_ENTRY_BYTES_LVL1] = 4;
        // chall_coeff = 0x0102 = 258 (LE).
        buf[130] = 2;
        buf[131] = 1;
        buf[146] = 0x0b; // hint_aux
        buf[147] = 0x17; // hint_chall

        let sig = SignatureData::from_bytes_lvl1(&buf).expect("decode succeeds");
        let expected_e_aux = Fp2::<Fp1Element>::new(
            crate::ec::biscalar::fp_small::<Fp1Element>(3),
            crate::ec::biscalar::fp_small::<Fp1Element>(7),
        );
        assert!(bool::from(sig.e_aux_a.ct_eq(&expected_e_aux)), "E_aux_A");
        assert_eq!(sig.backtracking, 2);
        assert_eq!(sig.two_resp_length, 5);
        assert_eq!(sig.mat[0][0], Uint::<8>::from_u8(1));
        assert_eq!(sig.mat[0][1], Uint::<8>::from_u8(2));
        assert_eq!(sig.mat[1][0], Uint::<8>::from_u8(3));
        assert_eq!(sig.mat[1][1], Uint::<8>::from_u8(4));
        assert_eq!(sig.chall_coeff, Uint::<8>::from_u16(258));
        assert_eq!(sig.hint_aux, 0x0b);
        assert_eq!(sig.hint_chall, 0x17);
    }

    fn small_sig(mat: [[Uint<8>; 2]; 2], backtracking: u8) -> SignatureData<Fp1Element> {
        SignatureData {
            e_aux_a: Fp2::<Fp1Element>::zero(),
            backtracking,
            two_resp_length: 0,
            mat,
            chall_coeff: Uint::<8>::ZERO,
            hint_aux: 0,
            hint_chall: 0,
        }
    }

    #[test]
    fn check_canonical_matrix_bounds_entries() {
        let ones = [[Uint::<8>::from_u8(1); 2]; 2];
        // backtracking = 0 ⇒ bound = 2^128; small entries accepted.
        assert!(check_canonical_basis_change_matrix::<Level1>(&small_sig(
            ones, 0
        )));
        // An entry equal to the bound (2^128) is rejected (C: bound ≤ entry).
        let mut m = ones;
        m[1][1] = Uint::<8>::ONE.shl_vartime(128);
        assert!(!check_canonical_basis_change_matrix::<Level1>(&small_sig(
            m, 0
        )));
        // Just below the bound (2^128 − 1) is accepted.
        let mut m2 = ones;
        m2[0][1] = Uint::<8>::ONE
            .shl_vartime(128)
            .wrapping_sub(&Uint::<8>::ONE);
        assert!(check_canonical_basis_change_matrix::<Level1>(&small_sig(
            m2, 0
        )));
        // backtracking shrinks the bound: backtracking = 128 ⇒ bound = 2^0 = 1,
        // so any non-zero entry is rejected.
        assert!(!check_canonical_basis_change_matrix::<Level1>(&small_sig(
            ones, 128
        )));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn compute_challenge_verify_e0_produces_valid_distinct_curve() {
        use crate::ec::montgomery::MontgomeryCurve;
        // E0 public curve (A = 0) ⇒ the E0 basis branch; small odd challenge.
        let epk = MontgomeryCurve::<Fp1Element>::e0();
        let sig = small_sig([[Uint::<8>::ONE; 2]; 2], 0);
        let sig = SignatureData {
            chall_coeff: Uint::<8>::from_u64(12345),
            ..sig
        };
        // hint_pk is ignored on E0 (the basis comes from the precomputed E0 set).
        let e_chall = compute_challenge_verify::<Level1>(&epk, &sig, 0);
        // A 2^248-isogeny lands on a valid curve, genuinely off E0.
        assert!(ec_curve_verify_a(&e_chall.a), "E_chall is a valid curve");
        assert_ne!(
            e_chall.a,
            Fp2::<Fp1Element>::zero(),
            "the challenge isogeny moved the curve off E0",
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn challenge_and_aux_basis_identity_matrix_preserves_reduced_basis() {
        use crate::ec::biscalar::ec_basis_e0_2f;
        use crate::ec::montgomery::MontgomeryCurve;
        // Identity change matrix [[1,0],[0,1]] must leave the (reduced)
        // challenge basis unchanged: R=[1]P+[0]Q=P, S=[0]P+[1]Q=Q,
        // R−S=[1]P+[−1]Q=P−Q.
        let identity = [
            [Uint::<8>::ONE, Uint::<8>::ZERO],
            [Uint::<8>::ZERO, Uint::<8>::ONE],
        ];
        let sig = small_sig(identity, 0); // two_resp_length = 0
        let e0 = MontgomeryCurve::<Fp1Element>::e0();
        let pow = 240usize; // ⇒ chall/aux doublings = 248−240−2 = 6, f = 242
        let (b_chall, _b_aux) =
            challenge_and_aux_basis_verify::<Level1>(&e0, &e0, &sig, pow).expect("bases recompute");

        // Independent expected reduced basis: E0 basis doubled 6 times.
        let a24 = e0.to_a24();
        let exp = ec_basis_e0_2f::<Level1>(248);
        let exp_p = a24.x_double_n(&exp.p, 6);
        let exp_q = a24.x_double_n(&exp.q, 6);
        let exp_pmq = a24.x_double_n(&exp.p_minus_q, 6);
        assert_eq!(b_chall.p.affine_x(), exp_p.affine_x(), "identity keeps P");
        assert_eq!(b_chall.q.affine_x(), exp_q.affine_x(), "identity keeps Q");
        assert_eq!(
            b_chall.p_minus_q.affine_x(),
            exp_pmq.affine_x(),
            "identity keeps P−Q",
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn two_response_isogeny_moves_curve_and_pushes_basis() {
        use crate::ec::biscalar::{ec_basis_e0_2f, is_on_curve};
        use crate::ec::montgomery::MontgomeryCurve;
        // Input challenge basis of order 2^8 on E0 (pow=4, two_resp=2 ⇒
        // order = 2^(HD_extra2 + pow4 + two_resp2) = 2^8).
        let e0 = MontgomeryCurve::<Fp1Element>::e0();
        let a24 = e0.to_a24();
        let base = ec_basis_e0_2f::<Level1>(248);
        let b8 = crate::ec::couple::EcBasis::new(
            a24.x_double_n(&base.p, 240),
            a24.x_double_n(&base.q, 240),
            a24.x_double_n(&base.p_minus_q, 240),
        );
        // mat[0][0] odd ⇒ kernel = P branch.
        let mat = [
            [Uint::<8>::ONE, Uint::<8>::ZERO],
            [Uint::<8>::ZERO, Uint::<8>::ONE],
        ];
        let sig = SignatureData {
            two_resp_length: 2,
            ..small_sig(mat, 0)
        };
        let (e_new, b_new) =
            two_response_isogeny_verify(&e0, &b8, &sig, 4).expect("short isogeny runs");
        // Degree-4 isogeny lands on a valid curve, off E0.
        assert!(ec_curve_verify_a(&e_new.a), "codomain is a valid curve");
        assert_ne!(e_new.a, Fp2::<Fp1Element>::zero(), "moved off E0");
        // Pushed basis points are non-trivial and lie on the codomain.
        assert!(!bool::from(b_new.p.z.is_zero()), "image of P is finite");
        assert!(!bool::from(b_new.q.z.is_zero()), "image of Q is finite");
        assert!(
            bool::from(is_on_curve(&b_new.p.affine_x(), &e_new.a)),
            "image of P is on the codomain",
        );
        assert!(
            bool::from(is_on_curve(&b_new.q.affine_x(), &e_new.a)),
            "image of Q is on the codomain",
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn compute_commitment_curve_pow0_requires_four_torsion() {
        use crate::ec::biscalar::ec_basis_e0_2f;
        use crate::ec::couple::EcBasis;
        use crate::ec::montgomery::MontgomeryCurve;
        let e0 = MontgomeryCurve::<Fp1Element>::e0();
        let a24 = e0.to_a24();
        let base = ec_basis_e0_2f::<Level1>(248);
        // Order-4 basis (doubled 246 ⇒ order 2^2): pow==0 ⇒ E_com = E_chall = E0.
        let b4 = EcBasis::new(
            a24.x_double_n(&base.p, 246),
            a24.x_double_n(&base.q, 246),
            a24.x_double_n(&base.p_minus_q, 246),
        );
        let e_com = compute_commitment_curve_verify(&b4, &b4, &e0, &e0, 0)
            .expect("4-torsion basis accepted");
        assert_eq!(e_com.a, e0.a, "E_com = E_chall when pow == 0");
        // Order-2 basis (doubled 247) ⇒ not 4-torsion ⇒ rejected.
        let b2 = EcBasis::new(
            a24.x_double_n(&base.p, 247),
            a24.x_double_n(&base.q, 247),
            a24.x_double_n(&base.p_minus_q, 247),
        );
        assert!(
            compute_commitment_curve_verify(&b2, &b2, &e0, &e0, 0).is_none(),
            "non-4-torsion basis rejected at pow == 0",
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn compute_commitment_curve_pow_positive_completes() {
        use crate::ec::biscalar::ec_basis_e0_2f;
        use crate::ec::couple::EcBasis;
        use crate::ec::montgomery::MontgomeryCurve;
        // Bases of order 2^(HD_extra2 + pow4) = 2^6 on E0; the dim-2 verify
        // chain must run to completion (Some ⇒ valid curve, None ⇒ no split)
        // without panicking on the empty eval-point list.
        let e0 = MontgomeryCurve::<Fp1Element>::e0();
        let a24 = e0.to_a24();
        let base = ec_basis_e0_2f::<Level1>(248);
        let b6 = EcBasis::new(
            a24.x_double_n(&base.p, 242),
            a24.x_double_n(&base.q, 242),
            a24.x_double_n(&base.p_minus_q, 242),
        );
        let result = compute_commitment_curve_verify(&b6, &b6, &e0, &e0, 4);
        if let Some(e_com) = result {
            assert!(ec_curve_verify_a(&e_com.a), "if it splits, E_com is valid");
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn hash_to_challenge_deterministic_and_sensitive() {
        let pk_a = Fp2::<Fp1Element>::new(
            crate::ec::biscalar::fp_small::<Fp1Element>(5),
            crate::ec::biscalar::fp_small::<Fp1Element>(9),
        );
        let com_a = Fp2::<Fp1Element>::new(
            crate::ec::biscalar::fp_small::<Fp1Element>(7),
            crate::ec::biscalar::fp_small::<Fp1Element>(2),
        );
        let h1 = hash_to_challenge::<Level1>(&pk_a, &com_a, b"msg");
        let h2 = hash_to_challenge::<Level1>(&pk_a, &com_a, b"msg");
        assert_eq!(h1, h2, "hash is deterministic");
        // Changing the commitment curve changes the challenge.
        let com_a2 = Fp2::<Fp1Element>::new(
            crate::ec::biscalar::fp_small::<Fp1Element>(11),
            crate::ec::biscalar::fp_small::<Fp1Element>(2),
        );
        assert_ne!(
            h1,
            hash_to_challenge::<Level1>(&pk_a, &com_a2, b"msg"),
            "E_com matters"
        );
        // Output is reduced below 2^122 (top 6 bits of the 128-bit value clear).
        let mut bytes = [0u8; 64];
        h1.to_le_bytes()
            .iter()
            .enumerate()
            .for_each(|(i, b)| bytes[i] = *b);
        assert_eq!(bytes[15] & 0xfc, 0, "challenge masked to 122 bits");
        assert!(bytes[16..].iter().all(|&b| b == 0), "challenge < 2^128");
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn protocols_verify_rejects_non_matching_signature_without_panic() {
        // E0 public key (A = 0, hint 0): 65 zero bytes.
        let pk = [0u8; PK_BYTES_LVL1];
        // A well-formed-but-non-matching signature: E_aux = E0, identity change
        // matrix (canonical), tiny challenge. Runs the full verify pipeline
        // (challenge isogeny → bases → dim-2 commitment → hash) and must return
        // false (the recomputed challenge won't equal the arbitrary one).
        let mut sig = [0u8; SIG_BYTES_LVL1];
        sig[66] = 1; // mat[0][0] = 1
        sig[66 + 3 * MAT_ENTRY_BYTES_LVL1] = 1; // mat[1][1] = 1
        sig[130] = 1; // chall_coeff = 1
        assert!(
            !protocols_verify::<Level1>(&sig, &pk, b"hello"),
            "non-matching signature rejected",
        );
    }

    #[test]
    fn signature_encode_decode_round_trips() {
        // Build a structured signature, encode, decode, and compare fields.
        let sig = SignatureData {
            e_aux_a: Fp2::<Fp1Element>::new(
                crate::ec::biscalar::fp_small::<Fp1Element>(13),
                crate::ec::biscalar::fp_small::<Fp1Element>(21),
            ),
            backtracking: 3,
            two_resp_length: 7,
            mat: [
                [Uint::<8>::from_u64(0x1234), Uint::<8>::from_u64(0x5678)],
                [Uint::<8>::from_u64(0x9abc), Uint::<8>::from_u64(0xdef0)],
            ],
            chall_coeff: Uint::<8>::from_u64(0x00ab_cdef),
            hint_aux: 0x2a,
            hint_chall: 0x3b,
        };
        let mut buf = [0u8; SIG_BYTES_LVL1];
        sig.to_bytes_lvl1(&mut buf).expect("encode");
        let decoded = SignatureData::from_bytes_lvl1(&buf).expect("decode");
        assert_eq!(decoded, sig, "signature survives encode→decode");

        // And bytes→struct→bytes is a fixed point.
        let mut buf2 = [0u8; SIG_BYTES_LVL1];
        decoded.to_bytes_lvl1(&mut buf2).expect("re-encode");
        assert_eq!(buf, buf2, "encode is stable");
    }

    #[test]
    fn public_key_encode_decode_round_trips() {
        let pk = PublicKeyData {
            curve_a: Fp2::<Fp1Element>::new(
                crate::ec::biscalar::fp_small::<Fp1Element>(4),
                crate::ec::biscalar::fp_small::<Fp1Element>(99),
            ),
            hint_pk: 0x0b,
        };
        let mut buf = [0u8; PK_BYTES_LVL1];
        pk.to_bytes_lvl1(&mut buf).expect("encode");
        assert_eq!(PublicKeyData::from_bytes_lvl1(&buf).expect("decode"), pk);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn dlog_2f_recovers_known_exponent() {
        use crate::ec::biscalar::ec_basis_e0_2f;
        use crate::ec::montgomery::MontgomeryCurve;
        use crate::ec::weil::weil;
        // A primitive 2^e root of unity from the Weil pairing of the E0 basis.
        let e: u32 = 24;
        let e0 = MontgomeryCurve::<Fp1Element>::e0();
        let b = ec_basis_e0_2f::<Level1>(e as usize);
        let pq = b.p.x_add(&b.q, &b.p_minus_q); // x(P + Q)
        let zeta = weil(e, &b.p, &b.q, &pq, &e0);
        assert!(!bool::from(zeta.is_one()), "pairing is a non-trivial root");

        for x_known in [1u32, 2, 39612, (1 << e) - 1] {
            let t = zeta.pow_vartime(&x_known.to_le_bytes());
            let recovered = dlog_2f(&zeta, &t, e);
            assert_eq!(
                recovered,
                Uint::<8>::from_u32(x_known),
                "dlog recovers exponent {x_known}",
            );
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn change_of_basis_matrix_inverts_matrix_application() {
        use crate::ec::biscalar::ec_basis_e0_2f;
        use crate::ec::couple::EcBasis;
        use crate::ec::montgomery::MontgomeryCurve;
        use crate::isogeny::endomorphism::matrix_application_even_basis;
        let f: u32 = 32;
        let e0 = MontgomeryCurve::<Fp1Element>::e0();
        let a24 = e0.a24();
        // Canonical order-2^f basis b2 on E0.
        let b2 = ec_basis_e0_2f::<Level1>(f as usize);
        // Apply a known invertible matrix M0 (det 13 odd) to get a basis b1.
        let m0 = [
            [Uint::<8>::from_u8(3), Uint::<8>::from_u8(1)],
            [Uint::<8>::from_u8(2), Uint::<8>::from_u8(5)],
        ];
        let (b1p, b1q, b1pmq) =
            matrix_application_even_basis(&b2.p, &b2.q, &b2.p_minus_q, &m0, f as usize, &a24)
                .expect("apply M0");
        let b1 = EcBasis::new(b1p, b1q, b1pmq);
        // change_of_basis must recover M0 (it inverts matrix_application).
        let m = change_of_basis_matrix::<Level1>(&b1, &b2, &e0, f).expect("change of basis");
        assert_eq!(m, m0, "change_of_basis_matrix recovers the applied matrix");
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn compute_and_set_basis_change_matrix_round_trips_known_matrix() {
        use crate::ec::biscalar::ec_curve_to_basis_2f_to_hint;
        use crate::ec::montgomery::MontgomeryCurve;
        use crate::isogeny::endomorphism::matrix_application_even_basis;
        // E_chall = E_aux_2 = E0; work at order 2^f.
        let f: usize = 32;
        let e0 = MontgomeryCurve::<Fp1Element>::e0();
        let a24 = e0.a24();
        let e_diff = u32::try_from(248 - f).expect("difference fits in u32");

        // Canonical challenge basis reduced to order 2^f.
        let (b_can_chall, _h) = ec_curve_to_basis_2f_to_hint::<Level1>(&e0, 248);
        let b_can_chall_f = ec_dbl_iter_basis(&b_can_chall, e_diff, &e0);

        // Supplied challenge basis = M_known applied to the canonical basis.
        let m_known = [
            [Uint::<8>::from_u8(5), Uint::<8>::from_u8(2)],
            [Uint::<8>::from_u8(3), Uint::<8>::from_u8(7)],
        ];
        let (cp, cq, cpmq) = matrix_application_even_basis(
            &b_can_chall_f.p,
            &b_can_chall_f.q,
            &b_can_chall_f.p_minus_q,
            &m_known,
            f,
            &a24,
        )
        .expect("apply M_known");
        let b_chall_2 = crate::ec::couple::EcBasis::new(cp, cq, cpmq);

        // Auxiliary basis = the reduced canonical basis ⇒ M_aux = identity.
        let b_aux_2 = b_can_chall_f;

        let mut sig = small_sig([[Uint::<8>::ZERO; 2]; 2], 0);
        assert!(compute_and_set_basis_change_matrix::<Level1>(
            &mut sig, &b_aux_2, &b_chall_2, &e0, &e0, f
        ));
        // With M_aux = identity, sig.mat must recover M_known.
        assert_eq!(sig.mat, m_known, "basis-change matrix recovers M_known");
        assert_eq!(sig.hint_chall, 0, "E0 challenge hint is 0");
        assert_eq!(sig.hint_aux, 0, "E0 aux hint is 0");
    }

    #[test]
    fn decode_int_le_handles_sign() {
        use crypto_bigint::Int;
        assert_eq!(
            decode_int_le::<8>(&[5, 0, 0, 0], false),
            Int::<8>::from_i64(5)
        );
        // −2 as 4-byte two's complement.
        assert_eq!(
            decode_int_le::<8>(&[0xFE, 0xFF, 0xFF, 0xFF], true),
            Int::<8>::from_i64(-2),
        );
        // −2 as 32-byte two's complement.
        let mut neg2 = [0xFFu8; 32];
        neg2[0] = 0xFE;
        assert_eq!(decode_int_le::<8>(&neg2, true), Int::<8>::from_i64(-2));
        // High bit set but unsigned ⇒ large positive.
        assert!(!bool::from(
            decode_int_le::<8>(&[0, 0, 0, 0x80], false).is_negative()
        ));
    }

    #[test]
    fn secret_key_decode_lvl1_reconstructs_secret_ideal() {
        use crate::quaternion::o0_mul::multiply_o0_basis;
        // Hand-build a 353-byte SK: E0 pk, gen = 1 + 2i (N_red = 5), norm = 5,
        // identity-ish change matrix.
        let mut buf = [0u8; SK_BYTES_LVL1];
        // pk: A = 0 (E0), hint_pk = 0 (bytes already zero).
        buf[65] = 5; // secret_ideal.norm = 5
        buf[97] = 1; // gen.coord[0] = 1
        buf[97 + SK_FIELD_BYTES_LVL1] = 2; // gen.coord[1] = 2
        buf[225] = 1; // mat[0][0] = 1
        buf[225 + 3 * SK_FIELD_BYTES_LVL1] = 1; // mat[1][1] = 1

        let sk = SecretKeyData::from_bytes_lvl1(&buf).expect("decode SK");
        assert_eq!(sk.curve_a, Fp2::<Fp1Element>::zero(), "E0 curve");
        assert_eq!(sk.hint_pk, 0);
        assert_eq!(sk.mat_bacan_to_ba0_two[0][0], Uint::<8>::ONE);
        assert_eq!(sk.mat_bacan_to_ba0_two[1][1], Uint::<8>::ONE);
        // The reconstructed secret ideal (norm 5) must be a valid left O_0-ideal.
        let p16 = crate::params::lvl1::prime().resize::<16>();
        for r in 0..4 {
            let g = sk.secret_ideal.basis[r];
            for k in 0..4 {
                let mut e = [crypto_bigint::Int::<16>::from_i64(0); 4];
                e[k] = crypto_bigint::Int::<16>::from_i64(1);
                assert!(
                    sk.secret_ideal
                        .contains(&multiply_o0_basis::<16>(&e, &g, &p16)),
                    "secret ideal must be left-O_0-closed",
                );
            }
        }
        assert_eq!(
            sk.secret_ideal.reduced_norm_vartime(),
            Some(Uint::<16>::from_u64(5)),
            "secret ideal norm = 5",
        );
    }

    #[test]
    fn signature_decode_rejects_short_buffer() {
        let buf = [0u8; SIG_BYTES_LVL1 - 1];
        assert!(SignatureData::from_bytes_lvl1(&buf).is_err());
    }
}
