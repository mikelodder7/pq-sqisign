// SPDX-License-Identifier: MIT OR Apache-2.0
//! Wire-format encoding primitives.
//!
//! SQIsign's compact public-key / secret-key / signature layouts pack
//! variable-bit-length fields back-to-back into a small byte budget
//! (Level-1: 65 / 353 / 148; Level-5: 129 / 701 / 292). The protocol
//! orchestrator builds an encoder once and pushes each field with the
//! exact bit width the spec dictates; the verifier mirrors with a reader.
//!
//! Two primitives:
//!
//! - [`BitWriter`] — push `(value, bits)` pairs into a caller-owned buffer.
//! - [`BitReader`] — pull `bits` at a time, returning the accumulated value.
//!
//! Bit order: **little-endian within each byte**, i.e. the first bit
//! written goes into the least-significant bit of `buf[0]`, and successive
//! bits walk up to bit 7 then continue into `buf[1]` at bit 0. This matches
//! the reference's `bit_pack` / `bit_unpack` macros in
//! `src/common/generic/include/tutil.h`.

use crate::error::{Error, Result};

/// Encoder over a caller-owned byte buffer.
///
/// The buffer is *not* zeroised on construction — the caller is expected to
/// zero it once before populating, then `push_bits` accumulates by OR-ing
/// into the existing bytes (bits already past `self.bit_pos` are assumed
/// `0`).
#[derive(Debug)]
pub struct BitWriter<'a> {
    buf: &'a mut [u8],
    bit_pos: usize,
}

impl<'a> BitWriter<'a> {
    /// Wrap a zeroed byte buffer.
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, bit_pos: 0 }
    }

    /// Current write position in bits from the start of the buffer.
    pub fn bit_pos(&self) -> usize {
        self.bit_pos
    }

    /// Push the low `num_bits` of `value` into the buffer.
    ///
    /// Returns `Err(BufferTooSmall)` if there aren't enough bits left.
    /// `num_bits` must be `≤ 64`.
    pub fn push_bits(&mut self, value: u64, num_bits: u32) -> Result<()> {
        debug_assert!(num_bits <= 64);
        let needed = self.bit_pos + num_bits as usize;
        if needed > self.buf.len() * 8 {
            return Err(Error::BufferTooSmall {
                required: needed.div_ceil(8),
                provided: self.buf.len(),
            });
        }
        let mask: u64 = if num_bits == 64 {
            u64::MAX
        } else {
            (1u64 << num_bits) - 1
        };
        let mut v = value & mask;
        let mut bits_left = num_bits as usize;
        while bits_left > 0 {
            let byte_idx = self.bit_pos / 8;
            let bit_off = self.bit_pos % 8;
            let take = core::cmp::min(8 - bit_off, bits_left);
            let chunk_mask = if take == 8 { 0xffu8 } else { (1u8 << take) - 1 };
            // masked to `take` ≤ 8 bits, so the value fits in a `u8`.
            let chunk = u8::try_from(v & u64::from(chunk_mask)).expect("masked to ≤ 8 bits");
            self.buf[byte_idx] |= chunk << bit_off;
            v >>= take;
            self.bit_pos += take;
            bits_left -= take;
        }
        Ok(())
    }
}

/// Decoder over a borrowed byte buffer.
#[derive(Debug)]
pub struct BitReader<'a> {
    buf: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    /// Wrap a buffer for reading.
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, bit_pos: 0 }
    }

    /// Current read position in bits.
    pub fn bit_pos(&self) -> usize {
        self.bit_pos
    }

    /// Pull the next `num_bits` bits as a `u64`. Bits beyond `num_bits`
    /// are zero. Returns `Err(BufferTooSmall)` if not enough remaining.
    /// `num_bits` must be `≤ 64`.
    pub fn read_bits(&mut self, num_bits: u32) -> Result<u64> {
        debug_assert!(num_bits <= 64);
        let needed = self.bit_pos + num_bits as usize;
        if needed > self.buf.len() * 8 {
            return Err(Error::BufferTooSmall {
                required: needed.div_ceil(8),
                provided: self.buf.len(),
            });
        }
        let mut out: u64 = 0;
        let mut bits_done: usize = 0;
        while bits_done < num_bits as usize {
            let byte_idx = self.bit_pos / 8;
            let bit_off = self.bit_pos % 8;
            let take = core::cmp::min(8 - bit_off, num_bits as usize - bits_done);
            let chunk_mask = if take == 8 { 0xffu8 } else { (1u8 << take) - 1 };
            let chunk = (self.buf[byte_idx] >> bit_off) & chunk_mask;
            out |= u64::from(chunk) << bits_done;
            self.bit_pos += take;
            bits_done += take;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_byte_round_trip() {
        let mut buf = [0u8; 1];
        let mut w = BitWriter::new(&mut buf);
        w.push_bits(0b101, 3).expect("fits");
        w.push_bits(0b1110, 4).expect("fits");
        let mut r = BitReader::new(&buf);
        assert_eq!(r.read_bits(3).expect("fits"), 0b101);
        assert_eq!(r.read_bits(4).expect("fits"), 0b1110);
    }

    #[test]
    fn cross_byte_boundary() {
        let mut buf = [0u8; 2];
        let mut w = BitWriter::new(&mut buf);
        // 12 bits — crosses the byte boundary.
        w.push_bits(0xabc, 12).expect("fits");
        let mut r = BitReader::new(&buf);
        assert_eq!(r.read_bits(12).expect("fits"), 0xabc);
    }

    #[test]
    fn full_byte_alignment() {
        let mut buf = [0u8; 4];
        let mut w = BitWriter::new(&mut buf);
        w.push_bits(0xab, 8).expect("fits");
        w.push_bits(0xcd, 8).expect("fits");
        w.push_bits(0xef, 8).expect("fits");
        w.push_bits(0x12, 8).expect("fits");
        assert_eq!(buf, [0xab, 0xcd, 0xef, 0x12]);
    }

    #[test]
    fn full_64_bits() {
        let mut buf = [0u8; 8];
        let mut w = BitWriter::new(&mut buf);
        w.push_bits(0x0123_4567_89ab_cdef, 64).expect("fits");
        let mut r = BitReader::new(&buf);
        assert_eq!(r.read_bits(64).expect("fits"), 0x0123_4567_89ab_cdef);
    }

    #[test]
    fn buffer_too_small_on_write() {
        let mut buf = [0u8; 1];
        let mut w = BitWriter::new(&mut buf);
        w.push_bits(0, 8).expect("fits");
        let r = w.push_bits(0, 1);
        assert!(matches!(r, Err(Error::BufferTooSmall { .. })));
    }

    #[test]
    fn buffer_too_small_on_read() {
        let buf = [0u8; 1];
        let mut r = BitReader::new(&buf);
        r.read_bits(8).expect("fits");
        let r2 = r.read_bits(1);
        assert!(matches!(r2, Err(Error::BufferTooSmall { .. })));
    }

    #[test]
    fn many_small_fields_pack_dense() {
        let mut buf = [0u8; 8];
        let mut w = BitWriter::new(&mut buf);
        // Pack 16 nibbles back-to-back.
        for i in 0..16u8 {
            w.push_bits(u64::from(i), 4).expect("fits");
        }
        assert_eq!(w.bit_pos(), 64);
        let mut r = BitReader::new(&buf);
        for i in 0..16u8 {
            assert_eq!(r.read_bits(4).expect("fits"), u64::from(i));
        }
    }

    #[test]
    fn mixed_widths() {
        let mut buf = [0u8; 16];
        let widths: &[u32] = &[1, 3, 5, 7, 11, 13, 17, 19, 23];
        let values: &[u64] = &[1, 5, 17, 99, 1023, 5000, 100_000, 500_000, 8_000_000];
        let mut w = BitWriter::new(&mut buf);
        for (&v, &b) in values.iter().zip(widths.iter()) {
            w.push_bits(v, b).expect("fits");
        }
        let mut r = BitReader::new(&buf);
        for (&expected, &b) in values.iter().zip(widths.iter()) {
            assert_eq!(r.read_bits(b).expect("fits"), expected);
        }
    }

    #[test]
    fn zero_bit_push_is_noop() {
        let mut buf = [0u8; 1];
        let mut w = BitWriter::new(&mut buf);
        w.push_bits(0xff, 0).expect("fits");
        assert_eq!(w.bit_pos(), 0);
        assert_eq!(buf, [0]);
    }
}
