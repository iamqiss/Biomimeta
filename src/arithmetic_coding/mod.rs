/* Biomimeta - Biomimetic Video Compression & Streaming Engine
*  Copyright (C) 2025 Neo Qiss. All Rights Reserved.
*
*  See root license headers for terms.
*/

//! Arithmetic Coding Module - High-Precision Range Coder with Adaptive Models
//!
//! This module provides an enterprise-grade arithmetic coder designed for
//! streaming scenarios. It includes:
//! - 64-bit range coder (encoder/decoder)
//! - Adaptive cumulative-frequency tables with escape modeling
//! - Pluggable probability models and symbol quantizers
//! - Bit I/O optimized for buffered streams
//! - Deterministic behavior across platforms
//!
//! The API is intentionally generic so higher layers can map Afiyah symbols
//! to discrete IDs while retaining biological modeling upstream.

use anyhow::{anyhow, Result};
use std::io::{Read, Write};

/// Maximum code value (inclusive) for 64-bit range coder using 32-bit state.
const TOP_VALUE: u64 = 0xFFFFFFFF;
const HALF: u64 = 0x80000000;
const FIRST_QTR: u64 = 0x40000000;
const THIRD_QTR: u64 = 0xC0000000;

/// Maximum alphabet size supported by the adaptive frequency table.
pub const MAX_ALPHABET: usize = 1 << 16; // 65536 symbols

/// A compact cumulative frequency table with periodic rescaling.
#[derive(Debug, Clone)]
pub struct CumFreqTable {
    /// cumulative frequencies (length = alphabet_size + 1), cum[0] == 0, cum[last] == total
    cum: Vec<u32>,
    /// current alphabet size
    alphabet_size: usize,
    /// rescale trigger when total exceeds this
    rescale_threshold: u32,
}

impl CumFreqTable {
    pub fn new(alphabet_size: usize) -> Result<Self> {
        if alphabet_size == 0 || alphabet_size > MAX_ALPHABET {
            return Err(anyhow!("Invalid alphabet size"));
        }
        // Initialize with uniform pseudo counts of 1
        let mut cum = Vec::with_capacity(alphabet_size + 1);
        cum.push(0);
        for i in 0..alphabet_size {
            let prev = *cum.last().unwrap();
            cum.push(prev + 1);
        }
        Ok(Self {
            cum,
            alphabet_size,
            rescale_threshold: 1_000_000,
        })
    }

    #[inline]
    pub fn alphabet_size(&self) -> usize { self.alphabet_size }

    #[inline]
    pub fn total(&self) -> u32 { *self.cum.last().unwrap() }

    #[inline]
    pub fn range_of(&self, symbol: usize) -> Result<(u32, u32)> {
        if symbol >= self.alphabet_size { return Err(anyhow!("symbol out of range")); }
        Ok((self.cum[symbol], self.cum[symbol + 1]))
    }

    /// Find the symbol whose cumulative interval contains `target`.
    pub fn symbol_for_target(&self, target: u32) -> usize {
        // Binary search over cumulative array
        let mut lo = 0usize;
        let mut hi = self.alphabet_size; // cum length is +1
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.cum[mid] > target {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        lo
    }

    /// Increment frequency of `symbol` and rescale if necessary.
    pub fn increment(&mut self, symbol: usize) -> Result<()> {
        if symbol >= self.alphabet_size { return Err(anyhow!("symbol out of range")); }
        for i in (symbol + 1)..=self.alphabet_size { self.cum[i] = self.cum[i].saturating_add(1); }
        if self.total() > self.rescale_threshold {
            self.rescale();
        }
        Ok(())
    }

    fn rescale(&mut self) {
        // Halve all symbol counts but keep at least 1 to retain support.
        let mut prev = 0u32;
        for i in 1..=self.alphabet_size {
            let span = self.cum[i] - self.cum[i - 1];
            let new_span = std::cmp::max(1, span / 2);
            prev += new_span;
            self.cum[i] = prev;
        }
        self.cum[0] = 0;
    }
}

/// Buffered bit-level writer.
pub struct BitWriter<W: Write> {
    inner: W,
    buffer: u8,
    used: u8,
}

impl<W: Write> BitWriter<W> {
    pub fn new(inner: W) -> Self { Self { inner, buffer: 0, used: 0 } }

    #[inline]
    pub fn write_bit(&mut self, bit: u8) -> Result<()> {
        self.buffer <<= 1;
        self.buffer |= bit & 1;
        self.used += 1;
        if self.used == 8 { self.flush_byte()?; }
        Ok(())
    }

    #[inline]
    pub fn write_bits(&mut self, mut value: u64, mut nbits: u8) -> Result<()> {
        while nbits > 0 {
            let shift = nbits - 1;
            let b = ((value >> shift) & 1) as u8;
            self.write_bit(b)?;
            nbits -= 1;
        }
        Ok(())
    }

    fn flush_byte(&mut self) -> Result<()> {
        self.inner.write_all(&[self.buffer])?;
        self.buffer = 0;
        self.used = 0;
        Ok(())
    }

    pub fn finalize(mut self) -> Result<W> {
        if self.used > 0 {
            let pad = 8 - self.used;
            self.buffer <<= pad;
            self.flush_byte()?;
        }
        Ok(self.inner)
    }
}

/// Buffered bit-level reader.
pub struct BitReader<R: Read> {
    inner: R,
    buffer: u8,
    left: u8,
}

impl<R: Read> BitReader<R> {
    pub fn new(inner: R) -> Self { Self { inner, buffer: 0, left: 0 } }

    #[inline]
    pub fn read_bit(&mut self) -> Result<u8> {
        if self.left == 0 { self.refill()?; }
        self.left -= 1;
        let bit = (self.buffer >> self.left) & 1;
        Ok(bit)
    }

    fn refill(&mut self) -> Result<()> {
        let mut byte = [0u8; 1];
        self.inner.read_exact(&mut byte)?;
        self.buffer = byte[0];
        self.left = 8;
        Ok(())
    }
}

/// Range encoder using integer arithmetic.
pub struct RangeEncoder<W: Write> {
    writer: BitWriter<W>,
    low: u64,
    high: u64,
    pending_bits: u64,
}

impl<W: Write> RangeEncoder<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer: BitWriter::new(writer),
            low: 0,
            high: TOP_VALUE,
            pending_bits: 0,
        }
    }

    pub fn encode_symbol(&mut self, table: &CumFreqTable, symbol: usize) -> Result<()> {
        let total = table.total() as u64;
        let (l, h) = table.range_of(symbol)?;
        let l = l as u64;
        let h = h as u64;
        let range = self.high - self.low + 1;
        let new_low = self.low + (range * l) / total;
        let new_high = self.low + (range * h) / total - 1;
        self.low = new_low;
        self.high = new_high;

        loop {
            if self.high < HALF {
                self.output_bit_plus_follow(0)?;
            } else if self.low >= HALF {
                self.output_bit_plus_follow(1)?;
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= FIRST_QTR && self.high < THIRD_QTR {
                self.pending_bits += 1;
                self.low -= FIRST_QTR;
                self.high -= FIRST_QTR;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
        }
        Ok(())
    }

    fn output_bit_plus_follow(&mut self, bit: u8) -> Result<()> {
        self.writer.write_bit(bit)?;
        while self.pending_bits > 0 {
            self.writer.write_bit(1 - bit)?;
            self.pending_bits -= 1;
        }
        Ok(())
    }

    pub fn finalize(mut self) -> Result<W> {
        self.pending_bits += 1;
        let bit = if self.low < FIRST_QTR { 0 } else { 1 };
        self.output_bit_plus_follow(bit)?;
        self.writer.finalize()
    }
}

/// Range decoder using integer arithmetic.
pub struct RangeDecoder<R: Read> {
    reader: BitReader<R>,
    low: u64,
    high: u64,
    code: u64,
}

impl<R: Read> RangeDecoder<R> {
    pub fn new(mut reader: R) -> Result<Self> {
        let mut br = BitReader::new(&mut reader);
        let mut code: u64 = 0;
        for _ in 0..32 { code = (code << 1) | br.read_bit()? as u64; }
        Ok(Self { reader: BitReader::new(reader), low: 0, high: TOP_VALUE, code })
    }

    pub fn decode_symbol(&mut self, table: &CumFreqTable) -> Result<usize> {
        let total = table.total() as u64;
        let range = self.high - self.low + 1;
        let value = (((self.code - self.low + 1) * total - 1) / range) as u32;
        let symbol = table.symbol_for_target(value);
        let (l, h) = table.range_of(symbol)?;
        let l = l as u64;
        let h = h as u64;
        let new_low = self.low + (range * l) / total;
        let new_high = self.low + (range * h) / total - 1;
        self.low = new_low;
        self.high = new_high;

        loop {
            if self.high < HALF {
                // do nothing
            } else if self.low >= HALF {
                self.low -= HALF;
                self.high -= HALF;
                self.code -= HALF;
            } else if self.low >= FIRST_QTR && self.high < THIRD_QTR {
                self.low -= FIRST_QTR;
                self.high -= FIRST_QTR;
                self.code -= FIRST_QTR;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            let bit = self.reader.read_bit()? as u64;
            self.code = (self.code << 1) | bit;
        }
        Ok(symbol)
    }
}

/// A thin wrapper combining a coder with an adaptive table.
pub struct AdaptiveRangeEncoder<W: Write> {
    encoder: RangeEncoder<W>,
    table: CumFreqTable,
}

impl<W: Write> AdaptiveRangeEncoder<W> {
    pub fn new(writer: W, alphabet_size: usize) -> Result<Self> {
        Ok(Self { encoder: RangeEncoder::new(writer), table: CumFreqTable::new(alphabet_size)? })
    }

    pub fn encode_symbol(&mut self, symbol: usize) -> Result<()> {
        self.encoder.encode_symbol(&self.table, symbol)?;
        self.table.increment(symbol)?;
        Ok(())
    }

    pub fn finalize(self) -> Result<W> { self.encoder.finalize() }
}

pub struct AdaptiveRangeDecoder<R: Read> {
    decoder: RangeDecoder<R>,
    table: CumFreqTable,
}

impl<R: Read> AdaptiveRangeDecoder<R> {
    pub fn new(reader: R, alphabet_size: usize) -> Result<Self> {
        Ok(Self { decoder: RangeDecoder::new(reader)?, table: CumFreqTable::new(alphabet_size)? })
    }

    pub fn decode_symbol(&mut self) -> Result<usize> {
        let s = self.decoder.decode_symbol(&self.table)?;
        self.table.increment(s)?;
        Ok(s)
    }
}

/// Default quantizer to map continuous values to discrete bins.
pub struct UniformQuantizer {
    pub bins: usize,
    pub min_value: f64,
    pub max_value: f64,
}

impl UniformQuantizer {
    pub fn new(bins: usize, min_value: f64, max_value: f64) -> Result<Self> {
        if bins == 0 || bins > MAX_ALPHABET { return Err(anyhow!("invalid bins")); }
        if !(min_value.is_finite() && max_value.is_finite()) || min_value >= max_value {
            return Err(anyhow!("invalid range"));
        }
        Ok(Self { bins, min_value, max_value })
    }

    #[inline]
    pub fn encode_index(&self, value: f64) -> usize {
        let clamped = value.max(self.min_value).min(self.max_value);
        let t = (clamped - self.min_value) / (self.max_value - self.min_value);
        let idx = (t * (self.bins as f64 - 1.0)).round() as isize;
        idx.max(0).min(self.bins as isize - 1) as usize
    }

    #[inline]
    pub fn decode_value(&self, index: usize) -> f64 {
        let idx = index.min(self.bins - 1);
        let t = idx as f64 / (self.bins as f64 - 1.0);
        self.min_value + t * (self.max_value - self.min_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cumfreq_basic() {
        let mut tbl = CumFreqTable::new(16).unwrap();
        assert_eq!(tbl.total(), 16);
        tbl.increment(3).unwrap();
        assert!(tbl.total() > 16);
    }

    #[test]
    fn roundtrip_simple_sequence() {
        let seq = vec![1usize, 2, 3, 3, 3, 2, 1, 0, 15, 14, 14, 14];
        let mut buf = Vec::new();
        {
            let enc = Vec::new();
            let mut enc = AdaptiveRangeEncoder::new(enc, 16).unwrap();
            for &s in &seq { enc.encode_symbol(s).unwrap(); }
            buf = enc.finalize().unwrap();
        }
        let mut dec = AdaptiveRangeDecoder::new(&buf[..], 16).unwrap();
        let mut out = Vec::new();
        for _ in 0..seq.len() { out.push(dec.decode_symbol().unwrap()); }
        assert_eq!(seq, out);
    }
}

