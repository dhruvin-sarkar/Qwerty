//! WAV reading and writing for recordings and replay (§1.18).
//!
//! Hand-rolled 32-bit IEEE-float WAVE_FORMAT_EXTENSIBLE, little-endian (no new crate). The writer
//! emits one fixed 80-byte header:
//!
//! ```text
//!  0 "RIFF" <u32 size = file length − 8> "WAVE"
//! 12 "fmt " <u32 40> tag 0xFFFE, channels, rate, byte_rate, block_align, bits 32,
//!                    cbSize 22, valid_bits 32, channel_mask, SubFormat IEEE_FLOAT
//! 60 "fact" <u32 4>  frames
//! 72 "data" <u32 size> samples…
//! ```

use crate::audio::AudioFormat;
use crate::units::{ChannelCount, IoFailure, SampleRate};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Largest data chunk a writer accepts: 2 GiB of sample data.
pub const WAV_DATA_LIMIT: u64 = 2 * 1024 * 1024 * 1024;

/// WAVE_FORMAT_IEEE_FLOAT.
const TAG_FLOAT: u16 = 3;
/// WAVE_FORMAT_EXTENSIBLE.
const TAG_EXTENSIBLE: u16 = 0xFFFE;
/// The only sample width read or written.
const BITS: u16 = 32;
/// Bytes per sample.
const SAMPLE_BYTES: u64 = 4;
/// KSDATAFORMAT_SUBTYPE_IEEE_FLOAT as stored on disk.
const SUBFORMAT_FLOAT: [u8; 16] = [
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];
/// Size of the extensible `fmt ` chunk body.
const FMT_EXTENSIBLE_SIZE: u32 = 40;
/// cbSize of the extensible `fmt ` chunk.
const EXTENSIBLE_CB_SIZE: u16 = 22;
/// The smallest `fmt ` body any reader can parse (the PCMWAVEFORMAT part).
const FMT_MIN_SIZE: u32 = 16;
/// Byte offsets of the size fields the writer patches.
const RIFF_SIZE_AT: u64 = 4;
const FACT_FRAMES_AT: u64 = 68;
const DATA_SIZE_AT: u64 = 76;
/// The writer's header length; sample data starts here.
const HEADER_BYTES: u64 = 80;
/// Bytes of the RIFF size that are not sample data: "WAVE" + fmt + fact + data chunk headers.
const RIFF_OVERHEAD: u64 = HEADER_BYTES - 8;
/// Staging buffer for sample (de)serialization, in bytes (a multiple of 4).
const IO_CHUNK_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WavSpec {
    pub format: AudioFormat,
    pub channel_mask: u32,
}

impl WavSpec {
    fn block_align(self) -> u64 {
        u64::from(self.format.channels.get()) * SAMPLE_BYTES
    }
}

/// Streams interleaved f32 frames into a new WAV file.
pub struct WavWriter {
    out: BufWriter<File>,
    spec: WavSpec,
    frames: u64,
    limit: u64,
}

impl WavWriter {
    /// Creates `path` with the 2 GiB data limit; fails if the file exists.
    pub fn create(path: &Path, spec: WavSpec) -> Result<Self, WavError> {
        Self::create_with_limit(path, spec, WAV_DATA_LIMIT)
    }

    /// Creates `path` with a data limit of `limit` bytes; fails if the file exists.
    pub fn create_with_limit(path: &Path, spec: WavSpec, limit: u64) -> Result<Self, WavError> {
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        let mut out = BufWriter::new(file);
        out.write_all(&header_bytes(spec, 0))?;
        Ok(WavWriter {
            out,
            spec,
            frames: 0,
            limit,
        })
    }

    /// Frames written so far.
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// Header plus sample bytes written so far.
    pub fn bytes(&self) -> u64 {
        HEADER_BYTES + self.data_bytes()
    }

    fn data_bytes(&self) -> u64 {
        self.frames * self.spec.block_align()
    }

    /// Appends whole interleaved frames. All or nothing: `SizeLimitReached` writes nothing.
    // SPEC: §1.18.1 names no error for a buffer that is not whole frames; it is refused with
    // `BadFormat` before anything is written, since a partial frame would shift every later one.
    pub fn write_interleaved(&mut self, samples: &[f32]) -> Result<(), WavError> {
        let channels = usize::from(self.spec.format.channels.get());
        if !samples.len().is_multiple_of(channels) {
            return Err(WavError::BadFormat {
                reason: format!(
                    "{} samples are not whole {channels}-channel frames",
                    samples.len()
                ),
            });
        }
        let added = samples.len() as u64 * SAMPLE_BYTES;
        if self.data_bytes() + added > self.limit {
            return Err(WavError::SizeLimitReached { limit: self.limit });
        }
        let mut staging = [0u8; IO_CHUNK_BYTES];
        for chunk in samples.chunks(IO_CHUNK_BYTES / SAMPLE_BYTES as usize) {
            for (bytes, s) in staging.chunks_exact_mut(SAMPLE_BYTES as usize).zip(chunk) {
                bytes.copy_from_slice(&s.to_le_bytes());
            }
            self.out
                .write_all(&staging[..chunk.len() * SAMPLE_BYTES as usize])?;
        }
        self.frames += (samples.len() / channels) as u64;
        Ok(())
    }

    /// Patches the RIFF, fact and data sizes for the frames written so far, then flushes and
    /// syncs the file data, so a crash afterwards leaves a readable file.
    pub fn flush_header(&mut self) -> Result<(), WavError> {
        let data = self.data_bytes();
        let frames = self.frames;
        self.out.seek(SeekFrom::Start(RIFF_SIZE_AT))?;
        self.out.write_all(&size_u32(RIFF_OVERHEAD + data)?)?;
        self.out.seek(SeekFrom::Start(FACT_FRAMES_AT))?;
        self.out.write_all(&size_u32(frames)?)?;
        self.out.seek(SeekFrom::Start(DATA_SIZE_AT))?;
        self.out.write_all(&size_u32(data)?)?;
        self.out.seek(SeekFrom::End(0))?;
        self.out.flush()?;
        self.out.get_ref().sync_data()?;
        Ok(())
    }

    /// Patches the header and closes the file; returns the frame count.
    pub fn finalize(mut self) -> Result<u64, WavError> {
        self.flush_header()?;
        Ok(self.frames)
    }
}

/// Reads the frames of a 32-bit float WAV file.
pub struct WavReader {
    input: BufReader<File>,
    spec: WavSpec,
    frames: u64,
    cursor: u64,
}

impl WavReader {
    /// Parses the header up to the data chunk. Accepts tag 3 (32-bit IEEE float) and tag 0xFFFE
    /// with the float SubFormat and 32 bits; skips unknown chunks (honouring the pad byte).
    pub fn open(path: &Path) -> Result<Self, WavError> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut input = BufReader::new(file);
        let layout = parse_layout(&mut input, file_len)?;
        let block_align = layout.spec.block_align();
        if layout.data_size > layout.remaining {
            return Err(WavError::SizeMismatch {
                header: layout.data_size,
                actual: layout.remaining,
            });
        }
        if !layout.data_size.is_multiple_of(block_align) {
            return Err(WavError::BadFormat {
                reason: format!(
                    "data size {} is not a whole number of {block_align}-byte frames",
                    layout.data_size
                ),
            });
        }
        Ok(WavReader {
            input,
            spec: layout.spec,
            frames: layout.data_size / block_align,
            cursor: 0,
        })
    }

    pub fn spec(&self) -> WavSpec {
        self.spec
    }

    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// Fills `out` with up to `out.len() / channels` interleaved frames; returns frames read, 0
    /// at EOF.
    // SPEC: §1.18.1 names no error for an output buffer that is not whole frames; it is refused
    // with `BadFormat` (a short buffer would otherwise read 0 frames and look like EOF).
    pub fn read_frames(&mut self, out: &mut [f32]) -> Result<usize, WavError> {
        let channels = usize::from(self.spec.format.channels.get());
        if !out.len().is_multiple_of(channels) {
            return Err(WavError::BadFormat {
                reason: format!(
                    "output buffer of {} samples is not whole {channels}-channel frames",
                    out.len()
                ),
            });
        }
        let wanted = (out.len() / channels) as u64;
        // ≤ wanted, which came from a usize.
        let frames = wanted.min(self.frames - self.cursor) as usize;
        let samples = &mut out[..frames * channels];
        let mut staging = [0u8; IO_CHUNK_BYTES];
        for chunk in samples.chunks_mut(IO_CHUNK_BYTES / SAMPLE_BYTES as usize) {
            let bytes = &mut staging[..chunk.len() * SAMPLE_BYTES as usize];
            self.input.read_exact(bytes)?;
            for (s, b) in chunk
                .iter_mut()
                .zip(bytes.chunks_exact(SAMPLE_BYTES as usize))
            {
                *s = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        self.cursor += frames as u64;
        Ok(frames)
    }
}

/// Rewrites the RIFF, fact and data sizes from the file length, keeping whole frames only (a
/// trailing partial frame is cut off). Explicit, never implicit: readers never call it. Returns
/// the frame count.
// SPEC: "whole frames only" is read as also truncating a trailing partial frame, so that the
// repaired file is exactly header + data and its RIFF size matches its length.
pub fn repair(path: &Path) -> Result<u64, WavError> {
    let file = OpenOptions::new().read(true).write(true).open(path)?;
    let file_len = file.metadata()?.len();
    let mut input = BufReader::new(file);
    let layout = parse_layout(&mut input, file_len)?;
    let block_align = layout.spec.block_align();
    let frames = layout.remaining / block_align;
    let data = frames * block_align;
    let mut file = input.into_inner();
    file.set_len(layout.data_offset + data)?;
    file.seek(SeekFrom::Start(RIFF_SIZE_AT))?;
    file.write_all(&size_u32(layout.data_offset + data - 8)?)?;
    if let Some(at) = layout.fact_frames_at {
        file.seek(SeekFrom::Start(at))?;
        file.write_all(&size_u32(frames)?)?;
    }
    file.seek(SeekFrom::Start(layout.data_offset - 4))?;
    file.write_all(&size_u32(data)?)?;
    file.sync_data()?;
    Ok(frames)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WavError {
    Io { error: IoFailure },
    NotRiffWave,
    MissingChunk { id: String },
    UnsupportedFormat { tag: u16, bits: u16 },
    SizeMismatch { header: u64, actual: u64 },
    SizeLimitReached { limit: u64 },
    BadFormat { reason: String },
}

impl From<std::io::Error> for WavError {
    fn from(e: std::io::Error) -> Self {
        WavError::Io {
            error: IoFailure::from(&e),
        }
    }
}

impl fmt::Display for WavError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WavError::Io { error } => {
                write!(f, "WAV I/O error ({}): {}", error.kind, error.message)
            }
            WavError::NotRiffWave => write!(f, "not a RIFF/WAVE file"),
            WavError::MissingChunk { id } => write!(f, "WAV file has no '{id}' chunk"),
            WavError::UnsupportedFormat { tag, bits } => {
                write!(f, "unsupported WAV format tag {tag:#06x} with {bits} bits")
            }
            WavError::SizeMismatch { header, actual } => write!(
                f,
                "WAV header claims {header} data bytes but only {actual} are present"
            ),
            WavError::SizeLimitReached { limit } => {
                write!(f, "WAV data limit of {limit} bytes reached")
            }
            WavError::BadFormat { reason } => write!(f, "malformed WAV: {reason}"),
        }
    }
}

impl std::error::Error for WavError {}

/// What `parse_layout` learned from the header.
struct Layout {
    spec: WavSpec,
    data_offset: u64,
    data_size: u64,
    /// Bytes from the data offset to the end of the file.
    remaining: u64,
    /// Byte offset of the fact chunk's frame count, if the file has one.
    fact_frames_at: Option<u64>,
}

/// Reads the RIFF header and the chunks up to `data`, leaving `input` at the first sample.
fn parse_layout(input: &mut BufReader<File>, file_len: u64) -> Result<Layout, WavError> {
    let mut riff = [0u8; 12];
    if file_len < 12 {
        return Err(WavError::NotRiffWave);
    }
    input.read_exact(&mut riff)?;
    if &riff[0..4] != b"RIFF" || &riff[8..12] != b"WAVE" {
        return Err(WavError::NotRiffWave);
    }
    let mut pos: u64 = 12;
    let mut spec: Option<WavSpec> = None;
    let mut fact_frames_at = None;
    loop {
        if file_len - pos < 8 {
            let missing = if spec.is_none() { "fmt " } else { "data" };
            return Err(WavError::MissingChunk {
                id: missing.to_string(),
            });
        }
        let mut head = [0u8; 8];
        input.read_exact(&mut head)?;
        pos += 8;
        let size = u64::from(u32::from_le_bytes([head[4], head[5], head[6], head[7]]));
        let pad = size & 1;
        match &head[0..4] {
            b"fmt " => {
                let body = read_body(input, size, file_len - pos)?;
                spec = Some(parse_fmt(&body)?);
                pos += size;
                skip(input, &mut pos, pad, file_len)?;
            }
            b"fact" => {
                if size >= 4 {
                    fact_frames_at = Some(pos);
                }
                skip(input, &mut pos, size + pad, file_len)?;
            }
            b"data" => {
                let Some(spec) = spec else {
                    return Err(WavError::MissingChunk {
                        id: "fmt ".to_string(),
                    });
                };
                return Ok(Layout {
                    spec,
                    data_offset: pos,
                    data_size: size,
                    remaining: file_len - pos,
                    fact_frames_at,
                });
            }
            _ => skip(input, &mut pos, size + pad, file_len)?,
        }
    }
}

/// Skips `n` bytes, or to the end of the file if fewer remain (the next chunk read then reports
/// the missing chunk).
fn skip(input: &mut BufReader<File>, pos: &mut u64, n: u64, file_len: u64) -> Result<(), WavError> {
    let n = n.min(file_len - *pos);
    input.seek_relative(i64::try_from(n).map_err(|_| bad("chunk size out of range"))?)?;
    *pos += n;
    Ok(())
}

/// Reads a `size`-byte chunk body, refusing one that runs past the end of the file.
fn read_body(input: &mut BufReader<File>, size: u64, left: u64) -> Result<Vec<u8>, WavError> {
    if size > left {
        return Err(WavError::SizeMismatch {
            header: size,
            actual: left,
        });
    }
    let mut body = vec![0u8; usize::try_from(size).map_err(|_| bad("fmt chunk too large"))?];
    input.read_exact(&mut body)?;
    Ok(body)
}

fn parse_fmt(body: &[u8]) -> Result<WavSpec, WavError> {
    if (body.len() as u64) < u64::from(FMT_MIN_SIZE) {
        return Err(bad("fmt chunk shorter than 16 bytes"));
    }
    let u16_at = |i: usize| u16::from_le_bytes([body[i], body[i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes([body[i], body[i + 1], body[i + 2], body[i + 3]]);
    let tag = u16_at(0);
    let channels = u16_at(2);
    let rate = u32_at(4);
    let byte_rate = u32_at(8);
    let block_align = u16_at(12);
    let bits = u16_at(14);
    let channel_mask = match tag {
        TAG_FLOAT if bits == BITS => 0,
        TAG_EXTENSIBLE if bits == BITS => {
            if (body.len() as u64) < u64::from(FMT_EXTENSIBLE_SIZE)
                || u16_at(16) < EXTENSIBLE_CB_SIZE
            {
                return Err(bad("extensible fmt chunk without its 22-byte extension"));
            }
            if body[24..40] != SUBFORMAT_FLOAT {
                return Err(WavError::UnsupportedFormat { tag, bits });
            }
            u32_at(20)
        }
        _ => return Err(WavError::UnsupportedFormat { tag, bits }),
    };
    let channels = ChannelCount::new(channels).map_err(|e| bad(&e.to_string()))?;
    let rate = SampleRate::new(rate).map_err(|e| bad(&e.to_string()))?;
    let spec = WavSpec {
        format: AudioFormat { rate, channels },
        channel_mask,
    };
    if u64::from(block_align) != spec.block_align() {
        return Err(bad(&format!(
            "block_align {block_align} is not {}",
            spec.block_align()
        )));
    }
    if u64::from(byte_rate) != u64::from(rate.hz()) * spec.block_align() {
        return Err(bad(&format!(
            "byte_rate {byte_rate} does not match the format"
        )));
    }
    Ok(spec)
}

/// The writer's 80-byte header for `frames` frames.
fn header_bytes(spec: WavSpec, frames: u32) -> [u8; HEADER_BYTES as usize] {
    let channels = spec.format.channels.get();
    let rate = spec.format.rate.hz();
    let block_align = channels * SAMPLE_BYTES as u16;
    let data = frames * u32::from(block_align);
    let mut h = [0u8; HEADER_BYTES as usize];
    let mut at = 0;
    let mut put = |bytes: &[u8]| {
        h[at..at + bytes.len()].copy_from_slice(bytes);
        at += bytes.len();
    };
    put(b"RIFF");
    put(&(RIFF_OVERHEAD as u32 + data).to_le_bytes());
    put(b"WAVE");
    put(b"fmt ");
    put(&FMT_EXTENSIBLE_SIZE.to_le_bytes());
    put(&TAG_EXTENSIBLE.to_le_bytes());
    put(&channels.to_le_bytes());
    put(&rate.to_le_bytes());
    put(&(rate * u32::from(block_align)).to_le_bytes());
    put(&block_align.to_le_bytes());
    put(&BITS.to_le_bytes());
    put(&EXTENSIBLE_CB_SIZE.to_le_bytes());
    put(&BITS.to_le_bytes());
    put(&spec.channel_mask.to_le_bytes());
    put(&SUBFORMAT_FLOAT);
    put(b"fact");
    put(&4u32.to_le_bytes());
    put(&frames.to_le_bytes());
    put(b"data");
    put(&data.to_le_bytes());
    h
}

/// A size field's bytes; sizes past u32 cannot be represented in RIFF.
fn size_u32(v: u64) -> Result<[u8; 4], WavError> {
    u32::try_from(v)
        .map(u32::to_le_bytes)
        .map_err(|_| bad(&format!("size {v} does not fit a RIFF size field")))
}

fn bad(reason: &str) -> WavError {
    WavError::BadFormat {
        reason: reason.to_string(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A fresh directory under `std::env::temp_dir()`, deleted on drop (§0.13).
    pub(crate) struct TestDir(PathBuf);

    impl TestDir {
        pub(crate) fn new(name: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("qwerty-test-{}-{name}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("create test dir");
            TestDir(dir)
        }

        pub(crate) fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn spec(rate: u32, channels: u16, channel_mask: u32) -> WavSpec {
        WavSpec {
            format: AudioFormat {
                rate: SampleRate::new(rate).expect("rate"),
                channels: ChannelCount::new(channels).expect("channels"),
            },
            channel_mask,
        }
    }

    /// Deterministic samples covering the awkward bit patterns: ±0, subnormals, extremes.
    fn samples(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = crate::rng::Rng64::new(seed);
        let specials = [
            0.0f32,
            -0.0,
            f32::MIN_POSITIVE / 2.0,
            -f32::MIN_POSITIVE,
            f32::MAX,
            f32::MIN,
            1.0,
            -1.0,
        ];
        (0..n)
            .map(|i| {
                specials
                    .get(i)
                    .copied()
                    .unwrap_or(rng.range(-1.5, 1.5) as f32)
            })
            .collect()
    }

    fn read_all(path: &Path) -> (WavSpec, Vec<f32>) {
        let mut r = WavReader::open(path).expect("open");
        let ch = usize::from(r.spec().format.channels.get());
        let mut out = vec![0f32; r.frames() as usize * ch];
        let mut got = 0;
        // Odd read sizes exercise the cursor.
        while got < out.len() {
            let end = (got + 37 * ch).min(out.len());
            let n = r.read_frames(&mut out[got..end]).expect("read");
            assert!(n > 0, "early EOF");
            got += n * ch;
        }
        assert_eq!(r.read_frames(&mut [0f32; 8 * 8]).expect("eof"), 0);
        (r.spec(), out)
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|s| s.to_bits()).collect()
    }

    #[test]
    fn w1_round_trip_is_bit_exact() {
        let dir = TestDir::new("w1");
        for channels in [1u16, 2, 8] {
            for rate in [16_000u32, 48_000, 192_000] {
                let path = dir.path().join(format!("{channels}-{rate}.wav"));
                let s = spec(rate, channels, 0x3);
                let data = samples(1000 * usize::from(channels), u64::from(rate) + 7);
                let mut w = WavWriter::create(&path, s).expect("create");
                w.write_interleaved(&data[..10 * usize::from(channels)])
                    .expect("write a");
                w.write_interleaved(&data[10 * usize::from(channels)..])
                    .expect("write b");
                assert_eq!(w.finalize().expect("finalize"), 1000);
                let (got_spec, got) = read_all(&path);
                assert_eq!(got_spec, s);
                assert_eq!(bits(&got), bits(&data), "{channels} ch at {rate} Hz");
            }
        }
        // Creating over an existing file fails.
        let existing = dir.path().join("1-16000.wav");
        assert!(matches!(
            WavWriter::create(&existing, spec(16_000, 1, 0)),
            Err(WavError::Io { .. })
        ));
        golden_header(&dir);
    }

    /// W1's golden header: 3 stereo frames at 48 kHz, hand-assembled from the §1.18.1 layout.
    fn golden_header(dir: &TestDir) {
        let path = dir.path().join("golden.wav");
        let mut w = WavWriter::create(&path, spec(48_000, 2, 0x3)).expect("create");
        w.write_interleaved(&[0.5, -0.5, 1.0, -1.0, 0.0, 0.25])
            .expect("write");
        assert_eq!(w.finalize().expect("finalize"), 3);
        let bytes = std::fs::read(&path).expect("read");
        #[rustfmt::skip]
        let golden: [u8; 80] = [
            b'R', b'I', b'F', b'F', 96, 0, 0, 0, b'W', b'A', b'V', b'E',
            b'f', b'm', b't', b' ', 40, 0, 0, 0,
            0xFE, 0xFF,             // tag
            2, 0,                   // channels
            0x80, 0xBB, 0, 0,       // 48 000 Hz
            0x00, 0xDC, 0x05, 0,    // byte rate 384 000 = 0x5DC00
            8, 0,                   // block align
            32, 0,                  // bits
            22, 0,                  // cbSize
            32, 0,                  // valid bits
            3, 0, 0, 0,             // channel mask
            0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00,
            0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
            b'f', b'a', b'c', b't', 4, 0, 0, 0, 3, 0, 0, 0,
            b'd', b'a', b't', b'a', 24, 0, 0, 0,
        ];
        assert_eq!(&bytes[..80], &golden[..]);
        assert_eq!(bytes.len(), 104);
        assert_eq!(&bytes[80..84], &0.5f32.to_le_bytes());
    }

    /// A minimal non-extensible file: `fmt ` of 16 (or 18) bytes, an odd unknown chunk, data.
    fn plain_file(tag: u16, bits_per: u16, channels: u16, payload: &[u8]) -> Vec<u8> {
        let rate = 44_100u32;
        let align = channels * bits_per / 8;
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&tag.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&(rate * u32::from(align)).to_le_bytes());
        fmt.extend_from_slice(&align.to_le_bytes());
        fmt.extend_from_slice(&bits_per.to_le_bytes());
        let mut body = Vec::new();
        body.extend_from_slice(b"WAVE");
        body.extend_from_slice(b"fmt ");
        body.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        body.extend_from_slice(&fmt);
        body.extend_from_slice(b"LIST");
        body.extend_from_slice(&3u32.to_le_bytes());
        body.extend_from_slice(&[1, 2, 3, 0]); // 3 bytes + pad
        body.extend_from_slice(b"data");
        body.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        body.extend_from_slice(payload);
        let mut file = b"RIFF".to_vec();
        file.extend_from_slice(&(body.len() as u32).to_le_bytes());
        file.extend_from_slice(&body);
        file
    }

    #[test]
    fn w2_tag3_reads_and_pcm16_is_unsupported() {
        let dir = TestDir::new("w2");
        let frames = [0.25f32, -0.75, 1.0, -0.0];
        let payload: Vec<u8> = frames.iter().flat_map(|s| s.to_le_bytes()).collect();
        let tag3 = dir.path().join("tag3.wav");
        std::fs::write(&tag3, plain_file(3, 32, 2, &payload)).expect("write tag3");
        let (s, got) = read_all(&tag3);
        assert_eq!(s, spec(44_100, 2, 0));
        assert_eq!(bits(&got), bits(&frames));

        let pcm = dir.path().join("pcm16.wav");
        std::fs::write(&pcm, plain_file(1, 16, 1, &[0, 0, 1, 0])).expect("write pcm");
        assert_eq!(
            WavReader::open(&pcm).err(),
            Some(WavError::UnsupportedFormat { tag: 1, bits: 16 })
        );

        let junk = dir.path().join("junk.wav");
        std::fs::write(&junk, b"RIFX\0\0\0\0WAVEfmt ").expect("write junk");
        assert_eq!(WavReader::open(&junk).err(), Some(WavError::NotRiffWave));
    }

    #[test]
    fn w3_truncated_file_mismatches_then_repairs() {
        // SPEC: "a truncated header" is read as a header whose sizes exceed the bytes present
        // (the file was cut short, mid-frame), which is what `repair` exists for.
        let dir = TestDir::new("w3");
        let path = dir.path().join("cut.wav");
        let data = samples(2 * 500, 3);
        let mut w = WavWriter::create(&path, spec(48_000, 2, 0x3)).expect("create");
        w.write_interleaved(&data).expect("write");
        w.finalize().expect("finalize");
        let full = std::fs::metadata(&path).expect("meta").len();
        // Drop 1.5 frames: 12 bytes.
        let f = OpenOptions::new().write(true).open(&path).expect("open rw");
        f.set_len(full - 12).expect("truncate");
        drop(f);
        assert_eq!(
            WavReader::open(&path).err(),
            Some(WavError::SizeMismatch {
                header: 4000,
                actual: 3988
            })
        );
        assert_eq!(repair(&path).expect("repair"), 498);
        assert_eq!(
            std::fs::metadata(&path).expect("meta").len(),
            HEADER_BYTES + 498 * 8
        );
        let (_, got) = read_all(&path);
        assert_eq!(bits(&got), bits(&data[..2 * 498]));

        // A writer that never patched its header (crash) is recovered the same way.
        let crashed = dir.path().join("crashed.wav");
        let mut w = WavWriter::create(&crashed, spec(16_000, 1, 0)).expect("create");
        w.write_interleaved(&data[..300]).expect("write");
        drop(w);
        assert_eq!(WavReader::open(&crashed).expect("open").frames(), 0);
        assert_eq!(repair(&crashed).expect("repair"), 300);
        let (_, got) = read_all(&crashed);
        assert_eq!(bits(&got), bits(&data[..300]));
    }

    #[test]
    fn w4_size_limit_writes_nothing_partial() {
        let dir = TestDir::new("w4");
        let path = dir.path().join("limit.wav");
        let data = samples(2048, 4);
        let mut w = WavWriter::create_with_limit(&path, spec(48_000, 1, 0), 4096).expect("create");
        w.write_interleaved(&data[..1000]).expect("4000 bytes fit");
        assert_eq!(
            w.write_interleaved(&data[1000..1100]),
            Err(WavError::SizeLimitReached { limit: 4096 })
        );
        assert_eq!(w.frames(), 1000);
        w.write_interleaved(&data[1000..1024])
            .expect("exactly the limit fits");
        assert_eq!(
            w.write_interleaved(&data[1024..1025]),
            Err(WavError::SizeLimitReached { limit: 4096 })
        );
        assert_eq!(w.finalize().expect("finalize"), 1024);
        assert_eq!(
            std::fs::metadata(&path).expect("meta").len(),
            HEADER_BYTES + 4096
        );
        let (_, got) = read_all(&path);
        assert_eq!(bits(&got), bits(&data[..1024]));
    }
}
