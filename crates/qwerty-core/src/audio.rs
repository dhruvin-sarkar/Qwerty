//! Audio formats, blocks, the sample→host clock map, channel layout analysis and the capture
//! device identity (§1.4).

use crate::units::{
    ChannelCount, Hertz, HostDuration, HostTime, Milliseconds, SampleIndex, SampleRate,
};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioFormat {
    pub rate: SampleRate,
    pub channels: ChannelCount,
}

/// One contiguous run of interleaved f32 frames. Borrowed: the processing thread reuses its buffer.
#[derive(Debug, Clone, Copy)]
pub struct AudioBlock<'a> {
    pub format: AudioFormat,
    /// Position of frame 0.
    pub start: SampleIndex,
    /// len = frames · channels, interleaved.
    pub samples: &'a [f32],
    pub time: BlockTime,
    /// true ⇒ samples before `start` are not contiguous with this block.
    pub discontinuity: bool,
}

impl<'a> AudioBlock<'a> {
    /// samples.len() / channels.
    pub fn frames(&self) -> usize {
        self.samples.len() / usize::from(self.format.channels.get())
    }

    /// `AudioError::RaggedBlock` if len % channels != 0.
    pub fn new(
        format: AudioFormat,
        start: SampleIndex,
        samples: &'a [f32],
        time: BlockTime,
        discontinuity: bool,
    ) -> Result<Self, AudioError> {
        let channels = format.channels.get();
        if !samples.len().is_multiple_of(usize::from(channels)) {
            return Err(AudioError::RaggedBlock {
                len: samples.len(),
                channels,
            });
        }
        Ok(AudioBlock {
            format,
            start,
            samples,
            time,
            discontinuity,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum BlockTime {
    /// host time of frame 0 (WASAPI packet timestamp, or replay anchor interpolation)
    Timestamped { host_time: HostTime },
    /// device reported AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR
    Untimed,
}

/// Number of (SampleIndex, HostTime) anchors a `ClockMap` keeps.
const CLOCK_ANCHORS: usize = 64;
/// An anchor older than this, relative to the queried sample, is too stale to map it.
const CLOCK_ANCHOR_MAX_AGE: Milliseconds = Milliseconds::new(2000.0);

/// Maps SampleIndex → HostTime from the most recent anchor at or before the sample, at the nominal
/// rate: t = t_a + (s − s_a)·10^7/fs (i128 math, rounded). Keeps the last 64 anchors.
///
/// A fixed ring: `observe` never allocates, so it is safe on the processing thread (§0.9).
#[derive(Debug, Clone)]
pub struct ClockMap {
    rate: SampleRate,
    anchors: [(SampleIndex, HostTime); CLOCK_ANCHORS],
    len: usize,
    next: usize,
}

impl ClockMap {
    pub fn new(rate: SampleRate) -> Self {
        ClockMap {
            rate,
            anchors: [(SampleIndex(0), HostTime(0)); CLOCK_ANCHORS],
            len: 0,
            next: 0,
        }
    }

    /// Records an anchor if the block is Timestamped.
    pub fn observe(&mut self, block: &AudioBlock<'_>) {
        match block.time {
            BlockTime::Timestamped { host_time } => {
                self.anchors[self.next] = (block.start, host_time);
                self.next = (self.next + 1) % CLOCK_ANCHORS;
                self.len = (self.len + 1).min(CLOCK_ANCHORS);
            }
            BlockTime::Untimed => {}
        }
    }

    /// None if there is no anchor at or before `s`, or the nearest anchor is more than 2 s older than s.
    /// (Also None if the mapped time leaves the i64 tick range, which needs an anchor within 2 s of
    /// i64::MAX ticks, about 29 000 years of uptime.)
    pub fn host_time_of(&self, s: SampleIndex) -> Option<HostTime> {
        // Oldest to newest, so that on equal sample indices the later observation wins.
        let oldest = if self.len < CLOCK_ANCHORS {
            0
        } else {
            self.next
        };
        let mut best: Option<(SampleIndex, HostTime)> = None;
        for k in 0..self.len {
            let anchor = self.anchors[(oldest + k) % CLOCK_ANCHORS];
            let at_or_before = anchor.0 <= s;
            let nearer = best.is_none_or(|b| anchor.0 >= b.0);
            if at_or_before && nearer {
                best = Some(anchor);
            }
        }
        let (s_a, t_a) = best?;
        let gap = s.0 - s_a.0;
        if gap > self.rate.samples_in(CLOCK_ANCHOR_MAX_AGE) as u64 {
            return None;
        }
        // Round half up (gap ≥ 0): (2·gap·10^7 + fs) / (2·fs).
        let fs = i128::from(self.rate.hz());
        let num = i128::from(gap) * i128::from(HostTime::TICKS_PER_SECOND);
        let ticks = (2 * num + fs) / (2 * fs);
        t_a.checked_add(HostDuration(i64::try_from(ticks).ok()?))
    }

    /// Forgets every anchor (on discontinuity).
    pub fn reset(&mut self) {
        self.len = 0;
        self.next = 0;
    }
}

/// Which channels carry independent signal. Computed once, from the calibration ambient step;
/// stored in the device fingerprint. Fields are private (RUST-RT-17): the only constructors are
/// `all_real`, `analyze` and deserialization, and deserialization goes through a private raw shape
/// and `TryFrom`, so a hand-edited profile cannot yield a layout that panics later.
/// Invariants: `real` is non-empty and strictly ascending; every index is < `channels`; the real,
/// duplicate and silent channels are disjoint and cover 0..channels exactly once; every `of` is a
/// real channel lower than its `channel`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RawChannelLayout")]
pub struct ChannelLayout {
    channels: ChannelCount,
    real: Vec<u16>,
    duplicates: Vec<ChannelDuplicate>,
    silent: Vec<u16>,
}

/// The unvalidated serde shape of `ChannelLayout`, used by deserialization only.
#[derive(Deserialize)]
struct RawChannelLayout {
    channels: ChannelCount,
    real: Vec<u16>,
    duplicates: Vec<ChannelDuplicate>,
    silent: Vec<u16>,
}

impl TryFrom<RawChannelLayout> for ChannelLayout {
    type Error = LayoutError;
    fn try_from(raw: RawChannelLayout) -> Result<Self, Self::Error> {
        ChannelLayout::validated(raw.channels, raw.real, raw.duplicates, raw.silent)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelDuplicate {
    pub channel: u16,
    pub of: u16,
}

/// High-pass cutoff applied before the layout statistics.
const LAYOUT_HIGHPASS: Hertz = Hertz::new(100.0);
/// `analyze` needs at least this much audio.
const LAYOUT_MIN_DURATION: Milliseconds = Milliseconds::new(2000.0);
/// A channel whose high-passed RMS is below this is silent (−140 dBFS).
const SILENT_RMS: f64 = 1e-7;
/// |ρ| above this (with a matching level) makes a channel a duplicate.
const DUPLICATE_MIN_CORRELATION: f64 = 0.999;
/// |level difference| below this (with a matching correlation) makes a channel a duplicate.
const DUPLICATE_MAX_LEVEL_DB: f64 = 0.1;
/// Floor for the power ratio's `log10` argument and for the correlation denominator.
const LAYOUT_POWER_FLOOR: f64 = 1e-30;

impl ChannelLayout {
    pub fn channels(&self) -> ChannelCount {
        self.channels
    }

    pub fn real(&self) -> &[u16] {
        &self.real
    }

    pub fn duplicates(&self) -> &[ChannelDuplicate] {
        &self.duplicates
    }

    pub fn silent(&self) -> &[u16] {
        &self.silent
    }

    /// All channels treated as real. Used until the ambient step has been analyzed.
    pub fn all_real(channels: ChannelCount) -> Self {
        ChannelLayout {
            channels,
            real: (0..channels.get()).collect(),
            duplicates: Vec::new(),
            silent: Vec::new(),
        }
    }

    /// Rules, in f64, on the 100 Hz high-passed signal over the whole buffer (≥ 2 s required):
    ///  silent:    RMS < 1e-7 (−140 dBFS)
    ///  duplicate: c > p, both non-silent, |ρ(c,p)| > 0.999 and |10·log10(P_c/P_p)| < 0.1 dB → duplicate of the lowest such p
    ///  real:      the rest
    pub fn analyze(format: AudioFormat, interleaved: &[f32]) -> Result<Self, LayoutError> {
        let n = usize::from(format.channels.get());
        // SPEC: §1.4 names no error for a trailing partial frame; only whole frames are analyzed.
        let frames = interleaved.len() / n;
        let needed = format.rate.samples_in(LAYOUT_MIN_DURATION);
        if frames < needed {
            return Err(LayoutError::TooShort {
                frames: frames as u64,
                needed: needed as u64,
            });
        }
        if let Some(frame) = interleaved
            .chunks_exact(n)
            .position(|f| f.iter().any(|v| !v.is_finite()))
        {
            return Err(LayoutError::NonFiniteSample {
                frame: frame as u64,
            });
        }

        // One pass: high-pass every channel and accumulate Σy, Σy² and Σy_c·y_p (p < c) in f64.
        let mut filters = vec![HighPass::butterworth(format.rate, LAYOUT_HIGHPASS); n];
        let mut sum = vec![0.0f64; n];
        let mut sum_sq = vec![0.0f64; n];
        let mut cross = vec![0.0f64; n * n]; // cross[c·n + p] = Σ y_c·y_p for p < c
        let mut y = vec![0.0f64; n];
        for frame in interleaved.chunks_exact(n) {
            for (c, (&x, filter)) in frame.iter().zip(filters.iter_mut()).enumerate() {
                let v = f64::from(filter.process(x));
                y[c] = v;
                sum[c] += v;
                sum_sq[c] += v * v;
                for (acc, &yp) in cross[c * n..c * n + c].iter_mut().zip(&y[..c]) {
                    *acc += v * yp;
                }
            }
        }

        let count = frames as f64;
        let mean: Vec<f64> = sum.iter().map(|s| s / count).collect();
        let power: Vec<f64> = sum_sq.iter().map(|s| s / count).collect();
        let variance: Vec<f64> = power
            .iter()
            .zip(&mean)
            .map(|(p, m)| (p - m * m).max(0.0))
            .collect();
        let is_silent = |c: usize| power[c].sqrt() < SILENT_RMS;
        let is_duplicate = |c: usize, p: usize| {
            let cov = cross[c * n + p] / count - mean[c] * mean[p];
            let rho = cov / (variance[c] * variance[p]).max(LAYOUT_POWER_FLOOR).sqrt();
            let level_db = 10.0
                * (power[c].max(LAYOUT_POWER_FLOOR) / power[p].max(LAYOUT_POWER_FLOOR)).log10();
            rho.abs() > DUPLICATE_MIN_CORRELATION && level_db.abs() < DUPLICATE_MAX_LEVEL_DB
        };

        if (0..n).all(is_silent) {
            return Err(LayoutError::AllChannelsSilent);
        }
        let mut real: Vec<u16> = Vec::new();
        let mut duplicates = Vec::new();
        let mut silent = Vec::new();
        for c in 0..format.channels.get() {
            let ci = usize::from(c);
            if is_silent(ci) {
                silent.push(c);
                continue;
            }
            // SPEC: "the lowest such p" is taken over the channels already classified real, so
            // that every `of` is a real channel (the invariant) even when duplication is not
            // transitive under the thresholds.
            match real
                .iter()
                .copied()
                .find(|&p| is_duplicate(ci, usize::from(p)))
            {
                Some(of) => duplicates.push(ChannelDuplicate { channel: c, of }),
                None => real.push(c),
            }
        }
        ChannelLayout::validated(format.channels, real, duplicates, silent)
    }

    /// Checks every invariant, in the order the `LayoutInvalid` variants are listed.
    fn validated(
        channels: ChannelCount,
        real: Vec<u16>,
        duplicates: Vec<ChannelDuplicate>,
        silent: Vec<u16>,
    ) -> Result<Self, LayoutError> {
        let invalid = |reason| LayoutError::Invalid { reason };
        if real.is_empty() {
            return Err(invalid(LayoutInvalid::NoRealChannel));
        }
        if real.windows(2).any(|w| w[0] >= w[1]) {
            return Err(invalid(LayoutInvalid::NotAscending));
        }
        let n = channels.get();
        let every_index = real
            .iter()
            .copied()
            .chain(duplicates.iter().flat_map(|d| [d.channel, d.of]))
            .chain(silent.iter().copied());
        if let Some(channel) = every_index.into_iter().find(|&c| c >= n) {
            return Err(invalid(LayoutInvalid::ChannelOutOfRange { channel }));
        }
        let mut listed = [0u32; ChannelCount::MAX as usize];
        let members = real
            .iter()
            .chain(duplicates.iter().map(|d| &d.channel))
            .chain(silent.iter());
        for &c in members {
            listed[usize::from(c)] = listed[usize::from(c)].saturating_add(1);
        }
        for channel in 0..n {
            match listed[usize::from(channel)] {
                0 => return Err(invalid(LayoutInvalid::Uncovered { channel })),
                1 => {}
                _ => return Err(invalid(LayoutInvalid::Overlap { channel })),
            }
        }
        if let Some(d) = duplicates
            .iter()
            .find(|d| d.of >= d.channel || real.binary_search(&d.of).is_err())
        {
            return Err(invalid(LayoutInvalid::DuplicateOfNotReal {
                channel: d.channel,
            }));
        }
        Ok(ChannelLayout {
            channels,
            real,
            duplicates,
            silent,
        })
    }
}

/// Private copy of `dsp::Biquad::highpass` (§1.5.1): 2nd-order Butterworth high-pass, RBJ
/// cookbook, transposed direct form II, f64 coefficients and state. WP-A1 replaces it with a call
/// to `dsp::Biquad::highpass` (§1.4).
#[derive(Debug, Clone)]
struct HighPass {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl HighPass {
    /// Q = 1/√2. w0 = 2π·fc/fs, α = sin w0/(2Q),
    /// b = [(1+cos w0)/2, −(1+cos w0), (1+cos w0)/2]/a0, a = [−2cos w0, 1−α]/a0, a0 = 1+α.
    /// The only caller passes 100 Hz, below Nyquist at every valid rate.
    fn butterworth(rate: SampleRate, cutoff: Hertz) -> Self {
        let w0 = std::f64::consts::TAU * cutoff.get() / f64::from(rate.hz());
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * std::f64::consts::FRAC_1_SQRT_2);
        let a0 = 1.0 + alpha;
        HighPass {
            b0: (1.0 + cos) / 2.0 / a0,
            b1: -(1.0 + cos) / a0,
            b2: (1.0 + cos) / 2.0 / a0,
            a1: -2.0 * cos / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// y = b0·x + z1; z1 = b1·x − a1·y + z2; z2 = b2·x − a2·y
    fn process(&mut self, x: f32) -> f32 {
        let x = f64::from(x);
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y as f32
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LayoutError {
    TooShort { frames: u64, needed: u64 },
    AllChannelsSilent,
    NonFiniteSample { frame: u64 },
    Invalid { reason: LayoutInvalid },
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutError::TooShort { frames, needed } => {
                write!(f, "channel analysis needs {needed} frames, got {frames}")
            }
            LayoutError::AllChannelsSilent => write!(f, "every channel is silent"),
            LayoutError::NonFiniteSample { frame } => {
                write!(f, "non-finite sample in frame {frame}")
            }
            LayoutError::Invalid { reason } => write!(f, "invalid channel layout: {reason}"),
        }
    }
}

impl std::error::Error for LayoutError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LayoutInvalid {
    NoRealChannel,
    NotAscending,
    ChannelOutOfRange { channel: u16 },
    Overlap { channel: u16 },
    Uncovered { channel: u16 },
    DuplicateOfNotReal { channel: u16 },
}

impl fmt::Display for LayoutInvalid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutInvalid::NoRealChannel => write!(f, "no real channel"),
            LayoutInvalid::NotAscending => write!(f, "real channels are not strictly ascending"),
            LayoutInvalid::ChannelOutOfRange { channel } => {
                write!(f, "channel {channel} is out of range")
            }
            LayoutInvalid::Overlap { channel } => {
                write!(f, "channel {channel} is listed more than once")
            }
            LayoutInvalid::Uncovered { channel } => write!(f, "channel {channel} is not listed"),
            LayoutInvalid::DuplicateOfNotReal { channel } => write!(
                f,
                "channel {channel} duplicates a channel that is not a lower real channel"
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AudioError {
    RaggedBlock { len: usize, channels: u16 },
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioError::RaggedBlock { len, channels } => write!(
                f,
                "block of {len} samples is not a whole number of {channels}-channel frames"
            ),
        }
    }
}

impl std::error::Error for AudioError {}

/// How the capture stream was opened. This is the only processing-mode visibility Windows gives apps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CaptureMode {
    Raw,
    Default { reason: DefaultModeReason },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DefaultModeReason {
    /// PKEY_Devices_AudioDevice_RawProcessingSupported is false or absent
    RawNotSupported,
    /// RAW was requested and IAudioClient::Initialize failed with this HRESULT
    RawRejected { hresult: i32 },
}

/// Identity of an opened capture stream: what calibration binds to and what live capture is compared against.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceIdentity {
    /// IMMDevice::GetId
    pub endpoint_id: String,
    /// Friendly name, for display only; never compared.
    pub name: String,
    /// Shared-mode mix format: rate + channels.
    pub format: AudioFormat,
    pub capture_mode: CaptureMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum IdentityField {
    EndpointId,
    SampleRate,
    Channels,
    CaptureMode,
}

impl DeviceIdentity {
    /// Fields that differ, in the order above. `name` and the DefaultModeReason are ignored.
    pub fn differences(&self, live: &DeviceIdentity) -> Vec<IdentityField> {
        let mut out = Vec::new();
        if self.endpoint_id != live.endpoint_id {
            out.push(IdentityField::EndpointId);
        }
        if self.format.rate != live.format.rate {
            out.push(IdentityField::SampleRate);
        }
        if self.format.channels != live.format.channels {
            out.push(IdentityField::Channels);
        }
        if !same_capture_mode(self.capture_mode, live.capture_mode) {
            out.push(IdentityField::CaptureMode);
        }
        out
    }
}

/// Compares capture modes by variant only, ignoring the DefaultModeReason.
fn same_capture_mode(a: CaptureMode, b: CaptureMode) -> bool {
    match (a, b) {
        (CaptureMode::Raw, CaptureMode::Raw) => true,
        (CaptureMode::Default { .. }, CaptureMode::Default { .. }) => true,
        (CaptureMode::Raw, CaptureMode::Default { .. })
        | (CaptureMode::Default { .. }, CaptureMode::Raw) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng64;

    fn stereo(rate_hz: u32) -> AudioFormat {
        AudioFormat {
            rate: SampleRate::new(rate_hz).expect("valid rate"),
            channels: ChannelCount::new(2).expect("valid count"),
        }
    }

    /// Interleaves two channels of `frames` frames.
    fn interleave(frames: usize, mut f: impl FnMut(usize) -> (f32, f32)) -> Vec<f32> {
        let mut out = Vec::with_capacity(frames * 2);
        for i in 0..frames {
            let (l, r) = f(i);
            out.push(l);
            out.push(r);
        }
        out
    }

    fn two_seconds(format: AudioFormat) -> usize {
        format.rate.samples_in(Milliseconds::new(2000.0))
    }

    #[test]
    fn ad1_channel_layout_analyze() {
        let format = stereo(48_000);
        let frames = two_seconds(format);

        // Identical channels: channel 1 duplicates channel 0.
        let mut rng = Rng64::new(1);
        let same = interleave(frames, |_| {
            let v = (0.1 * rng.normal()) as f32;
            (v, v)
        });
        let layout = ChannelLayout::analyze(format, &same).expect("identical channels");
        assert_eq!(layout.real(), &[0]);
        assert_eq!(
            layout.duplicates(),
            &[ChannelDuplicate { channel: 1, of: 0 }]
        );
        assert!(layout.silent().is_empty());

        // Channel 1 silent.
        let mut rng = Rng64::new(2);
        let one_silent = interleave(frames, |_| ((0.1 * rng.normal()) as f32, 0.0));
        let layout = ChannelLayout::analyze(format, &one_silent).expect("one silent channel");
        assert_eq!(layout.real(), &[0]);
        assert_eq!(layout.silent(), &[1]);
        assert!(layout.duplicates().is_empty());

        // Independent noise on each channel.
        let mut rng = Rng64::new(3);
        let independent = interleave(frames, |_| {
            ((0.1 * rng.normal()) as f32, (0.1 * rng.normal()) as f32)
        });
        let layout = ChannelLayout::analyze(format, &independent).expect("independent noise");
        assert_eq!(layout.real(), &[0, 1]);
        assert!(layout.duplicates().is_empty());
        assert!(layout.silent().is_empty());

        // Shorter than 2 s.
        let short = &independent[..(frames - 1) * 2];
        assert_eq!(
            ChannelLayout::analyze(format, short),
            Err(LayoutError::TooShort {
                frames: (frames - 1) as u64,
                needed: frames as u64,
            })
        );
    }

    fn raw_json(channels: u16, real: &str, duplicates: &str, silent: &str) -> String {
        format!(
            r#"{{"channels":{channels},"real":{real},"duplicates":{duplicates},"silent":{silent}}}"#
        )
    }

    fn invalid_reason(json: &str) -> LayoutInvalid {
        let raw: RawChannelLayout = serde_json::from_str(json).expect("raw shape parses");
        match ChannelLayout::try_from(raw) {
            Err(LayoutError::Invalid { reason }) => reason,
            other => panic!("expected Invalid for {json}, got {other:?}"),
        }
    }

    #[test]
    fn al1_layout_deserialization_validates() {
        let cases = [
            (
                raw_json(2, "[]", "[]", "[0,1]"),
                LayoutInvalid::NoRealChannel,
            ),
            (
                raw_json(2, "[1,0]", "[]", "[]"),
                LayoutInvalid::NotAscending,
            ),
            (
                raw_json(2, "[0,0]", "[]", "[1]"),
                LayoutInvalid::NotAscending,
            ),
            (
                raw_json(2, "[0,2]", "[]", "[]"),
                LayoutInvalid::ChannelOutOfRange { channel: 2 },
            ),
            (
                raw_json(2, "[0]", "[]", "[5]"),
                LayoutInvalid::ChannelOutOfRange { channel: 5 },
            ),
            (
                raw_json(2, "[0]", r#"[{"channel":1,"of":7}]"#, "[]"),
                LayoutInvalid::ChannelOutOfRange { channel: 7 },
            ),
            (
                raw_json(2, "[0,1]", "[]", "[1]"),
                LayoutInvalid::Overlap { channel: 1 },
            ),
            (
                raw_json(3, "[0]", "[]", "[2]"),
                LayoutInvalid::Uncovered { channel: 1 },
            ),
            (
                raw_json(3, "[0]", r#"[{"channel":1,"of":2}]"#, "[2]"),
                LayoutInvalid::DuplicateOfNotReal { channel: 1 },
            ),
            (
                raw_json(3, "[1]", r#"[{"channel":0,"of":1}]"#, "[2]"),
                LayoutInvalid::DuplicateOfNotReal { channel: 0 },
            ),
        ];
        for (json, expected) in &cases {
            assert_eq!(invalid_reason(json), *expected, "{json}");
            // The public type refuses it as a JSON error too.
            assert!(
                serde_json::from_str::<ChannelLayout>(json).is_err(),
                "{json}"
            );
        }

        // A valid layout deserializes.
        let ok: ChannelLayout =
            serde_json::from_str(&raw_json(3, "[0]", r#"[{"channel":1,"of":0}]"#, "[2]"))
                .expect("valid layout");
        assert_eq!(ok.real(), &[0]);

        // Every layout `analyze` returns round-trips, as does `all_real`.
        let format = stereo(44_100);
        let frames = two_seconds(format);
        let mut rng = Rng64::new(9);
        let mut layouts = vec![ChannelLayout::all_real(format.channels)];
        for kind in 0..3 {
            let buf = interleave(frames, |_| {
                let a = (0.05 * rng.normal()) as f32;
                let b = (0.05 * rng.normal()) as f32;
                match kind {
                    0 => (a, a),
                    1 => (0.0, b),
                    _ => (a, b),
                }
            });
            layouts.push(ChannelLayout::analyze(format, &buf).expect("analyze"));
        }
        for layout in layouts {
            let json = serde_json::to_string(&layout).expect("serialize");
            let back: ChannelLayout = serde_json::from_str(&json).expect("round-trip");
            assert_eq!(back, layout);
        }
    }

    #[test]
    fn analyze_reports_non_finite_and_all_silent() {
        let format = stereo(16_000);
        let frames = two_seconds(format);
        let mut buf = vec![0.0f32; frames * 2];
        assert_eq!(
            ChannelLayout::analyze(format, &buf),
            Err(LayoutError::AllChannelsSilent)
        );
        buf[2 * 100 + 1] = f32::NAN;
        assert_eq!(
            ChannelLayout::analyze(format, &buf),
            Err(LayoutError::NonFiniteSample { frame: 100 })
        );
    }

    #[test]
    fn block_rejects_ragged_and_counts_frames() {
        let format = stereo(48_000);
        let samples = [0.0f32; 6];
        let block = AudioBlock::new(format, SampleIndex(0), &samples, BlockTime::Untimed, false)
            .expect("whole frames");
        assert_eq!(block.frames(), 3);
        assert_eq!(
            AudioBlock::new(
                format,
                SampleIndex(0),
                &samples[..5],
                BlockTime::Untimed,
                false
            )
            .map(|b| b.frames()),
            Err(AudioError::RaggedBlock {
                len: 5,
                channels: 2
            })
        );
    }

    #[test]
    fn clock_map_maps_from_latest_anchor() {
        let format = stereo(48_000);
        let samples = [0.0f32; 2];
        let block = |start: u64, time: BlockTime| {
            AudioBlock::new(format, SampleIndex(start), &samples, time, false).expect("block")
        };
        let mut map = ClockMap::new(format.rate);
        assert_eq!(map.host_time_of(SampleIndex(0)), None);
        map.observe(&block(
            1000,
            BlockTime::Timestamped {
                host_time: HostTime(5_000_000),
            },
        ));
        map.observe(&block(2000, BlockTime::Untimed));
        // Before the only anchor: unmapped.
        assert_eq!(map.host_time_of(SampleIndex(999)), None);
        // 48 frames = 1 ms = 10_000 ticks; 1 frame = 208.33 ticks → 208.
        assert_eq!(
            map.host_time_of(SampleIndex(1048)),
            Some(HostTime(5_010_000))
        );
        assert_eq!(
            map.host_time_of(SampleIndex(1001)),
            Some(HostTime(5_000_208))
        );
        // Exactly 2 s after the anchor maps; one frame more does not.
        assert_eq!(
            map.host_time_of(SampleIndex(1000 + 96_000)),
            Some(HostTime(25_000_000))
        );
        assert_eq!(map.host_time_of(SampleIndex(1000 + 96_001)), None);
        // A later anchor takes over.
        map.observe(&block(
            48_000,
            BlockTime::Timestamped {
                host_time: HostTime(9_000_000),
            },
        ));
        assert_eq!(
            map.host_time_of(SampleIndex(48_048)),
            Some(HostTime(9_010_000))
        );
        assert_eq!(
            map.host_time_of(SampleIndex(47_999)),
            Some(HostTime(14_791_458))
        );
        // The ring keeps the last 64 anchors only.
        for k in 0..CLOCK_ANCHORS as u64 {
            map.observe(&block(
                100_000 + k,
                BlockTime::Timestamped {
                    host_time: HostTime(0),
                },
            ));
        }
        assert_eq!(map.host_time_of(SampleIndex(48_048)), None);
        map.reset();
        assert_eq!(map.host_time_of(SampleIndex(100_010)), None);
    }

    #[test]
    fn identity_differences_ignore_name_and_reason() {
        let base = DeviceIdentity {
            endpoint_id: "{0.0.1.00000000}.{abc}".to_string(),
            name: "Mic".to_string(),
            format: stereo(48_000),
            capture_mode: CaptureMode::Default {
                reason: DefaultModeReason::RawNotSupported,
            },
        };
        let mut live = base.clone();
        live.name = "Renamed".to_string();
        live.capture_mode = CaptureMode::Default {
            reason: DefaultModeReason::RawRejected { hresult: -1 },
        };
        assert!(base.differences(&live).is_empty());
        live.endpoint_id = "other".to_string();
        live.format = AudioFormat {
            rate: SampleRate::new(44_100).expect("valid rate"),
            channels: ChannelCount::new(1).expect("valid count"),
        };
        live.capture_mode = CaptureMode::Raw;
        assert_eq!(
            base.differences(&live),
            vec![
                IdentityField::EndpointId,
                IdentityField::SampleRate,
                IdentityField::Channels,
                IdentityField::CaptureMode,
            ]
        );
    }
}
