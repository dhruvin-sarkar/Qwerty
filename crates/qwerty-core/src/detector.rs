//! Onset detector: turns audio blocks into tap candidates and named rejections (§1.6).
//!
//! Every decision depends only on sample positions, never on block boundaries: blocks are consumed
//! in chunks that end exactly at the next event (a grid frame completing, an AIC window or an
//! anchored span becoming available), and the events at one position run in a fixed order (frame,
//! then AIC, then emission). Any block segmentation therefore gives bit-identical outputs (T4).

use crate::audio::{AudioBlock, AudioFormat, ChannelLayout, ClockMap};
use crate::descriptor::DescriptorLayout;
use crate::dsp::{BandLayout, Biquad, FrameGeometry, PowerSpectrum, BAND_COUNT};
use crate::units::{
    CandidateId, Decibels, Hertz, HostTime, Milliseconds, SampleIndex, SampleRate, Seconds,
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;

pub const FRONT_END_VERSION: u32 = 1;
pub const ANCHORED_FRAMES: usize = 7;
/// ≥ 22 grid frames of pre-roll so FrozenNoise re-analysis has a full peak-pick mean (19 + lag 2 + slack)
pub const CLIP_PRE: Milliseconds = Milliseconds::new(150.0);
pub const CLIP_POST: Milliseconds = Milliseconds::new(60.0);
/// time-domain analysis span [t0-4, t0+40) ms
pub const SPAN_BEFORE: Milliseconds = Milliseconds::new(4.0);
pub const SPAN_AFTER: Milliseconds = Milliseconds::new(40.0);
pub const HISTORY: Milliseconds = Milliseconds::new(400.0);
pub const MAX_PENDING: usize = 4;

/// `HealthWarning::NoiseTrackerStarved` (§2) is raised when `frozen_fraction` > STARVED_ENTER and
/// cleared when it falls below STARVED_EXIT (hysteresis).
pub const STARVED_ENTER: f64 = 0.5;
pub const STARVED_EXIT: f64 = 0.25;

/// Floor of the AIC segment variances (§1.6.3 step 13).
const AIC_VARIANCE_FLOOR: f64 = 1e-20;
/// Floor of the whitened mean excess in the SNR gate (§1.6.3 step 15).
const SNR_EXCESS_FLOOR: f64 = 1e-6;
/// Floor of every power ratio converted to dB in the gates, so digital silence stays finite.
const GATE_RATIO_FLOOR: f64 = 1e-30;
/// Floor of band powers converted to dB for telemetry (−300 dB).
const TELEMETRY_POWER_FLOOR: f64 = 1e-30;
/// Window of the rolling health statistics.
const HEALTH_WINDOW: Milliseconds = Milliseconds::new(10_000.0);
/// `estimate_noise` needs at least this many usable grid frames.
const MIN_AMBIENT_FRAMES: u64 = 100;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DetectorParams {
    pub hpf_cutoff: Hertz,                 // 100 Hz
    pub noise_tau: Seconds,                // 1.5 s, both directions (ADR-002 "gated EMA"; step 11)
    pub noise_freeze: Milliseconds,        // 300 ms (ADR-002)
    pub noise_release_snr: Decibels, // 0 dB: a candidate rejected with snr < this releases its freeze
    pub noise_recovery_ratio: f64, // 0.25: pooled mean_b E_b/N_b below this (a drop of about 6 dB or more) ...
    pub noise_recovery_frames: usize, // 3: ... on this many consecutive non-frozen frames re-warms the tracker (step 11)
    pub warmup: Milliseconds,         // 200 ms
    pub noise_floor_power: f64,       // 1e-12 (-120 dB; below any real mic)
    pub flux_lag_frames: usize,       // 2
    pub flux_max_filter_radius: usize, // 1 (0 = no max filter; a tunable per ADR-002)
    pub peak_local_max_frames: usize, // 6 (~32 ms, causal: frame m is the max of m-5..=m)
    pub peak_mean_frames: usize,      // 19 (~101 ms, frames m-19..=m-1)
    pub peak_delta: f64,              // 0.18 (sim)
    pub min_trigger_spacing: Milliseconds, // 30 ms
    pub supersede_ratio: f64,         // 2.0
    pub aic_before: Milliseconds,     // 10 ms
    pub aic_after: Milliseconds, // 6 ms (≥ L/2 + aic_min_segment = 5.83 ms at every rate, rounded up; DSP-ML-2)
    pub aic_min_segment: Milliseconds, // 0.5 ms
    pub anchor_offset: Milliseconds, // 2 ms (centre of anchored frame 0 = t0 + 2 ms)
    pub snr_gate: Decibels,      // 10 dB
    pub broadband_rise: Decibels, // 6 dB
    pub broadband_fraction: f64, // 0.40
    pub decay_valid_band_snr: Decibels, // 10 dB
    pub decay_min_valid_bands: usize, // 3
    pub decay_drop: Decibels,    // 8 dB
    pub decay_fraction: f64,     // 0.60
    pub clip_level: f32,         // 0.999
}

impl DetectorParams {
    pub const DEFAULT: DetectorParams = DetectorParams {
        hpf_cutoff: Hertz::new(100.0),
        noise_tau: Seconds::new(1.5),
        noise_freeze: Milliseconds::new(300.0),
        noise_release_snr: Decibels::new(0.0),
        noise_recovery_ratio: 0.25,
        noise_recovery_frames: 3,
        warmup: Milliseconds::new(200.0),
        noise_floor_power: 1e-12,
        flux_lag_frames: 2,
        flux_max_filter_radius: 1,
        peak_local_max_frames: 6,
        peak_mean_frames: 19,
        peak_delta: 0.18,
        min_trigger_spacing: Milliseconds::new(30.0),
        supersede_ratio: 2.0,
        aic_before: Milliseconds::new(10.0),
        aic_after: Milliseconds::new(6.0),
        aic_min_segment: Milliseconds::new(0.5),
        anchor_offset: Milliseconds::new(2.0),
        snr_gate: Decibels::new(10.0),
        broadband_rise: Decibels::new(6.0),
        broadband_fraction: 0.40,
        decay_valid_band_snr: Decibels::new(10.0),
        decay_min_valid_bands: 3,
        decay_drop: Decibels::new(8.0),
        decay_fraction: 0.60,
        clip_level: 0.999,
    };
}

// SPEC: §1.6.2 fixes `FrozenNoise { noise: NoiseSnapshot, .. }` by value; the mode is built once
// per detector, so the variant size difference costs nothing.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DetectorMode {
    /// emit when the anchored span [t0-3.33, t0+39.33] ms is complete
    Live,
    /// emit at max(span end, t0 + CLIP_POST) with a RawClip attached
    CalibrationClips,
    /// augmentation re-analysis: the tracker is fixed to `noise`, no warm-up, never updates.
    /// `anchor_jitter` (samples, may be negative) is added to t0 after AIC, before anchored analysis.
    FrozenNoise {
        noise: NoiseSnapshot,
        anchor_jitter: i32,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoiseSnapshot {
    pub pooled: [f64; BAND_COUNT],      // sum over real channels
    pub chosen: Vec<[f64; BAND_COUNT]>, // per DescriptorLayout channel: [a] or [a, c]
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateReport {
    pub snr: Decibels,
    pub broadband_fraction: f64,
    pub valid_bands: u8,
    pub decay_fraction: f64, // 0.0 when valid_bands < decay_min_valid_bands
    pub clipped: bool,
    pub trigger_flux: f64, // D(m) at the trigger frame
}

/// Everything DescriptorExtractor::extract needs; produced only for gate-passing candidates.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisWindow {
    pub rate: SampleRate,
    pub geometry: FrameGeometry,
    pub layout: DescriptorLayout,
    pub pooled: [[f64; BAND_COUNT]; ANCHORED_FRAMES], // E_b(j), summed over real channels
    pub chosen: Vec<[[f64; BAND_COUNT]; ANCHORED_FRAMES]>, // per chosen channel
    pub noise: NoiseSnapshot,                         // N* at the trigger
    pub span: Vec<Vec<f32>>, // HPF samples per chosen channel over [t0 - SPAN_BEFORE, t0 + SPAN_AFTER)
    pub span_onset: usize,   // index of t0 inside each span
}

#[derive(Debug, Clone, PartialEq)]
pub struct RawClip {
    pub format: AudioFormat,
    pub start: SampleIndex,
    pub samples: Vec<f32>, // raw (pre-HPF), interleaved, ALL channels, [t0 - CLIP_PRE, t0 + CLIP_POST)
    pub onset_offset: usize, // frames from clip start to t0
}

#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub id: CandidateId,
    pub trigger: SampleIndex,         // centre of the trigger frame
    pub onset: SampleIndex,           // refined t0
    pub onset_time: Option<HostTime>, // from the detector's ClockMap; None = untimed
    pub emitted_at: SampleIndex,      // newest sample consumed at emission
    pub gates: GateReport,
    pub analysis: AnalysisWindow,
    pub clip: Option<RawClip>, // Some iff mode == CalibrationClips
}

/// Rejected candidates never carry audio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RejectedCandidate {
    pub id: CandidateId,
    pub trigger: SampleIndex,
    pub onset: Option<SampleIndex>,
    pub reason: RejectReason,
    pub gates: Option<GateReport>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RejectReason {
    LowSnr,
    NotBroadband,
    InsufficientBandSnr,
    NotDecaying,
    OnsetNotLocalized,
    Superseded,
    SpanCrossesDiscontinuity,
    PendingOverflow,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InputFault {
    NonFinite {
        block_start: SampleIndex,
        count: u32,
    },
    Discontinuity {
        at: SampleIndex,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum DetectorOutput {
    Candidate(Box<Candidate>),
    Rejected(RejectedCandidate),
    InputFault(InputFault),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DetectorState {
    Warming { remaining: Milliseconds },
    Running,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DetectorHealth {
    pub frozen_fraction: f64,     // over the last 10 s of frames
    pub triggers_per_second: f64, // over the last 10 s
    pub superseded: u64,
    pub pending_overflow: u64,
    pub faults: u64,
    pub rewarms: u64, // step 11 recoveries since construction
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameTelemetry {
    pub at: SampleIndex,
    pub bands_db: [f32; BAND_COUNT],
    pub noise_db: [f32; BAND_COUNT],
    pub flux: f32,
    pub threshold: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DetectorError {
    FormatMismatch {
        expected: AudioFormat,
        got: AudioFormat,
    },
    LayoutChannelOutOfRange {
        channel: u16,
        channels: u16,
    },
    InsufficientAmbient {
        frames: u64,
        needed: u64,
    },
}

impl fmt::Display for DetectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DetectorError::FormatMismatch { expected, got } => write!(
                f,
                "block format {} Hz × {} does not match the detector's {} Hz × {}",
                got.rate.hz(),
                got.channels.get(),
                expected.rate.hz(),
                expected.channels.get()
            ),
            DetectorError::LayoutChannelOutOfRange { channel, channels } => {
                write!(f, "layout channel {channel} is outside 0..{channels}")
            }
            DetectorError::InsufficientAmbient { frames, needed } => {
                write!(f, "ambient has {frames} usable frames, {needed} needed")
            }
        }
    }
}

impl std::error::Error for DetectorError {}

/// The newest `cap` sample frames of `channels` channels, addressed by absolute sample index.
struct History {
    cap: usize,
    buf: Vec<f32>, // channel-major: buf[ch·cap + s % cap]
    valid_from: u64,
    next: u64,
}

impl History {
    fn new(channels: usize, cap: usize) -> Self {
        History {
            cap,
            buf: vec![0.0; channels * cap],
            valid_from: 0,
            next: 0,
        }
    }

    fn reset(&mut self, at: u64) {
        self.valid_from = at;
        self.next = at;
    }

    fn write(&mut self, ch: usize, x: f32) {
        let slot = (self.next % self.cap as u64) as usize;
        self.buf[ch * self.cap + slot] = x;
    }

    fn advance(&mut self) {
        self.next += 1;
    }

    /// [start, end) is held: after the last reset, not yet overwritten, and already consumed.
    fn covers(&self, start: u64, end: u64) -> bool {
        start >= self.valid_from && start + self.cap as u64 >= self.next && end <= self.next
    }

    /// Copies channel `ch` over [start, start + out.len()); the caller checked `covers`.
    fn copy(&self, ch: usize, start: u64, out: &mut [f32]) {
        let base = ch * self.cap;
        let first = (start % self.cap as u64) as usize;
        let head = out.len().min(self.cap - first);
        out[..head].copy_from_slice(&self.buf[base + first..base + first + head]);
        let rest = out.len() - head;
        out[head..].copy_from_slice(&self.buf[base..base + rest]);
    }

    /// One sample; the caller checked `covers`.
    fn get(&self, ch: usize, s: u64) -> f32 {
        self.buf[ch * self.cap + (s % self.cap as u64) as usize]
    }
}

/// Noise-tracker phase.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Tracker {
    /// Running mean over the first `total` frames (step 8; re-entered by the step 11 recovery).
    Warming { done: usize, total: usize },
    /// Gated EMA (step 11).
    Tracking,
    /// FrozenNoise: fixed to the given snapshot.
    Fixed,
}

/// One trigger, kept while it can still freeze the tracker or bound the trigger spacing.
#[derive(Debug, Clone, Copy)]
struct TriggerRecord {
    id: CandidateId,
    frame_start: u64,
    freeze_end: u64,
    flux: f64,
    released: bool,
    not_localized: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Stage {
    /// Waiting for HPF history to cover t_c + aic_after.
    Aic { ready: u64 },
    /// t0 is known; waiting for the anchored span (and the clip) to be complete.
    Analysis { onset: u64, ready: u64 },
}

#[derive(Debug, Clone)]
struct Pending {
    id: CandidateId,
    trigger: u64, // t_c
    flux: f64,
    noise: NoiseSnapshot,
    stage: Stage,
}

impl Pending {
    fn ready(&self) -> u64 {
        match self.stage {
            Stage::Aic { ready } | Stage::Analysis { ready, .. } => ready,
        }
    }

    fn onset(&self) -> Option<SampleIndex> {
        match self.stage {
            Stage::Aic { .. } => None,
            Stage::Analysis { onset, .. } => Some(SampleIndex(onset)),
        }
    }
}

/// Sample counts derived once from the params and the rate.
#[derive(Debug, Clone, Copy)]
struct Spans {
    aic_before: u64,
    aic_after: u64,
    aic_min: usize,
    anchor_offset: u64,
    span_before: u64,
    span_after: u64,
    clip_pre: u64,
    clip_post: u64,
    freeze: u64,
    spacing: u64,
}

pub struct Detector {
    format: AudioFormat,
    params: DetectorParams,
    clips: bool,
    jitter: i64,
    geometry: FrameGeometry,
    spans: Spans,
    spectrum: PowerSpectrum,
    bands: BandLayout,
    real: Vec<usize>,
    descriptor_layout: DescriptorLayout,
    chosen_pos: Vec<usize>, // positions of the chosen channels inside `real`
    hpf: Vec<Biquad>,
    raw: History,
    filtered: History,
    clock: ClockMap,
    // Frame grid.
    started: bool,
    restart: bool,
    origin: u64,
    next_frame: u64,
    // Noise tracker and flux.
    tracker: Tracker,
    warmup_frames: usize,
    alpha: f64,
    noise: NoiseSnapshot,
    first_flux_frame: u64,
    yhat: Vec<[f64; BAND_COUNT]>, // ring, frame m at m % len
    flux: Vec<f64>,               // ring, frame m at m % len
    low_ratio_run: usize,
    // Triggers and candidates.
    triggers: VecDeque<TriggerRecord>,
    pending: VecDeque<Pending>,
    next_id: u64,
    // Health.
    health_flags: Vec<u8>, // ring of per-frame flags: bit 0 frozen, bit 1 trigger
    health_next: usize,
    health_filled: usize,
    health_frozen: usize,
    health_triggers: usize,
    superseded: u64,
    pending_overflow: u64,
    faults: u64,
    rewarms: u64,
    last_frame: Option<FrameTelemetry>,
    // Scratch.
    frame_buf: Vec<f32>,
    power: Vec<f64>,
    channel_bands: Vec<[f64; BAND_COUNT]>,
    aic_x: Vec<f32>,
    aic_sums: Vec<(f64, f64)>,
}

const FLAG_FROZEN: u8 = 1;
const FLAG_TRIGGER: u8 = 2;

impl Detector {
    /// # Panics
    /// If `params.hpf_cutoff` is not inside (0, fs/2). `DetectorParams::DEFAULT` (100 Hz) always is.
    pub fn new(
        format: AudioFormat,
        layout: &ChannelLayout,
        params: &DetectorParams,
        mode: DetectorMode,
    ) -> Result<Self, DetectorError> {
        let channels = format.channels.get();
        if let Some(&channel) = layout.real().iter().find(|&&c| c >= channels) {
            return Err(DetectorError::LayoutChannelOutOfRange { channel, channels });
        }
        let rate = format.rate;
        let geometry = FrameGeometry::for_rate(rate);
        let real: Vec<usize> = layout.real().iter().map(|&c| usize::from(c)).collect();
        let descriptor_layout = DescriptorLayout::from_channels(layout);
        let chosen = descriptor_layout.chosen();
        let chosen_pos: Vec<usize> = chosen
            .iter()
            .map(|c| {
                // INVARIANT: DescriptorLayout::from_channels picks its channels from `real`.
                layout
                    .real()
                    .iter()
                    .position(|r| r == c)
                    .expect("chosen channels are real")
            })
            .collect();
        let (clips, jitter, fixed) = match &mode {
            DetectorMode::Live => (false, 0, None),
            DetectorMode::CalibrationClips => (true, 0, None),
            DetectorMode::FrozenNoise {
                noise,
                anchor_jitter,
            } => (false, i64::from(*anchor_jitter), Some(noise.clone())),
        };
        if let Some(noise) = &fixed {
            if noise.chosen.len() != chosen.len() {
                // SPEC: §1.6.2 names no error for a FrozenNoise snapshot taken under another
                // layout; it is reported as the layout's last chosen channel against the number
                // of channels the snapshot holds.
                return Err(DetectorError::LayoutChannelOutOfRange {
                    channel: chosen[chosen.len() - 1],
                    channels: noise.chosen.len() as u16,
                });
            }
        }
        let hpf = Biquad::highpass(rate, params.hpf_cutoff)
            .unwrap_or_else(|e| panic!("DetectorParams::hpf_cutoff is invalid: {e}"));
        let samples = |d: Milliseconds| rate.samples_in(d) as u64;
        let spans = Spans {
            aic_before: samples(params.aic_before),
            aic_after: samples(params.aic_after),
            aic_min: rate.samples_in(params.aic_min_segment),
            anchor_offset: samples(params.anchor_offset),
            span_before: samples(SPAN_BEFORE),
            span_after: samples(SPAN_AFTER),
            clip_pre: samples(CLIP_PRE),
            clip_post: samples(CLIP_POST),
            freeze: samples(params.noise_freeze),
            spacing: samples(params.min_trigger_spacing),
        };
        let fs = f64::from(rate.hz());
        let hop = geometry.hop as f64;
        // round(warmup/hop), half away from zero; exact for integral ms (37.5 → 38 at 48 kHz).
        let warmup_frames = (params.warmup.get() * fs / (1000.0 * hop)).round() as usize;
        let alpha = 1.0 - (-hop / (fs * params.noise_tau.get())).exp();
        let history = rate.samples_in(HISTORY);
        let health_len = rate.samples_in(HEALTH_WINDOW).div_ceil(geometry.hop);
        let flux_len = (params.peak_mean_frames + 1).max(params.peak_local_max_frames);
        let aic_len = (spans.aic_before + spans.aic_after) as usize;
        let (tracker, noise) = match fixed {
            Some(noise) => (Tracker::Fixed, noise),
            None => (
                Tracker::Warming {
                    done: 0,
                    total: warmup_frames,
                },
                NoiseSnapshot {
                    pooled: [0.0; BAND_COUNT],
                    chosen: vec![[0.0; BAND_COUNT]; chosen.len()],
                },
            ),
        };
        Ok(Detector {
            format,
            params: *params,
            clips,
            jitter,
            geometry,
            spans,
            spectrum: PowerSpectrum::new(geometry),
            bands: BandLayout::new(rate, geometry),
            hpf: vec![hpf; real.len()],
            raw: History::new(usize::from(channels), history),
            filtered: History::new(real.len(), history),
            clock: ClockMap::new(rate),
            channel_bands: vec![[0.0; BAND_COUNT]; real.len()],
            real,
            descriptor_layout,
            chosen_pos,
            started: false,
            restart: false,
            origin: 0,
            next_frame: 0,
            tracker,
            warmup_frames,
            alpha,
            noise,
            first_flux_frame: 0,
            yhat: vec![[0.0; BAND_COUNT]; params.flux_lag_frames + 1],
            flux: vec![0.0; flux_len],
            low_ratio_run: 0,
            triggers: VecDeque::with_capacity(16),
            pending: VecDeque::with_capacity(MAX_PENDING + 1),
            next_id: 0,
            health_flags: vec![0; health_len],
            health_next: 0,
            health_filled: 0,
            health_frozen: 0,
            health_triggers: 0,
            superseded: 0,
            pending_overflow: 0,
            faults: 0,
            rewarms: 0,
            last_frame: None,
            frame_buf: vec![0.0; geometry.len],
            power: vec![0.0; geometry.fft_len / 2 + 1],
            aic_x: vec![0.0; aic_len],
            aic_sums: vec![(0.0, 0.0); aic_len + 1],
        })
    }

    /// Appends outputs in emission order; never allocates when nothing is emitted.
    pub fn push(
        &mut self,
        block: &AudioBlock<'_>,
        out: &mut Vec<DetectorOutput>,
    ) -> Result<(), DetectorError> {
        if block.format != self.format {
            return Err(DetectorError::FormatMismatch {
                expected: self.format,
                got: block.format,
            });
        }
        let bad = block.samples.iter().filter(|v| !v.is_finite()).count();
        if bad > 0 {
            out.push(DetectorOutput::InputFault(InputFault::NonFinite {
                block_start: block.start,
                count: bad.min(u32::MAX as usize) as u32,
            }));
            self.faults += 1;
            // SPEC: step 3 treats the block as a discontinuity; the NonFinite fault is the one
            // report of it (no second Discontinuity fault), and the next block starts the new grid.
            self.break_continuity(out);
            self.restart = true;
            return Ok(());
        }
        let start = block.start.0;
        if !self.started || self.restart {
            self.restart_grid(start);
        } else if block.discontinuity || start != self.raw.next {
            // SPEC: a block that does not start where the previous one ended is not contiguous
            // with it either, so it is handled exactly like a flagged discontinuity.
            self.break_continuity(out);
            out.push(DetectorOutput::InputFault(InputFault::Discontinuity {
                at: block.start,
            }));
            self.faults += 1;
            self.restart_grid(start);
        }
        self.clock.observe(block);

        let channels = usize::from(self.format.channels.get());
        let mut frames = block.samples.chunks_exact(channels);
        let mut remaining = block.frames();
        loop {
            let pos = self.raw.next;
            let event = self.next_event();
            if event <= pos {
                self.handle_events(out);
                continue;
            }
            if remaining == 0 {
                return Ok(());
            }
            let take = remaining.min((event - pos) as usize);
            for frame in frames.by_ref().take(take) {
                self.consume(frame);
            }
            remaining -= take;
        }
    }

    pub fn state(&self) -> DetectorState {
        match self.tracker {
            Tracker::Warming { done, total } => DetectorState::Warming {
                remaining: self
                    .format
                    .rate
                    .duration_of(((total - done) * self.geometry.hop) as u64),
            },
            Tracker::Tracking | Tracker::Fixed => DetectorState::Running,
        }
    }

    pub fn noise(&self) -> &NoiseSnapshot {
        &self.noise
    }

    pub fn health(&self) -> DetectorHealth {
        let (frozen_fraction, triggers_per_second) = if self.health_filled == 0 {
            (0.0, 0.0)
        } else {
            let frames = self.health_filled as f64;
            let seconds = frames * self.geometry.hop as f64 / f64::from(self.format.rate.hz());
            (
                self.health_frozen as f64 / frames,
                self.health_triggers as f64 / seconds,
            )
        };
        DetectorHealth {
            frozen_fraction,
            triggers_per_second,
            superseded: self.superseded,
            pending_overflow: self.pending_overflow,
            faults: self.faults,
            rewarms: self.rewarms,
        }
    }

    pub fn last_frame(&self) -> Option<FrameTelemetry> {
        self.last_frame
    }

    fn consume(&mut self, frame: &[f32]) {
        for (ch, &x) in frame.iter().enumerate() {
            self.raw.write(ch, x);
        }
        for (ri, (&ch, hpf)) in self.real.iter().zip(&mut self.hpf).enumerate() {
            self.filtered.write(ri, hpf.process(frame[ch]));
        }
        self.raw.advance();
        self.filtered.advance();
    }

    fn frame_end(&self) -> u64 {
        self.origin + self.next_frame * self.geometry.hop as u64 + self.geometry.len as u64
    }

    fn next_event(&self) -> u64 {
        self.pending
            .iter()
            .map(Pending::ready)
            .fold(self.frame_end(), u64::min)
    }

    /// Runs every event due at the current position: frames, then AIC, then emission.
    fn handle_events(&mut self, out: &mut Vec<DetectorOutput>) {
        let pos = self.raw.next;
        while self.frame_end() <= pos {
            self.process_frame(out);
        }
        let mut i = 0;
        while i < self.pending.len() {
            match self.pending[i].stage {
                Stage::Aic { ready } if ready <= pos => {
                    if self.refine_onset(i, out) {
                        i += 1;
                    }
                }
                Stage::Aic { .. } | Stage::Analysis { .. } => i += 1,
            }
        }
        let mut i = 0;
        while i < self.pending.len() {
            match self.pending[i].stage {
                Stage::Analysis { onset, ready } if ready <= pos => {
                    // INVARIANT: i < pending.len() was checked by the loop condition.
                    let p = self.pending.remove(i).expect("index in range");
                    self.analyze(p, onset, out);
                }
                Stage::Aic { .. } | Stage::Analysis { .. } => i += 1,
            }
        }
    }

    fn restart_grid(&mut self, at: u64) {
        self.origin = at;
        self.next_frame = 0;
        self.raw.reset(at);
        self.filtered.reset(at);
        for h in &mut self.hpf {
            h.reset();
        }
        self.first_flux_frame = 0;
        self.low_ratio_run = 0;
        self.started = true;
        self.restart = false;
    }

    /// Step 2 except the grid restart: rejects every pending candidate and resets the clock map.
    /// The noise estimate is kept.
    fn break_continuity(&mut self, out: &mut Vec<DetectorOutput>) {
        while let Some(p) = self.pending.pop_front() {
            out.push(DetectorOutput::Rejected(RejectedCandidate {
                id: p.id,
                trigger: SampleIndex(p.trigger),
                onset: p.onset(),
                reason: RejectReason::SpanCrossesDiscontinuity,
                gates: None,
            }));
        }
        self.clock.reset();
    }

    /// Band powers of [start, start + L) for every real channel into `channel_bands`.
    fn real_channel_bands(&mut self, start: u64) {
        for ri in 0..self.real.len() {
            self.filtered.copy(ri, start, &mut self.frame_buf);
            self.spectrum.compute(&self.frame_buf, &mut self.power);
            self.bands.apply(&self.power, &mut self.channel_bands[ri]);
        }
    }

    fn pooled_bands(&self) -> [f64; BAND_COUNT] {
        let mut pooled = [0.0; BAND_COUNT];
        for e in &self.channel_bands {
            for (p, v) in pooled.iter_mut().zip(e) {
                *p += v;
            }
        }
        pooled
    }

    /// Steps 7-12 for grid frame `next_frame`.
    fn process_frame(&mut self, out: &mut Vec<DetectorOutput>) {
        let m = self.next_frame;
        let hop = self.geometry.hop as u64;
        let start = self.origin + m * hop;
        self.next_frame += 1;
        self.real_channel_bands(start);
        let e = self.pooled_bands();
        let floor = self.params.noise_floor_power;
        let centre = start + self.geometry.len as u64 / 2;

        if let Tracker::Warming { done, total } = self.tracker {
            let k = (done + 1) as f64;
            for (n, v) in self.noise.pooled.iter_mut().zip(&e) {
                *n = (*n + (v - *n) / k).max(floor);
            }
            for (n, &pos) in self.noise.chosen.iter_mut().zip(&self.chosen_pos) {
                for (n, v) in n.iter_mut().zip(&self.channel_bands[pos]) {
                    *n = (*n + (v - *n) / k).max(floor);
                }
            }
            self.tracker = if done + 1 >= total {
                self.first_flux_frame = m + 1;
                Tracker::Tracking
            } else {
                Tracker::Warming {
                    done: done + 1,
                    total,
                }
            };
            self.record_health(false, false);
            self.set_telemetry(centre, &e, None, None);
            return;
        }

        // Step 9: whitened log flux against N(m−1).
        let mut y = [0.0; BAND_COUNT];
        for ((y, v), n) in y.iter_mut().zip(&e).zip(&self.noise.pooled) {
            *y = (1.0 + v / n.max(floor)).log10();
        }
        let r = self.params.flux_max_filter_radius;
        let mut yhat = [0.0; BAND_COUNT];
        for (b, slot) in yhat.iter_mut().enumerate() {
            let lo = b.saturating_sub(r);
            let hi = (b + r).min(BAND_COUNT - 1);
            *slot = y[lo..=hi].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        }
        let lag = self.params.flux_lag_frames as u64;
        let yhat_len = self.yhat.len() as u64;
        let defined_from = self.first_flux_frame + lag;
        let d = (m >= defined_from).then(|| {
            let prev = &self.yhat[((m - lag) % yhat_len) as usize];
            y.iter()
                .zip(prev)
                .map(|(y, p)| (y - p).max(0.0))
                .sum::<f64>()
                / BAND_COUNT as f64
        });
        self.yhat[(m % yhat_len) as usize] = yhat;

        // Step 10: adaptive peak-pick.
        let mut triggered = false;
        let mut threshold = None;
        if let Some(d) = d {
            let flux_len = self.flux.len() as u64;
            self.flux[(m % flux_len) as usize] = d;
            let means = self.params.peak_mean_frames as u64;
            if m >= defined_from + means {
                let mean = (m - means..m)
                    .map(|k| self.flux[(k % flux_len) as usize])
                    .sum::<f64>()
                    / means as f64;
                let thr = mean + self.params.peak_delta;
                threshold = Some(thr);
                let local = self.params.peak_local_max_frames as u64;
                let local_max = (m + 1 - local.min(m + 1)..=m)
                    .filter(|&k| k >= defined_from)
                    .map(|k| self.flux[(k % flux_len) as usize])
                    .fold(f64::NEG_INFINITY, f64::max);
                if d >= local_max && d >= thr {
                    triggered = self.try_trigger(start, centre, d, out);
                }
            }
        }

        // Step 11: gated EMA, freeze and recovery.
        let mut frozen = false;
        if self.tracker == Tracker::Tracking {
            frozen = self.frozen_at(start);
            if frozen {
                self.low_ratio_run = 0;
            } else {
                let ratio = e
                    .iter()
                    .zip(&self.noise.pooled)
                    .map(|(v, n)| v / n.max(floor))
                    .sum::<f64>()
                    / BAND_COUNT as f64;
                let alpha = self.alpha;
                for (n, v) in self.noise.pooled.iter_mut().zip(&e) {
                    *n = (*n + alpha * (v - *n)).max(floor);
                }
                for (n, &pos) in self.noise.chosen.iter_mut().zip(&self.chosen_pos) {
                    for (n, v) in n.iter_mut().zip(&self.channel_bands[pos]) {
                        *n = (*n + alpha * (v - *n)).max(floor);
                    }
                }
                if ratio < self.params.noise_recovery_ratio {
                    self.low_ratio_run += 1;
                } else {
                    self.low_ratio_run = 0;
                }
                if self.low_ratio_run >= self.params.noise_recovery_frames {
                    self.low_ratio_run = 0;
                    self.rewarms += 1;
                    self.tracker = Tracker::Warming {
                        done: 0,
                        total: self.warmup_frames,
                    };
                }
            }
        }
        self.prune_triggers(start);
        self.record_health(frozen, triggered);
        self.set_telemetry(centre, &e, d, threshold);
    }

    /// Spacing and supersede rules (step 10 c); creates the pending candidate on a trigger.
    fn try_trigger(
        &mut self,
        start: u64,
        centre: u64,
        d: f64,
        out: &mut Vec<DetectorOutput>,
    ) -> bool {
        let last = self
            .triggers
            .iter()
            .rev()
            .find(|t| !t.not_localized)
            .copied();
        if let Some(last) = last {
            if start.saturating_sub(last.frame_start) < self.spans.spacing {
                let pending = self.pending.iter().position(|p| p.id == last.id);
                match pending {
                    Some(i) if d >= self.params.supersede_ratio * last.flux => {
                        // INVARIANT: `i` was just found in `pending`.
                        let p = self.pending.remove(i).expect("index in range");
                        self.superseded += 1;
                        self.release(p.id, false);
                        out.push(DetectorOutput::Rejected(RejectedCandidate {
                            id: p.id,
                            trigger: SampleIndex(p.trigger),
                            onset: p.onset(),
                            reason: RejectReason::Superseded,
                            gates: None,
                        }));
                    }
                    Some(_) | None => return false,
                }
            }
        }
        if self.pending.len() >= MAX_PENDING {
            if let Some(p) = self.pending.pop_front() {
                self.pending_overflow += 1;
                out.push(DetectorOutput::Rejected(RejectedCandidate {
                    id: p.id,
                    trigger: SampleIndex(p.trigger),
                    onset: p.onset(),
                    reason: RejectReason::PendingOverflow,
                    gates: None,
                }));
            }
        }
        let id = CandidateId(self.next_id);
        self.next_id += 1;
        self.triggers.push_back(TriggerRecord {
            id,
            frame_start: start,
            freeze_end: centre + self.spans.freeze,
            flux: d,
            released: false,
            not_localized: false,
        });
        self.pending.push_back(Pending {
            id,
            trigger: centre,
            flux: d,
            noise: self.noise.clone(),
            stage: Stage::Aic {
                ready: centre + self.spans.aic_after,
            },
        });
        true
    }

    /// Frozen ⇔ the frame start ≤ the max of (t_c + noise_freeze) over unreleased triggers.
    fn frozen_at(&self, frame_start: u64) -> bool {
        self.triggers
            .iter()
            .any(|t| !t.released && frame_start <= t.freeze_end)
    }

    fn release(&mut self, id: CandidateId, not_localized: bool) {
        if let Some(t) = self.triggers.iter_mut().find(|t| t.id == id) {
            t.released = true;
            t.not_localized |= not_localized;
        }
    }

    /// Drops leading records that can no longer freeze, bound spacing or be superseded.
    fn prune_triggers(&mut self, frame_start: u64) {
        while let Some(t) = self.triggers.front() {
            let freeze_over = t.released || t.freeze_end < frame_start;
            let spacing_over = frame_start.saturating_sub(t.frame_start) >= self.spans.spacing;
            let pending = self.pending.iter().any(|p| p.id == t.id);
            if freeze_over && spacing_over && !pending {
                self.triggers.pop_front();
            } else {
                return;
            }
        }
    }

    fn record_health(&mut self, frozen: bool, triggered: bool) {
        let flags = u8::from(frozen) * FLAG_FROZEN + u8::from(triggered) * FLAG_TRIGGER;
        let len = self.health_flags.len();
        let slot = &mut self.health_flags[self.health_next];
        if self.health_filled == len {
            self.health_frozen -= usize::from(*slot & FLAG_FROZEN != 0);
            self.health_triggers -= usize::from(*slot & FLAG_TRIGGER != 0);
        } else {
            self.health_filled += 1;
        }
        *slot = flags;
        self.health_frozen += usize::from(frozen);
        self.health_triggers += usize::from(triggered);
        self.health_next = (self.health_next + 1) % len;
    }

    fn set_telemetry(
        &mut self,
        centre: u64,
        e: &[f64; BAND_COUNT],
        flux: Option<f64>,
        threshold: Option<f64>,
    ) {
        let db = |p: f64| (10.0 * p.max(TELEMETRY_POWER_FLOOR).log10()) as f32;
        let mut bands_db = [0.0f32; BAND_COUNT];
        let mut noise_db = [0.0f32; BAND_COUNT];
        for ((b, n), (v, nv)) in bands_db
            .iter_mut()
            .zip(noise_db.iter_mut())
            .zip(e.iter().zip(&self.noise.pooled))
        {
            *b = db(*v);
            *n = db(*nv);
        }
        // Flux and threshold are shown as 0 while they are undefined (warm-up, refill).
        self.last_frame = Some(FrameTelemetry {
            at: SampleIndex(centre),
            bands_db,
            noise_db,
            flux: flux.map_or(0.0, |v| v as f32),
            threshold: threshold.map_or(0.0, |v| v as f32),
        });
    }

    /// Step 13 for pending[i]. Returns false if the candidate was removed (rejected).
    fn refine_onset(&mut self, i: usize, out: &mut Vec<DetectorOutput>) -> bool {
        let (id, tc) = (self.pending[i].id, self.pending[i].trigger);
        let start = tc - self.spans.aic_before;
        let end = tc + self.spans.aic_after;
        let mut reject = |this: &mut Self, reason: RejectReason, onset: Option<SampleIndex>| {
            // INVARIANT: i < pending.len(), checked by the caller's loop.
            let p = this.pending.remove(i).expect("index in range");
            if reason == RejectReason::OnsetNotLocalized {
                this.release(id, true);
            }
            out.push(DetectorOutput::Rejected(RejectedCandidate {
                id,
                trigger: SampleIndex(p.trigger),
                onset,
                reason,
                gates: None,
            }));
            false
        };
        if !self.filtered.covers(start, end) {
            return reject(self, RejectReason::SpanCrossesDiscontinuity, None);
        }
        let a = self.chosen_pos[0];
        self.filtered.copy(a, start, &mut self.aic_x);
        let k = aic_pick(&self.aic_x, self.spans.aic_min, &mut self.aic_sums);
        let Some(k) = k else {
            return reject(self, RejectReason::OnsetNotLocalized, None);
        };
        let Some(onset) = (start + k as u64).checked_add_signed(self.jitter) else {
            return reject(self, RejectReason::SpanCrossesDiscontinuity, None);
        };
        let g = self.geometry;
        let anchored_end = onset
            + self.spans.anchor_offset
            + (ANCHORED_FRAMES as u64 - 1) * g.hop as u64
            + g.len as u64 / 2;
        // SPEC: emission waits for max(anchored end, t0 + SPAN_AFTER): `AnalysisWindow.span` and
        // the clip flag need HPF and raw samples to t0 + 40 ms, later than the anchored end
        // (t0 + 39.33 ms at 48 kHz).
        let mut ready = anchored_end.max(onset + self.spans.span_after);
        if self.clips {
            ready = ready.max(onset + self.spans.clip_post);
        }
        self.pending[i].stage = Stage::Analysis { onset, ready };
        true
    }

    /// Steps 14-16 for one candidate whose span is complete.
    fn analyze(&mut self, p: Pending, onset: u64, out: &mut Vec<DetectorOutput>) {
        let g = self.geometry;
        let (hop, half) = (g.hop as u64, g.len as u64 / 2);
        let first_centre = onset + self.spans.anchor_offset;
        let span_start = onset.checked_sub(self.spans.span_before);
        let span_end = onset + self.spans.span_after;
        let anchored_start = first_centre.checked_sub(half);
        let anchored_end = first_centre + (ANCHORED_FRAMES as u64 - 1) * hop + half;
        let clip_start = onset.checked_sub(self.spans.clip_pre);
        let covered = match (span_start, anchored_start) {
            (Some(s), Some(a)) => {
                self.filtered.covers(s, span_end)
                    && self.filtered.covers(a, anchored_end)
                    && self.raw.covers(s, span_end)
            }
            (None, _) | (_, None) => false,
        };
        let clip_covered = !self.clips
            || clip_start.is_some_and(|c| self.raw.covers(c, onset + self.spans.clip_post));
        let (Some(span_start), true, true) = (span_start, covered, clip_covered) else {
            // SPEC: a span or clip reaching back before the last grid restart (possible only for
            // CalibrationClips taps in the first 150 ms after a discontinuity) is rejected
            // SpanCrossesDiscontinuity rather than filled with samples from before the break.
            out.push(DetectorOutput::Rejected(RejectedCandidate {
                id: p.id,
                trigger: SampleIndex(p.trigger),
                onset: Some(SampleIndex(onset)),
                reason: RejectReason::SpanCrossesDiscontinuity,
                gates: None,
            }));
            return;
        };

        let mut pooled = [[0.0; BAND_COUNT]; ANCHORED_FRAMES];
        let mut chosen = vec![[[0.0; BAND_COUNT]; ANCHORED_FRAMES]; self.chosen_pos.len()];
        for (j, pooled_j) in pooled.iter_mut().enumerate() {
            self.real_channel_bands(first_centre + j as u64 * hop - half);
            *pooled_j = self.pooled_bands();
            for (c, &pos) in chosen.iter_mut().zip(&self.chosen_pos) {
                c[j] = self.channel_bands[pos];
            }
        }
        let level = self.params.clip_level;
        let channels = usize::from(self.format.channels.get());
        let clipped = (span_start..span_end)
            .any(|s| (0..channels).any(|ch| self.raw.get(ch, s).abs() >= level));
        let (gates, failure) =
            evaluate_gates(&self.params, &pooled, &p.noise.pooled, clipped, p.flux);

        if let Some(reason) = failure {
            if gates.snr < self.params.noise_release_snr {
                self.release(p.id, false);
            }
            out.push(DetectorOutput::Rejected(RejectedCandidate {
                id: p.id,
                trigger: SampleIndex(p.trigger),
                onset: Some(SampleIndex(onset)),
                reason,
                gates: Some(gates),
            }));
            return;
        }

        let span_len = (span_end - span_start) as usize;
        let span = self
            .chosen_pos
            .iter()
            .map(|&pos| {
                let mut s = vec![0.0f32; span_len];
                self.filtered.copy(pos, span_start, &mut s);
                s
            })
            .collect();
        let clip = clip_start.filter(|_| self.clips).map(|c| {
            let len = (onset + self.spans.clip_post - c) as usize;
            let mut samples = Vec::with_capacity(len * channels);
            for s in c..c + len as u64 {
                samples.extend((0..channels).map(|ch| self.raw.get(ch, s)));
            }
            RawClip {
                format: self.format,
                start: SampleIndex(c),
                samples,
                onset_offset: self.spans.clip_pre as usize,
            }
        });
        out.push(DetectorOutput::Candidate(Box::new(Candidate {
            id: p.id,
            trigger: SampleIndex(p.trigger),
            onset: SampleIndex(onset),
            onset_time: self.clock.host_time_of(SampleIndex(onset)),
            emitted_at: SampleIndex(self.raw.next - 1),
            gates,
            analysis: AnalysisWindow {
                rate: self.format.rate,
                geometry: g,
                layout: self.descriptor_layout,
                pooled,
                chosen,
                noise: p.noise,
                span,
                span_onset: self.spans.span_before as usize,
            },
            clip,
        })));
    }
}

/// AIC change point (step 13) of x: the argmin over k ∈ [k_min, M − k_min] of
/// k·ln max(var x[0..k), 1e-20) + (M − k − 1)·ln max(var x[k..M), 1e-20), ties to the lowest k.
/// None when the minimum sits on either edge (OnsetNotLocalized).
fn aic_pick(x: &[f32], k_min: usize, sums: &mut [(f64, f64)]) -> Option<usize> {
    let m = x.len();
    if k_min == 0 || 2 * k_min > m {
        return None;
    }
    sums[0] = (0.0, 0.0);
    for (i, &v) in x.iter().enumerate() {
        let v = f64::from(v);
        sums[i + 1] = (sums[i].0 + v, sums[i].1 + v * v);
    }
    let (s1, s2) = sums[m];
    let variance = |n: f64, a: f64, b: f64| (b / n - (a / n) * (a / n)).max(AIC_VARIANCE_FLOOR);
    let mut best = (f64::INFINITY, k_min);
    for (k, &(a, b)) in sums.iter().enumerate().take(m - k_min + 1).skip(k_min) {
        let (left, right) = (k as f64, (m - k) as f64);
        let aic =
            left * variance(left, a, b).ln() + (right - 1.0) * variance(right, s1 - a, s2 - b).ln();
        if aic < best.0 {
            best = (aic, k);
        }
    }
    let k = best.1;
    (k != k_min && k != m - k_min).then_some(k)
}

/// Step 15 on the anchored pooled frames against N* (floored at noise_floor_power). Returns the
/// report and the first failing gate, in the order LowSnr → NotBroadband → InsufficientBandSnr →
/// NotDecaying.
fn evaluate_gates(
    params: &DetectorParams,
    e: &[[f64; BAND_COUNT]; ANCHORED_FRAMES],
    noise: &[f64; BAND_COUNT],
    clipped: bool,
    trigger_flux: f64,
) -> (GateReport, Option<RejectReason>) {
    let floor = params.noise_floor_power;
    let db = |ratio: f64| 10.0 * ratio.max(GATE_RATIO_FLOOR).log10();
    let n = |b: usize| noise[b].max(floor);
    let bands = BAND_COUNT as f64;

    let excess = (0..BAND_COUNT)
        .map(|b| (e[0][b] + e[1][b]) / 2.0 / n(b))
        .sum::<f64>()
        / bands
        - 1.0;
    let snr = 10.0 * excess.max(SNR_EXCESS_FLOOR).log10();
    let broadband = (0..BAND_COUNT)
        .filter(|&b| {
            let peak = e[..3].iter().map(|f| f[b]).fold(0.0, f64::max);
            db(peak / n(b)) >= params.broadband_rise.get()
        })
        .count() as f64
        / bands;
    let valid: Vec<usize> = (0..BAND_COUNT)
        .filter(|&b| {
            let peak = e.iter().map(|f| f[b]).fold(0.0, f64::max);
            db(peak / n(b)) >= params.decay_valid_band_snr.get()
        })
        .collect();
    let enough = valid.len() >= params.decay_min_valid_bands;
    let decay = if enough {
        valid
            .iter()
            .filter(|&&b| {
                let head = e[0][b] + e[1][b];
                let tail = (e[5][b] + e[6][b]).max(GATE_RATIO_FLOOR);
                db(head / tail) >= params.decay_drop.get()
            })
            .count() as f64
            / valid.len() as f64
    } else {
        0.0
    };
    let report = GateReport {
        // INVARIANT: the excess is floored at 1e-6, so its log is finite.
        snr: Decibels::try_new(snr).expect("floored snr is finite"),
        broadband_fraction: broadband,
        valid_bands: valid.len() as u8,
        decay_fraction: decay,
        clipped,
        trigger_flux,
    };
    let failure = if snr < params.snr_gate.get() {
        Some(RejectReason::LowSnr)
    } else if broadband < params.broadband_fraction {
        Some(RejectReason::NotBroadband)
    } else if !enough {
        Some(RejectReason::InsufficientBandSnr)
    } else if decay < params.decay_fraction {
        Some(RejectReason::NotDecaying)
    } else {
        None
    };
    (report, failure)
}

/// Ambient-step estimate: mean grid-frame band power over frames that overlap no `exclude` range
/// (frame indices relative to the buffer start). Requires >= 100 usable frames.
// SPEC: `exclude` holds half-open sample-frame ranges [start, end) relative to the buffer start;
// a grid frame [s, s + L) is unusable if it overlaps one. A frame holding a non-finite sample is
// unusable too (that channel's high-pass restarts after it), since the live path drops such
// blocks (step 3).
pub fn estimate_noise(
    format: AudioFormat,
    layout: &ChannelLayout,
    interleaved: &[f32],
    exclude: &[(u64, u64)],
) -> Result<NoiseSnapshot, DetectorError> {
    let channels = format.channels.get();
    if let Some(&channel) = layout.real().iter().find(|&&c| c >= channels) {
        return Err(DetectorError::LayoutChannelOutOfRange { channel, channels });
    }
    let params = DetectorParams::DEFAULT;
    let rate = format.rate;
    let geometry = FrameGeometry::for_rate(rate);
    let n = usize::from(channels);
    let frames = interleaved.len() / n;
    let real: Vec<usize> = layout.real().iter().map(|&c| usize::from(c)).collect();
    let chosen: Vec<usize> = DescriptorLayout::from_channels(layout)
        .chosen()
        .iter()
        .map(|c| {
            // INVARIANT: DescriptorLayout::from_channels picks its channels from `real`.
            layout
                .real()
                .iter()
                .position(|r| r == c)
                .expect("chosen channels are real")
        })
        .collect();
    // INVARIANT: 100 Hz is below Nyquist at every valid rate (fs ≥ 16 kHz).
    let hpf = Biquad::highpass(rate, params.hpf_cutoff).expect("100 Hz < fs/2");
    let mut filtered = vec![vec![0.0f32; frames]; real.len()];
    let mut bad = Vec::new();
    for (ri, &ch) in real.iter().enumerate() {
        let mut h = hpf.clone();
        for (s, out) in filtered[ri].iter_mut().enumerate() {
            let x = interleaved[s * n + ch];
            if x.is_finite() {
                *out = h.process(x);
            } else {
                h.reset();
                bad.push(s as u64);
            }
        }
    }
    bad.sort_unstable();
    let (len, hop) = (geometry.len, geometry.hop);
    let mut spectrum = PowerSpectrum::new(geometry);
    let bands = BandLayout::new(rate, geometry);
    let mut power = vec![0.0; geometry.fft_len / 2 + 1];
    let mut channel_bands = vec![[0.0; BAND_COUNT]; real.len()];
    let mut pooled = [0.0; BAND_COUNT];
    let mut chosen_sum = vec![[0.0; BAND_COUNT]; chosen.len()];
    let mut usable = 0u64;
    let mut s = 0usize;
    while s + len <= frames {
        let (a, b) = (s as u64, (s + len) as u64);
        let excluded = exclude.iter().any(|&(x0, x1)| x0 < b && a < x1);
        let first_bad = bad.partition_point(|&i| i < a);
        let has_bad = bad.get(first_bad).is_some_and(|&i| i < b);
        if !excluded && !has_bad {
            usable += 1;
            for (ri, f) in filtered.iter().enumerate() {
                spectrum.compute(&f[s..s + len], &mut power);
                bands.apply(&power, &mut channel_bands[ri]);
            }
            for e in &channel_bands {
                for (p, v) in pooled.iter_mut().zip(e) {
                    *p += v;
                }
            }
            for (c, &pos) in chosen_sum.iter_mut().zip(&chosen) {
                for (p, v) in c.iter_mut().zip(&channel_bands[pos]) {
                    *p += v;
                }
            }
        }
        s += hop;
    }
    if usable < MIN_AMBIENT_FRAMES {
        return Err(DetectorError::InsufficientAmbient {
            frames: usable,
            needed: MIN_AMBIENT_FRAMES,
        });
    }
    let count = usable as f64;
    let floor = params.noise_floor_power;
    let mean = |sum: [f64; BAND_COUNT]| sum.map(|v| (v / count).max(floor));
    Ok(NoiseSnapshot {
        pooled: mean(pooled),
        chosen: chosen_sum.into_iter().map(mean).collect(),
    })
}

/// Signal generators and helpers shared by the detector and descriptor tests.
#[cfg(test)]
pub(crate) mod testkit {
    use super::*;
    use crate::audio::BlockTime;
    use crate::rng::Rng64;
    use crate::units::ChannelCount;
    use std::f64::consts::TAU;

    /// −60 dBFS.
    pub(crate) const NOISE_RMS: f64 = 1e-3;

    pub(crate) fn rate(hz: u32) -> SampleRate {
        SampleRate::new(hz).expect("valid rate")
    }

    pub(crate) fn format(hz: u32, channels: u16) -> AudioFormat {
        AudioFormat {
            rate: rate(hz),
            channels: ChannelCount::new(channels).expect("valid channel count"),
        }
    }

    pub(crate) fn all_real(channels: u16) -> ChannelLayout {
        ChannelLayout::all_real(ChannelCount::new(channels).expect("valid channel count"))
    }

    pub(crate) fn samples(hz: u32, ms: f64) -> usize {
        rate(hz).samples_in(Milliseconds::new(ms))
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum Colour {
        White,
        Pink,
    }

    /// Streaming Gaussian noise, white or pink (Kellet's refined filter), scaled to `rms`.
    pub(crate) struct NoiseGen {
        rng: Rng64,
        colour: Colour,
        b: [f64; 7],
        scale: f64,
    }

    impl NoiseGen {
        pub(crate) fn new(seed: u64, colour: Colour, rms: f64) -> Self {
            let mut probe = NoiseGen {
                rng: Rng64::new(seed ^ 0x9E37_79B9),
                colour,
                b: [0.0; 7],
                scale: 1.0,
            };
            let n = 1 << 16;
            let ms = (0..n)
                .map(|_| {
                    let v = probe.raw();
                    v * v
                })
                .sum::<f64>()
                / f64::from(n);
            NoiseGen {
                rng: Rng64::new(seed),
                colour,
                b: [0.0; 7],
                scale: rms / ms.sqrt(),
            }
        }

        fn raw(&mut self) -> f64 {
            let w = self.rng.normal();
            match self.colour {
                Colour::White => w,
                Colour::Pink => {
                    let b = &mut self.b;
                    b[0] = 0.99886 * b[0] + w * 0.055_517_9;
                    b[1] = 0.99332 * b[1] + w * 0.075_075_9;
                    b[2] = 0.96900 * b[2] + w * 0.153_852;
                    b[3] = 0.86650 * b[3] + w * 0.310_485_6;
                    b[4] = 0.55000 * b[4] + w * 0.532_952_2;
                    b[5] = -0.7616 * b[5] - w * 0.016_898;
                    let out = b.iter().sum::<f64>() + w * 0.5362;
                    b[6] = w * 0.115_926;
                    out
                }
            }
        }

        pub(crate) fn sample(&mut self) -> f64 {
            self.raw() * self.scale
        }

        pub(crate) fn fill(&mut self, n: usize) -> Vec<f64> {
            (0..n).map(|_| self.sample()).collect()
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum TapKind {
        Partials,
        Click,
        Thud,
    }

    fn add_partials(
        x: &mut [f64],
        fs: f64,
        rng: &mut Rng64,
        count: usize,
        band: (f64, f64),
        tau_ms: (f64, f64),
    ) {
        for _ in 0..count {
            let f = rng.range(band.0, band.1);
            let tau = rng.range(tau_ms.0, tau_ms.1) * 1e-3 * fs;
            let amp = rng.range(0.3, 1.0);
            let phase = rng.range(0.0, TAU);
            for (i, v) in x.iter_mut().enumerate() {
                let t = i as f64;
                *v += amp * (-t / tau).exp() * (TAU * f * t / fs + phase).sin();
            }
        }
    }

    fn add_noise_burst(x: &mut [f64], fs: f64, rng: &mut Rng64, amp: f64, tau_ms: f64) {
        let tau = tau_ms * 1e-3 * fs;
        for (i, v) in x.iter_mut().enumerate() {
            *v += amp * rng.normal() * (-(i as f64) / tau).exp();
        }
    }

    /// A unit-peak tap, 100 ms long, whose onset is index 0.
    pub(crate) fn tap(kind: TapKind, hz: u32, rng: &mut Rng64) -> Vec<f64> {
        let fs = f64::from(hz);
        let mut x = vec![0.0; (0.1 * fs) as usize];
        match kind {
            TapKind::Partials => add_partials(
                &mut x,
                fs,
                rng,
                6,
                (300.0, (0.4 * fs).min(8000.0)),
                (3.0, 12.0),
            ),
            TapKind::Click => {
                let tau = rng.range(1.0, 3.0);
                add_noise_burst(&mut x, fs, rng, 1.0, tau);
            }
            TapKind::Thud => {
                add_partials(&mut x, fs, rng, 3, (120.0, 600.0), (8.0, 20.0));
                add_noise_burst(&mut x, fs, rng, 0.5, 0.5);
            }
        }
        let peak = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        x.iter_mut().for_each(|v| *v /= peak);
        x
    }

    /// x[at..] += gain · shape (truncated at the end of x).
    pub(crate) fn add(x: &mut [f64], at: usize, shape: &[f64], gain: f64) {
        for (v, s) in x[at..].iter_mut().zip(shape) {
            *v += gain * s;
        }
    }

    pub(crate) fn to_f32(x: &[f64]) -> Vec<f32> {
        x.iter().map(|&v| v as f32).collect()
    }

    /// Peak amplitude of a tap at `snr_db` above the −60 dBFS noise rms.
    pub(crate) fn amp(snr_db: f64) -> f64 {
        NOISE_RMS * 10f64.powf(snr_db / 20.0)
    }

    /// Pushes interleaved `x`, whose frame 0 is sample `start`, in untimed blocks of `block` frames.
    pub(crate) fn push_all(
        det: &mut Detector,
        fmt: AudioFormat,
        x: &[f32],
        start: u64,
        block: usize,
        out: &mut Vec<DetectorOutput>,
    ) {
        let ch = usize::from(fmt.channels.get());
        for (i, chunk) in x.chunks(block * ch).enumerate() {
            let b = AudioBlock::new(
                fmt,
                SampleIndex(start + (i * block) as u64),
                chunk,
                BlockTime::Untimed,
                false,
            )
            .expect("whole frames");
            det.push(&b, out).expect("matching format");
        }
    }

    pub(crate) fn candidates(out: &[DetectorOutput]) -> Vec<&Candidate> {
        out.iter()
            .filter_map(|o| match o {
                DetectorOutput::Candidate(c) => Some(&**c),
                DetectorOutput::Rejected(_) | DetectorOutput::InputFault(_) => None,
            })
            .collect()
    }

    pub(crate) fn rejections(out: &[DetectorOutput]) -> Vec<&RejectedCandidate> {
        out.iter()
            .filter_map(|o| match o {
                DetectorOutput::Rejected(r) => Some(r),
                DetectorOutput::Candidate(_) | DetectorOutput::InputFault(_) => None,
            })
            .collect()
    }

    /// Mono 100 Hz high-pass from rest, as the detector computes it.
    pub(crate) fn highpass(hz: u32, x: &[f32]) -> Vec<f32> {
        let mut h =
            Biquad::highpass(rate(hz), DetectorParams::DEFAULT.hpf_cutoff).expect("valid hpf");
        x.iter().map(|&v| h.process(v)).collect()
    }

    /// Step 14's band powers of a mono HPF signal, anchored at `t0`.
    pub(crate) fn anchored(
        hz: u32,
        hpf: &[f32],
        t0: usize,
    ) -> [[f64; BAND_COUNT]; ANCHORED_FRAMES] {
        let r = rate(hz);
        let g = FrameGeometry::for_rate(r);
        let mut spectrum = PowerSpectrum::new(g);
        let bands = BandLayout::new(r, g);
        let mut power = vec![0.0; g.fft_len / 2 + 1];
        let first = t0 + r.samples_in(DetectorParams::DEFAULT.anchor_offset);
        let mut out = [[0.0; BAND_COUNT]; ANCHORED_FRAMES];
        for (j, o) in out.iter_mut().enumerate() {
            let s = first + j * g.hop - g.len / 2;
            spectrum.compute(&hpf[s..s + g.len], &mut power);
            bands.apply(&power, o);
        }
        out
    }

    /// A Live mono detector at `hz` with `params`.
    pub(crate) fn live(hz: u32, params: &DetectorParams) -> Detector {
        Detector::new(format(hz, 1), &all_real(1), params, DetectorMode::Live)
            .expect("valid detector")
    }

    /// `channels` channels of independent −60 dBFS white noise, `len_ms` long, with each
    /// (onset ms, kind, peak SNR dB) tap added to every channel (channel c at gain 1 − 0.2·c).
    /// Returns the interleaved signal and the tap onsets in samples.
    pub(crate) fn scene(
        hz: u32,
        channels: u16,
        seed: u64,
        len_ms: f64,
        events: &[(f64, TapKind, f64)],
    ) -> (Vec<f32>, Vec<usize>) {
        let n = samples(hz, len_ms);
        let mut rng = Rng64::new(seed);
        let mut chans: Vec<Vec<f64>> = (0..channels)
            .map(|c| NoiseGen::new(seed + 1 + u64::from(c), Colour::White, NOISE_RMS).fill(n))
            .collect();
        let mut onsets = Vec::new();
        for &(at, kind, snr) in events {
            let t0 = samples(hz, at);
            let shape = tap(kind, hz, &mut rng);
            for (c, ch) in chans.iter_mut().enumerate() {
                add(ch, t0, &shape, amp(snr) * (1.0 - 0.2 * c as f64));
            }
            onsets.push(t0);
        }
        let x = (0..n)
            .flat_map(|s| chans.iter().map(move |ch| ch[s] as f32))
            .collect();
        (x, onsets)
    }
}

#[cfg(test)]
mod tests {
    use super::testkit::*;
    use super::*;
    use crate::audio::BlockTime;
    use crate::dsp::TABLE_RATES;
    use crate::rng::Rng64;

    const P: DetectorParams = DetectorParams::DEFAULT;

    /// One frame's telemetry, the pooled N after it and the re-warm count.
    type FrameRow = (FrameTelemetry, [f64; BAND_COUNT], u64);
    /// A test-signal generator: (rate, seed, length) -> samples.
    type Generator = fn(u32, u64, usize) -> Vec<f64>;

    fn ms(hz: u32, samples: i64) -> f64 {
        samples as f64 * 1000.0 / f64::from(hz)
    }

    /// Per-frame telemetry and pooled N of a mono stream pushed one hop at a time (the grid
    /// starts at sample 0, so every push after the first completes exactly one frame).
    fn per_frame(
        det: &mut Detector,
        hz: u32,
        x: &[f32],
        out: &mut Vec<DetectorOutput>,
    ) -> Vec<FrameRow> {
        let hop = FrameGeometry::for_rate(rate(hz)).hop;
        let fmt = format(hz, 1);
        let mut frames: Vec<FrameRow> = Vec::new();
        for (i, chunk) in x.chunks(hop).enumerate() {
            push_all(det, fmt, chunk, (i * hop) as u64, hop, out);
            if let Some(f) = det.last_frame() {
                if frames.last().is_none_or(|l| l.0.at != f.at) {
                    frames.push((f, det.noise().pooled, det.health().rewarms));
                }
            }
        }
        frames
    }

    #[test]
    fn t1_stationary_noise_has_no_gate_passing_candidates() {
        std::thread::scope(|s| {
            for hz in TABLE_RATES {
                s.spawn(move || {
                    for (k, colour) in [Colour::White, Colour::Pink].into_iter().enumerate() {
                        let seed = 0x7100 + 2 * u64::from(hz) + k as u64;
                        let mut gen = NoiseGen::new(seed, colour, NOISE_RMS);
                        let mut det = live(hz, &P);
                        let fmt = format(hz, 1);
                        let seconds = 120;
                        let total = hz as usize * seconds;
                        let block = 4096;
                        let mut buf = vec![0.0f32; block];
                        let mut out = Vec::new();
                        let (mut triggers, mut passing) = (0usize, 0usize);
                        let mut pos = 0;
                        while pos < total {
                            let n = block.min(total - pos);
                            for v in &mut buf[..n] {
                                *v = gen.sample() as f32;
                            }
                            push_all(&mut det, fmt, &buf[..n], pos as u64, block, &mut out);
                            for o in out.drain(..) {
                                match o {
                                    DetectorOutput::Candidate(_) => {
                                        triggers += 1;
                                        passing += 1;
                                    }
                                    DetectorOutput::Rejected(_) => triggers += 1,
                                    DetectorOutput::InputFault(f) => panic!("{hz}: fault {f:?}"),
                                }
                            }
                            pos += n;
                        }
                        let per_second = triggers as f64 / seconds as f64;
                        println!(
                            "T1 {hz:>6} {colour:?}: {per_second:.3} triggers/s, {passing} gate-passing"
                        );
                        assert!(per_second <= 0.15, "{hz} {colour:?}: {per_second} triggers/s");
                        assert_eq!(passing, 0, "{hz} {colour:?}: gate-passing candidates");
                    }
                });
            }
        });
    }

    fn t2_rate(hz: u32, seed: u64) {
        let g = FrameGeometry::for_rate(rate(hz));
        let mut rng = Rng64::new(seed);
        let spacing = samples(hz, 400.0);
        let lead = samples(hz, 1000.0);
        let mut kinds: Vec<TapKind> = [TapKind::Partials, TapKind::Click, TapKind::Thud]
            .into_iter()
            .flat_map(|k| std::iter::repeat_n(k, 60))
            .collect();
        for i in (1..kinds.len()).rev() {
            let j = rng.below(i as u64 + 1) as usize;
            kinds.swap(i, j);
        }
        let n = lead + (kinds.len() + 1) * spacing;
        let mut x = NoiseGen::new(seed, Colour::White, NOISE_RMS).fill(n);
        let mut taps = Vec::new();
        for (i, &kind) in kinds.iter().enumerate() {
            let t0 = lead + i * spacing + rng.below(g.hop as u64) as usize;
            let snr = rng.range(10.0, 55.0);
            let shape = tap(kind, hz, &mut rng);
            add(&mut x, t0, &shape, amp(snr));
            taps.push((t0, kind, snr));
        }
        let x = to_f32(&x);
        let hpf = highpass(hz, &x);
        let fmt = format(hz, 1);
        let mut det = live(hz, &P);
        let mut out = Vec::new();
        let mut noise_at = Vec::new();
        let mut pos = 0;
        for &(t0, _, _) in &taps {
            push_all(&mut det, fmt, &x[pos..t0], pos as u64, 480, &mut out);
            noise_at.push(det.noise().pooled);
            pos = t0;
        }
        push_all(&mut det, fmt, &x[pos..], pos as u64, 480, &mut out);
        let cands = candidates(&out);
        let rejected = rejections(&out);
        let (before, after) = (samples(hz, 20.0), samples(hz, 200.0));
        let (mut eligible, mut close) = (0usize, 0usize);
        for (&(t0, kind, snr), noise) in taps.iter().zip(&noise_at) {
            let (gates, _) = evaluate_gates(&P, &anchored(hz, &hpf, t0), noise, false, 0.0);
            let margin = gates.snr.get() >= P.snr_gate.get() + 1.0
                && gates.broadband_fraction >= P.broadband_fraction + 0.05
                && gates.decay_fraction >= P.decay_fraction + 0.05
                && usize::from(gates.valid_bands) > P.decay_min_valid_bands;
            if !margin {
                continue;
            }
            eligible += 1;
            let near = |s: SampleIndex| (t0 - before..t0 + after).contains(&(s.0 as usize));
            let found: Vec<_> = cands.iter().filter(|c| near(c.onset)).collect();
            assert_eq!(
                found.len(),
                1,
                "{hz}: {kind:?} tap at {t0} ({snr:.1} dB) gave {} candidates",
                found.len()
            );
            let err = ms(hz, found[0].onset.0 as i64 - t0 as i64);
            assert!(
                err.abs() <= 2.0,
                "{hz}: {kind:?} tap at {t0} ({snr:.1} dB): t0 error {err} ms"
            );
            close += usize::from(err.abs() <= 0.5);
            assert!(
                !rejected
                    .iter()
                    .any(|r| r.reason == RejectReason::OnsetNotLocalized && near(r.trigger)),
                "{hz}: {kind:?} tap at {t0} ({snr:.1} dB) gave OnsetNotLocalized"
            );
        }
        println!(
            "T2 {hz}: {eligible} of {} taps pass the oracle with margin, {close} within 0.5 ms",
            taps.len()
        );
        assert!(eligible >= 60, "{hz}: only {eligible} oracle-passing taps");
        assert!(
            close * 100 >= eligible * 95,
            "{hz}: {close}/{eligible} within 0.5 ms"
        );
    }

    #[test]
    fn t2_taps_in_white_noise_localize() {
        std::thread::scope(|s| {
            s.spawn(|| t2_rate(48_000, 0x7248));
            s.spawn(|| t2_rate(16_000, 0x7216));
        });
    }

    /// 3 s at `hz`: three taps of different strengths, and a weak click 10 ms before a strong tap.
    fn mixed_scene(hz: u32, channels: u16) -> Vec<f32> {
        let events = [
            (500.0, TapKind::Partials, 40.0),
            (1200.0, TapKind::Click, 25.0),
            (1800.0, TapKind::Click, 22.0),
            (1810.0, TapKind::Partials, 50.0),
            (2400.0, TapKind::Thud, 35.0),
        ];
        scene(hz, channels, 0x7404, 3000.0, &events).0
    }

    fn run_debug(hz: u32, channels: u16, x: &[f32], mode: &DetectorMode, block: usize) -> String {
        let fmt = format(hz, channels);
        let mut det = Detector::new(fmt, &all_real(channels), &P, mode.clone()).expect("detector");
        let mut out = Vec::new();
        push_all(&mut det, fmt, x, 0, block, &mut out);
        assert!(candidates(&out).len() >= 3, "scene too weak: {out:?}");
        format!(
            "{out:?}\n{:?}\n{:?}\n{:?}\n{:?}",
            det.health(),
            det.noise(),
            det.last_frame(),
            det.state()
        )
    }

    #[test]
    fn t4_block_size_does_not_change_outputs() {
        let x = mixed_scene(48_000, 1);
        for mode in [DetectorMode::Live, DetectorMode::CalibrationClips] {
            let reference = run_debug(48_000, 1, &x, &mode, 4096);
            for block in [1, 7, 480] {
                assert!(
                    run_debug(48_000, 1, &x, &mode, block) == reference,
                    "{mode:?}: block size {block} differs from 4096"
                );
            }
        }
    }

    fn fade(t: f64, start: f64, end: f64, ramp: f64) -> f64 {
        if t < start || t >= end {
            0.0
        } else if t < start + ramp {
            0.5 - 0.5 * (std::f64::consts::PI * (t - start) / ramp).cos()
        } else if t > end - ramp {
            0.5 - 0.5 * (std::f64::consts::PI * (end - t) / ramp).cos()
        } else {
            1.0
        }
    }

    fn tones(hz: u32, seed: u64, n: usize) -> Vec<f64> {
        let fs = f64::from(hz);
        let mut rng = Rng64::new(seed);
        let mut x = NoiseGen::new(seed, Colour::White, NOISE_RMS).fill(n);
        let mut t = 0.3;
        while t < n as f64 / fs - 2.0 {
            let dur = rng.range(0.4, 1.5);
            let notes = 1 + rng.below(4) as usize;
            for _ in 0..notes {
                let f = rng.range(200.0, 4000.0);
                let a = 0.1 / notes as f64;
                let phase = rng.range(0.0, std::f64::consts::TAU);
                let (s0, s1) = ((t * fs) as usize, ((t + dur) * fs) as usize);
                for (i, v) in x[s0..s1].iter_mut().enumerate() {
                    let tt = (s0 + i) as f64 / fs;
                    let w = std::f64::consts::TAU * f * tt + phase;
                    *v += a * fade(tt, t, t + dur, 0.02) * w.sin();
                }
            }
            t += dur + rng.range(0.2, 0.8);
        }
        x
    }

    fn speech(hz: u32, seed: u64, n: usize) -> Vec<f64> {
        let fs = f64::from(hz);
        let mut x = NoiseGen::new(seed, Colour::White, NOISE_RMS).fill(n);
        let tau = std::f64::consts::TAU;
        let formant = |f: f64| {
            1.0 + 2.0 * (-((f - 700.0) / 150.0).powi(2)).exp()
                + 1.5 * (-((f - 1200.0) / 200.0).powi(2)).exp()
        };
        let mut phase = 0.0;
        for (i, v) in x.iter_mut().enumerate() {
            let t = i as f64 / fs;
            let f0 = 140.0 * (1.0 + 0.03 * (tau * 5.0 * t).sin()) + 20.0 * (tau * 0.3 * t).sin();
            phase += tau * f0 / fs;
            let env = (std::f64::consts::PI * 4.0 * t).sin().powi(2);
            let mut s = 0.0;
            let mut k = 1.0;
            while k * f0 < 4000.0 {
                s += formant(k * f0) / k * (k * phase).sin();
                k += 1.0;
            }
            *v += 0.02 * env * s;
        }
        x
    }

    fn noise_steps(hz: u32, seed: u64, n: usize) -> Vec<f64> {
        let fs = f64::from(hz);
        let mut rng = Rng64::new(seed);
        let mut gen = NoiseGen::new(seed, Colour::Pink, 0.01);
        let ramp = 0.05 * fs;
        let (mut level, mut from, mut at) = (0.0f64, 0.0f64, 0.0f64);
        let mut next = rng.range(1.5, 3.0) * fs;
        (0..n)
            .map(|i| {
                let i = i as f64;
                if i >= next {
                    from = level;
                    level = if level == 0.0 { 10.0 } else { 0.0 };
                    at = next;
                    next += rng.range(1.5, 3.0) * fs;
                }
                let db = from + (level - from) * ((i - at) / ramp).min(1.0);
                gen.sample() * 10f64.powf(db / 20.0)
            })
            .collect()
    }

    #[test]
    fn t5_tones_speech_and_noise_steps_never_pass_the_gates() {
        let hz = 48_000;
        let n = samples(hz, 60_000.0);
        let signals: [(&str, Generator); 3] =
            [("tones", tones), ("speech", speech), ("steps", noise_steps)];
        std::thread::scope(|s| {
            for (k, (name, make)) in signals.into_iter().enumerate() {
                s.spawn(move || {
                    let x = to_f32(&make(hz, 0x7500 + k as u64, n));
                    let mut det = live(hz, &P);
                    let mut out = Vec::new();
                    push_all(&mut det, format(hz, 1), &x, 0, 480, &mut out);
                    let rejected = rejections(&out);
                    let mut reasons: Vec<(RejectReason, usize)> = Vec::new();
                    for r in &rejected {
                        match reasons.iter_mut().find(|(k, _)| *k == r.reason) {
                            Some((_, n)) => *n += 1,
                            None => reasons.push((r.reason, 1)),
                        }
                    }
                    println!("T5 {name}: {} rejections {reasons:?}", rejected.len());
                    assert_eq!(candidates(&out).len(), 0, "{name}: gate-passing candidates");
                });
            }
        });
    }

    /// Seconds after `step` until every band of N is within 1 dB of the mean band power of
    /// [step, end) and stays within it for 1 s; infinity if it never does.
    fn settle_seconds(hz: u32, frames: &[FrameRow], step: usize, end: usize) -> f64 {
        let half = FrameGeometry::for_rate(rate(hz)).len / 2;
        let inside: Vec<&FrameTelemetry> = frames
            .iter()
            .map(|f| &f.0)
            .filter(|f| f.at.0 as usize >= step + half && f.at.0 as usize + half <= end)
            .collect();
        let mut mean = [0.0f64; BAND_COUNT];
        for f in &inside {
            for (m, db) in mean.iter_mut().zip(&f.bands_db) {
                *m += 10f64.powf(f64::from(*db) / 10.0);
            }
        }
        let reference = mean.map(|m| 10.0 * (m / inside.len() as f64).log10());
        let after: Vec<(usize, bool)> = frames
            .iter()
            .map(|f| &f.0)
            .filter(|f| (step..end).contains(&(f.at.0 as usize)))
            .map(|f| {
                let ok = f
                    .noise_db
                    .iter()
                    .zip(&reference)
                    .all(|(n, r)| (f64::from(*n) - r).abs() <= 1.0);
                (f.at.0 as usize, ok)
            })
            .collect();
        let hold = samples(hz, 1000.0);
        for (i, &(at, _)) in after.iter().enumerate() {
            let window: Vec<bool> = after[i..]
                .iter()
                .take_while(|(a, _)| *a <= at + hold)
                .map(|&(_, ok)| ok)
                .collect();
            if after.last().is_some_and(|&(last, _)| last >= at + hold)
                && window.iter().all(|&ok| ok)
            {
                return (at - step) as f64 / f64::from(hz);
            }
        }
        f64::INFINITY
    }

    /// The weak-click-then-strong-tap scene of T8: (signal, weak onset, strong onset).
    fn supersede_scene() -> (Vec<f32>, usize, usize) {
        let (x, onsets) = scene(
            48_000,
            1,
            0x7808,
            1500.0,
            &[
                (1000.0, TapKind::Click, 22.0),
                (1010.0, TapKind::Partials, 50.0),
            ],
        );
        (x, onsets[0], onsets[1])
    }

    fn released(det: &Detector, id: CandidateId) -> bool {
        det.triggers.iter().all(|t| t.id != id || t.released)
    }

    #[test]
    fn t6_noise_tracker_steps_freeze_and_release() {
        // Step tracking: +12 dB for 7 s, then back down 12 dB for 3 s.
        let mut jobs = Vec::new();
        for colour in [Colour::White, Colour::Pink] {
            for hz in [48_000u32, 16_000] {
                for seed in 0..8u64 {
                    jobs.push((colour, hz, seed));
                }
            }
        }
        let results: Vec<(f64, f64)> = std::thread::scope(|s| {
            let handles: Vec<_> = jobs
                .iter()
                .map(|&(colour, hz, seed)| {
                    s.spawn(move || {
                        let mut gen = NoiseGen::new(0x7600 + seed, colour, NOISE_RMS);
                        let (up, down, end) = (
                            samples(hz, 3000.0),
                            samples(hz, 10_000.0),
                            samples(hz, 13_000.0),
                        );
                        let gain = 10f64.powf(12.0 / 20.0);
                        let x: Vec<f32> = (0..end)
                            .map(|i| {
                                let g = if (up..down).contains(&i) { gain } else { 1.0 };
                                (gen.sample() * g) as f32
                            })
                            .collect();
                        let mut det = live(hz, &P);
                        let frames = per_frame(&mut det, hz, &x, &mut Vec::new());
                        (
                            settle_seconds(hz, &frames, up, down),
                            settle_seconds(hz, &frames, down, end),
                        )
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("worker"))
                .collect()
        });
        for (&(colour, hz, seed), &(up, down)) in jobs.iter().zip(&results) {
            assert!(
                up <= 5.0,
                "{colour:?} {hz} seed {seed}: +12 dB settled after {up} s"
            );
            assert!(
                down <= 1.5,
                "{colour:?} {hz} seed {seed}: −12 dB settled after {down} s"
            );
        }
        let worst = |f: fn(&(f64, f64)) -> f64| results.iter().map(f).fold(0.0, f64::max);
        println!(
            "T6 worst settle: +12 dB {:.2} s, −12 dB {:.2} s",
            worst(|r| r.0),
            worst(|r| r.1)
        );

        // A gate-passing candidate freezes N in both directions for 300 ms.
        let hz = 48_000;
        let (x, onsets) = scene(hz, 1, 0x7606, 2500.0, &[(1500.0, TapKind::Partials, 40.0)]);
        let mut det = live(hz, &P);
        let mut out = Vec::new();
        let frames = per_frame(&mut det, hz, &x, &mut out);
        let cands = candidates(&out);
        assert_eq!(cands.len(), 1, "{out:?}");
        assert!((cands[0].onset.0 as i64 - onsets[0] as i64).abs() <= samples(hz, 2.0) as i64);
        let tc = cands[0].trigger.0;
        let half = FrameGeometry::for_rate(rate(hz)).len as u64 / 2;
        let freeze_end = tc + samples(hz, 300.0) as u64;
        let m = frames
            .iter()
            .position(|f| f.0.at.0 == tc)
            .expect("trigger frame");
        let before = frames[m - 1].1;
        for f in &frames[m..] {
            let start = f.0.at.0 - half;
            if start <= freeze_end {
                assert_eq!(
                    f.1, before,
                    "N changed at frame start {start} inside the freeze"
                );
            } else {
                assert_ne!(f.1, before, "N still frozen at frame start {start}");
                break;
            }
        }

        // Noise-only (snr < 0 dB) and OnsetNotLocalized rejections release their freeze.
        let params = DetectorParams {
            peak_delta: 0.02,
            ..P
        };
        let mut gen = NoiseGen::new(0x7607, Colour::White, NOISE_RMS);
        let x = to_f32(&gen.fill(samples(hz, 20_000.0)));
        let mut det = live(hz, &params);
        let hop = FrameGeometry::for_rate(rate(hz)).hop;
        let (mut low_snr, mut not_localized, mut resumed) = (0, 0, 0);
        let mut out = Vec::new();
        let mut last_noise = det.noise().pooled;
        let mut check_resume = false;
        for (i, chunk) in x.chunks(hop).enumerate() {
            let next_id = det.next_id;
            push_all(
                &mut det,
                format(hz, 1),
                chunk,
                (i * hop) as u64,
                hop,
                &mut out,
            );
            // A frame that triggers freezes itself, so only trigger-free frames are checked.
            if check_resume && det.next_id == next_id {
                assert_ne!(
                    det.noise().pooled,
                    last_noise,
                    "N did not resume after release"
                );
                resumed += 1;
            }
            check_resume = false;
            for r in rejections(&out) {
                let noise_only = r.reason == RejectReason::LowSnr
                    && r.gates.as_ref().is_some_and(|g| g.snr.get() < 0.0);
                if noise_only || r.reason == RejectReason::OnsetNotLocalized {
                    assert!(released(&det, r.id), "{r:?} kept its freeze");
                    low_snr += usize::from(noise_only);
                    not_localized += usize::from(r.reason == RejectReason::OnsetNotLocalized);
                    check_resume = det.triggers.iter().all(|t| t.released);
                }
            }
            last_noise = det.noise().pooled;
            out.clear();
        }
        println!(
            "T6 releases: {low_snr} noise-only, {not_localized} not localized, {resumed} resumed"
        );
        assert!(low_snr > 0 && not_localized > 0 && resumed > 0);

        // A Superseded candidate releases its freeze.
        let (x, _, _) = supersede_scene();
        let mut det = live(hz, &P);
        let mut superseded = 0;
        for (i, chunk) in x.chunks(hop).enumerate() {
            let mut out = Vec::new();
            push_all(
                &mut det,
                format(hz, 1),
                chunk,
                (i * hop) as u64,
                hop,
                &mut out,
            );
            for r in rejections(&out) {
                if r.reason == RejectReason::Superseded {
                    assert!(released(&det, r.id), "superseded candidate kept its freeze");
                    superseded += 1;
                }
            }
        }
        assert_eq!(superseded, 1);
    }

    #[test]
    fn t7_warmup_is_silent_and_counts_down() {
        let hz = 48_000;
        let (x, _) = scene(
            hz,
            1,
            0x7707,
            400.0,
            &[
                (50.0, TapKind::Partials, 40.0),
                (150.0, TapKind::Click, 40.0),
            ],
        );
        let hop = FrameGeometry::for_rate(rate(hz)).hop;
        let warm_end = samples(hz, 200.0);
        let mut det = live(hz, &P);
        let mut remaining = Vec::new();
        let mut running = false;
        let mut out = Vec::new();
        for (i, chunk) in x.chunks(hop).enumerate() {
            push_all(
                &mut det,
                format(hz, 1),
                chunk,
                (i * hop) as u64,
                hop,
                &mut out,
            );
            if i * hop + chunk.len() <= warm_end {
                assert!(out.is_empty(), "output during warm-up: {out:?}");
            }
            // The taps inflate the warm-up mean, so a step 11 re-warm may follow; only the
            // first warm-up is checked.
            match det.state() {
                DetectorState::Warming { remaining: r } if !running => remaining.push(r.get()),
                DetectorState::Warming { .. } => {}
                DetectorState::Running => running = true,
            }
        }
        assert!(running, "never left warm-up");
        assert!(remaining.windows(2).all(|w| w[1] <= w[0]), "{remaining:?}");
        assert!(
            remaining.windows(2).skip(1).all(|w| w[1] < w[0]),
            "{remaining:?}"
        );
        assert!(remaining[0] >= 200.0, "{remaining:?}");
    }

    #[test]
    fn t8_strong_tap_supersedes_a_noise_level_trigger() {
        let (x, weak, strong) = supersede_scene();
        let mut det = live(48_000, &P);
        let mut out = Vec::new();
        push_all(&mut det, format(48_000, 1), &x, 0, 480, &mut out);
        let cands = candidates(&out);
        assert_eq!(cands.len(), 1, "{out:?}");
        assert!((cands[0].onset.0 as i64 - strong as i64).abs() <= samples(48_000, 2.0) as i64);
        let rejected = rejections(&out);
        assert_eq!(rejected.len(), 1, "{out:?}");
        assert_eq!(rejected[0].reason, RejectReason::Superseded);
        let near = (rejected[0].trigger.0 as i64 - weak as i64).abs();
        assert!(near <= samples(48_000, 11.0) as i64, "{out:?}");
        assert_eq!(det.health().superseded, 1);
    }

    #[test]
    fn t9_clipped_tap_is_flagged_and_analyzed() {
        let hz = 48_000;
        let t0 = samples(hz, 1000.0);
        let mut y = NoiseGen::new(0x7909, Colour::White, NOISE_RMS).fill(samples(hz, 2000.0));
        let shape = tap(TapKind::Partials, hz, &mut Rng64::new(0x7909));
        add(&mut y, t0, &shape, 0.999);
        let mut x = to_f32(&y);
        // The tap's largest sample is set to exactly ±1.0.
        let peak = (t0..t0 + shape.len())
            .max_by(|&a, &b| x[a].abs().total_cmp(&x[b].abs()))
            .expect("non-empty tap");
        x[peak] = x[peak].signum();
        let mut det = live(hz, &P);
        let mut out = Vec::new();
        push_all(&mut det, format(hz, 1), &x, 0, 480, &mut out);
        let cands = candidates(&out);
        assert_eq!(cands.len(), 1, "{out:?}");
        assert!(cands[0].gates.clipped);
        assert!(cands[0].gates.snr.get() >= P.snr_gate.get());
        assert!(!cands[0].analysis.span[0].is_empty());
    }

    #[test]
    fn t10_nan_is_a_fault_and_detection_recovers() {
        let hz = 48_000;
        let nan_at = samples(hz, 1000.0) + 123;
        let tap_ms = (nan_at + samples(hz, 300.0)) as f64 * 1000.0 / f64::from(hz);
        let (mut x, onsets) = scene(hz, 1, 0x7A10, 2000.0, &[(tap_ms, TapKind::Partials, 40.0)]);
        x[nan_at] = f32::NAN;
        let mut det = live(hz, &P);
        let mut out = Vec::new();
        push_all(&mut det, format(hz, 1), &x, 0, 480, &mut out);
        let faults: Vec<&InputFault> = out
            .iter()
            .filter_map(|o| match o {
                DetectorOutput::InputFault(f) => Some(f),
                DetectorOutput::Candidate(_) | DetectorOutput::Rejected(_) => None,
            })
            .collect();
        assert_eq!(
            faults,
            vec![&InputFault::NonFinite {
                block_start: SampleIndex((nan_at / 480 * 480) as u64),
                count: 1
            }]
        );
        let cands = candidates(&out);
        assert_eq!(cands.len(), 1, "{out:?}");
        let err = cands[0].onset.0 as i64 - onsets[0] as i64;
        assert!(
            err.abs() <= samples(hz, 0.5) as i64,
            "t0 error {err} samples"
        );
        assert_eq!(det.health().faults, 1);
    }

    #[test]
    fn t11_discontinuity_cuts_spans_and_keeps_noise() {
        let hz = 48_000;
        let fmt = format(hz, 1);
        let (first, a) = scene(hz, 1, 0x7B11, 1500.0, &[(1000.0, TapKind::Partials, 40.0)]);
        let cut = a[0] + samples(hz, 20.0);
        let (second, b) = scene(hz, 1, 0x7B12, 1500.0, &[(500.0, TapKind::Partials, 40.0)]);
        let resume = cut + 5000;
        let mut det = live(hz, &P);
        let mut out = Vec::new();
        push_all(&mut det, fmt, &first[..cut], 0, 480, &mut out);
        let noise = det.noise().clone();
        let block = AudioBlock::new(
            fmt,
            SampleIndex(resume as u64),
            &second[..100],
            BlockTime::Untimed,
            true,
        )
        .expect("block");
        det.push(&block, &mut out).expect("push");
        assert_eq!(
            det.noise(),
            &noise,
            "discontinuity changed the noise estimate"
        );
        assert_eq!(det.state(), DetectorState::Running);
        push_all(
            &mut det,
            fmt,
            &second[100..],
            (resume + 100) as u64,
            480,
            &mut out,
        );

        assert!(
            out.contains(&DetectorOutput::InputFault(InputFault::Discontinuity {
                at: SampleIndex(resume as u64)
            }))
        );
        let cut_off: Vec<_> = rejections(&out)
            .into_iter()
            .filter(|r| r.reason == RejectReason::SpanCrossesDiscontinuity)
            .collect();
        assert_eq!(cut_off.len(), 1, "{out:?}");
        assert!((cut_off[0].trigger.0 as i64 - a[0] as i64).abs() <= samples(hz, 11.0) as i64);
        let cands = candidates(&out);
        assert_eq!(cands.len(), 1, "{out:?}");
        assert!((cands[0].onset.0 as i64 - (resume + b[0]) as i64).abs() <= 24);
        for c in &cands {
            let start = c.onset.0 - samples(hz, 4.0) as u64;
            let end = c.onset.0 + samples(hz, 40.0) as u64;
            assert!(
                start >= resume as u64 || end <= cut as u64,
                "span crosses: {c:?}"
            );
        }
    }

    #[test]
    fn t12_same_input_gives_identical_outputs() {
        let x = mixed_scene(48_000, 2);
        let mode = DetectorMode::CalibrationClips;
        let a = run_debug(48_000, 2, &x, &mode, 480);
        let b = run_debug(48_000, 2, &x, &mode, 480);
        assert!(a == b, "two runs differ");
    }

    #[test]
    fn t13_calibration_clips_hold_the_raw_input() {
        let hz = 48_000;
        let fmt = format(hz, 2);
        let (x, onsets) = scene(hz, 2, 0x7D13, 2000.0, &[(1000.0, TapKind::Partials, 40.0)]);
        let mut det =
            Detector::new(fmt, &all_real(2), &P, DetectorMode::CalibrationClips).expect("detector");
        let mut out = Vec::new();
        push_all(&mut det, fmt, &x, 0, 480, &mut out);
        let cands = candidates(&out);
        assert_eq!(cands.len(), 1, "{out:?}");
        let c = cands[0];
        assert!((c.onset.0 as i64 - onsets[0] as i64).abs() <= samples(hz, 0.5) as i64);
        let clip = c.clip.as_ref().expect("clip in CalibrationClips mode");
        let (pre, post) = (samples(hz, 150.0), samples(hz, 60.0));
        assert_eq!(clip.format, fmt);
        assert_eq!(clip.samples.len(), (pre + post) * 2);
        assert_eq!(clip.onset_offset, pre);
        assert_eq!(clip.start.0, c.onset.0 - pre as u64);
        let s = clip.start.0 as usize * 2;
        assert_eq!(clip.samples, x[s..s + (pre + post) * 2].to_vec());
        assert!(c.emitted_at.0 >= c.onset.0 + post as u64 - 1);
    }

    #[test]
    fn t15_live_tracker_agrees_with_estimate_noise() {
        let mut jobs = Vec::new();
        for hz in [48_000u32, 16_000] {
            for colour in [Colour::White, Colour::Pink] {
                jobs.push((hz, colour));
            }
        }
        std::thread::scope(|s| {
            for (k, &(hz, colour)) in jobs.iter().enumerate() {
                s.spawn(move || {
                    let mut gen = NoiseGen::new(0x7F15 + k as u64, colour, NOISE_RMS);
                    let x = to_f32(&gen.fill(samples(hz, 60_000.0)));
                    let mut det = live(hz, &P);
                    let frames = per_frame(&mut det, hz, &x, &mut Vec::new());
                    let mean_db = |sel: &dyn Fn(&FrameRow) -> bool,
                                   val: &dyn Fn(&FrameRow, usize) -> f64| {
                        let chosen: Vec<_> = frames.iter().filter(|f| sel(f)).collect();
                        let mut out = [0.0f64; BAND_COUNT];
                        for (b, o) in out.iter_mut().enumerate() {
                            let m = chosen.iter().map(|f| val(f, b)).sum::<f64>() / chosen.len() as f64;
                            *o = 10.0 * m.log10();
                        }
                        out
                    };
                    let in_range = |lo: f64, hi: f64| {
                        move |f: &FrameRow| {
                            (samples(hz, lo)..samples(hz, hi)).contains(&(f.0.at.0 as usize))
                        }
                    };
                    let live_n = |f: &FrameRow, b: usize| f.1[b];
                    let band_e = |f: &FrameRow, b: usize| {
                        10f64.powf(f64::from(f.0.bands_db[b]) / 10.0)
                    };
                    // Live N over 10-20 s against estimate_noise over a 3 s recording (20-23 s).
                    let live_db = mean_db(&in_range(10_000.0, 20_000.0), &live_n);
                    let rec = &x[samples(hz, 20_000.0)..samples(hz, 23_000.0)];
                    let est = estimate_noise(format(hz, 1), &all_real(1), rec, &[]).expect("estimate");
                    let diffs: Vec<f64> = live_db
                        .iter()
                        .zip(&est.pooled)
                        .map(|(l, e)| l - 10.0 * e.log10())
                        .collect();
                    let mean = diffs.iter().sum::<f64>() / BAND_COUNT as f64;
                    let worst = diffs.iter().fold(0.0f64, |m, d| m.max(d.abs()));
                    // Live N time-mean over 1-60 s against the mean band power.
                    let n60 = mean_db(&in_range(1000.0, 60_000.0), &live_n);
                    let e60 = mean_db(&in_range(1000.0, 60_000.0), &band_e);
                    let bias = n60.iter().zip(&e60).map(|(n, e)| n - e).sum::<f64>() / BAND_COUNT as f64;
                    println!("T15 {hz} {colour:?}: worst {worst:.3} dB, mean {mean:+.3} dB, 60 s bias {bias:+.3} dB");
                    assert!(worst <= 1.0, "{hz} {colour:?}: band differs by {worst} dB");
                    assert!(mean.abs() <= 0.25, "{hz} {colour:?}: mean difference {mean} dB");
                    assert!(bias.abs() <= 0.25, "{hz} {colour:?}: 60 s bias {bias} dB");
                });
            }
        });
    }

    #[test]
    fn t16_tracker_recovers_only_from_real_drops() {
        std::thread::scope(|s| {
            for hz in TABLE_RATES {
                s.spawn(move || {
                    let mut gen = NoiseGen::new(0x7016 + u64::from(hz), Colour::White, NOISE_RMS);
                    let mut det = live(hz, &P);
                    let fmt = format(hz, 1);
                    let block = 4096;
                    let total = samples(hz, 60_000.0);
                    let mut out = Vec::new();
                    let mut pos = 0;
                    while pos < total {
                        let n = block.min(total - pos);
                        let buf: Vec<f32> = (0..n).map(|_| gen.sample() as f32).collect();
                        push_all(&mut det, fmt, &buf, pos as u64, block, &mut out);
                        out.clear();
                        pos += n;
                    }
                    assert_eq!(
                        det.health().rewarms,
                        0,
                        "{hz}: re-warmed on stationary noise"
                    );
                });
            }
        });

        // A −12 dB step: one re-warm within 25 ms, then no trigger until warm-up and the 19
        // defined D frames have refilled.
        let hz = 48_000;
        let g = FrameGeometry::for_rate(rate(hz));
        let mut gen = NoiseGen::new(0x7016, Colour::White, NOISE_RMS);
        let (step, end) = (samples(hz, 3000.0) + 77, samples(hz, 5000.0));
        let gain = 10f64.powf(12.0 / 20.0);
        let x: Vec<f32> = (0..end)
            .map(|i| (gen.sample() * if i < step { gain } else { 1.0 }) as f32)
            .collect();
        let mut det = live(hz, &P);
        let mut out = Vec::new();
        let frames = per_frame(&mut det, hz, &x, &mut out);
        assert_eq!(det.health().rewarms, 1);
        let r = frames.iter().position(|f| f.2 == 1).expect("re-warm frame");
        let rewarm_end = frames[r].0.at.0 as usize + g.len / 2;
        assert!(
            rewarm_end >= step && rewarm_end - step <= samples(hz, 25.0),
            "re-warm at {rewarm_end}, step {step}"
        );
        let refill = det.warmup_frames + P.flux_lag_frames + P.peak_mean_frames;
        for f in &frames[r + 1..=r + refill] {
            assert_eq!(
                f.0.threshold, 0.0,
                "peak-pick defined during refill at {}",
                f.0.at.0
            );
        }
        assert!(frames[r + refill + 1].0.threshold > 0.0);
        let refilled = frames[r + refill].0.at.0 + g.len as u64 / 2;
        for o in &out {
            let trigger = match o {
                DetectorOutput::Candidate(c) => c.trigger,
                DetectorOutput::Rejected(r) => r.trigger,
                DetectorOutput::InputFault(f) => panic!("fault {f:?}"),
            };
            assert!(
                !(step as u64..refilled).contains(&trigger.0),
                "trigger at {} during the refill",
                trigger.0
            );
        }
    }
}
