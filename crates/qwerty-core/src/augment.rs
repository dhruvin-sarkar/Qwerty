//! Calibration augmentation: deterministic noise-mixed and jittered variants of calibration taps (§1.11).

use crate::audio::{AudioBlock, AudioFormat, BlockTime, ChannelLayout};
use crate::descriptor::{Descriptor, DescriptorError, DescriptorExtractor, DescriptorLayout};
use crate::detector::{
    estimate_noise, Candidate, Detector, DetectorError, DetectorMode, DetectorOutput,
    DetectorParams, NoiseSnapshot, RawClip, RejectReason, CLIP_POST, CLIP_PRE, SPAN_AFTER,
    SPAN_BEFORE,
};
use crate::dsp::{BandLayout, Biquad, FftConvolver, FrameGeometry, BAND_COUNT};
use crate::rng::Rng64;
use crate::session::AMBIENT_DURATION;
use crate::units::{Decibels, Hertz, Milliseconds, SampleIndex, SampleRate};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};
use std::f64::consts::TAU;

pub const VARIANTS_PER_TAP: usize = 8;
pub const SNR_SCHEDULE: [Option<Decibels>; VARIANTS_PER_TAP] = [
    None,
    None,
    Some(Decibels::new(24.0)),
    Some(Decibels::new(24.0)),
    Some(Decibels::new(18.0)),
    Some(Decibels::new(18.0)),
    Some(Decibels::new(12.0)),
    Some(Decibels::new(12.0)),
]; // None = no ambient mixed (∞)
   // No gain range: deviation m (DSP-ML-3).
pub const TILT_RANGE_DB_PER_OCTAVE: f64 = 1.5; // U[−1.5, +1.5] dB/oct
pub const TILT_REFERENCE: Hertz = Hertz::new(1000.0);
pub const TILT_LOW_LIMIT: Hertz = Hertz::new(50.0); // tilt held constant below this
/// Within 0.04 dB of target at 500 Hz / 2 kHz for s = ±1.5; 4 ms missed 500 Hz by 0.10-0.18 dB
/// (DSP-ML-11).
pub const TILT_FIR: Milliseconds = Milliseconds::new(8.0);
/// U[−0.2, +0.2] ms on the post-AIC anchor (deviation c): measured t0 error p90 0.04 ms, max 0.21 ms.
pub const JITTER_RANGE: Milliseconds = Milliseconds::new(0.2);
/// Mixing offsets clear of every excluded range (DSP-ML-6).
pub const AMBIENT_MIN_USABLE: Milliseconds = Milliseconds::new(1000.0);
pub const REANALYSIS_TOLERANCE: Milliseconds = Milliseconds::new(10.0);
pub const AMBIENT_SILENT_RMS: f64 = 1e-6; // −120 dBFS

/// The pre-onset stretch left out of the tap's noise estimate (§1.11.2 step 2).
const NOISE_GUARD: Milliseconds = Milliseconds::new(20.0);
/// E_tap never falls below this fraction of the span energy (§1.11.2 step 2).
const MIN_TAP_ENERGY_FRACTION: f64 = 0.1;

#[derive(Debug, Clone, PartialEq)]
pub struct AmbientRecording {
    pub format: AudioFormat,
    /// raw, interleaved, ≥ AMBIENT_DURATION
    pub samples: Vec<f32>,
    /// P^a: estimate_noise() over the recording (pooled + chosen)
    pub band_power: NoiseSnapshot,
    /// RMS over real channels < AMBIENT_SILENT_RMS
    pub silent: bool,
    /// frame ranges around ambient disturbances (§1.14.2 rule 1)
    pub exclude: Vec<(u64, u64)>,
}

impl AmbientRecording {
    /// Err(AmbientTooDisturbed) when fewer than samples_in(AMBIENT_MIN_USABLE) mixing offsets remain for a
    /// clip of CLIP_PRE + CLIP_POST, i.e. offsets o whose [o, o + clip_frames) lies inside no excluded range.
    // SPEC: `exclude` ranges are half-open [start, end) frame ranges, as `estimate_noise` reads them.
    // The disturbance check runs before `estimate_noise`: a recording disturbed enough to leave
    // estimate_noise too few frames always has too few mixing offsets, and is reported as such.
    pub fn new(
        format: AudioFormat,
        layout: &ChannelLayout,
        samples: Vec<f32>,
        exclude: &[(u64, u64)],
    ) -> Result<Self, AugmentError> {
        let channels = usize::from(format.channels.get());
        let frames = (samples.len() / channels) as u64;
        let needed = format.rate.samples_in(AMBIENT_DURATION) as u64;
        if frames < needed {
            return Err(AugmentError::AmbientTooShort { frames, needed });
        }
        let min_usable = format.rate.samples_in(AMBIENT_MIN_USABLE) as u64;
        let usable = count_offsets(&usable_ranges(frames, exclude, clip_frames(format.rate)));
        if usable < min_usable {
            return Err(AugmentError::AmbientTooDisturbed {
                usable_frames: usable,
                needed: min_usable,
            });
        }
        let band_power = estimate_noise(format, layout, &samples, exclude)
            .map_err(|error| AugmentError::Detector { error })?;
        // estimate_noise has checked that every real channel is < channels.
        let real: Vec<usize> = layout.real().iter().map(|&c| usize::from(c)).collect();
        let mut sum = 0.0f64;
        for frame in samples.chunks_exact(channels) {
            for &c in &real {
                sum += f64::from(frame[c]).powi(2);
            }
        }
        let count = frames as f64 * real.len() as f64;
        let silent = (sum / count).sqrt() < AMBIENT_SILENT_RMS;
        Ok(AmbientRecording {
            format,
            samples,
            band_power,
            silent,
            exclude: exclude.to_vec(),
        })
    }

    pub fn usable_offsets(&self, clip_frames: u64) -> u64 {
        count_offsets(&usable_ranges(self.frames(), &self.exclude, clip_frames))
    }

    fn frames(&self) -> u64 {
        (self.samples.len() / usize::from(self.format.channels.get())) as u64
    }

    /// The k-th usable offset in ascending order; k < usable_offsets(clip_frames).
    fn offset_at(&self, clip_frames: u64, k: u64) -> u64 {
        let mut k = k;
        for (lo, hi) in usable_ranges(self.frames(), &self.exclude, clip_frames) {
            let len = hi - lo + 1;
            if k < len {
                return lo + k;
            }
            k -= len;
        }
        // INVARIANT: every caller draws k with Rng64::below(usable_offsets(clip_frames)).
        unreachable!("offset index beyond the usable offsets")
    }
}

/// Frames of a calibration clip, [t0 − CLIP_PRE, t0 + CLIP_POST), as the detector cuts it.
fn clip_frames(rate: SampleRate) -> u64 {
    (rate.samples_in(CLIP_PRE) + rate.samples_in(CLIP_POST)) as u64
}

/// Usable mixing offsets as inclusive, ascending, disjoint ranges [lo, hi]: o ≤ frames − clip and
/// [o, o + clip) overlaps no half-open excluded range.
fn usable_ranges(frames: u64, exclude: &[(u64, u64)], clip: u64) -> Vec<(u64, u64)> {
    if clip == 0 || frames < clip {
        return Vec::new();
    }
    let last = frames - clip;
    // [o, o + clip) overlaps [x0, x1) iff x0 + 1 − clip ≤ o ≤ x1 − 1.
    let mut forbidden: Vec<(u64, u64)> = exclude
        .iter()
        .filter(|&&(x0, x1)| x0 < x1)
        .map(|&(x0, x1)| ((x0 + 1).saturating_sub(clip), x1 - 1))
        .collect();
    forbidden.sort_unstable();
    let mut out = Vec::new();
    let mut next = 0u64;
    for (f0, f1) in forbidden {
        if next > last {
            break;
        }
        if f0 > next {
            out.push((next, (f0 - 1).min(last)));
        }
        next = next.max(f1.saturating_add(1));
    }
    if next <= last {
        out.push((next, last));
    }
    out
}

fn count_offsets(ranges: &[(u64, u64)]) -> u64 {
    ranges.iter().map(|&(lo, hi)| hi - lo + 1).sum()
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VariantParams {
    pub index: u8,
    pub tilt_db_per_octave: f64,
    pub jitter: Milliseconds,
    pub snr: Option<Decibels>,
    pub ambient_offset: u64,
}

impl VariantParams {
    /// Variant 0: the live descriptor itself.
    const LIVE: VariantParams = VariantParams {
        index: 0,
        tilt_db_per_octave: 0.0,
        jitter: Milliseconds::new(0.0),
        snr: None,
        ambient_offset: 0,
    };
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum VariantDrop {
    NotDetected,
    MultipleCandidates { count: u8 },
    Rejected { reason: RejectReason },
    Descriptor { error: DescriptorError },
}

#[derive(Debug, Clone, PartialEq)]
pub struct AugmentOutcome {
    /// [0] = the live descriptor, then surviving variants in index order
    pub variants: Vec<Descriptor>,
    /// aligned with `variants`
    pub params: Vec<VariantParams>,
    pub dropped: Vec<(VariantParams, VariantDrop)>,
    /// true ⇒ noise mixing was skipped (named, surfaced in TrainingSummary.variants_dropped context)
    pub ambient_silent: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AugmentError {
    FormatMismatch {
        expected: AudioFormat,
        got: AudioFormat,
    },
    AmbientTooShort {
        frames: u64,
        needed: u64,
    },
    AmbientTooDisturbed {
        usable_frames: u64,
        needed: u64,
    },
    Detector {
        error: DetectorError,
    },
}

/// One synthesized variant before re-analysis (§1.11.2 steps 1-3).
struct Rendered {
    /// z: the tilted clip plus κ·a, raw, interleaved
    z: Vec<f32>,
    /// N': the FrozenNoise seed
    noise: NoiseSnapshot,
    /// the SNR actually mixed: `None` when no ambient was mixed
    snr: Option<Decibels>,
}

/// What a re-analysis found around the clip's onset (§1.11.2 step 4).
type Found = Result<Box<Candidate>, VariantDrop>;

pub struct Augmenter {
    format: AudioFormat,
    layout: ChannelLayout,
    params: DetectorParams,
    /// real channel indices, in layout order
    real: Vec<usize>,
    chosen: usize,
    centres: [Hertz; BAND_COUNT],
    hpf: Biquad,
    convolver: FftConvolver,
    planner: FftPlanner<f64>,
    extractor: DescriptorExtractor,
    // Scratch, reused across variants.
    spectrum: Vec<Complex<f64>>,
    fft_scratch: Vec<Complex<f64>>,
    taps: Vec<f64>,
    taps_f32: Vec<f32>,
    channel: Vec<f32>,
    filtered: Vec<f32>,
    outputs: Vec<DetectorOutput>,
}

impl Augmenter {
    /// # Panics
    /// If `params.hpf_cutoff` is not inside (0, fs/2), as `Detector::new`. `DetectorParams::DEFAULT`
    /// (100 Hz) always is.
    pub fn new(
        format: AudioFormat,
        layout: &ChannelLayout,
        params: &DetectorParams,
    ) -> Result<Self, AugmentError> {
        let channels = format.channels.get();
        if let Some(&channel) = layout.real().iter().find(|&&c| c >= channels) {
            return Err(AugmentError::Detector {
                error: DetectorError::LayoutChannelOutOfRange { channel, channels },
            });
        }
        let rate = format.rate;
        let hpf = Biquad::highpass(rate, params.hpf_cutoff)
            .unwrap_or_else(|e| panic!("DetectorParams::hpf_cutoff is invalid: {e}"));
        let descriptor_layout = DescriptorLayout::from_channels(layout);
        Ok(Augmenter {
            format,
            layout: layout.clone(),
            params: *params,
            real: layout.real().iter().map(|&c| usize::from(c)).collect(),
            chosen: descriptor_layout.chosen().len(),
            centres: BandLayout::new(rate, FrameGeometry::for_rate(rate)).centres,
            hpf,
            convolver: FftConvolver::new(),
            planner: FftPlanner::new(),
            extractor: DescriptorExtractor::new(rate, descriptor_layout),
            spectrum: Vec::new(),
            fft_scratch: Vec::new(),
            taps: Vec::new(),
            taps_f32: Vec::new(),
            channel: Vec::new(),
            filtered: Vec::new(),
            outputs: Vec::new(),
        })
    }

    /// `item` is a stable per-clip index (tap index for taps; 1_000_000 + n for negatives).
    pub fn augment(
        &mut self,
        live: &Descriptor,
        clip: &RawClip,
        noise: &NoiseSnapshot,
        ambient: &AmbientRecording,
        seed: u64,
        item: u64,
    ) -> Result<AugmentOutcome, AugmentError> {
        self.check_format(clip.format)?;
        self.check_format(ambient.format)?;
        self.check_snapshot(noise)?;
        self.check_snapshot(&ambient.band_power)?;
        let rate = self.format.rate;
        let frames = (clip.samples.len() / usize::from(self.format.channels.get())) as u64;
        let usable = ambient.usable_offsets(frames);
        let min_usable = rate.samples_in(AMBIENT_MIN_USABLE) as u64;
        if usable < min_usable {
            return Err(AugmentError::AmbientTooDisturbed {
                usable_frames: usable,
                needed: min_usable,
            });
        }
        let mut outcome = AugmentOutcome {
            variants: vec![*live],
            params: vec![VariantParams::LIVE],
            dropped: Vec::new(),
            ambient_silent: ambient.silent,
        };
        for (v, &scheduled) in SNR_SCHEDULE.iter().enumerate().skip(1) {
            let mut rng = Rng64::derive(seed, item, v as u64);
            let s = rng.range(-TILT_RANGE_DB_PER_OCTAVE, TILT_RANGE_DB_PER_OCTAVE);
            let j = rng.range(-JITTER_RANGE.get(), JITTER_RANGE.get());
            let offset = ambient.offset_at(frames, rng.below(usable));
            let snr = if ambient.silent { None } else { scheduled };
            let rendered = self.render(clip, noise, ambient, s, snr, offset);
            let params = VariantParams {
                index: v as u8,
                tilt_db_per_octave: s,
                // INVARIANT: j is drawn from [−0.2, 0.2).
                jitter: Milliseconds::try_new(j).expect("jitter is finite"),
                snr: rendered.snr,
                ambient_offset: offset,
            };
            let jitter = (j * f64::from(rate.hz()) / 1000.0).round() as i32;
            let found = self.reanalyze(&rendered.z, clip.onset_offset, rendered.noise, jitter)?;
            match self.describe(found) {
                Ok(descriptor) => {
                    outcome.variants.push(descriptor);
                    outcome.params.push(params);
                }
                Err(drop) => outcome.dropped.push((params, drop)),
            }
        }
        Ok(outcome)
    }

    /// The FrozenNoise re-analysis of an unmodified clip under this augmenter's layout, jitter 0
    /// (§1.14.2 rule 5a: an EnvironmentOnset clip's variant 0).
    pub(crate) fn reanalyze_clip(
        &mut self,
        clip: &RawClip,
        noise: &NoiseSnapshot,
    ) -> Result<Result<Descriptor, VariantDrop>, AugmentError> {
        self.check_format(clip.format)?;
        self.check_snapshot(noise)?;
        let found = self.reanalyze(&clip.samples, clip.onset_offset, noise.clone(), 0)?;
        Ok(self.describe(found))
    }

    fn check_format(&self, got: AudioFormat) -> Result<(), AugmentError> {
        if got == self.format {
            Ok(())
        } else {
            Err(AugmentError::FormatMismatch {
                expected: self.format,
                got,
            })
        }
    }

    /// A snapshot taken under another layout is reported as `Detector::new` reports it.
    fn check_snapshot(&self, noise: &NoiseSnapshot) -> Result<(), AugmentError> {
        if noise.chosen.len() == self.chosen {
            return Ok(());
        }
        let chosen = DescriptorLayout::from_channels(&self.layout).chosen();
        Err(AugmentError::Detector {
            error: DetectorError::LayoutChannelOutOfRange {
                channel: chosen[chosen.len() - 1],
                channels: noise.chosen.len() as u16,
            },
        })
    }

    /// §1.11.2 steps 1-3: tilt, ambient mix and the noise seed.
    fn render(
        &mut self,
        clip: &RawClip,
        noise: &NoiseSnapshot,
        ambient: &AmbientRecording,
        s: f64,
        snr: Option<Decibels>,
        offset: u64,
    ) -> Rendered {
        let rate = self.format.rate;
        let ch = usize::from(self.format.channels.get());
        let frames = clip.samples.len() / ch;
        let x = &clip.samples[..frames * ch];

        // 1. Tilt, per channel, "same" alignment (zero delay).
        self.design_tilt(s);
        let mut z = vec![0.0f32; frames * ch];
        for c in 0..ch {
            self.channel.clear();
            self.channel.extend(x.iter().skip(c).step_by(ch));
            self.convolver
                .convolve_same(&self.channel, &self.taps_f32, &mut self.filtered);
            for (out, &v) in z.iter_mut().skip(c).step_by(ch).zip(&self.filtered) {
                *out = v;
            }
        }

        // 2. Ambient mix.
        let (e_tap, span) = self.tap_energy(&z, clip.onset_offset);
        let mut kappa = 0.0;
        let mut mixed = None;
        if let Some(snr) = snr {
            let start = offset as usize * ch;
            let a = &ambient.samples[start..start + frames * ch];
            let (e_a, _) = self.hpf_energy(a, span, 0);
            if e_a > 0.0 {
                kappa = (e_tap / (e_a * 10f64.powf(snr.get() / 10.0))).sqrt();
                for (out, &v) in z.iter_mut().zip(a) {
                    *out = (f64::from(*out) + kappa * f64::from(v)) as f32;
                }
                mixed = Some(snr);
            }
        }

        // 3. Noise seed: N'_b = |H(f_b)|²·N*_b + κ²·P^a_b.
        let fs = f64::from(rate.hz());
        let mut gain = [0.0f64; BAND_COUNT];
        for (g, f) in gain.iter_mut().zip(&self.centres) {
            *g = response(&self.taps, fs, f.get()).powi(2);
        }
        let k2 = kappa * kappa;
        let seed = |n: &[f64; BAND_COUNT], p: &[f64; BAND_COUNT]| -> [f64; BAND_COUNT] {
            std::array::from_fn(|b| gain[b] * n[b] + k2 * p[b])
        };
        let seeded = NoiseSnapshot {
            pooled: seed(&noise.pooled, &ambient.band_power.pooled),
            chosen: noise
                .chosen
                .iter()
                .zip(&ambient.band_power.chosen)
                .map(|(n, p)| seed(n, p))
                .collect(),
        };
        Rendered {
            z,
            noise: seeded,
            snr: mixed,
        }
    }

    /// (E_tap, span) of §1.11.2 step 2 for the tilted clip `y`: the span is
    /// [onset − SPAN_BEFORE, onset + SPAN_AFTER), and E_tap = max(E_span − E_noise, 0.1·E_span)
    /// with E_noise the mean hpf(y)² over [0, onset − 20 ms) × span frames × n_real.
    fn tap_energy(&self, y: &[f32], onset_offset: usize) -> (f64, (usize, usize)) {
        let rate = self.format.rate;
        let frames = y.len() / usize::from(self.format.channels.get());
        let onset = onset_offset.min(frames);
        let span = (
            onset.saturating_sub(rate.samples_in(SPAN_BEFORE)),
            (onset + rate.samples_in(SPAN_AFTER)).min(frames),
        );
        let guard_end = onset.saturating_sub(rate.samples_in(NOISE_GUARD));
        let n_real = self.real.len() as f64;
        let (e_span, pre_sum) = self.hpf_energy(y, span, guard_end);
        let e_noise = if guard_end == 0 {
            0.0
        } else {
            pre_sum / (guard_end as f64 * n_real) * (span.1 - span.0) as f64 * n_real
        };
        (
            (e_span - e_noise).max(MIN_TAP_ENERGY_FRACTION * e_span),
            span,
        )
    }

    /// (Σ hpf(x)² over `span`, Σ hpf(x)² over [0, pre_end)) on the real channels, with fresh
    /// high-pass filters from frame 0.
    fn hpf_energy(&self, x: &[f32], span: (usize, usize), pre_end: usize) -> (f64, f64) {
        let ch = usize::from(self.format.channels.get());
        let end = span.1.max(pre_end);
        let (mut span_sum, mut pre_sum) = (0.0f64, 0.0f64);
        for &c in &self.real {
            let mut hpf = self.hpf.clone();
            hpf.reset();
            for (s, frame) in x.chunks_exact(ch).take(end).enumerate() {
                let v = f64::from(hpf.process(frame[c])).powi(2);
                if s < pre_end {
                    pre_sum += v;
                }
                if (span.0..span.1).contains(&s) {
                    span_sum += v;
                }
            }
        }
        (span_sum, pre_sum)
    }

    /// §1.11.2 step 1: the zero-phase tilt FIR for `s` dB/octave, into `taps` and `taps_f32`.
    fn design_tilt(&mut self, s: f64) {
        let rate = self.format.rate;
        let fs = f64::from(rate.hz());
        let t = tilt_length(rate);
        let n = (4 * t).next_power_of_two();
        self.spectrum.clear();
        self.spectrum.extend((0..n).map(|k| {
            let bin = k.min(n - k);
            Complex::new(tilt_target(bin as f64 * fs / n as f64, s), 0.0)
        }));
        let inverse = self.planner.plan_fft_inverse(n);
        self.fft_scratch
            .resize(inverse.get_inplace_scratch_len(), Complex::new(0.0, 0.0));
        inverse.process_with_scratch(&mut self.spectrum, &mut self.fft_scratch);
        let half = (t - 1) / 2;
        let scale = 1.0 / n as f64;
        self.taps.clear();
        for i in 0..t {
            let hann = 0.5 - 0.5 * (TAU * i as f64 / (t - 1) as f64).cos();
            self.taps
                .push(self.spectrum[(i + n - half) % n].re * scale * hann);
        }
        let at_reference = response(&self.taps, fs, TILT_REFERENCE.get());
        for v in &mut self.taps {
            *v /= at_reference;
        }
        self.taps_f32.clear();
        self.taps_f32.extend(self.taps.iter().map(|&v| v as f32));
    }

    /// §1.11.2 step 4: FrozenNoise re-analysis of `z` pushed as one untimed block from 0.
    fn reanalyze(
        &mut self,
        z: &[f32],
        onset_offset: usize,
        noise: NoiseSnapshot,
        jitter: i32,
    ) -> Result<Found, AugmentError> {
        let mode = DetectorMode::FrozenNoise {
            noise,
            anchor_jitter: jitter,
        };
        let mut detector = Detector::new(self.format, &self.layout, &self.params, mode)
            .map_err(|error| AugmentError::Detector { error })?;
        let ch = usize::from(self.format.channels.get());
        let whole = z.len() / ch * ch;
        // INVARIANT: `whole` is a multiple of the channel count.
        let block = AudioBlock::new(
            self.format,
            SampleIndex(0),
            &z[..whole],
            BlockTime::Untimed,
            false,
        )
        .expect("whole frames");
        self.outputs.clear();
        detector
            .push(&block, &mut self.outputs)
            .map_err(|error| AugmentError::Detector { error })?;
        let tolerance = self.format.rate.samples_in(REANALYSIS_TOLERANCE) as u64;
        let near = |s: SampleIndex| s.0.abs_diff(onset_offset as u64) <= tolerance;
        let mut found: Option<Box<Candidate>> = None;
        let mut count = 0u8;
        let mut rejected: Option<RejectReason> = None;
        for output in self.outputs.drain(..) {
            match output {
                DetectorOutput::Candidate(c) => {
                    if near(c.onset) {
                        count = count.saturating_add(1);
                        if found.is_none() {
                            found = Some(c);
                        }
                    }
                }
                DetectorOutput::Rejected(r) => {
                    let at = match r.onset {
                        Some(onset) => onset,
                        // No onset was localized: the trigger frame's centre places it.
                        None => r.trigger,
                    };
                    if rejected.is_none() && near(at) {
                        rejected = Some(r.reason);
                    }
                }
                // A non-finite clip sample: nothing is analyzed, which `NotDetected` reports.
                DetectorOutput::InputFault(_) => {}
            }
        }
        Ok(match (count, found) {
            (1, Some(candidate)) => Ok(candidate),
            (0, _) | (1, None) => Err(match rejected {
                Some(reason) => VariantDrop::Rejected { reason },
                None => VariantDrop::NotDetected,
            }),
            (count, _) => Err(VariantDrop::MultipleCandidates { count }),
        })
    }

    fn describe(&mut self, found: Found) -> Result<Descriptor, VariantDrop> {
        let candidate = found?;
        self.extractor
            .extract(&candidate.analysis)
            .map_err(|error| VariantDrop::Descriptor { error })
    }
}

/// T = 2·⌊samples_in(TILT_FIR)/2⌋ + 1 (385 taps at 48 kHz, 1537 at 192 kHz).
fn tilt_length(rate: SampleRate) -> usize {
    2 * (rate.samples_in(TILT_FIR) / 2) + 1
}

/// Target |H(f)| = 10^(s·log2(max(f, 50)/1000)/20); the max floors the log argument.
fn tilt_target(f: f64, s: f64) -> f64 {
    let octaves = (f.max(TILT_LOW_LIMIT.get()) / TILT_REFERENCE.get()).log2();
    10f64.powf(s * octaves / 20.0)
}

/// |H(f)| of the FIR `h` at `fs`, by direct DFT.
fn response(h: &[f64], fs: f64, f: f64) -> f64 {
    let w = TAU * f / fs;
    let (re, im) = h
        .iter()
        .enumerate()
        .fold((0.0f64, 0.0f64), |(re, im), (n, &v)| {
            let phase = w * n as f64;
            (re + v * phase.cos(), im - v * phase.sin())
        });
    re.hypot(im)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desk::{DeskConfig, ModalDesk, TapSpec};
    use crate::detector::testkit::*;
    use crate::rng::Rng64;

    const P: DetectorParams = DetectorParams::DEFAULT;

    /// A live calibration tap: the stream it came from, its onset, and what the live detector
    /// stored for it.
    struct LiveTap {
        stream: Vec<f32>,
        onset: usize,
        clip: RawClip,
        live: Descriptor,
        noise: NoiseSnapshot,
    }

    /// Runs `x` through Detector(CalibrationClips) and extracts the single candidate.
    fn live_tap(fmt: AudioFormat, layout: &ChannelLayout, x: Vec<f32>) -> LiveTap {
        let mut det =
            Detector::new(fmt, layout, &P, DetectorMode::CalibrationClips).expect("detector");
        let mut out = Vec::new();
        push_all(&mut det, fmt, &x, 0, 480, &mut out);
        let cands = candidates(&out);
        assert_eq!(cands.len(), 1, "{out:?}");
        let c = cands[0];
        let mut extractor =
            DescriptorExtractor::new(fmt.rate, DescriptorLayout::from_channels(layout));
        LiveTap {
            onset: c.onset.0 as usize,
            clip: c.clip.clone().expect("clip in CalibrationClips mode"),
            live: extractor.extract(&c.analysis).expect("live descriptor"),
            noise: c.analysis.noise.clone(),
            stream: x,
        }
    }

    fn desk_spec(zone: usize, force_db: f64) -> TapSpec {
        let (x, y) = ModalDesk::zone_centres(4)[zone];
        TapSpec {
            x,
            y,
            force: Decibels::new(force_db),
            contact: Milliseconds::new(1.0),
            position_jitter_m: 0.0,
        }
    }

    /// A stereo desk tap (zone 1 of 4, about 50 dB over the −80 dBFS floor) 1 s into a 1.4 s
    /// stream, and a 3 s ambient of the same desk's floor.
    fn desk_fixture(hz: u32, seed: u64) -> (Augmenter, LiveTap, AmbientRecording) {
        let fmt = format(hz, 2);
        let layout = all_real(2);
        let mut desk = ModalDesk::new(DeskConfig::default_for(rate(hz), seed)).expect("desk");
        let lead = samples(hz, 1000.0);
        let mut x = desk.floor(lead + samples(hz, 400.0));
        let tap = desk.tap(&desk_spec(1, -30.0), lead);
        for (v, t) in x.iter_mut().zip(&tap) {
            *v += t;
        }
        let tap = live_tap(fmt, &layout, x);
        let floor = desk.floor(samples(hz, 3000.0));
        let ambient = AmbientRecording::new(fmt, &layout, floor, &[]).expect("ambient");
        let augmenter = Augmenter::new(fmt, &layout, &P).expect("augmenter");
        (augmenter, tap, ambient)
    }

    /// Bit patterns of every variant, for exact comparisons.
    fn bits(o: &AugmentOutcome) -> Vec<Vec<u32>> {
        o.variants
            .iter()
            .map(|d| d.values.iter().map(|v| v.to_bits()).collect())
            .collect()
    }

    /// Every variant's params (kept and dropped) in index order, variant 0 included.
    fn all_params(o: &AugmentOutcome) -> Vec<VariantParams> {
        let mut p: Vec<VariantParams> = o
            .params
            .iter()
            .copied()
            .chain(o.dropped.iter().map(|d| d.0))
            .collect();
        p.sort_by_key(|p| p.index);
        p
    }

    /// Renders (s, snr, offset 0, jitter 0), re-analyzes and describes it.
    fn variant(
        aug: &mut Augmenter,
        clip: &RawClip,
        noise: &NoiseSnapshot,
        ambient: &AmbientRecording,
        s: f64,
        snr: Option<Decibels>,
    ) -> Descriptor {
        let r = aug.render(clip, noise, ambient, s, snr, 0);
        let found = aug
            .reanalyze(&r.z, clip.onset_offset, r.noise, 0)
            .expect("re-analysis");
        aug.describe(found).expect("variant survives")
    }

    /// A unit-peak click (a 1-3 ms noise burst).
    fn click(hz: u32, seed: u64) -> Vec<f64> {
        tap(TapKind::Click, hz, &mut Rng64::new(seed))
    }

    fn add_click(x: &mut [f32], channels: usize, at: usize, click: &[f64], gain: f64) {
        for (i, v) in click.iter().enumerate() {
            for c in 0..channels {
                x[(at + i) * channels + c] += (gain * v) as f32;
            }
        }
    }

    #[test]
    fn au1_same_seed_and_item_are_bit_identical() {
        let (mut aug, tap, ambient) = desk_fixture(48_000, 0xA001);
        let run = |aug: &mut Augmenter, item: u64| {
            aug.augment(&tap.live, &tap.clip, &tap.noise, &ambient, 0x5EED, item)
                .expect("augment")
        };
        let a = run(&mut aug, 3);
        let b = run(&mut aug, 3);
        let mut fresh = Augmenter::new(ambient.format, &all_real(2), &P).expect("augmenter");
        let c = run(&mut fresh, 3);
        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_eq!(bits(&a), bits(&b));
        assert_eq!(bits(&a), bits(&c));
        assert_eq!(a.variants.len(), VARIANTS_PER_TAP, "{:?}", a.dropped);

        let other = run(&mut aug, 4);
        let (pa, po) = (all_params(&a), all_params(&other));
        assert_eq!(pa.len(), VARIANTS_PER_TAP);
        assert_eq!(
            pa[0], po[0],
            "variant 0 is the live descriptor for every item"
        );
        for (x, y) in pa.iter().zip(&po).skip(1) {
            assert_ne!(x.tilt_db_per_octave, y.tilt_db_per_octave);
            assert_ne!(x.jitter, y.jitter);
            assert_ne!(x.ambient_offset, y.ambient_offset);
        }
    }

    #[test]
    fn au2_live_variant_ranges_and_realized_snr() {
        let hz = 48_000;
        let (mut aug, tap, ambient) = desk_fixture(hz, 0xA002);
        let clip = clip_frames(rate(hz));
        for item in 0..4u64 {
            let o = aug
                .augment(&tap.live, &tap.clip, &tap.noise, &ambient, 0xA2, item)
                .expect("augment");
            assert_eq!(o.variants[0], tap.live, "variant 0 is the live descriptor");
            assert_eq!(o.params[0], VariantParams::LIVE);
            assert!(!o.ambient_silent);
            assert_eq!(o.variants.len(), o.params.len());
            let params = all_params(&o);
            assert_eq!(params.len(), VARIANTS_PER_TAP);
            for p in params.into_iter().skip(1) {
                let v = usize::from(p.index);
                assert!((1..VARIANTS_PER_TAP).contains(&v));
                assert!(p.tilt_db_per_octave.abs() <= TILT_RANGE_DB_PER_OCTAVE);
                assert!(p.jitter.get().abs() <= JITTER_RANGE.get());
                assert_eq!(p.snr, SNR_SCHEDULE[v], "variant {v}");
                assert!(p.ambient_offset + clip <= ambient.frames());
                let Some(snr) = p.snr else { continue };
                // The realized SNR, by the same span-energy formula, against the target.
                let s = p.tilt_db_per_octave;
                let tilted = aug.render(&tap.clip, &tap.noise, &ambient, s, None, 0);
                let mixed = aug.render(
                    &tap.clip,
                    &tap.noise,
                    &ambient,
                    s,
                    Some(snr),
                    p.ambient_offset,
                );
                assert_eq!(mixed.snr, Some(snr));
                let (e_tap, span) = aug.tap_energy(&tilted.z, tap.clip.onset_offset);
                let added: Vec<f32> = mixed.z.iter().zip(&tilted.z).map(|(m, t)| m - t).collect();
                let (e_mix, _) = aug.hpf_energy(&added, span, 0);
                let realized = 10.0 * (e_tap / e_mix).log10();
                assert!(
                    (realized - snr.get()).abs() <= 1.0,
                    "variant {v}: realized {realized:.3} dB, target {}",
                    snr.get()
                );
            }
        }
    }

    #[test]
    fn au3_silent_ambient_mixes_no_noise() {
        let hz = 48_000;
        let (mut aug, tap, _) = desk_fixture(hz, 0xA003);
        let silence = vec![0.0f32; samples(hz, 3000.0) * 2];
        let ambient =
            AmbientRecording::new(format(hz, 2), &all_real(2), silence, &[]).expect("ambient");
        assert!(ambient.silent);
        let o = aug
            .augment(&tap.live, &tap.clip, &tap.noise, &ambient, 0xA3, 0)
            .expect("augment");
        assert!(o.ambient_silent);
        assert_eq!(o.variants.len(), VARIANTS_PER_TAP, "{:?}", o.dropped);
        assert!(o.params.iter().all(|p| p.snr.is_none()));
        assert!(o.dropped.is_empty());
    }

    #[test]
    fn au4_drop_rules() {
        let hz = 48_000;
        let (mut aug, tap, ambient) = desk_fixture(hz, 0xA004);
        let ch = 2;
        let (pre, post) = (samples(hz, 150.0), samples(hz, 60.0));

        // The tap sits 15 ms after the clip's onset_offset: outside ±10 ms, so NotDetected.
        let shift = samples(hz, 15.0);
        let start = tap.onset - pre - shift;
        let shifted = RawClip {
            format: tap.clip.format,
            start: SampleIndex(start as u64),
            samples: tap.stream[start * ch..(start + pre + post) * ch].to_vec(),
            onset_offset: pre,
        };
        let o = aug
            .augment(&tap.live, &shifted, &tap.noise, &ambient, 0xA4, 0)
            .expect("augment");
        assert_eq!(o.variants.len(), 1, "{:?}", o.params);
        assert_eq!(o.dropped.len(), VARIANTS_PER_TAP - 1);
        assert!(
            o.dropped.iter().all(|d| d.1 == VariantDrop::NotDetected),
            "{:?}",
            o.dropped
        );

        // A double onset 40 ms apart (desk taps at zone 1).
        // SPEC: AU4 expects `MultipleCandidates{2}` or a superseded candidate here. Under §1.11.2's
        // rule (candidates within ±10 ms of onset_offset) two candidates can never share the
        // window, since triggers are ≥ 30 ms apart and a superseded one is a rejection; and the
        // re-analysis cannot re-detect a tap 40 ms before t0, which lies inside the first ~101 ms
        // of the clip, before peak-picking has its 19-frame history. So the documented results
        // are: two well-localized taps keep every variant (each window holds its own onset), and
        // a live t0 placed off the second onset drops every variant as `NotDetected`
        // (evidence: G:/DeveloperData/Evidence/Qwerty/wp-b2/au4_double_onset_probe.txt).
        let fmt = format(hz, 2);
        let layout = all_real(2);
        let mut double = |first_db: f64, second_db: f64| -> Vec<AugmentOutcome> {
            let mut desk = ModalDesk::new(DeskConfig::default_for(rate(hz), 0xA44)).expect("desk");
            let lead = samples(hz, 1000.0);
            let mut x = desk.floor(lead + samples(hz, 500.0));
            for (at, db) in [(lead, first_db), (lead + samples(hz, 40.0), second_db)] {
                let t = desk.tap(&desk_spec(1, db), at);
                for (v, s) in x.iter_mut().zip(&t) {
                    *v += s;
                }
            }
            let mut det =
                Detector::new(fmt, &layout, &P, DetectorMode::CalibrationClips).expect("det");
            let mut out = Vec::new();
            push_all(&mut det, fmt, &x, 0, 480, &mut out);
            let mut extractor =
                DescriptorExtractor::new(fmt.rate, DescriptorLayout::from_channels(&layout));
            candidates(&out)
                .into_iter()
                .map(|c| {
                    let clip = c.clip.clone().expect("clip in CalibrationClips mode");
                    let live = extractor.extract(&c.analysis).expect("live descriptor");
                    aug.augment(&live, &clip, &c.analysis.noise, &ambient, 0xA4, 1)
                        .expect("augment")
                })
                .collect()
        };
        let equal = double(-30.0, -30.0);
        assert_eq!(equal.len(), 2, "both taps are live candidates");
        for o in &equal {
            assert_eq!(o.variants.len(), VARIANTS_PER_TAP, "{:?}", o.dropped);
        }
        let weak_second = double(-25.0, -40.0);
        assert_eq!(weak_second.len(), 2);
        assert_eq!(weak_second[0].variants.len(), VARIANTS_PER_TAP);
        assert_eq!(weak_second[1].variants.len(), 1);
        assert!(weak_second[1]
            .dropped
            .iter()
            .all(|d| d.1 == VariantDrop::NotDetected));
    }

    #[test]
    fn au5_tilt_fir_response_and_onset() {
        for hz in [16_000u32, 48_000, 192_000] {
            let mut aug = Augmenter::new(format(hz, 1), &all_real(1), &P).expect("augmenter");
            let fs = f64::from(hz);
            for s in [1.5, -1.5] {
                aug.design_tilt(s);
                assert_eq!(aug.taps.len(), tilt_length(rate(hz)));
                let db = |f: f64| {
                    20.0 * (response(&aug.taps, fs, f) / response(&aug.taps, fs, 1000.0)).log10()
                };
                let (up, down, low) = (db(2000.0), db(500.0), db(250.0));
                println!(
                    "AU5 {hz} Hz s = {s:+}: 2 kHz {up:+.3} dB, 500 Hz {down:+.3} dB, 250 Hz {low:+.3} dB"
                );
                assert!((up - s).abs() <= 0.1, "{hz} s={s}: 2 kHz {up}");
                assert!((down + s).abs() <= 0.1, "{hz} s={s}: 500 Hz {down}");
            }
        }
        assert_eq!(tilt_length(rate(48_000)), 385);
        assert_eq!(tilt_length(rate(192_000)), 1537);

        // A re-analyzed impulse under the tilt: the FIR is zero-phase, so the impulse's peak stays
        // on its sample (zero delay by construction), and the re-analysis still finds it.
        // SPEC: AU5 also asks that the AIC t0 move by ≤ 1 sample. With this exact design (whose
        // response matches §1.11.2's realized −1.52/+1.50/−3.12 dB), every gate-passing impulse moves
        // 2-57 samples EARLIER: the symmetric FIR's pre-ringing (h[c−1] = −13 % of h[c] at
        // s = +1.5) rises above the noise before the impulse (evidence:
        // G:/DeveloperData/Evidence/Qwerty/wp-b2/au5_tilt_t0_probe.txt). A zero-phase FIR and a
        // ≤ 1-sample AIC shift cannot both hold; the normative zero-phase design is kept, its zero
        // delay is asserted, and the AIC shift is printed.
        for hz in [16_000u32, 48_000, 192_000] {
            let fmt = format(hz, 1);
            let layout = all_real(1);
            let (mut x, _) = scene(hz, 1, 0xA5 + u64::from(hz), 1500.0, &[]);
            let impulse = samples(hz, 1000.0);
            x[impulse] += 0.5;
            let tap = live_tap(fmt, &layout, x);
            let at = impulse - tap.clip.start.0 as usize;
            let silence = vec![0.0; samples(hz, 3000.0)];
            let silent = AmbientRecording::new(fmt, &layout, silence, &[]).expect("ambient");
            let mut aug = Augmenter::new(fmt, &layout, &P).expect("augmenter");
            let mut probe = |s: f64| {
                let r = aug.render(&tap.clip, &tap.noise, &silent, s, None, 0);
                let peak = (0..r.z.len())
                    .max_by(|&a, &b| r.z[a].abs().total_cmp(&r.z[b].abs()))
                    .expect("non-empty clip");
                let found = aug
                    .reanalyze(&r.z, tap.clip.onset_offset, r.noise, 0)
                    .expect("re-analysis");
                (peak, found.expect("one candidate").onset.0 as i64)
            };
            let (peak, base) = probe(0.0);
            assert_eq!(peak, at, "{hz}: s = 0 moved the impulse");
            for s in [1.5, -1.5] {
                let (peak, t0) = probe(s);
                assert_eq!(peak, at, "{hz} s={s}: the tilt delayed the impulse");
                println!("AU5 {hz} Hz s = {s:+}: AIC t0 moved {} samples", t0 - base);
            }
        }
    }

    #[test]
    fn au6_every_axis_changes_the_descriptor_and_gain_does_not() {
        let hz = 48_000;
        let (mut aug, tap, ambient) = desk_fixture(hz, 0xA006);
        let base = variant(&mut aug, &tap.clip, &tap.noise, &ambient, 0.0, None);
        // The T14 re-analysis noise: an unmodified clip re-analyzed against the live descriptor.
        let found = aug
            .reanalyze(
                &tap.clip.samples,
                tap.clip.onset_offset,
                tap.noise.clone(),
                0,
            )
            .expect("re-analysis");
        let again = aug.describe(found).expect("re-analysis survives");
        // T14 bounds the re-analysis noise at 1e-3; the measured value is used when it is larger.
        let noise = again
            .max_abs_diff(&tap.live)
            .max(base.max_abs_diff(&tap.live))
            .max(1e-3);
        let tilt = variant(&mut aug, &tap.clip, &tap.noise, &ambient, 1.5, None);
        let twelve = Some(Decibels::new(12.0));
        let mixed = variant(&mut aug, &tap.clip, &tap.noise, &ambient, 0.0, twelve);
        let (dt, dm) = (tilt.max_abs_diff(&base), mixed.max_abs_diff(&base));
        println!("AU6 re-analysis noise {noise:e}; tilt changes {dt:e}, 12 dB mix changes {dm:e}");
        assert!(dt > 100.0 * noise, "tilt: {dt} vs noise {noise}");
        assert!(dm > 100.0 * noise, "mix: {dm} vs noise {noise}");

        for gain_db in [10.0f64, -10.0] {
            let g = 10f64.powf(gain_db / 20.0);
            let clip = RawClip {
                samples: tap
                    .clip
                    .samples
                    .iter()
                    .map(|&v| (f64::from(v) * g) as f32)
                    .collect(),
                ..tap.clip.clone()
            };
            let scale = |b: &[f64; BAND_COUNT]| b.map(|v| v * g * g);
            let noise = NoiseSnapshot {
                pooled: scale(&tap.noise.pooled),
                chosen: tap.noise.chosen.iter().map(scale).collect(),
            };
            let gained = variant(&mut aug, &clip, &noise, &ambient, 0.0, None);
            let d = gained.max_abs_diff(&base);
            println!("AU6 gain {gain_db:+} dB changes {d:e}");
            assert!(d <= 1e-5, "gain {gain_db} dB changed a dim by {d}");
        }
    }

    #[test]
    fn au7_mixing_offsets_avoid_excluded_ranges() {
        let hz = 48_000;
        let fmt = format(hz, 2);
        let layout = all_real(2);
        let (mut aug, tap, _) = desk_fixture(hz, 0xA007);
        let clip = clip_frames(rate(hz));
        let frames = samples(hz, 3000.0);
        let shape = click(hz, 0xA7);
        let (before, after) = (samples(hz, 50.0) as u64, samples(hz, 300.0) as u64);

        // One click at 1.5 s, excluded as the session excludes an ambient disturbance.
        let mut floor = ModalDesk::new(DeskConfig::default_for(rate(hz), 0xA7))
            .expect("desk")
            .floor(frames);
        let at = samples(hz, 1500.0);
        add_click(&mut floor, 2, at, &shape, 0.1);
        let range = (at as u64 - before, at as u64 + after);
        let ambient = AmbientRecording::new(fmt, &layout, floor, &[range]).expect("ambient");
        let total = frames as u64 - clip + 1;
        let forbidden = range.1 - (range.0 + 1 - clip);
        assert_eq!(ambient.usable_offsets(clip), total - forbidden);
        assert_eq!(ambient.offset_at(clip, 0), 0);
        assert_eq!(ambient.offset_at(clip, range.0 - clip), range.0 - clip);
        assert_eq!(ambient.offset_at(clip, range.0 - clip + 1), range.1);
        let mut mixed = 0;
        for item in 0..24u64 {
            let o = aug
                .augment(&tap.live, &tap.clip, &tap.noise, &ambient, 0xA7, item)
                .expect("augment");
            for p in all_params(&o).into_iter().skip(1) {
                let (o0, o1) = (p.ambient_offset, p.ambient_offset + clip);
                assert!(
                    o1 <= range.0 || o0 >= range.1,
                    "offset {o0} overlaps {range:?}"
                );
                mixed += usize::from(p.snr.is_some());
            }
        }
        assert!(mixed > 0);

        // Clicks every 0.4 s leave less than 1 s of usable offsets.
        let mut busy = ModalDesk::new(DeskConfig::default_for(rate(hz), 0xA8))
            .expect("desk")
            .floor(frames);
        let mut exclude = Vec::new();
        let mut at = samples(hz, 200.0);
        while at + shape.len() < frames {
            add_click(&mut busy, 2, at, &shape, 0.1);
            exclude.push((at as u64 - before, at as u64 + after));
            at += samples(hz, 400.0);
        }
        match AmbientRecording::new(fmt, &layout, busy, &exclude) {
            Err(AugmentError::AmbientTooDisturbed {
                usable_frames,
                needed,
            }) => {
                assert_eq!(needed, samples(hz, 1000.0) as u64);
                assert!(usable_frames < needed);
            }
            other => panic!("expected AmbientTooDisturbed, got {other:?}"),
        }
    }
}
