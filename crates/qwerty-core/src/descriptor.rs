//! Per-candidate descriptor: the level-free feature vector the zone model classifies (§1.7).

use crate::audio::ChannelLayout;
use crate::detector::{AnalysisWindow, ANCHORED_FRAMES};
use crate::dsp::{BandLayout, Biquad, BAND_COUNT};
use crate::units::{Hertz, Milliseconds, SampleRate};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use serde::de::Error as _;
use serde::ser::{Error as _, SerializeStruct};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::sync::Arc;

pub const MONO_DIMS: usize = 38;
pub const MULTI_DIMS: usize = 10;
pub const DESCRIPTOR_DIMS: usize = MONO_DIMS + MULTI_DIMS; // 48

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DescriptorLayout {
    Mono { a: u16 },
    Stereo { a: u16, c: u16 },
}

impl DescriptorLayout {
    /// real.len()==1 -> Mono{a: real[0]}; else Stereo{a: real[0], c: *real.last()} (pair choice: OQ-7)
    pub fn from_channels(layout: &ChannelLayout) -> Self {
        match layout.real() {
            [a] => DescriptorLayout::Mono { a: *a },
            [a, .., c] => DescriptorLayout::Stereo { a: *a, c: *c },
            // INVARIANT: every ChannelLayout constructor rejects an empty `real`
            // (LayoutInvalid::NoRealChannel, LayoutError::AllChannelsSilent).
            [] => unreachable!("ChannelLayout::real is never empty"),
        }
    }

    /// 38 or 48.
    pub const fn dims(self) -> usize {
        match self {
            DescriptorLayout::Mono { .. } => MONO_DIMS,
            DescriptorLayout::Stereo { .. } => DESCRIPTOR_DIMS,
        }
    }

    /// [a] or [a, c].
    pub fn chosen(self) -> Vec<u16> {
        match self {
            DescriptorLayout::Mono { a } => vec![a],
            DescriptorLayout::Stereo { a, c } => vec![a, c],
        }
    }
}

/// values[dims()..] are exactly 0.0 for Mono. Serialized as {"layout":..,"values":[dims() floats]}
/// by a hand-written impl (serde arrays stop at 32 elements).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Descriptor {
    pub layout: DescriptorLayout,
    pub values: [f32; DESCRIPTOR_DIMS],
}

impl Descriptor {
    /// &values[..dims()].
    pub fn as_slice(&self) -> &[f32] {
        &self.values[..self.layout.dims()]
    }

    /// Every one of the 48 stored values is finite.
    pub fn is_finite(&self) -> bool {
        self.values.iter().all(|v| v.is_finite())
    }

    /// Largest |self − other| over the layout's dims; f32::INFINITY if the layouts differ.
    /// A NaN in either operand gives NaN, so a tolerance check on the result fails.
    pub fn max_abs_diff(&self, other: &Descriptor) -> f32 {
        if self.layout != other.layout {
            return f32::INFINITY;
        }
        let mut max = 0.0f32;
        for (a, b) in self.as_slice().iter().zip(other.as_slice()) {
            let d = (a - b).abs();
            if d.is_nan() {
                return f32::NAN;
            }
            max = max.max(d);
        }
        max
    }
}

impl Serialize for Descriptor {
    /// Refuses a non-finite value: serde_json would write it as `null`, and the file would then
    /// fail to load (§0.5).
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if let Some(index) = self.as_slice().iter().position(|v| !v.is_finite()) {
            return Err(S::Error::custom(format!(
                "descriptor value {index} is not finite"
            )));
        }
        let mut s = serializer.serialize_struct("Descriptor", 2)?;
        s.serialize_field("layout", &self.layout)?;
        s.serialize_field("values", self.as_slice())?;
        s.end()
    }
}

/// The unvalidated serde shape of `Descriptor`, used by deserialization only.
#[derive(Deserialize)]
struct RawDescriptor {
    layout: DescriptorLayout,
    values: Vec<f32>,
}

impl<'de> Deserialize<'de> for Descriptor {
    /// Rejects a `values` length other than `layout.dims()` and any non-finite value.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawDescriptor::deserialize(deserializer)?;
        let dims = raw.layout.dims();
        if raw.values.len() != dims {
            return Err(D::Error::custom(format!(
                "descriptor has {} values, its layout needs {dims}",
                raw.values.len()
            )));
        }
        if let Some(index) = raw.values.iter().position(|v| !v.is_finite()) {
            return Err(D::Error::custom(format!(
                "descriptor value {index} is not finite"
            )));
        }
        let mut values = [0.0f32; DESCRIPTOR_DIMS];
        values[..dims].copy_from_slice(&raw.values);
        Ok(Descriptor {
            layout: raw.layout,
            values,
        })
    }
}

/// The golden dimension names of §1.7.3, in index order.
const DESCRIPTOR_NAMES: [&str; DESCRIPTOR_DIMS] = [
    "shape_c1",
    "shape_c2",
    "shape_c3",
    "shape_c4",
    "shape_c5",
    "shape_c6",
    "shape_c7",
    "shape_c8",
    "shape_c9",
    "shape_c10",
    "shape_c11",
    "shape_c12",
    "shape_c13",
    "shape_c14",
    "shape_c15",
    "shape_c16",
    "attack_c0",
    "attack_c1",
    "attack_c2",
    "attack_c3",
    "attack_c4",
    "attack_c5",
    "attack_c6",
    "attack_c7",
    "decay_g0",
    "decay_g1",
    "decay_g2",
    "decay_g3",
    "decay_g4",
    "decay_g5",
    "decay_g6",
    "decay_g7",
    "dispersion_o0",
    "dispersion_o1",
    "dispersion_o2",
    "dispersion_o3",
    "dispersion_o4",
    "attack_crest_db",
    "ild_g0",
    "ild_g1",
    "ild_g2",
    "ild_g3",
    "ild_g4",
    "ild_g5",
    "ild_g6",
    "ild_g7",
    "itd_ms",
    "coherence",
];

/// Golden list, §1.7.3: the first `layout.dims()` names.
pub fn descriptor_names(layout: DescriptorLayout) -> Vec<&'static str> {
    DESCRIPTOR_NAMES[..layout.dims()].to_vec()
}

// ---------------------------------------------------------------------------------------------
// Extractor (WP-A1, §1.7.2-§1.7.3).
// ---------------------------------------------------------------------------------------------

/// Power floor of every band-power log (§1.7.3 FLOOR).
const POWER_FLOOR: f64 = 1e-30;
/// Noise-subtraction floor: a band keeps at least β of its power (§1.7.3 β).
const BETA: f64 = 0.05;
/// Floor of the span RMS in `attack_crest_db` (§1.7.3).
const RMS_FLOOR: f64 = 1e-10;
/// Floor of the crest ratio before its log. max|x| ≥ rms x for every signal, so it only acts on
/// digital silence, where it gives 0 dB.
const CREST_RATIO_FLOOR: f64 = 1.0;
/// Floor of |X_a·conj(X_c)| in the GCC-PHAT weighting (§1.7.3).
const PHAT_FLOOR: f64 = 1e-20;
/// Octaves of the dispersion bank (§1.7.3); the sixth value is implied by the other five.
const OCTAVES: usize = 6;
/// Shape DCT coefficients c_1..c_16, attack DCT coefficients c_0..c_7.
const SHAPE_COEFFS: usize = 16;
const ATTACK_COEFFS: usize = 8;
/// Band groups of three for `decay_g*` and `ild_g*`.
const GROUPS: usize = BAND_COUNT / 3;
/// GCC-PHAT interpolation factor (§1.7.3 r₈).
const GCC_UPSAMPLE: usize = 8;

/// Dimension offsets, in §1.7.3 order.
const SHAPE_AT: usize = 0;
const ATTACK_AT: usize = SHAPE_AT + SHAPE_COEFFS;
const DECAY_AT: usize = ATTACK_AT + ATTACK_COEFFS;
const DISPERSION_AT: usize = DECAY_AT + GROUPS;
const CREST_AT: usize = DISPERSION_AT + OCTAVES - 1;
const ILD_AT: usize = CREST_AT + 1;
const ITD_AT: usize = ILD_AT + GROUPS;
const COHERENCE_AT: usize = ITD_AT + 1;

/// Time spans of §1.7.3 that the extractor reads around t0.
const DISPERSION_SEARCH: Milliseconds = Milliseconds::new(20.0);
/// Half of the centred 1 ms boxcar: the boxcar is 2·h + 1 samples, h = samples_in(0.5 ms).
const BOXCAR_HALF: Milliseconds = Milliseconds::new(0.5);
const CREST_SPAN: Milliseconds = Milliseconds::new(10.0);
const GCC_BEFORE: Milliseconds = Milliseconds::new(1.0);
const GCC_AFTER: Milliseconds = Milliseconds::new(5.0);
/// ITD search range ±1 ms.
const ITD_RANGE: Milliseconds = Milliseconds::new(1.0);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DescriptorError {
    LayoutMismatch {
        expected: DescriptorLayout,
        got: DescriptorLayout,
    },
    SpanTooShort {
        needed: usize,
        got: usize,
    },
    NonFinite {
        index: u16,
    },
}

impl fmt::Display for DescriptorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DescriptorError::LayoutMismatch { expected, got } => {
                write!(
                    f,
                    "analysis layout {got:?} does not match the extractor's {expected:?}"
                )
            }
            DescriptorError::SpanTooShort { needed, got } => {
                write!(
                    f,
                    "analysis span covers {got} samples around t0, {needed} needed"
                )
            }
            DescriptorError::NonFinite { index } => {
                write!(f, "descriptor dimension {index} is not finite")
            }
        }
    }
}

impl std::error::Error for DescriptorError {}

/// GCC-PHAT state for a Stereo layout: plans and buffers sized once for the rate.
struct Gcc {
    before: usize,
    after: usize,
    range: usize,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
    xa: Vec<Complex<f64>>,
    xc: Vec<Complex<f64>>,
    wide: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
}

impl Gcc {
    fn new(rate: SampleRate) -> Self {
        let before = rate.samples_in(GCC_BEFORE);
        let after = rate.samples_in(GCC_AFTER);
        let n = (2 * (before + after)).next_power_of_two();
        let mut planner = FftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(n);
        let inverse = planner.plan_fft_inverse(GCC_UPSAMPLE * n);
        let scratch_len = forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len());
        let zero = Complex::new(0.0, 0.0);
        Gcc {
            before,
            after,
            range: GCC_UPSAMPLE * rate.samples_in(ITD_RANGE),
            forward,
            inverse,
            xa: vec![zero; n],
            xc: vec![zero; n],
            wide: vec![zero; GCC_UPSAMPLE * n],
            scratch: vec![zero; scratch_len],
        }
    }

    /// (itd_ms, coherence) of a against c over [t0 − 1 ms, t0 + 5 ms) (§1.7.3 dims 46-47).
    fn measure(&mut self, a: &[f32], c: &[f32], t0: usize, fs: f64) -> (f64, f64) {
        let n = self.xa.len();
        let zero = Complex::new(0.0, 0.0);
        let seg = t0 - self.before..t0 + self.after;
        for (buf, x) in [(&mut self.xa, a), (&mut self.xc, c)] {
            buf.fill(zero);
            for (b, &v) in buf.iter_mut().zip(&x[seg.clone()]) {
                b.re = f64::from(v);
            }
            self.forward.process_with_scratch(buf, &mut self.scratch);
        }
        // R = X_a·conj(X_c)/max(|X_a·conj(X_c)|, 1e-20), zero-padded to 8n for the inverse.
        let wide_len = self.wide.len();
        self.wide.fill(zero);
        let half = n / 2;
        for k in 0..n {
            let cross = self.xa[k] * self.xc[k].conj();
            let r = cross / cross.norm().max(PHAT_FLOOR);
            if k < half {
                self.wide[k] = r;
            } else if k > half {
                self.wide[wide_len - (n - k)] = r;
            } else {
                // SPEC: §1.7.3 zero-pads R for the 8n inverse. The Nyquist bin k = n/2 is split
                // between +n/2 and −n/2, the band-limited interpolation that passes through r at
                // every integer lag (r₈[8τ] = r[τ]); keeping it whole would add R[n/2]·(−1)^τ/n and
                // let |r₈| exceed 1, against `coherence` ∈ [−1, 1].
                self.wide[k] = r * 0.5;
                self.wide[wide_len - half] = r * 0.5;
            }
        }
        self.inverse
            .process_with_scratch(&mut self.wide, &mut self.scratch);
        // r₈[t] = 8·IFFT_8n(R)[t]/(8n): rustfft's inverse is unnormalized, so divide by n.
        let scale = 1.0 / n as f64;
        let r8 = |lag: isize| -> f64 {
            let i = if lag >= 0 {
                lag as usize
            } else {
                wide_len - lag.unsigned_abs()
            };
            self.wide[i].re * scale
        };
        // Ties go to the smaller |τ₈| (and to +τ₈ before −τ₈ at equal |τ₈|).
        let mut best = (0isize, r8(0));
        for d in 1..=self.range as isize {
            for lag in [d, -d] {
                let v = r8(lag);
                if v > best.1 {
                    best = (lag, v);
                }
            }
        }
        let itd_ms = best.0 as f64 / (GCC_UPSAMPLE as f64 * fs) * 1000.0;
        (itd_ms, best.1)
    }
}

/// Computes the §1.7.3 descriptor of one gate-passing candidate.
pub struct DescriptorExtractor {
    rate: SampleRate,
    layout: DescriptorLayout,
    /// DCT-II basis rows c_0..c_16 over the 24 bands (orthonormal).
    dct: [[f64; BAND_COUNT]; SHAPE_COEFFS + 1],
    /// One band-pass per dispersion octave, at rest.
    octaves: [Biquad; OCTAVES],
    boxcar_half: usize,
    search: usize,
    crest: usize,
    gcc: Option<Gcc>,
    /// x_o² prefix sums over the span.
    energy: Vec<f64>,
}

impl DescriptorExtractor {
    pub fn new(rate: SampleRate, layout: DescriptorLayout) -> Self {
        let mut dct = [[0.0; BAND_COUNT]; SHAPE_COEFFS + 1];
        let bands = BAND_COUNT as f64;
        for (k, row) in dct.iter_mut().enumerate() {
            let s = if k == 0 {
                (1.0 / bands).sqrt()
            } else {
                (2.0 / bands).sqrt()
            };
            for (b, v) in row.iter_mut().enumerate() {
                *v = s * (std::f64::consts::PI * k as f64 * (b as f64 + 0.5) / bands).cos();
            }
        }
        let f_top = BandLayout::f_top(rate).get();
        let q = std::f64::consts::SQRT_2;
        let octaves = std::array::from_fn(|o| {
            let f_hi = f_top / f64::from(1u32 << o);
            // INVARIANT: the centre f_hi/√2 ≤ 0.45·fs/√2 lies inside (0, fs/2) and √2 is a valid Q.
            let centre = Hertz::try_new(f_hi / q).expect("octave centre is finite");
            Biquad::bandpass(rate, centre, q).expect("octave centre lies below Nyquist")
        });
        let gcc = match layout {
            DescriptorLayout::Mono { .. } => None,
            DescriptorLayout::Stereo { .. } => Some(Gcc::new(rate)),
        };
        DescriptorExtractor {
            rate,
            layout,
            dct,
            octaves,
            boxcar_half: rate.samples_in(BOXCAR_HALF),
            search: rate.samples_in(DISPERSION_SEARCH),
            crest: rate.samples_in(CREST_SPAN),
            gcc,
            energy: Vec::new(),
        }
    }

    /// # Panics
    /// If `w.rate` is not the rate the extractor was built for: the band-pass bank and the GCC
    /// plans are rate-specific, and a window from another rate is a wiring error.
    pub fn extract(&mut self, w: &AnalysisWindow) -> Result<Descriptor, DescriptorError> {
        if w.layout != self.layout {
            return Err(DescriptorError::LayoutMismatch {
                expected: self.layout,
                got: w.layout,
            });
        }
        // SPEC: §1.7.2 names no error for a window from another rate; it is a wiring error, so
        // it fails loudly here rather than producing a descriptor in the wrong units.
        assert_eq!(w.rate, self.rate, "analysis window rate != extractor rate");
        let channels = self.layout.chosen().len();
        if w.span.len() != channels
            || w.chosen.len() != channels
            || w.noise.chosen.len() != channels
        {
            // SPEC: §1.7.2 names no error for a window whose per-channel vectors disagree with
            // its own layout; it is reported as a mismatch against that layout.
            return Err(DescriptorError::LayoutMismatch {
                expected: self.layout,
                got: w.layout,
            });
        }
        self.check_span(w)?;
        if let Some(index) = non_finite_input(w) {
            return Err(DescriptorError::NonFinite { index });
        }

        let mut v = [0.0f64; DESCRIPTOR_DIMS];
        self.spectral_dims(w, &mut v);
        self.dispersion_dims(w, &mut v);
        let x = &w.span[0];
        let t0 = w.span_onset;
        v[CREST_AT] = crest_db(&x[t0..t0 + self.crest]);
        if let Some(gcc) = &mut self.gcc {
            ild_dims(w, &mut v);
            let fs = f64::from(self.rate.hz());
            let (itd, coherence) = gcc.measure(&w.span[0], &w.span[1], t0, fs);
            v[ITD_AT] = itd;
            v[COHERENCE_AT] = coherence;
        }

        let mut values = [0.0f32; DESCRIPTOR_DIMS];
        for (i, (out, &value)) in values.iter_mut().zip(&v).enumerate() {
            *out = value as f32;
            if !out.is_finite() {
                return Err(DescriptorError::NonFinite { index: i as u16 });
            }
        }
        Ok(Descriptor {
            layout: self.layout,
            values,
        })
    }

    /// Every span must cover [t0 − before, t0 + after) for the time-domain dims.
    fn check_span(&self, w: &AnalysisWindow) -> Result<(), DescriptorError> {
        let h = self.boxcar_half;
        // Dispersion reads e on [t0 − 1, t0 + search + 1] (parabolic neighbours), each e a
        // centred (2h + 1)-sample boxcar; the crest reads [t0, t0 + 10 ms).
        let (mut before, mut after) = (h + 1, (self.search + h + 2).max(self.crest));
        if let Some(gcc) = &self.gcc {
            before = before.max(gcc.before);
            after = after.max(gcc.after);
        }
        let got = w
            .span
            .iter()
            .map(|s| {
                w.span_onset.min(s.len()).min(before)
                    + s.len().saturating_sub(w.span_onset).min(after)
            })
            .fold(usize::MAX, usize::min);
        if got < before + after {
            return Err(DescriptorError::SpanTooShort {
                needed: before + after,
                got,
            });
        }
        Ok(())
    }

    /// shape_c1..c16, attack_c0..c7 and decay_g0..g7 from the pooled anchored frames.
    fn spectral_dims(&self, w: &AnalysisWindow, v: &mut [f64; DESCRIPTOR_DIMS]) {
        let e = &w.pooled;
        let n = &w.noise.pooled;
        let mut l = [0.0; BAND_COUNT];
        let mut attack = [0.0; BAND_COUNT];
        for b in 0..BAND_COUNT {
            let s: f64 = e.iter().map(|f| (f[b] - n[b]).max(BETA * f[b])).sum();
            l[b] = s.max(POWER_FLOOR).ln();
            let a0 = (e[0][b] - n[b]).max(BETA * e[0][b]).max(POWER_FLOOR);
            attack[b] = a0.ln() - l[b];
        }
        for k in 1..=SHAPE_COEFFS {
            v[SHAPE_AT + k - 1] = dot(&self.dct[k], &l);
        }
        for k in 0..ATTACK_COEFFS {
            v[ATTACK_AT + k] = dot(&self.dct[k], &attack);
        }
        // decay_g*: least-squares slope of P_g(j), j = 1..6, against τ_j = j·H/fs·1000 ms.
        let hop_ms = w.geometry.hop as f64 / f64::from(w.rate.hz()) * 1000.0;
        let frames = 1..ANCHORED_FRAMES;
        let count = frames.len() as f64;
        let tau_mean = frames.clone().map(|j| j as f64 * hop_ms).sum::<f64>() / count;
        let tau_var: f64 = frames
            .clone()
            .map(|j| (j as f64 * hop_ms - tau_mean).powi(2))
            .sum();
        for g in 0..GROUPS {
            let p = |j: usize| {
                let (mut excess, mut total) = (0.0, 0.0);
                for b in 3 * g..3 * g + 3 {
                    excess += e[j][b] - n[b];
                    total += e[j][b];
                }
                10.0 * excess.max(BETA * total).max(POWER_FLOOR).log10()
            };
            let p_mean = frames.clone().map(p).sum::<f64>() / count;
            let cov: f64 = frames
                .clone()
                .map(|j| (j as f64 * hop_ms - tau_mean) * (p(j) - p_mean))
                .sum();
            v[DECAY_AT + g] = cov / tau_var;
        }
    }

    /// dispersion_o0..o4: octave envelope-peak times relative to their mean across octaves.
    fn dispersion_dims(&mut self, w: &AnalysisWindow, v: &mut [f64; DESCRIPTOR_DIMS]) {
        let arrivals = self.octave_arrivals(&w.span[0], w.span_onset);
        let mean = arrivals.iter().sum::<f64>() / OCTAVES as f64;
        for o in 0..OCTAVES - 1 {
            v[DISPERSION_AT + o] = arrivals[o] - mean;
        }
    }

    /// a_o per octave: the parabolic-interpolated argmax of the 1 ms-smoothed band-passed energy
    /// on [t0, t0 + 20 ms], in ms after t0. The caller checked the span covers what this reads.
    fn octave_arrivals(&mut self, x: &[f32], t0: usize) -> [f64; OCTAVES] {
        let h = self.boxcar_half;
        let width = (2 * h + 1) as f64;
        let ms_per_sample = 1000.0 / f64::from(self.rate.hz());
        let mut arrivals = [0.0; OCTAVES];
        for (o, arrival) in arrivals.iter_mut().enumerate() {
            let mut filter = self.octaves[o].clone();
            filter.reset();
            self.energy.clear();
            self.energy.push(0.0);
            let mut acc = 0.0;
            for &s in x {
                let y = f64::from(filter.process(s));
                acc += y * y;
                self.energy.push(acc);
            }
            // e[i] = mean of x_o² over [i − h, i + h].
            let energy = &self.energy;
            let e = |i: usize| (energy[i + h + 1] - energy[i - h]) / width;
            let mut peak = (t0, e(t0));
            for i in t0 + 1..=t0 + self.search {
                let value = e(i);
                if value > peak.1 {
                    peak = (i, value);
                }
            }
            let (i, centre) = peak;
            let (left, right) = (e(i - 1), e(i + 1));
            let curvature = left - 2.0 * centre + right;
            let delta = if curvature < 0.0 {
                (0.5 * (left - right) / curvature).clamp(-0.5, 0.5)
            } else {
                0.0
            };
            *arrival = ((i - t0) as f64 + delta) * ms_per_sample;
        }
        arrivals
    }
}

/// Σ_b row_b·values_b.
fn dot(row: &[f64; BAND_COUNT], values: &[f64; BAND_COUNT]) -> f64 {
    row.iter().zip(values).map(|(r, v)| r * v).sum()
}

/// 20·log10(max|x| / max(rms x, 1e-10)) (§1.7.3 dim 37).
fn crest_db(x: &[f32]) -> f64 {
    let (mut peak, mut sum) = (0.0f64, 0.0f64);
    for &s in x {
        let s = f64::from(s);
        peak = peak.max(s.abs());
        sum += s * s;
    }
    let rms = (sum / x.len() as f64).sqrt();
    20.0 * (peak / rms.max(RMS_FLOOR)).max(CREST_RATIO_FLOOR).log10()
}

/// ild_g0..g7 (Stereo): noise-subtracted attack power of a over c per band group, in dB.
fn ild_dims(w: &AnalysisWindow, v: &mut [f64; DESCRIPTOR_DIMS]) {
    let attack = |c: usize, g: usize| -> f64 {
        let (e, n) = (&w.chosen[c], &w.noise.chosen[c]);
        let mut sum = 0.0;
        for b in 3 * g..3 * g + 3 {
            for frame in &e[..3] {
                sum += (frame[b] - n[b]).max(BETA * frame[b]);
            }
        }
        sum.max(POWER_FLOOR)
    };
    for g in 0..GROUPS {
        v[ILD_AT + g] = 10.0 * (attack(0, g).log10() - attack(1, g).log10());
    }
}

// SPEC: `NonFinite.index` is a descriptor dimension. A non-finite input is reported as the first
// dimension that reads it (pooled bands → 0, channel a's span → 32, chosen-channel bands → 38,
// channel c's span → 46), so a NaN anywhere in the window is an error even where an argmax or a
// max() would have stepped over it.
fn non_finite_input(w: &AnalysisWindow) -> Option<u16> {
    let bands_finite = |b: &[f64; BAND_COUNT]| b.iter().all(|v| v.is_finite());
    let samples_finite = |s: &[f32]| s.iter().all(|v| v.is_finite());
    if !(w.pooled.iter().all(bands_finite) && bands_finite(&w.noise.pooled)) {
        return Some(SHAPE_AT as u16);
    }
    if !samples_finite(&w.span[0]) {
        return Some(DISPERSION_AT as u16);
    }
    if w.span.len() > 1 {
        let chosen_finite = w.chosen.iter().all(|c| c.iter().all(bands_finite))
            && w.noise.chosen.iter().all(bands_finite);
        if !chosen_finite {
            return Some(ILD_AT as u16);
        }
        if !samples_finite(&w.span[1]) {
            return Some(ITD_AT as u16);
        }
    }
    None
}

#[cfg(test)]
mod type_tests {
    use super::*;
    use crate::units::ChannelCount;

    fn mono() -> Descriptor {
        let mut values = [0.0f32; DESCRIPTOR_DIMS];
        for (i, v) in values[..MONO_DIMS].iter_mut().enumerate() {
            *v = i as f32 * 0.25 - 3.0;
        }
        Descriptor {
            layout: DescriptorLayout::Mono { a: 0 },
            values,
        }
    }

    #[test]
    fn descriptor_serde_round_trips_and_validates() {
        let d = mono();
        let json = serde_json::to_string(&d).expect("serialize");
        let back: Descriptor = serde_json::from_str(&json).expect("round-trip");
        assert_eq!(back, d);
        assert_eq!(back.as_slice().len(), MONO_DIMS);

        let value: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(value["values"].as_array().map(Vec::len), Some(MONO_DIMS));

        // Wrong length for the layout.
        let short = json.replacen("[-3.0,", "[", 1);
        assert!(serde_json::from_str::<Descriptor>(&short).is_err());
        // Out of f32 range is non-finite after the cast, and refused.
        let huge = json.replacen("[-3.0,", "[1e39,", 1);
        assert!(serde_json::from_str::<Descriptor>(&huge).is_err());
        // A non-finite value cannot be serialized.
        let mut bad = d;
        bad.values[3] = f32::NAN;
        assert!(!bad.is_finite());
        assert!(serde_json::to_string(&bad).is_err());
    }

    #[test]
    fn descriptor_layout_and_diff() {
        let stereo = ChannelLayout::all_real(ChannelCount::new(4).expect("count"));
        let layout = DescriptorLayout::from_channels(&stereo);
        assert_eq!(layout, DescriptorLayout::Stereo { a: 0, c: 3 });
        assert_eq!(layout.dims(), DESCRIPTOR_DIMS);
        assert_eq!(layout.chosen(), vec![0, 3]);
        let mono_layout = DescriptorLayout::from_channels(&ChannelLayout::all_real(
            ChannelCount::new(1).expect("count"),
        ));
        assert_eq!(mono_layout, DescriptorLayout::Mono { a: 0 });
        assert_eq!(mono_layout.chosen(), vec![0]);

        let names = descriptor_names(layout);
        assert_eq!(names.len(), DESCRIPTOR_DIMS);
        assert_eq!(descriptor_names(mono_layout).len(), MONO_DIMS);
        assert_eq!(names[MONO_DIMS - 1], "attack_crest_db");
        assert_eq!(names[DESCRIPTOR_DIMS - 1], "coherence");

        let a = mono();
        let mut b = a;
        b.values[5] += 0.5;
        assert_eq!(a.max_abs_diff(&b), 0.5);
        b.values[6] = f32::NAN;
        assert!(a.max_abs_diff(&b).is_nan());
        let other = Descriptor {
            layout: DescriptorLayout::Mono { a: 1 },
            values: a.values,
        };
        assert_eq!(a.max_abs_diff(&other), f32::INFINITY);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::testkit::*;
    use crate::detector::{
        Candidate, Detector, DetectorMode, DetectorParams, NoiseSnapshot, SPAN_AFTER, SPAN_BEFORE,
    };
    use crate::dsp::FrameGeometry;
    use crate::rng::Rng64;
    use std::collections::BTreeSet;
    use std::f64::consts::TAU;

    const P: DetectorParams = DetectorParams::DEFAULT;

    /// Gate-passing candidates of a detector run over interleaved `x` in 480-frame blocks.
    fn run(hz: u32, channels: u16, x: &[f32], mode: DetectorMode) -> Vec<Candidate> {
        let fmt = format(hz, channels);
        let mut det = Detector::new(fmt, &all_real(channels), &P, mode).expect("detector");
        let mut out = Vec::new();
        push_all(&mut det, fmt, x, 0, 480, &mut out);
        candidates(&out).into_iter().cloned().collect()
    }

    fn extract(c: &Candidate) -> Result<Descriptor, DescriptorError> {
        DescriptorExtractor::new(c.analysis.rate, c.analysis.layout).extract(&c.analysis)
    }

    fn index_of(name: &str) -> usize {
        DESCRIPTOR_NAMES
            .iter()
            .position(|n| *n == name)
            .expect("golden name")
    }

    /// Detector-length span (SPAN_BEFORE + SPAN_AFTER) at `hz`.
    fn span_len(hz: u32) -> usize {
        samples(hz, SPAN_BEFORE.get()) + samples(hz, SPAN_AFTER.get())
    }

    /// A hand-made mono window at `hz` around `span`, whose t0 is samples_in(SPAN_BEFORE).
    fn window(hz: u32, span: Vec<f32>) -> AnalysisWindow {
        let r = rate(hz);
        AnalysisWindow {
            rate: r,
            geometry: FrameGeometry::for_rate(r),
            layout: DescriptorLayout::Mono { a: 0 },
            pooled: [[1.0; BAND_COUNT]; ANCHORED_FRAMES],
            chosen: vec![[[1.0; BAND_COUNT]; ANCHORED_FRAMES]],
            noise: NoiseSnapshot {
                pooled: [0.1; BAND_COUNT],
                chosen: vec![[0.1; BAND_COUNT]],
            },
            span: vec![span],
            span_onset: samples(hz, SPAN_BEFORE.get()),
        }
    }

    #[test]
    fn f1_names_match_the_golden_table() {
        let mut golden: Vec<String> = Vec::new();
        golden.extend((1..=16).map(|k| format!("shape_c{k}")));
        golden.extend((0..8).map(|k| format!("attack_c{k}")));
        golden.extend((0..8).map(|g| format!("decay_g{g}")));
        golden.extend((0..5).map(|o| format!("dispersion_o{o}")));
        golden.push("attack_crest_db".to_owned());
        golden.extend((0..8).map(|g| format!("ild_g{g}")));
        golden.push("itd_ms".to_owned());
        golden.push("coherence".to_owned());
        let mono = descriptor_names(DescriptorLayout::Mono { a: 0 });
        let stereo = descriptor_names(DescriptorLayout::Stereo { a: 0, c: 1 });
        assert_eq!(mono.len(), MONO_DIMS);
        assert_eq!(stereo.len(), DESCRIPTOR_DIMS);
        assert_eq!(stereo, golden);
        assert_eq!(mono, golden[..MONO_DIMS]);
        assert_eq!(
            stereo.iter().collect::<BTreeSet<_>>().len(),
            DESCRIPTOR_DIMS
        );
    }

    #[test]
    fn f2_input_gain_leaves_the_descriptor_unchanged() {
        let hz = 48_000;
        let events = [
            (500.0, TapKind::Partials, 60.0),
            (1100.0, TapKind::Click, 65.0),
            (1700.0, TapKind::Thud, 60.0),
        ];
        let (x, _) = scene(hz, 1, 0xF2, 2300.0, &events);
        let reference: Vec<Descriptor> = run(hz, 1, &x, DetectorMode::Live)
            .iter()
            .map(|c| extract(c).expect("descriptor"))
            .collect();
        assert_eq!(reference.len(), events.len());
        for gain_db in [20.0f64, -20.0] {
            let g = 10f64.powf(gain_db / 20.0);
            let scaled: Vec<f32> = x.iter().map(|&v| (f64::from(v) * g) as f32).collect();
            let got = run(hz, 1, &scaled, DetectorMode::Live);
            assert_eq!(got.len(), reference.len(), "{gain_db} dB");
            for (c, r) in got.iter().zip(&reference) {
                let diff = extract(c).expect("descriptor").max_abs_diff(r);
                println!("F2 {gain_db:+} dB: max |Δ| = {diff:.2e}");
                assert!(diff <= 1e-3, "{gain_db} dB: max |Δ| = {diff}");
            }
        }
    }

    #[test]
    fn f5_no_dimension_carries_an_absolute_level() {
        // Structural: no DCT c0 of L, no RMS and no peak level among the dimensions.
        let names = descriptor_names(DescriptorLayout::Stereo { a: 0, c: 1 });
        assert!(!names.contains(&"shape_c0"));
        for banned in ["rms", "peak", "level", "loud", "energy", "power", "gain"] {
            assert!(
                names.iter().all(|n| !n.contains(banned)),
                "a dimension name contains {banned:?}"
            );
        }
        // The extractor is homogeneous of degree 0: scaling the window's powers by g² and its
        // samples by g changes no dimension.
        let hz = 48_000;
        let (x, _) = scene(hz, 2, 0xF5, 1000.0, &[(500.0, TapKind::Thud, 40.0)]);
        let cands = run(hz, 2, &x, DetectorMode::Live);
        assert_eq!(cands.len(), 1);
        let w = &cands[0].analysis;
        let reference = extract(&cands[0]).expect("descriptor");
        for g in [4.0f64, 0.01, 30.0] {
            let p = g * g;
            let scale_bands = |b: &[f64; BAND_COUNT]| b.map(|v| v * p);
            let scaled = AnalysisWindow {
                pooled: w.pooled.map(|f| scale_bands(&f)),
                chosen: w
                    .chosen
                    .iter()
                    .map(|c| c.map(|f| scale_bands(&f)))
                    .collect(),
                noise: NoiseSnapshot {
                    pooled: scale_bands(&w.noise.pooled),
                    chosen: w.noise.chosen.iter().map(scale_bands).collect(),
                },
                span: w
                    .span
                    .iter()
                    .map(|s| s.iter().map(|&v| (f64::from(v) * g) as f32).collect())
                    .collect(),
                ..w.clone()
            };
            let d = DescriptorExtractor::new(w.rate, w.layout)
                .extract(&scaled)
                .expect("descriptor");
            let diff = d.max_abs_diff(&reference);
            assert!(diff <= 1e-4, "gain {g}: max |Δ| = {diff}");
        }
    }

    /// x delayed by `delay` samples through an FFT phase shift (a band-limited fractional delay).
    fn fractional_delay(x: &[f64], delay: f64) -> Vec<f64> {
        let n = (x.len() + 1024).next_power_of_two();
        let mut planner = FftPlanner::<f64>::new();
        let mut buf: Vec<Complex<f64>> = x
            .iter()
            .map(|&v| Complex::new(v, 0.0))
            .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
            .take(n)
            .collect();
        planner.plan_fft_forward(n).process(&mut buf);
        for (k, b) in buf.iter_mut().enumerate() {
            let f = if k <= n / 2 {
                k as f64
            } else {
                k as f64 - n as f64
            };
            let phase = TAU * f * delay / n as f64;
            if k == n / 2 {
                *b *= phase.cos();
            } else {
                *b *= Complex::from_polar(1.0, -phase);
            }
        }
        planner.plan_fft_inverse(n).process(&mut buf);
        buf[..x.len()].iter().map(|c| c.re / n as f64).collect()
    }

    /// Interleaves channel a and channel c.
    fn stereo(a: &[f64], c: &[f64]) -> Vec<f32> {
        a.iter()
            .zip(c)
            .flat_map(|(&a, &c)| [a as f32, c as f32])
            .collect()
    }

    #[test]
    fn f6_stereo_layout_itd_and_ild() {
        // L = R gives a Mono layout.
        let (mono, _) = scene(48_000, 1, 0xF6, 2500.0, &[(1000.0, TapKind::Click, 40.0)]);
        let same: Vec<f32> = mono.iter().flat_map(|&v| [v, v]).collect();
        let layout = ChannelLayout::analyze(format(48_000, 2), &same).expect("layout");
        assert_eq!(layout.real(), &[0]);
        assert_eq!(
            DescriptorLayout::from_channels(&layout),
            DescriptorLayout::Mono { a: 0 }
        );

        let (itd, coherence) = (index_of("itd_ms"), index_of("coherence"));
        let ild = index_of("ild_g0");
        for hz in [44_100u32, 48_000] {
            let events = [(800.0, TapKind::Click, 40.0)];
            let (x, _) = scene(hz, 1, 0xF6 + u64::from(hz), 1500.0, &events);
            let x: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
            // a lags c by 0.30 ms, a fractional delay at both rates.
            let delayed = fractional_delay(&x, 0.30e-3 * f64::from(hz));
            let cands = run(hz, 2, &stereo(&delayed, &x), DetectorMode::Live);
            assert_eq!(cands.len(), 1, "{hz}");
            let d = extract(&cands[0]).expect("descriptor");
            println!(
                "F6 {hz}: itd {:.4} ms, coherence {:.4}",
                d.values[itd], d.values[coherence]
            );
            assert!(
                (d.values[itd] - 0.30).abs() <= 0.01,
                "{hz}: itd {}",
                d.values[itd]
            );
            assert!(
                d.values[coherence] >= 0.9,
                "{hz}: coherence {}",
                d.values[coherence]
            );

            // +6 dB on channel a.
            let g = 10f64.powf(6.0 / 20.0);
            let louder: Vec<f64> = x.iter().map(|v| v * g).collect();
            let cands = run(hz, 2, &stereo(&louder, &x), DetectorMode::Live);
            assert_eq!(cands.len(), 1, "{hz}");
            let d = extract(&cands[0]).expect("descriptor");
            for (gi, v) in d.values[ild..ild + GROUPS].iter().enumerate() {
                assert!((v - 6.0).abs() <= 0.2, "{hz}: ild_g{gi} = {v}");
            }
        }
    }

    /// Group delay of `filter` at `f` Hz in ms, from its impulse response:
    /// Re(Σ n·h[n]·e^{−iωn} / Σ h[n]·e^{−iωn}).
    fn group_delay_ms(mut filter: Biquad, hz: u32, f: f64) -> f64 {
        let fs = f64::from(hz);
        let w = TAU * f / fs;
        let (mut h, mut nh) = (Complex::new(0.0, 0.0), Complex::new(0.0, 0.0));
        for n in 0..hz as usize {
            let y = f64::from(filter.process(if n == 0 { 1.0 } else { 0.0 }));
            let e = Complex::from_polar(1.0, -w * n as f64);
            h += e * y;
            nh += e * (y * n as f64);
        }
        (nh / h).re / fs * 1000.0
    }

    #[test]
    fn f7_octave_bursts_give_their_relative_delays() {
        // Octave o gets a Gaussian tone burst (σ = 0.7 ms) at its band-pass centre, peaking
        // 4 ms + 2·o ms after t0. Its expected arrival is that delay plus the band-pass group
        // delay at the centre, measured on the filter's impulse response.
        //
        // The 1 ms boxcar leaves a ripple at 2·fc in the octaves below ~1.4 kHz, so a_o snaps to
        // the nearest crest of the smoothed squared carrier. The carrier phase puts such a crest
        // at the expected peak (cos where the boxcar's response at 2·fc is positive, sin where it
        // is negative); any other phase is off by up to a quarter carrier period by construction.
        //
        // 16 kHz is not tested: its lowest octave is centred at 159 Hz, where the 2 ms spacing is
        // a third of a carrier period, so neighbouring bursts cannot be told apart in time.
        for hz in [32_000u32, 44_100, 48_000, 96_000] {
            let r = rate(hz);
            let fs = f64::from(hz);
            let t0 = samples(hz, SPAN_BEFORE.get());
            let f_top = BandLayout::f_top(r).get();
            let sigma = 0.7e-3;
            let width = (2 * samples(hz, BOXCAR_HALF.get()) + 1) as f64;
            let boxcar = |f: f64| {
                let x = std::f64::consts::PI * f / fs;
                (x * width).sin() / (width * x.sin())
            };
            let mut x = vec![0.0f64; span_len(hz)];
            let mut expected = [0.0; OCTAVES];
            for (o, e) in expected.iter_mut().enumerate() {
                let fc = f_top / f64::from(1u32 << o) / std::f64::consts::SQRT_2;
                let filter = Biquad::bandpass(r, Hertz::new(fc), std::f64::consts::SQRT_2)
                    .expect("octave filter");
                let gd = group_delay_ms(filter, hz, fc);
                let delay = 2.0 * o as f64;
                *e = delay + gd;
                let centre = (4.0 + delay) * 1e-3;
                let crest = centre + gd * 1e-3;
                let cosine = boxcar(2.0 * fc) >= 0.0;
                for (i, v) in x.iter_mut().enumerate() {
                    let t = (i as f64 - t0 as f64) / fs;
                    let phase = TAU * fc * (t - crest);
                    let carrier = if cosine { phase.cos() } else { phase.sin() };
                    *v += (-0.5 * ((t - centre) / sigma).powi(2)).exp() * carrier;
                }
            }
            let mean = expected.iter().sum::<f64>() / OCTAVES as f64;
            let w = window(hz, to_f32(&x));
            let d = DescriptorExtractor::new(r, w.layout)
                .extract(&w)
                .expect("descriptor");
            let first = index_of("dispersion_o0");
            for (o, e) in expected.iter().enumerate().take(OCTAVES - 1) {
                let want = e - mean;
                let got = f64::from(d.values[first + o]);
                println!("F7 {hz} o{o}: {got:+.4} ms (expected {want:+.4})");
                assert!((got - want).abs() <= 0.25, "{hz} o{o}: {got} vs {want}");
            }
        }
    }

    /// A broadband burst with a 2.5 ms attack: white noise × (t/τ)·e^{1 − t/τ}.
    fn slow_burst(hz: u32, rng: &mut Rng64) -> Vec<f64> {
        let tau = 2.5e-3 * f64::from(hz);
        (0..samples(hz, 60.0))
            .map(|i| {
                let u = i as f64 / tau;
                rng.normal() * u * (1.0 - u).exp()
            })
            .collect()
    }

    #[test]
    fn f8_anchor_jitter_leaves_dispersion_unchanged() {
        for hz in [48_000u32, 16_000] {
            let mut rng = Rng64::new(0xF8 + u64::from(hz));
            let n = samples(hz, 2600.0);
            let mut x = NoiseGen::new(0xF80, Colour::White, NOISE_RMS).fill(n);
            for at in [600.0, 1300.0, 2000.0] {
                let burst = slow_burst(hz, &mut rng);
                add(&mut x, samples(hz, at), &burst, amp(45.0));
            }
            let clips = run(hz, 1, &to_f32(&x), DetectorMode::CalibrationClips);
            assert_eq!(clips.len(), 3, "{hz}");
            let first = index_of("dispersion_o0");
            for c in &clips {
                let clip = c
                    .clip
                    .as_ref()
                    .expect("CalibrationClips candidates carry a clip");
                let reanalyze = |jitter: i32| -> Descriptor {
                    let mode = DetectorMode::FrozenNoise {
                        noise: c.analysis.noise.clone(),
                        anchor_jitter: jitter,
                    };
                    let got = run(hz, 1, &clip.samples, mode);
                    assert_eq!(got.len(), 1, "{hz} jitter {jitter}");
                    extract(&got[0]).expect("descriptor")
                };
                let reference = reanalyze(0);
                for ms in [0.2, 1.0] {
                    let j = samples(hz, ms) as i32;
                    for jitter in [j, -j] {
                        let d = reanalyze(jitter);
                        for o in 0..OCTAVES - 1 {
                            let delta = (d.values[first + o] - reference.values[first + o]).abs();
                            assert!(delta <= 0.01, "{hz} jitter {jitter}: o{o} moved {delta} ms");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn f9_sine_crest_is_3_db() {
        // The crest window is 10 ms, so each sine has ≥ 10 periods in it: with fewer, the RMS of
        // the partial period depends on the phase (440 Hz, 4.4 periods, gives 3.10 dB).
        for hz in [16_000u32, 48_000, 192_000] {
            for f in [1000.0, 2500.0, 3170.0] {
                let fs = f64::from(hz);
                let x: Vec<f64> = (0..span_len(hz))
                    .map(|i| 0.3 * (TAU * f * i as f64 / fs + 0.4).sin())
                    .collect();
                let w = window(hz, to_f32(&x));
                let d = DescriptorExtractor::new(w.rate, w.layout)
                    .extract(&w)
                    .expect("descriptor");
                let crest = d.values[index_of("attack_crest_db")];
                assert!(
                    (crest - 3.01).abs() <= 0.05,
                    "{hz} Hz, {f} Hz sine: {crest} dB"
                );
            }
        }
    }

    #[test]
    fn f10_non_finite_input_is_an_error() {
        let hz = 48_000;
        let (x, _) = scene(hz, 2, 0xF10, 1000.0, &[(500.0, TapKind::Partials, 40.0)]);
        let cands = run(hz, 2, &x, DetectorMode::Live);
        assert_eq!(cands.len(), 1);
        let w = &cands[0].analysis;
        let mut ex = DescriptorExtractor::new(w.rate, w.layout);
        assert!(ex.extract(w).expect("clean window").is_finite());
        let len = w.span[0].len();
        for ch in 0..2 {
            for at in [0, w.span_onset, w.span_onset + 100, len - 1] {
                for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    let mut broken = w.clone();
                    broken.span[ch][at] = bad;
                    let got = ex.extract(&broken);
                    assert!(
                        matches!(got, Err(DescriptorError::NonFinite { .. })),
                        "channel {ch}, sample {at}, {bad}: {got:?}"
                    );
                }
            }
        }
        let mut broken = w.clone();
        broken.pooled[3][7] = f64::NAN;
        assert_eq!(
            ex.extract(&broken),
            Err(DescriptorError::NonFinite { index: 0 })
        );
        let mut broken = w.clone();
        broken.noise.chosen[1][2] = f64::INFINITY;
        assert_eq!(
            ex.extract(&broken),
            Err(DescriptorError::NonFinite { index: 38 })
        );
        // Failures leave the extractor's state clean.
        assert_eq!(ex.extract(w), extract(&cands[0]));
    }

    #[test]
    fn extract_rejects_a_foreign_layout_and_a_short_span() {
        let hz = 48_000;
        let w = window(hz, vec![0.1; span_len(hz)]);
        let stereo = DescriptorLayout::Stereo { a: 0, c: 1 };
        assert_eq!(
            DescriptorExtractor::new(w.rate, stereo).extract(&w),
            Err(DescriptorError::LayoutMismatch {
                expected: stereo,
                got: DescriptorLayout::Mono { a: 0 },
            })
        );
        let mut short = w.clone();
        short.span[0].truncate(samples(hz, 4.0) + samples(hz, 15.0));
        let got = DescriptorExtractor::new(w.rate, w.layout).extract(&short);
        assert!(
            matches!(got, Err(DescriptorError::SpanTooShort { .. })),
            "{got:?}"
        );
    }
}
