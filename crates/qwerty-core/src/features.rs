//! Feature extraction: a fixed-length window of audio samples in, one
//! [`FeatureVector`] out.
//!
//! The vector has three groups (see `DATA_MODEL.md` → Runtime-only types):
//! temporal shape, spectral-band distribution, and cepstral coefficients.
//! Every group is deliberately built to be *robust to how hard the desk was
//! tapped* — the discriminative signal between zones is the acoustic
//! signature of the location, not the loudness of the hit:
//!
//! - spectral bands are normalized into a distribution that sums to 1, so
//!   only the *shape* of the spectrum survives, not its magnitude;
//! - cepstral coefficients start at c1 (c0, the overall log-energy term, is
//!   dropped) for the same reason.
//!
//! Extraction is deterministic: the same input window always yields the same
//! [`FeatureVector`], bit for bit (required by `TESTING.md`). There is no RNG
//! and no reliance on uninitialized memory — every array element is written.
//!
//! Allocation policy (`PERFORMANCE.md`): the FFT plan, scratch space, and
//! working buffers are built once when a [`FeatureExtractor`] is created and
//! reused for every window, so per-tap extraction on the processing thread
//! does not allocate. [`FeatureVector`] itself is fixed-size stack arrays,
//! never a `Vec`.

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Nominal sample rate the DSP core is tuned for. Real capture (Phase 2) may
/// report a different rate; the pipeline resamples/validates against this.
/// At 44.1 kHz, [`FEATURE_WINDOW_SAMPLES`] spans ≈92.9 ms — the "90 ms
/// feature window" of `ARCHITECTURE.md`.
pub const SAMPLE_RATE_HZ: u32 = 44_100;

/// Length of the analysis window in samples. A power of two so it doubles as
/// the FFT size with no zero-padding.
pub const FEATURE_WINDOW_SAMPLES: usize = 4096;

/// Number of time-domain shape features.
pub const TEMPORAL_FEATURE_COUNT: usize = 6;

/// Number of log-spaced spectral bands.
pub const SPECTRAL_BAND_COUNT: usize = 16;

/// Number of cepstral coefficients kept (c1..=c12; c0 is dropped).
pub const CEPSTRAL_COEFF_COUNT: usize = 12;

/// Total flattened feature dimensionality.
pub const FEATURE_DIM: usize =
    TEMPORAL_FEATURE_COUNT + SPECTRAL_BAND_COUNT + CEPSTRAL_COEFF_COUNT;

/// Lowest frequency (Hz) of the spectral band bank. Below this is mostly
/// desk rumble / DC drift and is not discriminative.
const BAND_LOW_HZ: f32 = 120.0;

/// A small constant added before every logarithm / division to keep the
/// output finite for silent or near-silent windows.
const EPS: f32 = 1e-9;

/// The 90 ms window's extracted feature representation. Fixed-size arrays —
/// see `DATA_MODEL.md`. This is a runtime value and is never serialized into
/// a `Profile` directly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureVector {
    /// log-energy, zero-crossing rate, peak, crest factor, attack position,
    /// temporal centroid — in that order.
    pub temporal: [f32; TEMPORAL_FEATURE_COUNT],
    /// Normalized log-spaced band energies (sum ≈ 1 before the log step).
    pub spectral_bands: [f32; SPECTRAL_BAND_COUNT],
    /// Cepstral coefficients c1..=c12 (DCT-II of the log band energies).
    pub cepstral: [f32; CEPSTRAL_COEFF_COUNT],
}

impl FeatureVector {
    /// Flatten the three groups into one contiguous array in a stable order
    /// (temporal, then spectral, then cepstral). Used by the classifier,
    /// which treats the vector as a single point in `FEATURE_DIM` space.
    pub fn as_array(&self) -> [f32; FEATURE_DIM] {
        let mut out = [0.0f32; FEATURE_DIM];
        let mut i = 0;
        for &v in &self.temporal {
            out[i] = v;
            i += 1;
        }
        for &v in &self.spectral_bands {
            out[i] = v;
            i += 1;
        }
        for &v in &self.cepstral {
            out[i] = v;
            i += 1;
        }
        out
    }

    /// True if every component is finite (no NaN/inf). A property the
    /// extractor guarantees; asserted in tests per `TESTING.md`.
    pub fn is_finite(&self) -> bool {
        self.temporal.iter().all(|v| v.is_finite())
            && self.spectral_bands.iter().all(|v| v.is_finite())
            && self.cepstral.iter().all(|v| v.is_finite())
    }
}

/// Reusable feature extractor. Build once (it plans an FFT and precomputes
/// the Hann window and band edges), then call [`FeatureExtractor::extract`]
/// per window with no further allocation.
pub struct FeatureExtractor {
    fft: Arc<dyn Fft<f32>>,
    hann: Vec<f32>,
    /// Inclusive-exclusive FFT-bin ranges for each spectral band.
    band_bins: Vec<(usize, usize)>,
    // Reused working buffers (never reallocated after construction).
    fft_buf: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    power: Vec<f32>,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor {
    /// Construct an extractor tuned for [`FEATURE_WINDOW_SAMPLES`] windows.
    pub fn new() -> Self {
        let n = FEATURE_WINDOW_SAMPLES;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);
        let scratch_len = fft.get_inplace_scratch_len();

        // Periodic Hann window reduces spectral leakage from the abrupt
        // window edges of a transient.
        let hann: Vec<f32> = (0..n)
            .map(|i| {
                let x = std::f32::consts::PI * i as f32 / n as f32;
                let s = x.sin();
                s * s
            })
            .collect();

        let band_bins = compute_band_bins(n, SAMPLE_RATE_HZ, SPECTRAL_BAND_COUNT);

        Self {
            fft,
            hann,
            band_bins,
            fft_buf: vec![Complex::new(0.0, 0.0); n],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            power: vec![0.0; n / 2],
        }
    }

    /// Extract features from exactly [`FEATURE_WINDOW_SAMPLES`] samples.
    ///
    /// # Panics
    /// Panics if `window.len() != FEATURE_WINDOW_SAMPLES` — this is a
    /// programmer error (the caller owns windowing), not an expected runtime
    /// failure, so it is asserted rather than returned as a `Result`
    /// (`ARCHITECTURE.md` → Error handling policy).
    pub fn extract(&mut self, window: &[f32]) -> FeatureVector {
        assert_eq!(
            window.len(),
            FEATURE_WINDOW_SAMPLES,
            "feature window must be exactly {FEATURE_WINDOW_SAMPLES} samples",
        );

        let temporal = temporal_features(window);

        // Windowed FFT -> per-bin power.
        for (dst, (&s, &w)) in self.fft_buf.iter_mut().zip(window.iter().zip(&self.hann)) {
            *dst = Complex::new(s * w, 0.0);
        }
        self.fft
            .process_with_scratch(&mut self.fft_buf, &mut self.scratch);
        for (p, c) in self.power.iter_mut().zip(&self.fft_buf) {
            *p = c.norm_sqr();
        }

        // Band energies, then normalize into a distribution (force-invariant).
        let mut bands = [0.0f32; SPECTRAL_BAND_COUNT];
        for (band, &(lo, hi)) in bands.iter_mut().zip(&self.band_bins) {
            let mut acc = 0.0f32;
            for &p in &self.power[lo..hi] {
                acc += p;
            }
            *band = acc;
        }
        let total: f32 = bands.iter().sum::<f32>() + EPS;
        for b in &mut bands {
            *b /= total;
        }

        // Cepstral: DCT-II of the log band energies, coefficients c1..
        let cepstral = cepstrum(&bands);

        FeatureVector {
            temporal,
            spectral_bands: bands,
            cepstral,
        }
    }
}

/// Compute the six time-domain shape features on the raw (un-windowed) window.
fn temporal_features(x: &[f32]) -> [f32; TEMPORAL_FEATURE_COUNT] {
    let n = x.len() as f32;

    let mut sum_sq = 0.0f32;
    let mut peak = 0.0f32;
    let mut peak_idx = 0usize;
    let mut zero_crossings = 0u32;
    let mut prev = 0.0f32;
    let mut centroid_num = 0.0f32; // sum(t * x^2)

    for (i, &s) in x.iter().enumerate() {
        let sq = s * s;
        sum_sq += sq;
        centroid_num += i as f32 * sq;
        let a = s.abs();
        if a > peak {
            peak = a;
            peak_idx = i;
        }
        if (s >= 0.0) != (prev >= 0.0) && i > 0 {
            zero_crossings += 1;
        }
        prev = s;
    }

    let rms = (sum_sq / n).sqrt();
    let log_energy = (rms + EPS).ln();
    let zcr = zero_crossings as f32 / n;
    let crest = peak / (rms + EPS);
    let attack_pos = peak_idx as f32 / n; // 0 = instant attack, 1 = late peak
    let temporal_centroid = centroid_num / (sum_sq + EPS) / n; // normalized 0..1

    [log_energy, zcr, peak, crest, attack_pos, temporal_centroid]
}

/// DCT-II of the log band energies, returning coefficients c1..=cN (c0, the
/// loudness term, is dropped for force-invariance).
fn cepstrum(bands: &[f32; SPECTRAL_BAND_COUNT]) -> [f32; CEPSTRAL_COEFF_COUNT] {
    let mut log_bands = [0.0f32; SPECTRAL_BAND_COUNT];
    for (dst, &b) in log_bands.iter_mut().zip(bands) {
        *dst = (b + EPS).ln();
    }

    let m = SPECTRAL_BAND_COUNT as f32;
    let mut out = [0.0f32; CEPSTRAL_COEFF_COUNT];
    for (k_idx, o) in out.iter_mut().enumerate() {
        let k = (k_idx + 1) as f32; // start at c1
        let mut acc = 0.0f32;
        for (n_idx, &lb) in log_bands.iter().enumerate() {
            let angle = std::f32::consts::PI / m * (n_idx as f32 + 0.5) * k;
            acc += lb * angle.cos();
        }
        *o = acc;
    }
    out
}

/// Precompute the FFT-bin range covered by each log-spaced band, from
/// [`BAND_LOW_HZ`] up to Nyquist. Ranges are contiguous and clamped to valid
/// bins; each band spans at least one bin.
fn compute_band_bins(fft_size: usize, sample_rate: u32, band_count: usize) -> Vec<(usize, usize)> {
    let nyquist = sample_rate as f32 / 2.0;
    let hz_per_bin = sample_rate as f32 / fft_size as f32;
    let max_bin = fft_size / 2;

    let log_lo = BAND_LOW_HZ.ln();
    let log_hi = nyquist.ln();

    let mut edges = Vec::with_capacity(band_count + 1);
    for i in 0..=band_count {
        let frac = i as f32 / band_count as f32;
        let hz = (log_lo + (log_hi - log_lo) * frac).exp();
        let bin = (hz / hz_per_bin).round() as usize;
        edges.push(bin.min(max_bin));
    }

    let mut ranges = Vec::with_capacity(band_count);
    for i in 0..band_count {
        let lo = edges[i];
        let hi = edges[i + 1].max(lo + 1).min(max_bin);
        ranges.push((lo, hi));
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE_HZ as f32).sin())
            .collect()
    }

    #[test]
    fn dimensionality_is_stable() {
        assert_eq!(FEATURE_DIM, 34);
        let mut ex = FeatureExtractor::new();
        let fv = ex.extract(&sine(1000.0, FEATURE_WINDOW_SAMPLES));
        assert_eq!(fv.as_array().len(), FEATURE_DIM);
    }

    #[test]
    fn extraction_is_deterministic_bit_for_bit() {
        let win = sine(2000.0, FEATURE_WINDOW_SAMPLES);
        let mut a = FeatureExtractor::new();
        let mut b = FeatureExtractor::new();
        let fa = a.extract(&win);
        // Same extractor, second call: must be identical to a fresh one.
        let fa2 = a.extract(&win);
        let fb = b.extract(&win);
        assert_eq!(fa.as_array(), fa2.as_array());
        assert_eq!(fa.as_array(), fb.as_array());
    }

    #[test]
    fn output_is_always_finite() {
        let mut ex = FeatureExtractor::new();
        for input in [
            vec![0.0; FEATURE_WINDOW_SAMPLES],               // silence
            sine(440.0, FEATURE_WINDOW_SAMPLES),             // tone
            vec![1.0; FEATURE_WINDOW_SAMPLES],               // DC / clipped
        ] {
            let fv = ex.extract(&input);
            assert!(fv.is_finite(), "non-finite features for an input");
        }
    }

    #[test]
    fn spectral_bands_form_a_distribution() {
        let mut ex = FeatureExtractor::new();
        let fv = ex.extract(&sine(3000.0, FEATURE_WINDOW_SAMPLES));
        let sum: f32 = fv.spectral_bands.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "bands should sum to ~1, got {sum}");
    }

    #[test]
    fn loudness_scaling_barely_moves_shape_features() {
        // Force-invariance: the same tone at 1x and 0.25x amplitude should
        // produce near-identical spectral shape and cepstral features.
        let mut ex = FeatureExtractor::new();
        let quiet = sine(2500.0, FEATURE_WINDOW_SAMPLES);
        let loud: Vec<f32> = quiet.iter().map(|s| s * 4.0).collect();
        let fq = ex.extract(&quiet);
        let fl = ex.extract(&loud);
        for (a, b) in fq.spectral_bands.iter().zip(&fl.spectral_bands) {
            assert!((a - b).abs() < 1e-4, "spectral shape shifted with loudness");
        }
        for (a, b) in fq.cepstral.iter().zip(&fl.cepstral) {
            assert!((a - b).abs() < 1e-3, "cepstral shifted with loudness");
        }
    }
}
