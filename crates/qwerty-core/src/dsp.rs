//! DSP building blocks: biquads, the band layout and spectral helpers (§1.5).

use crate::units::{Hertz, Milliseconds, SampleRate};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// RBJ-cookbook biquad in transposed direct form II, f64 coefficients and state (§1.5.1).
#[derive(Debug, Clone)]
pub struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DspError {
    CutoffOutOfRange { hz: f64, nyquist: f64 },
    BadQ { q: f64 },
}

impl fmt::Display for DspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DspError::CutoffOutOfRange { hz, nyquist } => {
                write!(f, "filter frequency {hz} Hz is outside (0, {nyquist}) Hz")
            }
            DspError::BadQ { q } => write!(f, "filter Q {q} must be finite and positive"),
        }
    }
}

impl std::error::Error for DspError {}

impl Biquad {
    /// 2nd-order high-pass, Q = 1/√2 (Butterworth). w0 = 2π·fc/fs, α = sin w0/(2Q),
    /// b = [(1+cos w0)/2, −(1+cos w0), (1+cos w0)/2]/a0, a = [−2cos w0, 1−α]/a0, a0 = 1+α.
    pub fn highpass(rate: SampleRate, cutoff: Hertz) -> Result<Self, DspError> {
        let (sin, cos) = angular(rate, cutoff)?.sin_cos();
        let alpha = sin / (2.0 * std::f64::consts::FRAC_1_SQRT_2);
        let a0 = 1.0 + alpha;
        Ok(Biquad::from_coefficients(
            [(1.0 + cos) / 2.0, -(1.0 + cos), (1.0 + cos) / 2.0],
            [a0, -2.0 * cos, 1.0 - alpha],
        ))
    }

    /// 2nd-order band-pass (constant 0 dB peak gain), centre fc, Q = fc/bandwidth.
    /// b = [α, 0, −α]/a0, a = [−2cos w0, 1−α]/a0, a0 = 1+α, α = sin w0/(2Q).
    pub fn bandpass(rate: SampleRate, centre: Hertz, q: f64) -> Result<Self, DspError> {
        if !(q.is_finite() && q > 0.0) {
            return Err(DspError::BadQ { q });
        }
        let (sin, cos) = angular(rate, centre)?.sin_cos();
        let alpha = sin / (2.0 * q);
        Ok(Biquad::from_coefficients(
            [alpha, 0.0, -alpha],
            [1.0 + alpha, -2.0 * cos, 1.0 - alpha],
        ))
    }

    fn from_coefficients(b: [f64; 3], a: [f64; 3]) -> Self {
        Biquad {
            b0: b[0] / a[0],
            b1: b[1] / a[0],
            b2: b[2] / a[0],
            a1: a[1] / a[0],
            a2: a[2] / a[0],
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// y = b0·x + z1; z1 = b1·x − a1·y + z2; z2 = b2·x − a2·y
    pub fn process(&mut self, x: f32) -> f32 {
        let x = f64::from(x);
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y as f32
    }

    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

/// w0 = 2π·f/fs, after checking 0 < f < fs/2.
fn angular(rate: SampleRate, freq: Hertz) -> Result<f64, DspError> {
    let fs = f64::from(rate.hz());
    let nyquist = fs / 2.0;
    let hz = freq.get();
    if !(hz > 0.0 && hz < nyquist) {
        return Err(DspError::CutoffOutOfRange { hz, nyquist });
    }
    Ok(std::f64::consts::TAU * hz / fs)
}

/// Frame geometry (§1.5.2): L even, H = L/2, N = next_pow2(L), zero-padded at the end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameGeometry {
    pub len: usize,
    pub hop: usize,
    pub fft_len: usize,
}

impl FrameGeometry {
    pub const FRAME: Milliseconds = Milliseconds::new(10.667);

    /// len = 2·round(fs·FRAME/2), hop = len/2, fft_len = len.next_power_of_two()
    pub fn for_rate(rate: SampleRate) -> Self {
        let half = rate.samples_in(Milliseconds::new(Self::FRAME.get() / 2.0));
        let len = 2 * half;
        FrameGeometry {
            len,
            hop: half,
            fft_len: len.next_power_of_two(),
        }
    }

    pub fn hop_duration(self, rate: SampleRate) -> Milliseconds {
        rate.duration_of(self.hop as u64)
    }
}

/// Windowed power spectrum of one frame (§1.5.3): periodic Hann, f32 complex FFT of size N,
/// P[k] = |X[k]|²/U for k = 0..=N/2 with U = Σw², so white noise of variance σ² gives E[P[k]] = σ².
pub struct PowerSpectrum {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    u: f64,
    scratch: Vec<Complex<f32>>,
    buf: Vec<Complex<f32>>,
}

impl PowerSpectrum {
    pub fn new(geom: FrameGeometry) -> Self {
        let fft = FftPlanner::<f32>::new().plan_fft_forward(geom.fft_len);
        let l = geom.len as f64;
        let window: Vec<f32> = (0..geom.len)
            .map(|n| (0.5 - 0.5 * (std::f64::consts::TAU * n as f64 / l).cos()) as f32)
            .collect();
        let u = window.iter().map(|&w| f64::from(w) * f64::from(w)).sum();
        let scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        PowerSpectrum {
            fft,
            window,
            u,
            scratch,
            buf: vec![Complex::new(0.0, 0.0); geom.fft_len],
        }
    }

    /// frame.len() == geom.len, out.len() == geom.fft_len/2+1; no allocation.
    pub fn compute(&mut self, frame: &[f32], out: &mut [f64]) {
        assert_eq!(frame.len(), self.window.len(), "frame length != geometry");
        assert_eq!(out.len(), self.buf.len() / 2 + 1, "power length != N/2+1");
        let (head, tail) = self.buf.split_at_mut(frame.len());
        for ((b, &x), &w) in head.iter_mut().zip(frame).zip(&self.window) {
            *b = Complex::new(x * w, 0.0);
        }
        tail.fill(Complex::new(0.0, 0.0));
        self.fft
            .process_with_scratch(&mut self.buf, &mut self.scratch);
        for (p, x) in out.iter_mut().zip(&self.buf) {
            let (re, im) = (f64::from(x.re), f64::from(x.im));
            *p = (re * re + im * im) / self.u;
        }
    }
}

pub const BAND_COUNT: usize = 24;
pub const BAND_LOW: Hertz = Hertz::new(150.0);
pub const BAND_HIGH_CAP: Hertz = Hertz::new(16_000.0);
pub const BAND_HIGH_FRACTION: f64 = 0.45;

/// 24 triangular bands over 26 log-spaced edges from 150 Hz to f_top, each half-width floored at
/// one FFT bin (deviation b), weights normalized to sum to 1 (§1.5.4).
pub struct BandLayout {
    pub centres: [Hertz; BAND_COUNT],
    weights: Vec<(usize /*first bin*/, Vec<f64>)>,
}

impl BandLayout {
    pub fn new(rate: SampleRate, geom: FrameGeometry) -> Self {
        let f_top = Self::f_top(rate).get();
        let low = BAND_LOW.get();
        let edges: Vec<f64> = (0..BAND_COUNT + 2)
            .map(|i| low * (f_top / low).powf(i as f64 / (BAND_COUNT + 1) as f64))
            .collect();
        let df = f64::from(rate.hz()) / geom.fft_len as f64;
        let bins = geom.fft_len / 2 + 1;
        let mut centres = [Hertz::new(0.0); BAND_COUNT];
        let mut weights = Vec::with_capacity(BAND_COUNT);
        for (b, centre_slot) in centres.iter_mut().enumerate() {
            let centre = edges[b + 1];
            let lower = edges[b].min(centre - df);
            let upper = edges[b + 2].max(centre + df);
            let raw: Vec<f64> = (0..bins)
                .map(|k| {
                    let f = k as f64 * df;
                    ((f - lower) / (centre - lower))
                        .min((upper - f) / (upper - centre))
                        .max(0.0)
                })
                .collect();
            // INVARIANT: (lower, upper) is at least 2Δf wide and lies inside (0, fs/2], because
            // f_top + Δf ≤ 0.45·fs + fs/256 < fs/2, so some bin falls strictly inside it and has
            // a positive weight.
            let first = raw
                .iter()
                .position(|&w| w > 0.0)
                .expect("band covers a bin");
            let last = raw
                .iter()
                .rposition(|&w| w > 0.0)
                .expect("band covers a bin");
            let run = &raw[first..=last];
            let total: f64 = run.iter().sum();
            weights.push((first, run.iter().map(|w| w / total).collect()));
            *centre_slot = finite_hz(centre);
        }
        BandLayout { centres, weights }
    }

    /// out[b] = Σ_k w_b[k]·P[k], the weighted mean of P over band b; no allocation.
    pub fn apply(&self, power: &[f64], out: &mut [f64; BAND_COUNT]) {
        for (o, (first, w)) in out.iter_mut().zip(&self.weights) {
            *o = w
                .iter()
                .zip(&power[*first..*first + w.len()])
                .map(|(w, p)| w * p)
                .sum();
        }
    }

    /// min(16 kHz, 0.45·fs)
    pub fn f_top(rate: SampleRate) -> Hertz {
        finite_hz((BAND_HIGH_FRACTION * f64::from(rate.hz())).min(BAND_HIGH_CAP.get()))
    }

    /// (first bin, weights) per band, for tests and diagnostics.
    #[cfg(test)]
    pub(crate) fn weights(&self) -> &[(usize, Vec<f64>)] {
        &self.weights
    }
}

/// A frequency computed from finite operands.
fn finite_hz(hz: f64) -> Hertz {
    // INVARIANT: every caller passes a product, power or min of finite rates and constants.
    Hertz::try_new(hz).expect("band frequencies are finite")
}

/// Linear convolution through one complex FFT (§1.5.5), used by the augmentation tilt filter.
/// Accumulates in f64 (§0.8): x and h are packed as the real and imaginary parts of one input.
pub struct FftConvolver {
    planner: FftPlanner<f64>,
    buf: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
}

// SPEC: §1.5.5 writes `#[derive(Default)]`, but rustfft 6.4's FftPlanner has no Default impl, so
// the impl is written out. It still satisfies RUST-RT-21 (clippy::new_without_default).
impl Default for FftConvolver {
    fn default() -> Self {
        FftConvolver {
            planner: FftPlanner::new(),
            buf: Vec::new(),
            scratch: Vec::new(),
        }
    }
}

impl FftConvolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Linear convolution via one complex FFT of size next_pow2(x.len()+h.len()−1), cropped to the
    /// "same" alignment: output[n] = Σ_k h[k]·x[n + (h.len()−1)/2 − k]. h.len() must be odd.
    ///
    /// # Panics
    /// If `h.len()` is even (including 0): "same" alignment needs a centre tap.
    pub fn convolve_same(&mut self, x: &[f32], h: &[f32], out: &mut Vec<f32>) {
        assert!(h.len() % 2 == 1, "convolve_same needs an odd-length h");
        out.clear();
        if x.is_empty() {
            return;
        }
        let n = (x.len() + h.len() - 1).next_power_of_two();
        self.buf.clear();
        self.buf.resize(n, Complex::new(0.0, 0.0));
        for (b, &v) in self.buf.iter_mut().zip(x) {
            b.re = f64::from(v);
        }
        for (b, &v) in self.buf.iter_mut().zip(h) {
            b.im = f64::from(v);
        }
        let forward = self.planner.plan_fft_forward(n);
        let inverse = self.planner.plan_fft_inverse(n);
        let scratch_len = forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len());
        self.scratch.resize(scratch_len, Complex::new(0.0, 0.0));
        forward.process_with_scratch(&mut self.buf, &mut self.scratch[..scratch_len]);
        // Z = X + iH with X, H Hermitian, so X[k]·H[k] = (Z[k]² − conj(Z[−k])²)/(4i).
        let quarter_i = Complex::new(0.0, 4.0);
        for k in 0..=n / 2 {
            let j = (n - k) % n;
            let (a, b) = (self.buf[k], self.buf[j]);
            self.buf[k] = (a * a - b.conj() * b.conj()) / quarter_i;
            if j != k {
                self.buf[j] = (b * b - a.conj() * a.conj()) / quarter_i;
            }
        }
        inverse.process_with_scratch(&mut self.buf, &mut self.scratch[..scratch_len]);
        let offset = (h.len() - 1) / 2;
        let scale = 1.0 / n as f64;
        out.extend(
            self.buf[offset..offset + x.len()]
                .iter()
                .map(|c| (c.re * scale) as f32),
        );
    }
}

/// The ten rates of the §1.5.2 table.
#[cfg(test)]
pub(crate) const TABLE_RATES: [u32; 10] = [
    16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000, 176_400, 192_000,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng64;
    use proptest::prelude::*;

    fn rate(hz: u32) -> SampleRate {
        SampleRate::new(hz).expect("table rate")
    }

    /// |H(f)| in dB from an impulse response.
    fn gain_db(h: &[f64], fs: f64, f: f64) -> f64 {
        let w = std::f64::consts::TAU * f / fs;
        let (mut re, mut im) = (0.0, 0.0);
        for (n, &v) in h.iter().enumerate() {
            let (s, c) = (w * n as f64).sin_cos();
            re += v * c;
            im -= v * s;
        }
        10.0 * (re * re + im * im).log10()
    }

    #[test]
    fn d1_highpass_magnitude_and_decay() {
        for hz in TABLE_RATES {
            let fs = f64::from(hz);
            let mut hpf = Biquad::highpass(rate(hz), Hertz::new(100.0)).expect("hpf");
            let h: Vec<f64> = (0..2 * hz as usize)
                .map(|n| f64::from(hpf.process(if n == 0 { 1.0 } else { 0.0 })))
                .collect();
            let at_100 = gain_db(&h, fs, 100.0);
            let at_10 = gain_db(&h, fs, 10.0);
            let at_1k = gain_db(&h, fs, 1000.0);
            assert!((at_100 + 3.01).abs() <= 0.1, "{hz}: {at_100} dB at 100 Hz");
            assert!(at_10 <= -39.0, "{hz}: {at_10} dB at 10 Hz");
            assert!(at_1k.abs() <= 0.05, "{hz}: {at_1k} dB at 1 kHz");
            let total: f64 = h.iter().map(|v| v * v).sum();
            let tail: f64 = h[hz as usize..].iter().map(|v| v * v).sum();
            assert!(tail < 1e-9 * total, "{hz}: tail energy {tail} of {total}");
        }
        assert!(matches!(
            Biquad::highpass(rate(16_000), Hertz::new(8_000.0)),
            Err(DspError::CutoffOutOfRange { .. })
        ));
        assert!(matches!(
            Biquad::bandpass(rate(16_000), Hertz::new(1_000.0), 0.0),
            Err(DspError::BadQ { .. })
        ));
    }

    #[test]
    fn d2_frame_geometry_table() {
        // fs, L, H, N, L ms, H ms, Δf Hz (§1.5.2)
        let table: [(u32, usize, usize, usize, f64, f64, f64); 10] = [
            (16000, 170, 85, 256, 10.625, 5.312, 62.50),
            (22050, 236, 118, 256, 10.703, 5.351, 86.13),
            (24000, 256, 128, 256, 10.667, 5.333, 93.75),
            (32000, 342, 171, 512, 10.688, 5.344, 62.50),
            (44100, 470, 235, 512, 10.658, 5.329, 86.13),
            (48000, 512, 256, 512, 10.667, 5.333, 93.75),
            (88200, 940, 470, 1024, 10.658, 5.329, 86.13),
            (96000, 1024, 512, 1024, 10.667, 5.333, 93.75),
            (176400, 1882, 941, 2048, 10.669, 5.334, 86.13),
            (192000, 2048, 1024, 2048, 10.667, 5.333, 93.75),
        ];
        for (hz, l, h, n, l_ms, h_ms, df) in table {
            let g = FrameGeometry::for_rate(rate(hz));
            assert_eq!((g.len, g.hop, g.fft_len), (l, h, n), "{hz}");
            let fs = f64::from(hz);
            // The table prints ms to 3 decimals (truncated or rounded) and Δf to 2.
            assert!((l as f64 * 1000.0 / fs - l_ms).abs() < 1e-3, "{hz} L ms");
            assert!(
                (g.hop_duration(rate(hz)).get() - h_ms).abs() < 1e-3,
                "{hz} H ms"
            );
            assert!((fs / n as f64 - df).abs() < 5e-3, "{hz} Δf");
            println!(
                "D2 fs={hz:>6} L={l:>4} H={h:>4} N={n:>4} L_ms={:.3} H_ms={:.3} df={:.2}",
                l as f64 * 1000.0 / fs,
                g.hop_duration(rate(hz)).get(),
                fs / n as f64
            );
        }
    }

    #[test]
    fn d3_white_noise_band_means() {
        for hz in [48_000, 16_000] {
            let r = rate(hz);
            let geom = FrameGeometry::for_rate(r);
            let mut spectrum = PowerSpectrum::new(geom);
            let bands = BandLayout::new(r, geom);
            let mut rng = Rng64::new(0xD3 ^ u64::from(hz));
            let sigma = 1e-4f64.sqrt();
            let mut frame = vec![0.0f32; geom.len];
            let mut power = vec![0.0f64; geom.fft_len / 2 + 1];
            let mut e = [0.0f64; BAND_COUNT];
            let mut sum = [0.0f64; BAND_COUNT];
            let frames = 20_000;
            for _ in 0..frames {
                for x in frame.iter_mut() {
                    *x = (sigma * rng.normal()) as f32;
                }
                spectrum.compute(&frame, &mut power);
                bands.apply(&power, &mut e);
                for (s, v) in sum.iter_mut().zip(&e) {
                    *s += v;
                }
            }
            for (b, s) in sum.iter().enumerate() {
                let mean = s / f64::from(frames);
                assert!(
                    (mean / 1e-4 - 1.0).abs() <= 0.05,
                    "{hz} band {b}: mean {mean}"
                );
            }
        }
    }

    #[test]
    fn d4_band_layout_at_every_rate() {
        for hz in TABLE_RATES {
            let r = rate(hz);
            let layout = BandLayout::new(r, FrameGeometry::for_rate(r));
            for (b, (_, w)) in layout.weights().iter().enumerate() {
                let nonzero = w.iter().filter(|&&v| v > 0.0).count();
                assert!(nonzero >= 2, "{hz} band {b}: {nonzero} bins");
                let total: f64 = w.iter().sum();
                assert!((total - 1.0).abs() <= 1e-9, "{hz} band {b}: sum {total}");
            }
            assert!(layout.centres.windows(2).all(|c| c[0] < c[1]), "{hz}");
            let expected = (0.45 * f64::from(hz)).min(16_000.0);
            assert_eq!(BandLayout::f_top(r).get(), expected, "{hz}");
        }
    }

    fn direct_same(x: &[f32], h: &[f32]) -> Vec<f64> {
        let half = (h.len() - 1) / 2;
        (0..x.len())
            .map(|n| {
                h.iter()
                    .enumerate()
                    .filter_map(|(k, &hk)| {
                        (n + half)
                            .checked_sub(k)
                            .and_then(|i| x.get(i))
                            .map(|&xi| f64::from(hk) * f64::from(xi))
                    })
                    .sum()
            })
            .collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]
        #[test]
        fn d5_convolve_same_matches_direct(
            x_len in 1usize..5000,
            half in 0usize..800,
            seed in any::<u64>(),
        ) {
            let mut rng = Rng64::new(seed);
            let x: Vec<f32> = (0..x_len).map(|_| rng.range(-1.0, 1.0) as f32).collect();
            let h: Vec<f32> = (0..2 * half + 1).map(|_| rng.range(-1.0, 1.0) as f32).collect();
            let mut out = Vec::new();
            FftConvolver::new().convolve_same(&x, &h, &mut out);
            let direct = direct_same(&x, &h);
            prop_assert_eq!(out.len(), x.len());
            let max = direct.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for (o, d) in out.iter().zip(&direct) {
                prop_assert!((f64::from(*o) - d).abs() <= 1e-5 * max, "{} vs {}", o, d);
            }
        }
    }
}
