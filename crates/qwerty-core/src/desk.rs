//! Synthetic desk fixtures for tests, benches and the soak (§1.19).
//!
//! For invariance tests, false-accept regression and stress only: never an accuracy gate, and
//! never tuned to chase one (ADR-006). Every parameter is frozen.

use crate::rng::Rng64;
use crate::units::{ChannelCount, Decibels, Hertz, Milliseconds, SampleRate, Seconds};
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, TAU};
use std::fmt;

// ---------------------------------------------------------------------------
// CleanSignatures: the frozen v1 fixture (§1.19.2), moved byte-for-byte from `soak.rs`.
// ---------------------------------------------------------------------------

/// The v1 fixture's sample rate, frozen locally (was `features::SAMPLE_RATE_HZ`).
const FIXTURE_RATE: u32 = 44_100;
/// The v1 fixture's clip length, frozen locally (was `features::FEATURE_WINDOW_SAMPLES`).
const FIXTURE_CLIP: usize = 4096;

// ---------------------------------------------------------------------------
// Deterministic PRNG (no dependency; keeps the soak reproducible)
// ---------------------------------------------------------------------------

pub struct Lcg(u64);
impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    /// Uniform in [0, 1).
    pub fn next_unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// Uniform in [-1, 1).
    pub fn next_bipolar(&mut self) -> f32 {
        self.next_unit() * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Synthetic desk: distinct acoustic signature per zone
// ---------------------------------------------------------------------------

const LEAD_SILENCE: usize = 256 * 8;

pub struct CleanSignatures {
    /// Resonant partials (Hz) for each zone — distinct sets make zones
    /// spectrally separable, mirroring how desk positions resonate differently.
    zone_partials: Vec<Vec<f32>>,
}

impl CleanSignatures {
    pub fn new(zone_count: usize) -> Self {
        // Base frequency per zone spread across the audible band, each with a
        // small harmonic stack. NOTE: this fixture gives every zone the *same*
        // harmonic-ratio template scaled by a per-zone fundamental, so it is a
        // faithful sanity check only at the default zone count it was validated
        // against (≈99% at 4 zones). It is deliberately NOT used to assert a
        // multi-zone *accuracy* floor: at higher zone counts the scaled-template
        // signatures crowd together in a way that reflects the fixture's own
        // degeneracy, not the classifier — real multi-zone accuracy is gated by
        // the hardware-based ≥92% row in ACCEPTANCE.md, not by this generator.
        // (The `--zones` flag on the soak example exists to *explore* that
        // synthetic separability, not to gate on it.)
        let zone_partials = (0..zone_count)
            .map(|z| {
                let base = 700.0 + z as f32 * 430.0;
                vec![base, base * 1.9, base * 2.7]
            })
            .collect();
        Self { zone_partials }
    }

    /// A tap clip for `zone`: lead silence, then a fast-decaying multi-partial
    /// impulse, then trailing silence so a full window fits after the onset.
    pub fn tap(&self, zone: usize, rng: &mut Lcg, jitter: f32) -> Vec<f32> {
        let n = FIXTURE_CLIP;
        let mut clip = vec![0.0f32; LEAD_SILENCE];
        let partials = &self.zone_partials[zone];
        for i in 0..n {
            let t = i as f32 / FIXTURE_RATE as f32;
            let env = (-t * 130.0).exp();
            let mut s = 0.0f32;
            for (k, &f) in partials.iter().enumerate() {
                let fj = f * (1.0 + jitter * rng.next_bipolar() * 0.02);
                let amp = 1.0 / (k as f32 + 1.0);
                s += amp * (2.0 * std::f32::consts::PI * fj * t).sin();
            }
            clip.push(0.6 * env * s);
        }
        clip.extend(std::iter::repeat_n(0.0, n));
        clip
    }

    /// A sustained (non-tap) clip: a steady tone or a slowly amplitude-modulated
    /// broadband sound — energy that does not decay, so it must be rejected.
    pub fn sustained(&self, rng: &mut Lcg) -> Vec<f32> {
        let n = FIXTURE_CLIP * 2;
        let mut clip = vec![0.0f32; LEAD_SILENCE];
        let speechlike = rng.next_unit() > 0.5;
        let f = 300.0 + rng.next_unit() * 500.0;
        for i in 0..n {
            let t = i as f32 / FIXTURE_RATE as f32;
            let s = if speechlike {
                // Amplitude-modulated multi-tone, like voiced speech.
                let am = 0.6 + 0.4 * (2.0 * std::f32::consts::PI * 5.0 * t).sin();
                am * ((2.0 * std::f32::consts::PI * f * t).sin()
                    + 0.5 * (2.0 * std::f32::consts::PI * f * 2.5 * t).sin())
            } else {
                (2.0 * std::f32::consts::PI * f * t).sin()
            };
            clip.push(0.5 * s);
        }
        clip.extend(std::iter::repeat_n(0.0, FIXTURE_CLIP));
        clip
    }

    /// A *foreign* tap-like transient: a fast-decaying broadband noise burst.
    /// Crucially it has the SAME sharp attack + fast decay as a real tap, so it
    /// PASSES the onset detector's sustain gate and reaches the classifier —
    /// unlike [`sustained`], which onset rejects before the classifier sees it.
    /// But its spectrum is broadband noise, matching no calibrated zone's peaky
    /// harmonic signature, so the classifier's *novelty gate* must reject it.
    /// This is what makes the novelty gate actually testable end-to-end.
    pub fn foreign_tap(&self, rng: &mut Lcg) -> Vec<f32> {
        let n = FIXTURE_CLIP;
        let mut clip = vec![0.0f32; LEAD_SILENCE];
        for i in 0..n {
            let t = i as f32 / FIXTURE_RATE as f32;
            let env = (-t * 130.0).exp(); // same decay envelope as a real tap
            clip.push(0.6 * env * rng.next_bipolar());
        }
        clip.extend(std::iter::repeat_n(0.0, n));
        clip
    }
}

// ---------------------------------------------------------------------------
// ModalDesk: a rectangular plate rung by taps, plus airborne distractors (§1.19.1).
// ---------------------------------------------------------------------------
//
// Conventions shared by every output (none of them is given numerically by §1.19.1; each is a
// frozen choice of this fixture):
// - dBFS is re 1.0, for peaks and for RMS alike.
// - `tap` and `distractor` return the sound alone; the device floor comes only from `floor`, so a
//   caller streams `floor` and adds events on top (the engine's fake capture, p2 §2.15, does).
// - An airborne source's carrier is normalized to the RMS of a unit-peak sine, and 0 dB is the
//   level at which such a source at the plate centre peaks at the −6 dBFS reference at the
//   nearest mic, the same reference as a 0 dB centre tap. Spreading is 1/r with r the source-mic
//   distance.

/// Speed of sound in air, m/s.
const SPEED_OF_SOUND: f64 = 343.0;
/// Modes are kept below min(MODE_CEILING, MODE_CEILING_FRACTION·fs).
const MODE_CEILING: Hertz = Hertz::new(12_000.0);
const MODE_CEILING_FRACTION: f64 = 0.45;
/// ψ_{i,mn} ~ U(−PHASE_SPREAD, PHASE_SPREAD) rad.
const PHASE_SPREAD: f64 = 0.3;
/// The contact click's peak re the modal peak at the same mic.
// SPEC: "−10 dB" has no stated reference; it is read as re the tap's own modal peak at that mic.
const CLICK_LEVEL: Decibels = Decibels::new(-10.0);
/// The diffuse tail starts this long after the direct arrival.
const DIFFUSE_DELAY: Milliseconds = Milliseconds::new(3.0);
/// ln(1000): a 60 dB decay over RT60.
const RT60_DECAY: f64 = 6.91;
/// A tap's length after its onset, and the span every plate impact and its tail is rendered over.
const EVENT_SPAN: Milliseconds = Milliseconds::new(250.0);
/// Peak of a 0 dB tap at the plate centre.
const REFERENCE_PEAK: Decibels = Decibels::new(-6.0);
/// Contact time of the tap that sets the output normalization.
// SPEC: the normalizing tap's contact time is not given; 1 ms, inside TapSpec's 0.5-3 ms range.
const REFERENCE_CONTACT: Milliseconds = Milliseconds::new(1.0);
/// `TapSpec::contact` range, in ms.
const CONTACT_MIN_MS: f64 = 0.5;
const CONTACT_MAX_MS: f64 = 3.0;
/// Below this |1 − (2fT_c)²| the force spectrum takes its limit π/4.
const FORCE_SINGULARITY: f64 = 1e-6;
/// Source-mic distances are floored here, so 1/r stays finite for a mic resting on the desk.
const MIN_DISTANCE_M: f64 = 0.01;
/// A mode is rendered until its envelope falls to e^(−27.6) ≈ 1e-12 of its start, far below f32
/// resolution; this bounds the work of fast-decaying modes.
const MODE_CUTOFF: f64 = 27.6;
/// Floor of the normalizing peak: a desk without modes is silent, never NaN.
const PEAK_FLOOR: f64 = 1e-12;
/// `DeskConfig::light`'s loss factor (DSP-ML-13).
const LIGHT_LOSS_FACTOR: f64 = 0.015;

/// Rng64 streams derived from the desk seed, so the floor never depends on how many taps ran.
const STREAM_PHASES: u64 = 1;
const STREAM_EVENTS: u64 = 2;
const STREAM_FLOOR: u64 = 3;
const STREAM_DIFFUSE: u64 = 4;

// Distractor sources. SPEC: §1.19.1 fixes the keyboard and mouse positions only; the talker sits
// 0.3 m beyond the user's (front, y = D) edge at mouth height, and music plays from the laptop's
// speakers just in front of the hinge mics.
const KEYBOARD: [f64; 3] = [0.6, 0.15, 0.0];
const MOUSE: [f64; 3] = [0.95, 0.30, 0.0];
const TALKER: [f64; 3] = [0.6, 0.90, 0.35];
const LAPTOP_SPEAKER: [f64; 3] = [0.6, 0.10, 0.02];
/// Random partials summed into band-limited noise.
const NOISE_PARTIALS: usize = 48;
const BROADBAND_PARTIALS: usize = 96;
const LOW_BAND_PARTIALS: usize = 32;
/// Keystroke: press at 0 dB, release at −6 dB; each a 2-5 kHz burst plus a key click.
const KEY_BAND: (f64, f64) = (2000.0, 5000.0);
const KEY_DECAY: f64 = 900.0;
const KEY_BURST: Milliseconds = Milliseconds::new(15.0);
const KEY_RELEASE_LEVEL: Decibels = Decibels::new(-6.0);
// SPEC: the key click's shape and level are not given: a 0.25 ms raised-cosine pulse at −6 dB re
// its keystroke.
const KEY_CLICK_WIDTH: Milliseconds = Milliseconds::new(0.25);
const KEY_CLICK_LEVEL: Decibels = Decibels::new(-6.0);
/// Mouse click: a 4-8 kHz burst decaying at 3000/s, −12 dB.
const MOUSE_BAND: (f64, f64) = (4000.0, 8000.0);
const MOUSE_DECAY: f64 = 3000.0;
const MOUSE_BURST: Milliseconds = Milliseconds::new(5.0);
const MOUSE_LEVEL: Decibels = Decibels::new(-12.0);
/// Plosive: 50-300 Hz noise, 5 ms rise, decay 60/s, 40 ms long, plus breath noise.
// SPEC: a hard stop at 40 ms would add a click no plosive has; the last 5 ms are a raised-cosine
// taper. Breath noise is 0.5-8 kHz at −20 dB re the plosive.
const PLOSIVE_BAND: (f64, f64) = (50.0, 300.0);
const PLOSIVE_RISE: Milliseconds = Milliseconds::new(5.0);
const PLOSIVE_DECAY: f64 = 60.0;
const PLOSIVE_LENGTH: Milliseconds = Milliseconds::new(40.0);
const PLOSIVE_TAPER: Milliseconds = Milliseconds::new(5.0);
const BREATH_BAND: (f64, f64) = (500.0, 8000.0);
const BREATH_LEVEL: Decibels = Decibels::new(-20.0);
/// Knock: contact 5-8 ms, knocks 150-250 ms apart.
const KNOCK_CONTACT_MS: (f64, f64) = (5.0, 8.0);
const KNOCK_SPACING_MS: (f64, f64) = (150.0, 250.0);
/// Object set down: a 10 ms thud and 3-5 small clicks within 80 ms.
// SPEC: a small click is a 0.5 ms plate impact at −20 dB within 3 cm of the thud.
const THUD_CONTACT: Milliseconds = Milliseconds::new(10.0);
const SETTLE_WINDOW: Milliseconds = Milliseconds::new(80.0);
const SETTLE_CLICK_CONTACT: Milliseconds = Milliseconds::new(0.5);
const SETTLE_CLICK_FORCE: Decibels = Decibels::new(-20.0);
const SETTLE_CLICK_SCATTER_M: f64 = 0.03;
/// Drums: 120 bpm in 4/4, kick on beats 1 and 3, snare on 2 and 4, over a sustained chord.
// SPEC: levels, decays and lengths not in §1.19.1 are: kick −10 dB over 400 ms with a 40 ms sweep
// time constant; snare −13 dB, 1-8 kHz noise decaying 20/s plus the 200 Hz tone at −6 dB re it
// decaying 30/s, over 300 ms; chord −22 dB with 20 ms fades (the kit peaks near −7 dBFS).
const BEAT: Seconds = Seconds::new(0.5);
const BEATS_PER_BAR: u32 = 4;
const KICK_SWEEP_HZ: (f64, f64) = (120.0, 50.0);
const KICK_SWEEP_TIME: Seconds = Seconds::new(0.04);
const KICK_DECAY: f64 = 15.0;
const KICK_LENGTH: Seconds = Seconds::new(0.4);
const KICK_LEVEL: Decibels = Decibels::new(-10.0);
const SNARE_LEVEL: Decibels = Decibels::new(-13.0);
const SNARE_BAND: (f64, f64) = (1000.0, 8000.0);
const SNARE_NOISE_DECAY: f64 = 20.0;
const SNARE_TONE: Hertz = Hertz::new(200.0);
const SNARE_TONE_LEVEL: Decibels = Decibels::new(-6.0);
const SNARE_TONE_DECAY: f64 = 30.0;
const SNARE_LENGTH: Seconds = Seconds::new(0.3);
const CHORD: [f64; 3] = [220.0, 277.0, 330.0];
const CHORD_LEVEL: Decibels = Decibels::new(-22.0);
const CHORD_FADE: Milliseconds = Milliseconds::new(20.0);
/// Speech-like: syllables at 4 Hz, each a harmonic series on its own f0 in 120-220 Hz, shaped by
/// three formants.
// SPEC: each 250 ms syllable draws its own f0 and formants (F1 300-800, F2 900-2300, F3
// 2400-3200 Hz; bandwidths 80/120/160 Hz) under a sin² envelope; harmonics fall at 1/h below
// 4 kHz. The utterance is rounded up to whole syllables.
const SYLLABLE: Seconds = Seconds::new(0.25);
const F0_RANGE: (f64, f64) = (120.0, 220.0);
const FORMANT_RANGES: [(f64, f64); 3] = [(300.0, 800.0), (900.0, 2300.0), (2400.0, 3200.0)];
const FORMANT_BANDWIDTHS: [f64; 3] = [80.0, 120.0, 160.0];
const SPEECH_CEILING: Hertz = Hertz::new(4000.0);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeskConfig {
    pub rate: SampleRate,
    /// 1.20
    pub width_m: f64,
    /// 0.60
    pub depth_m: f64,
    /// 45 Hz
    pub f11: Hertz,
    /// 0.06 (mode decay σ = π·f·η). No reason was recorded when it was frozen; it is 2-6× the
    /// research range η ∈ [0.01, 0.03] and makes the decay gate easy. It stays frozen (changing
    /// it re-baselines every suite); `DeskConfig::light` (η = 0.015) covers the research range
    /// (DSP-ML-13).
    pub loss_factor: f64,
    /// 160 lowest modes with f < min(12 kHz, 0.45·fs)
    pub max_modes: usize,
    /// 800 Hz: R(f) = (f/fc)²/(1+(f/fc)²)
    pub radiation_corner: Hertz,
    /// default: [(0.54, 0.05, 0.25), (0.66, 0.05, 0.25)] m (two mics 12 cm apart at the laptop
    /// hinge)
    pub mics: Vec<MicPos>,
    /// 0.35 s
    pub rt60: Seconds,
    /// −18 dB re direct RMS
    pub diffuse_level: Decibels,
    /// Pink at −80 dBFS RMS, independent per mic
    pub floor: NoiseFloor,
    pub seed: u64,
}

impl DeskConfig {
    /// The values above: the one source of the defaults.
    pub fn default_for(rate: SampleRate, seed: u64) -> DeskConfig {
        DeskConfig {
            rate,
            width_m: 1.20,
            depth_m: 0.60,
            f11: Hertz::new(45.0),
            loss_factor: 0.06,
            max_modes: 160,
            radiation_corner: Hertz::new(800.0),
            mics: vec![
                MicPos {
                    x: 0.54,
                    y: 0.05,
                    z: 0.25,
                },
                MicPos {
                    x: 0.66,
                    y: 0.05,
                    z: 0.25,
                },
            ],
            rt60: Seconds::new(0.35),
            diffuse_level: Decibels::new(-18.0),
            floor: NoiseFloor::Pink {
                rms: Decibels::new(-80.0),
            },
            seed,
        }
    }

    /// The same desk with loss_factor 0.015 (frozen).
    pub fn light(rate: SampleRate, seed: u64) -> DeskConfig {
        DeskConfig {
            loss_factor: LIGHT_LOSS_FACTOR,
            ..DeskConfig::default_for(rate, seed)
        }
    }
}

/// A microphone position in metres: x, y on the plate, z above it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MicPos {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum NoiseFloor {
    Silent,
    Pink { rms: Decibels },
}

/// A tap at (x, y) m. `position_jitter_m` moves it uniformly within a disc of that radius.
///
/// Levels: the plate centre is a node of every mode with an even m or n, so a centre tap is the
/// quiet case the normalization is pinned to. A 0 dB tap at a zone centre peaks above full scale
/// (about +1 dBFS at 1 ms contact, +8 dBFS at 0.5 ms); a 3 ms contact is about 30 dB quieter than
/// 0.5 ms. Callers choose `force` for the peak or SNR they need.
// SPEC: the jitter's distribution is not given; uniform over the disc, clamped to the plate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TapSpec {
    pub x: f64,
    pub y: f64,
    /// gain re 0 dB
    pub force: Decibels,
    /// 0.5..=3
    pub contact: Milliseconds,
    pub position_jitter_m: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Distractor {
    /// press + release (−6 dB), each a 2-5 kHz keycap burst σ=900/s + click; at the keyboard
    /// (0.6, 0.15)
    KeystrokePair { gap: Milliseconds },
    /// 4-8 kHz burst σ=3000/s, −12 dB, at (0.95, 0.30)
    MouseClick,
    /// 50-300 Hz band-limited noise, 5 ms rise, decay 60/s, 40 ms, + breath noise
    Plosive,
    /// contact 5-8 ms, 150-250 ms apart
    Knock { x: f64, y: f64, count: u8 },
    /// contact 10 ms thud + 3-5 small clicks within 80 ms
    ObjectSetDown { x: f64, y: f64 },
    /// kick (120→50 Hz sweep, decay 15/s) + snare (noise + 200 Hz) at 120 bpm over a
    /// 220/277/330 Hz chord
    DrumsOverTones { bars: u8 },
    /// f0 120-220 Hz harmonics, 3 formants, 4 Hz syllable envelope (sustained)
    SpeechLike { seconds: f64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DeskError {
    NoMics,
    OutOfPlate { x: f64, y: f64 },
    // SPEC: §1.19.1 has no error for more mics than a `ChannelCount` holds (32), which
    // `channels()` cannot represent; it is refused here instead of truncated.
    TooManyMics { count: u64 },
}

impl fmt::Display for DeskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeskError::NoMics => write!(f, "the desk has no microphones"),
            DeskError::OutOfPlate { x, y } => {
                write!(f, "a microphone at ({x}, {y}) m is not over the plate")
            }
            DeskError::TooManyMics { count } => write!(
                f,
                "{count} microphones exceed the {} channels a stream can carry",
                ChannelCount::MAX
            ),
        }
    }
}

impl std::error::Error for DeskError {}

/// One plate mode (m, n).
#[derive(Debug, Clone, Copy)]
struct Mode {
    m: u32,
    n: u32,
    freq: f64,
    /// Decay rate σ = π·f·η, 1/s.
    sigma: f64,
    /// R(f).
    radiation: f64,
}

/// Unit-RMS pink noise: Paul Kellet's refined pinking filter on unit Gaussian white noise. Its
/// 1/f slope holds to within 0.05 dB from fs·2·10⁻⁴ up, at every table rate.
#[derive(Debug, Clone)]
struct Pink {
    rng: Rng64,
    state: [f64; 7],
    /// 1/RMS of the filter's output for unit-variance input (closed form, `pink_rms`).
    scale: f64,
}

/// Kellet's one-pole branches (pole, gain), then the direct and one-sample-delayed gains.
const PINK_POLES: [(f64, f64); 6] = [
    (0.99886, 0.055_517_9),
    (0.99332, 0.075_075_9),
    (0.96900, 0.153_852_0),
    (0.86650, 0.310_485_6),
    (0.55000, 0.532_952_2),
    (-0.7616, -0.016_898_0),
];
const PINK_DIRECT: f64 = 0.5362;
const PINK_DELAYED: f64 = 0.115_926;

impl Pink {
    fn new(rng: Rng64) -> Self {
        Pink {
            rng,
            state: [0.0; 7],
            scale: 1.0 / pink_rms(),
        }
    }

    fn next(&mut self) -> f64 {
        let white = self.rng.normal();
        let mut sum = 0.0;
        for (b, &(pole, gain)) in self.state.iter_mut().zip(PINK_POLES.iter()) {
            *b = pole * *b + gain * white;
            sum += *b;
        }
        let pink = sum + self.state[6] + PINK_DIRECT * white;
        self.state[6] = PINK_DELAYED * white;
        pink * self.scale
    }
}

/// The pinking filter's output RMS for unit-variance white input: the root of Σ h_i² over its
/// impulse response h_0 = Σg + c0, h_1 = Σg·a + c1, h_i = Σ g·a^i.
fn pink_rms() -> f64 {
    let mut energy = PINK_DIRECT * PINK_DIRECT + PINK_DELAYED * PINK_DELAYED;
    for &(a, g) in &PINK_POLES {
        energy += 2.0 * PINK_DIRECT * g + 2.0 * PINK_DELAYED * g * a;
        for &(b, h) in &PINK_POLES {
            energy += g * h / (1.0 - a * b);
        }
    }
    energy.sqrt()
}

/// What an airborne sound oscillates with, before its envelope.
#[derive(Debug, Clone)]
enum Carrier {
    /// Σ amp·sin(2π·freq·t + phase), with Σ amp² = 1.
    Partials(Vec<Partial>),
    /// A unit sine whose frequency glides from `from` to `to` with time constant `time`.
    Sweep { from: f64, to: f64, time: f64 },
    /// A unit raised-cosine pulse lasting the sound's duration.
    Pulse,
}

#[derive(Debug, Clone, Copy)]
struct Partial {
    freq: f64,
    amp: f64,
    phase: f64,
}

#[derive(Debug, Clone, Copy)]
enum Envelope {
    /// e^(−rate·t)
    Decay { rate: f64 },
    /// Linear rise, then e^(−rate·(t − rise)), with a raised-cosine taper over the last `taper` s.
    RiseDecay { rise: f64, rate: f64, taper: f64 },
    /// sin²(π·t/duration)
    Syllable,
    /// 1, with raised-cosine fades of `fade` s at both ends.
    Flat { fade: f64 },
}

/// One airborne sound, `start` s after the distractor's onset.
#[derive(Debug, Clone)]
struct Sound {
    pos: [f64; 3],
    start: f64,
    duration: f64,
    gain: f64,
    carrier: Carrier,
    envelope: Envelope,
}

/// One plate impact, `start` s after the distractor's onset.
#[derive(Debug, Clone, Copy)]
struct Impact {
    x: f64,
    y: f64,
    start: f64,
    contact: f64,
    gain: f64,
}

pub struct ModalDesk {
    config: DeskConfig,
    fs: f64,
    channels: ChannelCount,
    modes: Vec<Mode>,
    /// ψ, mode-major: `phases[mode·mics + mic]`.
    phases: Vec<f64>,
    /// Output gain that puts a 0 dB centre tap's peak at REFERENCE_PEAK.
    norm: f64,
    /// Gain of a 0 dB airborne source at 1 m (see the conventions above).
    airborne: f64,
    /// Samples in EVENT_SPAN.
    span: usize,
    rng: Rng64,
    floor: Vec<Pink>,
    diffuse: Vec<Pink>,
}

impl ModalDesk {
    pub fn new(config: DeskConfig) -> Result<Self, DeskError> {
        if config.mics.is_empty() {
            return Err(DeskError::NoMics);
        }
        let count = config.mics.len();
        let channels = u16::try_from(count)
            .ok()
            .and_then(|n| ChannelCount::new(n).ok())
            .ok_or(DeskError::TooManyMics {
                count: count as u64,
            })?;
        for mic in &config.mics {
            let on_plate =
                (0.0..=config.width_m).contains(&mic.x) && (0.0..=config.depth_m).contains(&mic.y);
            if !on_plate {
                return Err(DeskError::OutOfPlate { x: mic.x, y: mic.y });
            }
        }
        let fs = f64::from(config.rate.hz());
        let modes = plate_modes(&config, MODE_CEILING.get().min(MODE_CEILING_FRACTION * fs));
        let mut phase_rng = Rng64::derive(config.seed, STREAM_PHASES, 0);
        let phases = (0..modes.len() * count)
            .map(|_| phase_rng.range(-PHASE_SPREAD, PHASE_SPREAD))
            .collect();
        let pinks = |stream: u64| -> Vec<Pink> {
            (0..count as u64)
                .map(|mic| Pink::new(Rng64::derive(config.seed, stream, mic)))
                .collect()
        };
        let centre = [config.width_m / 2.0, config.depth_m / 2.0, 0.0];
        let nearest = config
            .mics
            .iter()
            .map(|mic| distance(centre, mic))
            .fold(f64::INFINITY, f64::min);
        let mut desk = ModalDesk {
            fs,
            channels,
            modes,
            phases,
            norm: 1.0,
            airborne: amplitude(REFERENCE_PEAK) * nearest,
            span: config.rate.samples_in(EVENT_SPAN),
            rng: Rng64::derive(config.seed, STREAM_EVENTS, 0),
            floor: pinks(STREAM_FLOOR),
            diffuse: pinks(STREAM_DIFFUSE),
            config,
        };
        // The 0 dB reference tap at the plate centre, direct sound only (no randomness).
        let reference = Impact {
            x: centre[0],
            y: centre[1],
            start: 0.0,
            contact: REFERENCE_CONTACT.get() / 1000.0,
            gain: 1.0,
        };
        let mut peak: f64 = 0.0;
        for mic in 0..count {
            let r = distance(centre, &desk.config.mics[mic]);
            let (_, direct) =
                desk.direct_at_mic(mic, &reference, 1.0 / r, r / SPEED_OF_SOUND, usize::MAX);
            peak = direct.iter().fold(peak, |p, v| p.max(v.abs()));
        }
        desk.norm = amplitude(REFERENCE_PEAK) / peak.max(PEAK_FLOOR);
        Ok(desk)
    }

    pub fn channels(&self) -> ChannelCount {
        self.channels
    }

    /// Interleaved, onset at frame `lead`; length lead + 250 ms.
    ///
    /// # Panics
    /// If (x, y) is not on the plate, `contact` is outside 0.5..=3 ms, or the jitter is negative.
    pub fn tap(&mut self, spec: &TapSpec, lead: usize) -> Vec<f32> {
        self.assert_on_plate(spec.x, spec.y);
        let contact_ms = spec.contact.get();
        assert!(
            (CONTACT_MIN_MS..=CONTACT_MAX_MS).contains(&contact_ms),
            "tap contact {contact_ms} ms is outside 0.5..=3 ms"
        );
        assert!(
            spec.position_jitter_m >= 0.0,
            "negative position jitter {}",
            spec.position_jitter_m
        );
        let radius = spec.position_jitter_m * self.rng.uniform().sqrt();
        let angle = TAU * self.rng.uniform();
        let x = (spec.x + radius * angle.cos()).clamp(0.0, self.config.width_m);
        let y = (spec.y + radius * angle.sin()).clamp(0.0, self.config.depth_m);
        let len = lead + self.span;
        let mut out = vec![vec![0.0; len]; self.mic_count()];
        let impact = Impact {
            x,
            y,
            start: 0.0,
            contact: contact_ms / 1000.0,
            gain: amplitude(spec.force),
        };
        self.add_impact(&mut out, &impact, lead as f64 / self.fs);
        interleave(&out)
    }

    /// Interleaved, onset at frame `lead`; length lead + the distractor's own duration + 250 ms.
    ///
    /// # Panics
    /// If a knock or set-down position is not on the plate, a keystroke gap is negative, or the
    /// speech length is negative or not finite.
    pub fn distractor(&mut self, d: &Distractor, lead: usize) -> Vec<f32> {
        let (sounds, impacts) = self.distractor_parts(d);
        let active = sounds
            .iter()
            .map(|s| s.start + s.duration)
            .chain(impacts.iter().map(|i| i.start))
            .fold(0.0, f64::max);
        let len = lead + (active * self.fs).ceil() as usize + self.span;
        let mut out = vec![vec![0.0; len]; self.mic_count()];
        let onset = lead as f64 / self.fs;
        for impact in &impacts {
            self.add_impact(&mut out, impact, onset);
        }
        for sound in &sounds {
            self.add_sound(&mut out, sound, onset);
        }
        interleave(&out)
    }

    /// The device floor alone, continuous across calls (each mic's generator keeps its state).
    pub fn floor(&mut self, frames: usize) -> Vec<f32> {
        let channels = self.mic_count();
        match self.config.floor {
            NoiseFloor::Silent => vec![0.0; frames * channels],
            NoiseFloor::Pink { rms } => {
                let a = amplitude(rms);
                let mut out = Vec::with_capacity(frames * channels);
                for _ in 0..frames {
                    for pink in &mut self.floor {
                        out.push((a * pink.next()) as f32);
                    }
                }
                out
            }
        }
    }

    /// Zone centres for n ∈ {2,4,6,8}: a 2-column grid over x ∈ [0.25, 0.95], y ∈ [0.25, 0.55].
    ///
    /// # Panics
    /// If n is not 2, 4, 6 or 8.
    // SPEC: the grid's spacing is not given: the two columns sit at x = 0.25 and 0.95, the n/2
    // rows are evenly spaced over y ∈ [0.25, 0.55] including both ends (one row sits at 0.40),
    // and zones are numbered row by row, left column first.
    pub fn zone_centres(n: usize) -> Vec<(f64, f64)> {
        assert!(
            matches!(n, 2 | 4 | 6 | 8),
            "zone_centres takes 2, 4, 6 or 8 zones, not {n}"
        );
        const COLUMNS: [f64; 2] = [0.25, 0.95];
        const ROWS: (f64, f64) = (0.25, 0.55);
        let rows = n / 2;
        (0..rows)
            .flat_map(|r| {
                let y = if rows == 1 {
                    (ROWS.0 + ROWS.1) / 2.0
                } else {
                    ROWS.0 + (ROWS.1 - ROWS.0) * r as f64 / (rows - 1) as f64
                };
                COLUMNS.map(|x| (x, y))
            })
            .collect()
    }

    fn mic_count(&self) -> usize {
        self.config.mics.len()
    }

    fn assert_on_plate(&self, x: f64, y: f64) {
        assert!(
            (0.0..=self.config.width_m).contains(&x) && (0.0..=self.config.depth_m).contains(&y),
            "({x}, {y}) m is not on the {} × {} m plate",
            self.config.width_m,
            self.config.depth_m
        );
    }

    /// One mic's direct sound of an impact arriving at `arrival` s: the modes plus the contact
    /// click, over EVENT_SPAN and before frame `len`. Returns the first frame and the samples.
    fn direct_at_mic(
        &self,
        mic: usize,
        impact: &Impact,
        amp: f64,
        arrival: f64,
        len: usize,
    ) -> (usize, Vec<f64>) {
        let (x, y, contact) = (impact.x, impact.y, impact.contact);
        let first = (arrival * self.fs).ceil() as usize;
        let end = first.saturating_add(self.span).min(len);
        if first >= end {
            return (first, Vec::new());
        }
        let mut acc = vec![0.0; end - first];
        let dt0 = first as f64 / self.fs - arrival;
        let (w, d) = (self.config.width_m, self.config.depth_m);
        for (k, mode) in self.modes.iter().enumerate() {
            let shape =
                (f64::from(mode.m) * PI * x / w).sin() * (f64::from(mode.n) * PI * y / d).sin();
            let a = amp * force_spectrum(mode.freq, contact) * shape * mode.radiation;
            let psi = self.phases[k * self.mic_count() + mic];
            // A mode stops once its envelope is below e^(−MODE_CUTOFF) of its start.
            let ringing = (MODE_CUTOFF / mode.sigma * self.fs).ceil();
            let n = if ringing < acc.len() as f64 {
                ringing as usize
            } else {
                acc.len()
            };
            add_damped_sine(&mut acc[..n], a, mode.sigma, mode.freq, psi, dt0, self.fs);
        }
        // The contact click: the half-sine force's derivative, cos(πt/T_c) on [0, T_c), whose
        // differentiation is the high-pass.
        // SPEC: "force derivative, high-passed" is read as the derivative itself (a first-order
        // high-pass of the force), with no further filter.
        let modal_peak = acc.iter().fold(0.0f64, |p, v| p.max(v.abs()));
        let click = modal_peak * amplitude(CLICK_LEVEL);
        for (j, v) in acc.iter_mut().enumerate() {
            let t = dt0 + j as f64 / self.fs;
            if t >= contact {
                break;
            }
            *v += click * (PI * t / contact).cos();
        }
        (first, acc)
    }

    /// Adds an impact's direct sound and diffuse tail at every mic.
    fn add_impact(&mut self, out: &mut [Vec<f64>], impact: &Impact, onset: f64) {
        let pos = [impact.x, impact.y, 0.0];
        let rt60 = self.config.rt60.get();
        let diffuse_gain = amplitude(self.config.diffuse_level);
        for (mic, buf) in out.iter_mut().enumerate() {
            let r = distance(pos, &self.config.mics[mic]);
            let arrival = onset + impact.start + r / SPEED_OF_SOUND;
            let amp = self.norm * impact.gain / r;
            let (first, direct) = self.direct_at_mic(mic, impact, amp, arrival, buf.len());
            if direct.is_empty() {
                continue;
            }
            // SPEC: "−18 dB re direct RMS" is read as re the RMS of this mic's direct sound over
            // its 250 ms span, applied to unit-RMS pink noise before the decay envelope.
            let rms = (direct.iter().map(|v| v * v).sum::<f64>() / direct.len() as f64).sqrt();
            for (j, v) in direct.iter().enumerate() {
                buf[first + j] += v;
            }
            let start = arrival + DIFFUSE_DELAY.get() / 1000.0;
            let from = (start * self.fs).ceil() as usize;
            let to = from.saturating_add(self.span).min(buf.len());
            let level = rms * diffuse_gain;
            for (j, v) in buf.iter_mut().enumerate().take(to).skip(from) {
                let t = j as f64 / self.fs - start;
                *v += level * (-RT60_DECAY * t / rt60).exp() * self.diffuse[mic].next();
            }
        }
    }

    /// Adds an airborne sound at every mic, delayed by r/c and spread by 1/r.
    fn add_sound(&self, out: &mut [Vec<f64>], sound: &Sound, onset: f64) {
        for (mic, buf) in out.iter_mut().enumerate() {
            let r = distance(sound.pos, &self.config.mics[mic]);
            let arrival = onset + sound.start + r / SPEED_OF_SOUND;
            let first = (arrival * self.fs).ceil() as usize;
            let end = (((arrival + sound.duration) * self.fs).ceil() as usize).min(buf.len());
            if first >= end {
                continue;
            }
            let dt0 = first as f64 / self.fs - arrival;
            let mut carrier = vec![0.0; end - first];
            match &sound.carrier {
                Carrier::Partials(partials) => {
                    for p in partials {
                        add_damped_sine(&mut carrier, p.amp, 0.0, p.freq, p.phase, dt0, self.fs);
                    }
                }
                Carrier::Sweep { from, to, time } => {
                    for (j, v) in carrier.iter_mut().enumerate() {
                        let t = dt0 + j as f64 / self.fs;
                        let cycles = to * t + (from - to) * time * (1.0 - (-t / time).exp());
                        *v = (TAU * cycles).sin();
                    }
                }
                Carrier::Pulse => {
                    for (j, v) in carrier.iter_mut().enumerate() {
                        let t = dt0 + j as f64 / self.fs;
                        *v = 0.5 - 0.5 * (TAU * t / sound.duration).cos();
                    }
                }
            }
            let gain = sound.gain * self.airborne / r;
            for (j, c) in carrier.iter().enumerate() {
                let t = dt0 + j as f64 / self.fs;
                buf[first + j] += gain * envelope(sound.envelope, t, sound.duration) * c;
            }
        }
    }

    /// The airborne sounds and plate impacts that make up `d`, drawn from the event stream.
    fn distractor_parts(&mut self, d: &Distractor) -> (Vec<Sound>, Vec<Impact>) {
        let ceiling = MODE_CEILING_FRACTION * self.fs;
        let mut sounds = Vec::new();
        let mut impacts = Vec::new();
        match *d {
            Distractor::KeystrokePair { gap } => {
                assert!(gap.get() >= 0.0, "negative keystroke gap {} ms", gap.get());
                for (start, level) in [
                    (0.0, 1.0),
                    (gap.get() / 1000.0, amplitude(KEY_RELEASE_LEVEL)),
                ] {
                    sounds.push(Sound {
                        pos: KEYBOARD,
                        start,
                        duration: KEY_BURST.get() / 1000.0,
                        gain: level,
                        carrier: Carrier::Partials(band(
                            &mut self.rng,
                            NOISE_PARTIALS,
                            KEY_BAND,
                            ceiling,
                        )),
                        envelope: Envelope::Decay { rate: KEY_DECAY },
                    });
                    sounds.push(Sound {
                        pos: KEYBOARD,
                        start,
                        duration: KEY_CLICK_WIDTH.get() / 1000.0,
                        gain: level * amplitude(KEY_CLICK_LEVEL),
                        carrier: Carrier::Pulse,
                        envelope: Envelope::Flat { fade: 0.0 },
                    });
                }
            }
            Distractor::MouseClick => sounds.push(Sound {
                pos: MOUSE,
                start: 0.0,
                duration: MOUSE_BURST.get() / 1000.0,
                gain: amplitude(MOUSE_LEVEL),
                carrier: Carrier::Partials(band(
                    &mut self.rng,
                    NOISE_PARTIALS,
                    MOUSE_BAND,
                    ceiling,
                )),
                envelope: Envelope::Decay { rate: MOUSE_DECAY },
            }),
            Distractor::Plosive => {
                let envelope = Envelope::RiseDecay {
                    rise: PLOSIVE_RISE.get() / 1000.0,
                    rate: PLOSIVE_DECAY,
                    taper: PLOSIVE_TAPER.get() / 1000.0,
                };
                for (count, band_hz, gain) in [
                    (LOW_BAND_PARTIALS, PLOSIVE_BAND, 1.0),
                    (BROADBAND_PARTIALS, BREATH_BAND, amplitude(BREATH_LEVEL)),
                ] {
                    sounds.push(Sound {
                        pos: TALKER,
                        start: 0.0,
                        duration: PLOSIVE_LENGTH.get() / 1000.0,
                        gain,
                        carrier: Carrier::Partials(band(&mut self.rng, count, band_hz, ceiling)),
                        envelope,
                    });
                }
            }
            Distractor::Knock { x, y, count } => {
                self.assert_on_plate(x, y);
                let mut start = 0.0;
                for _ in 0..count {
                    let contact = self.rng.range(KNOCK_CONTACT_MS.0, KNOCK_CONTACT_MS.1) / 1000.0;
                    impacts.push(Impact {
                        x,
                        y,
                        start,
                        contact,
                        gain: 1.0,
                    });
                    start += self.rng.range(KNOCK_SPACING_MS.0, KNOCK_SPACING_MS.1) / 1000.0;
                }
            }
            Distractor::ObjectSetDown { x, y } => {
                self.assert_on_plate(x, y);
                impacts.push(Impact {
                    x,
                    y,
                    start: 0.0,
                    contact: THUD_CONTACT.get() / 1000.0,
                    gain: 1.0,
                });
                let clicks = 3 + self.rng.below(3);
                for _ in 0..clicks {
                    let start = self.rng.range(0.0, SETTLE_WINDOW.get() / 1000.0);
                    let radius = SETTLE_CLICK_SCATTER_M * self.rng.uniform().sqrt();
                    let angle = TAU * self.rng.uniform();
                    impacts.push(Impact {
                        x: (x + radius * angle.cos()).clamp(0.0, self.config.width_m),
                        y: (y + radius * angle.sin()).clamp(0.0, self.config.depth_m),
                        start,
                        contact: SETTLE_CLICK_CONTACT.get() / 1000.0,
                        gain: amplitude(SETTLE_CLICK_FORCE),
                    });
                }
            }
            Distractor::DrumsOverTones { bars } => {
                let beats = u32::from(bars) * BEATS_PER_BAR;
                let beat = BEAT.get();
                for b in 0..beats {
                    let start = f64::from(b) * beat;
                    if b % 2 == 0 {
                        sounds.push(Sound {
                            pos: LAPTOP_SPEAKER,
                            start,
                            duration: KICK_LENGTH.get(),
                            gain: amplitude(KICK_LEVEL),
                            carrier: Carrier::Sweep {
                                from: KICK_SWEEP_HZ.0,
                                to: KICK_SWEEP_HZ.1,
                                time: KICK_SWEEP_TIME.get(),
                            },
                            envelope: Envelope::Decay { rate: KICK_DECAY },
                        });
                    } else {
                        let snare = amplitude(SNARE_LEVEL);
                        sounds.push(Sound {
                            pos: LAPTOP_SPEAKER,
                            start,
                            duration: SNARE_LENGTH.get(),
                            gain: snare,
                            carrier: Carrier::Partials(band(
                                &mut self.rng,
                                BROADBAND_PARTIALS,
                                SNARE_BAND,
                                ceiling,
                            )),
                            envelope: Envelope::Decay {
                                rate: SNARE_NOISE_DECAY,
                            },
                        });
                        sounds.push(Sound {
                            pos: LAPTOP_SPEAKER,
                            start,
                            duration: SNARE_LENGTH.get(),
                            gain: snare * amplitude(SNARE_TONE_LEVEL),
                            carrier: Carrier::Partials(vec![Partial {
                                freq: SNARE_TONE.get(),
                                amp: 1.0,
                                phase: 0.0,
                            }]),
                            envelope: Envelope::Decay {
                                rate: SNARE_TONE_DECAY,
                            },
                        });
                    }
                }
                if beats > 0 {
                    let amp = 1.0 / (CHORD.len() as f64).sqrt();
                    sounds.push(Sound {
                        pos: LAPTOP_SPEAKER,
                        start: 0.0,
                        duration: f64::from(beats) * beat,
                        gain: amplitude(CHORD_LEVEL),
                        carrier: Carrier::Partials(
                            CHORD
                                .iter()
                                .map(|&freq| Partial {
                                    freq,
                                    amp,
                                    phase: 0.0,
                                })
                                .collect(),
                        ),
                        envelope: Envelope::Flat {
                            fade: CHORD_FADE.get() / 1000.0,
                        },
                    });
                }
            }
            Distractor::SpeechLike { seconds } => {
                assert!(
                    seconds.is_finite() && seconds >= 0.0,
                    "speech length {seconds} s is not a finite non-negative number"
                );
                let syllable = SYLLABLE.get();
                let syllables = (seconds / syllable).ceil() as u64;
                let top = SPEECH_CEILING.get().min(ceiling);
                for s in 0..syllables {
                    sounds.push(Sound {
                        pos: TALKER,
                        start: s as f64 * syllable,
                        duration: syllable,
                        gain: 1.0,
                        carrier: Carrier::Partials(voiced(&mut self.rng, top)),
                        envelope: Envelope::Syllable,
                    });
                }
            }
        }
        (sounds, impacts)
    }
}

/// The modes below `ceiling`, lowest first (ties by m, then n), at most `max_modes`.
fn plate_modes(c: &DeskConfig, ceiling: f64) -> Vec<Mode> {
    let (w, d) = (c.width_m, c.depth_m);
    let s = (1.0 / w).powi(2) + (1.0 / d).powi(2);
    let f11 = c.f11.get();
    let fc = c.radiation_corner.get();
    // f_mn ≥ f11·(m/W)²/S, so every mode below the ceiling has m ≤ W·√(ceiling·S/f11); likewise n.
    let bound = |side: f64| (side * (ceiling * s / f11).sqrt()).ceil() as u32 + 1;
    let mut modes = Vec::new();
    for m in 1..=bound(w) {
        for n in 1..=bound(d) {
            let freq = f11 * ((f64::from(m) / w).powi(2) + (f64::from(n) / d).powi(2)) / s;
            if freq < ceiling {
                let q = (freq / fc).powi(2);
                modes.push(Mode {
                    m,
                    n,
                    freq,
                    sigma: PI * freq * c.loss_factor,
                    radiation: q / (1.0 + q),
                });
            }
        }
    }
    modes.sort_by(|a, b| {
        a.freq
            .total_cmp(&b.freq)
            .then(a.m.cmp(&b.m))
            .then(a.n.cmp(&b.n))
    });
    modes.truncate(c.max_modes);
    modes
}

/// |F(f)| of a half-sine contact of duration T_c: |cos(πfT_c)/(1 − (2fT_c)²)|, π/4 at 2fT_c = 1.
fn force_spectrum(f: f64, contact: f64) -> f64 {
    let x = 2.0 * f * contact;
    let denom = 1.0 - x * x;
    if denom.abs() < FORCE_SINGULARITY {
        PI / 4.0
    } else {
        ((PI * f * contact).cos() / denom).abs()
    }
}

/// Adds amp·e^(−σt)·sin(2πft + phase) at t = dt0 + k/fs to `acc[k]`, by phasor recursion.
fn add_damped_sine(
    acc: &mut [f64],
    amp: f64,
    sigma: f64,
    freq: f64,
    phase: f64,
    dt0: f64,
    fs: f64,
) {
    let omega = TAU * freq;
    let start = amp * (-sigma * dt0).exp();
    let (mut zi, mut zr) = (omega * dt0 + phase).sin_cos();
    zr *= start;
    zi *= start;
    let decay = (-sigma / fs).exp();
    let (si, sr) = (omega / fs).sin_cos();
    let (sr, si) = (sr * decay, si * decay);
    for v in acc.iter_mut() {
        *v += zi;
        let r = zr * sr - zi * si;
        zi = zr * si + zi * sr;
        zr = r;
    }
}

fn envelope(e: Envelope, t: f64, duration: f64) -> f64 {
    match e {
        Envelope::Decay { rate } => (-rate * t).exp(),
        Envelope::RiseDecay { rise, rate, taper } => {
            let shape = if t < rise {
                t / rise
            } else {
                (-rate * (t - rise)).exp()
            };
            let into_taper = t - (duration - taper);
            if into_taper > 0.0 {
                shape * (0.5 + 0.5 * (PI * into_taper / taper).cos())
            } else {
                shape
            }
        }
        Envelope::Syllable => (PI * t / duration).sin().powi(2),
        Envelope::Flat { fade } => {
            let edge = t.min(duration - t);
            if edge < fade {
                0.5 - 0.5 * (PI * edge / fade).cos()
            } else {
                1.0
            }
        }
    }
}

/// `count` partials at uniform random frequencies in the band (its top capped at `ceiling`), with
/// random phases and equal amplitudes, Σ amp² = 1.
fn band(rng: &mut Rng64, count: usize, (lo, hi): (f64, f64), ceiling: f64) -> Vec<Partial> {
    let hi = hi.min(ceiling);
    let amp = 1.0 / (count as f64).sqrt();
    (0..count)
        .map(|_| Partial {
            freq: rng.range(lo, hi),
            amp,
            phase: rng.range(0.0, TAU),
        })
        .collect()
}

/// One voiced syllable: harmonics of a random f0 below `top`, at 1/h, shaped by three random
/// formants (Lorentzian resonances), Σ amp² = 1.
fn voiced(rng: &mut Rng64, top: f64) -> Vec<Partial> {
    let f0 = rng.range(F0_RANGE.0, F0_RANGE.1);
    let formants = FORMANT_RANGES.map(|(lo, hi)| rng.range(lo, hi));
    let mut partials: Vec<Partial> = (1..)
        .map(|h| f64::from(h) * f0)
        .take_while(|&f| f < top)
        .enumerate()
        .map(|(i, freq)| {
            let resonance: f64 = formants
                .iter()
                .zip(FORMANT_BANDWIDTHS)
                .map(|(&fm, bw)| 1.0 / (1.0 + ((freq - fm) / bw).powi(2)))
                .sum();
            Partial {
                freq,
                amp: resonance / (i + 1) as f64,
                phase: rng.range(0.0, TAU),
            }
        })
        .collect();
    let norm = partials.iter().map(|p| p.amp * p.amp).sum::<f64>().sqrt();
    for p in &mut partials {
        p.amp /= norm;
    }
    partials
}

fn distance(p: [f64; 3], mic: &MicPos) -> f64 {
    let d = ((p[0] - mic.x).powi(2) + (p[1] - mic.y).powi(2) + (p[2] - mic.z).powi(2)).sqrt();
    d.max(MIN_DISTANCE_M)
}

/// Linear amplitude of a dB gain.
fn amplitude(db: Decibels) -> f64 {
    10f64.powf(db.get() / 20.0)
}

/// Per-mic f64 buffers → interleaved f32 frames.
fn interleave(mics: &[Vec<f64>]) -> Vec<f32> {
    let frames = mics.first().map_or(0, Vec::len);
    let mut out = Vec::with_capacity(frames * mics.len());
    for j in 0..frames {
        for mic in mics {
            out.push(mic[j] as f32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FNV-1a 64 offset basis and prime: a hash that is stable across Rust versions, unlike
    /// `DefaultHasher`.
    const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
    /// K4 golden hash of 4 zones × 3 clips, recorded from the pre-move v1 generator.
    const K4_GOLDEN: u64 = 0x6DA1_1BAD_C8FF_61FB;

    /// Folds the clip length and every sample's little-endian bit pattern into `hash`.
    fn fnv1a_clip(mut hash: u64, clip: &[f32]) -> u64 {
        let len = clip.len() as u64;
        for byte in len
            .to_le_bytes()
            .into_iter()
            .chain(clip.iter().flat_map(|s| s.to_bits().to_le_bytes()))
        {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    #[test]
    fn k4_clean_signatures_match_v1_generator() {
        // SPEC: "4 zones × 3 clips" read as one clip from each of the three generators per
        // zone (tap, sustained, foreign tap), drawn from one Lcg stream seeded as the soak's.
        let desk = CleanSignatures::new(4);
        let mut rng = Lcg::new(1);
        let mut hash = FNV_OFFSET;
        for zone in 0..4 {
            hash = fnv1a_clip(hash, &desk.tap(zone, &mut rng, 1.0));
            hash = fnv1a_clip(hash, &desk.sustained(&mut rng));
            hash = fnv1a_clip(hash, &desk.foreign_tap(&mut rng));
        }
        assert_eq!(hash, K4_GOLDEN, "got {hash:#018X}");
    }

    /// The 10 rates of the §1.5.2 frame-geometry table.
    const TABLE_RATES: [u32; 10] = [
        16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000, 176_400, 192_000,
    ];

    fn tap_at(x: f64, y: f64, jitter: f64) -> TapSpec {
        TapSpec {
            x,
            y,
            force: Decibels::new(0.0),
            contact: Milliseconds::new(1.0),
            position_jitter_m: jitter,
        }
    }

    /// One pass over every output: floor, a tap per zone centre, every distractor, floor again.
    fn render(rate: SampleRate, seed: u64) -> Vec<Vec<f32>> {
        let mut desk = ModalDesk::new(DeskConfig::default_for(rate, seed)).expect("desk");
        let mut out = vec![desk.floor(rate.samples_in(Milliseconds::new(50.0)))];
        for (x, y) in ModalDesk::zone_centres(4) {
            out.push(desk.tap(&tap_at(x, y, 0.01), 64));
        }
        for d in [
            Distractor::KeystrokePair {
                gap: Milliseconds::new(80.0),
            },
            Distractor::MouseClick,
            Distractor::Plosive,
            Distractor::Knock {
                x: 0.3,
                y: 0.4,
                count: 2,
            },
            Distractor::ObjectSetDown { x: 0.8, y: 0.45 },
            Distractor::DrumsOverTones { bars: 1 },
            Distractor::SpeechLike { seconds: 0.5 },
        ] {
            out.push(desk.distractor(&d, 32));
        }
        out.push(desk.floor(rate.samples_in(Milliseconds::new(50.0))));
        out
    }

    fn bits(clips: &[Vec<f32>]) -> Vec<Vec<u32>> {
        clips
            .iter()
            .map(|c| c.iter().map(|s| s.to_bits()).collect())
            .collect()
    }

    fn peak_dbfs(clip: &[f32]) -> f64 {
        let peak = clip.iter().fold(0.0f64, |p, &s| p.max(f64::from(s).abs()));
        20.0 * peak.log10()
    }

    #[test]
    fn k1_same_seed_is_bit_identical_at_every_rate() {
        for hz in TABLE_RATES {
            let rate = SampleRate::new(hz).expect("rate");
            let a = render(rate, 7);
            let b = render(rate, 7);
            assert_eq!(bits(&a), bits(&b), "{hz} Hz: same seed, different output");
            assert!(
                a.iter().flatten().all(|s| s.is_finite()),
                "{hz} Hz: non-finite output"
            );
            let lead = 64 * 2;
            let tap_len = 2 * (64 + rate.samples_in(Milliseconds::new(250.0)));
            assert!(
                a[1..5].iter().all(|t| t.len() == tap_len),
                "{hz} Hz: tap length"
            );
            assert!(
                a[1..5].iter().all(|t| t[..lead].iter().all(|&s| s == 0.0)),
                "{hz} Hz: sound before the onset"
            );
        }

        // The seed matters, and the normalization holds: a 0 dB centre tap with the reference
        // contact peaks at −6 dBFS, and the floor's RMS is −80 dBFS.
        let rate = SampleRate::new(48_000).expect("rate");
        assert_ne!(bits(&render(rate, 7)), bits(&render(rate, 8)));
        let mut desk = ModalDesk::new(DeskConfig::default_for(rate, 7)).expect("desk");
        assert_eq!(desk.channels(), ChannelCount::new(2).expect("2"));
        let centre = desk.tap(&tap_at(0.6, 0.3, 0.0), 0);
        assert!(
            (peak_dbfs(&centre) + 6.0).abs() < 0.5,
            "centre tap peaks at {} dBFS",
            peak_dbfs(&centre)
        );
        let floor = desk.floor(48_000 * 4);
        let rms =
            (floor.iter().map(|&s| f64::from(s).powi(2)).sum::<f64>() / floor.len() as f64).sqrt();
        assert!(
            (20.0 * rms.log10() + 80.0).abs() < 0.5,
            "floor RMS {} dBFS",
            20.0 * rms.log10()
        );
        for n in [2, 4, 6, 8] {
            let centres = ModalDesk::zone_centres(n);
            assert_eq!(centres.len(), n);
            assert!(centres
                .iter()
                .all(|&(x, y)| (0.25..=0.95).contains(&x) && (0.25..=0.55).contains(&y)));
        }
        assert_eq!(
            ModalDesk::new(DeskConfig {
                mics: Vec::new(),
                ..DeskConfig::default_for(rate, 7)
            })
            .err(),
            Some(DeskError::NoMics)
        );
    }
}
