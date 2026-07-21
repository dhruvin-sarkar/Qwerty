//! Onset detection with a sustained-sound rejection gate.
//!
//! A *tap* is a short transient: sharp attack, fast decay. Sustained sounds
//! (speech, music, steady tones) keep their energy. The detector (a) finds a
//! sharp energy rise above an adaptive noise floor, then (b) rejects it as
//! non-tap if the captured window does not decay — the head-vs-tail energy
//! test. See `ARCHITECTURE.md` (onset.rs) and the "DSP core (synthetic)" rows
//! of `ACCEPTANCE.md` (zero false accepts on the sustained set).
//!
//! The detector is streaming-friendly: [`OnsetDetector::process`] may be
//! called repeatedly with successive sample chunks and carries its noise-floor
//! and refractory state across calls.

use crate::features::{FEATURE_WINDOW_SAMPLES, SAMPLE_RATE_HZ};

/// Analysis hop, in samples (~5.8 ms at 44.1 kHz). Energy is measured per hop.
const HOP: usize = 256;

/// A candidate onset must exceed the noise floor by at least this factor.
const ONSET_RATIO: f32 = 3.0;

/// …and rise at least this sharply versus the previous hop (fast attack).
const ATTACK_RATIO: f32 = 2.0;

/// Absolute RMS floor below which nothing triggers (keeps silence quiet).
const ABS_FLOOR: f32 = 1e-3;

/// Noise-floor EMA rate (only updated on quiet hops so taps don't raise it).
const NOISE_EMA_ALPHA: f32 = 0.05;

/// Sustained-sound gate: if the window's tail energy is at least this fraction
/// of its head energy, the sound did not decay → reject as non-tap.
const SUSTAIN_DECAY_MAX: f32 = 0.5;

/// Head / tail measurement spans (~15 ms each).
const EDGE_SAMPLES: usize = (SAMPLE_RATE_HZ as usize * 15) / 1000;

const EPS: f32 = 1e-12;

/// A detected onset and the fixed-length window captured from it.
#[derive(Debug, Clone)]
pub struct OnsetEvent {
    /// Index (within the buffer passed to `process`) where the window starts.
    pub start: usize,
    /// Exactly [`FEATURE_WINDOW_SAMPLES`] samples for feature extraction.
    pub window: Vec<f32>,
    /// `true` if it passed the sustained-sound gate (a real tap-like
    /// transient); `false` if rejected as sustained.
    pub is_transient: bool,
}

/// Streaming adaptive onset detector.
pub struct OnsetDetector {
    noise_floor: f32,
    prev_hop_rms: f32,
    /// Hops remaining before another onset may fire (refractory period).
    cooldown_hops: usize,
    initialized: bool,
}

impl Default for OnsetDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl OnsetDetector {
    pub fn new() -> Self {
        Self {
            noise_floor: ABS_FLOOR,
            prev_hop_rms: 0.0,
            cooldown_hops: 0,
            initialized: false,
        }
    }

    /// Process a chunk of samples, returning any onsets whose full
    /// [`FEATURE_WINDOW_SAMPLES`] window fits within this chunk. Rejected
    /// (sustained) onsets are still returned, flagged `is_transient = false`,
    /// so callers/soak can account for them.
    pub fn process(&mut self, samples: &[f32]) -> Vec<OnsetEvent> {
        let mut events = Vec::new();
        let hops = samples.len() / HOP;

        for h in 0..hops {
            let start = h * HOP;
            let rms = hop_rms(&samples[start..start + HOP]);

            let elevated = rms > (self.noise_floor * ONSET_RATIO).max(ABS_FLOOR);
            let sharp = rms > self.prev_hop_rms * ATTACK_RATIO;

            if self.cooldown_hops > 0 {
                self.cooldown_hops -= 1;
            } else if self.initialized && elevated && sharp {
                // Candidate onset. Capture the window if it fits.
                if start + FEATURE_WINDOW_SAMPLES <= samples.len() {
                    let window = samples[start..start + FEATURE_WINDOW_SAMPLES].to_vec();
                    let is_transient = passes_sustain_gate(&window);
                    events.push(OnsetEvent {
                        start,
                        window,
                        is_transient,
                    });
                    // Refractory: don't re-trigger within one window.
                    self.cooldown_hops = FEATURE_WINDOW_SAMPLES / HOP;
                }
            } else {
                // Quiet hop: let the noise floor track the background.
                self.noise_floor =
                    NOISE_EMA_ALPHA * rms + (1.0 - NOISE_EMA_ALPHA) * self.noise_floor;
            }

            self.prev_hop_rms = rms;
            self.initialized = true;
        }
        events
    }
}

fn hop_rms(hop: &[f32]) -> f32 {
    (hop.iter().map(|s| s * s).sum::<f32>() / hop.len() as f32).sqrt()
}

fn mean_sq(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32
}

/// True if the window decays like a tap (tail energy well below head energy).
fn passes_sustain_gate(window: &[f32]) -> bool {
    let n = window.len();
    let edge = EDGE_SAMPLES.min(n / 4);
    let head = mean_sq(&window[..edge]);
    let tail = mean_sq(&window[n - edge..]);
    let decay_ratio = tail / (head + EPS);
    decay_ratio < SUSTAIN_DECAY_MAX
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn silence(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    /// A tap-like impulse: instant attack, exponential decay.
    fn tap(n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE_HZ as f32;
                let env = (-t * 120.0).exp(); // fast decay
                let carrier = (2.0 * PI * 2500.0 * i as f32 / SAMPLE_RATE_HZ as f32).sin();
                amp * env * carrier
            })
            .collect()
    }

    /// A sustained tone: rises and stays energetic for the whole window.
    fn sustained_tone(n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (2.0 * PI * 440.0 * i as f32 / SAMPLE_RATE_HZ as f32).sin())
            .collect()
    }

    /// One signal surrounded by quiet, long enough to hold a full window.
    fn clip_with(signal: Vec<f32>) -> Vec<f32> {
        let mut v = silence(HOP * 8);
        v.extend(signal);
        v.extend(silence(FEATURE_WINDOW_SAMPLES));
        v
    }

    #[test]
    fn accepts_a_tap_as_transient() {
        let mut d = OnsetDetector::new();
        let events = d.process(&clip_with(tap(FEATURE_WINDOW_SAMPLES, 0.8)));
        assert_eq!(events.len(), 1, "expected exactly one onset for one tap");
        assert!(events[0].is_transient, "a tap must pass the sustained gate");
        assert_eq!(events[0].window.len(), FEATURE_WINDOW_SAMPLES);
    }

    #[test]
    fn rejects_a_sustained_tone_as_non_tap() {
        let mut d = OnsetDetector::new();
        let events = d.process(&clip_with(sustained_tone(FEATURE_WINDOW_SAMPLES * 2, 0.8)));
        // Either no onset, or an onset explicitly rejected — never an accepted
        // transient (zero false accepts on sustained sound).
        assert!(
            events.iter().all(|e| !e.is_transient),
            "a sustained tone must not be accepted as a tap",
        );
    }

    #[test]
    fn silence_produces_no_onsets() {
        let mut d = OnsetDetector::new();
        assert!(d.process(&silence(FEATURE_WINDOW_SAMPLES * 4)).is_empty());
    }

    #[test]
    fn amplitude_does_not_flip_the_tap_decision() {
        for amp in [0.2, 0.5, 1.0] {
            let mut d = OnsetDetector::new();
            let events = d.process(&clip_with(tap(FEATURE_WINDOW_SAMPLES, amp)));
            assert!(
                events.iter().any(|e| e.is_transient),
                "tap at amp {amp} should be accepted",
            );
        }
    }
}
