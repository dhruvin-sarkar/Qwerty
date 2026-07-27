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

/// Absolute RMS floor below which nothing triggers (keeps *digital* silence
/// quiet). Detection is otherwise SNR-relative (`noise_floor * ONSET_RATIO`),
/// so this is only a guard against triggering on true silence — not a minimum
/// tap loudness. It was originally -60 dB (`1e-3`), which turned out to be far
/// hotter than real low-gain USB mics deliver: a captured desk tap on a Razer
/// USB headset peaked at ~-81 dB with a per-hop RMS near -90 dB (evidence from
/// the `--monitor` diagnostic), i.e. entirely below the old floor, so no onset
/// could ever fire despite a clean ~60 dB tap-to-silence SNR. Lowered to
/// ~-120 dB so the relative gate does the real work and quiet-but-clean mics
/// are heard; measured room silence sat near -147 dB, well below this.
const ABS_FLOOR: f32 = 1e-6;

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

/// Streaming adaptive onset detector. Feed it successive sample chunks with
/// [`OnsetDetector::process`]; it buffers internally so a tap's window may span
/// many chunks, and drops consumed samples so memory stays bounded.
pub struct OnsetDetector {
    /// Samples received but not yet fully scanned (front is compacted away as
    /// hops are consumed, so this never grows unbounded).
    buf: Vec<f32>,
    /// Absolute sample index of `buf[0]`, so emitted `start`s are stream-global.
    base_index: usize,
    /// Hops within `buf` already scanned.
    processed_hops: usize,
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
            buf: Vec::new(),
            base_index: 0,
            processed_hops: 0,
            noise_floor: ABS_FLOOR,
            prev_hop_rms: 0.0,
            cooldown_hops: 0,
            initialized: false,
        }
    }

    /// Append a chunk of samples and return any onsets whose full
    /// [`FEATURE_WINDOW_SAMPLES`] window has now arrived. Rejected (sustained)
    /// onsets are still returned, flagged `is_transient = false`, so
    /// callers/soak can account for them. Passing an entire clip in one call
    /// yields exactly the onsets contained in it (the Phase 1 behavior).
    pub fn process(&mut self, samples: &[f32]) -> Vec<OnsetEvent> {
        self.buf.extend_from_slice(samples);
        let mut events = Vec::new();

        // Scan every hop that now has a full window of trailing samples, so a
        // detected onset can be emitted immediately — and the boundary hop is
        // not skipped. The guard also makes the window slice below panic-safe.
        loop {
            let start = self.processed_hops * HOP;
            if start + FEATURE_WINDOW_SAMPLES > self.buf.len() {
                break;
            }
            let rms = hop_rms(&self.buf[start..start + HOP]);

            // A corrupt hop (a NaN/Inf sample from a device glitch/underrun)
            // must never reach the noise-floor EMA: `alpha*NaN + (1-alpha)*x`
            // stays NaN forever, which then collapses the onset threshold to
            // `ABS_FLOOR` (since `NaN.max(x) == x`) and silently floods false
            // triggers with no recovery short of a restart. Consume the hop and
            // move on, leaving `noise_floor`/`prev_hop_rms` at their last good
            // value (`CLAUDE.md`: fail fast, no silent non-recoverable drift).
            if !rms.is_finite() {
                self.processed_hops += 1;
                continue;
            }

            let elevated = rms > (self.noise_floor * ONSET_RATIO).max(ABS_FLOOR);
            let sharp = rms > self.prev_hop_rms * ATTACK_RATIO;

            if self.cooldown_hops > 0 {
                self.cooldown_hops -= 1;
            } else if self.initialized && elevated && sharp {
                let window = self.buf[start..start + FEATURE_WINDOW_SAMPLES].to_vec();
                let is_transient = passes_sustain_gate(&window);
                events.push(OnsetEvent {
                    start: self.base_index + start,
                    window,
                    is_transient,
                });
                // Refractory: don't re-trigger within one window.
                self.cooldown_hops = FEATURE_WINDOW_SAMPLES / HOP;
            } else {
                // Quiet hop: let the noise floor track the background.
                self.noise_floor =
                    NOISE_EMA_ALPHA * rms + (1.0 - NOISE_EMA_ALPHA) * self.noise_floor;
            }

            self.prev_hop_rms = rms;
            self.initialized = true;
            self.processed_hops += 1;
        }

        // Compact: drop fully-scanned samples from the front (already copied
        // into any emitted window) so the buffer stays bounded.
        let drop = self.processed_hops * HOP;
        if drop > 0 {
            self.buf.drain(0..drop);
            self.base_index += drop;
            self.processed_hops = 0;
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

    #[test]
    fn a_nan_sample_does_not_poison_the_noise_floor() {
        let mut d = OnsetDetector::new();
        // Warm the adaptive floor above ABS_FLOOR with steady background energy.
        d.process(&sustained_tone(FEATURE_WINDOW_SAMPLES * 2, 0.05));
        assert!(d.noise_floor.is_finite() && d.noise_floor > ABS_FLOOR);

        // A single corrupt sample must not turn the floor into NaN (which would
        // collapse the onset threshold to ABS_FLOOR and flood false triggers).
        d.process(&[f32::NAN]);
        d.process(&sustained_tone(FEATURE_WINDOW_SAMPLES, 0.05));
        assert!(
            d.noise_floor.is_finite(),
            "noise floor became non-finite after a NaN sample: {}",
            d.noise_floor
        );

        // …and the detector still works: a clean tap is still accepted.
        let events = d.process(&clip_with(tap(FEATURE_WINDOW_SAMPLES, 0.8)));
        assert!(
            events.iter().any(|e| e.is_transient),
            "detector must still detect a real tap after a NaN sample"
        );
    }

    #[test]
    fn streaming_in_small_chunks_matches_single_call() {
        // A tap fed in sub-hop chunks (as real cpal callbacks deliver it) must
        // be detected identically to passing the whole clip at once.
        let clip = clip_with(tap(FEATURE_WINDOW_SAMPLES, 0.8));

        let mut streamed = OnsetDetector::new();
        let mut events = Vec::new();
        for chunk in clip.chunks(100) {
            events.extend(streamed.process(chunk));
        }

        let mut single = OnsetDetector::new();
        let one_shot = single.process(&clip);

        assert!(
            events.iter().any(|e| e.is_transient),
            "streaming across chunk boundaries should still detect the tap",
        );
        assert_eq!(
            events.len(),
            one_shot.len(),
            "chunked and single-call must agree on onset count",
        );
    }

    use proptest::prelude::*;

    proptest! {
        /// Invariant (ACCEPTANCE.md): no steady tone, at any frequency or
        /// amplitude, is ever accepted as a tap — the zero-false-accepts claim
        /// generalized over the input space. A steady tone's energy does not
        /// decay, so the head-vs-tail gate must always reject it.
        #[test]
        fn steady_tones_are_never_accepted_as_taps(
            freq in 150.0f32..3000.0,
            amp in 0.05f32..1.0,
        ) {
            let mut d = OnsetDetector::new();
            let n = FEATURE_WINDOW_SAMPLES * 2;
            let signal: Vec<f32> = (0..n)
                .map(|i| amp * (2.0 * PI * freq * i as f32 / SAMPLE_RATE_HZ as f32).sin())
                .collect();
            let events = d.process(&clip_with(signal));
            prop_assert!(events.iter().all(|e| !e.is_transient));
        }

        /// Invariant: a sharp, fast-decaying impulse is accepted as a tap across
        /// the plausible range of decay rates, carrier frequencies, and levels.
        #[test]
        fn fast_decay_impulses_are_accepted(
            decay in 90.0f32..240.0,
            freq in 500.0f32..6000.0,
            amp in 0.2f32..1.0,
        ) {
            let mut d = OnsetDetector::new();
            let n = FEATURE_WINDOW_SAMPLES;
            let signal: Vec<f32> = (0..n)
                .map(|i| {
                    let t = i as f32 / SAMPLE_RATE_HZ as f32;
                    amp * (-t * decay).exp() * (2.0 * PI * freq * i as f32 / SAMPLE_RATE_HZ as f32).sin()
                })
                .collect();
            let events = d.process(&clip_with(signal));
            prop_assert!(events.iter().any(|e| e.is_transient));
        }
    }
}
