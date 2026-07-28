//! Synthetic stress harness for the DSP core — no real audio hardware.
//!
//! It synthesizes distinct per-zone tap signatures and sustained (non-tap)
//! sounds, runs them through the full `onset → features → classifier`
//! pipeline, and tallies accept/reject/classify outcomes. This is the evidence
//! source for the "DSP core (synthetic)" rows of `ACCEPTANCE.md`
//! (`ROADMAP.md` Phase 1). The long-running memory/stability check lives in the
//! `soak` example, which drives [`Soak::run_batch`] in a loop under the
//! [`CountingAllocator`] defined here.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::classifier::ClassifierParams;
use crate::features::{FeatureExtractor, FeatureVector, FEATURE_WINDOW_SAMPLES, SAMPLE_RATE_HZ};
use crate::onset::OnsetDetector;
use crate::profile::ZoneId;

// ---------------------------------------------------------------------------
// Live-heap accounting allocator (used by the soak example for the
// "no unbounded memory growth" evidence).
// ---------------------------------------------------------------------------

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

/// A `GlobalAlloc` wrapper that tracks currently-live heap bytes. Install it in
/// a binary with `#[global_allocator]` to measure steady-state heap use across
/// a long run (see the `soak` example).
pub struct CountingAllocator;

// SAFETY: forwards every call to the system allocator unchanged, only adding
// atomic byte accounting around it.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

/// Currently-live heap bytes as tracked by [`CountingAllocator`].
pub fn allocated_bytes() -> usize {
    ALLOCATED.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (no dependency; keeps the soak reproducible)
// ---------------------------------------------------------------------------

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    /// Uniform in [0, 1).
    fn next_unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// Uniform in [-1, 1).
    fn next_bipolar(&mut self) -> f32 {
        self.next_unit() * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Synthetic desk: distinct acoustic signature per zone
// ---------------------------------------------------------------------------

const LEAD_SILENCE: usize = 256 * 8;

struct SyntheticDesk {
    /// Resonant partials (Hz) for each zone — distinct sets make zones
    /// spectrally separable, mirroring how desk positions resonate differently.
    zone_partials: Vec<Vec<f32>>,
}

impl SyntheticDesk {
    fn new(zone_count: usize) -> Self {
        // Base frequency per zone spread across the audible band, each with a
        // small harmonic stack.
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
    fn tap(&self, zone: usize, rng: &mut Lcg, jitter: f32) -> Vec<f32> {
        let n = FEATURE_WINDOW_SAMPLES;
        let mut clip = vec![0.0f32; LEAD_SILENCE];
        let partials = &self.zone_partials[zone];
        for i in 0..n {
            let t = i as f32 / SAMPLE_RATE_HZ as f32;
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
    fn sustained(&self, rng: &mut Lcg) -> Vec<f32> {
        let n = FEATURE_WINDOW_SAMPLES * 2;
        let mut clip = vec![0.0f32; LEAD_SILENCE];
        let speechlike = rng.next_unit() > 0.5;
        let f = 300.0 + rng.next_unit() * 500.0;
        for i in 0..n {
            let t = i as f32 / SAMPLE_RATE_HZ as f32;
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
        clip.extend(std::iter::repeat_n(0.0, FEATURE_WINDOW_SAMPLES));
        clip
    }
}

// ---------------------------------------------------------------------------
// Soak driver
// ---------------------------------------------------------------------------

/// Soak parameters.
#[derive(Debug, Clone)]
pub struct SoakConfig {
    pub zone_count: usize,
    pub calib_taps_per_zone: usize,
    pub sustained_per_batch: usize,
}

impl Default for SoakConfig {
    fn default() -> Self {
        Self {
            zone_count: 4,
            calib_taps_per_zone: 10,
            sustained_per_batch: 4,
        }
    }
}

/// Outcome counts from one or more batches.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BatchTally {
    /// Tap clips presented (one per zone per batch).
    pub taps: u32,
    /// Taps accepted and assigned to the correct zone.
    pub correct: u32,
    /// Real taps not credited as correct: no onset detected, the onset rejected
    /// as non-transient by the sustain gate, or an accepted onset rejected by
    /// the classifier's novelty gate.
    pub rejected_taps: u32,
    /// Sustained (non-tap) clips presented.
    pub sustained: u32,
    /// Sustained clips wrongly accepted as a tap — MUST stay zero.
    pub sustained_false_accepts: u32,
}

impl BatchTally {
    pub fn add(&mut self, o: &BatchTally) {
        self.taps += o.taps;
        self.correct += o.correct;
        self.rejected_taps += o.rejected_taps;
        self.sustained += o.sustained;
        self.sustained_false_accepts += o.sustained_false_accepts;
    }

    /// Fraction of presented taps assigned to the correct zone.
    pub fn accuracy(&self) -> f32 {
        if self.taps == 0 {
            0.0
        } else {
            self.correct as f32 / self.taps as f32
        }
    }
}

/// A fully-wired synthetic pipeline: trained classifier plus the generators
/// needed to keep producing fresh taps and sustained sounds.
pub struct Soak {
    classifier: ClassifierParams,
    extractor: FeatureExtractor,
    desk: SyntheticDesk,
    zone_ids: Vec<ZoneId>,
    rng: Lcg,
    config: SoakConfig,
    batch: u64,
}

impl Soak {
    /// Build the harness: synthesize calibration taps, extract their features,
    /// and train the classifier — so the pipeline is ready to evaluate.
    pub fn new(config: SoakConfig) -> Self {
        let desk = SyntheticDesk::new(config.zone_count);
        let zone_ids: Vec<ZoneId> = (0..config.zone_count)
            .map(|z| ZoneId::from_uuid(uuid::Uuid::from_u128(1000 + z as u128)))
            .collect();
        let mut extractor = FeatureExtractor::new();
        let mut rng = Lcg::new(1);

        let mut samples: Vec<(ZoneId, FeatureVector)> = Vec::new();
        for (z, &zid) in zone_ids.iter().enumerate() {
            for _ in 0..config.calib_taps_per_zone {
                let clip = desk.tap(z, &mut rng, 1.0);
                if let Some((fv, true)) = pipeline_first(&clip, &mut extractor) {
                    samples.push((zid, fv));
                }
            }
        }
        let classifier =
            ClassifierParams::train(&samples, &[]).expect("synthetic calibration produced samples");

        Self {
            classifier,
            extractor,
            desk,
            zone_ids,
            rng,
            config,
            batch: 0,
        }
    }

    /// Run one evaluation batch: one fresh tap per zone plus
    /// `sustained_per_batch` sustained sounds. Allocations here are transient
    /// (windows/feature buffers) and freed before returning, so live heap stays
    /// flat across many batches.
    pub fn run_batch(&mut self) -> BatchTally {
        self.batch += 1;
        let mut tally = BatchTally::default();

        for (z, &zid) in self.zone_ids.iter().enumerate() {
            let clip = self.desk.tap(z, &mut self.rng, 1.0);
            tally.taps += 1;
            if let Some((fv, true)) = pipeline_first(&clip, &mut self.extractor) {
                let c = self.classifier.classify(&fv);
                if !c.accepted {
                    tally.rejected_taps += 1;
                } else if c.zone_id == Some(zid) {
                    tally.correct += 1;
                }
            } else {
                tally.rejected_taps += 1;
            }
        }

        for _ in 0..self.config.sustained_per_batch {
            let clip = self.desk.sustained(&mut self.rng);
            tally.sustained += 1;
            let mut detector = OnsetDetector::new();
            for ev in detector.process(&clip) {
                if ev.is_transient {
                    let fv = self.extractor.extract(&ev.window);
                    if self.classifier.classify(&fv).accepted {
                        tally.sustained_false_accepts += 1;
                    }
                }
            }
        }

        tally
    }

    pub fn zone_count(&self) -> usize {
        self.config.zone_count
    }
}

/// Run one clip through onset → features, returning the first onset's feature
/// vector and whether it was a transient (tap-like) event.
fn pipeline_first(clip: &[f32], extractor: &mut FeatureExtractor) -> Option<(FeatureVector, bool)> {
    let mut detector = OnsetDetector::new();
    let events = detector.process(clip);
    events.into_iter().next().map(|ev| {
        let fv = extractor.extract(&ev.window);
        (fv, ev.is_transient)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_synthetic_accuracy_is_at_least_99_percent() {
        let mut soak = Soak::new(SoakConfig::default());
        let mut tally = BatchTally::default();
        for _ in 0..50 {
            let b = soak.run_batch();
            tally.add(&b);
        }
        assert!(tally.taps >= 200);
        assert!(
            tally.accuracy() >= 0.99,
            "clean synthetic accuracy {:.3} < 0.99",
            tally.accuracy()
        );
    }

    #[test]
    fn zero_false_accepts_on_sustained_sounds() {
        let mut soak = Soak::new(SoakConfig::default());
        let mut tally = BatchTally::default();
        for _ in 0..50 {
            tally.add(&soak.run_batch());
        }
        assert!(tally.sustained >= 100);
        assert_eq!(
            tally.sustained_false_accepts, 0,
            "sustained sounds produced false accepts"
        );
    }

    #[test]
    fn loop_runs_cleanly_many_times() {
        // Exercises the full pipeline repeatedly to shake out leaks/panics; the
        // real live-heap byte delta is asserted by the `soak` example.
        let mut soak = Soak::new(SoakConfig::default());
        for _ in 0..200 {
            let _ = soak.run_batch();
        }
    }
}
