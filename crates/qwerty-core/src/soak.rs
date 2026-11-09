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
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::classifier::ClassifierParams;
use crate::desk::{CleanSignatures, Lcg};
use crate::features::{FeatureExtractor, FeatureVector};
use crate::onset::OnsetDetector;
use crate::profile::ZoneId;

// ---------------------------------------------------------------------------
// Live-heap accounting allocator (used by the soak example for the
// "no unbounded memory growth" evidence).
// ---------------------------------------------------------------------------

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Successful `alloc`/`realloc` calls made by the current thread. `const`-initialized and
    /// without a destructor, so reading or bumping it never allocates and never fails, even
    /// from inside the allocator or during thread teardown.
    static THREAD_ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

/// Counts one allocation on the current thread.
fn count_thread_allocation() {
    THREAD_ALLOCATIONS.with(|n| n.set(n.get() + 1));
}

/// A `GlobalAlloc` wrapper that tracks currently-live heap bytes. Install it in
/// a binary with `#[global_allocator]` to measure steady-state heap use across
/// a long run (see the `soak` example).
pub struct CountingAllocator;

// SAFETY: forwards every call to the system allocator unchanged, only adding
// atomic byte accounting and a per-thread call count around it.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
            count_thread_allocation();
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            // The live-byte total moves exactly as the default alloc + dealloc pair moved it.
            ALLOCATED.fetch_add(new_size, Ordering::Relaxed);
            ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
            count_thread_allocation();
        }
        new_ptr
    }
}

/// Currently-live heap bytes as tracked by [`CountingAllocator`].
pub fn allocated_bytes() -> usize {
    ALLOCATED.load(Ordering::Relaxed)
}

/// Successful `alloc` and `realloc` calls the current thread has made through
/// [`CountingAllocator`] (0 when it is not the global allocator). Tests take the
/// difference across a region to assert it allocates nothing.
pub fn thread_allocations() -> u64 {
    THREAD_ALLOCATIONS.with(Cell::get)
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
///
/// Invariant: `taps == correct + rejected_taps + misclassified`. Splitting
/// "clean miss" (`rejected_taps`) from "confidently wrong zone"
/// (`misclassified`) matters for root-causing per the detective method — a
/// pooled `taps - correct` would hide which failure mode a regression hit.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BatchTally {
    /// Tap clips presented (one per zone per batch).
    pub taps: u32,
    /// Taps accepted and assigned to the correct zone.
    pub correct: u32,
    /// Real taps not credited as correct because they were *rejected*: no onset
    /// detected, the onset rejected as non-transient by the sustain gate, or an
    /// accepted onset rejected by the classifier's novelty gate.
    pub rejected_taps: u32,
    /// Real taps that were accepted but assigned to the *wrong* zone. Distinct
    /// from `rejected_taps`: the gate let them through, the classifier misread
    /// the location. Also depresses `accuracy()` (in the denominator, not the
    /// numerator), but is tracked separately so a soak log is diagnosable.
    pub misclassified: u32,
    /// Sustained (non-tap) clips presented.
    pub sustained: u32,
    /// Sustained clips wrongly accepted as a tap — MUST stay zero. Exercises the
    /// onset detector's sustain gate (sustained sounds don't decay, so they are
    /// rejected before the classifier ever sees them).
    pub sustained_false_accepts: u32,
    /// Foreign *transients* that reached the classifier: tap-like impulses
    /// (sharp attack, fast decay — so they PASS the onset sustain gate) whose
    /// acoustic signature matches no calibrated zone. This is the population the
    /// classifier's novelty gate exists to reject; counting it proves the gate
    /// is actually exercised (unlike sustained sounds, which never get past
    /// onset). MUST be > 0 for the false-accept check below to be meaningful.
    pub foreign_transients: u32,
    /// Foreign transients wrongly accepted as a calibrated zone — the novelty
    /// gate failing. MUST stay zero. A regression that widened the novelty
    /// threshold to accept-everything shows up here (and nowhere else).
    pub foreign_false_accepts: u32,
}

impl BatchTally {
    pub fn add(&mut self, o: &BatchTally) {
        self.taps += o.taps;
        self.correct += o.correct;
        self.rejected_taps += o.rejected_taps;
        self.misclassified += o.misclassified;
        self.sustained += o.sustained;
        self.sustained_false_accepts += o.sustained_false_accepts;
        self.foreign_transients += o.foreign_transients;
        self.foreign_false_accepts += o.foreign_false_accepts;
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
    desk: CleanSignatures,
    zone_ids: Vec<ZoneId>,
    rng: Lcg,
    /// A *separate* generator for the foreign-transient population, seeded
    /// independently so drawing foreign clips cannot perturb the RNG stream that
    /// drives tap jitter — otherwise adding the novelty-gate check would shift
    /// the very accuracy it sits beside (the tap/sustained streams stay bit-for-
    /// bit what they were before foreign transients existed).
    foreign_rng: Lcg,
    config: SoakConfig,
    batch: u64,
}

impl Soak {
    /// Build the harness: synthesize calibration taps, extract their features,
    /// and train the classifier — so the pipeline is ready to evaluate.
    pub fn new(config: SoakConfig) -> Self {
        let desk = CleanSignatures::new(config.zone_count);
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
            foreign_rng: Lcg::new(0xF0),
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
                } else {
                    // Accepted, but the classifier named the wrong zone.
                    tally.misclassified += 1;
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

        // Foreign transients: tap-like impulses that ARE accepted by the onset
        // sustain gate (they decay) but match no calibrated zone — the exact
        // population the classifier's novelty gate must reject. Counting the
        // ones that reach `classify()` proves the gate is genuinely exercised.
        for _ in 0..self.config.sustained_per_batch {
            let clip = self.desk.foreign_tap(&mut self.foreign_rng);
            let mut detector = OnsetDetector::new();
            for ev in detector.process(&clip) {
                if ev.is_transient {
                    tally.foreign_transients += 1;
                    let fv = self.extractor.extract(&ev.window);
                    if self.classifier.classify(&fv).accepted {
                        tally.foreign_false_accepts += 1;
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
    fn novelty_gate_rejects_foreign_transients() {
        // Closes the blind spot the sustained-sound test leaves open: sustained
        // sounds never pass the onset gate, so they never reach the classifier.
        // Foreign transients DO reach it (tap-like decay, non-zone spectrum), so
        // this is the only end-to-end test of the classifier's novelty gate. A
        // regression that widened the threshold to accept-everything fails here.
        let mut soak = Soak::new(SoakConfig::default());
        let mut tally = BatchTally::default();
        for _ in 0..50 {
            tally.add(&soak.run_batch());
        }
        assert!(
            tally.foreign_transients >= 100,
            "novelty gate was not exercised: only {} foreign transients reached it",
            tally.foreign_transients
        );
        assert_eq!(
            tally.foreign_false_accepts, 0,
            "novelty gate accepted a foreign (non-zone) transient as a tap"
        );
    }

    #[test]
    fn novelty_gate_holds_across_all_supported_zone_counts() {
        // The "never fire on a non-tap" safety property must hold at every zone
        // count the product offers (2/4/6/8), unlike synthetic *accuracy*, which
        // is only a faithful sanity check at the default config. This assertion
        // is fixture-independent: foreign broadband-noise transients must be
        // novelty-rejected regardless of how many zones are calibrated.
        for zone_count in [2usize, 4, 6, 8] {
            let mut soak = Soak::new(SoakConfig {
                zone_count,
                ..SoakConfig::default()
            });
            let mut tally = BatchTally::default();
            for _ in 0..20 {
                tally.add(&soak.run_batch());
            }
            assert!(
                tally.foreign_transients > 0,
                "{zone_count} zones: novelty gate not exercised"
            );
            assert_eq!(
                tally.foreign_false_accepts, 0,
                "{zone_count} zones: a foreign transient was accepted as a tap"
            );
        }
    }

    #[test]
    fn tally_accounts_for_every_presented_tap() {
        // The BatchTally invariant: every presented tap is exactly one of
        // correct / rejected / misclassified — no outcome silently dropped.
        let mut soak = Soak::new(SoakConfig::default());
        let mut tally = BatchTally::default();
        for _ in 0..30 {
            tally.add(&soak.run_batch());
        }
        assert_eq!(
            tally.taps,
            tally.correct + tally.rejected_taps + tally.misclassified,
            "a presented tap fell through all three outcome buckets"
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
