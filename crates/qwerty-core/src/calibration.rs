//! Guided calibration: the domain logic behind the wizard in `DESIGN.md`
//! (Calibration flow) and `ARCHITECTURE.md` (`calibration.rs`).
//!
//! This is pure, device-free logic — the analysis lives in free functions that
//! take samples/windows and return verdicts, so every rule is unit-testable
//! against synthetic signals (`qwerty-core` never touches real audio). A thin
//! [`CalibrationSession`] collects samples across the wizard's steps and calls
//! into those functions; the `qwerty-app` wizard drives it with real capture.
//!
//! The flow it supports (`DESIGN.md`): environment check → per-zone capture
//! with immediate per-tap feedback → leave-one-out consistency check →
//! optional negative examples → build a `Profile`.

use chrono::{DateTime, Utc};

use crate::classifier::{ClassifierError, ClassifierParams};
use crate::features::{FeatureExtractor, FeatureVector, FEATURE_WINDOW_SAMPLES, SAMPLE_RATE_HZ};
use crate::profile::{
    DeviceFingerprint, Profile, SensingMode, Zone, ZoneId, ZoneLayout, PROFILE_SCHEMA_VERSION,
};

/// Default accepted taps captured per zone (`DESIGN.md`: "ten accepted taps").
pub const DEFAULT_TAPS_PER_ZONE: usize = 10;

/// Minimum samples per zone required before a consistency check or build is
/// meaningful (leave-one-out needs at least one sample left per zone).
pub const MIN_SAMPLES_PER_ZONE: usize = 2;

/// A tap is clipped if its peak reaches this fraction of full scale.
const CLIP_PEAK: f32 = 0.98;
/// Onset-region RMS below which a window is treated as *digital silence*, not a
/// tap. This is only a silence guard, mirroring `onset.rs` ABS_FLOOR — the
/// meaningful "too quiet to use" decision is the SNR check below (onset energy
/// vs the room's noise floor). It was -40 dBFS (`0.01`), ~200-300x hotter than
/// a real low-gain USB mic delivers: the reference Razer headset's accepted
/// taps have an onset-region RMS near -87 dBFS (~4.5e-5), so every tap the
/// onset detector accepts (after the ABS_FLOOR fix) was still labelled TooWeak
/// here and calibration was impossible. Lowered to ~-120 dB so it clears only
/// true silence; NOTE this is not a remedy for a tap that sits under the noise
/// floor in a genuinely noisy room — uniform gain scales signal and noise
/// alike, so the ceiling there is the mic's SNR, and the SNR check handles it.
const WEAK_RMS: f32 = 1e-6;
/// A tap must exceed the measured noise floor by this factor, else it is
/// masked by ambient noise (mirrors the onset detector's `ONSET_RATIO`).
const MASK_RATIO: f32 = 3.0;
/// Environment noise-floor RMS thresholds for the pass/warn/fail indicator.
const ENV_QUIET_MAX: f32 = 0.01;
const ENV_MODERATE_MAX: f32 = 0.05;
/// A zone whose leave-one-out accuracy is below this is flagged inconsistent.
const CONSISTENCY_MIN_ACCURACY: f32 = 0.8;

/// Per-tap feedback shown immediately during capture (`DESIGN.md`: "accepted /
/// too weak / clipped / masked by noise"). Never a generic failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapFeedback {
    Accepted,
    TooWeak,
    Clipped,
    MaskedByNoise,
}

impl TapFeedback {
    pub fn is_accepted(self) -> bool {
        matches!(self, TapFeedback::Accepted)
    }

    /// A short, specific reason for the non-accepted cases (for inline UI text).
    pub fn reason(self) -> &'static str {
        match self {
            TapFeedback::Accepted => "accepted",
            TapFeedback::TooWeak => "too weak — tap a little harder",
            TapFeedback::Clipped => "clipped — tap more gently or lower input gain",
            TapFeedback::MaskedByNoise => "masked by background noise",
        }
    }
}

/// Pass/warn/fail verdict on the room before calibration (`DESIGN.md`:
/// "a pass/warn/fail indicator before the user can proceed").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvironmentQuality {
    /// Quiet enough to calibrate well.
    Quiet,
    /// Usable but noisier than ideal — warn the user.
    Moderate,
    /// Too loud to calibrate reliably — block until resolved.
    TooNoisy,
}

impl EnvironmentQuality {
    pub fn passes(self) -> bool {
        !matches!(self, EnvironmentQuality::TooNoisy)
    }
}

/// Result of the leave-one-out consistency check across captured zones.
#[derive(Debug, Clone, PartialEq)]
pub struct ConsistencyReport {
    /// Leave-one-out accuracy in `[0, 1]` per zone, in `zone_ids` order.
    pub per_zone_accuracy: Vec<f32>,
    /// Indices of zones whose accuracy fell below the consistency threshold —
    /// the wizard names these specifically and offers to redo them.
    pub inconsistent_zones: Vec<usize>,
}

impl ConsistencyReport {
    pub fn all_consistent(&self) -> bool {
        self.inconsistent_zones.is_empty()
    }
}

/// Errors from the calibration domain. Real variants so the wizard can react
/// specifically (`ARCHITECTURE.md` → Error handling policy).
#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationError {
    /// Fewer than two zones — a classifier needs at least two to be meaningful.
    NotEnoughZones { have: usize },
    /// A zone has too few accepted samples to build or check.
    NotEnoughSamples {
        zone: usize,
        have: usize,
        need: usize,
    },
    /// Per-zone metadata (labels/layouts) didn't match the zone count.
    ZoneMetaMismatch { zones: usize, meta: usize },
    /// The number of per-zone sample groups didn't match the zone count.
    SampleGroupCountMismatch { zones: usize, groups: usize },
    /// The underlying classifier could not be fit.
    Training(ClassifierError),
}

impl std::fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalibrationError::NotEnoughZones { have } => {
                write!(f, "need at least 2 zones to calibrate, got {have}")
            }
            CalibrationError::NotEnoughSamples { zone, have, need } => write!(
                f,
                "zone {} has {have} samples, needs at least {need}",
                zone + 1
            ),
            CalibrationError::ZoneMetaMismatch { zones, meta } => {
                write!(f, "{meta} zone labels/layouts provided for {zones} zones")
            }
            CalibrationError::SampleGroupCountMismatch { zones, groups } => {
                write!(f, "{groups} sample groups provided for {zones} zones")
            }
            CalibrationError::Training(e) => write!(f, "could not train classifier: {e}"),
        }
    }
}

impl std::error::Error for CalibrationError {}

impl From<ClassifierError> for CalibrationError {
    fn from(e: ClassifierError) -> Self {
        CalibrationError::Training(e)
    }
}

// --- Pure analysis (free functions, fully unit-testable) -------------------

/// Classify the room noise floor (an RMS amplitude) into pass/warn/fail.
pub fn assess_environment(noise_rms: f32) -> EnvironmentQuality {
    if !noise_rms.is_finite() || noise_rms >= ENV_MODERATE_MAX {
        EnvironmentQuality::TooNoisy
    } else if noise_rms >= ENV_QUIET_MAX {
        EnvironmentQuality::Moderate
    } else {
        EnvironmentQuality::Quiet
    }
}

/// Judge one captured window for calibration quality, given the room's measured
/// noise floor (0.0 if unknown, which disables the masking check).
///
/// Loudness is measured over the **onset region** (the first ~15 ms of the
/// window, where the tap's transient actually is), not the whole 90 ms window.
/// A tap is a brief transient; averaging over the long silent decay tail
/// systematically under-reads its level, so whole-window RMS would reject
/// genuine taps — the very taps the onset detector already accepted, which
/// gates on the transient's per-hop energy — as `TooWeak` or `MaskedByNoise`.
/// Clipping is still checked against the peak anywhere in the window.
pub fn assess_tap(window: &[f32], noise_floor: f32) -> TapFeedback {
    if window.is_empty() {
        return TapFeedback::TooWeak;
    }
    let mut peak = 0.0f32;
    for &s in window {
        if !s.is_finite() {
            return TapFeedback::TooWeak;
        }
        peak = peak.max(s.abs());
    }
    let edge = onset_edge(window.len());
    let onset_rms = rms(&window[..edge]);

    if peak >= CLIP_PEAK {
        TapFeedback::Clipped
    } else if onset_rms < WEAK_RMS {
        TapFeedback::TooWeak
    } else if noise_floor > 0.0 && onset_rms < noise_floor * MASK_RATIO {
        TapFeedback::MaskedByNoise
    } else {
        TapFeedback::Accepted
    }
}

/// RMS of a slice.
fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|s| (*s as f64) * (*s as f64)).sum::<f64>() / x.len() as f64).sqrt() as f32
}

/// The onset-region span: the first ~15 ms of the window (matching the onset
/// detector's edge span), clamped to the window length.
fn onset_edge(n: usize) -> usize {
    ((SAMPLE_RATE_HZ as usize * 15) / 1000).clamp(1, n)
}

/// Leave-one-out cross-validation across the captured zones: for each sample,
/// retrain on every *other* sample and classify the held-out one, then report
/// per-zone accuracy and which zones fall below the consistency threshold.
///
/// `per_zone_samples[i]` are the accepted feature vectors for `zone_ids[i]`.
pub fn leave_one_out_consistency(
    zone_ids: &[ZoneId],
    per_zone_samples: &[Vec<FeatureVector>],
) -> Result<ConsistencyReport, CalibrationError> {
    validate_counts(zone_ids, per_zone_samples)?;

    let mut per_zone_accuracy = Vec::with_capacity(zone_ids.len());
    let mut inconsistent_zones = Vec::new();

    for (zi, samples) in per_zone_samples.iter().enumerate() {
        let mut correct = 0usize;
        for (held_out, held_fv) in samples.iter().enumerate() {
            let mut training: Vec<(ZoneId, FeatureVector)> = Vec::new();
            for (oz, others) in per_zone_samples.iter().enumerate() {
                for (oi, fv) in others.iter().enumerate() {
                    if oz == zi && oi == held_out {
                        continue; // leave this one out
                    }
                    training.push((zone_ids[oz], *fv));
                }
            }
            let clf = ClassifierParams::train(&training, &[])?;
            let out = clf.classify(held_fv);
            if out.accepted && out.zone_id == Some(zone_ids[zi]) {
                correct += 1;
            }
        }
        let accuracy = correct as f32 / samples.len() as f32;
        if accuracy < CONSISTENCY_MIN_ACCURACY {
            inconsistent_zones.push(zi);
        }
        per_zone_accuracy.push(accuracy);
    }

    Ok(ConsistencyReport {
        per_zone_accuracy,
        inconsistent_zones,
    })
}

/// Per-zone label + on-diagram layout, supplied at the save step.
#[derive(Debug, Clone, PartialEq)]
pub struct ZoneMeta {
    pub label: String,
    pub layout: ZoneLayout,
}

/// Build a persistable [`Profile`] from captured samples and (optional)
/// negative examples. Trains the classifier and assembles the zones with empty
/// action bindings (the user assigns actions afterward, per `DESIGN.md`).
#[allow(clippy::too_many_arguments)]
pub fn build_profile(
    id: crate::profile::ProfileId,
    name: String,
    device_fingerprint: DeviceFingerprint,
    sensing_mode: SensingMode,
    zone_ids: &[ZoneId],
    zone_meta: &[ZoneMeta],
    per_zone_samples: &[Vec<FeatureVector>],
    negatives: &[FeatureVector],
    created_at: DateTime<Utc>,
) -> Result<Profile, CalibrationError> {
    validate_counts(zone_ids, per_zone_samples)?;
    if zone_meta.len() != zone_ids.len() {
        return Err(CalibrationError::ZoneMetaMismatch {
            zones: zone_ids.len(),
            meta: zone_meta.len(),
        });
    }

    let mut training: Vec<(ZoneId, FeatureVector)> = Vec::new();
    for (zi, samples) in per_zone_samples.iter().enumerate() {
        for fv in samples {
            training.push((zone_ids[zi], *fv));
        }
    }
    let classifier = ClassifierParams::train(&training, negatives)?;

    let zones = zone_ids
        .iter()
        .zip(zone_meta.iter())
        .zip(per_zone_samples.iter())
        .map(|((id, meta), samples)| Zone {
            id: *id,
            label: meta.label.clone(),
            layout: meta.layout,
            actions: Vec::new(),
            calibration_sample_count: samples.len() as u32,
        })
        .collect();

    Ok(Profile {
        schema_version: PROFILE_SCHEMA_VERSION,
        id,
        name,
        created_at,
        updated_at: created_at,
        device_fingerprint,
        sensing_mode,
        zones,
        classifier,
        negative_examples_trained: !negatives.is_empty(),
    })
}

fn validate_counts(
    zone_ids: &[ZoneId],
    per_zone_samples: &[Vec<FeatureVector>],
) -> Result<(), CalibrationError> {
    if zone_ids.len() < 2 {
        return Err(CalibrationError::NotEnoughZones {
            have: zone_ids.len(),
        });
    }
    // per_zone_samples must align with zone_ids.
    if per_zone_samples.len() != zone_ids.len() {
        return Err(CalibrationError::SampleGroupCountMismatch {
            zones: zone_ids.len(),
            groups: per_zone_samples.len(),
        });
    }
    for (zi, samples) in per_zone_samples.iter().enumerate() {
        if samples.len() < MIN_SAMPLES_PER_ZONE {
            return Err(CalibrationError::NotEnoughSamples {
                zone: zi,
                have: samples.len(),
                need: MIN_SAMPLES_PER_ZONE,
            });
        }
    }
    Ok(())
}

// --- Stateful session ------------------------------------------------------

/// Collects samples across the wizard's steps. Owns a [`FeatureExtractor`] so
/// windows become feature vectors as they are accepted. The wizard reads its
/// progress (`zone_sample_count`, `current_zone_complete`) to drive the UI.
pub struct CalibrationSession {
    extractor: FeatureExtractor,
    zone_ids: Vec<ZoneId>,
    taps_per_zone: usize,
    noise_floor: f32,
    per_zone_samples: Vec<Vec<FeatureVector>>,
    negatives: Vec<FeatureVector>,
    current_zone: usize,
}

impl CalibrationSession {
    /// Start a session for the given zones. `noise_floor` is the RMS measured in
    /// the environment-check step (pass 0.0 to skip the masking check).
    pub fn new(zone_ids: Vec<ZoneId>, taps_per_zone: usize, noise_floor: f32) -> Self {
        // At least one zone is a precondition the wizard upholds (zones are
        // placed before capture starts). Enforce it here so a violation fails
        // loudly at construction, not as an out-of-bounds panic on the first
        // `record_tap`/`undo_last`/`redo_current_zone` (`CLAUDE.md`: fail fast).
        assert!(
            !zone_ids.is_empty(),
            "CalibrationSession requires at least one zone"
        );
        let n = zone_ids.len();
        Self {
            extractor: FeatureExtractor::new(),
            zone_ids,
            taps_per_zone,
            noise_floor,
            per_zone_samples: vec![Vec::new(); n],
            negatives: Vec::new(),
            current_zone: 0,
        }
    }

    pub fn zone_ids(&self) -> &[ZoneId] {
        &self.zone_ids
    }
    pub fn current_zone(&self) -> usize {
        self.current_zone
    }
    pub fn taps_per_zone(&self) -> usize {
        self.taps_per_zone
    }
    pub fn negatives(&self) -> &[FeatureVector] {
        &self.negatives
    }
    pub fn per_zone_samples(&self) -> &[Vec<FeatureVector>] {
        &self.per_zone_samples
    }

    /// Accepted-sample count for a zone.
    pub fn zone_sample_count(&self, zone: usize) -> usize {
        self.per_zone_samples.get(zone).map_or(0, Vec::len)
    }

    /// Whether the current zone has reached its target tap count.
    pub fn current_zone_complete(&self) -> bool {
        self.zone_sample_count(self.current_zone) >= self.taps_per_zone
    }

    /// Offer a captured window for the current zone. Returns the per-tap
    /// feedback; a full zone or a non-accepted quality both leave the sample
    /// count unchanged (an accepted tap past the target is ignored, not stored).
    pub fn record_tap(&mut self, window: &[f32]) -> TapFeedback {
        let feedback = assess_tap(window, self.noise_floor);
        if feedback.is_accepted()
            && window.len() == FEATURE_WINDOW_SAMPLES
            && !self.current_zone_complete()
        {
            let fv = self.extractor.extract(window);
            self.per_zone_samples[self.current_zone].push(fv);
        }
        feedback
    }

    /// Undo the most recently accepted tap in the current zone
    /// (`DESIGN.md`: "Undo last sample"). Returns whether one was removed.
    pub fn undo_last(&mut self) -> bool {
        self.per_zone_samples[self.current_zone].pop().is_some()
    }

    /// Clear the current zone's samples to recapture it (`DESIGN.md`: "redo a
    /// whole zone").
    pub fn redo_current_zone(&mut self) {
        self.per_zone_samples[self.current_zone].clear();
    }

    /// Jump back to a specific zone and clear it for recapture — used when the
    /// consistency check flags a zone and the user chooses to redo it
    /// (`DESIGN.md`: "names that zone specifically and offers to redo it").
    /// Out-of-range indices are ignored.
    pub fn restart_zone(&mut self, zone: usize) {
        if zone < self.zone_ids.len() {
            self.current_zone = zone;
            self.per_zone_samples[zone].clear();
        }
    }

    /// Advance to the next zone; returns `false` if already at the last zone.
    pub fn advance_zone(&mut self) -> bool {
        if self.current_zone + 1 < self.zone_ids.len() {
            self.current_zone += 1;
            true
        } else {
            false
        }
    }

    /// Record a negative (rejection) example from a captured window. Negatives
    /// are non-tap sounds (typing, talking) captured to reduce false accepts, so
    /// they bypass the tap-quality gate; only non-finite garbage is dropped.
    pub fn record_negative(&mut self, window: &[f32]) -> bool {
        if window.len() != FEATURE_WINDOW_SAMPLES {
            return false;
        }
        let fv = self.extractor.extract(window);
        if fv.is_finite() {
            self.negatives.push(fv);
            true
        } else {
            false
        }
    }

    /// Run the leave-one-out consistency check over the captured zones.
    pub fn consistency(&self) -> Result<ConsistencyReport, CalibrationError> {
        leave_one_out_consistency(&self.zone_ids, &self.per_zone_samples)
    }

    /// Finish: build the persistable profile from what was captured.
    pub fn build_profile(
        &self,
        id: crate::profile::ProfileId,
        name: String,
        device_fingerprint: DeviceFingerprint,
        sensing_mode: SensingMode,
        zone_meta: &[ZoneMeta],
        created_at: DateTime<Utc>,
    ) -> Result<Profile, CalibrationError> {
        build_profile(
            id,
            name,
            device_fingerprint,
            sensing_mode,
            &self.zone_ids,
            zone_meta,
            &self.per_zone_samples,
            &self.negatives,
            created_at,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{CEPSTRAL_COEFF_COUNT, SPECTRAL_BAND_COUNT, TEMPORAL_FEATURE_COUNT};

    fn zone(n: u128) -> ZoneId {
        ZoneId::from_uuid(uuid::Uuid::from_u128(n))
    }

    /// A tight, distinct synthetic cluster around a seed (same shape the
    /// classifier tests use), so consistency/build can be exercised directly.
    fn fv_from(seed: f32, jitter: f32) -> FeatureVector {
        let mut temporal = [0.0f32; TEMPORAL_FEATURE_COUNT];
        for (i, t) in temporal.iter_mut().enumerate() {
            *t = seed + i as f32 * 0.1 + jitter;
        }
        let mut spectral_bands = [0.0f32; SPECTRAL_BAND_COUNT];
        for (i, s) in spectral_bands.iter_mut().enumerate() {
            *s = (seed * 0.5 + i as f32 * 0.05 + jitter).abs();
        }
        let mut cepstral = [0.0f32; CEPSTRAL_COEFF_COUNT];
        for (i, c) in cepstral.iter_mut().enumerate() {
            *c = seed - i as f32 * 0.07 + jitter;
        }
        FeatureVector {
            temporal,
            spectral_bands,
            cepstral,
        }
    }

    fn cluster(seed: f32, n: usize) -> Vec<FeatureVector> {
        (0..n).map(|j| fv_from(seed, j as f32 * 1e-4)).collect()
    }

    #[test]
    fn environment_thresholds_map_to_pass_warn_fail() {
        assert_eq!(assess_environment(0.005), EnvironmentQuality::Quiet);
        assert_eq!(assess_environment(0.02), EnvironmentQuality::Moderate);
        assert_eq!(assess_environment(0.2), EnvironmentQuality::TooNoisy);
        assert!(!assess_environment(f32::NAN).passes());
    }

    #[test]
    fn tap_quality_distinguishes_the_four_cases() {
        let n = FEATURE_WINDOW_SAMPLES;
        // A clean mid-level tap on a quiet floor.
        let good: Vec<f32> = (0..n).map(|i| 0.3 * (i as f32 * 0.05).sin()).collect();
        assert_eq!(assess_tap(&good, 0.001), TapFeedback::Accepted);
        // Clipped: reaches full scale.
        let clipped = vec![1.0f32; n];
        assert_eq!(assess_tap(&clipped, 0.001), TapFeedback::Clipped);
        // Too weak: essentially digital silence (below the silence guard).
        let weak = vec![1e-7f32; n];
        assert_eq!(assess_tap(&weak, 0.0), TapFeedback::TooWeak);
        // Masked: a real-ish level but the room floor is nearly as loud.
        assert_eq!(assess_tap(&good, 0.2), TapFeedback::MaskedByNoise);
    }

    #[test]
    fn low_gain_mic_tap_is_accepted_over_a_quiet_room() {
        // Reference hardware: a Razer USB headset whose accepted desk taps have
        // an onset-region RMS near -87 dBFS (~4.5e-5) over a near-silent room
        // (~-140 dBFS). This tap MUST be Accepted — the old WEAK_RMS=0.01 made
        // it TooWeak and calibration impossible on the real device.
        let n = FEATURE_WINDOW_SAMPLES;
        let edge = 600.min(n);
        let mut w = vec![0.0f32; n];
        for s in w.iter_mut().take(edge) {
            *s = 4.5e-5;
        }
        assert_eq!(assess_tap(&w, 1e-7), TapFeedback::Accepted);
        // Genuine digital silence is still rejected.
        assert_eq!(assess_tap(&vec![0.0f32; n], 0.0), TapFeedback::TooWeak);
        // And a tap only marginally above a loud room floor is still masked.
        assert_eq!(assess_tap(&w, 4.5e-5), TapFeedback::MaskedByNoise);
    }

    #[test]
    fn brief_transient_tap_is_accepted_not_weak_or_masked() {
        // A real tap: a short, clearly-audible onset followed by a long silent
        // decay tail. Its whole-90ms-window RMS is small, but its onset-region
        // energy is strong. The old whole-window measure rejected such taps as
        // MaskedByNoise (and, when softer, TooWeak); measuring the onset region
        // accepts them — matching what the onset detector already accepted.
        let n = FEATURE_WINDOW_SAMPLES;
        let onset = 200.min(n);
        let mut w = vec![0.0f32; n];
        for s in w.iter_mut().take(onset) {
            *s = 0.1;
        }
        let whole_rms =
            (w.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / n as f64).sqrt() as f32;
        // The whole-window RMS is under the old masking threshold at this floor,
        // which is exactly what used to (wrongly) reject the tap.
        assert!(whole_rms < 0.01 * MASK_RATIO, "whole_rms {whole_rms}");
        assert_eq!(assess_tap(&w, 0.0), TapFeedback::Accepted);
        assert_eq!(assess_tap(&w, 0.01), TapFeedback::Accepted);
    }

    #[test]
    fn consistent_zones_pass_leave_one_out() {
        let ids = [zone(1), zone(2), zone(3)];
        let samples = [cluster(0.0, 8), cluster(6.0, 8), cluster(12.0, 8)];
        let report = leave_one_out_consistency(&ids, &samples).unwrap();
        assert!(
            report.all_consistent(),
            "tight clusters should be consistent"
        );
        assert!(report.per_zone_accuracy.iter().all(|&a| a >= 0.8));
    }

    #[test]
    fn inconsistent_zone_is_flagged() {
        // Zone 0 is calibrated from two very different physical spots (two
        // clusters), exactly the ACCEPTANCE.md injected-inconsistency test.
        let ids = [zone(1), zone(2)];
        let mut split_zone = cluster(0.0, 5);
        split_zone.extend(cluster(12.0, 5)); // second spot, collides with zone 2
        let zone2 = cluster(12.0, 10);
        let report = leave_one_out_consistency(&ids, &[split_zone, zone2]).unwrap();
        assert!(
            report.inconsistent_zones.contains(&0),
            "the split zone must be flagged: {report:?}"
        );
    }

    #[test]
    fn too_few_zones_or_samples_are_named_errors() {
        assert!(matches!(
            leave_one_out_consistency(&[zone(1)], &[cluster(0.0, 4)]),
            Err(CalibrationError::NotEnoughZones { have: 1 })
        ));
        assert!(matches!(
            leave_one_out_consistency(&[zone(1), zone(2)], &[cluster(0.0, 1), cluster(5.0, 4)]),
            Err(CalibrationError::NotEnoughSamples {
                zone: 0,
                have: 1,
                ..
            })
        ));
    }

    #[test]
    fn sample_group_count_mismatch_is_a_distinct_named_error() {
        // Two zones but only one sample group: the diagnostic must name the
        // sample-group mismatch, not reuse the zone labels/layouts message.
        let err = leave_one_out_consistency(&[zone(1), zone(2)], &[cluster(0.0, 4)]).unwrap_err();
        assert!(
            matches!(
                err,
                CalibrationError::SampleGroupCountMismatch {
                    zones: 2,
                    groups: 1
                }
            ),
            "expected a sample-group mismatch, got {err:?}"
        );
        assert!(err.to_string().contains("sample groups"));
    }

    #[test]
    fn build_profile_produces_a_classifying_profile() {
        let ids = [zone(1), zone(2)];
        let samples = [cluster(0.0, 10), cluster(8.0, 10)];
        let meta = [
            ZoneMeta {
                label: "Left".into(),
                layout: ZoneLayout {
                    x: 0.0,
                    y: 0.0,
                    width: 0.5,
                    height: 1.0,
                },
            },
            ZoneMeta {
                label: "Right".into(),
                layout: ZoneLayout {
                    x: 0.5,
                    y: 0.0,
                    width: 0.5,
                    height: 1.0,
                },
            },
        ];
        let now = DateTime::parse_from_rfc3339("2026-07-21T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let profile = build_profile(
            crate::profile::ProfileId::from_uuid(uuid::Uuid::from_u128(99)),
            "Test desk".into(),
            DeviceFingerprint {
                device_name: "Built-in Microphone".into(),
                capability_hash: "hash".into(),
            },
            SensingMode::Passive,
            &ids,
            &meta,
            &samples,
            &[],
            now,
        )
        .unwrap();

        assert_eq!(profile.zones.len(), 2);
        assert_eq!(profile.zones[0].calibration_sample_count, 10);
        // The trained classifier assigns a fresh sample near zone 0 to zone 0.
        let out = profile.classifier.classify(&fv_from(0.0, 1e-3));
        assert_eq!(out.zone_id, Some(zone(1)));
        // And it round-trips through JSON with no field loss.
        let back = Profile::from_json(&profile.to_json(), "p.json").unwrap();
        assert_eq!(back, profile);
    }

    #[test]
    #[should_panic(expected = "at least one zone")]
    fn new_rejects_zero_zones() {
        // Zero zones would index-panic on the first tap; construction must
        // reject it up front instead (fail fast).
        CalibrationSession::new(vec![], 3, 0.0);
    }

    #[test]
    fn session_captures_and_advances() {
        let n = FEATURE_WINDOW_SAMPLES;
        let good: Vec<f32> = (0..n).map(|i| 0.3 * (i as f32 * 0.05).sin()).collect();
        let mut s = CalibrationSession::new(vec![zone(1), zone(2)], 3, 0.0);

        assert_eq!(s.record_tap(&good), TapFeedback::Accepted);
        assert_eq!(s.record_tap(&good), TapFeedback::Accepted);
        assert_eq!(s.zone_sample_count(0), 2);
        assert!(s.undo_last());
        assert_eq!(s.zone_sample_count(0), 1);

        // Fill to target; further accepted taps are ignored (not stored).
        s.record_tap(&good);
        s.record_tap(&good);
        assert!(s.current_zone_complete());
        s.record_tap(&good);
        assert_eq!(s.zone_sample_count(0), 3);

        assert!(s.advance_zone());
        assert_eq!(s.current_zone(), 1);
        assert!(!s.advance_zone()); // last zone
    }
}
