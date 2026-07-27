//! Held-out evaluation: the pure aggregation behind the Evaluation screen
//! (`DESIGN.md` → Evaluation, `ROADMAP.md` Phase 6) and the exact persisted
//! [`EvaluationReport`] of `DATA_MODEL.md`.
//!
//! This module is deliberately device- and UI-free, the same discipline as the
//! rest of `qwerty-core`: the guided test in `qwerty-app` captures real taps,
//! classifies each one against the profile's classifier, and feeds the outcome
//! here one tap at a time via [`EvaluationAccumulator::record`]. The accumulator
//! owns *only* the tallying — confusion matrix, per-zone and overall accuracy,
//! rejection count, confidence distribution, and latency percentiles — so every
//! one of those computations is unit-testable against synthetic outcomes with no
//! audio, no egui, and no real classifier involved.
//!
//! The report a run produces is the only accuracy evidence this project trusts
//! (`ACCEPTANCE.md`: "a theory is not evidence; a saved evaluation report is").
//! It is persisted per-profile at `Evaluations/<profile-id>/<timestamp>.json`
//! (`DATA_MODEL.md`), using the same schema-version + atomic-write policy as
//! `Profile`/`Config` (`crate::profile`), so a torn write can never damage a
//! prior report and a version bump refuses to load rather than silently coerce.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::classifier::Classification;
use crate::profile::{parse_versioned, write_atomic, PersistenceError, ProfileId, ZoneId};

/// Current schema version of a persisted [`EvaluationReport`]. Bumping it
/// requires a migration or an explicit refusal-to-load (`DATA_MODEL.md`).
pub const EVALUATION_SCHEMA_VERSION: u32 = 1;

/// Default held-out taps captured per zone (`ACCEPTANCE.md`: "default 15
/// taps/zone"). The wizard's *calibration* default is smaller
/// ([`crate::calibration::DEFAULT_TAPS_PER_ZONE`]); a held-out evaluation asks
/// for more so the accuracy estimate has tighter error bars.
pub const DEFAULT_EVAL_TAPS_PER_ZONE: u32 = 15;

/// A fixed-range histogram: `bins.len()` equal-width buckets spanning
/// `[min, max]`. `DATA_MODEL.md` names `confidence_distribution: Histogram` but
/// leaves its shape to the implementation; this is that shape — deliberately
/// general (a min/max range plus counts) rather than hard-coding the `[0, 1]`
/// confidence range, so the same type serves any future distribution view.
///
/// (Its full shape is recorded in `DATA_MODEL.md` under "Evaluation report"
/// alongside the field it fills, per the rule that a type the document only
/// named gets written down in both places in the same change.)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Histogram {
    pub min: f32,
    pub max: f32,
    /// Bucket counts, low bucket first. `bins[i]` covers
    /// `[min + i*w, min + (i+1)*w)` where `w = (max - min) / bins.len()`, with
    /// the top bucket closed on the right so `max` itself lands in the last bin.
    pub bins: Vec<u32>,
}

impl Histogram {
    /// Build a histogram of `values` over `[min, max]` with `bin_count` buckets.
    /// Values outside the range are clamped into the end buckets rather than
    /// dropped, so no observation silently disappears (`CLAUDE.md`: no silent
    /// discards). Non-finite values are ignored — they are not observations.
    pub fn from_values(values: &[f32], bin_count: usize, min: f32, max: f32) -> Self {
        let bin_count = bin_count.max(1);
        let mut bins = vec![0u32; bin_count];
        let span = (max - min).max(f32::MIN_POSITIVE);
        for &v in values {
            if !v.is_finite() {
                continue;
            }
            let frac = ((v - min) / span).clamp(0.0, 1.0);
            // frac == 1.0 (v == max) would index one past the end; clamp it into
            // the last bucket so the top of the range is counted, not lost.
            let idx = ((frac * bin_count as f32) as usize).min(bin_count - 1);
            bins[idx] += 1;
        }
        Self { min, max, bins }
    }

    /// Total observations recorded across all buckets.
    pub fn total(&self) -> u32 {
        self.bins.iter().sum()
    }
}

/// Latency percentiles over a run, in milliseconds (`DATA_MODEL.md`). The
/// acceptance target is median < 120 ms and p95 < 200 ms (`ACCEPTANCE.md`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LatencyStats {
    pub p50: f32,
    pub p95: f32,
    pub p99: f32,
    pub max: f32,
}

impl LatencyStats {
    /// Percentiles from per-tap latencies (any order; non-finite values are
    /// dropped). Uses the nearest-rank method on the sorted samples — for the
    /// small held-out sets here (tens to low hundreds of taps) it is the honest,
    /// assumption-free choice; interpolation would invent values between taps.
    /// An empty set yields all-zero stats (a run with no taps has no latency).
    pub fn from_millis(samples: &[f32]) -> Self {
        let mut xs: Vec<f32> = samples.iter().copied().filter(|s| s.is_finite()).collect();
        if xs.is_empty() {
            return Self {
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
                max: 0.0,
            };
        }
        xs.sort_by(f32::total_cmp);
        let n = xs.len();
        let pick = |p: f32| -> f32 {
            // Nearest-rank: rank = ceil(p * n), 1-based, clamped into range.
            let rank = (p * n as f32).ceil() as usize;
            let idx = rank.saturating_sub(1).min(n - 1);
            xs[idx]
        };
        Self {
            p50: pick(0.50),
            p95: pick(0.95),
            p99: pick(0.99),
            max: xs[n - 1],
        }
    }
}

/// The persisted result of one held-out evaluation run (`DATA_MODEL.md`).
///
/// `confusion_matrix` rows are the *actual* zone and columns the *predicted*
/// zone, both indexed in the profile's `zones` order (never the classifier's
/// private zone ordering — the eval lives outside `classifier` and must not peek
/// at [`crate::classifier::ClassifierParams`] internals). A rejected tap (the
/// novelty gate fired) is *not* placed in the matrix; it is counted in
/// `rejected_tap_count` and still counts against the zone's accuracy, because a
/// held-out tap the system failed to recognize is a miss from the user's point
/// of view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub schema_version: u32,
    pub profile_id: ProfileId,
    pub run_at: DateTime<Utc>,
    pub taps_per_zone: u32,
    pub per_zone_accuracy: HashMap<ZoneId, f32>,
    pub confusion_matrix: Vec<Vec<u32>>,
    pub rejected_tap_count: u32,
    pub confidence_distribution: Histogram,
    pub latency_ms: LatencyStats,
    pub overall_accuracy: f32,
}

impl EvaluationReport {
    /// Serialize to pretty JSON (the on-disk form).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("EvaluationReport serializes")
    }

    /// Parse from JSON with the shared schema-version policy applied, then check
    /// structural integrity. A file whose version matches but whose contents are
    /// internally inconsistent — a ragged (non-square) confusion matrix, or a
    /// per-zone map that disagrees with the matrix size — is *refused* with
    /// [`PersistenceError::InvalidData`] rather than loaded and silently
    /// mis-rendered, the same loudly-refuse policy `Profile::from_json` applies
    /// to its classifier blob.
    pub fn from_json(s: &str, file: &str) -> Result<Self, PersistenceError> {
        let report: Self = parse_versioned(s, EVALUATION_SCHEMA_VERSION, file)?;
        report
            .validate()
            .map_err(|reason| PersistenceError::InvalidData {
                file: file.to_string(),
                reason,
            })?;
        Ok(report)
    }

    /// Structural invariants of a report: the confusion matrix is square, and
    /// `per_zone_accuracy` carries exactly one entry per matrix row/column.
    fn validate(&self) -> Result<(), String> {
        let n = self.confusion_matrix.len();
        for (i, row) in self.confusion_matrix.iter().enumerate() {
            if row.len() != n {
                return Err(format!(
                    "confusion_matrix row {i} has {} cells, expected {n} \
                     (the matrix must be square)",
                    row.len()
                ));
            }
        }
        if self.per_zone_accuracy.len() != n {
            return Err(format!(
                "per_zone_accuracy has {} entries but the confusion matrix is {n}x{n}",
                self.per_zone_accuracy.len()
            ));
        }
        Ok(())
    }

    /// Load from a file path, applying the schema-version policy.
    pub fn load(path: &std::path::Path) -> Result<Self, PersistenceError> {
        let s = std::fs::read_to_string(path).map_err(PersistenceError::Io)?;
        Self::from_json(&s, &path.display().to_string())
    }

    /// Write to a file path atomically (temp + rename), as pretty JSON.
    pub fn save(&self, path: &std::path::Path) -> Result<(), PersistenceError> {
        write_atomic(path, &self.to_json()).map_err(PersistenceError::Io)
    }
}

/// Number of confidence-distribution buckets (`[0, 1]` split into tenths).
const CONFIDENCE_BINS: usize = 10;

/// Accumulates one held-out run tap-by-tap, then folds the tallies into an
/// [`EvaluationReport`]. Construct with the profile's zone ordering; feed each
/// captured tap's outcome via [`record`](Self::record) (or the classifier-typed
/// [`record_classification`](Self::record_classification)); call
/// [`finish`](Self::finish) once the guided test completes.
pub struct EvaluationAccumulator {
    zone_ids: Vec<ZoneId>,
    index: HashMap<ZoneId, usize>,
    taps_per_zone: u32,
    /// `matrix[actual][predicted]` — accepted taps only.
    matrix: Vec<Vec<u32>>,
    /// Rejected taps attributed to their actual zone (novelty gate fired).
    rejected_per_zone: Vec<u32>,
    /// Confidence of every accepted tap (rejected taps have no meaningful
    /// confidence, so they are not recorded here — only in the matrix denominator
    /// via `rejected_per_zone`).
    confidences: Vec<f32>,
    /// Per-tap latency in ms for *every* presented tap (accepted or rejected):
    /// latency is a pipeline property independent of whether the guess was right.
    /// Retained in full so the live screen can draw a latency histogram during
    /// the run; the saved report keeps only the percentiles ([`LatencyStats`]).
    latencies_ms: Vec<f32>,
}

impl EvaluationAccumulator {
    /// Start a run over `zone_ids` (the profile's `zones` order). `taps_per_zone`
    /// is recorded into the report for context; it does not gate `record`.
    pub fn new(zone_ids: Vec<ZoneId>, taps_per_zone: u32) -> Self {
        let n = zone_ids.len();
        let index = zone_ids.iter().enumerate().map(|(i, z)| (*z, i)).collect();
        Self {
            zone_ids,
            index,
            taps_per_zone,
            matrix: vec![vec![0u32; n]; n],
            rejected_per_zone: vec![0u32; n],
            confidences: Vec::new(),
            latencies_ms: Vec::new(),
        }
    }

    /// The per-tap latencies recorded so far, for a live in-run histogram (the
    /// saved report stores only percentiles, so this is the only source for one).
    pub fn latencies_ms(&self) -> &[f32] {
        &self.latencies_ms
    }

    /// Record one held-out tap: the `actual` zone the user was asked to tap, the
    /// `predicted` zone (`None` when the novelty gate rejected it), the model
    /// `confidence` (ignored when `predicted` is `None`), and the end-to-end
    /// `latency_ms`.
    ///
    /// # Panics
    /// Panics if `actual` is not one of this run's zones. That is a caller bug,
    /// not a runtime condition (the guided test only ever asks for zones from
    /// the profile the accumulator was built from), so it fails fast in *every*
    /// build — the same policy as [`crate::features::FeatureExtractor::extract`]
    /// on a wrong-length window. It deliberately does **not** use
    /// `debug_assert!`, which would compile away in release and let the tap be
    /// dropped silently, undercounting the report (`CLAUDE.md`: a precondition
    /// that isn't met must surface immediately and loudly, never be quietly
    /// worked around).
    ///
    /// A `predicted` zone not in the set (impossible for a classifier trained on
    /// this profile) is treated as a rejection so it can never index out of
    /// bounds.
    pub fn record(
        &mut self,
        actual: ZoneId,
        predicted: Option<ZoneId>,
        confidence: f32,
        latency_ms: f32,
    ) {
        let ai = *self
            .index
            .get(&actual)
            .expect("evaluation tap recorded for a zone that is not in this run");
        self.latencies_ms.push(latency_ms);
        match predicted.and_then(|p| self.index.get(&p).copied()) {
            Some(pi) => {
                self.matrix[ai][pi] += 1;
                self.confidences.push(confidence);
            }
            None => self.rejected_per_zone[ai] += 1,
        }
    }

    /// Convenience for the app: record straight from a classifier
    /// [`Classification`]. A tap that failed the novelty gate (`!accepted`) is a
    /// rejection regardless of the winning centroid.
    pub fn record_classification(
        &mut self,
        actual: ZoneId,
        outcome: Classification,
        latency_ms: f32,
    ) {
        let predicted = if outcome.accepted {
            outcome.zone_id
        } else {
            None
        };
        self.record(actual, predicted, outcome.confidence, latency_ms);
    }

    /// Total taps presented for a zone (correct + misclassified + rejected).
    fn presented(&self, zone_idx: usize) -> u32 {
        self.matrix[zone_idx].iter().sum::<u32>() + self.rejected_per_zone[zone_idx]
    }

    /// Fold the tallies into the persistable report. `run_at` is passed in (the
    /// caller stamps it) so this stays free of the wall clock and remains a pure
    /// function of what was recorded — matching how the rest of `qwerty-core`
    /// keeps timestamps at the boundary.
    ///
    /// Convention for `per_zone_accuracy`: a zone with zero presented taps is
    /// reported as `0.0`, the same value as a zone that was tested and missed
    /// every tap. The persisted schema is `HashMap<ZoneId, f32>`
    /// (`DATA_MODEL.md`), so the two cases are not distinguishable in the report
    /// — this is sound only because a real run always presents every zone: the
    /// guided test's "Finish" action is gated on every zone having reached
    /// `taps_per_zone`. Call `finish` only on a complete run; a partial run's
    /// report would understate the unpresented zones rather than mark them
    /// as unmeasured.
    pub fn finish(&self, profile_id: ProfileId, run_at: DateTime<Utc>) -> EvaluationReport {
        let n = self.zone_ids.len();

        let mut per_zone_accuracy = HashMap::with_capacity(n);
        let mut correct_total = 0u32;
        let mut presented_total = 0u32;
        for i in 0..n {
            let correct = self.matrix[i][i];
            let presented = self.presented(i);
            correct_total += correct;
            presented_total += presented;
            let acc = if presented > 0 {
                correct as f32 / presented as f32
            } else {
                0.0
            };
            per_zone_accuracy.insert(self.zone_ids[i], acc);
        }

        let overall_accuracy = if presented_total > 0 {
            correct_total as f32 / presented_total as f32
        } else {
            0.0
        };

        EvaluationReport {
            schema_version: EVALUATION_SCHEMA_VERSION,
            profile_id,
            run_at,
            taps_per_zone: self.taps_per_zone,
            per_zone_accuracy,
            confusion_matrix: self.matrix.clone(),
            rejected_tap_count: self.rejected_per_zone.iter().sum(),
            confidence_distribution: Histogram::from_values(
                &self.confidences,
                CONFIDENCE_BINS,
                0.0,
                1.0,
            ),
            latency_ms: LatencyStats::from_millis(&self.latencies_ms),
            overall_accuracy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn zone(n: u128) -> ZoneId {
        ZoneId::from_uuid(Uuid::from_u128(n))
    }

    #[test]
    fn histogram_buckets_and_clamps() {
        // 10 buckets over [0,1]. 0.0 -> bin 0, 1.0 -> last bin (not out of range),
        // 0.55 -> bin 5, out-of-range values clamp into the end buckets.
        let h = Histogram::from_values(&[0.0, 1.0, 0.55, -3.0, 9.0, f32::NAN], 10, 0.0, 1.0);
        assert_eq!(h.bins.len(), 10);
        assert_eq!(h.bins[0], 2); // 0.0 and the clamped -3.0
        assert_eq!(h.bins[5], 1); // 0.55
        assert_eq!(h.bins[9], 2); // 1.0 and the clamped 9.0
        assert_eq!(h.total(), 5); // NaN dropped
    }

    #[test]
    fn latency_percentiles_are_nearest_rank() {
        let samples: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let s = LatencyStats::from_millis(&samples);
        assert_eq!(s.p50, 50.0);
        assert_eq!(s.p95, 95.0);
        assert_eq!(s.p99, 99.0);
        assert_eq!(s.max, 100.0);
    }

    #[test]
    fn empty_latency_is_all_zero() {
        let s = LatencyStats::from_millis(&[]);
        assert_eq!((s.p50, s.p95, s.p99, s.max), (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn perfect_run_is_100_percent_with_identity_matrix() {
        let (za, zb) = (zone(1), zone(2));
        let mut acc = EvaluationAccumulator::new(vec![za, zb], 3);
        for _ in 0..3 {
            acc.record(za, Some(za), 0.9, 40.0);
            acc.record(zb, Some(zb), 0.85, 45.0);
        }
        let now = DateTime::parse_from_rfc3339("2026-07-24T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let r = acc.finish(ProfileId::from_uuid(Uuid::from_u128(9)), now);
        assert_eq!(r.confusion_matrix, vec![vec![3, 0], vec![0, 3]]);
        assert_eq!(r.overall_accuracy, 1.0);
        assert_eq!(r.per_zone_accuracy[&za], 1.0);
        assert_eq!(r.rejected_tap_count, 0);
        assert_eq!(r.confidence_distribution.total(), 6);
    }

    #[test]
    fn misclassification_and_rejection_lower_accuracy() {
        // Zone A: 2 correct, 1 predicted as B, 1 rejected -> 2/4 = 0.5.
        // Zone B: 4 correct -> 1.0. Overall 6/8 = 0.75.
        let (za, zb) = (zone(1), zone(2));
        let mut acc = EvaluationAccumulator::new(vec![za, zb], 4);
        acc.record(za, Some(za), 0.8, 50.0);
        acc.record(za, Some(za), 0.8, 50.0);
        acc.record(za, Some(zb), 0.4, 60.0); // misclassified
        acc.record(za, None, 0.0, 55.0); // rejected
        for _ in 0..4 {
            acc.record(zb, Some(zb), 0.9, 45.0);
        }
        let now = Utc::now();
        let r = acc.finish(ProfileId::from_uuid(Uuid::from_u128(1)), now);
        assert_eq!(r.confusion_matrix, vec![vec![2, 1], vec![0, 4]]);
        assert_eq!(r.rejected_tap_count, 1);
        assert_eq!(r.per_zone_accuracy[&za], 0.5);
        assert_eq!(r.per_zone_accuracy[&zb], 1.0);
        assert_eq!(r.overall_accuracy, 0.75);
        // Only the 7 accepted taps have a recorded confidence; the rejection has none.
        assert_eq!(r.confidence_distribution.total(), 7);
    }

    #[test]
    fn report_round_trips_through_json() {
        let (za, zb) = (zone(1), zone(2));
        let mut acc = EvaluationAccumulator::new(vec![za, zb], 2);
        acc.record(za, Some(za), 0.7, 40.0);
        acc.record(zb, None, 0.0, 90.0);
        let now = DateTime::parse_from_rfc3339("2026-07-24T09:30:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let r = acc.finish(ProfileId::from_uuid(Uuid::from_u128(5)), now);
        let back = EvaluationReport::from_json(&r.to_json(), "eval.json").unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn schema_mismatch_is_a_named_error() {
        let (za, zb) = (zone(1), zone(2));
        let mut acc = EvaluationAccumulator::new(vec![za, zb], 1);
        acc.record(za, Some(za), 0.6, 30.0);
        acc.record(zb, Some(zb), 0.6, 30.0);
        let r = acc.finish(ProfileId::from_uuid(Uuid::from_u128(2)), Utc::now());
        let mut v: serde_json::Value = serde_json::from_str(&r.to_json()).unwrap();
        v["schema_version"] = serde_json::json!(999);
        let err = EvaluationReport::from_json(&v.to_string(), "eval.json").unwrap_err();
        assert!(matches!(err, PersistenceError::SchemaMismatch { .. }));
    }

    #[test]
    #[should_panic(expected = "not in this run")]
    fn recording_a_zone_outside_the_run_fails_fast() {
        // A precondition violation must surface loudly in EVERY build, not be
        // dropped silently in release (which `debug_assert!` would have done).
        let (za, zb) = (zone(1), zone(2));
        let mut acc = EvaluationAccumulator::new(vec![za], 5);
        acc.record(zb, Some(za), 0.9, 40.0);
    }

    #[test]
    fn structurally_inconsistent_report_is_refused_on_load() {
        let (za, zb) = (zone(1), zone(2));
        let mut acc = EvaluationAccumulator::new(vec![za, zb], 1);
        acc.record(za, Some(za), 0.8, 40.0);
        acc.record(zb, Some(zb), 0.8, 40.0);
        let r = acc.finish(ProfileId::from_uuid(Uuid::from_u128(7)), Utc::now());

        // A ragged (non-square) confusion matrix is a named refusal, not a
        // silently-loaded report that would later mis-render.
        let mut v: serde_json::Value = serde_json::from_str(&r.to_json()).unwrap();
        v["confusion_matrix"] = serde_json::json!([[1, 0], [0]]);
        let err = EvaluationReport::from_json(&v.to_string(), "eval.json").unwrap_err();
        assert!(
            matches!(err, PersistenceError::InvalidData { .. }),
            "got {err:?}"
        );

        // So is a per-zone map that disagrees with the matrix size.
        let mut v2: serde_json::Value = serde_json::from_str(&r.to_json()).unwrap();
        v2["per_zone_accuracy"] = serde_json::json!({ za.to_string(): 1.0 });
        let err2 = EvaluationReport::from_json(&v2.to_string(), "eval.json").unwrap_err();
        assert!(
            matches!(err2, PersistenceError::InvalidData { .. }),
            "got {err2:?}"
        );
    }

    #[test]
    fn record_classification_maps_rejected_to_none() {
        let (za, zb) = (zone(1), zone(2));
        let mut acc = EvaluationAccumulator::new(vec![za, zb], 1);
        // Accepted classification counts in the matrix.
        acc.record_classification(
            za,
            Classification {
                zone_id: Some(za),
                confidence: 0.9,
                accepted: true,
            },
            40.0,
        );
        // A rejected classification (accepted=false) is a rejection even though a
        // centroid "won" — the novelty gate is authoritative.
        acc.record_classification(
            zb,
            Classification {
                zone_id: Some(zb),
                confidence: 0.9,
                accepted: false,
            },
            40.0,
        );
        let r = acc.finish(ProfileId::from_uuid(Uuid::from_u128(3)), Utc::now());
        assert_eq!(r.confusion_matrix, vec![vec![1, 0], vec![0, 0]]);
        assert_eq!(r.rejected_tap_count, 1);
    }
}
