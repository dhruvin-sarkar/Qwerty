//! Zone classifier: a regularized linear model (nearest-centroid in a
//! standardized feature space) paired with a nearest-example novelty gate for
//! rejecting sounds that don't resemble any calibrated zone.
//!
//! See `ARCHITECTURE.md` (classifier.rs) and `DATA_MODEL.md`
//! ([`ClassifierParams`] is the opaque, versioned blob persisted inside a
//! `Profile`; nothing outside this module pattern-matches its internals).

use serde::{Deserialize, Serialize};

use crate::features::{FeatureVector, FEATURE_DIM};
use crate::profile::ZoneId;

/// Schema version for the persisted classifier blob. Bumped only with an
/// explicit migration (see `DATA_MODEL.md`).
pub const CLASSIFIER_SCHEMA_VERSION: u32 = 1;

/// Std-dev floor: doubles as the ridge-style regularizer (keeps
/// standardization finite for near-constant features and shrinks their
/// influence).
const STD_FLOOR: f32 = 1e-6;

/// How many standard deviations beyond the mean intra-zone spread a query may
/// sit before the novelty gate rejects it, when no negative examples pin the
/// threshold tighter.
const NOVELTY_SIGMA: f32 = 4.0;

/// Softmax temperature for turning centroid distances into a confidence.
const CONFIDENCE_BETA: f32 = 1.0;

/// Errors that can arise while fitting or loading a classifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassifierError {
    /// No labeled samples were provided.
    EmptyTrainingSet,
    /// Training produced a non-finite parameter (NaN/inf) — refuse it rather
    /// than persist unusable data.
    NonFiniteParameters,
    /// Loaded parameters are internally inconsistent (e.g. a stale profile from
    /// a build with a different `FEATURE_DIM`). The string names the mismatch.
    MalformedParameters(String),
}

impl std::fmt::Display for ClassifierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClassifierError::EmptyTrainingSet => write!(f, "classifier training set was empty"),
            ClassifierError::NonFiniteParameters => {
                write!(f, "classifier training produced non-finite parameters")
            }
            ClassifierError::MalformedParameters(d) => {
                write!(f, "classifier parameters are malformed: {d}")
            }
        }
    }
}

impl std::error::Error for ClassifierError {}

/// A standardized calibration example, retained for the nearest-example
/// novelty gate. Stored as a plain float vector (not the `FeatureVector`
/// type) so the persisted `Profile` stays decoupled from the DSP core's
/// evolving feature layout (`DATA_MODEL.md`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct StdExample {
    zone: usize,
    v: Vec<f32>,
}

/// Fitted classifier parameters. Persisted inside a `Profile` and treated as
/// opaque everywhere except this module.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassifierParams {
    schema_version: u32,
    feature_mean: Vec<f32>,
    feature_std: Vec<f32>,
    zone_ids: Vec<ZoneId>,
    centroids: Vec<Vec<f32>>,
    examples: Vec<StdExample>,
    novelty_threshold: f32,
    negative_examples_trained: bool,
}

/// Outcome of classifying one feature vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Classification {
    /// The winning zone, or `None` if the novelty gate rejected the input.
    pub zone_id: Option<ZoneId>,
    /// Confidence in `[0, 1]` for the winning zone (meaningful only when
    /// `accepted`).
    pub confidence: f32,
    /// Whether the input passed the novelty gate.
    pub accepted: bool,
}

impl ClassifierParams {
    /// Whether this profile was trained with negative (rejection) examples.
    pub fn negative_examples_trained(&self) -> bool {
        self.negative_examples_trained
    }

    /// Number of zones this classifier can assign.
    pub fn zone_count(&self) -> usize {
        self.zone_ids.len()
    }

    /// Fit the classifier from labeled calibration samples and optional
    /// negative (rejection) examples.
    ///
    /// `negatives` are non-tap sounds that must be rejected; when present they
    /// pull the novelty threshold inward so those sounds fall outside it,
    /// without letting it drop below the real taps' own spread.
    pub fn train(
        samples: &[(ZoneId, FeatureVector)],
        negatives: &[FeatureVector],
    ) -> Result<Self, ClassifierError> {
        if samples.is_empty() {
            return Err(ClassifierError::EmptyTrainingSet);
        }

        // Stable zone ordering: first-seen order across the samples.
        let mut zone_ids: Vec<ZoneId> = Vec::new();
        for (zid, _) in samples {
            if !zone_ids.contains(zid) {
                zone_ids.push(*zid);
            }
        }
        let zone_index = |zid: &ZoneId| zone_ids.iter().position(|z| z == zid).unwrap();

        let raw: Vec<[f32; FEATURE_DIM]> = samples.iter().map(|(_, fv)| fv.as_array()).collect();

        // Standardization statistics over the whole training set.
        let n = raw.len() as f32;
        let mut mean = vec![0.0f32; FEATURE_DIM];
        for r in &raw {
            for (m, &x) in mean.iter_mut().zip(r) {
                *m += x;
            }
        }
        for m in &mut mean {
            *m /= n;
        }
        let mut std = vec![0.0f32; FEATURE_DIM];
        for r in &raw {
            for (s, (&x, &m)) in std.iter_mut().zip(r.iter().zip(&mean)) {
                let d = x - m;
                *s += d * d;
            }
        }
        for s in &mut std {
            *s = (*s / n).sqrt().max(STD_FLOOR);
        }

        let standardize = |r: &[f32; FEATURE_DIM]| -> Vec<f32> {
            r.iter()
                .zip(mean.iter().zip(&std))
                .map(|(&x, (&m, &sd))| (x - m) / sd)
                .collect()
        };

        // Standardized examples + per-zone centroids.
        let examples: Vec<StdExample> = samples
            .iter()
            .zip(&raw)
            .map(|((zid, _), r)| StdExample {
                zone: zone_index(zid),
                v: standardize(r),
            })
            .collect();

        let mut centroids = vec![vec![0.0f32; FEATURE_DIM]; zone_ids.len()];
        let mut counts = vec![0u32; zone_ids.len()];
        for ex in &examples {
            counts[ex.zone] += 1;
            for (c, &x) in centroids[ex.zone].iter_mut().zip(&ex.v) {
                *c += x;
            }
        }
        for (c, &cnt) in centroids.iter_mut().zip(&counts) {
            if cnt > 0 {
                for x in c.iter_mut() {
                    *x /= cnt as f32;
                }
            }
        }

        // Novelty threshold from the positive examples' distances to their own
        // centroid (intra-zone spread).
        let mut spreads: Vec<f32> = examples
            .iter()
            .map(|ex| euclidean(&ex.v, &centroids[ex.zone]))
            .collect();
        let (spread_mean, spread_std) = mean_std(&spreads);
        spreads.sort_by(|a, b| a.total_cmp(b));
        let spread_max = spreads.last().copied().unwrap_or(0.0);
        let mut novelty_threshold = (spread_mean + NOVELTY_SIGMA * spread_std).max(spread_max);
        if novelty_threshold <= 0.0 {
            novelty_threshold = STD_FLOOR;
        }

        // If negatives are supplied, pull the threshold inward so they get
        // rejected — but never below the real taps' own spread.
        let negative_examples_trained = !negatives.is_empty();
        if negative_examples_trained {
            let mut neg_min = f32::INFINITY;
            for fv in negatives {
                let z = standardize(&fv.as_array());
                let d = examples
                    .iter()
                    .map(|ex| euclidean(&z, &ex.v))
                    .fold(f32::INFINITY, f32::min);
                neg_min = neg_min.min(d);
            }
            if neg_min.is_finite() {
                // Halfway between the taps' spread and the nearest negative,
                // clamped so real taps still pass.
                let target = (spread_max + neg_min) * 0.5;
                novelty_threshold = novelty_threshold.min(target).max(spread_max);
            }
        }

        let params = Self {
            schema_version: CLASSIFIER_SCHEMA_VERSION,
            feature_mean: mean,
            feature_std: std,
            zone_ids,
            centroids,
            examples,
            novelty_threshold,
            negative_examples_trained,
        };
        params.validate()?;
        Ok(params)
    }

    /// Check that these parameters are internally consistent and finite. Run at
    /// the end of `train()` and again after a profile is loaded, so a stale or
    /// hand-edited blob fails loudly instead of silently truncating via `.zip()`
    /// and misclassifying (`CLAUDE.md`: fail fast, no silent fallbacks).
    pub fn validate(&self) -> Result<(), ClassifierError> {
        let dim_ok = |v: &[f32]| v.len() == FEATURE_DIM;
        if !dim_ok(&self.feature_mean) || !dim_ok(&self.feature_std) {
            return Err(ClassifierError::MalformedParameters(format!(
                "standardization vectors must be length {FEATURE_DIM}"
            )));
        }
        if self.centroids.len() != self.zone_ids.len() {
            return Err(ClassifierError::MalformedParameters(
                "centroid count does not match zone count".into(),
            ));
        }
        if self.centroids.iter().any(|c| !dim_ok(c)) {
            return Err(ClassifierError::MalformedParameters(format!(
                "each centroid must be length {FEATURE_DIM}"
            )));
        }
        if self
            .examples
            .iter()
            .any(|ex| !dim_ok(&ex.v) || ex.zone >= self.zone_ids.len())
        {
            return Err(ClassifierError::MalformedParameters(
                "an example has a bad dimension or zone index".into(),
            ));
        }
        let all_finite = self
            .feature_mean
            .iter()
            .chain(&self.feature_std)
            .chain(self.centroids.iter().flatten())
            .chain(self.examples.iter().flat_map(|e| &e.v))
            .chain(std::iter::once(&self.novelty_threshold))
            .all(|v| v.is_finite());
        if !all_finite {
            return Err(ClassifierError::NonFiniteParameters);
        }
        if self.feature_std.iter().any(|s| *s <= 0.0) {
            return Err(ClassifierError::MalformedParameters(
                "feature_std must be strictly positive".into(),
            ));
        }
        Ok(())
    }

    /// Classify one feature vector: assign the nearest-centroid zone, or reject
    /// via the nearest-example novelty gate.
    pub fn classify(&self, fv: &FeatureVector) -> Classification {
        // Garbage input (NaN/inf from a flaky driver) is rejected rather than
        // fed into the distance math, where a NaN compare would otherwise panic.
        if !fv.is_finite() {
            return Classification {
                zone_id: None,
                confidence: 0.0,
                accepted: false,
            };
        }
        let raw = fv.as_array();
        let z: Vec<f32> = raw
            .iter()
            .zip(self.feature_mean.iter().zip(&self.feature_std))
            .map(|(&x, (&m, &sd))| (x - m) / sd)
            .collect();

        // Nearest-example distance drives the novelty gate.
        let nearest_example = self
            .examples
            .iter()
            .map(|ex| euclidean(&z, &ex.v))
            .fold(f32::INFINITY, f32::min);
        let accepted = nearest_example <= self.novelty_threshold;

        // Nearest centroid (the linear model) picks the zone + confidence.
        let mut dists: Vec<f32> = self.centroids.iter().map(|c| euclidean(&z, c)).collect();
        let best = dists
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i);

        let (zone_id, confidence) = match best {
            Some(i) if accepted => {
                // Softmax over negative squared distances -> confidence.
                let min_d2 = dists.iter().map(|d| d * d).fold(f32::INFINITY, f32::min);
                let mut denom = 0.0f32;
                for d in &mut dists {
                    *d = (-CONFIDENCE_BETA * (*d * *d - min_d2)).exp();
                    denom += *d;
                }
                let conf = if denom > 0.0 { dists[i] / denom } else { 0.0 };
                (Some(self.zone_ids[i]), conf)
            }
            _ => (None, 0.0),
        };

        Classification {
            zone_id,
            confidence,
            accepted,
        }
    }
}

fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

fn mean_std(xs: &[f32]) -> (f32, f32) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let n = xs.len() as f32;
    let mean = xs.iter().sum::<f32>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{CEPSTRAL_COEFF_COUNT, SPECTRAL_BAND_COUNT, TEMPORAL_FEATURE_COUNT};

    /// Build a deterministic feature vector from a small seed, so tests can
    /// synthesize distinct, tight per-zone clusters without real audio.
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

    fn zone(n: u128) -> ZoneId {
        ZoneId::from_uuid(uuid::Uuid::from_u128(n))
    }

    #[test]
    fn empty_training_set_is_an_error() {
        assert_eq!(
            ClassifierParams::train(&[], &[]),
            Err(ClassifierError::EmptyTrainingSet)
        );
    }

    #[test]
    fn assigns_correct_zone_on_clean_clusters() {
        let (za, zb, zc) = (zone(1), zone(2), zone(3));
        let mut samples = Vec::new();
        for j in 0..10 {
            let jit = j as f32 * 1e-4;
            samples.push((za, fv_from(0.0, jit)));
            samples.push((zb, fv_from(5.0, jit)));
            samples.push((zc, fv_from(10.0, jit)));
        }
        let clf = ClassifierParams::train(&samples, &[]).unwrap();

        // Query near each cluster center classifies to that zone.
        for (seed, expected) in [(0.0, za), (5.0, zb), (10.0, zc)] {
            let out = clf.classify(&fv_from(seed, 1e-3));
            assert!(out.accepted, "clean tap should be accepted");
            assert_eq!(out.zone_id, Some(expected));
            assert!(out.confidence > 0.5);
        }
    }

    #[test]
    fn novelty_gate_rejects_far_out_input() {
        let (za, zb) = (zone(1), zone(2));
        let mut samples = Vec::new();
        for j in 0..10 {
            let jit = j as f32 * 1e-4;
            samples.push((za, fv_from(0.0, jit)));
            samples.push((zb, fv_from(4.0, jit)));
        }
        let clf = ClassifierParams::train(&samples, &[]).unwrap();
        // A vector wildly outside both clusters must be rejected.
        let out = clf.classify(&fv_from(1000.0, 0.0));
        assert!(!out.accepted);
        assert_eq!(out.zone_id, None);
    }

    #[test]
    fn negatives_tighten_the_novelty_threshold() {
        let (za, zb) = (zone(1), zone(2));
        let mut samples = Vec::new();
        for j in 0..10 {
            let jit = j as f32 * 1e-4;
            samples.push((za, fv_from(0.0, jit)));
            samples.push((zb, fv_from(4.0, jit)));
        }
        let without = ClassifierParams::train(&samples, &[]).unwrap();
        let negatives = vec![fv_from(20.0, 0.0), fv_from(25.0, 0.0)];
        let with = ClassifierParams::train(&samples, &negatives).unwrap();
        assert!(with.negative_examples_trained());
        assert!(!without.negative_examples_trained());
        assert!(with.novelty_threshold <= without.novelty_threshold);
    }
}
