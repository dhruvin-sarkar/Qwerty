//! Model training from calibration descriptors (§1.10).

use crate::descriptor::{Descriptor, DescriptorLayout};
use crate::linalg::{
    self, ledoit_wolf_diagonal, ledoit_wolf_identity, quantile_nearest_rank, shrink_to_diagonal,
    shrink_to_identity, Cholesky, CholeskyError, SymMatrix,
};
use crate::model::{
    log_likelihood_ratio, rank_zones, zone_index, Background, BackgroundRegime, GaussianFactor,
    MapPoint, ModelRejectReason, SoundMap, Standardization, Thresholds, TrainedModel,
    MODEL_SCHEMA_VERSION,
};
use crate::profile::ZoneId;
use crate::units::{SampleRate, TapIndex, ZoneIndex};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU8;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

pub const MIN_ZONES: usize = 2;
pub const MIN_TAPS_PER_ZONE: usize = 5;
pub const MAX_WORKERS: usize = 8;

// SPEC: `TrainedModel.front_end_version` is `detector::FRONT_END_VERSION` (§1.6.1), which WP-A1
// creates in parallel with this package; until both merge, training records the same value (1)
// from here. Replace with `crate::detector::FRONT_END_VERSION` at the merge.
const FRONT_END_VERSION_AT_TRAINING: u32 = 1;

/// A dim is excluded when its clean spread is at most this fraction of (1 + |mean|) (§1.10.3).
const EXCLUDE_RELATIVE_SPREAD: f64 = 1e-9;
/// The within-zone sd is floored at this fraction of the dim's total sd (§1.10.3, DSP-ML-7).
const WITHIN_SD_FLOOR_FRACTION: f64 = 0.05;
/// The Negatives regime also needs two clips: each background fold leaves one clip out (§1.10.4).
// SPEC: binds only when `bg_min_clips` < 2 (the default is 24); with one clip the clip fold's
// (C − 1) divisor is 0, so such a set stays PooledAll.
const MIN_BACKGROUND_FOLD_CLIPS: usize = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Workers {
    /// min(MAX_WORKERS, available_parallelism)
    Auto,
    /// 0 workers is unrepresentable (X01-10)
    Exactly { n: NonZeroU8 },
}

impl Workers {
    fn count(&self) -> usize {
        match self {
            Workers::Exactly { n } => usize::from(n.get()),
            Workers::Auto => {
                // The count only sets the speed: every count gives bit-identical results
                // (§1.10.4), so a platform that cannot report its parallelism gets one worker.
                let available = match std::thread::available_parallelism() {
                    Ok(n) => n.get(),
                    Err(_) => 1,
                };
                available.min(MAX_WORKERS)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingParams {
    pub alpha_in: f64,             // 0.02
    pub tau_margin: f64,           // 4.0
    pub bg_positive_quantile: f64, // 0.01
    pub bg_negative_quantile: f64, // 0.95
    /// 24: the Negatives regime needs ≥ 24 negative clips (≥ d/2 for both layouts: 19, 24).
    /// Counted in clips, not vectors: a clip's variants are strongly correlated (§1.8.3),
    /// so 14 clips × 8 variants are 14 observations, not 112 (DSP-ML-17a)
    pub bg_min_clips: u32,
    pub workers: Workers, // Auto
}

impl TrainingParams {
    pub const DEFAULT: TrainingParams = TrainingParams {
        alpha_in: 0.02,
        tau_margin: 4.0,
        bg_positive_quantile: 0.01,
        bg_negative_quantile: 0.95,
        bg_min_clips: 24,
        workers: Workers::Auto,
    };
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingSet {
    pub layout: DescriptorLayout,
    pub rate: SampleRate,
    pub zone_ids: Vec<ZoneId>,
    /// the plan seed the variants were built with; training only records it (TrainingSummary.seed)
    pub seed: u64,
    pub taps: Vec<TapGroup>,
    pub negatives: Vec<NegativeGroup>,
}

/// variants[0] = live (clean)
#[derive(Debug, Clone, PartialEq)]
pub struct TapGroup {
    pub zone: ZoneIndex,
    pub tap: TapIndex,
    pub variants: Vec<Descriptor>,
}

/// variants[0] = live (clean), as for taps
#[derive(Debug, Clone, PartialEq)]
pub struct NegativeGroup {
    pub source: NegativeSource,
    pub variants: Vec<Descriptor>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum NegativeSource {
    IgnoreStep,
    EnvironmentOnset,
    OsVetoed,
}

impl NegativeSource {
    const ALL: [NegativeSource; 3] = [
        NegativeSource::IgnoreStep,
        NegativeSource::EnvironmentOnset,
        NegativeSource::OsVetoed,
    ];
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TrainingProgress {
    Standardize,
    FitWithin,
    Background,
    Folds { done: u32, total: u32 },
    BackgroundFolds { done: u32, total: u32 },
    Thresholds,
    Done,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TrainingError {
    // SPEC: §1.10.1 names no error for a `TrainingParams` field out of range; `train` checks the
    // params before the set and reports the first bad field (see `validate_params`).
    InvalidParam {
        param: TrainingParam,
    },
    NotEnoughZones {
        have: u32,
    },
    NotEnoughTaps {
        zone: ZoneIndex,
        have: u32,
        need: u32,
    },
    EmptyTap {
        tap: TapIndex,
    },
    LayoutMismatch {
        tap: TapIndex,
    },
    NonFiniteDescriptor {
        tap: TapIndex,
        index: u16,
    },
    /// excluded + floored > d/2 (§1.10.3)
    DegenerateWithinZoneSpread {
        degenerate_dims: u16,
        dims: u16,
    },
    CovarianceNotPositiveDefinite {
        stage: FactorStage,
    },
    Cancelled,
}

/// The `TrainingParams` field an `InvalidParam` names. The value is not carried: it may be
/// non-finite, which a persisted float must never be (§0.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TrainingParam {
    AlphaIn,
    TauMargin,
    BgPositiveQuantile,
    BgNegativeQuantile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum FactorStage {
    Within,
    WithinFold { tap: TapIndex },
    Background,
    BackgroundFold { clip: u32 },
}

/// Held-out CONSISTENCY of the calibration taps (leave-one-tap-out). It is not an accuracy measurement:
/// only an `EvaluationReport` is (ADR-006), so the Review step never labels these figures "accuracy" (PARITY-12).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingSummary {
    pub seed: u64, // the plan seed the augmentation used; replay reuses it (RUST-RT-13)
    pub zones: Vec<ZoneTrainingStats>,
    pub negatives: Vec<NegativeCount>, // one entry per NegativeSource present
    pub negative_vectors: u32,
    pub variants_dropped: u32,
    pub within_lambda: f64,
    pub background_lambda: f64,
    pub background_regime: BackgroundRegime,
    pub thresholds: Thresholds,
    pub held_out_clean: OutcomeCounts,  // v0 of every tap
    pub held_out_all: OutcomeCounts,    // every variant
    pub confusion_clean: Vec<Vec<u32>>, // K × (K+1): accepted as zone j | rejected (last column)
    pub negatives_held_out: u32,
    pub negatives_held_out_accepted: u32, // passed all three gates
    pub quality: ModelQuality,
}

impl TrainingSummary {
    /// Every float the summary carries is finite (checked by `TrainedModel::validate`).
    pub(crate) fn is_finite(&self) -> bool {
        let t = &self.thresholds;
        self.within_lambda.is_finite()
            && self.background_lambda.is_finite()
            && [t.alpha_in, t.tau_in, t.tau_bg, t.tau_margin]
                .iter()
                .all(|v| v.is_finite())
            && self
                .zones
                .iter()
                .all(|z| z.clean_accuracy.is_finite() && z.clean_rejected.is_finite())
            && self.quality.issues().iter().all(QualityIssue::is_finite)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZoneTrainingStats {
    pub zone: ZoneId,
    pub taps: u32,
    pub vectors: u32,
    pub clean_accuracy: f64,
    pub clean_rejected: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NegativeCount {
    pub source: NegativeSource,
    pub clips: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct OutcomeCounts {
    pub total: u32,
    pub correct: u32,
    pub wrong: u32,
    pub rejected_ood: u32,
    pub rejected_bg: u32,
    pub rejected_ambiguous: u32,
}

impl OutcomeCounts {
    fn rejected(&self) -> u32 {
        self.rejected_ood + self.rejected_bg + self.rejected_ambiguous
    }

    fn add(&mut self, outcome: Outcome) {
        self.total += 1;
        match outcome {
            Outcome::Accepted { correct: true, .. } => self.correct += 1,
            Outcome::Accepted { correct: false, .. } => self.wrong += 1,
            Outcome::Rejected(ModelRejectReason::OutOfDistribution) => self.rejected_ood += 1,
            Outcome::Rejected(ModelRejectReason::Background) => self.rejected_bg += 1,
            Outcome::Rejected(ModelRejectReason::Ambiguous) => self.rejected_ambiguous += 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ModelQuality {
    Good,
    Marginal { issues: Vec<QualityIssue> },
    Poor { issues: Vec<QualityIssue> },
}

impl ModelQuality {
    fn issues(&self) -> &[QualityIssue] {
        match self {
            ModelQuality::Good => &[],
            ModelQuality::Marginal { issues } | ModelQuality::Poor { issues } => issues,
        }
    }
}

/// The quality rules' limits, public so UI copy and tests bind to them instead of repeating numbers (PARITY-34).
pub const QUALITY_MIN_ZONE_CONSISTENCY: f64 = 0.80; // v1 CONSISTENCY_MIN_ACCURACY
pub const QUALITY_MAX_CONFUSION: f64 = 0.10;
pub const QUALITY_MAX_BACKGROUND_REJECTION: f64 = 0.10;
pub const QUALITY_MAX_REJECTION: f64 = 0.15;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum QualityIssue {
    InconsistentZone {
        zone: ZoneId,
        clean_accuracy: f64,
    }, // < QUALITY_MIN_ZONE_CONSISTENCY
    ZonesConfusable {
        a: ZoneId,
        b: ZoneId,
        rate: f64,
    }, // clean taps of a accepted as b, or of b as a, > QUALITY_MAX_CONFUSION
    BackgroundOverlap {
        positive_rejection: f64,
    }, // clean rejected by τ_bg > QUALITY_MAX_BACKGROUND_REJECTION
    HighRejection {
        rate: f64,
    }, // clean rejected by any gate > QUALITY_MAX_REJECTION
    FewNegatives, // regime PooledAll
    DegenerateDims {
        excluded: Vec<u16>,
        floored: Vec<u16>,
    }, // §1.10.3 step 3
}

impl QualityIssue {
    fn is_finite(&self) -> bool {
        match self {
            QualityIssue::InconsistentZone { clean_accuracy, .. } => clean_accuracy.is_finite(),
            QualityIssue::ZonesConfusable { rate, .. } | QualityIssue::HighRejection { rate } => {
                rate.is_finite()
            }
            QualityIssue::BackgroundOverlap { positive_rejection } => {
                positive_rejection.is_finite()
            }
            QualityIssue::FewNegatives | QualityIssue::DegenerateDims { .. } => true,
        }
    }
}

/// `cancel` is checked after every fold chunk; once set, `train` returns `Err(Cancelled)` (RUST-RT-30).
pub fn train(
    set: &TrainingSet,
    params: &TrainingParams,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(TrainingProgress),
) -> Result<TrainedModel, TrainingError> {
    validate_params(params)?;
    validate_set(set)?;
    let d = set.layout.dims();
    let k = set.zone_ids.len();
    let g = set.taps.len();
    let zone_of: Vec<usize> = set.taps.iter().map(|t| usize::from(t.zone.0)).collect();

    // §1.10.3 steps 1-3.
    progress(TrainingProgress::Standardize);
    let standardization = fit_standardization(set, &zone_of)?;
    let standardize = |desc: &Descriptor| {
        let mut z = vec![0.0; d];
        standardization.apply(desc.as_slice(), &mut z);
        z
    };
    let tap_z: Vec<Vec<Vec<f64>>> = set
        .taps
        .iter()
        .map(|t| t.variants.iter().map(standardize).collect())
        .collect();
    // Negative clips with ≥ 1 surviving variant, as (position in `set.negatives`, z).
    let clip_z: Vec<(usize, Vec<Vec<f64>>)> = set
        .negatives
        .iter()
        .enumerate()
        .filter(|(_, n)| !n.variants.is_empty())
        .map(|(c, n)| (c, n.variants.iter().map(standardize).collect()))
        .collect();

    // Step 4: the within model.
    progress(TrainingProgress::FitWithin);
    let tap_moments: Vec<GroupMoments> = tap_z.iter().map(|v| GroupMoments::of(v, d)).collect();
    let clean: Vec<&[f64]> = tap_z.iter().map(|v| v[0].as_slice()).collect();
    let within_sums = WithinSums::new(&tap_moments, &clean, &zone_of, k, d);
    let within_groups: Vec<SymMatrix> = tap_z
        .iter()
        .zip(&zone_of)
        .map(|(v, &zk)| deviation_scatter(v, &within_sums.means[zk], d))
        .collect();
    let s_within = mean_of(&within_groups, d);
    let within_shrinkage = ledoit_wolf_identity(&within_groups, &s_within);
    let within_factor = factor(
        &shrink_to_identity(&s_within, within_shrinkage.lambda),
        FactorStage::Within,
    )?;

    // Step 5: the background.
    progress(TrainingProgress::Background);
    let clips = clip_z.len();
    let negative_vectors: usize = clip_z.iter().map(|(_, v)| v.len()).sum();
    let negatives_regime =
        clips >= params.bg_min_clips as usize && clips >= MIN_BACKGROUND_FOLD_CLIPS;
    let regime = if negatives_regime {
        BackgroundRegime::Negatives {
            clips: count_u32(clips),
            vectors: count_u32(negative_vectors),
        }
    } else {
        BackgroundRegime::PooledAll {
            negative_clips: count_u32(clips),
            needed: params.bg_min_clips,
        }
    };
    let clip_moments: Vec<GroupMoments> = if negatives_regime {
        clip_z.iter().map(|(_, v)| GroupMoments::of(v, d)).collect()
    } else {
        Vec::new()
    };
    let bg_moments = if negatives_regime {
        &clip_moments
    } else {
        &tap_moments
    };
    let bg_sums = PoolSums::new(bg_moments, d);
    let bg_groups: Vec<SymMatrix> = if negatives_regime {
        clip_z
            .iter()
            .map(|(_, v)| deviation_scatter(v, &bg_sums.mean, d))
            .collect()
    } else {
        tap_z
            .iter()
            .map(|v| deviation_scatter(v, &bg_sums.mean, d))
            .collect()
    };
    let s_bg = mean_of(&bg_groups, d);
    let bg_lambda = ledoit_wolf_diagonal(&bg_groups, &s_bg).lambda;
    let bg_factor = factor(
        &shrink_to_diagonal(&s_bg, bg_lambda),
        FactorStage::Background,
    )?;

    // Step 6: the sound map (display only).
    let clean_by_zone: Vec<(usize, &[f64])> =
        zone_of.iter().copied().zip(clean.iter().copied()).collect();
    let axes = map_axes(&within_factor, &within_sums.means, &clean_by_zone);
    let map = SoundMap {
        taps: set
            .taps
            .iter()
            .zip(&clean)
            .map(|(t, z)| MapPoint {
                zone: t.zone,
                at: SoundMap::project(&axes, z),
            })
            .collect(),
        background: clip_z
            .iter()
            .map(|(_, v)| SoundMap::project(&axes, &v[0]))
            .collect(),
        axes,
    };

    // §1.10.4: leave-one-tap-out folds.
    let workers = params.workers.count();
    let full_bg = FoldBackground {
        factor: &bg_factor,
        mean: &bg_sums.mean,
    };
    let tap_fold = |h: usize| -> Result<Vec<HeldOut>, TrainingError> {
        let zk = zone_of[h];
        let tap = set.taps[h].tap;
        let (m_prime, s_prime) = within_sums.fold(&tap_moments[h], clean[h], zk);
        let fold_within = factor(
            &shrink_to_identity(&s_prime, within_shrinkage.lambda),
            FactorStage::WithinFold { tap },
        )?;
        let pooled_fold;
        let background = if negatives_regime {
            full_bg
        } else {
            let (mean, s) = bg_sums.fold(&tap_moments[h]);
            // SPEC: §1.10.1 has no stage for a PooledAll background refit, which is a background
            // fold whose removed group is tap h; it is reported as BackgroundFold{clip: h's TapIndex}.
            let f = factor(
                &shrink_to_diagonal(&s, bg_lambda),
                FactorStage::BackgroundFold { clip: tap.0 },
            )?;
            pooled_fold = (f, mean);
            FoldBackground {
                factor: &pooled_fold.0,
                mean: &pooled_fold.1,
            }
        };
        let means = FoldMeans {
            all: &within_sums.means,
            replaced: Some((zk, &m_prime)),
        };
        Ok(score_all(&tap_z[h], &fold_within, &means, background, d))
    };
    let tap_scores = run_folds(
        g,
        workers,
        cancel,
        progress,
        |done, total| TrainingProgress::Folds { done, total },
        &tap_fold,
    )?;

    // Held-out negatives: background folds in the Negatives regime. In PooledAll no fit saw
    // them, so they are scored against the full model.
    let full_means = FoldMeans {
        all: &within_sums.means,
        replaced: None,
    };
    let negative_scores: Vec<Vec<HeldOut>> = if negatives_regime {
        let clip_fold = |c: usize| -> Result<Vec<HeldOut>, TrainingError> {
            let (mean, s) = bg_sums.fold(&clip_moments[c]);
            let f = factor(
                &shrink_to_diagonal(&s, bg_lambda),
                FactorStage::BackgroundFold {
                    clip: count_u32(clip_z[c].0),
                },
            )?;
            let background = FoldBackground {
                factor: &f,
                mean: &mean,
            };
            Ok(score_all(
                &clip_z[c].1,
                &within_factor,
                &full_means,
                background,
                d,
            ))
        };
        run_folds(
            clips,
            workers,
            cancel,
            progress,
            |done, total| TrainingProgress::BackgroundFolds { done, total },
            &clip_fold,
        )?
    } else {
        clip_z
            .iter()
            .map(|(_, v)| score_all(v, &within_factor, &full_means, full_bg, d))
            .collect()
    };

    // §1.10.5.
    progress(TrainingProgress::Thresholds);
    let all_positive: Vec<HeldOut> = tap_scores.iter().flatten().copied().collect();
    let negative_llr: Vec<f64> = negative_scores.iter().flatten().map(|s| s.llr).collect();
    let thresholds = thresholds_from(
        params,
        &all_positive,
        if negatives_regime {
            Some(&negative_llr)
        } else {
            None
        },
    );

    let summary = summarize(SummaryInputs {
        set,
        zone_of: &zone_of,
        tap_scores: &tap_scores,
        negative_scores: &negative_scores,
        clip_sources: clip_z
            .iter()
            .map(|(c, _)| set.negatives[*c].source)
            .collect(),
        negative_vectors,
        thresholds,
        within_lambda: within_shrinkage.lambda,
        background_lambda: bg_lambda,
        regime: regime.clone(),
        standardization: &standardization,
    });

    let model = TrainedModel {
        schema_version: MODEL_SCHEMA_VERSION,
        front_end_version: FRONT_END_VERSION_AT_TRAINING,
        layout: set.layout,
        rate: set.rate,
        zone_ids: set.zone_ids.clone(),
        standardization,
        zone_means: within_sums.means,
        within: GaussianFactor {
            factor: within_factor,
            shrinkage: within_shrinkage,
        },
        background: Background {
            regime,
            mean: bg_sums.mean,
            factor: bg_factor,
            lambda: bg_lambda,
        },
        thresholds,
        summary,
        map,
    };
    progress(TrainingProgress::Done);
    Ok(model)
}

/// usize → u32 for a count of taps, clips, variants or folds of one in-memory `TrainingSet`.
fn count_u32(n: usize) -> u32 {
    // INVARIANT: each counted item is a `Descriptor` (196 bytes) or a group of them held in
    // memory, so 2^32 of them would need > 800 GB.
    u32::try_from(n).expect("training counts fit in u32")
}

/// A descriptor dim index or dim count as u16.
fn dim_u16(i: usize) -> u16 {
    // INVARIANT: every index and count passed here is ≤ DESCRIPTOR_DIMS (48).
    u16::try_from(i).expect("descriptor dims fit in u16")
}

/// The ranges `train` needs, first bad field first: the quantile levels 1 − α_in, q_bg+ and q_bg−
/// lie in (0, 1] (nearest rank, §1.10.5), and τ_m is finite because the model stores it (§0.5).
// SPEC: §1.10.2 checks only the set. Unchecked, a level outside (0, 1] tripped
// `quantile_nearest_rank`'s debug assertion and was clamped to an end of the list in release,
// and a non-finite τ_m gave a model that its own `validate` rejects.
fn validate_params(params: &TrainingParams) -> Result<(), TrainingError> {
    let level = |p: f64| p > 0.0 && p <= 1.0;
    let in_range = [
        (
            TrainingParam::AlphaIn,
            (0.0..1.0).contains(&params.alpha_in),
        ),
        (TrainingParam::TauMargin, params.tau_margin.is_finite()),
        (
            TrainingParam::BgPositiveQuantile,
            level(params.bg_positive_quantile),
        ),
        (
            TrainingParam::BgNegativeQuantile,
            level(params.bg_negative_quantile),
        ),
    ];
    match in_range.iter().find(|&&(_, ok)| !ok) {
        Some(&(param, _)) => Err(TrainingError::InvalidParam { param }),
        None => Ok(()),
    }
}

/// §1.10.2, in order, fail fast.
fn validate_set(set: &TrainingSet) -> Result<(), TrainingError> {
    let k = set.zone_ids.len();
    if k < MIN_ZONES {
        return Err(TrainingError::NotEnoughZones { have: count_u32(k) });
    }
    if set.taps.iter().any(|t| usize::from(t.zone.0) >= k) {
        // SPEC: §1.10.2 names no error for a tap whose zone index is ≥ K. The set then holds
        // taps of a zone that `zone_ids` lacks, so it has too few zone ids: NotEnoughZones.
        return Err(TrainingError::NotEnoughZones { have: count_u32(k) });
    }
    let mut per_zone = vec![0usize; k];
    for t in &set.taps {
        per_zone[usize::from(t.zone.0)] += 1;
    }
    if let Some((zone, &have)) = per_zone
        .iter()
        .enumerate()
        .find(|&(_, &n)| n < MIN_TAPS_PER_ZONE)
    {
        return Err(TrainingError::NotEnoughTaps {
            zone: zone_index(zone),
            have: count_u32(have),
            need: count_u32(MIN_TAPS_PER_ZONE),
        });
    }
    if let Some(t) = set.taps.iter().find(|t| t.variants.is_empty()) {
        return Err(TrainingError::EmptyTap { tap: t.tap });
    }
    // SPEC: §1.10.2's layout and finiteness rules cover the negative clips too, but its errors
    // carry a `TapIndex`; a negative clip is reported by its position in `set.negatives`.
    let groups = || {
        set.taps.iter().map(|t| (t.tap, &t.variants)).chain(
            set.negatives
                .iter()
                .enumerate()
                .map(|(c, n)| (TapIndex(count_u32(c)), &n.variants)),
        )
    };
    for (tap, variants) in groups() {
        if variants.iter().any(|v| v.layout != set.layout) {
            return Err(TrainingError::LayoutMismatch { tap });
        }
    }
    for (tap, variants) in groups() {
        for v in variants {
            if let Some(index) = v.as_slice().iter().position(|x| !x.is_finite()) {
                return Err(TrainingError::NonFiniteDescriptor {
                    tap,
                    index: dim_u16(index),
                });
            }
        }
    }
    Ok(())
}

/// §1.10.3 steps 1-3, on the raw (f64) descriptors.
fn fit_standardization(
    set: &TrainingSet,
    zone_of: &[usize],
) -> Result<Standardization, TrainingError> {
    let d = set.layout.dims();
    let k = set.zone_ids.len();
    let g = set.taps.len() as f64;
    let raw = |desc: &Descriptor, i: usize| f64::from(desc.as_slice()[i]);

    // Step 1: raw zone means from the clean variants.
    let mut zone_raw = vec![vec![0.0; d]; k];
    let mut zone_n = vec![0usize; k];
    for (t, &zk) in set.taps.iter().zip(zone_of) {
        zone_n[zk] += 1;
        for (i, m) in zone_raw[zk].iter_mut().enumerate() {
            *m += raw(&t.variants[0], i);
        }
    }
    for (m, &n) in zone_raw.iter_mut().zip(&zone_n) {
        m.iter_mut().for_each(|v| *v /= n as f64);
    }

    // Step 2: grouped within-class variance and the total clean spread.
    let mut within_var = vec![0.0; d];
    for (t, &zk) in set.taps.iter().zip(zone_of) {
        let v_t = t.variants.len() as f64;
        for variant in &t.variants {
            for (i, w) in within_var.iter_mut().enumerate() {
                let dev = raw(variant, i) - zone_raw[zk][i];
                *w += dev * dev / v_t;
            }
        }
    }
    within_var.iter_mut().for_each(|w| *w /= g);
    let mut grand = vec![0.0; d];
    for t in &set.taps {
        for (i, m) in grand.iter_mut().enumerate() {
            *m += raw(&t.variants[0], i);
        }
    }
    grand.iter_mut().for_each(|m| *m /= g);
    let mut total_var = vec![0.0; d];
    for t in &set.taps {
        for (i, v) in total_var.iter_mut().enumerate() {
            let dev = raw(&t.variants[0], i) - grand[i];
            *v += dev * dev;
        }
    }
    total_var.iter_mut().for_each(|v| *v /= g);

    // Step 3: exclusion, floors, offset.
    let mut scale = vec![0.0; d];
    let mut excluded = Vec::new();
    let mut floored = Vec::new();
    for i in 0..d {
        let sd_tot = total_var[i].sqrt();
        let sd_within = within_var[i].sqrt();
        let dim = dim_u16(i);
        if sd_tot <= EXCLUDE_RELATIVE_SPREAD * (1.0 + grand[i].abs()) {
            excluded.push(dim);
            // SPEC: §1.10.3 sets no scale for an excluded dim (z_g ≡ 0); 1.0 keeps it finite.
            scale[i] = 1.0;
        } else {
            let floor = WITHIN_SD_FLOOR_FRACTION * sd_tot;
            if sd_within < floor {
                floored.push(dim);
            }
            scale[i] = sd_within.max(floor);
        }
    }
    let degenerate = excluded.len() + floored.len();
    if 2 * degenerate > d {
        return Err(TrainingError::DegenerateWithinZoneSpread {
            degenerate_dims: dim_u16(degenerate),
            dims: dim_u16(d),
        });
    }
    let mut offset = vec![0.0; d];
    for m in &zone_raw {
        for (o, v) in offset.iter_mut().zip(m) {
            *o += v / k as f64;
        }
    }
    Ok(Standardization {
        offset,
        scale,
        floored_dims: floored,
        excluded_dims: excluded,
    })
}

fn factor(a: &SymMatrix, stage: FactorStage) -> Result<Cholesky, TrainingError> {
    Cholesky::factor(a).map_err(|e| match e {
        CholeskyError::NonFinite | CholeskyError::NotPositiveDefinite { .. } => {
            TrainingError::CovarianceNotPositiveDefinite { stage }
        }
    })
}

/// (1/G)·Σ groups.
fn mean_of(groups: &[SymMatrix], d: usize) -> SymMatrix {
    let mut s = SymMatrix::zeros(d);
    let w = 1.0 / groups.len() as f64;
    for y in groups {
        s.add_scaled(y, w);
    }
    s
}

/// (1/V)·Σ_v (z_v − m)(z_v − m)ᵀ.
fn deviation_scatter(variants: &[Vec<f64>], mean: &[f64], d: usize) -> SymMatrix {
    let mut y = SymMatrix::zeros(d);
    let w = 1.0 / variants.len() as f64;
    let mut dev = vec![0.0; d];
    for z in variants {
        for ((o, a), b) in dev.iter_mut().zip(z).zip(mean) {
            *o = a - b;
        }
        y.add_outer(&dev, w);
    }
    y
}

/// Z(A, b, n, m) = A − b·mᵀ − m·bᵀ + n·m·mᵀ (§1.10.4).
fn scatter_about(a: &SymMatrix, b: &[f64], n: f64, m: &[f64]) -> SymMatrix {
    let mut z = a.clone();
    z.add_cross(b, m, -1.0);
    z.add_outer(m, n);
    z
}

/// A_t = (1/V_t)·Σ_v z·zᵀ and b_t = (1/V_t)·Σ_v z of one group (a tap or a negative clip).
#[derive(Debug, Clone)]
struct GroupMoments {
    a: SymMatrix,
    b: Vec<f64>,
}

impl GroupMoments {
    fn of(variants: &[Vec<f64>], d: usize) -> Self {
        let w = 1.0 / variants.len() as f64;
        let mut a = SymMatrix::zeros(d);
        let mut b = vec![0.0; d];
        for z in variants {
            a.add_outer(z, w);
            for (bi, zi) in b.iter_mut().zip(z) {
                *bi += w * zi;
            }
        }
        GroupMoments { a, b }
    }
}

/// Per-zone sufficient statistics of the within model (§1.10.4).
#[derive(Debug, Clone)]
struct WithinSums {
    zone_a: Vec<SymMatrix>,
    zone_b: Vec<Vec<f64>>,
    zone_n: Vec<usize>,
    /// m_k: the mean of the clean z of zone k.
    means: Vec<Vec<f64>>,
    zone_z: Vec<SymMatrix>,
    /// T = Σ_k Z_k
    total: SymMatrix,
    g: usize,
}

impl WithinSums {
    fn new(
        moments: &[GroupMoments],
        clean: &[&[f64]],
        zone_of: &[usize],
        k: usize,
        d: usize,
    ) -> Self {
        let mut zone_a = vec![SymMatrix::zeros(d); k];
        let mut zone_b = vec![vec![0.0; d]; k];
        let mut zone_n = vec![0usize; k];
        let mut sums = vec![vec![0.0; d]; k];
        for ((m, x), &zk) in moments.iter().zip(clean).zip(zone_of) {
            zone_a[zk].add_scaled(&m.a, 1.0);
            add_into(&mut zone_b[zk], &m.b, 1.0);
            add_into(&mut sums[zk], x, 1.0);
            zone_n[zk] += 1;
        }
        let means: Vec<Vec<f64>> = sums
            .iter()
            .zip(&zone_n)
            .map(|(s, &n)| s.iter().map(|v| v / n as f64).collect())
            .collect();
        let zone_z: Vec<SymMatrix> = (0..k)
            .map(|zk| scatter_about(&zone_a[zk], &zone_b[zk], zone_n[zk] as f64, &means[zk]))
            .collect();
        let mut total = SymMatrix::zeros(d);
        for z in &zone_z {
            total.add_scaled(z, 1.0);
        }
        WithinSums {
            zone_a,
            zone_b,
            zone_n,
            means,
            zone_z,
            total,
            g: moments.len(),
        }
    }

    /// Fold h in zone k: (m'_k, S') without tap h.
    fn fold(&self, h: &GroupMoments, x_h: &[f64], zk: usize) -> (Vec<f64>, SymMatrix) {
        let n = self.zone_n[zk] as f64;
        let m_prime: Vec<f64> = self.means[zk]
            .iter()
            .zip(x_h)
            .map(|(m, x)| (n * m - x) / (n - 1.0))
            .collect();
        let mut a = self.zone_a[zk].clone();
        a.add_scaled(&h.a, -1.0);
        let mut b = self.zone_b[zk].clone();
        add_into(&mut b, &h.b, -1.0);
        let z_prime = scatter_about(&a, &b, n - 1.0, &m_prime);
        let mut s = self.total.clone();
        s.add_scaled(&self.zone_z[zk], -1.0);
        s.add_scaled(&z_prime, 1.0);
        s.scale(1.0 / (self.g as f64 - 1.0));
        (m_prime, s)
    }
}

/// Sufficient statistics of a background built from groups (negative clips, or every tap in
/// PooledAll), each removed by its variant mean in a fold (§1.10.4, DSP-ML-17c).
#[derive(Debug, Clone)]
struct PoolSums {
    a: SymMatrix,
    b: Vec<f64>,
    n: usize,
    /// (1/n)·Σ b_g
    mean: Vec<f64>,
}

impl PoolSums {
    fn new(groups: &[GroupMoments], d: usize) -> Self {
        let mut a = SymMatrix::zeros(d);
        let mut b = vec![0.0; d];
        for m in groups {
            a.add_scaled(&m.a, 1.0);
            add_into(&mut b, &m.b, 1.0);
        }
        let n = groups.len();
        let mean = b.iter().map(|v| v / n as f64).collect();
        PoolSums { a, b, n, mean }
    }

    /// (m', S') without group `h`: m' = (n·m − b_h)/(n − 1), S' = Z(A − A_h, b − b_h, n − 1, m')/(n − 1).
    fn fold(&self, h: &GroupMoments) -> (Vec<f64>, SymMatrix) {
        let n = self.n as f64;
        let m_prime: Vec<f64> = self
            .mean
            .iter()
            .zip(&h.b)
            .map(|(m, b)| (n * m - b) / (n - 1.0))
            .collect();
        let mut a = self.a.clone();
        a.add_scaled(&h.a, -1.0);
        let mut b = self.b.clone();
        add_into(&mut b, &h.b, -1.0);
        let mut s = scatter_about(&a, &b, n - 1.0, &m_prime);
        s.scale(1.0 / (n - 1.0));
        (m_prime, s)
    }
}

/// a += w·b
fn add_into(a: &mut [f64], b: &[f64], w: f64) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += w * y;
    }
}

/// The zone means of one fold: `all`, with one zone's mean swapped in a within fold.
struct FoldMeans<'a> {
    all: &'a [Vec<f64>],
    replaced: Option<(usize, &'a [f64])>,
}

impl FoldMeans<'_> {
    fn get(&self, k: usize) -> &[f64] {
        match self.replaced {
            Some((zk, mean)) if zk == k => mean,
            Some(_) | None => &self.all[k],
        }
    }
}

#[derive(Clone, Copy)]
struct FoldBackground<'a> {
    factor: &'a Cholesky,
    mean: &'a [f64],
}

/// One held-out vector's scores (§1.9.2 steps 2-5, with a fold's parameters).
#[derive(Debug, Clone, Copy, PartialEq)]
struct HeldOut {
    best: usize,
    d2_min: f64,
    margin: f64,
    llr: f64,
}

fn score_all(
    variants: &[Vec<f64>],
    within: &Cholesky,
    means: &FoldMeans<'_>,
    background: FoldBackground<'_>,
    d: usize,
) -> Vec<HeldOut> {
    let k = means.all.len();
    let mut solve = vec![0.0; d];
    let mut d2 = vec![0.0; k];
    variants
        .iter()
        .map(|z| {
            for (j, slot) in d2.iter_mut().enumerate() {
                *slot = within.mahalanobis_sq(z, means.get(j), &mut solve);
            }
            let d2_bg = background
                .factor
                .mahalanobis_sq(z, background.mean, &mut solve);
            let r = rank_zones(&d2);
            HeldOut {
                best: r.best,
                d2_min: r.d2_min,
                margin: r.margin,
                llr: log_likelihood_ratio(
                    r.d2_min,
                    within.log_det,
                    d2_bg,
                    background.factor.log_det,
                ),
            }
        })
        .collect()
}

/// One fold as a helper reports it: the fold index, and the fold's result or its panic payload.
type FoldReport<T> = (usize, std::thread::Result<Result<T, TrainingError>>);

/// Runs `fold(i)` for i in 0..total in chunks of `workers` inside `std::thread::scope`, slot
/// i holding fold i, so the output is bit-identical for any worker count. Progress is reported
/// and `cancel` checked by the coordinating thread after every chunk.
///
/// The coordinating thread folds the first index of every chunk itself and `workers − 1`
/// helpers, spawned once per call, fold the rest. A thread spawn and join measured 7-13 ms on
/// the Windows dev machine: one spawn per fold made a d = 48, G = 80 training take ~485 ms with
/// 2 or 8 workers against 26 ms with 1 (`G:/DeveloperData/Evidence/Qwerty/wp-a2/`).
fn run_folds<T: Send>(
    total: usize,
    workers: usize,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(TrainingProgress),
    event: fn(u32, u32) -> TrainingProgress,
    fold: &(dyn Fn(usize) -> Result<T, TrainingError> + Sync),
) -> Result<Vec<T>, TrainingError> {
    let workers = workers.clamp(1, total.max(1));
    std::thread::scope(|s| {
        let (report_tx, report_rx) = mpsc::channel::<FoldReport<T>>();
        let helpers: Vec<mpsc::Sender<usize>> = (1..workers)
            .map(|_| {
                let (job_tx, job_rx) = mpsc::channel::<usize>();
                let report_tx = report_tx.clone();
                s.spawn(move || {
                    for i in job_rx {
                        let result = panic::catch_unwind(AssertUnwindSafe(|| fold(i)));
                        if report_tx.send((i, result)).is_err() {
                            return;
                        }
                    }
                });
                job_tx
            })
            .collect();
        drop(report_tx);

        let mut out = Vec::with_capacity(total);
        let mut start = 0;
        while start < total {
            let end = (start + workers).min(total);
            for (job, i) in helpers.iter().zip(start + 1..end) {
                // INVARIANT: a helper runs until its job channel closes (`helpers` lives to the
                // end of this closure) and catches a fold's panic, so its receiver is alive.
                job.send(i)
                    .expect("a fold helper runs until its job channel closes");
            }
            let mut chunk: Vec<Option<Result<T, TrainingError>>> =
                (start..end).map(|_| None).collect();
            chunk[0] = Some(fold(start));
            for _ in start + 1..end {
                // INVARIANT: every helper holds a report sender until its job channel closes,
                // and each job sent above is answered exactly once.
                let (i, report) = report_rx
                    .recv()
                    .expect("a fold helper answers every job it receives");
                match report {
                    Ok(result) => chunk[i - start] = Some(result),
                    Err(payload) => panic::resume_unwind(payload),
                }
            }
            for result in chunk {
                // INVARIANT: the loop above filled one slot per fold of the chunk.
                out.push(result.expect("every fold of the chunk reported")?);
            }
            progress(event(count_u32(end), count_u32(total)));
            if cancel.load(Ordering::Relaxed) {
                return Err(TrainingError::Cancelled);
            }
            start = end;
        }
        Ok(out)
    })
}

/// §1.10.5: pooled nearest-rank thresholds. `negative_llr` is `Some` in the Negatives regime.
fn thresholds_from(
    params: &TrainingParams,
    positives: &[HeldOut],
    negative_llr: Option<&[f64]>,
) -> Thresholds {
    let sorted = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v
    };
    let d2_min = sorted(positives.iter().map(|s| s.d2_min).collect());
    let llr = sorted(positives.iter().map(|s| s.llr).collect());
    let tau_in = quantile_nearest_rank(&d2_min, 1.0 - params.alpha_in);
    let tau_bg_pos = quantile_nearest_rank(&llr, params.bg_positive_quantile);
    let tau_bg = match negative_llr {
        None => tau_bg_pos,
        Some(neg) => {
            let neg = sorted(neg.to_vec());
            tau_bg_pos.max(quantile_nearest_rank(&neg, params.bg_negative_quantile))
        }
    };
    Thresholds {
        alpha_in: params.alpha_in,
        tau_in,
        tau_bg,
        tau_margin: params.tau_margin,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Outcome {
    Accepted { zone: usize, correct: bool },
    Rejected(ModelRejectReason),
}

fn outcome(s: &HeldOut, thresholds: &Thresholds, truth: Option<usize>) -> Outcome {
    match thresholds.first_failure(s.d2_min, s.llr, s.margin) {
        Some(reason) => Outcome::Rejected(reason),
        None => Outcome::Accepted {
            zone: s.best,
            correct: truth == Some(s.best),
        },
    }
}

struct SummaryInputs<'a> {
    set: &'a TrainingSet,
    zone_of: &'a [usize],
    tap_scores: &'a [Vec<HeldOut>],
    negative_scores: &'a [Vec<HeldOut>],
    clip_sources: Vec<NegativeSource>,
    negative_vectors: usize,
    thresholds: Thresholds,
    within_lambda: f64,
    background_lambda: f64,
    regime: BackgroundRegime,
    standardization: &'a Standardization,
}

/// §1.10.6.
fn summarize(inp: SummaryInputs<'_>) -> TrainingSummary {
    let set = inp.set;
    let k = set.zone_ids.len();
    let mut clean = OutcomeCounts::default();
    let mut all = OutcomeCounts::default();
    let mut confusion = vec![vec![0u32; k + 1]; k];
    let mut zone_taps = vec![0u32; k];
    let mut zone_vectors = vec![0u32; k];
    for ((scores, &zk), tap) in inp.tap_scores.iter().zip(inp.zone_of).zip(&set.taps) {
        zone_taps[zk] += 1;
        zone_vectors[zk] += count_u32(tap.variants.len());
        for (v, s) in scores.iter().enumerate() {
            let o = outcome(s, &inp.thresholds, Some(zk));
            all.add(o);
            if v == 0 {
                clean.add(o);
                let column = match o {
                    Outcome::Accepted { zone, .. } => zone,
                    Outcome::Rejected(_) => k,
                };
                confusion[zk][column] += 1;
            }
        }
    }
    let zones: Vec<ZoneTrainingStats> = (0..k)
        .map(|zk| {
            let n = f64::from(zone_taps[zk]);
            ZoneTrainingStats {
                zone: set.zone_ids[zk],
                taps: zone_taps[zk],
                vectors: zone_vectors[zk],
                clean_accuracy: f64::from(confusion[zk][zk]) / n,
                clean_rejected: f64::from(confusion[zk][k]) / n,
            }
        })
        .collect();

    let mut negatives_held_out = 0u32;
    let mut negatives_held_out_accepted = 0u32;
    for s in inp.negative_scores.iter().flatten() {
        negatives_held_out += 1;
        if matches!(outcome(s, &inp.thresholds, None), Outcome::Accepted { .. }) {
            negatives_held_out_accepted += 1;
        }
    }
    let negatives: Vec<NegativeCount> = NegativeSource::ALL
        .iter()
        .filter_map(|&source| {
            let clips = inp.clip_sources.iter().filter(|&&s| s == source).count();
            (clips > 0).then(|| NegativeCount {
                source,
                clips: count_u32(clips),
            })
        })
        .collect();

    // Quality (§1.10.6).
    let mut issues = Vec::new();
    for z in &zones {
        if z.clean_accuracy < QUALITY_MIN_ZONE_CONSISTENCY {
            issues.push(QualityIssue::InconsistentZone {
                zone: z.zone,
                clean_accuracy: z.clean_accuracy,
            });
        }
    }
    for a in 0..k {
        for b in (a + 1)..k {
            let rate_ab = f64::from(confusion[a][b]) / f64::from(zone_taps[a]);
            let rate_ba = f64::from(confusion[b][a]) / f64::from(zone_taps[b]);
            let rate = rate_ab.max(rate_ba);
            if rate > QUALITY_MAX_CONFUSION {
                issues.push(QualityIssue::ZonesConfusable {
                    a: set.zone_ids[a],
                    b: set.zone_ids[b],
                    rate,
                });
            }
        }
    }
    let total = f64::from(clean.total);
    // SPEC: "clean rejected by τ_bg" reads as the clean taps whose decision is
    // Reject{Background} (the gates run in order), as the live engine would report them.
    let positive_rejection = f64::from(clean.rejected_bg) / total;
    if positive_rejection > QUALITY_MAX_BACKGROUND_REJECTION {
        issues.push(QualityIssue::BackgroundOverlap { positive_rejection });
    }
    let rate = f64::from(clean.rejected()) / total;
    if rate > QUALITY_MAX_REJECTION {
        issues.push(QualityIssue::HighRejection { rate });
    }
    if matches!(inp.regime, BackgroundRegime::PooledAll { .. }) {
        issues.push(QualityIssue::FewNegatives);
    }
    let st = inp.standardization;
    if !st.excluded_dims.is_empty() || !st.floored_dims.is_empty() {
        issues.push(QualityIssue::DegenerateDims {
            excluded: st.excluded_dims.clone(),
            floored: st.floored_dims.clone(),
        });
    }
    let poor = issues.iter().any(|i| {
        matches!(
            i,
            QualityIssue::InconsistentZone { .. } | QualityIssue::ZonesConfusable { .. }
        )
    });
    let quality = if poor {
        ModelQuality::Poor { issues }
    } else if !issues.is_empty() {
        ModelQuality::Marginal { issues }
    } else {
        ModelQuality::Good
    };

    TrainingSummary {
        seed: set.seed,
        zones,
        negatives,
        negative_vectors: count_u32(inp.negative_vectors),
        // SPEC: augmentation drops (§1.11.2 step 6) happen before training, which never sees
        // them; the calibration session (WP-B2) fills this count in after `train`.
        variants_dropped: 0,
        within_lambda: inp.within_lambda,
        background_lambda: inp.background_lambda,
        background_regime: inp.regime,
        thresholds: inp.thresholds,
        held_out_clean: clean,
        held_out_all: all,
        confusion_clean: confusion,
        negatives_held_out,
        negatives_held_out_accepted,
        quality,
    }
}

/// §1.10.3 step 6: the two map axes in standardized space, axes[i] = L⁻ᵀ·e_i. `clean` holds
/// (zone, clean z) of every tap; it is read only when K = 2.
fn map_axes(within: &Cholesky, means: &[Vec<f64>], clean: &[(usize, &[f64])]) -> [Vec<f64>; 2] {
    let d = within.dim;
    let whiten = |v: &[f64]| {
        let mut w = vec![0.0; d];
        within.forward_solve(v, &mut w);
        w
    };
    let w: Vec<Vec<f64>> = means.iter().map(|m| whiten(m)).collect();
    let [e1, e2] = if means.len() >= 3 {
        top_axes::<2>(&w)
    } else {
        let [e1] = top_axes::<1>(&w);
        let residuals: Vec<Vec<f64>> = clean
            .iter()
            .map(|&(zk, z)| {
                let dev: Vec<f64> = z.iter().zip(&means[zk]).map(|(a, b)| a - b).collect();
                let mut r = whiten(&dev);
                let along: f64 = r.iter().zip(&e1).map(|(a, b)| a * b).sum();
                add_into(&mut r, &e1, -along);
                r
            })
            .collect();
        let [e2] = top_axes::<1>(&residuals);
        [e1, e2]
    };
    let unwhiten = |e: &[f64]| {
        let mut a = vec![0.0; d];
        within.back_solve(e, &mut a);
        a
    };
    [unwhiten(&e1), unwhiten(&e2)]
}

/// The top `N` eigenvectors of Σ_r r·rᵀ as an array.
fn top_axes<const N: usize>(rows: &[Vec<f64>]) -> [Vec<f64>; N] {
    // INVARIANT: `top_eigenvectors` returns exactly `count` vectors.
    linalg::top_eigenvectors(rows, N)
        .try_into()
        .expect("top_eigenvectors returns `count` vectors")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::DESCRIPTOR_DIMS;
    use crate::model::{ModelDecision, ModelError, ModelInvalid, ModelScores};
    use crate::rng::Rng64;
    use proptest::prelude::*;
    use uuid::Uuid;

    const MONO: DescriptorLayout = DescriptorLayout::Mono { a: 0 };
    const STEREO: DescriptorLayout = DescriptorLayout::Stereo { a: 0, c: 1 };
    /// sd of a tap's clean descriptor around its zone mean, per dim (raw units before the
    /// fixture's per-dim affine map).
    const TAP_SD: f64 = 1.0;
    /// sd of an augmented variant around its tap's clean descriptor.
    const VARIANT_SD: f64 = 0.7;

    fn rate() -> SampleRate {
        SampleRate::new(48_000).expect("48 kHz is valid")
    }

    fn zone_ids(k: usize) -> Vec<ZoneId> {
        (0..k)
            .map(|i| ZoneId::from_uuid(Uuid::from_u128(i as u128 + 1)))
            .collect()
    }

    fn noisy(rng: &mut Rng64, centre: &[f64], sd: f64) -> Vec<f64> {
        centre.iter().map(|c| c + sd * rng.normal()).collect()
    }

    fn random_means(rng: &mut Rng64, k: usize, d: usize, spread: f64) -> Vec<Vec<f64>> {
        (0..k).map(|_| noisy(rng, &vec![0.0; d], spread)).collect()
    }

    /// A per-dim affine map (mixed units, as real descriptor dims have) applied to every value.
    struct Units {
        offset: Vec<f64>,
        scale: Vec<f64>,
    }

    impl Units {
        fn random(rng: &mut Rng64, d: usize) -> Self {
            Units {
                offset: (0..d).map(|_| rng.range(-50.0, 50.0)).collect(),
                scale: (0..d).map(|_| 10f64.powf(rng.range(-2.0, 2.0))).collect(),
            }
        }

        fn identity(d: usize) -> Self {
            Units {
                offset: vec![0.0; d],
                scale: vec![1.0; d],
            }
        }

        fn descriptor(&self, layout: DescriptorLayout, x: &[f64]) -> Descriptor {
            let mut values = [0.0f32; DESCRIPTOR_DIMS];
            for (g, v) in values.iter_mut().take(layout.dims()).enumerate() {
                *v = (self.offset[g] + self.scale[g] * x[g]) as f32;
            }
            Descriptor { layout, values }
        }
    }

    /// Variant 0 is `centre` itself (the clean descriptor); the others add variant noise.
    fn variants_of(
        rng: &mut Rng64,
        units: &Units,
        layout: DescriptorLayout,
        centre: &[f64],
        count: usize,
        sd: f64,
    ) -> Vec<Descriptor> {
        (0..count)
            .map(|v| {
                if v == 0 {
                    units.descriptor(layout, centre)
                } else {
                    units.descriptor(layout, &noisy(rng, centre, sd))
                }
            })
            .collect()
    }

    /// `taps_per_zone` taps of every zone, zone by zone, each with `variants` variants.
    fn make_set(
        rng: &mut Rng64,
        units: &Units,
        layout: DescriptorLayout,
        means: &[Vec<f64>],
        taps_per_zone: usize,
        variants: usize,
    ) -> TrainingSet {
        let mut taps = Vec::new();
        for (zk, mean) in means.iter().enumerate() {
            for _ in 0..taps_per_zone {
                let centre = noisy(rng, mean, TAP_SD);
                taps.push(TapGroup {
                    zone: zone_index(zk),
                    tap: TapIndex(count_u32(taps.len())),
                    variants: variants_of(rng, units, layout, &centre, variants, VARIANT_SD),
                });
            }
        }
        TrainingSet {
            layout,
            rate: rate(),
            zone_ids: zone_ids(means.len()),
            seed: 0x5EED,
            taps,
            negatives: Vec::new(),
        }
    }

    /// How a fixture's negative clips are spread: clip centres around `centre` with sd `clip_sd`,
    /// variants around their clip with sd `variant_sd`.
    struct NegativeSpec<'a> {
        centre: &'a [f64],
        clips: usize,
        variants: usize,
        clip_sd: f64,
        variant_sd: f64,
    }

    fn negatives(
        rng: &mut Rng64,
        units: &Units,
        layout: DescriptorLayout,
        spec: NegativeSpec<'_>,
    ) -> Vec<NegativeGroup> {
        (0..spec.clips)
            .map(|c| {
                let clip = noisy(rng, spec.centre, spec.clip_sd);
                NegativeGroup {
                    source: NegativeSource::ALL[c % 3],
                    variants: variants_of(
                        rng,
                        units,
                        layout,
                        &clip,
                        spec.variants,
                        spec.variant_sd,
                    ),
                }
            })
            .collect()
    }

    fn params(workers: Workers) -> TrainingParams {
        TrainingParams {
            workers,
            ..TrainingParams::DEFAULT
        }
    }

    fn exactly(n: u8) -> Workers {
        Workers::Exactly {
            n: NonZeroU8::new(n).expect("non-zero worker count"),
        }
    }

    fn train_with(
        set: &TrainingSet,
        params: &TrainingParams,
    ) -> Result<TrainedModel, TrainingError> {
        train(set, params, &AtomicBool::new(false), &mut |_| {})
    }

    fn train_ok(set: &TrainingSet) -> TrainedModel {
        let model = train_with(set, &params(exactly(2))).expect("trains");
        model.validate().expect("a trained model validates");
        model
    }

    /// The raw descriptor whose standardized vector is `z` (excluded dims carry x_g = offset_g).
    fn probe_at(model: &TrainedModel, z: &[f64]) -> Descriptor {
        let st = &model.standardization;
        let x: Vec<f64> = z
            .iter()
            .zip(&st.offset)
            .zip(&st.scale)
            .map(|((z, o), s)| o + s * z)
            .collect();
        Units::identity(x.len()).descriptor(model.layout, &x)
    }

    fn decide(model: &TrainedModel, d: &Descriptor) -> ModelDecision {
        let mut scratch = model.scratch();
        model.decide(d, &mut scratch).expect("decides")
    }

    fn scores(decision: &ModelDecision) -> &ModelScores {
        match decision {
            ModelDecision::Accept { scores, .. } | ModelDecision::Reject { scores, .. } => scores,
        }
    }

    fn is_reject(decision: &ModelDecision, expected: ModelRejectReason) -> bool {
        matches!(decision, ModelDecision::Reject { reason, .. } if *reason == expected)
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn unit(v: &mut [f64]) {
        let n = dot(v, v).sqrt();
        v.iter_mut().for_each(|x| *x /= n);
    }

    fn rel_close(a: &[f64], b: &[f64], rel: f64) -> bool {
        let scale = b
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()))
            .max(f64::MIN_POSITIVE);
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= rel * scale)
    }

    fn sym_entries(s: &SymMatrix) -> Vec<f64> {
        let d = s.dim();
        (0..d * d).map(|i| s.get(i / d, i % d)).collect()
    }

    #[test]
    fn m1_separated_zones_train_accurately_and_deterministically() {
        let mut rng = Rng64::new(0xA201);
        let units = Units::random(&mut rng, 38);
        let means = random_means(&mut rng, 4, 38, 3.0);
        let set = make_set(&mut rng, &units, MONO, &means, 10, 8);
        let reference = train_with(&set, &params(exactly(1))).expect("trains");
        reference.validate().expect("validates");
        let clean = reference.summary.held_out_clean;
        assert_eq!(clean.total, 40);
        let accuracy = f64::from(clean.correct) / f64::from(clean.total);
        assert!(accuracy >= 0.99, "held-out clean accuracy {accuracy}");
        let t = reference.thresholds;
        println!(
            "M1 d=38 K=4 G=40 V=8: tau_in {:.2} tau_bg {:.2} lambda_w {:.3} lambda_bg {:.3} \
             clean {:?} all {:?}",
            t.tau_in,
            t.tau_bg,
            reference.summary.within_lambda,
            reference.summary.background_lambda,
            clean,
            reference.summary.held_out_all
        );
        let json = serde_json::to_string(&reference).expect("serializes");
        for workers in [
            exactly(1),
            exactly(1),
            exactly(2),
            exactly(8),
            Workers::Auto,
        ] {
            let model = train_with(&set, &params(workers.clone())).expect("trains");
            let again = serde_json::to_string(&model).expect("serializes");
            assert!(again == json, "{workers:?} changed the model JSON");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]
        #[test]
        fn m2_loto_sufficient_statistics_match_refit(
            seed in any::<u64>(),
            d in prop::sample::select(vec![5usize, 38, 48]),
            k in 2usize..=8,
        ) {
            let mut rng = Rng64::new(seed);
            let mut zone_of = Vec::new();
            let mut tap_z: Vec<Vec<Vec<f64>>> = Vec::new();
            for (zk, mean) in random_means(&mut rng, k, d, 3.0).iter().enumerate() {
                for _ in 0..(2 + rng.below(3)) {
                    let centre = noisy(&mut rng, mean, TAP_SD);
                    let v = 1 + rng.below(4) as usize;
                    zone_of.push(zk);
                    tap_z.push((0..v).map(|_| noisy(&mut rng, &centre, VARIANT_SD)).collect());
                }
            }
            let g = tap_z.len();
            let moments: Vec<GroupMoments> = tap_z.iter().map(|v| GroupMoments::of(v, d)).collect();
            let clean: Vec<&[f64]> = tap_z.iter().map(|v| v[0].as_slice()).collect();
            let sums = WithinSums::new(&moments, &clean, &zone_of, k, d);

            // Within folds against a brute-force refit without tap h.
            for h in 0..g {
                let (m_prime, s_prime) = sums.fold(&moments[h], clean[h], zone_of[h]);
                let keep: Vec<usize> = (0..g).filter(|&t| t != h).collect();
                let refit_means: Vec<Vec<f64>> = (0..k)
                    .map(|zk| {
                        let members: Vec<usize> =
                            keep.iter().copied().filter(|&t| zone_of[t] == zk).collect();
                        let mut m = vec![0.0; d];
                        for &t in &members {
                            add_into(&mut m, clean[t], 1.0 / members.len() as f64);
                        }
                        m
                    })
                    .collect();
                let mut refit_s = SymMatrix::zeros(d);
                for &t in &keep {
                    let y = deviation_scatter(&tap_z[t], &refit_means[zone_of[t]], d);
                    refit_s.add_scaled(&y, 1.0 / (g - 1) as f64);
                }
                prop_assert!(rel_close(&m_prime, &refit_means[zone_of[h]], 1e-9));
                prop_assert!(rel_close(&sym_entries(&s_prime), &sym_entries(&refit_s), 1e-9));
            }

            // Background folds: PooledAll (every tap a group) and Negatives (clip groups), each
            // removing the held-out group's variant mean.
            let clips: Vec<Vec<Vec<f64>>> = (0..(3 + rng.below(6)))
                .map(|_| {
                    let centre = noisy(&mut rng, &vec![0.0; d], 2.0);
                    let v = 1 + rng.below(4) as usize;
                    (0..v).map(|_| noisy(&mut rng, &centre, 0.5)).collect()
                })
                .collect();
            for groups in [&tap_z, &clips] {
                let n = groups.len();
                let gm: Vec<GroupMoments> = groups.iter().map(|v| GroupMoments::of(v, d)).collect();
                let pool = PoolSums::new(&gm, d);
                for (h, held_out) in gm.iter().enumerate() {
                    let (m_prime, s_prime) = pool.fold(held_out);
                    let keep: Vec<usize> = (0..n).filter(|&c| c != h).collect();
                    let mut refit_mean = vec![0.0; d];
                    for &c in &keep {
                        let variant_mean = GroupMoments::of(&groups[c], d).b;
                        add_into(&mut refit_mean, &variant_mean, 1.0 / (n - 1) as f64);
                    }
                    let mut refit_s = SymMatrix::zeros(d);
                    for &c in &keep {
                        let y = deviation_scatter(&groups[c], &refit_mean, d);
                        refit_s.add_scaled(&y, 1.0 / (n - 1) as f64);
                    }
                    prop_assert!(rel_close(&m_prime, &refit_mean, 1e-9));
                    prop_assert!(rel_close(&sym_entries(&s_prime), &sym_entries(&refit_s), 1e-9));
                }
            }
        }
    }

    #[test]
    fn m3_threshold_rules() {
        let mut rng = Rng64::new(0xA203);
        let held = |d2_min: f64, llr: f64| HeldOut {
            best: 0,
            d2_min,
            margin: 10.0,
            llr,
        };
        // 200 positives with d2_min = 1..=200 and llr = −99..=100, in shuffled order.
        let mut positives: Vec<HeldOut> = (1..=200)
            .map(|i| held(f64::from(i), f64::from(i) - 100.0))
            .collect();
        for i in (1..positives.len()).rev() {
            positives.swap(i, rng.below(i as u64 + 1) as usize);
        }
        let p = TrainingParams::DEFAULT;
        // PooledAll: τ_in = Q_0.98 = sorted[195], τ_bg = Q_0.01 = sorted[1].
        let pooled = thresholds_from(&p, &positives, None);
        assert_eq!(pooled.tau_in, 196.0);
        assert_eq!(pooled.tau_bg, -98.0);
        assert_eq!(pooled.alpha_in, 0.02);
        assert_eq!(pooled.tau_margin, 4.0);
        // Negatives: max(τ_bg_pos, Q_0.95 of the negative llr); 40 values → sorted[37].
        let high: Vec<f64> = (1..=40).rev().map(f64::from).collect();
        assert_eq!(thresholds_from(&p, &positives, Some(&high)).tau_bg, 38.0);
        let low: Vec<f64> = (1..=40).map(|i| -1000.0 - f64::from(i)).collect();
        assert_eq!(thresholds_from(&p, &positives, Some(&low)).tau_bg, -98.0);
        // α_in, τ_m and the quantiles are parameters.
        let custom = TrainingParams {
            alpha_in: 0.1,
            tau_margin: 6.0,
            bg_positive_quantile: 0.5,
            ..TrainingParams::DEFAULT
        };
        let t = thresholds_from(&custom, &positives, None);
        assert_eq!((t.tau_in, t.tau_bg, t.tau_margin), (180.0, 0.0, 6.0));

        // A trained model carries one pooled set of thresholds, recorded in its summary.
        let means = random_means(&mut rng, 3, 38, 3.0);
        let set = make_set(&mut rng, &Units::identity(38), MONO, &means, 6, 4);
        let model = train_ok(&set);
        assert_eq!(model.thresholds, model.summary.thresholds);
        assert_eq!(model.thresholds.alpha_in, 0.02);
        assert_eq!(model.thresholds.tau_margin, 4.0);
    }

    /// Zones 0 and 1 close together, zones 2 and 3 far away, and a tight negative cluster 2
    /// within-sd from zone 2 (Negatives regime).
    fn semantics_model() -> TrainedModel {
        let d = 38;
        let mut rng = Rng64::new(0xA204);
        let units = Units::random(&mut rng, d);
        let mut means = random_means(&mut rng, 4, d, 3.0);
        means[1] = means[0].clone();
        means[1][0] += 7.0;
        let mut neg_centre = means[2].clone();
        neg_centre[1] += 2.0;
        let mut set = make_set(&mut rng, &units, MONO, &means, 10, 6);
        let spec = NegativeSpec {
            centre: &neg_centre,
            clips: 30,
            variants: 4,
            clip_sd: 0.3,
            variant_sd: 0.2,
        };
        set.negatives = negatives(&mut rng, &units, MONO, spec);
        train_ok(&set)
    }

    #[test]
    fn m4_decision_semantics() {
        let model = semantics_model();
        assert_eq!(
            model.background.regime,
            BackgroundRegime::Negatives {
                clips: 30,
                vectors: 120
            }
        );
        let m = &model.zone_means;
        let midpoint: Vec<f64> = m[0].iter().zip(&m[1]).map(|(a, b)| 0.5 * (a + b)).collect();
        let decision = decide(&model, &probe_at(&model, &midpoint));
        assert!(
            is_reject(&decision, ModelRejectReason::Ambiguous),
            "midpoint: {decision:?}"
        );

        // 6 within-sd away from zone 0 in every dim: outside every zone.
        let far: Vec<f64> = m[0].iter().map(|v| v + 6.0).collect();
        let decision = decide(&model, &probe_at(&model, &far));
        assert!(
            is_reject(&decision, ModelRejectReason::OutOfDistribution),
            "far: {decision:?}"
        );

        // The negative cluster's centre, inside the zone hull: passes τ_in, fails τ_bg.
        let decision = decide(&model, &probe_at(&model, &model.background.mean));
        assert!(
            scores(&decision).d2_min <= model.thresholds.tau_in,
            "the negative centre is inside the zone hull: {decision:?}"
        );
        assert!(
            is_reject(&decision, ModelRejectReason::Background),
            "negative centre: {decision:?}"
        );

        // A zone mean away from the negative cluster is accepted as that zone.
        for (zk, mean) in m.iter().enumerate().filter(|&(zk, _)| zk != 2) {
            let decision = decide(&model, &probe_at(&model, mean));
            assert!(
                matches!(decision, ModelDecision::Accept { zone, .. } if zone == zone_index(zk)),
                "zone {zk} mean: {decision:?}"
            );
        }
    }

    /// The defect M5 injects into an otherwise valid set.
    #[derive(Debug, Clone, Copy)]
    enum Defect {
        None,
        OneZone,
        FourTaps,
        IdenticalVectors,
        NaN,
        Infinite,
        EmptyTap,
        Layout,
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        #[test]
        fn m5_degenerate_inputs_return_named_errors(
            seed in any::<u64>(),
            defect in prop::sample::select(vec![
                Defect::None, Defect::OneZone, Defect::FourTaps, Defect::IdenticalVectors,
                Defect::NaN, Defect::Infinite, Defect::EmptyTap, Defect::Layout,
            ]),
            k in 2usize..=4,
            taps in 5usize..=7,
            variants in 1usize..=3,
            negative_clips in 0usize..=3,
        ) {
            let mut rng = Rng64::new(seed);
            let means = random_means(&mut rng, k, 38, 3.0);
            let units = Units::identity(38);
            let mut set = make_set(&mut rng, &units, MONO, &means, taps, variants);
            let spec = NegativeSpec {
                centre: &means[0],
                clips: negative_clips,
                variants: 2,
                clip_sd: 1.0,
                variant_sd: 0.5,
            };
            set.negatives = negatives(&mut rng, &units, MONO, spec);
            let victim = rng.below(set.taps.len() as u64) as usize;
            let dim = rng.below(38) as usize;
            match defect {
                Defect::None => {}
                Defect::OneZone => {
                    set.zone_ids.truncate(1);
                    set.taps.retain(|t| t.zone == ZoneIndex(0));
                }
                Defect::FourTaps => {
                    // Zone 1's taps are the contiguous range taps..2·taps.
                    set.taps.drain(taps + 4..2 * taps);
                }
                Defect::IdenticalVectors => {
                    let one = set.taps[0].variants[0];
                    for t in &mut set.taps {
                        t.variants.iter_mut().for_each(|v| *v = one);
                    }
                }
                Defect::NaN => set.taps[victim].variants[0].values[dim] = f32::NAN,
                Defect::Infinite => set.taps[victim].variants[0].values[dim] = f32::INFINITY,
                Defect::EmptyTap => set.taps[victim].variants.clear(),
                Defect::Layout => set.taps[victim].variants[0].layout = STEREO,
            }
            let result = train_with(&set, &params(exactly(2)));
            let victim_tap = TapIndex(count_u32(victim));
            match defect {
                Defect::None => prop_assert!(result.is_ok(), "{:?}", result),
                Defect::OneZone => {
                    prop_assert_eq!(result, Err(TrainingError::NotEnoughZones { have: 1 }));
                }
                Defect::FourTaps => prop_assert_eq!(
                    result,
                    Err(TrainingError::NotEnoughTaps { zone: ZoneIndex(1), have: 4, need: 5 })
                ),
                Defect::IdenticalVectors => prop_assert_eq!(
                    result,
                    Err(TrainingError::DegenerateWithinZoneSpread { degenerate_dims: 38, dims: 38 })
                ),
                Defect::NaN | Defect::Infinite => prop_assert_eq!(
                    result,
                    Err(TrainingError::NonFiniteDescriptor { tap: victim_tap, index: dim_u16(dim) })
                ),
                Defect::EmptyTap => {
                    prop_assert_eq!(result, Err(TrainingError::EmptyTap { tap: victim_tap }));
                }
                Defect::Layout => {
                    prop_assert_eq!(result, Err(TrainingError::LayoutMismatch { tap: victim_tap }));
                }
            }
        }
    }

    #[test]
    fn train_rejects_out_of_range_params() {
        let mut rng = Rng64::new(0xA2F1);
        let means = random_means(&mut rng, 2, 38, 3.0);
        let set = make_set(&mut rng, &Units::identity(38), MONO, &means, 5, 2);
        use TrainingParam::{AlphaIn, BgNegativeQuantile, BgPositiveQuantile, TauMargin};
        // Each bad value is rejected up front, before any quantile is taken (no debug panic)
        // and before a model with a non-finite τ_m is built. The set has no negatives, so
        // q_bg− would never be used: it is still checked.
        type Case = (TrainingParam, fn(&mut TrainingParams));
        let cases: [Case; 11] = [
            (AlphaIn, |p| p.alpha_in = 1.0),
            (AlphaIn, |p| p.alpha_in = -0.01),
            (AlphaIn, |p| p.alpha_in = f64::NAN),
            (TauMargin, |p| p.tau_margin = f64::NAN),
            (TauMargin, |p| p.tau_margin = f64::INFINITY),
            (BgPositiveQuantile, |p| p.bg_positive_quantile = 0.0),
            (BgPositiveQuantile, |p| p.bg_positive_quantile = 1.5),
            (BgPositiveQuantile, |p| p.bg_positive_quantile = f64::NAN),
            (BgNegativeQuantile, |p| p.bg_negative_quantile = 0.0),
            (BgNegativeQuantile, |p| p.bg_negative_quantile = f64::NAN),
            // The first bad field is the one reported.
            (AlphaIn, |p| {
                p.alpha_in = 2.0;
                p.bg_negative_quantile = 2.0;
            }),
        ];
        for (i, (param, defect)) in cases.iter().enumerate() {
            let mut p = params(exactly(1));
            defect(&mut p);
            assert_eq!(
                train_with(&set, &p).err(),
                Some(TrainingError::InvalidParam { param: *param }),
                "case {i}"
            );
        }
        // The params are checked before the set.
        let mut one_zone = set.clone();
        one_zone.zone_ids.truncate(1);
        let p = TrainingParams {
            alpha_in: 1.0,
            ..params(exactly(1))
        };
        assert_eq!(
            train_with(&one_zone, &p).err(),
            Some(TrainingError::InvalidParam { param: AlphaIn })
        );
        // The range ends are valid: α_in = 0 and level 1 take the largest held-out value.
        let ends = TrainingParams {
            alpha_in: 0.0,
            bg_positive_quantile: 1.0,
            bg_negative_quantile: 1.0,
            ..params(exactly(1))
        };
        let model = train_with(&set, &ends).expect("the range ends train");
        model
            .validate()
            .expect("a model trained at the range ends validates");
    }

    #[test]
    fn m6_json_round_trip_keeps_decisions() {
        let model = semantics_model();
        let json = serde_json::to_string(&model).expect("serializes");
        let back: TrainedModel = serde_json::from_str(&json).expect("deserializes");
        back.validate().expect("the loaded model validates");
        let mut rng = Rng64::new(0xA206);
        let mut accepted = 0;
        let mut rejected = 0;
        for _ in 0..1000 {
            let zk = rng.below(model.zone_means.len() as u64) as usize;
            let spread = [0.5, 1.5, 3.0, 8.0][rng.below(4) as usize];
            let z = noisy(&mut rng, &model.zone_means[zk], spread);
            let probe = probe_at(&model, &z);
            let (a, b) = (decide(&model, &probe), decide(&back, &probe));
            match (&a, &b) {
                (
                    ModelDecision::Accept { zone: za, .. },
                    ModelDecision::Accept { zone: zb, .. },
                ) => {
                    assert_eq!(za, zb);
                    accepted += 1;
                }
                (
                    ModelDecision::Reject { reason: ra, .. },
                    ModelDecision::Reject { reason: rb, .. },
                ) => {
                    assert_eq!(ra, rb);
                    rejected += 1;
                }
                (ModelDecision::Accept { .. }, ModelDecision::Reject { .. })
                | (ModelDecision::Reject { .. }, ModelDecision::Accept { .. }) => {
                    panic!("the round trip changed the decision: {a:?} vs {b:?}")
                }
            }
            // SPEC: scores agree to rounding, not bit for bit: serde_json without its
            // `float_roundtrip` feature may parse a float 1 ULP off.
            let (sa, sb) = (scores(&a), scores(&b));
            let tol = 1e-9 * (1.0 + sa.d2.iter().fold(0.0f64, |m, v| m.max(v.abs())));
            assert_eq!((sa.best, sa.second), (sb.best, sb.second));
            assert!(rel_close(&sb.d2, &sa.d2, 1e-9));
            for (x, y) in [
                (sa.d2_min, sb.d2_min),
                (sa.margin, sb.margin),
                (sa.llr_bg, sb.llr_bg),
                (sa.posterior_best, sb.posterior_best),
            ] {
                assert!((x - y).abs() <= tol, "{x} vs {y}");
            }
        }
        assert!(accepted > 0 && rejected > 0, "{accepted} / {rejected}");
    }

    #[test]
    #[ignore = "timing, informative: cargo test --release -- --ignored bench_decide"]
    fn m7_bench_decide() {
        let mut rng = Rng64::new(0xA207);
        let units = Units::random(&mut rng, 48);
        let means = random_means(&mut rng, 8, 48, 3.0);
        let mut set = make_set(&mut rng, &units, STEREO, &means, 10, 8);
        let spec = NegativeSpec {
            centre: &means[0],
            clips: 30,
            variants: 8,
            clip_sd: 2.0,
            variant_sd: 0.5,
        };
        set.negatives = negatives(&mut rng, &units, STEREO, spec);
        let model = train_ok(&set);
        // Training time per worker count (G = 80, V = 8, 30 negative clips of 8).
        let mut train_ms = Vec::new();
        for n in [1u8, 2, 8] {
            let started = std::time::Instant::now();
            let again = train_with(&set, &params(exactly(n))).expect("trains");
            train_ms.push((n, started.elapsed().as_secs_f64() * 1e3));
            assert_eq!(again, model, "{n} workers changed the model");
        }
        let probes: Vec<Descriptor> = (0..1000)
            .map(|i| {
                let z = noisy(&mut rng, &model.zone_means[i % 8], 1.0);
                probe_at(&model, &z)
            })
            .collect();
        let mut scratch = model.scratch();
        let rounds = 50;
        let mut accepted = 0usize;
        let started = std::time::Instant::now();
        for _ in 0..rounds {
            for p in &probes {
                if let Ok(ModelDecision::Accept { .. }) = model.decide(p, &mut scratch) {
                    accepted += 1;
                }
            }
        }
        let calls = rounds * probes.len();
        let per_us = started.elapsed().as_secs_f64() * 1e6 / calls as f64;
        let t = model.thresholds;
        println!(
            "M7 decide d=48 K=8: {per_us:.2} us per call over {calls} calls (budget 50 us), \
             {accepted} accepted"
        );
        for (n, ms) in train_ms {
            println!(
                "M7 train d=48 K=8 G=80 V=8 + 30 negative clips x 8: {ms:.1} ms ({n} workers)"
            );
        }
        println!(
            "M7 thresholds d=48 K=8: tau_in {:.2} tau_bg {:.2} lambda_w {:.3} lambda_bg {:.3} \
             regime {:?} clean {:?}",
            t.tau_in,
            t.tau_bg,
            model.summary.within_lambda,
            model.summary.background_lambda,
            model.background.regime,
            model.summary.held_out_clean
        );
    }

    #[test]
    fn m8_background_regime_counts_clips() {
        let mut rng = Rng64::new(0xA208);
        let units = Units::identity(38);
        let means = random_means(&mut rng, 3, 38, 3.0);
        let mut set = make_set(&mut rng, &units, MONO, &means, 6, 4);
        let far = vec![20.0; 38];
        let spec = NegativeSpec {
            centre: &far,
            clips: 24,
            variants: 8,
            clip_sd: 1.0,
            variant_sd: 0.5,
        };
        let clips = negatives(&mut rng, &units, MONO, spec);

        set.negatives = clips[..23].to_vec();
        let model = train_ok(&set);
        let pooled = BackgroundRegime::PooledAll {
            negative_clips: 23,
            needed: 24,
        };
        assert_eq!(model.background.regime, pooled);
        assert_eq!(model.summary.background_regime, pooled);
        assert_eq!(model.summary.negative_vectors, 184);
        assert!(model
            .summary
            .quality
            .issues()
            .contains(&QualityIssue::FewNegatives));

        set.negatives = clips;
        let model = train_ok(&set);
        assert_eq!(
            model.background.regime,
            BackgroundRegime::Negatives {
                clips: 24,
                vectors: 192
            }
        );
        assert_eq!(model.summary.negatives_held_out, 192);
        let per_source: Vec<NegativeCount> = NegativeSource::ALL
            .iter()
            .map(|&source| NegativeCount { source, clips: 8 })
            .collect();
        assert_eq!(model.summary.negatives, per_source);
        assert!(!model
            .summary
            .quality
            .issues()
            .contains(&QualityIssue::FewNegatives));

        // A clip with no surviving variant does not count.
        set.negatives[5].variants.clear();
        let model = train_ok(&set);
        assert_eq!(model.background.regime, pooled);
    }

    #[test]
    fn m9_inconsistent_zone_is_flagged() {
        let mut rng = Rng64::new(0xA209);
        let units = Units::random(&mut rng, 38);
        let means = random_means(&mut rng, 6, 38, 3.0);
        let mut set = make_set(&mut rng, &units, MONO, &means, 10, 4);
        // Half of zone 2's taps come from zone 0's distribution.
        for t in set
            .taps
            .iter_mut()
            .filter(|t| t.zone == ZoneIndex(2))
            .take(5)
        {
            let centre = noisy(&mut rng, &means[0], TAP_SD);
            t.variants = variants_of(&mut rng, &units, MONO, &centre, 4, VARIANT_SD);
        }
        let model = train_ok(&set);
        let issues = model.summary.quality.issues();
        let inconsistent: Vec<ZoneId> = issues
            .iter()
            .filter_map(|i| match i {
                QualityIssue::InconsistentZone { zone, .. } => Some(*zone),
                QualityIssue::ZonesConfusable { .. }
                | QualityIssue::BackgroundOverlap { .. }
                | QualityIssue::HighRejection { .. }
                | QualityIssue::FewNegatives
                | QualityIssue::DegenerateDims { .. } => None,
            })
            .collect();
        assert_eq!(inconsistent, vec![set.zone_ids[2]], "{issues:?}");
        assert!(matches!(model.summary.quality, ModelQuality::Poor { .. }));
    }

    #[test]
    fn m10_degenerate_dims_are_excluded_or_floored() {
        let d = 38;
        let mut rng = Rng64::new(0xA20A);
        let units = Units::random(&mut rng, d);
        let means = random_means(&mut rng, 4, d, 3.0);
        let mut set = make_set(&mut rng, &units, MONO, &means, 8, 4);
        // Dim 5 is constant over every tap; dim 7 is zone-dependent with a tiny within spread.
        for t in &mut set.taps {
            let zone_value = 10.0 * f64::from(t.zone.0);
            for v in &mut t.variants {
                v.values[5] = 0.7;
                v.values[7] = (zone_value + 0.01 * rng.normal()) as f32;
            }
        }
        let model = train_ok(&set);
        assert_eq!(model.standardization.excluded_dims, vec![5]);
        assert_eq!(model.standardization.floored_dims, vec![7]);
        let degenerate = QualityIssue::DegenerateDims {
            excluded: vec![5],
            floored: vec![7],
        };
        assert!(model.summary.quality.issues().contains(&degenerate));
        assert!(!matches!(model.summary.quality, ModelQuality::Good));
        // Whatever value a probe carries in the excluded dim, the decision is the same.
        for _ in 0..1000 {
            let zk = rng.below(4) as usize;
            let z = noisy(&mut rng, &model.zone_means[zk], 1.5);
            let probe = probe_at(&model, &z);
            let mut other = probe;
            other.values[5] = rng.range(-1e6, 1e6) as f32;
            assert_eq!(decide(&model, &probe), decide(&model, &other));
        }

        // d/2 = 19 degenerate dims still train; 20 do not.
        let constant = |set: &mut TrainingSet, dims: usize| {
            for t in &mut set.taps {
                for v in &mut t.variants {
                    v.values[..dims].iter_mut().for_each(|x| *x = 1.5);
                }
            }
        };
        let mut set19 = make_set(&mut rng, &units, MONO, &means, 8, 4);
        constant(&mut set19, 19);
        let model = train_ok(&set19);
        assert_eq!(model.standardization.excluded_dims.len(), 19);
        let mut set20 = make_set(&mut rng, &units, MONO, &means, 8, 4);
        constant(&mut set20, 20);
        assert_eq!(
            train_with(&set20, &params(exactly(2))),
            Err(TrainingError::DegenerateWithinZoneSpread {
                degenerate_dims: 20,
                dims: 38
            })
        );
    }

    #[test]
    fn m11_constant_negative_dim_keeps_background_factorable() {
        let d = 38;
        let mut rng = Rng64::new(0xA20B);
        let units = Units::random(&mut rng, d);
        let means = random_means(&mut rng, 3, d, 3.0);
        let mut set = make_set(&mut rng, &units, MONO, &means, 8, 4);
        let spec = NegativeSpec {
            centre: &means[1],
            clips: 30,
            variants: 4,
            clip_sd: 1.5,
            variant_sd: 0.5,
        };
        set.negatives = negatives(&mut rng, &units, MONO, spec);
        for n in &mut set.negatives {
            n.variants.iter_mut().for_each(|v| v.values[3] = 2.25);
        }
        let model = train_ok(&set);
        assert!(matches!(
            model.background.regime,
            BackgroundRegime::Negatives { .. }
        ));
        assert_eq!(model.background.factor.jitter, 0.0);
        for deviation in [-50.0, -5.0, 0.5, 5.0, 50.0] {
            let mut z = model.zone_means[0].clone();
            z[3] += deviation;
            let decision = decide(&model, &probe_at(&model, &z));
            assert!(scores(&decision).llr_bg.is_finite(), "{decision:?}");
        }
    }

    #[test]
    fn m12_cancel_stops_after_first_chunk() {
        let mut rng = Rng64::new(0xA20C);
        let means = random_means(&mut rng, 3, 38, 3.0);
        let set = make_set(&mut rng, &Units::identity(38), MONO, &means, 6, 3);
        let mut events = Vec::new();
        let result = train(
            &set,
            &params(exactly(2)),
            &AtomicBool::new(true),
            &mut |p| events.push(p),
        );
        assert_eq!(result, Err(TrainingError::Cancelled));
        assert_eq!(
            events,
            vec![
                TrainingProgress::Standardize,
                TrainingProgress::FitWithin,
                TrainingProgress::Background,
                TrainingProgress::Folds { done: 2, total: 18 },
            ]
        );
    }

    /// m = L·w for the lower-triangular factor.
    fn lower_mul(c: &Cholesky, w: &[f64]) -> Vec<f64> {
        (0..c.dim)
            .map(|i| (0..=i).map(|j| c.lower[i * (i + 1) / 2 + j] * w[j]).sum())
            .collect()
    }

    /// e = Lᵀ·a.
    fn lower_t_mul(c: &Cholesky, a: &[f64]) -> Vec<f64> {
        (0..c.dim)
            .map(|j| {
                (j..c.dim)
                    .map(|i| c.lower[i * (i + 1) / 2 + j] * a[i])
                    .sum()
            })
            .collect()
    }

    #[test]
    fn m13_sound_map() {
        let d = 48;
        let mut rng = Rng64::new(0xA20D);
        // 4 zones whose whitened means are the corners of a square in a random plane: the map
        // keeps their pairwise distances (measured in f64 on axes·m; stored points are f32).
        for _ in 0..20 {
            let mut a = SymMatrix::zeros(d);
            for _ in 0..(d + 10) {
                a.add_outer(&noisy(&mut rng, &vec![0.0; d], 1.0), 1.0);
            }
            let l = Cholesky::factor(&a).expect("SPD");
            let mut p = noisy(&mut rng, &vec![0.0; d], 1.0);
            unit(&mut p);
            let mut q = noisy(&mut rng, &vec![0.0; d], 1.0);
            let along = dot(&q, &p);
            add_into(&mut q, &p, -along);
            unit(&mut q);
            let side = rng.range(0.5, 20.0);
            let corners = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
            let w: Vec<Vec<f64>> = corners
                .iter()
                .map(|(cp, cq)| {
                    p.iter()
                        .zip(&q)
                        .map(|(x, y)| side * (cp * x + cq * y))
                        .collect()
                })
                .collect();
            let means: Vec<Vec<f64>> = w.iter().map(|wk| lower_mul(&l, wk)).collect();
            let clean: Vec<(usize, &[f64])> =
                means.iter().map(|m| m.as_slice()).enumerate().collect();
            let axes = map_axes(&l, &means, &clean);
            let point = |m: &[f64]| [dot(&axes[0], m), dot(&axes[1], m)];
            for i in 0..4 {
                for j in (i + 1)..4 {
                    let (pi, pj) = (point(&means[i]), point(&means[j]));
                    let on_map = ((pi[0] - pj[0]).powi(2) + (pi[1] - pj[1]).powi(2)).sqrt();
                    let diff: Vec<f64> = w[i].iter().zip(&w[j]).map(|(a, b)| a - b).collect();
                    let true_dist = dot(&diff, &diff).sqrt();
                    assert!(
                        (on_map - true_dist).abs() <= 1e-9 * true_dist,
                        "{on_map} vs {true_dist}"
                    );
                }
            }
        }

        // K = 2 from a real training: e1 is parallel to the whitened mean difference, e2 is a
        // unit vector orthogonal to it.
        let units = Units::random(&mut rng, 38);
        let means = random_means(&mut rng, 2, 38, 3.0);
        let mut set = make_set(&mut rng, &units, MONO, &means, 8, 4);
        let spec = NegativeSpec {
            centre: &means[0],
            clips: 5,
            variants: 2,
            clip_sd: 2.0,
            variant_sd: 0.5,
        };
        set.negatives = negatives(&mut rng, &units, MONO, spec);
        let model = train_ok(&set);
        let l = &model.within.factor;
        let whiten = |m: &[f64]| {
            let mut w = vec![0.0; m.len()];
            l.forward_solve(m, &mut w);
            w
        };
        let (w0, w1) = (whiten(&model.zone_means[0]), whiten(&model.zone_means[1]));
        let mut u: Vec<f64> = w1.iter().zip(&w0).map(|(a, b)| a - b).collect();
        unit(&mut u);
        let e1 = lower_t_mul(l, &model.map.axes[0]);
        let e2 = lower_t_mul(l, &model.map.axes[1]);
        assert!((dot(&e1, &u).abs() - 1.0).abs() <= 1e-9);
        assert!(dot(&e2, &e1).abs() <= 1e-9);
        assert!((dot(&e2, &e2) - 1.0).abs() <= 1e-9);

        // Stored points are the live map points of the same descriptors, and training is
        // deterministic.
        let mut scratch = model.scratch();
        assert_eq!(model.map.taps.len(), set.taps.len());
        for (t, point) in set.taps.iter().zip(&model.map.taps) {
            assert_eq!(point.zone, t.zone);
            let live = model.map_point(&t.variants[0], &mut scratch).expect("maps");
            assert_eq!(live, point.at);
        }
        assert_eq!(model.map.background.len(), 5);
        for (n, at) in set.negatives.iter().zip(&model.map.background) {
            let live = model.map_point(&n.variants[0], &mut scratch).expect("maps");
            assert_eq!(&live, at);
        }
        assert_eq!(train_ok(&set).map, model.map);

        // validate() rejects a map zone ≥ K and an axis of the wrong length.
        let mut bad = model.clone();
        bad.map.taps[3].zone = ZoneIndex(2);
        assert_eq!(
            bad.validate(),
            Err(ModelError::Invalid {
                reason: ModelInvalid::MapZoneOutOfRange { zone: 2 }
            })
        );
        let mut bad = model.clone();
        bad.map.axes[1].push(0.0);
        assert_eq!(
            bad.validate(),
            Err(ModelError::Invalid {
                reason: ModelInvalid::Dims {
                    expected: 38,
                    got: 39
                }
            })
        );
    }
}
