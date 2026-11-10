//! Trained zone model and its per-candidate decision (§1.9).

use crate::descriptor::{Descriptor, DescriptorLayout};
use crate::linalg::{Cholesky, Shrinkage};
use crate::profile::ZoneId;
use crate::training::TrainingSummary;
use crate::units::{SampleRate, ZoneIndex};
use serde::{Deserialize, Serialize};

pub const MODEL_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainedModel {
    pub schema_version: u32,
    pub front_end_version: u32, // detector::FRONT_END_VERSION at training time
    pub layout: DescriptorLayout,
    pub rate: SampleRate,
    pub zone_ids: Vec<ZoneId>, // class order; K = len ≥ 2
    pub standardization: Standardization,
    pub zone_means: Vec<Vec<f64>>, // K × d
    pub within: GaussianFactor,
    pub background: Background,
    pub thresholds: Thresholds,
    pub summary: TrainingSummary, // §1.10.6
    pub map: SoundMap,            // display only (§1.10.3 step 6, XR-UI-10)
}

/// z = (x − offset)/scale; z_g = 0 for g ∈ excluded_dims (§1.10.3)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Standardization {
    pub offset: Vec<f64>,
    pub scale: Vec<f64>,
    pub floored_dims: Vec<u16>,
    pub excluded_dims: Vec<u16>,
}

impl Standardization {
    /// z = (x − offset)/scale over the first `offset.len()` values, then z_g = 0 for every
    /// excluded dim. The one standardization every caller uses (training, `decide`,
    /// `map_point`), so a stored point and a live point of the same descriptor are identical.
    pub(crate) fn apply(&self, x: &[f32], z: &mut [f64]) {
        for (((zg, &xg), off), sc) in z.iter_mut().zip(x).zip(&self.offset).zip(&self.scale) {
            *zg = (f64::from(xg) - off) / sc;
        }
        for &g in &self.excluded_dims {
            z[usize::from(g)] = 0.0;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GaussianFactor {
    pub factor: Cholesky,
    pub shrinkage: Shrinkage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Background {
    pub regime: BackgroundRegime,
    pub mean: Vec<f64>,
    pub factor: Cholesky,
    pub lambda: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum BackgroundRegime {
    /// ≥ `bg_min_clips` negative clips, each with ≥ 1 surviving variant
    Negatives { clips: u32, vectors: u32 },
    /// broad "any tap on this desk" Gaussian over all positive vectors; visible in the UI
    PooledAll { negative_clips: u32, needed: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Thresholds {
    pub alpha_in: f64,   // 0.02
    pub tau_in: f64,     // squared-Mahalanobis units
    pub tau_bg: f64,     // nats (LLR)
    pub tau_margin: f64, // squared-Mahalanobis units; = 2·ln(odds), default 4.0 ⇒ e² ≈ 7.39:1
}

impl Thresholds {
    /// The first failing gate, in the order OutOfDistribution → Background → Ambiguous;
    /// `None` accepts (§1.9.2 step 6).
    pub(crate) fn first_failure(
        &self,
        d2_min: f64,
        llr: f64,
        margin: f64,
    ) -> Option<ModelRejectReason> {
        if d2_min.is_nan() || d2_min > self.tau_in {
            Some(ModelRejectReason::OutOfDistribution)
        } else if llr.is_nan() || llr < self.tau_bg {
            Some(ModelRejectReason::Background)
        } else if margin.is_nan() || margin < self.tau_margin {
            Some(ModelRejectReason::Ambiguous)
        } else {
            None
        }
    }
}

/// The Diagnostics sound map (XR-UI-10, ui-v2 §4.6): a 2-D view of the calibration. Display only: no decision reads it,
/// and every statement about overlap comes from model distances (`QualityIssue::ZonesConfusable`), never from this view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoundMap {
    pub axes: [Vec<f64>; 2], // len d each, standardized space: a point is [axes[0]·z, axes[1]·z]
    pub taps: Vec<MapPoint>, // the clean variant (v = 0) of every calibration tap, tap order
    pub background: Vec<[f32; 2]>, // the clean variant of every negative clip
}

impl SoundMap {
    /// [axes[0]·z, axes[1]·z], each dot product summed in index order.
    pub(crate) fn project(axes: &[Vec<f64>; 2], z: &[f64]) -> [f32; 2] {
        let dot = |a: &[f64]| a.iter().zip(z).map(|(x, y)| x * y).sum::<f64>() as f32;
        [dot(&axes[0]), dot(&axes[1])]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MapPoint {
    pub zone: ZoneIndex,
    pub at: [f32; 2],
} // f32: a display coordinate, half the JSON

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelScores {
    pub d2: Vec<f64>, // per zone, model order
    pub best: ZoneIndex,
    pub second: ZoneIndex,
    pub d2_min: f64,
    pub margin: f64, // d2[second] − d2[best] ≥ 0
    pub llr_bg: f64,
    pub posterior_best: f64, // diagnostics only: 1/(1 + Σ_{k≠best} exp(−½(d2_k − d2_min)))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ModelDecision {
    Accept {
        zone: ZoneIndex,
        scores: ModelScores,
    },
    Reject {
        reason: ModelRejectReason,
        scores: ModelScores,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ModelRejectReason {
    OutOfDistribution,
    Background,
    Ambiguous,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ModelError {
    LayoutMismatch {
        model: DescriptorLayout,
        descriptor: DescriptorLayout,
    },
    NonFiniteDescriptor {
        index: u16,
    },
    Invalid {
        reason: ModelInvalid,
    }, // from validate() (RUST-RT-25: typed, not a String)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ModelInvalid {
    Dims { expected: u16, got: u16 },
    TooFewZones { zones: u16 },
    NonFiniteParameter,
    FactorSize { expected: u32, got: u32 },
    MapZoneOutOfRange { zone: u16 },
}

/// Per-caller working buffers for `decide` and `map_point`, each of len d.
#[derive(Debug, Clone)]
pub struct ModelScratch {
    z: Vec<f64>,
    solve: Vec<f64>,
}

/// best = argmin (ties to the lowest index), second = argmin over k ≠ best, and their gap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Ranked {
    pub best: usize,
    pub second: usize,
    pub d2_min: f64,
    pub margin: f64,
}

/// §1.9.2 step 3. `d2` has ≥ 2 entries.
pub(crate) fn rank_zones(d2: &[f64]) -> Ranked {
    debug_assert!(d2.len() >= 2);
    let mut best = 0;
    for (k, v) in d2.iter().enumerate() {
        if *v < d2[best] {
            best = k;
        }
    }
    let mut second = if best == 0 { 1 } else { 0 };
    for (k, v) in d2.iter().enumerate() {
        if k != best && *v < d2[second] {
            second = k;
        }
    }
    Ranked {
        best,
        second,
        d2_min: d2[best],
        margin: d2[second] - d2[best],
    }
}

/// §1.9.2 step 5: llr = −½·d2_best − ½·log det Σ_w + ½·d2_bg + ½·log det Σ_bg.
pub(crate) fn log_likelihood_ratio(
    d2_best: f64,
    within_log_det: f64,
    d2_bg: f64,
    background_log_det: f64,
) -> f64 {
    -0.5 * d2_best - 0.5 * within_log_det + 0.5 * d2_bg + 0.5 * background_log_det
}

/// usize → u16 for an error report, saturating (a corrupt file may carry any length).
fn report_u16(n: usize) -> u16 {
    n.min(usize::from(u16::MAX)) as u16
}

/// usize → u32 for an error report, saturating.
fn report_u32(n: usize) -> u32 {
    n.min(u32::MAX as usize) as u32
}

fn invalid(reason: ModelInvalid) -> ModelError {
    ModelError::Invalid { reason }
}

fn check_len(expected: usize, got: usize) -> Result<(), ModelError> {
    if expected == got {
        Ok(())
    } else {
        Err(invalid(ModelInvalid::Dims {
            expected: report_u16(expected),
            got: report_u16(got),
        }))
    }
}

fn check_factor_shape(f: &Cholesky, d: usize) -> Result<(), ModelError> {
    check_len(d, f.dim)?;
    let expected = d * (d + 1) / 2;
    if f.lower.len() != expected {
        return Err(invalid(ModelInvalid::FactorSize {
            expected: report_u32(expected),
            got: report_u32(f.lower.len()),
        }));
    }
    Ok(())
}

fn all_finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// Finite entries, finite jitter and log-det, and a strictly positive diagonal (a zero or
/// negative L_ii would make every d² non-finite).
fn factor_usable(f: &Cholesky) -> bool {
    all_finite(&f.lower)
        && f.jitter.is_finite()
        && f.log_det.is_finite()
        && (0..f.dim).all(|i| f.lower[i * (i + 1) / 2 + i] > 0.0)
}

impl TrainedModel {
    pub fn validate(&self) -> Result<(), ModelError> {
        let d = self.layout.dims();
        let k = self.zone_ids.len();
        if k < 2 {
            return Err(invalid(ModelInvalid::TooFewZones {
                zones: report_u16(k),
            }));
        }
        let st = &self.standardization;
        check_len(d, st.offset.len())?;
        check_len(d, st.scale.len())?;
        for &g in st.excluded_dims.iter().chain(&st.floored_dims) {
            if usize::from(g) >= d {
                // SPEC: §1.9.1 names no variant for a dim index outside the layout; it is the
                // layout's dimension count that it violates, so it is reported as Dims.
                return Err(invalid(ModelInvalid::Dims {
                    expected: report_u16(d),
                    got: g,
                }));
            }
        }
        // SPEC: §1.9.1 names no variant for a zone-mean count ≠ K; reported as Dims{K, count}.
        check_len(k, self.zone_means.len())?;
        for m in &self.zone_means {
            check_len(d, m.len())?;
        }
        check_factor_shape(&self.within.factor, d)?;
        check_len(d, self.background.mean.len())?;
        check_factor_shape(&self.background.factor, d)?;
        for axis in &self.map.axes {
            check_len(d, axis.len())?;
        }
        if let Some(p) = self.map.taps.iter().find(|p| usize::from(p.zone.0) >= k) {
            return Err(invalid(ModelInvalid::MapZoneOutOfRange { zone: p.zone.0 }));
        }

        let th = &self.thresholds;
        let s = &self.summary;
        // SPEC: a non-positive scale is reported as NonFiniteParameter: it makes z non-finite.
        let finite = all_finite(&st.offset)
            && st.scale.iter().all(|v| v.is_finite() && *v > 0.0)
            && self.zone_means.iter().all(|m| all_finite(m))
            && factor_usable(&self.within.factor)
            && self.within.shrinkage.lambda.is_finite()
            && self.within.shrinkage.nu.is_finite()
            && all_finite(&self.background.mean)
            && factor_usable(&self.background.factor)
            && self.background.lambda.is_finite()
            && all_finite(&[th.alpha_in, th.tau_in, th.tau_bg, th.tau_margin])
            && self.map.axes.iter().all(|a| all_finite(a))
            && self
                .map
                .taps
                .iter()
                .all(|p| p.at.iter().all(|v| v.is_finite()))
            && self.map.background.iter().flatten().all(|v| v.is_finite())
            && s.is_finite();
        if !finite {
            return Err(invalid(ModelInvalid::NonFiniteParameter));
        }
        Ok(())
    }

    pub fn scratch(&self) -> ModelScratch {
        let d = self.layout.dims();
        ModelScratch {
            z: vec![0.0; d],
            solve: vec![0.0; d],
        }
    }

    /// Layout and finiteness checks, then z into `scratch.z`.
    fn standardize(&self, d: &Descriptor, scratch: &mut ModelScratch) -> Result<(), ModelError> {
        if d.layout != self.layout {
            return Err(ModelError::LayoutMismatch {
                model: self.layout,
                descriptor: d.layout,
            });
        }
        let x = d.as_slice();
        if let Some(index) = x.iter().position(|v| !v.is_finite()) {
            return Err(ModelError::NonFiniteDescriptor {
                index: report_u16(index),
            });
        }
        self.standardization.apply(x, &mut scratch.z);
        Ok(())
    }

    /// §1.9.2. Allocates only the returned `ModelScores.d2`. Call `validate()` once first.
    pub fn decide(
        &self,
        d: &Descriptor,
        scratch: &mut ModelScratch,
    ) -> Result<ModelDecision, ModelError> {
        if self.zone_means.len() < 2 {
            return Err(invalid(ModelInvalid::TooFewZones {
                zones: report_u16(self.zone_means.len()),
            }));
        }
        self.standardize(d, scratch)?;
        let mut d2 = Vec::with_capacity(self.zone_means.len());
        for mean in &self.zone_means {
            d2.push(
                self.within
                    .factor
                    .mahalanobis_sq(&scratch.z, mean, &mut scratch.solve),
            );
        }
        let d2_bg = self.background.factor.mahalanobis_sq(
            &scratch.z,
            &self.background.mean,
            &mut scratch.solve,
        );
        let r = rank_zones(&d2);
        let llr = log_likelihood_ratio(
            r.d2_min,
            self.within.factor.log_det,
            d2_bg,
            self.background.factor.log_det,
        );
        let others: f64 = d2
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != r.best)
            .map(|(_, v)| (-0.5 * (v - r.d2_min)).exp())
            .sum();
        let best = zone_index(r.best);
        let scores = ModelScores {
            best,
            second: zone_index(r.second),
            d2_min: r.d2_min,
            margin: r.margin,
            llr_bg: llr,
            posterior_best: 1.0 / (1.0 + others),
            d2,
        };
        Ok(
            match self.thresholds.first_failure(r.d2_min, llr, r.margin) {
                None => ModelDecision::Accept { zone: best, scores },
                Some(reason) => ModelDecision::Reject { reason, scores },
            },
        )
    }

    /// Where a live descriptor falls on the sound map (the engine's Diagnostics trace). Standardizes into `scratch`.
    pub fn map_point(
        &self,
        d: &Descriptor,
        scratch: &mut ModelScratch,
    ) -> Result<[f32; 2], ModelError> {
        self.standardize(d, scratch)?;
        Ok(SoundMap::project(&self.map.axes, &scratch.z))
    }
}

/// A zone position as a `ZoneIndex`. K ≤ u16::MAX is guaranteed where models are built
/// (training) and checked by `validate` for loaded ones (every map zone is a u16 < K).
pub(crate) fn zone_index(k: usize) -> ZoneIndex {
    ZoneIndex(report_u16(k))
}
