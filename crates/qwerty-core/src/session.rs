//! Calibration session v2: the guided ambient, zone and negatives steps that produce a trained model (§1.14).

use serde::{Deserialize, Serialize};

/// Pre-placed by WP-00; WP-B2 MUST NOT change it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EnvironmentQuality {
    Quiet,
    Moderate,
    TooNoisy,
}

// ---- Session logic (§1.14, WP-B2). The pre-placed type above belongs to WP-00. ----

use crate::audio::{AudioBlock, AudioFormat, ChannelLayout, LayoutError};
use crate::augment::{AmbientRecording, AugmentError, Augmenter};
use crate::descriptor::{Descriptor, DescriptorError};
use crate::detector::{
    estimate_noise, Candidate, DetectorError, DetectorParams, InputFault, NoiseSnapshot, RawClip,
    RejectReason, RejectedCandidate,
};
use crate::gesture::GestureParams;
use crate::model::TrainedModel;
use crate::profile::ZoneId;
use crate::training::{
    self, NegativeGroup, NegativeSource, TapGroup, TrainingError, TrainingParams, TrainingProgress,
    TrainingSet, TrainingSummary, Workers, MAX_WORKERS, MIN_TAPS_PER_ZONE, MIN_ZONES,
};
use crate::units::{
    Decibels, HostDuration, HostTime, Milliseconds, SampleIndex, TapIndex, ZoneIndex,
};
use crate::veto::{VetoReason, VetoVerdict};
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

pub const DEFAULT_TAPS_PER_ZONE: u32 = 10;
pub const AMBIENT_DURATION: Milliseconds = Milliseconds::new(3000.0);
pub const NEGATIVES_TARGET: u32 = 24;
/// The engine timer fires at `negatives_deadline()`.
pub const NEGATIVES_MAX_DURATION: Milliseconds = Milliseconds::new(30_000.0);
/// OsVetoed negatives kept per session (PARITY-7). Typing gives several onsets per second and would fill all
/// 24 negatives within seconds with sounds the live input veto removes anyway; beyond 8 (a third of the
/// target), input-vetoed candidates give `InputActivity` and are dropped, so the rest must be other sounds.
pub const MAX_INPUT_NEGATIVES: u32 = 8;
/// Room verdict thresholds (rule 1). Public so the UI reads them instead of copying them (PARITY-34).
pub const ROOM_TOO_NOISY_RATE: f64 = 0.5; // disturbances per second
pub const ROOM_TOO_NOISY_FLOOR: Decibels = Decibels::new(-30.0);
pub const ROOM_MODERATE_RATE: f64 = 0.1;
pub const ROOM_MODERATE_FLOOR: Decibels = Decibels::new(-45.0);
pub const DEFAULT_AUDIO_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

/// The ambient noise estimate leaves out [onset − 50 ms, onset + 300 ms) around every ambient
/// candidate (rule 1).
const EXCLUDE_BEFORE: Milliseconds = Milliseconds::new(50.0);
const EXCLUDE_AFTER: Milliseconds = Milliseconds::new(300.0);
/// Floor for the noise-floor `log10` argument (§0.8).
const NOISE_FLOOR_LOG_FLOOR: f64 = 1e-30;
/// Augmentation item of negative clip n: 1_000_000 + n (§1.11.1).
const NEGATIVE_ITEM_BASE: u64 = 1_000_000;

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationPlan {
    pub zone_ids: Vec<ZoneId>,
    pub zone_labels: Vec<String>,
    /// ≥ MIN_TAPS_PER_ZONE
    pub taps_per_zone: u32,
    pub format: AudioFormat,
    pub seed: u64,
    /// DEFAULT_AUDIO_BUDGET_BYTES in the app; tests may lower it
    pub audio_budget_bytes: u64,
    /// DetectorParams::DEFAULT in the app
    pub detector: DetectorParams,
    /// TrainingParams::DEFAULT in the app
    pub training: TrainingParams,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CalibrationStep {
    /// Capture is open and the meter runs; every candidate gives `Ignored` and nothing is stored (XR-UI-01, PARITY-6).
    Ready,
    Ambient {
        elapsed: Milliseconds,
        needed: Milliseconds,
    },
    AmbientReview {
        environment: EnvironmentQuality,
        layout: ChannelLayout,
        noise_floor: Decibels,
        disturbances: u32,
        mixing: AmbientMixing,
    },
    Zone {
        index: ZoneIndex,
        accepted: u32,
        target: u32,
    },
    /// ends_at = the step's deadline (XR-UI-15)
    Negatives {
        collected: u32,
        target: u32,
        ends_at: HostTime,
    },
    ReadyToBuild,
    Building {
        progress: BuildProgress,
    },
    Review {
        summary: TrainingSummary,
    },
    Failed {
        error: SessionError,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StepKind {
    Ready,
    Ambient,
    AmbientReview,
    Zone,
    Negatives,
    ReadyToBuild,
    Building,
    Review,
    Failed,
}

impl CalibrationStep {
    fn kind(&self) -> StepKind {
        match self {
            CalibrationStep::Ready => StepKind::Ready,
            CalibrationStep::Ambient { .. } => StepKind::Ambient,
            CalibrationStep::AmbientReview { .. } => StepKind::AmbientReview,
            CalibrationStep::Zone { .. } => StepKind::Zone,
            CalibrationStep::Negatives { .. } => StepKind::Negatives,
            CalibrationStep::ReadyToBuild => StepKind::ReadyToBuild,
            CalibrationStep::Building { .. } => StepKind::Building,
            CalibrationStep::Review { .. } => StepKind::Review,
            CalibrationStep::Failed { .. } => StepKind::Failed,
        }
    }
}

/// Whether the ambient recording can feed the augmentation (§1.11, DSP-ML-6). `TooDisturbed` blocks
/// `accept_ambient`; only `retry_ambient` proceeds. (TooNoisy stays advisory.)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AmbientMixing {
    Usable,
    TooDisturbed {
        usable: Milliseconds,
        needed: Milliseconds,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TapFeedback {
    Accepted {
        zone: ZoneIndex,
        tap: TapIndex,
        count: u32,
    },
    ZoneFull {
        zone: ZoneIndex,
    },
    TooWeak {
        snr: Decibels,
    },
    NotBroadband {
        fraction: f64,
    },
    NotDecaying,
    NotLocalized,
    Clipped,
    /// OS-vetoed: kept as an OsVetoed negative
    InputActivity,
    SelfOutput,
    Refractory,
    /// a candidate during the ambient step: kept as an EnvironmentOnset negative
    AmbientDisturbance,
    NegativeRecorded {
        count: u32,
    },
    /// superseded / overflow / discontinuity, or a step that takes no taps
    Ignored,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZoneProgress {
    pub zone: ZoneId,
    pub label: String,
    pub accepted: u32,
    pub target: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationView {
    pub step: CalibrationStep,
    pub zones: Vec<ZoneProgress>,
    pub last_feedback: Option<TapFeedback>,
    pub memory_bytes: u64,
    pub layout: Option<ChannelLayout>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum BuildProgress {
    Augmenting { done: u32, total: u32 },
    Training { progress: TrainingProgress },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SessionError {
    NotInStep {
        expected: StepKind,
        actual: StepKind,
    },
    NothingToUndo,
    ZoneIncomplete {
        zone: ZoneIndex,
        accepted: u32,
        need: u32,
    },
    AmbientTooDisturbed {
        usable: Milliseconds,
        needed: Milliseconds,
    },
    BuildCancelled,
    AudioBudgetExceeded {
        bytes: u64,
        budget: u64,
    },
    Layout {
        error: LayoutError,
    },
    Augment {
        error: AugmentError,
    },
    Training {
        error: TrainingError,
    },
    Detector {
        error: DetectorError,
    },
    BuildPanicked {
        message: String,
    },
}

// SPEC: §1.15 places `CalibrationInput` in `pipeline.rs` (WP-B1), which is built in parallel with
// this package, and `on_input` cannot compile without it. Until both merge it is defined here,
// exactly as §1.15 writes it; the merge replaces this definition with
// `use crate::pipeline::CalibrationInput;`.
#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationInput {
    Candidate {
        candidate: Box<Candidate>,
        live: Descriptor,
        veto: VetoVerdict,
    },
    /// extraction failed: feedback `Ignored`, never stored
    Invalid {
        candidate: Box<Candidate>,
        error: DescriptorError,
    },
    Rejected {
        rejected: RejectedCandidate,
    },
    InputFault {
        fault: InputFault,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationResult {
    pub model: TrainedModel,
    pub layout: ChannelLayout,
    pub environment: EnvironmentQuality,
}

/// A stored calibration clip: an accepted tap or a negative. Raw audio lives only here and in a
/// build job, in memory (I4).
#[derive(Debug, Clone)]
struct StoredClip {
    clip: Arc<RawClip>,
    /// the descriptor the live front end computed (an EnvironmentOnset clip's is re-analyzed at build)
    live: Descriptor,
    /// N* at the trigger
    noise: NoiseSnapshot,
}

impl StoredClip {
    fn bytes(&self) -> u64 {
        (self.clip.samples.len() * std::mem::size_of::<f32>()) as u64
    }
}

#[derive(Debug, Clone)]
struct StoredTap {
    zone: ZoneIndex,
    onset: SampleIndex,
    stored: StoredClip,
}

#[derive(Debug, Clone)]
struct StoredNegative {
    source: NegativeSource,
    /// collected by the Negatives step (it counts toward `NEGATIVES_TARGET` and `undo_last`)
    in_negatives_step: bool,
    stored: StoredClip,
}

/// The analyzed ambient recording (rule 1).
#[derive(Debug, Clone)]
struct AmbientAnalysis {
    layout: ChannelLayout,
    environment: EnvironmentQuality,
    /// Some iff mixing is `Usable`
    recording: Option<AmbientRecording>,
}

/// How a candidate's veto verdict sorts it (rule 2 (1)-(2)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VetoClass {
    Clear,
    Input,
    SelfOutput,
}

impl VetoClass {
    // SPEC: rule 2 (2) names "overflow" without saying which; both overflows sort with self-output:
    // the OS-input ring overflow (`InputOverflow`) leaves the input window unknown, as
    // `ClockUnavailable` does, and `DecisionOverflow` is the front end's own queue overflow.
    fn of(verdict: &VetoVerdict) -> VetoClass {
        match verdict {
            VetoVerdict::Clear => VetoClass::Clear,
            VetoVerdict::Vetoed { reason } => match reason {
                VetoReason::Input { .. } => VetoClass::Input,
                VetoReason::SelfOutput { .. }
                | VetoReason::ClockUnavailable
                | VetoReason::InputOverflow
                | VetoReason::DecisionOverflow => VetoClass::SelfOutput,
            },
        }
    }
}

pub struct CalibrationSession {
    plan: CalibrationPlan,
    step: CalibrationStep,
    last_feedback: Option<TapFeedback>,
    /// raw ambient audio while the Ambient step records
    ambient_samples: Vec<f32>,
    ambient_start: Option<SampleIndex>,
    /// onsets of the candidates seen during the Ambient step
    ambient_onsets: Vec<SampleIndex>,
    ambient: Option<AmbientAnalysis>,
    taps: Vec<StoredTap>,
    negatives: Vec<StoredNegative>,
    /// the zone being redone, if any (rule 6)
    redo: Option<ZoneIndex>,
    result: Option<CalibrationResult>,
}

impl CalibrationSession {
    /// Starts in Ready.
    // SPEC: §1.14.1 names no error for a plan the session cannot complete; it is refused with the
    // training error the build would give: fewer than MIN_ZONES zones (or a label list of another
    // length, reported as the zones both lists cover) is `NotEnoughZones`, and fewer than
    // MIN_TAPS_PER_ZONE taps per zone is `NotEnoughTaps` for zone 0.
    pub fn new(plan: CalibrationPlan) -> Result<Self, SessionError> {
        let zones = plan.zone_ids.len().min(plan.zone_labels.len());
        if zones < MIN_ZONES
            || plan.zone_ids.len() != plan.zone_labels.len()
            || plan.zone_ids.len() > usize::from(u16::MAX)
        {
            return Err(SessionError::Training {
                error: TrainingError::NotEnoughZones {
                    have: zones.min(u32::MAX as usize) as u32,
                },
            });
        }
        if (plan.taps_per_zone as usize) < MIN_TAPS_PER_ZONE {
            return Err(SessionError::Training {
                error: TrainingError::NotEnoughTaps {
                    zone: ZoneIndex(0),
                    have: plan.taps_per_zone,
                    need: MIN_TAPS_PER_ZONE as u32,
                },
            });
        }
        Ok(CalibrationSession {
            plan,
            step: CalibrationStep::Ready,
            last_feedback: None,
            ambient_samples: Vec::new(),
            ambient_start: None,
            ambient_onsets: Vec::new(),
            ambient: None,
            taps: Vec::new(),
            negatives: Vec::new(),
            redo: None,
            result: None,
        })
    }

    pub fn view(&self) -> CalibrationView {
        let zones = self
            .plan
            .zone_ids
            .iter()
            .zip(&self.plan.zone_labels)
            .enumerate()
            .map(|(i, (id, label))| ZoneProgress {
                zone: *id,
                label: label.clone(),
                accepted: self.accepted(ZoneIndex(i as u16)),
                target: self.plan.taps_per_zone,
            })
            .collect();
        CalibrationView {
            step: self.step.clone(),
            zones,
            last_feedback: self.last_feedback.clone(),
            memory_bytes: self.memory_bytes(),
            layout: self.ambient.as_ref().map(|a| a.layout.clone()),
        }
    }

    pub fn step_kind(&self) -> StepKind {
        self.step.kind()
    }

    // ---- Ready / Ambient ----

    /// Ready → Ambient.
    pub fn start_ambient(&mut self) -> Result<(), SessionError> {
        self.expect(StepKind::Ready)?;
        self.begin_ambient();
        Ok(())
    }

    pub fn push_ambient(&mut self, block: &AudioBlock<'_>) -> Result<(), SessionError> {
        self.expect(StepKind::Ambient)?;
        let format = self.plan.format;
        if block.format != format {
            return Err(SessionError::Detector {
                error: DetectorError::FormatMismatch {
                    expected: format,
                    got: block.format,
                },
            });
        }
        if self.ambient_start.is_none() {
            self.ambient_start = Some(block.start);
        }
        let channels = usize::from(format.channels.get());
        let needed = format.rate.samples_in(AMBIENT_DURATION) * channels;
        let take = (needed - self.ambient_samples.len()).min(block.samples.len());
        self.ambient_samples
            .extend_from_slice(&block.samples[..take]);
        self.check_budget()?;
        if self.ambient_samples.len() < needed {
            self.step = CalibrationStep::Ambient {
                elapsed: format
                    .rate
                    .duration_of((self.ambient_samples.len() / channels) as u64),
                needed: AMBIENT_DURATION,
            };
            return Ok(());
        }
        match self.analyze_ambient() {
            Ok(step) => {
                self.step = step;
                Ok(())
            }
            Err(error) => Err(self.fail(error)),
        }
    }

    /// AmbientReview → Ambient; the recording and its disturbances are discarded.
    pub fn retry_ambient(&mut self) -> Result<(), SessionError> {
        self.expect(StepKind::AmbientReview)?;
        self.begin_ambient();
        Ok(())
    }

    /// AmbientReview → Zone{0}; Err(AmbientTooDisturbed) if mixing is TooDisturbed.
    pub fn accept_ambient(&mut self) -> Result<ChannelLayout, SessionError> {
        let CalibrationStep::AmbientReview { mixing, .. } = &self.step else {
            return Err(self.not_in(StepKind::AmbientReview));
        };
        match *mixing {
            AmbientMixing::Usable => {}
            AmbientMixing::TooDisturbed { usable, needed } => {
                return Err(SessionError::AmbientTooDisturbed { usable, needed })
            }
        }
        // INVARIANT: the AmbientReview step is only entered with the analysis stored.
        let layout = self
            .ambient
            .as_ref()
            .map(|a| a.layout.clone())
            .expect("AmbientReview holds the ambient analysis");
        self.enter_zone(ZoneIndex(0));
        Ok(layout)
    }

    // ---- Front-end outputs ----

    /// The one entry point for `CalibrationFrontEnd` output (§1.15); the veto is final.
    pub fn on_input(&mut self, input: CalibrationInput) -> Result<TapFeedback, SessionError> {
        let feedback = match &self.step {
            CalibrationStep::Ambient { .. } => self.on_ambient_input(input)?,
            CalibrationStep::Zone { index, .. } => {
                let zone = *index;
                self.on_zone_input(zone, input)?
            }
            CalibrationStep::Negatives { .. } => self.on_negatives_input(input)?,
            CalibrationStep::Ready
            | CalibrationStep::AmbientReview { .. }
            | CalibrationStep::ReadyToBuild
            | CalibrationStep::Building { .. }
            | CalibrationStep::Review { .. }
            | CalibrationStep::Failed { .. } => TapFeedback::Ignored,
        };
        self.last_feedback = Some(feedback.clone());
        Ok(feedback)
    }

    // ---- Navigation ----

    /// Rule 6: may walk back into the previous zone.
    pub fn undo_last(&mut self) -> Result<(), SessionError> {
        let from = match &self.step {
            CalibrationStep::Zone { index, .. } => *index,
            // SPEC: rule 6 covers Negatives with nothing collected only; with negatives collected,
            // undo removes the most recent one, as it removes the most recent tap of a zone.
            CalibrationStep::Negatives { .. } => {
                if let Some(pos) = self.negatives.iter().rposition(|n| n.in_negatives_step) {
                    self.negatives.remove(pos);
                    self.refresh_negatives_step();
                    return Ok(());
                }
                // Nothing collected: back to the last zone, as from an empty zone after it.
                ZoneIndex(self.zone_count() as u16)
            }
            CalibrationStep::Ready
            | CalibrationStep::Ambient { .. }
            | CalibrationStep::AmbientReview { .. }
            | CalibrationStep::ReadyToBuild
            | CalibrationStep::Building { .. }
            | CalibrationStep::Review { .. }
            | CalibrationStep::Failed { .. } => return Err(self.not_in(StepKind::Zone)),
        };
        if from.0 < self.zone_count() as u16 && self.remove_last_tap(from) {
            self.enter_zone(from);
            return Ok(());
        }
        // SPEC: a redo zone (rule 6) is undone within itself only; walking back from it would leave
        // the redo, so an empty redo zone gives NothingToUndo although other zones hold taps.
        if self.redo == Some(from) {
            return Err(SessionError::NothingToUndo);
        }
        for prev in (0..from.0).rev() {
            let zone = ZoneIndex(prev);
            if self.remove_last_tap(zone) {
                self.enter_zone(zone);
                return Ok(());
            }
        }
        Err(SessionError::NothingToUndo)
    }

    pub fn restart_zone(&mut self) -> Result<(), SessionError> {
        let zone = self.current_zone()?;
        self.taps.retain(|t| t.zone != zone);
        self.enter_zone(zone);
        Ok(())
    }

    /// true iff the current zone is full and another zone follows in plan order, and the zone is not a
    /// redo. The engine arms its auto-advance timer only then (XR-UI-02); a full last zone waits for the user.
    pub fn auto_advances(&self) -> bool {
        match &self.step {
            CalibrationStep::Zone { index, .. } => {
                self.zone_full(*index)
                    && usize::from(index.0) + 1 < self.zone_count()
                    && self.redo != Some(*index)
            }
            CalibrationStep::Ready
            | CalibrationStep::Ambient { .. }
            | CalibrationStep::AmbientReview { .. }
            | CalibrationStep::Negatives { .. }
            | CalibrationStep::ReadyToBuild
            | CalibrationStep::Building { .. }
            | CalibrationStep::Review { .. }
            | CalibrationStep::Failed { .. } => false,
        }
    }

    /// Zone{i} full → Zone{i+1} | Negatives{ends_at: now + NEGATIVES_MAX_DURATION} | (after redo) ReadyToBuild
    pub fn advance(&mut self, now: HostTime) -> Result<(), SessionError> {
        let zone = self.current_zone()?;
        self.require_full(zone)?;
        if self.redo == Some(zone) {
            self.redo = None;
            self.step = CalibrationStep::ReadyToBuild;
        } else if usize::from(zone.0) + 1 < self.zone_count() {
            self.enter_zone(ZoneIndex(zone.0 + 1));
        } else {
            let ends_at = HostTime(
                now.0
                    .saturating_add(HostDuration::from_ms(NEGATIVES_MAX_DURATION).0),
            );
            self.step = CalibrationStep::Negatives {
                collected: self.collected_negatives(),
                target: NEGATIVES_TARGET,
                ends_at,
            };
        }
        Ok(())
    }

    /// Some(ends_at) in Negatives; the engine timer calls finish_negatives then.
    pub fn negatives_deadline(&self) -> Option<HostTime> {
        match &self.step {
            CalibrationStep::Negatives { ends_at, .. } => Some(*ends_at),
            CalibrationStep::Ready
            | CalibrationStep::Ambient { .. }
            | CalibrationStep::AmbientReview { .. }
            | CalibrationStep::Zone { .. }
            | CalibrationStep::ReadyToBuild
            | CalibrationStep::Building { .. }
            | CalibrationStep::Review { .. }
            | CalibrationStep::Failed { .. } => None,
        }
    }

    /// Negatives, or a full last zone ("skip negatives") → ReadyToBuild.
    pub fn finish_negatives(&mut self) -> Result<(), SessionError> {
        match &self.step {
            CalibrationStep::Negatives { .. } => {}
            CalibrationStep::Zone { index, .. } => {
                let zone = *index;
                if usize::from(zone.0) + 1 != self.zone_count() || self.redo == Some(zone) {
                    return Err(self.not_in(StepKind::Negatives));
                }
                self.require_full(zone)?;
            }
            CalibrationStep::Ready
            | CalibrationStep::Ambient { .. }
            | CalibrationStep::AmbientReview { .. }
            | CalibrationStep::ReadyToBuild
            | CalibrationStep::Building { .. }
            | CalibrationStep::Review { .. }
            | CalibrationStep::Failed { .. } => return Err(self.not_in(StepKind::Negatives)),
        }
        self.step = CalibrationStep::ReadyToBuild;
        Ok(())
    }

    /// ReadyToBuild|Review → Zone{zone}, cleared.
    // SPEC: §1.14 names no error for a zone outside the plan; it is reported as that zone being
    // incomplete (it has no taps and can never have any).
    pub fn redo_zone(&mut self, zone: ZoneIndex) -> Result<(), SessionError> {
        match self.step.kind() {
            StepKind::ReadyToBuild | StepKind::Review => {}
            StepKind::Ready
            | StepKind::Ambient
            | StepKind::AmbientReview
            | StepKind::Zone
            | StepKind::Negatives
            | StepKind::Building
            | StepKind::Failed => return Err(self.not_in(StepKind::ReadyToBuild)),
        }
        if usize::from(zone.0) >= self.zone_count() {
            return Err(SessionError::ZoneIncomplete {
                zone,
                accepted: 0,
                need: self.plan.taps_per_zone,
            });
        }
        self.result = None;
        self.taps.retain(|t| t.zone != zone);
        self.redo = Some(zone);
        self.enter_zone(zone);
        Ok(())
    }

    // ---- Build ----

    /// ReadyToBuild → Building.
    pub fn take_build_job(&mut self) -> Result<BuildJob, SessionError> {
        self.expect(StepKind::ReadyToBuild)?;
        let ambient = match &self.ambient {
            Some(AmbientAnalysis {
                layout,
                recording: Some(recording),
                ..
            }) => (layout.clone(), recording.clone()),
            // INVARIANT: ReadyToBuild is reached through accept_ambient, which requires a usable
            // recording; nothing clears it afterwards.
            Some(AmbientAnalysis {
                recording: None, ..
            })
            | None => unreachable!("ReadyToBuild without a usable ambient recording"),
        };
        let mut taps = self.taps.clone();
        // Plan order, then acceptance order within a zone (the stable sort keeps it).
        taps.sort_by_key(|t| t.zone);
        let job = BuildJob {
            format: self.plan.format,
            layout: ambient.0,
            ambient: ambient.1,
            detector: self.plan.detector,
            training: self.plan.training.clone(),
            zone_ids: self.plan.zone_ids.clone(),
            seed: self.plan.seed,
            taps: taps.into_iter().map(|t| (t.zone, t.stored)).collect(),
            negatives: self
                .negatives
                .iter()
                .map(|n| (n.source, n.stored.clone()))
                .collect(),
        };
        self.step = CalibrationStep::Building {
            progress: BuildProgress::Augmenting {
                done: 0,
                total: job.items() as u32,
            },
        };
        Ok(job)
    }

    pub fn on_build_progress(&mut self, p: BuildProgress) {
        if let CalibrationStep::Building { progress } = &mut self.step {
            *progress = p;
        }
    }

    /// Building → Review | Failed. Any other step gives NotInStep{expected: Building}: a stale result from a
    /// build that no longer belongs to this step is refused, never applied (RUST-RT-14).
    pub fn finish_build(
        &mut self,
        result: Result<TrainedModel, SessionError>,
    ) -> Result<(), SessionError> {
        self.expect(StepKind::Building)?;
        match result {
            Ok(model) => {
                // INVARIANT: Building is entered by take_build_job, which requires the analysis.
                let ambient = self
                    .ambient
                    .as_ref()
                    .expect("Building holds the ambient analysis");
                self.step = CalibrationStep::Review {
                    summary: model.summary.clone(),
                };
                self.result = Some(CalibrationResult {
                    model,
                    layout: ambient.layout.clone(),
                    environment: ambient.environment,
                });
            }
            Err(error) => self.step = CalibrationStep::Failed { error },
        }
        Ok(())
    }

    /// Review only. The engine builds, validates and saves the profile from this, and calls `commit` only
    /// after the save succeeded; a failed save leaves the session in Review, audio included (PARITY-8, XR-UI-04).
    pub fn result(&self) -> Result<&CalibrationResult, SessionError> {
        self.expect(StepKind::Review)?;
        // INVARIANT: Review is entered only by finish_build(Ok), which stores the result, and
        // redo_zone (the only other place that clears it) leaves Review.
        Ok(self.result.as_ref().expect("Review holds the result"))
    }

    /// Review → result; drops all audio.
    pub fn commit(self) -> Result<CalibrationResult, SessionError> {
        self.expect(StepKind::Review)?;
        // INVARIANT: as in `result`.
        Ok(self.result.expect("Review holds the result"))
    }

    // ---- Internals ----

    fn expect(&self, expected: StepKind) -> Result<(), SessionError> {
        if self.step.kind() == expected {
            Ok(())
        } else {
            Err(self.not_in(expected))
        }
    }

    fn not_in(&self, expected: StepKind) -> SessionError {
        SessionError::NotInStep {
            expected,
            actual: self.step.kind(),
        }
    }

    fn current_zone(&self) -> Result<ZoneIndex, SessionError> {
        let CalibrationStep::Zone { index, .. } = &self.step else {
            return Err(self.not_in(StepKind::Zone));
        };
        Ok(*index)
    }

    fn zone_count(&self) -> usize {
        self.plan.zone_ids.len()
    }

    fn accepted(&self, zone: ZoneIndex) -> u32 {
        self.taps.iter().filter(|t| t.zone == zone).count() as u32
    }

    fn zone_full(&self, zone: ZoneIndex) -> bool {
        self.accepted(zone) >= self.plan.taps_per_zone
    }

    fn require_full(&self, zone: ZoneIndex) -> Result<(), SessionError> {
        let accepted = self.accepted(zone);
        if accepted < self.plan.taps_per_zone {
            return Err(SessionError::ZoneIncomplete {
                zone,
                accepted,
                need: self.plan.taps_per_zone,
            });
        }
        Ok(())
    }

    fn enter_zone(&mut self, zone: ZoneIndex) {
        self.step = CalibrationStep::Zone {
            index: zone,
            accepted: self.accepted(zone),
            target: self.plan.taps_per_zone,
        };
    }

    /// Removes the most recent tap of `zone`; false if it has none.
    fn remove_last_tap(&mut self, zone: ZoneIndex) -> bool {
        match self.taps.iter().rposition(|t| t.zone == zone) {
            Some(pos) => {
                self.taps.remove(pos);
                true
            }
            None => false,
        }
    }

    fn collected_negatives(&self) -> u32 {
        self.negatives
            .iter()
            .filter(|n| n.in_negatives_step)
            .count() as u32
    }

    fn os_vetoed_negatives(&self) -> u32 {
        self.negatives
            .iter()
            .filter(|n| n.source == NegativeSource::OsVetoed)
            .count() as u32
    }

    fn refresh_negatives_step(&mut self) {
        let collected = self.collected_negatives();
        if let CalibrationStep::Negatives { collected: c, .. } = &mut self.step {
            *c = collected;
        }
    }

    fn begin_ambient(&mut self) {
        self.ambient_samples = Vec::new();
        self.ambient_start = None;
        self.ambient_onsets.clear();
        self.ambient = None;
        // The disturbances of a discarded recording go with it.
        self.negatives
            .retain(|n| n.source != NegativeSource::EnvironmentOnset);
        self.step = CalibrationStep::Ambient {
            elapsed: Milliseconds::new(0.0),
            needed: AMBIENT_DURATION,
        };
    }

    /// Rule 1, at AMBIENT_DURATION: layout, exclusions, noise floor, environment and mixing.
    fn analyze_ambient(&mut self) -> Result<CalibrationStep, SessionError> {
        let format = self.plan.format;
        let rate = format.rate;
        let samples = std::mem::take(&mut self.ambient_samples);
        let layout = ChannelLayout::analyze(format, &samples)
            .map_err(|error| SessionError::Layout { error })?;
        let start = match self.ambient_start {
            Some(start) => start.0,
            // INVARIANT: the recording is complete, so at least one block set the start.
            None => unreachable!("a complete ambient recording has a first block"),
        };
        let (before, after) = (
            rate.samples_in(EXCLUDE_BEFORE) as u64,
            rate.samples_in(EXCLUDE_AFTER) as u64,
        );
        let exclude: Vec<(u64, u64)> = self
            .ambient_onsets
            .iter()
            .map(|onset| {
                let at = onset.0.saturating_sub(start);
                (at.saturating_sub(before), at.saturating_add(after))
            })
            .collect();
        let noise = match estimate_noise(format, &layout, &samples, &exclude) {
            Ok(noise) => noise,
            // SPEC: rule 1 does not cover exclusions that leave estimate_noise too few frames. Such
            // a recording also leaves too few mixing offsets (TooDisturbed, below), so the user must
            // record again; the advisory floor is then read from the whole recording.
            Err(DetectorError::InsufficientAmbient { .. }) => {
                estimate_noise(format, &layout, &samples, &[])
                    .map_err(|error| SessionError::Detector { error })?
            }
            Err(
                error @ (DetectorError::FormatMismatch { .. }
                | DetectorError::LayoutChannelOutOfRange { .. }),
            ) => return Err(SessionError::Detector { error }),
        };
        let n_real = layout.real().len() as f64;
        let mean = noise.pooled.iter().sum::<f64>() / noise.pooled.len() as f64;
        let floor = 10.0 * (mean / n_real).max(NOISE_FLOOR_LOG_FLOOR).log10();
        // INVARIANT: the log argument is floored at 1e-30 and the pooled powers are finite.
        let noise_floor = Decibels::try_new(floor).expect("floored log is finite");
        let disturbances = self.ambient_onsets.len() as u32;
        let per_second = f64::from(disturbances) / (AMBIENT_DURATION.get() / 1000.0);
        let environment = classify_environment(per_second, noise_floor);
        let (recording, mixing) = match AmbientRecording::new(format, &layout, samples, &exclude) {
            Ok(recording) => (Some(recording), AmbientMixing::Usable),
            Err(AugmentError::AmbientTooDisturbed {
                usable_frames,
                needed,
            }) => (
                None,
                AmbientMixing::TooDisturbed {
                    usable: rate.duration_of(usable_frames),
                    needed: rate.duration_of(needed),
                },
            ),
            Err(
                error @ (AugmentError::FormatMismatch { .. }
                | AugmentError::AmbientTooShort { .. }
                | AugmentError::Detector { .. }),
            ) => return Err(SessionError::Augment { error }),
        };
        self.ambient = Some(AmbientAnalysis {
            layout: layout.clone(),
            environment,
            recording,
        });
        Ok(CalibrationStep::AmbientReview {
            environment,
            layout,
            noise_floor,
            disturbances,
            mixing,
        })
    }

    fn on_ambient_input(&mut self, input: CalibrationInput) -> Result<TapFeedback, SessionError> {
        match input {
            CalibrationInput::Candidate {
                candidate,
                live,
                veto,
            } => {
                // Every candidate is a disturbance of the recording, whatever its veto.
                self.ambient_onsets.push(candidate.onset);
                // SPEC: rule 1 keeps ambient candidates as EnvironmentOnset negatives; Qwerty's own
                // output (and an untimed or overflowed candidate) is never kept, as in rule 3.
                match VetoClass::of(&veto) {
                    VetoClass::SelfOutput => return Ok(TapFeedback::SelfOutput),
                    VetoClass::Clear | VetoClass::Input => {}
                }
                let Some(stored) = stored_clip(*candidate, live) else {
                    return Ok(TapFeedback::Ignored);
                };
                self.negatives.push(StoredNegative {
                    source: NegativeSource::EnvironmentOnset,
                    in_negatives_step: false,
                    stored,
                });
                self.check_budget()?;
                Ok(TapFeedback::AmbientDisturbance)
            }
            CalibrationInput::Invalid { .. }
            | CalibrationInput::Rejected { .. }
            | CalibrationInput::InputFault { .. } => Ok(TapFeedback::Ignored),
        }
    }

    fn on_zone_input(
        &mut self,
        zone: ZoneIndex,
        input: CalibrationInput,
    ) -> Result<TapFeedback, SessionError> {
        let (candidate, live, veto) = match input {
            CalibrationInput::Candidate {
                candidate,
                live,
                veto,
            } => (candidate, live, veto),
            CalibrationInput::Rejected { rejected } => return Ok(rejection_feedback(&rejected)),
            CalibrationInput::Invalid { .. } | CalibrationInput::InputFault { .. } => {
                return Ok(TapFeedback::Ignored)
            }
        };
        // (1), (2)
        match VetoClass::of(&veto) {
            VetoClass::Input => return self.keep_input_negative(*candidate, live),
            VetoClass::SelfOutput => return Ok(TapFeedback::SelfOutput),
            VetoClass::Clear => {}
        }
        // (3)
        if candidate.gates.clipped {
            return Ok(TapFeedback::Clipped);
        }
        // (4)
        let refractory = self
            .plan
            .format
            .rate
            .samples_in(GestureParams::DEFAULT.refractory) as u64;
        let last = self
            .taps
            .iter()
            .rev()
            .find(|t| t.zone == zone)
            .map(|t| t.onset);
        if let Some(last) = last {
            if candidate.onset.0 < last.0.saturating_add(refractory) {
                return Ok(TapFeedback::Refractory);
            }
        }
        // (5)
        let accepted = self.accepted(zone);
        if accepted >= self.plan.taps_per_zone {
            return Ok(TapFeedback::ZoneFull { zone });
        }
        // (6)
        let onset = candidate.onset;
        let Some(stored) = stored_clip(*candidate, live) else {
            return Ok(TapFeedback::Ignored);
        };
        self.taps.push(StoredTap {
            zone,
            onset,
            stored,
        });
        self.check_budget()?;
        self.enter_zone(zone);
        // SPEC: `tap` is the tap's position within its zone (0-based); `count` is the zone's count.
        Ok(TapFeedback::Accepted {
            zone,
            tap: TapIndex(accepted),
            count: accepted + 1,
        })
    }

    fn on_negatives_input(&mut self, input: CalibrationInput) -> Result<TapFeedback, SessionError> {
        let (candidate, live, veto) = match input {
            CalibrationInput::Candidate {
                candidate,
                live,
                veto,
            } => (candidate, live, veto),
            CalibrationInput::Invalid { .. }
            | CalibrationInput::Rejected { .. }
            | CalibrationInput::InputFault { .. } => return Ok(TapFeedback::Ignored),
        };
        let source = match VetoClass::of(&veto) {
            VetoClass::SelfOutput => return Ok(TapFeedback::SelfOutput),
            VetoClass::Input => {
                if self.os_vetoed_negatives() >= MAX_INPUT_NEGATIVES {
                    return Ok(TapFeedback::InputActivity);
                }
                NegativeSource::OsVetoed
            }
            VetoClass::Clear => NegativeSource::IgnoreStep,
        };
        let Some(stored) = stored_clip(*candidate, live) else {
            return Ok(TapFeedback::Ignored);
        };
        self.negatives.push(StoredNegative {
            source,
            in_negatives_step: true,
            stored,
        });
        self.check_budget()?;
        let count = self.collected_negatives();
        if count >= NEGATIVES_TARGET {
            self.step = CalibrationStep::ReadyToBuild;
        } else {
            self.refresh_negatives_step();
        }
        Ok(TapFeedback::NegativeRecorded { count })
    }

    /// Rule 2 (1): an input-vetoed zone candidate is kept as an OsVetoed negative below the cap.
    fn keep_input_negative(
        &mut self,
        candidate: Candidate,
        live: Descriptor,
    ) -> Result<TapFeedback, SessionError> {
        if self.os_vetoed_negatives() < MAX_INPUT_NEGATIVES {
            if let Some(stored) = stored_clip(candidate, live) {
                self.negatives.push(StoredNegative {
                    source: NegativeSource::OsVetoed,
                    in_negatives_step: false,
                    stored,
                });
                self.check_budget()?;
            }
        }
        Ok(TapFeedback::InputActivity)
    }

    fn memory_bytes(&self) -> u64 {
        let clips: u64 = self
            .taps
            .iter()
            .map(|t| t.stored.bytes())
            .chain(self.negatives.iter().map(|n| n.stored.bytes()))
            .sum();
        let ambient = match &self.ambient {
            Some(AmbientAnalysis {
                recording: Some(recording),
                ..
            }) => recording.samples.len(),
            Some(AmbientAnalysis {
                recording: None, ..
            })
            | None => self.ambient_samples.len(),
        };
        clips + (ambient * std::mem::size_of::<f32>()) as u64
    }

    /// Rule 4: exceeding the budget fails the session.
    fn check_budget(&mut self) -> Result<(), SessionError> {
        let bytes = self.memory_bytes();
        let budget = self.plan.audio_budget_bytes;
        if bytes > budget {
            return Err(self.fail(SessionError::AudioBudgetExceeded { bytes, budget }));
        }
        Ok(())
    }

    fn fail(&mut self, error: SessionError) -> SessionError {
        self.step = CalibrationStep::Failed {
            error: error.clone(),
        };
        error
    }
}

// SPEC: the calibration front end runs its detector in CalibrationClips mode, so a candidate always
// carries its clip; one without (a wiring error) cannot be stored and gives `Ignored`.
fn stored_clip(candidate: Candidate, live: Descriptor) -> Option<StoredClip> {
    let Candidate { clip, analysis, .. } = candidate;
    clip.map(|clip| StoredClip {
        clip: Arc::new(clip),
        live,
        noise: analysis.noise,
    })
}

/// Rule 2's rejection mapping.
// SPEC: LowSnr and InsufficientBandSnr always carry their gate report; without one there is no SNR
// to show, and the rejection gives `Ignored`.
fn rejection_feedback(rejected: &RejectedCandidate) -> TapFeedback {
    match (rejected.reason, &rejected.gates) {
        (RejectReason::LowSnr | RejectReason::InsufficientBandSnr, Some(gates)) => {
            TapFeedback::TooWeak { snr: gates.snr }
        }
        (RejectReason::NotBroadband, Some(gates)) => TapFeedback::NotBroadband {
            fraction: gates.broadband_fraction,
        },
        (RejectReason::NotDecaying, _) => TapFeedback::NotDecaying,
        (RejectReason::OnsetNotLocalized, _) => TapFeedback::NotLocalized,
        (
            RejectReason::LowSnr | RejectReason::InsufficientBandSnr | RejectReason::NotBroadband,
            None,
        )
        | (
            RejectReason::Superseded
            | RejectReason::SpanCrossesDiscontinuity
            | RejectReason::PendingOverflow,
            _,
        ) => TapFeedback::Ignored,
    }
}

/// Rule 1's room verdict from the disturbance rate (per second) and the noise floor.
fn classify_environment(per_second: f64, noise_floor: Decibels) -> EnvironmentQuality {
    if per_second > ROOM_TOO_NOISY_RATE || noise_floor > ROOM_TOO_NOISY_FLOOR {
        EnvironmentQuality::TooNoisy
    } else if per_second > ROOM_MODERATE_RATE || noise_floor > ROOM_MODERATE_FLOOR {
        EnvironmentQuality::Moderate
    } else {
        EnvironmentQuality::Quiet
    }
}

/// The augmentation outcome of one stored clip.
struct Augmented {
    /// variants[0] = live; empty when the clip has no variant 0 (rule 5a)
    variants: Vec<Descriptor>,
    dropped: u32,
}

/// One augmented clip as a helper thread reports it: its index, and its result or panic payload.
type ItemReport = (usize, std::thread::Result<Result<Augmented, SessionError>>);

/// Owns Arc clones of the clips and a copy of the ambient recording. Pure; run on a worker thread.
pub struct BuildJob {
    format: AudioFormat,
    layout: ChannelLayout,
    ambient: AmbientRecording,
    detector: DetectorParams,
    training: TrainingParams,
    zone_ids: Vec<ZoneId>,
    seed: u64,
    /// plan order, then acceptance order
    taps: Vec<(ZoneIndex, StoredClip)>,
    negatives: Vec<(NegativeSource, StoredClip)>,
}

impl BuildJob {
    /// `cancel` is checked between augmented clips and after every fold chunk; once set ⇒ Err(BuildCancelled) (RUST-RT-30).
    pub fn run(
        self,
        cancel: &AtomicBool,
        progress: &mut dyn FnMut(BuildProgress),
    ) -> Result<TrainedModel, SessionError> {
        let total = self.items();
        progress(BuildProgress::Augmenting {
            done: 0,
            total: total as u32,
        });
        let augmented = self.augment_all(cancel, progress)?;
        let layout = crate::descriptor::DescriptorLayout::from_channels(&self.layout);
        let mut dropped = 0u32;
        let mut taps = Vec::with_capacity(self.taps.len());
        let mut negatives = Vec::with_capacity(self.negatives.len());
        for (i, item) in augmented.into_iter().enumerate() {
            dropped += item.dropped;
            if let Some(&(zone, _)) = self.taps.get(i) {
                taps.push(TapGroup {
                    zone,
                    tap: TapIndex(i as u32),
                    variants: item.variants,
                });
            } else if !item.variants.is_empty() {
                negatives.push(NegativeGroup {
                    source: self.negatives[i - self.taps.len()].0,
                    variants: item.variants,
                });
            }
        }
        let set = TrainingSet {
            layout,
            rate: self.format.rate,
            zone_ids: self.zone_ids.clone(),
            seed: self.seed,
            taps,
            negatives,
        };
        let mut model = training::train(&set, &self.training, cancel, &mut |p| {
            progress(BuildProgress::Training { progress: p })
        })
        .map_err(|error| match error {
            TrainingError::Cancelled => SessionError::BuildCancelled,
            error @ (TrainingError::InvalidParam { .. }
            | TrainingError::NotEnoughZones { .. }
            | TrainingError::NotEnoughTaps { .. }
            | TrainingError::EmptyTap { .. }
            | TrainingError::LayoutMismatch { .. }
            | TrainingError::NonFiniteDescriptor { .. }
            | TrainingError::DegenerateWithinZoneSpread { .. }
            | TrainingError::CovarianceNotPositiveDefinite { .. }) => {
                SessionError::Training { error }
            }
        })?;
        model.summary.variants_dropped = dropped;
        Ok(model)
    }

    fn items(&self) -> usize {
        self.taps.len() + self.negatives.len()
    }

    /// Rule 5 (a)-(b) for item i: taps first, then negatives.
    fn augment_item(&self, augmenter: &mut Augmenter, i: usize) -> Result<Augmented, SessionError> {
        let wrap = |error| SessionError::Augment { error };
        let (stored, item, environment) = match self.taps.get(i) {
            Some((_, stored)) => (stored, i as u64, false),
            None => {
                let n = i - self.taps.len();
                let (source, stored) = &self.negatives[n];
                let environment = match source {
                    NegativeSource::EnvironmentOnset => true,
                    NegativeSource::IgnoreStep | NegativeSource::OsVetoed => false,
                };
                (stored, NEGATIVE_ITEM_BASE + n as u64, environment)
            }
        };
        let (live, noise) = if environment {
            // (a) captured under all_real: re-analyzed under the final layout in FrozenNoise mode,
            // jitter 0, with the ambient noise snapshot. That descriptor is variant 0.
            let noise = &self.ambient.band_power;
            match augmenter
                .reanalyze_clip(&stored.clip, noise)
                .map_err(wrap)?
            {
                Ok(live) => (live, noise),
                // SPEC: rule 5 does not say what happens when the re-analysis finds no single
                // candidate: the clip has no variant 0, so it is left out and counts as one
                // dropped variant.
                Err(_) => {
                    return Ok(Augmented {
                        variants: Vec::new(),
                        dropped: 1,
                    })
                }
            }
        } else {
            (stored.live, &stored.noise)
        };
        let outcome = augmenter
            .augment(&live, &stored.clip, noise, &self.ambient, self.seed, item)
            .map_err(wrap)?;
        Ok(Augmented {
            variants: outcome.variants,
            dropped: outcome.dropped.len() as u32,
        })
    }

    /// Rule 5 (b), parallel per clip in chunks of `workers` consecutive clips, slot i holding clip
    /// i (bit-identical for any worker count). Progress is reported and `cancel` checked by the
    /// coordinating thread after every chunk (§1.11.3, §1.10.4).
    fn augment_all(
        &self,
        cancel: &AtomicBool,
        progress: &mut dyn FnMut(BuildProgress),
    ) -> Result<Vec<Augmented>, SessionError> {
        let total = self.items();
        let workers = worker_count(&self.training.workers).clamp(1, total.max(1));
        let new_augmenter = || {
            Augmenter::new(self.format, &self.layout, &self.detector)
                .map_err(|error| SessionError::Augment { error })
        };
        let mut own = new_augmenter()?;
        std::thread::scope(|s| {
            let (report_tx, report_rx) = mpsc::channel::<ItemReport>();
            let helpers: Vec<mpsc::Sender<usize>> = (1..workers)
                .map(|_| {
                    let (job_tx, job_rx) = mpsc::channel::<usize>();
                    let report_tx = report_tx.clone();
                    s.spawn(move || {
                        // INVARIANT: the coordinator built an Augmenter from the same arguments.
                        let mut augmenter = new_augmenter().expect("the arguments were checked");
                        for i in job_rx {
                            let result = panic::catch_unwind(AssertUnwindSafe(|| {
                                self.augment_item(&mut augmenter, i)
                            }));
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
                    // INVARIANT: a helper runs until its job channel closes (`helpers` lives to
                    // the end of this closure) and catches a clip's panic, so it is listening.
                    job.send(i)
                        .expect("an augmentation helper runs until its channel closes");
                }
                let mut chunk: Vec<Option<Result<Augmented, SessionError>>> =
                    (start..end).map(|_| None).collect();
                chunk[0] = Some(self.augment_item(&mut own, start));
                for _ in start + 1..end {
                    // INVARIANT: every helper holds a report sender until its job channel closes,
                    // and answers each job exactly once.
                    let (i, report) = report_rx
                        .recv()
                        .expect("an augmentation helper answers every job");
                    match report {
                        Ok(result) => chunk[i - start] = Some(result),
                        Err(payload) => panic::resume_unwind(payload),
                    }
                }
                for result in chunk {
                    // INVARIANT: the loop above filled one slot per clip of the chunk.
                    out.push(result.expect("every clip of the chunk reported")?);
                }
                progress(BuildProgress::Augmenting {
                    done: end as u32,
                    total: total as u32,
                });
                if cancel.load(Ordering::Relaxed) {
                    return Err(SessionError::BuildCancelled);
                }
                start = end;
            }
            Ok(out)
        })
    }
}

/// `Workers` as `training::train` resolves it: the count only sets the speed.
fn worker_count(workers: &Workers) -> usize {
    match workers {
        Workers::Exactly { n } => usize::from(n.get()),
        Workers::Auto => {
            // Every count gives bit-identical results, so a platform that cannot report its
            // parallelism gets one worker.
            let available = match std::thread::available_parallelism() {
                Ok(n) => n.get(),
                Err(_) => 1,
            };
            available.min(MAX_WORKERS)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::BlockTime;
    use crate::augment::AMBIENT_MIN_USABLE;
    use crate::descriptor::{DescriptorExtractor, DescriptorLayout, MONO_DIMS};
    use crate::desk::{DeskConfig, Distractor, ModalDesk, TapSpec};
    use crate::detector::{Detector, DetectorMode, DetectorOutput};
    use crate::rng::Rng64;
    use crate::units::{ChannelCount, SampleRate};
    use crate::veto::{InputEvent, InputKind, VetoDecision, VetoParams, VetoTimeline};
    use std::num::NonZeroU8;
    use uuid::Uuid;

    const HZ: u32 = 48_000;
    const BLOCK: usize = 480;
    /// Zone taps peak about 29 dB below full scale, about 50 dB over the desk floor.
    const TAP_FORCE: f64 = -30.0;

    fn rate() -> SampleRate {
        SampleRate::new(HZ).expect("rate")
    }

    fn stereo() -> AudioFormat {
        AudioFormat {
            rate: rate(),
            channels: ChannelCount::new(2).expect("2 channels"),
        }
    }

    fn frames(ms: f64) -> usize {
        rate().samples_in(Milliseconds::new(ms))
    }

    fn plan(zones: usize, taps: u32, seed: u64) -> CalibrationPlan {
        CalibrationPlan {
            zone_ids: (0..zones)
                .map(|z| ZoneId::from_uuid(Uuid::from_u128(0x5E55_0000 + z as u128)))
                .collect(),
            zone_labels: (0..zones).map(|z| format!("Zone {z}")).collect(),
            taps_per_zone: taps,
            format: stereo(),
            seed,
            audio_budget_bytes: DEFAULT_AUDIO_BUDGET_BYTES,
            detector: DetectorParams::DEFAULT,
            training: TrainingParams::DEFAULT,
        }
    }

    /// One stored clip at 48 kHz stereo: [t0 − 150 ms, t0 + 60 ms) × 2 channels × 4 B.
    fn clip_bytes() -> u64 {
        ((frames(150.0) + frames(60.0)) * 2 * 4) as u64
    }

    fn ambient_bytes() -> u64 {
        (frames(3000.0) * 2 * 4) as u64
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Mics {
        /// the desk's two mics
        Stereo,
        /// the desk's first mic on both channels
        FakeStereo,
    }

    fn front_end(layout: &ChannelLayout) -> (Detector, DescriptorExtractor) {
        let detector = Detector::new(
            stereo(),
            layout,
            &DetectorParams::DEFAULT,
            DetectorMode::CalibrationClips,
        )
        .expect("detector");
        let extractor = DescriptorExtractor::new(rate(), DescriptorLayout::from_channels(layout));
        (detector, extractor)
    }

    /// A stand-in for the calibration front end (§1.15, WP-B1): a desk streamed in timestamped
    /// blocks through Detector(CalibrationClips) and the extractor, every veto `Clear`.
    struct Rig {
        desk: ModalDesk,
        mics: Mics,
        rng: Rng64,
        detector: Detector,
        extractor: DescriptorExtractor,
        pos: u64,
    }

    impl Rig {
        fn new(seed: u64, mics: Mics) -> Rig {
            let mut config = DeskConfig::default_for(rate(), seed);
            if mics == Mics::FakeStereo {
                config.mics.truncate(1);
            }
            let (detector, extractor) = front_end(&ChannelLayout::all_real(stereo().channels));
            let mut rig = Rig {
                desk: ModalDesk::new(config).expect("desk"),
                mics,
                rng: Rng64::new(seed ^ 0x5E55),
                detector,
                extractor,
                pos: 0,
            };
            rig.play(300.0, &[]);
            rig
        }

        /// Reconfigures the front end for `layout`, as the engine does after accept_ambient.
        fn set_layout(&mut self, layout: &ChannelLayout) {
            (self.detector, self.extractor) = front_end(layout);
            self.play(300.0, &[]);
        }

        fn interleave(&self, x: Vec<f32>) -> Vec<f32> {
            match self.mics {
                Mics::Stereo => x,
                Mics::FakeStereo => x.iter().flat_map(|&v| [v, v]).collect(),
            }
        }

        /// A tap at zone `zone` of 4, its force within ±2 dB of `force_db`, contact 0.8-1.5 ms.
        fn tap(&mut self, zone: usize, force_db: f64) -> Vec<f32> {
            let force = force_db + self.rng.range(-2.0, 2.0);
            let contact = self.rng.range(0.8, 1.5);
            self.tap_exact(zone, force, contact)
        }

        fn tap_exact(&mut self, zone: usize, force: f64, contact: f64) -> Vec<f32> {
            let (x, y) = ModalDesk::zone_centres(4)[zone];
            let spec = TapSpec {
                x,
                y,
                force: Decibels::try_new(force).expect("finite"),
                contact: Milliseconds::try_new(contact).expect("finite"),
                position_jitter_m: 0.02,
            };
            let raw = self.desk.tap(&spec, 0);
            self.interleave(raw)
        }

        fn distractor(&mut self, d: &Distractor) -> Vec<f32> {
            let raw = self.desk.distractor(d, 0);
            self.interleave(raw)
        }

        fn now(&self) -> HostTime {
            time_of(self.pos)
        }

        /// Streams `len_ms` of the floor with each (at ms, interleaved sound) added. Returns the
        /// first frame's index, the raw audio and the front end's inputs.
        fn play(
            &mut self,
            len_ms: f64,
            sounds: &[(f64, Vec<f32>)],
        ) -> (SampleIndex, Vec<f32>, Vec<CalibrationInput>) {
            let floor = self.desk.floor(frames(len_ms));
            let mut x = self.interleave(floor);
            for (at, sound) in sounds {
                for (v, s) in x[frames(*at) * 2..].iter_mut().zip(sound) {
                    *v += s;
                }
            }
            let start = SampleIndex(self.pos);
            let mut outputs = Vec::new();
            for chunk in x.chunks(BLOCK * 2) {
                let time = BlockTime::Timestamped {
                    host_time: time_of(self.pos),
                };
                let block = AudioBlock::new(stereo(), SampleIndex(self.pos), chunk, time, false)
                    .expect("whole frames");
                self.detector.push(&block, &mut outputs).expect("format");
                self.pos += (chunk.len() / 2) as u64;
            }
            let inputs = outputs.into_iter().map(|o| self.input(o)).collect();
            (start, x, inputs)
        }

        fn input(&mut self, output: DetectorOutput) -> CalibrationInput {
            match output {
                DetectorOutput::Candidate(candidate) => {
                    match self.extractor.extract(&candidate.analysis) {
                        Ok(live) => CalibrationInput::Candidate {
                            candidate,
                            live,
                            veto: VetoVerdict::Clear,
                        },
                        Err(error) => CalibrationInput::Invalid { candidate, error },
                    }
                }
                DetectorOutput::Rejected(rejected) => CalibrationInput::Rejected { rejected },
                DetectorOutput::InputFault(fault) => CalibrationInput::InputFault { fault },
            }
        }
    }

    /// The rig's clock: frame s at s / fs seconds.
    fn time_of(s: u64) -> HostTime {
        HostTime((i128::from(s) * i128::from(HostTime::TICKS_PER_SECOND) / i128::from(HZ)) as i64)
    }

    fn feed(s: &mut CalibrationSession, inputs: Vec<CalibrationInput>) -> Vec<TapFeedback> {
        inputs
            .into_iter()
            .map(|i| s.on_input(i).expect("on_input"))
            .collect()
    }

    fn count(feedback: &[TapFeedback], pred: fn(&TapFeedback) -> bool) -> usize {
        feedback.iter().filter(|f| pred(f)).count()
    }

    fn is_accepted(f: &TapFeedback) -> bool {
        matches!(f, TapFeedback::Accepted { .. })
    }

    fn is_candidate(i: &CalibrationInput) -> bool {
        matches!(i, CalibrationInput::Candidate { .. })
    }

    /// The Ambient step: start it, stream 3 s of floor plus `sounds` through the front end,
    /// feed its inputs, then push the same audio as the ambient recording.
    fn record_ambient(
        rig: &mut Rig,
        s: &mut CalibrationSession,
        sounds: &[(f64, Vec<f32>)],
    ) -> Vec<TapFeedback> {
        s.start_ambient().expect("start_ambient");
        let (start, x, inputs) = rig.play(3000.0, sounds);
        let feedback = feed(s, inputs);
        for (i, chunk) in x.chunks(BLOCK * 2).enumerate() {
            let at = SampleIndex(start.0 + (i * BLOCK) as u64);
            let block =
                AudioBlock::new(stereo(), at, chunk, BlockTime::Untimed, false).expect("block");
            s.push_ambient(&block).expect("push_ambient");
        }
        feedback
    }

    /// Accepts the ambient and reconfigures the rig for its layout.
    fn accept(rig: &mut Rig, s: &mut CalibrationSession) -> ChannelLayout {
        let layout = s.accept_ambient().expect("accept_ambient");
        rig.set_layout(&layout);
        layout
    }

    /// `n` zone taps 300 ms apart.
    fn tap_zone(
        rig: &mut Rig,
        s: &mut CalibrationSession,
        zone: usize,
        n: usize,
    ) -> Vec<TapFeedback> {
        let mut feedback = Vec::new();
        for _ in 0..n {
            let tap = rig.tap(zone, TAP_FORCE);
            let (_, _, inputs) = rig.play(300.0, &[(50.0, tap)]);
            feedback.extend(feed(s, inputs));
        }
        feedback
    }

    /// Fills every zone of the plan in order, advancing after each (into Negatives at the end).
    fn fill_zones(rig: &mut Rig, s: &mut CalibrationSession) {
        let (zones, target) = (s.plan.zone_ids.len(), s.plan.taps_per_zone as usize);
        for zone in 0..zones {
            let feedback = tap_zone(rig, s, zone, target);
            assert_eq!(
                count(&feedback, is_accepted),
                target,
                "zone {zone}: {feedback:?}"
            );
            s.advance(rig.now()).expect("advance");
        }
    }

    const NEGATIVE_SOUNDS: [Distractor; 4] = [
        Distractor::Knock {
            x: 0.30,
            y: 0.40,
            count: 1,
        },
        Distractor::ObjectSetDown { x: 0.80, y: 0.45 },
        Distractor::Knock {
            x: 0.70,
            y: 0.20,
            count: 1,
        },
        Distractor::ObjectSetDown { x: 0.40, y: 0.15 },
    ];

    /// Plays negative sounds until the step leaves Negatives or `max` sounds were played; the
    /// veto of every candidate is `veto`. Returns the feedbacks.
    fn play_negatives(
        rig: &mut Rig,
        s: &mut CalibrationSession,
        max: usize,
        veto: VetoVerdict,
    ) -> Vec<TapFeedback> {
        let mut feedback = Vec::new();
        for i in 0..max {
            if s.step_kind() != StepKind::Negatives {
                break;
            }
            let sound = rig.distractor(&NEGATIVE_SOUNDS[i % NEGATIVE_SOUNDS.len()]);
            let (_, _, inputs) = rig.play(700.0, &[(50.0, sound)]);
            for input in inputs {
                feedback.push(s.on_input(with_veto(input, veto)).expect("on_input"));
            }
        }
        feedback
    }

    /// The input with `veto` as a candidate's verdict.
    fn with_veto(input: CalibrationInput, veto: VetoVerdict) -> CalibrationInput {
        match input {
            CalibrationInput::Candidate {
                candidate, live, ..
            } => CalibrationInput::Candidate {
                candidate,
                live,
                veto,
            },
            input @ (CalibrationInput::Invalid { .. }
            | CalibrationInput::Rejected { .. }
            | CalibrationInput::InputFault { .. }) => input,
        }
    }

    /// A session at ReadyToBuild: a clean ambient (plus `ambient_sounds`), every zone full and
    /// up to `negatives` negative sounds.
    fn ready_to_build(
        rig: &mut Rig,
        plan: CalibrationPlan,
        ambient_sounds: &[(f64, Vec<f32>)],
        negatives: usize,
    ) -> CalibrationSession {
        let mut s = CalibrationSession::new(plan).expect("session");
        record_ambient(rig, &mut s, ambient_sounds);
        accept(rig, &mut s);
        fill_zones(rig, &mut s);
        play_negatives(rig, &mut s, negatives, VetoVerdict::Clear);
        if s.step_kind() == StepKind::Negatives {
            s.finish_negatives().expect("finish_negatives");
        }
        assert_eq!(s.step_kind(), StepKind::ReadyToBuild);
        s
    }

    fn build(s: &mut CalibrationSession) -> (TrainedModel, Vec<BuildProgress>) {
        let job = s.take_build_job().expect("take_build_job");
        let mut progress = Vec::new();
        let model = job
            .run(&AtomicBool::new(false), &mut |p| progress.push(p))
            .expect("the model trains");
        (model, progress)
    }

    /// The largest `NegativeRecorded` count in `feedback` (0 if none).
    fn collected(feedback: &[TapFeedback]) -> u32 {
        feedback.iter().fold(0, |m, f| {
            if let TapFeedback::NegativeRecorded { count } = f {
                m.max(*count)
            } else {
                m
            }
        })
    }

    /// A 220 Hz harmonic tone (59 partials at 1/k) switched on abruptly and held for 400 ms.
    fn tone_onset() -> Vec<f32> {
        (0..frames(400.0))
            .flat_map(|i| {
                let t = i as f64 / f64::from(HZ);
                let v: f64 = (1..60)
                    .map(|k| (std::f64::consts::TAU * 220.0 * k as f64 * t).sin() / k as f64)
                    .sum();
                let v = (0.05 * v) as f32;
                [v, v]
            })
            .collect()
    }

    fn input_veto(offset_ms: f64) -> VetoVerdict {
        VetoVerdict::Vetoed {
            reason: VetoReason::Input {
                kind: InputKind::KeyDown,
                offset: Milliseconds::new(offset_ms),
            },
        }
    }

    #[test]
    fn s1_full_flow_counts_memory_and_training() {
        let p = plan(4, 10, 0x51);
        let mut rig = Rig::new(0x5101, Mics::Stereo);
        let mut s = CalibrationSession::new(p.clone()).expect("session");
        assert_eq!(s.step_kind(), StepKind::Ready);

        // Ready: candidates give Ignored and change no count.
        let tap = rig.tap(0, TAP_FORCE);
        let (_, _, inputs) = rig.play(400.0, &[(50.0, tap)]);
        assert!(inputs.iter().any(is_candidate));
        let before = s.view();
        assert!(feed(&mut s, inputs)
            .iter()
            .all(|f| *f == TapFeedback::Ignored));
        let after = s.view();
        assert_eq!(after.step, CalibrationStep::Ready);
        assert_eq!(after.zones, before.zones);
        assert_eq!(after.memory_bytes, 0);
        assert!(after
            .zones
            .iter()
            .all(|z| z.accepted == 0 && z.target == 10));

        // Ambient, with one knock: an EnvironmentOnset negative.
        let knock = rig.distractor(&NEGATIVE_SOUNDS[0]);
        let feedback = record_ambient(&mut rig, &mut s, &[(1500.0, knock)]);
        let disturbances = count(&feedback, |f| *f == TapFeedback::AmbientDisturbance);
        assert!(disturbances >= 1, "{feedback:?}");
        let v = s.view();
        let CalibrationStep::AmbientReview {
            layout,
            disturbances: d,
            mixing,
            ..
        } = &v.step
        else {
            panic!("not in AmbientReview: {:?}", v.step);
        };
        assert_eq!(*d as usize, disturbances);
        assert_eq!(*mixing, AmbientMixing::Usable);
        assert_eq!(layout.real(), &[0, 1]);
        assert_eq!(v.layout.as_ref(), Some(layout));
        let env = disturbances as u64;
        assert_eq!(v.memory_bytes, ambient_bytes() + env * clip_bytes());
        accept(&mut rig, &mut s);

        // Zones: exact counts after every tap.
        for zone in 0..4u16 {
            for k in 1..=10 {
                let tap = rig.tap(usize::from(zone), TAP_FORCE);
                let (_, _, inputs) = rig.play(300.0, &[(50.0, tap)]);
                let fb = feed(&mut s, inputs);
                assert_eq!(count(&fb, is_accepted), 1, "zone {zone} tap {k}: {fb:?}");
                let v = s.view();
                let step = CalibrationStep::Zone {
                    index: ZoneIndex(zone),
                    accepted: k,
                    target: 10,
                };
                assert_eq!(v.step, step);
                for (z, progress) in v.zones.iter().enumerate() {
                    let want = match (z as u16).cmp(&zone) {
                        std::cmp::Ordering::Less => 10,
                        std::cmp::Ordering::Equal => k,
                        std::cmp::Ordering::Greater => 0,
                    };
                    assert_eq!(progress.accepted, want, "zone {z}");
                    assert_eq!(progress.zone, p.zone_ids[z]);
                }
            }
            s.advance(rig.now()).expect("advance");
        }
        let ends_at = HostTime(rig.now().0 + 300_000_000);
        let step = CalibrationStep::Negatives {
            collected: 0,
            target: NEGATIVES_TARGET,
            ends_at,
        };
        assert_eq!(s.view().step, step);
        assert_eq!(s.negatives_deadline(), Some(ends_at));
        let feedback = play_negatives(&mut rig, &mut s, 60, VetoVerdict::Clear);
        let negatives = collected(&feedback);
        assert!(negatives > 0, "{feedback:?}");
        if s.step_kind() == StepKind::Negatives {
            s.finish_negatives().expect("finish_negatives");
        }
        assert_eq!(s.step_kind(), StepKind::ReadyToBuild);
        let clips = 40 + env + u64::from(negatives);
        assert_eq!(
            s.view().memory_bytes,
            clips * clip_bytes() + ambient_bytes()
        );

        // Build.
        let (model, progress) = build(&mut s);
        let total = clips as u32;
        assert_eq!(
            progress.first(),
            Some(&BuildProgress::Augmenting { done: 0, total })
        );
        assert!(progress.contains(&BuildProgress::Augmenting { done: total, total }));
        assert!(matches!(
            progress.last(),
            Some(BuildProgress::Training { .. })
        ));
        assert_eq!(model.zone_ids, p.zone_ids);
        assert_eq!(model.layout, DescriptorLayout::Stereo { a: 0, c: 1 });
        assert_eq!(model.summary.seed, p.seed);
        assert_eq!(model.summary.zones.iter().map(|z| z.taps).sum::<u32>(), 40);
        let kept: u32 = model.summary.negatives.iter().map(|n| n.clips).sum();
        assert!(kept + model.summary.variants_dropped >= negatives);
        println!(
            "S1: {clips} clips, {} variants dropped, negatives {:?}, quality {:?}",
            model.summary.variants_dropped, model.summary.negatives, model.summary.quality
        );
        let last = progress.last().cloned().expect("progress");
        s.on_build_progress(last.clone());
        assert_eq!(s.view().step, CalibrationStep::Building { progress: last });
        s.finish_build(Ok(model.clone())).expect("finish_build");
        let summary = model.summary.clone();
        assert_eq!(s.view().step, CalibrationStep::Review { summary });
        let result = s.commit().expect("commit");
        assert_eq!(result.model, model);
        assert_eq!(result.layout.real(), &[0, 1]);
    }

    #[test]
    fn s2_feedback_mapping() {
        let mut rig = Rig::new(0x5202, Mics::Stereo);
        let mut s = CalibrationSession::new(plan(2, 5, 0x52)).expect("session");
        record_ambient(&mut rig, &mut s, &[]);
        accept(&mut rig, &mut s);

        let weak = rig.tap_exact(0, -62.0, 1.0);
        let (_, _, inputs) = rig.play(300.0, &[(50.0, weak)]);
        let fb = feed(&mut s, inputs);
        assert!(
            fb.iter().any(|f| matches!(f, TapFeedback::TooWeak { .. })),
            "{fb:?}"
        );

        let mut loud = rig.tap_exact(0, TAP_FORCE, 1.0);
        let peak = loud.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        loud.iter_mut().for_each(|v| *v /= peak);
        let (_, _, inputs) = rig.play(300.0, &[(50.0, loud)]);
        let fb = feed(&mut s, inputs);
        assert!(fb.contains(&TapFeedback::Clipped), "{fb:?}");

        let (_, _, inputs) = rig.play(600.0, &[(50.0, tone_onset())]);
        let fb = feed(&mut s, inputs);
        assert!(fb.contains(&TapFeedback::NotDecaying), "{fb:?}");
        assert_eq!(s.view().zones[0].accepted, 0);

        // The tone's end re-warms the noise tracker (§1.6.3 step 11): let it settle first.
        let (_, _, inputs) = rig.play(1000.0, &[]);
        feed(&mut s, inputs);
        assert_eq!(
            rig.detector.state(),
            crate::detector::DetectorState::Running
        );

        // An input event 10 ms before t0, through the veto timeline.
        let tap = rig.tap(0, TAP_FORCE);
        let (_, _, inputs) = rig.play(300.0, &[(50.0, tap)]);
        let mut veto = VetoTimeline::new(VetoParams::DEFAULT);
        let mut vetoed = 0;
        for input in inputs {
            let CalibrationInput::Candidate { candidate, .. } = &input else {
                s.on_input(input).expect("on_input");
                continue;
            };
            let t0 = candidate.onset_time.expect("timed blocks");
            let ten = HostDuration::from_ms(Milliseconds::new(10.0));
            veto.push_input(InputEvent {
                time: HostTime(t0.0 - ten.0),
                kind: InputKind::KeyDown,
            });
            let now = HostTime(t0.0 + HostTime::TICKS_PER_SECOND);
            let VetoDecision::Final { verdict } = veto.decide(t0, now) else {
                panic!("the decision is final a second later");
            };
            assert!(matches!(
                verdict,
                VetoVerdict::Vetoed {
                    reason: VetoReason::Input { .. }
                }
            ));
            let fb = s.on_input(with_veto(input, verdict)).expect("on_input");
            assert_eq!(fb, TapFeedback::InputActivity);
            vetoed += 1;
        }
        assert_eq!(vetoed, 1);
        assert_eq!(s.view().zones[0].accepted, 0);

        // It is counted as an OsVetoed negative in the summary.
        fill_zones(&mut rig, &mut s);
        s.finish_negatives().expect("finish_negatives");
        let (model, _) = build(&mut s);
        let os = model
            .summary
            .negatives
            .iter()
            .find(|n| n.source == NegativeSource::OsVetoed)
            .map(|n| n.clips);
        assert_eq!(os, Some(1), "{:?}", model.summary.negatives);
    }

    #[test]
    fn s3_undo_restart_and_redo() {
        let mut rig = Rig::new(0x5303, Mics::Stereo);
        let mut s = CalibrationSession::new(plan(3, 5, 0x53)).expect("session");
        assert_eq!(
            s.undo_last(),
            Err(SessionError::NotInStep {
                expected: StepKind::Zone,
                actual: StepKind::Ready
            })
        );
        record_ambient(&mut rig, &mut s, &[]);
        accept(&mut rig, &mut s);
        let accepted = |s: &CalibrationSession| -> Vec<u32> {
            s.view().zones.iter().map(|z| z.accepted).collect()
        };
        let zone = |index: u16, accepted: u32| CalibrationStep::Zone {
            index: ZoneIndex(index),
            accepted,
            target: 5,
        };

        // v1: accept two, undo one.
        tap_zone(&mut rig, &mut s, 0, 2);
        assert_eq!(accepted(&s), [2, 0, 0]);
        s.undo_last().expect("undo");
        assert_eq!(s.view().step, zone(0, 1));
        // restart clears the zone; with no tap anywhere, undo has nothing to do.
        s.restart_zone().expect("restart");
        assert_eq!(s.view().step, zone(0, 0));
        assert_eq!(s.undo_last(), Err(SessionError::NothingToUndo));
        assert_eq!(
            s.advance(rig.now()),
            Err(SessionError::ZoneIncomplete {
                zone: ZoneIndex(0),
                accepted: 0,
                need: 5
            })
        );

        // v1: fill to the target; a further tap is not stored.
        let fb = tap_zone(&mut rig, &mut s, 0, 6);
        assert_eq!(count(&fb, is_accepted), 5);
        assert!(fb.contains(&TapFeedback::ZoneFull { zone: ZoneIndex(0) }));
        assert_eq!(accepted(&s), [5, 0, 0]);
        s.advance(rig.now()).expect("advance");
        tap_zone(&mut rig, &mut s, 1, 5);
        s.advance(rig.now()).expect("advance");
        assert_eq!(s.view().step, zone(2, 0));

        // Cross-zone: zone 1 full and zone 2 empty.
        s.undo_last().expect("undo");
        assert_eq!(s.view().step, zone(1, 4));
        s.undo_last().expect("undo");
        assert_eq!(s.view().step, zone(1, 3));
        assert_eq!(accepted(&s), [5, 3, 0]);
        tap_zone(&mut rig, &mut s, 1, 2);
        s.advance(rig.now()).expect("advance");
        tap_zone(&mut rig, &mut s, 2, 5);
        s.advance(rig.now()).expect("advance");
        assert_eq!(s.step_kind(), StepKind::Negatives);
        // From Negatives with nothing collected: back into the last zone.
        s.undo_last().expect("undo");
        assert_eq!(s.view().step, zone(2, 4));
        tap_zone(&mut rig, &mut s, 2, 1);
        s.finish_negatives().expect("skip negatives");
        assert_eq!(s.step_kind(), StepKind::ReadyToBuild);
        assert_eq!(
            s.restart_zone(),
            Err(SessionError::NotInStep {
                expected: StepKind::Zone,
                actual: StepKind::ReadyToBuild
            })
        );

        // Redo: clears that zone only and returns to ReadyToBuild when it is full again.
        s.redo_zone(ZoneIndex(1)).expect("redo");
        assert_eq!(s.view().step, zone(1, 0));
        assert_eq!(accepted(&s), [5, 0, 5]);
        assert_eq!(s.undo_last(), Err(SessionError::NothingToUndo));
        tap_zone(&mut rig, &mut s, 1, 5);
        s.advance(rig.now()).expect("advance");
        assert_eq!(s.step_kind(), StepKind::ReadyToBuild);
        assert_eq!(accepted(&s), [5, 5, 5]);
        assert_eq!(
            s.redo_zone(ZoneIndex(3)),
            Err(SessionError::ZoneIncomplete {
                zone: ZoneIndex(3),
                accepted: 0,
                need: 5
            })
        );
    }

    #[test]
    fn s4_audio_budget() {
        // The ambient alone (1.15 MB at 48 kHz stereo) exceeds 1 MB.
        let mut rig = Rig::new(0x5404, Mics::Stereo);
        let mut s = CalibrationSession::new(CalibrationPlan {
            audio_budget_bytes: 1 << 20,
            ..plan(2, 5, 0x54)
        })
        .expect("session");
        s.start_ambient().expect("start");
        let (start, x, _) = rig.play(3000.0, &[]);
        let mut failed = None;
        for (i, chunk) in x.chunks(BLOCK * 2).enumerate() {
            let at = SampleIndex(start.0 + (i * BLOCK) as u64);
            let block =
                AudioBlock::new(stereo(), at, chunk, BlockTime::Untimed, false).expect("block");
            if let Err(e) = s.push_ambient(&block) {
                failed = Some(e);
                break;
            }
        }
        let Some(SessionError::AudioBudgetExceeded { bytes, budget }) = failed else {
            panic!("expected AudioBudgetExceeded, got {failed:?}");
        };
        assert_eq!(budget, 1 << 20);
        assert!(bytes > budget);
        assert_eq!(
            s.view().step,
            CalibrationStep::Failed {
                error: SessionError::AudioBudgetExceeded { bytes, budget }
            }
        );

        // Clips count too: room for the ambient and two and a half clips.
        let budget = ambient_bytes() + 5 * clip_bytes() / 2;
        let mut s = CalibrationSession::new(CalibrationPlan {
            audio_budget_bytes: budget,
            ..plan(2, 5, 0x54)
        })
        .expect("session");
        record_ambient(&mut rig, &mut s, &[]);
        accept(&mut rig, &mut s);
        tap_zone(&mut rig, &mut s, 0, 2);
        let tap = rig.tap(0, TAP_FORCE);
        let (_, _, inputs) = rig.play(300.0, &[(50.0, tap)]);
        let results: Vec<_> = inputs.into_iter().map(|i| s.on_input(i)).collect();
        let error = SessionError::AudioBudgetExceeded {
            bytes: ambient_bytes() + 3 * clip_bytes(),
            budget,
        };
        assert!(results.contains(&Err(error.clone())), "{results:?}");
        assert_eq!(s.view().step, CalibrationStep::Failed { error });

        // R11 (p35 §5.2): memory at 192 kHz × 8 channels, 8 zones × 10 taps and 24 negatives.
        memory_at_192k_8ch();
    }

    /// Feeds one real 192 kHz × 8 ch candidate, re-timed 200 ms apart, through a full session
    /// and prints the memory against the default budget.
    fn memory_at_192k_8ch() {
        use crate::detector::testkit::{push_all, scene, TapKind};
        let (hz, ch) = (192_000u32, 8u16);
        let r = SampleRate::new(hz).expect("rate");
        let fmt = AudioFormat {
            rate: r,
            channels: ChannelCount::new(ch).expect("8 channels"),
        };
        let mut s = CalibrationSession::new(CalibrationPlan {
            format: fmt,
            ..plan(8, 10, 0x5480)
        })
        .expect("session");
        s.start_ambient().expect("start");
        let ambient_ms = 3000.0;
        let (noise, _) = scene(hz, ch, 0x5481, ambient_ms, &[]);
        let block =
            AudioBlock::new(fmt, SampleIndex(0), &noise, BlockTime::Untimed, false).expect("block");
        s.push_ambient(&block).expect("push_ambient");
        let layout = s.accept_ambient().expect("accept");
        let (x, _) = scene(hz, ch, 0x5482, 1500.0, &[(1000.0, TapKind::Partials, 40.0)]);
        let mut det = Detector::new(
            fmt,
            &layout,
            &DetectorParams::DEFAULT,
            DetectorMode::CalibrationClips,
        )
        .expect("detector");
        let mut out = Vec::new();
        push_all(&mut det, fmt, &x, 0, 4096, &mut out);
        let mut extractor =
            DescriptorExtractor::new(fmt.rate, DescriptorLayout::from_channels(&layout));
        let template = out
            .into_iter()
            .find_map(|o| match o {
                DetectorOutput::Candidate(c) => Some(c),
                DetectorOutput::Rejected(_) | DetectorOutput::InputFault(_) => None,
            })
            .expect("a candidate");
        let live = extractor.extract(&template.analysis).expect("descriptor");
        let mut next = 0u64;
        let mut feed_copy = |s: &mut CalibrationSession| {
            let mut candidate = template.clone();
            next += u64::from(hz) / 5;
            candidate.onset = SampleIndex(next);
            s.on_input(CalibrationInput::Candidate {
                candidate,
                live,
                veto: VetoVerdict::Clear,
            })
            .expect("on_input")
        };
        for _ in 0..8 {
            for _ in 0..10 {
                assert!(is_accepted(&feed_copy(&mut s)));
            }
            s.advance(HostTime(0)).expect("advance");
        }
        while s.step_kind() == StepKind::Negatives {
            feed_copy(&mut s);
        }
        let clip_frames =
            r.samples_in(Milliseconds::new(150.0)) + r.samples_in(Milliseconds::new(60.0));
        let clip = u64::from(ch) * clip_frames as u64 * 4;
        let ambient = u64::from(hz) * 3 * u64::from(ch) * 4;
        let memory = s.view().memory_bytes;
        assert_eq!(memory, 104 * clip + ambient);
        assert!(memory <= DEFAULT_AUDIO_BUDGET_BYTES);
        let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
        println!(
            "S4/R11 192 kHz x 8 ch: clip {:.2} MiB, ambient {:.2} MiB; 8 x 10 taps + 24 negatives = {:.1} MiB of {:.0} MiB",
            mib(clip),
            mib(ambient),
            mib(memory),
            mib(DEFAULT_AUDIO_BUDGET_BYTES)
        );
        for (channels, taps) in [(2u64, 10u64), (2, 20), (8, 10), (8, 20)] {
            let clip = channels * clip_frames as u64 * 4;
            let total = (8 * taps + 24) * clip + u64::from(hz) * 3 * channels * 4;
            println!(
                "S4/R11 192 kHz x {channels} ch, 8 x {taps} taps + 24 negatives: {:.1} MiB ({})",
                mib(total),
                if total <= DEFAULT_AUDIO_BUDGET_BYTES {
                    "within budget"
                } else {
                    "OVER budget"
                }
            );
        }
    }

    #[test]
    fn s5_fake_stereo_is_mono_throughout() {
        let mut rig = Rig::new(0x5505, Mics::FakeStereo);
        let mut s = CalibrationSession::new(plan(2, 5, 0x55)).expect("session");
        let knock = rig.distractor(&NEGATIVE_SOUNDS[2]);
        let feedback = record_ambient(&mut rig, &mut s, &[(1200.0, knock)]);
        assert!(
            feedback.contains(&TapFeedback::AmbientDisturbance),
            "{feedback:?}"
        );
        let layout = accept(&mut rig, &mut s);
        assert_eq!(layout.real(), &[0]);
        assert_eq!(
            DescriptorLayout::from_channels(&layout),
            DescriptorLayout::Mono { a: 0 }
        );
        for zone in 0..2 {
            for _ in 0..5 {
                let tap = rig.tap(zone, TAP_FORCE);
                let (_, _, inputs) = rig.play(300.0, &[(50.0, tap)]);
                for input in &inputs {
                    if let CalibrationInput::Candidate { live, .. } = input {
                        assert_eq!(live.layout.dims(), MONO_DIMS);
                    }
                }
                assert_eq!(count(&feed(&mut s, inputs), is_accepted), 1);
            }
            s.advance(rig.now()).expect("advance");
        }
        play_negatives(&mut rig, &mut s, 6, VetoVerdict::Clear);
        s.finish_negatives().expect("finish_negatives");
        let job = s.take_build_job().expect("job");
        // The EnvironmentOnset clip was captured as stereo; its re-analysis is mono.
        let mut augmenter =
            Augmenter::new(job.format, &job.layout, &job.detector).expect("augmenter");
        let env = job.taps.len()
            + job
                .negatives
                .iter()
                .position(|n| n.0 == NegativeSource::EnvironmentOnset)
                .expect("an EnvironmentOnset negative");
        let item = job.augment_item(&mut augmenter, env).expect("augment");
        assert!(!item.variants.is_empty());
        assert!(item.variants.iter().all(|d| d.layout.dims() == MONO_DIMS));
        let model = job
            .run(&AtomicBool::new(false), &mut |_| {})
            .expect("the model trains");
        assert_eq!(model.layout, DescriptorLayout::Mono { a: 0 });
        assert_eq!(model.standardization.offset.len(), MONO_DIMS);
        assert!(model.zone_means.iter().all(|m| m.len() == MONO_DIMS));
    }

    #[test]
    fn s6_environment_at_the_thresholds() {
        let quiet_floor = Decibels::new(-60.0);
        for (rate, want) in [
            (ROOM_TOO_NOISY_RATE + 0.01, EnvironmentQuality::TooNoisy),
            (ROOM_TOO_NOISY_RATE - 0.01, EnvironmentQuality::Moderate),
            (ROOM_MODERATE_RATE + 0.01, EnvironmentQuality::Moderate),
            (ROOM_MODERATE_RATE - 0.01, EnvironmentQuality::Quiet),
            (ROOM_TOO_NOISY_RATE, EnvironmentQuality::Moderate),
            (ROOM_MODERATE_RATE, EnvironmentQuality::Quiet),
        ] {
            assert_eq!(classify_environment(rate, quiet_floor), want, "rate {rate}");
        }
        let floor = |db: f64| Decibels::new(db);
        for (db, want) in [
            (
                ROOM_TOO_NOISY_FLOOR.get() + 0.5,
                EnvironmentQuality::TooNoisy,
            ),
            (
                ROOM_TOO_NOISY_FLOOR.get() - 0.5,
                EnvironmentQuality::Moderate,
            ),
            (
                ROOM_MODERATE_FLOOR.get() + 0.5,
                EnvironmentQuality::Moderate,
            ),
            (ROOM_MODERATE_FLOOR.get() - 0.5, EnvironmentQuality::Quiet),
        ] {
            assert_eq!(classify_environment(0.0, floor(db)), want, "floor {db}");
        }
        // Either cause alone raises the verdict.
        assert_eq!(
            classify_environment(ROOM_TOO_NOISY_RATE + 0.01, floor(-90.0)),
            EnvironmentQuality::TooNoisy
        );
        assert_eq!(
            classify_environment(0.0, floor(ROOM_TOO_NOISY_FLOOR.get() + 0.5)),
            EnvironmentQuality::TooNoisy
        );
    }

    #[test]
    fn s7_same_seed_and_inputs_give_a_bit_identical_model() {
        let run = |workers: Workers| {
            let mut rig = Rig::new(0x5707, Mics::Stereo);
            let p = CalibrationPlan {
                training: TrainingParams {
                    workers,
                    ..TrainingParams::DEFAULT
                },
                ..plan(3, 6, 0x57)
            };
            let knock = rig.distractor(&NEGATIVE_SOUNDS[1]);
            let mut s = ready_to_build(&mut rig, p, &[(1000.0, knock)], 8);
            build(&mut s).0
        };
        let a = run(Workers::Auto);
        let b = run(Workers::Exactly {
            n: NonZeroU8::new(1).expect("1"),
        });
        assert_eq!(a, b);
        let json = |m: &TrainedModel| serde_json::to_string(m).expect("serialize");
        assert_eq!(json(&a), json(&b));
    }

    #[test]
    fn s8_a_failed_save_keeps_the_review() {
        let mut rig = Rig::new(0x5808, Mics::Stereo);
        let mut s = ready_to_build(&mut rig, plan(2, 5, 0x58), &[], 4);
        let (model, _) = build(&mut s);
        s.finish_build(Ok(model)).expect("finish_build");
        let mut attempts = 0;
        let mut save = |r: &CalibrationResult| -> Result<(), String> {
            attempts += 1;
            assert!(r.model.validate().is_ok());
            if attempts == 1 {
                Err("disk full".into())
            } else {
                Ok(())
            }
        };
        let first = s.result().expect("result").clone();
        assert!(save(&first).is_err());
        assert_eq!(s.step_kind(), StepKind::Review);
        assert_eq!(s.result().expect("result after a failed save"), &first);
        assert!(s.view().memory_bytes > 0, "the audio is kept for a retry");
        save(s.result().expect("result")).expect("the retry saves");
        let committed = s.commit().expect("commit");
        assert_eq!(committed, first);
    }

    #[test]
    fn s9_typing_cap_in_negatives() {
        let mut rig = Rig::new(0x5909, Mics::Stereo);
        let mut s = CalibrationSession::new(plan(2, 5, 0x59)).expect("session");
        record_ambient(&mut rig, &mut s, &[]);
        accept(&mut rig, &mut s);
        fill_zones(&mut rig, &mut s);
        assert_eq!(s.step_kind(), StepKind::Negatives);
        let mut feedback = Vec::new();
        for _ in 0..200 {
            if feedback.len() >= 40 {
                break;
            }
            let more = play_negatives(&mut rig, &mut s, 1, input_veto(10.0));
            feedback.extend(more.into_iter().filter(|f| *f != TapFeedback::Ignored));
        }
        assert!(feedback.len() >= 40, "{} candidates", feedback.len());
        feedback.truncate(40);
        let max = MAX_INPUT_NEGATIVES as usize;
        let recorded: Vec<TapFeedback> = (1..=MAX_INPUT_NEGATIVES)
            .map(|count| TapFeedback::NegativeRecorded { count })
            .collect();
        assert_eq!(&feedback[..max], &recorded[..]);
        assert!(feedback[max..]
            .iter()
            .all(|f| *f == TapFeedback::InputActivity));
        let CalibrationStep::Negatives {
            collected: kept, ..
        } = s.view().step
        else {
            panic!("left Negatives");
        };
        assert_eq!(kept, MAX_INPUT_NEGATIVES);
        assert_eq!(s.os_vetoed_negatives(), MAX_INPUT_NEGATIVES);
        // Clear candidates are still collected past the cap.
        let fb = play_negatives(&mut rig, &mut s, 4, VetoVerdict::Clear);
        assert!(collected(&fb) > MAX_INPUT_NEGATIVES, "{fb:?}");
    }

    #[test]
    fn s10_auto_advance() {
        let mut rig = Rig::new(0x5A10, Mics::Stereo);
        let mut s = CalibrationSession::new(plan(4, 5, 0x5A)).expect("session");
        record_ambient(&mut rig, &mut s, &[]);
        accept(&mut rig, &mut s);
        assert!(!s.auto_advances());
        tap_zone(&mut rig, &mut s, 0, 4);
        assert!(!s.auto_advances(), "zone 0 is not full yet");
        tap_zone(&mut rig, &mut s, 0, 1);
        assert!(s.auto_advances(), "a full zone 0 of 4");
        s.advance(rig.now()).expect("advance");
        for zone in 1..3 {
            tap_zone(&mut rig, &mut s, zone, 5);
            assert!(s.auto_advances());
            s.advance(rig.now()).expect("advance");
        }
        tap_zone(&mut rig, &mut s, 3, 5);
        assert!(!s.auto_advances(), "the full last zone waits for the user");
        s.finish_negatives().expect("skip negatives");
        assert_eq!(s.step_kind(), StepKind::ReadyToBuild);
        assert_eq!(s.view().memory_bytes, 20 * clip_bytes() + ambient_bytes());
        s.redo_zone(ZoneIndex(1)).expect("redo");
        tap_zone(&mut rig, &mut s, 1, 5);
        assert!(!s.auto_advances(), "a full redo zone");
        assert_eq!(
            s.finish_negatives(),
            Err(SessionError::NotInStep {
                expected: StepKind::Negatives,
                actual: StepKind::Zone
            })
        );
        s.advance(rig.now()).expect("advance");
        assert_eq!(s.step_kind(), StepKind::ReadyToBuild);
        let job = s.take_build_job().expect("job");
        assert!(job.negatives.is_empty());
        assert_eq!(job.taps.len(), 20);
    }

    #[test]
    fn s11_disturbed_ambient_must_be_recorded_again() {
        let mut rig = Rig::new(0x5B11, Mics::Stereo);
        let mut s = CalibrationSession::new(plan(2, 5, 0x5B)).expect("session");
        let sounds: Vec<(f64, Vec<f32>)> = (0..7)
            .map(|k| (200.0 + 400.0 * f64::from(k), rig.tap(2, TAP_FORCE)))
            .collect();
        let feedback = record_ambient(&mut rig, &mut s, &sounds);
        assert!(
            count(&feedback, |f| *f == TapFeedback::AmbientDisturbance) >= 6,
            "{feedback:?}"
        );
        let CalibrationStep::AmbientReview {
            mixing,
            environment,
            ..
        } = s.view().step
        else {
            panic!("not in AmbientReview");
        };
        let AmbientMixing::TooDisturbed { usable, needed } = mixing else {
            panic!("mixing is {mixing:?}");
        };
        assert_eq!(needed, AMBIENT_MIN_USABLE);
        assert!(usable.get() < needed.get());
        assert_eq!(environment, EnvironmentQuality::TooNoisy);
        assert_eq!(
            s.accept_ambient(),
            Err(SessionError::AmbientTooDisturbed { usable, needed })
        );
        assert_eq!(s.step_kind(), StepKind::AmbientReview);
        s.retry_ambient().expect("retry");
        assert_eq!(
            s.view().step,
            CalibrationStep::Ambient {
                elapsed: Milliseconds::new(0.0),
                needed: AMBIENT_DURATION
            }
        );
        assert_eq!(
            s.view().memory_bytes,
            0,
            "the disturbed recording is discarded"
        );

        // A clean recording is usable.
        let (start, x, inputs) = rig.play(3000.0, &[]);
        feed(&mut s, inputs);
        for (i, chunk) in x.chunks(BLOCK * 2).enumerate() {
            let at = SampleIndex(start.0 + (i * BLOCK) as u64);
            let block =
                AudioBlock::new(stereo(), at, chunk, BlockTime::Untimed, false).expect("block");
            s.push_ambient(&block).expect("push_ambient");
        }
        let CalibrationStep::AmbientReview { mixing, .. } = s.view().step else {
            panic!("not in AmbientReview");
        };
        assert_eq!(mixing, AmbientMixing::Usable);
        s.accept_ambient().expect("accept");
    }

    #[test]
    fn s12_finish_build_outside_building_changes_nothing() {
        let mut s = CalibrationSession::new(plan(2, 5, 0x5C)).expect("session");
        let before = s.view();
        assert_eq!(
            s.finish_build(Err(SessionError::BuildCancelled)),
            Err(SessionError::NotInStep {
                expected: StepKind::Building,
                actual: StepKind::Ready
            })
        );
        assert_eq!(s.view(), before);

        let mut rig = Rig::new(0x5C12, Mics::Stereo);
        let mut s = ready_to_build(&mut rig, plan(2, 5, 0x5C), &[], 0);
        let before = s.view();
        assert_eq!(
            s.finish_build(Err(SessionError::BuildCancelled)),
            Err(SessionError::NotInStep {
                expected: StepKind::Building,
                actual: StepKind::ReadyToBuild
            })
        );
        assert_eq!(s.view(), before);

        // A cancelled build fails the step; its late result is refused.
        let job = s.take_build_job().expect("job");
        let result = job.run(&AtomicBool::new(true), &mut |_| {});
        assert_eq!(result, Err(SessionError::BuildCancelled));
        s.finish_build(result).expect("finish_build");
        let failed = s.view();
        assert_eq!(
            failed.step,
            CalibrationStep::Failed {
                error: SessionError::BuildCancelled
            }
        );
        assert_eq!(
            s.finish_build(Err(SessionError::BuildPanicked {
                message: "stale".into()
            })),
            Err(SessionError::NotInStep {
                expected: StepKind::Building,
                actual: StepKind::Failed
            })
        );
        assert_eq!(s.view(), failed);
    }
}
