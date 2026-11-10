//! Offline replay of recordings through the streaming pipeline (§1.18).
//!
//! `run_replay` reads a recording (`recording.rs`) and runs it through the v2 core (calibration
//! front end, calibration session, pipeline and evaluation session) and/or the v1 core, giving a
//! `ReplayReport`. Every sidecar event is applied at its own stream position, never by file order.
//! `render_desk_recording` writes a synthetic recording from `desk::ModalDesk`, so replay runs
//! without hardware (the `desk_render` example wraps it).

use crate::assessment::{
    AssessmentError, ConfusionMatrix, EvaluationEvent, EvaluationReport, EvaluationSession,
    LatencyStats, NegativeClassResult, OverallMetrics, PromptOutcome, PromptSchedule, ReportMeta,
    ScriptedNegative, ScriptedPrompt, ZoneMetrics, ZoneRef,
};
use crate::audio::{
    AudioBlock, AudioFormat, BlockTime, CaptureMode, ChannelLayout, ClockMap, DeviceIdentity,
};
use crate::classifier::ClassifierParams;
use crate::descriptor::{Descriptor, DescriptorError, DescriptorLayout, DESCRIPTOR_DIMS};
use crate::desk::{DeskConfig, DeskError, Distractor, ModalDesk, TapSpec};
use crate::detector::{
    DetectorParams, GateReport, RejectReason, RejectedCandidate, FRONT_END_VERSION,
};
use crate::features::FeatureVector;
use crate::gesture::{GestureParams, Trigger};
use crate::model::{BackgroundRegime, ModelRejectReason, ModelScores, TrainedModel};
use crate::onset::OnsetDetector;
use crate::pipeline::{
    CalibrationFrontEnd, CalibrationInput, DecisionOutcome, Pipeline, PipelineConfig,
    PipelineError, PipelineEvent, TapDecision,
};
use crate::profile::{ProfileId, ZoneId};
use crate::recording::{
    Anchor, Gap, NegativeClass, RecordedBlock, RecordingError, RecordingReader, RecordingSummary,
    RecordingWriter, SegmentKind, Sidecar, SidecarEvent, SidecarHeader, AUDIO_FILE,
};
use crate::rng::Rng64;
use crate::session::{
    self, CalibrationPlan, CalibrationSession, CalibrationStep, SessionError, StepKind,
    DEFAULT_AUDIO_BUDGET_BYTES,
};
use crate::store::{CalibrationState, Profile};
use crate::training::{TrainingParams, TrainingSummary};
use crate::units::{
    CandidateId, Decibels, HostTime, Milliseconds, SampleIndex, Seconds, ZoneIndex,
};
use crate::veto::{
    InputEvent, InputKind, OutputId, SelfOutputEvent, SelfOutputKind, VetoParams, VetoTimeline,
};
use crate::wav::WavReader;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use uuid::Uuid;

pub const REPLAY_SCHEMA_VERSION: u32 = 1;
/// `ReplayOptions::block`'s value in the CLI and the tests.
pub const DEFAULT_REPLAY_BLOCK: Milliseconds = Milliseconds::new(10.0);
/// `live_agreement` pairs a replay decision with a live one whose onsets are this close.
const AGREEMENT_TOLERANCE: Milliseconds = Milliseconds::new(1.0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ReplayEngine {
    V1,
    V2,
} // V1 deleted in WP-D1

/// No seed: calibration replays the recorded one (RUST-RT-13).
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayOptions {
    pub engines: Vec<ReplayEngine>,
    pub detector: DetectorParams,
    pub training: TrainingParams,
    /// DEFAULT_REPLAY_BLOCK (10 ms)
    pub block: Milliseconds,
    pub profile: Option<Profile>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayReport {
    /// REPLAY_SCHEMA_VERSION
    pub schema_version: u32,
    pub recording: String,
    pub format: AudioFormat,
    pub duration: Seconds,
    /// records any override
    pub params: DetectorParams,
    pub results: Vec<EngineReplay>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineReplay {
    pub engine: ReplayEngine,
    /// V2 only
    pub training: Option<TrainingSummary>,
    pub confusion: ConfusionMatrix,
    pub per_zone: Vec<ZoneMetrics>,
    pub overall: OverallMetrics,
    /// false accepts/hour per negative class (EvaluationNegative + Corpus)
    pub negatives: Vec<NegativeClassResult>,
    pub algorithmic_latency: LatencyStats,
    /// V2: replay decisions vs SidecarEvent::Decision
    pub live_agreement: Option<LiveAgreement>,
    pub errors: Vec<ErrorDump>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveAgreement {
    pub live: u32,
    pub matched: u32,
    pub mismatched: u32,
    pub missing: u32,
    pub extra: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorDump {
    pub prompt: u32,
    pub expected: ZoneId,
    pub outcome: PromptOutcome,
    pub onset: Option<SampleIndex>,
    pub descriptor: Option<Descriptor>,
    pub scores: Option<ModelScores>,
    pub v1_features: Option<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ReplayError {
    Recording { error: RecordingError },
    NoCalibrationAndNoProfile,
    NoCalibrationSeed,
    Session { error: SessionError },
    Assessment { error: AssessmentError },
    V1 { detail: String },
    // SPEC: §1.18.3's list has no variant for the pipeline, yet `Pipeline::new` refuses a profile
    // whose model does not fit the recording (rate, layout, front end) and `push_block` can fail;
    // this variant names those errors instead of dressing them as another kind.
    Pipeline { error: PipelineError },
}

impl From<RecordingError> for ReplayError {
    fn from(error: RecordingError) -> Self {
        ReplayError::Recording { error }
    }
}

impl From<SessionError> for ReplayError {
    fn from(error: SessionError) -> Self {
        ReplayError::Session { error }
    }
}

impl From<AssessmentError> for ReplayError {
    fn from(error: AssessmentError) -> Self {
        ReplayError::Assessment { error }
    }
}

impl From<PipelineError> for ReplayError {
    fn from(error: PipelineError) -> Self {
        ReplayError::Pipeline { error }
    }
}

/// Replays the recording in `dir` through each engine of `options.engines`, in that order.
pub fn run_replay(dir: &Path, options: &ReplayOptions) -> Result<ReplayReport, ReplayError> {
    let recording = Recording::open(dir)?;
    let mut results = Vec::with_capacity(options.engines.len());
    for engine in &options.engines {
        results.push(match engine {
            ReplayEngine::V1 => replay_v1(&recording, options)?,
            ReplayEngine::V2 => replay_v2(&recording, options)?,
        });
    }
    let rate = f64::from(recording.format.rate.hz());
    let duration = Seconds::try_new(recording.frames as f64 / rate)
        // INVARIANT: a u64 frame count over a rate of at least 8 kHz is finite.
        .expect("a frame count in seconds is finite");
    Ok(ReplayReport {
        schema_version: REPLAY_SCHEMA_VERSION,
        recording: dir.display().to_string(),
        format: recording.format,
        duration,
        params: options.detector,
        results,
    })
}

// ---- The recording, its events in stream order, and its clock ----

/// A segment: [start, end) of the stream (`end` None: to the end of the recording).
#[derive(Debug, Clone, PartialEq)]
struct Segment {
    start: SampleIndex,
    end: Option<SampleIndex>,
    kind: SegmentKind,
    reconfigured: bool,
}

/// A calibration run's plan (`SidecarEvent::CalibrationStarted`).
#[derive(Debug, Clone, Copy, PartialEq)]
struct CalibrationRun {
    start: SampleIndex,
    seed: u64,
    taps_per_zone: u32,
}

#[derive(Debug, Clone, PartialEq)]
struct RecordedPrompt {
    zone: ZoneId,
    opened_at: HostTime,
    deadline: HostTime,
    reopened: bool,
}

/// An event the veto timeline takes, at its host time.
#[derive(Debug, Clone, Copy, PartialEq)]
enum VetoEvent {
    Input(InputEvent),
    SelfOutput(SelfOutputEvent),
}

impl VetoEvent {
    fn time(&self) -> HostTime {
        match self {
            VetoEvent::Input(e) => e.time,
            VetoEvent::SelfOutput(SelfOutputEvent::Began { at, .. })
            | VetoEvent::SelfOutput(SelfOutputEvent::Ended { at, .. }) => *at,
        }
    }

    fn apply(&self, veto: &mut VetoTimeline) {
        match *self {
            VetoEvent::Input(e) => veto.push_input(e),
            VetoEvent::SelfOutput(e) => veto.self_output(e),
        }
    }
}

/// A recording's sidecar, its events sorted into stream order (§1.18.3): sample-positioned events
/// by sample, host-timed events by host time, ties in file order (every sort is stable).
struct Recording {
    dir: PathBuf,
    format: AudioFormat,
    sidecar: Sidecar,
    segments: Vec<Segment>,
    runs: Vec<CalibrationRun>,
    prompts: Vec<RecordedPrompt>,
    veto_events: Vec<VetoEvent>,
    /// live decisions: onset and accepted zone, by onset
    decisions: Vec<(SampleIndex, Option<ZoneId>)>,
    /// frames in the WAV
    frames: u64,
    /// the stream position just past the last recorded frame
    end: SampleIndex,
}

impl Recording {
    fn open(dir: &Path) -> Result<Recording, ReplayError> {
        let reader = RecordingReader::open(dir)?;
        let format = reader.format();
        let sidecar = reader.sidecar;
        let mut segments = Vec::new();
        let mut runs = Vec::new();
        let mut prompts = Vec::new();
        let mut veto_events = Vec::new();
        let mut decisions = Vec::new();
        for event in &sidecar.events {
            match event {
                SidecarEvent::Segment {
                    start,
                    kind,
                    reconfigured,
                } => segments.push(Segment {
                    start: *start,
                    end: None,
                    kind: kind.clone(),
                    reconfigured: *reconfigured,
                }),
                SidecarEvent::CalibrationStarted {
                    start,
                    seed,
                    taps_per_zone,
                } => runs.push(CalibrationRun {
                    start: *start,
                    seed: *seed,
                    taps_per_zone: *taps_per_zone,
                }),
                SidecarEvent::Prompt {
                    zone,
                    opened_at,
                    deadline,
                    reopened,
                } => prompts.push(RecordedPrompt {
                    zone: *zone,
                    opened_at: *opened_at,
                    deadline: *deadline,
                    reopened: *reopened,
                }),
                SidecarEvent::Input { event } => veto_events.push(VetoEvent::Input(*event)),
                SidecarEvent::SelfOutput { event } => {
                    veto_events.push(VetoEvent::SelfOutput(*event))
                }
                SidecarEvent::Decision {
                    onset,
                    accepted_zone,
                } => decisions.push((*onset, *accepted_zone)),
            }
        }
        segments.sort_by_key(|s| s.start);
        let ends: Vec<Option<SampleIndex>> = segments
            .iter()
            .skip(1)
            .map(|s| Some(s.start))
            .chain(std::iter::once(None))
            .collect();
        for (segment, end) in segments.iter_mut().zip(ends) {
            segment.end = end;
        }
        runs.sort_by_key(|r| r.start);
        prompts.sort_by_key(|p| p.opened_at);
        veto_events.sort_by_key(VetoEvent::time);
        decisions.sort_by_key(|d| d.0);
        let frames = WavReader::open(&dir.join(AUDIO_FILE))
            .map_err(|error| RecordingError::Wav { error })?
            .frames();
        let lost: u64 = sidecar.gaps.iter().map(|g| g.frames_lost).sum();
        Ok(Recording {
            frames,
            end: SampleIndex(frames + lost),
            dir: dir.to_path_buf(),
            format,
            sidecar,
            segments,
            runs,
            prompts,
            veto_events,
            decisions,
        })
    }

    fn clock(&self) -> Clock<'_> {
        Clock {
            format: self.format,
            anchors: &self.sidecar.anchors,
            gaps: &self.sidecar.gaps,
        }
    }

    /// A fresh pass over the audio, in `block` pieces cut at every segment start.
    fn pieces(&self, block: Milliseconds) -> Result<Pieces<'_>, ReplayError> {
        Ok(Pieces {
            reader: RecordingReader::open(&self.dir)?,
            frames: self.format.rate.samples_in(block).max(1),
            cuts: self.segments.iter().map(|s| s.start).collect(),
            queue: VecDeque::new(),
            clock: self.clock(),
        })
    }

    /// The labels the recording gives the zones of its calibration segments from `from` on, by
    /// zone id, in order of first appearance.
    fn calibration_zones(&self, from: SampleIndex) -> Vec<ZoneRef> {
        let mut zones: Vec<ZoneRef> = Vec::new();
        for segment in self.segments.iter().filter(|s| s.start >= from) {
            if let SegmentKind::CalibrationZone { zone, label } = &segment.kind {
                if !zones.iter().any(|z| z.id == *zone) {
                    zones.push(ZoneRef {
                        id: *zone,
                        label: label.clone(),
                    });
                }
            }
        }
        zones
    }
}

/// Sample → host time over the recording's anchors.
#[derive(Debug, Clone, Copy)]
struct Clock<'a> {
    format: AudioFormat,
    anchors: &'a [Anchor],
    gaps: &'a [Gap],
}

impl Clock<'_> {
    /// The host time the recording reader gives a block that starts at `s`: the `ClockMap` mapping
    /// of the newest anchor at or before `s` in `s`'s gap-free run, None where live had none.
    fn mapped(&self, s: SampleIndex) -> Option<HostTime> {
        let run_start = self
            .gaps
            .iter()
            .take_while(|g| g.at <= s)
            .last()
            .map(|g| g.at);
        let anchor = self.anchors[..self.anchors.partition_point(|a| a.sample <= s)].last()?;
        if run_start.is_some_and(|r| anchor.sample < r) {
            return None;
        }
        let mut clock = ClockMap::new(self.format.rate);
        clock.observe(&AudioBlock {
            format: self.format,
            start: anchor.sample,
            samples: &[],
            time: BlockTime::Timestamped {
                host_time: anchor.host_time,
            },
            discontinuity: false,
        });
        clock.host_time_of(s)
    }

    /// Replay's clock at `s`: `mapped(s)`, else extrapolated from the newest anchor before `s`
    /// (or back from the first one), else `s` on a free-running clock from 0.
    // SPEC: §1.18.3 does not say what `now` is for a block live had no timestamp for; live's `now`
    // came from the OS clock, which a recording does not hold, so replay extrapolates the anchors.
    fn at(&self, s: SampleIndex) -> HostTime {
        if let Some(t) = self.mapped(s) {
            return t;
        }
        let before = self.anchors[..self.anchors.partition_point(|a| a.sample <= s)].last();
        match (before, self.anchors.first()) {
            (Some(a), _) => HostTime(a.host_time.0.saturating_add(self.ticks(s.0 - a.sample.0))),
            (None, Some(a)) => HostTime(a.host_time.0.saturating_sub(self.ticks(a.sample.0 - s.0))),
            (None, None) => HostTime(self.ticks(s.0)),
        }
    }

    /// `frames` in host ticks, rounded half up, as `ClockMap` rounds.
    fn ticks(&self, frames: u64) -> i64 {
        let fs = i128::from(self.format.rate.hz());
        let num = i128::from(frames) * i128::from(HostTime::TICKS_PER_SECOND);
        // Saturates: only a corrupt gap or anchor puts a position 29,000 years away.
        ((2 * num + fs) / (2 * fs)).min(i128::from(i64::MAX)) as i64
    }
}

/// The recording's blocks, cut so that every segment starts a block.
struct Pieces<'a> {
    reader: RecordingReader,
    frames: usize,
    cuts: Vec<SampleIndex>,
    queue: VecDeque<RecordedBlock>,
    clock: Clock<'a>,
}

impl Pieces<'_> {
    fn next(&mut self) -> Result<Option<RecordedBlock>, ReplayError> {
        if let Some(piece) = self.queue.pop_front() {
            return Ok(Some(piece));
        }
        let Some(block) = self.reader.next_block(self.frames)? else {
            return Ok(None);
        };
        let channels = self.samples_per_frame();
        let end = block.start.0 + (block.samples.len() / channels) as u64;
        let inner: Vec<u64> = self
            .cuts
            .iter()
            .map(|c| c.0)
            .filter(|&c| block.start.0 < c && c < end)
            .collect();
        if inner.is_empty() {
            return Ok(Some(block));
        }
        let timed = matches!(block.time, BlockTime::Timestamped { .. });
        let mut from = block.start.0;
        for to in inner.into_iter().chain(std::iter::once(end)) {
            let range = (from - block.start.0) as usize * channels
                ..(to - block.start.0) as usize * channels;
            let start = SampleIndex(from);
            let time = if from == block.start.0 {
                block.time
            } else {
                match (timed, self.clock.mapped(start)) {
                    (true, Some(host_time)) => BlockTime::Timestamped { host_time },
                    (true, None) | (false, _) => BlockTime::Untimed,
                }
            };
            self.queue.push_back(RecordedBlock {
                start,
                samples: block.samples[range].to_vec(),
                time,
                discontinuity: block.discontinuity && from == block.start.0,
            });
            from = to;
        }
        Ok(self.queue.pop_front())
    }

    fn samples_per_frame(&self) -> usize {
        usize::from(self.reader.format().channels.get())
    }
}

/// A piece as an `AudioBlock`, with the sample just past its end.
fn audio_block(format: AudioFormat, piece: &RecordedBlock) -> (AudioBlock<'_>, SampleIndex) {
    let block = AudioBlock::new(
        format,
        piece.start,
        &piece.samples,
        piece.time,
        piece.discontinuity,
    )
    // INVARIANT: the reader returns whole frames and pieces are cut on frame boundaries.
    .expect("a recorded piece holds whole frames");
    let end = SampleIndex(piece.start.0 + block.frames() as u64);
    (block, end)
}

/// The recorded veto events not yet applied, in stream order.
struct VetoFeed<'a> {
    events: &'a [VetoEvent],
    next: usize,
}

impl VetoFeed<'_> {
    /// Applies every event due before a block whose last frame's host time is `last`.
    fn apply_until(&mut self, last: HostTime, veto: &mut VetoTimeline) {
        while let Some(e) = self.events.get(self.next) {
            if e.time() > last {
                break;
            }
            e.apply(veto);
            self.next += 1;
        }
    }
}

/// Walks the segments in stream order; `enter` gives the segments that start at or before a
/// position, once each.
struct SegmentCursor<'a> {
    segments: &'a [Segment],
    next: usize,
}

impl<'a> SegmentCursor<'a> {
    fn enter(&mut self, at: SampleIndex) -> &'a [Segment] {
        let from = self.next;
        while self.segments.get(self.next).is_some_and(|s| s.start <= at) {
            self.next += 1;
        }
        &self.segments[from..self.next]
    }
}

fn is_calibration(kind: &SegmentKind) -> bool {
    match kind {
        SegmentKind::Ambient
        | SegmentKind::CalibrationZone { .. }
        | SegmentKind::CalibrationNegatives => true,
        SegmentKind::Idle
        | SegmentKind::Listening
        | SegmentKind::EvaluationPrompts
        | SegmentKind::EvaluationNegative { .. }
        | SegmentKind::Corpus { .. } => false,
    }
}

/// The segments a live pipeline processed: listening and the evaluation's.
fn runs_pipeline(kind: &SegmentKind) -> bool {
    match kind {
        SegmentKind::Listening
        | SegmentKind::EvaluationPrompts
        | SegmentKind::EvaluationNegative { .. }
        | SegmentKind::Corpus { .. } => true,
        SegmentKind::Idle
        | SegmentKind::Ambient
        | SegmentKind::CalibrationZone { .. }
        | SegmentKind::CalibrationNegatives => false,
    }
}

fn negative_class(kind: &SegmentKind) -> Option<&NegativeClass> {
    match kind {
        SegmentKind::EvaluationNegative { class } | SegmentKind::Corpus { class } => Some(class),
        SegmentKind::Idle
        | SegmentKind::Listening
        | SegmentKind::Ambient
        | SegmentKind::CalibrationZone { .. }
        | SegmentKind::CalibrationNegatives
        | SegmentKind::EvaluationPrompts => None,
    }
}

// ---- V2 ----

/// The model replay evaluates with, its channel layout and its zones (model order).
struct Model {
    model: Arc<TrainedModel>,
    layout: ChannelLayout,
    zones: Vec<ZoneRef>,
    profile_id: Option<ProfileId>,
    trained_at: Option<DateTime<Utc>>,
}

fn replay_v2(rec: &Recording, options: &ReplayOptions) -> Result<EngineReplay, ReplayError> {
    let model = match &options.profile {
        Some(Profile {
            id,
            zones,
            calibration:
                CalibrationState::Calibrated {
                    device,
                    model,
                    calibrated_at,
                    ..
                },
            ..
        }) => Model {
            zones: model
                .zone_ids
                .iter()
                .map(|zone| ZoneRef {
                    id: *zone,
                    label: match zones.iter().find(|z| z.id == *zone) {
                        Some(z) => z.label.clone(),
                        None => zone.as_uuid().to_string(),
                    },
                })
                .collect(),
            model: Arc::new(model.clone()),
            layout: device.layout.clone(),
            profile_id: Some(*id),
            trained_at: Some(*calibrated_at),
        },
        Some(Profile {
            calibration: CalibrationState::NeedsRecalibration { .. },
            ..
        })
        | None => calibrate(rec, options)?,
    };
    evaluate_v2(rec, options, &model)
}

/// Step 2 of §1.18.3: calibrates from the recording's last calibration run, driving the session as
/// live did.
fn calibrate(rec: &Recording, options: &ReplayOptions) -> Result<Model, ReplayError> {
    if !rec.segments.iter().any(|s| is_calibration(&s.kind)) {
        return Err(ReplayError::NoCalibrationAndNoProfile);
    }
    let Some(&run) = rec.runs.last() else {
        return Err(ReplayError::NoCalibrationSeed);
    };
    let in_run = |s: &&Segment| s.start >= run.start && is_calibration(&s.kind);
    let Some(last) = rec.segments.iter().rfind(in_run) else {
        return Err(ReplayError::NoCalibrationAndNoProfile);
    };
    let run_end = last.end;
    let zones = rec.calibration_zones(run.start);
    let zone_ids: Vec<ZoneId> = zones.iter().map(|z| z.id).collect();
    let mut session = CalibrationSession::new(CalibrationPlan {
        zone_ids: zone_ids.clone(),
        zone_labels: zones.iter().map(|z| z.label.clone()).collect(),
        taps_per_zone: run.taps_per_zone,
        format: rec.format,
        seed: run.seed,
        audio_budget_bytes: DEFAULT_AUDIO_BUDGET_BYTES,
        detector: options.detector,
        training: options.training.clone(),
    })?;
    let clock = rec.clock();
    let mut pieces = rec.pieces(options.block)?;
    let mut cursor = SegmentCursor {
        segments: &rec.segments,
        next: 0,
    };
    let mut feed = VetoFeed {
        events: &rec.veto_events,
        next: 0,
    };
    let mut veto = VetoTimeline::new(VetoParams::DEFAULT);
    let mut layout: Option<ChannelLayout> = None;
    let mut front_end: Option<CalibrationFrontEnd> = None;
    let mut current: Option<&Segment> = None;
    let mut inputs = Vec::new();
    while let Some(piece) = pieces.next()? {
        if run_end.is_some_and(|end| piece.start >= end) {
            break;
        }
        let (block, end) = audio_block(rec.format, &piece);
        feed.apply_until(clock.at(SampleIndex(end.0 - 1)), &mut veto);
        for segment in cursor.enter(piece.start) {
            let prev = current.map(|s| &s.kind);
            current = Some(segment);
            if segment.start < run.start || !is_calibration(&segment.kind) {
                if segment.reconfigured {
                    front_end = None;
                }
                continue;
            }
            if let Some(accepted) = steer(
                &mut session,
                &segment.kind,
                prev,
                &zone_ids,
                clock.at(segment.start),
            )? {
                layout = Some(accepted);
            }
            if segment.reconfigured || front_end.is_none() {
                let channels = match &layout {
                    Some(l) => l.clone(),
                    None => ChannelLayout::all_real(rec.format.channels),
                };
                front_end = Some(
                    CalibrationFrontEnd::new(rec.format, &channels, &options.detector)
                        .map_err(|error| SessionError::Detector { error })?,
                );
            }
        }
        let Some(segment) = current.filter(|s| s.start >= run.start && is_calibration(&s.kind))
        else {
            continue;
        };
        let Some(fe) = front_end.as_mut() else {
            continue;
        };
        fe.push_block(&block, clock.at(end), &mut veto, &mut inputs)
            .map_err(|error| SessionError::Detector { error })?;
        for input in inputs.drain(..) {
            session.on_input(session_input(input))?;
        }
        let ambient = matches!(segment.kind, SegmentKind::Ambient);
        if ambient && session.step_kind() == StepKind::Ambient {
            session.push_ambient(&block)?;
        }
    }
    match session.step_kind() {
        StepKind::Negatives | StepKind::Zone => session.finish_negatives()?,
        StepKind::Ready
        | StepKind::Ambient
        | StepKind::AmbientReview
        | StepKind::ReadyToBuild
        | StepKind::Building
        | StepKind::Review
        | StepKind::Failed => {}
    }
    let job = session.take_build_job()?;
    let model = job.run(&AtomicBool::new(false), &mut |_| {})?;
    session.finish_build(Ok(model))?;
    let result = session.commit()?;
    Ok(Model {
        model: Arc::new(result.model),
        layout: result.layout,
        zones,
        profile_id: None,
        trained_at: None,
    })
}

/// Moves the session to the step a calibration segment stands for, with the calls live made: the
/// Ambient segment starts (or retries) the ambient; the first zone segment accepts it; a later
/// zone advances to it, walks back to it with undo, redoes it after the build step, or (the same
/// zone again: live re-Configured after `restart_zone`) restarts it; the Negatives segment
/// advances past the last zone. Returns the layout when the ambient is accepted.
// SPEC: §1.18.3 forces the session's current zone to the segment's zone, but the session has no
// call that sets a zone; replay reaches it with the session's own navigation, as live did. A
// session error here means the replayed taps did not fill a zone that live advanced past.
fn steer(
    session: &mut CalibrationSession,
    kind: &SegmentKind,
    prev: Option<&SegmentKind>,
    zone_ids: &[ZoneId],
    now: HostTime,
) -> Result<Option<ChannelLayout>, SessionError> {
    match kind {
        SegmentKind::Ambient => {
            match session.step_kind() {
                StepKind::AmbientReview => session.retry_ambient()?,
                StepKind::Ambient => {}
                StepKind::Ready
                | StepKind::Zone
                | StepKind::Negatives
                | StepKind::ReadyToBuild
                | StepKind::Building
                | StepKind::Review
                | StepKind::Failed => session.start_ambient()?,
            }
            Ok(None)
        }
        SegmentKind::CalibrationZone { zone, .. } => {
            let target = ZoneIndex(
                zone_ids
                    .iter()
                    .position(|id| id == zone)
                    // INVARIANT: the plan's zones are the run's zone segments' zones.
                    .expect("every zone segment's zone is in the plan") as u16,
            );
            let mut accepted = None;
            let restart = prev == Some(kind);
            loop {
                match session.view().step {
                    CalibrationStep::AmbientReview { .. } => {
                        accepted = Some(session.accept_ambient()?);
                    }
                    CalibrationStep::ReadyToBuild | CalibrationStep::Review { .. } => {
                        session.redo_zone(target)?;
                    }
                    CalibrationStep::Zone { index, .. } if index.0 < target.0 => {
                        session.advance(now)?;
                    }
                    CalibrationStep::Zone { index, .. } if index.0 > target.0 => {
                        session.undo_last()?;
                    }
                    CalibrationStep::Zone { .. } => break,
                    CalibrationStep::Negatives { .. } => session.undo_last()?,
                    CalibrationStep::Ready
                    | CalibrationStep::Ambient { .. }
                    | CalibrationStep::Building { .. }
                    | CalibrationStep::Failed { .. } => {
                        // accept_ambient names the step the session is in.
                        session.accept_ambient()?;
                    }
                }
            }
            if restart {
                session.restart_zone()?;
            }
            Ok(accepted)
        }
        SegmentKind::CalibrationNegatives => {
            match session.step_kind() {
                StepKind::Negatives => {}
                StepKind::Ready
                | StepKind::Ambient
                | StepKind::AmbientReview
                | StepKind::Zone
                | StepKind::ReadyToBuild
                | StepKind::Building
                | StepKind::Review
                | StepKind::Failed => session.advance(now)?,
            }
            Ok(None)
        }
        SegmentKind::Idle
        | SegmentKind::Listening
        | SegmentKind::EvaluationPrompts
        | SegmentKind::EvaluationNegative { .. }
        | SegmentKind::Corpus { .. } => Ok(None),
    }
}

// SPEC: §1.15 places `CalibrationInput` in pipeline.rs and the session takes it, but the merged
// session.rs still holds its own copy of the type (its WP-B2 note says the merge would replace it);
// replay converts the front end's input to the session's, variant for variant.
fn session_input(input: CalibrationInput) -> session::CalibrationInput {
    match input {
        CalibrationInput::Candidate {
            candidate,
            live,
            veto,
        } => session::CalibrationInput::Candidate {
            candidate,
            live,
            veto,
        },
        CalibrationInput::Invalid { candidate, error } => {
            session::CalibrationInput::Invalid { candidate, error }
        }
        CalibrationInput::Rejected { rejected } => session::CalibrationInput::Rejected { rejected },
        CalibrationInput::InputFault { fault } => session::CalibrationInput::InputFault { fault },
    }
}

/// Step 4 of §1.18.3: the pipeline over every segment live ran one in, scored by an
/// `EvaluationSession` on the recorded prompts and negative phases.
fn evaluate_v2(
    rec: &Recording,
    options: &ReplayOptions,
    model: &Model,
) -> Result<EngineReplay, ReplayError> {
    let clock = rec.clock();
    let mut assessor = Assessor::new(rec, &model.zones)?;
    let config = PipelineConfig {
        format: rec.format,
        layout: model.layout.clone(),
        // SPEC: triggers shape only gesture events, which the report does not use.
        triggers: vec![Trigger::Single; model.zones.len()],
        detector: options.detector,
        veto: VetoParams::DEFAULT,
        gesture: GestureParams::DEFAULT,
    };
    let mut pieces = rec.pieces(options.block)?;
    let mut cursor = SegmentCursor {
        segments: &rec.segments,
        next: 0,
    };
    let mut feed = VetoFeed {
        events: &rec.veto_events,
        next: 0,
    };
    let mut veto = VetoTimeline::new(VetoParams::DEFAULT);
    let mut pipeline: Option<Pipeline> = None;
    let mut events = Vec::new();
    let mut decided: Vec<(SampleIndex, Option<ZoneId>)> = Vec::new();
    while let Some(piece) = pieces.next()? {
        let (block, end) = audio_block(rec.format, &piece);
        let last = clock.at(SampleIndex(end.0 - 1));
        feed.apply_until(last, &mut veto);
        assessor.tick(last);
        for segment in cursor.enter(piece.start) {
            if segment.reconfigured || pipeline.is_none() {
                pipeline = if runs_pipeline(&segment.kind) {
                    Some(Pipeline::new(model.model.clone(), config.clone())?)
                } else {
                    None
                };
            }
        }
        let now = clock.at(end);
        if let Some(p) = pipeline.as_mut() {
            p.push_block(&block, now, &mut veto, &mut events)?;
        }
        for event in events.drain(..) {
            match event {
                PipelineEvent::Decision(d) => {
                    decided.push((d.onset, accepted_zone(&d, &model.zones)));
                    let evidence = Evidence {
                        onset: d.onset,
                        descriptor: d.descriptor().cloned(),
                        scores: d.scores().cloned(),
                        v1_features: None,
                    };
                    assessor.decision(&d, evidence);
                }
                PipelineEvent::DetectorRejected(r) => {
                    let at = clock.at(rejected_onset(&r));
                    assessor.rejected(&r, at);
                }
                PipelineEvent::Gesture(_) | PipelineEvent::InputFault(_) => {}
            }
        }
        assessor.tick(now);
    }
    let summary = &model.model.summary;
    let meta = ReportMeta {
        profile_id: report_profile(rec, model.profile_id),
        device: rec.sidecar.device.clone(),
        front_end_version: model.model.front_end_version,
        model_trained_at: match model.trained_at {
            Some(t) => t,
            None => rec.sidecar.created_at,
        },
        zone_taps: summary.zones.iter().map(|z| z.taps).collect(),
        negatives: summary.negatives.clone(),
        background_regime: model.model.background.regime.clone(),
        input_veto_enabled: true,
    };
    let (report, errors) = assessor.finish(meta, rec.sidecar.created_at)?;
    let agreement = live_agreement(rec, &decided);
    Ok(engine_replay(
        ReplayEngine::V2,
        Some(summary.clone()),
        report,
        Some(agreement),
        errors,
    ))
}

fn accepted_zone(d: &TapDecision, zones: &[ZoneRef]) -> Option<ZoneId> {
    match d.outcome {
        DecisionOutcome::Accepted { zone, .. } => Some(zones[usize::from(zone.0)].id),
        DecisionOutcome::Vetoed { .. }
        | DecisionOutcome::Rejected { .. }
        | DecisionOutcome::Invalid { .. } => None,
    }
}

fn rejected_onset(r: &RejectedCandidate) -> SampleIndex {
    match r.onset {
        Some(onset) => onset,
        None => r.trigger,
    }
}

/// The report's profile id: the profile's, else the recording's, else the nil id.
fn report_profile(rec: &Recording, profile: Option<ProfileId>) -> ProfileId {
    match (profile, rec.sidecar.profile_id) {
        (Some(id), _) | (None, Some(id)) => id,
        (None, None) => ProfileId::from_uuid(Uuid::nil()),
    }
}

/// Pairs each live decision, in onset order, with the nearest unpaired replay decision within
/// ±1 ms (the earlier on a tie).
fn live_agreement(rec: &Recording, replay: &[(SampleIndex, Option<ZoneId>)]) -> LiveAgreement {
    let tolerance = rec.format.rate.samples_in(AGREEMENT_TOLERANCE) as u64;
    let mut used = vec![false; replay.len()];
    let mut agreement = LiveAgreement {
        live: rec.decisions.len() as u32,
        matched: 0,
        mismatched: 0,
        missing: 0,
        extra: 0,
    };
    for (onset, zone) in &rec.decisions {
        let nearest = replay
            .iter()
            .enumerate()
            .filter(|(i, (o, _))| !used[*i] && o.0.abs_diff(onset.0) <= tolerance)
            .min_by_key(|(i, (o, _))| (o.0.abs_diff(onset.0), *i));
        match nearest {
            Some((i, (_, replayed))) => {
                used[i] = true;
                if replayed == zone {
                    agreement.matched += 1;
                } else {
                    agreement.mismatched += 1;
                }
            }
            None => agreement.missing += 1,
        }
    }
    agreement.extra = used.iter().filter(|u| !**u).count() as u32;
    agreement
}

fn engine_replay(
    engine: ReplayEngine,
    training: Option<TrainingSummary>,
    report: EvaluationReport,
    live_agreement: Option<LiveAgreement>,
    errors: Vec<ErrorDump>,
) -> EngineReplay {
    EngineReplay {
        engine,
        training,
        confusion: report.confusion,
        per_zone: report.per_zone,
        overall: report.overall,
        negatives: report.negatives,
        algorithmic_latency: report.decision_latency,
        live_agreement,
        errors,
    }
}

// ---- Evaluation shared by both engines ----

/// What a prompt was resolved by, for its error dump.
#[derive(Debug, Clone, PartialEq)]
struct Evidence {
    onset: SampleIndex,
    descriptor: Option<Descriptor>,
    scores: Option<ModelScores>,
    v1_features: Option<Vec<f32>>,
}

/// The evaluation session on the recorded prompts and negative phases, and the evidence of each
/// prompt's resolution.
struct Assessor {
    session: EvaluationSession,
    events: Vec<EvaluationEvent>,
    evidence: Vec<Option<Evidence>>,
}

impl Assessor {
    /// A `Scripted` schedule (§1.18.3 step 4): the recorded prompts, and a negative phase per
    /// EvaluationNegative or Corpus segment.
    // SPEC: a prompt for a zone the model does not have cannot be scripted; no AssessmentError
    // names it, so it is `ScriptUnordered`, the error for a script that cannot be replayed (as
    // assessment.rs reports a zone index out of range).
    fn new(rec: &Recording, zones: &[ZoneRef]) -> Result<Assessor, ReplayError> {
        let clock = rec.clock();
        let mut prompts = Vec::with_capacity(rec.prompts.len());
        for p in &rec.prompts {
            let Some(zone) = zones.iter().position(|z| z.id == p.zone) else {
                return Err(AssessmentError::ScriptUnordered.into());
            };
            prompts.push(ScriptedPrompt {
                zone: ZoneIndex(zone as u16),
                opened_at: p.opened_at,
                deadline: p.deadline,
                reopened: p.reopened,
            });
        }
        let negatives = rec
            .segments
            .iter()
            .filter_map(|s| {
                negative_class(&s.kind).map(|class| ScriptedNegative {
                    class: class.clone(),
                    start: clock.at(s.start),
                    end: clock.at(match s.end {
                        Some(e) => e,
                        None => rec.end,
                    }),
                })
            })
            .collect();
        let count = prompts.len();
        let session = EvaluationSession::new(
            zones.to_vec(),
            PromptSchedule::Scripted { prompts, negatives },
            clock.at(SampleIndex(0)),
        )?;
        Ok(Assessor {
            session,
            events: Vec::new(),
            evidence: vec![None; count],
        })
    }

    fn decision(&mut self, d: &TapDecision, evidence: Evidence) {
        self.session.on_decision(d, &mut self.events);
        self.absorb(Some(evidence));
    }

    fn rejected(&mut self, r: &RejectedCandidate, at: HostTime) {
        self.session.on_detector_rejected(r, at, &mut self.events);
        let evidence = Evidence {
            onset: rejected_onset(r),
            descriptor: None,
            scores: None,
            v1_features: None,
        };
        self.absorb(Some(evidence));
    }

    fn tick(&mut self, now: HostTime) {
        self.session.tick(now, &mut self.events);
        self.absorb(None);
    }

    /// Records `evidence` for every prompt the last call resolved; a reopened prompt loses its.
    fn absorb(&mut self, evidence: Option<Evidence>) {
        for event in self.events.drain(..) {
            match event {
                EvaluationEvent::PromptResolved { index, .. } => {
                    self.evidence[index as usize] = evidence.clone();
                }
                EvaluationEvent::PromptReopened { index, .. } => {
                    self.evidence[index as usize] = None;
                }
                EvaluationEvent::PromptOpened { .. }
                | EvaluationEvent::NegativeStarted { .. }
                | EvaluationEvent::NegativeEnded { .. }
                | EvaluationEvent::Finished => {}
            }
        }
    }

    /// Runs the schedule to its end (prompts left open are Missed) and reports it, with an error
    /// dump for every prompt not answered Correct, in prompt order.
    fn finish(
        mut self,
        meta: ReportMeta,
        run_at: DateTime<Utc>,
    ) -> Result<(EvaluationReport, Vec<ErrorDump>), ReplayError> {
        self.tick(HostTime(i64::MAX));
        let report = self.session.finish(meta, run_at)?;
        let errors = report
            .prompts
            .iter()
            .zip(self.evidence)
            .enumerate()
            .filter(|(_, (p, _))| !matches!(p.outcome, PromptOutcome::Correct { .. }))
            .map(|(i, (p, evidence))| {
                let (onset, descriptor, scores, v1_features) = match evidence {
                    Some(e) => (Some(e.onset), e.descriptor, e.scores, e.v1_features),
                    None => (None, None, None, None),
                };
                ErrorDump {
                    prompt: i as u32,
                    expected: p.zone,
                    outcome: p.outcome.clone(),
                    onset,
                    descriptor,
                    scores,
                    v1_features,
                }
            })
            .collect();
        Ok((report, errors))
    }
}

// ---- V1 ----

/// A v1 `OnsetDetector` on the mono downmix of one segment, mapping its sample count back to
/// stream positions across gaps.
struct MonoFeed {
    detector: OnsetDetector,
    /// (samples fed before, stream position) at the start and after every jump
    runs: Vec<(usize, SampleIndex)>,
    fed: usize,
    next: Option<SampleIndex>,
    mono: Vec<f32>,
}

/// A v1 onset at its stream position.
struct MonoOnset {
    onset: SampleIndex,
    window: Vec<f32>,
    transient: bool,
}

impl MonoFeed {
    fn new() -> MonoFeed {
        MonoFeed {
            detector: OnsetDetector::new(),
            runs: Vec::new(),
            fed: 0,
            next: None,
            mono: Vec::new(),
        }
    }

    /// Feeds the mean of the channels (summed in f64).
    fn push(&mut self, block: &AudioBlock<'_>) -> Vec<MonoOnset> {
        let channels = usize::from(block.format.channels.get());
        if self.next != Some(block.start) {
            self.runs.push((self.fed, block.start));
        }
        self.mono.clear();
        self.mono
            .extend(block.samples.chunks_exact(channels).map(|frame| {
                (frame.iter().map(|&v| f64::from(v)).sum::<f64>() / channels as f64) as f32
            }));
        self.fed += self.mono.len();
        self.next = Some(SampleIndex(block.start.0 + self.mono.len() as u64));
        let events = self.detector.process(&self.mono);
        events
            .into_iter()
            .map(|e| {
                let (fed, stream) = self.runs[self.runs.partition_point(|r| r.0 <= e.start) - 1];
                MonoOnset {
                    onset: SampleIndex(stream.0 + (e.start - fed) as u64),
                    window: e.window,
                    transient: e.is_transient,
                }
            })
            .collect()
    }
}

/// §1.18.3 V1 path: v1 onset detector and features on the mono downmix, a v1 calibration from the
/// CalibrationZone segments, v1 `classify` for the prompts; no veto, no background.
// SPEC: V1 replays the same calibration run as V2 (the segments from the last CalibrationStarted),
// or every zone segment when the recording has none: v1 needs no seed. v1's environment check is
// not part of the adapter, so the session's noise floor is 0 (its masking check is skipped).
fn replay_v1(rec: &Recording, options: &ReplayOptions) -> Result<EngineReplay, ReplayError> {
    let (from, taps_per_zone) = match rec.runs.last() {
        Some(run) => (run.start, run.taps_per_zone as usize),
        None => (SampleIndex(0), crate::calibration::DEFAULT_TAPS_PER_ZONE),
    };
    let zones = rec.calibration_zones(from);
    if zones.is_empty() {
        return Err(ReplayError::V1 {
            detail: "the recording has no CalibrationZone segment to train the v1 classifier from"
                .into(),
        });
    }
    let zone_ids: Vec<ZoneId> = zones.iter().map(|z| z.id).collect();
    let mut session =
        crate::calibration::CalibrationSession::new(zone_ids.clone(), taps_per_zone, 0.0);
    let mut pieces = rec.pieces(options.block)?;
    let mut cursor = SegmentCursor {
        segments: &rec.segments,
        next: 0,
    };
    let mut feed: Option<MonoFeed> = None;
    let mut prev: Option<&SegmentKind> = None;
    while let Some(piece) = pieces.next()? {
        let (block, _) = audio_block(rec.format, &piece);
        for segment in cursor.enter(piece.start) {
            let zone = match &segment.kind {
                SegmentKind::CalibrationZone { zone, .. } => {
                    (segment.start >= from).then_some(zone)
                }
                SegmentKind::Idle
                | SegmentKind::Listening
                | SegmentKind::Ambient
                | SegmentKind::CalibrationNegatives
                | SegmentKind::EvaluationPrompts
                | SegmentKind::EvaluationNegative { .. }
                | SegmentKind::Corpus { .. } => None,
            };
            if segment.reconfigured || zone.is_none() {
                feed = None;
            }
            if let Some(zone) = zone {
                // INVARIANT: `zones` lists every zone segment's zone from `from` on.
                let target = zone_ids
                    .iter()
                    .position(|id| id == zone)
                    .expect("a planned zone");
                if prev == Some(&segment.kind) || target < session.current_zone() {
                    session.restart_zone(target);
                }
                while session.current_zone() < target && session.advance_zone() {}
                feed.get_or_insert_with(MonoFeed::new);
            }
            prev = Some(&segment.kind);
        }
        if let Some(f) = feed.as_mut() {
            for onset in f.push(&block) {
                if onset.transient {
                    session.record_tap(&onset.window);
                }
            }
        }
    }
    let samples: Vec<(ZoneId, FeatureVector)> = session
        .per_zone_samples()
        .iter()
        .zip(&zone_ids)
        .flat_map(|(fvs, id)| fvs.iter().map(move |fv| (*id, *fv)))
        .collect();
    let classifier = ClassifierParams::train(&samples, &[]).map_err(|e| ReplayError::V1 {
        detail: e.to_string(),
    })?;
    let zone_taps = (0..zones.len())
        .map(|z| session.zone_sample_count(z) as u32)
        .collect();
    evaluate_v1(rec, options, &classifier, &zones, zone_taps)
}

/// The trained v1 classifier, its zones (in the evaluation's order) and the recording's clock.
struct V1Model<'a> {
    classifier: &'a ClassifierParams,
    zones: &'a [ZoneRef],
    clock: Clock<'a>,
}

fn evaluate_v1(
    rec: &Recording,
    options: &ReplayOptions,
    classifier: &ClassifierParams,
    zones: &[ZoneRef],
    zone_taps: Vec<u32>,
) -> Result<EngineReplay, ReplayError> {
    let clock = rec.clock();
    let v1 = V1Model {
        classifier,
        zones,
        clock,
    };
    let mut assessor = Assessor::new(rec, zones)?;
    let mut extractor = crate::features::FeatureExtractor::new();
    let mut pieces = rec.pieces(options.block)?;
    let mut cursor = SegmentCursor {
        segments: &rec.segments,
        next: 0,
    };
    let mut feed: Option<MonoFeed> = None;
    let mut candidates = 0u64;
    while let Some(piece) = pieces.next()? {
        let (block, end) = audio_block(rec.format, &piece);
        assessor.tick(clock.at(SampleIndex(end.0 - 1)));
        for segment in cursor.enter(piece.start) {
            if segment.reconfigured || feed.is_none() {
                feed = runs_pipeline(&segment.kind).then(MonoFeed::new);
            }
        }
        let now = clock.at(end);
        if let Some(f) = feed.as_mut() {
            for onset in f.push(&block) {
                candidates += 1;
                let id = CandidateId(candidates);
                if !onset.transient {
                    let r = RejectedCandidate {
                        id,
                        trigger: onset.onset,
                        onset: Some(onset.onset),
                        reason: RejectReason::NotDecaying,
                        gates: None,
                    };
                    assessor.rejected(&r, clock.at(onset.onset));
                    continue;
                }
                let fv = extractor.extract(&onset.window);
                let d = v1_decision(&v1, &fv, id, &onset, (end, now));
                let evidence = Evidence {
                    onset: onset.onset,
                    descriptor: None,
                    scores: None,
                    v1_features: fv.is_finite().then(|| fv.as_array().to_vec()),
                };
                assessor.decision(&d, evidence);
            }
        }
        assessor.tick(now);
    }
    let meta = ReportMeta {
        profile_id: report_profile(rec, None),
        device: rec.sidecar.device.clone(),
        front_end_version: FRONT_END_VERSION,
        model_trained_at: rec.sidecar.created_at,
        zone_taps,
        negatives: Vec::new(),
        // SPEC: v1 has no background model; the report's meta (which replay does not keep) needs
        // a regime, and PooledAll with no negative clips is the one that claims none.
        background_regime: BackgroundRegime::PooledAll {
            negative_clips: 0,
            needed: 0,
        },
        input_veto_enabled: false,
    };
    let (report, errors) = assessor.finish(meta, rec.sidecar.created_at)?;
    Ok(engine_replay(ReplayEngine::V1, None, report, None, errors))
}

/// A v1 classification as the `TapDecision` the evaluation session scores: Accepted for the
/// winning zone, Rejected(OutOfDistribution) when the novelty gate rejects, Invalid for a
/// non-finite feature vector. Decided when the 4096-sample window completed.
// SPEC: v1 has no descriptor, gates or background score. The decision carries a zero descriptor
// and zero gates that nothing reads (the V1 report keeps the v1 features instead), and scores
// from v1's feature space: d2 the squared centroid distances, llr_bg 0, posterior_best v1's
// confidence. That keeps one evaluation path (EvaluationSession) for both engines.
fn v1_decision(
    v1: &V1Model<'_>,
    fv: &FeatureVector,
    id: CandidateId,
    onset: &MonoOnset,
    (emitted_at, now): (SampleIndex, HostTime),
) -> TapDecision {
    let V1Model {
        classifier,
        zones,
        clock,
    } = v1;
    let gates = GateReport {
        snr: Decibels::new(0.0),
        broadband_fraction: 0.0,
        valid_bands: 0,
        decay_fraction: 0.0,
        clipped: false,
        trigger_flux: 0.0,
    };
    let outcome = match fv.as_array().iter().position(|v| !v.is_finite()) {
        Some(index) => DecisionOutcome::Invalid {
            error: DescriptorError::NonFinite {
                index: index as u16,
            },
        },
        None => {
            let classification = classifier.classify(fv);
            let space = classifier.feature_space(fv);
            let d2: Vec<f64> = zones
                .iter()
                .map(|z| {
                    let distance = space
                        .zones
                        .iter()
                        .find(|d| d.zone_id == z.id)
                        .map(|d| f64::from(d.centroid_distance))
                        // INVARIANT: the classifier was trained on exactly these zones.
                        .expect("every zone has a centroid");
                    distance * distance
                })
                .collect();
            let mut order: Vec<usize> = (0..d2.len()).collect();
            order.sort_by(|&a, &b| d2[a].total_cmp(&d2[b]).then(a.cmp(&b)));
            let (best, second) = (order[0], order[order.len().min(2) - 1]);
            let scores = ModelScores {
                best: ZoneIndex(best as u16),
                second: ZoneIndex(second as u16),
                d2_min: d2[best],
                margin: d2[second] - d2[best],
                llr_bg: 0.0,
                posterior_best: f64::from(classification.confidence),
                d2,
            };
            let descriptor = Descriptor {
                layout: DescriptorLayout::Mono { a: 0 },
                values: [0.0; DESCRIPTOR_DIMS],
            };
            match classification.zone_id {
                Some(zone) => DecisionOutcome::Accepted {
                    zone: ZoneIndex(
                        zones
                            .iter()
                            .position(|z| z.id == zone)
                            // INVARIANT: classify returns one of the zones it was trained on.
                            .expect("a trained zone") as u16,
                    ),
                    descriptor,
                    scores,
                },
                None => DecisionOutcome::Rejected {
                    reason: ModelRejectReason::OutOfDistribution,
                    descriptor,
                    scores,
                },
            }
        }
    };
    TapDecision {
        candidate: id,
        onset: onset.onset,
        onset_time: clock.mapped(onset.onset),
        emitted_at,
        decided_at: now,
        gates,
        outcome,
    }
}

// ---- Synthetic recordings (`desk_render`) ----

/// What `render_desk_recording` plays after the calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DeskScene {
    /// prompts, then the evaluation's Everyday background check
    CalibrateEvaluate,
    /// prompts, then one negative phase per everyday class (typing, clicking, speech, music,
    /// knocks, objects set down)
    Negatives,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeskRender {
    pub scene: DeskScene,
    /// 1 or 2 channels (the desk's two mics, or the first)
    pub format: AudioFormat,
    /// 2, 4, 6 or 8
    pub zones: u8,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RenderError {
    ZoneCount { zones: u8 },
    Channels { channels: u16 },
    Desk { error: DeskError },
    Recording { error: RecordingError },
}

impl From<RecordingError> for RenderError {
    fn from(error: RecordingError) -> Self {
        RenderError::Recording { error }
    }
}

/// Calibration taps per zone the rendered run plans (CalibrationStarted.taps_per_zone).
pub const RENDER_TAPS_PER_ZONE: u32 = 10;
/// Evaluation prompts per zone.
pub const RENDER_PROMPTS_PER_ZONE: u32 = 6;
/// Taps played per zone: a few beyond the target, as a user taps until the zone is full.
const RENDER_TAPS_PLAYED: usize = 12;
/// Zone taps peak about 29 dB below full scale, about 50 dB over the desk floor.
const RENDER_TAP_FORCE: f64 = -30.0;
/// Host time of the recording's first sample (100 s).
const RENDER_BASE_TICKS: i64 = 1_000_000_000;
const RENDER_BLOCK: Milliseconds = Milliseconds::new(10.0);
/// The front end needs about 312 ms after a re-Configure before it can trigger (p2 §2.8).
const SEGMENT_LEAD_MS: f64 = 400.0;
const TAP_SPACING_MS: f64 = 400.0;
const PROMPT_SPACING_MS: f64 = 1500.0;
/// A prompted tap lands this long after its prompt opens.
const PROMPT_REACTION_MS: f64 = 400.0;
const NEGATIVE_PHASE_MS: f64 = 4500.0;
/// OS key and button events precede the keycap's sound by this much.
const INPUT_LEAD_MS: f64 = 8.0;

/// Writes a synthetic recording under `root` (a new `<YYYYMMDDTHHMMSSZ>` directory, named after a
/// fixed creation time): an Ambient segment, one CalibrationZone segment per zone with
/// RENDER_TAPS_PLAYED taps, a CalibrationNegatives segment, then the scene's prompts (with their
/// Prompt events) and negative phases (with Input and SelfOutput events where the sounds are
/// typing, clicks or speech). Deterministic in `spec`.
pub fn render_desk_recording(
    root: &Path,
    spec: &DeskRender,
) -> Result<RecordingSummary, RenderError> {
    if !matches!(spec.zones, 2 | 4 | 6 | 8) {
        return Err(RenderError::ZoneCount { zones: spec.zones });
    }
    let channels = spec.format.channels.get();
    if !matches!(channels, 1 | 2) {
        return Err(RenderError::Channels { channels });
    }
    let mut config = DeskConfig::default_for(spec.format.rate, spec.seed);
    config.mics.truncate(usize::from(channels));
    let desk = ModalDesk::new(config).map_err(|error| RenderError::Desk { error })?;
    let created_at = DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
        // INVARIANT: a literal RFC 3339 time.
        .expect("a valid time")
        .with_timezone(&Utc);
    let header = SidecarHeader {
        app_version: "desk_render".into(),
        device: DeviceIdentity {
            endpoint_id: "desk_render".into(),
            name: "Synthetic desk".into(),
            format: spec.format,
            capture_mode: CaptureMode::Raw,
        },
        layout: None,
        profile_id: None,
        created_at,
    };
    let writer = RecordingWriter::create(root, header)?;
    let mut r = Render {
        desk,
        writer,
        format: spec.format,
        pos: 0,
        rng: Rng64::new(spec.seed ^ 0xDE5C_8E4D),
    };
    let zone_count = usize::from(spec.zones);
    let zones: Vec<(ZoneId, String)> = (0..zone_count)
        .map(|z| {
            let id = Uuid::from_u128((u128::from(spec.seed) << 64) | 0xDE5C_0000 | z as u128);
            (ZoneId::from_uuid(id), format!("Zone {}", z + 1))
        })
        .collect();
    let centres = ModalDesk::zone_centres(zone_count);

    r.segment(SegmentKind::Idle);
    r.play(500.0, Vec::new())?;
    let start = r.at();
    r.writer.push_event(SidecarEvent::CalibrationStarted {
        start,
        seed: spec.seed,
        taps_per_zone: RENDER_TAPS_PER_ZONE,
    });
    r.segment(SegmentKind::Ambient);
    r.play(3300.0, Vec::new())?;
    r.segment(SegmentKind::Idle);
    r.play(300.0, Vec::new())?;
    for ((id, label), centre) in zones.iter().zip(&centres) {
        r.segment(SegmentKind::CalibrationZone {
            zone: *id,
            label: label.clone(),
        });
        let taps = (0..RENDER_TAPS_PLAYED)
            .map(|i| (SEGMENT_LEAD_MS + i as f64 * TAP_SPACING_MS, r.tap(*centre)))
            .collect();
        r.play(
            SEGMENT_LEAD_MS + RENDER_TAPS_PLAYED as f64 * TAP_SPACING_MS + 200.0,
            taps,
        )?;
    }
    r.segment(SegmentKind::CalibrationNegatives);
    let set_downs = (0..8)
        .map(|i| {
            let d = match i % 2 {
                0 => Distractor::Knock {
                    x: 0.30 + 0.05 * i as f64,
                    y: 0.40,
                    count: 1,
                },
                _ => Distractor::ObjectSetDown {
                    x: 0.80 - 0.05 * i as f64,
                    y: 0.45,
                },
            };
            (SEGMENT_LEAD_MS + i as f64 * 700.0, r.desk.distractor(&d, 0))
        })
        .collect();
    r.play(SEGMENT_LEAD_MS + 8.0 * 700.0 + 300.0, set_downs)?;
    r.segment(SegmentKind::Idle);
    r.play(500.0, Vec::new())?;

    r.segment(SegmentKind::EvaluationPrompts);
    let mut order: Vec<usize> = (0..RENDER_PROMPTS_PER_ZONE as usize)
        .flat_map(|_| 0..zone_count)
        .collect();
    for i in (1..order.len()).rev() {
        let j = r.rng.below(i as u64 + 1) as usize;
        order.swap(i, j);
    }
    let mut taps = Vec::with_capacity(order.len());
    for (k, &zone) in order.iter().enumerate() {
        let opened = SEGMENT_LEAD_MS + k as f64 * PROMPT_SPACING_MS;
        let opened_at = r.time_in(opened);
        let deadline = r.time_in(opened + crate::assessment::PROMPT_WINDOW.get());
        r.writer.push_event(SidecarEvent::Prompt {
            zone: zones[zone].0,
            opened_at,
            deadline,
            reopened: false,
        });
        taps.push((opened + PROMPT_REACTION_MS, r.tap(centres[zone])));
    }
    r.play(
        SEGMENT_LEAD_MS + order.len() as f64 * PROMPT_SPACING_MS + 300.0,
        taps,
    )?;

    match spec.scene {
        DeskScene::CalibrateEvaluate => r.everyday_phase()?,
        DeskScene::Negatives => {
            r.typing_phase()?;
            r.clicking_phase()?;
            r.sound_phase(
                NegativeClass::Speech,
                &[Distractor::SpeechLike { seconds: 3.0 }],
                1500.0,
            )?;
            r.sound_phase(
                NegativeClass::Music,
                &[Distractor::DrumsOverTones { bars: 1 }],
                1500.0,
            )?;
            let knock = |x| Distractor::Knock {
                x,
                y: 0.30,
                count: 2,
            };
            r.sound_phase(
                NegativeClass::Knocks,
                &[knock(0.35), knock(0.6), knock(0.85)],
                1200.0,
            )?;
            let set_down = |x| Distractor::ObjectSetDown { x, y: 0.50 };
            r.sound_phase(
                NegativeClass::ObjectsSetDown,
                &[set_down(0.3), set_down(0.6), set_down(0.9)],
                1200.0,
            )?;
        }
    }
    r.segment(SegmentKind::Idle);
    r.play(500.0, Vec::new())?;
    Ok(r.writer.finish()?)
}

/// The renderer's state: the desk, the recording and the stream position.
struct Render {
    desk: ModalDesk,
    writer: RecordingWriter,
    format: AudioFormat,
    pos: u64,
    rng: Rng64,
}

impl Render {
    fn at(&self) -> SampleIndex {
        SampleIndex(self.pos)
    }

    fn frames(&self, ms: f64) -> usize {
        self.format.rate.samples_in(Milliseconds::new(ms))
    }

    /// Host time of the sample `ms` after the current position.
    fn time_in(&self, ms: f64) -> HostTime {
        self.time(self.pos + self.frames(ms) as u64)
    }

    fn time(&self, s: u64) -> HostTime {
        let fs = i128::from(self.format.rate.hz());
        let ticks = (i128::from(s) * i128::from(HostTime::TICKS_PER_SECOND) * 2 + fs) / (2 * fs);
        // INVARIANT: a rendered recording lasts minutes; its ticks fit an i64.
        HostTime(RENDER_BASE_TICKS + i64::try_from(ticks).expect("ticks fit"))
    }

    fn segment(&mut self, kind: SegmentKind) {
        self.writer.push_event(SidecarEvent::Segment {
            start: self.at(),
            kind,
            reconfigured: true,
        });
    }

    /// A tap at a zone centre, ±2 dB around RENDER_TAP_FORCE, contact 0.8-1.5 ms.
    fn tap(&mut self, (x, y): (f64, f64)) -> Vec<f32> {
        let force = RENDER_TAP_FORCE + self.rng.range(-2.0, 2.0);
        let contact = self.rng.range(0.8, 1.5);
        let spec = TapSpec {
            x,
            y,
            force: Decibels::new(force),
            contact: Milliseconds::new(contact),
            position_jitter_m: 0.02,
        };
        self.desk.tap(&spec, 0)
    }

    fn input(&mut self, ms: f64, kind: InputKind) {
        let time = self.time_in(ms);
        self.writer.push_event(SidecarEvent::Input {
            event: InputEvent { time, kind },
        });
    }

    /// Writes `ms` of the desk floor with each (at ms, interleaved sound) added, cut at the end.
    fn play(&mut self, ms: f64, sounds: Vec<(f64, Vec<f32>)>) -> Result<(), RenderError> {
        let channels = usize::from(self.format.channels.get());
        let frames = self.frames(ms);
        let mut audio = self.desk.floor(frames);
        for (at, sound) in sounds {
            let from = (self.frames(at) * channels).min(audio.len());
            for (v, s) in audio[from..].iter_mut().zip(sound) {
                *v += s;
            }
        }
        let block = self.format.rate.samples_in(RENDER_BLOCK) * channels;
        for chunk in audio.chunks(block) {
            let block = AudioBlock::new(
                self.format,
                self.at(),
                chunk,
                BlockTime::Timestamped {
                    host_time: self.time(self.pos),
                },
                false,
            )
            // INVARIANT: chunks of whole frames from an interleaved buffer of whole frames.
            .expect("whole frames");
            self.writer.write_block(&block)?;
            self.writer.checkpoint()?;
            self.pos += (chunk.len() / channels) as u64;
        }
        Ok(())
    }

    /// Everyday background check: typing, a click and a spoken confirmation (self-output).
    fn everyday_phase(&mut self) -> Result<(), RenderError> {
        self.segment(SegmentKind::EvaluationNegative {
            class: NegativeClass::Everyday,
        });
        let mut sounds = Vec::new();
        for (k, at) in [500.0, 1100.0, 1700.0].into_iter().enumerate() {
            let gap = 90.0 + 20.0 * k as f64;
            sounds.push(self.keystroke(at, gap));
        }
        sounds.push(self.click(2300.0));
        let (began, ended) = (self.time_in(2950.0), self.time_in(4000.0));
        self.writer.push_event(SidecarEvent::SelfOutput {
            event: SelfOutputEvent::Began {
                id: OutputId(1),
                kind: SelfOutputKind::Speech,
                at: began,
            },
        });
        self.writer.push_event(SidecarEvent::SelfOutput {
            event: SelfOutputEvent::Ended {
                id: OutputId(1),
                at: ended,
            },
        });
        let speech = self
            .desk
            .distractor(&Distractor::SpeechLike { seconds: 1.0 }, 0);
        sounds.push((3000.0, speech));
        self.play(NEGATIVE_PHASE_MS + 500.0, sounds)
    }

    fn typing_phase(&mut self) -> Result<(), RenderError> {
        self.segment(SegmentKind::EvaluationNegative {
            class: NegativeClass::Typing,
        });
        let sounds = (0..6)
            .map(|k| self.keystroke(400.0 + 600.0 * k as f64, 80.0 + 10.0 * k as f64))
            .collect();
        self.play(NEGATIVE_PHASE_MS, sounds)
    }

    fn clicking_phase(&mut self) -> Result<(), RenderError> {
        self.segment(SegmentKind::EvaluationNegative {
            class: NegativeClass::Clicking,
        });
        let sounds = (0..5)
            .map(|k| self.click(400.0 + 800.0 * k as f64))
            .collect();
        self.play(NEGATIVE_PHASE_MS, sounds)
    }

    /// A negative phase of `sounds`, `spacing` ms apart from 400 ms in, with no OS events.
    fn sound_phase(
        &mut self,
        class: NegativeClass,
        sounds: &[Distractor],
        spacing: f64,
    ) -> Result<(), RenderError> {
        self.segment(SegmentKind::EvaluationNegative { class });
        let sounds = sounds
            .iter()
            .enumerate()
            .map(|(k, d)| (400.0 + spacing * k as f64, self.desk.distractor(d, 0)))
            .collect();
        self.play(NEGATIVE_PHASE_MS, sounds)
    }

    /// A key press and release `gap` ms apart, with their KeyDown and KeyUp events.
    fn keystroke(&mut self, at: f64, gap: f64) -> (f64, Vec<f32>) {
        self.input(at - INPUT_LEAD_MS, InputKind::KeyDown);
        self.input(at + gap - INPUT_LEAD_MS, InputKind::KeyUp);
        let sound = Distractor::KeystrokePair {
            gap: Milliseconds::new(gap),
        };
        (at, self.desk.distractor(&sound, 0))
    }

    /// A mouse click with its button events.
    fn click(&mut self, at: f64) -> (f64, Vec<f32>) {
        self.input(at - INPUT_LEAD_MS, InputKind::MouseButtonDown);
        self.input(at + 60.0, InputKind::MouseButtonUp);
        (at, self.desk.distractor(&Distractor::MouseClick, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recording::SIDECAR_FILE;
    use crate::units::{ChannelCount, SampleRate};

    const ZONES: u8 = 2;

    /// A fresh directory under the temp dir, removed when dropped.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> TempDir {
            let dir =
                std::env::temp_dir().join(format!("qwerty-replay-{name}-{}", std::process::id()));
            if dir.exists() {
                std::fs::remove_dir_all(&dir).expect("clear the old test directory");
            }
            std::fs::create_dir_all(&dir).expect("make the test directory");
            TempDir(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            if let Err(e) = std::fs::remove_dir_all(&self.0) {
                eprintln!("could not remove {}: {e}", self.0.display());
            }
        }
    }

    fn format() -> AudioFormat {
        AudioFormat {
            rate: SampleRate::new(48_000).expect("rate"),
            channels: ChannelCount::new(2).expect("channels"),
        }
    }

    /// Renders `scene` under a fresh temp dir; returns the dir guard and the recording.
    fn render(name: &str, scene: DeskScene, seed: u64) -> (TempDir, PathBuf) {
        let tmp = TempDir::new(name);
        let spec = DeskRender {
            scene,
            format: format(),
            zones: ZONES,
            seed,
        };
        let summary = render_desk_recording(&tmp.0, &spec).expect("the desk renders");
        let dir = PathBuf::from(summary.dir);
        (tmp, dir)
    }

    fn options(engines: Vec<ReplayEngine>) -> ReplayOptions {
        ReplayOptions {
            engines,
            detector: DetectorParams::DEFAULT,
            training: TrainingParams::DEFAULT,
            block: DEFAULT_REPLAY_BLOCK,
            profile: None,
        }
    }

    fn json(report: &ReplayReport) -> String {
        serde_json::to_string_pretty(report).expect("a report serializes")
    }

    fn read_sidecar(dir: &Path) -> Sidecar {
        let text = std::fs::read_to_string(dir.join(SIDECAR_FILE)).expect("read the sidecar");
        serde_json::from_str(&text).expect("a valid sidecar")
    }

    fn write_sidecar(dir: &Path, sidecar: &Sidecar) {
        let text = serde_json::to_string_pretty(sidecar).expect("a sidecar serializes");
        std::fs::write(dir.join(SIDECAR_FILE), text).expect("write the sidecar");
    }

    fn total_prompts() -> u32 {
        u32::from(ZONES) * RENDER_PROMPTS_PER_ZONE
    }

    #[test]
    fn rp1_desk_render_replays_v2_deterministically() {
        let (_tmp, dir) = render("rp1", DeskScene::CalibrateEvaluate, 0xA1);
        let options = options(vec![ReplayEngine::V2]);
        let first = run_replay(&dir, &options).expect("the first replay");
        let second = run_replay(&dir, &options).expect("the second replay");
        assert_eq!(json(&first), json(&second), "two replays, one report");

        assert_eq!(first.schema_version, REPLAY_SCHEMA_VERSION);
        assert_eq!(first.format, format());
        assert_eq!(first.params, DetectorParams::DEFAULT);
        let [r] = &first.results[..] else {
            panic!("one engine, one result: {:?}", first.results);
        };
        assert_eq!(r.engine, ReplayEngine::V2);
        let training = r.training.as_ref().expect("V2 reports its training");
        assert_eq!(training.seed, 0xA1, "the recorded seed");
        assert_eq!(training.zones.len(), usize::from(ZONES));
        // Every prompt was resolved by a decision of the replayed pipeline: none ran out.
        assert_eq!(r.overall.prompts, total_prompts());
        assert_eq!(r.overall.missed, 0, "{:?}", r.errors);
        assert_eq!(r.confusion.counts.len(), usize::from(ZONES));
        let [everyday] = &r.negatives[..] else {
            panic!("one background check: {:?}", r.negatives);
        };
        assert_eq!(everyday.class, NegativeClass::Everyday);
        // The synthetic recording holds no live decisions: every replay decision is extra.
        let agreement = r.live_agreement.as_ref().expect("V2 compares with live");
        assert_eq!(
            (
                agreement.live,
                agreement.matched,
                agreement.mismatched,
                agreement.missing
            ),
            (0, 0, 0, 0)
        );
        assert!(agreement.extra >= total_prompts());
        eprintln!(
            "rp1: accuracy {:.3}, wrong {}, rejected {}, errors {}, everyday {:?}",
            r.overall.accuracy,
            r.overall.wrong,
            r.overall.rejected,
            r.errors.len(),
            everyday
        );
    }

    #[test]
    fn rp3_v1_adapter_reports_on_the_same_recording() {
        let (_tmp, dir) = render("rp3", DeskScene::CalibrateEvaluate, 0xA3);
        let report = run_replay(&dir, &options(vec![ReplayEngine::V1])).expect("the v1 replay");
        let [r] = &report.results[..] else {
            panic!("one engine, one result: {:?}", report.results);
        };
        assert_eq!(r.engine, ReplayEngine::V1);
        assert_eq!(r.training, None, "V1 has no v2 training summary");
        assert_eq!(r.live_agreement, None, "live agreement is V2's");
        assert_eq!(r.overall.prompts, total_prompts());
        assert_eq!(r.per_zone.len(), usize::from(ZONES));
        assert_eq!(r.negatives.len(), 1);
        for e in &r.errors {
            assert_eq!(e.descriptor, None, "V1 has no v2 descriptor");
            assert_eq!(e.scores, None);
        }
        let again = run_replay(&dir, &options(vec![ReplayEngine::V1])).expect("again");
        assert_eq!(json(&report), json(&again));
        eprintln!(
            "rp3: v1 accuracy {:.3}, wrong {}, rejected {}, missed {}",
            r.overall.accuracy, r.overall.wrong, r.overall.rejected, r.overall.missed
        );
    }

    #[test]
    fn rp4_last_calibration_run_seed_and_event_order() {
        let seed = 0xA4;
        let (tmp, dir) = render("rp4", DeskScene::CalibrateEvaluate, seed);
        let rendered = read_sidecar(&dir);

        // A first, abandoned calibration run in the leading Idle segment: another seed, another
        // zone and a tap count no segment could reach.
        let stale = ZoneId::from_uuid(Uuid::from_u128(0x57A1E));
        let mut two_runs = rendered.clone();
        let first = two_runs
            .events
            .iter_mut()
            .find(|e| matches!(e, SidecarEvent::Segment { start, .. } if start.0 == 0))
            .expect("the recording starts with a segment");
        *first = SidecarEvent::Segment {
            start: SampleIndex(0),
            kind: SegmentKind::CalibrationZone {
                zone: stale,
                label: "Stale".into(),
            },
            reconfigured: true,
        };
        two_runs.events.insert(
            0,
            SidecarEvent::CalibrationStarted {
                start: SampleIndex(0),
                seed: 0xBAD,
                taps_per_zone: 99,
            },
        );
        write_sidecar(&dir, &two_runs);
        let options = options(vec![ReplayEngine::V2]);
        let sorted = run_replay(&dir, &options).expect("calibrates from the second run");
        let training = sorted.results[0].training.as_ref().expect("a V2 summary");
        assert_eq!(training.seed, seed);
        let rendered_zones: Vec<ZoneId> = rendered
            .events
            .iter()
            .filter_map(|e| match e {
                SidecarEvent::Segment {
                    kind: SegmentKind::CalibrationZone { zone, .. },
                    ..
                } => Some(*zone),
                SidecarEvent::Segment { .. }
                | SidecarEvent::CalibrationStarted { .. }
                | SidecarEvent::Prompt { .. }
                | SidecarEvent::Input { .. }
                | SidecarEvent::SelfOutput { .. }
                | SidecarEvent::Decision { .. } => None,
            })
            .collect();
        let trained: Vec<ZoneId> = training.zones.iter().map(|z| z.zone).collect();
        assert_eq!(
            trained, rendered_zones,
            "the stale run's zone is not trained"
        );
        assert!(training
            .zones
            .iter()
            .all(|z| z.taps == RENDER_TAPS_PER_ZONE));

        // The same events written in reverse order replay to the same report.
        let reversed_dir = tmp.0.join("reversed");
        std::fs::create_dir_all(&reversed_dir).expect("make the copy");
        std::fs::copy(dir.join(AUDIO_FILE), reversed_dir.join(AUDIO_FILE)).expect("copy audio");
        let mut reversed = two_runs.clone();
        reversed.events.reverse();
        write_sidecar(&reversed_dir, &reversed);
        let mut replayed = run_replay(&reversed_dir, &options).expect("the reversed file replays");
        assert_ne!(replayed.recording, sorted.recording);
        replayed.recording = sorted.recording.clone();
        assert_eq!(json(&replayed), json(&sorted));

        // Calibration segments without a CalibrationStarted, and no profile: no seed to replay.
        let mut unseeded = rendered.clone();
        unseeded
            .events
            .retain(|e| !matches!(e, SidecarEvent::CalibrationStarted { .. }));
        write_sidecar(&dir, &unseeded);
        assert_eq!(
            run_replay(&dir, &options),
            Err(ReplayError::NoCalibrationSeed)
        );

        // Neither a profile nor calibration segments.
        let mut uncalibrated = rendered;
        for e in &mut uncalibrated.events {
            if let SidecarEvent::Segment { kind, .. } = e {
                if is_calibration(kind) {
                    *kind = SegmentKind::Idle;
                }
            }
        }
        write_sidecar(&dir, &uncalibrated);
        assert_eq!(
            run_replay(&dir, &options),
            Err(ReplayError::NoCalibrationAndNoProfile)
        );
    }
}
