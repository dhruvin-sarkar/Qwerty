//! The single streaming path: detector, descriptor, veto, model and gesture (§1.15).
//!
//! `Pipeline` turns audio blocks into final tap decisions and gesture events. Every gate-passing
//! candidate is described and scored at once, then waits in the pending-decision queue until the
//! veto timeline can settle it (t0 + input_after + settle), and is finalized in onset order.
//! `CalibrationFrontEnd` composes the same detector, extractor and queue for calibration clips.

use crate::audio::{AudioBlock, AudioFormat, ChannelLayout};
use crate::descriptor::{Descriptor, DescriptorError, DescriptorExtractor, DescriptorLayout};
use crate::detector::{
    Candidate, Detector, DetectorError, DetectorMode, DetectorOutput, DetectorParams, GateReport,
    InputFault, RejectedCandidate, FRONT_END_VERSION,
};
use crate::gesture::{CandidateOutcome, Gesture, GestureEvent, GestureParams, Trigger};
use crate::model::{
    ModelDecision, ModelError, ModelRejectReason, ModelScores, ModelScratch, TrainedModel,
};
use crate::units::{CandidateId, HostTime, SampleIndex, SampleRate, ZoneIndex};
use crate::veto::{VetoDecision, VetoParams, VetoReason, VetoTimeline, VetoVerdict};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;

pub const MAX_PENDING_DECISIONS: usize = 8;

/// Capacity of the reused detector-output buffer. One push yields at most one trigger per 30 ms
/// (the minimum trigger spacing) plus faults, so a block of up to ~2 s never grows it.
const DETECTED_CAPACITY: usize = 64;
/// Capacity of the reused gesture-event buffer: one candidate yields at most four events.
const GESTURE_CAPACITY: usize = 8;

#[derive(Debug, Clone, PartialEq)]
pub struct PipelineConfig {
    pub format: AudioFormat,
    pub layout: ChannelLayout,  // from the profile fingerprint
    pub triggers: Vec<Trigger>, // one per model zone, model order
    pub detector: DetectorParams,
    pub veto: VetoParams,
    pub gesture: GestureParams,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PipelineError {
    Detector { error: DetectorError },
    ModelIncompatible { reason: Incompatibility },
    TriggerCountMismatch { zones: u32, triggers: u32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Incompatibility {
    FrontEndVersion {
        model: u32,
        current: u32,
    },
    Layout {
        model: DescriptorLayout,
        current: DescriptorLayout,
    },
    Rate {
        model: SampleRate,
        current: SampleRate,
    },
    Invalid {
        error: ModelError,
    },
}

/// The descriptor and scores live in the variants that have them (RUST-RT-24): only `Invalid` has none.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DecisionOutcome {
    Accepted {
        zone: ZoneIndex,
        descriptor: Descriptor,
        scores: ModelScores,
    },
    Vetoed {
        reason: VetoReason,
        descriptor: Descriptor,
        scores: ModelScores,
    },
    Rejected {
        reason: ModelRejectReason,
        descriptor: Descriptor,
        scores: ModelScores,
    },
    Invalid {
        error: DescriptorError,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TapDecision {
    pub candidate: CandidateId,
    pub onset: SampleIndex,
    pub onset_time: Option<HostTime>,
    pub emitted_at: SampleIndex,
    /// `now` passed to push_block/advance when the decision became final
    pub decided_at: HostTime,
    pub gates: GateReport,
    pub outcome: DecisionOutcome,
}

impl TapDecision {
    /// None iff Invalid (derived from `outcome`, not stored twice).
    pub fn descriptor(&self) -> Option<&Descriptor> {
        match &self.outcome {
            DecisionOutcome::Accepted { descriptor, .. }
            | DecisionOutcome::Vetoed { descriptor, .. }
            | DecisionOutcome::Rejected { descriptor, .. } => Some(descriptor),
            DecisionOutcome::Invalid { .. } => None,
        }
    }

    pub fn scores(&self) -> Option<&ModelScores> {
        match &self.outcome {
            DecisionOutcome::Accepted { scores, .. }
            | DecisionOutcome::Vetoed { scores, .. }
            | DecisionOutcome::Rejected { scores, .. } => Some(scores),
            DecisionOutcome::Invalid { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineEvent {
    Decision(Box<TapDecision>),
    Gesture(GestureEvent),
    DetectorRejected(RejectedCandidate),
    InputFault(InputFault),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PipelineCounters {
    pub candidates: u64,
    pub accepted: u64,
    pub vetoed: u64,
    pub rejected: u64,
    pub invalid: u64,
    pub detector_rejected: u64,
    pub faults: u64,
    pub decision_overflow: u64,
}

// ---- The pending-decision queue, shared by `Pipeline` and `CalibrationFrontEnd` (RUST-RT-15). ----

/// A candidate waiting for its veto verdict.
#[derive(Debug)]
struct Queued<T> {
    onset: SampleIndex,
    onset_time: Option<HostTime>,
    item: T,
}

/// Candidates in onset order, each held until the veto timeline settles it. The one
/// implementation of the overflow rule and the veto resolution. Allocates only in `new`.
#[derive(Debug)]
struct DecisionQueue<T> {
    items: VecDeque<Queued<T>>,
}

impl<T> DecisionQueue<T> {
    fn new() -> Self {
        // One spare slot: an insert lands before the eviction that restores the cap.
        DecisionQueue {
            items: VecDeque::with_capacity(MAX_PENDING_DECISIONS + 1),
        }
    }

    /// Inserts in onset order. If more than MAX_PENDING_DECISIONS are then queued, the oldest
    /// is removed and returned: the caller finalizes it as `Vetoed{DecisionOverflow}`.
    fn push(&mut self, queued: Queued<T>) -> Option<Queued<T>> {
        let at = self.items.partition_point(|q| q.onset <= queued.onset);
        self.items.insert(at, queued);
        if self.items.len() > MAX_PENDING_DECISIONS {
            self.items.pop_front()
        } else {
            None
        }
    }

    /// The oldest item and its verdict once that verdict is final at `now`: untimed items are
    /// `Vetoed{ClockUnavailable}` at once, timed ones wait for `VetoTimeline::decide`. Later items
    /// never overtake a pending one, so decisions leave in onset order.
    fn pop_final(
        &mut self,
        veto: &VetoTimeline,
        now: HostTime,
    ) -> Option<(Queued<T>, VetoVerdict)> {
        let front = self.items.front()?;
        let verdict = match front.onset_time {
            None => VetoVerdict::Vetoed {
                reason: VetoReason::ClockUnavailable,
            },
            Some(t0) => match veto.decide(t0, now) {
                VetoDecision::Pending { .. } => return None,
                VetoDecision::Final { verdict } => verdict,
            },
        };
        self.items.pop_front().map(|queued| (queued, verdict))
    }
}

const OVERFLOW: VetoVerdict = VetoVerdict::Vetoed {
    reason: VetoReason::DecisionOverflow,
};

/// Runs the detector on `block`, leaving its outputs in `detected` (cleared first).
fn detect(
    detector: &mut Detector,
    block: &AudioBlock<'_>,
    detected: &mut Vec<DetectorOutput>,
) -> Result<(), DetectorError> {
    detected.clear();
    detector.push(block, detected)
}

// ---- Pipeline ----

/// A candidate after the model, before its veto.
#[derive(Debug)]
struct Judged {
    candidate: CandidateId,
    emitted_at: SampleIndex,
    gates: GateReport,
    /// An extraction error is final whatever the veto says (§1.15 flow, step 2).
    judgement: Result<Scored, DescriptorError>,
}

/// A described candidate and the model's decision on it.
#[derive(Debug)]
struct Scored {
    descriptor: Descriptor,
    decision: ModelDecision,
}

/// The pipeline does NOT own the veto timeline (RUST-RT-9). The processing thread, and replay, own one
/// `VetoTimeline` for their whole lifetime and lend it to every call, so a reconfigure (pause/resume, a mode
/// change, a trigger edit) keeps the open self-output intervals and the 2 s input history. Input, gap and
/// self-output events go straight to the timeline, and `VetoTimeline::close_stuck` is called on it directly.
/// Gesture state is per pipeline: a new pipeline starts with no burst hold and no pending double tap.
pub struct Pipeline {
    detector: Detector,
    extractor: DescriptorExtractor,
    model: Arc<TrainedModel>,
    scratch: ModelScratch,
    gesture: Gesture,
    pending: DecisionQueue<Judged>,
    detected: Vec<DetectorOutput>,
    gestures: Vec<GestureEvent>,
    counters: PipelineCounters,
}

impl Pipeline {
    pub fn new(model: Arc<TrainedModel>, config: PipelineConfig) -> Result<Self, PipelineError> {
        check_model(&model, &config)?;
        let zones = model.zone_ids.len();
        if config.triggers.len() != zones {
            return Err(PipelineError::TriggerCountMismatch {
                zones: saturating_u32(zones),
                triggers: saturating_u32(config.triggers.len()),
            });
        }
        let detector = Detector::new(
            config.format,
            &config.layout,
            &config.detector,
            DetectorMode::Live,
        )
        .map_err(|error| PipelineError::Detector { error })?;
        Ok(Pipeline {
            detector,
            extractor: DescriptorExtractor::new(config.format.rate, model.layout),
            scratch: model.scratch(),
            model,
            gesture: Gesture::new(config.triggers, config.gesture),
            pending: DecisionQueue::new(),
            detected: Vec::with_capacity(DETECTED_CAPACITY),
            gestures: Vec::with_capacity(GESTURE_CAPACITY),
            counters: PipelineCounters::default(),
        })
    }

    /// Detects, describes and scores, queues each candidate for its veto, passes rejections and
    /// faults straight through, then runs `advance(now)`. Allocates nothing when there is no
    /// candidate (P6).
    pub fn push_block(
        &mut self,
        block: &AudioBlock<'_>,
        now: HostTime,
        veto: &mut VetoTimeline,
        out: &mut Vec<PipelineEvent>,
    ) -> Result<(), PipelineError> {
        let mut detected = std::mem::take(&mut self.detected);
        let result = detect(&mut self.detector, block, &mut detected);
        for output in detected.drain(..) {
            match output {
                DetectorOutput::Candidate(candidate) => {
                    self.counters.candidates += 1;
                    let queued = self.judge(*candidate);
                    if let Some(oldest) = self.pending.push(queued) {
                        self.counters.decision_overflow += 1;
                        self.finalize(oldest, OVERFLOW, now, out);
                    }
                }
                DetectorOutput::Rejected(rejected) => {
                    self.counters.detector_rejected += 1;
                    out.push(PipelineEvent::DetectorRejected(rejected));
                }
                DetectorOutput::InputFault(fault) => {
                    self.counters.faults += 1;
                    out.push(PipelineEvent::InputFault(fault));
                }
            }
        }
        self.detected = detected;
        result.map_err(|error| PipelineError::Detector { error })?;
        self.advance(now, veto, out);
        Ok(())
    }

    /// Finalizes, in onset order, every queued candidate whose veto has settled at `now`: emits
    /// its `Decision`, then feeds the gesture and emits its events. Then advances the gesture
    /// clock and prunes the timeline.
    pub fn advance(
        &mut self,
        now: HostTime,
        veto: &mut VetoTimeline,
        out: &mut Vec<PipelineEvent>,
    ) {
        while let Some((queued, verdict)) = self.pending.pop_final(veto, now) {
            self.finalize(queued, verdict, now, out);
        }
        self.gesture.advance_to(now, &mut self.gestures);
        out.extend(self.gestures.drain(..).map(PipelineEvent::Gesture));
        veto.prune(now);
    }

    pub fn detector(&self) -> &Detector {
        &self.detector
    }

    pub fn counters(&self) -> PipelineCounters {
        self.counters
    }

    /// Extracts the descriptor and runs the model; the veto is still to come.
    fn judge(&mut self, candidate: Candidate) -> Queued<Judged> {
        let (model, scratch) = (&self.model, &mut self.scratch);
        let judgement = self
            .extractor
            .extract(&candidate.analysis)
            .map(|descriptor| {
                // INVARIANT: `new` checked that the model's layout is the extractor's layout and
                // that `validate()` passes (K ≥ 2), and `extract` returns only finite descriptors,
                // so none of `decide`'s errors (layout, non-finite, too few zones) can occur.
                let decision = model
                    .decide(&descriptor, scratch)
                    .expect("a compatible model decides every extracted descriptor");
                Scored {
                    descriptor,
                    decision,
                }
            });
        Queued {
            onset: candidate.onset,
            onset_time: candidate.onset_time,
            item: Judged {
                candidate: candidate.id,
                emitted_at: candidate.emitted_at,
                gates: candidate.gates,
                judgement,
            },
        }
    }

    fn finalize(
        &mut self,
        queued: Queued<Judged>,
        verdict: VetoVerdict,
        now: HostTime,
        out: &mut Vec<PipelineEvent>,
    ) {
        let Queued {
            onset,
            onset_time,
            item,
        } = queued;
        let outcome = match (item.judgement, verdict) {
            (Err(error), _) => DecisionOutcome::Invalid { error },
            (
                Ok(Scored {
                    descriptor,
                    decision,
                }),
                VetoVerdict::Vetoed { reason },
            ) => DecisionOutcome::Vetoed {
                reason,
                descriptor,
                scores: match decision {
                    ModelDecision::Accept { scores, .. } | ModelDecision::Reject { scores, .. } => {
                        scores
                    }
                },
            },
            (
                Ok(Scored {
                    descriptor,
                    decision: ModelDecision::Accept { zone, scores },
                }),
                VetoVerdict::Clear,
            ) => DecisionOutcome::Accepted {
                zone,
                descriptor,
                scores,
            },
            (
                Ok(Scored {
                    descriptor,
                    decision: ModelDecision::Reject { reason, scores },
                }),
                VetoVerdict::Clear,
            ) => DecisionOutcome::Rejected {
                reason,
                descriptor,
                scores,
            },
        };
        let gesture_outcome = match &outcome {
            DecisionOutcome::Accepted { zone, .. } => {
                self.counters.accepted += 1;
                CandidateOutcome::Accepted { zone: *zone }
            }
            DecisionOutcome::Vetoed { .. } => {
                self.counters.vetoed += 1;
                CandidateOutcome::Vetoed
            }
            DecisionOutcome::Rejected { .. } => {
                self.counters.rejected += 1;
                CandidateOutcome::Rejected
            }
            DecisionOutcome::Invalid { .. } => {
                self.counters.invalid += 1;
                // SPEC: §1.12's CandidateOutcome has no Invalid; the gesture must still see every
                // detector-passed candidate, and an undescribable one is not a tap, so it counts as
                // Rejected (it interrupts a pending double exactly as a rejection does).
                CandidateOutcome::Rejected
            }
        };
        // An untimed candidate enters the gesture at its decision time (§1.15 flow, step 4).
        let t0 = match onset_time {
            Some(t0) => t0,
            None => now,
        };
        out.push(PipelineEvent::Decision(Box::new(TapDecision {
            candidate: item.candidate,
            onset,
            onset_time,
            emitted_at: item.emitted_at,
            decided_at: now,
            gates: item.gates,
            outcome,
        })));
        self.gesture
            .on_candidate(item.candidate, t0, gesture_outcome, &mut self.gestures);
        out.extend(self.gestures.drain(..).map(PipelineEvent::Gesture));
    }
}

/// `Pipeline::new`'s model compatibility checks, in the order of §1.15.
fn check_model(model: &TrainedModel, config: &PipelineConfig) -> Result<(), PipelineError> {
    let incompatible = |reason| Err(PipelineError::ModelIncompatible { reason });
    if model.front_end_version != FRONT_END_VERSION {
        return incompatible(Incompatibility::FrontEndVersion {
            model: model.front_end_version,
            current: FRONT_END_VERSION,
        });
    }
    let current = DescriptorLayout::from_channels(&config.layout);
    if model.layout != current {
        return incompatible(Incompatibility::Layout {
            model: model.layout,
            current,
        });
    }
    if model.rate != config.format.rate {
        return incompatible(Incompatibility::Rate {
            model: model.rate,
            current: config.format.rate,
        });
    }
    match model.validate() {
        Ok(()) => Ok(()),
        Err(error) => incompatible(Incompatibility::Invalid { error }),
    }
}

/// A count for an error report; counts here are tiny, the saturation only keeps the cast lossless.
fn saturating_u32(n: usize) -> u32 {
    n.min(u32::MAX as usize) as u32
}

// ---- Calibration front end ----

/// The calibration front end (RUST-RT-15): Detector(CalibrationClips) + DescriptorExtractor + the same
/// pending-decision queue, overflow rule and veto resolution as `Pipeline` (one implementation, shared
/// privately by both types). The engine's Detect setup and replay V2 both use it; nothing else composes
/// these parts for calibration.
pub struct CalibrationFrontEnd {
    detector: Detector,
    extractor: DescriptorExtractor,
    pending: DecisionQueue<Extracted>,
    detected: Vec<DetectorOutput>,
}

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

/// A calibration candidate and its live descriptor, before its veto.
#[derive(Debug)]
struct Extracted {
    candidate: Box<Candidate>,
    live: Result<Descriptor, DescriptorError>,
}

impl CalibrationFrontEnd {
    pub fn new(
        format: AudioFormat,
        layout: &ChannelLayout,
        params: &DetectorParams,
    ) -> Result<Self, DetectorError> {
        let detector = Detector::new(format, layout, params, DetectorMode::CalibrationClips)?;
        Ok(CalibrationFrontEnd {
            detector,
            extractor: DescriptorExtractor::new(
                format.rate,
                DescriptorLayout::from_channels(layout),
            ),
            pending: DecisionQueue::new(),
            detected: Vec::with_capacity(DETECTED_CAPACITY),
        })
    }

    pub fn push_block(
        &mut self,
        block: &AudioBlock<'_>,
        now: HostTime,
        veto: &mut VetoTimeline,
        out: &mut Vec<CalibrationInput>,
    ) -> Result<(), DetectorError> {
        let mut detected = std::mem::take(&mut self.detected);
        let result = detect(&mut self.detector, block, &mut detected);
        for output in detected.drain(..) {
            match output {
                DetectorOutput::Candidate(candidate) => {
                    let live = self.extractor.extract(&candidate.analysis);
                    let queued = Queued {
                        onset: candidate.onset,
                        onset_time: candidate.onset_time,
                        item: Extracted { candidate, live },
                    };
                    if let Some(oldest) = self.pending.push(queued) {
                        out.push(calibration_input(oldest.item, OVERFLOW));
                    }
                }
                DetectorOutput::Rejected(rejected) => {
                    out.push(CalibrationInput::Rejected { rejected });
                }
                DetectorOutput::InputFault(fault) => {
                    out.push(CalibrationInput::InputFault { fault });
                }
            }
        }
        self.detected = detected;
        result?;
        self.advance(now, veto, out);
        Ok(())
    }

    /// Finalizes, in onset order, every queued candidate whose veto has settled at `now`, then
    /// prunes the timeline.
    pub fn advance(
        &mut self,
        now: HostTime,
        veto: &mut VetoTimeline,
        out: &mut Vec<CalibrationInput>,
    ) {
        while let Some((queued, verdict)) = self.pending.pop_final(veto, now) {
            out.push(calibration_input(queued.item, verdict));
        }
        veto.prune(now);
    }

    pub fn detector(&self) -> &Detector {
        &self.detector
    }
}

fn calibration_input(extracted: Extracted, veto: VetoVerdict) -> CalibrationInput {
    let Extracted { candidate, live } = extracted;
    match live {
        Ok(live) => CalibrationInput::Candidate {
            candidate,
            live,
            veto,
        },
        Err(error) => CalibrationInput::Invalid { candidate, error },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::BlockTime;
    use crate::desk::{DeskConfig, Distractor, ModalDesk, TapSpec};
    use crate::profile::ZoneId;
    use crate::training::{train, TapGroup, TrainingParams, TrainingSet};
    use crate::units::{ChannelCount, Decibels, Milliseconds, TapIndex};
    use crate::veto::{InputEvent, InputKind, OutputId, SelfOutputEvent, SelfOutputKind};
    use std::sync::atomic::AtomicBool;
    use std::sync::OnceLock;

    /// 16 kHz: a frame is exactly 625 host ticks, so the block timestamps of every block size map
    /// each sample to the same host time and P2 can compare decisions exactly.
    const HZ: u32 = 16_000;
    const TICKS_PER_FRAME: i64 = HostTime::TICKS_PER_SECOND / HZ as i64;
    const CHANNELS: u16 = 2;
    const ZONES: usize = 4;
    const TRAIN_TAPS_PER_ZONE: usize = 6;
    /// One desk for training and for every scene: its seed fixes the plate's modal phases.
    const DESK_SEED: u64 = 11;
    /// Detector warm-up (200 ms) plus margin before the first event of every scene.
    const LEAD_MS: f64 = 400.0;

    fn rate() -> SampleRate {
        SampleRate::new(HZ).expect("valid rate")
    }

    fn format() -> AudioFormat {
        AudioFormat {
            rate: rate(),
            channels: ChannelCount::new(CHANNELS).expect("valid channel count"),
        }
    }

    fn layout() -> ChannelLayout {
        ChannelLayout::all_real(ChannelCount::new(CHANNELS).expect("valid channel count"))
    }

    fn frames(ms: f64) -> usize {
        rate().samples_in(Milliseconds::new(ms))
    }

    fn host(frame: u64) -> HostTime {
        HostTime(frame as i64 * TICKS_PER_FRAME)
    }

    fn ms_of(d: crate::units::HostDuration) -> f64 {
        d.as_ms().get()
    }

    #[derive(Debug, Clone, Copy)]
    enum Sound {
        Tap(usize),
        Keys(f64),
    }

    fn tap_spec(zone: usize) -> TapSpec {
        let (x, y) = ModalDesk::zone_centres(ZONES)[zone];
        TapSpec {
            x,
            y,
            force: Decibels::new(-20.0),
            contact: Milliseconds::new(1.0),
            position_jitter_m: 0.01,
        }
    }

    /// The desk floor for `len_ms`, with each (onset ms, sound) added at its onset.
    fn render(desk: &mut ModalDesk, len_ms: f64, events: &[(f64, Sound)]) -> Vec<f32> {
        let ch = usize::from(CHANNELS);
        let mut x = desk.floor(frames(len_ms));
        for &(at, sound) in events {
            let clip = match sound {
                Sound::Tap(zone) => desk.tap(&tap_spec(zone), 0),
                Sound::Keys(gap) => desk.distractor(
                    &Distractor::KeystrokePair {
                        gap: Milliseconds::new(gap),
                    },
                    0,
                ),
            };
            let start = frames(at) * ch;
            for (v, c) in x[start..].iter_mut().zip(&clip) {
                *v += c;
            }
        }
        x
    }

    fn desk() -> ModalDesk {
        ModalDesk::new(DeskConfig::default_for(rate(), DESK_SEED)).expect("valid desk")
    }

    /// The test model: Detector(Live) + extractor on modal-desk taps, one variant per tap.
    fn model() -> Arc<TrainedModel> {
        static MODEL: OnceLock<Arc<TrainedModel>> = OnceLock::new();
        MODEL.get_or_init(|| Arc::new(fit_model())).clone()
    }

    fn fit_model() -> TrainedModel {
        let taps: Vec<(f64, Sound)> = (0..ZONES * TRAIN_TAPS_PER_ZONE)
            .map(|i| (LEAD_MS + 500.0 * i as f64, Sound::Tap(i % ZONES)))
            .collect();
        let len = LEAD_MS + 500.0 * taps.len() as f64 + 300.0;
        let x = render(&mut desk(), len, &taps);
        let mut detector = Detector::new(
            format(),
            &layout(),
            &DetectorParams::DEFAULT,
            DetectorMode::Live,
        )
        .expect("valid detector");
        let mut extractor =
            DescriptorExtractor::new(rate(), DescriptorLayout::from_channels(&layout()));
        let mut out = Vec::new();
        let ch = usize::from(CHANNELS);
        for (i, chunk) in x.chunks(160 * ch).enumerate() {
            let block = AudioBlock::new(
                format(),
                SampleIndex((i * 160) as u64),
                chunk,
                BlockTime::Untimed,
                false,
            )
            .expect("whole frames");
            detector.push(&block, &mut out).expect("matching format");
        }
        let mut groups: Vec<TapGroup> = Vec::new();
        for output in &out {
            if let DetectorOutput::Candidate(c) = output {
                let onset = c.onset.0 as i64;
                let (i, _) = taps
                    .iter()
                    .enumerate()
                    .find(|(_, (at, _))| (onset - frames(*at) as i64).abs() < frames(15.0) as i64)
                    .expect("every candidate is one of the taps");
                let descriptor = extractor.extract(&c.analysis).expect("a clean tap");
                groups.push(TapGroup {
                    zone: ZoneIndex((i % ZONES) as u16),
                    tap: TapIndex(i as u32),
                    variants: vec![descriptor],
                });
            }
        }
        assert_eq!(
            groups.len(),
            taps.len(),
            "every training tap is detected once"
        );
        let set = TrainingSet {
            layout: DescriptorLayout::from_channels(&layout()),
            rate: rate(),
            zone_ids: (0..ZONES)
                .map(|z| ZoneId::from_uuid(uuid::Uuid::from_u128(z as u128 + 1)))
                .collect(),
            seed: 0,
            taps: groups,
            negatives: Vec::new(),
        };
        train(
            &set,
            &TrainingParams::DEFAULT,
            &AtomicBool::new(false),
            &mut |_| {},
        )
        .expect("the desk taps train")
    }

    fn config(triggers: Vec<Trigger>) -> PipelineConfig {
        PipelineConfig {
            format: format(),
            layout: layout(),
            triggers,
            detector: DetectorParams::DEFAULT,
            veto: VetoParams::DEFAULT,
            gesture: GestureParams::DEFAULT,
        }
    }

    fn singles() -> Vec<Trigger> {
        vec![Trigger::Single; ZONES]
    }

    fn pipeline(triggers: Vec<Trigger>) -> Pipeline {
        Pipeline::new(model(), config(triggers)).expect("compatible model")
    }

    /// Device packets are 10 ms: a block that starts a packet carries its timestamp, any other
    /// block is untimed. (The ClockMap keeps 64 anchors, so stamping every 1-frame block would
    /// leave no anchor at or before an onset 40 ms back.) At 16 kHz every block size then maps
    /// each sample to the same host time.
    const PACKET: u64 = 160;

    /// Splits interleaved `x` (frame 0 at `start`) into blocks of `block` frames and calls
    /// `push(block, now)` for each, where `now` is the host time of the block's last frame (the
    /// replay clock).
    fn blocks(
        x: &[f32],
        start: u64,
        block: usize,
        mut push: impl FnMut(&AudioBlock<'_>, HostTime),
    ) {
        let ch = usize::from(CHANNELS);
        for (i, chunk) in x.chunks(block * ch).enumerate() {
            let first = start + (i * block) as u64;
            let len = (chunk.len() / ch) as u64;
            let time = if first.is_multiple_of(PACKET) {
                BlockTime::Timestamped {
                    host_time: host(first),
                }
            } else {
                BlockTime::Untimed
            };
            let b = AudioBlock::new(format(), SampleIndex(first), chunk, time, false)
                .expect("whole frames");
            push(&b, host(first + len - 1));
        }
    }

    /// Streams `x` through `p`. Before each block, every input event up to its `now` goes to
    /// the timeline.
    fn stream(
        p: &mut Pipeline,
        veto: &mut VetoTimeline,
        x: &[f32],
        start: u64,
        block: usize,
        inputs: &[InputEvent],
    ) -> Vec<PipelineEvent> {
        let mut out = Vec::new();
        let mut next = 0;
        blocks(x, start, block, |b, now| {
            while next < inputs.len() && inputs[next].time <= now {
                veto.push_input(inputs[next]);
                next += 1;
            }
            p.push_block(b, now, veto, &mut out)
                .expect("matching format");
        });
        out
    }

    fn decisions(events: &[PipelineEvent]) -> Vec<TapDecision> {
        events
            .iter()
            .filter_map(|e| match e {
                PipelineEvent::Decision(d) => Some((**d).clone()),
                PipelineEvent::Gesture(_)
                | PipelineEvent::DetectorRejected(_)
                | PipelineEvent::InputFault(_) => None,
            })
            .collect()
    }

    fn gestures(events: &[PipelineEvent]) -> Vec<GestureEvent> {
        events
            .iter()
            .filter_map(|e| match e {
                PipelineEvent::Gesture(g) => Some(g.clone()),
                PipelineEvent::Decision(_)
                | PipelineEvent::DetectorRejected(_)
                | PipelineEvent::InputFault(_) => None,
            })
            .collect()
    }

    fn fires(events: &[PipelineEvent]) -> usize {
        gestures(events)
            .iter()
            .filter(|g| matches!(g, GestureEvent::Fire { .. }))
            .count()
    }

    fn tap_events(onsets_ms: &[(f64, usize)]) -> Vec<(f64, Sound)> {
        onsets_ms
            .iter()
            .map(|&(at, zone)| (at, Sound::Tap(zone)))
            .collect()
    }

    #[test]
    fn p2_block_sizes_give_identical_decisions_and_gestures() {
        // Zone 2 is a double-tap zone: a pair 200 ms apart fires, a lone tap arms and expires.
        // Taps are ≥ 500 ms apart elsewhere, so no burst hold starts.
        let triggers = vec![
            Trigger::Single,
            Trigger::Single,
            Trigger::Double,
            Trigger::Single,
        ];
        let events = tap_events(&[
            (500.0, 0),
            (1100.0, 1),
            (1700.0, 2),
            (1900.0, 2),
            (2600.0, 3),
            (3200.0, 2),
            (3900.0, 0),
        ]);
        let x = render(&mut desk(), 4800.0, &events);
        let mut runs = Vec::new();
        for block in [1usize, 480, 4096] {
            let mut veto = VetoTimeline::new(VetoParams::DEFAULT);
            let mut p = pipeline(triggers.clone());
            let out = stream(&mut p, &mut veto, &x, 0, block, &[]);
            let block_ms = block as f64 * 1000.0 / f64::from(HZ);
            let mut ds = decisions(&out);
            for d in &mut ds {
                // decided_at is the `now` of the call that finalized it: the first block end at or
                // after t0 + input_after + settle (45 ms). Only it may depend on the block size.
                let t0 = d.onset_time.expect("timestamped blocks");
                let latency = ms_of(d.decided_at.since(t0));
                assert!(
                    (45.0..45.0 + block_ms + 0.1).contains(&latency),
                    "block {block}: latency {latency} ms"
                );
                d.decided_at = HostTime(0);
            }
            runs.push((block, ds, gestures(&out), p.counters()));
        }
        let (_, base_d, base_g, base_c) = &runs[0];
        assert_eq!(base_d.len(), events.len(), "every tap is a decision");
        let accepted = base_d
            .iter()
            .filter(|d| matches!(d.outcome, DecisionOutcome::Accepted { .. }))
            .count();
        assert!(
            accepted >= events.len() - 1,
            "accepted {accepted} of {}",
            events.len()
        );
        assert!(base_g
            .iter()
            .any(|g| matches!(g, GestureEvent::Fire { first: Some(_), .. })));
        assert!(base_g
            .iter()
            .any(|g| matches!(g, GestureEvent::DoubleExpired { .. })));
        for (block, d, g, c) in &runs[1..] {
            assert_eq!(d, base_d, "block {block}: decisions differ");
            assert_eq!(g, base_g, "block {block}: gestures differ");
            assert_eq!(c, base_c, "block {block}: counters differ");
        }
    }

    #[test]
    fn p3_keystrokes_with_matching_input_events_never_fire() {
        let pairs: Vec<f64> = (0..8).map(|i| 500.0 + 600.0 * f64::from(i)).collect();
        let gap = 90.0;
        let events: Vec<(f64, Sound)> = pairs.iter().map(|&at| (at, Sound::Keys(gap))).collect();
        let x = render(&mut desk(), 500.0 + 600.0 * 8.0 + 400.0, &events);
        let mut inputs = Vec::new();
        for &at in &pairs {
            inputs.push(InputEvent {
                time: host(frames(at) as u64),
                kind: InputKind::KeyDown,
            });
            inputs.push(InputEvent {
                time: host(frames(at + gap) as u64),
                kind: InputKind::KeyUp,
            });
        }
        let mut veto = VetoTimeline::new(VetoParams::DEFAULT);
        let mut p = pipeline(singles());
        let out = stream(&mut p, &mut veto, &x, 0, 160, &inputs);
        let ds = decisions(&out);
        assert!(!ds.is_empty(), "the keystrokes produce candidates");
        for d in &ds {
            assert!(
                matches!(
                    d.outcome,
                    DecisionOutcome::Vetoed {
                        reason: VetoReason::Input { .. },
                        ..
                    }
                ),
                "{:?}",
                d.outcome
            );
        }
        assert_eq!(fires(&out), 0);
        assert_eq!(p.counters().vetoed, ds.len() as u64);
    }

    #[test]
    fn p4_tap_during_open_speech_is_vetoed() {
        let events = tap_events(&[(600.0, 0), (1200.0, 1), (1800.0, 3)]);
        let x = render(&mut desk(), 2400.0, &events);
        let mut veto = VetoTimeline::new(VetoParams::DEFAULT);
        veto.self_output(SelfOutputEvent::Began {
            id: OutputId(1),
            kind: SelfOutputKind::Speech,
            at: host(frames(300.0) as u64),
        });
        let mut p = pipeline(singles());
        let out = stream(&mut p, &mut veto, &x, 0, 160, &[]);
        let ds = decisions(&out);
        assert_eq!(ds.len(), events.len());
        for d in &ds {
            assert!(
                matches!(
                    d.outcome,
                    DecisionOutcome::Vetoed {
                        reason: VetoReason::SelfOutput {
                            kind: SelfOutputKind::Speech
                        },
                        ..
                    }
                ),
                "{:?}",
                d.outcome
            );
        }
        assert_eq!(fires(&out), 0);
    }

    /// −60 dBFS white noise on both channels with a sharp click every `spacing_ms` from 300 ms.
    fn storm(len_ms: f64, spacing_ms: f64) -> Vec<f32> {
        use crate::detector::testkit::{scene, TapKind};
        let mut events = Vec::new();
        let mut at = 300.0;
        while at + 100.0 < len_ms {
            events.push((at, TapKind::Click, 40.0));
            at += spacing_ms;
        }
        scene(HZ, CHANNELS, 5, len_ms, &events).0
    }

    /// The storm's click spacing.
    // SPEC: P5 and P8 say "one every 30 ms". At 30 ms the next click lands inside the anchored
    // analysis span [t0, t0 + 40 ms) and the decay gate rejects every candidate (NotDecaying: 1
    // candidate in 3 s at 30 and 35 ms, 72 at 40 ms), so the storm uses 40 ms, the shortest
    // spacing whose clicks pass the gates.
    const STORM_SPACING_MS: f64 = 40.0;

    #[test]
    fn p5_candidate_storm_overflows_into_vetoes_without_growth() {
        // One click every 40 ms for 10 s, pushed in 1 s blocks (a backlog drained at once): far
        // more candidates arrive per call than the queue holds.
        let x = storm(10_300.0, STORM_SPACING_MS);
        let mut veto = VetoTimeline::new(VetoParams::DEFAULT);
        let mut p = pipeline(singles());
        let capacity = p.pending.items.capacity();
        let mut out = Vec::new();
        blocks(&x, 0, frames(1000.0), |b, now| {
            p.push_block(b, now, &mut veto, &mut out)
                .expect("matching format");
            assert!(p.pending.items.len() <= MAX_PENDING_DECISIONS);
            assert_eq!(
                p.pending.items.capacity(),
                capacity,
                "the queue never grows"
            );
        });
        let c = p.counters();
        assert!(c.candidates > 200, "{c:?}");
        assert!(c.decision_overflow > 0, "{c:?}");
        let overflowed = decisions(&out)
            .iter()
            .filter(|d| {
                matches!(
                    d.outcome,
                    DecisionOutcome::Vetoed {
                        reason: VetoReason::DecisionOverflow,
                        ..
                    }
                )
            })
            .count() as u64;
        let invalid_overflow = c.decision_overflow - overflowed;
        assert!(invalid_overflow <= c.invalid);
        assert_eq!(
            c.accepted + c.vetoed + c.rejected + c.invalid + p.pending.items.len() as u64,
            c.candidates
        );
        // Decisions leave in onset order.
        let onsets: Vec<u64> = decisions(&out).iter().map(|d| d.onset.0).collect();
        assert!(onsets.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn p7_timeline_survives_a_reconfigure() {
        let mut veto = VetoTimeline::new(VetoParams::DEFAULT);
        let mut d = desk();
        // The first pipeline runs 1 s of floor; the speech begins during it and never ends.
        let first = render(&mut d, 1000.0, &[]);
        let mut a = pipeline(singles());
        stream(&mut a, &mut veto, &first, 0, 160, &[]);
        veto.self_output(SelfOutputEvent::Began {
            id: OutputId(7),
            kind: SelfOutputKind::Speech,
            at: host(frames(900.0) as u64),
        });
        drop(a);
        // A reconfigure: a new pipeline, the same timeline, the stream continuing.
        let start = frames(1000.0) as u64;
        let second = render(&mut d, 1500.0, &tap_events(&[(LEAD_MS + 200.0, 2)]));
        let mut b = pipeline(singles());
        let out = stream(&mut b, &mut veto, &second, start, 160, &[]);
        let ds = decisions(&out);
        assert_eq!(ds.len(), 1);
        assert!(
            matches!(
                ds[0].outcome,
                DecisionOutcome::Vetoed {
                    reason: VetoReason::SelfOutput {
                        kind: SelfOutputKind::Speech
                    },
                    ..
                }
            ),
            "{:?}",
            ds[0].outcome
        );
    }

    fn calibrate(
        x: &[f32],
        block: usize,
        stale_now: bool,
    ) -> (Vec<CalibrationInput>, CalibrationFrontEnd) {
        let mut fe = CalibrationFrontEnd::new(format(), &layout(), &DetectorParams::DEFAULT)
            .expect("valid front end");
        let mut veto = VetoTimeline::new(VetoParams::DEFAULT);
        let capacity = fe.pending.items.capacity();
        let mut out = Vec::new();
        blocks(x, 0, block, |b, now| {
            let now = if stale_now { HostTime(0) } else { now };
            fe.push_block(b, now, &mut veto, &mut out)
                .expect("matching format");
            assert!(fe.pending.items.len() <= MAX_PENDING_DECISIONS);
            assert_eq!(
                fe.pending.items.capacity(),
                capacity,
                "the queue never grows"
            );
        });
        (out, fe)
    }

    #[test]
    fn p8_calibration_front_end_is_block_size_invariant_and_bounded() {
        let events = tap_events(&[(500.0, 0), (1000.0, 1), (1500.0, 2), (2000.0, 3)]);
        let x = render(&mut desk(), 2600.0, &events);
        let (base, _) = calibrate(&x, 1, false);
        let candidates = base
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    CalibrationInput::Candidate {
                        veto: VetoVerdict::Clear,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(candidates, events.len());
        for block in [480usize, 4096] {
            let (out, _) = calibrate(&x, block, false);
            assert_eq!(out, base, "block {block}");
        }

        // The storm, with `now` stuck before every candidate: nothing settles, so the queue
        // fills and each further candidate finalizes the oldest as DecisionOverflow.
        let x = storm(3_300.0, STORM_SPACING_MS);
        let (out, fe) = calibrate(&x, 160, true);
        let overflowed = out
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    CalibrationInput::Candidate {
                        veto: VetoVerdict::Vetoed {
                            reason: VetoReason::DecisionOverflow
                        },
                        ..
                    } | CalibrationInput::Invalid { .. }
                )
            })
            .count();
        assert!(overflowed > 50, "{overflowed}");
        assert_eq!(fe.pending.items.len(), MAX_PENDING_DECISIONS);
        let onsets: Vec<u64> = out
            .iter()
            .filter_map(|c| match c {
                CalibrationInput::Candidate { candidate, .. }
                | CalibrationInput::Invalid { candidate, .. } => Some(candidate.onset.0),
                CalibrationInput::Rejected { .. } | CalibrationInput::InputFault { .. } => None,
            })
            .collect();
        assert!(onsets.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn new_checks_model_compatibility_and_trigger_count() {
        let m = model();
        let mut stale = (*m).clone();
        stale.front_end_version = FRONT_END_VERSION + 1;
        assert_eq!(
            Pipeline::new(Arc::new(stale), config(singles())).err(),
            Some(PipelineError::ModelIncompatible {
                reason: Incompatibility::FrontEndVersion {
                    model: FRONT_END_VERSION + 1,
                    current: FRONT_END_VERSION
                }
            })
        );
        let mono = PipelineConfig {
            format: AudioFormat {
                rate: rate(),
                channels: ChannelCount::new(1).expect("valid"),
            },
            layout: ChannelLayout::all_real(ChannelCount::new(1).expect("valid")),
            ..config(singles())
        };
        assert_eq!(
            Pipeline::new(m.clone(), mono).err(),
            Some(PipelineError::ModelIncompatible {
                reason: Incompatibility::Layout {
                    model: DescriptorLayout::Stereo { a: 0, c: 1 },
                    current: DescriptorLayout::Mono { a: 0 }
                }
            })
        );
        let other_rate = SampleRate::new(48_000).expect("valid");
        let fast = PipelineConfig {
            format: AudioFormat {
                rate: other_rate,
                channels: format().channels,
            },
            ..config(singles())
        };
        assert_eq!(
            Pipeline::new(m.clone(), fast).err(),
            Some(PipelineError::ModelIncompatible {
                reason: Incompatibility::Rate {
                    model: rate(),
                    current: other_rate
                }
            })
        );
        let mut broken = (*m).clone();
        broken.zone_means.pop();
        assert!(matches!(
            Pipeline::new(Arc::new(broken), config(singles())).err(),
            Some(PipelineError::ModelIncompatible {
                reason: Incompatibility::Invalid { .. }
            })
        ));
        assert_eq!(
            Pipeline::new(m, config(vec![Trigger::Single; ZONES - 1])).err(),
            Some(PipelineError::TriggerCountMismatch {
                zones: ZONES as u32,
                triggers: ZONES as u32 - 1
            })
        );
    }

    #[test]
    fn untimed_candidates_are_vetoed_for_the_clock_at_once() {
        let events = tap_events(&[(600.0, 1)]);
        let x = render(&mut desk(), 1200.0, &events);
        let mut veto = VetoTimeline::new(VetoParams::DEFAULT);
        let mut p = pipeline(singles());
        let mut out = Vec::new();
        let ch = usize::from(CHANNELS);
        for (i, chunk) in x.chunks(160 * ch).enumerate() {
            let first = (i * 160) as u64;
            let b = AudioBlock::new(
                format(),
                SampleIndex(first),
                chunk,
                BlockTime::Untimed,
                false,
            )
            .expect("whole frames");
            p.push_block(&b, host(first + 159), &mut veto, &mut out)
                .expect("matching format");
        }
        let ds = decisions(&out);
        assert_eq!(ds.len(), 1);
        assert_eq!(ds[0].onset_time, None);
        assert!(matches!(
            ds[0].outcome,
            DecisionOutcome::Vetoed {
                reason: VetoReason::ClockUnavailable,
                ..
            }
        ));
        // Decided in the call that emitted it: the emission latency, not the veto hold.
        let emitted = ds[0].emitted_at.0;
        assert_eq!(ds[0].decided_at, host(emitted - emitted % 160 + 159));
        assert!(ds[0].descriptor().is_some() && ds[0].scores().is_some());
    }

    #[test]
    fn decision_serde_round_trips() {
        let events = tap_events(&[(600.0, 3)]);
        let x = render(&mut desk(), 1200.0, &events);
        let mut veto = VetoTimeline::new(VetoParams::DEFAULT);
        let mut p = pipeline(singles());
        let out = stream(&mut p, &mut veto, &x, 0, 160, &[]);
        let d = decisions(&out).pop().expect("one decision");
        let json = serde_json::to_string(&d).expect("serializes");
        let back: TapDecision = serde_json::from_str(&json).expect("parses");
        // serde_json without its `float_roundtrip` feature may parse an f64 1 ULP off, so the
        // f64 scores agree to rounding; everything else is exact.
        assert_eq!(
            (
                back.candidate,
                back.onset,
                back.onset_time,
                back.emitted_at,
                back.decided_at
            ),
            (
                d.candidate,
                d.onset,
                d.onset_time,
                d.emitted_at,
                d.decided_at
            )
        );
        assert_eq!(back.descriptor(), d.descriptor());
        let (a, b) = (back.scores().expect("scored"), d.scores().expect("scored"));
        assert_eq!((a.best, a.second), (b.best, b.second));
        for (x, y) in a.d2.iter().zip(&b.d2).chain([(&a.margin, &b.margin)]) {
            assert!((x - y).abs() <= 1e-12 * y.abs().max(1.0), "{x} vs {y}");
        }
        assert_eq!(
            std::mem::discriminant(&back.outcome),
            std::mem::discriminant(&d.outcome)
        );
    }
}
