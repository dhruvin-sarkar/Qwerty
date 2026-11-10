//! Evaluation v2: assessment of a trained model on held-out taps and replayed audio (§1.17).
//!
//! An `EvaluationSession` prompts each zone in a balanced random order, resolves every prompt from
//! the first relevant decision or detector rejection inside its window, then runs the negative
//! phases (the background check) and counts what would have fired. `finish` turns the run into a
//! v2 `EvaluationReport`. Replay drives the same session from a `Scripted` schedule.

use crate::audio::DeviceIdentity;
use crate::detector::{RejectReason, RejectedCandidate};
use crate::model::{BackgroundRegime, ModelRejectReason};
use crate::pipeline::{DecisionOutcome, TapDecision};
use crate::profile::{parse_versioned, PersistenceError, ProfileId, ZoneId};
use crate::recording::NegativeClass;
use crate::rng::Rng64;
use crate::training::NegativeCount;
use crate::units::{HostDuration, HostTime, Milliseconds, Seconds, ZoneIndex};
use crate::veto::VetoReason;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const EVALUATION_SCHEMA_VERSION: u32 = 2;
pub const DEFAULT_PROMPTS_PER_ZONE: u32 = 15;
pub const PROMPT_WINDOW: Milliseconds = Milliseconds::new(3000.0);
pub const PROMPT_GAP: Milliseconds = Milliseconds::new(700.0);
pub const LATENCY_BIN: Milliseconds = Milliseconds::new(5.0); // histogram 0..200 ms in 40 bins + 1 overflow bin
pub const MARGIN_BIN: f64 = 1.0; // margin histogram over [0, 20) in 20 bins + 1 overflow bin: 0-5 × the default τ_m = 4 (PARITY-10)
/// The ACCEPTANCE.md accuracy target. Each report stores the target in force when it was measured, so
/// history stays honest if the target changes (PARITY-12, XR-UI-07).
pub const ACCURACY_TARGET: f64 = 0.92;

/// Latency histogram: 40 bins of LATENCY_BIN over [0, 200) ms, then one overflow bin.
const LATENCY_BINS: usize = 40;
/// Margin histogram: 20 bins of MARGIN_BIN over [0, 20), then one overflow bin.
const MARGIN_BINS: usize = 20;
/// Confusion columns after the K zones: Rejected, Vetoed, DetectorRejected, Invalid, Missed.
const OUTCOME_COLUMNS: usize = 5;
/// The schema of the v1 reports `peek_legacy_report` reads.
const LEGACY_SCHEMA_VERSION: u32 = 1;
// SPEC: `to_json` takes no file name, yet `PersistenceError::InvalidData` names one; a report that
// fails validation before it has a file is named by what it is.
const UNSAVED_REPORT: &str = "evaluation report";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZoneRef {
    pub id: ZoneId,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NegativePhase {
    pub class: NegativeClass,
    pub duration: Seconds,
} // NegativeClass: recording.rs (§1.18.2)

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PromptSchedule {
    /// live: balanced, with no zone twice in a row (PARITY-9), drawn position by position from Rng64(seed). With R prompts
    /// left (this one included), a zone whose remaining count c has 2·c > R must be taken: at most one zone can, and it is
    /// never the previous one. Otherwise u = below(Σc over the zones ≠ the previous one) picks, walking those zones in
    /// index order, the first whose running sum of c exceeds u. Equal quotas with K ≥ 2 never force a repeat (1000 seeds
    /// × K ∈ {2, 4, 6, 8}: 0 repeats, exact balance; `scratch/p01_map_sched.out`). Prompts are opened by tick().
    Generated {
        prompts_per_zone: u32,
        seed: u64,
        negatives: Vec<NegativePhase>,
    },
    /// replay: prompts and negative phases exactly as recorded
    Scripted {
        prompts: Vec<ScriptedPrompt>,
        negatives: Vec<ScriptedNegative>,
    },
}

/// reopened: replaces the previous prompt's outcome, as undo_last did live
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScriptedPrompt {
    pub zone: ZoneIndex,
    pub opened_at: HostTime,
    pub deadline: HostTime,
    pub reopened: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScriptedNegative {
    pub class: NegativeClass,
    pub start: HostTime,
    pub end: HostTime,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PromptOutcome {
    Correct {
        latency: Option<Milliseconds>,
    },
    WrongZone {
        got: ZoneIndex,
        latency: Option<Milliseconds>,
    },
    Rejected {
        reason: ModelRejectReason,
    },
    Vetoed {
        reason: VetoReason,
    },
    DetectorRejected {
        reason: RejectReason,
    },
    Invalid,
    Missed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EvaluationStep {
    Prompt {
        index: u32,
        total: u32,
        zone: ZoneIndex,
        deadline: HostTime,
    },
    Gap {
        until: HostTime,
    },
    Negative {
        index: u32,
        class: NegativeClass,
        ends_at: HostTime,
        false_accepts: u32,
    },
    Finished,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EvaluationEvent {
    PromptOpened {
        index: u32,
        zone: ZoneIndex,
    },
    PromptResolved {
        index: u32,
        zone: ZoneIndex,
        outcome: PromptOutcome,
    },
    PromptReopened {
        index: u32,
        zone: ZoneIndex,
    }, // undo_last (XR-UI-05)
    NegativeStarted {
        index: u32,
        class: NegativeClass,
    },
    NegativeEnded {
        index: u32,
    },
    Finished,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationView {
    pub step: EvaluationStep,
    pub resolved: u32,
    pub total_prompts: u32,
    pub correct: u32,
    pub last: Option<PromptOutcome>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AssessmentError {
    NoZones,
    ZeroPrompts,
    ScriptUnordered,
    NotFinished,
    NothingToUndo,
}

/// Enough to explain a result and to compare two reports (PARITY-13).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportMeta {
    pub profile_id: ProfileId,
    /// the opened stream measured with (endpoint id, name, format, capture mode); it matches the
    /// profile's bound identity, or evaluation could not have started
    pub device: DeviceIdentity,
    pub front_end_version: u32,
    /// = the profile's `calibrated_at`. It identifies the calibration: a report is current iff it
    /// equals the profile's current `calibrated_at` (no separate id)
    pub model_trained_at: DateTime<Utc>,
    /// calibration taps per zone, model order (TrainingSummary.zones)
    pub zone_taps: Vec<u32>,
    /// TrainingSummary.negatives
    pub negatives: Vec<NegativeCount>,
    /// TrainingSummary.background_regime
    pub background_regime: BackgroundRegime,
    /// Config at evaluation start
    pub input_veto_enabled: bool,
}

// ---- Session ----

/// One prompt of the run, by index: its zone and its outcome so far.
#[derive(Debug, Clone, PartialEq)]
struct PromptResult {
    outcome: PromptOutcome,
    margin: Option<f64>,
}

/// A negative phase that has started.
#[derive(Debug, Clone, PartialEq)]
struct NegativeTally {
    class: NegativeClass,
    start: HostTime,
    end: HostTime,
    duration: Seconds,
    candidates: u32,
    false_accepts: u32,
    by_zone: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Phase {
    /// Waiting until `until`; then the next prompt, the next negative phase, or Finished.
    Gap {
        until: HostTime,
    },
    /// The window is [opened_at, deadline). A prompt reopened by undo opens PROMPT_GAP after the
    /// undo, so `opened_at` may still be ahead.
    Prompt {
        index: u32,
        zone: ZoneIndex,
        opened_at: HostTime,
        deadline: HostTime,
    },
    /// `tallies[index]` is running.
    Negative {
        index: u32,
    },
    Finished,
}

/// One recorded prompt opening, with the prompt index it stands for.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Entry {
    index: u32,
    zone: ZoneIndex,
    opened_at: HostTime,
    deadline: HostTime,
    reopened: bool,
}

impl Entry {
    /// When this entry takes effect: a fresh prompt at its opening, a reopened one at the undo
    /// that opened it, PROMPT_GAP earlier (undo_last reopens after PROMPT_GAP).
    fn start(&self) -> HostTime {
        if self.reopened {
            before(self.opened_at, gap())
        } else {
            self.opened_at
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Plan {
    Generated {
        /// The next prompt index never opened (rewound when an open prompt is withdrawn).
        next: u32,
        negatives: Vec<NegativePhase>,
    },
    Scripted {
        entries: Vec<Entry>,
        /// The next entry to take effect.
        next: usize,
        negatives: Vec<ScriptedNegative>,
    },
}

pub struct EvaluationSession {
    zones: Vec<ZoneRef>,
    prompts_per_zone: u32,
    /// Zone of each prompt index.
    prompt_zones: Vec<ZoneIndex>,
    plan: Plan,
    phase: Phase,
    results: Vec<Option<PromptResult>>,
    /// The last prompt to resolve, while its outcome may still be undone.
    undoable: Option<u32>,
    last: Option<PromptOutcome>,
    undone: u32,
    tallies: Vec<NegativeTally>,
}

impl EvaluationSession {
    pub fn new(
        zones: Vec<ZoneRef>,
        schedule: PromptSchedule,
        started: HostTime,
    ) -> Result<Self, AssessmentError> {
        if zones.is_empty() {
            return Err(AssessmentError::NoZones);
        }
        let k = zones.len();
        let (prompt_zones, prompts_per_zone, plan, first) = match schedule {
            PromptSchedule::Generated {
                prompts_per_zone,
                seed,
                negatives,
            } => {
                if prompts_per_zone == 0 {
                    return Err(AssessmentError::ZeroPrompts);
                }
                (
                    generated_order(k, prompts_per_zone, seed),
                    prompts_per_zone,
                    Plan::Generated { next: 0, negatives },
                    after(started, gap()),
                )
            }
            PromptSchedule::Scripted { prompts, negatives } => {
                let entries = script_entries(k, &prompts)?;
                check_negatives(&prompts, &negatives)?;
                let total = entries.iter().map(|e| e.index + 1).max();
                let total = match total {
                    Some(total) => total,
                    None => return Err(AssessmentError::ZeroPrompts),
                };
                let mut prompt_zones = vec![ZoneIndex(0); total as usize];
                for e in &entries {
                    prompt_zones[e.index as usize] = e.zone;
                }
                // SPEC: a script carries no prompts_per_zone; the report records the largest
                // per-zone count, which is the quota for the balanced schedules a live run records.
                let mut per_zone = vec![0u32; k];
                for z in &prompt_zones {
                    per_zone[usize::from(z.0)] += 1;
                }
                let per_zone = per_zone.into_iter().fold(0, u32::max);
                let first = entries[0].start();
                (
                    prompt_zones,
                    per_zone,
                    Plan::Scripted {
                        entries,
                        next: 0,
                        negatives,
                    },
                    first,
                )
            }
        };
        Ok(EvaluationSession {
            zones,
            prompts_per_zone,
            results: vec![None; prompt_zones.len()],
            prompt_zones,
            plan,
            phase: Phase::Gap { until: first },
            undoable: None,
            last: None,
            undone: 0,
            tallies: Vec::new(),
        })
    }

    pub fn view(&self) -> EvaluationView {
        let step = match self.phase {
            Phase::Gap { until } => EvaluationStep::Gap { until },
            Phase::Prompt {
                index,
                zone,
                deadline,
                ..
            } => EvaluationStep::Prompt {
                index,
                total: self.total(),
                zone,
                deadline,
            },
            Phase::Negative { index } => {
                let t = &self.tallies[index as usize];
                EvaluationStep::Negative {
                    index,
                    class: t.class.clone(),
                    ends_at: t.end,
                    false_accepts: t.false_accepts,
                }
            }
            Phase::Finished => EvaluationStep::Finished,
        };
        let resolved = self.results.iter().flatten().count() as u32;
        let correct = self
            .results
            .iter()
            .flatten()
            .filter(|r| matches!(r.outcome, PromptOutcome::Correct { .. }))
            .count() as u32;
        EvaluationView {
            step,
            resolved,
            total_prompts: self.total(),
            correct,
            last: self.last.clone(),
        }
    }

    /// Advances the schedule to `now`: opens prompts, resolves expired ones as Missed, starts and
    /// ends negative phases, and finishes. Every transition happens at its scheduled time, so a
    /// late tick catches up exactly.
    pub fn tick(&mut self, now: HostTime, out: &mut Vec<EvaluationEvent>) {
        while self.step(now, out) {}
    }

    /// A decision whose onset lies in a negative phase counts for that phase; one in the open
    /// prompt's window resolves the prompt; any other is ignored.
    ///
    /// # Panics
    /// If an `Accepted` zone is not one of the session's zones (the decisions and the zones come
    /// from one model).
    pub fn on_decision(&mut self, d: &TapDecision, out: &mut Vec<EvaluationEvent>) {
        if let DecisionOutcome::Accepted { zone, .. } = d.outcome {
            assert!(
                usize::from(zone.0) < self.zones.len(),
                "accepted zone {} is not one of the {} evaluated zones",
                zone.0,
                self.zones.len()
            );
        }
        // An untimed decision is placed at its decision time, as the gesture places it.
        let t = match d.onset_time {
            Some(t0) => t0,
            None => d.decided_at,
        };
        if let Some(tally) = self.tallies.iter_mut().find(|n| n.start <= t && t < n.end) {
            tally.candidates += 1;
            if let DecisionOutcome::Accepted { zone, .. } = d.outcome {
                tally.false_accepts += 1;
                tally.by_zone[usize::from(zone.0)] += 1;
            }
            return;
        }
        let Phase::Prompt {
            zone,
            opened_at,
            deadline,
            ..
        } = self.phase
        else {
            return;
        };
        if t < opened_at || t >= deadline {
            return;
        }
        let latency = d.onset_time.map(|t0| d.decided_at.since(t0).as_ms());
        let outcome = match &d.outcome {
            DecisionOutcome::Accepted { zone: got, .. } if *got == zone => {
                PromptOutcome::Correct { latency }
            }
            DecisionOutcome::Accepted { zone: got, .. } => {
                PromptOutcome::WrongZone { got: *got, latency }
            }
            DecisionOutcome::Rejected { reason, .. } => PromptOutcome::Rejected { reason: *reason },
            DecisionOutcome::Vetoed { reason, .. } => PromptOutcome::Vetoed { reason: *reason },
            DecisionOutcome::Invalid { .. } => PromptOutcome::Invalid,
        };
        let margin = d.scores().map(|s| s.margin);
        self.resolve(outcome, margin, d.decided_at, out);
    }

    pub fn on_detector_rejected(
        &mut self,
        r: &RejectedCandidate,
        at: HostTime,
        out: &mut Vec<EvaluationEvent>,
    ) {
        let counts = match r.reason {
            RejectReason::LowSnr
            | RejectReason::NotBroadband
            | RejectReason::InsufficientBandSnr
            | RejectReason::NotDecaying
            | RejectReason::OnsetNotLocalized => true,
            RejectReason::Superseded
            | RejectReason::SpanCrossesDiscontinuity
            | RejectReason::PendingOverflow => false,
        };
        let Phase::Prompt {
            opened_at,
            deadline,
            ..
        } = self.phase
        else {
            return;
        };
        if counts && opened_at <= at && at < deadline {
            self.resolve(
                PromptOutcome::DetectorRejected { reason: r.reason },
                None,
                at,
                out,
            );
        }
    }

    /// The user says the last tap went to the wrong place (XR-UI-05, ui-v2 §4.5.2). Accepted in a Prompt or Gap step when
    /// the last prompt to resolve ended Correct, WrongZone, Rejected or Invalid (the outcomes the run shows only as
    /// "Got it.", so undo can't be aimed at a known miss) and has not been undone; otherwise `NothingToUndo`. It drops
    /// that outcome, withdraws the next prompt if it is already open (it follows, unchanged), reopens the same zone after
    /// PROMPT_GAP with a fresh PROMPT_WINDOW, adds 1 to `undone` and emits `PromptReopened`. A reopened prompt that
    /// resolves can be undone like any other; an earlier prompt never can.
    pub fn undo_last(
        &mut self,
        now: HostTime,
        out: &mut Vec<EvaluationEvent>,
    ) -> Result<(), AssessmentError> {
        // SPEC: a Scripted (replayed) run already holds its undos as `reopened` prompts; a further
        // undo has no recorded window to reopen into, so it is refused.
        if !matches!(self.plan, Plan::Generated { .. }) {
            return Err(AssessmentError::NothingToUndo);
        }
        let index = match (self.phase, self.undoable) {
            (Phase::Gap { .. } | Phase::Prompt { .. }, Some(index)) => index,
            (Phase::Gap { .. } | Phase::Prompt { .. }, None)
            | (Phase::Negative { .. } | Phase::Finished, _) => {
                return Err(AssessmentError::NothingToUndo)
            }
        };
        let opened_at = after(now, gap());
        self.reopen(index, opened_at, after(opened_at, window()), out);
        Ok(())
    }

    pub fn finish(
        self,
        meta: ReportMeta,
        run_at: DateTime<Utc>,
    ) -> Result<EvaluationReport, AssessmentError> {
        if self.phase != Phase::Finished {
            return Err(AssessmentError::NotFinished);
        }
        let results: Option<Vec<PromptResult>> = self.results.iter().cloned().collect();
        let results = match results {
            Some(results) => results,
            None => return Err(AssessmentError::NotFinished),
        };
        Ok(self.into_report(meta, run_at, &results))
    }

    fn total(&self) -> u32 {
        self.prompt_zones.len() as u32
    }

    /// One transition due at `now`, if any.
    fn step(&mut self, now: HostTime, out: &mut Vec<EvaluationEvent>) -> bool {
        match self.phase {
            Phase::Finished => false,
            Phase::Gap { until } => {
                if now < until {
                    return false;
                }
                self.end_gap(until, out);
                true
            }
            Phase::Prompt { deadline, .. } => {
                // A recorded undo withdraws the open prompt at the time it was made.
                if let Some(undo) = self.scripted_undo() {
                    if undo.start() <= now && undo.start() < deadline {
                        self.take_entry();
                        self.reopen(undo.index, undo.opened_at, undo.deadline, out);
                        return true;
                    }
                }
                if now < deadline {
                    return false;
                }
                self.resolve(PromptOutcome::Missed, None, deadline, out);
                true
            }
            Phase::Negative { index } => {
                let end = self.tallies[index as usize].end;
                if now < end {
                    return false;
                }
                out.push(EvaluationEvent::NegativeEnded { index });
                let next = self.tallies.len();
                let start = match &self.plan {
                    Plan::Generated { negatives, .. } => negatives.get(next).map(|_| end),
                    Plan::Scripted { negatives, .. } => negatives.get(next).map(|n| n.start),
                };
                match start {
                    Some(start) => self.phase = Phase::Gap { until: start },
                    None => self.finish_run(out),
                }
                true
            }
        }
    }

    /// The gap ends at `at`: the next prompt opens, or the next negative phase starts, or the
    /// run finishes.
    fn end_gap(&mut self, at: HostTime, out: &mut Vec<EvaluationEvent>) {
        let total = self.total();
        match &mut self.plan {
            Plan::Generated { next, negatives } => {
                if *next < total {
                    let index = *next;
                    *next += 1;
                    let zone = self.prompt_zones[index as usize];
                    self.open(index, zone, at, after(at, window()), out);
                } else if let Some(phase) = negatives.get(self.tallies.len()) {
                    let (class, duration) = (phase.class.clone(), phase.duration);
                    self.start_negative(class, at, after(at, seconds(duration)), duration, out);
                } else {
                    self.finish_run(out);
                }
            }
            Plan::Scripted {
                entries,
                next,
                negatives,
            } => {
                if let Some(&entry) = entries.get(*next) {
                    *next += 1;
                    if entry.reopened {
                        self.reopen(entry.index, entry.opened_at, entry.deadline, out);
                    } else {
                        self.open(
                            entry.index,
                            entry.zone,
                            entry.opened_at,
                            entry.deadline,
                            out,
                        );
                    }
                } else if let Some(n) = negatives.get(self.tallies.len()) {
                    let (class, start, end) = (n.class.clone(), n.start, n.end);
                    let duration = Seconds::try_new(
                        end.since(start).0 as f64 / HostTime::TICKS_PER_SECOND as f64,
                    )
                    // INVARIANT: an i64 tick count divided by 10^7 is always finite.
                    .expect("a tick span in seconds is finite");
                    self.start_negative(class, start, end, duration, out);
                } else {
                    self.finish_run(out);
                }
            }
        }
    }

    /// The next scripted entry if it is a reopened prompt (a recorded undo).
    fn scripted_undo(&self) -> Option<Entry> {
        match &self.plan {
            Plan::Scripted { entries, next, .. } => {
                entries.get(*next).copied().filter(|e| e.reopened)
            }
            Plan::Generated { .. } => None,
        }
    }

    fn take_entry(&mut self) {
        if let Plan::Scripted { next, .. } = &mut self.plan {
            *next += 1;
        }
    }

    fn open(
        &mut self,
        index: u32,
        zone: ZoneIndex,
        opened_at: HostTime,
        deadline: HostTime,
        out: &mut Vec<EvaluationEvent>,
    ) {
        self.phase = Phase::Prompt {
            index,
            zone,
            opened_at,
            deadline,
        };
        out.push(EvaluationEvent::PromptOpened { index, zone });
    }

    /// Undo: drops prompt `index`'s outcome, withdraws a different prompt that is open (it
    /// follows, unchanged), and reopens `index` over [opened_at, deadline).
    fn reopen(
        &mut self,
        index: u32,
        opened_at: HostTime,
        deadline: HostTime,
        out: &mut Vec<EvaluationEvent>,
    ) {
        if let Phase::Prompt { index: open, .. } = self.phase {
            if open != index {
                if let Plan::Generated { next, .. } = &mut self.plan {
                    *next = open;
                }
            }
        }
        self.results[index as usize] = None;
        self.undone += 1;
        self.undoable = None;
        self.last = None;
        let zone = self.prompt_zones[index as usize];
        self.phase = Phase::Prompt {
            index,
            zone,
            opened_at,
            deadline,
        };
        out.push(EvaluationEvent::PromptReopened { index, zone });
    }

    /// Resolves the open prompt at `at`; the next step starts after PROMPT_GAP (a script's next
    /// entry, or its first negative phase, starts when it was recorded).
    fn resolve(
        &mut self,
        outcome: PromptOutcome,
        margin: Option<f64>,
        at: HostTime,
        out: &mut Vec<EvaluationEvent>,
    ) {
        let Phase::Prompt { index, zone, .. } = self.phase else {
            return;
        };
        let undoable = match outcome {
            PromptOutcome::Correct { .. }
            | PromptOutcome::WrongZone { .. }
            | PromptOutcome::Rejected { .. }
            | PromptOutcome::Invalid => true,
            PromptOutcome::Vetoed { .. }
            | PromptOutcome::DetectorRejected { .. }
            | PromptOutcome::Missed => false,
        };
        self.undoable = undoable.then_some(index);
        self.last = Some(outcome.clone());
        self.results[index as usize] = Some(PromptResult {
            outcome: outcome.clone(),
            margin,
        });
        out.push(EvaluationEvent::PromptResolved {
            index,
            zone,
            outcome,
        });
        let until = match &self.plan {
            Plan::Scripted {
                entries,
                next,
                negatives,
            } => match entries.get(*next) {
                Some(entry) => entry.start(),
                None => match negatives.get(self.tallies.len()) {
                    Some(n) => n.start,
                    None => after(at, gap()),
                },
            },
            Plan::Generated { .. } => after(at, gap()),
        };
        self.phase = Phase::Gap { until };
    }

    fn start_negative(
        &mut self,
        class: NegativeClass,
        start: HostTime,
        end: HostTime,
        duration: Seconds,
        out: &mut Vec<EvaluationEvent>,
    ) {
        let index = self.tallies.len() as u32;
        self.undoable = None;
        self.tallies.push(NegativeTally {
            class: class.clone(),
            start,
            end,
            duration,
            candidates: 0,
            false_accepts: 0,
            by_zone: vec![0; self.zones.len()],
        });
        self.phase = Phase::Negative { index };
        out.push(EvaluationEvent::NegativeStarted { index, class });
    }

    fn finish_run(&mut self, out: &mut Vec<EvaluationEvent>) {
        self.undoable = None;
        self.phase = Phase::Finished;
        out.push(EvaluationEvent::Finished);
    }
}

fn gap() -> HostDuration {
    HostDuration::from_ms(PROMPT_GAP)
}

fn window() -> HostDuration {
    HostDuration::from_ms(PROMPT_WINDOW)
}

fn seconds(s: Seconds) -> HostDuration {
    // Saturates at the i64 range, which no evaluation phase approaches.
    HostDuration((s.get() * HostTime::TICKS_PER_SECOND as f64).round() as i64)
}

/// t + d, saturating at the i64 range.
fn after(t: HostTime, d: HostDuration) -> HostTime {
    HostTime(t.0.saturating_add(d.0))
}

/// t − d, saturating at the i64 range.
fn before(t: HostTime, d: HostDuration) -> HostTime {
    HostTime(t.0.saturating_sub(d.0))
}

/// The Generated order (§1.17.1): forced zone when 2·c > R, else a draw over the zones other
/// than the previous one, weighted by their remaining counts.
fn generated_order(zones: usize, prompts_per_zone: u32, seed: u64) -> Vec<ZoneIndex> {
    let mut left = vec![u64::from(prompts_per_zone); zones];
    let mut remaining = zones as u64 * u64::from(prompts_per_zone);
    let mut rng = Rng64::new(seed);
    let mut prev: Option<usize> = None;
    let mut order = Vec::with_capacity(remaining as usize);
    while remaining > 0 {
        let zone = match (0..zones).find(|&z| 2 * left[z] > remaining) {
            Some(forced) => forced,
            None => {
                let others = |z: &usize| Some(*z) != prev;
                let weight: u64 = (0..zones).filter(others).map(|z| left[z]).sum();
                // INVARIANT: with nothing forced, the previous zone holds at most half of the
                // remaining prompts, so the other zones hold at least one (weight > 0).
                let u = rng.below(weight);
                let mut run = 0;
                (0..zones)
                    .filter(others)
                    .find(|&z| {
                        run += left[z];
                        run > u
                    })
                    // INVARIANT: the running sum reaches `weight` > u by the last zone.
                    .expect("the walk passes u")
            }
        };
        left[zone] -= 1;
        remaining -= 1;
        prev = Some(zone);
        order.push(ZoneIndex(zone as u16));
    }
    order
}

/// Assigns each recorded prompt its index. A fresh prompt takes the next index; a reopened one
/// is an undo of the last prompt to resolve: the entry before it, unless that entry is a fresh
/// prompt of another zone, which the undo withdrew (it follows, unchanged, with its index).
fn script_entries(zones: usize, prompts: &[ScriptedPrompt]) -> Result<Vec<Entry>, AssessmentError> {
    if prompts.is_empty() {
        return Err(AssessmentError::ZeroPrompts);
    }
    let mut entries: Vec<Entry> = Vec::with_capacity(prompts.len());
    let mut next: u32 = 0;
    for (j, p) in prompts.iter().enumerate() {
        // SPEC: no AssessmentError names a script whose zone is outside the zones given; it is
        // reported as ScriptUnordered, the one error for a script that cannot be replayed.
        let in_order = usize::from(p.zone.0) < zones
            && p.opened_at < p.deadline
            && entries.last().is_none_or(|e| e.opened_at < p.opened_at);
        if !in_order {
            return Err(AssessmentError::ScriptUnordered);
        }
        let index = if p.reopened {
            let prev = match j.checked_sub(1).map(|i| entries[i]) {
                Some(prev) => prev,
                None => return Err(AssessmentError::ScriptUnordered),
            };
            let undone = if !prev.reopened && prev.zone != p.zone {
                // `prev` was withdrawn: its index is next again, and the entry before it is the
                // prompt undone.
                next = prev.index;
                match j.checked_sub(2).map(|i| entries[i]) {
                    Some(undone) => undone,
                    None => return Err(AssessmentError::ScriptUnordered),
                }
            } else {
                prev
            };
            if undone.zone != p.zone {
                return Err(AssessmentError::ScriptUnordered);
            }
            undone.index
        } else {
            next += 1;
            next - 1
        };
        entries.push(Entry {
            index,
            zone: p.zone,
            opened_at: p.opened_at,
            deadline: p.deadline,
            reopened: p.reopened,
        });
    }
    Ok(entries)
}

/// Negative phases are non-empty, in order, disjoint, and start after the last prompt opened.
fn check_negatives(
    prompts: &[ScriptedPrompt],
    negatives: &[ScriptedNegative],
) -> Result<(), AssessmentError> {
    let mut floor = prompts.last().map(|p| p.opened_at);
    for n in negatives {
        if n.start >= n.end || floor.is_some_and(|f| n.start < f) {
            return Err(AssessmentError::ScriptUnordered);
        }
        floor = Some(n.end);
    }
    Ok(())
}

// ---- Report ----

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationReport {
    pub schema_version: u32,
    pub profile_id: ProfileId,
    pub run_at: DateTime<Utc>,
    pub meta: ReportMeta,
    pub zones: Vec<ZoneRef>,
    pub prompts_per_zone: u32,
    pub target: f64, // ACCURACY_TARGET when measured
    pub undone: u32, // prompts reopened by undo_last
    pub confusion: ConfusionMatrix,
    pub per_zone: Vec<ZoneMetrics>,
    pub overall: OverallMetrics,
    pub negatives: Vec<NegativeClassResult>,
    pub decision_latency: LatencyStats,
    /// one per prompt, index order (an undone outcome is replaced by its reopened prompt's); no audio (PARITY-10)
    pub prompts: Vec<PromptRecord>,
    /// MARGIN_BIN bins of `prompts[].margin`
    pub margin_histogram: Vec<u32>,
}

/// `margin` = `ModelScores.margin` for prompts that reached the model; latency lives in the outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptRecord {
    pub zone: ZoneId,
    pub outcome: PromptOutcome,
    pub margin: Option<f64>,
}

/// rows = prompted zone; columns = predicted zone 0..K, then Rejected, Vetoed, DetectorRejected, Invalid, Missed
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    pub zones: u32,
    pub counts: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RejectionBreakdown {
    pub out_of_distribution: u32,
    pub background: u32,
    pub ambiguous: u32,
    pub vetoed: u32,
    pub detector: u32,
    pub invalid: u32,
}

impl RejectionBreakdown {
    fn total(&self) -> u32 {
        self.out_of_distribution
            + self.background
            + self.ambiguous
            + self.vetoed
            + self.detector
            + self.invalid
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZoneMetrics {
    pub zone: ZoneId,
    pub prompts: u32,
    pub correct: u32,
    pub wrong: u32,
    pub rejected: RejectionBreakdown,
    pub missed: u32,
    pub recall: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverallMetrics {
    pub prompts: u32,
    pub correct: u32,
    pub wrong: u32,
    pub rejected: u32,
    pub missed: u32,
    pub accuracy: f64,          // correct / prompts  (ACCEPTANCE ≥ 92 % target)
    pub wrong_action_rate: f64, // wrong / (correct + wrong); 0.0 when no accepts  (hard precision gate, ADR-006)
    pub rejection_rate: f64,    // (rejected + missed) / prompts
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NegativeClassResult {
    pub class: NegativeClass,
    pub duration: Seconds,
    pub candidates: u32,
    pub false_accepts: u32,
    pub false_accepts_per_hour: f64,
    pub false_accepts_by_zone: Vec<u32>, /* model order */
}

/// nearest-rank; all 0 when count == 0
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencyStats {
    pub count: u32,
    pub untimed: u32,
    pub p50: Milliseconds,
    pub p95: Milliseconds,
    pub p99: Milliseconds,
    pub max: Milliseconds,
    pub histogram: Vec<u32>,
}

impl LatencyStats {
    /// Nearest-rank quantiles of the samples sorted with `total_cmp`, and the 5 ms histogram
    /// (40 bins + 1 overflow bin). `untimed` counts the decisions that had no onset time.
    ///
    /// # Panics
    /// If a sample is not finite: latencies are differences of host ticks, always finite.
    pub fn from_samples(samples_ms: &[f64], untimed: u32) -> LatencyStats {
        let mut sorted = samples_ms.to_vec();
        sorted.sort_by(f64::total_cmp);
        let mut histogram = vec![0u32; LATENCY_BINS + 1];
        for &v in &sorted {
            histogram[bin(v, LATENCY_BIN.get(), LATENCY_BINS)] += 1;
        }
        // Nearest rank: the ⌈percent·n/100⌉-th smallest, in integers so no rounding moves a rank.
        let rank = |percent: usize| -> Milliseconds {
            let n = sorted.len();
            if n == 0 {
                return Milliseconds::new(0.0);
            }
            let r = (percent * n).div_ceil(100);
            Milliseconds::new(sorted[r.clamp(1, n) - 1])
        };
        LatencyStats {
            count: sorted.len() as u32,
            untimed,
            p50: rank(50),
            p95: rank(95),
            p99: rank(99),
            max: match sorted.last() {
                Some(&v) => Milliseconds::new(v),
                None => Milliseconds::new(0.0),
            },
            histogram,
        }
    }
}

/// floor(v / width) in 0..bins, values ≥ bins·width in the overflow bin `bins`. Latencies and
/// margins are ≥ 0; a negative value would land in bin 0 (the saturating cast).
fn bin(v: f64, width: f64, bins: usize) -> usize {
    ((v / width).floor() as usize).min(bins)
}

fn ratio(num: u32, den: u32) -> f64 {
    if den == 0 {
        0.0
    } else {
        f64::from(num) / f64::from(den)
    }
}

impl EvaluationSession {
    /// The report of a finished run; `results` holds every prompt's outcome, index order.
    fn into_report(
        self,
        meta: ReportMeta,
        run_at: DateTime<Utc>,
        results: &[PromptResult],
    ) -> EvaluationReport {
        let EvaluationSession {
            zones,
            prompts_per_zone,
            prompt_zones,
            undone,
            tallies,
            ..
        } = self;
        let k = zones.len();
        let mut counts = vec![vec![0u32; k + OUTCOME_COLUMNS]; k];
        let mut per_zone: Vec<ZoneMetrics> = zones
            .iter()
            .map(|z| ZoneMetrics {
                zone: z.id,
                prompts: 0,
                correct: 0,
                wrong: 0,
                rejected: RejectionBreakdown::default(),
                missed: 0,
                recall: 0.0,
            })
            .collect();
        let mut latencies = Vec::new();
        let mut untimed = 0;
        let mut margin_histogram = vec![0u32; MARGIN_BINS + 1];
        let mut prompts = Vec::with_capacity(results.len());
        for (r, &zone) in results.iter().zip(&prompt_zones) {
            let z = usize::from(zone.0);
            let m = &mut per_zone[z];
            m.prompts += 1;
            let column = match &r.outcome {
                PromptOutcome::Correct { .. } => {
                    m.correct += 1;
                    z
                }
                PromptOutcome::WrongZone { got, .. } => {
                    m.wrong += 1;
                    usize::from(got.0)
                }
                PromptOutcome::Rejected { reason } => {
                    match reason {
                        ModelRejectReason::OutOfDistribution => m.rejected.out_of_distribution += 1,
                        ModelRejectReason::Background => m.rejected.background += 1,
                        ModelRejectReason::Ambiguous => m.rejected.ambiguous += 1,
                    }
                    k
                }
                PromptOutcome::Vetoed { .. } => {
                    m.rejected.vetoed += 1;
                    k + 1
                }
                PromptOutcome::DetectorRejected { .. } => {
                    m.rejected.detector += 1;
                    k + 2
                }
                PromptOutcome::Invalid => {
                    m.rejected.invalid += 1;
                    k + 3
                }
                PromptOutcome::Missed => {
                    m.missed += 1;
                    k + 4
                }
            };
            counts[z][column] += 1;
            match &r.outcome {
                PromptOutcome::Correct { latency } | PromptOutcome::WrongZone { latency, .. } => {
                    match latency {
                        Some(ms) => latencies.push(ms.get()),
                        None => untimed += 1,
                    }
                }
                PromptOutcome::Rejected { .. }
                | PromptOutcome::Vetoed { .. }
                | PromptOutcome::DetectorRejected { .. }
                | PromptOutcome::Invalid
                | PromptOutcome::Missed => {}
            }
            if let Some(margin) = r.margin {
                margin_histogram[bin(margin, MARGIN_BIN, MARGIN_BINS)] += 1;
            }
            prompts.push(PromptRecord {
                zone: zones[z].id,
                outcome: r.outcome.clone(),
                margin: r.margin,
            });
        }
        for m in &mut per_zone {
            m.recall = ratio(m.correct, m.prompts);
        }
        let sum = |f: fn(&ZoneMetrics) -> u32| per_zone.iter().map(f).sum::<u32>();
        let (correct, wrong, missed) = (sum(|m| m.correct), sum(|m| m.wrong), sum(|m| m.missed));
        let rejected = sum(|m| m.rejected.total());
        let total = results.len() as u32;
        let overall = OverallMetrics {
            prompts: total,
            correct,
            wrong,
            rejected,
            missed,
            accuracy: ratio(correct, total),
            wrong_action_rate: ratio(wrong, correct + wrong),
            rejection_rate: ratio(rejected + missed, total),
        };
        let negatives = tallies
            .iter()
            .map(|t| NegativeClassResult {
                class: t.class.clone(),
                duration: t.duration,
                candidates: t.candidates,
                false_accepts: t.false_accepts,
                // A phase with no accepts has rate 0 whatever its length (an empty one included).
                false_accepts_per_hour: if t.false_accepts == 0 {
                    0.0
                } else {
                    f64::from(t.false_accepts) * 3600.0 / t.duration.get()
                },
                false_accepts_by_zone: t.by_zone.clone(),
            })
            .collect();
        EvaluationReport {
            schema_version: EVALUATION_SCHEMA_VERSION,
            profile_id: meta.profile_id,
            run_at,
            meta,
            confusion: ConfusionMatrix {
                zones: k as u32,
                counts,
            },
            zones,
            prompts_per_zone,
            target: ACCURACY_TARGET,
            undone,
            per_zone,
            overall,
            negatives,
            decision_latency: LatencyStats::from_samples(&latencies, untimed),
            prompts,
            margin_histogram,
        }
    }
}

impl EvaluationReport {
    /// Validated first, so no save path can write an inconsistent or non-finite report (§0.5):
    /// failure is `PersistenceError::InvalidData`. Then pretty JSON.
    pub fn to_json(&self) -> Result<String, PersistenceError> {
        self.validate()
            .map_err(|reason| PersistenceError::InvalidData {
                file: UNSAVED_REPORT.to_string(),
                reason,
            })?;
        serde_json::to_string_pretty(self).map_err(PersistenceError::Json)
    }

    /// parse_versioned(…, 2, …), then the same validation as `to_json`.
    pub fn from_json(text: &str, file: &Path) -> Result<EvaluationReport, PersistenceError> {
        let name = file.display().to_string();
        let report: EvaluationReport = parse_versioned(text, EVALUATION_SCHEMA_VERSION, &name)?;
        report
            .validate()
            .map_err(|reason| PersistenceError::InvalidData { file: name, reason })?;
        Ok(report)
    }

    /// Shapes agree with the zone count, every prompt's zone is listed, and every float is finite.
    fn validate(&self) -> Result<(), String> {
        let k = self.zones.len();
        if k == 0 {
            return Err("the report lists no zones".to_string());
        }
        let shapes = [
            ("confusion.zones", self.confusion.zones as usize, k),
            ("confusion rows", self.confusion.counts.len(), k),
            ("per_zone", self.per_zone.len(), k),
            (
                "decision_latency.histogram",
                self.decision_latency.histogram.len(),
                LATENCY_BINS + 1,
            ),
            (
                "margin_histogram",
                self.margin_histogram.len(),
                MARGIN_BINS + 1,
            ),
        ];
        for (what, got, expected) in shapes {
            if got != expected {
                return Err(format!("{what} has {got} entries, {expected} expected"));
            }
        }
        if let Some(row) = self
            .confusion
            .counts
            .iter()
            .find(|r| r.len() != k + OUTCOME_COLUMNS)
        {
            return Err(format!(
                "a confusion row has {} columns, {} expected",
                row.len(),
                k + OUTCOME_COLUMNS
            ));
        }
        if let Some(n) = self
            .negatives
            .iter()
            .find(|n| n.false_accepts_by_zone.len() != k)
        {
            return Err(format!(
                "false_accepts_by_zone has {} entries, {k} expected",
                n.false_accepts_by_zone.len()
            ));
        }
        if let Some(p) = self
            .prompts
            .iter()
            .find(|p| !self.zones.iter().any(|z| z.id == p.zone))
        {
            return Err(format!("prompt zone {} is not a report zone", p.zone));
        }
        let floats = [self.target, self.overall.accuracy]
            .into_iter()
            .chain([self.overall.wrong_action_rate, self.overall.rejection_rate])
            .chain(self.per_zone.iter().map(|m| m.recall))
            .chain(self.negatives.iter().map(|n| n.false_accepts_per_hour))
            .chain(self.prompts.iter().filter_map(|p| p.margin));
        for v in floats {
            if !v.is_finite() {
                return Err("a metric is not finite".to_string());
            }
        }
        Ok(())
    }

    /// Blank-line-separated blocks, as v1 and DESIGN ask: (1) metric,value (accuracy, target, prompts, correct, wrong,
    /// rejected, missed, undone, wrong_action_rate, rejection_rate, latency p50/p95/p99); (2) the confusion matrix with
    /// each row's recall; (3) per zone: zone,prompts,correct,wrong,rejected_ood,rejected_bg,rejected_ambiguous,vetoed,
    /// detector,invalid,missed,recall; (4) the margin histogram; (5) negatives per class with false accepts per zone.
    pub fn to_csv(&self) -> String {
        let o = &self.overall;
        let l = &self.decision_latency;
        let mut s = String::from("metric,value\n");
        let metrics: [(&str, String); 13] = [
            ("accuracy", o.accuracy.to_string()),
            ("target", self.target.to_string()),
            ("prompts", o.prompts.to_string()),
            ("correct", o.correct.to_string()),
            ("wrong", o.wrong.to_string()),
            ("rejected", o.rejected.to_string()),
            ("missed", o.missed.to_string()),
            ("undone", self.undone.to_string()),
            ("wrong_action_rate", o.wrong_action_rate.to_string()),
            ("rejection_rate", o.rejection_rate.to_string()),
            ("latency_p50_ms", l.p50.get().to_string()),
            ("latency_p95_ms", l.p95.get().to_string()),
            ("latency_p99_ms", l.p99.get().to_string()),
        ];
        for (name, value) in metrics {
            s.push_str(&format!("{name},{value}\n"));
        }

        let labels: Vec<String> = self.zones.iter().map(|z| csv_field(&z.label)).collect();
        s.push_str(&format!(
            "\nprompted,{},rejected,vetoed,detector_rejected,invalid,missed,recall\n",
            labels.join(",")
        ));
        for ((label, row), m) in labels
            .iter()
            .zip(&self.confusion.counts)
            .zip(&self.per_zone)
        {
            let cells: Vec<String> = row.iter().map(u32::to_string).collect();
            s.push_str(&format!("{label},{},{}\n", cells.join(","), m.recall));
        }

        s.push_str(
            "\nzone,prompts,correct,wrong,rejected_ood,rejected_bg,rejected_ambiguous,vetoed,\
             detector,invalid,missed,recall\n",
        );
        for (label, m) in labels.iter().zip(&self.per_zone) {
            let r = &m.rejected;
            s.push_str(&format!(
                "{label},{},{},{},{},{},{},{},{},{},{},{}\n",
                m.prompts,
                m.correct,
                m.wrong,
                r.out_of_distribution,
                r.background,
                r.ambiguous,
                r.vetoed,
                r.detector,
                r.invalid,
                m.missed,
                m.recall
            ));
        }

        s.push_str("\nmargin_bin,count\n");
        for (i, count) in self.margin_histogram.iter().enumerate() {
            let lo = i as f64 * MARGIN_BIN;
            if i < MARGIN_BINS {
                s.push_str(&format!("{lo}-{},{count}\n", lo + MARGIN_BIN));
            } else {
                s.push_str(&format!("{lo}+,{count}\n"));
            }
        }

        s.push_str(&format!(
            "\nclass,duration_s,candidates,false_accepts,false_accepts_per_hour,{}\n",
            labels.join(",")
        ));
        for n in &self.negatives {
            let by_zone: Vec<String> = n.false_accepts_by_zone.iter().map(u32::to_string).collect();
            s.push_str(&format!(
                "{},{},{},{},{},{}\n",
                csv_field(&class_name(&n.class)),
                n.duration.get(),
                n.candidates,
                n.false_accepts,
                n.false_accepts_per_hour,
                by_zone.join(",")
            ));
        }
        s
    }

    /// index,zone,outcome,latency_ms,margin: one row per prompt, index order; empty cells for a
    /// latency or margin the prompt does not have.
    pub fn to_prompts_csv(&self) -> String {
        let mut s = String::from("index,zone,outcome,latency_ms,margin\n");
        for (i, p) in self.prompts.iter().enumerate() {
            // A zone missing from `zones` (an inconsistent report, which from_json refuses) is
            // printed by its id.
            let zone = match self.zones.iter().find(|z| z.id == p.zone) {
                Some(z) => csv_field(&z.label),
                None => p.zone.to_string(),
            };
            let (name, latency) = match &p.outcome {
                PromptOutcome::Correct { latency } => ("Correct", *latency),
                PromptOutcome::WrongZone { latency, .. } => ("WrongZone", *latency),
                PromptOutcome::Rejected { .. } => ("Rejected", None),
                PromptOutcome::Vetoed { .. } => ("Vetoed", None),
                PromptOutcome::DetectorRejected { .. } => ("DetectorRejected", None),
                PromptOutcome::Invalid => ("Invalid", None),
                PromptOutcome::Missed => ("Missed", None),
            };
            let latency = latency.map_or(String::new(), |ms| ms.get().to_string());
            let margin = p.margin.map_or(String::new(), |m| m.to_string());
            s.push_str(&format!("{i},{zone},{name},{latency},{margin}\n"));
        }
        s
    }
}

fn class_name(class: &NegativeClass) -> String {
    match class {
        NegativeClass::Typing => "Typing".to_string(),
        NegativeClass::Clicking => "Clicking".to_string(),
        NegativeClass::Speech => "Speech".to_string(),
        NegativeClass::Music => "Music".to_string(),
        NegativeClass::Knocks => "Knocks".to_string(),
        NegativeClass::ObjectsSetDown => "ObjectsSetDown".to_string(),
        NegativeClass::Everyday => "Everyday".to_string(),
        NegativeClass::Other { label } => label.clone(),
    }
}

/// RFC 4180: quoted, with embedded quotes doubled, when it holds a comma, quote or line break.
fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// The read-only header of a v1 report (schema 1), from a frozen private shape that survives WP-D1 (PARITY-35).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegacyReport {
    pub run_at: DateTime<Utc>,
    pub overall_accuracy: f64,
    pub taps_per_zone: u32,
}

/// The fields of a v1 report this module reads; every other v1 field is ignored.
#[derive(Debug, Deserialize)]
struct LegacyReportV1 {
    run_at: DateTime<Utc>,
    overall_accuracy: f64,
    taps_per_zone: u32,
}

/// v1 reports are not migrated (a report is evidence); they are listed by this header.
pub fn peek_legacy_report(text: &str, file: &Path) -> Result<LegacyReport, PersistenceError> {
    let v1: LegacyReportV1 =
        parse_versioned(text, LEGACY_SCHEMA_VERSION, &file.display().to_string())?;
    Ok(LegacyReport {
        run_at: v1.run_at,
        overall_accuracy: v1.overall_accuracy,
        taps_per_zone: v1.taps_per_zone,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{AudioFormat, CaptureMode};
    use crate::descriptor::{Descriptor, DescriptorError, DescriptorLayout, DESCRIPTOR_DIMS};
    use crate::detector::GateReport;
    use crate::model::ModelScores;
    use crate::units::{CandidateId, ChannelCount, Decibels, SampleIndex, SampleRate};
    use crate::veto::InputKind;
    use uuid::Uuid;

    /// Host ticks per millisecond.
    const MS: i64 = 10_000;
    /// The simulated user taps this long after a window opens.
    const REACTION_MS: i64 = 350;

    fn t(ms: i64) -> HostTime {
        HostTime(ms * MS)
    }

    fn ms_of(t: HostTime) -> i64 {
        t.0 / MS
    }

    fn zone_refs(k: usize) -> Vec<ZoneRef> {
        (0..k)
            .map(|z| ZoneRef {
                id: ZoneId::from_uuid(Uuid::from_u128(z as u128 + 1)),
                label: format!("Zone {z}"),
            })
            .collect()
    }

    fn scores(margin: f64) -> ModelScores {
        ModelScores {
            d2: vec![1.0, 1.0 + margin],
            best: ZoneIndex(0),
            second: ZoneIndex(1),
            d2_min: 1.0,
            margin,
            llr_bg: 5.0,
            posterior_best: 0.9,
        }
    }

    fn descriptor() -> Descriptor {
        Descriptor {
            layout: DescriptorLayout::Mono { a: 0 },
            values: [0.0; DESCRIPTOR_DIMS],
        }
    }

    fn accepted(zone: u16, margin: f64) -> DecisionOutcome {
        DecisionOutcome::Accepted {
            zone: ZoneIndex(zone),
            descriptor: descriptor(),
            scores: scores(margin),
        }
    }

    fn rejected(reason: ModelRejectReason, margin: f64) -> DecisionOutcome {
        DecisionOutcome::Rejected {
            reason,
            descriptor: descriptor(),
            scores: scores(margin),
        }
    }

    fn vetoed(margin: f64) -> DecisionOutcome {
        DecisionOutcome::Vetoed {
            reason: VetoReason::Input {
                kind: InputKind::KeyDown,
                offset: Milliseconds::new(12.0),
            },
            descriptor: descriptor(),
            scores: scores(margin),
        }
    }

    fn invalid() -> DecisionOutcome {
        DecisionOutcome::Invalid {
            error: DescriptorError::SpanTooShort { needed: 10, got: 0 },
        }
    }

    fn decision(onset_ms: Option<i64>, decided_ms: i64, outcome: DecisionOutcome) -> TapDecision {
        TapDecision {
            candidate: CandidateId(decided_ms as u64),
            onset: SampleIndex(0),
            onset_time: onset_ms.map(t),
            emitted_at: SampleIndex(0),
            decided_at: t(decided_ms),
            gates: GateReport {
                snr: Decibels::new(30.0),
                broadband_fraction: 1.0,
                valid_bands: 20,
                decay_fraction: 1.0,
                clipped: false,
                trigger_flux: 2.0,
            },
            outcome,
        }
    }

    fn run_at() -> DateTime<Utc> {
        "2026-11-10T12:00:00Z".parse().expect("valid time")
    }

    fn meta(k: usize) -> ReportMeta {
        ReportMeta {
            profile_id: ProfileId::from_uuid(Uuid::from_u128(99)),
            device: DeviceIdentity {
                endpoint_id: "test-endpoint".to_string(),
                name: "Test microphone".to_string(),
                format: AudioFormat {
                    rate: SampleRate::new(48_000).expect("valid rate"),
                    channels: ChannelCount::new(2).expect("valid channels"),
                },
                capture_mode: CaptureMode::Raw,
            },
            front_end_version: 1,
            model_trained_at: "2026-11-09T09:30:00Z".parse().expect("valid time"),
            zone_taps: vec![8; k],
            negatives: Vec::new(),
            background_regime: BackgroundRegime::PooledAll {
                negative_clips: 0,
                needed: 24,
            },
            input_veto_enabled: true,
        }
    }

    fn generated(
        k: usize,
        prompts_per_zone: u32,
        seed: u64,
        negatives: &[(NegativeClass, f64)],
    ) -> EvaluationSession {
        EvaluationSession::new(
            zone_refs(k),
            PromptSchedule::Generated {
                prompts_per_zone,
                seed,
                negatives: negatives
                    .iter()
                    .map(|(class, s)| NegativePhase {
                        class: class.clone(),
                        duration: Seconds::new(*s),
                    })
                    .collect(),
            },
            t(0),
        )
        .expect("valid schedule")
    }

    /// Everything a session was fed, in order, so a Scripted replay can be fed the same.
    #[derive(Debug, Clone)]
    enum Feed {
        Tick(HostTime),
        Decision(Box<TapDecision>),
        Rejected(RejectedCandidate, HostTime),
        Undo(HostTime),
    }

    /// Drives a session on a 10 ms tick, as a user would, and records what a live run writes to
    /// the sidecar: the prompts (from `view()` at each opening) and the negative phases.
    struct Driver {
        s: EvaluationSession,
        now_ms: i64,
        events: Vec<EvaluationEvent>,
        log: Vec<Feed>,
        script: Vec<ScriptedPrompt>,
        negatives: Vec<ScriptedNegative>,
        durations_ms: Vec<i64>,
    }

    impl Driver {
        fn new(s: EvaluationSession, durations_s: &[f64]) -> Self {
            Driver {
                s,
                now_ms: 0,
                events: Vec::new(),
                log: Vec::new(),
                script: Vec::new(),
                negatives: Vec::new(),
                durations_ms: durations_s.iter().map(|s| (s * 1000.0) as i64).collect(),
            }
        }

        fn absorb(&mut self, ev: Vec<EvaluationEvent>) {
            for e in &ev {
                match e {
                    EvaluationEvent::PromptOpened { zone, .. }
                    | EvaluationEvent::PromptReopened { zone, .. } => {
                        let EvaluationStep::Prompt { deadline, .. } = self.s.view().step else {
                            panic!("a prompt opens into a Prompt step")
                        };
                        self.script.push(ScriptedPrompt {
                            zone: *zone,
                            opened_at: before(deadline, window()),
                            deadline,
                            reopened: matches!(e, EvaluationEvent::PromptReopened { .. }),
                        });
                    }
                    EvaluationEvent::NegativeStarted { index, class } => {
                        let EvaluationStep::Negative { ends_at, .. } = self.s.view().step else {
                            panic!("a negative phase starts into a Negative step")
                        };
                        let len = self.durations_ms[*index as usize];
                        self.negatives.push(ScriptedNegative {
                            class: class.clone(),
                            start: t(ms_of(ends_at) - len),
                            end: ends_at,
                        });
                    }
                    EvaluationEvent::PromptResolved { .. }
                    | EvaluationEvent::NegativeEnded { .. }
                    | EvaluationEvent::Finished => {}
                }
            }
            self.events.extend(ev);
        }

        fn step(&mut self) {
            self.now_ms += 10;
            let now = t(self.now_ms);
            let mut ev = Vec::new();
            self.s.tick(now, &mut ev);
            self.log.push(Feed::Tick(now));
            self.absorb(ev);
        }

        fn tick_to(&mut self, ms: i64) {
            while self.now_ms < ms {
                self.step();
            }
        }

        /// Ticks until a prompt window is open; returns (index, zone, opened ms).
        fn wait_prompt(&mut self) -> (u32, ZoneIndex, i64) {
            loop {
                match self.s.view().step {
                    EvaluationStep::Prompt {
                        index,
                        zone,
                        deadline,
                        ..
                    } if ms_of(deadline) - 3000 <= self.now_ms => {
                        return (index, zone, ms_of(deadline) - 3000)
                    }
                    EvaluationStep::Finished => panic!("finished while waiting for a prompt"),
                    EvaluationStep::Prompt { .. }
                    | EvaluationStep::Gap { .. }
                    | EvaluationStep::Negative { .. } => self.step(),
                }
            }
        }

        fn deliver(&mut self, d: TapDecision) {
            let mut ev = Vec::new();
            self.s.on_decision(&d, &mut ev);
            self.log.push(Feed::Decision(Box::new(d)));
            self.absorb(ev);
        }

        /// A tap REACTION_MS into the open prompt, decided `latency_ms` later.
        fn tap(&mut self, outcome: DecisionOutcome, latency_ms: i64) -> (u32, ZoneIndex) {
            let (index, zone, opened) = self.wait_prompt();
            let onset = opened + REACTION_MS;
            self.tick_to(onset + latency_ms);
            self.deliver(decision(Some(onset), onset + latency_ms, outcome));
            (index, zone)
        }

        /// The prompt's zone, then a tap whose outcome is built from it.
        fn tap_with(
            &mut self,
            outcome: impl FnOnce(ZoneIndex) -> DecisionOutcome,
        ) -> (u32, ZoneIndex) {
            let (_, zone, _) = self.wait_prompt();
            self.tap(outcome(zone), 50)
        }

        fn untimed_tap(&mut self, outcome: DecisionOutcome) {
            let (_, _, opened) = self.wait_prompt();
            self.tick_to(opened + REACTION_MS + 50);
            self.deliver(decision(None, self.now_ms, outcome));
        }

        fn faint(&mut self, reason: RejectReason) {
            let (_, _, opened) = self.wait_prompt();
            self.tick_to(opened + REACTION_MS);
            let r = RejectedCandidate {
                id: CandidateId(self.now_ms as u64),
                trigger: SampleIndex(0),
                onset: None,
                reason,
                gates: None,
            };
            let at = t(self.now_ms);
            let mut ev = Vec::new();
            self.s.on_detector_rejected(&r, at, &mut ev);
            self.log.push(Feed::Rejected(r, at));
            self.absorb(ev);
        }

        fn miss(&mut self) {
            let (_, _, opened) = self.wait_prompt();
            self.tick_to(opened + 3000);
        }

        fn undo(&mut self) -> Result<(), AssessmentError> {
            let now = t(self.now_ms);
            let mut ev = Vec::new();
            let result = self.s.undo_last(now, &mut ev);
            if result.is_ok() {
                self.log.push(Feed::Undo(now));
            }
            self.absorb(ev);
            result
        }

        /// Ticks until a negative phase runs; returns its start ms.
        fn wait_negative(&mut self) -> i64 {
            loop {
                if let EvaluationStep::Negative { index, ends_at, .. } = self.s.view().step {
                    return ms_of(ends_at) - self.durations_ms[index as usize];
                }
                assert!(self.s.view().step != EvaluationStep::Finished);
                self.step();
            }
        }

        /// A decision `offset_ms` into the negative phase that began at `start`.
        fn background(&mut self, start: i64, offset_ms: i64, outcome: DecisionOutcome) {
            let onset = start + offset_ms;
            self.tick_to(onset + 50);
            self.deliver(decision(Some(onset), onset + 50, outcome));
        }

        fn run_out(&mut self) {
            while self.s.view().step != EvaluationStep::Finished {
                self.step();
            }
        }

        fn finish(self) -> EvaluationReport {
            let k = self.s.zones.len();
            self.s.finish(meta(k), run_at()).expect("the run finished")
        }
    }

    /// The same feed into a Scripted session built from the recorded prompts and negatives.
    fn replay(
        k: usize,
        prompts: Vec<ScriptedPrompt>,
        negatives: Vec<ScriptedNegative>,
        log: &[Feed],
    ) -> (Vec<EvaluationEvent>, EvaluationReport) {
        let mut s = EvaluationSession::new(
            zone_refs(k),
            PromptSchedule::Scripted { prompts, negatives },
            t(0),
        )
        .expect("a recorded script");
        let mut ev = Vec::new();
        for f in log {
            match f {
                Feed::Tick(now) | Feed::Undo(now) => s.tick(*now, &mut ev),
                Feed::Decision(d) => s.on_decision(d, &mut ev),
                Feed::Rejected(r, at) => s.on_detector_rejected(r, *at, &mut ev),
            }
        }
        (
            ev,
            s.finish(meta(k), run_at()).expect("the replay finished"),
        )
    }

    fn scripted(zones: &[u16], negatives: Vec<ScriptedNegative>) -> EvaluationSession {
        let prompts = zones
            .iter()
            .enumerate()
            .map(|(i, &z)| {
                let open = 1000 + 4000 * i as i64;
                ScriptedPrompt {
                    zone: ZoneIndex(z),
                    opened_at: t(open),
                    deadline: t(open + 3000),
                    reopened: false,
                }
            })
            .collect();
        let k = usize::from(zones.iter().copied().max().expect("zones") + 1);
        EvaluationSession::new(
            zone_refs(k),
            PromptSchedule::Scripted { prompts, negatives },
            t(0),
        )
        .expect("valid script")
    }

    #[test]
    fn e1_generated_schedule_is_balanced_deterministic_and_never_repeats() {
        for k in [2usize, 4, 6, 8] {
            for seed in 0..1000u64 {
                let order = generated_order(k, 15, seed);
                assert_eq!(order.len(), 15 * k);
                for z in 0..k {
                    assert_eq!(order.iter().filter(|o| usize::from(o.0) == z).count(), 15);
                }
                assert!(
                    order.windows(2).all(|w| w[0] != w[1]),
                    "K = {k}, seed {seed}: a zone twice in a row"
                );
                assert_eq!(order, generated_order(k, 15, seed), "deterministic");
            }
        }
        assert_ne!(generated_order(4, 15, 1), generated_order(4, 15, 2));

        // A full run opens the prompts in that order, and each confusion row sums to the quota.
        let k = 4;
        let mut d = Driver::new(generated(k, 15, 77, &[]), &[]);
        let mut opened = Vec::new();
        for i in 0..15 * k {
            match i % 4 {
                0 => d.tap_with(|z| accepted(z.0, 6.0)),
                1 => d.tap_with(|z| accepted((z.0 + 1) % 4, 6.0)),
                2 => d.tap_with(|_| rejected(ModelRejectReason::Ambiguous, 1.0)),
                _ => {
                    let (index, zone, _) = d.wait_prompt();
                    d.miss();
                    (index, zone)
                }
            };
            let EvaluationEvent::PromptOpened { zone, .. } = d
                .events
                .iter()
                .rev()
                .find(|e| matches!(e, EvaluationEvent::PromptOpened { .. }))
                .expect("a prompt opened")
                .clone()
            else {
                unreachable!()
            };
            opened.push(zone);
        }
        d.run_out();
        assert_eq!(opened, generated_order(k, 15, 77));
        let report = d.finish();
        for (row, m) in report.confusion.counts.iter().zip(&report.per_zone) {
            assert_eq!(row.iter().sum::<u32>(), 15);
            assert_eq!(m.prompts, 15);
        }
        assert_eq!(report.overall.prompts, 60);
    }

    #[test]
    fn e2_hand_built_outcomes_give_exact_metrics() {
        let zones = [0u16, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        let mut d = Driver::new(scripted(&zones, Vec::new()), &[]);
        d.tap(accepted(0, 5.0), 50); // z0 Correct
        d.tap(accepted(1, 5.0), 50); // z1 Correct
        d.tap(accepted(0, 5.0), 50); // z0 Correct
        d.tap(rejected(ModelRejectReason::OutOfDistribution, 0.5), 50); // z1
        d.tap(accepted(1, 5.0), 50); // z0 WrongZone
        d.tap(rejected(ModelRejectReason::Background, 0.5), 50); // z1
        d.tap(rejected(ModelRejectReason::Ambiguous, 0.5), 50); // z0
        d.tap(vetoed(5.0), 50); // z1
        d.faint(RejectReason::LowSnr); // z0
        d.tap(invalid(), 50); // z1
        d.miss(); // z0
        d.tap(accepted(0, 5.0), 50); // z1 WrongZone
        d.run_out();
        let r = d.finish();

        let z0 = &r.per_zone[0];
        assert_eq!((z0.prompts, z0.correct, z0.wrong, z0.missed), (6, 2, 1, 1));
        assert_eq!(
            z0.rejected,
            RejectionBreakdown {
                ambiguous: 1,
                detector: 1,
                ..RejectionBreakdown::default()
            }
        );
        assert_eq!(z0.recall, 2.0 / 6.0);
        let z1 = &r.per_zone[1];
        assert_eq!((z1.prompts, z1.correct, z1.wrong, z1.missed), (6, 1, 1, 0));
        assert_eq!(
            z1.rejected,
            RejectionBreakdown {
                out_of_distribution: 1,
                background: 1,
                vetoed: 1,
                invalid: 1,
                ..RejectionBreakdown::default()
            }
        );
        assert_eq!(z1.recall, 1.0 / 6.0);
        let o = &r.overall;
        assert_eq!(
            (o.prompts, o.correct, o.wrong, o.rejected, o.missed),
            (12, 3, 2, 6, 1)
        );
        assert_eq!(o.accuracy, 3.0 / 12.0);
        assert_eq!(o.wrong_action_rate, 2.0 / 5.0);
        assert_eq!(o.rejection_rate, 7.0 / 12.0);
        assert_eq!(
            r.confusion.counts,
            vec![vec![2, 1, 1, 0, 1, 0, 1], vec![1, 1, 2, 1, 0, 1, 0]]
        );
        assert_eq!(r.target, ACCURACY_TARGET);
        assert_eq!(r.prompts_per_zone, 6);

        // No accepts at all: wrong_action_rate is 0, not NaN.
        let mut d = Driver::new(scripted(&[0, 1], Vec::new()), &[]);
        d.miss();
        d.miss();
        d.run_out();
        let r = d.finish();
        assert_eq!(r.overall.wrong_action_rate, 0.0);
        assert_eq!(r.overall.rejection_rate, 1.0);
    }

    #[test]
    fn e3_negative_phase_false_accepts_per_hour() {
        let negatives = [(NegativeClass::Typing, 30.0), (NegativeClass::Speech, 45.0)];
        let mut d = Driver::new(generated(2, 1, 5, &negatives), &[30.0, 45.0]);
        d.tap_with(|z| accepted(z.0, 6.0));
        d.tap_with(|z| accepted(z.0, 6.0));
        let start = d.wait_negative();
        d.background(start, 1000, accepted(0, 6.0));
        d.background(start, 5000, rejected(ModelRejectReason::Background, 0.1));
        d.background(start, 9000, accepted(1, 6.0));
        d.background(start, 20_000, vetoed(6.0));
        d.background(start, 29_000, accepted(1, 6.0));
        d.tick_to(start + 30_000);
        let second = d.wait_negative();
        assert_eq!(second, start + 30_000, "phases run back to back");
        d.background(
            second,
            4000,
            rejected(ModelRejectReason::OutOfDistribution, 0.1),
        );
        d.run_out();
        let r = d.finish();
        let typing = &r.negatives[0];
        assert_eq!(typing.class, NegativeClass::Typing);
        assert_eq!(typing.duration, Seconds::new(30.0));
        assert_eq!((typing.candidates, typing.false_accepts), (5, 3));
        assert_eq!(typing.false_accepts_per_hour, 3600.0 * 3.0 / 30.0);
        assert_eq!(typing.false_accepts_by_zone, vec![1, 2]);
        let speech = &r.negatives[1];
        assert_eq!((speech.candidates, speech.false_accepts), (1, 0));
        assert_eq!(speech.false_accepts_per_hour, 0.0);
        // The background decisions never touched the prompts.
        assert_eq!(r.overall.correct, 2);
    }

    #[test]
    fn e4_latency_stats_nearest_rank_histogram_and_untimed() {
        let mut samples: Vec<f64> = (1..=100).map(f64::from).collect();
        samples.reverse();
        let s = LatencyStats::from_samples(&samples, 3);
        assert_eq!((s.count, s.untimed), (100, 3));
        assert_eq!(
            [s.p50, s.p95, s.p99, s.max].map(Milliseconds::get),
            [50.0, 95.0, 99.0, 100.0]
        );
        let s = LatencyStats::from_samples(&[30.0, 10.0, 20.0], 0);
        assert_eq!(
            [s.p50, s.p95, s.p99].map(Milliseconds::get),
            [20.0, 30.0, 30.0]
        );

        let s = LatencyStats::from_samples(&[0.0, 4.99, 5.0, 199.99, 200.0, 1000.0], 0);
        assert_eq!(s.histogram.len(), 41);
        let mut expected = vec![0u32; 41];
        expected[0] = 2;
        expected[1] = 1;
        expected[39] = 1;
        expected[40] = 2;
        assert_eq!(s.histogram, expected);

        let empty = LatencyStats::from_samples(&[], 0);
        assert_eq!(empty.count, 0);
        assert_eq!(
            [empty.p50, empty.p95, empty.p99, empty.max].map(Milliseconds::get),
            [0.0; 4]
        );
        assert_eq!(empty.histogram, vec![0; 41]);

        // Through a run: timed accepts give latencies, untimed ones are counted apart.
        let mut d = Driver::new(scripted(&[0, 1, 0, 1], Vec::new()), &[]);
        d.tap(accepted(0, 6.0), 50);
        d.tap(accepted(0, 6.0), 70); // WrongZone, timed
        d.untimed_tap(accepted(0, 6.0));
        d.tap(rejected(ModelRejectReason::Ambiguous, 1.0), 50); // no latency
        d.run_out();
        let r = d.finish();
        let l = &r.decision_latency;
        assert_eq!((l.count, l.untimed), (2, 1));
        assert_eq!([l.p50, l.max].map(Milliseconds::get), [50.0, 70.0]);
        assert_eq!(
            r.prompts[2].outcome,
            PromptOutcome::Correct { latency: None }
        );
    }

    /// Two zones (one label needs CSV quoting), four prompts and a 10 s typing check.
    fn small_report() -> EvaluationReport {
        let negatives = vec![ScriptedNegative {
            class: NegativeClass::Typing,
            start: t(20_000),
            end: t(30_000),
        }];
        let s = scripted(&[0, 1, 0, 1], negatives);
        let mut d = Driver::new(s, &[10.0]);
        d.s.zones[1].label = "Right, top".to_string();
        d.s.zones[0].label = "Left".to_string();
        d.tap(accepted(0, 6.5), 50);
        d.tap(accepted(0, 2.5), 60);
        d.tap(rejected(ModelRejectReason::Ambiguous, 1.25), 50);
        d.miss();
        let start = d.wait_negative();
        d.background(start, 2000, accepted(1, 9.0));
        d.background(start, 6000, rejected(ModelRejectReason::Background, 0.2));
        d.run_out();
        d.finish()
    }

    #[test]
    fn e5_json_round_trip_and_golden_csv() {
        let report = small_report();
        let json = report.to_json().expect("valid report");
        let back = EvaluationReport::from_json(&json, Path::new("report.json")).expect("loads");
        assert_eq!(back, report);

        let mut nan = report.clone();
        nan.overall.accuracy = f64::NAN;
        assert!(matches!(
            nan.to_json(),
            Err(PersistenceError::InvalidData { .. })
        ));
        let v1 = json.replacen("\"schema_version\": 2", "\"schema_version\": 1", 1);
        assert!(matches!(
            EvaluationReport::from_json(&v1, Path::new("old.json")),
            Err(PersistenceError::SchemaMismatch {
                found: 1,
                expected: 2,
                ..
            })
        ));
        let extra = json.replacen("{", "{\n  \"extra\": 1,", 1);
        assert!(matches!(
            EvaluationReport::from_json(&extra, Path::new("extra.json")),
            Err(PersistenceError::Json(_))
        ));

        let csv = report.to_csv();
        let blocks: Vec<&str> = csv.split("\n\n").collect();
        assert_eq!(blocks.len(), 5);
        assert_eq!(
            blocks[0],
            "metric,value\naccuracy,0.25\ntarget,0.92\nprompts,4\ncorrect,1\nwrong,1\n\
             rejected,1\nmissed,1\nundone,0\nwrong_action_rate,0.5\nrejection_rate,0.5\n\
             latency_p50_ms,50\nlatency_p95_ms,60\nlatency_p99_ms,60"
        );
        assert_eq!(
            blocks[1],
            "prompted,Left,\"Right, top\",rejected,vetoed,detector_rejected,invalid,missed,recall\n\
             Left,1,0,1,0,0,0,0,0.5\n\
             \"Right, top\",1,0,0,0,0,0,1,0"
        );
        assert_eq!(
            blocks[2],
            "zone,prompts,correct,wrong,rejected_ood,rejected_bg,rejected_ambiguous,vetoed,\
             detector,invalid,missed,recall\n\
             Left,2,1,0,0,0,1,0,0,0,0,0.5\n\
             \"Right, top\",2,0,1,0,0,0,0,0,0,1,0"
        );
        let mut margins = String::from("margin_bin,count");
        for i in 0..20 {
            let count = match i {
                1 | 2 | 6 => 1,
                _ => 0,
            };
            margins.push_str(&format!("\n{i}-{},{count}", i + 1));
        }
        margins.push_str("\n20+,0");
        assert_eq!(blocks[3], margins);
        assert_eq!(
            blocks[4],
            "class,duration_s,candidates,false_accepts,false_accepts_per_hour,Left,\"Right, top\"\n\
             Typing,10,2,1,360,0,1\n"
        );
        assert_eq!(
            report.to_prompts_csv(),
            "index,zone,outcome,latency_ms,margin\n\
             0,Left,Correct,50,6.5\n\
             1,\"Right, top\",WrongZone,60,2.5\n\
             2,Left,Rejected,,1.25\n\
             3,\"Right, top\",Missed,,\n"
        );
    }

    #[test]
    fn e6_scripted_replay_matches_the_generated_run_including_undos() {
        let k = 3;
        let mut d = Driver::new(
            generated(k, 3, 2024, &[(NegativeClass::Everyday, 2.0)]),
            &[2.0],
        );
        d.tap_with(|z| accepted(z.0, 6.0)); // idx 0 Correct
        d.tap_with(|z| accepted((z.0 + 1) % 3, 6.0)); // idx 1 WrongZone ...
        d.tick_to(d.now_ms + 300);
        d.undo().expect("undo in the gap"); // ... undone in the gap
        d.tap_with(|z| accepted(z.0, 6.0)); // idx 1 reopened: Correct
        d.tap_with(|_| vetoed(3.0)); // idx 2 Vetoed
        d.tap_with(|_| rejected(ModelRejectReason::OutOfDistribution, 0.3)); // idx 3 Rejected ...
        let (_, _, opened) = d.wait_prompt(); // idx 4 opens ...
        d.tick_to(opened + 100);
        d.undo().expect("undo with the next prompt open"); // ... and is withdrawn
        d.tap_with(|z| accepted(z.0, 6.0)); // idx 3 reopened: Correct
        d.faint(RejectReason::NotDecaying); // idx 4 again: DetectorRejected
        d.miss(); // idx 5 Missed
        d.tap_with(|_| invalid()); // idx 6 Invalid
        d.tap_with(|z| accepted(z.0, 6.0)); // idx 7
        d.tap_with(|z| accepted(z.0, 6.0)); // idx 8
        let start = d.wait_negative();
        d.background(start, 500, accepted(2, 6.0));
        d.background(start, 1500, rejected(ModelRejectReason::Background, 0.1));
        d.run_out();

        let script = d.script.clone();
        assert_eq!(script.iter().filter(|p| p.reopened).count(), 2);
        assert_eq!(
            script.len(),
            9 + 2 + 1,
            "9 prompts, 2 reopened, 1 withdrawn and reopened"
        );
        let (negatives, log, live_events) = (d.negatives.clone(), d.log.clone(), d.events.clone());
        let live = d.finish();
        assert_eq!(live.undone, 2);
        assert_eq!(live.overall.prompts, 9);

        let (events, replayed) = replay(k, script, negatives, &log);
        assert_eq!(events, live_events);
        assert_eq!(replayed, live);
    }

    #[test]
    fn e7_undo_last_rules() {
        let mut d = Driver::new(generated(4, 2, 9, &[(NegativeClass::Knocks, 1.0)]), &[1.0]);
        let (i0, z0) = d.tap_with(|z| accepted((z.0 + 1) % 4, 6.0)); // WrongZone
        assert_eq!(i0, 0);
        let (i1, z1, opened) = d.wait_prompt();
        assert_eq!(i1, 1);
        d.tick_to(opened + 100);
        d.undo()
            .expect("undo after a WrongZone, the next prompt open");
        assert_eq!(
            d.events.last(),
            Some(&EvaluationEvent::PromptReopened { index: 0, zone: z0 })
        );
        let EvaluationStep::Prompt {
            index,
            zone,
            deadline,
            ..
        } = d.s.view().step
        else {
            panic!("the reopened prompt is the step")
        };
        assert_eq!((index, zone), (0, z0));
        assert_eq!(
            ms_of(deadline),
            d.now_ms + 700 + 3000,
            "PROMPT_GAP, then a fresh window"
        );
        assert_eq!(
            d.undo(),
            Err(AssessmentError::NothingToUndo),
            "a second undo"
        );

        d.tap(accepted(z0.0, 6.0), 50); // the reopened prompt: Correct
        let (i, z, opened) = d.wait_prompt();
        assert_eq!((i, z), (i1, z1), "the withdrawn prompt follows, unchanged");
        d.tick_to(opened + 100);
        d.undo()
            .expect("a reopened prompt that resolved can be undone again");
        d.tap(accepted(z0.0, 6.0), 50);

        d.miss(); // idx 1 Missed
        d.tick_to(d.now_ms + 100);
        assert_eq!(
            d.undo(),
            Err(AssessmentError::NothingToUndo),
            "after Missed"
        );
        d.faint(RejectReason::LowSnr); // idx 2
        d.tick_to(d.now_ms + 100);
        assert_eq!(
            d.undo(),
            Err(AssessmentError::NothingToUndo),
            "after DetectorRejected"
        );
        d.tap(vetoed(5.0), 50); // idx 3
        d.tick_to(d.now_ms + 100);
        assert_eq!(
            d.undo(),
            Err(AssessmentError::NothingToUndo),
            "after Vetoed"
        );
        for _ in 4..8 {
            d.tap_with(|z| accepted(z.0, 6.0));
        }
        d.wait_negative();
        assert_eq!(
            d.undo(),
            Err(AssessmentError::NothingToUndo),
            "in a Negative step"
        );
        d.run_out();
        assert_eq!(
            d.undo(),
            Err(AssessmentError::NothingToUndo),
            "after Finished"
        );
        let r = d.finish();
        assert_eq!(r.undone, 2);
        assert!(matches!(
            r.prompts[0].outcome,
            PromptOutcome::Correct { .. }
        ));
        assert_eq!(r.prompts[1].outcome, PromptOutcome::Missed);
        assert_eq!(r.overall.prompts, 8);
    }

    #[test]
    fn e8_false_accepts_by_zone_and_prompt_records_match_the_decisions() {
        let mut d = Driver::new(
            generated(4, 1, 31, &[(NegativeClass::Everyday, 5.0)]),
            &[5.0],
        );
        let fed = [
            (6.0, 3.2),  // Correct
            (0.0, 0.4),  // WrongZone
            (1.0, 1.9),  // Rejected Ambiguous
            (2.0, 25.0), // Vetoed
        ];
        let mut zones = Vec::new();
        for &(kind, margin) in &fed {
            let (_, zone) = d.tap_with(|z| match kind as u8 {
                6 => accepted(z.0, margin),
                0 => accepted((z.0 + 1) % 4, margin),
                1 => rejected(ModelRejectReason::Ambiguous, margin),
                _ => vetoed(margin),
            });
            zones.push(zone);
        }
        let start = d.wait_negative();
        d.background(start, 500, accepted(0, 9.0));
        d.background(start, 1500, accepted(2, 9.0));
        d.background(start, 2500, rejected(ModelRejectReason::Background, 0.1));
        d.background(start, 3500, accepted(0, 9.0));
        d.run_out();
        let r = d.finish();
        assert_eq!(r.negatives[0].false_accepts_by_zone, vec![2, 0, 1, 0]);
        assert_eq!(r.negatives[0].false_accepts, 3);
        assert_eq!(r.negatives[0].candidates, 4);

        let expected: Vec<PromptRecord> = zones
            .iter()
            .zip(&fed)
            .map(|(z, &(kind, margin))| PromptRecord {
                zone: zone_refs(4)[usize::from(z.0)].id,
                outcome: match kind as u8 {
                    6 => PromptOutcome::Correct {
                        latency: Some(Milliseconds::new(50.0)),
                    },
                    0 => PromptOutcome::WrongZone {
                        got: ZoneIndex((z.0 + 1) % 4),
                        latency: Some(Milliseconds::new(50.0)),
                    },
                    1 => PromptOutcome::Rejected {
                        reason: ModelRejectReason::Ambiguous,
                    },
                    _ => PromptOutcome::Vetoed {
                        reason: VetoReason::Input {
                            kind: InputKind::KeyDown,
                            offset: Milliseconds::new(12.0),
                        },
                    },
                },
                margin: Some(margin),
            })
            .collect();
        assert_eq!(r.prompts, expected);
        let mut histogram = vec![0u32; 21];
        for bin in [3, 0, 1, 20] {
            histogram[bin] += 1;
        }
        assert_eq!(r.margin_histogram, histogram);
    }

    const GOLDEN_V1_REPORT: &str = r#"{
  "schema_version": 1,
  "profile_id": "00000000-0000-0000-0000-000000000001",
  "run_at": "2026-10-28T14:02:00Z",
  "taps_per_zone": 15,
  "per_zone_accuracy": {
    "00000000-0000-0000-0000-00000000000a": 1.0,
    "00000000-0000-0000-0000-00000000000b": 0.93333334
  },
  "confusion_matrix": [[15, 0], [1, 14]],
  "rejected_tap_count": 0,
  "confidence_distribution": { "min": 0.0, "max": 1.0, "bins": [0, 0, 1, 29] },
  "latency_ms": { "p50": 31.5, "p95": 44.0, "p99": 47.25, "max": 48.0 },
  "overall_accuracy": 0.96666664
}"#;

    #[test]
    fn e9_peek_legacy_report_reads_v1_headers() {
        let header =
            peek_legacy_report(GOLDEN_V1_REPORT, Path::new("v1.json")).expect("a v1 report");
        assert_eq!(
            header,
            LegacyReport {
                run_at: "2026-10-28T14:02:00Z".parse().expect("valid time"),
                overall_accuracy: 0.966_666_64,
                taps_per_zone: 15,
            }
        );
        let truncated = &GOLDEN_V1_REPORT[..GOLDEN_V1_REPORT.len() / 2];
        assert!(matches!(
            peek_legacy_report(truncated, Path::new("bad.json")),
            Err(PersistenceError::Json(_))
        ));
        let wrong_type =
            GOLDEN_V1_REPORT.replace("\"taps_per_zone\": 15", "\"taps_per_zone\": \"x\"");
        assert!(matches!(
            peek_legacy_report(&wrong_type, Path::new("bad.json")),
            Err(PersistenceError::Json(_))
        ));
        let v2 = small_report().to_json().expect("valid report");
        assert!(matches!(
            peek_legacy_report(&v2, Path::new("v2.json")),
            Err(PersistenceError::SchemaMismatch {
                found: 2,
                expected: 1,
                ..
            })
        ));
    }

    #[test]
    fn new_refuses_empty_or_disordered_schedules() {
        let refuse =
            |zones: Vec<ZoneRef>, schedule| EvaluationSession::new(zones, schedule, t(0)).err();
        assert_eq!(
            refuse(
                Vec::new(),
                PromptSchedule::Generated {
                    prompts_per_zone: 15,
                    seed: 1,
                    negatives: Vec::new()
                }
            ),
            Some(AssessmentError::NoZones)
        );
        assert_eq!(
            refuse(
                zone_refs(2),
                PromptSchedule::Generated {
                    prompts_per_zone: 0,
                    seed: 1,
                    negatives: Vec::new()
                }
            ),
            Some(AssessmentError::ZeroPrompts)
        );
        assert_eq!(
            refuse(
                zone_refs(2),
                PromptSchedule::Scripted {
                    prompts: Vec::new(),
                    negatives: Vec::new()
                }
            ),
            Some(AssessmentError::ZeroPrompts)
        );
        let prompt = |zone: u16, open: i64, reopened: bool| ScriptedPrompt {
            zone: ZoneIndex(zone),
            opened_at: t(open),
            deadline: t(open + 3000),
            reopened,
        };
        let unordered = [
            vec![prompt(0, 5000, false), prompt(1, 1000, false)],
            vec![prompt(2, 1000, false)],
            vec![prompt(0, 1000, true)],
            vec![prompt(0, 1000, false), prompt(1, 5000, true)],
        ];
        for prompts in unordered {
            assert_eq!(
                refuse(
                    zone_refs(2),
                    PromptSchedule::Scripted {
                        prompts,
                        negatives: Vec::new()
                    }
                ),
                Some(AssessmentError::ScriptUnordered)
            );
        }
        let overlapping = vec![
            ScriptedNegative {
                class: NegativeClass::Music,
                start: t(9000),
                end: t(12_000),
            },
            ScriptedNegative {
                class: NegativeClass::Music,
                start: t(11_000),
                end: t(14_000),
            },
        ];
        assert_eq!(
            refuse(
                zone_refs(2),
                PromptSchedule::Scripted {
                    prompts: vec![prompt(0, 1000, false)],
                    negatives: overlapping
                }
            ),
            Some(AssessmentError::ScriptUnordered)
        );
        let s = generated(2, 1, 1, &[]);
        assert_eq!(
            s.finish(meta(2), run_at()).err(),
            Some(AssessmentError::NotFinished)
        );
    }
}
