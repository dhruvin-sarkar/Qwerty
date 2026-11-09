//! Input and self-output veto: OS input events and Qwerty's own sound output that disqualify a candidate (§1.13).

use crate::units::{HostDuration, HostTime, Milliseconds};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InputKind {
    KeyDown,
    KeyUp,
    MouseButtonDown,
    MouseButtonUp,
}

/// Coarse OS input event. Never carries key identity, position or device (ADR-004 privacy).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputEvent {
    pub time: HostTime,
    pub kind: InputKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SelfOutputKind {
    Speech,
    Sound,
    Toast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OutputId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SelfOutputEvent {
    Began {
        id: OutputId,
        kind: SelfOutputKind,
        at: HostTime,
    },
    Ended {
        id: OutputId,
        at: HostTime,
    },
}

// ---- Timeline (§1.13.2, WP-A4). The shared types above belong to WP-00 and stay as they are. ----

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VetoParams {
    /// An input event up to this long BEFORE t0 vetoes (t0 − e ≤ +40 ms).
    pub input_before: Milliseconds,
    /// An input event up to this long AFTER t0 vetoes (t0 − e ≥ −30 ms).
    pub input_after: Milliseconds,
    /// Delivery margin for OS input events (deviation l).
    pub settle: Milliseconds,
    /// A Speech or Sound output vetoes until this long after it ends (ADR-004).
    pub self_output_tail: Milliseconds,
    /// A Toast vetoes this long after it begins, plus the tail (deviation e).
    pub toast_allowance: Milliseconds,
    /// An output open longer than this is closed by `close_stuck`.
    pub stuck_output_limit: Milliseconds,
    /// Input events older than newest − horizon are pruned.
    pub horizon: Milliseconds,
}

impl VetoParams {
    pub const DEFAULT: VetoParams = VetoParams {
        input_before: Milliseconds::new(40.0),
        input_after: Milliseconds::new(30.0),
        settle: Milliseconds::new(15.0),
        self_output_tail: Milliseconds::new(200.0),
        toast_allowance: Milliseconds::new(1500.0),
        stuck_output_limit: Milliseconds::new(120_000.0),
        horizon: Milliseconds::new(2000.0),
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum VetoReason {
    /// `offset` = t0 − e.
    Input {
        kind: InputKind,
        offset: Milliseconds,
    },
    SelfOutput {
        kind: SelfOutputKind,
    },
    /// The OS-input ring overflowed during the veto window.
    InputOverflow,
    /// The candidate is untimed.
    ClockUnavailable,
    /// Pipeline pending-decision queue overflow (§1.15).
    DecisionOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum VetoVerdict {
    Clear,
    Vetoed { reason: VetoReason },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum VetoDecision {
    Pending { until: HostTime },
    Final { verdict: VetoVerdict },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct VetoCounters {
    /// Input events stored (late ones are counted in `inputs_pruned_late` instead).
    pub inputs: u64,
    /// Input events already older than newest − horizon on arrival, dropped.
    pub inputs_pruned_late: u64,
    /// Gaps recorded: one per evicted input event and one per `mark_input_gap`.
    pub gaps: u64,
    /// Self-outputs begun.
    pub outputs: u64,
    pub stuck_closed: u64,
    pub unmatched_ends: u64,
}

/// Stored input events (§1.21 I14). At the cap the oldest is evicted and leaves a gap.
const INPUT_CAP: usize = 512;
/// Stored self-outputs, open and closed (§1.21 I14).
const OUTPUT_CAP: usize = 16;
/// Stored gap spans. Beyond it the two oldest merge into one span, which can only widen a veto.
// SPEC: §1.13.2 bounds events (512) and outputs (16) but not gaps; I14 needs a bound, so 16 here.
const GAP_CAP: usize = 16;

/// The parameters in host ticks, converted once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Spans {
    before: HostDuration,
    after: HostDuration,
    settle: HostDuration,
    tail: HostDuration,
    toast: HostDuration,
    stuck: HostDuration,
    horizon: HostDuration,
}

/// A span of host time in which input events may have been lost; closed at both ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Gap {
    from: HostTime,
    to: HostTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Output {
    id: OutputId,
    kind: SelfOutputKind,
    start: HostTime,
    /// `None` while open.
    end: Option<HostTime>,
}

/// The input and self-output history candidates are judged against. Its owner (the processing
/// thread, replay or the soak) keeps one for its whole lifetime (RUST-RT-9). Allocates only in
/// `new` and when `close_stuck` returns ids.
#[derive(Debug, Clone, PartialEq)]
pub struct VetoTimeline {
    spans: Spans,
    /// Sorted by time; equal times keep arrival order.
    inputs: VecDeque<InputEvent>,
    /// Sorted by `from`, disjoint.
    gaps: Vec<Gap>,
    /// In `Began` order.
    outputs: Vec<Output>,
    counters: VetoCounters,
}

impl VetoTimeline {
    pub fn new(params: VetoParams) -> Self {
        VetoTimeline {
            spans: Spans {
                before: HostDuration::from_ms(params.input_before),
                after: HostDuration::from_ms(params.input_after),
                settle: HostDuration::from_ms(params.settle),
                tail: HostDuration::from_ms(params.self_output_tail),
                toast: HostDuration::from_ms(params.toast_allowance),
                stuck: HostDuration::from_ms(params.stuck_output_limit),
                horizon: HostDuration::from_ms(params.horizon),
            },
            // One spare slot each: an insert lands before the eviction or merge that restores the cap.
            inputs: VecDeque::with_capacity(INPUT_CAP + 1),
            gaps: Vec::with_capacity(GAP_CAP + 1),
            outputs: Vec::with_capacity(OUTPUT_CAP),
            counters: VetoCounters::default(),
        }
    }

    /// Insertion-sorted; an event older than newest − horizon is dropped and counted. At the
    /// 512-event cap the oldest is evicted and a gap from its time to the new oldest event's time
    /// is recorded, so a window that reaches the evicted span gives `InputOverflow`, never a
    /// false `Clear` (RUST-RT-22).
    pub fn push_input(&mut self, e: InputEvent) {
        let newest = match self.inputs.back() {
            Some(back) => {
                if e.time < before(back.time, self.spans.horizon) {
                    self.counters.inputs_pruned_late += 1;
                    return;
                }
                back.time.max(e.time)
            }
            None => e.time,
        };
        let at = self.inputs.partition_point(|x| x.time <= e.time);
        self.inputs.insert(at, e);
        self.counters.inputs += 1;

        let cutoff = before(newest, self.spans.horizon);
        while self.inputs.front().is_some_and(|x| x.time < cutoff) {
            self.inputs.pop_front();
        }
        if self.inputs.len() > INPUT_CAP {
            if let (Some(evicted), Some(oldest)) =
                (self.inputs.pop_front(), self.inputs.front().copied())
            {
                self.record_gap(Gap {
                    from: evicted.time,
                    to: oldest.time,
                });
            }
        }
    }

    /// Input events may be missing in [from, to] (both ends inclusive; given in either order).
    pub fn mark_input_gap(&mut self, from: HostTime, to: HostTime) {
        self.record_gap(Gap {
            from: from.min(to),
            to: from.max(to),
        });
    }

    /// `Began{Speech}` closes any open Speech (interrupt). An `Ended` for an unknown or
    /// already-closed id is ignored and counted in `unmatched_ends` (processing maps speech ends
    /// to ids through a FIFO; X01-07).
    pub fn self_output(&mut self, e: SelfOutputEvent) {
        match e {
            SelfOutputEvent::Began { id, kind, at } => {
                self.counters.outputs += 1;
                if kind == SelfOutputKind::Speech {
                    for o in &mut self.outputs {
                        if o.kind == SelfOutputKind::Speech && o.end.is_none() {
                            o.end = Some(at);
                        }
                    }
                }
                if self.outputs.len() == OUTPUT_CAP {
                    self.evict_output();
                }
                self.outputs.push(Output {
                    id,
                    kind,
                    start: at,
                    end: None,
                });
            }
            SelfOutputEvent::Ended { id, at } => {
                match self
                    .outputs
                    .iter_mut()
                    .find(|o| o.id == id && o.end.is_none())
                {
                    Some(o) => o.end = Some(at),
                    None => self.counters.unmatched_ends += 1,
                }
            }
        }
    }

    /// `Pending` until now = t0 + input_after + settle; then the first match of: a gap in the
    /// window, an input event in the window, a self-output interval containing t0.
    pub fn decide(&self, t0: HostTime, now: HostTime) -> VetoDecision {
        let until = after(
            t0,
            HostDuration(self.spans.after.0.saturating_add(self.spans.settle.0)),
        );
        if now < until {
            return VetoDecision::Pending { until };
        }
        VetoDecision::Final {
            verdict: self.verdict(t0, now),
        }
    }

    /// Drops input events, gaps and closed outputs that end before now − horizon. Open outputs
    /// stay until they end or `close_stuck` closes them.
    pub fn prune(&mut self, now: HostTime) {
        let cutoff = before(now, self.spans.horizon);
        while self.inputs.front().is_some_and(|x| x.time < cutoff) {
            self.inputs.pop_front();
        }
        self.gaps.retain(|g| g.to >= cutoff);
        let spans = self.spans;
        self.outputs
            .retain(|o| o.end.is_none() || veto_interval(o, &spans, now).1 >= cutoff);
    }

    /// Outputs open longer than stuck_output_limit: closed at `now`, returned for a HealthWarning.
    pub fn close_stuck(&mut self, now: HostTime) -> Vec<OutputId> {
        let mut closed = Vec::new();
        for o in &mut self.outputs {
            if o.end.is_none() && now.since(o.start) > self.spans.stuck {
                o.end = Some(now);
                closed.push(o.id);
            }
        }
        self.counters.stuck_closed += closed.len() as u64;
        closed
    }

    pub fn counters(&self) -> VetoCounters {
        self.counters
    }

    /// The settled verdict: the first of gap, input event, self-output that matches.
    fn verdict(&self, t0: HostTime, now: HostTime) -> VetoVerdict {
        let lo = before(t0, self.spans.before);
        let hi = after(t0, self.spans.after);

        if self.gaps.iter().any(|g| g.from <= hi && g.to >= lo) {
            return VetoVerdict::Vetoed {
                reason: VetoReason::InputOverflow,
            };
        }

        // SPEC: §1.13.2 does not break a tie between two events equally near t0; the earlier
        // one (then the first to arrive) decides, which keeps the verdict deterministic.
        let first = self.inputs.partition_point(|x| x.time < lo);
        let mut nearest: Option<(u64, InputEvent)> = None;
        for e in self.inputs.range(first..) {
            if e.time > hi {
                break;
            }
            let distance = t0.since(e.time).0.unsigned_abs();
            if nearest.is_none_or(|(best, _)| distance < best) {
                nearest = Some((distance, *e));
            }
        }
        if let Some((_, e)) = nearest {
            return VetoVerdict::Vetoed {
                reason: VetoReason::Input {
                    kind: e.kind,
                    offset: t0.since(e.time).as_ms(),
                },
            };
        }

        for o in &self.outputs {
            let (start, end) = veto_interval(o, &self.spans, now);
            if start <= t0 && t0 <= end {
                return VetoVerdict::Vetoed {
                    reason: VetoReason::SelfOutput { kind: o.kind },
                };
            }
        }
        VetoVerdict::Clear
    }

    /// Adds a gap, keeping `gaps` sorted, disjoint and within GAP_CAP.
    fn record_gap(&mut self, gap: Gap) {
        self.counters.gaps += 1;
        let at = self.gaps.partition_point(|g| g.from <= gap.from);
        self.gaps.insert(at, gap);
        let mut i = 0;
        while i + 1 < self.gaps.len() {
            if self.gaps[i + 1].from <= self.gaps[i].to {
                let next = self.gaps.remove(i + 1);
                self.gaps[i].to = self.gaps[i].to.max(next.to);
            } else {
                i += 1;
            }
        }
        if self.gaps.len() > GAP_CAP {
            // Widening the oldest span over its neighbour can only turn a verdict into
            // InputOverflow, never into a false Clear.
            let second = self.gaps.remove(1);
            self.gaps[0].to = self.gaps[0].to.max(second.to);
        }
    }

    /// Frees one output slot: the closed output whose veto ends first, else the oldest open one.
    // SPEC: §1.13.2 caps outputs at 16 but names no eviction rule. Only a closed output's interval
    // is complete, so the one ending first goes; 16 outputs open at once (speech closes speech,
    // sounds play synchronously) would drop the oldest.
    fn evict_output(&mut self) {
        let spans = self.spans;
        let closed = self
            .outputs
            .iter()
            .enumerate()
            .filter_map(|(i, o)| o.end.map(|end| (i, veto_interval(o, &spans, end).1)))
            .min_by_key(|&(_, end)| end)
            .map(|(i, _)| i);
        let oldest_open = self
            .outputs
            .iter()
            .enumerate()
            .min_by_key(|(_, o)| o.start)
            .map(|(i, _)| i);
        if let Some(i) = closed.or(oldest_open) {
            self.outputs.remove(i);
        }
    }
}

/// The closed interval of host time in which `o` vetoes a candidate: Speech and Sound from their
/// start to (end, or `now` while open) + tail; Toast from its start to start + allowance + tail.
fn veto_interval(o: &Output, spans: &Spans, now: HostTime) -> (HostTime, HostTime) {
    let end = match o.kind {
        SelfOutputKind::Speech | SelfOutputKind::Sound => match o.end {
            Some(end) => end,
            None => now,
        },
        SelfOutputKind::Toast => after(o.start, spans.toast),
    };
    (o.start, after(end, spans.tail))
}

/// `t + d`, saturating at the i64 tick range, which real host times never approach.
fn after(t: HostTime, d: HostDuration) -> HostTime {
    HostTime(t.0.saturating_add(d.0))
}

/// `t − d`, saturating at the i64 tick range.
fn before(t: HostTime, d: HostDuration) -> HostTime {
    HostTime(t.0.saturating_sub(d.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 100-ns ticks per millisecond.
    const TICKS_PER_MS: f64 = 10_000.0;
    /// Every test timeline starts here, well away from zero.
    const BASE_TICKS: i64 = 1_000 * 10_000_000;
    const T0_MS: f64 = 10_000.0;

    fn at(ms: f64) -> HostTime {
        HostTime(BASE_TICKS + (ms * TICKS_PER_MS).round() as i64)
    }

    fn key(ms: f64) -> InputEvent {
        InputEvent {
            time: at(ms),
            kind: InputKind::KeyDown,
        }
    }

    fn timeline() -> VetoTimeline {
        VetoTimeline::new(VetoParams::DEFAULT)
    }

    /// The final verdict for a candidate at `t0_ms`, decided long after it.
    fn verdict(v: &VetoTimeline, t0_ms: f64) -> VetoVerdict {
        match v.decide(at(t0_ms), at(t0_ms + 1000.0)) {
            VetoDecision::Final { verdict } => verdict,
            VetoDecision::Pending { until } => panic!("still pending until {until:?}"),
        }
    }

    fn vetoed(reason: VetoReason) -> VetoVerdict {
        VetoVerdict::Vetoed { reason }
    }

    fn input(kind: InputKind, offset_ms: f64) -> VetoVerdict {
        vetoed(VetoReason::Input {
            kind,
            offset: Milliseconds::new(offset_ms),
        })
    }

    fn self_out(kind: SelfOutputKind) -> VetoVerdict {
        vetoed(VetoReason::SelfOutput { kind })
    }

    fn began(id: u64, kind: SelfOutputKind, ms: f64) -> SelfOutputEvent {
        SelfOutputEvent::Began {
            id: OutputId(id),
            kind,
            at: at(ms),
        }
    }

    fn ended(id: u64, ms: f64) -> SelfOutputEvent {
        SelfOutputEvent::Ended {
            id: OutputId(id),
            at: at(ms),
        }
    }

    fn stored_times(v: &VetoTimeline) -> Vec<HostTime> {
        v.inputs.iter().map(|e| e.time).collect()
    }

    #[test]
    fn v1_input_window_bounds_are_inclusive() {
        let cases = [
            (T0_MS - 40.0, Some(40.0)),
            (T0_MS - 40.1, None),
            (T0_MS + 30.0, Some(-30.0)),
            (T0_MS + 30.1, None),
        ];
        for (e_ms, offset) in cases {
            let mut v = timeline();
            v.push_input(key(e_ms));
            let expected = match offset {
                Some(o) => input(InputKind::KeyDown, o),
                None => VetoVerdict::Clear,
            };
            assert_eq!(
                verdict(&v, T0_MS),
                expected,
                "event at t0 {:+} ms",
                e_ms - T0_MS
            );
        }

        // Several events in the window: the one nearest t0 decides.
        let mut v = timeline();
        v.push_input(key(T0_MS - 35.0));
        v.push_input(InputEvent {
            time: at(T0_MS + 5.0),
            kind: InputKind::MouseButtonUp,
        });
        v.push_input(key(T0_MS + 25.0));
        assert_eq!(verdict(&v, T0_MS), input(InputKind::MouseButtonUp, -5.0));
    }

    #[test]
    fn v2_pending_until_settled_then_deterministic() {
        let mut v = timeline();
        v.push_input(key(T0_MS + 20.0));
        let until = at(T0_MS + 45.0);
        assert_eq!(
            v.decide(at(T0_MS), at(T0_MS)),
            VetoDecision::Pending { until }
        );
        assert_eq!(
            v.decide(at(T0_MS), HostTime(until.0 - 1)),
            VetoDecision::Pending { until }
        );
        let expected = VetoDecision::Final {
            verdict: input(InputKind::KeyDown, -20.0),
        };
        for later_ms in [45.0, 46.0, 500.0, 1999.0] {
            assert_eq!(v.decide(at(T0_MS), at(T0_MS + later_ms)), expected);
        }
        // An empty timeline is Clear once settled.
        assert_eq!(
            timeline().decide(at(T0_MS), until),
            VetoDecision::Final {
                verdict: VetoVerdict::Clear
            }
        );
    }

    #[test]
    fn v3_self_output_intervals() {
        let speech = self_out(SelfOutputKind::Speech);

        // An open Speech interval vetoes; before it began does not.
        let mut v = timeline();
        v.self_output(began(1, SelfOutputKind::Speech, 1000.0));
        assert_eq!(verdict(&v, 1000.0), speech);
        assert_eq!(verdict(&v, 5000.0), speech);
        assert_eq!(verdict(&v, 999.9), VetoVerdict::Clear);

        // Ended at e vetoes until e + 200 ms.
        v.self_output(ended(1, 3000.0));
        assert_eq!(verdict(&v, 3200.0), speech);
        assert_eq!(verdict(&v, 3200.1), VetoVerdict::Clear);
        assert_eq!(v.counters().unmatched_ends, 0);

        // A Sound behaves the same way.
        let mut v = timeline();
        v.self_output(began(1, SelfOutputKind::Sound, 1000.0));
        v.self_output(ended(1, 1100.0));
        assert_eq!(verdict(&v, 1300.0), self_out(SelfOutputKind::Sound));
        assert_eq!(verdict(&v, 1300.1), VetoVerdict::Clear);

        // A Toast vetoes on [s, s + 1700 ms], whenever it ends.
        let toast = self_out(SelfOutputKind::Toast);
        let mut v = timeline();
        v.self_output(began(1, SelfOutputKind::Toast, 1000.0));
        assert_eq!(verdict(&v, 999.9), VetoVerdict::Clear);
        assert_eq!(verdict(&v, 1000.0), toast);
        assert_eq!(verdict(&v, 2700.0), toast);
        assert_eq!(verdict(&v, 2700.1), VetoVerdict::Clear);
        v.self_output(ended(1, 1050.0));
        assert_eq!(verdict(&v, 2700.0), toast);
        assert_eq!(v.counters().unmatched_ends, 0);

        // A new Speech Began closes the previous one: once speech 2 has ended, nothing holds the
        // veto open, as a still-open speech 1 would.
        let mut v = timeline();
        v.self_output(began(1, SelfOutputKind::Speech, 1000.0));
        v.self_output(began(2, SelfOutputKind::Speech, 2000.0));
        v.self_output(ended(2, 2500.0));
        assert_eq!(verdict(&v, 1500.0), speech);
        assert_eq!(verdict(&v, 2700.0), speech);
        assert_eq!(verdict(&v, 2700.1), VetoVerdict::Clear);
        assert_eq!(verdict(&v, 5000.0), VetoVerdict::Clear);
        assert_eq!(v.counters().outputs, 2);

        // An Ended for the id that Began closed, or for one never opened, changes nothing and
        // adds 1 to unmatched_ends.
        let before = v.clone();
        v.self_output(ended(1, 2600.0));
        assert_eq!(v.counters().unmatched_ends, 1);
        v.self_output(ended(99, 2600.0));
        assert_eq!(v.counters().unmatched_ends, 2);
        let mut expected = before;
        expected.counters.unmatched_ends = 2;
        assert_eq!(v, expected);
    }

    #[test]
    fn v4_bounded_memory_and_overflow_gap() {
        // 10^6 pushes (1 ms apart, jittered out of order) keep at most 512 stored.
        let mut v = timeline();
        let input_capacity = v.inputs.capacity();
        let gap_capacity = v.gaps.capacity();
        let mut newest = at(0.0);
        for i in 0..1_000_000_i64 {
            let jitter = (i * 7919) % 37;
            let time = HostTime(BASE_TICKS + (i - jitter) * 10_000);
            v.push_input(InputEvent {
                time,
                kind: InputKind::KeyUp,
            });
            assert!(v.inputs.len() <= INPUT_CAP);
            assert!(v.gaps.len() <= GAP_CAP);
            newest = newest.max(time);
        }
        assert_eq!(v.inputs.capacity(), input_capacity);
        assert_eq!(v.gaps.capacity(), gap_capacity);
        assert!(stored_times(&v).windows(2).all(|w| w[0] <= w[1]));
        // prune bounds memory: once everything is past the horizon nothing is kept.
        v.prune(HostTime(newest.0 + 20_000_001));
        assert!(v.inputs.is_empty());
        assert!(v.gaps.is_empty());

        // 600 events 3 ms apart (inside one 2 s horizon) evict 88 and record the matching gap.
        let mut v = timeline();
        let t = |i: usize| 1000.0 + 3.0 * i as f64;
        for i in 0..600 {
            v.push_input(key(t(i)));
        }
        assert_eq!(v.inputs.len(), INPUT_CAP);
        assert_eq!(v.inputs.front().map(|e| e.time), Some(at(t(88))));
        assert_eq!(v.counters().gaps, 88);
        assert_eq!(v.counters().inputs, 600);
        assert_eq!(
            v.gaps,
            vec![Gap {
                from: at(t(0)),
                to: at(t(88)),
            }]
        );
        // A candidate whose window covers an evicted event is never Clear.
        assert_eq!(verdict(&v, t(10)), vetoed(VetoReason::InputOverflow));
        assert_eq!(verdict(&v, t(0) - 30.0), vetoed(VetoReason::InputOverflow));
        // One whose window ends before the evicted span is untouched; stored events still decide.
        assert_eq!(verdict(&v, t(0) - 30.1), VetoVerdict::Clear);
        assert_eq!(verdict(&v, t(300)), input(InputKind::KeyDown, 0.0));

        // mark_input_gap gives InputOverflow over its span, inclusive.
        let mut v = timeline();
        v.mark_input_gap(at(5100.0), at(5000.0));
        assert_eq!(v.counters().gaps, 1);
        assert_eq!(verdict(&v, 5140.0), vetoed(VetoReason::InputOverflow));
        assert_eq!(verdict(&v, 5140.1), VetoVerdict::Clear);
        assert_eq!(verdict(&v, 4970.0), vetoed(VetoReason::InputOverflow));
        assert_eq!(verdict(&v, 4969.9), VetoVerdict::Clear);
        // Many disjoint gaps stay bounded and never lose coverage.
        for i in 0..100 {
            let s = 10_000.0 + 1000.0 * f64::from(i);
            v.mark_input_gap(at(s), at(s + 10.0));
            assert!(v.gaps.len() <= GAP_CAP);
        }
        assert_eq!(v.gaps.capacity(), gap_capacity);
        for i in 0..100 {
            let s = 10_000.0 + 1000.0 * f64::from(i);
            assert_eq!(verdict(&v, s + 5.0), vetoed(VetoReason::InputOverflow));
        }
    }

    #[test]
    fn v5_out_of_order_and_late_inputs() {
        let mut v = timeline();
        v.push_input(key(1000.0));
        v.push_input(key(1100.0));
        v.push_input(key(1050.0));
        assert_eq!(stored_times(&v), vec![at(1000.0), at(1050.0), at(1100.0)]);
        assert_eq!(verdict(&v, 1060.0), input(InputKind::KeyDown, 10.0));

        // newest = 5000 ms: older than 3000 ms is late; exactly 3000 ms is kept.
        v.push_input(key(5000.0));
        assert_eq!(stored_times(&v), vec![at(5000.0)]);
        v.push_input(key(2999.9));
        assert_eq!(v.counters().inputs_pruned_late, 1);
        v.push_input(key(3000.0));
        assert_eq!(stored_times(&v), vec![at(3000.0), at(5000.0)]);
        assert_eq!(v.counters().inputs_pruned_late, 1);
        assert_eq!(v.counters().inputs, 5);
    }

    #[test]
    fn v6_close_stuck_outputs() {
        let mut v = timeline();
        v.self_output(began(7, SelfOutputKind::Sound, 1000.0));
        v.self_output(began(8, SelfOutputKind::Speech, 50_000.0));
        // Open exactly 120 s is not stuck.
        assert_eq!(v.close_stuck(at(121_000.0)), vec![]);
        assert_eq!(v.close_stuck(at(121_000.1)), vec![OutputId(7)]);
        assert_eq!(v.counters().stuck_closed, 1);
        assert_eq!(v.close_stuck(at(121_000.1)), vec![]);
        // Closed at `now`: it vetoes until now + tail, then stops.
        assert_eq!(v.close_stuck(at(170_000.1)), vec![OutputId(8)]);
        assert_eq!(v.counters().stuck_closed, 2);
        assert_eq!(verdict(&v, 170_200.1), self_out(SelfOutputKind::Speech));
        assert_eq!(verdict(&v, 170_200.2), VetoVerdict::Clear);
        // Its real end arrives later: already closed, so it is unmatched.
        v.self_output(ended(7, 171_000.0));
        assert_eq!(v.counters().unmatched_ends, 1);
    }
}
