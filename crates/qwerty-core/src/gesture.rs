//! Gesture recognition over classified taps: single and double triggers with a refractory (§1.12).
//!
//! `Gesture` sees every detector-passed candidate once, in onset order, with its final outcome
//! (accepted for a zone, vetoed or rejected). It turns that stream into user-visible events: a
//! `Fire` for a single tap or a completed double tap, the arming and disarming of a pending
//! double tap, and the burst hold that suppresses everything while the desk is busy (typing,
//! drumming, a dropped object bouncing).

use crate::units::{CandidateId, HostDuration, HostTime, Milliseconds, ZoneIndex};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// How a zone fires: on one tap, or on two taps of the same zone within the double window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Trigger {
    Single,
    Double,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GestureParams {
    /// Minimum spacing between two processed candidates (ADR-004).
    pub refractory: Milliseconds,
    /// Window in which `burst_count` candidates start a burst hold.
    pub burst_window: Milliseconds,
    pub burst_count: u32,
    /// A hold ends this long after the last candidate it saw.
    pub burst_quiet: Milliseconds,
    /// Inclusive bounds on the spacing of the two taps of a double.
    pub double_min: Milliseconds,
    pub double_max: Milliseconds,
}

impl GestureParams {
    pub const DEFAULT: GestureParams = GestureParams {
        refractory: Milliseconds::new(120.0),
        burst_window: Milliseconds::new(600.0),
        burst_count: 3,
        burst_quiet: Milliseconds::new(300.0),
        double_min: Milliseconds::new(120.0),
        double_max: Milliseconds::new(450.0),
    };
}

/// The final outcome of one detector-passed candidate, as the pipeline resolved it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CandidateOutcome {
    Accepted { zone: ZoneIndex },
    Vetoed,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GestureEvent {
    /// The zone's action fires. `first` is the first tap of a double.
    Fire {
        zone: ZoneIndex,
        candidate: CandidateId,
        t0: HostTime,
        first: Option<CandidateId>,
    },
    /// The first tap of a double: a second tap of the same zone up to `expires_at` fires.
    Armed {
        zone: ZoneIndex,
        candidate: CandidateId,
        expires_at: HostTime,
    },
    Disarmed {
        zone: ZoneIndex,
        reason: DisarmReason,
    },
    DoubleExpired {
        zone: ZoneIndex,
    },
    Suppressed {
        candidate: CandidateId,
        reason: SuppressReason,
    },
    BurstHeld {
        since: HostTime,
    },
    BurstReleased {
        at: HostTime,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DisarmReason {
    Interrupted,
    OtherZone,
    BurstHold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SuppressReason {
    Refractory,
    Burst,
}

/// Capacity of the recent-onset ring (§1.21 I14). Only `|recent| ≥ burst_count` is ever read, so
/// the cap changes no decision while `burst_count ≤ RECENT_CAP` (the default is 3).
const RECENT_CAP: usize = 16;

/// A pending double tap: the first tap's zone, candidate and onset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Arm {
    zone: ZoneIndex,
    candidate: CandidateId,
    t0: HostTime,
    expires_at: HostTime,
}

/// Per-pipeline gesture state. Allocates only in `new`; every later call writes into the
/// caller's `out` and touches fixed-size state.
#[derive(Debug, Clone, PartialEq)]
pub struct Gesture {
    /// Indexed by `ZoneIndex`.
    triggers: Vec<Trigger>,
    params: GestureParams,
    /// Onsets of every candidate within the last `burst_window`, oldest first.
    recent: VecDeque<HostTime>,
    /// The refractory anchor: the onset of the last candidate that was not suppressed.
    last_unsuppressed: Option<HostTime>,
    armed: Option<Arm>,
    /// `Some` while a burst hold is active.
    hold_until: Option<HostTime>,
}

impl Gesture {
    /// `triggers[z]` is zone `z`'s trigger; it MUST cover every zone the model can accept.
    pub fn new(triggers: Vec<Trigger>, params: GestureParams) -> Self {
        Gesture {
            triggers,
            params,
            recent: VecDeque::with_capacity(RECENT_CAP),
            last_unsuppressed: None,
            armed: None,
            hold_until: None,
        }
    }

    /// Feed EVERY detector-passed candidate exactly once, in onset order. Vetoed and rejected ones count too.
    ///
    /// # Panics
    /// If an `Accepted` zone has no entry in `triggers` (a caller bug: the triggers and the model
    /// come from one profile).
    pub fn on_candidate(
        &mut self,
        id: CandidateId,
        t0: HostTime,
        outcome: CandidateOutcome,
        out: &mut Vec<GestureEvent>,
    ) {
        // 1. Expire the arm and release the hold up to this onset.
        self.advance_to(t0, out);

        // 2. Every candidate counts toward a burst, whatever its outcome.
        let window = HostDuration::from_ms(self.params.burst_window);
        while let Some(&oldest) = self.recent.front() {
            if t0.since(oldest) > window {
                self.recent.pop_front();
            } else {
                break;
            }
        }
        if self.recent.len() == RECENT_CAP {
            self.recent.pop_front();
        }
        self.recent.push_back(t0);

        // 3. A hold swallows the candidate and extends itself.
        let quiet = HostDuration::from_ms(self.params.burst_quiet);
        if self.hold_until.is_some() {
            self.hold_until = Some(after(t0, quiet));
            out.push(GestureEvent::Suppressed {
                candidate: id,
                reason: SuppressReason::Burst,
            });
            return;
        }

        // 4. Too many candidates within the window start a hold.
        if self.recent.len() >= self.params.burst_count as usize {
            self.hold_until = Some(after(t0, quiet));
            out.push(GestureEvent::BurstHeld { since: t0 });
            if let Some(arm) = self.armed.take() {
                out.push(GestureEvent::Disarmed {
                    zone: arm.zone,
                    reason: DisarmReason::BurstHold,
                });
            }
            out.push(GestureEvent::Suppressed {
                candidate: id,
                reason: SuppressReason::Burst,
            });
            return;
        }

        // 5. Refractory; a suppressed candidate does not move the anchor.
        if let Some(last) = self.last_unsuppressed {
            if t0.since(last) < HostDuration::from_ms(self.params.refractory) {
                out.push(GestureEvent::Suppressed {
                    candidate: id,
                    reason: SuppressReason::Refractory,
                });
                return;
            }
        }

        // 6.
        self.last_unsuppressed = Some(t0);

        // 7.
        match outcome {
            CandidateOutcome::Vetoed | CandidateOutcome::Rejected => {
                if let Some(arm) = self.armed.take() {
                    out.push(GestureEvent::Disarmed {
                        zone: arm.zone,
                        reason: DisarmReason::Interrupted,
                    });
                }
            }
            CandidateOutcome::Accepted { zone } => match self.trigger_of(zone) {
                Trigger::Single => {
                    if let Some(arm) = self.armed.take() {
                        out.push(GestureEvent::Disarmed {
                            zone: arm.zone,
                            reason: DisarmReason::OtherZone,
                        });
                    }
                    out.push(GestureEvent::Fire {
                        zone,
                        candidate: id,
                        t0,
                        first: None,
                    });
                }
                Trigger::Double => self.on_double(zone, id, t0, out),
            },
        }
    }

    /// Expires a pending double tap and releases a burst hold whose time has come.
    pub fn advance_to(&mut self, now: HostTime, out: &mut Vec<GestureEvent>) {
        if let Some(arm) = self.armed {
            if now > arm.expires_at {
                out.push(GestureEvent::DoubleExpired { zone: arm.zone });
                self.armed = None;
            }
        }
        if let Some(until) = self.hold_until {
            if now >= until {
                out.push(GestureEvent::BurstReleased { at: now });
                self.hold_until = None;
            }
        }
    }

    /// Step 7 for an accepted tap of a double-trigger zone: complete the armed double, or arm.
    fn on_double(
        &mut self,
        zone: ZoneIndex,
        id: CandidateId,
        t0: HostTime,
        out: &mut Vec<GestureEvent>,
    ) {
        let min = HostDuration::from_ms(self.params.double_min);
        let max = HostDuration::from_ms(self.params.double_max);
        if let Some(arm) = self.armed.take() {
            let spacing = t0.since(arm.t0);
            if arm.zone == zone && min <= spacing && spacing <= max {
                out.push(GestureEvent::Fire {
                    zone,
                    candidate: id,
                    t0,
                    first: Some(arm.candidate),
                });
                return;
            }
            let reason = if arm.zone == zone {
                DisarmReason::Interrupted
            } else {
                DisarmReason::OtherZone
            };
            out.push(GestureEvent::Disarmed {
                zone: arm.zone,
                reason,
            });
        }
        let expires_at = after(t0, max);
        self.armed = Some(Arm {
            zone,
            candidate: id,
            t0,
            expires_at,
        });
        out.push(GestureEvent::Armed {
            zone,
            candidate: id,
            expires_at,
        });
    }

    fn trigger_of(&self, zone: ZoneIndex) -> Trigger {
        match self.triggers.get(usize::from(zone.0)) {
            Some(&trigger) => trigger,
            // SPEC: §1.12 gives no outcome for an accepted zone without a trigger; it can only be a
            // caller bug (triggers and model come from one profile), so it fails fast.
            None => panic!(
                "gesture: accepted zone {} has no trigger ({} configured)",
                zone.0,
                self.triggers.len()
            ),
        }
    }
}

/// `t + d`, saturating at the i64 tick range, which real host times never approach.
fn after(t: HostTime, d: HostDuration) -> HostTime {
    HostTime(t.0.saturating_add(d.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// 100-ns ticks per millisecond.
    const TICKS_PER_MS: f64 = 10_000.0;
    /// Every test stream starts here, well away from zero.
    const BASE_TICKS: i64 = 1_000 * 10_000_000;

    fn at(ms: f64) -> HostTime {
        HostTime(BASE_TICKS + (ms * TICKS_PER_MS).round() as i64)
    }

    fn z(i: u16) -> ZoneIndex {
        ZoneIndex(i)
    }

    fn c(i: u64) -> CandidateId {
        CandidateId(i)
    }

    fn accepted(zone: u16) -> CandidateOutcome {
        CandidateOutcome::Accepted { zone: z(zone) }
    }

    fn fire(zone: u16, id: u64, t0_ms: f64, first: Option<u64>) -> GestureEvent {
        GestureEvent::Fire {
            zone: z(zone),
            candidate: c(id),
            t0: at(t0_ms),
            first: first.map(c),
        }
    }

    fn armed(zone: u16, id: u64, t0_ms: f64) -> GestureEvent {
        GestureEvent::Armed {
            zone: z(zone),
            candidate: c(id),
            expires_at: at(t0_ms + GestureParams::DEFAULT.double_max.get()),
        }
    }

    fn suppressed(id: u64, reason: SuppressReason) -> GestureEvent {
        GestureEvent::Suppressed {
            candidate: c(id),
            reason,
        }
    }

    fn disarmed(zone: u16, reason: DisarmReason) -> GestureEvent {
        GestureEvent::Disarmed {
            zone: z(zone),
            reason,
        }
    }

    /// Feeds one candidate and returns exactly the events it produced.
    fn feed(g: &mut Gesture, id: u64, t0_ms: f64, outcome: CandidateOutcome) -> Vec<GestureEvent> {
        let mut out = Vec::new();
        g.on_candidate(c(id), at(t0_ms), outcome, &mut out);
        out
    }

    fn advance(g: &mut Gesture, now_ms: f64) -> Vec<GestureEvent> {
        let mut out = Vec::new();
        g.advance_to(at(now_ms), &mut out);
        out
    }

    #[test]
    fn g1_single_fires_then_refractory() {
        // Single fires on Accepted; a second candidate 100 ms later is Refractory whatever its outcome.
        for second in [
            accepted(0),
            CandidateOutcome::Vetoed,
            CandidateOutcome::Rejected,
        ] {
            let mut g = Gesture::new(vec![Trigger::Single], GestureParams::DEFAULT);
            assert_eq!(
                feed(&mut g, 1, 0.0, accepted(0)),
                vec![fire(0, 1, 0.0, None)]
            );
            assert_eq!(
                feed(&mut g, 2, 100.0, second),
                vec![suppressed(2, SuppressReason::Refractory)]
            );
        }

        // SPEC: G1's "one at 130 ms" is its own stream. After 0 ms and 100 ms, a third candidate at
        // 130 ms is three within burst_window and §1.12 step 4 holds it as a burst.
        let mut g = Gesture::new(vec![Trigger::Single], GestureParams::DEFAULT);
        assert_eq!(
            feed(&mut g, 1, 0.0, accepted(0)),
            vec![fire(0, 1, 0.0, None)]
        );
        assert_eq!(
            feed(&mut g, 2, 130.0, accepted(0)),
            vec![fire(0, 2, 130.0, None)]
        );

        // The refractory bound is exclusive: exactly 120 ms later is processed.
        let mut g = Gesture::new(vec![Trigger::Single], GestureParams::DEFAULT);
        assert_eq!(
            feed(&mut g, 1, 0.0, accepted(0)),
            vec![fire(0, 1, 0.0, None)]
        );
        assert_eq!(
            feed(&mut g, 2, 120.0, accepted(0)),
            vec![fire(0, 2, 120.0, None)]
        );

        // A suppressed candidate does not move the anchor (burst disabled to isolate the rule):
        // 130 ms is 130 ms after the anchor at 0, not 30 ms after the suppressed one at 100.
        let no_burst = GestureParams {
            burst_count: 16,
            ..GestureParams::DEFAULT
        };
        let mut g = Gesture::new(vec![Trigger::Single], no_burst);
        assert_eq!(
            feed(&mut g, 1, 0.0, accepted(0)),
            vec![fire(0, 1, 0.0, None)]
        );
        assert_eq!(
            feed(&mut g, 2, 100.0, accepted(0)),
            vec![suppressed(2, SuppressReason::Refractory)]
        );
        assert_eq!(
            feed(&mut g, 3, 130.0, accepted(0)),
            vec![fire(0, 3, 130.0, None)]
        );
    }

    #[test]
    fn g2_burst_hold_extends_and_releases() {
        let mut g = Gesture::new(vec![Trigger::Single], GestureParams::DEFAULT);
        assert_eq!(
            feed(&mut g, 1, 0.0, accepted(0)),
            vec![fire(0, 1, 0.0, None)]
        );
        assert_eq!(
            feed(&mut g, 2, 200.0, accepted(0)),
            vec![fire(0, 2, 200.0, None)]
        );
        // The third within 600 ms starts the hold.
        assert_eq!(
            feed(&mut g, 3, 400.0, accepted(0)),
            vec![
                GestureEvent::BurstHeld { since: at(400.0) },
                suppressed(3, SuppressReason::Burst),
            ]
        );
        // Candidates during the hold extend it: 600 → until 900, then 850 → until 1150.
        assert_eq!(
            feed(&mut g, 4, 600.0, accepted(0)),
            vec![suppressed(4, SuppressReason::Burst)]
        );
        assert_eq!(advance(&mut g, 849.9), vec![]);
        assert_eq!(
            feed(&mut g, 5, 850.0, CandidateOutcome::Rejected),
            vec![suppressed(5, SuppressReason::Burst)]
        );
        // Released exactly 300 ms after the last candidate, not before.
        assert_eq!(advance(&mut g, 1149.9), vec![]);
        assert_eq!(
            advance(&mut g, 1150.0),
            vec![GestureEvent::BurstReleased { at: at(1150.0) }]
        );
        assert_eq!(advance(&mut g, 1150.0), vec![]);
        // After the release a candidate is processed again (only 850 is still within 600 ms).
        assert_eq!(
            feed(&mut g, 6, 1300.0, accepted(0)),
            vec![fire(0, 6, 1300.0, None)]
        );

        // A burst disarms a pending double tap.
        let mut g = Gesture::new(vec![Trigger::Double], GestureParams::DEFAULT);
        assert_eq!(feed(&mut g, 1, 0.0, accepted(0)), vec![armed(0, 1, 0.0)]);
        assert_eq!(
            feed(&mut g, 2, 100.0, CandidateOutcome::Rejected),
            vec![suppressed(2, SuppressReason::Refractory)]
        );
        assert_eq!(
            feed(&mut g, 3, 130.0, accepted(0)),
            vec![
                GestureEvent::BurstHeld { since: at(130.0) },
                disarmed(0, DisarmReason::BurstHold),
                suppressed(3, SuppressReason::Burst),
            ]
        );
    }

    #[test]
    fn g3_double_trigger() {
        let double = || {
            Gesture::new(
                vec![Trigger::Double, Trigger::Double],
                GestureParams::DEFAULT,
            )
        };

        // Same zone 200 ms apart: one Fire with the second t0 and the first candidate.
        let mut g = double();
        assert_eq!(feed(&mut g, 1, 0.0, accepted(0)), vec![armed(0, 1, 0.0)]);
        assert_eq!(
            feed(&mut g, 2, 200.0, accepted(0)),
            vec![fire(0, 2, 200.0, Some(1))]
        );
        assert_eq!(advance(&mut g, 2000.0), vec![]);

        // The double window is inclusive: exactly 450 ms still fires.
        let mut g = double();
        assert_eq!(feed(&mut g, 1, 0.0, accepted(0)), vec![armed(0, 1, 0.0)]);
        assert_eq!(
            feed(&mut g, 2, 450.0, accepted(0)),
            vec![fire(0, 2, 450.0, Some(1))]
        );

        // 500 ms apart: Armed, DoubleExpired, Armed.
        let mut g = double();
        assert_eq!(feed(&mut g, 1, 0.0, accepted(0)), vec![armed(0, 1, 0.0)]);
        assert_eq!(
            feed(&mut g, 2, 500.0, accepted(0)),
            vec![
                GestureEvent::DoubleExpired { zone: z(0) },
                armed(0, 2, 500.0)
            ]
        );
        // Expiry is strictly after expires_at.
        let mut g = double();
        assert_eq!(feed(&mut g, 1, 0.0, accepted(0)), vec![armed(0, 1, 0.0)]);
        assert_eq!(advance(&mut g, 450.0), vec![]);
        assert_eq!(
            advance(&mut g, 450.1),
            vec![GestureEvent::DoubleExpired { zone: z(0) }]
        );

        // Other zone: Disarmed{OtherZone}, then the other zone arms.
        let mut g = double();
        assert_eq!(feed(&mut g, 1, 0.0, accepted(0)), vec![armed(0, 1, 0.0)]);
        assert_eq!(
            feed(&mut g, 2, 200.0, accepted(1)),
            vec![disarmed(0, DisarmReason::OtherZone), armed(1, 2, 200.0)]
        );

        // An intervening Rejected (or Vetoed) candidate: Disarmed{Interrupted}.
        for between in [CandidateOutcome::Rejected, CandidateOutcome::Vetoed] {
            let mut g = double();
            assert_eq!(feed(&mut g, 1, 0.0, accepted(0)), vec![armed(0, 1, 0.0)]);
            assert_eq!(
                feed(&mut g, 2, 200.0, between),
                vec![disarmed(0, DisarmReason::Interrupted)]
            );
            // The next accepted tap starts a new double instead of completing the old one.
            assert_eq!(
                feed(&mut g, 3, 700.0, accepted(0)),
                vec![armed(0, 3, 700.0)]
            );
        }

        // A single-trigger zone accepted while a double is armed: Disarmed{OtherZone}, then Fire.
        let mut g = Gesture::new(
            vec![Trigger::Double, Trigger::Single],
            GestureParams::DEFAULT,
        );
        assert_eq!(feed(&mut g, 1, 0.0, accepted(0)), vec![armed(0, 1, 0.0)]);
        assert_eq!(
            feed(&mut g, 2, 200.0, accepted(1)),
            vec![
                disarmed(0, DisarmReason::OtherZone),
                fire(1, 2, 200.0, None)
            ]
        );

        // Same zone faster than double_min (refractory lowered so the tap is processed):
        // Disarmed{Interrupted}, then it re-arms.
        let fast = GestureParams {
            refractory: Milliseconds::new(50.0),
            ..GestureParams::DEFAULT
        };
        let mut g = Gesture::new(vec![Trigger::Double], fast);
        assert_eq!(feed(&mut g, 1, 0.0, accepted(0)), vec![armed(0, 1, 0.0)]);
        assert_eq!(
            feed(&mut g, 2, 100.0, accepted(0)),
            vec![disarmed(0, DisarmReason::Interrupted), armed(0, 2, 100.0)]
        );
    }

    #[test]
    fn g4_vetoed_candidates_count_toward_burst() {
        let mut g = Gesture::new(vec![Trigger::Single], GestureParams::DEFAULT);
        assert_eq!(feed(&mut g, 1, 0.0, CandidateOutcome::Vetoed), vec![]);
        assert_eq!(feed(&mut g, 2, 200.0, CandidateOutcome::Vetoed), vec![]);
        assert_eq!(
            feed(&mut g, 3, 400.0, accepted(0)),
            vec![
                GestureEvent::BurstHeld { since: at(400.0) },
                suppressed(3, SuppressReason::Burst),
            ]
        );
    }

    /// One step of a random stream: a candidate or an `advance_to`, `gap_ms` after the previous step.
    #[derive(Debug, Clone)]
    enum Step {
        Candidate { gap_ms: u32, outcome: u8, zone: u16 },
        Advance { gap_ms: u32 },
    }

    fn step() -> impl Strategy<Value = Step> {
        prop_oneof![
            3 => (0u32..700, 0u8..4, 0u16..3)
                .prop_map(|(gap_ms, outcome, zone)| Step::Candidate { gap_ms, outcome, zone }),
            1 => (0u32..700).prop_map(|gap_ms| Step::Advance { gap_ms }),
        ]
    }

    /// The time an event is about, for the ordering check (`None` for events without one).
    fn event_time(e: &GestureEvent) -> Option<HostTime> {
        match e {
            GestureEvent::Fire { t0, .. } => Some(*t0),
            GestureEvent::BurstHeld { since } => Some(*since),
            GestureEvent::BurstReleased { at } => Some(*at),
            GestureEvent::Armed { .. }
            | GestureEvent::Disarmed { .. }
            | GestureEvent::DoubleExpired { .. }
            | GestureEvent::Suppressed { .. } => None,
        }
    }

    proptest! {
        #[test]
        fn g5_random_streams_keep_invariants(steps in prop::collection::vec(step(), 1..300)) {
            let params = GestureParams::DEFAULT;
            let refractory = HostDuration::from_ms(params.refractory);
            let mut g = Gesture::new(
                vec![Trigger::Single, Trigger::Double, Trigger::Double],
                params,
            );
            let initial_capacity = g.recent.capacity();
            let mut now = at(0.0);
            let mut next_id = 0u64;
            let mut held = false;
            let mut last_time: Option<HostTime> = None;
            let mut last_processed: Option<HostTime> = None;

            for s in steps {
                let mut out = Vec::new();
                match s {
                    Step::Candidate { gap_ms, outcome, zone } => {
                        now = HostTime(now.0 + i64::from(gap_ms) * 10_000);
                        next_id += 1;
                        let outcome = match outcome {
                            0 => CandidateOutcome::Vetoed,
                            1 => CandidateOutcome::Rejected,
                            _ => accepted(zone),
                        };
                        g.on_candidate(c(next_id), now, outcome, &mut out);
                        let suppressed_now = out
                            .iter()
                            .any(|e| matches!(e, GestureEvent::Suppressed { .. }));
                        // Never a Fire within refractory of the last processed candidate.
                        if out.iter().any(|e| matches!(e, GestureEvent::Fire { .. })) {
                            prop_assert!(!suppressed_now);
                            if let Some(last) = last_processed {
                                prop_assert!(now.since(last) >= refractory);
                            }
                        }
                        if !suppressed_now {
                            last_processed = Some(now);
                        }
                    }
                    Step::Advance { gap_ms } => {
                        now = HostTime(now.0 + i64::from(gap_ms) * 10_000);
                        g.advance_to(now, &mut out);
                        // advance_to is idempotent: a second call at the same time does nothing.
                        let before = g.clone();
                        let mut again = Vec::new();
                        g.advance_to(now, &mut again);
                        prop_assert!(again.is_empty());
                        prop_assert_eq!(&g, &before);
                    }
                }
                for e in &out {
                    match e {
                        GestureEvent::BurstHeld { .. } => held = true,
                        GestureEvent::BurstReleased { .. } => held = false,
                        // Never a Fire during a hold.
                        GestureEvent::Fire { .. } => prop_assert!(!held),
                        GestureEvent::Armed { .. }
                        | GestureEvent::Disarmed { .. }
                        | GestureEvent::DoubleExpired { .. }
                        | GestureEvent::Suppressed { .. } => {}
                    }
                    // Events are time-ordered and never ahead of the call that produced them.
                    if let Some(t) = event_time(e) {
                        prop_assert!(t <= now);
                        if let Some(prev) = last_time {
                            prop_assert!(t >= prev);
                        }
                        last_time = Some(t);
                    }
                }
                prop_assert_eq!(held, g.hold_until.is_some());
                // Memory is bounded: the ring never exceeds its cap or reallocates.
                prop_assert!(g.recent.len() <= RECENT_CAP);
                prop_assert_eq!(g.recent.capacity(), initial_capacity);
            }
        }
    }

    #[test]
    fn g6_default_double_window_within_half_a_second() {
        // The UI copy says "within half a second" (ui-v2 §10.5, XR-UI-13).
        assert!(GestureParams::DEFAULT.double_max.get() <= 500.0);
    }
}
