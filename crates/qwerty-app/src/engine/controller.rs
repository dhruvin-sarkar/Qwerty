//! The controller thread `qwerty-ctl` (p2 §2.8): the one owner of all mutable app state.
//!
//! This module also holds the two pure display helpers of the health aggregation, `close_call`
//! and `DriftWindow` (XR-UI-20). No decision reads either of them, so the pooled novelty threshold
//! (§1.9.3) is untouched.

use std::collections::VecDeque;

use qwerty_core::model::{ModelRejectReason, ModelScores};
use qwerty_core::pipeline::DecisionOutcome;
use qwerty_core::units::ZoneIndex;

use crate::engine::protocol::{CLOSE_CALL_RATIO, DRIFT_CLEAR, DRIFT_RAISE, DRIFT_WINDOW};

// ---------------------------------------------------------------------------
// Close calls and recognition drift (§2.8, XR-UI-20)
// ---------------------------------------------------------------------------

/// `Some(scores.second)` when an accepted decision's margin is below `CLOSE_CALL_RATIO × τ_m`
/// (the model's `Thresholds::tau_margin`), else `None`. Display only.
pub fn close_call(scores: &ModelScores, tau_margin: f64) -> Option<ZoneIndex> {
    if scores.margin < CLOSE_CALL_RATIO * tau_margin {
        Some(scores.second)
    } else {
        None
    }
}

/// A change of the `RecognitionDrift` warning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftChange {
    Raised {
        window: u32,
        ambiguous: u32,
        close_calls: u32,
    },
    Cleared,
}

/// One counted outcome in the drift window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DriftSample {
    Ambiguous,
    CloseCall,
    Clean,
}

/// The last `DRIFT_WINDOW` decisions the model placed among the zones (`Accepted`, or
/// `Rejected{Ambiguous}`), with the raise/clear hysteresis of §2.8. Display only.
#[derive(Debug, Clone)]
pub struct DriftWindow {
    samples: VecDeque<DriftSample>,
    raised: bool,
}

impl Default for DriftWindow {
    fn default() -> Self {
        DriftWindow::new()
    }
}

impl DriftWindow {
    pub fn new() -> DriftWindow {
        DriftWindow {
            samples: VecDeque::with_capacity(DRIFT_WINDOW as usize),
            raised: false,
        }
    }

    /// Forgets every outcome and the raised state (a profile switch, a committed recalibration or
    /// a capture reopen). The caller clears the warning itself.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.raised = false;
    }

    /// Counts `outcome` (an excluded outcome is ignored) and reports a change of the warning:
    /// raised when ambiguous + close calls ≥ `DRIFT_RAISE`, cleared at ≤ `DRIFT_CLEAR`.
    pub fn push(&mut self, outcome: &DecisionOutcome, tau_margin: f64) -> Option<DriftChange> {
        let sample = match outcome {
            DecisionOutcome::Accepted { scores, .. } => match close_call(scores, tau_margin) {
                Some(_) => DriftSample::CloseCall,
                None => DriftSample::Clean,
            },
            DecisionOutcome::Rejected { reason, .. } => match reason {
                ModelRejectReason::Ambiguous => DriftSample::Ambiguous,
                // Ordinary non-tap sounds land here (§2.8).
                ModelRejectReason::OutOfDistribution | ModelRejectReason::Background => {
                    return None
                }
            },
            DecisionOutcome::Vetoed { .. } | DecisionOutcome::Invalid { .. } => return None,
        };
        if self.samples.len() == DRIFT_WINDOW as usize {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
        let ambiguous = self.count(DriftSample::Ambiguous);
        let close_calls = self.count(DriftSample::CloseCall);
        let flagged = ambiguous + close_calls;
        if !self.raised && flagged >= DRIFT_RAISE {
            self.raised = true;
            Some(DriftChange::Raised {
                window: self.samples.len() as u32,
                ambiguous,
                close_calls,
            })
        } else if self.raised && flagged <= DRIFT_CLEAR {
            self.raised = false;
            Some(DriftChange::Cleared)
        } else {
            None
        }
    }

    fn count(&self, kind: DriftSample) -> u32 {
        self.samples.iter().filter(|s| **s == kind).count() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwerty_core::descriptor::{Descriptor, DescriptorLayout, DESCRIPTOR_DIMS};
    use qwerty_core::units::Milliseconds;
    use qwerty_core::veto::{InputKind, VetoReason};

    const TAU_MARGIN: f64 = 4.0;

    fn scores(margin: f64) -> ModelScores {
        ModelScores {
            d2: vec![1.0, 1.0 + margin],
            best: ZoneIndex(0),
            second: ZoneIndex(1),
            d2_min: 1.0,
            margin,
            llr_bg: 10.0,
            posterior_best: 0.9,
        }
    }

    fn descriptor() -> Descriptor {
        Descriptor {
            layout: DescriptorLayout::Mono { a: 0 },
            values: [0.0; DESCRIPTOR_DIMS],
        }
    }

    fn accepted(margin: f64) -> DecisionOutcome {
        DecisionOutcome::Accepted {
            zone: ZoneIndex(0),
            descriptor: descriptor(),
            scores: scores(margin),
        }
    }

    fn rejected(reason: ModelRejectReason) -> DecisionOutcome {
        DecisionOutcome::Rejected {
            reason,
            descriptor: descriptor(),
            scores: scores(0.5),
        }
    }

    #[test]
    fn en16_close_call_and_drift_window() {
        // (a) close_call: just below 1.5·τ_m is a close call; equal to it is not.
        let limit = CLOSE_CALL_RATIO * TAU_MARGIN;
        let below = limit - 1e-9;
        assert_eq!(close_call(&scores(below), TAU_MARGIN), Some(ZoneIndex(1)));
        assert_eq!(close_call(&scores(limit), TAU_MARGIN), None);

        // (b) DriftWindow: excluded outcomes are ignored.
        let mut drift = DriftWindow::new();
        let excluded = [
            rejected(ModelRejectReason::OutOfDistribution),
            rejected(ModelRejectReason::Background),
            DecisionOutcome::Vetoed {
                reason: VetoReason::Input {
                    kind: InputKind::KeyDown,
                    offset: Milliseconds::new(5.0),
                },
                descriptor: descriptor(),
                scores: scores(0.5),
            },
            DecisionOutcome::Invalid {
                error: qwerty_core::descriptor::DescriptorError::SpanTooShort { needed: 2, got: 1 },
            },
        ];
        for _ in 0..30 {
            for outcome in &excluded {
                assert_eq!(drift.push(outcome, TAU_MARGIN), None);
            }
        }

        // 12 clean accepts, then 4 ambiguous and 3 close calls: 7 flagged, not raised.
        for _ in 0..12 {
            assert_eq!(drift.push(&accepted(limit * 2.0), TAU_MARGIN), None);
        }
        for _ in 0..4 {
            assert_eq!(
                drift.push(&rejected(ModelRejectReason::Ambiguous), TAU_MARGIN),
                None
            );
        }
        for _ in 0..3 {
            assert_eq!(drift.push(&accepted(below), TAU_MARGIN), None);
        }
        // The 8th flagged outcome in the last 20 raises.
        assert_eq!(
            drift.push(&accepted(below), TAU_MARGIN),
            Some(DriftChange::Raised {
                window: 20,
                ambiguous: 4,
                close_calls: 4
            })
        );
        // Still raised while flagged > 3.
        for _ in 0..16 {
            assert_eq!(drift.push(&accepted(limit * 2.0), TAU_MARGIN), None);
        }
        // 20 clean outcomes would leave 0 flagged; the clear fires as the count reaches 3.
        // After 16 clean pushes the window holds 4 flagged (the last 4 close calls).
        assert_eq!(
            drift.push(&accepted(limit * 2.0), TAU_MARGIN),
            Some(DriftChange::Cleared)
        );

        // reset() forgets the window and the raised state.
        for _ in 0..8 {
            drift.push(&rejected(ModelRejectReason::Ambiguous), TAU_MARGIN);
        }
        drift.reset();
        for _ in 0..7 {
            assert_eq!(
                drift.push(&rejected(ModelRejectReason::Ambiguous), TAU_MARGIN),
                None
            );
        }
        assert_eq!(
            drift.push(&rejected(ModelRejectReason::Ambiguous), TAU_MARGIN),
            Some(DriftChange::Raised {
                window: 8,
                ambiguous: 8,
                close_calls: 0
            })
        );
    }
}
