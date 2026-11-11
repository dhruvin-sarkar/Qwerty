//! The mode state machine (p2 §2.10): a pure resolver from the controller's state to the `Mode`.
//! The controller performs every effect; `resolve` only names the resulting mode.

use qwerty_core::assessment::EvaluationView;
use qwerty_core::audio::DeviceIdentity;
use qwerty_core::pipeline::PipelineError;
use qwerty_core::profile::ProfileId;
use qwerty_core::store::{DeviceFingerprint, RecalibrationReason};

use crate::engine::protocol::{
    Blocker, CalibrationRun, Desired, MicUnavailableReason, Mode, PersistenceFailure, ThreadName,
    VetoStatus,
};

pub struct ModeInputs<'a> {
    pub fault: Option<(ThreadName, &'a str)>,
    /// until the first reconcile has attempted every open
    pub starting: bool,
    pub activity: Activity<'a>,
    pub profile: ProfileReadiness<'a>,
    pub desired: Desired,
    /// for the bound endpoint
    pub device: DeviceReadiness<'a>,
    /// the active profile's Pipeline
    pub pipeline: PipelineReadiness<'a>,
    pub veto: &'a VetoStatus,
    pub veto_enabled: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum Activity<'a> {
    None,
    Calibrating(&'a CalibrationRun),
    Evaluating(&'a EvaluationView),
}

#[derive(Debug, Clone, Copy)]
pub enum ProfileReadiness<'a> {
    None,
    Unreadable {
        file: &'a str,
        error: &'a PersistenceFailure,
    },
    Calibrated {
        id: ProfileId,
        bound: &'a DeviceFingerprint,
    },
    NeedsRecalibration {
        id: ProfileId,
        reason: &'a RecalibrationReason,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum DeviceReadiness<'a> {
    NotAttempted,
    Opened {
        identity: &'a DeviceIdentity,
    },
    Unavailable {
        endpoint_id: &'a str,
        name: &'a str,
        reason: &'a MicUnavailableReason,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum PipelineReadiness<'a> {
    NotBuilt,
    Ready,
    Failed { error: &'a PipelineError },
}

/// Precedence, first match wins (§2.10):
/// 1. fault → Faulted; 2. starting → Starting; 3. activity → Calibrating / Evaluating;
/// 4. no active profile → Idle; 5. active profile unreadable → Blocked(ProfileUnreadable);
/// 6. desired == Paused → Paused; 7. NeedsRecalibration → Blocked(NeedsRecalibration);
/// 8. device Unavailable → Blocked(MicUnavailable), NotAttempted → Starting;
/// 9. Opened with differences ≠ ∅ → Blocked(MicMismatch);
/// 10. pipeline Failed → Blocked(ProfileIncompatible);
/// 11. veto_enabled and veto ∈ {WaitingForShell, Registering} → Starting;
/// 12. veto_enabled and veto Unavailable → Blocked(InputVetoUnavailable); 13. → Listening.
pub fn resolve(inputs: &ModeInputs<'_>) -> Mode {
    if let Some((thread, message)) = inputs.fault {
        return Mode::Faulted {
            thread,
            message: message.to_string(),
        };
    }
    if inputs.starting {
        return Mode::Starting;
    }
    match inputs.activity {
        Activity::None => {}
        Activity::Calibrating(run) => {
            return Mode::Calibrating {
                run: Box::new(run.clone()),
            }
        }
        Activity::Evaluating(view) => return Mode::Evaluating { view: view.clone() },
    }
    let (profile, bound) = match inputs.profile {
        ProfileReadiness::None => return Mode::Idle,
        ProfileReadiness::Unreadable { file, error } => {
            return blocked(Blocker::ProfileUnreadable {
                file: file.to_string(),
                error: error.clone(),
            })
        }
        // Rule 6 (Paused) outranks rule 7 (NeedsRecalibration).
        ProfileReadiness::NeedsRecalibration { .. } | ProfileReadiness::Calibrated { .. }
            if inputs.desired == Desired::Paused =>
        {
            return Mode::Paused
        }
        ProfileReadiness::NeedsRecalibration { id, reason } => {
            return blocked(Blocker::NeedsRecalibration {
                profile: id,
                reason: reason.clone(),
            })
        }
        ProfileReadiness::Calibrated { id, bound } => (id, bound),
    };
    match inputs.device {
        DeviceReadiness::Unavailable {
            endpoint_id,
            name,
            reason,
        } => {
            return blocked(Blocker::MicUnavailable {
                endpoint_id: endpoint_id.to_string(),
                name: name.to_string(),
                reason: reason.clone(),
            })
        }
        DeviceReadiness::NotAttempted => return Mode::Starting,
        DeviceReadiness::Opened { identity } => {
            let fields = bound.identity.differences(identity);
            if !fields.is_empty() {
                return blocked(Blocker::MicMismatch {
                    bound: bound.identity.clone(),
                    current: identity.clone(),
                    fields,
                });
            }
        }
    }
    match inputs.pipeline {
        PipelineReadiness::Failed { error } => {
            return blocked(Blocker::ProfileIncompatible {
                profile,
                error: error.clone(),
            })
        }
        PipelineReadiness::NotBuilt | PipelineReadiness::Ready => {}
    }
    if inputs.veto_enabled {
        match inputs.veto {
            VetoStatus::WaitingForShell | VetoStatus::Registering => return Mode::Starting,
            VetoStatus::Unavailable { reason } => {
                return blocked(Blocker::InputVetoUnavailable {
                    reason: reason.clone(),
                })
            }
            VetoStatus::Disabled | VetoStatus::NotNeeded | VetoStatus::Active => {}
        }
    }
    Mode::Listening
}

fn blocked(blocker: Blocker) -> Mode {
    Mode::Blocked { blocker }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::protocol::{
        CalibrationTarget, CaptureError, InputDeviceClass, ModeKind, PersistenceFailureKind,
        RunSession, VetoUnavailableReason, ZoneDraft,
    };
    use qwerty_core::assessment::EvaluationStep;
    use qwerty_core::audio::{AudioFormat, CaptureMode, ChannelLayout, IdentityField};
    use qwerty_core::pipeline::Incompatibility;
    use qwerty_core::profile::ZoneLayout;
    use qwerty_core::units::{ChannelCount, SampleRate};

    /// A fixed ProfileId (the app has no uuid dependency; serde reads the uuid string).
    fn profile_id(n: u8) -> ProfileId {
        let text = format!("\"00000000-0000-0000-0000-0000000000{n:02x}\"");
        serde_json::from_str(&text).expect("a valid uuid string")
    }

    fn identity(endpoint: &str, hz: u32) -> DeviceIdentity {
        DeviceIdentity {
            endpoint_id: endpoint.to_string(),
            name: "Desk mic".to_string(),
            format: AudioFormat {
                rate: SampleRate::new(hz).expect("valid rate"),
                channels: ChannelCount::new(2).expect("valid channels"),
            },
            capture_mode: CaptureMode::Raw,
        }
    }

    fn fingerprint() -> DeviceFingerprint {
        DeviceFingerprint {
            identity: identity("mic-1", 48_000),
            layout: ChannelLayout::all_real(ChannelCount::new(2).expect("valid channels")),
        }
    }

    fn run() -> CalibrationRun {
        CalibrationRun {
            target: CalibrationTarget::New {
                name: "Desk".to_string(),
            },
            zones: vec![ZoneDraft {
                label: "Left".to_string(),
                layout: ZoneLayout {
                    x: 0.0,
                    y: 0.0,
                    width: 0.5,
                    height: 1.0,
                },
            }],
            device: identity("mic-1", 48_000),
            taps_per_zone: 10,
            session: RunSession::MicLost {
                reason: MicUnavailableReason::Unplugged,
            },
        }
    }

    fn view() -> EvaluationView {
        EvaluationView {
            step: EvaluationStep::Finished,
            resolved: 0,
            total_prompts: 0,
            correct: 0,
            last: None,
        }
    }

    fn failure() -> PersistenceFailure {
        PersistenceFailure {
            file: "Profiles/x.json".to_string(),
            kind: PersistenceFailureKind::Json,
            detail: "expected value".to_string(),
        }
    }

    fn pipeline_error() -> PipelineError {
        PipelineError::ModelIncompatible {
            reason: Incompatibility::FrontEndVersion {
                model: 1,
                current: 2,
            },
        }
    }

    fn veto_reason() -> VetoUnavailableReason {
        VetoUnavailableReason::OwnershipLost {
            class: InputDeviceClass::Keyboard,
        }
    }

    /// The fixtures every input borrows from.
    struct World {
        run: CalibrationRun,
        view: EvaluationView,
        failure: PersistenceFailure,
        bound: DeviceFingerprint,
        reason: RecalibrationReason,
        matching: DeviceIdentity,
        mismatched: DeviceIdentity,
        mic_reason: MicUnavailableReason,
        pipeline_error: PipelineError,
        vetoes: Vec<VetoStatus>,
    }

    fn world() -> World {
        World {
            run: run(),
            view: view(),
            failure: failure(),
            bound: fingerprint(),
            reason: RecalibrationReason::ZoneSetChanged,
            matching: identity("mic-1", 48_000),
            mismatched: identity("mic-1", 44_100),
            mic_reason: MicUnavailableReason::OpenFailed {
                error: CaptureError::AccessDenied,
            },
            pipeline_error: pipeline_error(),
            vetoes: vec![
                VetoStatus::Disabled,
                VetoStatus::NotNeeded,
                VetoStatus::WaitingForShell,
                VetoStatus::Registering,
                VetoStatus::Active,
                VetoStatus::Unavailable {
                    reason: veto_reason(),
                },
            ],
        }
    }

    /// A Listening world: calibrated, device open and matching, pipeline ready, veto Active.
    fn listening(w: &World) -> ModeInputs<'_> {
        ModeInputs {
            fault: None,
            starting: false,
            activity: Activity::None,
            profile: ProfileReadiness::Calibrated {
                id: profile_id(1),
                bound: &w.bound,
            },
            desired: Desired::Listening,
            device: DeviceReadiness::Opened {
                identity: &w.matching,
            },
            pipeline: PipelineReadiness::Ready,
            veto: &w.vetoes[4],
            veto_enabled: true,
        }
    }

    fn calibrated(w: &World) -> ProfileReadiness<'_> {
        ProfileReadiness::Calibrated {
            id: profile_id(1),
            bound: &w.bound,
        }
    }

    #[test]
    fn en1_mode_transitions() {
        let w = world();
        let unavailable = DeviceReadiness::Unavailable {
            endpoint_id: "mic-1",
            name: "Desk mic",
            reason: &w.mic_reason,
        };
        let needs = ProfileReadiness::NeedsRecalibration {
            id: profile_id(1),
            reason: &w.reason,
        };
        let unreadable = ProfileReadiness::Unreadable {
            file: "Profiles/x.json",
            error: &w.failure,
        };
        let veto_unavailable = &w.vetoes[5];
        let mic_unavailable = Mode::Blocked {
            blocker: Blocker::MicUnavailable {
                endpoint_id: "mic-1".to_string(),
                name: "Desk mic".to_string(),
                reason: w.mic_reason.clone(),
            },
        };
        let calibrating = Mode::Calibrating {
            run: Box::new(w.run.clone()),
        };

        // ---- Precedence table (§2.10), one row per rule ----
        let rows: Vec<(&str, ModeInputs<'_>, Mode)> = vec![
            (
                "1 fault → Faulted (outranks everything)",
                ModeInputs {
                    fault: Some((ThreadName::Capture, "boom")),
                    starting: true,
                    activity: Activity::Calibrating(&w.run),
                    ..listening(&w)
                },
                Mode::Faulted {
                    thread: ThreadName::Capture,
                    message: "boom".to_string(),
                },
            ),
            (
                "2 starting → Starting (outranks activity)",
                ModeInputs {
                    starting: true,
                    activity: Activity::Evaluating(&w.view),
                    ..listening(&w)
                },
                Mode::Starting,
            ),
            (
                "3 activity Calibrating (outranks a missing profile)",
                ModeInputs {
                    activity: Activity::Calibrating(&w.run),
                    profile: ProfileReadiness::None,
                    ..listening(&w)
                },
                calibrating.clone(),
            ),
            (
                "3 activity Evaluating",
                ModeInputs {
                    activity: Activity::Evaluating(&w.view),
                    device: unavailable,
                    ..listening(&w)
                },
                Mode::Evaluating {
                    view: w.view.clone(),
                },
            ),
            (
                "4 no active profile → Idle (outranks Paused)",
                ModeInputs {
                    profile: ProfileReadiness::None,
                    desired: Desired::Paused,
                    ..listening(&w)
                },
                Mode::Idle,
            ),
            (
                "5 unreadable → Blocked(ProfileUnreadable) (outranks Paused)",
                ModeInputs {
                    profile: unreadable,
                    desired: Desired::Paused,
                    ..listening(&w)
                },
                Mode::Blocked {
                    blocker: Blocker::ProfileUnreadable {
                        file: "Profiles/x.json".to_string(),
                        error: w.failure.clone(),
                    },
                },
            ),
            (
                "6 desired Paused → Paused (outranks NeedsRecalibration and the device)",
                ModeInputs {
                    profile: needs,
                    desired: Desired::Paused,
                    device: unavailable,
                    ..listening(&w)
                },
                Mode::Paused,
            ),
            (
                "7 NeedsRecalibration → Blocked(NeedsRecalibration) (outranks the device)",
                ModeInputs {
                    profile: needs,
                    device: unavailable,
                    ..listening(&w)
                },
                Mode::Blocked {
                    blocker: Blocker::NeedsRecalibration {
                        profile: profile_id(1),
                        reason: w.reason.clone(),
                    },
                },
            ),
            (
                "8 device Unavailable → Blocked(MicUnavailable) (outranks the pipeline)",
                ModeInputs {
                    device: unavailable,
                    pipeline: PipelineReadiness::Failed {
                        error: &w.pipeline_error,
                    },
                    ..listening(&w)
                },
                mic_unavailable.clone(),
            ),
            (
                "8 device NotAttempted → Starting",
                ModeInputs {
                    device: DeviceReadiness::NotAttempted,
                    ..listening(&w)
                },
                Mode::Starting,
            ),
            (
                "9 Opened with differences → Blocked(MicMismatch) (outranks the pipeline)",
                ModeInputs {
                    device: DeviceReadiness::Opened {
                        identity: &w.mismatched,
                    },
                    pipeline: PipelineReadiness::Failed {
                        error: &w.pipeline_error,
                    },
                    ..listening(&w)
                },
                Mode::Blocked {
                    blocker: Blocker::MicMismatch {
                        bound: w.bound.identity.clone(),
                        current: w.mismatched.clone(),
                        fields: vec![IdentityField::SampleRate],
                    },
                },
            ),
            (
                "10 pipeline Failed → Blocked(ProfileIncompatible) (outranks the veto)",
                ModeInputs {
                    pipeline: PipelineReadiness::Failed {
                        error: &w.pipeline_error,
                    },
                    veto: veto_unavailable,
                    ..listening(&w)
                },
                Mode::Blocked {
                    blocker: Blocker::ProfileIncompatible {
                        profile: profile_id(1),
                        error: w.pipeline_error.clone(),
                    },
                },
            ),
            (
                "11 veto WaitingForShell → Starting",
                ModeInputs {
                    veto: &w.vetoes[2],
                    ..listening(&w)
                },
                Mode::Starting,
            ),
            (
                "11 veto Registering → Starting",
                ModeInputs {
                    veto: &w.vetoes[3],
                    ..listening(&w)
                },
                Mode::Starting,
            ),
            (
                "12 veto Unavailable → Blocked(InputVetoUnavailable)",
                ModeInputs {
                    veto: veto_unavailable,
                    ..listening(&w)
                },
                Mode::Blocked {
                    blocker: Blocker::InputVetoUnavailable {
                        reason: veto_reason(),
                    },
                },
            ),
            (
                "11/12 a disabled veto is never waited on nor blocks",
                ModeInputs {
                    veto: veto_unavailable,
                    veto_enabled: false,
                    ..listening(&w)
                },
                Mode::Listening,
            ),
            ("13 → Listening", listening(&w), Mode::Listening),
            // ---- Transition table (§2.10): each row is the state after the controller's effect ----
            (
                "T SetListening{false} / Toggle from Listening → Paused",
                ModeInputs {
                    desired: Desired::Paused,
                    ..listening(&w)
                },
                Mode::Paused,
            ),
            (
                "T SetListening{true} / Toggle from Paused → Listening",
                ModeInputs {
                    desired: Desired::Listening,
                    ..listening(&w)
                },
                Mode::Listening,
            ),
            (
                "T StartCalibration from Blocked → Calibrating",
                ModeInputs {
                    activity: Activity::Calibrating(&w.run),
                    device: unavailable,
                    ..listening(&w)
                },
                calibrating.clone(),
            ),
            (
                "T Commit written: saved and activated, desired := Listening → Listening",
                ModeInputs {
                    profile: calibrated(&w),
                    desired: Desired::Listening,
                    ..listening(&w)
                },
                Mode::Listening,
            ),
            (
                "T Commit written but the world changed → Blocked",
                ModeInputs {
                    device: unavailable,
                    ..listening(&w)
                },
                mic_unavailable.clone(),
            ),
            (
                "T Commit write fails: the run stays → Calibrating",
                ModeInputs {
                    activity: Activity::Calibrating(&w.run),
                    ..listening(&w)
                },
                calibrating.clone(),
            ),
            (
                "T CancelCalibration resumes the prior desired (Paused)",
                ModeInputs {
                    desired: Desired::Paused,
                    ..listening(&w)
                },
                Mode::Paused,
            ),
            (
                "T the capture worker ends while Calibrating: run := MicLost → Calibrating",
                ModeInputs {
                    activity: Activity::Calibrating(&w.run),
                    device: unavailable,
                    ..listening(&w)
                },
                calibrating.clone(),
            ),
            (
                "T StartEvaluation from Listening → Evaluating",
                ModeInputs {
                    activity: Activity::Evaluating(&w.view),
                    ..listening(&w)
                },
                Mode::Evaluating {
                    view: w.view.clone(),
                },
            ),
            (
                "T Evaluation finished / cancelled / aborted resumes the prior desired (Listening)",
                listening(&w),
                Mode::Listening,
            ),
            (
                "T Blocked(MicUnavailable): DeviceMsg / RetryCapture / retry timer → open ok",
                listening(&w),
                Mode::Listening,
            ),
            (
                "T Blocked(MicMismatch): PropertyChanged / RetryCapture → identity matches",
                ModeInputs {
                    device: DeviceReadiness::Opened {
                        identity: &w.matching,
                    },
                    ..listening(&w)
                },
                Mode::Listening,
            ),
            (
                "T Blocked(InputVetoUnavailable): RetryInputVeto → Active",
                ModeInputs {
                    veto: &w.vetoes[4],
                    ..listening(&w)
                },
                Mode::Listening,
            ),
            (
                "T Blocked(InputVetoUnavailable): UpdateConfig{input_veto_enabled: false}",
                ModeInputs {
                    veto: veto_unavailable,
                    veto_enabled: false,
                    ..listening(&w)
                },
                Mode::Listening,
            ),
            (
                "T Blocked(NeedsRecalibration/…): SwitchProfile to a calibrated profile",
                ModeInputs {
                    profile: calibrated(&w),
                    ..listening(&w)
                },
                Mode::Listening,
            ),
            (
                "T Starting: first open attempted and the veto reports Active",
                ModeInputs {
                    starting: false,
                    ..listening(&w)
                },
                Mode::Listening,
            ),
            (
                "T Starting: the veto reports Unavailable",
                ModeInputs {
                    veto: veto_unavailable,
                    ..listening(&w)
                },
                Mode::Blocked {
                    blocker: Blocker::InputVetoUnavailable {
                        reason: veto_reason(),
                    },
                },
            ),
            (
                "T a worker panics → Faulted (terminal)",
                ModeInputs {
                    fault: Some((ThreadName::Dispatch, "worker panic")),
                    ..listening(&w)
                },
                Mode::Faulted {
                    thread: ThreadName::Dispatch,
                    message: "worker panic".to_string(),
                },
            ),
        ];
        for (name, inputs, expected) in &rows {
            assert_eq!(&resolve(inputs), expected, "row: {name}");
        }

        // ---- Totality and determinism: every combination of representative inputs ----
        let faults = [None, Some((ThreadName::Processing, "p"))];
        let activities = [
            Activity::None,
            Activity::Calibrating(&w.run),
            Activity::Evaluating(&w.view),
        ];
        let profiles = [ProfileReadiness::None, unreadable, needs, calibrated(&w)];
        let devices = [
            DeviceReadiness::NotAttempted,
            DeviceReadiness::Opened {
                identity: &w.matching,
            },
            DeviceReadiness::Opened {
                identity: &w.mismatched,
            },
            unavailable,
        ];
        let pipelines = [
            PipelineReadiness::NotBuilt,
            PipelineReadiness::Ready,
            PipelineReadiness::Failed {
                error: &w.pipeline_error,
            },
        ];
        let mut combinations = 0u32;
        for fault in faults {
            for starting in [false, true] {
                for activity in activities {
                    for profile in profiles {
                        for desired in [Desired::Listening, Desired::Paused] {
                            for device in devices {
                                for pipeline in pipelines {
                                    for veto in &w.vetoes {
                                        for veto_enabled in [false, true] {
                                            let inputs = ModeInputs {
                                                fault,
                                                starting,
                                                activity,
                                                profile,
                                                desired,
                                                device,
                                                pipeline,
                                                veto,
                                                veto_enabled,
                                            };
                                            let mode = resolve(&inputs);
                                            assert_eq!(mode, resolve(&inputs), "deterministic");
                                            check_invariants(&inputs, &mode);
                                            combinations += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(combinations, 2 * 2 * 3 * 4 * 2 * 4 * 3 * 6 * 2);
    }

    /// Precedence facts that hold for every input.
    fn check_invariants(inputs: &ModeInputs<'_>, mode: &Mode) {
        let kind = mode.kind();
        if inputs.fault.is_some() {
            assert_eq!(kind, ModeKind::Faulted);
            return;
        }
        assert_ne!(kind, ModeKind::Faulted, "only a fault faults");
        if inputs.starting {
            assert_eq!(kind, ModeKind::Starting);
            return;
        }
        match inputs.activity {
            Activity::Calibrating(_) => assert_eq!(kind, ModeKind::Calibrating),
            Activity::Evaluating(_) => assert_eq!(kind, ModeKind::Evaluating),
            Activity::None => {
                assert!(!matches!(
                    kind,
                    ModeKind::Calibrating | ModeKind::Evaluating
                ));
                if matches!(inputs.profile, ProfileReadiness::None) {
                    assert_eq!(kind, ModeKind::Idle);
                }
                if kind == ModeKind::Listening {
                    assert!(matches!(
                        inputs.profile,
                        ProfileReadiness::Calibrated { .. }
                    ));
                    assert_eq!(inputs.desired, Desired::Listening);
                }
            }
        }
    }
}
