//! Snapshot and event builders for tests and gallery scenes (shell-v2 §4.2 WP-S0): every `Mode`,
//! every `Blocker`, profiles in each status, activity rows, telemetry frames, one value of every
//! `Event` and `Command` variant. Deterministic: fixed ids, times and revisions.

use std::path::PathBuf;

use chrono::{DateTime, Utc};

use qwerty_core::assessment::{
    ConfusionMatrix, EvaluationEvent, EvaluationReport, EvaluationStep, EvaluationView,
    LatencyStats, NegativePhase, OverallMetrics, PromptOutcome, ReportMeta, ZoneRef,
};
use qwerty_core::audio::{
    AudioFormat, CaptureMode, ChannelLayout, DefaultModeReason, DeviceIdentity, IdentityField,
};
use qwerty_core::descriptor::DescriptorError;
use qwerty_core::dsp::BAND_COUNT;
use qwerty_core::gesture::Trigger;
use qwerty_core::migrate::MigrationNote;
use qwerty_core::model::{BackgroundRegime, ModelRejectReason, ModelScores, Thresholds};
use qwerty_core::pipeline::{DecisionOutcome, Incompatibility, PipelineError};
use qwerty_core::profile::{
    ActionSpec, KeyCombo, Modifier, ProfileId, SystemSound, ZoneId, ZoneLayout,
};
use qwerty_core::recording::{NegativeClass, RecordingError, SegmentKind};
use qwerty_core::session::{
    BuildProgress, CalibrationStep, CalibrationView, EnvironmentQuality, TapFeedback, ZoneProgress,
};
use qwerty_core::store::{
    Config, DeviceFingerprint, NoticeId, RecalibrationReason, WindowPlacement, Zone, ZoneCount,
    ZoneEdit,
};
use qwerty_core::training::{
    ModelQuality, NegativeCount, NegativeSource, OutcomeCounts, TrainingProgress, TrainingSummary,
    ZoneTrainingStats,
};
use qwerty_core::units::{
    CandidateId, ChannelCount, Decibels, HostTime, Milliseconds, SampleIndex, SampleRate, Seconds,
    TapIndex, ZoneIndex,
};
use qwerty_core::veto::{OutputId, VetoReason};

use crate::engine::protocol::{
    ActionResult, ActivityEntry, ActivityLog, AttentionId, AttentionItem, Blocker,
    CalibrationAction, CalibrationOutcome, CalibrationRun, CalibrationStatus, CalibrationTarget,
    CaptureError, CapturePurpose, CaptureStatus, Command, CommandOutcome, ConfigEdit, ConfigStatus,
    Desired, DeviceChoice, DeviceInfo, DeviceProbe, DeviceState, EngineCounters, EvaluationOutcome,
    Event, Gate, GateRow, GateUnit, Health, HealthWarning, HotkeyEdit, InputDeviceClass, JobOrigin,
    LatencyBreakdown, MicUnavailableReason, MmcssStatus, Mode, OpenedStream, Paths, PendingTest,
    PersistenceFailure, PersistenceFailureKind, PreviewId, ProbeOutcome, ProfileCalibration,
    ProfileContent, ProfileSummary, ProfileView, RecorderState, RecorderStatus, RecorderStopReason,
    RejectedBy, ReportContent, ReportFormat, ReportId, ReportSummary, RetryState, RunSession,
    SkipReason, Snapshot, StepOutcome, StoredRecordings, TapRef, TapReport, TelemetryFrame,
    TelemetryLevel, ThreadName, TraceEntry, TraceOutcome, VetoStatus, VetoUnavailableReason,
    ZoneDraft,
};
use crate::platform::ActionError;

// ---------------------------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------------------------

/// An RFC 3339 time.
pub fn at(text: &str) -> DateTime<Utc> {
    // INVARIANT: only called with the fixture's own RFC 3339 literals.
    text.parse().expect("an RFC 3339 time")
}

/// Profile `n`: a fixed uuid.
pub fn profile_id(n: u8) -> ProfileId {
    let text = format!("\"00000000-0000-0000-0000-0000000001{n:02x}\"");
    // INVARIANT: a 36-character hex uuid string always parses.
    serde_json::from_str(&text).expect("a valid uuid string")
}

/// Zone `n`: a fixed uuid.
pub fn zone_id(n: u8) -> ZoneId {
    let text = format!("\"00000000-0000-0000-0000-0000000002{n:02x}\"");
    // INVARIANT: a 36-character hex uuid string always parses.
    serde_json::from_str(&text).expect("a valid uuid string")
}

fn ms(v: f64) -> Milliseconds {
    Milliseconds::new(v)
}

fn db(v: f64) -> Decibels {
    Decibels::new(v)
}

pub fn format() -> AudioFormat {
    AudioFormat {
        // INVARIANT: 48 kHz and 2 channels are inside the core's allowed ranges.
        rate: SampleRate::new(48_000).expect("valid rate"),
        channels: ChannelCount::new(2).expect("valid channels"),
    }
}

/// The desk microphone, opened raw.
pub fn identity() -> DeviceIdentity {
    DeviceIdentity {
        endpoint_id: "{0.0.1.00000000}.{mic-1}".to_string(),
        name: "Desk mic".to_string(),
        format: format(),
        capture_mode: CaptureMode::Raw,
    }
}

/// The same microphone, opened processed (raw was refused).
pub fn processed_identity() -> DeviceIdentity {
    DeviceIdentity {
        capture_mode: CaptureMode::Default {
            reason: DefaultModeReason::RawRejected {
                hresult: -2_004_287_478,
            },
        },
        ..identity()
    }
}

pub fn fingerprint() -> DeviceFingerprint {
    DeviceFingerprint {
        identity: processed_identity(),
        // INVARIANT: 2 channels are inside the core's allowed range.
        layout: ChannelLayout::all_real(ChannelCount::new(2).expect("valid channels")),
    }
}

/// Zone `n` of `count` side-by-side zones.
pub fn layout(n: u8, count: u8) -> ZoneLayout {
    let width = 1.0 / f32::from(count);
    ZoneLayout {
        x: width * f32::from(n),
        y: 0.0,
        width,
        height: 1.0,
    }
}

/// The fixture desk's zones: "Left" (opens Spotify) and "Right" (no actions).
pub fn zones() -> Vec<Zone> {
    vec![
        Zone {
            id: zone_id(1),
            label: "Left".to_string(),
            layout: layout(0, 2),
            trigger: Trigger::Single,
            actions: vec![
                ActionSpec::OpenApp {
                    path: "C:\\Users\\Me\\AppData\\Roaming\\Spotify\\Spotify.exe".to_string(),
                },
                ActionSpec::PlaySound {
                    sound: SystemSound::Success,
                },
            ],
        },
        Zone {
            id: zone_id(2),
            label: "Right".to_string(),
            layout: layout(1, 2),
            trigger: Trigger::Double,
            actions: Vec::new(),
        },
    ]
}

pub fn failure() -> PersistenceFailure {
    PersistenceFailure {
        file: "Profiles/broken.json".to_string(),
        kind: PersistenceFailureKind::Json,
        detail: "expected value at line 1 column 1".to_string(),
    }
}

pub fn overall() -> OverallMetrics {
    OverallMetrics {
        prompts: 40,
        correct: 38,
        wrong: 0,
        rejected: 2,
        missed: 0,
        accuracy: 0.95,
        wrong_action_rate: 0.0,
        rejection_rate: 0.05,
    }
}

pub fn latency_stats() -> LatencyStats {
    LatencyStats {
        count: 2,
        untimed: 0,
        p50: ms(51.5),
        p95: ms(56.0),
        p99: ms(56.0),
        max: ms(56.0),
        histogram: vec![0, 1, 1],
    }
}

/// When the fixture desk was calibrated: the stamp a current report carries.
pub const CALIBRATED_AT: &str = "2026-11-10T18:00:00Z";

/// A readable report, current when `model_trained_at` is [`CALIBRATED_AT`].
pub fn report_summary(model_trained_at: &str) -> ReportSummary {
    ReportSummary {
        report: ReportId("20261111T084500000Z".to_string()),
        content: ReportContent::Readable {
            run_at: at("2026-11-11T08:45:00Z"),
            model_trained_at: at(model_trained_at),
            target: 0.92,
            overall: overall(),
        },
    }
}

pub fn training_summary() -> TrainingSummary {
    let counts = OutcomeCounts {
        total: 20,
        correct: 19,
        wrong: 0,
        rejected_ood: 1,
        rejected_bg: 0,
        rejected_ambiguous: 0,
    };
    TrainingSummary {
        seed: 42,
        zones: vec![
            ZoneTrainingStats {
                zone: zone_id(1),
                taps: 10,
                vectors: 80,
                clean_accuracy: 1.0,
                clean_rejected: 0.0,
            },
            ZoneTrainingStats {
                zone: zone_id(2),
                taps: 10,
                vectors: 80,
                clean_accuracy: 0.9,
                clean_rejected: 0.1,
            },
        ],
        negatives: vec![NegativeCount {
            source: NegativeSource::EnvironmentOnset,
            clips: 6,
        }],
        negative_vectors: 48,
        variants_dropped: 0,
        within_lambda: 0.125,
        background_lambda: 0.25,
        background_regime: BackgroundRegime::Negatives {
            clips: 6,
            vectors: 48,
        },
        thresholds: Thresholds {
            alpha_in: 0.02,
            tau_in: 40.5,
            tau_bg: 1.5,
            tau_margin: 4.0,
        },
        held_out_clean: counts,
        held_out_all: counts,
        confusion_clean: vec![vec![10, 0, 0], vec![1, 9, 0]],
        negatives_held_out: 6,
        negatives_held_out_accepted: 0,
        quality: ModelQuality::Good,
    }
}

pub fn calibrated_status() -> CalibrationStatus {
    CalibrationStatus::Calibrated {
        device: fingerprint(),
        calibrated_at: at(CALIBRATED_AT),
        environment: EnvironmentQuality::Quiet,
        summary: Box::new(training_summary()),
    }
}

pub fn calibration_view() -> CalibrationView {
    CalibrationView {
        step: CalibrationStep::Zone {
            index: ZoneIndex(0),
            accepted: 4,
            target: 10,
        },
        zones: vec![
            ZoneProgress {
                zone: zone_id(1),
                label: "Left".to_string(),
                accepted: 4,
                target: 10,
            },
            ZoneProgress {
                zone: zone_id(2),
                label: "Right".to_string(),
                accepted: 0,
                target: 10,
            },
        ],
        last_feedback: Some(TapFeedback::Accepted {
            zone: ZoneIndex(0),
            tap: TapIndex(3),
            count: 4,
        }),
        memory_bytes: 1024,
        layout: None,
    }
}

pub fn calibration_run(session: RunSession) -> CalibrationRun {
    CalibrationRun {
        target: CalibrationTarget::New {
            name: "My desk".to_string(),
        },
        zones: zones()
            .into_iter()
            .map(|zone| ZoneDraft {
                label: zone.label,
                layout: zone.layout,
            })
            .collect(),
        device: identity(),
        taps_per_zone: 10,
        session,
    }
}

pub fn evaluation_view() -> EvaluationView {
    EvaluationView {
        step: EvaluationStep::Prompt {
            index: 1,
            total: 40,
            zone: ZoneIndex(0),
            deadline: HostTime(123_456_789),
        },
        resolved: 1,
        total_prompts: 40,
        correct: 1,
        last: Some(PromptOutcome::Correct {
            latency: Some(ms(52.0)),
        }),
    }
}

pub fn evaluation_report() -> EvaluationReport {
    EvaluationReport {
        schema_version: 2,
        profile_id: profile_id(1),
        run_at: at("2026-11-11T08:45:00Z"),
        meta: ReportMeta {
            profile_id: profile_id(1),
            device: identity(),
            front_end_version: 1,
            model_trained_at: at(CALIBRATED_AT),
            zone_taps: vec![10, 10],
            negatives: vec![],
            background_regime: BackgroundRegime::PooledAll {
                negative_clips: 0,
                needed: 4,
            },
            input_veto_enabled: true,
        },
        zones: vec![
            ZoneRef {
                id: zone_id(1),
                label: "Left".to_string(),
            },
            ZoneRef {
                id: zone_id(2),
                label: "Right".to_string(),
            },
        ],
        prompts_per_zone: 20,
        target: 0.92,
        undone: 0,
        confusion: ConfusionMatrix {
            zones: 2,
            counts: vec![vec![19, 0, 1], vec![0, 19, 1]],
        },
        per_zone: vec![],
        overall: overall(),
        negatives: vec![],
        decision_latency: latency_stats(),
        prompts: vec![],
        margin_histogram: vec![],
    }
}

pub fn opened(identity: DeviceIdentity) -> OpenedStream {
    OpenedStream {
        identity,
        period: ms(10.0),
        buffer_frames: 960,
        mmcss: MmcssStatus::Active,
    }
}

pub fn tap_ref() -> TapRef {
    TapRef {
        candidate: CandidateId(7),
        t0: HostTime(1_000_000),
        decided_at: HostTime(1_510_000),
        fired_at: HostTime(1_520_000),
    }
}

pub fn action_result(steps: Vec<StepOutcome>) -> ActionResult {
    ActionResult {
        started_at: HostTime(1_530_000),
        finished_at: HostTime(1_540_000),
        steps,
    }
}

pub fn scores() -> ModelScores {
    ModelScores {
        d2: vec![1.5, 9.0],
        best: ZoneIndex(0),
        second: ZoneIndex(1),
        d2_min: 1.5,
        margin: 7.5,
        llr_bg: 3.25,
        posterior_best: 0.75,
    }
}

pub fn trace_entry() -> TraceEntry {
    let row = |gate, unit, value, limit| GateRow {
        gate,
        value,
        limit,
        unit,
        passed: true,
    };
    TraceEntry {
        at: HostTime(1_510_000),
        onset: Some(SampleIndex(4800)),
        gates: Some([
            row(Gate::Snr, GateUnit::Decibels, 18.5, 10.0),
            row(Gate::BroadbandFraction, GateUnit::Fraction, 0.75, 0.5),
            row(Gate::ValidBands, GateUnit::Count, 20.0, 12.0),
            row(Gate::DecayFraction, GateUnit::Fraction, 0.5, 0.25),
        ]),
        outcome: TraceOutcome::Decided {
            outcome: Box::new(DecisionOutcome::Invalid {
                error: DescriptorError::SpanTooShort {
                    needed: 4096,
                    got: 2048,
                },
            }),
            scores: Some(scores()),
        },
        map_point: Some([0.5, -0.25]),
    }
}

pub fn counters() -> EngineCounters {
    EngineCounters {
        counters_since: at("2026-11-11T08:00:00Z"),
        taps_fired: 3,
        input_vetoed: 12,
        actions_failed: 1,
        capture_dropped_frames: 0,
        capture_packets: 36_000,
        controller_dropped: 0,
        events_dropped: 0,
        telemetry_dropped: 0,
        input_dropped: 0,
        decisions_dropped: 0,
        dispatch_dropped: 0,
        recorder_dropped_frames: 0,
        recorder_dropped_events: 0,
        stale_messages: 0,
        lifecycle_lost: 0,
        faults: 0,
    }
}

pub fn device_info() -> DeviceInfo {
    DeviceInfo {
        endpoint_id: identity().endpoint_id,
        name: "Desk mic".to_string(),
        is_default: true,
        state: DeviceState::Active,
        probe: ProbeOutcome::Probed {
            probe: DeviceProbe {
                format: format(),
                raw_supported: true,
            },
        },
    }
}

/// A second, unplugged microphone.
pub fn unplugged_device_info() -> DeviceInfo {
    DeviceInfo {
        endpoint_id: "{0.0.1.00000000}.{mic-2}".to_string(),
        name: "USB headset".to_string(),
        is_default: false,
        state: DeviceState::Unplugged,
        probe: ProbeOutcome::NotActive,
    }
}

pub fn veto_reason() -> VetoUnavailableReason {
    VetoUnavailableReason::RegistrationFailed {
        class: InputDeviceClass::Mouse,
        code: 5,
    }
}

pub fn tap_report() -> TapReport {
    TapReport {
        candidate: CandidateId(7),
        zone: zone_id(1),
        label: "Left".to_string(),
        t0: HostTime(1_000_000),
        latency: LatencyBreakdown {
            decision: ms(51.0),
            controller: ms(1.0),
        },
    }
}

pub fn engine_faulted() -> Event {
    Event::EngineFaulted {
        thread: ThreadName::Processing,
        message: "the processing thread panicked".to_string(),
    }
}

// ---------------------------------------------------------------------------------------------
// One value of every variant
// ---------------------------------------------------------------------------------------------

/// Every `Blocker` variant.
pub fn blockers() -> Vec<Blocker> {
    vec![
        Blocker::ProfileUnreadable {
            file: "Profiles/broken.json".to_string(),
            error: failure(),
        },
        Blocker::NeedsRecalibration {
            profile: profile_id(1),
            reason: RecalibrationReason::MigratedFromV1 {
                v1_device_name: "Desk mic".to_string(),
            },
        },
        Blocker::ProfileIncompatible {
            profile: profile_id(1),
            error: PipelineError::ModelIncompatible {
                reason: Incompatibility::FrontEndVersion {
                    model: 1,
                    current: 2,
                },
            },
        },
        Blocker::MicUnavailable {
            endpoint_id: identity().endpoint_id,
            name: "Desk mic".to_string(),
            reason: MicUnavailableReason::Unplugged,
        },
        Blocker::MicMismatch {
            bound: identity(),
            current: processed_identity(),
            fields: vec![IdentityField::CaptureMode],
        },
        Blocker::InputVetoUnavailable {
            reason: veto_reason(),
        },
    ]
}

/// Every `Mode` variant; `Calibrating` once per `RunSession` variant and `Blocked` once per
/// `Blocker` variant.
pub fn modes() -> Vec<Mode> {
    let mut modes = vec![
        Mode::Starting,
        Mode::Idle,
        Mode::Listening,
        Mode::Paused,
        Mode::Calibrating {
            run: Box::new(calibration_run(RunSession::Active {
                view: Box::new(calibration_view()),
            })),
        },
        Mode::Calibrating {
            run: Box::new(calibration_run(RunSession::MicLost {
                reason: MicUnavailableReason::Unplugged,
            })),
        },
        Mode::Evaluating {
            view: evaluation_view(),
        },
        Mode::Faulted {
            thread: ThreadName::Processing,
            message: "the processing thread panicked".to_string(),
        },
    ];
    modes.extend(
        blockers()
            .into_iter()
            .map(|blocker| Mode::Blocked { blocker }),
    );
    modes
}

/// Every `HealthWarning` variant.
pub fn health_warnings() -> Vec<HealthWarning> {
    vec![
        HealthWarning::CaptureOverflow { frames: 3 },
        HealthWarning::CaptureStalled { after: ms(2000.0) },
        HealthWarning::MmcssUnavailable { code: 1552 },
        HealthWarning::NoiseTrackerStarved {
            frozen_fraction: 0.5,
        },
        HealthWarning::LowSignal {
            near_miss_fraction: 0.25,
        },
        HealthWarning::ControllerOverloaded,
        HealthWarning::SelfOutputStuck { id: OutputId(3) },
        HealthWarning::SpeechCallbackPanicked { count: 1 },
        HealthWarning::EventsDropped { count: 4 },
        HealthWarning::DiskSpaceLow {
            free_bytes: 50_000_000,
        },
        HealthWarning::PlatformRebuilt {
            panic: "the platform panicked".to_string(),
        },
        HealthWarning::AutostartFailed {
            detail: "Access is denied.".to_string(),
        },
        HealthWarning::RecognitionDrift {
            window: 20,
            ambiguous: 6,
            close_calls: 3,
        },
        HealthWarning::SpeechUnmatched { count: 1 },
    ]
}

/// Every `ConfigStatus` variant.
pub fn config_statuses() -> Vec<ConfigStatus> {
    vec![
        ConfigStatus::Loaded,
        ConfigStatus::Initial,
        ConfigStatus::Migrated { from_version: 1 },
        ConfigStatus::Unreadable {
            error: PersistenceFailure {
                file: "config.json".to_string(),
                kind: PersistenceFailureKind::SchemaMismatch {
                    found: 9,
                    expected: 2,
                },
                detail: "a newer version wrote this file".to_string(),
            },
        },
    ]
}

/// Every `RetryState` variant.
pub fn retry_states() -> Vec<RetryState> {
    vec![
        RetryState::Scheduled {
            attempt: 1,
            at: HostTime(20_000_000),
        },
        RetryState::Exhausted,
    ]
}

/// Every `CaptureStatus` variant; `Running` raw and processed, `Failed` once per `RetryState`.
pub fn capture_statuses() -> Vec<CaptureStatus> {
    let mut statuses = vec![
        CaptureStatus::Off,
        CaptureStatus::Running {
            opened: opened(identity()),
            purpose: CapturePurpose::Listening,
        },
        CaptureStatus::Running {
            opened: opened(processed_identity()),
            purpose: CapturePurpose::Diagnostics,
        },
    ];
    statuses.extend(
        retry_states()
            .into_iter()
            .map(|retry| CaptureStatus::Failed {
                endpoint_id: identity().endpoint_id,
                error: CaptureError::DeviceInvalidated,
                retry,
            }),
    );
    statuses
}

/// Every `VetoStatus` variant.
pub fn veto_statuses() -> Vec<VetoStatus> {
    vec![
        VetoStatus::Disabled,
        VetoStatus::NotNeeded,
        VetoStatus::WaitingForShell,
        VetoStatus::Registering,
        VetoStatus::Active,
        VetoStatus::Unavailable {
            reason: veto_reason(),
        },
    ]
}

/// Every `RecorderState` variant.
pub fn recorder_states() -> Vec<RecorderState> {
    vec![
        RecorderState::Off,
        RecorderState::Recording {
            dir: "DebugCaptures/20261111T084500Z".to_string(),
            bytes: 384_044,
            duration: Seconds::new(4.0),
            gaps: 0,
            dropped_events: 0,
        },
        RecorderState::Stopped {
            dir: "DebugCaptures/20261111T084500Z".to_string(),
            reason: RecorderStopReason::SizeLimit,
        },
        RecorderState::Failed {
            error: RecordingError::DirExists {
                dir: "DebugCaptures/20261111T084500Z".to_string(),
            },
        },
    ]
}

/// One profile in each status (U:1012-1032): active and calibrated with a current report;
/// calibrated with a report from before the last calibration; imported and calibrated on another
/// microphone; needing recalibration after the upgrade; new and never calibrated; unreadable.
pub fn profiles() -> Vec<ProfileSummary> {
    let readable = |n: u8, name: &str, active: bool, calibration, imported, report| {
        ProfileSummary {
            file: format!("Profiles/00000000-0000-0000-0000-0000000001{n:02x}.json"),
            is_active: active,
            content: ProfileContent::Readable {
                id: profile_id(n),
                name: name.to_string(),
                // INVARIANT: 2 is one of ZoneCount::ALLOWED.
                zones: ZoneCount::new(2).expect("allowed count"),
                calibration,
                imported_at: imported,
                latest_report: report,
            },
        }
    };
    let calibrated = |identity| ProfileCalibration::Calibrated {
        device: identity,
        calibrated_at: at(CALIBRATED_AT),
    };
    vec![
        readable(
            1,
            "My desk",
            true,
            calibrated(identity()),
            None,
            Some(report_summary(CALIBRATED_AT)),
        ),
        readable(
            2,
            "Desk 2",
            false,
            calibrated(identity()),
            None,
            Some(report_summary("2026-11-01T09:00:00Z")),
        ),
        readable(
            3,
            "Office",
            false,
            calibrated(DeviceIdentity {
                endpoint_id: "{0.0.1.00000000}.{other-pc}".to_string(),
                name: "Office mic".to_string(),
                ..identity()
            }),
            Some(at("2026-11-09T12:00:00Z")),
            None,
        ),
        readable(
            4,
            "Old desk",
            false,
            ProfileCalibration::NeedsRecalibration {
                reason: RecalibrationReason::MigratedFromV1 {
                    v1_device_name: "Desk mic".to_string(),
                },
            },
            None,
            None,
        ),
        readable(
            5,
            "Desk 3",
            false,
            ProfileCalibration::NeedsRecalibration {
                reason: RecalibrationReason::NewProfile,
            },
            None,
            None,
        ),
        ProfileSummary {
            file: "Profiles/broken.json".to_string(),
            is_active: false,
            content: ProfileContent::Unreadable { error: failure() },
        },
    ]
}

/// Recent-taps rows (U:627-636), newest first: recognized, close call with a runner-up, an action
/// that didn't run, one still running, and not recognized with each reason.
pub fn activity_rows() -> Vec<ActivityEntry> {
    let fired = |at_text: &str, result, close_call| ActivityEntry::Fired {
        at: at(at_text),
        zone: zone_id(1),
        label: "Left".to_string(),
        origin: JobOrigin::Tap,
        tap: Some(tap_ref()),
        latency: Some(ms(53.0)),
        result,
        close_call,
    };
    let mut rows = vec![
        fired(
            "2026-11-11T08:44:00Z",
            Some(action_result(vec![StepOutcome::Done, StepOutcome::Done])),
            None,
        ),
        fired(
            "2026-11-11T08:43:00Z",
            Some(action_result(vec![StepOutcome::Done])),
            Some(zone_id(2)),
        ),
        fired(
            "2026-11-11T08:42:00Z",
            Some(action_result(vec![StepOutcome::Failed {
                error: ActionError::TargetNotFound {
                    target: "C:\\Tools\\missing.exe".to_string(),
                },
            }])),
            None,
        ),
        fired("2026-11-11T08:41:30Z", None, None),
    ];
    rows.extend(
        [
            ModelRejectReason::OutOfDistribution,
            ModelRejectReason::Background,
            ModelRejectReason::Ambiguous,
        ]
        .into_iter()
        .map(|reason| ActivityEntry::NotRecognized {
            at: at("2026-11-11T08:41:00Z"),
            reason,
            best_zone: Some(zone_id(2)),
        }),
    );
    rows
}

pub fn attention_item() -> AttentionItem {
    AttentionItem {
        id: AttentionId(1),
        first_at: at("2026-11-11T08:40:00Z"),
        last_at: at("2026-11-11T08:42:00Z"),
        count: 2,
        zone: zone_id(1),
        label: "Left".to_string(),
        error: ActionError::TargetNotFound {
            target: "C:\\Tools\\missing.exe".to_string(),
        },
    }
}

// ---------------------------------------------------------------------------------------------
// Telemetry frames
// ---------------------------------------------------------------------------------------------

pub fn meter_frame() -> TelemetryFrame {
    TelemetryFrame::Meter {
        rms: db(-52.0),
        peak: db(-31.5),
        noise: db(-58.0),
    }
}

/// A Spectrum frame whose detector frame ends at `end`.
pub fn spectrum_frame(end: SampleIndex) -> TelemetryFrame {
    TelemetryFrame::Spectrum {
        end,
        bands: Box::new([db(-60.0); BAND_COUNT]),
        noise: [db(-70.0); BAND_COUNT],
        flux: 0.25,
        threshold: 0.5,
    }
}

/// A Waveform frame (256 columns over 100 ms) whose last sample is `end`.
pub fn waveform_frame(end: SampleIndex) -> TelemetryFrame {
    const COLUMNS: usize = 256;
    TelemetryFrame::Waveform {
        end,
        min: (0..COLUMNS).map(|i| -0.001 * (i % 16) as f32).collect(),
        max: (0..COLUMNS).map(|i| 0.001 * (i % 16) as f32).collect(),
    }
}

pub fn trace_frame() -> TelemetryFrame {
    TelemetryFrame::Trace {
        entries: vec![trace_entry()],
    }
}

/// One frame of each kind.
pub fn telemetry_frames() -> Vec<TelemetryFrame> {
    vec![
        meter_frame(),
        spectrum_frame(SampleIndex(9600)),
        waveform_frame(SampleIndex(9600)),
        trace_frame(),
    ]
}

// ---------------------------------------------------------------------------------------------
// Snapshots
// ---------------------------------------------------------------------------------------------

/// The base snapshot: revision 1, listening on the fixture desk.
pub fn snapshot() -> Snapshot {
    Snapshot {
        revision: 1,
        mode: Mode::Listening,
        desired: Desired::Listening,
        config: Config::initial(),
        config_status: ConfigStatus::Loaded,
        profiles: profiles(),
        active_profile: Some(ProfileView {
            id: profile_id(1),
            name: "My desk".to_string(),
            zones: zones(),
            calibration: calibrated_status(),
            updated_at: at(CALIBRATED_AT),
        }),
        devices: vec![device_info(), unplugged_device_info()],
        capture: CaptureStatus::Running {
            opened: opened(identity()),
            purpose: CapturePurpose::Listening,
        },
        veto: VetoStatus::Active,
        recorder: RecorderStatus {
            state: RecorderState::Off,
            stored: StoredRecordings { count: 0, bytes: 0 },
        },
        telemetry: TelemetryLevel::Meter,
        health: Health {
            counters: counters(),
            latency: latency_stats(),
            warnings: Vec::new(),
        },
        activity: ActivityLog {
            recent: activity_rows(),
            attention: Vec::new(),
        },
        pending_tests: Vec::new(),
        unsaved_report: None,
        paths: Paths {
            data_dir: PathBuf::from("D:\\QwertyData"),
            logs: PathBuf::from("D:\\QwertyData\\Logs"),
            recordings: PathBuf::from("D:\\QwertyData\\DebugCaptures"),
        },
    }
}

/// Every optional field set and every list non-empty (CV1 reads every key from it).
pub fn maximal_snapshot() -> Snapshot {
    let mut config = Config::initial();
    config.active_profile_id = Some(profile_id(1));
    config.global_hotkey = Some(KeyCombo {
        modifiers: vec![Modifier::Ctrl, Modifier::Alt],
        key: "Space".to_string(),
    });
    config.dismissed_notices = vec![NoticeId::FirstCloseToTray];
    config.window_placement = Some(WindowPlacement {
        x: 100,
        y: 80,
        w: 1120,
        h: 760,
        maximized: false,
    });
    let mut snapshot = snapshot();
    snapshot.config = config;
    snapshot.recorder.stored = StoredRecordings {
        count: 1,
        bytes: 2048,
    };
    snapshot.health.warnings = health_warnings();
    snapshot.activity.attention = vec![attention_item()];
    snapshot.pending_tests = vec![PendingTest {
        zone: zone_id(1),
        origin: JobOrigin::TestAction { action_index: 0 },
        fires_at: HostTime(9_000_000),
    }];
    snapshot.unsaved_report = Some(report_summary(CALIBRATED_AT));
    snapshot
}

/// The fixture set: the maximal snapshot, then the base snapshot in every mode, capture status,
/// veto status, recorder state and config status, with no profiles, and with no active profile.
/// Revisions are 1, 2, 3, … in order.
pub fn snapshots() -> Vec<Snapshot> {
    let base = snapshot();
    let mut out = vec![maximal_snapshot()];
    out.extend(modes().into_iter().map(|mode| Snapshot {
        mode,
        ..base.clone()
    }));
    out.extend(capture_statuses().into_iter().map(|capture| Snapshot {
        capture,
        ..base.clone()
    }));
    out.extend(veto_statuses().into_iter().map(|veto| Snapshot {
        veto,
        ..base.clone()
    }));
    out.extend(recorder_states().into_iter().map(|state| Snapshot {
        recorder: RecorderStatus {
            state,
            stored: base.recorder.stored,
        },
        ..base.clone()
    }));
    out.extend(config_statuses().into_iter().map(|config_status| Snapshot {
        config_status,
        ..base.clone()
    }));
    out.push(Snapshot {
        mode: Mode::Idle,
        profiles: Vec::new(),
        active_profile: None,
        capture: CaptureStatus::Off,
        activity: ActivityLog {
            recent: Vec::new(),
            attention: Vec::new(),
        },
        ..base.clone()
    });
    out.push(Snapshot {
        mode: Mode::Idle,
        active_profile: None,
        capture: CaptureStatus::Off,
        ..base
    });
    for (n, snapshot) in out.iter_mut().enumerate() {
        snapshot.revision = n as u64 + 1;
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Events and commands
// ---------------------------------------------------------------------------------------------

/// Every `Event` variant once, and `Telemetry` once per frame kind.
pub fn events() -> Vec<Event> {
    let mut events = vec![
        Event::StateChanged {
            snapshot: Box::new(snapshot()),
        },
        Event::TapFired { tap: tap_report() },
        Event::TapRejected {
            rejected: RejectedBy::Vetoed {
                reason: VetoReason::InputOverflow,
            },
            best_zone: Some(zone_id(1)),
        },
        Event::GestureArmed {
            zone: zone_id(2),
            expires_at: HostTime(2_000_000),
        },
        Event::GestureCleared { zone: zone_id(2) },
        Event::ActionCompleted {
            zone: zone_id(1),
            tap: Some(tap_ref()),
            origin: JobOrigin::Tap,
            result: action_result(vec![StepOutcome::Started]),
        },
        Event::ActionSkipped {
            zone: zone_id(1),
            origin: JobOrigin::TestChain,
            reason: SkipReason::TestCanceled,
        },
        Event::CommandFinished {
            zone: zone_id(1),
            origin: JobOrigin::TestAction { action_index: 0 },
            step: 0,
            outcome: CommandOutcome::Exited { code: Some(0) },
        },
        Event::CalibrationFeedback {
            feedback: TapFeedback::TooWeak { snr: db(4.5) },
        },
        Event::CalibrationProgress {
            progress: BuildProgress::Training {
                progress: TrainingProgress::Folds { done: 2, total: 5 },
            },
        },
        Event::CalibrationEnded {
            outcome: CalibrationOutcome::Committed {
                profile: profile_id(1),
            },
        },
        Event::EvaluationUpdate {
            event: EvaluationEvent::PromptResolved {
                index: 1,
                zone: ZoneIndex(0),
                outcome: PromptOutcome::Rejected {
                    reason: ModelRejectReason::Ambiguous,
                },
            },
        },
        Event::EvaluationEnded {
            outcome: EvaluationOutcome::SaveFailed {
                report: Box::new(evaluation_report()),
                error: failure(),
            },
        },
        Event::ProfileMigrated {
            id: profile_id(4),
            from_version: 1,
            notes: vec![MigrationNote::HotkeyDefaulted],
        },
        Event::ConfigMigrated {
            from_version: 1,
            notes: vec![MigrationNote::HotkeyDefaulted],
        },
        engine_faulted(),
        Event::Stopped,
    ];
    events.extend(
        telemetry_frames()
            .into_iter()
            .map(|frame| Event::Telemetry { frame }),
    );
    events
}

/// Every `Command` variant once.
pub fn commands() -> Vec<Command> {
    let path = || PathBuf::from("D:\\out\\file.json");
    let report = || ReportId("20261111T084500000Z".to_string());
    vec![
        Command::ShellReady,
        Command::GetSnapshot,
        Command::Shutdown,
        Command::SetListening { listening: false },
        Command::ToggleListening,
        Command::SetTelemetry {
            level: TelemetryLevel::Diagnostics,
        },
        Command::ListDevices,
        Command::RetryCapture,
        Command::RetryInputVeto,
        Command::UpdateConfig {
            edit: ConfigEdit {
                text_zoom: Some(125),
                global_hotkey: Some(HotkeyEdit::Clear),
                ..ConfigEdit::default()
            },
        },
        Command::ResetConfig,
        Command::ReconcileAutostart,
        Command::SwitchProfile { id: profile_id(1) },
        Command::RenameProfile {
            id: profile_id(1),
            name: "Desk 2".to_string(),
        },
        Command::DuplicateProfile {
            id: profile_id(1),
            name: "My desk (copy)".to_string(),
        },
        Command::DeleteProfile { id: profile_id(2) },
        Command::EditZone {
            zone: zone_id(1),
            edit: ZoneEdit {
                label: Some("Right".to_string()),
                layout: None,
                trigger: Some(Trigger::Double),
                actions: Some(Vec::new()),
            },
        },
        Command::ExportProfile {
            id: profile_id(1),
            path: path(),
        },
        Command::PreviewImport { path: path() },
        Command::ImportProfile {
            preview: PreviewId(3),
        },
        Command::RebindDevice {
            profile: profile_id(3),
            endpoint_id: identity().endpoint_id,
        },
        Command::StartCalibration {
            target: CalibrationTarget::Recalibrate {
                profile: profile_id(1),
            },
            zones: calibration_run(RunSession::MicLost {
                reason: MicUnavailableReason::Unplugged,
            })
            .zones,
            device: DeviceChoice::Endpoint {
                endpoint_id: identity().endpoint_id,
            },
            taps_per_zone: 10,
        },
        Command::Calibration {
            action: CalibrationAction::RedoZone { zone: zone_id(1) },
        },
        Command::CancelCalibration,
        Command::StartEvaluation {
            prompts_per_zone: 15,
            negatives: vec![NegativePhase {
                class: NegativeClass::Everyday,
                duration: Seconds::new(30.0),
            }],
        },
        Command::UndoLastPrompt,
        Command::CancelEvaluation,
        Command::RetrySaveReport,
        Command::DiscardUnsavedReport,
        Command::ExportUnsavedReport {
            format: ReportFormat::Csv,
            path: path(),
        },
        Command::ListEvaluations {
            profile: profile_id(1),
        },
        Command::GetEvaluation {
            profile: profile_id(1),
            report: report(),
        },
        Command::ExportEvaluation {
            profile: profile_id(1),
            report: report(),
            format: ReportFormat::PromptsCsv,
            path: path(),
        },
        Command::DeleteEvaluation {
            profile: profile_id(1),
            report: report(),
        },
        Command::TestAction {
            zone: zone_id(1),
            action_index: 0,
            delay: ms(3000.0),
        },
        Command::TestChain {
            zone: zone_id(1),
            delay: ms(0.0),
        },
        Command::CancelTest { zone: zone_id(1) },
        Command::DismissAttention { id: AttentionId(1) },
        Command::ShellToast,
        Command::GetInspection,
        Command::ExportInspection { path: path() },
        Command::GetSoundMap,
        Command::SaveLatencyReport,
        Command::MarkSegment {
            kind: SegmentKind::Corpus {
                class: NegativeClass::Typing,
            },
        },
        Command::DeleteDebugRecordings,
    ]
}

/// The user-visible text the fixture set carries (names, labels, paths, device names), the input
/// set for the copy checks (CO2).
pub fn texts() -> Vec<String> {
    let mut texts: Vec<String> = Vec::new();
    for profile in profiles() {
        texts.push(profile.file.clone());
        if let ProfileContent::Readable { name, .. } = profile.content {
            texts.push(name);
        }
    }
    for zone in zones() {
        texts.push(zone.label);
        for action in zone.actions {
            if let ActionSpec::OpenApp { path } = action {
                texts.push(path);
            }
        }
    }
    texts.push(device_info().name);
    texts.push(unplugged_device_info().name);
    let paths = snapshot().paths;
    for path in [paths.data_dir, paths.logs, paths.recordings] {
        texts.push(path.display().to_string());
    }
    texts.push("C:\\Tools\\missing.exe".to_string());
    texts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fx1_fixture_snapshots_round_trip_with_increasing_revisions() {
        let snapshots = snapshots();
        for snapshot in &snapshots {
            let json = serde_json::to_string(snapshot).expect("serializes");
            let back: Snapshot = serde_json::from_str(&json).expect("deserializes");
            assert_eq!(
                &back, snapshot,
                "round trip of revision {}",
                snapshot.revision
            );
        }
        let revisions: Vec<u64> = snapshots.iter().map(|s| s.revision).collect();
        assert!(
            revisions.windows(2).all(|w| w[0] < w[1]),
            "revisions unique and increasing: {revisions:?}"
        );
        for event in events() {
            let json = serde_json::to_string(&event).expect("serializes");
            let back: Event = serde_json::from_str(&json).expect("deserializes");
            assert_eq!(back, event);
        }
        for command in commands() {
            let json = serde_json::to_string(&command).expect("serializes");
            let back: Command = serde_json::from_str(&json).expect("deserializes");
            assert_eq!(back, command);
        }
    }
}
