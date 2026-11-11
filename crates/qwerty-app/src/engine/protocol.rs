//! The complete engine boundary (p2 §2.9): commands, replies, events, the snapshot, errors, the
//! view types and the public constants. Every type derives `Debug, Clone, PartialEq, Serialize,
//! Deserialize`; enums are `#[serde(tag = "type")]`; units are the core newtypes.

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use qwerty_core::assessment::{
    AssessmentError, EvaluationEvent, EvaluationReport, EvaluationView, LatencyStats,
    NegativePhase, OverallMetrics,
};
use qwerty_core::audio::{AudioFormat, DeviceIdentity, IdentityField};
use qwerty_core::descriptor::DescriptorError;
use qwerty_core::detector::RejectReason;
use qwerty_core::dsp::BAND_COUNT;
use qwerty_core::migrate::MigrationNote;
use qwerty_core::model::{ModelRejectReason, ModelScores, SoundMap};
use qwerty_core::pipeline::{DecisionOutcome, PipelineError};
use qwerty_core::profile::{KeyCombo, NotificationVerbosity, ProfileId, Theme, ZoneId, ZoneLayout};
use qwerty_core::recording::{RecordingError, RecordingSummary, SegmentKind};
use qwerty_core::session::{
    BuildProgress, CalibrationView, EnvironmentQuality, SessionError, TapFeedback,
};
use qwerty_core::store::{
    Config, ConfigError, DeviceFingerprint, NoticeId, ProfileError, RecalibrationReason,
    WindowPlacement, Zone, ZoneCount, ZoneEdit,
};
use qwerty_core::training::TrainingSummary;
use qwerty_core::units::{CandidateId, Decibels, HostTime, Milliseconds, SampleIndex, Seconds};
use qwerty_core::veto::{OutputId, VetoReason};

use crate::platform::ActionError;

// ---------------------------------------------------------------------------
// Public constants (§2.9.5). The shell reads these and never copies them.
// ---------------------------------------------------------------------------

/// The wizard's pip ring after a zone that is not the last becomes full (XR-UI-02).
pub const AUTO_ADVANCE_DELAY: Milliseconds = Milliseconds::new(600.0);
/// ADR-005 on started_at − t0.
pub const LATENCY_BUDGET_P50: Milliseconds = Milliseconds::new(70.0);
/// ADR-005 on started_at − t0.
pub const LATENCY_BUDGET_P95: Milliseconds = Milliseconds::new(100.0);
/// Fired taps in `Health.latency` and a `LatencyReport`.
pub const LATENCY_WINDOW: usize = 1024;
/// A `LatencyReport` passes only with ≥ 100 taps, so its p95 rests on ≥ 5 samples above it.
pub const LATENCY_MIN_TAPS: u32 = 100;
/// `ActivityLog.recent` (§2.8).
pub const RECENT_CAPACITY: usize = 50;
/// `ActivityLog.attention` (§2.8).
pub const ATTENTION_CAPACITY: usize = 16;
/// The inspection log (§2.8).
pub const INSPECTION_CAPACITY: usize = 200;
/// How long a started RunCommand child is watched (§2.6).
pub const COMMAND_WATCH: Milliseconds = Milliseconds::new(4000.0);
/// Children watched at once (§2.6).
pub const MAX_WATCHED_COMMANDS: usize = 16;
/// How long the platform waits for a speech start (§2.6).
pub const SPEECH_START_TIMEOUT: Milliseconds = Milliseconds::new(5000.0);
/// Bounds a capture open, and so the controller's longest handler (§2.3).
pub const CAPTURE_OPEN_TIMEOUT: Milliseconds = Milliseconds::new(2000.0);
/// Capture reopen attempts after a failure (§2.8, PARITY-24).
pub const RETRY_SCHEDULE: [Milliseconds; 3] = [
    Milliseconds::new(2000.0),
    Milliseconds::new(10000.0),
    Milliseconds::new(60000.0),
];
/// Below any real microphone floor, so it only replaces digital silence's −∞.
pub const TELEMETRY_DB_FLOOR: Decibels = Decibels::new(-160.0);
/// `EngineHandle::request` (§2.9.1).
pub const REQUEST_TIMEOUT: Milliseconds = Milliseconds::new(5000.0);
/// The whole shutdown sequence (§2.11).
pub const SHUTDOWN_DEADLINE: Milliseconds = Milliseconds::new(5000.0);
/// `Logs/qwerty.log` rotates past this size (§2.9.4).
pub const LOG_FILE_LIMIT: u64 = 4 * 1024 * 1024;
/// An `Accepted` margin below 1.5 × τ_m is a close call (§2.8, XR-UI-20).
pub const CLOSE_CALL_RATIO: f64 = 1.5;
/// `RecognitionDrift` window (§2.8, XR-UI-20).
pub const DRIFT_WINDOW: u32 = 20;
/// `RecognitionDrift` raise threshold (§2.8).
pub const DRIFT_RAISE: u32 = 8;
/// `RecognitionDrift` clear threshold (§2.8).
pub const DRIFT_CLEAR: u32 = 3;

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Command {
    // lifecycle
    ShellReady,
    GetSnapshot,
    Shutdown,
    // listening
    SetListening {
        listening: bool,
    },
    /// tray + global hotkey
    ToggleListening,
    SetTelemetry {
        level: TelemetryLevel,
    },
    // devices
    ListDevices,
    RetryCapture,
    RetryInputVeto,
    // config
    UpdateConfig {
        edit: ConfigEdit,
    },
    /// Only when config_status == Unreadable (else ConfigReadable): renames the file to
    /// config.json.unreadable-<ts>, writes Config::initial().
    ResetConfig,
    /// §2.9.3 (XR-UI-18)
    ReconcileAutostart,
    // profiles
    /// Loads before committing (v1 semantics).
    SwitchProfile {
        id: ProfileId,
    },
    RenameProfile {
        id: ProfileId,
        name: String,
    },
    /// §2.9.4
    DuplicateProfile {
        id: ProfileId,
        name: String,
    },
    /// Also deletes Evaluations/<id>/.
    DeleteProfile {
        id: ProfileId,
    },
    /// Active profile; never invalidates the model; refused while Calibrating.
    EditZone {
        zone: ZoneId,
        edit: ZoneEdit,
    },
    /// Atomic write to the path the save dialog chose.
    ExportProfile {
        id: ProfileId,
        path: PathBuf,
    },
    /// Reads, migrates, validates; never saves.
    PreviewImport {
        path: PathBuf,
    },
    /// Exactly what was previewed; never activated.
    ImportProfile {
        preview: PreviewId,
    },
    /// An import's calibration → a matching microphone here (§2.8, XR-UI-16).
    RebindDevice {
        profile: ProfileId,
        endpoint_id: String,
    },
    // calibration (a draft until Commit, §2.8)
    StartCalibration {
        target: CalibrationTarget,
        zones: Vec<ZoneDraft>,
        device: DeviceChoice,
        taps_per_zone: u32,
    },
    Calibration {
        action: CalibrationAction,
    },
    CancelCalibration,
    // evaluation
    /// Background check: NegativePhase{class: Everyday, duration: 30 s} (XR-P01-20).
    StartEvaluation {
        prompts_per_zone: u32,
        negatives: Vec<NegativePhase>,
    },
    UndoLastPrompt,
    CancelEvaluation,
    /// While an unsaved report is held.
    RetrySaveReport,
    DiscardUnsavedReport,
    ExportUnsavedReport {
        format: ReportFormat,
        path: PathBuf,
    },
    ListEvaluations {
        profile: ProfileId,
    },
    GetEvaluation {
        profile: ProfileId,
        report: ReportId,
    },
    ExportEvaluation {
        profile: ProfileId,
        report: ReportId,
        format: ReportFormat,
        path: PathBuf,
    },
    DeleteEvaluation {
        profile: ProfileId,
        report: ReportId,
    },
    // actions
    /// The shell uses 3000 ms for SendKeystroke.
    TestAction {
        zone: ZoneId,
        action_index: u32,
        delay: Milliseconds,
    },
    TestChain {
        zone: ZoneId,
        delay: Milliseconds,
    },
    CancelTest {
        zone: ZoneId,
    },
    DismissAttention {
        id: AttentionId,
    },
    /// Sent just before the shell shows its own toast (PARITY-25).
    ShellToast,
    // diagnostics and evidence
    GetInspection,
    ExportInspection {
        path: PathBuf,
    },
    GetSoundMap,
    SaveLatencyReport,
    /// Requires an active recording; re-Configures the current setup with this segment.
    MarkSegment {
        kind: SegmentKind,
    },
    /// Refused while recording.
    DeleteDebugRecordings,
}

/// The payload-free mirror of [`Command`], used in errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CommandKind {
    ShellReady,
    GetSnapshot,
    Shutdown,
    SetListening,
    ToggleListening,
    SetTelemetry,
    ListDevices,
    RetryCapture,
    RetryInputVeto,
    UpdateConfig,
    ResetConfig,
    ReconcileAutostart,
    SwitchProfile,
    RenameProfile,
    DuplicateProfile,
    DeleteProfile,
    EditZone,
    ExportProfile,
    PreviewImport,
    ImportProfile,
    RebindDevice,
    StartCalibration,
    Calibration,
    CancelCalibration,
    StartEvaluation,
    UndoLastPrompt,
    CancelEvaluation,
    RetrySaveReport,
    DiscardUnsavedReport,
    ExportUnsavedReport,
    ListEvaluations,
    GetEvaluation,
    ExportEvaluation,
    DeleteEvaluation,
    TestAction,
    TestChain,
    CancelTest,
    DismissAttention,
    ShellToast,
    GetInspection,
    ExportInspection,
    GetSoundMap,
    SaveLatencyReport,
    MarkSegment,
    DeleteDebugRecordings,
}

impl Command {
    pub fn kind(&self) -> CommandKind {
        match self {
            Command::ShellReady => CommandKind::ShellReady,
            Command::GetSnapshot => CommandKind::GetSnapshot,
            Command::Shutdown => CommandKind::Shutdown,
            Command::SetListening { .. } => CommandKind::SetListening,
            Command::ToggleListening => CommandKind::ToggleListening,
            Command::SetTelemetry { .. } => CommandKind::SetTelemetry,
            Command::ListDevices => CommandKind::ListDevices,
            Command::RetryCapture => CommandKind::RetryCapture,
            Command::RetryInputVeto => CommandKind::RetryInputVeto,
            Command::UpdateConfig { .. } => CommandKind::UpdateConfig,
            Command::ResetConfig => CommandKind::ResetConfig,
            Command::ReconcileAutostart => CommandKind::ReconcileAutostart,
            Command::SwitchProfile { .. } => CommandKind::SwitchProfile,
            Command::RenameProfile { .. } => CommandKind::RenameProfile,
            Command::DuplicateProfile { .. } => CommandKind::DuplicateProfile,
            Command::DeleteProfile { .. } => CommandKind::DeleteProfile,
            Command::EditZone { .. } => CommandKind::EditZone,
            Command::ExportProfile { .. } => CommandKind::ExportProfile,
            Command::PreviewImport { .. } => CommandKind::PreviewImport,
            Command::ImportProfile { .. } => CommandKind::ImportProfile,
            Command::RebindDevice { .. } => CommandKind::RebindDevice,
            Command::StartCalibration { .. } => CommandKind::StartCalibration,
            Command::Calibration { .. } => CommandKind::Calibration,
            Command::CancelCalibration => CommandKind::CancelCalibration,
            Command::StartEvaluation { .. } => CommandKind::StartEvaluation,
            Command::UndoLastPrompt => CommandKind::UndoLastPrompt,
            Command::CancelEvaluation => CommandKind::CancelEvaluation,
            Command::RetrySaveReport => CommandKind::RetrySaveReport,
            Command::DiscardUnsavedReport => CommandKind::DiscardUnsavedReport,
            Command::ExportUnsavedReport { .. } => CommandKind::ExportUnsavedReport,
            Command::ListEvaluations { .. } => CommandKind::ListEvaluations,
            Command::GetEvaluation { .. } => CommandKind::GetEvaluation,
            Command::ExportEvaluation { .. } => CommandKind::ExportEvaluation,
            Command::DeleteEvaluation { .. } => CommandKind::DeleteEvaluation,
            Command::TestAction { .. } => CommandKind::TestAction,
            Command::TestChain { .. } => CommandKind::TestChain,
            Command::CancelTest { .. } => CommandKind::CancelTest,
            Command::DismissAttention { .. } => CommandKind::DismissAttention,
            Command::ShellToast => CommandKind::ShellToast,
            Command::GetInspection => CommandKind::GetInspection,
            Command::ExportInspection { .. } => CommandKind::ExportInspection,
            Command::GetSoundMap => CommandKind::GetSoundMap,
            Command::SaveLatencyReport => CommandKind::SaveLatencyReport,
            Command::MarkSegment { .. } => CommandKind::MarkSegment,
            Command::DeleteDebugRecordings => CommandKind::DeleteDebugRecordings,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DeviceChoice {
    ProfileBound,
    SystemDefault,
    Endpoint { endpoint_id: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CalibrationTarget {
    New { name: String },
    Recalibrate { profile: ProfileId },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZoneDraft {
    pub label: String,
    pub layout: ZoneLayout,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CalibrationAction {
    RecordAmbient,
    AcceptAmbient,
    RetryAmbient,
    UndoLastTap,
    RestartZone,
    NextZone,
    SkipNegatives,
    RedoZone { zone: ZoneId },
    Commit { name: String },
}

/// A patch: every field None = unchanged. `#[serde(default)]`, so the shell sends only what
/// changed. UpdateConfig applies it to a copy and saves only if Config::validate() passes; a
/// failure replies EngineError::Config{error} (XR-P01-13).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ConfigEdit {
    pub theme: Option<Theme>,
    pub start_with_windows: Option<bool>,
    pub minimize_to_tray_on_close: Option<bool>,
    pub notification_verbosity: Option<NotificationVerbosity>,
    pub global_hotkey: Option<HotkeyEdit>,
    pub debug_capture_enabled: Option<bool>,
    pub input_veto_enabled: Option<bool>,
    /// No tap_hud until the HUD ships (§1.16.1).
    pub eval_prompts_per_zone: Option<u32>,
    pub eval_background_check: Option<bool>,
    /// Appends to Config.dismissed_notices; one already there is a no-op.
    pub dismiss_notice: Option<NoticeId>,
    pub window_placement: Option<WindowPlacement>,
    /// XR-UI-19: one of TEXT_ZOOM_STEPS, else Config{InvalidTextZoom{value}} from Config::validate().
    pub text_zoom: Option<u16>,
}

/// RUST-RT-4: "clear" survives serde.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum HotkeyEdit {
    Clear,
    Set { combo: KeyCombo },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TelemetryLevel {
    Off,
    Meter,
    Diagnostics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ReportFormat {
    Json,
    Csv,
    PromptsCsv,
}

/// File stem YYYYMMDDTHHMMSSmmmZ.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ReportId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PreviewId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AttentionId(pub u64);

// ---------------------------------------------------------------------------
// Replies
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Reply {
    Ack,
    Snapshot {
        snapshot: Box<Snapshot>,
    },
    Devices {
        devices: Vec<DeviceInfo>,
    },
    Saved {
        path: PathBuf,
    },
    ProfileCreated {
        id: ProfileId,
    },
    ImportPreview {
        preview: Box<ImportPreview>,
    },
    Calibration {
        run: Box<CalibrationRun>,
    },
    Committed {
        profile: ProfileId,
    },
    Evaluation {
        view: EvaluationView,
    },
    Evaluations {
        reports: Vec<ReportSummary>,
    },
    EvaluationReport {
        report: Box<EvaluationReport>,
    },
    TestScheduled {
        fires_at: HostTime,
    },
    Inspection {
        entries: Vec<TraceEntry>,
    },
    /// map: the active model's TrainedModel.map (XR-P01-22); overlaps: its ZonesConfusable pairs
    /// (XR-UI-10).
    SoundMap {
        map: SoundMap,
        overlaps: Vec<[ZoneId; 2]>,
    },
    Shutdown {
        report: ShutdownReport,
    },
}

/// The payload-free mirror of [`Reply`], used in errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ReplyKind {
    Ack,
    Snapshot,
    Devices,
    Saved,
    ProfileCreated,
    ImportPreview,
    Calibration,
    Committed,
    Evaluation,
    Evaluations,
    EvaluationReport,
    TestScheduled,
    Inspection,
    SoundMap,
    Shutdown,
}

impl Reply {
    pub fn kind(&self) -> ReplyKind {
        match self {
            Reply::Ack => ReplyKind::Ack,
            Reply::Snapshot { .. } => ReplyKind::Snapshot,
            Reply::Devices { .. } => ReplyKind::Devices,
            Reply::Saved { .. } => ReplyKind::Saved,
            Reply::ProfileCreated { .. } => ReplyKind::ProfileCreated,
            Reply::ImportPreview { .. } => ReplyKind::ImportPreview,
            Reply::Calibration { .. } => ReplyKind::Calibration,
            Reply::Committed { .. } => ReplyKind::Committed,
            Reply::Evaluation { .. } => ReplyKind::Evaluation,
            Reply::Evaluations { .. } => ReplyKind::Evaluations,
            Reply::EvaluationReport { .. } => ReplyKind::EvaluationReport,
            Reply::TestScheduled { .. } => ReplyKind::TestScheduled,
            Reply::Inspection { .. } => ReplyKind::Inspection,
            Reply::SoundMap { .. } => ReplyKind::SoundMap,
            Reply::Shutdown { .. } => ReplyKind::Shutdown,
        }
    }

    fn mismatch(&self, expected: ReplyKind) -> EngineError {
        EngineError::ProtocolMismatch {
            expected,
            got: self.kind(),
        }
    }

    pub fn into_ack(self) -> Result<(), EngineError> {
        let Reply::Ack = self else {
            return Err(self.mismatch(ReplyKind::Ack));
        };
        Ok(())
    }

    pub fn into_snapshot(self) -> Result<Box<Snapshot>, EngineError> {
        let Reply::Snapshot { snapshot } = self else {
            return Err(self.mismatch(ReplyKind::Snapshot));
        };
        Ok(snapshot)
    }

    pub fn into_devices(self) -> Result<Vec<DeviceInfo>, EngineError> {
        let Reply::Devices { devices } = self else {
            return Err(self.mismatch(ReplyKind::Devices));
        };
        Ok(devices)
    }

    pub fn into_saved(self) -> Result<PathBuf, EngineError> {
        let Reply::Saved { path } = self else {
            return Err(self.mismatch(ReplyKind::Saved));
        };
        Ok(path)
    }

    pub fn into_profile_created(self) -> Result<ProfileId, EngineError> {
        let Reply::ProfileCreated { id } = self else {
            return Err(self.mismatch(ReplyKind::ProfileCreated));
        };
        Ok(id)
    }

    pub fn into_import_preview(self) -> Result<Box<ImportPreview>, EngineError> {
        let Reply::ImportPreview { preview } = self else {
            return Err(self.mismatch(ReplyKind::ImportPreview));
        };
        Ok(preview)
    }

    pub fn into_calibration(self) -> Result<Box<CalibrationRun>, EngineError> {
        let Reply::Calibration { run } = self else {
            return Err(self.mismatch(ReplyKind::Calibration));
        };
        Ok(run)
    }

    pub fn into_committed(self) -> Result<ProfileId, EngineError> {
        let Reply::Committed { profile } = self else {
            return Err(self.mismatch(ReplyKind::Committed));
        };
        Ok(profile)
    }

    pub fn into_evaluation(self) -> Result<EvaluationView, EngineError> {
        let Reply::Evaluation { view } = self else {
            return Err(self.mismatch(ReplyKind::Evaluation));
        };
        Ok(view)
    }

    pub fn into_evaluations(self) -> Result<Vec<ReportSummary>, EngineError> {
        let Reply::Evaluations { reports } = self else {
            return Err(self.mismatch(ReplyKind::Evaluations));
        };
        Ok(reports)
    }

    pub fn into_evaluation_report(self) -> Result<Box<EvaluationReport>, EngineError> {
        let Reply::EvaluationReport { report } = self else {
            return Err(self.mismatch(ReplyKind::EvaluationReport));
        };
        Ok(report)
    }

    pub fn into_test_scheduled(self) -> Result<HostTime, EngineError> {
        let Reply::TestScheduled { fires_at } = self else {
            return Err(self.mismatch(ReplyKind::TestScheduled));
        };
        Ok(fires_at)
    }

    pub fn into_inspection(self) -> Result<Vec<TraceEntry>, EngineError> {
        let Reply::Inspection { entries } = self else {
            return Err(self.mismatch(ReplyKind::Inspection));
        };
        Ok(entries)
    }

    /// (map, overlaps)
    pub fn into_sound_map(self) -> Result<(SoundMap, Vec<[ZoneId; 2]>), EngineError> {
        let Reply::SoundMap { map, overlaps } = self else {
            return Err(self.mismatch(ReplyKind::SoundMap));
        };
        Ok((map, overlaps))
    }

    pub fn into_shutdown(self) -> Result<ShutdownReport, EngineError> {
        let Reply::Shutdown { report } = self else {
            return Err(self.mismatch(ReplyKind::Shutdown));
        };
        Ok(report)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportPreview {
    pub preview: PreviewId,
    pub name: String,
    pub zones: Vec<Zone>,
    pub calibration: CalibrationStatus,
}

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub revision: u64,
    pub mode: Mode,
    pub desired: Desired,
    pub config: Config,
    pub config_status: ConfigStatus,
    pub profiles: Vec<ProfileSummary>,
    pub active_profile: Option<ProfileView>,
    pub devices: Vec<DeviceInfo>,
    pub capture: CaptureStatus,
    pub veto: VetoStatus,
    pub recorder: RecorderStatus,
    pub telemetry: TelemetryLevel,
    pub health: Health,
    pub activity: ActivityLog,
    pub pending_tests: Vec<PendingTest>,
    /// XR-UI-06
    pub unsaved_report: Option<ReportSummary>,
    pub paths: Paths,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Mode {
    Starting,
    /// No active profile (first run: no profile files at all).
    Idle,
    Listening,
    Paused,
    Calibrating {
        run: Box<CalibrationRun>,
    },
    Evaluating {
        view: EvaluationView,
    },
    Blocked {
        blocker: Blocker,
    },
    /// Terminal for this engine instance.
    Faulted {
        thread: ThreadName,
        message: String,
    },
}

/// The payload-free mirror of [`Mode`], used in errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ModeKind {
    Starting,
    Idle,
    Listening,
    Paused,
    Calibrating,
    Evaluating,
    Blocked,
    Faulted,
}

impl Mode {
    pub fn kind(&self) -> ModeKind {
        match self {
            Mode::Starting => ModeKind::Starting,
            Mode::Idle => ModeKind::Idle,
            Mode::Listening => ModeKind::Listening,
            Mode::Paused => ModeKind::Paused,
            Mode::Calibrating { .. } => ModeKind::Calibrating,
            Mode::Evaluating { .. } => ModeKind::Evaluating,
            Mode::Blocked { .. } => ModeKind::Blocked,
            Mode::Faulted { .. } => ModeKind::Faulted,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationRun {
    pub target: CalibrationTarget,
    pub zones: Vec<ZoneDraft>,
    pub device: DeviceIdentity,
    pub taps_per_zone: u32,
    pub session: RunSession,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RunSession {
    /// Boxed by §2.1.1's boxing rule (large_enum_variant); serializes as the view.
    Active {
        view: Box<CalibrationView>,
    },
    MicLost {
        reason: MicUnavailableReason,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Desired {
    Listening,
    Paused,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Blocker {
    /// PARITY-23
    ProfileUnreadable {
        file: String,
        error: PersistenceFailure,
    },
    NeedsRecalibration {
        profile: ProfileId,
        reason: RecalibrationReason,
    },
    /// RUST-RT-17
    ProfileIncompatible {
        profile: ProfileId,
        error: PipelineError,
    },
    MicUnavailable {
        endpoint_id: String,
        name: String,
        reason: MicUnavailableReason,
    },
    MicMismatch {
        bound: DeviceIdentity,
        current: DeviceIdentity,
        fields: Vec<IdentityField>,
    },
    InputVetoUnavailable {
        reason: VetoUnavailableReason,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MicUnavailableReason {
    NotPresent,
    Disabled,
    Unplugged,
    AccessDenied,
    InUseExclusive,
    Stalled,
    OpenFailed { error: CaptureError },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ConfigStatus {
    Loaded,
    Initial,
    Migrated { from_version: u32 },
    Unreadable { error: PersistenceFailure },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersistenceFailure {
    pub file: String,
    pub kind: PersistenceFailureKind,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PersistenceFailureKind {
    Io,
    Json,
    SchemaMismatch { found: u32, expected: u32 },
    InvalidData,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileSummary {
    pub file: String,
    pub is_active: bool,
    pub content: ProfileContent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ProfileContent {
    Readable {
        id: ProfileId,
        name: String,
        zones: ZoneCount,
        calibration: ProfileCalibration,
        /// XR-UI-16
        imported_at: Option<DateTime<Utc>>,
        /// The newest Readable report.
        latest_report: Option<ReportSummary>,
    },
    Unreadable {
        error: PersistenceFailure,
    },
}

/// A report is current iff its model_trained_at == calibrated_at: both are the one value stamped
/// at Commit (PARITY-12).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ProfileCalibration {
    Calibrated {
        device: DeviceIdentity,
        calibrated_at: DateTime<Utc>,
    },
    NeedsRecalibration {
        reason: RecalibrationReason,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileView {
    pub id: ProfileId,
    pub name: String,
    pub zones: Vec<Zone>,
    pub calibration: CalibrationStatus,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CalibrationStatus {
    Calibrated {
        device: DeviceFingerprint,
        calibrated_at: DateTime<Utc>,
        environment: EnvironmentQuality,
        /// Boxed by §2.1.1's boxing rule (large_enum_variant).
        summary: Box<TrainingSummary>,
    },
    NeedsRecalibration {
        reason: RecalibrationReason,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportSummary {
    pub report: ReportId,
    pub content: ReportContent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ReportContent {
    Readable {
        run_at: DateTime<Utc>,
        model_trained_at: DateTime<Utc>,
        target: f64,
        overall: OverallMetrics,
    },
    /// A schema-1 (v1) report via assessment::peek_legacy_report: "older format, not comparable"
    /// (XR-P01-15).
    Legacy {
        run_at: DateTime<Utc>,
        overall_accuracy: f64,
    },
    Unreadable {
        error: PersistenceFailure,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CaptureStatus {
    Off,
    Running {
        opened: OpenedStream,
        purpose: CapturePurpose,
    },
    Failed {
        endpoint_id: String,
        error: CaptureError,
        retry: RetryState,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RetryState {
    Scheduled { attempt: u8, at: HostTime },
    Exhausted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CapturePurpose {
    Listening,
    Calibrating,
    Evaluating,
    Diagnostics,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum VetoStatus {
    Disabled,
    NotNeeded,
    WaitingForShell,
    Registering,
    Active,
    Unavailable { reason: VetoUnavailableReason },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecorderStatus {
    pub state: RecorderState,
    pub stored: StoredRecordings,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RecorderState {
    Off,
    Recording {
        dir: String,
        bytes: u64,
        duration: Seconds,
        gaps: u32,
        dropped_events: u32,
    },
    Stopped {
        dir: String,
        reason: RecorderStopReason,
    },
    Failed {
        error: RecordingError,
    },
}

/// DebugCaptures/: measured at startup, after each recording, after DeleteDebugRecordings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredRecordings {
    pub count: u32,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RecorderStopReason {
    Disabled,
    CaptureStopped,
    SizeLimit,
    DiskSpaceLow { free_bytes: u64 },
    StructureEventLost,
    NewCalibrationRun,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Health {
    pub counters: EngineCounters,
    /// Core §1.17; started_at − t0 over the last LATENCY_WINDOW fired taps.
    pub latency: LatencyStats,
    pub warnings: Vec<HealthWarning>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineCounters {
    /// Engine start (XR-UI-09).
    pub counters_since: DateTime<Utc>,
    pub taps_fired: u64,
    pub input_vetoed: u64,
    pub actions_failed: u64,
    pub capture_dropped_frames: u64,
    pub controller_dropped: u64,
    pub events_dropped: u64,
    pub telemetry_dropped: u64,
    pub input_dropped: u64,
    pub decisions_dropped: u64,
    pub dispatch_dropped: u64,
    pub recorder_dropped_frames: u64,
    pub recorder_dropped_events: u64,
    pub stale_messages: u64,
    pub lifecycle_lost: u64,
    pub faults: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum HealthWarning {
    CaptureOverflow {
        frames: u64,
    },
    CaptureStalled {
        after: Milliseconds,
    },
    MmcssUnavailable {
        code: u32,
    },
    NoiseTrackerStarved {
        frozen_fraction: f64,
    },
    LowSignal {
        near_miss_fraction: f64,
    },
    ControllerOverloaded,
    SelfOutputStuck {
        id: OutputId,
    },
    SpeechCallbackPanicked {
        count: u64,
    },
    EventsDropped {
        count: u64,
    },
    DiskSpaceLow {
        free_bytes: u64,
    },
    PlatformRebuilt {
        panic: String,
    },
    AutostartFailed {
        detail: String,
    },
    /// XR-UI-20, §2.8
    RecognitionDrift {
        window: u32,
        ambiguous: u32,
        close_calls: u32,
    },
    /// Sticky: an invariant broke, §2.8.
    SpeechUnmatched {
        count: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ThreadName {
    Controller,
    Capture,
    Processing,
    Dispatch,
    RawInput,
    Recorder,
    Build,
}

impl ThreadName {
    /// The OS thread name (§2.2).
    pub fn os_name(self) -> &'static str {
        match self {
            ThreadName::Controller => "qwerty-ctl",
            ThreadName::Capture => "qwerty-capture",
            ThreadName::Processing => "qwerty-dsp",
            ThreadName::Dispatch => "qwerty-dispatch",
            ThreadName::RawInput => "qwerty-rawinput",
            ThreadName::Recorder => "qwerty-rec",
            ThreadName::Build => "qwerty-build",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivityLog {
    /// Newest first.
    pub recent: Vec<ActivityEntry>,
    pub attention: Vec<AttentionItem>,
}

/// XR-UI-17
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ActivityEntry {
    Fired {
        at: DateTime<Utc>,
        zone: ZoneId,
        label: String,
        origin: JobOrigin,
        tap: Option<TapRef>,
        /// started_at − t0, origin Tap; None until Completed.
        latency: Option<Milliseconds>,
        /// None until Completed.
        result: Option<ActionResult>,
        /// XR-UI-20: the runner-up zone (model.zone_ids[scores.second]) when close_call() is Some;
        /// None otherwise and for tests.
        close_call: Option<ZoneId>,
    },
    NotRecognized {
        at: DateTime<Utc>,
        reason: ModelRejectReason,
        best_zone: Option<ZoneId>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionItem {
    pub id: AttentionId,
    pub first_at: DateTime<Utc>,
    pub last_at: DateTime<Utc>,
    pub count: u32,
    pub zone: ZoneId,
    pub label: String,
    pub error: ActionError,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingTest {
    pub zone: ZoneId,
    pub origin: JobOrigin,
    pub fires_at: HostTime,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Paths {
    pub data_dir: PathBuf,
    pub logs: PathBuf,
    pub recordings: PathBuf,
}

/// <data_dir>/Latency/<ts>.json (PARITY-11): the saved evidence for the end-to-end latency
/// acceptance row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencyReport {
    pub schema_version: u32,
    pub created_at: DateTime<Utc>,
    pub profile_id: ProfileId,
    pub device: DeviceIdentity,
    /// started_at − t0
    pub end_to_end: LatencyStats,
    pub decision: LatencyStats,
    pub controller: LatencyStats,
    pub dispatch: LatencyStats,
    pub budget_p50: Milliseconds,
    pub budget_p95: Milliseconds,
    /// end_to_end.count ≥ LATENCY_MIN_TAPS ∧ p50 < budget_p50 ∧ p95 < budget_p95
    pub passed: bool,
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Event {
    StateChanged {
        snapshot: Box<Snapshot>,
    },
    TapFired {
        tap: TapReport,
    },
    /// Decisions only; coalesced, ≤ 10 Hz.
    TapRejected {
        rejected: RejectedBy,
        best_zone: Option<ZoneId>,
    },
    GestureArmed {
        zone: ZoneId,
        expires_at: HostTime,
    },
    GestureCleared {
        zone: ZoneId,
    },
    ActionCompleted {
        zone: ZoneId,
        tap: Option<TapRef>,
        origin: JobOrigin,
        result: ActionResult,
    },
    ActionSkipped {
        zone: ZoneId,
        origin: JobOrigin,
        reason: SkipReason,
    },
    CommandFinished {
        zone: ZoneId,
        origin: JobOrigin,
        step: u32,
        outcome: CommandOutcome,
    },
    CalibrationFeedback {
        feedback: TapFeedback,
    },
    CalibrationProgress {
        progress: BuildProgress,
    },
    /// lifecycle
    CalibrationEnded {
        outcome: CalibrationOutcome,
    },
    EvaluationUpdate {
        event: EvaluationEvent,
    },
    /// lifecycle
    EvaluationEnded {
        outcome: EvaluationOutcome,
    },
    /// What the migration changed (XR-P01-14).
    ProfileMigrated {
        id: ProfileId,
        from_version: u32,
        notes: Vec<MigrationNote>,
    },
    ConfigMigrated {
        from_version: u32,
        notes: Vec<MigrationNote>,
    },
    Telemetry {
        frame: TelemetryFrame,
    },
    /// lifecycle
    EngineFaulted {
        thread: ThreadName,
        message: String,
    },
    /// lifecycle
    Stopped,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RejectedBy {
    Model { reason: ModelRejectReason },
    Vetoed { reason: VetoReason },
    Invalid { error: DescriptorError },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SkipReason {
    QueueFull,
    TestCanceled,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CalibrationOutcome {
    Committed { profile: ProfileId },
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EvaluationOutcome {
    Saved {
        report: ReportId,
        overall: OverallMetrics,
    },
    /// Held as Snapshot.unsaved_report.
    SaveFailed {
        report: Box<EvaluationReport>,
        error: PersistenceFailure,
    },
    Cancelled,
    Aborted {
        reason: MicUnavailableReason,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TapReport {
    pub candidate: CandidateId,
    pub zone: ZoneId,
    pub label: String,
    pub t0: HostTime,
    pub latency: LatencyBreakdown,
}

/// The dispatch wait is ActionResult.started_at − fired_at.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LatencyBreakdown {
    /// decided_at − t0
    pub decision: Milliseconds,
    /// fired_at − decided_at
    pub controller: Milliseconds,
}

/// Every dB value is 10·log10(power), floored at TELEMETRY_DB_FLOOR (−160 dB) so it is always
/// finite (serde_json writes non-finite floats as null, RUST-RT-16). Meter: `rms` = mean square and
/// `peak` = max square of the high-passed samples over the last 50 ms (one frame at the ≤ 20 Hz
/// Meter rate), all real channels, re full scale (a full-scale sine is −3 dB); `noise` =
/// 10·log10(mean_b N_b / n_real), the same formula as the session's noise floor. Spectrum: the
/// detector's band powers and noise estimate (§1.5.4 normalization), so bands and noise compare
/// directly (PARITY-38).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TelemetryFrame {
    Meter {
        rms: Decibels,
        peak: Decibels,
        noise: Decibels,
    },
    Spectrum {
        // SPEC: §2.1.1's boxing rule (large_enum_variant) boxes both arrays, so a Spectrum frame
        // makes two small allocations, where §2.5 calls Spectrum frames "fixed arrays".
        bands: Box<[Decibels; BAND_COUNT]>,
        noise: Box<[Decibels; BAND_COUNT]>,
        flux: f32,
        threshold: f32,
    },
    /// Diagnostics: 256 columns over the last 100 ms.
    Waveform { min: Vec<f32>, max: Vec<f32> },
    /// Diagnostics: from the controller (§2.8); ≤ 10 Hz, ≤ 32 entries, older ones counted in
    /// telemetry_dropped.
    Trace { entries: Vec<TraceEntry> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceEntry {
    pub at: HostTime,
    pub onset: Option<SampleIndex>,
    pub gates: Option<[GateRow; 4]>,
    pub outcome: TraceOutcome,
    pub map_point: Option<[f32; 2]>,
}

/// One detector gate, in the detector's evaluation order; limit = the DetectorParams field the
/// detector compares `value` with (§1.6); passed = the detector's own verdict for that gate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GateRow {
    pub gate: Gate,
    pub value: f64,
    pub limit: f64,
    pub unit: GateUnit,
    pub passed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Gate {
    Snr,
    BroadbandFraction,
    ValidBands,
    DecayFraction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GateUnit {
    Decibels,
    Fraction,
    Count,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TraceOutcome {
    DetectorRejected {
        reason: RejectReason,
    },
    /// Monitor: passed the gates; no model.
    Detected,
    Decided {
        /// Boxed by §2.1.1's boxing rule (large_enum_variant).
        outcome: Box<DecisionOutcome>,
        scores: Option<ModelScores>,
    },
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EngineError {
    ShuttingDown,
    /// The controller thread is gone.
    EngineStopped,
    /// No reply within REQUEST_TIMEOUT: a bug, never expected.
    Timeout {
        command: CommandKind,
    },
    ProtocolMismatch {
        expected: ReplyKind,
        got: ReplyKind,
    },
    InvalidInMode {
        command: CommandKind,
        mode: ModeKind,
    },
    NoActiveProfile,
    /// ProfileBound device choice, StartEvaluation or RebindDevice on a NeedsRecalibration profile.
    ProfileNotCalibrated,
    ProfileNotFound {
        id: ProfileId,
    },
    ReportNotFound {
        profile: ProfileId,
        report: ReportId,
    },
    Profile {
        error: ProfileError,
    },
    Persistence {
        error: PersistenceFailure,
    },
    /// Config writes refused until ResetConfig.
    ConfigUnreadable,
    /// ResetConfig refused: the config is readable (PARITY-38).
    ConfigReadable,
    /// Config::validate() refused an UpdateConfig (XR-P01-13).
    Config {
        error: ConfigError,
    },
    /// key ∉ the platform's vk_of set; a missing modifier is Config{HotkeyWithoutModifier}.
    UnsupportedHotkey {
        combo: KeyCombo,
    },
    /// GetEvaluation / ExportEvaluation of a schema-1 report (XR-P01-15).
    LegacyReport {
        report: ReportId,
    },
    Capture {
        error: CaptureError,
    },
    /// A processing setup could not be built.
    Setup {
        error: PipelineError,
    },
    Calibration {
        error: SessionError,
    },
    Evaluation {
        error: AssessmentError,
    },
    Veto {
        reason: VetoUnavailableReason,
    },
    /// The veto is enabled and still registering.
    VetoNotReady,
    /// StartEvaluation while an unsaved report is held.
    UnsavedReportHeld,
    NoUnsavedReport,
    /// ImportProfile with a preview that was replaced or consumed.
    PreviewExpired,
    TestPending {
        zone: ZoneId,
    },
    NoPendingTest {
        zone: ZoneId,
    },
    AttentionNotFound {
        id: AttentionId,
    },
    NotRecording,
    /// DeleteDebugRecordings while recording.
    RecordingActive,
    Autostart {
        detail: String,
    },
    AutostartNotManaged,
    ThreadSpawn {
        thread: ThreadName,
        detail: String,
    },
}

/// What `EngineHandle::shutdown` reports (§2.9.1, §2.11).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShutdownReport {
    pub joined: Vec<ThreadName>,
    pub timed_out: Vec<ThreadName>,
    pub recording: Option<RecordingSummary>,
}

/// Whether this engine instance may touch the Run key (§2.9.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AutostartPolicy {
    Reconcile,
    Untouched,
}

/// Never saved. The GUI and --v2-monitor pass FromConfig; --v2-live --record and --v2-corpus pass
/// ForceOn, so a CLI session's recording never outlives it (§2.13, §2.15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DebugCaptureMode {
    FromConfig,
    ForceOn,
}

// ---------------------------------------------------------------------------
// The serde types the snapshot embeds (§2.3, §2.4, §2.6, §2.7)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenedStream {
    pub identity: DeviceIdentity,
    pub period: Milliseconds,
    pub buffer_frames: u32,
    pub mmcss: MmcssStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MmcssStatus {
    Active,
    Unavailable { code: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CaptureStage {
    Spawn,
    Enumerate,
    Activate,
    Properties,
    MixFormat,
    Initialize,
    EventHandle,
    CaptureClient,
    Start,
    Read,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CaptureError {
    ComInit {
        hresult: i32,
    },
    DeviceNotFound {
        endpoint_id: String,
    },
    DeviceNotActive {
        endpoint_id: String,
        state: DeviceState,
    },
    /// DeviceChoice::SystemDefault while no capture endpoint is Active (§2.4, §2.8).
    NoDefaultDevice,
    /// E_ACCESSDENIED: Windows microphone privacy.
    AccessDenied,
    /// AUDCLNT_E_DEVICE_IN_USE
    InUseExclusive,
    /// tag = wFormatTag, or Data1 of the EXTENSIBLE subformat GUID.
    UnsupportedMixFormat {
        tag: u16,
        bits: u16,
    },
    UnsupportedSampleRate {
        hz: u32,
    },
    UnsupportedChannelCount {
        channels: u32,
    },
    /// AUDCLNT_E_DEVICE_INVALIDATED (unplug, format change).
    DeviceInvalidated,
    Stalled {
        after: Milliseconds,
    },
    OpenTimeout,
    /// Any other failing HRESULT, with where it happened.
    Os {
        stage: CaptureStage,
        hresult: i32,
    },
    /// wasapi variants impossible by construction (§2.3).
    Unexpected {
        stage: CaptureStage,
        detail: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub endpoint_id: String,
    pub name: String,
    pub is_default: bool,
    pub state: DeviceState,
    pub probe: ProbeOutcome,
}

/// One device's probe, never the listing's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ProbeOutcome {
    NotActive,
    Probed { probe: DeviceProbe },
    Failed { error: CaptureError },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceProbe {
    pub format: AudioFormat,
    pub raw_supported: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DeviceState {
    Active,
    Disabled,
    NotPresent,
    Unplugged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InputDeviceClass {
    Keyboard,
    Mouse,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum VetoUnavailableReason {
    RegistrationFailed { class: InputDeviceClass, code: u32 },
    RegistrationNotOwned { class: InputDeviceClass },
    OwnershipLost { class: InputDeviceClass },
    ThreadFailed { message: String },
}

/// A Fire is always timed: untimed ⇒ Vetoed{ClockUnavailable} (§1.15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TapRef {
    pub candidate: CandidateId,
    pub t0: HostTime,
    pub decided_at: HostTime,
    pub fired_at: HostTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum JobOrigin {
    Tap,
    TestAction { action_index: u32 },
    TestChain,
}

/// One entry per step that ran, in order. A Failed or Canceled entry is the last: later steps did
/// not run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionResult {
    pub started_at: HostTime,
    pub finished_at: HostTime,
    pub steps: Vec<StepOutcome>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StepOutcome {
    Done,
    /// RunCommand: the child runs; its exit arrives as Event::CommandFinished.
    Started,
    /// The user cancelled (screenshot region selection).
    Canceled,
    Failed {
        error: ActionError,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CommandOutcome {
    Exited { code: Option<i32> },
    StillRunning,
    WaitFailed { detail: String },
}
