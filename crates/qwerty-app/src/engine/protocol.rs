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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::fmt::Debug;

    use serde::de::DeserializeOwned;

    use qwerty_core::assessment::{
        ConfusionMatrix, EvaluationStep, PromptOutcome, ReportMeta, ZoneRef,
    };
    use qwerty_core::audio::{CaptureMode, ChannelLayout, DefaultModeReason};
    use qwerty_core::gesture::Trigger;
    use qwerty_core::model::{BackgroundRegime, MapPoint, Thresholds};
    use qwerty_core::pipeline::Incompatibility;
    use qwerty_core::profile::{ActionSpec, Modifier, SystemSound};
    use qwerty_core::recording::NegativeClass;
    use qwerty_core::session::{CalibrationStep, ZoneProgress};
    use qwerty_core::training::{
        ModelQuality, NegativeCount, NegativeSource, OutcomeCounts, TrainingProgress,
        ZoneTrainingStats,
    };
    use qwerty_core::units::{ChannelCount, SampleRate, TapIndex, ZoneIndex};
    use qwerty_core::veto::SelfOutputKind;

    /// serde round trip: the value comes back equal, and re-serializes to the same JSON.
    fn rt<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) -> serde_json::Value {
        let json = serde_json::to_string(value).expect("serializes");
        let back: T = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(&back, value, "round trip of {json}");
        assert_eq!(serde_json::to_string(&back).expect("serializes"), json);
        serde_json::from_str(&json).expect("valid JSON")
    }

    /// Round-trips every value of an enum and checks that `count` distinct variants were covered.
    fn rt_variants<T: Serialize + DeserializeOwned + PartialEq + Debug>(
        values: &[T],
        count: usize,
    ) {
        let tags: BTreeSet<String> = values
            .iter()
            .map(|v| match rt(v) {
                serde_json::Value::Object(map) => match map.get("type") {
                    Some(serde_json::Value::String(tag)) => tag.clone(),
                    other => panic!("no string type tag: {other:?}"),
                },
                other => panic!("an enum serializes as an object: {other}"),
            })
            .collect();
        assert_eq!(tags.len(), count, "variants covered: {tags:?}");
    }

    fn at(text: &str) -> DateTime<Utc> {
        text.parse().expect("an RFC 3339 time")
    }

    fn profile_id(n: u8) -> ProfileId {
        let text = format!("\"00000000-0000-0000-0000-0000000001{n:02x}\"");
        serde_json::from_str(&text).expect("a valid uuid string")
    }

    fn zone_id(n: u8) -> ZoneId {
        let text = format!("\"00000000-0000-0000-0000-0000000002{n:02x}\"");
        serde_json::from_str(&text).expect("a valid uuid string")
    }

    fn format() -> AudioFormat {
        AudioFormat {
            rate: SampleRate::new(48_000).expect("valid rate"),
            channels: ChannelCount::new(2).expect("valid channels"),
        }
    }

    fn identity() -> DeviceIdentity {
        DeviceIdentity {
            endpoint_id: "{0.0.1.00000000}.{mic-1}".to_string(),
            name: "Desk mic".to_string(),
            format: format(),
            capture_mode: CaptureMode::Raw,
        }
    }

    fn fingerprint() -> DeviceFingerprint {
        DeviceFingerprint {
            identity: DeviceIdentity {
                capture_mode: CaptureMode::Default {
                    reason: DefaultModeReason::RawRejected {
                        hresult: -2_004_287_478,
                    },
                },
                ..identity()
            },
            layout: ChannelLayout::all_real(ChannelCount::new(2).expect("valid channels")),
        }
    }

    fn layout() -> ZoneLayout {
        ZoneLayout {
            x: 0.0,
            y: 0.0,
            width: 0.5,
            height: 1.0,
        }
    }

    fn zone() -> Zone {
        Zone {
            id: zone_id(1),
            label: "Left".to_string(),
            layout: layout(),
            trigger: Trigger::Single,
            actions: vec![ActionSpec::PlaySound {
                sound: SystemSound::Success,
            }],
        }
    }

    fn failure() -> PersistenceFailure {
        PersistenceFailure {
            file: "Profiles/x.json".to_string(),
            kind: PersistenceFailureKind::Json,
            detail: "expected value at line 1 column 1".to_string(),
        }
    }

    fn overall() -> OverallMetrics {
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

    fn latency_stats() -> LatencyStats {
        LatencyStats {
            count: 2,
            untimed: 0,
            p50: Milliseconds::new(51.5),
            p95: Milliseconds::new(56.0),
            p99: Milliseconds::new(56.0),
            max: Milliseconds::new(56.0),
            histogram: vec![0, 1, 1],
        }
    }

    fn report_summary() -> ReportSummary {
        ReportSummary {
            report: ReportId("20261111T084500000Z".to_string()),
            content: ReportContent::Readable {
                run_at: at("2026-11-11T08:45:00Z"),
                model_trained_at: at("2026-11-10T18:00:00Z"),
                target: 0.92,
                overall: overall(),
            },
        }
    }

    fn training_summary() -> TrainingSummary {
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
            zones: vec![ZoneTrainingStats {
                zone: zone_id(1),
                taps: 10,
                vectors: 80,
                clean_accuracy: 1.0,
                clean_rejected: 0.0,
            }],
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
            confusion_clean: vec![vec![10, 0]],
            negatives_held_out: 6,
            negatives_held_out_accepted: 0,
            quality: ModelQuality::Good,
        }
    }

    fn calibration_view() -> CalibrationView {
        CalibrationView {
            step: CalibrationStep::Ambient {
                elapsed: Milliseconds::new(1500.0),
                needed: Milliseconds::new(5000.0),
            },
            zones: vec![ZoneProgress {
                zone: zone_id(1),
                label: "Left".to_string(),
                accepted: 0,
                target: 10,
            }],
            last_feedback: Some(TapFeedback::Ignored),
            memory_bytes: 1024,
            layout: None,
        }
    }

    fn evaluation_view() -> EvaluationView {
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
                latency: Some(Milliseconds::new(52.0)),
            }),
        }
    }

    fn evaluation_report() -> EvaluationReport {
        EvaluationReport {
            schema_version: 2,
            profile_id: profile_id(1),
            run_at: at("2026-11-11T08:45:00Z"),
            meta: ReportMeta {
                profile_id: profile_id(1),
                device: identity(),
                front_end_version: 1,
                model_trained_at: at("2026-11-10T18:00:00Z"),
                zone_taps: vec![10],
                negatives: vec![],
                background_regime: BackgroundRegime::PooledAll {
                    negative_clips: 0,
                    needed: 4,
                },
                input_veto_enabled: true,
            },
            zones: vec![ZoneRef {
                id: zone_id(1),
                label: "Left".to_string(),
            }],
            prompts_per_zone: 40,
            target: 0.92,
            undone: 0,
            confusion: ConfusionMatrix {
                zones: 1,
                counts: vec![vec![38, 2]],
            },
            per_zone: vec![],
            overall: overall(),
            negatives: vec![],
            decision_latency: latency_stats(),
            prompts: vec![],
            margin_histogram: vec![],
        }
    }

    fn calibration_run() -> CalibrationRun {
        CalibrationRun {
            target: CalibrationTarget::New {
                name: "Desk".to_string(),
            },
            zones: vec![ZoneDraft {
                label: "Left".to_string(),
                layout: layout(),
            }],
            device: identity(),
            taps_per_zone: 10,
            session: RunSession::Active {
                view: Box::new(calibration_view()),
            },
        }
    }

    fn opened() -> OpenedStream {
        OpenedStream {
            identity: identity(),
            period: Milliseconds::new(10.0),
            buffer_frames: 960,
            mmcss: MmcssStatus::Active,
        }
    }

    fn tap_ref() -> TapRef {
        TapRef {
            candidate: CandidateId(7),
            t0: HostTime(1_000_000),
            decided_at: HostTime(1_510_000),
            fired_at: HostTime(1_520_000),
        }
    }

    fn action_result() -> ActionResult {
        ActionResult {
            started_at: HostTime(1_530_000),
            finished_at: HostTime(1_540_000),
            steps: vec![StepOutcome::Done],
        }
    }

    fn scores() -> ModelScores {
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

    fn trace_entry() -> TraceEntry {
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

    fn counters() -> EngineCounters {
        EngineCounters {
            counters_since: at("2026-11-11T08:00:00Z"),
            taps_fired: 3,
            input_vetoed: 1,
            actions_failed: 0,
            capture_dropped_frames: 0,
            controller_dropped: 0,
            events_dropped: 0,
            telemetry_dropped: 0,
            input_dropped: 0,
            decisions_dropped: 0,
            dispatch_dropped: 0,
            recorder_dropped_frames: 0,
            recorder_dropped_events: 0,
            stale_messages: 2,
            lifecycle_lost: 0,
            faults: 0,
        }
    }

    fn device_info() -> DeviceInfo {
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

    fn veto_reason() -> VetoUnavailableReason {
        VetoUnavailableReason::RegistrationFailed {
            class: InputDeviceClass::Mouse,
            code: 5,
        }
    }

    fn calibrated_status() -> CalibrationStatus {
        CalibrationStatus::Calibrated {
            device: fingerprint(),
            calibrated_at: at("2026-11-10T18:00:00Z"),
            environment: EnvironmentQuality::Quiet,
            summary: Box::new(training_summary()),
        }
    }

    fn import_preview() -> ImportPreview {
        ImportPreview {
            preview: PreviewId(1),
            name: "Desk".to_string(),
            zones: vec![zone()],
            calibration: calibrated_status(),
        }
    }

    fn sound_map() -> SoundMap {
        SoundMap {
            axes: [vec![1.0, 0.0], vec![0.0, 1.0]],
            taps: vec![MapPoint {
                zone: ZoneIndex(0),
                at: [0.5, -0.25],
            }],
            background: vec![[0.0, 0.125]],
        }
    }

    fn shutdown_report() -> ShutdownReport {
        ShutdownReport {
            joined: vec![ThreadName::Capture, ThreadName::Processing],
            timed_out: vec![ThreadName::Dispatch],
            recording: Some(RecordingSummary {
                dir: "DebugCaptures/20261111T084500Z".to_string(),
                frames: 48_000,
                bytes: 384_044,
                events: 12,
                gaps: 0,
            }),
        }
    }

    /// The golden snapshot: every top-level field holds a representative value.
    fn snapshot() -> Snapshot {
        Snapshot {
            revision: 7,
            mode: Mode::Listening,
            desired: Desired::Listening,
            config: Config::initial(),
            config_status: ConfigStatus::Loaded,
            profiles: vec![
                ProfileSummary {
                    file: "Profiles/00000000-0000-0000-0000-000000000101.json".to_string(),
                    is_active: true,
                    content: ProfileContent::Readable {
                        id: profile_id(1),
                        name: "Desk".to_string(),
                        zones: ZoneCount::new(2).expect("allowed count"),
                        calibration: ProfileCalibration::Calibrated {
                            device: identity(),
                            calibrated_at: at("2026-11-10T18:00:00Z"),
                        },
                        imported_at: None,
                        latest_report: Some(report_summary()),
                    },
                },
                ProfileSummary {
                    file: "Profiles/broken.json".to_string(),
                    is_active: false,
                    content: ProfileContent::Unreadable { error: failure() },
                },
            ],
            active_profile: Some(ProfileView {
                id: profile_id(1),
                name: "Desk".to_string(),
                zones: vec![zone()],
                calibration: CalibrationStatus::NeedsRecalibration {
                    reason: RecalibrationReason::NewProfile,
                },
                updated_at: at("2026-11-10T18:00:00Z"),
            }),
            devices: vec![device_info()],
            capture: CaptureStatus::Running {
                opened: opened(),
                purpose: CapturePurpose::Listening,
            },
            veto: VetoStatus::Active,
            recorder: RecorderStatus {
                state: RecorderState::Off,
                stored: StoredRecordings {
                    count: 1,
                    bytes: 2048,
                },
            },
            telemetry: TelemetryLevel::Off,
            health: Health {
                counters: counters(),
                latency: latency_stats(),
                warnings: vec![HealthWarning::MmcssUnavailable { code: 1552 }],
            },
            activity: ActivityLog {
                recent: vec![ActivityEntry::Fired {
                    at: at("2026-11-11T08:44:00Z"),
                    zone: zone_id(1),
                    label: "Left".to_string(),
                    origin: JobOrigin::Tap,
                    tap: Some(tap_ref()),
                    latency: Some(Milliseconds::new(53.0)),
                    result: Some(action_result()),
                    close_call: None,
                }],
                attention: vec![AttentionItem {
                    id: AttentionId(1),
                    first_at: at("2026-11-11T08:40:00Z"),
                    last_at: at("2026-11-11T08:41:00Z"),
                    count: 2,
                    zone: zone_id(1),
                    label: "Left".to_string(),
                    error: ActionError::TargetNotFound {
                        target: "C:/Tools/missing.exe".to_string(),
                    },
                }],
            },
            pending_tests: vec![PendingTest {
                zone: zone_id(1),
                origin: JobOrigin::TestAction { action_index: 0 },
                fires_at: HostTime(9_000_000),
            }],
            unsaved_report: None,
            paths: Paths {
                data_dir: PathBuf::from("D:/QwertyData"),
                logs: PathBuf::from("D:/QwertyData/Logs"),
                recordings: PathBuf::from("D:/QwertyData/DebugCaptures"),
            },
        }
    }

    /// Every Command variant, once.
    fn commands() -> Vec<Command> {
        let path = || PathBuf::from("D:/out/file.json");
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
                name: "Copy".to_string(),
            },
            Command::DeleteProfile { id: profile_id(2) },
            Command::EditZone {
                zone: zone_id(1),
                edit: ZoneEdit {
                    label: Some("Right".to_string()),
                    layout: None,
                    trigger: Some(Trigger::Double),
                    actions: Some(vec![ActionSpec::VisualOnly]),
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
                profile: profile_id(1),
                endpoint_id: "{mic-2}".to_string(),
            },
            Command::StartCalibration {
                target: CalibrationTarget::Recalibrate {
                    profile: profile_id(1),
                },
                zones: vec![ZoneDraft {
                    label: "Left".to_string(),
                    layout: layout(),
                }],
                device: DeviceChoice::Endpoint {
                    endpoint_id: "{mic-1}".to_string(),
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
                action_index: 2,
                delay: Milliseconds::new(3000.0),
            },
            Command::TestChain {
                zone: zone_id(1),
                delay: Milliseconds::new(0.0),
            },
            Command::CancelTest { zone: zone_id(1) },
            Command::DismissAttention { id: AttentionId(4) },
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

    /// Every Reply variant, once.
    fn replies() -> Vec<Reply> {
        vec![
            Reply::Ack,
            Reply::Snapshot {
                snapshot: Box::new(snapshot()),
            },
            Reply::Devices {
                devices: vec![device_info()],
            },
            Reply::Saved {
                path: PathBuf::from("D:/out/file.json"),
            },
            Reply::ProfileCreated { id: profile_id(2) },
            Reply::ImportPreview {
                preview: Box::new(import_preview()),
            },
            Reply::Calibration {
                run: Box::new(calibration_run()),
            },
            Reply::Committed {
                profile: profile_id(1),
            },
            Reply::Evaluation {
                view: evaluation_view(),
            },
            Reply::Evaluations {
                reports: vec![report_summary()],
            },
            Reply::EvaluationReport {
                report: Box::new(evaluation_report()),
            },
            Reply::TestScheduled {
                fires_at: HostTime(9_000_000),
            },
            Reply::Inspection {
                entries: vec![trace_entry()],
            },
            Reply::SoundMap {
                map: sound_map(),
                overlaps: vec![[zone_id(1), zone_id(2)]],
            },
            Reply::Shutdown {
                report: shutdown_report(),
            },
        ]
    }

    fn capture_errors() -> Vec<CaptureError> {
        vec![
            CaptureError::ComInit {
                hresult: -2_147_417_850,
            },
            CaptureError::DeviceNotFound {
                endpoint_id: "{mic-9}".to_string(),
            },
            CaptureError::DeviceNotActive {
                endpoint_id: "{mic-9}".to_string(),
                state: DeviceState::Unplugged,
            },
            CaptureError::NoDefaultDevice,
            CaptureError::AccessDenied,
            CaptureError::InUseExclusive,
            CaptureError::UnsupportedMixFormat { tag: 1, bits: 16 },
            CaptureError::UnsupportedSampleRate { hz: 8000 },
            CaptureError::UnsupportedChannelCount { channels: 64 },
            CaptureError::DeviceInvalidated,
            CaptureError::Stalled {
                after: Milliseconds::new(2000.0),
            },
            CaptureError::OpenTimeout,
            CaptureError::Os {
                stage: CaptureStage::Initialize,
                hresult: -2_004_287_480,
            },
            CaptureError::Unexpected {
                stage: CaptureStage::Read,
                detail: "data length mismatch".to_string(),
            },
        ]
    }

    #[test]
    fn en9_protocol_serde_round_trips_and_goldens() {
        // ---- Commands, replies and their kinds ----
        let commands = commands();
        rt_variants(&commands, 45);
        let kinds: Vec<CommandKind> = commands.iter().map(Command::kind).collect();
        rt_variants(&kinds, 45);
        let replies = replies();
        rt_variants(&replies, 15);
        let reply_kinds: Vec<ReplyKind> = replies.iter().map(Reply::kind).collect();
        rt_variants(&reply_kinds, 15);
        rt_variants(
            &[
                DeviceChoice::ProfileBound,
                DeviceChoice::SystemDefault,
                DeviceChoice::Endpoint {
                    endpoint_id: "{mic-1}".to_string(),
                },
            ],
            3,
        );
        rt_variants(
            &[
                CalibrationTarget::New {
                    name: "Desk".to_string(),
                },
                CalibrationTarget::Recalibrate {
                    profile: profile_id(1),
                },
            ],
            2,
        );
        rt_variants(
            &[
                CalibrationAction::RecordAmbient,
                CalibrationAction::AcceptAmbient,
                CalibrationAction::RetryAmbient,
                CalibrationAction::UndoLastTap,
                CalibrationAction::RestartZone,
                CalibrationAction::NextZone,
                CalibrationAction::SkipNegatives,
                CalibrationAction::RedoZone { zone: zone_id(1) },
                CalibrationAction::Commit {
                    name: "Desk".to_string(),
                },
            ],
            9,
        );
        rt_variants(
            &[
                HotkeyEdit::Clear,
                HotkeyEdit::Set {
                    combo: KeyCombo {
                        modifiers: vec![Modifier::Ctrl, Modifier::Alt],
                        key: "K".to_string(),
                    },
                },
            ],
            2,
        );
        rt_variants(
            &[
                TelemetryLevel::Off,
                TelemetryLevel::Meter,
                TelemetryLevel::Diagnostics,
            ],
            3,
        );
        rt_variants(
            &[
                ReportFormat::Json,
                ReportFormat::Csv,
                ReportFormat::PromptsCsv,
            ],
            3,
        );
        let full_edit = ConfigEdit {
            theme: Some(Theme::Midnight),
            start_with_windows: Some(true),
            minimize_to_tray_on_close: Some(false),
            notification_verbosity: Some(NotificationVerbosity::Minimal),
            global_hotkey: Some(HotkeyEdit::Clear),
            debug_capture_enabled: Some(false),
            input_veto_enabled: Some(true),
            eval_prompts_per_zone: Some(20),
            eval_background_check: Some(false),
            dismiss_notice: Some(NoticeId::UnattendedActions),
            window_placement: Some(WindowPlacement {
                x: 10,
                y: 20,
                w: 800,
                h: 600,
                maximized: false,
            }),
            text_zoom: Some(125),
        };
        rt(&full_edit);
        // #[serde(default)]: the shell sends only what changed.
        let partial: ConfigEdit =
            serde_json::from_str(r#"{"text_zoom":105}"#).expect("a partial patch");
        assert_eq!(
            partial,
            ConfigEdit {
                text_zoom: Some(105),
                ..ConfigEdit::default()
            }
        );
        assert_eq!(
            rt(&ReportId("20261111T084500000Z".to_string())),
            "20261111T084500000Z"
        );
        assert_eq!(rt(&PreviewId(3)), 3);
        assert_eq!(rt(&AttentionId(4)), 4);
        rt(&ZoneDraft {
            label: "Left".to_string(),
            layout: layout(),
        });
        rt(&import_preview());

        // ---- Snapshot and its parts ----
        let modes = vec![
            Mode::Starting,
            Mode::Idle,
            Mode::Listening,
            Mode::Paused,
            Mode::Calibrating {
                run: Box::new(calibration_run()),
            },
            Mode::Evaluating {
                view: evaluation_view(),
            },
            Mode::Blocked {
                blocker: Blocker::InputVetoUnavailable {
                    reason: veto_reason(),
                },
            },
            Mode::Faulted {
                thread: ThreadName::Capture,
                message: "capture panicked".to_string(),
            },
        ];
        rt_variants(&modes, 8);
        let mode_kinds: Vec<ModeKind> = modes.iter().map(Mode::kind).collect();
        rt_variants(&mode_kinds, 8);
        rt(&calibration_run());
        rt_variants(
            &[
                RunSession::Active {
                    view: Box::new(calibration_view()),
                },
                RunSession::MicLost {
                    reason: MicUnavailableReason::Stalled,
                },
            ],
            2,
        );
        rt_variants(&[Desired::Listening, Desired::Paused], 2);
        rt_variants(
            &[
                Blocker::ProfileUnreadable {
                    file: "Profiles/x.json".to_string(),
                    error: failure(),
                },
                Blocker::NeedsRecalibration {
                    profile: profile_id(1),
                    reason: RecalibrationReason::FrontEndChanged {
                        trained: 1,
                        current: 2,
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
                    endpoint_id: "{mic-1}".to_string(),
                    name: "Desk mic".to_string(),
                    reason: MicUnavailableReason::NotPresent,
                },
                Blocker::MicMismatch {
                    bound: identity(),
                    current: fingerprint().identity,
                    fields: vec![IdentityField::CaptureMode],
                },
                Blocker::InputVetoUnavailable {
                    reason: veto_reason(),
                },
            ],
            6,
        );
        rt_variants(
            &[
                MicUnavailableReason::NotPresent,
                MicUnavailableReason::Disabled,
                MicUnavailableReason::Unplugged,
                MicUnavailableReason::AccessDenied,
                MicUnavailableReason::InUseExclusive,
                MicUnavailableReason::Stalled,
                MicUnavailableReason::OpenFailed {
                    error: CaptureError::OpenTimeout,
                },
            ],
            7,
        );
        rt_variants(
            &[
                ConfigStatus::Loaded,
                ConfigStatus::Initial,
                ConfigStatus::Migrated { from_version: 1 },
                ConfigStatus::Unreadable { error: failure() },
            ],
            4,
        );
        rt(&failure());
        rt_variants(
            &[
                PersistenceFailureKind::Io,
                PersistenceFailureKind::Json,
                PersistenceFailureKind::SchemaMismatch {
                    found: 3,
                    expected: 2,
                },
                PersistenceFailureKind::InvalidData,
            ],
            4,
        );
        rt_variants(
            &[
                ProfileContent::Readable {
                    id: profile_id(1),
                    name: "Desk".to_string(),
                    zones: ZoneCount::new(4).expect("allowed count"),
                    calibration: ProfileCalibration::NeedsRecalibration {
                        reason: RecalibrationReason::MigratedFromV1 {
                            v1_device_name: "Old mic".to_string(),
                        },
                    },
                    imported_at: Some(at("2026-11-09T10:00:00Z")),
                    latest_report: None,
                },
                ProfileContent::Unreadable { error: failure() },
            ],
            2,
        );
        rt_variants(
            &[
                ProfileCalibration::Calibrated {
                    device: identity(),
                    calibrated_at: at("2026-11-10T18:00:00Z"),
                },
                ProfileCalibration::NeedsRecalibration {
                    reason: RecalibrationReason::ZoneSetChanged,
                },
            ],
            2,
        );
        rt_variants(
            &[
                calibrated_status(),
                CalibrationStatus::NeedsRecalibration {
                    reason: RecalibrationReason::NewProfile,
                },
            ],
            2,
        );
        rt_variants(
            &[
                report_summary().content,
                ReportContent::Legacy {
                    run_at: at("2026-10-01T12:00:00Z"),
                    overall_accuracy: 0.875,
                },
                ReportContent::Unreadable { error: failure() },
            ],
            3,
        );
        rt_variants(
            &[
                CaptureStatus::Off,
                CaptureStatus::Running {
                    opened: opened(),
                    purpose: CapturePurpose::Diagnostics,
                },
                CaptureStatus::Failed {
                    endpoint_id: "{mic-1}".to_string(),
                    error: CaptureError::DeviceInvalidated,
                    retry: RetryState::Scheduled {
                        attempt: 1,
                        at: HostTime(20_000_000),
                    },
                },
            ],
            3,
        );
        rt_variants(
            &[
                RetryState::Scheduled {
                    attempt: 3,
                    at: HostTime(1),
                },
                RetryState::Exhausted,
            ],
            2,
        );
        rt_variants(
            &[
                CapturePurpose::Listening,
                CapturePurpose::Calibrating,
                CapturePurpose::Evaluating,
                CapturePurpose::Diagnostics,
            ],
            4,
        );
        rt_variants(
            &[
                VetoStatus::Disabled,
                VetoStatus::NotNeeded,
                VetoStatus::WaitingForShell,
                VetoStatus::Registering,
                VetoStatus::Active,
                VetoStatus::Unavailable {
                    reason: veto_reason(),
                },
            ],
            6,
        );
        rt_variants(
            &[
                RecorderState::Off,
                RecorderState::Recording {
                    dir: "DebugCaptures/20261111T084500Z".to_string(),
                    bytes: 4096,
                    duration: Seconds::new(1.5),
                    gaps: 1,
                    dropped_events: 0,
                },
                RecorderState::Stopped {
                    dir: "DebugCaptures/20261111T084500Z".to_string(),
                    reason: RecorderStopReason::SizeLimit,
                },
                RecorderState::Failed {
                    error: RecordingError::FormatMismatch,
                },
            ],
            4,
        );
        rt_variants(
            &[
                RecorderStopReason::Disabled,
                RecorderStopReason::CaptureStopped,
                RecorderStopReason::SizeLimit,
                RecorderStopReason::DiskSpaceLow { free_bytes: 1024 },
                RecorderStopReason::StructureEventLost,
                RecorderStopReason::NewCalibrationRun,
            ],
            6,
        );
        rt_variants(
            &[
                HealthWarning::CaptureOverflow { frames: 480 },
                HealthWarning::CaptureStalled {
                    after: Milliseconds::new(500.0),
                },
                HealthWarning::MmcssUnavailable { code: 1552 },
                HealthWarning::NoiseTrackerStarved {
                    frozen_fraction: 0.625,
                },
                HealthWarning::LowSignal {
                    near_miss_fraction: 0.5,
                },
                HealthWarning::ControllerOverloaded,
                HealthWarning::SelfOutputStuck { id: OutputId(9) },
                HealthWarning::SpeechCallbackPanicked { count: 1 },
                HealthWarning::EventsDropped { count: 3 },
                HealthWarning::DiskSpaceLow { free_bytes: 1024 },
                HealthWarning::PlatformRebuilt {
                    panic: "action panicked".to_string(),
                },
                HealthWarning::AutostartFailed {
                    detail: "access denied".to_string(),
                },
                HealthWarning::RecognitionDrift {
                    window: 20,
                    ambiguous: 5,
                    close_calls: 4,
                },
                HealthWarning::SpeechUnmatched { count: 1 },
            ],
            14,
        );
        rt_variants(
            &[
                ThreadName::Controller,
                ThreadName::Capture,
                ThreadName::Processing,
                ThreadName::Dispatch,
                ThreadName::RawInput,
                ThreadName::Recorder,
                ThreadName::Build,
            ],
            7,
        );
        rt_variants(
            &[
                ActivityEntry::Fired {
                    at: at("2026-11-11T08:44:00Z"),
                    zone: zone_id(1),
                    label: "Left".to_string(),
                    origin: JobOrigin::Tap,
                    tap: Some(tap_ref()),
                    latency: Some(Milliseconds::new(53.0)),
                    result: Some(action_result()),
                    close_call: Some(zone_id(2)),
                },
                ActivityEntry::NotRecognized {
                    at: at("2026-11-11T08:44:01Z"),
                    reason: ModelRejectReason::Ambiguous,
                    best_zone: Some(zone_id(2)),
                },
            ],
            2,
        );
        rt(&LatencyReport {
            schema_version: 1,
            created_at: at("2026-11-11T09:00:00Z"),
            profile_id: profile_id(1),
            device: identity(),
            end_to_end: latency_stats(),
            decision: latency_stats(),
            controller: latency_stats(),
            dispatch: latency_stats(),
            budget_p50: LATENCY_BUDGET_P50,
            budget_p95: LATENCY_BUDGET_P95,
            passed: false,
        });

        // ---- Events ----
        let events = vec![
            Event::StateChanged {
                snapshot: Box::new(snapshot()),
            },
            Event::TapFired {
                tap: TapReport {
                    candidate: CandidateId(7),
                    zone: zone_id(1),
                    label: "Left".to_string(),
                    t0: HostTime(1_000_000),
                    latency: LatencyBreakdown {
                        decision: Milliseconds::new(51.0),
                        controller: Milliseconds::new(1.0),
                    },
                },
            },
            Event::TapRejected {
                rejected: RejectedBy::Model {
                    reason: ModelRejectReason::OutOfDistribution,
                },
                best_zone: None,
            },
            Event::GestureArmed {
                zone: zone_id(1),
                expires_at: HostTime(4_000_000),
            },
            Event::GestureCleared { zone: zone_id(1) },
            Event::ActionCompleted {
                zone: zone_id(1),
                tap: Some(tap_ref()),
                origin: JobOrigin::Tap,
                result: action_result(),
            },
            Event::ActionSkipped {
                zone: zone_id(1),
                origin: JobOrigin::TestChain,
                reason: SkipReason::TestCanceled,
            },
            Event::CommandFinished {
                zone: zone_id(1),
                origin: JobOrigin::Tap,
                step: 1,
                outcome: CommandOutcome::Exited { code: Some(3) },
            },
            Event::CalibrationFeedback {
                feedback: TapFeedback::Accepted {
                    zone: ZoneIndex(0),
                    tap: TapIndex(3),
                    count: 4,
                },
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
                event: EvaluationEvent::PromptOpened {
                    index: 0,
                    zone: ZoneIndex(1),
                },
            },
            Event::EvaluationEnded {
                outcome: EvaluationOutcome::Cancelled,
            },
            Event::ProfileMigrated {
                id: profile_id(1),
                from_version: 1,
                notes: vec![MigrationNote::HotkeyDefaulted],
            },
            Event::ConfigMigrated {
                from_version: 1,
                notes: vec![],
            },
            Event::Telemetry {
                frame: TelemetryFrame::Meter {
                    rms: Decibels::new(-42.5),
                    peak: Decibels::new(-20.0),
                    noise: TELEMETRY_DB_FLOOR,
                },
            },
            Event::EngineFaulted {
                thread: ThreadName::Processing,
                message: "processing panicked".to_string(),
            },
            Event::Stopped,
        ];
        rt_variants(&events, 18);
        rt_variants(
            &[
                RejectedBy::Model {
                    reason: ModelRejectReason::Background,
                },
                RejectedBy::Vetoed {
                    reason: VetoReason::SelfOutput {
                        kind: SelfOutputKind::Speech,
                    },
                },
                RejectedBy::Invalid {
                    error: DescriptorError::SpanTooShort {
                        needed: 4096,
                        got: 10,
                    },
                },
            ],
            3,
        );
        rt_variants(&[SkipReason::QueueFull, SkipReason::TestCanceled], 2);
        rt_variants(
            &[
                CalibrationOutcome::Committed {
                    profile: profile_id(1),
                },
                CalibrationOutcome::Cancelled,
            ],
            2,
        );
        rt_variants(
            &[
                EvaluationOutcome::Saved {
                    report: ReportId("20261111T084500000Z".to_string()),
                    overall: overall(),
                },
                EvaluationOutcome::SaveFailed {
                    report: Box::new(evaluation_report()),
                    error: failure(),
                },
                EvaluationOutcome::Cancelled,
                EvaluationOutcome::Aborted {
                    reason: MicUnavailableReason::Unplugged,
                },
            ],
            4,
        );
        let floor = || Box::new([TELEMETRY_DB_FLOOR; BAND_COUNT]);
        rt_variants(
            &[
                TelemetryFrame::Meter {
                    rms: Decibels::new(-42.5),
                    peak: Decibels::new(-20.0),
                    noise: Decibels::new(-90.0),
                },
                TelemetryFrame::Spectrum {
                    bands: floor(),
                    noise: floor(),
                    flux: 0.5,
                    threshold: 1.25,
                },
                TelemetryFrame::Waveform {
                    min: vec![-0.5, -0.25],
                    max: vec![0.5, 0.25],
                },
                TelemetryFrame::Trace {
                    entries: vec![trace_entry()],
                },
            ],
            4,
        );
        rt_variants(
            &[
                Gate::Snr,
                Gate::BroadbandFraction,
                Gate::ValidBands,
                Gate::DecayFraction,
            ],
            4,
        );
        rt_variants(
            &[GateUnit::Decibels, GateUnit::Fraction, GateUnit::Count],
            3,
        );
        rt_variants(
            &[
                TraceOutcome::DetectorRejected {
                    reason: RejectReason::LowSnr,
                },
                TraceOutcome::Detected,
                trace_entry().outcome,
            ],
            3,
        );

        // ---- Errors and lifecycle ----
        let engine_errors = vec![
            EngineError::ShuttingDown,
            EngineError::EngineStopped,
            EngineError::Timeout {
                command: CommandKind::ListDevices,
            },
            EngineError::ProtocolMismatch {
                expected: ReplyKind::Devices,
                got: ReplyKind::Ack,
            },
            EngineError::InvalidInMode {
                command: CommandKind::EditZone,
                mode: ModeKind::Calibrating,
            },
            EngineError::NoActiveProfile,
            EngineError::ProfileNotCalibrated,
            EngineError::ProfileNotFound { id: profile_id(9) },
            EngineError::ReportNotFound {
                profile: profile_id(1),
                report: ReportId("20261111T084500000Z".to_string()),
            },
            EngineError::Profile {
                error: ProfileError::EmptyName,
            },
            EngineError::Persistence { error: failure() },
            EngineError::ConfigUnreadable,
            EngineError::ConfigReadable,
            EngineError::Config {
                error: ConfigError::InvalidTextZoom { value: 105 },
            },
            EngineError::UnsupportedHotkey {
                combo: KeyCombo {
                    modifiers: vec![Modifier::Win],
                    key: "Pause".to_string(),
                },
            },
            EngineError::LegacyReport {
                report: ReportId("20250101T000000000Z".to_string()),
            },
            EngineError::Capture {
                error: CaptureError::NoDefaultDevice,
            },
            EngineError::Setup {
                error: PipelineError::TriggerCountMismatch {
                    zones: 4,
                    triggers: 2,
                },
            },
            EngineError::Calibration {
                error: SessionError::NothingToUndo,
            },
            EngineError::Evaluation {
                error: AssessmentError::NothingToUndo,
            },
            EngineError::Veto {
                reason: veto_reason(),
            },
            EngineError::VetoNotReady,
            EngineError::UnsavedReportHeld,
            EngineError::NoUnsavedReport,
            EngineError::PreviewExpired,
            EngineError::TestPending { zone: zone_id(1) },
            EngineError::NoPendingTest { zone: zone_id(1) },
            EngineError::AttentionNotFound { id: AttentionId(5) },
            EngineError::NotRecording,
            EngineError::RecordingActive,
            EngineError::Autostart {
                detail: "access denied".to_string(),
            },
            EngineError::AutostartNotManaged,
            EngineError::ThreadSpawn {
                thread: ThreadName::Build,
                detail: "out of memory".to_string(),
            },
        ];
        rt_variants(&engine_errors, 33);
        rt(&shutdown_report());
        rt_variants(&[AutostartPolicy::Reconcile, AutostartPolicy::Untouched], 2);
        rt_variants(
            &[DebugCaptureMode::FromConfig, DebugCaptureMode::ForceOn],
            2,
        );

        // ---- The serde types the snapshot embeds ----
        rt(&opened());
        rt_variants(
            &[MmcssStatus::Active, MmcssStatus::Unavailable { code: 1552 }],
            2,
        );
        rt_variants(
            &[
                CaptureStage::Spawn,
                CaptureStage::Enumerate,
                CaptureStage::Activate,
                CaptureStage::Properties,
                CaptureStage::MixFormat,
                CaptureStage::Initialize,
                CaptureStage::EventHandle,
                CaptureStage::CaptureClient,
                CaptureStage::Start,
                CaptureStage::Read,
            ],
            10,
        );
        rt_variants(&capture_errors(), 14);
        rt(&device_info());
        rt_variants(
            &[
                ProbeOutcome::NotActive,
                device_info().probe,
                ProbeOutcome::Failed {
                    error: CaptureError::AccessDenied,
                },
            ],
            3,
        );
        rt_variants(
            &[
                DeviceState::Active,
                DeviceState::Disabled,
                DeviceState::NotPresent,
                DeviceState::Unplugged,
            ],
            4,
        );
        rt_variants(&[InputDeviceClass::Keyboard, InputDeviceClass::Mouse], 2);
        rt_variants(
            &[
                veto_reason(),
                VetoUnavailableReason::RegistrationNotOwned {
                    class: InputDeviceClass::Keyboard,
                },
                VetoUnavailableReason::OwnershipLost {
                    class: InputDeviceClass::Mouse,
                },
                VetoUnavailableReason::ThreadFailed {
                    message: "RegisterClassExW failed (1410)".to_string(),
                },
            ],
            4,
        );
        rt(&tap_ref());
        rt_variants(
            &[
                JobOrigin::Tap,
                JobOrigin::TestAction { action_index: 1 },
                JobOrigin::TestChain,
            ],
            3,
        );
        rt(&ActionResult {
            started_at: HostTime(1),
            finished_at: HostTime(2),
            steps: vec![
                StepOutcome::Done,
                StepOutcome::Started,
                StepOutcome::Canceled,
                StepOutcome::Failed {
                    error: ActionError::ForegroundIsQwerty,
                },
            ],
        });
        rt_variants(
            &[
                StepOutcome::Done,
                StepOutcome::Started,
                StepOutcome::Canceled,
                StepOutcome::Failed {
                    error: ActionError::SpeechTimeout,
                },
            ],
            4,
        );
        rt_variants(
            &[
                CommandOutcome::Exited { code: None },
                CommandOutcome::StillRunning,
                CommandOutcome::WaitFailed {
                    detail: "access denied".to_string(),
                },
            ],
            3,
        );

        // ---- Goldens ----
        let snapshot_json =
            serde_json::to_string_pretty(&snapshot()).expect("the snapshot serializes");
        assert_eq!(snapshot_json, GOLDEN_SNAPSHOT, "golden Snapshot JSON");

        let action_errors = [
            (
                ActionError::TargetNotFound {
                    target: "C:/Tools/missing.exe".to_string(),
                },
                r#"{"type":"TargetNotFound","target":"C:/Tools/missing.exe"}"#,
            ),
            (
                ActionError::LaunchFailed {
                    target: "notepad.exe".to_string(),
                    reason: "access denied".to_string(),
                },
                r#"{"type":"LaunchFailed","target":"notepad.exe","reason":"access denied"}"#,
            ),
            (
                ActionError::Clipboard {
                    detail: "clipboard busy".to_string(),
                },
                r#"{"type":"Clipboard","detail":"clipboard busy"}"#,
            ),
            (
                ActionError::Speech {
                    detail: "no voice".to_string(),
                },
                r#"{"type":"Speech","detail":"no voice"}"#,
            ),
            (ActionError::SpeechTimeout, r#"{"type":"SpeechTimeout"}"#),
            (
                ActionError::Keystroke {
                    detail: "unrecognized key".to_string(),
                },
                r#"{"type":"Keystroke","detail":"unrecognized key"}"#,
            ),
            (
                ActionError::ForegroundIsQwerty,
                r#"{"type":"ForegroundIsQwerty"}"#,
            ),
            (
                ActionError::Screenshot {
                    detail: "BitBlt failed".to_string(),
                },
                r#"{"type":"Screenshot","detail":"BitBlt failed"}"#,
            ),
            (
                ActionError::Notification {
                    detail: "no AUMID".to_string(),
                },
                r#"{"type":"Notification","detail":"no AUMID"}"#,
            ),
            (
                ActionError::Sound {
                    detail: "PlaySound(SystemHand) returned FALSE".to_string(),
                },
                r#"{"type":"Sound","detail":"PlaySound(SystemHand) returned FALSE"}"#,
            ),
            (
                ActionError::Panicked {
                    message: "index out of bounds".to_string(),
                },
                r#"{"type":"Panicked","message":"index out of bounds"}"#,
            ),
        ];
        let mut kinds = BTreeSet::new();
        for (error, golden) in &action_errors {
            assert_eq!(
                serde_json::to_string(error).expect("serializes"),
                *golden,
                "golden ActionError JSON"
            );
            assert_eq!(
                &serde_json::from_str::<ActionError>(golden).expect("parses"),
                error
            );
            // kind() is the variant name, which is also the serde tag.
            assert!(golden.starts_with(&format!(r#"{{"type":"{}""#, error.kind())));
            kinds.insert(error.kind());
        }
        assert_eq!(kinds.len(), 11, "every ActionError variant has a golden");
    }

    /// Recorded from `snapshot()`; a change here is a change to the shell boundary.
    const GOLDEN_SNAPSHOT: &str = r#"{
  "revision": 7,
  "mode": {
    "type": "Listening"
  },
  "desired": {
    "type": "Listening"
  },
  "config": {
    "schema_version": 2,
    "active_profile_id": null,
    "theme": "System",
    "start_with_windows": false,
    "minimize_to_tray_on_close": true,
    "notification_verbosity": "Normal",
    "global_hotkey": {
      "modifiers": [
        "Ctrl",
        "Alt"
      ],
      "key": "Space"
    },
    "debug_capture_enabled": false,
    "input_veto_enabled": true,
    "eval_prompts_per_zone": 15,
    "eval_background_check": true,
    "dismissed_notices": [],
    "window_placement": null,
    "text_zoom": 100
  },
  "config_status": {
    "type": "Loaded"
  },
  "profiles": [
    {
      "file": "Profiles/00000000-0000-0000-0000-000000000101.json",
      "is_active": true,
      "content": {
        "type": "Readable",
        "id": "00000000-0000-0000-0000-000000000101",
        "name": "Desk",
        "zones": 2,
        "calibration": {
          "type": "Calibrated",
          "device": {
            "endpoint_id": "{0.0.1.00000000}.{mic-1}",
            "name": "Desk mic",
            "format": {
              "rate": 48000,
              "channels": 2
            },
            "capture_mode": {
              "type": "Raw"
            }
          },
          "calibrated_at": "2026-11-10T18:00:00Z"
        },
        "imported_at": null,
        "latest_report": {
          "report": "20261111T084500000Z",
          "content": {
            "type": "Readable",
            "run_at": "2026-11-11T08:45:00Z",
            "model_trained_at": "2026-11-10T18:00:00Z",
            "target": 0.92,
            "overall": {
              "prompts": 40,
              "correct": 38,
              "wrong": 0,
              "rejected": 2,
              "missed": 0,
              "accuracy": 0.95,
              "wrong_action_rate": 0.0,
              "rejection_rate": 0.05
            }
          }
        }
      }
    },
    {
      "file": "Profiles/broken.json",
      "is_active": false,
      "content": {
        "type": "Unreadable",
        "error": {
          "file": "Profiles/x.json",
          "kind": {
            "type": "Json"
          },
          "detail": "expected value at line 1 column 1"
        }
      }
    }
  ],
  "active_profile": {
    "id": "00000000-0000-0000-0000-000000000101",
    "name": "Desk",
    "zones": [
      {
        "id": "00000000-0000-0000-0000-000000000201",
        "label": "Left",
        "layout": {
          "x": 0.0,
          "y": 0.0,
          "width": 0.5,
          "height": 1.0
        },
        "trigger": {
          "type": "Single"
        },
        "actions": [
          {
            "type": "PlaySound",
            "sound": "Success"
          }
        ]
      }
    ],
    "calibration": {
      "type": "NeedsRecalibration",
      "reason": {
        "type": "NewProfile"
      }
    },
    "updated_at": "2026-11-10T18:00:00Z"
  },
  "devices": [
    {
      "endpoint_id": "{0.0.1.00000000}.{mic-1}",
      "name": "Desk mic",
      "is_default": true,
      "state": {
        "type": "Active"
      },
      "probe": {
        "type": "Probed",
        "probe": {
          "format": {
            "rate": 48000,
            "channels": 2
          },
          "raw_supported": true
        }
      }
    }
  ],
  "capture": {
    "type": "Running",
    "opened": {
      "identity": {
        "endpoint_id": "{0.0.1.00000000}.{mic-1}",
        "name": "Desk mic",
        "format": {
          "rate": 48000,
          "channels": 2
        },
        "capture_mode": {
          "type": "Raw"
        }
      },
      "period": 10.0,
      "buffer_frames": 960,
      "mmcss": {
        "type": "Active"
      }
    },
    "purpose": {
      "type": "Listening"
    }
  },
  "veto": {
    "type": "Active"
  },
  "recorder": {
    "state": {
      "type": "Off"
    },
    "stored": {
      "count": 1,
      "bytes": 2048
    }
  },
  "telemetry": {
    "type": "Off"
  },
  "health": {
    "counters": {
      "counters_since": "2026-11-11T08:00:00Z",
      "taps_fired": 3,
      "input_vetoed": 1,
      "actions_failed": 0,
      "capture_dropped_frames": 0,
      "controller_dropped": 0,
      "events_dropped": 0,
      "telemetry_dropped": 0,
      "input_dropped": 0,
      "decisions_dropped": 0,
      "dispatch_dropped": 0,
      "recorder_dropped_frames": 0,
      "recorder_dropped_events": 0,
      "stale_messages": 2,
      "lifecycle_lost": 0,
      "faults": 0
    },
    "latency": {
      "count": 2,
      "untimed": 0,
      "p50": 51.5,
      "p95": 56.0,
      "p99": 56.0,
      "max": 56.0,
      "histogram": [
        0,
        1,
        1
      ]
    },
    "warnings": [
      {
        "type": "MmcssUnavailable",
        "code": 1552
      }
    ]
  },
  "activity": {
    "recent": [
      {
        "type": "Fired",
        "at": "2026-11-11T08:44:00Z",
        "zone": "00000000-0000-0000-0000-000000000201",
        "label": "Left",
        "origin": {
          "type": "Tap"
        },
        "tap": {
          "candidate": 7,
          "t0": 1000000,
          "decided_at": 1510000,
          "fired_at": 1520000
        },
        "latency": 53.0,
        "result": {
          "started_at": 1530000,
          "finished_at": 1540000,
          "steps": [
            {
              "type": "Done"
            }
          ]
        },
        "close_call": null
      }
    ],
    "attention": [
      {
        "id": 1,
        "first_at": "2026-11-11T08:40:00Z",
        "last_at": "2026-11-11T08:41:00Z",
        "count": 2,
        "zone": "00000000-0000-0000-0000-000000000201",
        "label": "Left",
        "error": {
          "type": "TargetNotFound",
          "target": "C:/Tools/missing.exe"
        }
      }
    ]
  },
  "pending_tests": [
    {
      "zone": "00000000-0000-0000-0000-000000000201",
      "origin": {
        "type": "TestAction",
        "action_index": 0
      },
      "fires_at": 9000000
    }
  ],
  "unsaved_report": null,
  "paths": {
    "data_dir": "D:/QwertyData",
    "logs": "D:/QwertyData/Logs",
    "recordings": "D:/QwertyData/DebugCaptures"
  }
}"#;
}
