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

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::Thread;
use std::time::Duration;

use chrono::Utc;
use crossbeam_channel::{
    Receiver, RecvTimeoutError, SendTimeoutError, Sender, TryRecvError, TrySendError,
};

use qwerty_core::assessment::{
    EvaluationEvent, EvaluationReport, EvaluationSession, EvaluationStep, LatencyStats,
    PromptSchedule, ReportMeta, ZoneRef, PROMPT_WINDOW,
};
use qwerty_core::audio::{AudioBlock, AudioFormat, ChannelLayout, DeviceIdentity};
use qwerty_core::detector::{Detector, DetectorMode, DetectorParams};
use qwerty_core::gesture::{GestureEvent, GestureParams};
use qwerty_core::migrate::Loaded;
use qwerty_core::model::{ModelScratch, TrainedModel};
use qwerty_core::pipeline::{
    CalibrationFrontEnd, CalibrationInput, Incompatibility, Pipeline, PipelineConfig,
    PipelineError, TapDecision,
};
use qwerty_core::profile::{ProfileId, ZoneId};
use qwerty_core::recording::{RecordingSummary, SegmentKind, SidecarEvent, SidecarHeader};
use qwerty_core::session::{
    self, CalibrationPlan, CalibrationSession, CalibrationStep, SessionError, StepKind, TapFeedback,
};
use qwerty_core::store::{CalibrationState, Config, Profile, ZoneCount, ZoneEdit};
use qwerty_core::training::TrainingParams;
use qwerty_core::units::{CandidateId, HostDuration, HostTime, Milliseconds};
use qwerty_core::veto::{SelfOutputKind, VetoParams};

use crate::engine::messages::{
    AmbientChunk, BuildMsg, CalibrationStart, CaptureRequest, CommandMsg, DeviceMsg, DeviceWatch,
    DispatchJob, DispatchMsg, DispatchQueue, InputRecord, JobKey, OsParts, ProcessingBody,
    ProcessingCmd, ProcessingHealth, ProcessingMsg, ProcessingSetup, RawInputMsg, RawInputRequest,
    RawInputStatus, RecorderCmd, RecorderFeed, RecorderMsg, SelfOutputLink, SetupEpoch, SpeechEnds,
    ThreadEnd, ThreadExit, Worker, WorkerId,
};
use crate::engine::modes::{
    self, Activity, DeviceReadiness, ModeInputs, PipelineReadiness, ProfileReadiness,
};
use crate::engine::persist::{self, DataDir, StoredReport};
use crate::engine::protocol::{
    ActionResult, ActivityEntry, ActivityLog, AttentionId, AttentionItem, AutostartPolicy, Blocker,
    CalibrationAction, CalibrationOutcome, CalibrationRun, CalibrationStatus, CalibrationTarget,
    CaptureError, CapturePurpose, CaptureStatus, Command, CommandKind, ConfigStatus,
    DebugCaptureMode, Desired, DeviceChoice, DeviceInfo, DeviceState, EngineCounters, EngineError,
    EvaluationOutcome, Event, Health, HealthWarning, ImportPreview, JobOrigin, LatencyBreakdown,
    LatencyReport, MicUnavailableReason, Mode, ModeKind, PendingTest, PersistenceFailure,
    PreviewId, ProbeOutcome, ProfileCalibration, ProfileContent, ProfileSummary, ProfileView,
    RecorderState, RecorderStatus, RecorderStopReason, RejectedBy, Reply, ReportSummary,
    RetryState, RunSession, ShutdownReport, SkipReason, Snapshot, StepOutcome, StoredRecordings,
    TapRef, TapReport, TelemetryFrame, TelemetryLevel, ThreadName, TraceEntry, VetoStatus,
    VetoUnavailableReason, ZoneDraft, ATTENTION_CAPACITY, AUTO_ADVANCE_DELAY, INSPECTION_CAPACITY,
    LATENCY_BUDGET_P50, LATENCY_BUDGET_P95, LATENCY_MIN_TAPS, LATENCY_WINDOW, RECENT_CAPACITY,
    RETRY_SCHEDULE, SHUTDOWN_DEADLINE,
};
use crate::engine::recorder;
use crate::engine::ring::{audio_ring, RingReader, RECORDER_RING};
use crate::engine::telemetry::{trace_entry, TraceSource};
use crate::platform::OutputObserver;

// ---------------------------------------------------------------------------
// The controller (§2.8)
// ---------------------------------------------------------------------------

/// Messages handled per loop turn after the first (§2.8 step 3).
const EXTRA_MESSAGES: usize = 63;
/// The state publish rate limit (§3.3: ≤ 30 Hz).
const STATE_PUBLISH: Milliseconds = Milliseconds::new(33.0);
const EVALUATION_TICK: Milliseconds = Milliseconds::new(50.0);
const DEVICE_DEBOUNCE: Milliseconds = Milliseconds::new(200.0);
const HEALTH_INTERVAL: Milliseconds = Milliseconds::new(1000.0);
/// TapRejected and NotRecognized coalescing (PARITY-2: ≤ 10 Hz).
const REJECT_INTERVAL: Milliseconds = Milliseconds::new(100.0);
/// Trace frames at Diagnostics: ≤ 10 Hz, ≤ 32 entries.
const TRACE_INTERVAL: Milliseconds = Milliseconds::new(100.0);
const TRACE_FRAME_ENTRIES: usize = 32;
/// Telemetry from the controller goes out only while fewer events than this are queued (§2.11).
const TELEMETRY_QUEUE_LIMIT: usize = 256;
/// Each worker stop is bounded by this (and by what is left of SHUTDOWN_DEADLINE).
const WORKER_STOP: Duration = Duration::from_secs(2);
/// Lifecycle events: send_timeout(1 s) (§2.11).
const LIFECYCLE_SEND: Duration = Duration::from_secs(1);
/// RecorderCmd: send_timeout(1 s) (§2.11).
const RECORDER_SEND: Duration = Duration::from_secs(1);
/// The fresh input ring per raw-input worker (§2.7).
const INPUT_RING: usize = 1024;
/// The recorder ring's packet capacity (§2.13).
const RECORDER_PACKETS: usize = 2048;
/// LowSignal: a near-miss is a LowSnr or InsufficientBandSnr rejection with gate SNR in [3, 10) dB.
const NEAR_MISS_LOW_DB: f64 = 3.0;
const NEAR_MISS_HIGH_DB: f64 = 10.0;
const LOW_SIGNAL_WINDOW: usize = 20;
const LOW_SIGNAL_RAISE: f64 = 0.5;
const LOW_SIGNAL_CLEAR: f64 = 0.25;
/// Decisions remembered for the close call of the Fire that follows them.
const RECENT_DECISIONS: usize = 16;
/// The Run-key argument (§2.14).
const AUTOSTART_ARGS: &str = "--background";
/// LatencyReport.schema_version.
// SPEC: §2.9 gives LatencyReport a schema_version but no constant; this is its first version.
const LATENCY_REPORT_SCHEMA_VERSION: u32 = 1;

/// The receivers the controller selects over, with the senders it keeps so a channel whose
/// producers come and go never disconnects.
pub(crate) struct Channels {
    pub commands: Receiver<CommandMsg>,
    pub processing: Receiver<ProcessingMsg>,
    pub dispatch: Receiver<DispatchMsg>,
    pub raw_input: Receiver<RawInputMsg>,
    pub raw_input_tx: Sender<RawInputMsg>,
    pub recorder: Receiver<RecorderMsg>,
    pub build: Receiver<BuildMsg>,
    pub build_tx: Sender<BuildMsg>,
    pub devices: Receiver<DeviceMsg>,
    pub devices_tx: Sender<DeviceMsg>,
    pub exits: Receiver<ThreadExit>,
    pub exits_tx: Sender<ThreadExit>,
    pub events: Sender<Event>,
    pub processing_cmds: Sender<ProcessingCmd>,
    pub recorder_cmds: Sender<RecorderCmd>,
    pub feed_events: Sender<SidecarEvent>,
}

/// The workers `Engine::start_with` spawns before the controller (§2.9.2 step 5).
pub(crate) struct StartWorkers {
    pub processing: Worker,
    pub processing_thread: Thread,
    pub dispatch: Worker,
    pub queue: DispatchQueue,
    pub recorder: Worker,
    pub link: Arc<SelfOutputLink>,
    pub speech: Arc<SpeechEnds>,
    pub next_worker: u64,
}

/// What `Engine::start_with` loaded before the controller starts (§2.9.2 steps 1-4).
pub(crate) struct Loaded0 {
    pub data: DataDir,
    pub config: Config,
    pub config_status: ConfigStatus,
    pub profiles: Vec<ProfileEntry>,
    pub active: Active,
}

pub(crate) struct ProfileEntry {
    pub file: String,
    pub loaded: Result<Profile, PersistenceFailure>,
    pub latest_report: Option<ReportSummary>,
}

/// The active profile: none, a readable one, or an unreadable file (PARITY-23).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Active {
    None,
    Profile(ProfileId),
    Unreadable { file: String },
}

struct ActiveModel {
    model: Arc<TrainedModel>,
    scratch: ModelScratch,
}

struct Capture {
    worker: Worker,
    endpoint_id: String,
    opened: crate::engine::protocol::OpenedStream,
    purpose: CapturePurpose,
}

#[derive(Debug, Clone, PartialEq)]
enum DeviceAttempt {
    Opened(DeviceIdentity),
    Unavailable {
        name: String,
        reason: MicUnavailableReason,
    },
}

#[derive(Debug, Clone, PartialEq)]
enum SetupSpec {
    Calibration {
        layout: Option<ChannelLayout>,
        forward_ambient: bool,
        segment: SegmentKind,
    },
    Pipeline {
        segment: SegmentKind,
    },
    Monitor {
        segment: SegmentKind,
    },
}

impl SetupSpec {
    fn segment(&self) -> &SegmentKind {
        match self {
            SetupSpec::Calibration { segment, .. }
            | SetupSpec::Pipeline { segment }
            | SetupSpec::Monitor { segment } => segment,
        }
    }
}

enum RunState {
    Active(Box<CalibrationSession>),
    MicLost(MicUnavailableReason),
}

struct Run {
    target: CalibrationTarget,
    zones: Vec<ZoneDraft>,
    device: DeviceIdentity,
    endpoint_id: String,
    taps_per_zone: u32,
    draft: Profile,
    zone_ids: Vec<ZoneId>,
    zone_labels: Vec<String>,
    state: RunState,
    /// The layout `accept_ambient` returned, for the setup rule.
    layout: Option<ChannelLayout>,
}

struct Build {
    worker: Worker,
    cancel: Arc<AtomicBool>,
    finished: bool,
}

struct Evaluation {
    session: EvaluationSession,
    zone_refs: Vec<ZoneRef>,
    veto_enabled: bool,
    prior: Desired,
    segment: SegmentKind,
}

struct JobInfo {
    zone: ZoneId,
    label: String,
    origin: JobOrigin,
    tap: Option<TapRef>,
    entry: Option<u64>,
}

struct Recording {
    dir: Option<String>,
    /// The epoch of the first Configure under this recording (SegmentLost filter, §2.8).
    first_epoch: SetupEpoch,
    has_calibration: bool,
}

#[derive(Debug, Clone, Copy)]
struct LatencySample {
    end_to_end: f64,
    decision: f64,
    controller: f64,
    dispatch: f64,
}

pub(crate) struct Controller {
    os: Arc<dyn OsParts>,
    autostart: AutostartPolicy,
    debug_capture: DebugCaptureMode,
    ch: Channels,
    slot: Arc<Mutex<Arc<Snapshot>>>,
    data: DataDir,
    // workers
    processing: Option<Worker>,
    processing_thread: Thread,
    dispatch: Option<Worker>,
    queue: DispatchQueue,
    recorder_worker: Option<Worker>,
    link: Arc<SelfOutputLink>,
    speech: Arc<SpeechEnds>,
    capture: Option<Capture>,
    raw_input: Option<Worker>,
    build: Option<Build>,
    build_lost: Arc<AtomicU64>,
    device_watch: Option<DeviceWatch>,
    devices_dropped: Arc<AtomicU64>,
    next_worker: u64,
    epoch: SetupEpoch,
    next_job: u64,
    next_entry: u64,
    next_attention: u64,
    next_preview: u64,
    // app state
    config: Config,
    config_status: ConfigStatus,
    profiles: Vec<ProfileEntry>,
    active: Active,
    model: Option<ActiveModel>,
    desired: Desired,
    telemetry: TelemetryLevel,
    mode: Mode,
    fault: Option<(ThreadName, String)>,
    starting: bool,
    shell_ready: bool,
    devices: Vec<DeviceInfo>,
    device_attempt: Option<(String, DeviceAttempt)>,
    pipeline_failed: Option<(ProfileId, PipelineError)>,
    capture_status: CaptureStatus,
    retry: Option<(String, u8, HostTime)>,
    veto: VetoStatus,
    raw_input_skipped: u64,
    recorder_state: RecorderState,
    stored: StoredRecordings,
    recording: Option<Recording>,
    last_summary: Option<RecordingSummary>,
    configured: Option<SetupSpec>,
    pending_ring: Option<RingReader>,
    pending_calibration_start: Option<CalibrationStart>,
    segment_override: Option<SegmentKind>,
    run: Option<Run>,
    /// The command (StartCalibration or Calibration) whose handler is installing a setup; a
    /// calibration setup that fails then is returned to it as EngineError::Setup.
    setup_caller: Option<CommandKind>,
    /// The setup error that ended the run while `setup_caller` was set.
    run_setup_error: Option<PipelineError>,
    auto_advance_at: Option<HostTime>,
    negatives_at: Option<HostTime>,
    evaluation: Option<Evaluation>,
    unsaved_report: Option<Box<EvaluationReport>>,
    preview: Option<(PreviewId, Profile)>,
    pending_tests: Vec<PendingTest>,
    jobs: HashMap<JobKey, JobInfo>,
    recent_decisions: Vec<(CandidateId, qwerty_core::model::ModelScores)>,
    counters: EngineCounters,
    latency: std::collections::VecDeque<LatencySample>,
    latency_stats: LatencyStats,
    /// Live decisions the recorder feed refused (part of recorder_dropped_events).
    decision_feed_dropped: u64,
    /// The workers a fault stopped, reported again by the Shutdown that follows.
    fault_stops: Option<ShutdownReport>,
    recent: VecDeque<(u64, ActivityEntry)>,
    attention: Vec<AttentionItem>,
    inspection: VecDeque<TraceEntry>,
    trace_pending: Vec<TraceEntry>,
    last_trace_frame: HostTime,
    last_reject: HostTime,
    /// The first setup epoch configured after the current model was loaded.
    model_since: SetupEpoch,
    /// The newest rejection inside the current 100 ms window, emitted when it ends.
    pending_reject: Option<(RejectedBy, Option<ZoneId>)>,
    warnings: Vec<HealthWarning>,
    health: Option<ProcessingHealth>,
    drift: DriftWindow,
    near_misses: VecDeque<bool>,
    speech_unmatched_seen: u64,
    capture_dropped_seen: u64,
    events_dropped_seen: u64,
    controller_dropped_seen: u64,
    decisions_dropped_seen: u64,
    /// Processing's lifetime totals already folded into the counters (ProcessingDrops).
    processing_messages_seen: u64,
    processing_telemetry_seen: u64,
    // timers
    eval_tick_at: Option<HostTime>,
    debounce_at: Option<HostTime>,
    health_at: HostTime,
    last_state_sent: Option<HostTime>,
    state_pending: bool,
    revision: u64,
    shutting_down: bool,
    exit: bool,
}

/// Stops a worker mid-run. A thread that misses the limit is detached by `Worker::stop`; its
/// exit still arrives on the exits channel when it ends, so only the timeout is logged here.
fn stop_worker(worker: Worker) {
    if let Err(timed_out) = worker.stop(WORKER_STOP) {
        tracing::warn!(
            thread = ?timed_out.thread,
            worker = timed_out.worker.0,
            "a worker missed its stop limit and was detached"
        );
    }
}

fn after(t: HostTime, ms: Milliseconds) -> HostTime {
    HostTime(t.0.saturating_add(HostDuration::from_ms(ms).0))
}

fn ms_between(later: HostTime, earlier: HostTime) -> f64 {
    later.since(earlier).as_ms().get()
}

/// The `Milliseconds` of a finite span (every engine span is finite).
fn ms(value: f64) -> Milliseconds {
    // INVARIANT: the value is a difference of two finite host times in ms, so it is finite.
    Milliseconds::try_new(value).expect("a span between host times is finite")
}

fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

/// The reason a capture error makes the microphone unavailable.
fn mic_reason(error: &CaptureError) -> MicUnavailableReason {
    match error {
        CaptureError::DeviceNotFound { .. } => MicUnavailableReason::NotPresent,
        CaptureError::DeviceNotActive { state, .. } => match state {
            DeviceState::Disabled => MicUnavailableReason::Disabled,
            DeviceState::Unplugged => MicUnavailableReason::Unplugged,
            DeviceState::NotPresent => MicUnavailableReason::NotPresent,
            DeviceState::Active => MicUnavailableReason::OpenFailed {
                error: error.clone(),
            },
        },
        CaptureError::AccessDenied => MicUnavailableReason::AccessDenied,
        CaptureError::InUseExclusive => MicUnavailableReason::InUseExclusive,
        CaptureError::Stalled { .. } => MicUnavailableReason::Stalled,
        CaptureError::DeviceInvalidated => MicUnavailableReason::Unplugged,
        CaptureError::ComInit { .. }
        | CaptureError::NoDefaultDevice
        | CaptureError::UnsupportedMixFormat { .. }
        | CaptureError::UnsupportedSampleRate { .. }
        | CaptureError::UnsupportedChannelCount { .. }
        | CaptureError::OpenTimeout
        | CaptureError::Os { .. }
        | CaptureError::Unexpected { .. } => MicUnavailableReason::OpenFailed {
            error: error.clone(),
        },
    }
}

fn is_lifecycle(event: &Event) -> bool {
    match event {
        Event::CalibrationEnded { .. }
        | Event::EvaluationEnded { .. }
        | Event::EngineFaulted { .. }
        | Event::Stopped => true,
        Event::StateChanged { .. }
        | Event::TapFired { .. }
        | Event::TapRejected { .. }
        | Event::GestureArmed { .. }
        | Event::GestureCleared { .. }
        | Event::ActionCompleted { .. }
        | Event::ActionSkipped { .. }
        | Event::CommandFinished { .. }
        | Event::CalibrationFeedback { .. }
        | Event::CalibrationProgress { .. }
        | Event::EvaluationUpdate { .. }
        | Event::ProfileMigrated { .. }
        | Event::ConfigMigrated { .. }
        | Event::Telemetry { .. } => false,
    }
}

/// The segment a calibration step records under (the setup rule, §2.8).
fn step_segment(step: &CalibrationStep, ids: &[ZoneId], labels: &[String]) -> SegmentKind {
    match step {
        CalibrationStep::Ambient { .. } => SegmentKind::Ambient,
        CalibrationStep::Zone { index, .. } => {
            let i = usize::from(index.0);
            SegmentKind::CalibrationZone {
                zone: ids[i],
                label: labels[i].clone(),
            }
        }
        CalibrationStep::Negatives { .. } => SegmentKind::CalibrationNegatives,
        CalibrationStep::Ready
        | CalibrationStep::AmbientReview { .. }
        | CalibrationStep::ReadyToBuild
        | CalibrationStep::Building { .. }
        | CalibrationStep::Review { .. }
        | CalibrationStep::Failed { .. } => SegmentKind::Idle,
    }
}

/// pipeline::CalibrationInput → the session's own copy (see messages.rs).
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

fn profile_calibration_status(profile: &Profile) -> CalibrationStatus {
    match &profile.calibration {
        CalibrationState::Calibrated {
            device,
            model,
            calibrated_at,
            environment,
        } => CalibrationStatus::Calibrated {
            device: device.clone(),
            calibrated_at: *calibrated_at,
            environment: *environment,
            summary: Box::new(model.summary.clone()),
        },
        CalibrationState::NeedsRecalibration { reason } => CalibrationStatus::NeedsRecalibration {
            reason: reason.clone(),
        },
    }
}

fn profile_file(id: ProfileId) -> String {
    format!("{}.json", id.as_uuid())
}

impl Controller {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        os: Arc<dyn OsParts>,
        autostart: AutostartPolicy,
        debug_capture: DebugCaptureMode,
        ch: Channels,
        slot: Arc<Mutex<Arc<Snapshot>>>,
        loaded: Loaded0,
        workers: StartWorkers,
    ) -> Controller {
        let now = os.now();
        let stored = match recorder::stored_recordings(&loaded.data.paths().recordings) {
            Ok(stored) => stored,
            Err(e) => {
                tracing::error!(code = e.raw_os_error(), "measuring DebugCaptures failed");
                StoredRecordings { count: 0, bytes: 0 }
            }
        };
        let mut controller = Controller {
            os,
            autostart,
            debug_capture,
            ch,
            slot,
            data: loaded.data,
            processing: Some(workers.processing),
            processing_thread: workers.processing_thread,
            dispatch: Some(workers.dispatch),
            queue: workers.queue,
            recorder_worker: Some(workers.recorder),
            link: workers.link,
            speech: workers.speech,
            capture: None,
            raw_input: None,
            build: None,
            build_lost: Arc::new(AtomicU64::new(0)),
            device_watch: None,
            devices_dropped: Arc::new(AtomicU64::new(0)),
            next_worker: workers.next_worker,
            epoch: SetupEpoch(0),
            next_job: 0,
            next_entry: 0,
            next_attention: 0,
            next_preview: 0,
            config: loaded.config,
            config_status: loaded.config_status,
            profiles: loaded.profiles,
            active: loaded.active,
            model: None,
            desired: Desired::Listening,
            telemetry: TelemetryLevel::Off,
            mode: Mode::Starting,
            fault: None,
            starting: true,
            shell_ready: false,
            devices: Vec::new(),
            device_attempt: None,
            pipeline_failed: None,
            capture_status: CaptureStatus::Off,
            retry: None,
            veto: VetoStatus::Disabled,
            raw_input_skipped: 0,
            recorder_state: RecorderState::Off,
            stored,
            recording: None,
            last_summary: None,
            configured: None,
            pending_ring: None,
            pending_calibration_start: None,
            segment_override: None,
            run: None,
            setup_caller: None,
            run_setup_error: None,
            auto_advance_at: None,
            negatives_at: None,
            evaluation: None,
            unsaved_report: None,
            preview: None,
            pending_tests: Vec::new(),
            jobs: HashMap::new(),
            recent_decisions: Vec::new(),
            counters: EngineCounters {
                counters_since: Utc::now(),
                taps_fired: 0,
                input_vetoed: 0,
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
                stale_messages: 0,
                lifecycle_lost: 0,
                faults: 0,
            },
            latency: VecDeque::new(),
            latency_stats: LatencyStats::from_samples(&[], 0),
            decision_feed_dropped: 0,
            fault_stops: None,
            recent: VecDeque::new(),
            attention: Vec::new(),
            inspection: VecDeque::new(),
            trace_pending: Vec::new(),
            last_trace_frame: now,
            last_reject: HostTime(i64::MIN / 2),
            model_since: SetupEpoch(0),
            pending_reject: None,
            warnings: Vec::new(),
            health: None,
            drift: DriftWindow::new(),
            near_misses: VecDeque::new(),
            speech_unmatched_seen: 0,
            capture_dropped_seen: 0,
            events_dropped_seen: 0,
            controller_dropped_seen: 0,
            decisions_dropped_seen: 0,
            processing_messages_seen: 0,
            processing_telemetry_seen: 0,
            eval_tick_at: None,
            debounce_at: None,
            health_at: after(now, HEALTH_INTERVAL),
            last_state_sent: None,
            state_pending: false,
            revision: 0,
            shutting_down: false,
            exit: false,
        };
        controller.load_active_model();
        controller
    }

    /// §2.9.2 steps 5-8 on the controller thread: the device watch, the device list, autostart,
    /// reconcile. Returns the first error.
    pub(crate) fn init(&mut self) -> Result<(), EngineError> {
        match self.os.watch_devices(
            self.ch.devices_tx.clone(),
            Arc::clone(&self.devices_dropped),
        ) {
            Ok(watch) => self.device_watch = Some(watch),
            Err(error) => return Err(EngineError::Capture { error }),
        }
        self.refresh_devices();
        if self.autostart == AutostartPolicy::Reconcile {
            if let Err(detail) = self.reconcile_autostart(self.config.start_with_windows) {
                tracing::warn!("autostart reconcile failed at startup");
                drop(detail);
            }
        }
        self.starting = false;
        self.reconcile();
        self.publish(true);
        Ok(())
    }

    fn now(&self) -> HostTime {
        self.os.now()
    }

    fn next_worker_id(&mut self) -> WorkerId {
        self.next_worker += 1;
        WorkerId(self.next_worker)
    }

    // -----------------------------------------------------------------------
    // The loop (§2.8)
    // -----------------------------------------------------------------------

    pub(crate) fn run(&mut self) {
        while !self.exit {
            let deadline = self.next_deadline();
            let timeout = self.os.timeout_until(deadline);
            self.wait_one(timeout);
            for _ in 0..EXTRA_MESSAGES {
                if self.exit || !self.try_one() {
                    break;
                }
            }
            if self.exit {
                break;
            }
            self.run_timers();
            self.publish(false);
        }
    }

    fn wait_one(&mut self, timeout: Duration) {
        let ch = &self.ch;
        crossbeam_channel::select! {
            recv(ch.commands) -> m => match m {
                Ok(m) => self.on_command(m),
                Err(_) => self.on_commands_closed(),
            },
            recv(ch.exits) -> m => if let Ok(m) = m { self.on_exit(m) },
            recv(ch.processing) -> m => match m {
                Ok(m) => self.on_processing(m),
                Err(_) => self.ch.processing = crossbeam_channel::never(),
            },
            recv(ch.dispatch) -> m => match m {
                Ok(m) => self.on_dispatch(m),
                Err(_) => self.ch.dispatch = crossbeam_channel::never(),
            },
            recv(ch.raw_input) -> m => if let Ok(m) = m { self.on_raw_input(m) },
            recv(ch.recorder) -> m => match m {
                Ok(m) => self.on_recorder(m),
                Err(_) => self.ch.recorder = crossbeam_channel::never(),
            },
            recv(ch.build) -> m => if let Ok(m) = m { self.on_build(m) },
            recv(ch.devices) -> m => if let Ok(m) = m { self.on_device(m) },
            default(timeout) => {},
        }
    }

    /// One already-queued message without blocking; false when every channel is empty.
    fn try_one(&mut self) -> bool {
        match self.ch.commands.try_recv() {
            Ok(m) => {
                self.on_command(m);
                return true;
            }
            Err(TryRecvError::Disconnected) => {
                self.on_commands_closed();
                return true;
            }
            Err(TryRecvError::Empty) => {}
        }
        if let Ok(m) = self.ch.exits.try_recv() {
            self.on_exit(m);
            return true;
        }
        if let Ok(m) = self.ch.processing.try_recv() {
            self.on_processing(m);
            return true;
        }
        if let Ok(m) = self.ch.dispatch.try_recv() {
            self.on_dispatch(m);
            return true;
        }
        if let Ok(m) = self.ch.raw_input.try_recv() {
            self.on_raw_input(m);
            return true;
        }
        if let Ok(m) = self.ch.recorder.try_recv() {
            self.on_recorder(m);
            return true;
        }
        if let Ok(m) = self.ch.build.try_recv() {
            self.on_build(m);
            return true;
        }
        if let Ok(m) = self.ch.devices.try_recv() {
            self.on_device(m);
            return true;
        }
        false
    }

    /// Every handle is gone without a Shutdown: shut down anyway, with nobody to reply to.
    fn on_commands_closed(&mut self) {
        self.ch.commands = crossbeam_channel::never();
        let report = self.shutdown_sequence();
        drop(report);
        self.exit = true;
    }

    fn next_deadline(&self) -> HostTime {
        let mut deadline = self.health_at;
        let mut consider = |t: Option<HostTime>| {
            if let Some(t) = t {
                if t < deadline {
                    deadline = t;
                }
            }
        };
        consider(self.eval_tick_at);
        consider(self.auto_advance_at);
        consider(self.negatives_at);
        consider(self.debounce_at);
        consider(self.retry.as_ref().map(|r| r.2));
        consider(self.pending_tests.iter().map(|t| t.fires_at).min());
        if self.state_pending {
            consider(Some(self.publish_due()));
        }
        if !self.trace_pending.is_empty() {
            consider(Some(after(self.last_trace_frame, TRACE_INTERVAL)));
        }
        if self.pending_reject.is_some() {
            consider(Some(after(self.last_reject, REJECT_INTERVAL)));
        }
        deadline
    }

    fn publish_due(&self) -> HostTime {
        match self.last_state_sent {
            Some(t) => after(t, STATE_PUBLISH),
            None => HostTime(i64::MIN / 2),
        }
    }

    fn run_timers(&mut self) {
        let now = self.now();
        if self.fault.is_some() {
            // Faulted (§2.11): no timer acts any more, but every deadline must move on or
            // next_deadline stays in the past and the loop spins on a zero timeout.
            self.eval_tick_at = None;
            self.auto_advance_at = None;
            self.negatives_at = None;
            self.debounce_at = None;
            self.retry = None;
            self.pending_tests.clear();
            self.pending_reject = None;
            if !self.trace_pending.is_empty() && now >= after(self.last_trace_frame, TRACE_INTERVAL)
            {
                self.flush_trace(now);
            }
            if now >= self.health_at {
                self.health_at = after(now, HEALTH_INTERVAL);
            }
            return;
        }
        if let Some(t) = self.eval_tick_at {
            if now >= t {
                self.eval_tick_at = Some(after(now, EVALUATION_TICK));
                let mut events = Vec::new();
                if let Some(eval) = self.evaluation.as_mut() {
                    eval.session.tick(now, &mut events);
                }
                self.on_evaluation_events(events);
            }
        }
        if let Some(t) = self.auto_advance_at {
            if now >= t {
                self.auto_advance_at = None;
                let result = self.with_session(|s, now| s.advance(now));
                if let Err(e) = result {
                    tracing::debug!(error = ?std::mem::discriminant(&e), "auto-advance refused");
                }
            }
        }
        if let Some(t) = self.negatives_at {
            if now >= t {
                self.negatives_at = None;
                let result = self.with_session(|s, _| s.finish_negatives());
                if result.is_err() {
                    tracing::debug!("negatives limit: not in the Negatives step");
                }
            }
        }
        if let Some(t) = self.debounce_at {
            if now >= t {
                self.debounce_at = None;
                self.refresh_devices();
                self.retry_if_active();
                self.reconcile();
            }
        }
        if let Some((endpoint, attempt, at)) = self.retry.clone() {
            if now >= at {
                self.retry_open(endpoint, attempt);
            }
        }
        let due: Vec<PendingTest> = self
            .pending_tests
            .iter()
            .filter(|t| t.fires_at <= now)
            .copied()
            .collect();
        if !due.is_empty() {
            self.pending_tests.retain(|t| t.fires_at > now);
            for test in due {
                self.fire_test(test);
            }
        }
        if !self.trace_pending.is_empty() && now >= after(self.last_trace_frame, TRACE_INTERVAL) {
            self.flush_trace(now);
        }
        if now >= after(self.last_reject, REJECT_INTERVAL) {
            if let Some((rejected, best_zone)) = self.pending_reject.take() {
                self.emit_rejected(rejected, best_zone);
            }
        }
        if now >= self.health_at {
            self.health_at = after(now, HEALTH_INTERVAL);
            self.aggregate_health();
        }
    }

    // -----------------------------------------------------------------------
    // Events and the snapshot
    // -----------------------------------------------------------------------

    fn emit(&mut self, event: Event) {
        if is_lifecycle(&event) {
            match self.ch.events.send_timeout(event, LIFECYCLE_SEND) {
                Ok(()) => {}
                Err(SendTimeoutError::Timeout(_)) | Err(SendTimeoutError::Disconnected(_)) => {
                    self.counters.lifecycle_lost += 1;
                }
            }
            return;
        }
        match self.ch.events.try_send(event) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => self.counters.events_dropped += 1,
            // The shell dropped its stream; nothing is listening.
            Err(TrySendError::Disconnected(_)) => {}
        }
    }

    /// Rebuilds the snapshot; on a change bumps the revision and stores it; sends StateChanged
    /// at ≤ 30 Hz (§2.8 steps 5-6).
    fn publish(&mut self, force: bool) {
        self.mode = self.resolve_mode();
        let mut snapshot = self.build_snapshot();
        let changed = {
            let current = lock_slot(&self.slot);
            snapshot.revision = current.revision;
            **current != snapshot
        };
        if changed || force {
            self.revision += 1;
            snapshot.revision = self.revision;
            *lock_slot(&self.slot) = Arc::new(snapshot);
            self.state_pending = true;
        }
        if self.state_pending && self.now() >= self.publish_due() {
            let snapshot = Box::new((**lock_slot(&self.slot)).clone());
            match self.ch.events.try_send(Event::StateChanged { snapshot }) {
                Ok(()) => {
                    self.state_pending = false;
                    self.last_state_sent = Some(self.now());
                }
                Err(TrySendError::Full(_)) => {
                    // Try again one publish interval later, not on every loop pass.
                    self.counters.events_dropped += 1;
                    self.last_state_sent = Some(self.now());
                }
                Err(TrySendError::Disconnected(_)) => self.state_pending = false,
            }
        }
    }

    fn build_snapshot(&self) -> Snapshot {
        let paths = self.data.paths();
        Snapshot {
            revision: self.revision,
            mode: self.mode.clone(),
            desired: self.desired,
            config: self.config.clone(),
            config_status: self.config_status.clone(),
            profiles: self.profile_summaries(),
            active_profile: self.active_profile().map(|p| ProfileView {
                id: p.id,
                name: p.name.clone(),
                zones: p.zones.clone(),
                calibration: profile_calibration_status(p),
                updated_at: p.updated_at,
            }),
            devices: self.devices.clone(),
            capture: self.capture_status.clone(),
            veto: self.veto.clone(),
            recorder: RecorderStatus {
                state: self.recorder_state.clone(),
                stored: self.stored,
            },
            telemetry: self.telemetry,
            health: Health {
                counters: self.counters.clone(),
                latency: self.latency_stats.clone(),
                warnings: self.warnings.clone(),
            },
            activity: ActivityLog {
                recent: self.recent.iter().map(|(_, e)| e.clone()).collect(),
                attention: self.attention.clone(),
            },
            pending_tests: self.pending_tests.clone(),
            unsaved_report: self.unsaved_report.as_ref().map(|r| ReportSummary {
                report: crate::engine::protocol::ReportId(
                    r.run_at.format("%Y%m%dT%H%M%S%3fZ").to_string(),
                ),
                content: crate::engine::protocol::ReportContent::Readable {
                    run_at: r.run_at,
                    model_trained_at: r.meta.model_trained_at,
                    target: r.target,
                    overall: r.overall.clone(),
                },
            }),
            paths,
        }
    }

    fn profile_summaries(&self) -> Vec<ProfileSummary> {
        let mut out: Vec<ProfileSummary> = self
            .profiles
            .iter()
            .map(|e| {
                let content = match &e.loaded {
                    Ok(p) => ProfileContent::Readable {
                        id: p.id,
                        name: p.name.clone(),
                        // INVARIANT: a loaded profile passed validate(), so its zone count is allowed.
                        zones: ZoneCount::new(p.zones.len() as u8)
                            .expect("a validated profile has an allowed zone count"),
                        calibration: match &p.calibration {
                            CalibrationState::Calibrated {
                                device,
                                calibrated_at,
                                ..
                            } => ProfileCalibration::Calibrated {
                                device: device.identity.clone(),
                                calibrated_at: *calibrated_at,
                            },
                            CalibrationState::NeedsRecalibration { reason } => {
                                ProfileCalibration::NeedsRecalibration {
                                    reason: reason.clone(),
                                }
                            }
                        },
                        imported_at: p.imported_at,
                        latest_report: e.latest_report.clone(),
                    },
                    Err(error) => ProfileContent::Unreadable {
                        error: error.clone(),
                    },
                };
                let is_active = match (&self.active, &e.loaded) {
                    (Active::Profile(id), Ok(p)) => p.id == *id,
                    (Active::Unreadable { file }, Err(_)) => *file == e.file,
                    (Active::None, _)
                    | (Active::Profile(_), Err(_))
                    | (Active::Unreadable { .. }, Ok(_)) => false,
                };
                ProfileSummary {
                    file: e.file.clone(),
                    is_active,
                    content,
                }
            })
            .collect();
        out.sort_by(|a, b| a.file.cmp(&b.file));
        out
    }

    fn resolve_mode(&self) -> Mode {
        let fault = self.fault.as_ref().map(|(t, m)| (*t, m.as_str()));
        let run_view = self.run.as_ref().map(|r| self.run_view(r));
        let eval_view = self.evaluation.as_ref().map(|e| e.session.view());
        let activity = match (&run_view, &eval_view) {
            (Some(run), _) => Activity::Calibrating(run),
            (None, Some(view)) => Activity::Evaluating(view),
            (None, None) => Activity::None,
        };
        let unreadable = match &self.active {
            Active::Unreadable { file } => self
                .profiles
                .iter()
                .find(|e| e.file == *file)
                .and_then(|e| e.loaded.as_ref().err().map(|err| (file.as_str(), err))),
            Active::None | Active::Profile(_) => None,
        };
        let profile = match (self.active_profile(), unreadable) {
            (_, Some((file, error))) => ProfileReadiness::Unreadable { file, error },
            (Some(p), None) => match &p.calibration {
                CalibrationState::Calibrated { device, .. } => ProfileReadiness::Calibrated {
                    id: p.id,
                    bound: device,
                },
                CalibrationState::NeedsRecalibration { reason } => {
                    ProfileReadiness::NeedsRecalibration { id: p.id, reason }
                }
            },
            (None, None) => ProfileReadiness::None,
        };
        let device = match (self.bound_endpoint(), &self.device_attempt) {
            (Some(bound), Some((endpoint, attempt))) if bound == *endpoint => match attempt {
                DeviceAttempt::Opened(identity) => DeviceReadiness::Opened { identity },
                DeviceAttempt::Unavailable { name, reason } => DeviceReadiness::Unavailable {
                    endpoint_id: endpoint,
                    name,
                    reason,
                },
            },
            (Some(_), Some(_)) | (Some(_), None) | (None, _) => DeviceReadiness::NotAttempted,
        };
        let pipeline = match (&self.pipeline_failed, self.active_id()) {
            (Some((id, error)), Some(active)) if *id == active => {
                PipelineReadiness::Failed { error }
            }
            (Some(_), _) | (None, _) => PipelineReadiness::NotBuilt,
        };

        modes::resolve(&ModeInputs {
            fault,
            starting: self.starting,
            activity,
            profile,
            desired: self.desired,
            device,
            pipeline,
            veto: &self.veto,
            veto_enabled: self.config.input_veto_enabled,
        })
    }

    fn run_view(&self, run: &Run) -> CalibrationRun {
        CalibrationRun {
            target: run.target.clone(),
            zones: run.zones.clone(),
            device: run.device.clone(),
            taps_per_zone: run.taps_per_zone,
            session: match &run.state {
                RunState::Active(s) => RunSession::Active {
                    view: Box::new(s.view()),
                },
                RunState::MicLost(reason) => RunSession::MicLost {
                    reason: reason.clone(),
                },
            },
        }
    }

    // -----------------------------------------------------------------------
    // Profiles
    // -----------------------------------------------------------------------

    fn active_id(&self) -> Option<ProfileId> {
        match &self.active {
            Active::Profile(id) => Some(*id),
            Active::None | Active::Unreadable { .. } => None,
        }
    }

    fn profile(&self, id: ProfileId) -> Option<&Profile> {
        self.profiles
            .iter()
            .find_map(|e| e.loaded.as_ref().ok().filter(|p| p.id == id))
    }

    fn active_profile(&self) -> Option<&Profile> {
        self.active_id().and_then(|id| self.profile(id))
    }

    fn bound_endpoint(&self) -> Option<String> {
        match &self.active_profile()?.calibration {
            CalibrationState::Calibrated { device, .. } => {
                Some(device.identity.endpoint_id.clone())
            }
            CalibrationState::NeedsRecalibration { .. } => None,
        }
    }

    fn load_active_model(&mut self) {
        self.model = match self.active_profile().map(|p| &p.calibration) {
            Some(CalibrationState::Calibrated { model, .. }) => {
                let model = Arc::new(model.clone());
                let scratch = model.scratch();
                Some(ActiveModel { model, scratch })
            }
            Some(CalibrationState::NeedsRecalibration { .. }) | None => None,
        };
        // Setups configured from now on run this model.
        self.model_since = SetupEpoch(self.epoch.0 + 1);
        self.reset_drift();
        self.pipeline_failed = None;
    }

    /// Forgets the drift window and clears its warning (§2.8: a profile switch, a committed
    /// recalibration or a capture reopen). Once reset, `push` cannot report `Cleared`, so the
    /// warning must go here.
    fn reset_drift(&mut self) {
        self.drift.reset();
        self.warnings
            .retain(|w| !matches!(w, HealthWarning::RecognitionDrift { .. }));
    }

    fn upsert_profile(&mut self, profile: Profile) {
        let file = profile_file(profile.id);
        match self.profiles.iter_mut().find(|e| e.file == file) {
            Some(entry) => entry.loaded = Ok(profile),
            None => self.profiles.push(ProfileEntry {
                file,
                loaded: Ok(profile),
                latest_report: None,
            }),
        }
    }

    fn refresh_latest_report(&mut self, id: ProfileId) {
        let latest = match self.data.list_reports(id) {
            Ok(reports) => reports.into_iter().find(|r| {
                matches!(
                    r.content,
                    crate::engine::protocol::ReportContent::Readable { .. }
                )
            }),
            Err(failure) => {
                tracing::warn!(file = %failure.file, kind = ?failure.kind, "listing reports failed");
                None
            }
        };
        let file = profile_file(id);
        if let Some(entry) = self.profiles.iter_mut().find(|e| e.file == file) {
            entry.latest_report = latest;
        }
    }

    fn save_config(&mut self, config: Config) -> Result<(), EngineError> {
        self.data
            .save_config(&config)
            .map_err(|error| EngineError::Persistence { error })?;
        self.config = config;
        Ok(())
    }

    fn refresh_devices(&mut self) {
        match self.os.list_devices() {
            Ok(devices) => self.devices = devices,
            Err(error) => {
                tracing::warn!(error = ?std::mem::discriminant(&error), "device listing failed");
            }
        }
    }

    fn reconcile_autostart(&mut self, enabled: bool) -> Result<(), String> {
        let result = crate::platform::reconcile_start_with_windows(enabled, AUTOSTART_ARGS);
        self.warnings
            .retain(|w| !matches!(w, HealthWarning::AutostartFailed { .. }));
        match result {
            Ok(_) => Ok(()),
            Err(detail) => {
                self.warnings.push(HealthWarning::AutostartFailed {
                    detail: detail.clone(),
                });
                Err(detail)
            }
        }
    }
}

fn lock_slot(slot: &Mutex<Arc<Snapshot>>) -> std::sync::MutexGuard<'_, Arc<Snapshot>> {
    match slot.lock() {
        Ok(guard) => guard,
        // A panic while holding the slot left a whole Arc in it; the snapshot is still valid.
        Err(poisoned) => poisoned.into_inner(),
    }
}

impl Controller {
    // -----------------------------------------------------------------------
    // Reconcile (§2.8)
    // -----------------------------------------------------------------------

    fn reconcile(&mut self) {
        if self.fault.is_some()
            || self.shutting_down
            || self.starting && self.device_watch.is_none()
        {
            return;
        }
        self.reconcile_veto();
        self.mode = self.resolve_mode();
        match self.capture_target() {
            None => {
                self.stop_capture();
                if !matches!(self.capture_status, CaptureStatus::Failed { .. })
                    || self.retry.is_none()
                {
                    self.capture_status = CaptureStatus::Off;
                    self.retry = None;
                }
            }
            Some((endpoint, purpose)) => {
                let running = self.capture.as_ref().map(|c| c.endpoint_id.clone());
                if running.as_deref() == Some(endpoint.as_str()) {
                    if let Some(c) = self.capture.as_mut() {
                        c.purpose = purpose;
                        self.capture_status = CaptureStatus::Running {
                            opened: c.opened.clone(),
                            purpose,
                        };
                        // A capture kept across a change of purpose (a Commit binds the profile
                        // to the stream the wizard ran on) is this endpoint's attempt.
                        self.device_attempt = Some((
                            endpoint.clone(),
                            DeviceAttempt::Opened(c.opened.identity.clone()),
                        ));
                    }
                } else {
                    let failed_here = matches!(&self.capture_status,
                        CaptureStatus::Failed { endpoint_id, .. } if *endpoint_id == endpoint);
                    if !failed_here {
                        if self.open(&endpoint, purpose).is_err() {
                            // fail_capture put the error and the retry schedule in the status.
                            tracing::warn!("the capture did not open; its retry schedule runs");
                        }
                    } else {
                        self.stop_capture();
                    }
                }
            }
        }
        self.mode = self.resolve_mode();
        self.ensure_setup(false);
    }

    /// The target capture (§2.8): device and purpose.
    fn capture_target(&self) -> Option<(String, CapturePurpose)> {
        if let Some(run) = &self.run {
            return match run.state {
                RunState::Active(_) => Some((run.endpoint_id.clone(), CapturePurpose::Calibrating)),
                RunState::MicLost(_) => None,
            };
        }
        let bound = self.bound_endpoint();
        if self.evaluation.is_some() {
            return bound.map(|b| (b, CapturePurpose::Evaluating));
        }
        let diagnostics = self.telemetry == TelemetryLevel::Diagnostics;
        let default = || {
            self.devices
                .iter()
                .find(|d| d.is_default)
                .map(|d| d.endpoint_id.clone())
        };
        match &self.mode {
            Mode::Listening | Mode::Starting => match bound {
                Some(b) if self.desired == Desired::Listening => {
                    Some((b, CapturePurpose::Listening))
                }
                Some(b) if diagnostics => Some((b, CapturePurpose::Diagnostics)),
                Some(_) | None => None,
            },
            Mode::Blocked {
                blocker: Blocker::MicUnavailable { .. },
            } => bound.map(|b| (b, CapturePurpose::Listening)),
            Mode::Blocked {
                blocker: Blocker::MicMismatch { .. } | Blocker::ProfileIncompatible { .. },
            } => None,
            Mode::Paused | Mode::Blocked { .. } | Mode::Idle if diagnostics => match bound {
                Some(b) => Some((b, CapturePurpose::Diagnostics)),
                None => default().map(|d| (d, CapturePurpose::Diagnostics)),
            },
            Mode::Paused
            | Mode::Blocked { .. }
            | Mode::Idle
            | Mode::Calibrating { .. }
            | Mode::Evaluating { .. }
            | Mode::Faulted { .. } => None,
        }
    }

    /// Opens `endpoint` (stopping a different running capture first). On success: the recorder
    /// starts if recording is on, and the next Configure carries the ring.
    fn open(&mut self, endpoint: &str, purpose: CapturePurpose) -> Result<(), CaptureError> {
        self.stop_capture();
        let worker = self.next_worker_id();
        let request = CaptureRequest {
            worker,
            endpoint_id: endpoint.to_string(),
            wake: self.processing_thread.clone(),
            exits: self.ch.exits_tx.clone(),
        };
        match self.os.open_capture(request) {
            Ok((worker, opened)) => {
                let identity = opened.stream.identity.clone();
                self.capture_status = CaptureStatus::Running {
                    opened: opened.stream.clone(),
                    purpose,
                };
                self.capture = Some(Capture {
                    worker,
                    endpoint_id: endpoint.to_string(),
                    opened: opened.stream.clone(),
                    purpose,
                });
                self.retry = None;
                self.device_attempt = Some((
                    endpoint.to_string(),
                    DeviceAttempt::Opened(identity.clone()),
                ));
                self.reset_drift();
                self.pending_ring = Some(opened.ring);
                self.configured = None;
                if self.recording_on() {
                    self.start_recording();
                }
                let mismatch = purpose != CapturePurpose::Calibrating
                    && self.bound_endpoint().as_deref() == Some(endpoint)
                    && self.active_profile().is_some_and(|p| match &p.calibration {
                        CalibrationState::Calibrated { device, .. } => {
                            !device.identity.differences(&identity).is_empty()
                        }
                        CalibrationState::NeedsRecalibration { .. } => false,
                    });
                if mismatch {
                    self.stop_capture();
                    self.capture_status = CaptureStatus::Off;
                }
                Ok(())
            }
            Err(error) => {
                self.fail_capture(endpoint, error.clone(), 1);
                Err(error)
            }
        }
    }

    fn fail_capture(&mut self, endpoint: &str, error: CaptureError, attempt: u8) {
        let name = self
            .devices
            .iter()
            .find(|d| d.endpoint_id == endpoint)
            .map(|d| d.name.clone())
            .unwrap_or_else(|| endpoint.to_string());
        self.device_attempt = Some((
            endpoint.to_string(),
            DeviceAttempt::Unavailable {
                name,
                reason: mic_reason(&error),
            },
        ));
        let retry = match RETRY_SCHEDULE.get(usize::from(attempt) - 1) {
            Some(delay) => {
                let at = after(self.now(), *delay);
                self.retry = Some((endpoint.to_string(), attempt, at));
                RetryState::Scheduled { attempt, at }
            }
            None => {
                self.retry = None;
                RetryState::Exhausted
            }
        };
        self.capture_status = CaptureStatus::Failed {
            endpoint_id: endpoint.to_string(),
            error,
            retry,
        };
    }

    fn retry_open(&mut self, endpoint: String, attempt: u8) {
        self.retry = None;
        let Some((target, purpose)) = self.capture_target_ignoring_failure() else {
            self.capture_status = CaptureStatus::Off;
            return;
        };
        if target != endpoint {
            self.reconcile();
            return;
        }
        if let Err(error) = self.open(&endpoint, purpose) {
            self.fail_capture(&endpoint, error, attempt + 1);
        }
        self.reconcile();
    }

    /// A device message made the failed endpoint Active again: try at once (§2.8 Retry).
    fn retry_if_active(&mut self) {
        let CaptureStatus::Failed { endpoint_id, .. } = &self.capture_status else {
            return;
        };
        let active = self
            .devices
            .iter()
            .any(|d| d.endpoint_id == *endpoint_id && d.state == DeviceState::Active);
        if active {
            let endpoint = endpoint_id.clone();
            self.retry_open(endpoint, 0);
        }
    }

    fn capture_target_ignoring_failure(&mut self) -> Option<(String, CapturePurpose)> {
        self.mode = self.resolve_mode();
        self.capture_target()
    }

    fn stop_capture(&mut self) {
        let Some(capture) = self.capture.take() else {
            return;
        };
        if let Err(timed_out) = capture.worker.stop(WORKER_STOP) {
            tracing::error!(
                thread = timed_out.thread.os_name(),
                "capture stop timed out"
            );
        }
        self.stop_recording(RecorderStopReason::CaptureStopped);
        self.configure_idle();
        if matches!(self.capture_status, CaptureStatus::Running { .. }) {
            self.capture_status = CaptureStatus::Off;
        }
    }

    // -----------------------------------------------------------------------
    // Processing setups (§2.5, §2.8 setup rule)
    // -----------------------------------------------------------------------

    fn desired_setup(&self) -> Option<SetupSpec> {
        let capture = self.capture.as_ref()?;
        let spec = match capture.purpose {
            CapturePurpose::Calibrating => {
                let run = self.run.as_ref()?;
                let RunState::Active(session) = &run.state else {
                    return None;
                };
                let view = session.view();
                SetupSpec::Calibration {
                    layout: run.layout.clone(),
                    forward_ambient: matches!(view.step, CalibrationStep::Ambient { .. }),
                    segment: step_segment(&view.step, &run.zone_ids, &run.zone_labels),
                }
            }
            CapturePurpose::Evaluating => SetupSpec::Pipeline {
                segment: self
                    .evaluation
                    .as_ref()
                    .map(|e| e.segment.clone())
                    .unwrap_or(SegmentKind::EvaluationPrompts),
            },
            CapturePurpose::Listening | CapturePurpose::Diagnostics => {
                let segment = self.segment_override.clone();
                if self.model.is_some()
                    && self.bound_endpoint().as_deref() == Some(&capture.endpoint_id)
                {
                    SetupSpec::Pipeline {
                        segment: segment.unwrap_or(SegmentKind::Listening),
                    }
                } else {
                    // SPEC: §2.8 names no segment for a Monitor setup; Idle (no tap activity).
                    SetupSpec::Monitor {
                        segment: segment.unwrap_or(SegmentKind::Idle),
                    }
                }
            }
        };
        Some(spec)
    }

    /// Sends a Configure when the desired setup differs from the installed one (or `force`).
    fn ensure_setup(&mut self, force: bool) {
        if self.fault.is_some() {
            return;
        }
        let desired = self.desired_setup();
        if !force && desired == self.configured && self.pending_ring.is_none() {
            return;
        }
        let Some(spec) = desired else {
            if self.configured.is_some() {
                self.configure_idle();
            }
            return;
        };
        match self.build_setup(&spec) {
            Ok(setup) => {
                self.epoch = SetupEpoch(self.epoch.0 + 1);
                let calibration_started = self.pending_calibration_start.take();
                if calibration_started.is_some() {
                    if let Some(rec) = self.recording.as_mut() {
                        rec.has_calibration = true;
                    }
                }
                let cmd = ProcessingCmd::Configure {
                    epoch: self.epoch,
                    setup,
                    segment: spec.segment().clone(),
                    calibration_started,
                    ring: self.pending_ring.take(),
                };
                self.send_processing(cmd);
                if let Some(rec) = self.recording.as_mut() {
                    if rec.first_epoch.0 == 0 {
                        rec.first_epoch = self.epoch;
                    }
                }
                self.configured = Some(spec);
            }
            Err(error) => self.on_setup_error(error),
        }
    }

    fn configure_idle(&mut self) {
        self.epoch = SetupEpoch(self.epoch.0 + 1);
        self.pending_ring = None;
        self.send_processing(ProcessingCmd::Configure {
            epoch: self.epoch,
            setup: ProcessingSetup::Idle,
            segment: SegmentKind::Idle,
            calibration_started: None,
            ring: None,
        });
        self.configured = None;
    }

    fn build_setup(&self, spec: &SetupSpec) -> Result<ProcessingSetup, PipelineError> {
        // INVARIANT: desired_setup returns a spec only while a capture runs.
        let capture = self
            .capture
            .as_ref()
            .expect("a setup is built only while capture runs");
        let format = capture.opened.identity.format;
        let all_real = ChannelLayout::all_real(format.channels);
        match spec {
            SetupSpec::Calibration {
                layout,
                forward_ambient,
                ..
            } => {
                let layout = layout.as_ref().unwrap_or(&all_real);
                let front_end = CalibrationFrontEnd::new(format, layout, &DetectorParams::DEFAULT)
                    .map_err(|error| PipelineError::Detector { error })?;
                Ok(ProcessingSetup::Calibration {
                    front_end: Box::new(front_end),
                    forward_ambient: *forward_ambient,
                })
            }
            SetupSpec::Pipeline { .. } => {
                // INVARIANT: a Pipeline spec requires the active model and a Calibrated profile.
                let model = self.model.as_ref().expect("a Pipeline setup has a model");
                let profile = self
                    .active_profile()
                    .expect("a Pipeline setup has an active profile");
                let CalibrationState::Calibrated { device, .. } = &profile.calibration else {
                    unreachable!("a model comes only from a Calibrated profile");
                };
                let triggers = model
                    .model
                    .zone_ids
                    .iter()
                    .map(|id| {
                        profile
                            .zones
                            .iter()
                            .find(|z| z.id == *id)
                            .map(|z| z.trigger)
                            // INVARIANT: Profile::validate (run on every load and save) checks
                            // that every model zone id names one of the profile's zones.
                            .expect("a validated profile has every model zone")
                    })
                    .collect();
                let pipeline = Pipeline::new(
                    Arc::clone(&model.model),
                    PipelineConfig {
                        format,
                        layout: device.layout.clone(),
                        triggers,
                        detector: DetectorParams::DEFAULT,
                        veto: VetoParams::DEFAULT,
                        gesture: GestureParams::DEFAULT,
                    },
                )?;
                Ok(ProcessingSetup::Pipeline {
                    pipeline: Box::new(pipeline),
                })
            }
            SetupSpec::Monitor { .. } => {
                let detector = Detector::new(
                    format,
                    &all_real,
                    &DetectorParams::DEFAULT,
                    DetectorMode::Live,
                )
                .map_err(|error| PipelineError::Detector { error })?;
                Ok(ProcessingSetup::Monitor {
                    detector: Box::new(detector),
                })
            }
        }
    }

    /// RUST-RT-17: a setup that cannot be built is routed by cause.
    fn on_setup_error(&mut self, error: PipelineError) {
        tracing::error!(error = ?std::mem::discriminant(&error), "a processing setup could not be built");
        if let PipelineError::ModelIncompatible {
            reason: Incompatibility::FrontEndVersion { .. },
        } = &error
        {
            self.enter_fault(
                ThreadName::Controller,
                "front-end version incompatibility reached construction".to_string(),
            );
            return;
        }
        if self.run.is_some() {
            // The calibration front end failed: the run cannot continue, and leaving it in place
            // would starve it of input with nothing shown. StartCalibration returns
            // EngineError::Setup (§2.8 Reconcile); a later failure ends the run.
            self.counters.faults += 1;
            self.end_run();
            // A run StartCalibration never returned has nothing to end for the shell.
            if self.setup_caller != Some(CommandKind::StartCalibration) {
                self.emit(Event::CalibrationEnded {
                    outcome: CalibrationOutcome::Cancelled,
                });
            }
            if self.setup_caller.is_some() {
                self.run_setup_error = Some(error);
            }
            self.reconcile();
            return;
        }
        if let Some(id) = self.active_id() {
            self.pipeline_failed = Some((id, error));
            if self.evaluation.take().is_some() {
                self.eval_tick_at = None;
            }
            self.stop_capture();
        }
    }

    fn send_processing(&mut self, cmd: ProcessingCmd) {
        if self.ch.processing_cmds.send(cmd).is_err() {
            tracing::error!("processing is gone; a command was not delivered");
        }
    }

    // -----------------------------------------------------------------------
    // The input veto (§2.7)
    // -----------------------------------------------------------------------

    fn veto_needed(&self) -> bool {
        self.run.is_some()
            || self.evaluation.is_some()
            || self
                .active_profile()
                .is_some_and(|p| matches!(p.calibration, CalibrationState::Calibrated { .. }))
    }

    fn reconcile_veto(&mut self) {
        let wanted = self.config.input_veto_enabled && self.veto_needed();
        if !self.config.input_veto_enabled {
            self.stop_raw_input();
            self.veto = VetoStatus::Disabled;
            return;
        }
        if !self.veto_needed() {
            self.stop_raw_input();
            self.veto = VetoStatus::NotNeeded;
            return;
        }
        if !self.shell_ready {
            self.veto = VetoStatus::WaitingForShell;
            return;
        }
        if wanted
            && self.raw_input.is_none()
            && !matches!(self.veto, VetoStatus::Unavailable { .. })
        {
            self.spawn_raw_input();
        }
    }

    fn spawn_raw_input(&mut self) {
        let (producer, consumer) = rtrb::RingBuffer::<InputRecord>::new(INPUT_RING);
        self.send_processing(ProcessingCmd::SetInputRing {
            inputs: Some(consumer),
        });
        let worker = self.next_worker_id();
        let request = RawInputRequest {
            worker,
            ring: producer,
            tx: self.ch.raw_input_tx.clone(),
            exits: self.ch.exits_tx.clone(),
        };
        match self.os.spawn_raw_input(request) {
            Ok(w) => {
                self.raw_input = Some(w);
                self.veto = VetoStatus::Registering;
            }
            Err(reason) => {
                self.send_processing(ProcessingCmd::SetInputRing { inputs: None });
                self.veto = VetoStatus::Unavailable { reason };
            }
        }
    }

    fn stop_raw_input(&mut self) {
        if let Some(worker) = self.raw_input.take() {
            if let Err(t) = worker.stop(WORKER_STOP) {
                tracing::error!(thread = t.thread.os_name(), "raw-input stop timed out");
            }
            self.send_processing(ProcessingCmd::SetInputRing { inputs: None });
        }
    }

    fn on_raw_input(&mut self, msg: RawInputMsg) {
        if self.raw_input.as_ref().map(|w| w.id()) != Some(msg.worker) {
            self.counters.stale_messages += 1;
            return;
        }
        if msg.skipped_records > self.raw_input_skipped {
            let increase = msg.skipped_records - self.raw_input_skipped;
            self.counters.faults += increase;
            tracing::error!(count = increase, "raw-input records skipped");
            self.raw_input_skipped = msg.skipped_records;
        }
        match msg.status {
            RawInputStatus::Active => self.veto = VetoStatus::Active,
            RawInputStatus::Unavailable { reason } => {
                self.stop_raw_input();
                self.veto = VetoStatus::Unavailable { reason };
            }
        }
        self.reconcile();
    }

    // -----------------------------------------------------------------------
    // The recorder (§2.13)
    // -----------------------------------------------------------------------

    fn recording_on(&self) -> bool {
        self.debug_capture == DebugCaptureMode::ForceOn || self.config.debug_capture_enabled
    }

    fn start_recording(&mut self) {
        let Some(capture) = self.capture.as_ref() else {
            return;
        };
        let format = capture.opened.identity.format;
        let (writer, reader) = audio_ring(
            format,
            RECORDER_RING,
            RECORDER_PACKETS,
            capture.opened.buffer_frames,
        );
        let header = SidecarHeader {
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            device: capture.opened.identity.clone(),
            layout: self.active_profile().and_then(|p| match &p.calibration {
                CalibrationState::Calibrated { device, .. }
                    if capture.purpose != CapturePurpose::Calibrating =>
                {
                    Some(device.layout.clone())
                }
                CalibrationState::Calibrated { .. }
                | CalibrationState::NeedsRecalibration { .. } => None,
            }),
            profile_id: self.active_id(),
            created_at: Utc::now(),
        };
        let cmd = RecorderCmd::Start {
            root: self.data.paths().recordings,
            header: Box::new(header),
            ring: reader,
        };
        if !self.send_recorder(cmd) {
            return;
        }
        self.recording = Some(Recording {
            dir: None,
            first_epoch: SetupEpoch(0),
            has_calibration: false,
        });
        self.send_processing(ProcessingCmd::SetRecorder {
            feed: Some(RecorderFeed {
                ring: writer,
                events: self.ch.feed_events.clone(),
            }),
        });
        self.configured = None;
    }

    fn stop_recording(&mut self, reason: RecorderStopReason) {
        if self.recording.take().is_none() {
            return;
        }
        self.send_processing(ProcessingCmd::SetRecorder { feed: None });
        if !self.send_recorder(RecorderCmd::Stop { reason }) {
            tracing::error!("the recorder did not take its stop command");
        }
    }

    /// send_timeout(1 s); false when the recorder could not take it.
    fn send_recorder(&mut self, cmd: RecorderCmd) -> bool {
        match self.ch.recorder_cmds.send_timeout(cmd, RECORDER_SEND) {
            Ok(()) => true,
            Err(SendTimeoutError::Timeout(_)) | Err(SendTimeoutError::Disconnected(_)) => false,
        }
    }

    /// A controller structure event (Prompt): a lost one stops the recording (§2.13).
    fn send_structure(&mut self, event: SidecarEvent) {
        if self.recording.is_none() {
            return;
        }
        if !self.send_recorder(RecorderCmd::Event { event }) {
            self.stop_recording(RecorderStopReason::StructureEventLost);
        }
    }

    fn on_recorder(&mut self, msg: RecorderMsg) {
        match msg {
            RecorderMsg::Started { dir } => {
                if let Some(rec) = self.recording.as_mut() {
                    rec.dir = Some(dir.clone());
                }
                self.recorder_state = RecorderState::Recording {
                    dir,
                    bytes: 0,
                    duration: qwerty_core::units::Seconds::new(0.0),
                    gaps: 0,
                    dropped_events: 0,
                };
            }
            RecorderMsg::Progress {
                bytes,
                duration,
                gaps,
                dropped_events,
            } => {
                if let RecorderState::Recording { dir, .. } = &self.recorder_state {
                    self.recorder_state = RecorderState::Recording {
                        dir: dir.clone(),
                        bytes,
                        duration,
                        gaps,
                        dropped_events,
                    };
                }
            }
            RecorderMsg::Stopped { summary, reason } => {
                let dir = match &self.recorder_state {
                    RecorderState::Recording { dir, .. } | RecorderState::Stopped { dir, .. } => {
                        dir.clone()
                    }
                    RecorderState::Off | RecorderState::Failed { .. } => String::new(),
                };
                if let RecorderStopReason::DiskSpaceLow { free_bytes } = reason {
                    self.warnings
                        .retain(|w| !matches!(w, HealthWarning::DiskSpaceLow { .. }));
                    self.warnings
                        .push(HealthWarning::DiskSpaceLow { free_bytes });
                }
                // The recorder stopped on its own (size, disk): the engine stops feeding it.
                if self.recording.is_some() && !matches!(reason, RecorderStopReason::CaptureStopped)
                {
                    self.recording = None;
                    self.send_processing(ProcessingCmd::SetRecorder { feed: None });
                }
                self.last_summary = Some(summary);
                self.recorder_state = RecorderState::Stopped { dir, reason };
            }
            RecorderMsg::Failed { error } => {
                self.recording = None;
                self.send_processing(ProcessingCmd::SetRecorder { feed: None });
                self.recorder_state = RecorderState::Failed { error };
            }
            RecorderMsg::Stored { stored } => self.stored = stored,
        }
    }

    // -----------------------------------------------------------------------
    // Devices and thread exits
    // -----------------------------------------------------------------------

    fn on_device(&mut self, msg: DeviceMsg) {
        drop(msg);
        self.debounce_at = Some(after(self.now(), DEVICE_DEBOUNCE));
    }

    fn on_exit(&mut self, exit: ThreadExit) {
        let current = match exit.thread {
            ThreadName::Capture => self.capture.as_ref().map(|c| c.worker.id()),
            ThreadName::RawInput => self.raw_input.as_ref().map(|w| w.id()),
            ThreadName::Build => self.build.as_ref().map(|b| b.worker.id()),
            ThreadName::Processing => self.processing.as_ref().map(|w| w.id()),
            ThreadName::Dispatch => self.dispatch.as_ref().map(|w| w.id()),
            ThreadName::Recorder => self.recorder_worker.as_ref().map(|w| w.id()),
            ThreadName::Controller => None,
        };
        if current != Some(exit.worker) {
            self.counters.stale_messages += 1;
            return;
        }
        if let ThreadEnd::Panicked { message } = &exit.end {
            self.enter_fault(exit.thread, message.clone());
            return;
        }
        match (exit.thread, exit.end) {
            (ThreadName::Capture, ThreadEnd::Capture { exit: capture_exit }) => {
                self.counters.capture_dropped_frames += capture_exit.dropped_frames;
                let error = match capture_exit.error {
                    Some(error) => error,
                    None => CaptureError::Unexpected {
                        stage: crate::engine::protocol::CaptureStage::Read,
                        detail: "the capture thread ended without an error".to_string(),
                    },
                };
                self.on_capture_lost(error);
            }
            (ThreadName::Build, ThreadEnd::Finished | ThreadEnd::Capture { .. }) => {
                // The build thread sends its result before it exits, but on another channel,
                // and the loop does not read channels in send order: handle anything still
                // queued first, so only a result the thread failed to send counts as lost.
                while let Ok(m) = self.ch.build.try_recv() {
                    self.on_build(m);
                }
                let lost = self.build.as_ref().is_some_and(|b| !b.finished);
                if let Some(build) = self.build.take() {
                    stop_worker(build.worker);
                }
                if lost {
                    // The result never arrived (send_timeout expired): the build failed.
                    self.counters.lifecycle_lost += self.build_lost.swap(0, Ordering::AcqRel);
                    self.finish_build(Err(SessionError::BuildPanicked {
                        message: "the build result was lost".to_string(),
                    }));
                }
            }
            (
                ThreadName::RawInput,
                ThreadEnd::Finished | ThreadEnd::Capture { .. } | ThreadEnd::Panicked { .. },
            ) => {
                self.raw_input = None;
                self.send_processing(ProcessingCmd::SetInputRing { inputs: None });
                if !matches!(self.veto, VetoStatus::Unavailable { .. }) {
                    self.veto = VetoStatus::Unavailable {
                        reason: VetoUnavailableReason::ThreadFailed {
                            message: "the raw-input thread ended".to_string(),
                        },
                    };
                }
                self.reconcile();
            }
            (thread @ ThreadName::Capture, ThreadEnd::Finished)
            | (
                thread @ (ThreadName::Processing
                | ThreadName::Dispatch
                | ThreadName::Recorder
                | ThreadName::Controller),
                ThreadEnd::Finished | ThreadEnd::Capture { .. },
            ) => {
                self.enter_fault(thread, "the thread ended unexpectedly".to_string());
            }
            // Panicked was handled above, before the match.
            (thread, ThreadEnd::Panicked { message }) => self.enter_fault(thread, message),
        }
    }

    /// The current capture worker ended with an error (§2.8 Retry; calibration MicLost;
    /// evaluation Aborted).
    fn on_capture_lost(&mut self, error: CaptureError) {
        let Some(capture) = self.capture.take() else {
            return;
        };
        stop_worker(capture.worker);
        self.stop_recording(RecorderStopReason::CaptureStopped);
        self.configure_idle();
        let reason = mic_reason(&error);
        self.fail_capture(&capture.endpoint_id, error, 1);
        if let Some(run) = self.run.as_mut() {
            run.state = RunState::MicLost(reason.clone());
            self.auto_advance_at = None;
            self.negatives_at = None;
            if let Some(build) = self.build.take() {
                build.cancel.store(true, Ordering::Release);
                stop_worker(build.worker);
            }
            self.retry = None;
        }
        if let Some(eval) = self.evaluation.take() {
            self.eval_tick_at = None;
            self.desired = eval.prior;
            self.emit(Event::EvaluationEnded {
                outcome: EvaluationOutcome::Aborted { reason },
            });
        }
        self.reconcile();
    }
}

impl Controller {
    // -----------------------------------------------------------------------
    // Processing messages
    // -----------------------------------------------------------------------

    fn on_processing(&mut self, msg: ProcessingMsg) {
        let current = msg.epoch == self.epoch;
        // Health and SegmentLost apply whatever their epoch; anything else from an old setup is
        // stale.
        let any_epoch = matches!(
            msg.body,
            ProcessingBody::Health { .. } | ProcessingBody::SegmentLost
        );
        if !current && !any_epoch {
            // A decision the old setup made before processing applied the new Configure is
            // still a live outcome of the recorded stream: replay reproduces it up to the
            // segment boundary (§2.13, RP2), so it is recorded although nothing acts on it.
            // Only from a setup that ran the current model (so its zone maps to the same id)
            // and that the recording covers.
            let recorded = self
                .recording
                .as_ref()
                .is_some_and(|r| r.first_epoch.0 != 0 && msg.epoch >= r.first_epoch);
            if recorded && msg.epoch >= self.model_since {
                if let ProcessingBody::Decision { decision } = &msg.body {
                    self.record_decision(decision);
                }
            }
            self.counters.stale_messages += 1;
            return;
        }
        match msg.body {
            ProcessingBody::Health { health } => self.on_health(health),
            ProcessingBody::SegmentLost => {
                let applies = self
                    .recording
                    .as_ref()
                    .is_some_and(|r| r.first_epoch.0 != 0 && msg.epoch >= r.first_epoch);
                if applies {
                    self.stop_recording(RecorderStopReason::StructureEventLost);
                } else {
                    self.counters.stale_messages += 1;
                }
            }
            ProcessingBody::Decision { decision } => self.on_decision(*decision),
            ProcessingBody::Gesture { event, at } => self.on_gesture(event, at),
            ProcessingBody::DetectorRejected { rejected, at } => {
                self.note_near_miss(&rejected);
                self.log_trace(
                    at,
                    TraceSource::DetectorRejected {
                        rejected: &rejected,
                    },
                );
                if self.evaluation.is_some() {
                    let mut events = Vec::new();
                    if let Some(eval) = self.evaluation.as_mut() {
                        eval.session
                            .on_detector_rejected(&rejected, at, &mut events);
                    }
                    self.on_evaluation_events(events);
                }
            }
            ProcessingBody::Detected {
                candidate: _,
                onset,
                gates,
                at,
            } => {
                self.note_detector_outcome(false);
                self.log_trace(
                    at,
                    TraceSource::Detected {
                        onset,
                        gates: &gates,
                    },
                );
            }
            ProcessingBody::Calibration { input } => self.on_calibration_input(input),
            ProcessingBody::SetupFailed { error } => self.on_setup_error(error),
            ProcessingBody::AmbientChunk { chunk } => self.on_ambient(chunk),
            ProcessingBody::Fault { fault } => {
                // An InputFault carries sample positions only, never audio.
                tracing::error!(?fault, "the processing input faulted");
                self.counters.faults += 1;
            }
        }
    }

    fn log_trace(&mut self, at: HostTime, source: TraceSource<'_>) {
        let model = self.model.as_mut().map(|m| (&*m.model, &mut m.scratch));
        let (entry, error) = trace_entry(at, source, model);
        if error.is_some() {
            tracing::error!("map_point failed on a validated model");
            self.counters.faults += 1;
        }
        if self.inspection.len() == INSPECTION_CAPACITY {
            self.inspection.pop_front();
        }
        self.inspection.push_back(entry.clone());
        if self.telemetry == TelemetryLevel::Diagnostics {
            self.trace_pending.push(entry);
        }
    }

    fn flush_trace(&mut self, now: HostTime) {
        self.last_trace_frame = now;
        let mut entries = std::mem::take(&mut self.trace_pending);
        if entries.len() > TRACE_FRAME_ENTRIES {
            let drop_count = entries.len() - TRACE_FRAME_ENTRIES;
            entries.drain(..drop_count);
            self.counters.telemetry_dropped += drop_count as u64;
        }
        if self.ch.events.len() >= TELEMETRY_QUEUE_LIMIT {
            self.counters.telemetry_dropped += entries.len() as u64;
            return;
        }
        match self.ch.events.try_send(Event::Telemetry {
            frame: TelemetryFrame::Trace { entries },
        }) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.counters.telemetry_dropped += 1;
            }
        }
    }

    fn on_decision(&mut self, decision: TapDecision) {
        self.record_decision(&decision);
        self.note_detector_outcome(false);
        self.log_trace(
            decision.decided_at,
            TraceSource::Decision {
                decision: &decision,
            },
        );
        if let Some(scores) = decision.scores() {
            if self.recent_decisions.len() == RECENT_DECISIONS {
                self.recent_decisions.remove(0);
            }
            self.recent_decisions
                .push((decision.candidate, scores.clone()));
        }
        if self.evaluation.is_some() {
            let mut events = Vec::new();
            if let Some(eval) = self.evaluation.as_mut() {
                eval.session.on_decision(&decision, &mut events);
            }
            self.on_evaluation_events(events);
            return;
        }
        if self.mode != Mode::Listening {
            return;
        }
        let tau = self.model.as_ref().map(|m| m.model.thresholds.tau_margin);
        if let Some(tau) = tau {
            if let Some(change) = self.drift.push(&decision.outcome, tau) {
                self.warnings
                    .retain(|w| !matches!(w, HealthWarning::RecognitionDrift { .. }));
                match change {
                    DriftChange::Raised {
                        window,
                        ambiguous,
                        close_calls,
                    } => self.warnings.push(HealthWarning::RecognitionDrift {
                        window,
                        ambiguous,
                        close_calls,
                    }),
                    DriftChange::Cleared => {}
                }
            }
        }
        let rejected = match &decision.outcome {
            DecisionOutcome::Accepted { .. } => None,
            DecisionOutcome::Vetoed { reason, .. } => {
                if matches!(reason, qwerty_core::veto::VetoReason::Input { .. }) {
                    self.counters.input_vetoed += 1;
                }
                Some(RejectedBy::Vetoed { reason: *reason })
            }
            DecisionOutcome::Rejected { reason, .. } => Some(RejectedBy::Model { reason: *reason }),
            DecisionOutcome::Invalid { error } => Some(RejectedBy::Invalid {
                error: error.clone(),
            }),
        };
        let Some(rejected) = rejected else {
            return;
        };
        let best_zone = decision.scores().and_then(|s| self.zone_id(s.best));
        // p35 §3.3: ≤ 10 Hz, only the latest in each 100 ms window. Inside the window the newest
        // rejection replaces the held one; the timer emits it when the window ends.
        if self.now() < after(self.last_reject, REJECT_INTERVAL) {
            self.pending_reject = Some((rejected, best_zone));
            return;
        }
        self.emit_rejected(rejected, best_zone);
    }

    /// Emits one coalesced `TapRejected` (and its `NotRecognized` entry) and opens a new window.
    fn emit_rejected(&mut self, rejected: RejectedBy, best_zone: Option<ZoneId>) {
        self.last_reject = self.now();
        self.pending_reject = None;
        if let RejectedBy::Model { reason } = &rejected {
            self.push_activity(ActivityEntry::NotRecognized {
                at: Utc::now(),
                reason: *reason,
                best_zone,
            });
        }
        self.emit(Event::TapRejected {
            rejected,
            best_zone,
        });
    }

    fn zone_id(&self, zone: qwerty_core::units::ZoneIndex) -> Option<ZoneId> {
        self.model
            .as_ref()
            .and_then(|m| m.model.zone_ids.get(usize::from(zone.0)).copied())
    }

    // SPEC: §2.13 has processing send `Decision{onset, accepted_zone}` through the feed, but
    // `ProcessingSetup::Pipeline` (WP-C0's type) carries no zone ids, so processing cannot name
    // the accepted zone. The controller, which holds the model's `zone_ids`, sends it instead,
    // on the same feed and with the same `try_send` and drop count.
    /// The live outcome of a decision, for live-vs-replay equality (RP2).
    fn record_decision(&mut self, decision: &TapDecision) {
        if self.recording.is_none() {
            return;
        }
        let accepted_zone = match decision.outcome {
            DecisionOutcome::Accepted { zone, .. } => self.zone_id(zone),
            DecisionOutcome::Vetoed { .. }
            | DecisionOutcome::Rejected { .. }
            | DecisionOutcome::Invalid { .. } => None,
        };
        let event = SidecarEvent::Decision {
            onset: decision.onset,
            accepted_zone,
        };
        if self.ch.feed_events.try_send(event).is_err() {
            self.decision_feed_dropped += 1;
            self.counters.recorder_dropped_events += 1;
        }
    }

    fn zone_info(&self, zone: ZoneId) -> Option<(String, Vec<qwerty_core::profile::ActionSpec>)> {
        self.active_profile()?
            .zones
            .iter()
            .find(|z| z.id == zone)
            .map(|z| (z.label.clone(), z.actions.clone()))
    }

    fn on_gesture(&mut self, event: GestureEvent, at: HostTime) {
        if self.mode != Mode::Listening || self.evaluation.is_some() {
            return;
        }
        match event {
            GestureEvent::Fire {
                zone,
                candidate,
                t0,
                ..
            } => self.fire_tap(zone, candidate, t0, at),
            GestureEvent::Armed {
                zone, expires_at, ..
            } => {
                if let Some(zone) = self.zone_id(zone) {
                    self.emit(Event::GestureArmed { zone, expires_at });
                }
            }
            GestureEvent::Disarmed { zone, .. } | GestureEvent::DoubleExpired { zone } => {
                if let Some(zone) = self.zone_id(zone) {
                    self.emit(Event::GestureCleared { zone });
                }
            }
            // Suppressions and the burst hold change no visible state.
            GestureEvent::Suppressed { .. }
            | GestureEvent::BurstHeld { .. }
            | GestureEvent::BurstReleased { .. } => {}
        }
    }

    fn fire_tap(
        &mut self,
        zone: qwerty_core::units::ZoneIndex,
        candidate: CandidateId,
        t0: HostTime,
        decided_at: HostTime,
    ) {
        let Some(zone_id) = self.zone_id(zone) else {
            tracing::error!("a Fire named a zone outside the model");
            self.counters.faults += 1;
            return;
        };
        let Some((label, actions)) = self.zone_info(zone_id) else {
            return;
        };
        let fired_at = self.now();
        let tap = TapRef {
            candidate,
            t0,
            decided_at,
            fired_at,
        };
        self.counters.taps_fired += 1;
        self.emit(Event::TapFired {
            tap: TapReport {
                candidate,
                zone: zone_id,
                label: label.clone(),
                t0,
                latency: LatencyBreakdown {
                    decision: ms(ms_between(decided_at, t0)),
                    controller: ms(ms_between(fired_at, decided_at)),
                },
            },
        });
        let tau = self.model.as_ref().map(|m| m.model.thresholds.tau_margin);
        let close = self
            .recent_decisions
            .iter()
            .rev()
            .find(|(c, _)| *c == candidate)
            .and_then(|(_, scores)| tau.and_then(|tau| close_call(scores, tau)))
            .and_then(|z| self.zone_id(z));
        let entry = self.push_activity(ActivityEntry::Fired {
            at: Utc::now(),
            zone: zone_id,
            label: label.clone(),
            origin: JobOrigin::Tap,
            tap: Some(tap),
            latency: None,
            result: None,
            close_call: close,
        });
        self.send_job(
            zone_id,
            label,
            actions,
            JobOrigin::Tap,
            Some(tap),
            Some(entry),
            fired_at,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn send_job(
        &mut self,
        zone: ZoneId,
        label: String,
        actions: Vec<qwerty_core::profile::ActionSpec>,
        origin: JobOrigin,
        tap: Option<TapRef>,
        entry: Option<u64>,
        fired_at: HostTime,
    ) {
        self.next_job += 1;
        let key = JobKey(self.next_job);
        let info = JobInfo {
            zone,
            label,
            origin,
            tap,
            entry,
        };
        if actions.is_empty() {
            self.jobs.insert(key, info);
            self.on_completed(
                key,
                ActionResult {
                    started_at: fired_at,
                    finished_at: fired_at,
                    steps: Vec::new(),
                },
            );
            return;
        }
        let job = DispatchJob {
            key,
            zone,
            actions,
            origin,
            tap,
        };
        match self.queue.try_send(job) {
            Ok(()) => {
                self.jobs.insert(key, info);
            }
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.counters.dispatch_dropped += 1;
                self.emit(Event::ActionSkipped {
                    zone,
                    origin,
                    reason: SkipReason::QueueFull,
                });
            }
        }
    }

    fn push_activity(&mut self, entry: ActivityEntry) -> u64 {
        self.next_entry += 1;
        if self.recent.len() == RECENT_CAPACITY {
            self.recent.pop_back();
        }
        self.recent.push_front((self.next_entry, entry));
        self.next_entry
    }

    fn on_dispatch(&mut self, msg: DispatchMsg) {
        match msg {
            DispatchMsg::PlatformReady => tracing::debug!("the platform is ready"),
            DispatchMsg::Completed { key, result } => self.on_completed(key, result),
            DispatchMsg::CommandFinished { key, step, outcome } => {
                if let Some(info) = self.jobs.get(&key) {
                    let (zone, origin) = (info.zone, info.origin);
                    self.emit(Event::CommandFinished {
                        zone,
                        origin,
                        step,
                        outcome,
                    });
                } else {
                    self.counters.stale_messages += 1;
                }
            }
            DispatchMsg::PlatformRebuilt { panic } => {
                tracing::error!("the platform was rebuilt after a panic");
                self.warnings
                    .retain(|w| !matches!(w, HealthWarning::PlatformRebuilt { .. }));
                self.warnings.push(HealthWarning::PlatformRebuilt { panic });
            }
        }
    }

    fn on_completed(&mut self, key: JobKey, result: ActionResult) {
        let started_command = result
            .steps
            .iter()
            .any(|s| matches!(s, StepOutcome::Started));
        let info = if started_command {
            self.jobs
                .get(&key)
                .map(|i| (i.zone, i.label.clone(), i.origin, i.tap, i.entry))
        } else {
            self.jobs
                .remove(&key)
                .map(|i| (i.zone, i.label, i.origin, i.tap, i.entry))
        };
        let Some((zone, label, origin, tap, entry)) = info else {
            self.counters.stale_messages += 1;
            return;
        };
        if let Some(StepOutcome::Failed { error }) = result.steps.last() {
            self.counters.actions_failed += 1;
            tracing::warn!(kind = ?error.kind(), step = result.steps.len() - 1, "an action failed");
            self.note_attention(zone, &label, error.clone());
        }
        let latency = match (origin, tap) {
            (JobOrigin::Tap, Some(tap)) => {
                let sample = LatencySample {
                    end_to_end: ms_between(result.started_at, tap.t0),
                    decision: ms_between(tap.decided_at, tap.t0),
                    controller: ms_between(tap.fired_at, tap.decided_at),
                    dispatch: ms_between(result.started_at, tap.fired_at),
                };
                if self.latency.len() == LATENCY_WINDOW {
                    self.latency.pop_front();
                }
                self.latency.push_back(sample);
                Some(ms(sample.end_to_end))
            }
            (JobOrigin::Tap, None) | (JobOrigin::TestAction { .. } | JobOrigin::TestChain, _) => {
                None
            }
        };
        if let Some(id) = entry {
            if let Some((
                _,
                ActivityEntry::Fired {
                    latency: l,
                    result: r,
                    ..
                },
            )) = self.recent.iter_mut().find(|(i, _)| *i == id)
            {
                *l = latency;
                *r = Some(result.clone());
            }
        }
        self.emit(Event::ActionCompleted {
            zone,
            tap,
            origin,
            result,
        });
    }

    fn note_attention(&mut self, zone: ZoneId, label: &str, error: crate::platform::ActionError) {
        let now = Utc::now();
        let same = |a: &AttentionItem| {
            a.zone == zone && std::mem::discriminant(&a.error) == std::mem::discriminant(&error)
        };
        if let Some(item) = self.attention.iter_mut().find(|a| same(a)) {
            item.count += 1;
            item.last_at = now;
            item.error = error;
            return;
        }
        if self.attention.len() == ATTENTION_CAPACITY {
            self.attention.remove(0);
        }
        self.next_attention += 1;
        self.attention.push(AttentionItem {
            id: AttentionId(self.next_attention),
            first_at: now,
            last_at: now,
            count: 1,
            zone,
            label: label.to_string(),
            error,
        });
    }

    fn fire_test(&mut self, test: PendingTest) {
        let Some((label, actions)) = self.zone_info(test.zone) else {
            return;
        };
        let actions = match test.origin {
            JobOrigin::TestAction { action_index } => actions
                .get(action_index as usize)
                .cloned()
                .into_iter()
                .collect(),
            JobOrigin::TestChain | JobOrigin::Tap => actions,
        };
        let now = self.now();
        let entry = self.push_activity(ActivityEntry::Fired {
            at: Utc::now(),
            zone: test.zone,
            label: label.clone(),
            origin: test.origin,
            tap: None,
            latency: None,
            result: None,
            close_call: None,
        });
        self.send_job(
            test.zone,
            label,
            actions,
            test.origin,
            None,
            Some(entry),
            now,
        );
    }

    fn cancel_tests(&mut self) {
        let tests = std::mem::take(&mut self.pending_tests);
        for test in tests {
            self.emit(Event::ActionSkipped {
                zone: test.zone,
                origin: test.origin,
                reason: SkipReason::TestCanceled,
            });
        }
    }

    // -----------------------------------------------------------------------
    // Health aggregation (§2.8)
    // -----------------------------------------------------------------------

    fn on_health(&mut self, health: ProcessingHealth) {
        let d = health.dropped;
        self.counters.decisions_dropped = d.decisions;
        // ProcessingDrops are processing's lifetime totals; add only what is new, since the
        // controller adds its own drops to the same counters (§2.11: Trace frames count in
        // telemetry_dropped; every failed non-decision send counts in controller_dropped).
        self.counters.telemetry_dropped +=
            d.telemetry.saturating_sub(self.processing_telemetry_seen);
        self.processing_telemetry_seen = self.processing_telemetry_seen.max(d.telemetry);
        self.counters.controller_dropped +=
            d.messages.saturating_sub(self.processing_messages_seen);
        self.processing_messages_seen = self.processing_messages_seen.max(d.messages);
        self.counters.recorder_dropped_frames = d.recorder_frames;
        self.counters.recorder_dropped_events = d.recorder_events + self.decision_feed_dropped;
        // SPEC: §2.11's table names `input_dropped` but no source for it; the raw-input worker
        // coalesces every overflow into one Gap record, and the timeline counts each Gap (and each
        // eviction) in `gaps`, the one lifetime count of input the veto could not see.
        self.counters.input_dropped = health.veto.gaps;
        if health.speech_unmatched > self.speech_unmatched_seen {
            let increase = health.speech_unmatched - self.speech_unmatched_seen;
            self.speech_unmatched_seen = health.speech_unmatched;
            self.counters.faults += increase;
            tracing::error!(
                count = increase,
                "speech finishes with no open speech output"
            );
            self.warnings
                .retain(|w| !matches!(w, HealthWarning::SpeechUnmatched { .. }));
            self.warnings.push(HealthWarning::SpeechUnmatched {
                count: health.speech_unmatched,
            });
        }
        self.health = Some(health);
    }

    fn note_near_miss(&mut self, rejected: &qwerty_core::detector::RejectedCandidate) {
        use qwerty_core::detector::RejectReason;
        // A candidate without a gate report (Superseded, PendingOverflow, ...) is no near miss.
        let near = match &rejected.gates {
            Some(gates) => {
                matches!(
                    rejected.reason,
                    RejectReason::LowSnr | RejectReason::InsufficientBandSnr
                ) && (NEAR_MISS_LOW_DB..NEAR_MISS_HIGH_DB).contains(&gates.snr.get())
            }
            None => false,
        };
        self.note_detector_outcome(near);
    }

    /// One detector outcome for LowSignal's window (§2.8: the last 20 detector outcomes while
    /// Listening or Evaluating). A candidate that passed the detector (a Decision or Detected)
    /// is an outcome too, never a near miss.
    fn note_detector_outcome(&mut self, near_miss: bool) {
        if !matches!(self.mode, Mode::Listening | Mode::Evaluating { .. }) {
            return;
        }
        if self.near_misses.len() == LOW_SIGNAL_WINDOW {
            self.near_misses.pop_front();
        }
        self.near_misses.push_back(near_miss);
    }

    fn aggregate_health(&mut self) {
        let mut keep: Vec<HealthWarning> = self
            .warnings
            .iter()
            .filter(|w| {
                matches!(
                    w,
                    HealthWarning::SpeechUnmatched { .. }
                        | HealthWarning::RecognitionDrift { .. }
                        | HealthWarning::AutostartFailed { .. }
                        | HealthWarning::PlatformRebuilt { .. }
                        | HealthWarning::DiskSpaceLow { .. }
                        | HealthWarning::NoiseTrackerStarved { .. }
                        | HealthWarning::LowSignal { .. }
                )
            })
            .cloned()
            .collect();
        let starved = keep
            .iter()
            .any(|w| matches!(w, HealthWarning::NoiseTrackerStarved { .. }));
        keep.retain(|w| !matches!(w, HealthWarning::NoiseTrackerStarved { .. }));
        if let Some(detector) = self.health.as_ref().and_then(|h| h.detector.as_ref()) {
            let f = detector.frozen_fraction;
            let on = if starved {
                f >= qwerty_core::detector::STARVED_EXIT
            } else {
                f > qwerty_core::detector::STARVED_ENTER
            };
            if on {
                keep.push(HealthWarning::NoiseTrackerStarved { frozen_fraction: f });
            }
        }
        let low = keep
            .iter()
            .any(|w| matches!(w, HealthWarning::LowSignal { .. }));
        keep.retain(|w| !matches!(w, HealthWarning::LowSignal { .. }));
        if self.near_misses.len() == LOW_SIGNAL_WINDOW {
            let fraction =
                self.near_misses.iter().filter(|n| **n).count() as f64 / LOW_SIGNAL_WINDOW as f64;
            let on = if low {
                fraction >= LOW_SIGNAL_CLEAR
            } else {
                fraction >= LOW_SIGNAL_RAISE
            };
            if on {
                keep.push(HealthWarning::LowSignal {
                    near_miss_fraction: fraction,
                });
            }
        }
        if let Some(health) = &self.health {
            for id in &health.stuck_outputs {
                keep.push(HealthWarning::SelfOutputStuck { id: *id });
            }
            if health.stalled {
                keep.push(HealthWarning::CaptureStalled {
                    after: crate::engine::processing::STALL_AFTER,
                });
            }
        }
        if let Some(capture) = &self.capture {
            if let crate::engine::protocol::MmcssStatus::Unavailable { code } = capture.opened.mmcss
            {
                keep.push(HealthWarning::MmcssUnavailable { code });
            }
        }
        if self.counters.capture_dropped_frames > self.capture_dropped_seen {
            keep.push(HealthWarning::CaptureOverflow {
                frames: self.counters.capture_dropped_frames - self.capture_dropped_seen,
            });
            self.capture_dropped_seen = self.counters.capture_dropped_frames;
        }
        let dropped_now = self.devices_dropped.swap(0, Ordering::AcqRel);
        self.counters.controller_dropped += dropped_now;
        // §2.11: a dropped decision and any other dropped message both mean the controller fell
        // behind.
        if self.counters.controller_dropped > self.controller_dropped_seen
            || self.counters.decisions_dropped > self.decisions_dropped_seen
        {
            keep.push(HealthWarning::ControllerOverloaded);
            self.controller_dropped_seen = self.counters.controller_dropped;
            self.decisions_dropped_seen = self.counters.decisions_dropped;
        }
        if self.counters.events_dropped > self.events_dropped_seen {
            keep.push(HealthWarning::EventsDropped {
                count: self.counters.events_dropped - self.events_dropped_seen,
            });
            self.events_dropped_seen = self.counters.events_dropped;
        }
        let panics = self.speech.callback_panics.load(Ordering::Relaxed);
        if panics > 0 {
            keep.push(HealthWarning::SpeechCallbackPanicked { count: panics });
        }
        self.warnings = keep;
        let samples: Vec<f64> = self.latency.iter().map(|s| s.end_to_end).collect();
        self.latency_stats = LatencyStats::from_samples(&samples, 0);
    }

    // -----------------------------------------------------------------------
    // Faults and shutdown (§2.11)
    // -----------------------------------------------------------------------

    fn enter_fault(&mut self, thread: ThreadName, message: String) {
        if self.fault.is_some() {
            return;
        }
        tracing::error!(
            thread = thread.os_name(),
            "an engine thread panicked; the engine is faulted"
        );
        self.fault_stops = Some(self.stop_workers());
        self.fault = Some((thread, message.clone()));
        self.run = None;
        self.evaluation = None;
        self.capture_status = CaptureStatus::Off;
        self.emit(Event::EngineFaulted { thread, message });
    }

    /// Steps 1-7 of the shutdown order, each stop bounded by min(2 s, what is left of 5 s).
    fn stop_workers(&mut self) -> ShutdownReport {
        let deadline =
            std::time::Instant::now() + Duration::from_secs_f64(SHUTDOWN_DEADLINE.get() / 1000.0);
        let mut report = ShutdownReport {
            joined: Vec::new(),
            timed_out: Vec::new(),
            recording: None,
        };
        let stop = |worker: Option<Worker>, thread: ThreadName, report: &mut ShutdownReport| {
            let Some(worker) = worker else { return };
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            match worker.stop(WORKER_STOP.min(left)) {
                Ok(()) => report.joined.push(thread),
                Err(_) => report.timed_out.push(thread),
            }
        };
        stop(self.raw_input.take(), ThreadName::RawInput, &mut report);
        if let Some(build) = self.build.take() {
            build.cancel.store(true, Ordering::Release);
            stop(Some(build.worker), ThreadName::Build, &mut report);
        }
        let had_recording = self.recording.is_some();
        stop(
            self.capture.take().map(|c| c.worker),
            ThreadName::Capture,
            &mut report,
        );
        stop(self.processing.take(), ThreadName::Processing, &mut report);
        stop(self.dispatch.take(), ThreadName::Dispatch, &mut report);
        if had_recording {
            self.recording = None;
            if self.send_recorder(RecorderCmd::Stop {
                reason: RecorderStopReason::CaptureStopped,
            }) {
                // The recorder answers Stopped with the summary.
                let left = deadline
                    .saturating_duration_since(std::time::Instant::now())
                    .min(WORKER_STOP);
                let until = std::time::Instant::now() + left;
                while let Some(wait) = until.checked_duration_since(std::time::Instant::now()) {
                    match self.ch.recorder.recv_timeout(wait) {
                        Ok(msg) => {
                            let stopped = matches!(
                                msg,
                                RecorderMsg::Stopped { .. } | RecorderMsg::Failed { .. }
                            );
                            self.on_recorder(msg);
                            if stopped {
                                break;
                            }
                        }
                        Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => {
                            break
                        }
                    }
                }
            }
        }
        stop(
            self.recorder_worker.take(),
            ThreadName::Recorder,
            &mut report,
        );
        self.device_watch = None;
        report.recording = self.last_summary.clone();
        report
    }

    fn shutdown_sequence(&mut self) -> ShutdownReport {
        self.shutting_down = true;
        let mut report = self.stop_workers();
        // A fault already stopped the workers; its stops are this shutdown's too.
        if let Some(earlier) = self.fault_stops.take() {
            let mut joined = earlier.joined;
            joined.append(&mut report.joined);
            report.joined = joined;
            let mut timed_out = earlier.timed_out;
            timed_out.append(&mut report.timed_out);
            report.timed_out = timed_out;
            if report.recording.is_none() {
                report.recording = earlier.recording;
            }
        }
        report
    }

    /// The controller's own panic arm (§2.11): drop every worker (each Drop signals stop), run the
    /// bounded stops, store a Faulted snapshot, send EngineFaulted and Stopped.
    pub(crate) fn after_panic(&mut self, message: String) {
        drop(self.stop_workers());
        self.fault = Some((ThreadName::Controller, message.clone()));
        self.run = None;
        self.evaluation = None;
        self.capture_status = CaptureStatus::Off;
        self.publish(true);
        self.emit(Event::EngineFaulted {
            thread: ThreadName::Controller,
            message,
        });
        self.emit(Event::Stopped);
    }
}

impl Controller {
    // -----------------------------------------------------------------------
    // Commands
    // -----------------------------------------------------------------------

    fn on_command(&mut self, msg: CommandMsg) {
        let CommandMsg { command, reply } = msg;
        let kind = command.kind();
        let result = if self.shutting_down {
            Err(EngineError::ShuttingDown)
        } else if let Command::Shutdown = command {
            let report = self.shutdown_sequence();
            self.publish(true);
            if reply.send(Ok(Reply::Shutdown { report })).is_err() {
                tracing::debug!("the shutdown requester is gone");
            }
            self.emit(Event::Stopped);
            self.exit = true;
            // Later commands find the channel closed and get EngineStopped from the handle.
            while let Ok(m) = self.ch.commands.try_recv() {
                drop(m.reply.send(Err(EngineError::ShuttingDown)));
            }
            return;
        } else if self.fault.is_some() && kind != CommandKind::GetSnapshot {
            Err(EngineError::InvalidInMode {
                command: kind,
                mode: ModeKind::Faulted,
            })
        } else {
            self.handle(command)
        };
        if reply.send(result).is_err() {
            tracing::debug!("a requester stopped waiting for its reply");
        }
        self.reconcile();
    }

    fn invalid(&self, command: CommandKind) -> EngineError {
        EngineError::InvalidInMode {
            command,
            mode: self.mode.kind(),
        }
    }

    fn handle(&mut self, command: Command) -> Result<Reply, EngineError> {
        let kind = command.kind();
        match command {
            Command::ShellReady => {
                self.shell_ready = true;
                Ok(Reply::Ack)
            }
            Command::GetSnapshot => {
                self.mode = self.resolve_mode();
                let mut snapshot = self.build_snapshot();
                snapshot.revision = self.revision;
                Ok(Reply::Snapshot {
                    snapshot: Box::new(snapshot),
                })
            }
            // INVARIANT: on_command answers Shutdown itself and never calls handle with it.
            Command::Shutdown => unreachable!("Shutdown is handled before dispatch"),
            Command::SetListening { listening } => {
                self.desired = if listening {
                    Desired::Listening
                } else {
                    Desired::Paused
                };
                Ok(Reply::Ack)
            }
            Command::ToggleListening => {
                self.desired = match self.desired {
                    Desired::Listening => Desired::Paused,
                    Desired::Paused => Desired::Listening,
                };
                Ok(Reply::Ack)
            }
            Command::SetTelemetry { level } => {
                self.telemetry = level;
                self.send_processing(ProcessingCmd::SetTelemetry { level });
                Ok(Reply::Ack)
            }
            Command::ListDevices => {
                let devices = self
                    .os
                    .list_devices()
                    .map_err(|error| EngineError::Capture { error })?;
                self.devices = devices.clone();
                Ok(Reply::Devices { devices })
            }
            Command::RetryCapture => {
                if let CaptureStatus::Failed { endpoint_id, .. } = self.capture_status.clone() {
                    self.capture_status = CaptureStatus::Off;
                    self.retry_open(endpoint_id, 0);
                } else {
                    self.device_attempt = None;
                    self.pipeline_failed = None;
                }
                Ok(Reply::Ack)
            }
            Command::RetryInputVeto => {
                self.stop_raw_input();
                if matches!(self.veto, VetoStatus::Unavailable { .. }) {
                    self.veto = VetoStatus::NotNeeded;
                }
                Ok(Reply::Ack)
            }
            Command::UpdateConfig { edit } => self.update_config(edit),
            Command::ResetConfig => {
                if !matches!(self.config_status, ConfigStatus::Unreadable { .. }) {
                    return Err(EngineError::ConfigReadable);
                }
                let config = self
                    .data
                    .reset_config(Utc::now())
                    .map_err(|error| EngineError::Persistence { error })?;
                self.config = config;
                self.config_status = ConfigStatus::Initial;
                Ok(Reply::Ack)
            }
            Command::ReconcileAutostart => {
                if self.autostart == AutostartPolicy::Untouched {
                    return Err(EngineError::AutostartNotManaged);
                }
                self.reconcile_autostart(self.config.start_with_windows)
                    .map_err(|detail| EngineError::Autostart { detail })?;
                Ok(Reply::Ack)
            }
            Command::SwitchProfile { id } => {
                if self.run.is_some() || self.evaluation.is_some() {
                    return Err(self.invalid(kind));
                }
                if self.profile(id).is_none() {
                    return Err(EngineError::ProfileNotFound { id });
                }
                self.writable_config()?;
                let mut config = self.config.clone();
                config.active_profile_id = Some(id);
                self.save_config(config)?;
                self.cancel_tests();
                self.stop_capture();
                self.active = Active::Profile(id);
                self.device_attempt = None;
                self.load_active_model();
                Ok(Reply::Ack)
            }
            Command::RenameProfile { id, name } => {
                let mut profile = self
                    .profile(id)
                    .cloned()
                    .ok_or(EngineError::ProfileNotFound { id })?;
                profile.name = name;
                profile.updated_at = Utc::now();
                profile
                    .validate()
                    .map_err(|error| EngineError::Profile { error })?;
                self.write_profile(profile)?;
                Ok(Reply::Ack)
            }
            Command::DuplicateProfile { id, name } => {
                let mut profile = self
                    .profile(id)
                    .cloned()
                    .ok_or(EngineError::ProfileNotFound { id })?;
                let now = Utc::now();
                profile.id = ProfileId::new();
                profile.name = name;
                profile.created_at = now;
                profile.updated_at = now;
                profile
                    .validate()
                    .map_err(|error| EngineError::Profile { error })?;
                let new_id = profile.id;
                self.write_profile(profile)?;
                Ok(Reply::ProfileCreated { id: new_id })
            }
            Command::DeleteProfile { id } => {
                let recalibrating = self
                    .run
                    .as_ref()
                    .is_some_and(|r| r.target == CalibrationTarget::Recalibrate { profile: id });
                if recalibrating || (self.evaluation.is_some() && self.active_id() == Some(id)) {
                    return Err(self.invalid(kind));
                }
                if self.profile(id).is_none() {
                    return Err(EngineError::ProfileNotFound { id });
                }
                self.data
                    .delete_profile(id)
                    .map_err(|error| EngineError::Persistence { error })?;
                let file = profile_file(id);
                self.profiles.retain(|e| e.file != file);
                if self.active_id() == Some(id) {
                    self.cancel_tests();
                    self.stop_capture();
                    self.active = Active::None;
                    self.load_active_model();
                    if self.writable_config().is_ok() {
                        let mut config = self.config.clone();
                        config.active_profile_id = None;
                        self.save_config(config)?;
                    }
                }
                Ok(Reply::Ack)
            }
            Command::EditZone { zone, edit } => {
                if self.run.is_some() {
                    return Err(self.invalid(kind));
                }
                let id = self.active_id().ok_or(EngineError::NoActiveProfile)?;
                let mut profile = self
                    .profile(id)
                    .cloned()
                    .ok_or(EngineError::NoActiveProfile)?;
                profile
                    .edit_zone(zone, edit, Utc::now())
                    .map_err(|error| EngineError::Profile { error })?;
                self.write_profile(profile)?;
                // Triggers may have changed: the pipeline is rebuilt.
                self.ensure_setup(self.capture.is_some());
                Ok(Reply::Ack)
            }
            Command::ExportProfile { id, path } => {
                let profile = self
                    .profile(id)
                    .ok_or(EngineError::ProfileNotFound { id })?;
                persist::export_profile(profile, &path)
                    .map_err(|error| EngineError::Persistence { error })?;
                Ok(Reply::Saved { path })
            }
            Command::PreviewImport { path } => {
                let profile = match persist::read_import(&path)
                    .map_err(|error| EngineError::Persistence { error })?
                {
                    Loaded::Current(p) => p,
                    Loaded::Migrated { value, .. } => value,
                };
                self.next_preview += 1;
                let preview = PreviewId(self.next_preview);
                let view = ImportPreview {
                    preview,
                    name: profile.name.clone(),
                    zones: profile.zones.clone(),
                    calibration: profile_calibration_status(&profile),
                };
                self.preview = Some((preview, profile));
                Ok(Reply::ImportPreview {
                    preview: Box::new(view),
                })
            }
            Command::ImportProfile { preview } => {
                let mut profile = match self.preview.take() {
                    Some((id, profile)) if id == preview => profile,
                    Some(other) => {
                        self.preview = Some(other);
                        return Err(EngineError::PreviewExpired);
                    }
                    None => return Err(EngineError::PreviewExpired),
                };
                profile.id = ProfileId::new();
                if let CalibrationState::Calibrated { device, .. } = &profile.calibration {
                    let here = self
                        .os
                        .list_devices()
                        .map_err(|error| EngineError::Capture { error })?
                        .iter()
                        .any(|d| d.endpoint_id == device.identity.endpoint_id);
                    profile.imported_at = if here { None } else { Some(Utc::now()) };
                }
                let new_id = profile.id;
                self.write_profile(profile)?;
                Ok(Reply::ProfileCreated { id: new_id })
            }
            Command::RebindDevice {
                profile,
                endpoint_id,
            } => self.rebind(kind, profile, endpoint_id),
            Command::StartCalibration {
                target,
                zones,
                device,
                taps_per_zone,
            } => self.start_calibration(kind, target, zones, device, taps_per_zone),
            Command::Calibration { action } => self.calibration_action(kind, action),
            Command::CancelCalibration => {
                if self.run.is_none() {
                    return Err(self.invalid(kind));
                }
                self.end_run();
                self.emit(Event::CalibrationEnded {
                    outcome: CalibrationOutcome::Cancelled,
                });
                Ok(Reply::Ack)
            }
            Command::StartEvaluation {
                prompts_per_zone,
                negatives,
            } => self.start_evaluation(kind, prompts_per_zone, negatives),
            Command::UndoLastPrompt => {
                let now = self.now();
                let eval = self.evaluation.as_mut().ok_or(EngineError::InvalidInMode {
                    command: kind,
                    mode: self.mode.kind(),
                })?;
                let mut events = Vec::new();
                eval.session
                    .undo_last(now, &mut events)
                    .map_err(|error| EngineError::Evaluation { error })?;
                self.on_evaluation_events(events);
                Ok(Reply::Ack)
            }
            Command::CancelEvaluation => {
                let eval = self.evaluation.take().ok_or(EngineError::InvalidInMode {
                    command: kind,
                    mode: self.mode.kind(),
                })?;
                self.eval_tick_at = None;
                self.desired = eval.prior;
                self.emit(Event::EvaluationEnded {
                    outcome: EvaluationOutcome::Cancelled,
                });
                Ok(Reply::Ack)
            }
            Command::RetrySaveReport => {
                let report = self
                    .unsaved_report
                    .as_ref()
                    .ok_or(EngineError::NoUnsavedReport)?;
                let id = self
                    .data
                    .save_report(report)
                    .map_err(|error| EngineError::Persistence { error })?;
                // INVARIANT: checked just above.
                let report = self
                    .unsaved_report
                    .take()
                    .expect("an unsaved report is held");
                self.refresh_latest_report(report.profile_id);
                self.emit(Event::EvaluationEnded {
                    outcome: EvaluationOutcome::Saved {
                        report: id,
                        overall: report.overall.clone(),
                    },
                });
                Ok(Reply::Ack)
            }
            Command::DiscardUnsavedReport => {
                self.unsaved_report
                    .take()
                    .ok_or(EngineError::NoUnsavedReport)?;
                Ok(Reply::Ack)
            }
            Command::ExportUnsavedReport { format, path } => {
                let report = self
                    .unsaved_report
                    .as_ref()
                    .ok_or(EngineError::NoUnsavedReport)?;
                persist::export_report(report, format, &path)
                    .map_err(|error| EngineError::Persistence { error })?;
                Ok(Reply::Saved { path })
            }
            Command::ListEvaluations { profile } => {
                if self.profile(profile).is_none() {
                    return Err(EngineError::ProfileNotFound { id: profile });
                }
                let reports = self
                    .data
                    .list_reports(profile)
                    .map_err(|error| EngineError::Persistence { error })?;
                Ok(Reply::Evaluations { reports })
            }
            Command::GetEvaluation { profile, report } => {
                let loaded = self.load_report(profile, &report)?;
                Ok(Reply::EvaluationReport { report: loaded })
            }
            Command::ExportEvaluation {
                profile,
                report,
                format,
                path,
            } => {
                let loaded = self.load_report(profile, &report)?;
                persist::export_report(&loaded, format, &path)
                    .map_err(|error| EngineError::Persistence { error })?;
                Ok(Reply::Saved { path })
            }
            Command::DeleteEvaluation { profile, report } => {
                let removed = self
                    .data
                    .delete_report(profile, &report)
                    .map_err(|error| EngineError::Persistence { error })?;
                if !removed {
                    return Err(EngineError::ReportNotFound { profile, report });
                }
                self.refresh_latest_report(profile);
                Ok(Reply::Ack)
            }
            Command::TestAction {
                zone,
                action_index,
                delay,
            } => self.schedule_test(kind, zone, JobOrigin::TestAction { action_index }, delay),
            Command::TestChain { zone, delay } => {
                self.schedule_test(kind, zone, JobOrigin::TestChain, delay)
            }
            Command::CancelTest { zone } => {
                let Some(i) = self.pending_tests.iter().position(|t| t.zone == zone) else {
                    return Err(EngineError::NoPendingTest { zone });
                };
                let test = self.pending_tests.remove(i);
                self.emit(Event::ActionSkipped {
                    zone,
                    origin: test.origin,
                    reason: SkipReason::TestCanceled,
                });
                Ok(Reply::Ack)
            }
            Command::DismissAttention { id } => {
                let before = self.attention.len();
                self.attention.retain(|a| a.id != id);
                if self.attention.len() == before {
                    return Err(EngineError::AttentionNotFound { id });
                }
                Ok(Reply::Ack)
            }
            Command::ShellToast => {
                // The shell's toast has no end signal: the stuck-output timer closes it (§2.6).
                self.link.output_started(SelfOutputKind::Toast);
                Ok(Reply::Ack)
            }
            Command::GetInspection => Ok(Reply::Inspection {
                entries: self.inspection.iter().cloned().collect(),
            }),
            Command::ExportInspection { path } => {
                let entries: Vec<TraceEntry> = self.inspection.iter().cloned().collect();
                persist::export_inspection(&entries, &path)
                    .map_err(|error| EngineError::Persistence { error })?;
                Ok(Reply::Saved { path })
            }
            Command::GetSoundMap => {
                let model = self
                    .model
                    .as_ref()
                    .ok_or(EngineError::ProfileNotCalibrated)?;
                use qwerty_core::training::{ModelQuality, QualityIssue};
                let issues: &[QualityIssue] = match &model.model.summary.quality {
                    ModelQuality::Good => &[],
                    ModelQuality::Marginal { issues } | ModelQuality::Poor { issues } => issues,
                };
                let overlaps = issues
                    .iter()
                    .filter_map(|issue| match issue {
                        QualityIssue::ZonesConfusable { a, b, .. } => Some([*a, *b]),
                        QualityIssue::InconsistentZone { .. }
                        | QualityIssue::BackgroundOverlap { .. }
                        | QualityIssue::HighRejection { .. }
                        | QualityIssue::FewNegatives
                        | QualityIssue::DegenerateDims { .. } => None,
                    })
                    .collect();
                Ok(Reply::SoundMap {
                    map: model.model.map.clone(),
                    overlaps,
                })
            }
            Command::SaveLatencyReport => {
                let profile = self.active_profile().ok_or(EngineError::NoActiveProfile)?;
                let CalibrationState::Calibrated { device, .. } = &profile.calibration else {
                    return Err(EngineError::ProfileNotCalibrated);
                };
                let stats = |f: fn(&LatencySample) -> f64| {
                    let v: Vec<f64> = self.latency.iter().map(f).collect();
                    LatencyStats::from_samples(&v, 0)
                };
                let end_to_end = stats(|s| s.end_to_end);
                let passed = end_to_end.count >= LATENCY_MIN_TAPS
                    && end_to_end.p50.get() < LATENCY_BUDGET_P50.get()
                    && end_to_end.p95.get() < LATENCY_BUDGET_P95.get();
                let report = LatencyReport {
                    schema_version: LATENCY_REPORT_SCHEMA_VERSION,
                    created_at: Utc::now(),
                    profile_id: profile.id,
                    device: device.identity.clone(),
                    end_to_end,
                    decision: stats(|s| s.decision),
                    controller: stats(|s| s.controller),
                    dispatch: stats(|s| s.dispatch),
                    budget_p50: LATENCY_BUDGET_P50,
                    budget_p95: LATENCY_BUDGET_P95,
                    passed,
                };
                let path = self
                    .data
                    .save_latency_report(&report)
                    .map_err(|error| EngineError::Persistence { error })?;
                Ok(Reply::Saved { path })
            }
            Command::MarkSegment { kind: segment } => {
                if self.recording.is_none() {
                    return Err(EngineError::NotRecording);
                }
                self.segment_override = Some(segment);
                self.ensure_setup(true);
                Ok(Reply::Ack)
            }
            Command::DeleteDebugRecordings => {
                if self.recording.is_some() {
                    return Err(EngineError::RecordingActive);
                }
                let root = self.data.paths().recordings;
                let io = |e: std::io::Error| EngineError::Persistence {
                    error: persist::failure(&root, qwerty_core::profile::PersistenceError::Io(e)),
                };
                recorder::delete_recordings(&root).map_err(io)?;
                self.stored = recorder::stored_recordings(&root).map_err(io)?;
                Ok(Reply::Ack)
            }
        }
    }

    fn writable_config(&self) -> Result<(), EngineError> {
        match self.config_status {
            ConfigStatus::Unreadable { .. } => Err(EngineError::ConfigUnreadable),
            ConfigStatus::Loaded | ConfigStatus::Initial | ConfigStatus::Migrated { .. } => Ok(()),
        }
    }

    fn write_profile(&mut self, profile: Profile) -> Result<(), EngineError> {
        self.data
            .save_profile(&profile)
            .map_err(|error| EngineError::Persistence { error })?;
        let active = self.active_id() == Some(profile.id);
        self.upsert_profile(profile);
        if active {
            let had_model = self.model.is_some();
            self.load_active_model();
            if had_model {
                self.configured = None;
            }
        }
        Ok(())
    }

    fn load_report(
        &self,
        profile: ProfileId,
        report: &crate::engine::protocol::ReportId,
    ) -> Result<Box<EvaluationReport>, EngineError> {
        match self
            .data
            .load_report(profile, report)
            .map_err(|error| EngineError::Persistence { error })?
        {
            None => Err(EngineError::ReportNotFound {
                profile,
                report: report.clone(),
            }),
            Some(StoredReport::Legacy) => Err(EngineError::LegacyReport {
                report: report.clone(),
            }),
            Some(StoredReport::Current(r)) => Ok(r),
        }
    }

    fn update_config(
        &mut self,
        edit: crate::engine::protocol::ConfigEdit,
    ) -> Result<Reply, EngineError> {
        self.writable_config()?;
        let mut config = self.config.clone();
        if let Some(enabled) = edit.start_with_windows {
            if self.autostart == AutostartPolicy::Untouched {
                return Err(EngineError::AutostartNotManaged);
            }
            self.reconcile_autostart(enabled)
                .map_err(|detail| EngineError::Autostart { detail })?;
            config.start_with_windows = enabled;
        }
        if let Some(v) = edit.theme {
            config.theme = v;
        }
        if let Some(v) = edit.minimize_to_tray_on_close {
            config.minimize_to_tray_on_close = v;
        }
        if let Some(v) = edit.notification_verbosity {
            config.notification_verbosity = v;
        }
        if let Some(v) = edit.global_hotkey {
            config.global_hotkey = match v {
                crate::engine::protocol::HotkeyEdit::Clear => None,
                crate::engine::protocol::HotkeyEdit::Set { combo } => Some(combo),
            };
        }
        if let Some(v) = edit.debug_capture_enabled {
            config.debug_capture_enabled = v;
        }
        if let Some(v) = edit.input_veto_enabled {
            config.input_veto_enabled = v;
        }
        if let Some(v) = edit.eval_prompts_per_zone {
            config.eval_prompts_per_zone = v;
        }
        if let Some(v) = edit.eval_background_check {
            config.eval_background_check = v;
        }
        if let Some(notice) = edit.dismiss_notice {
            if !config.dismissed_notices.contains(&notice) {
                config.dismissed_notices.push(notice);
                config.dismissed_notices.sort();
            }
        }
        if let Some(v) = edit.window_placement {
            config.window_placement = Some(v);
        }
        if let Some(v) = edit.text_zoom {
            config.text_zoom = v;
        }
        config
            .validate()
            .map_err(|error| EngineError::Config { error })?;
        let was_recording_on = self.recording_on();
        self.save_config(config)?;
        if was_recording_on && !self.recording_on() {
            self.stop_recording(RecorderStopReason::Disabled);
        } else if !was_recording_on && self.recording_on() && self.capture.is_some() {
            self.start_recording();
        }
        Ok(Reply::Ack)
    }

    fn schedule_test(
        &mut self,
        kind: CommandKind,
        zone: ZoneId,
        origin: JobOrigin,
        delay: Milliseconds,
    ) -> Result<Reply, EngineError> {
        if self.run.is_some() || self.evaluation.is_some() {
            return Err(self.invalid(kind));
        }
        if self.pending_tests.iter().any(|t| t.zone == zone) {
            return Err(EngineError::TestPending { zone });
        }
        if self.zone_info(zone).is_none() {
            return Err(EngineError::Profile {
                error: qwerty_core::store::ProfileError::ZoneNotFound { zone },
            });
        }
        let fires_at = after(self.now(), delay);
        self.pending_tests.push(PendingTest {
            zone,
            origin,
            fires_at,
        });
        Ok(Reply::TestScheduled { fires_at })
    }

    fn rebind(
        &mut self,
        kind: CommandKind,
        id: ProfileId,
        endpoint_id: String,
    ) -> Result<Reply, EngineError> {
        if self.run.is_some() || self.evaluation.is_some() {
            return Err(self.invalid(kind));
        }
        let mut profile = self
            .profile(id)
            .cloned()
            .ok_or(EngineError::ProfileNotFound { id })?;
        if !matches!(profile.calibration, CalibrationState::Calibrated { .. }) {
            return Err(EngineError::ProfileNotCalibrated);
        }
        let probe = self.active_probe(&endpoint_id)?;
        let live = DeviceIdentity {
            endpoint_id: endpoint_id.clone(),
            name: probe.0,
            format: probe.1.format,
            capture_mode: if probe.1.raw_supported {
                qwerty_core::audio::CaptureMode::Raw
            } else {
                qwerty_core::audio::CaptureMode::Default {
                    reason: qwerty_core::audio::DefaultModeReason::RawNotSupported,
                }
            },
        };
        profile
            .rebind_device(&live, Utc::now())
            .map_err(|error| EngineError::Profile { error })?;
        self.write_profile(profile)?;
        if self.active_id() == Some(id) {
            self.stop_capture();
            self.device_attempt = None;
        }
        Ok(Reply::Ack)
    }

    /// A fresh listing's Active endpoint with a successful probe: (name, probe).
    fn active_probe(
        &mut self,
        endpoint_id: &str,
    ) -> Result<(String, crate::engine::protocol::DeviceProbe), EngineError> {
        let devices = self
            .os
            .list_devices()
            .map_err(|error| EngineError::Capture { error })?;
        self.devices = devices.clone();
        let device = devices
            .iter()
            .find(|d| d.endpoint_id == endpoint_id)
            .ok_or_else(|| EngineError::Capture {
                error: CaptureError::DeviceNotFound {
                    endpoint_id: endpoint_id.to_string(),
                },
            })?;
        if device.state != DeviceState::Active {
            return Err(EngineError::Capture {
                error: CaptureError::DeviceNotActive {
                    endpoint_id: endpoint_id.to_string(),
                    state: device.state,
                },
            });
        }
        match &device.probe {
            ProbeOutcome::Probed { probe } => Ok((device.name.clone(), *probe)),
            ProbeOutcome::Failed { error } => Err(EngineError::Capture {
                error: error.clone(),
            }),
            ProbeOutcome::NotActive => Err(EngineError::Capture {
                error: CaptureError::DeviceNotActive {
                    endpoint_id: endpoint_id.to_string(),
                    state: device.state,
                },
            }),
        }
    }
}

impl Controller {
    // -----------------------------------------------------------------------
    // Calibration (§2.8)
    // -----------------------------------------------------------------------

    fn start_calibration(
        &mut self,
        kind: CommandKind,
        target: CalibrationTarget,
        zones: Vec<ZoneDraft>,
        device: DeviceChoice,
        taps_per_zone: u32,
    ) -> Result<Reply, EngineError> {
        if self.evaluation.is_some() {
            return Err(self.invalid(kind));
        }
        let count = u8::try_from(zones.len())
            .map_err(|_| EngineError::Profile {
                error: qwerty_core::store::ProfileError::InvalidZoneCount { n: u8::MAX },
            })
            .and_then(|n| ZoneCount::new(n).map_err(|error| EngineError::Profile { error }))?;
        let now = Utc::now();
        let mut draft = match &target {
            CalibrationTarget::New { name } => Profile::new(name.clone(), count, now)
                .map_err(|error| EngineError::Profile { error })?,
            CalibrationTarget::Recalibrate { profile } => {
                let mut p = self
                    .profile(*profile)
                    .cloned()
                    .ok_or(EngineError::ProfileNotFound { id: *profile })?;
                if p.zones.len() != zones.len() {
                    p.set_zone_count(count, now);
                }
                p
            }
        };
        let ids: Vec<ZoneId> = draft.zones.iter().map(|z| z.id).collect();
        for (id, zone) in ids.iter().zip(&zones) {
            draft
                .edit_zone(
                    *id,
                    ZoneEdit {
                        label: Some(zone.label.clone()),
                        layout: Some(zone.layout),
                        trigger: None,
                        actions: None,
                    },
                    now,
                )
                .map_err(|error| EngineError::Profile { error })?;
        }
        let endpoint = match &device {
            DeviceChoice::ProfileBound => match &target {
                CalibrationTarget::Recalibrate { profile } => match &self
                    .profile(*profile)
                    .ok_or(EngineError::ProfileNotFound { id: *profile })?
                    .calibration
                {
                    CalibrationState::Calibrated { device, .. } => {
                        device.identity.endpoint_id.clone()
                    }
                    CalibrationState::NeedsRecalibration { .. } => {
                        return Err(EngineError::ProfileNotCalibrated)
                    }
                },
                CalibrationTarget::New { .. } => return Err(EngineError::ProfileNotCalibrated),
            },
            DeviceChoice::Endpoint { endpoint_id } => {
                self.active_probe(endpoint_id)?;
                endpoint_id.clone()
            }
            DeviceChoice::SystemDefault => {
                let devices = self
                    .os
                    .list_devices()
                    .map_err(|error| EngineError::Capture { error })?;
                self.devices = devices.clone();
                devices
                    .iter()
                    .find(|d| d.is_default)
                    .map(|d| d.endpoint_id.clone())
                    .ok_or(EngineError::Capture {
                        error: CaptureError::NoDefaultDevice,
                    })?
            }
        };
        self.cancel_tests();
        // A new run replaces the old one: its session, audio and build are dropped.
        self.drop_run_state();
        let has_calibration = self.recording.as_ref().is_some_and(|r| r.has_calibration);
        let reopen =
            self.capture.as_ref().map(|c| c.endpoint_id.as_str()) != Some(endpoint.as_str());
        if reopen {
            self.open(&endpoint, CapturePurpose::Calibrating)
                .map_err(|error| EngineError::Capture { error })?;
        } else if has_calibration {
            self.stop_recording(RecorderStopReason::NewCalibrationRun);
            if self.recording_on() {
                self.start_recording();
            }
        }
        // INVARIANT: the capture was opened (or kept) just above.
        let capture = self
            .capture
            .as_mut()
            .expect("the calibration capture is open");
        capture.purpose = CapturePurpose::Calibrating;
        let identity = capture.opened.identity.clone();
        self.capture_status = CaptureStatus::Running {
            opened: capture.opened.clone(),
            purpose: CapturePurpose::Calibrating,
        };
        let seed = self.now().0 as u64;
        let labels: Vec<String> = zones.iter().map(|z| z.label.clone()).collect();
        let plan = CalibrationPlan {
            zone_ids: ids.clone(),
            zone_labels: labels.clone(),
            taps_per_zone,
            format: identity.format,
            seed,
            audio_budget_bytes: session::DEFAULT_AUDIO_BUDGET_BYTES,
            detector: DetectorParams::DEFAULT,
            training: TrainingParams::DEFAULT,
        };
        let session = match CalibrationSession::new(plan) {
            Ok(s) => s,
            Err(error) => {
                self.reconcile();
                return Err(EngineError::Calibration { error });
            }
        };
        self.run = Some(Run {
            target,
            zones,
            device: identity,
            endpoint_id: endpoint,
            taps_per_zone,
            draft,
            zone_ids: ids,
            zone_labels: labels,
            state: RunState::Active(Box::new(session)),
            layout: None,
        });
        self.pending_calibration_start = Some(CalibrationStart {
            seed,
            taps_per_zone,
        });
        self.setup_caller = Some(CommandKind::StartCalibration);
        self.ensure_setup(true);
        self.setup_caller = None;
        if let Some(error) = self.run_setup_error.take() {
            // on_setup_error ended the run and reconciled.
            self.pending_calibration_start = None;
            return Err(EngineError::Setup { error });
        }
        // INVARIANT: the run was stored just above.
        let run = self.run.as_ref().expect("the run was just stored");
        Ok(Reply::Calibration {
            run: Box::new(self.run_view(run)),
        })
    }

    /// Runs `f` on the active session, then applies the setup rule and the step's timers.
    fn with_session<T>(
        &mut self,
        f: impl FnOnce(&mut CalibrationSession, HostTime) -> Result<T, SessionError>,
    ) -> Result<T, EngineError> {
        let now = self.now();
        let before = self.session_step_kind();
        let result = match self.run.as_mut().map(|r| &mut r.state) {
            Some(RunState::Active(session)) => f(session, now),
            Some(RunState::MicLost(_)) | None => {
                return Err(EngineError::InvalidInMode {
                    command: CommandKind::Calibration,
                    mode: self.mode.kind(),
                })
            }
        };
        self.after_session_call(before);
        result.map_err(|error| EngineError::Calibration { error })
    }

    fn session_step_kind(&self) -> Option<StepKind> {
        match self.run.as_ref().map(|r| &r.state) {
            Some(RunState::Active(s)) => Some(s.step_kind()),
            Some(RunState::MicLost(_)) | None => None,
        }
    }

    /// The setup rule, the negatives timer and the build spawn after every session call.
    fn after_session_call(&mut self, before: Option<StepKind>) {
        let Some(after_kind) = self.session_step_kind() else {
            return;
        };
        if after_kind != StepKind::Zone {
            self.auto_advance_at = None;
        }
        if after_kind == StepKind::Negatives && before != Some(StepKind::Negatives) {
            if let Some(RunState::Active(s)) = self.run.as_ref().map(|r| &r.state) {
                self.negatives_at = s.negatives_deadline();
            }
        }
        if after_kind != StepKind::Negatives {
            self.negatives_at = None;
        }
        if after_kind == StepKind::ReadyToBuild {
            self.spawn_build();
        }
        self.ensure_setup(false);
    }

    fn calibration_action(
        &mut self,
        kind: CommandKind,
        action: CalibrationAction,
    ) -> Result<Reply, EngineError> {
        self.setup_caller = Some(CommandKind::Calibration);
        let result = self.calibration_step(kind, action);
        self.setup_caller = None;
        // A setup that could not be built ended the run (on_setup_error).
        match self.run_setup_error.take() {
            Some(error) => Err(EngineError::Setup { error }),
            None => result,
        }
    }

    fn calibration_step(
        &mut self,
        kind: CommandKind,
        action: CalibrationAction,
    ) -> Result<Reply, EngineError> {
        if self.run.is_none() {
            return Err(self.invalid(kind));
        }
        match action {
            CalibrationAction::RecordAmbient => {
                if self.config.input_veto_enabled {
                    match &self.veto {
                        VetoStatus::Active => {}
                        VetoStatus::Unavailable { reason } => {
                            return Err(EngineError::Veto {
                                reason: reason.clone(),
                            })
                        }
                        VetoStatus::Disabled
                        | VetoStatus::NotNeeded
                        | VetoStatus::WaitingForShell
                        | VetoStatus::Registering => return Err(EngineError::VetoNotReady),
                    }
                }
                self.with_session(|s, _| s.start_ambient())?;
            }
            CalibrationAction::AcceptAmbient => {
                let layout = self.with_session(|s, _| s.accept_ambient())?;
                if let Some(run) = self.run.as_mut() {
                    run.layout = Some(layout);
                }
                self.ensure_setup(false);
            }
            CalibrationAction::RetryAmbient => {
                self.with_session(|s, _| s.retry_ambient())?;
                if let Some(run) = self.run.as_mut() {
                    run.layout = None;
                }
                self.ensure_setup(false);
            }
            CalibrationAction::UndoLastTap => {
                self.auto_advance_at = None;
                self.with_session(|s, _| s.undo_last())?;
            }
            CalibrationAction::RestartZone => {
                self.auto_advance_at = None;
                self.with_session(|s, _| s.restart_zone())?;
                self.ensure_setup(true);
            }
            CalibrationAction::NextZone => {
                self.auto_advance_at = None;
                self.with_session(|s, now| s.advance(now))?;
            }
            CalibrationAction::SkipNegatives => {
                self.with_session(|s, _| s.finish_negatives())?;
            }
            CalibrationAction::RedoZone { zone } => {
                let index = self
                    .run
                    .as_ref()
                    .and_then(|r| r.zone_ids.iter().position(|z| *z == zone))
                    .ok_or(EngineError::Profile {
                        error: qwerty_core::store::ProfileError::ZoneNotFound { zone },
                    })?;
                // INVARIANT: a run has at most MAX_ZONES zones, far below u16::MAX.
                let index = qwerty_core::units::ZoneIndex(
                    u16::try_from(index).expect("a zone position fits in u16"),
                );
                self.with_session(|s, _| s.redo_zone(index))?;
                self.ensure_setup(true);
            }
            CalibrationAction::Commit { name } => return self.commit(name),
        }
        // Only a setup that could not be built ends the run above; calibration_action then
        // returns that EngineError::Setup instead of this.
        let Some(run) = self.run.as_ref() else {
            return Err(self.invalid(kind));
        };
        Ok(Reply::Calibration {
            run: Box::new(self.run_view(run)),
        })
    }

    fn on_calibration_input(&mut self, input: CalibrationInput) {
        if self.run.is_none() {
            self.counters.stale_messages += 1;
            return;
        }
        let fault = matches!(input, CalibrationInput::InputFault { .. });
        let input = session_input(input);
        match self.with_session(|s, _| s.on_input(input)) {
            Ok(feedback) => {
                if fault {
                    self.counters.faults += 1;
                }
                // XR-UI-02: arm when a zone that is not the last becomes full. The core gives
                // the filling tap `Accepted` and only a later extra tap `ZoneFull`, so both
                // arm; an armed timer is never pushed later.
                let fills = matches!(
                    feedback,
                    TapFeedback::Accepted { .. } | TapFeedback::ZoneFull { .. }
                );
                if fills && self.auto_advance_at.is_none() {
                    let auto = matches!(self.run.as_ref().map(|r| &r.state), Some(RunState::Active(s)) if s.auto_advances());
                    if auto {
                        self.auto_advance_at = Some(after(self.now(), AUTO_ADVANCE_DELAY));
                    }
                }
                self.emit(Event::CalibrationFeedback { feedback });
            }
            Err(e) => {
                tracing::debug!(error = ?std::mem::discriminant(&e), "calibration input refused")
            }
        }
    }

    fn on_ambient(&mut self, chunk: AmbientChunk) {
        let Some(capture) = self.capture.as_ref() else {
            self.counters.stale_messages += 1;
            return;
        };
        let format: AudioFormat = capture.opened.identity.format;
        if self.session_step_kind() != Some(StepKind::Ambient) {
            return;
        }
        let block = AudioBlock {
            format,
            start: chunk.start,
            samples: &chunk.samples,
            time: chunk.time,
            discontinuity: chunk.discontinuity,
        };
        if let Err(e) = self.with_session(|s, _| s.push_ambient(&block)) {
            tracing::debug!(error = ?std::mem::discriminant(&e), "ambient chunk refused");
        }
    }

    fn spawn_build(&mut self) {
        let job = match self.run.as_mut().map(|r| &mut r.state) {
            Some(RunState::Active(s)) => s.take_build_job(),
            Some(RunState::MicLost(_)) | None => return,
        };
        let job = match job {
            Ok(job) => job,
            Err(e) => {
                tracing::error!(error = ?std::mem::discriminant(&e), "take_build_job refused in ReadyToBuild");
                self.counters.faults += 1;
                return;
            }
        };
        let id = self.next_worker_id();
        let cancel = Arc::new(AtomicBool::new(false));
        let tx = self.ch.build_tx.clone();
        let lost = Arc::clone(&self.build_lost);
        let body_cancel = Arc::clone(&cancel);
        let worker = Worker::spawn(
            id,
            ThreadName::Build,
            Box::new(BuildStop(Arc::clone(&cancel))),
            self.ch.exits_tx.clone(),
            move || {
                let progress_tx = tx.clone();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    job.run(&body_cancel, &mut |progress| {
                        drop(progress_tx.try_send(BuildMsg::Progress {
                            worker: id,
                            progress,
                        }));
                    })
                }));
                let result = match result {
                    Ok(r) => r.map(Box::new),
                    Err(payload) => Err(SessionError::BuildPanicked {
                        message: panic_text(payload.as_ref()),
                    }),
                };
                if tx
                    .send_timeout(BuildMsg::Finished { worker: id, result }, LIFECYCLE_SEND)
                    .is_err()
                {
                    lost.fetch_add(1, Ordering::AcqRel);
                }
                ThreadEnd::Finished
            },
        );
        match worker {
            Ok(worker) => {
                self.build = Some(Build {
                    worker,
                    cancel,
                    finished: false,
                })
            }
            Err(e) => {
                drop(e);
                self.finish_build(Err(SessionError::BuildPanicked {
                    message: "the build thread could not be spawned".to_string(),
                }));
            }
        }
    }

    fn on_build(&mut self, msg: BuildMsg) {
        let current = self.build.as_ref().map(|b| b.worker.id());
        match msg {
            BuildMsg::Progress { worker, progress } => {
                if current != Some(worker) {
                    self.counters.stale_messages += 1;
                    return;
                }
                if let Some(RunState::Active(s)) = self.run.as_mut().map(|r| &mut r.state) {
                    s.on_build_progress(progress.clone());
                }
                self.emit(Event::CalibrationProgress { progress });
            }
            BuildMsg::Finished { worker, result } => {
                if current != Some(worker) {
                    self.counters.stale_messages += 1;
                    return;
                }
                if let Some(build) = self.build.as_mut() {
                    build.finished = true;
                }
                self.finish_build(result.map(|m| *m));
            }
        }
    }

    fn finish_build(&mut self, result: Result<TrainedModel, SessionError>) {
        let outcome = match self.run.as_mut().map(|r| &mut r.state) {
            Some(RunState::Active(s)) => s.finish_build(result),
            Some(RunState::MicLost(_)) | None => {
                self.counters.stale_messages += 1;
                return;
            }
        };
        if let Err(SessionError::NotInStep { .. }) = outcome {
            self.counters.stale_messages += 1;
        }
        self.ensure_setup(false);
    }

    fn commit(&mut self, name: String) -> Result<Reply, EngineError> {
        let now = Utc::now();
        let (mut draft, result_parts) = {
            let run = self.run.as_ref().ok_or(EngineError::InvalidInMode {
                command: CommandKind::Calibration,
                mode: self.mode.kind(),
            })?;
            let RunState::Active(session) = &run.state else {
                return Err(self.invalid(CommandKind::Calibration));
            };
            let result = session
                .result()
                .map_err(|error| EngineError::Calibration { error })?;
            (
                run.draft.clone(),
                (
                    result.model.clone(),
                    result.layout.clone(),
                    result.environment,
                    run.device.clone(),
                ),
            )
        };
        let (model, layout, environment, identity) = result_parts;
        draft.name = name;
        draft.updated_at = now;
        draft.calibration = CalibrationState::Calibrated {
            device: qwerty_core::store::DeviceFingerprint { identity, layout },
            model,
            calibrated_at: now,
            environment,
        };
        draft
            .validate()
            .map_err(|error| EngineError::Profile { error })?;
        self.data
            .save_profile(&draft)
            .map_err(|error| EngineError::Persistence { error })?;
        // Success: the session is consumed and its audio dropped.
        // INVARIANT: the run exists (read above).
        let run = self.run.take().expect("the run exists");
        if let RunState::Active(session) = run.state {
            drop(session.commit());
        }
        self.auto_advance_at = None;
        self.negatives_at = None;
        let id = draft.id;
        self.upsert_profile(draft);
        if self.writable_config().is_ok() {
            let mut config = self.config.clone();
            config.active_profile_id = Some(id);
            if let Err(e) = self.save_config(config) {
                tracing::error!("saving the active profile after a commit failed");
                drop(e);
            }
        }
        self.active = Active::Profile(id);
        self.device_attempt = None;
        self.load_active_model();
        self.configured = None;
        self.desired = Desired::Listening;
        self.emit(Event::CalibrationEnded {
            outcome: CalibrationOutcome::Committed { profile: id },
        });
        Ok(Reply::Committed { profile: id })
    }

    fn drop_run_state(&mut self) {
        if let Some(build) = self.build.take() {
            build.cancel.store(true, Ordering::Release);
            stop_worker(build.worker);
        }
        self.run = None;
        self.auto_advance_at = None;
        self.negatives_at = None;
    }

    /// CancelCalibration: drops the run, its audio, its draft and any build; `desired` is kept.
    fn end_run(&mut self) {
        self.drop_run_state();
        self.configured = None;
    }

    // -----------------------------------------------------------------------
    // Evaluation (§2.8)
    // -----------------------------------------------------------------------

    fn start_evaluation(
        &mut self,
        kind: CommandKind,
        prompts_per_zone: u32,
        negatives: Vec<qwerty_core::assessment::NegativePhase>,
    ) -> Result<Reply, EngineError> {
        if self.run.is_some() || self.evaluation.is_some() {
            return Err(self.invalid(kind));
        }
        if self.unsaved_report.is_some() {
            return Err(EngineError::UnsavedReportHeld);
        }
        let profile = self.active_profile().ok_or(EngineError::NoActiveProfile)?;
        let CalibrationState::Calibrated { device, model, .. } = &profile.calibration else {
            return Err(EngineError::ProfileNotCalibrated);
        };
        let endpoint = device.identity.endpoint_id.clone();
        let zone_refs: Vec<ZoneRef> = model
            .zone_ids
            .iter()
            .map(|id| {
                profile
                    .zones
                    .iter()
                    .find(|z| z.id == *id)
                    .map(|z| ZoneRef {
                        id: *id,
                        label: z.label.clone(),
                    })
                    .ok_or(EngineError::Profile {
                        error: qwerty_core::store::ProfileError::ZoneNotFound { zone: *id },
                    })
            })
            .collect::<Result<_, _>>()?;
        if self.config.input_veto_enabled {
            match &self.veto {
                VetoStatus::Active => {}
                VetoStatus::Unavailable { reason } => {
                    return Err(EngineError::Veto {
                        reason: reason.clone(),
                    })
                }
                VetoStatus::Disabled
                | VetoStatus::NotNeeded
                | VetoStatus::WaitingForShell
                | VetoStatus::Registering => return Err(EngineError::VetoNotReady),
            }
        }
        if self.capture.as_ref().map(|c| c.endpoint_id.as_str()) != Some(endpoint.as_str()) {
            self.open(&endpoint, CapturePurpose::Evaluating)
                .map_err(|error| EngineError::Capture { error })?;
            if self.capture.is_none() {
                return Err(EngineError::Capture {
                    error: CaptureError::DeviceNotFound {
                        endpoint_id: endpoint,
                    },
                });
            }
        }
        let now = self.now();
        let session = EvaluationSession::new(
            zone_refs.clone(),
            PromptSchedule::Generated {
                prompts_per_zone,
                seed: now.0 as u64,
                negatives,
            },
            now,
        )
        .map_err(|error| EngineError::Evaluation { error })?;
        self.cancel_tests();
        self.evaluation = Some(Evaluation {
            session,
            zone_refs,
            veto_enabled: self.config.input_veto_enabled,
            prior: self.desired,
            segment: SegmentKind::EvaluationPrompts,
        });
        if let Some(c) = self.capture.as_mut() {
            c.purpose = CapturePurpose::Evaluating;
        }
        self.eval_tick_at = Some(after(now, EVALUATION_TICK));
        self.ensure_setup(true);
        let mut events = Vec::new();
        if let Some(eval) = self.evaluation.as_mut() {
            eval.session.tick(now, &mut events);
        }
        self.on_evaluation_events(events);
        let view =
            self.evaluation
                .as_ref()
                .map(|e| e.session.view())
                .ok_or(EngineError::Evaluation {
                    error: qwerty_core::assessment::AssessmentError::NotFinished,
                })?;
        Ok(Reply::Evaluation { view })
    }

    fn on_evaluation_events(&mut self, events: Vec<EvaluationEvent>) {
        for event in events {
            match &event {
                EvaluationEvent::PromptOpened { zone, .. }
                | EvaluationEvent::PromptReopened { zone, .. } => {
                    let reopened = matches!(event, EvaluationEvent::PromptReopened { .. });
                    let prompt = self.evaluation.as_ref().and_then(|e| {
                        let id = e.zone_refs.get(usize::from(zone.0))?.id;
                        match e.session.view().step {
                            EvaluationStep::Prompt { deadline, .. } => Some((id, deadline)),
                            EvaluationStep::Gap { .. }
                            | EvaluationStep::Negative { .. }
                            | EvaluationStep::Finished => None,
                        }
                    });
                    if let Some((zone, deadline)) = prompt {
                        let opened_at =
                            HostTime(deadline.0 - HostDuration::from_ms(PROMPT_WINDOW).0);
                        self.send_structure(SidecarEvent::Prompt {
                            zone,
                            opened_at,
                            deadline,
                            reopened,
                        });
                    }
                }
                EvaluationEvent::NegativeStarted { class, .. } => {
                    if let Some(e) = self.evaluation.as_mut() {
                        e.segment = SegmentKind::EvaluationNegative {
                            class: class.clone(),
                        };
                    }
                    self.ensure_setup(true);
                }
                EvaluationEvent::NegativeEnded { .. } => {
                    if let Some(e) = self.evaluation.as_mut() {
                        e.segment = SegmentKind::EvaluationPrompts;
                    }
                    self.ensure_setup(true);
                }
                EvaluationEvent::PromptResolved { .. } | EvaluationEvent::Finished => {}
            }
            let finished = matches!(event, EvaluationEvent::Finished);
            self.emit(Event::EvaluationUpdate { event });
            if finished {
                self.finish_evaluation();
            }
        }
    }

    fn finish_evaluation(&mut self) {
        let Some(eval) = self.evaluation.take() else {
            return;
        };
        self.eval_tick_at = None;
        self.desired = eval.prior;
        let Some(profile) = self.active_profile() else {
            return;
        };
        let CalibrationState::Calibrated {
            device,
            model,
            calibrated_at,
            ..
        } = &profile.calibration
        else {
            return;
        };
        let meta = ReportMeta {
            profile_id: profile.id,
            device: device.identity.clone(),
            front_end_version: model.front_end_version,
            model_trained_at: *calibrated_at,
            zone_taps: model.summary.zones.iter().map(|z| z.taps).collect(),
            negatives: model.summary.negatives.clone(),
            background_regime: model.summary.background_regime.clone(),
            input_veto_enabled: eval.veto_enabled,
        };
        let report = match eval.session.finish(meta, Utc::now()) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(error = ?std::mem::discriminant(&e), "a finished evaluation would not report");
                self.counters.faults += 1;
                return;
            }
        };
        match self.data.save_report(&report) {
            Ok(id) => {
                self.refresh_latest_report(report.profile_id);
                self.emit(Event::EvaluationEnded {
                    outcome: EvaluationOutcome::Saved {
                        report: id,
                        overall: report.overall.clone(),
                    },
                });
            }
            Err(error) => {
                tracing::warn!(file = %error.file, kind = ?error.kind, "saving an evaluation report failed");
                self.unsaved_report = Some(Box::new(report.clone()));
                self.emit(Event::EvaluationEnded {
                    outcome: EvaluationOutcome::SaveFailed {
                        report: Box::new(report),
                        error,
                    },
                });
            }
        }
        self.configured = None;
    }
}

/// The build worker's stop signal: the cancel flag the job polls.
struct BuildStop(Arc<AtomicBool>);

impl crate::engine::messages::StopSignal for BuildStop {
    fn signal(&self) {
        self.0.store(true, Ordering::Release);
    }
}

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
    /// a capture reopen). The caller clears the warning itself (`Controller::reset_drift`).
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
