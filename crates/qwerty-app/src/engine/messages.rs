//! The engine's internal messages, ids, workers and OS seam (p2 §2.2-§2.8, §2.13). Not serde:
//! nothing here crosses the shell boundary (that is `protocol`).

use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Sender, TrySendError};
use serde::{Deserialize, Serialize};
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::Threading::{CreateEventW, SetEvent};

use qwerty_core::audio::BlockTime;
use qwerty_core::detector::{Detector, DetectorHealth, GateReport, InputFault, RejectedCandidate};
use qwerty_core::gesture::GestureEvent;
use qwerty_core::model::TrainedModel;
use qwerty_core::pipeline::{
    CalibrationFrontEnd, CalibrationInput, Pipeline, PipelineCounters, PipelineError, TapDecision,
};
use qwerty_core::profile::{ActionSpec, ZoneId};
use qwerty_core::recording::{
    RecordingError, RecordingSummary, SegmentKind, SidecarEvent, SidecarHeader,
};
use qwerty_core::session::{BuildProgress, SessionError};
use qwerty_core::units::{CandidateId, HostTime, SampleIndex, Seconds};
use qwerty_core::veto::{InputEvent, OutputId, SelfOutputEvent, SelfOutputKind, VetoCounters};

use crate::engine::protocol::{
    ActionResult, CaptureError, Command, CommandOutcome, DeviceInfo, DeviceState, EngineError,
    JobOrigin, OpenedStream, RecorderStopReason, Reply, StoredRecordings, TapRef, TelemetryLevel,
    ThreadName, VetoUnavailableReason,
};
use crate::engine::ring::{RingReader, RingWriter};
use crate::platform::{OutputObserver, PlatformActions};

// ---------------------------------------------------------------------------
// Workers, ids and the OS seam (§2.2)
// ---------------------------------------------------------------------------

/// One spawned thread instance. Strictly increasing, assigned by the controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkerId(pub u64);

/// One processing setup (§2.5). Strictly increasing, assigned by the controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SetupEpoch(pub u64);

/// Idempotent and non-blocking: SetEvent, PostMessageW, or an AtomicBool + unpark, per thread.
pub trait StopSignal: Send {
    fn signal(&self);
}

/// How often `Worker::stop` polls for the thread to finish.
const STOP_POLL: Duration = Duration::from_millis(5);

pub struct Worker {
    id: WorkerId,
    thread: ThreadName,
    stop: Box<dyn StopSignal>,
    /// None once `stop` ran (joined or detached).
    handle: Option<JoinHandle<()>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StopTimedOut {
    pub thread: ThreadName,
    pub worker: WorkerId,
}

impl Worker {
    /// Runs `body` under catch_unwind(AssertUnwindSafe(..)). A panic payload (&str or String;
    /// anything else becomes "non-string panic payload") becomes ThreadEnd::Panicked. The thread's
    /// last act is exits.try_send(ThreadExit): the exits channel holds 16 > the live worker count,
    /// so it is never full, and Disconnected means the controller is gone and nobody is left to
    /// tell.
    pub fn spawn(
        id: WorkerId,
        thread: ThreadName,
        stop: Box<dyn StopSignal>,
        exits: Sender<ThreadExit>,
        body: impl FnOnce() -> ThreadEnd + Send + 'static,
    ) -> Result<Worker, EngineError> {
        let handle = std::thread::Builder::new()
            .name(thread.os_name().to_string())
            .spawn(move || {
                let end = match catch_unwind(AssertUnwindSafe(body)) {
                    Ok(end) => end,
                    Err(payload) => ThreadEnd::Panicked {
                        message: panic_message(payload.as_ref()),
                    },
                };
                match exits.try_send(ThreadExit {
                    worker: id,
                    thread,
                    end,
                }) {
                    Ok(()) => {}
                    // The controller is gone; nobody is left to tell.
                    Err(TrySendError::Disconnected(_)) => {}
                    Err(TrySendError::Full(_)) => {
                        tracing::error!(
                            thread = thread.os_name(),
                            "the exits channel was full: more live workers than its capacity"
                        );
                    }
                }
            })
            .map_err(|e| EngineError::ThreadSpawn {
                thread,
                detail: e.to_string(),
            })?;
        Ok(Worker {
            id,
            thread,
            stop,
            handle: Some(handle),
        })
    }

    pub fn id(&self) -> WorkerId {
        self.id
    }

    /// signal(); then poll JoinHandle::is_finished every 5 ms up to `limit`, and join only a
    /// finished thread. On timeout the JoinHandle is dropped: the thread is detached and leaks
    /// until it ends.
    pub fn stop(mut self, limit: Duration) -> Result<(), StopTimedOut> {
        self.stop.signal();
        // Taking the handle tells Drop that stop() ran.
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        let deadline = Instant::now() + limit;
        loop {
            if handle.is_finished() {
                // The body runs under catch_unwind, so the thread can only have ended normally; a
                // panic in its last act (the exits send) still means it has ended, which is all
                // stop() promises.
                match handle.join() {
                    Ok(()) | Err(_) => return Ok(()),
                }
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(StopTimedOut {
                    thread: self.thread,
                    worker: self.id,
                });
            }
            std::thread::sleep(STOP_POLL.min(deadline - now));
        }
    }
}

impl Drop for Worker {
    /// signal() unless stop() already ran; never joins.
    fn drop(&mut self) {
        if self.handle.is_some() {
            self.stop.signal();
        }
    }
}

/// The text of a panic payload: &str or String; anything else is "non-string panic payload".
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThreadExit {
    pub worker: WorkerId,
    pub thread: ThreadName,
    pub end: ThreadEnd,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ThreadEnd {
    Finished,
    Capture { exit: CaptureExit },
    Panicked { message: String },
}

/// The engine's only OS seam (RUST-RT-3). `os::WindowsParts` is the product; `testkit::FakeOs`
/// drives the EN tests (§2.11).
pub trait OsParts: Send + Sync + 'static {
    fn now(&self) -> HostTime;
    /// The real wait until `deadline` on this clock (WindowsParts: deadline − now; FakeOs: scaled
    /// by its speed).
    fn timeout_until(&self, deadline: HostTime) -> Duration;
    /// Controller thread (COM MTA).
    fn list_devices(&self) -> Result<Vec<DeviceInfo>, CaptureError>;
    /// `dropped` feeds controller_dropped.
    fn watch_devices(
        &self,
        tx: Sender<DeviceMsg>,
        dropped: Arc<AtomicU64>,
    ) -> Result<DeviceWatch, CaptureError>;
    /// Spawns the capture thread and waits ≤ CAPTURE_OPEN_TIMEOUT for its open result (§2.3).
    fn open_capture(
        &self,
        request: CaptureRequest,
    ) -> Result<(Worker, OpenedCapture), CaptureError>;
    fn spawn_raw_input(&self, request: RawInputRequest) -> Result<Worker, VetoUnavailableReason>;
    /// Called once on the dispatch thread after CoInitializeEx(STA); returns a platform with
    /// `observer` set.
    fn platform(&self, observer: Arc<dyn OutputObserver>) -> Box<dyn PlatformActions>;
    /// The recorder's disk seam (§2.13), so tests control free space and write stalls.
    fn recorder_disk(&self) -> Arc<dyn RecorderDisk>;
}

pub trait RecorderDisk: Send + Sync + 'static {
    /// WindowsParts: GetDiskFreeSpaceExW for `dir`'s volume. FakeOs: `set_free_bytes` (default
    /// u64::MAX).
    fn free_bytes(&self, dir: &Path) -> std::io::Result<u64>;
    /// Called before every `write_block` and `checkpoint`. WindowsParts: no-op. FakeOs: blocks for
    /// `stall_recorder`'s time.
    fn before_write(&self);
}

/// Opaque registration guard; dropping it unregisters. Created and dropped on the controller
/// thread.
pub struct DeviceWatch {
    _registration: Box<dyn Any>,
}

impl DeviceWatch {
    /// Wraps the registration; dropping the watch drops it.
    pub fn new(registration: Box<dyn Any>) -> DeviceWatch {
        DeviceWatch {
            _registration: registration,
        }
    }
}

// ---------------------------------------------------------------------------
// Capture and devices (§2.3, §2.4)
// ---------------------------------------------------------------------------

pub struct CaptureRequest {
    pub worker: WorkerId,
    pub endpoint_id: String,
    /// processing
    pub wake: std::thread::Thread,
    pub exits: Sender<ThreadExit>,
}

pub struct OpenedCapture {
    pub stream: OpenedStream,
    pub ring: RingReader,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PacketMeta {
    pub start: SampleIndex,
    pub frames: u32,
    pub time: BlockTime,
    pub discontinuity: bool,
}

// SPEC: §2.3 shows CaptureExit with Serialize/Deserialize although §2.1.1 calls messages.rs
// "not serde"; the derive is kept as §2.3 writes it (it changes nothing on the wire).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptureExit {
    pub frames: u64,
    pub dropped_frames: u64,
    pub packets: u64,
    pub error: Option<CaptureError>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceMsg {
    Added { id: String },
    Removed { id: String },
    StateChanged { id: String, state: DeviceState },
    DefaultChanged { id: Option<String> },
    PropertyChanged { id: String },
}

// ---------------------------------------------------------------------------
// Controller (§2.8)
// ---------------------------------------------------------------------------

/// reply: bounded(1)
pub struct CommandMsg {
    pub command: Command,
    pub reply: Sender<Result<Reply, EngineError>>,
}

pub enum BuildMsg {
    Progress {
        worker: WorkerId,
        progress: BuildProgress,
    },
    Finished {
        worker: WorkerId,
        result: Result<Box<TrainedModel>, SessionError>,
    },
}

/// Controller-assigned, strictly increasing; maps a DispatchMsg back to (zone, tap, origin).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobKey(pub u64);

// ---------------------------------------------------------------------------
// Processing (§2.5)
// ---------------------------------------------------------------------------

pub enum ProcessingCmd {
    /// Installs a setup. The first block processed under it pushes SidecarEvent::Segment{start:
    /// block.start, kind: segment, reconfigured: true} into the recorder feed, so a segment is
    /// exactly one setup's life (§2.13, XR-P01-16, X01-06): every engine segment boundary is a
    /// re-Configure, so the engine never produces `reconfigured: false`. With
    /// `calibration_started: Some(c)` (set only on the Configure that `StartCalibration` sends,
    /// §2.8), that first block first pushes SidecarEvent::CalibrationStarted{start: block.start,
    /// seed: c.seed, taps_per_zone: c.taps_per_zone}. `ring: Some` replaces the capture ring (sent
    /// with every capture (re)open); None keeps it. `Idle` also drops the ring.
    Configure {
        epoch: SetupEpoch,
        setup: ProcessingSetup,
        segment: SegmentKind,
        calibration_started: Option<CalibrationStart>,
        ring: Option<RingReader>,
    },
    /// A fresh ring per raw-input worker (RUST-RT-8); None = no input events (veto off or not
    /// running).
    SetInputRing {
        inputs: Option<rtrb::Consumer<InputRecord>>,
    },
    SetTelemetry {
        level: TelemetryLevel,
    },
    SetRecorder {
        feed: Option<RecorderFeed>,
    },
}

/// A calibration run's plan for the recording (SidecarEvent::CalibrationStarted), stamped by
/// processing (XR-P01-16).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalibrationStart {
    pub seed: u64,
    pub taps_per_zone: u32,
}

/// Built by the controller (it holds the format from `OpenedStream`) and moved into processing, so
/// construction errors surface on the controller (§2.8). Processing never constructs DSP state.
pub enum ProcessingSetup {
    /// capture off: drops the ring
    Idle,
    /// Calibrating: the core CalibrationFrontEnd (Detector(CalibrationClips), the
    /// DescriptorExtractor and ≤ 8 candidates awaiting the veto, §1.15), the same code replay runs
    /// (RUST-RT-15). forward_ambient ⇒ also send AmbientChunk.
    Calibration {
        front_end: Box<CalibrationFrontEnd>,
        forward_ambient: bool,
    },
    /// Listening / Evaluating / Diagnostics with a calibrated profile (the controller decides
    /// whether a Fire dispatches).
    Pipeline { pipeline: Box<Pipeline> },
    /// Diagnostics without a calibrated profile: Detector(Live, all real channels); candidates
    /// appear only in the inspection log.
    Monitor { detector: Box<Detector> },
}

pub struct ProcessingMsg {
    pub epoch: SetupEpoch,
    pub body: ProcessingBody,
}

pub enum ProcessingBody {
    Decision {
        decision: Box<TapDecision>,
    },
    /// `at` = processing now at emission (for Fire: the decided_at of the triggering decision)
    Gesture {
        event: GestureEvent,
        at: HostTime,
    },
    DetectorRejected {
        rejected: RejectedCandidate,
        at: HostTime,
    },
    /// Monitor setup: a candidate passed the detector gates (no model)
    Detected {
        candidate: CandidateId,
        onset: SampleIndex,
        gates: GateReport,
        at: HostTime,
    },
    /// Calibration setup: exactly what CalibrationSession::on_input consumes (CalibrationInput,
    /// §1.15)
    // SPEC: the merged core defines CalibrationInput twice (pipeline.rs, which the front end
    // emits, and session.rs, which on_input takes; session.rs says the merge replaces its copy
    // with pipeline's). §1.15 places it in pipeline.rs, so that one is used here.
    Calibration {
        input: CalibrationInput,
    },
    /// push_block returned an error; the setup dropped to Idle
    SetupFailed {
        error: PipelineError,
    },
    /// the recorder feed refused this setup's Segment (or CalibrationStarted) event, so the
    /// recording can no longer be replayed faithfully (§2.13)
    SegmentLost,
    AmbientChunk {
        chunk: AmbientChunk,
    },
    Fault {
        fault: InputFault,
    },
    /// every 1 s; applied whatever its epoch
    Health {
        health: ProcessingHealth,
    },
}

/// ~100 ms, capture format.
pub struct AmbientChunk {
    pub start: SampleIndex,
    pub samples: Vec<f32>,
    pub time: BlockTime,
    pub discontinuity: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProcessingHealth {
    /// None in Idle
    pub detector: Option<DetectorHealth>,
    /// Pipeline setup only
    pub pipeline: Option<PipelineCounters>,
    /// the one timeline, lifetime totals
    pub veto: VetoCounters,
    /// closed by close_stuck this second → HealthWarning::SelfOutputStuck
    pub stuck_outputs: Vec<OutputId>,
    /// speech finishes with no open speech output (lifetime; an invariant counter: any increase is
    /// a fault, §2.8)
    pub speech_unmatched: u64,
    /// lifetime totals, so a lost Health loses nothing
    pub dropped: ProcessingDrops,
    /// no packet for ≥ 500 ms while a setup other than Idle is active
    pub stalled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProcessingDrops {
    pub decisions: u64,
    pub messages: u64,
    pub telemetry: u64,
    pub recorder_frames: u64,
    pub recorder_events: u64,
}

// ---------------------------------------------------------------------------
// Dispatch (§2.6)
// ---------------------------------------------------------------------------

pub struct DispatchJob {
    pub key: JobKey,
    pub zone: ZoneId,
    pub actions: Vec<ActionSpec>,
    pub origin: JobOrigin,
    pub tap: Option<TapRef>,
}

/// The dispatch thread's auto-reset job event: `DispatchQueue::try_send` sets it, so the STA's
/// message wait wakes for jobs. Closed on drop.
// SPEC: §2.6 shows DispatchQueue holding "the job event" without a type; this owner of the
// handle is the one addition, so the controller's queue and the dispatch thread's wait share
// it safely (Arc) and it is closed exactly once.
pub struct JobEvent {
    handle: HANDLE,
}

// SAFETY: a Win32 event handle is a process-wide kernel object handle; SetEvent and the waits are
// thread-safe, and the handle is closed only in Drop, after every Arc holder is gone.
unsafe impl Send for JobEvent {}
// SAFETY: as above; &JobEvent only exposes SetEvent and the raw handle for waiting.
unsafe impl Sync for JobEvent {}

impl JobEvent {
    /// An unnamed auto-reset event, initially unset.
    pub fn new() -> windows::core::Result<JobEvent> {
        // SAFETY: no security attributes and no name; the returned handle is owned by JobEvent.
        let handle = unsafe { CreateEventW(None, false, false, None) }?;
        Ok(JobEvent { handle })
    }

    /// The raw handle, for the dispatch thread's MsgWaitForMultipleObjectsEx.
    pub fn handle(&self) -> HANDLE {
        self.handle
    }

    pub fn set(&self) {
        // SAFETY: `handle` is a valid event handle until Drop.
        let set = unsafe { SetEvent(self.handle) };
        // INVARIANT: SetEvent fails only on an invalid handle, and this one stays open until Drop.
        set.expect("SetEvent on an open event handle cannot fail");
    }
}

impl Drop for JobEvent {
    fn drop(&mut self) {
        // SAFETY: the handle was created in `new` and is closed only here.
        if let Err(e) = unsafe { CloseHandle(self.handle) } {
            tracing::error!(
                hresult = e.code().0,
                "CloseHandle of the dispatch job event failed"
            );
        }
    }
}

/// try_send, then SetEvent(job event), so the STA's message wait wakes for jobs.
pub struct DispatchQueue {
    /// bounded 16
    jobs: Sender<DispatchJob>,
    event: Arc<JobEvent>,
}

impl DispatchQueue {
    pub fn new(jobs: Sender<DispatchJob>, event: Arc<JobEvent>) -> DispatchQueue {
        DispatchQueue { jobs, event }
    }

    pub fn try_send(&self, job: DispatchJob) -> Result<(), TrySendError<DispatchJob>> {
        self.jobs.try_send(job)?;
        self.event.set();
        Ok(())
    }
}

pub enum DispatchMsg {
    PlatformReady,
    Completed {
        key: JobKey,
        result: ActionResult,
    },
    CommandFinished {
        key: JobKey,
        step: u32,
        outcome: CommandOutcome,
    },
    PlatformRebuilt {
        panic: String,
    },
}

/// The engine's platform::OutputObserver.
pub struct SelfOutputLink {
    /// bounded 64, to processing
    began: Sender<SelfOutputEvent>,
    next_id: AtomicU64,
    speech: Arc<SpeechEnds>,
    os: Arc<dyn OsParts>,
}

impl SelfOutputLink {
    pub fn new(
        began: Sender<SelfOutputEvent>,
        speech: Arc<SpeechEnds>,
        os: Arc<dyn OsParts>,
    ) -> SelfOutputLink {
        SelfOutputLink {
            began,
            next_id: AtomicU64::new(0),
            speech,
            os,
        }
    }

    /// Blocking send to processing, which drains every ≤ 50 ms.
    fn send(&self, event: SelfOutputEvent) {
        if self.began.send(event).is_err() {
            // Disconnected: processing has stopped (shutdown or fault), so no veto is left to
            // inform.
            tracing::debug!("self-output event not delivered: processing has stopped");
        }
    }
}

impl OutputObserver for SelfOutputLink {
    fn output_started(&self, kind: SelfOutputKind) -> OutputId {
        let id = OutputId(self.next_id.fetch_add(1, Ordering::Relaxed));
        self.send(SelfOutputEvent::Began {
            id,
            kind,
            at: self.os.now(),
        });
        id
    }

    fn output_ended(&self, id: OutputId) {
        self.send(SelfOutputEvent::Ended {
            id,
            at: self.os.now(),
        });
    }

    /// Under catch_unwind: last_at.store(now, Relaxed), then finished.fetch_add(1, Release). No
    /// lock, allocation or channel, so it is safe inside tts's locks; a caught panic increments
    /// callback_panics.
    fn speech_finished(&self) {
        let speech = &self.speech;
        let os = &self.os;
        let stored = catch_unwind(AssertUnwindSafe(|| {
            speech.last_at.store(os.now().0, Ordering::Relaxed);
            speech.finished.fetch_add(1, Ordering::Release);
        }));
        match stored {
            Ok(()) => {}
            Err(_) => {
                speech.callback_panics.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct SpeechEnds {
    pub finished: AtomicU64,
    pub last_at: AtomicI64,
    pub callback_panics: AtomicU64,
}

// ---------------------------------------------------------------------------
// Raw input (§2.7)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputRecord {
    Event(InputEvent),
    Gap { from: HostTime, to: HostTime },
}

pub struct RawInputRequest {
    pub worker: WorkerId,
    pub ring: rtrb::Producer<InputRecord>,
    pub tx: Sender<RawInputMsg>,
    pub exits: Sender<ThreadExit>,
}

/// skipped_records: lifetime count of WM_INPUT records skipped for a larger required size. Sent
/// with every status message and on the 5 s WM_TIMER whenever it changed, never once per event.
#[derive(Debug, Clone, PartialEq)]
pub struct RawInputMsg {
    pub worker: WorkerId,
    pub status: RawInputStatus,
    pub skipped_records: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RawInputStatus {
    Active,
    Unavailable { reason: VetoUnavailableReason },
}

// ---------------------------------------------------------------------------
// Recorder (§2.13)
// ---------------------------------------------------------------------------

pub struct RecorderFeed {
    /// audio_ring(format, RECORDER_RING, 2048, buffer_frames)
    pub ring: RingWriter,
    /// bounded 1024
    pub events: Sender<SidecarEvent>,
}

pub enum RecorderCmd {
    /// format = header.device.format
    Start {
        root: PathBuf,
        /// Boxed by §2.1.1's boxing rule (large_enum_variant).
        header: Box<SidecarHeader>,
        ring: RingReader,
    },
    /// Structure events from the controller: Prompt (RUST-RT-32); Segments and CalibrationStarted
    /// come from processing (XR-P01-16).
    Event {
        event: SidecarEvent,
    },
    Stop {
        reason: RecorderStopReason,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecorderMsg {
    Started {
        dir: String,
    },
    Progress {
        bytes: u64,
        duration: Seconds,
        gaps: u32,
        dropped_events: u32,
    },
    Stopped {
        summary: RecordingSummary,
        reason: RecorderStopReason,
    },
    Failed {
        error: RecordingError,
    },
    Stored {
        stored: StoredRecordings,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    struct FlagStop(Arc<AtomicBool>);

    impl StopSignal for FlagStop {
        fn signal(&self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn worker_reports_panics_and_stops_on_signal() {
        let (exits_tx, exits_rx) = crossbeam_channel::bounded(16);

        let worker = Worker::spawn(
            WorkerId(1),
            ThreadName::Build,
            Box::new(FlagStop(Arc::new(AtomicBool::new(false)))),
            exits_tx.clone(),
            || panic!("boom"),
        )
        .expect("spawn");
        let exit = exits_rx.recv_timeout(Duration::from_secs(5)).expect("exit");
        assert_eq!(exit.worker, WorkerId(1));
        assert_eq!(
            exit.end,
            ThreadEnd::Panicked {
                message: "boom".to_string()
            }
        );
        worker.stop(Duration::from_secs(1)).expect("joined");

        let flag = Arc::new(AtomicBool::new(false));
        let seen = Arc::clone(&flag);
        let worker = Worker::spawn(
            WorkerId(2),
            ThreadName::Recorder,
            Box::new(FlagStop(Arc::clone(&flag))),
            exits_tx,
            move || {
                while !seen.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                ThreadEnd::Finished
            },
        )
        .expect("spawn");
        assert_eq!(worker.id(), WorkerId(2));
        worker.stop(Duration::from_secs(2)).expect("joined");
        let exit = exits_rx.recv_timeout(Duration::from_secs(5)).expect("exit");
        assert_eq!(exit.end, ThreadEnd::Finished);
    }
}
