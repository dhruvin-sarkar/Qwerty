//! `Engine`, `EngineHandle`, `EngineOptions` and `EventStream` (p2 §2.9.1): the shell's only way
//! into the engine. `Engine::start_with` runs startup (§2.9.2) and spawns the controller.

use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, SendTimeoutError};
use windows::Win32::System::Com::{CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED};

use qwerty_core::migrate::Loaded;
use qwerty_core::recording::SidecarEvent;
use qwerty_core::store::Config;

use crate::engine::controller::{
    Active, Channels, Controller, Loaded0, ProfileEntry, StartWorkers,
};
use crate::engine::messages::{CommandMsg, OsParts, SelfOutputLink, SpeechEnds, WorkerId};
use crate::engine::os::WindowsParts;
use crate::engine::persist::DataDir;
use crate::engine::protocol::{
    AutostartPolicy, ConfigStatus, DebugCaptureMode, EngineError, Event, Mode, Reply,
    ShutdownReport, Snapshot, ThreadName, REQUEST_TIMEOUT, SHUTDOWN_DEADLINE,
};
use crate::engine::{dispatch, processing, recorder};

/// Channel capacities (§2.11).
const COMMANDS: usize = 64;
const PROCESSING_TO_CONTROLLER: usize = 256;
const DISPATCH_TO_CONTROLLER: usize = 64;
const RAW_INPUT_TO_CONTROLLER: usize = 8;
const RECORDER_TO_CONTROLLER: usize = 16;
const BUILD_TO_CONTROLLER: usize = 64;
const DEVICES_TO_CONTROLLER: usize = 64;
const EXITS: usize = 16;
const EVENTS: usize = 512;
const PROCESSING_CMDS: usize = 32;
const SELF_OUTPUT: usize = 64;
const RECORDER_CMDS: usize = 64;
const FEED_EVENTS: usize = 1024;

/// Explicit, always; there is no default location inside the engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineOptions {
    pub data_dir: PathBuf,
    pub autostart: AutostartPolicy,
    pub debug_capture: DebugCaptureMode,
}

pub struct Engine;

impl Engine {
    /// `start_with(options, Arc::new(WindowsParts::new()))`.
    pub fn start(options: EngineOptions) -> Result<(EngineHandle, EventStream), EngineError> {
        Engine::start_with(options, Arc::new(WindowsParts::new()))
    }

    /// Startup (§2.9.2): returns once the controller has listed devices and reconciled, or with
    /// the first error.
    pub fn start_with(
        options: EngineOptions,
        os: Arc<dyn OsParts>,
    ) -> Result<(EngineHandle, EventStream), EngineError> {
        install_panic_hook();
        let persistence = |error| EngineError::Persistence { error };
        let (events_tx, events_rx) = crossbeam_channel::bounded::<Event>(EVENTS);
        // 1-2. The data folder and the config.
        let data = DataDir::open(options.data_dir.clone()).map_err(persistence)?;
        let load = data.load_config().map_err(persistence)?;
        let mut config: Config = load.config;
        if let ConfigStatus::Migrated { from_version } = load.status {
            send_startup(
                &events_tx,
                Event::ConfigMigrated {
                    from_version,
                    notes: load.notes,
                },
            );
        }
        // 3. The profiles.
        let mut profiles = Vec::new();
        for scan in data.scan_profiles().map_err(persistence)? {
            let loaded = match scan.loaded {
                Ok(Loaded::Current(p)) => Ok(p),
                Ok(Loaded::Migrated {
                    value,
                    from_version,
                    notes,
                }) => {
                    send_startup(
                        &events_tx,
                        Event::ProfileMigrated {
                            id: value.id,
                            from_version,
                            notes,
                        },
                    );
                    Ok(value)
                }
                Err(failure) => {
                    tracing::warn!(file = %failure.file, kind = ?failure.kind, "a profile is unreadable");
                    Err(failure)
                }
            };
            let latest_report = match &loaded {
                Ok(p) => data
                    .list_reports(p.id)
                    .map_err(persistence)?
                    .into_iter()
                    .find(|r| {
                        matches!(
                            r.content,
                            crate::engine::protocol::ReportContent::Readable { .. }
                        )
                    }),
                Err(_) => None,
            };
            profiles.push(ProfileEntry {
                file: scan.file,
                loaded,
                latest_report,
            });
        }
        // 4. The active profile.
        let config_writable = !matches!(load.status, ConfigStatus::Unreadable { .. });
        let active = match config.active_profile_id {
            None => Active::None,
            Some(id) => {
                let file = format!("{}.json", id.as_uuid());
                match profiles.iter().find(|e| e.file == file) {
                    Some(entry) => match &entry.loaded {
                        Ok(_) => Active::Profile(id),
                        Err(_) => Active::Unreadable { file },
                    },
                    None => {
                        config.active_profile_id = None;
                        if config_writable {
                            data.save_config(&config).map_err(persistence)?;
                        }
                        Active::None
                    }
                }
            }
        };
        // 5. Processing, dispatch and the recorder, then the controller.
        let (commands_tx, commands_rx) = crossbeam_channel::bounded::<CommandMsg>(COMMANDS);
        let (processing_tx, processing_rx) = crossbeam_channel::bounded(PROCESSING_TO_CONTROLLER);
        let (dispatch_tx, dispatch_rx) = crossbeam_channel::bounded(DISPATCH_TO_CONTROLLER);
        let (raw_tx, raw_rx) = crossbeam_channel::bounded(RAW_INPUT_TO_CONTROLLER);
        let (recorder_tx, recorder_rx) = crossbeam_channel::bounded(RECORDER_TO_CONTROLLER);
        let (build_tx, build_rx) = crossbeam_channel::bounded(BUILD_TO_CONTROLLER);
        let (devices_tx, devices_rx) = crossbeam_channel::bounded(DEVICES_TO_CONTROLLER);
        let (exits_tx, exits_rx) = crossbeam_channel::bounded(EXITS);
        let (pcmd_tx, pcmd_rx) = crossbeam_channel::bounded(PROCESSING_CMDS);
        let (self_tx, self_rx) = crossbeam_channel::bounded(SELF_OUTPUT);
        let (rcmd_tx, rcmd_rx) = crossbeam_channel::bounded(RECORDER_CMDS);
        let (feed_tx, feed_rx) = crossbeam_channel::bounded::<SidecarEvent>(FEED_EVENTS);
        let speech = Arc::new(SpeechEnds::default());
        let link = Arc::new(SelfOutputLink::new(
            self_tx,
            Arc::clone(&speech),
            Arc::clone(&os),
        ));
        let (processing_worker, processing_thread) = processing::spawn(
            WorkerId(1),
            pcmd_rx,
            processing_tx,
            events_tx.clone(),
            self_rx,
            Arc::clone(&speech),
            Arc::clone(&os),
            exits_tx.clone(),
        )?;
        let sends_dropped = Arc::new(AtomicU64::new(0));
        let (dispatch_worker, queue) = dispatch::spawn(
            WorkerId(2),
            dispatch_tx,
            Arc::clone(&link),
            Arc::clone(&os),
            exits_tx.clone(),
            Arc::clone(&sends_dropped),
        )?;
        let recorder_worker = recorder::spawn(
            WorkerId(3),
            rcmd_rx,
            feed_rx,
            recorder_tx,
            os.recorder_disk(),
            exits_tx.clone(),
            Arc::clone(&sends_dropped),
        )?;
        let channels = Channels {
            commands: commands_rx,
            processing: processing_rx,
            dispatch: dispatch_rx,
            raw_input: raw_rx,
            raw_input_tx: raw_tx,
            recorder: recorder_rx,
            build: build_rx,
            build_tx,
            devices: devices_rx,
            devices_tx,
            exits: exits_rx,
            exits_tx,
            events: events_tx.clone(),
            processing_cmds: pcmd_tx,
            recorder_cmds: rcmd_tx,
            feed_events: feed_tx,
        };
        let workers = StartWorkers {
            processing: processing_worker,
            processing_thread,
            dispatch: dispatch_worker,
            queue,
            recorder: recorder_worker,
            link,
            speech,
            sends_dropped,
            next_worker: 3,
        };
        let loaded = Loaded0 {
            data,
            config,
            config_status: load.status,
            profiles,
            active,
        };
        let slot = Arc::new(Mutex::new(Arc::new(starting_snapshot())));
        let (ready_tx, ready_rx) = crossbeam_channel::bounded::<Result<(), EngineError>>(1);
        let thread_slot = Arc::clone(&slot);
        let thread_events = events_tx;
        let autostart = options.autostart;
        let debug_capture = options.debug_capture;
        let controller = std::thread::Builder::new()
            .name(ThreadName::Controller.os_name().to_string())
            .spawn(move || {
                // SAFETY: COM is initialized once for this thread's life and uninitialized when the
                // controller exits, after the device watch is dropped (RUST-RT-18).
                let com = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
                if com.is_err() {
                    let hresult = com.0;
                    drop(ready_tx.send(Err(EngineError::Capture {
                        error: crate::engine::protocol::CaptureError::ComInit { hresult },
                    })));
                    return;
                }
                run_controller(
                    os,
                    autostart,
                    debug_capture,
                    channels,
                    thread_slot,
                    loaded,
                    workers,
                    ready_tx,
                    thread_events,
                );
                // SAFETY: paired with the successful CoInitializeEx above, on this thread.
                unsafe { CoUninitialize() };
            })
            .map_err(|e| EngineError::ThreadSpawn {
                thread: ThreadName::Controller,
                detail: e.to_string(),
            })?;
        match ready_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                drop(controller.join());
                return Err(error);
            }
            Err(_) => {
                drop(controller.join());
                return Err(EngineError::EngineStopped);
            }
        }
        Ok((
            EngineHandle {
                commands: commands_tx,
                snapshot: slot,
                controller: Arc::new(Mutex::new(Some(controller))),
            },
            EventStream { events: events_rx },
        ))
    }
}

/// A startup event (a migration report) before the stream is handed out: the channel is empty,
/// so it cannot be full.
fn send_startup(events: &crossbeam_channel::Sender<Event>, event: Event) {
    if events.try_send(event).is_err() {
        tracing::error!("a startup event did not fit the empty event channel");
    }
}

/// §2.11 Panic policy: a tracing panic hook logs the thread, the panic location (file:line) and
/// the message, so a worker's panic reaches the log and not only stderr (which the GUI process
/// lacks). Installed once per process; the previous hook still runs after it.
fn install_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let thread = std::thread::current();
            let location = info
                .location()
                .map(|l| format!("{}:{}", l.file(), l.line()))
                .unwrap_or_else(|| "an unknown location".to_string());
            let payload = info.payload();
            let message = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "non-string panic payload".to_string()
            };
            tracing::error!(
                thread = thread.name().unwrap_or("unnamed"),
                %location,
                %message,
                "a thread panicked"
            );
            previous(info);
        }));
    });
}

#[allow(clippy::too_many_arguments)]
fn run_controller(
    os: Arc<dyn OsParts>,
    autostart: AutostartPolicy,
    debug_capture: DebugCaptureMode,
    channels: Channels,
    slot: Arc<Mutex<Arc<Snapshot>>>,
    loaded: Loaded0,
    workers: StartWorkers,
    ready: crossbeam_channel::Sender<Result<(), EngineError>>,
    events: crossbeam_channel::Sender<Event>,
) {
    let mut controller = Controller::new(
        os,
        autostart,
        debug_capture,
        channels,
        slot,
        loaded,
        workers,
    );
    if let Err(error) = controller.init() {
        drop(ready.send(Err(error)));
        controller.after_panic("startup failed".to_string());
        return;
    }
    drop(ready.send(Ok(())));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| controller.run()));
    if let Err(payload) = outcome {
        let message = if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "non-string panic payload".to_string()
        };
        tracing::error!(
            thread = ThreadName::Controller.os_name(),
            "the controller panicked"
        );
        controller.after_panic(message);
    }
    drop(events);
}

/// The snapshot the slot holds before the controller publishes its first one.
pub(crate) fn starting_snapshot() -> Snapshot {
    Snapshot {
        revision: 0,
        mode: Mode::Starting,
        desired: crate::engine::protocol::Desired::Listening,
        config: Config::initial(),
        config_status: ConfigStatus::Initial,
        profiles: Vec::new(),
        active_profile: None,
        devices: Vec::new(),
        capture: crate::engine::protocol::CaptureStatus::Off,
        veto: crate::engine::protocol::VetoStatus::Disabled,
        recorder: crate::engine::protocol::RecorderStatus {
            state: crate::engine::protocol::RecorderState::Off,
            stored: crate::engine::protocol::StoredRecordings { count: 0, bytes: 0 },
        },
        telemetry: crate::engine::protocol::TelemetryLevel::Off,
        health: crate::engine::protocol::Health {
            counters: crate::engine::protocol::EngineCounters {
                counters_since: chrono::Utc::now(),
                taps_fired: 0,
                input_vetoed: 0,
                actions_failed: 0,
                capture_dropped_frames: 0,
                capture_packets: 0,
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
            latency: qwerty_core::assessment::LatencyStats::from_samples(&[], 0),
            warnings: Vec::new(),
        },
        activity: crate::engine::protocol::ActivityLog {
            recent: Vec::new(),
            attention: Vec::new(),
        },
        pending_tests: Vec::new(),
        unsaved_report: None,
        paths: crate::engine::protocol::Paths {
            data_dir: PathBuf::new(),
            logs: PathBuf::new(),
            recordings: PathBuf::new(),
        },
    }
}

/// The shell's handle: commands in, the latest snapshot out.
#[derive(Clone)]
pub struct EngineHandle {
    commands: crossbeam_channel::Sender<CommandMsg>,
    snapshot: Arc<Mutex<Arc<Snapshot>>>,
    controller: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl EngineHandle {
    /// send_timeout(REQUEST_TIMEOUT), then wait ≤ REQUEST_TIMEOUT for the reply; call off the UI
    /// thread. A closed channel is EngineStopped.
    pub fn request(&self, command: crate::engine::protocol::Command) -> Result<Reply, EngineError> {
        let kind = command.kind();
        let timeout = Duration::from_secs_f64(REQUEST_TIMEOUT.get() / 1000.0);
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        match self.commands.send_timeout(
            CommandMsg {
                command,
                reply: reply_tx,
            },
            timeout,
        ) {
            Ok(()) => {}
            Err(SendTimeoutError::Timeout(_)) => {
                return Err(EngineError::Timeout { command: kind })
            }
            Err(SendTimeoutError::Disconnected(_)) => return Err(EngineError::EngineStopped),
        }
        match reply_rx.recv_timeout(timeout) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => Err(EngineError::Timeout { command: kind }),
            Err(RecvTimeoutError::Disconnected) => Err(EngineError::EngineStopped),
        }
    }

    pub fn snapshot(&self) -> Arc<Snapshot> {
        match self.snapshot.lock() {
            Ok(slot) => Arc::clone(&slot),
            Err(poisoned) => Arc::clone(&poisoned.into_inner()),
        }
    }

    /// request(Shutdown).into_shutdown(), then join the controller (≤ SHUTDOWN_DEADLINE + 1 s).
    pub fn shutdown(self) -> Result<ShutdownReport, EngineError> {
        let report = self
            .request(crate::engine::protocol::Command::Shutdown)?
            .into_shutdown()?;
        let handle = match self.controller.lock() {
            Ok(mut slot) => slot.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        };
        if let Some(handle) = handle {
            let limit = Duration::from_secs_f64(SHUTDOWN_DEADLINE.get() / 1000.0 + 1.0);
            let deadline = Instant::now() + limit;
            while !handle.is_finished() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            if handle.is_finished() {
                drop(handle.join());
            } else {
                tracing::error!("the controller did not exit within the shutdown deadline");
            }
        }
        Ok(report)
    }
}

/// The shell's event stream.
pub struct EventStream {
    events: Receiver<Event>,
}

impl EventStream {
    /// A disconnect is EngineStopped: treat it as Stopped.
    pub fn recv(&self) -> Result<Event, EngineError> {
        self.events.recv().map_err(|_| EngineError::EngineStopped)
    }

    pub fn recv_timeout(&self, d: Duration) -> Result<Option<Event>, EngineError> {
        match self.events.recv_timeout(d) {
            Ok(event) => Ok(Some(event)),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(EngineError::EngineStopped),
        }
    }

    pub fn receiver(&self) -> &Receiver<Event> {
        &self.events
    }
}
