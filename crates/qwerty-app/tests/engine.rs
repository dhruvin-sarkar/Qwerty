//! The engine tests (p2 §2.11): EN2-EN8, EN10-EN15, EN17, RC3 and RP2, plus shell-v2's DN1
//! through the engine's commands. Each runs the real
//! controller, processing, dispatch and recorder through `Engine::start_with(options,
//! Arc::new(FakeOs))`, on a temporary data folder.
//!
//! None asserts which zone a scripted tap was classified as: the desk fixture is never an
//! accuracy gate (§1.19). Scripted taps are ≥ 310 ms apart, outside the gesture refractory and
//! burst hold (§1.12).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use qwerty_app::engine::protocol::{
    ActionResult, ActivityEntry, AutostartPolicy, Blocker, CalibrationAction, CalibrationTarget,
    CaptureError, CapturePurpose, CaptureStatus, Command, DebugCaptureMode, DeviceChoice,
    DeviceInfo, DeviceState, EngineError, Event, HealthWarning, Mode, PersistenceFailureKind,
    ProbeOutcome, RecorderState, RecorderStopReason, Reply, ReplyKind, RetryState, RunSession,
    SkipReason, Snapshot, StepOutcome, TapRef, TapReport, TelemetryLevel, ThreadName, VetoStatus,
    ZoneDraft,
};
use qwerty_app::engine::testkit::{desk_device, CaptureScript, FakeOs, InputScript, ScriptedTap};
use qwerty_app::engine::{Engine, EngineHandle, EngineOptions, EventStream};
use qwerty_app::platform::ActionError;
use qwerty_core::assessment::LatencyStats;
use qwerty_core::audio::AudioFormat;
use qwerty_core::desk::{ModalDesk, TapSpec};
use qwerty_core::profile::{ActionSpec, ZoneId, ZoneLayout};
use qwerty_core::session::CalibrationStep;
use qwerty_core::store::{ProfileError, ZoneEdit};
use qwerty_core::units::{ChannelCount, Decibels, HostTime, Milliseconds, SampleRate, ZoneIndex};

/// Fake time runs this many times real time.
const SPEED: f64 = 2.0;
/// The endpoint every test calibrates on.
const MIC: &str = "mic-desk";
/// Real-time limit for any one wait.
const WAIT: Duration = Duration::from_secs(90);
/// The minimum calibration (MIN_TAPS_PER_ZONE).
const TAPS_PER_ZONE: u32 = 5;
const ZONES: usize = 4;
/// Spacing of scripted taps (≥ 310 ms, §2.11).
const TAP_GAP_MS: f64 = 400.0;

fn format() -> AudioFormat {
    AudioFormat {
        rate: SampleRate::new(48_000).expect("rate"),
        channels: ChannelCount::new(2).expect("channels"),
    }
}

fn desk_script() -> CaptureScript {
    CaptureScript::Desk {
        format: format(),
        period: Milliseconds::new(10.0),
        ambient: Decibels::new(-80.0),
        seed: 11,
        taps: Vec::new(),
    }
}

fn tap_on(zone: usize) -> TapSpec {
    let (x, y) = ModalDesk::zone_centres(ZONES)[zone];
    TapSpec {
        x,
        y,
        force: Decibels::new(-20.0),
        contact: Milliseconds::new(1.0),
        position_jitter_m: 0.01,
    }
}

/// `n` taps on `zone`, the first `first_ms` from now, then `TAP_GAP_MS` apart.
fn taps(zone: usize, n: usize, first_ms: f64) -> Vec<ScriptedTap> {
    (0..n)
        .map(|i| ScriptedTap {
            at: Milliseconds::new(first_ms + TAP_GAP_MS * i as f64),
            tap: tap_on(zone),
        })
        .collect()
}

fn drafts() -> Vec<ZoneDraft> {
    ModalDesk::zone_centres(ZONES)
        .iter()
        .enumerate()
        .map(|(i, (x, y))| ZoneDraft {
            label: format!("Zone {}", i + 1),
            layout: ZoneLayout {
                x: (*x / 1.2 - 0.05) as f32,
                y: (*y / 0.6 - 0.05) as f32,
                width: 0.1,
                height: 0.1,
            },
        })
        .collect()
}

/// The reply kind §2.9's Command → Reply table gives each command (EN8).
fn expected_reply(command: &Command) -> ReplyKind {
    match command {
        Command::GetSnapshot => ReplyKind::Snapshot,
        Command::ListDevices => ReplyKind::Devices,
        Command::ExportProfile { .. }
        | Command::ExportEvaluation { .. }
        | Command::ExportUnsavedReport { .. }
        | Command::ExportInspection { .. }
        | Command::SaveLatencyReport => ReplyKind::Saved,
        Command::DuplicateProfile { .. } | Command::ImportProfile { .. } => {
            ReplyKind::ProfileCreated
        }
        Command::PreviewImport { .. } => ReplyKind::ImportPreview,
        Command::StartCalibration { .. } => ReplyKind::Calibration,
        Command::Calibration { action } => match action {
            CalibrationAction::Commit { .. } => ReplyKind::Committed,
            CalibrationAction::RecordAmbient
            | CalibrationAction::AcceptAmbient
            | CalibrationAction::RetryAmbient
            | CalibrationAction::UndoLastTap
            | CalibrationAction::RestartZone
            | CalibrationAction::NextZone
            | CalibrationAction::SkipNegatives
            | CalibrationAction::RedoZone { .. } => ReplyKind::Calibration,
        },
        Command::StartEvaluation { .. } => ReplyKind::Evaluation,
        Command::ListEvaluations { .. } => ReplyKind::Evaluations,
        Command::GetEvaluation { .. } => ReplyKind::EvaluationReport,
        Command::TestAction { .. } | Command::TestChain { .. } => ReplyKind::TestScheduled,
        Command::GetInspection => ReplyKind::Inspection,
        Command::GetSoundMap => ReplyKind::SoundMap,
        Command::Shutdown => ReplyKind::Shutdown,
        Command::ShellReady
        | Command::SetListening { .. }
        | Command::ToggleListening
        | Command::SetTelemetry { .. }
        | Command::RetryCapture
        | Command::RetryInputVeto
        | Command::UpdateConfig { .. }
        | Command::ResetConfig
        | Command::ReconcileAutostart
        | Command::SwitchProfile { .. }
        | Command::RenameProfile { .. }
        | Command::DeleteProfile { .. }
        | Command::EditZone { .. }
        | Command::RebindDevice { .. }
        | Command::CancelCalibration
        | Command::UndoLastPrompt
        | Command::CancelEvaluation
        | Command::RetrySaveReport
        | Command::DiscardUnsavedReport
        | Command::DeleteEvaluation { .. }
        | Command::CancelTest { .. }
        | Command::DismissAttention { .. }
        | Command::ShellToast
        | Command::MarkSegment { .. }
        | Command::DeleteDebugRecordings => ReplyKind::Ack,
    }
}

/// A unique temporary data folder, removed by `Rig::finish`.
fn temp_dir(name: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "qwerty-engine-{name}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("remove a stale temp dir");
    }
    dir
}

/// How many engines may run at once in this binary. Each runs its processing thread on fake
/// audio at `SPEED` × real time; with all 16 tests at once on a loaded machine processing fell
/// behind, the capture ring overflowed, and every overflow is a discontinuity that restarts the
/// detector (hundreds of InputFaults, no candidate at all), so scripted taps were never seen.
fn engine_permits() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get() / 3)
        .unwrap_or(2)
        .clamp(2, 6)
}

/// A slot among `engine_permits()`, held for a rig's whole life.
struct Permit;

static RUNNING: Mutex<usize> = Mutex::new(0);
static FREED: std::sync::Condvar = std::sync::Condvar::new();

impl Permit {
    fn take() -> Permit {
        let limit = engine_permits();
        let mut running = RUNNING.lock().unwrap_or_else(|p| p.into_inner());
        while *running >= limit {
            running = FREED.wait(running).unwrap_or_else(|p| p.into_inner());
        }
        *running += 1;
        Permit
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        let mut running = RUNNING.lock().unwrap_or_else(|p| p.into_inner());
        *running -= 1;
        FREED.notify_one();
    }
}

/// One engine on a fake OS, with its events collected on a thread.
struct Rig {
    _permit: Permit,
    os: Arc<FakeOs>,
    handle: EngineHandle,
    events: Arc<Mutex<Vec<Event>>>,
    collector: Option<JoinHandle<()>>,
    dir: PathBuf,
    t0: Instant,
}

struct RigOptions {
    debug_capture: DebugCaptureMode,
    devices: Vec<(DeviceInfo, CaptureScript)>,
}

impl RigOptions {
    fn desk() -> RigOptions {
        RigOptions {
            debug_capture: DebugCaptureMode::FromConfig,
            devices: vec![(desk_device(MIC, "Desk mic", true, format()), desk_script())],
        }
    }
}

impl Rig {
    fn new(name: &str) -> Rig {
        Rig::with(name, RigOptions::desk())
    }

    fn with(name: &str, options: RigOptions) -> Rig {
        Rig::start(temp_dir(name), Arc::new(FakeOs::new(SPEED)), options)
    }

    fn start(dir: PathBuf, os: Arc<FakeOs>, options: RigOptions) -> Rig {
        let permit = Permit::take();
        for (info, script) in options.devices {
            os.add_device(info, script);
        }
        let (handle, stream) = Engine::start_with(
            EngineOptions {
                data_dir: dir.clone(),
                autostart: AutostartPolicy::Untouched,
                debug_capture: options.debug_capture,
            },
            Arc::clone(&os) as Arc<dyn qwerty_app::engine::messages::OsParts>,
        )
        .expect("the engine starts");
        let events = Arc::new(Mutex::new(Vec::new()));
        let collector = Some(collect(stream, Arc::clone(&events)));
        let rig = Rig {
            _permit: permit,
            os,
            handle,
            events,
            collector,
            dir,
            t0: Instant::now(),
        };
        rig.ok(Command::ShellReady);
        rig
    }

    /// A request whose Ok reply must match the Command → Reply table.
    fn request(&self, command: Command) -> Result<Reply, EngineError> {
        let expected = expected_reply(&command);
        let reply = self.handle.request(command)?;
        assert_eq!(reply.kind(), expected, "the reply kind breaks the table");
        Ok(reply)
    }

    fn ok(&self, command: Command) -> Reply {
        let text = format!("{command:?}");
        match self.request(command) {
            Ok(reply) => reply,
            Err(e) => panic!("{text} failed: {e:?}"),
        }
    }

    fn snapshot(&self) -> Arc<Snapshot> {
        self.handle.snapshot()
    }

    /// Polls the snapshot until `pred` holds.
    fn wait(&self, what: &str, pred: impl Fn(&Snapshot) -> bool) -> Arc<Snapshot> {
        let deadline = Instant::now() + WAIT;
        loop {
            let s = self.snapshot();
            if pred(&s) {
                eprintln!("[{:?}] {what}", self.t0.elapsed());
                return s;
            }
            if Instant::now() > deadline {
                panic!(
                    "timed out waiting for {what}; mode {:?}, veto {:?}\n{}",
                    s.mode,
                    s.veto,
                    self.diagnosis()
                );
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Waits until the collected events satisfy `pred`; returns them.
    fn wait_events(&self, what: &str, pred: impl Fn(&[Event]) -> bool) -> Vec<Event> {
        let deadline = Instant::now() + WAIT;
        loop {
            let events = self.events();
            if pred(&events) {
                return events;
            }
            if Instant::now() > deadline {
                panic!(
                    "timed out waiting for {what}; mode {:?}\n{}",
                    self.snapshot().mode,
                    self.diagnosis()
                );
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn events(&self) -> Vec<Event> {
        self.events.lock().expect("events lock").clone()
    }

    /// What became of the scripted taps, for a timeout's message: the counters, the rejections
    /// and the newest inspection entries.
    fn diagnosis(&self) -> String {
        let s = self.snapshot();
        let rejected: Vec<String> = self
            .events()
            .iter()
            .filter_map(|e| match e {
                Event::TapRejected { rejected, .. } => Some(format!("{rejected:?}")),
                Event::CalibrationFeedback { feedback } => Some(format!("{feedback:?}")),
                _ => None,
            })
            .collect();
        let inspection = match self.handle.request(Command::GetInspection) {
            Ok(Reply::Inspection { entries }) => entries
                .iter()
                .rev()
                .take(12)
                .map(|e| format!("  {:?} {:?}", e.onset, e.outcome))
                .collect::<Vec<_>>()
                .join("\n"),
            other => format!("  no inspection: {other:?}"),
        };
        format!(
            "counters {:?}\nwarnings {:?}\ncapture {:?}\nrejected {rejected:?}\ninspection (newest first):\n{inspection}",
            s.health.counters, s.health.warnings, s.capture
        )
    }

    fn fired(&self) -> Vec<TapReport> {
        self.events()
            .into_iter()
            .filter_map(|e| match e {
                Event::TapFired { tap } => Some(tap),
                _ => None,
            })
            .collect()
    }

    fn completed(&self) -> Vec<(ZoneId, Option<TapRef>, ActionResult)> {
        self.events()
            .into_iter()
            .filter_map(|e| match e {
                Event::ActionCompleted {
                    zone, tap, result, ..
                } => Some((zone, tap, result)),
                _ => None,
            })
            .collect()
    }

    /// The full calibration choreography of EN2; returns once a snapshot shows Listening.
    fn calibrate(&self, name: &str) {
        self.calibrate_to_build(name);
        self.wait("Review", |s| {
            matches!(step(s), Some(CalibrationStep::Review { .. }))
        });
        self.ok(Command::Calibration {
            action: CalibrationAction::Commit {
                name: name.to_string(),
            },
        });
        self.wait("Listening", |s| s.mode == Mode::Listening);
    }

    /// Starts a run for a new profile on the default mic and waits for its Ready step.
    fn start_run(&self, name: &str) {
        self.start_run_on(CalibrationTarget::New {
            name: name.to_string(),
        });
    }

    /// Starts a run for `target` on the default mic and waits for its Ready step.
    fn start_run_on(&self, target: CalibrationTarget) {
        self.ok(Command::StartCalibration {
            target,
            zones: drafts(),
            device: DeviceChoice::SystemDefault,
            taps_per_zone: TAPS_PER_ZONE,
        });
        // RecordAmbient needs the input veto running (VetoNotReady otherwise).
        self.wait("Ready with the veto", |s| {
            matches!(step(s), Some(CalibrationStep::Ready))
                && matches!(s.veto, VetoStatus::Active | VetoStatus::Disabled)
        });
    }

    /// EN2's choreography up to and including SkipNegatives (the build then starts).
    fn calibrate_to_build(&self, name: &str) {
        self.calibrate_to_build_on(CalibrationTarget::New {
            name: name.to_string(),
        });
    }

    /// `calibrate_to_build` for any target.
    fn calibrate_to_build_on(&self, target: CalibrationTarget) {
        self.start_run_on(target);
        self.ok(Command::Calibration {
            action: CalibrationAction::RecordAmbient,
        });
        self.wait("AmbientReview", |s| {
            matches!(step(s), Some(CalibrationStep::AmbientReview { .. }))
        });
        self.ok(Command::Calibration {
            action: CalibrationAction::AcceptAmbient,
        });
        for zone in 0..ZONES {
            let index = ZoneIndex(zone as u16);
            self.wait(
                "the zone step",
                |s| matches!(step(s), Some(CalibrationStep::Zone { index: i, .. }) if i == index),
            );
            self.os
                .release_taps(MIC, taps(zone, TAPS_PER_ZONE as usize + 1, TAP_GAP_MS));
            let last = zone + 1 == ZONES;
            self.wait("the zone to fill", |s| match step(s) {
                Some(CalibrationStep::Zone {
                    index: i,
                    accepted,
                    target,
                }) => i.0 as usize > zone || (last && accepted >= target),
                _ => false,
            });
        }
        self.ok(Command::Calibration {
            action: CalibrationAction::SkipNegatives,
        });
    }

    /// The active profile's zone ids, in order.
    fn zone_ids(&self) -> Vec<ZoneId> {
        self.snapshot()
            .active_profile
            .as_ref()
            .expect("an active profile")
            .zones
            .iter()
            .map(|z| z.id)
            .collect()
    }

    fn set_actions(&self, zone: ZoneId, actions: Vec<ActionSpec>) {
        self.ok(Command::EditZone {
            zone,
            edit: ZoneEdit {
                label: None,
                layout: None,
                trigger: None,
                actions: Some(actions),
            },
        });
    }

    fn profile_files(&self) -> usize {
        match std::fs::read_dir(self.dir.join("Profiles")) {
            Ok(entries) => entries
                .filter_map(Result::ok)
                .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
                .count(),
            Err(e) => panic!("Profiles is unreadable: {e}"),
        }
    }

    /// Shuts the engine down, joins the collector and removes the data folder.
    fn finish(mut self) -> qwerty_app::engine::protocol::ShutdownReport {
        let report = self.handle.clone().shutdown().expect("shutdown");
        if let Some(c) = self.collector.take() {
            c.join().expect("the collector ends");
        }
        remove_dir(&self.dir);
        report
    }
}

fn remove_dir(dir: &Path) {
    if dir.exists() {
        if let Err(e) = std::fs::remove_dir_all(dir) {
            panic!("the temp data folder could not be removed: {e}");
        }
    }
}

fn collect(stream: EventStream, into: Arc<Mutex<Vec<Event>>>) -> JoinHandle<()> {
    std::thread::spawn(move || {
        while let Ok(event) = stream.recv() {
            into.lock().expect("events lock").push(event);
        }
    })
}

fn step(s: &Snapshot) -> Option<CalibrationStep> {
    match &s.mode {
        Mode::Calibrating { run } => match &run.session {
            RunSession::Active { view } => Some(view.step.clone()),
            RunSession::MicLost { .. } => None,
        },
        _ => None,
    }
}

fn copy(text: &str) -> ActionSpec {
    ActionSpec::CopyToClipboard {
        text: text.to_string(),
    }
}

fn ms_between(later: HostTime, earlier: HostTime) -> f64 {
    later.since(earlier).0 as f64 / 10_000.0
}

/// Calibrates, gives every zone a one-copy chain, releases one tap per zone and returns the
/// fired taps (EN2's listening phase).
fn fire_one_per_zone(rig: &Rig) -> Vec<TapReport> {
    let ids = rig.zone_ids();
    for (i, id) in ids.iter().enumerate() {
        rig.set_actions(*id, vec![copy(&format!("chain {i}"))]);
    }
    let before = rig.fired().len();
    for zone in 0..ZONES {
        rig.os
            .release_taps(MIC, taps(zone, 1, TAP_GAP_MS * (zone as f64 + 1.0)));
    }
    rig.wait_events("the taps to fire", |e| {
        e.iter()
            .filter(|e| matches!(e, Event::TapFired { .. }))
            .count()
            > before
    });
    // Let the rest of the scripted taps land.
    std::thread::sleep(Duration::from_secs_f64(
        TAP_GAP_MS * (ZONES as f64 + 2.0) / 1000.0 / SPEED,
    ));
    rig.fired()[before..].to_vec()
}

#[test]
fn en2_lifecycle() {
    let rig = Rig::new("en2");
    let first = rig.wait("Idle", |s| s.mode == Mode::Idle);
    assert_eq!(first.capture, CaptureStatus::Off);
    assert_eq!(first.telemetry, TelemetryLevel::Off);

    rig.calibrate("Desk");
    assert_eq!(rig.profile_files(), 1);

    // Taps fire their chains in order.
    let fired = fire_one_per_zone(&rig);
    assert!(!fired.is_empty(), "no scripted tap fired");
    let ids = rig.zone_ids();
    rig.wait_events("every chain to complete", |e| {
        e.iter()
            .filter(|e| matches!(e, Event::ActionCompleted { .. }))
            .count()
            >= fired.len()
    });
    let log: Vec<ActionSpec> = rig
        .os
        .platform_log()
        .into_iter()
        .map(|c| c.action)
        .collect();
    let expected: Vec<ActionSpec> = fired
        .iter()
        .map(|t| {
            let i = ids.iter().position(|z| *z == t.zone).expect("a known zone");
            copy(&format!("chain {i}"))
        })
        .collect();
    assert_eq!(log, expected, "the chains ran in the order the taps fired");

    // Toggle: Paused with capture off; Diagnostics telemetry opens the bound endpoint.
    rig.ok(Command::ToggleListening);
    let paused = rig.wait("Paused", |s| s.mode == Mode::Paused);
    assert_eq!(paused.capture, CaptureStatus::Off);
    rig.ok(Command::SetTelemetry {
        level: TelemetryLevel::Diagnostics,
    });
    let diag = rig.wait("Diagnostics capture", |s| {
        matches!(
            &s.capture,
            CaptureStatus::Running { purpose: CapturePurpose::Diagnostics, opened }
                if opened.identity.endpoint_id == MIC
        )
    });
    assert_eq!(diag.mode, Mode::Paused);
    rig.ok(Command::SetTelemetry {
        level: TelemetryLevel::Off,
    });
    rig.wait("capture off", |s| s.capture == CaptureStatus::Off);
    rig.ok(Command::ToggleListening);
    rig.wait("Listening", |s| s.mode == Mode::Listening);

    // A cancelled wizard resumes Listening and leaves no profile file.
    rig.ok(Command::StartCalibration {
        target: CalibrationTarget::New {
            name: "Cancelled".to_string(),
        },
        zones: drafts(),
        device: DeviceChoice::SystemDefault,
        taps_per_zone: TAPS_PER_ZONE,
    });
    rig.wait("Calibrating", |s| {
        matches!(s.mode, Mode::Calibrating { .. })
    });
    rig.ok(Command::CancelCalibration);
    rig.wait("Listening", |s| s.mode == Mode::Listening);
    assert_eq!(rig.profile_files(), 1);

    // An evaluation resumes Listening.
    rig.ok(Command::StartEvaluation {
        prompts_per_zone: 2,
        negatives: Vec::new(),
    });
    rig.wait("Evaluating", |s| matches!(s.mode, Mode::Evaluating { .. }));
    rig.ok(Command::CancelEvaluation);
    rig.wait("Listening", |s| s.mode == Mode::Listening);

    // Unplugging the mic blocks with a scheduled retry; plugging it back resumes.
    rig.os.remove_device(MIC);
    rig.wait("MicUnavailable", |s| {
        matches!(
            &s.mode,
            Mode::Blocked {
                blocker: Blocker::MicUnavailable { .. }
            }
        ) && matches!(
            &s.capture,
            CaptureStatus::Failed {
                retry: RetryState::Scheduled { .. },
                ..
            }
        )
    });
    rig.os
        .add_device(desk_device(MIC, "Desk mic", true, format()), desk_script());
    rig.wait("Listening again", |s| s.mode == Mode::Listening);
    rig.finish();
}

/// The snapshot counters, now.
fn counters(rig: &Rig) -> qwerty_app::engine::protocol::EngineCounters {
    rig.snapshot().health.counters.clone()
}

/// Two-step chains [CopyToClipboard, OpenUrl] on every zone, distinct per zone.
fn two_step_chains(rig: &Rig) {
    for (i, id) in rig.zone_ids().iter().enumerate() {
        rig.set_actions(
            *id,
            vec![
                copy(&format!("chain {i}")),
                ActionSpec::OpenUrl {
                    url: format!("https://example.invalid/{i}"),
                },
            ],
        );
    }
}

fn fired_count(e: &[Event]) -> usize {
    e.iter()
        .filter(|e| matches!(e, Event::TapFired { .. }))
        .count()
}

fn completed_count(e: &[Event]) -> usize {
    e.iter()
        .filter(|e| matches!(e, Event::ActionCompleted { .. }))
        .count()
}

/// Releases two taps `gap_ms` apart on zones 0 and 1 and waits for at least `want` new fires.
fn two_taps(rig: &Rig, gap_ms: f64) -> usize {
    let before = fired_count(&rig.events());
    rig.os.release_taps(
        MIC,
        vec![
            ScriptedTap {
                at: Milliseconds::new(TAP_GAP_MS),
                tap: tap_on(0),
            },
            ScriptedTap {
                at: Milliseconds::new(TAP_GAP_MS + gap_ms),
                tap: tap_on(1),
            },
        ],
    );
    rig.wait_events("two taps to fire", |e| fired_count(e) >= before + 2);
    before
}

#[test]
fn en3_chains_run_in_order_and_never_overlap() {
    let rig = Rig::new("en3");
    rig.calibrate("Desk");
    two_step_chains(&rig);
    rig.os.set_action_time(Milliseconds::new(400.0));
    let completed_before = completed_count(&rig.events());
    two_taps(&rig, 310.0);
    rig.wait_events("both chains", |e| {
        completed_count(e) >= completed_before + 2
    });
    let log = rig.os.platform_log();
    assert_eq!(log.len(), 4, "two chains of two calls each: {log:?}");
    // Each chain's two calls run back to back, and the second chain starts after the first
    // chain's last call ends.
    let chain = |c: &qwerty_app::engine::testkit::PlatformCall| match &c.action {
        ActionSpec::CopyToClipboard { text } => text.clone(),
        ActionSpec::OpenUrl { url } => format!(
            "chain {}",
            url.rsplit('/').next().expect("a url with a path")
        ),
        other => panic!("unexpected call {other:?}"),
    };
    assert_eq!(chain(&log[0]), chain(&log[1]));
    assert_eq!(chain(&log[2]), chain(&log[3]));
    assert!(matches!(log[0].action, ActionSpec::CopyToClipboard { .. }));
    assert!(matches!(log[1].action, ActionSpec::OpenUrl { .. }));
    for w in log.windows(2) {
        assert!(w[1].started >= w[0].ended, "calls overlap: {w:?}");
    }

    // A failing first step ends its chain: the OpenUrl never runs.
    rig.os.set_action_time(Milliseconds::new(0.0));
    rig.os.fail_next_action(ActionError::Clipboard {
        detail: "scripted failure".to_string(),
    });
    let completed_before = completed_count(&rig.events());
    let fired_before = fired_count(&rig.events());
    rig.os.release_taps(MIC, taps(0, 1, TAP_GAP_MS));
    rig.wait_events("the tap", |e| fired_count(e) > fired_before);
    rig.wait_events("its chain", |e| completed_count(e) > completed_before);
    let (_, _, result) = rig.completed()[completed_before].clone();
    assert!(
        matches!(result.steps.as_slice(), [StepOutcome::Failed { .. }]),
        "{:?}",
        result.steps
    );
    let log = rig.os.platform_log();
    assert_eq!(log.len(), 5, "only the failed copy ran: {log:?}");
    assert!(matches!(log[4].action, ActionSpec::CopyToClipboard { .. }));
    rig.finish();
}

#[test]
fn en4_a_stuck_action_fills_the_queue_and_the_next_fire_is_skipped() {
    let rig = Rig::new("en4");
    rig.calibrate("Desk");
    for id in rig.zone_ids() {
        rig.set_actions(id, vec![copy("stuck")]);
    }
    rig.os.set_action_time(Milliseconds::new(10_000.0));
    let before = counters(&rig).dispatch_dropped;
    // One running chain, 16 queued behind it, then QueueFull: release taps until it shows.
    let mut released = 0;
    let deadline = Instant::now() + WAIT;
    loop {
        if rig.events().iter().any(|e| {
            matches!(
                e,
                Event::ActionSkipped {
                    reason: SkipReason::QueueFull,
                    ..
                }
            )
        }) {
            break;
        }
        assert!(
            Instant::now() < deadline && released < 60,
            "no QueueFull after {released} taps"
        );
        rig.os.release_taps(
            MIC,
            (0..6)
                .map(|i| ScriptedTap {
                    at: Milliseconds::new(TAP_GAP_MS + 310.0 * i as f64),
                    tap: tap_on(i % ZONES),
                })
                .collect(),
        );
        released += 6;
        std::thread::sleep(Duration::from_secs_f64(6.0 * 0.31 / SPEED + 0.2));
    }
    rig.wait("dispatch_dropped", |s| {
        s.health.counters.dispatch_dropped > before
    });
    // TapFired precedes each Fire's job: the first QueueFull is exactly the 18th fire (one
    // running chain, 16 queued behind it, then this one).
    let events = rig.events();
    let skip = events
        .iter()
        .position(|e| {
            matches!(
                e,
                Event::ActionSkipped {
                    reason: SkipReason::QueueFull,
                    ..
                }
            )
        })
        .expect("a QueueFull");
    assert_eq!(
        fired_count(&events[..skip]),
        18,
        "a running chain, 16 queued and one more"
    );
    rig.finish();
}

#[test]
fn en5_shutdown_with_every_thread_busy() {
    let rig = Rig::with(
        "en5",
        RigOptions {
            debug_capture: DebugCaptureMode::ForceOn,
            ..RigOptions::desk()
        },
    );
    rig.calibrate("Desk");
    for id in rig.zone_ids() {
        rig.set_actions(id, vec![copy("busy"), copy("busy")]);
    }
    rig.os.set_action_time(Milliseconds::new(1000.0));
    rig.os.release_taps(MIC, taps(0, 1, TAP_GAP_MS));
    rig.wait_events("a busy chain", |e| fired_count(e) >= 1);
    // Typing keeps the raw-input worker busy (after the tap, which it would veto).
    rig.os.input(InputScript {
        events: (0..200)
            .map(|i| qwerty_core::veto::InputEvent {
                time: HostTime(i * 100_000),
                kind: qwerty_core::veto::InputKind::KeyDown,
            })
            .collect(),
    });
    std::thread::sleep(Duration::from_millis(100));
    let dir = rig.dir.clone();
    let start = Instant::now();
    let report = rig.handle.clone().shutdown().expect("shutdown");
    let took = start.elapsed();
    assert!(
        took.as_secs_f64() * 1000.0 < SHUTDOWN_DEADLINE_MS,
        "shutdown took {took:?}"
    );
    assert!(report.timed_out.is_empty(), "{report:?}");
    for thread in [
        ThreadName::Capture,
        ThreadName::Processing,
        ThreadName::Dispatch,
        ThreadName::Recorder,
        ThreadName::RawInput,
    ] {
        assert!(
            report.joined.contains(&thread),
            "{thread:?} not joined: {report:?}"
        );
    }
    let summary = report.recording.clone().expect("a recording was running");
    let reader = qwerty_core::recording::RecordingReader::open(&recording_dir(&dir, &summary.dir))
        .expect("the recording reads back");
    assert!(reader.sidecar.finished, "the sidecar is finished");
    let mut rig = rig;
    if let Some(c) = rig.collector.take() {
        c.join().expect("the collector ends");
    }
    remove_dir(&dir);
}

const SHUTDOWN_DEADLINE_MS: f64 = 5000.0;

/// A recording's directory: `summary.dir` as written, under DebugCaptures when relative.
fn recording_dir(data_dir: &Path, dir: &str) -> PathBuf {
    let p = PathBuf::from(dir);
    if p.is_absolute() {
        p
    } else {
        data_dir.join("DebugCaptures").join(p)
    }
}

#[test]
fn en6_a_capture_panic_faults_the_engine() {
    let rig = Rig::with(
        "en6",
        RigOptions {
            debug_capture: DebugCaptureMode::FromConfig,
            devices: vec![(
                desk_device(MIC, "Panicking mic", true, format()),
                CaptureScript::Panic {
                    after: Milliseconds::new(200.0),
                },
            )],
        },
    );
    rig.ok(Command::StartCalibration {
        target: CalibrationTarget::New {
            name: "Doomed".to_string(),
        },
        zones: drafts(),
        device: DeviceChoice::SystemDefault,
        taps_per_zone: TAPS_PER_ZONE,
    });
    rig.wait("Faulted", |s| {
        matches!(
            &s.mode,
            Mode::Faulted {
                thread: ThreadName::Capture,
                ..
            }
        )
    });
    rig.wait_events("EngineFaulted", |e| {
        e.iter().any(|e| {
            matches!(
                e,
                Event::EngineFaulted {
                    thread: ThreadName::Capture,
                    ..
                }
            )
        })
    });
    let dir = rig.dir.clone();
    let report = rig
        .handle
        .clone()
        .shutdown()
        .expect("shutdown after a fault");
    assert!(report.timed_out.is_empty(), "{report:?}");
    // The run had the veto on (a run needs it), so the raw-input worker was running too.
    assert_eq!(rig.os.raw_input_workers().len(), 1);
    for thread in [
        ThreadName::RawInput,
        ThreadName::Processing,
        ThreadName::Dispatch,
        ThreadName::Recorder,
    ] {
        assert!(
            report.joined.contains(&thread),
            "{thread:?} not joined: {report:?}"
        );
    }
    let mut rig = rig;
    if let Some(c) = rig.collector.take() {
        c.join().expect("the collector ends");
    }
    remove_dir(&dir);
}

#[test]
fn en7_latency_identities() {
    let rig = Rig::new("en7");
    rig.calibrate("Desk");
    let ids = rig.zone_ids();
    for (i, id) in ids.iter().enumerate() {
        rig.set_actions(*id, vec![copy(&format!("chain {i}"))]);
    }
    rig.os.set_action_time(Milliseconds::new(400.0));
    let fired_before = fired_count(&rig.events());
    let completed_before = completed_count(&rig.events());
    let scheduled_before = rig
        .os
        .scheduled_taps(MIC)
        .expect("a running capture")
        .1
        .len();
    two_taps(&rig, 310.0);
    rig.wait_events("both chains", |e| {
        completed_count(e) >= completed_before + 2
    });
    // Let the activity entries settle in the snapshot.
    std::thread::sleep(Duration::from_millis(200));
    let fired = rig.fired()[fired_before..].to_vec();
    let completed = rig.completed();
    let (stream_start, starts) = rig.os.scheduled_taps(MIC).expect("a running capture");
    let starts = &starts[scheduled_before..];
    let snapshot = rig.snapshot();
    for t in &fired {
        let (_, tap, result) = completed
            .iter()
            .find(|(_, tap, _)| tap.is_some_and(|r| r.candidate == t.candidate))
            .cloned()
            .expect("every fired tap completes");
        let tap = tap.expect("a tap ref");
        assert_eq!(tap.t0, t.t0);
        let decision = ms_between(tap.decided_at, tap.t0);
        assert!((t.latency.decision.get() - decision).abs() < 1e-6);
        assert!(decision >= 45.0, "decision latency {decision} ms < 45 ms");
        let controller = ms_between(tap.fired_at, tap.decided_at);
        assert!((t.latency.controller.get() - controller).abs() < 1e-6);
        assert!(controller >= 0.0);
        let entry_latency = snapshot
            .activity
            .recent
            .iter()
            .find_map(|e| match e {
                ActivityEntry::Fired {
                    tap: Some(r),
                    latency,
                    ..
                } if r.candidate == t.candidate => Some(*latency),
                _ => None,
            })
            .expect("an activity entry for the tap")
            .expect("its latency once completed");
        assert!((entry_latency.get() - ms_between(result.started_at, t.t0)).abs() < 1e-6);
        // The detected onset lies within §1.6.4's T2 bound of a scripted tap's arrival at a mic:
        // its start plus the propagation r / c, with r from the tap's position (two_taps puts
        // the first on zone 0 and the second on zone 1) to that mic.
        let positions = [tap_on(0), tap_on(1)];
        let nearest = starts
            .iter()
            .zip(&positions)
            .flat_map(|(s, tap)| {
                let start = qwerty_app::engine::testkit::sample_time(stream_start, *s, format());
                desk_mics().into_iter().map(move |m| {
                    let r = ((tap.x - m.x).powi(2) + (tap.y - m.y).powi(2) + m.z.powi(2)).sqrt();
                    (ms_between(t.t0, start) - r / SPEED_OF_SOUND * 1000.0).abs()
                })
            })
            .fold(f64::INFINITY, f64::min);
        assert!(
            nearest <= ONSET_TOLERANCE_MS,
            "t0 is {nearest} ms from the nearest scripted arrival"
        );
    }
    // Health.latency is the stats of started_at − t0 over the fired taps.
    let samples: Vec<f64> = completed
        .iter()
        .filter_map(|(_, tap, result)| tap.map(|r| ms_between(result.started_at, r.t0)))
        .collect();
    let window = &samples[samples
        .len()
        .saturating_sub(qwerty_app::engine::protocol::LATENCY_WINDOW)..];
    // Health is recomputed on the 1 s health tick.
    let snapshot = rig.wait("Health.latency to catch up", |s| {
        s.health.latency.count as usize == window.len()
    });
    assert_eq!(
        snapshot.health.latency,
        LatencyStats::from_samples(window, 0),
        "Health.latency over the fired taps"
    );
    // The second chain waited for the first: its dispatch wait covers the rest of the first
    // chain's 400 ms.
    if fired.len() >= 2 {
        let first = completed
            .iter()
            .find(|(_, tap, _)| tap.is_some_and(|r| r.candidate == fired[0].candidate))
            .expect("the first chain")
            .clone();
        let second = completed
            .iter()
            .find(|(_, tap, _)| tap.is_some_and(|r| r.candidate == fired[1].candidate))
            .expect("the second chain")
            .clone();
        let second_ref = second.1.expect("a tap ref");
        let first_ref = first.1.expect("a tap ref");
        let wait = ms_between(second.2.started_at, second_ref.fired_at);
        // SPEC: §2.11 states the bound as 400 − 310 ms minus the controller hop; the fire times
        // differ from the script's 310 ms by the packet phase, so the bound uses the measured gap.
        let gap = ms_between(second_ref.fired_at, first_ref.fired_at);
        let first_wait = ms_between(first.2.started_at, first_ref.fired_at);
        assert!(
            wait >= 400.0 - gap + first_wait - 1.0,
            "the second chain waited {wait} ms (gap {gap} ms)"
        );
        assert!(second.2.started_at >= first.2.finished_at);
    }
    rig.finish();
}

/// The detector's onset error against the true arrival: p01 §1.6.4 T2's hard bound (T3's
/// 0.25 ms holds for one mic at a known position), plus the 0.01 m position jitter of the
/// scripted taps (0.03 ms at the speed of sound).
const ONSET_TOLERANCE_MS: f64 = 2.0 + 0.01 / SPEED_OF_SOUND * 1000.0;
/// The desk model's speed of sound (m/s).
const SPEED_OF_SOUND: f64 = 343.0;

/// The fake capture's mics: the default desk at the scripts' rate and seed (`desk_script`).
fn desk_mics() -> Vec<qwerty_core::desk::MicPos> {
    qwerty_core::desk::DeskConfig::default_for(format().rate, 11).mics
}

#[test]
fn en8_command_reply_table_and_typed_extraction() {
    let rig = Rig::new("en8");
    rig.wait("Idle", |s| s.mode == Mode::Idle);
    // Every command an Idle engine accepts answers with its table kind (Rig::request asserts
    // it); the refusals carry no reply.
    let export = rig.dir.join("export.json");
    let commands = vec![
        Command::GetSnapshot,
        Command::ListDevices,
        Command::SetListening { listening: false },
        Command::SetListening { listening: true },
        Command::ToggleListening,
        Command::ToggleListening,
        Command::SetTelemetry {
            level: TelemetryLevel::Meter,
        },
        Command::SetTelemetry {
            level: TelemetryLevel::Off,
        },
        Command::RetryCapture,
        Command::RetryInputVeto,
        Command::UpdateConfig {
            edit: Default::default(),
        },
        Command::GetInspection,
        Command::ExportInspection {
            path: export.clone(),
        },
        Command::ShellToast,
        Command::DeleteDebugRecordings,
    ];
    let mut refused = Vec::new();
    for command in commands {
        let text = format!("{command:?}");
        if let Err(e) = rig.request(command) {
            refused.push(format!("{text}: {e:?}"));
        }
    }
    assert!(refused.is_empty(), "refused: {refused:#?}");
    // These two need a calibrated active profile; Idle refuses them.
    assert_eq!(
        rig.request(Command::GetSoundMap).err(),
        Some(EngineError::ProfileNotCalibrated)
    );
    assert_eq!(
        rig.request(Command::SaveLatencyReport).err(),
        Some(EngineError::NoActiveProfile)
    );

    // The rest of the table needs a calibrated profile and a saved evaluation report.
    rig.calibrate("Desk");
    let profile = rig
        .snapshot()
        .active_profile
        .as_ref()
        .expect("an active profile")
        .id;
    let zone = rig.zone_ids()[0];
    rig.set_actions(zone, vec![copy("test")]);
    rig.ok(Command::GetSoundMap);
    rig.ok(Command::SaveLatencyReport);
    rig.ok(Command::ExportProfile {
        id: profile,
        path: rig.dir.join("profile-export.json"),
    });
    rig.ok(Command::DuplicateProfile {
        id: profile,
        name: "Copy".to_string(),
    });
    rig.ok(Command::TestAction {
        zone,
        action_index: 0,
        delay: Milliseconds::new(0.0),
    });
    rig.ok(Command::ListEvaluations { profile });
    // An evaluation with no taps misses every prompt and ends with a report; with Evaluations
    // a regular file, saving it fails, so the report is held unsaved.
    let evaluations = rig.dir.join("Evaluations");
    std::fs::remove_dir_all(&evaluations).expect("remove Evaluations");
    std::fs::write(&evaluations, b"not a directory").expect("a file named Evaluations");
    rig.ok(Command::StartEvaluation {
        prompts_per_zone: 1,
        negatives: Vec::new(),
    });
    rig.wait("Evaluating", |s| matches!(s.mode, Mode::Evaluating { .. }));
    rig.wait("the evaluation to end", |s| s.mode == Mode::Listening);
    rig.ok(Command::ExportUnsavedReport {
        format: qwerty_app::engine::protocol::ReportFormat::Json,
        path: rig.dir.join("unsaved.json"),
    });
    std::fs::remove_file(&evaluations).expect("remove the file");
    std::fs::create_dir(&evaluations).expect("recreate Evaluations");
    rig.ok(Command::RetrySaveReport);
    let reports = rig
        .ok(Command::ListEvaluations { profile })
        .into_evaluations()
        .expect("Evaluations");
    let report = reports.first().expect("the saved report").report.clone();
    rig.ok(Command::GetEvaluation {
        profile,
        report: report.clone(),
    });
    rig.ok(Command::ExportEvaluation {
        profile,
        report,
        format: qwerty_app::engine::protocol::ReportFormat::Csv,
        path: rig.dir.join("report.csv"),
    });
    // Typed extraction: each into_* refuses every other variant with ProtocolMismatch.
    let snapshot = rig.ok(Command::GetSnapshot);
    let replies = vec![
        Reply::Ack,
        snapshot,
        Reply::Devices {
            devices: Vec::new(),
        },
        Reply::Saved {
            path: PathBuf::from("x"),
        },
        Reply::Evaluations {
            reports: Vec::new(),
        },
        Reply::Inspection {
            entries: Vec::new(),
        },
        Reply::TestScheduled {
            fires_at: HostTime(0),
        },
    ];
    for reply in &replies {
        let kind = reply.kind();
        let results: Vec<(ReplyKind, Option<EngineError>)> = vec![
            (ReplyKind::Ack, reply.clone().into_ack().err()),
            (ReplyKind::Snapshot, reply.clone().into_snapshot().err()),
            (ReplyKind::Devices, reply.clone().into_devices().err()),
            (ReplyKind::Saved, reply.clone().into_saved().err()),
            (
                ReplyKind::ProfileCreated,
                reply.clone().into_profile_created().err(),
            ),
            (
                ReplyKind::ImportPreview,
                reply.clone().into_import_preview().err(),
            ),
            (
                ReplyKind::Calibration,
                reply.clone().into_calibration().err(),
            ),
            (ReplyKind::Committed, reply.clone().into_committed().err()),
            (ReplyKind::Evaluation, reply.clone().into_evaluation().err()),
            (
                ReplyKind::Evaluations,
                reply.clone().into_evaluations().err(),
            ),
            (
                ReplyKind::EvaluationReport,
                reply.clone().into_evaluation_report().err(),
            ),
            (
                ReplyKind::TestScheduled,
                reply.clone().into_test_scheduled().err(),
            ),
            (ReplyKind::Inspection, reply.clone().into_inspection().err()),
            (ReplyKind::SoundMap, reply.clone().into_sound_map().err()),
            (ReplyKind::Shutdown, reply.clone().into_shutdown().err()),
        ];
        for (expected, error) in results {
            if expected == kind {
                assert_eq!(error, None, "into_{expected:?} refused its own variant");
            } else {
                assert_eq!(
                    error,
                    Some(EngineError::ProtocolMismatch {
                        expected,
                        got: kind
                    }),
                    "into_{expected:?} on {kind:?}"
                );
            }
        }
    }
    rig.finish();
}

#[test]
fn en10_a_failed_commit_keeps_the_review() {
    let rig = Rig::new("en10");
    let profiles = rig.dir.join("Profiles");
    std::fs::remove_dir_all(&profiles).expect("remove Profiles");
    std::fs::write(&profiles, b"not a directory").expect("a file named Profiles");
    rig.calibrate_to_build("Desk");
    let review = step(&rig.wait("Review", |s| {
        matches!(step(s), Some(CalibrationStep::Review { .. }))
    }));
    let commit = || Command::Calibration {
        action: CalibrationAction::Commit {
            name: "Desk".to_string(),
        },
    };
    match rig.request(commit()) {
        Err(EngineError::Persistence { error }) => {
            assert_eq!(error.kind, PersistenceFailureKind::Io, "{error:?}")
        }
        other => panic!("the commit did not fail on I/O: {other:?}"),
    }
    let s = rig.snapshot();
    assert!(
        matches!(step(&s), Some(CalibrationStep::Review { .. })),
        "the session left Review: {:?}",
        s.mode
    );
    assert_eq!(step(&s), review, "the Review result changed");
    std::fs::remove_file(&profiles).expect("remove the file");
    std::fs::create_dir(&profiles).expect("recreate Profiles");
    let id = rig.ok(commit()).into_committed().expect("Committed");
    rig.wait("Listening", |s| s.mode == Mode::Listening);
    assert!(profiles.join(format!("{}.json", id.as_uuid())).is_file());
    rig.finish();
}

/// DN1 (shell-v2 WP-E1, amendment AB-9) through the engine's own commands: creating (the
/// wizard's `Commit`), renaming, duplicating or importing to a stored profile's name after trim
/// and lowercase fails with `DuplicateName` and writes nothing; the same name in another case on
/// the profile itself succeeds (a rename, and a recalibration's commit).
#[test]
fn dn1_duplicate_profile_names_are_refused_by_the_engine() {
    let rig = Rig::new("dn1");
    let duplicate = Some(EngineError::Profile {
        error: ProfileError::DuplicateName,
    });
    rig.calibrate("Desk");
    let desk = rig
        .snapshot()
        .active_profile
        .as_ref()
        .expect("an active profile")
        .id;

    // Duplicate.
    let err = rig
        .request(Command::DuplicateProfile {
            id: desk,
            name: " DESK ".to_string(),
        })
        .err();
    assert_eq!(err, duplicate, "duplicate to a stored name");
    assert_eq!(rig.profile_files(), 1);
    let copy = rig
        .ok(Command::DuplicateProfile {
            id: desk,
            name: "Copy".to_string(),
        })
        .into_profile_created()
        .expect("ProfileCreated");

    // Rename.
    let err = rig
        .request(Command::RenameProfile {
            id: copy,
            name: "desk".to_string(),
        })
        .err();
    assert_eq!(err, duplicate, "rename to another profile's name");
    rig.ok(Command::RenameProfile {
        id: copy,
        name: "COPY".to_string(),
    });

    // Import: the file's profile keeps its name "Desk".
    let file = rig.dir.join("desk-export.json");
    rig.ok(Command::ExportProfile {
        id: desk,
        path: file.clone(),
    });
    let preview = |rig: &Rig| {
        rig.ok(Command::PreviewImport { path: file.clone() })
            .into_import_preview()
            .expect("ImportPreview")
            .preview
    };
    let err = rig
        .request(Command::ImportProfile {
            preview: preview(&rig),
        })
        .err();
    assert_eq!(err, duplicate, "import under a stored name");
    assert_eq!(rig.profile_files(), 2);
    rig.ok(Command::RenameProfile {
        id: desk,
        name: "Old desk".to_string(),
    });
    rig.ok(Command::ImportProfile {
        preview: preview(&rig),
    });
    assert_eq!(rig.profile_files(), 3);

    // Create: the wizard names the new profile at Save, after the run started under another.
    rig.calibrate_to_build("New desk");
    let review = |rig: &Rig| {
        rig.wait("Review", |s| {
            matches!(step(s), Some(CalibrationStep::Review { .. }))
        })
    };
    review(&rig);
    let commit = |name: &str| Command::Calibration {
        action: CalibrationAction::Commit {
            name: name.to_string(),
        },
    };
    assert_eq!(
        rig.request(commit(" copy")).err(),
        duplicate,
        "a new profile saved under a stored name"
    );
    assert!(
        matches!(step(&rig.snapshot()), Some(CalibrationStep::Review { .. })),
        "a refused commit keeps the Review"
    );
    assert_eq!(rig.profile_files(), 3);
    let created = rig
        .ok(commit("Desk 2"))
        .into_committed()
        .expect("Committed");
    rig.wait("Listening", |s| s.mode == Mode::Listening);
    assert_eq!(rig.profile_files(), 4);

    // Recalibrate: its commit may not take another profile's name, but may recase its own.
    rig.calibrate_to_build_on(CalibrationTarget::Recalibrate { profile: created });
    review(&rig);
    assert_eq!(
        rig.request(commit("old DESK")).err(),
        duplicate,
        "a recalibration renamed to another profile's name"
    );
    assert_eq!(
        rig.ok(commit("DESK 2")).into_committed(),
        Ok(created),
        "a recalibration keeps its id"
    );
    rig.wait("Listening", |s| s.mode == Mode::Listening);
    assert_eq!(rig.profile_files(), 4);
    rig.finish();
}

#[test]
fn en11_the_veto_comes_back_with_a_fresh_ring() {
    let rig = Rig::new("en11");
    rig.start_run("Veto");
    rig.wait("veto Active", |s| s.veto == VetoStatus::Active);
    let veto = |on: bool| Command::UpdateConfig {
        edit: qwerty_app::engine::protocol::ConfigEdit {
            input_veto_enabled: Some(on),
            ..Default::default()
        },
    };
    // Each return to Active comes from a new raw-input request, which brings its own ring.
    let workers = || rig.os.raw_input_workers();
    let first = workers();
    assert_eq!(first.len(), 1, "one raw-input worker so far: {first:?}");
    rig.ok(veto(false));
    rig.wait("veto Disabled", |s| s.veto == VetoStatus::Disabled);
    rig.ok(veto(true));
    rig.wait("veto Active again", |s| s.veto == VetoStatus::Active);
    let second = workers();
    assert_eq!(
        second.len(),
        2,
        "toggling on spawned a fresh worker: {second:?}"
    );
    assert_ne!(second[1], second[0]);
    // An unavailable veto comes back with RetryInputVeto.
    rig.os.fail_next_raw_input(
        qwerty_app::engine::protocol::VetoUnavailableReason::ThreadFailed {
            message: "scripted".to_string(),
        },
    );
    rig.ok(veto(false));
    rig.wait("veto Disabled", |s| s.veto == VetoStatus::Disabled);
    rig.ok(veto(true));
    rig.wait("veto Unavailable", |s| {
        matches!(s.veto, VetoStatus::Unavailable { .. })
    });
    rig.ok(Command::RetryInputVeto);
    rig.wait("veto Active after the retry", |s| {
        s.veto == VetoStatus::Active
    });
    // The failed spawn started no worker; the retry did, with its own ring.
    let third = workers();
    assert_eq!(
        third.len(),
        3,
        "the retry spawned a fresh worker: {third:?}"
    );
    assert!(!third[..2].contains(&third[2]));
    rig.finish();
}

#[test]
fn en12_late_messages_are_stale() {
    const OTHER: &str = "mic-other";
    let mut options = RigOptions::desk();
    options.devices.push((
        desk_device(OTHER, "Other mic", false, format()),
        desk_script(),
    ));
    let rig = Rig::with("en12", options);
    // Listening on a calibrated profile first: a recalibration on the same mic then keeps the
    // capture and the veto across CancelCalibration, so neither worker's exit can be the stale
    // message counted below.
    rig.calibrate("Desk");
    let workers = rig.os.raw_input_workers().len();
    rig.calibrate_to_build("Stale");
    rig.wait("Building", |s| {
        matches!(step(s), Some(CalibrationStep::Building { .. }))
    });
    let before = counters(&rig).stale_messages;
    rig.ok(Command::CancelCalibration);
    rig.wait("Listening", |s| s.mode == Mode::Listening);
    rig.wait("the cancelled build's late message to count", |s| {
        s.health.counters.stale_messages > before
    });
    assert_eq!(
        rig.os.raw_input_workers().len(),
        workers,
        "the veto was kept, so its worker cannot be the stale sender"
    );
    // A capture exit from a replaced worker: a run on the other mic replaces the Listening
    // capture, and cancelling it replaces that one in turn.
    let before = counters(&rig).stale_messages;
    rig.ok(Command::StartCalibration {
        target: CalibrationTarget::New {
            name: "Elsewhere".to_string(),
        },
        zones: drafts(),
        device: DeviceChoice::Endpoint {
            endpoint_id: OTHER.to_string(),
        },
        taps_per_zone: TAPS_PER_ZONE,
    });
    rig.wait("the other mic", |s| {
        matches!(&s.capture, CaptureStatus::Running { opened, .. }
            if opened.identity.endpoint_id == OTHER)
    });
    rig.wait("the replaced capture's exit to count", |s| {
        s.health.counters.stale_messages > before
    });
    let before = counters(&rig).stale_messages;
    rig.ok(Command::CancelCalibration);
    rig.wait("Listening on the desk mic", |s| {
        s.mode == Mode::Listening
            && matches!(&s.capture, CaptureStatus::Running { opened, .. }
                if opened.identity.endpoint_id == MIC)
    });
    rig.wait("the other capture's exit to count", |s| {
        s.health.counters.stale_messages > before
    });
    assert_eq!(rig.os.raw_input_workers().len(), workers);
    rig.finish();
}

fn device(id: &str, state: DeviceState, probe: ProbeOutcome) -> DeviceInfo {
    DeviceInfo {
        endpoint_id: id.to_string(),
        name: id.to_string(),
        is_default: false,
        state,
        probe,
    }
}

#[test]
fn en13_device_listing_edge_cases() {
    // (a) Only Disabled or Unplugged devices.
    let rig = Rig::with(
        "en13a",
        RigOptions {
            debug_capture: DebugCaptureMode::FromConfig,
            devices: vec![
                (
                    device("mic-off", DeviceState::Disabled, ProbeOutcome::NotActive),
                    desk_script(),
                ),
                (
                    device("mic-gone", DeviceState::Unplugged, ProbeOutcome::NotActive),
                    desk_script(),
                ),
            ],
        },
    );
    let devices = rig
        .ok(Command::ListDevices)
        .into_devices()
        .expect("Devices");
    assert_eq!(devices.len(), 2, "{devices:?}");
    assert!(devices.iter().all(|d| !d.is_default));
    rig.ok(Command::SetTelemetry {
        level: TelemetryLevel::Diagnostics,
    });
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(rig.snapshot().capture, CaptureStatus::Off);
    match rig.request(Command::StartCalibration {
        target: CalibrationTarget::New {
            name: "None".to_string(),
        },
        zones: drafts(),
        device: DeviceChoice::SystemDefault,
        taps_per_zone: TAPS_PER_ZONE,
    }) {
        Err(EngineError::Capture {
            error: CaptureError::NoDefaultDevice,
        }) => {}
        other => panic!("expected NoDefaultDevice, got {other:?}"),
    }
    let file = rig.dir.join("import.json");
    let profile = qwerty_core::store::Profile::new(
        "Imported".to_string(),
        qwerty_core::store::ZoneCount::new(4).expect("4 zones"),
        chrono::Utc::now(),
    )
    .expect("a new profile");
    qwerty_app::engine::persist::export_profile(&profile, &file).expect("export");
    let preview = rig
        .ok(Command::PreviewImport { path: file })
        .into_import_preview()
        .expect("ImportPreview");
    rig.ok(Command::ImportProfile {
        preview: preview.preview,
    })
    .into_profile_created()
    .expect("ProfileCreated");
    rig.finish();

    // (b) Two Active devices, one whose probe fails.
    let failing = CaptureError::AccessDenied;
    let rig = Rig::with(
        "en13b",
        RigOptions {
            debug_capture: DebugCaptureMode::FromConfig,
            devices: vec![
                (desk_device(MIC, "Desk mic", true, format()), desk_script()),
                (
                    device(
                        "mic-locked",
                        DeviceState::Active,
                        ProbeOutcome::Failed {
                            error: failing.clone(),
                        },
                    ),
                    desk_script(),
                ),
            ],
        },
    );
    let devices = rig
        .ok(Command::ListDevices)
        .into_devices()
        .expect("Devices");
    assert_eq!(devices.len(), 2, "{devices:?}");
    rig.calibrate("Desk");
    let profile = rig
        .snapshot()
        .active_profile
        .as_ref()
        .expect("an active profile")
        .id;
    match rig.request(Command::RebindDevice {
        profile,
        endpoint_id: "mic-locked".to_string(),
    }) {
        Err(EngineError::Capture { error }) => assert_eq!(error, failing),
        other => panic!("expected the probe's error, got {other:?}"),
    }
    rig.finish();
}

fn recording_dirs(data_dir: &Path) -> usize {
    match std::fs::read_dir(data_dir.join("DebugCaptures")) {
        Ok(entries) => entries
            .filter_map(Result::ok)
            .filter(|e| e.path().is_dir())
            .count(),
        Err(_) => 0,
    }
}

#[test]
fn en14_debug_capture_never_outlives_a_cli_session() {
    let dir = temp_dir("en14");
    let os = Arc::new(FakeOs::new(SPEED));
    let rig = Rig::start(
        dir.clone(),
        Arc::clone(&os),
        RigOptions {
            debug_capture: DebugCaptureMode::ForceOn,
            ..RigOptions::desk()
        },
    );
    assert!(!rig.snapshot().config.debug_capture_enabled);
    rig.start_run("Recorded");
    rig.wait("a recording", |s| {
        matches!(s.recorder.state, RecorderState::Recording { .. })
    });
    assert_eq!(recording_dirs(&dir), 1);
    let mut rig = rig;
    rig.handle.clone().shutdown().expect("shutdown");
    if let Some(c) = rig.collector.take() {
        c.join().expect("the collector ends");
    }
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("config.json")).expect("config.json"),
    )
    .expect("json");
    assert_eq!(
        config["debug_capture_enabled"],
        serde_json::Value::Bool(false)
    );
    // The first engine is shut down; give its slot back before the restart takes one.
    drop(rig);

    let rig = Rig::start(
        dir.clone(),
        Arc::new(FakeOs::new(SPEED)),
        RigOptions::desk(),
    );
    rig.start_run("Not recorded");
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(rig.snapshot().recorder.state, RecorderState::Off);
    assert_eq!(recording_dirs(&dir), 1, "a FromConfig restart recorded");
    rig.finish();
}

#[test]
fn en15_an_extra_speech_finish_is_a_fault() {
    let rig = Rig::new("en15");
    rig.calibrate("Desk");
    let zone = rig.zone_ids()[0];
    rig.set_actions(
        zone,
        vec![ActionSpec::Speak {
            text: "hello".to_string(),
        }],
    );
    rig.os.speech_finishes(2);
    let faults = counters(&rig).faults;
    rig.ok(Command::TestChain {
        zone,
        delay: Milliseconds::new(0.0),
    });
    let s = rig.wait("SpeechUnmatched", |s| {
        s.health.warnings.iter().any(|w| {
            matches!(
                w,
                qwerty_app::engine::protocol::HealthWarning::SpeechUnmatched { count: 1 }
            )
        })
    });
    assert_eq!(s.health.counters.faults, faults + 1);
    rig.finish();
}

#[test]
fn en17_config_through_the_engine() {
    let rig = Rig::new("en17");
    rig.wait("Idle", |s| s.mode == Mode::Idle);
    let edit = |e: qwerty_app::engine::protocol::ConfigEdit| Command::UpdateConfig { edit: e };
    let before = rig.snapshot().config.clone();
    match rig.request(edit(qwerty_app::engine::protocol::ConfigEdit {
        text_zoom: Some(105),
        ..Default::default()
    })) {
        Err(EngineError::Config {
            error: qwerty_core::store::ConfigError::InvalidTextZoom { value: 105 },
        }) => {}
        other => panic!("expected InvalidTextZoom, got {other:?}"),
    }
    assert_eq!(rig.snapshot().config, before);
    rig.ok(edit(qwerty_app::engine::protocol::ConfigEdit {
        text_zoom: Some(125),
        ..Default::default()
    }));
    rig.wait("text_zoom 125", |s| s.config.text_zoom == 125);
    let saved: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(rig.dir.join("config.json")).expect("config.json"),
    )
    .expect("json");
    assert_eq!(saved["text_zoom"], serde_json::json!(125));
    let dismiss = || {
        edit(qwerty_app::engine::protocol::ConfigEdit {
            dismiss_notice: Some(qwerty_core::store::NoticeId::UnattendedActions),
            ..Default::default()
        })
    };
    rig.ok(dismiss());
    rig.ok(dismiss());
    let notices = rig
        .wait("the dismissal", |s| !s.config.dismissed_notices.is_empty())
        .config
        .dismissed_notices
        .clone();
    assert_eq!(
        notices
            .iter()
            .filter(|n| **n == qwerty_core::store::NoticeId::UnattendedActions)
            .count(),
        1,
        "{notices:?}"
    );
    rig.finish();
}

#[test]
fn rc3_an_undeliverable_segment_stops_the_recording() {
    let rig = Rig::with(
        "rc3",
        RigOptions {
            debug_capture: DebugCaptureMode::ForceOn,
            ..RigOptions::desk()
        },
    );
    rig.start_run("Recorded");
    rig.wait("recording and veto", |s| {
        matches!(s.recorder.state, RecorderState::Recording { .. }) && s.veto == VetoStatus::Active
    });
    rig.os.stall_recorder(Milliseconds::new(10_000.0));
    // Wait for the recorder to block in a write.
    std::thread::sleep(Duration::from_millis(300));
    rig.os.input(InputScript {
        events: (0..1500)
            .map(|i| qwerty_core::veto::InputEvent {
                // 2 s spread, in 100-ns ticks.
                time: HostTime(i * 2 * 10_000_000 / 1500),
                kind: qwerty_core::veto::InputKind::KeyDown,
            })
            .collect(),
    });
    std::thread::sleep(Duration::from_secs_f64(2.5 / SPEED));
    rig.ok(Command::MarkSegment {
        kind: qwerty_core::recording::SegmentKind::Corpus {
            class: qwerty_core::recording::NegativeClass::Typing,
        },
    });
    rig.os.stall_recorder(Milliseconds::new(0.0));
    rig.wait("StructureEventLost", |s| {
        matches!(
            &s.recorder.state,
            RecorderState::Stopped {
                reason: RecorderStopReason::StructureEventLost,
                ..
            }
        )
    });
    rig.finish();
}

#[test]
fn rp2_live_and_replay_agree() {
    let rig = Rig::with(
        "rp2",
        RigOptions {
            debug_capture: DebugCaptureMode::ForceOn,
            ..RigOptions::desk()
        },
    );
    rig.calibrate("Desk");
    let fired = fire_one_per_zone(&rig);
    assert!(!fired.is_empty(), "no scripted tap fired");
    let id = rig
        .snapshot()
        .active_profile
        .as_ref()
        .expect("a profile")
        .id;
    let dir = rig.dir.clone();
    let mut rig = rig;
    let report = rig.handle.clone().shutdown().expect("shutdown");
    if let Some(c) = rig.collector.take() {
        c.join().expect("the collector ends");
    }
    let summary = report.recording.expect("the session was recorded");
    let recording = recording_dir(&dir, &summary.dir);
    let file = dir.join("Profiles").join(format!("{}.json", id.as_uuid()));
    let text = std::fs::read_to_string(&file).expect("the committed profile");
    let profile = match qwerty_core::migrate::load_profile(&text, &file).expect("it loads") {
        qwerty_core::migrate::Loaded::Current(p) => p,
        qwerty_core::migrate::Loaded::Migrated { value, .. } => value,
    };
    for with_profile in [Some(profile), None] {
        let label = if with_profile.is_some() {
            "with the committed profile"
        } else {
            "calibrated from the recording"
        };
        let options = qwerty_core::replay::ReplayOptions {
            engines: vec![qwerty_core::replay::ReplayEngine::V2],
            detector: qwerty_core::detector::DetectorParams::DEFAULT,
            training: qwerty_core::training::TrainingParams::DEFAULT,
            block: qwerty_core::replay::DEFAULT_REPLAY_BLOCK,
            profile: with_profile,
        };
        let replay =
            qwerty_core::replay::run_replay(&recording, &options).expect("the replay runs");
        let agreement = replay.results[0]
            .live_agreement
            .clone()
            .expect("V2 reports live agreement");
        assert!(agreement.live > 0, "{label}: no live decision recorded");
        assert_eq!(
            (agreement.mismatched, agreement.missing, agreement.extra),
            (0, 0, 0),
            "{label}: {agreement:?}"
        );
    }
    remove_dir(&dir);
}

/// §2.11: ring drops surface as `capture_dropped_frames` and `CaptureOverflow` while the stream
/// still runs, and a stop the controller starts loses none of them.
#[test]
fn capture_drops_show_while_the_stream_runs() {
    let rig = Rig::new("capture-drops");
    rig.ok(Command::SetTelemetry {
        level: TelemetryLevel::Diagnostics,
    });
    rig.wait("the Diagnostics capture", |s| {
        matches!(s.capture, CaptureStatus::Running { .. })
    });
    rig.os.drop_packets(MIC, 3);
    // 3 packets of 10 ms at 48 kHz.
    let dropped = 3 * 480;
    rig.wait("CaptureOverflow", |s| {
        s.health
            .warnings
            .iter()
            .any(|w| matches!(w, HealthWarning::CaptureOverflow { .. }))
    });
    rig.wait("the drops and packets while the stream runs", |s| {
        s.health.counters.capture_dropped_frames == dropped
            && s.health.counters.capture_packets > 3
            && matches!(s.capture, CaptureStatus::Running { .. })
    });
    rig.ok(Command::SetTelemetry {
        level: TelemetryLevel::Off,
    });
    rig.wait("the capture stopped", |s| s.capture == CaptureStatus::Off);
    // The stopped worker's exit arrives after the stop; it must not count the drops again.
    std::thread::sleep(Duration::from_millis(1500));
    assert_eq!(
        rig.snapshot().health.counters.capture_dropped_frames,
        dropped
    );
    rig.finish();
}
