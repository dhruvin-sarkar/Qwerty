//! The `--v2-*` CLI modes (p2 §2.15): hardware evidence for the v2 engine, side by side with the
//! v1 modes until WP-D1.
//!
//! - `--v2-list-devices`: every capture endpoint, with its probe.
//! - `--v2-monitor [secs] [device] --data-dir <dir>`: Diagnostics telemetry (meter and every
//!   detector candidate with its gates and reject reason).
//! - `--v2-live [zones] [taps] [device] --data-dir <dir> [--record]`: a terminal wizard (ambient,
//!   zones, negatives, build, review, commit), then Listening with decisions, fires and latency
//!   stats; writes a `LatencyReport` at exit.
//! - `--v2-corpus <class> <secs> [device] --data-dir <dir>`: a recording marked as a negative
//!   corpus segment, made while Listening on the data folder's calibrated profile (whose actions
//!   must all be VisualOnly, as `--v2-live` leaves them).
//!
//! Recording is switched on only through `EngineOptions.debug_capture` (`--record` and
//! `--v2-corpus`), never through the config (ADR-006). Every engine mode prints the recorder state.

use std::io::BufRead;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, TryRecvError};

use qwerty_core::profile::{ActionSpec, ZoneLayout};
use qwerty_core::recording::{NegativeClass, SegmentKind};
use qwerty_core::session::{CalibrationStep, TapFeedback};
use qwerty_core::store::{ZoneCount, ZoneEdit};
use qwerty_core::training::MIN_TAPS_PER_ZONE;

use crate::engine::handle::{Engine, EngineHandle, EngineOptions, EventStream};
use crate::engine::logfile::RotatingLog;
use crate::engine::messages::OsParts;
use crate::engine::os::WindowsParts;
use crate::engine::protocol::{
    AutostartPolicy, CalibrationAction, CalibrationTarget, CaptureStatus, Command,
    DebugCaptureMode, DeviceChoice, DeviceInfo, EngineError, Event, Mode, ProbeOutcome,
    RecorderState, RunSession, Snapshot, TelemetryFrame, TelemetryLevel, TraceEntry, TraceOutcome,
    ZoneDraft,
};

/// Exit codes: 0 success, 1 a runtime failure, 2 a usage error.
pub const EXIT_FAILURE: i32 = 1;
pub const EXIT_USAGE: i32 = 2;

/// The longest the CLI waits on the engine alone (a step change, a capture or recorder start).
const ENGINE_WAIT: Duration = Duration::from_secs(30);
/// The longest it waits for the optional negatives step plus the model build.
const BUILD_WAIT: Duration = Duration::from_secs(600);

pub const USAGE: &str = "\
v2 engine modes (hardware evidence; v1 stays the shipping app until WP-D1):
  qwerty-app --v2-list-devices
  qwerty-app --v2-monitor [secs] [device] --data-dir <dir>         (default 20 s)
  qwerty-app --v2-live [zones] [taps] [device] --data-dir <dir> [--record]
                                                                   (default 4 zones, 10 taps per zone)
  qwerty-app --v2-corpus <class> <secs> [device] --data-dir <dir>
                                                                   (a --data-dir --v2-live calibrated)
  [device] = an index from --v2-list-devices, a name substring matching one device, or an endpoint id.
  <class>  = typing | clicking | speech | music | knocks | objects-set-down | everyday";

/// One parsed `--v2-*` invocation.
#[derive(Debug, Clone, PartialEq)]
pub enum CliMode {
    ListDevices,
    Monitor {
        secs: u64,
        device: Option<String>,
        data_dir: PathBuf,
    },
    Live {
        zones: ZoneCount,
        taps: u32,
        device: Option<String>,
        data_dir: PathBuf,
        record: bool,
    },
    Corpus {
        class: NegativeClass,
        secs: u64,
        device: Option<String>,
        data_dir: PathBuf,
    },
}

/// A malformed command line: printed with the usage, exit 2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsageError(pub String);

/// True for an argument this module routes (`main.rs` sends every `--v2-*` here).
pub fn is_v2(arg: &str) -> bool {
    arg.starts_with("--v2-")
}

/// Runs a `--v2-*` mode from the arguments after the program name; returns the exit code.
pub fn run(args: &[String]) -> i32 {
    let mode = match parse(args) {
        Ok(mode) => mode,
        Err(UsageError(message)) => return usage(&message),
    };
    let result = match mode {
        CliMode::ListDevices => list_devices(),
        CliMode::Monitor {
            secs,
            device,
            data_dir,
        } => monitor(secs, device, data_dir),
        CliMode::Live {
            zones,
            taps,
            device,
            data_dir,
            record,
        } => live(zones, taps, device, data_dir, record),
        CliMode::Corpus {
            class,
            secs,
            device,
            data_dir,
        } => corpus(class, secs, device, data_dir),
    };
    match result {
        Ok(()) => 0,
        Err(CliError::Usage(UsageError(message))) => usage(&message),
        Err(CliError::Engine(e)) => {
            eprintln!("error: {e:?}");
            EXIT_FAILURE
        }
        Err(CliError::Io(e)) => {
            eprintln!("error: {e}");
            EXIT_FAILURE
        }
        Err(CliError::Stuck(why)) => {
            eprintln!("error: {why}");
            EXIT_FAILURE
        }
    }
}

fn usage(message: &str) -> i32 {
    eprintln!("error: {message}");
    eprintln!("{USAGE}");
    EXIT_USAGE
}

enum CliError {
    Usage(UsageError),
    Engine(EngineError),
    Io(std::io::Error),
    /// A wait that can no longer end: the engine faulted, the microphone was lost, the
    /// recording ended, or the time limit passed (fail fast instead of hanging).
    Stuck(String),
}

impl From<EngineError> for CliError {
    fn from(e: EngineError) -> CliError {
        CliError::Engine(e)
    }
}

impl From<UsageError> for CliError {
    fn from(e: UsageError) -> CliError {
        CliError::Usage(e)
    }
}

/// Splits `--data-dir <dir>` and `--record` from the positionals.
struct Split {
    positionals: Vec<String>,
    data_dir: Option<PathBuf>,
    record: bool,
}

fn split(args: &[String]) -> Result<Split, UsageError> {
    let mut out = Split {
        positionals: Vec::new(),
        data_dir: None,
        record: false,
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--data-dir" => match it.next() {
                Some(dir) if !dir.starts_with("--") => out.data_dir = Some(PathBuf::from(dir)),
                Some(_) | None => return Err(UsageError("--data-dir needs a folder".into())),
            },
            "--record" => out.record = true,
            other if other.starts_with("--") => {
                return Err(UsageError(format!("unknown option {other:?}")))
            }
            other => out.positionals.push(other.to_string()),
        }
    }
    Ok(out)
}

/// A positional, defaulted only when absent: a malformed value is a usage error (v1's `arg_or`).
fn arg_or<T: std::str::FromStr>(
    args: &[String],
    index: usize,
    default: T,
    name: &str,
) -> Result<T, UsageError> {
    match args.get(index) {
        None => Ok(default),
        Some(s) => s
            .parse()
            .map_err(|_| UsageError(format!("invalid {name}: {s:?}"))),
    }
}

fn required_data_dir(split: &Split, mode: &str) -> Result<PathBuf, UsageError> {
    split
        .data_dir
        .clone()
        .ok_or_else(|| UsageError(format!("{mode} needs --data-dir <dir>")))
}

fn too_many(split: &Split, max: usize, mode: &str) -> Result<(), UsageError> {
    if split.positionals.len() > max {
        return Err(UsageError(format!(
            "{mode} takes at most {max} arguments, got {:?}",
            split.positionals
        )));
    }
    Ok(())
}

/// Parses the arguments after the program name (the first is the `--v2-*` mode).
pub fn parse(args: &[String]) -> Result<CliMode, UsageError> {
    let Some((mode, rest)) = args.split_first() else {
        return Err(UsageError("no mode given".into()));
    };
    let split = split(rest)?;
    let p = &split.positionals;
    match mode.as_str() {
        "--v2-list-devices" => {
            too_many(&split, 0, mode)?;
            if split.data_dir.is_some() || split.record {
                return Err(UsageError("--v2-list-devices takes no options".into()));
            }
            Ok(CliMode::ListDevices)
        }
        "--v2-monitor" => {
            too_many(&split, 2, mode)?;
            no_record(&split, mode)?;
            Ok(CliMode::Monitor {
                secs: positive(arg_or(p, 0, 20u64, "seconds")?, "seconds")?,
                device: p.get(1).cloned(),
                data_dir: required_data_dir(&split, mode)?,
            })
        }
        "--v2-live" => {
            too_many(&split, 3, mode)?;
            let n: u8 = arg_or(p, 0, 4u8, "zones")?;
            let zones = ZoneCount::new(n).map_err(|_| {
                UsageError(format!("zones must be one of {:?}", ZoneCount::ALLOWED))
            })?;
            let taps: u32 = arg_or(p, 1, 10u32, "taps")?;
            if (taps as usize) < MIN_TAPS_PER_ZONE {
                return Err(UsageError(format!(
                    "taps must be at least {MIN_TAPS_PER_ZONE}"
                )));
            }
            Ok(CliMode::Live {
                zones,
                taps,
                device: p.get(2).cloned(),
                data_dir: required_data_dir(&split, mode)?,
                record: split.record,
            })
        }
        "--v2-corpus" => {
            too_many(&split, 3, mode)?;
            no_record(&split, mode)?;
            let class = match p.first() {
                Some(c) => parse_class(c)?,
                None => return Err(UsageError("--v2-corpus needs a <class>".into())),
            };
            let secs = match p.get(1) {
                Some(_) => positive(arg_or(p, 1, 0u64, "seconds")?, "seconds")?,
                None => return Err(UsageError("--v2-corpus needs <secs>".into())),
            };
            Ok(CliMode::Corpus {
                class,
                secs,
                device: p.get(2).cloned(),
                data_dir: required_data_dir(&split, mode)?,
            })
        }
        other => Err(UsageError(format!("unknown mode {other:?}"))),
    }
}

fn no_record(split: &Split, mode: &str) -> Result<(), UsageError> {
    if split.record {
        return Err(UsageError(format!("{mode} does not take --record")));
    }
    Ok(())
}

fn positive(n: u64, name: &str) -> Result<u64, UsageError> {
    if n == 0 {
        return Err(UsageError(format!("{name} must be positive")));
    }
    Ok(n)
}

pub fn parse_class(s: &str) -> Result<NegativeClass, UsageError> {
    match s.to_ascii_lowercase().as_str() {
        "typing" => Ok(NegativeClass::Typing),
        "clicking" => Ok(NegativeClass::Clicking),
        "speech" => Ok(NegativeClass::Speech),
        "music" => Ok(NegativeClass::Music),
        "knocks" => Ok(NegativeClass::Knocks),
        "objects-set-down" => Ok(NegativeClass::ObjectsSetDown),
        "everyday" => Ok(NegativeClass::Everyday),
        other => Err(UsageError(format!("unknown corpus class {other:?}"))),
    }
}

/// `[device]` (PARITY-37): an index, an endpoint id, or a case-insensitive name substring that
/// matches exactly one device. Anything else lists the candidates.
pub fn resolve_device<'a>(
    devices: &'a [DeviceInfo],
    arg: &str,
) -> Result<&'a DeviceInfo, UsageError> {
    if let Ok(i) = arg.parse::<usize>() {
        return devices.get(i).ok_or_else(|| candidates(devices, arg));
    }
    if let Some(d) = devices.iter().find(|d| d.endpoint_id == arg) {
        return Ok(d);
    }
    let needle = arg.to_lowercase();
    let mut matches = devices
        .iter()
        .filter(|d| d.name.to_lowercase().contains(&needle));
    match (matches.next(), matches.next()) {
        (Some(d), None) => Ok(d),
        (Some(_), Some(_)) | (None, _) => Err(candidates(devices, arg)),
    }
}

fn candidates(devices: &[DeviceInfo], arg: &str) -> UsageError {
    let list: Vec<String> = devices
        .iter()
        .enumerate()
        .map(|(i, d)| format!("  [{i}] {} ({})", d.name, d.endpoint_id))
        .collect();
    UsageError(format!(
        "{arg:?} names no single device; the candidates are:\n{}",
        list.join("\n")
    ))
}

fn list_devices() -> Result<(), CliError> {
    let devices = WindowsParts::new()
        .list_devices()
        .map_err(|error| CliError::Engine(EngineError::Capture { error }))?;
    println!("Capture endpoints ({}):", devices.len());
    for (i, d) in devices.iter().enumerate() {
        let probe = match &d.probe {
            ProbeOutcome::Probed { probe } => format!(
                "{} Hz, {} ch, RAW {}",
                probe.format.rate.hz(),
                probe.format.channels.get(),
                if probe.raw_supported { "yes" } else { "no" }
            ),
            ProbeOutcome::NotActive => "not active".to_string(),
            ProbeOutcome::Failed { error } => format!("probe failed: {error:?}"),
        };
        println!(
            "  [{i}] {}{}\n      id: {}\n      state: {:?}; {probe}",
            d.name,
            if d.is_default { " (default)" } else { "" },
            d.endpoint_id,
            d.state,
        );
    }
    Ok(())
}

/// One engine session for a CLI mode: logs to `<data-dir>/Logs`, the shell is ready at once.
struct Session {
    handle: EngineHandle,
    events: EventStream,
    recorder: Option<String>,
}

impl Session {
    fn start(data_dir: PathBuf, debug_capture: DebugCaptureMode) -> Result<Session, CliError> {
        std::fs::create_dir_all(&data_dir).map_err(CliError::Io)?;
        let log = RotatingLog::new(&data_dir.join("Logs")).map_err(CliError::Io)?;
        if tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(log)
            .try_init()
            .is_err()
        {
            eprintln!("note: a log subscriber was already installed");
        }
        let (handle, events) = Engine::start(EngineOptions {
            data_dir,
            autostart: AutostartPolicy::Untouched,
            debug_capture,
        })?;
        handle.request(Command::ShellReady)?.into_ack()?;
        let session = Session {
            handle,
            events,
            recorder: None,
        };
        Ok(session)
    }

    fn devices(&self) -> Result<Vec<DeviceInfo>, CliError> {
        Ok(self.handle.request(Command::ListDevices)?.into_devices()?)
    }

    /// Prints the recorder state whenever it changes (ADR-006: a recording is always visible).
    fn show_recorder(&mut self, s: &Snapshot) {
        let line = match &s.recorder.state {
            RecorderState::Off => None,
            RecorderState::Recording { dir, .. } => Some(format!("recording to {dir}")),
            RecorderState::Stopped { dir, reason } => {
                Some(format!("recording {dir} stopped: {reason:?}"))
            }
            RecorderState::Failed { error } => Some(format!("recording failed: {error:?}")),
        };
        if line != self.recorder {
            if let Some(l) = &line {
                println!("[recorder] {l}");
            }
            self.recorder = line;
        }
    }

    /// The next event within `wait`, printing recorder changes on the way.
    fn next(&mut self, wait: Duration) -> Result<Option<Event>, CliError> {
        let event = self.events.recv_timeout(wait)?;
        if let Some(Event::StateChanged { snapshot }) = &event {
            self.show_recorder(snapshot);
        }
        Ok(event)
    }

    /// Waits for a snapshot satisfying `pred`, handing every other event to `on_event`. Fails
    /// with `Stuck` when the engine faults, when a calibration run loses its microphone, or
    /// after `limit` (None: a wait on the user, such as their taps).
    fn wait_for(
        &mut self,
        pred: impl Fn(&Snapshot) -> bool,
        on_event: &mut dyn FnMut(&Event),
        limit: Option<Duration>,
    ) -> Result<Arc<Snapshot>, CliError> {
        let deadline = limit.map(|l| Instant::now() + l);
        loop {
            let s = self.handle.snapshot();
            if pred(&s) {
                return Ok(s);
            }
            if let Some(why) = stuck(&s) {
                return Err(CliError::Stuck(why));
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Err(CliError::Stuck(format!(
                    "nothing happened within {} s; mode {:?}, capture {:?}",
                    limit.map_or(0, |l| l.as_secs()),
                    s.mode,
                    s.capture
                )));
            }
            match self.next(Duration::from_millis(50))? {
                Some(Event::EngineFaulted { thread, message }) => {
                    return Err(CliError::Stuck(format!(
                        "the engine faulted on {thread:?}: {message}"
                    )))
                }
                Some(e) => on_event(&e),
                None => {}
            }
        }
    }

    /// Runs for `secs`, handing every event to `on_event`. Fails as soon as the engine faults or
    /// `broken` names a reason, so a run that failed partway never reports success.
    fn run_for(
        &mut self,
        secs: u64,
        broken: impl Fn(&Snapshot) -> Option<String>,
        on_event: &mut dyn FnMut(&Event),
    ) -> Result<(), CliError> {
        let end = Instant::now() + Duration::from_secs(secs);
        while Instant::now() < end {
            if let Some(why) = broken(&self.handle.snapshot()) {
                return Err(CliError::Stuck(why));
            }
            match self.next(Duration::from_millis(50))? {
                Some(Event::EngineFaulted { thread, message }) => {
                    return Err(CliError::Stuck(format!(
                        "the engine faulted on {thread:?}: {message}"
                    )))
                }
                Some(e) => on_event(&e),
                None => {}
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<(), CliError> {
        let report = self.handle.shutdown()?;
        if let Some(summary) = report.recording {
            println!(
                "[recorder] {} finished: {} frames, {} events, {} gaps",
                summary.dir, summary.frames, summary.events, summary.gaps
            );
        }
        if !report.timed_out.is_empty() {
            eprintln!(
                "warning: threads missed the shutdown deadline: {:?}",
                report.timed_out
            );
        }
        Ok(())
    }
}

// SPEC: §2.15 gives `--v2-monitor` a `[device]`, but with no calibrated profile the engine's
// Diagnostics capture opens the system default endpoint (§2.8 capture target), and no command
// selects another one outside a calibration run. This mode therefore accepts only the default
// endpoint and refuses any other with a usage error naming it. (`--v2-corpus` records while
// Listening, so its `[device]` must be the profile's bound endpoint.)
fn require_default(session: &Session, device: &Option<String>) -> Result<(), CliError> {
    let Some(arg) = device else {
        return Ok(());
    };
    let devices = session.devices()?;
    let d = resolve_device(&devices, arg)?;
    if !d.is_default {
        return Err(CliError::Usage(UsageError(format!(
            "{} is not the default capture device; this mode monitors the default device, so make it the default in Windows Sound settings first",
            d.name
        ))));
    }
    Ok(())
}

fn monitor(secs: u64, device: Option<String>, data_dir: PathBuf) -> Result<(), CliError> {
    let mut session = Session::start(data_dir, DebugCaptureMode::FromConfig)?;
    require_default(&session, &device)?;
    session
        .handle
        .request(Command::SetTelemetry {
            level: TelemetryLevel::Diagnostics,
        })?
        .into_ack()?;
    session.wait_for(
        |s| matches!(s.capture, CaptureStatus::Running { .. }),
        &mut |_| {},
        Some(ENGINE_WAIT),
    )?;
    println!("Monitoring for {secs} s (Diagnostics telemetry)...");
    let mut last_meter = Instant::now();
    let ran = session.run_for(secs, monitor_broken, &mut |event| match event {
        Event::Telemetry {
            frame: TelemetryFrame::Meter { rms, peak, noise },
        } if last_meter.elapsed() >= Duration::from_millis(500) => {
            last_meter = Instant::now();
            println!(
                "meter  rms {:6.1} dB  peak {:6.1} dB  noise {:6.1} dB",
                rms.get(),
                peak.get(),
                noise.get()
            );
        }
        Event::Telemetry {
            frame: TelemetryFrame::Trace { entries },
        } => entries.iter().for_each(print_trace),
        _ => {}
    });
    let s = session.handle.snapshot();
    println!(
        "capture: {:?}\npackets: {}\ndropped frames: {}",
        s.capture, s.health.counters.capture_packets, s.health.counters.capture_dropped_frames
    );
    // Shut down cleanly either way; a failed run is still the error the exit code reports.
    let finished = session.finish();
    ran.and(finished)
}

/// Why a `--v2-monitor` run is no longer valid evidence: the engine faulted or its microphone
/// stopped.
fn monitor_broken(s: &Snapshot) -> Option<String> {
    stuck(s).or_else(|| match &s.capture {
        CaptureStatus::Running { .. } => None,
        CaptureStatus::Off | CaptureStatus::Failed { .. } => Some(format!(
            "the microphone stopped during the run: {:?}",
            s.capture
        )),
    })
}

/// Why a `--v2-corpus` run is no longer valid evidence: the engine faulted, left Listening (a
/// lost microphone blocks it), or the recording stopped.
fn corpus_broken(s: &Snapshot) -> Option<String> {
    stuck(s)
        .or_else(|| {
            (s.mode != Mode::Listening)
                .then(|| format!("the engine left Listening during the run: {:?}", s.mode))
        })
        .or_else(|| match &s.recorder.state {
            RecorderState::Recording { .. } => None,
            RecorderState::Off | RecorderState::Stopped { .. } | RecorderState::Failed { .. } => {
                Some(format!(
                    "the recording stopped during the run: {:?}",
                    s.recorder.state
                ))
            }
        })
}

fn print_trace(e: &TraceEntry) {
    let gates: Vec<String> = e
        .gates
        .iter()
        .flatten()
        .map(|g| {
            format!(
                "{:?} {:.2}/{:.2}{}",
                g.gate,
                g.value,
                g.limit,
                if g.passed { "" } else { " FAIL" }
            )
        })
        .collect();
    let outcome = match &e.outcome {
        TraceOutcome::DetectorRejected { reason } => format!("rejected: {reason:?}"),
        TraceOutcome::Detected => "detected".to_string(),
        TraceOutcome::Decided { outcome, .. } => format!("decided: {}", outcome_name(outcome)),
    };
    println!(
        "candidate onset {:?}  [{}]  {outcome}",
        e.onset.map(|o| o.0),
        gates.join(", ")
    );
}

fn outcome_name(o: &qwerty_core::pipeline::DecisionOutcome) -> String {
    use qwerty_core::pipeline::DecisionOutcome;
    match o {
        DecisionOutcome::Accepted { zone, .. } => format!("accepted zone {}", zone.0 + 1),
        DecisionOutcome::Vetoed { reason, .. } => format!("vetoed ({reason:?})"),
        DecisionOutcome::Rejected { reason, .. } => format!("rejected ({reason:?})"),
        DecisionOutcome::Invalid { .. } => "invalid descriptor".to_string(),
    }
}

/// Lines typed on stdin, read on their own thread.
fn stdin_lines() -> Receiver<String> {
    let (tx, rx) = crossbeam_channel::bounded(16);
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            match line {
                Ok(l) => {
                    if tx.send(l).is_err() {
                        return;
                    }
                }
                Err(_) => return,
            }
        }
    });
    rx
}

/// A state no CLI wait can leave: the engine faulted, or the calibration run lost its microphone.
fn stuck(s: &Snapshot) -> Option<String> {
    match &s.mode {
        Mode::Faulted { .. } => Some(format!("the engine faulted: {:?}", s.mode)),
        Mode::Calibrating { run } => match &run.session {
            RunSession::MicLost { .. } => Some(format!(
                "the microphone was lost during calibration: {:?}",
                s.capture
            )),
            RunSession::Active { .. } => None,
        },
        Mode::Starting
        | Mode::Idle
        | Mode::Listening
        | Mode::Paused
        | Mode::Evaluating { .. }
        | Mode::Blocked { .. } => None,
    }
}

fn step(s: &Snapshot) -> Option<CalibrationStep> {
    match &s.mode {
        Mode::Calibrating { run } => match &run.session {
            RunSession::Active { view } => Some(view.step.clone()),
            RunSession::MicLost { .. } => None,
        },
        Mode::Starting
        | Mode::Idle
        | Mode::Listening
        | Mode::Paused
        | Mode::Evaluating { .. }
        | Mode::Blocked { .. }
        | Mode::Faulted { .. } => None,
    }
}

/// Zones in a grid of two columns.
fn drafts(zones: ZoneCount) -> Vec<ZoneDraft> {
    let n = zones.get() as usize;
    let rows = n / 2;
    (0..n)
        .map(|i| {
            let (col, row) = (i % 2, i / 2);
            ZoneDraft {
                label: format!("Zone {}", i + 1),
                layout: ZoneLayout {
                    x: 0.05 + 0.5 * col as f32,
                    y: 0.05 + row as f32 / rows as f32,
                    width: 0.4,
                    height: 0.9 / rows as f32 - 0.05,
                },
            }
        })
        .collect()
}

fn feedback_line(f: &TapFeedback) -> String {
    match f {
        TapFeedback::Accepted { zone, count, .. } => {
            format!("  zone {}: tap {count} accepted", zone.0 + 1)
        }
        other => format!("  {other:?}"),
    }
}

fn calibration(session: &Session, action: CalibrationAction) -> Result<(), CliError> {
    match action {
        CalibrationAction::Commit { .. } => {
            session
                .handle
                .request(Command::Calibration { action })?
                .into_committed()?;
        }
        CalibrationAction::RecordAmbient
        | CalibrationAction::AcceptAmbient
        | CalibrationAction::RetryAmbient
        | CalibrationAction::UndoLastTap
        | CalibrationAction::RestartZone
        | CalibrationAction::NextZone
        | CalibrationAction::SkipNegatives
        | CalibrationAction::RedoZone { .. } => {
            session
                .handle
                .request(Command::Calibration { action })?
                .into_calibration()?;
        }
    }
    Ok(())
}

fn prompt(lines: &Receiver<String>, text: &str) -> Result<String, CliError> {
    println!("{text}");
    lines.recv().map_err(|_| {
        CliError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "stdin closed",
        ))
    })
}

fn live(
    zones: ZoneCount,
    taps: u32,
    device: Option<String>,
    data_dir: PathBuf,
    record: bool,
) -> Result<(), CliError> {
    let debug_capture = if record {
        DebugCaptureMode::ForceOn
    } else {
        DebugCaptureMode::FromConfig
    };
    let mut session = Session::start(data_dir, debug_capture)?;
    let choice = match &device {
        Some(arg) => {
            let devices = session.devices()?;
            DeviceChoice::Endpoint {
                endpoint_id: resolve_device(&devices, arg)?.endpoint_id.clone(),
            }
        }
        None => DeviceChoice::SystemDefault,
    };
    let lines = stdin_lines();
    let name = format!("CLI {}", chrono::Utc::now().format("%Y-%m-%d %H:%M"));
    session
        .handle
        .request(Command::StartCalibration {
            target: CalibrationTarget::New { name: name.clone() },
            zones: drafts(zones),
            device: choice,
            taps_per_zone: taps,
        })?
        .into_calibration()?;
    let mut show = |e: &Event| {
        if let Event::CalibrationFeedback { feedback } = e {
            println!("{}", feedback_line(feedback));
        }
    };
    session.wait_for(
        |s| {
            matches!(step(s), Some(CalibrationStep::Ready))
                && s.veto != crate::engine::protocol::VetoStatus::Registering
        },
        &mut show,
        Some(ENGINE_WAIT),
    )?;
    prompt(
        &lines,
        "Ambient: stay quiet, then press Enter to record 3 s of room sound.",
    )?;
    calibration(&session, CalibrationAction::RecordAmbient)?;
    let review = session.wait_for(
        |s| matches!(step(s), Some(CalibrationStep::AmbientReview { .. })),
        &mut show,
        Some(ENGINE_WAIT),
    )?;
    if let Some(CalibrationStep::AmbientReview {
        environment,
        noise_floor,
        disturbances,
        ..
    }) = step(&review)
    {
        println!(
            "ambient: {environment:?}, noise floor {:.1} dB, {disturbances} disturbances",
            noise_floor.get()
        );
    }
    calibration(&session, CalibrationAction::AcceptAmbient)?;
    for zone in 0..zones.get() {
        println!("Tap zone {} {taps} times.", zone + 1);
        let next = zone + 1;
        let last = next == zones.get();
        session.wait_for(
            |s| match step(s) {
                Some(CalibrationStep::Zone {
                    index,
                    accepted,
                    target,
                }) => index.0 >= u16::from(next) || (last && accepted >= target),
                Some(_) | None => false,
            },
            &mut show,
            // The user's taps: no limit.
            None,
        )?;
    }
    let answer = prompt(
        &lines,
        "Negatives: type 'y' and Enter, then type, click and talk normally until it ends; or press Enter to skip.",
    )?;
    if answer.trim().eq_ignore_ascii_case("y") {
        calibration(&session, CalibrationAction::NextZone)?;
    } else {
        calibration(&session, CalibrationAction::SkipNegatives)?;
    }
    println!("Building the model...");
    let review = session.wait_for(
        |s| {
            matches!(
                step(s),
                Some(CalibrationStep::Review { .. }) | Some(CalibrationStep::Failed { .. })
            )
        },
        &mut show,
        // Negatives (if chosen) end on their own deadline; then the build.
        Some(BUILD_WAIT),
    )?;
    match step(&review) {
        Some(CalibrationStep::Review { summary }) => {
            println!("model quality: {:?}", summary.quality)
        }
        Some(CalibrationStep::Failed { error }) => {
            return Err(CliError::Engine(EngineError::Calibration { error }))
        }
        Some(_) | None => {}
    }
    calibration(&session, CalibrationAction::Commit { name })?;
    // Fires do nothing on the machine: every zone's chain is VisualOnly.
    let s = session.wait_for(
        |s| s.mode == Mode::Listening,
        &mut |_| {},
        Some(ENGINE_WAIT),
    )?;
    let zone_ids: Vec<_> = s
        .active_profile
        .as_ref()
        .ok_or(CliError::Engine(EngineError::NoActiveProfile))?
        .zones
        .iter()
        .map(|z| z.id)
        .collect();
    for zone in &zone_ids {
        session
            .handle
            .request(Command::EditZone {
                zone: *zone,
                edit: ZoneEdit {
                    label: None,
                    layout: None,
                    trigger: None,
                    actions: Some(vec![ActionSpec::VisualOnly]),
                },
            })?
            .into_ack()?;
    }
    println!("Listening. Tap the zones; press Enter to stop.");
    let mut fired = 0u64;
    loop {
        match lines.try_recv() {
            Ok(_) | Err(TryRecvError::Disconnected) => break,
            Err(TryRecvError::Empty) => {}
        }
        match session.next(Duration::from_millis(50))? {
            Some(Event::TapFired { tap }) => {
                fired += 1;
                println!(
                    "FIRE {} (decision {:.1} ms, controller {:.1} ms)",
                    tap.label,
                    tap.latency.decision.get(),
                    tap.latency.controller.get()
                );
                if fired.is_multiple_of(10) {
                    let l = session.handle.snapshot().health.latency.clone();
                    println!(
                        "latency (started - t0) over {} taps: p50 {:.1} ms, p95 {:.1} ms, max {:.1} ms",
                        l.count,
                        l.p50.get(),
                        l.p95.get(),
                        l.max.get()
                    );
                }
            }
            Some(Event::TapRejected {
                rejected,
                best_zone,
            }) => println!("not recognized: {rejected:?} (best {best_zone:?})"),
            Some(Event::EngineFaulted { thread, message }) => {
                eprintln!("the engine faulted on {thread:?}: {message}");
                break;
            }
            Some(_) | None => {}
        }
    }
    match session
        .handle
        .request(Command::SaveLatencyReport)
        .and_then(|r| r.into_saved())
    {
        Ok(path) => println!("latency report: {}", path.display()),
        Err(e) => eprintln!("the latency report was not saved: {e:?}"),
    }
    session.finish()
}

fn corpus(
    class: NegativeClass,
    secs: u64,
    device: Option<String>,
    data_dir: PathBuf,
) -> Result<(), CliError> {
    let mut session = Session::start(data_dir, DebugCaptureMode::ForceOn)?;
    // §2.9.2 step 7: only --v2-monitor changes the telemetry level, so this mode records the
    // capture a calibrated profile runs while Listening (with its veto, as in use), on the
    // profile's bound endpoint.
    let settled = session.wait_for(|s| s.mode != Mode::Starting, &mut |_| {}, Some(ENGINE_WAIT))?;
    if settled.mode != Mode::Listening {
        return Err(CliError::Stuck(format!(
            "--v2-corpus records while Listening, so --data-dir needs a calibrated active profile (run --v2-live with it first); mode {:?}",
            settled.mode
        )));
    }
    // Listening fires the profile's chains; a corpus session must never run real actions.
    let acts = settled.active_profile.as_ref().is_some_and(|p| {
        p.zones
            .iter()
            .flat_map(|z| &z.actions)
            .any(|a| *a != ActionSpec::VisualOnly)
    });
    if acts {
        return Err(CliError::Stuck(
            "the active profile has actions that would run on every fire; use a --data-dir whose profile only has VisualOnly actions (as --v2-live leaves it)".to_string(),
        ));
    }
    if let Some(arg) = &device {
        let devices = session.devices()?;
        let wanted = resolve_device(&devices, arg)?;
        let bound = match &settled.capture {
            CaptureStatus::Running { opened, .. } => Some(opened.identity.endpoint_id.as_str()),
            CaptureStatus::Off | CaptureStatus::Failed { .. } => None,
        };
        if bound != Some(wanted.endpoint_id.as_str()) {
            return Err(CliError::Usage(UsageError(format!(
                "{} is not the active profile's microphone; this mode records the profile's bound device",
                wanted.name
            ))));
        }
    }
    let s = session.wait_for(
        |s| !matches!(s.recorder.state, RecorderState::Off),
        &mut |_| {},
        Some(ENGINE_WAIT),
    )?;
    if !matches!(s.recorder.state, RecorderState::Recording { .. }) {
        return Err(CliError::Stuck(format!(
            "the recording did not start: {:?}",
            s.recorder.state
        )));
    }
    session
        .handle
        .request(Command::MarkSegment {
            kind: SegmentKind::Corpus {
                class: class.clone(),
            },
        })?
        .into_ack()?;
    println!("Recording the {class:?} corpus for {secs} s...");
    let ran = session.run_for(secs, corpus_broken, &mut |_| {});
    // Shut down cleanly either way, so the recording is closed; a failed run is still the error.
    let finished = session.finish();
    ran.and(finished)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::protocol::{
        Blocker, CapturePurpose, DeviceState, MicUnavailableReason, MmcssStatus, OpenedStream,
        RecorderStopReason, ThreadName,
    };
    use qwerty_core::audio::{AudioFormat, CaptureMode, DeviceIdentity};
    use qwerty_core::units::{ChannelCount, Milliseconds, SampleRate, Seconds};

    fn args(s: &[&str]) -> Vec<String> {
        s.iter().map(|a| a.to_string()).collect()
    }

    fn running() -> CaptureStatus {
        CaptureStatus::Running {
            opened: OpenedStream {
                identity: DeviceIdentity {
                    endpoint_id: "mic".to_string(),
                    name: "Mic".to_string(),
                    format: AudioFormat {
                        rate: SampleRate::new(48_000).expect("rate"),
                        channels: ChannelCount::new(1).expect("channels"),
                    },
                    capture_mode: CaptureMode::Raw,
                },
                period: Milliseconds::new(10.0),
                buffer_frames: 960,
                mmcss: MmcssStatus::Active,
            },
            purpose: CapturePurpose::Listening,
        }
    }

    fn faulted() -> Mode {
        Mode::Faulted {
            thread: ThreadName::Processing,
            message: "boom".to_string(),
        }
    }

    #[test]
    fn a_monitor_run_breaks_on_a_fault_or_a_stopped_mic() {
        let mut s = crate::engine::handle::starting_snapshot();
        s.mode = Mode::Idle;
        s.capture = running();
        assert_eq!(monitor_broken(&s), None);
        s.mode = faulted();
        assert!(monitor_broken(&s).is_some());
        s.mode = Mode::Idle;
        s.capture = CaptureStatus::Off;
        assert!(monitor_broken(&s).is_some());
    }

    #[test]
    fn a_corpus_run_breaks_on_a_fault_a_lost_mic_or_a_stopped_recording() {
        let mut s = crate::engine::handle::starting_snapshot();
        s.mode = Mode::Listening;
        s.capture = running();
        s.recorder.state = RecorderState::Recording {
            dir: "d".to_string(),
            bytes: 0,
            duration: Seconds::new(1.0),
            gaps: 0,
            dropped_events: 0,
        };
        assert_eq!(corpus_broken(&s), None);
        s.mode = faulted();
        assert!(corpus_broken(&s).is_some());
        s.mode = Mode::Blocked {
            blocker: Blocker::MicUnavailable {
                endpoint_id: "mic".to_string(),
                name: "Mic".to_string(),
                reason: MicUnavailableReason::Unplugged,
            },
        };
        assert!(corpus_broken(&s).is_some());
        s.mode = Mode::Listening;
        s.recorder.state = RecorderState::Stopped {
            dir: "d".to_string(),
            reason: RecorderStopReason::SizeLimit,
        };
        assert!(corpus_broken(&s).is_some());
    }

    fn device(id: &str, name: &str, is_default: bool) -> DeviceInfo {
        DeviceInfo {
            endpoint_id: id.to_string(),
            name: name.to_string(),
            is_default,
            state: DeviceState::Active,
            probe: ProbeOutcome::NotActive,
        }
    }

    #[test]
    fn parse_requires_a_data_dir_and_keeps_arg_or() {
        assert_eq!(
            parse(&args(&["--v2-list-devices"])),
            Ok(CliMode::ListDevices)
        );
        assert!(parse(&args(&["--v2-monitor", "10"])).is_err());
        assert_eq!(
            parse(&args(&["--v2-monitor", "--data-dir", "d"])),
            Ok(CliMode::Monitor {
                secs: 20,
                device: None,
                data_dir: PathBuf::from("d"),
            })
        );
        assert!(parse(&args(&["--v2-monitor", "ten", "--data-dir", "d"])).is_err());
        assert!(parse(&args(&["--v2-live", "3", "--data-dir", "d"])).is_err());
        assert!(parse(&args(&["--v2-live", "4", "4", "--data-dir", "d"])).is_err());
        assert_eq!(
            parse(&args(&[
                "--v2-live",
                "4",
                "10",
                "mic",
                "--data-dir",
                "d",
                "--record"
            ])),
            Ok(CliMode::Live {
                zones: ZoneCount::new(4).expect("4"),
                taps: 10,
                device: Some("mic".to_string()),
                data_dir: PathBuf::from("d"),
                record: true,
            })
        );
        assert_eq!(
            parse(&args(&["--v2-corpus", "Typing", "60", "--data-dir", "d"])),
            Ok(CliMode::Corpus {
                class: NegativeClass::Typing,
                secs: 60,
                device: None,
                data_dir: PathBuf::from("d"),
            })
        );
        assert!(parse(&args(&["--v2-corpus", "humming", "60", "--data-dir", "d"])).is_err());
        assert!(parse(&args(&["--v2-nope"])).is_err());
    }

    #[test]
    fn device_by_index_id_or_unique_substring() {
        let devices = vec![
            device("{a}", "Realtek Mic", true),
            device("{b}", "USB Mic", false),
        ];
        assert_eq!(
            resolve_device(&devices, "1").map(|d| &d.endpoint_id),
            Ok(&"{b}".to_string())
        );
        assert_eq!(
            resolve_device(&devices, "{a}").map(|d| &d.endpoint_id),
            Ok(&"{a}".to_string())
        );
        assert_eq!(
            resolve_device(&devices, "realtek").map(|d| &d.endpoint_id),
            Ok(&"{a}".to_string())
        );
        assert!(resolve_device(&devices, "mic").is_err(), "ambiguous");
        assert!(resolve_device(&devices, "2").is_err(), "out of range");
        assert!(resolve_device(&devices, "bluetooth").is_err(), "unknown");
    }
}
