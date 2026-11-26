//! `FakeOs`, `FakePlatform` and the scripts (p2 §2.11; RUST-RT-3).
//!
//! Public test support, like `qwerty_core::soak`: the integration tests in
//! `crates/qwerty-app/tests/` drive the real controller, processing, dispatch and recorder through
//! `Engine::start_with(options, Arc::new(fake))`. The product never constructs any of this.
//!
//! Fake time runs `speed`× real time. Every wait goes through `OsParts::timeout_until`, so the
//! engine's deadlines and this clock never disagree. The fake capture stamps each packet with the
//! fake time of its first sample, so fake time and stream position share one timebase.

use std::path::Path;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crossbeam_channel::Sender;

use qwerty_core::audio::{AudioFormat, BlockTime, CaptureMode, DefaultModeReason, DeviceIdentity};
use qwerty_core::desk::{DeskConfig, ModalDesk, NoiseFloor, TapSpec};
use qwerty_core::profile::{ActionSpec, KeyCombo, ScreenshotMode, SystemSound};
use qwerty_core::units::{Decibels, HostDuration, HostTime, Milliseconds, SampleIndex};
use qwerty_core::veto::{InputEvent, OutputId, SelfOutputKind};

use crate::engine::messages::{
    CaptureCounters, CaptureExit, CaptureRequest, DeviceMsg, DeviceWatch, InputRecord,
    OpenedCapture, OsParts, PacketMeta, RawInputMsg, RawInputRequest, RawInputStatus, RecorderDisk,
    StopSignal, ThreadEnd, Worker, WorkerId,
};
use crate::engine::protocol::{
    CaptureError, DeviceInfo, DeviceProbe, DeviceState, MmcssStatus, OpenedStream, ProbeOutcome,
    ThreadName, VetoUnavailableReason,
};
use crate::engine::ring::{audio_ring, PushOutcome, CAPTURE_RING};
use crate::platform::{ActionError, OutputObserver, PlatformActions, Screenshot, ToastSpec};

/// The fake clock's origin: 1000 s, so no engine time is ever near zero.
const CLOCK_ORIGIN: HostTime = HostTime(1000 * HostTime::TICKS_PER_SECOND);
/// The longest real sleep a fake thread takes before re-checking its stop flag.
const POLL: Duration = Duration::from_millis(5);
/// How long after a Speak returns the fake tts end callback fires (fake time, §2.11).
const SPEECH_END_AFTER: Milliseconds = Milliseconds::new(500.0);
/// The fake raw-input ring (§2.11: rtrb InputRecord, 1024).
pub const INPUT_RING: usize = 1024;

/// Locks a testkit mutex. A poisoned lock means a test thread panicked while holding it; the
/// test has already failed, so the panic is propagated.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // INVARIANT: testkit locks are held only for plain field updates that cannot panic.
    m.lock().expect("a testkit lock is never poisoned")
}

/// A capture device's behaviour (§2.11).
#[derive(Debug, Clone, PartialEq)]
pub enum CaptureScript {
    /// `qwerty_core::desk::ModalDesk` audio (§1.19.1): `DeskConfig::default_for(format.rate, seed)`
    /// with floor `Pink{rms: ambient}` (format.channels = its 2 mics), taps at fixed times from
    /// stream start, released in packets of `period`. Every open restarts the script at sample 0.
    Desk {
        format: AudioFormat,
        period: Milliseconds,
        ambient: Decibels,
        seed: u64,
        taps: Vec<ScriptedTap>,
    },
    FailOpen {
        error: CaptureError,
    },
    Panic {
        after: Milliseconds,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScriptedTap {
    /// From stream start in `Desk.taps`; from the current position in `release_taps`.
    pub at: Milliseconds,
    /// §1.19.1; positions from `ModalDesk::zone_centres`.
    pub tap: TapSpec,
}

/// Raw input the fake raw-input worker replays after it reports Active. Each event's `time` is
/// an offset from the replay start (`HostTime(ticks)`); the worker stamps the event with the fake
/// time at which it delivers it.
#[derive(Debug, Clone, PartialEq)]
pub struct InputScript {
    pub events: Vec<InputEvent>,
}

/// One platform call with its start and end time (fake clock).
#[derive(Debug, Clone, PartialEq)]
pub struct PlatformCall {
    pub action: ActionSpec,
    pub started: HostTime,
    pub ended: HostTime,
}

struct FakeDevice {
    info: DeviceInfo,
    script: CaptureScript,
}

/// A running fake capture, shared with the test.
struct CaptureControl {
    endpoint_id: String,
    /// Frames released so far.
    position: AtomicU64,
    /// Taps to mix, by absolute start sample.
    released: Mutex<Vec<(u64, TapSpec)>>,
    /// Every tap ever scheduled on this stream, by absolute start sample.
    scheduled: Mutex<Vec<u64>>,
    /// The fake time of sample 0.
    stream_start: HostTime,
    invalidated: AtomicBool,
    stop: AtomicBool,
    /// The next this many packets are dropped as if the ring were full.
    drop_next: AtomicU64,
}

struct PlatformState {
    calls: Vec<PlatformCall>,
    fail_next: Option<ActionError>,
    action_time: Milliseconds,
    speech_finishes: u32,
}

struct FakeState {
    speed: f64,
    start: Instant,
    devices: Mutex<Vec<FakeDevice>>,
    device_tx: Mutex<Option<Sender<DeviceMsg>>>,
    captures: Mutex<Vec<Arc<CaptureControl>>>,
    input: Mutex<Option<InputScript>>,
    raw_input_failure: Mutex<Option<VetoUnavailableReason>>,
    /// The worker of every raw-input request that started, in order (each brought its own ring).
    raw_input_workers: Mutex<Vec<WorkerId>>,
    platform: Mutex<PlatformState>,
    free_bytes: AtomicU64,
    recorder_stall: Mutex<Milliseconds>,
}

impl FakeState {
    fn now(&self) -> HostTime {
        let real = self.start.elapsed().as_secs_f64();
        let ticks = real * self.speed * HostTime::TICKS_PER_SECOND as f64;
        HostTime(CLOCK_ORIGIN.0 + ticks as i64)
    }

    fn timeout_until(&self, deadline: HostTime) -> Duration {
        let left = deadline.since(self.now()).0;
        if left <= 0 {
            return Duration::ZERO;
        }
        Duration::from_secs_f64(left as f64 / HostTime::TICKS_PER_SECOND as f64 / self.speed)
    }

    /// Sleeps until `deadline` in fake time, waking every `POLL` to check `stop`. True if stopped.
    fn sleep_until(&self, deadline: HostTime, stop: &AtomicBool) -> bool {
        loop {
            if stop.load(Ordering::Acquire) {
                return true;
            }
            let left = self.timeout_until(deadline);
            if left.is_zero() {
                return false;
            }
            std::thread::sleep(left.min(POLL));
        }
    }

    fn send_device(&self, msg: DeviceMsg) {
        if let Some(tx) = lock(&self.device_tx).as_ref() {
            if tx.try_send(msg).is_err() {
                tracing::debug!("fake device message dropped");
            }
        }
    }
}

/// The fake OS parts (§2.11). Share it with the engine as `Arc<FakeOs>`; the test keeps a clone.
pub struct FakeOs {
    state: Arc<FakeState>,
}

impl FakeOs {
    /// Fake time runs `speed`× real time; `timeout_until` scales by it.
    pub fn new(speed: f64) -> FakeOs {
        assert!(
            speed.is_finite() && speed > 0.0,
            "FakeOs speed must be positive"
        );
        FakeOs {
            state: Arc::new(FakeState {
                speed,
                start: Instant::now(),
                devices: Mutex::new(Vec::new()),
                device_tx: Mutex::new(None),
                captures: Mutex::new(Vec::new()),
                input: Mutex::new(None),
                raw_input_failure: Mutex::new(None),
                raw_input_workers: Mutex::new(Vec::new()),
                platform: Mutex::new(PlatformState {
                    calls: Vec::new(),
                    fail_next: None,
                    action_time: Milliseconds::new(0.0),
                    speech_finishes: 1,
                }),
                free_bytes: AtomicU64::new(u64::MAX),
                recorder_stall: Mutex::new(Milliseconds::new(0.0)),
            }),
        }
    }

    /// Adds (or replaces) a device and sends `DeviceMsg::Added`. `info.probe` is what
    /// `list_devices` reports.
    pub fn add_device(&self, info: DeviceInfo, script: CaptureScript) {
        let id = info.endpoint_id.clone();
        {
            let mut devices = lock(&self.state.devices);
            devices.retain(|d| d.info.endpoint_id != id);
            devices.push(FakeDevice { info, script });
        }
        self.state.send_device(DeviceMsg::Added { id });
    }

    /// Removes a device: its running fake capture ends with `DeviceInvalidated`, then
    /// `DeviceMsg::Removed`.
    pub fn remove_device(&self, endpoint_id: &str) {
        lock(&self.state.devices).retain(|d| d.info.endpoint_id != endpoint_id);
        for c in lock(&self.state.captures).iter() {
            if c.endpoint_id == endpoint_id {
                c.invalidated.store(true, Ordering::Release);
            }
        }
        self.state.send_device(DeviceMsg::Removed {
            id: endpoint_id.to_string(),
        });
    }

    /// Drops the endpoint's next `packets` packets at the ring, as a processing stall would.
    pub fn drop_packets(&self, endpoint_id: &str, packets: u64) {
        let captures = lock(&self.state.captures);
        let Some(capture) = captures
            .iter()
            .rev()
            .find(|c| c.endpoint_id == endpoint_id && !c.stop.load(Ordering::Acquire))
        else {
            panic!("drop_packets: no running fake capture on {endpoint_id}");
        };
        capture.drop_next.fetch_add(packets, Ordering::AcqRel);
    }

    /// Mixes `taps` into the endpoint's running fake capture; each `at` is an offset from the
    /// stream's current sample position.
    pub fn release_taps(&self, endpoint_id: &str, taps: Vec<ScriptedTap>) {
        let captures = lock(&self.state.captures);
        let Some(capture) = captures
            .iter()
            .rev()
            .find(|c| c.endpoint_id == endpoint_id && !c.stop.load(Ordering::Acquire))
        else {
            panic!("release_taps: no running fake capture on {endpoint_id}");
        };
        let rate = lock(&self.state.devices)
            .iter()
            .filter(|d| d.info.endpoint_id == endpoint_id)
            .find_map(|d| {
                if let CaptureScript::Desk { format, .. } = &d.script {
                    Some(format.rate)
                } else {
                    None
                }
            })
            // INVARIANT: a running fake capture exists only for a Desk device (FailOpen never
            // opens; a Panic capture is test misuse and fails the test here).
            .expect("release_taps: the endpoint has a Desk script");
        let position = capture.position.load(Ordering::Acquire);
        let mut released = lock(&capture.released);
        let mut scheduled = lock(&capture.scheduled);
        for t in taps {
            let start = position + rate.samples_in(t.at) as u64;
            released.push((start, t.tap));
            scheduled.push(start);
        }
    }

    // SPEC: §2.11 lists no way to read where a released tap landed; EN7 needs it to compare t0
    // with the scripted onset, so the testkit adds this query.
    /// The running fake capture on `endpoint_id`: its stream start (the fake time of sample 0)
    /// and the absolute start sample of every tap scheduled on it, in release order.
    pub fn scheduled_taps(&self, endpoint_id: &str) -> Option<(HostTime, Vec<SampleIndex>)> {
        lock(&self.state.captures)
            .iter()
            .rev()
            .find(|c| c.endpoint_id == endpoint_id && !c.stop.load(Ordering::Acquire))
            .map(|c| {
                let starts = lock(&c.scheduled).iter().map(|s| SampleIndex(*s)).collect();
                (c.stream_start, starts)
            })
    }

    // SPEC: §2.11 lists no way to make the fake raw input unavailable; EN11 needs one for
    // `RetryInputVeto` after `Unavailable`, so the testkit adds this switch.
    /// The next raw-input spawn fails with `reason` (once).
    pub fn fail_next_raw_input(&self, reason: VetoUnavailableReason) {
        *lock(&self.state.raw_input_failure) = Some(reason);
    }

    // SPEC: §2.11 lists no way to see which input ring a raw-input worker got; EN11 needs it to
    // show each return to Active came with a fresh ring, so the testkit adds this query.
    /// The worker id of every raw-input request that started, oldest first. Each request carries
    /// its own new ring, so a new id here is a fresh input ring.
    pub fn raw_input_workers(&self) -> Vec<WorkerId> {
        lock(&self.state.raw_input_workers).clone()
    }

    /// Replayed by the fake raw-input worker after it reports Active (a running worker picks it
    /// up within a poll).
    pub fn input(&self, script: InputScript) {
        *lock(&self.state.input) = Some(script);
    }

    /// The next platform call returns this error.
    pub fn fail_next_action(&self, error: ActionError) {
        lock(&self.state.platform).fail_next = Some(error);
    }

    /// Every later fake action blocks this long, in fake time (default 0).
    pub fn set_action_time(&self, time: Milliseconds) {
        lock(&self.state.platform).action_time = time;
    }

    /// Every later fake Speak calls speech_finished `n` times (default 1).
    pub fn speech_finishes(&self, n: u32) {
        lock(&self.state.platform).speech_finishes = n;
    }

    /// `RecorderDisk::free_bytes` (default u64::MAX, so no test depends on the real disk).
    pub fn set_free_bytes(&self, bytes: u64) {
        self.state.free_bytes.store(bytes, Ordering::Release);
    }

    /// Every later `RecorderDisk::before_write` blocks this long, in fake time (default 0).
    pub fn stall_recorder(&self, time: Milliseconds) {
        *lock(&self.state.recorder_stall) = time;
    }

    /// Every action call with its start and end time.
    pub fn platform_log(&self) -> Vec<PlatformCall> {
        lock(&self.state.platform).calls.clone()
    }

    /// The fake clock (also `OsParts::now`).
    pub fn fake_now(&self) -> HostTime {
        self.state.now()
    }

    /// The running fake capture's position on `endpoint_id` (frames released so far).
    pub fn stream_position(&self, endpoint_id: &str) -> Option<SampleIndex> {
        lock(&self.state.captures)
            .iter()
            .rev()
            .find(|c| c.endpoint_id == endpoint_id && !c.stop.load(Ordering::Acquire))
            .map(|c| SampleIndex(c.position.load(Ordering::Acquire)))
    }
}

/// A Desk device's `DeviceInfo` as an Active endpoint with a successful probe.
pub fn desk_device(
    endpoint_id: &str,
    name: &str,
    is_default: bool,
    format: AudioFormat,
) -> DeviceInfo {
    DeviceInfo {
        endpoint_id: endpoint_id.to_string(),
        name: name.to_string(),
        is_default,
        state: DeviceState::Active,
        probe: ProbeOutcome::Probed {
            probe: DeviceProbe {
                format,
                raw_supported: true,
            },
        },
    }
}

/// The host time of stream sample `start`, with sample 0 at `stream_start`.
pub fn sample_time(stream_start: HostTime, start: SampleIndex, format: AudioFormat) -> HostTime {
    let ticks =
        i128::from(start.0) * i128::from(HostTime::TICKS_PER_SECOND) / i128::from(format.rate.hz());
    HostTime(stream_start.0.saturating_add(ticks as i64))
}

/// `t + ms`, saturating at the end of the clock (fake deadlines are never near it).
fn after(t: HostTime, ms: Milliseconds) -> HostTime {
    HostTime(t.0.saturating_add(HostDuration::from_ms(ms).0))
}

struct FlagStop(Arc<AtomicBool>);

impl StopSignal for FlagStop {
    fn signal(&self) {
        self.0.store(true, Ordering::Release);
    }
}

/// Unregisters the device watch when the engine drops it.
struct WatchGuard {
    state: Arc<FakeState>,
}

impl Drop for WatchGuard {
    fn drop(&mut self) {
        *lock(&self.state.device_tx) = None;
    }
}

impl OsParts for FakeOs {
    fn now(&self) -> HostTime {
        self.state.now()
    }

    fn timeout_until(&self, deadline: HostTime) -> Duration {
        self.state.timeout_until(deadline)
    }

    fn list_devices(&self) -> Result<Vec<DeviceInfo>, CaptureError> {
        Ok(lock(&self.state.devices)
            .iter()
            .map(|d| d.info.clone())
            .collect())
    }

    fn watch_devices(
        &self,
        tx: Sender<DeviceMsg>,
        _dropped: Arc<AtomicU64>,
    ) -> Result<DeviceWatch, CaptureError> {
        *lock(&self.state.device_tx) = Some(tx);
        Ok(DeviceWatch::new(Box::new(WatchGuard {
            state: Arc::clone(&self.state),
        })))
    }

    fn open_capture(
        &self,
        request: CaptureRequest,
    ) -> Result<(Worker, OpenedCapture), CaptureError> {
        let (info, script) = {
            let devices = lock(&self.state.devices);
            let Some(d) = devices
                .iter()
                .find(|d| d.info.endpoint_id == request.endpoint_id)
            else {
                return Err(CaptureError::DeviceNotFound {
                    endpoint_id: request.endpoint_id,
                });
            };
            (d.info.clone(), d.script.clone())
        };
        if info.state != DeviceState::Active {
            return Err(CaptureError::DeviceNotActive {
                endpoint_id: request.endpoint_id,
                state: info.state,
            });
        }
        let raw = match &info.probe {
            ProbeOutcome::Probed { probe } => probe.raw_supported,
            ProbeOutcome::NotActive => false,
            ProbeOutcome::Failed { error } => return Err(error.clone()),
        };
        let (format, period, ambient, seed, taps, panic_after) = match script {
            CaptureScript::FailOpen { error } => return Err(error),
            CaptureScript::Desk {
                format,
                period,
                ambient,
                seed,
                taps,
            } => (format, period, ambient, seed, taps, None),
            CaptureScript::Panic { after } => {
                let format = match &info.probe {
                    ProbeOutcome::Probed { probe } => probe.format,
                    ProbeOutcome::NotActive | ProbeOutcome::Failed { .. } => {
                        return Err(CaptureError::DeviceNotActive {
                            endpoint_id: request.endpoint_id,
                            state: info.state,
                        })
                    }
                };
                (
                    format,
                    Milliseconds::new(10.0),
                    Decibels::new(-80.0),
                    0,
                    Vec::new(),
                    Some(after),
                )
            }
        };
        let period_frames = format.rate.samples_in(period).max(1);
        let (mut writer, reader) = audio_ring(format, CAPTURE_RING, 1024, period_frames as u32);
        let stream_start = self.state.now();
        let control = Arc::new(CaptureControl {
            endpoint_id: request.endpoint_id.clone(),
            position: AtomicU64::new(0),
            released: Mutex::new(
                taps.iter()
                    .map(|t| (format.rate.samples_in(t.at) as u64, t.tap))
                    .collect(),
            ),
            invalidated: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            drop_next: AtomicU64::new(0),
            scheduled: Mutex::new(
                taps.iter()
                    .map(|t| format.rate.samples_in(t.at) as u64)
                    .collect(),
            ),
            stream_start,
        });
        {
            let mut captures = lock(&self.state.captures);
            captures.retain(|c| !c.stop.load(Ordering::Acquire));
            captures.push(Arc::clone(&control));
        }
        let stop = Arc::new(AtomicBool::new(false));
        let state = Arc::clone(&self.state);
        let body_stop = Arc::clone(&stop);
        let body_control = Arc::clone(&control);
        let wake = request.wake;
        let live = request.counters;
        let worker = Worker::spawn(
            request.worker,
            ThreadName::Capture,
            Box::new(FlagStop(stop)),
            request.exits,
            move || {
                let exit = run_fake_capture(
                    &state,
                    &body_control,
                    &body_stop,
                    FakeStream {
                        format,
                        period_frames,
                        ambient,
                        seed,
                        stream_start,
                        panic_after,
                    },
                    &mut writer,
                    &wake,
                    &live,
                );
                body_control.stop.store(true, Ordering::Release);
                ThreadEnd::Capture { exit }
            },
        )
        .map_err(|e| CaptureError::Unexpected {
            stage: crate::engine::protocol::CaptureStage::Spawn,
            detail: format!("{e:?}"),
        })?;
        let stream = OpenedStream {
            identity: DeviceIdentity {
                endpoint_id: info.endpoint_id,
                name: info.name,
                format,
                capture_mode: if raw {
                    CaptureMode::Raw
                } else {
                    CaptureMode::Default {
                        reason: DefaultModeReason::RawNotSupported,
                    }
                },
            },
            period,
            buffer_frames: period_frames as u32,
            mmcss: MmcssStatus::Active,
        };
        Ok((
            worker,
            OpenedCapture {
                stream,
                ring: reader,
            },
        ))
    }

    fn spawn_raw_input(&self, request: RawInputRequest) -> Result<Worker, VetoUnavailableReason> {
        if let Some(reason) = lock(&self.state.raw_input_failure).take() {
            return Err(reason);
        }
        let stop = Arc::new(AtomicBool::new(false));
        let state = Arc::clone(&self.state);
        let body_stop = Arc::clone(&stop);
        let RawInputRequest {
            worker,
            mut ring,
            tx,
            exits,
        } = request;
        lock(&self.state.raw_input_workers).push(worker);
        Worker::spawn(
            worker,
            ThreadName::RawInput,
            Box::new(FlagStop(stop)),
            exits,
            move || {
                let active = RawInputMsg {
                    worker,
                    status: RawInputStatus::Active,
                    skipped_records: 0,
                };
                if tx.try_send(active).is_err() {
                    return ThreadEnd::Finished;
                }
                while !body_stop.load(Ordering::Acquire) {
                    let script = lock(&state.input).take();
                    match script {
                        Some(script) => replay_input(&state, &body_stop, script, &mut ring),
                        None => std::thread::sleep(POLL),
                    }
                }
                ThreadEnd::Finished
            },
        )
        .map_err(|e| VetoUnavailableReason::ThreadFailed {
            message: format!("{e:?}"),
        })
    }

    fn platform(&self, observer: Arc<dyn OutputObserver>) -> Box<dyn PlatformActions> {
        Box::new(FakePlatform {
            state: Arc::clone(&self.state),
            observer,
            pending_speech: Arc::new(Mutex::new(Vec::new())),
        })
    }

    fn recorder_disk(&self) -> Arc<dyn RecorderDisk> {
        Arc::new(FakeDisk {
            state: Arc::clone(&self.state),
        })
    }
}

fn replay_input(
    state: &FakeState,
    stop: &AtomicBool,
    script: InputScript,
    ring: &mut rtrb::Producer<InputRecord>,
) {
    let start = state.now();
    let mut gap_from: Option<HostTime> = None;
    for event in script.events {
        let due = HostTime(start.0.saturating_add(event.time.0));
        if state.sleep_until(due, stop) {
            return;
        }
        let stamped = InputEvent {
            time: state.now(),
            kind: event.kind,
        };
        if let Some(from) = gap_from {
            if ring
                .push(InputRecord::Gap {
                    from,
                    to: stamped.time,
                })
                .is_ok()
            {
                gap_from = None;
            }
        }
        if ring.push(InputRecord::Event(stamped)).is_err() && gap_from.is_none() {
            gap_from = Some(stamped.time);
        }
    }
}

struct FakeStream {
    format: AudioFormat,
    period_frames: usize,
    ambient: Decibels,
    seed: u64,
    stream_start: HostTime,
    panic_after: Option<Milliseconds>,
}

fn run_fake_capture(
    state: &FakeState,
    control: &CaptureControl,
    stop: &AtomicBool,
    stream: FakeStream,
    writer: &mut crate::engine::ring::RingWriter,
    wake: &std::thread::Thread,
    live: &CaptureCounters,
) -> CaptureExit {
    let FakeStream {
        format,
        period_frames,
        ambient,
        seed,
        stream_start,
        panic_after,
    } = stream;
    let mut exit = CaptureExit {
        frames: 0,
        dropped_frames: 0,
        packets: 0,
        error: None,
    };
    let config = DeskConfig {
        floor: NoiseFloor::Pink { rms: ambient },
        ..DeskConfig::default_for(format.rate, seed)
    };
    // INVARIANT: the default desk has two microphones and every scripted format here is the
    // desk's own (format.channels = its 2 mics, §2.11).
    let mut desk = ModalDesk::new(config).expect("the default desk is valid");
    assert_eq!(
        desk.channels(),
        format.channels,
        "a Desk script's format.channels must be the desk's 2 mics"
    );
    let channels = usize::from(format.channels.get());
    let mut active: Vec<(u64, Vec<f32>)> = Vec::new();
    let mut discontinuity = false;
    let mut position: u64 = 0;
    loop {
        let end = position + period_frames as u64;
        let deadline = sample_time(stream_start, SampleIndex(end), format);
        if state.sleep_until(deadline, stop) {
            return exit;
        }
        if control.invalidated.load(Ordering::Acquire) {
            exit.error = Some(CaptureError::DeviceInvalidated);
            return exit;
        }
        if let Some(after) = panic_after {
            if deadline.since(stream_start) >= HostDuration::from_ms(after) {
                panic!("scripted capture panic");
            }
        }
        // New taps whose start is due before this packet ends.
        {
            let mut released = lock(&control.released);
            let mut i = 0;
            while i < released.len() {
                if released[i].0 < end {
                    let (at, spec) = released.swap_remove(i);
                    active.push((at, desk.tap(&spec, 0)));
                } else {
                    i += 1;
                }
            }
        }
        let mut packet = desk.floor(period_frames);
        active.retain(|(at, samples)| {
            let len = (samples.len() / channels) as u64;
            for frame in position.max(*at)..end.min(at + len) {
                let src = ((frame - at) as usize) * channels;
                let dst = ((frame - position) as usize) * channels;
                for c in 0..channels {
                    packet[dst + c] += samples[src + c];
                }
            }
            at + len > end
        });
        let meta = PacketMeta {
            start: SampleIndex(position),
            frames: period_frames as u32,
            time: BlockTime::Timestamped {
                host_time: sample_time(stream_start, SampleIndex(position), format),
            },
            discontinuity,
        };
        let scripted_drop = control
            .drop_next
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
            .is_ok();
        let outcome = if scripted_drop {
            PushOutcome::Dropped {
                frames: period_frames as u32,
            }
        } else {
            writer.push(meta, &packet)
        };
        match outcome {
            PushOutcome::Written => discontinuity = false,
            PushOutcome::Dropped { frames } => {
                exit.dropped_frames += u64::from(frames);
                live.dropped_frames
                    .fetch_add(u64::from(frames), Ordering::Relaxed);
                discontinuity = true;
            }
        }
        exit.frames += period_frames as u64;
        exit.packets += 1;
        live.packets.fetch_add(1, Ordering::Relaxed);
        position = end;
        control.position.store(position, Ordering::Release);
        wake.unpark();
    }
}

/// The fake recorder disk: free space and a per-write stall from `FakeOs`.
struct FakeDisk {
    state: Arc<FakeState>,
}

impl RecorderDisk for FakeDisk {
    fn free_bytes(&self, _dir: &Path) -> std::io::Result<u64> {
        Ok(self.state.free_bytes.load(Ordering::Acquire))
    }

    fn before_write(&self) {
        let stall = *lock(&self.state.recorder_stall);
        if stall.get() > 0.0 {
            let until = after(self.state.now(), stall);
            let never = AtomicBool::new(false);
            self.state.sleep_until(until, &never);
        }
    }
}

/// A Speak's pending tts end callbacks.
struct PendingSpeech {
    due: HostTime,
    remaining: u32,
}

/// The fake platform (§2.11): records every call into `FakeOs`'s log. Speak reports
/// `output_started(Speech)`, returns after the action time, and calls `speech_finished` 500 ms
/// (fake) later from its own timer, as tts's end callback would; `stop_speech` fires every pending
/// `speech_finished` at once; PlaySound reports Began/Ended around it.
pub struct FakePlatform {
    state: Arc<FakeState>,
    observer: Arc<dyn OutputObserver>,
    pending_speech: Arc<Mutex<Vec<PendingSpeech>>>,
}

impl FakePlatform {
    /// Runs one recorded call: blocks for the action time, then returns the pending failure, if
    /// any.
    fn call(&self, action: ActionSpec) -> Result<(), ActionError> {
        let started = self.state.now();
        let (time, fail) = {
            let mut p = lock(&self.state.platform);
            (p.action_time, p.fail_next.take())
        };
        if time.get() > 0.0 {
            let until = after(started, time);
            let never = AtomicBool::new(false);
            self.state.sleep_until(until, &never);
        }
        let ended = self.state.now();
        lock(&self.state.platform).calls.push(PlatformCall {
            action,
            started,
            ended,
        });
        match fail {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn schedule_speech_end(&self, finishes: u32) {
        let due = after(self.state.now(), SPEECH_END_AFTER);
        lock(&self.pending_speech).push(PendingSpeech {
            due,
            remaining: finishes,
        });
        let state = Arc::clone(&self.state);
        let pending = Arc::clone(&self.pending_speech);
        let observer = Arc::clone(&self.observer);
        let spawned = std::thread::Builder::new()
            .name("fake-tts".to_string())
            .spawn(move || {
                let never = AtomicBool::new(false);
                state.sleep_until(due, &never);
                // Fire every finish now due, oldest first (stop_speech may have fired them).
                let fired: Vec<PendingSpeech> = {
                    let mut p = lock(&pending);
                    let now = state.now();
                    let (due, later): (Vec<_>, Vec<_>) = p.drain(..).partition(|s| s.due <= now);
                    *p = later;
                    due
                };
                for s in fired {
                    for _ in 0..s.remaining {
                        observer.speech_finished();
                    }
                }
            });
        if let Err(e) = spawned {
            panic!("the fake tts timer thread could not start: {e}");
        }
    }
}

impl PlatformActions for FakePlatform {
    fn open_uri(&self, uri: &str) -> Result<(), ActionError> {
        self.call(ActionSpec::OpenUrl {
            url: uri.to_string(),
        })
    }

    fn open_path(&self, path: &Path) -> Result<(), ActionError> {
        // OpenApp and OpenPath both reach open_path; the log records OpenPath.
        self.call(ActionSpec::OpenPath {
            path: path.to_string_lossy().into_owned(),
        })
    }

    fn run_command(&self, cmd: &str, args: &[String]) -> Result<Child, ActionError> {
        self.call(ActionSpec::RunCommand {
            command: cmd.to_string(),
            args: args.to_vec(),
        })?;
        // A real, immediately exiting child: `Child` has no public constructor. The scripted
        // command itself never runs.
        Command::new("cmd")
            .args(["/C", "exit", "0"])
            .spawn()
            .map_err(|e| ActionError::LaunchFailed {
                target: cmd.to_string(),
                reason: e.to_string(),
            })
    }

    fn copy_to_clipboard(&self, text: &str) -> Result<(), ActionError> {
        self.call(ActionSpec::CopyToClipboard {
            text: text.to_string(),
        })
    }

    fn speak(&self, text: &str) -> Result<(), ActionError> {
        let _id: OutputId = self.observer.output_started(SelfOutputKind::Speech);
        let finishes = lock(&self.state.platform).speech_finishes;
        let result = self.call(ActionSpec::Speak {
            text: text.to_string(),
        });
        // Every Began{Speech} gets its finishes, whether the speech succeeds or fails (§2.6).
        self.schedule_speech_end(finishes);
        result
    }

    fn send_keystroke(&self, combo: &KeyCombo) -> Result<(), ActionError> {
        self.call(ActionSpec::SendKeystroke {
            combo: combo.clone(),
        })
    }

    fn play_sound(&self, sound: SystemSound) -> Result<(), ActionError> {
        let id = self.observer.output_started(SelfOutputKind::Sound);
        let result = self.call(ActionSpec::PlaySound { sound });
        self.observer.output_ended(id);
        result
    }

    fn screenshot_to_clipboard(&self, mode: ScreenshotMode) -> Result<Screenshot, ActionError> {
        self.call(ActionSpec::Screenshot { mode })
            .map(|()| Screenshot::Copied)
    }

    fn notify(&self, toast: &ToastSpec) -> Result<(), ActionError> {
        self.observer.output_started(SelfOutputKind::Toast);
        self.call(ActionSpec::ShowToast {
            title: toast.title.clone(),
            body: toast.body.clone(),
        })
    }

    /// The observer is fixed at construction (`FakeOs::platform(observer)`); the engine never
    /// registers a second one.
    fn set_output_observer(&self, observer: Arc<dyn OutputObserver>) {
        drop(observer);
    }

    /// Ends all queued speech: each pending finish fires now, in order (§2.6 `stop_speech`).
    fn stop_speech(&self) {
        let pending: Vec<PendingSpeech> = std::mem::take(&mut *lock(&self.pending_speech));
        for p in pending {
            for _ in 0..p.remaining {
                self.observer.speech_finished();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::messages::{ThreadExit, WorkerId};
    use qwerty_core::units::{ChannelCount, SampleRate};

    fn format() -> AudioFormat {
        AudioFormat {
            rate: SampleRate::new(48_000).expect("rate"),
            channels: ChannelCount::new(2).expect("channels"),
        }
    }

    #[test]
    fn fake_capture_stamps_packets_on_the_fake_clock() {
        let os = FakeOs::new(4.0);
        os.add_device(
            desk_device("mic-a", "Mic A", true, format()),
            CaptureScript::Desk {
                format: format(),
                period: Milliseconds::new(10.0),
                ambient: Decibels::new(-80.0),
                seed: 3,
                taps: Vec::new(),
            },
        );
        let (exits_tx, exits_rx) = crossbeam_channel::bounded::<ThreadExit>(16);
        let (worker, mut opened) = os
            .open_capture(CaptureRequest {
                worker: WorkerId(1),
                endpoint_id: "mic-a".to_string(),
                wake: std::thread::current(),
                exits: exits_tx,
                counters: Arc::default(),
            })
            .expect("open");
        let mut buf = Vec::new();
        let mut packets = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while packets.len() < 10 && Instant::now() < deadline {
            match opened.ring.pop(&mut buf) {
                Some(meta) => packets.push(meta),
                None => std::thread::park_timeout(Duration::from_millis(20)),
            }
        }
        assert_eq!(packets.len(), 10);
        let BlockTime::Timestamped { host_time: first } = packets[0].time else {
            panic!("untimed packet");
        };
        for (i, p) in packets.iter().enumerate() {
            assert_eq!(p.start, SampleIndex(480 * i as u64));
            assert_eq!(
                p.time,
                BlockTime::Timestamped {
                    host_time: sample_time(first, p.start, format()),
                }
            );
        }
        worker.stop(Duration::from_secs(2)).expect("stopped");
        let exit = exits_rx.recv_timeout(Duration::from_secs(2)).expect("exit");
        assert!(matches!(exit.end, ThreadEnd::Capture { exit } if exit.error.is_none()));
    }

    #[test]
    fn timeout_until_scales_by_speed() {
        let os = FakeOs::new(10.0);
        let now = os.fake_now();
        let later = now
            .checked_add(HostDuration::from_ms(Milliseconds::new(1000.0)))
            .expect("in range");
        let wait = os.timeout_until(later);
        assert!(wait <= Duration::from_millis(100) && wait > Duration::from_millis(50));
        assert_eq!(os.timeout_until(now), Duration::ZERO);
    }
}
