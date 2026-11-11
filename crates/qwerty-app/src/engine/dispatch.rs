//! Dispatch thread (p2 §2.6). One COM STA thread for the engine's life that owns ONE platform (so
//! speech is never cut off by a dropped per-tap instance) and runs every chain in order (so chains
//! never overlap). It waits on its job event and its message queue at once, so the STA pumps and
//! COM, WinRT and shell callbacks complete while it is parked. Started RunCommand children are
//! watched for `COMMAND_WATCH` (PARITY-15).

use std::any::Any;
use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::process::Child;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender, TrySendError};
use windows::Win32::Foundation::WAIT_FAILED;
use windows::Win32::System::Com::{CoInitializeEx, CoUninitialize, COINIT_APARTMENTTHREADED};
use windows::Win32::System::Threading::INFINITE;
use windows::Win32::UI::WindowsAndMessaging::{
    DispatchMessageW, MsgWaitForMultipleObjectsEx, PeekMessageW, TranslateMessage, MSG,
    MWMO_INPUTAVAILABLE, PM_REMOVE, QS_ALLINPUT,
};

use qwerty_core::units::{HostDuration, HostTime};

use crate::engine::messages::{
    DispatchJob, DispatchMsg, DispatchQueue, JobEvent, JobKey, OsParts, SelfOutputLink, StopSignal,
    ThreadEnd, ThreadExit, Worker, WorkerId,
};
use crate::engine::protocol::{
    ActionResult, CommandOutcome, EngineError, StepOutcome, ThreadName, COMMAND_WATCH,
    MAX_WATCHED_COMMANDS,
};
use crate::platform::{self, ActionError, Dispatched, OutputObserver, PlatformActions};

/// The controller → dispatch job queue (p2 §2.11: bounded 16).
const JOB_QUEUE: usize = 16;

/// The wait while children are watched: each is polled with `try_wait` this often (p2 §2.6).
const WATCH_POLL_MS: u32 = 100;

/// Spawns the dispatch thread (p2 §2.6). The returned queue is the controller's only way in; its
/// `try_send` sets the job event, which is also how `Worker::stop` wakes the thread
/// (StopSignal = AtomicBool + SetEvent).
pub fn spawn(
    worker: WorkerId,
    tx: Sender<DispatchMsg>,
    link: Arc<SelfOutputLink>,
    os: Arc<dyn OsParts>,
    exits: Sender<ThreadExit>,
) -> Result<(Worker, DispatchQueue), EngineError> {
    let event = Arc::new(JobEvent::new().map_err(|e| EngineError::ThreadSpawn {
        thread: ThreadName::Dispatch,
        detail: format!("CreateEventW for the job event failed: {}", e.message()),
    })?);
    let (jobs_tx, jobs) = crossbeam_channel::bounded(JOB_QUEUE);
    let stop = Arc::new(AtomicBool::new(false));
    let signal = DispatchStop {
        stop: Arc::clone(&stop),
        event: Arc::clone(&event),
    };
    let thread = DispatchThread {
        jobs,
        event: Arc::clone(&event),
        stop,
        out: Outbox { tx, os },
        link,
    };
    let handle = Worker::spawn(
        worker,
        ThreadName::Dispatch,
        Box::new(signal),
        exits,
        move || thread.run(),
    )?;
    Ok((handle, DispatchQueue::new(jobs_tx, event)))
}

/// Sets the stop flag, then the job event, so a parked thread wakes and sees it. Idempotent and
/// non-blocking.
struct DispatchStop {
    stop: Arc<AtomicBool>,
    event: Arc<JobEvent>,
}

impl StopSignal for DispatchStop {
    fn signal(&self) {
        self.stop.store(true, Ordering::Release);
        self.event.set();
    }
}

/// What the dispatch thread sends, and the clock it stamps with.
struct Outbox {
    tx: Sender<DispatchMsg>,
    os: Arc<dyn OsParts>,
}

impl Outbox {
    /// `try_send`: no worker ever blocks on the controller (RUST-RT-5).
    // SPEC: §2.11 counts a failed dispatch → controller send as controller_dropped, but §2.6's
    // spawn signature carries no counter; the channel cannot fill (16 queued jobs give ≤ 34
    // messages, §2.11), so a Full is logged as the broken invariant it is.
    fn send(&self, msg: DispatchMsg) {
        match self.tx.try_send(msg) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                tracing::error!("the dispatch to controller channel was full; a message was lost");
            }
            Err(TrySendError::Disconnected(_)) => {
                tracing::debug!("dispatch message not delivered: the controller has stopped");
            }
        }
    }

    fn now(&self) -> HostTime {
        self.os.now()
    }
}

struct DispatchThread {
    jobs: Receiver<DispatchJob>,
    event: Arc<JobEvent>,
    stop: Arc<AtomicBool>,
    out: Outbox,
    link: Arc<SelfOutputLink>,
}

impl DispatchThread {
    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    /// `CoInitializeEx(COINIT_APARTMENTTHREADED)` once, for the thread's life (RUST-RT-18).
    fn run(self) -> ThreadEnd {
        // SAFETY: no reserved pointer; this thread has no apartment yet.
        let com = unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED) };
        if com.is_err() {
            // SPEC: §2.6 gives no outcome for a failed STA init and ThreadEnd has no variant for
            // it; it ends the thread the way a panic would, so the controller faults the engine.
            return ThreadEnd::Panicked {
                message: "CoInitializeEx(COINIT_APARTMENTTHREADED) failed on the dispatch thread"
                    .to_string(),
            };
        }
        let end = self.serve();
        // SAFETY: balances the successful CoInitializeEx above; the platform is already dropped.
        unsafe { CoUninitialize() };
        end
    }

    fn serve(self) -> ThreadEnd {
        let mut platform = self.build_platform();
        self.out.send(DispatchMsg::PlatformReady);
        let mut watched = Watched::default();
        let end = loop {
            if self.stopped() {
                break ThreadEnd::Finished;
            }
            if !self.wait(watched.is_empty()) {
                break ThreadEnd::Panicked {
                    message: "MsgWaitForMultipleObjectsEx failed on the dispatch thread"
                        .to_string(),
                };
            }
            pump_messages();
            while !self.stopped() {
                let Ok(job) = self.jobs.try_recv() else {
                    break;
                };
                platform = self.run_job(platform, &mut watched, job);
            }
            watched.poll(&self.out);
        };
        // Shutdown (§2.11 step 5): the current action has finished; end the speech queue, then
        // drop the platform. Watched children are released, never killed or waited on.
        stop_speech(platform.as_ref());
        drop(platform);
        drop(watched);
        end
    }

    fn build_platform(&self) -> Box<dyn PlatformActions> {
        let observer: Arc<dyn OutputObserver> = self.link.clone();
        self.out.os.platform(observer)
    }

    /// `MsgWaitForMultipleObjectsEx([job event], timeout, QS_ALLINPUT, MWMO_INPUTAVAILABLE)`:
    /// INFINITE while no child is watched, 100 ms otherwise. False only if the wait itself failed.
    fn wait(&self, no_children: bool) -> bool {
        let timeout = if no_children { INFINITE } else { WATCH_POLL_MS };
        let handles = [self.event.handle()];
        // SAFETY: the job event stays open while `self.event` (an Arc) lives.
        let woke = unsafe {
            MsgWaitForMultipleObjectsEx(Some(&handles), timeout, QS_ALLINPUT, MWMO_INPUTAVAILABLE)
        };
        woke != WAIT_FAILED
    }

    /// Runs one chain sequentially, in order, then sends `Completed`. `started_at` is `os.now()`
    /// before the first action and `finished_at` after the last (§2.12).
    fn run_job(
        &self,
        mut platform: Box<dyn PlatformActions>,
        watched: &mut Watched,
        job: DispatchJob,
    ) -> Box<dyn PlatformActions> {
        let started_at = self.out.now();
        let mut steps = Vec::with_capacity(job.actions.len());
        let mut panicked = None;
        for (step, action) in (0u32..).zip(job.actions.iter()) {
            if step > 0 && self.stopped() {
                // Shutdown: the current action finished; the rest of the chain is abandoned with
                // the engine, so no Completed is sent for it.
                return platform;
            }
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                platform::dispatch(platform.as_ref(), action)
            }));
            match outcome {
                Ok(Ok(Dispatched::Done)) => steps.push(StepOutcome::Done),
                Ok(Ok(Dispatched::Spawned(child))) => {
                    steps.push(StepOutcome::Started);
                    watched.add(&self.out, job.key, step, child);
                }
                Ok(Ok(Dispatched::Canceled)) => {
                    steps.push(StepOutcome::Canceled);
                    break;
                }
                Ok(Err(error)) => {
                    steps.push(StepOutcome::Failed { error });
                    break;
                }
                Err(payload) => {
                    let message = panic_text(payload.as_ref());
                    steps.push(StepOutcome::Failed {
                        error: ActionError::Panicked {
                            message: message.clone(),
                        },
                    });
                    panicked = Some(message);
                    break;
                }
            }
        }
        let finished_at = self.out.now();
        if let Some(panic) = panicked {
            platform = self.rebuild(platform, panic);
        }
        self.out.send(DispatchMsg::Completed {
            key: job.key,
            result: ActionResult {
                started_at,
                finished_at,
                steps,
            },
        });
        platform
    }

    /// After an action panic: the old platform's speech queue is ended, the platform is dropped
    /// and a new one built, then `PlatformRebuilt` (§2.6).
    fn rebuild(&self, old: Box<dyn PlatformActions>, panic: String) -> Box<dyn PlatformActions> {
        stop_speech(old.as_ref());
        drop(old);
        let platform = self.build_platform();
        self.out.send(DispatchMsg::PlatformRebuilt { panic });
        platform
    }
}

/// `platform.stop_speech()` under `catch_unwind`: a panic there is logged and changes nothing
/// else.
fn stop_speech(platform: &dyn PlatformActions) {
    if catch_unwind(AssertUnwindSafe(|| platform.stop_speech())).is_err() {
        tracing::error!("stop_speech panicked on the dispatch thread");
    }
}

/// `PeekMessageW(PM_REMOVE)` + `TranslateMessage` + `DispatchMessageW` until the queue is empty.
fn pump_messages() {
    let mut msg = MSG::default();
    // SAFETY: `msg` is a valid MSG for each call; None reads every message of this thread.
    while unsafe { PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE) }.as_bool() {
        // SAFETY: `msg` was just filled by PeekMessageW.
        unsafe {
            // The result says only whether a character message was posted.
            let _translated = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

/// The text of a panic payload: &str or String; anything else is "non-string panic payload".
// The same rule as `messages::Worker::spawn`, whose helper is private to that module.
fn panic_text(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

struct WatchedChild {
    key: JobKey,
    step: u32,
    child: Child,
    until: HostTime,
}

/// Started children (PARITY-15): at most `MAX_WATCHED_COMMANDS`, each polled with `try_wait` for
/// `COMMAND_WATCH`. A released child is dropped: never killed, never waited on.
#[derive(Default)]
struct Watched {
    children: VecDeque<WatchedChild>,
}

impl Watched {
    fn is_empty(&self) -> bool {
        self.children.is_empty()
    }

    /// A child beyond the limit releases the oldest as `StillRunning`.
    fn add(&mut self, out: &Outbox, key: JobKey, step: u32, child: Child) {
        if self.children.len() == MAX_WATCHED_COMMANDS {
            if let Some(oldest) = self.children.pop_front() {
                out.send(DispatchMsg::CommandFinished {
                    key: oldest.key,
                    step: oldest.step,
                    outcome: CommandOutcome::StillRunning,
                });
            }
        }
        // INVARIANT: a QPC host time plus 4 s cannot overflow i64 ticks (~29 000 years).
        let until = out
            .now()
            .checked_add(HostDuration::from_ms(COMMAND_WATCH))
            .expect("now + COMMAND_WATCH fits the tick range");
        self.children.push_back(WatchedChild {
            key,
            step,
            child,
            until,
        });
    }

    /// An exit gives `Exited{code}`; at `COMMAND_WATCH`, `StillRunning` and the child is released;
    /// a `try_wait` error gives `WaitFailed`.
    fn poll(&mut self, out: &Outbox) {
        if self.children.is_empty() {
            return;
        }
        let now = out.now();
        self.children.retain_mut(|watched| {
            let outcome = match watched.child.try_wait() {
                Ok(Some(status)) => CommandOutcome::Exited {
                    code: status.code(),
                },
                Ok(None) if now >= watched.until => CommandOutcome::StillRunning,
                Ok(None) => return true,
                Err(e) => CommandOutcome::WaitFailed {
                    detail: e.to_string(),
                },
            };
            out.send(DispatchMsg::CommandFinished {
                key: watched.key,
                step: watched.step,
                outcome,
            });
            false
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::os::windows::process::CommandExt;
    use std::path::Path;
    use std::process::{Command, Stdio};
    use std::sync::atomic::{AtomicU64, AtomicUsize};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    use qwerty_core::profile::{ActionSpec, KeyCombo, ScreenshotMode, SystemSound, ZoneId};
    use qwerty_core::veto::{OutputId, SelfOutputEvent, SelfOutputKind};
    use windows::core::w;
    use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DefWindowProcW, PostMessageW, RegisterClassW, HWND_MESSAGE,
        WINDOW_EX_STYLE, WINDOW_STYLE, WM_APP, WNDCLASSW,
    };

    use crate::engine::clock;
    use crate::engine::messages::{
        CaptureRequest, DeviceMsg, DeviceWatch, OpenedCapture, RawInputRequest, RecorderDisk,
        SpeechEnds,
    };
    use crate::engine::protocol::{CaptureError, DeviceInfo, JobOrigin, VetoUnavailableReason};
    use crate::platform::{Screenshot, ToastSpec};

    /// Run a launched command without flashing a console window.
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    type Log = Arc<Mutex<Vec<String>>>;

    fn push(log: &Log, entry: String) {
        log.lock().expect("log lock").push(entry);
    }

    fn snapshot(log: &Log) -> Vec<String> {
        log.lock().expect("log lock").clone()
    }

    /// Records every observer call into the shared log, then forwards it to the engine's link, so
    /// the self-output channel sees exactly what the platform reported.
    struct FakeObserver {
        log: Log,
        link: Arc<dyn OutputObserver>,
    }

    impl OutputObserver for FakeObserver {
        fn output_started(&self, kind: SelfOutputKind) -> OutputId {
            let id = self.link.output_started(kind);
            push(&self.log, format!("observer started {kind:?} {}", id.0));
            id
        }

        fn output_ended(&self, id: OutputId) {
            push(&self.log, format!("observer ended {}", id.0));
            self.link.output_ended(id);
        }

        fn speech_finished(&self) {
            push(&self.log, "observer speech_finished".to_string());
            self.link.speech_finished();
        }
    }

    /// A scripted platform: `open_uri("fail")` fails, `open_uri("panic")` panics, sounds and speech
    /// report to the observer, `run_command` starts a real child. Every call is logged with the
    /// instance number, and so is its drop.
    struct FakePlatform {
        n: usize,
        log: Log,
        observer: RefCell<Option<Arc<dyn OutputObserver>>>,
    }

    impl FakePlatform {
        fn record(&self, call: &str) {
            push(&self.log, format!("p{} {call}", self.n));
        }

        fn observer(&self) -> Arc<dyn OutputObserver> {
            // INVARIANT: TestOs::platform sets the observer before returning the platform.
            self.observer
                .borrow()
                .clone()
                .expect("the observer is set at construction")
        }
    }

    impl Drop for FakePlatform {
        fn drop(&mut self) {
            self.record("drop");
        }
    }

    impl PlatformActions for FakePlatform {
        fn open_uri(&self, uri: &str) -> Result<(), ActionError> {
            self.record(&format!("open_uri {uri}"));
            match uri {
                "fail" => Err(ActionError::TargetNotFound {
                    target: uri.to_string(),
                }),
                "panic" => panic!("scripted action panic"),
                _ => Ok(()),
            }
        }
        fn open_path(&self, _: &Path) -> Result<(), ActionError> {
            self.record("open_path");
            Ok(())
        }
        fn run_command(&self, cmd: &str, args: &[String]) -> Result<Child, ActionError> {
            self.record("run_command");
            Command::new(cmd)
                .args(args)
                .creation_flags(CREATE_NO_WINDOW)
                .spawn()
                .map_err(|e| ActionError::LaunchFailed {
                    target: cmd.to_string(),
                    reason: e.to_string(),
                })
        }
        fn copy_to_clipboard(&self, text: &str) -> Result<(), ActionError> {
            self.record(&format!("copy_to_clipboard {text}"));
            Ok(())
        }
        fn speak(&self, _: &str) -> Result<(), ActionError> {
            self.record("speak");
            let observer = self.observer();
            observer.output_started(SelfOutputKind::Speech);
            observer.speech_finished();
            Ok(())
        }
        fn send_keystroke(&self, _: &KeyCombo) -> Result<(), ActionError> {
            self.record("send_keystroke");
            Ok(())
        }
        fn play_sound(&self, _: SystemSound) -> Result<(), ActionError> {
            self.record("play_sound");
            let observer = self.observer();
            let id = observer.output_started(SelfOutputKind::Sound);
            observer.output_ended(id);
            Ok(())
        }
        fn screenshot_to_clipboard(&self, _: ScreenshotMode) -> Result<Screenshot, ActionError> {
            self.record("screenshot_to_clipboard");
            Ok(Screenshot::Canceled)
        }
        fn notify(&self, _: &ToastSpec) -> Result<(), ActionError> {
            self.record("notify");
            Ok(())
        }
        fn set_output_observer(&self, observer: Arc<dyn OutputObserver>) {
            self.record("set_output_observer");
            *self.observer.borrow_mut() = Some(observer);
        }
        fn stop_speech(&self) {
            self.record("stop_speech");
        }
    }

    /// The OS seam for these tests: the real clock, and FakePlatforms wrapped in a FakeObserver.
    /// `on_platform` runs on the dispatch thread inside `platform()` (DP7 makes its window there).
    struct TestOs {
        log: Log,
        built: AtomicUsize,
        on_platform: Option<Box<dyn Fn() + Send + Sync>>,
    }

    impl OsParts for TestOs {
        fn now(&self) -> HostTime {
            clock::now()
        }
        fn timeout_until(&self, deadline: HostTime) -> Duration {
            let ticks = deadline.since(clock::now()).0.max(0);
            Duration::from_nanos(ticks.unsigned_abs() * 100)
        }
        fn list_devices(&self) -> Result<Vec<DeviceInfo>, CaptureError> {
            unimplemented!("the dispatch tests use only now, timeout_until and platform")
        }
        fn watch_devices(
            &self,
            _: Sender<DeviceMsg>,
            _: Arc<std::sync::atomic::AtomicU64>,
        ) -> Result<DeviceWatch, CaptureError> {
            unimplemented!("the dispatch tests use only now, timeout_until and platform")
        }
        fn open_capture(&self, _: CaptureRequest) -> Result<(Worker, OpenedCapture), CaptureError> {
            unimplemented!("the dispatch tests use only now, timeout_until and platform")
        }
        fn spawn_raw_input(&self, _: RawInputRequest) -> Result<Worker, VetoUnavailableReason> {
            unimplemented!("the dispatch tests use only now, timeout_until and platform")
        }
        fn platform(&self, observer: Arc<dyn OutputObserver>) -> Box<dyn PlatformActions> {
            if let Some(hook) = &self.on_platform {
                hook();
            }
            let platform = FakePlatform {
                n: self.built.fetch_add(1, Ordering::SeqCst),
                log: Arc::clone(&self.log),
                observer: RefCell::new(None),
            };
            platform.set_output_observer(Arc::new(FakeObserver {
                log: Arc::clone(&self.log),
                link: observer,
            }));
            Box::new(platform)
        }
        fn recorder_disk(&self) -> Arc<dyn RecorderDisk> {
            unimplemented!("the dispatch tests use only now, timeout_until and platform")
        }
    }

    struct Harness {
        worker: Option<Worker>,
        queue: DispatchQueue,
        rx: Receiver<DispatchMsg>,
        began: Receiver<SelfOutputEvent>,
        exits: Receiver<ThreadExit>,
        log: Log,
        next_key: u64,
    }

    impl Harness {
        fn start(on_platform: Option<Box<dyn Fn() + Send + Sync>>) -> Harness {
            let log: Log = Arc::default();
            let os = Arc::new(TestOs {
                log: Arc::clone(&log),
                built: AtomicUsize::new(0),
                on_platform,
            });
            let (began_tx, began) = crossbeam_channel::bounded(64);
            let link = Arc::new(SelfOutputLink::new(
                began_tx,
                Arc::new(SpeechEnds::default()),
                Arc::clone(&os) as Arc<dyn OsParts>,
            ));
            let (tx, rx) = crossbeam_channel::bounded(64);
            let (exits_tx, exits) = crossbeam_channel::bounded(16);
            let (worker, queue) =
                spawn(WorkerId(7), tx, link, os, exits_tx).expect("spawn the dispatch thread");
            let harness = Harness {
                worker: Some(worker),
                queue,
                rx,
                began,
                exits,
                log,
                next_key: 0,
            };
            match harness.next(Duration::from_secs(5)) {
                DispatchMsg::PlatformReady => {}
                other => panic!("expected PlatformReady, got {}", describe(&other)),
            }
            harness
        }

        fn send(&mut self, actions: Vec<ActionSpec>) -> JobKey {
            self.next_key += 1;
            let key = JobKey(self.next_key);
            let job = DispatchJob {
                key,
                zone: ZoneId::new(),
                actions,
                origin: JobOrigin::TestChain,
                tap: None,
            };
            if self.queue.try_send(job).is_err() {
                panic!("the dispatch queue refused a job");
            }
            key
        }

        fn next(&self, limit: Duration) -> DispatchMsg {
            self.rx
                .recv_timeout(limit)
                .expect("a dispatch message within the limit")
        }

        fn completed(&self, key: JobKey, limit: Duration) -> ActionResult {
            match self.next(limit) {
                DispatchMsg::Completed { key: got, result } if got == key => result,
                other => panic!("expected Completed for {key:?}, got {}", describe(&other)),
            }
        }

        /// Stops the thread and returns the log: the platform's last calls are its stop_speech,
        /// then its drop.
        fn shutdown(mut self) -> Vec<String> {
            let worker = self.worker.take().expect("running");
            worker
                .stop(Duration::from_secs(2))
                .expect("the dispatch thread stops in time");
            let exit = self
                .exits
                .recv_timeout(Duration::from_secs(1))
                .expect("a thread exit");
            assert_eq!(exit.thread, ThreadName::Dispatch);
            assert_eq!(exit.worker, WorkerId(7));
            assert_eq!(exit.end, ThreadEnd::Finished);
            snapshot(&self.log)
        }
    }

    fn describe(msg: &DispatchMsg) -> String {
        match msg {
            DispatchMsg::PlatformReady => "PlatformReady".to_string(),
            DispatchMsg::Completed { key, result } => format!("Completed {key:?} {result:?}"),
            DispatchMsg::CommandFinished { key, step, outcome } => {
                format!("CommandFinished {key:?} step {step} {outcome:?}")
            }
            DispatchMsg::PlatformRebuilt { panic } => format!("PlatformRebuilt {panic}"),
        }
    }

    fn url(target: &str) -> ActionSpec {
        ActionSpec::OpenUrl {
            url: target.to_string(),
        }
    }

    fn command(cmd: &str, args: &[&str]) -> ActionSpec {
        ActionSpec::RunCommand {
            command: cmd.to_string(),
            args: args.iter().map(|a| a.to_string()).collect(),
        }
    }

    #[test]
    fn dp1_a_chain_runs_in_order_and_stops_at_the_first_error() {
        let mut h = Harness::start(None);
        let key = h.send(vec![
            ActionSpec::PlaySound {
                sound: SystemSound::Success,
            },
            ActionSpec::CopyToClipboard {
                text: "a".to_string(),
            },
            url("fail"),
            ActionSpec::Speak {
                text: "never".to_string(),
            },
            url("never"),
        ]);
        let result = h.completed(key, Duration::from_secs(5));
        assert_eq!(
            result.steps,
            vec![
                StepOutcome::Done,
                StepOutcome::Done,
                StepOutcome::Failed {
                    error: ActionError::TargetNotFound {
                        target: "fail".to_string()
                    }
                },
            ]
        );
        assert!(result.started_at <= result.finished_at);
        // The FakeObserver forwards to the engine's link: processing sees Began/Ended for the
        // sound that ran and nothing for the speech after the failure.
        let events: Vec<SelfOutputEvent> = h.began.try_iter().collect();
        assert_eq!(events.len(), 2, "{events:?}");
        assert!(matches!(
            events[0],
            SelfOutputEvent::Began {
                id: OutputId(0),
                kind: SelfOutputKind::Sound,
                ..
            }
        ));
        assert!(matches!(
            events[1],
            SelfOutputEvent::Ended {
                id: OutputId(0),
                ..
            }
        ));
        let log = h.shutdown();
        assert_eq!(
            log,
            vec![
                "p0 set_output_observer",
                "p0 play_sound",
                "observer started Sound 0",
                "observer ended 0",
                "p0 copy_to_clipboard a",
                "p0 open_uri fail",
                "p0 stop_speech",
                "p0 drop",
            ]
        );
    }

    #[test]
    fn dp2_a_panicking_action_rebuilds_the_platform_and_the_next_job_runs() {
        let mut h = Harness::start(None);
        let key = h.send(vec![url("panic"), url("never")]);
        match h.next(Duration::from_secs(5)) {
            DispatchMsg::PlatformRebuilt { panic } => {
                assert_eq!(panic, "scripted action panic");
                // Snapshot at once: the old platform's stop_speech ran before its drop, and both
                // before the new platform was built and PlatformRebuilt sent.
                let log = snapshot(&h.log);
                let at = |entry: &str| {
                    log.iter()
                        .position(|e| e == entry)
                        .unwrap_or_else(|| panic!("{entry} missing from {log:?}"))
                };
                assert!(at("p0 stop_speech") < at("p0 drop"));
                assert!(at("p0 drop") < at("p1 set_output_observer"));
            }
            other => panic!("expected PlatformRebuilt, got {}", describe(&other)),
        }
        let result = h.completed(key, Duration::from_secs(5));
        assert_eq!(
            result.steps,
            vec![StepOutcome::Failed {
                error: ActionError::Panicked {
                    message: "scripted action panic".to_string()
                }
            }]
        );

        let next = h.send(vec![ActionSpec::CopyToClipboard {
            text: "after".to_string(),
        }]);
        assert_eq!(
            h.completed(next, Duration::from_secs(5)).steps,
            vec![StepOutcome::Done]
        );
        let log = h.shutdown();
        assert!(!log.iter().any(|e| e.contains("open_uri never")), "{log:?}");
        assert!(log.contains(&"p1 copy_to_clipboard after".to_string()));
        assert_eq!(
            &log[log.len() - 2..],
            &["p1 stop_speech".to_string(), "p1 drop".to_string()]
        );
    }

    #[test]
    fn dp5_a_child_is_watched_until_it_exits_or_for_four_seconds() {
        let marker_dir = std::env::temp_dir().join(format!(
            "qwerty-dp5-{}-{}",
            std::process::id(),
            clock::now().0
        ));
        std::fs::create_dir_all(&marker_dir).expect("marker dir");
        let marker = marker_dir.join("still-ran.txt");

        let mut h = Harness::start(None);
        let exits = h.send(vec![command("cmd", &["/C", "exit 3"])]);
        assert_eq!(
            h.completed(exits, Duration::from_secs(5)).steps,
            vec![StepOutcome::Started]
        );
        match h.next(Duration::from_secs(3)) {
            DispatchMsg::CommandFinished { key, step, outcome } => {
                assert_eq!((key, step), (exits, 0));
                assert_eq!(outcome, CommandOutcome::Exited { code: Some(3) });
            }
            other => panic!("expected CommandFinished, got {}", describe(&other)),
        }

        let script = format!(
            "Start-Sleep -Seconds 6; Set-Content -LiteralPath '{}' -Value done",
            marker.display()
        );
        let lingers = h.send(vec![
            ActionSpec::VisualOnly,
            command(
                "powershell",
                &["-NoProfile", "-NonInteractive", "-Command", &script],
            ),
        ]);
        assert_eq!(
            h.completed(lingers, Duration::from_secs(5)).steps,
            vec![StepOutcome::Done, StepOutcome::Started]
        );
        let started = Instant::now();
        match h.next(Duration::from_secs(8)) {
            DispatchMsg::CommandFinished { key, step, outcome } => {
                assert_eq!((key, step), (lingers, 1));
                assert_eq!(outcome, CommandOutcome::StillRunning);
                let waited = started.elapsed();
                assert!(
                    waited >= Duration::from_millis(3800) && waited < Duration::from_secs(6),
                    "released after {waited:?}"
                );
            }
            other => panic!("expected CommandFinished, got {}", describe(&other)),
        }
        // Released, not killed: the child finishes its work after the watch ends.
        let deadline = Instant::now() + Duration::from_secs(20);
        while !marker.exists() {
            assert!(
                Instant::now() < deadline,
                "the released child did not keep running"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
        h.shutdown();
        std::fs::remove_dir_all(&marker_dir).expect("remove the marker dir");
    }

    #[test]
    fn a_seventeenth_child_releases_the_oldest_as_still_running() {
        let (tx, rx) = crossbeam_channel::bounded(64);
        let out = Outbox {
            tx,
            os: Arc::new(TestOs {
                log: Arc::default(),
                built: AtomicUsize::new(0),
                on_platform: None,
            }),
        };
        // Children that wait on their (never written) stdin, and exit once a released child's
        // pipe closes.
        let waiting = || {
            Command::new("cmd")
                .args(["/C", "set /p line="])
                .stdin(Stdio::piped())
                .creation_flags(CREATE_NO_WINDOW)
                .spawn()
                .expect("spawn cmd")
        };
        let mut watched = Watched::default();
        for step in 0..MAX_WATCHED_COMMANDS as u32 {
            watched.add(&out, JobKey(1), step, waiting());
        }
        assert!(rx.try_recv().is_err(), "nothing released at the limit");
        watched.add(&out, JobKey(2), 0, waiting());
        match rx.try_recv().expect("the oldest is released") {
            DispatchMsg::CommandFinished { key, step, outcome } => {
                assert_eq!((key, step), (JobKey(1), 0));
                assert_eq!(outcome, CommandOutcome::StillRunning);
            }
            other => panic!("expected CommandFinished, got {}", describe(&other)),
        }
        assert_eq!(watched.children.len(), MAX_WATCHED_COMMANDS);
        assert_eq!(watched.children[0].step, 1);
    }

    #[test]
    fn dp6_a_job_wakes_the_parked_thread() {
        let mut h = Harness::start(None);
        // No watched child, so the wait is INFINITE; no window message is posted.
        std::thread::sleep(Duration::from_millis(300));
        let key = h.send(vec![ActionSpec::VisualOnly]);
        let result = h.completed(key, Duration::from_secs(1));
        assert_eq!(result.steps, vec![StepOutcome::Done]);
        h.shutdown();
    }

    static DP7_DISPATCHED: AtomicU64 = AtomicU64::new(0);
    const DP7_MESSAGE: u32 = WM_APP + 7;

    extern "system" fn dp7_window_proc(
        hwnd: HWND,
        msg: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        if msg == DP7_MESSAGE {
            DP7_DISPATCHED.fetch_add(1, Ordering::SeqCst);
            return LRESULT(0);
        }
        // SAFETY: the arguments are the ones Windows passed to this window procedure.
        unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
    }

    /// A message-only window (never shown, never focused) owned by the calling thread.
    fn dp7_message_window() -> isize {
        // SAFETY: None asks for this executable's module handle.
        let module = unsafe { GetModuleHandleW(None) }.expect("module handle");
        let class = WNDCLASSW {
            lpfnWndProc: Some(dp7_window_proc),
            hInstance: module.into(),
            lpszClassName: w!("QwertyDispatchDp7"),
            ..Default::default()
        };
        // SAFETY: `class` is a valid WNDCLASSW whose strings are static.
        let atom = unsafe { RegisterClassW(&class) };
        assert_ne!(atom, 0, "RegisterClassW");
        // SAFETY: a message-only window of the class registered above.
        let hwnd = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE(0),
                w!("QwertyDispatchDp7"),
                w!(""),
                WINDOW_STYLE(0),
                0,
                0,
                0,
                0,
                Some(HWND_MESSAGE),
                None,
                Some(module.into()),
                None,
            )
        }
        .expect("CreateWindowExW");
        hwnd.0 as isize
    }

    #[test]
    fn dp7_the_thread_pumps_its_window_messages() {
        let (hwnd_tx, hwnd_rx) = crossbeam_channel::bounded(1);
        let h = Harness::start(Some(Box::new(move || {
            hwnd_tx
                .send(dp7_message_window())
                .expect("the test waits for the window");
        })));
        let hwnd = HWND(hwnd_rx.recv().expect("the window") as *mut std::ffi::c_void);
        std::thread::sleep(Duration::from_millis(100));
        let before = DP7_DISPATCHED.load(Ordering::SeqCst);
        // SAFETY: `hwnd` is the dispatch thread's live message-only window.
        unsafe { PostMessageW(Some(hwnd), DP7_MESSAGE, WPARAM(0), LPARAM(0)) }.expect("post");
        let deadline = Instant::now() + Duration::from_secs(1);
        while DP7_DISPATCHED.load(Ordering::SeqCst) == before {
            assert!(
                Instant::now() < deadline,
                "the message was not dispatched within 1 s"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(h.rx.try_recv().is_err(), "no job ran");
        h.shutdown();
    }

    /// DP8 (hardware evidence): over ≥ 200 parked-then-sent jobs, (`started_at` − send time) has
    /// p99 ≤ 5 ms. Run by the user: `cargo test -p qwerty-app --lib dp8_ -- --ignored --nocapture`.
    #[test]
    #[ignore = "hardware evidence: wake latency of the parked dispatch thread"]
    fn dp8_parked_wake_latency_p99_is_within_5_ms() {
        let mut h = Harness::start(None);
        let mut waits_ms = Vec::with_capacity(220);
        for _ in 0..220 {
            std::thread::sleep(Duration::from_millis(20));
            let sent = clock::now();
            let key = h.send(vec![ActionSpec::VisualOnly]);
            let result = h.completed(key, Duration::from_secs(1));
            waits_ms.push(result.started_at.since(sent).as_ms().get());
        }
        waits_ms.sort_by(f64::total_cmp);
        let at = |q: f64| waits_ms[((waits_ms.len() as f64 - 1.0) * q).round() as usize];
        println!(
            "DP8 over {} jobs: p50 {:.3} ms, p99 {:.3} ms, max {:.3} ms",
            waits_ms.len(),
            at(0.5),
            at(0.99),
            at(1.0)
        );
        assert!(at(0.99) <= 5.0, "p99 {:.3} ms", at(0.99));
        h.shutdown();
    }
}
