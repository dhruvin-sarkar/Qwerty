//! Raw-input veto thread (p2 §2.7): a message-only window registered for keyboard and mouse Raw
//! Input with `RIDEV_INPUTSINK`. Every key and button transition is stamped on the QPC clock and
//! pushed to processing's input ring; nothing else about the input leaves the handler.

use std::ffi::c_void;
use std::mem::{size_of, MaybeUninit};
use std::ptr::addr_of;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use crossbeam_channel::{Sender, TrySendError};
use qwerty_core::units::HostTime;
use qwerty_core::veto::{InputEvent, InputKind};
use windows::core::{w, PCWSTR};
use windows::Win32::Foundation::{GetLastError, HINSTANCE, HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{
    GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_HIGHEST,
};
use windows::Win32::UI::Input::{
    GetRawInputData, GetRegisteredRawInputDevices, RegisterRawInputDevices, HRAWINPUT, RAWINPUT,
    RAWINPUTDEVICE, RAWINPUTDEVICE_FLAGS, RAWINPUTHEADER, RIDEV_INPUTSINK, RIDEV_REMOVE, RID_INPUT,
    RIM_TYPEKEYBOARD, RIM_TYPEMOUSE,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, GetMessageW, KillTimer,
    PostMessageW, PostQuitMessage, RegisterClassExW, SetTimer, HWND_MESSAGE, MSG, RI_KEY_BREAK,
    RI_MOUSE_BUTTON_4_DOWN, RI_MOUSE_BUTTON_4_UP, RI_MOUSE_BUTTON_5_DOWN, RI_MOUSE_BUTTON_5_UP,
    RI_MOUSE_LEFT_BUTTON_DOWN, RI_MOUSE_LEFT_BUTTON_UP, RI_MOUSE_MIDDLE_BUTTON_DOWN,
    RI_MOUSE_MIDDLE_BUTTON_UP, RI_MOUSE_RIGHT_BUTTON_DOWN, RI_MOUSE_RIGHT_BUTTON_UP,
    WINDOW_EX_STYLE, WINDOW_STYLE, WM_CLOSE, WM_INPUT, WM_TIMER, WNDCLASSEXW,
};

use crate::engine::clock;
use crate::engine::messages::{
    try_send_counted, InputRecord, RawInputMsg, RawInputRequest, RawInputStatus, StopSignal,
    ThreadEnd, Worker, WorkerId,
};
use crate::engine::protocol::{InputDeviceClass, ThreadName, VetoUnavailableReason};

/// Each spawn gets a fresh ring of this many records (§2.7). 1024 key and button events cannot
/// arrive within processing's ≤ 50 ms drain interval; only a processing stall overflows it.
pub const INPUT_RING_RECORDS: usize = 1024;

/// HID usage page and usages the veto registers: generic desktop keyboard and mouse.
const USAGE_PAGE_GENERIC: u16 = 0x01;
const USAGE_KEYBOARD: u16 = 0x06;
const USAGE_MOUSE: u16 = 0x02;

/// The ownership re-check (step 3) and the pending-gap retry (step 4).
const OWNERSHIP_TIMER: usize = 1;
const OWNERSHIP_PERIOD_MS: u32 = 5_000;
const GAP_TIMER: usize = 2;
const GAP_RETRY_MS: u32 = 10;

const BUTTON_DOWN_FLAGS: u32 = RI_MOUSE_LEFT_BUTTON_DOWN
    | RI_MOUSE_RIGHT_BUTTON_DOWN
    | RI_MOUSE_MIDDLE_BUTTON_DOWN
    | RI_MOUSE_BUTTON_4_DOWN
    | RI_MOUSE_BUTTON_5_DOWN;
const BUTTON_UP_FLAGS: u32 = RI_MOUSE_LEFT_BUTTON_UP
    | RI_MOUSE_RIGHT_BUTTON_UP
    | RI_MOUSE_MIDDLE_BUTTON_UP
    | RI_MOUSE_BUTTON_4_UP
    | RI_MOUSE_BUTTON_5_UP;

/// The usages in registration and ownership-check order.
const USAGES: [(InputDeviceClass, u16); 2] = [
    (InputDeviceClass::Keyboard, USAGE_KEYBOARD),
    (InputDeviceClass::Mouse, USAGE_MOUSE),
];

// ---------------------------------------------------------------------------
// The ring-full logic (step 4), pure so RI4 tests it without input
// ---------------------------------------------------------------------------

/// What the caller does with the 10 ms gap-retry timer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimerAction {
    Arm10ms,
    Kill,
    None,
}

/// Coalesces every event dropped on a full ring into one pending `Gap{from, to}`, pushed ahead of
/// any later record as soon as the ring has room, so a gap never waits for the next input.
#[derive(Debug, Default)]
pub struct GapCoalescer {
    pending: Option<(HostTime, HostTime)>,
}

impl GapCoalescer {
    pub fn new() -> GapCoalescer {
        GapCoalescer { pending: None }
    }

    /// Pushes the pending gap if there is one. True when no gap is left pending.
    fn flush(&mut self, ring: &mut rtrb::Producer<InputRecord>) -> bool {
        match self.pending {
            None => true,
            Some((from, to)) => match ring.push(InputRecord::Gap { from, to }) {
                Ok(()) => {
                    self.pending = None;
                    true
                }
                Err(rtrb::PushError::Full(_)) => false,
            },
        }
    }

    /// Starts a pending gap at `now`, or extends the pending one to `now`.
    fn drop_at(&mut self, now: HostTime) {
        self.pending = Some(match self.pending {
            None => (now, now),
            Some((from, _)) => (from, now),
        });
    }

    pub fn on_input(
        &mut self,
        ring: &mut rtrb::Producer<InputRecord>,
        event: InputEvent,
        now: HostTime,
    ) -> TimerAction {
        let had_gap = self.pending.is_some();
        if !self.flush(ring) {
            // The gap is still waiting for room, so this event is lost inside it.
            self.drop_at(now);
            return TimerAction::Arm10ms;
        }
        match ring.push(InputRecord::Event(event)) {
            Ok(()) if had_gap => TimerAction::Kill,
            Ok(()) => TimerAction::None,
            Err(rtrb::PushError::Full(_)) => {
                self.drop_at(now);
                TimerAction::Arm10ms
            }
        }
    }

    /// `_now` is part of the §2.7 signature; the retry only moves the pending gap, whose span is
    /// fixed by the events it covers.
    pub fn on_timer(
        &mut self,
        ring: &mut rtrb::Producer<InputRecord>,
        _now: HostTime,
    ) -> TimerAction {
        if self.flush(ring) {
            TimerAction::Kill
        } else {
            TimerAction::Arm10ms
        }
    }
}

// ---------------------------------------------------------------------------
// Registration queries (shared by the ownership check, the stop cleanup and the tests)
// ---------------------------------------------------------------------------

/// The process's Raw Input targets for the two usages the veto registers; None = not registered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisteredTargets {
    pub keyboard: Option<HWND>,
    pub mouse: Option<HWND>,
}

impl RegisteredTargets {
    fn target(&self, class: InputDeviceClass) -> Option<HWND> {
        match class {
            InputDeviceClass::Keyboard => self.keyboard,
            InputDeviceClass::Mouse => self.mouse,
        }
    }

    /// The first usage (keyboard, then mouse) whose target is not `hwnd`.
    fn not_owned_by(&self, hwnd: HWND) -> Option<InputDeviceClass> {
        USAGES
            .iter()
            .map(|&(class, _)| class)
            .find(|&class| self.target(class) != Some(hwnd))
    }
}

/// `GetRegisteredRawInputDevices`, reduced to the keyboard and mouse targets. Err = the
/// GetLastError code.
pub fn registered_targets() -> Result<RegisteredTargets, u32> {
    let entry_size = size_of::<RAWINPUTDEVICE>() as u32;
    let mut count = 0u32;
    // SAFETY: a None buffer asks only for the count, written to the valid `count`.
    if unsafe { GetRegisteredRawInputDevices(None, &mut count, entry_size) } == u32::MAX {
        return Err(last_error());
    }
    let mut devices = vec![RAWINPUTDEVICE::default(); count as usize];
    let mut capacity = count;
    if count > 0 {
        // SAFETY: `devices` holds `capacity` writable entries of `entry_size` bytes each.
        let written = unsafe {
            GetRegisteredRawInputDevices(Some(devices.as_mut_ptr()), &mut capacity, entry_size)
        };
        if written == u32::MAX {
            return Err(last_error());
        }
        devices.truncate(written as usize);
    }
    let target = |usage: u16| {
        devices
            .iter()
            .find(|d| d.usUsagePage == USAGE_PAGE_GENERIC && d.usUsage == usage)
            .map(|d| d.hwndTarget)
    };
    Ok(RegisteredTargets {
        keyboard: target(USAGE_KEYBOARD),
        mouse: target(USAGE_MOUSE),
    })
}

fn last_error() -> u32 {
    // SAFETY: GetLastError only reads thread-local state.
    unsafe { GetLastError() }.0
}

fn device(usage: u16, flags: RAWINPUTDEVICE_FLAGS, target: HWND) -> RAWINPUTDEVICE {
    RAWINPUTDEVICE {
        usUsagePage: USAGE_PAGE_GENERIC,
        usUsage: usage,
        dwFlags: flags,
        hwndTarget: target,
    }
}

// ---------------------------------------------------------------------------
// Spawn and stop
// ---------------------------------------------------------------------------

/// Built before `Worker::spawn` with hwnd = 0 (the window does not exist yet).
struct RawInputStop {
    stop: Arc<AtomicBool>,
    hwnd: Arc<AtomicIsize>,
}

impl StopSignal for RawInputStop {
    /// stop = true, then post WM_CLOSE to the sink window if it exists. Never posts with a null
    /// HWND (that posts to the calling thread's own queue).
    fn signal(&self) {
        self.stop.store(true, Ordering::SeqCst);
        let h = self.hwnd.load(Ordering::SeqCst);
        if h != 0 {
            // SAFETY: posting a message never dereferences the window; a stale HWND only fails.
            let posted = unsafe {
                PostMessageW(Some(HWND(h as *mut c_void)), WM_CLOSE, WPARAM(0), LPARAM(0))
            };
            if let Err(e) = posted {
                // The window is already gone: the thread is ending on its own.
                tracing::debug!(error = %e, "the raw-input sink was already closed");
            }
        }
    }
}

pub fn spawn(request: RawInputRequest) -> Result<Worker, VetoUnavailableReason> {
    let RawInputRequest {
        worker,
        ring,
        tx,
        exits,
        dropped,
    } = request;
    let stop = Arc::new(AtomicBool::new(false));
    let hwnd = Arc::new(AtomicIsize::new(0));
    let signal = RawInputStop {
        stop: Arc::clone(&stop),
        hwnd: Arc::clone(&hwnd),
    };
    Worker::spawn(
        worker,
        ThreadName::RawInput,
        Box::new(signal),
        exits,
        move || {
            run(worker, ring, tx, dropped, &stop, &hwnd);
            ThreadEnd::Finished
        },
    )
    // Worker::spawn fails only with EngineError::ThreadSpawn; its Debug form names the thread and
    // the OS detail.
    .map_err(|e| VetoUnavailableReason::ThreadFailed {
        message: format!("{e:?}"),
    })
}

// ---------------------------------------------------------------------------
// The thread
// ---------------------------------------------------------------------------

/// Status messages to the controller. The last delivered status and skip count are remembered,
/// so the 5 s timer sends only on a change; a send that finds the channel full is retried then.
struct Reporter {
    worker: WorkerId,
    tx: Sender<RawInputMsg>,
    /// The controller's `controller_dropped` feed (§2.11): each refused send counts, resent or
    /// not.
    dropped: Arc<AtomicU64>,
    delivered: Option<(RawInputStatus, u64)>,
}

impl Reporter {
    fn report(&mut self, status: RawInputStatus, skipped_records: u64) {
        let current = (status, skipped_records);
        if self.delivered.as_ref() == Some(&current) {
            return;
        }
        let msg = RawInputMsg {
            worker: self.worker,
            status: current.0.clone(),
            skipped_records,
        };
        match try_send_counted(&self.tx, msg, &self.dropped) {
            Ok(()) => self.delivered = Some(current),
            Err(TrySendError::Full(_)) => {
                tracing::warn!(
                    "the raw-input channel was full; the status is resent on the next check"
                );
            }
            // The controller is gone; nobody is left to tell.
            Err(TrySendError::Disconnected(_)) => {}
        }
    }

    fn unavailable(&mut self, reason: VetoUnavailableReason, skipped_records: u64) {
        self.report(RawInputStatus::Unavailable { reason }, skipped_records);
    }
}

/// The window procedure: every message the loop does not handle itself goes to DefWindowProcW,
/// including WM_INPUT after it is read (the system frees the raw input buffer there).
unsafe extern "system" fn sink_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: forwarded unchanged from the system's call.
    unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
}

/// The class atom, registered by the first spawn and never unregistered (a detached worker's
/// window can outlive its stop, and unregistering would fail with ERROR_CLASS_HAS_WINDOWS).
static SINK_CLASS: OnceLock<Result<u16, u32>> = OnceLock::new();

fn module() -> Result<HINSTANCE, u32> {
    // SAFETY: None asks for the executable's own module handle, which needs no release.
    unsafe { GetModuleHandleW(PCWSTR::null()) }
        .map(HINSTANCE::from)
        .map_err(|_| last_error())
}

fn sink_class() -> Result<u16, u32> {
    *SINK_CLASS.get_or_init(|| {
        let instance = module()?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(sink_proc),
            hInstance: instance,
            lpszClassName: w!("QwertyRawInputSink"),
            ..Default::default()
        };
        // SAFETY: `class` is a fully initialised WNDCLASSEXW whose strings are static.
        match unsafe { RegisterClassExW(&class) } {
            0 => Err(last_error()),
            atom => Ok(atom),
        }
    })
}

fn create_sink() -> Result<HWND, String> {
    let atom = sink_class().map_err(|code| format!("RegisterClassExW failed (error {code})"))?;
    let instance = module().map_err(|code| format!("GetModuleHandleW failed (error {code})"))?;
    // SAFETY: the class name is the registered atom (MAKEINTATOM); the window is message-only and
    // owned by this thread, which destroys it in `close`.
    unsafe {
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            PCWSTR(usize::from(atom) as *const u16),
            w!("Qwerty raw input"),
            WINDOW_STYLE(0),
            0,
            0,
            0,
            0,
            Some(HWND_MESSAGE),
            None,
            Some(instance),
            None,
        )
    }
    .map_err(|_| format!("CreateWindowExW failed (error {})", last_error()))
}

/// Step 5: remove each usage that still targets `hwnd` (RIDEV_REMOVE is process-wide per usage,
/// so a registration another window now owns is left alone), then destroy the window.
fn close(hwnd: HWND, hwnd_slot: &AtomicIsize) {
    match registered_targets() {
        Ok(targets) => {
            for (class, usage) in USAGES {
                if targets.target(class) == Some(hwnd) {
                    let remove = [device(usage, RIDEV_REMOVE, HWND::default())];
                    // SAFETY: one valid RAWINPUTDEVICE with the size the API expects.
                    let removed = unsafe {
                        RegisterRawInputDevices(&remove, size_of::<RAWINPUTDEVICE>() as u32)
                    };
                    if let Err(e) = removed {
                        tracing::error!(?class, error = %e, "could not remove the Raw Input registration");
                    }
                }
            }
        }
        Err(code) => {
            tracing::error!(
                code,
                "GetRegisteredRawInputDevices failed; the registrations were not removed"
            );
        }
    }
    // No signal() posts to this window once it is being destroyed.
    hwnd_slot.store(0, Ordering::SeqCst);
    // SAFETY: `hwnd` is this thread's own live window.
    if let Err(e) = unsafe { DestroyWindow(hwnd) } {
        tracing::error!(error = %e, "could not destroy the raw-input sink window");
    }
}

/// Register both usages, each separately, so a failure names its class.
fn register(hwnd: HWND) -> Result<(), VetoUnavailableReason> {
    for (class, usage) in USAGES {
        let add = [device(usage, RIDEV_INPUTSINK, hwnd)];
        // SAFETY: one valid RAWINPUTDEVICE with the size the API expects.
        if unsafe { RegisterRawInputDevices(&add, size_of::<RAWINPUTDEVICE>() as u32) }.is_err() {
            return Err(VetoUnavailableReason::RegistrationFailed {
                class,
                code: last_error(),
            });
        }
    }
    Ok(())
}

fn run(
    worker: WorkerId,
    ring: rtrb::Producer<InputRecord>,
    tx: Sender<RawInputMsg>,
    dropped: Arc<AtomicU64>,
    stop: &AtomicBool,
    hwnd_slot: &AtomicIsize,
) {
    let mut reporter = Reporter {
        worker,
        tx,
        dropped,
        delivered: None,
    };

    // Step 1: input stamps must not queue behind normal-priority work.
    // SAFETY: GetCurrentThread is a pseudo handle valid on this thread.
    if let Err(e) = unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST) } {
        reporter.unavailable(
            VetoUnavailableReason::ThreadFailed {
                message: format!("SetThreadPriority failed: {e}"),
            },
            0,
        );
        return;
    }

    // Step 2.
    let hwnd = match create_sink() {
        Ok(hwnd) => hwnd,
        Err(message) => {
            reporter.unavailable(VetoUnavailableReason::ThreadFailed { message }, 0);
            return;
        }
    };
    // Step 2a: either signal() sees the hwnd and posts WM_CLOSE, or this sees the flag. Nothing
    // runs between the window's creation and the registration, so this one check is both the
    // post-CreateWindowExW check and the pre-RegisterRawInputDevices one.
    hwnd_slot.store(hwnd.0 as isize, Ordering::SeqCst);
    if stop.load(Ordering::SeqCst) {
        close(hwnd, hwnd_slot);
        return;
    }
    if let Err(reason) = register(hwnd) {
        reporter.unavailable(reason, 0);
        close(hwnd, hwnd_slot);
        return;
    }

    // Step 3.
    match registered_targets() {
        Ok(targets) => {
            if let Some(class) = targets.not_owned_by(hwnd) {
                reporter.unavailable(VetoUnavailableReason::RegistrationNotOwned { class }, 0);
                close(hwnd, hwnd_slot);
                return;
            }
        }
        Err(code) => {
            reporter.unavailable(
                VetoUnavailableReason::ThreadFailed {
                    message: format!("GetRegisteredRawInputDevices failed (error {code})"),
                },
                0,
            );
            close(hwnd, hwnd_slot);
            return;
        }
    }
    // SAFETY: `hwnd` is this thread's own window; no TIMERPROC, so WM_TIMER is queued.
    if unsafe { SetTimer(Some(hwnd), OWNERSHIP_TIMER, OWNERSHIP_PERIOD_MS, None) } == 0 {
        reporter.unavailable(
            VetoUnavailableReason::ThreadFailed {
                message: format!("SetTimer failed (error {})", last_error()),
            },
            0,
        );
        close(hwnd, hwnd_slot);
        return;
    }
    if stop.load(Ordering::SeqCst) {
        close(hwnd, hwnd_slot);
        return;
    }
    reporter.report(RawInputStatus::Active, 0);

    let mut sink = Sink {
        hwnd,
        ring,
        coalescer: GapCoalescer::new(),
        skipped_records: 0,
        reporter,
    };
    sink.pump(hwnd_slot);
}

/// The running sink: everything the message loop's handlers touch.
struct Sink {
    hwnd: HWND,
    ring: rtrb::Producer<InputRecord>,
    coalescer: GapCoalescer,
    skipped_records: u64,
    reporter: Reporter,
}

impl Sink {
    fn pump(&mut self, hwnd_slot: &AtomicIsize) {
        let mut open = true;
        let mut msg = MSG::default();
        loop {
            // SAFETY: `msg` is a valid, writable MSG.
            let got = unsafe { GetMessageW(&mut msg, None, 0, 0) };
            match got.0 {
                0 => return, // WM_QUIT
                -1 => {
                    self.reporter.unavailable(
                        VetoUnavailableReason::ThreadFailed {
                            message: format!("GetMessageW failed (error {})", last_error()),
                        },
                        self.skipped_records,
                    );
                    if open {
                        close(self.hwnd, hwnd_slot);
                    }
                    return;
                }
                _ => {}
            }
            if !open || msg.hwnd != self.hwnd {
                // SAFETY: `msg` was just filled by GetMessageW.
                unsafe { DispatchMessageW(&msg) };
                continue;
            }
            match msg.message {
                WM_INPUT => {
                    self.on_input(HRAWINPUT(msg.lParam.0 as *mut c_void));
                    // SAFETY: as above; DefWindowProcW frees the raw input buffer.
                    unsafe { DispatchMessageW(&msg) };
                }
                WM_TIMER => self.on_timer(msg.wParam.0),
                WM_CLOSE => {
                    open = false;
                    close(self.hwnd, hwnd_slot);
                    // SAFETY: ends this thread's GetMessageW loop.
                    unsafe { PostQuitMessage(0) };
                }
                _ => {
                    // SAFETY: `msg` was just filled by GetMessageW.
                    unsafe { DispatchMessageW(&msg) };
                }
            }
        }
    }

    /// Step 4: stamp first, then read the record into a correctly aligned RAWINPUT. Only
    /// (now, kind) leaves this function; the RAWINPUT bytes are never stored or logged.
    fn on_input(&mut self, handle: HRAWINPUT) {
        let now = clock::now();
        let mut raw = MaybeUninit::<RAWINPUT>::uninit();
        let mut size = size_of::<RAWINPUT>() as u32;
        // SAFETY: `raw` is an aligned, writable buffer of `size` bytes, and `handle` is the
        // HRAWINPUT of the WM_INPUT being handled, before DefWindowProcW frees it.
        let read = unsafe {
            GetRawInputData(
                handle,
                RID_INPUT,
                Some(raw.as_mut_ptr().cast()),
                &mut size,
                size_of::<RAWINPUTHEADER>() as u32,
            )
        };
        if read == u32::MAX || (read as usize) < size_of::<RAWINPUTHEADER>() {
            // A record larger than RAWINPUT (keyboard and mouse records always fit, so this is a
            // fault the controller reports).
            self.skipped_records += 1;
            return;
        }
        let p = raw.as_ptr();
        // SAFETY: the call succeeded and wrote at least the header.
        let kind = unsafe { addr_of!((*p).header.dwType).read() };
        let mut events = [None, None];
        if kind == RIM_TYPEKEYBOARD.0 {
            // SAFETY: a keyboard record holds a RAWKEYBOARD after the header.
            let flags = u32::from(unsafe { addr_of!((*p).data.keyboard.Flags).read() });
            events[0] = Some(if flags & RI_KEY_BREAK != 0 {
                InputKind::KeyUp
            } else {
                InputKind::KeyDown
            });
        } else if kind == RIM_TYPEMOUSE.0 {
            // SAFETY: a mouse record holds a RAWMOUSE after the header.
            let buttons = u32::from(unsafe {
                addr_of!((*p).data.mouse.Anonymous.Anonymous.usButtonFlags).read()
            });
            // One record can carry a press and a release together; each gives its event. Moves
            // and the wheel carry neither and are ignored.
            if buttons & BUTTON_DOWN_FLAGS != 0 {
                events[0] = Some(InputKind::MouseButtonDown);
            }
            if buttons & BUTTON_UP_FLAGS != 0 {
                events[1] = Some(InputKind::MouseButtonUp);
            }
        }
        for kind in events.into_iter().flatten() {
            let action =
                self.coalescer
                    .on_input(&mut self.ring, InputEvent { time: now, kind }, now);
            self.apply(action);
        }
    }

    fn on_timer(&mut self, id: usize) {
        match id {
            GAP_TIMER => {
                let action = self.coalescer.on_timer(&mut self.ring, clock::now());
                self.apply(action);
            }
            OWNERSHIP_TIMER => {
                let status = match registered_targets() {
                    Ok(targets) => match targets.not_owned_by(self.hwnd) {
                        None => RawInputStatus::Active,
                        Some(class) => RawInputStatus::Unavailable {
                            reason: VetoUnavailableReason::OwnershipLost { class },
                        },
                    },
                    Err(code) => RawInputStatus::Unavailable {
                        reason: VetoUnavailableReason::ThreadFailed {
                            message: format!("GetRegisteredRawInputDevices failed (error {code})"),
                        },
                    },
                };
                // The sink stays up either way; only a change is sent.
                self.reporter.report(status, self.skipped_records);
            }
            other => {
                tracing::error!(timer = other, "the raw-input sink got an unknown timer");
            }
        }
    }

    fn apply(&mut self, action: TimerAction) {
        match action {
            TimerAction::Arm10ms => {
                // SAFETY: `self.hwnd` is this thread's own window.
                if unsafe { SetTimer(Some(self.hwnd), GAP_TIMER, GAP_RETRY_MS, None) } == 0 {
                    tracing::error!(code = last_error(), "could not arm the input-gap timer");
                }
            }
            TimerAction::Kill => {
                // SAFETY: as above. Killing a timer that is not set only fails.
                if let Err(e) = unsafe { KillTimer(Some(self.hwnd), GAP_TIMER) } {
                    tracing::debug!(error = %e, "the input-gap timer was not running");
                }
            }
            TimerAction::None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(ticks: i64) -> InputEvent {
        InputEvent {
            time: HostTime(ticks),
            kind: InputKind::KeyDown,
        }
    }

    #[test]
    fn ri4_full_ring_coalesces_drops_into_one_gap() {
        let (mut ring, mut out) = rtrb::RingBuffer::<InputRecord>::new(4);
        let mut gaps = GapCoalescer::new();

        for t in 1..=4 {
            assert_eq!(
                gaps.on_input(&mut ring, key(t), HostTime(t)),
                TimerAction::None
            );
        }
        for t in 5..=7 {
            assert_eq!(
                gaps.on_input(&mut ring, key(t), HostTime(t)),
                TimerAction::Arm10ms
            );
        }

        assert_eq!(out.pop(), Ok(InputRecord::Event(key(1))));
        assert_eq!(gaps.on_timer(&mut ring, HostTime(8)), TimerAction::Kill);

        let rest: Vec<InputRecord> = std::iter::from_fn(|| out.pop().ok()).collect();
        assert_eq!(
            rest,
            vec![
                InputRecord::Event(key(2)),
                InputRecord::Event(key(3)),
                InputRecord::Event(key(4)),
                InputRecord::Gap {
                    from: HostTime(5),
                    to: HostTime(7),
                },
            ]
        );
        // Nothing is left pending: the next timer has nothing to push.
        assert_eq!(gaps.on_timer(&mut ring, HostTime(9)), TimerAction::Kill);
        assert!(out.pop().is_err());
    }
}
