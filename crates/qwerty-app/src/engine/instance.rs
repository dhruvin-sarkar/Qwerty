//! Single instance (p2 §2.14): a named mutex decides which process is the Primary, and a named
//! auto-reset event lets a second launch wake the Primary ("open your window") before it exits.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::thread::JoinHandle;

use serde::{Deserialize, Serialize};
use windows::core::{HSTRING, PCWSTR};
use windows::Win32::Foundation::{
    CloseHandle, DuplicateHandle, GetLastError, SetLastError, DUPLICATE_SAME_ACCESS,
    ERROR_ALREADY_EXISTS, HANDLE, WAIT_OBJECT_0, WIN32_ERROR,
};
use windows::Win32::System::Threading::{
    CreateEventW, CreateMutexW, GetCurrentProcess, SetEvent, WaitForMultipleObjects, INFINITE,
};

/// The product's object-name prefix (`acquire`). Tests pass their own unique prefix.
const PRODUCT_PREFIX: &str = "Local\\Qwerty.TapZones";

pub enum InstanceGuard {
    Primary(PrimaryInstance),
    AlreadyRunning,
}

/// Owns the mutex and the activation event; `Drop` closes both handles.
pub struct PrimaryInstance {
    mutex: HANDLE,
    activate: HANDLE,
}

// SAFETY: kernel object handles are process-wide values, valid on any thread; PrimaryInstance only
// closes them in Drop and duplicates `activate`, both of which are thread-agnostic.
unsafe impl Send for PrimaryInstance {}
// SAFETY: as above; `&PrimaryInstance` only reads the handle values.
unsafe impl Sync for PrimaryInstance {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InstanceError {
    CreateMutex { code: u32 },
    CreateEvent { code: u32 },
    // SPEC: §2.14 gives `listen` no error path, but DuplicateHandle and the thread spawn can fail
    // (handle or thread exhaustion); the project forbids an unproven expect, so `listen` returns
    // these instead of panicking.
    DuplicateHandle { code: u32 },
    ListenerSpawn { detail: String },
}

impl std::fmt::Display for InstanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstanceError::CreateMutex { code } => {
                write!(
                    f,
                    "could not create the single-instance mutex (error {code})"
                )
            }
            InstanceError::CreateEvent { code } => {
                write!(f, "could not create the activation event (error {code})")
            }
            InstanceError::DuplicateHandle { code } => {
                write!(f, "could not duplicate the activation event (error {code})")
            }
            InstanceError::ListenerSpawn { detail } => {
                write!(f, "could not start the activation listener: {detail}")
            }
        }
    }
}

impl std::error::Error for InstanceError {}

/// The calling thread's last Win32 error. The `windows` wrappers build their `Err` from
/// GetLastError without changing it, so this is the failed call's code.
fn last_error() -> u32 {
    // SAFETY: GetLastError only reads thread-local state.
    unsafe { GetLastError() }.0
}

/// `CreateEventW(None, false /*auto-reset*/, false, "<prefix>.Activate")`: opens the event or
/// creates it.
fn activation_event(prefix: &str) -> Result<HANDLE, InstanceError> {
    let name = HSTRING::from(format!("{prefix}.Activate"));
    // SAFETY: `name` is a valid NUL-terminated wide string that outlives the call.
    unsafe { CreateEventW(None, false, false, &name) }
        .map_err(|_| InstanceError::CreateEvent { code: last_error() })
}

fn close(handle: HANDLE, what: &str) {
    // SAFETY: `handle` is a handle this module opened and has not closed yet.
    if let Err(e) = unsafe { CloseHandle(handle) } {
        tracing::error!(handle = what, error = %e, "CloseHandle failed");
    }
}

/// `acquire_named("Local\\Qwerty.TapZones")`.
pub fn acquire() -> Result<InstanceGuard, InstanceError> {
    acquire_named(PRODUCT_PREFIX)
}

/// `CreateMutexW(None, true, "<prefix>.Instance")`. If the mutex already existed, signal the
/// activation event, close every handle and return AlreadyRunning (no handle survives this path,
/// so the named mutex dies with its Primary). Otherwise Primary, with the activation event
/// created the same way.
pub fn acquire_named(prefix: &str) -> Result<InstanceGuard, InstanceError> {
    let name = HSTRING::from(format!("{prefix}.Instance"));
    // SAFETY: SetLastError only writes thread-local state. Clearing it first means the
    // ERROR_ALREADY_EXISTS read below can only come from this CreateMutexW.
    unsafe { SetLastError(WIN32_ERROR(0)) };
    // SAFETY: `name` is a valid NUL-terminated wide string that outlives the call.
    let mutex = unsafe { CreateMutexW(None, true, &name) }
        .map_err(|_| InstanceError::CreateMutex { code: last_error() })?;
    // The handle is valid either way; ERROR_ALREADY_EXISTS means another process (or an earlier
    // acquire in this one) is the Primary.
    let existed = last_error() == ERROR_ALREADY_EXISTS.0;

    let activate = match activation_event(prefix) {
        Ok(event) => event,
        Err(e) => {
            close(mutex, "instance mutex");
            return Err(e);
        }
    };

    if existed {
        // SAFETY: `activate` is the valid event handle just opened.
        if let Err(e) = unsafe { SetEvent(activate) } {
            tracing::error!(error = %e, "could not signal the running instance");
        }
        close(activate, "activation event");
        close(mutex, "instance mutex");
        return Ok(InstanceGuard::AlreadyRunning);
    }
    Ok(InstanceGuard::Primary(PrimaryInstance { mutex, activate }))
}

/// A handle value handed to the listener thread. The `ActivationListener` owns and closes it,
/// only after the thread has been joined.
#[derive(Clone, Copy)]
struct ThreadHandle(HANDLE);

// SAFETY: a kernel handle value is valid on any thread of the process; the listener keeps it open
// until the thread that uses it has been joined.
unsafe impl Send for ThreadHandle {}

impl PrimaryInstance {
    /// Spawns the `qwerty-activate` thread (a plain named thread: it lives outside the engine, so
    /// it is not a `Worker`). It waits on a DuplicateHandle copy of the activation event and its
    /// own unnamed manual-reset stop event: index 0 calls `on_activate()` (under catch_unwind; a
    /// panic is logged and ends the thread), index 1 returns. Owning a duplicate means the wait
    /// never uses a closed handle, whichever of the listener and the PrimaryInstance drops first.
    pub fn listen(
        &self,
        on_activate: Box<dyn Fn() + Send>,
    ) -> Result<ActivationListener, InstanceError> {
        let mut duplicate = HANDLE::default();
        // SAFETY: both process handles are the current-process pseudo handle, `self.activate` is
        // a live handle owned by `self`, and `duplicate` is a valid out pointer.
        unsafe {
            DuplicateHandle(
                GetCurrentProcess(),
                self.activate,
                GetCurrentProcess(),
                &mut duplicate,
                0,
                false,
                DUPLICATE_SAME_ACCESS,
            )
        }
        .map_err(|_| InstanceError::DuplicateHandle { code: last_error() })?;

        // SAFETY: no name and no attributes; the returned handle is owned below.
        let stop = match unsafe { CreateEventW(None, true, false, PCWSTR::null()) } {
            Ok(stop) => stop,
            Err(_) => {
                let code = last_error();
                close(duplicate, "activation event duplicate");
                return Err(InstanceError::CreateEvent { code });
            }
        };

        let handles = [ThreadHandle(duplicate), ThreadHandle(stop)];
        let spawned = std::thread::Builder::new()
            .name("qwerty-activate".to_string())
            .spawn(move || activation_loop(handles, on_activate));
        match spawned {
            Ok(thread) => Ok(ActivationListener {
                stop,
                activate: duplicate,
                thread: Some(thread),
            }),
            Err(e) => {
                close(stop, "activation stop event");
                close(duplicate, "activation event duplicate");
                Err(InstanceError::ListenerSpawn {
                    detail: e.to_string(),
                })
            }
        }
    }
}

fn activation_loop(handles: [ThreadHandle; 2], on_activate: Box<dyn Fn() + Send>) {
    let waits = [handles[0].0, handles[1].0];
    loop {
        // SAFETY: both handles stay open until the ActivationListener has joined this thread.
        let woke = unsafe { WaitForMultipleObjects(&waits, false, INFINITE) };
        if woke == WAIT_OBJECT_0 {
            if catch_unwind(AssertUnwindSafe(&on_activate)).is_err() {
                tracing::error!("the activation callback panicked; the listener has ended");
                return;
            }
        } else if woke.0 == WAIT_OBJECT_0.0 + 1 {
            return;
        } else {
            tracing::error!(
                result = woke.0,
                error = last_error(),
                "waiting for activation failed; the listener has ended"
            );
            return;
        }
    }
}

impl Drop for PrimaryInstance {
    fn drop(&mut self) {
        close(self.activate, "activation event");
        close(self.mutex, "instance mutex");
    }
}

/// Drop: SetEvent(stop), join (the thread wakes at once), then CloseHandle of its duplicate and
/// of the stop event.
pub struct ActivationListener {
    stop: HANDLE,
    activate: HANDLE,
    thread: Option<JoinHandle<()>>,
}

impl Drop for ActivationListener {
    fn drop(&mut self) {
        // SAFETY: `self.stop` is the live stop event this listener owns.
        if let Err(e) = unsafe { SetEvent(self.stop) } {
            // The thread cannot be woken, so it is detached and its handles stay open: closing
            // them under its live wait would let the wait use a recycled handle value.
            tracing::error!(error = %e, "could not stop the activation listener; it is detached");
            return;
        }
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                tracing::error!("the activation listener panicked outside its callback");
            }
        }
        close(self.activate, "activation event duplicate");
        close(self.stop, "activation stop event");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    fn unique_prefix() -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("the clock is after 1970")
            .as_nanos();
        format!("Local\\Qwerty.Test.{}.{nanos}", std::process::id())
    }

    #[test]
    fn in1_second_instance_wakes_the_primary_once() {
        let prefix = unique_prefix();
        let InstanceGuard::Primary(primary) = acquire_named(&prefix).expect("first acquire") else {
            panic!("the first acquire with a fresh prefix must be Primary");
        };

        let woken = Arc::new(AtomicU32::new(0));
        let (tx, rx) = crossbeam_channel::bounded::<()>(4);
        let count = Arc::clone(&woken);
        let listener = primary
            .listen(Box::new(move || {
                count.fetch_add(1, Ordering::SeqCst);
                tx.try_send(()).expect("the test channel has room");
            }))
            .expect("listen");

        let InstanceGuard::AlreadyRunning = acquire_named(&prefix).expect("second acquire") else {
            panic!("a second acquire while the Primary lives must be AlreadyRunning");
        };
        rx.recv_timeout(Duration::from_secs(1))
            .expect("the listener woke within 1 s");
        // Exactly once: no second wake arrives.
        assert!(rx.recv_timeout(Duration::from_millis(300)).is_err());
        assert_eq!(woken.load(Ordering::SeqCst), 1);

        let started = Instant::now();
        drop(listener);
        let joined = started.elapsed();
        assert!(
            joined < Duration::from_millis(500),
            "dropping the listener took {joined:?}"
        );

        drop(primary);
        let InstanceGuard::Primary(third) = acquire_named(&prefix).expect("third acquire") else {
            panic!("after the Primary is dropped, a new acquire must be Primary");
        };
        drop(third);
    }
}
