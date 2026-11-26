//! The one inbox and the event pump (shell-v2 §2.1-§2.2). Every producer (the event pump, the
//! request worker, the engine-start thread, activation, tray, hotkey, toasts, the settings
//! listener, the `WM_SIZE` subclass) pushes a [`ShellInput`] into the [`Inbox`] and wakes the UI
//! thread, which drains it in order. The inbox is unbounded because the engine already bounds
//! every producer (MB:287), so the pump never drops and never blocks on it (B:11).

use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use crossbeam_channel::Receiver;

use qwerty_core::profile::ProfileId;

use crate::engine::handle::{EngineHandle, EventStream};
use crate::engine::protocol::{EngineError, Event, Reply};
use crate::shell::model::requests::RequestId;
use crate::shell::model::route::Route;

/// Everything that reaches the UI thread, in arrival order (§2.2).
pub enum ShellInput {
    /// One engine event, read by the pump (`qwerty-shell-events`).
    Engine(Event),
    /// One request's reply, posted by the worker (`qwerty-shell-requests`).
    Reply {
        id: RequestId,
        result: Result<Reply, EngineError>,
    },
    /// `Engine::start`'s result, posted by `qwerty-shell-engine-start`.
    EngineStarted(Result<(EngineHandle, EventStream), EngineError>),
    /// A second instance asked this one to open its window (MB:31).
    Activate,
    /// A tray menu item (AB-13).
    Tray(TrayCommand),
    /// The global hotkey's `Pressed` edge.
    Hotkey,
    /// A status toast was activated; open on this route.
    ToastActivated(Route),
    /// The settings listener window saw a system change (WP-W2).
    SettingChanged(SystemChange),
    /// The window's own state (WP-W2's `WM_SIZE` subclass, the view's destroy path).
    WindowState(WindowState),
}

/// One variant per tray menu action (§9 AB-13; U:1113-1121, U:1271).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayCommand {
    /// "Pause listening" / "Resume listening" → `ToggleListening`.
    ToggleListening,
    /// A profile radio item → `SwitchProfile{profile}`.
    SwitchProfile { profile: ProfileId },
    /// "Recalibrate “{profile}”…" → open the window on the wizard route.
    Recalibrate { profile: ProfileId },
    /// "Open Qwerty" → open the window.
    Open,
    /// "Stop debug recording" → `UpdateConfig{debug_capture_enabled: false}`.
    StopDebugRecording,
    /// "Restart Qwerty" (fault only) → `Shutdown`, wait for `Stopped`, `Engine::start` (AB-5).
    Restart,
    /// "Quit Qwerty" → `Shutdown`.
    Quit,
}

/// What the settings listener window saw (WP-W2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemChange {
    /// `WM_SETTINGCHANGE`.
    Settings,
    /// `WM_SYSCOLORCHANGE`.
    SysColors,
}

/// The window's state as the model needs it: Rune counts a minimised window as shown (§2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowState {
    Shown,
    Minimised,
    Restored,
    Destroyed,
}

/// Wakes the UI thread's loop; supplied by the view (Rune: `AppHandle::wake`).
pub type Waker = Box<dyn Fn() + Send + Sync>;

/// `Mutex<Vec<ShellInput>>` plus the waker: `push` never blocks for longer than the lock and
/// never drops.
pub struct Inbox {
    inputs: Mutex<Vec<ShellInput>>,
    waker: Waker,
}

impl Inbox {
    pub fn new(waker: Waker) -> Inbox {
        Inbox {
            inputs: Mutex::new(Vec::new()),
            waker,
        }
    }

    /// Appends one input, then wakes the UI thread.
    pub fn push(&self, input: ShellInput) {
        self.lock().push(input);
        (self.waker)();
    }

    /// Appends several inputs in order, then wakes the UI thread once.
    pub fn push_all(&self, inputs: Vec<ShellInput>) {
        if inputs.is_empty() {
            return;
        }
        self.lock().extend(inputs);
        (self.waker)();
    }

    /// Everything pushed since the last drain, in arrival order.
    pub fn drain(&self) -> Vec<ShellInput> {
        std::mem::take(&mut *self.lock())
    }

    /// A producer that panicked while holding the lock left a complete `Vec` behind (a push is
    /// one `Vec::push` or `extend`), so the inputs are still valid and are kept.
    fn lock(&self) -> MutexGuard<'_, Vec<ShellInput>> {
        match self.inputs.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

/// The OS name of the pump thread (§2.1).
pub const EVENTS_THREAD: &str = "qwerty-shell-events";

/// The `qwerty-shell-events` thread: the only reader of the event stream (B:10).
pub struct EventPump {
    thread: JoinHandle<()>,
}

impl EventPump {
    /// Spawns the pump over `events` (`EventStream::receiver()`, MB:78). Each wake carries every
    /// event already waiting, so a burst costs one wake. When the stream disconnects without a
    /// `Stopped`, the pump posts one `Engine(Stopped)` so the shell sees the end (BR2), then ends.
    pub fn spawn(events: Receiver<Event>, inbox: Arc<Inbox>) -> std::io::Result<EventPump> {
        let thread = std::thread::Builder::new()
            .name(EVENTS_THREAD.to_string())
            .spawn(move || pump(&events, &inbox))?;
        Ok(EventPump { thread })
    }

    /// True once the pump has ended (the stream disconnected).
    pub fn is_finished(&self) -> bool {
        self.thread.is_finished()
    }

    /// Waits for the pump to end; it ends only when the stream disconnects.
    pub fn join(self) -> std::thread::Result<()> {
        self.thread.join()
    }
}

fn pump(events: &Receiver<Event>, inbox: &Inbox) {
    let mut stopped = false;
    while let Ok(first) = events.recv() {
        let mut batch = vec![first];
        batch.extend(events.try_iter());
        stopped = stopped || batch.iter().any(|event| matches!(event, Event::Stopped));
        inbox.push_all(batch.into_iter().map(ShellInput::Engine).collect());
    }
    // SPEC: BR2 asks for "one Engine(Stopped)-equivalent input" at a disconnect; it is the
    // engine's own `Stopped` event, added only when the engine did not already send one, so the
    // shell sees exactly one end of the stream.
    if !stopped {
        inbox.push(ShellInput::Engine(Event::Stopped));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use qwerty_core::profile::ZoneId;

    use crate::engine::protocol::Event;

    fn gesture(n: u32) -> Event {
        // The zone id carries the sequence number, so order is checkable.
        let text = format!("\"00000000-0000-0000-0000-{n:012x}\"");
        // INVARIANT: a 36-character hex uuid string always parses.
        let zone: ZoneId = serde_json::from_str(&text).expect("a valid uuid string");
        Event::GestureCleared { zone }
    }

    fn sequence(input: &ShellInput) -> u32 {
        let ShellInput::Engine(Event::GestureCleared { zone }) = input else {
            panic!("only the test's own events are pushed");
        };
        let text = zone.to_string();
        u32::from_str_radix(&text[24..], 16).expect("the test's own hex suffix")
    }

    #[test]
    fn br1_pump_keeps_every_event_in_order_without_blocking() {
        const EVENTS: u32 = 10_000;
        let wakes = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&wakes);
        let inbox = Arc::new(Inbox::new(Box::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
        })));
        let (tx, rx) = crossbeam_channel::bounded(512);
        let pump = EventPump::spawn(rx, Arc::clone(&inbox)).expect("spawns");

        // The producer fills a 512-slot channel 20 times over while the consumer below sleeps:
        // it finishes only because the pump moves events into the unbounded inbox at once.
        let producer = std::thread::spawn(move || {
            let start = Instant::now();
            for n in 0..EVENTS {
                tx.send(gesture(n)).expect("the pump is reading");
            }
            start.elapsed()
        });

        let mut received = Vec::with_capacity(EVENTS as usize);
        let mut wakes_seen = 0;
        let deadline = Instant::now() + Duration::from_secs(20);
        while received.len() < EVENTS as usize + 1 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(15)); // a slow consumer
            let batch = inbox.drain();
            if !batch.is_empty() {
                let now = wakes.load(Ordering::SeqCst);
                assert!(
                    now > wakes_seen,
                    "a non-empty batch arrived with no wake since the last"
                );
                wakes_seen = now;
            }
            received.extend(batch);
        }
        let sending = producer.join().expect("producer");
        assert!(
            sending < Duration::from_secs(5),
            "the producer was held up for {sending:?}"
        );
        pump.join().expect("the pump ends at the disconnect");
        received.extend(inbox.drain());

        // Every event, in order, then the one end-of-stream input.
        assert_eq!(received.len(), EVENTS as usize + 1);
        assert!(matches!(
            received.last(),
            Some(ShellInput::Engine(Event::Stopped))
        ));
        for (n, input) in received[..EVENTS as usize].iter().enumerate() {
            assert_eq!(sequence(input), n as u32, "order kept");
        }
    }

    #[test]
    fn br2_disconnect_ends_the_pump_with_one_stopped() {
        let inbox = Arc::new(Inbox::new(Box::new(|| {})));
        let (tx, rx) = crossbeam_channel::bounded(512);
        let pump = EventPump::spawn(rx, Arc::clone(&inbox)).expect("spawns");
        tx.send(gesture(1)).expect("the pump is reading");
        let start = Instant::now();
        drop(tx);
        while !pump.is_finished() && start.elapsed() < Duration::from_millis(500) {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(pump.is_finished(), "the pump ended within 500 ms");
        pump.join().expect("joins");
        assert!(start.elapsed() < Duration::from_millis(500));
        let inputs = inbox.drain();
        assert_eq!(inputs.len(), 2);
        assert_eq!(sequence(&inputs[0]), 1);
        assert!(matches!(inputs[1], ShellInput::Engine(Event::Stopped)));

        // An engine that sent its own Stopped gets no second one.
        let inbox = Arc::new(Inbox::new(Box::new(|| {})));
        let (tx, rx) = crossbeam_channel::bounded(512);
        let pump = EventPump::spawn(rx, Arc::clone(&inbox)).expect("spawns");
        tx.send(Event::Stopped).expect("the pump is reading");
        drop(tx);
        pump.join().expect("joins");
        let inputs = inbox.drain();
        assert_eq!(inputs.len(), 1);
        assert!(matches!(inputs[0], ShellInput::Engine(Event::Stopped)));
    }
}
