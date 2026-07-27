//! Global hotkey to toggle listening (`DESIGN.md` → Live status & feedback:
//! "a global hotkey … toggles listening on/off without touching the mouse").
//!
//! `global-hotkey` runs its own OS message loop on a background thread and
//! delivers hotkeys on a single process-global channel that hands each event to
//! exactly one consumer. A single event-pump thread (see `ui::shell`) is that
//! consumer: it forwards every event into the app-owned channel this type
//! drains, and calls `Context::request_repaint` so the UI reacts immediately
//! without polling on a timer. Reading the global channel here too would race
//! that pump thread and lose events.
//!
//! Phase 3 registers one fixed default combo. User-assignable rebinding (the
//! `Config::global_hotkey` field, plus a Settings editor) is a later phase —
//! this proves the plumbing toggles listening reliably, per `ROADMAP.md`.

use std::sync::mpsc::Receiver;

use global_hotkey::hotkey::{Code, HotKey, Modifiers};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};

/// Owns the registered global hotkey. Dropping it unregisters (RAII), so the
/// shell keeps this alive for the app's lifetime.
pub struct Hotkeys {
    _manager: GlobalHotKeyManager,
    toggle_id: u32,
    /// App-owned channel fed by the event-pump thread in `ui::shell`; the UI
    /// drains this rather than reading the process-global receiver directly.
    hotkey_rx: Receiver<GlobalHotKeyEvent>,
}

impl Hotkeys {
    /// Human-readable label for the default toggle combo, for the UI.
    pub const DEFAULT_LABEL: &'static str = "Ctrl+Alt+Space";

    /// Register the default listening-toggle hotkey (`Ctrl+Alt+Space`).
    pub fn new(hotkey_rx: Receiver<GlobalHotKeyEvent>) -> Result<Self, global_hotkey::Error> {
        let manager = GlobalHotKeyManager::new()?;
        let combo = HotKey::new(Some(Modifiers::CONTROL | Modifiers::ALT), Code::Space);
        let toggle_id = combo.id();
        manager.register(combo)?;
        Ok(Self {
            _manager: manager,
            toggle_id,
            hotkey_rx,
        })
    }

    /// Drain pending hotkey events; return `true` if the toggle combo was
    /// pressed at least once since the last call. Only the `Pressed` edge
    /// counts, so one physical press is one toggle (not press + release).
    pub fn poll_toggle(&self) -> bool {
        let mut toggled = false;
        while let Ok(ev) = self.hotkey_rx.try_recv() {
            if ev.id == self.toggle_id && ev.state == HotKeyState::Pressed {
                toggled = true;
            }
        }
        toggled
    }
}
