//! Global hotkey to toggle listening (`DESIGN.md` → Live status & feedback:
//! "a global hotkey … toggles listening on/off without touching the mouse").
//!
//! `global-hotkey` runs its own OS message loop on a background thread, so
//! registered hotkeys are delivered to a process-global channel independently
//! of eframe's event loop. The shell drains that channel each frame; a small
//! wake-thread (see `ui::shell`) calls `Context::request_repaint` when an event
//! arrives so the UI reacts immediately without polling on a timer.
//!
//! Phase 3 registers one fixed default combo. User-assignable rebinding (the
//! `Config::global_hotkey` field, plus a Settings editor) is a later phase —
//! this proves the plumbing toggles listening reliably, per `ROADMAP.md`.

use global_hotkey::hotkey::{Code, HotKey, Modifiers};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};

/// Owns the registered global hotkey. Dropping it unregisters (RAII), so the
/// shell keeps this alive for the app's lifetime.
pub struct Hotkeys {
    _manager: GlobalHotKeyManager,
    toggle_id: u32,
}

impl Hotkeys {
    /// Human-readable label for the default toggle combo, for the UI.
    pub const DEFAULT_LABEL: &'static str = "Ctrl+Alt+Space";

    /// Register the default listening-toggle hotkey (`Ctrl+Alt+Space`).
    pub fn new() -> Result<Self, global_hotkey::Error> {
        let manager = GlobalHotKeyManager::new()?;
        let combo = HotKey::new(Some(Modifiers::CONTROL | Modifiers::ALT), Code::Space);
        let toggle_id = combo.id();
        manager.register(combo)?;
        Ok(Self {
            _manager: manager,
            toggle_id,
        })
    }

    /// Drain pending hotkey events; return `true` if the toggle combo was
    /// pressed at least once since the last call. Only the `Pressed` edge
    /// counts, so one physical press is one toggle (not press + release).
    pub fn poll_toggle(&self) -> bool {
        let mut toggled = false;
        while let Ok(ev) = GlobalHotKeyEvent::receiver().try_recv() {
            if ev.id == self.toggle_id && ev.state == HotKeyState::Pressed {
                toggled = true;
            }
        }
        toggled
    }
}
