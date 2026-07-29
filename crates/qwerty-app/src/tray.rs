//! System tray icon and menu (`DESIGN.md` → Live status & feedback:
//! "Tray icon reflects state at a glance … its menu offers: pause/resume …
//! open main window, quit").
//!
//! The icon image swaps instantly on state change (no animation — `MOTION.md`
//! → Tray icon state change). Icons are generated in memory as colored discs,
//! so no image assets are committed to the repo (`CLAUDE.md`: no committed
//! binaries). Menu clicks and tray-icon clicks each arrive on their own
//! process-global channel (one per `tray-icon` event source), each handing a
//! message to exactly one consumer — so each has its own sole-reader event-pump
//! thread (see `ui::shell`) that forwards every event into the matching
//! app-owned channel this type drains, and wakes the UI. The tray must never
//! read those global channels directly, or it would race the pump threads and
//! lose events.

use std::sync::mpsc::Receiver;

use tray_icon::menu::{Menu, MenuEvent, MenuId, MenuItem};
use tray_icon::{Icon, MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent};

use crate::app_state::ListeningState;

/// A user intent decoded from a tray interaction, applied by the shell (the one
/// place state mutates — `ARCHITECTURE.md` → Threading model).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayCommand {
    ToggleListening,
    OpenWindow,
    Quit,
}

/// Errors building the tray icon or its menu. A tray failure is logged and the
/// GUI still runs (the tray is an affordance, not a precondition), so this is
/// surfaced but not fatal.
#[derive(Debug)]
pub enum TrayError {
    Menu(tray_icon::menu::Error),
    Build(tray_icon::Error),
}

impl std::fmt::Display for TrayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrayError::Menu(e) => write!(f, "could not build the tray menu: {e}"),
            TrayError::Build(e) => write!(f, "could not create the tray icon: {e}"),
        }
    }
}

impl std::error::Error for TrayError {}

/// Owns the live tray icon and the ids of its menu items (for matching events).
pub struct Tray {
    icon: TrayIcon,
    toggle_id: MenuId,
    open_id: MenuId,
    quit_id: MenuId,
    state: ListeningState,
    /// App-owned channels fed by the event-pump threads in `ui::shell`. The
    /// tray/menu crates deliver on a single process-global channel, so the UI
    /// drains these forwarded copies rather than reading that global directly.
    menu_rx: Receiver<MenuEvent>,
    tray_rx: Receiver<TrayIconEvent>,
}

impl Tray {
    /// Build the tray icon (reflecting `state`) and its menu. `menu_rx`/`tray_rx`
    /// are the app-owned channels the shell's event pump forwards menu and
    /// tray-click events into; [`poll`](Self::poll) drains them.
    pub fn new(
        state: ListeningState,
        menu_rx: Receiver<MenuEvent>,
        tray_rx: Receiver<TrayIconEvent>,
    ) -> Result<Self, TrayError> {
        let toggle = MenuItem::new("Toggle listening", true, None);
        let open = MenuItem::new("Open Qwerty", true, None);
        let quit = MenuItem::new("Quit", true, None);
        let (toggle_id, open_id, quit_id) =
            (toggle.id().clone(), open.id().clone(), quit.id().clone());

        let menu = Menu::new();
        menu.append(&toggle).map_err(TrayError::Menu)?;
        menu.append(&open).map_err(TrayError::Menu)?;
        menu.append(&quit).map_err(TrayError::Menu)?;

        let icon = TrayIconBuilder::new()
            .with_tooltip(tooltip(state))
            .with_menu(Box::new(menu))
            .with_icon(status_icon(state))
            .build()
            .map_err(TrayError::Build)?;

        Ok(Self {
            icon,
            toggle_id,
            open_id,
            quit_id,
            state,
            menu_rx,
            tray_rx,
        })
    }

    /// Swap the icon + tooltip to match `state`, only when it actually changed
    /// (avoids redundant OS calls every frame).
    pub fn set_state(&mut self, state: ListeningState) {
        if state == self.state {
            return;
        }
        self.state = state;
        let _ = self.icon.set_icon(Some(status_icon(state)));
        let _ = self.icon.set_tooltip(Some(tooltip(state)));
    }

    /// Drain pending menu + tray-click events into commands for the shell.
    /// A left click on the icon opens the window.
    pub fn poll(&self) -> Vec<TrayCommand> {
        let mut cmds = Vec::new();
        while let Ok(ev) = self.menu_rx.try_recv() {
            if ev.id == self.toggle_id {
                cmds.push(TrayCommand::ToggleListening);
            } else if ev.id == self.open_id {
                cmds.push(TrayCommand::OpenWindow);
            } else if ev.id == self.quit_id {
                cmds.push(TrayCommand::Quit);
            }
        }
        while let Ok(ev) = self.tray_rx.try_recv() {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = ev
            {
                cmds.push(TrayCommand::OpenWindow);
            }
        }
        cmds
    }
}

fn tooltip(state: ListeningState) -> String {
    format!("Qwerty — {}", state.label())
}

/// The tray icon: the keycap app mark with a state-colored status dot in the
/// corner (`DESIGN.md`: the tray reflects listening/paused/error at a glance).
/// The RGBA is produced by [`crate::ui::icon::tray_rgba`], which decodes the
/// bundled keycap once and composites the dot per state.
fn status_icon(state: ListeningState) -> Icon {
    let (rgba, w, h) = crate::ui::icon::tray_rgba(state);
    // A fixed-size, in-memory RGBA buffer is always a valid icon; a failure here
    // would be a programmer error, not a runtime condition.
    Icon::from_rgba(rgba, w, h).expect("keycap tray icon is valid RGBA")
}
