//! System tray icon and menu (`DESIGN.md` → Live status & feedback:
//! "Tray icon reflects state at a glance … its menu offers: pause/resume …
//! open main window, quit").
//!
//! The icon image swaps instantly on state change (no animation — `MOTION.md`
//! → Tray icon state change). Icons are generated in memory as colored discs,
//! so no image assets are committed to the repo (`CLAUDE.md`: no committed
//! binaries). Tray/menu events land on process-global channels; the shell
//! drains them each frame and a wake-thread requests a repaint on arrival.

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
}

impl Tray {
    /// Build the tray icon (reflecting `state`) and its menu.
    pub fn new(state: ListeningState) -> Result<Self, TrayError> {
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
        while let Ok(ev) = MenuEvent::receiver().try_recv() {
            if ev.id == self.toggle_id {
                cmds.push(TrayCommand::ToggleListening);
            } else if ev.id == self.open_id {
                cmds.push(TrayCommand::OpenWindow);
            } else if ev.id == self.quit_id {
                cmds.push(TrayCommand::Quit);
            }
        }
        while let Ok(ev) = TrayIconEvent::receiver().try_recv() {
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

/// A 32×32 RGBA icon: a filled disc in the status color on a transparent
/// background. Distinct color per state (`DESIGN.md`: listening/paused/error),
/// generated deterministically so no image file is needed.
fn status_icon(state: ListeningState) -> Icon {
    let (r, g, b) = match state {
        ListeningState::Listening => (0x35, 0xC4, 0x6B), // success green
        ListeningState::Paused => (0x9A, 0xA6, 0xB2),    // neutral gray
        ListeningState::Error => (0xF0, 0x6A, 0x6A),     // danger red
    };
    const SIZE: u32 = 32;
    let (cx, cy, radius) = (15.5_f32, 15.5_f32, 14.0_f32);
    let mut rgba = vec![0u8; (SIZE * SIZE * 4) as usize];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            if (dx * dx + dy * dy).sqrt() <= radius {
                let i = ((y * SIZE + x) * 4) as usize;
                rgba[i] = r;
                rgba[i + 1] = g;
                rgba[i + 2] = b;
                rgba[i + 3] = 255;
            }
        }
    }
    // A hardcoded 32×32 buffer is always a valid icon; a failure here would be
    // a programmer error, not a runtime condition.
    Icon::from_rgba(rgba, SIZE, SIZE).expect("32x32 RGBA is a valid tray icon")
}
