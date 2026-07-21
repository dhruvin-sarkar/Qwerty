//! Shared UI-thread state (`ARCHITECTURE.md` → Threading model).
//!
//! There is exactly one place app state is mutated: the UI thread. This struct
//! is that state. In Phase 3 it holds the active screen, the (still manually
//! toggled) listening state, the loaded [`Config`], and the theme cross-fade
//! animation. Later phases add the processing-thread `ZoneEvent` receiver and
//! the `PlatformActions` handle here — this is the single owner they attach to.
//!
//! State transitions here are pure (no disk I/O): persistence is the shell's
//! concern, so these stay unit-testable without touching the real user config.

use std::path::PathBuf;
use std::time::Instant;

use qwerty_core::profile::{Config, Profile, Theme};

/// The main-navigation destinations (`DESIGN.md` → Layout: a persistent
/// left-hand rail, every area one click away). The calibration wizard is
/// deliberately *not* here — it takes over the whole window (Phase 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Home,
    Zones,
    Diagnostics,
    Evaluation,
    Settings,
}

impl Screen {
    /// Rail order, top to bottom (`DESIGN.md` → Layout).
    pub const ALL: [Screen; 5] = [
        Screen::Home,
        Screen::Zones,
        Screen::Diagnostics,
        Screen::Evaluation,
        Screen::Settings,
    ];

    /// The rail label shown to the user.
    pub fn label(self) -> &'static str {
        match self {
            Screen::Home => "Home",
            Screen::Zones => "Zones & profiles",
            Screen::Diagnostics => "Diagnostics",
            Screen::Evaluation => "Evaluation",
            Screen::Settings => "Settings",
        }
    }

    /// A leading glyph for the rail. Text-based so it renders in the default
    /// system font without bundling an icon font, and always paired with the
    /// text label (no information by glyph alone, per `DESIGN.md`).
    pub fn glyph(self) -> &'static str {
        match self {
            Screen::Home => "⌂",
            Screen::Zones => "▦",
            Screen::Diagnostics => "∿",
            Screen::Evaluation => "✓",
            Screen::Settings => "⚙",
        }
    }
}

/// Whether the app is currently acting on taps. In Phase 3 this is toggled
/// manually (a Home button and, next, the global hotkey); the real audio
/// pipeline drives it from Phase 4 on. `Error` is the fail-fast state for a
/// missing/failed device (`PERFORMANCE.md` → Startup path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListeningState {
    Listening,
    Paused,
    Error,
}

impl ListeningState {
    pub fn label(self) -> &'static str {
        match self {
            ListeningState::Listening => "Listening",
            ListeningState::Paused => "Paused",
            ListeningState::Error => "Error",
        }
    }
}

/// An in-flight theme cross-fade (`MOTION.md` → Theme switch): interpolate from
/// one theme's tokens to another over the standard duration, then settle. Held
/// as explicit state (not per-widget) so the shell can decide, each frame,
/// whether an animation is still active and therefore whether to repaint again.
#[derive(Debug, Clone, Copy)]
pub struct ThemeTransition {
    pub from: Theme,
    pub to: Theme,
    pub started_at: Instant,
}

/// The one owner of UI-thread state.
pub struct AppState {
    pub config: Config,
    /// Where `config` persists. `None` if `%APPDATA%` was unavailable (the app
    /// still runs with in-memory settings) or in tests.
    pub config_path: Option<PathBuf>,
    pub screen: Screen,
    pub listening: ListeningState,
    /// Set while a theme cross-fade is animating; cleared once it settles.
    pub theme_transition: Option<ThemeTransition>,
    /// Last-known OS dark-mode preference, consulted for [`Theme::System`].
    pub system_dark: bool,
}

impl AppState {
    /// Build initial state, loading `config.json` from `%APPDATA%` if present.
    /// A *corrupt* config is logged and falls back to defaults rather than
    /// stopping launch — the user can then reach Settings to fix it.
    pub fn load(system_dark: bool) -> Self {
        let config_path = config_file_path();
        let config = match &config_path {
            Some(path) if path.exists() => match Config::load(path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!(
                        "warning: could not load {}: {e}; using defaults",
                        path.display()
                    );
                    Config::default()
                }
            },
            _ => Config::default(),
        };
        Self::with_config(config, config_path, system_dark)
    }

    /// Core constructor from an already-resolved config and (optional) path.
    /// Tests pass `None` for the path so no real user file is ever touched.
    pub fn with_config(config: Config, config_path: Option<PathBuf>, system_dark: bool) -> Self {
        Self {
            config,
            config_path,
            screen: Screen::Home,
            listening: ListeningState::Paused,
            theme_transition: None,
            system_dark,
        }
    }

    /// Persist the current config best-effort. A write failure is surfaced on
    /// stderr; the in-memory setting still applies for this session. No-op when
    /// there is no path (tests, or `%APPDATA%` unavailable).
    pub fn save_config(&self) {
        let Some(path) = &self.config_path else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = self.config.save(path) {
            eprintln!("warning: could not save {}: {e}", path.display());
        }
    }

    /// Persist a freshly calibrated profile to `%APPDATA%\Qwerty\Profiles\
    /// <id>.json` and make it the active profile. Returns a human-readable
    /// error string for the wizard to surface. No-op-safe when there is no
    /// config path (returns an error rather than silently dropping the profile).
    pub fn save_profile(&mut self, profile: &Profile) -> Result<(), String> {
        let Some(cfg_path) = &self.config_path else {
            return Err("no %APPDATA% location is available to save the profile".into());
        };
        let dir = cfg_path
            .parent()
            .map(|p| p.join("Profiles"))
            .ok_or("invalid config path")?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join(format!("{}.json", profile.id));
        profile.save(&path).map_err(|e| e.to_string())?;
        self.config.active_profile_id = Some(profile.id);
        self.save_config();
        Ok(())
    }

    /// Toggle between Listening and Paused. Does nothing in the `Error` state —
    /// a device error must be resolved, not toggled past (`ARCHITECTURE.md` →
    /// fail fast). Returns the new state.
    pub fn toggle_listening(&mut self) -> ListeningState {
        self.listening = match self.listening {
            ListeningState::Listening => ListeningState::Paused,
            ListeningState::Paused => ListeningState::Listening,
            ListeningState::Error => ListeningState::Error,
        };
        self.listening
    }

    /// Begin switching to `theme`. Records the cross-fade transition and updates
    /// the in-memory config, but does **not** persist — the shell calls
    /// [`save_config`](Self::save_config) when this returns `true`. Returns
    /// `false` (no-op) if `theme` is already active.
    pub fn begin_theme_switch(&mut self, theme: Theme, now: Instant) -> bool {
        if self.config.theme == theme {
            return false;
        }
        self.theme_transition = Some(ThemeTransition {
            from: self.config.theme,
            to: theme,
            started_at: now,
        });
        self.config.theme = theme;
        true
    }
}

/// The on-disk config location: `%APPDATA%\Qwerty\config.json`
/// (`ARCHITECTURE.md` → Persistence). `None` if `%APPDATA%` is not set.
fn config_file_path() -> Option<PathBuf> {
    let appdata = std::env::var_os("APPDATA")?;
    Some(PathBuf::from(appdata).join("Qwerty").join("config.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_state() -> AppState {
        // No path → save_config is a no-op, so tests never touch the real
        // %APPDATA% config.json.
        AppState::with_config(Config::default(), None, true)
    }

    #[test]
    fn toggle_flips_between_listening_and_paused() {
        let mut s = test_state();
        s.listening = ListeningState::Paused;
        assert_eq!(s.toggle_listening(), ListeningState::Listening);
        assert_eq!(s.toggle_listening(), ListeningState::Paused);
    }

    #[test]
    fn toggle_does_not_escape_error_state() {
        let mut s = test_state();
        s.listening = ListeningState::Error;
        assert_eq!(s.toggle_listening(), ListeningState::Error);
    }

    #[test]
    fn switching_to_same_theme_is_a_noop() {
        let mut s = test_state();
        let current = s.config.theme;
        assert!(!s.begin_theme_switch(current, Instant::now()));
        assert!(s.theme_transition.is_none());
    }

    #[test]
    fn switching_theme_records_transition_and_updates_config() {
        let mut s = test_state();
        s.config.theme = Theme::Daylight;
        assert!(s.begin_theme_switch(Theme::Midnight, Instant::now()));
        assert_eq!(s.config.theme, Theme::Midnight);
        let t = s.theme_transition.expect("transition recorded");
        assert_eq!(t.from, Theme::Daylight);
        assert_eq!(t.to, Theme::Midnight);
    }

    #[test]
    fn screen_rail_has_all_five_and_labels() {
        assert_eq!(Screen::ALL.len(), 5);
        assert_eq!(Screen::Home.label(), "Home");
        assert_eq!(Screen::Settings.label(), "Settings");
    }
}
