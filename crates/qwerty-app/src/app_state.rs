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

use qwerty_core::evaluation::EvaluationReport;
use qwerty_core::profile::{Config, Profile, ProfileId, Theme, ZoneId};

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

/// A lightweight summary of a saved profile for the switcher (`DESIGN.md` →
/// "Multiple desk profiles, switchable"). Avoids handing the whole [`Profile`]
/// to the UI just to render a name in a list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileSummary {
    pub id: ProfileId,
    pub name: String,
    pub zone_count: usize,
    /// Whether this is the currently active profile.
    pub is_active: bool,
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
    /// The currently-active profile, loaded from disk (`None` until one is
    /// calibrated/selected, or if its file is missing/corrupt).
    pub active_profile: Option<Profile>,
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
        let mut state = Self::with_config(config, config_path, system_dark);
        state.reload_active_profile();
        state
    }

    /// The `Profiles/` directory beside `config.json`, or `None` without a path.
    fn profiles_dir(&self) -> Option<PathBuf> {
        self.config_path
            .as_ref()
            .and_then(|p| p.parent())
            .map(|p| p.join("Profiles"))
    }

    /// The `Evaluations/` directory beside `config.json` (sibling of `Profiles/`,
    /// per `ARCHITECTURE.md` → Persistence), or `None` without a path.
    fn evaluations_dir(&self) -> Option<PathBuf> {
        self.config_path
            .as_ref()
            .and_then(|p| p.parent())
            .map(|p| p.join("Evaluations"))
    }

    /// A profile's evaluation-report directory: `Evaluations/<profile-id>/`
    /// (`DATA_MODEL.md`). Reports are kept per-profile so the Evaluation screen
    /// can show one profile's accuracy trend over time.
    fn profile_eval_dir(&self, profile_id: ProfileId) -> Option<PathBuf> {
        self.evaluations_dir()
            .map(|d| d.join(profile_id.to_string()))
    }

    /// (Re)load the active profile named by `config.active_profile_id` from
    /// disk into [`active_profile`](Self::active_profile). A missing or corrupt
    /// profile is logged and leaves it `None` — the app still runs and the user
    /// can recalibrate, rather than failing to start (fail loud, not fatal).
    pub fn reload_active_profile(&mut self) {
        self.active_profile = None;
        let (Some(id), Some(dir)) = (self.config.active_profile_id, self.profiles_dir()) else {
            return;
        };
        let path = dir.join(format!("{id}.json"));
        if !path.exists() {
            eprintln!(
                "warning: active profile {id} not found at {}",
                path.display()
            );
            return;
        }
        match Profile::load(&path) {
            Ok(p) => self.active_profile = Some(p),
            Err(e) => eprintln!("warning: could not load active profile: {e}"),
        }
    }

    /// List every saved profile by scanning `Profiles/`, sorted by name
    /// (`DESIGN.md` → "Multiple desk profiles, switchable"). A file that fails to
    /// load is logged and skipped rather than aborting the whole list — exactly
    /// how [`list_evaluation_reports`](Self::list_evaluation_reports) treats a bad
    /// report — so one corrupt profile never hides the others. A missing
    /// directory is the ordinary "no profiles yet" case (empty, silent); any
    /// other read error is logged, so an I/O fault is not disguised as "none".
    pub fn list_profiles(&self) -> Vec<ProfileSummary> {
        let Some(dir) = self.profiles_dir() else {
            return Vec::new();
        };
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
            Err(e) => {
                eprintln!(
                    "warning: could not read profiles directory {}: {e}",
                    dir.display()
                );
                return Vec::new();
            }
        };
        let active = self.config.active_profile_id;
        let mut out = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            match Profile::load(&path) {
                Ok(p) => out.push(ProfileSummary {
                    is_active: Some(p.id) == active,
                    id: p.id,
                    name: p.name,
                    zone_count: p.zones.len(),
                }),
                Err(e) => eprintln!(
                    "warning: skipping unreadable profile {}: {e}",
                    path.display()
                ),
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    /// Make `id` the active profile: persist the choice and load it into memory.
    /// The profile is loaded *before* the switch is committed, so a missing or
    /// corrupt target fails fast and leaves the current active profile untouched
    /// (no dangling `active_profile_id` pointing at an unloadable file). A
    /// config-write failure rolls the id back. A no-op (`Ok`) when `id` is
    /// already active.
    pub fn switch_profile(&mut self, id: ProfileId) -> Result<(), String> {
        if self.config.active_profile_id == Some(id) {
            return Ok(());
        }
        let dir = self
            .profiles_dir()
            .ok_or("no %APPDATA% location is available to switch profiles")?;
        let path = dir.join(format!("{id}.json"));
        // Prove it loads before committing the switch.
        Profile::load(&path).map_err(|e| e.to_string())?;
        let previous = self.config.active_profile_id;
        self.config.active_profile_id = Some(id);
        if let Err(e) = self.save_config() {
            self.config.active_profile_id = previous; // undo on persist failure
            return Err(e);
        }
        self.reload_active_profile();
        Ok(())
    }

    /// The `Exports/` interchange folder beside `config.json` (sibling of
    /// `Profiles/`) — where portable profile files are written by export and
    /// read by import (`DESIGN.md`: "Import / export a profile"). `None` without
    /// an `%APPDATA%` location.
    fn exports_dir(&self) -> Option<PathBuf> {
        self.config_path
            .as_ref()
            .and_then(|p| p.parent())
            .map(|p| p.join("Exports"))
    }

    /// Export the active profile to `Exports/<name>.json` as a portable,
    /// human-inspectable file the user can copy to another machine or share
    /// (`DESIGN.md`). Returns the path written. Errors (never silently) if there
    /// is no active profile or no `%APPDATA%` location.
    pub fn export_active_profile(&self) -> Result<PathBuf, String> {
        let profile = self
            .active_profile
            .as_ref()
            .ok_or("no active profile to export")?;
        let dir = self
            .exports_dir()
            .ok_or("no %APPDATA% location is available to export")?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join(format!("{}.json", sanitize_filename(&profile.name)));
        profile.save(&path).map_err(|e| e.to_string())?;
        Ok(path)
    }

    /// List the importable profile files in `Exports/` as `(path, display name)`,
    /// sorted by name. The display name is the profile's own name when the file
    /// loads, else the file stem, so a partially-corrupt drop is still listed by
    /// filename rather than vanishing. Empty without an `Exports/` directory.
    pub fn list_importable_profiles(&self) -> Vec<(PathBuf, String)> {
        let Some(dir) = self.exports_dir() else {
            return Vec::new();
        };
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
            Err(e) => {
                eprintln!(
                    "warning: could not read exports directory {}: {e}",
                    dir.display()
                );
                return Vec::new();
            }
        };
        let mut out = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let name = Profile::load(&path).map(|p| p.name).unwrap_or_else(|_| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("profile")
                    .to_string()
            });
            out.push((path, name));
        }
        out.sort_by(|a, b| a.1.cmp(&b.1));
        out
    }

    /// Import a profile from `path`: load + validate, assign a **fresh**
    /// `ProfileId` so an import never overwrites an existing profile (even one
    /// exported from this same machine), save it into `Profiles/`, and make it
    /// active (`DESIGN.md`). Returns the new id. A missing/corrupt/version-
    /// mismatched file fails fast via [`Profile::load`], leaving state unchanged.
    pub fn import_profile(&mut self, path: &std::path::Path) -> Result<ProfileId, String> {
        let mut profile = Profile::load(path).map_err(|e| e.to_string())?;
        profile.id = ProfileId::new();
        profile.updated_at = chrono::Utc::now();
        self.save_profile(&profile)?; // persists, sets active, reloads into memory
        Ok(profile.id)
    }

    /// Duplicate the active profile under `new_name`: a fresh `ProfileId` with
    /// the same calibration, zones, and action bindings, saved and made active.
    /// This is what makes one desk calibration reusable for a second context —
    /// e.g. duplicate a "Work" profile into "Game", then rebind the copy's zones
    /// — without calibrating the same desk twice. An empty/blank `new_name`
    /// falls back to "<source> copy". Errors if there is no active profile.
    pub fn duplicate_active_profile(&mut self, new_name: &str) -> Result<ProfileId, String> {
        let mut copy = self
            .active_profile
            .clone()
            .ok_or("no active profile to duplicate")?;
        copy.id = ProfileId::new();
        let name = new_name.trim();
        copy.name = if name.is_empty() {
            format!("{} copy", copy.name)
        } else {
            name.to_string()
        };
        let now = chrono::Utc::now();
        copy.created_at = now;
        copy.updated_at = now;
        self.save_profile(&copy)?; // persists, sets active, reloads into memory
        Ok(copy.id)
    }

    /// Delete a profile: remove its `Profiles/<id>.json` and its
    /// `Evaluations/<id>/` history (which would otherwise be orphaned), and if it
    /// was the active profile, clear the active selection so the app drops to the
    /// no-profile state rather than pointing at a file that no longer exists.
    /// Idempotent: deleting an already-absent profile is `Ok` (the end state —
    /// gone — is the same), so a double-click can't error. The eval-history
    /// removal is best-effort; its absence is not a failure.
    pub fn delete_profile(&mut self, id: ProfileId) -> Result<(), String> {
        let dir = self
            .profiles_dir()
            .ok_or("no %APPDATA% location is available to delete profiles")?;
        let path = dir.join(format!("{id}.json"));
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            // Already gone → the target state is already achieved; proceed to
            // clear the active selection if needed rather than reporting failure.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.to_string()),
        }
        if let Some(eval_dir) = self.profile_eval_dir(id) {
            let _ = std::fs::remove_dir_all(&eval_dir); // best-effort cleanup
        }
        if self.config.active_profile_id == Some(id) {
            self.config.active_profile_id = None;
            self.active_profile = None;
            self.save_config()?;
        }
        Ok(())
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
            active_profile: None,
        }
    }

    /// Persist the current config. Returns `Err` with a human-readable message
    /// on a write failure so the caller can surface it (Settings routes it into
    /// the save-error banner, `save_profile` propagates it) — the in-memory
    /// setting still applies for the session. `Ok(())` no-op when there is no
    /// path (tests, or `%APPDATA%` unavailable).
    pub fn save_config(&self) -> Result<(), String> {
        let Some(path) = &self.config_path else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        self.config.save(path).map_err(|e| e.to_string())
    }

    /// Persist a freshly calibrated profile to `%APPDATA%\Qwerty\Profiles\
    /// <id>.json` and make it the active profile. Returns a human-readable
    /// error string for the wizard to surface. No-op-safe when there is no
    /// config path (returns an error rather than silently dropping the profile).
    pub fn save_profile(&mut self, profile: &Profile) -> Result<(), String> {
        let dir = self
            .profiles_dir()
            .ok_or("no %APPDATA% location is available to save the profile")?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join(format!("{}.json", profile.id));
        profile.save(&path).map_err(|e| e.to_string())?;
        self.config.active_profile_id = Some(profile.id);
        self.save_config()?;
        // Reflect the newly-saved profile in memory immediately.
        self.active_profile = Some(profile.clone());
        Ok(())
    }

    /// Persist edits made in place to the active profile (e.g. action bindings),
    /// stamping `updated_at`. No-op error if there is no active profile.
    pub fn save_active_profile(&mut self) -> Result<(), String> {
        let mut p = self
            .active_profile
            .clone()
            .ok_or("no active profile to save")?;
        p.updated_at = chrono::Utc::now();
        self.save_profile(&p)
    }

    /// Persist the active profile on shutdown, but *only if* its in-memory
    /// state differs from what is already on disk. Returns whether a write
    /// actually happened.
    ///
    /// Free-text action fields commit on focus loss (`action_editor`), so a
    /// quit — which exits without a final blur — would drop a value that was
    /// typed but not yet committed, even though it is already live in the
    /// in-memory profile. Saving here closes that gap.
    ///
    /// The equality check is exact because editing a profile never stamps
    /// `updated_at`: an uncommitted edit differs in content while `updated_at`
    /// still matches the last save, so the profiles compare unequal and we save
    /// (stamping a fresh time, correctly, since a real edit occurred); an
    /// untouched profile round-trips equal to its on-disk copy (guaranteed by
    /// `qwerty_core`'s `profile_round_trips_with_no_field_loss`) and is skipped,
    /// so a no-op quit neither rewrites the file nor advances `updated_at`.
    pub fn save_active_profile_if_changed(&mut self) -> Result<bool, String> {
        let Some(current) = self.active_profile.clone() else {
            return Ok(false);
        };
        // A profile that loads and equals the in-memory copy has no unsaved
        // edit. Every other case (differs, file missing, unreadable) means the
        // good in-memory profile should win, so fall through and save.
        if let Some(dir) = self.profiles_dir() {
            let path = dir.join(format!("{}.json", current.id));
            if Profile::load(&path).is_ok_and(|on_disk| on_disk == current) {
                return Ok(false);
            }
        }
        self.save_active_profile()?;
        Ok(true)
    }

    /// Persist a finished evaluation report to
    /// `Evaluations/<profile-id>/<timestamp>.json` and return the path written.
    /// The filename uses a filesystem-safe compact timestamp (`:` is illegal in
    /// Windows paths, so RFC 3339 can't be the filename); the RFC-3339 `run_at`
    /// inside the JSON is unchanged. Returns an error string (never silently
    /// drops the report) when there is no `%APPDATA%` location.
    pub fn save_evaluation_report(&self, report: &EvaluationReport) -> Result<PathBuf, String> {
        let dir = self
            .profile_eval_dir(report.profile_id)
            .ok_or("no %APPDATA% location is available to save the report")?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let stamp = report.run_at.format("%Y%m%dT%H%M%S%3fZ").to_string();
        let path = dir.join(format!("{stamp}.json"));
        report.save(&path).map_err(|e| e.to_string())?;
        Ok(path)
    }

    /// Export a finished report as CSV beside its JSON at
    /// `Evaluations/<profile-id>/<timestamp>.csv` (`DESIGN.md` → Evaluation:
    /// "Exportable JSON/CSV"). Reuses the same filesystem-safe timestamp as
    /// [`save_evaluation_report`](Self::save_evaluation_report), so the CSV and
    /// JSON of one run pair up by name. `zone_ids`/`zone_labels` are in
    /// confusion-matrix order (the Evaluation screen holds both from the run).
    /// Returns the path written, or an error string — never a silent drop.
    pub fn export_evaluation_csv(
        &self,
        report: &EvaluationReport,
        zone_ids: &[ZoneId],
        zone_labels: &[String],
    ) -> Result<PathBuf, String> {
        let dir = self
            .profile_eval_dir(report.profile_id)
            .ok_or("no %APPDATA% location is available to export the report")?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let stamp = report.run_at.format("%Y%m%dT%H%M%S%3fZ").to_string();
        let path = dir.join(format!("{stamp}.csv"));
        std::fs::write(&path, report.to_csv(zone_ids, zone_labels)).map_err(|e| e.to_string())?;
        Ok(path)
    }

    /// Load a profile's saved evaluation reports, oldest first (by `run_at`), for
    /// the history/trend view. A report that fails to load (corrupt or a schema
    /// mismatch from an older build) is logged and skipped rather than aborting
    /// the whole list — the history view is read-only presentation, not the
    /// primary persistence path, so one bad historical file must not hide the
    /// rest. Returns empty without an `%APPDATA%` location or profile directory.
    pub fn list_evaluation_reports(&self, profile_id: ProfileId) -> Vec<EvaluationReport> {
        let Some(dir) = self.profile_eval_dir(profile_id) else {
            return Vec::new();
        };
        // A missing directory is the ordinary "never evaluated yet" case and is
        // silent; any *other* read failure (permissions, a lock, the path being
        // a file) is a real problem and is logged, matching how an unreadable
        // individual report below is reported. Collapsing both into an empty
        // list would disguise an I/O fault as "no history".
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
            Err(e) => {
                eprintln!(
                    "warning: could not read evaluations directory {}: {e}",
                    dir.display()
                );
                return Vec::new();
            }
        };
        let mut reports = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            match EvaluationReport::load(&path) {
                Ok(r) => reports.push(r),
                Err(e) => eprintln!(
                    "warning: skipping unreadable evaluation report {}: {e}",
                    path.display()
                ),
            }
        }
        reports.sort_by_key(|r| r.run_at);
        reports
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

/// Make a profile name safe to use as an export filename: replace the
/// Windows-illegal path characters (`< > : " / \ | ? *`) and control characters
/// with `_`, trim surrounding whitespace and dots, and fall back to `profile`
/// when nothing usable remains. The profile's real name inside the JSON is
/// unchanged — this only shapes the filename.
fn sanitize_filename(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if matches!(c, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*') || c.is_control() {
                '_'
            } else {
                c
            }
        })
        .collect();
    let trimmed = cleaned.trim().trim_matches('.').trim();
    if trimmed.is_empty() {
        "profile".to_string()
    } else {
        trimmed.to_string()
    }
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

    /// A throwaway state whose config/profiles live under a unique temp dir, so
    /// profile-management I/O is exercised without touching the real %APPDATA%.
    fn temp_state(tag: &str) -> (AppState, PathBuf) {
        let dir = std::env::temp_dir().join(format!("qwerty_test_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("Profiles")).unwrap();
        let config_path = dir.join("config.json");
        (
            AppState::with_config(Config::default(), Some(config_path), true),
            dir,
        )
    }

    #[test]
    fn list_profiles_skips_non_json_and_corrupt_files() {
        let (s, dir) = temp_state("list_skips");
        let profiles = dir.join("Profiles");
        std::fs::write(profiles.join("notes.txt"), "not a profile").unwrap();
        std::fs::write(profiles.join("broken.json"), "{ not valid profile").unwrap();
        // Neither loads as a profile, so the list is empty — and never panics on
        // the junk file.
        assert!(s.list_profiles().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_profiles_is_empty_without_a_profiles_dir() {
        // No config path → no %APPDATA% location → empty, never an error.
        assert!(test_state().list_profiles().is_empty());
    }

    #[test]
    fn switch_to_a_missing_profile_fails_and_keeps_the_current_active() {
        let (mut s, dir) = temp_state("switch_missing");
        let original = s.config.active_profile_id;
        assert!(s.switch_profile(ProfileId::new()).is_err());
        // The failed switch left the active id untouched (no dangling pointer).
        assert_eq!(s.config.active_profile_id, original);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn switching_to_the_already_active_profile_is_a_noop_ok() {
        let mut s = test_state();
        let id = ProfileId::new();
        s.config.active_profile_id = Some(id);
        // Already active → Ok early, without needing a path or disk access.
        assert!(s.switch_profile(id).is_ok());
    }

    #[test]
    fn sanitize_filename_replaces_illegal_chars_and_has_a_fallback() {
        assert_eq!(sanitize_filename("Home desk"), "Home desk");
        assert_eq!(sanitize_filename("a/b:c*d?"), "a_b_c_d_");
        assert_eq!(sanitize_filename("  ..  "), "profile");
        assert_eq!(sanitize_filename(""), "profile");
    }

    #[test]
    fn exporting_without_an_active_profile_errors() {
        // No active profile → a clear error, never a silent no-op.
        assert!(test_state().export_active_profile().is_err());
    }

    #[test]
    fn importing_a_missing_or_corrupt_file_fails_and_changes_nothing() {
        let (mut s, dir) = temp_state("import_bad");
        // Missing file.
        assert!(s.import_profile(&dir.join("nope.json")).is_err());
        // Corrupt file.
        let bad = dir.join("bad.json");
        std::fs::write(&bad, "{ not a profile").unwrap();
        assert!(s.import_profile(&bad).is_err());
        // Neither touched the active profile.
        assert!(s.active_profile.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_importable_is_empty_without_an_exports_dir() {
        assert!(test_state().list_importable_profiles().is_empty());
    }

    #[test]
    fn duplicating_without_an_active_profile_errors() {
        // Nothing to copy → a clear error, never a silent empty profile.
        assert!(test_state().duplicate_active_profile("Game").is_err());
    }

    #[test]
    fn deleting_the_active_profile_removes_its_file_and_clears_active() {
        let (mut s, dir) = temp_state("delete_active");
        let id = ProfileId::new();
        let path = dir.join("Profiles").join(format!("{id}.json"));
        std::fs::write(&path, "{}").unwrap(); // a stand-in file (delete is a file op)
        s.config.active_profile_id = Some(id);

        assert!(s.delete_profile(id).is_ok());
        assert!(!path.exists(), "the profile file should be gone");
        assert_eq!(s.config.active_profile_id, None, "active must be cleared");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deleting_a_non_active_profile_leaves_the_active_selection() {
        let (mut s, dir) = temp_state("delete_other");
        let active = ProfileId::new();
        let other = ProfileId::new();
        let other_path = dir.join("Profiles").join(format!("{other}.json"));
        std::fs::write(&other_path, "{}").unwrap();
        s.config.active_profile_id = Some(active);

        assert!(s.delete_profile(other).is_ok());
        assert!(!other_path.exists());
        assert_eq!(s.config.active_profile_id, Some(active));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deleting_an_absent_profile_is_ok_and_idempotent() {
        let (mut s, dir) = temp_state("delete_absent");
        // Never created — deleting it still succeeds (target state already met).
        assert!(s.delete_profile(ProfileId::new()).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
