//! The platform-actions seam (`ARCHITECTURE.md` → The `PlatformActions` trait).
//!
//! Everything from "action dispatcher" onward is platform-specific and lives
//! behind this one trait. `qwerty-core` owns the [`ActionSpec`] *data* enum;
//! *acting* on it is platform territory, so it lives here in `qwerty-app`. The
//! routing from an `ActionSpec` to a trait call is a pure function
//! ([`dispatch`]) that is unit-tested against a mock — the only OS-touching
//! code is the one concrete implementation (`windows.rs`). This is the seam a
//! future Linux port implements; nothing else changes.

#[cfg(windows)]
pub mod windows;

use std::path::Path;

use qwerty_core::profile::{ActionSpec, KeyCombo, ScreenshotMode, SystemSound};

/// A toast notification to surface to the user. Platform-neutral; the concrete
/// WinRT toast XML is built inside the Windows implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToastSpec {
    pub title: String,
    pub body: String,
}

/// Why an action failed. A real enum (not a stringly-typed catch-all) so the UI
/// can tell the user *why* — a missing target reads differently from a denied
/// permission (`ARCHITECTURE.md` → Error handling policy).
#[derive(Debug)]
pub enum ActionError {
    /// The app/file/URL to open does not exist or could not be resolved.
    TargetNotFound(String),
    /// Launching an app/path/URL or running a command failed at the OS level.
    LaunchFailed { target: String, reason: String },
    /// Reading or writing the system clipboard failed.
    Clipboard(String),
    /// Text-to-speech was unavailable or failed (from the `tts` backend).
    Speech(String),
    /// Synthesizing a keystroke failed (e.g. an unrecognized key name).
    Keystroke(String),
    /// Capturing the screen failed.
    Screenshot(String),
    /// Showing a toast notification failed, or this app's notification identity
    /// (AUMID) could not be registered with Windows.
    Notification(String),
    /// Playing a system sound failed.
    Sound(String),
    /// An action variant the current build cannot perform (today: region-select
    /// screenshot). A loud, honest state, not a silent no-op.
    Unsupported(&'static str),
}

impl std::fmt::Display for ActionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActionError::TargetNotFound(t) => write!(f, "could not find \"{t}\""),
            ActionError::LaunchFailed { target, reason } => {
                write!(f, "could not launch \"{target}\": {reason}")
            }
            ActionError::Clipboard(e) => write!(f, "clipboard error: {e}"),
            ActionError::Speech(e) => write!(f, "text-to-speech error: {e}"),
            ActionError::Keystroke(e) => write!(f, "keystroke error: {e}"),
            ActionError::Screenshot(e) => write!(f, "screenshot error: {e}"),
            ActionError::Notification(e) => write!(f, "notification error: {e}"),
            ActionError::Sound(e) => write!(f, "sound error: {e}"),
            ActionError::Unsupported(what) => {
                write!(f, "not supported on this build: {what}")
            }
        }
    }
}

impl std::error::Error for ActionError {}

/// One trait, one implementation per OS. Windows is the only implementation
/// today; the signatures stay platform-neutral so Linux can follow without
/// touching the DSP core (`ARCHITECTURE.md`). The UI thread is the only caller
/// (`ARCHITECTURE.md` → Threading model).
pub trait PlatformActions {
    fn open_uri(&self, uri: &str) -> Result<(), ActionError>;
    fn open_path(&self, path: &Path) -> Result<(), ActionError>;
    fn run_command(&self, cmd: &str, args: &[String]) -> Result<(), ActionError>;
    fn copy_to_clipboard(&self, text: &str) -> Result<(), ActionError>;
    fn speak(&self, text: &str) -> Result<(), ActionError>;
    fn send_keystroke(&self, combo: &KeyCombo) -> Result<(), ActionError>;
    fn play_sound(&self, sound: SystemSound) -> Result<(), ActionError>;
    fn screenshot_to_clipboard(&self, mode: ScreenshotMode) -> Result<(), ActionError>;
    fn notify(&self, toast: &ToastSpec) -> Result<(), ActionError>;
}

/// Route one [`ActionSpec`] to the matching platform call. `VisualOnly` is a
/// no-op here — it produces only in-app visual feedback, no side effect.
pub fn dispatch(action: &ActionSpec, platform: &dyn PlatformActions) -> Result<(), ActionError> {
    match action {
        // Launching an executable and opening a file/folder both go through the
        // shell open verb on Windows; the distinction is intent, not mechanism.
        ActionSpec::OpenApp { path } | ActionSpec::OpenPath { path } => {
            platform.open_path(Path::new(path))
        }
        ActionSpec::OpenUrl { url } => platform.open_uri(url),
        ActionSpec::RunCommand { command, args } => platform.run_command(command, args),
        ActionSpec::CopyToClipboard { text } => platform.copy_to_clipboard(text),
        ActionSpec::Speak { text } => platform.speak(text),
        ActionSpec::PlaySound { sound } => platform.play_sound(*sound),
        ActionSpec::ShowToast { title, body } => platform.notify(&ToastSpec {
            title: title.clone(),
            body: body.clone(),
        }),
        ActionSpec::SendKeystroke { combo } => platform.send_keystroke(combo),
        ActionSpec::Screenshot { mode } => platform.screenshot_to_clipboard(*mode),
        ActionSpec::VisualOnly => Ok(()),
    }
}

/// Run a zone's macro chain in order, stopping at the first failure (`DESIGN.md`
/// → macro chains). Returns the index of the failing action alongside its error
/// so the UI can point at exactly which step failed. Called live from the Home
/// listening loop when an accepted tap classifies to a zone (`shell.rs` →
/// `reconcile_capture`), and from the per-action Test button.
pub fn dispatch_chain(
    actions: &[ActionSpec],
    platform: &dyn PlatformActions,
) -> Result<(), (usize, ActionError)> {
    for (i, action) in actions.iter().enumerate() {
        dispatch(action, platform).map_err(|e| (i, e))?;
    }
    Ok(())
}

/// Construct the action backend for the current OS. Windows-first: the concrete
/// backend is `windows::WindowsPlatform`. Construction is infallible (it degrades
/// with a warning like the tray/hotkey setup, rather than failing startup).
#[cfg(windows)]
pub fn platform_for_this_os() -> Box<dyn PlatformActions> {
    Box::new(windows::WindowsPlatform::new())
}

/// Whether the OS asks for reduced motion (`DESIGN.md`/`ACCEPTANCE.md`:
/// accessibility). Read once at startup and cached in `AppState`. On Windows
/// this consults the "show animations" accessibility setting; other platforms
/// (none today) default to full motion until they implement their own read.
#[cfg(windows)]
pub fn os_prefers_reduced_motion() -> bool {
    windows::os_prefers_reduced_motion()
}

#[cfg(not(windows))]
pub fn os_prefers_reduced_motion() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Records the calls dispatched to it; every method can be made to fail.
    struct MockPlatform {
        calls: RefCell<Vec<String>>,
        fail_on: Option<&'static str>,
    }

    impl MockPlatform {
        fn new() -> Self {
            Self {
                calls: RefCell::new(Vec::new()),
                fail_on: None,
            }
        }
        fn failing(method: &'static str) -> Self {
            Self {
                calls: RefCell::new(Vec::new()),
                fail_on: Some(method),
            }
        }
        fn record(&self, m: &'static str) -> Result<(), ActionError> {
            self.calls.borrow_mut().push(m.to_string());
            if self.fail_on == Some(m) {
                Err(ActionError::LaunchFailed {
                    target: m.into(),
                    reason: "mock".into(),
                })
            } else {
                Ok(())
            }
        }
    }

    impl PlatformActions for MockPlatform {
        fn open_uri(&self, _: &str) -> Result<(), ActionError> {
            self.record("open_uri")
        }
        fn open_path(&self, _: &Path) -> Result<(), ActionError> {
            self.record("open_path")
        }
        fn run_command(&self, _: &str, _: &[String]) -> Result<(), ActionError> {
            self.record("run_command")
        }
        fn copy_to_clipboard(&self, _: &str) -> Result<(), ActionError> {
            self.record("copy_to_clipboard")
        }
        fn speak(&self, _: &str) -> Result<(), ActionError> {
            self.record("speak")
        }
        fn send_keystroke(&self, _: &KeyCombo) -> Result<(), ActionError> {
            self.record("send_keystroke")
        }
        fn play_sound(&self, _: SystemSound) -> Result<(), ActionError> {
            self.record("play_sound")
        }
        fn screenshot_to_clipboard(&self, _: ScreenshotMode) -> Result<(), ActionError> {
            self.record("screenshot_to_clipboard")
        }
        fn notify(&self, _: &ToastSpec) -> Result<(), ActionError> {
            self.record("notify")
        }
    }

    #[test]
    fn each_action_routes_to_the_right_call() {
        let cases: Vec<(ActionSpec, &str)> = vec![
            (ActionSpec::OpenApp { path: "a".into() }, "open_path"),
            (ActionSpec::OpenPath { path: "a".into() }, "open_path"),
            (ActionSpec::OpenUrl { url: "u".into() }, "open_uri"),
            (
                ActionSpec::RunCommand {
                    command: "c".into(),
                    args: vec![],
                },
                "run_command",
            ),
            (
                ActionSpec::CopyToClipboard { text: "t".into() },
                "copy_to_clipboard",
            ),
            (ActionSpec::Speak { text: "t".into() }, "speak"),
            (
                ActionSpec::PlaySound {
                    sound: SystemSound::Success,
                },
                "play_sound",
            ),
            (
                ActionSpec::ShowToast {
                    title: "t".into(),
                    body: "b".into(),
                },
                "notify",
            ),
            (
                ActionSpec::Screenshot {
                    mode: ScreenshotMode::FullScreen,
                },
                "screenshot_to_clipboard",
            ),
        ];
        for (spec, expected) in cases {
            let mock = MockPlatform::new();
            dispatch(&spec, &mock).unwrap();
            assert_eq!(mock.calls.borrow().as_slice(), &[expected.to_string()]);
        }
    }

    #[test]
    fn visual_only_is_a_no_op() {
        let mock = MockPlatform::new();
        dispatch(&ActionSpec::VisualOnly, &mock).unwrap();
        assert!(mock.calls.borrow().is_empty());
    }

    #[test]
    fn chain_runs_in_order_and_stops_at_first_failure() {
        let mock = MockPlatform::failing("speak");
        let chain = vec![
            ActionSpec::CopyToClipboard { text: "t".into() },
            ActionSpec::Speak { text: "t".into() }, // fails here (index 1)
            ActionSpec::OpenUrl { url: "u".into() }, // must not run
        ];
        let err = dispatch_chain(&chain, &mock).unwrap_err();
        assert_eq!(err.0, 1);
        assert_eq!(
            mock.calls.borrow().as_slice(),
            &["copy_to_clipboard".to_string(), "speak".to_string()]
        );
    }
}
