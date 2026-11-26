//! The complete `Route` of ui-v2 §3.5 (U:483-505): primary routes, subpages, takeovers and the
//! deep-link targets. Data only: navigation (transitions, back, memory) is `model/navigation.rs`
//! (WP-S1).

use serde::{Deserialize, Serialize};

use qwerty_core::profile::{ProfileId, ZoneId};

use crate::engine::protocol::ReportId;

/// Where the window is. Serde, so a toast's launch argument can carry it (WP-W4).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Route {
    /// Takeover.
    FirstRun { step: FirstRunStep },
    /// Primary (Ctrl+1).
    Home,
    /// Primary (Ctrl+2); a deep link may target one zone.
    Zones {
        selected: Option<ZoneId>,
        layout_editing: bool,
    },
    /// Primary (Ctrl+3).
    Evaluation,
    /// Subpage of Evaluation.
    EvaluationReport { report_id: ReportId },
    /// Takeover.
    EvaluationRun,
    /// Primary (Ctrl+4).
    Diagnostics,
    /// Subpage of Settings.
    Profiles,
    /// Primary (Ctrl+5); a deep link may target one section.
    Settings { section: Option<SettingsSection> },
    /// Subpage of Settings.
    About,
    /// Takeover.
    Wizard { mode: WizardMode, step: WizardStep },
}

/// The two first-run pages (U:534-560).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum FirstRunStep {
    Welcome,
    BeforeYouStart,
}

/// The Settings sections (U:1038-1090), the targets of `Settings { section }` deep links.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SettingsSection {
    Appearance,
    Listening,
    Microphone,
    Notifications,
    StartupAndWindow,
    Privacy,
    ProfilesAndAbout,
}

/// The wizard's two modes (U:734). `Recalibrate(profile)` is the deep-link target of U:504.
// SPEC: U:734 lists first run under New profile; whether a New run is the first run (no Cancel,
// U:729) is not a separate mode here, because first run is decided by an empty
// `Snapshot.profiles` (U:530), which the wizard view-model (WP-S6) reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WizardMode {
    New,
    Recalibrate { profile: ProfileId },
}

/// The wizard's six steps (U:730).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WizardStep {
    Zones,
    Room,
    Taps,
    Ignore,
    Check,
    Save,
}
