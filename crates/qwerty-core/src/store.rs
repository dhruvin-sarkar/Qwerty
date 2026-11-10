//! Config and Profile v2 persistence (§1.16).
//!
//! The v2 `Config` and `Profile`, their validation, the profile edits the engine applies, and the
//! validated JSON writers. Loading, and the explicit v1 migration, live in `migrate`. The core holds
//! no paths (§1.16.3): a `file` argument only names the source in an error.

use std::fmt;
use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::audio::{ChannelLayout, DeviceIdentity, IdentityField};
use crate::detector::FRONT_END_VERSION;
use crate::gesture::Trigger;
use crate::model::TrainedModel;
use crate::profile::{
    ActionSpec, KeyCombo, Modifier, NotificationVerbosity, PersistenceError, ProfileId, Theme,
    ZoneId, ZoneLayout,
};
use crate::session::EnvironmentQuality;

pub const CONFIG_SCHEMA_VERSION: u32 = 2;
pub const PROFILE_SCHEMA_VERSION: u32 = 2;

// SPEC: §1.16.1 gives `Config::initial()` assessment::DEFAULT_PROMPTS_PER_ZONE (15). That constant is
// WP-B1's (assessment.rs), built in parallel with this package, so the initial value is named here.
/// The evaluation prompts per zone a new or migrated config starts with.
const INITIAL_EVAL_PROMPTS_PER_ZONE: u32 = 15;
/// The text zoom a new or migrated config starts with (percent).
const INITIAL_TEXT_ZOOM: u16 = 100;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub schema_version: u32,
    pub active_profile_id: Option<ProfileId>,
    pub theme: Theme,
    pub start_with_windows: bool,
    pub minimize_to_tray_on_close: bool,
    pub notification_verbosity: NotificationVerbosity,
    /// v2 honours this; v1 files migrate None → Some(Ctrl+Alt+Space) (§1.16.2).
    pub global_hotkey: Option<KeyCombo>,
    pub debug_capture_enabled: bool,
    /// "Ignore sounds while I'm typing or clicking" (ADR-004).
    pub input_veto_enabled: bool,
    // Shell-persisted settings (PARITY-3, XR-UI-12). The engine stores and validates them and
    // never interprets them.
    /// The last evaluation setting.
    pub eval_prompts_per_zone: u32,
    pub eval_background_check: bool,
    /// Ascending, no duplicates.
    pub dismissed_notices: Vec<NoticeId>,
    pub window_placement: Option<WindowPlacement>,
    /// Percent, one of `TEXT_ZOOM_STEPS` (XR-UI-19, ui-v2 §8.7).
    pub text_zoom: u16,
}

pub const TEXT_ZOOM_STEPS: [u16; 6] = [100, 110, 125, 150, 175, 200];

/// Appended only, so the derived order of existing values never changes. `UnattendedActions`: the
/// actions safety note (ui-v2 §4.3), dismissed through `ConfigEdit.dismiss_notice`; the engine never
/// reads it (XR-UI-21).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum NoticeId {
    FirstCloseToTray,
    UpgradeV2TypingFilter,
    UnattendedActions,
}

/// `GetWindowPlacement` semantics: the normal (restored) rectangle in workspace coordinates, plus
/// `maximized`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WindowPlacement {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub maximized: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ConfigError {
    HotkeyWithoutModifier,
    ZeroEvalPrompts,
    EmptyWindowPlacement,
    DuplicateNotice { notice: NoticeId },
    InvalidTextZoom { value: u16 },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ConfigError {}

impl Config {
    /// System theme, autostart off, close-to-tray on, Normal verbosity, Some(Ctrl+Alt+Space),
    /// debug capture off, input veto on, DEFAULT_PROMPTS_PER_ZONE (15) with the background
    /// check on, no dismissed notices, no window placement, text zoom 100.
    pub fn initial() -> Config {
        Config {
            schema_version: CONFIG_SCHEMA_VERSION,
            active_profile_id: None,
            theme: Theme::System,
            start_with_windows: false,
            minimize_to_tray_on_close: true,
            notification_verbosity: NotificationVerbosity::Normal,
            global_hotkey: Some(default_global_hotkey()),
            debug_capture_enabled: false,
            input_veto_enabled: true,
            eval_prompts_per_zone: INITIAL_EVAL_PROMPTS_PER_ZONE,
            eval_background_check: true,
            dismissed_notices: Vec::new(),
            window_placement: None,
            text_zoom: INITIAL_TEXT_ZOOM,
        }
    }

    /// A hotkey must include Ctrl, Alt or Win (Shift alone is not enough; PARITY-32);
    /// eval_prompts_per_zone ≥ 1; a placement has w > 0 and h > 0; notices are unique;
    /// text_zoom ∈ TEXT_ZOOM_STEPS (else InvalidTextZoom{value}).
    pub fn validate(&self) -> Result<(), ConfigError> {
        if let Some(hotkey) = &self.global_hotkey {
            if !has_chord_modifier(hotkey) {
                return Err(ConfigError::HotkeyWithoutModifier);
            }
        }
        if self.eval_prompts_per_zone == 0 {
            return Err(ConfigError::ZeroEvalPrompts);
        }
        if let Some(p) = &self.window_placement {
            if p.w <= 0 || p.h <= 0 {
                return Err(ConfigError::EmptyWindowPlacement);
            }
        }
        // SPEC: the field is documented "ascending, no duplicates", but ConfigError names only a
        // duplicate, so uniqueness is what is checked; any pair is compared, not just neighbours.
        let notices = &self.dismissed_notices;
        if let Some(notice) = notices
            .iter()
            .enumerate()
            .find(|(i, n)| notices[..*i].contains(n))
            .map(|(_, n)| *n)
        {
            return Err(ConfigError::DuplicateNotice { notice });
        }
        if !TEXT_ZOOM_STEPS.contains(&self.text_zoom) {
            return Err(ConfigError::InvalidTextZoom {
                value: self.text_zoom,
            });
        }
        Ok(())
    }
}

/// Ctrl+Alt+Space: the hotkey v1 always registered (v1 `hotkeys.rs:33-40`) and v2's initial one.
pub(crate) fn default_global_hotkey() -> KeyCombo {
    KeyCombo {
        modifiers: vec![Modifier::Ctrl, Modifier::Alt],
        key: "Space".to_string(),
    }
}

/// Ctrl, Alt or Win is present; Shift alone does not make a global chord (PARITY-32).
fn has_chord_modifier(combo: &KeyCombo) -> bool {
    combo.modifiers.iter().any(|m| match m {
        Modifier::Ctrl | Modifier::Alt | Modifier::Win => true,
        Modifier::Shift => false,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub struct ZoneCount(u8); // ∈ ZoneCount::ALLOWED (ACCEPTANCE)

impl ZoneCount {
    pub const ALLOWED: [u8; 4] = [2, 4, 6, 8];

    pub fn new(n: u8) -> Result<Self, ProfileError> {
        if Self::ALLOWED.contains(&n) {
            Ok(ZoneCount(n))
        } else {
            Err(ProfileError::InvalidZoneCount { n })
        }
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}

impl TryFrom<u8> for ZoneCount {
    type Error = ProfileError;
    fn try_from(n: u8) -> Result<Self, Self::Error> {
        ZoneCount::new(n)
    }
}

impl From<ZoneCount> for u8 {
    fn from(count: ZoneCount) -> u8 {
        count.0
    }
}

/// The count of `len` zones, if it is allowed (a length beyond u8 reports as 255).
pub(crate) fn zone_count_of(len: usize) -> Result<ZoneCount, ProfileError> {
    ZoneCount::new(len.min(usize::from(u8::MAX)) as u8)
}

/// Validation limits, public so the UI binds to them (PARITY-32). Lengths are in `char`s, after
/// trimming.
pub const PROFILE_NAME_MAX: usize = 60;
pub const ZONE_LABEL_MAX: usize = 32; // ui-v2 §4.3

const fn layout(x: f32, y: f32, width: f32, height: f32) -> ZoneLayout {
    ZoneLayout {
        x,
        y,
        width,
        height,
    }
}

type DefaultZone = (&'static str, ZoneLayout);

const ZONES_2: [DefaultZone; 2] = [
    ("Left", layout(0.03, 0.08, 0.29, 0.84)),
    ("Right", layout(0.68, 0.08, 0.29, 0.84)),
];
const ZONES_4: [DefaultZone; 4] = [
    ("Left back", layout(0.03, 0.08, 0.29, 0.40)),
    ("Right back", layout(0.68, 0.08, 0.29, 0.40)),
    ("Left front", layout(0.03, 0.52, 0.29, 0.40)),
    ("Right front", layout(0.68, 0.52, 0.29, 0.40)),
];
const NEAR_LEFT: DefaultZone = ("Near left", layout(0.35, 0.64, 0.145, 0.28));
const NEAR_RIGHT: DefaultZone = ("Near right", layout(0.505, 0.64, 0.145, 0.28));
const ZONES_6: [DefaultZone; 6] = [
    ZONES_4[0], ZONES_4[1], ZONES_4[2], ZONES_4[3], NEAR_LEFT, NEAR_RIGHT,
];
const ZONES_8: [DefaultZone; 8] = [
    ("Left back", layout(0.03, 0.08, 0.29, 0.2533)),
    ("Right back", layout(0.68, 0.08, 0.29, 0.2533)),
    ("Left middle", layout(0.03, 0.3733, 0.29, 0.2533)),
    ("Right middle", layout(0.68, 0.3733, 0.29, 0.2533)),
    ("Left front", layout(0.03, 0.6667, 0.29, 0.2533)),
    ("Right front", layout(0.68, 0.6667, 0.29, 0.2533)),
    NEAR_LEFT,
    NEAR_RIGHT,
];

/// The single source of default zone names and positions (ui-v2 Appendix A; PARITY-33).
/// (x, y, width, height) on the 2:1 desk canvas; no default overlaps another or the laptop footprint
/// (x 0.35-0.65, y 0.08-0.58).
pub fn default_zones(n: ZoneCount) -> Vec<(&'static str, ZoneLayout)> {
    let table: &[DefaultZone] = match n.get() {
        2 => &ZONES_2,
        4 => &ZONES_4,
        6 => &ZONES_6,
        8 => &ZONES_8,
        // INVARIANT: every ZoneCount comes from `ZoneCount::new` (deserialization included, through
        // `try_from`), which admits only ZoneCount::ALLOWED = [2, 4, 6, 8].
        other => unreachable!("zone count {other} is not in ZoneCount::ALLOWED"),
    };
    table.to_vec()
}

/// Finite, inside the [0, 1]² canvas, with width and height > 0 (PARITY-19, as v1). Compared in
/// f32, the stored type, so a zone the editor clamps flush to an edge in f32 stays valid.
pub(crate) fn layout_is_valid(l: &ZoneLayout) -> bool {
    let finite = [l.x, l.y, l.width, l.height].iter().all(|v| v.is_finite());
    finite
        && l.x >= 0.0
        && l.y >= 0.0
        && l.width > 0.0
        && l.height > 0.0
        && l.x + l.width <= 1.0
        && l.y + l.height <= 1.0
}

/// The form two labels are compared in: trimmed and lowercased (PARITY-32).
pub(crate) fn label_key(label: &str) -> String {
    label.trim().to_lowercase()
}

/// `base`, or the first of "base 2", "base 3", … whose key is not in `taken` (label keys).
// SPEC: §1.16 appends " 2", " 3", … but does not say what happens when that pushes a label past
// ZONE_LABEL_MAX; the stem is shortened so the result always passes `Profile::validate`.
pub(crate) fn unique_label(base: &str, taken: &[String]) -> String {
    if !taken.contains(&label_key(base)) {
        return base.to_string();
    }
    let mut k: u32 = 2;
    loop {
        let suffix = format!(" {k}");
        let room = ZONE_LABEL_MAX.saturating_sub(suffix.chars().count());
        let candidate: String = base.chars().take(room).chain(suffix.chars()).collect();
        if !taken.contains(&label_key(&candidate)) {
            return candidate;
        }
        k += 1;
    }
}

/// usize → u32 for an error report or a count, saturating.
pub(crate) fn report_u32(n: usize) -> u32 {
    n.min(u32::MAX as usize) as u32
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Zone {
    pub id: ZoneId,
    pub label: String,
    pub layout: ZoneLayout,
    pub trigger: Trigger,
    pub actions: Vec<ActionSpec>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceFingerprint {
    pub identity: DeviceIdentity,
    pub layout: ChannelLayout,
}

// SPEC: §1.16.1 fixes `Calibrated { model: TrainedModel, .. }` by value; a profile holds one state
// and is never in a hot path, so the variant size difference costs nothing.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", deny_unknown_fields)] // §0.5 (was "state"; RUST-RT-24)
pub enum CalibrationState {
    Calibrated {
        device: DeviceFingerprint,
        model: TrainedModel,
        calibrated_at: DateTime<Utc>,
        environment: EnvironmentQuality,
    },
    NeedsRecalibration {
        reason: RecalibrationReason,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum RecalibrationReason {
    NewProfile,
    MigratedFromV1 { v1_device_name: String },
    ZoneSetChanged,
    FrontEndChanged { trained: u32, current: u32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub schema_version: u32,
    pub id: ProfileId,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub zones: Vec<Zone>,
    pub calibration: CalibrationState,
    /// Set by the engine's import (p2 §2.9.4) when the bound endpoint id is not on this PC; None
    /// from `new` and from v1 migration. No core rule reads it (XR-UI-16, X01-14).
    pub imported_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ProfileError {
    InvalidZoneCount {
        n: u8,
    },
    EmptyName,
    /// PROFILE_NAME_MAX
    NameTooLong {
        len: u32,
        max: u32,
    },
    EmptyZoneLabel {
        zone: ZoneId,
    },
    /// ZONE_LABEL_MAX
    ZoneLabelTooLong {
        zone: ZoneId,
        len: u32,
        max: u32,
    },
    /// Equal after trimming and lowercasing (PARITY-32).
    DuplicateZoneLabel {
        zone: ZoneId,
        other: ZoneId,
    },
    /// Non-finite, or not inside [0, 1]² with width, height > 0 (PARITY-19, as v1).
    InvalidLayout {
        zone: ZoneId,
    },
    /// ActionSpec::VisualOnly: an empty chain already means "lights up only" (PARITY-4); deleted
    /// with the variant in WP-D1.
    UnsupportedAction {
        zone: ZoneId,
        index: u32,
    },
    DuplicateZoneId {
        zone: ZoneId,
    },
    ZoneNotFound {
        zone: ZoneId,
    },
    ActionIndexOutOfRange {
        zone: ZoneId,
        index: u32,
        len: u32,
    },
    /// Calibrated model.zone_ids != zones ids (in order).
    ModelZonesMismatch,
    /// rebind_device on NeedsRecalibration (X01-14).
    NotCalibrated,
    /// rebind_device: the differences other than EndpointId (X01-14).
    DeviceDoesNotMatch {
        fields: Vec<IdentityField>,
        name_differs: bool,
    },
}

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ProfileError {}

/// Whether a v2 zone may hold this action. Exhaustive, so a new variant must decide.
fn action_supported(action: &ActionSpec) -> bool {
    match action {
        ActionSpec::OpenApp { .. }
        | ActionSpec::OpenPath { .. }
        | ActionSpec::OpenUrl { .. }
        | ActionSpec::RunCommand { .. }
        | ActionSpec::CopyToClipboard { .. }
        | ActionSpec::Speak { .. }
        | ActionSpec::PlaySound { .. }
        | ActionSpec::ShowToast { .. }
        | ActionSpec::SendKeystroke { .. }
        | ActionSpec::Screenshot { .. } => true,
        ActionSpec::VisualOnly => false,
    }
}

impl Profile {
    /// NeedsRecalibration{NewProfile}; names and layouts from `default_zones(n)`; Trigger::Single;
    /// no actions; imported_at None.
    pub fn new(
        name: String,
        zones: ZoneCount,
        now: DateTime<Utc>,
    ) -> Result<Profile, ProfileError> {
        let profile = Profile {
            schema_version: PROFILE_SCHEMA_VERSION,
            id: ProfileId::new(),
            name,
            created_at: now,
            updated_at: now,
            zones: default_zones(zones)
                .into_iter()
                .map(|(label, layout)| Zone {
                    id: ZoneId::new(),
                    label: label.to_string(),
                    layout,
                    trigger: Trigger::Single,
                    actions: Vec::new(),
                })
                .collect(),
            calibration: CalibrationState::NeedsRecalibration {
                reason: RecalibrationReason::NewProfile,
            },
            imported_at: None,
        };
        profile.validate()?;
        Ok(profile)
    }

    pub fn validate(&self) -> Result<(), ProfileError> {
        let name_len = self.name.trim().chars().count();
        if name_len == 0 {
            return Err(ProfileError::EmptyName);
        }
        if name_len > PROFILE_NAME_MAX {
            return Err(ProfileError::NameTooLong {
                len: report_u32(name_len),
                max: report_u32(PROFILE_NAME_MAX),
            });
        }
        zone_count_of(self.zones.len())?;
        for (i, zone) in self.zones.iter().enumerate() {
            let earlier = &self.zones[..i];
            if earlier.iter().any(|o| o.id == zone.id) {
                return Err(ProfileError::DuplicateZoneId { zone: zone.id });
            }
            let label_len = zone.label.trim().chars().count();
            if label_len == 0 {
                return Err(ProfileError::EmptyZoneLabel { zone: zone.id });
            }
            if label_len > ZONE_LABEL_MAX {
                return Err(ProfileError::ZoneLabelTooLong {
                    zone: zone.id,
                    len: report_u32(label_len),
                    max: report_u32(ZONE_LABEL_MAX),
                });
            }
            let key = label_key(&zone.label);
            if let Some(other) = earlier.iter().find(|o| label_key(&o.label) == key) {
                return Err(ProfileError::DuplicateZoneLabel {
                    zone: zone.id,
                    other: other.id,
                });
            }
            if !layout_is_valid(&zone.layout) {
                return Err(ProfileError::InvalidLayout { zone: zone.id });
            }
            if let Some(index) = zone.actions.iter().position(|a| !action_supported(a)) {
                return Err(ProfileError::UnsupportedAction {
                    zone: zone.id,
                    index: report_u32(index),
                });
            }
        }
        match &self.calibration {
            CalibrationState::Calibrated { model, .. } => {
                if !model.zone_ids.iter().eq(self.zones.iter().map(|z| &z.id)) {
                    return Err(ProfileError::ModelZonesMismatch);
                }
            }
            CalibrationState::NeedsRecalibration { .. } => {}
        }
        Ok(())
    }

    /// Keeps the first min(old, new) zones (id, label, trigger, actions); added zones take their
    /// names from `default_zones(new)` (a name equal to a kept label gets " 2", " 3", …); every
    /// layout is reset to `default_zones(new)`, so the arrangement never overlaps;
    /// → NeedsRecalibration{ZoneSetChanged} if the count changed.
    pub fn set_zone_count(&mut self, zones: ZoneCount, now: DateTime<Utc>) {
        let defaults = default_zones(zones);
        let old = self.zones.len();
        self.zones.truncate(defaults.len());
        for &(name, default_layout) in defaults.iter().skip(self.zones.len()) {
            let taken: Vec<String> = self.zones.iter().map(|z| label_key(&z.label)).collect();
            self.zones.push(Zone {
                id: ZoneId::new(),
                label: unique_label(name, &taken),
                layout: default_layout,
                trigger: Trigger::Single,
                actions: Vec::new(),
            });
        }
        for (zone, &(_, default_layout)) in self.zones.iter_mut().zip(&defaults) {
            zone.layout = default_layout;
        }
        if self.zones.len() != old {
            self.calibration = CalibrationState::NeedsRecalibration {
                reason: RecalibrationReason::ZoneSetChanged,
            };
        }
        self.updated_at = now;
    }

    /// Applies the patch to one zone; never invalidates the model. An edit that would leave the
    /// profile invalid is refused with its error and changes nothing.
    pub fn edit_zone(
        &mut self,
        zone: ZoneId,
        edit: ZoneEdit,
        now: DateTime<Utc>,
    ) -> Result<(), ProfileError> {
        let index = self
            .zones
            .iter()
            .position(|z| z.id == zone)
            .ok_or(ProfileError::ZoneNotFound { zone })?;
        let previous = self.zones[index].clone();
        let ZoneEdit {
            label,
            layout,
            trigger,
            actions,
        } = edit;
        let target = &mut self.zones[index];
        if let Some(label) = label {
            target.label = label;
        }
        if let Some(layout) = layout {
            target.layout = layout;
        }
        if let Some(trigger) = trigger {
            target.trigger = trigger;
        }
        if let Some(actions) = actions {
            target.actions = actions;
        }
        if let Err(e) = self.validate() {
            self.zones[index] = previous;
            return Err(e);
        }
        self.updated_at = now;
        Ok(())
    }

    /// Binds a Calibrated profile to the same microphone under a new endpoint id (an import from
    /// another PC; XR-UI-16, X01-14, p2 §2.8). NeedsRecalibration → NotCalibrated. Accepted only
    /// when `live.name` equals the bound name and `device.identity.differences(live)` is exactly
    /// [EndpointId]: then only `device.identity.endpoint_id` (= live's) and `updated_at` (= now)
    /// change, and the layout, the model, `calibrated_at`, `environment` and `imported_at` are
    /// kept. Otherwise DeviceDoesNotMatch{fields: those differences without EndpointId,
    /// name_differs}, so the already-bound identity gives {fields: [], name_differs: false}. Every
    /// error leaves the profile unchanged. This is the one place the name is compared; §1.4's
    /// "never compared" refers to `differences()`, which ignores it.
    pub fn rebind_device(
        &mut self,
        live: &DeviceIdentity,
        now: DateTime<Utc>,
    ) -> Result<(), ProfileError> {
        let device = match &mut self.calibration {
            CalibrationState::Calibrated { device, .. } => device,
            CalibrationState::NeedsRecalibration { .. } => return Err(ProfileError::NotCalibrated),
        };
        let differences = device.identity.differences(live);
        let name_differs = device.identity.name != live.name;
        if !name_differs && differences == [IdentityField::EndpointId] {
            device.identity.endpoint_id = live.endpoint_id.clone();
            self.updated_at = now;
            return Ok(());
        }
        Err(ProfileError::DeviceDoesNotMatch {
            fields: differences
                .into_iter()
                .filter(|f| *f != IdentityField::EndpointId)
                .collect(),
            name_differs,
        })
    }

    /// Applied on every load: Calibrated with model.front_end_version != FRONT_END_VERSION
    /// → NeedsRecalibration{FrontEndChanged}. Returns true if it changed anything (the caller
    /// writes back).
    pub fn normalize(&mut self) -> bool {
        let trained = match &self.calibration {
            CalibrationState::Calibrated { model, .. } => model.front_end_version,
            CalibrationState::NeedsRecalibration { .. } => return false,
        };
        if trained == FRONT_END_VERSION {
            return false;
        }
        self.calibration = CalibrationState::NeedsRecalibration {
            reason: RecalibrationReason::FrontEndChanged {
                trained,
                current: FRONT_END_VERSION,
            },
        };
        true
    }
}

/// `Option` fields mean "unchanged". This is a patch type, not a fallback.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZoneEdit {
    pub label: Option<String>,
    pub layout: Option<ZoneLayout>,
    pub trigger: Option<Trigger>,
    pub actions: Option<Vec<ActionSpec>>,
}

/// `PersistenceError::InvalidData` for `file`, its reason naming the typed error.
pub(crate) fn invalid_data(file: &Path, error: &impl fmt::Debug) -> PersistenceError {
    PersistenceError::InvalidData {
        file: file.display().to_string(),
        reason: format!("{error:?}"),
    }
}

/// `Profile::validate`, then the model's own `validate` when Calibrated: every check a profile must
/// pass to be written or loaded.
// SPEC: §1.16.2 validates "Profile, including its model", but ProfileError has no variant that
// carries a ModelError, so the model is validated here, beside `Profile::validate`, and its typed
// error is named in the InvalidData reason the same way.
pub(crate) fn check_profile(p: &Profile, file: &Path) -> Result<(), PersistenceError> {
    p.validate().map_err(|e| invalid_data(file, &e))?;
    match &p.calibration {
        CalibrationState::Calibrated { model, .. } => {
            model.validate().map_err(|e| invalid_data(file, &e))
        }
        CalibrationState::NeedsRecalibration { .. } => Ok(()),
    }
}

/// `validate()` first, so no save path can write an invalid or non-finite value (§0.5,
/// PARITY-19): failure is `PersistenceError::InvalidData{file, reason}`, naming the typed error.
/// Then serde_json::to_string_pretty.
pub fn config_to_json(c: &Config, file: &Path) -> Result<String, PersistenceError> {
    c.validate().map_err(|e| invalid_data(file, &e))?;
    serde_json::to_string_pretty(c).map_err(PersistenceError::Json)
}

pub fn profile_to_json(p: &Profile, file: &Path) -> Result<String, PersistenceError> {
    check_profile(p, file)?;
    serde_json::to_string_pretty(p).map_err(PersistenceError::Json)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::audio::{AudioFormat, CaptureMode, DefaultModeReason};
    use crate::descriptor::{Descriptor, DescriptorLayout, DESCRIPTOR_DIMS};
    use crate::migrate::{load_config, load_profile, Loaded};
    use crate::profile::SystemSound;
    use crate::rng::Rng64;
    use crate::training::{train, TapGroup, TrainingParams, TrainingSet, Workers};
    use crate::units::{ChannelCount, SampleRate, TapIndex, ZoneIndex};
    use serde_json::{json, Value};
    use std::num::NonZeroU8;
    use std::sync::atomic::AtomicBool;
    use std::sync::OnceLock;
    use uuid::Uuid;

    pub(crate) fn ts(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s)
            .expect("a valid RFC 3339 time")
            .with_timezone(&Utc)
    }

    pub(crate) fn file() -> &'static Path {
        Path::new("profile.json")
    }

    pub(crate) fn count(n: u8) -> ZoneCount {
        ZoneCount::new(n).expect("an allowed zone count")
    }

    fn fixture_zone_ids() -> Vec<ZoneId> {
        (0..2u128)
            .map(|i| ZoneId::from_uuid(Uuid::from_u128(0xB300 + i)))
            .collect()
    }

    /// A 2-zone Mono model trained on Gaussian descriptor clusters, trained once per test binary.
    fn trained_model() -> TrainedModel {
        static MODEL: OnceLock<TrainedModel> = OnceLock::new();
        MODEL
            .get_or_init(|| {
                let layout = DescriptorLayout::Mono { a: 0 };
                let d = layout.dims();
                let mut rng = Rng64::new(0xB3);
                let means: Vec<Vec<f64>> = (0..2)
                    .map(|_| (0..d).map(|_| 3.0 * rng.normal()).collect())
                    .collect();
                let mut taps = Vec::new();
                for (zone, mean) in means.iter().enumerate() {
                    for _ in 0..6 {
                        let centre: Vec<f64> = mean.iter().map(|m| m + rng.normal()).collect();
                        let variants = (0..4)
                            .map(|v| {
                                let mut values = [0.0f32; DESCRIPTOR_DIMS];
                                for (x, c) in values.iter_mut().zip(&centre) {
                                    let jitter = if v == 0 { 0.0 } else { 0.7 * rng.normal() };
                                    *x = (c + jitter) as f32;
                                }
                                Descriptor { layout, values }
                            })
                            .collect();
                        taps.push(TapGroup {
                            zone: ZoneIndex(zone as u16),
                            tap: TapIndex(taps.len() as u32),
                            variants,
                        });
                    }
                }
                let set = TrainingSet {
                    layout,
                    rate: SampleRate::new(48_000).expect("48 kHz is valid"),
                    zone_ids: fixture_zone_ids(),
                    seed: 0xB3,
                    taps,
                    negatives: Vec::new(),
                };
                let params = TrainingParams {
                    workers: Workers::Exactly {
                        n: NonZeroU8::new(1).expect("one worker"),
                    },
                    ..TrainingParams::DEFAULT
                };
                train(&set, &params, &AtomicBool::new(false), &mut |_| {})
                    .expect("the fixture trains")
            })
            .clone()
    }

    pub(crate) fn bound_identity() -> DeviceIdentity {
        DeviceIdentity {
            endpoint_id: "{0.0.1.00000000}.{b3-microphone}".into(),
            name: "USB Microphone".into(),
            format: AudioFormat {
                rate: SampleRate::new(48_000).expect("48 kHz is valid"),
                channels: ChannelCount::new(1).expect("mono is valid"),
            },
            capture_mode: CaptureMode::Raw,
        }
    }

    /// A Calibrated 2-zone profile (with a trained model) that was imported from another PC.
    pub(crate) fn calibrated_profile() -> Profile {
        let mut p = Profile::new("Home desk".into(), count(2), ts("2026-11-01T09:00:00Z"))
            .expect("a valid new profile");
        p.id = ProfileId::from_uuid(Uuid::from_u128(0xB3));
        for (zone, id) in p.zones.iter_mut().zip(fixture_zone_ids()) {
            zone.id = id;
        }
        p.zones[0].actions = vec![
            ActionSpec::Speak {
                text: "left".into(),
            },
            ActionSpec::SendKeystroke {
                combo: KeyCombo {
                    modifiers: vec![Modifier::Ctrl, Modifier::Shift],
                    key: "M".into(),
                },
            },
        ];
        p.zones[1].trigger = Trigger::Double;
        p.zones[1].actions = vec![ActionSpec::PlaySound {
            sound: SystemSound::Success,
        }];
        p.calibration = CalibrationState::Calibrated {
            device: DeviceFingerprint {
                identity: bound_identity(),
                layout: ChannelLayout::all_real(ChannelCount::new(1).expect("mono is valid")),
            },
            model: trained_model(),
            calibrated_at: ts("2026-11-01T09:30:00Z"),
            environment: EnvironmentQuality::Quiet,
        };
        p.imported_at = Some(ts("2026-11-02T08:00:00Z"));
        p.validate().expect("the fixture is valid");
        p
    }

    pub(crate) fn model_mut(p: &mut Profile) -> &mut TrainedModel {
        match &mut p.calibration {
            CalibrationState::Calibrated { model, .. } => model,
            CalibrationState::NeedsRecalibration { reason } => {
                panic!("the fixture is calibrated, got {reason:?}")
            }
        }
    }

    fn current<T: fmt::Debug>(loaded: Loaded<T>) -> T {
        match loaded {
            Loaded::Current(value) => value,
            other => panic!("expected Current, got {other:?}"),
        }
    }

    fn json_with(text: &str, pointer: &str, key: &str, value: Value) -> String {
        let mut v: Value = serde_json::from_str(text).expect("valid JSON");
        v.pointer_mut(pointer)
            .and_then(Value::as_object_mut)
            .expect("an object at the pointer")
            .insert(key.to_string(), value);
        v.to_string()
    }

    fn overlaps(a: &ZoneLayout, b: &ZoneLayout) -> bool {
        a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height
    }

    /// `a` and `b` have the same structure and the same values, except that two numbers may differ
    /// by the rounding error of serde_json's default float reader (a few ulps).
    fn assert_json_close(a: &Value, b: &Value, path: &str) {
        match (a, b) {
            (Value::Number(x), Value::Number(y)) => {
                if x.is_f64() || y.is_f64() {
                    let (x, y) = (x.as_f64().expect("a number"), y.as_f64().expect("a number"));
                    let scale = x.abs().max(y.abs());
                    assert!((x - y).abs() <= 1e-15 * scale, "{path}: {x} vs {y}");
                } else {
                    assert_eq!(x, y, "{path}");
                }
            }
            (Value::Array(x), Value::Array(y)) => {
                assert_eq!(x.len(), y.len(), "{path}");
                for (i, (u, v)) in x.iter().zip(y).enumerate() {
                    assert_json_close(u, v, &format!("{path}/{i}"));
                }
            }
            (Value::Object(x), Value::Object(y)) => {
                assert!(x.keys().eq(y.keys()), "{path}");
                for (k, u) in x {
                    assert_json_close(u, &y[k], &format!("{path}/{k}"));
                }
            }
            (x, y) => assert_eq!(x, y, "{path}"),
        }
    }

    /// Serialize, parse with `load_profile`, serialize again: the two texts.
    fn profile_round_trip(p: &Profile) -> (String, String) {
        let first = profile_to_json(p, file()).expect("a valid profile serializes");
        let parsed = current(load_profile(&first, file()).expect("loads"));
        let second = profile_to_json(&parsed, file()).expect("serializes again");
        (first, second)
    }

    #[test]
    fn st4_v2_round_trip_is_text_identical() {
        let fresh =
            Profile::new("Fresh".into(), count(8), ts("2026-11-03T10:00:00Z")).expect("valid");
        let (first, second) = profile_round_trip(&fresh);
        assert_eq!(first, second);
        assert_eq!(current(load_profile(&first, file()).expect("loads")), fresh);

        // SPEC: ST4 asks for text identity. Every config field and every profile field outside the
        // model round-trips text-identically, but the model's f64s cannot: the workspace builds
        // serde_json without its `float_roundtrip` feature, whose default reader rounds about one in
        // ten 17-digit values by an ulp (measured: 217 of 2065 lines of this fixture), and this
        // package may not edit a manifest. The model's numbers are compared within that error.
        let calibrated = calibrated_profile();
        let (first, second) = profile_round_trip(&calibrated);
        let mut a: Value = serde_json::from_str(&first).expect("valid JSON");
        let mut b: Value = serde_json::from_str(&second).expect("valid JSON");
        let take_model = |v: &mut Value| {
            v["calibration"]
                .as_object_mut()
                .and_then(|c| c.remove("model"))
                .expect("a calibrated profile has a model")
        };
        let (model_a, model_b) = (take_model(&mut a), take_model(&mut b));
        assert_eq!(a, b, "everything outside the model is identical");
        assert_json_close(&model_a, &model_b, "model");

        let config = Config {
            active_profile_id: Some(ProfileId::from_uuid(Uuid::from_u128(0xB3))),
            theme: Theme::Grape,
            dismissed_notices: vec![NoticeId::FirstCloseToTray, NoticeId::UnattendedActions],
            window_placement: Some(WindowPlacement {
                x: -1920,
                y: 40,
                w: 1280,
                h: 800,
                maximized: true,
            }),
            text_zoom: 125,
            ..Config::initial()
        };
        for c in [Config::initial(), config] {
            let first = config_to_json(&c, file()).expect("a valid config serializes");
            let parsed = current(load_config(&first, file()).expect("loads"));
            assert_eq!(parsed, c);
            assert_eq!(config_to_json(&parsed, file()).expect("serializes"), first);
        }
    }

    #[test]
    fn st5_identity_differences() {
        let bound = bound_identity();
        assert_eq!(bound.differences(&bound), []);

        let mut rate = bound.clone();
        rate.format.rate = SampleRate::new(44_100).expect("valid");
        assert_eq!(bound.differences(&rate), [IdentityField::SampleRate]);

        let mut name = bound.clone();
        name.name = "Another name".into();
        assert_eq!(bound.differences(&name), []);

        let mut mode = bound.clone();
        mode.capture_mode = CaptureMode::Default {
            reason: DefaultModeReason::RawNotSupported,
        };
        assert_eq!(bound.differences(&mode), [IdentityField::CaptureMode]);

        let mut other_reason = mode.clone();
        other_reason.capture_mode = CaptureMode::Default {
            reason: DefaultModeReason::RawRejected {
                hresult: -2_004_287_478,
            },
        };
        assert_eq!(mode.differences(&other_reason), []);
    }

    #[test]
    fn st6_edits_normalize_and_unknown_fields() {
        let later = ts("2026-11-05T12:00:00Z");

        // set_zone_count invalidates the model.
        let mut p = calibrated_profile();
        p.set_zone_count(count(4), later);
        assert_eq!(
            p.calibration,
            CalibrationState::NeedsRecalibration {
                reason: RecalibrationReason::ZoneSetChanged
            }
        );
        assert_eq!(p.updated_at, later);
        p.validate().expect("still valid");

        // edit_zone never invalidates the model.
        let mut p = calibrated_profile();
        let calibration = p.calibration.clone();
        let zone = p.zones[0].id;
        let new_layout = layout(0.05, 0.10, 0.25, 0.30);
        p.edit_zone(
            zone,
            ZoneEdit {
                label: Some("Mouse pad".into()),
                layout: Some(new_layout),
                trigger: Some(Trigger::Double),
                actions: Some(Vec::new()),
            },
            later,
        )
        .expect("a valid edit");
        assert_eq!(p.calibration, calibration);
        assert_eq!(p.updated_at, later);
        assert_eq!(p.zones[0].label, "Mouse pad");
        assert_eq!(p.zones[0].layout, new_layout);
        assert_eq!(p.zones[0].trigger, Trigger::Double);
        assert!(p.zones[0].actions.is_empty());

        // An invalid edit is refused and changes nothing; an unknown zone is named.
        let before = p.clone();
        let blank = ZoneEdit {
            label: Some("   ".into()),
            layout: None,
            trigger: None,
            actions: None,
        };
        assert_eq!(
            p.edit_zone(zone, blank.clone(), ts("2026-11-06T00:00:00Z")),
            Err(ProfileError::EmptyZoneLabel { zone })
        );
        assert_eq!(p, before);
        let missing = ZoneId::from_uuid(Uuid::from_u128(0xDEAD));
        assert_eq!(
            p.edit_zone(missing, blank, later),
            Err(ProfileError::ZoneNotFound { zone: missing })
        );

        // normalize: a model from another front end needs recalibration; a current one is kept.
        let mut p = calibrated_profile();
        assert!(!p.normalize());
        let stale_version = FRONT_END_VERSION - 1;
        model_mut(&mut p).front_end_version = stale_version;
        assert!(p.normalize());
        assert_eq!(
            p.calibration,
            CalibrationState::NeedsRecalibration {
                reason: RecalibrationReason::FrontEndChanged {
                    trained: stale_version,
                    current: FRONT_END_VERSION
                }
            }
        );
        assert!(!p.normalize());

        // Unknown fields fail to parse, at every level of a profile and a config.
        let text = profile_to_json(&calibrated_profile(), file()).expect("serializes");
        for pointer in ["", "/zones/0", "/calibration", "/calibration/device"] {
            let edited = json_with(&text, pointer, "extra", json!(1));
            assert!(
                matches!(
                    load_profile(&edited, file()),
                    Err(PersistenceError::Json(_))
                ),
                "an unknown field at '{pointer}' must fail to parse"
            );
        }
        let mut stale = calibrated_profile();
        model_mut(&mut stale).front_end_version = stale_version;
        assert!(stale.normalize());
        let text = profile_to_json(&stale, file()).expect("serializes");
        // A struct variant's unknown field is refused; serde's internally tagged unit variants
        // (NewProfile, ZoneSetChanged) have no fields to check.
        let edited = json_with(&text, "/calibration/reason", "extra", json!(1));
        assert!(matches!(
            load_profile(&edited, file()),
            Err(PersistenceError::Json(_))
        ));
        let config = Config {
            window_placement: Some(WindowPlacement {
                x: 0,
                y: 0,
                w: 800,
                h: 600,
                maximized: false,
            }),
            ..Config::initial()
        };
        let text = config_to_json(&config, file()).expect("serializes");
        for pointer in ["", "/window_placement"] {
            let edited = json_with(&text, pointer, "extra", json!(true));
            assert!(
                matches!(load_config(&edited, file()), Err(PersistenceError::Json(_))),
                "an unknown config field at '{pointer}' must fail to parse"
            );
        }
    }

    #[test]
    fn st8_limits() {
        let now = ts("2026-11-05T12:00:00Z");
        let fresh = Profile::new("Desk".into(), count(2), now).expect("valid");
        let (a, b) = (fresh.zones[0].id, fresh.zones[1].id);
        let with = |edit: &dyn Fn(&mut Profile)| {
            let mut p = fresh.clone();
            edit(&mut p);
            p.validate()
        };

        // Labels: 32 chars pass, 33 do not; equal after trimming and lowercasing is a duplicate.
        assert_eq!(with(&|p| p.zones[0].label = "x".repeat(32)), Ok(()));
        assert_eq!(
            with(&|p| p.zones[0].label = "x".repeat(33)),
            Err(ProfileError::ZoneLabelTooLong {
                zone: a,
                len: 33,
                max: 32
            })
        );
        assert_eq!(
            with(&|p| p.zones[0].label = format!("  {}  ", "é".repeat(32))),
            Ok(())
        );
        assert_eq!(
            with(&|p| p.zones[0].label = " \t ".into()),
            Err(ProfileError::EmptyZoneLabel { zone: a })
        );
        assert_eq!(
            with(&|p| {
                p.zones[0].label = "Desk".into();
                p.zones[1].label = "  desk ".into();
            }),
            Err(ProfileError::DuplicateZoneLabel { zone: b, other: a })
        );

        // Names.
        assert_eq!(with(&|p| p.name = "n".repeat(60)), Ok(()));
        assert_eq!(
            with(&|p| p.name = "n".repeat(61)),
            Err(ProfileError::NameTooLong { len: 61, max: 60 })
        );
        assert_eq!(
            with(&|p| p.name = "  ".into()),
            Err(ProfileError::EmptyName)
        );

        // Layouts: NaN, negative size and off-canvas are invalid; flush with an edge is fine.
        let bad_layouts = [
            layout(f32::NAN, 0.1, 0.2, 0.2),
            layout(0.1, 0.1, f32::INFINITY, 0.2),
            layout(0.1, 0.1, -0.2, 0.2),
            layout(0.1, 0.1, 0.2, 0.0),
            layout(0.9, 0.1, 0.2, 0.2),
            layout(0.1, -0.1, 0.2, 0.2),
            layout(0.1, 0.9, 0.2, 0.2),
            layout(1.5, 0.1, 0.2, 0.2),
        ];
        for bad in bad_layouts {
            assert_eq!(
                with(&|p| p.zones[0].layout = bad),
                Err(ProfileError::InvalidLayout { zone: a }),
                "{bad:?}"
            );
        }
        assert_eq!(
            with(&|p| p.zones[0].layout = layout(0.0, 0.0, 1.0, 1.0)),
            Ok(())
        );
        assert_eq!(
            with(&|p| p.zones[0].layout = layout(0.7, 0.0, 0.3, 0.2)),
            Ok(())
        );

        // The ported v1 `save_rejects_a_non_finite_zone_layout`: a NaN layout is never written.
        let mut nan = fresh.clone();
        nan.zones[0].layout.x = f32::NAN;
        match profile_to_json(&nan, file()) {
            Err(PersistenceError::InvalidData { file, reason }) => {
                assert_eq!(file, "profile.json");
                assert!(reason.contains("InvalidLayout"), "{reason}");
            }
            other => panic!("expected InvalidData, got {other:?}"),
        }

        // Actions, ids, counts and the model's zones.
        assert_eq!(
            with(&|p| p.zones[1].actions = vec![
                ActionSpec::Speak { text: "hi".into() },
                ActionSpec::VisualOnly
            ]),
            Err(ProfileError::UnsupportedAction { zone: b, index: 1 })
        );
        assert_eq!(
            with(&|p| p.zones[1].id = a),
            Err(ProfileError::DuplicateZoneId { zone: a })
        );
        assert_eq!(
            with(&|p| {
                p.zones.pop();
            }),
            Err(ProfileError::InvalidZoneCount { n: 1 })
        );
        let mut swapped = calibrated_profile();
        swapped.zones.swap(0, 1);
        assert_eq!(swapped.validate(), Err(ProfileError::ModelZonesMismatch));

        // A hotkey needs Ctrl, Alt or Win; Shift alone is not enough.
        let hotkey = |modifiers: Vec<Modifier>| Config {
            global_hotkey: Some(KeyCombo {
                modifiers,
                key: "A".into(),
            }),
            ..Config::initial()
        };
        assert_eq!(
            hotkey(vec![Modifier::Shift]).validate(),
            Err(ConfigError::HotkeyWithoutModifier)
        );
        assert_eq!(
            hotkey(Vec::new()).validate(),
            Err(ConfigError::HotkeyWithoutModifier)
        );
        for m in [Modifier::Ctrl, Modifier::Alt, Modifier::Win] {
            assert_eq!(hotkey(vec![Modifier::Shift, m]).validate(), Ok(()));
        }
        assert!(matches!(
            config_to_json(&hotkey(vec![Modifier::Shift]), file()),
            Err(PersistenceError::InvalidData { .. })
        ));

        // The other config limits.
        let config = |edit: &dyn Fn(&mut Config)| {
            let mut c = Config::initial();
            edit(&mut c);
            c.validate()
        };
        assert_eq!(config(&|c| c.global_hotkey = None), Ok(()));
        assert_eq!(
            config(&|c| c.eval_prompts_per_zone = 0),
            Err(ConfigError::ZeroEvalPrompts)
        );
        let placement = |w, h| WindowPlacement {
            x: 10,
            y: 10,
            w,
            h,
            maximized: false,
        };
        assert_eq!(
            config(&|c| c.window_placement = Some(placement(0, 600))),
            Err(ConfigError::EmptyWindowPlacement)
        );
        assert_eq!(
            config(&|c| c.window_placement = Some(placement(800, -1))),
            Err(ConfigError::EmptyWindowPlacement)
        );
        assert_eq!(
            config(&|c| c.dismissed_notices = vec![
                NoticeId::UnattendedActions,
                NoticeId::FirstCloseToTray,
                NoticeId::UnattendedActions
            ]),
            Err(ConfigError::DuplicateNotice {
                notice: NoticeId::UnattendedActions
            })
        );
    }

    #[test]
    fn st9_default_zones_and_zone_count() {
        // The §1.16.1 table, written out independently of `default_zones`.
        let l = |name: &'static str, x: f32, y: f32, w: f32, h: f32| (name, layout(x, y, w, h));
        let four = [
            l("Left back", 0.03, 0.08, 0.29, 0.40),
            l("Right back", 0.68, 0.08, 0.29, 0.40),
            l("Left front", 0.03, 0.52, 0.29, 0.40),
            l("Right front", 0.68, 0.52, 0.29, 0.40),
        ];
        let near = [
            l("Near left", 0.35, 0.64, 0.145, 0.28),
            l("Near right", 0.505, 0.64, 0.145, 0.28),
        ];
        let table: [(u8, Vec<(&str, ZoneLayout)>); 4] = [
            (
                2,
                vec![
                    l("Left", 0.03, 0.08, 0.29, 0.84),
                    l("Right", 0.68, 0.08, 0.29, 0.84),
                ],
            ),
            (4, four.to_vec()),
            (6, four.iter().chain(&near).copied().collect()),
            (
                8,
                vec![
                    l("Left back", 0.03, 0.08, 0.29, 0.2533),
                    l("Right back", 0.68, 0.08, 0.29, 0.2533),
                    l("Left middle", 0.03, 0.3733, 0.29, 0.2533),
                    l("Right middle", 0.68, 0.3733, 0.29, 0.2533),
                    l("Left front", 0.03, 0.6667, 0.29, 0.2533),
                    l("Right front", 0.68, 0.6667, 0.29, 0.2533),
                    near[0],
                    near[1],
                ],
            ),
        ];
        let laptop = layout(0.35, 0.08, 0.30, 0.50);
        let now = ts("2026-11-05T12:00:00Z");
        for (n, expected) in &table {
            assert_eq!(&default_zones(count(*n)), expected, "n = {n}");
            let p = Profile::new("Desk".into(), count(*n), now).expect("valid");
            assert_eq!(p.zones.len(), usize::from(*n));
            for (zone, (name, zl)) in p.zones.iter().zip(expected) {
                assert_eq!((zone.label.as_str(), zone.layout), (*name, *zl));
                assert_eq!(zone.trigger, Trigger::Single);
                assert!(zone.actions.is_empty());
                assert!(layout_is_valid(&zone.layout));
                assert!(
                    !overlaps(&zone.layout, &laptop),
                    "{name} overlaps the laptop"
                );
            }
            for (i, a) in expected.iter().enumerate() {
                for b in &expected[i + 1..] {
                    assert!(!overlaps(&a.1, &b.1), "{} overlaps {}", a.0, b.0);
                }
            }
            assert_eq!(
                p.calibration,
                CalibrationState::NeedsRecalibration {
                    reason: RecalibrationReason::NewProfile
                }
            );
            assert_eq!(
                (p.created_at, p.updated_at, p.imported_at),
                (now, now, None)
            );
        }

        // ZoneCount admits exactly ALLOWED, in Rust and in JSON.
        for n in 0..=u8::MAX {
            assert_eq!(
                ZoneCount::new(n).is_ok(),
                ZoneCount::ALLOWED.contains(&n),
                "{n}"
            );
        }
        assert_eq!(
            ZoneCount::new(3),
            Err(ProfileError::InvalidZoneCount { n: 3 })
        );
        assert!(serde_json::from_str::<ZoneCount>("3").is_err());
        assert_eq!(
            serde_json::from_str::<ZoneCount>("6").expect("6 is allowed"),
            count(6)
        );
        assert_eq!(serde_json::to_string(&count(8)).expect("serializes"), "8");

        // set_zone_count 4 → 6 keeps zones 1-4 and resets every layout to the 6-zone table.
        let mut p = Profile::new("Desk".into(), count(4), now).expect("valid");
        p.zones[0].label = "Mouse".into();
        p.zones[0].layout = layout(0.1, 0.1, 0.1, 0.1);
        p.zones[1].trigger = Trigger::Double;
        p.zones[2].actions = vec![ActionSpec::OpenUrl {
            url: "https://example.org".into(),
        }];
        let kept: Vec<(ZoneId, String, Trigger, Vec<ActionSpec>)> = p
            .zones
            .iter()
            .map(|z| (z.id, z.label.clone(), z.trigger, z.actions.clone()))
            .collect();
        let later = ts("2026-11-06T12:00:00Z");
        p.set_zone_count(count(6), later);
        assert_eq!(p.zones.len(), 6);
        for (zone, (id, label, trigger, actions)) in p.zones.iter().zip(&kept) {
            assert_eq!(
                (&zone.id, &zone.label, &zone.trigger, &zone.actions),
                (id, label, trigger, actions)
            );
        }
        for (zone, (name, zl)) in p.zones.iter().zip(&table[2].1).skip(4) {
            assert_eq!(
                (zone.label.as_str(), zone.trigger),
                (*name, Trigger::Single)
            );
            assert!(zone.actions.is_empty());
            assert!(
                !kept.iter().any(|k| k.0 == zone.id),
                "a new zone gets a new id"
            );
            assert_eq!(zone.layout, *zl);
        }
        for (zone, (_, zl)) in p.zones.iter().zip(&table[2].1) {
            assert_eq!(zone.layout, *zl);
        }
        assert_eq!(p.updated_at, later);
        p.validate().expect("valid");

        // A default name equal to a kept label is numbered; shrinking keeps the first zones.
        let mut p = Profile::new("Desk".into(), count(4), now).expect("valid");
        p.zones[0].label = "near LEFT".into();
        p.set_zone_count(count(6), later);
        assert_eq!(p.zones[4].label, "Near left 2");
        p.validate().expect("valid");
        let first_two: Vec<ZoneId> = p.zones[..2].iter().map(|z| z.id).collect();
        p.set_zone_count(count(2), later);
        assert_eq!(p.zones.iter().map(|z| z.id).collect::<Vec<_>>(), first_two);
        assert_eq!(p.zones[0].layout, table[0].1[0].1);
        p.validate().expect("valid");
    }

    #[test]
    fn st11_rebind_device() {
        let now = ts("2026-11-07T12:00:00Z");
        let original = calibrated_profile();
        let before: Value = serde_json::to_value(&original).expect("serializes");
        let bound = bound_identity();

        // Only the endpoint id differs: rebound, and only endpoint_id and updated_at change.
        let mut moved = bound.clone();
        moved.endpoint_id = "{0.0.1.00000000}.{other-pc}".into();
        let mut p = original.clone();
        p.rebind_device(&moved, now)
            .expect("same microphone, new endpoint id");
        let after: Value = serde_json::to_value(&p).expect("serializes");
        let mut expected = before.clone();
        expected["calibration"]["device"]["identity"]["endpoint_id"] = json!(moved.endpoint_id);
        expected["updated_at"] = serde_json::to_value(now).expect("serializes");
        assert_eq!(after, expected);
        assert_eq!(p.imported_at, original.imported_at);

        // Anything else is refused and leaves the profile unchanged.
        let mut rate = moved.clone();
        rate.format.rate = SampleRate::new(44_100).expect("valid");
        let mut renamed = moved.clone();
        renamed.name = "Other microphone".into();
        let cases = [
            (rate, vec![IdentityField::SampleRate], false),
            (renamed, Vec::new(), true),
            (bound.clone(), Vec::new(), false),
        ];
        for (live, fields, name_differs) in cases {
            let mut p = original.clone();
            assert_eq!(
                p.rebind_device(&live, now),
                Err(ProfileError::DeviceDoesNotMatch {
                    fields,
                    name_differs
                })
            );
            assert_eq!(serde_json::to_value(&p).expect("serializes"), before);
        }

        // A profile that needs recalibration has nothing to rebind.
        let mut fresh = Profile::new("Fresh".into(), count(2), now).expect("valid");
        let fresh_before = fresh.clone();
        assert_eq!(
            fresh.rebind_device(&moved, now),
            Err(ProfileError::NotCalibrated)
        );
        assert_eq!(fresh, fresh_before);

        // imported_at: None from `new` and from the v1 migration; Some(t) round-trips.
        assert_eq!(fresh.imported_at, None);
        let migrated = match load_profile(crate::migrate::tests::GOLDEN_V1_PROFILE, file()) {
            Ok(Loaded::Migrated { value, .. }) => value,
            other => panic!("expected Migrated, got {other:?}"),
        };
        assert_eq!(migrated.imported_at, None);
        let (text, _) = profile_round_trip(&original);
        assert!(
            text.contains("\"imported_at\": \"2026-11-02T08:00:00Z\""),
            "{text}"
        );
        let parsed = current(load_profile(&text, file()).expect("loads"));
        assert_eq!(parsed.imported_at, original.imported_at);
        // Text identity with no model in the file (see ST4 for the model's floats).
        fresh.imported_at = Some(ts("2026-11-02T08:00:00.123456789Z"));
        let (first, second) = profile_round_trip(&fresh);
        assert_eq!(first, second);
        assert!(first.contains("\"imported_at\": \"2026-11-02T08:00:00.123456789Z\""));
    }

    #[test]
    fn st12_text_zoom_and_notices() {
        assert_eq!(Config::initial().text_zoom, 100);
        let migrated = match load_config(crate::migrate::tests::GOLDEN_V1_CONFIG, file()) {
            Ok(Loaded::Migrated { value, .. }) => value,
            other => panic!("expected Migrated, got {other:?}"),
        };
        assert_eq!(migrated.text_zoom, 100);

        for step in TEXT_ZOOM_STEPS {
            let c = Config {
                text_zoom: step,
                ..Config::initial()
            };
            assert_eq!(c.validate(), Ok(()), "{step}");
        }
        let odd = Config {
            text_zoom: 105,
            ..Config::initial()
        };
        assert_eq!(
            odd.validate(),
            Err(ConfigError::InvalidTextZoom { value: 105 })
        );
        match config_to_json(&odd, file()) {
            Err(PersistenceError::InvalidData { reason, .. }) => {
                assert!(reason.contains("InvalidTextZoom"), "{reason}");
            }
            other => panic!("expected InvalidData, got {other:?}"),
        }
        let text = config_to_json(&Config::initial(), file()).expect("serializes");
        let mut v: Value = serde_json::from_str(&text).expect("valid JSON");
        v["text_zoom"] = json!(105);
        match load_config(&v.to_string(), file()) {
            Err(PersistenceError::InvalidData { reason, .. }) => {
                assert!(reason.contains("InvalidTextZoom"), "{reason}");
                assert!(reason.contains("105"), "{reason}");
            }
            other => panic!("expected InvalidData, got {other:?}"),
        }

        // NoticeId: appended only, so UnattendedActions sorts last, and every value round-trips.
        assert!(NoticeId::FirstCloseToTray < NoticeId::UpgradeV2TypingFilter);
        assert!(NoticeId::UpgradeV2TypingFilter < NoticeId::UnattendedActions);
        for notice in [
            NoticeId::FirstCloseToTray,
            NoticeId::UpgradeV2TypingFilter,
            NoticeId::UnattendedActions,
        ] {
            let text = serde_json::to_string(&notice).expect("serializes");
            assert_eq!(
                serde_json::from_str::<NoticeId>(&text).expect("parses"),
                notice
            );
        }
        assert_eq!(
            serde_json::to_string(&NoticeId::UnattendedActions).expect("serializes"),
            r#"{"type":"UnattendedActions"}"#
        );
    }
}
