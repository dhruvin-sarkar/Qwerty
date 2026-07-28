//! Persisted data model and its (de)serialization — the single Rust mirror of
//! `DATA_MODEL.md`. If this file and that document disagree, the document is
//! right and this is a bug.
//!
//! Policy (`ARCHITECTURE.md` → Persistence): every persisted struct leads with
//! `schema_version: u32`. On load, a version other than the current one is a
//! *named refusal* ([`PersistenceError::SchemaMismatch`]), never a silent
//! coercion; corrupt or truncated JSON is a *named error*
//! ([`PersistenceError::Json`]), never a panic.

use chrono::{DateTime, Utc};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::classifier::ClassifierParams;

/// Current schema versions. Bumping either requires a migration or an explicit
/// refusal-to-load (`DATA_MODEL.md`).
pub const CONFIG_SCHEMA_VERSION: u32 = 1;
pub const PROFILE_SCHEMA_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Identifiers (UUID v4 newtypes, serialized transparently as strings)
// ---------------------------------------------------------------------------

macro_rules! uuid_newtype {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub struct $name(Uuid);

        impl $name {
            /// Generate a fresh random (v4) id.
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }
            /// Wrap an existing UUID (used by tests and migrations).
            pub fn from_uuid(u: Uuid) -> Self {
                Self(u)
            }
            /// The underlying UUID.
            pub fn as_uuid(&self) -> Uuid {
                self.0
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

uuid_newtype!(
    /// Stable identity of a desk profile.
    ProfileId
);
uuid_newtype!(
    /// Stable identity of a zone, unchanged when *other* zones are recalibrated.
    ZoneId
);

// ---------------------------------------------------------------------------
// Enums (serialized as their PascalCase variant names — see DATA_MODEL.md
// "Naming and versioning conventions": no rename_all, so the JSON greps
// directly against the Rust source)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Theme {
    System,
    Daylight,
    Midnight,
    Aurora,
    HighContrast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotificationVerbosity {
    Silent,
    Minimal,
    Normal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensingMode {
    Passive,
    Active,
    Hybrid,
}

/// A keyboard modifier for a [`KeyCombo`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Modifier {
    Ctrl,
    Shift,
    Alt,
    Win,
}

/// A modifier+key chord, e.g. `Ctrl+Shift+Q`. `key` is a platform-neutral key
/// name resolved by the platform layer when synthesized.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyCombo {
    pub modifiers: Vec<Modifier>,
    pub key: String,
}

/// A semantic system sound. The concrete Windows sound is chosen by the
/// platform layer; this stays platform-neutral in `qwerty-core`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SystemSound {
    Default,
    Success,
    Error,
    Notification,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScreenshotMode {
    FullScreen,
    RegionSelect,
}

/// One action in a zone's macro chain. Tagged by `type` in JSON — the single
/// representation of an action (`DATA_MODEL.md`). A new action type is a new
/// variant here, reviewed against `ARCHITECTURE.md` → out of scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ActionSpec {
    OpenApp { path: String },
    OpenPath { path: String },
    OpenUrl { url: String },
    RunCommand { command: String, args: Vec<String> },
    CopyToClipboard { text: String },
    Speak { text: String },
    PlaySound { sound: SystemSound },
    ShowToast { title: String, body: String },
    SendKeystroke { combo: KeyCombo },
    Screenshot { mode: ScreenshotMode },
    VisualOnly,
}

impl ActionSpec {
    /// A short reason this action is not yet runnable because a required field
    /// is blank, or `None` when it is complete. This is non-blocking editor
    /// guidance: the action editor shows it so a half-filled action reads as
    /// visibly incomplete *before* a tap fires it, instead of the action
    /// silently no-op'ing or failing at dispatch. It is deliberately **not**
    /// enforced by [`Profile::validate`] — a half-typed action is a normal
    /// mid-edit state, not a corrupt profile. Exhaustive, so a new variant must
    /// decide whether it has a required field.
    pub fn incomplete_reason(&self) -> Option<&'static str> {
        let blank = |s: &str| s.trim().is_empty();
        match self {
            ActionSpec::OpenApp { path } => blank(path).then_some("needs an application path"),
            ActionSpec::OpenPath { path } => blank(path).then_some("needs a file or folder path"),
            ActionSpec::OpenUrl { url } => blank(url).then_some("needs a URL"),
            ActionSpec::RunCommand { command, .. } => blank(command).then_some("needs a command"),
            ActionSpec::CopyToClipboard { text } => blank(text).then_some("needs text to copy"),
            ActionSpec::Speak { text } => blank(text).then_some("needs text to speak"),
            ActionSpec::ShowToast { title, body } => {
                (blank(title) && blank(body)).then_some("needs a title or body")
            }
            ActionSpec::SendKeystroke { combo } => blank(&combo.key).then_some("needs a key"),
            ActionSpec::PlaySound { .. }
            | ActionSpec::Screenshot { .. }
            | ActionSpec::VisualOnly => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Config (config.json)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub schema_version: u32,
    pub active_profile_id: Option<ProfileId>,
    pub theme: Theme,
    pub start_with_windows: bool,
    pub minimize_to_tray_on_close: bool,
    pub notification_verbosity: NotificationVerbosity,
    pub global_hotkey: Option<KeyCombo>,
    pub debug_capture_enabled: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            schema_version: CONFIG_SCHEMA_VERSION,
            active_profile_id: None,
            theme: Theme::System,
            start_with_windows: false,
            minimize_to_tray_on_close: true,
            notification_verbosity: NotificationVerbosity::Normal,
            global_hotkey: None,
            debug_capture_enabled: false,
        }
    }
}

impl Config {
    /// Serialize to pretty JSON (the on-disk form).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("Config serializes")
    }

    /// Parse from JSON with the schema-version policy applied. `file` names the
    /// source for error messages.
    pub fn from_json(s: &str, file: &str) -> Result<Self, PersistenceError> {
        parse_versioned(s, CONFIG_SCHEMA_VERSION, file)
    }

    /// Load from a file path, applying the schema-version policy.
    pub fn load(path: &std::path::Path) -> Result<Self, PersistenceError> {
        let s = std::fs::read_to_string(path).map_err(PersistenceError::Io)?;
        Self::from_json(&s, &path.display().to_string())
    }

    /// Write to a file path as pretty JSON.
    pub fn save(&self, path: &std::path::Path) -> Result<(), PersistenceError> {
        write_atomic(path, &self.to_json()).map_err(PersistenceError::Io)
    }
}

// ---------------------------------------------------------------------------
// Profile (Profiles/<id>.json)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceFingerprint {
    pub device_name: String,
    pub capability_hash: String,
}

impl DeviceFingerprint {
    /// Build a fingerprint from a live device's name and negotiated sample rate.
    /// The single place the `capability_hash` shape is defined, so calibration
    /// (which stamps it into the profile) and the live device-change check
    /// (which compares against it) can never drift apart. `ARCHITECTURE.md` →
    /// Microphone identity notes the full channel/format fingerprint is a
    /// follow-up; the sample rate is the stand-in until then.
    pub fn from_device(device_name: impl Into<String>, sample_rate: u32) -> Self {
        Self {
            device_name: device_name.into(),
            capability_hash: format!("sr{sample_rate}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ZoneLayout {
    /// All normalized 0.0–1.0 against the desk-diagram canvas.
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Zone {
    pub id: ZoneId,
    pub label: String,
    pub layout: ZoneLayout,
    pub actions: Vec<ActionSpec>,
    pub calibration_sample_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    pub schema_version: u32,
    pub id: ProfileId,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub device_fingerprint: DeviceFingerprint,
    pub sensing_mode: SensingMode,
    pub zones: Vec<Zone>,
    /// Opaque to everything except `qwerty-core::classifier`.
    pub classifier: ClassifierParams,
    pub negative_examples_trained: bool,
}

impl Profile {
    /// Serialize to pretty JSON (the on-disk form).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("Profile serializes")
    }

    /// Whether recalibration should be suggested because the live capture device
    /// differs from the one this profile was calibrated on (`DESIGN.md`:
    /// "prompts recalibration … when the bound microphone changes"). A tap's
    /// acoustic signature is device-specific, so a classifier fitted on one mic
    /// is not trustworthy on another; the UI surfaces this rather than silently
    /// classifying against a stale model. Compares the full fingerprint, so a
    /// changed name *or* changed capability (e.g. sample rate) both trigger it.
    pub fn needs_recalibration_for(&self, current_device: &DeviceFingerprint) -> bool {
        self.device_fingerprint != *current_device
    }

    /// How many of this profile's zones have no action bound yet. A freshly
    /// calibrated profile is saved with zones but no actions (the wizard's save
    /// step says exactly this), so those zones tap-detect but do nothing until
    /// the user binds them. Home uses this to nudge toward the action editor
    /// rather than leaving silently-inert zones unexplained.
    pub fn unbound_zone_count(&self) -> usize {
        self.zones.iter().filter(|z| z.actions.is_empty()).count()
    }

    /// Parse from JSON with the schema-version policy applied.
    pub fn from_json(s: &str, file: &str) -> Result<Self, PersistenceError> {
        let profile: Profile = parse_versioned(s, PROFILE_SCHEMA_VERSION, file)?;
        // Enforce the cross-field invariants a bare deserialize cannot, so a
        // stale or hand-edited profile fails loudly rather than misbehaving later.
        profile
            .validate()
            .map_err(|reason| PersistenceError::InvalidData {
                file: file.to_string(),
                reason,
            })?;
        Ok(profile)
    }

    /// Cross-field invariants a plain deserialize cannot enforce: the classifier
    /// blob's internal consistency, and that every `ZoneLayout` coordinate is
    /// finite. `serde_json` writes a non-finite `f32` as JSON `null`, which then
    /// fails to deserialize — so validating here on **save** stops `write_atomic`
    /// from atomically replacing a good profile with a permanently unloadable
    /// one, and on **load** makes a corrupt file fail loudly. Returns a human
    /// reason string on failure.
    fn validate(&self) -> Result<(), String> {
        self.classifier.validate().map_err(|e| e.to_string())?;
        for zone in &self.zones {
            let l = &zone.layout;
            if !(l.x.is_finite() && l.y.is_finite() && l.width.is_finite() && l.height.is_finite())
            {
                return Err(format!("zone {:?} has a non-finite layout", zone.id));
            }
        }
        Ok(())
    }

    /// Load from a file path, applying the schema-version policy.
    pub fn load(path: &std::path::Path) -> Result<Self, PersistenceError> {
        let s = std::fs::read_to_string(path).map_err(PersistenceError::Io)?;
        Self::from_json(&s, &path.display().to_string())
    }

    /// Write to a file path as pretty JSON. Validates first, so a corrupt
    /// in-memory profile is never atomically written over a good one on disk.
    pub fn save(&self, path: &std::path::Path) -> Result<(), PersistenceError> {
        self.validate()
            .map_err(|reason| PersistenceError::InvalidData {
                file: path.display().to_string(),
                reason,
            })?;
        write_atomic(path, &self.to_json()).map_err(PersistenceError::Io)
    }
}

// ---------------------------------------------------------------------------
// Persistence error + version-checked parse
// ---------------------------------------------------------------------------

/// Errors from loading persisted state. Distinct variants so the UI can react
/// differently (`ARCHITECTURE.md` → Error handling policy).
#[derive(Debug)]
pub enum PersistenceError {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// The file's `schema_version` did not match the current one — recalibrate
    /// or migrate; never silently coerced.
    SchemaMismatch {
        file: String,
        found: u32,
        expected: u32,
    },
    /// The file parsed and its version matched, but its contents fail an
    /// internal integrity check (e.g. classifier dimensions inconsistent with
    /// this build). Loaded loudly-refused, never silently accepted.
    InvalidData {
        file: String,
        reason: String,
    },
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PersistenceError::Io(e) => write!(f, "I/O error: {e}"),
            PersistenceError::Json(e) => write!(f, "malformed JSON: {e}"),
            PersistenceError::SchemaMismatch {
                file,
                found,
                expected,
            } => write!(
                f,
                "{file}: schema_version {found} does not match expected {expected} \
                 — this file must be migrated or recalibrated, not loaded as-is",
            ),
            PersistenceError::InvalidData { file, reason } => {
                write!(f, "{file}: invalid contents — {reason}")
            }
        }
    }
}

impl std::error::Error for PersistenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PersistenceError::Io(e) => Some(e),
            PersistenceError::Json(e) => Some(e),
            PersistenceError::SchemaMismatch { .. } => None,
            PersistenceError::InvalidData { .. } => None,
        }
    }
}

/// Write `contents` to `path` atomically: serialize to a sibling `*.tmp` file,
/// then rename it over the target. A crash or full disk mid-write leaves the
/// previously-good file untouched instead of truncating it — losing a profile
/// (with its trained classifier) to a torn write would force a full
/// recalibration (`DATA_MODEL.md` → durability of the primary mechanism).
///
/// `pub(crate)` so [`crate::evaluation`] persists reports through the *same* one
/// atomic-write path rather than a second, subtly-different copy (`CLAUDE.md`:
/// one way to do things).
pub(crate) fn write_atomic(path: &std::path::Path, contents: &str) -> std::io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    // Flush the temp file's data all the way to disk *before* the rename. Plain
    // `fs::write` only pushes bytes into the OS page cache; on NTFS (the target)
    // directory metadata is journaled while file data is not, with no ordering
    // guarantee, so a power loss after the rename's metadata is durable but
    // before the data pages are flushed could leave the rename pointing at a
    // still-zero-length file — making the rename the *only* copy while it is
    // empty, exactly the torn write this function exists to prevent. `sync_all`
    // closes that window so the docstring's durability promise actually holds.
    {
        use std::io::Write as _;
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(contents.as_bytes())?;
        f.sync_all()?;
    }
    // `rename` replaces an existing destination on Windows and is atomic within
    // a volume; tmp is a sibling so it is always the same volume.
    std::fs::rename(&tmp, path)
}

/// Parse `s` into `T`, but only after confirming its `schema_version` matches
/// `expected`. A missing version is treated as `0` (a mismatch). Corrupt JSON
/// surfaces as [`PersistenceError::Json`] — never a panic or partial load.
///
/// `pub(crate)` so [`crate::evaluation`] loads reports under the identical
/// schema-version policy as `Profile`/`Config` (one versioned-load path).
pub(crate) fn parse_versioned<T: DeserializeOwned>(
    s: &str,
    expected: u32,
    file: &str,
) -> Result<T, PersistenceError> {
    let value: serde_json::Value = serde_json::from_str(s).map_err(PersistenceError::Json)?;
    let found = value
        .get("schema_version")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .unwrap_or(0);
    if found != expected {
        return Err(PersistenceError::SchemaMismatch {
            file: file.to_string(),
            found,
            expected,
        });
    }
    serde_json::from_value(value).map_err(PersistenceError::Json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{
        FeatureVector, CEPSTRAL_COEFF_COUNT, SPECTRAL_BAND_COUNT, TEMPORAL_FEATURE_COUNT,
    };

    #[test]
    fn unbound_zone_count_counts_zones_without_actions() {
        let mut p = sample_profile(); // its one zone has actions
        assert_eq!(p.unbound_zone_count(), 0);
        p.zones[0].actions.clear();
        assert_eq!(p.unbound_zone_count(), 1);
    }

    #[test]
    fn incomplete_reason_flags_only_blank_required_fields() {
        // Blank required field → a reason; filled → none.
        assert!(ActionSpec::OpenApp {
            path: String::new()
        }
        .incomplete_reason()
        .is_some());
        assert!(ActionSpec::OpenApp { path: "app".into() }
            .incomplete_reason()
            .is_none());
        assert!(ActionSpec::SendKeystroke {
            combo: KeyCombo {
                modifiers: vec![],
                key: "   ".into()
            }
        }
        .incomplete_reason()
        .is_some());
        // No-required-field actions are always complete.
        assert!(ActionSpec::VisualOnly.incomplete_reason().is_none());
        assert!(ActionSpec::PlaySound {
            sound: SystemSound::Default
        }
        .incomplete_reason()
        .is_none());
        // A toast needs at least one of title/body.
        assert!(ActionSpec::ShowToast {
            title: String::new(),
            body: String::new()
        }
        .incomplete_reason()
        .is_some());
        assert!(ActionSpec::ShowToast {
            title: "Hi".into(),
            body: String::new()
        }
        .incomplete_reason()
        .is_none());
    }

    #[test]
    fn device_fingerprint_from_device_uses_the_sample_rate_hash() {
        let fp = DeviceFingerprint::from_device("USB Mic", 48_000);
        assert_eq!(fp.device_name, "USB Mic");
        assert_eq!(fp.capability_hash, "sr48000");
    }

    #[test]
    fn needs_recalibration_only_when_the_device_differs() {
        let p = sample_profile();
        // Same device the profile was calibrated on → no prompt.
        assert!(!p.needs_recalibration_for(&p.device_fingerprint.clone()));
        // A different microphone → prompt.
        let other = DeviceFingerprint::from_device("A Different Mic", 48_000);
        assert!(p.needs_recalibration_for(&other));
    }

    fn tiny_classifier() -> ClassifierParams {
        // Two crude clusters so a real (non-empty) classifier blob exists to
        // round-trip inside a Profile.
        let fv = |s: f32| FeatureVector {
            temporal: [s; TEMPORAL_FEATURE_COUNT],
            spectral_bands: [s * 0.5; SPECTRAL_BAND_COUNT],
            cepstral: [s * 0.25; CEPSTRAL_COEFF_COUNT],
        };
        let za = ZoneId::from_uuid(Uuid::from_u128(10));
        let zb = ZoneId::from_uuid(Uuid::from_u128(20));
        let samples: Vec<_> = (0..6)
            .flat_map(|j| {
                let jit = j as f32 * 1e-3;
                [(za, fv(jit)), (zb, fv(3.0 + jit))]
            })
            .collect();
        ClassifierParams::train(&samples, &[]).unwrap()
    }

    fn sample_profile() -> Profile {
        let ts = DateTime::parse_from_rfc3339("2026-07-21T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        Profile {
            schema_version: PROFILE_SCHEMA_VERSION,
            id: ProfileId::from_uuid(Uuid::from_u128(1)),
            name: "Home desk".into(),
            created_at: ts,
            updated_at: ts,
            device_fingerprint: DeviceFingerprint {
                device_name: "Built-in Microphone".into(),
                capability_hash: "abc123".into(),
            },
            sensing_mode: SensingMode::Passive,
            zones: vec![Zone {
                id: ZoneId::from_uuid(Uuid::from_u128(2)),
                label: "Top-left".into(),
                layout: ZoneLayout {
                    x: 0.0,
                    y: 0.0,
                    width: 0.5,
                    height: 0.5,
                },
                actions: vec![
                    ActionSpec::OpenApp {
                        path: "C:\\Windows\\System32\\WindowsTerminal.exe".into(),
                    },
                    ActionSpec::ShowToast {
                        title: "Terminal".into(),
                        body: "Opened via top-left tap".into(),
                    },
                ],
                calibration_sample_count: 10,
            }],
            classifier: tiny_classifier(),
            negative_examples_trained: false,
        }
    }

    #[test]
    fn config_default_round_trips() {
        let c = Config::default();
        let back = Config::from_json(&c.to_json(), "config.json").unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn config_with_hotkey_round_trips() {
        let c = Config {
            global_hotkey: Some(KeyCombo {
                modifiers: vec![Modifier::Ctrl, Modifier::Shift],
                key: "Q".into(),
            }),
            active_profile_id: Some(ProfileId::from_uuid(Uuid::from_u128(7))),
            ..Default::default()
        };
        let back = Config::from_json(&c.to_json(), "config.json").unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn profile_round_trips_with_no_field_loss() {
        let p = sample_profile();
        let back = Profile::from_json(&p.to_json(), "p.json").unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn save_rejects_a_non_finite_zone_layout() {
        // serde_json writes a non-finite f32 as JSON `null`, which then fails to
        // load — so a profile with a NaN layout must be refused rather than
        // atomically written over a good file (validate runs before write).
        let mut p = sample_profile();
        assert!(p.validate().is_ok(), "the sample profile is valid");
        p.zones[0].layout.x = f32::NAN;
        assert!(
            p.validate().is_err(),
            "a non-finite zone layout must be rejected"
        );
    }

    #[test]
    fn schema_version_mismatch_is_a_named_error() {
        let mut v: serde_json::Value = serde_json::from_str(&Config::default().to_json()).unwrap();
        v["schema_version"] = serde_json::json!(999);
        let err = Config::from_json(&v.to_string(), "config.json").unwrap_err();
        match err {
            PersistenceError::SchemaMismatch {
                file,
                found,
                expected,
            } => {
                assert_eq!(file, "config.json");
                assert_eq!(found, 999);
                assert_eq!(expected, CONFIG_SCHEMA_VERSION);
            }
            other => panic!("expected SchemaMismatch, got {other:?}"),
        }
    }

    #[test]
    fn corrupt_json_is_an_error_not_a_panic() {
        let truncated = &Config::default().to_json()[..20];
        assert!(matches!(
            Config::from_json(truncated, "config.json"),
            Err(PersistenceError::Json(_))
        ));
    }

    #[test]
    fn actionspec_json_is_greppable_against_variant_names() {
        let p = sample_profile();
        let json = p.to_json();
        assert!(json.contains("\"type\": \"OpenApp\""));
        assert!(json.contains("\"type\": \"ShowToast\""));
        assert!(json.contains("\"path\":"));
    }

    #[test]
    fn profile_with_malformed_classifier_is_refused_on_load() {
        // A stale/hand-edited profile whose classifier vectors are the wrong
        // length must fail loudly on load, not silently truncate at classify.
        let p = sample_profile();
        let mut v: serde_json::Value = serde_json::from_str(&p.to_json()).unwrap();
        v["classifier"]["feature_mean"] = serde_json::json!([0.0, 1.0]);
        let err = Profile::from_json(&v.to_string(), "p.json").unwrap_err();
        assert!(matches!(err, PersistenceError::InvalidData { .. }));
    }
}
