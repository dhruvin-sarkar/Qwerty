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
        std::fs::write(path, self.to_json()).map_err(PersistenceError::Io)
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

    /// Parse from JSON with the schema-version policy applied.
    pub fn from_json(s: &str, file: &str) -> Result<Self, PersistenceError> {
        let profile: Profile = parse_versioned(s, PROFILE_SCHEMA_VERSION, file)?;
        // The classifier blob is only version-checked at the outer schema level;
        // validate its internal invariants so a stale or hand-edited profile
        // fails loudly instead of silently truncating and misclassifying.
        profile
            .classifier
            .validate()
            .map_err(|e| PersistenceError::InvalidData {
                file: file.to_string(),
                reason: e.to_string(),
            })?;
        Ok(profile)
    }

    /// Load from a file path, applying the schema-version policy.
    pub fn load(path: &std::path::Path) -> Result<Self, PersistenceError> {
        let s = std::fs::read_to_string(path).map_err(PersistenceError::Io)?;
        Self::from_json(&s, &path.display().to_string())
    }

    /// Write to a file path as pretty JSON.
    pub fn save(&self, path: &std::path::Path) -> Result<(), PersistenceError> {
        std::fs::write(path, self.to_json()).map_err(PersistenceError::Io)
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
    InvalidData { file: String, reason: String },
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

/// Parse `s` into `T`, but only after confirming its `schema_version` matches
/// `expected`. A missing version is treated as `0` (a mismatch). Corrupt JSON
/// surfaces as [`PersistenceError::Json`] — never a panic or partial load.
fn parse_versioned<T: DeserializeOwned>(
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
    fn schema_version_mismatch_is_a_named_error() {
        let mut v: serde_json::Value =
            serde_json::from_str(&Config::default().to_json()).unwrap();
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
