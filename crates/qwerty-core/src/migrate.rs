//! Explicit migration of v1 config and profile files to v2 (§1.16).
//!
//! Loading dispatches on the file's `schema_version`: 2 is parsed, normalized and validated; 1 is
//! parsed into frozen private copies of the v1 shapes and mapped field by field; anything else is a
//! named refusal. The frozen copies cover every v1 type the v1 files embed, so no later edit to a
//! live type (and no deletion in WP-D1) can break v1 migration (PARITY-4). Only the id newtypes are
//! shared: they are identities that no schema version changes.

use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize};

use crate::gesture::Trigger;
use crate::profile::{
    ActionSpec, KeyCombo, Modifier, NotificationVerbosity, PersistenceError, ProfileId,
    ScreenshotMode, SystemSound, Theme, ZoneId, ZoneLayout,
};
use crate::store::{
    check_profile, default_global_hotkey, default_zones, invalid_data, label_key, layout_is_valid,
    report_u32, unique_label, zone_count_of, CalibrationState, Config, Profile,
    RecalibrationReason, Zone, CONFIG_SCHEMA_VERSION, PROFILE_SCHEMA_VERSION, ZONE_LABEL_MAX,
};

/// The only older schema version there is a migration from.
const V1_SCHEMA_VERSION: u32 = 1;
/// The version `peek_schema_version` reports for a file without a `schema_version` (§1.16.2): a
/// mismatch for every loader, never a guess.
const ABSENT_SCHEMA_VERSION: u32 = 0;

#[derive(Debug, Clone, PartialEq)]
pub enum Loaded<T> {
    Current(T),
    Migrated {
        value: T,
        from_version: u32,
        notes: Vec<MigrationNote>,
    },
}

/// Everything a migration changed beyond a field copy, so the caller can show it
/// (`Event::ProfileMigrated`/`ConfigMigrated`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MigrationNote {
    /// v1 None → Some(Ctrl+Alt+Space): v1 always registered it.
    HotkeyDefaulted,
    VisualOnlyRemoved {
        zone: ZoneId,
        count: u32,
    },
    /// To ZONE_LABEL_MAX chars.
    LabelTruncated {
        zone: ZoneId,
    },
    /// Empty → its default name, or a duplicate → " 2", " 3", …
    LabelRenamed {
        zone: ZoneId,
        label: String,
    },
    /// Failed the v2 layout check → its `default_zones` entry.
    LayoutReset {
        zone: ZoneId,
    },
}

/// `{schema_version: Option<u32>}`; absent ⇒ 0. Malformed JSON, or a version that is not a u32, is
/// `PersistenceError::Json`.
// SPEC: the signature carries `file`, but every error this function can return is
// `PersistenceError::Json`, which carries no file name, so it is not read.
pub fn peek_schema_version(text: &str, _file: &Path) -> Result<u32, PersistenceError> {
    #[derive(Deserialize)]
    struct Peek {
        schema_version: Option<u32>,
    }
    let peek: Peek = serde_json::from_str(text).map_err(PersistenceError::Json)?;
    // Absence is a version of its own (0), not a silenced error: it then fails dispatch loudly.
    Ok(peek.schema_version.unwrap_or(ABSENT_SCHEMA_VERSION))
}

pub fn load_config(text: &str, file: &Path) -> Result<Loaded<Config>, PersistenceError> {
    match peek_schema_version(text, file)? {
        CONFIG_SCHEMA_VERSION => {
            let config: Config = serde_json::from_str(text).map_err(PersistenceError::Json)?;
            config.validate().map_err(|e| invalid_data(file, &e))?;
            Ok(Loaded::Current(config))
        }
        V1_SCHEMA_VERSION => {
            let v1: V1Config = serde_json::from_str(text).map_err(PersistenceError::Json)?;
            let (config, notes) = migrate_config(v1);
            // A v1 hotkey is kept as it is, so one without Ctrl, Alt or Win is refused here
            // rather than handed back as a value no save path accepts.
            config.validate().map_err(|e| invalid_data(file, &e))?;
            Ok(Loaded::Migrated {
                value: config,
                from_version: V1_SCHEMA_VERSION,
                notes,
            })
        }
        found => Err(PersistenceError::SchemaMismatch {
            file: file.display().to_string(),
            found,
            expected: CONFIG_SCHEMA_VERSION,
        }),
    }
}

pub fn load_profile(text: &str, file: &Path) -> Result<Loaded<Profile>, PersistenceError> {
    match peek_schema_version(text, file)? {
        PROFILE_SCHEMA_VERSION => {
            let mut profile: Profile =
                serde_json::from_str(text).map_err(PersistenceError::Json)?;
            // Normalizing first lets a model from an older front end become
            // NeedsRecalibration{FrontEndChanged} instead of failing a dims check.
            profile.normalize();
            check_profile(&profile, file)?;
            Ok(Loaded::Current(profile))
        }
        V1_SCHEMA_VERSION => {
            let v1: V1Profile = serde_json::from_str(text).map_err(PersistenceError::Json)?;
            // SPEC: §1.16.2 sets updated_at = now, but load_profile takes no clock argument, so the
            // migration reads the wall clock here; it is the only non-pure step of a load.
            let (profile, notes) = migrate_profile(v1, Utc::now(), file)?;
            Ok(Loaded::Migrated {
                value: profile,
                from_version: V1_SCHEMA_VERSION,
                notes,
            })
        }
        found => Err(PersistenceError::SchemaMismatch {
            file: file.display().to_string(),
            found,
            expected: PROFILE_SCHEMA_VERSION,
        }),
    }
}

/// Every v1 config field through the frozen types; `input_veto_enabled = true`; the shell fields at
/// their `initial()` values; a v1 `None` hotkey becomes Ctrl+Alt+Space (PARITY-18).
fn migrate_config(v1: V1Config) -> (Config, Vec<MigrationNote>) {
    let V1Config {
        schema_version,
        active_profile_id,
        theme,
        start_with_windows,
        minimize_to_tray_on_close,
        notification_verbosity,
        global_hotkey,
        debug_capture_enabled,
    } = v1;
    debug_assert_eq!(
        schema_version, V1_SCHEMA_VERSION,
        "dispatched on the peeked version"
    );
    let mut notes = Vec::new();
    let global_hotkey = match global_hotkey {
        Some(combo) => map_key_combo(combo),
        None => {
            // v1 ignored the field and always registered Ctrl+Alt+Space (v1 hotkeys.rs:33-40).
            notes.push(MigrationNote::HotkeyDefaulted);
            default_global_hotkey()
        }
    };
    let initial = Config::initial();
    let config = Config {
        schema_version: CONFIG_SCHEMA_VERSION,
        active_profile_id,
        theme: map_theme(theme),
        start_with_windows,
        minimize_to_tray_on_close,
        notification_verbosity: map_verbosity(notification_verbosity),
        global_hotkey: Some(global_hotkey),
        debug_capture_enabled,
        input_veto_enabled: true,
        eval_prompts_per_zone: initial.eval_prompts_per_zone,
        eval_background_check: initial.eval_background_check,
        dismissed_notices: initial.dismissed_notices,
        window_placement: initial.window_placement,
        text_zoom: initial.text_zoom,
    };
    (config, notes)
}

/// §1.16.2: id, name and created_at kept; updated_at = now; every zone keeps id, label, layout and
/// actions with Trigger::Single; VisualOnly removed; labels trimmed, truncated and de-duplicated;
/// invalid layouts reset; NeedsRecalibration{MigratedFromV1}; imported_at None.
fn migrate_profile(
    v1: V1Profile,
    now: DateTime<Utc>,
    file: &Path,
) -> Result<(Profile, Vec<MigrationNote>), PersistenceError> {
    let V1Profile {
        schema_version,
        id,
        name,
        created_at,
        updated_at: _replaced_by_now,
        device_fingerprint,
        sensing_mode: _dropped_sensing_mode, // only Passive exists (deviation g)
        zones,
        classifier: _dropped_classifier, // v2 retrains (NeedsRecalibration)
    } = v1;
    debug_assert_eq!(
        schema_version, V1_SCHEMA_VERSION,
        "dispatched on the peeked version"
    );
    let V1DeviceFingerprint {
        device_name,
        capability_hash: _dropped_capability_hash, // the v2 identity is captured at calibration
    } = device_fingerprint;

    let count = zone_count_of(zones.len()).map_err(|e| invalid_data(file, &e))?;
    let mut notes = Vec::new();
    let mut migrated: Vec<Zone> = Vec::with_capacity(zones.len());
    for (v1_zone, (default_name, default_layout)) in zones.into_iter().zip(default_zones(count)) {
        let V1Zone {
            id: zone,
            label,
            layout,
            actions,
            calibration_sample_count: _dropped_sample_count,
        } = v1_zone;

        let v1_len = actions.len();
        let actions: Vec<ActionSpec> = actions.into_iter().filter_map(map_action).collect();
        if actions.len() != v1_len {
            notes.push(MigrationNote::VisualOnlyRemoved {
                zone,
                count: report_u32(v1_len - actions.len()),
            });
        }

        let trimmed = label.trim();
        let label: String = if trimmed.chars().count() > ZONE_LABEL_MAX {
            notes.push(MigrationNote::LabelTruncated { zone });
            trimmed.chars().take(ZONE_LABEL_MAX).collect()
        } else {
            trimmed.to_string()
        };
        let taken: Vec<String> = migrated.iter().map(|z| label_key(&z.label)).collect();
        let base = if label.is_empty() {
            default_name
        } else {
            label.as_str()
        };
        let unique = unique_label(base, &taken);
        let label = if unique == label {
            label
        } else {
            notes.push(MigrationNote::LabelRenamed {
                zone,
                label: unique.clone(),
            });
            unique
        };

        let layout = map_layout(layout);
        let layout = if layout_is_valid(&layout) {
            layout
        } else {
            notes.push(MigrationNote::LayoutReset { zone });
            default_layout
        };

        migrated.push(Zone {
            id: zone,
            label,
            layout,
            trigger: Trigger::Single,
            actions,
        });
    }

    let profile = Profile {
        schema_version: PROFILE_SCHEMA_VERSION,
        id,
        name,
        created_at,
        updated_at: now,
        zones: migrated,
        calibration: CalibrationState::NeedsRecalibration {
            reason: RecalibrationReason::MigratedFromV1 {
                v1_device_name: device_name,
            },
        },
        imported_at: None,
    };
    // SPEC: "the result passes validate(), which the migration asserts". The label, layout and
    // action repairs guarantee it for every zone, but the v1 name, zone count and zone ids are kept
    // as they are, so a v1 file whose name is empty or too long, whose zone count is not allowed,
    // or whose zone ids repeat is refused as InvalidData (listed Unreadable) instead of panicking.
    check_profile(&profile, file)?;
    Ok((profile, notes))
}

// ---------------------------------------------------------------------------
// Frozen v1 shapes (private). Each is a byte-for-byte copy of v1's serde shape (profile.rs at
// schema version 1), mapped to the live types by the explicit, exhaustive functions below.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct V1Config {
    schema_version: u32,
    active_profile_id: Option<ProfileId>,
    theme: V1Theme,
    start_with_windows: bool,
    minimize_to_tray_on_close: bool,
    notification_verbosity: V1NotificationVerbosity,
    global_hotkey: Option<V1KeyCombo>,
    debug_capture_enabled: bool,
}

#[derive(Deserialize)]
struct V1DeviceFingerprint {
    device_name: String,
    capability_hash: String,
}

#[derive(Deserialize)]
struct V1Zone {
    id: ZoneId,
    label: String,
    layout: V1ZoneLayout,
    actions: Vec<V1ActionSpec>,
    calibration_sample_count: u32,
}

#[derive(Deserialize)]
struct V1Profile {
    schema_version: u32,
    id: ProfileId,
    name: String,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    device_fingerprint: V1DeviceFingerprint,
    sensing_mode: serde::de::IgnoredAny,
    zones: Vec<V1Zone>,
    classifier: serde::de::IgnoredAny,
}

#[derive(Deserialize)]
enum V1Theme {
    System,
    Daylight,
    Midnight,
    Aurora,
    HighContrast,
    Ember,
    Grape,
    Translucent,
}

#[derive(Deserialize)]
enum V1NotificationVerbosity {
    Silent,
    Minimal,
    Normal,
}

#[derive(Deserialize)]
enum V1Modifier {
    Ctrl,
    Shift,
    Alt,
    Win,
}

#[derive(Deserialize)]
struct V1KeyCombo {
    modifiers: Vec<V1Modifier>,
    key: String,
}

#[derive(Deserialize)]
enum V1SystemSound {
    Default,
    Success,
    Error,
    Notification,
}

#[derive(Deserialize)]
enum V1ScreenshotMode {
    FullScreen,
    RegionSelect,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum V1ActionSpec {
    OpenApp { path: String },
    OpenPath { path: String },
    OpenUrl { url: String },
    RunCommand { command: String, args: Vec<String> },
    CopyToClipboard { text: String },
    Speak { text: String },
    PlaySound { sound: V1SystemSound },
    ShowToast { title: String, body: String },
    SendKeystroke { combo: V1KeyCombo },
    Screenshot { mode: V1ScreenshotMode },
    VisualOnly,
}

// SPEC: ST2 migrates "a NaN layout" with LayoutReset, and v1 wrote a non-finite coordinate as JSON
// `null` (serde_json), which v1's own `f32` field could not read back. The frozen shape keeps v1's
// fields and types but reads such a `null` as NaN, so the v2 layout check resets it.
#[derive(Deserialize)]
struct V1ZoneLayout {
    #[serde(deserialize_with = "v1_coordinate")]
    x: f32,
    #[serde(deserialize_with = "v1_coordinate")]
    y: f32,
    #[serde(deserialize_with = "v1_coordinate")]
    width: f32,
    #[serde(deserialize_with = "v1_coordinate")]
    height: f32,
}

/// A v1 layout coordinate: a number, or the `null` v1 wrote for a non-finite one (read as NaN, which
/// the v2 layout check rejects, so the layout is reset and noted).
fn v1_coordinate<'de, D: Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    Ok(Option::<f32>::deserialize(d)?.unwrap_or(f32::NAN))
}

fn map_theme(t: V1Theme) -> Theme {
    match t {
        V1Theme::System => Theme::System,
        V1Theme::Daylight => Theme::Daylight,
        V1Theme::Midnight => Theme::Midnight,
        V1Theme::Aurora => Theme::Aurora,
        V1Theme::HighContrast => Theme::HighContrast,
        V1Theme::Ember => Theme::Ember,
        V1Theme::Grape => Theme::Grape,
        V1Theme::Translucent => Theme::Translucent,
    }
}

fn map_verbosity(v: V1NotificationVerbosity) -> NotificationVerbosity {
    match v {
        V1NotificationVerbosity::Silent => NotificationVerbosity::Silent,
        V1NotificationVerbosity::Minimal => NotificationVerbosity::Minimal,
        V1NotificationVerbosity::Normal => NotificationVerbosity::Normal,
    }
}

fn map_modifier(m: V1Modifier) -> Modifier {
    match m {
        V1Modifier::Ctrl => Modifier::Ctrl,
        V1Modifier::Shift => Modifier::Shift,
        V1Modifier::Alt => Modifier::Alt,
        V1Modifier::Win => Modifier::Win,
    }
}

fn map_key_combo(c: V1KeyCombo) -> KeyCombo {
    let V1KeyCombo { modifiers, key } = c;
    KeyCombo {
        modifiers: modifiers.into_iter().map(map_modifier).collect(),
        key,
    }
}

fn map_sound(s: V1SystemSound) -> SystemSound {
    match s {
        V1SystemSound::Default => SystemSound::Default,
        V1SystemSound::Success => SystemSound::Success,
        V1SystemSound::Error => SystemSound::Error,
        V1SystemSound::Notification => SystemSound::Notification,
    }
}

fn map_screenshot_mode(m: V1ScreenshotMode) -> ScreenshotMode {
    match m {
        V1ScreenshotMode::FullScreen => ScreenshotMode::FullScreen,
        V1ScreenshotMode::RegionSelect => ScreenshotMode::RegionSelect,
    }
}

/// `None` for VisualOnly, which v2 removes: an empty chain already means "lights up only".
fn map_action(a: V1ActionSpec) -> Option<ActionSpec> {
    Some(match a {
        V1ActionSpec::OpenApp { path } => ActionSpec::OpenApp { path },
        V1ActionSpec::OpenPath { path } => ActionSpec::OpenPath { path },
        V1ActionSpec::OpenUrl { url } => ActionSpec::OpenUrl { url },
        V1ActionSpec::RunCommand { command, args } => ActionSpec::RunCommand { command, args },
        V1ActionSpec::CopyToClipboard { text } => ActionSpec::CopyToClipboard { text },
        V1ActionSpec::Speak { text } => ActionSpec::Speak { text },
        V1ActionSpec::PlaySound { sound } => ActionSpec::PlaySound {
            sound: map_sound(sound),
        },
        V1ActionSpec::ShowToast { title, body } => ActionSpec::ShowToast { title, body },
        V1ActionSpec::SendKeystroke { combo } => ActionSpec::SendKeystroke {
            combo: map_key_combo(combo),
        },
        V1ActionSpec::Screenshot { mode } => ActionSpec::Screenshot {
            mode: map_screenshot_mode(mode),
        },
        V1ActionSpec::VisualOnly => return None,
    })
}

fn map_layout(l: V1ZoneLayout) -> ZoneLayout {
    let V1ZoneLayout {
        x,
        y,
        width,
        height,
    } = l;
    ZoneLayout {
        x,
        y,
        width,
        height,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::detector::FRONT_END_VERSION;
    use crate::store::tests::{calibrated_profile, count, file, model_mut, ts};
    use crate::store::{config_to_json, profile_to_json};
    use serde_json::{json, Value};
    use uuid::Uuid;

    /// A hand-written v1 config.json (schema version 1), never real user data.
    pub(crate) const GOLDEN_V1_CONFIG: &str = r#"{
  "schema_version": 1,
  "active_profile_id": "00000000-0000-0000-0000-0000000000b3",
  "theme": "Ember",
  "start_with_windows": true,
  "minimize_to_tray_on_close": false,
  "notification_verbosity": "Minimal",
  "global_hotkey": null,
  "debug_capture_enabled": true
}"#;

    /// A hand-written v1 profile (schema version 1), never real user data. Zone 1 chains
    /// [VisualOnly, Speak]; zones 1 and 2 are labelled "Desk" and "desk"; zone 3 has a 40-char label
    /// and the `null` v1 wrote for a NaN coordinate; zone 4's label carries spaces. The classifier
    /// blob is present and deliberately malformed.
    pub(crate) const GOLDEN_V1_PROFILE: &str = r#"{
  "schema_version": 1,
  "id": "00000000-0000-0000-0000-0000000000b3",
  "name": "Home desk",
  "created_at": "2026-07-21T12:00:00Z",
  "updated_at": "2026-07-22T08:30:00Z",
  "device_fingerprint": {
    "device_name": "Built-in Microphone",
    "capability_hash": "sr48000"
  },
  "sensing_mode": "Passive",
  "zones": [
    {
      "id": "00000000-0000-0000-0000-0000000000a1",
      "label": "Desk",
      "layout": { "x": 0.03, "y": 0.08, "width": 0.29, "height": 0.4 },
      "actions": [
        { "type": "VisualOnly" },
        { "type": "Speak", "text": "hello" }
      ],
      "calibration_sample_count": 10
    },
    {
      "id": "00000000-0000-0000-0000-0000000000a2",
      "label": "desk",
      "layout": { "x": 0.68, "y": 0.08, "width": 0.29, "height": 0.4 },
      "actions": [
        { "type": "OpenApp", "path": "C:\\Tools\\editor.exe" },
        { "type": "SendKeystroke", "combo": { "modifiers": ["Ctrl", "Alt", "Shift", "Win"], "key": "T" } },
        { "type": "PlaySound", "sound": "Success" },
        { "type": "Screenshot", "mode": "RegionSelect" }
      ],
      "calibration_sample_count": 12
    },
    {
      "id": "00000000-0000-0000-0000-0000000000a3",
      "label": "abcdefghijklmnopqrstuvwxyz0123456789ABCD",
      "layout": { "x": null, "y": 0.52, "width": 0.29, "height": 0.4 },
      "actions": [
        { "type": "OpenUrl", "url": "https://example.org" },
        { "type": "RunCommand", "command": "tool", "args": ["--flag", "value"] },
        { "type": "CopyToClipboard", "text": "snippet" },
        { "type": "ShowToast", "title": "Hi", "body": "there" },
        { "type": "OpenPath", "path": "D:\\Notes" },
        { "type": "PlaySound", "sound": "Default" },
        { "type": "PlaySound", "sound": "Error" },
        { "type": "PlaySound", "sound": "Notification" },
        { "type": "Screenshot", "mode": "FullScreen" },
        { "type": "VisualOnly" }
      ],
      "calibration_sample_count": 9
    },
    {
      "id": "00000000-0000-0000-0000-0000000000a4",
      "label": "  Right front  ",
      "layout": { "x": 0.6, "y": 0.5, "width": 0.3, "height": 0.4 },
      "actions": [],
      "calibration_sample_count": 11
    }
  ],
  "classifier": {
    "feature_mean": [0.0, "NaN"],
    "centroids": null,
    "nested": { "deeper": [[1, 2], { "x": false }] }
  }
}"#;

    fn zone(n: u128) -> ZoneId {
        ZoneId::from_uuid(Uuid::from_u128(n))
    }

    fn layout(x: f32, y: f32, width: f32, height: f32) -> ZoneLayout {
        ZoneLayout {
            x,
            y,
            width,
            height,
        }
    }

    fn migrated<T: std::fmt::Debug>(loaded: Loaded<T>) -> (T, Vec<MigrationNote>) {
        match loaded {
            Loaded::Migrated {
                value,
                from_version,
                notes,
            } => {
                assert_eq!(from_version, 1);
                (value, notes)
            }
            other => panic!("expected Migrated, got {other:?}"),
        }
    }

    fn invalid_reason<T: std::fmt::Debug>(result: Result<T, PersistenceError>) -> String {
        match result {
            Err(PersistenceError::InvalidData { file, reason }) => {
                assert_eq!(file, "profile.json");
                reason
            }
            other => panic!("expected InvalidData, got {other:?}"),
        }
    }

    fn edited(text: &str, edit: impl FnOnce(&mut Value)) -> String {
        let mut v: Value = serde_json::from_str(text).expect("valid JSON");
        edit(&mut v);
        v.to_string()
    }

    #[test]
    fn st1_v1_config_migrates() {
        let (config, notes) = migrated(load_config(GOLDEN_V1_CONFIG, file()).expect("migrates"));
        let initial = Config::initial();
        assert_eq!(
            config,
            Config {
                schema_version: CONFIG_SCHEMA_VERSION,
                active_profile_id: Some(ProfileId::from_uuid(Uuid::from_u128(0xB3))),
                theme: Theme::Ember,
                start_with_windows: true,
                minimize_to_tray_on_close: false,
                notification_verbosity: NotificationVerbosity::Minimal,
                global_hotkey: Some(KeyCombo {
                    modifiers: vec![Modifier::Ctrl, Modifier::Alt],
                    key: "Space".into(),
                }),
                debug_capture_enabled: true,
                input_veto_enabled: true,
                eval_prompts_per_zone: initial.eval_prompts_per_zone,
                eval_background_check: initial.eval_background_check,
                dismissed_notices: initial.dismissed_notices.clone(),
                window_placement: initial.window_placement,
                text_zoom: initial.text_zoom,
            }
        );
        assert_eq!(config.global_hotkey, initial.global_hotkey);
        assert_eq!(notes, [MigrationNote::HotkeyDefaulted]);
        // The caller writes it back: it serializes and then loads as the current version.
        let text = config_to_json(&config, file()).expect("a migrated config is valid");
        assert_eq!(
            load_config(&text, file()).expect("loads"),
            Loaded::Current(config)
        );

        // A v1 Some(k) stays Some(k), with no note; every v1 theme and verbosity maps by name.
        let themes = [
            ("System", Theme::System),
            ("Daylight", Theme::Daylight),
            ("Midnight", Theme::Midnight),
            ("Aurora", Theme::Aurora),
            ("HighContrast", Theme::HighContrast),
            ("Ember", Theme::Ember),
            ("Grape", Theme::Grape),
            ("Translucent", Theme::Translucent),
        ];
        let verbosities = [
            ("Silent", NotificationVerbosity::Silent),
            ("Minimal", NotificationVerbosity::Minimal),
            ("Normal", NotificationVerbosity::Normal),
        ];
        for (i, (theme_name, theme)) in themes.iter().enumerate() {
            let (verbosity_name, verbosity) = verbosities[i % verbosities.len()];
            let text = edited(GOLDEN_V1_CONFIG, |v| {
                v["theme"] = json!(theme_name);
                v["notification_verbosity"] = json!(verbosity_name);
                v["global_hotkey"] = json!({ "modifiers": ["Shift", "Win"], "key": "Q" });
                v["active_profile_id"] = Value::Null;
            });
            let (config, notes) = migrated(load_config(&text, file()).expect("migrates"));
            assert_eq!(config.theme, *theme);
            assert_eq!(config.notification_verbosity, verbosity);
            assert_eq!(config.active_profile_id, None);
            assert_eq!(
                config.global_hotkey,
                Some(KeyCombo {
                    modifiers: vec![Modifier::Shift, Modifier::Win],
                    key: "Q".into(),
                })
            );
            assert!(notes.is_empty(), "{notes:?}");
        }
    }

    #[test]
    fn st2_v1_profile_migrates() {
        let before = Utc::now();
        let (profile, notes) = migrated(load_profile(GOLDEN_V1_PROFILE, file()).expect("migrates"));
        let after = Utc::now();

        assert_eq!(profile.schema_version, PROFILE_SCHEMA_VERSION);
        assert_eq!(profile.id, ProfileId::from_uuid(Uuid::from_u128(0xB3)));
        assert_eq!(profile.name, "Home desk");
        assert_eq!(profile.created_at, ts("2026-07-21T12:00:00Z"));
        assert!(before <= profile.updated_at && profile.updated_at <= after);
        assert_eq!(
            profile.calibration,
            CalibrationState::NeedsRecalibration {
                reason: RecalibrationReason::MigratedFromV1 {
                    v1_device_name: "Built-in Microphone".into()
                }
            }
        );
        assert_eq!(profile.imported_at, None);

        let ids: Vec<ZoneId> = profile.zones.iter().map(|z| z.id).collect();
        assert_eq!(ids, [zone(0xA1), zone(0xA2), zone(0xA3), zone(0xA4)]);
        assert!(profile.zones.iter().all(|z| z.trigger == Trigger::Single));
        let labels: Vec<&str> = profile.zones.iter().map(|z| z.label.as_str()).collect();
        assert_eq!(
            labels,
            [
                "Desk",
                "desk 2",
                "abcdefghijklmnopqrstuvwxyz012345",
                "Right front"
            ]
        );
        assert_eq!(profile.zones[2].label.chars().count(), ZONE_LABEL_MAX);
        let layouts: Vec<ZoneLayout> = profile.zones.iter().map(|z| z.layout).collect();
        assert_eq!(
            layouts,
            [
                layout(0.03, 0.08, 0.29, 0.4),
                layout(0.68, 0.08, 0.29, 0.4),
                default_zones(count(4))[2].1,
                layout(0.6, 0.5, 0.3, 0.4),
            ]
        );
        assert_eq!(
            profile.zones[0].actions,
            [ActionSpec::Speak {
                text: "hello".into()
            }]
        );
        assert_eq!(
            profile.zones[1].actions,
            [
                ActionSpec::OpenApp {
                    path: "C:\\Tools\\editor.exe".into()
                },
                ActionSpec::SendKeystroke {
                    combo: KeyCombo {
                        modifiers: vec![
                            Modifier::Ctrl,
                            Modifier::Alt,
                            Modifier::Shift,
                            Modifier::Win
                        ],
                        key: "T".into()
                    }
                },
                ActionSpec::PlaySound {
                    sound: SystemSound::Success
                },
                ActionSpec::Screenshot {
                    mode: ScreenshotMode::RegionSelect
                },
            ]
        );
        assert_eq!(
            profile.zones[2].actions,
            [
                ActionSpec::OpenUrl {
                    url: "https://example.org".into()
                },
                ActionSpec::RunCommand {
                    command: "tool".into(),
                    args: vec!["--flag".into(), "value".into()]
                },
                ActionSpec::CopyToClipboard {
                    text: "snippet".into()
                },
                ActionSpec::ShowToast {
                    title: "Hi".into(),
                    body: "there".into()
                },
                ActionSpec::OpenPath {
                    path: "D:\\Notes".into()
                },
                ActionSpec::PlaySound {
                    sound: SystemSound::Default
                },
                ActionSpec::PlaySound {
                    sound: SystemSound::Error
                },
                ActionSpec::PlaySound {
                    sound: SystemSound::Notification
                },
                ActionSpec::Screenshot {
                    mode: ScreenshotMode::FullScreen
                },
            ]
        );
        assert!(profile.zones[3].actions.is_empty());
        assert_eq!(
            notes,
            [
                MigrationNote::VisualOnlyRemoved {
                    zone: zone(0xA1),
                    count: 1
                },
                MigrationNote::LabelRenamed {
                    zone: zone(0xA2),
                    label: "desk 2".into()
                },
                MigrationNote::VisualOnlyRemoved {
                    zone: zone(0xA3),
                    count: 1
                },
                MigrationNote::LabelTruncated { zone: zone(0xA3) },
                MigrationNote::LayoutReset { zone: zone(0xA3) },
            ]
        );
        profile.validate().expect("the migration result is valid");
        // The caller writes it back immediately; the written file loads as the current version.
        let text = profile_to_json(&profile, file()).expect("a migrated profile is valid");
        assert_eq!(
            load_profile(&text, file()).expect("loads"),
            Loaded::Current(profile.clone())
        );

        // An empty label takes its default name; an off-canvas layout is reset like a NaN one; a
        // duplicate of a full-length label is numbered within ZONE_LABEL_MAX.
        let text = edited(GOLDEN_V1_PROFILE, |v| {
            v["zones"][3]["label"] = json!("   ");
            v["zones"][1]["layout"]["x"] = json!(1.5);
            v["zones"][0]["label"] = json!("abcdefghijklmnopqrstuvwxyz012345");
        });
        let (profile, notes) = migrated(load_profile(&text, file()).expect("migrates"));
        assert_eq!(profile.zones[0].label, "abcdefghijklmnopqrstuvwxyz012345");
        assert_eq!(profile.zones[2].label, "abcdefghijklmnopqrstuvwxyz0123 2");
        assert_eq!(profile.zones[3].label, "Right front");
        assert_eq!(profile.zones[1].layout, default_zones(count(4))[1].1);
        assert!(notes.contains(&MigrationNote::LayoutReset { zone: zone(0xA2) }));
        assert!(notes.contains(&MigrationNote::LabelRenamed {
            zone: zone(0xA3),
            label: "abcdefghijklmnopqrstuvwxyz0123 2".into()
        }));
        assert!(notes.contains(&MigrationNote::LabelRenamed {
            zone: zone(0xA4),
            label: "Right front".into()
        }));
        profile.validate().expect("valid");

        // What the repairs cannot fix is refused by name, never panicked on or half-loaded.
        let three_zones = edited(GOLDEN_V1_PROFILE, |v| {
            v["zones"].as_array_mut().expect("an array").pop();
        });
        assert!(invalid_reason(load_profile(&three_zones, file())).contains("InvalidZoneCount"));
        let long_name = edited(GOLDEN_V1_PROFILE, |v| v["name"] = json!("n".repeat(61)));
        assert!(invalid_reason(load_profile(&long_name, file())).contains("NameTooLong"));
        let no_classifier = edited(GOLDEN_V1_PROFILE, |v| {
            v.as_object_mut().expect("an object").remove("classifier");
        });
        assert!(matches!(
            load_profile(&no_classifier, file()),
            Err(PersistenceError::Json(_))
        ));
        let unknown_action = edited(GOLDEN_V1_PROFILE, |v| {
            v["zones"][0]["actions"][0] = json!({ "type": "Teleport" });
        });
        assert!(matches!(
            load_profile(&unknown_action, file()),
            Err(PersistenceError::Json(_))
        ));
    }

    #[test]
    fn st3_unknown_schema_versions_are_refused() {
        let mismatch = |result: Result<u32, PersistenceError>| match result {
            Err(PersistenceError::SchemaMismatch {
                file,
                found,
                expected,
            }) => {
                assert_eq!(file, "profile.json");
                assert_eq!(expected, 2);
                found
            }
            other => panic!("expected SchemaMismatch, got {other:?}"),
        };
        let v3 = edited(GOLDEN_V1_PROFILE, |v| v["schema_version"] = json!(3));
        let missing = edited(GOLDEN_V1_PROFILE, |v| {
            v.as_object_mut()
                .expect("an object")
                .remove("schema_version");
        });
        let null = edited(GOLDEN_V1_PROFILE, |v| v["schema_version"] = Value::Null);
        for (text, found) in [(&v3, 3), (&missing, 0), (&null, 0)] {
            assert_eq!(mismatch(load_profile(text, file()).map(|_| 0)), found);
            assert_eq!(mismatch(load_config(text, file()).map(|_| 0)), found);
        }
        let config_v3 = edited(GOLDEN_V1_CONFIG, |v| v["schema_version"] = json!(3));
        assert_eq!(mismatch(load_config(&config_v3, file()).map(|_| 0)), 3);
        assert_eq!(peek_schema_version(&missing, file()).expect("parses"), 0);
        assert_eq!(
            peek_schema_version(GOLDEN_V1_PROFILE, file()).expect("parses"),
            1
        );

        // Malformed JSON, or a version that is not a u32, is a Json error and never a guess.
        for text in [
            "{",
            "",
            r#"{"schema_version": "2"}"#,
            r#"{"schema_version": 4294967297}"#,
        ] {
            assert!(matches!(
                peek_schema_version(text, file()),
                Err(PersistenceError::Json(_))
            ));
            assert!(matches!(
                load_profile(text, file()),
                Err(PersistenceError::Json(_))
            ));
        }
    }

    #[test]
    fn st10_v2_validation_on_load() {
        let good = profile_to_json(&calibrated_profile(), file()).expect("serializes");

        // A wrong-length zone mean is refused, naming the model's typed error.
        let short_mean = edited(&good, |v| {
            v["calibration"]["model"]["zone_means"][0]
                .as_array_mut()
                .expect("an array")
                .pop();
        });
        let reason = invalid_reason(load_profile(&short_mean, file()));
        assert!(reason.contains("Dims"), "{reason}");

        // An off-canvas layout is refused, naming the profile's typed error.
        let off_canvas = edited(&good, |v| v["zones"][0]["layout"]["x"] = json!(1.5));
        let reason = invalid_reason(load_profile(&off_canvas, file()));
        assert!(reason.contains("InvalidLayout"), "{reason}");

        // A NaN cannot reach a file: it would be `null`, which fails to parse.
        let null = edited(&good, |v| v["zones"][0]["layout"]["x"] = Value::Null);
        assert!(matches!(
            load_profile(&null, file()),
            Err(PersistenceError::Json(_))
        ));

        // The same broken model from an older front end needs recalibration instead.
        let mut stale = calibrated_profile();
        let old = FRONT_END_VERSION - 1;
        model_mut(&mut stale).front_end_version = old;
        let stale_text = serde_json::to_string_pretty(&stale).expect("serializes");
        let stale_short = edited(&stale_text, |v| {
            v["calibration"]["model"]["zone_means"][0]
                .as_array_mut()
                .expect("an array")
                .pop();
        });
        for text in [stale_text.as_str(), stale_short.as_str()] {
            match load_profile(text, file()).expect("loads") {
                Loaded::Current(p) => assert_eq!(
                    p.calibration,
                    CalibrationState::NeedsRecalibration {
                        reason: RecalibrationReason::FrontEndChanged {
                            trained: old,
                            current: FRONT_END_VERSION
                        }
                    }
                ),
                other => panic!("expected Current, got {other:?}"),
            }
        }

        // A v2 config whose hotkey is Shift+A is refused as HotkeyWithoutModifier.
        let config = config_to_json(&Config::initial(), file()).expect("serializes");
        let shift_a = edited(&config, |v| {
            v["global_hotkey"] = json!({ "modifiers": ["Shift"], "key": "A" });
        });
        let reason = invalid_reason(load_config(&shift_a, file()));
        assert!(reason.contains("HotkeyWithoutModifier"), "{reason}");
        let zero = edited(&config, |v| v["eval_prompts_per_zone"] = json!(0));
        assert!(invalid_reason(load_config(&zero, file())).contains("ZeroEvalPrompts"));
    }
}
