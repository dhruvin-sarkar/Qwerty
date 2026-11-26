//! The coverage table of shell-v2 §6.7 as data (B:146): every key of the engine boundary the shell
//! receives or sends has an owner and a test, or a "not shown" reason (§6.6). CV1 serializes the
//! fixture boundary values and fails when their keys and type tags differ from these rows.
//!
//! A key is a path through the JSON form: `.field` for an object key and `:Variant` for a
//! `type` tag, from the roots `Snapshot`, `Event` and `Command`; e.g.
//! `Snapshot.capture:Failed.retry:Exhausted`. A row covers its key and everything below it, so the
//! rows are the leaves of the table's granularity: a key above a row (`Snapshot.capture`) is
//! structure, and a key that is neither a row, below one nor above one has no owner.

/// Who derives and shows a key, or why it is not shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Owner {
    /// The model package that derives it and the view package that shows (or sends) it.
    Shown(&'static [&'static str]),
    /// Not shown (or, for a command, not sent), with the reason (§6.6).
    NotShown(&'static str),
}

/// One row of §6.7.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Row {
    pub key: &'static str,
    pub owner: Owner,
    /// The tests that check the mapping; empty for a key nothing derives.
    pub tests: &'static [&'static str],
}

const fn shown(
    key: &'static str,
    owners: &'static [&'static str],
    tests: &'static [&'static str],
) -> Row {
    Row {
        key,
        owner: Owner::Shown(owners),
        tests,
    }
}

const fn not_shown(key: &'static str, reason: &'static str, tests: &'static [&'static str]) -> Row {
    Row {
        key,
        owner: Owner::NotShown(reason),
        tests,
    }
}

const NONE: &[&str] = &[];

const MODE_STATE: &[&str] = &["S2", "R7", "R8"];
const MODE_STATE_TESTS: &[&str] = &["LS1", "HS3"];
const SETTINGS: &[&str] = &["S9", "R13"];
const SETTINGS_TESTS: &[&str] = &["SE1", "SE3", "SE4", "SE5", "SC1"];
const BLOCKER: &[&str] = &["S2", "R8"];
const BLOCKER_TESTS: &[&str] = &["AT5", "AT7"];
const ENGINE_INTERNAL: &str = "engine-internal; the engine logs it (§6.6)";
const COUNTER: &str = "an engine counter (§6.6)";
const METER: &[&str] = &["S1", "S4", "R8"];
const METER_TESTS: &[&str] = &["TP2", "HM2"];
const SPECTRUM: &[&str] = &["S8", "R6", "R12"];
const SPECTRUM_TESTS: &[&str] = &["CH1", "DS1"];
const WAVEFORM: &[&str] = &["S8", "R6", "R10", "R12"];
const WAVEFORM_TESTS: &[&str] = &["DG6", "CH1"];
const PROFILE_COMMAND: &[&str] = &["S9", "W3", "R13"];
const PROFILE_COMMAND_TESTS: &[&str] = &["PF1", "FD1", "SC1"];
const CALIBRATION_COMMAND: &[&str] = &["S6", "R10", "S2", "R7"];
const CALIBRATION_COMMAND_TESTS: &[&str] = &["WZ1", "AT7", "SH2"];
const EVALUATION_RUN_COMMAND: &[&str] = &["S7", "R11"];
const EVALUATION_RUN_COMMAND_TESTS: &[&str] = &["EV3", "EV8"];
const REPORT_COMMAND: &[&str] = &["S7", "W3", "R11"];
const REPORT_COMMAND_TESTS: &[&str] = &["EV7", "ES1"];
const TEST_COMMAND: &[&str] = &["S5", "R9"];
const TEST_COMMAND_TESTS: &[&str] = &["AC2", "AC3"];
const INSPECTION_COMMAND: &[&str] = &["S8", "W3", "R12"];
const INSPECTION_COMMAND_TESTS: &[&str] = &["DG1", "DG3"];
const CLI_ONLY: &str = "not sent: CLI evidence commands (§6.6)";

/// The rows of §6.7, in its order.
pub const ROWS: &[Row] = &[
    // ---- Snapshot ----
    shown("Snapshot.revision", &["S0"], &["SQ1"]),
    shown("Snapshot.mode:Starting", MODE_STATE, MODE_STATE_TESTS),
    shown("Snapshot.mode:Idle", MODE_STATE, MODE_STATE_TESTS),
    shown("Snapshot.mode:Listening", MODE_STATE, MODE_STATE_TESTS),
    shown("Snapshot.mode:Paused", MODE_STATE, MODE_STATE_TESTS),
    shown("Snapshot.mode:Calibrating", &["S6", "R10"], &["WZ1"]),
    shown("Snapshot.mode:Evaluating", &["S7", "R11"], &["EV8"]),
    shown(
        "Snapshot.mode:Blocked.blocker:ProfileUnreadable",
        BLOCKER,
        BLOCKER_TESTS,
    ),
    shown(
        "Snapshot.mode:Blocked.blocker:NeedsRecalibration",
        BLOCKER,
        BLOCKER_TESTS,
    ),
    shown(
        "Snapshot.mode:Blocked.blocker:ProfileIncompatible",
        BLOCKER,
        BLOCKER_TESTS,
    ),
    shown(
        "Snapshot.mode:Blocked.blocker:MicUnavailable",
        BLOCKER,
        BLOCKER_TESTS,
    ),
    shown(
        "Snapshot.mode:Blocked.blocker:MicMismatch",
        BLOCKER,
        BLOCKER_TESTS,
    ),
    shown(
        "Snapshot.mode:Blocked.blocker:InputVetoUnavailable",
        BLOCKER,
        BLOCKER_TESTS,
    ),
    shown(
        "Snapshot.mode:Faulted",
        &["S9", "R13", "R7"],
        &["ER1", "SH4"],
    ),
    shown("Snapshot.desired", &["S2", "R7"], &["LS3"]),
    shown("Snapshot.config.theme", SETTINGS, SETTINGS_TESTS),
    shown(
        "Snapshot.config.notification_verbosity",
        SETTINGS,
        SETTINGS_TESTS,
    ),
    shown(
        "Snapshot.config.debug_capture_enabled",
        SETTINGS,
        SETTINGS_TESTS,
    ),
    shown(
        "Snapshot.config.input_veto_enabled",
        SETTINGS,
        SETTINGS_TESTS,
    ),
    shown(
        "Snapshot.config.start_with_windows",
        SETTINGS,
        SETTINGS_TESTS,
    ),
    shown(
        "Snapshot.config.global_hotkey",
        &["S9", "W1", "R13"],
        &["SE2", "HK1"],
    ),
    shown(
        "Snapshot.config.minimize_to_tray_on_close",
        &["S1", "S9", "R13"],
        &["WN3"],
    ),
    shown(
        "Snapshot.config.active_profile_id",
        &["S9", "R7", "W1"],
        &["PF1", "TM2"],
    ),
    shown(
        "Snapshot.config.eval_prompts_per_zone",
        &["S7", "R11"],
        &["ES1"],
    ),
    shown(
        "Snapshot.config.eval_background_check",
        &["S7", "R11"],
        &["ES1"],
    ),
    shown(
        "Snapshot.config.dismissed_notices",
        &["S4", "S5"],
        &["FR2", "ZN3"],
    ),
    shown(
        "Snapshot.config.window_placement",
        &["S1", "W2", "R7"],
        &["WN4", "PL1"],
    ),
    shown("Snapshot.config.text_zoom", &["S1"], &["TZ1"]),
    not_shown(
        "Snapshot.config.schema_version",
        "a file-format number",
        NONE,
    ),
    not_shown(
        "Snapshot.config_status:Loaded",
        "a normal start (first run is decided by an empty `profiles`, U:352)",
        NONE,
    ),
    not_shown(
        "Snapshot.config_status:Initial",
        "a normal start (first run is decided by an empty `profiles`, U:352)",
        NONE,
    ),
    not_shown("Snapshot.config_status:Migrated", "§6.6", NONE),
    shown(
        "Snapshot.config_status:Unreadable",
        &["S9", "R13"],
        &["ER1"],
    ),
    shown(
        "Snapshot.profiles",
        &["S9", "R13", "R7", "W1"],
        &["PF1", "TM2"],
    ),
    shown(
        "Snapshot.active_profile",
        &["S4", "S5", "R8", "R9"],
        &["HM1", "ZN1"],
    ),
    shown(
        "Snapshot.devices",
        &["S6", "S9", "R10", "R13"],
        &["WS1", "PF1"],
    ),
    shown(
        "Snapshot.capture:Off",
        &["S4", "S8", "R8", "R12"],
        &["HM2", "DG8"],
    ),
    shown(
        "Snapshot.capture:Running",
        &["S4", "S8", "R8", "R12"],
        &["HM2", "DG8"],
    ),
    shown(
        "Snapshot.capture:Failed.endpoint_id",
        &["S2", "R8"],
        &["AT7"],
    ),
    shown("Snapshot.capture:Failed.error", &["S2", "R8"], &["AT7"]),
    shown(
        "Snapshot.capture:Failed.retry:Scheduled",
        &["S2", "R8"],
        &["AT7"],
    ),
    shown(
        "Snapshot.capture:Failed.retry:Exhausted",
        &["S2", "R8"],
        &["AT7"],
    ),
    shown("Snapshot.veto:Disabled", &["S2"], &["LS1"]),
    shown("Snapshot.veto:NotNeeded", &["S2"], &["LS1"]),
    shown("Snapshot.veto:WaitingForShell", &["S2"], &["LS1"]),
    shown("Snapshot.veto:Registering", &["S2"], &["LS1"]),
    shown("Snapshot.veto:Active", &["S2"], &["LS1"]),
    shown("Snapshot.veto:Unavailable", &["S2"], &["LS1"]),
    shown(
        "Snapshot.recorder.state:Off",
        &["S9", "R7", "S2", "W1"],
        &["SE5", "TM1"],
    ),
    shown(
        "Snapshot.recorder.state:Recording",
        &["S9", "R7", "S2", "W1"],
        &["SE5", "TM1"],
    ),
    shown("Snapshot.recorder.state:Stopped", &["S9", "R13"], &["SE5"]),
    shown("Snapshot.recorder.state:Failed", &["S9", "R13"], &["SE5"]),
    shown("Snapshot.recorder.stored", &["S9", "R13"], &["SE5"]),
    not_shown("Snapshot.telemetry", "the shell's own policy (§6.6)", NONE),
    shown(
        "Snapshot.health.counters.input_vetoed",
        &["S4", "R8"],
        &["HM6"],
    ),
    shown(
        "Snapshot.health.counters.counters_since",
        &["S4", "R8"],
        &["HM6"],
    ),
    shown(
        "Snapshot.health.counters.capture_dropped_frames",
        &["S8", "R12"],
        &["DG8"],
    ),
    not_shown("Snapshot.health.counters.taps_fired", COUNTER, NONE),
    not_shown("Snapshot.health.counters.actions_failed", COUNTER, NONE),
    not_shown(
        "Snapshot.health.counters.capture_packets",
        "capture packet count: evidence for the CLI monitor only",
        NONE,
    ),
    not_shown("Snapshot.health.counters.controller_dropped", COUNTER, NONE),
    not_shown("Snapshot.health.counters.events_dropped", COUNTER, NONE),
    not_shown("Snapshot.health.counters.telemetry_dropped", COUNTER, NONE),
    not_shown("Snapshot.health.counters.input_dropped", COUNTER, NONE),
    not_shown("Snapshot.health.counters.decisions_dropped", COUNTER, NONE),
    not_shown("Snapshot.health.counters.dispatch_dropped", COUNTER, NONE),
    not_shown(
        "Snapshot.health.counters.recorder_dropped_frames",
        COUNTER,
        NONE,
    ),
    not_shown(
        "Snapshot.health.counters.recorder_dropped_events",
        COUNTER,
        NONE,
    ),
    not_shown("Snapshot.health.counters.stale_messages", COUNTER, NONE),
    not_shown("Snapshot.health.counters.lifecycle_lost", COUNTER, NONE),
    not_shown("Snapshot.health.counters.faults", COUNTER, NONE),
    not_shown(
        "Snapshot.health.latency",
        "the report's latency histogram covers it (§6.6)",
        NONE,
    ),
    shown(
        "Snapshot.health.warnings:CaptureOverflow",
        &["S2"],
        &["AT5"],
    ),
    shown(
        "Snapshot.health.warnings:LowSignal",
        &["S2"],
        &["AT3", "AT5"],
    ),
    shown(
        "Snapshot.health.warnings:RecognitionDrift",
        &["S2"],
        &["AT3", "AT5"],
    ),
    shown(
        "Snapshot.health.warnings:AutostartFailed",
        &["S2"],
        &["AT4"],
    ),
    shown("Snapshot.health.warnings:DiskSpaceLow", &["S2"], &["AT5"]),
    not_shown(
        "Snapshot.health.warnings:CaptureStalled",
        "the blocking MicUnavailable that follows is shown (§6.6)",
        &["AT5"],
    ),
    not_shown(
        "Snapshot.health.warnings:MmcssUnavailable",
        ENGINE_INTERNAL,
        &["AT5"],
    ),
    not_shown(
        "Snapshot.health.warnings:NoiseTrackerStarved",
        ENGINE_INTERNAL,
        &["AT5"],
    ),
    not_shown(
        "Snapshot.health.warnings:ControllerOverloaded",
        ENGINE_INTERNAL,
        &["AT5"],
    ),
    not_shown(
        "Snapshot.health.warnings:SelfOutputStuck",
        ENGINE_INTERNAL,
        &["AT5"],
    ),
    not_shown(
        "Snapshot.health.warnings:SpeechCallbackPanicked",
        ENGINE_INTERNAL,
        &["AT5"],
    ),
    not_shown(
        "Snapshot.health.warnings:EventsDropped",
        ENGINE_INTERNAL,
        &["AT5"],
    ),
    not_shown(
        "Snapshot.health.warnings:PlatformRebuilt",
        ENGINE_INTERNAL,
        &["AT5"],
    ),
    not_shown(
        "Snapshot.health.warnings:SpeechUnmatched",
        ENGINE_INTERNAL,
        &["AT5"],
    ),
    shown("Snapshot.activity.recent", &["S4", "R8"], &["HM3"]),
    shown("Snapshot.activity.attention", &["S2", "R8"], &["AT1"]),
    shown("Snapshot.pending_tests", &["S5", "R9"], &["AC3"]),
    shown("Snapshot.unsaved_report", &["S7", "R11"], &["EV8"]),
    shown("Snapshot.paths.data_dir", &["S9", "R13"], &["SE5", "ER1"]),
    shown("Snapshot.paths.logs", &["S9", "R13"], &["SE5", "ER1"]),
    shown("Snapshot.paths.recordings", &["S9", "R13"], &["SE5", "ER1"]),
    // ---- Events and telemetry frames ----
    shown("Event:StateChanged", &["S0"], &["SQ1"]),
    shown("Event:TapFired", &["S0", "R5a", "R8"], &["SQ2", "DV5"]),
    shown(
        "Event:TapRejected",
        &["S0", "S2", "R7"],
        &["SQ3", "LS4", "SH8"],
    ),
    shown("Event:GestureArmed", &["S8"], &["DG1"]),
    shown("Event:GestureCleared", &["S8"], &["DG1"]),
    shown("Event:ActionCompleted", &["S5", "R9"], &["AC2"]),
    shown("Event:ActionSkipped", &["S5", "R9"], &["AC2"]),
    shown("Event:CommandFinished", &["S5", "R9"], &["AC2"]),
    shown(
        "Event:CalibrationFeedback",
        &["S6", "R10"],
        &["WZ1", "WZ5", "WZ6"],
    ),
    shown(
        "Event:CalibrationProgress",
        &["S6", "R10"],
        &["WZ1", "WZ5", "WZ6"],
    ),
    shown(
        "Event:CalibrationEnded",
        &["S6", "R10"],
        &["WZ1", "WZ5", "WZ6"],
    ),
    shown("Event:EvaluationUpdate", &["S7", "R11"], &["EV8"]),
    shown("Event:EvaluationEnded", &["S7", "R11"], &["EV8"]),
    not_shown(
        "Event:ProfileMigrated",
        "logged by the shell at info (§6.6)",
        NONE,
    ),
    not_shown(
        "Event:ConfigMigrated",
        "logged by the shell at info (§6.6)",
        NONE,
    ),
    shown("Event:EngineFaulted", &["S9", "R13", "R7"], &["ER1", "SH4"]),
    shown("Event:Stopped", &["R7"], &["SH2"]),
    // SPEC: §6.7 has one row per frame kind; the frames' fields are rows here, so a field the
    // engine adds to a frame (as WP-E1 added `end`) fails CV1 until it has a row. Each field row
    // carries its frame's owners and tests.
    shown("Event:Telemetry.frame:Meter.rms", METER, METER_TESTS),
    shown("Event:Telemetry.frame:Meter.peak", METER, METER_TESTS),
    shown("Event:Telemetry.frame:Meter.noise", METER, METER_TESTS),
    shown(
        "Event:Telemetry.frame:Spectrum.end",
        SPECTRUM,
        SPECTRUM_TESTS,
    ),
    shown(
        "Event:Telemetry.frame:Spectrum.bands",
        SPECTRUM,
        SPECTRUM_TESTS,
    ),
    shown(
        "Event:Telemetry.frame:Spectrum.noise",
        SPECTRUM,
        SPECTRUM_TESTS,
    ),
    shown(
        "Event:Telemetry.frame:Spectrum.flux",
        SPECTRUM,
        SPECTRUM_TESTS,
    ),
    shown(
        "Event:Telemetry.frame:Spectrum.threshold",
        SPECTRUM,
        SPECTRUM_TESTS,
    ),
    shown(
        "Event:Telemetry.frame:Waveform.end",
        WAVEFORM,
        WAVEFORM_TESTS,
    ),
    shown(
        "Event:Telemetry.frame:Waveform.min",
        WAVEFORM,
        WAVEFORM_TESTS,
    ),
    shown(
        "Event:Telemetry.frame:Waveform.max",
        WAVEFORM,
        WAVEFORM_TESTS,
    ),
    shown(
        "Event:Telemetry.frame:Trace.entries",
        &["S8", "R12"],
        &["DG1"],
    ),
    // ---- Commands (sent, or "not sent") ----
    shown("Command:ShellReady", &["R7"], &["SH1"]),
    not_shown(
        "Command:GetSnapshot",
        "not sent: `handle.snapshot()` is the pull path",
        NONE,
    ),
    shown(
        "Command:Shutdown",
        &["R7", "W1"],
        &["SH2", "SH4", "SH7", "TY1"],
    ),
    shown("Command:SetListening", &["S2", "R7"], &["LS3"]),
    shown("Command:ToggleListening", &["W1"], &["HK1", "TY1"]),
    shown("Command:SetTelemetry", &["S1", "R7"], &["TP1", "SH5"]),
    shown("Command:ListDevices", &["S6", "R10"], &["WS1"]),
    shown("Command:RetryCapture", &["S2"], &["AT7"]),
    shown("Command:RetryInputVeto", &["S2"], &["AT7"]),
    shown(
        "Command:UpdateConfig",
        &["S9", "R13", "S1", "S4", "W1"],
        &["SE1", "SE2", "SE3", "SE4", "SE5", "WN4", "FR1", "TY1"],
    ),
    shown("Command:ResetConfig", &["S9", "R13"], &["ER1"]),
    shown("Command:ReconcileAutostart", &["S9", "R13", "S2"], &["SE3"]),
    shown(
        "Command:SwitchProfile",
        &["S9", "R7", "W1", "S2"],
        &["PF1", "TY1", "AT7"],
    ),
    shown(
        "Command:RenameProfile",
        PROFILE_COMMAND,
        PROFILE_COMMAND_TESTS,
    ),
    shown(
        "Command:DuplicateProfile",
        PROFILE_COMMAND,
        PROFILE_COMMAND_TESTS,
    ),
    shown(
        "Command:DeleteProfile",
        PROFILE_COMMAND,
        PROFILE_COMMAND_TESTS,
    ),
    shown(
        "Command:ExportProfile",
        PROFILE_COMMAND,
        PROFILE_COMMAND_TESTS,
    ),
    shown(
        "Command:PreviewImport",
        PROFILE_COMMAND,
        PROFILE_COMMAND_TESTS,
    ),
    shown(
        "Command:ImportProfile",
        PROFILE_COMMAND,
        PROFILE_COMMAND_TESTS,
    ),
    shown(
        "Command:RebindDevice",
        PROFILE_COMMAND,
        PROFILE_COMMAND_TESTS,
    ),
    shown(
        "Command:EditZone",
        &["S5", "R9"],
        &["ZN1", "AC4", "AC5", "LE1"],
    ),
    shown(
        "Command:StartCalibration",
        CALIBRATION_COMMAND,
        CALIBRATION_COMMAND_TESTS,
    ),
    shown(
        "Command:Calibration",
        CALIBRATION_COMMAND,
        CALIBRATION_COMMAND_TESTS,
    ),
    shown(
        "Command:CancelCalibration",
        CALIBRATION_COMMAND,
        CALIBRATION_COMMAND_TESTS,
    ),
    shown(
        "Command:StartEvaluation",
        EVALUATION_RUN_COMMAND,
        EVALUATION_RUN_COMMAND_TESTS,
    ),
    shown(
        "Command:UndoLastPrompt",
        EVALUATION_RUN_COMMAND,
        EVALUATION_RUN_COMMAND_TESTS,
    ),
    shown(
        "Command:CancelEvaluation",
        EVALUATION_RUN_COMMAND,
        EVALUATION_RUN_COMMAND_TESTS,
    ),
    shown(
        "Command:RetrySaveReport",
        EVALUATION_RUN_COMMAND,
        EVALUATION_RUN_COMMAND_TESTS,
    ),
    shown(
        "Command:DiscardUnsavedReport",
        EVALUATION_RUN_COMMAND,
        EVALUATION_RUN_COMMAND_TESTS,
    ),
    shown(
        "Command:ExportUnsavedReport",
        EVALUATION_RUN_COMMAND,
        EVALUATION_RUN_COMMAND_TESTS,
    ),
    shown(
        "Command:ListEvaluations",
        REPORT_COMMAND,
        REPORT_COMMAND_TESTS,
    ),
    shown(
        "Command:GetEvaluation",
        REPORT_COMMAND,
        REPORT_COMMAND_TESTS,
    ),
    shown(
        "Command:ExportEvaluation",
        REPORT_COMMAND,
        REPORT_COMMAND_TESTS,
    ),
    shown(
        "Command:DeleteEvaluation",
        REPORT_COMMAND,
        REPORT_COMMAND_TESTS,
    ),
    shown("Command:TestAction", TEST_COMMAND, TEST_COMMAND_TESTS),
    shown("Command:TestChain", TEST_COMMAND, TEST_COMMAND_TESTS),
    shown("Command:CancelTest", TEST_COMMAND, TEST_COMMAND_TESTS),
    shown("Command:DismissAttention", &["S2"], &["AT1"]),
    not_shown(
        "Command:ShellToast",
        "not sent: status toasts are silent (§6.6, U-40)",
        NONE,
    ),
    shown(
        "Command:GetInspection",
        INSPECTION_COMMAND,
        INSPECTION_COMMAND_TESTS,
    ),
    shown(
        "Command:ExportInspection",
        INSPECTION_COMMAND,
        INSPECTION_COMMAND_TESTS,
    ),
    shown(
        "Command:GetSoundMap",
        INSPECTION_COMMAND,
        INSPECTION_COMMAND_TESTS,
    ),
    not_shown("Command:SaveLatencyReport", CLI_ONLY, NONE),
    not_shown("Command:MarkSegment", CLI_ONLY, NONE),
    shown("Command:DeleteDebugRecordings", &["S9", "R13"], &["SE5"]),
];

/// The row for `key`, if `key` is one.
pub fn row(key: &str) -> Option<&'static Row> {
    ROWS.iter().find(|row| row.key == key)
}

/// True when `key` is `row` or lies below it.
pub fn covers(row: &str, key: &str) -> bool {
    key == row
        || key
            .strip_prefix(row)
            .is_some_and(|rest| rest.starts_with('.') || rest.starts_with(':'))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::BTreeSet;

    use serde::de::DeserializeOwned;
    use serde::Serialize;
    use serde_json::Value;

    use crate::engine::protocol::{
        Blocker, CaptureStatus, Command, ConfigStatus, Event, HealthWarning, Mode, RecorderState,
        RetryState, TelemetryFrame, VetoStatus,
    };
    use crate::shell::model::fixture;

    /// Every key below `path` in `value`: `.field` per object key, `:Variant` per `type` tag.
    fn collect(path: &str, value: &Value, out: &mut BTreeSet<String>) {
        match value {
            Value::Object(map) => {
                let path = match map.get("type") {
                    Some(Value::String(tag)) => {
                        let tagged = format!("{path}:{tag}");
                        out.insert(tagged.clone());
                        tagged
                    }
                    Some(other) => panic!("a non-string `type` at {path}: {other}"),
                    None => path.to_string(),
                };
                for (key, child) in map {
                    if key == "type" {
                        continue;
                    }
                    let child_path = format!("{path}.{key}");
                    out.insert(child_path.clone());
                    collect(&child_path, child, out);
                }
            }
            Value::Array(items) => {
                for item in items {
                    collect(path, item, out);
                }
            }
            Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
        }
    }

    fn json<T: Serialize>(value: &T) -> Value {
        serde_json::to_value(value).expect("serializes")
    }

    /// The variant names serde knows for an internally tagged enum, read from its "unknown
    /// variant" error, so a variant the engine adds is seen without a hand-kept list.
    fn variants<T: DeserializeOwned>() -> BTreeSet<String> {
        let error = match serde_json::from_str::<T>(r#"{"type":"?"}"#) {
            Ok(_) => panic!("no variant is named `?`"),
            Err(error) => error.to_string(),
        };
        let expected = error
            .split_once("expected")
            .unwrap_or_else(|| panic!("an unknown-variant error: {error}"))
            .1;
        expected
            .split('`')
            .skip(1)
            .step_by(2)
            .map(str::to_string)
            .collect()
    }

    fn tags<T: Serialize>(values: &[T]) -> BTreeSet<String> {
        values
            .iter()
            .map(|value| match json(value).get("type") {
                Some(Value::String(tag)) => tag.clone(),
                other => panic!("an internally tagged enum, got {other:?}"),
            })
            .collect()
    }

    fn every<T: Serialize + DeserializeOwned>(name: &str, values: &[T]) {
        assert_eq!(tags(values), variants::<T>(), "the fixture's {name} values");
    }

    #[test]
    fn cv1_coverage_table_matches_the_serialized_boundary() {
        // The fixture holds one value of every variant of each listed enum.
        every("Event", &fixture::events());
        every("TelemetryFrame", &fixture::telemetry_frames());
        every("Command", &fixture::commands());
        every("HealthWarning", &fixture::health_warnings());
        every("Blocker", &fixture::blockers());
        every("Mode", &fixture::modes());
        every("ConfigStatus", &fixture::config_statuses());
        every("CaptureStatus", &fixture::capture_statuses());
        every("RetryState", &fixture::retry_states());
        every("VetoStatus", &fixture::veto_statuses());
        every("RecorderState", &fixture::recorder_states());

        // Every key, each enum's values placed where the snapshot or event holds them.
        let mut keys = BTreeSet::new();
        collect("Snapshot", &json(&fixture::maximal_snapshot()), &mut keys);
        let mut place = |path: &str, values: Vec<Value>| {
            for value in values {
                collect(path, &value, &mut keys);
            }
        };
        place(
            "Snapshot.mode",
            fixture::modes().iter().map(json::<Mode>).collect(),
        );
        place(
            "Snapshot.mode:Blocked.blocker",
            fixture::blockers().iter().map(json::<Blocker>).collect(),
        );
        place(
            "Snapshot.config_status",
            fixture::config_statuses()
                .iter()
                .map(json::<ConfigStatus>)
                .collect(),
        );
        place(
            "Snapshot.capture",
            fixture::capture_statuses()
                .iter()
                .map(json::<CaptureStatus>)
                .collect(),
        );
        place(
            "Snapshot.capture:Failed.retry",
            fixture::retry_states()
                .iter()
                .map(json::<RetryState>)
                .collect(),
        );
        place(
            "Snapshot.veto",
            fixture::veto_statuses()
                .iter()
                .map(json::<VetoStatus>)
                .collect(),
        );
        place(
            "Snapshot.recorder.state",
            fixture::recorder_states()
                .iter()
                .map(json::<RecorderState>)
                .collect(),
        );
        place(
            "Snapshot.health.warnings",
            fixture::health_warnings()
                .iter()
                .map(json::<HealthWarning>)
                .collect(),
        );
        place(
            "Event",
            fixture::events().iter().map(json::<Event>).collect(),
        );
        place(
            "Event:Telemetry.frame",
            fixture::telemetry_frames()
                .iter()
                .map(json::<TelemetryFrame>)
                .collect(),
        );
        place(
            "Command",
            fixture::commands().iter().map(json::<Command>).collect(),
        );

        // The table is well formed: unique keys, none below another.
        let rows: BTreeSet<&str> = ROWS.iter().map(|row| row.key).collect();
        assert_eq!(rows.len(), ROWS.len(), "duplicate row keys");
        for a in ROWS {
            for b in ROWS {
                assert!(
                    a.key == b.key || !covers(a.key, b.key),
                    "row {} lies below row {}",
                    b.key,
                    a.key
                );
            }
            if let Owner::Shown(owners) = a.owner {
                assert!(!owners.is_empty(), "row {} has no owner", a.key);
            }
        }

        // Every row is a real key, and every key is a row, below one, or structure above one.
        let stale: Vec<&str> = rows
            .iter()
            .copied()
            .filter(|row| !keys.contains(*row))
            .collect();
        let unowned: Vec<&String> = keys
            .iter()
            .filter(|key| {
                let covered = ROWS.iter().any(|row| covers(row.key, key));
                let structure = ROWS
                    .iter()
                    .any(|row| row.key != key.as_str() && covers(key, row.key));
                !covered && !structure
            })
            .collect();
        assert!(
            stale.is_empty() && unowned.is_empty(),
            "§6.7 differs from the engine's keys.\nRows with no key: {stale:#?}\nKeys with no row: {unowned:#?}"
        );
    }
}
