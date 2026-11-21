//! Data-folder persistence (p2 §2.9.4): stateless file I/O for the config, profiles, evaluation
//! reports, latency reports and the exports. The controller (WP-C5) owns every piece of state;
//! this module only reads and writes the files. Paths are unchanged from v1:
//!
//! ```text
//! <data_dir>/config.json
//! <data_dir>/Profiles/<profile-id>.json
//! <data_dir>/Evaluations/<profile-id>/<report-id>.json        report-id = YYYYMMDDTHHMMSSmmmZ
//! <data_dir>/Latency/<YYYYMMDDTHHMMSSmmmZ>.json
//! <data_dir>/DebugCaptures/<YYYYMMDDTHHMMSSZ>/{audio.wav, session.json}
//! <data_dir>/Logs/qwerty.log, qwerty.1.log
//! ```

use std::ffi::OsStr;
use std::io::Write as _;
use std::path::{Component, Path, PathBuf};

use chrono::{DateTime, Utc};

use qwerty_core::assessment::{peek_legacy_report, EvaluationReport};
use qwerty_core::migrate::{load_config, load_profile, peek_schema_version, Loaded, MigrationNote};
use qwerty_core::profile::{PersistenceError, ProfileId};
use qwerty_core::store::{config_to_json, profile_to_json, Config, Profile};

use crate::engine::protocol::{
    ConfigStatus, LatencyReport, Paths, PersistenceFailure, PersistenceFailureKind, ReportContent,
    ReportFormat, ReportId, ReportSummary, TraceEntry,
};

const CONFIG_FILE: &str = "config.json";
const PROFILES_DIR: &str = "Profiles";
const EVALUATIONS_DIR: &str = "Evaluations";
const LATENCY_DIR: &str = "Latency";
const DEBUG_CAPTURES_DIR: &str = "DebugCaptures";
const LOGS_DIR: &str = "Logs";
const JSON_EXTENSION: &str = "json";
/// Report ids and latency file stems: the run's UTC time to the millisecond.
const REPORT_STAMP_FORMAT: &str = "%Y%m%dT%H%M%S%3fZ";
/// The suffix an unreadable config.json is renamed with: the reset's UTC time.
const UNREADABLE_STAMP_FORMAT: &str = "%Y%m%dT%H%M%SZ";
/// The schema version of a v1 report, listed as Legacy and never migrated (PARITY-35).
const LEGACY_REPORT_SCHEMA_VERSION: u32 = 1;

pub struct DataDir {
    root: PathBuf,
}

pub struct ConfigLoad {
    pub config: Config,
    pub status: ConfigStatus,
    /// Empty unless Migrated.
    pub notes: Vec<MigrationNote>,
}

pub struct ProfileScan {
    /// The file name inside Profiles/.
    pub file: String,
    pub loaded: Result<Loaded<Profile>, PersistenceFailure>,
}

pub enum StoredReport {
    Current(Box<EvaluationReport>),
    Legacy,
}

impl DataDir {
    /// create_dir_all of the root and of Profiles, Evaluations, Latency, DebugCaptures and Logs.
    pub fn open(root: PathBuf) -> Result<DataDir, PersistenceFailure> {
        std::fs::create_dir_all(&root).map_err(io_failure(&root))?;
        for dir in [
            PROFILES_DIR,
            EVALUATIONS_DIR,
            LATENCY_DIR,
            DEBUG_CAPTURES_DIR,
            LOGS_DIR,
        ] {
            std::fs::create_dir_all(root.join(dir)).map_err(io_failure(Path::new(dir)))?;
        }
        Ok(DataDir { root })
    }

    pub fn paths(&self) -> Paths {
        Paths {
            data_dir: self.root.clone(),
            logs: self.root.join(LOGS_DIR),
            recordings: self.root.join(DEBUG_CAPTURES_DIR),
        }
    }

    /// Missing: Config::initial(), written, Initial. v1: migrated and written back,
    /// Migrated{from_version}, with notes. Unreadable: Config::initial() in memory,
    /// Unreadable{error}, nothing written. Err only when a write-back fails.
    pub fn load_config(&self) -> Result<ConfigLoad, PersistenceFailure> {
        let file = Path::new(CONFIG_FILE);
        let text = match std::fs::read_to_string(self.root.join(file)) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let config = Config::initial();
                self.save_config(&config)?;
                return Ok(ConfigLoad {
                    config,
                    status: ConfigStatus::Initial,
                    notes: Vec::new(),
                });
            }
            Err(e) => return Ok(unreadable_config(failure(file, PersistenceError::Io(e)))),
        };
        match load_config(&text, file) {
            Ok(Loaded::Current(config)) => Ok(ConfigLoad {
                config,
                status: ConfigStatus::Loaded,
                notes: Vec::new(),
            }),
            Ok(Loaded::Migrated {
                value,
                from_version,
                notes,
            }) => {
                self.save_config(&value)?;
                Ok(ConfigLoad {
                    config: value,
                    status: ConfigStatus::Migrated { from_version },
                    notes,
                })
            }
            Err(e) => Ok(unreadable_config(failure(file, e))),
        }
    }

    pub fn save_config(&self, config: &Config) -> Result<(), PersistenceFailure> {
        let file = Path::new(CONFIG_FILE);
        let json = config_to_json(config, file).map_err(|e| failure(file, e))?;
        self.write(file, &json)
    }

    /// Renames config.json to config.json.unreadable-<YYYYMMDDTHHMMSSZ>, then writes and returns
    /// Config::initial().
    pub fn reset_config(&self, now: DateTime<Utc>) -> Result<Config, PersistenceFailure> {
        let file = Path::new(CONFIG_FILE);
        let kept = format!(
            "{CONFIG_FILE}.unreadable-{}",
            now.format(UNREADABLE_STAMP_FORMAT)
        );
        std::fs::rename(self.root.join(file), self.root.join(kept)).map_err(io_failure(file))?;
        let config = Config::initial();
        self.save_config(&config)?;
        Ok(config)
    }

    /// Every Profiles/*.json in file-name order; a Migrated file is written back before it is
    /// returned (Err only when that write fails). A file that fails to load is listed with its
    /// failure, never repaired or deleted.
    pub fn scan_profiles(&self) -> Result<Vec<ProfileScan>, PersistenceFailure> {
        let mut scans = Vec::new();
        for name in self.json_files(Path::new(PROFILES_DIR))? {
            let file = Path::new(PROFILES_DIR).join(&name);
            let loaded = match std::fs::read_to_string(self.root.join(&file)) {
                Ok(text) => load_profile(&text, &file).map_err(|e| failure(&file, e)),
                Err(e) => Err(failure(&file, PersistenceError::Io(e))),
            };
            if let Ok(Loaded::Migrated { value, .. }) = &loaded {
                let json = profile_to_json(value, &file).map_err(|e| failure(&file, e))?;
                self.write(&file, &json)?;
            }
            scans.push(ProfileScan { file: name, loaded });
        }
        Ok(scans)
    }

    /// Profiles/<id>.json
    pub fn save_profile(&self, profile: &Profile) -> Result<(), PersistenceFailure> {
        let file = profile_file(profile.id);
        let json = profile_to_json(profile, &file).map_err(|e| failure(&file, e))?;
        self.write(&file, &json)
    }

    /// The file and Evaluations/<id>/; idempotent.
    pub fn delete_profile(&self, id: ProfileId) -> Result<(), PersistenceFailure> {
        let file = profile_file(id);
        match std::fs::remove_file(self.root.join(&file)) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(failure(&file, PersistenceError::Io(e))),
        }
        let reports = reports_dir(id);
        match std::fs::remove_dir_all(self.root.join(&reports)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(failure(&reports, PersistenceError::Io(e))),
        }
    }

    /// Evaluations/<profile_id>/<report-id>.json, report-id = run_at as YYYYMMDDTHHMMSSmmmZ.
    pub fn save_report(&self, report: &EvaluationReport) -> Result<ReportId, PersistenceFailure> {
        let id = ReportId(report.run_at.format(REPORT_STAMP_FORMAT).to_string());
        let file = report_file(report.profile_id, &id);
        let json = report.to_json().map_err(|e| failure(&file, e))?;
        self.write(&file, &json)?;
        Ok(id)
    }

    /// Newest first; a missing folder is empty; schema 1 → Legacy via
    /// assessment::peek_legacy_report; a bad file → Unreadable.
    pub fn list_reports(&self, id: ProfileId) -> Result<Vec<ReportSummary>, PersistenceFailure> {
        let dir = reports_dir(id);
        let mut names = self.json_files(&dir)?;
        // The stems are UTC times in a fixed-width format, so name order is time order.
        names.reverse();
        let mut summaries = Vec::with_capacity(names.len());
        for name in names {
            let file = dir.join(&name);
            // INVARIANT: json_files returns only names whose extension is exactly "json".
            let stem = name.strip_suffix(".json").expect("a .json file name");
            let report = ReportId(stem.to_string());
            let read = std::fs::read_to_string(self.root.join(&file)).map_err(io_failure(&file));
            let content = match read.and_then(|text| parse_report(&text, &file)) {
                Ok(StoredReadable::Current(r)) => ReportContent::Readable {
                    run_at: r.run_at,
                    model_trained_at: r.meta.model_trained_at,
                    target: r.target,
                    overall: r.overall,
                },
                Ok(StoredReadable::Legacy {
                    run_at,
                    overall_accuracy,
                }) => ReportContent::Legacy {
                    run_at,
                    overall_accuracy,
                },
                Err(error) => ReportContent::Unreadable { error },
            };
            summaries.push(ReportSummary { report, content });
        }
        Ok(summaries)
    }

    /// None: no such file.
    pub fn load_report(
        &self,
        id: ProfileId,
        report: &ReportId,
    ) -> Result<Option<StoredReport>, PersistenceFailure> {
        if !is_report_id(report) {
            return Ok(None);
        }
        let file = report_file(id, report);
        let text = match std::fs::read_to_string(self.root.join(&file)) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(failure(&file, PersistenceError::Io(e))),
        };
        Ok(Some(match parse_report(&text, &file)? {
            StoredReadable::Current(r) => StoredReport::Current(r),
            StoredReadable::Legacy { .. } => StoredReport::Legacy,
        }))
    }

    /// false: no such file.
    pub fn delete_report(
        &self,
        id: ProfileId,
        report: &ReportId,
    ) -> Result<bool, PersistenceFailure> {
        if !is_report_id(report) {
            return Ok(false);
        }
        let file = report_file(id, report);
        match std::fs::remove_file(self.root.join(&file)) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(failure(&file, PersistenceError::Io(e))),
        }
    }

    /// Latency/<created_at as YYYYMMDDTHHMMSSmmmZ>.json; returns the full path written.
    pub fn save_latency_report(
        &self,
        report: &LatencyReport,
    ) -> Result<PathBuf, PersistenceFailure> {
        let file = Path::new(LATENCY_DIR).join(format!(
            "{}.{JSON_EXTENSION}",
            report.created_at.format(REPORT_STAMP_FORMAT)
        ));
        let json = serde_json::to_string_pretty(report)
            .map_err(|e| failure(&file, PersistenceError::Json(e)))?;
        self.write(&file, &json)?;
        Ok(self.root.join(file))
    }

    /// Writes `file` (relative to the root) through `write_atomic`, after create_dir_all of its
    /// parent, so a folder removed at runtime is recreated.
    fn write(&self, file: &Path, contents: &str) -> Result<(), PersistenceFailure> {
        write_file(&self.root.join(file), file, contents)
    }

    /// The `*.json` file names in `dir` (relative to the root), in name order; a missing folder
    /// is empty.
    fn json_files(&self, dir: &Path) -> Result<Vec<String>, PersistenceFailure> {
        let entries = match std::fs::read_dir(self.root.join(dir)) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(failure(dir, PersistenceError::Io(e))),
        };
        let mut names = Vec::new();
        for entry in entries {
            let entry = entry.map_err(io_failure(dir))?;
            let path = entry.path();
            let is_json = path.extension().is_some_and(|x| x == JSON_EXTENSION);
            if is_json && entry.file_type().map_err(io_failure(dir))?.is_file() {
                names.push(entry.file_name().to_string_lossy().into_owned());
            }
        }
        names.sort();
        Ok(names)
    }
}

/// A schema-1 file is Legacy (never migrated); anything else must be a valid schema-2 report.
fn parse_report(text: &str, file: &Path) -> Result<StoredReadable, PersistenceFailure> {
    let version = peek_schema_version(text, file).map_err(|e| failure(file, e))?;
    if version == LEGACY_REPORT_SCHEMA_VERSION {
        let legacy = peek_legacy_report(text, file).map_err(|e| failure(file, e))?;
        return Ok(StoredReadable::Legacy {
            run_at: legacy.run_at,
            overall_accuracy: legacy.overall_accuracy,
        });
    }
    let report = EvaluationReport::from_json(text, file).map_err(|e| failure(file, e))?;
    Ok(StoredReadable::Current(Box::new(report)))
}

/// A report file's readable content.
enum StoredReadable {
    Current(Box<EvaluationReport>),
    Legacy {
        run_at: DateTime<Utc>,
        overall_accuracy: f64,
    },
}

fn unreadable_config(error: PersistenceFailure) -> ConfigLoad {
    ConfigLoad {
        config: Config::initial(),
        status: ConfigStatus::Unreadable { error },
        notes: Vec::new(),
    }
}

fn profile_file(id: ProfileId) -> PathBuf {
    Path::new(PROFILES_DIR).join(format!("{id}.{JSON_EXTENSION}"))
}

fn reports_dir(id: ProfileId) -> PathBuf {
    Path::new(EVALUATIONS_DIR).join(id.to_string())
}

fn report_file(id: ProfileId, report: &ReportId) -> PathBuf {
    reports_dir(id).join(format!("{}.{JSON_EXTENSION}", report.0))
}

/// A report id names a file in the profile's folder: `<id>.json` is one plain file name whose
/// stem is the id, which is exactly what list_reports lists. An id with a path separator, a root
/// or a drive is refused, so an id from outside can never reach another path.
fn is_report_id(report: &ReportId) -> bool {
    let name = format!("{}.{JSON_EXTENSION}", report.0);
    let name = Path::new(&name);
    let mut parts = name.components();
    let one_file_name = matches!(
        (parts.next(), parts.next()),
        (Some(Component::Normal(_)), None)
    );
    one_file_name && name.file_stem() == Some(OsStr::new(&report.0))
}

/// Import and export touch only the path the shell's dialog chose (PARITY-27): never the data
/// folder, never a renamed path. load_profile (migrate, normalize, validate); writes nothing.
pub fn read_import(path: &Path) -> Result<Loaded<Profile>, PersistenceFailure> {
    let text = std::fs::read_to_string(path).map_err(io_failure(path))?;
    load_profile(&text, path).map_err(|e| failure(path, e))
}

pub fn export_profile(profile: &Profile, path: &Path) -> Result<(), PersistenceFailure> {
    let json = profile_to_json(profile, path).map_err(|e| failure(path, e))?;
    write_file(path, path, &json)
}

/// Json: the report; Csv: `to_csv`; PromptsCsv: `to_prompts_csv` (XR-UI-07).
pub fn export_report(
    report: &EvaluationReport,
    format: ReportFormat,
    path: &Path,
) -> Result<(), PersistenceFailure> {
    let text = match format {
        ReportFormat::Json => report.to_json().map_err(|e| failure(path, e))?,
        ReportFormat::Csv => report.to_csv(),
        ReportFormat::PromptsCsv => report.to_prompts_csv(),
    };
    write_file(path, path, &text)
}

/// The inspection log as a JSON array.
pub fn export_inspection(entries: &[TraceEntry], path: &Path) -> Result<(), PersistenceFailure> {
    let json = serde_json::to_string_pretty(entries)
        .map_err(|e| failure(path, PersistenceError::Json(e)))?;
    write_file(path, path, &json)
}

/// The one PersistenceError → PersistenceFailure mapping: Io → Io, Json → Json,
/// SchemaMismatch{found, expected} → the same, InvalidData → InvalidData. `file` = the path
/// relative to the data folder, or the full path outside it; `detail` = the error's Display text
/// (shown to the user, never logged). Every std::io::Error of a persist operation goes through
/// PersistenceError::Io first.
pub fn failure(file: &Path, error: PersistenceError) -> PersistenceFailure {
    let detail = error.to_string();
    let kind = match error {
        PersistenceError::Io(_) => PersistenceFailureKind::Io,
        PersistenceError::Json(_) => PersistenceFailureKind::Json,
        PersistenceError::SchemaMismatch {
            found, expected, ..
        } => PersistenceFailureKind::SchemaMismatch { found, expected },
        PersistenceError::InvalidData { .. } => PersistenceFailureKind::InvalidData,
    };
    PersistenceFailure {
        file: file.display().to_string(),
        kind,
        detail,
    }
}

fn io_failure(file: &Path) -> impl FnOnce(std::io::Error) -> PersistenceFailure + '_ {
    move |e| failure(file, PersistenceError::Io(e))
}

/// Writes `path` through `write_atomic` after create_dir_all of its parent (a regular file in a
/// folder's place fails the write, EN10). `shown` is the path a failure names.
fn write_file(path: &Path, shown: &Path, contents: &str) -> Result<(), PersistenceFailure> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(io_failure(shown))?;
    }
    write_atomic(path, contents).map_err(io_failure(shown))
}

/// `profile::write_atomic`'s procedure: write a sibling temp file, sync it to disk, then rename it
/// over the target, so a crash or a full disk mid-write leaves the previous file whole.
// SPEC: §2.9.4 routes every write through `profile::write_atomic`, but the merged core declares it
// `pub(crate)` and this package may not edit qwerty-core, so the same steps are repeated here. The
// temp file is the target's name plus ".tmp" (core: the extension replaced by "json.tmp"), so an
// export's temp file never takes another file's name.
fn write_atomic(path: &Path, contents: &str) -> std::io::Result<()> {
    let Some(name) = path.file_name() else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "the path names no file",
        ));
    };
    let mut tmp_name = name.to_os_string();
    tmp_name.push(".tmp");
    let tmp = path.with_file_name(tmp_name);
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(contents.as_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::logfile::{RotatingLog, LOG_FILE};
    use qwerty_core::assessment::{
        EvaluationSession, LatencyStats, PromptSchedule, ReportMeta, ScriptedNegative,
        ScriptedPrompt, ZoneRef,
    };
    use qwerty_core::audio::{AudioFormat, CaptureMode, DeviceIdentity};
    use qwerty_core::descriptor::{Descriptor, DescriptorLayout, DESCRIPTOR_DIMS};
    use qwerty_core::detector::GateReport;
    use qwerty_core::model::{BackgroundRegime, ModelRejectReason, ModelScores};
    use qwerty_core::pipeline::{DecisionOutcome, TapDecision};
    use qwerty_core::profile::ZoneId;
    use qwerty_core::recording::NegativeClass;
    use qwerty_core::store::ZoneCount;
    use qwerty_core::training::{NegativeCount, NegativeSource};
    use qwerty_core::units::{
        CandidateId, ChannelCount, Decibels, HostTime, Milliseconds, SampleIndex, SampleRate,
        ZoneIndex,
    };
    use std::collections::BTreeMap;

    /// A fresh directory under the temp dir, removed when dropped.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> TempDir {
            let dir = std::env::temp_dir().join(format!("qwerty-c4-{name}-{}", std::process::id()));
            if dir.exists() {
                std::fs::remove_dir_all(&dir).expect("clear the old test directory");
            }
            std::fs::create_dir_all(&dir).expect("make the test directory");
            TempDir(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            // Cleanup only: a leftover temp folder fails nothing.
            if let Err(e) = std::fs::remove_dir_all(&self.0) {
                eprintln!("could not remove {}: {e}", self.0.display());
            }
        }
    }

    /// Every file and folder under `root`, relative and with '/' separators, with each file's
    /// bytes (None for a folder).
    fn tree(root: &Path) -> BTreeMap<String, Option<Vec<u8>>> {
        fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Option<Vec<u8>>>) {
            for entry in std::fs::read_dir(dir).expect("lists the folder") {
                let path = entry.expect("an entry").path();
                let rel = path
                    .strip_prefix(root)
                    .expect("under the root")
                    .to_string_lossy()
                    .replace('\\', "/");
                if path.is_dir() {
                    out.insert(rel, None);
                    walk(root, &path, out);
                } else {
                    out.insert(rel, Some(std::fs::read(&path).expect("reads the file")));
                }
            }
        }
        let mut out = BTreeMap::new();
        walk(root, root, &mut out);
        out
    }

    fn files(root: &Path) -> Vec<String> {
        tree(root)
            .into_iter()
            .filter_map(|(path, bytes)| bytes.map(|_| path))
            .collect()
    }

    fn ts(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s)
            .expect("a valid RFC 3339 time")
            .with_timezone(&Utc)
    }

    fn profile_id(n: u32) -> ProfileId {
        serde_json::from_str(&format!("\"00000000-0000-0000-0000-{n:012x}\"")).expect("a uuid")
    }

    fn zone_id(n: u32) -> ZoneId {
        serde_json::from_str(&format!("\"00000000-0000-0000-0000-{n:012x}\"")).expect("a uuid")
    }

    fn profile() -> Profile {
        let mut p = Profile::new(
            "Desk".into(),
            ZoneCount::new(2).expect("2 zones"),
            ts("2026-11-01T09:00:00Z"),
        )
        .expect("a valid new profile");
        p.id = profile_id(0x57);
        p
    }

    fn identity() -> DeviceIdentity {
        DeviceIdentity {
            endpoint_id: "{0.0.1.00000000}.{c4-microphone}".into(),
            name: "USB Microphone".into(),
            format: AudioFormat {
                rate: SampleRate::new(48_000).expect("48 kHz"),
                channels: ChannelCount::new(1).expect("mono"),
            },
            capture_mode: CaptureMode::Raw,
        }
    }

    fn at_ms(ms: i64) -> HostTime {
        HostTime(1_000_000_000 + ms * 10_000)
    }

    fn scores(best: u16) -> ModelScores {
        let d2 = if best == 0 {
            vec![1.0, 9.0]
        } else {
            vec![9.0, 1.0]
        };
        ModelScores {
            d2,
            best: ZoneIndex(best),
            second: ZoneIndex(1 - best),
            d2_min: 1.0,
            margin: 8.0,
            llr_bg: 3.5,
            posterior_best: 0.875,
        }
    }

    fn descriptor() -> Descriptor {
        let mut values = [0.0f32; DESCRIPTOR_DIMS];
        for (i, v) in values.iter_mut().enumerate() {
            *v = i as f32 * 0.25 - 3.0;
        }
        Descriptor {
            layout: DescriptorLayout::Mono { a: 0 },
            values,
        }
    }

    /// A decision on a tap at `t0_ms`, final 45 ms later.
    fn decision(t0_ms: i64, outcome: DecisionOutcome) -> TapDecision {
        let onset = SampleIndex(t0_ms as u64 * 48);
        TapDecision {
            candidate: CandidateId(t0_ms as u64),
            onset,
            onset_time: Some(at_ms(t0_ms)),
            emitted_at: SampleIndex(onset.0 + 2_000),
            decided_at: at_ms(t0_ms + 45),
            gates: GateReport {
                snr: Decibels::new(24.5),
                broadband_fraction: 0.625,
                valid_bands: 6,
                decay_fraction: 0.75,
                clipped: false,
                trigger_flux: 2.5,
            },
            outcome,
        }
    }

    /// A 2-zone run of 4 prompts (3 correct, 1 rejected) and a short Everyday phase, for the
    /// profile of `profile()`.
    fn report() -> EvaluationReport {
        let zones: Vec<ZoneRef> = [(zone_id(0xA1), "Left"), (zone_id(0xA2), "Right")]
            .into_iter()
            .map(|(id, label)| ZoneRef {
                id,
                label: label.into(),
            })
            .collect();
        let prompts = (0..4)
            .map(|i| ScriptedPrompt {
                zone: ZoneIndex(i % 2),
                opened_at: at_ms(2000 * i64::from(i)),
                deadline: at_ms(2000 * i64::from(i) + 1500),
                reopened: false,
            })
            .collect();
        let negatives = vec![ScriptedNegative {
            class: NegativeClass::Everyday,
            start: at_ms(10_000),
            end: at_ms(14_000),
        }];
        let mut session = EvaluationSession::new(
            zones,
            PromptSchedule::Scripted { prompts, negatives },
            at_ms(0),
        )
        .expect("a valid script");
        let accepted = |zone: u16| DecisionOutcome::Accepted {
            zone: ZoneIndex(zone),
            descriptor: descriptor(),
            scores: scores(zone),
        };
        let outcomes = [
            accepted(0),
            accepted(1),
            accepted(0),
            DecisionOutcome::Rejected {
                reason: ModelRejectReason::Ambiguous,
                descriptor: descriptor(),
                scores: scores(0),
            },
        ];
        let mut events = Vec::new();
        for (i, outcome) in outcomes.into_iter().enumerate() {
            let opened = 2000 * i as i64;
            session.tick(at_ms(opened), &mut events);
            session.on_decision(&decision(opened + 400, outcome), &mut events);
        }
        session.tick(at_ms(14_000), &mut events);
        let meta = ReportMeta {
            profile_id: profile_id(0x57),
            device: identity(),
            front_end_version: 1,
            model_trained_at: ts("2026-11-01T09:30:00Z"),
            zone_taps: vec![10, 10],
            negatives: vec![NegativeCount {
                source: NegativeSource::IgnoreStep,
                clips: 12,
            }],
            background_regime: BackgroundRegime::PooledAll {
                negative_clips: 12,
                needed: 24,
            },
            input_veto_enabled: true,
        };
        session
            .finish(meta, ts("2026-11-03T10:00:00.250Z"))
            .expect("the run finished")
    }

    fn latency_report() -> LatencyReport {
        let stats = LatencyStats::from_samples(&[40.0, 45.0, 50.0], 0);
        LatencyReport {
            schema_version: 1,
            created_at: ts("2026-11-04T12:30:45.125Z"),
            profile_id: profile_id(0x57),
            device: identity(),
            end_to_end: stats.clone(),
            decision: stats.clone(),
            controller: stats.clone(),
            dispatch: stats,
            budget_p50: Milliseconds::new(70.0),
            budget_p95: Milliseconds::new(100.0),
            passed: false,
        }
    }

    /// A hand-written v1 profile (schema version 1), never real user data: zone a1 chains
    /// [VisualOnly, Speak] and zones a1 and a2 are labelled "Desk" and "desk".
    const GOLDEN_V1_PROFILE: &str = r#"{
  "schema_version": 1,
  "id": "00000000-0000-0000-0000-0000000000b3",
  "name": "Home desk",
  "created_at": "2026-07-21T12:00:00Z",
  "updated_at": "2026-07-22T08:30:00Z",
  "device_fingerprint": { "device_name": "Built-in Microphone", "capability_hash": "sr48000" },
  "sensing_mode": "Passive",
  "zones": [
    {
      "id": "00000000-0000-0000-0000-0000000000a1",
      "label": "Desk",
      "layout": { "x": 0.03, "y": 0.08, "width": 0.29, "height": 0.4 },
      "actions": [{ "type": "VisualOnly" }, { "type": "Speak", "text": "hello" }],
      "calibration_sample_count": 10
    },
    {
      "id": "00000000-0000-0000-0000-0000000000a2",
      "label": "desk",
      "layout": { "x": 0.68, "y": 0.08, "width": 0.29, "height": 0.4 },
      "actions": [],
      "calibration_sample_count": 12
    }
  ],
  "classifier": { "feature_mean": [0.0, 1.0] }
}"#;

    #[test]
    fn ps1_golden_paths() {
        let tmp = TempDir::new("ps1");
        let root = tmp.0.join("data");
        let data = DataDir::open(root.clone()).expect("opens");
        assert_eq!(
            data.paths(),
            Paths {
                data_dir: root.clone(),
                logs: root.join("Logs"),
                recordings: root.join("DebugCaptures"),
            }
        );
        for dir in [
            "Profiles",
            "Evaluations",
            "Latency",
            "DebugCaptures",
            "Logs",
        ] {
            assert!(root.join(dir).is_dir(), "open creates {dir}/");
        }

        data.save_config(&Config::initial())
            .expect("saves the config");
        let profile = profile();
        data.save_profile(&profile).expect("saves the profile");
        let report = report();
        let id = data.save_report(&report).expect("saves the report");
        assert_eq!(id, ReportId("20261103T100000250Z".into()));
        let latency = data
            .save_latency_report(&latency_report())
            .expect("saves the latency report");
        assert_eq!(
            latency,
            root.join("Latency").join("20261104T123045125Z.json")
        );
        let log = RotatingLog::new(&data.paths().logs).expect("opens the log");
        drop(log);

        let pid = "00000000-0000-0000-0000-000000000057";
        assert_eq!(
            files(&root),
            [
                format!("Evaluations/{pid}/20261103T100000250Z.json"),
                "Latency/20261104T123045125Z.json".to_string(),
                format!("Logs/{LOG_FILE}"),
                format!("Profiles/{pid}.json"),
                "config.json".to_string(),
            ]
        );
        // What was written reads back as what was saved.
        let loaded = data.load_config().expect("loads");
        assert_eq!(loaded.status, ConfigStatus::Loaded);
        assert_eq!(loaded.config, Config::initial());
        let scans = data.scan_profiles().expect("scans");
        assert_eq!(scans.len(), 1);
        assert_eq!(scans[0].file, format!("{pid}.json"));
        assert_eq!(scans[0].loaded, Ok(Loaded::Current(profile)));
        match data.load_report(report.profile_id, &id).expect("loads") {
            Some(StoredReport::Current(back)) => assert_eq!(*back, report),
            Some(StoredReport::Legacy) | None => panic!("expected the saved report"),
        }
        let listed = data.list_reports(report.profile_id).expect("lists");
        assert_eq!(
            listed,
            [ReportSummary {
                report: id,
                content: ReportContent::Readable {
                    run_at: report.run_at,
                    model_trained_at: report.meta.model_trained_at,
                    target: report.target,
                    overall: report.overall.clone(),
                },
            }]
        );
    }

    #[test]
    fn ps2_exports_write_exactly_the_given_path() {
        let tmp = TempDir::new("ps2");
        let out = tmp.0.join("chosen folder");
        let profile = profile();
        let report = report();

        // A missing parent folder is created; an existing file is replaced, never suffixed.
        let profile_path = out.join("nested").join("my desk.json");
        export_profile(&profile, &profile_path).expect("exports the profile");
        std::fs::write(&profile_path, "old").expect("plants an old file");
        export_profile(&profile, &profile_path).expect("replaces it");
        let expected = profile_to_json(&profile, &profile_path).expect("serializes");
        assert_eq!(
            std::fs::read_to_string(&profile_path).expect("reads"),
            expected
        );

        let exports = [
            (
                ReportFormat::Json,
                "report.json",
                report.to_json().expect("valid"),
            ),
            (ReportFormat::Csv, "report.csv", report.to_csv()),
            (
                ReportFormat::PromptsCsv,
                "prompts.csv",
                report.to_prompts_csv(),
            ),
        ];
        for (format, name, text) in &exports {
            let path = out.join(name);
            std::fs::write(&path, "old").expect("plants an old file");
            export_report(&report, *format, &path).expect("exports the report");
            assert_eq!(&std::fs::read_to_string(&path).expect("reads"), text);
        }

        // Nothing else was written: no suffixed copy, no renamed file, no temp file left.
        assert_eq!(
            files(&out),
            [
                "nested/my desk.json",
                "prompts.csv",
                "report.csv",
                "report.json"
            ]
        );
    }

    #[test]
    fn ps3_read_import_migrates_and_writes_nothing() {
        let tmp = TempDir::new("ps3");
        let data = DataDir::open(tmp.0.join("data")).expect("opens");
        let source = tmp.0.join("import").join("v1 profile.json");
        std::fs::create_dir_all(source.parent().expect("a parent")).expect("makes the folder");
        std::fs::write(&source, GOLDEN_V1_PROFILE).expect("writes the source");
        let before = tree(&tmp.0);

        match read_import(&source).expect("imports") {
            Loaded::Migrated {
                value,
                from_version,
                notes,
            } => {
                assert_eq!(from_version, 1);
                assert_eq!(value.name, "Home desk");
                assert_eq!(
                    notes,
                    [
                        MigrationNote::VisualOnlyRemoved {
                            zone: zone_id(0xA1),
                            count: 1
                        },
                        MigrationNote::LabelRenamed {
                            zone: zone_id(0xA2),
                            label: "desk 2".into()
                        },
                    ]
                );
            }
            Loaded::Current(_) => panic!("a v1 profile migrates"),
        }
        assert_eq!(
            tree(&tmp.0),
            before,
            "the source and the data folder are unchanged"
        );
        assert!(data.scan_profiles().expect("scans").is_empty());

        // A file that parses but fails validation is InvalidData, named by its full path.
        let invalid = tmp.0.join("import").join("invalid.json");
        let mut value: serde_json::Value =
            serde_json::from_str(&profile_to_json(&profile(), &invalid).expect("serializes"))
                .expect("valid JSON");
        value["name"] = serde_json::Value::String(String::new());
        std::fs::write(&invalid, value.to_string()).expect("writes");
        let error = read_import(&invalid).expect_err("an empty name is refused");
        assert_eq!(error.kind, PersistenceFailureKind::InvalidData);
        assert_eq!(error.file, invalid.display().to_string());
    }

    #[test]
    fn ps4_delete_profile_is_idempotent() {
        let tmp = TempDir::new("ps4");
        let root = tmp.0.join("data");
        let data = DataDir::open(root.clone()).expect("opens");
        let profile = profile();
        let mut other = profile.clone();
        other.id = profile_id(0x58);
        data.save_profile(&profile).expect("saves");
        data.save_profile(&other).expect("saves");
        data.save_report(&report()).expect("saves the report");
        let pid = "00000000-0000-0000-0000-000000000057";
        assert!(root.join("Evaluations").join(pid).is_dir());

        data.delete_profile(profile.id).expect("deletes");
        assert!(!root.join("Profiles").join(format!("{pid}.json")).exists());
        assert!(!root.join("Evaluations").join(pid).exists());
        assert!(root
            .join("Profiles")
            .join(format!("{}.json", other.id))
            .is_file());

        let after = tree(&root);
        data.delete_profile(profile.id)
            .expect("a second delete is Ok");
        assert_eq!(tree(&root), after, "a second delete changes nothing");
    }

    #[test]
    fn ps5_corrupt_config_is_unreadable_until_reset() {
        let tmp = TempDir::new("ps5");
        let root = tmp.0.join("data");
        let data = DataDir::open(root.clone()).expect("opens");
        let corrupt = b"{ \"schema_version\": 2, \"theme\": ";
        std::fs::write(root.join("config.json"), corrupt).expect("plants a corrupt config");
        let before = tree(&root);

        let load = data
            .load_config()
            .expect("an unreadable config is not an Err");
        assert_eq!(load.config, Config::initial());
        assert!(load.notes.is_empty());
        match load.status {
            ConfigStatus::Unreadable { error } => {
                assert_eq!(error.kind, PersistenceFailureKind::Json);
                assert_eq!(error.file, "config.json");
            }
            other => panic!("expected Unreadable, got {other:?}"),
        }
        assert_eq!(tree(&root), before, "loading wrote nothing");

        let config = data
            .reset_config(ts("2026-11-05T07:08:09.999Z"))
            .expect("resets");
        assert_eq!(config, Config::initial());
        let kept = root.join("config.json.unreadable-20261105T070809Z");
        assert_eq!(std::fs::read(&kept).expect("kept"), corrupt);
        let reloaded = data.load_config().expect("loads");
        assert_eq!(reloaded.status, ConfigStatus::Loaded);
        assert_eq!(reloaded.config, Config::initial());
    }

    /// Every id list_reports gives names a file that load_report reads and delete_report
    /// removes, whatever the file is called (here the copy Explorer makes of a report).
    #[test]
    fn listed_report_ids_load_and_delete() {
        let tmp = TempDir::new("listed-ids");
        let root = tmp.0.join("data");
        let data = DataDir::open(root.clone()).expect("opens");
        let report = report();
        let saved = data.save_report(&report).expect("saves the report");
        let dir = root.join("Evaluations").join(report.profile_id.to_string());
        let copy = ReportId(format!("{} - Copy", saved.0));
        std::fs::copy(
            dir.join(format!("{}.json", saved.0)),
            dir.join(format!("{}.json", copy.0)),
        )
        .expect("copies the report");

        let listed: Vec<ReportId> = data
            .list_reports(report.profile_id)
            .expect("lists")
            .into_iter()
            .map(|summary| summary.report)
            .collect();
        assert_eq!(listed, [saved, copy]);
        for id in &listed {
            match data.load_report(report.profile_id, id).expect("loads") {
                Some(StoredReport::Current(back)) => assert_eq!(*back, report),
                Some(StoredReport::Legacy) | None => panic!("{id:?} is listed, so it loads"),
            }
        }
        for id in &listed {
            assert!(
                data.delete_report(report.profile_id, id).expect("deletes"),
                "{id:?} is listed, so it is deleted"
            );
        }
        assert!(data
            .list_reports(report.profile_id)
            .expect("lists")
            .is_empty());
    }

    /// An id that is not the stem of one plain file name in the profile's folder reaches
    /// nothing: a file one folder up, a file elsewhere by its full path, and the ".json" file
    /// the listing never shows all stay as they are.
    #[test]
    fn report_ids_outside_the_folder_reach_nothing() {
        let tmp = TempDir::new("outside-ids");
        let root = tmp.0.join("data");
        let data = DataDir::open(root.clone()).expect("opens");
        let report = report();
        data.save_report(&report).expect("saves the report");
        let json = report.to_json().expect("valid");
        let dir = root.join("Evaluations").join(report.profile_id.to_string());
        let planted = [
            root.join("Evaluations").join("outside.json"),
            tmp.0.join("elsewhere.json"),
            dir.join(".json"),
        ];
        for path in &planted {
            std::fs::write(path, &json).expect("plants a report");
        }

        let mut ids = vec![
            "../outside".to_string(),
            tmp.0.join("elsewhere").display().to_string(),
            String::new(),
        ];
        if cfg!(windows) {
            ids.push("..\\outside".to_string());
        }
        for id in ids.into_iter().map(ReportId) {
            let loaded = data.load_report(report.profile_id, &id).expect("no error");
            assert!(loaded.is_none(), "{id:?} loads nothing");
            let deleted = data
                .delete_report(report.profile_id, &id)
                .expect("no error");
            assert!(!deleted, "{id:?} deletes nothing");
        }
        for path in &planted {
            assert!(path.is_file(), "{} is untouched", path.display());
        }
    }
}
