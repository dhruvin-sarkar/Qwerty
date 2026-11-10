//! Offline replay of a recording through the v1 and/or v2 core (§1.18.3).
//!
//! Usage:
//!   cargo run -p qwerty-core --release --example replay -- <recording-dir>
//!       [--engine v1|v2|both] [--profile <profile.json>] [--out <report.json>]
//!       [--param <name>=<value>]... [--repair]
//!
//! The report is printed to stdout, or written to `--out`. `--param` overrides a `DetectorParams`
//! field by name (the value is a JSON number). `--repair` runs `wav::repair` on the recording's
//! audio first. Exit codes: 0 on success; 1 on a recording or processing error; 2 on bad arguments.

use qwerty_core::detector::DetectorParams;
use qwerty_core::migrate::{load_profile, Loaded};
use qwerty_core::recording::AUDIO_FILE;
use qwerty_core::replay::{run_replay, ReplayEngine, ReplayOptions, DEFAULT_REPLAY_BLOCK};
use qwerty_core::store::Profile;
use qwerty_core::training::TrainingParams;
use qwerty_core::wav;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str =
    "usage: replay <recording-dir> [--engine v1|v2|both] [--profile <profile.json>] \
                     [--out <report.json>] [--param <name>=<value>]... [--repair]";

struct Args {
    dir: PathBuf,
    engines: Vec<ReplayEngine>,
    profile: Option<PathBuf>,
    out: Option<PathBuf>,
    params: Vec<(String, String)>,
    repair: bool,
}

fn main() -> ExitCode {
    let args = match parse(std::env::args().skip(1).collect()) {
        Ok(args) => args,
        Err(message) => return bad_arguments(&message),
    };
    let detector = match override_params(&args.params) {
        Ok(detector) => detector,
        Err(message) => return bad_arguments(&message),
    };
    match run(&args, detector) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("replay: {message}");
            ExitCode::from(1)
        }
    }
}

fn bad_arguments(message: &str) -> ExitCode {
    eprintln!("replay: {message}\n{USAGE}");
    ExitCode::from(2)
}

fn parse(argv: Vec<String>) -> Result<Args, String> {
    let mut dir = None;
    let mut engines = vec![ReplayEngine::V2];
    let mut profile = None;
    let mut out = None;
    let mut params = Vec::new();
    let mut repair = false;
    let mut it = argv.into_iter();
    while let Some(arg) = it.next() {
        let mut value = |flag: &str| it.next().ok_or(format!("{flag} needs a value"));
        match arg.as_str() {
            "--engine" => {
                engines = match value("--engine")?.as_str() {
                    "v1" => vec![ReplayEngine::V1],
                    "v2" => vec![ReplayEngine::V2],
                    "both" => vec![ReplayEngine::V1, ReplayEngine::V2],
                    other => return Err(format!("unknown engine {other:?}")),
                }
            }
            "--profile" => profile = Some(PathBuf::from(value("--profile")?)),
            "--out" => out = Some(PathBuf::from(value("--out")?)),
            "--param" => {
                let p = value("--param")?;
                let Some((name, v)) = p.split_once('=') else {
                    return Err(format!("--param {p:?} is not <name>=<value>"));
                };
                params.push((name.to_string(), v.to_string()));
            }
            "--repair" => repair = true,
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
            path => {
                if dir.replace(PathBuf::from(path)).is_some() {
                    return Err("more than one recording directory".into());
                }
            }
        }
    }
    let Some(dir) = dir else {
        return Err("no recording directory".into());
    };
    Ok(Args {
        dir,
        engines,
        profile,
        out,
        params,
        repair,
    })
}

/// DetectorParams::DEFAULT with each `name=value` applied to the field of that name.
fn override_params(params: &[(String, String)]) -> Result<DetectorParams, String> {
    let mut json = serde_json::to_value(DetectorParams::DEFAULT).map_err(|e| e.to_string())?;
    let Some(fields) = json.as_object_mut() else {
        return Err("DetectorParams is not a JSON object".into());
    };
    for (name, value) in params {
        let Some(field) = fields.get_mut(name) else {
            return Err(format!("DetectorParams has no field {name:?}"));
        };
        let parsed: Value = serde_json::from_str(value)
            .map_err(|e| format!("--param {name}={value}: not a JSON value ({e})"))?;
        *field = parsed;
    }
    serde_json::from_value(json).map_err(|e| format!("--param: {e}"))
}

fn run(args: &Args, detector: DetectorParams) -> Result<(), String> {
    if args.repair {
        let audio = args.dir.join(AUDIO_FILE);
        let frames =
            wav::repair(&audio).map_err(|e| format!("repair {}: {e:?}", audio.display()))?;
        eprintln!("replay: repaired {} ({frames} frames)", audio.display());
    }
    let profile = match &args.profile {
        Some(path) => Some(load(path)?),
        None => None,
    };
    let options = ReplayOptions {
        engines: args.engines.clone(),
        detector,
        training: TrainingParams::DEFAULT,
        block: DEFAULT_REPLAY_BLOCK,
        profile,
    };
    let report = run_replay(&args.dir, &options).map_err(|e| format!("{e:?}"))?;
    let json = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    match &args.out {
        Some(path) => std::fs::write(path, json + "\n")
            .map_err(|e| format!("write {}: {e}", path.display()))?,
        None => println!("{json}"),
    }
    Ok(())
}

fn load(path: &Path) -> Result<Profile, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    match load_profile(&text, path).map_err(|e| format!("{}: {e:?}", path.display()))? {
        Loaded::Current(profile) | Loaded::Migrated { value: profile, .. } => Ok(profile),
    }
}
