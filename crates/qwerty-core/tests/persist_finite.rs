//! ST7 (§1.16, §0.5, RUST-RT-16): no non-finite value is ever written. A fixture of every
//! persisted type (`Config`, `Profile` with a `TrainedModel`, `EvaluationReport`, `Sidecar`,
//! `ReplayReport`) serializes with no `null` where the schema has a number, and round-trips
//! text-identically.

use chrono::{DateTime, Utc};
use qwerty_core::assessment::{
    EvaluationReport, EvaluationSession, PromptSchedule, ReportMeta, ScriptedNegative,
    ScriptedPrompt, ZoneRef,
};
use qwerty_core::audio::{
    AudioBlock, AudioFormat, BlockTime, CaptureMode, ChannelLayout, DeviceIdentity,
};
use qwerty_core::descriptor::{Descriptor, DescriptorLayout, DESCRIPTOR_DIMS};
use qwerty_core::detector::GateReport;
use qwerty_core::migrate::{load_config, load_profile, Loaded};
use qwerty_core::model::{ModelRejectReason, ModelScores, TrainedModel};
use qwerty_core::pipeline::{DecisionOutcome, TapDecision};
use qwerty_core::profile::{ActionSpec, KeyCombo, Modifier, ProfileId, Theme, ZoneId};
use qwerty_core::recording::{
    NegativeClass, RecordingReader, RecordingWriter, SegmentKind, SidecarEvent, SidecarHeader,
    SIDECAR_FILE,
};
use qwerty_core::replay::{
    render_desk_recording, run_replay, DeskRender, DeskScene, ReplayEngine, ReplayOptions,
    DEFAULT_REPLAY_BLOCK,
};
use qwerty_core::rng::Rng64;
use qwerty_core::session::EnvironmentQuality;
use qwerty_core::store::{
    config_to_json, profile_to_json, CalibrationState, Config, DeviceFingerprint, NoticeId,
    Profile, WindowPlacement, ZoneCount,
};
use qwerty_core::training::{
    train, NegativeCount, NegativeSource, TapGroup, TrainingParams, TrainingSet, Workers,
};
use qwerty_core::units::{
    CandidateId, ChannelCount, Decibels, HostTime, Milliseconds, SampleIndex, SampleRate, TapIndex,
    ZoneIndex,
};
use qwerty_core::veto::{
    InputEvent, InputKind, OutputId, SelfOutputEvent, SelfOutputKind, VetoReason,
};
use serde_json::Value;
use std::num::NonZeroU8;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use uuid::Uuid;

/// The persisted fields whose schema is an `Option`: the only places a `null` may appear. A
/// non-finite number serializes as `null` (serde_json), so a `null` anywhere else is one.
const OPTIONAL_FIELDS: [&str; 15] = [
    "active_profile_id",
    "global_hotkey",
    "window_placement",
    "imported_at",
    "layout",
    "profile_id",
    "margin",
    "latency",
    "training",
    "live_agreement",
    "onset",
    "descriptor",
    "scores",
    "v1_features",
    "last",
];

/// Panics at the first `null` that is not the value of an optional field.
fn assert_no_number_null(v: &Value, path: &str) {
    match v {
        Value::Null => panic!("{path}: null where the schema has a value"),
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                assert_no_number_null(item, &format!("{path}/{i}"));
            }
        }
        Value::Object(fields) => {
            for (key, value) in fields {
                if value.is_null() && OPTIONAL_FIELDS.contains(&key.as_str()) {
                    continue;
                }
                assert_no_number_null(value, &format!("{path}/{key}"));
            }
        }
        Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
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

/// `first` (a fixture's JSON) has no number written as null, and `second` (the same text parsed
/// and serialized again) is text-identical to it, except inside the subtrees at `trained` (JSON
/// pointers; "" is the whole document), whose numbers are compared within the reader's rounding.
// SPEC: as ST4 found, computed 17-digit f64s cannot round-trip text-identically: the workspace
// builds serde_json without `float_roundtrip`, whose default reader rounds about one in ten such
// values by an ulp, and this package may not edit a manifest. Subtrees holding them (a trained
// model; a replay report's metrics and scores) are compared within that error; the fixtures with
// chosen values (config, evaluation report, sidecar, the profile outside its model) are
// text-identical.
fn assert_round_trip(what: &str, first: &str, second: &str, trained: &[&str]) {
    let mut a: Value = serde_json::from_str(first).expect("valid JSON");
    let mut b: Value = serde_json::from_str(second).expect("valid JSON");
    assert_no_number_null(&a, what);
    if trained.is_empty() {
        assert_eq!(first, second, "{what} round-trips text-identically");
        return;
    }
    for pointer in trained {
        let take = |v: &mut Value| v.pointer_mut(pointer).map(Value::take);
        let (x, y) = (take(&mut a), take(&mut b));
        let x = x.unwrap_or_else(|| panic!("{what}: nothing at {pointer}"));
        let y = y.unwrap_or_else(|| panic!("{what}: nothing at {pointer} after the round trip"));
        assert_json_close(&x, &y, &format!("{what}{pointer}"));
    }
    let text = |v: &Value| serde_json::to_string_pretty(v).expect("serializes");
    assert_eq!(
        text(&a),
        text(&b),
        "{what} round-trips text-identically outside {trained:?}"
    );
}

fn ts(s: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(s)
        .expect("a valid RFC 3339 time")
        .with_timezone(&Utc)
}

fn file(name: &str) -> PathBuf {
    PathBuf::from(name)
}

fn rate() -> SampleRate {
    SampleRate::new(48_000).expect("48 kHz")
}

fn zone_ids() -> Vec<ZoneId> {
    (0..2u128)
        .map(|i| ZoneId::from_uuid(Uuid::from_u128(0x57_0000 + i)))
        .collect()
}

fn identity() -> DeviceIdentity {
    DeviceIdentity {
        endpoint_id: "{0.0.1.00000000}.{st7-microphone}".into(),
        name: "USB Microphone".into(),
        format: AudioFormat {
            rate: rate(),
            channels: ChannelCount::new(1).expect("mono"),
        },
        capture_mode: CaptureMode::Raw,
    }
}

/// A 2-zone Mono model trained on Gaussian descriptor clusters.
fn trained_model() -> TrainedModel {
    let layout = DescriptorLayout::Mono { a: 0 };
    let d = layout.dims();
    let mut rng = Rng64::new(0x57);
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
        rate: rate(),
        zone_ids: zone_ids(),
        seed: 0x57,
        taps,
        negatives: Vec::new(),
    };
    let params = TrainingParams {
        workers: Workers::Exactly {
            n: NonZeroU8::new(1).expect("one worker"),
        },
        ..TrainingParams::DEFAULT
    };
    train(&set, &params, &AtomicBool::new(false), &mut |_| {}).expect("the fixture trains")
}

fn current<T>(loaded: Loaded<T>) -> T {
    match loaded {
        Loaded::Current(value) => value,
        Loaded::Migrated { from_version, .. } => panic!("migrated from {from_version}"),
    }
}

fn config_fixture() -> Config {
    Config {
        active_profile_id: Some(ProfileId::from_uuid(Uuid::from_u128(0x57))),
        theme: Theme::Midnight,
        global_hotkey: Some(KeyCombo {
            modifiers: vec![Modifier::Ctrl, Modifier::Alt],
            key: "Q".into(),
        }),
        debug_capture_enabled: true,
        dismissed_notices: vec![NoticeId::FirstCloseToTray],
        window_placement: Some(WindowPlacement {
            x: -1280,
            y: 24,
            w: 1100,
            h: 760,
            maximized: false,
        }),
        text_zoom: 150,
        ..Config::initial()
    }
}

fn profile_fixture(model: TrainedModel) -> Profile {
    let mut p = Profile::new(
        "Desk".into(),
        ZoneCount::new(2).expect("2 zones"),
        ts("2026-11-01T09:00:00Z"),
    )
    .expect("a valid new profile");
    p.id = ProfileId::from_uuid(Uuid::from_u128(0x57));
    for (zone, id) in p.zones.iter_mut().zip(zone_ids()) {
        zone.id = id;
    }
    p.zones[0].actions = vec![ActionSpec::Speak {
        text: "left".into(),
    }];
    p.calibration = CalibrationState::Calibrated {
        device: DeviceFingerprint {
            identity: identity(),
            layout: ChannelLayout::all_real(ChannelCount::new(1).expect("mono")),
        },
        model,
        calibrated_at: ts("2026-11-01T09:30:00Z"),
        environment: EnvironmentQuality::Moderate,
    };
    p.imported_at = Some(ts("2026-11-02T08:00:00Z"));
    p.validate().expect("the fixture is valid");
    p
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

/// 8 prompts (3 correct, 1 wrong, 4 rejected or vetoed: dyadic rates) and a 4 s Everyday phase
/// with one false accept, every prompt reaching the model (so every margin is a number).
fn evaluation_fixture() -> EvaluationReport {
    let zones: Vec<ZoneRef> = zone_ids()
        .into_iter()
        .zip(["Left", "Right"])
        .map(|(id, label)| ZoneRef {
            id,
            label: label.into(),
        })
        .collect();
    let prompts = (0..8)
        .map(|i| ScriptedPrompt {
            zone: ZoneIndex(i % 2),
            opened_at: at_ms(2000 * i64::from(i)),
            deadline: at_ms(2000 * i64::from(i) + 1500),
            reopened: false,
        })
        .collect();
    let negatives = vec![ScriptedNegative {
        class: NegativeClass::Everyday,
        start: at_ms(20_000),
        end: at_ms(24_000),
    }];
    let mut session = EvaluationSession::new(
        zones,
        PromptSchedule::Scripted { prompts, negatives },
        at_ms(0),
    )
    .expect("a valid script");
    let accepted = |zone: u16, best: u16| DecisionOutcome::Accepted {
        zone: ZoneIndex(zone),
        descriptor: descriptor(),
        scores: scores(best),
    };
    let rejected = |reason| DecisionOutcome::Rejected {
        reason,
        descriptor: descriptor(),
        scores: scores(0),
    };
    let outcomes = [
        accepted(0, 0),
        accepted(1, 1),
        accepted(0, 0),
        accepted(0, 0),
        rejected(ModelRejectReason::OutOfDistribution),
        rejected(ModelRejectReason::Background),
        DecisionOutcome::Vetoed {
            reason: VetoReason::Input {
                kind: InputKind::KeyDown,
                offset: Milliseconds::new(12.5),
            },
            descriptor: descriptor(),
            scores: scores(0),
        },
        rejected(ModelRejectReason::Ambiguous),
    ];
    let mut events = Vec::new();
    for (i, outcome) in outcomes.into_iter().enumerate() {
        let opened = 2000 * i as i64;
        session.tick(at_ms(opened), &mut events);
        session.on_decision(&decision(opened + 400, outcome), &mut events);
    }
    session.tick(at_ms(20_000), &mut events);
    session.on_decision(&decision(21_000, accepted(1, 1)), &mut events);
    session.tick(at_ms(24_000), &mut events);
    let meta = ReportMeta {
        profile_id: ProfileId::from_uuid(Uuid::from_u128(0x57)),
        device: identity(),
        front_end_version: 1,
        model_trained_at: ts("2026-11-01T09:30:00Z"),
        zone_taps: vec![10, 10],
        negatives: vec![NegativeCount {
            source: NegativeSource::IgnoreStep,
            clips: 12,
        }],
        background_regime: qwerty_core::model::BackgroundRegime::PooledAll {
            negative_clips: 12,
            needed: 24,
        },
        input_veto_enabled: true,
    };
    let report = session
        .finish(meta, ts("2026-11-03T10:00:00Z"))
        .expect("the run finished");
    assert_eq!(report.overall.correct, 3);
    assert_eq!(report.overall.wrong, 1);
    assert!(report.prompts.iter().all(|p| p.margin.is_some()));
    report
}

/// A fresh directory under the temp dir, removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> TempDir {
        let dir = std::env::temp_dir().join(format!("qwerty-st7-{name}-{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).expect("clear the old test directory");
        }
        std::fs::create_dir_all(&dir).expect("make the test directory");
        TempDir(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_dir_all(&self.0) {
            eprintln!("could not remove {}: {e}", self.0.display());
        }
    }
}

/// A finished recording whose sidecar holds every field and every event kind; returns its dir.
fn sidecar_fixture(root: &Path) -> PathBuf {
    let format = identity().format;
    let header = SidecarHeader {
        app_version: "0.1.0".into(),
        device: identity(),
        layout: Some(ChannelLayout::all_real(format.channels)),
        profile_id: Some(ProfileId::from_uuid(Uuid::from_u128(0x57))),
        created_at: ts("2026-11-03T10:00:00Z"),
    };
    let mut writer = RecordingWriter::create(root, header).expect("a new recording");
    let samples = vec![0.125f32; 480];
    for (start, ms) in [(0u64, 0i64), (480, 10), (1_440, 30)] {
        if start == 1_440 {
            writer.mark_gap(SampleIndex(start), 480);
        }
        let block = AudioBlock::new(
            format,
            SampleIndex(start),
            &samples,
            BlockTime::Timestamped {
                host_time: at_ms(ms),
            },
            false,
        )
        .expect("whole frames");
        writer.write_block(&block).expect("written");
    }
    let zone = zone_ids()[0];
    for event in [
        SidecarEvent::CalibrationStarted {
            start: SampleIndex(0),
            seed: 0x57,
            taps_per_zone: 10,
        },
        SidecarEvent::Segment {
            start: SampleIndex(0),
            kind: SegmentKind::CalibrationZone {
                zone,
                label: "Left".into(),
            },
            reconfigured: true,
        },
        SidecarEvent::Segment {
            start: SampleIndex(960),
            kind: SegmentKind::EvaluationNegative {
                class: NegativeClass::Other {
                    label: "Fan".into(),
                },
            },
            reconfigured: true,
        },
        SidecarEvent::Prompt {
            zone,
            opened_at: at_ms(5),
            deadline: at_ms(3005),
            reopened: true,
        },
        SidecarEvent::Input {
            event: InputEvent {
                time: at_ms(7),
                kind: InputKind::MouseButtonUp,
            },
        },
        SidecarEvent::SelfOutput {
            event: SelfOutputEvent::Began {
                id: OutputId(3),
                kind: SelfOutputKind::Toast,
                at: at_ms(8),
            },
        },
        SidecarEvent::SelfOutput {
            event: SelfOutputEvent::Ended {
                id: OutputId(3),
                at: at_ms(9),
            },
        },
        SidecarEvent::Decision {
            onset: SampleIndex(100),
            accepted_zone: Some(zone),
        },
    ] {
        writer.push_event(event);
    }
    let summary = writer.finish().expect("finished");
    PathBuf::from(summary.dir)
}

#[test]
fn st7_persisted_types_write_no_non_finite_value() {
    // Config.
    for config in [Config::initial(), config_fixture()] {
        let first = config_to_json(&config, &file("config.json")).expect("serializes");
        let parsed = current(load_config(&first, &file("config.json")).expect("loads"));
        let second = config_to_json(&parsed, &file("config.json")).expect("serializes again");
        assert_round_trip("config", &first, &second, &[]);
    }

    // Profile with a TrainedModel; a non-finite model value is refused, never written.
    let model = trained_model();
    let profile = profile_fixture(model.clone());
    let first = profile_to_json(&profile, &file("profile.json")).expect("serializes");
    let parsed = current(load_profile(&first, &file("profile.json")).expect("loads"));
    let second = profile_to_json(&parsed, &file("profile.json")).expect("serializes again");
    assert_round_trip("profile", &first, &second, &["/calibration/model"]);
    let mut broken = profile.clone();
    if let CalibrationState::Calibrated { model, .. } = &mut broken.calibration {
        model.thresholds.tau_in = f64::NAN;
    }
    assert!(profile_to_json(&broken, &file("profile.json")).is_err());

    // EvaluationReport; a non-finite metric is refused, never written.
    let report = evaluation_fixture();
    let first = report.to_json().expect("serializes");
    let parsed = EvaluationReport::from_json(&first, &file("evaluation.json")).expect("loads");
    let second = parsed.to_json().expect("serializes again");
    assert_round_trip("evaluation report", &first, &second, &[]);
    let mut broken = report.clone();
    broken.overall.accuracy = f64::INFINITY;
    assert!(broken.to_json().is_err());

    // Sidecar: as the recorder wrote it, and as the reader reads it back.
    let tmp = TempDir::new("sidecar");
    let dir = sidecar_fixture(&tmp.0);
    let first = std::fs::read_to_string(dir.join(SIDECAR_FILE)).expect("the sidecar");
    let sidecar = RecordingReader::open(&dir).expect("readable").sidecar;
    assert!(sidecar.layout.is_some() && sidecar.profile_id.is_some());
    assert_eq!(sidecar.events.len(), 8);
    assert_eq!(sidecar.gaps.len(), 1);
    let second = serde_json::to_string_pretty(&sidecar).expect("serializes again");
    assert_round_trip("sidecar", &first, &second, &[]);

    // ReplayReport, from a real replay of a synthetic recording through both engines (the V2
    // result also has a training summary, live agreement and error dumps).
    let render = TempDir::new("replay");
    let spec = DeskRender {
        scene: DeskScene::CalibrateEvaluate,
        format: AudioFormat {
            rate: rate(),
            channels: ChannelCount::new(2).expect("stereo"),
        },
        zones: 2,
        seed: 0x57,
    };
    let recording = render_desk_recording(&render.0, &spec).expect("renders");
    let options = ReplayOptions {
        engines: vec![ReplayEngine::V1, ReplayEngine::V2],
        detector: qwerty_core::detector::DetectorParams::DEFAULT,
        training: TrainingParams::DEFAULT,
        block: DEFAULT_REPLAY_BLOCK,
        // A profile that still needs calibration: V2 calibrates from the recorded run.
        profile: Some(
            Profile::new(
                "Uncalibrated".into(),
                ZoneCount::new(2).expect("2 zones"),
                ts("2026-11-03T10:00:00Z"),
            )
            .expect("a valid new profile"),
        ),
    };
    let replay = run_replay(Path::new(&recording.dir), &options).expect("replays");
    let first = serde_json::to_string_pretty(&replay).expect("serializes");
    let parsed: qwerty_core::replay::ReplayReport = serde_json::from_str(&first).expect("parses");
    let second = serde_json::to_string_pretty(&parsed).expect("serializes again");
    let v2 = replay
        .results
        .iter()
        .position(|r| r.engine == ReplayEngine::V2)
        .expect("a V2 result");
    assert!(replay.results[v2].training.is_some());
    assert!(replay.results[v2].live_agreement.is_some());
    // Every float of a replay report is computed (metrics like 11/12, model scores), so the whole
    // report is compared within the reader's rounding.
    assert_round_trip("replay report", &first, &second, &[""]);
}
