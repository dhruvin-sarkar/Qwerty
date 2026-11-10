//! Writes a synthetic recording from the modal desk, so replay runs without hardware (§1.18.3).
//!
//! Usage:
//!   cargo run -p qwerty-core --release --example desk_render -- --out <dir>
//!       --scene calibrate-evaluate|negatives --rate <hz> --channels 1|2 --zones 2|4|6|8
//!       --seed <u64>
//!
//! The recording is a new `<YYYYMMDDTHHMMSSZ>` directory under `--out`; its summary (with the
//! directory) is printed to stdout as JSON. Exit codes: 0 on success; 1 on a render or write
//! error; 2 on bad arguments.

use qwerty_core::audio::AudioFormat;
use qwerty_core::replay::{render_desk_recording, DeskRender, DeskScene};
use qwerty_core::units::{ChannelCount, SampleRate};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage: desk_render --out <dir> --scene calibrate-evaluate|negatives \
                     --rate <hz> --channels 1|2 --zones 2|4|6|8 --seed <u64>";

fn main() -> ExitCode {
    let (out, spec) = match parse(std::env::args().skip(1).collect()) {
        Ok(parsed) => parsed,
        Err(message) => {
            eprintln!("desk_render: {message}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match render_desk_recording(&out, &spec) {
        Ok(summary) => match serde_json::to_string_pretty(&summary) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("desk_render: {e}");
                ExitCode::from(1)
            }
        },
        Err(e) => {
            eprintln!("desk_render: {e:?}");
            ExitCode::from(1)
        }
    }
}

fn parse(argv: Vec<String>) -> Result<(PathBuf, DeskRender), String> {
    let (mut out, mut scene, mut rate, mut channels, mut zones, mut seed) =
        (None, None, None, None, None, None);
    let mut it = argv.into_iter();
    while let Some(flag) = it.next() {
        let Some(value) = it.next() else {
            return Err(format!("{flag} needs a value"));
        };
        match flag.as_str() {
            "--out" => out = Some(PathBuf::from(value)),
            "--scene" => {
                scene = Some(match value.as_str() {
                    "calibrate-evaluate" => DeskScene::CalibrateEvaluate,
                    "negatives" => DeskScene::Negatives,
                    other => return Err(format!("unknown scene {other:?}")),
                })
            }
            "--rate" => {
                let hz = number::<u32>("--rate", &value)?;
                rate = Some(SampleRate::new(hz).map_err(|e| format!("--rate {hz}: {e:?}"))?);
            }
            "--channels" => {
                let n = number::<u16>("--channels", &value)?;
                channels =
                    Some(ChannelCount::new(n).map_err(|e| format!("--channels {n}: {e:?}"))?);
            }
            "--zones" => zones = Some(number::<u8>("--zones", &value)?),
            "--seed" => seed = Some(number::<u64>("--seed", &value)?),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    let missing = |flag: &str| format!("{flag} is required");
    let format = AudioFormat {
        rate: rate.ok_or_else(|| missing("--rate"))?,
        channels: channels.ok_or_else(|| missing("--channels"))?,
    };
    let spec = DeskRender {
        scene: scene.ok_or_else(|| missing("--scene"))?,
        format,
        zones: zones.ok_or_else(|| missing("--zones"))?,
        seed: seed.ok_or_else(|| missing("--seed"))?,
    };
    Ok((out.ok_or_else(|| missing("--out"))?, spec))
}

fn number<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} {value:?} is not a valid number"))
}
