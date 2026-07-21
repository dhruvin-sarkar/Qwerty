//! qwerty-app — the Qwerty desktop binary.
//!
//! Phase 2: a CLI validation harness (no GUI yet). It exercises the real audio
//! path — device enumeration and a live onset monitor against the mic — so the
//! DSP core proven synthetically in Phase 1 can be validated on real hardware
//! before any GUI is built (`ROADMAP.md`). The egui shell, tray, hotkeys, and
//! platform action dispatch arrive in later phases.

mod audio_thread;

use audio_thread::{list_input_devices, run_live, run_monitor};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("--list-devices") => match list_input_devices() {
            Ok(names) => {
                println!("Audio input devices ({}):", names.len());
                for n in &names {
                    println!("  - {n}");
                }
            }
            Err(e) => fail(&e),
        },
        Some("--monitor") => {
            let seconds = arg_or(&args, 2, 20u64, "seconds");
            if let Err(e) = run_monitor(seconds) {
                fail(&e);
            }
        }
        Some("--live") => {
            let zones = arg_or(&args, 2, 4usize, "zones");
            let taps = arg_or(&args, 3, 10usize, "taps");
            if let Err(e) = run_live(zones, taps) {
                fail(&e);
            }
        }
        None => print_usage(),
        Some(other) => {
            eprintln!("error: unknown command: {other}");
            print_usage();
            std::process::exit(2);
        }
    }
}

/// Parse a positional CLI arg, defaulting only when it is ABSENT — a present
/// but malformed value is a loud error, never silently defaulted (mirrors the
/// soak example's arg handling; `CLAUDE.md`: fail fast).
fn arg_or<T: std::str::FromStr>(args: &[String], idx: usize, default: T, name: &str) -> T {
    match args.get(idx) {
        None => default,
        Some(s) => s.parse().unwrap_or_else(|_| {
            eprintln!("error: invalid {name}: {s:?}");
            std::process::exit(2);
        }),
    }
}

fn print_usage() {
    println!(
        "Qwerty {} (qwerty-core {})",
        env!("CARGO_PKG_VERSION"),
        qwerty_core::CORE_VERSION,
    );
    println!("Phase 2 CLI validation harness (no GUI yet).");
    println!();
    println!("Usage:");
    println!("  qwerty-app --list-devices          list audio input devices");
    println!("  qwerty-app --monitor [seconds]     live onset monitor (default 20s) — tap the desk");
    println!("  qwerty-app --live [zones] [taps]   calibrate then classify taps live (default 4 zones, 10 taps)");
}

fn fail(e: &dyn std::error::Error) -> ! {
    eprintln!("error: {e}");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    #[test]
    fn links_against_core() {
        // Smoke test: the binary crate compiles and can reach into qwerty-core
        // across the workspace boundary.
        assert!(!qwerty_core::CORE_VERSION.is_empty());
    }
}
