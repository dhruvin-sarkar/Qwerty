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
            let seconds = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
            if let Err(e) = run_monitor(seconds) {
                fail(&e);
            }
        }
        Some("--live") => {
            let zones = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
            let taps = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
            if let Err(e) = run_live(zones, taps) {
                fail(&e);
            }
        }
        _ => print_usage(),
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
