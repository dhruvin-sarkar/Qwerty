//! qwerty-app — the Qwerty desktop binary.
//!
//! Launched with no arguments it opens the GUI (Phase 3 onward). The Phase 2
//! CLI validation harnesses remain available as explicit subcommands — they
//! exercise the real audio path (device enumeration, live onset monitor, a
//! terminal calibrate-then-classify preview) so the DSP core can still be
//! validated on real hardware without the GUI. Tray, hotkeys, and platform
//! action dispatch arrive in later phases.

mod app_state;
mod audio_thread;
mod hotkeys;
mod tray;
mod ui;

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
            let device = args.get(3).map(String::as_str);
            if let Err(e) = run_monitor(seconds, device) {
                fail(&e);
            }
        }
        Some("--live") => {
            let zones = arg_or(&args, 2, 4usize, "zones");
            let taps = arg_or(&args, 3, 10usize, "taps");
            let device = args.get(4).map(String::as_str);
            if let Err(e) = run_live(zones, taps, device) {
                fail(&e);
            }
        }
        Some("--themes") => ui::theme::print_report(),
        None => {
            if let Err(e) = ui::shell::run() {
                eprintln!("error: could not start the GUI: {e}");
                std::process::exit(1);
            }
        }
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
    println!("Run with no arguments to open the GUI. Debug subcommands:");
    println!();
    println!("Usage:");
    println!("  qwerty-app                                open the GUI (default)");
    println!("  qwerty-app --list-devices                 list audio input devices");
    println!("  qwerty-app --monitor [secs] [device]      live onset monitor + level meter (default 20s, default mic)");
    println!("  qwerty-app --live [zones] [taps] [device] calibrate then classify taps live (default 4 zones, 10 taps)");
    println!("  qwerty-app --themes                       show the 5 theme palettes + WCAG contrast check");
    println!();
    println!("  [device] = a list index (see --list-devices) or a name substring, e.g. \"Realtek\".");
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
