//! Synthetic soak v2 (§1.20): the frozen v1 fixture calibrated through `CalibrationSession` and
//! streamed through `Pipeline` for a wall-clock duration, under a live-heap-counting allocator.
//!
//! Gates (all must pass):
//!   1. no false accept on sustained sound;
//!   2. no false accept on foreign transients, with enough of them reaching the model;
//!   3. live-heap growth from the end of warm-up to the end < 5 %;
//!   4. no panic;
//!   5. clean accuracy ≥ 0.99, at `--zones 4` only (other zone counts print it ungated).
//!
//! Usage:
//!   cargo run -p qwerty-core --release --example soak -- --duration 1800 [--zones 2|4|6|8]
//!
//! Defaults: `--duration 30`, `--zones 4`. Exit codes: 0 pass, 1 gate failure, 2 bad arguments.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use qwerty_core::soak::{
    allocated_bytes, gate, heap_growth_percent, BatchTally, CountingAllocator, Soak, SoakConfig,
    WARMUP_BATCHES,
};
use qwerty_core::units::Seconds;

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Seconds between progress lines.
const BEAT: Duration = Duration::from_secs(10);

struct Args {
    duration: u64,
    zones: usize,
}

fn main() -> ExitCode {
    let args = match parse_args(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            eprintln!("usage: soak [--duration <seconds>] [--zones 2|4|6|8]");
            return ExitCode::from(2);
        }
    };
    let config = SoakConfig {
        zone_count: args.zones,
        ..SoakConfig::default()
    };
    println!(
        "qwerty-core soak v2 (core {}): {} zones, duration {}s, seed {:#x}",
        qwerty_core::CORE_VERSION,
        config.zone_count,
        args.duration,
        config.seed
    );
    match run(config, Duration::from_secs(args.duration)) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            println!("\nRESULT: FAIL\n  - the run stopped: {error}");
            ExitCode::from(1)
        }
    }
}

/// Runs the soak and prints its report. Ok(true) iff every gate passed.
fn run(config: SoakConfig, duration: Duration) -> Result<bool, qwerty_core::soak::SoakError> {
    let built = Instant::now();
    let mut soak = Soak::new(config)?;
    println!(
        "calibrated in {:.1}s; {WARMUP_BATCHES} warm-up batches",
        built.elapsed().as_secs_f64()
    );
    for _ in 0..WARMUP_BATCHES {
        soak.run_batch()?;
    }
    let heap_start = allocated_bytes();

    let mut tally = BatchTally::default();
    let mut batches = 0u64;
    let start = Instant::now();
    let mut last_beat = start;
    let mut heap_peak = heap_start;
    while start.elapsed() < duration {
        tally.add(&soak.run_batch()?);
        batches += 1;
        heap_peak = heap_peak.max(allocated_bytes());
        let now = Instant::now();
        if now.duration_since(last_beat) >= BEAT {
            last_beat = now;
            println!(
                "  t={:>5}s  batches={:>8}  acc={}  sustained_fa={}  foreign_fa={}  foreign_at_model={}  live_heap={} KB",
                start.elapsed().as_secs(),
                batches,
                accuracy_text(&tally),
                tally.sustained_false_accepts,
                tally.foreign_false_accepts,
                tally.foreign_reaching_model,
                allocated_bytes() / 1024,
            );
        }
    }
    let heap_end = allocated_bytes();
    let elapsed = start.elapsed();
    let growth = heap_growth_percent(heap_start, heap_end);

    println!("\n=== SOAK REPORT ===");
    println!("elapsed:                 {:.1}s", elapsed.as_secs_f64());
    println!("batches:                 {batches}");
    println!("taps presented:          {}", tally.taps);
    println!("correct zone:            {}", tally.correct);
    println!("rejected taps (miss):    {}", tally.rejected_taps);
    println!("misclassified (wrong):   {}", tally.misclassified);
    println!("clean accuracy:          {}", accuracy_text(&tally));
    println!("sustained presented:     {}", tally.sustained);
    println!("sustained FALSE accept:  {}", tally.sustained_false_accepts);
    println!("foreign presented:       {}", tally.foreign_transients);
    println!("foreign reaching model:  {}", tally.foreign_reaching_model);
    println!("foreign FALSE accept:    {}", tally.foreign_false_accepts);
    println!("live heap start/end:     {heap_start} / {heap_end} bytes  (peak {heap_peak} bytes)");
    println!("live heap growth:        {growth:+.3}%");

    let failures = gate(
        &tally,
        config.zone_count,
        Seconds::new(duration.as_secs_f64()),
        growth,
    );
    if failures.is_empty() {
        println!("\nRESULT: PASS: every soak gate met.");
        Ok(true)
    } else {
        println!("\nRESULT: FAIL");
        for f in &failures {
            println!("  - {f}");
        }
        Ok(false)
    }
}

fn accuracy_text(tally: &BatchTally) -> String {
    match tally.accuracy() {
        Some(a) => format!("{a:.4}"),
        None => "n/a".to_string(),
    }
}

/// `--duration <seconds>` (default 30) and `--zones 2|4|6|8` (default 4); anything else is an error.
fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut parsed = Args {
        duration: 30,
        zones: 4,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--duration" => {
                parsed.duration = args
                    .next()
                    .and_then(|v| v.parse::<u64>().ok())
                    .ok_or("--duration requires a whole number of seconds")?;
            }
            "--zones" => {
                parsed.zones = match args.next().and_then(|v| v.parse::<usize>().ok()) {
                    Some(n @ (2 | 4 | 6 | 8)) => n,
                    Some(_) | None => return Err("--zones requires one of: 2, 4, 6, 8".into()),
                };
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(parsed)
}
