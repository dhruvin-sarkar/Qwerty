//! Synthetic DSP soak test — the Phase 1 stress binary (`ROADMAP.md`).
//!
//! Runs the full onset → features → classifier pipeline against synthetic taps
//! and sustained sounds for a wall-clock duration, under a live-heap-counting
//! allocator, and reports the evidence for the "DSP core (synthetic)" rows of
//! `ACCEPTANCE.md`:
//!   * clean synthetic zone accuracy (sanity floor ≥ 99%),
//!   * zero false accepts on sustained sound,
//!   * no unbounded live-heap growth (< 5%),
//!   * no panics / clean exit.
//!
//! Usage:
//!   cargo run -p qwerty-core --example soak -- --duration 1800
//!
//! Exits non-zero if any synthetic criterion fails, so it doubles as a gate.

use std::time::{Duration, Instant};

use qwerty_core::soak::{allocated_bytes, BatchTally, CountingAllocator, Soak, SoakConfig};

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn main() {
    let duration = Duration::from_secs(parse_duration_secs());

    let config = SoakConfig {
        zone_count: parse_zone_count(),
        ..SoakConfig::default()
    };
    println!(
        "qwerty-core soak: {} zones, duration {}s",
        config.zone_count,
        duration.as_secs()
    );

    let mut soak = Soak::new(config);

    // Warm up so the heap reaches steady state before the baseline reading.
    for _ in 0..50 {
        let _ = soak.run_batch();
    }
    let heap_start = allocated_bytes();

    let mut tally = BatchTally::default();
    let start = Instant::now();
    let mut last_beat = start;
    let mut heap_peak = heap_start;

    while start.elapsed() < duration {
        tally.add(&soak.run_batch());
        let now = Instant::now();
        heap_peak = heap_peak.max(allocated_bytes());
        if now.duration_since(last_beat) >= Duration::from_secs(10) {
            last_beat = now;
            println!(
                "  t={:>5}s  batches={:>8}  acc={:.4}  false_accepts={}  live_heap={} KB",
                start.elapsed().as_secs(),
                tally.taps / soak.zone_count().max(1) as u32,
                tally.accuracy(),
                tally.sustained_false_accepts,
                allocated_bytes() / 1024,
            );
        }
    }

    let heap_end = allocated_bytes();
    let elapsed = start.elapsed();

    let growth_pct = if heap_start > 0 {
        (heap_end as f64 - heap_start as f64) / heap_start as f64 * 100.0
    } else {
        0.0
    };

    println!("\n=== SOAK REPORT ===");
    println!("elapsed:               {:.1}s", elapsed.as_secs_f64());
    println!("taps presented:        {}", tally.taps);
    println!("correct zone:          {}", tally.correct);
    println!("rejected taps (miss):  {}", tally.rejected_taps);
    println!("misclassified (wrong): {}", tally.misclassified);
    println!("clean accuracy:        {:.4}", tally.accuracy());
    println!("sustained presented:   {}", tally.sustained);
    println!("sustained FALSE accept:{}", tally.sustained_false_accepts);
    println!("foreign transients:    {}", tally.foreign_transients);
    println!("foreign FALSE accept:  {}", tally.foreign_false_accepts);
    println!(
        "live heap start/end:   {} / {} bytes  (peak {} bytes)",
        heap_start, heap_end, heap_peak
    );
    println!("live heap growth:      {growth_pct:+.3}%");

    // Gate the synthetic acceptance criteria.
    let mut failures = Vec::new();
    if tally.accuracy() < 0.99 {
        failures.push(format!("accuracy {:.4} < 0.99", tally.accuracy()));
    }
    if tally.sustained_false_accepts != 0 {
        failures.push(format!(
            "{} false accepts on sustained sound (must be 0)",
            tally.sustained_false_accepts
        ));
    }
    if growth_pct.abs() >= 5.0 {
        failures.push(format!("live heap growth {growth_pct:+.3}% exceeds ±5%"));
    }
    if tally.foreign_false_accepts != 0 {
        failures.push(format!(
            "{} foreign transients accepted by the novelty gate (must be 0)",
            tally.foreign_false_accepts
        ));
    }
    // Guard against a vacuous novelty check: if no foreign transient ever
    // reached the classifier, the false-accept count above proves nothing.
    if tally.foreign_transients == 0 {
        failures.push(
            "no foreign transients reached the novelty gate — the gate was not exercised".into(),
        );
    }

    if failures.is_empty() {
        println!("\nRESULT: PASS — all synthetic acceptance criteria met.");
    } else {
        println!("\nRESULT: FAIL");
        for f in &failures {
            println!("  - {f}");
        }
        std::process::exit(1);
    }
}

/// Parse `--duration <seconds>`; default 30s for a quick local run.
fn parse_duration_secs() -> u64 {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--duration" {
            if let Some(v) = args.next() {
                if let Ok(secs) = v.parse::<u64>() {
                    return secs;
                }
            }
            eprintln!("--duration requires a number of seconds");
            std::process::exit(2);
        }
    }
    30
}

/// Parse `--zones <n>`; default 4. Restricted to the product's supported zone
/// counts (2/4/6/8, per `DESIGN.md`) — this both matches what the app offers
/// and keeps `zone_count` ≥ 2, so `Soak::new` always has zones to calibrate.
fn parse_zone_count() -> usize {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--zones" {
            match args.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(n @ (2 | 4 | 6 | 8)) => return n,
                _ => {
                    eprintln!("--zones requires one of: 2, 4, 6, 8");
                    std::process::exit(2);
                }
            }
        }
    }
    4
}
