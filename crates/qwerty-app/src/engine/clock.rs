//! The engine's host clock: QueryPerformanceCounter scaled to 100-ns `HostTime` ticks, the domain
//! WASAPI stamps capture packets in (p2 §2.2, §2.12).

use std::sync::OnceLock;

use qwerty_core::units::HostTime;
use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};

/// QueryPerformanceFrequency is fixed at boot, so it is read once per process.
static QPC_FREQUENCY: OnceLock<i64> = OnceLock::new();

fn frequency() -> i64 {
    *QPC_FREQUENCY.get_or_init(|| {
        let mut freq = 0i64;
        // SAFETY: `freq` is a valid, writable i64 for the duration of the call.
        let read = unsafe { QueryPerformanceFrequency(&mut freq) };
        // INVARIANT: QueryPerformanceFrequency always succeeds on Windows XP and later
        // (documented), and never reports 0.
        read.expect("QueryPerformanceFrequency cannot fail on Windows XP and later");
        freq
    })
}

/// The raw counter.
fn counter() -> i64 {
    let mut qpc = 0i64;
    // SAFETY: `qpc` is a valid, writable i64 for the duration of the call.
    let read = unsafe { QueryPerformanceCounter(&mut qpc) };
    // INVARIANT: QueryPerformanceCounter always succeeds on Windows XP and later (documented).
    read.expect("QueryPerformanceCounter cannot fail on Windows XP and later");
    qpc
}

/// Now, on the QPC clock in 100-ns ticks. Non-decreasing (QPC is monotonic). Lock-free after the
/// first call, so the raw-input thread may stamp input with it.
pub fn now() -> HostTime {
    let qpc = counter();
    // INVARIANT: qpc·10^7/freq overflows i64 only after ~29 000 years of uptime.
    HostTime::from_qpc(qpc, frequency()).expect("QPC uptime fits the i64 tick range")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Raw QPC read through the core conversion, independent of `now()`'s code path.
    fn reference() -> HostTime {
        let mut qpc = 0i64;
        let mut freq = 0i64;
        // SAFETY: both are valid, writable i64s for the duration of the calls.
        unsafe {
            QueryPerformanceCounter(&mut qpc).expect("QPC");
            QueryPerformanceFrequency(&mut freq).expect("QPF");
        }
        HostTime::from_qpc(qpc, freq).expect("in range")
    }

    #[test]
    fn ck1_now_is_monotonic_and_matches_qpc() {
        let mut last = now();
        for _ in 0..100_000 {
            let t = now();
            assert!(t >= last, "now() went backwards: {last:?} -> {t:?}");
            last = t;
        }

        // Agreement within 1 ms: bracket now() between two reference reads. A bracket wider than
        // 1 ms means the thread was preempted between the reads, so it is retried.
        const ONE_MS_TICKS: i64 = 10_000;
        let mut agreed = false;
        for _ in 0..1000 {
            let before = reference();
            let t = now();
            let after = reference();
            if after.0 - before.0 > ONE_MS_TICKS {
                continue;
            }
            assert!(
                before <= t && t <= after,
                "{before:?} <= {t:?} <= {after:?}"
            );
            agreed = true;
            break;
        }
        assert!(agreed, "no bracket narrower than 1 ms in 1000 attempts");
    }
}
