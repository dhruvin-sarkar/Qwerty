//! RI2 (p2 §2.7), hardware evidence: it injects input, so it is `#[ignore]` and never part of a
//! default run. Run it at the desk, workstation unlocked, a non-elevated window in the
//! foreground, hands off the keyboard and mouse:
//! `cargo test -p qwerty-app --test rawinput_ri2 -- --ignored --nocapture`.
//! Raw Input registration is process-global, so this is the only `#[test]` in its binary.

use std::mem::size_of;
use std::time::{Duration, Instant};

use crossbeam_channel::bounded;
use qwerty_app::engine::clock;
use qwerty_app::engine::messages::{
    InputRecord, RawInputMsg, RawInputRequest, RawInputStatus, WorkerId,
};
use qwerty_app::engine::rawinput::{self, INPUT_RING_RECORDS};
use qwerty_core::units::HostTime;
use qwerty_core::veto::InputKind;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYBD_EVENT_FLAGS, KEYEVENTF_KEYUP,
    VK_LSHIFT,
};

const PAIRS: usize = 200;
/// p99 bound on (record stamp − send stamp): 5 ms in 100-ns ticks.
const P99_LIMIT_TICKS: i64 = 50_000;

/// One VK_LSHIFT transition: no character and no focus change.
fn inject(kind: InputKind) {
    let flags = match kind {
        InputKind::KeyDown => KEYBD_EVENT_FLAGS(0),
        InputKind::KeyUp => KEYEVENTF_KEYUP,
        InputKind::MouseButtonDown | InputKind::MouseButtonUp => {
            unreachable!("RI2 injects keys only")
        }
    };
    let input = INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VK_LSHIFT,
                wScan: 0,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    };
    // SAFETY: one fully initialised INPUT with the size the API expects.
    let sent = unsafe { SendInput(&[input], size_of::<INPUT>() as i32) };
    assert_eq!(sent, 1, "SendInput was blocked");
}

fn ms(ticks: i64) -> f64 {
    ticks as f64 / 10_000.0
}

#[test]
#[ignore = "hardware evidence: injects VK_LSHIFT with SendInput; run at the desk with --ignored"]
fn ri2_injected_input_is_stamped_within_5_ms_p99() {
    let id = WorkerId(1);
    let (ring, mut inputs) = rtrb::RingBuffer::new(INPUT_RING_RECORDS);
    let (tx, status) = bounded(16);
    let (exits_tx, _exits) = bounded(16);
    let worker = rawinput::spawn(RawInputRequest {
        worker: id,
        ring,
        tx,
        exits: exits_tx,
    })
    .expect("spawn");
    assert_eq!(
        status
            .recv_timeout(Duration::from_secs(5))
            .expect("a status"),
        RawInputMsg {
            worker: id,
            status: RawInputStatus::Active,
            skipped_records: 0,
        }
    );

    let mut sends: Vec<(HostTime, InputKind)> = Vec::with_capacity(2 * PAIRS);
    for _ in 0..PAIRS {
        for kind in [InputKind::KeyDown, InputKind::KeyUp] {
            let sent_at = clock::now();
            inject(kind);
            sends.push((sent_at, kind));
        }
        std::thread::sleep(Duration::from_millis(2));
    }

    let mut records = Vec::with_capacity(2 * PAIRS);
    let deadline = Instant::now() + Duration::from_secs(2);
    while records.len() < sends.len() && Instant::now() < deadline {
        while let Ok(record) = inputs.pop() {
            records.push(record);
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    worker.stop(Duration::from_millis(500)).expect("joined");

    assert_eq!(
        records.len(),
        sends.len(),
        "records lost, or other input arrived during the test"
    );
    let mut latencies: Vec<i64> = sends
        .iter()
        .zip(&records)
        .map(|(&(sent_at, kind), record)| match *record {
            InputRecord::Event(event) => {
                assert_eq!(
                    event.kind, kind,
                    "the records are out of step with the sends"
                );
                assert!(event.time >= sent_at, "stamped before its send");
                event.time.0 - sent_at.0
            }
            InputRecord::Gap { from, to } => panic!("the ring overflowed: {from:?}..{to:?}"),
        })
        .collect();
    latencies.sort_unstable();
    let pick = |q: f64| latencies[((q * latencies.len() as f64).ceil() as usize).max(1) - 1];
    let (p50, p99, max) = (pick(0.50), pick(0.99), latencies[latencies.len() - 1]);
    println!(
        "RI2: {} records, stamp - send: p50 {:.3} ms, p99 {:.3} ms, max {:.3} ms",
        latencies.len(),
        ms(p50),
        ms(p99),
        ms(max)
    );
    assert!(p99 <= P99_LIMIT_TICKS, "p99 {:.3} ms > 5 ms", ms(p99));
}
