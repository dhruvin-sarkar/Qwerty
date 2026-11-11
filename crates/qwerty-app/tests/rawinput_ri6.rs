//! RI6 (p2 §2.7), hardware evidence: it injects input, so it is `#[ignore]` and never part of a
//! default run. Run it at the desk, workstation unlocked, a non-elevated window in the
//! foreground, hands off the keyboard and mouse:
//! `cargo test -p qwerty-app --test rawinput_ri6 -- --ignored --nocapture`.
//! Raw Input registration is process-global, so this is the only `#[test]` in its binary.

use std::mem::size_of;
use std::time::{Duration, Instant};

use crossbeam_channel::bounded;
use qwerty_app::engine::clock;
use qwerty_app::engine::messages::{
    InputRecord, RawInputMsg, RawInputRequest, RawInputStatus, WorkerId,
};
use qwerty_app::engine::rawinput;
use qwerty_core::units::HostTime;
use qwerty_core::veto::{InputEvent, InputKind};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYBD_EVENT_FLAGS, KEYEVENTF_KEYUP,
    VK_LSHIFT,
};

const RING: usize = 4;

/// One VK_LSHIFT transition: no character and no focus change.
fn inject(flags: KEYBD_EVENT_FLAGS) {
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

fn filler(n: i64) -> InputRecord {
    InputRecord::Event(InputEvent {
        time: HostTime(n),
        kind: InputKind::KeyUp,
    })
}

#[test]
#[ignore = "hardware evidence: injects VK_LSHIFT with SendInput; run at the desk with --ignored"]
fn ri6_a_gap_follows_a_freed_slot_without_further_input() {
    let (mut ring, mut inputs) = rtrb::RingBuffer::new(RING);
    for n in 0..RING as i64 {
        ring.push(filler(n)).expect("room while pre-filling");
    }
    let id = WorkerId(1);
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

    let injected_from = clock::now();
    for _ in 0..3 {
        inject(KEYBD_EVENT_FLAGS(0));
        inject(KEYEVENTF_KEYUP);
    }
    let injected_to = clock::now();
    // Every injected event meets a full ring and joins the one pending gap.
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(inputs.slots(), RING, "the full ring took a record");

    assert_eq!(inputs.pop(), Ok(filler(0)));
    let popped = Instant::now();
    while inputs.slots() < RING && popped.elapsed() < Duration::from_millis(50) {
        std::thread::sleep(Duration::from_millis(1));
    }
    let waited = popped.elapsed();
    assert_eq!(inputs.slots(), RING, "no gap within 50 ms of the pop");
    println!("RI6: the gap arrived {waited:?} after the pop");

    for n in 1..RING as i64 {
        assert_eq!(inputs.pop(), Ok(filler(n)));
    }
    match inputs.pop() {
        Ok(InputRecord::Gap { from, to }) => {
            assert!(injected_from <= from && from <= to, "{from:?}..{to:?}");
            assert!(
                to.0 - injected_to.0 < 2_000_000,
                "the gap ends long after the injection"
            );
        }
        other => panic!("expected the gap, got {other:?}"),
    }
    // One gap, and nothing after it without further input.
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(inputs.slots(), 0);

    worker.stop(Duration::from_millis(500)).expect("joined");
}
