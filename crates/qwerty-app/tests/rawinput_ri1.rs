//! RI1 (p2 §2.7). Raw Input registration is process-global, so this is the only `#[test]` in its
//! binary (RUST-RT-20).

use std::mem::size_of;
use std::time::{Duration, Instant};

use crossbeam_channel::bounded;
use qwerty_app::engine::messages::{
    RawInputMsg, RawInputRequest, RawInputStatus, ThreadEnd, WorkerId,
};
use qwerty_app::engine::protocol::{InputDeviceClass, ThreadName, VetoUnavailableReason};
use qwerty_app::engine::rawinput::{self, registered_targets, INPUT_RING_RECORDS};
use windows::core::w;
use windows::Win32::Foundation::HWND;
use windows::Win32::UI::Input::{
    RegisterRawInputDevices, RAWINPUTDEVICE, RIDEV_INPUTSINK, RIDEV_REMOVE,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DestroyWindow, HWND_MESSAGE, WINDOW_EX_STYLE, WINDOW_STYLE,
};

fn keyboard(
    flags: windows::Win32::UI::Input::RAWINPUTDEVICE_FLAGS,
    target: HWND,
) -> RAWINPUTDEVICE {
    RAWINPUTDEVICE {
        usUsagePage: 0x01,
        usUsage: 0x06,
        dwFlags: flags,
        hwndTarget: target,
    }
}

#[test]
fn ri1_ownership_loss_is_reported_and_stop_spares_the_new_owner() {
    let id = WorkerId(1);
    let (ring, _inputs) = rtrb::RingBuffer::new(INPUT_RING_RECORDS);
    let (tx, status) = bounded(16);
    let (exits_tx, exits) = bounded(16);
    let worker = rawinput::spawn(RawInputRequest {
        worker: id,
        ring,
        tx,
        exits: exits_tx,
        dropped: std::sync::Arc::default(),
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

    // A second message-only window in this process takes the keyboard usage.
    // SAFETY: "STATIC" is a system class; the window is message-only and destroyed below.
    let other = unsafe {
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            w!("STATIC"),
            w!("ri1 second owner"),
            WINDOW_STYLE(0),
            0,
            0,
            0,
            0,
            Some(HWND_MESSAGE),
            None,
            None,
            None,
        )
    }
    .expect("the second message-only window");
    // SAFETY: one valid RAWINPUTDEVICE with the size the API expects.
    unsafe {
        RegisterRawInputDevices(
            &[keyboard(RIDEV_INPUTSINK, other)],
            size_of::<RAWINPUTDEVICE>() as u32,
        )
    }
    .expect("the second registration");
    let registered = Instant::now();

    let lost = status
        .recv_timeout(Duration::from_secs(6))
        .expect("OwnershipLost within 6 s");
    assert!(registered.elapsed() < Duration::from_secs(6));
    assert_eq!(
        lost,
        RawInputMsg {
            worker: id,
            status: RawInputStatus::Unavailable {
                reason: VetoUnavailableReason::OwnershipLost {
                    class: InputDeviceClass::Keyboard,
                },
            },
            skipped_records: 0,
        }
    );

    worker.stop(Duration::from_millis(500)).expect("joined");
    let exit = exits.recv_timeout(Duration::from_secs(1)).expect("an exit");
    assert_eq!(exit.worker, id);
    assert_eq!(exit.thread, ThreadName::RawInput);
    assert_eq!(exit.end, ThreadEnd::Finished);

    // The stop removed only what still targeted the sink: the mouse usage.
    let targets = registered_targets().expect("query");
    assert_eq!(targets.keyboard, Some(other));
    assert_eq!(targets.mouse, None);

    // SAFETY: RIDEV_REMOVE takes a null target; `other` is this thread's own window.
    unsafe {
        RegisterRawInputDevices(
            &[keyboard(RIDEV_REMOVE, HWND::default())],
            size_of::<RAWINPUTDEVICE>() as u32,
        )
        .expect("remove the second registration");
        DestroyWindow(other).expect("destroy the second window");
    }
}
