//! RI3 (p2 §2.7). Raw Input registration is process-global, so this is the only `#[test]` in its
//! binary (RUST-RT-20).

use std::time::{Duration, Instant};

use crossbeam_channel::bounded;
use qwerty_app::engine::messages::{
    RawInputMsg, RawInputRequest, RawInputStatus, ThreadEnd, WorkerId,
};
use qwerty_app::engine::protocol::ThreadName;
use qwerty_app::engine::rawinput::{self, registered_targets, INPUT_RING_RECORDS};

#[test]
fn ri3_stop_joins_and_a_fresh_spawn_is_active_again() {
    // Three spawns in one process: the second and third find the window class registered.
    for n in 1..=3 {
        let id = WorkerId(n);
        let (ring, _inputs) = rtrb::RingBuffer::new(INPUT_RING_RECORDS);
        let (tx, status) = bounded(16);
        let (exits_tx, exits) = bounded(16);
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
            },
            "spawn {n}"
        );
        assert_eq!(worker.id(), id);

        let started = Instant::now();
        worker
            .stop(Duration::from_millis(500))
            .unwrap_or_else(|e| panic!("spawn {n} did not join within 500 ms: {e:?}"));
        let joined = started.elapsed();
        assert!(joined < Duration::from_millis(500), "spawn {n}: {joined:?}");

        let exit = exits.recv_timeout(Duration::from_secs(1)).expect("an exit");
        assert_eq!(exit.worker, id);
        assert_eq!(exit.thread, ThreadName::RawInput);
        assert_eq!(exit.end, ThreadEnd::Finished);

        let targets = registered_targets().expect("query");
        assert_eq!(targets.keyboard, None, "spawn {n} left its keyboard usage");
        assert_eq!(targets.mouse, None, "spawn {n} left its mouse usage");
    }
}
