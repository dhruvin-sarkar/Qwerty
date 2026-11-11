//! RI5 (p2 §2.7). Raw Input registration is process-global, so this is the only `#[test]` in its
//! binary (RUST-RT-20).

use std::time::{Duration, Instant};

use crossbeam_channel::bounded;
use qwerty_app::engine::messages::{RawInputRequest, ThreadEnd, WorkerId};
use qwerty_app::engine::rawinput::{self, registered_targets, INPUT_RING_RECORDS};

#[test]
fn ri5_stop_right_after_spawn_joins_and_leaves_no_registration() {
    for n in 1..=50 {
        let id = WorkerId(n);
        let (ring, _inputs) = rtrb::RingBuffer::new(INPUT_RING_RECORDS);
        let (tx, _status) = bounded(16);
        let (exits_tx, exits) = bounded(16);
        let worker = rawinput::spawn(RawInputRequest {
            worker: id,
            ring,
            tx,
            exits: exits_tx,
        })
        .expect("spawn");

        // No wait for Active: the stop can land at any point of the registration.
        let started = Instant::now();
        worker
            .stop(Duration::from_millis(500))
            .unwrap_or_else(|e| panic!("round {n} did not join within 500 ms: {e:?}"));
        let joined = started.elapsed();
        assert!(joined < Duration::from_millis(500), "round {n}: {joined:?}");
        let exit = exits.recv_timeout(Duration::from_secs(1)).expect("an exit");
        assert_eq!((exit.worker, exit.end), (id, ThreadEnd::Finished));

        // Nothing else in this process registers Raw Input, so any keyboard or mouse usage left
        // would be a Qwerty sink's.
        let targets = registered_targets().expect("query");
        assert_eq!(targets.keyboard, None, "round {n} left its keyboard usage");
        assert_eq!(targets.mouse, None, "round {n} left its mouse usage");
    }
}
