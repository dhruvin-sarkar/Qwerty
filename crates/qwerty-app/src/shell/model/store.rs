//! The store (shell-v2 §1.3, §2.2): the latest engine snapshot by revision, bounded telemetry
//! and tap buffers (≤ 10 s, B:113), and the ordered queues the UI thread drains. Everything the
//! shell renders is a function of this, the events since and ephemeral view state.

use std::collections::VecDeque;
use std::sync::Arc;

use qwerty_core::units::{HostDuration, HostTime, Milliseconds};

use crate::engine::protocol::{Event, Snapshot, TapReport, TelemetryFrame};
use crate::shell::model::bridge::ShellInput;

/// The longest any drawing buffer reaches back (B:113).
pub const BUFFER_SPAN: Milliseconds = Milliseconds::new(10_000.0);

/// The boundary's frame rates (B:84-85; `Trace` ≤ 10 Hz, B:75): the most frames of each kind that
/// [`BUFFER_SPAN`] can hold.
// SPEC: the engine's cadence constants live in `engine::telemetry`, which the model may not import
// (§1.2's "May use" list); these are the boundary's documented maximum rates, not copies.
pub const METER_CAPACITY: usize = 200; // ≤ 20 Hz × 10 s
pub const SPECTRUM_CAPACITY: usize = 300; // ≤ 30 Hz × 10 s
pub const WAVEFORM_CAPACITY: usize = 300; // ≤ 30 Hz × 10 s (33.3 ms cadence)
pub const TRACE_CAPACITY: usize = 100; // ≤ 10 Hz × 10 s

/// One telemetry frame kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryKind {
    Meter,
    Spectrum,
    Waveform,
    Trace,
}

impl TelemetryKind {
    pub fn of(frame: &TelemetryFrame) -> TelemetryKind {
        match frame {
            TelemetryFrame::Meter { .. } => TelemetryKind::Meter,
            TelemetryFrame::Spectrum { .. } => TelemetryKind::Spectrum,
            TelemetryFrame::Waveform { .. } => TelemetryKind::Waveform,
            TelemetryFrame::Trace { .. } => TelemetryKind::Trace,
        }
    }

    pub fn capacity(self) -> usize {
        match self {
            TelemetryKind::Meter => METER_CAPACITY,
            TelemetryKind::Spectrum => SPECTRUM_CAPACITY,
            TelemetryKind::Waveform => WAVEFORM_CAPACITY,
            TelemetryKind::Trace => TRACE_CAPACITY,
        }
    }
}

/// What one or more applied inputs changed. Combine with `|`.
// SPEC: §4.2 writes `Changes::Rejected`; the store returns a set because a drained batch can change
// several things, so "exactly `Changes::Rejected`" is `Changes { rejected: true, ..Changes::NONE }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Changes {
    /// A newer snapshot replaced the held one.
    pub snapshot: bool,
    pub meter: bool,
    pub spectrum: bool,
    pub waveform: bool,
    pub trace: bool,
    /// A `TapFired` entered the tap buffer.
    pub tap: bool,
    /// A `TapRejected`: the Listening glyph's dip (U:1601); nothing is buffered.
    pub rejected: bool,
    /// A lifecycle event (`CalibrationEnded`, `EvaluationEnded`, `EngineFaulted`, `Stopped`) was
    /// queued; see [`Store::take_events`].
    pub lifecycle: bool,
    /// Another discrete engine event was queued (gestures, action results, calibration feedback
    /// and progress, evaluation updates, migrations); see [`Store::take_events`].
    pub event: bool,
    /// A non-engine input was queued for the app wiring; see [`Store::take_inputs`].
    pub input: bool,
}

impl Changes {
    pub const NONE: Changes = Changes {
        snapshot: false,
        meter: false,
        spectrum: false,
        waveform: false,
        trace: false,
        tap: false,
        rejected: false,
        lifecycle: false,
        event: false,
        input: false,
    };

    pub fn is_empty(self) -> bool {
        self == Changes::NONE
    }

    fn telemetry(kind: TelemetryKind) -> Changes {
        match kind {
            TelemetryKind::Meter => Changes {
                meter: true,
                ..Changes::NONE
            },
            TelemetryKind::Spectrum => Changes {
                spectrum: true,
                ..Changes::NONE
            },
            TelemetryKind::Waveform => Changes {
                waveform: true,
                ..Changes::NONE
            },
            TelemetryKind::Trace => Changes {
                trace: true,
                ..Changes::NONE
            },
        }
    }
}

impl std::ops::BitOr for Changes {
    type Output = Changes;
    fn bitor(self, other: Changes) -> Changes {
        Changes {
            snapshot: self.snapshot || other.snapshot,
            meter: self.meter || other.meter,
            spectrum: self.spectrum || other.spectrum,
            waveform: self.waveform || other.waveform,
            trace: self.trace || other.trace,
            tap: self.tap || other.tap,
            rejected: self.rejected || other.rejected,
            lifecycle: self.lifecycle || other.lifecycle,
            event: self.event || other.event,
            input: self.input || other.input,
        }
    }
}

impl std::ops::BitOrAssign for Changes {
    fn bitor_assign(&mut self, other: Changes) {
        *self = *self | other;
    }
}

/// The store. Owned by the UI thread.
#[derive(Default)]
pub struct Store {
    snapshot: Option<Arc<Snapshot>>,
    meter: VecDeque<TelemetryFrame>,
    spectrum: VecDeque<TelemetryFrame>,
    waveform: VecDeque<TelemetryFrame>,
    trace: VecDeque<TelemetryFrame>,
    taps: VecDeque<TapReport>,
    events: Vec<Event>,
    inputs: Vec<ShellInput>,
}

impl Store {
    pub fn new() -> Store {
        Store::default()
    }

    /// Applies one inbox input. Engine events update the store; every other input is queued, in
    /// order, for the app wiring ([`Store::take_inputs`]).
    // SPEC: §2.2 has `Store::apply(input)` for each drained input. The store holds no engine handle,
    // request flags or window state, so the inputs that carry them pass through it in order.
    pub fn apply(&mut self, input: ShellInput) -> Changes {
        match input {
            ShellInput::Engine(event) => self.apply_event(event),
            ShellInput::Reply { .. }
            | ShellInput::EngineStarted(_)
            | ShellInput::Activate
            | ShellInput::Tray(_)
            | ShellInput::Hotkey
            | ShellInput::ToastActivated(_)
            | ShellInput::SettingChanged(_)
            | ShellInput::WindowState(_) => {
                self.inputs.push(input);
                Changes {
                    input: true,
                    ..Changes::NONE
                }
            }
        }
    }

    /// A snapshot pulled with `handle.snapshot()` (MB:69): kept only when its revision is newer.
    pub fn apply_pulled(&mut self, snapshot: Arc<Snapshot>) -> Changes {
        self.offer(snapshot)
    }

    fn apply_event(&mut self, event: Event) -> Changes {
        match event {
            Event::StateChanged { snapshot } => self.offer(Arc::from(snapshot)),
            Event::Telemetry { frame } => {
                let kind = TelemetryKind::of(&frame);
                let buffer = self.buffer_mut(kind);
                if buffer.len() == kind.capacity() {
                    buffer.pop_front();
                }
                buffer.push_back(frame);
                Changes::telemetry(kind)
            }
            Event::TapFired { tap } => {
                self.push_tap(tap);
                Changes {
                    tap: true,
                    ..Changes::NONE
                }
            }
            Event::TapRejected { .. } => Changes {
                rejected: true,
                ..Changes::NONE
            },
            Event::CalibrationEnded { .. }
            | Event::EvaluationEnded { .. }
            | Event::EngineFaulted { .. }
            | Event::Stopped => {
                self.events.push(event);
                Changes {
                    lifecycle: true,
                    ..Changes::NONE
                }
            }
            Event::GestureArmed { .. }
            | Event::GestureCleared { .. }
            | Event::ActionCompleted { .. }
            | Event::ActionSkipped { .. }
            | Event::CommandFinished { .. }
            | Event::CalibrationFeedback { .. }
            | Event::CalibrationProgress { .. }
            | Event::EvaluationUpdate { .. }
            | Event::ProfileMigrated { .. }
            | Event::ConfigMigrated { .. } => {
                self.events.push(event);
                Changes {
                    event: true,
                    ..Changes::NONE
                }
            }
        }
    }

    /// Keeps the snapshot with the highest revision; an older or equal one is discarded.
    fn offer(&mut self, snapshot: Arc<Snapshot>) -> Changes {
        let newer = match &self.snapshot {
            Some(held) => snapshot.revision > held.revision,
            None => true,
        };
        if newer {
            self.snapshot = Some(snapshot);
            Changes {
                snapshot: true,
                ..Changes::NONE
            }
        } else {
            Changes::NONE
        }
    }

    /// Appends a tap and drops every tap more than [`BUFFER_SPAN`] older than the newest, on the
    /// engine clock (`TapReport.t0`).
    fn push_tap(&mut self, tap: TapReport) {
        self.taps.push_back(tap);
        // INVARIANT: the deque holds at least the tap pushed on the line above.
        let newest: HostTime = self.taps.iter().map(|t| t.t0).max().expect("one tap");
        let span = HostDuration::from_ms(BUFFER_SPAN);
        self.taps.retain(|t| newest.since(t.t0) <= span);
    }

    fn buffer_mut(&mut self, kind: TelemetryKind) -> &mut VecDeque<TelemetryFrame> {
        match kind {
            TelemetryKind::Meter => &mut self.meter,
            TelemetryKind::Spectrum => &mut self.spectrum,
            TelemetryKind::Waveform => &mut self.waveform,
            TelemetryKind::Trace => &mut self.trace,
        }
    }

    /// The held snapshot, if any has arrived.
    pub fn snapshot(&self) -> Option<&Arc<Snapshot>> {
        self.snapshot.as_ref()
    }

    /// The buffered frames of one kind, oldest first.
    pub fn frames(&self, kind: TelemetryKind) -> &VecDeque<TelemetryFrame> {
        match kind {
            TelemetryKind::Meter => &self.meter,
            TelemetryKind::Spectrum => &self.spectrum,
            TelemetryKind::Waveform => &self.waveform,
            TelemetryKind::Trace => &self.trace,
        }
    }

    /// The buffered taps (≤ 10 s by engine clock), oldest first.
    pub fn taps(&self) -> &VecDeque<TapReport> {
        &self.taps
    }

    /// The discrete engine events (lifecycle first-class among them) since the last call, in
    /// arrival order.
    pub fn take_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }

    /// The non-engine inputs since the last call, in arrival order.
    pub fn take_inputs(&mut self) -> Vec<ShellInput> {
        std::mem::take(&mut self.inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::engine::protocol::RejectedBy;
    use crate::shell::model::bridge::WindowState;
    use crate::shell::model::fixture;
    use qwerty_core::model::ModelRejectReason;
    use qwerty_core::units::SampleIndex;

    fn state(revision: u64) -> ShellInput {
        let mut snapshot = fixture::snapshot();
        snapshot.revision = revision;
        ShellInput::Engine(Event::StateChanged {
            snapshot: Box::new(snapshot),
        })
    }

    fn held(store: &Store) -> u64 {
        store.snapshot().expect("a snapshot is held").revision
    }

    #[test]
    fn sq1_stale_revision_is_discarded() {
        let mut store = Store::new();
        assert!(store.apply(state(5)).snapshot);
        assert_eq!(store.apply(state(4)), Changes::NONE, "revision 4 after 5");
        assert_eq!(held(&store), 5);

        let mut pulled = fixture::snapshot();
        pulled.revision = 7;
        assert!(store.apply_pulled(Arc::new(pulled)).snapshot);
        assert_eq!(held(&store), 7, "a pulled 7 replaces 5");
        let mut stale = fixture::snapshot();
        stale.revision = 6;
        assert_eq!(store.apply_pulled(Arc::new(stale)), Changes::NONE);

        // A drained batch of three keeps only the newest.
        let mut changes = Changes::NONE;
        for revision in [9, 11, 10] {
            changes |= store.apply(state(revision));
        }
        assert!(changes.snapshot);
        assert_eq!(held(&store), 11);
    }

    #[test]
    fn sq2_buffers_hold_at_most_ten_seconds() {
        let mut store = Store::new();
        // 400 Waveform frames at the 33.3 ms cadence: 1600 samples apart at 48 kHz.
        for n in 0..400u64 {
            store.apply(ShellInput::Engine(Event::Telemetry {
                frame: fixture::waveform_frame(SampleIndex(4800 + n * 1600)),
            }));
        }
        let frames = store.frames(TelemetryKind::Waveform);
        assert_eq!(frames.len(), WAVEFORM_CAPACITY);
        let end = |frame: &TelemetryFrame| {
            let TelemetryFrame::Waveform { end, .. } = frame else {
                panic!("a Waveform buffer holds Waveform frames, got {frame:?}");
            };
            end.0
        };
        let (first, last) = (
            end(frames.front().expect("full")),
            end(frames.back().expect("full")),
        );
        assert_eq!(last, 4800 + 399 * 1600, "the newest is kept");
        let span_ms = (last - first + 1600) as f64 / 48.0;
        assert!(span_ms <= 10_000.0, "{span_ms} ms buffered");

        // Taps: one every 250 ms for 20 s; only the last 10 s by engine clock stay.
        for n in 0..80i64 {
            let mut tap = fixture::tap_report();
            tap.t0 = HostTime(1_000_000_000 + n * 2_500_000);
            store.apply(ShellInput::Engine(Event::TapFired { tap }));
        }
        let taps = store.taps();
        let newest = taps.back().expect("taps").t0;
        let oldest = taps.front().expect("taps").t0;
        assert_eq!(newest, HostTime(1_000_000_000 + 79 * 2_500_000));
        assert!(newest.since(oldest) <= HostDuration::from_ms(BUFFER_SPAN));
        assert_eq!(taps.len(), 41, "10 s at 250 ms, both ends included");
    }

    #[test]
    fn sq3_apply_names_exactly_what_changed() {
        let mut store = Store::new();
        assert_eq!(
            store.apply(state(1)),
            Changes {
                snapshot: true,
                ..Changes::NONE
            }
        );
        let telemetry = [
            (fixture::meter_frame(), TelemetryKind::Meter),
            (
                fixture::spectrum_frame(SampleIndex(9600)),
                TelemetryKind::Spectrum,
            ),
            (
                fixture::waveform_frame(SampleIndex(9600)),
                TelemetryKind::Waveform,
            ),
            (fixture::trace_frame(), TelemetryKind::Trace),
        ];
        for (frame, kind) in telemetry {
            assert_eq!(
                store.apply(ShellInput::Engine(Event::Telemetry { frame })),
                Changes::telemetry(kind)
            );
            assert_eq!(store.frames(kind).len(), 1);
        }
        assert_eq!(
            store.apply(ShellInput::Engine(Event::TapFired {
                tap: fixture::tap_report()
            })),
            Changes {
                tap: true,
                ..Changes::NONE
            }
        );

        // TapRejected: exactly `rejected`, and nothing buffered anywhere.
        let taps = store.taps().len();
        let rejected = store.apply(ShellInput::Engine(Event::TapRejected {
            rejected: RejectedBy::Model {
                reason: ModelRejectReason::OutOfDistribution,
            },
            best_zone: None,
        }));
        assert_eq!(
            rejected,
            Changes {
                rejected: true,
                ..Changes::NONE
            }
        );
        assert_eq!(store.taps().len(), taps);
        assert!(store.take_events().is_empty());
        for kind in [
            TelemetryKind::Meter,
            TelemetryKind::Spectrum,
            TelemetryKind::Waveform,
            TelemetryKind::Trace,
        ] {
            assert_eq!(store.frames(kind).len(), 1);
        }

        // Lifecycle events are queued in order.
        assert_eq!(
            store.apply(ShellInput::Engine(fixture::engine_faulted())),
            Changes {
                lifecycle: true,
                ..Changes::NONE
            }
        );
        assert_eq!(
            store.apply(ShellInput::Engine(Event::Stopped)),
            Changes {
                lifecycle: true,
                ..Changes::NONE
            }
        );
        let events = store.take_events();
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], Event::EngineFaulted { .. }));
        assert_eq!(events[1], Event::Stopped);

        // A non-engine input passes through to the app wiring.
        assert_eq!(
            store.apply(ShellInput::WindowState(WindowState::Minimised)),
            Changes {
                input: true,
                ..Changes::NONE
            }
        );
        assert!(matches!(
            store.take_inputs().as_slice(),
            [ShellInput::WindowState(WindowState::Minimised)]
        ));
    }
}
