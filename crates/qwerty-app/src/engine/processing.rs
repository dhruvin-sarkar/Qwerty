//! The processing thread `qwerty-dsp` (p2 §2.5). It never waits: every send is `try_send`, every
//! receive `try_recv`, and the only sleep is `park_timeout(50 ms)`, which capture's unpark after
//! every packet cuts short.
//!
//! It owns one `VetoTimeline` for its whole life and lends it to every setup, so open self-output
//! intervals and recent input survive every `Configure` (RUST-RT-9). It never constructs DSP
//! state: the controller builds each setup and moves it in.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::Thread;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, TrySendError};

use qwerty_core::audio::{AudioBlock, AudioFormat};
use qwerty_core::detector::{Detector, DetectorOutput};
use qwerty_core::pipeline::{
    CalibrationFrontEnd, CalibrationInput, Pipeline, PipelineError, PipelineEvent,
};
use qwerty_core::recording::{SegmentKind, SidecarEvent};
use qwerty_core::units::{HostDuration, HostTime, Milliseconds, SampleIndex};
use qwerty_core::veto::{OutputId, SelfOutputEvent, SelfOutputKind, VetoParams, VetoTimeline};

use crate::engine::messages::{
    AmbientChunk, CalibrationStart, InputRecord, OsParts, PacketMeta, ProcessingBody,
    ProcessingCmd, ProcessingDrops, ProcessingHealth, ProcessingMsg, ProcessingSetup, RecorderFeed,
    SetupEpoch, SpeechEnds, StopSignal, ThreadEnd, ThreadExit, Worker, WorkerId,
};
use crate::engine::protocol::{EngineError, Event, TelemetryFrame, ThreadName};
use crate::engine::ring::{PushOutcome, RingReader};
use crate::engine::telemetry::TelemetryBuilder;

/// The loop's idle wait; capture unparks the thread after every packet (§2.5 step 9).
pub const PARK_TIMEOUT: Duration = Duration::from_millis(50);
/// No packet for this long while a setup other than Idle is active is a stall (§2.5 watchdog).
pub const STALL_AFTER: Milliseconds = Milliseconds::new(500.0);
/// `Health` once per second (§2.5 step 7).
pub const HEALTH_INTERVAL: Milliseconds = Milliseconds::new(1000.0);
/// One `AmbientChunk` per ~100 ms of audio (§2.5).
pub const AMBIENT_CHUNK: Milliseconds = Milliseconds::new(100.0);
/// Telemetry events go out only while fewer than this many events are queued: half the events
/// channel (512), the other half being reserved for controller events (§2.5 Messages).
pub const TELEMETRY_QUEUE_LIMIT: usize = 256;
/// The speech FIFO's fixed size (§2.5 step 4).
pub const SPEECH_FIFO: usize = 16;

/// AtomicBool + unpark (§2.5). The thread registers itself on its first instruction; a signal
/// before that is still seen, because the loop checks the flag before it ever parks.
struct ProcessingStop {
    flag: Arc<AtomicBool>,
    thread: Arc<OnceLock<Thread>>,
}

impl StopSignal for ProcessingStop {
    fn signal(&self) {
        self.flag.store(true, Ordering::Release);
        if let Some(thread) = self.thread.get() {
            thread.unpark();
        }
    }
}

/// Spawns `qwerty-dsp`.
// SPEC: §2.5 returns `Result<Worker, EngineError>`, but the controller must pass processing's
// `Thread` to every capture open (`CaptureRequest.wake`, §2.3) and `Worker` exposes no thread, so
// spawn also returns the thread handle it registered.
#[allow(clippy::too_many_arguments)]
pub fn spawn(
    worker: WorkerId,
    cmds: Receiver<ProcessingCmd>,
    tx: Sender<ProcessingMsg>,
    events: Sender<Event>,
    self_output: Receiver<SelfOutputEvent>,
    speech: Arc<SpeechEnds>,
    os: Arc<dyn OsParts>,
    exits: Sender<ThreadExit>,
) -> Result<(Worker, Thread), EngineError> {
    let flag = Arc::new(AtomicBool::new(false));
    let registered = Arc::new(OnceLock::new());
    let stop = ProcessingStop {
        flag: Arc::clone(&flag),
        thread: Arc::clone(&registered),
    };
    let (thread_tx, thread_rx) = crossbeam_channel::bounded::<Thread>(1);
    let body_registered = Arc::clone(&registered);
    let worker = Worker::spawn(
        worker,
        ThreadName::Processing,
        Box::new(stop),
        exits,
        move || {
            let me = std::thread::current();
            if body_registered.set(me.clone()).is_err() {
                // This body runs once, so the cell is empty here.
                tracing::error!("the processing thread registered twice");
            }
            // spawn() waits on this receiver until the send lands.
            if thread_tx.send(me).is_err() {
                tracing::error!("processing spawn stopped waiting for the thread handle");
                return ThreadEnd::Finished;
            }
            let mut processing = Processing::new(tx, events, self_output, speech, os);
            processing.run(&cmds, &flag);
            ThreadEnd::Finished
        },
    )?;
    let thread = thread_rx.recv().map_err(|_| EngineError::ThreadSpawn {
        thread: ThreadName::Processing,
        detail: "the processing thread ended before it registered".to_string(),
    })?;
    Ok((worker, thread))
}

/// The installed setup, as processing runs it.
enum Setup {
    Idle,
    Calibration {
        front_end: Box<CalibrationFrontEnd>,
        forward_ambient: bool,
    },
    Pipeline {
        pipeline: Box<Pipeline>,
    },
    Monitor {
        detector: Box<Detector>,
    },
}

impl Setup {
    fn from_message(setup: ProcessingSetup) -> Setup {
        match setup {
            ProcessingSetup::Idle => Setup::Idle,
            ProcessingSetup::Calibration {
                front_end,
                forward_ambient,
            } => Setup::Calibration {
                front_end,
                forward_ambient,
            },
            ProcessingSetup::Pipeline { pipeline } => Setup::Pipeline { pipeline },
            ProcessingSetup::Monitor { detector } => Setup::Monitor { detector },
        }
    }

    fn detector(&self) -> Option<&Detector> {
        match self {
            Setup::Idle => None,
            Setup::Calibration { front_end, .. } => Some(front_end.detector()),
            Setup::Pipeline { pipeline } => Some(pipeline.detector()),
            Setup::Monitor { detector } => Some(detector),
        }
    }

    fn is_idle(&self) -> bool {
        matches!(self, Setup::Idle)
    }
}

/// The speech FIFO (§2.5 step 4): a fixed array of 16 ids. On overflow the oldest id is dropped,
/// which a later `Began{Speech}` has already closed (§1.13.2).
struct SpeechFifo {
    ids: [OutputId; SPEECH_FIFO],
    head: usize,
    len: usize,
}

impl SpeechFifo {
    fn new() -> SpeechFifo {
        SpeechFifo {
            ids: [OutputId(0); SPEECH_FIFO],
            head: 0,
            len: 0,
        }
    }

    fn push(&mut self, id: OutputId) {
        if self.len == SPEECH_FIFO {
            self.head = (self.head + 1) % SPEECH_FIFO;
            self.len -= 1;
        }
        self.ids[(self.head + self.len) % SPEECH_FIFO] = id;
        self.len += 1;
    }

    fn pop(&mut self) -> Option<OutputId> {
        if self.len == 0 {
            return None;
        }
        let id = self.ids[self.head];
        self.head = (self.head + 1) % SPEECH_FIFO;
        self.len -= 1;
        Some(id)
    }
}

/// The ~100 ms chunk being collected while the calibration step is Ambient.
struct AmbientCollector {
    chunk: Option<AmbientChunk>,
}

impl AmbientCollector {
    /// Appends `block`; returns a chunk once it holds ≥ `AMBIENT_CHUNK`, or the previous chunk
    /// when `block` starts a discontinuity (the new chunk then starts at `block`).
    fn push(&mut self, block: &AudioBlock<'_>, out: &mut Vec<AmbientChunk>) {
        let channels = usize::from(block.format.channels.get());
        if block.discontinuity {
            if let Some(chunk) = self.chunk.take() {
                out.push(chunk);
            }
        }
        let chunk = self.chunk.get_or_insert_with(|| AmbientChunk {
            start: block.start,
            samples: Vec::with_capacity(
                block.format.rate.samples_in(AMBIENT_CHUNK) * channels + block.samples.len(),
            ),
            time: block.time,
            discontinuity: block.discontinuity,
        });
        chunk.samples.extend_from_slice(block.samples);
        if chunk.samples.len() / channels >= block.format.rate.samples_in(AMBIENT_CHUNK) {
            if let Some(chunk) = self.chunk.take() {
                out.push(chunk);
            }
        }
    }

    fn clear(&mut self) {
        self.chunk = None;
    }
}

/// The recorder feed and its pending structure events for the current setup.
struct Feed {
    feed: RecorderFeed,
    /// A `Dropped` push marks the next pushed block `discontinuity` (§2.13).
    next_discontinuity: bool,
}

struct Processing {
    tx: Sender<ProcessingMsg>,
    events: Sender<Event>,
    self_output: Receiver<SelfOutputEvent>,
    speech: Arc<SpeechEnds>,
    os: Arc<dyn OsParts>,

    epoch: SetupEpoch,
    setup: Setup,
    segment: SegmentKind,
    calibration_started: Option<CalibrationStart>,
    /// The next block is the first under the current Configure.
    first_block: bool,

    ring: Option<RingReader>,
    buf: Vec<f32>,
    inputs: Option<rtrb::Consumer<InputRecord>>,
    feed: Option<Feed>,

    veto: VetoTimeline,
    fifo: SpeechFifo,
    speech_seen: u64,
    speech_unmatched: u64,

    telemetry: TelemetryBuilder,
    frames: Vec<TelemetryFrame>,
    pipeline_events: Vec<PipelineEvent>,
    calibration_inputs: Vec<CalibrationInput>,
    detected: Vec<DetectorOutput>,
    ambient: AmbientCollector,
    ambient_chunks: Vec<AmbientChunk>,

    drops: ProcessingDrops,
    stuck: Vec<OutputId>,
    next_health: Option<HostTime>,
    last_packet: Option<HostTime>,
    stalled: bool,
}

impl Processing {
    fn new(
        tx: Sender<ProcessingMsg>,
        events: Sender<Event>,
        self_output: Receiver<SelfOutputEvent>,
        speech: Arc<SpeechEnds>,
        os: Arc<dyn OsParts>,
    ) -> Processing {
        Processing {
            tx,
            events,
            self_output,
            speech,
            os,
            epoch: SetupEpoch(0),
            setup: Setup::Idle,
            segment: SegmentKind::Idle,
            calibration_started: None,
            first_block: false,
            ring: None,
            buf: Vec::new(),
            inputs: None,
            feed: None,
            veto: VetoTimeline::new(VetoParams::DEFAULT),
            fifo: SpeechFifo::new(),
            speech_seen: 0,
            speech_unmatched: 0,
            telemetry: TelemetryBuilder::new(),
            frames: Vec::with_capacity(4),
            pipeline_events: Vec::with_capacity(32),
            calibration_inputs: Vec::with_capacity(32),
            detected: Vec::with_capacity(64),
            ambient: AmbientCollector { chunk: None },
            ambient_chunks: Vec::with_capacity(4),
            drops: ProcessingDrops::default(),
            stuck: Vec::new(),
            next_health: None,
            last_packet: None,
            stalled: false,
        }
    }

    fn run(&mut self, cmds: &Receiver<ProcessingCmd>, stop: &AtomicBool) {
        loop {
            // 1. One clock read per iteration.
            let now = self.os.now();
            // 2. Commands, then the stop flag.
            while let Ok(cmd) = cmds.try_recv() {
                self.command(cmd);
            }
            if stop.load(Ordering::Acquire) {
                return;
            }
            // 3. Input.
            self.drain_inputs();
            // 4. Self-output.
            self.drain_self_output();
            // 5. Audio.
            self.drain_audio(now);
            // 6. Advance the setup.
            self.advance(now);
            // 7. Health and the watchdog.
            self.health(now);
            // 8. Telemetry.
            self.telemetry_due(now);
            // 9. Wait.
            std::thread::park_timeout(PARK_TIMEOUT);
        }
    }

    fn command(&mut self, cmd: ProcessingCmd) {
        match cmd {
            ProcessingCmd::Configure {
                epoch,
                setup,
                segment,
                calibration_started,
                ring,
            } => {
                self.epoch = epoch;
                self.setup = Setup::from_message(setup);
                self.segment = segment;
                self.calibration_started = calibration_started;
                self.first_block = true;
                self.ambient.clear();
                self.last_packet = None;
                self.stalled = false;
                if let Some(ring) = ring {
                    self.buf = Vec::with_capacity(ring.max_packet_samples());
                    self.telemetry.set_format(ring.format());
                    self.ring = Some(ring);
                }
                if self.setup.is_idle() {
                    self.ring = None;
                    self.telemetry.clear_format();
                }
            }
            ProcessingCmd::SetInputRing { inputs } => self.inputs = inputs,
            ProcessingCmd::SetTelemetry { level } => {
                self.telemetry.set_level(level);
                if let Some(ring) = &self.ring {
                    self.telemetry.set_format(ring.format());
                }
            }
            ProcessingCmd::SetRecorder { feed } => {
                self.feed = feed.map(|feed| Feed {
                    feed,
                    next_discontinuity: false,
                });
            }
        }
    }

    fn drain_inputs(&mut self) {
        let Some(inputs) = self.inputs.as_mut() else {
            return;
        };
        while let Ok(record) = inputs.pop() {
            match record {
                InputRecord::Event(event) => {
                    self.veto.push_input(event);
                    if let Some(feed) = &self.feed {
                        if feed
                            .feed
                            .events
                            .try_send(SidecarEvent::Input { event })
                            .is_err()
                        {
                            self.drops.recorder_events += 1;
                        }
                    }
                }
                InputRecord::Gap { from, to } => self.veto.mark_input_gap(from, to),
            }
        }
    }

    fn drain_self_output(&mut self) {
        // (1) the finish count before draining, so every Began sent before a finish is applied
        // first (Acquire pairs with the observer's Release).
        let finished = self.speech.finished.load(Ordering::Acquire);
        let last_at = HostTime(self.speech.last_at.load(Ordering::Relaxed));
        // (2)
        while let Ok(event) = self.self_output.try_recv() {
            if let SelfOutputEvent::Began {
                id,
                kind: SelfOutputKind::Speech,
                ..
            } = event
            {
                self.fifo.push(id);
            }
            self.apply_self_output(event);
        }
        // (3)
        let new = finished.saturating_sub(self.speech_seen);
        self.speech_seen = finished;
        for _ in 0..new {
            match self.fifo.pop() {
                Some(id) => self.apply_self_output(SelfOutputEvent::Ended { id, at: last_at }),
                None => self.speech_unmatched += 1,
            }
        }
    }

    fn apply_self_output(&mut self, event: SelfOutputEvent) {
        self.veto.self_output(event);
        if let Some(feed) = &self.feed {
            if feed
                .feed
                .events
                .try_send(SidecarEvent::SelfOutput { event })
                .is_err()
            {
                self.drops.recorder_events += 1;
            }
        }
    }

    fn drain_audio(&mut self, now: HostTime) {
        // Taken for the loop; `block` never touches the ring, so it is put back unchanged.
        let Some(mut ring) = self.ring.take() else {
            return;
        };
        let format = ring.format();
        let mut buf = std::mem::take(&mut self.buf);
        while let Some(meta) = ring.pop(&mut buf) {
            self.last_packet = Some(now);
            self.stalled = false;
            self.block(format, meta, &buf, now);
        }
        self.buf = buf;
        self.ring = Some(ring);
    }

    fn block(&mut self, format: AudioFormat, meta: PacketMeta, samples: &[f32], now: HostTime) {
        let block = AudioBlock {
            format,
            start: meta.start,
            samples,
            time: meta.time,
            discontinuity: meta.discontinuity,
        };
        if self.first_block {
            self.first_block = false;
            self.structure_events(block.start);
        }
        if let Some(feed) = self.feed.as_mut() {
            let pushed = PacketMeta {
                discontinuity: meta.discontinuity || feed.next_discontinuity,
                ..meta
            };
            match feed.feed.ring.push(pushed, samples) {
                PushOutcome::Written => feed.next_discontinuity = false,
                PushOutcome::Dropped { frames } => {
                    self.drops.recorder_frames += u64::from(frames);
                    feed.next_discontinuity = true;
                }
            }
        }
        self.telemetry.on_block(&block);
        let result = match &mut self.setup {
            Setup::Idle => Ok(()),
            Setup::Pipeline { pipeline } => {
                pipeline.push_block(&block, now, &mut self.veto, &mut self.pipeline_events)
            }
            Setup::Calibration {
                front_end,
                forward_ambient,
            } => {
                if *forward_ambient {
                    self.ambient.push(&block, &mut self.ambient_chunks);
                }
                front_end
                    .push_block(&block, now, &mut self.veto, &mut self.calibration_inputs)
                    .map_err(|error| PipelineError::Detector { error })
            }
            Setup::Monitor { detector } => {
                self.detected.clear();
                detector
                    .push(&block, &mut self.detected)
                    .map_err(|error| PipelineError::Detector { error })
            }
        };
        // Outputs produced before an error still go out, in the order the setup produced them.
        self.flush_pipeline_events(now);
        self.flush_calibration_inputs();
        self.flush_detected(now);
        self.flush_ambient();
        if let Err(error) = result {
            self.send(ProcessingBody::SetupFailed { error });
            self.setup = Setup::Idle;
        }
    }

    /// The first block under a Configure: CalibrationStarted (if that Configure carried it), then
    /// the Segment, both at `start` (§2.5, §2.13). A refusal of either sends `SegmentLost`.
    fn structure_events(&mut self, start: SampleIndex) {
        let Some(feed) = &self.feed else {
            return;
        };
        let mut lost = false;
        if let Some(c) = self.calibration_started {
            lost |= feed
                .feed
                .events
                .try_send(SidecarEvent::CalibrationStarted {
                    start,
                    seed: c.seed,
                    taps_per_zone: c.taps_per_zone,
                })
                .is_err();
        }
        lost |= feed
            .feed
            .events
            .try_send(SidecarEvent::Segment {
                start,
                kind: self.segment.clone(),
                reconfigured: true,
            })
            .is_err();
        if lost {
            self.drops.recorder_events += 1;
            self.send(ProcessingBody::SegmentLost);
        }
    }

    fn advance(&mut self, now: HostTime) {
        match &mut self.setup {
            Setup::Idle | Setup::Monitor { .. } => {}
            Setup::Pipeline { pipeline } => {
                pipeline.advance(now, &mut self.veto, &mut self.pipeline_events);
                self.flush_pipeline_events(now);
            }
            Setup::Calibration { front_end, .. } => {
                front_end.advance(now, &mut self.veto, &mut self.calibration_inputs);
                self.flush_calibration_inputs();
            }
        }
    }

    fn flush_ambient(&mut self) {
        let mut chunks = std::mem::take(&mut self.ambient_chunks);
        for chunk in chunks.drain(..) {
            self.send(ProcessingBody::AmbientChunk { chunk });
        }
        self.ambient_chunks = chunks;
    }

    fn flush_pipeline_events(&mut self, now: HostTime) {
        let mut events = std::mem::take(&mut self.pipeline_events);
        for event in events.drain(..) {
            let body = match event {
                PipelineEvent::Decision(decision) => ProcessingBody::Decision { decision },
                PipelineEvent::Gesture(event) => ProcessingBody::Gesture { event, at: now },
                PipelineEvent::DetectorRejected(rejected) => {
                    ProcessingBody::DetectorRejected { rejected, at: now }
                }
                PipelineEvent::InputFault(fault) => ProcessingBody::Fault { fault },
            };
            self.send(body);
        }
        self.pipeline_events = events;
    }

    fn flush_calibration_inputs(&mut self) {
        let mut inputs = std::mem::take(&mut self.calibration_inputs);
        for input in inputs.drain(..) {
            self.send(ProcessingBody::Calibration { input });
        }
        self.calibration_inputs = inputs;
    }

    fn flush_detected(&mut self, now: HostTime) {
        let mut detected = std::mem::take(&mut self.detected);
        for output in detected.drain(..) {
            let body = match output {
                DetectorOutput::Candidate(candidate) => ProcessingBody::Detected {
                    candidate: candidate.id,
                    onset: candidate.onset,
                    gates: candidate.gates,
                    at: now,
                },
                DetectorOutput::Rejected(rejected) => {
                    ProcessingBody::DetectorRejected { rejected, at: now }
                }
                DetectorOutput::InputFault(fault) => ProcessingBody::Fault { fault },
            };
            self.send(body);
        }
        self.detected = detected;
    }

    /// `try_send` to the controller; a failed Decision or Gesture counts `decisions`, any other
    /// `messages` (§2.5 Messages).
    fn send(&mut self, body: ProcessingBody) {
        let decision = matches!(
            body,
            ProcessingBody::Decision { .. } | ProcessingBody::Gesture { .. }
        );
        match self.tx.try_send(ProcessingMsg {
            epoch: self.epoch,
            body,
        }) {
            Ok(()) => {}
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
                if decision {
                    self.drops.decisions += 1;
                } else {
                    self.drops.messages += 1;
                }
            }
        }
    }

    fn health(&mut self, now: HostTime) {
        // The watchdog: no packet for 500 ms while a setup other than Idle is active.
        if !self.setup.is_idle() && !self.stalled {
            let since = match self.last_packet {
                Some(at) => now.since(at),
                None => HostDuration(0),
            };
            if self.last_packet.is_none() {
                // The setup's clock starts at its first loop iteration.
                self.last_packet = Some(now);
            } else if since >= HostDuration::from_ms(STALL_AFTER) {
                self.stalled = true;
                self.send_health();
            }
        }
        let due = match self.next_health {
            Some(at) => now >= at,
            None => true,
        };
        if due {
            self.next_health = now.checked_add(HostDuration::from_ms(HEALTH_INTERVAL));
            self.stuck = self.veto.close_stuck(now);
            self.veto.prune(now);
            self.send_health();
            self.stuck.clear();
        }
    }

    fn send_health(&mut self) {
        let pipeline = match &self.setup {
            Setup::Pipeline { pipeline } => Some(pipeline.counters()),
            Setup::Idle | Setup::Calibration { .. } | Setup::Monitor { .. } => None,
        };
        let health = ProcessingHealth {
            detector: self.setup.detector().map(Detector::health),
            pipeline,
            veto: self.veto.counters(),
            stuck_outputs: self.stuck.clone(),
            speech_unmatched: self.speech_unmatched,
            dropped: self.drops,
            stalled: self.stalled,
        };
        self.send(ProcessingBody::Health { health });
    }

    fn telemetry_due(&mut self, now: HostTime) {
        let mut frames = std::mem::take(&mut self.frames);
        self.telemetry.due(now, self.setup.detector(), &mut frames);
        for frame in frames.drain(..) {
            if self.events.len() >= TELEMETRY_QUEUE_LIMIT {
                self.drops.telemetry += 1;
                continue;
            }
            match self.events.try_send(Event::Telemetry { frame }) {
                Ok(()) => {}
                Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
                    self.drops.telemetry += 1;
                }
            }
        }
        self.frames = frames;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speech_fifo_drops_the_oldest_on_overflow() {
        let mut fifo = SpeechFifo::new();
        for i in 0..20 {
            fifo.push(OutputId(i));
        }
        let popped: Vec<u64> = std::iter::from_fn(|| fifo.pop()).map(|id| id.0).collect();
        assert_eq!(popped, (4..20).collect::<Vec<u64>>());
        assert_eq!(fifo.pop(), None);
    }
}
