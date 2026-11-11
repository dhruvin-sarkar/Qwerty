//! Debug-capture session recorder (p2 §2.13). The `qwerty-rec` thread owns the one
//! `RecordingWriter`: it drains the recorder ring that processing fills, records the sidecar events
//! that processing (the feed) and the controller (`RecorderCmd::Event`) send, and reports to the
//! controller with `RecorderMsg`. Recordings live only under `<data_dir>/DebugCaptures`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Select, Sender, TryRecvError, TrySendError};

use qwerty_core::audio::{AudioBlock, AudioFormat};
use qwerty_core::recording::{
    RecordingError, RecordingSummary, RecordingWriter, SidecarEvent, SidecarHeader,
};
use qwerty_core::units::{SampleIndex, Seconds};

use crate::engine::messages::{
    RecorderCmd, RecorderDisk, RecorderMsg, StopSignal, ThreadEnd, ThreadExit, Worker, WorkerId,
};
use crate::engine::protocol::{EngineError, RecorderStopReason, StoredRecordings, ThreadName};
use crate::engine::ring::RingReader;

/// Refuse to start below 2 GiB free (RecorderDisk::free_bytes; an Err is an I/O error), and stop
/// a recording once free space falls below it.
pub const MIN_FREE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
/// ≈ 93 min of 48 kHz stereo f32; far inside WAV's 4 GiB.
pub const RECORDING_SIZE_LIMIT: u64 = 2 * 1024 * 1024 * 1024;

/// The ring has no wake-up, so while recording the loop polls it this often (it holds 2 s).
const RECORDING_POLL: Duration = Duration::from_millis(20);
/// The longest wait (§2.13: ≤ 100 ms), so a stop signal is seen promptly.
const IDLE_POLL: Duration = Duration::from_millis(100);
/// Progress, and the free-space check, once per second.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(1);
/// Bytes per f32 sample in the WAV (the capture format).
const SAMPLE_BYTES: u64 = std::mem::size_of::<f32>() as u64;

/// `disk` = os.recorder_disk() (§2.2): free space and a per-write hook, so tests control both.
/// The stop signal is an AtomicBool; the loop waits ≤ 100 ms.
pub fn spawn(
    worker: WorkerId,
    cmds: Receiver<RecorderCmd>,
    feed_events: Receiver<SidecarEvent>,
    tx: Sender<RecorderMsg>,
    disk: Arc<dyn RecorderDisk>,
    exits: Sender<ThreadExit>,
) -> Result<Worker, EngineError> {
    let stop = Arc::new(AtomicBool::new(false));
    let stopped = Arc::clone(&stop);
    let recorder = Recorder {
        cmds,
        feed_events,
        tx,
        disk,
        active: None,
    };
    Worker::spawn(
        worker,
        ThreadName::Recorder,
        Box::new(FlagStop(stop)),
        exits,
        move || {
            recorder.run(&stopped);
            ThreadEnd::Finished
        },
    )
}

/// DebugCaptures/ measured (PARITY-28): the directories directly under `root` and the bytes of
/// every file inside them. A missing folder holds nothing.
// SPEC: §2.13 has the recorder measure DebugCaptures/ at startup and after DeleteDebugRecordings,
// but `spawn` takes no folder and `RecorderCmd` has no delete, so the thread can only measure after
// a recording ends (it learns the folder from `Start`). The controller calls this function, and
// `delete_recordings`, for the other two cases.
pub fn stored_recordings(root: &Path) -> std::io::Result<StoredRecordings> {
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(StoredRecordings { count: 0, bytes: 0 })
        }
        Err(e) => return Err(e),
    };
    let mut stored = StoredRecordings { count: 0, bytes: 0 };
    for entry in entries {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            stored.count += 1;
            stored.bytes += folder_bytes(&entry.path())?;
        }
    }
    Ok(stored)
}

/// DeleteDebugRecordings: removes every directory under `root`. The controller refuses it while a
/// recording runs. A missing folder is already empty.
pub fn delete_recordings(root: &Path) -> std::io::Result<()> {
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    for entry in entries {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            std::fs::remove_dir_all(entry.path())?;
        }
    }
    Ok(())
}

/// The bytes of every regular file under `dir`, recursively (links are not followed).
fn folder_bytes(dir: &Path) -> std::io::Result<u64> {
    let mut bytes = 0;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            bytes += folder_bytes(&entry.path())?;
        } else if kind.is_file() {
            bytes += entry.metadata()?.len();
        }
    }
    Ok(bytes)
}

struct FlagStop(Arc<AtomicBool>);

impl StopSignal for FlagStop {
    fn signal(&self) {
        self.0.store(true, Ordering::Release);
    }
}

struct Recorder {
    cmds: Receiver<RecorderCmd>,
    feed_events: Receiver<SidecarEvent>,
    tx: Sender<RecorderMsg>,
    disk: Arc<dyn RecorderDisk>,
    active: Option<Active>,
}

/// One running recording.
struct Active {
    writer: RecordingWriter,
    ring: RingReader,
    /// DebugCaptures/, measured when the recording ends.
    root: PathBuf,
    format: AudioFormat,
    /// Stream position the next block continues at; None before the first block.
    next: Option<SampleIndex>,
    /// Frames written to the WAV.
    frames: u64,
    /// Gaps recorded, counted as `RecordingWriter::write_block` records them.
    gaps: u32,
    /// One packet's samples, reused.
    samples: Vec<f32>,
    last_progress: Instant,
}

/// Whether the loop goes on: a disconnected command or feed channel means the controller is gone.
enum Flow {
    Continue,
    Exit,
}

/// How a drain of the ring ended.
enum Drained {
    Empty,
    /// The next packet would take the recording past RECORDING_SIZE_LIMIT.
    LimitReached,
}

impl Recorder {
    fn run(mut self, stop: &AtomicBool) {
        while !stop.load(Ordering::Acquire) {
            match self.step() {
                Flow::Continue => self.wait(),
                Flow::Exit => break,
            }
        }
        // Shutdown sends Stop{CaptureStopped} first (§2.11); a recording still running here is
        // finished the same way, so its files are complete.
        self.stop(RecorderStopReason::CaptureStopped);
    }

    /// Commands, then feed events, then the ring, then the once-per-second checks.
    fn step(&mut self) -> Flow {
        // Commands first: the controller sends Start before SetRecorder, so a recording is in
        // place before its feed's first event.
        loop {
            match self.cmds.try_recv() {
                Ok(cmd) => self.on_cmd(cmd),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return Flow::Exit,
            }
        }
        loop {
            match self.feed_events.try_recv() {
                Ok(event) => {
                    if let Flow::Exit = self.place(event) {
                        return Flow::Exit;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return Flow::Exit,
            }
        }
        self.drain_ring();
        self.tick();
        Flow::Continue
    }

    /// Waits until a command or an event is ready, or the poll interval passes; the next step
    /// takes whatever arrived.
    fn wait(&self) {
        let poll = match self.active {
            Some(_) => RECORDING_POLL,
            None => IDLE_POLL,
        };
        let mut select = Select::new();
        select.recv(&self.cmds);
        select.recv(&self.feed_events);
        match select.ready_timeout(poll) {
            Ok(_) | Err(crossbeam_channel::ReadyTimeoutError) => {}
        }
    }

    fn on_cmd(&mut self, cmd: RecorderCmd) {
        match cmd {
            RecorderCmd::Start { root, header, ring } => self.start(root, *header, ring),
            RecorderCmd::Event { event } => self.record(event),
            RecorderCmd::Stop { reason } => self.stop(reason),
        }
    }

    /// A feed event belongs to the recording its feed was set for. A command that arrived after
    /// the last command drain was sent before the event's feed existed if it is a Start, so it is
    /// applied first; a Stop already queued ends the recording this event was recorded for.
    // SPEC: the feed events of every recording share one channel (`spawn`'s `feed_events`) with no
    // per-recording mark, so a recorder that lags processing by a whole Stop → Start → SetRecorder
    // cycle would put the new recording's first events into the stopped one. Commands are applied
    // in this order to make that the only case.
    fn place(&mut self, event: SidecarEvent) -> Flow {
        loop {
            match self.cmds.try_recv() {
                Ok(RecorderCmd::Stop { reason }) => {
                    self.record(event);
                    self.stop(reason);
                    return Flow::Continue;
                }
                Ok(cmd) => self.on_cmd(cmd),
                Err(TryRecvError::Empty) => {
                    self.record(event);
                    return Flow::Continue;
                }
                Err(TryRecvError::Disconnected) => {
                    self.record(event);
                    return Flow::Exit;
                }
            }
        }
    }

    /// Into the running recording; with none running the event has no recording to belong to.
    fn record(&mut self, event: SidecarEvent) {
        if let Some(active) = &mut self.active {
            active.writer.push_event(event);
        }
    }

    fn start(&mut self, root: PathBuf, header: SidecarHeader, ring: RingReader) {
        if self.active.is_some() {
            tracing::error!(
                "the recorder got Start while a recording ran; that recording stops as \
                 CaptureStopped"
            );
            self.stop(RecorderStopReason::CaptureStopped);
        }
        let format = header.device.format;
        if ring.format() != format {
            self.send(RecorderMsg::Failed {
                error: RecordingError::FormatMismatch,
            });
            return;
        }
        // The free-space query needs the folder to exist.
        if let Err(e) = std::fs::create_dir_all(&root) {
            self.send(RecorderMsg::Failed { error: e.into() });
            return;
        }
        match self.disk.free_bytes(&root) {
            Err(e) => {
                self.send(RecorderMsg::Failed { error: e.into() });
                return;
            }
            Ok(free_bytes) if free_bytes < MIN_FREE_BYTES => {
                // SPEC: a refused start has no recording directory and nothing recorded; the
                // summary names the folder it would have been made in.
                self.send(RecorderMsg::Stopped {
                    summary: RecordingSummary {
                        dir: root.display().to_string(),
                        frames: 0,
                        bytes: 0,
                        events: 0,
                        gaps: 0,
                    },
                    reason: RecorderStopReason::DiskSpaceLow { free_bytes },
                });
                return;
            }
            Ok(_) => {}
        }
        match RecordingWriter::create(&root, header) {
            Ok(writer) => {
                self.send(RecorderMsg::Started {
                    dir: writer.dir().display().to_string(),
                });
                let samples = Vec::with_capacity(ring.max_packet_samples());
                self.active = Some(Active {
                    writer,
                    ring,
                    root,
                    format,
                    next: None,
                    frames: 0,
                    gaps: 0,
                    samples,
                    last_progress: Instant::now(),
                });
            }
            Err(error) => {
                self.send(RecorderMsg::Failed { error });
                self.report_stored(&root);
            }
        }
    }

    /// A stop the controller asked for (or shutdown): the recording keeps everything that already
    /// reached the recorder, the ring and the events queued now, then is finished.
    fn stop(&mut self, reason: RecorderStopReason) {
        let Some(mut active) = self.active.take() else {
            // Already ended on its own (size, disk space, an I/O error): nothing to stop.
            return;
        };
        let reason = match active.drain(self.disk.as_ref()) {
            Ok(Drained::Empty) => reason,
            Ok(Drained::LimitReached) => RecorderStopReason::SizeLimit,
            Err(error) => {
                self.fail(active, error);
                return;
            }
        };
        for _ in 0..self.feed_events.len() {
            match self.feed_events.try_recv() {
                Ok(event) => active.writer.push_event(event),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
        self.finish(active, reason);
    }

    /// Finalizes the WAV and writes the finished sidecar.
    fn finish(&mut self, active: Active, reason: RecorderStopReason) {
        let root = active.root.clone();
        match active.writer.finish() {
            Ok(summary) => self.send(RecorderMsg::Stopped { summary, reason }),
            Err(error) => self.send(RecorderMsg::Failed { error }),
        }
        self.report_stored(&root);
    }

    /// An I/O error ends the recording where it is; its sidecar stays `finished: false`.
    fn fail(&mut self, active: Active, error: RecordingError) {
        let root = active.root.clone();
        drop(active);
        self.send(RecorderMsg::Failed { error });
        self.report_stored(&root);
    }

    fn drain_ring(&mut self) {
        let Some(active) = &mut self.active else {
            return;
        };
        let drained = active.drain(self.disk.as_ref());
        match drained {
            Ok(Drained::Empty) => {}
            Ok(Drained::LimitReached) => {
                if let Some(active) = self.active.take() {
                    self.finish(active, RecorderStopReason::SizeLimit);
                }
            }
            Err(error) => {
                if let Some(active) = self.active.take() {
                    self.fail(active, error);
                }
            }
        }
    }

    /// Once per second while recording: the free-space check, then Progress.
    fn tick(&mut self) {
        let Some(active) = &mut self.active else {
            return;
        };
        if active.last_progress.elapsed() < PROGRESS_INTERVAL {
            return;
        }
        active.last_progress = Instant::now();
        let free = self.disk.free_bytes(&active.root);
        let progress = active.progress();
        match free {
            Ok(free_bytes) if free_bytes < MIN_FREE_BYTES => {
                if let Some(active) = self.active.take() {
                    self.finish(active, RecorderStopReason::DiskSpaceLow { free_bytes });
                }
            }
            Ok(_) => self.send(progress),
            Err(e) => {
                if let Some(active) = self.active.take() {
                    self.fail(active, e.into());
                }
            }
        }
    }

    /// After each recording ends: DebugCaptures/ measured, reported as Stored.
    fn report_stored(&self, root: &Path) {
        match stored_recordings(root) {
            Ok(stored) => self.send(RecorderMsg::Stored { stored }),
            Err(e) => tracing::error!(
                kind = ?e.kind(),
                code = ?e.raw_os_error(),
                "could not measure the debug captures folder"
            ),
        }
    }

    /// try_send: the controller drains its channel continuously and progress is 1 Hz, so a full
    /// channel means a stalled controller; Disconnected means it is gone.
    fn send(&self, msg: RecorderMsg) {
        match self.tx.try_send(msg) {
            Ok(()) | Err(TrySendError::Disconnected(_)) => {}
            Err(TrySendError::Full(msg)) => tracing::warn!(
                message = msg_name(&msg),
                "the recorder channel was full; a message was dropped"
            ),
        }
    }
}

impl Active {
    /// Writes every packet in the ring. On a `discontinuity` the frames lost since the last block
    /// are marked as a gap first (§2.13).
    fn drain(&mut self, disk: &dyn RecorderDisk) -> Result<Drained, RecordingError> {
        while let Some(meta) = self.ring.pop(&mut self.samples) {
            let bytes = self.samples.len() as u64 * SAMPLE_BYTES;
            if self.writer.bytes() + bytes > RECORDING_SIZE_LIMIT {
                return Ok(Drained::LimitReached);
            }
            let opens_gap = match self.next {
                None => meta.start.0 > 0,
                Some(expected) => {
                    // A block before the recording's end is refused by write_block below.
                    if meta.discontinuity && meta.start >= expected {
                        self.writer.mark_gap(meta.start, meta.start.0 - expected.0);
                    }
                    meta.start != expected || meta.discontinuity
                }
            };
            disk.before_write();
            self.writer.write_block(&AudioBlock {
                format: self.format,
                start: meta.start,
                samples: &self.samples,
                time: meta.time,
                discontinuity: meta.discontinuity,
            })?;
            if opens_gap {
                self.gaps += 1;
            }
            let frames = u64::from(meta.frames);
            self.next = Some(SampleIndex(meta.start.0 + frames));
            self.frames += frames;
            disk.before_write();
            self.writer.checkpoint()?;
        }
        Ok(Drained::Empty)
    }

    fn progress(&self) -> RecorderMsg {
        let ms = self.format.rate.duration_of(self.frames);
        // INVARIANT: a frame count over a valid sample rate is finite.
        let duration = Seconds::try_new(ms.get() / 1000.0).expect("a finite duration");
        RecorderMsg::Progress {
            bytes: self.writer.bytes(),
            duration,
            gaps: self.gaps,
            // SPEC: feed events are dropped by processing's try_send, which the recorder never
            // sees; processing counts them (ProcessingDrops.recorder_events) for the controller.
            // The recorder itself drops none.
            dropped_events: 0,
        }
    }
}

/// The variant name, for logs.
fn msg_name(msg: &RecorderMsg) -> &'static str {
    match msg {
        RecorderMsg::Started { .. } => "Started",
        RecorderMsg::Progress { .. } => "Progress",
        RecorderMsg::Stopped { .. } => "Stopped",
        RecorderMsg::Failed { .. } => "Failed",
        RecorderMsg::Stored { .. } => "Stored",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::messages::PacketMeta;
    use crate::engine::ring::{audio_ring, PushOutcome, RingWriter, RECORDER_RING};
    use chrono::{DateTime, Utc};
    use qwerty_core::audio::{BlockTime, CaptureMode, DeviceIdentity};
    use qwerty_core::profile::ZoneId;
    use qwerty_core::recording::{Gap, RecordingReader, SegmentKind};
    use qwerty_core::units::{ChannelCount, HostTime, SampleRate};

    /// Frames per packet: 10 ms at 48 kHz.
    const FRAMES: u32 = 480;
    const WAIT: Duration = Duration::from_secs(5);

    /// The test-local disk (§2.13): unlimited free space, and writes never stall.
    struct TestDisk {
        free: u64,
    }

    impl RecorderDisk for TestDisk {
        fn free_bytes(&self, _dir: &Path) -> std::io::Result<u64> {
            Ok(self.free)
        }

        fn before_write(&self) {}
    }

    /// A fresh directory under the temp dir, removed when dropped.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> TempDir {
            let dir =
                std::env::temp_dir().join(format!("qwerty-c4-rec-{name}-{}", std::process::id()));
            if dir.exists() {
                std::fs::remove_dir_all(&dir).expect("clear the old test directory");
            }
            std::fs::create_dir_all(&dir).expect("make the test directory");
            TempDir(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            // Cleanup only: a leftover temp folder fails nothing.
            if let Err(e) = std::fs::remove_dir_all(&self.0) {
                eprintln!("could not remove {}: {e}", self.0.display());
            }
        }
    }

    /// The recorder thread, driven from the test as the controller and processing would.
    struct Harness {
        cmds: Sender<RecorderCmd>,
        events: Sender<SidecarEvent>,
        msgs: Receiver<RecorderMsg>,
        worker: Worker,
        _exits: Receiver<ThreadExit>,
    }

    impl Harness {
        fn new(free: u64) -> Harness {
            let (cmds, cmds_rx) = crossbeam_channel::bounded(64);
            let (events, events_rx) = crossbeam_channel::bounded(1024);
            let (msgs_tx, msgs) = crossbeam_channel::bounded(16);
            let (exits, exits_rx) = crossbeam_channel::bounded(16);
            let worker = spawn(
                WorkerId(1),
                cmds_rx,
                events_rx,
                msgs_tx,
                Arc::new(TestDisk { free }),
                exits,
            )
            .expect("spawns the recorder");
            Harness {
                cmds,
                events,
                msgs,
                worker,
                _exits: exits_rx,
            }
        }

        fn command(&self, cmd: RecorderCmd) {
            self.cmds.send(cmd).expect("the recorder is running");
        }

        fn event(&self, event: SidecarEvent) {
            self.events.try_send(event).expect("the feed has room");
        }

        /// The next message that is not Progress.
        fn next(&self) -> RecorderMsg {
            loop {
                match self.msgs.recv_timeout(WAIT).expect("a recorder message") {
                    RecorderMsg::Progress { .. } => {}
                    msg => return msg,
                }
            }
        }

        fn start(&self, root: &Path, ring: crate::engine::ring::RingReader) -> PathBuf {
            self.command(RecorderCmd::Start {
                root: root.to_path_buf(),
                header: Box::new(header()),
                ring,
            });
            match self.next() {
                RecorderMsg::Started { dir } => PathBuf::from(dir),
                other => panic!("expected Started, got {other:?}"),
            }
        }

        /// Stop{CaptureStopped}; returns the summary, after the Stored that follows it.
        fn stop(&self) -> RecordingSummary {
            self.command(RecorderCmd::Stop {
                reason: RecorderStopReason::CaptureStopped,
            });
            let summary = match self.next() {
                RecorderMsg::Stopped { summary, reason } => {
                    assert_eq!(reason, RecorderStopReason::CaptureStopped);
                    summary
                }
                other => panic!("expected Stopped, got {other:?}"),
            };
            match self.next() {
                RecorderMsg::Stored { stored } => assert_eq!(stored.count, 1),
                other => panic!("expected Stored, got {other:?}"),
            }
            summary
        }

        fn shut_down(self) {
            self.worker.stop(WAIT).expect("the recorder stops");
        }
    }

    fn format() -> AudioFormat {
        AudioFormat {
            rate: SampleRate::new(48_000).expect("48 kHz"),
            channels: ChannelCount::new(1).expect("mono"),
        }
    }

    fn header() -> SidecarHeader {
        SidecarHeader {
            app_version: "c4-test".into(),
            device: DeviceIdentity {
                endpoint_id: "{0.0.1.00000000}.{c4-microphone}".into(),
                name: "USB Microphone".into(),
                format: format(),
                capture_mode: CaptureMode::Raw,
            },
            layout: None,
            profile_id: None,
            created_at: DateTime::parse_from_rfc3339("2026-11-11T09:00:00Z")
                .expect("a valid time")
                .with_timezone(&Utc),
        }
    }

    fn zone(n: u32) -> ZoneId {
        serde_json::from_str(&format!("\"00000000-0000-0000-0000-{n:012x}\"")).expect("a uuid")
    }

    /// Processing's push of the block at stream position `start`.
    fn push(ring: &mut RingWriter, start: u64, discontinuity: bool) -> PushOutcome {
        let samples: Vec<f32> = (0..u64::from(FRAMES))
            .map(|i| ((start + i) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let meta = PacketMeta {
            start: SampleIndex(start),
            frames: FRAMES,
            time: BlockTime::Timestamped {
                host_time: HostTime(1_000_000 + (start * 10_000_000 / 48_000) as i64),
            },
            discontinuity,
        };
        ring.push(meta, &samples)
    }

    fn segment(start: u64, kind: SegmentKind) -> SidecarEvent {
        SidecarEvent::Segment {
            start: SampleIndex(start),
            kind,
            reconfigured: true,
        }
    }

    #[test]
    fn rc1_segments_tile_the_recording_in_configure_order() {
        let tmp = TempDir::new("rc1");
        let root = tmp.0.join("DebugCaptures");
        let h = Harness::new(u64::MAX);
        let (mut ring, reader) = audio_ring(format(), RECORDER_RING, 2048, FRAMES);
        let dir = h.start(&root, reader);

        // The controller's Configure order. Each setup's first processed block pushes its Segment
        // (and the calibration's CalibrationStarted just before it), then its blocks follow.
        let setups = [
            (SegmentKind::Listening, 3, false),
            (
                SegmentKind::CalibrationZone {
                    zone: zone(0xA1),
                    label: "Left".into(),
                },
                5,
                true,
            ),
            (SegmentKind::CalibrationNegatives, 2, false),
            (SegmentKind::Listening, 4, false),
        ];
        // The recording starts 2 s into the capture stream.
        let first = 96_000u64;
        let mut pos = first;
        let mut expected = Vec::new();
        for (kind, blocks, calibration) in setups {
            if calibration {
                h.event(SidecarEvent::CalibrationStarted {
                    start: SampleIndex(pos),
                    seed: 0xC4,
                    taps_per_zone: 10,
                });
            }
            h.event(segment(pos, kind.clone()));
            expected.push((pos, kind));
            for b in 0..blocks {
                if b == 1 {
                    h.event(SidecarEvent::Decision {
                        onset: SampleIndex(pos + 100),
                        accepted_zone: None,
                    });
                }
                assert_eq!(push(&mut ring, pos, false), PushOutcome::Written);
                pos += u64::from(FRAMES);
            }
        }
        h.command(RecorderCmd::Event {
            event: SidecarEvent::Prompt {
                zone: zone(0xA1),
                opened_at: HostTime(5_000_000),
                deadline: HostTime(20_000_000),
                reopened: false,
            },
        });
        let summary = h.stop();
        h.shut_down();
        assert_eq!(summary.frames, pos - first);
        assert_eq!(PathBuf::from(&summary.dir), dir);

        let mut reader = RecordingReader::open(&dir).expect("the recording reads back");
        let sidecar = reader.sidecar.clone();
        assert!(sidecar.finished);
        let segments: Vec<(u64, SegmentKind)> = sidecar
            .events
            .iter()
            .filter_map(|e| match e {
                SidecarEvent::Segment {
                    start,
                    kind,
                    reconfigured,
                } => {
                    assert!(reconfigured, "the engine only re-Configures");
                    Some((start.0, kind.clone()))
                }
                SidecarEvent::CalibrationStarted { .. }
                | SidecarEvent::Prompt { .. }
                | SidecarEvent::Input { .. }
                | SidecarEvent::SelfOutput { .. }
                | SidecarEvent::Decision { .. } => None,
            })
            .collect();
        // Each segment starts at the first sample processed under its Configure, in order.
        assert_eq!(segments, expected);
        // The first starts at the recording's first sample, and they tile it to its end.
        let first_block = reader
            .next_block(FRAMES as usize)
            .expect("reads")
            .expect("a first block");
        assert_eq!(first_block.start, SampleIndex(first));
        assert_eq!(segments[0].0, first);
        assert!(segments.windows(2).all(|w| w[0].0 < w[1].0));
        let end = first + sidecar.frames;
        assert_eq!(end, pos);
        assert!(segments.last().is_some_and(|s| s.0 < end));
        // CalibrationStarted comes just before its Segment, with the same start.
        let at = sidecar
            .events
            .iter()
            .position(|e| matches!(e, SidecarEvent::CalibrationStarted { .. }))
            .expect("the calibration's plan is recorded");
        assert_eq!(
            sidecar.events[at + 1],
            segment(expected[1].0, expected[1].1.clone())
        );
        assert!(sidecar
            .events
            .iter()
            .any(|e| matches!(e, SidecarEvent::Prompt { .. })));
    }

    #[test]
    fn rc2_full_feed_ring_gives_an_exact_gap() {
        let tmp = TempDir::new("rc2");
        let root = tmp.0.join("DebugCaptures");
        let h = Harness::new(u64::MAX);
        let (mut ring, reader) = audio_ring(format(), RECORDER_RING, 2048, FRAMES);

        // Processing fills the ring before the recorder drains it: once a push is Dropped, those
        // frames are lost and processing marks the next pushed block `discontinuity`.
        let mut pos = 0u64;
        let mut written = 0u64;
        while push(&mut ring, pos, false) == PushOutcome::Written {
            pos += u64::from(FRAMES);
            written += u64::from(FRAMES);
        }
        let mut lost = u64::from(FRAMES);
        pos += u64::from(FRAMES);
        for _ in 0..3 {
            assert_eq!(
                push(&mut ring, pos, false),
                PushOutcome::Dropped { frames: FRAMES }
            );
            lost += u64::from(FRAMES);
            pos += u64::from(FRAMES);
        }

        let dir = h.start(&root, reader);
        h.event(segment(0, SegmentKind::Listening));
        // Until the recorder drains, every push is dropped too; the first block that lands
        // carries the discontinuity.
        let gap_at = loop {
            if push(&mut ring, pos, true) == PushOutcome::Written {
                break pos;
            }
            lost += u64::from(FRAMES);
            pos += u64::from(FRAMES);
            std::thread::sleep(Duration::from_millis(2));
        };
        pos += u64::from(FRAMES);
        written += u64::from(FRAMES);
        // Stop drains what the ring still holds before the recording is finished.
        let summary = h.stop();
        h.shut_down();
        assert_eq!(summary.gaps, 1);
        assert_eq!(summary.frames, written);

        let mut reader = RecordingReader::open(&dir).expect("the recording reads back");
        assert_eq!(
            reader.sidecar.gaps,
            [Gap {
                at: SampleIndex(gap_at),
                frames_lost: lost
            }]
        );
        assert_eq!(written + lost, pos);
        // Read back, the block after the gap is at its live position, marked discontinuous.
        let mut after_gap = None;
        while let Some(block) = reader.next_block(FRAMES as usize).expect("reads") {
            if block.discontinuity {
                after_gap = Some(block.start);
            }
        }
        assert_eq!(after_gap, Some(SampleIndex(gap_at)));
    }

    #[test]
    fn a_start_below_the_free_space_floor_is_refused() {
        let tmp = TempDir::new("space");
        let root = tmp.0.join("DebugCaptures");
        let free = MIN_FREE_BYTES - 1;
        let h = Harness::new(free);
        let (_ring, reader) = audio_ring(format(), RECORDER_RING, 2048, FRAMES);
        h.command(RecorderCmd::Start {
            root: root.clone(),
            header: Box::new(header()),
            ring: reader,
        });
        match h.next() {
            RecorderMsg::Stopped { summary, reason } => {
                assert_eq!(
                    reason,
                    RecorderStopReason::DiskSpaceLow { free_bytes: free }
                );
                assert_eq!(summary.frames, 0);
            }
            other => panic!("expected Stopped, got {other:?}"),
        }
        // A Stop after the refusal has nothing to stop.
        h.command(RecorderCmd::Stop {
            reason: RecorderStopReason::Disabled,
        });
        h.shut_down();
        assert!(no_recording_under(&root));
    }

    /// No recording directory was made under `root`.
    fn no_recording_under(root: &Path) -> bool {
        stored_recordings(root).expect("measures") == StoredRecordings { count: 0, bytes: 0 }
    }

    #[test]
    fn stored_recordings_counts_folders_and_their_bytes() {
        let tmp = TempDir::new("stored");
        let root = tmp.0.join("DebugCaptures");
        assert_eq!(
            stored_recordings(&root).expect("a missing folder is empty"),
            StoredRecordings { count: 0, bytes: 0 }
        );
        for (name, bytes) in [("20261111T090000Z", 100usize), ("20261111T090500Z", 23)] {
            let dir = root.join(name);
            std::fs::create_dir_all(&dir).expect("makes a recording folder");
            std::fs::write(dir.join("audio.wav"), vec![0u8; bytes]).expect("writes");
            std::fs::write(dir.join("session.json"), b"{}").expect("writes");
        }
        std::fs::write(root.join("note.txt"), b"not a recording").expect("writes");
        assert_eq!(
            stored_recordings(&root).expect("measures"),
            StoredRecordings {
                count: 2,
                bytes: 127
            }
        );
        delete_recordings(&root).expect("deletes");
        assert_eq!(
            stored_recordings(&root).expect("measures"),
            StoredRecordings { count: 0, bytes: 0 }
        );
        assert!(
            root.join("note.txt").is_file(),
            "only folders are recordings"
        );
    }
}
