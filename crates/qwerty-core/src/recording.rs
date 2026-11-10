//! Opt-in raw recorder for replay corpora (§1.18).
//!
//! A recording is a directory `<root>/<YYYYMMDDTHHMMSSZ>/` holding `audio.wav` (32-bit float, the
//! capture format) and `session.json` (the [`Sidecar`]). Positions are stream positions
//! ([`SampleIndex`] as processing saw them): the WAV holds only the frames that reached the
//! recorder, and the sidecar's [`Gap`]s say where frames are missing, so the reader rebuilds the
//! live positions exactly.

use crate::audio::{AudioBlock, AudioFormat, BlockTime, ChannelLayout, ClockMap, DeviceIdentity};
use crate::profile::{parse_versioned, write_atomic, PersistenceError, ProfileId, ZoneId};
use crate::units::{HostTime, IoFailure, Milliseconds, SampleIndex};
use crate::veto::{InputEvent, SelfOutputEvent};
use crate::wav::{WavError, WavReader, WavSpec, WavWriter};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

pub const RECORDING_SCHEMA_VERSION: u32 = 1;
pub const AUDIO_FILE: &str = "audio.wav";
pub const SIDECAR_FILE: &str = "session.json";
pub const ANCHOR_INTERVAL: Milliseconds = Milliseconds::new(1000.0);
pub const HEADER_CHECKPOINT: Milliseconds = Milliseconds::new(1000.0);
pub const SIDECAR_CHECKPOINT: Milliseconds = Milliseconds::new(10_000.0);

/// The recording directory's name: its creation time in UTC.
const DIR_NAME_FORMAT: &str = "%Y%m%dT%H%M%SZ";
/// WAV channel mask of a recording: 0, "no speaker positions", since the channels are microphones.
// SPEC: §1.18 does not give the recorder's channel mask; 0 is the WAVEFORMATEXTENSIBLE value for
// channels that are not mapped to speaker positions.
const RECORDING_CHANNEL_MASK: u32 = 0;
/// `detector::FRONT_END_VERSION` (§1.6.1), recorded so replay knows which front end ran live.
// SPEC: §1.18.2 does not say where `front_end_version` comes from; it is the detector's
// FRONT_END_VERSION, which WP-A1 creates in parallel with this package. Until both have landed this
// mirrors its value (1); once `crate::detector::FRONT_END_VERSION` exists it replaces this constant.
const FRONT_END_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum NegativeClass {
    Typing,
    Clicking,
    Speech,
    Music,
    Knocks,
    ObjectsSetDown,
    /// Mixed typing, clicking and talking: the evaluation's background check (PARITY-9).
    Everyday,
    Other {
        label: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SegmentKind {
    Idle,
    Listening,
    Ambient,
    CalibrationZone {
        zone: ZoneId,
        label: String,
    },
    CalibrationNegatives,
    EvaluationPrompts,
    EvaluationNegative {
        class: NegativeClass,
    },
    /// CLI corpus-building (§2.13).
    Corpus {
        class: NegativeClass,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SidecarEvent {
    /// Contiguous; each ends where the next starts. Pushed by the engine's processing thread
    /// (p2 §2.5, §2.13). With `reconfigured`, `start` is the first sample processed under the new
    /// setup, so every detector/pipeline rebuild is a boundary; without it (a calibration step
    /// change and no re-Configure), the last onset the session consumed + 1 (RUST-RT-13, X01-06).
    /// The engine re-Configures at every segment change, so it only writes `reconfigured: true`.
    Segment {
        start: SampleIndex,
        kind: SegmentKind,
        reconfigured: bool,
    },
    /// A calibration run's plan, so replay rebuilds the same augmentation. A later run replaces an
    /// earlier one in the same recording. Pushed by the processing thread just before that run's
    /// first Segment, with the same `start`.
    CalibrationStarted {
        start: SampleIndex,
        seed: u64,
        taps_per_zone: u32,
    },
    /// `reopened`: opened by undo_last (XR-UI-05).
    Prompt {
        zone: ZoneId,
        opened_at: HostTime,
        deadline: HostTime,
        reopened: bool,
    },
    Input {
        event: InputEvent,
    },
    SelfOutput {
        event: SelfOutputEvent,
    },
    /// Live outcome, for live-vs-replay equality.
    Decision {
        onset: SampleIndex,
        accepted_zone: Option<ZoneId>,
    },
}

/// The host time of a stream position, recorded every [`ANCHOR_INTERVAL`] and after every gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Anchor {
    pub sample: SampleIndex,
    pub host_time: HostTime,
}

/// `frames_lost` frames are missing from the recording immediately before stream position `at`;
/// the block at `at` is a discontinuity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gap {
    pub at: SampleIndex,
    pub frames_lost: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sidecar {
    pub schema_version: u32,
    pub created_at: DateTime<Utc>,
    pub app_version: String,
    pub core_version: String,
    pub front_end_version: u32,
    pub device: DeviceIdentity,
    /// From the active profile, if any.
    pub layout: Option<ChannelLayout>,
    pub profile_id: Option<ProfileId>,
    pub anchors: Vec<Anchor>,
    pub gaps: Vec<Gap>,
    pub events: Vec<SidecarEvent>,
    pub frames: u64,
    /// false ⇒ the writer did not finish (crash); still readable.
    pub finished: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SidecarHeader {
    pub app_version: String,
    pub device: DeviceIdentity,
    pub layout: Option<ChannelLayout>,
    pub profile_id: Option<ProfileId>,
    pub created_at: DateTime<Utc>,
}

/// Writes one recording; the engine's recorder thread owns it.
pub struct RecordingWriter {
    wav: WavWriter,
    sidecar: Sidecar,
    dir: PathBuf,
    format: AudioFormat,
    /// Stream position the next block continues at; None before the first block.
    next: Option<SampleIndex>,
    /// `sidecar.gaps[..reconciled]` have been matched to the blocks that follow them.
    reconciled: usize,
    /// The next timestamped block is anchored (first block, and after every gap).
    anchor_due: bool,
    last_anchor: Option<SampleIndex>,
    anchor_frames: u64,
    header_frames: u64,
    sidecar_frames: u64,
    /// WAV frames at the last header patch / sidecar write.
    header_at: u64,
    sidecar_at: u64,
}

impl RecordingWriter {
    /// Makes `<root>/<created_at as YYYYMMDDTHHMMSSZ>/` (creating `root` if needed) and writes the
    /// empty WAV and the initial sidecar. Fails with `DirExists` if the directory exists.
    pub fn create(root: &Path, header: SidecarHeader) -> Result<Self, RecordingError> {
        let dir = root.join(header.created_at.format(DIR_NAME_FORMAT).to_string());
        std::fs::create_dir_all(root)?;
        if let Err(e) = std::fs::create_dir(&dir) {
            return Err(if e.kind() == std::io::ErrorKind::AlreadyExists {
                RecordingError::DirExists {
                    dir: dir.display().to_string(),
                }
            } else {
                e.into()
            });
        }
        let format = header.device.format;
        let wav = WavWriter::create(
            &dir.join(AUDIO_FILE),
            WavSpec {
                format,
                channel_mask: RECORDING_CHANNEL_MASK,
            },
        )?;
        let sidecar = Sidecar {
            schema_version: RECORDING_SCHEMA_VERSION,
            created_at: header.created_at,
            app_version: header.app_version,
            core_version: crate::CORE_VERSION.to_string(),
            front_end_version: FRONT_END_VERSION,
            device: header.device,
            layout: header.layout,
            profile_id: header.profile_id,
            anchors: Vec::new(),
            gaps: Vec::new(),
            events: Vec::new(),
            frames: 0,
            finished: false,
        };
        let rate = format.rate;
        let mut writer = RecordingWriter {
            wav,
            sidecar,
            dir,
            format,
            next: None,
            reconciled: 0,
            anchor_due: true,
            last_anchor: None,
            anchor_frames: rate.samples_in(ANCHOR_INTERVAL) as u64,
            header_frames: rate.samples_in(HEADER_CHECKPOINT) as u64,
            sidecar_frames: rate.samples_in(SIDECAR_CHECKPOINT) as u64,
            header_at: 0,
            sidecar_at: 0,
        };
        writer.write_sidecar()?;
        Ok(writer)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Appends a block. Anchors the first timestamped block, every [`ANCHOR_INTERVAL`] and the
    /// first timestamped block after each gap.
    ///
    /// A block that jumps past the recording's end, or is flagged `discontinuity`, starts a gap:
    /// either the one `mark_gap` recorded for it, which must count exactly the missing frames, or
    /// one recorded here. The first block opens the recording; if it starts after sample 0 the
    /// frames before it are recorded as a gap, so positions stay the stream's.
    // SPEC: §1.18.2 leaves the recording's first stream position implicit and has no field for it;
    // the frames before a first block at s > 0 are recorded as `Gap{at: s, frames_lost: s}`, which
    // keeps every position equal to the live one (needed by replay's live agreement, RP2).
    // SPEC: a block that overlaps the recording, or a marked gap that does not match the block
    // after it, would make the sidecar describe the audio wrongly; it is refused with
    // `RecordingError::Sidecar`, the closest named error, before anything is written.
    pub fn write_block(&mut self, block: &AudioBlock<'_>) -> Result<(), RecordingError> {
        if block.format != self.format {
            return Err(RecordingError::FormatMismatch);
        }
        let new_gap = self.reconcile(block)?;
        self.wav.write_interleaved(block.samples)?;
        if let Some(gap) = new_gap {
            self.sidecar.gaps.push(gap);
            self.anchor_due = true;
        }
        self.reconciled = self.sidecar.gaps.len();
        if let BlockTime::Timestamped { host_time } = block.time {
            let interval_elapsed = self
                .last_anchor
                .is_some_and(|a| block.start.0 - a.0 >= self.anchor_frames);
            if self.anchor_due || interval_elapsed {
                self.sidecar.anchors.push(Anchor {
                    sample: block.start,
                    host_time,
                });
                self.last_anchor = Some(block.start);
                self.anchor_due = false;
            }
        }
        self.next = Some(SampleIndex(block.start.0 + block.frames() as u64));
        Ok(())
    }

    /// Checks `block` against the recording's end and the gaps marked since the last block;
    /// returns the gap to record for it, if `mark_gap` did not.
    fn reconcile(&self, block: &AudioBlock<'_>) -> Result<Option<Gap>, RecordingError> {
        let pending = &self.sidecar.gaps[self.reconciled..];
        let (expected, opens_gap) = match self.next {
            None => (SampleIndex(0), block.start.0 > 0),
            Some(next) => (next, block.start != next || block.discontinuity),
        };
        if block.start < expected {
            return Err(sidecar_error(format!(
                "a block at sample {} overlaps the recording, which continues at {}",
                block.start.0, expected.0
            )));
        }
        let lost = block.start.0 - expected.0;
        match (opens_gap, pending) {
            (false, []) => Ok(None),
            (true, []) => Ok(Some(Gap {
                at: block.start,
                frames_lost: lost,
            })),
            (true, [gap]) if gap.at == block.start && gap.frames_lost == lost => Ok(None),
            (_, marked) => Err(sidecar_error(format!(
                "gaps {marked:?} were marked, but the block at sample {} follows the recording's \
                 end at {} with {lost} frames missing",
                block.start.0, expected.0
            ))),
        }
    }

    /// Records that `frames_lost` frames before stream position `at` never reached the recorder.
    /// The engine calls it before writing the block at `at`, which `write_block` then checks.
    pub fn mark_gap(&mut self, at: SampleIndex, frames_lost: u64) {
        self.sidecar.gaps.push(Gap { at, frames_lost });
        self.anchor_due = true;
    }

    pub fn push_event(&mut self, e: SidecarEvent) {
        self.sidecar.events.push(e);
    }

    /// Patches the WAV header once per [`HEADER_CHECKPOINT`] of audio and rewrites the sidecar
    /// atomically once per [`SIDECAR_CHECKPOINT`], both timed by frames written. The header is
    /// patched before every sidecar write, so the WAV never holds fewer frames than the sidecar
    /// counts.
    pub fn checkpoint(&mut self) -> Result<(), RecordingError> {
        let frames = self.wav.frames();
        if frames - self.sidecar_at >= self.sidecar_frames {
            self.wav.flush_header()?;
            self.header_at = frames;
            self.write_sidecar()?;
            self.sidecar_at = frames;
        } else if frames - self.header_at >= self.header_frames {
            self.wav.flush_header()?;
            self.header_at = frames;
        }
        Ok(())
    }

    /// Bytes of audio written: the WAV header plus its samples.
    pub fn bytes(&self) -> u64 {
        self.wav.bytes()
    }

    /// Finalizes the WAV and writes the finished sidecar.
    pub fn finish(mut self) -> Result<RecordingSummary, RecordingError> {
        let bytes = self.wav.bytes();
        let events = count(self.sidecar.events.len(), "events")?;
        let gaps = count(self.sidecar.gaps.len(), "gaps")?;
        let frames = self.wav.finalize()?;
        self.sidecar.finished = true;
        self.sidecar.frames = frames;
        write_sidecar_file(&self.dir, &self.sidecar)?;
        Ok(RecordingSummary {
            dir: self.dir.display().to_string(),
            frames,
            bytes,
            events,
            gaps,
        })
    }

    fn write_sidecar(&mut self) -> Result<(), RecordingError> {
        self.sidecar.frames = self.wav.frames();
        write_sidecar_file(&self.dir, &self.sidecar)
    }
}

fn write_sidecar_file(dir: &Path, sidecar: &Sidecar) -> Result<(), RecordingError> {
    let json = serde_json::to_string_pretty(sidecar).map_err(|e| sidecar_error(e.to_string()))?;
    write_atomic(&dir.join(SIDECAR_FILE), &json)?;
    Ok(())
}

fn count(n: usize, what: &str) -> Result<u32, RecordingError> {
    u32::try_from(n).map_err(|_| sidecar_error(format!("{n} {what} do not fit the summary")))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordingSummary {
    pub dir: String,
    pub frames: u64,
    pub bytes: u64,
    pub events: u32,
    pub gaps: u32,
}

/// Reads a recording back as the blocks processing saw: live stream positions, host times from
/// the anchors, and a discontinuity at every gap.
pub struct RecordingReader {
    pub sidecar: Sidecar,
    wav: WavReader,
    /// WAV frames read so far.
    read: u64,
    /// Stream position of the next WAV frame.
    pos: SampleIndex,
    /// The first gap not yet passed.
    next_gap: usize,
    /// The first anchor not yet given to `clock`.
    next_anchor: usize,
    /// Fed the anchors of the current gap-free run, the same sample→host mapping live used.
    clock: ClockMap,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecordedBlock {
    pub start: SampleIndex,
    pub samples: Vec<f32>,
    pub time: BlockTime,
    pub discontinuity: bool,
}

impl RecordingReader {
    /// Opens `dir`: the sidecar (schema checked first), then the WAV, whose format must be the
    /// recorded device's. An unfinished recording reads up to its last WAV header patch.
    pub fn open(dir: &Path) -> Result<Self, RecordingError> {
        let text = std::fs::read_to_string(dir.join(SIDECAR_FILE))?;
        let sidecar: Sidecar = parse_versioned(&text, RECORDING_SCHEMA_VERSION, SIDECAR_FILE)
            .map_err(|e| match e {
                PersistenceError::SchemaMismatch {
                    found, expected, ..
                } => RecordingError::SchemaMismatch { found, expected },
                PersistenceError::Json(e) => sidecar_error(e.to_string()),
                PersistenceError::Io(e) => e.into(),
                PersistenceError::InvalidData { reason, .. } => sidecar_error(reason),
            })?;
        let wav = WavReader::open(&dir.join(AUDIO_FILE))?;
        if wav.spec().format != sidecar.device.format {
            return Err(RecordingError::FormatMismatch);
        }
        check_consistent(&sidecar, wav.frames())?;
        Ok(RecordingReader {
            clock: ClockMap::new(sidecar.device.format.rate),
            sidecar,
            wav,
            read: 0,
            pos: SampleIndex(0),
            next_gap: 0,
            next_anchor: 0,
        })
    }

    pub fn format(&self) -> AudioFormat {
        self.sidecar.device.format
    }

    /// Blocks of exactly `frames` (last may be shorter); BlockTime from anchors; discontinuity at
    /// gaps.
    ///
    /// # Panics
    /// If `frames` is 0, like `slice::chunks(0)`: no block could ever be returned.
    // SPEC: a block never spans a gap, so the block that reaches a gap is also cut short there; the
    // next one starts at the gap with `discontinuity`. A block's time is the anchors' ClockMap
    // mapping (§1.4) of its first frame, `Untimed` where no anchor of the current run applies.
    pub fn next_block(&mut self, frames: usize) -> Result<Option<RecordedBlock>, RecordingError> {
        assert!(frames > 0, "RecordingReader::next_block needs frames > 0");
        let mut discontinuity = false;
        while let Some(gap) = self.sidecar.gaps.get(self.next_gap) {
            if gap.at.0 - gap.frames_lost != self.pos.0 {
                break;
            }
            self.pos = gap.at;
            self.next_gap += 1;
            discontinuity = true;
            self.clock.reset();
            while self
                .sidecar
                .anchors
                .get(self.next_anchor)
                .is_some_and(|a| a.sample < gap.at)
            {
                self.next_anchor += 1;
            }
        }
        let until_gap = self
            .sidecar
            .gaps
            .get(self.next_gap)
            .map_or(u64::MAX, |g| g.at.0 - g.frames_lost - self.pos.0);
        let n = (frames as u64)
            .min(self.wav.frames() - self.read)
            .min(until_gap);
        if n == 0 {
            return Ok(None);
        }
        let start = self.pos;
        while let Some(anchor) = self.sidecar.anchors.get(self.next_anchor) {
            if anchor.sample > start {
                break;
            }
            self.clock.observe(&AudioBlock {
                format: self.format(),
                start: anchor.sample,
                samples: &[],
                time: BlockTime::Timestamped {
                    host_time: anchor.host_time,
                },
                discontinuity: false,
            });
            self.next_anchor += 1;
        }
        let time = match self.clock.host_time_of(start) {
            Some(host_time) => BlockTime::Timestamped { host_time },
            None => BlockTime::Untimed,
        };
        let channels = usize::from(self.format().channels.get());
        // INVARIANT: n ≤ frames, which is a usize.
        let n_frames = usize::try_from(n).expect("n ≤ frames");
        let mut samples = vec![0f32; n_frames * channels];
        let got = self.wav.read_frames(&mut samples)?;
        if got != n_frames {
            return Err(RecordingError::Wav {
                error: WavError::SizeMismatch {
                    header: n,
                    actual: got as u64,
                },
            });
        }
        self.read += n;
        self.pos = SampleIndex(start.0 + n);
        Ok(Some(RecordedBlock {
            start,
            samples,
            time,
            discontinuity,
        }))
    }
}

/// Anchors ascend; gaps ascend without overlapping; every gap lies within the audio; a finished
/// recording's frame count is the WAV's.
fn check_consistent(sidecar: &Sidecar, wav_frames: u64) -> Result<(), RecordingError> {
    if sidecar
        .anchors
        .windows(2)
        .any(|w| w[1].sample <= w[0].sample)
    {
        return Err(sidecar_error(
            "anchors are not in ascending sample order".into(),
        ));
    }
    let mut prev_at = 0u64;
    let mut lost_before = 0u64;
    for gap in &sidecar.gaps {
        let Some(gap_start) = gap.at.0.checked_sub(gap.frames_lost) else {
            return Err(sidecar_error(format!("{gap:?} starts before sample 0")));
        };
        if gap_start < prev_at {
            return Err(sidecar_error(format!("{gap:?} overlaps the gap before it")));
        }
        // INVARIANT: gap_start ≥ prev_at = the previous gap's at ≥ every frame lost before it.
        let wav_frame = gap_start - lost_before;
        if wav_frame > wav_frames {
            return Err(sidecar_error(format!(
                "{gap:?} lies past the {wav_frames} recorded frames"
            )));
        }
        prev_at = gap.at.0;
        lost_before += gap.frames_lost;
    }
    if sidecar.finished && sidecar.frames != wav_frames {
        return Err(sidecar_error(format!(
            "the finished sidecar counts {} frames but the WAV holds {wav_frames}",
            sidecar.frames
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RecordingError {
    Io { error: IoFailure },
    Wav { error: WavError },
    Sidecar { detail: String },
    SchemaMismatch { found: u32, expected: u32 },
    DirExists { dir: String },
    FormatMismatch,
}

fn sidecar_error(detail: String) -> RecordingError {
    RecordingError::Sidecar { detail }
}

impl From<std::io::Error> for RecordingError {
    fn from(e: std::io::Error) -> Self {
        RecordingError::Io {
            error: IoFailure::from(&e),
        }
    }
}

impl From<WavError> for RecordingError {
    fn from(error: WavError) -> Self {
        RecordingError::Wav { error }
    }
}

impl fmt::Display for RecordingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RecordingError::Io { error } => {
                write!(f, "recording I/O error ({}): {}", error.kind, error.message)
            }
            RecordingError::Wav { error } => write!(f, "recording audio: {error}"),
            RecordingError::Sidecar { detail } => write!(f, "recording sidecar: {detail}"),
            RecordingError::SchemaMismatch { found, expected } => write!(
                f,
                "recording sidecar schema {found} is not the supported {expected}"
            ),
            RecordingError::DirExists { dir } => write!(f, "recording directory {dir} exists"),
            RecordingError::FormatMismatch => {
                write!(
                    f,
                    "audio format does not match the recording's device format"
                )
            }
        }
    }
}

impl std::error::Error for RecordingError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::CaptureMode;
    use crate::rng::Rng64;
    use crate::units::{ChannelCount, SampleRate};
    use crate::veto::InputKind;
    use crate::wav::tests::TestDir;
    use chrono::TimeZone;
    use uuid::Uuid;

    const RATE: u32 = 48_000;
    const BLOCK: usize = 480;
    /// 100-ns ticks per 480-frame block at 48 kHz.
    const BLOCK_TICKS: i64 = 100_000;
    const T0: i64 = 7_000_000_000;

    fn format() -> AudioFormat {
        AudioFormat {
            rate: SampleRate::new(RATE).expect("rate"),
            channels: ChannelCount::new(2).expect("channels"),
        }
    }

    fn header(second: u32) -> SidecarHeader {
        SidecarHeader {
            app_version: "0.0.0-test".into(),
            device: DeviceIdentity {
                endpoint_id: "{test-endpoint}".into(),
                name: "Test microphone".into(),
                format: format(),
                capture_mode: CaptureMode::Raw,
            },
            layout: Some(ChannelLayout::all_real(format().channels)),
            profile_id: Some(ProfileId::from_uuid(Uuid::from_u128(1))),
            created_at: Utc
                .with_ymd_and_hms(2026, 11, 10, 12, 0, second)
                .single()
                .expect("valid time"),
        }
    }

    fn block_samples(rng: &mut Rng64) -> Vec<f32> {
        (0..BLOCK * 2)
            .map(|_| rng.range(-1.0, 1.0) as f32)
            .collect()
    }

    fn timed(start: u64) -> BlockTime {
        BlockTime::Timestamped {
            host_time: HostTime(T0 + start as i64 / BLOCK as i64 * BLOCK_TICKS),
        }
    }

    fn write(
        w: &mut RecordingWriter,
        start: u64,
        samples: &[f32],
        time: BlockTime,
        discontinuity: bool,
    ) {
        let block = AudioBlock::new(format(), SampleIndex(start), samples, time, discontinuity)
            .expect("block");
        w.write_block(&block).expect("write block");
        w.checkpoint().expect("checkpoint");
    }

    /// Reads every block of `frames`; returns them all.
    fn read_all(dir: &Path, frames: usize) -> Vec<RecordedBlock> {
        let mut r = RecordingReader::open(dir).expect("open");
        let mut blocks = Vec::new();
        while let Some(b) = r.next_block(frames).expect("next block") {
            blocks.push(b);
        }
        blocks
    }

    #[test]
    fn r1_writer_reader_round_trip() {
        let root = TestDir::new("r1");
        let zone = ZoneId::from_uuid(Uuid::from_u128(10));
        let mut rng = Rng64::new(1);
        let mut w = RecordingWriter::create(root.path(), header(0)).expect("create");
        assert_eq!(
            w.dir(),
            root.path().join("20261110T120000Z").as_path(),
            "directory named by creation time"
        );
        w.push_event(SidecarEvent::Segment {
            start: SampleIndex(0),
            kind: SegmentKind::CalibrationZone {
                zone,
                label: "Left".into(),
            },
            reconfigured: true,
        });
        let mut written: Vec<(u64, Vec<f32>)> = Vec::new();
        // Run 1: 2.5 s from sample 0, timestamped.
        for k in 0..250u64 {
            let s = block_samples(&mut rng);
            write(&mut w, k * BLOCK as u64, &s, timed(k * BLOCK as u64), false);
            written.push((k * BLOCK as u64, s));
        }
        w.push_event(SidecarEvent::Input {
            event: InputEvent {
                time: HostTime(T0 + 5_000_000),
                kind: InputKind::KeyDown,
            },
        });
        // Two blocks never reach the recorder; the engine marks the gap before the next block.
        let resume = 252 * BLOCK as u64;
        w.mark_gap(SampleIndex(resume), 2 * BLOCK as u64);
        // Run 2: the first block after the gap is untimed, so the anchor lands on the next one.
        for k in 0..150u64 {
            let start = resume + k * BLOCK as u64;
            let s = block_samples(&mut rng);
            let time = if k == 0 {
                BlockTime::Untimed
            } else {
                timed(start)
            };
            write(&mut w, start, &s, time, k == 0);
            written.push((start, s));
        }
        w.push_event(SidecarEvent::Decision {
            onset: SampleIndex(resume + 1000),
            accepted_zone: Some(zone),
        });
        let mut expected = w.sidecar.clone();
        let summary = w.finish().expect("finish");
        expected.frames = 400 * BLOCK as u64;
        expected.finished = true;

        assert_eq!(summary.frames, 400 * BLOCK as u64);
        assert_eq!((summary.events, summary.gaps), (3, 1));
        assert_eq!(summary.bytes, 80 + 400 * BLOCK as u64 * 8);
        let anchors: Vec<u64> = expected.anchors.iter().map(|a| a.sample.0).collect();
        assert_eq!(
            anchors,
            [0, 48_000, 96_000, resume + 480, resume + 480 + 48_000],
            "anchors every 1 s and at the first timestamped block after the gap"
        );
        assert_eq!(
            expected.gaps,
            [Gap {
                at: SampleIndex(resume),
                frames_lost: 960
            }]
        );

        let reader = RecordingReader::open(&root.path().join("20261110T120000Z")).expect("open");
        assert_eq!(reader.sidecar, expected, "identical sidecar");
        assert_eq!(reader.format(), format());

        // Blocks of the recorded size come back exactly as written, with their times.
        let dir = root.path().join("20261110T120000Z");
        let blocks = read_all(&dir, BLOCK);
        assert_eq!(blocks.len(), written.len());
        for (k, (b, (start, s))) in blocks.iter().zip(&written).enumerate() {
            assert_eq!(b.start, SampleIndex(*start), "block {k}");
            assert_eq!(b.samples, *s, "block {k}");
            assert_eq!(b.discontinuity, *start == resume, "block {k}");
            let want = if *start == resume {
                BlockTime::Untimed
            } else {
                timed(*start)
            };
            assert_eq!(b.time, want, "block {k}");
        }

        // An odd block size: no block spans the gap, and the samples concatenate identically.
        let blocks = read_all(&dir, 700);
        assert!(blocks.iter().all(|b| b.samples.len() <= 1400));
        let cut = blocks
            .iter()
            .find(|b| b.start.0 < resume && b.start.0 + (b.samples.len() / 2) as u64 > 120_000);
        assert!(cut.is_none(), "a block spans the gap");
        let after = blocks
            .iter()
            .position(|b| b.start == SampleIndex(resume))
            .expect("a block starts at the gap");
        assert!(blocks[after].discontinuity);
        assert_eq!(
            blocks[after - 1].start.0 + 300,
            120_000,
            "cut short at the gap"
        );
        assert_eq!(
            blocks.iter().filter(|b| b.discontinuity).count(),
            1,
            "only the gap is a discontinuity"
        );
        let got: Vec<f32> = blocks.iter().flat_map(|b| b.samples.clone()).collect();
        let want: Vec<f32> = written.iter().flat_map(|(_, s)| s.clone()).collect();
        assert_eq!(got, want);

        // A recording that opens mid-stream keeps the stream's positions.
        let mut w = RecordingWriter::create(root.path(), header(1)).expect("create");
        let s = block_samples(&mut rng);
        write(&mut w, 5000, &s, timed(4800), false);
        w.finish().expect("finish");
        let blocks = read_all(&root.path().join("20261110T120001Z"), BLOCK);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].start, SampleIndex(5000));
        assert!(blocks[0].discontinuity);
        assert_eq!(blocks[0].samples, s);
    }

    #[test]
    fn r2_schema_mismatch_and_unfinished_recording() {
        let root = TestDir::new("r2");
        let mut rng = Rng64::new(2);
        let mut w = RecordingWriter::create(root.path(), header(0)).expect("create");
        let dir = w.dir().to_path_buf();
        let mut written = Vec::new();
        // 1.5 s: the header is patched at 1 s; the sidecar is not rewritten before 10 s.
        for k in 0..150u64 {
            let s = block_samples(&mut rng);
            write(&mut w, k * BLOCK as u64, &s, timed(k * BLOCK as u64), false);
            written.extend(s);
        }
        assert_eq!(w.bytes(), 80 + 150 * BLOCK as u64 * 8);
        // A crash: the writer never finishes.
        drop(w);
        let mut r = RecordingReader::open(&dir).expect("unfinished recording opens");
        assert!(!r.sidecar.finished);
        assert_eq!(r.sidecar.frames, 0, "the sidecar from creation");
        let mut got = Vec::new();
        while let Some(b) = r.next_block(BLOCK).expect("next block") {
            got.extend(b.samples);
        }
        assert_eq!(
            got,
            written[..48_000 * 2],
            "frames up to the 1 s header patch"
        );

        // A future schema is refused before anything else is read.
        let path = dir.join(SIDECAR_FILE);
        let text = std::fs::read_to_string(&path).expect("read sidecar");
        let bumped = text.replacen("\"schema_version\": 1", "\"schema_version\": 2", 1);
        assert_ne!(bumped, text);
        std::fs::write(&path, bumped).expect("write sidecar");
        assert_eq!(
            RecordingReader::open(&dir).err(),
            Some(RecordingError::SchemaMismatch {
                found: 2,
                expected: 1
            })
        );

        // The same second again: the directory exists.
        assert_eq!(
            RecordingWriter::create(root.path(), header(0)).err(),
            Some(RecordingError::DirExists {
                dir: dir.display().to_string()
            })
        );
        // A block in another format, an overlapping block and a mismatched gap are refused.
        let mut w = RecordingWriter::create(root.path(), header(2)).expect("create");
        let mono = AudioFormat {
            rate: format().rate,
            channels: ChannelCount::new(1).expect("channels"),
        };
        let s = block_samples(&mut rng);
        let wrong = AudioBlock::new(mono, SampleIndex(0), &s, timed(0), false).expect("block");
        assert_eq!(w.write_block(&wrong), Err(RecordingError::FormatMismatch));
        write(&mut w, 0, &s, timed(0), false);
        let overlap = AudioBlock::new(format(), SampleIndex(100), &s, timed(0), false).expect("b");
        assert!(matches!(
            w.write_block(&overlap),
            Err(RecordingError::Sidecar { .. })
        ));
        w.mark_gap(SampleIndex(2000), 7);
        let after = AudioBlock::new(format(), SampleIndex(2000), &s, timed(0), true).expect("b");
        assert!(matches!(
            w.write_block(&after),
            Err(RecordingError::Sidecar { .. })
        ));
        assert_eq!(
            w.bytes(),
            80 + BLOCK as u64 * 8,
            "nothing written by a refused block"
        );
    }
}
