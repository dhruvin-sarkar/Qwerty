//! Soak v2 (§1.20): the frozen v1 fixture calibrated through `CalibrationSession` and streamed
//! through `Pipeline`, with no audio hardware.
//!
//! The `soak` example drives [`Soak::run_batch`] in a loop under the [`CountingAllocator`] defined
//! here and gates the run with [`gate`]. [`Capture`] and [`calibrate`] are the one synthetic
//! capture and calibration path; the desk regression suites (`tests/desk_regression.rs`) use them
//! too.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::audio::{AudioBlock, AudioError, AudioFormat, BlockTime, ChannelLayout};
use crate::desk::{CleanSignatures, DeskConfig, DeskError, Lcg, ModalDesk};
use crate::detector::{DetectorError, DetectorParams};
use crate::gesture::{GestureParams, Trigger};
use crate::pipeline::{
    self, CalibrationFrontEnd, DecisionOutcome, Pipeline, PipelineConfig, PipelineError,
    PipelineEvent, TapDecision,
};
use crate::profile::ZoneId;
use crate::rng::Rng64;
use crate::session::{
    self, CalibrationPlan, CalibrationResult, CalibrationSession, CalibrationStep, SessionError,
    StepKind, AMBIENT_DURATION, DEFAULT_AUDIO_BUDGET_BYTES, DEFAULT_TAPS_PER_ZONE,
};
use crate::training::TrainingParams;
use crate::units::{
    ChannelCount, HostTime, Milliseconds, SampleIndex, SampleRate, Seconds, ZoneIndex,
};
use crate::veto::{VetoParams, VetoTimeline};

// ---------------------------------------------------------------------------
// Live-heap accounting allocator (used by the soak example for the
// "no unbounded memory growth" evidence).
// ---------------------------------------------------------------------------

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Successful `alloc`/`realloc` calls made by the current thread. `const`-initialized and
    /// without a destructor, so reading or bumping it never allocates and never fails, even
    /// from inside the allocator or during thread teardown.
    static THREAD_ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

/// Counts one allocation on the current thread.
fn count_thread_allocation() {
    THREAD_ALLOCATIONS.with(|n| n.set(n.get() + 1));
}

/// A `GlobalAlloc` wrapper that tracks currently-live heap bytes. Install it in
/// a binary with `#[global_allocator]` to measure steady-state heap use across
/// a long run (see the `soak` example).
pub struct CountingAllocator;

// SAFETY: forwards every call to the system allocator unchanged, only adding
// atomic byte accounting and a per-thread call count around it.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
            count_thread_allocation();
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            // The live-byte total moves exactly as the default alloc + dealloc pair moved it.
            ALLOCATED.fetch_add(new_size, Ordering::Relaxed);
            ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
            count_thread_allocation();
        }
        new_ptr
    }
}

/// Currently-live heap bytes as tracked by [`CountingAllocator`].
pub fn allocated_bytes() -> usize {
    ALLOCATED.load(Ordering::Relaxed)
}

/// Successful `alloc` and `realloc` calls the current thread has made through
/// [`CountingAllocator`] (0 when it is not the global allocator). Tests take the
/// difference across a region to assert it allocates nothing.
pub fn thread_allocations() -> u64 {
    THREAD_ALLOCATIONS.with(Cell::get)
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a synthetic run could not continue. Every variant names the stage that failed.
#[derive(Debug, Clone, PartialEq)]
pub enum SoakError {
    Audio {
        error: AudioError,
    },
    Desk {
        error: DeskError,
    },
    Detector {
        error: DetectorError,
    },
    Session {
        error: SessionError,
    },
    Pipeline {
        error: PipelineError,
    },
    /// A zone was still short of taps after `presented` taps were played into it.
    ZoneNotFilled {
        zone: ZoneIndex,
        accepted: u32,
        presented: u32,
    },
    /// A decision whose onset lies before the batch that finalized it. Every batch ends with a
    /// floor gap far longer than the veto settle time, so this is a candidate in the previous
    /// batch's final 56 ms of floor: a finding to diagnose, not to drop.
    DecisionOutsideBatch {
        onset: SampleIndex,
    },
}

impl fmt::Display for SoakError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SoakError::Audio { error } => write!(f, "audio block: {error}"),
            SoakError::Desk { error } => write!(f, "desk fixture: {error}"),
            SoakError::Detector { error } => write!(f, "calibration front end: {error}"),
            SoakError::Session { error } => write!(f, "calibration session: {error:?}"),
            SoakError::Pipeline { error } => write!(f, "pipeline: {error:?}"),
            SoakError::ZoneNotFilled {
                zone,
                accepted,
                presented,
            } => write!(
                f,
                "zone {} accepted only {accepted} of {presented} calibration taps",
                zone.0
            ),
            SoakError::DecisionOutsideBatch { onset } => write!(
                f,
                "a decision at sample {} precedes the batch that finalized it",
                onset.0
            ),
        }
    }
}

impl std::error::Error for SoakError {}

impl From<AudioError> for SoakError {
    fn from(error: AudioError) -> Self {
        SoakError::Audio { error }
    }
}

impl From<DeskError> for SoakError {
    fn from(error: DeskError) -> Self {
        SoakError::Desk { error }
    }
}

impl From<DetectorError> for SoakError {
    fn from(error: DetectorError) -> Self {
        SoakError::Detector { error }
    }
}

impl From<SessionError> for SoakError {
    fn from(error: SessionError) -> Self {
        SoakError::Session { error }
    }
}

impl From<PipelineError> for SoakError {
    fn from(error: PipelineError) -> Self {
        SoakError::Pipeline { error }
    }
}

// ---------------------------------------------------------------------------
// Synthetic capture and calibration (shared with the desk regression suites)
// ---------------------------------------------------------------------------

/// The synthetic capture's block: the engine's 10 ms device packet (§1.20).
pub const BLOCK: Milliseconds = Milliseconds::new(10.0);

/// The synthetic host clock (§1.20): frame s of a stream at fs is host time s·10^7/fs, rounded.
pub fn host_time(rate: SampleRate, frame: SampleIndex) -> HostTime {
    let fs = i128::from(rate.hz());
    let ticks = (i128::from(frame.0) * i128::from(HostTime::TICKS_PER_SECOND) + fs / 2) / fs;
    // Fits i64 for every frame below 2^63/10^7·fs: at 16 kHz that is about 29 000 years of audio.
    HostTime(ticks as i64)
}

/// A synthetic capture device: one continuous stream of interleaved frames, cut into 10 ms
/// blocks that each carry the synthetic host time of their first frame.
#[derive(Debug, Clone)]
pub struct Capture {
    format: AudioFormat,
    block_frames: usize,
    next: u64,
}

impl Capture {
    pub fn new(format: AudioFormat) -> Capture {
        Capture {
            format,
            block_frames: format.rate.samples_in(BLOCK),
            next: 0,
        }
    }

    pub fn format(&self) -> AudioFormat {
        self.format
    }

    /// The frame the next streamed sample lands on.
    pub fn position(&self) -> SampleIndex {
        SampleIndex(self.next)
    }

    /// The host time of the newest streamed frame (frame 0's before any).
    pub fn now(&self) -> HostTime {
        host_time(self.format.rate, SampleIndex(self.next.saturating_sub(1)))
    }

    /// Streams interleaved `x` from `position()` in 10 ms blocks. `push` gets each block and
    /// `now` = the host time of the block's last frame (the replay clock, §1.15).
    pub fn stream<E: From<AudioError>>(
        &mut self,
        x: &[f32],
        mut push: impl FnMut(&AudioBlock<'_>, HostTime) -> Result<(), E>,
    ) -> Result<(), E> {
        let channels = usize::from(self.format.channels.get());
        for chunk in x.chunks(self.block_frames * channels) {
            let start = self.position();
            let time = BlockTime::Timestamped {
                host_time: host_time(self.format.rate, start),
            };
            let block = AudioBlock::new(self.format, start, chunk, time, false)?;
            self.next += block.frames() as u64;
            push(&block, self.now())?;
        }
        Ok(())
    }
}

/// Taps a zone may be played before calibration gives up on it, as a multiple of its target.
const MAX_PRESENTED_PER_TARGET: u32 = 3;

/// Calibrates through `CalibrationSession`, fed by `CalibrationFrontEnd` through `on_input`
/// exactly as the engine does (§1.20), on one stream and one veto timeline:
/// - `start_ambient` from Ready, then `ambient` (at least 3 s of audio) as the ambient step;
/// - the front end rebuilt for the accepted layout, as the engine does;
/// - for the current zone, `segment(position, zone)` is played until the zone is full: the
///   audio it returns starts at `position` and holds one tap at least 60 ms before its end, so
///   the tap's decision settles inside it;
/// - `advance(now)` whenever the session would auto-advance (the engine's timer), and
///   `finish_negatives` from the full last zone (negatives skipped: PooledAll);
/// - the build job run to completion, then `finish_build` and `commit`.
pub fn calibrate(
    plan: CalibrationPlan,
    capture: &mut Capture,
    veto: &mut VetoTimeline,
    ambient: &[f32],
    mut segment: impl FnMut(SampleIndex, ZoneIndex) -> Vec<f32>,
) -> Result<CalibrationResult, SoakError> {
    let format = plan.format;
    let params = plan.detector;
    let mut session = CalibrationSession::new(plan)?;
    let mut front =
        CalibrationFrontEnd::new(format, &ChannelLayout::all_real(format.channels), &params)?;
    session.start_ambient()?;
    play(&mut session, &mut front, capture, veto, ambient)?;
    let layout = session.accept_ambient()?;
    front = CalibrationFrontEnd::new(format, &layout, &params)?;

    let mut zone = ZoneIndex(0);
    let mut presented = 0u32;
    loop {
        let (index, accepted, target) = zone_step(&session)?;
        if index != zone {
            (zone, presented) = (index, 0);
        }
        if accepted < target {
            if presented >= target.saturating_mul(MAX_PRESENTED_PER_TARGET) {
                return Err(SoakError::ZoneNotFilled {
                    zone,
                    accepted,
                    presented,
                });
            }
            let x = segment(capture.position(), zone);
            play(&mut session, &mut front, capture, veto, &x)?;
            presented += 1;
        } else if session.auto_advances() {
            session.advance(capture.now())?;
        } else {
            break;
        }
    }
    session.finish_negatives()?;
    let job = session.take_build_job()?;
    let model = job.run(&AtomicBool::new(false), &mut |_| {})?;
    session.finish_build(Ok(model))?;
    Ok(session.commit()?)
}

/// Streams `x` through the front end and feeds every input to the session; while the session
/// is in its ambient step, every block is also its ambient recording.
fn play(
    session: &mut CalibrationSession,
    front: &mut CalibrationFrontEnd,
    capture: &mut Capture,
    veto: &mut VetoTimeline,
    x: &[f32],
) -> Result<(), SoakError> {
    let mut inputs = Vec::new();
    capture.stream(x, |block, now| -> Result<(), SoakError> {
        front.push_block(block, now, veto, &mut inputs)?;
        for input in inputs.drain(..) {
            session.on_input(session_input(input))?;
        }
        if session.step_kind() == StepKind::Ambient {
            session.push_ambient(block)?;
        }
        Ok(())
    })
}

/// The zone step's (index, accepted, target); any other step is `NotInStep{Zone}`.
fn zone_step(session: &CalibrationSession) -> Result<(ZoneIndex, u32, u32), SoakError> {
    match session.view().step {
        CalibrationStep::Zone {
            index,
            accepted,
            target,
        } => Ok((index, accepted, target)),
        CalibrationStep::Ready
        | CalibrationStep::Ambient { .. }
        | CalibrationStep::AmbientReview { .. }
        | CalibrationStep::Negatives { .. }
        | CalibrationStep::ReadyToBuild
        | CalibrationStep::Building { .. }
        | CalibrationStep::Review { .. }
        | CalibrationStep::Failed { .. } => Err(SoakError::Session {
            error: SessionError::NotInStep {
                expected: StepKind::Zone,
                actual: session.step_kind(),
            },
        }),
    }
}

// SPEC: §1.15 defines `CalibrationInput` once, in `pipeline.rs`, but `session.rs` still carries
// the interim copy its own SPEC note says the B1/B2 merge would replace. Neither file is this
// package's to change, so the front end's output is mapped onto the session's copy field for field.
fn session_input(input: pipeline::CalibrationInput) -> session::CalibrationInput {
    match input {
        pipeline::CalibrationInput::Candidate {
            candidate,
            live,
            veto,
        } => session::CalibrationInput::Candidate {
            candidate,
            live,
            veto,
        },
        pipeline::CalibrationInput::Invalid { candidate, error } => {
            session::CalibrationInput::Invalid { candidate, error }
        }
        pipeline::CalibrationInput::Rejected { rejected } => {
            session::CalibrationInput::Rejected { rejected }
        }
        pipeline::CalibrationInput::InputFault { fault } => {
            session::CalibrationInput::InputFault { fault }
        }
    }
}

/// A live pipeline for a calibration result: one Single trigger per zone and the default
/// detector, veto and gesture parameters, as a fresh profile has them.
pub fn live_pipeline(
    result: CalibrationResult,
    format: AudioFormat,
) -> Result<Pipeline, SoakError> {
    let zones = result.model.zone_ids.len();
    let config = PipelineConfig {
        format,
        layout: result.layout,
        triggers: vec![Trigger::Single; zones],
        detector: DetectorParams::DEFAULT,
        veto: VetoParams::DEFAULT,
        gesture: GestureParams::DEFAULT,
    };
    Ok(Pipeline::new(Arc::new(result.model), config)?)
}

// ---------------------------------------------------------------------------
// The soak (§1.20)
// ---------------------------------------------------------------------------

/// The frozen fixture's rate: `CleanSignatures` is a 44.1 kHz mono generator.
const FIXTURE_RATE: u32 = 44_100;
/// `CleanSignatures` partial jitter (±2 % at 1.0), as the v1 soak drew it.
const TAP_JITTER: f32 = 1.0;
/// Batch composition.
const SUSTAINED_PER_BATCH: usize = 4;
const FOREIGN_PER_BATCH: usize = 2;
/// Floor gaps between clips, uniform in this range.
const GAP_MIN: Milliseconds = Milliseconds::new(300.0);
const GAP_MAX: Milliseconds = Milliseconds::new(600.0);
/// Floor streamed into a new pipeline before the first batch: the detector's 200 ms warm-up and
/// its peak-picking history.
const PIPELINE_WARMUP: Milliseconds = Milliseconds::new(500.0);
/// Lcg streams of the fixture (§1.20: A calibration taps, B batch taps, C foreign transients).
const STREAM_A: u64 = 0xA;
const STREAM_B: u64 = 0xB;
const STREAM_C: u64 = 0xC;
/// Rng64 stream of the calibration gaps.
const STREAM_CALIBRATION_GAPS: u64 = 0xCA1;

/// Batches run before measurement starts (§1.20).
pub const WARMUP_BATCHES: u64 = 50;
pub const DEFAULT_SEED: u64 = 0x50AC;

/// Soak parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoakConfig {
    /// 2, 4, 6 or 8 (the zone counts `ModalDesk` and the product offer).
    pub zone_count: usize,
    pub seed: u64,
}

impl Default for SoakConfig {
    fn default() -> Self {
        SoakConfig {
            zone_count: 4,
            seed: DEFAULT_SEED,
        }
    }
}

/// Outcome counts from one or more batches.
///
/// Invariant: `taps == correct + rejected_taps + misclassified`. Splitting a clean miss
/// (`rejected_taps`) from a confidently wrong zone (`misclassified`) is what the detective
/// method needs: a pooled `taps - correct` would hide which failure a regression hit.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BatchTally {
    /// Tap clips presented (one per zone per batch).
    pub taps: u64,
    /// Taps whose first decision is Accepted for their own zone.
    pub correct: u64,
    /// Taps with no decision, or whose first decision is Rejected, Vetoed or Invalid.
    pub rejected_taps: u64,
    /// Taps whose first decision is Accepted for another zone.
    pub misclassified: u64,
    /// Sustained (non-tap) clips presented.
    pub sustained: u64,
    /// Accepted decisions on sustained clips. MUST stay 0.
    pub sustained_false_accepts: u64,
    /// Foreign transients presented: tap-like broadband bursts that match no zone.
    pub foreign_transients: u64,
    /// Foreign transients with at least one decision the model scored: the population the
    /// model's novelty gates must reject. Gate 2 needs enough of them to mean something.
    pub foreign_reaching_model: u64,
    /// Accepted decisions on foreign transients. MUST stay 0.
    pub foreign_false_accepts: u64,
}

impl BatchTally {
    pub fn add(&mut self, o: &BatchTally) {
        self.taps += o.taps;
        self.correct += o.correct;
        self.rejected_taps += o.rejected_taps;
        self.misclassified += o.misclassified;
        self.sustained += o.sustained;
        self.sustained_false_accepts += o.sustained_false_accepts;
        self.foreign_transients += o.foreign_transients;
        self.foreign_reaching_model += o.foreign_reaching_model;
        self.foreign_false_accepts += o.foreign_false_accepts;
    }

    /// correct / taps; None before any tap was presented.
    pub fn accuracy(&self) -> Option<f64> {
        (self.taps > 0).then(|| self.correct as f64 / self.taps as f64)
    }
}

/// One clip of a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Item {
    Tap { zone: ZoneIndex },
    Sustained,
    Foreign,
}

/// A clip as streamed: its item and its first frame. Its frames run to the next clip's start,
/// the floor gap after it included.
#[derive(Debug, Clone, Copy)]
struct Placed {
    item: Item,
    start: u64,
}

/// The calibrated pipeline and the generators that keep producing fresh clips.
pub struct Soak {
    pipeline: Pipeline,
    veto: VetoTimeline,
    capture: Capture,
    fixture: CleanSignatures,
    /// The source of the continuous −80 dBFS pink floor (a one-mic desk's `floor`).
    floor: ModalDesk,
    zone_count: usize,
    seed: u64,
    taps: Lcg,
    foreign: Lcg,
    batch: u64,
    events: Vec<PipelineEvent>,
}

impl Soak {
    /// Calibrates `CleanSignatures` through the session (§1.20) and builds the live pipeline.
    pub fn new(config: SoakConfig) -> Result<Soak, SoakError> {
        let format = AudioFormat {
            // INVARIANT: 44.1 kHz and one channel are inside the SampleRate and ChannelCount ranges.
            rate: SampleRate::new(FIXTURE_RATE).expect("44.1 kHz is a valid rate"),
            channels: ChannelCount::new(1).expect("1 is a valid channel count"),
        };
        let mut desk = DeskConfig::default_for(format.rate, config.seed);
        desk.mics.truncate(1);
        let mut floor = ModalDesk::new(desk)?;
        let fixture = CleanSignatures::new(config.zone_count);
        let mut capture = Capture::new(format);
        let mut veto = VetoTimeline::new(VetoParams::DEFAULT);

        let plan = CalibrationPlan {
            zone_ids: (0..config.zone_count)
                .map(|z| ZoneId::from_uuid(uuid::Uuid::from_u128(1000 + z as u128)))
                .collect(),
            zone_labels: (0..config.zone_count)
                .map(|z| format!("Zone {}", z + 1))
                .collect(),
            taps_per_zone: DEFAULT_TAPS_PER_ZONE,
            format,
            seed: config.seed,
            audio_budget_bytes: DEFAULT_AUDIO_BUDGET_BYTES,
            detector: DetectorParams::DEFAULT,
            training: TrainingParams::DEFAULT,
        };
        let ambient = floor.floor(format.rate.samples_in(AMBIENT_DURATION));
        let mut taps_a = Lcg::new(config.seed ^ STREAM_A);
        let mut gaps = Rng64::derive(config.seed, STREAM_CALIBRATION_GAPS, 0);
        let result = calibrate(plan, &mut capture, &mut veto, &ambient, |_, zone| {
            let gap = gap_frames(format.rate, &mut gaps);
            let clip = fixture.tap(usize::from(zone.0), &mut taps_a, TAP_JITTER);
            let mut x = floor.floor(gap + clip.len());
            mix(&mut x[gap..], &clip);
            x
        })?;

        let mut pipeline = live_pipeline(result, format)?;
        let mut events = Vec::new();
        let warmup = floor.floor(format.rate.samples_in(PIPELINE_WARMUP));
        capture.stream(&warmup, |block, now| -> Result<(), SoakError> {
            Ok(pipeline.push_block(block, now, &mut veto, &mut events)?)
        })?;
        events.clear();

        Ok(Soak {
            pipeline,
            veto,
            capture,
            fixture,
            floor,
            zone_count: config.zone_count,
            seed: config.seed,
            taps: Lcg::new(config.seed ^ STREAM_B),
            foreign: Lcg::new(config.seed ^ STREAM_C),
            batch: 0,
            events,
        })
    }

    pub fn zone_count(&self) -> usize {
        self.zone_count
    }

    /// One batch (§1.20): one tap per zone and 4 sustained clips (stream B) and 2 foreign
    /// transients (stream C), shuffled with `Rng64(seed ^ batch)`, each followed by a uniform
    /// 300-600 ms floor gap, all on the continuous floor and streamed in 10 ms blocks. The
    /// final gap settles every decision of the batch before it returns.
    pub fn run_batch(&mut self) -> Result<BatchTally, SoakError> {
        self.batch += 1;
        let mut clips: Vec<(Item, Vec<f32>)> = Vec::new();
        for z in 0..self.zone_count {
            let clip = self.fixture.tap(z, &mut self.taps, TAP_JITTER);
            // INVARIANT: `Soak::new` calibrated these zones, and `CalibrationSession::new` refuses
            // a plan with more zones than a u16 counts.
            let zone = ZoneIndex(u16::try_from(z).expect("the session accepted this zone count"));
            clips.push((Item::Tap { zone }, clip));
        }
        // SPEC: §1.20 names no stream for the sustained clips; they come from stream B, as the
        // v1 soak drew them from its tap generator.
        for _ in 0..SUSTAINED_PER_BATCH {
            clips.push((Item::Sustained, self.fixture.sustained(&mut self.taps)));
        }
        for _ in 0..FOREIGN_PER_BATCH {
            clips.push((Item::Foreign, self.fixture.foreign_tap(&mut self.foreign)));
        }
        let mut rng = Rng64::new(self.seed ^ self.batch);
        for i in (1..clips.len()).rev() {
            let j = rng.below(i as u64 + 1) as usize;
            clips.swap(i, j);
        }

        let first = self.capture.position().0;
        let mut placed = Vec::with_capacity(clips.len());
        self.events.clear();
        for (item, clip) in clips {
            let gap = gap_frames(self.capture.format().rate, &mut rng);
            let mut x = self.floor.floor(clip.len() + gap);
            mix(&mut x, &clip);
            placed.push(Placed {
                item,
                start: self.capture.position().0,
            });
            let (pipeline, veto, events) = (&mut self.pipeline, &mut self.veto, &mut self.events);
            self.capture
                .stream(&x, |block, now| -> Result<(), SoakError> {
                    Ok(pipeline.push_block(block, now, veto, events)?)
                })?;
        }
        tally(&placed, first, &self.events)
    }
}

/// A uniform 300-600 ms gap in frames.
fn gap_frames(rate: SampleRate, rng: &mut Rng64) -> usize {
    rate.samples_in(Milliseconds::new(rng.range(GAP_MIN.get(), GAP_MAX.get())))
}

/// Adds `clip` onto the start of `x`.
fn mix(x: &mut [f32], clip: &[f32]) {
    for (v, c) in x.iter_mut().zip(clip) {
        *v += c;
    }
}

/// Tallies a batch's decisions by the clip whose frames hold their onset.
fn tally(placed: &[Placed], first: u64, events: &[PipelineEvent]) -> Result<BatchTally, SoakError> {
    let mut per_clip: Vec<Vec<&TapDecision>> = vec![Vec::new(); placed.len()];
    for event in events {
        let decision = match event {
            PipelineEvent::Decision(d) => d,
            PipelineEvent::Gesture(_)
            | PipelineEvent::DetectorRejected(_)
            | PipelineEvent::InputFault(_) => continue,
        };
        let onset = decision.onset.0;
        if onset < first {
            return Err(SoakError::DecisionOutsideBatch {
                onset: decision.onset,
            });
        }
        // The clips tile [first, end of batch): the onset belongs to the last clip starting at
        // or before it.
        let i = placed.partition_point(|p| p.start <= onset) - 1;
        per_clip[i].push(decision);
    }

    let mut t = BatchTally::default();
    for (p, decisions) in placed.iter().zip(&per_clip) {
        let accepted = decisions
            .iter()
            .filter(|d| matches!(d.outcome, DecisionOutcome::Accepted { .. }))
            .count() as u64;
        match p.item {
            Item::Tap { zone } => {
                t.taps += 1;
                match decisions.first().map(|d| &d.outcome) {
                    Some(DecisionOutcome::Accepted { zone: got, .. }) if *got == zone => {
                        t.correct += 1
                    }
                    Some(DecisionOutcome::Accepted { .. }) => t.misclassified += 1,
                    Some(
                        DecisionOutcome::Vetoed { .. }
                        | DecisionOutcome::Rejected { .. }
                        | DecisionOutcome::Invalid { .. },
                    )
                    | None => t.rejected_taps += 1,
                }
            }
            Item::Sustained => {
                t.sustained += 1;
                t.sustained_false_accepts += accepted;
            }
            Item::Foreign => {
                t.foreign_transients += 1;
                if decisions.iter().any(|d| d.scores().is_some()) {
                    t.foreign_reaching_model += 1;
                }
                t.foreign_false_accepts += accepted;
            }
        }
    }
    Ok(t)
}

// ---------------------------------------------------------------------------
// Gates (§1.20)
// ---------------------------------------------------------------------------

/// Gate 5's floor on clean accuracy, applied at `GATED_ZONE_COUNT` zones only: the ACCEPTANCE
/// sanity floor, not a claim.
pub const ACCURACY_FLOOR: f64 = 0.99;
pub const GATED_ZONE_COUNT: usize = 4;
/// Gate 3: live-heap growth from the end of warm-up to the end of the run stays below this.
pub const HEAP_GROWTH_LIMIT_PERCENT: f64 = 5.0;
/// Gate 2: foreign transients that must reach the model, for runs at least `LONG_RUN` long and
/// for shorter ones.
pub const LONG_RUN: Seconds = Seconds::new(600.0);
pub const FOREIGN_NEEDED_LONG: u64 = 100;
pub const FOREIGN_NEEDED_SHORT: u64 = 10;

/// A failed soak gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateFailure {
    SustainedFalseAccepts {
        count: u64,
    },
    ForeignFalseAccepts {
        count: u64,
    },
    ForeignNotExercised {
        reached: u64,
        needed: u64,
    },
    HeapGrowth {
        percent: f64,
    },
    /// No tap was presented, so gate 5 has nothing to measure.
    NoTaps,
    Accuracy {
        accuracy: f64,
    },
}

impl fmt::Display for GateFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GateFailure::SustainedFalseAccepts { count } => {
                write!(f, "{count} false accepts on sustained sound (must be 0)")
            }
            GateFailure::ForeignFalseAccepts { count } => {
                write!(f, "{count} foreign transients accepted (must be 0)")
            }
            GateFailure::ForeignNotExercised { reached, needed } => write!(
                f,
                "only {reached} foreign transients reached the model (need {needed})"
            ),
            GateFailure::HeapGrowth { percent } => write!(
                f,
                "live heap grew {percent:+.3}% (limit {HEAP_GROWTH_LIMIT_PERCENT}%)"
            ),
            GateFailure::NoTaps => write!(f, "no tap was presented"),
            GateFailure::Accuracy { accuracy } => {
                write!(f, "accuracy {accuracy:.4} < {ACCURACY_FLOOR}")
            }
        }
    }
}

/// Live-heap growth from `start` to `end` bytes, in percent of `start`.
pub fn heap_growth_percent(start: usize, end: usize) -> f64 {
    (end as f64 - start as f64) / (start.max(1) as f64) * 100.0
}

/// Gates 1, 2, 3 and 5 of §1.20 (gate 4, no panic, is the run reaching this call). Empty when
/// the run passes.
pub fn gate(
    tally: &BatchTally,
    zone_count: usize,
    duration: Seconds,
    heap_growth_percent: f64,
) -> Vec<GateFailure> {
    let mut failures = Vec::new();
    if tally.sustained_false_accepts != 0 {
        failures.push(GateFailure::SustainedFalseAccepts {
            count: tally.sustained_false_accepts,
        });
    }
    if tally.foreign_false_accepts != 0 {
        failures.push(GateFailure::ForeignFalseAccepts {
            count: tally.foreign_false_accepts,
        });
    }
    let needed = if duration.get() >= LONG_RUN.get() {
        FOREIGN_NEEDED_LONG
    } else {
        FOREIGN_NEEDED_SHORT
    };
    if tally.foreign_reaching_model < needed {
        failures.push(GateFailure::ForeignNotExercised {
            reached: tally.foreign_reaching_model,
            needed,
        });
    }
    if heap_growth_percent >= HEAP_GROWTH_LIMIT_PERCENT {
        failures.push(GateFailure::HeapGrowth {
            percent: heap_growth_percent,
        });
    }
    if zone_count == GATED_ZONE_COUNT {
        match tally.accuracy() {
            None => failures.push(GateFailure::NoTaps),
            Some(accuracy) if accuracy < ACCURACY_FLOOR => {
                failures.push(GateFailure::Accuracy { accuracy })
            }
            Some(_) => {}
        }
    }
    failures
}

#[cfg(test)]
mod tests {
    use super::*;

    fn passing() -> BatchTally {
        BatchTally {
            taps: 400,
            correct: 400,
            sustained: 400,
            foreign_transients: 200,
            foreign_reaching_model: 150,
            ..BatchTally::default()
        }
    }

    #[test]
    fn host_clock_is_frames_times_ticks_over_rate() {
        let r = SampleRate::new(44_100).expect("rate");
        assert_eq!(host_time(r, SampleIndex(0)), HostTime(0));
        assert_eq!(host_time(r, SampleIndex(44_100)), HostTime(10_000_000));
        // 10^7/44100 = 226.757…: frame 1 rounds up, frame 2 (453.51…) up as well.
        assert_eq!(host_time(r, SampleIndex(1)), HostTime(227));
        assert_eq!(host_time(r, SampleIndex(2)), HostTime(454));
    }

    #[test]
    fn capture_streams_10_ms_blocks_with_the_last_frame_as_now() {
        let format = AudioFormat {
            rate: SampleRate::new(48_000).expect("rate"),
            channels: ChannelCount::new(2).expect("channels"),
        };
        let mut capture = Capture::new(format);
        let x = vec![0.0f32; 1000 * 2];
        let mut seen = Vec::new();
        capture
            .stream(&x, |b, now| -> Result<(), SoakError> {
                seen.push((b.start, b.frames(), b.time, now));
                Ok(())
            })
            .expect("whole frames");
        let starts: Vec<_> = seen.iter().map(|s| (s.0 .0, s.1)).collect();
        assert_eq!(starts, vec![(0, 480), (480, 480), (960, 40)]);
        assert_eq!(
            seen[1].2,
            BlockTime::Timestamped {
                host_time: host_time(format.rate, SampleIndex(480))
            }
        );
        assert_eq!(seen[2].3, host_time(format.rate, SampleIndex(999)));
        assert_eq!(capture.position(), SampleIndex(1000));
        let ragged = capture.stream(&[0.0f32; 3], |_, _| -> Result<(), SoakError> { Ok(()) });
        assert!(matches!(ragged, Err(SoakError::Audio { .. })));
    }

    #[test]
    fn gates_pass_a_clean_run_and_name_each_failure() {
        let long = Seconds::new(1800.0);
        assert!(gate(&passing(), 4, long, 0.4).is_empty());

        let mut t = passing();
        t.sustained_false_accepts = 1;
        t.foreign_false_accepts = 2;
        t.foreign_reaching_model = 99;
        t.correct = 395;
        t.rejected_taps = 5;
        assert_eq!(
            gate(&t, 4, long, 5.0),
            vec![
                GateFailure::SustainedFalseAccepts { count: 1 },
                GateFailure::ForeignFalseAccepts { count: 2 },
                GateFailure::ForeignNotExercised {
                    reached: 99,
                    needed: 100
                },
                GateFailure::HeapGrowth { percent: 5.0 },
                GateFailure::Accuracy {
                    accuracy: 395.0 / 400.0
                },
            ]
        );
        // A short run needs 10 foreign transients at the model; other zone counts are not gated
        // on accuracy; a shrinking heap is not growth.
        assert!(gate(&t, 6, Seconds::new(60.0), -3.0)
            .iter()
            .all(|f| !matches!(
                f,
                GateFailure::ForeignNotExercised { .. }
                    | GateFailure::Accuracy { .. }
                    | GateFailure::HeapGrowth { .. }
            )));
        assert_eq!(
            gate(&BatchTally::default(), 4, Seconds::new(60.0), 0.0).last(),
            Some(&GateFailure::NoTaps)
        );
    }

    #[test]
    fn short_soak_calibrates_and_every_tap_lands_in_one_bucket() {
        let mut soak = Soak::new(SoakConfig::default()).expect("the fixture calibrates");
        let mut t = BatchTally::default();
        for _ in 0..8 {
            t.add(&soak.run_batch().expect("batch"));
        }
        assert_eq!(t.taps, 32);
        assert_eq!(t.taps, t.correct + t.rejected_taps + t.misclassified);
        assert_eq!(t.sustained, 32);
        assert_eq!(t.foreign_transients, 16);
        assert!(t.foreign_reaching_model <= t.foreign_transients);
        assert_eq!(t.sustained_false_accepts, 0, "{t:?}");
        assert_eq!(t.foreign_false_accepts, 0, "{t:?}");
    }
}
