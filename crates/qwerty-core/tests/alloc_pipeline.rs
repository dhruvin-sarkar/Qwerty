//! Allocation checks of the streaming path (§0.9): P6 and M13's `map_point` check (§1.10.7).
//!
//! Its own test binary so `CountingAllocator` can be the global allocator; counts are per thread
//! (`soak::thread_allocations`), so the harness and other tests cannot disturb them.

use qwerty_core::audio::{AudioBlock, AudioFormat, BlockTime, ChannelLayout};
use qwerty_core::descriptor::{Descriptor, DescriptorLayout, DESCRIPTOR_DIMS};
use qwerty_core::detector::DetectorParams;
use qwerty_core::gesture::{GestureParams, Trigger};
use qwerty_core::model::TrainedModel;
use qwerty_core::pipeline::{Pipeline, PipelineConfig};
use qwerty_core::profile::ZoneId;
use qwerty_core::rng::Rng64;
use qwerty_core::soak::{thread_allocations, CountingAllocator};
use qwerty_core::training::{train, TapGroup, TrainingParams, TrainingSet};
use qwerty_core::units::{ChannelCount, HostTime, SampleIndex, SampleRate, TapIndex, ZoneIndex};
use qwerty_core::veto::{VetoParams, VetoTimeline};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const HZ: u32 = 48_000;
const ZONES: usize = 2;
const TAPS_PER_ZONE: usize = 8;
/// 10 ms blocks, the engine's processing block.
const BLOCK: usize = 480;

fn rate() -> SampleRate {
    SampleRate::new(HZ).expect("valid rate")
}

fn mono() -> ChannelCount {
    ChannelCount::new(1).expect("valid channel count")
}

fn layout() -> DescriptorLayout {
    DescriptorLayout::from_channels(&ChannelLayout::all_real(mono()))
}

/// A descriptor of N(0, 1) values offset by `shift` (mono: the tail beyond 38 dims stays 0).
fn descriptor(rng: &mut Rng64, shift: f64) -> Descriptor {
    let mut values = [0.0f32; DESCRIPTOR_DIMS];
    for v in values.iter_mut().take(layout().dims()) {
        *v = (shift + rng.normal()) as f32;
    }
    Descriptor {
        layout: layout(),
        values,
    }
}

/// A valid mono 48 kHz model from synthetic Gaussian descriptors (the audio test below never
/// reaches it: noise makes no candidate).
fn model() -> TrainedModel {
    let mut rng = Rng64::new(42);
    let taps = (0..ZONES * TAPS_PER_ZONE)
        .map(|i| {
            let zone = i % ZONES;
            TapGroup {
                zone: ZoneIndex(zone as u16),
                tap: TapIndex(i as u32),
                variants: vec![descriptor(&mut rng, 4.0 * zone as f64)],
            }
        })
        .collect();
    let set = TrainingSet {
        layout: layout(),
        rate: rate(),
        zone_ids: (0..ZONES)
            .map(|z| ZoneId::from_uuid(uuid::Uuid::from_u128(z as u128 + 1)))
            .collect(),
        seed: 0,
        taps,
        negatives: Vec::new(),
    };
    train(
        &set,
        &TrainingParams::DEFAULT,
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .expect("synthetic descriptors train")
}

#[test]
fn p6_push_block_on_noise_allocates_nothing() {
    let format = AudioFormat {
        rate: rate(),
        channels: mono(),
    };
    let config = PipelineConfig {
        format,
        layout: ChannelLayout::all_real(mono()),
        triggers: vec![Trigger::Single; ZONES],
        detector: DetectorParams::DEFAULT,
        veto: VetoParams::DEFAULT,
        gesture: GestureParams::DEFAULT,
    };
    let mut pipeline = Pipeline::new(Arc::new(model()), config).expect("compatible model");
    let mut veto = VetoTimeline::new(VetoParams::DEFAULT);

    // 0.5 s of warm-up, then 10 s measured: stationary −60 dBFS white noise, generated up front.
    let warmup = HZ as usize / 2;
    let measured = 10 * HZ as usize;
    let mut rng = Rng64::new(7);
    let x: Vec<f32> = (0..warmup + measured)
        .map(|_| (1e-3 * rng.normal()) as f32)
        .collect();
    let mut out = Vec::with_capacity(256);
    let ticks_per_frame = HostTime::TICKS_PER_SECOND as f64 / f64::from(HZ);
    let host = |frame: usize| HostTime((frame as f64 * ticks_per_frame).round() as i64);
    let mut push = |pipeline: &mut Pipeline, veto: &mut VetoTimeline, first: usize| {
        let block = AudioBlock::new(
            format,
            SampleIndex(first as u64),
            &x[first..first + BLOCK],
            BlockTime::Timestamped {
                host_time: host(first),
            },
            false,
        )
        .expect("whole frames");
        out.clear();
        pipeline
            .push_block(&block, host(first + BLOCK - 1), veto, &mut out)
            .expect("matching format");
    };
    for first in (0..warmup).step_by(BLOCK) {
        push(&mut pipeline, &mut veto, first);
    }

    // The counter is live in this binary: one boxed value is one allocation.
    let probe = thread_allocations();
    std::hint::black_box(Box::new(probe));
    assert_eq!(thread_allocations() - probe, 1);

    let before = thread_allocations();
    for first in (warmup..warmup + measured).step_by(BLOCK) {
        push(&mut pipeline, &mut veto, first);
    }
    let allocations = thread_allocations() - before;

    assert_eq!(
        pipeline.counters().candidates,
        0,
        "noise makes no candidate"
    );
    assert_eq!(allocations, 0, "push_block allocated {allocations} times");
}

/// M13 (§1.10.7): `map_point` makes 0 allocations. The rest of M13 is `training::m13_sound_map`.
#[test]
fn map_point_allocates_nothing() {
    let model = model();
    let mut scratch = model.scratch();
    let mut rng = Rng64::new(3);
    let live: Vec<Descriptor> = (0..100).map(|_| descriptor(&mut rng, 2.0)).collect();
    let before = thread_allocations();
    let mut sum = 0.0f32;
    for d in &live {
        let at = model
            .map_point(d, &mut scratch)
            .expect("a compatible descriptor");
        sum += at[0] + at[1];
    }
    let allocations = thread_allocations() - before;
    assert!(sum.is_finite());
    assert_eq!(allocations, 0, "map_point allocated {allocations} times");
}
