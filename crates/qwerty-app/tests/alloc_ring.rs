//! RG3 (p2 §2.3.1): after creation, push/pop pairs through the audio ring make no allocation.
//!
//! Its own test binary so `CountingAllocator` can be the global allocator; counts are per thread
//! (`soak::thread_allocations`), so the harness and other tests cannot disturb them.

use qwerty_app::engine::messages::PacketMeta;
use qwerty_app::engine::ring::{audio_ring, PushOutcome, CAPTURE_RING};
use qwerty_core::audio::{AudioFormat, BlockTime};
use qwerty_core::soak::{thread_allocations, CountingAllocator};
use qwerty_core::units::{ChannelCount, HostTime, SampleIndex, SampleRate};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn rg3_push_pop_makes_no_allocations_after_creation() {
    const FRAMES: u32 = 480;
    let format = AudioFormat {
        rate: SampleRate::new(48_000).expect("valid rate"),
        channels: ChannelCount::new(2).expect("valid channel count"),
    };
    let (mut writer, mut reader) = audio_ring(format, CAPTURE_RING, 1024, FRAMES);
    let packet = vec![0.5f32; FRAMES as usize * 2];
    let mut out = Vec::with_capacity(reader.max_packet_samples());

    let before = thread_allocations();
    for i in 0..10_000u64 {
        let meta = PacketMeta {
            start: SampleIndex(i * u64::from(FRAMES)),
            frames: FRAMES,
            time: BlockTime::Timestamped {
                host_time: HostTime(i as i64),
            },
            discontinuity: false,
        };
        assert_eq!(writer.push(meta, &packet), PushOutcome::Written);
        assert_eq!(reader.pop(&mut out), Some(meta));
    }
    let allocations = thread_allocations() - before;
    assert_eq!(allocations, 0, "push/pop allocated {allocations} times");
}
