//! The audio ring (p2 §2.3.1): one SPSC pair of `rtrb` rings, interleaved f32 samples plus one
//! `PacketMeta` per packet, used capture → processing and processing → recorder.
//!
//! The writer pushes a packet's samples first, then its meta; the reader pops the meta first,
//! then exactly its samples. A reader therefore never sees a partial packet, and a push is all or
//! nothing. Neither side allocates after creation.

use qwerty_core::audio::AudioFormat;
use qwerty_core::units::Milliseconds;

use crate::engine::messages::PacketMeta;

/// Capture → processing: 100 periods of headroom, so only a processing stall of ~1 s drops audio
/// (1024 packets).
pub const CAPTURE_RING: Milliseconds = Milliseconds::new(1000.0);
/// Processing → recorder: disk writes stall longer than DSP (2048 packets).
pub const RECORDER_RING: Milliseconds = Milliseconds::new(2000.0);

/// `max_frames`: the largest packet the writer will push (buffer_frames); readers reserve
/// max_frames × channels once.
pub fn audio_ring(
    format: AudioFormat,
    capacity: Milliseconds,
    max_packets: usize,
    max_frames: u32,
) -> (RingWriter, RingReader) {
    let channels = usize::from(format.channels.get());
    let max_packet_samples = max_frames as usize * channels;
    // The sample ring always holds at least one largest packet, so a packet within max_frames can
    // always be written once the reader drains.
    let sample_slots = (format.rate.samples_in(capacity) * channels).max(max_packet_samples);
    let (samples_tx, samples_rx) = rtrb::RingBuffer::<f32>::new(sample_slots);
    let (meta_tx, meta_rx) = rtrb::RingBuffer::<PacketMeta>::new(max_packets);
    (
        RingWriter {
            samples: samples_tx,
            metas: meta_tx,
            channels,
            max_frames,
        },
        RingReader {
            samples: samples_rx,
            metas: meta_rx,
            format,
            max_frames,
        },
    )
}

pub struct RingWriter {
    samples: rtrb::Producer<f32>,
    metas: rtrb::Producer<PacketMeta>,
    channels: usize,
    max_frames: u32,
}

pub struct RingReader {
    samples: rtrb::Consumer<f32>,
    metas: rtrb::Consumer<PacketMeta>,
    format: AudioFormat,
    max_frames: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushOutcome {
    Written,
    Dropped { frames: u32 },
}

impl RingWriter {
    /// Writes the meta and meta.frames·channels samples, or nothing at all. Never allocates.
    /// A packet larger than max_frames is Dropped (impossible from WASAPI; counted like any drop).
    ///
    /// Panics if `interleaved.len() != meta.frames · channels`: the reader pops exactly that many
    /// samples per meta, so a mismatch would desynchronize the stream (a caller bug).
    pub fn push(&mut self, meta: PacketMeta, interleaved: &[f32]) -> PushOutcome {
        assert_eq!(
            interleaved.len(),
            meta.frames as usize * self.channels,
            "a packet's samples must be meta.frames × channels"
        );
        let dropped = PushOutcome::Dropped {
            frames: meta.frames,
        };
        if meta.frames > self.max_frames {
            return dropped;
        }
        // Single producer: the free space checked here can only grow before the writes below.
        if self.metas.slots() == 0 || self.samples.slots() < interleaved.len() {
            return dropped;
        }
        // INVARIANT: the sample ring has interleaved.len() free slots (checked above, and only
        // the reader changes it, by freeing more).
        self.samples
            .push_entire_slice(interleaved)
            .expect("the sample ring had room for the whole packet");
        // INVARIANT: the meta ring has a free slot (checked above, and only the reader changes it,
        // by freeing more).
        self.metas
            .push(meta)
            .expect("the meta ring had room for the packet's meta");
        PushOutcome::Written
    }
}

impl RingReader {
    pub fn format(&self) -> AudioFormat {
        self.format
    }

    pub fn max_packet_samples(&self) -> usize {
        self.max_frames as usize * usize::from(self.format.channels.get())
    }

    /// Pops the next whole packet into `out` (cleared, then filled; capacity reserved once). None
    /// if empty.
    pub fn pop(&mut self, out: &mut Vec<f32>) -> Option<PacketMeta> {
        let meta = self.metas.pop().ok()?;
        let n = meta.frames as usize * usize::from(self.format.channels.get());
        out.clear();
        // A no-op once `out` holds max_packet_samples of capacity, so pop allocates at most once
        // per buffer.
        out.reserve(self.max_packet_samples());
        // INVARIANT: the writer commits a packet's samples before its meta, and the meta is
        // visible, so all n samples are readable.
        let chunk = self
            .samples
            .read_chunk(n)
            .expect("a packet's samples are committed before its meta");
        let (first, second) = chunk.as_slices();
        out.extend_from_slice(first);
        out.extend_from_slice(second);
        chunk.commit_all();
        Some(meta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwerty_core::audio::BlockTime;
    use qwerty_core::rng::Rng64;
    use qwerty_core::units::{ChannelCount, HostTime, SampleIndex, SampleRate};

    fn format(channels: u16) -> AudioFormat {
        AudioFormat {
            rate: SampleRate::new(48_000).expect("valid rate"),
            channels: ChannelCount::new(channels).expect("valid channel count"),
        }
    }

    fn meta(start: u64, frames: u32) -> PacketMeta {
        PacketMeta {
            start: SampleIndex(start),
            frames,
            time: BlockTime::Timestamped {
                host_time: HostTime(start as i64),
            },
            discontinuity: start.is_multiple_of(7),
        }
    }

    /// The samples' bit patterns, compared as integers (bytewise, and NaN-safe).
    fn as_bits(samples: &[f32]) -> &[u32] {
        // SAFETY: f32 and u32 have the same size and alignment, and every bit pattern is a valid
        // u32.
        unsafe { std::slice::from_raw_parts(samples.as_ptr().cast::<u32>(), samples.len()) }
    }

    #[test]
    fn rg1_random_packets_round_trip_bit_exactly_through_wraparound() {
        const MAX_FRAMES: u32 = 4096;
        const PACKETS: usize = 100_000;
        let mut rng = Rng64::new(0x5EED_0001);
        // Arbitrary bit patterns, NaN payloads included: the ring must copy them untouched.
        let mut bits = 0u32;
        let pattern: Vec<f32> = (0..2 * MAX_FRAMES as usize * 8)
            .map(|_| {
                bits = bits.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                f32::from_bits(bits)
            })
            .collect();
        // 10^5 packets in all, spread over the three channel counts.
        for (k, channels) in [1u16, 2, 8].into_iter().enumerate() {
            let fmt = format(channels);
            // A small ring (200 ms: two largest packets) so the packets wrap it many times.
            let (mut writer, mut reader) =
                audio_ring(fmt, Milliseconds::new(200.0), 64, MAX_FRAMES);
            let mut out = Vec::with_capacity(reader.max_packet_samples());
            let mut start = 0u64;
            let packets = PACKETS / 3 + usize::from(k == 0) * (PACKETS % 3);
            let mut queued = std::collections::VecDeque::new();
            for i in 0..packets {
                let frames = 1 + rng.below(u64::from(MAX_FRAMES)) as u32;
                let len = frames as usize * usize::from(channels);
                let offset = rng.below((pattern.len() - len) as u64) as usize;
                let m = meta(start, frames);
                assert_eq!(
                    writer.push(m, &pattern[offset..offset + len]),
                    PushOutcome::Written
                );
                queued.push_back((m, offset, len));
                start += u64::from(frames);
                // Drain every other packet, so the reader also sees two packets queued at once.
                if i % 2 == 1 || i + 1 == packets {
                    while let Some((want, offset, len)) = queued.pop_front() {
                        let got = reader.pop(&mut out).expect("a packet was pushed");
                        assert_eq!(got, want);
                        assert_eq!(out.len(), len);
                        assert!(
                            as_bits(&out) == as_bits(&pattern[offset..offset + len]),
                            "packet samples differ"
                        );
                    }
                    assert!(reader.pop(&mut out).is_none());
                }
            }
        }
    }

    #[test]
    fn rg2_full_ring_drops_whole_packets_and_recovers_after_a_drain() {
        let fmt = format(2);
        let frames = 480u32;
        let samples = vec![0.25f32; frames as usize * 2];

        // Sample-bound: 50 ms at 48 kHz holds 2400 frames, five 480-frame packets.
        let (mut writer, mut reader) = audio_ring(fmt, Milliseconds::new(50.0), 64, frames);
        for i in 0..5 {
            assert_eq!(
                writer.push(meta(i * 480, frames), &samples),
                PushOutcome::Written
            );
        }
        assert_eq!(
            writer.push(meta(2400, frames), &samples),
            PushOutcome::Dropped { frames }
        );
        let mut out = Vec::new();
        for i in 0..5 {
            let got = reader.pop(&mut out).expect("written packet");
            assert_eq!(got, meta(i * 480, frames));
            assert_eq!(out.len(), samples.len(), "never a partial packet");
        }
        assert!(
            reader.pop(&mut out).is_none(),
            "the dropped packet left nothing"
        );
        assert_eq!(
            writer.push(meta(2880, frames), &samples),
            PushOutcome::Written
        );
        assert_eq!(reader.pop(&mut out), Some(meta(2880, frames)));

        // Meta-bound: two packets fill a 2-packet ring although the samples still fit.
        let (mut writer, mut reader) = audio_ring(fmt, Milliseconds::new(1000.0), 2, frames);
        assert_eq!(writer.push(meta(0, frames), &samples), PushOutcome::Written);
        assert_eq!(
            writer.push(meta(480, frames), &samples),
            PushOutcome::Written
        );
        assert_eq!(
            writer.push(meta(960, frames), &samples),
            PushOutcome::Dropped { frames }
        );
        assert_eq!(reader.pop(&mut out), Some(meta(0, frames)));
        assert_eq!(out.len(), samples.len());
        assert_eq!(
            writer.push(meta(960, frames), &samples),
            PushOutcome::Written
        );

        // Larger than max_frames: Dropped, nothing written.
        let (mut writer, mut reader) = audio_ring(fmt, Milliseconds::new(1000.0), 8, frames);
        let big = vec![0.0f32; (frames as usize + 1) * 2];
        assert_eq!(
            writer.push(meta(0, frames + 1), &big),
            PushOutcome::Dropped { frames: frames + 1 }
        );
        assert!(reader.pop(&mut out).is_none());
    }
}
