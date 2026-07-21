//! The audio-source boundary (`ARCHITECTURE.md` → The `AudioSource` boundary).
//!
//! `qwerty-core` never touches `cpal`. The real stream lives in
//! `qwerty-app::audio_thread`, which owns the device and feeds mono samples
//! across this seam into the onset detector. Keeping the trait here — and the
//! cpal glue out — is what lets the DSP core be tested with synthetic buffers
//! and ported later without a rewrite.

/// Minimal description of an audio input, implemented by the platform capture
/// layer so the DSP core never depends on `cpal`.
pub trait AudioSource {
    fn sample_rate(&self) -> u32;
    fn channels(&self) -> u16;
}

/// Average one interleaved frame's channels into a single mono sample.
///
/// Pure arithmetic, no allocation — safe to call inside the real-time audio
/// callback (`PERFORMANCE.md`: the callback thread must not allocate). An empty
/// frame yields silence rather than a division by zero.
#[inline]
pub fn frame_to_mono(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    frame.iter().sum::<f32>() / frame.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_averages_channels() {
        assert_eq!(frame_to_mono(&[0.5]), 0.5);
        assert_eq!(frame_to_mono(&[1.0, -1.0]), 0.0);
        assert_eq!(frame_to_mono(&[0.2, 0.4, 0.6]), 0.4);
    }

    #[test]
    fn empty_frame_is_silence() {
        assert_eq!(frame_to_mono(&[]), 0.0);
    }
}
