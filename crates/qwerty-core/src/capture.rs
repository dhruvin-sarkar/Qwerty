//! The audio-source boundary (`ARCHITECTURE.md` → The `AudioSource` boundary).
//!
//! `qwerty-core` never touches `cpal`. The real stream lives in
//! `qwerty-app::audio_thread`, which owns the device and feeds mono samples
//! across this seam into the onset detector. Keeping the trait here — and the
//! cpal glue out — is what lets the DSP core be tested with synthetic buffers
//! and ported later without a rewrite.

/// Minimal description of an audio input, implemented by the platform capture
/// layer so the DSP core never depends on `cpal`. The platform layer downmixes
/// to mono itself (necessarily, since it converts from the device's native
/// sample type first) and feeds the resulting samples across this seam.
pub trait AudioSource {
    fn sample_rate(&self) -> u32;
    fn channels(&self) -> u16;
}
