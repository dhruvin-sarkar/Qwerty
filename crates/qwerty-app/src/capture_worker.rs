//! Live capture bridge: streams the real microphone into the UI thread
//! (`ARCHITECTURE.md` → Threading model — the processing thread that reads the
//! ring, runs onset detection, and posts to the UI over a channel).
//!
//! `cpal::Stream` is `!Send` on Windows, so the stream can never cross a thread
//! boundary. The worker thread therefore *opens the device itself* and owns the
//! stream for its whole life; the outcome of opening arrives as the first event
//! on the channel (`Started` or `Failed`), not as a `Result` from `start`.
//!
//! The worker requests a UI repaint after every event, so the window wakes only
//! when there is real audio activity — matching the redraw discipline in
//! `PERFORMANCE.md` (the level meter is a bounded-rate live view, not a
//! free-running loop).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use qwerty_core::features::{
    FeatureExtractor, FEATURE_WINDOW_SAMPLES, SAMPLE_RATE_HZ, SPECTRAL_BAND_COUNT,
};
use qwerty_core::onset::OnsetDetector;

use crate::audio_thread::{AudioCapture, InputNormalizer};

/// How often the worker emits an input-level sample (bounded-rate meter).
const LEVEL_INTERVAL: Duration = Duration::from_millis(50);
/// Worker poll cadence when draining the capture ring.
const POLL_INTERVAL: Duration = Duration::from_millis(15);
/// Diagnostics frame cadence. `MOTION.md` prescribes a fixed ~30 fps for the
/// live views ("30fps is sufficient for legibility, not 60+"). The worker polls
/// every [`POLL_INTERVAL`] (15 ms), so a frame can only fire on a poll tick:
/// 30 ms = two ticks ≈ 33 fps, the closest clean multiple to 30. (33 ms would
/// round up to three ticks ≈ 22 fps, below the target — so this is deliberately
/// a multiple of the poll interval, not the naive 1000/30.)
const FRAME_INTERVAL: Duration = Duration::from_millis(30);
/// How much recent audio the Diagnostics views keep (0.25 s). Bounded, and
/// comfortably larger than one feature window so the spectrum always has input.
const RECENT_SPAN: usize = SAMPLE_RATE_HZ as usize / 4;
/// Display points in one waveform frame (an envelope, not raw samples).
const WAVEFORM_POINTS: usize = 256;

/// How much the worker computes per frame. The extra Diagnostics work (an FFT
/// every 33 ms plus a waveform envelope) is opt-in so Home listening and the
/// wizard stay at the idle-CPU budget in `PERFORMANCE.md` — there is one worker
/// implementation, not two, just a bounded amount of extra output when asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureDetail {
    /// Level meter + onsets only (Home, wizard, evaluation).
    Standard,
    /// Adds waveform + spectrum frames for the Diagnostics screen.
    Diagnostics,
}

/// An event from the capture worker to the UI thread.
#[derive(Debug, Clone)]
pub enum CaptureEvent {
    /// The device opened and streaming started. `sample_rate` is consumed by
    /// the calibration wizard (Phase 4b.2) to flag device/DSP-rate mismatch;
    /// Home listening reads only `device_name`.
    Started {
        device_name: String,
        #[allow(dead_code)]
        sample_rate: u32,
    },
    /// A periodic input-level sample (linear amplitudes in `[0, 1]`).
    Level { rms: f32, peak: f32 },
    /// A detected onset and its feature window. `is_transient` distinguishes a
    /// tap from a rejected sustained sound; `window` is fed to
    /// `CalibrationSession::record_tap` by the wizard (Phase 4b.2). Home
    /// listening reads only `is_transient` for its tap/reject counters.
    ///
    /// `detected_at` is stamped when the worker finishes assembling the window
    /// (i.e. once the full 90 ms window after the transient has been captured).
    /// The Evaluation screen (Phase 6) uses it to measure host-side processing
    /// latency; added to the fixed ~92.9 ms window-fill it forms the end-to-end
    /// latency the report records (`ACCEPTANCE.md`).
    Onset {
        window: Vec<f32>,
        is_transient: bool,
        detected_at: Instant,
    },
    /// A Diagnostics display frame, emitted at a fixed 30 fps and only when the
    /// worker was started with [`CaptureDetail::Diagnostics`].
    ///
    /// `waveform` is a max-abs envelope of the recent input decimated to
    /// [`WAVEFORM_POINTS`] points (not raw samples — the UI never needs 11k
    /// points to draw a 300 px trace). `bands` is the classifier's own
    /// normalized spectral-band vector for the most recent feature window, so
    /// the spectrogram shows exactly what the model sees rather than a prettier
    /// but unrelated STFT. It is the fixed-size array straight from the extractor
    /// (`Copy`, no per-frame allocation).
    Frame {
        waveform: Vec<f32>,
        bands: [f32; SPECTRAL_BAND_COUNT],
    },
    /// The device failed to open, or the stream faulted mid-session (unplugged,
    /// permission revoked). Terminal — the worker exits after sending this.
    Failed(String),
}

/// A running capture worker. Dropping it stops the worker thread and closes the
/// device (RAII), so the UI simply drops it to pause listening.
pub struct LiveCapture {
    rx: Receiver<CaptureEvent>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl LiveCapture {
    /// Start capturing from `device` (list index or name substring; `None` =
    /// system default). Returns immediately; watch for `Started`/`Failed` on
    /// [`poll`](Self::poll). `ctx` is pinged so the UI wakes on each event.
    /// `detail` selects whether Diagnostics display frames are computed.
    pub fn start(device: Option<String>, ctx: egui::Context, detail: CaptureDetail) -> Self {
        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_worker = stop.clone();
        let handle = std::thread::spawn(move || {
            run_worker(device, tx, ctx, stop_worker, detail);
        });
        Self {
            rx,
            stop,
            handle: Some(handle),
        }
    }

    /// Drain all pending events (non-blocking).
    pub fn poll(&self) -> Vec<CaptureEvent> {
        self.rx.try_iter().collect()
    }
}

impl Drop for LiveCapture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// The worker body: open the device (reporting the outcome), then drain the
/// ring, detect onsets, and emit level samples until asked to stop.
fn run_worker(
    device: Option<String>,
    tx: mpsc::Sender<CaptureEvent>,
    ctx: egui::Context,
    stop: Arc<AtomicBool>,
    detail: CaptureDetail,
) {
    let mut cap = match AudioCapture::open(device.as_deref()) {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(CaptureEvent::Failed(e.to_string()));
            ctx.request_repaint();
            return;
        }
    };
    let _ = tx.send(CaptureEvent::Started {
        device_name: cap.device_name.clone(),
        sample_rate: {
            use qwerty_core::capture::AudioSource;
            cap.sample_rate()
        },
    });
    ctx.request_repaint();

    let mut detector = OnsetDetector::new();
    let mut normalizer = InputNormalizer::new();
    let mut buf: Vec<f32> = Vec::new();
    let (mut sum_sq, mut count, mut peak) = (0.0f64, 0usize, 0.0f32);
    let mut last_level = Instant::now();

    // Diagnostics-only state: a bounded rolling tail of recent audio plus the
    // extractor that turns its newest feature window into a spectrum column.
    // Both are absent entirely in `Standard` detail, so the ordinary listening
    // path pays neither the memory nor the per-frame FFT.
    let diagnostics = detail == CaptureDetail::Diagnostics;
    let mut recent: Vec<f32> = Vec::new();
    let mut extractor = diagnostics.then(FeatureExtractor::new);
    let mut last_frame = Instant::now();

    while !stop.load(Ordering::Relaxed) {
        if cap.stream_failed() {
            let _ = tx.send(CaptureEvent::Failed(
                "the audio input stream failed (device unplugged or access lost)".into(),
            ));
            ctx.request_repaint();
            return;
        }

        buf.clear();
        cap.drain(&mut buf);
        if !buf.is_empty() {
            // Normalize to a mic-independent level before both the meter and
            // onset detection, so the wizard's level-based gates work on any mic.
            normalizer.process(&mut buf);
            for &s in &buf {
                sum_sq += (s as f64) * (s as f64);
                peak = peak.max(s.abs());
            }
            count += buf.len();
            if diagnostics {
                // Keep only the most recent RECENT_SPAN samples, so memory is
                // bounded no matter how long the screen stays open.
                recent.extend_from_slice(&buf);
                if recent.len() > RECENT_SPAN {
                    recent.drain(..recent.len() - RECENT_SPAN);
                }
            }
            let onsets = detector.process(&buf);
            let had_onset = !onsets.is_empty();
            for ev in onsets {
                let _ = tx.send(CaptureEvent::Onset {
                    window: ev.window,
                    is_transient: ev.is_transient,
                    // Stamp when the window finished assembling; the UI turns this
                    // into host-side processing latency at classify time.
                    detected_at: Instant::now(),
                });
            }
            // Wake the UI immediately on a real onset (like Started/Level/Failed
            // do). Without this, an onset waits for the next bounded-rate frame
            // (~33 ms), and that scheduling delay would be folded into the
            // Evaluation screen's measured per-tap latency, inflating the report.
            if had_onset {
                ctx.request_repaint();
            }
        }

        if last_level.elapsed() >= LEVEL_INTERVAL {
            let rms = if count > 0 {
                (sum_sq / count as f64).sqrt() as f32
            } else {
                0.0
            };
            let _ = tx.send(CaptureEvent::Level { rms, peak });
            ctx.request_repaint();
            sum_sq = 0.0;
            count = 0;
            peak = 0.0;
            last_level = Instant::now();
        }

        // Diagnostics display frame at a fixed 30 fps — the one place a
        // continuous live view is allowed, and deliberately rate-limited here in
        // the worker rather than repainting on every audio callback
        // (`PERFORMANCE.md`, `MOTION.md`).
        if let Some(ex) = extractor.as_mut() {
            if last_frame.elapsed() >= FRAME_INTERVAL && recent.len() >= FEATURE_WINDOW_SAMPLES {
                let waveform = envelope(&recent, WAVEFORM_POINTS);
                let window = &recent[recent.len() - FEATURE_WINDOW_SAMPLES..];
                let bands = ex.extract(window).spectral_bands;
                let _ = tx.send(CaptureEvent::Frame { waveform, bands });
                ctx.request_repaint();
                last_frame = Instant::now();
            }
        }

        std::thread::sleep(POLL_INTERVAL);
    }
}

/// Decimate `samples` to `points` buckets, each the maximum absolute amplitude
/// in its bucket. This is a peak envelope: drawn symmetrically about the centre
/// line it reads as a waveform, and unlike naive stride-sampling it cannot miss
/// a transient that falls between sample points — which matters here, because
/// the whole app is about spotting taps.
fn envelope(samples: &[f32], points: usize) -> Vec<f32> {
    if samples.is_empty() || points == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(points);
    for i in 0..points {
        let lo = i * samples.len() / points;
        let hi = ((i + 1) * samples.len() / points).max(lo + 1).min(samples.len());
        let mut m = 0.0f32;
        for &s in &samples[lo..hi] {
            m = m.max(s.abs());
        }
        out.push(m);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_has_requested_length_and_catches_peaks() {
        // A single spike anywhere must survive decimation (a stride-sampler
        // could step right over it).
        let mut samples = vec![0.0f32; 4000];
        samples[1234] = 0.9;
        let env = envelope(&samples, 256);
        assert_eq!(env.len(), 256);
        assert!(env.iter().any(|v| (*v - 0.9).abs() < 1e-6), "peak was lost");
        assert!(env.iter().all(|v| *v >= 0.0), "envelope is magnitude only");
    }

    #[test]
    fn envelope_handles_degenerate_input() {
        assert!(envelope(&[], 16).is_empty());
        assert!(envelope(&[0.5, 0.5], 0).is_empty());
        // More requested points than samples must not panic or produce empties.
        assert_eq!(envelope(&[0.2, 0.4], 8).len(), 8);
    }
}
