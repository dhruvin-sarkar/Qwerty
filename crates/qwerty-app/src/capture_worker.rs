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

use qwerty_core::onset::OnsetDetector;

use crate::audio_thread::AudioCapture;

/// How often the worker emits an input-level sample (bounded-rate meter).
const LEVEL_INTERVAL: Duration = Duration::from_millis(50);
/// Worker poll cadence when draining the capture ring.
const POLL_INTERVAL: Duration = Duration::from_millis(15);

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
    Onset {
        #[allow(dead_code)]
        window: Vec<f32>,
        is_transient: bool,
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
    pub fn start(device: Option<String>, ctx: egui::Context) -> Self {
        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_worker = stop.clone();
        let handle = std::thread::spawn(move || {
            run_worker(device, tx, ctx, stop_worker);
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
    let mut buf: Vec<f32> = Vec::new();
    let (mut sum_sq, mut count, mut peak) = (0.0f64, 0usize, 0.0f32);
    let mut last_level = Instant::now();

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
            for &s in &buf {
                sum_sq += (s as f64) * (s as f64);
                peak = peak.max(s.abs());
            }
            count += buf.len();
            for ev in detector.process(&buf) {
                let _ = tx.send(CaptureEvent::Onset {
                    window: ev.window,
                    is_transient: ev.is_transient,
                });
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

        std::thread::sleep(POLL_INTERVAL);
    }
}
