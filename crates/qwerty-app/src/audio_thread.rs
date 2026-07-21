//! Real audio capture (Phase 2). Owns the `cpal` input stream, downmixes to
//! mono in the callback, and hands samples to the DSP core across the
//! `AudioSource` seam (`ARCHITECTURE.md` → Threading model).
//!
//! The audio callback obeys the real-time rule from `PERFORMANCE.md`: it does
//! not allocate, lock, or log — it only converts samples and pushes them into a
//! lock-free SPSC ring buffer (`rtrb`). If the consumer falls behind and the
//! ring fills, samples are dropped and counted rather than blocking the
//! callback.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample};
use rtrb::{Consumer, Producer, RingBuffer};

use qwerty_core::capture::AudioSource;
use qwerty_core::features::{FEATURE_WINDOW_SAMPLES, SAMPLE_RATE_HZ};
use qwerty_core::onset::OnsetDetector;

/// Errors from opening or enumerating audio input. Distinct variants so the UI
/// (Phase 3+) can react differently (`ARCHITECTURE.md` → Error handling).
#[derive(Debug)]
pub enum CaptureError {
    NoInputDevice,
    Enumerate(cpal::DevicesError),
    DeviceName(cpal::DeviceNameError),
    DefaultConfig(cpal::DefaultStreamConfigError),
    UnsupportedFormat(cpal::SampleFormat),
    BuildStream(cpal::BuildStreamError),
    Play(cpal::PlayStreamError),
}

impl std::fmt::Display for CaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CaptureError::NoInputDevice => write!(f, "no audio input device is available"),
            CaptureError::Enumerate(e) => write!(f, "could not enumerate input devices: {e}"),
            CaptureError::DeviceName(e) => write!(f, "could not read a device name: {e}"),
            CaptureError::DefaultConfig(e) => write!(f, "no default input config: {e}"),
            CaptureError::UnsupportedFormat(fmt) => {
                write!(f, "unsupported sample format: {fmt:?}")
            }
            CaptureError::BuildStream(e) => write!(f, "could not build the input stream: {e}"),
            CaptureError::Play(e) => write!(f, "could not start the input stream: {e}"),
        }
    }
}

impl std::error::Error for CaptureError {}

/// List the names of all available audio input devices.
pub fn list_input_devices() -> Result<Vec<String>, CaptureError> {
    let host = cpal::default_host();
    let devices = host.input_devices().map_err(CaptureError::Enumerate)?;
    let mut names = Vec::new();
    for d in devices {
        names.push(d.name().map_err(CaptureError::DeviceName)?);
    }
    Ok(names)
}

/// An open capture stream on the default input device. Holds the live `cpal`
/// stream and the consumer end of the sample ring.
pub struct AudioCapture {
    _stream: cpal::Stream,
    consumer: Consumer<f32>,
    dropped: Arc<AtomicUsize>,
    pub device_name: String,
    sample_rate: u32,
    channels: u16,
}

impl AudioSource for AudioCapture {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn channels(&self) -> u16 {
        self.channels
    }
}

impl AudioCapture {
    /// Open the system default input device and start streaming.
    pub fn open_default() -> Result<Self, CaptureError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or(CaptureError::NoInputDevice)?;
        let device_name = device.name().map_err(CaptureError::DeviceName)?;
        let supported = device
            .default_input_config()
            .map_err(CaptureError::DefaultConfig)?;

        let sample_format = supported.sample_format();
        let channels = supported.channels();
        let sample_rate = supported.sample_rate().0;
        let config: cpal::StreamConfig = supported.into();

        // One second of headroom, at least a few windows' worth.
        let capacity = (sample_rate as usize).max(FEATURE_WINDOW_SAMPLES * 4);
        let (producer, consumer) = RingBuffer::<f32>::new(capacity);
        let dropped = Arc::new(AtomicUsize::new(0));
        let ch = channels as usize;

        let stream = match sample_format {
            cpal::SampleFormat::F32 => {
                build_input_stream::<f32>(&device, &config, producer, ch, dropped.clone())
            }
            cpal::SampleFormat::I16 => {
                build_input_stream::<i16>(&device, &config, producer, ch, dropped.clone())
            }
            cpal::SampleFormat::U16 => {
                build_input_stream::<u16>(&device, &config, producer, ch, dropped.clone())
            }
            other => return Err(CaptureError::UnsupportedFormat(other)),
        }
        .map_err(CaptureError::BuildStream)?;

        stream.play().map_err(CaptureError::Play)?;

        Ok(Self {
            _stream: stream,
            consumer,
            dropped,
            device_name,
            sample_rate,
            channels,
        })
    }

    /// Drain all currently-available mono samples into `out`.
    pub fn drain(&mut self, out: &mut Vec<f32>) {
        while let Ok(s) = self.consumer.pop() {
            out.push(s);
        }
    }

    /// Number of samples dropped because the ring was full (consumer behind).
    pub fn dropped(&self) -> usize {
        self.dropped.load(Ordering::Relaxed)
    }
}

/// Build a typed input stream whose callback downmixes to mono and pushes into
/// the ring. Real-time safe: no allocation, no locking, no logging on the hot
/// path (a full ring drops + counts instead of blocking).
fn build_input_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut producer: Producer<f32>,
    channels: usize,
    dropped: Arc<AtomicUsize>,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let ch = channels.max(1);
    device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            for frame in data.chunks(ch) {
                let mut sum = 0.0f32;
                for &s in frame {
                    sum += f32::from_sample(s);
                }
                let mono = sum / frame.len() as f32;
                if producer.push(mono).is_err() {
                    dropped.fetch_add(1, Ordering::Relaxed);
                }
            }
        },
        move |err| eprintln!("audio stream error: {err}"),
        None,
    )
}

/// Phase 2 CLI validation: open the default mic and print live onset decisions
/// as the user taps the desk. Runs for `seconds`, then prints a summary. This
/// is the "tap a real desk and see decisions printed live" harness from
/// `ROADMAP.md` (zone classification arrives with calibration in Phase 4).
pub fn run_monitor(seconds: u64) -> Result<(), CaptureError> {
    let mut cap = AudioCapture::open_default()?;
    println!(
        "Listening on \"{}\" @ {} Hz, {} ch for {}s. Tap the desk near the mic…",
        cap.device_name,
        cap.sample_rate(),
        cap.channels(),
        seconds,
    );
    if cap.sample_rate() != SAMPLE_RATE_HZ {
        println!(
            "(note: device runs at {} Hz vs the {} Hz the DSP core is tuned for; onset \
             detection still works — the classification path resamples at calibration time.)",
            cap.sample_rate(),
            SAMPLE_RATE_HZ,
        );
    }

    let mut detector = OnsetDetector::new();
    let mut buf: Vec<f32> = Vec::with_capacity(cap.sample_rate() as usize);
    let (mut taps, mut rejects) = (0u32, 0u32);

    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(seconds) {
        buf.clear();
        cap.drain(&mut buf);
        if !buf.is_empty() {
            for ev in detector.process(&buf) {
                if ev.is_transient {
                    taps += 1;
                    println!("  TAP  #{taps:<4} transient accepted");
                } else {
                    rejects += 1;
                    println!("  ---  rejected (sustained / noise)");
                }
            }
        }
        std::thread::sleep(Duration::from_millis(15));
    }

    println!(
        "\nMonitor done: {taps} taps accepted, {rejects} rejected, {} samples dropped.",
        cap.dropped(),
    );
    Ok(())
}
