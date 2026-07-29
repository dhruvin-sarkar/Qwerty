//! Real audio capture (Phase 2). Owns the `cpal` input stream, downmixes to
//! mono in the callback, and hands samples to the DSP core across the
//! `AudioSource` seam (`ARCHITECTURE.md` → Threading model).
//!
//! The audio callback obeys the real-time rule from `PERFORMANCE.md`: it does
//! not allocate, lock, or log — it only converts samples and pushes them into a
//! lock-free SPSC ring buffer (`rtrb`). If the consumer falls behind and the
//! ring fills, samples are dropped and counted rather than blocking the
//! callback.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
    /// A device selector (index or name substring) matched nothing.
    DeviceNotFound(String),
    Enumerate(cpal::DevicesError),
    DeviceName(cpal::DeviceNameError),
    DefaultConfig(cpal::DefaultStreamConfigError),
    UnsupportedFormat(cpal::SampleFormat),
    BuildStream(cpal::BuildStreamError),
    Play(cpal::PlayStreamError),
    /// The stream reported an asynchronous error after starting (device
    /// unplugged, permission revoked mid-session, driver fault).
    StreamFailed,
}

impl std::fmt::Display for CaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CaptureError::NoInputDevice => write!(f, "no audio input device is available"),
            CaptureError::DeviceNotFound(s) => write!(f, "no input device matched \"{s}\""),
            CaptureError::Enumerate(e) => write!(f, "could not enumerate input devices: {e}"),
            CaptureError::DeviceName(e) => write!(f, "could not read a device name: {e}"),
            CaptureError::DefaultConfig(e) => write!(f, "no default input config: {e}"),
            CaptureError::UnsupportedFormat(fmt) => {
                write!(f, "unsupported sample format: {fmt:?}")
            }
            CaptureError::BuildStream(e) => write!(f, "could not build the input stream: {e}"),
            CaptureError::Play(e) => write!(f, "could not start the input stream: {e}"),
            CaptureError::StreamFailed => write!(
                f,
                "the audio input stream failed (see stderr for the device error)"
            ),
        }
    }
}

impl std::error::Error for CaptureError {}

/// Errors from the end-to-end live harness: capture failed, or calibration
/// (classifier training) failed. A real enum so `--live` never reports a
/// training failure as success (`CLAUDE.md`: fail fast, no silent fallbacks).
#[derive(Debug)]
pub enum LiveError {
    Capture(CaptureError),
    Training(qwerty_core::classifier::ClassifierError),
}

impl std::fmt::Display for LiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiveError::Capture(e) => write!(f, "{e}"),
            LiveError::Training(e) => write!(f, "calibration failed: {e}"),
        }
    }
}

impl std::error::Error for LiveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LiveError::Capture(e) => Some(e),
            LiveError::Training(e) => Some(e),
        }
    }
}

impl From<CaptureError> for LiveError {
    fn from(e: CaptureError) -> Self {
        LiveError::Capture(e)
    }
}

impl From<qwerty_core::classifier::ClassifierError> for LiveError {
    fn from(e: qwerty_core::classifier::ClassifierError) -> Self {
        LiveError::Training(e)
    }
}

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

/// Resolve an input device by list index or case-insensitive name substring;
/// `None` selects the system default.
fn resolve_device(selector: Option<&str>) -> Result<cpal::Device, CaptureError> {
    let host = cpal::default_host();
    let Some(sel) = selector else {
        return host
            .default_input_device()
            .ok_or(CaptureError::NoInputDevice);
    };
    let devices: Vec<cpal::Device> = host
        .input_devices()
        .map_err(CaptureError::Enumerate)?
        .collect();
    if let Ok(idx) = sel.parse::<usize>() {
        return devices
            .into_iter()
            .nth(idx)
            .ok_or_else(|| CaptureError::DeviceNotFound(sel.to_string()));
    }
    let needle = sel.to_lowercase();
    for d in devices {
        if d.name().is_ok_and(|n| n.to_lowercase().contains(&needle)) {
            return Ok(d);
        }
    }
    Err(CaptureError::DeviceNotFound(sel.to_string()))
}

/// Choose the input config to open. Prefers a config that supports the DSP
/// core's tuned rate (`SAMPLE_RATE_HZ`, 44.1 kHz) exactly, because `onset` and
/// `features` compute their window timing (`HOP`, `EDGE_SAMPLES`) and spectral
/// band→frequency mapping (`compute_band_bins`) against that constant — a
/// device negotiated at, say, 48 kHz (a common Windows default) would shift the
/// ~92.9 ms window and misalign every band. This closes the "the pipeline
/// resamples/validates against this" gap documented in `features.rs`: we pick
/// 44.1 kHz when the hardware offers it, rather than blindly taking whatever
/// `default_input_config()` returns. Only when no supported range covers
/// 44.1 kHz do we fall back to the device default (still internally consistent,
/// since calibration and live classification then use the same rate) — and the
/// negotiated rate remains visible via [`AudioSource::sample_rate`] for the UI.
fn choose_input_config(
    device: &cpal::Device,
) -> Result<cpal::SupportedStreamConfig, CaptureError> {
    let target = cpal::SampleRate(SAMPLE_RATE_HZ);
    if let Ok(ranges) = device.supported_input_configs() {
        let mut best: Option<cpal::SupportedStreamConfigRange> = None;
        for r in ranges {
            if r.min_sample_rate() <= target && target <= r.max_sample_rate() {
                // Prefer an f32 range (no per-sample conversion); otherwise take
                // the first range that can hit 44.1 kHz.
                let better = match &best {
                    None => true,
                    Some(b) => {
                        r.sample_format() == cpal::SampleFormat::F32
                            && b.sample_format() != cpal::SampleFormat::F32
                    }
                };
                if better {
                    best = Some(r);
                }
            }
        }
        if let Some(r) = best {
            return Ok(r.with_sample_rate(target));
        }
    }
    // No range covers 44.1 kHz (or the device wouldn't enumerate configs):
    // fall back to the device's own default so capture still works.
    device
        .default_input_config()
        .map_err(CaptureError::DefaultConfig)
}

/// An open capture stream on the default input device. Holds the live `cpal`
/// stream and the consumer end of the sample ring.
pub struct AudioCapture {
    _stream: cpal::Stream,
    consumer: Consumer<f32>,
    dropped: Arc<AtomicUsize>,
    failed: Arc<AtomicBool>,
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
    /// Open a chosen input device (by list index or case-insensitive name
    /// substring; `None` = system default) and start streaming.
    pub fn open(selector: Option<&str>) -> Result<Self, CaptureError> {
        let device = resolve_device(selector)?;
        let device_name = device.name().map_err(CaptureError::DeviceName)?;
        let supported = choose_input_config(&device)?;

        let sample_format = supported.sample_format();
        let channels = supported.channels();
        let sample_rate = supported.sample_rate().0;
        let config: cpal::StreamConfig = supported.into();

        // One second of headroom, at least a few windows' worth.
        let capacity = (sample_rate as usize).max(FEATURE_WINDOW_SAMPLES * 4);
        let (producer, consumer) = RingBuffer::<f32>::new(capacity);
        let dropped = Arc::new(AtomicUsize::new(0));
        let failed = Arc::new(AtomicBool::new(false));
        let ch = channels as usize;

        let stream = match sample_format {
            cpal::SampleFormat::F32 => build_input_stream::<f32>(
                &device,
                &config,
                producer,
                ch,
                dropped.clone(),
                failed.clone(),
            ),
            cpal::SampleFormat::I16 => build_input_stream::<i16>(
                &device,
                &config,
                producer,
                ch,
                dropped.clone(),
                failed.clone(),
            ),
            cpal::SampleFormat::U16 => build_input_stream::<u16>(
                &device,
                &config,
                producer,
                ch,
                dropped.clone(),
                failed.clone(),
            ),
            other => return Err(CaptureError::UnsupportedFormat(other)),
        }
        .map_err(CaptureError::BuildStream)?;

        stream.play().map_err(CaptureError::Play)?;

        Ok(Self {
            _stream: stream,
            consumer,
            dropped,
            failed,
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

    /// Whether the input stream has reported an asynchronous failure.
    pub fn stream_failed(&self) -> bool {
        self.failed.load(Ordering::Relaxed)
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
    failed: Arc<AtomicBool>,
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
        move |err| {
            eprintln!("audio stream error: {err}");
            failed.store(true, Ordering::Relaxed);
        },
        None,
    )
}

/// Adaptive input-gain normalization for the capture path.
///
/// Low-gain mics (e.g. a USB headset) deliver taps 20-40 dB below what the
/// DSP's absolute gates were tuned for. This brings any mic into one level
/// domain by tracking a slow peak envelope and applying a gain toward a target
/// peak. It lives in `qwerty-app` (not `qwerty-core`), so the soak path — which
/// drives the core directly — is provably unaffected.
///
/// Safety (the reason this can ship without live tuning): the gain changes by
/// at most [`GAIN_STEP`] *per block*, so it can never produce the ≥2×-in-one-hop
/// rise the onset detector needs to see a "sharp" attack — a gain ramp cannot
/// masquerade as a tap. It also never attenuates and never chases *silence*
/// upward. Both properties are covered by unit tests below.
pub struct InputNormalizer {
    peak_env: f32,
    gain: f32,
}

/// Target peak the normalizer aims the loudest recent sample at (~-15 dBFS).
const TARGET_PEAK: f32 = 0.18;
/// Gain ceiling (+36 dB) and floor (never attenuate).
const MAX_GAIN: f32 = 64.0;
const MIN_GAIN: f32 = 1.0;
/// Applied gain at or above which the input counts as "low signal": the
/// normalizer is boosting by more than +30 dB (half its range) to reach the
/// target peak, i.e. the mic is hearing the desk only faintly. The Home level
/// meter shows the *normalized* level, so a faint mic still paints an active
/// bar; this threshold is the honest tell the UI uses to warn instead
/// (`CLAUDE.md`: surface degradation, don't hide it behind normalization).
pub const LOW_SIGNAL_GAIN: f32 = MAX_GAIN / 2.0;

/// Whether an applied normalizer `gain` indicates a low-signal mic (see
/// [`LOW_SIGNAL_GAIN`]). A pure decision rule so the "warn the user" gate is
/// unit-tested without a live device — the capture path that produces the gain
/// cannot be.
pub fn gain_indicates_low_signal(gain: f32) -> bool {
    gain >= LOW_SIGNAL_GAIN
}
/// Envelope attack/release per block (fast up, slow down → multi-second release).
const ENV_ATTACK: f32 = 0.5;
const ENV_RELEASE: f32 = 0.02;
/// Max fractional gain change per block — the anti-false-trigger guarantee.
const GAIN_STEP: f32 = 0.01;
/// Block peaks below this are treated as silence: hold gain, don't ramp up.
const SILENCE_PEAK: f32 = 1e-5;

impl Default for InputNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl InputNormalizer {
    pub fn new() -> Self {
        Self {
            peak_env: TARGET_PEAK,
            gain: MIN_GAIN,
        }
    }

    /// The gain currently being applied (for a low-signal UI indicator).
    pub fn gain(&self) -> f32 {
        self.gain
    }

    /// Normalize a drained mono block in place.
    pub fn process(&mut self, buf: &mut [f32]) {
        let block_peak = buf.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        // Peak envelope: fast attack, slow release.
        let rate = if block_peak > self.peak_env {
            ENV_ATTACK
        } else {
            ENV_RELEASE
        };
        self.peak_env += (block_peak - self.peak_env) * rate;

        // Desired gain — but only chase when THIS block has real signal; hold
        // through silence (gated on the block peak, not the slowly-releasing
        // envelope) so a quiet stretch never ramps the floor up into a false
        // trigger.
        let desired = if block_peak > SILENCE_PEAK {
            (TARGET_PEAK / self.peak_env).clamp(MIN_GAIN, MAX_GAIN)
        } else {
            self.gain
        };

        // Move gain toward `desired` by at most GAIN_STEP (fractional) per block.
        let max_delta = GAIN_STEP * self.gain;
        let delta = (desired - self.gain).clamp(-max_delta, max_delta);
        self.gain = (self.gain + delta).clamp(MIN_GAIN, MAX_GAIN);

        for s in buf.iter_mut() {
            *s *= self.gain;
        }
    }
}

/// Phase 2 CLI validation: open the default mic and print live onset decisions
/// as the user taps the desk. Runs for `seconds`, then prints a summary. This
/// is the "tap a real desk and see decisions printed live" harness from
/// `ROADMAP.md` (zone classification arrives with calibration in Phase 4).
pub fn run_monitor(seconds: u64, device: Option<&str>) -> Result<(), CaptureError> {
    let mut cap = AudioCapture::open(device)?;
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
             detection still works.)",
            cap.sample_rate(),
            SAMPLE_RATE_HZ,
        );
    }
    println!(
        "A per-second level meter prints below. If it reads 'silent' while you tap, the mic \
         is not hearing the desk — pick another device, e.g. `--monitor {seconds} \"Realtek\"`.",
    );

    let mut detector = OnsetDetector::new();
    let mut normalizer = InputNormalizer::new();
    let mut buf: Vec<f32> = Vec::with_capacity(cap.sample_rate() as usize);
    let (mut taps, mut rejects) = (0u32, 0u32);
    // Per-second signal accounting for the level meter (evidence, not a guess).
    // Measured on the RAW signal so the meter still reveals the true mic level.
    let (mut sec_sumsq, mut sec_count, mut sec_peak) = (0.0f64, 0usize, 0.0f32);

    let start = Instant::now();
    let mut last_tick = start;
    while start.elapsed() < Duration::from_secs(seconds) {
        if cap.stream_failed() {
            return Err(CaptureError::StreamFailed);
        }
        buf.clear();
        cap.drain(&mut buf);
        if !buf.is_empty() {
            for &s in &buf {
                sec_sumsq += (s as f64) * (s as f64);
                sec_peak = sec_peak.max(s.abs());
            }
            sec_count += buf.len();
            // Normalize for detection (mic-independent level); the meter above
            // already captured the raw level.
            normalizer.process(&mut buf);
            for ev in detector.process(&buf) {
                // Per-onset diagnostics: the numbers that discriminate why a
                // real tap does or doesn't register (see the theory in the
                // commit that added this). `assess_tap` is the exact gate the
                // calibration wizard applies to an accepted transient; running
                // it here (noise_floor = 0, isolating the weak/clip test) shows
                // whether the wizard would then reject a tap the detector took.
                let (peak, wrms, decay) = window_stats(&ev.window);
                if ev.is_transient {
                    taps += 1;
                    let verdict = qwerty_core::calibration::assess_tap(&ev.window, 0.0);
                    println!(
                        "  TAP  #{taps:<4} peak {}  win-rms {}  decay {decay:.2}  →  wizard: {verdict:?} ({})",
                        fmt_dbfs(peak),
                        fmt_dbfs(wrms),
                        verdict.reason(),
                    );
                } else {
                    rejects += 1;
                    println!(
                        "  ---  rejected as sustained: peak {}  win-rms {}  decay {decay:.2}  (tap gate needs decay < 0.50)",
                        fmt_dbfs(peak),
                        fmt_dbfs(wrms),
                    );
                }
            }
        }
        if last_tick.elapsed() >= Duration::from_secs(1) {
            let rms = if sec_count > 0 {
                (sec_sumsq / sec_count as f64).sqrt() as f32
            } else {
                0.0
            };
            println!(
                "  level  {sec_count:>6} samp/s  rms {:<8}  peak {:<8}  gain x{:<4.0}  taps={taps} rejects={rejects}",
                fmt_dbfs(rms),
                fmt_dbfs(sec_peak),
                normalizer.gain(),
            );
            sec_sumsq = 0.0;
            sec_count = 0;
            sec_peak = 0.0;
            last_tick = Instant::now();
        }
        std::thread::sleep(Duration::from_millis(15));
    }

    println!(
        "\nMonitor done: {taps} taps accepted, {rejects} rejected, {} samples dropped.",
        cap.dropped(),
    );
    Ok(())
}

/// Diagnostic stats for one captured window: peak amplitude, whole-window RMS,
/// and the head→tail energy decay ratio (the exact quantity the onset
/// detector's sustained-sound gate thresholds at 0.50). Computed here from the
/// event's window so no change to `qwerty-core` is needed just to observe it.
fn window_stats(w: &[f32]) -> (f32, f32, f32) {
    let n = w.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let mut peak = 0.0f32;
    let mut sumsq = 0.0f64;
    for &s in w {
        peak = peak.max(s.abs());
        sumsq += (s as f64) * (s as f64);
    }
    let rms = (sumsq / n as f64).sqrt() as f32;
    // Same ~15 ms edges the sustain gate uses (onset.rs EDGE_SAMPLES).
    let edge = ((SAMPLE_RATE_HZ as usize * 15) / 1000).min(n / 4).max(1);
    let mean_sq = |sl: &[f32]| -> f64 {
        if sl.is_empty() {
            0.0
        } else {
            sl.iter().map(|s| (*s as f64) * (*s as f64)).sum::<f64>() / sl.len() as f64
        }
    };
    let head = mean_sq(&w[..edge]);
    let tail = mean_sq(&w[n - edge..]);
    let decay = (tail / (head + 1e-12)) as f32;
    (peak, rms, decay)
}

/// Format a linear amplitude (0..1) as dBFS, or "silent" for ~zero.
fn fmt_dbfs(amp: f32) -> String {
    if amp <= 1e-9 {
        "silent".to_string()
    } else {
        format!("{:>5.1}dB", 20.0 * amp.log10())
    }
}

fn wait_for_enter() {
    let mut s = String::new();
    // Block until the user presses Enter; a closed/piped stdin simply proceeds.
    let _ = std::io::stdin().read_line(&mut s);
}

/// Collect `taps_per_zone` accepted taps from the mic, extracting a feature
/// vector for each. Blocks until enough transients are captured.
fn capture_taps(
    cap: &mut AudioCapture,
    extractor: &mut qwerty_core::features::FeatureExtractor,
    taps_per_zone: usize,
) -> Result<Vec<qwerty_core::features::FeatureVector>, CaptureError> {
    let mut detector = OnsetDetector::new();
    let mut buf = Vec::new();
    let mut out = Vec::new();
    while out.len() < taps_per_zone {
        if cap.stream_failed() {
            return Err(CaptureError::StreamFailed);
        }
        buf.clear();
        cap.drain(&mut buf);
        for ev in detector.process(&buf) {
            if ev.is_transient {
                out.push(extractor.extract(&ev.window));
                println!("  captured {}/{taps_per_zone}", out.len());
                if out.len() >= taps_per_zone {
                    break;
                }
            }
        }
        std::thread::sleep(Duration::from_millis(15));
    }
    Ok(out)
}

/// Phase 2 end-to-end CLI: calibrate `zone_count` zones from the mic, train the
/// Phase 1 classifier, then classify taps live — "tap a real desk and see
/// correct zone decisions printed live" (`ROADMAP.md`). This is a terminal-only
/// preview of the calibration wizard built properly in Phase 4; it does not
/// persist a profile.
pub fn run_live(
    zone_count: usize,
    taps_per_zone: usize,
    device: Option<&str>,
) -> Result<(), LiveError> {
    use qwerty_core::classifier::ClassifierParams;
    use qwerty_core::features::{FeatureExtractor, FeatureVector};
    use qwerty_core::profile::ZoneId;

    let mut cap = AudioCapture::open(device)?;
    println!(
        "Live harness on \"{}\" @ {} Hz — {} zones, {} taps each.",
        cap.device_name,
        cap.sample_rate(),
        zone_count,
        taps_per_zone,
    );

    let mut extractor = FeatureExtractor::new();
    let zone_ids: Vec<ZoneId> = (0..zone_count).map(|_| ZoneId::new()).collect();
    let mut samples: Vec<(ZoneId, FeatureVector)> = Vec::new();

    for (i, &zid) in zone_ids.iter().enumerate() {
        print!(
            "\nZone {} — press Enter, then tap that spot {} times… ",
            i + 1,
            taps_per_zone,
        );
        use std::io::Write;
        let _ = std::io::stdout().flush(); // best-effort flush before the prompt
        wait_for_enter();
        // Discard any audio buffered before the user was ready.
        let mut stale = Vec::new();
        cap.drain(&mut stale);

        for fv in capture_taps(&mut cap, &mut extractor, taps_per_zone)? {
            samples.push((zid, fv));
        }
    }

    let classifier = ClassifierParams::train(&samples, &[])?;
    println!(
        "\nTrained on {} taps across {} zones. Tap live now (Ctrl-C to stop)…",
        samples.len(),
        zone_count,
    );

    let mut detector = OnsetDetector::new();
    let mut buf = Vec::new();
    loop {
        if cap.stream_failed() {
            return Err(CaptureError::StreamFailed.into());
        }
        buf.clear();
        cap.drain(&mut buf);
        for ev in detector.process(&buf) {
            if ev.is_transient {
                let fv = extractor.extract(&ev.window);
                let decision = classifier.classify(&fv);
                match decision.zone_id {
                    Some(z) => {
                        let idx = zone_ids.iter().position(|x| *x == z).unwrap() + 1;
                        println!("  ZONE {idx}   (confidence {:.2})", decision.confidence);
                    }
                    None => println!("  rejected (unknown tap)"),
                }
            }
        }
        std::thread::sleep(Duration::from_millis(15));
    }
}

#[cfg(test)]
mod normalizer_tests {
    use super::*;

    /// A block with the given peak (two opposite-sign spikes; rest silent).
    fn block(peak: f32, n: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        if n >= 2 {
            v[0] = peak;
            v[n / 2] = -peak;
        }
        v
    }

    #[test]
    fn never_attenuates_and_stays_capped() {
        let mut nz = InputNormalizer::new();
        for _ in 0..2000 {
            let mut b = block(0.3, 512);
            nz.process(&mut b);
            assert!(nz.gain() >= MIN_GAIN - 1e-6 && nz.gain() <= MAX_GAIN + 1e-6);
        }
    }

    #[test]
    fn gain_step_per_block_cannot_fake_an_attack() {
        // The safety guarantee: gain changes by at most GAIN_STEP per block, so
        // it can never rise >=2x within a hop and read as a sharp onset.
        let mut nz = InputNormalizer::new();
        let mut prev = nz.gain();
        for i in 0..1000 {
            let mut b = block(if i % 2 == 0 { 1e-3 } else { 0.5 }, 512);
            nz.process(&mut b);
            let ratio = nz.gain() / prev;
            assert!(
                (ratio - 1.0).abs() <= GAIN_STEP + 1e-4,
                "gain jumped {prev} -> {} (ratio {ratio})",
                nz.gain(),
            );
            prev = nz.gain();
        }
    }

    #[test]
    fn boosts_a_quiet_but_clean_signal() {
        let mut nz = InputNormalizer::new();
        for _ in 0..3000 {
            let mut b = block(5e-4, 512); // ~-66 dBFS, like a low-gain mic tap
            nz.process(&mut b);
        }
        assert!(nz.gain() > 10.0, "gain only reached {}", nz.gain());
        assert!(5e-4 * nz.gain() > 0.02, "signal not meaningfully lifted");
    }

    #[test]
    fn low_signal_gate_trips_only_when_boost_is_heavy() {
        // A mic at unity or light boost is fine; a heavily-boosted one (>= half
        // the gain range) is the "faint mic, taps may be missed" state the Home
        // warning is for. The boundary is inclusive at LOW_SIGNAL_GAIN.
        assert!(!gain_indicates_low_signal(MIN_GAIN));
        assert!(!gain_indicates_low_signal(LOW_SIGNAL_GAIN - 0.1));
        assert!(gain_indicates_low_signal(LOW_SIGNAL_GAIN));
        assert!(gain_indicates_low_signal(MAX_GAIN));
    }

    #[test]
    fn does_not_amplify_silence() {
        let mut nz = InputNormalizer::new();
        for _ in 0..2000 {
            let mut b = vec![0.0f32; 512];
            nz.process(&mut b);
        }
        assert!(
            (nz.gain() - MIN_GAIN).abs() < 1e-6,
            "silence pushed gain to {}",
            nz.gain(),
        );
    }
}
