//! Telemetry (p2 §2.5 step 8, §2.8; p35 §3.3): the processing thread's `Meter`, `Spectrum` and
//! `Waveform` frames, and the controller's one inspection-log builder, `trace_entry`.
//!
//! Every dB value is 10·log10(power) floored at `TELEMETRY_DB_FLOOR`, so it is always finite
//! (RUST-RT-16). The engine computes nothing above the requested level.

use qwerty_core::audio::{AudioBlock, AudioFormat};
use qwerty_core::detector::{Detector, DetectorParams, GateReport, RejectedCandidate};
use qwerty_core::dsp::{Biquad, BAND_COUNT};
use qwerty_core::model::{ModelError, ModelScratch, TrainedModel};
use qwerty_core::pipeline::TapDecision;
use qwerty_core::units::{Decibels, HostDuration, HostTime, Milliseconds, SampleIndex};

use crate::engine::protocol::{
    Gate, GateRow, GateUnit, TelemetryFrame, TelemetryLevel, TraceEntry, TraceOutcome,
    TELEMETRY_DB_FLOOR,
};

/// `Meter` at ≤ 20 Hz: one frame per 50 ms (p35 §3.3).
pub const METER_INTERVAL: Milliseconds = Milliseconds::new(50.0);
/// `Spectrum` and `Waveform` at ≤ 30 Hz each.
pub const DIAGNOSTICS_INTERVAL: Milliseconds = Milliseconds::new(1000.0 / 30.0);
/// `Waveform`: the last 100 ms of the first real channel.
pub const WAVEFORM_SPAN: Milliseconds = Milliseconds::new(100.0);
/// `Waveform`: 256 min/max columns.
pub const WAVEFORM_COLUMNS: usize = 256;

/// 10·log10(power), floored at `TELEMETRY_DB_FLOOR` (a zero, negative or non-finite power gives
/// the floor).
pub fn power_db(power: f64) -> Decibels {
    floored_db(10.0 * power.log10())
}

/// A level that is already in dB (the detector's `FrameTelemetry`), floored the same way.
pub fn level_db(db: f32) -> Decibels {
    floored_db(f64::from(db))
}

/// `db` when finite and above the floor, else the floor (NaN, −∞ and log of 0 all floor).
fn floored_db(db: f64) -> Decibels {
    if db.is_finite() && db > TELEMETRY_DB_FLOOR.get() {
        // INVARIANT: checked finite on the line above.
        Decibels::try_new(db).expect("a finite level is a valid Decibels")
    } else {
        TELEMETRY_DB_FLOOR
    }
}

/// The processing thread's telemetry builder (§2.5 step 8). It allocates only in `set_format`
/// (once per capture ring) and for each `Waveform` frame's two vectors (level Diagnostics only).
pub struct TelemetryBuilder {
    level: TelemetryLevel,
    format: Option<AudioFormat>,
    /// One 100 Hz high-pass per channel, as the detector filters (`DetectorParams::hpf_cutoff`).
    hpf: Vec<Biquad>,
    meter_sum: f64,
    meter_peak: f64,
    meter_count: u64,
    /// The last `WAVEFORM_SPAN` of channel 0, a ring.
    wave: Vec<f32>,
    wave_next: usize,
    wave_len: usize,
    next_meter: Option<HostTime>,
    next_diagnostics: Option<HostTime>,
}

impl Default for TelemetryBuilder {
    fn default() -> Self {
        TelemetryBuilder::new()
    }
}

impl TelemetryBuilder {
    /// Starts at `TelemetryLevel::Off` (§2.5).
    pub fn new() -> TelemetryBuilder {
        TelemetryBuilder {
            level: TelemetryLevel::Off,
            format: None,
            hpf: Vec::new(),
            meter_sum: 0.0,
            meter_peak: 0.0,
            meter_count: 0,
            wave: Vec::new(),
            wave_next: 0,
            wave_len: 0,
            next_meter: None,
            next_diagnostics: None,
        }
    }

    pub fn level(&self) -> TelemetryLevel {
        self.level
    }

    pub fn set_level(&mut self, level: TelemetryLevel) {
        self.level = level;
        self.reset_windows();
    }

    /// A new capture format (a ring arrived): rebuilds the filters and the waveform ring.
    pub fn set_format(&mut self, format: AudioFormat) {
        let hpf = Biquad::highpass(format.rate, DetectorParams::DEFAULT.hpf_cutoff)
            // INVARIANT: 100 Hz is below Nyquist at every valid rate (fs ≥ 16 kHz).
            .expect("the detector's 100 Hz high-pass is valid at every sample rate");
        self.hpf = vec![hpf; usize::from(format.channels.get())];
        self.wave = vec![0.0; format.rate.samples_in(WAVEFORM_SPAN).max(1)];
        self.format = Some(format);
        self.reset_windows();
    }

    /// Capture stopped: no frames until the next format.
    pub fn clear_format(&mut self) {
        self.format = None;
        self.reset_windows();
    }

    fn reset_windows(&mut self) {
        for f in &mut self.hpf {
            f.reset();
        }
        self.meter_sum = 0.0;
        self.meter_peak = 0.0;
        self.meter_count = 0;
        self.wave_next = 0;
        self.wave_len = 0;
        self.next_meter = None;
        self.next_diagnostics = None;
    }

    /// Accumulates one block: the high-passed mean and max square for `Meter`, and channel 0
    /// for `Waveform`. Nothing at level Off.
    // SPEC: p35 §3.3 measures the Meter over "all real channels" and the Waveform on "the first
    // real channel", but ProcessingSetup carries no ChannelLayout and the core Detector exposes
    // none, so processing measures every channel of the capture format and uses channel 0.
    pub fn on_block(&mut self, block: &AudioBlock<'_>) {
        if self.level == TelemetryLevel::Off || self.format != Some(block.format) {
            return;
        }
        let channels = self.hpf.len();
        for frame in block.samples.chunks_exact(channels) {
            for (x, f) in frame.iter().zip(self.hpf.iter_mut()) {
                let y = f64::from(f.process(*x));
                let sq = y * y;
                self.meter_sum += sq;
                if sq > self.meter_peak {
                    self.meter_peak = sq;
                }
            }
            self.meter_count += channels as u64;
            if self.level == TelemetryLevel::Diagnostics {
                let cap = self.wave.len();
                self.wave[self.wave_next] = frame[0];
                self.wave_next = (self.wave_next + 1) % cap;
                self.wave_len = (self.wave_len + 1).min(cap);
            }
        }
    }

    /// The frames due at `now`, appended to `out`: `Meter` at ≥ Meter, `Spectrum` and `Waveform`
    /// at Diagnostics. `detector` is the current setup's (None in Idle: no Meter noise and no
    /// Spectrum, so no frame at all).
    pub fn due(
        &mut self,
        now: HostTime,
        detector: Option<&Detector>,
        out: &mut Vec<TelemetryFrame>,
    ) {
        let Some(detector) = detector else {
            return;
        };
        if self.format.is_none() {
            return;
        }
        match self.level {
            TelemetryLevel::Off => {}
            TelemetryLevel::Meter => self.meter_due(now, detector, out),
            TelemetryLevel::Diagnostics => {
                self.meter_due(now, detector, out);
                self.diagnostics_due(now, detector, out);
            }
        }
    }

    fn meter_due(&mut self, now: HostTime, detector: &Detector, out: &mut Vec<TelemetryFrame>) {
        if !is_due(&mut self.next_meter, now, METER_INTERVAL) || self.meter_count == 0 {
            return;
        }
        let rms = power_db(self.meter_sum / self.meter_count as f64);
        let peak = power_db(self.meter_peak);
        let pooled = &detector.noise().pooled;
        let mean = pooled.iter().sum::<f64>() / pooled.len() as f64;
        let channels = self.hpf.len().max(1) as f64;
        out.push(TelemetryFrame::Meter {
            rms,
            peak,
            noise: power_db(mean / channels),
        });
        self.meter_sum = 0.0;
        self.meter_peak = 0.0;
        self.meter_count = 0;
    }

    fn diagnostics_due(
        &mut self,
        now: HostTime,
        detector: &Detector,
        out: &mut Vec<TelemetryFrame>,
    ) {
        if !is_due(&mut self.next_diagnostics, now, DIAGNOSTICS_INTERVAL) {
            return;
        }
        if let Some(frame) = detector.last_frame() {
            let mut bands = Box::new([TELEMETRY_DB_FLOOR; BAND_COUNT]);
            let mut noise = [TELEMETRY_DB_FLOOR; BAND_COUNT];
            for b in 0..BAND_COUNT {
                bands[b] = level_db(frame.bands_db[b]);
                noise[b] = level_db(frame.noise_db[b]);
            }
            out.push(TelemetryFrame::Spectrum {
                bands,
                noise,
                flux: finite_f32(frame.flux),
                threshold: finite_f32(frame.threshold),
            });
        }
        if self.wave_len > 0 {
            out.push(self.waveform());
        }
    }

    /// Min/max of the waveform ring, oldest to newest, in `WAVEFORM_COLUMNS` columns.
    fn waveform(&self) -> TelemetryFrame {
        let cap = self.wave.len();
        let len = self.wave_len;
        let oldest = (self.wave_next + cap - len) % cap;
        let columns = WAVEFORM_COLUMNS.min(len);
        let mut min = Vec::with_capacity(columns);
        let mut max = Vec::with_capacity(columns);
        for c in 0..columns {
            let from = c * len / columns;
            let to = ((c + 1) * len / columns).max(from + 1);
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for i in from..to {
                let x = finite_f32(self.wave[(oldest + i) % cap]);
                lo = lo.min(x);
                hi = hi.max(x);
            }
            min.push(lo);
            max.push(hi);
        }
        TelemetryFrame::Waveform { min, max }
    }
}

/// True (and the next deadline set) when `next` is unset or reached.
fn is_due(next: &mut Option<HostTime>, now: HostTime, interval: Milliseconds) -> bool {
    match *next {
        Some(at) if now < at => false,
        Some(_) | None => {
            *next = now.checked_add(HostDuration::from_ms(interval));
            true
        }
    }
}

/// A non-finite sample (a faulty driver) is shown as 0 so the frame stays serializable.
fn finite_f32(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// The inspection log (§2.8): one builder for every TraceEntry
// ---------------------------------------------------------------------------

/// The four detector gates in the detector's evaluation order, each with the
/// `DetectorParams::DEFAULT` limit it is compared with and the detector's own verdict (§1.6).
pub fn gate_rows(gates: &GateReport) -> [GateRow; 4] {
    let p = DetectorParams::DEFAULT;
    let valid_bands = f64::from(gates.valid_bands);
    let min_bands = p.decay_min_valid_bands as f64;
    [
        GateRow {
            gate: Gate::Snr,
            value: gates.snr.get(),
            limit: p.snr_gate.get(),
            unit: GateUnit::Decibels,
            passed: gates.snr.get() >= p.snr_gate.get(),
        },
        GateRow {
            gate: Gate::BroadbandFraction,
            value: finite_f64(gates.broadband_fraction),
            limit: p.broadband_fraction,
            unit: GateUnit::Fraction,
            passed: gates.broadband_fraction >= p.broadband_fraction,
        },
        GateRow {
            gate: Gate::ValidBands,
            value: valid_bands,
            limit: min_bands,
            unit: GateUnit::Count,
            passed: valid_bands >= min_bands,
        },
        GateRow {
            gate: Gate::DecayFraction,
            value: finite_f64(gates.decay_fraction),
            limit: p.decay_fraction,
            unit: GateUnit::Fraction,
            passed: gates.decay_fraction >= p.decay_fraction,
        },
    ]
}

fn finite_f64(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// What one inspection-log entry describes.
pub enum TraceSource<'a> {
    /// A final pipeline decision (Listening, Evaluating, Diagnostics with a profile).
    Decision { decision: &'a TapDecision },
    /// A detector rejection (any setup).
    DetectorRejected { rejected: &'a RejectedCandidate },
    /// Monitor: a candidate passed the gates; there is no model.
    Detected {
        onset: SampleIndex,
        gates: &'a GateReport,
    },
}

/// The one `TraceEntry` builder (§2.8). `model` is the active model and the controller's own
/// scratch (`model.scratch()`, once per model load). The second value is `map_point`'s error,
/// impossible after `validate()`: the caller logs it and counts it in `faults`; the entry then
/// has no map point.
pub fn trace_entry(
    at: HostTime,
    source: TraceSource<'_>,
    model: Option<(&TrainedModel, &mut ModelScratch)>,
) -> (TraceEntry, Option<ModelError>) {
    match source {
        TraceSource::Decision { decision } => {
            let (map_point, error) = match (decision.descriptor(), model) {
                (Some(descriptor), Some((model, scratch))) => {
                    match model.map_point(descriptor, scratch) {
                        Ok(point) => (Some(point), None),
                        Err(error) => (None, Some(error)),
                    }
                }
                (Some(_), None) | (None, Some(_)) | (None, None) => (None, None),
            };
            let entry = TraceEntry {
                at,
                onset: Some(decision.onset),
                gates: Some(gate_rows(&decision.gates)),
                outcome: TraceOutcome::Decided {
                    outcome: Box::new(decision.outcome.clone()),
                    scores: decision.scores().cloned(),
                },
                map_point,
            };
            (entry, error)
        }
        TraceSource::DetectorRejected { rejected } => (
            TraceEntry {
                at,
                onset: rejected.onset,
                gates: rejected.gates.as_ref().map(gate_rows),
                outcome: TraceOutcome::DetectorRejected {
                    reason: rejected.reason,
                },
                map_point: None,
            },
            None,
        ),
        TraceSource::Detected { onset, gates } => (
            TraceEntry {
                at,
                onset: Some(onset),
                gates: Some(gate_rows(gates)),
                outcome: TraceOutcome::Detected,
                map_point: None,
            },
            None,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwerty_core::audio::BlockTime;
    use qwerty_core::units::{ChannelCount, SampleRate};

    fn format() -> AudioFormat {
        AudioFormat {
            rate: SampleRate::new(48_000).expect("rate"),
            channels: ChannelCount::new(2).expect("channels"),
        }
    }

    #[test]
    fn power_db_is_floored_and_finite() {
        assert_eq!(power_db(0.0), TELEMETRY_DB_FLOOR);
        assert_eq!(power_db(-1.0), TELEMETRY_DB_FLOOR);
        assert_eq!(power_db(f64::NAN), TELEMETRY_DB_FLOOR);
        assert!((power_db(0.5).get() - (-3.0103)).abs() < 1e-3);
        assert_eq!(level_db(f32::NEG_INFINITY), TELEMETRY_DB_FLOOR);
    }

    #[test]
    fn gate_rows_follow_the_detector_verdicts() {
        let gates = GateReport {
            snr: Decibels::new(12.0),
            broadband_fraction: 0.2,
            valid_bands: 5,
            decay_fraction: 0.9,
            clipped: false,
            trigger_flux: 1.0,
        };
        let rows = gate_rows(&gates);
        assert_eq!(
            rows.map(|r| r.gate),
            [
                Gate::Snr,
                Gate::BroadbandFraction,
                Gate::ValidBands,
                Gate::DecayFraction
            ]
        );
        assert_eq!(rows.map(|r| r.passed), [true, false, true, true]);
    }

    #[test]
    fn nothing_accumulates_at_level_off_and_waveform_has_256_columns() {
        let mut t = TelemetryBuilder::new();
        t.set_format(format());
        let samples: Vec<f32> = (0..9600).map(|i| ((i % 48) as f32 / 48.0) - 0.5).collect();
        let block = AudioBlock {
            format: format(),
            start: SampleIndex(0),
            samples: &samples,
            time: BlockTime::Untimed,
            discontinuity: false,
        };
        t.on_block(&block);
        assert_eq!(t.meter_count, 0);
        t.set_level(TelemetryLevel::Diagnostics);
        t.on_block(&block);
        assert_eq!(t.meter_count, 9600);
        assert_eq!(t.wave_len, 4800);
        match t.waveform() {
            TelemetryFrame::Waveform { min, max } => {
                assert_eq!(min.len(), WAVEFORM_COLUMNS);
                assert_eq!(max.len(), WAVEFORM_COLUMNS);
                assert!(min.iter().zip(&max).all(|(lo, hi)| lo <= hi));
            }
            other => panic!("expected a waveform, got {other:?}"),
        }
    }
}
