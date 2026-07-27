//! The Diagnostics screen (`DESIGN.md` → Diagnostics; `ROADMAP.md` Phase 6).
//!
//! Three live views over the real input:
//!
//! - a scrolling **waveform** (peak envelope of the recent input),
//! - a scrolling **spectrogram** built from the classifier's own normalized
//!   spectral bands — deliberately the model's view of the sound, not a
//!   prettier but unrelated STFT,
//! - a **feature-space** view of the most recent tap: how far it sat from every
//!   calibrated zone, and where it fell relative to the novelty gate.
//!
//! That last one is the point of the screen (`DESIGN.md`): "a user or agent can
//! *see* why a tap was misclassified instead of guessing" — the detective method
//! made visible. Two zones at nearly equal distance explains a confusion; a
//! nearest-example distance past the threshold explains a rejection.
//!
//! Redraw discipline (`PERFORMANCE.md`, `MOTION.md`): this is the one screen
//! allowed a continuous loop, and only while it is actually visible. The worker
//! emits frames at a fixed 30 fps rather than on every audio callback, and the
//! shell drops this screen's capture the moment the user navigates away, which
//! stops both the microphone and the repaints within a frame.

use std::collections::VecDeque;
use std::time::Duration;

use qwerty_core::classifier::FeatureSpaceView;
use qwerty_core::features::{FeatureExtractor, FEATURE_WINDOW_SAMPLES, SPECTRAL_BAND_COUNT};
use qwerty_core::profile::{Profile, SensingMode};

use crate::app_state::AppState;
use crate::capture_worker::{CaptureDetail, CaptureEvent, LiveCapture};
use crate::ui::shell::{mix, section, Palette};
use crate::ui::style::{space, text};

/// Spectrogram scrollback, in columns. At the worker's ~33 fps frame rate this
/// is a few seconds of history — enough to see a tap arrive and decay.
const SPECTROGRAM_COLUMNS: usize = 90;

/// The most recent tap, projected against the calibrated model.
struct TapView {
    space: FeatureSpaceView,
    /// Whether the onset detector judged it a tap-like transient at all.
    is_transient: bool,
}

/// Live diagnostics state. Owns its microphone only while the screen is on
/// screen; the shell calls [`start`](Self::start) / [`stop`](Self::stop).
#[derive(Default)]
pub struct DiagnosticsScreen {
    capture: Option<LiveCapture>,
    /// Built alongside the capture so an unopened screen never plans an FFT.
    extractor: Option<FeatureExtractor>,
    device_name: Option<String>,
    error: Option<String>,
    waveform: Vec<f32>,
    spectrogram: VecDeque<[f32; SPECTRAL_BAND_COUNT]>,
    last_tap: Option<TapView>,
}

impl DiagnosticsScreen {
    /// Whether the shell should (re)start capture: no live stream *and* no
    /// recorded failure. A failure leaves `error` set and `capture` cleared, so
    /// this returns `false` — the shell must not respawn a device that just
    /// failed on a loop; the error stays on screen until the user navigates away
    /// (which calls [`stop`](Self::stop) and resets it).
    pub fn needs_start(&self) -> bool {
        self.capture.is_none() && self.error.is_none()
    }

    /// Begin live capture. The shell calls this *after* closing the Home
    /// microphone, so the device is never held by two streams at once.
    pub fn start(&mut self, ctx: &egui::Context) {
        self.error = None;
        self.extractor = Some(FeatureExtractor::new());
        self.capture = Some(LiveCapture::start(
            None,
            ctx.clone(),
            CaptureDetail::Diagnostics,
        ));
    }

    /// Stop capturing and discard the live traces (dropping the capture closes
    /// the device). Called by the shell when the user navigates away, so neither
    /// the microphone nor the 30 fps repaint continues in the background.
    pub fn stop(&mut self) {
        self.capture = None;
        self.extractor = None;
        self.waveform.clear();
        self.spectrogram.clear();
        self.last_tap = None;
        self.device_name = None;
        // Clear the recorded failure too, so a later visit gets a fresh retry
        // via `needs_start()`. Without this the screen is locked out for the rest
        // of the session after one mic failure — even once the device recovers —
        // because `needs_start()` requires `error.is_none()`. (Retry-loop safety
        // is unaffected: `stop()` runs only on navigate-away/hide, not per frame.)
        self.error = None;
    }

    /// Absorb worker events: display frames drive the waveform/spectrogram, and
    /// each onset is projected against the active profile's classifier.
    fn drain(&mut self, profile: Option<&Profile>) {
        let Some(capture) = &self.capture else {
            return;
        };
        for ev in capture.poll() {
            match ev {
                CaptureEvent::Started { device_name, .. } => self.device_name = Some(device_name),
                CaptureEvent::Level { .. } => {}
                CaptureEvent::Frame { waveform, bands } => {
                    self.waveform = waveform;
                    self.spectrogram.push_back(bands);
                    while self.spectrogram.len() > SPECTROGRAM_COLUMNS {
                        self.spectrogram.pop_front();
                    }
                }
                CaptureEvent::Onset {
                    window,
                    is_transient,
                    ..
                } => {
                    // Projecting needs both a calibrated profile and the
                    // extractor that only exists while capturing.
                    let (Some(profile), Some(extractor)) = (profile, self.extractor.as_mut())
                    else {
                        continue;
                    };
                    if window.len() != FEATURE_WINDOW_SAMPLES {
                        continue;
                    }
                    let fv = extractor.extract(&window);
                    self.last_tap = Some(TapView {
                        space: profile.classifier.feature_space(&fv),
                        is_transient,
                    });
                }
                CaptureEvent::Failed(msg) => {
                    self.error = Some(msg);
                    self.capture = None;
                }
            }
        }
    }

    /// Render the screen.
    pub fn ui(&mut self, ui: &mut egui::Ui, pal: &Palette, state: &AppState) {
        ui.heading(
            egui::RichText::new("Diagnostics")
                .color(pal.text)
                .size(text::TITLE),
        );
        ui.add_space(space::XS);

        let profile = state.active_profile.as_ref();
        self.drain(profile);

        if let Some(err) = &self.error {
            ui.label(egui::RichText::new(format!("⚠ Microphone: {err}")).color(pal.danger));
            ui.add_space(space::SM);
        } else {
            let mic = self.device_name.as_deref().unwrap_or("opening microphone…");
            ui.label(
                egui::RichText::new(format!("Listening on {mic}"))
                    .color(pal.secondary)
                    .small(),
            );
        }
        ui.add_space(space::MD);

        egui::ScrollArea::vertical().show(ui, |ui| {
            section(ui, pal, "Waveform");
            draw_waveform(ui, pal, &self.waveform);
            ui.add_space(space::LG);

            section(ui, pal, "Spectrogram");
            ui.label(
                egui::RichText::new(
                    "The classifier's own frequency bands over time — low bands at the \
                     bottom, newest on the right.",
                )
                .color(pal.secondary)
                .small(),
            );
            ui.add_space(space::XS);
            draw_spectrogram(ui, pal, &self.spectrogram);
            ui.add_space(space::LG);

            section(ui, pal, "Feature space — last tap");
            match (profile, &self.last_tap) {
                (None, _) => {
                    ui.label(
                        egui::RichText::new(
                            "Calibrate a profile to see how a tap compares with each zone.",
                        )
                        .color(pal.secondary),
                    );
                }
                (Some(_), None) => {
                    ui.label(
                        egui::RichText::new("Tap the desk to see where the sound lands.")
                            .color(pal.secondary),
                    );
                }
                (Some(profile), Some(tap)) => draw_feature_space(ui, pal, profile, tap),
            }
            ui.add_space(space::LG);

            section(ui, pal, "Sensing mode");
            draw_sensing_mode(ui, pal, profile);
        });

        // The one sanctioned continuous view — and only while it is on screen.
        // The worker already rate-limits frames to 30 fps; this simply keeps the
        // UI awake to consume them, and stops as soon as the shell drops the
        // capture on navigate-away (`PERFORMANCE.md`).
        if self.capture.is_some() {
            ui.ctx().request_repaint_after(Duration::from_millis(33));
        }
    }
}

/// Draw the peak envelope symmetrically about a centre line, which reads as a
/// conventional waveform. The envelope (rather than stride-sampled points) means
/// a brief transient can never be skipped between pixels.
fn draw_waveform(ui: &mut egui::Ui, pal: &Palette, envelope: &[f32]) {
    let size = egui::vec2(ui.available_width().min(560.0), 90.0);
    let (resp, painter) = ui.allocate_painter(size, egui::Sense::hover());
    let r = resp.rect;
    painter.rect_filled(r, egui::CornerRadius::same(4), pal.surface);
    let mid = r.center().y;
    painter.line_segment(
        [egui::pos2(r.min.x, mid), egui::pos2(r.max.x, mid)],
        egui::Stroke::new(1.0_f32, pal.border),
    );
    if envelope.is_empty() {
        return;
    }
    let half = r.height() * 0.5 - 2.0;
    let step = r.width() / envelope.len() as f32;
    let mut shapes = Vec::with_capacity(envelope.len());
    for (i, &amp) in envelope.iter().enumerate() {
        // sqrt keeps quiet-but-real input visible without faking its level.
        let h = amp.clamp(0.0, 1.0).sqrt() * half;
        let x = r.min.x + i as f32 * step;
        shapes.push(egui::Shape::line_segment(
            [egui::pos2(x, mid - h), egui::pos2(x, mid + h)],
            egui::Stroke::new(1.0_f32, pal.accent),
        ));
    }
    painter.extend(shapes);
}

/// Draw the band-energy history as a heat grid: one column per frame, oldest at
/// the left, lowest band at the bottom.
fn draw_spectrogram(
    ui: &mut egui::Ui,
    pal: &Palette,
    history: &VecDeque<[f32; SPECTRAL_BAND_COUNT]>,
) {
    let size = egui::vec2(ui.available_width().min(560.0), 120.0);
    let (resp, painter) = ui.allocate_painter(size, egui::Sense::hover());
    let r = resp.rect;
    painter.rect_filled(r, egui::CornerRadius::same(4), pal.surface);
    let bands = SPECTRAL_BAND_COUNT;
    if history.is_empty() {
        return;
    }

    let col_w = r.width() / SPECTROGRAM_COLUMNS as f32;
    let row_h = r.height() / bands as f32;
    // Normalize against the loudest cell on screen so a quiet room still shows
    // structure rather than a uniformly flat panel.
    let peak = history
        .iter()
        .flat_map(|c| c.iter())
        .fold(0.0f32, |m, v| m.max(*v))
        .max(1e-6);

    let mut shapes = Vec::with_capacity(history.len() * bands);
    for (ci, column) in history.iter().enumerate() {
        // Newest column pinned to the right edge.
        let x = r.max.x - (history.len() - ci) as f32 * col_w;
        for (bi, &v) in column.iter().enumerate() {
            let intensity = (v / peak).clamp(0.0, 1.0).sqrt();
            if intensity <= 0.01 {
                continue;
            }
            // Low band at the bottom.
            let y = r.max.y - (bi + 1) as f32 * row_h;
            shapes.push(egui::Shape::rect_filled(
                egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(col_w, row_h)),
                egui::CornerRadius::ZERO,
                mix(pal.surface, pal.accent, intensity),
            ));
        }
    }
    painter.extend(shapes);
}

/// Show the last tap's distance to every zone, plus the novelty gate. Zones are
/// listed in the *profile's* order and matched by `ZoneId` — the classifier's
/// own zone ordering is private and never relied on here.
fn draw_feature_space(ui: &mut egui::Ui, pal: &Palette, profile: &Profile, tap: &TapView) {
    if !tap.is_transient {
        ui.label(
            egui::RichText::new(
                "▲ The last sound was not tap-like (it did not decay), so it was rejected \
                 before reaching the classifier.",
            )
            .color(pal.warning),
        );
        ui.add_space(space::SM);
    }

    // A non-finite input yields infinite distances from `feature_space` (its
    // documented defensive behaviour). Bail before the bar math, or `d / max_d`
    // would be inf/inf = NaN and paint garbage-width rects.
    if tap
        .space
        .zones
        .iter()
        .any(|z| !z.centroid_distance.is_finite())
        || !tap.space.nearest_example_distance.is_finite()
    {
        ui.label(
            egui::RichText::new(
                "The last tap's features were not finite (a glitchy audio frame); \
                 nothing to compare.",
            )
            .color(pal.warning),
        );
        return;
    }

    // Scale every bar against the same reference so the threshold marker below
    // is directly comparable with the per-zone distances.
    let max_d = tap
        .space
        .zones
        .iter()
        .map(|z| z.centroid_distance)
        .fold(0.0f32, f32::max)
        .max(tap.space.novelty_threshold)
        .max(1e-6);

    let winner = tap
        .space
        .zones
        .iter()
        .min_by(|a, b| a.centroid_distance.total_cmp(&b.centroid_distance))
        .map(|z| z.zone_id);

    for zone in &profile.zones {
        let Some(zd) = tap.space.zones.iter().find(|z| z.zone_id == zone.id) else {
            continue;
        };
        let is_winner = winner == Some(zone.id) && tap.space.accepted;
        let frac = (zd.centroid_distance / max_d).clamp(0.0, 1.0);
        ui.horizontal(|ui| {
            let (resp, painter) =
                ui.allocate_painter(egui::vec2(220.0, 16.0), egui::Sense::hover());
            let r = resp.rect;
            painter.rect_filled(r, egui::CornerRadius::same(3), pal.surface);
            // Shorter distance = closer = more likely, so fill the *remaining*
            // space: a long bar means "close to this zone".
            let w = r.width() * (1.0 - frac);
            painter.rect_filled(
                egui::Rect::from_min_size(r.min, egui::vec2(w, r.height())),
                egui::CornerRadius::same(3),
                if is_winner {
                    pal.success
                } else {
                    pal.secondary
                },
            );
            let marker = if is_winner { "●" } else { "○" };
            ui.label(
                egui::RichText::new(format!(
                    "{marker} {}  (distance {:.2})",
                    zone.label, zd.centroid_distance
                ))
                .color(if is_winner { pal.text } else { pal.secondary }),
            );
        });
    }

    ui.add_space(space::MD);
    let s = &tap.space;
    let (glyph, color, verdict) = if s.accepted {
        ("✓", pal.success, "accepted")
    } else {
        ("▲", pal.warning, "rejected as not matching any zone")
    };
    ui.label(
        egui::RichText::new(format!(
            "{glyph} Novelty gate: {verdict} — nearest calibrated example {:.2}, \
             threshold {:.2}",
            s.nearest_example_distance, s.novelty_threshold
        ))
        .color(color),
    );
    ui.label(
        egui::RichText::new(
            "Two zones at nearly equal distance is what a confusion looks like; a \
             nearest-example distance past the threshold is what a rejection looks like.",
        )
        .color(pal.secondary)
        .small(),
    );
}

/// The sensing-mode comparison (`DESIGN.md`: "A/B compare modes on their exact
/// desk before committing").
///
/// Only **Passive** sensing exists in this build. Active mode is defined as an
/// inaudible probe chirp plus response correlation, which needs an audio
/// *output* path and a correlation feature the DSP core does not have — no phase
/// has built it, and `ARCHITECTURE.md`'s threading model describes capture only.
/// So rather than render an A/B table whose Active and Hybrid columns could only
/// be invented, this states plainly what can and cannot be measured here.
/// Fabricating that comparison would break the very rule the Evaluation screen
/// exists to enforce: never imply a confidence the app has not earned.
fn draw_sensing_mode(ui: &mut egui::Ui, pal: &Palette, profile: Option<&Profile>) {
    match profile.map(|p| p.sensing_mode) {
        Some(mode) => ui.label(
            egui::RichText::new(format!(
                "This profile was calibrated with {} sensing.",
                mode_name(mode)
            ))
            .color(pal.text),
        ),
        None => ui.label(egui::RichText::new("No profile is active.").color(pal.secondary)),
    };
    ui.add_space(space::SM);

    for (mode, blurb, available) in [
        (
            SensingMode::Passive,
            "microphone only — listens for the sound the tap itself makes",
            true,
        ),
        (
            SensingMode::Active,
            "emits an inaudible probe tone and correlates the response",
            false,
        ),
        (
            SensingMode::Hybrid,
            "both passive listening and the active probe",
            false,
        ),
    ] {
        let (glyph, color, note) = if available {
            ("●", pal.success, "available")
        } else {
            ("○", pal.secondary, "not available in this build")
        };
        ui.label(
            egui::RichText::new(format!("{glyph} {} — {blurb} ({note})", mode_name(mode)))
                .color(color),
        );
    }

    ui.add_space(space::SM);
    ui.label(
        egui::RichText::new(
            "Active and Hybrid sensing need an inaudible probe-tone subsystem — audio \
             output plus response correlation — that this build does not contain. Showing \
             an A/B table would mean inventing their numbers, so Qwerty says so instead. \
             To measure Passive sensing on this exact desk and microphone, run an \
             evaluation.",
        )
        .color(pal.secondary)
        .small(),
    );
}

/// Display name for a sensing mode.
fn mode_name(mode: SensingMode) -> &'static str {
    match mode {
        SensingMode::Passive => "Passive",
        SensingMode::Active => "Active",
        SensingMode::Hybrid => "Hybrid",
    }
}
