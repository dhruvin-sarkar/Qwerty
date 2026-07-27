//! The Evaluation screen (`DESIGN.md` → Evaluation; `ROADMAP.md` Phase 6).
//!
//! A guided held-out accuracy test: for each zone the user is asked to tap the
//! same spot N times (default 15, *separate* from the calibration data), every
//! accepted tap is classified against the profile's already-fitted classifier,
//! and the outcomes are folded into an [`EvaluationReport`] — confusion matrix,
//! per-zone and overall accuracy, rejected-tap count, confidence distribution,
//! and latency percentiles (`qwerty-core::evaluation`). The report is saved
//! per-profile and shown, and the profile's past reports are browsable as an
//! accuracy trend. This screen is the *only* place the app earns the right to
//! state an accuracy for a device/desk (`DESIGN.md` → honest-accuracy rule);
//! a calibrated-but-never-evaluated profile says exactly that.
//!
//! Unlike the calibration wizard this does **not** take over the window — it
//! lives in the content area with the nav rail visible. It owns its own
//! [`LiveCapture`] while a run is active (the shell suppresses Home listening
//! and drops the run when the user navigates away, so the mic and the live
//! repaint both stop — `PERFORMANCE.md`).

use std::path::PathBuf;
use std::time::Duration;

use chrono::Utc;

use qwerty_core::classifier::ClassifierParams;
use qwerty_core::evaluation::{EvaluationAccumulator, EvaluationReport, DEFAULT_EVAL_TAPS_PER_ZONE};
use qwerty_core::features::{FeatureExtractor, FEATURE_WINDOW_SAMPLES, SAMPLE_RATE_HZ};
use qwerty_core::profile::{Profile, ProfileId, ZoneId};

use crate::app_state::AppState;
use crate::capture_worker::{CaptureDetail, CaptureEvent, LiveCapture};
use crate::ui::shell::{mix, section, Palette};

/// Configurable held-out tap counts (`DESIGN.md`: "N taps per zone (default 15,
/// configurable)"). 15 is [`DEFAULT_EVAL_TAPS_PER_ZONE`].
const TAP_COUNT_CHOICES: [u32; 4] = [10, 15, 20, 25];

/// The acceptance target this project holds itself to (`ACCEPTANCE.md`: ≥92%
/// overall accuracy). Rendered as a reference line and a pass/needs-work verdict
/// — never a claim, always measured against the saved report.
const TARGET_ACCURACY: f32 = 0.92;

/// The irreducible acoustic-window latency: the classifier cannot decide until
/// the full 90 ms window after the transient is captured. Added to the measured
/// host-side processing time (transport + feature extraction + classify), this
/// is the honest end-to-end latency the report records. It excludes the sub-hop
/// cpal input-buffer latency and the ≤15 ms worker poll interval (a small, known
/// under-count, documented so no one mistakes this for a hardware-exact figure).
const WINDOW_FILL_MS: f32 = FEATURE_WINDOW_SAMPLES as f32 / SAMPLE_RATE_HZ as f32 * 1000.0;

/// One recorded held-out tap, buffered in the run (not the pure accumulator) so
/// the last one can be undone before the report is built at finish.
struct TapRecord {
    zone_idx: usize,
    predicted: Option<ZoneId>,
    confidence: f32,
    latency_ms: f32,
}

/// Immediate feedback for the last tap (`DESIGN.md`: per-tap feedback, like the
/// wizard's capture step, but here it also names the predicted zone so a
/// misclassification is visible as it happens — the detective-method view).
struct LastTap {
    predicted_label: Option<String>,
    correct: bool,
    accepted: bool,
}

/// An in-progress guided evaluation. Owns the mic for its lifetime; dropping it
/// stops capture (RAII), which the shell does on navigate-away.
struct EvalRun {
    capture: LiveCapture,
    extractor: FeatureExtractor,
    /// A clone of the profile's fitted classifier — the run classifies against
    /// it read-only and never retrains (a held-out test must use the same model
    /// production would). Cloned so the run does not borrow `AppState`.
    classifier: ClassifierParams,
    profile_id: ProfileId,
    zone_ids: Vec<ZoneId>,
    zone_labels: Vec<String>,
    taps_per_zone: u32,
    records: Vec<TapRecord>,
    current_zone: usize,
    armed: bool,
    last: Option<LastTap>,
    device_name: Option<String>,
    capture_error: Option<String>,
}

impl EvalRun {
    fn new(ctx: &egui::Context, profile: &Profile, taps_per_zone: u32) -> Self {
        Self {
            capture: LiveCapture::start(None, ctx.clone(), CaptureDetail::Standard),
            extractor: FeatureExtractor::new(),
            classifier: profile.classifier.clone(),
            profile_id: profile.id,
            zone_ids: profile.zones.iter().map(|z| z.id).collect(),
            zone_labels: profile.zones.iter().map(|z| z.label.clone()).collect(),
            taps_per_zone,
            records: Vec::new(),
            current_zone: 0,
            armed: false,
            last: None,
            device_name: None,
            capture_error: None,
        }
    }

    /// Accepted taps recorded so far for a given zone index.
    fn count_for(&self, zone_idx: usize) -> u32 {
        self.records.iter().filter(|r| r.zone_idx == zone_idx).count() as u32
    }

    fn current_count(&self) -> u32 {
        self.count_for(self.current_zone)
    }

    fn current_zone_complete(&self) -> bool {
        self.current_count() >= self.taps_per_zone
    }

    fn is_last_zone(&self) -> bool {
        self.current_zone + 1 >= self.zone_ids.len()
    }

    fn all_zones_complete(&self) -> bool {
        (0..self.zone_ids.len()).all(|z| self.count_for(z) >= self.taps_per_zone)
    }

    /// Absorb capture events; classify each armed, transient tap for the current
    /// zone. A non-transient onset is not a tap attempt and is ignored (same as
    /// the wizard's capture step) — the novelty gate still turns a genuine but
    /// unrecognized tap into a rejection, which the report counts against the
    /// zone's accuracy.
    fn drain(&mut self) {
        for ev in self.capture.poll() {
            match ev {
                CaptureEvent::Started { device_name, .. } => self.device_name = Some(device_name),
                CaptureEvent::Level { .. } => {}
                CaptureEvent::Onset {
                    window,
                    is_transient,
                    detected_at,
                } => {
                    if !self.armed
                        || !is_transient
                        || self.current_zone_complete()
                        || window.len() != FEATURE_WINDOW_SAMPLES
                    {
                        continue;
                    }
                    let fv = self.extractor.extract(&window);
                    let outcome = self.classifier.classify(&fv);
                    let latency_ms = WINDOW_FILL_MS + detected_at.elapsed().as_secs_f32() * 1000.0;
                    let predicted = if outcome.accepted {
                        outcome.zone_id
                    } else {
                        None
                    };
                    let correct = predicted == Some(self.zone_ids[self.current_zone]);
                    self.last = Some(LastTap {
                        predicted_label: predicted
                            .and_then(|p| self.zone_ids.iter().position(|z| *z == p))
                            .map(|i| self.zone_labels[i].clone()),
                        correct,
                        accepted: outcome.accepted,
                    });
                    self.records.push(TapRecord {
                        zone_idx: self.current_zone,
                        predicted,
                        confidence: outcome.confidence,
                        latency_ms,
                    });
                    if self.current_zone_complete() {
                        self.armed = false;
                    }
                }
                // Evaluation requests `CaptureDetail::Standard`; display frames
                // are a Diagnostics-only concern.
                CaptureEvent::Frame { .. } => {}
                CaptureEvent::Failed(msg) => self.capture_error = Some(msg),
            }
        }
    }

    /// Remove the most recent tap recorded for the current zone (`DESIGN.md`
    /// per-tap correction — a fumbled tap should not corrupt the report).
    fn undo_last(&mut self) {
        if let Some(pos) = self
            .records
            .iter()
            .rposition(|r| r.zone_idx == self.current_zone)
        {
            self.records.remove(pos);
            self.last = None;
        }
    }

    fn advance_zone(&mut self) {
        if !self.is_last_zone() {
            self.current_zone += 1;
            self.armed = false;
            self.last = None;
        }
    }

    /// Build the report from the buffered records (replays them through the pure
    /// accumulator all at once) plus the per-tap latencies for the live view and
    /// the zone id/label order for rendering.
    fn finish(&self) -> FinishedRun {
        let mut acc = EvaluationAccumulator::new(self.zone_ids.clone(), self.taps_per_zone);
        for r in &self.records {
            acc.record(self.zone_ids[r.zone_idx], r.predicted, r.confidence, r.latency_ms);
        }
        FinishedRun {
            report: acc.finish(self.profile_id, Utc::now()),
            latencies_ms: self.records.iter().map(|r| r.latency_ms).collect(),
            zone_ids: self.zone_ids.clone(),
            zone_labels: self.zone_labels.clone(),
            saved: None,
        }
    }
}

/// A finished run's report plus what it needs to render and its save outcome.
/// `zone_ids`/`zone_labels` are in confusion-matrix order (`zones.order()`), so
/// the per-zone accuracy map is looked up by id without leaking the classifier's
/// private ordering (`DATA_MODEL.md` encapsulation boundary).
struct FinishedRun {
    report: EvaluationReport,
    latencies_ms: Vec<f32>,
    zone_ids: Vec<ZoneId>,
    zone_labels: Vec<String>,
    /// `None` until persisted on the first result frame; then the save outcome.
    saved: Option<Result<PathBuf, String>>,
}

/// The persistent Evaluation-screen state (held by the shell across frames, like
/// the Zones action editor). Owns the optional in-progress run and the last
/// finished result, and caches the profile's saved-report history for the trend.
#[derive(Default)]
pub struct EvaluationScreen {
    taps_per_zone: u32,
    run: Option<EvalRun>,
    result: Option<FinishedRun>,
    history: Vec<EvaluationReport>,
    history_for: Option<ProfileId>,
    /// Set when the user asks to start a run; the *shell* acts on it at end of
    /// frame. The run is deliberately not created here: the shell must close the
    /// Home microphone first, so the device is never held by two capture streams
    /// at once. Same deferred-launch pattern as the calibration wizard.
    pending_start: Option<u32>,
}

impl EvaluationScreen {
    /// Whether a guided run is currently capturing (the shell checks this to
    /// suppress Home listening and to stop the run when navigating away).
    pub fn is_running(&self) -> bool {
        self.run.is_some()
    }

    /// Take a pending "start a run" request, if the user made one this frame.
    /// The shell calls this at end of frame, closes the Home mic, then calls
    /// [`begin_run`](Self::begin_run).
    pub fn take_pending_start(&mut self) -> Option<u32> {
        self.pending_start.take()
    }

    /// Actually start the guided run (opens this screen's own microphone). Only
    /// the shell calls this, and only after it has dropped the Home capture.
    pub fn begin_run(&mut self, ctx: &egui::Context, profile: &Profile, taps_per_zone: u32) {
        self.result = None;
        self.run = Some(EvalRun::new(ctx, profile, taps_per_zone));
    }

    /// Stop and discard any in-progress run (drops the mic). Called by the shell
    /// when the user navigates away from this screen.
    pub fn stop_run(&mut self) {
        self.run = None;
    }

    /// Render the screen. `state` is read-only here: the run holds its own clone
    /// of the classifier, and saving/listing reports are `&self` on `AppState`.
    pub fn ui(&mut self, ui: &mut egui::Ui, pal: &Palette, state: &AppState) {
        ui.heading(egui::RichText::new("Evaluation").color(pal.text).size(26.0));
        ui.add_space(4.0);

        let Some(profile) = state.active_profile.as_ref() else {
            ui.label(
                egui::RichText::new(
                    "Calibrate a profile first. Evaluation measures how accurately \
                     this exact desk and mic recognize each zone.",
                )
                .color(pal.secondary),
            );
            return;
        };
        let profile_id = profile.id;

        // (Re)load the saved-report history when the active profile changes.
        if self.history_for != Some(profile_id) {
            self.history = state.list_evaluation_reports(profile_id);
            self.history_for = Some(profile_id);
        }
        if self.taps_per_zone == 0 {
            self.taps_per_zone = DEFAULT_EVAL_TAPS_PER_ZONE;
        }

        if self.run.is_some() {
            self.running_ui(ui, pal);
        } else if self.result.is_some() {
            self.result_ui(ui, pal, state);
        } else {
            self.idle_ui(ui, pal, profile);
        }
    }

    /// Idle: choose taps/zone, start a run, and browse this profile's history.
    fn idle_ui(&mut self, ui: &mut egui::Ui, pal: &Palette, profile: &Profile) {
        ui.label(
            egui::RichText::new(format!(
                "A held-out test taps each of the {} zone{} several times — separate \
                 from the calibration taps — and reports the real accuracy.",
                profile.zones.len(),
                if profile.zones.len() == 1 { "" } else { "s" },
            ))
            .color(pal.secondary),
        );
        ui.add_space(12.0);

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Taps per zone:").color(pal.text));
            for n in TAP_COUNT_CHOICES {
                if ui
                    .selectable_label(self.taps_per_zone == n, n.to_string())
                    .clicked()
                {
                    self.taps_per_zone = n;
                }
            }
        });
        ui.add_space(14.0);

        if ui.button("Run evaluation").clicked() {
            // Request only — the shell starts the run at end of frame, after it
            // has closed the Home microphone (see `pending_start`).
            self.pending_start = Some(self.taps_per_zone.max(1));
            ui.ctx().request_repaint();
        }

        ui.add_space(20.0);
        self.history_ui(ui, pal);
    }

    /// The saved-report history + accuracy trend for the active profile.
    fn history_ui(&self, ui: &mut egui::Ui, pal: &Palette) {
        section(ui, pal, "History");
        if self.history.is_empty() {
            ui.label(
                egui::RichText::new(
                    "No evaluations yet. This profile has not earned an accuracy \
                     figure — run one to produce a saved report.",
                )
                .color(pal.secondary)
                .italics(),
            );
            return;
        }

        draw_trend(ui, pal, &self.history);
        ui.add_space(8.0);
        egui::ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
            // Most recent first.
            for r in self.history.iter().rev() {
                let pass = r.overall_accuracy >= TARGET_ACCURACY;
                let (glyph, color) = if pass {
                    ("✓", pal.success)
                } else {
                    ("▲", pal.warning)
                };
                ui.label(
                    egui::RichText::new(format!(
                        "{glyph}  {}   ·   {:.1}% overall   ·   {} taps/zone",
                        r.run_at.format("%Y-%m-%d %H:%M UTC"),
                        r.overall_accuracy * 100.0,
                        r.taps_per_zone,
                    ))
                    .color(color),
                );
            }
        });
    }

    /// A running guided test: prompt the current zone, show progress + per-tap
    /// feedback, and arm/advance/finish controls. Repaints at a bounded 30 fps
    /// while active (live meter) and stops the instant the run ends.
    fn running_ui(&mut self, ui: &mut egui::Ui, pal: &Palette) {
        let mut finish = false;
        let mut cancel = false;
        {
            let run = self.run.as_mut().expect("run present");
            run.drain();

            if let Some(err) = &run.capture_error {
                ui.label(egui::RichText::new(format!("⚠ Microphone: {err}")).color(pal.danger));
                ui.add_space(8.0);
            }

            let zone = run.current_zone;
            let total = run.zone_ids.len();
            let count = run.current_count();
            let target = run.taps_per_zone;
            let label = run.zone_labels.get(zone).cloned().unwrap_or_default();
            let complete = run.current_zone_complete();
            let is_last = run.is_last_zone();
            let all_done = run.all_zones_complete();

            ui.label(
                egui::RichText::new(format!(
                    "Zone {} of {total}: “{label}”. Tap this spot {target} times.",
                    zone + 1
                ))
                .color(pal.text),
            );
            ui.add_space(8.0);

            // Progress pips (color + shape, never color alone — DESIGN.md a11y).
            ui.horizontal_wrapped(|ui| {
                for i in 0..target {
                    let (glyph, color) = if i < count {
                        ("●", pal.success)
                    } else {
                        ("○", pal.secondary)
                    };
                    ui.label(egui::RichText::new(glyph).color(color).size(18.0));
                }
                ui.add_space(8.0);
                ui.label(egui::RichText::new(format!("{count}/{target}")).color(pal.secondary));
            });
            ui.add_space(8.0);

            // Last-tap feedback: names the predicted zone so a miss is visible.
            match &run.last {
                Some(t) if !t.accepted => {
                    ui.label(
                        egui::RichText::new("rejected — not recognized as any zone")
                            .color(pal.warning),
                    );
                }
                Some(t) if t.correct => {
                    ui.label(egui::RichText::new("✓ correct").color(pal.success));
                }
                Some(t) => {
                    let got = t.predicted_label.as_deref().unwrap_or("another zone");
                    ui.label(
                        egui::RichText::new(format!("✗ misclassified as “{got}”"))
                            .color(pal.danger),
                    );
                }
                None => {
                    ui.label(
                        egui::RichText::new("Arm the zone, then tap.")
                            .color(pal.secondary)
                            .small(),
                    );
                }
            }
            ui.add_space(12.0);

            ui.horizontal(|ui| {
                let arm_label = if run.armed { "Stop" } else { "Arm zone" };
                if ui
                    .add_enabled(!complete, egui::Button::new(arm_label))
                    .clicked()
                {
                    run.armed = !run.armed;
                }
                if ui.button("Undo last").clicked() {
                    run.undo_last();
                }
            });
            ui.add_space(10.0);

            ui.horizontal(|ui| {
                if !is_last
                    && ui
                        .add_enabled(complete, egui::Button::new("Next zone"))
                        .clicked()
                {
                    run.advance_zone();
                }
                if ui
                    .add_enabled(all_done, egui::Button::new("Finish & save report"))
                    .clicked()
                {
                    finish = true;
                }
                if ui.button("Cancel").clicked() {
                    cancel = true;
                }
            });
        } // `run` borrow ends here, so the transitions below can touch `self`.

        if finish {
            if let Some(run) = self.run.take() {
                self.result = Some(run.finish());
            }
        } else if cancel {
            self.run = None;
        }

        if finish || cancel {
            // Render the result/idle view on the very next frame.
            ui.ctx().request_repaint();
        } else if self.run.is_some() {
            // Bounded-rate repaint only while the run is live (MOTION.md: 30 fps
            // is enough for a live meter; PERFORMANCE.md: no free-running loop).
            ui.ctx().request_repaint_after(Duration::from_millis(33));
        }
    }

    /// Render a completed run's report and persist it once (on first entry).
    fn result_ui(&mut self, ui: &mut egui::Ui, pal: &Palette, state: &AppState) {
        // Persist on the first frame after finishing (we now hold `state`).
        if let Some(res) = &mut self.result {
            if res.saved.is_none() {
                res.saved = Some(state.save_evaluation_report(&res.report));
                // Force a history refresh so this run appears next idle view.
                self.history_for = None;
            }
        }
        let res = self.result.as_ref().expect("result present");
        let report = &res.report;

        // Headline accuracy + honest pass/needs-work verdict.
        let pass = report.overall_accuracy >= TARGET_ACCURACY;
        let (glyph, color, verdict) = if pass {
            ("✓", pal.success, "meets the ≥92% target")
        } else {
            ("▲", pal.warning, "below the 92% target")
        };
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!("{:.1}%", report.overall_accuracy * 100.0))
                    .color(color)
                    .size(34.0)
                    .strong(),
            );
            ui.add_space(10.0);
            ui.label(egui::RichText::new(format!("{glyph} overall — {verdict}")).color(color));
        });
        ui.add_space(6.0);

        // A failed save must not cost the user the whole run: the computed
        // report is still in memory, so offer a retry rather than forcing a
        // repeat of the entire guided test.
        let mut retry_save = false;
        match &res.saved {
            Some(Ok(path)) => {
                ui.label(
                    egui::RichText::new(format!("Saved report: {}", path.display()))
                        .color(pal.secondary)
                        .small(),
                );
            }
            Some(Err(e)) => {
                ui.label(
                    egui::RichText::new(format!("⚠ Could not save report: {e}")).color(pal.danger),
                );
                ui.horizontal(|ui| {
                    if ui.button("Retry save").clicked() {
                        retry_save = true;
                    }
                    ui.label(
                        egui::RichText::new(
                            "The results below are still in memory — retrying will not \
                             require redoing the test.",
                        )
                        .color(pal.secondary)
                        .small(),
                    );
                });
            }
            None => {
                ui.label(egui::RichText::new("Saving…").color(pal.secondary).small());
            }
        }
        ui.add_space(14.0);

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Per-zone accuracy (accounts for rejections — DATA_MODEL.md), looked
            // up by the ordered zone ids captured at run time.
            section(ui, pal, "Per-zone accuracy");
            for (i, label) in res.zone_labels.iter().enumerate() {
                let acc = res
                    .zone_ids
                    .get(i)
                    .and_then(|z| report.per_zone_accuracy.get(z).copied())
                    .unwrap_or(0.0);
                bar_row(ui, pal, label, acc);
            }
            ui.add_space(14.0);

            section(ui, pal, "Confusion matrix");
            ui.label(
                egui::RichText::new(
                    "Rows = the zone tapped, columns = the zone predicted. A strong \
                     diagonal is good; off-diagonal cells are confusions.",
                )
                .color(pal.secondary)
                .small(),
            );
            ui.add_space(6.0);
            draw_confusion_matrix(ui, pal, &report.confusion_matrix, &res.zone_labels);
            ui.add_space(6.0);
            ui.label(
                egui::RichText::new(format!(
                    "Rejected taps (heard, but recognized as no zone): {}",
                    report.rejected_tap_count
                ))
                .color(if report.rejected_tap_count > 0 {
                    pal.warning
                } else {
                    pal.secondary
                }),
            );
            ui.add_space(14.0);

            section(ui, pal, "Latency");
            let l = &report.latency_ms;
            let lat_ok = l.p50 < 120.0 && l.p95 < 200.0;
            ui.label(
                egui::RichText::new(format!(
                    "p50 {:.0} ms   ·   p95 {:.0} ms   ·   p99 {:.0} ms   ·   max {:.0} ms",
                    l.p50, l.p95, l.p99, l.max
                ))
                .color(if lat_ok { pal.success } else { pal.warning }),
            );
            ui.label(
                egui::RichText::new("Target: median < 120 ms, p95 < 200 ms (ACCEPTANCE.md).")
                    .color(pal.secondary)
                    .small(),
            );
            ui.add_space(6.0);
            draw_latency_histogram(ui, pal, &res.latencies_ms);
            ui.add_space(14.0);

            section(ui, pal, "Confidence distribution");
            draw_confidence_histogram(ui, pal, report);
        });

        ui.add_space(14.0);
        let run_again = ui.button("Run another evaluation").clicked();

        // Apply the transitions after every read of `res` is done.
        if retry_save {
            if let Some(r) = &mut self.result {
                r.saved = None; // re-attempts the save on the next frame
            }
            ui.ctx().request_repaint();
        } else if run_again {
            self.result = None;
            ui.ctx().request_repaint();
        }
    }
}

/// A labeled horizontal accuracy bar in `[0, 1]`.
fn bar_row(ui: &mut egui::Ui, pal: &Palette, label: &str, frac: f32) {
    let frac = frac.clamp(0.0, 1.0);
    let color = if frac >= TARGET_ACCURACY {
        pal.success
    } else if frac >= 0.75 {
        pal.warning
    } else {
        pal.danger
    };
    ui.horizontal(|ui| {
        let (resp, painter) = ui.allocate_painter(egui::vec2(220.0, 18.0), egui::Sense::hover());
        let r = resp.rect;
        painter.rect_filled(r, egui::CornerRadius::same(4), pal.surface);
        let filled = egui::Rect::from_min_size(r.min, egui::vec2(r.width() * frac, r.height()));
        painter.rect_filled(filled, egui::CornerRadius::same(4), color);
        ui.label(egui::RichText::new(format!("{:.0}%  {label}", frac * 100.0)).color(pal.text));
    });
}

/// Draw the N×N confusion matrix as a heat grid with 1-based numeric headers and
/// a numbered legend below (rotated text is avoided — numbers keep it legible at
/// any zone count). Cell intensity is the fraction of that actual zone's
/// *accepted* taps that landed in the cell.
fn draw_confusion_matrix(ui: &mut egui::Ui, pal: &Palette, matrix: &[Vec<u32>], labels: &[String]) {
    let n = matrix.len();
    if n == 0 {
        return;
    }
    let cell = 40.0f32;
    let gutter = 26.0f32; // row/column header band
    let size = egui::vec2(gutter + n as f32 * cell, gutter + n as f32 * cell);
    let (resp, painter) = ui.allocate_painter(size, egui::Sense::hover());
    let origin = resp.rect.min;

    // Iterate by value (not index) so clippy is happy; `i`/`j` are used only for
    // cell geometry and header numbers, never to index `matrix` (it is square).
    for (i, row) in matrix.iter().enumerate() {
        // Row header (actual zone number).
        painter.text(
            egui::pos2(
                origin.x + gutter * 0.5,
                origin.y + gutter + i as f32 * cell + cell * 0.5,
            ),
            egui::Align2::CENTER_CENTER,
            format!("{}", i + 1),
            egui::FontId::monospace(12.0),
            pal.secondary,
        );
        // Column header (predicted zone number).
        painter.text(
            egui::pos2(
                origin.x + gutter + i as f32 * cell + cell * 0.5,
                origin.y + gutter * 0.5,
            ),
            egui::Align2::CENTER_CENTER,
            format!("{}", i + 1),
            egui::FontId::monospace(12.0),
            pal.secondary,
        );

        let row_sum: u32 = row.iter().sum();
        for (j, &count) in row.iter().enumerate() {
            let intensity = if row_sum > 0 {
                count as f32 / row_sum as f32
            } else {
                0.0
            };
            let base = if i == j { pal.success } else { pal.danger };
            let fill = mix(pal.surface, base, intensity);
            let rect = egui::Rect::from_min_size(
                egui::pos2(
                    origin.x + gutter + j as f32 * cell,
                    origin.y + gutter + i as f32 * cell,
                ),
                egui::vec2(cell, cell),
            )
            .shrink(1.0);
            painter.rect_filled(rect, egui::CornerRadius::same(3), fill);
            if count > 0 {
                let text_color = if intensity > 0.5 { pal.base } else { pal.text };
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    count.to_string(),
                    egui::FontId::monospace(13.0),
                    text_color,
                );
            }
        }
    }

    // Numbered legend mapping the header numbers to zone labels.
    ui.add_space(4.0);
    for (i, label) in labels.iter().enumerate() {
        ui.label(
            egui::RichText::new(format!("{} = {label}", i + 1))
                .color(pal.secondary)
                .small(),
        );
    }
}

/// Vertical-bar histogram of the confidence distribution (10 buckets over
/// `[0, 1]`, accepted taps only — `qwerty-core::evaluation`).
fn draw_confidence_histogram(ui: &mut egui::Ui, pal: &Palette, report: &EvaluationReport) {
    let h = &report.confidence_distribution;
    let max = h.bins.iter().copied().max().unwrap_or(0).max(1);
    let n = h.bins.len().max(1);
    let width = (n as f32 * 30.0).min(360.0);
    let (resp, painter) = ui.allocate_painter(egui::vec2(width, 90.0), egui::Sense::hover());
    let r = resp.rect;
    let bw = r.width() / n as f32;
    for (i, &c) in h.bins.iter().enumerate() {
        let frac = c as f32 / max as f32;
        let bh = r.height() * frac;
        let bar = egui::Rect::from_min_size(
            egui::pos2(r.min.x + i as f32 * bw, r.max.y - bh),
            egui::vec2(bw - 2.0, bh),
        );
        painter.rect_filled(bar, egui::CornerRadius::same(2), pal.accent);
    }
    ui.label(
        egui::RichText::new("left = low confidence, right = high (accepted taps).")
            .color(pal.secondary)
            .small(),
    );
}

/// Vertical-bar histogram of per-tap latencies (live-only view; the report
/// persists only percentiles — DATA_MODEL.md). Buckets span 0..250 ms.
fn draw_latency_histogram(ui: &mut egui::Ui, pal: &Palette, latencies_ms: &[f32]) {
    const BUCKETS: usize = 10;
    const RANGE_MS: f32 = 250.0;
    if latencies_ms.is_empty() {
        return;
    }
    let mut bins = [0u32; BUCKETS];
    for &l in latencies_ms {
        let frac = (l / RANGE_MS).clamp(0.0, 1.0);
        let idx = ((frac * BUCKETS as f32) as usize).min(BUCKETS - 1);
        bins[idx] += 1;
    }
    let max = bins.iter().copied().max().unwrap_or(1).max(1);
    let (resp, painter) = ui.allocate_painter(egui::vec2(300.0, 80.0), egui::Sense::hover());
    let r = resp.rect;
    let bw = r.width() / BUCKETS as f32;
    for (i, &c) in bins.iter().enumerate() {
        let bh = r.height() * (c as f32 / max as f32);
        let bar = egui::Rect::from_min_size(
            egui::pos2(r.min.x + i as f32 * bw, r.max.y - bh),
            egui::vec2(bw - 2.0, bh),
        );
        painter.rect_filled(bar, egui::CornerRadius::same(2), pal.accent);
    }
    ui.label(
        egui::RichText::new(format!("0 to {RANGE_MS:.0} ms per tap"))
            .color(pal.secondary)
            .small(),
    );
}

/// Draw the overall-accuracy trend across saved reports (oldest→newest), with a
/// reference line at the 92% target (`DESIGN.md`: "viewable as a trend over
/// time").
fn draw_trend(ui: &mut egui::Ui, pal: &Palette, history: &[EvaluationReport]) {
    if history.is_empty() {
        return;
    }
    let (resp, painter) = ui.allocate_painter(egui::vec2(360.0, 90.0), egui::Sense::hover());
    let r = resp.rect;
    painter.rect_filled(r, egui::CornerRadius::same(4), pal.surface);

    // Map accuracy in [0.5, 1.0] to the plot height (below 50% pins to the floor).
    let y_of = |acc: f32| {
        let a = ((acc - 0.5) / 0.5).clamp(0.0, 1.0);
        r.max.y - a * r.height()
    };

    // Target reference line at 92%.
    let ty = y_of(TARGET_ACCURACY);
    painter.line_segment(
        [egui::pos2(r.min.x, ty), egui::pos2(r.max.x, ty)],
        egui::Stroke::new(1.0_f32, pal.border),
    );

    let n = history.len();
    let x_of = |i: usize| {
        if n <= 1 {
            r.center().x
        } else {
            r.min.x + r.width() * i as f32 / (n - 1) as f32
        }
    };
    let pts: Vec<egui::Pos2> = history
        .iter()
        .enumerate()
        .map(|(i, rep)| egui::pos2(x_of(i), y_of(rep.overall_accuracy)))
        .collect();
    if pts.len() >= 2 {
        painter.add(egui::Shape::line(pts, egui::Stroke::new(2.0_f32, pal.accent)));
    }
    for (i, rep) in history.iter().enumerate() {
        let color = if rep.overall_accuracy >= TARGET_ACCURACY {
            pal.success
        } else {
            pal.warning
        };
        painter.circle_filled(egui::pos2(x_of(i), y_of(rep.overall_accuracy)), 3.0, color);
    }
    ui.label(
        egui::RichText::new("Overall accuracy over time (line at 92% target).")
            .color(pal.secondary)
            .small(),
    );
}

