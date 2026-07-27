//! The guided calibration wizard (`DESIGN.md` → Calibration flow; `ROADMAP.md`
//! Phase 4). A full-window, step-indicator-driven flow distinct from the main
//! navigation shell.
//!
//! It owns its own [`LiveCapture`] (the mic is on for the duration) and a
//! [`CalibrationSession`] from `qwerty-core`, which holds all the domain logic;
//! this module is only presentation + step flow. On save it hands the built
//! [`Profile`] back to the shell, which persists it.
//!
//! Steps: Environment → Layout → per-zone Capture → Consistency → optional
//! Negatives → Save. Zone layout is auto-arranged for now; a drag editor on the
//! desk diagram is a follow-up (the desk diagram itself is drawn here).

use chrono::Utc;

use qwerty_core::calibration::{
    assess_environment, CalibrationSession, ConsistencyReport, EnvironmentQuality, TapFeedback,
    ZoneMeta, DEFAULT_TAPS_PER_ZONE,
};
use qwerty_core::profile::{
    DeviceFingerprint, Profile, ProfileId, SensingMode, ZoneId, ZoneLayout,
};

use crate::capture_worker::{CaptureDetail, CaptureEvent, LiveCapture};
use crate::ui::shell::Palette;
use crate::ui::style::{space, text};

/// Supported zone counts (`DESIGN.md`: 2, 4, 6, 8).
const ZONE_COUNTS: [usize; 4] = [2, 4, 6, 8];

/// What the wizard hands back to the shell when it closes.
pub enum WizardOutcome {
    /// The user saved; the shell persists this profile and makes it active.
    Saved(Box<Profile>),
    /// The user cancelled; nothing is saved.
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    Environment,
    Layout,
    Capture,
    Consistency,
    Negatives,
    Save,
}

impl Step {
    /// (index, total) for the step indicator.
    fn position(self) -> (usize, usize) {
        let order = [
            Step::Environment,
            Step::Layout,
            Step::Capture,
            Step::Consistency,
            Step::Negatives,
            Step::Save,
        ];
        (
            order.iter().position(|s| *s == self).unwrap() + 1,
            order.len(),
        )
    }

    fn title(self) -> &'static str {
        match self {
            Step::Environment => "Check your environment",
            Step::Layout => "Lay out your zones",
            Step::Capture => "Capture each zone",
            Step::Consistency => "Consistency check",
            Step::Negatives => "Teach it what to ignore",
            Step::Save => "Save your profile",
        }
    }
}

pub struct Wizard {
    step: Step,
    cancelled: bool,
    capture: LiveCapture,
    device_name: Option<String>,
    sample_rate: Option<u32>,
    capture_error: Option<String>,
    level_peak: f32,

    // Environment step: running mean of the input RMS → noise floor.
    env_rms_accum: f64,
    env_rms_count: u64,
    noise_floor: f32,

    // Layout step.
    zone_count: usize,
    labels: Vec<String>,

    // Capture / negatives.
    session: Option<CalibrationSession>,
    armed: bool,
    last_feedback: Option<TapFeedback>,
    neg_armed: bool,
    neg_count: usize,

    // Consistency.
    consistency: Option<Result<ConsistencyReport, String>>,

    // Save.
    profile_name: String,
    save_error: Option<String>,

    /// Whether the main window is visible. While hidden to the tray the wizard
    /// stops its 33 fps repaint and mutes its capture worker (`PERFORMANCE.md`:
    /// no frames behind an invisible window).
    visible: bool,
}

impl Wizard {
    pub fn new(ctx: &egui::Context) -> Self {
        let capture = LiveCapture::start(None, ctx.clone(), CaptureDetail::Standard);
        let zone_count = 4;
        Self {
            step: Step::Environment,
            cancelled: false,
            capture,
            device_name: None,
            sample_rate: None,
            capture_error: None,
            level_peak: 0.0,
            env_rms_accum: 0.0,
            env_rms_count: 0,
            noise_floor: 0.0,
            zone_count,
            labels: default_labels(zone_count),
            session: None,
            armed: false,
            last_feedback: None,
            neg_armed: false,
            neg_count: 0,
            consistency: None,
            profile_name: "My desk".to_string(),
            save_error: None,
            visible: true,
        }
    }

    /// Set whether the main window is visible; forwards to the capture worker so
    /// a wizard hidden to the tray stops pumping frames and the mic meter.
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
        self.capture.set_visible(visible);
    }

    /// Surface a disk-save failure (raised by the shell) on the Save step, so the
    /// user can retry the save without redoing the calibration.
    pub fn set_save_error(&mut self, msg: String) {
        self.save_error = Some(msg);
    }

    /// Render one frame of the wizard. Returns `Some(..)` when it should close.
    pub fn ui(&mut self, ui: &mut egui::Ui, pal: &Palette) -> Option<WizardOutcome> {
        self.drain_capture();

        // Header: title + step indicator + cancel.
        let (idx, total) = self.step.position();
        ui.horizontal(|ui| {
            ui.heading(
                egui::RichText::new(self.step.title())
                    .color(pal.text)
                    .size(text::TITLE),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Cancel").clicked() {
                    self.cancelled = true;
                }
                ui.label(
                    egui::RichText::new(format!("Step {idx} of {total}")).color(pal.secondary),
                );
            });
        });
        ui.add_space(space::XS);
        if let Some(err) = &self.capture_error {
            ui.label(egui::RichText::new(format!("⚠ Microphone: {err}")).color(pal.danger));
        }
        ui.separator();
        ui.add_space(space::MD);

        if self.cancelled {
            return Some(WizardOutcome::Cancelled);
        }

        let outcome = match self.step {
            Step::Environment => self.environment_step(ui, pal),
            Step::Layout => self.layout_step(ui, pal),
            Step::Capture => self.capture_step(ui, pal),
            Step::Consistency => self.consistency_step(ui, pal),
            Step::Negatives => self.negatives_step(ui, pal),
            Step::Save => self.save_step(ui, pal),
        };

        // Keep the meter/level live while a mic step is on screen — but not while
        // hidden to the tray (no frames behind an invisible window).
        if self.visible
            && matches!(
                self.step,
                Step::Environment | Step::Capture | Step::Negatives
            )
        {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(33));
        }
        outcome
    }

    /// Absorb capture events, updating shared fields and feeding the active step.
    fn drain_capture(&mut self) {
        for ev in self.capture.poll() {
            match ev {
                CaptureEvent::Started {
                    device_name,
                    sample_rate,
                } => {
                    self.device_name = Some(device_name);
                    self.sample_rate = Some(sample_rate);
                }
                CaptureEvent::Level { peak, rms } => {
                    self.level_peak = peak;
                    if self.step == Step::Environment {
                        self.env_rms_accum += rms as f64;
                        self.env_rms_count += 1;
                    }
                }
                CaptureEvent::Onset {
                    window,
                    is_transient,
                    ..
                } => match self.step {
                    Step::Capture if self.armed && is_transient => {
                        if let Some(s) = &mut self.session {
                            self.last_feedback = Some(s.record_tap(&window));
                            if s.current_zone_complete() {
                                self.armed = false;
                            }
                        }
                    }
                    Step::Negatives if self.neg_armed => {
                        if let Some(s) = &mut self.session {
                            if s.record_negative(&window) {
                                self.neg_count = s.negatives().len();
                            }
                        }
                    }
                    _ => {}
                },
                // The wizard requests `CaptureDetail::Standard`, so the worker
                // never computes display frames for it. Named explicitly rather
                // than swept up by a catch-all arm.
                CaptureEvent::Frame { .. } => {}
                CaptureEvent::Failed(msg) => self.capture_error = Some(msg),
            }
        }
    }

    fn environment_step(&mut self, ui: &mut egui::Ui, pal: &Palette) -> Option<WizardOutcome> {
        ui.label(
            egui::RichText::new(
                "Stay quiet for a moment while Qwerty measures the room. \
                 We'll tell you if it's quiet enough to calibrate well.",
            )
            .color(pal.secondary),
        );
        ui.add_space(space::MD);

        let mic = self.device_name.as_deref().unwrap_or("opening microphone…");
        ui.label(
            egui::RichText::new(format!("Mic: {mic}"))
                .color(pal.secondary)
                .small(),
        );
        let frac = self.level_peak.clamp(0.0, 1.0).sqrt();
        ui.add(
            egui::ProgressBar::new(frac)
                .desired_width(320.0)
                .text("input"),
        );
        ui.add_space(space::MD);

        // Live provisional verdict from the running mean.
        let floor = self.current_noise_floor();
        let quality = assess_environment(floor);
        let (color, label) = match quality {
            EnvironmentQuality::Quiet => (pal.success, "Quiet — good to go"),
            EnvironmentQuality::Moderate => (
                pal.warning,
                "A bit noisy — calibration may be less reliable",
            ),
            EnvironmentQuality::TooNoisy => {
                (pal.danger, "Too loud — quiet the room before continuing")
            }
        };
        ui.label(egui::RichText::new(label).color(color).strong());
        ui.add_space(space::LG);

        // fail blocks; warn allows proceeding.
        let enabled = quality.passes() && self.env_rms_count > 0;
        if ui
            .add_enabled(enabled, egui::Button::new("Next: lay out zones"))
            .clicked()
        {
            self.noise_floor = floor;
            self.step = Step::Layout;
        }
        None
    }

    fn layout_step(&mut self, ui: &mut egui::Ui, pal: &Palette) -> Option<WizardOutcome> {
        ui.label(
            egui::RichText::new("How many zones, and what should they be called?")
                .color(pal.secondary),
        );
        ui.add_space(space::MD);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Zones:").color(pal.text));
            for n in ZONE_COUNTS {
                if ui
                    .selectable_label(self.zone_count == n, n.to_string())
                    .clicked()
                {
                    self.zone_count = n;
                    self.labels = resize_labels(std::mem::take(&mut self.labels), n);
                }
            }
        });
        ui.add_space(space::MD);

        let layouts = auto_layouts(self.zone_count);
        draw_desk_diagram(ui, pal, &layouts, &self.labels);
        ui.add_space(space::MD);

        // Editable labels.
        egui::Grid::new("zone_labels")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                for (i, label) in self.labels.iter_mut().enumerate() {
                    ui.label(egui::RichText::new(format!("Zone {}", i + 1)).color(pal.secondary));
                    ui.text_edit_singleline(label);
                    ui.end_row();
                }
            });
        ui.add_space(space::LG);

        if ui.button("Next: capture taps").clicked() {
            let zone_ids: Vec<ZoneId> = (0..self.zone_count).map(|_| ZoneId::new()).collect();
            self.session = Some(CalibrationSession::new(
                zone_ids,
                DEFAULT_TAPS_PER_ZONE,
                self.noise_floor,
            ));
            self.armed = false;
            self.last_feedback = None;
            self.step = Step::Capture;
        }
        None
    }

    fn capture_step(&mut self, ui: &mut egui::Ui, pal: &Palette) -> Option<WizardOutcome> {
        let Some(session) = &self.session else {
            return None;
        };
        let zone = session.current_zone();
        let count = session.zone_sample_count(zone);
        let target = session.taps_per_zone();
        let total_zones = session.zone_ids().len();
        let label = self.labels.get(zone).cloned().unwrap_or_default();

        ui.label(
            egui::RichText::new(format!(
                "Zone {} of {total_zones}: “{label}”. Tap this exact spot {target} times.",
                zone + 1
            ))
            .color(pal.text),
        );
        ui.add_space(space::MD);

        // Progress pips (color + shape, never color alone).
        ui.horizontal(|ui| {
            for i in 0..target {
                let (glyph, color) = if i < count {
                    ("●", pal.success)
                } else {
                    ("○", pal.secondary)
                };
                ui.label(egui::RichText::new(glyph).color(color).size(20.0));
            }
            ui.add_space(space::SM);
            ui.label(egui::RichText::new(format!("{count}/{target}")).color(pal.secondary));
        });
        ui.add_space(space::SM);

        // Last-tap feedback: specific reason, never generic.
        if let Some(fb) = self.last_feedback {
            let color = if fb.is_accepted() {
                pal.success
            } else {
                pal.warning
            };
            ui.label(egui::RichText::new(fb.reason()).color(color));
        } else {
            ui.label(
                egui::RichText::new("Arm the zone, then tap.")
                    .color(pal.secondary)
                    .small(),
            );
        }
        ui.add_space(space::MD);

        let complete = session.current_zone_complete();
        let is_last = zone + 1 >= total_zones;
        ui.horizontal(|ui| {
            let arm_label = if self.armed { "Stop" } else { "Arm zone" };
            if ui
                .add_enabled(!complete, egui::Button::new(arm_label))
                .clicked()
            {
                self.armed = !self.armed;
            }
            if ui.button("Undo last").clicked() {
                if let Some(s) = &mut self.session {
                    s.undo_last();
                    // Clear the stale per-tap feedback so it doesn't keep showing
                    // "accepted" for a tap that was just undone (mirrors Redo).
                    self.last_feedback = None;
                }
            }
            if ui.button("Redo zone").clicked() {
                if let Some(s) = &mut self.session {
                    s.redo_current_zone();
                    self.armed = false;
                    self.last_feedback = None;
                }
            }
        });
        ui.add_space(space::MD);

        if !is_last {
            if ui
                .add_enabled(complete, egui::Button::new("Next zone"))
                .clicked()
            {
                if let Some(s) = &mut self.session {
                    s.advance_zone();
                    self.armed = false;
                    self.last_feedback = None;
                }
            }
        } else if ui
            .add_enabled(complete, egui::Button::new("Check consistency"))
            .clicked()
        {
            self.armed = false;
            self.consistency = Some(
                self.session
                    .as_ref()
                    .unwrap()
                    .consistency()
                    .map_err(|e| e.to_string()),
            );
            self.step = Step::Consistency;
        }
        None
    }

    fn consistency_step(&mut self, ui: &mut egui::Ui, pal: &Palette) -> Option<WizardOutcome> {
        match &self.consistency {
            Some(Ok(report)) => {
                ui.label(
                    egui::RichText::new(
                        "Leave-one-out check: how reliably each zone tells itself apart.",
                    )
                    .color(pal.secondary),
                );
                ui.add_space(space::MD);
                let mut redo: Option<usize> = None;
                for (i, acc) in report.per_zone_accuracy.iter().enumerate() {
                    let flagged = report.inconsistent_zones.contains(&i);
                    let color = if flagged { pal.warning } else { pal.success };
                    let label = self.labels.get(i).cloned().unwrap_or_default();
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "{}  {label}: {:.0}%",
                                if flagged { "▲" } else { "✓" },
                                acc * 100.0
                            ))
                            .color(color),
                        );
                        if flagged && ui.button(format!("Redo zone {}", i + 1)).clicked() {
                            redo = Some(i);
                        }
                    });
                }
                ui.add_space(space::LG);
                if report.all_consistent() {
                    ui.label(egui::RichText::new("All zones look consistent.").color(pal.success));
                } else {
                    ui.label(
                        egui::RichText::new(
                            "Some zones overlap. Redo a flagged zone, or continue anyway.",
                        )
                        .color(pal.warning),
                    );
                }
                ui.add_space(space::MD);
                let cont = ui.button("Continue").clicked();

                if let Some(i) = redo {
                    if let Some(s) = &mut self.session {
                        s.restart_zone(i);
                    }
                    self.armed = false;
                    self.last_feedback = None;
                    self.step = Step::Capture;
                } else if cont {
                    self.step = Step::Negatives;
                }
            }
            Some(Err(e)) => {
                ui.label(
                    egui::RichText::new(format!("Consistency check failed: {e}")).color(pal.danger),
                );
                if ui.button("Back to capture").clicked() {
                    self.step = Step::Capture;
                }
            }
            None => {
                ui.label(egui::RichText::new("No consistency result.").color(pal.secondary));
            }
        }
        None
    }

    fn negatives_step(&mut self, ui: &mut egui::Ui, pal: &Palette) -> Option<WizardOutcome> {
        ui.label(
            egui::RichText::new(
                "Optional: with this armed, make the sounds you DON'T want to trigger \
                 actions — typing, talking, moving the mouse. This reduces false taps.",
            )
            .color(pal.secondary),
        );
        ui.add_space(space::MD);
        ui.horizontal(|ui| {
            let arm_label = if self.neg_armed {
                "Stop"
            } else {
                "Arm negatives"
            };
            if ui.button(arm_label).clicked() {
                self.neg_armed = !self.neg_armed;
            }
            ui.label(egui::RichText::new(format!("{} captured", self.neg_count)).color(pal.text));
        });
        ui.add_space(space::LG);
        if ui.button("Next: save").clicked() {
            self.neg_armed = false;
            self.step = Step::Save;
        }
        None
    }

    fn save_step(&mut self, ui: &mut egui::Ui, pal: &Palette) -> Option<WizardOutcome> {
        ui.label(egui::RichText::new("Name this profile and save it.").color(pal.secondary));
        ui.add_space(space::MD);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Name:").color(pal.text));
            ui.text_edit_singleline(&mut self.profile_name);
        });
        ui.add_space(space::MD);
        ui.label(
            egui::RichText::new(
                "Actions aren't assigned yet — after saving, open Zones & profiles to \
                 bind an action to each zone.",
            )
            .color(pal.secondary)
            .small(),
        );
        ui.add_space(space::LG);

        if let Some(err) = &self.save_error {
            ui.label(egui::RichText::new(format!("⚠ {err}")).color(pal.danger));
            ui.add_space(space::SM);
        }

        let can_save = !self.profile_name.trim().is_empty() && self.session.is_some();
        if ui
            .add_enabled(can_save, egui::Button::new("Save profile"))
            .clicked()
        {
            match self.build() {
                Ok(profile) => return Some(WizardOutcome::Saved(Box::new(profile))),
                Err(e) => self.save_error = Some(e),
            }
        }
        None
    }

    /// Build the profile from the session (called at save).
    fn build(&self) -> Result<Profile, String> {
        let session = self.session.as_ref().ok_or("no calibration session")?;
        let layouts = auto_layouts(self.zone_count);
        let zone_meta: Vec<ZoneMeta> = self
            .labels
            .iter()
            .zip(layouts)
            .map(|(label, layout)| ZoneMeta {
                label: label.clone(),
                layout,
            })
            .collect();
        // One shared constructor so the hash the profile is stamped with matches
        // what the live device-change check compares against (the full
        // channel/format fingerprint from ARCHITECTURE.md → Microphone identity
        // is a follow-up; sample rate is the stand-in).
        let fingerprint = DeviceFingerprint::from_device(
            self.device_name.clone().unwrap_or_else(|| "unknown".into()),
            self.sample_rate.unwrap_or(0),
        );
        session
            .build_profile(
                ProfileId::new(),
                self.profile_name.trim().to_string(),
                fingerprint,
                SensingMode::Passive,
                &zone_meta,
                Utc::now(),
            )
            .map_err(|e| e.to_string())
    }

    fn current_noise_floor(&self) -> f32 {
        if self.env_rms_count == 0 {
            0.0
        } else {
            (self.env_rms_accum / self.env_rms_count as f64) as f32
        }
    }
}

fn default_labels(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("Zone {}", i + 1)).collect()
}

/// Resize the label list to `n`, keeping existing entries and adding defaults.
fn resize_labels(mut labels: Vec<String>, n: usize) -> Vec<String> {
    let existing = labels.len();
    labels.truncate(n);
    for i in existing..n {
        labels.push(format!("Zone {}", i + 1));
    }
    labels
}

/// Auto-arrange `n` zones in a 2-column grid over the normalized desk canvas.
fn auto_layouts(n: usize) -> Vec<ZoneLayout> {
    let cols = 2usize;
    let rows = n.div_ceil(cols);
    let (w, h) = (1.0 / cols as f32, 1.0 / rows as f32);
    (0..n)
        .map(|i| {
            let (c, r) = (i % cols, i / cols);
            ZoneLayout {
                x: c as f32 * w,
                y: r as f32 * h,
                width: w,
                height: h,
            }
        })
        .collect()
}

/// Draw a simple top-down desk diagram: the normalized zone rectangles filled
/// in the elevated color with their labels, and a laptop marker in the center.
fn draw_desk_diagram(ui: &mut egui::Ui, pal: &Palette, layouts: &[ZoneLayout], labels: &[String]) {
    let size = egui::vec2(ui.available_width().min(460.0), 200.0);
    let (resp, painter) = ui.allocate_painter(size, egui::Sense::hover());
    let area = resp.rect;
    painter.rect_filled(area, egui::CornerRadius::same(8), pal.surface);

    for (i, layout) in layouts.iter().enumerate() {
        let rect = egui::Rect::from_min_size(
            egui::pos2(
                area.min.x + layout.x * area.width(),
                area.min.y + layout.y * area.height(),
            ),
            egui::vec2(layout.width * area.width(), layout.height * area.height()),
        )
        .shrink(4.0);
        painter.rect_filled(rect, egui::CornerRadius::same(6), pal.elevated);
        let label = labels.get(i).cloned().unwrap_or_default();
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(13.0),
            pal.text,
        );
    }

    // Laptop marker across the middle.
    let laptop = egui::Rect::from_center_size(area.center(), egui::vec2(area.width() * 0.34, 26.0));
    painter.rect_filled(laptop, egui::CornerRadius::same(4), pal.accent);
    painter.text(
        laptop.center(),
        egui::Align2::CENTER_CENTER,
        "laptop",
        egui::FontId::proportional(12.0),
        pal.base,
    );
}
