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

use crate::audio_thread::list_input_devices;
use crate::capture_worker::{CaptureDetail, CaptureEvent, LiveCapture};
use crate::ui::motion::{self, AnimState};
use crate::ui::shell::{c32, divider, mix, primary_button, with_alpha, Palette};
use crate::ui::style::{self, radius, space, text};
use egui_phosphor::regular as icon;
use crate::ui::theme::{on_color, Color};

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

/// The steps in flow order — the single source of truth the stepper indicator
/// and the `index()` accessor both derive from, so a step can never be numbered
/// inconsistently between them.
const STEP_ORDER: [Step; 6] = [
    Step::Environment,
    Step::Layout,
    Step::Capture,
    Step::Consistency,
    Step::Negatives,
    Step::Save,
];

impl Step {
    /// 0-based position in [`STEP_ORDER`], for the painted stepper.
    fn index(self) -> usize {
        STEP_ORDER.iter().position(|s| *s == self).unwrap()
    }

    /// A one- or two-word label for the stepper circle, distinct from the
    /// step's full sentence `title()` (which is too long under a 32 px circle).
    fn short_label(self) -> &'static str {
        match self {
            Step::Environment => "Room",
            Step::Layout => "Layout",
            Step::Capture => "Capture",
            Step::Consistency => "Check",
            Step::Negatives => "Ignore",
            Step::Save => "Save",
        }
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

    // Microphone selection (Environment step). `devices` is the list of input
    // devices, queried once at construction; `selected_device` is the user's
    // pick (`None` = the system default, matching `LiveCapture::start(None,..)`).
    // Calibrating on the chosen mic records *its* fingerprint on the profile, so
    // the mic is a property of the calibration, not a global setting — and the
    // live mismatch gate then enforces it at listen time.
    devices: Vec<String>,
    selected_device: Option<String>,

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

    // --- Presentation animation (Part 4; visual only, never gates logic) ---
    /// Fires on every step change: drives the per-step fade-and-rise arrival
    /// and the current-step ring pulse in the painted stepper.
    step_anim: AnimState,
    /// Taps recorded for the current zone as of last frame, so a fresh tap is
    /// detected (count increased) and its pip's ink-drop fill can be triggered.
    pip_count: usize,
    /// The just-landed pip: `(pip index, its fill animation)`. One-shot.
    pip_fill: Option<(usize, AnimState)>,

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
            // Enumerate input devices once; an enumeration failure just leaves
            // the list empty, so the picker still offers "System default".
            devices: list_input_devices().unwrap_or_default(),
            selected_device: None,
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
            step_anim: AnimState::start(motion::DELIBERATE),
            pip_count: 0,
            pip_fill: None,
            visible: true,
        }
    }

    /// Move to `step`, restarting the arrival transition and clearing the
    /// per-tap pip tracking so it re-primes for the new step. The single choke
    /// point for step changes, so no transition can be forgotten at a call site.
    fn goto(&mut self, step: Step) {
        self.step = step;
        self.step_anim = AnimState::start(motion::DELIBERATE);
        self.pip_count = 0;
        self.pip_fill = None;
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

        // Header: the painted stepper carries the position now, so the title
        // drops the redundant "Step N of M" text and just names the step, with
        // Cancel parked top-right.
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
            });
        });
        ui.add_space(space::SM);
        draw_stepper(ui, pal, self.step, &self.step_anim);
        ui.add_space(space::XS);
        if let Some(err) = &self.capture_error {
            ui.label(egui::RichText::new(format!("⚠ Microphone: {err}")).color(pal.danger));
        }
        divider(ui, pal);
        ui.add_space(space::SM);

        if self.cancelled {
            return Some(WizardOutcome::Cancelled);
        }

        // Per-step arrival transition: the body fades up from 25% opacity while
        // rising a few pixels into place (STANDARD_EASE over DELIBERATE). Only
        // the *painted* content moves — the allocated layout is unchanged once
        // settled, so nothing below reflows. Fire-and-forget via `step_anim`.
        let tp = self.step_anim.t(motion::STANDARD_EASE);
        let outcome = ui
            .scope(|ui| {
                ui.multiply_opacity(0.25 + 0.75 * tp);
                ui.add_space((1.0 - tp) * 10.0);
                match self.step {
                    Step::Environment => self.environment_step(ui, pal),
                    Step::Layout => self.layout_step(ui, pal),
                    Step::Capture => self.capture_step(ui, pal),
                    Step::Consistency => self.consistency_step(ui, pal),
                    Step::Negatives => self.negatives_step(ui, pal),
                    Step::Save => self.save_step(ui, pal),
                }
            })
            .inner;

        // Repaint discipline (PERFORMANCE.md): pump frames only while something
        // is actually live, and never behind a window hidden to the tray.
        //  - a mic step needs the meter/level refreshed at ~33 fps;
        //  - the step-arrival transition and the pip ink-drop are one-shots that
        //    must animate to completion, after which repaints stop and idle CPU
        //    returns to zero.
        if self.visible {
            let mic_live = matches!(
                self.step,
                Step::Environment | Step::Capture | Step::Negatives
            );
            let anim_live = !self.step_anim.is_complete()
                || self.pip_fill.as_ref().is_some_and(|(_, a)| !a.is_complete());
            if mic_live {
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_millis(33));
            } else if anim_live {
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_millis(16));
            }
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

        // Microphone picker: choose which mic to calibrate on. Its identity is
        // recorded on the profile, so switching mics later prompts a recalibrate
        // (and the live gate blocks actions on a mismatch) — hence pick the mic
        // you'll actually use. Changing it reopens the stream on the new device
        // and restarts the room measurement (a different mic has its own noise
        // floor). Only offered here on the Environment step, before capture, so
        // the device can't change mid-calibration.
        let mut new_choice: Option<Option<String>> = None;
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Microphone").color(pal.text));
            egui::ComboBox::from_id_salt("wizard_mic")
                .selected_text(self.selected_device.as_deref().unwrap_or("System default"))
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(self.selected_device.is_none(), "System default")
                        .clicked()
                    {
                        new_choice = Some(None);
                    }
                    for d in &self.devices {
                        let sel = self.selected_device.as_deref() == Some(d.as_str());
                        if ui.selectable_label(sel, d).clicked() {
                            new_choice = Some(Some(d.clone()));
                        }
                    }
                });
        });
        if let Some(choice) = new_choice {
            if choice != self.selected_device {
                self.selected_device = choice;
                // Reopen on the new device (the old LiveCapture stops via Drop),
                // and reset the device info + room measurement for the new mic.
                self.capture = LiveCapture::start(
                    self.selected_device.clone(),
                    ui.ctx().clone(),
                    CaptureDetail::Standard,
                );
                self.capture.set_visible(self.visible);
                self.device_name = None;
                self.sample_rate = None;
                self.level_peak = 0.0;
                self.env_rms_accum = 0.0;
                self.env_rms_count = 0;
                self.noise_floor = 0.0;
            }
        }
        ui.add_space(space::SM);

        let mic = self.device_name.as_deref().unwrap_or("opening microphone…");
        ui.label(
            egui::RichText::new(format!("Active: {mic}"))
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
            .add_enabled_ui(enabled, |ui| primary_button(ui, pal, "Next: lay out zones"))
            .inner
            .clicked()
        {
            self.noise_floor = floor;
            self.goto(Step::Layout);
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

        if primary_button(ui, pal, "Next: capture taps").clicked() {
            let zone_ids: Vec<ZoneId> = (0..self.zone_count).map(|_| ZoneId::new()).collect();
            self.session = Some(CalibrationSession::new(
                zone_ids,
                DEFAULT_TAPS_PER_ZONE,
                self.noise_floor,
            ));
            self.armed = false;
            self.last_feedback = None;
            self.goto(Step::Capture);
        }
        None
    }

    fn capture_step(&mut self, ui: &mut egui::Ui, pal: &Palette) -> Option<WizardOutcome> {
        // Pull everything this frame needs out of the session in one borrow, so
        // the immutable borrow ends before the pip-tracking self-mutation below
        // (and before the Arm/Undo/Redo handlers re-borrow `&mut self.session`).
        let (zone, count, target, total_zones, complete) = {
            let Some(session) = &self.session else {
                return None;
            };
            let zone = session.current_zone();
            (
                zone,
                session.zone_sample_count(zone),
                session.taps_per_zone(),
                session.zone_ids().len(),
                session.current_zone_complete(),
            )
        };
        let is_last = zone + 1 >= total_zones;
        let label = self.labels.get(zone).cloned().unwrap_or_default();

        // A freshly-recorded tap (count grew since last frame) triggers that
        // pip's ink-drop fill; an undo (count shrank) or a zone change just
        // re-syncs the counter without animating.
        if count > self.pip_count {
            self.pip_fill = Some((count - 1, AnimState::start(motion::QUICK)));
        } else if count < self.pip_count {
            self.pip_fill = None;
        }
        self.pip_count = count;

        ui.label(
            egui::RichText::new(format!(
                "Zone {} of {total_zones}: “{label}”. Tap this exact spot {target} times.",
                zone + 1
            ))
            .color(pal.text),
        );
        ui.add_space(space::MD);

        // Progress pips: custom-painted circles (Part 4). Filled for recorded
        // taps, the just-landed one blooming an ink-drop ring, the next one
        // ringed in accent, the rest hollow — shape + fill carry state, never
        // colour alone (DESIGN.md accessibility rule).
        draw_tap_pips(
            ui,
            pal,
            count,
            target,
            self.pip_fill.as_ref().map(|(i, a)| (*i, a)),
        );
        ui.add_space(space::XS);
        ui.label(
            egui::RichText::new(format!("{count} of {target} taps"))
                .color(pal.secondary)
                .size(text::CAPTION),
        );
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
                .add_enabled_ui(complete, |ui| primary_button(ui, pal, "Next zone"))
                .inner
                .clicked()
            {
                if let Some(s) = &mut self.session {
                    s.advance_zone();
                    self.armed = false;
                    self.last_feedback = None;
                }
            }
        } else if ui
            .add_enabled_ui(complete, |ui| primary_button(ui, pal, "Check consistency"))
            .inner
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
            self.goto(Step::Consistency);
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
                                "{}  {label}: {:.1}%",
                                if flagged { icon::WARNING } else { icon::CHECK },
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
                let cont = primary_button(ui, pal, "Continue").clicked();

                if let Some(i) = redo {
                    if let Some(s) = &mut self.session {
                        s.restart_zone(i);
                    }
                    self.armed = false;
                    self.last_feedback = None;
                    self.goto(Step::Capture);
                } else if cont {
                    self.goto(Step::Negatives);
                }
            }
            Some(Err(e)) => {
                ui.label(
                    egui::RichText::new(format!("Consistency check failed: {e}")).color(pal.danger),
                );
                if ui.button("Back to capture").clicked() {
                    self.goto(Step::Capture);
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
        if primary_button(ui, pal, "Next: save").clicked() {
            self.neg_armed = false;
            self.goto(Step::Save);
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
            .add_enabled_ui(can_save, |ui| primary_button(ui, pal, "Save profile"))
            .inner
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

/// A WCAG-picked contrasting ink for text drawn on the accent fill — the same
/// helper the primary button uses, so the laptop/stepper labels stay legible on
/// every theme's accent (yellow High-Contrast, teal Aurora) rather than
/// assuming a fixed light-on-dark.
fn ink_on_accent(pal: &Palette) -> egui::Color32 {
    c32(on_color(Color::rgb(pal.accent.r(), pal.accent.g(), pal.accent.b())))
}

/// The step-progress indicator (Part 4): a horizontal row of circles joined by
/// connector segments. Completed steps are filled with a check; the current
/// step is filled with its number and, on arrival, a one-shot ring that pulses
/// outward then rests (driven by `step_anim`, so it plays on entry and stops —
/// no perpetual repaint, honouring PERFORMANCE.md); upcoming steps are hollow.
/// Each circle is labelled beneath. Replaces the old "Step N of M" text.
fn draw_stepper(ui: &mut egui::Ui, pal: &Palette, current: Step, step_anim: &AnimState) {
    let n = STEP_ORDER.len();
    let cur = current.index();
    let r = 13.0_f32;
    let height = r * 2.0 + 22.0;
    let (resp, painter) = ui.allocate_painter(
        egui::vec2(ui.available_width().min(560.0), height),
        egui::Sense::hover(),
    );
    let area = resp.rect;
    let cy = area.min.y + r + 2.0;
    let x0 = area.min.x + r;
    let x1 = area.max.x - r;
    let step_x = if n > 1 { (x1 - x0) / (n - 1) as f32 } else { 0.0 };
    let cx = |i: usize| x0 + step_x * i as f32;

    // Connector segments behind the circles: accent up to the current step,
    // border beyond it.
    for i in 0..n.saturating_sub(1) {
        let col = if i < cur { pal.accent } else { pal.border };
        painter.line_segment(
            [egui::pos2(cx(i) + r, cy), egui::pos2(cx(i + 1) - r, cy)],
            egui::Stroke::new(2.0_f32, col),
        );
    }

    let pulse = 1.0 - step_anim.linear(); // 1 → 0 over the arrival transition
    for (i, step) in STEP_ORDER.iter().enumerate() {
        let center = egui::pos2(cx(i), cy);
        let done = i < cur;
        let is_cur = i == cur;
        if done || is_cur {
            painter.circle_filled(center, r, pal.accent);
            let glyph = if done {
                icon::CHECK.to_string()
            } else {
                (i + 1).to_string()
            };
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                glyph,
                egui::FontId::proportional(14.0),
                ink_on_accent(pal),
            );
            if is_cur && pulse > 0.01 {
                painter.circle_stroke(
                    center,
                    r + pulse * 8.0,
                    egui::Stroke::new(2.0_f32, with_alpha(pal.accent, (pulse * 180.0) as u8)),
                );
            }
        } else {
            painter.circle_filled(center, r, pal.surface);
            painter.circle_stroke(center, r, egui::Stroke::new(1.5_f32, pal.border));
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                (i + 1).to_string(),
                egui::FontId::proportional(13.0),
                pal.secondary,
            );
        }
        painter.text(
            egui::pos2(cx(i), cy + r + 10.0),
            egui::Align2::CENTER_CENTER,
            step.short_label(),
            egui::FontId::proportional(text::CAPTION),
            if is_cur { pal.text } else { pal.secondary },
        );
    }
    // The position ("Step N of M") now lives only in this painted stepper, so
    // announce it to screen readers — otherwise a Narrator user loses the wizard
    // progress the old text label used to convey. Changes only on step change.
    resp.widget_info(|| {
        egui::WidgetInfo::labeled(
            egui::WidgetType::Label,
            true,
            format!("Step {} of {}: {}", cur + 1, n, current.title()),
        )
    });
}

/// The per-zone tap pips (Part 4): one custom-painted circle per expected tap.
/// Recorded taps are filled `success`; the just-landed pip (`fill`) blooms an
/// expanding, fading ink-drop ring while its fill scales in (one-shot); the
/// next expected pip is ringed in accent ("here next"); the rest are hollow.
fn draw_tap_pips(
    ui: &mut egui::Ui,
    pal: &Palette,
    count: usize,
    target: usize,
    fill: Option<(usize, &AnimState)>,
) {
    let r = 9.0_f32;
    let gap = 14.0_f32;
    let w = target as f32 * (r * 2.0) + target.saturating_sub(1) as f32 * gap;
    let (resp, painter) = ui.allocate_painter(
        egui::vec2(w.max(1.0), r * 2.0 + 12.0),
        egui::Sense::hover(),
    );
    let area = resp.rect;
    let cy = area.center().y;
    let cx = |i: usize| area.min.x + r + i as f32 * (r * 2.0 + gap);

    for i in 0..target {
        let center = egui::pos2(cx(i), cy);
        let dropping = matches!(fill, Some((idx, _)) if idx == i).then(|| fill.unwrap().1);
        if i < count {
            // Filled. During its drop the fill scales in and a ring expands out.
            let scale = dropping.map_or(1.0, |a| 0.4 + 0.6 * a.t(motion::EMPHASIZED_EASE));
            painter.circle_filled(center, r * scale, pal.success);
            if let Some(a) = dropping {
                let lin = a.linear();
                painter.circle_stroke(
                    center,
                    r + 2.0 + lin * 10.0,
                    egui::Stroke::new(2.0_f32, with_alpha(pal.success, ((1.0 - lin) * 200.0) as u8)),
                );
            }
        } else if i == count {
            painter.circle_filled(center, r, mix(pal.surface, pal.accent, 0.12));
            painter.circle_stroke(center, r, egui::Stroke::new(2.0_f32, pal.accent));
        } else {
            painter.circle_filled(center, r, pal.surface);
            painter.circle_stroke(center, r, egui::Stroke::new(1.5_f32, pal.border));
        }
    }
}

/// A top-down desk diagram (Part 4): a raised desk surface (soft drop shadow +
/// border) carrying the normalized zone rectangles — each highlighting toward
/// accent on hover (`animate_bool_with_time`) — with a small two-part laptop
/// (base slab + raised screen) marking the centre.
fn draw_desk_diagram(ui: &mut egui::Ui, pal: &Palette, layouts: &[ZoneLayout], labels: &[String]) {
    let size = egui::vec2(ui.available_width().min(460.0), 210.0);
    let (resp, painter) = ui.allocate_painter(size, egui::Sense::hover());
    let area = resp.rect.shrink(4.0);
    let round = style::rounding(radius::MD);
    let hover_pos = resp.hover_pos();

    // Depth: a soft shadow slab under the surface, then the surface + border.
    painter.rect_filled(
        area.translate(egui::vec2(0.0, 3.0)),
        round,
        with_alpha(egui::Color32::BLACK, 40),
    );
    painter.rect_filled(area, round, pal.surface);
    painter.rect_stroke(
        area,
        round,
        egui::Stroke::new(1.0_f32, pal.border),
        egui::StrokeKind::Inside,
    );

    let tile_round = style::rounding(radius::SM);
    for (i, layout) in layouts.iter().enumerate() {
        let rect = egui::Rect::from_min_size(
            egui::pos2(
                area.min.x + layout.x * area.width(),
                area.min.y + layout.y * area.height(),
            ),
            egui::vec2(layout.width * area.width(), layout.height * area.height()),
        )
        .shrink(5.0);
        let hovered = hover_pos.is_some_and(|p| rect.contains(p));
        let hl =
            ui.ctx()
                .animate_bool_with_time(resp.id.with(i), hovered, motion::INSTANT.as_secs_f32());
        painter.rect_filled(rect, tile_round, mix(pal.elevated, pal.accent, hl * 0.18));
        painter.rect_stroke(
            rect,
            tile_round,
            egui::Stroke::new(1.0 + hl, mix(pal.border, pal.accent, hl)),
            egui::StrokeKind::Inside,
        );
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            labels.get(i).cloned().unwrap_or_default(),
            egui::FontId::proportional(13.0),
            pal.text,
        );
    }

    // Laptop across the middle: a base slab plus a raised screen.
    let lw = area.width() * 0.34;
    let base = egui::Rect::from_center_size(area.center() + egui::vec2(0.0, 7.0), egui::vec2(lw, 13.0));
    let screen = egui::Rect::from_center_size(
        area.center() - egui::vec2(0.0, 9.0),
        egui::vec2(lw * 0.9, 17.0),
    );
    painter.rect_filled(screen, style::rounding(3.0), mix(pal.accent, pal.base, 0.2));
    painter.rect_stroke(
        screen,
        style::rounding(3.0),
        egui::Stroke::new(1.0_f32, pal.accent),
        egui::StrokeKind::Inside,
    );
    painter.rect_filled(base, style::rounding(3.0), pal.accent);
    painter.text(
        base.center(),
        egui::Align2::CENTER_CENTER,
        "laptop",
        egui::FontId::proportional(11.0),
        ink_on_accent(pal),
    );
}
