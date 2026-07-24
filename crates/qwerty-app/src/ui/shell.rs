//! The eframe application shell (`DESIGN.md` → Layout, `ROADMAP.md` Phase 3).
//!
//! This is the single main window: a persistent left navigation rail plus a
//! central content area that swaps between screens. It owns the boundary where
//! the pure, egui-free theme [`Tokens`](super::theme::Tokens) are mapped to
//! egui [`Visuals`](egui::Visuals), and it is the one place the
//! redraw-discipline rule from `PERFORMANCE.md` is enforced: a repaint is
//! scheduled only while an animation (currently the theme cross-fade) is
//! in flight, so a static screen costs no frames.

use std::time::Instant;

use qwerty_core::profile::{Theme, ZoneId};

use crate::app_state::{AppState, ListeningState, Screen};
use crate::capture_worker::{CaptureDetail, CaptureEvent, LiveCapture};
use crate::hotkeys::Hotkeys;
use crate::platform::{platform_for_this_os, PlatformActions};
use crate::tray::{Tray, TrayCommand};
use crate::ui::action_editor::ActionEditor;
use crate::ui::diagnostics::DiagnosticsScreen;
use crate::ui::evaluation::EvaluationScreen;
use crate::ui::motion;
use crate::ui::theme::{tokens_for, Color, Tokens};
use crate::ui::wizard::{Wizard, WizardOutcome};

/// Live input status while listening, drained from the capture worker.
#[derive(Default)]
struct LiveStatus {
    device: Option<String>,
    rms: f32,
    peak: f32,
    taps: u32,
    rejects: u32,
    error: Option<String>,
}

/// Launch the GUI. Blocks until the window is closed.
pub fn run() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Qwerty")
            .with_inner_size([1040.0, 720.0])
            .with_min_inner_size([840.0, 560.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Qwerty",
        options,
        Box::new(|cc| Ok(Box::new(QwertyApp::new(cc)))),
    )
}

struct QwertyApp {
    state: AppState,
    /// The tray icon + menu. `None` if the tray failed to initialize — the GUI
    /// still runs (the tray is an affordance, not a precondition).
    tray: Option<Tray>,
    /// The global listening-toggle hotkey. `None` if registration failed
    /// (e.g. the combo is already claimed by another app).
    hotkeys: Option<Hotkeys>,
    /// The live microphone capture, present only while listening.
    capture: Option<LiveCapture>,
    /// Live input status drained from the capture worker.
    live: LiveStatus,
    /// The calibration wizard, present only while it is taking over the window.
    wizard: Option<Wizard>,
    /// Set by a "Calibrate" button; the wizard is launched at end of frame so
    /// the launch doesn't fight the in-progress render borrow.
    launch_wizard: bool,
    /// The OS action backend (constructed once, on the UI thread). Boxed trait
    /// object so the DSP/UI never names a Windows type directly.
    platform: Box<dyn PlatformActions>,
    /// Transient state for the Zones action editor.
    action_editor: ActionEditor,
    /// Which zone is selected in the Zones editor.
    selected_zone: Option<ZoneId>,
    /// The Evaluation screen (guided held-out test + report history). Owns its
    /// own mic while a run is active; the shell yields the Home mic to it.
    evaluation: EvaluationScreen,
    /// The Diagnostics screen (live waveform/spectrogram/feature space). Owns
    /// its own mic while that screen is open; the shell starts and stops it.
    diagnostics: DiagnosticsScreen,
}

impl QwertyApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let system_dark = read_system_dark(&cc.egui_ctx);
        let state = AppState::load(system_dark);
        // Paint the correct theme on the very first frame — no flash of the
        // egui default palette (`PERFORMANCE.md`: no flash of unstyled content).
        let tokens = tokens_for(state.config.theme, system_dark);
        cc.egui_ctx.set_visuals(visuals_for(&tokens));

        // Wake the UI when a tray/menu/hotkey event arrives. Each thread blocks
        // on its process-global receiver and requests a repaint — so idle costs
        // nothing (the thread parks) yet the UI reacts the instant an event
        // lands, even with the window hidden to the tray (`PERFORMANCE.md`).
        spawn_wake_thread(cc.egui_ctx.clone());

        // The tray must be created on this (the main / event-loop) thread on
        // Windows, which is where the eframe creator runs.
        let tray = match Tray::new(state.listening) {
            Ok(t) => Some(t),
            Err(e) => {
                eprintln!("warning: tray icon unavailable: {e}");
                None
            }
        };
        let hotkeys = match Hotkeys::new() {
            Ok(h) => Some(h),
            Err(e) => {
                eprintln!("warning: global hotkey unavailable: {e}");
                None
            }
        };

        Self {
            state,
            tray,
            hotkeys,
            capture: None,
            live: LiveStatus::default(),
            wizard: None,
            launch_wizard: false,
            platform: platform_for_this_os(),
            action_editor: ActionEditor::default(),
            selected_zone: None,
            evaluation: EvaluationScreen::default(),
            diagnostics: DiagnosticsScreen::default(),
        }
    }

    /// Start/stop the microphone to match the listening state, then drain any
    /// capture events. Listening opens the device; pausing (or an error) drops
    /// the worker, which stops the stream via RAII. A capture failure moves the
    /// app into the `Error` state with a specific message (fail fast).
    fn reconcile_capture(&mut self, ctx: &egui::Context) {
        match self.state.listening {
            ListeningState::Listening if self.capture.is_none() => {
                self.live = LiveStatus::default();
                self.capture = Some(LiveCapture::start(
                    None,
                    ctx.clone(),
                    CaptureDetail::Standard,
                ));
            }
            ListeningState::Paused | ListeningState::Error if self.capture.is_some() => {
                self.capture = None; // Drop stops the worker + closes the device.
            }
            _ => {}
        }

        let events = self.capture.as_ref().map(LiveCapture::poll).unwrap_or_default();
        for ev in events {
            match ev {
                CaptureEvent::Started { device_name, .. } => self.live.device = Some(device_name),
                CaptureEvent::Level { rms, peak } => {
                    self.live.rms = rms;
                    self.live.peak = peak;
                }
                CaptureEvent::Onset { is_transient, .. } => {
                    if is_transient {
                        self.live.taps += 1;
                    } else {
                        self.live.rejects += 1;
                    }
                }
                // Home listening requests `CaptureDetail::Standard`; display
                // frames belong to the Diagnostics screen's own capture.
                CaptureEvent::Frame { .. } => {}
                CaptureEvent::Failed(msg) => {
                    self.live.error = Some(msg);
                    self.state.listening = ListeningState::Error;
                    self.capture = None;
                }
            }
        }
    }

    /// Apply tray/hotkey events and the close-to-tray policy. Returns after
    /// mutating `self.state` and issuing any viewport commands.
    fn handle_external_events(&mut self, ctx: &egui::Context) {
        if let Some(hk) = &self.hotkeys {
            if hk.poll_toggle() {
                self.state.toggle_listening();
            }
        }
        if let Some(tray) = &mut self.tray {
            for cmd in tray.poll() {
                match cmd {
                    TrayCommand::ToggleListening => {
                        self.state.toggle_listening();
                    }
                    TrayCommand::OpenWindow => {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
                        ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
                    }
                    TrayCommand::Quit => {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                }
            }
            // Keep the tray icon in sync however the state changed (hotkey,
            // tray menu, or an in-window button).
            tray.set_state(self.state.listening);
        }

        // Close-to-tray: intercept the window-close request and hide instead,
        // when configured and a tray exists to restore from. Otherwise let the
        // close proceed and the app exits.
        if ctx.input(|i| i.viewport().close_requested())
            && self.state.config.minimize_to_tray_on_close
            && self.tray.is_some()
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
        }
    }
}

impl eframe::App for QwertyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let now = Instant::now();

        // Keep the OS dark/light preference current so `Theme::System` tracks it
        // live. A change while on System simply recomputes effective tokens.
        self.state.system_dark = read_system_dark(ctx);

        // Apply tray/hotkey events and the close-to-tray policy before painting.
        self.handle_external_events(ctx);

        // The calibration wizard takes over the entire window when active.
        if self.wizard.is_some() {
            self.capture = None; // the wizard owns its own mic
            if self.state.listening == ListeningState::Listening {
                self.state.listening = ListeningState::Paused;
            }
            let tokens = self.effective_tokens(ctx, now);
            ctx.set_visuals(visuals_for(&tokens));
            let pal = Palette::from_tokens(&tokens);
            let mut outcome = None;
            egui::CentralPanel::default()
                .frame(
                    egui::Frame::default()
                        .fill(pal.base)
                        .inner_margin(egui::Margin::same(28)),
                )
                .show(ctx, |ui| {
                    outcome = self.wizard.as_mut().and_then(|w| w.ui(ui, &pal));
                });
            if let Some(o) = outcome {
                if let WizardOutcome::Saved(profile) = o {
                    if let Err(e) = self.state.save_profile(&profile) {
                        eprintln!("warning: could not save profile: {e}");
                    }
                    self.state.screen = Screen::Home;
                }
                self.wizard = None;
            }
            return;
        }

        // An evaluation run owns the mic while it is active, and only on its own
        // screen: navigating away ends it, so the mic and the live repaint both
        // stop (PERFORMANCE.md — the Diagnostics/live views must not keep running
        // in the background). The same rule keeps a second capture stream from
        // fighting Home listening for the device.
        if self.state.screen != Screen::Evaluation {
            self.evaluation.stop_run();
        }
        if self.state.screen != Screen::Diagnostics {
            // Leaving Diagnostics closes its mic and, with it, the 30 fps
            // repaint — the "redraw stops within one frame of navigating away"
            // requirement in `ACCEPTANCE.md`.
            self.diagnostics.stop();
        }

        let diagnostics_open = self.state.screen == Screen::Diagnostics;
        if self.evaluation.is_running() || diagnostics_open {
            // A screen-owned capture takes the device. Close the Home stream
            // first, then hand it over — never both open at once.
            if self.state.listening == ListeningState::Listening {
                self.state.listening = ListeningState::Paused;
            }
            self.capture = None; // Drop stops the Home worker + closes the device.
            if diagnostics_open && !self.diagnostics.is_active() {
                self.diagnostics.start(ctx);
            }
        } else {
            // Start/stop the mic to match listening state and absorb capture events.
            self.reconcile_capture(ctx);
        }

        // Resolve this frame's effective tokens, advancing any theme cross-fade.
        let tokens = self.effective_tokens(ctx, now);
        ctx.set_visuals(visuals_for(&tokens));
        let pal = Palette::from_tokens(&tokens);

        self.nav_rail(ctx, &pal);

        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .fill(pal.base)
                    .inner_margin(egui::Margin::same(24)),
            )
            .show(ctx, |ui| match self.state.screen {
                Screen::Home => self.home(ui, &pal),
                Screen::Zones => self.zones(ui, &pal),
                Screen::Diagnostics => self.diagnostics.ui(ui, &pal, &self.state),
                Screen::Evaluation => self.evaluation.ui(ui, &pal, &self.state),
                Screen::Settings => self.settings(ui, &pal, now),
            });

        // Launch the wizard at end of frame (set by a Calibrate button), after
        // the render borrow of `self` is released.
        if self.launch_wizard {
            self.launch_wizard = false;
            self.state.listening = ListeningState::Paused;
            self.capture = None;
            self.wizard = Some(Wizard::new(ctx));
            ctx.request_repaint();
        }

        // Start a requested evaluation run here, at end of frame, for the same
        // reason the wizard launches here: the Home microphone must be closed
        // *before* the run opens its own, or the device would briefly be held by
        // two capture streams (harmless on most drivers, a spurious device-busy
        // failure on some). Deciding this at the top of the frame would be too
        // early — the button that requests it is only rendered further down.
        if let Some(taps) = self.evaluation.take_pending_start() {
            if self.state.listening == ListeningState::Listening {
                self.state.listening = ListeningState::Paused;
            }
            self.capture = None; // Drop stops the Home worker + closes the device.
            if let Some(profile) = self.state.active_profile.as_ref() {
                self.evaluation.begin_run(ctx, profile, taps);
            }
            ctx.request_repaint();
        }
    }
}

impl QwertyApp {
    /// Tokens for this frame. If a theme cross-fade is active, interpolate the
    /// two token sets by the eased time fraction and schedule the next frame;
    /// once the fade's elapsed time passes `motion::STANDARD`, settle on the
    /// target and stop animating (redraw discipline).
    fn effective_tokens(&mut self, ctx: &egui::Context, now: Instant) -> Tokens {
        let Some(tr) = self.state.theme_transition else {
            return tokens_for(self.state.config.theme, self.state.system_dark);
        };
        let elapsed = now.saturating_duration_since(tr.started_at);
        let from = tokens_for(tr.from, self.state.system_dark);
        let to = tokens_for(tr.to, self.state.system_dark);
        if elapsed >= motion::STANDARD {
            self.state.theme_transition = None;
            return to;
        }
        let frac = elapsed.as_secs_f32() / motion::STANDARD.as_secs_f32();
        let eased = motion::STANDARD_EASE.eval(frac);
        // Still animating: ask egui for another frame, then stop once settled.
        ctx.request_repaint();
        Tokens::lerp(&from, &to, eased)
    }

    fn nav_rail(&mut self, ctx: &egui::Context, pal: &Palette) {
        egui::SidePanel::left("nav_rail")
            .resizable(false)
            .exact_width(224.0)
            .frame(
                egui::Frame::default()
                    .fill(pal.surface)
                    .inner_margin(egui::Margin::same(16)),
            )
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.heading(egui::RichText::new("Qwerty").color(pal.text));
                ui.label(
                    egui::RichText::new("acoustic tap zones")
                        .color(pal.secondary)
                        .small(),
                );
                ui.add_space(20.0);

                for screen in Screen::ALL {
                    let selected = self.state.screen == screen;
                    let label = format!("{}   {}", screen.glyph(), screen.label());
                    let text = egui::RichText::new(label)
                        .color(if selected { pal.accent } else { pal.text });
                    if ui.selectable_label(selected, text).clicked() {
                        self.state.screen = screen;
                    }
                    ui.add_space(4.0);
                }

                // Pin a compact listening-status pill to the bottom of the rail.
                ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                    status_pill(ui, pal, self.state.listening);
                });
            });
    }

    fn home(&mut self, ui: &mut egui::Ui, pal: &Palette) {
        ui.heading(egui::RichText::new("Home").color(pal.text).size(26.0));
        ui.add_space(4.0);
        if let Some(profile) = &self.state.active_profile {
            ui.label(
                egui::RichText::new(format!(
                    "Active profile: {}  ·  {} zone{}",
                    profile.name,
                    profile.zones.len(),
                    if profile.zones.len() == 1 { "" } else { "s" },
                ))
                .color(pal.text),
            );
        } else {
            ui.label(
                egui::RichText::new("Calibrate a profile to map taps on your desk to actions.")
                    .color(pal.secondary),
            );
        }
        ui.add_space(12.0);
        let calibrate_label = if self.state.active_profile.is_some() {
            "Recalibrate / new profile"
        } else {
            "Calibrate a new profile"
        };
        if ui.button(calibrate_label).clicked() {
            self.launch_wizard = true;
        }
        ui.add_space(20.0);

        // Status card.
        card(ui, pal, |ui| {
            ui.horizontal(|ui| {
                status_pill(ui, pal, self.state.listening);
                ui.add_space(12.0);
                let verb = match self.state.listening {
                    ListeningState::Listening => "Pause",
                    ListeningState::Paused => "Start listening",
                    ListeningState::Error => "Resolve device error first",
                };
                let enabled = self.state.listening != ListeningState::Error;
                if ui.add_enabled(enabled, egui::Button::new(verb)).clicked() {
                    self.state.toggle_listening();
                }
            });
            if self.hotkeys.is_some() {
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new(format!(
                        "Toggle listening from anywhere with {}.",
                        Hotkeys::DEFAULT_LABEL
                    ))
                    .color(pal.secondary)
                    .small(),
                );
            }

            match self.state.listening {
                ListeningState::Listening => {
                    ui.add_space(10.0);
                    ui.separator();
                    ui.add_space(6.0);
                    let mic = self.live.device.as_deref().unwrap_or("opening microphone…");
                    ui.label(egui::RichText::new(format!("Mic: {mic}")).color(pal.secondary).small());
                    ui.add_space(4.0);
                    // Peak level, sqrt-shaped so quiet signals are still visible.
                    let frac = self.live.peak.clamp(0.0, 1.0).sqrt();
                    ui.add(
                        egui::ProgressBar::new(frac)
                            .desired_width(260.0)
                            .text(format!("input {:.0}%", frac * 100.0)),
                    );
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(format!(
                            "taps {}   ·   rejected {}",
                            self.live.taps, self.live.rejects
                        ))
                        .color(pal.text),
                    );
                }
                ListeningState::Error => {
                    ui.add_space(10.0);
                    let msg = self.live.error.as_deref().unwrap_or("audio device error");
                    ui.label(egui::RichText::new(format!("⚠ {msg}")).color(pal.danger));
                }
                ListeningState::Paused => {}
            }
        });

        ui.add_space(16.0);
        // Honest-accuracy framing (DESIGN.md): never imply an unearned number.
        ui.label(
            egui::RichText::new(
                "This profile has not been evaluated. Accuracy is shown only after \
                 a saved evaluation report exists for this exact desk and mic.",
            )
            .color(pal.secondary)
            .italics(),
        );
    }

    fn zones(&mut self, ui: &mut egui::Ui, pal: &Palette) {
        ui.heading(
            egui::RichText::new("Zones & profiles")
                .color(pal.text)
                .size(26.0),
        );
        ui.add_space(8.0);

        // Disjoint field borrows so the editor can hold the profile + platform
        // at once.
        let Self {
            state,
            action_editor,
            platform,
            selected_zone,
            launch_wizard,
            ..
        } = self;

        let Some(profile) = state.active_profile.as_mut() else {
            ui.label(
                egui::RichText::new(
                    "No profile yet. Calibrate one to define zones and bind actions.",
                )
                .color(pal.secondary),
            );
            ui.add_space(10.0);
            if ui.button("Calibrate a new profile").clicked() {
                *launch_wizard = true;
            }
            return;
        };

        // Default/repair the selection.
        if selected_zone.is_none() || !profile.zones.iter().any(|z| Some(z.id) == *selected_zone) {
            *selected_zone = profile.zones.first().map(|z| z.id);
        }

        let mut changed = false;
        ui.columns(2, |cols| {
            cols[0].label(egui::RichText::new(&profile.name).color(pal.text).strong());
            cols[0].add_space(6.0);
            for z in &profile.zones {
                let sel = Some(z.id) == *selected_zone;
                let text =
                    egui::RichText::new(&z.label).color(if sel { pal.accent } else { pal.text });
                if cols[0].selectable_label(sel, text).clicked() {
                    *selected_zone = Some(z.id);
                }
            }

            let right = &mut cols[1];
            match selected_zone.and_then(|sel| profile.zones.iter_mut().find(|z| z.id == sel)) {
                Some(zone) => {
                    right.label(
                        egui::RichText::new(format!("Actions for “{}”", zone.label))
                            .color(pal.text)
                            .strong(),
                    );
                    right.add_space(6.0);
                    let zid = zone.id;
                    egui::ScrollArea::vertical().show(right, |ui| {
                        changed = action_editor.ui(ui, pal, zid, &mut zone.actions, platform.as_ref());
                    });
                }
                None => {
                    right.label(
                        egui::RichText::new("This profile has no zones.").color(pal.secondary),
                    );
                }
            }
        });

        if changed {
            if let Err(e) = state.save_active_profile() {
                eprintln!("warning: could not save profile: {e}");
            }
        }
    }

    fn settings(&mut self, ui: &mut egui::Ui, pal: &Palette, now: Instant) {
        ui.heading(egui::RichText::new("Settings").color(pal.text).size(26.0));
        ui.add_space(16.0);

        section(ui, pal, "Theme");
        ui.horizontal_wrapped(|ui| {
            for (theme, name) in THEME_CHOICES {
                let selected = self.state.config.theme == theme;
                let text = egui::RichText::new(name)
                    .color(if selected { pal.accent } else { pal.text });
                if ui.selectable_label(selected, text).clicked()
                    && self.state.begin_theme_switch(theme, now)
                {
                    self.state.save_config();
                }
            }
        });
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new("System follows Windows; the rest are fixed themes.")
                .color(pal.secondary)
                .small(),
        );

        ui.add_space(20.0);
        section(ui, pal, "Startup");
        if ui
            .checkbox(
                &mut self.state.config.start_with_windows,
                "Start Qwerty when I sign in",
            )
            .changed()
        {
            self.state.save_config();
        }
        if ui
            .checkbox(
                &mut self.state.config.minimize_to_tray_on_close,
                "Minimize to the tray when the window is closed",
            )
            .changed()
        {
            self.state.save_config();
        }

        ui.add_space(20.0);
        section(ui, pal, "Privacy");
        if ui
            .checkbox(
                &mut self.state.config.debug_capture_enabled,
                "Enable debug audio capture (off by default)",
            )
            .changed()
        {
            self.state.save_config();
        }
        if self.state.config.debug_capture_enabled {
            ui.label(
                egui::RichText::new(
                    "⚠ Debug capture is ON — raw audio windows will be written to disk.",
                )
                .color(pal.warning),
            );
        } else {
            ui.label(
                egui::RichText::new("Everything runs and stays on this machine. No network calls.")
                    .color(pal.secondary)
                    .small(),
            );
        }
    }
}

/// The 5 selectable themes and their display names (`DESIGN.md` → Themes).
const THEME_CHOICES: [(Theme, &str); 5] = [
    (Theme::System, "System"),
    (Theme::Daylight, "Daylight"),
    (Theme::Midnight, "Midnight"),
    (Theme::Aurora, "Aurora"),
    (Theme::HighContrast, "High contrast"),
];

/// A status pill combining a color *and* a glyph *and* text — never color alone
/// (`DESIGN.md` → Accessibility).
fn status_pill(ui: &mut egui::Ui, pal: &Palette, state: ListeningState) {
    let (color, glyph) = match state {
        ListeningState::Listening => (pal.success, "●"),
        ListeningState::Paused => (pal.secondary, "❚❚"),
        ListeningState::Error => (pal.danger, "▲"),
    };
    ui.label(
        egui::RichText::new(format!("{glyph}  {}", state.label()))
            .color(color)
            .strong(),
    );
}

/// A titled section label.
fn section(ui: &mut egui::Ui, pal: &Palette, title: &str) {
    ui.label(egui::RichText::new(title).color(pal.text).strong().size(16.0));
    ui.add_space(6.0);
}

/// A simple elevated card frame. `pub(crate)` so the action editor reuses it.
pub(crate) fn card(ui: &mut egui::Ui, pal: &Palette, add: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::default()
        .fill(pal.elevated)
        .stroke(egui::Stroke::new(1.0_f32, pal.border))
        .inner_margin(egui::Margin::same(16))
        .corner_radius(egui::CornerRadius::same(8))
        .show(ui, add);
}

/// egui `Color32` versions of the semantic tokens, for painting this frame.
/// `pub(crate)` so the calibration wizard can paint with the same palette.
pub(crate) struct Palette {
    pub(crate) base: egui::Color32,
    pub(crate) surface: egui::Color32,
    pub(crate) elevated: egui::Color32,
    pub(crate) text: egui::Color32,
    pub(crate) secondary: egui::Color32,
    pub(crate) border: egui::Color32,
    pub(crate) accent: egui::Color32,
    pub(crate) success: egui::Color32,
    pub(crate) warning: egui::Color32,
    pub(crate) danger: egui::Color32,
}

impl Palette {
    pub(crate) fn from_tokens(t: &Tokens) -> Self {
        Self {
            base: c32(t.bg_base),
            surface: c32(t.bg_surface),
            elevated: c32(t.bg_elevated),
            text: c32(t.text_primary),
            secondary: c32(t.text_secondary),
            border: c32(t.border),
            accent: c32(t.accent),
            success: c32(t.success),
            warning: c32(t.warning),
            danger: c32(t.danger),
        }
    }
}

pub(crate) fn c32(c: Color) -> egui::Color32 {
    egui::Color32::from_rgb(c.r, c.g, c.b)
}

/// Spawn one background thread per external event source (tray, menu, hotkey).
/// Each blocks on its process-global receiver and requests a repaint when an
/// event arrives — parking (zero CPU) while idle. The receivers are `'static`
/// and never disconnect, so `recv()` only returns on a real event.
fn spawn_wake_thread(ctx: egui::Context) {
    let sources: [fn() -> bool; 3] = [
        || tray_icon::TrayIconEvent::receiver().recv().is_ok(),
        || tray_icon::menu::MenuEvent::receiver().recv().is_ok(),
        || global_hotkey::GlobalHotKeyEvent::receiver().recv().is_ok(),
    ];
    for wait_for_event in sources {
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            while wait_for_event() {
                ctx.request_repaint();
            }
        });
    }
}

/// Read the OS dark-mode preference; default to dark when unknown (Midnight is
/// the OLED-friendly default per `DESIGN.md`).
fn read_system_dark(ctx: &egui::Context) -> bool {
    match ctx.input(|i| i.raw.system_theme) {
        Some(egui::Theme::Light) => false,
        Some(egui::Theme::Dark) | None => true,
    }
}

/// Map the pure theme tokens to an egui [`Visuals`]. Starts from egui's own
/// dark/light base (so rounding/spacing stay sensible across egui versions) and
/// overrides only the *color* tokens — the one boundary between the egui-free
/// token system and egui itself.
fn visuals_for(t: &Tokens) -> egui::Visuals {
    // Derive light/dark from the (possibly mid-fade) base brightness rather than
    // a Theme enum, which does not exist during a cross-fade.
    let dark = (t.bg_base.r as u32 + t.bg_base.g as u32 + t.bg_base.b as u32) < 3 * 128;
    let mut v = if dark {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };
    v.dark_mode = dark;

    v.panel_fill = c32(t.bg_base);
    v.window_fill = c32(t.bg_elevated);
    v.extreme_bg_color = c32(t.bg_base);
    v.faint_bg_color = c32(t.bg_surface);
    v.hyperlink_color = c32(t.accent);
    v.error_fg_color = c32(t.danger);
    v.warn_fg_color = c32(t.warning);

    // Default text + borders.
    v.widgets.noninteractive.fg_stroke.color = c32(t.text_primary);
    v.widgets.noninteractive.bg_stroke.color = c32(t.border);
    v.window_stroke.color = c32(t.border);

    // Inactive controls sit on the surface layer with a border.
    v.widgets.inactive.bg_fill = c32(t.bg_surface);
    v.widgets.inactive.weak_bg_fill = c32(t.bg_surface);
    v.widgets.inactive.bg_stroke.color = c32(t.border);
    v.widgets.inactive.fg_stroke.color = c32(t.text_primary);

    // Hover raises to the elevated layer with an accent edge.
    v.widgets.hovered.bg_fill = c32(t.bg_elevated);
    v.widgets.hovered.weak_bg_fill = c32(t.bg_elevated);
    v.widgets.hovered.bg_stroke.color = c32(t.accent);
    v.widgets.hovered.fg_stroke.color = c32(t.text_primary);

    // Active / pressed uses the accent as the fill.
    v.widgets.active.bg_fill = c32(t.accent);
    v.widgets.active.weak_bg_fill = c32(t.accent);
    v.widgets.active.bg_stroke.color = c32(t.accent);

    // Selection (and thus selectable_label highlight) uses a translucent accent.
    let a = t.accent;
    v.selection.bg_fill = egui::Color32::from_rgba_unmultiplied(a.r, a.g, a.b, 90);
    v.selection.stroke.color = c32(t.accent);

    v
}
