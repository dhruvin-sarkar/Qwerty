//! The eframe application shell (`DESIGN.md` → Layout, `ROADMAP.md` Phase 3).
//!
//! This is the single main window: a persistent left navigation rail plus a
//! central content area that swaps between screens. It owns the boundary where
//! the pure, egui-free theme [`Tokens`](super::theme::Tokens) are mapped to
//! egui [`Visuals`](egui::Visuals), and it is the one place the
//! redraw-discipline rule from `PERFORMANCE.md` is enforced: a repaint is
//! scheduled only while an animation (currently the theme cross-fade) is
//! in flight, so a static screen costs no frames.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use qwerty_core::features::{FeatureExtractor, FEATURE_WINDOW_SAMPLES};
use qwerty_core::profile::{DeviceFingerprint, Theme, Zone, ZoneId};

use crate::ui::motion::AnimState;

use crate::app_state::{AppState, ListeningState, ProfileSummary, Screen};
use crate::capture_worker::{CaptureDetail, CaptureEvent, LiveCapture};
use crate::hotkeys::Hotkeys;
use crate::platform::{platform_for_this_os, PlatformActions};
use crate::tray::{Tray, TrayCommand};
use crate::ui::action_editor::ActionEditor;
use crate::ui::diagnostics::DiagnosticsScreen;
use crate::ui::evaluation::EvaluationScreen;
use crate::ui::motion;
use crate::ui::paint;
use crate::ui::style::{self, radius, space, text};
use crate::ui::theme::{on_color, tokens_for, Color, Tokens};
use crate::ui::wizard::{Wizard, WizardOutcome};

/// Live input status while listening, drained from the capture worker.
#[derive(Default)]
struct LiveStatus {
    device: Option<String>,
    /// The live device's negotiated sample rate, paired with `device` to form
    /// the current [`DeviceFingerprint`] for the device-change check.
    sample_rate: Option<u32>,
    rms: f32,
    peak: f32,
    /// A slowly-decaying peak, so the level meter's peak needle holds briefly at
    /// a transient's crest and eases down rather than snapping — the behaviour of
    /// a real audio input meter.
    peak_hold: f32,
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
    /// A profile-save failure to surface to the user as a dismissible banner
    /// above the active screen. `None` when there is nothing to report.
    /// Transient UI state only — never persisted.
    save_error: Option<String>,
    /// Set by the tray "Quit" item so the close-to-tray guard lets the close
    /// proceed to a real exit instead of hiding the window. One-shot: once the
    /// user has chosen to quit there is no path back, so it is never reset.
    force_quit: bool,
    /// Whether the main window is visible (vs hidden to the tray). Drives
    /// suppression of the live/level repaints while hidden (`PERFORMANCE.md`: a
    /// tray-hidden window must pump no frames).
    window_visible: bool,
    /// Cached summaries of all saved profiles, backing the Zones profile
    /// switcher (`DESIGN.md` → "Multiple desk profiles, switchable"). Refreshed
    /// only when the set changes — at startup, after the wizard saves a profile,
    /// and after a switch/export/import — so the switcher never rescans the disk
    /// per frame.
    profiles: Vec<ProfileSummary>,
    /// Cached `(path, display name)` of importable profile files in `Exports/`
    /// (`DESIGN.md` → import/export). Refreshed alongside `profiles`.
    importable: Vec<(PathBuf, String)>,
    /// The last export/import success message (e.g. the path a profile was
    /// exported to), shown inline under the Zones import/export controls.
    /// Failures use the shared save-error banner instead. Transient UI only.
    io_message: Option<String>,
    /// The name typed for a "Duplicate profile" action, before it is applied.
    /// Cleared after a successful duplicate. Transient UI only.
    duplicate_name: String,
    /// Whether the destructive "Delete profile" action is armed (awaiting a
    /// confirm click). Disarmed on any profile-set change (see `refresh_profiles`)
    /// so it never carries over to a different active profile. Transient UI only.
    confirm_delete: bool,
    /// The screen shown last frame, used to detect a navigation change and start
    /// the screen-transition fade (`MOTION.md` → Screen transition).
    last_screen: Screen,
    /// When the current screen-transition fade began; `None` once it has settled.
    screen_fade_start: Option<Instant>,
    /// Per-zone accept/reject pulse animations, keyed by `ZoneId`. Inserted (or
    /// replaced) each time a Home tap classifies to a zone; the value is the
    /// animation plus whether the tap was accepted (drives accept-glow vs.
    /// reject-shake). Completed entries are pruned by the repaint scheduler.
    zone_anim: HashMap<ZoneId, (AnimState, bool)>,
    /// A feature extractor for classifying Home taps *read-only*, purely to
    /// drive the zone-accept pulse — no action is dispatched here (that stays in
    /// the platform layer). Mirrors how Diagnostics/Evaluation classify.
    home_extractor: FeatureExtractor,
}

impl QwertyApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let system_dark = read_system_dark(&cc.egui_ctx);
        let state = AppState::load(system_dark);
        // Paint the correct theme on the very first frame — no flash of the
        // egui default palette (`PERFORMANCE.md`: no flash of unstyled content).
        let tokens = tokens_for(state.config.theme, system_dark);
        cc.egui_ctx.set_visuals(visuals_for(&tokens));
        // Spacing and roundness are theme-independent, so egui's global Spacing
        // is configured once here from the design tokens. The per-frame
        // `set_visuals` cross-fade replaces only the *color* half of the style,
        // leaving this breathing room intact (`style::configure_spacing`).
        cc.egui_ctx
            .style_mut(|s| style::configure_spacing(&mut s.spacing));
        // Re-skin every glyph in the app with the native Windows typeface
        // (`DESIGN.md` → Typography: Segoe UI is the platform-native choice),
        // replacing egui's bundled default. Best-effort — see `install_fonts`.
        install_fonts(&cc.egui_ctx);

        // Single-reader event pump. `tray-icon` and `global-hotkey` each deliver
        // events on one process-global channel that hands each message to
        // exactly one consumer. Create an app-owned channel per source, build
        // Tray/Hotkeys with the receiving ends, then spawn the forwarder threads
        // — each the sole reader of its global channel, forwarding into the app
        // channel and waking the UI. Spawning only after successful construction
        // means a failed Tray/Hotkeys never leaves a forwarder parked forever on
        // its global receiver. Waking a hidden (tray-only) window is intended so
        // a tray "Open"/"Quit" is processed promptly (`PERFORMANCE.md`).
        let (menu_tx, menu_rx) = std::sync::mpsc::channel();
        let (tray_tx, tray_rx) = std::sync::mpsc::channel();
        let (hotkey_tx, hotkey_rx) = std::sync::mpsc::channel();

        // The tray must be created on this (the main / event-loop) thread on
        // Windows, which is where the eframe creator runs.
        let tray = match Tray::new(state.listening, menu_rx, tray_rx) {
            Ok(t) => {
                let ctx = cc.egui_ctx.clone();
                spawn_forward(
                    menu_tx,
                    || tray_icon::menu::MenuEvent::receiver().recv().ok(),
                    ctx.clone(),
                );
                spawn_forward(
                    tray_tx,
                    || tray_icon::TrayIconEvent::receiver().recv().ok(),
                    ctx,
                );
                Some(t)
            }
            Err(e) => {
                eprintln!("warning: tray icon unavailable: {e}");
                None
            }
        };
        let hotkeys = match Hotkeys::new(hotkey_rx) {
            Ok(h) => {
                spawn_forward(
                    hotkey_tx,
                    || global_hotkey::GlobalHotKeyEvent::receiver().recv().ok(),
                    cc.egui_ctx.clone(),
                );
                Some(h)
            }
            Err(e) => {
                eprintln!("warning: global hotkey unavailable: {e}");
                None
            }
        };

        let profiles = state.list_profiles();
        let importable = state.list_importable_profiles();
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
            save_error: None,
            force_quit: false,
            window_visible: true,
            profiles,
            importable,
            io_message: None,
            duplicate_name: String::new(),
            confirm_delete: false,
            last_screen: Screen::Home,
            screen_fade_start: None,
            zone_anim: HashMap::new(),
            home_extractor: FeatureExtractor::new(),
        }
    }

    /// Start/stop the microphone to match the listening state, then drain any
    /// capture events. Listening opens the device; pausing (or an error) drops
    /// the worker, which stops the stream via RAII. A capture failure moves the
    /// app into the `Error` state with a specific message (fail fast).
    fn reconcile_capture(&mut self, ctx: &egui::Context, showing: bool) {
        match self.state.listening {
            ListeningState::Listening if self.capture.is_none() => {
                self.live = LiveStatus::default();
                let cap = LiveCapture::start(None, ctx.clone(), CaptureDetail::Standard);
                self.capture = Some(cap);
            }
            ListeningState::Paused | ListeningState::Error if self.capture.is_some() => {
                self.capture = None; // Drop stops the worker + closes the device.
            }
            _ => {}
        }

        let events = self
            .capture
            .as_ref()
            .map(LiveCapture::poll)
            .unwrap_or_default();
        for ev in events {
            match ev {
                CaptureEvent::Started {
                    device_name,
                    sample_rate,
                } => {
                    self.live.device = Some(device_name);
                    self.live.sample_rate = Some(sample_rate);
                }
                CaptureEvent::Level { rms, peak } => {
                    self.live.rms = rms;
                    self.live.peak = peak;
                    // Peak-hold: snap up to the crest, otherwise ease down at
                    // ~0.5/sec (Level arrives ~20 Hz → ~0.025 per sample).
                    self.live.peak_hold = (self.live.peak_hold - 0.025).max(peak);
                }
                CaptureEvent::Onset {
                    window,
                    is_transient,
                    ..
                } => {
                    if is_transient {
                        self.live.taps += 1;
                        // Drive the zone-tile pulse only while Home is the active
                        // screen: the tiles (and their pulse) are painted nowhere
                        // else, so classifying a tap and scheduling a 16 ms repaint
                        // off-Home would animate something invisible — burning
                        // frames against PERFORMANCE.md's redraw discipline. The
                        // tap tally above still updates on every screen, so the
                        // Home counters remain correct when the user returns.
                        if self.state.screen == Screen::Home {
                            // Light the zone the tap landed on (accept pulse) or
                            // shake the nearest one (reject). Visual only — no
                            // action is dispatched here. Use `feature_space`, not
                            // `classify`: `classify` returns zone_id=None whenever
                            // the novelty gate rejects, which discards the nearest
                            // centroid and left the reject-shake branch in
                            // `zone_tiles` unreachable. `feature_space` always
                            // reports every zone's distance plus `accepted`, so the
                            // nearest zone is known even on a reject ("heard a tap
                            // near here, but rejected it").
                            if let Some(profile) = self.state.active_profile.as_ref() {
                                if window.len() == FEATURE_WINDOW_SAMPLES {
                                    let fv = self.home_extractor.extract(&window);
                                    let space = profile.classifier.feature_space(&fv);
                                    if let Some(nearest) = space.zones.iter().min_by(|a, b| {
                                        a.centroid_distance.total_cmp(&b.centroid_distance)
                                    }) {
                                        let dur = if space.accepted {
                                            motion::QUICK
                                        } else {
                                            motion::INSTANT
                                        };
                                        self.zone_anim.insert(
                                            nearest.zone_id,
                                            (AnimState::start(dur), space.accepted),
                                        );
                                    }
                                }
                            }
                        }
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

        // The Home meter only paints on Home while the window is showing.
        // Everywhere else the mic keeps listening (onsets still fire and wake the
        // UI) but its 50 ms level report drives a repaint no one can see, so mute
        // it. Pushed every frame: a screen change, minimize, or restore delivers
        // no discrete event, and this is the single owner of Home-capture
        // visibility (the tray/close handlers only flip `window_visible`).
        if let Some(cap) = &self.capture {
            cap.set_visible(showing && self.state.screen == Screen::Home);
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
                        // `window_visible` alone; capture/wizard visibility is
                        // reconciled from `showing` every frame in `update`.
                        self.window_visible = true;
                    }
                    TrayCommand::Quit => {
                        // Force a real exit. Without this, the close-to-tray
                        // guard below turns this Close into a hide whenever
                        // `minimize_to_tray_on_close` is set (its default), so
                        // "Quit" would only hide the window and never exit.
                        self.force_quit = true;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                }
            }
            // Keep the tray icon in sync however the state changed (hotkey,
            // tray menu, or an in-window button).
            tray.set_state(self.state.listening);
        }

        // Close-to-tray: a user window-close (the X) hides to the tray instead
        // of exiting, when configured and a tray exists to restore from. The
        // tray "Quit" item sets `force_quit` to opt out of this so it always
        // reaches a real exit. Otherwise the close proceeds and the app exits.
        if ctx.input(|i| i.viewport().close_requested())
            && self.state.config.minimize_to_tray_on_close
            && self.tray.is_some()
            && !self.force_quit
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
            // `window_visible` alone; capture/wizard visibility is reconciled
            // from `showing` every frame in `update`.
            self.window_visible = false;
        }
    }
}

impl eframe::App for QwertyApp {
    /// Persist the active profile once on real shutdown.
    ///
    /// A free-text action field commits to disk on focus loss, but exiting
    /// leaves no final frame in which the focused field blurs, so a just-typed
    /// value — already live in the in-memory profile — would never be written.
    /// eframe calls `on_exit` exactly once on a genuine exit and *not* on the
    /// minimize-to-tray `CancelClose`, so it is the one choke point covering
    /// every real quit route (tray "Quit", which forces the close past the
    /// close-to-tray guard, and the window button when not minimizing to tray)
    /// without touching the hot path. `save_active_profile_if_changed` no-ops
    /// when nothing was edited.
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if let Err(e) = self.state.save_active_profile_if_changed() {
            eprintln!("warning: could not save profile on exit: {e}");
        }
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let now = Instant::now();

        // Keep the OS dark/light preference current so `Theme::System` tracks it
        // live. A change while on System simply recomputes effective tokens.
        self.state.system_dark = read_system_dark(ctx);

        // Apply tray/hotkey events and the close-to-tray policy before painting.
        self.handle_external_events(ctx);

        // A window hidden to the tray *or* minimized to the taskbar shows nothing,
        // so no live view may drive the mic meter or a repaint behind it
        // (`PERFORMANCE.md`). `window_visible` tracks the tray hide/show only; the
        // OS title-bar minimize is a separate signal the tray path never sees.
        // Fold both into one "is the window actually showing" truth and let every
        // visibility gate below derive from it — a minimize or restore delivers no
        // discrete event, so each gate is re-evaluated every frame from `showing`.
        let minimized = ctx.input(|i| i.viewport().minimized).unwrap_or(false);
        let showing = self.window_visible && !minimized;

        // The calibration wizard takes over the entire window when active.
        if self.wizard.is_some() {
            self.capture = None; // the wizard owns its own mic
            if self.state.listening == ListeningState::Listening {
                self.state.listening = ListeningState::Paused;
            }
            if let Some(w) = self.wizard.as_mut() {
                w.set_visible(showing);
            }
            let tokens = self.effective_tokens(ctx, now);
            ctx.set_visuals(visuals_for(&tokens));
            let pal = Palette::from_tokens(&tokens);
            let mut outcome = None;
            egui::CentralPanel::default()
                .frame(
                    egui::Frame::default()
                        .fill(pal.base)
                        .inner_margin(style::margin(space::XXL)),
                )
                .show(ctx, |ui| {
                    outcome = self.wizard.as_mut().and_then(|w| w.ui(ui, &pal));
                });
            if let Some(o) = outcome {
                match o {
                    WizardOutcome::Saved(profile) => match self.state.save_profile(&profile) {
                        Ok(()) => {
                            self.state.screen = Screen::Home;
                            self.wizard = None;
                            self.refresh_profiles();
                        }
                        Err(e) => {
                            eprintln!("warning: could not save profile: {e}");
                            // Keep the wizard — its calibration session is intact —
                            // and show the error on its Save step so the user can
                            // retry without redoing the whole wizard. (The shell's
                            // save-error banner can't show here: the wizard owns
                            // the whole window.)
                            if let Some(w) = self.wizard.as_mut() {
                                w.set_save_error(format!("Could not save “{}”: {e}", profile.name));
                            }
                        }
                    },
                    WizardOutcome::Cancelled => self.wizard = None,
                }
            }
            return;
        }

        // An evaluation run owns the mic while it is active, and only on its own
        // screen: navigating away ends it, so the mic and the live repaint both
        // stop (PERFORMANCE.md — the Diagnostics/live views must not keep running
        // in the background). The same rule keeps a second capture stream from
        // fighting Home listening for the device.
        // A hidden window ends the live screens too, exactly as navigating away
        // does: no mic and no 30 fps repaint run behind a tray-hidden window
        // (`PERFORMANCE.md`), and a hidden Evaluation run stops like any
        // navigate-away. On leaving Diagnostics the mic + repaint stop within one
        // frame (`ACCEPTANCE.md`).
        if !showing || self.state.screen != Screen::Evaluation {
            self.evaluation.stop_run();
        }
        if !showing || self.state.screen != Screen::Diagnostics {
            self.diagnostics.stop();
        }

        // Must agree with the stop condition above: a hidden Diagnostics screen
        // stays stopped, or `needs_start()` would restart the mic the same frame
        // it was stopped, churning open/close every frame behind a hidden window.
        let diagnostics_open = showing && self.state.screen == Screen::Diagnostics;
        if self.evaluation.is_running() || diagnostics_open {
            // A screen-owned capture takes the device. Close the Home stream
            // first, then hand it over — never both open at once.
            if self.state.listening == ListeningState::Listening {
                self.state.listening = ListeningState::Paused;
            }
            self.capture = None; // Drop stops the Home worker + closes the device.
                                 // Start the Diagnostics capture only when it has neither a live
                                 // stream nor an already-recorded failure. Checking `needs_start`
                                 // (not just "no live stream") is what stops a failed device from
                                 // being respawned every frame in a silent retry loop — the failure
                                 // stays visible on screen until the user leaves and returns.
            if diagnostics_open && self.diagnostics.needs_start() {
                self.diagnostics.start(ctx);
            }
        } else {
            // Start/stop the mic to match listening state and absorb capture events.
            self.reconcile_capture(ctx, showing);
        }

        // Resolve this frame's effective tokens, advancing any theme cross-fade.
        let tokens = self.effective_tokens(ctx, now);
        ctx.set_visuals(visuals_for(&tokens));
        let pal = Palette::from_tokens(&tokens);

        self.nav_rail(ctx, &pal);

        // Screen-transition fade (`MOTION.md` → Screen transition): when the nav
        // selection changes the incoming content fades in over `motion::STANDARD`
        // rather than hard-cutting. The panel background and rail stay put; only
        // the content Ui's opacity ramps. Repaints only while the fade is live.
        if self.state.screen != self.last_screen {
            self.last_screen = self.state.screen;
            self.screen_fade_start = Some(now);
        }
        let content_opacity = match self.screen_fade_start {
            Some(started_at) => {
                let elapsed = now.saturating_duration_since(started_at);
                if elapsed >= motion::STANDARD {
                    self.screen_fade_start = None;
                    1.0
                } else {
                    ctx.request_repaint();
                    motion::STANDARD_EASE
                        .eval(elapsed.as_secs_f32() / motion::STANDARD.as_secs_f32())
                }
            }
            None => 1.0,
        };

        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .fill(pal.base)
                    .inner_margin(style::margin(space::XL)),
            )
            .show(ctx, |ui| {
                if content_opacity < 1.0 {
                    ui.multiply_opacity(content_opacity);
                }
                self.config_error_banner(ui, &pal);
                self.save_error_banner(ui, &pal);
                match self.state.screen {
                    Screen::Home => self.home(ui, &pal),
                    Screen::Zones => self.zones(ui, &pal),
                    Screen::Diagnostics => self.diagnostics.ui(ui, &pal, &self.state),
                    Screen::Evaluation => self.evaluation.ui(ui, &pal, &self.state),
                    Screen::Settings => self.settings(ui, &pal, now),
                }
            });

        // Keep frames coming while any zone pulse is mid-flight, then drop the
        // completed ones so an idle Home schedules no repaints (`PERFORMANCE.md`
        // redraw discipline). Each pulse was already painted this frame at its
        // current `t`; pruning here means the next frame paints it settled.
        self.zone_anim.retain(|_, (anim, _)| !anim.is_complete());
        if !self.zone_anim.is_empty() {
            ctx.request_repaint_after(Duration::from_millis(16));
        }

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

    /// A dismissible banner shown above the active screen when a profile save
    /// fails. Without it, a persistence error would reach only stderr, leaving
    /// the user believing an edit was stored when it was not — the silent
    /// failure `CLAUDE.md` bans. It holds until dismissed (or cleared by a later
    /// successful save), so it needs no timer and no continuous repaint.
    fn save_error_banner(&mut self, ui: &mut egui::Ui, pal: &Palette) {
        let Some(msg) = self.save_error.clone() else {
            return;
        };
        // Neutral raised fill with a danger border + glyph — not a danger-tinted
        // fill. The error reads from color + glyph + border (DESIGN.md: never
        // color alone), while the heading and message stay in text_primary so
        // they clear WCAG AA. Danger *text* on a danger-tinted fill did not
        // (4.05:1 in Daylight); every color used here is a pairing the theme
        // contrast tests already cover (text on a layer; danger on bg_elevated).
        let inner = egui::Frame::default()
            .fill(pal.elevated)
            .stroke(egui::Stroke::new(1.0_f32, pal.danger))
            .inner_margin(style::margin(space::MD))
            .corner_radius(style::rounding(radius::MD))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("⚠").color(pal.danger).strong());
                    ui.label(
                        egui::RichText::new("Couldn’t save")
                            .color(pal.text)
                            .strong(),
                    );
                    ui.label(egui::RichText::new(&msg).color(pal.text));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .button(egui::RichText::new("✕").color(pal.secondary))
                            .on_hover_text("Dismiss")
                            .clicked()
                        {
                            self.save_error = None;
                        }
                    });
                });
            });
        // A 4px danger accent bar down the banner's left edge (Part 8) — a
        // stronger "this is an error" cue than the thin all-round border alone.
        let r = inner.response.rect;
        ui.painter().rect_filled(
            egui::Rect::from_min_max(r.min, egui::pos2(r.min.x + 4.0, r.max.y)),
            style::rounding(radius::MD),
            pal.danger,
        );
        ui.add_space(space::MD);
    }

    /// A persistent banner shown when `config.json` existed but could not be
    /// loaded (`AppState::config_load_error`). Unlike the save-error banner this
    /// is *not* dismissible: the app is running on in-memory defaults and is
    /// deliberately refusing to overwrite the on-disk file, so hiding the notice
    /// would leave the user believing their settings persist when a save would
    /// be refused. It surfaces the underlying reason and tells the user how to
    /// resolve it — the loud half of the fail-fast fix (the silent overwrite is
    /// prevented in `AppState::save_config`). No timer, no continuous repaint.
    fn config_error_banner(&mut self, ui: &mut egui::Ui, pal: &Palette) {
        let Some(reason) = self.state.config_load_error.clone() else {
            return;
        };
        // Same neutral-raised-fill + danger-border + glyph language as the
        // save-error banner, so the two read as one error vocabulary; text stays
        // in text_primary for WCAG AA on every theme (danger is carried by the
        // glyph and border, never text alone).
        let inner = egui::Frame::default()
            .fill(pal.elevated)
            .stroke(egui::Stroke::new(1.0_f32, pal.danger))
            .inner_margin(style::margin(space::MD))
            .corner_radius(style::rounding(radius::MD))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("⚠").color(pal.danger).strong());
                    ui.vertical(|ui| {
                        ui.label(
                            egui::RichText::new("Your settings can’t be saved")
                                .color(pal.text)
                                .strong(),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "Qwerty couldn’t load your config — {reason}. It’s running on \
                                 defaults and will not overwrite the file. Fix or remove it, then \
                                 restart."
                            ))
                            .color(pal.text),
                        );
                    });
                });
            });
        let r = inner.response.rect;
        ui.painter().rect_filled(
            egui::Rect::from_min_max(r.min, egui::pos2(r.min.x + 4.0, r.max.y)),
            style::rounding(radius::MD),
            pal.danger,
        );
        ui.add_space(space::MD);
    }

    fn nav_rail(&mut self, ctx: &egui::Context, pal: &Palette) {
        egui::SidePanel::left("nav_rail")
            .resizable(false)
            .exact_width(224.0)
            .frame(
                egui::Frame::default()
                    .fill(pal.surface)
                    .inner_margin(style::margin(space::LG)),
            )
            .show(ctx, |ui| {
                // Brand lockup: wordmark + quiet descriptor.
                ui.add_space(space::XS);
                ui.label(
                    egui::RichText::new("Qwerty")
                        .color(pal.text)
                        .strong()
                        .size(text::HEADING),
                );
                ui.label(
                    egui::RichText::new("acoustic tap zones")
                        .color(pal.secondary)
                        .size(text::CAPTION),
                );
                ui.add_space(space::LG);

                for screen in Screen::ALL {
                    let selected = self.state.screen == screen;
                    if nav_item(ui, pal, screen.glyph(), screen.label(), selected) {
                        self.state.screen = screen;
                    }
                    ui.add_space(space::XXS);
                }

                // Pin a compact listening-status pill to the bottom of the rail.
                ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                    status_pill(ui, pal, self.state.listening);
                });
            });
    }

    fn home(&mut self, ui: &mut egui::Ui, pal: &Palette) {
        screen_header(ui, pal, "Home", "Listening status and quick controls");

        // Empty state when there's no calibrated profile yet (Part 6).
        if self.state.active_profile.is_none() {
            if empty_state(
                ui,
                pal,
                "🎯",
                "Nothing mapped yet",
                "Calibrate your desk to map tap zones to actions.",
                "Calibrate now →",
            ) {
                self.launch_wizard = true;
            }
            return;
        }

        // Status card.
        card(ui, pal, |ui| {
            ui.horizontal(|ui| {
                status_pill(ui, pal, self.state.listening);
                ui.add_space(space::MD);
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
                ui.add_space(space::SM);
                ui.label(
                    egui::RichText::new(format!(
                        "Toggle listening from anywhere with {}.",
                        Hotkeys::DEFAULT_LABEL
                    ))
                    .color(pal.secondary)
                    .size(text::CAPTION),
                );
            }

            match self.state.listening {
                ListeningState::Listening => {
                    ui.add_space(space::XS);
                    divider(ui, pal);
                    let mic = self.live.device.as_deref().unwrap_or("opening microphone…");
                    ui.label(
                        egui::RichText::new(format!("Mic: {mic}"))
                            .color(pal.secondary)
                            .size(text::CAPTION),
                    );
                    ui.add_space(space::SM);
                    level_meter(ui, pal, &self.live);
                    ui.add_space(space::SM);
                    ui.horizontal(|ui| {
                        mini_pill(
                            ui,
                            pal,
                            "●",
                            &format!("{} accepted", self.live.taps),
                            pal.success,
                        );
                        ui.add_space(space::SM);
                        mini_pill(
                            ui,
                            pal,
                            "○",
                            &format!("{} ignored", self.live.rejects),
                            pal.secondary,
                        );
                    });

                    // Device-change check: a tap's acoustic signature is
                    // mic-specific, so a classifier fitted on one device is not
                    // trustworthy on another. If the live device differs from the
                    // one this profile was calibrated on, suggest recalibration
                    // rather than silently classifying against a stale model
                    // (`DESIGN.md`: prompt recalibration when the bound mic
                    // changes; honest-accuracy framing).
                    if let (Some(profile), Some(name), Some(sr)) = (
                        self.state.active_profile.as_ref(),
                        self.live.device.as_deref(),
                        self.live.sample_rate,
                    ) {
                        let current = DeviceFingerprint::from_device(name, sr);
                        if profile.needs_recalibration_for(&current) {
                            ui.add_space(space::SM);
                            ui.label(
                                egui::RichText::new(format!(
                                    "⚠ Calibrated for “{}”, but “{name}” is active — \
                                     recalibrate for reliable accuracy.",
                                    profile.device_fingerprint.device_name
                                ))
                                .color(pal.warning),
                            );
                        }
                    }
                }
                ListeningState::Error => {
                    ui.add_space(space::MD);
                    let msg = self.live.error.as_deref().unwrap_or("audio device error");
                    ui.label(egui::RichText::new(format!("⚠ {msg}")).color(pal.danger));
                }
                ListeningState::Paused => {
                    ui.add_space(space::SM);
                    ui.label(
                        egui::RichText::new("Paused — start listening to light up a tap.")
                            .color(pal.secondary)
                            .size(text::CAPTION),
                    );
                }
            }
        });

        // The desk: calibrated zones as a tile grid that pulses on each tap.
        ui.add_space(space::LG);
        if let Some(profile) = self.state.active_profile.as_ref() {
            zone_tiles(ui, pal, &profile.zones, &self.zone_anim);
        }
        ui.add_space(space::MD);
        if primary_button(ui, pal, "Recalibrate / new profile").clicked() {
            self.launch_wizard = true;
        }

        // Nudge toward binding actions: a freshly calibrated profile has zones
        // but no actions, so those zones detect taps yet do nothing until bound.
        if let Some(profile) = &self.state.active_profile {
            let unbound = profile.unbound_zone_count();
            if unbound > 0 {
                ui.add_space(space::LG);
                ui.label(
                    egui::RichText::new(format!(
                        "⚠ Unbound zones: {unbound} of {}. Open Zones & profiles to bind an \
                         action to each.",
                        profile.zones.len()
                    ))
                    .color(pal.warning),
                );
            }
        }

        ui.add_space(space::LG);
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

    /// Re-read the saved-profile summaries and the importable-file list from disk
    /// into their caches. Called at the few points either set changes (startup,
    /// wizard save, switch, export, import), never per frame.
    fn refresh_profiles(&mut self) {
        self.profiles = self.state.list_profiles();
        self.importable = self.state.list_importable_profiles();
        // Any change to the profile set (switch/duplicate/import/delete/wizard
        // save) disarms a pending delete so it can't fall through onto a
        // different active profile than the one the user armed it for.
        self.confirm_delete = false;
    }

    /// Export/import controls for the Zones screen (`DESIGN.md`: "Import / export
    /// a profile"). Export writes the active profile to the `Exports/` folder;
    /// import reads a profile from that folder as a new profile. Failures surface
    /// in the shared save-error banner; a success path/name is shown inline.
    fn profile_io(&mut self, ui: &mut egui::Ui, pal: &Palette) {
        ui.horizontal(|ui| {
            if self.state.active_profile.is_some() && ui.button("Export profile").clicked() {
                match self.state.export_active_profile() {
                    Ok(path) => {
                        self.io_message = Some(format!("Exported to {}", path.display()));
                        self.save_error = None;
                        self.refresh_profiles();
                    }
                    Err(e) => {
                        self.io_message = None;
                        self.save_error = Some(format!("Could not export profile: {e}"));
                    }
                }
            }
        });

        // Duplicate the active profile to reuse its calibration for another
        // context (the Work→Game case): one calibration, a second set of zone
        // bindings. Deferred-apply so the closure only sets a flag (it borrows
        // `duplicate_name` for the field), keeping the mutation out of the
        // closure's borrow.
        if self.state.active_profile.is_some() {
            ui.add_space(space::SM);
            let mut do_duplicate = false;
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Duplicate as")
                        .color(pal.secondary)
                        .size(text::CAPTION),
                );
                ui.add(
                    egui::TextEdit::singleline(&mut self.duplicate_name)
                        .hint_text("new profile name")
                        .desired_width(180.0),
                );
                do_duplicate = ui.button("Duplicate").clicked();
            });
            if do_duplicate {
                let name = self.duplicate_name.clone();
                match self.state.duplicate_active_profile(&name) {
                    Ok(_) => {
                        self.duplicate_name.clear();
                        self.selected_zone = None;
                        self.io_message =
                            Some("Duplicated — rebind this copy's zones to the new context.".into());
                        self.save_error = None;
                        self.refresh_profiles();
                    }
                    Err(e) => {
                        self.io_message = None;
                        self.save_error = Some(format!("Could not duplicate profile: {e}"));
                    }
                }
            }
        }

        if !self.importable.is_empty() {
            ui.add_space(space::SM);
            ui.label(
                egui::RichText::new("Import a profile from the Exports folder")
                    .color(pal.secondary)
                    .size(text::CAPTION),
            );
            // Importing activates someone else's action bindings, which can run
            // apps, commands, and keystrokes on every accepted tap — the same
            // unattended-execution risk DESIGN.md flags a safety note for when a
            // shell/keystroke action is first configured. Warn with a glyph +
            // colour + text (never colour alone), so importing a file from an
            // untrusted source is a considered choice.
            ui.label(
                egui::RichText::new(
                    "⚠ Imported profiles can run apps, commands, and keystrokes on every \
                     tap — only import files you trust.",
                )
                .color(pal.warning)
                .size(text::CAPTION),
            );
            ui.add_space(space::XS);
            let mut to_import: Option<PathBuf> = None;
            ui.horizontal_wrapped(|ui| {
                for (path, name) in &self.importable {
                    if ui.button(format!("Import “{name}”")).clicked() {
                        to_import = Some(path.clone());
                    }
                }
            });
            if let Some(path) = to_import {
                match self.state.import_profile(&path) {
                    Ok(_) => {
                        self.selected_zone = None;
                        self.io_message = Some("Imported a copy as a new profile.".to_string());
                        self.save_error = None;
                        self.refresh_profiles();
                    }
                    Err(e) => {
                        self.io_message = None;
                        self.save_error = Some(format!("Could not import profile: {e}"));
                    }
                }
            }
        }

        // Delete (destructive: removes the profile's calibration and evaluation
        // history). A two-click confirm — obviously destructive, in danger
        // colour — so a stray click can't wipe a calibration (`DESIGN.md`: make
        // destructive actions obviously destructive).
        if self.state.active_profile.is_some() {
            ui.add_space(space::SM);
            if self.confirm_delete {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Delete this profile and its evaluation history?")
                            .color(pal.danger),
                    );
                    if ui
                        .button(egui::RichText::new("Delete").color(pal.danger).strong())
                        .clicked()
                    {
                        let id = self.state.active_profile.as_ref().map(|p| p.id);
                        self.confirm_delete = false;
                        if let Some(id) = id {
                            match self.state.delete_profile(id) {
                                Ok(()) => {
                                    self.selected_zone = None;
                                    self.io_message = Some("Profile deleted.".to_string());
                                    self.save_error = None;
                                    self.refresh_profiles();
                                }
                                Err(e) => {
                                    self.save_error =
                                        Some(format!("Could not delete profile: {e}"));
                                }
                            }
                        }
                    }
                    if ui.button("Cancel").clicked() {
                        self.confirm_delete = false;
                    }
                });
            } else if ui
                .button(egui::RichText::new("Delete profile").color(pal.danger))
                .clicked()
            {
                self.confirm_delete = true;
            }
        }

        if let Some(msg) = &self.io_message {
            ui.add_space(space::XS);
            ui.label(
                egui::RichText::new(msg)
                    .color(pal.secondary)
                    .size(text::CAPTION),
            );
        }
        ui.add_space(space::MD);
    }

    /// A profile switcher for the Zones screen (`DESIGN.md`: "profile switcher"
    /// on Zones & profiles). Only shown when more than one profile exists —
    /// with a single profile there is nothing to switch between and the header
    /// already names it. Switching persists the choice, reloads the profile, and
    /// refreshes the cache; a load/persist failure surfaces in the save-error
    /// banner rather than silently leaving the wrong profile active.
    fn profile_switcher(&mut self, ui: &mut egui::Ui, pal: &Palette) {
        if self.profiles.len() < 2 {
            return;
        }
        let active_name = self
            .state
            .active_profile
            .as_ref()
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "Select a profile".to_string());
        let mut target: Option<qwerty_core::profile::ProfileId> = None;
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Profile").color(pal.secondary));
            egui::ComboBox::from_id_salt("profile_switcher")
                .selected_text(active_name)
                .show_ui(ui, |ui| {
                    for p in &self.profiles {
                        let label = format!(
                            "{}   ·   {} zone{}",
                            p.name,
                            p.zone_count,
                            if p.zone_count == 1 { "" } else { "s" }
                        );
                        if ui.selectable_label(p.is_active, label).clicked() && !p.is_active {
                            target = Some(p.id);
                        }
                    }
                });
        });
        if let Some(id) = target {
            match self.state.switch_profile(id) {
                Ok(()) => {
                    self.selected_zone = None;
                    self.refresh_profiles();
                }
                Err(e) => self.save_error = Some(format!("Could not switch profile: {e}")),
            }
        }
        ui.add_space(space::MD);
    }

    fn zones(&mut self, ui: &mut egui::Ui, pal: &Palette) {
        screen_header(
            ui,
            pal,
            "Zones & profiles",
            "Switch profiles, bind actions to each zone, and share calibrations",
        );
        self.profile_switcher(ui, pal);
        self.profile_io(ui, pal);

        // Disjoint field borrows so the editor can hold the profile + platform
        // at once.
        let Self {
            state,
            action_editor,
            platform,
            selected_zone,
            launch_wizard,
            save_error,
            ..
        } = self;

        let Some(profile) = state.active_profile.as_mut() else {
            if empty_state(
                ui,
                pal,
                "◻",
                "No zones defined",
                "Run calibration to create tap zones and bind actions.",
                "Open calibration",
            ) {
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
            cols[0].add_space(space::SM);
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
                    right.add_space(space::SM);
                    let zid = zone.id;
                    egui::ScrollArea::vertical().show(right, |ui| {
                        changed =
                            action_editor.ui(ui, pal, zid, &mut zone.actions, platform.as_ref());
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
            match state.save_active_profile() {
                Ok(()) => *save_error = None,
                Err(e) => {
                    eprintln!("warning: could not save profile: {e}");
                    *save_error = Some(format!("Could not save your changes: {e}"));
                }
            }
        }
    }

    fn settings(&mut self, ui: &mut egui::Ui, pal: &Palette, now: Instant) {
        screen_header(ui, pal, "Settings", "Appearance, startup, and privacy");

        // Each group is a card, so Settings reads as distinct panels at a glance
        // (DESIGN.md: settings understandable at a glance) rather than a flat run
        // of labels and checkboxes.
        card(ui, pal, |ui| {
            section(ui, pal, "Theme");
            ui.horizontal_wrapped(|ui| {
                for (theme, name) in THEME_CHOICES {
                    let selected = self.state.config.theme == theme;
                    let tokens = tokens_for(theme, self.state.system_dark);
                    if theme_swatch(ui, pal, name, &tokens, selected)
                        && self.state.begin_theme_switch(theme, now)
                    {
                        if let Err(e) = self.state.save_config() {
                            self.save_error = Some(e);
                        }
                    }
                    ui.add_space(space::SM);
                }
            });
            ui.add_space(space::SM);
            ui.label(
                egui::RichText::new("System follows Windows; the rest are fixed themes.")
                    .color(pal.secondary)
                    .size(text::CAPTION),
            );
        });
        ui.add_space(space::MD);

        card(ui, pal, |ui| {
            section(ui, pal, "Startup");
            // "Start with Windows" has no OS startup entry wired behind it in
            // this build, so it is shown disabled rather than as a toggle that
            // silently persists a value nothing acts on (DESIGN.md honest
            // framing; mirrors the Diagnostics "not available in this build"
            // treatment of Active/Hybrid sensing). Registering a real startup
            // entry is deferred to the packaging phase (`ROADMAP.md` Phase 9).
            ui.add_enabled(
                false,
                egui::Checkbox::new(
                    &mut self.state.config.start_with_windows,
                    "Start Qwerty when I sign in",
                ),
            );
            ui.label(
                egui::RichText::new("Not available in this build yet.")
                    .color(pal.secondary)
                    .size(text::CAPTION),
            );
            ui.add_space(space::SM);
            // This one *is* wired — the close-to-tray guard reads it.
            if ui
                .checkbox(
                    &mut self.state.config.minimize_to_tray_on_close,
                    "Minimize to the tray when the window is closed",
                )
                .changed()
            {
                if let Err(e) = self.state.save_config() {
                    self.save_error = Some(e);
                }
            }
        });
        ui.add_space(space::MD);

        card(ui, pal, |ui| {
            section(ui, pal, "Privacy");
            ui.label(
                egui::RichText::new("Everything runs and stays on this machine. No network calls.")
                    .color(pal.text),
            );
            ui.add_space(space::SM);
            // Debug capture is not consumed anywhere in this build — no audio
            // window is ever written to disk — so the control is shown disabled
            // rather than a toggle whose warning claims a write that never
            // happens. The privacy panel above all must not overstate what the
            // app does (DESIGN.md honest framing). When the capture-to-disk path
            // is built, this becomes a live toggle with the persistent
            // "capture is ON" indicator DESIGN.md calls for.
            ui.add_enabled(
                false,
                egui::Checkbox::new(
                    &mut self.state.config.debug_capture_enabled,
                    "Save raw audio windows to disk (debug)",
                ),
            );
            ui.label(
                egui::RichText::new("Not available in this build yet.")
                    .color(pal.secondary)
                    .size(text::CAPTION),
            );
        });
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

/// One navigation row: a full-width, left-aligned pill. The selected row fills
/// with a faint accent tint *and* shows a leading accent bar — position and
/// color, so selection never rides on color alone (`DESIGN.md` → Accessibility).
/// Selection and hover glide rather than snap (`MOTION.md` → Navigation
/// selection): the tint, bar, and label color cross-fade over `motion::QUICK`
/// on selection and `motion::INSTANT` on hover, driven by egui's
/// `animate_bool_with_time`, which schedules a repaint only while a row is
/// mid-transition and stops once it settles (redraw discipline). The row is
/// keyboard-focusable with a visible accent focus ring and carries a
/// `WidgetInfo` so the screen reader announces it as a selectable item with its
/// label and selected state.
fn nav_item(ui: &mut egui::Ui, pal: &Palette, glyph: &str, label: &str, selected: bool) -> bool {
    let height = 36.0;
    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), height), egui::Sense::click());

    // Eased 0→1 progress for selection and (unselected-only) hover. Distinct ids
    // per row so each animates independently; the built-in stops repainting once
    // a value settles, so an idle rail costs no frames (`PERFORMANCE.md`).
    let sel_t = ui.ctx().animate_bool_with_time(
        egui::Id::new(("nav_sel", label)),
        selected,
        motion::QUICK.as_secs_f32(),
    );
    let hover_t = ui.ctx().animate_bool_with_time(
        egui::Id::new(("nav_hover", label)),
        resp.hovered() && !selected,
        motion::INSTANT.as_secs_f32(),
    );

    let painter = ui.painter();
    // Selected tint dominates; hover previews a fainter tint of the same accent.
    let tint = 0.16 * sel_t + 0.07 * hover_t;
    if tint > 0.001 {
        painter.rect_filled(
            rect,
            style::rounding(radius::SM),
            mix(pal.surface, pal.accent, tint),
        );
    }
    // Leading accent bar grows from the vertical center as the row is selected.
    // A vertical gradient — full accent at the centre, fading to transparent at
    // both ends — reads as a soft glowing indicator rather than a hard rect.
    if sel_t > 0.001 {
        let bar_h = height * sel_t;
        let (x, w) = (rect.min.x, 3.0);
        let (mid, half) = (rect.center().y, bar_h * 0.5);
        let faded = with_alpha(pal.accent, 0);
        paint::vertical_gradient(
            painter,
            egui::Rect::from_min_max(egui::pos2(x, mid - half), egui::pos2(x + w, mid)),
            faded,
            pal.accent,
        );
        paint::vertical_gradient(
            painter,
            egui::Rect::from_min_max(egui::pos2(x, mid), egui::pos2(x + w, mid + half)),
            pal.accent,
            faded,
        );
    }
    if resp.has_focus() {
        painter.rect_stroke(
            rect,
            style::rounding(radius::SM),
            egui::Stroke::new(2.0_f32, pal.accent),
            egui::StrokeKind::Inside,
        );
    }
    // Label color cross-fades text → accent as the row becomes selected.
    painter.text(
        egui::pos2(rect.min.x + space::MD, rect.center().y),
        egui::Align2::LEFT_CENTER,
        format!("{glyph}   {label}"),
        egui::FontId::proportional(text::BODY),
        mix(pal.text, pal.accent, sel_t),
    );
    resp.widget_info(|| {
        egui::WidgetInfo::selected(egui::WidgetType::SelectableLabel, true, selected, label)
    });
    resp.clicked()
}

/// A status pill combining a color *and* a glyph *and* text — never color alone
/// (`DESIGN.md` → Accessibility).
fn status_pill(ui: &mut egui::Ui, pal: &Palette, state: ListeningState) {
    let (color, glyph) = match state {
        ListeningState::Listening => (pal.success, "●"),
        ListeningState::Paused => (pal.secondary, "❚❚"),
        ListeningState::Error => (pal.danger, "▲"),
    };
    // A real rounded badge: a faint status-tinted fill and edge, the status
    // colour on the glyph, the label in primary text (so it stays legible on
    // any theme — the colour is carried by the glyph, never text alone).
    egui::Frame::default()
        .fill(mix(pal.surface, color, 0.14))
        .stroke(egui::Stroke::new(1.0_f32, mix(pal.surface, color, 0.45)))
        .corner_radius(style::rounding(radius::MD))
        .inner_margin(egui::Margin::symmetric(space::MD as i8, space::XS as i8))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = space::SM;
                ui.label(egui::RichText::new(glyph).color(color).strong());
                ui.label(egui::RichText::new(state.label()).color(pal.text).strong());
            });
        });
}

/// A screen's header: a large title plus a one-line subtitle in secondary text,
/// then a block of space. Every top-level screen uses it, so they share one
/// hierarchy (a bare heading over a wall of controls is the "plain screen" look
/// this replaces). `pub(crate)` so the Diagnostics/Evaluation screens share it.
pub(crate) fn screen_header(ui: &mut egui::Ui, pal: &Palette, title: &str, subtitle: &str) {
    ui.heading(egui::RichText::new(title).color(pal.text).size(text::TITLE));
    ui.add_space(space::XXS);
    ui.label(
        egui::RichText::new(subtitle)
            .color(pal.secondary)
            .size(text::BODY),
    );
    ui.add_space(space::LG);
}

/// A titled section label. `pub(crate)` so every screen shares one definition
/// (the Settings, Evaluation, and Diagnostics screens all use it).
pub(crate) fn section(ui: &mut egui::Ui, pal: &Palette, title: &str) {
    ui.label(
        egui::RichText::new(title)
            .color(pal.text)
            .strong()
            .size(text::SECTION),
    );
    ui.add_space(space::SM);
}

/// Linear blend between two colors, `t` in `[0, 1]` (0 = `a`, 1 = `b`).
/// `pub(crate)` so the heat-grid views (confusion matrix, spectrogram) tint from
/// one shared helper rather than each re-deriving it.
pub(crate) fn mix(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    let lerp = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    egui::Color32::from_rgb(lerp(a.r(), b.r()), lerp(a.g(), b.g()), lerp(a.b(), b.b()))
}

/// A simple elevated card frame. `pub(crate)` so the action editor reuses it.
pub(crate) fn card(ui: &mut egui::Ui, pal: &Palette, add: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::default()
        .fill(pal.elevated)
        .stroke(egui::Stroke::new(1.0_f32, pal.border))
        .inner_margin(style::margin(space::LG))
        .corner_radius(style::rounding(radius::MD))
        // A soft drop shadow lifts the card off the base for real depth (a
        // larger blur + downward offset than a hairline, so cards read as
        // stacked planes). On dark themes the border still does most of the
        // separating — a black shadow barely shows on a near-black base — so
        // this stays a moderate lift rather than a heavy, theme-aware drop.
        .shadow(egui::Shadow {
            offset: [0, 4],
            blur: 16,
            spread: 0,
            color: egui::Color32::from_black_alpha(40),
        })
        .show(ui, add);
}

/// `color` with an explicit alpha byte, for painted glows/needles/tints.
/// `pub(crate)` so the calibration wizard's painted stepper/pips/desk diagram
/// tint from the same helper rather than re-deriving it.
pub(crate) fn with_alpha(c: egui::Color32, a: u8) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a)
}

/// A custom-painted 1px horizontal rule with breathing room, replacing
/// `ui.separator()` (Part 8) — a calm divider rather than egui's default line.
/// `pub(crate)` so the calibration wizard's header rule matches every screen's.
pub(crate) fn divider(ui: &mut egui::Ui, pal: &Palette) {
    ui.add_space(space::SM);
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 1.0), egui::Sense::hover());
    ui.painter()
        .hline(rect.x_range(), rect.center().y, egui::Stroke::new(1.0_f32, pal.border));
    ui.add_space(space::SM);
}

/// A centered empty-state column: a large quiet glyph, a title, one line of
/// context, and a primary call-to-action (Part 6). Returns whether the CTA was
/// clicked so the caller can route the fix.
fn empty_state(
    ui: &mut egui::Ui,
    pal: &Palette,
    glyph: &str,
    title: &str,
    body: &str,
    cta: &str,
) -> bool {
    ui.add_space(space::XXL);
    ui.vertical_centered(|ui| {
        ui.label(egui::RichText::new(glyph).size(64.0).color(pal.border));
        ui.add_space(space::MD);
        ui.label(
            egui::RichText::new(title)
                .size(text::HEADING)
                .color(pal.text)
                .strong(),
        );
        ui.add_space(space::XS);
        ui.label(
            egui::RichText::new(body)
                .size(text::BODY)
                .color(pal.secondary),
        );
        ui.add_space(space::LG);
        primary_button(ui, pal, cta).clicked()
    })
    .inner
}

/// An empty-state column with no call-to-action, for screens rendered from a
/// read-only `&AppState` that can't route a fix from where they sit. Same calm,
/// centered layout as [`empty_state`] minus the button (Part 6). `pub(crate)`
/// so the Evaluation/Diagnostics screens share it.
pub(crate) fn empty_state_note(ui: &mut egui::Ui, pal: &Palette, glyph: &str, title: &str, body: &str) {
    ui.add_space(space::XXL);
    ui.vertical_centered(|ui| {
        ui.label(egui::RichText::new(glyph).size(64.0).color(pal.border));
        ui.add_space(space::MD);
        ui.label(
            egui::RichText::new(title)
                .size(text::HEADING)
                .color(pal.text)
                .strong(),
        );
        ui.add_space(space::XS);
        ui.label(
            egui::RichText::new(body)
                .size(text::BODY)
                .color(pal.secondary),
        );
    });
}

/// A custom input-level meter (Part 3): a background track, an RMS fill, a soft
/// clip-warning zone when hot, and a decaying peak needle. Reads like an audio
/// app's input meter, not a loading bar — no text is overlaid on it.
fn level_meter(ui: &mut egui::Ui, pal: &Palette, live: &LiveStatus) {
    let (resp, painter) = ui.allocate_painter(
        egui::vec2(ui.available_width().min(420.0), 20.0),
        egui::Sense::hover(),
    );
    let r = resp.rect;
    let round = style::rounding(radius::XS);
    painter.rect_filled(r, round, pal.surface);

    // RMS fill: a horizontal gradient the GPU blends across the level — healthy
    // accent → brighter accent → warning (hot) → danger (clipping). The fill
    // width is the RMS fraction (sqrt-shaped so quiet input stays visible), but
    // the gradient is laid out in full-track space and *clipped* to that width,
    // so the fill's right-edge colour reflects how hot the signal actually is.
    // Reads like an audio app's meter, not a one-colour loading bar.
    let rms_frac = live.rms.clamp(0.0, 1.0).sqrt();
    if rms_frac > 0.0 {
        let fill = egui::Rect::from_min_size(r.min, egui::vec2(r.width() * rms_frac, r.height()));
        let bright = mix(pal.accent, egui::Color32::WHITE, 0.28);
        let stops = [
            (0.0, pal.accent),
            (0.6, bright),
            (0.75, pal.warning),
            (0.9, pal.warning),
            (1.0, pal.danger),
        ];
        paint::horizontal_gradient(&painter.with_clip_rect(fill), r, &stops);
    }

    let peak_frac = live.peak_hold.clamp(0.0, 1.0).sqrt();
    // Soft clip-warning zone (rightmost 7%) — a warning, not a hard color flip.
    if peak_frac > 0.93 {
        let warn =
            egui::Rect::from_min_max(egui::pos2(r.min.x + r.width() * 0.93, r.min.y), r.max);
        painter.rect_filled(warn, round, with_alpha(pal.danger, 60));
    }
    // Peak needle.
    let x = r.min.x + r.width() * peak_frac;
    painter.vline(x, r.y_range(), egui::Stroke::new(2.0_f32, with_alpha(pal.text, 200)));

    // Announce the meter to screen readers with a *stable* label — a live
    // percentage here would make Narrator chatter (the value changes ~20×/s, as
    // Level events arrive every 50 ms). The ProgressIndicator role conveys it is
    // a level readout.
    resp.widget_info(|| {
        egui::WidgetInfo::labeled(
            egui::WidgetType::ProgressIndicator,
            true,
            "Microphone input level",
        )
    });
}

/// A theme picker swatch (Part 7): the theme's own surface as the fill, an
/// accent stripe along the bottom, a text-contrast dot, and its name beneath —
/// a design-tool color chip rather than a dropdown row. Selected gets an accent
/// border; hover adds a faint accent tint; keyboard focus draws a ring. Returns
/// whether it was clicked.
fn theme_swatch(ui: &mut egui::Ui, pal: &Palette, name: &str, tokens: &Tokens, selected: bool) -> bool {
    ui.vertical(|ui| {
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(80.0, 56.0), egui::Sense::click());
        let hover = ui
            .ctx()
            .animate_bool_with_time(resp.id, resp.hovered(), motion::INSTANT.as_secs_f32());
        let round = style::rounding(radius::SM);
        let painter = ui.painter();
        painter.rect_filled(rect, round, c32(tokens.bg_surface));
        if hover > 0.01 {
            painter.rect_filled(rect, round, with_alpha(c32(tokens.accent), (hover * 32.0) as u8));
        }
        // Accent stripe across the bottom.
        let stripe = egui::Rect::from_min_max(egui::pos2(rect.min.x, rect.max.y - 8.0), rect.max);
        painter.rect_filled(stripe, round, c32(tokens.accent));
        // Text-contrast dot, center-top.
        painter.circle_filled(
            egui::pos2(rect.center().x, rect.min.y + 14.0),
            4.0,
            c32(tokens.text_primary),
        );
        let (bw, bc) = if selected {
            (2.0_f32, pal.accent)
        } else {
            (1.0_f32, pal.border)
        };
        painter.rect_stroke(rect, round, egui::Stroke::new(bw, bc), egui::StrokeKind::Inside);
        if resp.has_focus() {
            painter.rect_stroke(
                rect.expand(2.0),
                round,
                egui::Stroke::new(2.0_f32, pal.accent),
                egui::StrokeKind::Outside,
            );
        }
        ui.add_space(space::XXS);
        ui.label(
            egui::RichText::new(name)
                .size(text::CAPTION)
                .color(if selected { pal.text } else { pal.secondary }),
        );
        // Announce the swatch to screen readers: it is a focusable radio-style
        // choice, so Narrator must read its name and selected state (the painted
        // `name` label below is a separate non-focusable node, not tied to the
        // focusable swatch response). Mirrors nav_item's WidgetInfo::selected.
        resp.widget_info(|| {
            egui::WidgetInfo::selected(egui::WidgetType::RadioButton, true, selected, name)
        });
        resp.clicked()
    })
    .inner
}

/// A compact counter pill (the small sibling of `status_pill`) for the
/// accepted/ignored tap tallies (Part 3).
fn mini_pill(ui: &mut egui::Ui, pal: &Palette, glyph: &str, label: &str, color: egui::Color32) {
    egui::Frame::default()
        .fill(mix(pal.surface, color, 0.14))
        .stroke(egui::Stroke::new(1.0_f32, mix(pal.surface, color, 0.4)))
        .corner_radius(style::rounding(radius::MD))
        .inner_margin(egui::Margin::symmetric(space::SM as i8, space::XXS as i8))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = space::XS;
                ui.label(egui::RichText::new(glyph).color(color).size(text::CAPTION));
                ui.label(egui::RichText::new(label).color(pal.text).size(text::CAPTION));
            });
        });
}

/// The calibrated zones as a grid of tiles (Parts 2 + 3). Each tile shows its
/// label + action count; on a classified tap it pulses — an accepted tap scales
/// up 6% with a fading accent glow ring and a fill flash (`EMPHASIZED_EASE`), a
/// rejected tap shakes horizontally with a danger-tint flash. Only the *painted*
/// rect animates; the allocated layout is fixed, so the grid never reflows.
fn zone_tiles(
    ui: &mut egui::Ui,
    pal: &Palette,
    zones: &[Zone],
    zone_anim: &HashMap<ZoneId, (AnimState, bool)>,
) {
    if zones.is_empty() {
        return;
    }
    let cols = match zones.len() {
        0..=4 => 2,
        5..=6 => 3,
        _ => 4,
    };
    let rows = zones.len().div_ceil(cols);
    let gap = space::MD;
    let avail_w = ui.available_width().min(560.0);
    let tile_w = (avail_w - gap * (cols as f32 - 1.0)) / cols as f32;
    let tile_h = tile_w * 0.6;
    let total_h = tile_h * rows as f32 + gap * (rows as f32 - 1.0).max(0.0);
    let (resp, painter) =
        ui.allocate_painter(egui::vec2(avail_w, total_h), egui::Sense::hover());
    let origin = resp.rect.min;
    let round = style::rounding(radius::MD);

    for (i, zone) in zones.iter().enumerate() {
        let (c, row) = (i % cols, i / cols);
        let base = egui::Rect::from_min_size(
            egui::pos2(
                origin.x + c as f32 * (tile_w + gap),
                origin.y + row as f32 * (tile_h + gap),
            ),
            egui::vec2(tile_w, tile_h),
        );

        // Derive the animated properties from this zone's pulse, if any.
        // `bloom` (0→1→0 arc) drives a soft radial halo behind an accepted tile.
        let (scale, glow, flash, shake, accept, bloom) = match zone_anim.get(&zone.id) {
            Some((anim, true)) => {
                // Accept: a 0→1→0 arc so scale/flash/bloom rise then settle.
                let arc = (anim.t(motion::EMPHASIZED_EASE) * std::f32::consts::PI).sin();
                (1.0 + 0.06 * arc, (1.0 - anim.linear()) * 0.7, 0.10 * arc, 0.0, true, arc)
            }
            Some((anim, false)) => {
                // Reject: quick decaying horizontal shake + danger flash.
                let lin = anim.linear();
                let shake = 3.0 * (lin * 4.0 * std::f32::consts::PI).sin() * (1.0 - lin);
                (1.0, 0.0, 0.10 * (1.0 - lin), shake, false, 0.0)
            }
            None => (1.0, 0.0, 0.0, 0.0, true, 0.0),
        };

        let rect = egui::Rect::from_center_size(
            base.center() + egui::vec2(shake, 0.0),
            base.size() * scale,
        );
        // Soft radial halo behind an accepting tile (painted first so the
        // opaque tile fill masks the core and only the surrounding bloom shows
        // in the gap between tiles). Layered 3-pass mesh bloom, no shader.
        if bloom > 0.001 {
            let radius = rect.size().length() * 0.5 * 1.8;
            paint::radial_bloom(&painter, rect.center(), radius, pal.accent, bloom * 0.8);
        }
        let flash_color = if accept { pal.accent } else { pal.danger };
        painter.rect_filled(rect, round, mix(pal.elevated, flash_color, flash));
        painter.rect_stroke(
            rect,
            round,
            egui::Stroke::new(1.0_f32, pal.border),
            egui::StrokeKind::Inside,
        );
        if glow > 0.001 {
            painter.rect_stroke(
                rect.expand(3.0),
                round,
                egui::Stroke::new(2.0_f32, with_alpha(pal.accent, (glow * 255.0) as u8)),
                egui::StrokeKind::Outside,
            );
        }
        painter.text(
            egui::pos2(rect.center().x, rect.center().y - 9.0),
            egui::Align2::CENTER_CENTER,
            &zone.label,
            egui::FontId::proportional(text::BODY),
            pal.text,
        );
        let n = zone.actions.len();
        let sub = if n == 0 {
            "no action".to_string()
        } else {
            format!("{n} action{}", if n == 1 { "" } else { "s" })
        };
        painter.text(
            egui::pos2(rect.center().x, rect.center().y + 11.0),
            egui::Align2::CENTER_CENTER,
            sub,
            egui::FontId::proportional(text::CAPTION),
            pal.secondary,
        );
    }
    // Screen-reader summary: the tiles are painted pixels, invisible to AT, so
    // announce each zone's name and action count on the grid's response. This
    // single node changes only when the profile changes (not per animation
    // frame), so it does not churn Narrator.
    resp.widget_info(|| {
        egui::WidgetInfo::labeled(
            egui::WidgetType::Label,
            true,
            zones
                .iter()
                .map(|z| {
                    let n = z.actions.len();
                    format!("{}: {} action{}", z.label, n, if n == 1 { "" } else { "s" })
                })
                .collect::<Vec<_>>()
                .join(", "),
        )
    });
}

/// A filled-accent primary button — the one unmistakable call to action on a
/// screen (`DESIGN.md`: "Make primary actions unmistakable"). Its label is
/// painted in `on_color(accent)`, so it clears WCAG AA on every theme's accent
/// (proved by the theme `on_accent_ink_clears_aa_for_every_theme` test) rather
/// than assuming a fixed light-on-dark that breaks on the yellow High-Contrast
/// or teal Aurora accents. `pub(crate)` so the Home and Zones calibrate buttons
/// share exactly one primary-CTA shape.
pub(crate) fn primary_button(ui: &mut egui::Ui, pal: &Palette, label: &str) -> egui::Response {
    let ink = c32(on_color(Color::rgb(pal.accent.r(), pal.accent.g(), pal.accent.b())));
    ui.add(
        egui::Button::new(egui::RichText::new(label).color(ink).strong())
            .fill(pal.accent)
            .corner_radius(style::rounding(radius::SM))
            .min_size(egui::vec2(0.0, 34.0)),
    )
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

/// Spawn the single reader for one process-global event channel.
///
/// `tray-icon` and `global-hotkey` each deliver events through one
/// process-global crossbeam channel, and a crossbeam message goes to exactly
/// one consumer — so each channel must have exactly one reader. This thread is
/// it: it blocks on `recv_one`, forwards every event into `tx` (an app-owned
/// channel drained by `Tray`/`Hotkeys`), and wakes the UI. (Previously a
/// wake-thread `recv()` and the UI-thread `try_recv()` read the *same* global
/// channel, so the parked wake-thread won every event and the UI never saw a
/// tray/hotkey command.) Spawned only *after* its consumer is constructed, so a
/// failed `Tray`/`Hotkeys` never leaves a reader parked forever on the global.
fn spawn_forward<T, F>(tx: std::sync::mpsc::Sender<T>, mut recv_one: F, ctx: egui::Context)
where
    T: Send + 'static,
    F: FnMut() -> Option<T> + Send + 'static,
{
    std::thread::spawn(move || {
        while let Some(ev) = recv_one() {
            if tx.send(ev).is_err() {
                break; // the UI dropped its receiver — nothing left to wake
            }
            ctx.request_repaint();
        }
    });
}

/// Install the native Windows UI typeface (Segoe UI) as egui's proportional
/// font, so the app uses the platform-native face `DESIGN.md` specifies instead
/// of egui's bundled default. Best-effort: if no candidate file is present
/// (a non-Windows host, or a stripped install) egui keeps its default — a
/// missing *cosmetic* font is not a precondition worth failing launch over, so
/// this is the one place a quiet fallback is correct rather than a hidden bug.
/// The static `segoeui.ttf` is tried before the variable `SegoeUIVF.ttf`
/// because `ab_glyph` (egui's rasterizer) renders static faces reliably.
fn install_fonts(ctx: &egui::Context) {
    let candidates = [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\SegoeUIVF.ttf",
    ];
    let Some(bytes) = candidates.iter().find_map(|p| std::fs::read(p).ok()) else {
        return;
    };
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "segoe".to_owned(),
        std::sync::Arc::new(egui::FontData::from_owned(bytes)),
    );
    // First in the Proportional family so it wins; egui's default stays as a
    // fallback for any glyph Segoe UI lacks. Monospace is left untouched (the
    // Diagnostics/confusion-matrix numerics rely on it).
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "segoe".to_owned());
    ctx.set_fonts(fonts);
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

    // Inactive controls carry a raised elevated fill + border, so a button
    // reads as a tactile surface rather than flat text on the panel.
    v.widgets.inactive.bg_fill = c32(t.bg_elevated);
    v.widgets.inactive.weak_bg_fill = c32(t.bg_elevated);
    v.widgets.inactive.bg_stroke.color = c32(t.border);
    v.widgets.inactive.fg_stroke.color = c32(t.text_primary);
    v.widgets.inactive.expansion = 0.0;

    // Hover: an accent-tinted fill, an accent edge, and a hair of growth — the
    // control visibly lifts toward the pointer (egui animates `expansion`).
    let hover_fill = c32(Color::lerp(t.bg_elevated, t.accent, 0.10));
    v.widgets.hovered.bg_fill = hover_fill;
    v.widgets.hovered.weak_bg_fill = hover_fill;
    v.widgets.hovered.bg_stroke.color = c32(t.accent);
    v.widgets.hovered.fg_stroke.color = c32(t.text_primary);
    v.widgets.hovered.expansion = 1.5;

    // Active / pressed: accent fill, with the label in `on_color(accent)` so it
    // stays legible on every theme's accent, and the growth released (presses
    // "in" rather than staying lifted).
    v.widgets.active.bg_fill = c32(t.accent);
    v.widgets.active.weak_bg_fill = c32(t.accent);
    v.widgets.active.bg_stroke.color = c32(t.accent);
    v.widgets.active.fg_stroke.color = c32(on_color(t.accent));
    v.widgets.active.expansion = 0.0;

    // Selection (and thus selectable_label highlight) uses a translucent accent.
    let a = t.accent;
    v.selection.bg_fill = egui::Color32::from_rgba_unmultiplied(a.r, a.g, a.b, 90);
    v.selection.stroke.color = c32(t.accent);

    // Roundness (radius::SM) applied to every widget state in one place, so no
    // control keeps egui's default corner radius; menus/popovers/windows take
    // the softer card radius (radius::MD). This is the geometry half of the
    // design system meeting egui, alongside the spacing set once at startup.
    let control = style::rounding(radius::SM);
    for w in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        w.corner_radius = control;
    }
    v.window_corner_radius = style::rounding(radius::MD);
    v.menu_corner_radius = style::rounding(radius::MD);

    // Subtle elevation: menus, popovers, and windows lift off the surface with a
    // soft shadow (DESIGN.md: "real depth from subtle elevation rather than
    // gradients"). Dark themes need more alpha for a shadow to read against a
    // near-black base than light themes do.
    let shadow_color = if dark {
        egui::Color32::from_black_alpha(130)
    } else {
        egui::Color32::from_black_alpha(48)
    };
    v.window_shadow = egui::Shadow {
        offset: [0, 6],
        blur: 18,
        spread: 0,
        color: shadow_color,
    };
    v.popup_shadow = egui::Shadow {
        offset: [0, 4],
        blur: 12,
        spread: 0,
        color: shadow_color,
    };

    v
}
