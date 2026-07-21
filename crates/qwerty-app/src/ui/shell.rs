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

use qwerty_core::profile::Theme;

use crate::app_state::{AppState, ListeningState, Screen};
use crate::ui::motion;
use crate::ui::theme::{tokens_for, Color, Tokens};

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
}

impl QwertyApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let system_dark = read_system_dark(&cc.egui_ctx);
        let state = AppState::load(system_dark);
        // Paint the correct theme on the very first frame — no flash of the
        // egui default palette (`PERFORMANCE.md`: no flash of unstyled content).
        let tokens = tokens_for(state.config.theme, system_dark);
        cc.egui_ctx.set_visuals(visuals_for(&tokens));
        Self { state }
    }
}

impl eframe::App for QwertyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let now = Instant::now();

        // Keep the OS dark/light preference current so `Theme::System` tracks it
        // live. A change while on System simply recomputes effective tokens.
        self.state.system_dark = read_system_dark(ctx);

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
                Screen::Zones => placeholder(ui, &pal, "Zones & profiles", "Phase 4–5"),
                Screen::Diagnostics => placeholder(ui, &pal, "Diagnostics", "Phase 6"),
                Screen::Evaluation => placeholder(ui, &pal, "Evaluation", "Phase 6"),
                Screen::Settings => self.settings(ui, &pal, now),
            });
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
        ui.label(
            egui::RichText::new("No profile yet — calibration arrives in Phase 4.")
                .color(pal.secondary),
        );
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

fn placeholder(ui: &mut egui::Ui, pal: &Palette, title: &str, phase: &str) {
    ui.heading(egui::RichText::new(title).color(pal.text).size(26.0));
    ui.add_space(8.0);
    ui.label(egui::RichText::new(format!("This screen arrives in {phase}.")).color(pal.secondary));
}

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

/// A simple elevated card frame.
fn card(ui: &mut egui::Ui, pal: &Palette, add: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::default()
        .fill(pal.elevated)
        .stroke(egui::Stroke::new(1.0_f32, pal.border))
        .inner_margin(egui::Margin::same(16))
        .corner_radius(egui::CornerRadius::same(8))
        .show(ui, add);
}

/// egui `Color32` versions of the semantic tokens, for painting this frame.
struct Palette {
    base: egui::Color32,
    surface: egui::Color32,
    elevated: egui::Color32,
    text: egui::Color32,
    secondary: egui::Color32,
    border: egui::Color32,
    accent: egui::Color32,
    success: egui::Color32,
    warning: egui::Color32,
    danger: egui::Color32,
}

impl Palette {
    fn from_tokens(t: &Tokens) -> Self {
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

fn c32(c: Color) -> egui::Color32 {
    egui::Color32::from_rgb(c.r, c.g, c.b)
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
