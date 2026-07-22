//! The per-zone action editor (Phase 5, `DESIGN.md` → Zones & profiles).
//!
//! Edits one zone's `Vec<ActionSpec>` (the macro chain): add, change type, edit
//! fields, remove, reorder, and a per-action "Test" button that fires the action
//! through the real [`PlatformActions`] backend. `ui` returns `true` on a
//! committed change so the shell can persist the profile (not per keystroke —
//! free-text fields commit on focus loss).

use egui::{Button, ComboBox, RichText};

use qwerty_core::profile::{ActionSpec, Modifier, ScreenshotMode, SystemSound, ZoneId};

use crate::platform::{dispatch, PlatformActions};
use crate::ui::shell::{card, Palette};

/// Transient editor state: the last Test result, scoped to the zone + row it
/// was run on so it never bleeds onto a different zone's row.
#[derive(Default)]
pub struct ActionEditor {
    last_test: Option<(ZoneId, usize, Result<(), String>)>,
}

/// The action variants as a pickable list — an exhaustive mirror of
/// [`ActionSpec`], so adding a variant there forces the arms here to update.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    OpenApp,
    OpenPath,
    OpenUrl,
    RunCommand,
    CopyToClipboard,
    Speak,
    PlaySound,
    ShowToast,
    SendKeystroke,
    Screenshot,
    VisualOnly,
}

impl Kind {
    const ALL: [Kind; 11] = [
        Kind::OpenApp,
        Kind::OpenPath,
        Kind::OpenUrl,
        Kind::RunCommand,
        Kind::CopyToClipboard,
        Kind::Speak,
        Kind::PlaySound,
        Kind::ShowToast,
        Kind::SendKeystroke,
        Kind::Screenshot,
        Kind::VisualOnly,
    ];

    fn label(self) -> &'static str {
        match self {
            Kind::OpenApp => "Open app",
            Kind::OpenPath => "Open file / folder",
            Kind::OpenUrl => "Open URL",
            Kind::RunCommand => "Run command",
            Kind::CopyToClipboard => "Copy to clipboard",
            Kind::Speak => "Speak text",
            Kind::PlaySound => "Play sound",
            Kind::ShowToast => "Show notification",
            Kind::SendKeystroke => "Send keystroke",
            Kind::Screenshot => "Screenshot",
            Kind::VisualOnly => "Visual only",
        }
    }

    fn of(spec: &ActionSpec) -> Kind {
        match spec {
            ActionSpec::OpenApp { .. } => Kind::OpenApp,
            ActionSpec::OpenPath { .. } => Kind::OpenPath,
            ActionSpec::OpenUrl { .. } => Kind::OpenUrl,
            ActionSpec::RunCommand { .. } => Kind::RunCommand,
            ActionSpec::CopyToClipboard { .. } => Kind::CopyToClipboard,
            ActionSpec::Speak { .. } => Kind::Speak,
            ActionSpec::PlaySound { .. } => Kind::PlaySound,
            ActionSpec::ShowToast { .. } => Kind::ShowToast,
            ActionSpec::SendKeystroke { .. } => Kind::SendKeystroke,
            ActionSpec::Screenshot { .. } => Kind::Screenshot,
            ActionSpec::VisualOnly => Kind::VisualOnly,
        }
    }

    /// A fresh empty value of this kind (used when the user switches type).
    fn default_spec(self) -> ActionSpec {
        match self {
            Kind::OpenApp => ActionSpec::OpenApp { path: String::new() },
            Kind::OpenPath => ActionSpec::OpenPath { path: String::new() },
            Kind::OpenUrl => ActionSpec::OpenUrl { url: String::new() },
            Kind::RunCommand => ActionSpec::RunCommand {
                command: String::new(),
                args: Vec::new(),
            },
            Kind::CopyToClipboard => ActionSpec::CopyToClipboard { text: String::new() },
            Kind::Speak => ActionSpec::Speak { text: String::new() },
            Kind::PlaySound => ActionSpec::PlaySound {
                sound: SystemSound::Default,
            },
            Kind::ShowToast => ActionSpec::ShowToast {
                title: String::new(),
                body: String::new(),
            },
            Kind::SendKeystroke => ActionSpec::SendKeystroke {
                combo: qwerty_core::profile::KeyCombo {
                    modifiers: Vec::new(),
                    key: String::new(),
                },
            },
            Kind::Screenshot => ActionSpec::Screenshot {
                mode: ScreenshotMode::FullScreen,
            },
            Kind::VisualOnly => ActionSpec::VisualOnly,
        }
    }
}

/// A structural edit deferred until after the iteration (never mutate the Vec
/// mid-loop).
enum Op {
    Remove(usize),
    MoveUp(usize),
    MoveDown(usize),
}

impl ActionEditor {
    /// Render the editor for `actions`. Returns `true` if a committed change was
    /// made this frame (caller persists).
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        pal: &Palette,
        zone_id: ZoneId,
        actions: &mut Vec<ActionSpec>,
        platform: &dyn PlatformActions,
    ) -> bool {
        let mut changed = false;
        let mut op: Option<Op> = None;
        let mut test_req: Option<usize> = None;
        let n = actions.len();

        if actions.is_empty() {
            ui.label(
                RichText::new("No actions yet — add one below.")
                    .color(pal.secondary)
                    .italics(),
            );
        }

        for (i, spec) in actions.iter_mut().enumerate() {
            let row_status = self
                .last_test
                .as_ref()
                .and_then(|(zid, ti, r)| (*zid == zone_id && *ti == i).then_some(r));
            ui.push_id(i, |ui| {
                card(ui, pal, |ui| {
                    ui.horizontal(|ui| {
                        let mut kind = Kind::of(spec);
                        ComboBox::from_id_salt("kind")
                            .selected_text(kind.label())
                            .show_ui(ui, |ui| {
                                for k in Kind::ALL {
                                    ui.selectable_value(&mut kind, k, k.label());
                                }
                            });
                        if kind != Kind::of(spec) {
                            *spec = kind.default_spec();
                            changed = true;
                        }

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("Remove").clicked() {
                                op = Some(Op::Remove(i));
                            }
                            if ui.add_enabled(i + 1 < n, Button::new("↓")).clicked() {
                                op = Some(Op::MoveDown(i));
                            }
                            if ui.add_enabled(i > 0, Button::new("↑")).clicked() {
                                op = Some(Op::MoveUp(i));
                            }
                            if ui.button("Test").clicked() {
                                test_req = Some(i);
                            }
                            match row_status {
                                Some(Ok(())) => {
                                    ui.label(RichText::new("✓ ran").color(pal.success));
                                }
                                Some(Err(e)) => {
                                    ui.label(RichText::new(format!("⚠ {e}")).color(pal.danger));
                                }
                                None => {}
                            }
                        });
                    });
                    changed |= action_fields(ui, pal, spec);
                });
            });
            ui.add_space(6.0);
        }

        if ui.button("+ Add action").clicked() {
            actions.push(Kind::OpenApp.default_spec());
            changed = true;
        }

        // Apply a structural edit after the loop; a structural change clears the
        // stale test marker (indices shift).
        if let Some(op) = op {
            match op {
                Op::Remove(i) => {
                    actions.remove(i);
                }
                Op::MoveUp(i) => actions.swap(i, i - 1),
                Op::MoveDown(i) => actions.swap(i, i + 1),
            }
            changed = true;
        }
        // Any committed edit (type switch, field edit, add, reorder, remove)
        // invalidates a stale Test badge — its row/content has moved.
        if changed {
            self.last_test = None;
        }
        if let Some(i) = test_req {
            if let Some(spec) = actions.get(i) {
                let res = dispatch(spec, platform).map_err(|e| e.to_string());
                self.last_test = Some((zone_id, i, res));
            }
        }

        changed
    }
}

/// Per-variant field widgets. Returns `true` on a committed edit (free text
/// commits on focus loss; combos/checkboxes/list edits commit immediately).
fn action_fields(ui: &mut egui::Ui, pal: &Palette, spec: &mut ActionSpec) -> bool {
    let mut changed = false;
    match spec {
        ActionSpec::OpenApp { path } | ActionSpec::OpenPath { path } => {
            changed |= labeled_text(ui, "Path", path);
        }
        ActionSpec::OpenUrl { url } => changed |= labeled_text(ui, "URL", url),
        ActionSpec::RunCommand { command, args } => {
            changed |= labeled_text(ui, "Command", command);
            let mut remove: Option<usize> = None;
            for (j, arg) in args.iter_mut().enumerate() {
                ui.push_id(j, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("arg {}", j + 1)).color(pal.secondary));
                        if ui.text_edit_singleline(arg).lost_focus() {
                            changed = true;
                        }
                        if ui.button("x").clicked() {
                            remove = Some(j);
                        }
                    });
                });
            }
            if let Some(j) = remove {
                args.remove(j);
                changed = true;
            }
            if ui.button("+ arg").clicked() {
                args.push(String::new());
                changed = true;
            }
        }
        ActionSpec::CopyToClipboard { text } | ActionSpec::Speak { text } => {
            if ui.text_edit_multiline(text).lost_focus() {
                changed = true;
            }
        }
        ActionSpec::ShowToast { title, body } => {
            changed |= labeled_text(ui, "Title", title);
            changed |= labeled_text(ui, "Body", body);
        }
        ActionSpec::PlaySound { sound } => {
            ComboBox::from_id_salt("sound")
                .selected_text(format!("{sound:?}"))
                .show_ui(ui, |ui| {
                    for s in [
                        SystemSound::Default,
                        SystemSound::Success,
                        SystemSound::Error,
                        SystemSound::Notification,
                    ] {
                        if ui.selectable_value(sound, s, format!("{s:?}")).clicked() {
                            changed = true;
                        }
                    }
                });
        }
        ActionSpec::Screenshot { mode } => {
            ComboBox::from_id_salt("mode")
                .selected_text(format!("{mode:?}"))
                .show_ui(ui, |ui| {
                    for m in [ScreenshotMode::FullScreen, ScreenshotMode::RegionSelect] {
                        if ui.selectable_value(mode, m, format!("{m:?}")).clicked() {
                            changed = true;
                        }
                    }
                });
        }
        ActionSpec::SendKeystroke { combo } => {
            ui.horizontal(|ui| {
                for m in [Modifier::Ctrl, Modifier::Shift, Modifier::Alt, Modifier::Win] {
                    let mut on = combo.modifiers.contains(&m);
                    if ui.checkbox(&mut on, format!("{m:?}")).changed() {
                        if on {
                            combo.modifiers.push(m);
                        } else {
                            combo.modifiers.retain(|x| *x != m);
                        }
                        changed = true;
                    }
                }
            });
            changed |= labeled_text(ui, "Key", &mut combo.key);
        }
        ActionSpec::VisualOnly => {
            ui.label(
                RichText::new("Visual feedback only — no action is fired.")
                    .color(pal.secondary)
                    .small(),
            );
        }
    }
    changed
}

/// A label + single-line text field that reports a change only on focus loss.
fn labeled_text(ui: &mut egui::Ui, label: &str, value: &mut String) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        if ui.text_edit_singleline(value).lost_focus() {
            changed = true;
        }
    });
    changed
}
