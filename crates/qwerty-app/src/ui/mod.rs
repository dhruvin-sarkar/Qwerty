//! The egui UI layer (`ARCHITECTURE.md` → qwerty-app modules).
//!
//! Phase 3 builds this up: the theme token system lands first (pure, testable),
//! then the motion tokens, then the eframe window/navigation rail/screens that
//! read from both. The tray/hotkey wiring is layered on next.

pub mod action_editor;
pub mod diagnostics;
pub mod evaluation;
pub mod icon;
pub mod motion;
pub mod paint;
pub mod shell;
pub mod style;
pub mod theme;
pub mod wizard;
