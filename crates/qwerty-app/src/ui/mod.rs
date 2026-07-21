//! The egui UI layer (`ARCHITECTURE.md` → qwerty-app modules).
//!
//! Phase 3 builds this up: the theme token system lands first (pure, testable,
//! below), then the eframe window, navigation rail, and screens read from it.
//! The window/tray/hotkey wiring is added once it can be visually verified.

pub mod theme;
