//! The egui UI layer (`ARCHITECTURE.md` → qwerty-app modules).
//!
//! Phase 3 builds this up: the theme token system lands first (pure, testable),
//! then the motion tokens, then the eframe window/navigation rail/screens that
//! read from both. The tray/hotkey wiring is layered on next.

pub mod motion;
pub mod shell;
pub mod theme;
