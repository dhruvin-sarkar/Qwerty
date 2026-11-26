//! The v2 shell: the window side of the app (shell-v2 §1). Three layers, one job each: `model`
//! (framework-free and OS-free), `win` (Win32 and the tray/hotkey crates, no GUI framework) and
//! `view` (the GUI framework). The layer rule is enforced by `tests/shell_layers.rs` (SB1).

pub mod model;
pub mod view;
pub mod win;
