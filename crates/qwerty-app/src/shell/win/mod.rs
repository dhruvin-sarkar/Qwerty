//! The shell's WINDOWS layer (shell-v2 §1.2): tray, hotkey, keycap rasters, toasts, DWM, system
//! settings, placement, the settings listener window, the `WM_SIZE` subclass and file dialogs. It
//! imports no GUI framework; a window is an `isize` HWND handed in by the view layer.

pub mod altgr;
pub mod dwm;
pub mod file_dialog;
pub mod hotkey;
pub mod icon;
pub mod listener;
pub mod placement;
pub mod subclass;
pub mod system;
pub mod toasts;
pub mod tray;
