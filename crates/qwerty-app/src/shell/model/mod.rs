//! The shell's MODEL layer (shell-v2 §1.2): the store, the event bridge and the request worker,
//! navigation, per-screen view-models as plain structs, state machines, tokens and copy. It imports
//! no GUI framework and no Windows API.

pub mod announcer;
pub mod argv;
pub mod attention;
pub mod bridge;
pub mod contrast;
pub mod copy;
pub mod coverage;
pub mod fixture;
pub mod keymap;
pub mod launch;
pub mod layout;
pub mod listening;
pub mod live_policy;
pub mod motion;
pub mod navigation;
pub mod pending_edits;
pub mod requests;
pub mod route;
pub mod store;
pub mod text_zoom;
pub mod timers;
pub mod toast_model;
pub mod tokens;
pub mod tray_model;
pub mod vm;
pub mod window_policy;
