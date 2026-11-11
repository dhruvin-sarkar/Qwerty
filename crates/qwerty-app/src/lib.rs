//! qwerty-app library: the GUI-agnostic v2 engine and the platform-actions seam.
//!
//! The `qwerty-app` binary consumes this library; nothing in `engine` imports a GUI crate
//! (p2 §2.1).

pub mod engine;
pub mod platform;
