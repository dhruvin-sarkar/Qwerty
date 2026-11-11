//! The GUI-agnostic v2 engine (p2 §2). Every module is `pub`, so its items are library API
//! while the packages land in parallel (p2 §2.1).

pub mod capture;
pub mod cli;
pub mod clock;
pub mod controller;
pub mod devices;
pub mod dispatch;
pub mod handle;
pub mod instance;
pub mod logfile;
pub mod messages;
pub mod modes;
pub mod os;
pub mod persist;
pub mod processing;
pub mod protocol;
pub mod rawinput;
pub mod recorder;
pub mod ring;
pub mod telemetry;
pub mod testkit;
