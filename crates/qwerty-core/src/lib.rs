//! qwerty-core — the platform-agnostic signal-processing and classification
//! core for Qwerty.
//!
//! The audio path is `mic → onset detector → 90ms feature window → zone
//! classifier` (see ARCHITECTURE.md). Everything in this crate is pure Rust
//! with no OS-specific or GUI dependencies, so it can be exercised entirely
//! against synthetic signals (see the `soak` example and the Phase 1 tests).
//!
//! Modules are introduced phase by phase per ROADMAP.md; Phase 0 is
//! scaffolding only.

pub mod calibration;
pub mod capture;
pub mod classifier;
pub mod features;
pub mod onset;
pub mod profile;
pub mod soak;

/// Crate version string, surfaced so the app and soak binary can log which
/// core they were built against.
pub const CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_version_is_reported() {
        // Phase 0 smoke test: the crate builds and links, and its version
        // constant is populated from the manifest.
        assert!(!CORE_VERSION.is_empty());
    }
}
