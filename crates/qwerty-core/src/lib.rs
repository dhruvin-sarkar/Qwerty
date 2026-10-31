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

pub mod assessment; // v2 evaluation (WP-B1)
pub mod audio; // v2 (WP-00)
pub mod augment; // v2 (WP-B2)
pub mod calibration; // v1
pub mod capture; // v1
pub mod classifier; // v1
pub mod descriptor; // v2 (types WP-00, computation WP-A1)
pub mod desk; // v2 test/bench fixtures (WP-A3)
pub mod detector; // v2 (WP-A1)
pub mod dsp; // v2 (WP-A1)
pub mod evaluation; // v1
pub mod features; // v1
pub mod gesture; // v2 (WP-A4)
pub mod linalg; // v2 (WP-A2)
pub mod migrate; // v2 (WP-B3)
pub mod model; // v2 (WP-A2)
pub mod onset; // v1
pub mod pipeline; // v2 (WP-B1)
pub mod profile; // shared (v1 Config/Profile/Zone/DeviceFingerprint/SensingMode removed in WP-D1)
pub mod recording; // v2 (WP-A3)
pub mod replay; // v2 (WP-B4)
pub mod rng; // v2 (WP-00)
pub mod session; // v2 calibration (WP-B2)
pub mod soak; // rewritten onto v2 (WP-B5)
pub mod store; // v2 config/profile (WP-B3)
pub mod training; // v2 (WP-A2)
pub mod units; // v2 (WP-00)
pub mod veto; // v2 (WP-A4)
pub mod wav; // v2 (WP-A3)

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
