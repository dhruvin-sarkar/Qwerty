//! qwerty-app — the Qwerty desktop binary.
//!
//! Phase 0 scaffolding: this is a minimal entry point that links against
//! `qwerty-core` and reports both versions. The real application (audio
//! thread, egui shell, tray, hotkeys, platform action dispatch) is built up
//! in later phases per ROADMAP.md — nothing here is speculative.

fn main() {
    println!(
        "Qwerty {} (qwerty-core {})",
        env!("CARGO_PKG_VERSION"),
        qwerty_core::CORE_VERSION,
    );
}

#[cfg(test)]
mod tests {
    #[test]
    fn links_against_core() {
        // Phase 0 smoke test: the binary crate compiles and can reach into
        // qwerty-core across the workspace boundary.
        assert!(!qwerty_core::CORE_VERSION.is_empty());
    }
}
