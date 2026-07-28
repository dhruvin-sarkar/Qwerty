//! The non-color half of the design system: spacing, corner-radius, and
//! type-size scales (`DESIGN.md` → Visual design system — "generous whitespace",
//! "rounded surfaces", "restrained" type differentiated by size + weight).
//!
//! Colors live in [`theme`](super::theme) and time lives in
//! [`motion`](super::motion); this is the geometry. It is the same "one way to
//! do things" discipline applied to layout: a screen adds space, sets a margin,
//! or sizes a heading from *these* tokens, never a hand-picked literal, so the
//! vertical rhythm and roundness stay identical across every screen and the
//! whole surface can be re-tuned from one place.
//!
//! The numeric scales are plain `f32` (unit-testable, no egui needed); the thin
//! constructors and [`configure_spacing`] are where those numbers meet egui's
//! `i8`/`u8` margin/radius types and its global [`egui::Spacing`].

/// Spacing scale in logical points — a 4-pt base grid with a 2-pt half-step for
/// hairline inline gaps. Chosen to match (and formalize) the values already in
/// use across the screens rather than impose a foreign rhythm.
pub mod space {
    /// Hairline gap inside a single line (glyph → text).
    pub const XXS: f32 = 2.0;
    /// Label → its control; the tightest vertical stack.
    pub const XS: f32 = 4.0;
    /// Between closely related items.
    pub const SM: f32 = 8.0;
    /// Between control groups within a block.
    pub const MD: f32 = 12.0;
    /// Card inner padding; separation between blocks.
    pub const LG: f32 = 16.0;
    /// Screen content margin; separation between sections.
    pub const XL: f32 = 24.0;
    /// Major section breaks; the wizard's focused canvas margin.
    pub const XXL: f32 = 32.0;
}

/// Corner-radius scale. `DESIGN.md`: rounded surfaces with depth from elevation,
/// not gradients. Small for controls, medium for cards/menus, large for the
/// full-window focus surfaces (the calibration wizard).
pub mod radius {
    /// Micro-marks: chart bars, heat-map cells, meter tracks, sparklines — the
    /// small painted rectangles inside a card, kept a hair rounder than square
    /// but below the control radius. One token so these read identically across
    /// the Evaluation, Diagnostics, and Home level-meter views instead of
    /// drifting between hand-picked 2/3/4 px literals.
    pub const XS: f32 = 3.0;
    /// Buttons, inputs, combo boxes, nav pills.
    pub const SM: f32 = 6.0;
    /// Cards, banners, popovers, menus.
    pub const MD: f32 = 10.0;
    /// Hero / full-window surfaces (the wizard canvas). Part of the complete
    /// radius vocabulary; consumed as the wizard adopts the scale (matches the
    /// `motion.rs` pattern of defining the full token set up front).
    #[allow(dead_code)]
    pub const LG: f32 = 14.0;
}

/// Type-size scale in logical points. One family (the platform default, per
/// `DESIGN.md`), differentiated by size + weight only — never all-caps.
pub mod text {
    /// The single hero metric (the Evaluation headline accuracy number).
    pub const DISPLAY: f32 = 34.0;
    /// Screen headings ("Home", "Settings", …).
    pub const TITLE: f32 = 26.0;
    /// Sub-headings and the wizard step title.
    pub const HEADING: f32 = 20.0;
    /// Titled section labels within a screen.
    pub const SECTION: f32 = 16.0;
    /// Default body text (matches egui's default body size).
    pub const BODY: f32 = 14.0;
    /// Secondary / helper / caption text.
    pub const CAPTION: f32 = 12.0;
}

/// An `egui::Margin` from a spacing token. Rounds to egui's `i8` margin unit.
pub fn margin(pts: f32) -> egui::Margin {
    egui::Margin::same(pts.round() as i8)
}

/// An `egui::CornerRadius` from a radius token. Rounds to egui's `u8` unit.
pub fn rounding(pts: f32) -> egui::CornerRadius {
    egui::CornerRadius::same(pts.round() as u8)
}

/// Apply the spacing scale to egui's global [`Spacing`] once, at startup.
///
/// Spacing is theme-independent, so it is set a single time rather than rebuilt
/// each frame — the per-frame `set_visuals` for the theme cross-fade replaces
/// only the *color* half of the style and leaves this intact. This is what
/// replaces egui's cramped defaults with the design system's breathing room
/// coherently, in one place, for every widget on every screen.
pub fn configure_spacing(spacing: &mut egui::Spacing) {
    // A gentle, consistent vertical rhythm between consecutive widgets (egui's
    // default packs them tightly at 3pt vertically).
    spacing.item_spacing = egui::vec2(space::SM, space::SM);
    // Roomier, more tactile controls — the single biggest cue that lifts the UI
    // out of the "default framework" look, applied to every button/combo/input.
    spacing.button_padding = egui::vec2(space::MD, 7.0);
    // Comfortable minimum control height for pointer + touch targets.
    spacing.interact_size.y = 22.0;
    spacing.indent = space::LG;
    spacing.menu_margin = margin(space::SM);
    spacing.window_margin = margin(space::LG);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spacing_scale_is_strictly_increasing() {
        // The scale is a rhythm: each step must be strictly larger than the
        // last, or "SM vs MD" stops meaning anything at the call sites.
        let steps = [
            space::XXS,
            space::XS,
            space::SM,
            space::MD,
            space::LG,
            space::XL,
            space::XXL,
        ];
        for w in steps.windows(2) {
            assert!(
                w[1] > w[0],
                "spacing scale not increasing: {} !> {}",
                w[1],
                w[0]
            );
        }
    }

    #[test]
    fn type_scale_is_strictly_decreasing_from_display_to_caption() {
        let sizes = [
            text::DISPLAY,
            text::TITLE,
            text::HEADING,
            text::SECTION,
            text::BODY,
            text::CAPTION,
        ];
        for w in sizes.windows(2) {
            assert!(w[1] < w[0], "type scale not decreasing: {} !< {}", w[1], w[0]);
        }
    }

    #[test]
    fn radius_scale_is_strictly_increasing() {
        let steps = [radius::XS, radius::SM, radius::MD, radius::LG];
        for w in steps.windows(2) {
            assert!(w[1] > w[0], "radius scale not increasing: {} !> {}", w[1], w[0]);
        }
    }
}
