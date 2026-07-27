//! The 5-theme token system (`DESIGN.md` → Visual design system).
//!
//! Each theme is one [`Tokens`] set. UI code reads a *semantic token*
//! (`accent`, `bg_surface`, `danger`, …), never a literal color, so adding a
//! sixth theme later is a data-only change — no component touches a hardcoded
//! color. Tokens live here as plain sRGB (no egui dependency) so they stay
//! unit-testable; the shell maps them to egui `Visuals` at its boundary.
//!
//! Accessibility is a test, not a claim: WCAG relative-luminance contrast is
//! computed here, and every theme's primary text — plus the accent and status
//! colors, which are also rendered as text/glyphs — clears AA (≥4.5:1) against
//! its base, with High Contrast clearing AAA (≥7:1) — see the tests.

use qwerty_core::profile::Theme;

/// An 8-bit sRGB color.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// `#RRGGBB` form, for the `--themes` inspector and debugging.
    pub fn hex(&self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }

    /// Linearly interpolate between two colors, `t` clamped to `[0, 1]`.
    /// Used by the theme cross-fade (`MOTION.md` → Theme switch), which
    /// precomputes both token sets and interpolates rather than swapping the
    /// style object mid-frame. Kept here (pure, no egui) so it stays testable.
    pub fn lerp(a: Color, b: Color, t: f32) -> Color {
        let t = t.clamp(0.0, 1.0);
        let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
        Color::rgb(mix(a.r, b.r), mix(a.g, b.g), mix(a.b, b.b))
    }
}

/// One theme's full set of semantic tokens (`DESIGN.md`: background layers,
/// text layers, border, accent, and status colors).
#[derive(Clone, Copy, Debug)]
pub struct Tokens {
    /// Deepest background layer (window base).
    pub bg_base: Color,
    /// Panels / cards.
    pub bg_surface: Color,
    /// Raised surfaces (menus, popovers).
    pub bg_elevated: Color,
    pub text_primary: Color,
    pub text_secondary: Color,
    pub border: Color,
    pub accent: Color,
    pub success: Color,
    pub warning: Color,
    pub danger: Color,
}

impl Tokens {
    /// Interpolate every token between two theme token sets, for the cross-fade
    /// in `MOTION.md` (Theme switch). At `t = 0` this is `from`, at `t = 1` it
    /// is `to`; the shell maps the result to egui `Visuals` each frame while the
    /// switch animation is in flight, then stops repainting.
    pub fn lerp(from: &Tokens, to: &Tokens, t: f32) -> Tokens {
        let c = |a: Color, b: Color| Color::lerp(a, b, t);
        Tokens {
            bg_base: c(from.bg_base, to.bg_base),
            bg_surface: c(from.bg_surface, to.bg_surface),
            bg_elevated: c(from.bg_elevated, to.bg_elevated),
            text_primary: c(from.text_primary, to.text_primary),
            text_secondary: c(from.text_secondary, to.text_secondary),
            border: c(from.border, to.border),
            accent: c(from.accent, to.accent),
            success: c(from.success, to.success),
            warning: c(from.warning, to.warning),
            danger: c(from.danger, to.danger),
        }
    }
}

/// Resolve a [`Theme`] to its concrete [`Tokens`]. `system_prefers_dark` is the
/// OS light/dark setting, consulted only for [`Theme::System`] (`DESIGN.md`:
/// System follows Windows automatically).
pub fn tokens_for(theme: Theme, system_prefers_dark: bool) -> Tokens {
    match theme {
        Theme::System => {
            if system_prefers_dark {
                MIDNIGHT
            } else {
                DAYLIGHT
            }
        }
        Theme::Daylight => DAYLIGHT,
        Theme::Midnight => MIDNIGHT,
        Theme::Aurora => AURORA,
        Theme::HighContrast => HIGH_CONTRAST,
    }
}

// The concrete palettes. Daylight and Midnight share one accent (DESIGN.md);
// Aurora reserves its accent for visualizations; High Contrast is validated to
// WCAG AAA.

const DAYLIGHT: Tokens = Tokens {
    // Three distinct near-white layers (base < surface < elevated) instead of a
    // flat white-on-white — base is left as-is because its accent contrast is
    // already near the 4.5 floor, so the layering comes from lifting surface
    // toward white and reserving pure white for elevated (cards/menus).
    bg_base: Color::rgb(0xFA, 0xF8, 0xF5),
    bg_surface: Color::rgb(0xFC, 0xFB, 0xF9),
    bg_elevated: Color::rgb(0xFF, 0xFF, 0xFF),
    text_primary: Color::rgb(0x1B, 0x1B, 0x1F),
    text_secondary: Color::rgb(0x55, 0x55, 0x5C),
    // A more defined border so panels and controls read crisply (borders carry
    // no text, so this is free of the contrast constraints the fills have).
    border: Color::rgb(0xD5, 0xCF, 0xC5),
    accent: Color::rgb(0x2D, 0x6C, 0xDF),
    // Darkened from 0x1E874B (4.29:1 — below AA on bg_base) to 5.07:1, matching
    // this theme's other status colors (danger 5.0, warning 5.1). See the
    // semantic-color contrast test.
    success: Color::rgb(0x1A, 0x7A, 0x43),
    warning: Color::rgb(0x8A, 0x63, 0x00),
    danger: Color::rgb(0xC2, 0x39, 0x34),
};

const MIDNIGHT: Tokens = Tokens {
    bg_base: Color::rgb(0x0A, 0x0A, 0x0C),
    bg_surface: Color::rgb(0x13, 0x13, 0x17),
    bg_elevated: Color::rgb(0x1C, 0x1C, 0x22),
    text_primary: Color::rgb(0xEC, 0xEC, 0xEF),
    text_secondary: Color::rgb(0xA0, 0xA0, 0xA8),
    border: Color::rgb(0x32, 0x32, 0x3C),
    accent: Color::rgb(0x4C, 0x8D, 0xFF),
    success: Color::rgb(0x35, 0xC4, 0x6B),
    warning: Color::rgb(0xE0, 0xA9, 0x3B),
    danger: Color::rgb(0xF0, 0x6A, 0x6A),
};

const AURORA: Tokens = Tokens {
    bg_base: Color::rgb(0x0E, 0x11, 0x16),
    bg_surface: Color::rgb(0x16, 0x1B, 0x22),
    bg_elevated: Color::rgb(0x1D, 0x24, 0x2D),
    text_primary: Color::rgb(0xE8, 0xED, 0xF2),
    text_secondary: Color::rgb(0x9A, 0xA6, 0xB2),
    border: Color::rgb(0x2E, 0x39, 0x46),
    accent: Color::rgb(0x45, 0xC4, 0xB4),
    success: Color::rgb(0x35, 0xC4, 0x6B),
    warning: Color::rgb(0xE0, 0xA9, 0x3B),
    danger: Color::rgb(0xF0, 0x6A, 0x6A),
};

const HIGH_CONTRAST: Tokens = Tokens {
    bg_base: Color::rgb(0x00, 0x00, 0x00),
    bg_surface: Color::rgb(0x00, 0x00, 0x00),
    bg_elevated: Color::rgb(0x0A, 0x0A, 0x0A),
    text_primary: Color::rgb(0xFF, 0xFF, 0xFF),
    text_secondary: Color::rgb(0xFF, 0xFF, 0xFF),
    border: Color::rgb(0xFF, 0xFF, 0xFF),
    accent: Color::rgb(0xFF, 0xD6, 0x00),
    success: Color::rgb(0x00, 0xE6, 0x76),
    warning: Color::rgb(0xFF, 0xD6, 0x00),
    danger: Color::rgb(0xFF, 0x52, 0x52),
};

/// The four concrete (non-`System`) themes, for iteration.
pub const CONCRETE_THEMES: [Theme; 4] = [
    Theme::Daylight,
    Theme::Midnight,
    Theme::Aurora,
    Theme::HighContrast,
];

// --- WCAG contrast ---------------------------------------------------------

fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.039_28 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn relative_luminance(c: Color) -> f32 {
    0.2126 * srgb_to_linear(c.r) + 0.7152 * srgb_to_linear(c.g) + 0.0722 * srgb_to_linear(c.b)
}

/// WCAG contrast ratio between two colors, in `[1.0, 21.0]`.
pub fn contrast_ratio(a: Color, b: Color) -> f32 {
    let (la, lb) = (relative_luminance(a), relative_luminance(b));
    let (hi, lo) = if la >= lb { (la, lb) } else { (lb, la) };
    (hi + 0.05) / (lo + 0.05)
}

/// Black or white — whichever has the higher WCAG contrast against `bg` — for
/// text/glyphs painted *on* a filled color (e.g. the accent fill of a primary
/// button). This keeps on-accent labels legible across every theme without
/// adding a per-theme `on_accent` token: the accents range from a light yellow
/// (High Contrast) to a mid blue (Midnight), and no single fixed ink is legible
/// on all of them. The `on_accent_ink_clears_aa_for_every_theme` test proves
/// the chosen ink clears WCAG AA for every theme's accent.
pub fn on_color(bg: Color) -> Color {
    let black = Color::rgb(0, 0, 0);
    let white = Color::rgb(255, 255, 255);
    if contrast_ratio(white, bg) >= contrast_ratio(black, bg) {
        white
    } else {
        black
    }
}

/// Print each theme's key tokens and its text/background contrast + WCAG
/// verdict — the terminal inspector for the theme system before the GUI exists.
pub fn print_report() {
    println!("Qwerty theme tokens — DESIGN.md 5-theme system\n");
    for theme in CONCRETE_THEMES {
        let t = tokens_for(theme, false);
        let c = contrast_ratio(t.text_primary, t.bg_base);
        let verdict = if c >= 7.0 {
            "AA + AAA"
        } else if c >= 4.5 {
            "AA"
        } else {
            "FAILS AA"
        };
        println!("{theme:?}");
        println!(
            "  bg {} / {} / {}   text {} / {}   border {}",
            t.bg_base.hex(),
            t.bg_surface.hex(),
            t.bg_elevated.hex(),
            t.text_primary.hex(),
            t.text_secondary.hex(),
            t.border.hex(),
        );
        println!(
            "  accent {}   success {}   warning {}   danger {}",
            t.accent.hex(),
            t.success.hex(),
            t.warning.hex(),
            t.danger.hex(),
        );
        println!("  text/bg contrast {c:.1}:1  [{verdict}]\n");
    }
    println!("System follows the OS: dark -> Midnight, light -> Daylight.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contrast_ratio_extremes_are_correct() {
        let black = Color::rgb(0, 0, 0);
        let white = Color::rgb(255, 255, 255);
        assert!((contrast_ratio(black, white) - 21.0).abs() < 0.1);
        assert!((contrast_ratio(white, white) - 1.0).abs() < 0.001);
    }

    #[test]
    fn every_theme_primary_text_meets_wcag_aa() {
        for theme in CONCRETE_THEMES {
            let t = tokens_for(theme, false);
            let c = contrast_ratio(t.text_primary, t.bg_base);
            assert!(
                c >= 4.5,
                "{theme:?} primary text contrast {c:.2} < 4.5 (AA)"
            );
        }
    }

    #[test]
    fn high_contrast_theme_meets_wcag_aaa() {
        let t = tokens_for(Theme::HighContrast, false);
        let c = contrast_ratio(t.text_primary, t.bg_base);
        assert!(c >= 7.0, "High Contrast text contrast {c:.2} < 7.0 (AAA)");
    }

    #[test]
    fn secondary_text_is_at_least_large_text_aa() {
        for theme in CONCRETE_THEMES {
            let t = tokens_for(theme, false);
            let c = contrast_ratio(t.text_secondary, t.bg_base);
            assert!(c >= 3.0, "{theme:?} secondary text contrast {c:.2} < 3.0");
        }
    }

    #[test]
    fn primary_text_is_legible_on_every_background_layer() {
        // The shell paints primary text on all three background layers (central
        // panel = bg_base, nav rail = bg_surface, cards/menus = bg_elevated).
        // Each pairing must clear WCAG AA for normal text (ACCEPTANCE.md → UI:
        // "no illegible text in any theme" — the automated half of that row).
        for theme in CONCRETE_THEMES {
            let t = tokens_for(theme, false);
            for (name, bg) in [
                ("bg_base", t.bg_base),
                ("bg_surface", t.bg_surface),
                ("bg_elevated", t.bg_elevated),
            ] {
                let c = contrast_ratio(t.text_primary, bg);
                assert!(
                    c >= 4.5,
                    "{theme:?} primary text on {name} contrast {c:.2} < 4.5 (AA)"
                );
            }
        }
    }

    #[test]
    fn on_accent_ink_clears_aa_for_every_theme() {
        // A primary button fills with the accent and paints its label in
        // `on_color(accent)`. That label must clear WCAG AA (4.5:1) on every
        // theme's accent, or the app's single most prominent control would be
        // hard to read in some theme — the same accessibility bar the other
        // color tests hold, extended to on-accent ink.
        for theme in CONCRETE_THEMES {
            let t = tokens_for(theme, false);
            let ink = on_color(t.accent);
            let c = contrast_ratio(ink, t.accent);
            assert!(
                c >= 4.5,
                "{theme:?} on-accent ink {} contrast {c:.2} < 4.5 (AA)",
                ink.hex()
            );
        }
    }

    #[test]
    fn on_color_picks_the_higher_contrast_ink() {
        // Light fill → black ink; dark fill → white ink.
        assert_eq!(on_color(Color::rgb(0xFF, 0xD6, 0x00)), Color::rgb(0, 0, 0));
        assert_eq!(on_color(Color::rgb(0x0A, 0x0A, 0x0C)), Color::rgb(255, 255, 255));
    }

    #[test]
    fn color_lerp_hits_both_endpoints() {
        let a = Color::rgb(0, 0, 0);
        let b = Color::rgb(255, 100, 50);
        assert_eq!(Color::lerp(a, b, 0.0), a);
        assert_eq!(Color::lerp(a, b, 1.0), b);
        // Midpoint rounds to nearest.
        assert_eq!(Color::lerp(a, b, 0.5), Color::rgb(128, 50, 25));
    }

    #[test]
    fn color_lerp_clamps_out_of_range_t() {
        let a = Color::rgb(10, 20, 30);
        let b = Color::rgb(200, 200, 200);
        assert_eq!(Color::lerp(a, b, -1.0), a);
        assert_eq!(Color::lerp(a, b, 2.0), b);
    }

    #[test]
    fn tokens_lerp_endpoints_are_exact_themes() {
        let from = tokens_for(Theme::Daylight, false);
        let to = tokens_for(Theme::Midnight, false);
        assert_eq!(Tokens::lerp(&from, &to, 0.0).bg_base, from.bg_base);
        assert_eq!(Tokens::lerp(&from, &to, 1.0).accent, to.accent);
    }

    #[test]
    fn system_theme_follows_os_preference() {
        // System is not its own palette — it resolves to a concrete one.
        assert_eq!(
            tokens_for(Theme::System, true).bg_base,
            tokens_for(Theme::Midnight, false).bg_base
        );
        assert_eq!(
            tokens_for(Theme::System, false).bg_base,
            tokens_for(Theme::Daylight, false).bg_base
        );
    }

    #[test]
    fn semantic_colors_are_legible_as_text_on_backgrounds() {
        // accent + status colors are not just fills — they are rendered as
        // text/glyphs: the action editor's "✓ ran" / "⚠ …" badges, the
        // selected-zone and nav labels (accent), the save-error banner's ⚠ glyph
        // and border (danger on the elevated fill). Like primary text they must
        // clear WCAG AA (4.5:1) on whatever panel they can appear over. Collect
        // every shortfall so one run reports the whole picture, not just the first.
        let mut failures = Vec::new();
        for theme in CONCRETE_THEMES {
            let t = tokens_for(theme, false);
            for (cname, color) in [
                ("accent", t.accent),
                ("success", t.success),
                ("warning", t.warning),
                ("danger", t.danger),
            ] {
                for (bname, bg) in [
                    ("bg_base", t.bg_base),
                    ("bg_surface", t.bg_surface),
                    ("bg_elevated", t.bg_elevated),
                ] {
                    let c = contrast_ratio(color, bg);
                    if c < 4.5 {
                        failures.push(format!("{theme:?} {cname} on {bname}: {c:.2}:1"));
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "semantic colors below WCAG AA (4.5:1) as text:\n{}",
            failures.join("\n")
        );
    }
}
