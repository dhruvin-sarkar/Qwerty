//! Motion tokens and the redraw-discipline primitive (`MOTION.md`).
//!
//! `MOTION.md` requires exactly one shared set of durations and easing curves,
//! used everywhere — never a hand-picked duration per component. They live here
//! as constants plus a pure cubic-bezier evaluator, so the actual animation
//! math is unit-testable with no egui or window involved.
//!
//! The redraw-discipline rule from `PERFORMANCE.md` is expressed by the shell:
//! it schedules the next frame (`ctx.request_repaint_after`) *only while* an
//! animation is mid-flight — once every animated value has settled, repaints
//! stop and idle CPU returns to zero.

use std::time::Duration;

// --- Duration tokens (MOTION.md → Motion tokens) ---------------------------
//
// MOTION.md requires the *complete* token set to exist as one vocabulary, up
// front (ROADMAP.md Phase 3: establish the motion foundation before animated
// screens). `STANDARD` (theme cross-fade) and `INSTANT`/`QUICK` (navigation
// hover + selection glide, MOTION.md → Navigation selection) are consumed;
// `DELIBERATE` remains for the Phase 4 wizard step transition, so it alone
// still carries `#[allow(dead_code)]` rather than being omitted and re-derived
// per component.

/// Tiny state flips: hover, toggle, checkbox.
pub const INSTANT: Duration = Duration::from_millis(100);
/// Button press feedback, tab switches, zone highlight pulse.
pub const QUICK: Duration = Duration::from_millis(180);
/// Screen/step transitions, panel expand/collapse, theme cross-fade.
pub const STANDARD: Duration = Duration::from_millis(250);
/// Calibration wizard step transitions — slower on purpose (Phase 4).
#[allow(dead_code)]
pub const DELIBERATE: Duration = Duration::from_millis(400);

// --- Easing curves (MOTION.md → easing.*) ----------------------------------

/// `easing.standard` — the default for nearly everything: fast start, gentle
/// settle. `cubic-bezier(0.2, 0.0, 0.0, 1.0)`.
pub const STANDARD_EASE: Bezier = Bezier::new(0.2, 0.0, 0.0, 1.0);
/// `easing.emphasized` — a bit more spring, for the zone-accept pulse and
/// success states (Phase 4). `cubic-bezier(0.3, 0.0, 0.1, 1.0)`.
#[allow(dead_code)]
pub const EMPHASIZED_EASE: Bezier = Bezier::new(0.3, 0.0, 0.1, 1.0);

/// A CSS-style cubic-bezier easing curve with fixed endpoints `(0,0)`→`(1,1)`
/// and two control points. Evaluating it maps a linear time fraction `x` in
/// `[0, 1]` to an eased output `y`.
#[derive(Clone, Copy, Debug)]
pub struct Bezier {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

impl Bezier {
    pub const fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// Eased output for a linear time fraction `x` (clamped to `[0, 1]`).
    ///
    /// A cubic-bezier is parameterized by `s`, not by `x` directly, so we first
    /// solve `bezier_x(s) = x` for `s` (Newton-Raphson, then a bisection
    /// fallback), then return `bezier_y(s)`.
    pub fn eval(&self, x: f32) -> f32 {
        let x = x.clamp(0.0, 1.0);
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        let s = self.solve_for_s(x);
        bezier_axis(s, self.y1, self.y2)
    }

    fn solve_for_s(&self, x: f32) -> f32 {
        // Newton-Raphson from a good initial guess (s ≈ x).
        let mut s = x;
        for _ in 0..8 {
            let err = bezier_axis(s, self.x1, self.x2) - x;
            if err.abs() < 1e-5 {
                return s;
            }
            let d = bezier_axis_derivative(s, self.x1, self.x2);
            if d.abs() < 1e-6 {
                break;
            }
            s -= err / d;
        }
        // Bisection fallback if the derivative was ill-behaved.
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        s = x;
        for _ in 0..24 {
            let cx = bezier_axis(s, self.x1, self.x2);
            if (cx - x).abs() < 1e-5 {
                break;
            }
            if cx < x {
                lo = s;
            } else {
                hi = s;
            }
            s = 0.5 * (lo + hi);
        }
        s
    }
}

/// One axis of a cubic bezier with fixed endpoints 0 and 1: `B(s)` for control
/// values `c1`, `c2` at `s` in `[0, 1]`.
fn bezier_axis(s: f32, c1: f32, c2: f32) -> f32 {
    let u = 1.0 - s;
    // 3(1-s)²s·c1 + 3(1-s)s²·c2 + s³   (the P0=0, P3=1 terms simplify).
    3.0 * u * u * s * c1 + 3.0 * u * s * s * c2 + s * s * s
}

fn bezier_axis_derivative(s: f32, c1: f32, c2: f32) -> f32 {
    let u = 1.0 - s;
    3.0 * u * u * c1 + 6.0 * u * s * (c2 - c1) + 3.0 * s * s * (1.0 - c2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bezier_pins_endpoints() {
        for b in [STANDARD_EASE, EMPHASIZED_EASE] {
            assert!((b.eval(0.0) - 0.0).abs() < 1e-4, "start should be 0");
            assert!((b.eval(1.0) - 1.0).abs() < 1e-4, "end should be 1");
        }
    }

    #[test]
    fn bezier_clamps_and_is_monotonic() {
        let b = STANDARD_EASE;
        assert_eq!(b.eval(-0.5), 0.0);
        assert_eq!(b.eval(1.5), 1.0);
        let mut prev = 0.0;
        for i in 0..=20 {
            let y = b.eval(i as f32 / 20.0);
            assert!(y >= prev - 1e-4, "curve must not decrease at x={i}");
            prev = y;
        }
    }

    #[test]
    fn standard_ease_front_loads_progress() {
        // easing.standard is "fast start": at the time midpoint it is already
        // well past the halfway output.
        assert!(STANDARD_EASE.eval(0.5) > 0.5);
    }
}
