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
// screens). Every token is now consumed: `STANDARD` (theme cross-fade),
// `INSTANT`/`QUICK` (navigation hover + selection glide, zone-accept pulse),
// and `DELIBERATE` (the calibration wizard's per-step arrival transition).

/// Tiny state flips: hover, toggle, checkbox.
pub const INSTANT: Duration = Duration::from_millis(100);
/// Button press feedback, tab switches, zone highlight pulse.
pub const QUICK: Duration = Duration::from_millis(180);
/// Screen/step transitions, panel expand/collapse, theme cross-fade.
pub const STANDARD: Duration = Duration::from_millis(250);
/// Calibration wizard step transitions — slower on purpose (Phase 4).
pub const DELIBERATE: Duration = Duration::from_millis(400);

// --- Ambient tokens (the "listening heartbeat" + background breathe) -------
//
// These pace the ambient system (`shell.rs`). Kept here so the whole ambient
// vocabulary is one place, never a hand-picked interval at a call site.

/// How often the listening-heartbeat ring fires while idle + listening.
pub const HEARTBEAT_PERIOD: Duration = Duration::from_secs(6);
/// How long a single heartbeat ring takes to expand and fade.
pub const HEARTBEAT_PULSE_LEN: Duration = Duration::from_millis(2200);
/// The repaint cadence while any ambient effect is live (~30 fps). The shell's
/// single ambient repaint decision schedules the next frame this far out.
pub const AMBIENT_FRAME: Duration = Duration::from_millis(33);
/// Full period of the barely-perceptible background "breathe" (a slow sine).
// Consumed by the continuous-breathe commit (deferred, needs CPU measurement).
#[allow(dead_code)]
pub const BACKGROUND_BREATHE_PERIOD_SECS: f32 = 12.0;

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

/// A one-shot, forward-only animation from 0.0 to 1.0, driven by wall-clock
/// time. Used for choreographed sequences (the zone-accept pulse, wizard pip
/// fill, save-banner slide-in) that `Context::animate_bool_with_time` doesn't
/// cover because they are *fire-and-forget* rather than bool toggles: the caller
/// holds an `Option<AnimState>`, replaces it with `Some(AnimState::start(dur))`
/// to trigger, reads `t()`/`linear()` each frame, and the shell's repaint
/// scheduler keeps requesting frames until `is_complete()`.
#[derive(Clone, Copy, Debug)]
pub struct AnimState {
    started_at: std::time::Instant,
    duration: std::time::Duration,
}

impl AnimState {
    pub fn start(duration: std::time::Duration) -> Self {
        Self {
            started_at: std::time::Instant::now(),
            duration,
        }
    }

    /// Like [`start`](Self::start) but the animation does not begin until
    /// `starts_at` (which may be in the future). Used for staggered reveals
    /// (`MOTION.md`: cascade a list rather than snapping it in): item `i` starts
    /// `i * step` later. Read progress with the `*_at` methods, passing the
    /// frame's shared `now`, so a not-yet-started item reads exactly 0.0 without
    /// relying on the platform-specific behaviour of `Instant::elapsed` for a
    /// future instant.
    // Consumed by the staggered-reveal pass (Part 5); allow until then.
    #[allow(dead_code)]
    pub fn start_delayed(duration: std::time::Duration, starts_at: std::time::Instant) -> Self {
        Self {
            started_at: starts_at,
            duration,
        }
    }

    /// Raw linear progress evaluated against an explicit `now`, saturating to
    /// 0.0 before `started_at` — the delay-safe companion to [`linear`](Self::linear).
    #[allow(dead_code)] // consumed with start_delayed (Part 5)
    pub fn linear_at(&self, now: std::time::Instant) -> f32 {
        let secs = self.duration.as_secs_f32();
        if secs <= 0.0 {
            return 1.0;
        }
        (now.saturating_duration_since(self.started_at).as_secs_f32() / secs).clamp(0.0, 1.0)
    }

    /// Eased progress against an explicit `now` — the delay-safe [`t`](Self::t).
    #[allow(dead_code)] // consumed with start_delayed (Part 5)
    pub fn t_at(&self, now: std::time::Instant, easing: Bezier) -> f32 {
        easing.eval(self.linear_at(now))
    }

    /// Whether the animation has finished as of `now` (false before it starts).
    #[allow(dead_code)] // consumed with start_delayed (Part 5)
    pub fn is_complete_at(&self, now: std::time::Instant) -> bool {
        now.saturating_duration_since(self.started_at) >= self.duration
    }

    /// Eased progress in `[0.0, 1.0]`, reaching 1.0 once complete.
    pub fn t(&self, easing: Bezier) -> f32 {
        easing.eval(self.linear())
    }

    /// Raw linear progress in `[0.0, 1.0]` — for deriving two differently-eased
    /// values from one `AnimState` (e.g. the pulse's scale and glow alpha).
    pub fn linear(&self) -> f32 {
        let secs = self.duration.as_secs_f32();
        if secs <= 0.0 {
            return 1.0;
        }
        (self.started_at.elapsed().as_secs_f32() / secs).clamp(0.0, 1.0)
    }

    pub fn is_complete(&self) -> bool {
        self.started_at.elapsed() >= self.duration
    }
}

/// A physically-simulated, interruptible spring — the second motion primitive.
///
/// `AnimState` is right for discrete, fire-and-forget transitions with a fixed
/// start and end. It is the *wrong* tool for anything the user can interrupt
/// mid-flight (hover-follow, press depression, a value chasing a moving
/// target): a duration tween re-targeted mid-flight jump-cuts, whereas a spring
/// re-targeted mid-flight redirects smoothly because it integrates from the
/// current position and velocity, not from a fixed start/end pair. Integrated
/// with semi-implicit Euler; pure and egui-free, so the physics is unit-tested
/// with no window involved.
#[derive(Clone, Copy, Debug)]
pub struct Spring {
    pub value: f32,
    pub velocity: f32,
    pub target: f32,
    stiffness: f32,
    damping: f32,
}

impl Spring {
    /// A spring with the default (`Smooth`) feel. Part of the primitive's API
    /// and exercised by the unit tests; the shell's `spring_to` uses
    /// `with_preset` directly, so this carries an allow in the binary build.
    #[allow(dead_code)]
    pub fn new(initial: f32) -> Self {
        Self::with_preset(initial, SpringPreset::Smooth)
    }

    pub fn with_preset(initial: f32, preset: SpringPreset) -> Self {
        let (stiffness, damping) = preset.constants();
        Self {
            value: initial,
            velocity: 0.0,
            target: initial,
            stiffness,
            damping,
        }
    }

    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }

    /// Advance by `dt` seconds. Pull `dt` from egui's `stable_dt` (the frame
    /// delta it uses for its own animations); clamp it here so a stall (window
    /// drag, breakpoint) can't inject a huge step and blow up the simulation —
    /// this cap is belt-and-suspenders on top of the caller's `stable_dt.min`.
    pub fn step(&mut self, dt: f32) -> f32 {
        let dt = dt.min(1.0 / 30.0);
        let displacement = self.value - self.target;
        let accel = -self.stiffness * displacement - self.damping * self.velocity;
        self.velocity += accel * dt;
        self.value += self.velocity * dt;
        self.value
    }

    /// Jump straight to the target with zero velocity. The reduced-motion path
    /// is handled at the consumer (`spring_to` early-returns the target without
    /// stepping), so this is currently exercised only by the unit tests; kept as
    /// part of the primitive's API for a direct consumer that stores a spring.
    #[allow(dead_code)]
    pub fn snap_to_target(&mut self) {
        self.value = self.target;
        self.velocity = 0.0;
    }

    pub fn is_settled(&self) -> bool {
        (self.value - self.target).abs() < 0.0015 && self.velocity.abs() < 0.0015
    }
}

/// Spring feel presets (`stiffness`, `damping`). Kept as three named feels so a
/// call site picks an intent, never raw physics constants.
// Smooth (nav/swatch hover) and Stiff (number count-up) are in use; Bouncy is
// reserved for the toggle-knob interaction (a later Part 2 commit).
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub enum SpringPreset {
    /// No overshoot. Precise — focus rings, meter needles, anything that should
    /// never look playful.
    Stiff,
    /// A little life, no wobble. The default for hover lift and press depression.
    Smooth,
    /// Visible overshoot and settle. Reserved for a single moment (the
    /// zone-accept tile scale and the toggle knob) — never chrome, never two at
    /// once in a view, or it reads as noisy rather than alive.
    Bouncy,
}

impl SpringPreset {
    fn constants(self) -> (f32, f32) {
        match self {
            SpringPreset::Stiff => (320.0, 32.0),
            SpringPreset::Smooth => (210.0, 20.0),
            SpringPreset::Bouncy => (180.0, 12.0),
        }
    }
}

/// One shared per-frame clock for every ambient effect (heartbeat baseline,
/// RMS-reactive glow, background breathe), computed once in the shell and passed
/// by value so no effect reads its own timer. `seconds` is the elapsed time
/// since the ambient epoch (the first frame). Copy + egui-free so it threads
/// through painters without borrow hazards.
// Consumed by the ambient-system commit (Parts 3-4); allow until wired.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AmbientClock {
    pub seconds: f32,
}

#[allow(dead_code)] // consumed by the ambient-system commit (Parts 3-4)
impl AmbientClock {
    /// A smooth 0→1→0 breathing value over `period_secs`: a raised sine, so it
    /// eases at both ends rather than resetting hard at the loop boundary.
    pub fn breathe(&self, period_secs: f32) -> f32 {
        use std::f32::consts::TAU;
        (((self.seconds / period_secs) * TAU).sin() * 0.5 + 0.5).clamp(0.0, 1.0)
    }
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
    fn anim_state_starts_near_zero_incomplete_and_bounded() {
        // A long duration so the test's own runtime can't advance it past ~0.
        let a = AnimState::start(std::time::Duration::from_secs(3600));
        assert!(a.linear() < 0.01, "just-started progress should be ~0");
        assert!(!a.is_complete());
        let t = a.t(STANDARD_EASE);
        assert!((0.0..=1.0).contains(&t), "eased t must stay in [0, 1]");
    }

    #[test]
    fn anim_state_zero_duration_is_immediately_complete() {
        // Guard: a zero duration returns 1.0 rather than NaN from a 0/0 divide.
        let a = AnimState::start(std::time::Duration::ZERO);
        assert_eq!(a.linear(), 1.0);
        assert!(a.is_complete());
    }

    #[test]
    fn standard_ease_front_loads_progress() {
        // easing.standard is "fast start": at the time midpoint it is already
        // well past the halfway output.
        assert!(STANDARD_EASE.eval(0.5) > 0.5);
    }

    #[test]
    fn spring_converges_to_target_from_displacement() {
        // Released from 0 toward 1, a stable spring must arrive and settle.
        let mut s = Spring::new(0.0);
        s.set_target(1.0);
        // ~2s of 60fps steps is ample for every preset to settle.
        for _ in 0..120 {
            s.step(1.0 / 60.0);
        }
        assert!(s.is_settled(), "spring should settle: value={}", s.value);
        assert!((s.value - 1.0).abs() < 0.01, "should reach target");
    }

    #[test]
    fn spring_redirect_midflight_does_not_reset() {
        // The whole reason Spring exists over a tween: re-targeting mid-flight
        // redirects *continuously* from the current position, never snapping
        // back to a keyframe. Take a few steps, capture the value, reverse the
        // target, and confirm the very next step is near where it was (no jump)
        // — then that it settles at the new target.
        let mut s = Spring::new(0.0);
        s.set_target(1.0);
        for _ in 0..6 {
            s.step(1.0 / 60.0);
        }
        let before = s.value;
        assert!(before > 0.0, "should have moved off the start, got {before}");
        s.set_target(0.0); // reverse mid-flight
        let after = s.step(1.0 / 60.0);
        assert!(
            (after - before).abs() < 0.25,
            "redirect must be continuous, not a jump: {before} -> {after}"
        );
        for _ in 0..300 {
            s.step(1.0 / 60.0);
        }
        assert!(
            s.is_settled() && s.value.abs() < 0.02,
            "should settle at the new target, got {}",
            s.value
        );
    }

    #[test]
    fn spring_step_clamps_huge_dt() {
        // A monstrous dt (a stall) must not explode the simulation into NaN/inf.
        let mut s = Spring::with_preset(0.0, SpringPreset::Bouncy);
        s.set_target(1.0);
        let v = s.step(10.0); // clamped to 1/30 internally
        assert!(v.is_finite(), "clamped step must stay finite");
    }

    #[test]
    fn spring_snap_to_target_is_instant_and_still() {
        // Reduced-motion path: no travel, no velocity.
        let mut s = Spring::new(0.0);
        s.set_target(1.0);
        s.step(1.0 / 60.0);
        s.snap_to_target();
        assert_eq!(s.value, 1.0);
        assert_eq!(s.velocity, 0.0);
        assert!(s.is_settled());
    }

    #[test]
    fn start_delayed_reads_zero_before_it_begins() {
        let now = std::time::Instant::now();
        let starts = now + std::time::Duration::from_secs(10);
        let a = AnimState::start_delayed(std::time::Duration::from_millis(200), starts);
        assert_eq!(a.linear_at(now), 0.0, "not started yet → 0");
        assert!(!a.is_complete_at(now));
        // Well after it would have finished, it reads complete.
        let later = starts + std::time::Duration::from_secs(1);
        assert!(a.is_complete_at(later));
        assert_eq!(a.linear_at(later), 1.0);
    }

    #[test]
    fn ambient_breathe_stays_in_unit_range_and_varies() {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for i in 0..240 {
            let c = AmbientClock {
                seconds: i as f32 * 0.05,
            };
            let b = c.breathe(BACKGROUND_BREATHE_PERIOD_SECS);
            assert!((0.0..=1.0).contains(&b), "breathe out of range: {b}");
            min = min.min(b);
            max = max.max(b);
        }
        assert!(max - min > 0.5, "breathe should actually oscillate");
    }
}
