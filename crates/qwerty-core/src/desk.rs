//! Synthetic desk fixtures for tests, benches and the soak (§1.19).
//!
//! For invariance tests, false-accept regression and stress only: never an accuracy gate, and
//! never tuned to chase one (ADR-006). Every parameter is frozen.

// ---------------------------------------------------------------------------
// CleanSignatures: the frozen v1 fixture (§1.19.2), moved byte-for-byte from `soak.rs`.
// ---------------------------------------------------------------------------

/// The v1 fixture's sample rate, frozen locally (was `features::SAMPLE_RATE_HZ`).
const FIXTURE_RATE: u32 = 44_100;
/// The v1 fixture's clip length, frozen locally (was `features::FEATURE_WINDOW_SAMPLES`).
const FIXTURE_CLIP: usize = 4096;

// ---------------------------------------------------------------------------
// Deterministic PRNG (no dependency; keeps the soak reproducible)
// ---------------------------------------------------------------------------

pub struct Lcg(u64);
impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    /// Uniform in [0, 1).
    pub fn next_unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// Uniform in [-1, 1).
    pub fn next_bipolar(&mut self) -> f32 {
        self.next_unit() * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Synthetic desk: distinct acoustic signature per zone
// ---------------------------------------------------------------------------

const LEAD_SILENCE: usize = 256 * 8;

pub struct CleanSignatures {
    /// Resonant partials (Hz) for each zone — distinct sets make zones
    /// spectrally separable, mirroring how desk positions resonate differently.
    zone_partials: Vec<Vec<f32>>,
}

impl CleanSignatures {
    pub fn new(zone_count: usize) -> Self {
        // Base frequency per zone spread across the audible band, each with a
        // small harmonic stack. NOTE: this fixture gives every zone the *same*
        // harmonic-ratio template scaled by a per-zone fundamental, so it is a
        // faithful sanity check only at the default zone count it was validated
        // against (≈99% at 4 zones). It is deliberately NOT used to assert a
        // multi-zone *accuracy* floor: at higher zone counts the scaled-template
        // signatures crowd together in a way that reflects the fixture's own
        // degeneracy, not the classifier — real multi-zone accuracy is gated by
        // the hardware-based ≥92% row in ACCEPTANCE.md, not by this generator.
        // (The `--zones` flag on the soak example exists to *explore* that
        // synthetic separability, not to gate on it.)
        let zone_partials = (0..zone_count)
            .map(|z| {
                let base = 700.0 + z as f32 * 430.0;
                vec![base, base * 1.9, base * 2.7]
            })
            .collect();
        Self { zone_partials }
    }

    /// A tap clip for `zone`: lead silence, then a fast-decaying multi-partial
    /// impulse, then trailing silence so a full window fits after the onset.
    pub fn tap(&self, zone: usize, rng: &mut Lcg, jitter: f32) -> Vec<f32> {
        let n = FIXTURE_CLIP;
        let mut clip = vec![0.0f32; LEAD_SILENCE];
        let partials = &self.zone_partials[zone];
        for i in 0..n {
            let t = i as f32 / FIXTURE_RATE as f32;
            let env = (-t * 130.0).exp();
            let mut s = 0.0f32;
            for (k, &f) in partials.iter().enumerate() {
                let fj = f * (1.0 + jitter * rng.next_bipolar() * 0.02);
                let amp = 1.0 / (k as f32 + 1.0);
                s += amp * (2.0 * std::f32::consts::PI * fj * t).sin();
            }
            clip.push(0.6 * env * s);
        }
        clip.extend(std::iter::repeat_n(0.0, n));
        clip
    }

    /// A sustained (non-tap) clip: a steady tone or a slowly amplitude-modulated
    /// broadband sound — energy that does not decay, so it must be rejected.
    pub fn sustained(&self, rng: &mut Lcg) -> Vec<f32> {
        let n = FIXTURE_CLIP * 2;
        let mut clip = vec![0.0f32; LEAD_SILENCE];
        let speechlike = rng.next_unit() > 0.5;
        let f = 300.0 + rng.next_unit() * 500.0;
        for i in 0..n {
            let t = i as f32 / FIXTURE_RATE as f32;
            let s = if speechlike {
                // Amplitude-modulated multi-tone, like voiced speech.
                let am = 0.6 + 0.4 * (2.0 * std::f32::consts::PI * 5.0 * t).sin();
                am * ((2.0 * std::f32::consts::PI * f * t).sin()
                    + 0.5 * (2.0 * std::f32::consts::PI * f * 2.5 * t).sin())
            } else {
                (2.0 * std::f32::consts::PI * f * t).sin()
            };
            clip.push(0.5 * s);
        }
        clip.extend(std::iter::repeat_n(0.0, FIXTURE_CLIP));
        clip
    }

    /// A *foreign* tap-like transient: a fast-decaying broadband noise burst.
    /// Crucially it has the SAME sharp attack + fast decay as a real tap, so it
    /// PASSES the onset detector's sustain gate and reaches the classifier —
    /// unlike [`sustained`], which onset rejects before the classifier sees it.
    /// But its spectrum is broadband noise, matching no calibrated zone's peaky
    /// harmonic signature, so the classifier's *novelty gate* must reject it.
    /// This is what makes the novelty gate actually testable end-to-end.
    pub fn foreign_tap(&self, rng: &mut Lcg) -> Vec<f32> {
        let n = FIXTURE_CLIP;
        let mut clip = vec![0.0f32; LEAD_SILENCE];
        for i in 0..n {
            let t = i as f32 / FIXTURE_RATE as f32;
            let env = (-t * 130.0).exp(); // same decay envelope as a real tap
            clip.push(0.6 * env * rng.next_bipolar());
        }
        clip.extend(std::iter::repeat_n(0.0, n));
        clip
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FNV-1a 64 offset basis and prime: a hash that is stable across Rust versions, unlike
    /// `DefaultHasher`.
    const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
    /// K4 golden hash of 4 zones × 3 clips, recorded from the pre-move v1 generator.
    const K4_GOLDEN: u64 = 0x6DA1_1BAD_C8FF_61FB;

    /// Folds the clip length and every sample's little-endian bit pattern into `hash`.
    fn fnv1a_clip(mut hash: u64, clip: &[f32]) -> u64 {
        let len = clip.len() as u64;
        for byte in len
            .to_le_bytes()
            .into_iter()
            .chain(clip.iter().flat_map(|s| s.to_bits().to_le_bytes()))
        {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    #[test]
    fn k4_clean_signatures_match_v1_generator() {
        // SPEC: "4 zones × 3 clips" read as one clip from each of the three generators per
        // zone (tap, sustained, foreign tap), drawn from one Lcg stream seeded as the soak's.
        let desk = CleanSignatures::new(4);
        let mut rng = Lcg::new(1);
        let mut hash = FNV_OFFSET;
        for zone in 0..4 {
            hash = fnv1a_clip(hash, &desk.tap(zone, &mut rng, 1.0));
            hash = fnv1a_clip(hash, &desk.sustained(&mut rng));
            hash = fnv1a_clip(hash, &desk.foreign_tap(&mut rng));
        }
        assert_eq!(hash, K4_GOLDEN, "got {hash:#018X}");
    }
}
