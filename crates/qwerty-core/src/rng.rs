//! Deterministic, seedable pseudo-random numbers for every v2 core function (§1.3, §0.7).

/// SplitMix64 increment (the 64-bit golden ratio).
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
/// Second stream-derivation multiplier (the xxHash/Murmur prime).
const DERIVE_MUL_B: u64 = 0xC2B2_AE3D_27D4_EB4F;
/// 2^-53: maps the top 53 bits of a `u64` onto [0, 1).
const UNIT_SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;
/// Floor for the Box-Muller `ln` argument: the smallest non-zero `uniform()` output, 2^-53.
const BOX_MULLER_LN_FLOOR: f64 = UNIT_SCALE;

/// SplitMix64 (Steele/Lea/Flood 2014). Deterministic, seedable, not cryptographic.
#[derive(Debug, Clone)]
pub struct Rng64 {
    state: u64,
}

impl Rng64 {
    pub const fn new(seed: u64) -> Self {
        Rng64 { state: seed }
    }

    /// Derive an independent stream: SplitMix64 of (seed ^ a·0x9E3779B97F4A7C15 ^ b·0xC2B2AE3D27D4EB4F).
    // SPEC: "SplitMix64 of x" is the first output of the generator seeded with x, and that output
    // seeds the stream. Seeding with x itself is not independent: a's multiplier is the increment,
    // so a carry-free x is new(seed ^ b·δ) advanced a draws. Products wrap modulo 2^64.
    pub fn derive(seed: u64, a: u64, b: u64) -> Self {
        let combined = seed ^ a.wrapping_mul(GOLDEN_GAMMA) ^ b.wrapping_mul(DERIVE_MUL_B);
        Rng64::new(Rng64::new(combined).next_u64())
    }

    /// state += 0x9E3779B97F4A7C15; z=state; z=(z^(z>>30))*0xBF58476D1CE4E5B9;
    /// z=(z^(z>>27))*0x94D049BB133111EB; z^(z>>31)
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// (next_u64() >> 11) as f64 · 2^-53, in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * UNIT_SCALE
    }

    /// lo + (hi − lo)·uniform().
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Lemire multiply-shift onto 0..n; n > 0. One `next_u64` per call (no rejection loop), so
    /// the call count is deterministic.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "Rng64::below needs n > 0");
        ((u128::from(self.next_u64()) * u128::from(n)) >> 64) as u64
    }

    /// Standard normal by Box-Muller, no caching: exactly two `uniform` draws per call.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(BOX_MULLER_LN_FLOOR);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn derived_streams_are_independent() {
        // Seeds 0, 2, 8, 42 share no low bits with a·γ, and 7 none with 8·γ: a raw-seeded derive
        // made those streams shifted copies of new(seed).
        for seed in [0, 2, 7, 8, 42] {
            let mut seen = BTreeSet::new();
            let mut parent = Rng64::new(seed);
            for _ in 0..64 {
                assert!(seen.insert(parent.next_u64()));
            }
            for a in 0..256 {
                for b in 0..4 {
                    let mut rng = Rng64::derive(seed, a, b);
                    for _ in 0..4 {
                        let draw = rng.next_u64();
                        assert!(seen.insert(draw), "derive({seed}, {a}, {b}) repeats a draw");
                    }
                }
            }
        }
        // The stream is seeded with SplitMix64's output for the combined value: RN1's first value here.
        assert_eq!(
            Rng64::derive(0, 0, 0).next_u64(),
            Rng64::new(0xE220_A839_7B1D_CDAF).next_u64()
        );
    }

    #[test]
    fn rn1_splitmix64_known_answer() {
        let mut rng = Rng64::new(0);
        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(rng.next_u64(), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn rng_draws_stay_in_range_and_are_deterministic() {
        let mut a = Rng64::derive(7, 3, 2);
        let mut b = Rng64::derive(7, 3, 2);
        for _ in 0..10_000 {
            let u = a.uniform();
            assert!((0.0..1.0).contains(&u));
            assert_eq!(u.to_bits(), b.uniform().to_bits());
            let k = a.below(5);
            assert!(k < 5);
            assert_eq!(k, b.below(5));
            let r = a.range(-2.0, 3.0);
            assert!((-2.0..3.0).contains(&r));
            assert_eq!(r.to_bits(), b.range(-2.0, 3.0).to_bits());
            let z = a.normal();
            assert!(z.is_finite());
            assert_eq!(z.to_bits(), b.normal().to_bits());
        }
        assert_ne!(
            Rng64::derive(7, 1, 0).next_u64(),
            Rng64::derive(7, 0, 1).next_u64()
        );
    }
}
