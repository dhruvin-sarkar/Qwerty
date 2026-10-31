//! Unit newtypes for time, frequency, rate, level and stream positions (§1.2).
//!
//! Raw `f64`/`u32` values for time, frequency, rate or level never appear in a
//! public signature (§0.4). The finite float newtypes cannot hold NaN or ±inf by
//! any path, because serde_json would write those as `null` and the file would
//! then fail to load (§0.5).

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum UnitError {
    SampleRateOutOfRange { hz: u32 },
    ChannelCountOutOfRange { channels: u32 },
    NotFinite { what: String },
    HostTimeOverflow,
}

impl fmt::Display for UnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnitError::SampleRateOutOfRange { hz } => write!(
                f,
                "sample rate {hz} Hz is outside {}..={} Hz",
                SampleRate::MIN_HZ,
                SampleRate::MAX_HZ
            ),
            UnitError::ChannelCountOutOfRange { channels } => write!(
                f,
                "channel count {channels} is outside 1..={}",
                ChannelCount::MAX
            ),
            UnitError::NotFinite { what } => write!(f, "{what} must be finite"),
            UnitError::HostTimeOverflow => write!(f, "host time does not fit in 64-bit ticks"),
        }
    }
}

impl std::error::Error for UnitError {}

/// Native device sample rate in Hz. Valid range 16_000..=192_000.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "u32", into = "u32")]
pub struct SampleRate(u32);

impl SampleRate {
    pub const MIN_HZ: u32 = 16_000;
    pub const MAX_HZ: u32 = 192_000;

    pub fn new(hz: u32) -> Result<Self, UnitError> {
        if (Self::MIN_HZ..=Self::MAX_HZ).contains(&hz) {
            Ok(SampleRate(hz))
        } else {
            Err(UnitError::SampleRateOutOfRange { hz })
        }
    }

    pub const fn hz(self) -> u32 {
        self.0
    }

    /// round-half-away-from-zero(fs · ms / 1000); the ONLY ms→samples conversion in the codebase.
    // SPEC: §1.2 does not define a negative duration; the float→usize cast saturates it to 0.
    pub fn samples_in(self, d: Milliseconds) -> usize {
        (f64::from(self.0) * d.get() / 1000.0).round() as usize
    }

    /// Duration of `frames` frames at this rate.
    pub fn duration_of(self, frames: u64) -> Milliseconds {
        // A u64 frame count at ≥ 16 kHz is at most ~1.2e18 ms, always finite.
        Milliseconds(frames as f64 * 1000.0 / f64::from(self.0))
    }
}

impl TryFrom<u32> for SampleRate {
    type Error = UnitError;
    fn try_from(hz: u32) -> Result<Self, Self::Error> {
        Self::new(hz)
    }
}

impl From<SampleRate> for u32 {
    fn from(rate: SampleRate) -> u32 {
        rate.0
    }
}

/// Number of interleaved channels in a capture stream. Valid range 1..=32.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "u16", into = "u16")]
pub struct ChannelCount(u16);

impl ChannelCount {
    pub const MAX: u16 = 32;

    pub fn new(n: u16) -> Result<Self, UnitError> {
        if (1..=Self::MAX).contains(&n) {
            Ok(ChannelCount(n))
        } else {
            Err(UnitError::ChannelCountOutOfRange {
                channels: u32::from(n),
            })
        }
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

impl TryFrom<u16> for ChannelCount {
    type Error = UnitError;
    fn try_from(n: u16) -> Result<Self, Self::Error> {
        Self::new(n)
    }
}

impl From<ChannelCount> for u16 {
    fn from(count: ChannelCount) -> u16 {
        count.0
    }
}

/// f64 newtypes. `new` is const and asserts finiteness. It is for literals and constants, where a
/// non-finite value becomes a compile error (`f64::is_finite` is const-stable). Computed values use
/// `try_new`. serde uses try_from. No path can build NaN or ±inf, which serde_json would write as
/// `null` (§0.5, RUST-RT-16).
macro_rules! finite_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
        #[serde(try_from = "f64", into = "f64")]
        pub struct $name(f64);

        impl $name {
            /// For literals and constants only: a non-finite argument fails const evaluation.
            pub const fn new(v: f64) -> Self {
                assert!(v.is_finite(), concat!(stringify!($name), " must be finite"));
                Self(v)
            }

            /// For computed values: `Err(NotFinite)` on NaN or ±inf.
            pub fn try_new(v: f64) -> Result<Self, UnitError> {
                if v.is_finite() {
                    Ok(Self(v))
                } else {
                    Err(UnitError::NotFinite {
                        what: stringify!($name).to_string(),
                    })
                }
            }

            pub const fn get(self) -> f64 {
                self.0
            }
        }

        impl TryFrom<f64> for $name {
            type Error = UnitError;
            fn try_from(v: f64) -> Result<Self, Self::Error> {
                Self::try_new(v)
            }
        }

        impl From<$name> for f64 {
            fn from(v: $name) -> f64 {
                v.0
            }
        }
    };
}

finite_newtype!(Hertz);
finite_newtype!(Milliseconds);
finite_newtype!(Seconds);
finite_newtype!(
    /// Power decibels: 10·log10(ratio) unless a doc comment says otherwise.
    ///
    /// A finite literal builds a constant:
    ///
    /// ```
    /// use qwerty_core::units::Decibels;
    /// const FLOOR: Decibels = Decibels::new(-140.0);
    /// assert_eq!(FLOOR.get(), -140.0);
    /// ```
    ///
    /// A non-finite literal does not compile (U2):
    ///
    /// ```compile_fail
    /// use qwerty_core::units::Decibels;
    /// const X: Decibels = Decibels::new(f64::NAN);
    /// let _ = X.get();
    /// ```
    Decibels
);

/// Absolute stream position in frames since the capture stream (or replay) started.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SampleIndex(pub u64);

/// Host monotonic clock in 100-ns ticks. On Windows this is QPC scaled to 100 ns, the same
/// domain as WASAPI's capture timestamp (`BufferInfo::timestamp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct HostTime(pub i64);

/// Signed difference of two host times, in 100-ns ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct HostDuration(pub i64);

/// 100-ns ticks per millisecond.
const TICKS_PER_MS: f64 = 10_000.0;

impl HostTime {
    pub const TICKS_PER_SECOND: i64 = 10_000_000;

    /// ticks = qpc · 10^7 / freq, computed in i128; Err(HostTimeOverflow) if it exceeds i64.
    // SPEC: §1.2 does not cover freq = 0 (QPC never reports it); the quotient is unbounded, so it
    // is reported as HostTimeOverflow instead of panicking on division by zero.
    pub fn from_qpc(qpc: i64, freq: i64) -> Result<Self, UnitError> {
        let ticks = (i128::from(qpc) * i128::from(Self::TICKS_PER_SECOND))
            .checked_div(i128::from(freq))
            .ok_or(UnitError::HostTimeOverflow)?;
        i64::try_from(ticks)
            .map(HostTime)
            .map_err(|_| UnitError::HostTimeOverflow)
    }

    pub fn checked_add(self, d: HostDuration) -> Option<Self> {
        self.0.checked_add(d.0).map(HostTime)
    }

    /// self − earlier (may be negative). Saturates at the i64 range, which real host times
    /// (QPC scaled to 100 ns) never approach.
    pub fn since(self, earlier: HostTime) -> HostDuration {
        HostDuration(self.0.saturating_sub(earlier.0))
    }
}

impl HostDuration {
    /// Rounds to the nearest tick (half away from zero); saturates at the i64 range.
    pub fn from_ms(ms: Milliseconds) -> Self {
        HostDuration((ms.get() * TICKS_PER_MS).round() as i64)
    }

    pub fn as_ms(self) -> Milliseconds {
        // Any i64 divided by 10^4 is finite.
        Milliseconds(self.0 as f64 / TICKS_PER_MS)
    }
}

/// Index of a zone inside one trained model / profile (0..K). Never persisted alone: always next
/// to the `Vec<ZoneId>` that defines the order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ZoneIndex(pub u16);

/// Index of a calibration tap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TapIndex(pub u32);

/// Detector-assigned, strictly increasing per Detector instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CandidateId(pub u64);

/// Serializable I/O failure, so error enums that cross the boundary can derive serde.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IoFailure {
    /// `format!("{:?}", e.kind())`
    pub kind: String,
    pub message: String,
}

impl From<&std::io::Error> for IoFailure {
    fn from(e: &std::io::Error) -> Self {
        IoFailure {
            kind: format!("{:?}", e.kind()),
            message: e.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NON_FINITE: [f64; 3] = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

    #[test]
    fn un1_finite_newtypes_reject_non_finite() {
        for v in NON_FINITE {
            assert!(Hertz::try_new(v).is_err(), "Hertz accepted {v}");
            assert!(
                Milliseconds::try_new(v).is_err(),
                "Milliseconds accepted {v}"
            );
            assert!(Seconds::try_new(v).is_err(), "Seconds accepted {v}");
            assert!(Decibels::try_new(v).is_err(), "Decibels accepted {v}");
        }
        assert!(serde_json::from_str::<Milliseconds>("1e400").is_err());
        // Finite values pass through and round-trip.
        let ms = Milliseconds::try_new(12.5).expect("finite");
        let json = serde_json::to_string(&ms).expect("serialize");
        assert_eq!(json, "12.5");
        assert_eq!(
            serde_json::from_str::<Milliseconds>(&json).expect("parse"),
            ms
        );
    }

    #[test]
    fn u2_non_finite_is_not_finite_and_null_is_refused() {
        for v in NON_FINITE {
            assert_eq!(
                Decibels::try_new(v),
                Err(UnitError::NotFinite {
                    what: "Decibels".to_string()
                })
            );
        }
        assert!(serde_json::from_str::<Decibels>("null").is_err());
        // The compile_fail half of U2 is the doctest on `Decibels`.
    }

    #[test]
    fn un2_host_time_from_qpc() {
        for qpc in [0, 1, -5, 123_456_789, i64::MAX, i64::MIN] {
            assert_eq!(HostTime::from_qpc(qpc, 10_000_000), Ok(HostTime(qpc)));
        }
        assert_eq!(HostTime::from_qpc(3, 3_000_000), Ok(HostTime(10)));
        assert_eq!(
            HostTime::from_qpc(i64::MAX, 1),
            Err(UnitError::HostTimeOverflow)
        );
    }

    #[test]
    fn un3_rate_and_channel_count_bounds() {
        assert_eq!(SampleRate::new(16_000).map(SampleRate::hz), Ok(16_000));
        assert_eq!(SampleRate::new(192_000).map(SampleRate::hz), Ok(192_000));
        assert_eq!(
            SampleRate::new(15_999),
            Err(UnitError::SampleRateOutOfRange { hz: 15_999 })
        );
        assert_eq!(
            SampleRate::new(192_001),
            Err(UnitError::SampleRateOutOfRange { hz: 192_001 })
        );
        assert!(serde_json::from_str::<SampleRate>("15999").is_err());
        assert!(serde_json::from_str::<SampleRate>("192001").is_err());
        assert_eq!(
            serde_json::from_str::<SampleRate>("48000")
                .map(SampleRate::hz)
                .ok(),
            Some(48_000)
        );

        assert_eq!(ChannelCount::new(1).map(ChannelCount::get), Ok(1));
        assert_eq!(ChannelCount::new(32).map(ChannelCount::get), Ok(32));
        assert_eq!(
            ChannelCount::new(0),
            Err(UnitError::ChannelCountOutOfRange { channels: 0 })
        );
        assert_eq!(
            ChannelCount::new(33),
            Err(UnitError::ChannelCountOutOfRange { channels: 33 })
        );
        assert!(serde_json::from_str::<ChannelCount>("0").is_err());
        assert!(serde_json::from_str::<ChannelCount>("33").is_err());
    }

    #[test]
    fn samples_in_rounds_half_away_from_zero() {
        let r48 = SampleRate::new(48_000).expect("valid rate");
        let r44 = SampleRate::new(44_100).expect("valid rate");
        assert_eq!(r48.samples_in(Milliseconds::new(1.0)), 48);
        assert_eq!(r48.samples_in(Milliseconds::new(2000.0)), 96_000);
        // 44.1 · 0.2 = 8.82 → 9; 44.1 · 10 = 441 exactly.
        assert_eq!(r44.samples_in(Milliseconds::new(0.2)), 9);
        assert_eq!(r44.samples_in(Milliseconds::new(10.0)), 441);
        // 16 kHz · 1/32 ms = 0.5 → 1 (half away from zero).
        let r16 = SampleRate::new(16_000).expect("valid rate");
        assert_eq!(r16.samples_in(Milliseconds::new(1.0 / 32.0)), 1);
        assert_eq!(r48.duration_of(480).get(), 10.0);
    }

    #[test]
    fn host_duration_ms_round_trip() {
        assert_eq!(
            HostDuration::from_ms(Milliseconds::new(1.5)),
            HostDuration(15_000)
        );
        assert_eq!(HostDuration(25_000).as_ms().get(), 2.5);
        assert_eq!(HostTime(30).since(HostTime(50)), HostDuration(-20));
        assert_eq!(HostTime(i64::MAX).checked_add(HostDuration(1)), None);
    }
}
