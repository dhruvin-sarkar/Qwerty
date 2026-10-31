//! Per-candidate descriptor: the level-free feature vector the zone model classifies (§1.7).

use crate::audio::ChannelLayout;
use serde::de::Error as _;
use serde::ser::{Error as _, SerializeStruct};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub const MONO_DIMS: usize = 38;
pub const MULTI_DIMS: usize = 10;
pub const DESCRIPTOR_DIMS: usize = MONO_DIMS + MULTI_DIMS; // 48

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DescriptorLayout {
    Mono { a: u16 },
    Stereo { a: u16, c: u16 },
}

impl DescriptorLayout {
    /// real.len()==1 -> Mono{a: real[0]}; else Stereo{a: real[0], c: *real.last()} (pair choice: OQ-7)
    pub fn from_channels(layout: &ChannelLayout) -> Self {
        match layout.real() {
            [a] => DescriptorLayout::Mono { a: *a },
            [a, .., c] => DescriptorLayout::Stereo { a: *a, c: *c },
            // INVARIANT: every ChannelLayout constructor rejects an empty `real`
            // (LayoutInvalid::NoRealChannel, LayoutError::AllChannelsSilent).
            [] => unreachable!("ChannelLayout::real is never empty"),
        }
    }

    /// 38 or 48.
    pub const fn dims(self) -> usize {
        match self {
            DescriptorLayout::Mono { .. } => MONO_DIMS,
            DescriptorLayout::Stereo { .. } => DESCRIPTOR_DIMS,
        }
    }

    /// [a] or [a, c].
    pub fn chosen(self) -> Vec<u16> {
        match self {
            DescriptorLayout::Mono { a } => vec![a],
            DescriptorLayout::Stereo { a, c } => vec![a, c],
        }
    }
}

/// values[dims()..] are exactly 0.0 for Mono. Serialized as {"layout":..,"values":[dims() floats]}
/// by a hand-written impl (serde arrays stop at 32 elements).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Descriptor {
    pub layout: DescriptorLayout,
    pub values: [f32; DESCRIPTOR_DIMS],
}

impl Descriptor {
    /// &values[..dims()].
    pub fn as_slice(&self) -> &[f32] {
        &self.values[..self.layout.dims()]
    }

    /// Every one of the 48 stored values is finite.
    pub fn is_finite(&self) -> bool {
        self.values.iter().all(|v| v.is_finite())
    }

    /// Largest |self − other| over the layout's dims; f32::INFINITY if the layouts differ.
    /// A NaN in either operand gives NaN, so a tolerance check on the result fails.
    pub fn max_abs_diff(&self, other: &Descriptor) -> f32 {
        if self.layout != other.layout {
            return f32::INFINITY;
        }
        let mut max = 0.0f32;
        for (a, b) in self.as_slice().iter().zip(other.as_slice()) {
            let d = (a - b).abs();
            if d.is_nan() {
                return f32::NAN;
            }
            max = max.max(d);
        }
        max
    }
}

impl Serialize for Descriptor {
    /// Refuses a non-finite value: serde_json would write it as `null`, and the file would then
    /// fail to load (§0.5).
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if let Some(index) = self.as_slice().iter().position(|v| !v.is_finite()) {
            return Err(S::Error::custom(format!(
                "descriptor value {index} is not finite"
            )));
        }
        let mut s = serializer.serialize_struct("Descriptor", 2)?;
        s.serialize_field("layout", &self.layout)?;
        s.serialize_field("values", self.as_slice())?;
        s.end()
    }
}

/// The unvalidated serde shape of `Descriptor`, used by deserialization only.
#[derive(Deserialize)]
struct RawDescriptor {
    layout: DescriptorLayout,
    values: Vec<f32>,
}

impl<'de> Deserialize<'de> for Descriptor {
    /// Rejects a `values` length other than `layout.dims()` and any non-finite value.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawDescriptor::deserialize(deserializer)?;
        let dims = raw.layout.dims();
        if raw.values.len() != dims {
            return Err(D::Error::custom(format!(
                "descriptor has {} values, its layout needs {dims}",
                raw.values.len()
            )));
        }
        if let Some(index) = raw.values.iter().position(|v| !v.is_finite()) {
            return Err(D::Error::custom(format!(
                "descriptor value {index} is not finite"
            )));
        }
        let mut values = [0.0f32; DESCRIPTOR_DIMS];
        values[..dims].copy_from_slice(&raw.values);
        Ok(Descriptor {
            layout: raw.layout,
            values,
        })
    }
}

/// The golden dimension names of §1.7.3, in index order.
const DESCRIPTOR_NAMES: [&str; DESCRIPTOR_DIMS] = [
    "shape_c1",
    "shape_c2",
    "shape_c3",
    "shape_c4",
    "shape_c5",
    "shape_c6",
    "shape_c7",
    "shape_c8",
    "shape_c9",
    "shape_c10",
    "shape_c11",
    "shape_c12",
    "shape_c13",
    "shape_c14",
    "shape_c15",
    "shape_c16",
    "attack_c0",
    "attack_c1",
    "attack_c2",
    "attack_c3",
    "attack_c4",
    "attack_c5",
    "attack_c6",
    "attack_c7",
    "decay_g0",
    "decay_g1",
    "decay_g2",
    "decay_g3",
    "decay_g4",
    "decay_g5",
    "decay_g6",
    "decay_g7",
    "dispersion_o0",
    "dispersion_o1",
    "dispersion_o2",
    "dispersion_o3",
    "dispersion_o4",
    "attack_crest_db",
    "ild_g0",
    "ild_g1",
    "ild_g2",
    "ild_g3",
    "ild_g4",
    "ild_g5",
    "ild_g6",
    "ild_g7",
    "itd_ms",
    "coherence",
];

/// Golden list, §1.7.3: the first `layout.dims()` names.
pub fn descriptor_names(layout: DescriptorLayout) -> Vec<&'static str> {
    DESCRIPTOR_NAMES[..layout.dims()].to_vec()
}

#[cfg(test)]
mod type_tests {
    use super::*;
    use crate::units::ChannelCount;

    fn mono() -> Descriptor {
        let mut values = [0.0f32; DESCRIPTOR_DIMS];
        for (i, v) in values[..MONO_DIMS].iter_mut().enumerate() {
            *v = i as f32 * 0.25 - 3.0;
        }
        Descriptor {
            layout: DescriptorLayout::Mono { a: 0 },
            values,
        }
    }

    #[test]
    fn descriptor_serde_round_trips_and_validates() {
        let d = mono();
        let json = serde_json::to_string(&d).expect("serialize");
        let back: Descriptor = serde_json::from_str(&json).expect("round-trip");
        assert_eq!(back, d);
        assert_eq!(back.as_slice().len(), MONO_DIMS);

        let value: serde_json::Value = serde_json::from_str(&json).expect("json");
        assert_eq!(value["values"].as_array().map(Vec::len), Some(MONO_DIMS));

        // Wrong length for the layout.
        let short = json.replacen("[-3.0,", "[", 1);
        assert!(serde_json::from_str::<Descriptor>(&short).is_err());
        // Out of f32 range is non-finite after the cast, and refused.
        let huge = json.replacen("[-3.0,", "[1e39,", 1);
        assert!(serde_json::from_str::<Descriptor>(&huge).is_err());
        // A non-finite value cannot be serialized.
        let mut bad = d;
        bad.values[3] = f32::NAN;
        assert!(!bad.is_finite());
        assert!(serde_json::to_string(&bad).is_err());
    }

    #[test]
    fn descriptor_layout_and_diff() {
        let stereo = ChannelLayout::all_real(ChannelCount::new(4).expect("count"));
        let layout = DescriptorLayout::from_channels(&stereo);
        assert_eq!(layout, DescriptorLayout::Stereo { a: 0, c: 3 });
        assert_eq!(layout.dims(), DESCRIPTOR_DIMS);
        assert_eq!(layout.chosen(), vec![0, 3]);
        let mono_layout = DescriptorLayout::from_channels(&ChannelLayout::all_real(
            ChannelCount::new(1).expect("count"),
        ));
        assert_eq!(mono_layout, DescriptorLayout::Mono { a: 0 });
        assert_eq!(mono_layout.chosen(), vec![0]);

        let names = descriptor_names(layout);
        assert_eq!(names.len(), DESCRIPTOR_DIMS);
        assert_eq!(descriptor_names(mono_layout).len(), MONO_DIMS);
        assert_eq!(names[MONO_DIMS - 1], "attack_crest_db");
        assert_eq!(names[DESCRIPTOR_DIMS - 1], "coherence");

        let a = mono();
        let mut b = a;
        b.values[5] += 0.5;
        assert_eq!(a.max_abs_diff(&b), 0.5);
        b.values[6] = f32::NAN;
        assert!(a.max_abs_diff(&b).is_nan());
        let other = Descriptor {
            layout: DescriptorLayout::Mono { a: 1 },
            values: a.values,
        };
        assert_eq!(a.max_abs_diff(&other), f32::INFINITY);
    }
}
