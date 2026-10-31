//! Calibration session v2: the guided ambient, zone and negatives steps that produce a trained model (§1.14).

use serde::{Deserialize, Serialize};

/// Pre-placed by WP-00; WP-B2 MUST NOT change it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EnvironmentQuality {
    Quiet,
    Moderate,
    TooNoisy,
}
