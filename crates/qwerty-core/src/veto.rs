//! Input and self-output veto: OS input events and Qwerty's own sound output that disqualify a candidate (§1.13).

use crate::units::HostTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InputKind {
    KeyDown,
    KeyUp,
    MouseButtonDown,
    MouseButtonUp,
}

/// Coarse OS input event. Never carries key identity, position or device (ADR-004 privacy).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputEvent {
    pub time: HostTime,
    pub kind: InputKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SelfOutputKind {
    Speech,
    Sound,
    Toast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OutputId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SelfOutputEvent {
    Began {
        id: OutputId,
        kind: SelfOutputKind,
        at: HostTime,
    },
    Ended {
        id: OutputId,
        at: HostTime,
    },
}
