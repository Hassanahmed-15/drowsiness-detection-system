"""Data types for SG-3 (yawn and head-pose cues)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class CueFlag(str, Enum):
    """Cues SG-3 can report per frame. New values need a team-level change request."""

    HEAD_NOD = "head_nod"                # head drooping / nodding forward (from pitch)
    HEAD_SIDE_TILT = "head_side_tilt"    # head tilted sideways (from roll)


@dataclass(frozen=True)
class HeadPose:
    """Head pose in DEGREES.

    Convention used by SG-3 (upstream must match, or set `pitch_down_sign` in config):
      pitch > 0  -> head pitched DOWN (nodding forward)
      roll       -> sideways tilt, sign is irrelevant (SG-3 uses |roll|)
      yaw        -> accepted but not used by SG-3 yet
    """

    pitch: float
    yaw: float
    roll: float


@dataclass
class MouthOutput:
    """The `mouth` output block."""

    frame_id: int
    timestamp_ms: int
    mouth_valid: bool
    yawn_prob: Optional[float]               # 0..1, None if mouth_valid is False
    cue_flags: Optional[List[CueFlag]]       # [] = none observed, None = not measured

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "mouth_valid": self.mouth_valid,
            "yawn_prob": self.yawn_prob,
            "cue_flags": None if self.cue_flags is None else [c.value for c in self.cue_flags],
        }
