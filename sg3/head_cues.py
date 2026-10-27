"""Head-pose cues: head_nod (from pitch) and head_side_tilt (from roll)."""

from __future__ import annotations

import math
from typing import List, Optional

from .config import SG3Config
from .types import CueFlag, HeadPose


def _finite(*vals: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)


def head_cues(head_pose: Optional[HeadPose], cfg: SG3Config) -> Optional[List[CueFlag]]:
    """Return observed cues, [] if none observed, or None if head pose is unusable."""
    if head_pose is None or not _finite(head_pose.pitch, head_pose.roll):
        return None

    flags: List[CueFlag] = []
    if cfg.pitch_down_sign * head_pose.pitch >= cfg.nod_pitch_deg:
        flags.append(CueFlag.HEAD_NOD)
    if abs(head_pose.roll) >= cfg.tilt_roll_deg:
        flags.append(CueFlag.HEAD_SIDE_TILT)
    return flags
