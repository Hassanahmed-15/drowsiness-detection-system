"""SG-3 entry point: per-frame yawn probability (from MAR) + head-pose cues."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import SG3Config
from .head_cues import head_cues
from .types import HeadPose, MouthOutput
from .yawn import mouth_aspect_ratio, yawn_prob_from_mar


class SG3:
    """Stateless per-frame processor.

    Yawn evidence comes only from the Mouth Aspect Ratio computed on `mouth_pts`.
    `mouth_roi` is accepted to match the module interface but is not used by MAR.

    Validity rules (decisions made here; confirm in the spec):
      * face_valid == False       -> mouth_valid False, yawn_prob None, cue_flags None
      * mouth_pts missing/invalid -> mouth_valid False, yawn_prob None
      * cue_flags depends only on face_valid and head_pose, NOT on mouth_valid:
          head_pose missing/invalid -> None, otherwise a list (possibly empty)
    SG-3 never smooths over time. Turning yawn_prob into events is SG-4's job.
    """

    def __init__(self, cfg: Optional[SG3Config] = None):
        self.cfg = cfg or SG3Config()

    def process(
        self,
        frame_id: int,
        timestamp_ms: int,
        face_valid: bool,
        mouth_roi: Optional[np.ndarray] = None,   # unused by MAR; kept for the interface
        head_pose: Optional[HeadPose] = None,
        mouth_pts=None,
    ) -> MouthOutput:
        if not face_valid:
            return MouthOutput(frame_id, timestamp_ms, False, None, None)

        mar = mouth_aspect_ratio(mouth_pts)
        yawn_prob = None if mar is None else yawn_prob_from_mar(mar, self.cfg)
        return MouthOutput(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            mouth_valid=yawn_prob is not None,
            yawn_prob=yawn_prob,
            cue_flags=head_cues(head_pose, self.cfg),
        )