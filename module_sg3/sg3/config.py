"""Tunable settings for SG-3. Thresholds are SG-3's to define (see spec)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SG3Config:
    # ---- head cues (degrees) ----
    nod_pitch_deg: float = 20.0        # pitch >= this (head down) -> head_nod
    tilt_roll_deg: float = 20.0        # |roll| >= this -> head_side_tilt
    pitch_down_sign: float = 1.0       # set to -1.0 if upstream reports pitch>0 for head UP

    # ---- yawn from MAR ----
    mar_center: float = 0.60           # MAR at which yawn_prob = 0.5 (the decision threshold)
    mar_scale: float = 0.08            # sigmoid steepness (smaller = sharper)
