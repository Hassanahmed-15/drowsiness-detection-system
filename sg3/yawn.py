"""Per-frame yawn probability from the Mouth Aspect Ratio (MAR).

MAR (inner-lip, 8-point form used in the yawn-detection literature):

    MAR = ( ||p2 - p8|| + ||p3 - p7|| + ||p4 - p6|| ) / ( 2 * ||p1 - p5|| )

          p2  p3  p4
    p1 ---                --- p5        (p1, p5 = mouth corners)
          p8  p7  p6

`mouth_pts` convention: array of shape (8, 2), ordered
    [p1 left corner, p2 upper-left, p3 upper-mid, p4 upper-right,
     p5 right corner, p6 lower-right, p7 lower-mid, p8 lower-left]
in any consistent coordinate space (pixels or normalized); MAR is a ratio.

A closed mouth gives MAR near 0, talking roughly 0.2 to 0.4, a yawn 0.6 and above.
`yawn_prob` maps MAR through a sigmoid centred on `cfg.mar_center` so that SG-4 gets a
smooth 0..1 score. The raw decision threshold is therefore `mar_center`.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .config import SG3Config


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def mouth_aspect_ratio(mouth_pts: Optional[Sequence[Sequence[float]]]) -> Optional[float]:
    """8-point MAR. Returns None if the points are missing, malformed, non-finite or degenerate."""
    if mouth_pts is None:
        return None
    pts = np.asarray(mouth_pts, dtype=float)
    if pts.shape != (8, 2) or not np.all(np.isfinite(pts)):
        return None
    p1, p2, p3, p4, p5, p6, p7, p8 = pts
    width = float(np.linalg.norm(p1 - p5))
    if width < 1e-6:
        return None
    vertical = (
        np.linalg.norm(p2 - p8) + np.linalg.norm(p3 - p7) + np.linalg.norm(p4 - p6)
    )
    return float(vertical) / (2.0 * width)


def yawn_prob_from_mar(mar: float, cfg: SG3Config) -> float:
    return _sigmoid((mar - cfg.mar_center) / cfg.mar_scale)
