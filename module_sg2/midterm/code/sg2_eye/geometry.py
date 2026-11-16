"""Eye geometry helpers: Eye Aspect Ratio (EAR) and eye-crop extraction.

Input points follow the SG1 V1 ordering (see module_sg1 landmark_map.md):
    p1 outer corner, p2/p3 upper lid, p4 inner corner, p5/p6 lower lid
in full-frame pixel coordinates.
"""
import cv2
import numpy as np


def eye_aspect_ratio(pts):
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|)  (Soukupova & Cech, 2016).

    Returns None if the points are missing or degenerate.
    """
    if pts is None:
        return None
    p = np.asarray(pts, dtype=np.float32)
    if p.shape != (6, 2):
        return None
    horizontal = np.linalg.norm(p[0] - p[3])
    if horizontal < 1e-6:
        return None
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    return float(vertical / (2.0 * horizontal))


def eye_width(pts):
    """Corner-to-corner eye width in pixels (used for pose-aware fusion)."""
    p = np.asarray(pts, dtype=np.float32)
    return float(np.linalg.norm(p[0] - p[3]))


def eye_crop(gray, pts, size=32, scale=1.6):
    """Square, roll-corrected grey crop centred on the eye.

    The crop side is `scale` x eye width and the eye corners are rotated to
    horizontal, so one cv2.warpAffine call gives a size x size patch.
    """
    p = np.asarray(pts, dtype=np.float32)
    centre = p.mean(axis=0)
    v = p[3] - p[0]
    if v[0] < 0:  # always measure the angle image-left -> image-right
        v = -v
    angle = np.degrees(np.arctan2(v[1], v[0]))
    side = max(scale * float(np.linalg.norm(v)), 4.0)
    m = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), angle, size / side)
    m[0, 2] += size / 2.0 - centre[0]
    m[1, 2] += size / 2.0 - centre[1]
    return cv2.warpAffine(gray, m, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
