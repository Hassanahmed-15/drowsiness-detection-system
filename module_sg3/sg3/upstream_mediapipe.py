"""Optional upstream front end: MediaPipe FaceMesh -> face_valid, mouth ROI, mouth_pts, head_pose.

SG-3 itself does NOT depend on this file. It exists so SG-3 can be run on a raw video
before the team's real face / head-pose modules are wired in. Requires `mediapipe` and `opencv-python`.
"""

from __future__ import annotations


from dataclasses import dataclass
from typing import Optional, Tuple
import mediapipe as mp
import numpy as np
import cv2
from .types import HeadPose

# FaceMesh landmark indices
_POSE_IDX = (1, 152, 33, 263, 61, 291)          # nose, chin, eye-L, eye-R, mouth-L, mouth-R
# 8-point inner-lip MAR order: p1 left corner, p2 upper-left, p3 upper-mid, p4 upper-right,
#                              p5 right corner, p6 lower-right, p7 lower-mid, p8 lower-left
_MOUTH_PTS_IDX = (78, 82, 13, 312, 308, 317, 14, 87)
_LIPS_OUTER_IDX = (61, 291, 0, 17)              # for the ROI box

# Generic 3D face model (mm), y up, matching _POSE_IDX order
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),            # nose tip
        (0.0, -330.0, -65.0),       # chin
        (-225.0, 170.0, -135.0),    # eye outer, image-left
        (225.0, 170.0, -135.0),     # eye outer, image-right
        (-150.0, -150.0, -125.0),   # mouth corner, image-left
        (150.0, -150.0, -125.0),    # mouth corner, image-right
    ],
    dtype=np.float64,
)

# +1 if the raw angle already follows "pitch > 0 = head down"; verified in tests/test_head_pose.py
PITCH_SIGN = 1.0


def estimate_head_pose(image_points: np.ndarray, frame_w: int, frame_h: int) -> Optional[HeadPose]:
    """solvePnP on 6 face points (pixels, order = _POSE_IDX). Returns degrees."""
    

    focal = float(frame_w)
    cam = np.array([[focal, 0, frame_w / 2.0], [0, focal, frame_h / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS, np.asarray(image_points, dtype=np.float64), cam, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rmat)[0]           # degrees, about x, y, z
    pitch = ((angles[0] % 360.0) - 180.0) * PITCH_SIGN
    yaw = float(angles[1])
    roll = float(angles[2])
    return HeadPose(pitch=float(pitch), yaw=yaw, roll=roll)


@dataclass
class UpstreamResult:
    face_valid: bool
    mouth_roi: Optional[np.ndarray] = None
    mouth_pts: Optional[np.ndarray] = None
    head_pose: Optional[HeadPose] = None


class MediaPipeUpstream:
    def __init__(self, roi_pad: float = 0.4):
        
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self.roi_pad = roi_pad

    def close(self) -> None:
        self._mesh.close()

    def __call__(self, frame_bgr: np.ndarray) -> UpstreamResult:

        h, w = frame_bgr.shape[:2]
        res = self._mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return UpstreamResult(face_valid=False)

        lm = res.multi_face_landmarks[0].landmark
        xy = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float64)

        head_pose = estimate_head_pose(xy[list(_POSE_IDX)], w, h)
        mouth_pts = xy[list(_MOUTH_PTS_IDX)]
        roi = _crop_mouth(frame_bgr, xy[list(_LIPS_OUTER_IDX)], self.roi_pad)
        return UpstreamResult(True, roi, mouth_pts, head_pose)


def _crop_mouth(frame: np.ndarray, lip_pts: np.ndarray, pad: float) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x0, y0 = lip_pts.min(axis=0)
    x1, y1 = lip_pts.max(axis=0)
    side = max(x1 - x0, y1 - y0) * (1.0 + 2 * pad)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    xa, xb = int(max(0, cx - side / 2)), int(min(w, cx + side / 2))
    ya, yb = int(max(0, cy - side / 2)), int(min(h, cy + side / 2))
    if xb <= xa or yb <= ya:
        return None
    return frame[ya:yb, xa:xb].copy()
