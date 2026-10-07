"""SG1-compatible landmark source so SG2 can be tested without waiting for SG1.

Re-implements the SG1 V1 face block with the SAME MediaPipe Face Landmarker
model and the SAME landmark indices as module_sg1 (landmark_map.md). When the
real SG1 module is merged, SG2 consumes SG1's blocks directly instead.

Extra (non-V1) field: `blendshapes` (eyeBlinkLeft/Right), only filled when
with_blendshapes=True. It is used for the Week 5 Config E experiment.
"""
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

# SG1 landmark_map.md (driver's own left/right)
LEFT_EYE = [263, 387, 385, 362, 380, 373]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 13, 291, 14]
HEAD_POSE = [1, 152, 33, 263, 61, 291]
# generic 3D face model (mm) for the HEAD_POSE points, image-left = -x
MODEL_3D = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
                     (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)])


def _box(pts, margin=0.0):
    p = np.asarray(pts)
    x1, y1 = p.min(axis=0)
    x2, y2 = p.max(axis=0)
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return [round(float(x1 - mx), 1), round(float(y1 - my), 1),
            round(float(x2 + mx), 1), round(float(y2 + my), 1)]


def head_pose(pts2d, w, h):
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, np.asarray(pts2d, dtype=np.float64), cam, np.zeros(4),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles
    pitch = (pitch + 180) % 360 - 180  # camera looks along -z of the model
    pitch = pitch - 180 if pitch > 90 else (pitch + 180 if pitch < -90 else pitch)
    return {"yaw_deg": round(float(yaw), 1), "pitch_deg": round(float(pitch), 1),
            "roll_deg": round(float(roll), 1)}


class SG1LandmarkSource:
    def __init__(self, model_path, min_confidence=0.5, with_blendshapes=False):
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_face_blendshapes=with_blendshapes)
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.with_blendshapes = with_blendshapes

    def process(self, frame_bgr, frame_id, timestamp_ms, keep_landmarks=False):
        h, w = frame_bgr.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self.landmarker.detect_for_video(image, int(timestamp_ms))
        block = {"frame_id": frame_id, "timestamp_ms": timestamp_ms, "face_detected": False,
                 "face_valid": False, "face_confidence": None, "face_box": None,
                 "left_eye_pts": None, "right_eye_pts": None, "mouth_pts": None, "landmarks": None,
                 "left_eye_roi_box": None, "right_eye_roi_box": None, "mouth_roi_box": None,
                 "head_pose": None}
        if not res.face_landmarks:
            return block
        lm = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], dtype=np.float32)
        pick = lambda idx: [[round(float(x), 2), round(float(y), 2)] for x, y in lm[idx]]
        block.update({
            "face_detected": True, "face_valid": True,
            "face_box": _box(lm),
            "left_eye_pts": pick(LEFT_EYE), "right_eye_pts": pick(RIGHT_EYE), "mouth_pts": pick(MOUTH),
            "left_eye_roi_box": _box(lm[LEFT_EYE], 0.3), "right_eye_roi_box": _box(lm[RIGHT_EYE], 0.3),
            "mouth_roi_box": _box(lm[MOUTH], 0.3),
            "head_pose": head_pose(lm[HEAD_POSE], w, h),
        })
        if keep_landmarks:
            block["landmarks"] = lm.round(2).tolist()
        if self.with_blendshapes and res.face_blendshapes:
            block["blendshapes"] = {c.category_name: round(float(c.score), 4)
                                    for c in res.face_blendshapes[0] if "Blink" in c.category_name}
        return block

    def close(self):
        self.landmarker.close()
