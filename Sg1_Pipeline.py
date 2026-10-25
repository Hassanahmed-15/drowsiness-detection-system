#!/usr/bin/env python3
"""
SG-1 face and landmarks module, aligned to "Driver Drowsiness Monitoring:
Interface Specification V1 (DRAFT)".

Converted from week5_CV_project_SG_1.ipynb and reworked so that its output
follows the spec:

  * Serialized record (JSON, one per frame):
        {"interface_version": "1.0", "frame_id": int, "timestamp_ms": int,
         "face": { ...SG-1 block, spec section 4... }}
    SG-1 only produces the `face` block. The `eyes`, `mouth`, `temporal` and
    `decision` blocks belong to SG-2 to SG-5 and are NOT emitted here.
  * In-memory packet: the record plus the three ROI images
    (left_eye_roi, right_eye_roi, mouth_roi) as uint8 BGR arrays.
  * Boxes are {x, y, width, height} in full-frame pixels (top-left origin).
  * "left" = the driver's own left eye (appears on the image's right side).
  * JSON true/false and null (never 0/1, NaN, or last-good-value).

Usage:
  python sg1_face_pipeline.py --video "path/to/Video Project.mp4"
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

INTERFACE_VERSION = "1.0"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = "face_landmarker.task"

# ----------------------------------------------------------------------------
# Landmark map (spec D-02). Record the same table in documentation/landmark_map.md.
# MediaPipe Face Landmarker (478 landmarks). EAR point order, spec section 5:
#   p1 = outer corner, p2/p3 = upper lid, p4 = inner corner, p5/p6 = lower lid
# "Left" = the driver's own left eye (right side of an un-mirrored image).
# ----------------------------------------------------------------------------
LEFT_EYE_IDX = [263, 387, 385, 362, 380, 373]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
# Mouth order: corner (image-left), upper lip centre, corner (image-right), lower lip centre
MOUTH_IDX = [61, 13, 291, 14]

# ROI margins in pixels (spec D-16: native-resolution crop with margin)
EYE_ROI_MARGIN = 15
MOUTH_ROI_MARGIN = 20

# face_valid plausibility limits. PROPOSED values, not in the spec. Tune them.
MIN_FACE_SIZE_PX = 40
MIN_ROI_SIZE_PX = 8

# Shared parameter (spec section 2). Normally read from config/shared.yaml.
DEFAULT_FACE_CONF_MIN = 0.5

FACE_BLOCK_KEYS = [
    "face_detected", "face_valid", "face_confidence", "face_box",
    "left_eye_pts", "right_eye_pts", "mouth_pts",
    "left_eye_roi_box", "right_eye_roi_box", "mouth_roi_box", "head_pose",
]
NULLABLE_WHEN_NOT_VALID = [
    "left_eye_pts", "right_eye_pts", "mouth_pts",
    "left_eye_roi_box", "right_eye_roi_box", "mouth_roi_box", "head_pose",
]


# ----------------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------------
def ensure_model(model_path=MODEL_PATH):
    """Download the Face Landmarker model if it is not already on disk."""
    if os.path.exists(model_path):
        print(f"Model found: {model_path}")
        return
    print(f"Downloading model to {model_path} ...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("Model downloaded")


def load_face_conf_min(config_path):
    """Read face_conf_min from config/shared.yaml (spec section 2), if given."""
    if not config_path:
        return DEFAULT_FACE_CONF_MIN
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML is needed for --config. Run: pip install pyyaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "face_conf_min" not in cfg:
        print(f"'face_conf_min' not found in {config_path}, using "
              f"{DEFAULT_FACE_CONF_MIN}")
        return DEFAULT_FACE_CONF_MIN
    return float(cfg["face_conf_min"])


def create_detector(running_mode, face_conf_min=DEFAULT_FACE_CONF_MIN,
                    max_faces=3, tracking_confidence=0.5):
    """Create a Face Landmarker.

    face_conf_min is applied as the detection and presence confidence, which is
    how this module implements "face found with confidence >= face_conf_min".
    max_faces > 1 lets us pick the driver face ourselves (spec D-13).
    """
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=running_mode,
        num_faces=max_faces,
        min_face_detection_confidence=face_conf_min,
        min_face_presence_confidence=face_conf_min,
        min_tracking_confidence=tracking_confidence,
    )
    return vision.FaceLandmarker.create_from_options(options)


def to_mp_image(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def frame_timestamp_ms(frame_id, fps):
    """Timestamp for an offline video file.

    The spec says SG-6 assigns capture timestamps on a monotonic clock, and
    replay uses recorded timestamps. A plain video file has none, so the
    timestamp is derived from frame_id and the container FPS.
    """
    return int(round(frame_id / fps * 1000))


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def lm_to_pixel(lm, w, h):
    return [int(lm.x * w), int(lm.y * h)]


def get_points(landmarks, indices, w, h):
    return [lm_to_pixel(landmarks[i], w, h) for i in indices]


def box_dict(x1, y1, x2, y2):
    """Spec box format: {x, y, width, height}, (x, y) = top-left corner."""
    return {"x": int(x1), "y": int(y1),
            "width": int(x2 - x1), "height": int(y2 - y1)}


def make_roi_box(points, w, h, margin):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = max(0, min(xs) - margin)
    y1 = max(0, min(ys) - margin)
    x2 = min(w, max(xs) + margin)
    y2 = min(h, max(ys) + margin)
    return box_dict(x1, y1, x2, y2)


def get_face_box(landmarks, w, h):
    pts = [lm_to_pixel(lm, w, h) for lm in landmarks]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return box_dict(max(0, min(xs)), max(0, min(ys)),
                    min(w, max(xs)), min(h, max(ys)))


def crop_box(frame, box):
    """Native-resolution crop of a box (spec D-16). Returns a uint8 BGR array."""
    x, y, bw, bh = box["x"], box["y"], box["width"], box["height"]
    return frame[y:y + bh, x:x + bw].copy()


def points_inside_frame(points, w, h):
    return all(0 <= x < w and 0 <= y < h for x, y in points)


def box_is_plausible(box, min_size):
    return box["width"] >= min_size and box["height"] >= min_size


# ----------------------------------------------------------------------------
# Driver face selection (spec D-13)
# ----------------------------------------------------------------------------
def select_driver_face(result, w, h, region):
    """Pick the largest face whose box centre lies inside the driver region.

    region = (x1, y1, x2, y2) as fractions of the frame. (0, 0, 1, 1) means the
    whole frame. The real driver region is not defined in the spec yet.
    Returns the index into result.face_landmarks, or None.
    """
    rx1, ry1, rx2, ry2 = region[0] * w, region[1] * h, region[2] * w, region[3] * h
    best_idx, best_area = None, -1
    for i, landmarks in enumerate(result.face_landmarks):
        fb = get_face_box(landmarks, w, h)
        cx = fb["x"] + fb["width"] / 2
        cy = fb["y"] + fb["height"] / 2
        if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
            continue
        area = fb["width"] * fb["height"]
        if area > best_area:
            best_idx, best_area = i, area
    return best_idx


# ----------------------------------------------------------------------------
# Head pose (spec D-01: SG-1 computes it from landmarks with solvePnP)
# ----------------------------------------------------------------------------
def _wrap_angle(angle):
    """Remove the common +/-180 degree Euler ambiguity."""
    if angle > 90:
        return 180 - angle
    if angle < -90:
        return -180 - angle
    return angle


def estimate_head_pose(landmarks, w, h):
    """Approximate head pose (degrees) from six landmarks via solvePnP.

    The sign convention still has to be documented and agreed with SG-3
    (spec D-01). The convention below is whatever this generic 3D face model
    plus RQDecomp3x3 produces. It has not been verified against labelled
    images.
    """
    image_points = np.array(
        [
            lm_to_pixel(landmarks[1], w, h),    # nose tip
            lm_to_pixel(landmarks[152], w, h),  # chin
            lm_to_pixel(landmarks[33], w, h),   # driver's right eye outer corner
            lm_to_pixel(landmarks[263], w, h),  # driver's left eye outer corner
            lm_to_pixel(landmarks[61], w, h),   # mouth corner
            lm_to_pixel(landmarks[291], w, h),  # mouth corner
        ],
        dtype=np.float64,
    )
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ],
        dtype=np.float64,
    )
    focal_length = float(w)
    camera_matrix = np.array(
        [[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    return {
        "yaw_deg": round(_wrap_angle(float(angles[1])), 2),
        "pitch_deg": round(_wrap_angle(float(angles[0])), 2),
        "roll_deg": round(_wrap_angle(float(angles[2])), 2),
    }


# ----------------------------------------------------------------------------
# SG-1 face block (spec section 4)
# ----------------------------------------------------------------------------
def _null_face_block(face_detected=False, face_box=None):
    block = {k: None for k in FACE_BLOCK_KEYS}
    block["face_detected"] = face_detected
    block["face_valid"] = False
    block["face_box"] = face_box
    return block


def build_face_block(result, frame, region=(0.0, 0.0, 1.0, 1.0),
                     include_landmarks=False):
    """Return (face_block, rois). rois holds in-memory ROI images or None."""
    h, w = frame.shape[:2]
    no_rois = {"left_eye_roi": None, "right_eye_roi": None, "mouth_roi": None}

    # Rule: face_detected false -> every other field null, face_valid false.
    if not result.face_landmarks:
        block = _null_face_block()
        if include_landmarks:
            block["landmarks"] = None
        return block, no_rois

    idx = select_driver_face(result, w, h, region)
    if idx is None:  # faces exist, but none inside the driver region
        block = _null_face_block()
        if include_landmarks:
            block["landmarks"] = None
        return block, no_rois

    landmarks = result.face_landmarks[idx]
    face_box = get_face_box(landmarks, w, h)

    left_eye_pts = get_points(landmarks, LEFT_EYE_IDX, w, h)
    right_eye_pts = get_points(landmarks, RIGHT_EYE_IDX, w, h)
    mouth_pts = get_points(landmarks, MOUTH_IDX, w, h)

    left_eye_roi_box = make_roi_box(left_eye_pts, w, h, EYE_ROI_MARGIN)
    right_eye_roi_box = make_roi_box(right_eye_pts, w, h, EYE_ROI_MARGIN)
    mouth_roi_box = make_roi_box(mouth_pts, w, h, MOUTH_ROI_MARGIN)
    head_pose = estimate_head_pose(landmarks, w, h)

    # face_valid = face_detected AND landmarks and all three ROIs usable
    # (inside the frame, plausible size).
    face_valid = (
        box_is_plausible(face_box, MIN_FACE_SIZE_PX)
        and points_inside_frame(left_eye_pts, w, h)
        and points_inside_frame(right_eye_pts, w, h)
        and points_inside_frame(mouth_pts, w, h)
        and box_is_plausible(left_eye_roi_box, MIN_ROI_SIZE_PX)
        and box_is_plausible(right_eye_roi_box, MIN_ROI_SIZE_PX)
        and box_is_plausible(mouth_roi_box, MIN_ROI_SIZE_PX)
        and head_pose is not None
    )

    # Rule: face_valid false -> points, ROI boxes and head_pose are null.
    if not face_valid:
        block = _null_face_block(face_detected=True, face_box=face_box)
        if include_landmarks:
            block["landmarks"] = None
        return block, no_rois

    block = {
        "face_detected": True,
        "face_valid": True,
        # The Face Landmarker Tasks API does not return a detector score.
        "face_confidence": None,
        "face_box": face_box,
        "left_eye_pts": left_eye_pts,
        "right_eye_pts": right_eye_pts,
        "mouth_pts": mouth_pts,
        "left_eye_roi_box": left_eye_roi_box,
        "right_eye_roi_box": right_eye_roi_box,
        "mouth_roi_box": mouth_roi_box,
        "head_pose": head_pose,
    }
    if include_landmarks:
        block["landmarks"] = [lm_to_pixel(lm, w, h) for lm in landmarks]

    rois = {
        "left_eye_roi": crop_box(frame, left_eye_roi_box),
        "right_eye_roi": crop_box(frame, right_eye_roi_box),
        "mouth_roi": crop_box(frame, mouth_roi_box),
    }
    return block, rois


def build_sg1_packet(result, frame, frame_id, timestamp_ms,
                     region=(0.0, 0.0, 1.0, 1.0), include_landmarks=False):
    """Build the SG-1 output for one frame.

    Returns (record, rois):
      record = serialized JSON-safe dict (what gets logged)
      rois   = in-memory ROI images for SG-2 and SG-3 (not part of the JSON)
    """
    face_block, rois = build_face_block(result, frame, region, include_landmarks)
    record = {
        "interface_version": INTERFACE_VERSION,
        "frame_id": int(frame_id),
        "timestamp_ms": int(timestamp_ms),
        "face": face_block,
    }
    return record, rois


def check_face_block(face):
    """Check the SG-1 rules from spec section 4. Returns a list of problems."""
    problems = []
    for k in FACE_BLOCK_KEYS:
        if k not in face:
            problems.append(f"missing key: {k}")
    if problems:
        return problems

    for k in ("face_detected", "face_valid"):
        if not isinstance(face[k], bool):
            problems.append(f"{k} is not a bool")

    if face["face_valid"] and not face["face_detected"]:
        problems.append("face_valid true while face_detected false")

    if not face["face_detected"]:
        for k in FACE_BLOCK_KEYS:
            if k not in ("face_detected", "face_valid") and face.get(k) is not None:
                problems.append(f"{k} should be null when face_detected is false")
        if face.get("landmarks") is not None:
            problems.append("landmarks should be null when face_detected is false")
    elif not face["face_valid"]:
        for k in NULLABLE_WHEN_NOT_VALID:
            if face[k] is not None:
                problems.append(f"{k} should be null when face_valid is false")
        if face.get("landmarks") is not None:
            problems.append("landmarks should be null when face_valid is false")
    else:
        if face["face_box"] is None:
            problems.append("face_box is null while face_valid is true")
        if face["left_eye_pts"] is None or len(face["left_eye_pts"]) != 6:
            problems.append("left_eye_pts must have 6 points when face_valid")
        if face["right_eye_pts"] is None or len(face["right_eye_pts"]) != 6:
            problems.append("right_eye_pts must have 6 points when face_valid")
        for k in ("mouth_pts", "left_eye_roi_box", "right_eye_roi_box",
                  "mouth_roi_box", "head_pose"):
            if face[k] is None:
                problems.append(f"{k} is null while face_valid is true")
    return problems


# ----------------------------------------------------------------------------
# Drawing (debug overlay, not part of the interface)
# ----------------------------------------------------------------------------
def draw_sg1_output(frame, record):
    face = record["face"]
    vis = frame.copy()

    fb = face["face_box"]
    if fb is not None:
        x1, y1 = fb["x"], fb["y"]
        x2, y2 = x1 + fb["width"], y1 + fb["height"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(vis, "FACE", (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def draw_roi(box, label, color):
        if box is not None:
            x1, y1 = box["x"], box["y"]
            x2, y2 = x1 + box["width"], y1 + box["height"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    draw_roi(face["left_eye_roi_box"], "LEFT EYE", (255, 0, 0))
    draw_roi(face["right_eye_roi_box"], "RIGHT EYE", (0, 0, 255))
    draw_roi(face["mouth_roi_box"], "MOUTH", (0, 255, 255))

    def draw_points(points, prefix, color):
        if points is not None:
            for i, (x, y) in enumerate(points, start=1):
                cv2.circle(vis, (x, y), 4, color, -1)
                cv2.putText(vis, f"{prefix}{i}", (x + 4, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    draw_points(face["left_eye_pts"], "L", (255, 255, 0))
    draw_points(face["right_eye_pts"], "R", (255, 0, 255))
    draw_points(face["mouth_pts"], "M", (0, 255, 255))

    hp = face["head_pose"]
    if hp is not None:
        cv2.putText(vis, f"Yaw: {hp['yaw_deg']} deg", (30, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Pitch: {hp['pitch_deg']} deg", (30, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Roll: {hp['roll_deg']} deg", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(vis, f"frame_id: {record['frame_id']}  t: {record['timestamp_ms']} ms",
                (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(vis, f"face_detected: {face['face_detected']}", (30, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2)
    cv2.putText(vis, f"face_valid: {face['face_valid']}", (30, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2)
    return vis


# ----------------------------------------------------------------------------
# Pipeline steps
# ----------------------------------------------------------------------------
def read_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    return frame if success else None


def check_first_frame(video_path, out_dir, face_conf_min, max_faces, region,
                      include_landmarks):
    """Single-frame check in IMAGE mode. Saves the record and an overlay."""
    frame = read_first_frame(video_path)
    if frame is None:
        raise RuntimeError(f"Could not read a frame from: {video_path}")

    detector = create_detector(vision.RunningMode.IMAGE, face_conf_min, max_faces)
    try:
        result = detector.detect(to_mp_image(frame))
    finally:
        detector.close()

    print("Faces found in first frame:", len(result.face_landmarks))
    record, rois = build_sg1_packet(result, frame, 0, 0, region, include_landmarks)

    print("\n--- SG-1 record for frame 0 ---")
    print(json.dumps(record, indent=2))
    for name, img in rois.items():
        print(f"{name}: {None if img is None else img.shape}")

    problems = check_face_block(record["face"])
    print("Interface rule check:", "OK" if not problems else problems)

    rec_path = os.path.join(out_dir, "sg1_frame0_record.json")
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Saved: {rec_path}")

    img_path = os.path.join(out_dir, "sg1_annotated_frame.jpg")
    cv2.imwrite(img_path, draw_sg1_output(frame, record))
    print(f"Saved: {img_path}")


def process_video(video_path, out_dir, face_conf_min, max_faces, region,
                  include_landmarks, write_video):
    """One pass over the video: write JSONL records and the annotated video."""
    detector = create_detector(vision.RunningMode.VIDEO, face_conf_min, max_faces)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    jsonl_path = os.path.join(out_dir, "sg1_records.jsonl")
    video_out = os.path.join(out_dir, "sg1_annotated_video.mp4")
    writer = None
    if write_video:
        writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))

    frame_id = 0
    violations = 0
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        while True:
            success, frame = cap.read()
            if not success:
                break

            ts = frame_timestamp_ms(frame_id, fps)
            result = detector.detect_for_video(to_mp_image(frame), ts)
            record, _rois = build_sg1_packet(
                result, frame, frame_id, ts, region, include_landmarks
            )

            problems = check_face_block(record["face"])
            if problems:
                violations += 1
                if violations <= 5:
                    print(f"Frame {frame_id} rule problems: {problems}")

            jf.write(json.dumps(record, separators=(",", ":")) + "\n")
            if writer is not None:
                writer.write(draw_sg1_output(frame, record))
            frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()
    detector.close()

    print(f"\nProcessed {frame_id} frames")
    print(f"Saved records: {jsonl_path}")
    if write_video:
        print(f"Saved annotated video: {video_out}")
    print(f"Frames with interface rule problems: {violations}")
    return video_out if write_video else None


def run_configuration(video_path, config_name, face_conf_min, max_faces, region):
    detector = create_detector(vision.RunningMode.VIDEO, face_conf_min, max_faces)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    total = detected = valid = 0
    times, failures = [], []

    print(f"\nRunning: {config_name} (face_conf_min = {face_conf_min})")
    while True:
        success, frame = cap.read()
        if not success:
            break
        ts = frame_timestamp_ms(total, fps)
        mp_image = to_mp_image(frame)

        start = time.perf_counter()
        result = detector.detect_for_video(mp_image, ts)
        times.append((time.perf_counter() - start) * 1000)

        record, _ = build_sg1_packet(result, frame, total, ts, region)
        if record["face"]["face_detected"]:
            detected += 1
        if record["face"]["face_valid"]:
            valid += 1
        else:
            failures.append(total)
        total += 1

    cap.release()
    detector.close()

    avg_latency = float(np.mean(times)) if times else 0
    return {
        "configuration": config_name,
        "face_conf_min": face_conf_min,
        "total_frames": total,
        "face_detected_frames": detected,
        "face_valid_frames": valid,
        "failed_frames": total - valid,
        "detection_rate_percent": round(detected / total * 100, 2) if total else 0,
        "reliability_percent": round(valid / total * 100, 2) if total else 0,
        "avg_latency_ms": round(avg_latency, 2),
        "processing_fps": round(1000 / avg_latency, 2) if avg_latency > 0 else 0,
        "failure_frame_ids": failures[:20],
    }


def compare_configurations(video_path, out_csv, max_faces, region):
    df = pd.DataFrame([
        run_configuration(video_path, "Config A", 0.50, max_faces, region),
        run_configuration(video_path, "Config B", 0.70, max_faces, region),
    ])
    cols = ["configuration", "face_conf_min", "total_frames", "face_valid_frames",
            "failed_frames", "reliability_percent", "avg_latency_ms",
            "processing_fps"]
    print("\n=== Configuration comparison ===")
    print(df[cols].to_string(index=False))
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    return df


def reencode_h264(in_path, out_path):
    """Re-encode to H.264 so the video plays in browsers and most players."""
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH, skipping H.264 re-encode.")
        return False
    subprocess.run(["ffmpeg", "-y", "-i", in_path, "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p", out_path], check=True)
    print(f"Saved H.264 video as: {out_path}")
    return True


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_region(text):
    try:
        vals = tuple(float(v) for v in text.split(","))
        assert len(vals) == 4 and 0 <= vals[0] < vals[2] <= 1 and 0 <= vals[1] < vals[3] <= 1
        return vals
    except Exception:
        raise argparse.ArgumentTypeError(
            "driver region must be 'x1,y1,x2,y2' as fractions of the frame, "
            "e.g. 0.0,0.0,0.6,1.0")


def parse_args():
    p = argparse.ArgumentParser(
        description="SG-1 face/landmark module, output per Interface Spec V1")
    p.add_argument("--video", required=True, help="Path to the input video file")
    p.add_argument("--out-dir", default="outputs", help="Folder for output files")
    p.add_argument("--config", default=None,
                   help="Path to config/shared.yaml (reads face_conf_min)")
    p.add_argument("--face-conf-min", type=float, default=None,
                   help="Override face_conf_min (default: from --config, else 0.5)")
    p.add_argument("--max-faces", type=int, default=3,
                   help="Faces the model may return; the driver is chosen from them")
    p.add_argument("--driver-region", type=parse_region, default=(0.0, 0.0, 1.0, 1.0),
                   help="Driver region as 'x1,y1,x2,y2' fractions (default: whole frame)")
    p.add_argument("--include-landmarks", action="store_true",
                   help="Also log the full 478-point landmark list (large)")
    p.add_argument("--skip-compare", action="store_true",
                   help="Skip the Config A vs Config B comparison")
    p.add_argument("--skip-video", action="store_true",
                   help="Skip writing the annotated video")
    p.add_argument("--no-h264", action="store_true",
                   help="Skip the ffmpeg H.264 re-encode")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.video):
        sys.exit(f"Video not found: {args.video}")

    face_conf_min = (args.face_conf_min if args.face_conf_min is not None
                     else load_face_conf_min(args.config))

    os.makedirs(args.out_dir, exist_ok=True)
    print("MediaPipe:", getattr(mp, "__version__", "unknown"))
    print("OpenCV:", cv2.__version__)
    print("Using video:", args.video)
    print("face_conf_min:", face_conf_min)

    ensure_model()

    check_first_frame(args.video, args.out_dir, face_conf_min, args.max_faces,
                      args.driver_region, args.include_landmarks)

    if not args.skip_compare:
        compare_configurations(args.video,
                               os.path.join(args.out_dir, "config_comparison.csv"),
                               args.max_faces, args.driver_region)

    raw_video = process_video(args.video, args.out_dir, face_conf_min,
                              args.max_faces, args.driver_region,
                              args.include_landmarks, not args.skip_video)

    if raw_video and not args.no_h264:
        reencode_h264(raw_video,
                      os.path.join(args.out_dir, "sg1_annotated_video_h264.mp4"))

    print("\nDone.")


if __name__ == "__main__":
    main()