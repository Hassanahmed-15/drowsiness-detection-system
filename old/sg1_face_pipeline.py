#!/usr/bin/env python3
"""
SG1 face / eye / mouth landmark pipeline (converted from week5_CV_project_SG_1.ipynb).

What it does:
  1. Downloads the MediaPipe Face Landmarker model (once).
  2. Runs a single-frame sanity check on the first frame of the video.
  3. Builds the SG1 interface block (face box, 6-point eye landmarks, mouth
     points, ROI boxes, head pose) for every frame.
  4. Compares two detection-confidence configurations (A = 0.50, B = 0.70).
  5. Saves an annotated first frame and an annotated video.
  6. Optionally re-encodes the video to H.264 with ffmpeg (if installed).

Usage:
  python sg1_face_pipeline.py --video "path/to/Video Project.mp4"
"""

import argparse
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

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = "face_landmarker.task"

# MediaPipe Face Mesh landmark indices for the 6-point EAR ordering:
# p1 = outer corner, p2/p3 = upper lid, p4 = inner corner, p5/p6 = lower lid.
# "Left" is the driver's own left, as required by the SG1 interface.
LEFT_EYE_IDX = [263, 387, 385, 362, 380, 373]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Simple mouth landmarks
MOUTH_IDX = [61, 13, 291, 14]


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


def create_detector(running_mode, confidence=0.5, tracking_confidence=0.5):
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    if running_mode == vision.RunningMode.IMAGE:
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=1,
        )
    else:
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=1,
            min_face_detection_confidence=confidence,
            min_face_presence_confidence=confidence,
            min_tracking_confidence=tracking_confidence,
        )
    return vision.FaceLandmarker.create_from_options(options)


def to_mp_image(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


# ----------------------------------------------------------------------------
# Landmark / geometry helpers
# ----------------------------------------------------------------------------
def lm_to_pixel(lm, w, h):
    return [int(lm.x * w), int(lm.y * h)]


def get_points(landmarks, indices, w, h):
    return [lm_to_pixel(landmarks[i], w, h) for i in indices]


def make_roi_box(points, w, h, margin=15):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = max(0, min(xs) - margin)
    y1 = max(0, min(ys) - margin)
    x2 = min(w, max(xs) + margin)
    y2 = min(h, max(ys) + margin)
    return [x1, y1, x2, y2]


def get_face_box(landmarks, w, h):
    pts = [lm_to_pixel(lm, w, h) for lm in landmarks]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x = max(0, min(xs))
    y = max(0, min(ys))
    x2 = min(w, max(xs))
    y2 = min(h, max(ys))
    return {"x": x, "y": y, "width": x2 - x, "height": y2 - y}


def _wrap_angle(angle):
    """Remove the common +/-180 degree Euler ambiguity."""
    if angle > 90:
        return 180 - angle
    if angle < -90:
        return -180 - angle
    return angle


def estimate_head_pose(landmarks, w, h):
    """Approximate head pose from nose, chin, eye corners and mouth corners.

    This is the corrected version from the notebook (the one with the
    +/-180 degree ambiguity fix).
    """
    image_points = np.array(
        [
            lm_to_pixel(landmarks[1], w, h),    # nose
            lm_to_pixel(landmarks[152], w, h),  # chin
            lm_to_pixel(landmarks[33], w, h),   # driver's right eye outer area
            lm_to_pixel(landmarks[263], w, h),  # driver's left eye outer area
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
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    pitch = _wrap_angle(float(angles[0]))
    yaw = _wrap_angle(float(angles[1]))
    roll = _wrap_angle(float(angles[2]))

    return {
        "yaw_deg": round(yaw, 2),
        "pitch_deg": round(pitch, 2),
        "roll_deg": round(roll, 2),
    }


# ----------------------------------------------------------------------------
# SG1 interface block
# ----------------------------------------------------------------------------
def _empty_block(frame_id, timestamp_ms, face_detected=False,
                 face_confidence=None, face_box=None):
    return {
        "frame_id": frame_id,
        "timestamp_ms": timestamp_ms,
        "face_detected": face_detected,
        "face_valid": False,
        "face_confidence": face_confidence,
        "face_box": face_box,
        "left_eye_pts": None,
        "right_eye_pts": None,
        "mouth_pts": None,
        "landmarks": None,
        "left_eye_roi_box": None,
        "right_eye_roi_box": None,
        "mouth_roi_box": None,
        "head_pose": None,
    }


def build_sg1_face_block(result, frame, frame_id=0, timestamp_ms=0):
    h, w = frame.shape[:2]

    if not result.face_landmarks:
        return _empty_block(frame_id, timestamp_ms)

    landmarks = result.face_landmarks[0]

    left_eye_pts = get_points(landmarks, LEFT_EYE_IDX, w, h)
    right_eye_pts = get_points(landmarks, RIGHT_EYE_IDX, w, h)
    mouth_pts = get_points(landmarks, MOUTH_IDX, w, h)

    left_eye_roi_box = make_roi_box(left_eye_pts, w, h, margin=15)
    right_eye_roi_box = make_roi_box(right_eye_pts, w, h, margin=15)
    mouth_roi_box = make_roi_box(mouth_pts, w, h, margin=20)

    face_box = get_face_box(landmarks, w, h)
    head_pose = estimate_head_pose(landmarks, w, h)

    # The Face Landmarker Tasks API does not expose a detector confidence
    # score, so this stays None for now.
    face_confidence = None

    face_valid = (
        left_eye_roi_box is not None
        and right_eye_roi_box is not None
        and mouth_roi_box is not None
        and head_pose is not None
    )

    if not face_valid:
        return _empty_block(
            frame_id, timestamp_ms,
            face_detected=True,
            face_confidence=face_confidence,
            face_box=face_box,
        )

    all_landmarks_px = [lm_to_pixel(lm, w, h) for lm in landmarks]

    return {
        "frame_id": frame_id,
        "timestamp_ms": timestamp_ms,
        "face_detected": True,
        "face_valid": True,
        "face_confidence": face_confidence,
        "face_box": face_box,
        "left_eye_pts": left_eye_pts,
        "right_eye_pts": right_eye_pts,
        "mouth_pts": mouth_pts,
        "landmarks": all_landmarks_px,
        "left_eye_roi_box": left_eye_roi_box,
        "right_eye_roi_box": right_eye_roi_box,
        "mouth_roi_box": mouth_roi_box,
        "head_pose": head_pose,
    }


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def draw_sg1_output(frame, face_block):
    vis = frame.copy()

    # Face box
    if face_block["face_box"] is not None:
        fb = face_block["face_box"]
        x1, y1 = fb["x"], fb["y"]
        x2, y2 = x1 + fb["width"], y1 + fb["height"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(vis, "FACE", (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def draw_roi(box, label, color):
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    draw_roi(face_block["left_eye_roi_box"], "LEFT EYE", (255, 0, 0))
    draw_roi(face_block["right_eye_roi_box"], "RIGHT EYE", (0, 0, 255))
    draw_roi(face_block["mouth_roi_box"], "MOUTH", (0, 255, 255))

    def draw_points(points, prefix, color):
        if points is not None:
            for i, (x, y) in enumerate(points, start=1):
                cv2.circle(vis, (x, y), 4, color, -1)
                cv2.putText(vis, f"{prefix}{i}", (x + 4, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    draw_points(face_block["left_eye_pts"], "L", (255, 255, 0))
    draw_points(face_block["right_eye_pts"], "R", (255, 0, 255))
    draw_points(face_block["mouth_pts"], "M", (0, 255, 255))

    # Head pose text
    hp = face_block["head_pose"]
    if hp is not None:
        cv2.putText(vis, f"Yaw: {hp['yaw_deg']} deg", (30, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Pitch: {hp['pitch_deg']} deg", (30, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Roll: {hp['roll_deg']} deg", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(vis, f"face_detected: {face_block['face_detected']}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2)
    cv2.putText(vis, f"face_valid: {face_block['face_valid']}", (30, 160),
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


def check_first_frame(video_path):
    """Single-frame sanity check using IMAGE mode. Returns (frame, face_block)."""
    frame = read_first_frame(video_path)
    if frame is None:
        raise RuntimeError(f"Could not read a frame from: {video_path}")

    detector = create_detector(vision.RunningMode.IMAGE)
    try:
        result = detector.detect(to_mp_image(frame))
    finally:
        detector.close()

    if result.face_landmarks:
        print("Face detected")
        print("Number of faces:", len(result.face_landmarks))
        print("Landmarks on face:", len(result.face_landmarks[0]))
    else:
        print("No face detected in the first frame")

    face_block = build_sg1_face_block(result, frame, frame_id=0, timestamp_ms=0)

    print("\n--- SG1 block for frame 0 ---")
    for key in ("face_detected", "face_valid", "face_box", "left_eye_pts",
                "right_eye_pts", "mouth_pts", "left_eye_roi_box",
                "right_eye_roi_box", "mouth_roi_box", "head_pose"):
        print(f"{key}: {face_block[key]}")

    return frame, face_block


def run_configuration(video_path, config_name, confidence_threshold):
    detector = create_detector(
        vision.RunningMode.VIDEO,
        confidence=confidence_threshold,
        tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    total_frames = 0
    detected_frames = 0
    valid_frames = 0
    processing_times = []
    failure_frames = []

    print(f"\nRunning: {config_name}")
    print(f"Confidence: {confidence_threshold}")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_id = total_frames
        timestamp_ms = int((frame_id / fps) * 1000)  # required by VIDEO mode
        mp_image = to_mp_image(frame)

        start = time.perf_counter()
        result = detector.detect_for_video(mp_image, timestamp_ms)
        processing_times.append((time.perf_counter() - start) * 1000)

        face_block = build_sg1_face_block(
            result, frame, frame_id=frame_id, timestamp_ms=timestamp_ms
        )

        if face_block["face_detected"]:
            detected_frames += 1
        if face_block["face_valid"]:
            valid_frames += 1
        else:
            failure_frames.append(frame_id)

        total_frames += 1

    cap.release()
    detector.close()

    detection_rate = detected_frames / total_frames * 100 if total_frames else 0
    reliability = valid_frames / total_frames * 100 if total_frames else 0
    avg_latency = float(np.mean(processing_times)) if processing_times else 0
    processing_fps = 1000 / avg_latency if avg_latency > 0 else 0

    return {
        "configuration": config_name,
        "confidence_threshold": confidence_threshold,
        "total_frames": total_frames,
        "face_detected_frames": detected_frames,
        "face_valid_frames": valid_frames,
        "failed_frames": total_frames - valid_frames,
        "detection_rate_percent": round(detection_rate, 2),
        "reliability_percent": round(reliability, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "processing_fps": round(processing_fps, 2),
        "failure_frame_ids": failure_frames[:20],
    }


def compare_configurations(video_path, out_csv):
    results = [
        run_configuration(video_path, "Config A", 0.50),
        run_configuration(video_path, "Config B", 0.70),
    ]
    df = pd.DataFrame(results)

    cols = ["configuration", "confidence_threshold", "total_frames",
            "face_valid_frames", "failed_frames", "reliability_percent",
            "avg_latency_ms", "processing_fps"]
    print("\n=== Configuration comparison ===")
    print(df[cols].to_string(index=False))

    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    return df


def save_annotated_frame(frame, face_block, out_path):
    annotated = draw_sg1_output(frame, face_block)
    cv2.imwrite(out_path, annotated)
    print(f"Saved: {out_path}")


def write_annotated_video(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    detector = create_detector(vision.RunningMode.VIDEO)

    frame_id = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        timestamp_ms = int((frame_id / fps) * 1000)
        result = detector.detect_for_video(to_mp_image(frame), timestamp_ms)
        face_block = build_sg1_face_block(
            result, frame, frame_id=frame_id, timestamp_ms=timestamp_ms
        )
        writer.write(draw_sg1_output(frame, face_block))
        frame_id += 1

    cap.release()
    writer.release()
    detector.close()
    print(f"Saved annotated video as: {out_path}")


def print_video_info(path):
    print("File exists:", os.path.exists(path))
    if not os.path.exists(path):
        return
    print("File size (MB):", round(os.path.getsize(path) / (1024 * 1024), 2))
    cap = cv2.VideoCapture(path)
    print("Frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("FPS:", cap.get(cv2.CAP_PROP_FPS))
    print("Width:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("Height:", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()


def reencode_h264(in_path, out_path):
    """Re-encode to H.264 so the video plays in browsers and most players."""
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH, skipping H.264 re-encode "
              "(the mp4v file is still usable in VLC and similar players).")
        return False
    cmd = ["ffmpeg", "-y", "-i", in_path, "-vcodec", "libx264",
           "-pix_fmt", "yuv420p", out_path]
    subprocess.run(cmd, check=True)
    print(f"Saved H.264 video as: {out_path}")
    return True


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SG1 face/eye/mouth landmark pipeline")
    p.add_argument("--video", required=True, help="Path to the input video file")
    p.add_argument("--out-dir", default="outputs", help="Folder for output files")
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

    os.makedirs(args.out_dir, exist_ok=True)
    print("MediaPipe:", getattr(mp, "__version__", "unknown"))
    print("OpenCV:", cv2.__version__)
    print("Using video:", args.video)

    ensure_model()

    # 1. First-frame check + annotated image
    frame, face_block = check_first_frame(args.video)
    save_annotated_frame(
        frame, face_block, os.path.join(args.out_dir, "sg1_annotated_frame.jpg")
    )

    # 2. Config comparison
    if not args.skip_compare:
        compare_configurations(
            args.video, os.path.join(args.out_dir, "config_comparison.csv")
        )

    # 3. Annotated video
    if not args.skip_video:
        raw_path = os.path.join(args.out_dir, "sg1_annotated_video.mp4")
        write_annotated_video(args.video, raw_path)
        print_video_info(raw_path)
        if not args.no_h264:
            reencode_h264(
                raw_path, os.path.join(args.out_dir, "sg1_annotated_video_h264.mp4")
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
