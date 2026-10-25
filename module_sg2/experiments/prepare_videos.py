"""Run the SG1-compatible landmark source once per test video and cache
everything the SG2 methods need (eye points, eye crops, head pose,
blendshapes, brightness) plus the ground-truth labels.

Run from module_sg2/:   python experiments/prepare_videos.py --workers 4
Output: experiments/cache/<video>.npz and <video>_lowlight.npz (TEST only)
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from common import CACHE_DIR, DATA_ROOT, MODULE_DIR, TEST, VIDEOS, lowlight, parse_tag  # noqa: E402
from sg1_landmarks import SG1LandmarkSource  # noqa: E402
from sg2_eye.geometry import eye_crop  # noqa: E402

CROP_SIZE, CROP_SCALE = 32, 1.6


def prepare(job):
    name, dark = job
    out = CACHE_DIR / f"{name}{'_lowlight' if dark else ''}.npz"
    if out.exists():
        return f"{out.name} (cached)"
    stem = DATA_ROOT / VIDEOS[name]
    cap = cv2.VideoCapture(str(stem) + ".avi")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source = SG1LandmarkSource(MODULE_DIR / "models" / "face_landmarker.task", with_blendshapes=True)
    rng = np.random.default_rng(477)
    nan = np.full((6, 2), np.nan, np.float32)
    rows = {k: [] for k in ("valid", "left_pts", "right_pts", "pose", "blink_bs", "brightness",
                            "left_crop", "right_crop")}
    t0, i = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if dark:
            frame = lowlight(frame, rng)
        b = source.process(frame, i, i * 1000.0 / fps)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        valid = b["face_valid"]
        rows["valid"].append(valid)
        rows["left_pts"].append(np.array(b["left_eye_pts"], np.float32) if valid else nan)
        rows["right_pts"].append(np.array(b["right_eye_pts"], np.float32) if valid else nan)
        hp = b["head_pose"] if valid and b["head_pose"] else None
        rows["pose"].append([hp["yaw_deg"], hp["pitch_deg"], hp["roll_deg"]] if hp else [np.nan] * 3)
        bs = b.get("blendshapes") or {}
        rows["blink_bs"].append([bs.get("eyeBlinkLeft", np.nan), bs.get("eyeBlinkRight", np.nan)])
        if valid:
            x1, y1, x2, y2 = (int(max(v, 0)) for v in b["face_box"])
            rows["brightness"].append(float(gray[y1:y2, x1:x2].mean()) if x2 > x1 and y2 > y1 else np.nan)
            rows["left_crop"].append(eye_crop(gray, b["left_eye_pts"], CROP_SIZE, CROP_SCALE))
            rows["right_crop"].append(eye_crop(gray, b["right_eye_pts"], CROP_SIZE, CROP_SCALE))
        else:
            rows["brightness"].append(np.nan)
            rows["left_crop"].append(np.zeros((CROP_SIZE, CROP_SIZE), np.uint8))
            rows["right_crop"].append(np.zeros((CROP_SIZE, CROP_SIZE), np.uint8))
        i += 1
    source.close()
    label, blink_id, meta = parse_tag(str(stem) + ".tag", i)
    np.savez_compressed(
        out, fps=fps, timestamp_ms=np.arange(i) * 1000.0 / fps,
        valid=np.array(rows["valid"]), left_pts=np.stack(rows["left_pts"]),
        right_pts=np.stack(rows["right_pts"]), pose=np.array(rows["pose"], np.float32),
        blink_bs=np.array(rows["blink_bs"], np.float32), brightness=np.array(rows["brightness"], np.float32),
        left_crop=np.stack(rows["left_crop"]), right_crop=np.stack(rows["right_crop"]),
        label=label, blink_id=blink_id, glasses=meta.get("glasses", "NO"), lowlight=dark)
    return f"{out.name}: {i} frames, {np.mean(rows['valid']):.3f} face-valid, {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(v, False) for v in VIDEOS] + [(v, True) for v in TEST]
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(prepare, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
