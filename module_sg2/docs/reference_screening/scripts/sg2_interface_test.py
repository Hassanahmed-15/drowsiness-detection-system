"""SG2 interface feasibility test on talking.avi.

1. Simulate SG1 output with MediaPipe Face Landmarker: 6 ordered eye landmarks per eye
   (p1 outer corner, p2/p3 upper lid, p4 inner corner, p5/p6 lower lid) in full-frame pixels,
   plus a square eye ROI box [x1,y1,x2,y2].
2. Candidate A (e-candeloro/Driver-State-Detection): call the repo's own
   EyeDetector._calc_EAR_eye on the SG1 points (re-ordered to the repo's order).
3. Candidate B (OMZ open-closed-eye-0001 ONNX): crop the eye ROI, resize to 32x32 BGR,
   normalise (x-127)/255, run onnxruntime, read [open, closed] softmax.
4. Turn per-frame closed flags into blink events and compare with talking.tag ground truth.

Usage: python sg2_interface_test.py <dsd_repo> <onnx> <video> <tag> <out_dir>
"""
import sys
import time
import json
import numpy as np

dsd_repo, onnx_path, video, tag_path, out_dir = sys.argv[1:6]
sys.path.insert(0, dsd_repo)

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import onnxruntime as ort  # noqa: E402
from driver_state_detection.eye_detector import EyeDetector  # noqa: E402

# MediaPipe indices in SG1 order p1..p6 (p1 outer, p2/p3 upper, p4 inner, p5/p6 lower)
SG1_IDX = {"left": [33, 160, 158, 133, 153, 144], "right": [263, 387, 385, 362, 380, 373]}
ROI_SCALE = float(sys.argv[6]) if len(sys.argv) > 6 else 1.8  # OMZ demo default scale = 1.8
# OMZ demo eye_state_estimator.cpp: eyeState(open) = out[0] < out[1]  -> index 0 = closed, index 1 = open
CLOSED_IDX = 0


def sg1_to_repo_order(p):
    # repo expects [corner, corner, top1, bottom1, top2, bottom2] = [p1, p4, p2, p6, p3, p5]
    return p[[0, 3, 1, 5, 2, 4]]


def roi_box(p, w, h):
    c = (p[0] + p[3]) / 2.0
    side = ROI_SCALE * np.linalg.norm(p[0] - p[3])
    x1, y1 = int(max(0, c[0] - side / 2)), int(max(0, c[1] - side / 2))
    x2, y2 = int(min(w, c[0] + side / 2)), int(min(h, c[1] + side / 2))
    return [x1, y1, x2, y2]


def read_gt(tag_path):
    blinks = {}
    started = False
    for line in open(tag_path):
        line = line.strip()
        if line == "#start":
            started = True
            continue
        if not started or not line or line.startswith("#"):
            continue
        parts = line.split(":")
        fr, bid = int(parts[0]), int(parts[1])
        if bid != -1:
            blinks.setdefault(bid, []).append(fr)
    return [(min(v), max(v)) for v in blinks.values()]


def events(flags, min_len=1):
    ev, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_len:
                ev.append((start, i - 1))
            start = None
    return ev


def match(gt, pred, tol=2):
    def ov(a, b):
        return a[0] - tol <= b[1] and b[0] - tol <= a[1]
    tp = sum(any(ov(g, p) for p in pred) for g in gt)          # GT blinks that were found
    fp = sum(not any(ov(p, g) for g in gt) for p in pred)      # predicted events with no GT blink
    fn = len(gt) - tp
    prec = (len(pred) - fp) / len(pred) if pred else 0.0
    rec = tp / len(gt) if gt else 0.0
    return dict(gt=len(gt), pred=len(pred), gt_found=tp, false_events=fp, missed=fn,
                precision=round(prec, 3), recall=round(rec, 3))


opts = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=f"{dsd_repo}/models/face_landmarker.task"),
    running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1)
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
out_name = sess.get_outputs()[0].name
print("ONNX input:", inp.name, inp.shape, "output:", out_name, sess.get_outputs()[0].shape)

cap = cv2.VideoCapture(video)
rows, t_ear, t_cnn, t_lm = [], [], [], []
fi = 0
with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as det:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        t = time.perf_counter()
        res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), fi * 33 + 1)
        t_lm.append(time.perf_counter() - t)
        row = {"frame": fi}
        if res.face_landmarks:
            lm = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
            for eye, idx in SG1_IDX.items():
                p = lm[idx]                      # SG1-style 6 points, full-frame pixels
                box = roi_box(p, w, h)           # SG1-style eye ROI [x1,y1,x2,y2]
                t = time.perf_counter()
                row[f"ear_{eye}"] = EyeDetector._calc_EAR_eye(sg1_to_repo_order(p))
                t_ear.append(time.perf_counter() - t)
                crop = frame[box[1]:box[3], box[0]:box[2]]
                if crop.size:
                    x = cv2.resize(crop, (32, 32)).astype(np.float32)
                    x = ((x - 127.0) / 255.0).transpose(2, 0, 1)[None]
                    t = time.perf_counter()
                    prob = sess.run([out_name], {inp.name: x})[0].reshape(-1)
                    t_cnn.append(time.perf_counter() - t)
                    row[f"pclosed_{eye}"] = float(prob[CLOSED_IDX])
        rows.append(row)
        fi += 1

gt = read_gt(tag_path)
n = len(rows)


def get(r, k):
    v = r.get(k)
    return np.nan if v is None else v


ear = np.array([np.nanmean([get(r, "ear_left"), get(r, "ear_right")]) if "ear_left" in r else np.nan for r in rows])
pcl = np.array([np.nanmean([get(r, "pclosed_left"), get(r, "pclosed_right")]) if "pclosed_left" in r else np.nan for r in rows])

summary = {"roi_scale": ROI_SCALE, "frames": n, "frames_with_face": int(np.sum(~np.isnan(ear))), "gt_blinks": len(gt),
           "median_ms": {"mediapipe_landmarker_per_frame": round(1000 * float(np.median(t_lm)), 3),
                         "ear_per_eye": round(1000 * float(np.median(t_ear)), 4),
                         "open_closed_eye_onnx_per_eye": round(1000 * float(np.median(t_cnn)), 4)}}
for th in (0.15, 0.20):
    summary[f"EAR<{th}"] = match(gt, events(np.nan_to_num(ear, nan=1.0) < th))
summary["onnx_pclosed>0.5"] = match(gt, events(np.nan_to_num(pcl, nan=0.0) > 0.5))
summary["onnx_frames_closed_fraction"] = round(float(np.nanmean(pcl > 0.5)), 4)

# Frame-level agreement with GT "closed/blink" frames (any frame inside a GT blink interval)
gt_flag = np.zeros(n, bool)
for a, b in gt:
    gt_flag[a:b + 1] = True
for name, pred in (("EAR<0.20", np.nan_to_num(ear, nan=1.0) < 0.20),
                   ("onnx_pclosed>0.5", np.nan_to_num(pcl, nan=0.0) > 0.5)):
    summary[f"frame_acc_{name}"] = round(float(np.mean(pred == gt_flag)), 4)

with open(f"{out_dir}/sg2_interface_frames.csv", "w") as f:
    f.write("frame,ear_left,ear_right,pclosed_left,pclosed_right,gt_blink\n")
    for r in rows:
        f.write(",".join(str(r.get(k, "")) for k in ("frame", "ear_left", "ear_right", "pclosed_left", "pclosed_right"))
                + f",{int(gt_flag[r['frame']])}\n")
print(json.dumps(summary, indent=1))
