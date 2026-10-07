"""Load altaga/DBSE-monitor BlinkModel.t7 (24x24 grayscale eye CNN) and run it on SG1-style
eye crops from talking.avi. Usage: python dbse_blinkmodel_test.py <dbse_dir> <dsd_repo> <video> <tag>"""
import sys
import time
import numpy as np

dbse, dsd_repo, video, tag = sys.argv[1:5]
sys.path.insert(0, f"{dbse}/Drowsiness")
import torch  # noqa: E402
import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import model as dbse_model  # repo's model.py  # noqa: E402

ckpt_path = f"{dbse}/Drowsiness/Model/BlinkModel.t7"
try:
    ckpt = torch.load(ckpt_path, map_location="cpu")  # torch>=2.6 default weights_only=True
    print("torch.load default (weights_only=True): OK")
except Exception as e:  # report and stop: we do not unpickle arbitrary objects
    print("torch.load default FAILED:", type(e).__name__, str(e)[:300])
    raise SystemExit(1)
print("checkpoint keys:", list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt))
net = dbse_model.Model(num_classes=2)
missing = net.load_state_dict(ckpt["net"])
print("load_state_dict:", missing)
net.eval()
print("params:", sum(p.numel() for p in net.parameters()))
CLASSES = ["Close", "Open"]  # from repo notebook.py

SG1_IDX = {"left": [33, 160, 158, 133, 153, 144], "right": [263, 387, 385, 362, 380, 373]}
SCALE = 1.8


def read_gt(p):
    b, started = {}, False
    for line in open(p):
        line = line.strip()
        if line == "#start":
            started = True
            continue
        if not started or not line or line.startswith("#"):
            continue
        fr, bid = map(int, line.split(":")[:2])
        if bid != -1:
            b.setdefault(bid, []).append(fr)
    return [(min(v), max(v)) for v in b.values()]


def events(flags):
    ev, s = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and s is None:
            s = i
        elif not f and s is not None:
            ev.append((s, i - 1))
            s = None
    return ev


def match(gt, pred, tol=2):
    ov = lambda a, b: a[0] - tol <= b[1] and b[0] - tol <= a[1]  # noqa: E731
    found = sum(any(ov(g, p) for p in pred) for g in gt)
    fp = sum(not any(ov(p, g) for g in gt) for p in pred)
    return dict(gt=len(gt), pred=len(pred), gt_found=found, false_events=fp, missed=len(gt) - found,
                precision=round((len(pred) - fp) / len(pred), 3) if pred else 0.0,
                recall=round(found / len(gt), 3))


opts = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=f"{dsd_repo}/models/face_landmarker.task"),
    running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1)
cap = cv2.VideoCapture(video)
closed, times, fi = [], [], 0
with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as det, torch.no_grad():
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), fi * 33 + 1)
        votes = []
        if res.face_landmarks:
            lm = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for idx in SG1_IDX.values():
                p = lm[idx]
                c = (p[0] + p[3]) / 2
                side = SCALE * np.linalg.norm(p[0] - p[3])
                x1, y1 = int(max(0, c[0] - side / 2)), int(max(0, c[1] - side / 2))
                crop = gray[y1:int(c[1] + side / 2), x1:int(c[0] + side / 2)]
                x = torch.from_numpy(cv2.resize(crop, (24, 24)).astype(np.float32) / 255.0)[None, None]
                t = time.perf_counter()
                out = net(x)
                times.append(time.perf_counter() - t)
                votes.append(int(out.argmax(1)) == 0)  # 0 = Close
        closed.append(bool(votes) and all(votes))
        fi += 1

gt = read_gt(tag)
print("frames:", fi, "closed_fraction:", round(float(np.mean(closed)), 4),
      "median_ms_per_eye_torch_cpu:", round(1000 * float(np.median(times)), 3))
print("blink events (both eyes Close):", match(gt, events(closed)))
