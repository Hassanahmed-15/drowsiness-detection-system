"""MediaPipe Face Landmarker blendshapes eyeBlinkLeft/Right on talking.avi vs talking.tag.
Usage: python mp_blendshape_test.py <face_landmarker.task> <video> <tag>"""
import sys
import time
import numpy as np
import cv2
import mediapipe as mp

model, video, tag = sys.argv[1:4]
sys.path.insert(0, sys.path[0])
from dbse_blinkmodel_test_helpers import read_gt, events, match  # noqa: E402

opts = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model),
    running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True)
cap = cv2.VideoCapture(video)
scores, times, names, fi = [], [], None, 0
with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as det:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.perf_counter()
        r = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), fi * 33 + 1)
        times.append(time.perf_counter() - t)
        if r.face_blendshapes:
            d = {c.category_name: c.score for c in r.face_blendshapes[0]}
            names = names or list(d)
            scores.append((d["eyeBlinkLeft"] + d["eyeBlinkRight"]) / 2)
        else:
            scores.append(0.0)
        fi += 1
print("num_blendshapes:", len(names), "eye names:", [n for n in names if n.startswith("eyeBlink")])
s = np.array(scores)
print("frames:", fi, "median_ms_per_frame (landmarks+blendshapes):", round(1000 * float(np.median(times)), 2))
print("eyeBlink score min/median/max:", round(float(s.min()), 3), round(float(np.median(s)), 3), round(float(s.max()), 3))
gt = read_gt(tag)
for th in (0.4, 0.5):
    print(f"blink events eyeBlink>{th}:", match(gt, events(s > th)))
