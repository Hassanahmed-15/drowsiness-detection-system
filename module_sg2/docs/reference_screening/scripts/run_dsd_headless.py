"""Headless runner for e-candeloro/Driver-State-Detection main() on a video file.

Patching (done here at runtime, repo files are NOT edited):
  * cv2.VideoCapture(int) -> cv2.VideoCapture(<video path>)
  * cv2.imshow / cv2.waitKey / cv2.destroyAllWindows -> no-ops (no GUI)
  * EyeDetector.get_EAR wrapped to log per-frame EAR values
  * stops after MAX_FRAMES frames
Usage: python run_dsd_headless.py <repo_dir> <video> <out_csv> <max_frames>
"""
import sys
import time

repo, video, out_csv, max_frames = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, repo)

import cv2  # noqa: E402

_RealCap = cv2.VideoCapture
state = {"frames": 0}


class FileCap:
    def __init__(self, _index):
        self.cap = _RealCap(video)

    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        if state["frames"] >= max_frames:
            return False, None
        state["frames"] += 1
        return self.cap.read()

    def release(self):
        self.cap.release()


cv2.VideoCapture = FileCap
cv2.imshow = lambda *a, **k: None
cv2.waitKey = lambda *a, **k: -1
cv2.destroyAllWindows = lambda *a, **k: None

from driver_state_detection import eye_detector, main as dsd_main  # noqa: E402

ear_log = []
_orig = eye_detector.EyeDetector.get_EAR


def logged_get_EAR(self, landmarks, frame_size):
    ear = _orig(self, landmarks, frame_size)
    ear_log.append((state["frames"] - 1, ear))
    return ear


eye_detector.EyeDetector.get_EAR = logged_get_EAR

t0 = time.perf_counter()
dsd_main.main(["--camera", "1", "--model-path", f"{repo}/models/face_landmarker.task"])
dt = time.perf_counter() - t0

with open(out_csv, "w") as f:
    f.write("frame,ear\n")
    for fr, e in ear_log:
        f.write(f"{fr},{'' if e is None else f'{e:.4f}'}\n")

vals = [e for _, e in ear_log if e is not None]
print(f"frames_read={state['frames']} frames_with_EAR={len(vals)} wall_time_s={dt:.2f} "
      f"approx_fps_full_pipeline={state['frames']/dt:.1f}")
if vals:
    import statistics
    print(f"EAR min={min(vals):.3f} median={statistics.median(vals):.3f} max={max(vals):.3f}")
