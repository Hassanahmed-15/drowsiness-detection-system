"""Feed real SG1 V1 JSONL blocks (mock) into e-candeloro EyeDetector._calc_EAR_eye.
Usage: python dsd_on_sg1_mock.py <dsd_repo> <mock_sg1.jsonl>"""
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from driver_state_detection.eye_detector import EyeDetector
ORDER = [0, 3, 1, 5, 2, 4]  # SG1 p1..p6 -> repo [p1, p4, p2, p6, p3, p5]
ears, n = [], 0
for line in open(sys.argv[2]):
    b = json.loads(line); n += 1
    if not b.get("face_valid", b.get("face_detected")) or b["left_eye_pts"] is None:
        ears.append(None); continue
    l = EyeDetector._calc_EAR_eye(np.array(b["left_eye_pts"], float)[ORDER])
    r = EyeDetector._calc_EAR_eye(np.array(b["right_eye_pts"], float)[ORDER])
    ears.append((l + r) / 2)
closed = [e is not None and e < 0.20 for e in ears]
ev, s = [], None
for i, c in enumerate(closed + [False]):
    if c and s is None: s = i
    elif not c and s is not None: ev.append((s, i - 1)); s = None
valid = [e for e in ears if e is not None]
print(f"blocks={n} valid={len(valid)} EAR median={np.median(valid):.3f} closure_events(EAR<0.20)={len(ev)}")
