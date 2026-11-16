"""Export a 20-second REAL test sample as SG1 V1 blocks + hand labels.

Source: Eyeblink8 TEST video 9 (never used for tuning). The window with the
most annotated blinks is chosen. Only landmark coordinates and labels are
written (no images), so the sample is small and contains no raw video.
SG1 blocks come from the SG1-compatible MediaPipe source (same model and
indices as module_sg1). `landmarks` (478 points) is left null to keep it small.

Run from module_sg2/:  python experiments/export_real_sample.py
Output: interfaces/sg2/examples/eb8_09_real_sg1.jsonl
        interfaces/sg2/examples/eb8_09_real_sg1.labels.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from common import DATA_ROOT, MODULE_DIR, REPO_DIR, VIDEOS, blink_intervals, parse_tag  # noqa: E402
from sg1_landmarks import SG1LandmarkSource  # noqa: E402

VIDEO, SECONDS = "eb8_09", 20
OUT = REPO_DIR / "interfaces" / "sg2" / "examples"


def main():
    stem = DATA_ROOT / VIDEOS[VIDEO]
    cap = cv2.VideoCapture(str(stem) + ".avi")
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30.0
    label, blink_id, meta = parse_tag(str(stem) + ".tag", n)
    win = int(SECONDS * fps)
    starts = [s for s, _ in blink_intervals(blink_id)]
    best = max(range(0, n - win, 30), key=lambda a: sum(a <= s < a + win - 15 for s in starts))
    cap.set(cv2.CAP_PROP_POS_FRAMES, best)
    src = SG1LandmarkSource(MODULE_DIR / "models" / "face_landmarker.task")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{VIDEO}_real_sg1.jsonl", "w") as fh:
        for i in range(win):
            ok, frame = cap.read()
            if not ok:
                break
            b = src.process(frame, i, round(i * 1000.0 / fps, 1))
            fh.write(json.dumps(b) + "\n")
    src.close()
    lab = label[best:best + win]
    blinks = [(s - best, e - best) for s, e in blink_intervals(blink_id) if best <= s and e < best + win]
    info = {"source": f"Eyeblink8 video 9 ({stem.name}.avi), TEST split", "source_first_frame": best,
            "fps": fps, "frames": int(len(lab)), "glasses": meta.get("glasses", "NO"),
            "label_rule": "1 = annotated fully closed, 0 = open (outside blinks), -1 = ignore",
            "label": lab.astype(int).tolist(), "blinks": blinks,
            "citation": "Drutarovsky & Fogelton, ECCV Workshops 2014 (blinkingmatters.com); research use"}
    (OUT / f"{VIDEO}_real_sg1.labels.json").write_text(json.dumps(info))
    print(f"exported frames {best}-{best + win - 1} of {VIDEO}: {len(blinks)} annotated blinks, "
          f"{int((lab == 1).sum())} closed frames")


if __name__ == "__main__":
    main()
