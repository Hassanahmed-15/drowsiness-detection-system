"""SG2 latency / FPS benchmark (midterm submission).

Times every stage separately on the SAME frames of a video:
  decode    - cv2.VideoCapture.read()
  landmarks - SG1-compatible MediaPipe Face Landmarker (this is SG1's job in the full system)
  SG2       - EyeStateModule.process() for the PRIMARY (config G) and BACKUP (config C) configs
End-to-end FPS = 1000 / (decode + landmarks + SG2) per frame, offline (not limited by a camera).

Run from module_sg2/midterm/code/:
  python tools/benchmark_latency.py --video <test video> --frames 3000
  python tools/benchmark_latency.py --sg1-jsonl examples/eb8_09_real_sg1.jsonl   # SG2 only, no video needed
Writes ../latency.json (raw numbers) and prints a markdown table.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(HERE))
from camera import FrameClock  # noqa: E402
from sg2_eye import EyeStateModule  # noqa: E402

WARMUP = 30


def stats(ms):
    a = np.asarray(ms[WARMUP:] if len(ms) > 2 * WARMUP else ms, dtype=float)
    return {"frames": int(a.size), "mean_ms": round(float(a.mean()), 3), "median_ms": round(float(np.median(a)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3), "max_ms": round(float(a.max()), 3),
            "fps_equivalent": round(1000.0 / float(a.mean()), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--sg1-jsonl")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--out", default=str(CODE.parent / "latency.json"))
    args = ap.parse_args()

    primary = EyeStateModule(CODE / "configs" / "primary.yaml")
    backup = EyeStateModule(CODE / "configs" / "backup.yaml")
    t = {"decode": [], "landmarks": [], "sg2_primary": [], "sg2_backup": []}
    faces = n = 0
    if args.video:
        from sg1_landmarks import SG1LandmarkSource
        lm = SG1LandmarkSource(CODE / "models" / "face_landmarker.task", min_confidence=0.2)
        cap = cv2.VideoCapture(args.video)
        clock = FrameClock("file", cap.get(cv2.CAP_PROP_FPS))
        resolution = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        while n < args.frames:
            a = time.perf_counter(); ok, frame = cap.read(); b = time.perf_counter()
            if not ok:
                break
            block = lm.process(frame, n, clock.next()); c = time.perf_counter()
            primary.process(block, frame); d = time.perf_counter()
            backup.process(block, frame); e = time.perf_counter()
            t["decode"].append(1000 * (b - a)); t["landmarks"].append(1000 * (c - b))
            t["sg2_primary"].append(1000 * (d - c)); t["sg2_backup"].append(1000 * (e - d))
            faces += bool(block.get("face_valid")); n += 1
        cap.release()
        source = f"{Path(args.video).name} ({resolution})"
    else:
        blocks = [json.loads(line) for line in open(args.sg1_jsonl)]
        for block in blocks[: args.frames]:
            c = time.perf_counter(); primary.process(block); d = time.perf_counter()
            backup.process(block); e = time.perf_counter()
            t["sg2_primary"].append(1000 * (d - c)); t["sg2_backup"].append(1000 * (e - d))
            faces += bool(block.get("face_valid")); n += 1
        source = f"{Path(args.sg1_jsonl).name} (SG1 blocks, no frames: the CNN check is skipped)"

    res = {"source": source, "frames": n, "face_found_pct": round(100 * faces / max(n, 1), 1),
           "machine": f"{platform.machine()} / {platform.platform()}", "python": platform.python_version(),
           "opencv": cv2.__version__, "primary_cnn_loaded": bool(getattr(primary.classifier, "cnn_available", False)),
           "stages": {k: stats(v) for k, v in t.items() if v}}
    if t["decode"]:
        for cfg in ("primary", "backup"):
            total = np.add(np.add(t["decode"], t["landmarks"]), t[f"sg2_{cfg}"])
            res["stages"][f"end_to_end_{cfg}"] = stats(list(total))
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"source: {source} | frames {n} | face found {res['face_found_pct']}% | CNN loaded {res['primary_cnn_loaded']}")
    print("| Stage | Mean ms | Median ms | p95 ms | Max ms | FPS (1000/mean) |\n|---|---|---|---|---|---|")
    for k, s in res["stages"].items():
        print(f"| {k} | {s['mean_ms']} | {s['median_ms']} | {s['p95_ms']} | {s['max_ms']} | {s['fps_equivalent']} |")


if __name__ == "__main__":
    main()
