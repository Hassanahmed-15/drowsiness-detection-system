#!/usr/bin/env python3
"""
Latency / FPS benchmark for the SG-1 face and landmark module.

Measures, per frame and for both configurations (face_conf_min 0.50 and 0.70):
  convert : BGR -> RGB -> mp.Image
  detect  : FaceLandmarker.detect_for_video
  packet  : build_sg1_packet (driver face pick, landmarks, ROI boxes, ROI crops,
            solvePnP head pose, rule checks inputs)
  total   : convert + detect + packet

Video decoding and drawing the debug overlay are NOT included, because they are
not part of what SG-1 hands to SG-2 / SG-3 at run time.

Writes latency.md next to this script's parent folder (midterm/latency.md)
unless --out is given.

Usage:
  python benchmark_latency.py --video "Video Project.mp4"
  python benchmark_latency.py --video "Video Project.mp4" --warmup 10 --repeat 3
"""

import argparse
import datetime
import os
import platform
import sys
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

try:
    import Sg1_Pipeline as sg1
except ImportError:
    try:
        import sg1_face_pipeline as sg1
    except ImportError:
        sys.exit("Put this file in the same folder as Sg1_Pipeline.py "
                 "(or sg1_face_pipeline.py).")

CONFIGS = [("Config A", 0.50), ("Config B", 0.70)]


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def run_once(video, conf, max_faces, region, warmup, max_frames):
    detector = sg1.create_detector(vision.RunningMode.VIDEO, conf, max_faces)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    convert, detect, packet, total = [], [], [], []
    n = valid = detected = measured = 0
    while True:
        if max_frames and n >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        ts = sg1.frame_timestamp_ms(n, fps)

        t0 = time.perf_counter()
        img = sg1.to_mp_image(frame)
        t1 = time.perf_counter()
        result = detector.detect_for_video(img, ts)
        t2 = time.perf_counter()
        record, _ = sg1.build_sg1_packet(result, frame, n, ts, region)
        t3 = time.perf_counter()

        if n >= warmup:  # skip warm-up frames (model init, caches)
            convert.append((t1 - t0) * 1000)
            detect.append((t2 - t1) * 1000)
            packet.append((t3 - t2) * 1000)
            total.append((t3 - t0) * 1000)
            measured += 1
            detected += int(record["face"]["face_detected"])
            valid += int(record["face"]["face_valid"])
        n += 1

    cap.release()
    detector.close()
    return {
        "w": w, "h": h, "video_fps": fps, "frames_read": n, "measured": measured,
        "detected": detected, "valid": valid,
        "convert": convert, "detect": detect, "packet": packet, "total": total,
    }


def environment():
    return {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "mediapipe": getattr(mp, "__version__", "unknown"),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }


def main():
    p = argparse.ArgumentParser(description="SG-1 latency / FPS benchmark")
    p.add_argument("--video", required=True)
    p.add_argument("--out", default=None, help="Path of latency.md to write")
    p.add_argument("--max-faces", type=int, default=3)
    p.add_argument("--driver-region", type=sg1.parse_region,
                   default=(0.0, 0.0, 1.0, 1.0))
    p.add_argument("--warmup", type=int, default=10,
                   help="Frames excluded from the statistics")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after N frames (0 = whole video)")
    p.add_argument("--repeat", type=int, default=1,
                   help="Runs per configuration; the last run is reported")
    p.add_argument("--device-label", default="",
                   help="Free text, e.g. 'Laptop i5, CPU only' or 'Jetson Orin Nano'")
    args = p.parse_args()

    if not os.path.isfile(args.video):
        sys.exit(f"Video not found: {args.video}")
    sg1.ensure_model()

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out or os.path.join(here, "latency.md")

    rows = []
    video_info = None
    for name, conf in CONFIGS:
        print(f"Benchmarking {name} (face_conf_min={conf}) ...")
        res = None
        for r in range(args.repeat):
            res = run_once(args.video, conf, args.max_faces, args.driver_region,
                           args.warmup, args.max_frames)
            print(f"  run {r + 1}/{args.repeat}: "
                  f"mean total {np.mean(res['total']):.2f} ms")
        video_info = res
        rows.append((name, conf, res))

    env = environment()
    lines = []
    lines.append("# SG-1 Latency / FPS Results\n")
    lines.append("Measured by `code/benchmark_latency.py`. "
                 "All numbers below were produced on the machine described "
                 "here and will differ on other hardware (including the Jetson).\n")
    lines.append("## Test setup\n")
    lines.append("| Item | Value |")
    lines.append("|------|-------|")
    lines.append(f"| Device label | {args.device_label or '(not given)'} |")
    lines.append(f"| Date | {env['date']} |")
    lines.append(f"| OS | {env['platform']} |")
    lines.append(f"| Processor | {env['processor']} ({env['cpu_count']} logical cores) |")
    lines.append(f"| Python / MediaPipe / OpenCV / NumPy | "
                 f"{env['python']} / {env['mediapipe']} / {env['opencv']} / {env['numpy']} |")
    lines.append(f"| Video | {os.path.basename(args.video)}, "
                 f"{video_info['w']}x{video_info['h']}, "
                 f"{video_info['video_fps']:.2f} FPS nominal |")
    lines.append(f"| Frames read / measured | {video_info['frames_read']} / "
                 f"{video_info['measured']} (first {args.warmup} excluded as warm-up) |")
    lines.append(f"| Mode | MediaPipe Face Landmarker, VIDEO running mode, "
                 f"max_faces={args.max_faces}, CPU delegate (default) |")
    lines.append("")
    lines.append("## Summary (end-to-end SG-1, per frame)\n")
    lines.append("`total` = convert + detect + packet. It excludes video decoding "
                 "and the debug overlay. FPS is 1000 / mean total latency.\n")
    lines.append("| Configuration | face_conf_min | Mean (ms) | p50 (ms) | p95 (ms) | "
                 "Max (ms) | FPS (from mean) | FPS (from p95) | Detection rate | face_valid rate |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, conf, res in rows:
        s = stats(res["total"])
        m = res["measured"] or 1
        lines.append(
            f"| {name} | {conf:.2f} | {s['mean']:.2f} | {s['p50']:.2f} | "
            f"{s['p95']:.2f} | {s['max']:.2f} | {1000 / s['mean']:.1f} | "
            f"{1000 / s['p95']:.1f} | {res['detected'] / m * 100:.1f}% | "
            f"{res['valid'] / m * 100:.1f}% |")
    lines.append("")
    lines.append("## Per-stage breakdown (mean / p95, ms)\n")
    lines.append("| Configuration | convert | detect | packet | total |")
    lines.append("|---|---|---|---|---|")
    for name, conf, res in rows:
        cells = []
        for key in ("convert", "detect", "packet", "total"):
            s = stats(res[key])
            cells.append(f"{s['mean']:.2f} / {s['p95']:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Notes\n")
    lines.append("- `detect` is the MediaPipe inference call; `packet` covers "
                 "landmark extraction, ROI boxes, ROI crops and the solvePnP head pose.")
    lines.append("- Timing uses `time.perf_counter()` around each stage, single "
                 "process, one frame at a time (no batching, no threading).")
    lines.append("- The `avg_latency_ms` / `processing_fps` columns in "
                 "`config_comparison.csv` time only the `detect` call, so they "
                 "are lower-latency / higher-FPS than the end-to-end numbers here. "
                 "Use the end-to-end numbers when sharing with other subgroups.")
    lines.append("- Jetson Orin Nano figures are NOT included unless this "
                 "script was run on the Jetson (see Device label).")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
