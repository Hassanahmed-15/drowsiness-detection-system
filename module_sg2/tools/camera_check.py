"""Video-acquisition readiness test (USB / Jetson CSI camera, or a file as fallback).

Opens the source, grabs N frames, measures the real frame rate and
resolution, saves the first and last frame, and (optionally) runs the
SG1-compatible landmarks + SG2 on every frame to show the camera feeds the
pipeline. A file source only proves the software path, not a live camera
(as the midterm brief says), and the report marks it that way.

Run from module_sg2/:
    python tools/camera_check.py --source 0            # USB webcam
    python tools/camera_check.py --source csi          # Jetson CSI camera (GStreamer)
    python tools/camera_check.py --source 0 --pipeline # + landmarks + SG2 on each frame
    python tools/camera_check.py --source <video.avi>  # prerecorded fallback
Outputs (default ../embedded_sg6/reports/camera_<host>_<source>/): report.json, report.md, first.jpg, last.jpg
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
MODULE_DIR = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MODULE_DIR))
from camera import FrameClock, open_source  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="0/1/... (USB), csi or csi:N (Jetson CSI), or a video file")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--pipeline", action="store_true", help="also run SG1-compatible landmarks + SG2 on each frame")
    ap.add_argument("--config", default=str(MODULE_DIR / "configs" / "primary.yaml"))
    ap.add_argument("--out-dir")
    args = ap.parse_args()

    tag = str(args.source).replace("/", "_").replace(":", "")[-40:]
    out = Path(args.out_dir or MODULE_DIR.parent / "embedded_sg6" / "reports" / f"camera_{platform.node()}_{tag}")
    out.mkdir(parents=True, exist_ok=True)
    report = {"date": time.strftime("%Y-%m-%d %H:%M"), "host": platform.node(), "machine": platform.machine(),
              "source": args.source, "opencv": cv2.__version__}

    t_open = time.perf_counter()
    cap, kind = open_source(args.source, args.width, args.height)
    report["kind"] = kind
    report["proves_live_camera"] = kind in ("usb", "csi")
    report["opened"] = bool(cap.isOpened())
    report["open_seconds"] = round(time.perf_counter() - t_open, 2)
    if not cap.isOpened():
        report["result"] = "FAILED to open"
        report["next_actions"] = [
            "USB: check `ls /dev/video*` and `v4l2-ctl --list-devices`; try --source 1",
            "CSI: check ribbon cable orientation, `systemctl status nvargus-daemon`, and that OpenCV has GStreamer "
            "(tools/check_env.py)",
            "macOS: give the terminal/VS Code Camera permission (System Settings > Privacy & Security > Camera)"]
        finish(report, out)
        sys.exit(1)

    module = source = None
    if args.pipeline:
        from sg1_landmarks import SG1LandmarkSource
        from sg2_eye import EyeStateModule

        source = SG1LandmarkSource(MODULE_DIR / "models" / "face_landmarker.task")
        module = EyeStateModule(args.config)
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    clock = FrameClock(kind, reported_fps)
    stamps, first, last, faces, states, sg1_ms, sg2_ms, failures = [], None, None, 0, {}, [], [], 0
    t0 = time.perf_counter()
    while len(stamps) < args.frames:
        ok, frame = cap.read()
        if not ok:
            failures += 1
            if failures > 30 or kind == "file":
                break
            continue
        stamps.append(time.perf_counter())
        first = frame if first is None else first
        last = frame
        if args.pipeline:
            a = time.perf_counter()
            block = source.process(frame, len(stamps) - 1, clock.next())
            b = time.perf_counter()
            rec = module.process(block, frame)
            c = time.perf_counter()
            sg1_ms.append(1000 * (b - a))
            sg2_ms.append(1000 * (c - b))
            faces += block["face_valid"]
            states[rec["eye_state"]] = states.get(rec["eye_state"], 0) + 1
    cap.release()
    elapsed = time.perf_counter() - t0
    n = len(stamps)
    report.update({
        "frames_read": n, "read_failures": failures,
        "resolution": None if first is None else f"{first.shape[1]}x{first.shape[0]}",
        "reported_fps": round(reported_fps or 0, 1),
        "measured_fps": round((n - 1) / (stamps[-1] - stamps[0]), 1) if n > 1 else 0.0,
        "mean_frame_brightness": None if first is None else round(float(np.mean(last)), 1),
        "elapsed_s": round(elapsed, 2),
    })
    if args.pipeline and n:
        report["pipeline"] = {"config": Path(args.config).name, "face_valid_rate": round(faces / n, 3),
                              "eye_states": states, "sg1_landmarks_ms_mean": round(float(np.mean(sg1_ms)), 2),
                              "sg2_ms_mean": round(float(np.mean(sg2_ms)), 3),
                              "end_to_end_fps": round(n / elapsed, 1),
                              "cnn_check_loaded": getattr(module.classifier, "cnn_available", None)}
        source.close()
    if first is not None:
        cv2.imwrite(str(out / "first.jpg"), first)
        cv2.imwrite(str(out / "last.jpg"), last)
    report["result"] = "OK" if n >= min(args.frames, 30) else "FAILED (too few frames)"
    if kind == "file":
        report["note"] = "prerecorded-video fallback: proves the software path only, NOT live camera integration"
    finish(report, out)


def finish(report, out):
    (out / "report.json").write_text(json.dumps(report, indent=2))
    lines = [f"# Camera check — {report['host']} — source `{report['source']}`", ""]
    lines += [f"| {k} | {json.dumps(v) if isinstance(v, (dict, list)) else v} |" for k, v in report.items()]
    lines.insert(2, "| Item | Value |\n|---|---|")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))
    print(f"[camera_check] report + frames in {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
