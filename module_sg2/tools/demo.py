"""SG2 live demonstration: eye state + blinks, with ground truth when available.

Run from module_sg2/. Keys in the window: q = quit, space = pause, r = reset.

  # 1. live webcam (laptop or Jetson USB camera); Jetson CSI camera: --source csi
  python tools/demo.py --source 0

  # 2. saved TEST video with its hand-made answer file (.tag found automatically)
  python tools/demo.py --source ../datasets/raw/sg2/eyeblink8/eyeblink8/8/27122013_151644_cam.avi

  # 3. no camera, no SG1: mock SG1 V1 stream with scripted ground truth
  python tools/demo.py --sg1-jsonl ../interfaces/sg2/mock/mock_sg1_drowsy.jsonl

Options: --config configs/backup.yaml (default configs/primary.yaml),
         --out records.jsonl (save sg2-v1 records), --print-json (show records in the terminal),
         --save demo.mp4 (record the window), --no-window (headless, e.g. over SSH)
"""
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODULE_DIR = HERE.parent
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MODULE_DIR / "experiments"))
from camera import FrameClock, open_source  # noqa: E402
from sg2_eye import EyeStateModule  # noqa: E402
from sg2_eye.module import to_jsonable  # noqa: E402

GREEN, RED, GREY, WHITE, AMBER = (60, 170, 60), (50, 50, 220), (150, 150, 150), (245, 245, 245), (40, 160, 235)
DARK = (32, 30, 28)
PANEL_W, PLOT_H, VIEW_H, HISTORY = 360, 170, 480, 180


# ----------------------------------------------------------- ground truth
def load_ground_truth(args):
    """Per-frame labels (1 closed / 0 open / -1 ignore) and blink intervals, if available."""
    if args.sg1_jsonl:
        truth = Path(args.sg1_jsonl).with_name("mock_ground_truth.json")
        if not truth.exists():
            return None
        name = Path(args.sg1_jsonl).stem.replace("mock_sg1_", "")
        info = json.loads(truth.read_text()).get(name)
        if not info:
            return None
        n, fps = info["frames"], info["fps"]
        label = np.zeros(n, np.int8)
        blinks = []
        for e in info["scripted_events"]:
            s = int(round(e["start_ms"] * fps / 1000))
            d = max(1, int(round(e["duration_ms"] * fps / 1000)))
            if e["type"] == "face_lost":
                label[s:s + d] = -1
            else:
                label[s:s + d] = 1
                blinks.append((s, s + d - 1))
        return {"label": label, "blinks": blinks, "source": "scripted mock ground truth"}
    tag = Path(args.labels) if args.labels else Path(args.source).with_suffix(".tag")
    if args.source and tag.exists():
        from common import blink_intervals, parse_tag

        n = int(cv2.VideoCapture(str(args.source)).get(cv2.CAP_PROP_FRAME_COUNT))
        label, blink_id, _ = parse_tag(str(tag), n)
        return {"label": label, "blinks": blink_intervals(blink_id), "source": f"hand labels {tag.name}"}
    return None


class Score:
    """Running agreement between SG2 and ground truth (closed = positive)."""

    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def add(self, gt, pred_closed):
        if gt == 1:
            self.tp += pred_closed
            self.fn += not pred_closed
        elif gt == 0:
            self.fp += pred_closed
            self.tn += not pred_closed

    def text(self):
        n = self.tp + self.fp + self.fn + self.tn
        if n == 0:
            return "no labelled frames yet"
        f1 = 2 * self.tp / max(2 * self.tp + self.fp + self.fn, 1)
        return f"agree {(self.tp + self.tn) / n:.1%}  F1 closed {f1:.2f}  (FP {self.fp}, FN {self.fn})"


# ------------------------------------------------------------- drawing
def schematic(block, size=(640, 480)):
    """No-image mode: draw the SG1 eye points on a blank canvas."""
    canvas = np.full((size[1], size[0], 3), 52, np.uint8)
    cv2.putText(canvas, "mock SG1 input (no camera image)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
    pts = [p for k in ("left_eye_pts", "right_eye_pts") for p in (block.get(k) or [])]
    if not pts:
        return canvas, None
    p = np.array(pts, np.float32)
    centre = p.mean(axis=0)
    scale = 3.0
    return canvas, (centre, scale, np.array(size, np.float32) / 2)


def to_canvas(pt, xf):
    if xf is None:
        return int(pt[0]), int(pt[1])
    centre, scale, mid = xf
    q = (np.asarray(pt, np.float32) - centre) * scale + mid
    return int(q[0]), int(q[1])


def draw_eyes(img, block, rec, xf):
    colour = {"open": GREEN, "closed": RED}.get(rec["eye_state"], GREY)
    for key in ("left_eye_pts", "right_eye_pts"):
        pts = block.get(key)
        if not pts:
            continue
        poly = np.array([to_canvas(p, xf) for p in pts], np.int32)
        cv2.polylines(img, [poly], True, colour, 2 if xf else 1, cv2.LINE_AA)
        for q in poly:
            cv2.circle(img, tuple(int(v) for v in q), 3 if xf else 2, colour, -1, cv2.LINE_AA)


def panel(rec, module, gt_now, score, fps, ms, crops, cfg_name, truth):
    p = np.full((VIEW_H, PANEL_W, 3), DARK, np.uint8)
    state = rec["eye_state"].upper()
    colour = {"OPEN": GREEN, "CLOSED": RED}.get(state, GREY)
    cv2.putText(p, "SG2 eye state", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1, cv2.LINE_AA)
    cv2.putText(p, state, (16, 78), cv2.FONT_HERSHEY_DUPLEX, 1.6, colour, 2, cv2.LINE_AA)
    y = 112
    if truth is not None:
        g = {1: "CLOSED", 0: "OPEN"}.get(gt_now, "not labelled")
        ok = (gt_now == 1) == (rec["eye_state"] == "closed") if gt_now in (0, 1) and rec["eye_valid"] else None
        mark = {True: ("correct", GREEN), False: ("wrong", RED), None: ("-", GREY)}[ok]
        cv2.putText(p, f"ground truth: {g}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        cv2.putText(p, mark[0], (250, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mark[1], 2, cv2.LINE_AA)
        y += 26
    d = module.last_decision or {}
    lines = [
        f"EAR {rec['ear'] if rec['ear'] is not None else '-'}   threshold {rec['threshold'] if rec['threshold'] is not None else '-'}",
        f"CNN P(closed) {d['cnn_p_closed']:.2f}" if d.get("cnn_p_closed") is not None else "CNN check: only when EAR says closed",
        f"closed for {rec['closed_duration_ms']:.0f} ms",
        f"blinks {rec['blink_count']}    long closures {rec['long_closure_count']}",
    ]
    if truth is not None:
        lines.append(f"annotated closures so far {sum(1 for _, e in truth['blinks'] if e <= rec['frame_id'])}")
    for line in lines:
        cv2.putText(p, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        y += 24
    if rec["blink_event"]:
        ev = rec["blink_event"]
        cv2.putText(p, f"{ev['type'].replace('_', ' ')}: {ev['duration_ms']:.0f} ms", (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, AMBER, 2, cv2.LINE_AA)
    if crops is not None:
        cy = 304
        cv2.putText(p, "eye crops seen by the CNN (R | L)", (16, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1)
        for i, c in enumerate(crops):
            big = cv2.resize(c, (80, 80), interpolation=cv2.INTER_NEAREST)
            p[cy:cy + 80, 16 + i * 92:96 + i * 92] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    if truth is not None:
        cv2.putText(p, score.text(), (16, VIEW_H - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
        cv2.putText(p, f"labels: {truth['source']}", (16, VIEW_H - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREY, 1)
    cv2.putText(p, f"config {cfg_name}"[:52], (16, VIEW_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREY, 1)
    cv2.putText(p, f"processing {fps:4.1f} fps | SG1 {ms[0]:.1f} ms | SG2 {ms[1]:.2f} ms", (16, VIEW_H - 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1, cv2.LINE_AA)
    return p


def plot(hist, width):
    g = np.full((PLOT_H, width, 3), 24, np.uint8)
    if len(hist) < 2:
        return g
    ys = lambda v: int(PLOT_H - 18 - min(max(v, 0.0), 0.45) / 0.45 * (PLOT_H - 36))  # noqa: E731
    xs = lambda i: int(10 + i * (width - 20) / (HISTORY - 1))  # noqa: E731
    for i, h in enumerate(hist):
        if h["pred"]:
            cv2.rectangle(g, (xs(i) - 2, 8), (xs(i) + 2, PLOT_H - 18), (40, 40, 110), -1)
        if h["gt"] == 1:
            cv2.rectangle(g, (xs(i) - 2, PLOT_H - 14), (xs(i) + 2, PLOT_H - 6), (235, 160, 40), -1)
    for i in range(1, len(hist)):
        a, b = hist[i - 1], hist[i]
        if a["ear"] is not None and b["ear"] is not None:
            cv2.line(g, (xs(i - 1), ys(a["ear"])), (xs(i), ys(b["ear"])), WHITE, 1, cv2.LINE_AA)
        if a["thr"] is not None and b["thr"] is not None:
            cv2.line(g, (xs(i - 1), ys(a["thr"])), (xs(i), ys(b["thr"])), AMBER, 1, cv2.LINE_AA)
    cv2.putText(g, "EAR (white), threshold (orange), SG2 closed (red band), ground-truth closed (blue ticks)",
                (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1, cv2.LINE_AA)
    return g


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="camera index (0), 'csi' for a Jetson CSI camera, or a video file")
    src.add_argument("--sg1-jsonl", help="SG1 V1 blocks (e.g. the mock streams) - no camera needed")
    ap.add_argument("--labels", help="ground-truth .tag file (default: <video>.tag if it exists)")
    ap.add_argument("--config", default=str(MODULE_DIR / "configs" / "primary.yaml"))
    ap.add_argument("--landmark-model", default=str(MODULE_DIR / "models" / "face_landmarker.task"))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", help="write sg2-v1 records (JSONL)")
    ap.add_argument("--print-json", action="store_true", help="print every record to the terminal")
    ap.add_argument("--save", help="save the demo window to a video file")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    module = EyeStateModule(args.config)
    truth = load_ground_truth(args)
    score, hist = Score(), deque(maxlen=HISTORY)
    out = open(args.out, "w") if args.out else None
    writer, paused, n, t_last, fps = None, False, 0, time.perf_counter(), 0.0
    sg1_ms_avg, sg2_ms_avg = 0.0, 0.0

    if args.sg1_jsonl:
        stream = (json.loads(line) for line in open(args.sg1_jsonl))
        frames = None
        delay_ms = 33
    else:
        from sg1_landmarks import SG1LandmarkSource

        cap, kind = open_source(args.source, args.width, args.height)
        if not cap.isOpened():
            sys.exit(f"[demo] could not open video source {args.source!r} "
                     f"(camera permission? wrong index? for Jetson CSI use --source csi)")
        clock = FrameClock(kind, cap.get(cv2.CAP_PROP_FPS))
        source = SG1LandmarkSource(args.landmark_model,
                                   with_blendshapes=module.classifier.name in ("blendshape", "ear_adaptive_blendshape"))
        delay_ms = 1
        print(f"[demo] source={args.source} ({kind}), config={module.config_name}, "
              f"ground truth={'yes: ' + truth['source'] if truth else 'none'}", file=sys.stderr)

    while True:
        key = cv2.waitKey(delay_ms if not paused else 50) & 0xFF if not args.no_window else 255
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("r"):
            module.reset()
            score = Score()
        if paused:
            continue
        if args.sg1_jsonl:
            block = next(stream, None)
            if block is None:
                break
            frame, sg1_ms = None, 0.0
        else:
            ok, frame = cap.read()
            if not ok:
                break
            ts = clock.next()
            t0 = time.perf_counter()
            block = source.process(frame, n, ts)
            sg1_ms = 1000 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        rec = module.process(block, frame)
        sg2_ms = 1000 * (time.perf_counter() - t0)
        sg1_ms_avg = 0.9 * sg1_ms_avg + 0.1 * sg1_ms if n else sg1_ms
        sg2_ms_avg = 0.9 * sg2_ms_avg + 0.1 * sg2_ms if n else sg2_ms
        fid = rec["frame_id"]
        gt_now = int(truth["label"][fid]) if truth is not None and fid < len(truth["label"]) else None
        if gt_now is not None and rec["eye_valid"]:
            score.add(gt_now, rec["eye_state"] == "closed")
        hist.append({"ear": rec["ear"], "thr": rec["threshold"], "pred": rec["eye_state"] == "closed", "gt": gt_now})
        if out:
            out.write(json.dumps(to_jsonable(rec)) + "\n")
        if args.print_json and (rec["blink_event"] or n % 30 == 0):
            print(json.dumps(to_jsonable(rec)))
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6) if n else 0.0
        t_last = now

        if frame is not None:
            view = frame.copy()
            xf = None
        else:
            view, xf = schematic(block)
        draw_eyes(view, block, rec, xf)
        scale = VIEW_H / view.shape[0]
        view = cv2.resize(view, (int(view.shape[1] * scale), VIEW_H))
        f = module.last_features
        crops = (f["right_crop"], f["left_crop"]) if f and f.get("left_crop") is not None else None
        side = panel(rec, module, gt_now, score, fps, (sg1_ms_avg, sg2_ms_avg), crops, module.config_name, truth)
        top = np.hstack([view, side])
        canvas = np.vstack([top, plot(hist, top.shape[1])])
        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                         (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
        if not args.no_window:
            cv2.imshow("SG2 eye state & blink analysis", canvas)
        n += 1
        if args.max_frames and n >= args.max_frames:
            break

    if writer:
        writer.release()
    if out:
        out.close()
    if not args.sg1_jsonl:
        source.close()
        cap.release()
    cv2.destroyAllWindows()
    summary = (f"[demo] {n} frames | blinks {module.blink.blink_count} | long closures "
               f"{module.blink.long_closure_count}")
    if truth is not None:
        gt_blinks = sum(1 for _, e in truth["blinks"] if e < n)
        summary += f" | annotated closures {gt_blinks} | {score.text()}"
    print(summary, file=sys.stderr)


if __name__ == "__main__":
    main()
