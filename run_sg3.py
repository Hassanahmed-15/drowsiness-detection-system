#!/usr/bin/env python3
"""Run SG-3 on a video file or webcam and write the per-frame `mouth` blocks as JSON.

Examples:
    python run_sg3.py --input clip.mp4 --output results/sg3_output.json
    python run_sg3.py --input 0 --show
    python run_sg3.py --input clip.mp4 --mar-thresh 0.55
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from sg3 import SG3, SG3Config
from sg3.upstream_mediapipe import MediaPipeUpstream
from sg3.yawn import mouth_aspect_ratio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SG-3 yawn and head-pose cues")
    p.add_argument("--input", required=True, help="video path, or camera index such as 0")
    p.add_argument("--output", default="results/sg3_output.json", help="JSON output path")
    p.add_argument("--mar-thresh", type=float, default=None,
                   help="MAR value that maps to yawn_prob 0.5 (default from SG3Config)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    p.add_argument("--show", action="store_true", help="display a live overlay (press q to quit)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open input: {args.input}", file=sys.stderr)
        return 1

    cfg = SG3Config()
    if args.mar_thresh is not None:
        cfg.mar_center = args.mar_thresh
    sg3 = SG3(cfg)
    upstream = MediaPipeUpstream()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    results = []
    frame_id = 0
    t0 = time.perf_counter()
    if args.show:
        cv2.namedWindow("SG-3", cv2.WINDOW_NORMAL)   # makes the window resizable
        cv2.resizeWindow("SG-3", 960, 540)           # starting width, height in pixels
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts_ms = int(round(frame_id * 1000.0 / fps))
        up = upstream(frame)
        out = sg3.process(
            frame_id=frame_id,
            timestamp_ms=ts_ms,
            face_valid=up.face_valid,
            head_pose=up.head_pose,
            mouth_pts=up.mouth_pts,
        )
        results.append(out.to_dict())

        if args.show:
            mar = mouth_aspect_ratio(up.mouth_pts)
            txt = (f"MAR={mar:.2f} yawn={out.yawn_prob:.2f}"
                   if out.yawn_prob is not None and mar is not None else "MAR=n/a")
            cues = ",".join(c.value for c in out.cue_flags) if out.cue_flags else "-"
            cv2.putText(frame, f"{txt}  cues={cues}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if up.mouth_pts is not None:
                for x, y in up.mouth_pts:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv2.imshow("SG-3", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_id += 1
        if args.max_frames and frame_id >= args.max_frames:
            break

    elapsed = time.perf_counter() - t0
    cap.release()
    upstream.close()
    if args.show:
        cv2.destroyAllWindows()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"{frame_id} frames in {elapsed:.1f}s ({1000 * elapsed / max(frame_id, 1):.1f} ms/frame, includes upstream) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
