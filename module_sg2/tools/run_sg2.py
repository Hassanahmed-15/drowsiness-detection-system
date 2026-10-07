"""Run SG2 on a video (or on SG1 JSONL output) and write SG2 V1 records as JSONL.

Examples (run from module_sg2/):
  python tools/run_sg2.py --video driver.mp4 --config configs/baseline_w4_ear_fixed.yaml --out results/sample_output.jsonl
  python tools/run_sg2.py --sg1-jsonl ../interfaces/sg2/examples/mock_sg1_input.jsonl --config configs/baseline_w4_ear_fixed.yaml
  python tools/run_sg2.py --video driver.mp4 --config configs/w6_selected.yaml --overlay demo.mp4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sg2_eye import EyeStateModule  # noqa: E402
from sg2_eye.module import to_jsonable  # noqa: E402

DEFAULT_LANDMARK_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def video_frames(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, round(i * 1000.0 / fps, 1), frame
        i += 1
    cap.release()


def draw(frame, block, rec):
    colour = {"open": (0, 200, 0), "closed": (0, 0, 255), "unknown": (128, 128, 128)}[rec["eye_state"]]
    for key in ("left_eye_pts", "right_eye_pts"):
        for x, y in block.get(key) or []:
            cv2.circle(frame, (int(x), int(y)), 2, colour, -1)
    lines = [f"eye: {rec['eye_state']}  EAR: {rec['ear']}  thr: {rec['threshold']}",
             f"blinks: {rec['blink_count']}  long closures: {rec['long_closure_count']}  closed for: {rec['closed_duration_ms']} ms"]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="driver video; landmarks come from the SG1-compatible MediaPipe source")
    src.add_argument("--sg1-jsonl", help="SG1 V1 face blocks, one JSON per line")
    ap.add_argument("--frames-video", help="video matching --sg1-jsonl (only needed for the CNN method)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="-", help="output JSONL (default stdout)")
    ap.add_argument("--overlay", help="optional annotated output video (with --video)")
    ap.add_argument("--landmark-model", default=str(DEFAULT_LANDMARK_MODEL))
    args = ap.parse_args()

    module = EyeStateModule(args.config)
    out = sys.stdout if args.out == "-" else open(args.out, "w")
    writer, n, t_sg2 = None, 0, 0.0

    if args.video:
        from sg1_landmarks import SG1LandmarkSource

        source = SG1LandmarkSource(args.landmark_model,
                                   with_blendshapes=module.classifier.name == "blendshape")
        for fid, ts, frame in video_frames(args.video):
            block = source.process(frame, fid, ts)
            t0 = time.perf_counter()
            rec = module.process(block, frame)
            t_sg2 += time.perf_counter() - t0
            out.write(json.dumps(to_jsonable(rec)) + "\n")
            if args.overlay:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.overlay, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                writer.write(draw(frame, block, rec))
            n += 1
        source.close()
    else:
        frames = video_frames(args.frames_video) if args.frames_video else None
        with open(args.sg1_jsonl) as fh:
            for line in fh:
                block = json.loads(line)
                frame = next(frames)[2] if frames else None
                t0 = time.perf_counter()
                rec = module.process(block, frame)
                t_sg2 += time.perf_counter() - t0
                out.write(json.dumps(to_jsonable(rec)) + "\n")
                n += 1

    if writer:
        writer.release()
    if out is not sys.stdout:
        out.close()
    print(f"[sg2] {n} frames, config={module.config_name}, blinks={module.blink.blink_count}, "
          f"long_closures={module.blink.long_closure_count}, "
          f"SG2 time {1000 * t_sg2 / max(n, 1):.3f} ms/frame", file=sys.stderr)


if __name__ == "__main__":
    main()
