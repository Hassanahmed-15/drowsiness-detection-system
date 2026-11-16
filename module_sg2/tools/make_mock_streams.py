"""Generate deterministic mock streams so neighbouring sub-groups can work
without the live camera pipeline.

For each scenario an eye-opening (EAR) curve with known blinks / long
closures is scripted, converted into SG1 V1 face blocks (6-point eyes whose
EAR equals the scripted value), and passed through the REAL SG2 module.

Outputs (interfaces/sg2/mock/):
  mock_sg1_<scenario>.jsonl    SG1 V1 blocks  -> input for SG2 tests
  mock_sg2_<scenario>.jsonl    SG2 V1 records -> input for SG4 tests
  mock_ground_truth.json       scripted events per scenario

Run from module_sg2/:  python tools/make_mock_streams.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sg2_eye import EyeStateModule  # noqa: E402
from sg2_eye.module import to_jsonable  # noqa: E402

FPS = 30.0
SECONDS = 60
OUT = Path(__file__).resolve().parents[2] / "interfaces" / "sg2" / "mock"
CONFIG = Path(__file__).resolve().parent.parent / "configs" / "baseline_w4_ear_fixed.yaml"

SCENARIOS = {
    # open EAR, blink gap (s), blink duration (ms), long closures [(start s, duration ms)], face lost [(s, ms)]
    "alert": dict(open_ear=0.30, gap=(2.5, 5.0), blink_ms=(120, 300), long=[], lost=[]),
    "drowsy": dict(open_ear=0.24, gap=(1.5, 3.5), blink_ms=(300, 480),
                   long=[(14.0, 1200), (31.0, 2000), (47.0, 2800)], lost=[]),
    "face_lost": dict(open_ear=0.29, gap=(2.5, 5.0), blink_ms=(120, 300), long=[],
                      lost=[(20.0, 1000), (40.0, 400)]),
}


def eye_points(cx, cy, width, ear, outer_left):
    """6 SG1-ordered points (p1 outer, p2/p3 upper, p4 inner, p5/p6 lower) with the given EAR."""
    h = ear * width / 2.0  # EAR = (2h + 2h) / (2 w)
    sign = -1 if outer_left else 1  # driver's right eye has its outer corner on the image left
    x1, x4 = cx + sign * width / 2, cx - sign * width / 2
    xa, xb = cx + sign * width / 6, cx - sign * width / 6
    pts = [(x1, cy), (xa, cy - h), (xb, cy - h), (x4, cy), (xb, cy + h), (xa, cy + h)]
    return [[round(x, 2), round(y, 2)] for x, y in pts]


def script(name, p, rng):
    n = int(SECONDS * FPS)
    ear = p["open_ear"] + rng.normal(0, 0.008, n)
    events = []

    def closure(start_f, dur_ms, kind):
        frames = max(1, int(round(dur_ms / 1000 * FPS)))
        ramp = 1  # one partially-closed frame on each side (lid moving)
        for k in range(-ramp, frames + ramp):
            f = start_f + k
            if 0 <= f < n:
                depth = 1.0 if 0 <= k < frames else 0.15
                ear[f] = p["open_ear"] * (1 - depth) + 0.04 * depth + rng.normal(0, 0.004)
        events.append({"type": kind, "start_ms": round(start_f * 1000 / FPS, 1), "duration_ms": dur_ms})

    long_frames = {int(s * FPS) for s, _ in p["long"]}
    for s, d in p["long"]:
        closure(int(s * FPS), d, "long_closure")
    t = rng.uniform(*p["gap"])
    while t < SECONDS - 1:
        f = int(t * FPS)
        if all(abs(f - lf) > 4 * FPS for lf in long_frames):
            closure(f, int(rng.uniform(*p["blink_ms"])), "blink")
        t += rng.uniform(*p["gap"])
    lost = np.zeros(n, bool)
    for s, d in p["lost"]:
        lost[int(s * FPS):int((s + d / 1000) * FPS)] = True
        events.append({"type": "face_lost", "start_ms": s * 1000, "duration_ms": d})
    return ear, lost, sorted(events, key=lambda e: e["start_ms"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(477)
    truth = {}
    for name, p in SCENARIOS.items():
        ear, lost, events = script(name, p, rng)
        module = EyeStateModule(CONFIG)
        with open(OUT / f"mock_sg1_{name}.jsonl", "w") as f1, open(OUT / f"mock_sg2_{name}.jsonl", "w") as f2:
            for i, e in enumerate(ear):
                ts = round(i * 1000 / FPS, 1)
                ok = not lost[i]
                block = {"frame_id": i, "timestamp_ms": ts, "face_detected": ok, "face_valid": ok,
                         "face_confidence": None, "face_box": [250.0, 150.0, 410.0, 360.0] if ok else None,
                         "left_eye_pts": eye_points(370, 230, 40, float(e), outer_left=False) if ok else None,
                         "right_eye_pts": eye_points(290, 230, 40, float(e), outer_left=True) if ok else None,
                         "mouth_pts": None, "landmarks": None,
                         "left_eye_roi_box": [344.0, 218.0, 396.0, 242.0] if ok else None,
                         "right_eye_roi_box": [264.0, 218.0, 316.0, 242.0] if ok else None,
                         "mouth_roi_box": None,
                         "head_pose": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0} if ok else None}
                f1.write(json.dumps(block) + "\n")
                f2.write(json.dumps(to_jsonable(module.process(block))) + "\n")
        truth[name] = {"fps": FPS, "frames": len(ear), "description": p, "scripted_events": events,
                       "sg2_detected_blinks": module.blink.blink_count,
                       "sg2_detected_long_closures": module.blink.long_closure_count}
        print(f"{name}: scripted {sum(e['type'] == 'blink' for e in events)} blinks / "
              f"{sum(e['type'] == 'long_closure' for e in events)} long closures -> SG2 found "
              f"{module.blink.blink_count} / {module.blink.long_closure_count}")
    (OUT / "mock_ground_truth.json").write_text(json.dumps(truth, indent=2))


if __name__ == "__main__":
    main()
