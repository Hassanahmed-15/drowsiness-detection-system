"""Quick self-checks for SG2. Run from module_sg2/:  python -m pytest tests -q
(or plain `python tests/test_sg2.py`). Needs no dataset or camera.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "tools"))
from make_mock_streams import eye_points  # noqa: E402
from sg2_eye import BlinkDetector, EyeStateModule, eye_aspect_ratio, eye_crop  # noqa: E402

MOCK = HERE.parents[1] / "interfaces" / "sg2" / "mock"
BASELINE = HERE.parent / "configs" / "baseline_w4_ear_fixed.yaml"


def test_ear_matches_formula():
    pts = eye_points(100, 100, 40, 0.30, outer_left=True)
    assert abs(eye_aspect_ratio(pts) - 0.30) < 1e-3
    assert eye_aspect_ratio(None) is None


def test_ear_same_for_both_eye_orderings():
    left = eye_points(200, 100, 40, 0.25, outer_left=False)
    right = eye_points(100, 100, 40, 0.25, outer_left=True)
    assert abs(eye_aspect_ratio(left) - eye_aspect_ratio(right)) < 1e-6


def test_eye_crop_shape():
    gray = np.random.default_rng(0).integers(0, 255, (480, 640), dtype=np.uint8)
    crop = eye_crop(gray, eye_points(300, 200, 40, 0.3, outer_left=True))
    assert crop.shape == (32, 32) and crop.dtype == np.uint8


def test_blink_vs_long_closure():
    b = BlinkDetector(max_blink_ms=500)
    states = ["open"] * 3 + ["closed"] * 6 + ["open"] * 3 + ["closed"] * 30 + ["open"]
    events = [e for i, s in enumerate(states) if (e := b.update(i, i * 33.3, s))]
    assert [e["type"] for e in events] == ["blink", "long_closure"]
    assert b.blink_count == 1 and b.long_closure_count == 1


def test_short_face_loss_does_not_split_closure():
    b = BlinkDetector(max_gap_frames=2)
    states = ["closed"] * 3 + ["unknown"] * 2 + ["closed"] * 3 + ["open"]
    events = [e for i, s in enumerate(states) if (e := b.update(i, i * 33.3, s))]
    assert len(events) == 1 and events[0]["start_frame"] == 0 and events[0]["end_frame"] == 7


def test_invalid_face_gives_unknown():
    m = EyeStateModule(BASELINE)
    rec = m.process({"frame_id": 0, "timestamp_ms": 0.0, "face_valid": False,
                     "left_eye_pts": None, "right_eye_pts": None})
    assert rec["eye_state"] == "unknown" and rec["eye_valid"] is False


def test_mock_drowsy_stream_reproduces_ground_truth():
    truth = json.loads((MOCK / "mock_ground_truth.json").read_text())["drowsy"]
    m = EyeStateModule(BASELINE)
    with open(MOCK / "mock_sg1_drowsy.jsonl") as fh:
        for line in fh:
            m.process(json.loads(line))
    assert m.blink.blink_count == truth["sg2_detected_blinks"]
    assert m.blink.long_closure_count == truth["sg2_detected_long_closures"] == 3


def test_output_has_all_schema_fields():
    schema = json.loads((HERE.parents[1] / "interfaces" / "sg2" / "sg2_eye_state_v1.schema.json").read_text())
    m = EyeStateModule(BASELINE)
    with open(MOCK / "mock_sg1_alert.jsonl") as fh:
        rec = m.process(json.loads(fh.readline()))
    assert set(schema["required"]) <= set(rec)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
