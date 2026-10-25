"""V1 interface contract tests (midterm). Run from module_sg2/:  python -m pytest tests -q

Checks, without any dataset or camera:
  * SG1 V1 input: every block in the real sample and the mock streams has the
    fields SG2 reads, with 6 ordered [x, y] points per eye
  * SG2 output: every record from the primary and backup configs validates
    against interfaces/sg2/sg2_eye_state_v1.schema.json
  * primary (G) falls back to backup (C) when the CNN model is missing or no
    frame is given, instead of failing
  * the real 20 s sample reproduces its hand-labelled blinks
"""
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
REPO = MODULE.parent
sys.path.insert(0, str(MODULE))
from sg2_eye import EyeStateModule, load_config  # noqa: E402

SCHEMA = json.loads((REPO / "interfaces" / "sg2" / "sg2_eye_state_v1.schema.json").read_text())
REAL = REPO / "interfaces" / "sg2" / "examples" / "eb8_09_real_sg1.jsonl"
MOCKS = sorted((REPO / "interfaces" / "sg2" / "mock").glob("mock_sg1_*.jsonl"))
PRIMARY, BACKUP = MODULE / "configs" / "primary.yaml", MODULE / "configs" / "backup.yaml"
SG1_FIELDS = {"frame_id", "timestamp_ms", "face_detected", "face_valid", "face_confidence", "face_box",
              "left_eye_pts", "right_eye_pts", "mouth_pts", "landmarks", "left_eye_roi_box",
              "right_eye_roi_box", "mouth_roi_box", "head_pose"}


def blocks(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh]


@pytest.mark.parametrize("path", [REAL, *MOCKS], ids=lambda p: p.name)
def test_sg1_v1_input_contract(path):
    prev_ts = -1
    for b in blocks(path):
        assert SG1_FIELDS <= set(b), f"missing SG1 fields: {SG1_FIELDS - set(b)}"
        assert b["timestamp_ms"] > prev_ts
        prev_ts = b["timestamp_ms"]
        if b["face_valid"]:
            for key in ("left_eye_pts", "right_eye_pts"):
                pts = np.asarray(b[key], float)
                assert pts.shape == (6, 2), f"{key} must be 6 ordered [x, y] points"
                assert np.isfinite(pts).all()


@pytest.mark.parametrize("cfg", [PRIMARY, BACKUP], ids=["primary", "backup"])
@pytest.mark.parametrize("path", [REAL, *MOCKS], ids=lambda p: p.name)
def test_sg2_output_matches_schema(cfg, path):
    m = EyeStateModule(cfg)
    validator = jsonschema.Draft202012Validator(SCHEMA)
    for b in blocks(path):
        rec = json.loads(json.dumps(m.process(b)))
        errors = list(validator.iter_errors(rec))
        assert not errors, errors[0].message
        assert rec["frame_id"] == b["frame_id"] and rec["timestamp_ms"] == b["timestamp_ms"]


def run_counts(cfg, path, frames=None):
    m = EyeStateModule(cfg)
    states = [m.process(b)["eye_state"] for b in blocks(path)]
    return m.blink.blink_count, m.blink.long_closure_count, states


def test_primary_without_frame_equals_backup():
    assert run_counts(PRIMARY, REAL) == run_counts(BACKUP, REAL)


def test_primary_falls_back_when_model_missing(capsys):
    cfg = load_config(PRIMARY)
    cfg["classifier"]["model_path"] = "/nonexistent/eye_state_cnn_v2.onnx"
    m = EyeStateModule(cfg)
    assert m.classifier.cnn_available is False
    assert "falling back" in capsys.readouterr().err
    frame = np.zeros((480, 640, 3), np.uint8)
    b = blocks(REAL)[0]
    assert m.process(b, frame)["eye_state"] in ("open", "closed")


def test_real_sample_matches_hand_labels():
    info = json.loads(REAL.with_suffix(".labels.json").read_text())
    blinks, long_closures, states = run_counts(PRIMARY, REAL)
    assert blinks + long_closures == len(info["blinks"]) == 8
    lab = np.array(info["label"])
    pred = np.array([s == "closed" for s in states])
    tp, fp = int((pred & (lab == 1)).sum()), int((pred & (lab == 0)).sum())
    fn = int((~pred & (lab == 1)).sum())
    assert 2 * tp / (2 * tp + fp + fn) > 0.9


def test_demo_runs_headless_on_mock():
    r = subprocess.run([sys.executable, str(MODULE / "tools" / "demo.py"), "--sg1-jsonl",
                        str(REPO / "interfaces" / "sg2" / "mock" / "mock_sg1_alert.jsonl"), "--no-window"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert "blinks 16" in r.stderr and "agree 100.0%" in r.stderr
