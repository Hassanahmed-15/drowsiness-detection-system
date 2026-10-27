"""Tests for mock generator, replay, evaluate and sweep."""
import csv
import json

import pytest

from conftest import cfg, run
from evaluate import (derive_closure_runs, derive_yawn_detections, evaluate,
                      load_truth, match_by_iou, stability)
from mock.generate import SCENARIOS, MockOptions, generate, main as gen_main
from sg4 import SG4Config, read_jsonl
import replay
import sweep


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_generator_deterministic_and_has_truth(scenario):
    opts = MockOptions(scenario, 90, 30, seed=4, drop_prob=0.02, jitter_ms=3,
                       flicker_prob=0.01)
    a, ta = generate(opts)
    b, tb = generate(opts)
    assert a == b and ta == tb
    ts = [r["timestamp_ms"] for r in a]
    assert ts == sorted(ts) and len(set(ts)) == len(ts)
    assert any(kind == "closure" for _, _, kind in ta)
    if scenario in ("yawning", "mixed"):
        assert any(kind == "yawn" for _, _, kind in ta)
    if scenario in ("face_lost", "mixed"):
        assert any(not r["face_detected"] for r in a)


def test_generator_different_seed_differs_and_fps():
    a, _ = generate(MockOptions("drowsy", 60, 30, seed=1))
    b, _ = generate(MockOptions("drowsy", 60, 30, seed=2))
    c, _ = generate(MockOptions("drowsy", 60, 15, seed=1))
    assert a != b and len(c) == len(a) // 2


def test_generator_bad_timestamps_and_cli(tmp_path):
    recs, _ = generate(MockOptions("normal", 30, 30, seed=1, bad_ts_prob=0.1))
    ts = [r["timestamp_ms"] for r in recs]
    assert len(set(ts)) < len(ts)  # repeated timestamps were injected
    gen_main(["--scenario", "mixed", "--duration", "40", "--seed", "1",
              "--out-jsonl", str(tmp_path / "i.jsonl"),
              "--out-truth", str(tmp_path / "g.csv")])
    with open(tmp_path / "g.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["start_ms", "end_ms", "type"]


def test_end_to_end_clean_stream_scores_well():
    recs, truth_rows = generate(MockOptions("mixed", 180, 30, seed=1))
    truth = {"closure": [], "yawn": []}
    for s, e, k in truth_rows:
        truth[k].append((s, e))
    m = evaluate(run(cfg(warmup_ms=5000), recs), truth)
    assert m["sustained_closure"]["recall"] >= 0.9
    assert m["sustained_closure"]["precision"] >= 0.9
    assert m["yawn"]["recall"] >= 0.9
    assert m["detection_delay"]["missed"] == 0


def test_metric_helpers():
    assert match_by_iou([(0, 1000)], [(100, 1100)], 0.3) == 1
    assert match_by_iou([(0, 1000)], [(2000, 3000)], 0.3) == 0
    outs = [{"timestamp_ms": t, "current_closure_ms": c}
            for t, c in [(0, 0.0), (50, 50.0), (100, 100.0), (150, 0.0), (200, 0.0)]]
    assert derive_closure_runs(outs) == [(0, 150)]
    yo = [{"timestamp_ms": t, "yawn_count": c} for t, c in
          [(0, None), (10, 1), (20, 1), (30, 2), (40, 1)]]
    assert derive_yawn_detections(yo) == [30]
    so = [{"timestamp_ms": i * 1000, "perclos": v} for i, v in
          enumerate([0.0, 0.2, 0.0, None, 0.3])]
    assert stability(so, "perclos", 0.15)["flips"] == 3


def test_replay_cli_flat_nested_and_bad_lines(tmp_path, capsys):
    from sg4 import nest_packet
    recs, _ = generate(MockOptions("mixed", 30, 30, seed=7))
    lines = [json.dumps(r if i % 2 else nest_packet(r)) for i, r in enumerate(recs)]
    lines.insert(5, "this is not json")
    src, dst, cfg_path = tmp_path / "in.jsonl", tmp_path / "out.jsonl", tmp_path / "c.json"
    src.write_text("\n".join(lines) + "\n")
    SG4Config(warmup_ms=1000).save(cfg_path)
    assert replay.main(["--config", str(cfg_path), "--input", str(src),
                        "--output", str(dst)]) == 0
    out = list(read_jsonl(dst))
    assert len(out) == len(recs)
    assert out == run(SG4Config(warmup_ms=1000), recs)
    assert "malformed_lines=1" in capsys.readouterr().err


def test_sweep_grid_and_invalid_combo(tmp_path):
    recs, truth_rows = generate(MockOptions("mixed", 90, 30, seed=2))
    truth = {"closure": [], "yawn": []}
    for s, e, k in truth_rows:
        truth[k].append((s, e))
    grid = {"window_ms": [10000, 30000], "yawn_on_threshold": [0.5, 0.9]}
    rows = sweep.sweep(recs, truth, grid, SG4Config())
    assert len(rows) == 4 and all(r["error"] == "" for r in rows)
    bad = sweep.sweep(recs, truth, {"yawn_off_threshold": [0.2, 0.9]}, SG4Config())
    assert bad[0]["error"] == "" and "yawn_off_threshold" in bad[1]["error"]
    p = tmp_path / "r.csv"
    sweep.write_results(p, rows, list(grid))
    with open(p, newline="") as fh:
        got = list(csv.DictReader(fh))
    assert len(got) == 4 and "stability_flips" in got[0] and "delay_mean_excess_ms" in got[0]
    assert sweep.parse_params(["a=1,2.5"]) == {"a": [1, 2.5]}
