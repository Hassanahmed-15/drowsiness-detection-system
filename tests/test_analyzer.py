"""Spec section 10 tests 1-12 (hand-computable expectations)."""
import copy

import pytest

from conftest import cfg, rec, run, stream
from mock.generate import MockOptions, generate
from sg4 import MEASUREMENT_KEYS, OUTPUT_KEYS, SG4Config, TemporalAnalyzer

STEP = 50
TOL = STEP + 1


def test_01_all_open_10s():
    out = run(cfg(), stream(STEP, [(10000, "open")]))[-1]
    assert out["valid"] is True
    assert out["perclos"] == 0.0
    assert out["longest_closure_ms"] == 0.0
    assert out["current_closure_ms"] == 0.0
    assert out["blink_rate_per_min"] == 0.0
    assert out["mean_blink_duration_ms"] is None
    assert out["yawn_count"] == 0


def test_02_continuous_2s_closure():
    out = run(cfg(), stream(STEP, [(1000, "open"), (2000, "closed")]))[-1]
    assert out["current_closure_ms"] == pytest.approx(2000, abs=TOL)
    assert out["longest_closure_ms"] == pytest.approx(2000, abs=TOL)


def test_03_half_closed_no_blinks():
    outs = run(cfg(), stream(STEP, [(1000, "open"), (1000, "closed")] * 10))
    out = outs[-1]
    assert out["perclos"] == pytest.approx(0.5, abs=0.02)
    assert out["blink_rate_per_min"] == 0.0
    assert out["mean_blink_duration_ms"] is None


def test_04_blinks():
    segs = [(1000, "open")]
    for _ in range(10):
        segs += [(200, "closed"), (2800, "open")]
    out = run(cfg(), stream(STEP, segs))[-1]
    # 4-sample closure every 3 s for 30 s -> ~10 blinks, 200 ms, ~20/min
    assert out["mean_blink_duration_ms"] == pytest.approx(200, abs=TOL)
    assert out["blink_rate_per_min"] == pytest.approx(20, abs=2.5)


def test_05_face_lost_40_percent():
    cycle = [(6000, "open"), (4000, "lost")]
    records = stream(STEP, cycle * 3)
    low = run(cfg(min_valid_fraction=0.5), records)[-1]
    assert low["window_valid_fraction"] == pytest.approx(0.6, abs=0.03)
    assert low["valid"] is True
    high = run(cfg(min_valid_fraction=0.7), records)[-1]
    assert high["valid"] is False
    assert isinstance(high["window_valid_fraction"], float)
    assert all(high[k] is None for k in MEASUREMENT_KEYS)


def test_06_warmup():
    records = stream(STEP, [(8000, "open")])
    outs = run(cfg(warmup_ms=5000), records)
    for o in outs:
        if o["timestamp_ms"] < 5000:
            assert o["valid"] is False
            assert all(o[k] is None for k in MEASUREMENT_KEYS)
            assert isinstance(o["window_valid_fraction"], float)
    assert outs[-1]["valid"] is True


def test_07_gap_bridging():
    def build(gap_ms):
        return stream(STEP, [(1000, "open"), (500, "closed"),
                             (gap_ms, "eyes_invalid"), (500, "closed"),
                             (1000, "open")])
    short = TemporalAnalyzer(cfg())
    outs = [short.update(r) for r in build(150)]
    assert len(short.closure.completed) == 1
    assert outs[-1]["longest_closure_ms"] == pytest.approx(1150, abs=TOL)

    long_ = TemporalAnalyzer(cfg())
    outs = [long_.update(r) for r in build(400)]
    assert len(long_.closure.completed) == 2
    assert outs[-1]["longest_closure_ms"] == pytest.approx(500, abs=TOL)


def test_08_yawn():
    def yawn_count(bursts):
        segs = [(3000, "open", 0.0)]
        for i, (dur, gap) in enumerate(bursts):
            segs += [(dur, "open", 0.9), (gap, "open", 0.0)]
        return run(cfg(), stream(STEP, segs))[-1]["yawn_count"]

    assert yawn_count([(2000, 3000)]) == 1
    assert yawn_count([(300, 3000)]) == 0
    # second burst begins inside the 3 s refractory period -> still 1
    assert yawn_count([(2000, 500), (2000, 3000)]) == 1
    # well separated bursts -> 2
    assert yawn_count([(2000, 5000), (2000, 3000)]) == 2


def test_09_window_expiry():
    c = cfg(window_ms=10000)
    records = stream(STEP, [(1000, "open"), (2000, "closed"), (17000, "open")])
    outs = run(c, records)
    assert outs[-1]["longest_closure_ms"] == 0.0
    assert outs[-1]["perclos"] == 0.0
    # still visible while inside the window
    mid = [o for o in outs if o["timestamp_ms"] == 9000][0]
    assert mid["longest_closure_ms"] == pytest.approx(2000, abs=TOL)


def test_10_determinism_and_reset():
    records, _ = generate(MockOptions("mixed", 90, 30, seed=3, drop_prob=0.03,
                                      jitter_ms=4, flicker_prob=0.02))
    first = run(cfg(), records)
    second = run(cfg(), records)
    assert first == second
    analyzer = TemporalAnalyzer(cfg())
    for r in records[: len(records) // 2]:
        analyzer.update(r)
    analyzer.reset()
    assert analyzer.dropped_samples == 0
    assert [analyzer.update(r) for r in records] == first


def test_11_bad_timestamps():
    a = TemporalAnalyzer(cfg())
    a.update(rec(0, 0))
    a.update(rec(50, 1))
    before = a.update(rec(100, 2))
    same = a.update(rec(100, 3))            # repeated
    back = a.update(rec(40, 4))             # decreasing
    nots = a.update({"frame_id": 5})        # no timestamp
    junk = a.update("not a record")         # type: ignore[arg-type]
    assert a.dropped_samples == 4
    assert same == back == nots == junk == before
    after = a.update(rec(150, 6))
    assert after["timestamp_ms"] == 150 and a.dropped_samples == 4


def test_11b_first_record_bad_does_not_crash():
    a = TemporalAnalyzer(cfg())
    out = a.update({"frame_id": 0})
    assert out["valid"] is False and a.dropped_samples == 1
    assert isinstance(out["window_valid_fraction"], float)


def _check_contract(o):
    assert tuple(o.keys()) == OUTPUT_KEYS
    assert type(o["frame_id"]) is int and type(o["timestamp_ms"]) is int
    assert type(o["window_ms"]) is int
    assert type(o["valid"]) is bool
    assert type(o["window_valid_fraction"]) is float
    if not o["valid"]:
        assert all(o[k] is None for k in MEASUREMENT_KEYS)
    else:
        for k in MEASUREMENT_KEYS:
            if k == "mean_blink_duration_ms" and o[k] is None:
                continue
            assert o[k] is not None, k
        for k in ("perclos", "blink_rate_per_min", "longest_closure_ms",
                  "current_closure_ms"):
            assert type(o[k]) is float
        assert type(o["yawn_count"]) is int
        assert 0.0 <= o["perclos"] <= 1.0


def test_12_output_contract():
    records, _ = generate(MockOptions("mixed", 120, 30, seed=5, drop_prob=0.05,
                                      flicker_prob=0.02))
    seen_valid = seen_invalid = False
    for o in run(cfg(warmup_ms=5000), records):
        _check_contract(o)
        seen_valid |= o["valid"]
        seen_invalid |= not o["valid"]
    assert seen_valid and seen_invalid


def test_bad_values_never_crash():
    a = TemporalAnalyzer(cfg())
    weird = [
        {"timestamp_ms": 0, "face_detected": 1, "eyes_closed": "yes"},
        {"timestamp_ms": 50, "face_detected": True, "eyes_valid": True,
         "eyes_closed": float("nan"), "yawn_prob": float("nan"), "mouth_valid": True},
        {"timestamp_ms": 100, "face_detected": True, "eyes_valid": True,
         "eyes_closed": False, "yawn_prob": 7.0, "mouth_valid": True},
        {"timestamp_ms": 150.0, "face_detected": None},
    ]
    for r in weird:
        _check_contract(a.update(copy.deepcopy(r)))


def test_diagnostic_mouth_valid_fraction():
    a = TemporalAnalyzer(cfg())
    for r in stream(STEP, [(2000, "open")]):
        a.update(r)
    assert a.mouth_valid_fraction == pytest.approx(1.0, abs=0.03)
    b = TemporalAnalyzer(cfg())
    for r in stream(STEP, [(1000, "open"), (1000, "lost")]):
        b.update(r)
    assert b.mouth_valid_fraction == pytest.approx(0.5, abs=0.05)


def test_config_validation_and_roundtrip(tmp_path):
    c = SG4Config(window_ms=12345)
    p = tmp_path / "c.json"
    c.save(p)
    assert SG4Config.load(p) == c
    with pytest.raises(ValueError):
        SG4Config.from_dict({"window_msec": 1})
    with pytest.raises(ValueError):
        SG4Config(yawn_on_threshold=0.3, yawn_off_threshold=0.5)
    with pytest.raises(ValueError):
        SG4Config(t_blink_min_ms=600, t_blink_max_ms=500)
