"""Spec section 10 test 13 plus JSONL helpers."""
from conftest import cfg, rec, run
from mock.generate import MockOptions, generate
from sg4 import flatten_packet, nest_packet, read_jsonl, write_jsonl
from sg4.io_utils import FLAT_KEYS


def test_13_flatten_matches_flat():
    flat = rec(1234, 7, "closed", yawn=0.4)
    nested = nest_packet(flat)
    assert "face" in nested and "eyes" in nested and "mouth" in nested
    assert flatten_packet(nested) == flat
    assert set(flatten_packet(nested)) == set(FLAT_KEYS)


def test_13_nested_stream_gives_same_outputs():
    flat, _ = generate(MockOptions("mixed", 60, 30, seed=2))
    nested = [nest_packet(r) for r in flat]
    assert run(cfg(), flat) == run(cfg(), nested)
    assert run(cfg(), flat) == run(cfg(), [flatten_packet(n) for n in nested])


def test_flatten_missing_blocks():
    out = flatten_packet({"frame_id": 1, "timestamp_ms": 5})
    assert out["face_detected"] is False
    assert out["eyes_valid"] is False and out["mouth_valid"] is False
    assert out["eyes_closed"] is None and out["yawn_prob"] is None


def test_jsonl_roundtrip_and_bad_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    records = [rec(0, 0), rec(50, 1, "closed")]
    assert write_jsonl(p, records) == 2
    assert list(read_jsonl(p)) == records
    p.write_text('{"a": 1}\nnot json\n\n[1,2]\n{"b": 2}\n', encoding="utf-8")
    errors = []
    got = list(read_jsonl(p, on_error=lambda n, l, e: errors.append(n)))
    assert got == [{"a": 1}, {"b": 2}] and errors == [2, 4]
