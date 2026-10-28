"""Deterministic synthetic cue-stream generator with ground truth.

Writes an input JSONL (one record per captured frame) and a ground-truth CSV
with columns ``start_ms,end_ms,type`` where ``type`` is ``closure`` or
``yawn``. ``closure`` rows include short blinks as well as sustained closures;
the evaluator filters by duration.

Example::

    python mock/generate.py --scenario mixed --duration 120 --fps 30 --seed 1 \
        --out-jsonl in.jsonl --out-truth gt.csv --drop-prob 0.02 \
        --jitter-ms 3 --flicker-prob 0.01
"""
from __future__ import annotations

import argparse
import bisect
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sg4.io_utils import nest_packet, write_jsonl  # noqa: E402

SCENARIOS = ("normal", "drowsy", "yawning", "face_lost", "mixed")
Interval = Tuple[int, int]

# (gap_min_ms, gap_max_ms, dur_min_ms, dur_max_ms) per event kind and scenario
_PLAN: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {
    "normal": {"blink": (2000, 5000, 100, 350)},
    "drowsy": {"blink": (1500, 4000, 150, 450),
               "long": (6000, 12000, 600, 1500),
               "micro": (15000, 30000, 1500, 3500)},
    "yawning": {"blink": (2000, 5000, 100, 350),
                "yawn": (12000, 22000, 2000, 4000)},
    "face_lost": {"blink": (2000, 5000, 100, 350),
                  "lost": (6000, 12000, 1000, 4000)},
    "mixed": {"blink": (1500, 4000, 120, 400),
              "long": (10000, 20000, 600, 1500),
              "micro": (25000, 45000, 1500, 3000),
              "yawn": (15000, 30000, 2000, 4000),
              "lost": (10000, 20000, 1000, 3000)},
}
_MARGIN_MS = 200  # keep events apart so closures never merge
_TAIL_MS = 2000   # no events at the very end (detectors need time to react)


@dataclass
class MockOptions:
    """Generator parameters."""

    scenario: str = "mixed"
    duration_s: float = 120.0
    fps: float = 30.0
    seed: int = 0
    drop_prob: float = 0.0       # probability a captured frame is missing
    jitter_ms: int = 0           # +/- timestamp jitter
    flicker_prob: float = 0.0    # probability eyes_closed is flipped
    bad_ts_prob: float = 0.0     # probability of a repeated timestamp
    nested: bool = False         # write nested records instead of flat ones


def _overlaps(ev: Interval, others: Sequence[Interval]) -> bool:
    return any(ev[0] < o[1] + _MARGIN_MS and o[0] < ev[1] + _MARGIN_MS
               for o in others)


def _schedule(rng: random.Random, duration_ms: int,
              spec: Tuple[int, int, int, int],
              forbidden: Sequence[Interval]) -> List[Interval]:
    """Sequentially place events of one kind, skipping forbidden overlaps."""
    gap_lo, gap_hi, dur_lo, dur_hi = spec
    events: List[Interval] = []
    t = rng.randint(gap_lo, gap_hi) + 5000  # leave the start of the stream quiet
    while t < duration_ms - dur_hi:
        dur = rng.randint(dur_lo, dur_hi)
        # on a clash, slide the event later a few times instead of dropping it
        for _ in range(30):
            ev = (t, t + dur)
            if ev[1] > duration_ms - _TAIL_MS:
                return events  # keep the end of the stream free of events
            if not _overlaps(ev, forbidden):
                events.append(ev)
                break
            t += rng.randint(300, 1500)
        t += dur + rng.randint(gap_lo, gap_hi)
    return events


def build_timeline(opts: MockOptions) -> Dict[str, List[Interval]]:
    """Return event intervals: ``closure`` (all), ``yawn`` and ``lost``."""
    if opts.scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {opts.scenario!r}; use {SCENARIOS}")
    rng = random.Random(opts.seed)
    duration_ms = int(opts.duration_s * 1000)
    plan = _PLAN[opts.scenario]
    lost: List[Interval] = []
    if "lost" in plan:
        lost = _schedule(rng, duration_ms, plan["lost"], [])
    closures: List[Interval] = []
    yawns: List[Interval] = []
    for kind in ("micro", "long"):
        if kind in plan:
            closures += _schedule(rng, duration_ms, plan[kind], lost + closures)
    if "yawn" in plan:
        yawns = _schedule(rng, duration_ms, plan["yawn"], lost + closures)
    closures += _schedule(rng, duration_ms, plan["blink"], lost + closures + yawns)
    return {"closure": sorted(closures), "yawn": sorted(yawns), "lost": sorted(lost)}


def _inside(events: Sequence[Interval], starts: Sequence[int], t: int) -> bool:
    i = bisect.bisect_right(starts, t) - 1
    return i >= 0 and events[i][0] <= t < events[i][1]


def generate(opts: MockOptions) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int, str]]]:
    """Generate (records, ground_truth). Fully determined by ``opts``."""
    timeline = build_timeline(opts)
    rng = random.Random(opts.seed * 7919 + 13)  # separate stream for per-frame noise
    closures, yawns, lost = timeline["closure"], timeline["yawn"], timeline["lost"]
    c_starts = [e[0] for e in closures]
    y_starts = [e[0] for e in yawns]
    l_starts = [e[0] for e in lost]

    n_frames = int(opts.duration_s * opts.fps)
    records: List[Dict[str, Any]] = []
    prev_ts: Optional[int] = None
    for k in range(n_frames):
        t_ideal = int(round(k * 1000.0 / opts.fps))
        # draw all random numbers every frame so the stream is stable
        r_drop, r_flick, r_bad = rng.random(), rng.random(), rng.random()
        jitter = rng.randint(-opts.jitter_ms, opts.jitter_ms) if opts.jitter_ms else 0
        noise = [rng.random() for _ in range(5)]
        if r_drop < opts.drop_prob:
            continue

        ts = t_ideal + jitter
        if prev_ts is not None:
            ts = max(ts, prev_ts + 1)
            if r_bad < opts.bad_ts_prob:
                ts = prev_ts  # deliberately bad: repeated timestamp
        prev_ts = ts

        if _inside(lost, l_starts, t_ideal):
            rec: Dict[str, Any] = {
                "frame_id": k, "timestamp_ms": ts, "face_detected": False,
                "left_open_prob": None, "right_open_prob": None,
                "ear_left": None, "ear_right": None,
                "eyes_closed": None, "eyes_valid": False,
                "yawn_prob": None, "cue_flags": None, "mouth_valid": False,
            }
        else:
            closed = _inside(closures, c_starts, t_ideal)
            if r_flick < opts.flicker_prob:
                closed = not closed
            yawning = _inside(yawns, y_starts, t_ideal)
            if closed:
                left, right = 0.02 + 0.23 * noise[0], 0.02 + 0.23 * noise[1]
                ear = 0.05 + 0.10 * noise[2]
            else:
                left, right = 0.75 + 0.23 * noise[0], 0.75 + 0.23 * noise[1]
                ear = 0.25 + 0.10 * noise[2]
            yprob = 0.75 + 0.22 * noise[3] if yawning else 0.15 * noise[3]
            rec = {
                "frame_id": k, "timestamp_ms": ts, "face_detected": True,
                "left_open_prob": round(left, 4), "right_open_prob": round(right, 4),
                "ear_left": round(ear, 4), "ear_right": round(ear, 4),
                "eyes_closed": closed, "eyes_valid": True,
                "yawn_prob": round(yprob, 4), "cue_flags": [], "mouth_valid": True,
            }
        records.append(nest_packet(rec) if opts.nested else rec)

    truth = [(s, e, "closure") for s, e in closures] + [(s, e, "yawn") for s, e in yawns]
    truth.sort()
    return records, truth


def write_truth_csv(path: Any, truth: Sequence[Tuple[int, int, str]]) -> None:
    """Write the ground-truth CSV (``start_ms,end_ms,type``)."""
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["start_ms", "end_ms", "type"])
        writer.writerows(truth)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate a synthetic SG-4 input stream.")
    ap.add_argument("--scenario", choices=SCENARIOS, default="mixed")
    ap.add_argument("--duration", type=float, default=120.0, help="seconds")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop-prob", type=float, default=0.0)
    ap.add_argument("--jitter-ms", type=int, default=0)
    ap.add_argument("--flicker-prob", type=float, default=0.0)
    ap.add_argument("--bad-ts-prob", type=float, default=0.0)
    ap.add_argument("--nested", action="store_true", help="write nested records")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-truth", required=True)
    a = ap.parse_args(argv)
    opts = MockOptions(a.scenario, a.duration, a.fps, a.seed, a.drop_prob,
                       a.jitter_ms, a.flicker_prob, a.bad_ts_prob, a.nested)
    records, truth = generate(opts)
    write_jsonl(a.out_jsonl, records)
    write_truth_csv(a.out_truth, truth)
    print(f"wrote {len(records)} frames to {a.out_jsonl} and "
          f"{len(truth)} truth events to {a.out_truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
