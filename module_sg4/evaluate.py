#!/usr/bin/env python3
"""Evaluate SG-4 outputs against ground truth.

    python evaluate.py --output out.jsonl --truth gt.csv [--plot fig.png]

Metrics
-------
(a) Event precision/recall.
    * Sustained closures: predicted runs are reconstructed from
      ``current_closure_ms`` (start = timestamp - current_closure_ms). Truth and
      prediction are kept if they last >= ``--closure-min-ms`` and matched
      one-to-one by temporal IoU >= ``--iou``.
    * Yawns: a detection is a rise of ``yawn_count`` between consecutive valid
      outputs; it matches a truth yawn if it falls inside
      [start - tol, end + tol] (``--yawn-tol-ms``).
    Truth events that end before the first ``valid`` output are excluded
    (unobservable during warm-up) and reported.
(b) Detection delay: for each truth closure >= ``--delay-level-ms``, the time
    from its true start until the chosen field (``--delay-field``) first
    reaches that level (closures shorter than level + ``--delay-margin-ms`` are
    skipped as borderline). ``delay_ms`` includes the level itself (a closure
    cannot be seen above L before it has lasted L); ``excess_ms`` = delay - L.
(c) Stability: number of flips of ``(field >= threshold)`` over valid outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sg4.io_utils import read_jsonl  # noqa: E402

Event = Tuple[int, int]


# ------------------------------------------------------------------- loading
def load_truth(path: Any) -> Dict[str, List[Event]]:
    """Read the ground-truth CSV into {'closure': [...], 'yawn': [...]}."""
    truth: Dict[str, List[Event]] = {"closure": [], "yawn": []}
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            kind = row["type"].strip()
            if kind not in truth:
                raise ValueError(f"unknown truth type {kind!r}")
            truth[kind].append((int(float(row["start_ms"])), int(float(row["end_ms"]))))
    for events in truth.values():
        events.sort()
    return truth


def load_outputs(path: Any) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


# ------------------------------------------------------------------ matching
def _iou(a: Event, b: Event) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match_by_iou(truth: Sequence[Event], pred: Sequence[Event],
                 iou_threshold: float) -> int:
    """One-to-one greedy matching by descending IoU; returns #true positives."""
    pairs = sorted(((_iou(t, p), i, j) for i, t in enumerate(truth)
                    for j, p in enumerate(pred)), reverse=True)
    used_t, used_p, tp = set(), set(), 0
    for score, i, j in pairs:
        if score < iou_threshold or score <= 0:
            break
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        tp += 1
    return tp


def _prf(n_truth: int, n_pred: int, tp: int) -> Dict[str, Any]:
    precision = tp / n_pred if n_pred else None
    recall = tp / n_truth if n_truth else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else (0.0 if precision is not None and recall is not None else None))
    return {"n_truth": n_truth, "n_pred": n_pred, "tp": tp,
            "precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------- derivations
def derive_closure_runs(outputs: Sequence[Dict[str, Any]]) -> List[Event]:
    """Reconstruct closure runs (start, end) from ``current_closure_ms``."""
    runs: List[Event] = []
    start: Optional[int] = None
    last_ts: Optional[int] = None
    for o in outputs:
        cur, ts = o.get("current_closure_ms"), o["timestamp_ms"]
        if cur is not None and cur > 0:
            if start is None:
                start = int(ts - cur)
            last_ts = ts
        elif start is not None:
            runs.append((start, ts if cur is not None else int(last_ts or ts)))
            start = None
    if start is not None and last_ts is not None:
        runs.append((start, last_ts))
    return runs


def derive_yawn_detections(outputs: Sequence[Dict[str, Any]]) -> List[int]:
    """Timestamps where ``yawn_count`` rose between consecutive valid outputs."""
    times: List[int] = []
    prev: Optional[int] = None
    for o in outputs:
        count = o.get("yawn_count")
        if count is None:
            prev = None
            continue
        if prev is not None and count > prev:
            times.extend([o["timestamp_ms"]] * (count - prev))
        prev = count
    return times


def _field_value(o: Dict[str, Any], field: str) -> Optional[float]:
    if field == "either":
        vals = [o.get("current_closure_ms"), o.get("longest_closure_ms")]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None
    return o.get(f"{field}_closure_ms")


def detection_delays(outputs: Sequence[Dict[str, Any]], closures: Sequence[Event],
                     level_ms: float, field: str, search_ms: int,
                     margin_ms: float = 100) -> Dict[str, Any]:
    """Delay from true closure start until ``field`` reaches ``level_ms``.

    Closures shorter than ``level_ms + margin_ms`` are skipped: a closure that
    barely reaches the level can end before one more frame confirms it, which
    would count as a spurious miss.
    """
    if field not in ("current", "longest", "either"):
        raise ValueError("delay field must be current, longest or either")
    delays: List[float] = []
    missed = unobservable = 0
    for s, e in closures:
        if e - s < level_ms + margin_ms:
            continue
        before = [o for o in outputs if o["timestamp_ms"] < s]
        prior = _field_value(before[-1], field) if before else None
        if field != "current" and prior is not None and prior >= level_ms:
            unobservable += 1  # an older closure already sits above the level
            continue
        hit = None
        for o in outputs:
            ts = o["timestamp_ms"]
            if ts < s:
                continue
            if ts > e + search_ms:
                break
            v = _field_value(o, field)
            if v is not None and v >= level_ms:
                hit = ts
                break
        if hit is None:
            missed += 1
        else:
            delays.append(hit - s)
    summary: Dict[str, Any] = {
        "level_ms": level_ms, "margin_ms": margin_ms, "field": field, "n_events": len(delays) + missed,
        "detected": len(delays), "missed": missed, "unobservable": unobservable,
        "mean_ms": None, "median_ms": None, "max_ms": None, "mean_excess_ms": None,
    }
    if delays:
        summary.update(mean_ms=statistics.fmean(delays),
                       median_ms=statistics.median(delays), max_ms=max(delays),
                       mean_excess_ms=statistics.fmean(delays) - level_ms)
    return summary


def stability(outputs: Sequence[Dict[str, Any]], field: str,
              threshold: float) -> Dict[str, Any]:
    """Number of times ``field >= threshold`` flips over valid outputs."""
    flips, prev_state = 0, None
    first_ts = last_ts = None
    for o in outputs:
        v = o.get(field)
        if v is None:
            continue
        state = v >= threshold
        if prev_state is not None and state != prev_state:
            flips += 1
        prev_state = state
        first_ts = o["timestamp_ms"] if first_ts is None else first_ts
        last_ts = o["timestamp_ms"]
    span_min = ((last_ts - first_ts) / 60000.0) if first_ts is not None else 0.0
    return {"field": field, "threshold": threshold, "flips": flips,
            "flips_per_min": (flips / span_min) if span_min > 0 else None}


# ----------------------------------------------------------------- top level
def evaluate(outputs: Sequence[Dict[str, Any]], truth: Dict[str, List[Event]], *,
             closure_min_ms: int = 500, iou_threshold: float = 0.3,
             yawn_tol_ms: int = 1000, delay_level_ms: float = 1000,
             delay_field: str = "current", delay_search_ms: int = 1000, delay_margin_ms: float = 100,
             stability_field: str = "perclos",
             stability_threshold: float = 0.15) -> Dict[str, Any]:
    """Compute all metrics; returns a JSON-serialisable dict."""
    first_valid = next((o["timestamp_ms"] for o in outputs if o.get("valid")), None)
    n_valid = sum(1 for o in outputs if o.get("valid"))

    def observable(events: Sequence[Event]) -> Tuple[List[Event], int]:
        if first_valid is None:
            return [], len(events)
        kept = [ev for ev in events if ev[1] >= first_valid]
        return kept, len(events) - len(kept)

    closures, excl_c = observable(truth["closure"])
    yawns, excl_y = observable(truth["yawn"])

    sustained_truth = [ev for ev in closures if ev[1] - ev[0] >= closure_min_ms]
    pred_runs = [r for r in derive_closure_runs(outputs) if r[1] - r[0] >= closure_min_ms]
    closure_tp = match_by_iou(sustained_truth, pred_runs, iou_threshold)

    detections = derive_yawn_detections(outputs)
    used, yawn_tp = set(), 0
    for d in sorted(detections):
        for i, (s, e) in enumerate(yawns):
            if i not in used and s - yawn_tol_ms <= d <= e + yawn_tol_ms:
                used.add(i)
                yawn_tp += 1
                break

    return {
        "n_outputs": len(outputs),
        "valid_output_fraction": (n_valid / len(outputs)) if outputs else 0.0,
        "first_valid_ms": first_valid,
        "excluded_warmup": {"closure": excl_c, "yawn": excl_y},
        "sustained_closure": {**_prf(len(sustained_truth), len(pred_runs), closure_tp),
                              "min_ms": closure_min_ms, "iou": iou_threshold},
        "yawn": {**_prf(len(yawns), len(detections), yawn_tp), "tol_ms": yawn_tol_ms},
        "detection_delay": detection_delays(outputs, closures, delay_level_ms,
                                            delay_field, delay_search_ms,
                                            delay_margin_ms),
        "stability": stability(outputs, stability_field, stability_threshold),
    }


def plot_outputs(outputs: Sequence[Dict[str, Any]], truth: Dict[str, List[Event]],
                 path: Any, closure_min_ms: int = 500) -> None:
    """Plot outputs vs time with truth events shaded (needs matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for --plot") from exc

    def series(key: str) -> Tuple[List[float], List[float]]:
        pts = [(o["timestamp_ms"] / 1000.0, o[key]) for o in outputs
               if o.get(key) is not None]
        return [p[0] for p in pts], [p[1] for p in pts]

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, keys, label in ((axes[0], ["perclos"], "PERCLOS"),
                            (axes[1], ["current_closure_ms", "longest_closure_ms"], "closure (ms)"),
                            (axes[2], ["yawn_count"], "yawn count")):
        for key in keys:
            x, y = series(key)
            ax.plot(x, y, label=key, linewidth=1)
        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=8)
        for s, e in truth["closure"]:
            if e - s >= closure_min_ms:
                ax.axvspan(s / 1000.0, e / 1000.0, color="tab:red", alpha=0.25)
        for s, e in truth["yawn"]:
            ax.axvspan(s / 1000.0, e / 1000.0, color="tab:orange", alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("SG-4 outputs (red = sustained closure, orange = yawn)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def add_eval_args(ap: argparse.ArgumentParser) -> None:
    """Metric options shared with sweep.py."""
    ap.add_argument("--closure-min-ms", type=int, default=500)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--yawn-tol-ms", type=int, default=1000)
    ap.add_argument("--delay-level-ms", type=float, default=1000)
    ap.add_argument("--delay-field", choices=("current", "longest", "either"),
                    default="current")
    ap.add_argument("--delay-search-ms", type=int, default=1000)
    ap.add_argument("--delay-margin-ms", type=float, default=100,
                    help="skip closures shorter than level + margin")
    ap.add_argument("--stability-field", default="perclos")
    ap.add_argument("--stability-threshold", type=float, default=0.15)


def eval_kwargs(a: argparse.Namespace) -> Dict[str, Any]:
    return dict(closure_min_ms=a.closure_min_ms, iou_threshold=a.iou,
                yawn_tol_ms=a.yawn_tol_ms, delay_level_ms=a.delay_level_ms,
                delay_field=a.delay_field, delay_search_ms=a.delay_search_ms,
                delay_margin_ms=a.delay_margin_ms,
                stability_field=a.stability_field,
                stability_threshold=a.stability_threshold)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate SG-4 output against ground truth.")
    ap.add_argument("--output", required=True, help="SG-4 output JSONL")
    ap.add_argument("--truth", required=True, help="ground-truth CSV")
    ap.add_argument("--json-out", help="also write metrics to this JSON file")
    ap.add_argument("--plot", help="save a Matplotlib figure to this path")
    add_eval_args(ap)
    a = ap.parse_args(argv)
    outputs, truth = load_outputs(a.output), load_truth(a.truth)
    metrics = evaluate(outputs, truth, **eval_kwargs(a))
    text = json.dumps(metrics, indent=2)
    print(text)
    if a.json_out:
        Path(a.json_out).write_text(text + "\n", encoding="utf-8")
    if a.plot:
        plot_outputs(outputs, truth, a.plot, a.closure_min_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
