#!/usr/bin/env python3
"""Parameter sweep: run SG-4 for every config in a grid and score each run.

    python sweep.py --input in.jsonl --truth gt.csv \
        --param window_ms=10000,30000 --param t_blink_max_ms=400,500 \
        --param yawn_on_threshold=0.5,0.6 --output results.csv

or give a JSON grid file: ``--grid grid.json`` with
``{"window_ms": [10000, 30000], "yawn_on_threshold": [0.5, 0.6]}``.
``--base-config`` supplies the values of every parameter not in the grid.
Invalid combinations (e.g. off threshold above on threshold) are written with
an ``error`` message and empty metrics instead of aborting the sweep.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import add_eval_args, eval_kwargs, evaluate, load_truth  # noqa: E402
from sg4 import SG4Config, run_stream  # noqa: E402
from sg4.io_utils import normalize_record, read_jsonl  # noqa: E402

METRIC_COLUMNS = (
    "closure_precision", "closure_recall", "closure_f1",
    "yawn_precision", "yawn_recall", "yawn_f1",
    "delay_detected", "delay_missed", "delay_mean_ms", "delay_mean_excess_ms",
    "stability_flips", "stability_flips_per_min", "valid_output_fraction",
)


def parse_params(items: Sequence[str]) -> Dict[str, List[Any]]:
    """Parse ``key=v1,v2`` strings; each value is read as JSON (int/float)."""
    grid: Dict[str, List[Any]] = {}
    for item in items:
        key, _, values = item.partition("=")
        if not key or not values:
            raise ValueError(f"bad --param {item!r}; expected key=v1,v2,...")
        grid[key.strip()] = [json.loads(v) for v in values.split(",")]
    return grid


def flatten_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "closure_precision": m["sustained_closure"]["precision"],
        "closure_recall": m["sustained_closure"]["recall"],
        "closure_f1": m["sustained_closure"]["f1"],
        "yawn_precision": m["yawn"]["precision"],
        "yawn_recall": m["yawn"]["recall"],
        "yawn_f1": m["yawn"]["f1"],
        "delay_detected": m["detection_delay"]["detected"],
        "delay_missed": m["detection_delay"]["missed"],
        "delay_mean_ms": m["detection_delay"]["mean_ms"],
        "delay_mean_excess_ms": m["detection_delay"]["mean_excess_ms"],
        "stability_flips": m["stability"]["flips"],
        "stability_flips_per_min": m["stability"]["flips_per_min"],
        "valid_output_fraction": m["valid_output_fraction"],
    }


def sweep(records: Sequence[Dict[str, Any]], truth: Dict[str, List[Any]],
          grid: Dict[str, List[Any]], base: SG4Config,
          **eval_opts: Any) -> List[Dict[str, Any]]:
    """Run every combination in ``grid``; returns one result row per config."""
    keys = list(grid)
    rows: List[Dict[str, Any]] = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        row: Dict[str, Any] = dict(params)
        try:
            config = base.with_overrides(**params)
        except ValueError as exc:
            row["error"] = str(exc)
            rows.append(row)
            continue
        outputs = run_stream(config, records)
        row.update(flatten_metrics(evaluate(outputs, truth, **eval_opts)))
        row["error"] = ""
        rows.append(row)
    return rows


def write_results(path: Any, rows: Sequence[Dict[str, Any]],
                  param_keys: Sequence[str]) -> None:
    columns = list(param_keys) + list(METRIC_COLUMNS) + ["error"]
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if row.get(c) is None else row.get(c))
                             for c in columns})


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Sweep SG-4 parameters against ground truth.")
    ap.add_argument("--input", required=True, help="input JSONL (flat or nested)")
    ap.add_argument("--truth", required=True, help="ground-truth CSV")
    ap.add_argument("--output", required=True, help="results CSV")
    ap.add_argument("--base-config", help="base SG4Config JSON/YAML")
    ap.add_argument("--grid", help="JSON file {param: [values]}")
    ap.add_argument("--param", action="append", default=[],
                    help="key=v1,v2,... (repeatable)")
    add_eval_args(ap)
    a = ap.parse_args(argv)

    grid: Dict[str, List[Any]] = {}
    if a.grid:
        grid.update(json.loads(Path(a.grid).read_text(encoding="utf-8")))
    grid.update(parse_params(a.param))
    if not grid:
        ap.error("give --grid and/or at least one --param")

    base = SG4Config.load(a.base_config) if a.base_config else SG4Config()
    records = [normalize_record(r) for r in read_jsonl(a.input)]
    rows = sweep(records, load_truth(a.truth), grid, base, **eval_kwargs(a))
    write_results(a.output, rows, list(grid))
    print(f"{len(rows)} configurations -> {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
