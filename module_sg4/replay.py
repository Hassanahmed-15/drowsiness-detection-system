#!/usr/bin/env python3
"""Replay a JSONL stream through SG-4.

    python replay.py --config cfg.json --input in.jsonl --output out.jsonl

Each input line is a flat or a nested record. Each output line is the
``temporal`` dict for that input line (a dropped record repeats the previous
output; see README). ``--config`` is optional (defaults are used otherwise).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sg4 import SG4Config, TemporalAnalyzer  # noqa: E402
from sg4.io_utils import normalize_record, read_jsonl, write_jsonl  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Replay a JSONL cue stream through SG-4.")
    ap.add_argument("--config", help="SG4Config JSON/YAML file (optional)")
    ap.add_argument("--input", required=True, help="input JSONL (flat or nested)")
    ap.add_argument("--output", required=True, help="output JSONL")
    ap.add_argument("--quiet", action="store_true", help="no summary on stderr")
    args = ap.parse_args(argv)

    config = SG4Config.load(args.config) if args.config else SG4Config()
    analyzer = TemporalAnalyzer(config)
    bad_lines = []

    def on_error(lineno: int, _line: str, exc: Exception) -> None:
        bad_lines.append(lineno)
        print(f"warning: skipped malformed line {lineno}: {exc}", file=sys.stderr)

    outputs = (analyzer.update(normalize_record(r))
               for r in read_jsonl(args.input, on_error=on_error))
    n = write_jsonl(args.output, outputs)
    if not args.quiet:
        print(f"processed {n} records -> {args.output}; "
              f"dropped_samples={analyzer.dropped_samples}, "
              f"malformed_lines={len(bad_lines)}, "
              f"mouth_valid_fraction={analyzer.mouth_valid_fraction:.3f}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
