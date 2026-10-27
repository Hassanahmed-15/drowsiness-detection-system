# SG-4 Temporal Behaviour Analysis

Stateful, deterministic, time-based processor for the driver-drowsiness
pipeline (CS-477, SEECS CV Face-Off 2026). It turns per-frame eye and mouth
cues into sliding-window indicators: PERCLOS, blink rate and duration, longest
and current eye closure, and yawn count. No image processing, no deep learning.
Python 3.10, standard library only in the core. It runs with **no upstream
module present**: use the mock generator or recorded JSONL.

## Layout

```
config/sg4_default.json   placeholder thresholds
sg4/config.py             SG4Config (validated dataclass, JSON/YAML load/save)
sg4/analyzer.py           TemporalAnalyzer: update(), reset(), run_stream()
sg4/closure.py            closure-run state machine + blink classification
sg4/yawn.py               yawn tracker
sg4/io_utils.py           flatten_packet, nest_packet, JSONL read/write
mock/generate.py          synthetic stream + ground-truth CSV
replay.py  evaluate.py  sweep.py
tests/                    pytest suite
```

## Use

```python
from sg4 import SG4Config, TemporalAnalyzer
analyzer = TemporalAnalyzer(SG4Config.load("config/sg4_default.json"))
out = analyzer.update(record)      # flat dict (or nested: auto-flattened)
analyzer.reset()                   # new session
analyzer.dropped_samples           # bad-timestamp counter
analyzer.mouth_valid_fraction      # diagnostic: low value => yawns may be undercounted
```

Input and output fields are exactly those in the spec (sections 5 and 6).
Only `timestamp_ms` is used for time; the wall clock is never read.

## Run

```bash
pip install pytest            # tests; numpy/matplotlib optional (plots)
python -m pytest -q tests

python mock/generate.py --scenario mixed --duration 180 --fps 30 --seed 1 \
    --drop-prob 0.02 --jitter-ms 3 --flicker-prob 0.01 \
    --out-jsonl in.jsonl --out-truth gt.csv
python replay.py   --config config/sg4_default.json --input in.jsonl --output out.jsonl
python evaluate.py --output out.jsonl --truth gt.csv --plot fig.png
python sweep.py    --input in.jsonl --truth gt.csv --output results.csv \
    --param window_ms=10000,30000 --param t_blink_max_ms=400,500 \
    --param yawn_on_threshold=0.5,0.6,0.7
```

Scenarios: `normal`, `drowsy`, `yawning`, `face_lost`, `mixed`. Options: `--fps`,
`--drop-prob` (missing frames), `--jitter-ms`, `--flicker-prob` (flips
`eyes_closed`), `--bad-ts-prob` (repeated timestamps), `--nested`.
Ground truth `closure` rows include short blinks; evaluation keeps only those
>= `--closure-min-ms` (default 500) as "sustained".

## Evaluation (evaluate.py)

* **Precision/recall**: sustained closures are reconstructed from
  `current_closure_ms` and matched one-to-one by temporal IoU (default 0.3).
  Yawns count when `yawn_count` rises while the true yawn is in progress
  (+/- `--yawn-tol-ms`). Events ending before the first `valid` output are
  excluded and reported.
* **Detection delay**: true start -> first output where `current_closure_ms`
  (or `longest`/`either`, `--delay-field`) reaches `--delay-level-ms`. The raw
  delay includes the level itself; `mean_excess_ms` subtracts it.
* **Stability**: flips of `(--stability-field >= --stability-threshold)`.

For the course trade-off, compare two configs with `sweep.py` (e.g. a short and
a long `window_ms`) and read `delay_mean_excess_ms` against `stability_flips`.

## Assumptions the spec did not fully fix

1. **Gap measure.** Invalid time is counted from the end of the last valid
   sample's capped hold. A closure run survives while this is <= `max_gap_ms`.
2. **Dropped sample output.** A record with a repeated/decreasing/missing
   timestamp changes no state and returns the previous output again (or an
   invalid placeholder, `frame_id`/`timestamp_ms` = -1 if unknown).
3. **`valid` also needs `valid_eye_time > 0`**, so a measurement field is never
   computed from zero data when `min_valid_fraction` is 0.
4. **Strict types.** Flags must be real booleans; `0/1`, strings, NaN, or
   probabilities outside [0, 1] make that sample invalid rather than coerced.
5. **Bridged gaps** keep `current_closure_ms` counting; the gap time is still
   not counted as valid time.
6. **Yawn candidate** is cancelled if the smoothed value drops below the *on*
   threshold before confirmation; the *off* threshold only ends a confirmed
   yawn. The refractory period applies after confirmed yawns only. The moving
   average uses fewer samples until `yawn_smooth_n` have arrived. A yawn that
   is still in progress counts once confirmed.
7. **Blink/closure window membership** uses the run's end time; run durations
   are not clipped at the window start. `longest_closure_ms` ignores
   `t_blink_*` (it covers blinks too).
8. `frame_id` missing -> previous + 1 (output only).

## Current limitations

* All defaults are unvalidated placeholders; tune from labelled recordings.
* `ear_*`, `left/right_open_prob` and `cue_flags` are accepted and unused.
* A closure is only known to be long once it is long: delay is at least the
  alert level.
* Mock data is synthetic: scores on it show the code works, not that the
  thresholds are right.

## Open items (spec section 13)

Implemented as configurable/as written, not decided: blink-rate scaling by
valid time vs full window, PERCLOS denominator, placeholder thresholds, unused
head-nod cues, SG-5's handling of face-lost windows, yawn counting at low mouth
validity, and the PERCLOS closure-level definition.
