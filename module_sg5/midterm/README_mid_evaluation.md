# SG-5: Drowsiness Decision Module — Midterm Evaluation

**Group:** D1 · **Subgroup:** SG-5
**Members:** Alveena Shahid, Abdullah Umar

SG-5 is the **decision stage** of the driver drowsiness detection pipeline. It takes the temporal eye
and mouth measurements produced by **SG-4** over a sliding window and classifies the driver's state as
**NORMAL**, **WARNING** or **DROWSY**, raising an alarm when the driver is drowsy.

At the midterm, SG-5 was built and tested on its own using mock SG-4 inputs, before integration with
SG-4 and deployment on the Jetson.

```
Camera → ... → SG-4 (temporal features over a window) → SG-5 (decision) → Alarm / UI
```

---

## Selected algorithms

| Role | Method | File |
|---|---|---|
| **Primary** | Hybrid fusion | `code/hybrid_fusion.py` → `hybrid_decision()` |
| **Backup** | Rule-based thresholds | `code/decision_rules.py` → `make_decision()` |
| Compared, rejected | Pure weighted fusion | `code/weighted_fusion.py` → `weighted_decision()` |

**Why hybrid fusion as primary:** it matches the rule-based method's accuracy on every test case
(including the borderline ones) while also producing a continuous `drowsiness_score` that reflects
combined evidence. A single strong cue cannot be "averaged away", and several weak cues can still add
up to a WARNING.

**Why rule-based as backup:** it is the simplest and most transparent method, gives the same states
as hybrid on all test cases, and is the cheapest to run (≈0.9 µs per decision). Its drawback is that
it only outputs three fixed scores (0.20 / 0.50 / 0.80).

**Why not pure weighted fusion:** it dilutes strong single cues. A PERCLOS of 0.38 contributes only
0.19 to the score, so a clearly drowsy driver is classed as NORMAL. It scored 3/8 on the borderline set.

---

## Input (from SG-4)

Each input is a Python dictionary:

| Field | Meaning |
|---|---|
| `frame_id`, `timestamp_ms` | Identify the frame / moment the decision refers to |
| `window_ms` | Length of the temporal window (10000 ms) |
| `valid` | Whether SG-4 had enough reliable data |
| `window_valid_fraction` | Fraction of the window with usable frames (0–1) |
| `perclos` | Percentage of eye closure over the window (0–1) |
| `blink_rate_per_min` | Blinks per minute |
| `mean_blink_duration_ms` | Average blink duration |
| `longest_closure_ms` | Longest eye closure in the window |
| `current_closure_ms` | Duration of the ongoing closure (0 if eyes are open) |
| `yawn_count` | Number of yawns in the window |

The decision methods currently use PERCLOS, longest closure, current closure and yawn count.
Blink rate and mean blink duration are passed through but not yet used.

## Output (to downstream / alarm)

| Field | Meaning |
|---|---|
| `frame_id`, `timestamp_ms` | Copied from the input |
| `decision_valid` | `False` if the input could not be trusted |
| `state` | `"NORMAL"`, `"WARNING"`, `"DROWSY"` or `None` if invalid |
| `drowsiness_score` | Numeric score (0–1) or `None` if invalid |
| `alarm_flag` | `True` only when state is DROWSY |
| `alarm_type` | `"DROWSINESS"` when alarmed, otherwise `None` |

## Input validation (all methods)

- `valid == False` → `decision_valid = False`, no state, no alarm
- `window_valid_fraction < 0.70` → same as above

This stops the system from raising (or suppressing) alarms based on poor tracking, e.g. when the face
is lost for most of the window.

---

## The three decision methods

### 1. Rule-based (backup)
**DROWSY** if any strong cue: PERCLOS ≥ 0.35, longest closure ≥ 1500 ms, current closure ≥ 2000 ms, or yawns ≥ 3.
**WARNING** otherwise if any moderate cue: PERCLOS ≥ 0.20, longest closure ≥ 800 ms, or yawns ≥ 2.
**NORMAL** otherwise. Scores are fixed per state (0.20 / 0.50 / 0.80).

### 2. Weighted fusion (rejected)
```
perclos_score = perclos               (clipped to 0–1)
closure_score = longest_closure / 2000 (clipped to 0–1)
yawn_score    = yawns / 3              (clipped to 0–1)
score = 0.50·perclos_score + 0.30·closure_score + 0.20·yawn_score
```
score < 0.30 → NORMAL · 0.30–<0.60 → WARNING · ≥ 0.60 → DROWSY.
Safety override: current closure ≥ 2000 ms → DROWSY, score 1.0.

### 3. Hybrid fusion (primary)
1. Strong cues (same as rule-based) → DROWSY directly, score 0.80.
2. Otherwise compute the weighted score as above.
3. WARNING if any moderate cue is present **or** the weighted score ≥ 0.30.
4. Otherwise NORMAL.

---

## Results

| Test set | Rule-based | Weighted fusion | Hybrid fusion |
|---|---|---|---|
| Basic (6 cases) | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) |
| Borderline (8 cases) | 8/8 (100%) | 3/8 (37.5%) | 8/8 (100%) |

Borderline comparison:

| Test case | Expected | Rule-based | Weighted | Hybrid |
|---|---|---|---|---|
| PERCLOS Below Warning (0.19) | NORMAL | NORMAL | NORMAL | NORMAL |
| PERCLOS Above Warning (0.21) | WARNING | WARNING | NORMAL ✗ | WARNING |
| Closure Below Drowsy (1450 ms) | WARNING | WARNING | NORMAL ✗ | WARNING |
| Closure Above Drowsy (1550 ms) | DROWSY | DROWSY | WARNING ✗ | DROWSY |
| Yawns Only (2) | WARNING | WARNING | NORMAL ✗ | WARNING |
| Low Valid Fraction (0.60) | None | None | None | None |
| Moderate Combination | WARNING | WARNING | WARNING | WARNING |
| High PERCLOS Only (0.38) | DROWSY | DROWSY | NORMAL ✗ | DROWSY |

Latency: see [`latency.md`](latency.md). Hybrid fusion takes ≈1.7 µs per decision on a laptop CPU,
so SG-5 adds negligible latency to the pipeline.

## Limitations

- All test cases are **hand-written mock data**, and the expected labels were set using the same
  thresholds the rule-based method uses. 100% accuracy confirms the logic is implemented as designed;
  it is not evidence of real-world performance.
- Thresholds and weights are initial choices and still need tuning on real SG-4 output (ideally a
  labelled drowsiness dataset).
- Decisions are made per window with no smoothing, so the state could flicker between frames.
  Hysteresis or a minimum alarm duration would help.
- Blink rate and mean blink duration are not yet used.

---

## How to run

Requires Python ≥ 3.8. The scripts use only the standard library.

```bash
cd module_sg5/midterm/code
python run_evaluation.py        # runs all 3 methods on basic + borderline cases, writes sg5_method_comparison.csv
python benchmark_latency.py     # per-decision latency of each method
```

Original notebook: `code/CV_Project_SG5.ipynb`. Open it in Google Colab (or Jupyter with `pip install -r requirements.txt`) and run all cells in order.

Using the primary method in the pipeline:

```python
from hybrid_fusion import hybrid_decision

result = hybrid_decision(sg4_output_dict)
if result["decision_valid"] and result["alarm_flag"]:
    trigger_alarm()
```

## Files

| File | Purpose |
|---|---|
| `code/mock_inputs.py` | Six mock SG-4 input dictionaries |
| `code/test_cases.py` | Basic + borderline test cases with expected states |
| `code/decision_rules.py` | Rule-based method (backup) |
| `code/weighted_fusion.py` | Weighted-fusion method (rejected) |
| `code/hybrid_fusion.py` | Hybrid-fusion method (primary) |
| `code/run_evaluation.py` | Accuracy of all three methods, writes the comparison CSV |
| `code/benchmark_latency.py` | Latency / throughput measurement |
| `code/sg5_method_comparison.csv` | Borderline comparison table |
| `code/CV_Project_SG5.ipynb` | Original midterm notebook |
| `latency.md` | Latency / FPS results |
| `demo_link.md` | Demo video link |

## Next steps

- Integrate with live SG-4 output and test on the Jetson.
- Tune thresholds and weights on real labelled data.
- Add temporal smoothing / hysteresis to avoid flickering alarms.
- Evaluate with more realistic metrics (per-class precision and recall, false-alarm rate, detection latency).

---

## AI-generated / open-source code disclosure

- **ChatGPT (OpenAI)** was used for help writing the code in the midterm notebook (`CV_Project_SG5.ipynb`)
  and the decision-method scripts derived from it.
- **Claude (Anthropic)** was used to convert the notebook into standalone scripts
  (`test_cases.py`, `run_evaluation.py`, `benchmark_latency.py`) and to format this README.
  The decision logic itself is unchanged from the notebook.
- Open-source code used: none.
