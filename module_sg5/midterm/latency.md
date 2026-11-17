# SG-5 Latency / FPS

**Measured:** 17 Nov 2026 · **Script:** `code/benchmark_latency.py` (20,000 rounds × 14 test inputs per method)
**Hardware:** Intel Core i7-1185G7 @ 3.00 GHz (laptop CPU), Windows 11, Python 3.13 (single thread)

| Method | Mean (µs) | Median (µs) | p99 (µs) | Decisions / s |
|---|---|---|---|---|
| **Hybrid fusion (primary)** | 1.68 | 1.60 | 4.40 | ~595,000 |
| Rule-based (backup) | 0.89 | 0.70 | 2.10 | ~1,125,000 |
| Weighted fusion (rejected) | 1.82 | 1.50 | 4.40 | ~550,000 |

## What this means for the other subgroups

- SG-5 runs once per SG-4 output. Its latency is **under 5 µs (0.005 ms) per decision**, even at p99.
- At 30 FPS the frame budget is 33 ms, so SG-5 uses **< 0.02 %** of it. **SG-5 will not limit pipeline FPS.**
  End-to-end FPS will be set by the upstream vision stages (SG-1 to SG-4).
- SG-5 has no model weights or GPU dependency. It is pure Python arithmetic and comparisons.

## Not yet measured

- Latency on the **Jetson** (Stage 2). It is expected to be slower than on a laptop, but should stay in the microsecond range.
- End-to-end latency with live SG-4 input.

To reproduce:

```bash
cd module_sg5/midterm/code
python benchmark_latency.py
```
