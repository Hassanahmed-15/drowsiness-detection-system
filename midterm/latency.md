# SG-1 Latency / FPS Results

Measured by `code/benchmark_latency.py`. All numbers below were produced on the machine described here and will differ on other hardware (including the Jetson).

## Test setup

| Item | Value |
|------|-------|
| Device label | Mujtaba |
| Date | 2026-11-17 21:46 |
| OS | Windows-11-10.0.26200-SP0 |
| Processor | Intel64 Family 6 Model 140 Stepping 1, GenuineIntel (8 logical cores) |
| Python / MediaPipe / OpenCV / NumPy | 3.13.12 / 1.1.1 / 5.0.0 / 2.5.4 |
| Video | Subject1.mp4, 1920x1080, 30.00 FPS nominal |
| Frames read / measured | 1067 / 1057 (first 10 excluded as warm-up) |
| Mode | MediaPipe Face Landmarker, VIDEO running mode, max_faces=3, CPU delegate (default) |

## Summary (end-to-end SG-1, per frame)

`total` = convert + detect + packet. It excludes video decoding and the debug overlay. FPS is 1000 / mean total latency.

| Configuration | face_conf_min | Mean (ms) | p50 (ms) | p95 (ms) | Max (ms) | FPS (from mean) | FPS (from p95) | Detection rate | face_valid rate |
|---|---|---|---|---|---|---|---|---|---|
| Config A | 0.50 | 20.82 | 16.33 | 45.19 | 61.41 | 48.0 | 22.1 | 100.0% | 100.0% |
| Config B | 0.70 | 18.65 | 17.17 | 28.04 | 53.76 | 53.6 | 35.7 | 100.0% | 100.0% |

## Per-stage breakdown (mean / p95, ms)

| Configuration | convert | detect | packet | total |
|---|---|---|---|---|
| Config A | 4.04 / 8.54 | 15.26 / 32.92 | 1.52 / 3.44 | 20.82 / 45.19 |
| Config B | 3.66 / 5.47 | 13.67 / 20.70 | 1.33 / 2.31 | 18.65 / 28.04 |

## Notes

- `detect` is the MediaPipe inference call; `packet` covers landmark extraction, ROI boxes, ROI crops and the solvePnP head pose.
- Timing uses `time.perf_counter()` around each stage, single process, one frame at a time (no batching, no threading).
- The `avg_latency_ms` / `processing_fps` columns in `config_comparison.csv` time only the `detect` call, so they are lower-latency / higher-FPS than the end-to-end numbers here. Use the end-to-end numbers when sharing with other subgroups.
- Jetson Orin Nano figures are NOT included unless this script was run on the Jetson (see Device label).
