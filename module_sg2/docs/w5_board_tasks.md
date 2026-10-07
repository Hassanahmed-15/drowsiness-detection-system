# SG2 — Week 5 Project-board tasks

One task per actionable item, named and labelled as the Week 5 guideline asks.
Labels: `W5`, `SG2`, plus one type label (`Research` / `Code` / `Test` / `Integration` / `Documentation`).
Branch: `sg2/w5-eye-state-experiments` → PR into `module_sg2` → PR into `main`.

| # | Title | Type | Owner | Completion evidence | Status |
|---|---|---|---|---|---|
| 1 | W5 \| SG2 \| Recreate Week 4 EAR baseline as runnable module | Code | @Hassanahmed-15 | `sg2_eye/`, `configs/baseline_w4_ear_fixed.yaml`, `tools/run_sg2.py` commit | Done |
| 2 | W5 \| SG2 \| Define sg2-v1 output interface + JSON schema for SG4 | Integration | @Hassanahmed-15 | `interfaces/sg2/README.md`, `sg2_eye_state_v1.schema.json` | Review (SG4 to confirm) |
| 3 | W5 \| SG2 \| Publish mock SG1/SG2 streams so SG4 can work independently | Integration | @Hassanahmed-15 | `interfaces/sg2/mock/*.jsonl` + `mock_ground_truth.json` | Done |
| 4 | W5 \| SG2 \| Prepare labelled test data (Eyeblink8 + Talking Face, DEV/TEST split) | Test | @Hassanahmed-15 | `datasets/sg2/`, `experiments/prepare_videos.py`, `results/dataset_summary.csv` | Done |
| 5 | W5 \| SG2 \| Tune fixed EAR threshold (sweep 0.10–0.30) | Test | @Hassanahmed-15 | `results/threshold_sweep.csv`, `results/figures/ear_threshold_sweep.png` | Done |
| 6 | W5 \| SG2 \| Implement per-driver adaptive EAR threshold | Code | @Hassanahmed-15 | `EARAdaptive` in `sg2_eye/classifiers.py`, `configs/w5_C_ear_adaptive.yaml` | Done |
| 7 | W5 \| SG2 \| Train CNN eye-state classifier (MRL + CEW) and export ONNX | Code | @Hassanahmed-15 | `experiments/train_cnn.py`, `results/cnn_training.json`, `results/cnn_mrl_difficult_cases.csv` | Done |
| 8 | W5 \| SG2 \| Compare configs A–E on common TEST set (frame + blink metrics) | Test | @Hassanahmed-15 | `results/week5_comparison.csv`, `results/per_video_results.csv` | Done |
| 9 | W5 \| SG2 \| Difficult cases: glasses, head pose, low light | Test | @Hassanahmed-15 | `results/difficult_cases.csv`, `results/figures/difficult_cases_f1.png` | Done |
| 10 | W5 \| SG2 \| Failure-case analysis and Week 6 decision | Documentation | @Hassanahmed-15 | `docs/week5_report.md` §5–6, `results/figures/failure_cases.png` | Done |
| 11 | W5 \| SG2 \| Reference implementation screening (≥3 candidates, ranked) | Research | @Hassanahmed-15 | `docs/reference_screening/` | Done |
| 12 | W5 \| SG2 \| Module README (purpose, deps, run, I/O, data, limits) | Documentation | @Hassanahmed-15 | `module_sg2/README.md` | Done |
| 13 | W5 \| SG2 \| Ask SG1 about adding `blendshapes` field (V1.1 proposal) | Integration | @Hassanahmed-15 | Comment/issue answered by SG1 | Blocked: waiting for SG1 decision |
| 14 | W5 \| SG2 \| Hand ONNX model + latency numbers to SG6 for Jetson test | Integration | @Hassanahmed-15 | SG6 confirms model runs on Jetson | To Do (rolls to W6) |

All tasks are owned by @Hassanahmed-15 (SG2). They are created as GitHub issues with the labels above.
