# Module — Sub-group 2: Eye State & Blink Analysis

**Owner:** SG2: Hassan Ahmed ([@Hassanahmed-15](https://github.com/Hassanahmed-15)) · **Status (25 Oct 2026, midterm):** working prototype, primary + backup selected, V1 contract tested · **Board:** [team board](https://github.com/users/Hassanahmed-15/projects/2/views/2) · [SG2 board](https://github.com/users/Hassanahmed-15/projects/2/views/3)

## Purpose and role in the system
For every video frame, SG2 decides whether the driver's eyes are **open or closed**. It turns runs of closed frames into **blink** events (≤ 500 ms) and **long-closure** events (> 500 ms, a possible micro-sleep), and passes this per-frame eye evidence to SG4.

```
Camera ─► SG1 face + 6 eye points ─► SG2 eye state + blinks ─► SG4 temporal analysis ─► SG5 drowsiness decision ─► alert
```
SG2 provides the visual cue only. It does **not** decide drowsiness: that is SG4/SG5's job.

## Selected algorithm (midterm)

| Role | Config | How it works | TEST closed-frame F1 [95% CI] | Blink F1 | Uses |
|---|---|---|---|---|---|
| **Primary** | **G** [`configs/primary.yaml`](configs/primary.yaml) | Adaptive EAR proposes "closed" (threshold = 0.425 × this driver's normal open-eye EAR). A small CNN looks at the eye crop and must **confirm** it | **0.947 [0.928–0.962]** | **0.951** | SG1 eye points + frame |
| **Backup** | **C** [`configs/backup.yaml`](configs/backup.yaml) | Adaptive EAR only, with no model file (numpy + OpenCV). G switches to C automatically if the CNN or onnxruntime is unavailable | 0.918 [0.842–0.962] | 0.944 | SG1 eye points |

**Why G:** it cuts false "closed" alarms by **74%** (80 → 21) for 5 fewer true closed frames. It fixes the main Week 5 weakness: with the head tilted ≥ 10° from the driver's normal pose, false alarms fall from **70 to 12**. It raises the worst test video (eb8_08) from 0.70 to 0.89, keeps the frozen V1 interface, and costs 0.06 ms/frame.
**Caveat:** the paired bootstrap gives G > C in 79% of resamples, so the gain is likely but not statistically conclusive. That is why C stays as the fallback.
Alternatives tested: A, B, D, D2, E, F (see [`docs/midterm_evaluation.md`](docs/midterm_evaluation.md) §2). E/F (MediaPipe eyeBlink) score as well as G but need a field SG1 does not send in V1.

## Dependencies
- Python ≥ 3.10 (tested 3.12 on macOS and 3.10 with Jetson-equivalent versions: all 24 tests pass on both)
- Runtime: `numpy`, `pyyaml`, `opencv-python` (GUI window for the demo). Add `onnxruntime` for the primary's CNN check, and `mediapipe==1.1.0` to run directly on a camera or video
- Tests: `jsonschema`, `pytest` · Experiments only: `pandas`, `scikit-learn`, `matplotlib`, `torch`
- [`requirements.txt`](requirements.txt). Jetson notes, including a MediaPipe/OpenCV install trap: [`docs/jetson_deployment.md`](docs/jetson_deployment.md)

## Input / Output Interface (V1)
Contract: [`interfaces/sg2/README.md`](../interfaces/sg2/README.md) · schema [`sg2_eye_state_v1.schema.json`](../interfaces/sg2/sg2_eye_state_v1.schema.json) · tests [`tests/test_contract.py`](tests/test_contract.py)

- **Input (SG1 V1, unchanged):** `frame_id`, `timestamp_ms`, `face_valid`, `left_eye_pts`, `right_eye_pts` (6 ordered points: p1 outer, p2/p3 upper lid, p4 inner, p5/p6 lower lid, full-frame pixels, driver's own left/right), plus the BGR frame for the CNN check.
- **Output (`sg2-v1`, one record per frame):** `left_ear`, `right_ear`, `ear`, `left_eye_state`, `right_eye_state`, `eye_state` (`open`/`closed`/`unknown`), `closed_score`, `threshold`, `confidence`, `closed_duration_ms`, `blink_event` (on the frame a closure ends), `blink_count`, `long_closure_count`.

```python
from sg2_eye import EyeStateModule
sg2 = EyeStateModule("configs/primary.yaml")
record = sg2.process(sg1_block, frame)      # frame optional: without it the primary acts as the backup
```

## How to Run
```bash
# setup (from the repo root)
python3 -m venv .venv && source .venv/bin/activate
pip install -r module_sg2/requirements.txt
cd module_sg2 && bash models/download_models.sh        # landmark model + SG2 CNNs (SHA-256 checked)

python -m pytest tests -q                               # 24 checks, no data or camera needed

# DEMO (window: q quit, space pause, r reset; add --no-window --save demo.mp4 when headless)
python tools/demo.py --source 0                                              # live webcam (face confidence 0.2: works with a niqab)
python tools/demo.py --sg1-jsonl ../interfaces/sg2/examples/eb8_09_real_sg1.jsonl   # real test data + hand labels
python tools/demo.py --sg1-jsonl ../interfaces/sg2/mock/mock_sg1_drowsy.jsonl       # mock SG1: blinks + long closures
python tools/demo.py --source <eyeblink8 .avi>                               # full test video + hand labels (.tag)
python tools/demo.py --source csi                                            # Jetson CSI camera

# SG2 as a pipeline step (JSONL in -> JSONL out)
python tools/run_sg2.py --sg1-jsonl ../interfaces/sg2/mock/mock_sg1_drowsy.jsonl --config configs/primary.yaml --out /tmp/sg2.jsonl

# reproduce all experiments (~700 MB download, ~35 min on a laptop)
bash ../datasets/sg2/download_sg2_data.sh
python experiments/prepare_videos.py --workers 4        # landmarks + crops cache (+ low-light copy)
python experiments/train_cnn.py                         # CNN v1 (Week 5)
python experiments/train_cnn.py --with-dev-video        # CNN v2 (midterm)
python experiments/evaluate.py                          # Week 5 tables + figures
python experiments/evaluate_midterm.py                  # midterm tables, CIs, figures

# Jetson / camera readiness (writes evidence files)
python tools/check_env.py --out ../embedded_sg6/reports/env_$(hostname).md
python tools/camera_check.py --source 0 --pipeline      # or --source csi
```

## Test Data
Public, frame-annotated blink videos: [`datasets/sg2/README.md`](../datasets/sg2/README.md).
- **DEV (tuning only):** Talking Face + Eyeblink8 videos 1, 2, 3 (video 3 is held out of CNN v2 training and used to tune the primary's CNN threshold)
- **TEST (reporting only):** Eyeblink8 videos 4, 8, 9, 10, 11 (video 11 has glasses). 35,174 frames, 852 closed frames, 217 blinks. Also a synthetic low-light copy
- **CNN training:** MRL Eye (split by person) + CEW. CNN v2 adds webcam crops of DEV videos talking, 1, 2
- **In the repo:** 20 s real sample [`interfaces/sg2/examples/`](../interfaces/sg2/examples/) and 3 mock scenarios [`interfaces/sg2/mock/`](../interfaces/sg2/mock/)

## Current Performance (TEST, measured 25 Oct 2026)

| Config | F1 closed [95% CI] | False-closed | False-open | Blink F1 | Low-light F1 | SG2 ms/frame* |
|---|---|---|---|---|---|---|
| A Week 4 baseline EAR 0.20 | 0.361 [0.284–0.466] | 3,020 frames | 0 | 0.857 | 0.350 | 0.012 |
| C adaptive EAR (**backup**) | 0.918 [0.842–0.962] | 80 | 62 | 0.944 | 0.891 | 0.043 |
| **G adaptive EAR + CNN check (primary)** | **0.947 [0.928–0.962]** | **21** | 67 | **0.951** | 0.889 | 0.055 |
| E MediaPipe eyeBlink (needs SG1 change) | 0.945 [0.910–0.972] | 62 | 33 | 0.943 | 0.895 | 0.012 |

\*SG2 only, Apple M4 CPU. It excludes landmarks (~3.5–7 ms, SG1's job). Live webcam end-to-end: 29.5 fps (camera-limited). Not yet measured on Jetson.

### Drivers who wear a niqab (face covered except the eyes)
Tested on the same 5 TEST videos with a synthetic niqab: dark fabric over the head, face and neck, with an eye slit placed from the hand-annotated eye corners (`experiments/occlusion.py`). Results: [`results/occlusion/`](results/occlusion/).

| Face-detection confidence | Frames with a face found | Closed frames with no answer | Primary F1 | Primary blink recall / F1 |
|---|---|---|---|---|
| 0.5 (SG1's current setting) | 93.6% (glasses + niqab video: **72%**) | 100 | 0.921 | 0.834 / 0.889 |
| **0.2 (SG2 live tools default)** | **99.3%** (glasses + niqab video: **98%**) | **34** | 0.907 | **0.894 / 0.928** |
| normal faces, 0.5 → 0.2 | 99.7% → 99.7% | 0 → 0 | 0.947 → 0.948 | 0.931 → 0.931 |

- When a face is found, the eye points stay accurate under the niqab (median corner error 2.7% of the eye distance). SG2's EAR-based primary and backup keep working.
- The problem is **finding the face**. Lowering MediaPipe's confidence to 0.2 recovered 94% of the lost frames. The cost is a few more false "faces" when nothing is visible (+10 of 600 frames in the worst video).
- CNN-only methods fail under a niqab (D/D2 F1 ≈ 0.16–0.21: the dark fabric at the crop edge confuses them). This is another reason the CNN only confirms, never decides alone.

## Evidence links
| What | Where |
|---|---|
| Midterm rubric → evidence map + demo script | [`docs/midterm_evaluation.md`](docs/midterm_evaluation.md) |
| Primary vs backup tables, CIs, paired tests | [`results/midterm/`](results/midterm/) ([CI figure](results/midterm/figures/confidence_intervals.png), [difficult cases](results/midterm/figures/difficult_cases.png)) |
| Week 5 experiment (A–E), failure cases | [`docs/week5_report.md`](docs/week5_report.md), [`results/`](results/), [`results/figures/failure_cases.png`](results/figures/failure_cases.png) |
| Reference implementations (ranked, executed) | [`docs/reference_screening/`](docs/reference_screening/) |
| Decisions and dependencies | [`docs/decision_log.md`](docs/decision_log.md), [team tracker](../documentation/coordination/interface_dependency_tracker.md) |
| Niqab / face covering | [`results/occlusion/`](results/occlusion/), [`experiments/evaluate_occlusion.py`](experiments/evaluate_occlusion.py) |
| Jetson / camera readiness | [`docs/jetson_deployment.md`](docs/jetson_deployment.md), [`docs/env_report_dev_pc.md`](docs/env_report_dev_pc.md) |
| Viva questions | [`docs/viva_prep.md`](docs/viva_prep.md) |

## Current Limitations / failure cases
- **Not run on the Jetson yet** (no board access for SG2). The tools are ready ([`docs/jetson_deployment.md`](docs/jetson_deployment.md)).
- Only 4 TEST people (1 with glasses). No in-car, night or infrared footage. Low light is synthetic.
- The primary misses ~8% of closed frames (mostly half-closed frames at the start or end of a blink). Blink recall is 93%.
- The adaptive threshold calibrates on the first ~1 s of open eyes. If the driver starts with closed or downcast eyes, the first seconds can be wrong. A 0.10 lower clip limits this.
- CNN v2 alone over-calls "closed" on new people (Config D2 F1 0.70), so it is used only as a confirmation step.
- **Niqab / face covering:** if the face detector finds no face, SG2 has no eye points and reports `unknown` (shown as NO FACE). At confidence 0.2 this drops to 0.7% of frames on the simulated niqab, but real fabrics and lighting are untested. SG1's module still uses 0.5 (issue raised with SG1).
- The SG4 sign-off on `sg2-v1` is still pending (issue #7).

## Folder layout
```
module_sg2/
├── sg2_eye/      the module: geometry (EAR, crops), classifiers A–G, blink detector, EyeStateModule
├── configs/      primary.yaml (G), backup.yaml (C), Week 5 + midterm configs
├── tools/        demo.py, run_sg2.py, camera.py, camera_check.py, check_env.py, sg1_landmarks.py, make_mock_streams.py
├── experiments/  prepare_videos.py, train_cnn.py, evaluate.py, evaluate_midterm.py, export_real_sample.py
├── results/      Week 5 results, results/midterm/ (midterm)
├── tests/        test_sg2.py, test_contract.py (24 checks)
├── models/       download_models.sh (models are not committed)
└── docs/         midterm_evaluation.md, decision_log.md, jetson_deployment.md, week5_report.md, viva_prep.md, reference_screening/
```
