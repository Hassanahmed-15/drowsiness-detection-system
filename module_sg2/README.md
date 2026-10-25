# Module — Sub-group 2: Eye State & Blink Analysis

**Owner:** SG2: Hassan Ahmed ([@Hassanahmed-15](https://github.com/Hassanahmed-15))

## Purpose
For every video frame, SG2 decides whether the driver's eyes are **open or closed**. From those decisions it detects **blinks** (closures ≤ 500 ms) and **long closures** (> 500 ms, possible micro-sleeps). It passes this per-frame eye evidence to SG4.

```
SG1 (face + eye landmarks) ──► SG2 eye state + blinks ──► SG4 temporal analysis ──► SG5 decision/alert
```
SG2 does not decide drowsiness. It only provides the visual cue.

## Dependencies
- Python 3.12 (tested on macOS Apple M4, 7 Oct 2026)
- Runtime: `numpy`, `opencv-python-headless`, `pyyaml`. Add `onnxruntime` for the CNN config. `mediapipe` is only needed to run directly on a video without SG1
- Experiments: `pandas`, `scikit-learn`, `matplotlib`, `torch`
- See [`requirements.txt`](requirements.txt). The models are not committed: `bash models/download_models.sh` fetches the MediaPipe landmark model, and `experiments/train_cnn.py` makes `models/eye_state_cnn.onnx`

## Input / Output Interface
Full contract: [`interfaces/sg2/README.md`](../interfaces/sg2/README.md) · schema [`sg2_eye_state_v1.schema.json`](../interfaces/sg2/sg2_eye_state_v1.schema.json)

- **Input (SG1 V1, unchanged):** `frame_id`, `timestamp_ms`, `face_valid`, `left_eye_pts`, `right_eye_pts`. These are 6 ordered points per eye (p1 outer, p2/p3 upper lid, p4 inner, p5/p6 lower lid) in full-frame pixels, using the driver's own left/right. The BGR frame is needed only for the CNN config.
- **Output (`sg2-v1`, one per frame):** `left_ear`, `right_ear`, `ear`, `left_eye_state`, `right_eye_state`, `eye_state` (`open`/`closed`/`unknown`), `closed_score`, `threshold`, `confidence`, `closed_duration_ms`, `blink_event` (only on the frame where a closure ends), `blink_count`, `long_closure_count`.
- **Mock data** for SG4 (and SG1-format input for SG2): [`interfaces/sg2/mock/`](../interfaces/sg2/mock/)

```python
from sg2_eye import EyeStateModule
sg2 = EyeStateModule("configs/w6_selected.yaml")
record = sg2.process(sg1_block, frame)      # frame is optional except for the CNN config
```

## How to Run
```bash
# setup (from the repo root)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r module_sg2/requirements.txt
bash module_sg2/models/download_models.sh
cd module_sg2

# 1. self-checks (no data needed)
python tests/test_sg2.py

# 2. run on SG1 output (here: the mock SG1 stream)
python tools/run_sg2.py --sg1-jsonl ../interfaces/sg2/mock/mock_sg1_drowsy.jsonl \
       --config configs/w6_selected.yaml --out /tmp/sg2_out.jsonl
#   -> [sg2] 1800 frames, ... blinks=14, long_closures=3

# 3. run directly on a driver video (uses the SG1-compatible landmark source) + demo overlay
python tools/run_sg2.py --video driver.mp4 --config configs/w6_selected.yaml \
       --out results/driver_sg2.jsonl --overlay /tmp/driver_overlay.mp4

# 4. reproduce the Week 5 experiment (~700 MB download, ~25 min on a laptop)
bash ../datasets/sg2/download_sg2_data.sh
python experiments/prepare_videos.py --workers 4   # landmarks/crops cache (+ low-light copy)
python experiments/train_cnn.py                    # Config D model -> models/eye_state_cnn.onnx
python experiments/evaluate.py                     # tables + figures -> results/
```

| Config file | Method |
|---|---|
| `configs/baseline_w4_ear_fixed.yaml` | **A**: Week 4 baseline, EAR < 0.20 |
| `configs/w5_B_ear_fixed_tuned.yaml` | **B**: EAR < 0.13 (tuned on DEV) |
| `configs/w5_C_ear_adaptive.yaml` | **C**: adaptive per-driver EAR threshold (k = 0.425) |
| `configs/w5_D_cnn.yaml` | **D**: CNN on the 32×32 eye crop (needs the ONNX model) |
| `configs/w5_E_blendshape.yaml` | **E**: MediaPipe eyeBlink blendshape (needs a non-V1 SG1 field) |
| **`configs/w6_selected.yaml`** | **selected for Week 6 = C** |

## Test Data
Public, frame-annotated blink videos. Download them with [`datasets/sg2/download_sg2_data.sh`](../datasets/sg2/); details are in [`datasets/sg2/README.md`](../datasets/sg2/README.md).
- **DEV (tuning):** Talking Face + Eyeblink8 videos 1, 2, 3
- **TEST (reporting):** Eyeblink8 videos 4, 8, 9, 10, 11 (video 11: driver wears glasses). 35,174 frames, 852 closed frames, 217 blinks
- **TEST low-light:** the same videos, synthetically darkened
- **CNN training:** MRL Eye Dataset (split by person) + CEW eye patches

## Current Performance
Measured 7 Oct 2026 on the Eyeblink8 TEST videos. Full tables are in [`results/week5_comparison.csv`](results/week5_comparison.csv) and [`docs/week5_report.md`](docs/week5_report.md).

| Config | F1 closed frames | False-closed rate | False-open rate | Blink F1 | SG2 time / frame |
|---|---|---|---|---|---|
| A baseline EAR 0.20 | 0.361 | 9.14% | 0.0% | 0.857 | 0.012 ms |
| B EAR 0.13 | 0.833 | 0.89% | 4.1% | 0.934 | 0.012 ms |
| **C adaptive EAR (selected)** | **0.918** | **0.24%** | 7.3% | **0.944** | 0.040 ms |
| D CNN | 0.872 | 0.05% | 21.1% | 0.901 | 0.46 ms |
| E blendshape (needs SG1 change) | 0.945 | 0.19% | 3.9% | 0.943 | 0.012 ms |

Timings were measured on an Apple M4 CPU and do not include SG1's landmark step (~4.1 ms/frame). They are not Jetson numbers.

## Current Limitations
- Looking down (|pitch| ≥ 15°) is the weakest case for EAR methods (C: F1 0.75).
- Only 4 TEST people (1 with glasses). No real in-car, night or infrared video yet. Low light is synthetic.
- The adaptive threshold needs about 1 s of open-eye frames to warm up. Until then it uses the fixed 0.13 threshold.
- The CNN (D) loses recall on webcam video because of domain shift from its infrared/photo training data.
- Not yet measured on a Jetson (planned with SG6 in Week 6).

## Folder layout
```
module_sg2/
├── sg2_eye/            # the module: geometry (EAR, crops), classifiers A–E, blink detector, EyeStateModule
├── configs/            # one YAML per configuration (A–E) + w6_selected.yaml
├── tools/              # run_sg2.py (CLI), sg1_landmarks.py (SG1-compatible source), make_mock_streams.py
├── experiments/        # prepare_videos.py, train_cnn.py, evaluate.py, common.py (split + labels)
├── results/            # Week 5 CSVs + figures/
├── tests/              # test_sg2.py
├── models/             # downloaded / trained models (not committed)
└── docs/               # week5_report.md, viva_prep.md, w5_board_tasks.md, reference_screening/
```
