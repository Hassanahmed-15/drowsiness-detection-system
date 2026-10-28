# SG2 — Eye State & Blink Analysis

**Driver Drowsiness Monitoring · CS-477 · Sub-group 2**

| | |
|---|---|
| **Owner** | Hassan Ahmed ([@Hassanahmed-15](https://github.com/Hassanahmed-15)) |
| **Branch** | `module_sg2` (goes into `main` through PR [#31](https://github.com/Hassanahmed-15/drowsiness-detection-system/pull/31), waiting for review) |
| **Status** | Working module, main method and backup selected, input/output format tested (midterm, Oct 2026) |
| **Board** | [SG2 tasks](https://github.com/users/Hassanahmed-15/projects/2/views/3) · [team board](https://github.com/users/Hassanahmed-15/projects/2/views/2) |

---

## Purpose

For every video frame, SG2 answers one question: **are the driver's eyes open or closed?** It also turns closed frames into events: a **blink** (closed for up to 0.5 s) or a **long closure** (more than 0.5 s, a possible micro-sleep). SG2 sends this to SG4. It does **not** decide whether the driver is drowsy; SG4 and SG5 do that.

```
Camera ─► SG1: face + 6 points per eye ─► SG2: open / closed, blinks ─► SG4: patterns over time ─► SG5: drowsy? alert
```

### How it works (every frame)

| Step | What happens |
|---|---|
| 1. **EAR** | Eye Aspect Ratio = eye height ÷ eye width, from SG1's 6 eye points. About 0.3 when open, about 0.05 when closed. |
| 2. **Compare** | The eye counts as closed when EAR drops below **0.425 × this driver's normal open-eye EAR**, learned over the last 30 s. Every driver's eyes are a different size, so a fixed threshold does not work. |
| 3. **CNN check** | A small neural network looks at a 32×32 picture of each eye. It must **agree** before we say closed. This removes false alarms when the driver smiles or looks down. |
| 4. **Count** | Closed frames in a row become a **blink** (≤ 0.5 s) or a **long closure** (> 0.5 s). |

| Role | Steps | Config file | Why |
|---|---|---|---|
| **Main method (G)** | 1 + 2 + 3 + 4 | [`configs/primary.yaml`](configs/primary.yaml) | Best accuracy and fewest false alarms (see Current Performance) |
| **Backup (C)** | 1 + 2 + 4 | [`configs/backup.yaml`](configs/backup.yaml) | Needs no model file and runs anywhere. G switches to C by itself if the CNN cannot run |

---

## Dependencies

| Package | Needed for |
|---|---|
| Python ≥ 3.10 | everything (tested on 3.12, and on 3.10, the Jetson's version) |
| `numpy`, `pyyaml`, `opencv-python` | the module itself and the demo window |
| `onnxruntime` | the CNN check in the main method (on Jetson / Python 3.10: `onnxruntime==1.23.2`) |
| `mediapipe==1.1.0` | finding the eye points when SG2 runs directly on a camera or video |
| `jsonschema`, `pytest` | tests |
| `pandas`, `scikit-learn`, `matplotlib`, `torch` | experiments and CNN training only |

All of these are listed in [`requirements.txt`](requirements.txt). For the Jetson, read [`docs/jetson_deployment.md`](docs/jetson_deployment.md) first: a plain `pip install mediapipe` replaces the Jetson's camera-ready OpenCV.

Models are **not** stored in git. [`models/download_models.sh`](models/download_models.sh) downloads them and checks their SHA-256: the MediaPipe face model plus SG2's CNN (285 KB, from release [`sg2-models-v2`](https://github.com/Hassanahmed-15/drowsiness-detection-system/releases/tag/sg2-models-v2)).

---

## Input / Output Interface

Full contract: [`interfaces/sg2/README.md`](../interfaces/sg2/README.md) · JSON schema: [`sg2_eye_state_v1.schema.json`](../interfaces/sg2/sg2_eye_state_v1.schema.json) · checked by [`tests/test_contract.py`](tests/test_contract.py)

**Input:** one SG1 V1 block per frame (SG1's format, unchanged):

| Field | Meaning |
|---|---|
| `frame_id`, `timestamp_ms` | frame number and time |
| `face_valid` | `false` when SG1 found no face; SG2 then answers `unknown` |
| `left_eye_pts`, `right_eye_pts` | 6 points per eye in pixels: p1 outer corner, p2/p3 upper lid, p4 inner corner, p5/p6 lower lid |
| video frame | used only to cut the eye pictures for the CNN check (without it, the main method works as the backup) |

**Output:** one `sg2-v1` record per frame, sent to SG4:

| Field | Meaning |
|---|---|
| `eye_state` | `open`, `closed` or `unknown` (also `left_eye_state`, `right_eye_state`) |
| `ear`, `left_ear`, `right_ear` | the measured Eye Aspect Ratio |
| `threshold`, `closed_score`, `confidence` | the threshold used, the method's score, and how sure it is (0–1) |
| `closed_duration_ms` | how long the eyes have been closed so far (0 when open) |
| `blink_event` | on the frame a closure ends: `blink` or `long_closure`, with start, end and duration |
| `blink_count`, `long_closure_count` | running totals |

Real example (frame 31 of the test sample, at the end of a 133 ms blink):
```json
{"interface_version": "sg2-v1", "frame_id": 31, "timestamp_ms": 1033.3, "eye_valid": true,
 "method": "PRIMARY_G_ear_adaptive_cnn_check_p0.525", "left_ear": 0.1438, "right_ear": 0.1724, "ear": 0.1581,
 "left_eye_state": "open", "right_eye_state": "open", "eye_state": "open",
 "closed_score": -0.6108, "threshold": 0.11, "confidence": 0.962, "closed_duration_ms": 0.0,
 "blink_event": {"type": "blink", "start_frame": 27, "end_frame": 30, "start_ms": 900.0, "end_ms": 1000.0, "duration_ms": 133.3},
 "blink_count": 1, "long_closure_count": 0}
```

Using it from Python:
```python
from sg2_eye import EyeStateModule
sg2 = EyeStateModule("configs/primary.yaml")
record = sg2.process(sg1_block, frame)
```

---

## How to Run

**1. Set up (once).** Run from the repo root:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r module_sg2/requirements.txt
cd module_sg2
bash models/download_models.sh
```

**2. Check it works.** No data or camera needed:
```bash
python -m pytest tests -q            # expect: 24 passed
```

**3. Demo.** In the window: `q` quit, `space` pause, `r` reset. On a server, add `--no-window --save demo.mp4`.
```bash
python tools/demo.py --source 0                                                     # live webcam
python tools/demo.py --sg1-jsonl ../interfaces/sg2/examples/eb8_09_real_sg1.jsonl   # real test data, answers on screen
python tools/demo.py --sg1-jsonl ../interfaces/sg2/mock/mock_sg1_drowsy.jsonl       # mock SG1 input
python tools/demo.py --source csi                                                   # Jetson CSI camera
```
For the live webcam: look at the camera for 1 second first, so it learns your normal eye.

**4. Use SG2 as a pipeline step.** SG1 records in, SG2 records out:
```bash
python tools/run_sg2.py --sg1-jsonl ../interfaces/sg2/examples/eb8_09_real_sg1.jsonl \
       --config configs/primary.yaml --out results/sg2_output.jsonl
```

**5. Reproduce the experiments** (about 700 MB of data, about 35 minutes):
```bash
bash ../datasets/sg2/download_sg2_data.sh
python experiments/prepare_videos.py --workers 4      # eye points + eye pictures for every test frame
python experiments/train_cnn.py --with-dev-video      # train the CNN used by the main method
python experiments/evaluate_midterm.py                # all comparison tables and charts
```

**6. Jetson and camera checks** (they write report files):
```bash
python tools/check_env.py --out ../embedded_sg6/reports/env_$(hostname).md
python tools/camera_check.py --source 0 --pipeline    # or --source csi
```

---

## Test Data

All public research datasets with labels made by hand. They are not stored in git: [`datasets/sg2/download_sg2_data.sh`](../datasets/sg2/download_sg2_data.sh) downloads them. Details and citations: [`datasets/sg2/README.md`](../datasets/sg2/README.md).

| Dataset | What it is | Used for |
|---|---|---|
| **Eyeblink8** | 8 webcam videos, 4 people (one with glasses), every blink marked | **TEST:** videos 4, 8, 9, 10, 11. **Tuning:** videos 1, 2, 3 |
| **Talking Face** | 1 video, 5,000 frames, 61 blinks marked | tuning |
| **MRL Eye** | 84,898 infrared eye pictures, open or closed | training the CNN |
| **CEW** | 4,846 eye pictures from the internet | training the CNN |

- The **TEST** videos (35,174 frames, 852 closed frames, 217 blinks) were **never** used for tuning or training.
- Already in the repo, no download needed: a 20 s real test sample with hand labels ([`interfaces/sg2/examples/`](../interfaces/sg2/examples/)) and 3 mock scenarios for SG4 ([`interfaces/sg2/mock/`](../interfaces/sg2/mock/)).

---

## Current Performance

Measured on 25 Oct 2026, on the 5 TEST videos. F1 = how well closed eyes are found (1.0 = perfect). Details: [`results/midterm/`](results/midterm/).

| Method | F1, closed eyes (95% range) | Blink F1 | False alarms | Speed |
|---|---|---|---|---|
| Week 4 start: fixed EAR < 0.20 | 0.361 (0.28–0.47) | 0.857 | 3,020 | 0.01 ms |
| **Backup (C)**: adaptive EAR | 0.918 (0.84–0.96) | 0.944 | 80 | 0.04 ms |
| **Main method (G)**: adaptive EAR + CNN check | **0.947 (0.93–0.96)** | **0.951** | **21** | 0.06 ms |

- **Real sample in the repo:** 8 of 8 blinks found, F1 0.98.
- **Live webcam:** 29.5 frames per second end to end on a laptop (limited by the camera). Speeds above are SG2's own time on a laptop CPU; not yet measured on the Jetson.
- **Hard cases:**
  - glasses video: F1 0.97
  - simulated dark video: F1 0.89
  - head tilted 10° or more: false alarms 70 → 12 thanks to the CNN check
- **Niqab** (simulated, only the eyes visible): with face-detection confidence lowered from 0.5 to 0.2, the face is found in 98% of frames on the glasses + niqab video instead of 72%, and blink F1 is 0.93 ([`results/occlusion/`](results/occlusion/)).

### Known limitations
- Not yet run on the Jetson; the scripts are ready ([`docs/jetson_deployment.md`](docs/jetson_deployment.md)).
- Only 4 people in the test videos, all indoors. Dark video and the niqab were simulated.
- About 8% of closed frames are missed, mostly half-closed eyes at the start or end of a blink.
- The first second is less reliable while it learns the driver's normal eye.
- SG4 has not confirmed the output format yet (issue [#7](https://github.com/Hassanahmed-15/drowsiness-detection-system/issues/7)).

---

## Folder layout

```
module_sg2/
├── sg2_eye/       the module: EAR, eye pictures, the methods, blink counter
├── configs/       primary.yaml (main method), backup.yaml (backup), experiment settings
├── tools/         demo.py, run_sg2.py, camera_check.py, check_env.py
├── tests/         24 automatic tests
├── experiments/   data preparation, CNN training, evaluation scripts
├── results/       tables and charts from the experiments
├── models/        download_models.sh (the models themselves are not in git)
└── docs/          reports, decision log, Jetson guide, viva prep
```

## More documents
- [`docs/midterm_evaluation.md`](docs/midterm_evaluation.md): midterm rubric, evidence and demo script
- [`docs/decision_log.md`](docs/decision_log.md): what we decided and why
- [`docs/week5_report.md`](docs/week5_report.md): the Week 5 experiment
- [`docs/reference_screening/`](docs/reference_screening/): other projects we compared against
- [`docs/viva_prep.md`](docs/viva_prep.md): likely questions and answers
