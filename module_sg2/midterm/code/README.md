# SG2 midterm code: run commands and packages

This is a frozen, self-contained copy of the SG2 code as demonstrated at the midterm. It holds the primary and backup methods, the demo, and the latency benchmark. Development continues in [`module_sg2/`](../../). This folder does not change, so the midterm results can always be re-run.

| Method | Config | What it does |
| --- | --- | --- |
| **Primary, config G** | [`configs/primary.yaml`](configs/primary.yaml) | Adaptive EAR threshold (0.425 × the driver's 80th-percentile open-eye EAR over 30 s) **plus** a CNN v2 eye-crop check (closed only if P(closed) > 0.525). TEST closed-frame F1 0.947, blink F1 0.951. |
| **Backup, config C** | [`configs/backup.yaml`](configs/backup.yaml) | The same adaptive EAR without the CNN. Needs no model file. TEST F1 0.918, blink F1 0.944. The primary falls back to this automatically if the CNN cannot run. |

**Why this choice:** G was the most accurate, and cut false "closed" alarms from 80 to 21 against C on the TEST videos. C is kept as the backup because it needs only NumPy and OpenCV and runs on any device.

## Install (Python ≥ 3.10; tested on 3.12 on macOS, and on 3.10 like the Jetson)

```bash
cd module_sg2/midterm/code
pip install -r requirements.txt          # numpy, pyyaml, opencv-python, onnxruntime, mediapipe==1.1.0 (+ test/training extras)
bash models/download_models.sh           # face_landmarker.task + eye_state_cnn_v2.onnx, checks SHA-256
```

On the Jetson, use the system OpenCV (it has GStreamer for the CSI camera); see [`../../docs/jetson_deployment.md`](../../docs/jetson_deployment.md).

## Run (all commands from this folder)

| Demo | Command | Needs |
| --- | --- | --- |
| Real 20 s test sample with hand labels | `python tools/demo.py --sg1-jsonl examples/eb8_09_real_sg1.jsonl` | nothing else |
| Mock SG1 stream: drowsy driver | `python tools/demo.py --sg1-jsonl examples/mock_sg1_drowsy.jsonl` | nothing else |
| Live webcam (laptop / Jetson USB) | `python tools/demo.py --source 0` | camera |
| Jetson CSI camera | `python tools/demo.py --source csi` | camera |
| Saved test video + its `.tag` labels | `python tools/demo.py --source <eyeblink8>/8/27122013_151644_cam.avi` | dataset (`bash ../../../datasets/sg2/download_sg2_data.sh`) |
| Backup method instead of primary | add `--config configs/backup.yaml` | |
| Headless / record the window | add `--no-window --save demo.mp4` | |
| sg2-v1 JSON records for SG4 | `python tools/run_sg2.py --sg1-jsonl examples/eb8_09_real_sg1.jsonl --config configs/primary.yaml --out records.jsonl` | |
| Latency / FPS | `python tools/benchmark_latency.py --video <test video> --frames 20000` (results in [`../latency.md`](../latency.md)) | dataset |
| Environment check (e.g. on the Jetson) | `python tools/check_env.py` | |
| Camera readiness | `python tools/camera_check.py --source 0 --pipeline` | camera |

Expected output of the first command (checked 16 Nov 2026):

```
[demo] 600 frames | blinks 8 | long closures 0 | annotated closures 8 | agree 99.6%  F1 closed 0.98  (FP 0, FN 2)
```

Keys in the demo window: `q` quit, `space` pause, `r` reset.

## Contents

```
code/
├── sg2_eye/      the module (EAR, eye crops, classifiers, blink counter, V1 record)
├── tools/        demo.py, run_sg2.py, benchmark_latency.py, camera.py, camera_check.py, check_env.py, sg1_landmarks.py
├── configs/      primary.yaml (G), backup.yaml (C)
├── models/       download_models.sh (model files are not committed)
├── examples/     real 20 s sample as SG1 blocks + hand labels, mock drowsy stream + scripted ground truth
├── experiments/  common.py (reads the Eyeblink8 .tag labels for the demo)
└── requirements.txt
```

Input: SG1 V1 face blocks (6 points per eye). Output: one `sg2-v1` record per frame (eye state, EAR, blink and long-closure events). The full contract is in [`../../../interfaces/sg2/README.md`](../../../interfaces/sg2/README.md).
