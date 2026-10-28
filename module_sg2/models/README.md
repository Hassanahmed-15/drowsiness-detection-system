# SG2 models (not committed: repo rule says no model files in git)

Get them all with `bash download_models.sh`. It also verifies the SHA-256 of the SG2 CNNs.

| File | Source | Used by |
|---|---|---|
| `face_landmarker.task` (3.6 MB) | Google MediaPipe (the same model SG1 uses) | `tools/sg1_landmarks.py`, `tools/demo.py`, `tools/run_sg2.py --video` |
| `eye_state_cnn_v2.onnx` (285 KB, 72k params) | [release `sg2-models-v2`](https://github.com/Hassanahmed-15/drowsiness-detection-system/releases/tag/sg2-models-v2), or `python experiments/train_cnn.py --with-dev-video` | **Primary config G** (CNN check), Config D2 |
| `eye_state_cnn.onnx` (285 KB) | same release, or `python experiments/train_cnn.py` | Config D (Week 5) |

ONNX input `eye`: float32 `(N, 1, 32, 32)`. Each crop is standardised to zero mean and unit std (`sg2_eye.classifiers.standardise`). Output `closed_logit`: `(N, 1)`, and sigmoid gives P(closed). Opset 17. It runs on onnxruntime 1.23.2 (the newest version with Python 3.10 aarch64 wheels, for the Jetson) and on 1.30.

If `eye_state_cnn_v2.onnx` is missing, the primary config prints a warning and runs as the backup config C.
