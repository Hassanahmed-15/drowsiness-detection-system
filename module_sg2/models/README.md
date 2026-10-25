# SG2 models (not committed — repo rule: no model files in git)

| File | How to get it | Used by |
|---|---|---|
| `face_landmarker.task` (3.6 MB) | `bash download_models.sh` (Google MediaPipe, the same model SG1 uses) | `tools/sg1_landmarks.py`, `tools/run_sg2.py --video` |
| `eye_state_cnn.onnx` (285 KB, 72k params) | `python experiments/train_cnn.py` (~3 min on Apple M4, needs MRL + CEW) | Config D (`configs/w5_D_cnn.yaml`) |

ONNX input `eye`: float32 `(N, 1, 32, 32)`, each crop standardised to zero mean and unit std (`sg2_eye.classifiers.standardise`). Output `closed_logit`: `(N, 1)`, where sigmoid → P(closed).
