#!/usr/bin/env bash
# Downloads the MediaPipe Face Landmarker model (same model SG1 uses).
# The SG2 CNN (eye_state_cnn.onnx) is produced by experiments/train_cnn.py.
set -euo pipefail
cd "$(dirname "$0")"
curl -L -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
echo "saved models/face_landmarker.task"
