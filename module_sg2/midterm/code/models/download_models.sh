#!/usr/bin/env bash
# Downloads every model SG2 needs into module_sg2/models/ (model files are not committed to git).
#   face_landmarker.task   - Google MediaPipe Face Landmarker (the same model SG1 uses)
#   eye_state_cnn_v2.onnx  - SG2 CNN check used by the PRIMARY config G (+ Config D2)
#   eye_state_cnn.onnx     - SG2 Week 5 CNN (Config D)
# Re-create the CNNs instead with experiments/train_cnn.py [--with-dev-video].
set -euo pipefail
cd "$(dirname "$0")"
REL=https://github.com/Hassanahmed-15/drowsiness-detection-system/releases/download/sg2-models-v2

curl -fL -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
curl -fL -o eye_state_cnn_v2.onnx "$REL/eye_state_cnn_v2.onnx"
curl -fL -o eye_state_cnn.onnx "$REL/eye_state_cnn.onnx"

# integrity check (shasum on macOS, sha256sum on Linux/Jetson)
SUM=$(command -v sha256sum || echo "shasum -a 256")
$SUM -c - <<'SHA'
e819f4757ed8dd671f2284f394b6429b2fd1f484061ddbf641a9152d361ee955  eye_state_cnn_v2.onnx
78557ca53ab09d1a03a4addde174150ea991295a866b072685d1a8070c5184cd  eye_state_cnn.onnx
SHA
echo "SG2 models ready in $(pwd)"
