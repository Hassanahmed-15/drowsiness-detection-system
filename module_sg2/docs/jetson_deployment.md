# SG2 on the Jetson Orin Nano 8GB: dependencies, setup and readiness checks

**For:** SG6 (who leads the common Jetson setup) and every subgroup that needs to know SG2's deployment needs.
**Status (25 Oct 2026):** SG2 has **not** run on the Jetson yet, because the board was not available to SG2. Everything below marked ✅ was verified on a development PC. Items marked **Not Verified** must be confirmed on the board with the two scripts in §4.

## 1. What SG2 needs on the Jetson

| Need | Backup config C | Primary config G | Verified? |
|---|---|---|---|
| Python ≥ 3.10 (JetPack 6 ships 3.10) | ✅ | ✅ | ✅ All 24 SG2 tests pass on **Python 3.10.20** (macOS arm64, 25 Oct) |
| numpy, pyyaml | ✅ | ✅ | ✅ numpy 2.2.6 under Python 3.10 |
| OpenCV | ✅ (crops, drawing) | ✅ | Use the **system** OpenCV that comes with JetPack (it has GStreamer for the CSI camera) — **Not Verified** on our board |
| onnxruntime | – | ✅ CNN check (285 KB model, 2 crops/frame) | ✅ tests pass with **onnxruntime 1.23.2**. PyPI has aarch64 wheels for Python 3.10 only up to **1.23.2** (1.30 needs Python ≥ 3.11) |
| MediaPipe Face Landmarker | only to run SG2 standalone on a camera (in the full system the landmarks come from SG1) | same | PyPI has a `py3` `manylinux_2_28_aarch64` wheel for mediapipe 1.1.0 — install **Not Verified** on Jetson |
| GPU / CUDA / PyTorch | not needed | not needed (CPU is enough: 0.46 ms per frame on an M4 CPU) | – |

If onnxruntime or the model file is missing, the primary config **automatically runs as the backup** (it prints `falling back to adaptive EAR only`). SG2 therefore never blocks the pipeline.

## 2. Install trap that also affects SG1 (verified on PyPI, 25 Oct)

`mediapipe 1.1.0` declares `opencv-contrib-python` as a dependency. The pip OpenCV wheel has **no GStreamer**. If pip installs it into the environment, `import cv2` picks it up instead of the JetPack OpenCV, and the **CSI camera stops working** (`nvarguscamerasrc` pipelines fail to open). Avoid it like this:

```bash
python3 -m venv --system-site-packages ~/cv-venv      # can see the JetPack OpenCV
source ~/cv-venv/bin/activate
pip install "numpy<2.3" pyyaml jsonschema pytest "onnxruntime==1.23.2"
pip install --no-deps "mediapipe==1.1.0"               # do NOT let it pull opencv-contrib-python
pip install "absl-py~=2.3" "flatbuffers~=25.9" certifi matplotlib "sounddevice~=0.5"
python3 -c "import cv2; print(cv2.__version__, 'GStreamer' , 'YES' in [l for l in cv2.getBuildInformation().splitlines() if 'GStreamer' in l][0])"
```

`tools/check_env.py` shows whether `import cv2` resolves to a GStreamer-enabled build.

## 3. Setup steps (about 15 min, needs internet)

```bash
git clone https://github.com/Hassanahmed-15/drowsiness-detection-system.git
cd drowsiness-detection-system && git checkout sg2/midterm      # or main once merged
# environment: see section 2
cd module_sg2
bash models/download_models.sh        # face landmarker + SG2 CNN models
python -m pytest tests -q             # expect 24 passed
```

## 4. Readiness checks to run on the board (they write evidence files)

| Midterm criterion (Part C) | Command (run in `module_sg2/`) | Evidence produced |
|---|---|---|
| Boots, OS / storage identified (2) | `python tools/check_env.py --out ../embedded_sg6/reports/env_jetson.md` | board model, L4T/JetPack, root device (NVMe or microSD), free space, RAM, power mode |
| Stack documented: Python, Git, OpenCV, CUDA/PyTorch (1) | same report | versions + OpenCV GStreamer/CUDA flags + onnxruntime providers + PyTorch CUDA |
| Camera / video acquisition (2) | USB: `python tools/camera_check.py --source 0 --pipeline` <br> CSI: `python tools/camera_check.py --source csi --pipeline` | `report.json/.md`, first/last frame, measured FPS and resolution, face-detection rate, landmark + SG2 time, end-to-end FPS |
| Live SG2 on the board | `python tools/demo.py --source csi --no-window --save demo_jetson.mp4 --out demo_jetson.jsonl` (or with a monitor: no `--no-window`) | recorded demo video + sg2-v1 records |

A video file (`--source some.avi`) proves only the software path. The report says so (`proves_live_camera: false`), as the midterm brief requires.

## 5. Reference numbers from the development PC (not Jetson)

| Measurement | Apple M4 MacBook Air (25 Oct 2026) | Jetson Orin Nano |
|---|---|---|
| USB webcam capture | 640×480, measured 29.7–29.9 fps | Not Verified |
| Live webcam → landmarks → SG2 (primary) | landmarks 7.2 ms, SG2 1.07 ms, end-to-end 29.5 fps (camera-limited), face found 100% | Not Verified |
| SG2 only, per frame (real sample) | primary 0.034 ms, backup 0.033 ms | Not Verified |
| CNN v2 check, 2 eye crops, CPU | 0.46 ms | Not Verified |
| Landmarks on a 640×480 video file | 3.5 ms/frame | Not Verified |

## 6. Known blockers / next actions (Week 8 → Lab 9)

| Item | Owner | Status |
|---|---|---|
| Run §4 checks on the team's Jetson and commit the reports to `embedded_sg6/reports/` | SG6 with SG2 | **Open:** SG2 has no board access |
| Confirm the JetPack OpenCV has GStreamer and that MediaPipe installs without replacing it (§2) | SG6, SG1 | **Open** |
| Confirm which camera the team uses (USB or CSI) and its resolution/FPS | SG6 | **Open** |
| Measure SG2 primary latency on the board (target well under 33 ms per frame at 30 fps) | SG2 + SG6 | **Open** (issue #24) |
