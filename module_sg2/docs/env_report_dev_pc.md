# Environment report — Hassans-MacBook-Air-3.local

Generated 2026-10-25 11:25 by `module_sg2/tools/check_env.py`. **Not a Jetson** (development PC).

## Platform

| Item | Value |
|---|---|
| Hostname | Hassans-MacBook-Air-3.local |
| OS | Darwin 24.6.0 |
| Machine | arm64 |
| Board | Apple M4 (development PC, NOT a Jetson) |

## Storage / memory

| Item | Value |
|---|---|
| Root filesystem | 245 GB total, 17 GB free |
| Memory | 17 GB RAM |

## Development stack

| Item | Value |
|---|---|
| Python | 3.12.13 (/Users/hassanahmed/Desktop/cv/.venv/bin/python) |
| Git | git version 2.52.0 |
| OpenCV | 5.0.0 / GStreamer: NO / V4L/V4L2: NO / CUDA: NO / GUI: COCOA |
| CUDA toolkit | not found |
| PyTorch | 2.14.1 / CUDA available: False / device: Apple MPS |
| onnxruntime | 1.30.0 / providers: ['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider'] |
| numpy | 2.5.3 |
| yaml | 6.0.3 |
| mediapipe | 1.1.0 |
| jsonschema | 4.26.0 |

## Cameras

| Item | Value |
|---|---|
| Cameras | use tools/camera_check.py --source 0 (macOS needs Camera permission for the terminal) |

## SG2 smoke test

| Item | Value |
|---|---|
| SG2 primary on real sample | blinks 8/8 annotated, 0.034 ms/frame, CNN model loaded: True |
| SG2 backup on real sample | blinks 8/8 annotated, 0.033 ms/frame |
| CNN v2 inference (2 eyes, CPU) | 0.459 ms |

## Readiness checklist

- ✅ Python 3.10+
- ✅ Git
- ✅ OpenCV importable
- ❌ OpenCV GStreamer (needed for CSI camera)
- ✅ onnxruntime (primary config G)
- ✅ MediaPipe (landmarks on device)
