# SG2: latency / FPS (midterm submission)

**Module:** SG2, Eye State & Blink Analysis · **Measured:** 16 Nov 2026 · **Script:** [`code/tools/benchmark_latency.py`](code/tools/benchmark_latency.py) · **Raw numbers:** [`latency_video.json`](latency_video.json), [`latency_sg1_blocks.json`](latency_sg1_blocks.json)

## Summary for the other subgroups

| What | Per frame | FPS |
| --- | ---: | ---: |
| **SG2 primary** (config G: adaptive EAR + CNN v2 check) | **0.150 ms** (p95 0.195 ms) | ~6,700 |
| **SG2 backup** (config C: adaptive EAR only) | **0.046 ms** (p95 0.052 ms) | ~21,800 |
| SG1-compatible landmarks (MediaPipe Face Landmarker), for reference | 4.03 ms (p95 4.18 ms) | ~248 |
| **End to end** on a video file: decode + landmarks + SG2 primary | **4.32 ms** (p95 4.51 ms) | **~232** |

SG2 adds **0.15 ms per frame**, about 3.5% of the frame cost. At 30 fps the budget is 33 ms per frame, so SG2 uses less than 0.5% of it. Downstream modules (SG3/SG4) can count on an SG2 record within the same frame. All numbers are from an Apple M4 laptop CPU. **Jetson Orin Nano numbers are not measured yet** (see the last section).

## Setup

| Item | Value |
| --- | --- |
| Input | Eyeblink8 video 8 (`27122013_151644_cam.avi`), held-out TEST video, 640×480, all **10,663 frames**, face found in 100% of frames |
| Machine | Apple M4 (arm64), macOS 15, CPU only (no GPU used), Python 3.12.13, OpenCV 5.0.0, onnxruntime 1.30.0, MediaPipe 1.1.0 |
| Method | Each stage timed with `time.perf_counter()` around the call, on the same frames. The first 30 frames are dropped as warm-up. FPS = 1000 / mean ms. End to end is measured offline from a file, so it is not limited by a camera's frame rate. |
| Primary CNN | Loaded (`eye_state_cnn_v2.onnx`). It only runs when the EAR says "closed", which is why the primary's mean is close to the backup's. |

## Full results: TEST video (10,663 frames)

| Stage | Mean ms | Median ms | p95 ms | Max ms | FPS (1000/mean) |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode (`cv2.VideoCapture.read`) | 0.137 | 0.131 | 0.183 | 1.777 | 7,278 |
| landmarks (SG1's step, MediaPipe) | 4.029 | 4.026 | 4.179 | 8.909 | 248 |
| SG2 primary (config G) | 0.150 | 0.133 | 0.195 | 1.654 | 6,679 |
| SG2 backup (config C) | 0.046 | 0.046 | 0.052 | 0.186 | 21,760 |
| end to end, primary | 4.316 | 4.305 | 4.512 | 10.954 | 232 |
| end to end, backup | 4.212 | 4.206 | 4.381 | 10.743 | 237 |

## SG2 only, from SG1 V1 blocks (no frames)

This is SG2's cost when it receives SG1's JSON output and no camera image. The CNN check needs the image, so it is skipped. Input: the real 20 s sample [`code/examples/eb8_09_real_sg1.jsonl`](code/examples/eb8_09_real_sg1.jsonl), 600 frames.

| Config | Mean ms | Median ms | p95 ms |
| --- | ---: | ---: | ---: |
| primary (CNN skipped, no frames) | 0.039 | 0.033 | 0.037 |
| backup | 0.033 | 0.032 | 0.036 |

## Live camera (earlier measurement)

During the midterm preparation (25 Oct 2026), the live webcam path was measured on the same laptop with [`code/tools/camera_check.py`](code/tools/camera_check.py) `--source 0 --pipeline`:
- USB webcam at 640×480 delivered **29.7 fps**.
- End to end, webcam → landmarks → SG2, ran at **29.5 fps**. The camera was the limit, and the face was found in 100% of frames.

That run included the camera driver and OpenCV's capture thread, so its per-stage times are higher than the offline numbers above (see [`../docs/jetson_deployment.md`](../docs/jetson_deployment.md)). The table above is the reference measurement for this submission.

## Reproduce

```bash
cd module_sg2/midterm/code
pip install -r requirements.txt && bash models/download_models.sh
python tools/benchmark_latency.py --video <path>/eyeblink8/8/27122013_151644_cam.avi --frames 20000
python tools/benchmark_latency.py --sg1-jsonl examples/eb8_09_real_sg1.jsonl --out ../latency_sg1_blocks.json
```

## Jetson Orin Nano: pending

SG2 has not had access to the board yet. The same benchmark and `camera_check.py` are ready for SG6 to run on the Jetson (issue [#24](https://github.com/Hassanahmed-15/drowsiness-detection-system/issues/24), open). Expected impact: SG2 is NumPy plus one 285 KB ONNX model, so it should stay far below the 33 ms frame budget on the Orin Nano CPU. This is still to be confirmed with a measurement.
