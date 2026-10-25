# SG2 decision log

Each technical decision, when it was made, the evidence behind it, and where it is recorded. Newest first.

| Date | Decision | Evidence / reason | Where |
|---|---|---|---|
| 2026-10-25 | **Primary = Config G** (adaptive EAR + CNN v2 check). **Backup = Config C** (adaptive EAR). G falls back to C automatically | TEST F1 0.947 [0.928–0.962] vs 0.918 [0.842–0.962]. False alarms 80 → 21. P(G > C) = 0.79 in the paired bootstrap | `results/midterm/`, `configs/primary.yaml`, `configs/backup.yaml` |
| 2026-10-25 | CNN v2 is used **only as a confirmation step**, never alone | D2 alone: F1 0.699, 718 false alarms (over-calls "closed" on unseen people) | `results/midterm/comparison.csv` |
| 2026-10-25 | Train CNN v2 with webcam eye crops from DEV videos talking / eb8_01 / eb8_02. Hold out eb8_03 for epoch and threshold choice | Week 5 CNN lost recall on webcam video (domain shift). TEST stays untouched | `experiments/train_cnn.py --with-dev-video` |
| 2026-10-25 | Measure head pitch **relative to each driver's median**, not absolute | Median pitch differs by camera setup (−5° to +14° across videos) | `experiments/evaluate_midterm.py` |
| 2026-10-25 | E and F (MediaPipe eyeBlink) are **not** adopted while SG1 V1 has no `blendshapes` field | They match G on accuracy but would change the frozen SG1 interface | issue #18 |
| 2026-10-25 | Ship CNN models as a GitHub release with SHA-256, not in git | Repo rule: no model files in git. Peers still need them to reproduce | release `sg2-models-v2`, `models/download_models.sh` |
| 2026-10-25 | Jetson environment: JetPack system OpenCV (GStreamer) + `mediapipe --no-deps` + `onnxruntime==1.23.2` | mediapipe requires `opencv-contrib-python` (no GStreamer → CSI breaks). onnxruntime ≥ 1.24 has no Python 3.10 aarch64 wheel. All tests pass on Py3.10 + ort 1.23.2 | `docs/jetson_deployment.md` |
| 2026-10-07 | Week 6 candidate = Config C (adaptive EAR) | Best V1-compatible config in Week 5: F1 0.918, blink F1 0.944 | `docs/week5_report.md` §6 |
| 2026-10-07 | Pose-aware (width-weighted) eye fusion **not** adopted | No gain: F1 0.8328 vs 0.8325 | `results/fusion_study.csv` |
| 2026-10-07 | Output interface `sg2-v1`: per-frame record with `eye_state`, EARs, `closed_duration_ms`, `blink_event` (blink ≤ 500 ms / long_closure > 500 ms). SG2 makes no drowsiness decision | Keeps the cue → temporal → decision hierarchy from the brief. `closed_duration_ms` lets SG4 react before the eyes reopen | `interfaces/sg2/README.md`, issue #7 (waiting for SG4) |
| 2026-10-07 | SG2 reads only SG1 V1 fields, with SG1's MediaPipe indices and 6-point order | Respect the frozen V1 interface (SG1 `landmark_map.md`) | `tools/sg1_landmarks.py`, `tests/test_contract.py` |
| 2026-10-07 | Test protocol: Eyeblink8 + Talking Face, split by video into DEV (tuning) and TEST (reporting) | Public frame-level labels. No tuning on test data | `experiments/common.py`, `datasets/sg2/README.md` |
| 2026-10-07 | Re-create the Week 4 baseline (EAR < 0.20) as code | The Week 4 code was not in the shared repo | `configs/baseline_w4_ear_fixed.yaml` |
