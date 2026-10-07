# SG2 Reference Implementation Feasibility Screening

**Module:** SG2 – Eye State & Blink Analysis (CS-477 Driver Drowsiness Monitoring)
**Date of screening:** 2026-10-07
**Target hardware:** NVIDIA Jetson Orin Nano 8GB

## 1. Method

1. We searched GitHub (`gh search repos`, for example "drowsiness jetson", "eye blink detection", "eye state classification", "mrl eye dataset", and topic `jetson-nano`) and the web for eye-state and blink methods.
2. For every candidate we opened the real source: repo metadata, last commit, licence and README through `gh api`, model files, and the papers (PDF text).
3. We shortlisted 5 candidates and scored them with the course criteria (total 100).
4. We cloned and ran the top candidates on a real video (`talking.avi`, 5000 frames, with 61 hand-labelled blinks in `talking.tag`) on a Mac (Apple M4). **No test was run on a Jetson.**

**Verification rule:** every number in this file was read from a repo, a paper or our own run. If we could not check something, we wrote **Not Verified**. Mac timings are not Jetson timings.

Files in this folder:
- `reference_scores.csv`: scores, one row per candidate
- `execution_log.txt`: exact commands and outputs
- `scripts/`: the small test scripts we wrote (the repos were not edited)

## 2. Candidate long-list

| # | Candidate | Link | Kept / dropped | Why |
|---|---|---|---|---|
| A | e-candeloro/Driver-State-Detection | https://github.com/e-candeloro/Driver-State-Detection | **Kept** | MediaPipe + EAR in Python, MIT, last commit 2026-08-18, has tests. Same 6-point EAR as our SG1 input. |
| B | OpenVINO Open Model Zoo `open-closed-eye-0001` | https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/open-closed-eye-0001 | **Kept** | Tiny open/closed CNN (32×32 eye crop), ONNX download, Apache-2.0, accuracy reported on MRL Eye. |
| C | altaga/DBSE-monitor (Drowsiness part) | https://github.com/altaga/DBSE-monitor | **Kept** | PyTorch eye CNN with weights in the repo and a Jetson Nano build folder. |
| D | Tobias-Fischer/rt_gene (RT-BENE) | https://github.com/Tobias-Fischer/rt_gene | **Kept** | Peer-reviewed blink CNN (ICCV-W 2019) with code, weights and dataset. |
| E | MediaPipe Face Landmarker blendshapes (`eyeBlinkLeft/Right`) | https://github.com/google-ai-edge/mediapipe | **Kept** | Official Google model, gives 52 blendshape scores, including 2 for blinks. |
| F | akshaybahadur21/Drowsiness_Detection | https://github.com/akshaybahadur21/Drowsiness_Detection | Dropped | One 1.8 KB script, needs dlib (long compile, a known Jetson problem), no tests, no metrics. It is the same EAR idea as A, but older. |
| G | Layheng-Hok/jetson-nano-drowsiness-detection | https://github.com/Layheng-Hok/jetson-nano-drowsiness-detection | Dropped | Has Jetson Nano claims (15–20 FPS, README only), but it is a whole-face YOLOv5n with classes Alert/Yawn/MicroSleep. There is no per-eye state, so it does not fit the SG2 interface. 1 star, no licence. |
| H | rubendflorezzela/hcbf-driver-monitoring (arXiv 2606.08123) | https://github.com/rubendflorezzela/hcbf-driver-monitoring | Dropped as a reference, **used as evidence** | It is only an analysis repo: no trained weights and no inference code. But its frozen results contain **measured Jetson Nano TensorRT FP32 latency** for 6 eye-state CNNs (see Section 6). |
| I | pathak-ashutosh/Eye-blink-detection | https://github.com/pathak-ashutosh/Eye-blink-detection | Dropped | dlib EAR, last push 2019-07-16. |

## 3. Candidate Screening Template (shortlisted candidates)

| Item | A: Driver-State-Detection | B: OMZ open-closed-eye-0001 | C: DBSE-monitor | D: RT-BENE (rt_gene) | E: MediaPipe blendshapes |
|---|---|---|---|---|---|
| Candidate / repository | github.com/e-candeloro/Driver-State-Detection (commit d932759, 2026-08-18) | github.com/openvinotoolkit/open_model_zoo `models/public/open-closed-eye-0001` (commit a6946b6, 2026-09-21) | github.com/altaga/DBSE-monitor (commit 18cda42, 2021-11-08) | github.com/Tobias-Fischer/rt_gene (last commit 2026-07-14) | github.com/google-ai-edge/mediapipe, `face_landmarker.task` |
| Supporting paper | EAR method: Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks", CVWW 2016 (read). The README also cites a ResearchGate paper on driver attention (**Not Verified**, not opened). | None. Model card only. | None | Cortacero, Fischer, Demiris, "RT-BENE", ICCV Workshops 2019 (read) | Google model cards (blendshape model card **Not Verified**, not opened) |
| End-to-end? | Yes: webcam → landmarks → EAR, gaze, head pose, PERCLOS → alerts. No blink counter. | No: model only, plus a C++ OpenVINO gaze demo. We wrote a short Python wrapper. | Yes, but hardware-bound: Haar face/eye → CNN → alarm. The Jetson app also needs an accelerometer, VLC and Twilio. | Yes: face → landmarks → eye patches → blink (Pixi + ROS 2) | Yes: image → landmarks + blendshapes. Blink events need our own logic. |
| Python/source available? | Yes, Python, MIT | ONNX + C++ demo, Apache-2.0. Training code **Not Verified**. | Yes, Python incl. `train.py`, MIT | Yes, Python, CC BY-NC-SA 4.0 (non-commercial; OK for course) | Yes, Apache-2.0 library. Model training code not public. |
| Jetson/edge evidence? | None in repo. Issue #8 (2023): a user building a similar MediaPipe system reports ~5 FPS on the old Jetson Nano with MediaPipe 0.8.9. | Built for edge (OpenVINO). No Jetson evidence. | Jetson Nano setup folder + hardware photos. No FPS numbers. | None. Pixi platforms are osx-arm64, linux-64, win-64 only (no aarch64). | PyPI has a `manylinux_2_28_aarch64` wheel (CPU). No Jetson numbers. |
| Reported accuracy/metric | None in repo. Paper: EAR methods shown as precision-recall curves on ZJU and Eyeblink8. | 95.84% accuracy on MRL Eye (every 10th image is the test set) | None reported | F1 = 0.721 ± 0.044 (RT-BENE, ensemble); F1 = 0.976 ± 0.018 (Eyeblink8, ensemble) | None found on the docs page |
| Reported FPS/latency | None in repo | 0.0014 GFLOPs, 0.0113 M params (no FPS given) | None | 20.5–42.2 FPS on the authors' desktop (3× GTX 1080, i9-7960X) | None on the docs page |
| Pretrained weights? | Uses Google `face_landmarker.task` (3,758,596 B, SHA-256 pinned). EAR needs no training. | Yes, ONNX 46,164 B, SHA-384 matches model.yml | Yes, `BlinkModel.t7` 227,924 B (51,186 params) | Yes, downloaded from box.com on demand (download checked) | Yes, `face_landmarker.task` |
| Dataset/test data | No labelled data in repo. Demo video only. | MRL Eye: 84,898 images, 37 people, public zip (341,866,898 B) | Eye image zip in repo (4,846 images per README; the source dataset name is **Not Verified**) | RT-BENE on Zenodo (record 3685316, CC BY-NC-SA 4.0) | None |
| Successfully cloned? | Yes (blobless sparse; the full clone stalled because of ~109 MB of demo media) | Yes (sparse); ONNX downloaded | Yes (sparse, model files only; full repo ~593 MB) | No (not attempted) | N/A (pip package) |
| Successfully executed? | **Yes.** Tests: 33/35 pass (2 Linux-Qt GUI tests fail on macOS). Full app headless on 5000 frames. EAR on SG1-format points: 61/61 blinks found. | **Yes.** 60/61 blinks found, 3 false events. Label order had to be fixed (see notes). | **Partly.** CNN only: 60/61 blinks, 1 false event. The original app was not run. | No | **Yes.** 61/61 blinks, 0 false events at threshold 0.4 |
| Major dependency/risk | Main loop is webcam-only (needed a patch). The MediaPipe step is SG1's job, so SG2 only needs numpy. Fixed EAR threshold may not suit every driver. | README says output is `[open, closed]`, but the demo code and our test show index 0 = closed. Accuracy depends on crop size. MRL is infrared and our camera may be RGB. | Old (2021), JetPack 4-era setup, inactive. Unknown training data licence. | ROS 2 + Pixi, no aarch64, heavy two-stream ResNet/DenseNet, non-commercial licence | SG1 must send blendshapes (not in V1). Thresholds are not documented. CPU only on Jetson. |
| Weighted feasibility score /100 | **79** | **71** | **67** | **65** | **67** |
| Decision | **PRIMARY** | **BACKUP** | Extra comparison only | Reject as primary (accuracy reference only) | Optional experiment (Config E) |

## 4. Score breakdown (one line per score)

Weights: E2E 15, Source 15, Jetson 15, Accuracy 10, Reproducibility 10, Weights 10, Orin real-time 10, Data 5, V1 interface 5, Maturity 5.

### A. e-candeloro/Driver-State-Detection: 79/100
| Criterion | Score | Justification |
|---|---|---|
| End-to-end | 14/15 | Full app ran on all 5000 frames. We only had to patch the webcam/GUI calls. |
| Source code | 15/15 | Full Python package, MIT licence, `uv.lock`. |
| Jetson evidence | 5/15 | No official Jetson test. One user issue (a similar MediaPipe system) reports ~5 FPS on the old Nano. |
| Accuracy evidence | 5/10 | The repo reports no metric. The EAR paper and our 61/61 blink test give method-level evidence only. |
| Reproducibility | 9/10 | Clear README, locked deps, checksum-pinned model download, 35 tests. |
| Weights | 9/10 | Uses Google's pinned Face Landmarker model. EAR has nothing to train. |
| Orin real-time | 9/10 | EAR takes 0.025 ms per eye on Mac. Landmarks come from SG1, so the extra cost is close to zero. Not measured on Jetson. |
| Data | 3/5 | No labelled data in the repo. EAR needs no training data. |
| V1 interface | 5/5 | Reads SG1 `left_eye_pts`/`right_eye_pts` with only a point re-order (tested on SG1 mock files). |
| Maturity | 5/5 | 155 stars, commit 2026-08-18, 0 open issues, MIT. |

### B. OMZ open-closed-eye-0001: 71/100
| Criterion | Score | Justification |
|---|---|---|
| End-to-end | 9/15 | Model only. The OMZ pipeline is a C++ OpenVINO demo. Our ~30-line Python wrapper ran on 5000 frames. |
| Source code | 8/15 | ONNX file and C++ demo are open (Apache-2.0). Training code is **Not Verified**. |
| Jetson evidence | 6/15 | Designed for edge devices, but there is no Jetson test. Jetson relies on onnxruntime/TensorRT (not tested). |
| Accuracy evidence | 8/10 | 95.84% on MRL Eye in the model card and accuracy-check config. We found 60/61 blinks. |
| Reproducibility | 6/10 | Input and preprocessing are documented. The README label order is wrong, and there is no training recipe. |
| Weights | 10/10 | 46,164-byte ONNX, checksum matches. |
| Orin real-time | 10/10 | 0.0014 GFLOPs. 0.21 ms per eye on Mac CPU. |
| Data | 5/5 | MRL Eye is public (download checked). |
| V1 interface | 5/5 | Needs the BGR frame + eye points to cut a 32×32 crop, which is already allowed in V1 (Config D). Outputs open/closed. |
| Maturity | 4/5 | 4,431 stars, active commits, but the README says "maintenance mode". |

### C. altaga/DBSE-monitor: 67/100
| Criterion | Score | Justification |
|---|---|---|
| End-to-end | 7/15 | The app needs a webcam, an accelerometer, VLC and Twilio, so we did not run it. The CNN alone ran fine. |
| Source code | 13/15 | Python model, training and app code, MIT. |
| Jetson evidence | 9/15 | Has a Jetson Nano build folder and hardware photos, but no measured FPS. It targets the old Nano, not the Orin. |
| Accuracy evidence | 4/10 | No metric in the repo. Only our own 60/61 blink test. |
| Reproducibility | 5/10 | Notebook-style docs. The setup uses an old JetBot SD image. |
| Weights | 9/10 | `BlinkModel.t7` is in the repo and loads in torch 2.14 with `weights_only=True`. |
| Orin real-time | 9/10 | 51k-parameter CNN, 0.74 ms per eye on Mac CPU. |
| Data | 4/5 | Training zip is in the repo. Its source dataset and licence are **Not Verified**. |
| V1 interface | 5/5 | Takes a 24×24 gray eye crop and outputs Close/Open. |
| Maturity | 2/5 | Last commit 2021-11-08. Open issues are mostly dependabot. |

### D. Tobias-Fischer/rt_gene (RT-BENE): 65/100
| Criterion | Score | Justification |
|---|---|---|
| End-to-end | 9/15 | Full pipeline with demos exists, but we did not run it (needs Pixi + ROS 2). |
| Source code | 13/15 | Python source is available. The licence is non-commercial (CC BY-NC-SA 4.0). |
| Jetson evidence | 2/15 | No Jetson evidence, and the build has no linux-aarch64 platform. |
| Accuracy evidence | 9/10 | The paper reports F1 on 3 datasets with ± values. |
| Reproducibility | 7/10 | Good README and tests, but a heavy ROS 2/Pixi stack. |
| Weights | 8/10 | Downloadable (checked), but hosted on an external box.com link. |
| Orin real-time | 4/10 | Two ResNet/DenseNet streams plus a heavy face detector. 20–42 FPS was on a desktop GTX 1080. |
| Data | 5/5 | RT-BENE dataset is on Zenodo (checked). |
| V1 interface | 4/5 | Takes both eye patches (60×36, ImageNet normalisation) and outputs a blink probability. Needs both eyes at the same time. |
| Maturity | 4/5 | 447 stars, commit 2026-07-14, but the licence is non-commercial. |

### E. MediaPipe Face Landmarker blendshapes: 67/100
| Criterion | Score | Justification |
|---|---|---|
| End-to-end | 13/15 | Official Python API ran on 5000 frames. Blink events need our own logic. |
| Source code | 11/15 | Library is open (Apache-2.0). Model training code is not public. |
| Jetson evidence | 5/15 | An aarch64 wheel exists (CPU). No Jetson numbers. One user report of ~5 FPS with the old MediaPipe. |
| Accuracy evidence | 5/10 | No blink metric found in the docs. Our test: 61/61 at 0.4, 58/61 at 0.5. |
| Reproducibility | 8/10 | Good official docs. |
| Weights | 10/10 | `face_landmarker.task` is public. |
| Orin real-time | 7/10 | SG1 already runs this model. Blendshapes add a small model. Not measured on Orin. |
| Data | 1/5 | No labelled eye-state data. |
| V1 interface | 2/5 | Needs SG1 to send `blendshapes`, which is not in V1 (only an optional field). |
| Maturity | 5/5 | 37k stars, commit 2026-10-07. |

## 5. Ranking and decision

| Rank | Candidate | Score | Decision |
|---|---|---|---|
| 1 | A – e-candeloro/Driver-State-Detection (EAR) | 79 | **PRIMARY** |
| 2 | B – OMZ open-closed-eye-0001 (CNN) | 71 | **BACKUP** |
| 3 | C – DBSE-monitor BlinkModel | 67 | Extra comparison |
| 3 | E – MediaPipe blendshapes | 67 | Optional experiment |
| 5 | D – RT-BENE | 65 | Not primary. Use the paper numbers as a reference. |

(C and E have the same score. C is listed first because it fits V1 without any change.)

**Why A is PRIMARY**
- It uses exactly the input that SG1 gives us: 6 ordered eye points in pixels. We only re-order the points to `[p1, p4, p2, p6, p3, p5]`.
- It is active (commit 2026-08-18), has tests and an MIT licence, and we ran it end-to-end ourselves.
- On Jetson it costs almost nothing, because SG1 already computes the landmarks.
- **Gap:** A has no blink counter. SG2 must add blink and long-closure events (V1 rules). A's thresholds (0.15 default) still need tuning on our data.

**Why B is BACKUP**
- It is a different kind of method (appearance-based CNN). It can help when the landmarks are poor, for example with partly closed eyes or bad lid points.
- It is tiny (46 KB, 0.0014 GFLOPs) and is an ONNX file, so the TensorRT/onnxruntime path on Jetson is simple (**Not Verified** on device).
- **Gap:** use output index 0 = closed and the demo crop size (1.8 × eye width). Both were confirmed in our test.

No candidate hits a hard rejection rule, except D for Jetson use (no aarch64 build).

## 6. Execution evidence (summary – full commands in `execution_log.txt`)

All runs: Mac M4, Python 3.12 venv (mediapipe 1.1.0, OpenCV 5.0.0 headless, numpy 2.5.3, torch 2.14.1, onnxruntime 1.30.0). Video: `talking.avi` (5000 frames, 30 fps). Ground truth: `talking.tag`, 61 blinks. A blink counts as found if a predicted closed run overlaps it (±2 frames).

| Run | Command (short) | Result |
|---|---|---|
| A tests | `PYTHONPATH=. python -m pytest -q` | `2 failed, 33 passed`. The 2 failures are Linux/KDE Qt GUI tests. EAR + scorer tests: `8 passed`. |
| A model | `python -m driver_state_detection.download_model` | `Model ready`, SHA-256 `64184e22…c9ff` matches the pinned value |
| A full app | `run_dsd_headless.py` (patched webcam + imshow) | `frames_read=5000 frames_with_EAR=5000 … approx_fps_full_pipeline=52.7` (Mac) |
| A on SG1 points | `sg2_interface_test.py` | EAR<0.15: 61/61 found, 1 false event. EAR<0.20: 61/61 found, 2 false events. 0.025 ms per eye. |
| A on SG1 V1 mock | `dsd_on_sg1_mock.py` | alert 16 events (16 scripted); drowsy 17 (14 blinks + 3 long closures); face_lost 14 (15 scripted, see log) |
| B ONNX | `curl …/open_closed_eye.onnx` | 46,164 B, SHA-384 matches model.yml |
| B run (README label order) | `sg2_interface_test.py` | 97% of frames "closed" → order is inverted |
| B run (index 0 = closed, crop 1.8×) | `sg2_interface_test.py … 1.8` | 60/61 found, 3 false events, 0.21 ms per eye |
| B run (crop 1.2×) | `sg2_interface_test.py … 1.2` | 51/61 found, 8 false events → sensitive to crop size |
| C CNN | `dbse_blinkmodel_test.py` | `All keys matched`, 51,186 params, 60/61 found, 1 false event, 0.74 ms per eye |
| E blendshapes | `mp_blendshape_test.py` | 52 blendshapes incl. `eyeBlinkLeft/Right`. >0.4: 61/61, 0 false. >0.5: 58/61. |

**Patching we did:** only in our own wrapper scripts, at runtime. We replaced `cv2.VideoCapture(0)` with the video file, made `cv2.imshow` / `waitKey` / `destroyAllWindows` do nothing, and wrapped `get_EAR` to log values. No repo file was changed.

**Limits of this test:** one person, frontal face, good light, no glasses. The thresholds come from the repo or the paper and were not tuned, but the test is still easy. These results show that the methods **work**. They do not show how accurate the methods are in a car.

**Extra Jetson evidence (from candidate H, read only):** `results/frozen/figure_data/deployment_summary.csv` in hcbf-driver-monitoring reports Jetson Nano TensorRT FP32 latency for eye-state CNNs. Batch-1 means are ShuffleNetV2 7.62 ms and MobileNetV3-Large 13.59 ms. The paper abstract says only these two met the 33.333 ms binocular-pair deadline. Our candidate B has about 100× fewer parameters (0.0113 M vs 1.26 M for ShuffleNetV2), and C has about 25× fewer (0.051 M). So they should be faster. **This is an inference, not a measurement.**

## 7. Verification notes

**Checked (with `gh api`, curl, PDF text or our own runs):**
- Stars, forks, licence, last commit and open issues for every repo in the long-list.
- READMEs of A, B, C, D, F, G, H. Source of A (`eye_detector.py`, `main.py`, `parser.py`), C (`model.py`, Jetson `notebook.py`), D (blink model and download code), and B (`eye_state_estimator.cpp/.hpp`).
- B: model.yml (size, checksum, URL, licence), README (95.84%, MRL), accuracy-check.yml.
- Soukupová & Čech 2016 PDF: EAR + SVM, datasets ZJU and Eyeblink8, EAR threshold baseline 0.2.
- RT-BENE paper PDF: Tables 2 and 3 (F1, FPS) and the hardware used.
- arXiv 2606.08123 abstract, plus the hcbf repo CSVs (Jetson Nano latency, no checkpoints released).
- Downloads work: MRL zip, face_landmarker.task, OMZ ONNX, RT-BENE ResNet18 weights (partial GET), RT-BENE Zenodo record.
- PyPI aarch64 wheels: mediapipe 1.1.0, onnxruntime 1.30.0 (cp312), opencv-contrib-python 5.0.0.93.

**Not Verified:**
- Any run on a Jetson (Nano or Orin Nano): FPS, latency, memory, TensorRT conversion of B or C, and installing mediapipe/onnxruntime GPU on JetPack.
- The training code for B (OMZ points to openvino_training_extensions, which has moved).
- The source dataset and licence of C's training zip. The content of C's YouTube demo videos.
- The content of the ResearchGate paper cited by A. The blendshape model card for E.
- Whether the MRL Eye dataset has a formal licence (the page only says it is publicly available).
- RT-BENE end-to-end run (D was not executed).
- Accuracy of any candidate on real in-car, night, glasses or infrared video.
