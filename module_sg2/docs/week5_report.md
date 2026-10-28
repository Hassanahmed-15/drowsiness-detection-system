# SG2 — Eye State & Blink Analysis: Week 5 report

**Project:** Driver Drowsiness Monitoring (Stream B) · **Sub-group:** SG2 · **Date:** 7 October 2026
**Branch:** `sg2/w5-eye-state-experiments` · **Results:** [`results/`](../results/) · **Selected config:** [`configs/w6_selected.yaml`](../configs/w6_selected.yaml)

---

## 1. Baseline and task progress (rubric: working baseline)

**What SG2 does.** SG1 sends the driver's eye points for every frame. SG2 decides whether the eyes are **open or closed**. It turns runs of closed frames into **blink** events (≤ 500 ms) or **long-closure** events (> 500 ms). It sends one `sg2-v1` record per frame to SG4.

| | |
|---|---|
| **Input** | SG1 V1 block: `left_eye_pts`, `right_eye_pts` (6 ordered points, full-frame pixels), `face_valid`, `frame_id`, `timestamp_ms`. The frame is used only by the CNN method |
| **Processing** | EAR per eye → open/closed decision (the method is chosen in a YAML config) → blink / long-closure detector |
| **Output** | `sg2-v1` record: EARs, `eye_state`, `closed_duration_ms`, `blink_event`, counters. See [`interfaces/sg2/README.md`](../../interfaces/sg2/README.md) |

**Baseline (Config A, the Week 4 reference).** Eye Aspect Ratio (Soukupová & Čech 2016), averaged over both eyes, closed if EAR < **0.20**. The Week 4 code was not in the shared repository, so it was re-created as the module [`sg2_eye/`](../sg2_eye/) with the textbook settings. It runs end-to-end: `tools/run_sg2.py` takes a video or SG1 JSONL in and gives SG2 JSONL out.

**Known baseline limitation, now measured.** A single fixed threshold does not fit every driver. In 4 of the 5 TEST videos the normal open-eye EAR is only **0.225–0.26**, against 0.28–0.31 in the DEV videos ([`dataset_summary.csv`](../results/dataset_summary.csv)). The 0.20 threshold therefore calls many open eyes "closed": 9.1% false-closed rate, precision 0.22.

## 2. Week 5 experiment (rubric: alternative / parameter study)

Five configurations go through the same module and the same blink detector. Only the open/closed rule changes.

| Config | Method | What changes vs baseline |
|---|---|---|
| **A** | EAR, fixed 0.20 | Week 4 baseline |
| **B** | EAR, fixed, **threshold tuned** (sweep 0.08–0.30 on DEV → **0.13**) | parameter study |
| **C** | EAR, **adaptive per-driver threshold** = k × the driver's own open-eye EAR (80th percentile of the last 30 s, causal). Tuned **k = 0.425** | new method, same V1 input |
| **D** | **Learned CNN** on a roll-corrected 32×32 grey eye crop. 72k parameters, trained on MRL Eye (by person) + CEW, exported to ONNX. P(closed) threshold tuned: 0.995 | learned classifier vs geometric measure |
| **E** | MediaPipe Face Landmarker **eyeBlink blendshape** score. Threshold tuned: 0.425 | learned score. **Needs an extra SG1 field (not in V1)** |

Also tested: **pose-aware eye fusion** (each eye weighted by its visible width) against a plain average ([`fusion_study.csv`](../results/fusion_study.csv)). The CNN was also checked on the 6 held-out MRL people, with labels for glasses, reflections and lighting.

## 3. Quantitative evidence (rubric: metrics, common test conditions)

**Test data (public, frame-annotated).** Eyeblink8 (8 webcam videos, 4 people, one with glasses) + Talking Face. Labels come from the official `.tag` files: *closed* = annotated fully closed; *open* = outside every blink; half-closed frames are ignored.
- **DEV** (tuning only): Talking Face, Eyeblink8 videos 1, 2, 3 → 41,182 frames, 252 blinks
- **TEST** (reporting only): Eyeblink8 videos 4, 8, 9, 10, 11 → 35,174 frames, **852 closed / 33,040 open frames, 217 blinks**
- **TEST low-light:** the same TEST videos, darkened (γ = 2, gain 0.6, noise σ = 3). Median face brightness drops from about 103–114 to about 27–37

**Common conditions.** Landmarks were extracted **once** with the SG1-compatible MediaPipe source (same model and indices as SG1) and cached. Every config sees identical inputs. Thresholds were tuned on DEV, then frozen. The final numbers come from replaying the real `EyeStateModule`.

**Metrics.** Closed frames are only 2.5% of the data, so we report precision / recall / **F1 of the closed class**, the **false-closed rate** (open frame called closed, which causes false alarms later) and the **false-open rate** (closed frame missed, which causes missed alarms later). For blinks we report event precision / recall / **blink F1**: a detected closure must overlap an annotated blink (±2 frames, one-to-one).

### 3.1 Main result — TEST set ([`week5_comparison.csv`](../results/week5_comparison.csv))

| Config | Precision | Recall | **F1 (closed)** | False-closed | False-open | Blinks found / events | **Blink F1** | SG2 ms/frame* |
|---|---|---|---|---|---|---|---|---|
| A baseline EAR 0.20 | 0.220 | 1.000 | 0.361 | 9.14% | 0.0% | 213 / 280 | 0.857 | 0.012 |
| B EAR tuned 0.13 | 0.736 | 0.959 | 0.833 | 0.89% | 4.1% | 206 / 224 | 0.934 | 0.012 |
| **C EAR adaptive k=0.425** | 0.908 | 0.927 | **0.918** | 0.24% | 7.3% | 202 / 211 | **0.944** | 0.040 |
| D CNN p>0.995 | 0.975 | 0.789 | 0.872 | 0.05% | 21.1% | 187 / 198 | 0.901 | 0.464 |
| E blendshape p>0.425 | 0.930 | 0.961 | **0.945** | 0.19% | 3.9% | 206 / 220 | 0.943 | 0.012 |

\*SG2 time only, Apple M4 CPU, single thread. It excludes the landmark step, which takes about 4.1 ms/frame and belongs to SG1. Jetson numbers are still to be measured by SG6.

"Blinks found / events" = annotated blinks matched / closure events SG2 reported. TEST has 217 annotated blinks.

### 3.2 Difficult cases — F1 of closed frames on TEST ([`difficult_cases.csv`](../results/difficult_cases.csv))

| Condition (closed frames) | A | B | C | D | E |
|---|---|---|---|---|---|
| Glasses — eb8_11 (144) | 0.223 | 0.691 | **0.966** | 0.898 | 0.902 |
| No glasses (708) | 0.412 | 0.871 | 0.907 | 0.867 | **0.955** |
| \|yaw\| < 10° (706) | 0.360 | 0.844 | 0.923 | 0.900 | **0.947** |
| \|yaw\| 10–20° (111) | 0.305 | 0.729 | 0.863 | 0.671 | **0.918** |
| \|yaw\| ≥ 20° (35, small sample) | 1.000 | 0.971 | 0.986 | 0.833 | 1.000 |
| **\|pitch\| ≥ 15°** (113) | 0.153 | 0.594 | 0.753 | 0.864 | **0.907** |
| Normal light (852) | 0.361 | 0.833 | 0.918 | 0.872 | **0.945** |
| Low light, synthetic (852) | 0.350 | **0.921** | 0.891 | 0.840 | 0.895 |

Head pose comes from SG1-style solvePnP on 6 landmarks (an approximate estimate).

**CNN on held-out MRL people (infrared)** ([`cnn_mrl_difficult_cases.csv`](../results/cnn_mrl_difficult_cases.csv)): accuracy 98.1% overall, F1 0.98. With glasses F1 is 0.975. With **small reflections** recall drops to 0.73. With bad lighting F1 is 0.984.

**Pose-aware fusion:** F1 is 0.8328 with a mean and 0.8325 width-weighted on all TEST. For |yaw| ≥ 20° it is 0.971 vs 0.986, but on only 35 closed frames. That is no real gain, so it is **not adopted**.

### 3.3 Figures ([`results/figures/`](../results/figures/))
- `ear_threshold_sweep.png`: F1 and error rates against the EAR threshold, DEV and TEST. Shows why 0.20 is too high for these drivers.
- `precision_recall_test.png`: precision–recall curve of every method on TEST.
- `blink_timeline_eb8_04.png`: EAR, adaptive threshold and CNN P(closed) over 20 s, with the annotated blinks shaded (blink evidence).
- `difficult_cases_f1.png`: the table in §3.2 as a chart.
- `failure_cases.png`: eye crops of false-open / false-closed frames for B and D.
- `eye_crop_examples.png`: our video crops next to the MRL/CEW training crops.

## 4. GitHub and interface readiness

- Code is in `module_sg2/` on branch `sg2/w5-eye-state-experiments`, with incremental commits (baseline → interface → alternatives → data/tests → experiment → screening → docs). It merges through a PR into `module_sg2`, then `main`.
- **V1 input respected:** SG2 uses only SG1 V1 fields, with SG1's exact landmark indices and point order. SG1's interface does not change. Config E is kept only as a **V1.1 proposal** for SG1 to decide on.
- **Output interface for SG4:** [`interfaces/sg2/README.md`](../../interfaces/sg2/README.md) + JSON schema. **Mock streams** (alert / drowsy / face_lost, with ground truth) are in `interfaces/sg2/mock/`, so SG4 can work without SG1/SG2.
- **Reproducible:** `datasets/sg2/download_sg2_data.sh` → `experiments/prepare_videos.py` → `experiments/train_cnn.py` → `experiments/evaluate.py`. The unit tests (`tests/test_sg2.py`, 8 checks) need no data. Configs and results are versioned with the code.

## 5. Analysis — failures and trade-offs

1. **Fixed thresholds do not transfer between drivers.** Open-eye EAR ranges from 0.225 to 0.311 between drivers (it depends on eye shape and camera angle). The 0.20 threshold suits DEV drivers but floods TEST with false closures (3,020 false-closed frames). Tuning on DEV (B, 0.13) fixes most of this. Per-driver adaptation (C) fixes more: it cuts false-closed frames from 293 to 80 and lifts glasses-wearer F1 from 0.69 to 0.97.
2. **Trade-off.** C trades a few more missed closed frames (false-open 7.3% vs 4.1% for B) for far fewer false closures (0.24% vs 0.89%). False closures are what would cause false drowsiness alerts downstream, so this is the better side to err on. Blink F1 rises to 0.944.
3. **Looking down (|pitch| ≥ 15°) is the main weakness of every EAR method** (C: 0.753). From above, the upper lid covers the eye and the 2-D EAR shrinks even when the eye is open. The appearance-based methods cope better (D 0.864, E 0.907). In the failure montage most B false-closed crops are downward gaze or glasses frames.
4. **The CNN suffers from domain shift.** It scores 98% on held-out MRL (infrared) people but only 0.872 F1 on webcam video. It treats narrowed eyes as closed, so its threshold had to go up to 0.995, which then misses 21% of closed frames. It is also about 10–40× slower than EAR (still only 0.46 ms). It could improve by fine-tuning on webcam crops, but it is not selected now.
5. **Low light** (synthetic): the landmarks get noisier and closed-eye EAR rises (per-video median 0.05–0.07 → 0.07–0.10). Recall drops for C (0.927 → 0.812) and E. B gains precision because its threshold is low. No method fails completely. The face was still found in ≥ 99% of frames.
6. **E is the most accurate overall** (F1 0.945, best on pose). But it depends on SG1 sending two extra numbers per frame, which is an interface change that SG1 must agree to.
7. **Label noise.** Some "false-closed" crops look really closed (for example eb8_08#3427, EAR 0.08). The Eyeblink8 annotation marks only *fully* closed frames inside blinks, so part of the remaining error is ambiguity in the ground truth.

## 6. Week 6 decision

- **Selected: Config C, adaptive EAR** ([`configs/w6_selected.yaml`](../configs/w6_selected.yaml)). It is the best configuration that keeps the frozen V1 interface (F1 0.918, blink F1 0.944). It handles glasses best, needs no model file and costs 0.04 ms/frame.
- **Week 6 rigorous comparison:** C vs E, plus a **C + E fusion**. Report per-video spread / bootstrap confidence intervals on TEST, not one pooled number.
- **Action for the pitch weakness:** use SG1's `head_pose.pitch_deg` to lower `confidence` (or switch to E) when |pitch| ≥ 15°.
- **Interface:** ask SG1 whether `blendshapes` (2 floats) can be added as an optional V1.1 field. Until then E stays experimental. *Status: blocked on SG1's reply.*
- **With SG6:** measure C (and the D ONNX model as a backup) on the Jetson Orin Nano.
- **Not continued:** pose-aware width fusion (no gain). CNN (D) stays as the backup only, with a fine-tune on webcam crops as an optional item.

## 7. Current limitations
- TEST has 4 people (one with glasses). There is no real in-car, night or infrared footage yet. Low light is synthetic.
- The head-pose bins rely on an approximate pose estimate. The |yaw| ≥ 20° bin has only 35 closed frames.
- The adaptive threshold needs about 1 s of open-eye frames to warm up. Before that it uses the fixed 0.13 threshold.
- No Jetson measurement yet. All timings are from an Apple M4 CPU.
- The Week 3 metric sheet was not in the repository. The metrics above (closed-class precision/recall/F1, false-closed/false-open rate, blink F1) should be checked against it.

## 8. Reference implementations (bonus)
See [`docs/reference_screening/reference_screening.md`](reference_screening/reference_screening.md). It scores 5 candidates on the 100-point criteria and cloned and executed the top ones. **Primary:** e-candeloro/Driver-State-Detection, EAR-based (79/100). **Backup:** OpenVINO `open-closed-eye-0001` CNN (71/100). This agrees with our own experiment: geometric EAR is selected and a small eye-crop CNN is the backup.

## Data sources
Eyeblink8 and Talking Face blink annotations: Drutarovsky & Fogelton (blinkingmatters.com). Talking Face video: FGNet / T. Cootes. MRL Eye Dataset: Fusek, VŠB Ostrava. CEW: Song et al., Pattern Recognition 2014. All are used for research/teaching only. Raw data is not committed.
