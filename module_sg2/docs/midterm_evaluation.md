# SG2 — Midterm Project Evaluation (Stage 1, Week 8): evidence and demo script

**Subgroup:** SG2, Eye State & Blink Analysis · **Project:** Driver Drowsiness Monitoring · **Student:** Hassan Ahmed (@Hassanahmed-15)
**Code:** branch `module_sg2` (merged from `sg2/w5-eye-state-experiments` and `sg2/midterm`) · **Board:** https://github.com/users/Hassanahmed-15/projects/2 · **Prepared:** 25 Oct 2026

Every claim below links to a file in the repository that can be re-run. All numbers come from `results/midterm/` and `results/`.

---

## 1. Rubric → evidence map

### A. Subgroup algorithm selection & working demonstration (15)

| Criterion (marks) | What we show | Evidence |
|---|---|---|
| **Working prototype, live or saved inputs, observable correct output (7)** | `tools/demo.py` on **(a)** the live webcam, **(b)** real test data with hand labels, **(c)** mock SG1 input. The window shows the eye state, EAR vs threshold, the CNN check, blink counts, the eye crops and a **running agreement score with the ground truth** | Real 20 s sample: 8/8 blinks, F1 0.98. Full TEST video eb8_08 (10,663 frames): 99.8% agreement, F1 0.89. Mock drowsy: 14 blinks + 3 long closures, 100%. Live webcam: 29.5 fps end-to-end, face found 100% of frames |
| **Primary + backup with comparative rationale/evidence (4)** | Primary **G** (adaptive EAR + CNN check), backup **C** (adaptive EAR). 8 configurations compared on the same TEST videos with 95% bootstrap CIs and paired tests | `results/midterm/comparison.csv`, `paired_vs_C.csv`, `figures/confidence_intervals.png`; reference screening `docs/reference_screening/` |
| **V1 I/O contract, reproducibility, module role (2)** | SG1 V1 input is unchanged; the `sg2-v1` output has a JSON schema; contract tests; models downloadable with checksums; tests pass on Python 3.12 and 3.10 | `interfaces/sg2/README.md`, `tests/test_contract.py` (24 passed), `models/download_models.sh` |
| **Preliminary testing/results + limitations/failure cases (2)** | TEST metrics, difficult cases (glasses, head pitch, low light), failure montage, limitations list | `results/midterm/difficult_cases.csv`, `results/figures/failure_cases.png`, README "Current Limitations" |

### B. GitHub repository & project management (10, whole team; SG2's part)

| Criterion (marks) | SG2's contribution | Evidence |
|---|---|---|
| Shared repo, module structure, meaningful commits (3) | `module_sg2/`, `interfaces/sg2/`, `datasets/sg2/`. Feature branches `sg2/w5-eye-state-experiments` → `sg2/midterm`, merged into `module_sg2` by PR. One commit per technical step | `git log --oneline origin/module_sg2` |
| Project board: owners, status, progress, blockers (3) | Board in the Week 5 structure: Backlog → To Do → In Progress → Review/Blocked → Done. Titles `W# \| SG# \| task`. Every item has an owner and Week / Sub-group / Type labels (except #41, which waits for the group leader to name an SG6 owner). Blocker and Evidence fields. Views: Team board, SG2 board, Blockers, Week 8. Milestones W5/W6/W8 show % done. A status update lists each subgroup's blockers | [Team board](https://github.com/users/Hassanahmed-15/projects/2/views/2) · [Blockers](https://github.com/users/Hassanahmed-15/projects/2/views/4) · [milestones](https://github.com/Hassanahmed-15/drowsiness-detection-system/milestones) |
| Module README, deps, run instructions, evidence links (2) | Rewritten for the midterm | `module_sg2/README.md` |
| Coordination: meeting record, decisions, interface/dependency tracking (2) | SG2 decision log; team interface/dependency tracker; meeting record template (**the team must fill in a real meeting**) | `docs/decision_log.md`, `documentation/coordination/` |

### C. Jetson Orin Nano & camera readiness (5, SG6 leads; SG2 supports)

| Criterion (marks) | SG2's contribution | Status |
|---|---|---|
| Jetson boots, OS/storage identified (2) | `tools/check_env.py` reports board, L4T/JetPack, NVMe vs microSD, free space, RAM, power mode | **Run on the team's Jetson** (SG2 has no board access) |
| Stack documented: Python, Git, OpenCV, CUDA/PyTorch (1) | Same report, including whether OpenCV has GStreamer. Jetson install guide with the verified MediaPipe/OpenCV trap and onnxruntime 1.23.2 for Python 3.10 | `docs/jetson_deployment.md` |
| Camera/video acquisition demonstrated, blockers + next actions (2) | `tools/camera_check.py` for USB / CSI / file: measured FPS, frames, face rate, SG2 on every frame | Verified on the dev PC webcam (29.7 fps). Jetson run pending (issue #24) |

---

## 2. Algorithm selection (the "why", in 60 seconds)

All configs use the **same** SG1-compatible landmarks on the **same** 5 TEST videos (852 closed frames, 217 blinks). Thresholds were tuned on DEV videos only.

| Config | Idea | F1 closed [95% CI] | False alarms | Blink F1 | V1 only? | Verdict |
|---|---|---|---|---|---|---|
| A | EAR < 0.20 (Week 4 baseline) | 0.361 [0.28–0.47] | 3,020 | 0.857 | ✅ | fails: one threshold doesn't fit all eye shapes |
| B | EAR < 0.13 (tuned) | 0.833 [0.71–0.95] | 293 | 0.934 | ✅ | better, still driver-dependent |
| **C** | **adaptive EAR (per-driver)** | **0.918 [0.84–0.96]** | 80 | 0.944 | ✅ | **backup** |
| D | CNN v1 only | 0.872 [0.84–0.90] | 17 | 0.901 | ✅ | misses 21% of closed frames |
| D2 | CNN v2 only | 0.699 [0.57–0.84] | 718 | 0.789 | ✅ | over-calls "closed" on new people |
| E | MediaPipe eyeBlink | 0.945 [0.91–0.97] | 62 | 0.943 | ❌ | needs a new SG1 field |
| F | C + eyeBlink fusion | 0.933 [0.88–0.97] | 68 | 0.949 | ❌ | needs a new SG1 field |
| **G** | **C, confirmed by CNN v2** | **0.947 [0.93–0.96]** | **21** | **0.951** | ✅ | **primary** |

- **G vs C:** false alarms −74%, and with the head ≥ 10° from normal pitch they fall 70 → 12. Worst video eb8_08: 0.70 → 0.89. P(G better than C) = 0.79 in the paired bootstrap, so the gain is likely but not conclusive. G therefore falls back to C automatically.
- **Backup feasibility:** C is G without the CNN step. It needs no model and has a proven F1 of 0.918. It is the guaranteed path on the Jetson if onnxruntime is a problem.
- **External evidence:** the reference screening ranked an EAR implementation first (79/100) and a small eye-crop CNN second (71/100). That matches our choice of geometry first, CNN second.

---

## 3. Demo script (about 8 minutes)

**Before the session:** run `bash models/download_models.sh` and `python -m pytest tests -q` (expect `24 passed`). Give the terminal or VS Code camera permission (macOS: System Settings → Privacy & Security → Camera). Close other camera apps.

| # | Say | Run (in `module_sg2/`) | What the evaluator should see |
|---|---|---|---|
| 1 | "SG2 turns SG1's 6 eye points into open/closed and blink events for SG4." | open `README.md` diagram | role in the pipeline |
| 2 | "Our input/output contract is tested automatically." | `python -m pytest tests -q` | `24 passed` |
| 3 | "Real test data, never used for tuning, checked against hand labels." | `python tools/demo.py --sg1-jsonl ../interfaces/sg2/examples/eb8_09_real_sg1.jsonl` | 8 blinks, "correct" marks, agree ≈ 99.6%, F1 0.98 |
| 4 | "Live: I blink, then close my eyes for 2 seconds." Face the camera; keep eyes open for the first second (calibration) | `python tools/demo.py --source 0` | OPEN/CLOSED follows the eyes, blink count rises, the 2 s closure is reported as a **long closure** |
| 5 | "Without SG1 or a camera, using the agreed mock input." | `python tools/demo.py --sg1-jsonl ../interfaces/sg2/mock/mock_sg1_drowsy.jsonl` | 14 blinks + 3 long closures, agree 100% |
| 6 | "Why G is primary and C is backup." | open `results/midterm/figures/confidence_intervals.png` | G has the highest F1 and the narrowest CI |
| 7 | "Where it fails, and what we fixed." | open `results/midterm/figures/difficult_cases.png`, `results/figures/failure_cases.png`, `results/occlusion/figures/niqab_f1.png` | head pitch, glasses, low light, half-closed frames. **Niqab:** the face was lost at confidence 0.5; at 0.2 eyes are found in 98–100% of frames |
| 8 | "The output SG4 receives." | `python tools/run_sg2.py --sg1-jsonl ../interfaces/sg2/examples/eb8_09_real_sg1.jsonl --config configs/primary.yaml --out /tmp/sg2.jsonl && head -3 /tmp/sg2.jsonl` | `sg2-v1` JSON records |

Optional, if the dataset is downloaded: `python tools/demo.py --source ../datasets/raw/sg2/eyeblink8/eyeblink8/8/27122013_151644_cam.avi` (the hardest TEST video, with hand labels).

**If the camera fails in the room:** use steps 3 and 5 (the brief accepts saved inputs and mock upstream data), and say that `tools/camera_check.py` shows the webcam path working at 29.7 fps on this laptop.

---

## 4. Known limitations (say them before you are asked)
0. **Niqab / face covering:** SG2 needs SG1 to find the face. Under a simulated niqab, SG1's confidence 0.5 lost 28% of the frames on the glasses video. Our live tools use 0.2, which loses 1.7% with the same eye-point accuracy. Primary F1 is 0.91 and blink F1 0.93 with a niqab ([`results/occlusion/`](../results/occlusion/)). Real fabrics and lighting are not tested yet.
1. Not yet run on the Jetson. The scripts are ready (§1C) and the code passes all tests with the Jetson's Python 3.10 / onnxruntime 1.23.2.
2. TEST has only 4 people (1 with glasses), recorded indoors. Low light is synthetic.
3. G still misses ~8% of closed frames (mostly half-closed frames at blink edges). Blink recall is 93%.
4. The adaptive threshold needs about 1 s of open eyes to calibrate.
5. The G vs C improvement is likely (79%) but not statistically conclusive on 5 videos.
6. The SG4 sign-off on `sg2-v1` is pending. SG4 has not pushed code yet, so SG2 published mock streams for them.

## 5. Integration plan (Stage 2)
| Lab | SG2 action | Depends on |
|---|---|---|
| Lab 8 (first multi-module integration) | Replace the SG1-compatible source with SG1's real output. Check the 6-point order on 100 frames (contract test) | SG1 merges its module and output format |
| Lab 8 | SG4 consumes `sg2-v1` (mock streams already published) | SG4 confirms the interface (#7) |
| Lab 9 (individual Jetson deployment) | Run `check_env.py`, `camera_check.py --source csi --pipeline` and `demo.py --no-window --save` on the board. Record latency | SG6 board setup (#24) |
| Lab 11 (full embedded pipeline) | SG2 runs inside the team pipeline at ≥ 30 fps | all modules |
