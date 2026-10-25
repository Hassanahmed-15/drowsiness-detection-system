# Interface & dependency tracker — Driver Drowsiness Monitoring

**Started by SG2 on 25 Oct 2026. Every subgroup should update its own rows** (open a PR, or edit and commit on your module branch).
Status is taken **only from what is in this repository** on 25 Oct 2026. Anything not in the repo is marked *not in repo*. Update a row when your interface changes, and record the change in your module README.

## 1. Interfaces (who sends what to whom)

| # | From → To | Interface | Defined where | Status (25 Oct) | Open issue |
|---|---|---|---|---|---|
| I1 | Camera → SG1 | BGR frame + `frame_id` + `timestamp_ms` | SG1 README (PR #1) | documented, PR not merged | — |
| I2 | SG1 → SG2, SG3 | SG1 V1 face block: `face_valid`, `left_eye_pts`/`right_eye_pts` (6 ordered points), `mouth_pts`, ROI boxes, `head_pose`, full-frame pixels | SG1 README (PR #1) + `landmark_map.md` (PR #2) | documented, **PRs #1–#3 not merged** | SG1 to merge into `module_sg1` |
| I3 | SG2 → SG4 | `sg2-v1` per-frame eye record (`eye_state`, EARs, `closed_duration_ms`, `blink_event`, counters) | `interfaces/sg2/README.md` + JSON schema + mocks | documented, tested, **waiting for SG4 confirmation** | #7 |
| I4 | SG3 → SG4 | mouth / yawn packet (`mouth` block, `yawn_prob`, `cue_flags`) | SG3 `gate.py` refers to "interface spec V1, section 6" | partly in code (`module_sg3`). **The spec document is not in the repo** | SG3 to add their spec section to `interfaces/` |
| I5 | SG4 → SG5 | temporal indicators. SG5's code expects `valid`, `window_valid_fraction`, `perclos`, `longest_closure_ms`, `current_closure_ms`, `yawn_count` | SG5 mock inputs (branch `alveenashahidd-patch-1-1`, PR #4) | SG5 assumes this. **SG4 has no code or README in the repo** | SG4 to confirm the fields |
| I6 | SG5 → alert | `decision_valid`, `state`, `drowsiness_score`, `alarm_flag`, `alarm_type` | SG5 `decision_rules` (branch) | in code, not merged | — |
| I7 | all → SG6 | Jetson runtime (Python, OpenCV with GStreamer, onnxruntime, MediaPipe) | `module_sg2/docs/jetson_deployment.md` (SG2's needs) | **SG6 folder empty** | #24 |

How SG2's output feeds SG5's expected inputs, through SG4:

| SG5 expects (from SG4) | Can be computed from `sg2-v1` |
|---|---|
| `perclos` | share of `eye_state == "closed"` frames in SG4's window |
| `current_closure_ms` | `closed_duration_ms` (SG2 gives it directly, every frame) |
| `longest_closure_ms` | max `blink_event.duration_ms` in the window (+ the ongoing `closed_duration_ms`) |
| `window_valid_fraction` | share of `eye_valid == true` frames in the window |
| `yawn_count` | from SG3, not SG2 |

## 2. Dependencies and blockers

| Blocked item | Needs | From | Since | Effect if still open | Workaround in place |
|---|---|---|---|---|---|
| SG2 real-input integration | SG1 module merged, with the same 6-point order | SG1 | 7 Oct | SG2 tested with the SG1-compatible source only | `module_sg2/tools/sg1_landmarks.py` uses SG1's model and indices; contract test checks the order |
| SG2 → SG4 handover | SG4 confirms `sg2-v1` (#7) | SG4 | 7 Oct | SG4 may expect different field names | mock streams published (`interfaces/sg2/mock/`) |
| SG5 inputs | SG4 output definition | SG4 | — | SG5 is built on assumptions | SG5 mock inputs |
| Jetson deployment of every module | board set up, OS/storage/stack verified, camera tested | SG6 | — | Part C of the midterm (5 marks) at risk | SG2 scripts `check_env.py` and `camera_check.py` are ready to run on the board |
| Optional V1.1 `blendshapes` field (2 floats) | SG1 agreement | SG1 | 7 Oct | SG2 Configs E/F cannot be used | primary G does not need it (#18) |
| Team V1 interface spec document | copy committed to `interfaces/` | team lead | — | modules cannot be checked against it in the repo | each module documents its own I/O |

## 3. Merge status (25 Oct)

| Branch / PR | Into | State |
|---|---|---|
| PR #1–#3 (SG1) | `main` | open since 7 Oct |
| PR #4 (SG5) | `main` | open since 7 Oct |
| PR #5 (SG2 Week 5) + `sg2/midterm` | `module_sg2` | open. Waiting for review |
| `module_sg3` (gate.py) | — | on its branch, no PR |
| `main` | — | contains only the initial structure (no module merged yet) |
