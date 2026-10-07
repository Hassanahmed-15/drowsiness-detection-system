# SG2 Interface — Eye State & Blink Analysis (`sg2-v1`)

**Owner:** SG2. **Consumers:** SG4 (temporal analysis), SG5 (via SG4), SG6 (integration).
**Status:** V1 proposed in Week 5. SG4 should confirm it. Any change after confirmation needs SG4's agreement.

```
SG1 face block (per frame)  ──►  SG2 EyeStateModule  ──►  SG2 eye-state record (per frame)  ──►  SG4
```

## 1. Input from SG1 (V1, unchanged)

SG2 reads only these SG1 V1 fields. Everything else in the block is ignored.

| SG1 field | Used for |
|---|---|
| `frame_id`, `timestamp_ms` | copied to the output; blink timing |
| `face_valid` (falls back to `face_detected`) | if false, the eye state is `unknown` |
| `left_eye_pts`, `right_eye_pts` | 6 ordered points each: p1 outer, p2/p3 upper lid, p4 inner, p5/p6 lower lid. Full-frame pixels. Driver's own left/right ([SG1 landmark map](../../module_sg1/)) |
| BGR frame | **only** for the CNN method (Config D). SG2 cuts its own roll-corrected 32×32 eye crop from the eye points |

SG2 does **not** need `landmarks`, `mouth_pts`, ROI boxes or `head_pose`. Head pose is used only offline, to sort test results by difficulty.

**Optional, not part of V1:** `blendshapes: {"eyeBlinkLeft": float, "eyeBlinkRight": float}`. This is needed only by the experimental Config E. SG2 does not depend on it.

## 2. Output to SG4 (`sg2-v1`, one record per frame)

Schema: [`sg2_eye_state_v1.schema.json`](sg2_eye_state_v1.schema.json)

| Field | Type | Meaning |
|---|---|---|
| `interface_version` | `"sg2-v1"` | version tag |
| `frame_id`, `timestamp_ms` | int, float | copied from SG1 |
| `eye_valid` | bool | false = no usable eye points this frame |
| `method` | str | SG2 config that made the record |
| `left_ear`, `right_ear`, `ear` | float / null | Eye Aspect Ratio. Always reported, whatever the method |
| `left_eye_state`, `right_eye_state`, `eye_state` | `open` / `closed` / `unknown` | `eye_state` is the fused decision SG4 should use |
| `closed_score` | float / null | the method's raw score (higher = more closed) |
| `threshold` | float / null | threshold used on this frame |
| `confidence` | 0..1 / null | how far the score is from the threshold |
| `closed_duration_ms` | float | how long the eyes have been closed **so far** (0 when open). SG4 can react to a long closure before the eyes reopen |
| `blink_event` | null / object | set only on the frame where a closure **ends**: `{type: blink \| long_closure, start_frame, end_frame, start_ms, end_ms, duration_ms}` |
| `blink_count`, `long_closure_count` | int | running totals since `reset()` |

### Rules
- **blink** = closure of ≥ 1 frame and ≤ 500 ms. **long_closure** = closure > 500 ms (a candidate micro-sleep). SG4/SG5 decide whether that means drowsiness.
- If the face is lost for ≤ 2 frames, an ongoing closure continues. A longer loss ends it.
- SG2 does **no** windowing, PERCLOS or drowsiness decision. That belongs to SG4/SG5 (cue → temporal evidence → decision → alert).

### Example record (real output, `mock/mock_sg2_alert.jsonl`, frame 119)
```json
{"interface_version": "sg2-v1", "frame_id": 119, "timestamp_ms": 3966.7, "eye_valid": true,
 "method": "A_baseline_ear_fixed_0.20", "left_ear": 0.2605, "right_ear": 0.2605, "ear": 0.2605,
 "left_eye_state": "open", "right_eye_state": "open", "eye_state": "open",
 "closed_score": -0.2605, "threshold": 0.2, "confidence": 1.0, "closed_duration_ms": 0.0,
 "blink_event": {"type": "blink", "start_frame": 111, "end_frame": 118, "start_ms": 3700.0,
                 "end_ms": 3933.3, "duration_ms": 266.6},
 "blink_count": 1, "long_closure_count": 0}
```

## 3. Mock data (so SG4 does not have to wait for SG1/SG2)

Folder [`mock/`](mock/). Each file is 60 s at 30 fps, deterministic (seed 477). Made by `module_sg2/tools/make_mock_streams.py`.

| Scenario | Scripted events | `mock_sg1_*.jsonl` | `mock_sg2_*.jsonl` |
|---|---|---|---|
| `alert` | 16 normal blinks (120–300 ms), open EAR ≈ 0.30 | SG1 V1 blocks (input to SG2) | SG2 V1 records (input to SG4) |
| `drowsy` | 14 slow blinks (300–480 ms) + 3 long closures (1.2 s, 2.0 s, 2.8 s), open EAR ≈ 0.24 | ✓ | ✓ |
| `face_lost` | 15 blinks + face missing for 1.0 s and 0.4 s | ✓ | ✓ |

[`mock/mock_ground_truth.json`](mock/mock_ground_truth.json) lists every scripted event. SG4 can use it to check their temporal logic.

Quick check:
```bash
cd module_sg2
python tools/run_sg2.py --sg1-jsonl ../interfaces/sg2/mock/mock_sg1_drowsy.jsonl \
    --config configs/baseline_w4_ear_fixed.yaml --out /tmp/sg2_drowsy.jsonl
# [sg2] 1800 frames ... blinks=14, long_closures=3
```
