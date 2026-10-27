# SG-3: Yawn and Head-Pose Cues

## Purpose

SG-3 measures per-frame yawn evidence using the **Mouth Aspect Ratio (MAR)** and reports head-pose cues (`head_nod`, `head_side_tilt`). It reports **per-frame evidence only**. Turning `yawn_prob` into yawn events is SG-4's job (PROPOSED, D-07).

## Method: Mouth Aspect Ratio

MAR compares how far the inner lips are apart with how wide the mouth is (8 inner-lip landmarks):

```
MAR = ( ||p2 - p8|| + ||p3 - p7|| + ||p4 - p6|| ) / ( 2 * ||p1 - p5|| )

        p2   p3   p4
p1 ------                ------ p5      p1, p5 = mouth corners
        p8   p7   p6
```

- Closed mouth: MAR near 0. Talking: roughly 0.2 to 0.4. Yawn: 0.6 and above.
- `yawn_prob = sigmoid((MAR - mar_center) / mar_scale)`, with `mar_center = 0.60` (the decision threshold) and `mar_scale = 0.08`. This gives SG-4 a smooth 0 to 1 score instead of a hard flag.
- MAR is a ratio, so it does not depend on image size or distance from the camera.
- The thresholds are starting values from the literature, not tuned on your data. Tune them on your own clips (`--mar-thresh` or `SG3Config`).

## Dependencies

- Python 3.9+, NumPy (the core module needs only this)
- `opencv-python` for `run_sg3.py` and the head-pose test
- `mediapipe` only for the demo front end (`sg3/upstream_mediapipe.py`)
- `pytest` for tests

```bash
pip install -r requirements.txt
```

## Layout

```
SG-3/
  run_sg3.py                  # run on a video or webcam, writes JSON
  requirements.txt
  sg3/
    types.py                  # CueFlag, HeadPose, MouthOutput
    config.py                 # thresholds (SG3Config)
    yawn.py                   # MAR and MAR -> yawn_prob
    head_cues.py              # head_nod / head_side_tilt from head_pose
    module.py                 # SG3 class: the module entry point
    upstream_mediapipe.py     # optional: FaceMesh -> face_valid, ROI, mouth_pts, head_pose
  tests/
```

## Input / Output Interface

```python
from sg3 import SG3, HeadPose

sg3 = SG3()
out = sg3.process(
    frame_id=1042, timestamp_ms=34733, face_valid=True,
    mouth_roi=roi_bgr,                 # accepted for the interface, not used by MAR
    head_pose=HeadPose(pitch=24, yaw=3, roll=-2),
    mouth_pts=pts,                     # (8, 2) inner-lip landmarks, required for yawn_prob
)
out.to_dict()
```

### Inputs

| Field | Type | Notes |
|---|---|---|
| `frame_id`, `timestamp_ms` | int | Copied to output. |
| `face_valid` | bool | False forces an invalid output. |
| `mouth_roi` | ndarray or None | Cropped mouth region. Part of the interface but not used by MAR. |
| `head_pose` | `HeadPose` or None | Degrees. **`pitch > 0` means head down.** `roll` sign is ignored. Flip with `SG3Config.pitch_down_sign` if upstream differs. |
| `mouth_pts` | ndarray (8, 2) or None | Order: p1 left corner, p2 upper-left, p3 upper-mid, p4 upper-right, p5 right corner, p6 lower-right, p7 lower-mid, p8 lower-left. Any consistent coordinate space. Needed for `yawn_prob`. |

With MediaPipe FaceMesh the indices are `78, 82, 13, 312, 308, 317, 14, 87`, in that order.

### Output (`mouth` block)

| Field | Type | Notes |
|---|---|---|
| `frame_id`, `timestamp_ms` | int | Copied through. |
| `mouth_valid` | bool | False if MAR could not be computed. |
| `yawn_prob` | float or null | 0 to 1. Null if `mouth_valid` is false. |
| `cue_flags` | list or null | `[]` none observed, `null` not measured. Values: `head_nod`, `head_side_tilt`. |

### Validity rules (decisions to confirm in the spec)

- `face_valid = false` gives `mouth_valid = false`, `yawn_prob = null`, `cue_flags = null`.
- Missing, malformed, non-finite or zero-width `mouth_pts` gives `mouth_valid = false`.
- `cue_flags` depends only on `face_valid` and `head_pose`, not on `mouth_valid`. Missing or NaN `head_pose` gives `null`.

### Head-cue thresholds (`sg3/config.py`)

`head_nod`: pitch >= 20 deg. `head_side_tilt`: |roll| >= 20 deg.

## How to Run

```bash
python run_sg3.py --input path/to/video.mp4 --output results/sg3_output.json
python run_sg3.py --input 0 --show                        # webcam, overlay shows MAR and landmarks
python run_sg3.py --input clip.mp4 --mar-thresh 0.55      # custom MAR threshold
```

## Tests

```bash
pytest tests/
```

The tests check the MAR formula against hand-computed values, scale invariance, the yawn_prob mapping, validity rules, head cues, and the pitch sign convention of the MediaPipe head-pose helper on a synthetic face.

## Current Performance

Code-level tests pass (25/25). 

## Known Limitations

- Output is per-frame; no temporal smoothing (SG-4's job). A single high-MAR frame can also be wide-open talking or laughing, which is why SG-4 should require MAR to stay high over time.
- MAR needs reliable inner-lip landmarks. Occlusion (hand over mouth), extreme head rotation or poor lighting degrade them.
- Nod detection from pitch alone is noisy on single frames.
- The MediaPipe front end is a demo stand-in for the team's upstream modules.
