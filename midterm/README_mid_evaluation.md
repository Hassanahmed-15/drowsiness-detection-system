# SG-1 Midterm Evaluation: Face, Eye and Mouth Landmark Module

CS-477 Computer Vision | SEECS CV Face-Off 2026 | Project 2: Driver Drowsiness Monitoring

| Item | Value |
|------|-------|
| Group number | [FILL: group no.] |
| Subgroup | SG-1 (face detection and landmarks) |
| Members | [FILL: names of all SG-1 members] |
| Branch / folder | `module_SG1/midterm/` |
| Target platform (Stage 2) | NVIDIA Jetson Orin Nano 8GB (JetPack 6.1, Python 3.10) |

Items marked `[FILL]` could not be taken from the code and need your input before submission.

---

## 1. Role of SG-1 in the system

SG-1 is the first stage of the pipeline. For every video frame it finds the driver's face and
produces everything the downstream subgroups need:

| Output | Consumed by |
|--------|-------------|
| Face box, `face_detected`, `face_valid` | All subgroups (gating) |
| 6-point left and right eye landmarks (EAR point order p1 to p6) | SG-2 eye state |
| 4 mouth points | SG-3 yawn |
| ROI boxes and cropped ROI images for left eye, right eye, mouth | SG-2, SG-3 |
| Head pose (yaw, pitch, roll in degrees) | SG-3 and later stages |
| `frame_id`, `timestamp_ms` | SG-4 temporal |

The output follows "Driver Drowsiness Monitoring: Interface Specification V1 (DRAFT)":
one JSON record per frame, boxes as `{x, y, width, height}` in full-frame pixels with a
top-left origin, `true`/`false`/`null` only (never 0/1 or NaN), and "left" meaning the
driver's own left eye (right side of an un-mirrored image).

## 2. Selected algorithms

| | Algorithm | Reason |
|---|-----------|--------|
| Primary | MediaPipe Face Landmarker (Tasks API, `face_landmarker.task`, float16 model, 478 landmarks) | One model gives the face, the eye and mouth landmarks and the points needed for head pose, so ROI boxes and EAR points all come from the same frame without a second detector. It runs without training or a GPU in our code. |
| Backup / alternative | [FILL: your team's backup algorithm] | [FILL: short reason] |

Edit the reason for the primary choice if your team's actual reasoning was different. The
backup is not stated anywhere in the code or README provided, so it has to come from the team.

Related design choices in the code:

- Head pose is computed by SG-1 from six landmarks with `cv2.solvePnP` (spec D-01).
- Up to 3 faces are requested from the model and the driver face is chosen as the largest
  face whose centre is inside a configurable driver region (spec D-13).
- ROIs are native-resolution crops with a margin: 15 px for eyes, 20 px for mouth (spec D-16).

## 3. Landmark map

Indices are MediaPipe Face Landmarker indices.

| Feature | Indices (in order) | Order |
|---------|--------------------|-------|
| Left eye (driver's left) | 263, 387, 385, 362, 380, 373 | p1 outer corner, p2/p3 upper lid, p4 inner corner, p5/p6 lower lid |
| Right eye | 33, 160, 158, 133, 153, 144 | same as above |
| Mouth | 61, 13, 291, 14 | corner (image-left), upper lip centre, corner (image-right), lower lip centre |
| Head pose points | 1, 152, 33, 263, 61, 291 | nose tip, chin, both eye outer corners, both mouth corners |

## 4. Output record (per frame)

```json
{
  "interface_version": "1.0",
  "frame_id": 0,
  "timestamp_ms": 0,
  "face": {
    "face_detected": true,
    "face_valid": true,
    "face_confidence": null,
    "face_box": {"x": 0, "y": 0, "width": 0, "height": 0},
    "left_eye_pts": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    "right_eye_pts": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    "mouth_pts": [[0, 0], [0, 0], [0, 0], [0, 0]],
    "left_eye_roi_box": {"x": 0, "y": 0, "width": 0, "height": 0},
    "right_eye_roi_box": {"x": 0, "y": 0, "width": 0, "height": 0},
    "mouth_roi_box": {"x": 0, "y": 0, "width": 0, "height": 0},
    "head_pose": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}
  }
}
```

The values above are placeholders showing the shape only. Real records are in `sg1_records.jsonl`.

Rules enforced and self-checked in code (`check_face_block`):

- `face_detected` false: every other field is `null`, `face_valid` is false.
- `face_detected` true but `face_valid` false: points, ROI boxes and head pose are `null`;
  only `face_box` is kept.
- `face_valid` true: all fields present, eyes have exactly 6 points.
- The three ROI images (uint8 BGR) are passed in memory and are not part of the JSON.

`face_valid` is true only if the face box is at least 40 px, all landmark points are inside
the frame, all three ROI boxes are at least 8 px, and head pose was solved.

## 5. Configurations compared

The comparison varies one shared parameter, `face_conf_min`, which the code applies to both
MediaPipe's face detection confidence and face presence confidence.

| Configuration | `face_conf_min` | Intent |
|---------------|-----------------|--------|
| Config A | 0.50 | More permissive, accepts lower-confidence detections |
| Config B | 0.70 | Stricter, requires higher-confidence detections |

Results on the demo video (fill from `outputs/config_comparison.csv` after running):

| Configuration | Total frames | face_valid frames | Failed frames | Reliability (%) |
|---------------|--------------|-------------------|---------------|-----------------|
| Config A (0.50) | [FILL] | [FILL] | [FILL] | [FILL] |
| Config B (0.70) | [FILL] | [FILL] | [FILL] | [FILL] |

Selected configuration: [FILL: A or B, with one line of reason based on the table above].

Reliability here means the share of frames where `face_valid` is true. It is not accuracy
against ground truth, because no labelled landmarks were used.

## 6. How to run

### Packages

```text
mediapipe
opencv-python
numpy
pandas
pyyaml        # only needed for --config
```

Exact versions should be taken from your working environment with `pip freeze`
(see `code/requirements.txt`). Python 3.9 to 3.12 is the safer range for MediaPipe on a PC.
The model file `face_landmarker.task` is downloaded on first run, so internet is needed once.
`ffmpeg` on PATH is optional (H.264 re-encode only).

### Setup

```bash
cd module_SG1/midterm/code
python -m venv venv
# Windows PowerShell: venv\Scripts\Activate.ps1      Linux/macOS: source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run

```bash
python Sg1_Pipeline.py --video "Video Project.mp4"
```

The file name is case-sensitive on Linux and Jetson, so use `Sg1_Pipeline.py` exactly as
committed.

| Option | Meaning |
|--------|---------|
| `--out-dir DIR` | Output folder (default `outputs`) |
| `--config PATH` | Read `face_conf_min` from `config/shared.yaml` |
| `--face-conf-min X` | Override `face_conf_min` (default 0.5) |
| `--max-faces N` | Faces requested from the model (default 3) |
| `--driver-region x1,y1,x2,y2` | Driver region as frame fractions (default whole frame) |
| `--include-landmarks` | Also log all 478 landmarks (large files) |
| `--skip-compare` | Skip the Config A vs Config B comparison |
| `--skip-video` | Skip writing the annotated video |
| `--no-h264` | Skip the ffmpeg H.264 re-encode |

Quick check on the first frame only:

```bash
python Sg1_Pipeline.py --video "Video Project.mp4" --skip-compare --skip-video
```

### Outputs (in `outputs/`)

| File | Description |
|------|-------------|
| `sg1_frame0_record.json` | SG-1 record for frame 0 |
| `sg1_annotated_frame.jpg` | First frame with boxes, points and head pose drawn |
| `config_comparison.csv` | Config A vs Config B results, including latency and FPS of the detect call |
| `sg1_records.jsonl` | One compact JSON record per frame |
| `sg1_annotated_video.mp4` | Annotated video (mp4v codec) |
| `sg1_annotated_video_h264.mp4` | H.264 re-encode, only if `ffmpeg` is installed |

## 7. Latency / FPS

Full results are in [`latency.md`](latency.md), produced by:

```bash
python benchmark_latency.py --video "Video Project.mp4" --device-label "your device name"
```

Headline numbers (copy from `latency.md` after running):

| Configuration | Mean total latency (ms) | FPS |
|---------------|-------------------------|-----|
| Config A (0.50) | [FILL] | [FILL] |
| Config B (0.70) | [FILL] | [FILL] |

Measured on: [FILL: device, CPU/GPU]. Jetson Orin Nano numbers: [FILL or "not yet measured"].

## 8. Demo evidence

- Demo video: `demo_video.mp4` [FILL or use `demo_link.md`]
- Annotated outputs from the same video: `outputs/sg1_annotated_video_h264.mp4`, `outputs/sg1_annotated_frame.jpg`

## 9. AI-generated and open-source code disclosure

AI-generated or AI-assisted:

| File | Tool | Notes |
|------|------|-------|
| `Sg1_Pipeline.py` (converted from `week5_CV_project_SG_1.ipynb`) | [FILL: confirm tool, e.g. Claude by Anthropic] | Notebook-to-script conversion and interface-spec alignment. Reviewed and run by: [FILL name] |
| `benchmark_latency.py` | Claude (Anthropic) | Latency benchmark written for this submission |
| `README_mid_evaluation.md`, `latency.md` template | Claude (Anthropic) | Drafted from the code and the advisory |

Open-source components used:

| Component | Use | Link |
|-----------|-----|------|
| MediaPipe Face Landmarker | Face detection and 478 landmarks | https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker |
| `face_landmarker.task` model | Downloaded from `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task` | Same page as above |
| OpenCV | Video I/O, `solvePnP`, drawing | https://opencv.org |
| NumPy, pandas | Arrays and the comparison table | https://numpy.org , https://pandas.pydata.org |

Check each library's and the model's licence terms yourself before listing them in the repo.

## 10. Known limitations and open items

- `face_confidence` is always `null`, because the Face Landmarker Tasks API does not return a
  detector score. `face_conf_min` therefore acts only as the model's own detection and
  presence threshold.
- The `face_valid` limits (face at least 40 px, ROI at least 8 px) are proposed values, not
  from the spec, and have not been tuned.
- The head pose sign convention is whatever the generic 3D model and `RQDecomp3x3` produce.
  It has not been verified against labelled images and still has to be agreed with the
  consuming subgroup (spec D-01).
- The driver region is not defined in the spec, so it defaults to the whole frame. With
  several people in view the largest face wins.
- The latency columns in `config_comparison.csv` time only the model call. Use `latency.md`
  for end-to-end numbers.
- Not yet tested on the Jetson Orin Nano. Whether the MediaPipe package used here installs
  and runs on JetPack 6.1 (aarch64, Python 3.10) is unconfirmed; check this before Stage 2.
- Reliability is the `face_valid` rate on one demo video, not an accuracy measure.

## 11. Midterm submission checklist (from the advisory)

- [ ] Subgroup folder/branch with code, primary and backup algorithm and reason
- [ ] Project board updated (owner, status, blockers)
- [ ] `midterm/` subfolder with this README, `code/`, `latency.md`, demo video or link
- [ ] AI-generated and open-source code reported (section 9)
- [ ] Demo video emailed to unawaz.msee25seecs@seecs.edu.pk, one email per subgroup,
      with all member names, group number and subgroup number
- [ ] Email subject: `CS-477 Midterm Demo – Group <No.> – SG-<No.>`
- [ ] Deadline: Nov 17, 2026
