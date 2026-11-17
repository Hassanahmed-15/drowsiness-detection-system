# SG1 Face / Eye / Mouth Landmark Pipeline

## Configurations

The pipeline compares two face-confidence configurations. This threshold is used for
MediaPipe's face detection and face presence checks:

| Configuration | `face_conf_min` | Description |
|---------------|-----------------|-------------|
| Config A | `0.50` | Less strict threshold, allowing detections with lower confidence. |
| Config B | `0.70` | Stricter threshold, requiring higher-confidence detections. |

It uses MediaPipe's Face Landmarker to process a driver video and produce, per frame:
face box, 6-point left/right eye landmarks, mouth points, ROI boxes for both eyes and the
mouth, all 478 landmarks, and head pose (yaw, pitch, roll).

## What it outputs

Everything goes into the `outputs/` folder (change with `--out-dir`):

| File | Description |
|------|-------------|
| `sg1_annotated_frame.jpg` | First frame with boxes, landmarks and head pose drawn |
| `config_comparison.csv` | Config A (0.50) vs Config B (0.70) detection confidence results |
| `sg1_annotated_video.mp4` | Full annotated video (mp4v codec) |
| `sg1_annotated_video_h264.mp4` | H.264 re-encode, only if `ffmpeg` is installed |

The model file `face_landmarker.task` is downloaded automatically into the working
directory on the first run (internet needed that one time).

## Setup (virtual environment)

Python 3.9 to 3.12 is the safest choice for MediaPipe. I am not certain which newer Python
versions have MediaPipe wheels, so check the MediaPipe install docs if `pip install` fails.

### Windows (PowerShell)

```powershell
cd sg1_face_pipeline
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in that window and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows (Command Prompt)

```bat
cd sg1_face_pipeline
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
cd sg1_face_pipeline
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Put your video somewhere local (for example next to the script) and run:

```bash
python sg1_face_pipeline.py --video "Video Project.mp4"
```

Useful options:

```text
--out-dir DIR      Output folder (default: outputs)
--skip-compare     Skip the Config A vs Config B comparison
--skip-video       Skip writing the annotated video
--no-h264          Skip the ffmpeg H.264 re-encode
```

Quick test on just the first frame:

```bash
python sg1_face_pipeline.py --video "Video Project.mp4" --skip-compare --skip-video
```

## Optional: ffmpeg

The H.264 re-encode step needs `ffmpeg` on your PATH (it is a system program, not a pip
package). If it is missing, the script skips that step and still produces the `mp4v` video.
Install it with `winget install ffmpeg` (Windows), `brew install ffmpeg` (macOS) or
`sudo apt install ffmpeg` (Ubuntu/Debian).

## Notes

- `face_confidence` is always `None`, same as the notebook, because the Face Landmarker
  Tasks API does not expose a detector confidence score.
- Timestamps in VIDEO mode must increase on every frame; the script derives them from the
  video's FPS (falls back to 25 FPS if the FPS cannot be read).
- I could not run this against your video or download the model in my environment, so I only
  syntax-checked the script. Please run it once and tell me if anything errors.
- The pinned minimum versions in `requirements.txt` are my best estimate of compatible
  versions, not versions I tested. The notebook output reported MediaPipe 1.1.0 and OpenCV
  5.0.0, which I could not verify. If you want an exact match to your Colab setup, run
  `pip freeze` in your working venv and save the result.
