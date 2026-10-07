# Module — Sub-group 1

**Owner:** Tehreem Faisal and muhammad ammar

## Purpose
This module detects the driver's face and facial landmarks for the Driver Drowsiness Monitoring system.

It provides face information, eye landmarks, eye/mouth ROIs and head pose for downstream subgroups.

## Dependencies
- Python 3
- mediapipe
- opencv-python-headless
- numpy
- pandas

## Input / Output Interface
## Input

- BGR video frame
- frame_id
- timestamp_ms

## Output

The SG1 face block contains:

- frame_id
- timestamp_ms
- face_detected
- face_valid
- face_confidence
- face_box
- left_eye_pts
- right_eye_pts
- mouth_pts
- landmarks
- left_eye_roi_box
- right_eye_roi_box
- mouth_roi_box
- head_pose

## How to Run

1. Open the SG1 notebook in Google Colab.
2. Install the required dependencies.
3. Download the MediaPipe Face Landmarker model.
4. Upload the driver test video when prompted.
5. Run the notebook cells in order.
6. The notebook performs face and landmark detection, extracts eye/mouth ROIs, estimates head pose and evaluates the two confidence configurations.
7. 
MediaPipe Face Landmarker is used for face and landmark detection. The detector returns 478 facial landmarks.
The module extracts:
- 6 ordered points for the driver's left eye
- 6 ordered points for the driver's right eye
- mouth points
- left/right eye ROI boxes
- mouth ROI box
- face bounding box
- head pose: yaw, pitch and roll

## Test Data
The Week 5 experiment used a short driver-face video containing 571 frames.
The same video was used for both detector configurations to ensure a fair comparison.
The test video is uploaded on the google drive to be able to reuse.
It can be placed in the shared `datasets/` folder and referenced from this module.

Current test conditions mainly include a clearly visible frontal driver face. More difficult cases such as large head rotation, partial occlusion and low illumination will be added in later testing.

## Current Performance
Latest evaluation: 7 October 2026
Dataset: `Video_Project.mp4` — 571 frames
| Configuration | Confidence Threshold | Valid Frames | Failed Frames | Reliability | Avg. Latency | Processing FPS |
|---|---:|---:|---:|---:|---:|---:|
| Config A | 0.50 | 571/571 | 0 | 100% | 22.48 ms | 44.49 FPS |
| Config B | 0.70 | 571/571 | 0 | 100% | 23.65 ms | 42.28 FPS |

**Selected configuration:** Config A (`0.50` confidence threshold).

Both configurations achieved 100% face/landmark reliability on the current test video. Config A was selected because it achieved slightly lower latency and higher processing FPS.

## Interface Notes

Coordinates use full-frame pixel coordinates with the top-left corner as the origin.
ROI boxes use:
[x1, y1, x2, y2]
Left/right refer to the driver's own left and right.

## Current Limitations
- Current Week 5 video did not contain severe occlusion or difficult illumination.
- Face confidence is currently stored as null because the MediaPipe Face Landmarker Tasks API does not expose a simple detector confidence value in the same way as the previous API.
- Colab FPS is only a preliminary performance measurement and is not equivalent to Jetson performance.
