# Embedded — Sub-group 6

**Project:** Driver Drowsiness Detection System  
**Target:** NVIDIA Jetson Orin Nano Developer Kit, 8 GB  
**Camera:** USB webcam (generic UVC camera assumed; use `/dev/video0` / OpenCV camera index `0`)  
**Owner:** Embedded Sub-group 6

## Purpose

Deploy the team's driver drowsiness detection pipeline on the Jetson Orin Nano. This subgroup handles camera access, device setup, running the integrated model, displaying alerts, and measuring performance on the target hardware.

## Development Status

The team has successfully run the camera on the Jetson Orin Nano. The current milestone verifies camera access and live video only; drowsiness detection, warning behavior, and performance benchmarking have not yet been reported as tested. A short camera-test video was provided by the team as a record of this milestone.

## Dependencies and Hardware

### Hardware

- NVIDIA Jetson Orin Nano Developer Kit, 8 GB
- USB webcam, aimed at the driver's face
- Display connected to the Jetson for the live preview and on-screen warning
- Stable power supply and sufficient storage for the OS, model, and test videos

The camera interface and 640 × 480 capture size are starter defaults. Check that the webcam supports the selected resolution and adjust it for the final test setup.

### Software

- NVIDIA JetPack / Jetson Linux installed on the board. Record the exact version used for the final test; Jetson package compatibility depends on the installed release. See [NVIDIA JetPack downloads and release notes](https://developer.nvidia.com/embedded/jetpack/downloads).
- Python 3 provided by the installed Jetson Linux image
- OpenCV for camera capture and preview (prefer the Jetson-compatible system package)
- The inference runtime and model format supplied by the integration team; use the TensorRT version bundled for the installed JetPack release when the model is delivered for TensorRT

Record the tested Python and package versions after setup. Do not install generic desktop CUDA/TensorRT packages over the Jetson-provided stack.

## Input and Output Interface

### Inputs

- Live video from the USB webcam, initially camera index `0`
- The integrated model and its configuration, supplied by the integration team
- Optional local video file for repeatable tests

### Outputs

- Live preview with the detected drowsiness state overlaid
- On-screen warning when the demo drowsiness condition is met
- Optional CSV log containing timestamps and state changes; do not log or retain driver video unless the team has agreed on data handling

**Starter detection policy:** raise a warning when the integrated detector reports closed eyes continuously for 2 seconds. Treat this as a project demo setting, not a validated safety threshold; make the duration configurable and tune it during evaluation. The integration team should confirm the model's actual states and interface before connecting this policy.

See [`../interfaces/`](../interfaces/) for the shared interface schema. Add the camera, model, and output field names there when the integration contract is finalized.

## Setup

1. Prepare the Jetson with the JetPack image agreed by the project team.
2. Connect the USB webcam and display.
3. From a terminal on the Jetson, check the installed versions and camera device:

   ```bash
   python3 --version
   dpkg-query -W nvidia-jetpack
   ls /dev/video*
   ```

4. Install OpenCV using the package source selected for that JetPack image. Verify `import cv2` works before running the pipeline.
5. Copy or download the integrated model using the team's approved method. Keep model weights and caches out of Git unless maintainers approve them.

## How to Run

The team has completed a camera-only run, but the exact command used was not provided. The project source and integrated entry point are not present in this workspace, so the following is a proposed command-line contract, not a verified command. Update it to the command that works in the repository:

```bash
# From the repository root; --source 0 selects the first webcam
python3 embedded_sg6/run.py --source 0 --width 640 --height 480
```

For a repeatable video test, pass a video path instead of `0`:

```bash
python3 embedded_sg6/run.py --source path/to/test-video.mp4
```

The application should exit cleanly when the user presses `q` or Ctrl+C and release the camera.

## Test Data

The provided camera-test clip documents camera operation only; it is not a drowsiness-detection evaluation. For detection tests, use a short, consented test video or the dataset selected by the project team. Follow [`../datasets/`](../datasets/) for dataset manifests and download/preparation instructions. Do not commit raw datasets or identifiable driver footage to Git.

## Current Performance

The camera-only test ran successfully on the Jetson Orin Nano. No model inference or drowsiness detection results have been reported yet. Fill in technical measurements after the integrated pipeline is tested; do not use estimates as measured results.

| Metric | Result |
|---|---|
| Camera-only test | Successful on Jetson Orin Nano |
| Camera model and input resolution | Not recorded |
| JetPack and Python versions | Not recorded |
| Drowsiness model and detection results | Not tested yet |
| Dataset or test video | Select and record during test |
| Average FPS and end-to-end latency | Not measured |
| CPU, GPU, and memory use | Not measured |
| Alert behavior and latency | Not tested yet |

## Troubleshooting

- **Camera not detected:** reconnect the webcam, check `ls /dev/video*`, and confirm the selected camera index.
- **OpenCV import fails:** check the OpenCV package source and Python version for the installed JetPack image.
- **Model/runtime fails to load:** confirm that the model format and inference runtime match the JetPack release and the integration team's export settings.
- **Low frame rate:** record camera resolution, model, runtime, and FPS before changing settings; share the measurements with the integration team.
- **No warning appears:** verify that the detector output states match the interface contract and that the demo closure duration is configured as expected.

## Team Workflow

- Work on the `embedded_sg6` branch or an agreed feature branch; merge into `main` through a reviewed pull request.
- Commit focused changes with descriptive messages.
- Keep raw datasets, captured driver footage, model caches, and environment folders out of Git unless approved.
- Record tested software versions and target-device measurements so another team member can reproduce the setup.
