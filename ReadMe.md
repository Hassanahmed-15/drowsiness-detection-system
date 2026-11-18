# Embedded — Sub-group 6

## Phase 1: hardware inspection

On the Jetson, run from any directory:

```bash
python3 /home/sanan/Desktop/'hammad embedded SG06'/drowsiness-detection-system/embedded_sg6/hardware_test.py
```

The script requires Python 3 and checks OpenCV, GStreamer, USB, V4L2, Jetson/Argus support, device permissions, memory and disk without installing software or changing permissions. It writes `embedded_sg6/reports/hardware_report.json`, replacing the previous report. Exit code 1 means at least one critical check failed; warnings require review. The report is an inventory, not a live camera test: a video node might be metadata or another non-camera device. CSI availability and capture must be verified separately. `v4l2-ctl` and other system tools are optional for the inspection but missing tools limit the diagnosis. OpenCV CUDA device count describes OpenCV's CUDA support, not system-wide CUDA availability.

Connect a compatible camera, rerun this inspection, and verify the actual capture device and live frames before proceeding to performance measurements or integration. No software packages are installed by this script.

## Live USB/V4L2 preview

```bash
python3 "$HOME/Desktop/hammad embedded SG06/drowsiness-detection-system/embedded_sg6/camera_preview.py"
```

The script probes video nodes until one produces a frame. Press Q or Escape while the preview window is focused to exit. To choose a specific node, add `--device /dev/video0`. This preview requires a graphical desktop and does not test CSI/Argus sources.
