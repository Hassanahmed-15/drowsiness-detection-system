"""Open a video source the same way on a laptop and on the Jetson Orin Nano.

    source "0", "1", ...   USB / laptop webcam (V4L2 on Linux, AVFoundation on macOS)
    source "csi" / "csi:1" Jetson CSI camera through GStreamer (nvarguscamerasrc)
    source "<file>"        prerecorded video (fallback; proves software path only)
"""
import platform
import time

import cv2


def csi_pipeline(sensor_id=0, width=1280, height=720, fps=30):
    return (f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! "
            f"nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=1 max-buffers=1")


def open_source(source, width=None, height=None, fps=30):
    """Return (cv2.VideoCapture, kind) where kind is 'usb', 'csi' or 'file'."""
    source = str(source)
    if source.startswith("csi"):
        sensor = int(source.split(":")[1]) if ":" in source else 0
        cap = cv2.VideoCapture(csi_pipeline(sensor, width or 1280, height or 720, fps), cv2.CAP_GSTREAMER)
        return cap, "csi"
    if source.isdigit():
        backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(source), backend)
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        return cap, "usb"
    return cv2.VideoCapture(source), "file"


class FrameClock:
    """Monotonic timestamps in ms: real time for cameras, frame index / fps for files."""

    def __init__(self, kind, fps):
        self.kind, self.fps = kind, fps or 30.0
        self.t0, self.i = time.perf_counter(), 0

    def next(self):
        if self.kind == "file":
            ts = self.i * 1000.0 / self.fps
        else:
            ts = (time.perf_counter() - self.t0) * 1000.0
        self.i += 1
        return round(ts, 1)
