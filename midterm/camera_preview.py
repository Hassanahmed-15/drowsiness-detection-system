#!/usr/bin/env python3
"""Show a live camera preview. Press Q or Escape to exit."""
import argparse
import glob
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', help='V4L2 device path, e.g. /dev/video2; otherwise probe available nodes')
    args = parser.parse_args()
    try:
        import cv2
    except ImportError:
        print('OpenCV is unavailable for this Python interpreter.', file=sys.stderr)
        return 1
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        print('No graphical display is available; run this from the Jetson desktop session.', file=sys.stderr)
        return 1
    paths = [args.device] if args.device else sorted(glob.glob('/dev/video[0-9]*'))
    if not paths:
        print('No /dev/video* devices found.', file=sys.stderr)
        return 1
    camera = None
    for path in paths:
        print('Trying', path, flush=True)
        candidate = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if candidate.isOpened():
            ok, frame = candidate.read()
            if ok and frame is not None:
                camera = candidate
                break
        candidate.release()
    if camera is None:
        print('No tested device produced a frame. Check connection and camera permissions.', file=sys.stderr)
        return 1
    window = 'Camera preview - press Q or Esc to quit'
    print(f'Previewing {path}: {frame.shape[1]}x{frame.shape[0]}. Press Q or Escape to quit.', flush=True)
    try:
        while True:
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), ord('Q'), 27):
                break
            ok, frame = camera.read()
            if not ok or frame is None:
                print('Camera stopped delivering frames.', file=sys.stderr)
                return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())
