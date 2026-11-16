"""Synthetic niqab occlusion for the labelled Eyeblink8 / Talking Face videos.

Uses the hand-annotated face box and eye corners in each .tag file (independent
of MediaPipe) to cover the head and face with dark fabric, leaving only an eye
slit. This makes it possible to measure detection and eye-state accuracy for
drivers who wear a niqab on the same labelled TEST videos as everything else.
"""
import numpy as np


def parse_geometry(path, n_frames):
    """Per-frame face box (x, y, w, h) and eye corners (4 points) from a .tag file."""
    face = np.full((n_frames, 4), np.nan, np.float32)
    eyes = np.full((n_frames, 4, 2), np.nan, np.float32)
    with open(path, errors="ignore") as fh:
        for line in fh:
            parts = line.strip().split(":")
            if len(parts) < 19 or not parts[0].isdigit():
                continue
            fid = int(parts[0])
            if fid >= n_frames:
                continue
            try:
                vals = [float(v) for v in parts[7:19]]
            except ValueError:
                continue
            face[fid] = vals[:4]
            eyes[fid] = np.array(vals[4:12]).reshape(4, 2)
    return face, eyes


def niqab(frame, face_box, eye_corners, rng, slit=(0.30, 0.25), colour=(22, 20, 24)):
    """Cover head + face + neck with dark fabric except an eye slit.

    slit = (above, below) eye-line, as fractions of the distance between the eye centres.
    """
    if np.isnan(face_box).any() or np.isnan(eye_corners).any():
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    x, y, fw, fh = face_box
    le, re = eye_corners[:2].mean(axis=0), eye_corners[2:].mean(axis=0)
    d = float(np.linalg.norm(re - le)) or fw / 3
    cy = 0.5 * (le[1] + re[1])
    cover = np.zeros((h, w), bool)
    x0, x1 = int(max(0, x - 0.35 * fw)), int(min(w, x + 1.35 * fw))
    cover[int(max(0, y - 0.7 * fh)):h, x0:x1] = True             # head, face, neck and shoulders
    sx0 = int(max(0, eye_corners[:, 0].min() - 0.30 * d))
    sx1 = int(min(w, eye_corners[:, 0].max() + 0.30 * d))
    sy0, sy1 = int(max(0, cy - slit[0] * d)), int(min(h, cy + slit[1] * d))
    cover[sy0:sy1, sx0:sx1] = False                               # eye slit
    fabric = np.array(colour, np.float32) + rng.normal(0, 4, (h, w, 1))
    out[cover] = np.clip(fabric[cover], 0, 255).astype(np.uint8)
    return out
