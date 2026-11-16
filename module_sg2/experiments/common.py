"""Shared paths, splits and label parsing for the SG2 Week 5 experiments."""
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = MODULE_DIR.parent
DATA_ROOT = REPO_DIR / "datasets" / "raw" / "sg2"
CACHE_DIR = MODULE_DIR / "experiments" / "cache"
RESULTS_DIR = MODULE_DIR / "results"

# Fixed Week 5 split (by video). Thresholds are tuned on DEV only and every
# configuration is reported on the same TEST videos.
VIDEOS = {
    "talking": "talkingFace/talkingFace/talking",
    "eb8_01": "eyeblink8/eyeblink8/1/26122013_223310_cam",
    "eb8_02": "eyeblink8/eyeblink8/2/26122013_224532_cam",
    "eb8_03": "eyeblink8/eyeblink8/3/26122013_230103_cam",
    "eb8_04": "eyeblink8/eyeblink8/4/26122013_230654_cam",
    "eb8_08": "eyeblink8/eyeblink8/8/27122013_151644_cam",
    "eb8_09": "eyeblink8/eyeblink8/9/27122013_152435_cam",
    "eb8_10": "eyeblink8/eyeblink8/10/27122013_153916_cam",
    "eb8_11": "eyeblink8/eyeblink8/11/27122013_154548_cam",  # driver wears glasses
}
DEV = ["talking", "eb8_01", "eb8_02", "eb8_03"]
TEST = ["eb8_04", "eb8_08", "eb8_09", "eb8_10", "eb8_11"]

# Synthetic low-light stress condition applied to the TEST videos
LOWLIGHT_GAMMA = 2.0
LOWLIGHT_GAIN = 0.6
LOWLIGHT_NOISE_STD = 3.0


def lowlight(frame, rng):
    x = (frame.astype(np.float32) / 255.0) ** LOWLIGHT_GAMMA * 255.0 * LOWLIGHT_GAIN
    x += rng.normal(0.0, LOWLIGHT_NOISE_STD, size=x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def parse_tag(path, n_frames):
    """Eyeblink8 / Talking Face .tag -> per-frame labels.

    Row: frameID:blinkID:NF:LE_FC:LE_NV:RE_FC:RE_NV:face box:eye corners
    label  1 = closed  (either eye annotated Fully Closed)
           0 = open    (not part of any blink, eyes visible)
          -1 = ignore  (blink transition frames that are not fully closed,
                        eye not visible, or frame not annotated)
    blink_id = annotated blink id (-1 outside blinks)
    """
    label = np.full(n_frames, -1, np.int8)
    blink_id = np.full(n_frames, -1, np.int32)
    meta = {}
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#"):
                if ":" in line:
                    k, v = line[1:].split(":", 1)
                    meta[k.strip()] = v.strip()
                continue
            parts = line.split(":")
            if len(parts) < 7 or not parts[0].isdigit():
                continue
            fid, bid = int(parts[0]), int(parts[1])
            if fid >= n_frames:
                continue
            le_fc, le_nv, re_fc, re_nv = parts[3], parts[4], parts[5], parts[6]
            blink_id[fid] = bid
            if le_fc == "C" or re_fc == "C":
                label[fid] = 1
            elif bid == -1 and le_nv != "N" and re_nv != "N":
                label[fid] = 0
    return label, blink_id, meta


def blink_intervals(blink_id):
    """Ground-truth blink events as (start, end) frame intervals."""
    out, cur, start = [], -1, 0
    for i, b in enumerate(np.append(blink_id, -1)):
        if b != cur:
            if cur != -1:
                out.append((start, i - 1))
            cur, start = b, i
    return out


def load_cache(name):
    return dict(np.load(CACHE_DIR / f"{name}.npz", allow_pickle=True))
