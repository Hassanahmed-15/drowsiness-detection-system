"""SG2 Eye State & Blink Analysis module (V1 interface).

Input : one SG1 V1 face block per frame (+ the BGR frame, only needed by the
        CNN method). See interfaces/sg2/README.md.
Output: one SG2 V1 eye-state record per frame for SG4.
"""
from pathlib import Path

import cv2
import numpy as np
import yaml

from .blink import BlinkDetector
from .classifiers import build_classifier
from .geometry import eye_aspect_ratio, eye_crop, eye_width

SG2_INTERFACE_VERSION = "sg2-v1"
MODULE_DIR = Path(__file__).resolve().parent.parent


def load_config(path):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    model = cfg["classifier"].get("model_path")
    if model and not Path(model).is_absolute():
        cfg["classifier"]["model_path"] = str(MODULE_DIR / model)
    return cfg


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


class EyeStateModule:
    def __init__(self, cfg):
        if isinstance(cfg, (str, Path)):
            cfg = load_config(cfg)
        self.cfg = cfg
        self.config_name = cfg.get("name", "unnamed")
        self.classifier = build_classifier(cfg["classifier"])
        self.blink = BlinkDetector(**cfg.get("blink", {}))
        crop = cfg.get("crop", {})
        self.crop_size = int(crop.get("size", 32))
        self.crop_scale = float(crop.get("scale", 1.6))
        self.needs_crops = self.classifier.name == "cnn"

    def reset(self):
        """Call when a new driver / video starts."""
        self.classifier.reset()
        self.blink.reset()

    def features(self, block, frame=None, crops=None):
        """Build the classifier feature dict from an SG1 block (None if invalid)."""
        left, right = block.get("left_eye_pts"), block.get("right_eye_pts")
        if not block.get("face_valid", block.get("face_detected", False)) or left is None or right is None:
            return None
        left_ear, right_ear = eye_aspect_ratio(left), eye_aspect_ratio(right)
        if left_ear is None or right_ear is None:
            return None
        f = {"left_ear": left_ear, "right_ear": right_ear,
             "left_width": eye_width(left), "right_width": eye_width(right),
             "left_crop": None, "right_crop": None,
             "blendshapes": block.get("blendshapes")}
        if self.needs_crops:
            if crops is not None:
                f["left_crop"], f["right_crop"] = crops
            elif frame is not None:
                gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                f["left_crop"] = eye_crop(gray, left, self.crop_size, self.crop_scale)
                f["right_crop"] = eye_crop(gray, right, self.crop_size, self.crop_scale)
            else:
                return None
        return f

    def process(self, block, frame=None, crops=None):
        """One SG1 block (+ frame) -> one SG2 V1 record."""
        fid, ts = block["frame_id"], block["timestamp_ms"]
        f = self.features(block, frame, crops)
        decision = self.classifier(f) if f is not None else None
        if decision is None:
            state, left_state, right_state = "unknown", "unknown", "unknown"
        else:
            state = "closed" if decision["closed"] else "open"
            left_state = "closed" if decision["left_closed"] else "open"
            right_state = "closed" if decision["right_closed"] else "open"
        event = self.blink.update(fid, ts, state)
        return {
            "interface_version": SG2_INTERFACE_VERSION,
            "frame_id": fid,
            "timestamp_ms": ts,
            "eye_valid": decision is not None,
            "method": self.config_name,
            "left_ear": _r(f and f["left_ear"]),
            "right_ear": _r(f and f["right_ear"]),
            "ear": _r(f and 0.5 * (f["left_ear"] + f["right_ear"])),
            "left_eye_state": left_state,
            "right_eye_state": right_state,
            "eye_state": state,
            "closed_score": _r(decision and decision["closed_score"]),
            "threshold": _r(decision and decision["threshold"]),
            "confidence": _r(decision and min(1.0, decision["margin"]), 3),
            "closed_duration_ms": round(self.blink.closed_duration_ms(ts), 1),
            "blink_event": event,
            "blink_count": self.blink.blink_count,
            "long_closure_count": self.blink.long_closure_count,
        }

    def process_stream(self, blocks, frames=None):
        frames = frames if frames is not None else [None] * len(blocks)
        return [self.process(b, fr) for b, fr in zip(blocks, frames)]


def to_jsonable(record):
    """numpy-safe copy for json.dumps."""
    out = {}
    for k, v in record.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        elif isinstance(v, np.bool_):
            v = bool(v)
        out[k] = v
    return out
