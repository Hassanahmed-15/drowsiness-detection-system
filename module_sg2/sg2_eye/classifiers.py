"""Frame-level eye-state classifiers compared in Week 5.

Every classifier takes a per-frame feature dict and returns a "closed score"
(higher = more closed) plus the open/closed decision, so all methods can be
swept and compared with the same evaluation code.

    features = {
        "left_ear": float, "right_ear": float,          # always present
        "left_width": float, "right_width": float,      # eye width in px
        "left_crop": uint8[32,32] | None,               # only for the CNN
        "right_crop": uint8[32,32] | None,
        "blendshapes": {"eyeBlinkLeft": .., "eyeBlinkRight": ..} | None,
    }
"""
from collections import deque

import numpy as np


def fuse(left, right, left_w, right_w, mode):
    """Combine the two eyes into one value.

    mean           - plain average (Week 4 baseline behaviour)
    width_weighted - weight each eye by its visible width, so the eye turned
                     away from the camera (head yaw) counts less
    """
    if mode == "width_weighted":
        total = left_w + right_w
        if total > 1e-6:
            return (left * left_w + right * right_w) / total
    return 0.5 * (left + right)


class EARFixed:
    """Closed if fused EAR < threshold (Week 4 baseline, Config A/B)."""

    name = "ear_fixed"

    def __init__(self, threshold=0.20, fusion="mean", **_):
        self.threshold = float(threshold)
        self.fusion = fusion

    def reset(self):
        pass

    def __call__(self, f):
        ear = fuse(f["left_ear"], f["right_ear"], f["left_width"], f["right_width"], self.fusion)
        return {
            "closed_score": -ear,
            "left_closed": f["left_ear"] < self.threshold,
            "right_closed": f["right_ear"] < self.threshold,
            "closed": ear < self.threshold,
            "threshold": self.threshold,
            "margin": abs(ear - self.threshold) / 0.05,
        }


class EARAdaptive:
    """Per-driver threshold = k x (open-eye EAR level of this driver).

    The open-eye level is a running high percentile of the driver's recent
    fused EAR (eyes are open most of the time, so the 80th percentile is a
    robust "open" reference). Causal: only past frames are used. Until
    `warmup_frames` valid frames have been seen, the fixed fallback is used.
    """

    name = "ear_adaptive"

    def __init__(self, k=0.70, window_frames=900, percentile=80, warmup_frames=30,
                 fallback_threshold=0.20, min_threshold=0.10, max_threshold=0.35,
                 fusion="mean", **_):
        self.k = float(k)
        self.window_frames = int(window_frames)
        self.percentile = float(percentile)
        self.warmup_frames = int(warmup_frames)
        self.fallback_threshold = float(fallback_threshold)
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self.fusion = fusion
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.window_frames)

    def current_threshold(self):
        if len(self.history) < self.warmup_frames:
            return self.fallback_threshold
        baseline = float(np.percentile(np.fromiter(self.history, np.float32), self.percentile))
        return float(np.clip(self.k * baseline, self.min_threshold, self.max_threshold))

    def __call__(self, f):
        ear = fuse(f["left_ear"], f["right_ear"], f["left_width"], f["right_width"], self.fusion)
        thr = self.current_threshold()
        self.history.append(ear)
        return {
            "closed_score": -ear / (thr / self.k) if thr > 0 else -ear,  # EAR / open level
            "left_closed": f["left_ear"] < thr,
            "right_closed": f["right_ear"] < thr,
            "closed": ear < thr,
            "threshold": thr,
            "margin": abs(ear - thr) / 0.05,
        }


def standardise(crops):
    """Per-image zero-mean / unit-std normalisation (illumination robustness)."""
    x = crops.astype(np.float32)
    mean = x.mean(axis=(-1, -2), keepdims=True)
    std = x.std(axis=(-1, -2), keepdims=True)
    return (x - mean) / (std + 1e-3)


class CNNEyeState:
    """Learned open/closed classifier on 32x32 grey eye crops (ONNX model).

    Outputs P(closed) per eye; closed if fused probability > threshold.
    """

    name = "cnn"

    def __init__(self, model_path, threshold=0.5, fusion="mean", **_):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=ort.get_available_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = float(threshold)
        self.fusion = fusion

    def reset(self):
        pass

    def predict_proba(self, crops):
        """crops: uint8 array (N, 32, 32) -> P(closed) array (N,)."""
        x = standardise(np.asarray(crops))[:, None, :, :]
        logits = self.session.run(None, {self.input_name: x})[0].reshape(-1)
        return 1.0 / (1.0 + np.exp(-logits))

    def __call__(self, f):
        p_left, p_right = self.predict_proba(np.stack([f["left_crop"], f["right_crop"]]))
        p = fuse(p_left, p_right, f["left_width"], f["right_width"], self.fusion)
        return {
            "closed_score": float(p),
            "left_closed": bool(p_left > self.threshold),
            "right_closed": bool(p_right > self.threshold),
            "closed": bool(p > self.threshold),
            "threshold": self.threshold,
            "margin": abs(p - self.threshold) / 0.25,
        }


class BlendshapeEyeState:
    """MediaPipe Face Landmarker eyeBlink blendshape score (learned by Google).

    Needs SG1 to also forward `blendshapes`, which is NOT in the SG1 V1
    interface; evaluated in Week 5 only as a candidate for a V1.1 proposal.
    """

    name = "blendshape"

    def __init__(self, threshold=0.5, **_):
        self.threshold = float(threshold)

    def reset(self):
        pass

    def __call__(self, f):
        b = f.get("blendshapes") or {}
        left, right = b.get("eyeBlinkLeft"), b.get("eyeBlinkRight")
        if left is None or right is None:
            return None
        p = 0.5 * (left + right)
        return {
            "closed_score": float(p),
            "left_closed": left > self.threshold,
            "right_closed": right > self.threshold,
            "closed": p > self.threshold,
            "threshold": self.threshold,
            "margin": abs(p - self.threshold) / 0.25,
        }


CLASSIFIERS = {c.name: c for c in (EARFixed, EARAdaptive, CNNEyeState, BlendshapeEyeState)}


def build_classifier(cfg):
    params = dict(cfg)
    return CLASSIFIERS[params.pop("method")](**params)
