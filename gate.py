"""SG-3 step 1: input checks and the mouth_valid gate.

Implements the first stage of the SG-3 pipeline (interface spec V1, section 6):
decide whether this frame can be measured at all. Nothing here computes
yawn_prob or cue_flags yet; those come in later steps.

Python 3.8 compatible (no `X | None`, no built-in generics).
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

# SG-3-local parameter (not part of the interface). Move to config/shared.yaml
# once the team agrees on a value. Placeholder until you inspect real ROI sizes.
DEFAULT_MIN_ROI_PX = 16

# Reasons are for logging/debugging only. They are NOT part of the packet,
# so the interface contract stays unchanged.
REASON_OK = "ok"
REASON_FACE_INVALID = "face_invalid"
REASON_ROI_MISSING = "mouth_roi_missing"
REASON_ROI_BAD_TYPE = "mouth_roi_bad_type"
REASON_ROI_BAD_SHAPE = "mouth_roi_bad_shape"
REASON_ROI_TOO_SMALL = "mouth_roi_too_small"


@dataclass
class MouthInput:
    """SG-3 inputs (spec section 6)."""
    frame_id: int
    timestamp_ms: int
    face_valid: bool
    mouth_roi: Optional[np.ndarray]          # uint8, H x W x 3, BGR, native-res crop
    head_pose: Optional[Dict[str, float]]    # {yaw_deg, pitch_deg, roll_deg} or None
    mouth_pts: Optional[np.ndarray] = None   # optional, N x 2, full-frame pixels


def _check_identity(inp: MouthInput) -> None:
    """frame_id and timestamp_ms must be plain ints; they are copied through unchanged.

    bool is a subclass of int in Python, so it is rejected explicitly.
    Raises instead of returning invalid, because without identity we cannot
    emit a meaningful packet at all.
    """
    for name in ("frame_id", "timestamp_ms"):
        value = getattr(inp, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("%s must be an int, got %r" % (name, type(value).__name__))


def check_mouth_roi(roi: Any, min_roi_px: int = DEFAULT_MIN_ROI_PX) -> Tuple[bool, str]:
    """Return (usable, reason) for a mouth ROI image."""
    if roi is None:
        return False, REASON_ROI_MISSING
    if not isinstance(roi, np.ndarray) or roi.dtype != np.uint8:
        return False, REASON_ROI_BAD_TYPE
    if roi.ndim != 3 or roi.shape[2] != 3:
        return False, REASON_ROI_BAD_SHAPE
    h, w = roi.shape[0], roi.shape[1]
    if h < min_roi_px or w < min_roi_px:
        return False, REASON_ROI_TOO_SMALL
    return True, REASON_OK


def decide_mouth_valid(inp: MouthInput,
                       min_roi_px: int = DEFAULT_MIN_ROI_PX) -> Tuple[bool, str]:
    """The gate. Returns (mouth_valid, reason).

    mouth_valid is True only when the face is valid AND the mouth ROI is a
    usable image. SG-3 never skips emitting a packet; this only decides
    whether measurement is possible.
    """
    _check_identity(inp)

    # Spec F-07: booleans are real booleans. A non-bool here is an upstream bug,
    # so treat it as "not valid" rather than trusting truthiness (e.g. 1 or "false").
    if inp.face_valid is not True:
        return False, REASON_FACE_INVALID

    return check_mouth_roi(inp.mouth_roi, min_roi_px)


def build_mouth_block(inp: MouthInput, mouth_valid: bool) -> Dict[str, Any]:
    """Skeleton `mouth` block (spec section 6).

    For now yawn_prob and cue_flags are always None ("not measured").
    Later steps fill them in when mouth_valid is True.
    """
    return {
        "frame_id": int(inp.frame_id),
        "timestamp_ms": int(inp.timestamp_ms),
        "mouth_valid": bool(mouth_valid),
        "yawn_prob": None,
        "cue_flags": None,
    }


def run_step1(inp: MouthInput,
              min_roi_px: int = DEFAULT_MIN_ROI_PX) -> Tuple[Dict[str, Any], str]:
    """Step 1 entry point: gate + skeleton block. Returns (block, reason)."""
    valid, reason = decide_mouth_valid(inp, min_roi_px)
    return build_mouth_block(inp, valid), reason


# --------------------------------------------------------------------------
# Self-test: run `python sg3_gate.py`
# --------------------------------------------------------------------------
def _selftest() -> None:
    good_roi = np.zeros((48, 64, 3), dtype=np.uint8)
    pose = {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}

    def make(**kw: Any) -> MouthInput:
        base = dict(frame_id=0, timestamp_ms=0, face_valid=True,
                    mouth_roi=good_roi, head_pose=pose)
        base.update(kw)
        return MouthInput(**base)

    cases = [
        ("valid",            make(),                                   True,  REASON_OK),
        ("face invalid",     make(face_valid=False, mouth_roi=None),   False, REASON_FACE_INVALID),
        ("face flag is 1",   make(face_valid=1),                       False, REASON_FACE_INVALID),
        ("roi missing",      make(mouth_roi=None),                     False, REASON_ROI_MISSING),
        ("roi float dtype",  make(mouth_roi=good_roi.astype(np.float32)), False, REASON_ROI_BAD_TYPE),
        ("roi grayscale",    make(mouth_roi=np.zeros((48, 64), np.uint8)), False, REASON_ROI_BAD_SHAPE),
        ("roi too small",    make(mouth_roi=np.zeros((8, 8, 3), np.uint8)), False, REASON_ROI_TOO_SMALL),
        ("pose missing ok",  make(head_pose=None),                     True,  REASON_OK),
    ]
    for name, inp, want_valid, want_reason in cases:
        block, reason = run_step1(inp)
        assert block["mouth_valid"] is want_valid, name
        assert reason == want_reason, (name, reason)
        assert block["yawn_prob"] is None and block["cue_flags"] is None, name

    try:
        run_step1(make(frame_id="7"))
    except TypeError:
        pass
    else:
        raise AssertionError("string frame_id should raise TypeError")

    print("sg3_gate self-test: all %d cases passed" % (len(cases) + 1))


if __name__ == "__main__":
    _selftest()