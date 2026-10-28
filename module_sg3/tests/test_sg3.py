import numpy as np
import pytest

from sg3 import SG3, CueFlag, HeadPose, SG3Config
from sg3.yawn import mouth_aspect_ratio, yawn_prob_from_mar

W = 100.0


def pts(height):
    """8-point mouth of width W whose three vertical pairs are each `height` apart.

    Order: p1 left corner, p2 upper-left, p3 upper-mid, p4 upper-right,
           p5 right corner, p6 lower-right, p7 lower-mid, p8 lower-left.
    """
    h = height / 2.0
    return np.array(
        [[0, 0], [25, -h], [50, -h], [75, -h], [W, 0], [75, h], [50, h], [25, h]], dtype=float
    )


def roi(h=64, w=64, value=128):
    return np.full((h, w, 3), value, dtype=np.uint8)


def run(sg3=None, **kw):
    sg3 = sg3 or SG3()
    args = dict(frame_id=7, timestamp_ms=233, face_valid=True, mouth_roi=roi(),
                head_pose=HeadPose(0, 0, 0), mouth_pts=pts(10))
    args.update(kw)
    return sg3.process(**args)


def test_mar_formula_exact():
    # three vertical pairs of height 60 over width 100 -> 3*60 / (2*100) = 0.9
    assert mouth_aspect_ratio(pts(60)) == pytest.approx(0.9)
    # closed mouth
    assert mouth_aspect_ratio(pts(0)) == pytest.approx(0.0)


def test_mar_is_scale_invariant():
    assert mouth_aspect_ratio(pts(40) * 3.7) == pytest.approx(mouth_aspect_ratio(pts(40)))


@pytest.mark.parametrize("bad", [None, np.zeros((8, 2)), np.zeros((4, 2)), np.full((8, 2), np.nan)])
def test_mar_rejects_bad_points(bad):
    assert mouth_aspect_ratio(bad) is None


def test_yawn_prob_monotonic_and_centered():
    cfg = SG3Config()
    assert yawn_prob_from_mar(cfg.mar_center, cfg) == pytest.approx(0.5)
    probs = [yawn_prob_from_mar(m / 10, cfg) for m in range(0, 15)]
    assert probs == sorted(probs)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_ids_copied_through():
    out = run()
    assert (out.frame_id, out.timestamp_ms) == (7, 233)


def test_face_invalid_gates_everything():
    out = run(face_valid=False)
    assert out.mouth_valid is False and out.yawn_prob is None and out.cue_flags is None


def test_closed_vs_wide_open_mouth():
    closed = run(mouth_pts=pts(5)).yawn_prob      # MAR 0.075
    yawn = run(mouth_pts=pts(60)).yawn_prob       # MAR 0.9
    assert closed < 0.05 and yawn > 0.95


def test_missing_or_bad_pts_invalid_but_cues_survive():
    out = run(mouth_pts=None, head_pose=HeadPose(30, 0, 0))
    assert out.mouth_valid is False and out.yawn_prob is None
    assert out.cue_flags == [CueFlag.HEAD_NOD]       # cues are independent of the mouth
    assert run(mouth_pts=np.zeros((8, 2))).mouth_valid is False


def test_roi_not_needed():
    assert run(mouth_roi=None).mouth_valid is True


def test_custom_threshold():
    sg3 = SG3(SG3Config(mar_center=0.3))
    assert run(sg3, mouth_pts=pts(60 * 0.3 / 0.9)).yawn_prob == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize(
    "pose,expected",
    [
        (HeadPose(0, 0, 0), []),
        (HeadPose(25, 0, 0), [CueFlag.HEAD_NOD]),
        (HeadPose(-25, 0, 0), []),                    # head up is not a nod
        (HeadPose(0, 0, 25), [CueFlag.HEAD_SIDE_TILT]),
        (HeadPose(0, 0, -25), [CueFlag.HEAD_SIDE_TILT]),
        (HeadPose(25, 0, 25), [CueFlag.HEAD_NOD, CueFlag.HEAD_SIDE_TILT]),
    ],
)
def test_head_cues(pose, expected):
    assert run(head_pose=pose).cue_flags == expected


def test_head_pose_missing_or_nan_gives_none():
    assert run(head_pose=None).cue_flags is None
    assert run(head_pose=HeadPose(float("nan"), 0, 0)).cue_flags is None


def test_pitch_sign_config():
    sg3 = SG3(SG3Config(pitch_down_sign=-1.0))
    assert run(sg3, head_pose=HeadPose(-25, 0, 0)).cue_flags == [CueFlag.HEAD_NOD]


def test_to_dict_is_json_ready():
    import json
    d = run(head_pose=HeadPose(25, 0, 25)).to_dict()
    json.dumps(d)
    assert d["cue_flags"] == ["head_nod", "head_side_tilt"]
    assert set(d) == {"frame_id", "timestamp_ms", "mouth_valid", "yawn_prob", "cue_flags"}
