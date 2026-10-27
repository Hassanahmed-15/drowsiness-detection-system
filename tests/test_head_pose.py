"""Checks the solvePnP head-pose sign convention with a synthetic face (needs opencv)."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from sg3.upstream_mediapipe import _MODEL_POINTS, estimate_head_pose  # noqa: E402

W, H = 640, 480


def project(rot_x_deg=0.0, rot_z_deg=0.0):
    """Project the generic face model, rotated, 800 mm in front of the camera."""
    ax, az = np.radians(rot_x_deg), np.radians(rot_z_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    pts = _MODEL_POINTS @ (rz @ rx).T
    # model is y-up, camera is y-down and looks along +z: flip y and z, then translate
    cam_pts = pts * np.array([1, -1, -1]) + np.array([0, 0, 800.0])
    focal = float(W)
    img = np.stack([focal * cam_pts[:, 0] / cam_pts[:, 2] + W / 2,
                    focal * cam_pts[:, 1] / cam_pts[:, 2] + H / 2], axis=1)
    return img, cam_pts


def test_frontal_face_is_near_zero():
    img, _ = project()
    pose = estimate_head_pose(img, W, H)
    assert abs(pose.pitch) < 3 and abs(pose.roll) < 3


def test_pitch_down_is_positive():
    # Find which rotation direction moves the nose DOWN in the image relative to the eyes.
    down = None
    for deg in (+25.0, -25.0):
        img, _ = project(rot_x_deg=deg)
        nose_y, eyes_y = img[0, 1], (img[2, 1] + img[3, 1]) / 2
        base_img, _ = project()
        base_gap = base_img[0, 1] - (base_img[2, 1] + base_img[3, 1]) / 2
        if (nose_y - eyes_y) > base_gap:
            down = deg
    assert down is not None
    img, _ = project(rot_x_deg=down)
    pose = estimate_head_pose(img, W, H)
    assert pose.pitch > 15, pose


def test_roll_magnitude():
    img, _ = project(rot_z_deg=25.0)
    pose = estimate_head_pose(img, W, H)
    assert abs(abs(pose.roll) - 25.0) < 5, pose
