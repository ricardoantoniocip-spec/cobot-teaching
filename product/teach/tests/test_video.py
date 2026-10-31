"""Render a synthetic video with two ArUco markers and check the recovered hand path."""

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from retarget.kinematics import make_transform, rpy_to_matrix
from retarget.video import CameraModel, MarkerSetup, marker_object_points, video_to_demo

W, H = 1280, 720


def _draw_marker(img, dictionary, marker_id, cam_to_marker, size, cam):
    side = 200
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
    border = side // 4  # white quiet zone around the marker
    tile = np.full((side + 2 * border, side + 2 * border), 255, np.uint8)
    tile[border:-border, border:-border] = marker
    s = size * (side + 2 * border) / side
    obj = marker_object_points(s)
    rvec, _ = cv2.Rodrigues(cam_to_marker[:3, :3])
    pts, _ = cv2.projectPoints(obj, rvec, cam_to_marker[:3, 3], cam.matrix, cam.dist)
    src = np.float32([[0, 0], [tile.shape[1], 0], [tile.shape[1], tile.shape[0]], [0, tile.shape[0]]])
    h = cv2.getPerspectiveTransform(src, pts.reshape(4, 2).astype(np.float32))
    warped = cv2.warpPerspective(tile, h, (W, H), borderValue=0)
    mask = cv2.warpPerspective(np.full_like(tile, 255), h, (W, H)) > 0
    img[mask] = warped[mask]


def test_video_to_demo_recovers_hand_path(tmp_path):
    cam = CameraModel.guess(W, H, 70)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # Camera 0.6 m above the table looking down; table marker 0.25 m in front of the robot base.
    cam_to_table = make_transform(rpy_to_matrix(np.pi, 0, 0), np.array([0.0, 0.05, 0.6]))
    base_to_table = make_transform(np.eye(3), np.array([0.25, 0.0, 0.0]))
    markers = {
        "table_id": 0,
        "table_size_m": 0.06,
        "hand_id": 1,
        "hand_size_m": 0.04,
        "base_to_table": {"xyz": [0.25, 0.0, 0.0], "rpy": [0, 0, 0]},
    }
    (tmp_path / "markers.json").write_text(json.dumps(markers))

    video = tmp_path / "demo.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (W, H))
    truth = []
    for k in range(30):
        table_to_hand = make_transform(rpy_to_matrix(0, 0, 0.3), np.array([-0.1 + 0.2 * k / 29, 0.1, 0.12]))
        img = np.full((H, W), 128, np.uint8)
        _draw_marker(img, dictionary, 0, cam_to_table, 0.06, cam)
        _draw_marker(img, dictionary, 1, cam_to_table @ table_to_hand, 0.04, cam)
        writer.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        truth.append((base_to_table @ table_to_hand)[:3, 3])
    writer.release()

    demo, warnings = video_to_demo(video, MarkerSetup.from_json(tmp_path / "markers.json"), cam, [(0.3, 0.7)])
    assert len(demo) >= 28, warnings
    idx = np.rint(demo.t * 30).astype(int)
    err_mm = np.linalg.norm(demo.poses[:, :3, 3] - np.array(truth)[idx], axis=1) * 1000
    # One camera, depth from marker size: about 5 mm in this synthetic test. Gate A target is 10 mm.
    assert np.median(err_mm) < 10.0, err_mm
    assert demo.grip[idx == 15][0] == 1.0 and demo.grip[idx == 0][0] == 0.0
