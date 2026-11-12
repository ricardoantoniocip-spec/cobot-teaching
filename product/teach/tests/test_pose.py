"""Arm tracking geometry, tested on made-up landmarks (no video and no MediaPipe needed)."""

import numpy as np
import pytest

from retarget import pose
from retarget.examples import synthetic_operator_frames
from retarget.kinematics import SerialChain, make_transform, rpy_to_matrix
from retarget.retarget import retarget


@pytest.fixture(scope="module")
def chain():
    return SerialChain.from_urdf()


def test_elbow_angle_and_grip_ratio():
    assert pose.elbow_angle_deg(np.array([0, 0, 0.0]), np.array([0.3, 0, 0]), np.array([0.3, 0.3, 0])) == pytest.approx(90)
    f = synthetic_operator_frames()
    assert pose.grip_ratio(f[0].hand_world) > 0.9  # open at the start
    states = pose.grip_states(np.array([1.0, 0.6, 0.45, 0.6, 0.75, 0.6]), 0.5, 0.7)
    assert states.tolist() == [0, 0, 1, 1, 0, 0]  # 0.6 keeps whatever state it was in


def test_body_axes_follow_the_operator_not_the_camera():
    frames = synthetic_operator_frames()
    assert np.allclose(pose.body_axes(frames), [[0, 1, 0], [0, 0, -1], [-1, 0, 0]], atol=1e-9)
    # Turn the camera 40 degrees around the operator: the hand path in the robot frame must not change.
    rot = rpy_to_matrix(0.0, np.radians(40), 0.0)
    turned = synthetic_operator_frames()
    for f in turned:
        f.pose_world, f.hand_world = f.pose_world @ rot.T, f.hand_world @ rot.T
    c = SerialChain.from_urdf()
    a = pose.frames_to_demo(frames, c).demo.poses[:, :3, 3]
    b = pose.frames_to_demo(turned, c).demo.poses[:, :3, 3]
    assert np.abs(a - b).max() < 1e-6


def test_body_mode_demo_lands_in_the_robot_workspace(chain):
    res = pose.frames_to_demo(synthetic_operator_frames(noise_m=0.003), chain)
    assert res.mode == "body" and 0.35 < res.scale < 0.6
    steps = pose.segment_steps(res.demo)
    assert [s["step"] for s in steps] == ["reach", "grasp", "move", "release", "retreat"]
    pick, place = pose.pick_and_place_points(steps)
    assert pick[1] > 0.05 and place[1] < -0.05  # picked on the left, placed on the right
    assert pick[2] < 0.02 and place[2] < 0.02  # both at table height
    _, report = retarget(res.demo, chain)
    assert report.reached_fraction > 0.95


def test_lost_hand_is_filled_and_reported(chain):
    frames = synthetic_operator_frames()
    for f in frames[100:130]:  # one second without a hand
        f.hand_world = f.hand_px = None
    res = pose.frames_to_demo(frames, chain)
    assert any("hand lost for 1.0 s" in w for w in res.warnings)
    assert np.isfinite(res.demo.poses).all()


def test_table_marker_mode_recovers_true_positions(chain):
    cv2 = pytest.importorskip("cv2")
    from retarget.video import CameraModel, MarkerSetup

    cam = CameraModel.guess(1280, 720)
    base_to_table = make_transform(np.eye(3), np.array([0.20, 0.0, 0.0]))
    # Camera 60 cm in front of and 50 cm above the table marker, looking down at it.
    cam_to_table = make_transform(rpy_to_matrix(np.radians(-140), 0.0, 0.0), np.array([0.0, 0.05, 0.75]))
    setup = MarkerSetup(table_id=0, table_size_m=0.06, hand_id=1, hand_size_m=0.04, base_to_table=base_to_table)
    base_to_cam = base_to_table @ np.linalg.inv(cam_to_table)
    cam_to_base = np.linalg.inv(base_to_cam)

    frames = synthetic_operator_frames()
    true = []
    for i, f in enumerate(frames):
        # Put the made-up hand at a known place in the robot frame: a slow line across the table.
        target = np.array([0.22 + 0.0002 * i, -0.10 + 0.0007 * i, 0.03])
        pinch = (f.hand_world[4] + f.hand_world[8]) / 2
        hand_cam = (f.hand_world - pinch) + (cam_to_base @ np.append(target, 1))[:3]
        rvec = np.zeros(3)
        px, _ = cv2.projectPoints(hand_cam, rvec, np.zeros(3), cam.matrix, cam.dist)
        f.hand_px, f.cam_to_table = px.reshape(-1, 2), cam_to_table
        true.append(target)
    res = pose.frames_to_demo(frames, chain, pose.PoseConfig(smooth_s=0.0), markers=setup, cam=cam)
    got = res.demo.poses[:, :3, 3]
    assert res.mode == "table-marker"
    assert np.abs(got - np.array(true)[: len(got)]).max() < 0.002
