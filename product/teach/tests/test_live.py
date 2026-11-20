"""Live follower on made-up landmarks: calibration, following, speed limit, hand lost. No camera needed."""

import copy

import numpy as np

from retarget.examples import synthetic_operator_frames
from retarget.kinematics import SerialChain
from retarget.live import CALIBRATING, FOLLOWING, HOLD, WAITING, LiveConfig, LiveFollower

HZ = 30.0


def operator_frames():
    """2.5 s with the hand resting at the pick spot (calibration), then the pick and place."""
    demo = synthetic_operator_frames(hz=HZ)
    still = [copy.deepcopy(demo[2 * int(HZ)]) for _ in range(int(2.5 * HZ))]  # frame where the hand is down at the pick
    frames = still + demo
    for i, f in enumerate(frames):
        f.t = i / HZ
    return frames, len(still)


def run(frames, **cfg):
    fol = LiveFollower(SerialChain.from_urdf(), LiveConfig(**cfg))
    return fol, [copy.deepcopy(fol.update(f)) for f in frames]


def test_calibrates_then_follows_the_hand():
    frames, n_still = operator_frames()
    fol, states = run(frames)
    assert states[0].mode == CALIBRATING and 0 < states[30].calib_progress < 1
    first = next(i for i, s in enumerate(states) if s.mode == FOLLOWING)
    assert abs(first / HZ - 2.0) < 0.1
    assert 0.3 < fol.scale < 0.7  # a 70 cm arm onto a 33 cm reach
    following = [s for s in states[first + 1 :] if s.target is not None]
    assert len(following) > 200
    # the hand rested on the table during calibration, so the lowest target is the table height
    assert abs(min(s.target[2] for s in following) - fol.cfg.table_z_m) < 0.003
    # the pick is to the operator's left, the place to the right: the robot's y changes sign
    ys = [s.target[1] for s in following]
    assert max(ys) > 0.03 and min(ys) < -0.03
    # once it has caught up, IK reaches the target and the speed limit is not holding it back
    late = following[60:]
    assert np.median([s.ik_error_mm for s in late]) < 5
    assert np.median([s.lag_deg for s in late]) < 5


def test_gripper_follows_the_hand():
    frames, _ = operator_frames()
    _, states = run(frames)
    grips = np.array([s.gripper_open for s in states if s.mode == FOLLOWING])
    assert grips.max() > 0.95 and grips.min() < 0.05  # open hand opens it, closed hand closes it
    assert grips[-1] > 0.95


def test_joints_never_move_faster_than_the_limit():
    frames, _ = operator_frames()
    _, states = run(frames, max_joint_speed_deg_s=45.0)
    q = np.array([s.q for s in states])
    speed = np.degrees(np.abs(np.diff(q, axis=0))).max() * HZ
    assert speed <= 45.0 + 1e-6


def test_holds_when_the_hand_is_lost_and_stays_in_the_working_area():
    frames, n_still = operator_frames()
    for f in frames[n_still + 60 : n_still + 90]:  # one second without a hand
        f.hand_world = None
    for f in frames[n_still + 150 :]:  # then the hand goes far out to the side
        f.pose_world = f.pose_world.copy()
        f.pose_world[16] += np.array([0.0, 0.0, -1.0])
    fol, states = run(frames)
    lost = states[n_still + 60 : n_still + 90]
    assert lost[-1].mode == HOLD
    assert np.allclose(lost[-1].q, lost[10].q)  # no motion while holding
    far = [s for s in states[n_still + 160 :] if s.target is not None]
    assert any(s.limited for s in far)
    for s in far:
        assert np.linalg.norm(s.target[:2]) <= fol.cfg.max_radius_m + 1e-6 and s.target[2] <= fol.cfg.max_z_m + 1e-6


def test_waits_until_a_person_is_seen():
    frames, _ = operator_frames()
    for f in frames[:20]:
        f.pose_world = None
    _, states = run(frames)
    assert states[0].mode == WAITING and "shoulders" in states[0].message
    assert states[-1].mode == FOLLOWING
