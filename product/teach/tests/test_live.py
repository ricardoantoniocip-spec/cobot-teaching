"""Live follower on made-up landmarks: calibration, following, speed limit, hand lost. No camera needed."""

import copy

import numpy as np
from retarget.examples import synthetic_operator_frames
from retarget.kinematics import SerialChain, axis_angle_to_matrix
from retarget.live import (
    CALIBRATING,
    FOLLOWING,
    HOLD,
    WAITING,
    LiveConfig,
    LiveFollower,
)
from retarget.pose import H_WRIST, body_axes, wrist_angles

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


# ---------------------------------------------------------------- shoulder and wrist (2026-11-28, first real-arm test)


def rotate_hand(f, axis_body, angle, axes):
    """Turn the hand about its wrist by angle (rad) about an axis given in the operator's axes; the arm stays."""
    r = axes @ axis_angle_to_matrix(np.asarray(axis_body, float), angle) @ axes.T
    h = f.hand_world
    f.hand_world = (h - h[H_WRIST]) @ r.T + h[H_WRIST]


def test_wrist_angles_measure_tilt_and_roll():
    f = synthetic_operator_frames()[0]
    axes = body_axes([f])
    ref = f.hand_world.copy()
    fwd = (ref[9] - ref[0]) @ axes
    fwd /= np.linalg.norm(fwd)
    side = np.cross([0, 0, 1.0], fwd)
    side /= np.linalg.norm(side)
    g = copy.deepcopy(f)
    rotate_hand(g, side, np.radians(20), axes)  # about the left axis: fingers 20 deg further down
    pitch, roll = wrist_angles(g.hand_world, ref, axes)
    assert abs(np.degrees(pitch) - 20) < 1 and abs(np.degrees(roll)) < 1
    g = copy.deepcopy(f)
    rotate_hand(g, fwd, np.radians(40), axes)  # palm turned 40 deg about the hand's own forward axis
    pitch, roll = wrist_angles(g.hand_world, ref, axes)
    assert abs(np.degrees(roll) - 40) < 1 and abs(np.degrees(pitch)) < 1


def wrist_run(turn, **cfg):
    """Calibrate, then hold the hand still above the pick and turn the wrist from second 1 on."""
    frames, n_still = operator_frames()
    hold = copy.deepcopy(frames[n_still + 15])  # hand on the way to the pick, open, above the table
    axes = body_axes(frames[:n_still])
    out = frames[:n_still]
    for i in range(int(3 * HZ)):
        g = copy.deepcopy(hold)
        g.t = (n_still + i) / HZ
        if i >= HZ:
            turn(g, axes)
        out.append(g)
    fol, states = run(out, **cfg)
    return states[n_still + int(HZ) - 1], states[-1], fol


def test_hand_roll_drives_wrist_roll():
    def turn(g, axes):
        h = g.hand_world
        fwd = (h[9] - h[0]) / np.linalg.norm(h[9] - h[0])
        rotate_hand(g, axes.T @ fwd, np.radians(45), axes)

    before, after, fol = wrist_run(turn)
    assert abs(after.wrist_roll_deg - 45) < 2
    d = np.degrees(after.q - before.q)
    assert abs(abs(d[4]) - 45) < 5  # wrist_roll turns about as much as the palm
    assert np.all(np.abs(d[:3]) < 6)  # the rest only corrects a little (the tool point is a bit off the roll axis)
    assert np.linalg.norm(fol.chain.fk(after.q)[:3, 3] - after.target) < 0.005
    # same sense of turning: the gripper turns about its fingers the way the palm turned about the hand's forward axis
    r_rel = fol.chain.fk(before.q)[:3, :3].T @ fol.chain.fk(after.q)[:3, :3]
    assert np.degrees(np.arctan2(r_rel[1, 0], r_rel[0, 0])) > 35
    # switched off: no roll
    before, after, _ = wrist_run(turn, wrist=False)
    assert abs(np.degrees(after.q[4] - before.q[4])) < 2


def test_hand_tilt_drives_gripper_tilt():
    def turn(g, axes):
        h = g.hand_world
        fwd = axes.T @ (h[9] - h[0])
        side = np.cross([0, 0, 1.0], fwd / np.linalg.norm(fwd))
        rotate_hand(g, -side / np.linalg.norm(side), np.radians(30), axes)  # fingers 30 deg UP

    before, after, fol = wrist_run(turn)
    assert abs(after.wrist_flex_deg + 30) < 2

    def finger_down_deg(q):
        z = fol.chain.fk(q)[:3, 2]  # along the fingers
        return np.degrees(np.arcsin(-z[2]))

    assert finger_down_deg(before.q) - finger_down_deg(after.q) > 20  # the gripper tipped up with the hand
    assert abs(np.degrees(after.q[3] - before.q[3])) > 10  # mostly through wrist_flex
    assert np.linalg.norm(fol.chain.fk(after.q)[:3, 3] - after.target) < 0.005  # the tool point stays put


def test_follows_with_an_unsure_shoulder_after_calibration():
    frames, n_still = operator_frames()
    for f in frames[n_still:]:
        f.pose_vis = f.pose_vis.copy()
        f.pose_vis[12] = 0.05  # right shoulder hidden or out of the picture
    _, states = run(frames)
    assert states[-1].mode == FOLLOWING and states[-1].target is not None


def test_calibration_asks_for_the_shoulder():
    frames, _ = operator_frames()
    for f in frames[:20]:
        f.pose_vis = f.pose_vis.copy()
        f.pose_vis[12] = 0.05
    _, states = run(frames)
    assert states[0].mode == WAITING and "shoulder" in states[0].message
    assert states[-1].mode == FOLLOWING
