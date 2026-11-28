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
    """3.5 s with the hand resting at the pick spot (calibration), then the pick and place."""
    demo = synthetic_operator_frames(hz=HZ)
    still = [copy.deepcopy(demo[2 * int(HZ)]) for _ in range(int(3.5 * HZ))]  # frame where the hand is down at the pick
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
    # the pick is to the operator's left, the place to the right: the robot moves sideways the same way
    ys = [s.target[1] for s in following]
    assert max(ys) - min(ys) > 0.08 and abs(following[0].target[1] - fol.cfg.rest_xy_m[1]) < 0.01
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
    """Calibrate, then hold the hand still above the pick (2 s, the robot catches up) and turn the wrist after that."""
    frames, n_still = operator_frames()
    hold = copy.deepcopy(frames[n_still + 15])  # hand on the way to the pick, open, above the table
    axes = body_axes(frames[:n_still])
    out = frames[:n_still]
    for i in range(int(4 * HZ)):
        g = copy.deepcopy(hold)
        g.t = (n_still + i) / HZ
        if i >= 2 * HZ:
            turn(g, axes)
        out.append(g)
    fol, states = run(out, **cfg)
    return states[n_still + int(2 * HZ) - 1], states[-1], fol


def test_hand_roll_drives_wrist_roll():
    def turn(g, axes):
        h = g.hand_world
        fwd = (h[9] - h[0]) / np.linalg.norm(h[9] - h[0])
        rotate_hand(g, axes.T @ fwd, np.radians(45), axes)

    before, after, fol = wrist_run(turn)
    assert abs(after.wrist_roll_deg - 45) < 2
    d = np.degrees(after.q - before.q)
    assert abs(abs(d[4]) - (45 - fol.cfg.wrist_deadband_deg)) < 3  # wrist_roll turns as much as the palm, less the dead band
    assert abs(d[3]) < 1  # wrist_flex stays (the arm joints follow the grasp point, which moves a little as the hand turns)
    assert after.ik_error_mm < 5
    # same sense of turning: the gripper turns about its fingers the way the palm turned about the hand's forward axis
    r_rel = fol.chain.fk(before.q)[:3, :3].T @ fol.chain.fk(after.q)[:3, :3]
    assert np.degrees(np.arctan2(r_rel[1, 0], r_rel[0, 0])) > 30
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

    assert finger_down_deg(before.q) - finger_down_deg(after.q) > 18  # the gripper tipped up with the hand
    d = np.degrees(after.q - before.q)
    assert abs(abs(d[3]) - (30 - fol.cfg.wrist_deadband_deg)) < 3  # wrist_flex turns by the same angle (arm joints only follow the moved grasp point)
    assert after.ik_error_mm < 5 and not after.wrist_at_limit


def test_ik_never_jams_against_the_wrist():
    """28 Nov real test: the hand tilted 40-70 deg and IK, asked for that tilt, jammed wrist_flex at 95 deg with 5 cm misses."""
    def turn(g, axes):
        h = g.hand_world
        fwd = axes.T @ (h[9] - h[0])
        side = np.cross([0, 0, 1.0], fwd / np.linalg.norm(fwd))
        rotate_hand(g, side / np.linalg.norm(side), np.radians(70), axes)  # fingers 70 deg further down

    _, after, _ = wrist_run(turn)
    assert after.ik_error_mm < 5


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


def test_calm_when_still_and_restarts_calibration_if_you_move():
    rng = np.random.default_rng(1)
    frames, n_still = operator_frames()
    hold = frames[n_still + 15]
    noisy = frames[:n_still]
    for i in range(int(3 * HZ)):  # holding still, tracker noise 5 mm on the wrist and 3 deg on the hand
        g = copy.deepcopy(hold)
        g.t = (n_still + i) / HZ
        g.pose_world = g.pose_world.copy()
        g.pose_world[16] += rng.normal(0, 0.005, 3)
        axes = body_axes(frames[:n_still])
        rotate_hand(g, rng.normal(0, 1, 3) / np.sqrt(3), np.radians(rng.normal(0, 3)), axes)
        noisy.append(g)
    _, states = run(noisy)
    late = np.array([s.q for s in states[-int(HZ):]])
    assert np.degrees(np.ptp(late, axis=0)).max() < 4  # the robot barely shakes

    moving = operator_frames()[0][n_still:]  # no still part: the hand never rests
    _, states = run(moving)
    assert all(s.mode != FOLLOWING for s in states[: int(2.5 * HZ)])


def test_stretching_the_arm_moves_the_robot_forward_even_with_bad_depth():
    """28 Nov real test: camera in front, the tracker's depth was poor and stretching the arm moved the robot down and back.
    Here, after calibration, the wrist's depth is squashed to a third but the elbow angle is right (MediaPipe's
    3D body keeps bone lengths and joint angles plausible even when its distance from the camera is off)."""
    frames, n_still = operator_frames()

    def squash(fs):
        out = []
        for f in fs:
            g = copy.deepcopy(f)
            g.pose_world = g.pose_world.copy()
            sh = g.pose_world[12]
            d = g.pose_world[16] - sh
            elbow_ang = np.degrees(np.arccos(np.clip(np.dot(sh - g.pose_world[14], g.pose_world[16] - g.pose_world[14])
                                                     / np.linalg.norm(sh - g.pose_world[14]) / np.linalg.norm(g.pose_world[16] - g.pose_world[14]), -1, 1)))
            g.pose_world[16] = sh + d * np.array([1.0, 1.0, 1 / 3])  # camera z = depth, squashed
            g.elbow_true = elbow_ang
            out.append(g)
        return out

    import retarget.live as L
    true_angle = L.elbow_angle_deg
    try:  # the tracker's elbow angle stays right even when its depth is off
        cur = {}
        L.elbow_angle_deg = lambda a, b, c: cur["ang"]
        fol = LiveFollower(SerialChain.from_urdf(), LiveConfig())
        xs, states = [], []
        for f in frames[:n_still] + squash(frames[n_still:]):
            cur["ang"] = getattr(f, "elbow_true", None) or true_angle(f.pose_world[12], f.pose_world[14], f.pose_world[16])
            states.append(fol.update(f))
        good = LiveFollower(SerialChain.from_urdf(), LiveConfig())
        ref = []
        for f in frames:
            cur["ang"] = true_angle(f.pose_world[12], f.pose_world[14], f.pose_world[16])
            ref.append(good.update(f))
    finally:
        L.elbow_angle_deg = true_angle
    a = np.array([s.target for s in states if s.target is not None])
    b = np.array([s.target for s in ref if s.target is not None])
    assert np.median(np.linalg.norm(a - b, axis=1)) < 0.01  # squashed depth gives the same robot path
