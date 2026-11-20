"""A synthetic pick-and-place hand demo, to test the pipeline before the sleeve exists."""

from __future__ import annotations

import numpy as np

from .demo import HandDemo
from .kinematics import SerialChain


def example_pick_place(chain: SerialChain, hz: float = 30.0) -> HandDemo:
    """Hand moves above a part, down, grips, lifts, moves across, places, releases.

    Waypoints are made with forward kinematics from joint poses, so every pose is
    reachable. A real demo will not be this kind.
    """
    # Joint waypoints (rad): above pick, at pick, above pick, above place, at place, above place.
    waypoints = np.array(
        [
            [-0.5, -0.3, 0.3, 1.0, 0.0],
            [-0.5, 0.1, 0.3, 0.9, 0.0],
            [-0.5, -0.3, 0.3, 1.0, 0.0],
            [0.5, -0.3, 0.3, 1.0, 0.0],
            [0.5, 0.1, 0.3, 0.9, 0.0],
            [0.5, -0.3, 0.3, 1.0, 0.0],
        ]
    )
    seg_s = 1.5
    grip_by_segment = [0, 1, 1, 1, 0]  # closed while carrying: from reaching the pick to reaching the place
    ts, poses, grips = [], [], []
    n = int(seg_s * hz)
    for s in range(len(waypoints) - 1):
        for k in range(n):
            a = 0.5 - 0.5 * np.cos(np.pi * k / n)  # smooth start and stop
            q = waypoints[s] + a * (waypoints[s + 1] - waypoints[s])
            ts.append((s * n + k) / hz)
            poses.append(chain.fk(q))
            grips.append(float(grip_by_segment[s]))
    ts.append(len(ts) / hz)
    poses.append(chain.fk(waypoints[-1]))
    grips.append(0.0)
    return HandDemo(np.array(ts), np.array(poses), np.array(grips))


def synthetic_operator_frames(pick_body=(0.40, 0.22, -0.40), place_body=(0.40, -0.16, -0.40), hz: float = 30.0, seg_s: float = 1.0, noise_m: float = 0.0, seed: int = 0):
    """Landmarks of a made-up operator doing a pick and place, as the pose tracker would give them.

    For tests without a real video. The operator faces the camera. pick_body / place_body are the
    grasp point relative to the right shoulder in the operator's axes (forward, left, up), metres.
    Returns a list of pose.ArmFrame.
    """
    from .pose import ArmFrame, grasp_point

    rng = np.random.default_rng(seed)
    axes = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])  # columns: forward, left, up in camera axes
    world = np.zeros((33, 3))
    world[12], world[11] = [-0.18, -0.50, 0.0], [0.18, -0.50, 0.0]  # right and left shoulder
    world[24], world[23] = [-0.10, 0.0, 0.0], [0.10, 0.0, 0.0]  # hips
    shoulder = world[12]
    pick, place = np.array(pick_body), np.array(place_body)
    up = np.array([0.0, 0.0, 0.12])
    rest = np.array([0.25, 0.05, -0.30])
    keys = [(rest, 0), (pick + up, 0), (pick, 0), (pick, 1), (pick + up, 1), (place + up, 1), (place, 1), (place, 0), (place + up, 0), (rest, 0)]
    frames = []
    n = int(seg_s * hz)
    for s in range(len(keys) - 1):
        (p0, g0), (p1, g1) = keys[s], keys[s + 1]
        for k in range(n):
            a = 0.5 - 0.5 * np.cos(np.pi * k / n)
            closed = g1 if g1 != g0 and k > n // 2 else g0
            pinch = shoulder + axes @ (p0 + a * (p1 - p0)) + rng.normal(0, noise_m, 3)
            fwd, side, down = axes @ np.array([0.8, 0.0, -0.6]), axes[:, 1], axes @ np.array([-0.6, 0.0, -0.8])
            hand = _hand_points(bool(closed), fwd, side, down)
            hand += pinch - grasp_point(hand)  # put the grasp point where the demo wants it
            wrist = hand[0]
            w = world.copy()
            w[16] = wrist
            d = wrist - shoulder
            dist = np.linalg.norm(d)
            drop = np.sqrt(max(0.31**2 - (dist / 2) ** 2, 0.0))  # 31 cm upper arm and forearm, elbow hangs down
            w[14] = shoulder + d / 2 + drop * np.array([0.0, 1.0, 0.0])
            frames.append(ArmFrame(t=len(frames) / hz, pose_world=w, pose_vis=np.ones(33), pose_px=np.zeros((33, 2)), hand_world=hand - hand.mean(axis=0), hand_px=np.zeros((21, 2))))
    return frames


def _hand_points(closed: bool, fwd: np.ndarray, side: np.ndarray, down: np.ndarray) -> np.ndarray:
    """21 hand landmarks (MediaPipe numbering) with the wrist at the origin. Open: fingers straight
    along fwd. Closed: the four fingers curl towards the palm (down) and the thumb stays out to the side, as on a whole-hand grasp."""
    hand = np.zeros((21, 3))
    lengths = (0.040, 0.025, 0.020)
    curl = np.radians([55.0, 125.0, 190.0]) if closed else np.zeros(3)
    for f, off in enumerate((0.03, 0.01, -0.01, -0.03)):
        p = 0.09 * fwd + off * side
        hand[5 + 4 * f] = p
        for k in range(3):
            p = p + lengths[k] * (np.cos(curl[k]) * fwd + np.sin(curl[k]) * down)
            hand[6 + 4 * f + k] = p
    thumb_tip = (0.04 * fwd + 0.07 * side + 0.02 * down) if closed else (0.07 * fwd + 0.08 * side)
    for k in range(1, 5):
        hand[k] = thumb_tip * k / 4
    return hand
