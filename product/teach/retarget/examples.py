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
