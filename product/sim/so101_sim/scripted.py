"""A scripted "operator" for the twin: makes a pick-and-place hand demo for any part position.

It writes the same HandDemo that the video and the sleeve produce, so the rest of the
pipeline (retarget -> sim replay -> episode) is the real one. Use it to make many varied
demos for training before the real arm and the sleeve exist.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import scene
from .replay import SerialChain  # noqa: F401  (also sets up the import path to product/teach)
from retarget.demo import HandDemo  # noqa: E402
from retarget.kinematics import axis_angle_to_matrix, make_transform  # noqa: E402

# Grasp geometry, measured in the twin: the tool point is the tip of the fixed finger.
# The moving jaw is on the outer side (away from the base), so the tool point sits on the
# inner side of the part.
FINGER_HALF_THICKNESS = 0.006  # m
TIP_HEIGHT_ABOVE_TABLE = 0.008  # m, fingers stop this high above the table
APPROACH_HEIGHT = 0.05  # m above the grasp for the approach and the retreat
HOME = np.array([0.18, 0.0, 0.06])  # tool point at the start and end of every demo


@lru_cache(maxsize=1)
def _chain() -> SerialChain:
    return SerialChain.from_urdf()


def tool_down(yaw: float) -> np.ndarray:
    """Tool rotation with fingers pointing straight down, jaws closing radially at angle yaw."""
    r0 = _chain().fk(np.array([0.0, 0.1, 0.3, 0.9, 0.0]))[:3, :3]
    z = r0[:, 2]
    # Tilt about the base y axis so the finger axis is exactly vertical.
    tilt = np.arctan2(z[0], -z[2])
    r_down = axis_angle_to_matrix(np.array([0.0, 1.0, 0.0]), tilt) @ r0
    return axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), yaw) @ r_down


def grasp_point(part_xy: np.ndarray, part_size=scene.PART_SIZE) -> np.ndarray:
    """Tool point position for a part centred at part_xy (base frame)."""
    xy = np.asarray(part_xy, dtype=float)
    radial = xy / np.linalg.norm(xy)
    inner = xy - radial * (part_size[0] / 2 + FINGER_HALF_THICKNESS)
    return np.array([inner[0], inner[1], TIP_HEIGHT_ABOVE_TABLE])


def pick_place_demo(part_xy, place_xy, part_size=scene.PART_SIZE, hz: float = 30.0, seg_s: float = 1.2) -> HandDemo:
    """Approach, descend, close, lift, carry, descend, open, retreat. Grip 1 = closed."""
    pick = grasp_point(part_xy, part_size)
    place = grasp_point(place_xy, part_size)
    place[2] += 0.004  # release a little higher, so the part is not pushed into the table
    up = np.array([0.0, 0.0, APPROACH_HEIGHT])
    # (position, yaw, grip at the end of the segment)
    keys = [
        (HOME, 0),
        (pick + up, 0),
        (pick, 0),
        (pick, 1),  # close in place
        (pick + up, 1),
        (place + up, 1),
        (place, 1),
        (place, 0),  # open in place
        (place + up, 0),
        (HOME, 0),
    ]
    yaw = lambda p: np.arctan2(p[1], p[0])  # noqa: E731  (HOME has y = 0, so yaw 0)
    ts, poses, grips = [], [], []
    n = int(seg_s * hz)
    for s in range(len(keys) - 1):
        (p0, g0), (p1, g1) = keys[s], keys[s + 1]
        for k in range(n):
            a = 0.5 - 0.5 * np.cos(np.pi * k / n)
            p = p0 + a * (p1 - p0)
            ts.append(len(ts) / hz)
            poses.append(make_transform(tool_down(yaw(p)), p))
            grips.append(float(g1 if g1 != g0 and k > n // 2 else g0))
    ts.append(len(ts) / hz)
    poses.append(make_transform(tool_down(yaw(keys[-1][0])), keys[-1][0]))
    grips.append(0.0)
    return HandDemo(np.array(ts), np.array(poses), np.array(grips))


def home_q() -> np.ndarray:
    """Joint angles of the HOME pose (start of every demo and of every evaluation trial)."""
    from retarget.kinematics import solve_ik_with_restarts

    return solve_ik_with_restarts(_chain(), make_transform(tool_down(0.0), HOME), np.array([0.0, 0.0, 0.3, 1.0, 0.0])).q
