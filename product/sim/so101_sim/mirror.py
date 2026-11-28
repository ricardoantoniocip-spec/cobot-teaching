"""Drive the real SO-101 from the twin: move the twin in the 3D viewer, the physical arm follows.

    python -m so101_sim view --port COM4

The twin starts in the real arm's pose. The real arm starts PAUSED (it holds still).
Open the "Control" panel on the right of the viewer and drag the joint sliders: the twin moves.
Press SPACE (with the viewer window focused) to let the real arm follow, SPACE again to stop it.
Joints are capped at --max-speed (default 30 deg/s). The Teach panel (teach_panel.py) opens with it. Close the window to quit: the motors then
switch off and the arm goes limp, so keep a hand near it.
"""

from __future__ import annotations

import time

import numpy as np

from . import scene
from .replay import SerialChain  # noqa: F401  (sets up the import path to product/teach)

SPACE = 32


def twin_joints(data) -> tuple[np.ndarray, float]:
    q = np.array([data.joint(j).qpos[0] for j in scene.ARM_JOINTS])
    g = (data.joint("gripper").qpos[0] - scene.GRIPPER_CLOSED_RAD) / (scene.GRIPPER_OPEN_RAD - scene.GRIPPER_CLOSED_RAD)
    return q, float(np.clip(g, 0.0, 1.0))


def step_towards(q_cmd: np.ndarray, q_goal: np.ndarray, max_step: float) -> np.ndarray:
    return q_cmd + np.clip(q_goal - q_cmd, -max_step, max_step)


def run(port: str, robot_id: str = "follower", max_speed_deg_s: float = 30.0, hz: float = 30.0) -> int:
    """Kept for old scripts: the view with the Teach panel does the mirroring now."""
    from .teach_panel import run as run_teach

    return run_teach(port, robot_id, max_speed_deg_s, hz=hz)
