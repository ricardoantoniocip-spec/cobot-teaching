"""Drive the real SO-101 from the twin: move the twin in the 3D viewer, the physical arm follows.

    python -m so101_sim view --port COM4

The twin starts in the real arm's pose. The real arm starts PAUSED (it holds still).
Open the "Control" panel on the right of the viewer and drag the joint sliders: the twin moves.
Press SPACE (with the viewer window focused) to let the real arm follow, SPACE again to stop it.
Joints are capped at --max-speed (default 30 deg/s). Close the window to quit: the motors then
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
    import mujoco
    import mujoco.viewer
    from retarget import robot as rr

    model, data = scene.load()
    robot = rr.connect(port, robot_id)
    mapping = rr.JointMapping.load()
    q_cmd = rr.current_joints(robot, mapping)
    g_cmd = 1.0
    scene.set_pose(model, data, q_cmd, g_cmd)  # twin starts where the real arm is

    state = {"follow": False}

    def on_key(key: int) -> None:
        if key == SPACE:
            state["follow"] = not state["follow"]
            print("real arm FOLLOWS the twin" if state["follow"] else "real arm PAUSED (holds still)", flush=True)

    print(f"SO-101 on {port}: connected. PAUSED. Drag the sliders in the viewer's Control panel; SPACE = follow / pause.")
    max_step = np.radians(max_speed_deg_s) / hz
    physics_steps = max(1, round((1.0 / hz) / model.opt.timestep))
    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
            while viewer.is_running():
                t0 = time.monotonic()
                for _ in range(physics_steps):
                    mujoco.mj_step(model, data)
                if state["follow"]:
                    q_twin, g_twin = twin_joints(data)
                    q_cmd = step_towards(q_cmd, q_twin, max_step)
                    g_cmd = float(np.clip(g_twin, g_cmd - 0.05, g_cmd + 0.05))
                robot.send_action(mapping.to_action(q_cmd, g_cmd))  # paused: keep holding the last pose
                viewer.sync()
                time.sleep(max(0.0, 1.0 / hz - (time.monotonic() - t0)))
    finally:
        print("Closing: the motors switch off now, the arm goes limp.")
        robot.disconnect()
    return 0
