"""Send a joint trajectory to the SO-101 follower through LeRobot.

UNTESTED ON HARDWARE. Written against LeRobot 0.6.1 (lerobot.robots.so_follower). First run: --check, then replay at the default speed cap
with a hand ready on the power switch.

Units: the follower is created with use_degrees=True, so body joints are in
degrees and the gripper is 0-100. The URDF zero pose is assumed to match the
LeRobot calibration zero; JointMapping lets you correct sign and offset per
joint once --check shows a difference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .demo import JointTrajectory
from .kinematics import ARM_JOINTS


@dataclass
class JointMapping:
    signs: dict[str, float] = field(default_factory=lambda: {j: 1.0 for j in ARM_JOINTS})
    offsets_deg: dict[str, float] = field(default_factory=lambda: {j: 0.0 for j in ARM_JOINTS})
    gripper_open_pct: float = 40.0
    gripper_closed_pct: float = 5.0

    def to_action(self, q: np.ndarray, gripper_open: float) -> dict[str, float]:
        action = {f"{j}.pos": self.signs[j] * float(np.degrees(v)) + self.offsets_deg[j] for j, v in zip(ARM_JOINTS, q)}
        g = self.gripper_closed_pct + gripper_open * (self.gripper_open_pct - self.gripper_closed_pct)
        action["gripper.pos"] = g
        return action

    def from_observation(self, obs: dict[str, float]) -> np.ndarray:
        return np.array([np.radians((obs[f"{j}.pos"] - self.offsets_deg[j]) / self.signs[j]) for j in ARM_JOINTS])


def connect(port: str, robot_id: str = "follower", max_step_deg: float = 5.0):
    """Connect to the follower arm. Needs LeRobot installed and the arm calibrated."""
    try:  # LeRobot 0.6.x
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:  # older LeRobot
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

    # max_relative_target: LeRobot clips every single step to at most this many degrees.
    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, use_degrees=True, max_relative_target=max_step_deg))
    robot.connect()
    return robot


def current_joints(robot, mapping: JointMapping) -> np.ndarray:
    return mapping.from_observation(robot.get_observation())


def move_to(robot, mapping: JointMapping, q_from: np.ndarray, q_to: np.ndarray, gripper_open: float, max_speed_rad_s: float, hz: float = 30.0) -> None:
    """Straight joint-space move at no more than max_speed."""
    duration = max(float(np.abs(q_to - q_from).max()) / max_speed_rad_s, 1.0)
    steps = int(duration * hz)
    for k in range(1, steps + 1):
        q = q_from + (q_to - q_from) * k / steps
        robot.send_action(mapping.to_action(q, gripper_open))
        time.sleep(1 / hz)


def play(robot, traj: JointTrajectory, mapping: JointMapping, max_speed_rad_s: float, hz: float = 30.0) -> None:
    """Go slowly to the first pose, then play the trajectory in real time."""
    q0, g0 = traj.sample(0.0)
    move_to(robot, mapping, current_joints(robot, mapping), q0, g0, max_speed_rad_s, hz)
    start = time.monotonic()
    while True:
        t = time.monotonic() - start
        if t > traj.t[-1]:
            break
        q, g = traj.sample(t)
        robot.send_action(mapping.to_action(q, g))
        time.sleep(1 / hz)
