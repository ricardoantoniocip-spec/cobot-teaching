"""Turn a hand demonstration into an SO-101 joint trajectory.

Steps: hand pose -> tool pose (fixed hand-to-tool offset) -> IK per sample
(warm-started from the previous sample) -> smooth -> slow down to the speed limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .demo import HandDemo, JointTrajectory
from .kinematics import SerialChain, solve_ik_with_restarts


@dataclass
class RetargetConfig:
    # Tool pose = hand pose @ hand_to_tool. Identity until it is calibrated
    # against the leader arm (the leader arm is the ground truth).
    hand_to_tool: np.ndarray = field(default_factory=lambda: np.eye(4))
    orientation_weight: float = 0.05
    reach_tolerance_m: float = 0.005  # a sample counts as reached within 5 mm
    max_joint_speed_rad_s: float = np.radians(30)  # replay speed cap
    smooth_window: int = 5  # samples, centred moving average on joints
    grip_threshold: float = 0.5  # grip >= threshold closes the gripper
    q_start: np.ndarray = field(default_factory=lambda: np.zeros(5))


@dataclass
class RetargetReport:
    samples: int
    reached_fraction: float
    worst_error_mm: float
    time_scale: float  # >1 means the replay is slower than the demo

    def summary(self) -> str:
        return (
            f"{self.samples} samples, {self.reached_fraction:.0%} reached within tolerance, "
            f"worst error {self.worst_error_mm:.1f} mm, replay {self.time_scale:.2f}x demo duration"
        )


def retarget(demo: HandDemo, chain: SerialChain, cfg: RetargetConfig | None = None) -> tuple[JointTrajectory, RetargetReport]:
    cfg = cfg or RetargetConfig()
    q_prev = np.asarray(cfg.q_start, dtype=float)
    qs, errors = [], []
    for pose in demo.poses:
        target = pose @ cfg.hand_to_tool
        res = solve_ik_with_restarts(chain, target, q_prev, orientation_weight=cfg.orientation_weight)
        qs.append(res.q)
        errors.append(res.position_error_m)
        q_prev = res.q
    q = smooth(np.array(qs), cfg.smooth_window)
    t, scale = limit_speed(demo.t - demo.t[0], q, cfg.max_joint_speed_rad_s)
    gripper_open = (demo.grip < cfg.grip_threshold).astype(float)
    errors_arr = np.array(errors)
    traj = JointTrajectory(t=t, q=q, gripper_open=gripper_open, pos_error_m=errors_arr)
    report = RetargetReport(
        samples=len(demo),
        reached_fraction=float(np.mean(errors_arr <= cfg.reach_tolerance_m)),
        worst_error_mm=float(errors_arr.max() * 1000),
        time_scale=scale,
    )
    return traj, report


def smooth(q: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average per joint; keeps the first and last sample exact."""
    if window <= 1 or len(q) < window:
        return q.copy()
    pad = window // 2
    padded = np.pad(q, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    out = np.column_stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(q.shape[1])])
    out[0], out[-1] = q[0], q[-1]
    return out


def limit_speed(t: np.ndarray, q: np.ndarray, max_speed: float) -> tuple[np.ndarray, float]:
    """Stretch time uniformly so no joint moves faster than max_speed (rad/s)."""
    if len(t) < 2:
        return t.copy(), 1.0
    speeds = np.abs(np.diff(q, axis=0)) / np.diff(t)[:, None]
    peak = float(speeds.max())
    scale = max(1.0, peak / max_speed)
    return t * scale, scale
