"""Read and write hand demonstrations and joint trajectories as CSV.

Hand demo CSV (one row per sample, poses in the robot base frame, metres):
    t, x, y, z, qw, qx, qy, qz, grip
grip: 0 = hand open, 1 = hand closed (from EMG/FSR later, or a grip-events file now).

Joint trajectory CSV:
    t, shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper_open, pos_error_mm
Joint angles in radians, gripper_open from 0 (closed) to 1 (open).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .kinematics import ARM_JOINTS, make_transform

DEMO_COLUMNS = ("t", "x", "y", "z", "qw", "qx", "qy", "qz", "grip")


def quat_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=float)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(r: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix to (qw, qx, qy, qz), numerically stable branch."""
    tr = np.trace(r)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
        v = [0.0, 0.0, 0.0]
        v[i] = 0.25 * s
        v[j] = (r[j, i] + r[i, j]) / s
        v[k] = (r[k, i] + r[i, k]) / s
        q = ((r[k, j] - r[j, k]) / s, *v)
    w, x, y, z = q
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    return float(w), float(x), float(y), float(z)


@dataclass
class HandDemo:
    t: np.ndarray  # (N,) seconds
    poses: np.ndarray  # (N, 4, 4) hand pose in robot base frame
    grip: np.ndarray  # (N,) 0 open .. 1 closed

    def __len__(self) -> int:
        return len(self.t)


def read_demo(path: Path) -> HandDemo:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    missing = set(DEMO_COLUMNS) - set(rows[0].keys()) if rows else set(DEMO_COLUMNS)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    t = np.array([float(r["t"]) for r in rows])
    poses = np.array(
        [
            make_transform(
                quat_to_matrix(float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])),
                np.array([float(r["x"]), float(r["y"]), float(r["z"])]),
            )
            for r in rows
        ]
    )
    grip = np.array([float(r["grip"]) for r in rows])
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{path}: time column must strictly increase")
    return HandDemo(t, poses, grip)


def write_demo(path: Path, demo: HandDemo) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(DEMO_COLUMNS)
        for t, pose, g in zip(demo.t, demo.poses, demo.grip):
            w.writerow([f"{t:.4f}", *(f"{v:.5f}" for v in pose[:3, 3]), *(f"{v:.6f}" for v in matrix_to_quat(pose[:3, :3])), f"{g:.3f}"])


@dataclass
class JointTrajectory:
    t: np.ndarray  # (N,)
    q: np.ndarray  # (N, 5) radians
    gripper_open: np.ndarray  # (N,) 0 closed .. 1 open
    pos_error_m: np.ndarray  # (N,) IK position error per sample

    def sample(self, t: float) -> tuple[np.ndarray, float]:
        """Linear interpolation of joints and gripper at time t (clamped)."""
        q = np.array([np.interp(t, self.t, self.q[:, i]) for i in range(self.q.shape[1])])
        return q, float(np.interp(t, self.t, self.gripper_open))


def write_trajectory(path: Path, traj: JointTrajectory) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", *ARM_JOINTS, "gripper_open", "pos_error_mm"])
        for t, q, g, e in zip(traj.t, traj.q, traj.gripper_open, traj.pos_error_m):
            w.writerow([f"{t:.4f}", *(f"{v:.5f}" for v in q), f"{g:.3f}", f"{e * 1000:.2f}"])


def read_trajectory(path: Path) -> JointTrajectory:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return JointTrajectory(
        t=np.array([float(r["t"]) for r in rows]),
        q=np.array([[float(r[j]) for j in ARM_JOINTS] for r in rows]),
        gripper_open=np.array([float(r["gripper_open"]) for r in rows]),
        pos_error_m=np.array([float(r["pos_error_mm"]) / 1000 for r in rows]),
    )
