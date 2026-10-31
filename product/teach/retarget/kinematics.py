"""SO-101 forward and inverse kinematics, read from the official URDF.

No link lengths are typed here: every joint frame comes from the URDF file.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_URDF = REPO_ROOT / "hardware" / "so101" / "so101_new_calib.urdf"
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues formula for a unit axis."""
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def matrix_to_rotvec(r: np.ndarray) -> np.ndarray:
    """Rotation matrix to rotation vector (axis * angle)."""
    cos_a = np.clip((np.trace(r) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(cos_a)
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-6:
        # Near 180 degrees: take the axis from the largest diagonal term.
        i = int(np.argmax(np.diag(r)))
        axis = r[:, i] + np.eye(3)[i]
        return axis / np.linalg.norm(axis) * angle
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return w / (2 * np.sin(angle)) * angle


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


@dataclass(frozen=True)
class Joint:
    name: str
    origin: np.ndarray  # 4x4 parent -> joint frame at q = 0
    axis: np.ndarray  # unit axis in the joint frame
    revolute: bool
    lower: float
    upper: float


class SerialChain:
    """A serial chain from base_link to a tip link, loaded from a URDF."""

    def __init__(self, joints: list[Joint]):
        self.joints = joints
        self.active = [j for j in joints if j.revolute]
        self.names = [j.name for j in self.active]
        self.lower = np.array([j.lower for j in self.active])
        self.upper = np.array([j.upper for j in self.active])

    @classmethod
    def from_urdf(
        cls,
        path: Path = DEFAULT_URDF,
        base: str = "base_link",
        tip: str = "gripper_frame_link",
    ) -> "SerialChain":
        root = ET.parse(path).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain: list[Joint] = []
        link = tip
        while link != base:
            if link not in by_child:
                raise ValueError(f"No joint leads to link '{link}' in {path}")
            el = by_child[link]
            chain.append(_parse_joint(el))
            link = el.find("parent").get("link")
        chain.reverse()
        return cls(chain)

    def clip(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.lower, self.upper)

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Tool pose (4x4) in the base frame for joint angles q (radians)."""
        return self._fk_frames(q)[0]

    def _fk_frames(self, q: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        t = np.eye(4)
        axes, points = [], []
        i = 0
        for j in self.joints:
            t = t @ j.origin
            if j.revolute:
                axes.append(t[:3, :3] @ j.axis)
                points.append(t[:3, 3].copy())
                t = t @ make_transform(axis_angle_to_matrix(j.axis, q[i]), np.zeros(3))
                i += 1
        return t, axes, points

    def jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Geometric Jacobian (6 x n, linear rows first) and the tool pose."""
        tool, axes, points = self._fk_frames(q)
        p = tool[:3, 3]
        jac = np.zeros((6, len(self.active)))
        for i, (z, o) in enumerate(zip(axes, points)):
            jac[:3, i] = np.cross(z, p - o)
            jac[3:, i] = z
        return jac, tool


def _parse_joint(el: ET.Element) -> Joint:
    origin_el = el.find("origin")
    xyz = np.array([float(v) for v in origin_el.get("xyz", "0 0 0").split()])
    rpy = [float(v) for v in origin_el.get("rpy", "0 0 0").split()]
    kind = el.get("type")
    revolute = kind in ("revolute", "continuous")
    axis_el = el.find("axis")
    axis = np.array([float(v) for v in axis_el.get("xyz").split()]) if axis_el is not None else np.zeros(3)
    if revolute:
        axis = axis / np.linalg.norm(axis)
    limit = el.find("limit")
    lower = float(limit.get("lower", -np.pi)) if limit is not None else -np.pi
    upper = float(limit.get("upper", np.pi)) if limit is not None else np.pi
    return Joint(el.get("name"), make_transform(rpy_to_matrix(*rpy), xyz), axis, revolute, lower, upper)


@dataclass
class IKResult:
    q: np.ndarray
    position_error_m: float
    orientation_error_rad: float


def solve_ik(
    chain: SerialChain,
    target: np.ndarray,
    q0: np.ndarray,
    orientation_weight: float = 0.05,
    damping: float = 0.01,
    iterations: int = 200,
    tolerance_m: float = 1e-4,
) -> IKResult:
    """Damped least squares IK with weighted position and orientation error.

    The SO-101 has 5 joints, so it cannot match every orientation. A low
    orientation weight makes position win: the arm reaches the point and gets
    as close as it can to the hand's orientation.
    """
    w = np.sqrt(np.array([1.0, 1.0, 1.0] + [orientation_weight] * 3))
    q = chain.clip(np.asarray(q0, dtype=float).copy())
    for _ in range(iterations):
        jac, tool = chain.jacobian(q)
        e_pos = target[:3, 3] - tool[:3, 3]
        e_rot = matrix_to_rotvec(target[:3, :3] @ tool[:3, :3].T)
        if np.linalg.norm(e_pos) < tolerance_m and orientation_weight == 0:
            break
        err = np.concatenate([e_pos, e_rot]) * w
        jw = jac * w[:, None]
        dq = jw.T @ np.linalg.solve(jw @ jw.T + damping**2 * np.eye(6), err)
        step = np.linalg.norm(dq)
        if step > 0.2:  # keep each step small and stable
            dq *= 0.2 / step
        q = chain.clip(q + dq)
        if step < 1e-7:
            break
    return _result(chain, target, q)


def solve_ik_with_restarts(
    chain: SerialChain,
    target: np.ndarray,
    q0: np.ndarray,
    accept_m: float = 0.002,
    restarts: int = 8,
    seed: int = 0,
    **kwargs: float,
) -> IKResult:
    """Try the warm start first, then random starts if the error stays large."""
    best = solve_ik(chain, target, q0, **kwargs)
    rng = np.random.default_rng(seed)
    for _ in range(restarts):
        if best.position_error_m <= accept_m:
            break
        start = rng.uniform(chain.lower * 0.9, chain.upper * 0.9)
        trial = solve_ik(chain, target, start, **kwargs)
        if trial.position_error_m < best.position_error_m:
            best = trial
    return best


def _result(chain: SerialChain, target: np.ndarray, q: np.ndarray) -> IKResult:
    tool = chain.fk(q)
    return IKResult(
        q=q,
        position_error_m=float(np.linalg.norm(target[:3, 3] - tool[:3, 3])),
        orientation_error_rad=float(np.linalg.norm(matrix_to_rotvec(target[:3, :3] @ tool[:3, :3].T))),
    )
