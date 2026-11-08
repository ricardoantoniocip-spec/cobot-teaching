"""Play a retarget trajectory (traj.csv) on the digital twin and check the result.

What it checks, the same way you would on the real cell:
  1. tracking: how far the simulated joints lag the commanded joints (servo model from upstream)
  2. tool point: where the gripper tip really went vs where retarget's IK said it would go
  3. task: did the part end up in the place zone?
It can also save a video and an episode file (joints, actions, camera images) for training.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import scene

# The retarget package lives in product/teach. Import it from there so both use one copy.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "teach"))
from retarget.demo import JointTrajectory, read_trajectory  # noqa: E402
from retarget.kinematics import SerialChain  # noqa: E402

FPS = 30  # same as the SO-101 cameras and LeRobot recordings


@dataclass
class ReplayResult:
    max_track_err_deg: float
    max_tcp_err_mm: float  # simulated tool point vs IK tool point (commanded joints)
    part_start: np.ndarray
    part_end: np.ndarray
    place_err_mm: float
    success: bool
    episode: dict = field(repr=False, default_factory=dict)

    def report(self) -> str:
        return (
            f"joint tracking, worst: {self.max_track_err_deg:.1f} deg\n"
            f"tool point vs IK, worst: {self.max_tcp_err_mm:.1f} mm\n"
            f"part moved {np.linalg.norm(self.part_end[:2] - self.part_start[:2]) * 1000:.0f} mm, "
            f"ends {self.place_err_mm:.0f} mm from the place zone centre\n"
            f"task: {'SUCCESS' if self.success else 'FAIL'}"
        )


def replay(
    traj: JointTrajectory,
    model: mujoco.MjModel | None = None,
    data: mujoco.MjData | None = None,
    camera: str | None = None,
    settle_s: float = 1.0,
    place_tol_m: float = 0.03,
) -> ReplayResult:
    """Run the trajectory at FPS with physics in between. camera='top' also records images."""
    if model is None:
        model, data = scene.load()
    chain = SerialChain.from_urdf()
    renderer = mujoco.Renderer(model, 240, 320) if camera else None

    q0, g0 = traj.sample(0.0)
    scene.set_pose(model, data, q0, g0)
    for _ in range(int(0.3 / model.opt.timestep)):  # let the part settle on the table
        mujoco.mj_step(model, data)
    part_start = data.body("part").xpos.copy()

    substeps = max(1, round(1 / (FPS * model.opt.timestep)))
    n = int((traj.t[-1] + settle_s) * FPS) + 1
    ep = {k: [] for k in ("t", "state", "action", "images")}
    track, tcp = [], []
    for k in range(n):
        t = k / FPS
        q_cmd, g = traj.sample(t)
        action = np.append(q_cmd, scene.gripper_angle(g))
        state = np.array([data.joint(j).qpos[0] for j in (*scene.ARM_JOINTS, "gripper")])
        ep["t"].append(t)
        ep["state"].append(state)
        ep["action"].append(action)
        if renderer:
            renderer.update_scene(data, camera=camera)
            ep["images"].append(renderer.render().copy())
        for name, v in zip((*scene.ARM_JOINTS, "gripper"), action):
            data.actuator(name).ctrl[0] = v
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        q_sim = np.array([data.joint(j).qpos[0] for j in scene.ARM_JOINTS])
        track.append(np.abs(q_sim - q_cmd).max())
        tcp.append(np.linalg.norm(data.site(scene.TCP_SITE).xpos - chain.fk(q_cmd)[:3, 3]))

    part_end = data.body("part").xpos.copy()
    place = data.site("place_zone").xpos
    place_err = float(np.linalg.norm(part_end[:2] - place[:2]))
    on_table = part_end[2] < scene.PART_SIZE[2]  # resting, not stuck in the gripper
    episode = {k: np.array(v) for k, v in ep.items() if v}
    return ReplayResult(
        max_track_err_deg=float(np.degrees(max(track))),
        max_tcp_err_mm=float(max(tcp) * 1000),
        part_start=part_start,
        part_end=part_end,
        place_err_mm=place_err * 1000,
        success=bool(place_err < place_tol_m and on_table),
        episode=episode,
    )


def save_video(images: np.ndarray, path: Path, fps: int = FPS) -> None:
    import cv2  # opencv is already in product/teach/requirements.txt

    h, w = images.shape[1:3]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for im in images:
        out.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    out.release()


def replay_file(path: Path, **kw) -> ReplayResult:
    return replay(read_trajectory(path), **kw)
