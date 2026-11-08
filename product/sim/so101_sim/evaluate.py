"""Run a trained LeRobot policy (for example ACT) in the twin and score it.

This is the twin version of a capability study: N trials with random part and place
positions, count the successes. Needs lerobot[dataset,training]==0.6.1.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from . import scene, scripted
from .episodes import GRIPPER_RANGE_RAD, TASK, to_lerobot_units
from .replay import FPS


def from_lerobot_units(a: np.ndarray) -> np.ndarray:
    """(6,) degrees / gripper 0-100 -> sim radians."""
    out = np.radians(a.astype(float))
    lo, hi = GRIPPER_RANGE_RAD
    out[5] = lo + a[5] / 100 * (hi - lo)
    return out


def evaluate(policy_path: Path, trials: int = 10, seed: int = 100, max_s: float = 25.0, device: str = "cpu", video_dir: Path | None = None) -> int:
    import torch
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(policy_path, config=cfg)
    policy.eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(policy_path))

    home = scripted.home_q()
    rng = np.random.default_rng(seed)  # different seed from the training demos
    wins = 0
    for i in range(trials):
        part = np.array([rng.uniform(0.18, 0.28), rng.uniform(0.05, 0.15)])
        place = np.array([rng.uniform(0.18, 0.28), rng.uniform(-0.15, -0.05)])
        model, data = scene.load(part_pos=part, place_pos=place)
        renderer = mujoco.Renderer(model, 240, 320)
        scene.set_pose(model, data, home, 1.0)  # same start as the demos
        for _ in range(int(0.3 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        policy.reset()
        substeps = max(1, round(1 / (FPS * model.opt.timestep)))
        frames = []
        for _ in range(int(max_s * FPS)):
            state = np.array([data.joint(j).qpos[0] for j in (*scene.ARM_JOINTS, "gripper")])
            renderer.update_scene(data, camera="top")
            img = renderer.render()
            if video_dir:
                frames.append(img.copy())
            obs = {
                "observation.state": torch.from_numpy(to_lerobot_units(state[None])[0]).unsqueeze(0),
                "observation.images.top": torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0),
                "task": [TASK],
            }
            with torch.inference_mode():
                action = post(policy.select_action(pre(obs)))
            q = from_lerobot_units(action.squeeze(0).cpu().numpy())
            for name, v in zip((*scene.ARM_JOINTS, "gripper"), q):
                data.actuator(name).ctrl[0] = v
            for _ in range(substeps):
                mujoco.mj_step(model, data)
        end = data.body("part").xpos
        err = float(np.linalg.norm(end[:2] - data.site("place_zone").xpos[:2]))
        ok = err < 0.03 and end[2] < scene.PART_SIZE[2]
        wins += ok
        print(f"trial {i}: part ends {err * 1000:.0f} mm from the zone -> {'SUCCESS' if ok else 'FAIL'}")
        if video_dir:
            from .replay import save_video

            video_dir.mkdir(parents=True, exist_ok=True)
            save_video(np.array(frames), video_dir / f"trial_{i:02d}.mp4")
    print(f"policy success in the twin: {wins}/{trials}")
    return wins
