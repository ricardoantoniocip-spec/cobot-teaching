"""Save twin episodes for training.

Units follow LeRobot's SO-101 follower (use_degrees=True), so a policy trained on sim data
sends the same kind of numbers as one trained on real data:
  joints in degrees, gripper 0 (closed) .. 100 (open), names like "shoulder_pan.pos".
This assumes the MuJoCo zero (middle of each joint range, "new calibration") matches the
LeRobot calibration zero. Check it with `retarget check` on the real arm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .scene import ARM_JOINTS

NAMES = [f"{j}.pos" for j in (*ARM_JOINTS, "gripper")]
GRIPPER_RANGE_RAD = (-0.17453, 1.74533)  # gripper joint range in the MJCF
TASK = "Pick the blue block and place it on the green zone."


def to_lerobot_units(x: np.ndarray) -> np.ndarray:
    """(N, 6) sim radians -> degrees for joints, 0-100 for the gripper."""
    out = np.degrees(x).astype(np.float32)
    lo, hi = GRIPPER_RANGE_RAD
    out[:, 5] = np.clip((x[:, 5] - lo) / (hi - lo) * 100, 0, 100)
    return out


def save_npz(episode: dict, path: Path) -> None:
    """One file per episode: t, state, action (LeRobot units), images (N, H, W, 3) uint8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        t=episode["t"],
        state=to_lerobot_units(episode["state"]),
        action=to_lerobot_units(episode["action"]),
        images=episode.get("images", np.zeros((0, 1, 1, 3), np.uint8)),
        names=np.array(NAMES),
    )


def npz_to_lerobot(files: list[Path], repo_id: str, root: Path, fps: int = 30, push: bool = False) -> Path:
    """Convert saved episodes into a LeRobotDataset (needs lerobot==0.6.1 installed).

    The result trains with the normal LeRobot command, for example:
      lerobot-train --dataset.repo_id=<repo_id> --dataset.root=<root> --policy.type=act ...
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    first = np.load(files[0])
    h, w = first["images"].shape[1:3]
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": NAMES},
        "action": {"dtype": "float32", "shape": (6,), "names": NAMES},
        "observation.images.top": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]},
    }
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=root, robot_type="so101_follower", use_videos=True)
    for f in files:
        ep = np.load(f)
        for s, a, im in zip(ep["state"], ep["action"], ep["images"]):
            ds.add_frame({"observation.state": s, "action": a, "observation.images.top": im, "task": TASK})
        ds.save_episode()
    ds.finalize()
    if push:
        ds.push_to_hub(private=True)
    return root
