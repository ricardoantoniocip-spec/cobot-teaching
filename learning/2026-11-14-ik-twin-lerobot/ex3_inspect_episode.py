"""Exercise 3: what is inside a twin episode, and what LeRobot (and ACT) will see.

Run after exercise 2, from the same folder:
    python learning\\2026-11-14-ik-twin-lerobot\\ex3_inspect_episode.py l4_out\\episode_000.npz

Reads the .npz written by product/sim/so101_sim/episodes.py (save_npz). If lerobot 0.6.1 is
installed it also builds a one-episode LeRobotDataset in l4_out/lerobot (no upload).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "product" / "sim"))

import numpy as np  # noqa: E402

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("l4_out/episode_000.npz")
ep = np.load(path)
names = [str(n) for n in ep["names"]]
fps = 1 / np.median(np.diff(ep["t"]))

# 1. Shapes: one row per frame.
print(f"1. {path.name}: {len(ep['t'])} frames at {fps:.0f} fps = {ep['t'][-1]:.1f} s")
for k in ("state", "action", "images"):
    print(f"   {k:7} {ep[k].shape} {ep[k].dtype}")
print(f"   names: {names}")

# 2. Units: degrees for joints, 0 (closed) .. 100 (open) for the gripper, like LeRobot's SO-101 follower.
print("\n2. Ranges over the episode")
for i, n in enumerate(names):
    print(f"   {n:18} state {ep['state'][:, i].min():7.1f} .. {ep['state'][:, i].max():6.1f}   action {ep['action'][:, i].min():7.1f} .. {ep['action'][:, i].max():6.1f}")

# 3. The action leads the state: the servo follows the command a little later (lesson 1).
j = names.index("shoulder_lift.pos")
s, a = ep["state"][:, j], ep["action"][:, j]
lags = range(0, 10)
err = [np.abs(s[k:] - a[: len(a) - k]).mean() for k in lags]
best = int(np.argmin(err))
print(f"\n3. shoulder_lift: state matches the action best {best} frame(s) later ({best / fps * 1000:.0f} ms)."
      f" Mean gap {err[0]:.2f} deg with no shift, {err[best]:.2f} deg with the shift.")

# 4. The gripper: when does it close and open?
g = ep["action"][:, names.index("gripper.pos")]
closed = g < 50
edges = np.flatnonzero(np.diff(closed.astype(int)))
print(f"4. gripper command crosses 50 at frames {edges.tolist()} = {np.round(edges / fps, 2).tolist()} s")

# 5. What to-lerobot makes of it (product/sim/so101_sim/episodes.py: npz_to_lerobot).
print("\n5. LeRobotDataset features: observation.state (6), action (6), observation.images.top"
      f" {ep['images'].shape[1:]} video, task text, fps 30, robot_type so101_follower")
try:
    import lerobot  # noqa: F401
except ImportError:
    print("   lerobot is not installed in this venv: skipped building the dataset.")
    print('   To try it: pip install "lerobot[dataset,training]==0.6.1", then run this script again.')
    sys.exit(0)
from so101_sim.episodes import npz_to_lerobot  # noqa: E402

root = path.parent / "lerobot"
if root.exists():
    print(f"   {root} exists already; delete it to build again.")
    sys.exit(0)
npz_to_lerobot([path], "local/take1_twin", root)
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

ds = LeRobotDataset("local/take1_twin", root=root)
print(f"   built {root}: {ds.num_episodes} episode, {ds.num_frames} frames. Frame 0 keys: {sorted(ds[0].keys())}")
