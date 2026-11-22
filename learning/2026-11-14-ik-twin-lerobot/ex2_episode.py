"""Exercise 2: turn take 1's pick and place points into a training episode in the twin.

Run (from the repo root, venv active). About 1-2 minutes on a laptop CPU:
    python learning\\2026-11-14-ik-twin-lerobot\\ex2_episode.py

This is the 'steps' half of `so101_sim from-video` (product/sim/so101_sim/from_video.py:79-90):
clean scripted motion between the block and the zone -> IK (retarget) -> MuJoCo replay -> save
only if it succeeds. Writes l4_out/episode_000.npz and l4_out/replay_steps.mp4 in the current folder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "product" / "sim"))

import numpy as np  # noqa: E402

from so101_sim import scene, scripted  # noqa: E402  (also puts product/teach on the path)
from so101_sim.episodes import save_npz  # noqa: E402
from so101_sim.replay import replay, save_video  # noqa: E402
from retarget.retarget import RetargetConfig, retarget  # noqa: E402

res_json = json.loads((HERE / "data/take1_result.json").read_text())
part_xy, place_xy = np.array(res_json["part_xy"]), np.array(res_json["place_xy"])
print(f"block at {part_xy.tolist()} m, place zone at {place_xy.tolist()} m (from take 1 result.json)")

# 1. A clean hand path: home, above, down, close, up, across, down, open, up, home.
demo = scripted.pick_place_demo(part_xy, place_xy)
print(f"1. scripted path: {len(demo)} samples, {demo.t[-1]:.1f} s, grip closed for {demo.grip.sum() / 30:.1f} s")

# 2. IK for every sample (same settings as from_video.py:72).
cfg = RetargetConfig(q_start=scripted.home_q(), max_joint_speed_rad_s=np.radians(60))
traj, rep = retarget(demo, scripted._chain(), cfg)
print(f"2. retarget: {rep.summary()}")

# 3. Physics replay with the top camera, like the training data.
model, data = scene.load(part_pos=part_xy, place_pos=place_xy)
r = replay(traj, model, data, camera="top")
print("3. twin replay:\n   " + r.report().replace("\n", "\n   "))

# 4. Save only a success.
out = Path("l4_out")
out.mkdir(exist_ok=True)
if r.success:
    save_npz(r.episode, out / "episode_000.npz")
    save_video(r.episode["images"], out / "replay_steps.mp4")
    print(f"4. saved {(out / 'episode_000.npz').resolve()} and replay_steps.mp4")
else:
    print("4. not saved: the replay failed")
print(f"   take 1 result.json said: steps success {res_json['steps']['success']}, {res_json['steps']['place_err_mm']} mm from the zone")
