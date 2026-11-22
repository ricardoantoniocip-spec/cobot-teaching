"""Exercise 3: body mode. How your arm is shrunk onto the SO-101.

Run (from the repo root, venv active). No video needed:
    python learning\\2026-11-14-vision-front-end\\ex3_body_mode.py

Redoes the scale sum from product/teach/retarget/pose.py (frames_to_demo, body mode) by hand
and checks it against the real function. Then turns the "camera" to show that body mode
uses your own axes, not the camera's.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "product" / "teach"))

import numpy as np  # noqa: E402

from retarget import pose  # noqa: E402
from retarget.examples import synthetic_operator_frames  # noqa: E402
from retarget.kinematics import SerialChain, rpy_to_matrix  # noqa: E402

chain = SerialChain.from_urdf()
cfg = pose.PoseConfig()
frames = synthetic_operator_frames()
j = pose.POSE["right"]

# 1. The operator's own axes (forward, left, up), from shoulders and hips.
axes = pose.body_axes(frames)
sh = np.median([f.pose_world[j["shoulder"]] for f in frames], axis=0)
el = np.array([(f.pose_world[j["elbow"]] - sh) @ axes for f in frames])
wr = np.array([(f.pose_world[j["wrist"]] - sh) @ axes for f in frames])
pi = np.array([w + ((f.hand_world[4] + f.hand_world[8]) / 2 - f.hand_world[0]) @ axes for w, f in zip(wr, frames)])

# 2. Arm length = upper arm + forearm + wrist-to-pinch (medians over the video).
upper = np.median(np.linalg.norm(el, axis=1))
fore = np.median(np.linalg.norm(wr - el, axis=1))
hand = np.median(np.linalg.norm(pi - wr, axis=1))
scale = cfg.robot_reach_m / (upper + fore + hand)
print("1. Your arm (made-up operator), metres")
print(f"   upper arm {upper:.3f} + forearm {fore:.3f} + wrist-to-pinch {hand:.3f} = {upper + fore + hand:.3f}")
print(f"   scale = robot reach {cfg.robot_reach_m} / {upper + fore + hand:.3f} = {scale:.3f}")

# 3. The real function, for comparison. It may shrink more to stay within max_radius_m.
res = pose.frames_to_demo(frames, chain, cfg)
print(f"2. frames_to_demo says: mode {res.mode}, scale {res.scale:.3f}"
      + ("  (same: the path fits)" if abs(res.scale - scale) < 1e-6 else "  (smaller: shrunk to stay within reach)"))

p = res.demo.poses[:, :3, 3]
steps = pose.segment_steps(res.demo)
pick, place = pose.pick_and_place_points(steps)
print(f"3. Pick at  {np.round(pick * 1000).astype(int)} mm in the robot base frame (x forward, y left, z up)")
print(f"   Place at {np.round(place * 1000).astype(int)} mm")
print(f"   Lowest tool height {p[:, 2].min() * 1000:.0f} mm (table_z_m = {cfg.table_z_m * 1000:.0f} mm: your lowest hand point becomes the table)")
print(f"   Farthest from the base {np.linalg.norm(p[:, :2], axis=1).max() * 1000:.0f} mm (limit max_radius_m = {cfg.max_radius_m * 1000:.0f} mm)")

# 4. Turn the camera 40 degrees around the operator. Body mode should not care.
turned = synthetic_operator_frames()
rot = rpy_to_matrix(0.0, np.radians(40), 0.0)
for f in turned:
    f.pose_world, f.hand_world = f.pose_world @ rot.T, f.hand_world @ rot.T
p2 = pose.frames_to_demo(turned, chain, cfg).demo.poses[:, :3, 3]
print(f"4. Camera turned 40 deg: path changes by at most {np.abs(p2 - p).max() * 1000:.4f} mm")

# 5. Reach far forward: the scale must drop so the robot can still get there.
far = synthetic_operator_frames(pick_body=(0.60, 0.30, -0.40))
res3 = pose.frames_to_demo(far, chain, cfg)
print(f"5. Operator reaches 60 cm forward: scale {res3.scale:.3f} (was {res.scale:.3f})")
