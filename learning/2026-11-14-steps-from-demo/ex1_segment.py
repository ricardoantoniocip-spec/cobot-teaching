"""Exercise 1: how the demo is cut into steps, on take 1's own numbers.

Run (from the repo root, venv active):
    python learning\\2026-11-14-steps-from-demo\\ex1_segment.py

It rebuilds a hand path that passes through take 1's step points, runs the real cutter
(segment_steps in product/teach/retarget/pose.py), and checks the result against steps.json.
Then it adds one bad frame to the grip signal to show how a rule-based cutter reacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "product" / "teach"))

import numpy as np  # noqa: E402

from retarget import pose  # noqa: E402
from retarget.demo import HandDemo  # noqa: E402
from retarget.kinematics import make_transform  # noqa: E402

doc = json.loads((REPO / "product/teach/taskprog/examples/take1/steps.json").read_text())
steps = doc["steps"]

# 1. What steps.json says (made by the video app on 12 Nov from take 1, grip set with --grips).
print("1. take 1 steps.json")
for i, s in enumerate(steps):
    print(f"   {i}  {s['step']:8} {s['t_start']:6.3f} -> {s['t_end']:6.3f} s  holding={s['holding']!s:5}  to {s['to_xyz']} m")

# 2. Rebuild a 30 Hz hand path through the same points, with the grip closed while holding.
keys_t = [steps[0]["t_start"]] + [s["t_end"] for s in steps if s["step"] in ("reach", "move", "retreat")]
keys_p = [steps[0]["from_xyz"]] + [s["to_xyz"] for s in steps if s["step"] in ("reach", "move", "retreat")]
t = np.arange(0, keys_t[-1] + 1e-9, 1 / 30)
xyz = np.column_stack([np.interp(t, keys_t, np.array(keys_p)[:, k]) for k in range(3)])
t_grasp = next(s["t_start"] for s in steps if s["step"] == "grasp")
t_release = next(s["t_start"] for s in steps if s["step"] == "release")
grip = ((t >= t_grasp - 1e-6) & (t < t_release - 1e-6)).astype(float)
demo = HandDemo(t, np.array([make_transform(np.eye(3), p) for p in xyz]), grip)

again = pose.segment_steps(demo)
print("\n2. segment_steps on the rebuilt path")
print("   " + " > ".join(s["step"] for s in again))
same = [a["step"] == b["step"] and abs(a["t_start"] - b["t_start"]) < 0.04 for a, b in zip(again, steps)]
print(f"   same step names and times as steps.json: {all(same) and len(again) == len(steps)}")

# 3. Where the twin puts the block: the grasp point pushed out by half the block (10 mm)
#    plus half a finger (6 mm), along the line from the robot base (product/sim/so101_sim/scripted.py:39).
pick, place = pose.pick_and_place_points(steps)  # the grasp and release points in steps.json
for name, p in (("block", pick), ("place zone", place)):
    xy = p[:2] + p[:2] / np.linalg.norm(p[:2]) * (0.010 + 0.006)
    print(f"3. {name:10} at {np.round(xy, 4).tolist()} m")
res = json.loads((HERE / "data/take1_result.json").read_text())
print(f"   result.json says part_xy {res['part_xy']}, place_xy {res['place_xy']}")

# 4. One bad frame: the tracker loses the grip for 1/30 s in the middle of the carry.
bad = demo.grip.copy()
bad[np.searchsorted(t, 6.0)] = 0.0
broken = pose.segment_steps(HandDemo(t, demo.poses, bad))
print("\n4. Same demo, grip lost for ONE frame at 6.0 s")
print("   " + " > ".join(s["step"] for s in broken))
print(f"   {len(broken)} steps instead of {len(again)}. pick_and_place_points still uses the first grasp and the next release:")
pp = pose.pick_and_place_points(broken)
print(f"   place point now {np.round(pp[1], 3).tolist()} m instead of {np.round(place, 3).tolist()} m")
