"""Exercise 1: SO-101 forward and inverse kinematics, on take 1's grasp point.

Run (from the repo root, venv active):
    python learning\\2026-11-14-ik-twin-lerobot\\ex1_ik.py

Uses product/teach/retarget/kinematics.py: the chain is read from the official URDF
(hardware/so101/so101_new_calib.urdf), IK is damped least squares with restarts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "product" / "teach"))

import numpy as np  # noqa: E402

from retarget.kinematics import SerialChain, make_transform, solve_ik_with_restarts, tool_down  # noqa: E402

chain = SerialChain.from_urdf()
deg = lambda q: np.round(np.degrees(q), 1).tolist()  # noqa: E731
mm = lambda p: np.round(np.asarray(p) * 1000).astype(int).tolist()  # noqa: E731

# 1. The chain from the URDF: 5 joints, with limits.
print("1. SO-101 joints (from the URDF)")
for n, lo, hi in zip(chain.names, chain.lower, chain.upper):
    print(f"   {n:14} {np.degrees(lo):7.1f} .. {np.degrees(hi):6.1f} deg")

# 2. Forward kinematics: joints -> tool point (tip of the fixed finger).
home = np.radians(json.loads((REPO / "product/teach/taskprog/cells/so101_twin.json").read_text())["home_joints_deg"])
for name, q in (("all zero", np.zeros(5)), ("home", home)):
    t = chain.fk(q)
    print(f"2. FK {name:8}: joints {deg(q)} -> tool point {mm(t[:3, 3])} mm, finger axis {np.round(t[:3, 2], 2).tolist()}")

# 3. Inverse kinematics for take 1's grasp point, fingers straight down.
steps = json.loads((REPO / "product/teach/taskprog/examples/take1/steps.json").read_text())["steps"]
p = np.array(next(s for s in steps if s["step"] == "grasp")["to_xyz"])
yaw = np.arctan2(p[1], p[0])  # the code always closes the jaws along the line from the base
target = make_transform(tool_down(chain, yaw), p)
res = solve_ik_with_restarts(chain, target, home, orientation_weight=0.05)
print(f"\n3. IK to take 1's grasp point {mm(p)} mm, yaw {np.degrees(yaw):.1f} deg (radial)")
print(f"   joints {deg(res.q)}")
print(f"   misses by {res.position_error_m * 1000:.2f} mm and {np.degrees(res.orientation_error_rad):.1f} deg")
print(f"   check with FK: {mm(chain.fk(res.q)[:3, 3])} mm")

# 4. What 5 joints can and cannot do.
#    a) Jaws turned 45 deg about the vertical: wrist_roll does that while the fingers point down.
t45 = make_transform(tool_down(chain, yaw + np.radians(45)), p)
r45 = solve_ik_with_restarts(chain, t45, home, orientation_weight=0.05)
print(f"\n4a. Same point, jaws turned 45 deg about the vertical: misses by {r45.position_error_m * 1000:.1f} mm"
      f" and {np.degrees(r45.orientation_error_rad):.1f} deg. wrist_roll {deg(res.q)[4]} -> {deg(r45.q)[4]} deg")
#    b) Fingers tilted 30 deg sideways, out of the vertical plane through the base: no joint can do that.
radial = np.array([np.cos(yaw), np.sin(yaw), 0.0])
k = np.array([[0, -radial[2], radial[1]], [radial[2], 0, -radial[0]], [-radial[1], radial[0], 0]])
a = np.radians(30)
tilt = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k  # 30 deg about the radial line
tt = make_transform(tilt @ tool_down(chain, yaw), p)
rt = solve_ik_with_restarts(chain, tt, home, orientation_weight=0.05)
print(f"4b. Same point, fingers tilted 30 deg sideways: misses by {rt.position_error_m * 1000:.1f} mm"
      f" and {np.degrees(rt.orientation_error_rad):.1f} deg: a compromise, orientation counts only 0.05.")

# 5. Out of reach.
far = make_transform(tool_down(chain, 0.0), np.array([0.45, 0.0, 0.05]))
rf = solve_ik_with_restarts(chain, far, home, orientation_weight=0.05)
print(f"5. Point 450 mm in front: misses by {rf.position_error_m * 1000:.0f} mm (retarget counts a sample as reached within 5 mm)")
