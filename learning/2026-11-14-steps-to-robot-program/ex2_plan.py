"""Exercise 2: what one 'pick' becomes. The plan between the task file and the robot code.

Run (from the repo root, venv active):
    python learning\\2026-11-14-steps-to-robot-program\\ex2_plan.py

Expands take 1's task with make_plan (product/teach/taskprog/plan.py), runs the UR5e reach
check like `check` does (ur.py reach_check), and shows the SO-101 trajectory's worst IK error.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "product" / "teach"))

import numpy as np  # noqa: E402

from taskprog.plan import make_plan, work_to_base  # noqa: E402
from taskprog.task import load_cell, load_task  # noqa: E402
from taskprog.ur import reach_check  # noqa: E402

TP = REPO / "product/teach/taskprog"
ur_cell = load_cell(TP / "cells/ur5e_table.json")
task = load_task(TP / "examples/take1/task.json", ur_cell)

# 1. The plan: every motion, with the location it refers to and the height above it.
print(f"1. Plan for '{task.name}' on {ur_cell.name}: {len(make_plan(task, ur_cell))} motions")
inv = np.linalg.inv(work_to_base(ur_cell))
rows = reach_check(make_plan(task, ur_cell), ur_cell)
print("   step kind    location   height  tool point in work frame (mm)   UR5e joints (deg)")
for r in rows:
    m = r.motion
    where = f"{m.location or '':10} {m.height_m * 1000:5.0f}" if m.location else " " * 16
    xyz = np.round((inv @ m.pose)[:3, 3] * 1000).astype(int).tolist() if m.pose is not None else ""
    q = np.round(np.degrees(r.q)).astype(int).tolist() if m.kind in ("joints", "ptp", "line") else ""
    print(f"   {m.step:4} {m.kind:7} {where}  {str(xyz):28}  {q}   {m.note}" + (f"  !! {r.problem}" if r.problem else ""))

# 2. The SO-101 version of the same plan.
from taskprog.so101 import trajectory  # noqa: E402

so_cell = load_cell(TP / "cells/so101_twin.json")
traj, report = trajectory(load_task(TP / "examples/take1/task.json", so_cell), so_cell)
worst = max(report, key=lambda x: x[1])
print(f"\n2. SO-101: {len(traj.t)} samples at 30 Hz = {traj.t[-1]:.1f} s."
      f" Worst IK miss {worst[1] * 1000:.1f} mm ({worst[0].note}), limit 5 mm.")
