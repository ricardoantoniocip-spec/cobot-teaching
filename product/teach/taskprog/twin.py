"""Play the SO-101 trajectory of a task in the MuJoCo twin (product/sim) and score it.

The twin is a flat table with one block, so the task must have exactly one pick and one place,
both on surfaces at height 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .so101 import trajectory
from .task import Cell, Task

SIM_DIR = Path(__file__).resolve().parents[2] / "sim"


def run_twin(task: Task, cell: Cell, video: Path | None = None):
    picks = [s for s in task.steps if s.do == "pick"]
    places = [s for s in task.steps if s.do == "place"]
    if len(picks) != 1 or len(places) != 1:
        raise ValueError("the twin check needs exactly one pick and one place")
    pick, place = task.locations[picks[0].at], task.locations[places[0].at]
    if pick.xyz_mm[2] != 0 or place.xyz_mm[2] != 0:
        raise ValueError("the twin has one flat table: pick and place surfaces must be at z = 0 (use --flat-table)")
    if cell.work_frame_mm_deg != (0.0, 0.0, 0.0, 0.0):
        raise ValueError("use the so101 cell (work frame = robot base) for the twin")
    sys.path.insert(0, str(SIM_DIR))
    from so101_sim import scene  # noqa: E402
    from so101_sim.replay import replay, save_video  # noqa: E402

    obj = task.objects[picks[0].obj]
    traj, _ = trajectory(task, cell)
    model, data = scene.load(part_pos=np.array(pick.xyz_mm[:2]) / 1000, place_pos=np.array(place.xyz_mm[:2]) / 1000,
                             part_size=tuple(v / 1000 for v in obj.size_mm))
    res = replay(traj, model, data, camera="front" if video else None)
    if video:
        save_video(res.episode["images"], video)
    return traj, res
