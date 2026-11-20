"""Play the SO-101 trajectory of a task in the MuJoCo twin (product/sim) and score it.

The twin is a flat table with one block. The task may pick and place that block several
times (a part moved twice in the demo), as long as every later pick is where the block was last
put down. The score is where the block ends: at the last place or not.

look=True: before every pick after the first, the robot "looks": the pick position is taken from
where the block really stands in the twin, not from where the program put it down. In the twin the
look is perfect (the simulator knows where the block is). It proves the logic, not a camera.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .so101 import trajectory
from .task import Cell, Location, Task

SIM_DIR = Path(__file__).resolve().parents[2] / "sim"


def run_twin(task: Task, cell: Cell, video: Path | None = None, look: bool = False, notes: list[str] | None = None):
    moves = [s for s in task.steps if s.do in ("pick", "place")]
    picks = [s for s in moves if s.do == "pick"]
    places = [s for s in moves if s.do == "place"]
    if not picks or len(picks) != len(places):
        raise ValueError("the twin check needs at least one pick and place")
    if len({s.obj for s in moves}) != 1 or any(s.slot is not None for s in moves):
        raise ValueError("the twin has one block and no trays: one object, no grid slots")
    if any(task.locations[s.at].xyz_mm[2] != 0 for s in moves):
        raise ValueError("the twin has one flat table: pick and place surfaces must be at z = 0 (use --flat-table)")
    for prev, nxt in zip(places[:-1], picks[1:]):
        a, b = task.locations[prev.at].xyz_mm, task.locations[nxt.at].xyz_mm
        if np.hypot(a[0] - b[0], a[1] - b[1]) > 5.0:
            raise ValueError(f"step {nxt.id} picks at {nxt.at}, but the block was last put down at {prev.at}")
    pick, place = task.locations[picks[0].at], task.locations[places[-1].at]
    if cell.work_frame_mm_deg != (0.0, 0.0, 0.0, 0.0):
        raise ValueError("use the so101 cell (work frame = robot base) for the twin")
    sys.path.insert(0, str(SIM_DIR))
    from so101_sim import scene  # noqa: E402
    from so101_sim.replay import replay, save_video  # noqa: E402

    obj = task.objects[picks[0].obj]

    def load():
        return scene.load(part_pos=np.array(pick.xyz_mm[:2]) / 1000, place_pos=np.array(place.xyz_mm[:2]) / 1000,
                          part_size=tuple(v / 1000 for v in obj.size_mm))  # fmt: skip

    if look:
        import copy

        task = copy.deepcopy(task)
        for k in range(1, len(picks)):
            # Play the program up to the place before this pick, then look where the block stands.
            prev_place = next(s for s in task.steps if s.id == places[k - 1].id)
            upto = copy.deepcopy(task)
            upto.steps = task.steps[: task.steps.index(prev_place) + 1]
            before = replay(trajectory(upto, cell)[0], *load())
            seen = before.part_end
            planned = task.locations[picks[k].at].xyz_mm
            off = float(np.hypot(seen[0] * 1000 - planned[0], seen[1] * 1000 - planned[1]))
            name = f"seen_before_pick_{k + 1}"
            task.locations[name] = Location(name, (round(float(seen[0]) * 1000, 1), round(float(seen[1]) * 1000, 1), 0.0), note="where the block stood when the robot looked")
            next(s for s in task.steps if s.id == picks[k].id).at = name
            if notes is not None:
                notes.append(f"pick {k + 1}: looked first. The block stood {off:.0f} mm from where the program put it down; the pick was moved there."
                             + ("" if before.upright else " The block has fallen over."))  # fmt: skip

    traj, _ = trajectory(task, cell)
    model, data = load()
    res = replay(traj, model, data, camera="front" if video else None)
    if video:
        save_video(res.episode["images"], video)
    return traj, res
