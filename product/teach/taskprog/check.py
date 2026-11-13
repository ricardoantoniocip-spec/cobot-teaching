"""Offline checks before any simulator: geometry, reach, singularities, open review points.

Each finding is ERROR (do not run), WARN (an engineer must look) or INFO.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .plan import make_plan, work_to_base
from .task import Cell, Task


@dataclass
class Finding:
    level: str  # ERROR, WARN, INFO
    step: int
    text: str

    def __str__(self) -> str:
        where = f"step {self.step}" if self.step else "task"
        return f"{self.level:5} {where}: {self.text}"


def check(task: Task, cell: Cell) -> list[Finding]:
    out: list[Finding] = []
    plan = make_plan(task, cell)
    inv = np.linalg.inv(work_to_base(cell))

    for m in plan:
        if m.pose is None:
            continue
        p_work = (inv @ m.pose)[:3, 3] * 1000
        if p_work[2] < cell.min_tool_height_mm:
            out.append(Finding("ERROR", m.step, f"tool point {p_work[2]:.0f} mm above the floor ({m.note}); minimum is {cell.min_tool_height_mm:g} mm"))
        if cell.work_area_mm:
            x0, x1, y0, y1 = cell.work_area_mm
            if not (x0 <= p_work[0] <= x1 and y0 <= p_work[1] <= y1):
                out.append(Finding("ERROR", m.step, f"{m.location} ({p_work[0]:.0f}, {p_work[1]:.0f}) mm is outside the cell's work area"))

    if cell.robot == "so101":
        from .so101 import REACH_TOL_M, trajectory

        _, rows = trajectory(task, cell)
        for m, err in rows:
            if err > REACH_TOL_M:
                out.append(Finding("ERROR", m.step, f"SO-101 cannot reach ({m.note}): {err * 1000:.0f} mm short"))
        for loc in task.locations.values():
            if loc.yaw_deg:
                out.append(Finding("INFO", 0, f"{loc.name}: yaw {loc.yaw_deg:g} deg ignored, the SO-101 always grips along the line from its base"))
    else:
        from .ur import reach_check

        for row in reach_check(plan, cell):
            if row.problem:
                level = "ERROR" if ("reach" in row.problem or "jump" in row.problem) else "WARN"
                out.append(Finding(level, row.motion.step, f"{row.problem} ({row.motion.note})"))

    used = {s.signal: s for s in reversed(task.steps) if s.signal}
    if cell.robot == "so101":
        if used:
            out.append(Finding("WARN", 0, "the SO-101 has no I/O: signal steps become short pauses"))
    else:
        for name, s in sorted(used.items()):
            sg = cell.signals.get(name)
            if sg is None:
                out.append(Finding("ERROR", s.id, f"signal '{name}' is not in the cell's signals (which pin is it?)"))
            elif task.signals[name] == "output" and not sg.io.endswith("_out"):
                out.append(Finding("ERROR", s.id, f"signal '{name}' is an output in the task but the cell maps it to {sg.io}"))
            elif task.signals[name] == "input" and sg.io.endswith("_out"):
                out.append(Finding("WARN", s.id, f"input '{name}' reads back an output ({sg.io} {sg.pin}): fine in a simulator, not on a real cell"))
        if cell.simulate:
            out.append(Finding("WARN", 0, "this cell simulates a machine: the program is for URSim only"))
        if cell.gripper.type == "digital_out":
            for name in used:
                sg = cell.signals.get(name)
                if sg and sg.io == f"{cell.gripper.output}_out" and sg.pin == cell.gripper.pin:
                    out.append(Finding("ERROR", used[name].id, f"signal '{name}' uses the gripper's output pin"))
    waits = [s for s in task.steps if s.do == "wait_input"]
    if waits:
        out.append(Finding("INFO", 0, "on a timeout the program stops where it is and leaves outputs as they are"))

    for r in task.review:
        out.append(Finding("WARN", 0, f"review: {r}"))
    if not any(s.do == "home" for s in task.steps[-1:]):
        out.append(Finding("INFO", 0, "the task does not end at home"))
    return out


def passed(findings: list[Finding]) -> bool:
    return not any(f.level == "ERROR" for f in findings)
