"""Teach pendant paths: waypoints (MoveJ) and gripper steps, saved as a small JSON file.

Like a cobot teach pendant: jog the arm to a pose, record it as a waypoint, press o / c to open or
close the gripper, save the path under a name, run it again later.

    {"format": "so101-path/0.1", "name": "box_to_tray", "steps": [
        {"do": "movej", "joints_deg": [0, -20, 30, 80, 0], "gripper": 1.0, "tool_mm": [180, 0, 60]},
        {"do": "gripper", "gripper": 0.0},
        ...]}

joints_deg are the 5 SO-101 arm joints in URDF order (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll), the same angles the twin sliders show. gripper is 0 closed .. 1 open (the retarget
convention). tool_mm is the tool point in the robot base frame, for reading only.

A path turns into:
  - a joint trajectory CSV (to_trajectory + write_trajectory): `python -m retarget replay` plays it on
    the real arm and `python -m so101_sim replay` in the twin;
  - a task file (to_task) for the URScript / SO-101 generator: every close becomes a pick and every
    open a place at the waypoint just before it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .demo import JointTrajectory

FORMAT = "so101-path/0.1"
OPEN, CLOSED = 1.0, 0.0


@dataclass
class Step:
    do: str  # "movej" or "gripper"
    gripper: float  # 0 closed .. 1 open: the gripper at the end of this step
    joints_deg: list[float] | None = None  # movej only
    tool_mm: list[float] | None = None  # movej only, for reading

    def to_dict(self) -> dict:
        d = {"do": self.do}
        if self.do == "movej":
            d["joints_deg"] = [round(float(v), 3) for v in self.joints_deg]
        d["gripper"] = round(float(self.gripper), 3)
        if self.tool_mm is not None:
            d["tool_mm"] = [round(float(v), 1) for v in self.tool_mm]
        return d

    def text(self) -> str:
        if self.do == "gripper":
            return "gripper OPEN" if self.gripper >= 0.5 else "gripper CLOSE"
        where = f"  tool {self.tool_mm[0]:.0f}, {self.tool_mm[1]:.0f}, {self.tool_mm[2]:.0f} mm" if self.tool_mm else ""
        return "MoveJ " + " ".join(f"{v:.0f}" for v in self.joints_deg) + where


@dataclass
class TeachPath:
    name: str = "path"
    steps: list[Step] = field(default_factory=list)
    note: str = ""

    # --- teaching -------------------------------------------------------------------------------------
    def add_waypoint(self, q_rad: np.ndarray, gripper_open: float, chain=None) -> Step:
        tool = None
        if chain is not None:
            tool = list(chain.fk(np.asarray(q_rad, dtype=float))[:3, 3] * 1000)
        s = Step("movej", float(np.clip(gripper_open, 0, 1)), [float(v) for v in np.degrees(q_rad)], tool)
        self.steps.append(s)
        return s

    def add_gripper(self, open_: bool) -> Step:
        s = Step("gripper", OPEN if open_ else CLOSED)
        self.steps.append(s)
        return s

    def undo(self) -> Step | None:
        return self.steps.pop() if self.steps else None

    @property
    def n_waypoints(self) -> int:
        return sum(s.do == "movej" for s in self.steps)

    def lines(self) -> list[str]:
        return [f"{i + 1:2d}  {s.text()}" for i, s in enumerate(self.steps)]

    # --- files ----------------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = {"format": FORMAT, "name": self.name, "units": {"angle": "deg", "length": "mm"}}
        if self.note:
            d["note"] = self.note
        d["steps"] = [s.to_dict() for s in self.steps]
        return d

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "TeachPath":
        if d.get("format") != FORMAT:
            raise ValueError(f"not a teach path (format {d.get('format')!r}, expected {FORMAT!r})")
        steps = []
        for i, s in enumerate(d["steps"], 1):
            if s["do"] == "movej":
                if len(s["joints_deg"]) != 5:
                    raise ValueError(f"step {i}: joints_deg needs 5 angles")
                steps.append(Step("movej", float(s.get("gripper", OPEN)), [float(v) for v in s["joints_deg"]], s.get("tool_mm")))
            elif s["do"] == "gripper":
                steps.append(Step("gripper", float(s["gripper"])))
            else:
                raise ValueError(f"step {i}: unknown step {s['do']!r} (movej or gripper)")
        return cls(d.get("name", "path"), steps, d.get("note", ""))

    @classmethod
    def load(cls, path: Path) -> "TeachPath":
        return cls.from_dict(json.loads(Path(path).read_text()))


def safe_name(name: str) -> str:
    """A file-safe snake_case name (also valid as a task / program name)."""
    s = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_")
    if not s or not s[0].isalpha():
        s = "path_" + s
    return s[:40].rstrip("_")


def _min_jerk(s: np.ndarray) -> np.ndarray:
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def to_trajectory(path: TeachPath, q_start: np.ndarray | None = None, gripper_start: float | None = None,
                  max_speed_deg_s: float = 20.0, gripper_time_s: float = 1.0, dwell_s: float = 0.2,
                  hz: float = 30.0, min_move_s: float = 0.5) -> JointTrajectory:
    """Play the path as MoveJ moves: all joints start and stop together, smooth (minimum jerk) speed profile,
    peak joint speed = max_speed_deg_s. Gripper steps open/close over gripper_time_s with the arm still.

    q_start: where the arm is now (radians). Given, the trajectory starts with a move from there to the
    first waypoint. Not given, it starts at the first waypoint (retarget replay then goes there slowly itself).
    """
    moves = [s for s in path.steps if s.do == "movej"]
    if not moves:
        raise ValueError("the path has no waypoints: record at least one")
    q = np.radians(moves[0].joints_deg) if q_start is None else np.asarray(q_start, dtype=float)
    g = (moves[0].gripper if path.steps[0].do == "movej" else OPEN) if gripper_start is None else float(gripper_start)
    vmax = np.radians(max_speed_deg_s)
    dt = 1.0 / hz
    ts, qs, gs = [0.0], [q.copy()], [g]

    def segment(q_to: np.ndarray, g_to: float, duration: float) -> None:
        nonlocal q, g
        n = max(1, int(np.ceil(duration * hz)))
        s = _min_jerk(np.arange(1, n + 1) / n)
        for k in range(n):
            ts.append(ts[-1] + dt)
            qs.append(q + (q_to - q) * s[k])
            gs.append(g + (g_to - g) * s[k])
        q, g = q_to.copy(), g_to

    def hold(duration: float) -> None:
        for _ in range(int(round(duration * hz))):
            ts.append(ts[-1] + dt)
            qs.append(q.copy())
            gs.append(g)

    for st in path.steps:
        if st.do == "movej":
            q_to = np.radians(st.joints_deg)
            dq = float(np.abs(q_to - q).max())
            # A minimum-jerk move peaks at 1.875x its average speed.
            duration = max(1.875 * dq / vmax, min_move_s) if (dq > 1e-6 or abs(st.gripper - g) > 1e-3) else 0.0
            if duration > 0:
                segment(q_to, st.gripper, duration)
                hold(dwell_s)
        else:
            if abs(st.gripper - g) > 1e-3:
                segment(q, st.gripper, gripper_time_s)
                hold(dwell_s)
    n = len(ts)
    return JointTrajectory(np.array(ts), np.array(qs), np.array(gs), np.zeros(n))


def to_task(path: TeachPath, chain, object_size_mm=(20.0, 20.0, 40.0), grasp_height_mm: float = 20.0) -> dict | None:
    """A cobot-task/0.2 task file: each gripper close = pick, each open (while holding) = place, at the tool point
    of the waypoint before it. Waypoints between picks and places are not in the task file (the generator plans its
    own approach and travel moves). None if the path never closes and opens the gripper."""
    locations: dict = {}
    steps: list[dict] = [{"do": "home", "label": "start at home"}]
    last_q = None
    holding = False
    g = OPEN
    n_pick = n_place = 0
    for i, st in enumerate(path.steps, 1):
        if st.do == "movej":
            last_q = np.radians(st.joints_deg)
            g = st.gripper
            continue
        if last_q is None:
            g = st.gripper
            continue
        tool = chain.fk(last_q)[:3, 3] * 1000
        surface = max(0.0, float(tool[2]) - grasp_height_mm)
        xyz = [round(float(tool[0]), 1), round(float(tool[1]), 1), round(surface, 1)]
        if st.gripper < 0.5 <= g and not holding:
            n_pick += 1
            name = f"pick_{n_pick}"
            locations[name] = {"xyz_mm": xyz, "note": f"taught: gripper closed at path step {i}"}
            steps.append({"do": "pick", "object": "part", "at": name, "label": f"pick at taught point {n_pick}"})
            holding = True
        elif st.gripper >= 0.5 > g and holding:
            n_place += 1
            name = f"place_{n_place}"
            locations[name] = {"xyz_mm": xyz, "note": f"taught: gripper opened at path step {i}"}
            steps.append({"do": "place", "object": "part", "at": name, "label": f"place at taught point {n_place}"})
            holding = False
        g = st.gripper
    if n_pick == 0 or holding:
        return None
    steps.append({"do": "home", "label": "back home"})
    for k, s in enumerate(steps, 1):
        s["id"] = k
    return {
        "format": "cobot-task/0.2",
        "name": safe_name(path.name),
        "description": f"Taught on the SO-101 with the teach panel ({path.n_waypoints} waypoints).",
        "units": {"length": "mm", "angle": "deg", "time": "s"},
        "source": {"teach_path": path.name},
        "objects": {"part": {"size_mm": [float(v) for v in object_size_mm], "grasp_height_mm": float(grasp_height_mm),
                             "description": "the part held between close and open (size not measured, check it)"}},
        "locations": locations,
        "steps": steps,
        "review": ["Object size and grasp height are defaults: measure the real part.",
                   "Waypoints between picks and places are not kept here; the CSV from the same path plays them exactly."],
    }
