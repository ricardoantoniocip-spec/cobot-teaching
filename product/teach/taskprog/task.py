"""The task file and the cell file: load, validate, and give clear error messages.

task.json  = WHAT to do (brand-neutral): objects, named locations, a list of skills.
cell.json  = WHERE and WITH WHICH ROBOT: work frame on the robot, tool, gripper, speeds.

The format is in TASK_FORMAT.md. The LLM never writes these numbers; it only names and
labels things (see label.py). Every check here is strict on purpose: unknown keys,
unknown skills and out-of-range values are errors, not warnings.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

TASK_FORMAT = "cobot-task/0.1"
CELL_FORMAT = "cobot-cell/0.1"
SKILLS = ("home", "move", "pick", "place", "wait")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
# Words that would clash with URScript or Python if used as a name.
RESERVED = {"def", "end", "if", "else", "elif", "while", "for", "return", "thread", "run", "kill",
            "global", "local", "halt", "sync", "true", "false", "not", "and", "or", "xor", "pass"}


class TaskError(ValueError):
    """The file is not a valid task or cell. The message lists every problem found."""


@dataclass
class Obj:
    name: str
    size_mm: tuple[float, float, float]  # width across the jaws, depth, height
    grasp_height_mm: float  # where the tool point closes, measured up from the object's bottom
    description: str = ""


@dataclass
class Location:
    name: str
    xyz_mm: tuple[float, float, float]  # x, y in the work frame; z = height of the surface the object stands on
    yaw_deg: float = 0.0  # object turn about the vertical; the tool always points straight down
    note: str = ""


@dataclass
class Step:
    id: int
    do: str
    obj: str | None = None  # pick / place
    at: str | None = None  # pick / place / move: a location name
    height_mm: float | None = None  # move: tool point height above the location's surface
    seconds: float | None = None  # wait
    speed: str | None = None
    approach_mm: float | None = None
    label: str = ""
    demo_t: tuple[float, float] | None = None


@dataclass
class Task:
    name: str
    description: str
    objects: dict[str, Obj]
    locations: dict[str, Location]
    steps: list[Step]
    approach_mm: float = 50.0
    travel_height_mm: float = 100.0
    grip_time_s: float = 0.5
    speed: str = "normal"
    source: dict = field(default_factory=dict)
    review: list[str] = field(default_factory=list)  # open questions for the engineer


@dataclass
class Speed:
    joint_deg_s: float
    joint_deg_s2: float
    linear_mm_s: float
    linear_mm_s2: float


@dataclass
class Gripper:
    type: str  # "digital_out" (UR tool or controller output) or "so101" (servo)
    output: str = "tool"  # digital_out: "tool" or "standard"
    pin: int = 0
    close_value: bool = True


@dataclass
class Cell:
    name: str
    robot: str  # "ur5e", "ur3e", "ur10e", "so101"
    work_frame_mm_deg: tuple[float, ...]  # x, y, z, yaw of the work frame in the robot base frame
    tcp_mm: tuple[float, float, float]  # tool point from the flange (UR) - ignored for so101 (URDF)
    payload_kg: float
    gripper: Gripper
    home_joints_deg: tuple[float, ...]
    speeds: dict[str, Speed]
    release_gap_mm: float = 2.0  # open the gripper this much higher than the grasp height
    min_tool_height_mm: float = 5.0  # the tool point never goes lower than this above the work floor
    work_area_mm: tuple[float, ...] | None = None  # xmin, xmax, ymin, ymax in the work frame


# Hard caps for any cell file, so a typo cannot ask for a fast robot.
MAX_JOINT_DEG_S = 120.0
MAX_LINEAR_MM_S = 500.0


class _Errors:
    def __init__(self, what: str):
        self.what = what
        self.items: list[str] = []

    def add(self, msg: str) -> None:
        self.items.append(msg)

    def raise_if_any(self) -> None:
        if self.items:
            raise TaskError(f"{self.what} has {len(self.items)} problem(s):\n  - " + "\n  - ".join(self.items))


def _num(v, path: str, err: _Errors, lo: float | None = None, hi: float | None = None) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        err.add(f"{path}: must be a number, got {v!r}")
        return None
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        err.add(f"{path}: {v} is outside {lo} .. {hi}")
        return None
    return float(v)


def _vec(v, n: int, path: str, err: _Errors, lo=None, hi=None) -> tuple[float, ...] | None:
    if not isinstance(v, list) or len(v) != n:
        err.add(f"{path}: must be a list of {n} numbers")
        return None
    out = [_num(x, f"{path}[{i}]", err, lo, hi) for i, x in enumerate(v)]
    return None if any(x is None for x in out) else tuple(out)


def _name(v, path: str, err: _Errors) -> str | None:
    if not isinstance(v, str) or not NAME_RE.match(v) or v in RESERVED:
        err.add(f"{path}: {v!r} is not a valid name (lower case letters, digits, _; start with a letter; max 40)")
        return None
    return v


def _keys(d, allowed: set[str], required: set[str], path: str, err: _Errors) -> bool:
    if not isinstance(d, dict):
        err.add(f"{path}: must be an object")
        return False
    for k in sorted(set(d) - allowed):
        err.add(f"{path}: unknown key '{k}'")
    for k in sorted(required - set(d)):
        err.add(f"{path}: missing '{k}'")
    return True


def clean_text(s, limit: int = 200) -> str:
    """Printable ASCII on one line. Labels end up as comments in robot code, so no newlines."""
    if not isinstance(s, str):
        return ""
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = "".join(c if 32 <= ord(c) < 127 else "?" for c in s)
    return re.sub(r"\s+", " ", s).strip()[:limit]


def parse_task(data: dict, cell_speeds: set[str] | None = None) -> Task:
    err = _Errors("task file")
    top = {"format", "name", "description", "units", "source", "objects", "locations", "defaults", "steps", "review"}
    if not _keys(data, top, {"format", "name", "objects", "locations", "steps"}, "task", err):
        err.raise_if_any()
    if data.get("format") != TASK_FORMAT:
        err.add(f"format: must be '{TASK_FORMAT}', got {data.get('format')!r}")
    if "units" in data and data["units"] != {"length": "mm", "angle": "deg", "time": "s"}:
        err.add('units: only {"length": "mm", "angle": "deg", "time": "s"} is supported')
    name = _name(data.get("name"), "name", err)

    objects: dict[str, Obj] = {}
    for oname, o in (data.get("objects") or {}).items():
        p = f"objects.{oname}"
        if _name(oname, p, err) and _keys(o, {"size_mm", "grasp_height_mm", "description"}, {"size_mm"}, p, err):
            size = _vec(o.get("size_mm"), 3, f"{p}.size_mm", err, 1, 1000)
            if size:
                gh = o.get("grasp_height_mm", min(size[2] / 2, 30.0))
                gh = _num(gh, f"{p}.grasp_height_mm", err, 0, size[2])
                if gh is not None:
                    objects[oname] = Obj(oname, size, gh, clean_text(o.get("description", "")))
    if not objects:
        err.add("objects: at least one object is needed")

    locations: dict[str, Location] = {}
    for lname, loc in (data.get("locations") or {}).items():
        p = f"locations.{lname}"
        if _name(lname, p, err) and _keys(loc, {"xyz_mm", "yaw_deg", "note"}, {"xyz_mm"}, p, err):
            xyz = _vec(loc.get("xyz_mm"), 3, f"{p}.xyz_mm", err, -5000, 5000)
            yaw = _num(loc.get("yaw_deg", 0.0), f"{p}.yaw_deg", err, -180, 180)
            if xyz and yaw is not None:
                locations[lname] = Location(lname, xyz, yaw, clean_text(loc.get("note", "")))

    d = data.get("defaults", {})
    _keys(d, {"approach_mm", "travel_height_mm", "grip_time_s", "speed"}, set(), "defaults", err)
    approach = _num(d.get("approach_mm", 50.0), "defaults.approach_mm", err, 10, 300) or 50.0
    travel = _num(d.get("travel_height_mm", 100.0), "defaults.travel_height_mm", err, 10, 1000) or 100.0
    grip_t = _num(d.get("grip_time_s", 0.5), "defaults.grip_time_s", err, 0, 5)
    grip_t = 0.5 if grip_t is None else grip_t
    speed = d.get("speed", "normal")
    if cell_speeds is not None and speed not in cell_speeds:
        err.add(f"defaults.speed: '{speed}' is not one of the cell's speeds {sorted(cell_speeds)}")

    steps: list[Step] = []
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        err.add("steps: must be a non-empty list")
        raw_steps = []
    allowed = {
        "home": set(),
        "move": {"to", "height_mm"},
        "pick": {"object", "at", "approach_mm"},
        "place": {"object", "at", "approach_mm"},
        "wait": {"seconds"},
    }
    needed = {"home": set(), "move": {"to"}, "pick": {"object", "at"}, "place": {"object", "at"}, "wait": {"seconds"}}
    for i, s in enumerate(raw_steps):
        p = f"steps[{i}]"
        if not isinstance(s, dict) or s.get("do") not in SKILLS:
            err.add(f"{p}: 'do' must be one of {list(SKILLS)}, got {s.get('do') if isinstance(s, dict) else s!r}")
            continue
        do = s["do"]
        common = {"do", "id", "label", "speed", "demo"}
        if not _keys(s, common | allowed[do], {"do"} | needed[do], f"{p} ({do})", err):
            continue
        step = Step(id=i + 1, do=do, label=clean_text(s.get("label", "")))
        if "id" in s and s["id"] != i + 1:
            err.add(f"{p}: id must be {i + 1} (steps are numbered 1, 2, 3 in order)")
        if "speed" in s:
            if cell_speeds is not None and s["speed"] not in cell_speeds:
                err.add(f"{p}: speed '{s['speed']}' is not one of the cell's speeds {sorted(cell_speeds)}")
            step.speed = s["speed"]
        if "demo" in s:
            dm = s["demo"]
            if _keys(dm, {"t_start", "t_end"}, {"t_start", "t_end"}, f"{p}.demo", err):
                a, b = _num(dm["t_start"], f"{p}.demo.t_start", err, 0), _num(dm["t_end"], f"{p}.demo.t_end", err, 0)
                if a is not None and b is not None:
                    step.demo_t = (a, b)
        if do in ("pick", "place"):
            step.obj, step.at = s["object"], s["at"]
            if step.obj not in objects:
                err.add(f"{p}: object '{step.obj}' is not in objects")
            if step.at not in locations:
                err.add(f"{p}: location '{step.at}' is not in locations")
            if "approach_mm" in s:
                step.approach_mm = _num(s["approach_mm"], f"{p}.approach_mm", err, 10, 300)
        elif do == "move":
            step.at = s["to"]
            if step.at not in locations:
                err.add(f"{p}: location '{step.at}' is not in locations")
            if "height_mm" in s:
                step.height_mm = _num(s["height_mm"], f"{p}.height_mm", err, 10, 1000)
        elif do == "wait":
            step.seconds = _num(s["seconds"], f"{p}.seconds", err, 0, 60)
        steps.append(step)

    # Sequence logic: you can only place what you hold, and only hold one thing.
    holding: str | None = None
    for s in steps:
        if s.do == "pick":
            if holding:
                err.add(f"step {s.id}: pick {s.obj} while still holding {holding}")
            holding = s.obj
        elif s.do == "place":
            if holding != s.obj:
                err.add(f"step {s.id}: place {s.obj} but the gripper holds {holding or 'nothing'}")
            holding = None
    if holding:
        err.add(f"the task ends while still holding {holding}")

    review = [clean_text(r, 300) for r in data.get("review", []) if isinstance(r, str)]
    err.raise_if_any()
    return Task(name, clean_text(data.get("description", ""), 300), objects, locations, steps,
                approach, travel, grip_t, speed, data.get("source", {}), review)


def parse_cell(data: dict) -> Cell:
    err = _Errors("cell file")
    top = {"format", "name", "robot", "work_frame", "tcp_mm", "payload_kg", "gripper", "home_joints_deg",
           "speeds", "release_gap_mm", "min_tool_height_mm", "work_area_mm", "note"}
    if not _keys(data, top, {"format", "name", "robot", "work_frame", "gripper", "home_joints_deg", "speeds"}, "cell", err):
        err.raise_if_any()
    if data.get("format") != CELL_FORMAT:
        err.add(f"format: must be '{CELL_FORMAT}'")
    robot = data.get("robot")
    if robot not in ("ur3e", "ur5e", "ur10e", "so101"):
        err.add(f"robot: {robot!r} is not supported yet (ur3e, ur5e, ur10e, so101)")
    wf = data.get("work_frame", {})
    wf_ok = _keys(wf, {"xyz_mm", "yaw_deg"}, {"xyz_mm"}, "work_frame", err)
    xyz = _vec(wf.get("xyz_mm"), 3, "work_frame.xyz_mm", err, -3000, 3000) if wf_ok else None
    yaw = _num(wf.get("yaw_deg", 0.0), "work_frame.yaw_deg", err, -180, 180) if wf_ok else None
    tcp = _vec(data.get("tcp_mm", [0, 0, 0]), 3, "tcp_mm", err, -500, 500)
    payload = _num(data.get("payload_kg", 0.0), "payload_kg", err, 0, 20)
    g = data.get("gripper", {})
    gripper = None
    if _keys(g, {"type", "output", "pin", "close_value"}, {"type"}, "gripper", err):
        if g["type"] not in ("digital_out", "so101"):
            err.add("gripper.type: must be 'digital_out' or 'so101'")
        elif g.get("output", "tool") not in ("tool", "standard"):
            err.add("gripper.output: must be 'tool' or 'standard'")
        else:
            pin = g.get("pin", 0)
            if not isinstance(pin, int) or isinstance(pin, bool) or not 0 <= pin <= 7:
                err.add("gripper.pin: must be an integer 0..7")
            else:
                gripper = Gripper(g["type"], g.get("output", "tool"), pin, bool(g.get("close_value", True)))
    n_joints = 5 if robot == "so101" else 6
    home = _vec(data.get("home_joints_deg"), n_joints, "home_joints_deg", err, -360, 360)
    speeds: dict[str, Speed] = {}
    for sname, sp in (data.get("speeds") or {}).items():
        p = f"speeds.{sname}"
        keys = {"joint_deg_s", "joint_deg_s2", "linear_mm_s", "linear_mm_s2"}
        if _keys(sp, keys, keys, p, err):
            vals = [
                _num(sp["joint_deg_s"], f"{p}.joint_deg_s", err, 1, MAX_JOINT_DEG_S),
                _num(sp["joint_deg_s2"], f"{p}.joint_deg_s2", err, 1, 1000),
                _num(sp["linear_mm_s"], f"{p}.linear_mm_s", err, 1, MAX_LINEAR_MM_S),
                _num(sp["linear_mm_s2"], f"{p}.linear_mm_s2", err, 1, 5000),
            ]
            if all(v is not None for v in vals):
                speeds[sname] = Speed(*vals)
    if "normal" not in speeds:
        err.add("speeds: a 'normal' speed is needed")
    gap = _num(data.get("release_gap_mm", 2.0), "release_gap_mm", err, 0, 50)
    min_h = _num(data.get("min_tool_height_mm", 5.0), "min_tool_height_mm", err, 0, 500)
    area = None
    if "work_area_mm" in data:
        area = _vec(data["work_area_mm"], 4, "work_area_mm", err, -5000, 5000)
    err.raise_if_any()
    return Cell(data["name"], robot, (*xyz, yaw), tcp, payload, gripper, home, speeds, gap, min_h, area)


def load_cell(path: Path) -> Cell:
    return parse_cell(json.loads(Path(path).read_text()))


def load_task(path: Path, cell: Cell | None = None) -> Task:
    return parse_task(json.loads(Path(path).read_text()), set(cell.speeds) if cell else None)
