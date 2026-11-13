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

TASK_FORMAT = "cobot-task/0.2"
CELL_FORMAT = "cobot-cell/0.2"
TASK_FORMATS = ("cobot-task/0.1", TASK_FORMAT)  # 0.2 only adds things, so 0.1 files stay valid
CELL_FORMATS = ("cobot-cell/0.1", CELL_FORMAT)
SKILLS = ("home", "move", "pick", "place", "wait", "set_output", "wait_input")
MAX_STEPS = 500  # after repeats are written out
# Cell signal kinds. An input signal may also read back an output (used to simulate a machine).
SIGNAL_IO = {"standard_in": 7, "configurable_in": 7, "tool_in": 1,
             "standard_out": 7, "configurable_out": 7, "tool_out": 1}  # kind -> highest pin
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
class Grid:
    """A tray or pallet: one taught location (slot 1) and a regular pattern of slots."""
    cols: int  # slots along the location's x
    rows: int  # slots along the location's y
    pitch_mm: tuple[float, float]  # centre to centre, along x and y (may be negative)
    layers: int = 1
    layer_height_mm: float = 0.0

    @property
    def count(self) -> int:
        return self.cols * self.rows * self.layers

    def index(self, slot: int) -> tuple[int, int, int]:
        """Slot number (1, 2, 3 ...) -> column, row, layer. Fills along x, then y, then up."""
        i = slot - 1
        per_layer = self.cols * self.rows
        return i % self.cols, (i % per_layer) // self.cols, i // per_layer


@dataclass
class Location:
    name: str
    xyz_mm: tuple[float, float, float]  # x, y in the work frame; z = height of the surface the object stands on
    yaw_deg: float = 0.0  # object turn about the vertical; the tool always points straight down
    note: str = ""
    grid: Grid | None = None  # if set, xyz_mm is slot 1

    def slot_offset_mm(self, slot: int) -> tuple[float, float, float]:
        """Offset of a slot from slot 1, in the location's own x, y (turned by yaw) and up."""
        c, r, layer = self.grid.index(slot)
        return c * self.grid.pitch_mm[0], r * self.grid.pitch_mm[1], layer * self.grid.layer_height_mm

    def slot(self, slot: int | None) -> "Location":
        """The single place a step works at: the location itself, or one slot of its grid."""
        if slot is None or self.grid is None:
            return self
        dx, dy, dz = self.slot_offset_mm(slot)
        a = math.radians(self.yaw_deg)
        x = self.xyz_mm[0] + dx * math.cos(a) - dy * math.sin(a)
        y = self.xyz_mm[1] + dx * math.sin(a) + dy * math.cos(a)
        return Location(target_name(self.name, slot), (x, y, self.xyz_mm[2] + dz), self.yaw_deg)


def target_name(location: str, slot: int | None) -> str:
    return location if slot is None else f"{location}_s{slot}"


@dataclass
class Step:
    id: int
    do: str
    obj: str | None = None  # pick / place
    at: str | None = None  # pick / place / move: a location name
    slot: int | None = None  # pick / place / move at a grid location: slot number from 1
    height_mm: float | None = None  # move: tool point height above the location's surface
    seconds: float | None = None  # wait
    signal: str | None = None  # set_output / wait_input: a signal name from the task's signals
    value: bool = True  # set_output: the value to set; wait_input: the value to wait for
    pulse_s: float | None = None  # set_output: go back to the opposite value after this time
    timeout_s: float | None = None  # wait_input: stop the program if it takes longer
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
    signals: dict[str, str] = field(default_factory=dict)  # name -> "input" or "output"

    def target(self, s: Step) -> Location:
        return self.locations[s.at].slot(s.slot)


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
class Signal:
    io: str  # one of SIGNAL_IO
    pin: int


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
    signals: dict[str, Signal] = field(default_factory=dict)  # task signal name -> robot I/O pin
    # Simulator only: (start signal, done signal, seconds). A pretend machine in the program.
    simulate: list[tuple[str, str, float]] = field(default_factory=list)


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
    top = {"format", "name", "description", "units", "source", "objects", "locations", "defaults", "steps", "review",
           "signals"}
    if not _keys(data, top, {"format", "name", "objects", "locations", "steps"}, "task", err):
        err.raise_if_any()
    if data.get("format") not in TASK_FORMATS:
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
        if _name(lname, p, err) and _keys(loc, {"xyz_mm", "yaw_deg", "note", "grid"}, {"xyz_mm"}, p, err):
            xyz = _vec(loc.get("xyz_mm"), 3, f"{p}.xyz_mm", err, -5000, 5000)
            yaw = _num(loc.get("yaw_deg", 0.0), f"{p}.yaw_deg", err, -180, 180)
            grid = _grid(loc["grid"], f"{p}.grid", err) if "grid" in loc else None
            if xyz and yaw is not None and not ("grid" in loc and grid is None):
                locations[lname] = Location(lname, xyz, yaw, clean_text(loc.get("note", "")), grid)
    for lname, loc in locations.items():
        if loc.grid:
            clash = [n for n in locations if re.fullmatch(re.escape(lname) + r"_s\d+", n)]
            if clash:
                err.add(f"locations.{clash[0]}: the name clashes with the slots of grid '{lname}'")

    signals: dict[str, str] = {}
    if not isinstance(data.get("signals", {}), dict):
        err.add("signals: must be an object like {\"machine_done\": \"input\"}")
    else:
        for sname, direction in data.get("signals", {}).items():
            if _name(sname, f"signals.{sname}", err):
                if direction not in ("input", "output"):
                    err.add(f"signals.{sname}: must be 'input' or 'output'")
                else:
                    signals[sname] = direction

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
        "move": {"to", "slot", "height_mm"},
        "pick": {"object", "at", "slot", "approach_mm"},
        "place": {"object", "at", "slot", "approach_mm"},
        "wait": {"seconds"},
        "set_output": {"signal", "value", "pulse_s"},
        "wait_input": {"signal", "value", "timeout_s"},
    }
    needed = {"home": set(), "move": {"to"}, "pick": {"object", "at"}, "place": {"object", "at"}, "wait": {"seconds"},
              "set_output": {"signal"}, "wait_input": {"signal", "timeout_s"}}
    flat = _flatten(raw_steps, err)
    if len(flat) > MAX_STEPS:
        err.add(f"steps: {len(flat)} steps after repeats; the limit is {MAX_STEPS}")
        flat = []
    slot_now: dict[str, int] = {}  # per grid location: the last slot a step used
    for i, (p, s, in_repeat) in enumerate(flat):
        if not isinstance(s, dict) or s.get("do") not in SKILLS:
            err.add(f"{p}: 'do' must be one of {list(SKILLS)}, got {s.get('do') if isinstance(s, dict) else s!r}")
            continue
        do = s["do"]
        common = {"do", "id", "label", "speed", "demo"}
        if not _keys(s, common | allowed[do], {"do"} | needed[do], f"{p} ({do})", err):
            continue
        step = Step(id=i + 1, do=do, label=clean_text(s.get("label", "")))
        if "id" in s and (in_repeat or s["id"] != i + 1):
            err.add(f"{p}: id must be {i + 1} (steps are numbered 1, 2, 3 in order; leave ids out inside a repeat)")
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
            else:
                step.slot = _slot(s, locations[step.at], slot_now, p, err)
            if "approach_mm" in s:
                step.approach_mm = _num(s["approach_mm"], f"{p}.approach_mm", err, 10, 300)
        elif do == "move":
            step.at = s["to"]
            if step.at not in locations:
                err.add(f"{p}: location '{step.at}' is not in locations")
            else:
                step.slot = _slot(s, locations[step.at], slot_now, p, err)
            if "height_mm" in s:
                step.height_mm = _num(s["height_mm"], f"{p}.height_mm", err, 10, 1000)
        elif do == "wait":
            step.seconds = _num(s["seconds"], f"{p}.seconds", err, 0, 60)
        elif do in ("set_output", "wait_input"):
            step.signal = s["signal"]
            want = "output" if do == "set_output" else "input"
            if signals.get(step.signal) != want:
                err.add(f"{p}: '{step.signal}' is not an {want} in the task's signals")
            if not isinstance(s.get("value", True), bool):
                err.add(f"{p}.value: must be true or false")
            step.value = s.get("value", True) is True
            if "pulse_s" in s:
                step.pulse_s = _num(s["pulse_s"], f"{p}.pulse_s", err, 0.05, 10)
            if "timeout_s" in s:
                step.timeout_s = _num(s["timeout_s"], f"{p}.timeout_s", err, 0.1, 3600)
        steps.append(step)

    # Sequence logic: you can only place what you hold, and only hold one thing.
    # A grid slot holds one thing: no second place into a filled slot, no second pick from an emptied one.
    holding: str | None = None
    filled: dict[tuple[str, int], bool] = {}
    for s in steps:
        if s.do == "pick":
            if holding:
                err.add(f"step {s.id}: pick {s.obj} while still holding {holding}")
            holding = s.obj
            if s.slot is not None:
                if filled.get((s.at, s.slot)) is False:
                    err.add(f"step {s.id}: pick from {s.at} slot {s.slot}, which an earlier step already emptied")
                filled[(s.at, s.slot)] = False
        elif s.do == "place":
            if holding != s.obj:
                err.add(f"step {s.id}: place {s.obj} but the gripper holds {holding or 'nothing'}")
            holding = None
            if s.slot is not None:
                if filled.get((s.at, s.slot)) is True:
                    err.add(f"step {s.id}: place at {s.at} slot {s.slot}, which an earlier step already filled")
                filled[(s.at, s.slot)] = True
    if holding:
        err.add(f"the task ends while still holding {holding}")

    review = [clean_text(r, 300) for r in data.get("review", []) if isinstance(r, str)]
    err.raise_if_any()
    return Task(name, clean_text(data.get("description", ""), 300), objects, locations, steps,
                approach, travel, grip_t, speed, data.get("source", {}), review, signals)


def _grid(g, path: str, err: _Errors) -> Grid | None:
    if not _keys(g, {"count", "pitch_mm", "layers", "layer_height_mm"}, {"count", "pitch_mm"}, path, err):
        return None
    count = g.get("count")
    if not (isinstance(count, list) and len(count) == 2 and all(isinstance(c, int) and not isinstance(c, bool) and 1 <= c <= 50 for c in count)):
        err.add(f"{path}.count: must be [columns, rows], whole numbers 1..50")
        return None
    pitch = _vec(g.get("pitch_mm"), 2, f"{path}.pitch_mm", err, -1000, 1000)
    layers = g.get("layers", 1)
    if not isinstance(layers, int) or isinstance(layers, bool) or not 1 <= layers <= 20:
        err.add(f"{path}.layers: must be a whole number 1..20")
        return None
    lh = _num(g.get("layer_height_mm", 0.0), f"{path}.layer_height_mm", err, 0, 1000)
    if pitch is None or lh is None:
        return None
    if (count[0] > 1 and pitch[0] == 0) or (count[1] > 1 and pitch[1] == 0):
        err.add(f"{path}.pitch_mm: the pitch cannot be 0 when there is more than one slot in that direction")
        return None
    if layers > 1 and lh <= 0:
        err.add(f"{path}.layer_height_mm: needed when layers is more than 1")
        return None
    return Grid(count[0], count[1], pitch, layers, lh)


def _flatten(raw_steps: list, err: _Errors) -> list[tuple[str, dict, bool]]:
    """Write repeats out: [(path for messages, step, inside a repeat)]. Repeats do not nest."""
    flat: list[tuple[str, dict, bool]] = []
    for i, s in enumerate(raw_steps):
        p = f"steps[{i}]"
        if isinstance(s, dict) and s.get("do") == "repeat":
            if not _keys(s, {"do", "times", "steps", "label"}, {"do", "times", "steps"}, f"{p} (repeat)", err):
                continue
            times, inner = s["times"], s["steps"]
            if not isinstance(times, int) or isinstance(times, bool) or not 1 <= times <= 200:
                err.add(f"{p}.times: must be a whole number 1..200")
                continue
            if not isinstance(inner, list) or not inner:
                err.add(f"{p}.steps: must be a non-empty list")
                continue
            if any(isinstance(x, dict) and x.get("do") == "repeat" for x in inner):
                err.add(f"{p}: a repeat inside a repeat is not supported")
                continue
            for k in range(times):
                flat += [(f"{p} round {k + 1} step {j + 1}", x, True) for j, x in enumerate(inner)]
        else:
            flat.append((p, s, False))
    return flat


def _slot(s: dict, loc: Location, slot_now: dict[str, int], p: str, err: _Errors) -> int | None:
    """Resolve 'slot': a number, "next" (count up from 1) or "same" (the slot used last)."""
    v = s.get("slot")
    if loc.grid is None:
        if v is not None:
            err.add(f"{p}: '{loc.name}' has no grid, so it has no slots")
        return None
    if v is None:
        err.add(f"{p}: '{loc.name}' is a grid; say which slot (a number, \"next\" or \"same\")")
        return None
    if v == "next":
        v = slot_now.get(loc.name, 0) + 1
    elif v == "same":
        if loc.name not in slot_now:
            err.add(f"{p}: slot \"same\" but no earlier step used '{loc.name}'")
            return None
        v = slot_now[loc.name]
    if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= loc.grid.count:
        err.add(f"{p}: slot {v!r} is outside 1..{loc.grid.count} of '{loc.name}'")
        return None
    slot_now[loc.name] = v
    return v


def parse_cell(data: dict) -> Cell:
    err = _Errors("cell file")
    top = {"format", "name", "robot", "work_frame", "tcp_mm", "payload_kg", "gripper", "home_joints_deg",
           "speeds", "release_gap_mm", "min_tool_height_mm", "work_area_mm", "note", "signals", "simulate"}
    if not _keys(data, top, {"format", "name", "robot", "work_frame", "gripper", "home_joints_deg", "speeds"}, "cell", err):
        err.raise_if_any()
    if data.get("format") not in CELL_FORMATS:
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
    signals: dict[str, Signal] = {}
    if not isinstance(data.get("signals", {}), dict):
        err.add("signals: must be an object")
    else:
        for sname, sg in data.get("signals", {}).items():
            p = f"signals.{sname}"
            if _name(sname, p, err) and _keys(sg, {"io", "pin", "note"}, {"io", "pin"}, p, err):
                pin = sg["pin"]
                if sg["io"] not in SIGNAL_IO:
                    err.add(f"{p}.io: must be one of {sorted(SIGNAL_IO)}")
                elif not isinstance(pin, int) or isinstance(pin, bool) or not 0 <= pin <= SIGNAL_IO[sg["io"]]:
                    err.add(f"{p}.pin: must be a whole number 0..{SIGNAL_IO[sg['io']]}")
                else:
                    signals[sname] = Signal(sg["io"], pin)
    simulate: list[tuple[str, str, float]] = []
    sims = data.get("simulate", [])
    if not isinstance(sims, list):
        err.add("simulate: must be a list")
        sims = []
    for i, sm in enumerate(sims):
        p = f"simulate[{i}]"
        if _keys(sm, {"when", "then", "after_s"}, {"when", "then", "after_s"}, p, err):
            after = _num(sm["after_s"], f"{p}.after_s", err, 0, 600)
            if sm["when"] not in signals or sm["then"] not in signals:
                err.add(f"{p}: 'when' and 'then' must be names in the cell's signals")
            elif not signals[sm["then"]].io.endswith("_out"):
                err.add(f"{p}: '{sm['then']}' must be mapped to an output, so the pretend machine can set it")
            elif after is not None:
                simulate.append((sm["when"], sm["then"], after))
    err.raise_if_any()
    return Cell(data["name"], robot, (*xyz, yaw), tcp, payload, gripper, home, speeds, gap, min_h, area, signals,
                simulate)


def load_cell(path: Path) -> Cell:
    return parse_cell(json.loads(Path(path).read_text()))


def load_task(path: Path, cell: Cell | None = None) -> Task:
    return parse_task(json.loads(Path(path).read_text()), set(cell.speeds) if cell else None)
