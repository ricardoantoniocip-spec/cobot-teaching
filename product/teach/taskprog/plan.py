"""Expand a task into robot motions. Fixed rules, no AI: the same task always gives the same plan.

  home         joint move to the cell's home joints
  move  L      joint move to `height` above L
  pick  O at L joint move to approach height above L, straight line down to the grasp height,
               close, wait, straight line back up
  place O at L joint move above L, straight line down to grasp height + release gap,
               open, wait, straight line back up
  wait  s      wait s seconds
  set_output   set a named signal (optionally as a pulse)
  wait_input   wait until a named signal has the value, stop the program after the timeout

A step at a grid location works at one slot; the motion's location is then "tray_s3".

Every motion keeps the location name and the height above it, so a code generator can write
`pose_trans(loc_bin_a, p[0, 0, -0.05, 0, 0, 0])` instead of bare numbers, and an engineer can
re-teach one location without touching the rest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .task import Cell, Location, Task


@dataclass
class Motion:
    kind: str  # "joints", "ptp" (joint move to a pose), "line", "grip", "wait", "output", "input"
    step: int  # task step id
    note: str  # one line for the program comment
    speed: str
    location: str | None = None
    height_m: float = 0.0  # tool point above the location's surface
    pose: np.ndarray | None = None  # 4x4 tool pose in the robot base frame (metres)
    joints: np.ndarray | None = None  # radians, for kind "joints"
    close: bool | None = None  # for kind "grip"
    seconds: float = 0.0  # "grip" (settle time), "wait", "output" (pulse, 0 = none), "input" (timeout)
    signal: str | None = None  # for "output" and "input"
    value: bool = True


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


TOOL_DOWN = np.diag([1.0, -1.0, -1.0])  # rotation of pi about x: the tool z axis points down


def work_to_base(cell: Cell) -> np.ndarray:
    x, y, z, yaw = cell.work_frame_mm_deg
    t = np.eye(4)
    t[:3, :3] = rot_z(np.radians(yaw))
    t[:3, 3] = np.array([x, y, z]) / 1000.0
    return t


def location_pose(loc: Location, cell: Cell) -> np.ndarray:
    """Pose of the location in the robot base frame: on the surface, tool z pointing down."""
    t = np.eye(4)
    t[:3, :3] = rot_z(np.radians(loc.yaw_deg)) @ TOOL_DOWN
    t[:3, 3] = np.array(loc.xyz_mm) / 1000.0
    return work_to_base(cell) @ t


def raise_pose(pose: np.ndarray, height_m: float) -> np.ndarray:
    """The pose `height_m` above (along -tool z, which is straight up for a tool-down pose)."""
    off = np.eye(4)
    off[2, 3] = -height_m
    return pose @ off


def make_plan(task: Task, cell: Cell) -> list[Motion]:
    out: list[Motion] = []

    def at(kind: str, s, loc: str, h: float, note: str, speed: str) -> Motion:
        return Motion(kind, s.id, note, speed, loc, h, raise_pose(location_pose(task.target(s), cell), h))

    # Start known: open gripper at home.
    out.append(Motion("joints", 0, "start at home", task.speed, joints=np.radians(cell.home_joints_deg)))
    out.append(Motion("grip", 0, "open gripper", task.speed, close=False, seconds=task.grip_time_s))
    for s in task.steps:
        sp = s.speed or task.speed
        if s.do == "home":
            out.append(Motion("joints", s.id, "home", sp, joints=np.radians(cell.home_joints_deg)))
        elif s.do == "move":
            h = (s.height_mm or task.travel_height_mm) / 1000.0
            out.append(at("ptp", s, task.target(s).name, h, f"above {where(s)}", sp))
        elif s.do == "wait":
            out.append(Motion("wait", s.id, f"wait {s.seconds:g} s", sp, seconds=s.seconds))
        elif s.do == "set_output":
            note = f"{s.signal} {'on' if s.value else 'off'}" + (f" for {s.pulse_s:g} s" if s.pulse_s else "")
            out.append(Motion("output", s.id, note, sp, signal=s.signal, value=s.value, seconds=s.pulse_s or 0.0))
        elif s.do == "wait_input":
            note = f"wait for {s.signal} {'on' if s.value else 'off'}, at most {s.timeout_s:g} s"
            out.append(Motion("input", s.id, note, sp, signal=s.signal, value=s.value, seconds=s.timeout_s))
        elif s.do in ("pick", "place"):
            obj = task.objects[s.obj]
            grasp = obj.grasp_height_mm / 1000.0
            if s.do == "place":
                grasp += cell.release_gap_mm / 1000.0
            app = (s.approach_mm or task.approach_mm) / 1000.0
            verb = "grasp" if s.do == "pick" else "release"
            loc = task.target(s).name
            out.append(at("ptp", s, loc, grasp + app, f"above {where(s)}", sp))
            out.append(at("line", s, loc, grasp, f"down to {verb} height", sp))
            out.append(Motion("grip", s.id, "close gripper" if s.do == "pick" else "open gripper", sp,
                              close=(s.do == "pick"), seconds=task.grip_time_s))
            out.append(at("line", s, loc, grasp + app, f"back up from {where(s)}", sp))
    return out


def where(s) -> str:
    return s.at if s.slot is None else f"{s.at} slot {s.slot}"
