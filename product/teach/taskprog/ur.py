"""Universal Robots: URScript generator and an offline reach check (UR3e / UR5e / UR10e).

The generator is plain string templates over the plan. It writes numbers that came from the
task file and cell file, comments that came from the (cleaned) labels, and nothing else.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from . import __version__
from .plan import TOOL_DOWN, Motion, make_plan, rot_z, where
from .task import Cell, Task, target_name

# Standard DH parameters published by Universal Robots (d1, a2, a3, d4, d5, d6), metres.
# https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
DH = {
    "ur3e": (0.15185, -0.24355, -0.2132, 0.13105, 0.08535, 0.0921),
    "ur5e": (0.1625, -0.425, -0.3922, 0.1333, 0.0997, 0.0996),
    "ur10e": (0.1807, -0.6127, -0.57155, 0.17415, 0.11985, 0.11655),
}
ALPHA = (np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0)


def matrix_to_rotvec(r: np.ndarray) -> np.ndarray:
    """Rotation matrix to UR rotation vector, robust at 180 degrees (tool-down poses sit there)."""
    cos_a = np.clip((np.trace(r) - 1) / 2, -1.0, 1.0)
    angle = float(np.arccos(cos_a))
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-4:
        # axis axis^T = (sym(R) - cos(a) I) / (1 - cos(a)); the skew part gives the sign.
        b = ((r + r.T) / 2 - cos_a * np.eye(3)) / (1 - cos_a)
        i = int(np.argmax(np.diag(b)))
        axis = b[:, i] / np.sqrt(b[i, i])
        w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
        if axis @ w < 0:
            axis = -axis
        return axis / np.linalg.norm(axis) * angle
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return w / (2 * np.sin(angle)) * angle


def rotvec_to_matrix(v: np.ndarray) -> np.ndarray:
    a = np.linalg.norm(v)
    if a < 1e-12:
        return np.eye(3)
    x, y, z = v / a
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def pose_to_p(t: np.ndarray) -> list[float]:
    return [*t[:3, 3], *matrix_to_rotvec(t[:3, :3])]


class URKinematics:
    def __init__(self, robot: str, tcp_mm=(0.0, 0.0, 0.0)):
        self.d1, self.a2, self.a3, self.d4, self.d5, self.d6 = DH[robot]
        self.tcp = np.eye(4)
        self.tcp[:3, 3] = np.array(tcp_mm) / 1000.0

    def fk(self, q: np.ndarray) -> np.ndarray:
        d = (self.d1, 0, 0, self.d4, self.d5, self.d6)
        a = (0, self.a2, self.a3, 0, 0, 0)
        t = np.eye(4)
        for i in range(6):
            ct, st = np.cos(q[i]), np.sin(q[i])
            ca, sa = np.cos(ALPHA[i]), np.sin(ALPHA[i])
            t = t @ np.array([[ct, -st * ca, st * sa, a[i] * ct], [st, ct * ca, -ct * sa, a[i] * st], [0, sa, ca, d[i]], [0, 0, 0, 1]])
        return t @ self.tcp

    def _err(self, target: np.ndarray, q: np.ndarray) -> np.ndarray:
        t = self.fk(q)
        return np.concatenate([target[:3, 3] - t[:3, 3], matrix_to_rotvec(target[:3, :3] @ t[:3, :3].T)])

    def ik(self, target: np.ndarray, qnear: np.ndarray, iterations: int = 200) -> tuple[np.ndarray, float, float]:
        """Damped least squares from qnear (like the controller's get_inverse_kin with qnear).
        Returns joints, position error (m), orientation error (rad)."""
        q = np.asarray(qnear, dtype=float).copy()
        for _ in range(iterations):
            e = self._err(target, q)
            if np.linalg.norm(e[:3]) < 1e-6 and np.linalg.norm(e[3:]) < 1e-5:
                break
            jac = np.zeros((6, 6))
            for i in range(6):
                dq = np.zeros(6)
                dq[i] = 1e-6
                t1, t0 = self.fk(q + dq), self.fk(q)
                jac[:3, i] = (t1[:3, 3] - t0[:3, 3]) / 1e-6
                jac[3:, i] = matrix_to_rotvec(t1[:3, :3] @ t0[:3, :3].T) / 1e-6
            step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), e)
            n = np.linalg.norm(step)
            q += step * (0.3 / n if n > 0.3 else 1.0)
        e = self._err(target, q)
        return q, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))

    def singularity(self, q: np.ndarray) -> str | None:
        """Name the nearby singularity, if any (UR's three kinds)."""
        if abs(np.sin(q[4])) < 0.1:
            return "wrist (joint 5 near 0 or 180 deg)"
        if abs(np.sin(q[2])) < 0.1:
            return "elbow (arm almost straight)"
        # Shoulder: wrist centre close to the base z axis.
        flange = self.fk(q) @ np.linalg.inv(self.tcp)
        wc = flange[:3, 3] - self.d6 * flange[:3, 2]
        if np.hypot(wc[0], wc[1]) < 0.08:
            return "shoulder (wrist above the base)"
        return None


@dataclass
class ReachRow:
    motion: Motion
    q: np.ndarray
    pos_err_mm: float
    problem: str | None


def reach_check(plan: list[Motion], cell: Cell, line_samples: int = 20) -> list[ReachRow]:
    """IK every target like the controller would, and walk every straight line in small steps."""
    kin = URKinematics(cell.robot, cell.tcp_mm)
    home = np.radians(cell.home_joints_deg)
    q_now = home.copy()
    pose_now = kin.fk(q_now)
    rows: list[ReachRow] = []
    for m in plan:
        problem = None
        err_mm = 0.0
        if m.kind == "joints":
            q_now = m.joints.copy()
            pose_now = kin.fk(q_now)
        elif m.kind == "ptp":
            q, pe, oe = kin.ik(m.pose, home)
            err_mm = pe * 1000
            if pe > 0.001 or oe > np.radians(0.5):
                problem = f"out of reach ({err_mm:.0f} mm short)"
            elif np.abs(q - home).max() > np.radians(170):
                problem = "joint solution far from home (the arm would swing a joint more than 170 deg)"
            elif not -170 <= np.degrees(q[1]) <= -10 or q[2] * home[2] < 0:
                problem = "odd arm posture (leaning over backwards or elbow flipped): is the work frame on the side the home pose faces?"
            q_now, pose_now = q, m.pose
        elif m.kind == "line":
            q = q_now.copy()
            for k in range(1, line_samples + 1):
                a = k / line_samples
                target = m.pose.copy()
                target[:3, 3] = pose_now[:3, 3] + a * (m.pose[:3, 3] - pose_now[:3, 3])
                q_new, pe, oe = kin.ik(target, q)
                err_mm = max(err_mm, pe * 1000)
                if pe > 0.001 or oe > np.radians(0.5):
                    problem = f"straight line leaves the reach ({pe * 1000:.0f} mm short)"
                    break
                if np.abs(q_new - q).max() > np.radians(15):
                    problem = "straight line needs a sudden joint jump (singularity on the way)"
                    break
                q = q_new
            q_now, pose_now = q, m.pose
        if problem is None and m.kind in ("ptp", "line"):
            sing = kin.singularity(q_now)
            if sing:
                problem = f"close to a {sing} singularity"
        rows.append(ReachRow(m, q_now.copy(), err_mm, problem))
    return rows


# ------------------------------------------------------------------ URScript


def _f(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _p(v) -> str:
    return "p[" + ", ".join(_f(x) for x in v) + "]"


def _q(v) -> str:
    return "[" + ", ".join(_f(x) for x in v) + "]"


def step_title(s) -> str:
    if s.do in ("pick", "place"):
        return f"{s.do} {s.obj} at {where(s)}"
    if s.do == "move":
        return f"move to {where(s)}"
    if s.do == "wait":
        return f"wait {s.seconds:g} s"
    if s.do == "set_output":
        return f"set {s.signal} {'on' if s.value else 'off'}" + (f" for {s.pulse_s:g} s" if s.pulse_s else "")
    if s.do == "wait_input":
        return f"wait for {s.signal} {'on' if s.value else 'off'}"
    return "home"


def _io(cell: Cell, signal: str) -> tuple[str, str, int]:
    """(read call, write function or '', pin) for a task signal on this cell."""
    sg = cell.signals[signal]
    bank, direction = sg.io.rsplit("_", 1)
    read = f"get_{bank}_digital_{direction}({sg.pin})"
    write = f"set_{bank}_digital_out" if direction == "out" else ""
    return read, write, sg.pin


def _bool(v: bool) -> str:
    return "True" if v else "False"


def urscript(task: Task, cell: Cell, task_text: str = "", task_file: str = "task.json") -> str:
    plan = make_plan(task, cell)
    digest = hashlib.sha256(task_text.encode()).hexdigest()[:12] if task_text else "unknown"
    x, y, z, yaw = cell.work_frame_mm_deg
    work = [x / 1000, y / 1000, z / 1000, *matrix_to_rotvec(rot_z(np.radians(yaw)))]
    g = cell.gripper
    out_fn = "set_tool_digital_out" if g.output == "tool" else "set_standard_digital_out"
    value = "close" if g.close_value else "not close"
    lines = [
        "# " + "-" * 70,
        f"# Program : {task.name}",
        f"# Task    : {task.description or '-'}",
        f"# Made by : cobot-teaching taskprog {__version__} from {task_file} (sha256 {digest}), cell {cell.name}",
        "# Status  : NOT APPROVED. Run it in URSim first, then on the robot at reduced speed with an",
        "#           engineer present. Do not edit by hand: change the task file and generate again.",
        "# Units   : metres, radians, seconds (the task file uses mm and degrees)",
        *(["# WARNING : this cell SIMULATES a machine (see 'SIMULATION ONLY' below). Not for a real robot."]
          if cell.simulate else []),
        "# " + "-" * 70,
        f"def {task.name}():",
        f"  set_tcp({_p([*(np.array(cell.tcp_mm) / 1000), 0, 0, 0])})",
        f"  set_payload({_f(cell.payload_kg)})",
        "",
        "  # Work frame: where the task's table sits in the robot base frame. Re-teach this one",
        "  # pose to move the whole task.",
        f"  work = {_p(work)}",
        "",
        "  # Locations, in the work frame: on the surface the object stands on, tool pointing down.",
        "  # Heights above a location are pose_trans offsets along -tool z (straight up).",
    ]
    for name, loc in task.locations.items():
        local = [loc.xyz_mm[0] / 1000, loc.xyz_mm[1] / 1000, loc.xyz_mm[2] / 1000,
                 *matrix_to_rotvec(rot_z(np.radians(loc.yaw_deg)) @ TOOL_DOWN)]
        note = f"  # {loc.note}" if loc.note else ""
        lines.append(f"  loc_{name} = pose_trans(work, {_p(local)}){note}")
        if loc.grid:
            g_ = loc.grid
            layers = f", {g_.layers} layers of {g_.layer_height_mm:g} mm" if g_.layers > 1 else ""
            lines.append(f"  # {name} is a grid: {g_.cols} x {g_.rows} slots, pitch {g_.pitch_mm[0]:g} x {g_.pitch_mm[1]:g} mm{layers}."
                         f" loc_{name} is slot 1; re-teach it to move every slot.")
            for slot in sorted({s.slot for s in task.steps if s.at == name and s.slot is not None}):
                dx, dy, dz = loc.slot_offset_mm(slot)
                # The location frame has the tool pointing down: its y and z are flipped.
                off = [dx / 1000, -dy / 1000, -dz / 1000, 0, 0, 0]
                c, r, layer = g_.index(slot)
                lines.append(f"  loc_{target_name(name, slot)} = pose_trans(loc_{name}, {_p(off)})"
                             f"  # column {c + 1}, row {r + 1}" + (f", layer {layer + 1}" if g_.layers > 1 else ""))
    lines += [
        "",
        f"  home_q = {_q(np.radians(cell.home_joints_deg))}",
        "",
        "  def ct_gripper(close, settle_s):",
        f"    {out_fn}({g.pin}, {value})",
        "    sleep(settle_s)",
        "  end",
        "",
    ]
    used = sorted({s.signal for s in task.steps if s.signal})
    if used:
        lines.append("  # Signals (task name = robot I/O):")
        for name in used:
            sg = cell.signals[name]
            lines.append(f"  #   {name} = {sg.io.replace('_', ' ')} {sg.pin} ({task.signals[name]})")
        lines.append("")
    for k, (start, done, after_s) in enumerate(cell.simulate):
        r_start, _, _ = _io(cell, start)
        _, w_done, pin = _io(cell, done)
        lines += [
            f"  # SIMULATION ONLY: a pretend machine. {after_s:g} s after {start} comes on it turns {done} on.",
            f"  thread ct_sim_machine_{k}():",
            "    while True:",
            f"      if {r_start}:",
            f"        {w_done}({pin}, False)",
            f"        sleep({_f(after_s)})",
            f"        {w_done}({pin}, True)",
            f"        while {r_start}:",
            "          sync()",
            "        end",
            "      end",
            "      sync()",
            "    end",
            "  end",
            f"  ct_sim_{k} = run ct_sim_machine_{k}()",
            "",
        ]
    by_step = {s.id: s for s in task.steps}
    current = None
    n = len(task.steps)
    for m in plan:
        if m.step != current:
            if current is not None:
                lines.append(f'  textmsg("CT step {current} done ", get_actual_tcp_pose())')
            current = m.step
            if m.step == 0:
                lines.append("  # start: home, gripper open")
            else:
                s = by_step[m.step]
                what = step_title(s)
                lines.append("")
                lines.append(f"  # step {s.id}/{n}: {what}" + (f"  ({s.label})" if s.label else ""))
                lines.append(f'  textmsg("CT step {s.id} {what}")')
        sp = cell.speeds[m.speed]
        aj, vj = np.radians(sp.joint_deg_s2), np.radians(sp.joint_deg_s)
        al, vl = sp.linear_mm_s2 / 1000, sp.linear_mm_s / 1000
        if m.kind == "joints":
            lines.append(f"  movej(home_q, a={_f(aj)}, v={_f(vj)})  # {m.note}")
        elif m.kind in ("ptp", "line"):
            target = f"loc_{m.location}" if m.height_m == 0 else f"pose_trans(loc_{m.location}, {_p([0, 0, -m.height_m, 0, 0, 0])})"
            if m.kind == "ptp":
                lines.append(f"  movej(get_inverse_kin({target}, qnear=home_q), a={_f(aj)}, v={_f(vj)})  # {m.note}")
            else:
                lines.append(f"  movel({target}, a={_f(al)}, v={_f(vl)})  # {m.note}")
        elif m.kind == "grip":
            lines.append(f"  ct_gripper({'True' if m.close else 'False'}, {_f(m.seconds)})  # {m.note}")
        elif m.kind == "wait":
            lines.append(f"  sleep({_f(m.seconds)})")
        elif m.kind == "output":
            _, write, pin = _io(cell, m.signal)
            lines.append(f"  {write}({pin}, {_bool(m.value)})  # {m.note}")
            if m.seconds:
                lines.append(f"  sleep({_f(m.seconds)})")
                lines.append(f"  {write}({pin}, {_bool(not m.value)})")
        elif m.kind == "input":
            read, _, _ = _io(cell, m.signal)
            waiting = f"not {read}" if m.value else read
            lines += [
                f"  # {m.note}",
                "  ct_waited_s = 0.0",
                f"  while ({waiting}) and (ct_waited_s < {_f(m.seconds)}):",
                "    sleep(0.05)",
                "    ct_waited_s = ct_waited_s + 0.05",
                "  end",
                f"  if {waiting}:",
                f'    textmsg("CT ERROR step {m.step}: {m.signal} did not turn {"on" if m.value else "off"} within {m.seconds:g} s")',
                "    halt",
                "  end",
            ]
    lines.append(f'  textmsg("CT step {current} done ", get_actual_tcp_pose())')
    lines.append('  textmsg("CT program finished")')
    lines.append("end")
    return "\n".join(lines) + "\n"
