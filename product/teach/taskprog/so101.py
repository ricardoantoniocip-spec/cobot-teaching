"""SO-101: turn the same plan into a joint trajectory (traj.csv) for the twin and the real arm.

The SO-101 has 5 joints, so it cannot turn the gripper freely about the vertical: the jaws
always close along the line from the robot base to the object. A location's yaw is ignored
here (reported by `check`). Its tool point is the tip of the fixed finger, not the middle of
the jaws, so the target is shifted by half the object width plus the finger thickness, the
same geometry the scripted demo in product/sim uses.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from retarget.demo import JointTrajectory
from retarget.kinematics import SerialChain, axis_angle_to_matrix, solve_ik_with_restarts

from .plan import Motion, make_plan
from .task import Cell, Task

FINGER_HALF_THICKNESS_M = 0.006
HZ = 30.0
REACH_TOL_M = 0.005


@lru_cache(maxsize=1)
def chain() -> SerialChain:
    return SerialChain.from_urdf()


@lru_cache(maxsize=1)
def _down() -> np.ndarray:
    r0 = chain().fk(np.array([0.0, 0.1, 0.3, 0.9, 0.0]))[:3, :3]
    z = r0[:, 2]
    tilt = np.arctan2(z[0], -z[2])
    return axis_angle_to_matrix(np.array([0.0, 1.0, 0.0]), tilt) @ r0


def tool_target(m: Motion, task: Task) -> np.ndarray:
    """SO-101 tool pose for a plan pose: fingers down, jaws radial, tip on the inner side."""
    p = m.pose[:3, 3].copy()
    radial = p[:2] / max(np.linalg.norm(p[:2]), 1e-6)
    obj = _object_for(m, task)
    if obj is not None:
        p[:2] -= radial * (obj.size_mm[0] / 2000.0 + FINGER_HALF_THICKNESS_M)
    t = np.eye(4)
    t[:3, :3] = axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), np.arctan2(radial[1], radial[0])) @ _down()
    t[:3, 3] = p
    return t


def _object_for(m: Motion, task: Task):
    step = next((s for s in task.steps if s.id == m.step), None)
    return task.objects.get(step.obj) if step is not None and step.obj else None


def _profile(n: int) -> np.ndarray:
    k = np.arange(1, n + 1) / n
    return 0.5 - 0.5 * np.cos(np.pi * k)


def trajectory(task: Task, cell: Cell) -> tuple[JointTrajectory, list[tuple[Motion, float]]]:
    """Plan -> joint trajectory at 30 Hz. Also returns the worst IK error (m) per motion."""
    plan = make_plan(task, cell)
    ch = chain()
    q = np.radians(cell.home_joints_deg)
    grip = 1.0  # 1 open, 0 closed (retarget convention)
    ts, qs, gs, es = [0.0], [q.copy()], [grip], [0.0]
    report: list[tuple[Motion, float]] = []
    tool_now = ch.fk(q)

    def push(q_new, g, e):
        ts.append(ts[-1] + 1 / HZ)
        qs.append(q_new.copy())
        gs.append(g)
        es.append(e)

    for m in plan:
        sp = cell.speeds[m.speed]
        worst = 0.0
        if m.kind in ("joints", "ptp"):
            if m.kind == "joints":
                q_t, e = m.joints.copy(), 0.0
            else:
                res = solve_ik_with_restarts(ch, tool_target(m, task), q, orientation_weight=0.05)
                q_t, e = res.q, res.position_error_m
            worst = e
            dur = max(np.abs(q_t - q).max() / np.radians(sp.joint_deg_s), 0.3)
            for a in _profile(int(np.ceil(dur * HZ))):
                push(q + a * (q_t - q), grip, e)
            q = q_t
            tool_now = tool_target(m, task) if m.kind == "ptp" else ch.fk(q)
        elif m.kind == "line":
            target = tool_target(m, task)
            dist = np.linalg.norm(target[:3, 3] - tool_now[:3, 3])
            dur = max(dist / (sp.linear_mm_s / 1000.0), 0.3)
            start = tool_now[:3, 3].copy()
            for a in _profile(int(np.ceil(dur * HZ))):
                t = target.copy()
                t[:3, 3] = start + a * (target[:3, 3] - start)
                res = solve_ik_with_restarts(ch, t, q, orientation_weight=0.05)
                q = res.q
                worst = max(worst, res.position_error_m)
                push(q, grip, res.position_error_m)
            tool_now = target
        elif m.kind == "grip":
            g_t = 0.0 if m.close else 1.0
            n = max(int(np.ceil(m.seconds * HZ)), 1)
            for k in range(1, n + 1):
                push(q, grip + (g_t - grip) * min(1.0, 2 * k / n), 0.0)  # move in the first half, then settle
            grip = g_t
        elif m.kind == "wait":
            for _ in range(int(np.ceil(m.seconds * HZ))):
                push(q, grip, 0.0)
        report.append((m, worst))
    traj = JointTrajectory(t=np.array(ts), q=np.array(qs), gripper_open=np.array(gs), pos_error_m=np.array(es))
    return traj, report
