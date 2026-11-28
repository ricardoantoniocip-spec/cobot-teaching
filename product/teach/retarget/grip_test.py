"""Gripper-only test: find the motor load that means "the jaws feel a part", without risking the servo.

    python -m retarget grip-test --port COM4

Only the gripper moves. Start with the arm in its rest pose (folded down): the arm motors hold it during
the test and switch off at the end (the arm goes limp, same as every other command).

Three closes, each one slow (10 % of the jaw range a second) and stopped early:
  A. nothing between the jaws   -> the load of an empty close (the baseline)
  B. a soft part (sponge, foam, folded cloth)
  C. your real box (optional)
Every close stops at once when the load passes --limit (default 400 of 1000; LeRobot already caps the
gripper motor at 50 % torque), or when the jaw stops following for 3 readings. Then the goal is set to
where the jaw is (no squeeze) and, after a second, the gripper opens again.

At the end it prints the --grip-load to use in the Teach pendant and writes every reading to grip_test.csv.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

from .robot import JointMapping

STEP_PCT = 0.35  # goal step per reading (30 readings a second -> ~10 %/s)
HZ = 30.0
LAG_PCT = 2.5  # jaw behind the goal by more than this = blocked


@dataclass
class CloseResult:
    label: str
    stopped_by: str  # "load", "blocked" or "closed"
    contact_pct: float  # where the jaw stopped (gripper %)
    peak_load: float
    rows: list = field(default_factory=list)


def read(bus) -> tuple[float, float]:
    pos = float(bus.read("Present_Position", "gripper"))
    try:
        load = abs(float(bus.read("Present_Load", "gripper")))
    except Exception:  # noqa: BLE001
        load = 0.0
    return pos, load


def move_gripper(bus, target_pct: float, sleep=time.sleep) -> None:
    """Slow open (no load check needed: opening pushes on nothing)."""
    pos, _ = read(bus)
    goal = pos
    while abs(goal - target_pct) > 1e-6:
        goal += max(-STEP_PCT * 2, min(STEP_PCT * 2, target_pct - goal))
        bus.write("Goal_Position", "gripper", goal)
        sleep(1 / HZ)


def close_slowly(bus, label: str, closed_pct: float, limit: float, sleep=time.sleep, log=print) -> CloseResult:
    pos, load = read(bus)
    goal = pos
    peak, lag_n, t0 = load, 0, time.monotonic()
    rows = []
    stopped = "closed"
    while goal > closed_pct:
        goal = max(closed_pct, goal - STEP_PCT)
        bus.write("Goal_Position", "gripper", goal)
        sleep(1 / HZ)
        pos, load = read(bus)
        peak = max(peak, load)
        rows.append((label, round(time.monotonic() - t0, 3), round(goal, 2), round(pos, 2), load))
        lag_n = lag_n + 1 if pos - goal > LAG_PCT else 0
        if load > limit:
            stopped = "load"
            break
        if lag_n >= 3:
            stopped = "blocked"
            break
    bus.write("Goal_Position", "gripper", pos)  # stop pushing: goal = where the jaw is
    log(f"  {label}: stopped by {stopped} at {pos:.1f} % (goal {goal:.1f} %), peak load {peak:.0f}")
    return CloseResult(label, stopped, pos, peak, rows)


def suggest(empty: CloseResult, parts: list[CloseResult]) -> float | None:
    """A load threshold above the empty close and below what a part produced. None if they overlap."""
    if not parts:
        return None
    contact = min(p.peak_load for p in parts)
    low = empty.peak_load * 1.3 + 20
    if contact <= low:
        return None
    return round((low + contact) / 2 / 10) * 10


def grip_test(robot, mapping: JointMapping, limit: float = 400.0, ask=input, sleep=time.sleep, log=print,
              out: Path = Path("grip_test.csv")) -> int:
    bus = robot.bus
    pos, load = read(bus)
    log(f"Gripper now at {pos:.1f} %, load {load:.0f}. Open = {mapping.gripper_open_pct:.0f} %, "
        f"closed = {mapping.gripper_closed_pct:.0f} %. Safety stop at load {limit:.0f}.")
    results: list[CloseResult] = []
    tests = [("A_empty", "A. Nothing between the jaws. Press Enter to close (Ctrl+C stops). "),
             ("B_soft", "B. Hold a SOFT part (sponge, foam, folded cloth) between the jaws. Press Enter. "),
             ("C_box", "C. Optional: your real box between the jaws. Press Enter, or type s to skip. ")]
    for label, prompt in tests:
        log("Opening...")
        move_gripper(bus, mapping.gripper_open_pct, sleep)
        if ask(prompt).strip().lower().startswith("s"):
            continue
        results.append(close_slowly(bus, label, mapping.gripper_closed_pct, limit, sleep, log))
        sleep(1.0)
    log("Opening...")
    move_gripper(bus, mapping.gripper_open_pct, sleep)

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test", "t_s", "goal_pct", "jaw_pct", "load"])
        for r in results:
            w.writerows(r.rows)
    log(f"\nAll readings in {out.resolve()}")
    empty = next((r for r in results if r.label == "A_empty"), None)
    parts = [r for r in results if r.label != "A_empty"]
    log("Summary:")
    for r in results:
        log(f"  {r.label:8s} peak load {r.peak_load:4.0f}   stopped by {r.stopped_by:7s} at {r.contact_pct:5.1f} %")
    g = suggest(empty, parts) if empty else None
    if g is not None:
        log(f"\nUse:  python -m so101_sim view --port COM4 --grip-load {g:.0f}")
    elif parts and all(p.stopped_by == "blocked" for p in parts):
        log("\nThe load hardly rose with a part, but the jaw was seen to stop: the pendant's blocked-jaw check grips anyway. "
            "Keep the default --grip-load.")
    else:
        log("\nNo clear threshold: send me the summary above and grip_test.csv.")
    return 0
