"""First power-up of the SO-101 follower: check every joint against the model, one at a time.

    python -m retarget bringup --port COM5

Do this after LeRobot's own setup (motor ids, calibration; see hardware/so101/first-power-up.md).
For each joint the arm moves 10 degrees, slowly, and you say whether the gripper tip went the way
the model predicts. A joint that goes the other way gets its sign flipped. The answers are saved
in product/teach/so101_mapping.json, which replay, check and live mode then use.

UNTESTED ON HARDWARE (written before the servos arrived; tested against a stand-in robot).
Keep a hand on the power switch. Every move is at 15 degrees a second.
"""

from __future__ import annotations

import numpy as np

from .kinematics import ARM_JOINTS, SerialChain, make_transform, solve_ik_with_restarts, tool_down
from .robot import JointMapping, current_joints, move_to

STEP_DEG = 10.0
SPEED = np.radians(15.0)
FIRST_POSE_M = np.array([0.18, 0.0, 0.06])  # tool point of the first real move: 18 cm in front of the base, 6 cm up


def direction_words(d_mm: np.ndarray) -> str:
    """A tool displacement in the base frame as words. x = away from the base, y = left, z = up, seen from behind the arm."""
    names = (("forward", "back"), ("left", "right"), ("up", "down"))
    parts = [f"{abs(v):.0f} mm {names[i][0] if v > 0 else names[i][1]}" for i, v in enumerate(d_mm) if abs(v) >= 5]
    return " and ".join(parts) if parts else "almost nothing (it only turns)"


def bringup(robot, chain: SerialChain, ask=input, say=print, mapping: JointMapping | None = None, save: bool = True) -> int:
    mapping = mapping or JointMapping()
    say("STEP 1 of 4: reading the joints. Nothing moves.")
    q = current_joints(robot, mapping)
    for j, v, lo, hi in zip(ARM_JOINTS, q, chain.lower, chain.upper):
        note = "" if lo - 0.1 <= v <= hi + 0.1 else "   <-- outside the model's range: calibrate again (lerobot-calibrate)"
        say(f"  {j:14s} {np.degrees(v):7.1f} deg{note}")
    if ask("Put the arm in a middle pose, clear of the table, with space around it. Type 'yes' to start the joint check: ").strip().lower() != "yes":
        say("stopped")
        return 1

    say(f"\nSTEP 2 of 4: each joint moves {STEP_DEG:.0f} degrees and back. Look at the gripper tip from behind the arm.")
    problems = []
    for i, j in enumerate(ARM_JOINTS):
        q = current_joints(robot, mapping)
        step = STEP_DEG if q[i] + np.radians(STEP_DEG) <= chain.upper[i] else -STEP_DEG  # away from the end stop
        q2 = q.copy()
        q2[i] += np.radians(step)
        expect = (chain.fk(q2)[:3, 3] - chain.fk(q)[:3, 3]) * 1000
        say(f"\n{j}: the model says the tip moves {direction_words(expect)}." + ("  (wrist roll: watch the jaws turn instead.)" if j == "wrist_roll" else ""))
        move_to(robot, mapping, q, q2, 1.0, SPEED)
        moved = np.degrees(current_joints(robot, mapping)[i] - q[i])
        say(f"  asked {step:+.0f} deg, the motor reports {moved:+.1f} deg")
        if abs(moved - step) > 4:
            problems.append(f"{j}: asked {step:+.0f} deg but the motor reports {moved:+.1f}. Check the cable, the motor id and that nothing blocks it.")
        ans = ask("  Did it move that way? y = yes, n = the other way, x = it did not move, q = stop now: ").strip().lower()
        move_to(robot, mapping, q2, q, 1.0, SPEED)
        if ans == "q":
            say("stopped")
            return 1
        if ans == "n":
            mapping.signs[j] = -mapping.signs[j]
            mapping.offsets_deg[j] = -mapping.offsets_deg[j]
            say(f"  noted: {j} runs the other way. Sign flipped. Run bringup once more after this to confirm.")
        elif ans != "y":
            problems.append(f"{j}: did not move as expected (answer '{ans}').")

    say("\nSTEP 3 of 4: the gripper opens, then closes.")
    q = current_joints(robot, mapping)
    for g, word in ((1.0, "open"), (0.0, "closed"), (1.0, "open")):
        robot.send_action(mapping.to_action(q, g))
        ask(f"  The gripper should now be {word}. Press ENTER. ")

    if save:
        say(f"\nSaved the joint mapping to {mapping.save()}")
    if problems:
        say("\nFix these before moving on:")
        for p in problems:
            say("  - " + p)
        return 1
    if any(s < 0 for s in mapping.signs.values()):
        say("A sign was flipped. Run bringup again: every joint should then answer 'y'.")
        return 1

    say(f"\nSTEP 4 of 4: first real move. The gripper goes to a point {FIRST_POSE_M[0] * 100:.0f} cm in front of the base, {FIRST_POSE_M[2] * 100:.0f} cm above the table, pointing down.")
    if ask("Clear the table in front of the arm. Type 'yes' to move: ").strip().lower() != "yes":
        say("skipped")
        return 0
    q = current_joints(robot, mapping)
    goal = solve_ik_with_restarts(chain, make_transform(tool_down(chain, 0.0), FIRST_POSE_M), np.array([0.0, 0.0, 0.3, 1.0, 0.0])).q
    move_to(robot, mapping, q, goal, 1.0, SPEED)
    tip = chain.fk(current_joints(robot, mapping))[:3, 3] * 1000
    say(f"The model says the tip is now {tip[0]:.0f} mm in front of the base centre, {tip[1]:.0f} mm to the left, {tip[2]:.0f} mm above the table.")
    say("Measure it with a ruler. Within 10 mm: good, go on to replay and live mode. More: write the numbers down and tell Claude.")
    return 0
