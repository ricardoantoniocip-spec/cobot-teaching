"""Command line: python -m taskprog <command> ...  (run from product/teach)

  frames    VIDEO STEPS OUT_DIR             frames + prompt for labelling by hand (no API key needed)
  label     VIDEO STEPS OUT_LABELS          vision LLM labels the demo steps (needs an API key)
  from-steps STEPS OUT_TASK [--labels L]    demo steps + labels -> task.json
  check     TASK --cell CELL                offline checks (reach, singularities, heights, review notes)
  urscript  TASK --cell CELL OUT.script     task -> URScript (refuses if check finds an ERROR)
  so101     TASK --cell CELL OUT.csv        task -> SO-101 joint trajectory (plays with retarget replay)
  twin      TASK --cell CELL [--video V]    play the SO-101 trajectory in the MuJoCo twin
  ursim     TASK --cell CELL [--host H]     generate URScript and run it in URSim, compare poses
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .check import check, passed
from .task import TaskError, load_cell, load_task


def _load(args):
    cell = load_cell(args.cell)
    return load_task(args.task, cell), cell


def _print_check(task, cell) -> bool:
    findings = check(task, cell)
    for f in findings:
        print("  " + str(f))
    ok = passed(findings)
    print(f"  check: {'PASS' if ok else 'FAIL'} ({sum(f.level == 'ERROR' for f in findings)} errors, "
          f"{sum(f.level == 'WARN' for f in findings)} warnings for the engineer)")
    return ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m taskprog", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("frames")
    s.add_argument("video", type=Path)
    s.add_argument("steps", type=Path)
    s.add_argument("out", type=Path)

    s = sub.add_parser("label")
    s.add_argument("video", type=Path)
    s.add_argument("steps", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--model", default=None)

    s = sub.add_parser("from-steps")
    s.add_argument("steps", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--labels", type=Path)
    s.add_argument("--object-size", type=float, nargs=3, default=[20.0, 20.0, 40.0], metavar=("W", "D", "H"),
                   help="object size in mm: width across the jaws, depth, height (default: the twin's block)")
    s.add_argument("--grasp-height", type=float, help="mm above the object's bottom where the gripper closes")
    s.add_argument("--measured", action="store_true", help="the object size was measured (no review note)")
    s.add_argument("--flat-table", action="store_true", help="put every location on one table at z = 0")

    for name in ("check", "urscript", "so101", "twin", "ursim"):
        s = sub.add_parser(name)
        s.add_argument("task", type=Path)
        s.add_argument("--cell", type=Path, required=True)
        if name in ("urscript", "so101"):
            s.add_argument("out", type=Path)
        if name == "twin":
            s.add_argument("--video", type=Path)
        if name == "ursim":
            s.add_argument("--host", default="localhost")
            s.add_argument("--save", type=Path, help="also write the program here")

    args = p.parse_args(argv)
    try:
        return _run(args)
    except TaskError as e:
        print(e)
        return 2


def _run(args) -> int:
    if args.cmd == "frames":
        from .label import write_frames

        out = write_frames(args.video, json.loads(args.steps.read_text()), args.out)
        print(f"Frames and prompt in {out}. Write labels.json in the shape at the end of prompt.txt.")
        return 0

    if args.cmd == "label":
        from .label import MODEL, label

        labels = label(args.video, json.loads(args.steps.read_text()), args.model or MODEL)
        args.out.write_text(json.dumps(labels, indent=2) + "\n")
        print(f"Labels written to {args.out} (by {labels['_labelled_by']}).")
        return 0

    if args.cmd == "from-steps":
        from .from_steps import convert

        task = convert(args.steps, args.out, args.labels, object_size_mm=tuple(args.object_size),
                       grasp_height_mm=args.grasp_height, flat_table=args.flat_table, size_measured=args.measured)
        print(f"Task '{task['name']}' with {len(task['steps'])} steps written to {args.out}.")
        for r in task["review"]:
            print(f"  review: {r}")
        return 0

    task, cell = _load(args)
    print(f"Task '{task.name}' on cell '{cell.name}' ({cell.robot})")

    if args.cmd == "check":
        return 0 if _print_check(task, cell) else 1

    if args.cmd in ("urscript", "ursim"):
        if cell.robot == "so101":
            print("  this is an SO-101 cell; use the so101 or twin command")
            return 2
        if not _print_check(task, cell):
            print("  no program written: fix the errors first")
            return 1
        from .ur import urscript

        program = urscript(task, cell, args.task.read_text(), args.task.name)
        out = args.out if args.cmd == "urscript" else args.save
        if out:
            out.write_text(program)
            print(f"  URScript written to {out} ({program.count(chr(10))} lines)")
        if args.cmd == "urscript":
            return 0
        return _ursim(task, cell, program, args.host)

    if args.cmd == "so101":
        if cell.robot != "so101":
            print("  this is not an SO-101 cell")
            return 2
        if not _print_check(task, cell):
            print("  no trajectory written: fix the errors first")
            return 1
        from retarget.demo import write_trajectory

        from .so101 import trajectory

        traj, _ = trajectory(task, cell)
        write_trajectory(args.out, traj)
        print(f"  {len(traj.t)} samples, {traj.t[-1]:.1f} s, written to {args.out}")
        print("  play it: python -m retarget replay OUT.csv --port COM5   (or the twin command)")
        return 0

    if args.cmd == "twin":
        from .twin import run_twin

        _, res = run_twin(task, cell, args.video)
        print("  " + res.report().replace("\n", "\n  "))
        return 0 if res.success else 1
    return 2


def _ursim(task, cell, program: str, host: str) -> int:
    import numpy as np

    from .plan import make_plan
    from .ur import URKinematics, reach_check
    from .ursim import compare, run

    print(f"  sending to URSim at {host} ...")
    res = run(host, program)
    print(f"  URSim: {'finished' if res.finished else 'NOT finished'} in {res.seconds:.0f} s, {len(res.step_poses)} step poses")
    for e in res.errors:
        print(f"  URSim error: {e}")
    # Expected pose after each step = where the last motion of that step ends.
    kin = URKinematics(cell.robot, cell.tcp_mm)
    rows = reach_check(make_plan(task, cell), cell)
    expected = {}
    for r in rows:
        expected[r.motion.step] = r.motion.pose if r.motion.pose is not None else kin.fk(r.q)
    problems = compare(res, expected)
    for pr in problems:
        print(f"  pose: {pr}")
    if res.finished and not res.errors and not problems:
        worst = max((np.linalg.norm(np.array(res.step_poses[k][:3]) - v[:3, 3]) * 1000 for k, v in expected.items()), default=0)
        print(f"  URSim: PASS (every step ends within {max(worst, 0.1):.1f} mm of the plan; URSim reports poses to 0.1 mm)")
        return 0
    print("  URSim: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
