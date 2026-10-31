"""Command line: python -m retarget <command> ...  (run from product/teach)

  example  OUT.csv                       write a synthetic hand demo
  video    VIDEO markers.json OUT.csv    hand demo from a video with ArUco markers
  convert  DEMO.csv OUT.csv              hand demo -> SO-101 joint trajectory (no robot needed)
  check    --port COM5                   read the arm's joints, print the tool pose (FK check)
  replay   TRAJ.csv --port COM5          play a trajectory on the follower (asks to confirm)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .demo import read_demo, read_trajectory, write_demo, write_trajectory
from .kinematics import ARM_JOINTS, SerialChain
from .retarget import RetargetConfig, retarget


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m retarget", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("example")
    s.add_argument("out", type=Path)

    s = sub.add_parser("video")
    s.add_argument("video", type=Path)
    s.add_argument("markers", type=Path, help="markers.json (see markers.example.json)")
    s.add_argument("out", type=Path)
    s.add_argument("--camera", type=Path, help="camera.json with fx, fy, cx, cy, dist")
    s.add_argument("--grip-events", type=Path, help="CSV with t_close,t_open per grasp")

    s = sub.add_parser("convert")
    s.add_argument("demo", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--max-speed-deg", type=float, default=30.0)
    s.add_argument("--tool-offset-z", type=float, default=0.0, help="metres along the hand pod z axis to the tool point")

    s = sub.add_parser("check")
    s.add_argument("--port", required=True)

    s = sub.add_parser("replay")
    s.add_argument("traj", type=Path)
    s.add_argument("--port", required=True)
    s.add_argument("--max-speed-deg", type=float, default=30.0)
    s.add_argument("--yes", action="store_true", help="skip the typed confirmation (not recommended)")

    a = p.parse_args(argv)
    chain = SerialChain.from_urdf()

    if a.cmd == "example":
        from .examples import example_pick_place

        write_demo(a.out, example_pick_place(chain))
        print(f"wrote {a.out}")
    elif a.cmd == "video":
        from .video import CameraModel, MarkerSetup, read_grip_events, video_to_demo

        cam = CameraModel.from_json(a.camera) if a.camera else None
        if cam is None:
            print("warning: no --camera file, guessing a 60 degree lens. Calibrate the camera for mm accuracy.")
        demo, warnings = video_to_demo(a.video, MarkerSetup.from_json(a.markers), cam, read_grip_events(a.grip_events))
        for w in warnings:
            print("warning:", w)
        write_demo(a.out, demo)
        print(f"wrote {a.out}: {len(demo)} samples, {demo.t[-1] - demo.t[0]:.1f} s")
    elif a.cmd == "convert":
        cfg = RetargetConfig(max_joint_speed_rad_s=np.radians(a.max_speed_deg))
        cfg.hand_to_tool[2, 3] = a.tool_offset_z
        traj, report = retarget(read_demo(a.demo), chain, cfg)
        write_trajectory(a.out, traj)
        print(report.summary())
        if report.reached_fraction < 0.95:
            print("warning: under 95% of the hand path is reachable. Move the task closer to the arm or check the frames.")
        print(f"wrote {a.out}")
    elif a.cmd == "check":
        from .robot import JointMapping, connect, current_joints

        robot = connect(a.port)
        try:
            q = current_joints(robot, JointMapping())
            for j, v in zip(ARM_JOINTS, q):
                print(f"{j:14s} {np.degrees(v):7.1f} deg")
            tool = chain.fk(q)
            print("tool point in base frame (mm):", np.round(tool[:3, 3] * 1000, 1))
            print("Measure the real gripper tip with a ruler and compare. Note the difference in mm.")
        finally:
            robot.disconnect()
    elif a.cmd == "replay":
        from .robot import JointMapping, connect, play

        traj = read_trajectory(a.traj)
        print(f"{a.traj}: {traj.t[-1]:.1f} s, speed cap {a.max_speed_deg:.0f} deg/s, worst IK error {traj.pos_error_m.max() * 1000:.1f} mm")
        if not a.yes and input("Clear the workspace. Type 'yes' to move the arm: ").strip().lower() != "yes":
            print("cancelled")
            return 1
        robot = connect(a.port)
        try:
            play(robot, traj, JointMapping(), np.radians(a.max_speed_deg))
        except KeyboardInterrupt:
            print("stopped by user")
        finally:
            # Note: LeRobot turns torque off on disconnect, so the arm goes limp. Support it.
            robot.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
