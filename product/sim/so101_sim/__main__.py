"""SO-101 digital twin. Run from product/sim:  python -m so101_sim <command>

  fetch                         download the official SO-101 MuJoCo model (once, ~16 MB)
  check                         compare the twin with retarget's kinematics (should be < 0.1 mm)
  view                          3D viewer + Teach panel: jog with the sliders, record waypoints, o/c gripper, save, replay
                                (--port COM4: the real arm follows; --plain: viewer only)
  path NAME.path.json           show a saved teach path and write its trajectory CSV / task file again
  scripted-demo OUT.csv         write a scripted hand demo for the default part (input for `retarget convert`)
  replay TRAJ.csv               play a retarget trajectory in the twin and check the task result
  demos N                       make N varied pick-and-place demos, keep the successful ones as episodes
  from-video VIDEO              track your arm in a plain video, replay it in the twin, save a training episode
  live                          LIVE: your arm leads, the robot follows (twin, or the real arm with --port); shows all the tracking
  devices                       what this computer uses: graphics card, 3D view, camera
  to-lerobot DIR REPO_ID        turn saved episodes into a LeRobotDataset (needs lerobot installed)
  evaluate POLICY_DIR           run a trained policy (e.g. ACT) in the twin, N random trials, count successes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m so101_sim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    sub.add_parser("check")
    s = sub.add_parser("view")
    s.add_argument("--port", help="serial port of the real SO-101, e.g. COM4: the real arm follows the twin (SPACE = follow / pause)")
    s.add_argument("--robot-id", default="follower", help="the id used when the arm was calibrated in LeRobot")
    s.add_argument("--max-speed", type=float, default=60.0, help="fastest joint speed of the real arm in deg/s")
    s.add_argument("--path", type=Path, help="start with this saved teach path loaded (NAME.path.json)")
    s.add_argument("--paths-dir", type=Path, default=Path("paths"), help="where Save and Load look (default: paths)")
    s.add_argument("--speed", type=float, default=40.0, help="replay speed in deg/s (peak joint speed), changeable in the panel")
    s.add_argument("--plain", action="store_true", help="only the 3D viewer, no Teach panel (twin only)")
    s.add_argument("--viewer", action="store_true", help="also open MuJoCo's own 3D viewer window (old style)")
    s.add_argument("--grip-load", type=float, default=150.0, help="gripper motor load (0-1000) that counts as 'part felt' when closing")

    s = sub.add_parser("path")
    s.add_argument("file", type=Path, help="a saved NAME.path.json")
    s.add_argument("--speed", type=float, default=40.0, help="peak joint speed in deg/s for the CSV")

    s = sub.add_parser("scripted-demo")
    s.add_argument("out", type=Path)

    s = sub.add_parser("replay")
    s.add_argument("traj", type=Path)
    s.add_argument("--video", type=Path, help="save a video from the top camera, e.g. replay.mp4")
    s.add_argument("--camera", default="front", choices=["top", "front"])

    s = sub.add_parser("demos")
    s.add_argument("n", type=int)
    s.add_argument("--out", type=Path, default=Path("sim_data"))
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--video", action="store_true", help="also save one mp4 per episode")

    s = sub.add_parser("from-video")
    s.add_argument("video", type=Path)
    s.add_argument("--out", type=Path, default=Path("video_data"))
    s.add_argument("--arm", default="right", choices=["right", "left"], help="which of your arms does the task")
    s.add_argument("--use", default="steps", choices=["steps", "copy"], help="which robot motion becomes the training episode")
    s.add_argument("--markers", type=Path, help="markers.json: switch to table-marker mode (true scale)")
    s.add_argument("--camera", type=Path, help="camera.json with fx, fy, cx, cy, dist")
    s.add_argument("--hand-width-mm", type=float, help="your knuckle width, index to little finger (table-marker mode)")
    s.add_argument("--close-ratio", type=float, default=0.78, help="hand counts as closed below this openness (1 = flat hand, 0.45 = fist)")
    s.add_argument("--open-ratio", type=float, default=0.84, help="hand counts as open again above this openness")
    s.add_argument("--grips", help="set the grip by hand: seconds from grasp to release, e.g. 3.0-8.5 (times are shown on tracked.mp4), "
                        "or 'object' to find them from the object's motion")
    s.add_argument("--object", dest="obj", help="what you move, in plain words, e.g. \"juice carton\": an object detector then finds it, "
                        "follows it and reads grasp and release from it (needs requirements-vision.txt)")
    s.add_argument("--start", type=float, help="use the video from this second")
    s.add_argument("--end", type=float, help="use the video up to this second")

    sub.add_parser("devices")
    s = sub.add_parser("live")
    s.add_argument("--camera", type=int, default=0, help="webcam number (0 = the built-in one)")
    s.add_argument("--video", type=Path, help="use a recorded video instead of the webcam")
    s.add_argument("--arm", default="right", choices=["right", "left"], help="which of your arms leads")
    s.add_argument("--objects", nargs="?", const="", help="also find objects: no value = common things, or a name like \"juice carton\"")
    s.add_argument("--device", choices=["cuda", "cpu"], help="where the object detector runs (default: the graphics card if torch can use it)")
    s.add_argument("--port", help="serial port of the real SO-101, e.g. COM5. Without it only the twin moves")
    s.add_argument("--robot-id", default="follower", help="the id used when the arm was calibrated in LeRobot")
    s.add_argument("--max-speed", type=float, help="fastest joint speed in deg/s (default 90 in the twin, 45 on the real arm)")
    s.add_argument("--min-height", type=float, help="lowest tool point in mm (default 8 in the twin, 40 on the real arm)")
    s.add_argument("--mirror", action=argparse.BooleanOptionalAction, help="show the picture like a mirror (default: yes for a webcam)")
    s.add_argument("--no-window", action="store_true", help="no window (for tests)")
    s.add_argument("--no-twin", action="store_true", help="do not run the twin")
    s.add_argument("--save", type=Path, help="record the window to this mp4")
    s.add_argument("--log", type=Path, help="write every picture's numbers to this csv")
    s.add_argument("--seconds", type=float, help="stop after this many seconds")
    s.add_argument("--absolute", action="store_true", help="no clutch: your calibration hand is a fixed spot in the robot's work area")
    s.add_argument("--ik", choices=["exact", "numeric"], default="exact", help="exact closed-form SO-101 IK (default) or the general numeric solver")
    s.add_argument("--pose-model", choices=["lite", "full", "heavy"], default="full", help="body tracker size: lite is fastest, heavy most accurate")
    s.add_argument("--reach", choices=["elbow", "depth"], default="elbow", help="forward/back from how straight your arm is (default, camera in front) or from the tracker's depth (camera at your side)")
    s.add_argument("--no-wrist", action="store_true", help="do not use your wrist: gripper tilt from hand height only, roll fixed")
    s.add_argument("--roll-gain", type=float, default=1.0, help="gripper roll per degree of palm turn (-1 if it turns the wrong way)")
    s.add_argument("--flex-gain", type=float, default=1.0, help="gripper tilt per degree of hand tilt (-1 if it tilts the wrong way)")

    s = sub.add_parser("to-lerobot")
    s.add_argument("dir", type=Path, help="folder with episode_*.npz from `demos` or `from-video`")
    s.add_argument("repo_id", help="e.g. ricardocipriano/so101_sim_pick_place")
    s.add_argument("--root", type=Path, default=Path("sim_data/lerobot"))
    s.add_argument("--push", action="store_true", help="upload to the Hugging Face Hub as a private dataset")

    s = sub.add_parser("evaluate")
    s.add_argument("policy", type=Path, help="e.g. outputs/train/act_sim/checkpoints/last/pretrained_model")
    s.add_argument("--trials", type=int, default=10)
    s.add_argument("--max-s", type=float, default=25.0, help="seconds per trial")
    s.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    s.add_argument("--video-dir", type=Path)

    a = p.parse_args(argv)

    if a.cmd == "fetch":
        from .fetch import fetch

        fetch(force=True)
        return 0

    if a.cmd == "check":
        return check()

    if a.cmd == "view" and not a.plain:
        from .teach_panel import run

        return run(a.port, a.robot_id, a.max_speed, a.path, a.paths_dir, a.speed, viewer=a.viewer, grip_load=a.grip_load)

    if a.cmd == "path":
        from .teach_panel import export_path

        return export_path(a.file, a.speed)

    if a.cmd == "view":
        import mujoco.viewer

        from . import scene

        model, data = scene.load()
        scene.set_pose(model, data, np.zeros(5), 1.0)
        mujoco.viewer.launch(model, data)
        return 0

    if a.cmd == "scripted-demo":
        from . import scene, scripted
        from retarget.demo import write_demo  # import path set up by .scripted

        write_demo(a.out, scripted.pick_place_demo(scene.PART_POS, scene.PLACE_POS))
        print(f"wrote {a.out}")
        return 0

    if a.cmd == "replay":
        from .replay import replay_file, save_video

        res = replay_file(a.traj, camera=a.camera if a.video else None)
        print(res.report())
        if a.video:
            save_video(res.episode["images"], a.video)
            print(f"wrote {a.video}")
        return 0 if res.success else 1

    if a.cmd == "demos":
        return demos(a.n, a.out, a.seed, a.video)

    if a.cmd == "from-video":
        from .from_video import from_video

        return from_video(a.video, a.out, a.arm, a.use, a.markers, a.camera, a.hand_width_mm, a.close_ratio, a.open_ratio, a.grips, a.start, a.end, a.obj)

    if a.cmd == "devices":
        from .live import devices

        return devices()

    if a.cmd == "live":
        from .live import print_summary, run_live

        print_summary(run_live(a.video if a.video else a.camera, a.arm, a.objects, a.port, a.robot_id, a.max_speed, a.mirror, not a.no_window,
                               not a.no_twin, a.save, a.log, a.seconds, a.device, min_height_mm=a.min_height,
                               wrist=not a.no_wrist, roll_gain=a.roll_gain, flex_gain=a.flex_gain, reach=a.reach,
                               clutch=not a.absolute, ik=a.ik, pose_model=a.pose_model))
        return 0

    if a.cmd == "to-lerobot":
        from .episodes import npz_to_lerobot

        files = sorted(a.dir.glob("episode_*.npz"))
        if not files:
            print(f"No episode_*.npz in {a.dir}")
            return 1
        root = npz_to_lerobot(files, a.repo_id, a.root, push=a.push)
        print(f"LeRobotDataset with {len(files)} episodes in {root}")
        return 0
    if a.cmd == "evaluate":
        from .evaluate import evaluate

        evaluate(a.policy, a.trials, max_s=a.max_s, device=a.device, video_dir=a.video_dir)
        return 0
    return 1


def check() -> int:
    """Tool point from the MuJoCo twin vs retarget's URDF kinematics, at 200 random poses."""
    import mujoco

    from . import scene
    from .replay import SerialChain

    chain = SerialChain.from_urdf()
    model, data = scene.load()
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        q = rng.uniform(chain.lower, chain.upper)
        data.qpos[: len(q)] = 0
        for name, v in zip(scene.ARM_JOINTS, q):
            data.joint(name).qpos[0] = v
        mujoco.mj_kinematics(model, data)
        worst = max(worst, float(np.linalg.norm(data.site(scene.TCP_SITE).xpos - chain.fk(q)[:3, 3])))
    ok = worst < 1e-4
    print(f"twin vs retarget kinematics, worst tool point difference: {worst * 1000:.4f} mm -> {'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


def demos(n: int, out: Path, seed: int, video: bool) -> int:
    """Random part and place positions -> scripted demo -> retarget -> twin replay -> keep successes."""
    from . import scene, scripted
    from .episodes import save_npz
    from .replay import replay, save_video
    from retarget.retarget import RetargetConfig, retarget  # import path set up by .replay

    rng = np.random.default_rng(seed)
    chain = scripted._chain()
    kept = 0
    for i in range(n):
        part = np.array([rng.uniform(0.18, 0.28), rng.uniform(0.05, 0.15)])
        place = np.array([rng.uniform(0.18, 0.28), rng.uniform(-0.15, -0.05)])
        demo = scripted.pick_place_demo(part, place)
        # 60 deg/s: faster than the 30 deg/s used for a first replay on the real arm, to keep episodes short.
        cfg = RetargetConfig(q_start=scripted.home_q(), max_joint_speed_rad_s=np.radians(60))
        traj, rep = retarget(demo, chain, cfg)
        model, data = scene.load(part_pos=part, place_pos=place)
        res = replay(traj, model, data, camera="top")
        status = "kept" if res.success else "FAILED, dropped"
        print(f"episode {i}: part {np.round(part, 3)} -> {np.round(place, 3)}, {len(res.episode['t'])} frames, {status}")
        if res.success:
            save_npz(res.episode, out / f"episode_{kept:03d}.npz")
            if video:
                save_video(res.episode["images"], out / f"episode_{kept:03d}.mp4")
            kept += 1
    print(f"{kept}/{n} demos succeeded in the twin, saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
