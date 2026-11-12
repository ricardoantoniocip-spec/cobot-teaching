"""SO-101 digital twin. Run from product/sim:  python -m so101_sim <command>

  fetch                         download the official SO-101 MuJoCo model (once, ~16 MB)
  check                         compare the twin with retarget's kinematics (should be < 0.1 mm)
  view                          open the 3D viewer; drag the actuator sliders to move the arm
  scripted-demo OUT.csv         write a scripted hand demo for the default part (input for `retarget convert`)
  replay TRAJ.csv               play a retarget trajectory in the twin and check the task result
  demos N                       make N varied pick-and-place demos, keep the successful ones as episodes
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
    sub.add_parser("view")

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

    s = sub.add_parser("to-lerobot")
    s.add_argument("dir", type=Path, help="folder with episode_*.npz from `demos`")
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
