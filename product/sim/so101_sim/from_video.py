"""From a plain video of the operator to a checked training episode, in one go.

    video -> arm tracking (retarget.pose) -> hand demo -> IK -> replay in the twin -> episode file

The twin is set up from the demo: the block is put where your hand closed, the green zone
where it opened. Two robot motions are made and both are replayed:

  copy   the robot follows your hand path sample by sample. Shows how well the motion carries
         over. Often fails the grasp: a tracked hand is a centimetre or two off.
  steps  the demo is cut into steps (reach, grasp, move, release) and a clean robot motion is
         built from the grasp and release points. This is the one used for training by default.

Only a replay that succeeds in the twin is saved as an episode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import scene, scripted
from .episodes import save_npz
from .replay import ReplayResult, replay, save_video
from retarget.demo import HandDemo, write_trajectory  # noqa: E402  (import path set up by .replay)
from retarget.retarget import RetargetConfig, retarget  # noqa: E402

# Area on the table where the scripted pick-and-place was tested 20/20 (README, 8 Nov 2026).
TESTED_X = (0.18, 0.28)
TESTED_ABS_Y = (0.05, 0.15)


@dataclass
class VideoEpisode:
    part_xy: np.ndarray
    place_xy: np.ndarray
    copy: ReplayResult
    steps: ReplayResult
    copy_reached: float
    saved: Path | None
    notes: list[str]


def _in_tested_area(xy: np.ndarray) -> bool:
    return TESTED_X[0] <= xy[0] <= TESTED_X[1] and TESTED_ABS_Y[0] <= abs(xy[1]) <= TESTED_ABS_Y[1]


def next_episode_path(folder: Path) -> Path:
    """One past the highest number in the folder, so a deleted episode never gets another one overwritten."""
    used = [int(p.stem.split("_")[1]) for p in folder.glob("episode_*.npz") if p.stem.split("_")[1].isdigit()]
    return folder / f"episode_{max(used, default=-1) + 1:03d}.npz"


def demo_to_episode(demo: HandDemo, steps: list[dict], out_dir: Path, episodes_dir: Path, use: str = "steps", video: bool = True) -> VideoEpisode:
    """Replay a tracked demo in the twin both ways and save the chosen one if it succeeds."""
    from retarget.pose import pick_and_place_points

    points = pick_and_place_points(steps)
    if points is None:
        raise RuntimeError("The demo has no grasp followed by a release, so there is nothing to pick and place. Check tracked.mp4: does the pinch dot turn red when you grip?")
    part_xy, place_xy = scripted.part_centre(points[0]), scripted.part_centre(points[1])
    notes = []
    for name, xy in (("pick", part_xy), ("place", place_xy)):
        if not _in_tested_area(xy):
            notes.append(f"{name} point {np.round(xy, 3)} m is outside the area tested in the twin (x 0.18-0.28, |y| 0.05-0.15)")
    if np.linalg.norm(part_xy - place_xy) < 0.05:
        notes.append("pick and place are under 5 cm apart")

    chain = scripted._chain()
    cfg = RetargetConfig(q_start=scripted.home_q(), max_joint_speed_rad_s=np.radians(60))
    cam = "top"  # the same camera the scripted demos are recorded with, so the datasets match

    traj_copy, rep = retarget(demo, chain, cfg)
    write_trajectory(out_dir / "traj_copy.csv", traj_copy)
    res_copy = replay(traj_copy, *scene.load(part_pos=part_xy, place_pos=place_xy), camera=cam if video or use == "copy" else None)

    traj_steps, _ = retarget(scripted.pick_place_demo(part_xy, place_xy), chain, cfg)
    write_trajectory(out_dir / "traj_steps.csv", traj_steps)
    res_steps = replay(traj_steps, *scene.load(part_pos=part_xy, place_pos=place_xy), camera=cam if video or use == "steps" else None)

    if video:
        save_video(res_copy.episode["images"], out_dir / "replay_copy.mp4")
        save_video(res_steps.episode["images"], out_dir / "replay_steps.mp4")
    chosen = res_steps if use == "steps" else res_copy
    saved = None
    if chosen.success:
        saved = next_episode_path(episodes_dir)
        save_npz(chosen.episode, saved)
    (out_dir / "result.json").write_text(json.dumps({
        "part_xy": np.round(part_xy, 4).tolist(), "place_xy": np.round(place_xy, 4).tolist(),
        "copy": {"success": res_copy.success, "reached_fraction": round(rep.reached_fraction, 3), "place_err_mm": round(res_copy.place_err_mm, 1)},
        "steps": {"success": res_steps.success, "place_err_mm": round(res_steps.place_err_mm, 1)},
        "used": use, "episode": str(saved) if saved else None, "notes": notes,
    }, indent=2))  # fmt: skip
    return VideoEpisode(part_xy, place_xy, res_copy, res_steps, rep.reached_fraction, saved, notes)


def parse_grips(text: str | None) -> list[tuple[float, float]] | None:
    """'3.0-8.5' or '3.0-8.5,12-15': seconds where the hand holds the part (close-open)."""
    if not text:
        return None
    out = []
    for part in text.split(","):
        try:
            close, open_ = (float(v) for v in part.split("-"))
        except ValueError:
            raise ValueError(f"--grips '{part}': write seconds as GRASP-RELEASE, e.g. 3.0-8.5 (or 'object')") from None
        if not 0 <= close < open_:
            raise ValueError(f"--grips '{part}': the release must come after the grasp")
        out.append((close, open_))
    return out


def grips_from_object(video: Path, out_dir: Path, start: float | None, end: float | None) -> list[tuple[float, float]]:
    """Grasp and release times from the object's motion in the video (retarget.object_motion).
    For hands that hide the pinch, like a whole-hand grip on a carton. Raises NoObjectMotion."""
    from retarget.object_motion import debug_image, find_object_motion, read_small

    t0 = start or 0.0
    frames, fps, scale = read_small(video, t0, end)
    m = find_object_motion(frames, fps, scale)
    m.t_grasp += t0
    m.t_release += t0
    for note in m.notes:
        print("  note:", note)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_image(video, m, out_dir / "object_motion.jpg")
    print(f"  grasp at {m.t_grasp:.2f} s, release at {m.t_release:.2f} s (match {m.match:.2f}). Check {out_dir / 'object_motion.jpg'}")
    return [(m.t_grasp, m.t_release)]


def from_video(video: Path, out: Path, arm: str = "right", use: str = "steps", markers: Path | None = None, camera: Path | None = None,
               hand_width_mm: float | None = None, close_ratio: float = 0.5, open_ratio: float = 0.7,
               grips: str | None = None, start: float | None = None, end: float | None = None) -> int:  # fmt: skip
    from retarget import pose
    from retarget.video import CameraModel, MarkerSetup

    from retarget.object_motion import NoObjectMotion

    out_dir = out / video.stem
    try:
        if grips == "object":
            print("0/3 grasp and release from the object's motion")
            events = grips_from_object(video, out_dir, start, end)
        else:
            events = parse_grips(grips)
    except NoObjectMotion as e:
        print(f"  stopped: no object motion found ({e}). Give the times by hand: --grips 3.0-8.5")
        return 1
    except ValueError as e:
        print(f"  stopped: {e}")
        return 1
    cfg = pose.PoseConfig(arm=arm, hand_width_mm=hand_width_mm, close_ratio=close_ratio, open_ratio=open_ratio,
                          grip_events=events, t_start=start, t_end=end)  # fmt: skip
    print(f"1/3 tracking your {arm} arm in {video.name} (about as long as the video itself)")
    res, steps = pose.video_to_demo(video, out_dir, scripted._chain(), cfg,
                                    MarkerSetup.from_json(markers) if markers else None, CameraModel.from_json(camera) if camera else None)  # fmt: skip
    for w in res.warnings:
        print("  warning:", w)
    print(f"  {len(res.demo)} samples, {res.demo.t[-1]:.1f} s, mode: {res.mode}" + (f", your arm scaled x{res.scale:.2f}" if res.mode == "body" else ""))
    print("  steps: " + " > ".join(s["step"] for s in steps))

    print("2/3 robot motion (IK) and replay in the twin")
    try:
        ep = demo_to_episode(res.demo, steps, out_dir, out / "episodes", use)
    except RuntimeError as e:
        print("  stopped:", e)
        if grips is None:
            print("  if you gripped with the whole hand, try:  --grips object  (finds grasp and release from the object's motion)")
        return 1
    for n in ep.notes:
        print("  note:", n)
    print(f"  block at {np.round(ep.part_xy, 3)} m, place zone at {np.round(ep.place_xy, 3)} m (robot base frame)")
    print(f"  copy  (follows your hand): {ep.copy_reached:.0%} of the path reachable, task {'SUCCESS' if ep.copy.success else 'FAIL'}, block ends {ep.copy.place_err_mm:.0f} mm from the zone")
    print(f"  steps (rebuilt from grasp and release): task {'SUCCESS' if ep.steps.success else 'FAIL'}, block ends {ep.steps.place_err_mm:.0f} mm from the zone")

    print("3/3 training episode")
    if ep.saved:
        print(f"  saved {ep.saved} (from '{use}')")
        print(f"  when you have enough episodes:  python -m so101_sim to-lerobot {out / 'episodes'} USER/NAME --root {out / 'lerobot'}")
    else:
        print(f"  not saved: the '{use}' replay failed in the twin. Look at {out_dir / ('replay_' + use + '.mp4')}")
    print(f"Files to look at in {out_dir}: tracked.mp4 (was your arm found?), replay_copy.mp4, replay_steps.mp4")
    return 0 if ep.saved else 1
