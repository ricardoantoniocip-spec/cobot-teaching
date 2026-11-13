"""Tracked operator (made-up landmarks) -> twin replay -> episode file. No video or MediaPipe needed."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

mujoco = pytest.importorskip("mujoco")

from so101_sim import scene, scripted  # noqa: E402
from so101_sim.from_video import demo_to_episode  # noqa: E402
from retarget import pose  # noqa: E402
from retarget.examples import synthetic_operator_frames  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def model_files():
    try:
        scene.ensure_model()
    except OSError as e:  # no internet
        pytest.skip(f"SO-101 model not downloaded: {e}")


def test_tracked_demo_becomes_a_successful_episode(tmp_path):
    res = pose.frames_to_demo(synthetic_operator_frames(noise_m=0.002), scripted._chain())
    steps = pose.segment_steps(res.demo)
    ep = demo_to_episode(res.demo, steps, tmp_path, tmp_path / "episodes", use="steps", video=False)
    assert ep.steps.success, ep.steps.report()
    assert ep.saved is not None and ep.saved.name == "episode_000.npz"
    data = np.load(ep.saved)
    assert data["state"].shape[1] == 6 and data["images"].shape[1:] == (240, 320, 3)
    print("copy:", ep.copy.report(), ep.copy_reached, ep.notes)


def test_demo_without_a_grasp_is_refused(tmp_path):
    frames = synthetic_operator_frames()
    for f in frames:  # hand stays open the whole time
        f.hand_world = frames[0].hand_world
    res = pose.frames_to_demo(frames, scripted._chain())
    with pytest.raises(RuntimeError, match="no grasp"):
        demo_to_episode(res.demo, pose.segment_steps(res.demo), tmp_path, tmp_path / "episodes")


def test_grip_times_are_checked():
    from so101_sim.from_video import parse_grips

    assert parse_grips("3-8.5,12-15") == [(3.0, 8.5), (12.0, 15.0)]
    assert parse_grips(None) is None
    for bad in ("8-3", "abc", "3", "3-4-5"):
        with pytest.raises(ValueError, match="--grips"):
            parse_grips(bad)


def test_next_episode_never_overwrites(tmp_path):
    from so101_sim.from_video import next_episode_path

    assert next_episode_path(tmp_path).name == "episode_000.npz"
    for n in (0, 2):
        (tmp_path / f"episode_{n:03d}.npz").write_bytes(b"")
    assert next_episode_path(tmp_path).name == "episode_003.npz"


def test_grips_from_the_objects_motion(tmp_path):
    """A made-up video: a hand takes a printed box at 2 s and sets it down at 4 s."""
    cv2 = pytest.importorskip("cv2")
    from so101_sim.from_video import grips_from_object

    rng = np.random.default_rng(0)
    bg = np.full((360, 640, 3), 120, np.uint8) + rng.integers(0, 12, (360, 640, 3), dtype=np.uint8)
    box = cv2.resize(rng.integers(0, 255, (18, 10, 3), dtype=np.uint8), (40, 72), interpolation=cv2.INTER_NEAREST)
    box[:, :, 2] = np.clip(box[:, :, 2].astype(int) + 80, 0, 255)
    video = tmp_path / "demo.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 360))
    for i in range(180):
        t = i / 30
        a = min(max((t - 2.0) / 2.0, 0.0), 1.0)
        x, y = int(120 + 320 * a), int(120 + 100 * a)
        img = bg.copy()
        img[y:y + 72, x:x + 40] = box
        if 1.0 < t < 5.0:  # the hand covers the top third while it holds the box
            img[y - 12:y + 24, x - 8:x + 48] = (150, 170, 210)
        w.write(img)
    w.release()
    (t_grasp, t_release), = grips_from_object(video, tmp_path / "out", None, None)
    assert t_grasp == pytest.approx(2.0, abs=0.4) and t_release == pytest.approx(4.0, abs=0.4)
    assert (tmp_path / "out" / "object_motion.jpg").exists()
