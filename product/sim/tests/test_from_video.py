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
