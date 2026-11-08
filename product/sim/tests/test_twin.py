import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

mujoco = pytest.importorskip("mujoco")

from so101_sim import scene, scripted  # noqa: E402
from so101_sim.__main__ import check  # noqa: E402
from so101_sim.episodes import to_lerobot_units  # noqa: E402
from so101_sim.replay import replay  # noqa: E402
from retarget.retarget import RetargetConfig, retarget  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def model_files():
    try:
        scene.ensure_model()
    except OSError as e:  # no internet
        pytest.skip(f"SO-101 model not downloaded: {e}")


def test_twin_matches_retarget_kinematics():
    assert check() == 0


def test_scripted_pick_place_succeeds_in_twin():
    demo = scripted.pick_place_demo(scene.PART_POS, scene.PLACE_POS)
    traj, rep = retarget(demo, scripted._chain(), RetargetConfig(q_start=scripted.home_q()))
    assert rep.reached_fraction == 1.0
    model, data = scene.load()
    res = replay(traj, model, data)
    assert res.max_track_err_deg < 2.0
    assert res.success, res.report()


def test_lerobot_units():
    x = np.array([[0.0, np.pi / 2, 0.0, 0.0, 0.0, -0.17453], [0.0, 0.0, 0.0, 0.0, 0.0, 1.74533]])
    out = to_lerobot_units(x)
    assert out[0, 1] == pytest.approx(90.0)
    assert out[0, 5] == pytest.approx(0.0, abs=1e-3)  # closed
    assert out[1, 5] == pytest.approx(100.0, abs=1e-3)  # open
