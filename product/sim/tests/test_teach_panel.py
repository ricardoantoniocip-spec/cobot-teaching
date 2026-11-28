import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
mujoco = pytest.importorskip("mujoco")

from so101_sim import scene  # noqa: E402
from so101_sim.teach_panel import TeachSession, export_path  # noqa: E402

if not (scene.MODEL_DIR / "so101_new_calib.xml").exists():
    pytest.skip("run `python -m so101_sim fetch` first", allow_module_level=True)

HOME = np.radians([0, -21.782, 29.215, 82.567, 0])
PICK = np.radians([25, 10, 20, 70, 0])


def session(tmp_path):
    model, data = scene.load()
    scene.set_pose(model, data, HOME, 1.0)
    return TeachSession(model, data, tmp_path / "paths")


def test_teach_save_load_play(tmp_path):
    s = session(tmp_path)
    s.record()
    assert s.path.n_waypoints == 0  # Teach is off
    s.toggle_teach()
    s.record()
    s.set_target(PICK, 1.0)  # what dragging the sliders does
    s.record()
    s.gripper(False)  # key c
    s.set_target(HOME, 0.0)
    s.record()
    s.gripper(True)  # key o
    assert [st.do for st in s.path.steps] == ["movej", "movej", "gripper", "movej", "gripper"]

    files = s.save("Test path")
    assert [f.name for f in files] == ["test_path.path.json", "test_path.csv", "test_path.task.json"]

    s2 = session(tmp_path)
    s2.load(files[0])
    assert s2.path.n_waypoints == 3
    assert s2.play(speed_deg_s=60)
    seen_pick = False
    for _ in range(len(s2.traj.t) + 1):
        s2.tick()
        for _ in range(16):
            mujoco.mj_step(s2.model, s2.data)
        q, _ = s2.target()
        seen_pick |= np.abs(q - PICK).max() < 1e-6
    assert not s2.playing and seen_pick
    q, g = s2.target()
    np.testing.assert_allclose(q, HOME, atol=1e-9)
    assert g == pytest.approx(1.0)
    # the twin's real joints followed the commands
    q_real = np.array([s2.data.joint(j).qpos[0] for j in scene.ARM_JOINTS])
    assert np.degrees(np.abs(q_real - HOME)).max() < 5


def test_stop_and_undo(tmp_path):
    s = session(tmp_path)
    s.toggle_teach()
    s.record()
    s.set_target(PICK, 1.0)
    s.record()
    s.undo()
    assert s.path.n_waypoints == 1
    s.set_target(PICK, 1.0)
    s.play()
    s.tick()
    s.stop()
    assert not s.playing
    assert s.save("x") and not s.teaching  # play turns teaching off


def test_export_command(tmp_path, capsys):
    s = session(tmp_path)
    s.toggle_teach()
    s.record()
    s.set_target(PICK, 1.0)
    s.record()
    f = s.save("only_moves")[0]
    (tmp_path / "paths" / "only_moves.csv").unlink()
    assert export_path(f, 15) == 0
    assert (tmp_path / "paths" / "only_moves.csv").exists()
    assert "no task file" in capsys.readouterr().out
