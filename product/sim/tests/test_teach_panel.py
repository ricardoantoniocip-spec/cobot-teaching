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


class FakeBus:
    def __init__(self):
        self.torque = {j: True for j in (*scene.ARM_JOINTS, "gripper")}
        self.goals = {}

    def disable_torque(self, motors):
        for m in motors:
            self.torque[m] = False

    def enable_torque(self, motors):
        for m in motors:
            self.torque[m] = True

    def sync_write(self, name, values):
        assert name == "Goal_Position"
        self.goals.update(values)


class FakeRobot:
    """Joints in degrees like LeRobot's SO101Follower(use_degrees=True). Motors off: the hand sets `pos`."""

    def __init__(self, q_deg):
        self.bus = FakeBus()
        self.pos = {f"{j}.pos": float(v) for j, v in zip(scene.ARM_JOINTS, q_deg)} | {"gripper.pos": 40.0}
        self.sent = []

    def get_observation(self):
        return dict(self.pos)

    def send_action(self, action):
        self.sent.append(dict(action))
        for k, v in action.items():
            if self.bus.torque[k.removesuffix(".pos")]:
                self.pos[k] = v  # perfect servo
        return action


def test_freedrive_records_hand_guided_pose(tmp_path):
    from retarget.robot import JointMapping
    from so101_sim.teach_panel import RealArm, handle_command

    s = session(tmp_path)
    robot = FakeRobot(np.degrees(HOME))
    arm = RealArm(robot, JointMapping(), np.radians(30) / 30)
    handle_command(s, arm, "freedrive")
    assert arm.freedrive and s.teaching
    assert not any(robot.bus.torque[j] for j in scene.ARM_JOINTS) and robot.bus.torque["gripper"]
    n_sent = len(robot.sent)

    # move the arm by hand
    robot.pos.update({f"{j}.pos": float(v) for j, v in zip(scene.ARM_JOINTS, np.degrees(PICK))})
    arm.step(s)
    assert len(robot.sent) == n_sent  # no arm commands while in freedrive
    q, _ = s.target()
    np.testing.assert_allclose(q, PICK, atol=1e-9)  # twin copies the hand-guided pose
    handle_command(s, arm, "record")
    handle_command(s, arm, "close")
    arm.step(s)
    assert robot.bus.goals["gripper"] < 40  # gripper still driven
    handle_command(s, arm, "play")
    assert not s.playing  # not while in freedrive

    # leave freedrive: goal = where the arm is, then motors on; holds there
    handle_command(s, arm, "freedrive")
    assert not arm.freedrive and all(robot.bus.torque.values())
    last = robot.sent[-1]
    for j, v in zip(scene.ARM_JOINTS, np.degrees(PICK)):
        assert last[f"{j}.pos"] == pytest.approx(v)
    arm.step(s)
    for j, v in zip(scene.ARM_JOINTS, np.degrees(PICK)):
        assert robot.sent[-1][f"{j}.pos"] == pytest.approx(v)
    assert [st.do for st in s.path.steps] == ["movej", "gripper"]
    np.testing.assert_allclose(np.radians(s.path.steps[0].joints_deg), PICK, atol=1e-6)


def test_freedrive_needs_real_arm(tmp_path):
    from so101_sim.teach_panel import handle_command

    s = session(tmp_path)
    handle_command(s, None, "freedrive")
    assert "--port" in s.message


def test_goto_and_delete(tmp_path):
    from so101_sim.teach_panel import handle_command

    s = session(tmp_path)
    handle_command(s, None, "teach")
    s.record()
    s.set_target(PICK, 1.0)
    s.record()
    s.gripper(False)
    s.set_target(HOME, 0.0)
    handle_command(s, None, "goto", 1)
    while s.playing:
        s.tick()
    np.testing.assert_allclose(s.target()[0], PICK, atol=1e-9)
    handle_command(s, None, "goto", 2)  # a gripper step: refused
    assert not s.playing and "MoveJ" in s.message
    handle_command(s, None, "delete", 0)
    assert [st.do for st in s.path.steps] == ["movej", "gripper"]
