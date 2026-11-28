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
        self.robot = None
        self.load = 0.0

    def sync_read(self, name, motors):
        return {m: self.read(name, m) for m in motors}

    def read(self, name, motor):
        if name == "Present_Position":
            return self.robot.pos[f"{motor}.pos"]
        if name == "Present_Load":
            return self.load
        raise KeyError(name)

    def disable_torque(self, motors):
        for m in motors:
            self.torque[m] = False

    def enable_torque(self, motors):
        for m in motors:
            self.torque[m] = True

    def sync_write(self, name, values):
        assert name == "Goal_Position"
        self.goals.update(values)
        for m, v in values.items():
            if self.torque[m]:
                self.robot._set(f"{m}.pos", v)


class FakeRobot:
    """Joints in degrees like LeRobot's SO101Follower(use_degrees=True). Motors off: the hand sets `pos`."""

    def __init__(self, q_deg):
        self.bus = FakeBus()
        self.bus.robot = self
        self.pos = {f"{j}.pos": float(v) for j, v in zip(scene.ARM_JOINTS, q_deg)} | {"gripper.pos": 40.0}
        self.sent = []
        self.part_pct = None  # a part between the jaws stops the gripper here

        self.jaw_step = None  # a slow jaw: most it moves per command (gripper %)

    def _set(self, k, v):
        if k == "gripper.pos":
            if self.jaw_step is not None:
                cur = self.pos[k]
                v = cur + max(-self.jaw_step, min(self.jaw_step, v - cur))
            if self.part_pct is not None:
                v = max(v, self.part_pct)
        self.pos[k] = v

    def get_observation(self):
        return dict(self.pos)

    def send_action(self, action):
        self.sent.append(dict(action))
        for k, v in action.items():
            if self.bus.torque[k.removesuffix(".pos")]:
                self._set(k, v)  # perfect servo, except where a part blocks the jaws
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


def test_soft_grip_stops_at_the_part(tmp_path):
    from retarget.robot import JointMapping
    from so101_sim.teach_panel import RealArm, handle_command

    s = session(tmp_path)
    robot = FakeRobot(np.degrees(HOME))
    m = JointMapping()  # gripper 5 % closed .. 40 % open
    arm = RealArm(robot, m, np.radians(30) / 30)
    for _ in range(5):
        arm.step(s)
    robot.part_pct = 20.0  # part between the jaws at 20 % = (20-5)/35 = 0.43 open
    handle_command(s, arm, "close")
    for _ in range(90):
        arm.step(s)
    assert arm.gripped_at == pytest.approx(15 / 35, abs=0.02)
    assert arm.g_cmd == pytest.approx(arm.gripped_at - arm.grip_squeeze, abs=1e-6)  # squeezes a little, not to fully closed
    assert "gripped" in s.message
    sent_g = robot.sent[-1]["gripper.pos"]
    assert 5.0 < sent_g < 20.0

    handle_command(s, arm, "open")
    for _ in range(40):
        arm.step(s)
    assert arm.gripped_at is None and robot.pos["gripper.pos"] == pytest.approx(40.0)

    # no part: closes all the way
    robot.part_pct = None
    handle_command(s, arm, "close")
    for _ in range(90):
        arm.step(s)
    assert arm.gripped_at is None and robot.pos["gripper.pos"] == pytest.approx(5.0)


def test_soft_grip_by_motor_load(tmp_path):
    from retarget.robot import JointMapping
    from so101_sim.teach_panel import RealArm, handle_command

    s = session(tmp_path)
    robot = FakeRobot(np.degrees(HOME))
    arm = RealArm(robot, JointMapping(), np.radians(30) / 30)
    arm.step(s)
    handle_command(s, arm, "close")
    for _ in range(10):
        arm.step(s)
    robot.bus.load = 500  # load spike while the jaw still moves: not a part
    arm.step(s)
    assert arm.gripped_at is None
    robot.part_pct = robot.pos["gripper.pos"]  # soft part: jaw stops right at the command, motor works hard
    for _ in range(5):
        arm.step(s)
    assert arm.gripped_at is not None


def test_slow_jaw_is_not_a_part(tmp_path):
    from retarget.robot import JointMapping
    from so101_sim.teach_panel import RealArm, handle_command

    s = session(tmp_path)
    robot = FakeRobot(np.degrees(HOME))
    robot.jaw_step = 0.6  # % per tick, slower than the 0.8 s close command
    arm = RealArm(robot, JointMapping(), np.radians(30) / 30)
    arm.step(s)
    handle_command(s, arm, "close")
    for _ in range(120):
        arm.step(s)
    assert arm.gripped_at is None and robot.pos["gripper.pos"] == pytest.approx(5.0)


def test_program_waits_for_gripper_and_close_goes_to_rest(tmp_path, monkeypatch):
    from retarget.robot import JointMapping
    from so101_sim import teach_panel as tp

    monkeypatch.setattr(tp.time, "sleep", lambda s: None)
    s = session(tmp_path)
    robot = FakeRobot(np.degrees(HOME))
    arm = tp.RealArm(robot, JointMapping(), np.radians(30) / 30)
    arm.g_cmd = 1.0
    s.set_target(HOME, 0.0)  # twin says closed, real gripper still open
    arm.follow = True
    s.traj = object()  # pretend a program is running
    arm._gripper(s)
    assert arm.gripper_busy
    for _ in range(60):
        arm._gripper(s)
    assert not arm.gripper_busy
    s.traj = None

    robot.pos.update({f"{j}.pos": float(v) for j, v in zip(scene.ARM_JOINTS, np.degrees(PICK))})
    arm.go_rest()
    for j, v in zip(scene.ARM_JOINTS, np.degrees(HOME)):
        assert robot.pos[f"{j}.pos"] == pytest.approx(v)
    steps = [a["shoulder_pan.pos"] for a in robot.sent]
    assert max(abs(b - a) for a, b in zip(steps, steps[1:])) < 1.5  # small steps = slow


def test_jog(tmp_path):
    from so101_sim.teach_panel import handle_command

    s = session(tmp_path)
    handle_command(s, None, "jog", (0, np.radians(30)))
    handle_command(s, None, "jog", (5, 0.25))
    q, g = s.target()
    assert q[0] == pytest.approx(np.radians(30)) and g == pytest.approx(0.25)
    handle_command(s, None, "jog", (1, 99.0))  # clipped to the joint limit
    assert s.target()[0][1] == pytest.approx(s.chain.upper[1])


def test_set_box_at_gripper(tmp_path):
    from so101_sim.teach_panel import handle_command

    s = session(tmp_path)
    # gripper tip held just above the box (tool 60 mm up)
    handle_command(s, None, "box")
    tool = s.chain.fk(HOME)[:3, 3]
    np.testing.assert_allclose(s.data.joint("part").qpos[:2], tool[:2], atol=1e-9)
    for _ in range(200):  # it stands on the table and stays there
        mujoco.mj_step(s.model, s.data)
    assert s.data.joint("part").qpos[2] == pytest.approx(scene.PART_SIZE[2] / 2, abs=0.003)
    # saved with the program and restored on load
    handle_command(s, None, "teach")
    s.record()
    f = s.save("with_box")[0]
    s2 = session(tmp_path)
    s2.load(f)
    np.testing.assert_allclose(s2.data.joint("part").qpos[:2], tool[:2], atol=1e-4)
