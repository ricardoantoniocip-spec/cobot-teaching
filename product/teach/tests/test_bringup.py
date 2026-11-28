"""First power-up script against a stand-in robot (no hardware, no LeRobot)."""

import numpy as np

from retarget import bringup as bu
from retarget.kinematics import ARM_JOINTS, SerialChain
from retarget.robot import JointMapping


class FakeRobot:
    """Holds motor angles in degrees. wrong: joints wired so that they run the other way. dead: joints that do not move."""

    def __init__(self, wrong=(), dead=()):
        self.pos = {f"{j}.pos": 5.0 for j in ARM_JOINTS} | {"gripper.pos": 40.0}
        self.wrong, self.dead, self.grips = set(wrong), set(dead), []

    def get_observation(self):
        return dict(self.pos)

    def send_action(self, action):
        for k, v in action.items():
            if k.split(".")[0] not in self.dead:
                self.pos[k] = v
        self.grips.append(action["gripper.pos"])


def run(robot, answers, monkeypatch, tmp_path, mapping=None):
    monkeypatch.setattr(bu, "SPEED", 100.0)  # no waiting in tests
    monkeypatch.setattr("retarget.robot.time.sleep", lambda s: None)
    monkeypatch.setattr("retarget.robot.MAPPING_FILE", tmp_path / "so101_mapping.json")
    said, it = [], iter(answers)
    mapping = mapping or JointMapping()
    code = bu.bringup(robot, SerialChain.from_urdf(), ask=lambda prompt: next(it), say=said.append, mapping=mapping)
    return code, mapping, "\n".join(said)


def test_good_arm_passes_and_ends_at_the_first_pose(monkeypatch, tmp_path):
    robot = FakeRobot()
    code, mapping, text = run(robot, ["yes", *"yyyyy", "", "", "", "yes", ""], monkeypatch, tmp_path)
    assert code == 0 and all(s == 1.0 for s in mapping.signs.values())
    assert (tmp_path / "so101_mapping.json").exists()
    assert "180 mm in front" in text and "80 mm above" in text
    assert min(robot.grips) == mapping.gripper_closed_pct and max(robot.grips) == mapping.gripper_open_pct


def test_a_joint_that_runs_the_other_way_gets_its_sign_flipped(monkeypatch, tmp_path):
    code, mapping, text = run(FakeRobot(), ["yes", "y", "n", "y", "y", "y", "", "", ""], monkeypatch, tmp_path)
    assert code == 1 and mapping.signs["shoulder_lift"] == -1.0
    assert "Run bringup again" in text
    assert JointMapping.load(tmp_path / "so101_mapping.json").signs["shoulder_lift"] == -1.0


def test_a_dead_motor_is_reported(monkeypatch, tmp_path):
    code, _, text = run(FakeRobot(dead=["elbow_flex"]), ["yes", "y", "y", "x", "y", "y", "", "", ""], monkeypatch, tmp_path)
    assert code == 1 and "elbow_flex: asked +10 deg but the motor reports +0.0" in text


def test_direction_words():
    assert bu.direction_words(np.array([30.0, -12.0, 2.0])) == "30 mm forward and 12 mm right"
    assert "only turns" in bu.direction_words(np.zeros(3))
