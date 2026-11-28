from retarget.grip_test import grip_test, suggest, CloseResult
from retarget.robot import JointMapping


class Bus:
    """Gripper servo stand-in: follows the goal, a part blocks it at part_pct and the load then rises with the squeeze."""

    def __init__(self):
        self.goal = self.pos = 40.0
        self.part_pct = None
        self.free_load = 60.0

    def write(self, name, motor, v):
        assert name == "Goal_Position" and motor == "gripper"
        self.goal = v
        self.pos = v if self.part_pct is None else max(v, self.part_pct)

    def read(self, name, motor):
        if name == "Present_Position":
            return self.pos
        squeeze = max(0.0, self.pos - self.goal)
        return self.free_load + 120 * squeeze


class Robot:
    def __init__(self):
        self.bus = Bus()


def test_grip_test_finds_threshold_and_never_exceeds_limit(tmp_path):
    robot = Robot()
    answers = iter(["", "", ""])
    bus = robot.bus

    def ask(prompt):
        bus.part_pct = None if prompt.startswith("A") else (22.0 if prompt.startswith("B") else 25.0)
        return next(answers)

    out = []
    grip_test(robot, JointMapping(), limit=400, ask=ask, sleep=lambda s: None, log=out.append, out=tmp_path / "g.csv")
    text = "\n".join(out)
    assert "--grip-load" in text
    assert (tmp_path / "g.csv").exists()
    rows = (tmp_path / "g.csv").read_text().splitlines()[1:]
    assert max(float(r.split(",")[-1]) for r in rows) < 400 + 120  # stops right after crossing, never keeps squeezing
    assert bus.goal == JointMapping().gripper_open_pct  # left open


def test_skip_and_suggest():
    empty = CloseResult("A_empty", "closed", 5, 60)
    assert suggest(empty, []) is None
    assert suggest(empty, [CloseResult("B_soft", "load", 20, 300)]) == 200
    assert suggest(empty, [CloseResult("B_soft", "blocked", 20, 80)]) is None
