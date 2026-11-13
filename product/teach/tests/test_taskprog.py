"""Tests for taskprog: task file rules, demo -> task, plan, URScript, UR and SO-101 kinematics."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from taskprog.check import check, passed
from taskprog.from_steps import steps_to_task
from taskprog.label import LABELS_SCHEMA, frame_times
from taskprog.plan import make_plan
from taskprog.task import TaskError, load_cell, parse_task
from taskprog.ur import URKinematics, matrix_to_rotvec, reach_check, rotvec_to_matrix, urscript

HERE = Path(__file__).resolve().parents[1] / "taskprog"
TAKE1 = HERE / "examples" / "take1"
UR = load_cell(HERE / "cells" / "ur5e_table.json")
SO = load_cell(HERE / "cells" / "so101_twin.json")


def take1_task() -> dict:
    return json.loads((TAKE1 / "task.json").read_text())


def test_take1_task_matches_what_from_steps_makes():
    steps = json.loads((TAKE1 / "steps.json").read_text())
    labels = json.loads((TAKE1 / "labels.json").read_text())
    assert steps_to_task(steps, labels, flat_table=True) == take1_task()


def test_take1_locations_come_from_the_demo_not_the_llm():
    t = take1_task()
    assert t["locations"]["armrest"]["xyz_mm"] == [176.3, 46.1, 0.0]  # steps.json grasp point, m -> mm
    assert t["locations"]["sofa_seat"]["xyz_mm"] == [222.7, -72.9, 0.0]
    assert [s["do"] for s in t["steps"]] == ["home", "pick", "place", "home"]


def test_bad_llm_names_fall_back_and_ask_for_review():
    steps = json.loads((TAKE1 / "steps.json").read_text())
    labels = json.loads((TAKE1 / "labels.json").read_text())
    labels["task_name"] = "Move Carton!"
    labels["steps"][1]["location"] = "def"  # reserved word
    t = steps_to_task(steps, labels, flat_table=True)
    assert t["name"] == "demo_task" and "pick_1" in t["locations"]
    assert any("not valid" in r for r in t["review"]) and any("invalid" in r for r in t["review"])


def test_no_labels_still_gives_a_valid_task():
    steps = json.loads((TAKE1 / "steps.json").read_text())
    t = steps_to_task(steps, None)
    parse_task(t)
    assert set(t["locations"]) == {"pick_1", "place_1"}
    assert t["locations"]["pick_1"]["xyz_mm"][2] == pytest.approx(81.0 - 20.0)  # surface = grasp height - grasp offset


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda t: t["steps"][1].update(do="weld"), "'do' must be one of"),
        (lambda t: t["steps"][1].update(force=10), "unknown key 'force'"),
        (lambda t: t["steps"].pop(1), "place juice_carton but the gripper holds nothing"),
        (lambda t: t["steps"].pop(2), "ends while still holding"),
        (lambda t: t["locations"]["armrest"].update(xyz_mm=[1, 2]), "list of 3 numbers"),
        (lambda t: t.update(name="move carton"), "not a valid name"),
        (lambda t: t["steps"][1].update(at="kitchen"), "location 'kitchen' is not in locations"),
        (lambda t: t["defaults"].update(approach_mm=5), "outside 10"),
    ],
)
def test_invalid_tasks_are_rejected(edit, message):
    t = take1_task()
    edit(t)
    with pytest.raises(TaskError, match=message):
        parse_task(t, set(UR.speeds))


def test_unknown_speed_is_rejected_against_the_cell():
    t = take1_task()
    t["steps"][1]["speed"] = "turbo"
    with pytest.raises(TaskError, match="turbo"):
        parse_task(t, set(UR.speeds))


def test_cell_speed_caps():
    from taskprog.task import parse_cell

    c = json.loads((HERE / "cells" / "ur5e_table.json").read_text())
    c["speeds"]["normal"]["linear_mm_s"] = 2000
    with pytest.raises(TaskError, match="linear_mm_s"):
        parse_cell(c)


def test_pick_expands_to_approach_down_close_up():
    task = parse_task(take1_task())
    plan = make_plan(task, UR)
    pick = [m for m in plan if m.step == 2]
    assert [m.kind for m in pick] == ["ptp", "line", "grip", "line"]
    assert [round(m.height_m, 3) for m in pick if m.kind != "grip"] == [0.07, 0.02, 0.07]
    assert pick[2].close is True
    place = [m for m in plan if m.step == 3]
    assert place[1].height_m == pytest.approx(0.022)  # grasp height + release gap
    assert place[2].close is False


def test_urscript_matches_the_reviewed_example():
    task = parse_task(take1_task(), set(UR.speeds))
    text = (TAKE1 / "task.json").read_text()
    assert urscript(task, UR, text, "task.json") == (TAKE1 / "move_carton_to_seat.script").read_text()


def test_labels_cannot_inject_code():
    t = take1_task()
    t["steps"][1]["label"] = 'grip it\nset_tool_digital_out(1, True)\n# "'
    t["locations"]["armrest"]["note"] = "x\r\nmovej([0,0,0,0,0,0])"
    prog = urscript(parse_task(t), UR)
    code = [ln.split("#")[0] for ln in prog.splitlines()]
    assert not any("set_tool_digital_out(1" in c for c in code)
    assert not any("movej([0,0,0" in c for c in code)


def test_ur5e_fk_matches_ursim_home_pose():
    # URSim 5.26 (UR5e) reported p[-0.4919, -0.1333, 0.3379, 2.22144, 2.22144, 0] at home with a 150 mm TCP.
    kin = URKinematics("ur5e", (0, 0, 150))
    t = kin.fk(np.radians([0, -90, 90, -90, -90, 0]))
    assert t[:3, 3] == pytest.approx([-0.4919, -0.1333, 0.3379], abs=1e-4)
    assert matrix_to_rotvec(t[:3, :3]) == pytest.approx([2.22144, 2.22144, 0], abs=1e-4)


def test_rotvec_round_trip_at_180_degrees():
    r = rotvec_to_matrix(np.array([np.pi, 0, 0]))
    assert rotvec_to_matrix(matrix_to_rotvec(r)) == pytest.approx(r, abs=1e-9)
    r2 = rotvec_to_matrix(np.array([2.22144, 2.22144, 0]))
    assert rotvec_to_matrix(matrix_to_rotvec(r2)) == pytest.approx(r2, abs=1e-6)


def test_ur_ik_round_trip():
    kin = URKinematics("ur5e", (0, 0, 150))
    q_true = np.radians([10, -80, 100, -110, -90, 20])
    q, pe, oe = kin.ik(kin.fk(q_true), np.radians([0, -90, 90, -90, -90, 0]))
    assert pe < 1e-5 and oe < 1e-4


def test_take1_passes_checks_on_both_cells():
    task = parse_task(take1_task())
    for cell in (UR, SO):
        f = check(task, cell)
        assert passed(f), [str(x) for x in f]


def test_far_location_is_caught_before_the_robot_sees_it():
    t = take1_task()
    t["locations"]["armrest"]["xyz_mm"] = [900, 0, 0]  # 1.15 m from the UR5e base: out of reach
    findings = check(parse_task(t), UR)
    assert not passed(findings)
    assert any("reach" in f.text for f in findings)
    assert not passed(check(parse_task(t), SO))


def test_ur_reach_rows_line_moves_are_straight():
    task = parse_task(take1_task())
    rows = reach_check(make_plan(task, UR), UR)
    assert all(r.problem is None for r in rows)
    assert max(r.pos_err_mm for r in rows) < 0.01


def test_so101_trajectory_reaches_every_target():
    from taskprog.so101 import trajectory

    traj, rows = trajectory(parse_task(take1_task()), SO)
    assert max(e for _, e in rows) < 0.005
    assert traj.gripper_open[0] == 1.0 and traj.gripper_open[-1] == 1.0 and traj.gripper_open.min() == 0.0
    assert np.all(np.diff(traj.t) > 0)


def test_twin_runs_the_take1_task():
    pytest.importorskip("mujoco")
    from taskprog.twin import run_twin

    try:
        _, res = run_twin(parse_task(take1_task()), SO)
    except OSError as e:  # first run downloads the arm model
        pytest.skip(f"twin model not available: {e}")
    assert res.success, res.report()


def test_labels_schema_is_strict():
    def walk(s):
        if s.get("type") == "object":
            assert s["additionalProperties"] is False
            assert set(s["required"]) == set(s["properties"])
            for v in s["properties"].values():
                walk(v)
        if s.get("type") == "array":
            walk(s["items"])

    walk(LABELS_SCHEMA)
    jsonschema = pytest.importorskip("jsonschema")
    labels = json.loads((TAKE1 / "labels.json").read_text())
    labels.pop("_labelled_by")
    jsonschema.validate(labels, LABELS_SCHEMA)


def test_frame_times_three_per_step_one_for_instant_steps():
    steps = json.loads((TAKE1 / "steps.json").read_text())["steps"]
    ft = frame_times(steps)
    assert [i for i, _ in ft].count(0) == 3 and [i for i, _ in ft].count(1) == 1


def test_label_call_shape_with_a_fake_client(monkeypatch):
    """No API key in CI: check the request we send and how the answer is read."""
    anthropic = pytest.importorskip("anthropic")
    import taskprog.label as lab

    sent = {}
    answer = json.loads((TAKE1 / "labels.json").read_text())
    answer.pop("_labelled_by")

    class Block:
        type = "text"
        text = json.dumps(answer)

    class Resp:
        stop_reason = "end_turn"
        content = [Block()]
        model = "claude-opus-5-5"

    class Fake:
        def __init__(self, *a, **k):
            self.beta = self
            self.messages = self

        def create(self, **kw):
            sent.update(kw)
            return Resp()

    monkeypatch.setattr(anthropic, "Anthropic", Fake)
    monkeypatch.setattr(lab, "grab_frames", lambda video, times: [(i, t, b"jpg") for i, t in times])
    steps = json.loads((TAKE1 / "steps.json").read_text())
    labels = lab.label(Path("take1.mp4"), steps)
    assert labels["task_name"] == "move_carton_to_seat" and labels["_labelled_by"] == "claude-opus-5-5"
    assert sent["output_config"]["format"]["schema"] is LABELS_SCHEMA
    assert sum(c["type"] == "image" for c in sent["messages"][0]["content"]) == 11
