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
from taskprog.task import TaskError, load_cell, parse_cell, parse_task
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


def test_work_frame_behind_the_home_pose_is_flagged():
    # Found on the first URSim recording: with home facing -x and the table at +x the arm
    # reached over backwards. The program still "passed", so the check now warns.
    import dataclasses

    backwards = dataclasses.replace(UR, home_joints_deg=(0.0, -90.0, 90.0, -90.0, -90.0, 0.0))
    findings = check(parse_task(take1_task()), backwards)
    assert any("odd arm posture" in f.text for f in findings)
    assert not any("odd arm posture" in f.text for f in check(parse_task(take1_task()), UR))


# ---------------------------------------------------------------- 0.2: grids, signals, repeat

EX = HERE / "examples"


def _example(name: str) -> dict:
    return json.loads((EX / name / "task.json").read_text())


def _cell(name: str):
    return parse_cell(json.loads((HERE / "cells" / f"{name}.json").read_text()))


def test_grid_slots_fill_along_x_then_y_then_up():
    t = _example("kitting_tray")
    t["locations"]["tray"]["grid"] = {"count": [3, 2], "pitch_mm": [60, 50], "layers": 2, "layer_height_mm": 40}
    t["locations"]["tray"]["yaw_deg"] = 90
    tray = parse_task(t).locations["tray"]
    assert tray.slot(1).xyz_mm == pytest.approx((150, 100, 0))
    assert tray.slot(3).xyz_mm == pytest.approx((150, 220, 0))  # x of the tray points along +y at yaw 90
    assert tray.slot(4).xyz_mm == pytest.approx((100, 100, 0))
    assert tray.slot(12).xyz_mm == pytest.approx((100, 220, 40))


def test_repeat_is_written_out_and_next_counts_slots():
    task = parse_task(_example("kitting_tray"))
    assert [s.do for s in task.steps] == ["home"] + ["pick", "place"] * 6 + ["home"]
    assert [s.slot for s in task.steps if s.do == "place"] == [1, 2, 3, 4, 5, 6]
    assert [s.id for s in task.steps] == list(range(1, 15))


@pytest.mark.parametrize("edit, message", [
    (lambda t: t["steps"][1].update(times=7), "outside 1..6"),
    (lambda t: t["steps"][1]["steps"][1].update(slot=2), "already filled"),
    (lambda t: t["steps"][1]["steps"][1].pop("slot"), "say which slot"),
    (lambda t: t["steps"][1]["steps"][0].update(slot=1), "has no grid"),
    (lambda t: t["locations"].update(tray_s2={"xyz_mm": [0, 0, 0]}), "clashes"),
    (lambda t: t["locations"]["tray"]["grid"].update(pitch_mm=[0, 60]), "pitch cannot be 0"),
    (lambda t: t["steps"][1]["steps"].append({"do": "repeat", "times": 2, "steps": [{"do": "home"}]}), "inside a repeat"),
])
def test_invalid_grids_and_repeats_are_rejected(edit, message):
    t = _example("kitting_tray")
    edit(t)
    with pytest.raises(TaskError, match=message):
        parse_task(t)


def test_urscript_slot_poses_match_the_plan():
    """The script derives slots from the taught slot 1; the plan computes them in the work frame.
    Both must give the same pose (URSim checks the same thing on the controller)."""
    from taskprog.ur import rotvec_to_matrix

    t = _example("kitting_tray")
    t["locations"]["tray"]["yaw_deg"] = 30
    task = parse_task(t, set(UR.speeds))
    script = urscript(task, UR)

    def mat(nums):
        m = np.eye(4)
        m[:3, 3], m[:3, :3] = nums[:3], rotvec_to_matrix(np.array(nums[3:]))
        return m

    def nums(line):
        return [float(v) for v in line.split("p[")[1].split("]")[0].split(",")]

    lines = {ln.split("=")[0].strip(): ln for ln in script.splitlines() if ln.strip().startswith(("loc_", "work ="))}
    base = mat(nums(lines["work"])) @ mat(nums(lines["loc_tray"]))
    planned = {m.location: m.pose for m in make_plan(task, UR) if m.kind == "line" and m.height_m < 0.03}
    for slot in range(1, 7):
        got = base @ mat(nums(lines[f"loc_tray_s{slot}"]))
        want = planned[f"tray_s{slot}"]
        assert got[:3, 3] == pytest.approx(want[:3, 3] + [0, 0, -0.027], abs=1e-5)  # grasp 25 + gap 2 mm


def test_example_programs_match_the_reviewed_files():
    for name, cell, script in (("kitting_tray", UR, "fill_kit_tray.script"),
                               ("machine_tending", _cell("ur5e_tending"), "tend_press.script")):
        text = (EX / name / "task.json").read_text()
        task = parse_task(json.loads(text), set(cell.speeds))
        assert passed(check(task, cell))
        assert urscript(task, cell, text, "task.json") == (EX / name / script).read_text()


def test_signals_need_a_pin_in_the_cell_and_the_right_direction():
    task = parse_task(_example("machine_tending"))
    assert any(f.level == "ERROR" and "cycle_done" in f.text for f in check(task, UR))  # ur5e_table has no signals
    wrong = json.loads((HERE / "cells" / "ur5e_tending.json").read_text())
    wrong["signals"]["cycle_start"] = {"io": "standard_in", "pin": 2}
    assert any(f.level == "ERROR" and "cycle_start" in f.text for f in check(task, parse_cell(wrong)))
    wrong["signals"]["cycle_start"] = {"io": "tool_out", "pin": 0}
    assert any("gripper's output pin" in f.text for f in check(task, parse_cell(wrong)))


@pytest.mark.parametrize("edit, message", [
    (lambda t: t["signals"].pop("cycle_done"), "not an input"),
    (lambda t: t["signals"].update(cycle_start="input"), "not an output"),
    (lambda t: t["steps"][1]["steps"][4].pop("timeout_s"), "missing 'timeout_s'"),
    (lambda t: t["steps"][1]["steps"][3].update(value="yes"), "true or false"),
])
def test_invalid_signal_steps_are_rejected(edit, message):
    t = _example("machine_tending")
    edit(t)
    with pytest.raises(TaskError, match=message):
        parse_task(t)


def test_wait_input_has_a_timeout_that_stops_the_program():
    task = parse_task(_example("machine_tending"))
    script = urscript(task, _cell("ur5e_tending"))
    assert "while (not get_standard_digital_in(1)) and (ct_waited_s < 30):" in script
    assert 'textmsg("CT ERROR step 6: cycle_done did not turn on within 30 s")\n    halt' in script
    assert "SIMULATION" not in script


def test_pretend_machine_only_in_a_simulation_cell():
    task = parse_task(_example("machine_tending"))
    sim = _cell("ur5e_tending_sim")
    script = urscript(task, sim)
    assert "thread ct_sim_machine_0():" in script and "Not for a real robot" in script
    assert any("URSim only" in f.text for f in check(task, sim))
    bad = json.loads((HERE / "cells" / "ur5e_tending.json").read_text())
    bad["simulate"] = [{"when": "cycle_start", "then": "cycle_done", "after_s": 2}]
    with pytest.raises(TaskError, match="must be mapped to an output"):
        parse_cell(bad)


def test_old_0_1_files_still_load():
    t = take1_task()
    t["format"] = "cobot-task/0.1"
    parse_task(t)
    cell = json.loads((HERE / "cells" / "ur5e_table.json").read_text())
    assert cell["format"] == "cobot-cell/0.1"
    parse_cell(cell)
