import json

import numpy as np
import pytest

from retarget.demo import read_trajectory, write_trajectory
from retarget.kinematics import SerialChain
from retarget.teach import TeachPath, safe_name, to_task, to_trajectory
from taskprog.check import check
from taskprog.task import load_cell, parse_task

HOME = np.radians([0, -21.782, 29.215, 82.567, 0])
PICK = np.radians([25, 10, 20, 70, 0])
PLACE = np.radians([-25, 10, 20, 70, 0])


@pytest.fixture(scope="module")
def chain():
    return SerialChain.from_urdf()


def pick_place(chain) -> TeachPath:
    p = TeachPath("Box to tray")
    p.add_waypoint(HOME, 1.0, chain)
    p.add_waypoint(PICK, 1.0, chain)
    p.add_gripper(False)
    p.add_waypoint(HOME, 0.0, chain)
    p.add_waypoint(PLACE, 0.0, chain)
    p.add_gripper(True)
    p.add_waypoint(HOME, 1.0, chain)
    return p


def test_save_load_roundtrip(tmp_path, chain):
    p = pick_place(chain)
    f = p.save(tmp_path / "box.path.json")
    q = TeachPath.load(f)
    assert q.name == "Box to tray" and len(q.steps) == 7 and q.n_waypoints == 5
    np.testing.assert_allclose(np.radians(q.steps[1].joints_deg), PICK, atol=1e-4)
    assert q.steps[2].do == "gripper" and q.steps[2].gripper == 0.0
    assert "MoveJ" in q.lines()[0] and "CLOSE" in q.lines()[2]


def test_load_rejects_other_files(tmp_path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"format": "cobot-task/0.2", "steps": []}))
    with pytest.raises(ValueError):
        TeachPath.load(f)


def test_trajectory_hits_waypoints_and_respects_speed(chain):
    p = pick_place(chain)
    tr = to_trajectory(p, max_speed_deg_s=20, hz=30)
    np.testing.assert_allclose(tr.q[0], HOME)
    np.testing.assert_allclose(tr.q[-1], HOME)
    speed = np.degrees(np.abs(np.diff(tr.q, axis=0)).max(axis=1)) * 30
    assert speed.max() <= 20.5
    # every waypoint is reached exactly
    for wp in (PICK, PLACE):
        assert np.abs(tr.q - wp).max(axis=1).min() < 1e-9
    # gripper: open, closed while carrying, open at the end
    assert tr.gripper_open[0] == 1.0 and tr.gripper_open[-1] == 1.0
    i_pick = int(np.argmin(np.abs(tr.q - PICK).max(axis=1)))
    i_place = int(np.argmin(np.abs(tr.q - PLACE).max(axis=1)))
    assert tr.gripper_open[i_pick] == 1.0  # arrive open
    assert tr.gripper_open[i_place] == 0.0  # arrive closed
    # gripper only moves while the arm stands still
    moving_g = np.abs(np.diff(tr.gripper_open)) > 0
    assert np.abs(np.diff(tr.q, axis=0))[moving_g].max() == 0.0
    # slower speed -> longer run
    assert to_trajectory(p, max_speed_deg_s=10).t[-1] > tr.t[-1] * 1.5


def test_trajectory_from_current_pose(chain):
    p = TeachPath()
    p.add_waypoint(PICK, 1.0, chain)
    tr = to_trajectory(p, q_start=HOME, gripper_start=0.0)
    np.testing.assert_allclose(tr.q[0], HOME)
    np.testing.assert_allclose(tr.q[-1], PICK)
    assert tr.gripper_open[-1] == 1.0


def test_empty_path_has_no_trajectory():
    with pytest.raises(ValueError):
        to_trajectory(TeachPath())


def test_csv_works_for_retarget_replay(tmp_path, chain):
    tr = to_trajectory(pick_place(chain))
    write_trajectory(tmp_path / "t.csv", tr)
    back = read_trajectory(tmp_path / "t.csv")
    assert len(back.t) == len(tr.t)
    np.testing.assert_allclose(back.q, tr.q, atol=1e-5)


def test_task_export_passes_taskprog_check(chain):
    p = pick_place(chain)
    task_d = to_task(p, chain)
    assert task_d["name"] == "box_to_tray"
    assert [s["do"] for s in task_d["steps"]] == ["home", "pick", "place", "home"]
    pick_xyz = task_d["locations"]["pick_1"]["xyz_mm"]
    tool = chain.fk(PICK)[:3, 3] * 1000
    assert abs(pick_xyz[0] - tool[0]) < 0.1 and abs(pick_xyz[1] - tool[1]) < 0.1
    cell = load_cell(__import__("pathlib").Path(__file__).parents[1] / "taskprog" / "cells" / "so101_twin.json")
    task = parse_task(task_d, set(cell.speeds))
    errors = [f for f in check(task, cell) if f.level == "ERROR"]
    assert not errors, errors


def test_no_task_without_pick(chain):
    p = TeachPath()
    p.add_waypoint(HOME, 1.0, chain)
    p.add_waypoint(PICK, 1.0, chain)
    assert to_task(p, chain) is None


def test_safe_name():
    assert safe_name("Box to tray!") == "box_to_tray"
    assert safe_name("1st") == "path_1st"
