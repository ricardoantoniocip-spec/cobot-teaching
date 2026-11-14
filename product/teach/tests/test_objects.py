"""Object tracking from detector boxes, on made-up videos with a stand-in detector (no model needed)."""

import json

import numpy as np
import pytest

from retarget import objects as o
from retarget.object_motion import NoObjectMotion
from taskprog.from_steps import steps_to_task

FPS = 15.0
W, H = 640, 360
BOX = (40, 72)  # width, height of the made-up part


def part_xy(t: float, moves) -> tuple[int, int]:
    """Top-left corner of the part at time t. moves: [(t_lift, t_down, from_xy, to_xy), ...]"""
    xy = np.array(moves[0][2], dtype=float)
    for t_lift, t_down, a, b in moves:
        if t >= t_lift:
            s = min(1.0, (t - t_lift) / (t_down - t_lift))
            xy = np.array(a) + s * (np.array(b) - np.array(a))
    return int(xy[0]), int(xy[1])


def make_video(moves, seconds=12.0):
    rng = np.random.default_rng(0)
    bg = np.full((H, W, 3), 120, np.uint8) + rng.integers(0, 12, (H, W, 3), dtype=np.uint8)
    part = np.kron(rng.integers(0, 255, (18, 10, 3), dtype=np.uint8), np.ones((4, 4, 1), np.uint8))
    frames = []
    for i in range(int(seconds * FPS)):
        f = bg.copy()
        x, y = part_xy(i / FPS, moves)
        f[y:y + BOX[1], x:x + BOX[0]] = part
        frames.append(f)
    return frames


class FakeDetector:
    """Says where the part is (as a real detector would), plus a thing that never moves."""

    def __init__(self, moves, label="carton", miss_every=0):
        self.moves, self.label, self.miss_every, self.calls = moves, label, miss_every, 0

    def find(self, image, names, min_score=0.3):
        i = self.calls
        self.calls += 1
        t = i * round(o.EVERY_S * FPS) / FPS
        x, y = part_xy(t, self.moves)
        out = [o.Detection("box", 0.5, (500, 40, 560, 100)), o.Detection("hand", 0.9, (x - 30, y, x + 20, y + 40))]
        if not (self.miss_every and i % self.miss_every == self.miss_every - 1):
            out += [o.Detection(self.label, 0.8, (x, y, x + BOX[0], y + BOX[1])), o.Detection("object", 0.4, (x + 1, y, x + BOX[0], y + BOX[1]))]
        return out


ONE = [(3.0, 5.0, (100, 60), (400, 200))]
TWO = [(3.0, 5.0, (100, 60), (400, 200)), (8.0, 9.5, (400, 200), (250, 250))]


def test_one_move_gives_grasp_release_and_places():
    tr = o.track_object(make_video(ONE), FPS, None, FakeDetector(ONE))
    assert tr.name == "carton" and len(tr.moves) == 1  # the still box and the hand are not the object
    m = tr.moves[0]
    assert m.t_grasp == pytest.approx(2.9, abs=0.35) and m.t_release == pytest.approx(5.1, abs=0.35)
    assert m.pick_px == pytest.approx((120, 132), abs=3)  # bottom centre: where it stands
    assert m.place_px == pytest.approx((420, 272), abs=3)
    assert any("no object name given" in n for n in tr.notes)


def test_a_part_moved_twice_gives_two_picks():
    tr = o.track_object(make_video(TWO), FPS, "juice carton", FakeDetector(TWO, "juice carton"))
    assert tr.name == "juice carton" and tr.notes == []
    assert [round(m.t_grasp) for m in tr.moves] == [3, 8]
    assert tr.moves[1].pick_px == pytest.approx(tr.moves[0].place_px, abs=3)
    assert tr.grip_events() == [(m.t_grasp, m.t_release) for m in tr.moves]


def test_missed_pictures_are_bridged():
    tr = o.track_object(make_video(ONE), FPS, None, FakeDetector(ONE, miss_every=4))
    assert len(tr.moves) == 1 and tr.seen_fraction < 0.8
    assert tr.moves[0].t_grasp == pytest.approx(2.9, abs=0.6)


def test_nothing_moved_is_an_error():
    still = [(99.0, 100.0, (100, 60), (100, 60))]
    with pytest.raises(NoObjectMotion, match="stand still, get moved"):
        o.track_object(make_video(still), FPS, None, FakeDetector(still))


def test_hand_on_the_object_is_checked():
    frames = make_video(ONE)
    on = {i: np.array(part_xy(i / FPS, ONE)) + (20, 20) for i in range(len(frames))}
    away = {i: np.array([600.0, 340.0]) for i in range(len(frames))}
    assert o.track_object(frames, FPS, None, FakeDetector(ONE), hand_px=on).moves[0].hand_on_object is True
    tr = o.track_object(frames, FPS, None, FakeDetector(ONE), hand_px=away)
    assert tr.moves[0].hand_on_object is False and any("hand is not on it" in n for n in tr.notes)


def test_a_slow_carry_is_not_a_rest():
    centres = [np.array([100.0 + 6 * k, 100.0]) for k in range(12)]  # creeps 6 px per look, box diagonal 82 px
    assert o.rests_and_moves(centres, [np.array(BOX, dtype=float)] * 12) == []
    centres = [np.array([100.0, 100.0])] * 4 + [np.array([300.0, 100.0])] * 4
    assert o.rests_and_moves(centres, [np.array(BOX, dtype=float)] * 8) == [(0, 3), (4, 7)]


def test_same_thing_under_two_names_is_one_box():
    dets = [o.Detection("carton", 0.5, (10, 10, 50, 90)), o.Detection("bottle", 0.7, (11, 10, 50, 91)), o.Detection("hand", 0.9, (0, 0, 30, 30)),
            o.Detection("box", 0.4, (0, 0, 600, 350))]  # fmt: skip
    kept = o.clean(dets, (H, W, 3))
    assert len(kept) == 1 and kept[0][0].label == "bottle" and sorted(kept[0][1]) == ["bottle", "carton"]


def _steps_doc(obj: dict) -> dict:
    step = lambda kind, t, xyz: {"step": kind, "t_start": t, "t_end": t, "from_xyz": xyz, "to_xyz": xyz, "holding": kind == "grasp", "label": None}  # noqa: E731
    return {"video": "v.mp4", "mode": "body", "arm": "right", "object": obj,
            "steps": [step("grasp", 3.0, [0.2, 0.1, 0.03]), step("release", 5.0, [0.2, -0.1, 0.03]),
                      step("grasp", 8.0, [0.2, -0.1, 0.03]), step("release", 9.5, [0.25, 0.0, 0.03])]}  # fmt: skip


def test_the_detected_object_reaches_the_task_file():
    tr = o.track_object(make_video(TWO), FPS, "juice carton", FakeDetector(TWO, "juice carton"))
    task = steps_to_task(_steps_doc({**json.loads(json.dumps(tr.to_json())), "named_by": "you"}), flat_table=True)
    assert list(task["objects"]) == ["juice_carton"]
    assert [s["do"] for s in task["steps"]] == ["home", "pick", "place", "pick", "place", "home"]
    # The second pick starts where the first place ended (same box in the picture): one location.
    assert [s.get("at") for s in task["steps"]] == [None, "pick_1", "place_1", "place_1", "place_2", None]
    assert not any("detector's guess" in r for r in task["review"])


def test_a_guessed_name_and_a_missing_hand_ask_for_review():
    obj = {"name": "bottle", "score": 0.4, "named_by": "detector", "notes": [],
           "moves": [{"t_grasp": 3.0, "hand_on_object": False}, {"t_grasp": 8.0, "hand_on_object": True}]}
    task = steps_to_task(_steps_doc(obj), flat_table=True)
    assert list(task["objects"]) == ["bottle"]
    assert any("detector's guess" in r for r in task["review"])
    assert any("pick 1" in r and "hand was not on it" in r for r in task["review"])
    labelled = steps_to_task(_steps_doc(obj), {"objects": [{"name": "juice_carton"}]}, flat_table=True)
    assert list(labelled["objects"]) == ["juice_carton"]  # a label wins over the detector's word


def test_take1_with_the_detector_matches_the_reviewed_example():
    """steps.json written by the video app with --object "juice carton" -> task -> URScript."""
    from pathlib import Path

    from taskprog.check import check, passed
    from taskprog.from_steps import dumps
    from taskprog.task import load_cell, parse_task
    from taskprog.ur import urscript

    here = Path(__file__).resolve().parents[1] / "taskprog"
    ex = here / "examples" / "take1_object_detector"
    doc = json.loads((ex / "steps.json").read_text())
    assert doc["object"]["name"] == "juice carton" and len(doc["object"]["moves"]) == 2
    task = steps_to_task(doc, flat_table=True)
    text = (ex / "task.json").read_text()
    assert dumps(task) == text
    cell = load_cell(here / "cells" / "ur5e_table.json")
    parsed = parse_task(task, set(cell.speeds))
    assert passed(check(parsed, cell))
    assert urscript(parsed, cell, text, "task.json") == (ex / "demo_task.script").read_text()


def test_twin_refuses_a_pick_where_the_block_is_not():
    from pathlib import Path

    from taskprog.task import load_cell, parse_task
    from taskprog.twin import run_twin

    here = Path(__file__).resolve().parents[1] / "taskprog"
    task = json.loads((here / "examples" / "take1_object_detector" / "task.json").read_text())
    task["locations"]["elsewhere"] = {"xyz_mm": [200, 100, 0]}
    task["steps"][3]["at"] = "elsewhere"
    with pytest.raises(ValueError, match="the block was last put down at place_1"):
        run_twin(parse_task(task), load_cell(here / "cells" / "so101_twin.json"))
