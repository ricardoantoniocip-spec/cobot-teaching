"""The live loop end to end with a stand-in body tracker and object detector: no webcam, MediaPipe or torch needed."""

import copy
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

mujoco = pytest.importorskip("mujoco")
cv2 = pytest.importorskip("cv2")

from so101_sim import scene  # noqa: E402
from so101_sim.live import Seen, run_live  # noqa: E402
from retarget.examples import synthetic_operator_frames  # noqa: E402
from retarget.objects import Detection  # noqa: E402

HZ = 30


@pytest.fixture(scope="module", autouse=True)
def model_files():
    try:
        scene.ensure_model()
    except OSError as e:  # no internet
        pytest.skip(f"SO-101 model not downloaded: {e}")


class FakeTracker:
    """Hands out made-up landmarks: 2.5 s resting at the pick spot, then a pick and place."""

    def __init__(self):
        demo = synthetic_operator_frames(hz=HZ)
        self.frames = [copy.deepcopy(demo[2 * HZ]) for _ in range(int(2.5 * HZ))] + demo
        self.i = 0

    def step(self, img, t):
        f = copy.deepcopy(self.frames[min(self.i, len(self.frames) - 1)])
        f.t = t
        self.i += 1
        return Seen(arm=f, body_px=np.full((33, 2), 50.0), body_vis=np.ones(33), hands_px=[f.hand_px + 60])


class FakeDetector:
    device = "test"

    def find(self, image, names, min_score=0.3):
        return [Detection(names[0], 0.9, (10.0, 10.0, 40.0, 50.0))]


def blank_video(path: Path, n: int) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), HZ, (320, 240))
    for _ in range(n):
        w.write(np.full((240, 320, 3), 90, np.uint8))
    w.release()
    return path


def test_live_loop_follows_logs_and_records(tmp_path):
    tracker = FakeTracker()
    video = blank_video(tmp_path / "in.mp4", len(tracker.frames))
    try:
        s = run_live(video, objects="block", window=False, save=tmp_path / "out.mp4", log=tmp_path / "log.csv", tracker=tracker, detector=FakeDetector())
    except Exception as e:  # no OpenGL on this machine
        if "gl" in str(e).lower() or "display" in str(e).lower():
            pytest.skip(f"cannot render: {e}")
        raise
    assert s["pictures"] == len(tracker.frames)
    assert s["following"] > 200 and s["ik_miss_mm_median"] < 5
    assert s["object_looks"] >= 10 and s["object_device"] == "test"
    rows = list(csv.DictReader(open(tmp_path / "log.csv")))
    assert len(rows) == s["pictures"]
    assert rows[0]["mode"] == "calibrating" and rows[-1]["mode"] == "following"
    assert "block:0.90" in rows[-1]["objects"]
    grips = [float(r["gripper_open"]) for r in rows if r["mode"] == "following"]
    assert min(grips) < 0.05 and max(grips) > 0.95
    cap = cv2.VideoCapture(str(tmp_path / "out.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == s["pictures"]
    assert (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (1280, 600)


def test_live_without_twin_or_objects(tmp_path):
    tracker = FakeTracker()
    s = run_live(blank_video(tmp_path / "in.mp4", 120), window=False, twin=False, tracker=tracker, max_seconds=3.0)
    assert 85 <= s["pictures"] <= 92 and s["object_looks"] == 0
