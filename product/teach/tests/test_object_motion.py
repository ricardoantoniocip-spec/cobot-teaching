"""Grasp and release times from the object's motion, on a made-up video with known times."""

import numpy as np
import pytest

from retarget.object_motion import NoObjectMotion, find_object_motion

FPS = 15.0
W, H = 320, 180


def make_video(t_lift=3.0, t_down=6.0, seconds=9.0, move=True):
    rng = np.random.default_rng(0)
    bg = np.full((H, W, 3), 120, np.uint8) + rng.integers(0, 12, (H, W, 3), dtype=np.uint8)
    obj = rng.integers(0, 255, (36, 20, 3), dtype=np.uint8)  # a box with a busy print
    obj[:, :, 2] = np.clip(obj[:, :, 2].astype(int) + 80, 0, 255)
    a, b = np.array([60.0, 60.0]), np.array([220.0, 110.0])
    frames = []
    for i in range(int(seconds * FPS)):
        t = i / FPS
        f = bg.copy()
        s = float(np.clip((t - t_lift) / (t_down - t_lift), 0, 1)) if move else 0.0
        x, y = (a + s * (b - a)).astype(int)
        f[y:y + 36, x:x + 20] = obj
        if t_lift - 1.0 < t < t_down + 1.0:  # the hand covers the top third while it holds the object
            f[y - 6:y + 12, x - 4:x + 24] = (150, 170, 210)
        frames.append(f)
    return np.array(frames)


def test_finds_lift_and_set_down():
    m = find_object_motion(make_video(), FPS)
    assert m.t_grasp == pytest.approx(3.0, abs=0.4)
    assert m.t_release == pytest.approx(6.0, abs=0.4)
    assert m.pick_px == pytest.approx((70, 78), abs=4)
    assert m.place_px == pytest.approx((230, 128), abs=4)
    assert m.grips_arg() == f"{m.t_grasp:.2f}-{m.t_release:.2f}"


def test_nothing_moved_is_an_error():
    with pytest.raises(NoObjectMotion):
        find_object_motion(make_video(move=False), FPS)
