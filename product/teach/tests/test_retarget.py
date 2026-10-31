import numpy as np

from retarget.demo import HandDemo, read_demo, read_trajectory, write_demo, write_trajectory
from retarget.examples import example_pick_place
from retarget.kinematics import SerialChain
from retarget.retarget import RetargetConfig, limit_speed, retarget, smooth


def test_example_demo_retargets_fully_and_respects_speed(tmp_path):
    chain = SerialChain.from_urdf()
    demo = example_pick_place(chain)
    path = tmp_path / "demo.csv"
    write_demo(path, demo)
    cfg = RetargetConfig(max_joint_speed_rad_s=np.radians(20))
    traj, report = retarget(read_demo(path), chain, cfg)
    assert report.reached_fraction == 1.0
    speeds = np.abs(np.diff(traj.q, axis=0)) / np.diff(traj.t)[:, None]
    assert speeds.max() <= np.radians(20) * 1.0001
    assert traj.gripper_open[0] == 1.0 and traj.gripper_open.min() == 0.0


def test_unreachable_point_is_reported():
    chain = SerialChain.from_urdf()
    far = np.eye(4)
    far[:3, 3] = [2.0, 0.0, 0.2]  # 2 m away, far outside the SO-101 reach
    demo = HandDemo(np.array([0.0, 0.1]), np.array([far, far]), np.zeros(2))
    _, report = retarget(demo, chain)
    assert report.reached_fraction == 0.0
    assert report.worst_error_mm > 1000


def test_trajectory_csv_round_trip(tmp_path):
    chain = SerialChain.from_urdf()
    traj, _ = retarget(example_pick_place(chain), chain)
    path = tmp_path / "traj.csv"
    write_trajectory(path, traj)
    back = read_trajectory(path)
    assert np.allclose(back.q, traj.q, atol=1e-4)
    q_mid, _ = back.sample(back.t[-1] / 2)
    assert q_mid.shape == (5,)


def test_smooth_keeps_endpoints_and_limit_speed_stretches_time():
    q = np.cumsum(np.ones((10, 5)), axis=0)
    s = smooth(q, 5)
    assert np.allclose(s[0], q[0]) and np.allclose(s[-1], q[-1])
    t, scale = limit_speed(np.arange(10) * 0.1, q, max_speed=5.0)
    assert np.isclose(scale, 2.0) and np.isclose(t[-1], 1.8)
