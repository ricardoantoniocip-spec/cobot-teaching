import numpy as np

from retarget.kinematics import ARM_JOINTS, SerialChain, matrix_to_rotvec, rpy_to_matrix, solve_ik_with_restarts


def test_chain_has_five_arm_joints_from_urdf():
    chain = SerialChain.from_urdf()
    assert chain.names == list(ARM_JOINTS)
    assert np.all(chain.lower < chain.upper)


def test_fk_pose_is_rigid():
    chain = SerialChain.from_urdf()
    r = chain.fk(np.array([0.3, -0.2, 0.4, 0.5, -0.1]))[:3, :3]
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)


def test_jacobian_matches_finite_difference():
    chain = SerialChain.from_urdf()
    q = np.array([0.2, -0.4, 0.5, 0.3, 0.1])
    jac, tool = chain.jacobian(q)
    eps = 1e-6
    for i in range(5):
        dq = np.zeros(5)
        dq[i] = eps
        numeric = (chain.fk(q + dq)[:3, 3] - tool[:3, 3]) / eps
        assert np.allclose(jac[:3, i], numeric, atol=1e-5)


def test_ik_round_trip_reaches_within_1mm():
    chain = SerialChain.from_urdf()
    rng = np.random.default_rng(1)
    for _ in range(30):
        q = rng.uniform(chain.lower * 0.8, chain.upper * 0.8)
        res = solve_ik_with_restarts(chain, chain.fk(q), np.zeros(5))
        assert res.position_error_m < 0.001
        assert np.all(res.q >= chain.lower - 1e-9) and np.all(res.q <= chain.upper + 1e-9)


def test_rotvec_of_known_rotation():
    assert np.allclose(matrix_to_rotvec(rpy_to_matrix(0.0, 0.0, 0.7)), [0, 0, 0.7])
