"""Unit tests for the pure math helpers in loco_mujoco.core.utils.math.

These functions are backend-parametric (numpy / jax.numpy) and are otherwise
only exercised indirectly through the observation/reward pipeline. Testing them
directly pins down their contracts (roundtrips, relative/global inverses, frame
transforms) on both backends.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.spatial.transform import Rotation as sRot

from loco_mujoco.core.utils import math as m

jax.config.update("jax_platform_name", "cpu")

BACKENDS = [np, jnp]
_ATOL = 1e-5


@pytest.fixture
def bk(request):
    return request.param


# --------------------------- rotate_obs ---------------------------

def test_rotate_obs():
    # state layout: [rot, xvel, yvel]
    state = [0.0, 1.0, 0.0]
    rotated = m.rotate_obs(state, np.pi / 2, idx_rot=0, idx_xvel=1, idx_yvel=2)
    # rotation entry gets the added angle (wrapped to [-pi, pi])
    assert rotated[0] == pytest.approx(np.pi / 2, abs=_ATOL)
    # a +90deg rotation maps x-velocity onto +y
    assert rotated[1] == pytest.approx(0.0, abs=_ATOL)
    assert rotated[2] == pytest.approx(1.0, abs=_ATOL)


# --------------------------- angle <-> matrix ---------------------------

@pytest.mark.parametrize("angle", [-2.0, -0.3, 0.0, 0.7, 2.5])
def test_angle_mat_roundtrip(angle):
    mat = m.angle2mat_xy(angle)
    assert mat.shape == (3, 3)
    recovered = m.mat2angle_xy(mat.reshape(9))
    assert recovered == pytest.approx(angle, abs=_ATOL)


@pytest.mark.parametrize("raw,expected", [
    (0.0, 0.0),
    (np.pi / 2, np.pi / 2),
    (2 * np.pi, 0.0),
    (3 * np.pi, -np.pi),
    (-3 * np.pi, -np.pi),
])
def test_transform_angle_2pi(raw, expected):
    got = m.transform_angle_2pi(raw)
    assert got == pytest.approx(expected, abs=_ATOL)
    assert -np.pi - _ATOL <= got <= np.pi + _ATOL


# --------------------------- relative positions / velocities ---------------------------

@pytest.mark.parametrize("backend", BACKENDS)
def test_calc_rel_positions(backend):
    xpos = backend.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    main = backend.array([1.0, 1.0, 1.0])
    rel = m.calc_rel_positions(xpos, main, backend)
    np.testing.assert_allclose(np.asarray(rel), [[0, 1, 2], [3, 4, 5]], atol=_ATOL)


@pytest.mark.parametrize("backend", BACKENDS)
def test_calc_rel_velocities(backend):
    xvel = backend.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    main = backend.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    rel = m.calculate_relative_velocities(xvel, main, backend)
    np.testing.assert_allclose(np.asarray(rel), [0, 1, 2, 3, 4, 5], atol=_ATOL)


@pytest.mark.parametrize("backend", BACKENDS)
def test_calc_rel_quaternions_identity(backend):
    # relative quaternion of a rotation with itself is the identity (scalar-last)
    q = sRot.from_euler("xyz", [0.3, -0.2, 0.5]).as_quat()  # scalar-last
    q = backend.array(q)
    rel = m.calc_rel_quaternions(q, q, backend)
    rel = np.asarray(rel)
    # identity is (0,0,0,±1); compare via angle
    ang = sRot.from_quat(rel).magnitude()
    assert ang == pytest.approx(0.0, abs=_ATOL)


# --------------------------- rotation matrices: relative <-> global ---------------------------

@pytest.mark.parametrize("backend", BACKENDS)
def test_relative_global_rotation_roundtrip(backend):
    main = sRot.from_euler("z", 0.4).as_matrix()
    others = np.stack([
        sRot.from_euler("x", 0.2).as_matrix(),
        sRot.from_euler("y", -0.6).as_matrix(),
    ])
    main_b = backend.array(main)
    others_b = backend.array(others)

    rel = m.calculate_relative_rotation_matrices(main_b, others_b, backend)
    glob = m.calculate_global_rotation_matrices(main_b, rel, backend)

    assert np.asarray(rel).shape == (2, 3, 3)
    np.testing.assert_allclose(np.asarray(glob), others, atol=_ATOL)


@pytest.mark.parametrize("backend", BACKENDS)
def test_relative_velocity_in_local_frame_zero(backend):
    # identical velocities with identity rotations -> zero relative velocity
    vel = backend.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    eye = backend.array(np.eye(3))
    rel = m.calculate_relative_velocity_in_local_frame(vel, vel, eye, eye, backend)
    np.testing.assert_allclose(np.asarray(rel), 0.0, atol=_ATOL)


# --------------------------- quaternion helpers ---------------------------

@pytest.mark.parametrize("backend", BACKENDS)
def test_quaternion_angular_distance(backend):
    q1 = backend.array(sRot.from_euler("z", 0.0).as_quat())
    q2 = backend.array(sRot.from_euler("z", 0.5).as_quat())
    d = m.quaternion_angular_distance(q1, q2, backend)
    assert float(np.asarray(d)) == pytest.approx(0.5, abs=_ATOL)
    # distance to self is zero
    d0 = m.quaternion_angular_distance(q1, q1, backend)
    assert float(np.asarray(d0)) == pytest.approx(0.0, abs=_ATOL)


@pytest.mark.parametrize("backend", BACKENDS)
def test_quat2angle(backend):
    q = backend.array(sRot.from_rotvec([0.0, 0.0, 0.7]).as_quat())
    rotvec = np.asarray(m.quat2angle(q, backend))
    np.testing.assert_allclose(rotvec, [0.0, 0.0, 0.7], atol=_ATOL)


def test_quat_scalar_order_roundtrip():
    q_first = np.array([0.1, 0.2, 0.3, 0.4])  # (w, x, y, z)
    q_last = m.quat_scalarfirst2scalarlast(q_first)
    np.testing.assert_allclose(q_last, [0.2, 0.3, 0.4, 0.1])
    back = m.quat_scalarlast2scalarfirst(q_last)
    np.testing.assert_allclose(back, q_first)


@pytest.mark.parametrize("backend", BACKENDS)
def test_atleast_3d(backend):
    v = backend.array([1.0, 2.0, 3.0])
    out = m.atleast_3d(v, backend)
    assert out.ndim == 3
    assert out.shape == (1, 1, 3)
    # already-3d passes through unchanged
    w = backend.array(np.zeros((2, 3, 4)))
    assert m.atleast_3d(w, backend).shape == (2, 3, 4)
