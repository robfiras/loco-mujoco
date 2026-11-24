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
    rel = m.calculate_relative_velocity_in_local_frame(vel, vel, eye, backend)
    np.testing.assert_allclose(np.asarray(rel), 0.0, atol=_ATOL)


@pytest.mark.parametrize("backend", BACKENDS)
def test_relative_velocity_in_local_frame_rotated(backend):
    # the difference vel_b - vel_a is rotated from the world into the frame of A
    rot_a = sRot.from_euler("xyz", [0.3, -0.5, 1.2]).as_matrix()
    vel_a = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    vel_b = np.array([[1.0, -0.5, 0.2, -0.3, 0.7, 0.1],
                      [0.0, 0.4, -1.1, 0.9, 0.0, -0.2]])
    expected = np.concatenate([(vel_b[:, :3] - vel_a[:3]) @ rot_a,
                               (vel_b[:, 3:] - vel_a[3:]) @ rot_a], axis=1)
    rel = m.calculate_relative_velocity_in_local_frame(backend.array(vel_a), backend.array(vel_b),
                                                       backend.array(rot_a), backend)
    np.testing.assert_allclose(np.asarray(rel), expected, atol=_ATOL)


_REL_SITE_XML = """
<mujoco>
  <worldbody>
    <body name="root" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.2 0.3" mass="1"/>
      <site name="main" pos="0.05 0.02 0.1" euler="0.2 0.1 -0.3"/>
      <body name="arm" pos="0.1 0.3 0">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.5"/>
        <site name="hand" pos="0.3 0 0" euler="0.4 0 0.2"/>
        <body name="tip" pos="0.3 0 0">
          <joint name="ball" type="ball"/>
          <geom type="sphere" size="0.04" mass="0.2"/>
          <site name="finger" pos="0.05 0.01 0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _rel_site_quantities(model, data, qpos, qvel, backend):
    import mujoco
    from types import SimpleNamespace
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    site_ids = np.arange(model.nsite)
    body_ids = model.site_bodyid[site_ids]
    fields = ("site_xpos", "site_xmat", "cvel", "subtree_com")
    d = SimpleNamespace(**{f: backend.array(getattr(data, f)) for f in fields})
    out = m.calculate_relative_site_quatities(d, backend.array(site_ids), backend.array(body_ids),
                                              backend.array(model.body_rootid), backend)
    return [np.asarray(x) for x in out], data.site_xpos.copy(), data.site_xmat.copy().reshape(-1, 3, 3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_relative_site_quantities_do_not_depend_on_world_pose(backend):
    # Rotating and moving the whole system in the world must not change any quantity that is relative to
    # the main site. Before the fix the positions and velocities changed with the heading of the root.
    import mujoco
    model = mujoco.MjModel.from_xml_string(_REL_SITE_XML)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    qpos = np.zeros(model.nq)
    qpos[:3] = [0.3, -0.2, 1.0]
    qpos[3:7] = np.roll(sRot.from_euler("xyz", [0.1, -0.2, 0.4]).as_quat(), 1)
    qpos[7] = 0.6
    qpos[8:12] = np.roll(sRot.from_euler("xyz", [0.3, 0.2, -0.5]).as_quat(), 1)
    qvel = rng.normal(size=model.nv)
    base, _, _ = _rel_site_quantities(model, data, qpos, qvel, backend)

    for world_rot in [sRot.from_euler("z", np.pi / 2), sRot.from_euler("xyz", [0.7, -0.4, 2.0])]:
        qpos_w, qvel_w = qpos.copy(), qvel.copy()
        qpos_w[:3] = world_rot.apply(qpos[:3]) + np.array([1.5, -2.0, 0.3])
        qpos_w[3:7] = np.roll((world_rot * sRot.from_quat(np.roll(qpos[3:7], -1))).as_quat(), 1)
        qvel_w[:3] = world_rot.apply(qvel[:3])  # linear velocity of a free joint is in the world frame
        moved, _, _ = _rel_site_quantities(model, data, qpos_w, qvel_w, backend)
        for name, b, mv in zip(["site_rpos", "site_rangles", "site_rvel"], base, moved):
            np.testing.assert_allclose(mv, b, atol=_ATOL, err_msg=name)


@pytest.mark.parametrize("backend", BACKENDS)
def test_relative_site_quantities_match_mujoco(backend):
    # positions and velocities of the other sites relative to the main site, in the frame of the main site,
    # with the site velocities taken from MuJoCo's mj_objectVelocity
    import mujoco
    model = mujoco.MjModel.from_xml_string(_REL_SITE_XML)
    data = mujoco.MjData(model)
    qpos = np.zeros(model.nq)
    qpos[:7] = [0.2, 0.1, 1.0] + list(np.roll(sRot.from_euler("xyz", [0.2, -0.1, 1.0]).as_quat(), 1))
    qpos[7] = 1.0
    qpos[8:12] = np.roll(sRot.from_euler("y", 0.5).as_quat(), 1)
    qvel = np.random.default_rng(1).normal(size=model.nv)
    (rpos, _, rvel), site_xpos, site_xmat = _rel_site_quantities(model, data, qpos, qvel, backend)

    site_vel = np.zeros((model.nsite, 6))
    for i in range(model.nsite):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, i, site_vel[i], 0)
    main_rot = site_xmat[0]
    np.testing.assert_allclose(rpos, (site_xpos[1:] - site_xpos[0]) @ main_rot, atol=_ATOL)
    np.testing.assert_allclose(rvel[:, :3], (site_vel[1:, :3] - site_vel[0, :3]) @ main_rot, atol=_ATOL)
    np.testing.assert_allclose(rvel[:, 3:], (site_vel[1:, 3:] - site_vel[0, 3:]) @ main_rot, atol=_ATOL)


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


def test_quaternion_angular_distance_jax_matches_numpy_with_finite_gradients():
    q1 = sRot.random(64, random_state=0).as_quat()
    q2 = sRot.random(64, random_state=1).as_quat()
    np.testing.assert_allclose(np.asarray(m.quaternion_angular_distance(jnp.asarray(q1), jnp.asarray(q2), jnp)),
                               m.quaternion_angular_distance(q1, q2, np), atol=1e-5)
    # equal rotations: jax.scipy's magnitude() has a NaN gradient here
    g = jax.grad(lambda q: jnp.sum(m.quaternion_angular_distance(q, jnp.asarray(q1), jnp)))(jnp.asarray(q1))
    assert np.all(np.isfinite(np.asarray(g)))



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


def test_rotvec_from_matrix_jax_matches_scipy_and_has_finite_gradients():
    rots = sRot.random(64, random_state=0)
    special = sRot.from_rotvec([[0.0, 0.0, 0.0], [np.pi, 0.0, 0.0], [0.0, np.pi, 0.0],
                                [0.0, 0.0, np.pi], [0.0, 0.0, 1e-4]])
    mats = np.concatenate([rots.as_matrix(), special.as_matrix()])
    got = np.asarray(m.rotvec_from_matrix_jax(jnp.asarray(mats)))
    # compare as rotations: at angle pi, v and -v are the same rotation
    np.testing.assert_allclose(sRot.from_rotvec(got).as_matrix(), mats, atol=1e-5)
    np.testing.assert_allclose(got[:64], rots.as_rotvec(), atol=1e-5)

    # the jax.scipy implementation this replaces has NaN gradients at the identity
    grad = jax.grad(lambda x: jnp.sum(m.rotvec_from_matrix_jax(x)))(jnp.asarray(mats))
    assert np.all(np.isfinite(np.asarray(grad)))


# ---------------------------------------------------------------------------
# Rotation helpers in MuJoCo's quaternion convention (w, x, y, z), checked against SciPy
# ---------------------------------------------------------------------------

def _random_rotations(n, seed=0):
    """SciPy rotations and the matching scalar-first quaternions."""
    from scipy.spatial.transform import Rotation
    rot = Rotation.random(n, random_state=seed)
    return rot, np.roll(rot.as_quat(), 1, axis=-1)   # scipy is scalar-last


def test_quaternion_algebra_matches_scipy():
    rot_a, qa = _random_rotations(16, seed=0)
    rot_b, qb = _random_rotations(16, seed=1)
    v = np.random.default_rng(2).normal(size=(16, 3))

    def same_rotation(q, rot):
        expected = np.roll(rot.as_quat(), 1, axis=-1)
        q = np.asarray(q)
        return np.allclose(q * np.sign(np.sum(q * expected, axis=-1, keepdims=True)), expected, atol=1e-5)

    assert same_rotation(m.quat_mul(jnp.array(qa), jnp.array(qb)), rot_a * rot_b)
    assert same_rotation(m.quat_conj(jnp.array(qa)), rot_a.inv())
    assert np.allclose(m.quat_rotate(jnp.array(qa), jnp.array(v)), rot_a.apply(v), atol=1e-5)
    assert np.allclose(m.quat_rotate_inv(jnp.array(qa), jnp.array(v)), rot_a.inv().apply(v), atol=1e-5)
    assert np.allclose(np.linalg.norm(m.quat_normalize(3.0 * jnp.array(qa)), axis=-1), 1.0, atol=1e-5)


def test_rotation_vector_conversions_match_scipy():
    from scipy.spatial.transform import Rotation
    rotvec = np.random.default_rng(3).normal(size=(16, 3))
    rot = Rotation.from_rotvec(rotvec)

    q = np.asarray(m.quat_from_rotvec(jnp.array(rotvec)))
    expected = np.roll(rot.as_quat(), 1, axis=-1)
    assert np.allclose(q * np.sign(np.sum(q * expected, axis=-1, keepdims=True)), expected, atol=1e-5)
    # back to the rotation vector, also from the antipodal quaternion
    assert np.allclose(m.rotvec_from_quat(jnp.array(q)), rot.as_rotvec(), atol=1e-4)
    assert np.allclose(m.rotvec_from_quat(-jnp.array(q)), rot.as_rotvec(), atol=1e-4)


@pytest.mark.parametrize("backend", (np, jnp))
def test_rotation_matrix_helpers_match_scipy(backend):
    rot, q = _random_rotations(8, seed=4)
    mats = rot.as_matrix()

    assert np.allclose(m.quat_to_mat(backend.asarray(q), backend), mats, atol=1e-5)          # batched
    assert np.allclose(m.quat_to_mat(backend.asarray(q[0]), backend), mats[0], atol=1e-5)    # single

    # the two 6D layouts: columns one after the other, and row by row
    assert np.allclose(m.rot6d_from_quat(jnp.array(q)), np.concatenate([mats[:, :, 0], mats[:, :, 1]], axis=-1),
                       atol=1e-5)
    assert np.allclose(m.rot6d_from_mat(backend.asarray(mats[0]), backend), mats[0][:, :2].reshape(-1), atol=1e-6)

    # heading: rotation about z by the yaw of the body x axis
    yaw = np.arctan2(mats[0][1, 0], mats[0][0, 0])
    expected = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(m.heading_mat(backend.asarray(mats[0]), backend), expected, atol=1e-5)
    assert np.allclose(m.quat_to_mat(m.heading_quat(jnp.array(q[0]))), expected, atol=1e-5)

    # angle between two rotations
    angles = (rot[:4].inv() * rot[4:]).magnitude()
    assert np.allclose(m.rotation_angle(backend.asarray(mats[:4]), backend.asarray(mats[4:]), backend), angles,
                       atol=1e-4)


def test_rotation_helpers_have_finite_gradients_at_zero_rotation():
    """The learned dynamics are differentiated through these at zero angular velocity and equal rotations."""
    identity = jnp.array([1.0, 0.0, 0.0, 0.0])

    grads = [jax.jacobian(m.quat_from_rotvec)(jnp.zeros(3)),
             jax.jacobian(m.rotvec_from_quat)(identity),
             jax.grad(lambda v: m.eps_norm(v)[0])(jnp.zeros(3)),
             jax.grad(lambda R: m.rotation_angle(R, jnp.eye(3)))(jnp.eye(3))]
    for g in grads:
        assert np.all(np.isfinite(np.asarray(g)))
    # d rotvec / d (vector part of q) is 2 at the identity
    assert np.allclose(grads[1][:, 1:], 2.0 * np.eye(3), atol=1e-4)
