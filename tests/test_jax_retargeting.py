"""Tests for the JAX retargeting pipeline, on a small hand-written model and BVH file."""
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from loco_mujoco.retargeting import (Calibration, compile_forward_kinematics,
                                     compile_mjx_forward_kinematics, global_transforms, load_bvh,
                                     make_solver, pad_clips, retract, site_targets)
from loco_mujoco.retargeting.soma import WORLD_FROM_BVH, _rotation_between

XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1" quat="0.9 0.1 0.2 0.3">
      <freejoint/>
      <geom size="0.1"/>
      <site name="base_site" pos="0.1 0 0"/>
      <body name="upper" pos="0.2 0.1 0" quat="0.7 0 0.7 0">
        <joint name="hinge_a" axis="0 1 0" pos="0.05 0 0" range="-2 2" limited="true"/>
        <geom size="0.1"/>
        <body name="lower" pos="0 0 -0.3">
          <joint name="hinge_b" axis="1 0 0" range="-2 2" limited="true"/>
          <joint name="slide" type="slide" axis="0 0 1" range="-0.1 0.1" limited="true"/>
          <geom size="0.1"/>
          <site name="tip" pos="0 0.05 -0.3" quat="0.5 0.5 0.5 0.5"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0 100 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT Spine
  {
    OFFSET 0 10 0
    CHANNELS 3 Zrotation Yrotation Xrotation
    End Site
    {
      OFFSET 0 5 0
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.5
0 100 0 0 0 0 0 0 0
10 100 0 90 0 0 0 0 90
"""


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string(XML)


def _site_ids(model):
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in ("base_site", "tip")]


def _random_qpos(model, rng):
    q = np.zeros(model.nq)
    q[:3] = rng.normal(size=3)
    q[3:7] = rng.normal(size=4)
    q[3:7] /= np.linalg.norm(q[3:7])
    q[7:] = rng.uniform(-0.1, 0.1, model.nq - 7)
    return q


def test_forward_kinematics_matches_mujoco(model):
    site_ids = _site_ids(model)
    lower = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lower")
    point = np.array([0.01, 0.02, 0.03])
    fk = jax.jit(compile_forward_kinematics(model, site_ids, [lower], [point]))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)

    for _ in range(10):
        data.qpos[:] = _random_qpos(model, rng)
        mujoco.mj_kinematics(model, data)
        pos, rot, points = fk(jnp.asarray(data.qpos, jnp.float32))
        np.testing.assert_allclose(pos, data.site_xpos[site_ids], atol=1e-5)
        np.testing.assert_allclose(rot, data.site_xmat[site_ids].reshape(-1, 3, 3), atol=1e-5)
        expected = data.xpos[lower] + data.xmat[lower].reshape(3, 3) @ point
        np.testing.assert_allclose(points[0], expected, atol=1e-5)


def test_mjx_and_slim_forward_kinematics_agree(model):
    site_ids = _site_ids(model)
    lower = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lower")
    point = np.array([0.01, 0.02, 0.03])
    slim = jax.jit(compile_forward_kinematics(model, site_ids, [lower], [point]))
    full = jax.jit(compile_mjx_forward_kinematics(model, site_ids, [lower], [point]))
    q = jnp.asarray(_random_qpos(model, np.random.default_rng(3)), jnp.float32)
    for a, b in zip(slim(q), full(q)):
        np.testing.assert_allclose(a, b, atol=1e-5)


def test_mjx_forward_kinematics_keeps_model_collisions(model):
    contype = model.geom_contype.copy()
    compile_mjx_forward_kinematics(model, _site_ids(model))
    np.testing.assert_array_equal(model.geom_contype, contype)


def test_site_targets_keep_robot_bone_length():
    bone = np.array([0.0, 0.0, -0.4])
    calibration = Calibration(0.5, np.array([0, 1]), 0, np.zeros((2, 3)),
                              np.tile(np.eye(3), (2, 1, 1)), ((1, 0, bone),))
    turn = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    joint_pos = jnp.asarray([[1.0, 0.0, 2.0], [5.0, 5.0, 5.0]])
    joint_rot = jnp.asarray(np.stack([turn, np.eye(3)]))
    pos, rot = site_targets(joint_pos, joint_rot, calibration)

    np.testing.assert_allclose(pos[0], [0.5, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(pos[1], pos[0] + turn @ bone, atol=1e-6)
    np.testing.assert_allclose(rot, joint_rot, atol=1e-6)


def test_forward_kinematics_rejects_fixed_base():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint axis="0 0 1"/><geom size="1"/></body></worldbody></mujoco>')
    with pytest.raises(ValueError):
        compile_forward_kinematics(model, [])


def test_retract_keeps_unit_quaternion_and_identity(model):
    q = jnp.asarray(_random_qpos(model, np.random.default_rng(1)), jnp.float32)
    np.testing.assert_allclose(retract(q, jnp.zeros(model.nv)), q, atol=1e-6)
    moved = retract(q, jnp.ones(model.nv) * 0.3)
    np.testing.assert_allclose(jnp.linalg.norm(moved[3:7]), 1.0, atol=1e-6)


def test_solver_recovers_reachable_targets(model):
    site_ids = _site_ids(model)
    fk = compile_forward_kinematics(model, site_ids)
    rng = np.random.default_rng(2)

    q = np.tile(_random_qpos(model, rng), (8, 1))
    q[:, 7:] += np.linspace(0, 0.2, 8)[:, None]
    q = jnp.asarray(q, jnp.float32)
    pos, rot, _ = jax.vmap(fk)(q)

    solve = make_solver(fk, model.nv, -2 * np.ones(3), 2 * np.ones(3), np.ones(2), np.ones(2))
    start = q[0].at[7:].add(0.05)
    out = solve(start[None], pos[None], rot[None])[0]

    final_pos, _, _ = jax.vmap(fk)(out)
    np.testing.assert_allclose(final_pos, pos, atol=1e-3)


def test_load_bvh(tmp_path):
    path = tmp_path / "clip.bvh"
    path.write_text(BVH)
    motion = load_bvh(str(path))

    assert motion.skeleton.names == ("Hips", "Spine")
    assert motion.skeleton.parents == (-1, 0)
    assert motion.fps == 2.0
    np.testing.assert_allclose(motion.local_pos[0, 0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(motion.local_pos[1, 0], [0.1, 1.0, 0.0])
    np.testing.assert_allclose(motion.local_pos[:, 1], [[0.0, 0.1, 0.0]] * 2)
    # 90 degrees about z takes x onto y
    np.testing.assert_allclose(motion.local_rot[1, 0] @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_load_bvh_rejects_truncated_motion(tmp_path):
    path = tmp_path / "clip.bvh"
    path.write_text(BVH.rsplit("\n", 2)[0] + "\n")
    with pytest.raises(ValueError):
        load_bvh(str(path))


def test_load_bvh_rejects_nan(tmp_path):
    path = tmp_path / "clip.bvh"
    path.write_text(BVH.replace("10 100 0 90", "10 100 0 nan"))
    with pytest.raises(ValueError):
        load_bvh(str(path))


def test_global_transforms(tmp_path):
    path = tmp_path / "clip.bvh"
    path.write_text(BVH)
    motion = load_bvh(str(path))
    pos, rot = global_transforms(motion.skeleton.parents, jnp.asarray(motion.local_pos),
                                 jnp.asarray(motion.local_rot), [0, 1])

    # BVH +y is up, so the rest pose stacks the spine above the hips in world z
    np.testing.assert_allclose(pos[0, 0], [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(pos[0, 1], [0.0, 0.0, 1.1], atol=1e-6)
    np.testing.assert_allclose(rot[0, 0], WORLD_FROM_BVH, atol=1e-6)
    # In frame two the hips turn the spine offset onto BVH -x
    expected = WORLD_FROM_BVH @ np.array([0.1 - 0.1, 1.0, 0.0])
    np.testing.assert_allclose(pos[1, 1], expected, atol=1e-6)


@pytest.mark.parametrize("a, b", [([1, 0, 0], [0, 1, 0]), ([0, 0, 1], [0, 0, 1]), ([0, 1, 0], [0, -1, 0])])
def test_rotation_between(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    r = _rotation_between(a, b)
    np.testing.assert_allclose(r @ a, b, atol=1e-7)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-7)


def test_pad_clips_repeats_last_frame():
    clips = (np.array([[1.0], [2.0]]), np.array([[3.0]]))
    padded = pad_clips(clips, 3)
    assert padded.shape == (2, 3, 1)
    np.testing.assert_array_equal(padded[1, :, 0], [3.0, 3.0, 3.0])
    np.testing.assert_array_equal(padded[0, :, 0], [1.0, 2.0, 2.0])
