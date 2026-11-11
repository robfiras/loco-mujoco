"""Tests for the JAX retargeting pipeline, on a small hand-written model and BVH file."""
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from loco_mujoco.retargeting import (Calibration, body_contact_points, compile_forward_kinematics,
                                     compile_mjx_forward_kinematics, detect_contacts, global_transforms,
                                     hold_contacts, load_bvh, make_solver, pad_clips, retract, site_targets,
                                     soften_contacts)
from loco_mujoco.retargeting.solver import SolverConfig, rotation_error, soft_min
from loco_mujoco.retargeting.soma import WORLD_FROM_BVH, _rotation_between, limb_dof_damping, limit_dofs

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

    solve = make_solver(fk, model.nv, -2 * np.ones(3), 2 * np.ones(3), np.ones(2), np.ones(2),
                        SolverConfig(damping=1e-2))
    start = q[0].at[7:].add(0.05)
    out = solve(start[None], pos[None], rot[None])[0]

    final_pos, _, _ = jax.vmap(fk)(out)
    np.testing.assert_allclose(final_pos, pos, atol=1e-3)


def test_solver_moves_other_joints_when_a_limit_stops_one():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body name="root"><freejoint/><geom size="0.1"/><site name="root"/>
      <body name="link"><joint name="hinge" axis="0 1 0"/><geom size="0.1"/>
        <site name="tip" pos="0.5 0 0"/>
      </body>
    </body></worldbody></mujoco>""")
    fk = compile_forward_kinematics(model, [0, 1])
    limit = 0.2
    angle = 1.0
    target_pos = jnp.asarray([[0.0, 0.0, 0.0], [0.5 * np.cos(angle), 0.0, -0.5 * np.sin(angle)]], jnp.float32)
    target_rot = jnp.tile(jnp.eye(3, dtype=jnp.float32), (2, 1, 1))
    start = jnp.asarray([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], jnp.float32)

    def tip_error(limit_passes):
        solve = make_solver(fk, model.nv, -limit * np.ones(1), limit * np.ones(1), np.array([1.0, 10.0]),
                            np.zeros(2), SolverConfig(iterations=1, init_iterations=1, limit_passes=limit_passes))
        out = solve(start, target_pos[None, None], target_rot[None, None])[0]
        assert abs(float(out[0, 7])) <= limit + 1e-6
        return float(jnp.linalg.norm(fk(out[0])[0][1] - target_pos[1]))

    assert tip_error(3) < 0.2 * tip_error(1)


@pytest.mark.parametrize("angle", [0.0, 1e-5, 0.5, 2.0, 3.1, 3.14159, np.pi])
def test_rotation_error_is_finite_with_finite_jacobian(angle):
    c, s = np.cos(angle), np.sin(angle)
    rot = jnp.asarray([[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]], jnp.float32)
    target = jnp.eye(3, dtype=jnp.float32)[None]

    assert np.isfinite(rotation_error(target, rot)).all()
    jac = jax.jacfwd(lambda r: rotation_error(target, r))(rot)
    assert np.isfinite(jac).all()
    assert np.abs(jac).max() < 1e3


def test_rotation_error_is_the_rotation_vector():
    angle = 2.0
    c, s = np.cos(angle), np.sin(angle)
    rot = jnp.asarray([[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]], jnp.float32)
    # the gain bound only applies very close to 180 degrees
    np.testing.assert_allclose(rotation_error(jnp.eye(3)[None], rot)[0], [0, 0, angle], atol=1e-5)


def _two_arm_model():
    arm = """
    <body name="{0}_upper" pos="0 0 0">
      <joint name="{0}_a" axis="1 0 0"/><joint name="{0}_b" axis="0 1 0"/><geom size="0.1"/>
      <body name="{0}_fore" pos="0 0 -0.3">
        <joint name="{0}_c" axis="0 1 0"/><geom size="0.1"/><site name="{0}_elbow_mimic"/>
        <body name="{0}_hand" pos="0 0 -0.3">
          <joint name="{0}_d" axis="1 0 0"/><geom size="0.1"/><site name="{0}_hand_mimic"/>
        </body>
      </body>
    </body>"""
    return mujoco.MjModel.from_xml_string(f"""
    <mujoco><worldbody><body name="torso"><freejoint/><geom size="0.1"/>
      <site name="upper_body_mimic"/>{arm.format("left")}{arm.format("right")}
    </body></worldbody></mujoco>""")


def test_limit_dofs_leaves_out_only_the_elbows():
    model = _two_arm_model()
    mask = limit_dofs(model)
    for j in range(1, model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        assert mask[model.jnt_dofadr[j]] == (not name.endswith("_c"))


def test_limb_dof_damping_leaves_elbows_undamped():
    model = _two_arm_model()
    damping = limb_dof_damping(model, 10.0, 1.0)

    def dof(name):
        return model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]

    np.testing.assert_array_equal(damping[:6], 0.0)
    for side in ("left", "right"):
        assert damping[dof(f"{side}_a")] == 10.0 and damping[dof(f"{side}_b")] == 10.0
        assert damping[dof(f"{side}_c")] == 0.0
        assert damping[dof(f"{side}_d")] == 1.0


def test_soft_min_stays_between_minimum_and_mean():
    values = jnp.asarray([0.0, 0.0, 0.0, 0.5])
    assert soft_min(values, 0.0) == 0.0
    assert abs(float(soft_min(values, 0.01))) < 1e-6
    close = jnp.asarray([0.0, 0.01])
    assert 0.0 < float(soft_min(close, 0.01)) < 0.005


def test_body_contact_points_lie_on_the_surface_and_include_fingers():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body name="arm"><freejoint/><geom type="box" size="0.1 0.2 0.3"/>
      <site name="pelvis_mimic"/>
      <body name="hand" pos="1 0 0" euler="0 0 90"><joint axis="0 1 0"/><geom type="box" size="0.1 0.1 0.1"/>
        <site name="left_hand_mimic"/>
        <body name="finger" pos="0.5 0 0"><joint axis="0 1 0"/><geom type="box" size="0.1 0.1 0.1"/></body>
      </body>
    </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    bodies, points, groups = body_contact_points(model, data, ["pelvis_mimic", "left_hand_mimic"])

    pelvis = np.array([points[i] for i in groups[0]])
    np.testing.assert_allclose(np.abs(pelvis), np.tile([0.1, 0.2, 0.3], (len(pelvis), 1)), atol=1e-6)
    hand = np.array([points[i] for i in groups[1]])
    assert hand[:, 0].max() == pytest.approx(0.6)
    assert {bodies[i] for i in groups[1]} == {2}


def test_detect_contacts_needs_low_and_slow():
    frames = 30
    height = np.full(frames, 0.05)
    height[10:20] = 0.5
    x = np.zeros(frames)
    x[25:] = np.arange(5) * 0.1
    joint_pos = jnp.asarray(np.stack([x, 0 * x, height], axis=1)[:, None], jnp.float32)
    contact = np.asarray(detect_contacts(joint_pos, jnp.asarray([0.1]), jnp.asarray([[True]]), 30.0))[:, 0]
    assert contact[:9].all() and contact[21:25].all()
    assert not contact[11:20].any() and not contact[27:].any()


def test_detect_contacts_ignores_single_frames():
    height = np.full(12, 0.5)
    height[5] = 0.0
    joint_pos = jnp.asarray(np.stack([0 * height, 0 * height, height], axis=1)[:, None], jnp.float32)
    assert not np.asarray(detect_contacts(joint_pos, jnp.asarray([0.1]), jnp.asarray([[True]]), 30.0)).any()


def test_soften_contacts_keeps_contacts_whole_and_fades_around_them():
    contact = np.zeros((12, 1), bool)
    contact[4:7] = True
    weight = np.asarray(soften_contacts(jnp.asarray(contact), blend=1))[:, 0]
    np.testing.assert_allclose(weight[3:8], [0.5, 1.0, 1.0, 1.0, 0.5])
    assert not weight[:3].any() and not weight[8:].any()


def test_soften_contacts_without_any_contact():
    assert not np.asarray(soften_contacts(jnp.zeros((5, 1), bool))).any()


def test_hold_contacts_holds_each_contact_at_its_mean():
    frames = 20
    target = np.zeros((frames, 2, 3), np.float32)
    target[:, 0, 0] = np.arange(frames)
    target[:, 0, 2] = 0.3
    target[:, 1, 0] = 7.0
    contact = np.zeros((frames, 1), bool)
    contact[2:5] = True
    contact[10:14] = True
    held = np.asarray(hold_contacts(jnp.asarray(target), jnp.asarray(contact), [0], blend=1))

    np.testing.assert_allclose(held[2:5, 0, 0], 3.0)
    np.testing.assert_allclose(held[10:14, 0, 0], 11.5)
    np.testing.assert_allclose(held[7, 0, 0], 7.0)
    np.testing.assert_allclose(held[:, 0, 2], 0.3)
    np.testing.assert_array_equal(held[:, 1], target[:, 1])


def test_solver_pulls_a_body_in_contact_onto_the_floor():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body name="root"><freejoint/><geom size="0.1"/><site name="root"/>
    </body></worldbody></mujoco>""")
    fk = compile_forward_kinematics(model, [0], [1], [np.zeros(3)])
    target_pos = jnp.tile(jnp.asarray([0.0, 0.0, 0.1]), (4, 1, 1))
    target_rot = jnp.tile(jnp.eye(3), (4, 1, 1, 1))
    start = jnp.asarray([[0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]])
    solve = make_solver(fk, model.nv, np.zeros(0), np.zeros(0), np.ones(1), np.ones(1),
                        SolverConfig(damping=1e-3), contact_groups=[[0]])

    free = solve(start, target_pos[None], target_rot[None])
    held = solve(start, target_pos[None], target_rot[None], jnp.ones((1, 4, 1)))
    assert float(free[0, -1, 2]) == pytest.approx(0.1, abs=1e-3)
    assert abs(float(held[0, -1, 2])) < 0.01


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
