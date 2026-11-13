"""Tests for reading AMASS clips onto an SMPL-H skeleton, on a small made-up body model."""
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from loco_mujoco.retargeting import global_transforms
from loco_mujoco.retargeting.amass import (SMPL_JOINTS, SMPL_PARENTS, WORLD_FROM_SMPL, load_amass, place_on_floor,
                                           smpl_rest_pose, smpl_skeleton)

N_JOINTS = len(SMPL_JOINTS)


@pytest.fixture
def model_path(tmp_path):
    """Body model whose joints are its first 22 vertices, with more vertices below them."""
    rng = np.random.default_rng(0)
    vertices = rng.uniform([-0.5, 0.0, -0.2], [0.5, 1.6, 0.2], size=(30, 3))
    regressor = np.zeros((52, 30))
    regressor[np.arange(N_JOINTS), np.arange(N_JOINTS)] = 1.0
    path = tmp_path / "model.npz"
    np.savez(path, v_template=vertices, J_regressor=regressor, shapedirs=rng.normal(0, 0.01, (30, 3, 16)))
    return str(path)


def _clip(tmp_path, poses, trans, fps=120.0):
    path = tmp_path / "clip.npz"
    np.savez(path, poses=poses, trans=trans, mocap_framerate=np.array(fps), betas=np.zeros(16))
    return str(path)


def _smpl_joint_positions(joints, poses, trans):
    """Joint positions as SMPL computes them, rotating each bone about its parent joint."""
    rot = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(N_JOINTS, 3, 3)
    world_rot, world_pos = [None] * N_JOINTS, [None] * N_JOINTS
    for j, parent in enumerate(SMPL_PARENTS):
        if parent == -1:
            world_rot[j], world_pos[j] = rot[j], joints[j] + trans
        else:
            world_rot[j] = world_rot[parent] @ rot[j]
            world_pos[j] = world_pos[parent] + world_rot[parent] @ (joints[j] - joints[parent])
    return np.array(world_pos)


def test_smpl_skeleton_offsets_and_pelvis_height(model_path):
    skeleton, pelvis_height = smpl_skeleton(model_path)
    with np.load(model_path) as model:
        vertices = model["v_template"]
    joints = vertices[:N_JOINTS]
    assert skeleton.names == SMPL_JOINTS and skeleton.parents == SMPL_PARENTS
    np.testing.assert_allclose(skeleton.offsets[0], joints[0])
    for j, parent in enumerate(SMPL_PARENTS[1:], start=1):
        np.testing.assert_allclose(skeleton.offsets[j], joints[j] - joints[parent])
    assert pelvis_height == pytest.approx(joints[0, 1] - vertices[:, 1].min())


def test_smpl_skeleton_applies_betas(model_path):
    betas = np.linspace(-2.0, 2.0, 16)
    skeleton, _ = smpl_skeleton(model_path, betas)
    with np.load(model_path) as model:
        joints = (model["v_template"] + model["shapedirs"] @ betas)[:N_JOINTS]
    np.testing.assert_allclose(skeleton.offsets[4], joints[4] - joints[1])


def test_load_amass_matches_smpl_joint_positions(model_path, tmp_path):
    rng = np.random.default_rng(1)
    poses = np.concatenate([rng.normal(0, 0.5, (5, 3 * N_JOINTS)), np.zeros((5, 90))], axis=1)
    trans = rng.normal(0, 1.0, (5, 3))
    skeleton, _ = smpl_skeleton(model_path)
    motion = load_amass(_clip(tmp_path, poses, trans), skeleton)
    pos, _ = global_transforms(skeleton.parents, jnp.asarray(motion.local_pos), jnp.asarray(motion.local_rot),
                               range(N_JOINTS), np.eye(3))
    with np.load(model_path) as model:
        joints = model["v_template"][:N_JOINTS]
    for t in range(5):
        np.testing.assert_allclose(pos[t], _smpl_joint_positions(joints, poses[t, :3 * N_JOINTS], trans[t]), atol=1e-5)
    assert motion.fps == 120.0


def test_load_amass_resamples_by_slerp(model_path, tmp_path):
    poses = np.zeros((3, 156))
    poses[:, 2] = [0.0, 1.0, 2.0]
    trans = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    skeleton, _ = smpl_skeleton(model_path)
    motion = load_amass(_clip(tmp_path, poses, trans, fps=2.0), skeleton, fps=4.0)
    assert motion.fps == 4.0 and len(motion.local_pos) == 5
    np.testing.assert_allclose(motion.local_pos[:, 0, 0] - skeleton.offsets[0, 0], [0.0, 0.5, 1.0, 1.5, 2.0], atol=1e-9)
    angles = Rotation.from_matrix(motion.local_rot[:, 0]).as_rotvec()[:, 2]
    np.testing.assert_allclose(angles, [0.0, 0.5, 1.0, 1.5, 2.0], atol=1e-9)


def test_load_amass_rejects_nan(model_path, tmp_path):
    poses = np.zeros((3, 156))
    poses[1, 5] = np.nan
    skeleton, _ = smpl_skeleton(model_path)
    with pytest.raises(ValueError):
        load_amass(_clip(tmp_path, poses, np.zeros((3, 3))), skeleton)


def test_rest_pose_stands_on_the_sole(model_path):
    skeleton, pelvis_height = smpl_skeleton(model_path)
    local_pos, local_rot = smpl_rest_pose(skeleton, pelvis_height)
    pos, _ = global_transforms(skeleton.parents, jnp.asarray(local_pos), jnp.asarray(local_rot), range(N_JOINTS),
                               np.eye(3))
    with np.load(model_path) as model:
        vertices = model["v_template"]
    np.testing.assert_allclose(pos[:, 2], vertices[:N_JOINTS, 1] - vertices[:, 1].min(), atol=1e-6)


def test_place_on_floor_puts_the_feet_on_the_floor(model_path, tmp_path):
    skeleton, pelvis_height = smpl_skeleton(model_path)
    poses = np.zeros((4, 156))
    poses[:, :3] = Rotation.from_matrix(WORLD_FROM_SMPL).as_rotvec()
    trans = np.tile([0.2, -0.1, pelvis_height + 0.3], (4, 1)) - skeleton.offsets[0]
    motion = place_on_floor(load_amass(_clip(tmp_path, poses, trans), skeleton), pelvis_height)
    rest_pos, rest_rot = smpl_rest_pose(skeleton, pelvis_height)
    pos, _ = global_transforms(skeleton.parents, jnp.asarray(motion.local_pos), jnp.asarray(motion.local_rot),
                               range(N_JOINTS), np.eye(3))
    rest, _ = global_transforms(skeleton.parents, jnp.asarray(rest_pos), jnp.asarray(rest_rot), range(N_JOINTS),
                                np.eye(3))
    np.testing.assert_allclose(pos[..., 2], np.broadcast_to(rest[:, 2], pos.shape[:2]), atol=1e-5)
