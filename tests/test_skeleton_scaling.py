"""Tests for the body scaling and the arm-less variant of the skeleton environments."""
import jax
import mujoco
import numpy as np
import pytest

from loco_mujoco.environments.humanoids import SkeletonTorque, SkeletonMuscle

from conftest import _synthetic_traj_from_env

jax.config.update("jax_platform_name", "cpu")

_SCALING = 0.8
_KWARGS = dict(headless=True, goal_type="NoGoal", reward_type="NoReward")


@pytest.mark.parametrize("env_cls", (SkeletonTorque, SkeletonMuscle))
def test_scaling_follows_the_geometric_scaling_laws(env_cls):
    reference = env_cls(**_KWARGS)
    scaled = env_cls(scaling=_SCALING, **_KWARGS)
    m_ref, m_scaled = reference._model, scaled._model

    assert m_scaled.nq == m_ref.nq and m_scaled.nu == m_ref.nu
    # lengths scale with s, masses with s^3, inertias with s^5
    assert np.allclose(m_scaled.body_pos, _SCALING * m_ref.body_pos, atol=1e-6)
    assert np.allclose(m_scaled.body_mass, _SCALING ** 3 * m_ref.body_mass, rtol=1e-5)
    # the model's compiler clamps inertias from below (boundinertia), also after scaling
    has_mass = m_ref.body_mass > 0
    bound = m_ref.body_inertia[has_mass].min()
    assert np.allclose(m_scaled.body_inertia[has_mass],
                       np.maximum(_SCALING ** 5 * m_ref.body_inertia[has_mass], bound), rtol=1e-4)

    # forces scale with the cross-sectional area s^2: motor gears and peak muscle forces
    motors = m_ref.actuator_gaintype != mujoco.mjtGain.mjGAIN_MUSCLE
    assert np.allclose(m_scaled.actuator_gear[motors], _SCALING ** 2 * m_ref.actuator_gear[motors])
    assert np.allclose(m_scaled.actuator_gainprm[~motors, 2], _SCALING ** 2 * m_ref.actuator_gainprm[~motors, 2])
    assert np.allclose(m_scaled.actuator_biasprm[~motors, 2], _SCALING ** 2 * m_ref.actuator_biasprm[~motors, 2])
    assert (~motors).any() == (env_cls is SkeletonMuscle)


def test_scaled_skeleton_scales_the_root_translation_of_a_trajectory():
    reference = SkeletonTorque(**_KWARGS)
    scaled = SkeletonTorque(scaling=_SCALING, **_KWARGS)
    traj = _synthetic_traj_from_env(reference, n_steps=20)

    processed_ref = reference.process_trajectory(traj)
    processed_scaled = scaled.process_trajectory(traj)

    root = scaled.free_jnt_qpos_id[:, :3].reshape(-1)
    joints = np.setdiff1d(np.arange(scaled._model.nq), scaled.free_jnt_qpos_id.reshape(-1))
    assert np.allclose(processed_scaled.data.qpos[:, root], _SCALING * processed_ref.data.qpos[:, root], atol=1e-6)
    assert np.allclose(processed_scaled.data.qpos[:, joints], processed_ref.data.qpos[:, joints], atol=1e-6)
    # body positions are recomputed for the scaled model
    assert processed_scaled.data.xpos.shape == processed_ref.data.xpos.shape
    assert not np.allclose(processed_scaled.data.xpos, processed_ref.data.xpos)


def test_disable_arms_removes_arm_joints_and_motors():
    full = SkeletonTorque(**_KWARGS)
    armless = SkeletonTorque(disable_arms=True, **_KWARGS)

    assert armless._model.nq == full._model.nq - 14
    assert armless.info.action_space.shape[0] == full.info.action_space.shape[0] - 14
    assert armless.info.observation_space.shape[0] < full.info.observation_space.shape[0]
