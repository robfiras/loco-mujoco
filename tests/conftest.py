"""Shared pytest fixtures.

The imitation-learning tests (GAIL / AMP) normally fetch a *default* expert
trajectory from the HuggingFace Hub via
``ImitationFactory.get_default_traj``.  That download makes the test-suite
depend on network access, which is unavailable in CI sandboxes (and slow /
flaky everywhere else).

The ``offline_default_traj`` fixture below is ``autouse`` so that, for every
test, ``get_default_traj`` is replaced by a routine that *synthesises* a
structurally-valid expert trajectory directly from the environment's own
MuJoCo model.  Because the trajectory is built from ``env._model`` the joint /
body / site layout matches the environment exactly, so it flows through
``env.process_trajectory`` and the mimic goals/rewards just like a real
downloaded dataset would -- only offline and deterministically.

Numerical realism is intentionally *not* a goal here: the synthetic clip is a
single forward-kinematics frame tiled over time (a "balance"/standing clip).
That is sufficient to exercise the algorithm build/compile/save/load code
paths, which is what these tests assert.
"""
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from loco_mujoco.core.trajectory import (
    Trajectory,
    TrajectoryInfo,
    TrajectoryModel,
    TrajectoryData,
)


def _synthetic_traj_from_env(env, n_steps: int = 200) -> Trajectory:
    """Build a static (standing) trajectory from an env's MuJoCo model."""
    model = env._model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def names(obj_type, n):
        return [mujoco.mj_id2name(model, obj_type, i) for i in range(n)]

    joint_names = names(mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    body_names = names(mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    site_names = names(mujoco.mjtObj.mjOBJ_SITE, model.nsite)

    traj_model = TrajectoryModel(
        model.njnt,
        np.array(model.jnt_type),
        model.nbody,
        np.array(model.body_rootid),
        np.array(model.body_weldid),
        np.array(model.body_mocapid),
        np.array(model.body_pos),
        np.array(model.body_quat),
        np.array(model.body_ipos),
        np.array(model.body_iquat),
        model.nsite,
        np.array(model.site_bodyid),
        np.array(model.site_pos),
        np.array(model.site_quat),
    )

    frequency = 1.0 / env.dt
    info = TrajectoryInfo(joint_names, traj_model, frequency, body_names, site_names)

    def tile(x):
        x = np.asarray(x)
        return jnp.array(np.tile(x, (n_steps,) + (1,) * x.ndim))

    # guard against degenerate (zero-norm) quaternions from an unposed model
    xquat = np.asarray(data.xquat).copy()
    zero = np.linalg.norm(xquat, axis=-1) == 0
    xquat[zero] = np.array([1.0, 0.0, 0.0, 0.0])

    traj_data = TrajectoryData(
        tile(data.qpos),
        tile(data.qvel),
        tile(data.xpos),
        tile(xquat),
        tile(data.cvel),
        tile(data.subtree_com),
        tile(data.site_xpos),
        tile(data.site_xmat),
        split_points=jnp.array([0, n_steps]),
    )
    return Trajectory(info=info, data=traj_data)


@pytest.fixture(autouse=True)
def offline_default_traj(monkeypatch):
    """Replace the HuggingFace default-dataset download with a local synth."""
    from loco_mujoco.task_factories.imitation_factory import ImitationFactory

    def _fake_get_default_traj(env, default_dataset_conf):
        tasks = default_dataset_conf.task
        if isinstance(tasks, str):
            tasks = [tasks]
        trajs = [_synthetic_traj_from_env(env) for _ in tasks]
        return Trajectory.concatenate(trajs) if len(trajs) > 1 else trajs[0]

    monkeypatch.setattr(
        ImitationFactory, "get_default_traj", staticmethod(_fake_get_default_traj)
    )
