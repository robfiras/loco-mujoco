"""Offline-reachable branches of ImitationFactory.

Most of imitation_factory.py downloads trajectories from HuggingFace or does
SMPL retargeting, which is not CI-safe. But a few branches never touch the
network and are worth pinning down:

* ``make`` rejecting an unregistered env name (KeyError),
* the "unknown dataset group" ``ValueError`` guards in ``get_amass_traj`` /
  ``get_lafan1_traj`` (raised *before* any download), and
* the whole ``get_custom_dataset`` path, which is just
  ``TrajectoryHandler.process`` on a user-supplied trajectory.

The custom path is driven with the synthetic ``standing_trajectory`` fixture and
a DummyHumamoidEnv, so no data is fetched.
"""
import numpy as np
import jax
import pytest

from test_conf import DummyHumamoidEnv
from test_conf import *  # noqa: F401,F403  (standing_trajectory fixture, etc.)

from loco_mujoco.task_factories import ImitationFactory
from loco_mujoco.task_factories.dataset_confs import (
    AMASSDatasetConf, LAFAN1DatasetConf, CustomDatasetConf,
)
from loco_mujoco.core.trajectory import Trajectory

jax.config.update('jax_platform_name', 'cpu')

DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}


def _dummy_env():
    return DummyHumamoidEnv(enable_mjx=False, goal_type="NoGoal",
                            reward_type="NoReward", **DEFAULTS)


def test_make_rejects_unregistered_env():
    with pytest.raises(KeyError):
        ImitationFactory.make("DefinitelyNotARegisteredEnv")


def test_get_amass_traj_unknown_group():
    env = _dummy_env()
    conf = AMASSDatasetConf(dataset_group="BOGUS_GROUP")
    with pytest.raises(ValueError):
        ImitationFactory.get_amass_traj(env, conf)


def test_get_lafan1_traj_unknown_group():
    env = _dummy_env()
    conf = LAFAN1DatasetConf(dataset_group="BOGUS_GROUP")
    with pytest.raises(ValueError):
        ImitationFactory.get_lafan1_traj(env, conf)


def test_get_custom_dataset(standing_trajectory):
    env = _dummy_env()
    conf = CustomDatasetConf(traj=standing_trajectory)

    traj = ImitationFactory.get_custom_dataset(env, conf)

    assert isinstance(traj, Trajectory)
    # the processed trajectory keeps finite qpos/qvel
    assert np.all(np.isfinite(np.asarray(traj.data.qpos)))
    assert np.all(np.isfinite(np.asarray(traj.data.qvel)))
