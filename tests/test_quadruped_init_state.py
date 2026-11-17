"""The quadrupeds' initial state: their standing pose by default, the caller's handler and parameters otherwise."""
import jax
import numpy as np
import pytest

from loco_mujoco.environments.base import LocoEnv
from loco_mujoco.task_factories import RLFactory

jax.config.update("jax_platform_name", "cpu")

QUADRUPEDS = ["UnitreeA1", "UnitreeGo2", "BDSpot"]


@pytest.mark.parametrize("name", QUADRUPEDS)
def test_rl_factory_starts_in_the_standing_pose(name):
    # RLFactory passes init_state_type="DefaultInitialStateHandler" without parameters
    env, _ = RLFactory.make(name)
    env.reset(jax.random.PRNGKey(0))
    np.testing.assert_allclose(env._data.qpos, env.init_qpos)
    np.testing.assert_allclose(env._data.qvel, env.init_qvel)


@pytest.mark.parametrize("name", QUADRUPEDS)
def test_the_callers_init_state_is_kept(name):
    env_cls = LocoEnv.registered_envs[name]
    # e.g. ImitationFactory's default
    env = env_cls(init_state_type="TrajInitialStateHandler")
    assert type(env._init_state_handler).__name__ == "TrajInitialStateHandler"

    qpos = np.array(env.init_qpos)
    qpos[2] += 0.1
    env = env_cls(init_state_type="DefaultInitialStateHandler", init_state_params=dict(qpos_init=qpos))
    env.reset(jax.random.PRNGKey(0))
    np.testing.assert_allclose(env._data.qpos, qpos)
