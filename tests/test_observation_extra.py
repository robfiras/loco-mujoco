"""Extra coverage for core/observations/base.py.

Two clusters were left uncovered by the existing observation tests:

1. ``ObservationContainer`` bookkeeping helpers (equality, group filtering,
   randomizable-index gathering, duplicate/locked guards) - pure logic, no sim.
2. Several observation *types* that no default env spec includes: ``LastAction``,
   ``ModelInfo``, ``Force`` (collision force) and ``RelativeSiteQuantaties``.
   Adding them to a custom observation spec and stepping the env drives their
   ``_init_from_mj`` / ``get_obs_and_update_state`` in both backends.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from test_conf import DummyHumamoidEnv
from test_conf import *  # noqa: F401,F403
from loco_mujoco.core.observations import ObservationType
from loco_mujoco.core.observations.base import ObservationContainer

jax.config.update('jax_platform_name', 'cpu')

DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}


class ExtraObsDummyEnv(DummyHumamoidEnv):
    """Dummy env whose observation spec additionally exposes the obs types that
    no shipped env includes, so their code paths get exercised."""

    @staticmethod
    def _get_observation_specification(spec):
        obs = DummyHumamoidEnv._get_observation_specification(spec)
        obs += [
            ObservationType.LastAction("last_action_obs"),
            ObservationType.ModelInfo("model_info_obs", model_attributes="body_mass"),
            ObservationType.Force("grf_obs", xml_name_geom1="right_foot", xml_name_geom2="floor"),
            ObservationType.RelativeSiteQuantaties("rel_sites_obs"),
        ]
        return obs


# --------------------------------------------------------------------------- #
# 1. ObservationContainer bookkeeping helpers (pure logic)
# --------------------------------------------------------------------------- #
def test_observation_container_helpers():
    env = DummyHumamoidEnv(enable_mjx=False, goal_type="NoGoal", reward_type="NoReward", **DEFAULTS)
    container = env.obs_container

    # names()/entries() are thin aliases of keys()/values()
    assert list(container.names()) == list(container.keys())
    assert list(container.entries()) == list(container.values())

    # group helpers
    groups = container.get_all_group_names()
    assert isinstance(groups, list)

    randomizable = container.get_randomizable_obs_indices()
    assert randomizable.dtype == int

    # __eq__: identical container is equal; different types / key-sets are not
    assert container == container
    assert not (container == "not-a-container")
    empty = ObservationContainer()
    assert not (container == empty)


def test_observation_container_group_filter():
    env = DummyHumamoidEnv(enable_mjx=False, goal_type="NoGoal", reward_type="NoReward", **DEFAULTS)
    container = env.obs_container
    key = jax.random.PRNGKey(0)
    obs = env.reset(key)

    groups = container.get_all_group_names()
    # a group present on at least one obs -> non-empty indices + filtered slice
    for g in groups:
        if g is not None:
            ind = container.get_obs_ind_by_group(g)
            filtered = container.filter_by_group(obs, g)
            assert filtered.shape[-1] == len(ind)
            break

    # unknown group -> empty index array
    assert len(container.get_obs_ind_by_group("__no_such_group__")) == 0


def test_observation_container_guards():
    # a *real* env locks its container after setup, so build a fresh (unlocked)
    # one to exercise both the happy-path insert and the duplicate/locked guards.
    env = DummyHumamoidEnv(enable_mjx=False, goal_type="NoGoal", reward_type="NoReward", **DEFAULTS)
    sample_obs = list(env.obs_container.values())[0]

    container = ObservationContainer()
    container[sample_obs.name] = sample_obs  # happy-path insert (unlocked)
    assert sample_obs.name in container

    # duplicate key rejected
    with pytest.raises(KeyError):
        container[sample_obs.name] = sample_obs

    # locked container rejects any mutation
    container._locked = True
    with pytest.raises(ValueError):
        container["some_new_key"] = sample_obs


# --------------------------------------------------------------------------- #
# 2. Extra observation types
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_extra_observation_types(backend):
    key = jax.random.PRNGKey(0)
    env = ExtraObsDummyEnv(enable_mjx=True, goal_type="NoGoal", reward_type="NoReward", **DEFAULTS)

    # all four extra obs types are present
    for name in ["last_action_obs", "model_info_obs", "grf_obs", "rel_sites_obs"]:
        assert name in env.obs_container

    if backend == "numpy":
        obs = env.reset(key)
        obs2, *_ = env.step(np.zeros(env.info.action_space.shape[0]))
        assert np.all(np.isfinite(np.asarray(obs)))
        assert np.all(np.isfinite(np.asarray(obs2)))
    else:
        state = env.mjx_reset(key)
        assert jnp.all(jnp.isfinite(state.observation))
        state = env.mjx_step(state, jnp.zeros(env.info.action_space.shape[0]))
        assert jnp.all(jnp.isfinite(state.observation))
