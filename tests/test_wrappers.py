"""Tests for the observation-history wrapper (NStepWrapper) of the MJX backend."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from loco_mujoco.core.wrappers.mjx import LocoMjxWrapper, NStepWrapper

from test_conf import DummyHumamoidEnv

jax.config.update("jax_platform_name", "cpu")

_N_STEPS = 3
_GROUP = "history"
_GROUP_TERMS = ("q_root", "dq_root")


def _make_env(with_group=False):
    env = DummyHumamoidEnv(enable_mjx=True, horizon=10, gamma=0.99, n_envs=1)
    if with_group:
        for name in _GROUP_TERMS:
            env.obs_container[name].group = [_GROUP]
    return env


def _rollout(wrapper, n=2):
    """Resets the wrapper and steps it n times with constant, distinct actions."""
    obs, state = wrapper.reset(jax.random.PRNGKey(0))
    out = [obs]
    actions = []
    for i in range(n):
        action = jnp.full(wrapper.info.action_space.shape, 0.1 * (i + 1))
        obs, _, _, _, _, state = wrapper.step(state, action)
        out.append(obs)
        actions.append(action)
    return out, actions, state


def test_nstep_stacks_whole_observation():
    env = _make_env()
    obs_dim = env.info.observation_space.shape[0]
    wrapper = NStepWrapper(LocoMjxWrapper(env), n_steps=_N_STEPS)

    assert wrapper.info.observation_space.shape == (_N_STEPS * obs_dim,)
    assert env.info.observation_space.shape == (obs_dim,)   # the wrapped info is not modified

    (obs_0, obs_1, obs_2), _, state = _rollout(wrapper)
    frames = [o.reshape(_N_STEPS, obs_dim) for o in (obs_0, obs_1, obs_2)]

    # reset: no history yet, the current frame sits in the last slot
    assert jnp.all(frames[0][:-1] == 0.0)
    assert jnp.allclose(frames[0][-1], state.observation_buffer[0])

    # every step shifts the window by one frame, oldest first
    assert jnp.allclose(frames[1][-2], frames[0][-1])
    assert jnp.allclose(frames[2][-3], frames[0][-1])
    assert jnp.allclose(frames[2][-2], frames[1][-1])
    assert jnp.allclose(frames[2], state.observation_buffer)


def test_nstep_include_actions_appends_the_action_to_every_frame():
    env = _make_env()
    obs_dim = env.info.observation_space.shape[0]
    action_dim = env.info.action_space.shape[0]
    wrapper = NStepWrapper(LocoMjxWrapper(env), n_steps=_N_STEPS, include_actions=True)

    entry_dim = obs_dim + action_dim
    assert wrapper.info.observation_space.shape == (_N_STEPS * entry_dim,)
    high = wrapper.info.observation_space.high.reshape(_N_STEPS, entry_dim)
    assert np.all(np.isinf(high[:, obs_dim:]))

    (obs_0, obs_1, obs_2), actions, _ = _rollout(wrapper)
    assert jnp.all(obs_0.reshape(_N_STEPS, entry_dim)[-1, obs_dim:] == 0.0)   # no action at reset
    last = obs_2.reshape(_N_STEPS, entry_dim)
    assert jnp.allclose(last[-1, obs_dim:], actions[1])
    assert jnp.allclose(last[-2, obs_dim:], actions[0])


def test_nstep_group_appends_only_past_frames_of_the_group():
    env = _make_env(with_group=True)
    obs_dim = env.info.observation_space.shape[0]
    group_ind = np.asarray(env.obs_container.get_obs_ind_by_group(_GROUP), dtype=int)
    wrapper = NStepWrapper(LocoMjxWrapper(env), n_steps=_N_STEPS, group=_GROUP)

    n_hist = (_N_STEPS - 1) * group_ind.size
    assert wrapper.info.observation_space.shape == (obs_dim + n_hist,)
    assert np.array_equal(wrapper.info.observation_space.high[obs_dim:],
                          np.tile(env.info.observation_space.high[group_ind], _N_STEPS - 1))

    (obs_0, obs_1, obs_2), _, _ = _rollout(wrapper)
    current = [o[:obs_dim] for o in (obs_0, obs_1, obs_2)]
    history = [o[obs_dim:].reshape(_N_STEPS - 1, group_ind.size) for o in (obs_0, obs_1, obs_2)]

    # reset: every past slot is seeded with the current reading of the group
    assert jnp.allclose(history[0], jnp.tile(current[0][group_ind], (_N_STEPS - 1, 1)))
    # afterwards the history is newest first: block 0 is t-1, block 1 is t-2
    assert jnp.allclose(history[2][0], current[1][group_ind])
    assert jnp.allclose(history[2][1], current[0][group_ind])


def test_nstep_group_rejects_invalid_configurations():
    env = _make_env(with_group=True)
    with pytest.raises(ValueError, match="include_actions is not supported"):
        NStepWrapper(LocoMjxWrapper(env), n_steps=_N_STEPS, include_actions=True, group=_GROUP)
    with pytest.raises(ValueError, match="is empty"):
        NStepWrapper(LocoMjxWrapper(env), n_steps=_N_STEPS, group="no_such_group")
