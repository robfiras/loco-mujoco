from copy import deepcopy
from functools import partial
from typing import Optional, Tuple, Union, Any

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct

from loco_mujoco.core import MjxState, Mjx
from loco_mujoco.core.utils.env import Box


class LocoMjxWrapper:

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        """
        Allow proxy access to regular attributes of Mjx env.
        """
        return getattr(self.env, name)

    def reset(self, rng_key, traj=None):
        state = self.env.mjx_reset(rng_key, traj)
        return state.observation, state

    def step(self, state, action, traj=None):
        next_state = self.env.mjx_step(state, action, traj)
        next_obs = jnp.where(next_state.done, next_state.additional_carry.final_observation, next_state.observation)
        return (next_obs, next_state.reward, next_state.absorbing, next_state.done,
                next_state.info, next_state)


@struct.dataclass
class BaseWrapperState:

    def __getattr__(self, name):
        """
        Allow proxy access to all attributes of all States.
        """
        try:
            if name in self.__dict__.keys():
                return self.__dict__[name]
            else:
                return getattr(self.env_state, name)
        except AttributeError as e:
            raise AttributeError(f"Attribute '{name}' not found in any env state nor the MjxState.") from e

    def find(self, cls):

        if isinstance(self, cls):
            return self
        elif isinstance(self.env_state, MjxState) and cls != MjxState:
            raise AttributeError(f"Class '{cls}' not found")
        else:
            return self.env_state.find(cls)


class BaseWrapper:

    def __init__(self, env):
        # if it's the bare Mjx class, wrap it in the LocoMjxWrapper first
        if issubclass(env.__class__, Mjx):
            self.env = LocoMjxWrapper(env)
        else:
            self.env = env

    def reset(self, rng_key, traj=None):
        return self.env.reset(rng_key, traj)

    def step(self, state, action, traj=None):
        return self.env.step(state, action, traj)

    def __getattr__(self, name):

        return getattr(self.env, name)

    def find_attr(self, state, attr_name):
        # Recursively search for the attribute
        if hasattr(state, attr_name):
            return getattr(state, attr_name)

        # If the attribute is not found, check env_state recursively
        if hasattr(state, 'env_state') and state.env_state is not None:
            return self.find_attr(state.env_state, attr_name)

        # If the attribute or env_state isn't found
        raise AttributeError(f"Attribute '{attr_name}' not found")

    def unwrapped(self):
        # find first env which is not a subclass of BaseWrapper
        if isinstance(self.env, BaseWrapper):
            return self.env.unwrapped()
        else:
            return self.env.env

@struct.dataclass
class SummaryMetrics:
    mean_episode_return: float = 0.0
    mean_episode_length: float = 0.0
    max_timestep: int = 0.0


@struct.dataclass
class Metrics:
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int
    done: bool


@struct.dataclass
class LogEnvState(BaseWrapperState):
    env_state: MjxState
    metrics: Metrics


class LogWrapper(BaseWrapper):
    """Log the episode returns and lengths."""

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng_key, traj=None):
        obs, env_state = self.env.reset(rng_key, traj)
        state = LogEnvState(env_state, metrics=Metrics(0, 0, 0, 0, 0, False))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: LogEnvState, action: Union[int, float], traj=None):

        # make a step
        next_observation, reward, absorbing, done, info, env_state = self.env.step(state.env_state, action, traj)

        new_episode_return = state.metrics.episode_returns + reward
        new_episode_length = state.metrics.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            metrics=Metrics(
                episode_returns=new_episode_return * (1 - done),
                episode_lengths=new_episode_length * (1 - done),
                returned_episode_returns=state.metrics.returned_episode_returns * (1 - done)
                                         + new_episode_return * done,
                returned_episode_lengths=state.metrics.returned_episode_lengths * (1 - done)
                                         + new_episode_length * done,
                timestep=state.metrics.timestep + 1,
                done=done,),
        )
        return next_observation, reward, absorbing, done, info, state


@struct.dataclass
class NStepWrapperState(BaseWrapperState):
    env_state: MjxState
    observation_buffer: jnp.ndarray


class NStepWrapper(BaseWrapper):

    def __init__(self, env, n_steps):
        super().__init__(env)
        self.n_steps = n_steps
        self.info = self.update_info(env.info)

    def update_info(self, info):
        new_info = deepcopy(info)
        high = np.tile(info.observation_space.high, self.n_steps)
        low = np.tile(info.observation_space.low, self.n_steps)
        observation_space = Box(low, high)
        new_info.observation_space = observation_space
        return new_info

    def reset(self, rng_key, traj=None):
        obs, env_state = self.env.reset(rng_key, traj)
        observation_buffer = jnp.tile(jnp.zeros_like(obs), (self.n_steps, 1))
        observation_buffer = observation_buffer.at[-1].set(obs)
        state = NStepWrapperState(env_state, observation_buffer)
        obs = jnp.reshape(observation_buffer, (-1,))
        return obs, state

    def step(self, state: NStepWrapperState, action: Union[int, float], traj=None):

        # make a step
        next_observation, reward, absorbing, done, info, env_state = self.env.step(state.env_state, action, traj)

        # add observation to the buffer
        observation_buffer = state.observation_buffer
        observation_buffer = jnp.roll(observation_buffer, shift=-1, axis=0)
        observation_buffer = observation_buffer.at[-1].set(next_observation)
        state = NStepWrapperState(env_state, observation_buffer)
        next_observation = jnp.reshape(observation_buffer, (-1,))

        return next_observation, reward, absorbing, done, info, state


class VecEnv(BaseWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self.env.reset, in_axes=(0, None))
        self.step = jax.vmap(self.env.step, in_axes=(0, 0, None))


@struct.dataclass
class NormalizeVecRewEnvState(BaseWrapperState):
    env_state: MjxState
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float


class NormalizeVecReward(BaseWrapper):

    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, traj=None):
        obs, state = self.env.reset(key, traj)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, state, action, traj=None):
        next_observation, reward, absorbing, done, info, env_state = self.env.step(state.env_state, action, traj)

        return_val = (state.return_val * self.gamma * (1 - done) + reward)

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = next_observation.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )

        return next_observation, reward / jnp.sqrt(state.var + 1e-8), absorbing, done, info, state

