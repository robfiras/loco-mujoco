from copy import deepcopy
from functools import partial
from typing import Optional, Tuple, Union, Any

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct

from loco_mujoco.core import MjxState, Mjx
from loco_mujoco.core.utils.env import Box

#: Name of the vmap axis :class:`VecEnv` maps the environments over. Naming it lets any
#: code inside the traced region recover its own environment index with
#: ``jax.lax.axis_index(VECENV_AXIS_NAME)``, which is how ``AdditionalCarry.env_id`` is
#: populated. That index is what makes MjWarp's *global* contact buffer usable: the
#: buffer is shared by every world, and ``contact__worldid`` only becomes meaningful
#: once an environment knows which world it is.
VECENV_AXIS_NAME = "env"

#: Name for the OUTER vmap over seeds, applied by the training drivers (not here) when
#: ``n_seeds > 1``. MjWarp flattens nested batch axes into a single world dimension, so
#: with a seed axis present ``axis_index(VECENV_AXIS_NAME)`` alone no longer equals
#: ``worldid`` -- the two indices have to be combined. Drivers should import this rather
#: than hardcoding the string.
SEED_AXIS_NAME = "seed"


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

    """Append a rolling history of past observations to the current one.

    Two modes:

    * ``group=None`` (legacy): the WHOLE observation is stacked ``n_steps`` times, so
      the output is ``n_steps * obs_dim`` and every consumer -- actor and critic alike --
      sees a stacked vector. Kept for backward compatibility.
    * ``group="<name>"``: only that observation group is stacked, and only its PAST
      frames are appended. Layout::

          [ current full observation ][ group @ t-1 ][ t-2 ] ... [ t-(n_steps-1) ]

      The current frame stays at its original indices, so every existing index (actor,
      critic, per-term) keeps pointing at the right place and only the actor needs
      widening. That is what lets the critic stay un-stacked, matching mjlab, whose
      ``privileged_state`` group has no history.

    Note the history holds PAST frames only -- the current frame is already present in
    full at the head, so including it again would feed the network the same numbers
    twice. mjlab reaches the same arrangement differently, by pushing the current frame
    out of a 4-frame window with a one-step lag.
    """

    def __init__(self, env, n_steps, include_actions=False, group=None):
        super().__init__(env)
        self.n_steps = n_steps
        self.include_actions = include_actions
        self.group = group
        if group is not None:
            if include_actions:
                raise ValueError("include_actions is not supported with a group filter; "
                                 "put the action term in the stacked group instead.")
            # numpy, NOT jnp. _wrap_env runs inside _train_fn, which is pmapped, so
            # this __init__ and update_info execute during tracing -- a jnp array
            # created here is traced, and update_info's np.asarray on it raises
            # TracerArrayConversionError. Plain numpy indices sidestep that entirely and
            # index jax arrays just as well.
            self._group_ind = np.asarray(
                env.obs_container.get_obs_ind_by_group(group), dtype=np.int64)
            if self._group_ind.size == 0:
                raise ValueError(f"observation group {group!r} is empty -- nothing to "
                                 "stack. Check the group name in the observation spec.")
        self.info = self.update_info(env.info)

    def update_info(self, info):
        new_info = deepcopy(info)
        obs_dim = info.observation_space.high.shape[0]
        if self.group is not None:
            # current full obs + (n_steps - 1) past frames of the group only
            n_hist = (self.n_steps - 1) * int(self._group_ind.size)
            g_hi = np.asarray(info.observation_space.high)[self._group_ind]
            g_lo = np.asarray(info.observation_space.low)[self._group_ind]
            high = np.concatenate([info.observation_space.high,
                                   np.tile(g_hi, self.n_steps - 1)])
            low = np.concatenate([info.observation_space.low,
                                  np.tile(g_lo, self.n_steps - 1)])
            new_info.observation_space = Box(low, high)
            self._entry_dim = obs_dim
            self._n_hist = n_hist
            return new_info
        if self.include_actions:
            action_dim = info.action_space.shape[0]
            entry_dim = obs_dim + action_dim
            high = np.tile(np.concatenate([info.observation_space.high,
                                           np.full(action_dim, np.inf)]), self.n_steps)
            low = np.tile(np.concatenate([info.observation_space.low,
                                          np.full(action_dim, -np.inf)]), self.n_steps)
        else:
            entry_dim = obs_dim
            high = np.tile(info.observation_space.high, self.n_steps)
            low = np.tile(info.observation_space.low, self.n_steps)
        new_info.observation_space = Box(low, high)
        self._entry_dim = entry_dim
        return new_info

    def reset(self, rng_key, traj=None):
        obs, env_state = self.env.reset(rng_key, traj)
        if self.group is not None:
            # Seed every past slot with the CURRENT reading rather than zeros: at t=0
            # there is no history, and zeros would be a state the robot is never in.
            cur = obs[self._group_ind]
            observation_buffer = jnp.tile(cur, (self.n_steps - 1, 1))
            state = NStepWrapperState(env_state, observation_buffer)
            # Emitted NEWEST-FIRST: block i is t-(i+1). The buffer itself is
            # oldest-first (roll/append is cheapest that way), so flip on emit.
            # init_agent_conf's index math assumes newest-first.
            return jnp.concatenate(
                [obs, jnp.flip(observation_buffer, axis=0).reshape(-1)]), state
        if self.include_actions:
            action_dim = self.info.action_space.shape[0]
            entry = jnp.concatenate([obs, jnp.zeros(action_dim)])
        else:
            entry = obs
        observation_buffer = jnp.tile(jnp.zeros_like(entry), (self.n_steps, 1))
        observation_buffer = observation_buffer.at[-1].set(entry)
        state = NStepWrapperState(env_state, observation_buffer)
        obs = jnp.reshape(observation_buffer, (-1,))
        return obs, state

    def step(self, state: NStepWrapperState, action: Union[int, float], traj=None):

        # make a step
        next_observation, reward, absorbing, done, info, env_state = self.env.step(state.env_state, action, traj)

        if self.group is not None:
            # buffer holds t-1 .. t-(n_steps-1), oldest first; push the frame that is
            # about to stop being current.
            prev = state.observation_buffer
            cur = next_observation[self._group_ind]
            observation_buffer = jnp.roll(prev, shift=-1, axis=0).at[-1].set(cur)
            new_state = NStepWrapperState(env_state, observation_buffer)
            out = jnp.concatenate(
                [next_observation, jnp.flip(prev, axis=0).reshape(-1)])
            return out, reward, absorbing, done, info, new_state

        # build buffer entry: concat(obs, action) if include_actions else obs
        if self.include_actions:
            entry = jnp.concatenate([next_observation, action])
        else:
            entry = next_observation

        # add entry to the buffer
        observation_buffer = state.observation_buffer
        observation_buffer = jnp.roll(observation_buffer, shift=-1, axis=0)
        observation_buffer = observation_buffer.at[-1].set(entry)
        state = NStepWrapperState(env_state, observation_buffer)
        next_observation = jnp.reshape(observation_buffer, (-1,))

        return next_observation, reward, absorbing, done, info, state


class VecEnv(BaseWrapper):

    def __init__(self, env):
        super().__init__(env)
        # axis_name is what lets AdditionalCarry.env_id be filled in from inside the
        # vmap; without it there is no way for an environment to know which world it is,
        # which is exactly what reading MjWarp's shared contact buffer requires.
        self.reset = jax.vmap(self.env.reset, in_axes=(0, None),
                              axis_name=VECENV_AXIS_NAME)
        self.step = jax.vmap(self.env.step, in_axes=(0, 0, None),
                             axis_name=VECENV_AXIS_NAME)


# NormalizeVecReward has been removed. Reward normalization now lives on the
# agent state (see loco_mujoco.algorithms.common.dataclasses.RewardNormStats)
# and is applied inside the algorithm _train_fn, so stats survive env resets
# and persist across save/load.

