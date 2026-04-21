from typing import NamedTuple
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from flax.training import train_state

from loco_mujoco.environments.base import TrajState
from loco_mujoco.core.wrappers.mjx import Metrics


class Transition(NamedTuple):
    done: jnp.ndarray
    absorbing: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    traj_state: TrajState
    metrics: Metrics


class MetricHandlerTransition(NamedTuple):
    env_state: Any
    logged_metrics: Metrics


class TrainState(train_state.TrainState):
    run_stats: Any


@struct.dataclass
class TrainStateBuffer:
    train_states: TrainState
    n: int
    size: int   # buffer size

    @classmethod
    def create(cls, train_state: TrainState, size: int):
        return TrainStateBuffer(
            train_states=jax.tree.map(lambda x: jnp.stack([x] * size), train_state),
            n=0,
            size=size
        )

    @classmethod
    def add(cls, train_state_buffer, train_state: TrainState):
        index = train_state_buffer.n
        # Add the new train state at index n
        train_states_updated = jax.tree.map(
            lambda buffer, new: buffer.at[index].set(new),
            train_state_buffer.train_states,
            train_state
        )
        return train_state_buffer.replace(
            train_states=train_states_updated,
            n=index + 1,
        )


@struct.dataclass
class RewardNormStats:
    """
    Running statistics for PPO-style reward normalization. Lives on the agent
    state so it survives env resets and is saved with the agent.
    """
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray
    return_val: jnp.ndarray

    @classmethod
    def create(cls, num_envs: int):
        return cls(
            mean=jnp.zeros(()),
            var=jnp.ones(()),
            count=jnp.full((), 1e-4),
            return_val=jnp.zeros((num_envs,)),
        )


def update_reward_norm(stats: "RewardNormStats", reward: jnp.ndarray,
                       done: jnp.ndarray, gamma: float):
    """
    Update running stats with a batch of rewards and return the normalized
    reward plus the new stats. Mirrors the previous NormalizeVecReward wrapper.
    """
    return_val = stats.return_val * gamma * (1.0 - done) + reward

    batch_mean = jnp.mean(return_val, axis=0)
    batch_var = jnp.var(return_val, axis=0)
    batch_count = return_val.shape[0]

    delta = batch_mean - stats.mean
    tot_count = stats.count + batch_count

    new_mean = stats.mean + delta * batch_count / tot_count
    m_a = stats.var * stats.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * stats.count * batch_count / tot_count
    new_var = M2 / tot_count

    new_stats = RewardNormStats(
        mean=new_mean,
        var=new_var,
        count=tot_count,
        return_val=return_val,
    )
    normalized_reward = reward / jnp.sqrt(new_stats.var + 1e-8)
    return normalized_reward, new_stats


@struct.dataclass
class ReplayBuffer:
    """Simple circular replay buffer for off-policy / imitation algorithms.
    Stores (obs, next_obs, action, reward, done) batches and samples uniformly.
    """
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    ptr: int
    size: int

    @classmethod
    def create(cls, obs_dim: int, action_dim: int, capacity: int):
        return cls(
            obs=jnp.zeros((capacity, obs_dim)),
            next_obs=jnp.zeros((capacity, obs_dim)),
            action=jnp.zeros((capacity, action_dim)),
            reward=jnp.zeros((capacity,)),
            done=jnp.zeros((capacity,), dtype=jnp.float32),
            ptr=0,
            size=0,
        )

    def add_batch(self, obs, next_obs, action, reward, done):
        capacity = self.obs.shape[0]
        batch_size = obs.shape[0]
        indices = (self.ptr + jnp.arange(batch_size)) % capacity
        return self.replace(
            obs=self.obs.at[indices].set(obs),
            next_obs=self.next_obs.at[indices].set(next_obs),
            action=self.action.at[indices].set(action),
            reward=self.reward.at[indices].set(reward),
            done=self.done.at[indices].set(done.astype(jnp.float32)),
            ptr=(self.ptr + batch_size) % capacity,
            size=jnp.minimum(self.size + batch_size, capacity),
        )

    def sample(self, rng, batch_size: int):
        indices = jax.random.randint(rng, (batch_size,), 0, self.size)
        return (
            self.obs[indices],
            self.next_obs[indices],
            self.action[indices],
            self.reward[indices],
            self.done[indices],
        )


@struct.dataclass
class BestTrainStates:
    train_states: TrainState
    metrics: jnp.array
    iterations: jnp.array
    cur_worst_perf: float
    step: int
    n: int
    size: int

    @classmethod
    def create(cls, train_state: TrainState, n: int):
        return BestTrainStates(
            train_states=jax.tree.map(lambda x: jnp.stack([x] * n), train_state),
            metrics=jnp.full((n,), -jnp.inf),
            iterations=jnp.zeros((n,)),
            cur_worst_perf=-jnp.inf,
            n=n,
            size=0
        )
