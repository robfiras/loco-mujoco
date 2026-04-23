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

    Core columns: (obs, next_obs, action, reward, done). `value_target` is
    an optional column used by value-distillation-style critic learning
    (e.g. VanillaDagger stores `V_teacher(s)` there at collect time).

    Any of `next_obs` / `value_target` can be turned off at `create` time;
    disabled columns become zero-sized dummies (0 bytes). Flags are static
    metadata so `if self.store_*` branches are baked at trace time.
    """
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value_target: jnp.ndarray
    ptr: int
    size: int
    store_next_obs: bool = struct.field(pytree_node=False, default=True)
    store_value_target: bool = struct.field(pytree_node=False, default=False)

    @classmethod
    def create(cls, obs_dim: int, action_dim: int, capacity: int,
               store_next_obs: bool = True,
               store_value_target: bool = False):
        next_obs_shape = (capacity, obs_dim) if store_next_obs else (0, 0)
        value_shape = (capacity,) if store_value_target else (0,)
        return cls(
            obs=jnp.zeros((capacity, obs_dim)),
            next_obs=jnp.zeros(next_obs_shape),
            action=jnp.zeros((capacity, action_dim)),
            reward=jnp.zeros((capacity,)),
            done=jnp.zeros((capacity,), dtype=jnp.float32),
            value_target=jnp.zeros(value_shape),
            ptr=0,
            size=0,
            store_next_obs=store_next_obs,
            store_value_target=store_value_target,
        )

    def add_batch(self, obs, next_obs, action, reward, done, value_target=None):
        capacity = self.obs.shape[0]
        batch_size = obs.shape[0]
        indices = (self.ptr + jnp.arange(batch_size)) % capacity
        new_next_obs = (self.next_obs.at[indices].set(next_obs)
                        if self.store_next_obs else self.next_obs)
        if self.store_value_target:
            assert value_target is not None, (
                "buffer was created with store_value_target=True but "
                "add_batch received value_target=None"
            )
            new_value_target = self.value_target.at[indices].set(value_target)
        else:
            new_value_target = self.value_target
        return self.replace(
            obs=self.obs.at[indices].set(obs),
            next_obs=new_next_obs,
            action=self.action.at[indices].set(action),
            reward=self.reward.at[indices].set(reward),
            done=self.done.at[indices].set(done.astype(jnp.float32)),
            value_target=new_value_target,
            ptr=(self.ptr + batch_size) % capacity,
            size=jnp.minimum(self.size + batch_size, capacity),
        )

    def sample(self, rng, batch_size: int):
        indices = jax.random.randint(rng, (batch_size,), 0, self.size)
        if self.store_next_obs:
            next_obs = self.next_obs[indices]
        else:
            next_obs = jnp.zeros((batch_size, 0), dtype=self.obs.dtype)
        return (
            self.obs[indices],
            next_obs,
            self.action[indices],
            self.reward[indices],
            self.done[indices],
        )


@struct.dataclass
class ReservoirBuffer:
    """Reservoir-sampled long-term memory.

    Approximates a uniform sample of all transitions ever passed through
    `add_batch`. Capacity stays fixed; early additions fill sequentially,
    later additions probabilistically replace random slots. At steady
    state each global transition has probability `capacity / total_seen`
    of being in the buffer.

    Columns mirror `ReplayBuffer` (obs / action / value_target). No
    `next_obs` or `reward` / `done` — this is intended as a
    long-term BC + value-distillation store, not a TD buffer.

    `min_include_prob` optionally floors the per-item inclusion rate. The
    pure reservoir's rate decays as k/n indefinitely, which is correct
    for stationary streams but wastes signal when the data distribution
    drifts (e.g. DAgger rollouts shift as the student learns). Setting
    `min_include_prob > 0` caps the effective n at `ceil(k / p_min)`, so
    the buffer behaves like a uniform sample over the last `k / p_min`
    transitions once that window has filled. Default 0.0 = pure reservoir.
    """
    obs: jnp.ndarray
    action: jnp.ndarray
    value_target: jnp.ndarray
    total_seen: jnp.ndarray       # scalar int32, counts all transitions ever added
    size: jnp.ndarray             # scalar int32, min(total_seen, capacity)
    store_value_target: bool = struct.field(pytree_node=False, default=False)
    min_include_prob: float = struct.field(pytree_node=False, default=0.0)

    @classmethod
    def create(cls, obs_dim: int, action_dim: int, capacity: int,
               store_value_target: bool = False,
               min_include_prob: float = 0.0):
        value_shape = (capacity,) if store_value_target else (0,)
        return cls(
            obs=jnp.zeros((capacity, obs_dim)),
            action=jnp.zeros((capacity, action_dim)),
            value_target=jnp.zeros(value_shape),
            total_seen=jnp.zeros((), dtype=jnp.int32),
            size=jnp.zeros((), dtype=jnp.int32),
            store_value_target=store_value_target,
            min_include_prob=float(min_include_prob),
        )

    def add_batch(self, obs, action, value_target, rng):
        """Reservoir-insert a batch. Vectorised: for each element i with
        global index g_i = total_seen + i, draw j_i ~ uniform_int[0, n_eff_i)
        where `n_eff_i = min(g_i+1, ceil(capacity / min_include_prob))`.
        Write at slot g_i if we're still in the fill phase, else write at
        j_i when j_i < capacity. Collisions between two include=True
        elements writing to the same slot are resolved by whichever wins
        the scatter — a tiny bias but costs nothing.
        """
        import math
        capacity = self.obs.shape[0]
        B = obs.shape[0]
        batch_idx = self.total_seen + jnp.arange(B, dtype=jnp.int32)

        # Effective denominator for the reservoir decision. A positive
        # min_include_prob caps it at ceil(k/p_min), flooring the
        # inclusion probability at p_min. Python-level branch — static.
        if self.min_include_prob > 0.0:
            cap_n = int(math.ceil(capacity / self.min_include_prob))
            n_eff = jnp.minimum(batch_idx + 1, jnp.int32(cap_n))
        else:
            n_eff = batch_idx + 1

        u = jax.random.uniform(rng, (B,))
        j = jnp.floor(u * n_eff.astype(jnp.float32)).astype(jnp.int32)

        fill = batch_idx < capacity
        include = fill | (j < capacity)
        slot = jnp.where(fill, batch_idx, j)
        slot = jnp.clip(slot, 0, capacity - 1)

        # Masked scatter: for non-include entries, write back the current
        # slot contents (no-op). Collisions between include=True entries
        # are rare for capacity >> batch_size.
        fresh_obs = jnp.where(include[:, None], obs, self.obs[slot])
        fresh_action = jnp.where(include[:, None], action, self.action[slot])
        new_obs = self.obs.at[slot].set(fresh_obs)
        new_action = self.action.at[slot].set(fresh_action)

        if self.store_value_target:
            assert value_target is not None, (
                "reservoir was created with store_value_target=True but "
                "add_batch received value_target=None"
            )
            fresh_vt = jnp.where(include, value_target, self.value_target[slot])
            new_vt = self.value_target.at[slot].set(fresh_vt)
        else:
            new_vt = self.value_target

        new_total = self.total_seen + B
        new_size = jnp.minimum(jnp.int32(capacity), new_total)
        return self.replace(
            obs=new_obs,
            action=new_action,
            value_target=new_vt,
            total_seen=new_total,
            size=new_size,
        )

    def sample(self, rng, batch_size: int):
        indices = jax.random.randint(rng, (batch_size,), 0, jnp.maximum(1, self.size))
        return (
            self.obs[indices],
            self.action[indices],
            (self.value_target[indices] if self.store_value_target
             else jnp.zeros((batch_size,), dtype=self.obs.dtype)),
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
