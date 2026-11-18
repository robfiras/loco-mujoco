"""
Data containers shared by the learning algorithms.

Contains the rollout transition tuples, the Flax train state with running statistics, running reward
normalization, and the replay and reservoir buffers used by the off-policy and distillation algorithms. All
containers are JAX pytrees and can be carried through ``jax.lax.scan``.

"""

from typing import NamedTuple
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from flax.training import train_state

from loco_mujoco.environments.base import TrajState
from loco_mujoco.core.wrappers.mjx import Metrics


class Transition(NamedTuple):
    """
    One step of on-policy experience, as collected by PPO and the algorithms built on it.

    Inside a rollout every field has a leading ``(num_steps, n_envs)`` shape.

    Attributes:
        done (jax.Array): Whether the episode ended at this step (terminal state or horizon reached).
        absorbing (jax.Array): Whether the state reached at this step is absorbing (terminal); no value is
            bootstrapped from absorbing states.
        action (jax.Array): Action taken, shape ``(action_dim,)``.
        value (jax.Array): Critic estimate for ``obs``.
        reward (jax.Array): Reward received (normalized if reward normalization is enabled).
        log_prob (jax.Array): Log probability of ``action`` under the policy that collected the data.
        obs (jax.Array): Observation the action was computed from, shape ``(obs_dim,)``.
        info (dict): Info dictionary returned by the environment step.
        traj_state (TrajState): Trajectory state of the environment after the step.
        metrics (Metrics): Episode statistics of the :class:`~loco_mujoco.core.wrappers.mjx.LogWrapper`.

    """
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
    """
    One step of a validation rollout, as consumed by :class:`~loco_mujoco.utils.metrics.MetricsHandler`.

    Attributes:
        env_state: Full (wrapped) environment state after the step.
        logged_metrics (Metrics): Episode statistics of the :class:`~loco_mujoco.core.wrappers.mjx.LogWrapper`.

    """
    env_state: Any
    logged_metrics: Metrics


class TrainState(train_state.TrainState):
    """
    Flax train state extended by the running statistics of the network.

    Attributes:
        step (int): Number of gradient steps applied so far.
        apply_fn (Callable): The ``apply`` function of the network.
        params: Network parameters.
        tx (optax.GradientTransformation): Optimizer.
        opt_state: Optimizer state.
        run_stats: The ``run_stats`` variable collection of the network, i.e. the running mean, variance and count
            of the :class:`~loco_mujoco.algorithms.common.networks.RunningMeanStd` layers.

    """
    run_stats: Any


@struct.dataclass
class TrainStateBuffer:
    """
    Fixed-size buffer of train states, used to keep snapshots taken during training.

    Attributes:
        train_states (TrainState): Train-state pytree whose leaves have a leading axis of length ``size``.
        n (int): Number of train states added so far, which is also the index the next one is written to.
        size (int): Capacity of the buffer.

    """
    train_states: TrainState
    n: int
    size: int   # buffer size

    @classmethod
    def create(cls, train_state: TrainState, size: int):
        """
        Create a buffer whose slots are all initialized with copies of ``train_state``.

        Args:
            train_state (TrainState): Train state used as template and initial content of every slot.
            size (int): Capacity of the buffer.

        Returns:
            TrainStateBuffer: The new buffer with ``n = 0``.

        """
        return TrainStateBuffer(
            train_states=jax.tree.map(lambda x: jnp.stack([x] * size), train_state),
            n=0,
            size=size
        )

    @classmethod
    def add(cls, train_state_buffer, train_state: TrainState):
        """
        Write a train state to the next free slot.

        There is no bounds check; the caller must not add more than ``size`` train states.

        Args:
            train_state_buffer (TrainStateBuffer): Buffer to add to.
            train_state (TrainState): Train state to store at index ``train_state_buffer.n``.

        Returns:
            TrainStateBuffer: The updated buffer with ``n`` incremented by one.

        """
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
    Running statistics for PPO-style reward normalization.

    The statistics live on the agent state, so they survive environment resets and are saved with the agent.
    They are updated with :func:`update_reward_norm`.

    Attributes:
        mean (jax.Array): Running mean of the discounted return, scalar.
        var (jax.Array): Running variance of the discounted return, scalar.
        count (jax.Array): Number of samples the statistics are based on, scalar.
        return_val (jax.Array): Current discounted return of every environment, shape ``(num_envs,)``.

    """
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray
    return_val: jnp.ndarray

    @classmethod
    def create(cls, num_envs: int):
        """
        Create initial statistics (zero mean, unit variance).

        Args:
            num_envs (int): Number of parallel environments.

        Returns:
            RewardNormStats: The initial statistics.

        """
        return cls(
            mean=jnp.zeros(()),
            var=jnp.ones(()),
            count=jnp.full((), 1e-4),
            return_val=jnp.zeros((num_envs,)),
        )


def update_reward_norm(stats: "RewardNormStats", reward: jnp.ndarray,
                       done: jnp.ndarray, gamma: float):
    """
    Update the running return statistics with a batch of rewards and normalize the rewards.

    A discounted return is accumulated per environment (and reset where ``done`` is set). The rewards are divided
    by the running standard deviation of this return; the mean is not subtracted. This mirrors the former
    ``NormalizeVecReward`` wrapper.

    Args:
        stats (RewardNormStats): Current statistics.
        reward (jax.Array): Rewards of the current step, shape ``(num_envs,)``.
        done (jax.Array): Done flags of the current step, shape ``(num_envs,)``.
        gamma (float): Discount factor.

    Returns:
        Tuple[jax.Array, RewardNormStats]: The normalized rewards, shape ``(num_envs,)``, and the updated
        statistics.

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
    """
    Simple circular replay buffer for off-policy and imitation algorithms.

    Core columns: ``(obs, next_obs, action, reward, done)``. ``value_target`` is an optional column used by
    value-distillation-style critic learning (e.g. VanillaDagger stores ``V_teacher(s)`` there at collect time).

    ``next_obs`` and ``value_target`` can be turned off at :meth:`create` time; disabled columns become zero-sized
    dummies (0 bytes). The flags are static metadata, so ``if self.store_*`` branches are baked in at trace time.

    Optional Bernoulli sub-sampling at write time: with ``add_subsample=True`` and ``add_prob=p``, each transition
    in an incoming batch is kept with probability ``p``. The effective temporal coverage becomes roughly
    ``capacity / p`` without growing GPU memory. :meth:`add_batch` requires an ``rng`` when sub-sampling is on.

    The buffer is immutable: :meth:`add_batch` returns a new buffer.

    Attributes:
        obs (jax.Array): Observations, shape ``(capacity, obs_dim)``.
        next_obs (jax.Array): Next observations, shape ``(capacity, obs_dim)``, or ``(0, 0)`` if disabled.
        action (jax.Array): Actions, shape ``(capacity, action_dim)``.
        reward (jax.Array): Rewards, shape ``(capacity,)``.
        done (jax.Array): Done flags as ``float32``, shape ``(capacity,)``.
        value_target (jax.Array): Value targets, shape ``(capacity,)``, or ``(0,)`` if disabled.
        ptr (int): Index of the slot the next transition is written to.
        size (int): Number of valid transitions in the buffer.
        store_next_obs (bool): Whether the ``next_obs`` column is stored (static).
        store_value_target (bool): Whether the ``value_target`` column is stored (static).
        add_subsample (bool): Whether incoming transitions are sub-sampled (static).
        add_prob (float): Probability with which an incoming transition is kept when sub-sampling (static).

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
    add_subsample: bool = struct.field(pytree_node=False, default=False)
    add_prob: float = struct.field(pytree_node=False, default=1.0)

    @classmethod
    def create(cls, obs_dim: int, action_dim: int, capacity: int,
               store_next_obs: bool = True,
               store_value_target: bool = False,
               add_subsample: bool = False,
               add_prob: float = 1.0):
        """
        Create an empty buffer.

        Args:
            obs_dim (int): Dimension of the observations.
            action_dim (int): Dimension of the actions.
            capacity (int): Maximum number of transitions.
            store_next_obs (bool): Whether to allocate the ``next_obs`` column.
            store_value_target (bool): Whether to allocate the ``value_target`` column.
            add_subsample (bool): Whether :meth:`add_batch` keeps each transition only with probability
                ``add_prob``.
            add_prob (float): Keep probability used when ``add_subsample`` is True.

        Returns:
            ReplayBuffer: The empty buffer.

        """
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
            add_subsample=bool(add_subsample),
            add_prob=float(add_prob),
        )

    def add_batch(self, obs, next_obs, action, reward, done,
                  value_target=None, rng=None):
        """
        Add a batch of transitions, overwriting the oldest ones once the buffer is full.

        Args:
            obs (jax.Array): Observations, shape ``(batch, obs_dim)``.
            next_obs (jax.Array): Next observations, shape ``(batch, obs_dim)``; ignored if the column is disabled.
            action (jax.Array): Actions, shape ``(batch, action_dim)``.
            reward (jax.Array): Rewards, shape ``(batch,)``.
            done (jax.Array): Done flags, shape ``(batch,)``; stored as ``float32``.
            value_target (jax.Array): Value targets, shape ``(batch,)``; required if the buffer stores value targets.
            rng (jax.Array): PRNG key; required if the buffer was created with ``add_subsample=True``.

        Returns:
            ReplayBuffer: The updated buffer.

        """
        capacity = self.obs.shape[0]
        batch_size = obs.shape[0]

        if self.add_subsample:
            assert rng is not None, (
                "buffer was created with add_subsample=True but "
                "add_batch received rng=None"
            )
            # Per-element Bernoulli keep mask.
            keep = jax.random.uniform(rng, (batch_size,)) < self.add_prob
            kept_cnt = jnp.sum(keep.astype(jnp.int32))
            # Kept entries are written to consecutive slots; dropped entries get the
            # out-of-range index `capacity`, which mode="drop" discards. They must not be
            # routed to a real slot: a scatter with duplicate indices applies its updates
            # in an unspecified order, so a dropped entry could overwrite a kept one.
            offsets = jnp.cumsum(keep.astype(jnp.int32)) - 1
            indices = jnp.where(keep, (self.ptr + offsets) % capacity, capacity)
            if self.store_next_obs:
                new_next_obs = self.next_obs.at[indices].set(next_obs, mode="drop")
            else:
                new_next_obs = self.next_obs
            if self.store_value_target:
                assert value_target is not None, (
                    "buffer was created with store_value_target=True but "
                    "add_batch received value_target=None"
                )
                new_value_target = self.value_target.at[indices].set(value_target, mode="drop")
            else:
                new_value_target = self.value_target
            return self.replace(
                obs=self.obs.at[indices].set(obs, mode="drop"),
                next_obs=new_next_obs,
                action=self.action.at[indices].set(action, mode="drop"),
                reward=self.reward.at[indices].set(reward, mode="drop"),
                done=self.done.at[indices].set(done.astype(jnp.float32), mode="drop"),
                value_target=new_value_target,
                ptr=(self.ptr + kept_cnt) % capacity,
                size=jnp.minimum(self.size + kept_cnt, capacity),
            )

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
        """
        Sample transitions uniformly with replacement from the valid part of the buffer.

        The buffer must not be empty. The ``value_target`` column is not returned.

        Args:
            rng (jax.Array): PRNG key.
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]: ``(obs, next_obs, action, reward, done)``
            with a leading dimension of ``batch_size``. If ``next_obs`` is not stored, an array of shape
            ``(batch_size, 0)`` is returned in its place.

        """
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
    """
    Reservoir-sampled long-term memory.

    Approximates a uniform sample of all transitions ever passed through :meth:`add_batch`. The capacity stays
    fixed; early additions fill the buffer sequentially, later additions probabilistically replace random slots.
    At steady state each transition seen so far has probability ``capacity / total_seen`` of being in the buffer.

    The columns are ``obs``, ``action`` and (optionally) ``value_target``. There is no ``next_obs``, ``reward`` or
    ``done``: this is intended as a long-term behavior-cloning and value-distillation store, not a TD buffer.

    ``min_include_prob`` optionally floors the per-item inclusion rate. The rate of a pure reservoir decays as
    ``k / n`` indefinitely, which is correct for stationary streams but wastes signal when the data distribution
    drifts (e.g. DAgger rollouts shift as the student learns). Setting ``min_include_prob > 0`` caps the effective
    ``n`` at ``ceil(k / p_min)``, so the buffer behaves like a uniform sample over the last ``k / p_min``
    transitions once that window has filled. The default ``0.0`` gives a pure reservoir.

    Attributes:
        obs (jax.Array): Observations, shape ``(capacity, obs_dim)``.
        action (jax.Array): Actions, shape ``(capacity, action_dim)``.
        value_target (jax.Array): Value targets, shape ``(capacity,)``, or ``(0,)`` if disabled.
        total_seen (jax.Array): Scalar ``int32`` counting all transitions ever added.
        size (jax.Array): Scalar ``int32`` number of valid transitions, ``min(total_seen, capacity)``.
        store_value_target (bool): Whether the ``value_target`` column is stored (static).
        min_include_prob (float): Lower bound on the inclusion probability of a new transition (static).

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
        """
        Create an empty reservoir.

        Args:
            obs_dim (int): Dimension of the observations.
            action_dim (int): Dimension of the actions.
            capacity (int): Number of transitions the reservoir holds.
            store_value_target (bool): Whether to allocate the ``value_target`` column.
            min_include_prob (float): Lower bound on the inclusion probability; ``0.0`` gives a pure reservoir.

        Returns:
            ReservoirBuffer: The empty reservoir.

        """
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
        """
        Reservoir-insert a batch of transitions.

        Vectorized: for each element ``i`` with global index ``g_i = total_seen + i``, an index ``j_i`` is drawn
        uniformly from ``[0, n_eff_i)``, where ``n_eff_i = g_i + 1``, capped at ``ceil(capacity / min_include_prob)``
        if ``min_include_prob > 0``. The element is written to slot ``g_i`` while the buffer is still filling, and
        otherwise to slot ``j_i`` if ``j_i < capacity``. If two included elements of a batch map to the same slot,
        whichever wins the scatter is kept, which introduces a tiny bias.

        Args:
            obs (jax.Array): Observations, shape ``(batch, obs_dim)``.
            action (jax.Array): Actions, shape ``(batch, action_dim)``.
            value_target (jax.Array): Value targets, shape ``(batch,)``; required if the reservoir stores value
                targets, ignored otherwise.
            rng (jax.Array): PRNG key.

        Returns:
            ReservoirBuffer: The updated reservoir.

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
        """
        Sample transitions uniformly with replacement from the valid part of the reservoir.

        Args:
            rng (jax.Array): PRNG key.
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple[jax.Array, jax.Array, jax.Array]: ``(obs, action, value_target)`` with a leading dimension of
            ``batch_size``. If value targets are not stored, zeros of shape ``(batch_size,)`` are returned in their
            place.

        """
        indices = jax.random.randint(rng, (batch_size,), 0, jnp.maximum(1, self.size))
        return (
            self.obs[indices],
            self.action[indices],
            (self.value_target[indices] if self.store_value_target
             else jnp.zeros((batch_size,), dtype=self.obs.dtype)),
        )


@struct.dataclass
class BestTrainStates:
    """
    Container for the ``n`` best train states seen during training, together with their performance.

    Attributes:
        train_states (TrainState): Train-state pytree whose leaves have a leading axis of length ``n``.
        metrics (jax.Array): Performance metric of every stored train state, shape ``(n,)``.
        iterations (jax.Array): Training iteration at which every train state was stored, shape ``(n,)``.
        cur_worst_perf (float): Worst performance among the stored train states.
        step (int): Step counter.
        n (int): Number of train states to keep.
        size (int): Number of train states currently stored.

    """
    train_states: TrainState
    metrics: jnp.array
    iterations: jnp.array
    cur_worst_perf: float
    step: int
    n: int
    size: int

    @classmethod
    def create(cls, train_state: TrainState, n: int):
        """
        Create a container whose slots are all initialized with copies of ``train_state``.

        Args:
            train_state (TrainState): Train state used as template and initial content of every slot.
            n (int): Number of train states to keep.

        Returns:
            BestTrainStates: The new container with all metrics set to ``-inf``.

        """
        return BestTrainStates(
            train_states=jax.tree.map(lambda x: jnp.stack([x] * n), train_state),
            metrics=jnp.full((n,), -jnp.inf),
            iterations=jnp.zeros((n,)),
            cur_worst_perf=-jnp.inf,
            n=n,
            size=0
        )
