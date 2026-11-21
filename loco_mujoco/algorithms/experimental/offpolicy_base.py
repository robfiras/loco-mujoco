"""
Off-policy actor-critic base class shared by SAC and TD3.

This module owns:

- the Q-network ensemble (:class:`OffPolicyCriticNet`, twin critics by default),
- the target critic with soft (Polyak) updates,
- environment collection and insertion into the
  :class:`~loco_mujoco.algorithms.common.dataclasses.ReplayBuffer` (re-exported here for backwards
  compatibility),
- the outer scan over environment steps (one transition per environment per step, followed by
  ``gradient_steps`` gradient updates),
- the ``learning_starts`` logic and the skeleton of the logged metrics.

Subclasses of :class:`OffPolicyBase` provide the algorithm-specific pieces via hook methods:

- ``_build_actor_net(action_dim, exp)``
- ``_init_extra_state(rng, exp)``: optional extra state (the temperature for SAC, the target actor for TD3).
- ``_select_action(actor_state, obs, rng, exp, extra_state, deterministic)``: action for environment
  interaction.
- ``_next_action_and_q_bonus(actor_state, next_obs, rng, exp, extra_state)``: returns
  ``(next_action, next_bonus, aux)`` for the target
  ``reward + gamma * (1 - done) * (min(Q1, Q2)(next_obs, next_action) + next_bonus)``. SAC uses
  ``next_bonus = -alpha * log_pi(next_action)``, TD3 uses ``next_bonus = 0`` with a target-smoothed action.
- ``_actor_loss(actor_params, ..., extra_state)``: ``alpha * log_pi - Q`` for SAC, ``-Q`` for TD3.
- ``_update_extra(extra_state, aux, rng, exp)``: e.g. the temperature update of SAC.
- ``_should_update_actor(step_count, exp)``: for the policy delay of TD3 (default: always).
- ``_extra_metrics(actor_state, obs, rng, extra_state, exp)``
- ``_build_summary_metric(base_kwargs, extra)``

The module additionally contains helpers for optional critic variants (batch normalization, weight
normalization, categorical critics, pessimistic ensemble targets) that are used by
:class:`~loco_mujoco.algorithms.experimental.td3_jax.TD3Jax`.

"""

from typing import Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import flax
import optax
from flax import struct
from omegaconf import DictConfig

from loco_mujoco.algorithms import (JaxRLAlgorithmBase, AgentConfBase,
                                    AgentStateBase, TrainState, ReplayBuffer)
from loco_mujoco.algorithms.common.networks import (RunningMeanStd,
                                                    get_activation_fn)
from loco_mujoco.core.wrappers import (LogWrapper, LogEnvState, VecEnv)
from loco_mujoco.utils import MetricsHandler, ValidationSummary


# ---------------------------------------------------------------------------
# Networks (shared)
# ---------------------------------------------------------------------------

class _QNet(nn.Module):
    """
    One Q-network sub-module mapping the concatenated ``(obs, action)`` input to a Q-value.

    Optional pieces (all gated; the defaults give a plain MLP critic):

    - ``use_batch_norm``: insert BatchNorm into each hidden block. With ``pre_activation_bn=True`` a block is
      ``Dense(no bias) -> BN -> activation``, with ``pre_activation_bn=False`` it is ``Dense -> activation -> BN``.
    - ``num_atoms > 1``: replace the scalar head with a categorical head of ``num_atoms`` logits. The network then
      returns the scalar value ``sum(softmax * z)`` with ``z = linspace(min_v, max_v, num_atoms)``. The per-atom
      log-probabilities are stashed via ``self.sow("log_probs_collection", ...)`` so that a categorical
      cross-entropy loss can retrieve them.

    """

    hidden_layer_dims: tuple = (256, 256)
    activation: str = "tanh"
    use_batch_norm: bool = False
    pre_activation_bn: bool = True
    num_atoms: int = 1
    min_v: float = -5.0
    max_v: float = 5.0

    @nn.compact
    def __call__(self, x, *, training=False):
        activation_fn = get_activation_fn(self.activation)
        for dim in self.hidden_layer_dims:
            if self.use_batch_norm:
                if self.pre_activation_bn:
                    # XQC pre-activation: Dense(no bias) -> BN -> act
                    x = nn.Dense(dim, use_bias=False,
                                 kernel_init=orthogonal(jnp.sqrt(2)))(x)
                    x = nn.BatchNorm(use_running_average=not training,
                                     momentum=0.99, epsilon=0.001)(x)
                    x = activation_fn(x)
                else:
                    # CrossQ post-activation: Dense -> act -> BN
                    x = nn.Dense(dim, kernel_init=orthogonal(jnp.sqrt(2)),
                                 bias_init=constant(0.0))(x)
                    x = activation_fn(x)
                    x = nn.BatchNorm(use_running_average=not training,
                                     momentum=0.99, epsilon=0.001)(x)
            else:
                x = nn.Dense(dim, kernel_init=orthogonal(jnp.sqrt(2)),
                             bias_init=constant(0.0))(x)
                x = activation_fn(x)

        # Head
        n_out = max(1, int(self.num_atoms))
        head = nn.Dense(n_out, kernel_init=orthogonal(1.0),
                        bias_init=constant(0.0))(x)
        if n_out == 1:
            return jnp.squeeze(head, axis=-1)
        # Categorical: convert logits -> log_probs over `num_atoms` bins,
        # extract scalar value via support.
        log_probs = nn.log_softmax(head, axis=-1)
        # Stash log-probs so the CE loss can find them via mutable collection.
        self.sow("log_probs_collection", "log_probs", log_probs)
        bin_values = jnp.linspace(self.min_v, self.max_v, n_out)
        value = jnp.sum(jnp.exp(log_probs) * bin_values, axis=-1)
        return value


def aggregate_q_target(qs, pessimism_penalty=None):
    """
    Reduce an ensemble of target Q-values to a single bootstrap value.

    The spread term of the pessimistic variant is the ensemble *range* rather than the standard deviation, so that
    the two-critic case reproduces the usual formula exactly: for two heads ``max - min == |Q1 - Q2|``, hence
    ``mean - k * range`` with ``k = 0.5`` is identical to ``min(Q1, Q2)``. This makes ``pessimism_penalty`` a
    continuous dial: ``k = 0`` is a plain ensemble mean (no pessimism), ``k = 0.5`` is standard TD3 and
    ``k > 0.5`` is more conservative than TD3.

    Args:
        qs (Sequence[jax.Array]): ``K`` arrays of shape ``(batch,)``, one per critic head.
        pessimism_penalty (float): If ``None``, the elementwise minimum over the heads is taken, i.e. the standard
            TD3/SAC worst-case pessimism. Otherwise ``mean(Q) - pessimism_penalty * (max(Q) - min(Q))`` is used.

    Returns:
        jax.Array: The aggregated target Q-values, shape ``(batch,)``.

    """
    q_stack = jnp.stack(qs, axis=0)
    if pessimism_penalty is None:
        return jnp.min(q_stack, axis=0)
    k = float(pessimism_penalty)
    q_range = jnp.max(q_stack, axis=0) - jnp.min(q_stack, axis=0)
    return jnp.mean(q_stack, axis=0) - k * q_range


def pessimistic_mixture_weights(num_critics, pessimism_penalty):
    """
    Compute the weights for blending the categorical target distributions of ``num_critics`` critic heads.

    A distributional critic cannot average Q-values, it needs a single coherent distribution, so the arithmetic of
    :func:`aggregate_q_target` does not transfer. The analogue is a mixture that concentrates on the most
    pessimistic head as ``k = pessimism_penalty`` rises: the head with the smallest Q-value gets the weight
    ``1/K + (k / 0.5) * (1 - 1/K)``, clipped to ``[1/K, 1]``, and the other heads share the remainder equally.
    ``k = 0.5`` puts all mass on the most pessimistic head (a hard pick) and ``k = 0`` gives the uniform mixture
    (no pessimism). The mixing is meant to be done on probabilities, not log-probabilities, so that the result
    stays a valid distribution for the C51 projection.

    Args:
        num_critics (int): Number of critic heads ``K``.
        pessimism_penalty (float): Pessimism coefficient ``k``.

    Returns:
        Tuple[float, float]: ``(w_argmin, w_other)``, the weight of the most pessimistic head and the weight of
        each other head, as Python floats.

    Raises:
        ValueError: If ``num_critics < 1``.

    """
    K = int(num_critics)
    if K < 1:
        raise ValueError(f"num_critics must be >= 1, got {K}")
    if K == 1:
        return 1.0, 0.0
    k = float(pessimism_penalty)
    w_argmin = 1.0 / K + (k / 0.5) * (1.0 - 1.0 / K)
    w_argmin = min(max(w_argmin, 1.0 / K), 1.0)
    return w_argmin, (1.0 - w_argmin) / (K - 1)


class OffPolicyCriticNet(nn.Module):
    """
    Ensemble of ``num_critics`` Q-networks, optionally with running normalization of the observation.

    The default is two critics, the classic TD3/SAC twin. The sub-modules are named ``q1`` ... ``qK`` and the
    network returns a ``K``-tuple of Q-values, so that with two critics ``q1, q2 = critic(obs, action)`` works.

    Attributes:
        hidden_layer_dims (tuple): Number of units of each hidden layer of every Q-network.
        activation (str): Name of the activation function in ``flax.linen``.
        use_obs_norm (bool): If True, the observation is normalized by a
            :class:`~loco_mujoco.algorithms.common.networks.RunningMeanStd` layer (adds a ``run_stats``
            collection).
        use_batch_norm (bool): If True, BatchNorm is inserted into every hidden block (adds a ``batch_stats``
            collection).
        pre_activation_bn (bool): If True, BatchNorm is applied before the activation, otherwise after it.
        num_atoms (int): Number of atoms of the categorical value head; ``1`` gives a scalar head.
        min_v (float): Lower end of the support of the categorical head.
        max_v (float): Upper end of the support of the categorical head.
        obs_ind (jax.Array): Indices of the observation entries the critics see; ``None`` means all.
        num_critics (int): Number of Q-networks in the ensemble.

    """

    hidden_layer_dims: tuple = (256, 256)
    activation: str = "tanh"
    use_obs_norm: bool = True
    use_batch_norm: bool = False
    pre_activation_bn: bool = True
    num_atoms: int = 1
    min_v: float = -5.0
    max_v: float = 5.0
    obs_ind: jnp.ndarray = None
    num_critics: int = 2

    @nn.compact
    def __call__(self, obs, action, *, training=False):
        """
        Evaluate all Q-networks.

        Args:
            obs (jax.Array): Observations, shape ``(batch, obs_dim)``.
            action (jax.Array): Actions, shape ``(batch, action_dim)``.
            training (bool): Whether BatchNorm layers use batch statistics (and update their running averages)
                instead of the running averages.

        Returns:
            Tuple[jax.Array, ...]: One array of Q-values of shape ``(batch,)`` per critic.

        """
        if self.obs_ind is not None:
            obs = obs[..., self.obs_ind]
        if self.use_obs_norm:
            obs = RunningMeanStd()(obs)
        x = jnp.concatenate([obs, action], axis=-1)
        kwargs = dict(
            hidden_layer_dims=self.hidden_layer_dims,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
            pre_activation_bn=self.pre_activation_bn,
            num_atoms=self.num_atoms,
            min_v=self.min_v,
            max_v=self.max_v,
        )
        # Named q1..qK so that K=2 yields exactly the historical q1/q2 param
        # tree -- old checkpoints load without migration.
        return tuple(
            _QNet(**kwargs, name=f"q{i + 1}")(x, training=training)
            for i in range(int(self.num_critics))
        )


# ReplayBuffer has moved to loco_mujoco.algorithms.common.dataclasses;
# re-imported above and re-exported here for back-compat.


# ---------------------------------------------------------------------------
# Weight-normalization helper (Salimans-Kingma / XQC-style post-step renorm)
# ---------------------------------------------------------------------------


def _normalize_dense_kernels(params, normalize_last_layer: bool = True):
    """
    Renormalize the weights of every Dense layer to unit norm per output unit.

    For each Dense layer the kernel (stacked with the bias, if present) is divided by its L2 norm along the input
    axis (``axis=-2``). Mirrors ``xqc/networks/common.py::norm_network``.

    Args:
        params: Flax parameter pytree.
        normalize_last_layer (bool): If False, skip the final Dense layer of each Q-network (matches the XQC
            ``normalize_last_layer`` flag).

    Returns:
        A new parameter pytree with the same structure.

    """
    flat = flax.traverse_util.flatten_dict(params, sep="/")

    dense_paths = sorted({
        "/".join(k.split("/")[:-1])
        for k in flat
        if k.endswith("/kernel") and "Dense" in k
    })

    if not normalize_last_layer and dense_paths:
        # Drop the last Dense under each q1/q2 (or other) prefix — that's the
        # predictor head.
        from collections import defaultdict
        last_per_prefix = defaultdict(lambda: ("", -1))
        for path in dense_paths:
            parts = path.split("/")
            prefix = "/".join(parts[:-1])
            dense_name = parts[-1]
            if dense_name.startswith("Dense_"):
                try:
                    idx = int(dense_name[len("Dense_"):])
                except ValueError:
                    continue
                if idx > last_per_prefix[prefix][1]:
                    last_per_prefix[prefix] = (path, idx)
        skip = {p for (p, _) in last_per_prefix.values()}
        dense_paths = [p for p in dense_paths if p not in skip]

    for path in dense_paths:
        kernel_key = f"{path}/kernel"
        bias_key = f"{path}/bias"
        kernel = flat[kernel_key]
        if bias_key in flat:
            bias = flat[bias_key]
            w = jnp.concatenate([kernel, jnp.expand_dims(bias, -2)], axis=-2)
            norm = jnp.linalg.norm(w, axis=-2, keepdims=True) + 1e-12
            flat[kernel_key] = kernel / norm
            flat[bias_key] = bias / norm.squeeze(-2)
        else:
            norm = jnp.linalg.norm(kernel, axis=-2, keepdims=True) + 1e-12
            flat[kernel_key] = kernel / norm

    return flax.traverse_util.unflatten_dict(flat, sep="/")


def _c51_project_target(target_log_probs, target_bin_values, num_atoms,
                        min_v, max_v):
    """
    Project a categorical target distribution with shifted bin centers back onto the fixed C51 support.

    Mirrors ``xqc.agents.xqc.critic.categorical_td_loss``: the probability mass of each shifted bin is distributed
    between the two nearest bins of the support using floor/ceil weights.

    Args:
        target_log_probs (jax.Array): Log-probabilities of the next-state distribution, shape
            ``(batch, num_atoms)``.
        target_bin_values (jax.Array): Bin centers after the bootstrap shift
            ``reward + gamma * (bin + bonus) * (1 - done)``, shape ``(batch, num_atoms)``. Values are clipped to
            ``[min_v, max_v]``.
        num_atoms (int): Number of atoms of the support.
        min_v (float): Lower end of the support.
        max_v (float): Upper end of the support.

    Returns:
        jax.Array: The target distribution, shape ``(batch, num_atoms)``, ready for the cross-entropy against the
        predicted log-probabilities.

    """
    target_bin_values = jnp.clip(target_bin_values, min_v, max_v)

    # `b` indexes new-bin positions for each shifted old bin center.
    b = (target_bin_values - min_v) / ((max_v - min_v) / (num_atoms - 1))
    l = jnp.floor(b)
    u = jnp.ceil(b)
    l_mask = jax.nn.one_hot(l.reshape(-1), num_atoms).reshape(
        (-1, num_atoms, num_atoms)
    )
    u_mask = jax.nn.one_hot(u.reshape(-1), num_atoms).reshape(
        (-1, num_atoms, num_atoms)
    )

    target_probs_old = jnp.exp(target_log_probs)
    # When floor == ceil (target value lands exactly on a bin center), put all
    # mass on that bin; else split (u-b) to lower and (b-l) to upper.
    m_l = (target_probs_old * (u + (l == u).astype(jnp.float32) - b)).reshape(
        (-1, num_atoms, 1)
    )
    m_u = (target_probs_old * (b - l)).reshape((-1, num_atoms, 1))
    target_probs = jnp.sum(m_l * l_mask + m_u * u_mask, axis=1)
    return target_probs


# ---------------------------------------------------------------------------
# Base agent conf / state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OffPolicyAgentConf(AgentConfBase):
    """
    Common static configuration fields of off-policy algorithms; subclasses can extend it.

    Attributes:
        config (DictConfig): The full configuration; the hyperparameters live under ``config.experiment``.
        actor_net: The actor network definition.
        critic_net: The critic network definition.
        actor_tx: The Optax optimizer of the actor.
        critic_tx: The Optax optimizer of the critic.

    """
    config: DictConfig
    actor_net: Any
    critic_net: Any
    actor_tx: Any
    critic_tx: Any


@struct.dataclass
class OffPolicyAgentState(AgentStateBase):
    """
    Dynamic state of an off-policy agent.

    Attributes:
        actor_state (TrainState): Parameters, optimizer state and running statistics of the actor.
        critic_state (TrainState): Parameters and optimizer state of the critic ensemble. Its ``run_stats`` field
            holds the mutable variable collections of the critic (running observation statistics and, with
            batch normalization, batch statistics).
        target_critic_params: Parameters of the target critic.
        target_critic_run_stats: Mutable variable collections of the target critic.
        extra_state: Algorithm-specific state (e.g. the temperature for SAC, the target actor for TD3).
        replay_buffer (ReplayBuffer): The replay buffer.
        env_state: Environment state at the end of the last training chunk, carried over so that training can
            continue seamlessly; not saved to disk.
        last_obs (jax.Array): Last observations of the previous training chunk; not saved to disk.

    """
    actor_state: TrainState
    critic_state: TrainState
    target_critic_params: Any
    target_critic_run_stats: Any
    extra_state: Any                # algorithm-specific (e.g. log_alpha for SAC)
    replay_buffer: ReplayBuffer
    env_state: Any = None
    last_obs: Any = None

    def serialize(self):
        """
        Serialize the agent state.

        The replay buffer, ``env_state`` and ``last_obs`` are not serialized.

        Returns:
            dict: Dictionary with the keys ``"actor_state"``, ``"critic_state"``, ``"target_critic"`` and
            ``"extra_state"``.

        """
        return {
            "actor_state": flax.serialization.to_state_dict(self.actor_state),
            "critic_state": flax.serialization.to_state_dict(self.critic_state),
            "target_critic": flax.serialization.to_state_dict(
                {"params": self.target_critic_params,
                 "run_stats": self.target_critic_run_stats}
            ),
            "extra_state": flax.serialization.to_state_dict(self.extra_state),
        }


# ---------------------------------------------------------------------------
# OffPolicyBase
# ---------------------------------------------------------------------------

class OffPolicyBase(JaxRLAlgorithmBase):
    """
    Shared off-policy actor-critic engine.

    Implements the training loop common to SAC and TD3: in every iteration all environments are stepped once, the
    transitions are added to the replay buffer and, once the buffer holds ``learning_starts`` transitions,
    ``gradient_steps`` updates of critic, actor and target critic are performed. Subclasses must override the
    hooks listed in the docstring of this module (see
    :class:`~loco_mujoco.algorithms.experimental.sac_jax.SACJax` and
    :class:`~loco_mujoco.algorithms.experimental.td3_jax.TD3Jax`).

    """

    # ---------- Hooks (override in subclasses) ---------------------------
    @classmethod
    def _build_actor_net(cls, action_dim: int, exp):
        """
        Build the actor network; must be implemented by subclasses that use it.

        Args:
            action_dim (int): Dimension of the action space.
            exp (DictConfig): The ``experiment`` node of the configuration.

        Returns:
            nn.Module: The actor network definition.

        """
        raise NotImplementedError

    @classmethod
    def _init_extra_state(cls, rng, exp):
        """
        Initialize the algorithm-specific extra state.

        Args:
            rng (jax.Array): PRNG key.
            exp (DictConfig): The ``experiment`` node of the configuration.

        Returns:
            A pytree; empty by default.

        """
        return {}

    @classmethod
    def _select_action(cls, actor_state, obs, rng, exp,
                       extra_state, deterministic=False):
        """
        Select the action used for environment interaction.

        Args:
            actor_state (TrainState): Train state of the actor.
            obs (jax.Array): Observations, shape ``(num_envs, obs_dim)``.
            rng (jax.Array): PRNG key.
            exp (DictConfig): The ``experiment`` node of the configuration.
            extra_state: Algorithm-specific extra state.
            deterministic (bool): If True, no exploration noise is applied.

        Returns:
            Tuple[jax.Array, TrainState]: The actions and the actor train state with updated running statistics.

        """
        raise NotImplementedError

    @classmethod
    def _next_action_and_q_bonus(cls, actor_state, next_obs, rng, exp,
                                 extra_state):
        """
        Compute the next action and the additive bonus of the Q-target.

        The Q-target is ``r + gamma * (1 - d) * (agg(Q)(next_obs, next_action) + next_q_bonus)``, where ``agg`` is the
        minimum over the critics by default. SAC returns ``next_q_bonus = -alpha * log_pi(next_action)``; TD3 returns
        zero (its next action is target-policy smoothed).

        Args:
            actor_state (TrainState): Train state of the actor.
            next_obs (jax.Array): Next observations, shape ``(batch, obs_dim)``.
            rng (jax.Array): PRNG key.
            exp (DictConfig): The ``experiment`` node of the configuration.
            extra_state: Algorithm-specific extra state.

        Returns:
            Tuple[jax.Array, jax.Array, dict]: ``(next_action, next_q_bonus, aux)`` with shapes
            ``(batch, action_dim)`` and ``(batch,)``; ``aux`` can carry the log-probability or any other signal.

        """
        raise NotImplementedError

    @classmethod
    def _actor_loss(cls, actor_params, actor_apply_fn, actor_run_stats,
                    critic_st, critic_apply_fn,
                    obs_b, rng, exp, extra_state):
        """
        Compute the actor loss on a batch of observations.

        Args:
            actor_params: Actor parameters the loss is differentiated with respect to.
            actor_apply_fn (Callable): ``apply`` function of the actor network.
            actor_run_stats: Running statistics of the actor.
            critic_st (TrainState): Train state of the critic (after its update).
            critic_apply_fn (Callable): ``apply`` function of the critic network.
            obs_b (jax.Array): Batch of observations, shape ``(batch, obs_dim)``.
            rng (jax.Array): PRNG key.
            exp (DictConfig): The ``experiment`` node of the configuration.
            extra_state: Algorithm-specific extra state.

        Returns:
            Tuple[jax.Array, dict]: The scalar loss and an auxiliary dictionary (e.g. the log-probabilities for SAC).
            If it contains the key ``"run_stats"``, these become the new running statistics of the actor.

        """
        raise NotImplementedError

    @classmethod
    def _update_extra(cls, extra_state, actor_aux, rng, exp):
        """
        Update the algorithm-specific extra state (e.g. the temperature of SAC).

        Called only in gradient steps in which the actor is updated.

        Args:
            extra_state: Algorithm-specific extra state.
            actor_aux (dict): Auxiliary output of :meth:`_actor_loss`.
            rng (jax.Array): PRNG key.
            exp (DictConfig): The ``experiment`` node of the configuration.

        Returns:
            Tuple[Any, dict]: The new extra state and a dictionary of auxiliary metrics.

        """
        return extra_state, {}

    @classmethod
    def _should_update_actor(cls, step_count, exp):
        """
        Decide whether the actor is updated in the current gradient step.

        Used for the policy delay of TD3; by default the actor is always updated.

        Args:
            step_count (jax.Array): Index of the gradient step within the current iteration.
            exp (DictConfig): The ``experiment`` node of the configuration.

        Returns:
            jax.Array: Scalar boolean.

        """
        return jnp.array(True)

    @classmethod
    def _extra_metrics(cls, actor_state, obs, rng, extra_state, exp):
        """
        Compute additional metrics for logging (e.g. the temperature and the entropy for SAC).

        Args:
            actor_state (TrainState): Train state of the actor.
            obs (jax.Array): Current observations, shape ``(num_envs, obs_dim)``.
            rng (jax.Array): PRNG key.
            extra_state: Algorithm-specific extra state.
            exp (DictConfig): The ``experiment`` node of the configuration.

        Returns:
            dict: Additional metrics; empty by default.

        """
        return {}

    @classmethod
    def _build_summary_metric(cls, base_kwargs, extra):
        """
        Combine the base metrics and the extra metrics into the summary struct of the algorithm.

        Args:
            base_kwargs (dict): Base metrics: ``mean_episode_return``, ``mean_episode_length``, ``max_timestep``,
                ``mean_critic_loss``, ``mean_actor_loss`` and ``buffer_size``.
            extra (dict): Output of :meth:`_extra_metrics`.

        Returns:
            The summary metrics struct (a subclass of :class:`~loco_mujoco.core.wrappers.mjx.SummaryMetrics`).

        """
        raise NotImplementedError

    # ---------- Optimisers (override if you need separate optimisers) ----
    @classmethod
    def _build_optimisers(cls, exp):
        """
        Build the optimizers of actor and critic: gradient clipping by global norm followed by Adam.

        Reads ``lr_actor`` and ``lr_critic`` and the optional ``max_grad_norm`` (default 0.5),
        ``max_actor_grad_norm`` and ``max_critic_grad_norm`` (both default to ``max_grad_norm``).

        Args:
            exp (DictConfig): The ``experiment`` node of the configuration.

        Returns:
            Tuple[optax.GradientTransformation, optax.GradientTransformation]: ``(actor_tx, critic_tx)``.

        """
        max_grad = float(getattr(exp, 'max_grad_norm', 0.5))
        actor_tx = optax.chain(
            optax.clip_by_global_norm(float(getattr(exp, 'max_actor_grad_norm', max_grad))),
            optax.adam(float(exp.lr_actor)),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(float(getattr(exp, 'max_critic_grad_norm', max_grad))),
            optax.adam(float(exp.lr_critic)),
        )
        return actor_tx, critic_tx

    # ---------- Conf helpers (subclasses can override) -------------------
    @classmethod
    def _build_critic_net(cls, exp, obs_ind=None):
        """
        Build the critic ensemble from the configuration.

        Reads ``hidden_layers`` and ``activation`` and the optional keys ``use_obs_norm``, ``use_batch_norm``,
        ``pre_activation_bn``, ``num_atoms``, ``min_v``, ``max_v`` and ``num_critics``.

        Args:
            exp (DictConfig): The ``experiment`` node of the configuration.
            obs_ind (jax.Array): Indices of the observation entries the critic sees; ``None`` means all.

        Returns:
            OffPolicyCriticNet: The critic network definition.

        """
        import ast
        from omegaconf import ListConfig
        hidden = (exp.hidden_layers if isinstance(exp.hidden_layers, (list, ListConfig))
                  else ast.literal_eval(exp.hidden_layers))
        return OffPolicyCriticNet(
            hidden_layer_dims=tuple(hidden),
            activation=str(exp.activation),
            use_obs_norm=bool(getattr(exp, 'use_obs_norm', False)),
            use_batch_norm=bool(getattr(exp, 'use_batch_norm', False)),
            pre_activation_bn=bool(getattr(exp, 'pre_activation_bn', True)),
            num_atoms=int(getattr(exp, 'num_atoms', 1)),
            min_v=float(getattr(exp, 'min_v', -5.0)),
            max_v=float(getattr(exp, 'max_v', 5.0)),
            obs_ind=obs_ind,
            num_critics=int(getattr(exp, 'num_critics', 2)),
        )

    # ---------- Training loop (the shared engine) ------------------------
    @classmethod
    def _train_fn(cls, rng, env,
                  agent_conf,
                  agent_state=None,
                  traj=None,
                  mh: MetricsHandler = None):
        """
        Run the shared off-policy training loop.

        ``num_updates`` iterations are performed. Each one steps all ``num_envs`` environments once with the
        exploration policy, adds the transitions to the replay buffer (the ``absorbing`` flag is stored as the done
        flag, so values are bootstrapped through time limits) and, if the buffer holds at least ``learning_starts``
        transitions, runs ``gradient_steps`` gradient steps. A gradient step updates the critics on the mean squared
        error to the Q-target, optionally renormalizes the critic weights (``use_weight_norm``), updates the actor and
        the extra state if :meth:`_should_update_actor` allows it, and soft-updates the target critic with rate
        ``tau``. If ``agent_state`` carries an environment state from a previous call, training continues from it.

        Args:
            rng (jax.Array): PRNG key.
            env: The (unwrapped) environment.
            agent_conf (OffPolicyAgentConf): Static agent configuration.
            agent_state (OffPolicyAgentState): Agent state to start or resume from. If ``None``, a new one is
                initialized.
            traj: Trajectory data passed on to ``env.reset`` and ``env.step``; ``None`` if the task has no trajectory.
            mh (MetricsHandler): Unused; no validation is run during off-policy training.

        Returns:
            dict: Dictionary with the keys

            - ``"agent_state"``: the agent state after training, including replay buffer, last environment state
              and last observation.
            - ``"training_metrics"``: the summary metrics of the algorithm with a leading axis of length
              ``max(1, num_updates // log_every)``; each entry aggregates ``log_every`` (default 100) iterations.
              The episode return and length are the means over all environments of their last completed episode.
            - ``"validation_metrics"`` (:class:`~loco_mujoco.utils.metrics.ValidationSummary`): always empty.

        """

        exp = agent_conf.config.experiment
        actor_net = agent_conf.actor_net
        critic_net = agent_conf.critic_net

        env = cls._wrap_env(env, exp)

        # ----- restore or init -----
        if agent_state is not None:
            actor_state = agent_state.actor_state.replace(apply_fn=actor_net.apply)
            critic_state = agent_state.critic_state.replace(apply_fn=critic_net.apply)
            target_critic_params = agent_state.target_critic_params
            target_critic_run_stats = agent_state.target_critic_run_stats
            extra_state = agent_state.extra_state
            replay_buffer = agent_state.replay_buffer
        else:
            rng, rng_a, rng_c, rng_e = jax.random.split(rng, 4)
            obs_dim = int(exp.obs_dim)
            action_dim = int(exp.action_dim)

            actor_params = actor_net.init(rng_a, jnp.zeros((1, obs_dim)))
            actor_state = TrainState.create(
                apply_fn=actor_net.apply,
                params=actor_params["params"],
                run_stats=actor_params.get("run_stats", {}),
                tx=agent_conf.actor_tx,
            )
            critic_params = critic_net.init(
                rng_c, jnp.zeros((1, obs_dim)), jnp.zeros((1, action_dim))
            )
            # Bundle every non-params variable collection (run_stats, batch_stats)
            # into `critic_state.run_stats` so subsequent apply / target-update
            # sites can plumb a single object regardless of whether BN is on.
            # `log_probs_collection` is a sow output, not long-lived state.
            _BUNDLE_EXCLUDE = {"params", "log_probs_collection"}
            critic_bundle = {k: v for k, v in critic_params.items() if k not in _BUNDLE_EXCLUDE}
            critic_state = TrainState.create(
                apply_fn=critic_net.apply,
                params=critic_params["params"],
                run_stats=critic_bundle,
                tx=agent_conf.critic_tx,
            )
            target_critic_params = critic_params["params"]
            target_critic_run_stats = critic_bundle
            extra_state = cls._init_extra_state(rng_e, exp)
            replay_buffer = ReplayBuffer.create(
                int(exp.obs_dim), int(exp.action_dim), int(exp.buffer_size),
                add_subsample=bool(getattr(exp, 'buffer_add_subsample', False)),
                add_prob=float(getattr(exp, 'buffer_add_prob', 1.0)),
            )

        # ----- env init -----
        if agent_state is not None and agent_state.env_state is not None:
            env_state = agent_state.env_state
            last_obs = agent_state.last_obs
        else:
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, exp.num_envs)
            last_obs, env_state = env.reset(reset_rng, traj)

        # ----- helpers / config -----
        learning_starts = int(getattr(exp, 'learning_starts', exp.batch_size))
        batch_size = int(exp.batch_size)
        tau = float(getattr(exp, 'tau', 0.005))
        gamma = float(exp.gamma)
        gradient_steps = int(getattr(exp, 'gradient_steps', 1))

        def _critic_forward_target(obs, action, params, run_stats):
            # `run_stats` is the full mutable-variables bundle for the critic
            # (run_stats + optional batch_stats). Splatting `**run_stats` lets
            # us handle both old (only run_stats) and BN (run_stats + batch_stats)
            # critics without conditional plumbing through every call site.
            mutables = list(run_stats.keys())
            qs, updates = critic_net.apply(
                {"params": params, **run_stats},
                obs, action, mutable=mutables, training=False,
            )
            new_bundle = {k: updates.get(k, run_stats[k]) for k in run_stats}
            return qs, new_bundle

        # ----- collect transition -----
        def _collect_transition(actor_state, replay_buffer, env_state,
                                last_obs, rng, extra_state):
            rng, rng_act = jax.random.split(rng)
            action, actor_state = cls._select_action(
                actor_state, last_obs, rng_act, exp, extra_state, deterministic=False
            )
            next_obs, reward, absorbing, done, info, env_state = env.step(
                env_state, action, traj
            )
            if replay_buffer.add_subsample:
                rng, rng_buf = jax.random.split(rng)
                replay_buffer = replay_buffer.add_batch(
                    last_obs, next_obs, action, reward,
                    absorbing.astype(jnp.float32), rng=rng_buf,
                )
            else:
                replay_buffer = replay_buffer.add_batch(
                    last_obs, next_obs, action, reward, absorbing.astype(jnp.float32)
                )
            return actor_state, replay_buffer, env_state, next_obs, rng

        # ----- single gradient update -----
        def _single_gradient_update(carry, step_idx):
            (actor_st, critic_st, tgt_p, tgt_rs,
             ex_st, buf, rng_up) = carry

            rng_up, rng_sample = jax.random.split(rng_up)
            obs_b, nobs_b, act_b, rew_b, done_b = buf.sample(rng_sample, batch_size)

            # -- critic loss --
            rng_up, rng_next = jax.random.split(rng_up)
            next_action, next_q_bonus, _ = cls._next_action_and_q_bonus(
                actor_st, nobs_b, rng_next, exp, ex_st
            )
            qs_next, new_tgt_rs = _critic_forward_target(
                nobs_b, next_action, tgt_p, tgt_rs
            )
            # Q-target aggregation across the critic ensemble. Default is the
            # standard TD3/SAC `min` (worst-case pessimism); `pessimism_penalty`
            # switches to `mean(Q) - k * range(Q)`, which treats inter-head
            # disagreement as an uncertainty estimate. See aggregate_q_target.
            q_next = aggregate_q_target(
                qs_next, getattr(exp, "pessimism_penalty", None)
            ) + next_q_bonus
            q_target = rew_b + gamma * (1.0 - done_b) * q_next
            q_target = jax.lax.stop_gradient(q_target)

            def _critic_loss_fn(params):
                run_stats = critic_st.run_stats
                mutables = list(run_stats.keys())
                qs, updates = critic_net.apply(
                    {"params": params, **run_stats},
                    obs_b, act_b, mutable=mutables, training=False,
                )
                # Sum (not mean) over heads, so each head's gradient magnitude
                # is independent of K -- matches the historical q1 + q2 form.
                loss = sum(jnp.mean((q - q_target) ** 2) for q in qs)
                new_bundle = {k: updates.get(k, run_stats[k]) for k in run_stats}
                return loss, new_bundle

            (critic_loss, new_critic_rs), critic_grads = jax.value_and_grad(
                _critic_loss_fn, has_aux=True
            )(critic_st.params)
            critic_st = critic_st.apply_gradients(grads=critic_grads)
            critic_st = critic_st.replace(run_stats=new_critic_rs)

            # Post-step weight normalization (XQC / Salimans-Kingma).
            # When `use_weight_norm=True`, renormalize each Dense kernel after
            # the gradient step: W <- W / ||W||_axis=-2 (per-output-unit norm).
            # `normalize_last_layer=True` also renormalizes the predictor head.
            # No-op when the flag is unset, so SAC/TD3 baselines are unchanged.
            if bool(getattr(exp, 'use_weight_norm', False)):
                critic_st = critic_st.replace(
                    params=_normalize_dense_kernels(
                        critic_st.params,
                        normalize_last_layer=bool(getattr(exp, 'normalize_last_layer', True)),
                    )
                )

            # -- actor loss + extra update (delayed for TD3) --
            rng_up, rng_actor = jax.random.split(rng_up)
            do_actor_update = cls._should_update_actor(step_idx, exp)

            def _actor_loss_fn(params):
                return cls._actor_loss(
                    params, actor_st.apply_fn, actor_st.run_stats,
                    critic_st, critic_net.apply,
                    obs_b, rng_actor, exp, ex_st,
                )

            (actor_loss, actor_aux), actor_grads = jax.value_and_grad(
                _actor_loss_fn, has_aux=True
            )(actor_st.params)

            def _apply_actor(args):
                a_st, ex_st_in = args
                a_st = a_st.apply_gradients(grads=actor_grads)
                # update extra (e.g. alpha) only when we updated actor
                rng_inner = jax.random.fold_in(rng_actor, step_idx)
                new_ex_st, _ = cls._update_extra(ex_st_in, actor_aux, rng_inner, exp)
                return a_st, new_ex_st

            def _skip_actor(args):
                return args

            actor_st, ex_st = jax.lax.cond(
                do_actor_update, _apply_actor, _skip_actor, (actor_st, ex_st)
            )
            # if actor_run_stats produced inside loss aux, reapply (for SAC RunningMeanStd)
            new_actor_rs = actor_aux.get("run_stats", None) if isinstance(actor_aux, dict) else None
            if new_actor_rs is not None:
                actor_st = actor_st.replace(run_stats=new_actor_rs)

            # -- soft target update -- (params + the bundle of mutable
            # variables; jax.tree.map over an empty dict is a no-op, so the
            # bundle update is safe when BN/obs_norm are off).
            new_tgt_p = jax.tree.map(
                lambda tp, cp: tau * cp + (1.0 - tau) * tp,
                tgt_p, critic_st.params,
            )
            new_tgt_rs = jax.tree.map(
                lambda tp, cp: tau * cp + (1.0 - tau) * tp,
                new_tgt_rs, critic_st.run_stats,
            )

            new_carry = (actor_st, critic_st, new_tgt_p, new_tgt_rs,
                         ex_st, buf, rng_up)
            losses = (critic_loss, actor_loss)
            return new_carry, losses

        # ----- N gradient updates -----
        def _do_updates(actor_st, critic_st, tgt_p, tgt_rs, ex_st, buf, rng_up):
            carry = (actor_st, critic_st, tgt_p, tgt_rs, ex_st, buf, rng_up)
            carry, losses = jax.lax.scan(
                _single_gradient_update, carry, jnp.arange(gradient_steps)
            )
            actor_st, critic_st, tgt_p, tgt_rs, ex_st, _, _ = carry
            critic_loss = jnp.mean(losses[0])
            actor_loss = jnp.mean(losses[1])
            return (actor_st, critic_st, tgt_p, tgt_rs, ex_st,
                    critic_loss, actor_loss)

        def _skip_updates(actor_st, critic_st, tgt_p, tgt_rs, ex_st, buf, rng_up):
            return (actor_st, critic_st, tgt_p, tgt_rs, ex_st,
                    jnp.array(0.0), jnp.array(0.0))

        # ----- outer step -----
        def _update_step(runner_state, unused):
            (actor_state, critic_state, tgt_params, tgt_run_stats,
             ex_state, replay_buffer, env_state, last_obs, rng) = runner_state

            actor_state, replay_buffer, env_state, next_obs, rng = \
                _collect_transition(actor_state, replay_buffer, env_state,
                                    last_obs, rng, ex_state)

            rng, rng_update = jax.random.split(rng)
            result = jax.lax.cond(
                replay_buffer.size >= learning_starts,
                lambda args: _do_updates(*args),
                lambda args: _skip_updates(*args),
                (actor_state, critic_state, tgt_params, tgt_run_stats,
                 ex_state, replay_buffer, rng_update),
            )
            (actor_state, critic_state, tgt_params, tgt_run_stats,
             ex_state, critic_loss, actor_loss) = result

            log_env_state = env_state.find(LogEnvState)
            logged_metrics = log_env_state.metrics
            # `returned_episode_returns` / `returned_episode_lengths` carry the
            # last-completed episode return/length for each env, held constant
            # until that env finishes another one. Averaging across envs gives
            # a smooth running estimate. (Slots are 0 until each env has
            # completed its first episode — biases the very early curve only.)
            base_kwargs = dict(
                mean_episode_return=jnp.mean(logged_metrics.returned_episode_returns),
                mean_episode_length=jnp.mean(logged_metrics.returned_episode_lengths),
                max_timestep=jnp.max(logged_metrics.timestep * exp.num_envs),
                mean_critic_loss=critic_loss,
                mean_actor_loss=actor_loss,
                buffer_size=replay_buffer.size,
            )
            rng, rng_metrics = jax.random.split(rng)
            extra = cls._extra_metrics(actor_state, next_obs, rng_metrics,
                                       ex_state, exp)
            metric = cls._build_summary_metric(base_kwargs, extra)

            runner_state = (actor_state, critic_state, tgt_params, tgt_run_stats,
                            ex_state, replay_buffer, env_state, next_obs, rng)
            return runner_state, metric

        # ----- log-interval wrapper: run `log_every` inner steps and emit one
        # aggregated metric row. Reduces wandb log volume and smooths noise.
        log_every = max(1, int(getattr(exp, 'log_every', 100)))
        num_outer = max(1, int(exp.num_updates) // log_every)

        def _logged_step(runner_state, unused):
            runner_state, inner_metrics = jax.lax.scan(
                _update_step, runner_state, None, log_every
            )
            aggregated = jax.tree.map(lambda x: jnp.mean(x, axis=0), inner_metrics)
            # monotone counters: take the value at the end of the window
            aggregated = aggregated.replace(
                max_timestep=jnp.max(inner_metrics.max_timestep),
                buffer_size=inner_metrics.buffer_size[-1],
            )
            return runner_state, aggregated

        # ----- main scan -----
        rng, _rng = jax.random.split(rng)
        runner_state = (
            actor_state, critic_state, target_critic_params, target_critic_run_stats,
            extra_state, replay_buffer, env_state, last_obs, _rng,
        )
        runner_state, training_metrics = jax.lax.scan(
            _logged_step, runner_state, None, num_outer
        )
        (actor_state, critic_state, tgt_params, tgt_run_stats,
         ex_state, replay_buffer, env_state, last_obs, _) = runner_state

        agent_state_out = cls._agent_state(
            actor_state=actor_state,
            critic_state=critic_state,
            target_critic_params=tgt_params,
            target_critic_run_stats=tgt_run_stats,
            extra_state=ex_state,
            replay_buffer=replay_buffer,
            env_state=env_state,
            last_obs=last_obs,
        )
        return {
            "agent_state": agent_state_out,
            "training_metrics": training_metrics,
            "validation_metrics": ValidationSummary(),
        }

    # ---------- env wrap (subclasses can override) -----------------------
    @staticmethod
    def _wrap_env(env, config):
        """
        Wrap the environment for training.

        Applies a :class:`~loco_mujoco.core.wrappers.mjx.LogWrapper` and a
        :class:`~loco_mujoco.core.wrappers.mjx.VecEnv`. Rewards are not normalized.

        Args:
            env: The (unwrapped) environment.
            config (DictConfig): The ``experiment`` node of the configuration (unused).

        Returns:
            The wrapped, vectorized environment.

        """
        env = LogWrapper(env)
        env = VecEnv(env)
        # Reward normalization no longer lives on the env wrapper; off-policy
        # algorithms would need to track stats on the agent state if desired.
        return env
