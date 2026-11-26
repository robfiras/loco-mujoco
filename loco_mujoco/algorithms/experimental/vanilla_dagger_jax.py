"""
Vanilla DAgger distillation with a swappable student, teacher and trajectory.

Implements teacher-student distillation with Dataset Aggregation (DAgger; Ross et al., 2011). The rollout uses
a sticky per-environment mixture: each environment independently picks either the student or the teacher with
probability ``teacher_beta`` and commits to that choice for a uniformly random number of steps in
``[sticky_min, sticky_max]``. Whoever acts, the transition is *labeled with the teacher's action on that
observation*; that is the DAgger trick.

Training is supervised: minibatches are drawn from the buffers and the negative log-likelihood of the stored
teacher action under the student's Gaussian policy is minimized. Optionally the student's critic is trained
to match the teacher's value estimate.

Swap semantics (everything is part of the JAX pytree, so nothing has to be recompiled):

- trajectory: passed as an argument to ``train_fn(rng, agent_state, traj)``.
- teacher: ``agent_state.replace(teacher_params=..., teacher_run_stats=...)``.
- student: ``agent_state.replace(student_train_state=...)``.
- replay buffer: preserved across training chunks by default, so that data collected under one teacher stays
  useful after a teacher or trajectory swap.

Example: ``examples/training_examples/jax_distill``.

"""
import ast
from dataclasses import dataclass
from typing import Any
import warnings

import jax
import jax.numpy as jnp
from flax import struct
import flax
import optax
from omegaconf import DictConfig, OmegaConf, ListConfig

from loco_mujoco.algorithms import (JaxRLAlgorithmBase, AgentConfBase, AgentStateBase,
                                    ActorCritic, TrainState, ReplayBuffer,
                                    MetricHandlerTransition)
from loco_mujoco.algorithms.common.dataclasses import ReservoirBuffer
from loco_mujoco.core.wrappers import LogWrapper, LogEnvState, VecEnv, SummaryMetrics
from loco_mujoco.utils import MetricsHandler, ValidationSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_store_next_obs(exp) -> bool:
    """
    Return whether the replay buffer should materialize ``next_obs``.

    VanillaDagger does not use ``next_obs`` itself: the critic is trained by distilling ``V_teacher(s)``, not by
    TD bootstrapping. The option exists so that the buffer machinery stays reusable for algorithms that do need
    it. The default is False; set ``store_next_obs: true`` in the configuration to opt in.

    """
    if hasattr(exp, "store_next_obs") and exp.get("store_next_obs") is not None:
        return bool(exp.store_next_obs)
    return False


def _resolve_store_value_target(exp) -> bool:
    """
    Return whether the buffers store ``V_teacher(s)`` per transition.

    This is required for critic learning (value distillation). It follows ``use_critic_learning`` unless
    ``store_value_target`` is set explicitly.

    """
    if hasattr(exp, "store_value_target") and exp.get("store_value_target") is not None:
        return bool(exp.store_value_target)
    return bool(getattr(exp, "use_critic_learning", True))


def _resolve_long_term_buffer_size(exp) -> int:
    """
    Return the capacity of the reservoir-sampled long-term memory (``long_term_buffer_size``).

    A capacity of 0 disables it. Defaults to 1,000,000.

    """
    if hasattr(exp, "long_term_buffer_size") and exp.get("long_term_buffer_size") is not None:
        return int(exp.long_term_buffer_size)
    return 1_000_000


def _resolve_long_term_min_include_prob(exp) -> float:
    """
    Return the floor on the reservoir's per-item inclusion probability (``long_term_min_include_prob``).

    A value of 0 keeps pure reservoir semantics (uniform over all history). A positive value caps the effective
    denominator, so the buffer approximates a uniform sample over the last ``capacity / p_min`` transitions. This
    is useful in non-stationary settings like DAgger, where the state distribution of the rollouts drifts as the
    student learns. Defaults to 0.0.

    """
    if hasattr(exp, "long_term_min_include_prob") and exp.get("long_term_min_include_prob") is not None:
        return float(exp.long_term_min_include_prob)
    return 0.0


# ---------------------------------------------------------------------------
# Metrics + rollout state
# ---------------------------------------------------------------------------

@struct.dataclass
class DaggerSummaryMetrics(SummaryMetrics):
    """
    Training metrics logged by VanillaDagger once per training iteration.

    Extends :class:`~loco_mujoco.core.wrappers.mjx.SummaryMetrics`. The episode return and length are the means
    over all environments of their last completed episode. The losses are means over the gradient steps of the
    iteration and zero while no updates are performed.

    Attributes:
        mean_bc_loss (float): Negative log-likelihood of the teacher actions under the student policy (the
            quantity that is minimized).
        mean_mse_action (float): Diagnostic: mean squared error between the student's mean action and the teacher
            action.
        mean_critic_loss (float): Mean squared error between the student's and the teacher's value estimate; zero
            if critic learning is off.
        buffer_size (int): Number of transitions in the short-term replay buffer.
        frac_teacher_steps (float): Fraction of environments currently controlled by the teacher.

    """
    mean_bc_loss: float = 0.0       # actor NLL (what gets minimised)
    mean_mse_action: float = 0.0    # diagnostic: ||pi.mean - a_teacher||^2
    mean_critic_loss: float = 0.0   # MSE to V_teacher, 0 when critic learning is off
    buffer_size: int = 0
    frac_teacher_steps: float = 0.0


@struct.dataclass
class DaggerRolloutState:
    """
    State of the sticky student/teacher mixture, one entry per environment.

    Attributes:
        using_teacher (jax.Array): Boolean array of shape ``(num_envs,)``; True where the teacher currently
            controls the environment.
        steps_remaining (jax.Array): Integer array of shape ``(num_envs,)`` with the number of steps left before
            the choice is resampled for the environment.

    """
    using_teacher: jnp.ndarray
    steps_remaining: jnp.ndarray

    @classmethod
    def create(cls, num_envs: int, rng, beta: float, h_min: int, h_max: int):
        """
        Sample an initial mixture state.

        Args:
            num_envs (int): Number of parallel environments.
            rng (jax.Array): PRNG key.
            beta (float): Probability of picking the teacher.
            h_min (int): Minimum number of steps a choice is kept.
            h_max (int): Maximum number of steps a choice is kept (inclusive).

        Returns:
            DaggerRolloutState: The initial mixture state.

        """
        k1, k2 = jax.random.split(rng)
        using_teacher = jax.random.uniform(k1, (num_envs,)) < beta
        steps_remaining = jax.random.randint(k2, (num_envs,), h_min, h_max + 1)
        return cls(using_teacher=using_teacher, steps_remaining=steps_remaining)


# ---------------------------------------------------------------------------
# Agent conf / state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VanillaDaggerAgentConf(AgentConfBase):
    """
    Static configuration of a VanillaDagger agent.

    Attributes:
        config (DictConfig): The full configuration; the hyperparameters live under ``config.experiment``.
        student_net (ActorCritic): The network definition of the student.
        teacher_net (ActorCritic): The network definition of the teacher.
        student_tx: The Optax optimizer of the student.

    """
    config: DictConfig
    student_net: ActorCritic
    teacher_net: ActorCritic
    student_tx: Any

    def serialize(self):
        """
        Serialize the agent configuration and the network definitions.

        The optimizer is not serialized; it is rebuilt from the configuration in :meth:`from_dict`.

        Returns:
            dict: Dictionary with the keys ``"config"``, ``"student_net"`` and ``"teacher_net"``.

        """
        conf_dict = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
        return {
            "config": conf_dict,
            "student_net": flax.serialization.to_state_dict(self.student_net),
            "teacher_net": flax.serialization.to_state_dict(self.teacher_net),
        }

    @classmethod
    def from_dict(cls, d):
        """
        Restore the agent configuration from a dictionary.

        The networks are rebuilt from the configuration, which contains the resolved observation indices, so no
        environment is needed.

        Args:
            d (dict): Dictionary created by :meth:`serialize`.

        Returns:
            VanillaDaggerAgentConf: The restored configuration with a newly built optimizer.

        """
        config = OmegaConf.create(d["config"])
        student_tx = VanillaDaggerJax._get_optimizer(config)
        student_net = flax.serialization.from_state_dict(
            VanillaDaggerJax._build_net(config, role="student"), d["student_net"]
        )
        teacher_net = flax.serialization.from_state_dict(
            VanillaDaggerJax._build_net(config, role="teacher"), d["teacher_net"]
        )
        return cls(config=config, student_net=student_net,
                   teacher_net=teacher_net, student_tx=student_tx)


@struct.dataclass
class VanillaDaggerAgentState(AgentStateBase):
    """
    Dynamic state of a VanillaDagger agent.

    All swappable entities live here: ``student_train_state`` and ``teacher_params``/``teacher_run_stats`` (the
    trajectory is passed separately to the training function).

    There are two buffers: ``replay_buffer`` is a short-term rolling window of the most recent transitions,
    ``long_term_buffer`` is a reservoir-sampled long-term memory (approximately uniform over all transitions ever
    collected). At training time half of every batch is sampled from each to mitigate forgetting across
    teacher/trajectory swaps.

    Attributes:
        student_train_state (TrainState): Parameters, optimizer state and running statistics of the student.
        teacher_params: Parameters of the frozen teacher network.
        teacher_run_stats: Running observation statistics of the teacher network.
        replay_buffer (ReplayBuffer): Short-term replay buffer; created by the training function if ``None``.
        long_term_buffer (ReservoirBuffer): Long-term reservoir; created by the training function if ``None``.
        rollout_state (DaggerRolloutState): State of the student/teacher mixture; created by the training
            function if ``None``.
        env_state: Environment state at the end of the last training chunk; not saved to disk.
        last_obs (jax.Array): Last observations of the previous training chunk; not saved to disk.

    """
    student_train_state: TrainState
    teacher_params: Any
    teacher_run_stats: Any
    replay_buffer: Any = None          # ReplayBuffer; lazily initialised in _train_fn
    long_term_buffer: Any = None       # ReservoirBuffer; lazily initialised
    rollout_state: Any = None          # DaggerRolloutState; lazily initialised
    env_state: Any = None
    last_obs: Any = None

    def serialize(self):
        """
        Serialize the agent state.

        Only the student train state and the teacher parameters and running statistics are serialized; the buffers,
        the mixture state, ``env_state`` and ``last_obs`` are dropped.

        Returns:
            dict: Dictionary with the keys ``"student_train_state"``, ``"teacher_params"`` and
            ``"teacher_run_stats"``.

        """
        return {
            "student_train_state": flax.serialization.to_state_dict(self.student_train_state),
            "teacher_params": flax.serialization.to_state_dict(self.teacher_params),
            "teacher_run_stats": flax.serialization.to_state_dict(self.teacher_run_stats),
        }

    @classmethod
    def from_dict(cls, d, agent_conf):
        """
        Restore the agent state from a dictionary.

        The buffers are created empty and the mixture state is freshly sampled, at the configured shapes, so that
        the loaded state has the same pytree structure as one produced by ``init_agent_state``.

        Args:
            d (dict): Dictionary created by :meth:`serialize`.
            agent_conf (VanillaDaggerAgentConf): Agent configuration providing the network and the optimizer.

        Returns:
            VanillaDaggerAgentState: The restored agent state; ``env_state`` and ``last_obs`` are ``None``.

        """
        student_net = agent_conf.student_net
        train_state = TrainState.create(
            apply_fn=student_net.apply,
            tx=agent_conf.student_tx,
            params=d["student_train_state"]["params"],
            run_stats=d["student_train_state"]["run_stats"],
        )
        opt_state = flax.serialization.from_state_dict(
            train_state.opt_state, d["student_train_state"]["opt_state"]
        )
        train_state = train_state.replace(opt_state=opt_state)
        # Rebuild buffers + rollout_state at the configured shape so the
        # loaded agent has the same pytree structure as one produced by
        # init_agent_state — avoids a recompile on the first train_fn call.
        exp = agent_conf.config.experiment
        replay_buffer = ReplayBuffer.create(
            int(exp.obs_dim), int(exp.action_dim), int(exp.buffer_size),
            store_next_obs=_resolve_store_next_obs(exp),
            store_value_target=_resolve_store_value_target(exp),
        )
        long_term_buffer = ReservoirBuffer.create(
            int(exp.obs_dim), int(exp.action_dim),
            _resolve_long_term_buffer_size(exp),
            store_value_target=_resolve_store_value_target(exp),
            min_include_prob=_resolve_long_term_min_include_prob(exp),
        )
        rollout_state = DaggerRolloutState.create(
            int(exp.num_envs), jax.random.PRNGKey(0),
            beta=float(getattr(exp, "teacher_beta", 0.5)),
            h_min=int(getattr(exp, "sticky_min", 5)),
            h_max=int(getattr(exp, "sticky_max", 50)),
        )
        return cls(
            student_train_state=train_state,
            teacher_params=d["teacher_params"],
            teacher_run_stats=d["teacher_run_stats"],
            replay_buffer=replay_buffer,
            long_term_buffer=long_term_buffer,
            rollout_state=rollout_state,
        )


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class VanillaDaggerJax(JaxRLAlgorithmBase):
    """
    DAgger distillation with a frozen teacher and a trainable student, both
    :class:`~loco_mujoco.algorithms.common.networks.ActorCritic` networks.

    The training loop interleaves one environment step (per parallel environment) with ``gradient_steps``
    behavior-cloning updates, once the buffers contain at least ``learning_starts`` transitions. See the
    docstring of this module for the rollout mixture and the swap semantics. Usage::

        agent_conf = VanillaDaggerJax.init_agent_conf(env, config)
        agent_state = VanillaDaggerJax.init_agent_state(
            env, agent_conf, rng,
            teacher_params=teacher_state.train_state.params,
            teacher_run_stats=teacher_state.train_state.run_stats)
        train_fn = jax.jit(VanillaDaggerJax.build_train_fn(env, agent_conf))
        out = train_fn(rng, agent_state, traj)

    where ``teacher_state`` is, for example, a loaded
    :class:`~loco_mujoco.algorithms.ppo_jax.PPOAgentState`. Complete scripts are in
    ``examples/training_examples/jax_distill``.

    """

    _agent_conf = VanillaDaggerAgentConf
    _agent_state = VanillaDaggerAgentState

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    @classmethod
    def _build_net(cls, config: DictConfig, role: str) -> ActorCritic:
        """
        Build the :class:`~loco_mujoco.algorithms.common.networks.ActorCritic` of the student or the teacher.

        Reads ``config.experiment.<role>.*`` if present and falls back to ``config.experiment.*`` otherwise, so a
        minimal configuration with only shared fields works out of the box.

        ``actor_obs_ind`` and ``critic_obs_ind`` (lists of integers) are set by :meth:`init_agent_conf` after
        resolving ``actor_obs_group`` and ``critic_obs_group`` against the environment. They survive saving and
        loading because they live in the configuration.

        Args:
            config (DictConfig): The full configuration.
            role (str): ``"student"`` or ``"teacher"``.

        Returns:
            ActorCritic: The network definition.

        """
        exp = config.experiment
        sub = getattr(exp, role, None)

        def field(name, default=None):
            if sub is not None and name in sub:
                return sub[name]
            return getattr(exp, name, default)

        hidden_layers = field("hidden_layers", [512, 256])
        if not isinstance(hidden_layers, (list, ListConfig)):
            hidden_layers = ast.literal_eval(hidden_layers)

        kwargs = dict(
            action_dim=int(exp.action_dim),
            activation=str(field("activation", "tanh")),
            init_std=float(field("init_std", 1.0)),
            learnable_std=bool(field("learnable_std", True)),
            hidden_layer_dims=tuple(hidden_layers),
        )
        actor_ind = field("actor_obs_ind", None)
        critic_ind = field("critic_obs_ind", None)
        if actor_ind is not None:
            kwargs["actor_obs_ind"] = jnp.asarray(list(actor_ind), dtype=jnp.int32)
        if critic_ind is not None:
            kwargs["critic_obs_ind"] = jnp.asarray(list(critic_ind), dtype=jnp.int32)
        return ActorCritic(**kwargs)

    @classmethod
    def _get_optimizer(cls, config: DictConfig):
        """
        Build the optimizer of the student: gradient clipping by global norm followed by AdamW.

        Args:
            config (DictConfig): The full configuration; ``lr``, ``max_grad_norm`` (default 0.5) and
                ``weight_decay`` (default 0.0) are read from ``config.experiment``.

        Returns:
            optax.GradientTransformation: The optimizer.

        """
        exp = config.experiment
        max_grad = float(getattr(exp, "max_grad_norm", 0.5))
        lr = float(exp.lr)
        weight_decay = float(getattr(exp, "weight_decay", 0.0))
        return optax.chain(
            optax.clip_by_global_norm(max_grad),
            optax.adamw(lr, weight_decay=weight_decay, eps=1e-5),
        )

    # ------------------------------------------------------------------
    # Conf / state init
    # ------------------------------------------------------------------

    @classmethod
    def init_agent_conf(cls, env, config: DictConfig):
        """
        Create the static agent configuration: derived hyperparameters, the networks and the optimizer.

        The hyperparameters are read from ``config.experiment``:

        - ``student``, ``teacher``: sub-configurations with ``hidden_layers``, ``activation``, ``init_std``,
          ``learnable_std`` and optionally ``actor_obs_group`` and ``critic_obs_group``. Keys missing there are
          looked up at the top level. The teacher settings must match the loaded teacher checkpoint.
        - ``teacher_beta``, ``sticky_min``, ``sticky_max``: probability of acting with the teacher and the range of
          the number of steps a choice is kept.
        - ``use_critic_learning``, ``critic_loss_coef``: whether the student's critic is trained to match the
          teacher's value estimate, and the weight of that loss.
        - ``buffer_size``, ``long_term_buffer_size``, ``long_term_min_include_prob``: capacity of the short-term
          buffer, and capacity (0 disables it) and minimum inclusion probability of the long-term reservoir.
        - ``batch_size``, ``learning_starts``, ``gradient_steps``: minibatch size, number of stored transitions
          before learning starts and gradient steps per environment step.
        - ``lr``, ``max_grad_norm``, ``weight_decay``: optimizer settings of the student.
        - ``num_envs``, ``total_timesteps``: number of parallel environments and number of environment steps per
          call of the training function.

        The keys ``obs_dim``, ``action_dim``, ``num_updates`` (``total_timesteps // num_envs``) and, if observation
        groups are used, ``<role>.actor_obs_ind`` and ``<role>.critic_obs_ind`` are written to ``config.experiment``
        (the configuration is modified in place).

        Args:
            env: The (unwrapped) environment the agent is trained on.
            config (DictConfig): Configuration with the hyperparameters under ``config.experiment``.

        Returns:
            VanillaDaggerAgentConf: The agent configuration.

        """
        from omegaconf import open_dict, OmegaConf
        obs_dim = int(env.info.observation_space.shape[0])

        def _resolve_group(group):
            """Group name → list[int] observation indices, or None if no filter."""
            if group is None:
                return None
            return [int(i) for i in env.obs_container.get_obs_ind_by_group(group)]

        with open_dict(config.experiment):
            config.experiment.num_updates = int(config.experiment.total_timesteps) // int(config.experiment.num_envs)
            config.experiment.obs_dim = obs_dim
            config.experiment.action_dim = int(env.info.action_space.shape[0])

            # Optional top-level filters (fallback if a role doesn't specify its own).
            top_actor_group = config.experiment.get("actor_obs_group", None)
            top_critic_group = config.experiment.get("critic_obs_group", None)

            # Resolve per-role obs filters against the env and bake the
            # resulting index lists into the config. This keeps from_dict
            # env-independent — the loaded conf rebuilds the same network.
            for role in ("student", "teacher"):
                if config.experiment.get(role, None) is None:
                    config.experiment[role] = OmegaConf.create({})
                sub = config.experiment[role]
                actor_group = sub.get("actor_obs_group", top_actor_group)
                critic_group = sub.get("critic_obs_group", top_critic_group)
                actor_ind = _resolve_group(actor_group)
                critic_ind = _resolve_group(critic_group)
                if actor_ind is not None:
                    sub.actor_obs_ind = actor_ind
                if critic_ind is not None:
                    sub.critic_obs_ind = critic_ind

        student_net = cls._build_net(config, role="student")
        teacher_net = cls._build_net(config, role="teacher")
        student_tx = cls._get_optimizer(config)
        return cls._agent_conf(config=config, student_net=student_net,
                               teacher_net=teacher_net, student_tx=student_tx)

    @classmethod
    def init_agent_state(cls, env, agent_conf: VanillaDaggerAgentConf, rng,
                         teacher_params=None, teacher_run_stats=None):
        """
        Initialize the agent state: the student, the teacher, empty buffers and the rollout-mixture state.

        Args:
            env: The environment (unused; the dimensions are taken from the configuration).
            agent_conf (VanillaDaggerAgentConf): Agent configuration created by :meth:`init_agent_conf`.
            rng (jax.Array): PRNG key.
            teacher_params: Parameters of a pretrained teacher, e.g. ``train_state.params`` of a PPO agent.
            teacher_run_stats: Running statistics of the pretrained teacher, e.g. ``train_state.run_stats``. If
                either of the two is omitted, the teacher is randomly initialized (with a warning), which is only
                useful as a placeholder.

        Returns:
            VanillaDaggerAgentState: The initial agent state; ``env_state`` and ``last_obs`` are ``None`` and get
            initialized by the training function.

        """
        exp = agent_conf.config.experiment
        rng, rng_s, rng_t = jax.random.split(rng, 3)
        init_x = jnp.zeros((1, int(exp.obs_dim)))

        student_init = agent_conf.student_net.init(rng_s, init_x)
        student_train_state = TrainState.create(
            apply_fn=agent_conf.student_net.apply,
            params=student_init["params"],
            run_stats=student_init["run_stats"],
            tx=agent_conf.student_tx,
        )

        if teacher_params is None or teacher_run_stats is None:
            warnings.warn(
                "VanillaDagger: teacher params/run_stats not provided — using "
                "random init. Call `.replace(teacher_params=..., teacher_run_stats=...)` "
                "with a pretrained checkpoint before training."
            )
            teacher_init = agent_conf.teacher_net.init(rng_t, init_x)
            teacher_params = teacher_init["params"]
            teacher_run_stats = teacher_init["run_stats"]

        # Eagerly build the replay buffer and rollout-mixture state here rather
        # than lazily inside _train_fn. Keeps the agent_state pytree structure
        # stable across chunks so the jitted train_fn doesn't recompile when
        # the user passes a state with a populated buffer on chunk 2+.
        rng, rng_r = jax.random.split(rng)
        replay_buffer = ReplayBuffer.create(
            int(exp.obs_dim), int(exp.action_dim), int(exp.buffer_size),
            store_next_obs=_resolve_store_next_obs(exp),
            store_value_target=_resolve_store_value_target(exp),
        )
        long_term_buffer = ReservoirBuffer.create(
            int(exp.obs_dim), int(exp.action_dim),
            _resolve_long_term_buffer_size(exp),
            store_value_target=_resolve_store_value_target(exp),
            min_include_prob=_resolve_long_term_min_include_prob(exp),
        )
        rollout_state = DaggerRolloutState.create(
            int(exp.num_envs), rng_r,
            beta=float(getattr(exp, "teacher_beta", 0.5)),
            h_min=int(getattr(exp, "sticky_min", 5)),
            h_max=int(getattr(exp, "sticky_max", 50)),
        )

        return cls._agent_state(
            student_train_state=student_train_state,
            teacher_params=teacher_params,
            teacher_run_stats=teacher_run_stats,
            replay_buffer=replay_buffer,
            long_term_buffer=long_term_buffer,
            rollout_state=rollout_state,
        )

    # ------------------------------------------------------------------
    # Env wrap
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_env(env, config):
        """
        Wrap the environment for training.

        Applies a :class:`~loco_mujoco.core.wrappers.mjx.LogWrapper` and a
        :class:`~loco_mujoco.core.wrappers.mjx.VecEnv`. Rewards are not used and hence not normalized.

        Args:
            env: The (unwrapped) environment.
            config (DictConfig): The ``experiment`` node of the configuration (unused).

        Returns:
            The wrapped, vectorized environment.

        """
        env = LogWrapper(env)
        env = VecEnv(env)
        # DAgger doesn't use reward, so no reward normalization.
        return env

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def _train_fn(cls, rng, env,
                  agent_conf: VanillaDaggerAgentConf,
                  agent_state: VanillaDaggerAgentState = None,
                  traj=None,
                  mh: MetricsHandler = None):
        """
        Run DAgger for ``num_updates`` iterations.

        Each iteration steps all environments once. Per environment the acting policy is the teacher (mean action)
        or the student (sampled action), according to the sticky mixture; the observation is stored together with
        the teacher's mean action and, with critic learning, the teacher's value estimate in the short-term buffer
        and in the long-term reservoir. Once the buffers hold at least ``learning_starts`` transitions each,
        ``gradient_steps`` updates are performed on batches that are drawn half from each buffer (entirely from the
        short-term buffer if the reservoir is disabled). The loss is the negative log-likelihood of the teacher
        action plus, with critic learning, ``critic_loss_coef`` times the squared error to the teacher value. The
        teacher is never updated. If ``agent_state`` carries an environment state from a previous call, training
        continues from it.

        Args:
            rng (jax.Array): PRNG key.
            env: The (unwrapped) environment.
            agent_conf (VanillaDaggerAgentConf): Static agent configuration.
            agent_state (VanillaDaggerAgentState): Agent state holding the teacher; required.
            traj: Trajectory data passed on to ``env.reset`` and ``env.step``; ``None`` if the task has no trajectory.
            mh (MetricsHandler): Unused; use :meth:`_eval_fn` (via ``build_eval_fn``) for validation.

        Returns:
            dict: Dictionary with the keys

            - ``"agent_state"`` (:class:`VanillaDaggerAgentState`): the agent state after training, including the
              buffers, the mixture state, the last environment state and the last observation.
            - ``"training_metrics"`` (:class:`DaggerSummaryMetrics`): metrics with a leading axis of length
              ``num_updates``.
            - ``"validation_metrics"`` (:class:`~loco_mujoco.utils.metrics.ValidationSummary`): always empty.

        Raises:
            ValueError: If ``agent_state`` is ``None``, or if critic learning is enabled but a buffer does not store
                value targets.

        """
        if agent_state is None:
            raise ValueError(
                "VanillaDagger._train_fn requires an agent_state with teacher "
                "params. Build it via init_agent_state(..., teacher_params=..., "
                "teacher_run_stats=...)."
            )

        exp = agent_conf.config.experiment
        student_net = agent_conf.student_net
        teacher_net = agent_conf.teacher_net
        env = cls._wrap_env(env, exp)

        # Reattach apply_fn (not serializable through flax save)
        student_ts = agent_state.student_train_state.replace(apply_fn=student_net.apply)
        teacher_params = agent_state.teacher_params
        teacher_run_stats = agent_state.teacher_run_stats

        # Hyperparameters
        num_envs = int(exp.num_envs)
        batch_size = int(exp.batch_size)
        learning_starts = int(getattr(exp, "learning_starts", batch_size))
        gradient_steps = int(getattr(exp, "gradient_steps", 1))
        teacher_beta = float(getattr(exp, "teacher_beta", 0.5))
        sticky_min = int(getattr(exp, "sticky_min", 5))
        sticky_max = int(getattr(exp, "sticky_max", 50))
        # Critic learning via teacher-value distillation: store V_teacher(s)
        # at collect time and minimise MSE(V_student(s), V_teacher(s)).
        # No bootstrap, no next_obs required.
        use_critic_learning = bool(getattr(exp, "use_critic_learning", True))
        critic_loss_coef = float(getattr(exp, "critic_loss_coef", 0.5))

        # ENV (reuse carry if present)
        if agent_state.env_state is not None:
            env_state = agent_state.env_state
            last_obs = agent_state.last_obs
        else:
            rng, _rng = jax.random.split(rng)
            last_obs, env_state = env.reset(jax.random.split(_rng, num_envs), traj)

        # SHORT-TERM REPLAY BUFFER (rolling FIFO, preserve across chunks —
        # data from old (expert, traj) stays fresh after each swap).
        if agent_state.replay_buffer is not None:
            buf = agent_state.replay_buffer
        else:
            buf = ReplayBuffer.create(int(exp.obs_dim), int(exp.action_dim),
                                      int(exp.buffer_size),
                                      store_next_obs=_resolve_store_next_obs(exp),
                                      store_value_target=_resolve_store_value_target(exp))

        # LONG-TERM MEMORY (reservoir-sampled). Size 0 disables it.
        long_capacity = _resolve_long_term_buffer_size(exp)
        has_long_term = long_capacity > 0
        if agent_state.long_term_buffer is not None:
            long_buf = agent_state.long_term_buffer
        else:
            long_buf = ReservoirBuffer.create(
                int(exp.obs_dim), int(exp.action_dim), long_capacity,
                store_value_target=_resolve_store_value_target(exp),
                min_include_prob=_resolve_long_term_min_include_prob(exp),
            )

        if use_critic_learning and not buf.store_value_target:
            raise ValueError(
                "use_critic_learning=True requires a replay buffer with "
                "store_value_target=True (V_teacher is the critic target). "
                "Either leave store_value_target unset (auto-ties to "
                "use_critic_learning) or set use_critic_learning=false."
            )
        if has_long_term and use_critic_learning and not long_buf.store_value_target:
            raise ValueError(
                "use_critic_learning=True also requires the long-term "
                "reservoir to store value_target."
            )

        # ROLLOUT MIXTURE STATE
        if agent_state.rollout_state is not None:
            rollout_st = agent_state.rollout_state
        else:
            rng, _rng = jax.random.split(rng)
            rollout_st = DaggerRolloutState.create(
                num_envs, _rng, teacher_beta, sticky_min, sticky_max
            )

        # --------------------------------------------------------------
        # Helpers
        # --------------------------------------------------------------

        def _teacher_mean_and_value(obs):
            # Frozen teacher. RunningMeanStd always writes, so we pass
            # mutable=["run_stats"] to keep Flax happy but drop the updates —
            # teacher_run_stats stays put. (The ephemeral per-batch Welford
            # update is negligible against the teacher's converged stats.)
            # Returns (mean_action, value_estimate).
            (pi, v), _ = teacher_net.apply(
                {"params": teacher_params, "run_stats": teacher_run_stats}, obs,
                mutable=["run_stats"],
            )
            return pi.mean(), v

        def _student_sample(s_ts, obs, rng_s):
            (pi, _), updates = student_net.apply(
                {"params": s_ts.params, "run_stats": s_ts.run_stats}, obs,
                mutable=["run_stats"],
            )
            s_ts = s_ts.replace(run_stats=updates["run_stats"])
            return pi.sample(seed=rng_s), s_ts

        def _resample_expired(rollout_st, rng_r):
            expired = rollout_st.steps_remaining <= 0
            k1, k2 = jax.random.split(rng_r)
            new_teacher = jax.random.uniform(k1, rollout_st.using_teacher.shape) < teacher_beta
            new_remaining = jax.random.randint(
                k2, rollout_st.steps_remaining.shape, sticky_min, sticky_max + 1
            )
            return DaggerRolloutState(
                using_teacher=jnp.where(expired, new_teacher, rollout_st.using_teacher),
                steps_remaining=jnp.where(expired, new_remaining, rollout_st.steps_remaining),
            )

        def _collect_transition(s_ts, buf, long_buf, rollout_st,
                                env_state, last_obs, rng):
            # Compute student + teacher actions in parallel (and teacher value
            # to use as the critic-distillation target).
            rng, rng_s = jax.random.split(rng)
            a_student, s_ts = _student_sample(s_ts, last_obs, rng_s)
            a_teacher, v_teacher = _teacher_mean_and_value(last_obs)

            # Per-env sticky pick
            mask = rollout_st.using_teacher[:, None]
            acting = jnp.where(mask, a_teacher, a_student)

            # Step env with the acting policy
            next_obs, reward, absorbing, done, info, env_state = env.step(
                env_state, acting, traj
            )

            # Label with teacher action + teacher value regardless of who acted.
            # If store_value_target is False, the buffer ignores v_teacher.
            value_target = v_teacher if buf.store_value_target else None
            buf = buf.add_batch(last_obs, next_obs, a_teacher, reward,
                                done.astype(jnp.float32),
                                value_target=value_target)
            if has_long_term:
                rng, rng_long = jax.random.split(rng)
                long_value = v_teacher if long_buf.store_value_target else None
                long_buf = long_buf.add_batch(last_obs, a_teacher, long_value, rng_long)

            # Sticky countdown + resample expired slots
            rollout_st = rollout_st.replace(
                steps_remaining=rollout_st.steps_remaining - 1
            )
            rng, rng_r = jax.random.split(rng)
            rollout_st = _resample_expired(rollout_st, rng_r)

            return s_ts, buf, long_buf, rollout_st, env_state, next_obs, rng

        # --------------------------------------------------------------
        # Update (actor BC + optional critic distillation)
        # --------------------------------------------------------------

        def _update_loss(params, s_ts, obs_b, act_b, value_target_b):
            (pi, v_s), updates = student_net.apply(
                {"params": params, "run_stats": s_ts.run_stats}, obs_b,
                mutable=["run_stats"],
            )
            actor_loss = -jnp.mean(pi.log_prob(act_b))
            mse_action = jnp.mean((pi.mean() - act_b) ** 2)

            if use_critic_learning:
                # Teacher-value distillation: match V_teacher(s).
                critic_loss = jnp.mean((v_s - value_target_b) ** 2)
                total_loss = actor_loss + critic_loss_coef * critic_loss
            else:
                critic_loss = jnp.array(0.0)
                total_loss = actor_loss

            return total_loss, (updates["run_stats"], actor_loss, critic_loss, mse_action)

        # Half-split batch between short-term and long-term memory (when the
        # long-term reservoir is active). If not active, we sample the full
        # batch from short-term. `half` is static at trace time.
        if has_long_term:
            half = batch_size // 2
            short_half = batch_size - half
        else:
            half = 0
            short_half = batch_size

        def _single_update(carry, _):
            s_ts, buf, long_buf, rng = carry
            rng, rng_short, rng_long = jax.random.split(rng, 3)

            # Short-term sample
            short_idx = jax.random.randint(rng_short, (short_half,), 0, buf.size)
            obs_short = buf.obs[short_idx]
            act_short = buf.action[short_idx]
            if use_critic_learning:
                vt_short = buf.value_target[short_idx]
            else:
                vt_short = jnp.zeros((short_half,), dtype=buf.obs.dtype)

            # Long-term sample (only if the reservoir is active and has data)
            if has_long_term:
                long_idx = jax.random.randint(rng_long, (half,), 0,
                                              jnp.maximum(1, long_buf.size))
                obs_long = long_buf.obs[long_idx]
                act_long = long_buf.action[long_idx]
                if use_critic_learning:
                    vt_long = long_buf.value_target[long_idx]
                else:
                    vt_long = jnp.zeros((half,), dtype=long_buf.obs.dtype)

                obs_b = jnp.concatenate([obs_short, obs_long], axis=0)
                act_b = jnp.concatenate([act_short, act_long], axis=0)
                value_target_b = jnp.concatenate([vt_short, vt_long], axis=0)
            else:
                obs_b, act_b, value_target_b = obs_short, act_short, vt_short

            (_, aux), grads = jax.value_and_grad(_update_loss, has_aux=True)(
                s_ts.params, s_ts, obs_b, act_b, value_target_b
            )
            new_rs, actor_loss, critic_loss, mse_action = aux
            s_ts = s_ts.apply_gradients(grads=grads)
            s_ts = s_ts.replace(run_stats=new_rs)
            return (s_ts, buf, long_buf, rng), (actor_loss, critic_loss, mse_action)

        def _do_updates(s_ts, buf, long_buf, rng):
            (s_ts, _, _, _), aux = jax.lax.scan(
                _single_update, (s_ts, buf, long_buf, rng),
                jnp.arange(gradient_steps)
            )
            return s_ts, jnp.mean(aux[0]), jnp.mean(aux[1]), jnp.mean(aux[2])

        def _skip_updates(s_ts, buf, long_buf, rng):
            return s_ts, jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)

        # --------------------------------------------------------------
        # Outer step
        # --------------------------------------------------------------

        def _update_step(runner_state, unused):
            s_ts, buf, long_buf, rollout_st, env_state, last_obs, rng = runner_state

            s_ts, buf, long_buf, rollout_st, env_state, next_obs, rng = _collect_transition(
                s_ts, buf, long_buf, rollout_st, env_state, last_obs, rng
            )

            rng, rng_upd = jax.random.split(rng)
            # Start updates once both buffers have enough data (long_buf.size
            # ramps up from 0 via reservoir fill). When long-term is off,
            # long_buf.size stays 0 but has_long_term=False short-circuits the
            # condition below.
            if has_long_term:
                ready = (buf.size >= learning_starts) & (long_buf.size >= learning_starts)
            else:
                ready = buf.size >= learning_starts
            s_ts, bc_loss, critic_loss, mse_action = jax.lax.cond(
                ready,
                lambda args: _do_updates(*args),
                lambda args: _skip_updates(*args),
                (s_ts, buf, long_buf, rng_upd),
            )

            # Metrics. `returned_episode_returns` / `returned_episode_lengths`
            # carry the last-completed episode return/length for each env,
            # held constant until that env finishes another one. Averaging
            # across envs gives a smooth running estimate. (Slots are 0 until
            # each env has completed its first episode — biases the very
            # early curve only.) See dev commit 42c6413 (sac/td3 fix).
            logged = env_state.find(LogEnvState).metrics
            metric = DaggerSummaryMetrics(
                mean_episode_return=jnp.mean(logged.returned_episode_returns),
                mean_episode_length=jnp.mean(logged.returned_episode_lengths),
                max_timestep=jnp.max(logged.timestep * num_envs),
                mean_bc_loss=bc_loss,
                mean_mse_action=mse_action,
                mean_critic_loss=critic_loss,
                buffer_size=buf.size,
                frac_teacher_steps=jnp.mean(rollout_st.using_teacher.astype(jnp.float32)),
            )

            runner_state = (s_ts, buf, long_buf, rollout_st, env_state, next_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (student_ts, buf, long_buf, rollout_st, env_state, last_obs, _rng)
        runner_state, training_metrics = jax.lax.scan(
            _update_step, runner_state, None, int(exp.num_updates)
        )
        student_ts, buf, long_buf, rollout_st, env_state, last_obs, _ = runner_state

        agent_state_out = cls._agent_state(
            student_train_state=student_ts,
            teacher_params=teacher_params,
            teacher_run_stats=teacher_run_stats,
            replay_buffer=buf,
            long_term_buffer=long_buf,
            rollout_state=rollout_st,
            env_state=env_state,
            last_obs=last_obs,
        )
        return {
            "agent_state": agent_state_out,
            "training_metrics": training_metrics,
            "validation_metrics": ValidationSummary(),
        }

    # ------------------------------------------------------------------
    # Evaluation (standalone, same signature as train_fn)
    # ------------------------------------------------------------------

    @classmethod
    def _eval_fn(cls, rng, env,
                 agent_conf: VanillaDaggerAgentConf,
                 agent_state: VanillaDaggerAgentState,
                 traj=None,
                 mh: MetricsHandler = None):
        """
        Run a deterministic student-only rollout for evaluation.

        Resets ``eval_num_envs`` (default: ``num_envs``) fresh environments and runs the student policy with its
        mean action for ``eval_num_steps`` steps (default 500). The teacher, the buffers and the training state are
        not touched.

        Args:
            rng (jax.Array): PRNG key.
            env: The (unwrapped) environment.
            agent_conf (VanillaDaggerAgentConf): Static agent configuration.
            agent_state (VanillaDaggerAgentState): Agent state holding the student to evaluate.
            traj: Trajectory data passed on to ``env.reset`` and ``env.step``; ``None`` if the task has no trajectory.
            mh (MetricsHandler): Optional metrics handler computing the validation metrics.

        Returns:
            dict: Dictionary with the keys ``"eval_summary"``
            (:class:`~loco_mujoco.core.wrappers.mjx.SummaryMetrics` with the mean return and length of the episodes
            finished during the rollout) and ``"validation_metrics"``
            (:class:`~loco_mujoco.utils.metrics.ValidationSummary`; empty if ``mh`` is ``None``).

        """
        exp = agent_conf.config.experiment
        student_net = agent_conf.student_net
        env = cls._wrap_env(env, exp)

        student_ts = agent_state.student_train_state.replace(apply_fn=student_net.apply)

        eval_num_envs = int(getattr(exp, "eval_num_envs", exp.num_envs))
        eval_num_steps = int(getattr(exp, "eval_num_steps", 500))

        def _eval_step(carry, unused):
            s_ts, env_state, last_obs, rng = carry
            (pi, _), updates = student_net.apply(
                {"params": s_ts.params, "run_stats": s_ts.run_stats}, last_obs,
                mutable=["run_stats"],
            )
            s_ts = s_ts.replace(run_stats=updates["run_stats"])
            action = pi.mean()  # deterministic

            obs, reward, absorbing, done, info, env_state = env.step(
                env_state, action, traj
            )
            logged = env_state.find(LogEnvState).metrics
            transition = MetricHandlerTransition(env_state, logged)
            return (s_ts, env_state, obs, rng), transition

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, eval_num_envs)
        last_obs, env_state = env.reset(reset_rng, traj)

        (_, _, _, _), traj_batch = jax.lax.scan(
            _eval_step, (student_ts, env_state, last_obs, rng),
            None, eval_num_steps,
        )

        logged = traj_batch.logged_metrics
        done_count = jnp.maximum(jnp.sum(logged.done), 1)
        summary = SummaryMetrics(
            mean_episode_return=jnp.sum(
                jnp.where(logged.done, logged.returned_episode_returns, 0.0)
            ) / done_count,
            mean_episode_length=jnp.sum(
                jnp.where(logged.done, logged.returned_episode_lengths, 0.0)
            ) / done_count,
            max_timestep=jnp.max(logged.timestep * eval_num_envs),
        )

        if mh is None:
            validation_metrics = ValidationSummary()
        else:
            validation_metrics = mh(traj_batch.env_state)

        return {"eval_summary": summary, "validation_metrics": validation_metrics}

    # ------------------------------------------------------------------
    # Play
    # ------------------------------------------------------------------

    @classmethod
    def play_policy(cls, env,
                    agent_conf: VanillaDaggerAgentConf,
                    agent_state: VanillaDaggerAgentState,
                    n_envs: int, n_steps=None, render=True,
                    record=False, rng=None, deterministic=False,
                    use_mujoco=False, wrap_env=True,
                    use_teacher=False, traj=None):
        """
        Roll out the student (default) or the teacher in a Python loop and render it.

        By default the policy runs in ``n_envs`` parallel MJX environments; with ``use_mujoco`` it runs in the CPU
        MuJoCo environment instead. The environment is stopped at the end.

        Args:
            env: The (unwrapped) environment.
            agent_conf (VanillaDaggerAgentConf): Agent configuration.
            agent_state (VanillaDaggerAgentState): Agent state holding the student and the teacher.
            n_envs (int): Number of parallel environments; must be 1 with ``use_mujoco``.
            n_steps (int): Number of steps to run. If ``None``, the loop runs until it is interrupted.
            render (bool): Unused; the environment is rendered at every step.
            record (bool): Whether to record a video of the MJX rollout. The MuJoCo rollout is always recorded.
            rng (jax.Array): PRNG key for environment resets and action sampling. Defaults to ``jax.random.key(0)``.
            deterministic (bool): If True, the student applies its mean action. Has no effect with ``use_teacher``,
                where actions are always sampled from the teacher's policy.
            use_mujoco (bool): If True, use the CPU MuJoCo backend instead of MJX.
            wrap_env (bool): If True (and not ``use_mujoco``), the environment is wrapped as during training.
                Pass False if ``env`` is already wrapped.
            use_teacher (bool): If True, the teacher is played instead of the student, e.g. as a sanity check.
            traj: Trajectory data passed on to ``env.reset`` and ``env.step`` of the MJX environment.

        """
        import numpy as np

        if use_mujoco:
            assert n_envs == 1, "Only one mujoco env can be run at a time."

        if wrap_env and not use_mujoco:
            env = cls._wrap_env(env, agent_conf.config.experiment)

        if rng is None:
            rng = jax.random.key(0)

        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]

        student_net = agent_conf.student_net
        teacher_net = agent_conf.teacher_net

        s_params = agent_state.student_train_state.params
        s_run_stats = agent_state.student_train_state.run_stats
        t_params = agent_state.teacher_params
        t_run_stats = agent_state.teacher_run_stats

        if deterministic and not use_teacher:
            # Collapse the Gaussian to its mean.
            s_params = dict(s_params)
            s_params["log_std"] = np.ones_like(s_params["log_std"]) * -np.inf

        @jax.jit
        def _act(obs, rng_a):
            if use_teacher:
                (pi, _), _ = teacher_net.apply(
                    {"params": t_params, "run_stats": t_run_stats}, obs,
                    mutable=["run_stats"],
                )
            else:
                (pi, _), _ = student_net.apply(
                    {"params": s_params, "run_stats": s_run_stats}, obs,
                    mutable=["run_stats"],
                )
            return pi.sample(seed=rng_a)

        # reset env
        if use_mujoco:
            obs = env.reset()
            env_state = None
        else:
            obs, env_state = env.reset(env_keys, traj)

        if n_steps is None:
            n_steps = np.iinfo(np.int32).max

        for _ in range(n_steps):
            rng, rng_a = jax.random.split(rng)
            action = _act(obs, rng_a)
            action = jnp.atleast_2d(action)

            if use_mujoco:
                obs, reward, absorbing, done, info = env.step(action)
                env.render(record=True)
                if done:
                    obs = env.reset()
            else:
                obs, reward, absorbing, done, info, env_state = env.step(
                    env_state, action, traj
                )
                env.mjx_render(env_state, record=record)

        env.stop()
