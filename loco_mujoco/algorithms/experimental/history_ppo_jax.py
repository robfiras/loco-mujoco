"""
History-aware PPO in JAX.

A variant of :class:`~loco_mujoco.algorithms.ppo_jax.PPOJax` whose actor-critic network sees a fixed-length
window of the last ``history_len`` observations (and optionally the preceding actions), encoded either by an
MLP or by Transformer blocks. The history is carried by the algorithm itself instead of an observation
wrapper, so that the rollout buffer only stores the raw observations.

This algorithm is experimental. Example: ``examples/training_examples/jax_rl_mimic_history``.

"""

import ast
from dataclasses import dataclass
from typing import Sequence
from omegaconf import OmegaConf, ListConfig, open_dict

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import flax
import distrax

from loco_mujoco.algorithms import (Transition, TrainState, TrainStateBuffer, MetricHandlerTransition,
                                    RewardNormStats, update_reward_norm)
from loco_mujoco.algorithms.common.networks import RunningMeanStd, get_activation_fn
from loco_mujoco.algorithms.common.policies import PPOPolicy
from loco_mujoco.algorithms.ppo_jax import PPOJax, PPOAgentConf, PPOAgentState, PPOSummaryMetrics
from loco_mujoco.core.wrappers import LogWrapper, LogEnvState, VecEnv
from loco_mujoco.utils import MetricsHandler, ValidationSummary


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Post-norm Transformer encoder block.

    Multi-head self-attention and a two-layer feed-forward network (hidden width ``4 * d_model``, ReLU), each
    followed by a residual connection and layer normalization.

    Attributes:
        d_model (int): Feature dimension of the tokens.
        n_heads (int): Number of attention heads.

    """
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x):
        """
        Apply the block.

        Args:
            x (jax.Array): Token sequence of shape ``(batch, seq_len, d_model)``.

        Returns:
            jax.Array: Encoded sequence of shape ``(batch, seq_len, d_model)``.

        """
        attn_out = nn.MultiHeadDotProductAttention(num_heads=self.n_heads)(x, x)
        x = nn.LayerNorm()(x + attn_out)
        ff_out = nn.Dense(self.d_model)(nn.relu(nn.Dense(4 * self.d_model)(x)))
        return nn.LayerNorm()(x + ff_out)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ActorCriticHistory(nn.Module):
    """
    Actor-critic network for History-PPO.

    Expects a flat input of shape ``(batch, history_len * entry_dim)``, where the entries are ordered from the
    oldest to the newest step and ``entry_dim = obs_dim + action_dim`` if ``include_actions`` is set (each entry
    is an observation followed by the action that preceded it) and ``entry_dim = obs_dim`` otherwise.

    The observations of all steps are normalized with the *same*
    :class:`~loco_mujoco.algorithms.common.networks.RunningMeanStd` statistics; actions are not normalized.
    Two encoders are available:

    - ``"mlp"``: per-step dense projection to ``n_features``, flattening over the window, then an MLP with
      ``hidden_layer_dims``.
    - ``"transformer"``: per-step dense projection to ``d_model``, ``n_layers`` :class:`TransformerBlock`,
      then mean pooling over the window.

    Actor and critic share the encoder: the action mean and the value are linear heads on the encoded features.
    The policy is a diagonal Gaussian with a state-independent standard deviation (parameter ``log_std``).

    Attributes:
        action_dim (int): Dimension of the action space.
        obs_dim (int): Dimension of a single (raw) observation.
        history_len (int): Number of steps in the history window, including the current one.
        include_actions (bool): Whether every history entry also contains an action.
        encoder_type (str): ``"transformer"`` selects the Transformer encoder; any other value the MLP encoder.
        hidden_layer_dims (Sequence[int]): Hidden layers of the MLP encoder.
        n_features (int): Width of the per-step projection of the MLP encoder.
        d_model (int): Token dimension of the Transformer encoder.
        n_heads (int): Number of attention heads of the Transformer encoder.
        n_layers (int): Number of Transformer blocks.
        activation (str): Name of the activation function in ``flax.linen``.
        init_std (float): Initial standard deviation of the policy.
        learnable_std (bool): If False, the gradient with respect to ``log_std`` is stopped.

    """

    action_dim: int
    obs_dim: int
    history_len: int
    include_actions: bool = False
    encoder_type: str = "mlp"
    hidden_layer_dims: Sequence[int] = (512, 256)
    n_features: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    activation: str = "tanh"
    init_std: float = 1.0
    learnable_std: bool = True

    def setup(self):
        """Resolve the activation function from its name."""
        self.activation_fn = get_activation_fn(self.activation)

    @nn.compact
    def __call__(self, x):
        """
        Compute the action distribution and the value estimate from a history window.

        Args:
            x (jax.Array): Flattened history windows of shape ``(batch, history_len * entry_dim)`` or
                ``(history_len * entry_dim,)``.

        Returns:
            Tuple[distrax.MultivariateNormalDiag, jax.Array]: The action distribution and the value estimates of
            shape ``(batch,)``.

        """
        x = jnp.atleast_2d(x)                                   # (B, N*entry_dim)
        B = x.shape[0]
        entry_dim = self.obs_dim + (self.action_dim if self.include_actions else 0)

        x_hist = x.reshape(B, self.history_len, entry_dim)       # (B, N, entry_dim)
        obs_part = x_hist[:, :, :self.obs_dim]

        # Normalize: identical running stats for every step in the window
        obs_norm = RunningMeanStd()(
            obs_part.reshape(-1, self.obs_dim)
        ).reshape(B, self.history_len, self.obs_dim)

        if self.include_actions:
            x_norm = jnp.concatenate([obs_norm, x_hist[:, :, self.obs_dim:]], axis=-1)
        else:
            x_norm = obs_norm

        if self.encoder_type == "transformer":
            h = self.activation_fn(nn.Dense(self.d_model)(x_norm))
            for _ in range(self.n_layers):
                h = TransformerBlock(self.d_model, self.n_heads)(h)
            features = jnp.mean(h, axis=1)                       # (B, d_model)
        else:
            h = self.activation_fn(nn.Dense(self.n_features)(x_norm))
            features = h.reshape(B, -1)                          # (B, N*n_features)
            for dim in self.hidden_layer_dims:
                features = self.activation_fn(
                    nn.Dense(dim, kernel_init=orthogonal(jnp.sqrt(2)),
                             bias_init=constant(0.0))(features)
                )

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(features)
        logstd = self.param("log_std", nn.initializers.constant(jnp.log(self.init_std)),
                            (self.action_dim,))
        if not self.learnable_std:
            logstd = jax.lax.stop_gradient(logstd)
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(logstd))

        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(features)
        return pi, jnp.squeeze(value, axis=-1)


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HistoryAgentConf(PPOAgentConf):
    """
    Static configuration of a History-PPO agent.

    Same fields as :class:`~loco_mujoco.algorithms.ppo_jax.PPOAgentConf`, with an
    :class:`ActorCriticHistory` as network.

    """

    @classmethod
    def from_dict(cls, d):
        """
        Restore the agent configuration from a dictionary.

        The network definition is rebuilt from the ``experiment`` node of the stored configuration.

        Args:
            d (dict): Dictionary created by ``serialize``.

        Returns:
            HistoryAgentConf: The restored configuration with a newly built optimizer.

        """
        config = OmegaConf.create(d["config"])
        tx = PPOJax._get_optimizer(config)
        exp = config.experiment

        hidden_layers = (exp.hidden_layers
                         if isinstance(exp.hidden_layers, (list, ListConfig))
                         else ast.literal_eval(exp.hidden_layers))

        network_struct = ActorCriticHistory(
            action_dim=int(exp.action_dim),
            obs_dim=int(exp.obs_dim),
            history_len=int(exp.history_len),
            include_actions=bool(getattr(exp, 'include_actions_in_history', False)),
            encoder_type=str(getattr(exp, 'encoder_type', 'mlp')),
            hidden_layer_dims=hidden_layers,
            n_features=int(exp.n_features),
            d_model=int(getattr(exp, 'd_model', exp.n_features)),
            n_heads=int(getattr(exp, 'n_heads', 4)),
            n_layers=int(getattr(exp, 'n_layers', 2)),
            activation=str(exp.activation),
            init_std=float(exp.init_std),
            learnable_std=bool(exp.learnable_std),
        )
        network = flax.serialization.from_state_dict(network_struct, d["network"])
        return cls(config=config, network=network, tx=tx)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class HistoryPPOJax(PPOJax):
    """
    PPO with a fixed N-step observation history, encoded by an MLP or a Transformer.

    **Memory design.** Storing ``N * obs_dim`` values per transition (as an
    :class:`~loco_mujoco.core.wrappers.mjx.NStepWrapper` does) inflates the rollout buffer N-fold. Instead, only
    the raw observation is stored in each transition, and history windows are never materialized for the whole
    batch:

    1. After the rollout, the history carried over from before the rollout is prepended to the collected
       observations (and actions, if ``include_actions_in_history`` is set), giving buffers of shape
       ``(N - 1 + T, num_envs, obs_dim)`` and ``(N + T, num_envs, action_dim)``.
    2. In each minibatch, the N-step window of every sample is gathered on the fly from these buffers with
       vectorized indexing, giving an array of shape ``(minibatch_size, N * entry_dim)``.

    This keeps the observation-related peak memory proportional to ``obs_dim`` instead of ``N * obs_dim``.

    The history starts with zeros. During training it is carried across episode boundaries and across updates
    within one call of the training function.

    The interface is that of :class:`~loco_mujoco.algorithms.ppo_jax.PPOJax`, and agent states are
    :class:`~loco_mujoco.algorithms.ppo_jax.PPOAgentState`; see
    ``examples/training_examples/jax_rl_mimic_history`` for a training script.

    """

    _agent_conf = HistoryAgentConf

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @classmethod
    def init_agent_conf(cls, env, config):
        """
        Create the static agent configuration: derived hyperparameters, the network and the optimizer.

        In addition to the PPO hyperparameters (see :meth:`PPOJax.init_agent_conf
        <loco_mujoco.algorithms.ppo_jax.PPOJax.init_agent_conf>`), the following keys are read from
        ``config.experiment``:

        - ``history_len``: number of steps in the history window, including the current one.
        - ``include_actions_in_history`` (optional, default False): whether the previous actions are part of the
          window.
        - ``encoder_type`` (optional, default ``"mlp"``): ``"mlp"`` or ``"transformer"``.
        - ``n_features``: width of the per-step projection of the MLP encoder.
        - ``d_model`` (optional, default ``n_features``), ``n_heads`` (optional, default 4), ``n_layers``
          (optional, default 2): Transformer settings.

        The keys ``num_updates``, ``minibatch_size``, ``validation_interval``, ``obs_dim``, ``action_dim`` and a
        recomputed ``validation.num`` are written to ``config.experiment`` in place.

        Args:
            env: The (unwrapped) environment the agent is trained on.
            config (DictConfig): Configuration with the hyperparameters under ``config.experiment``.

        Returns:
            HistoryAgentConf: The agent configuration.

        """
        with open_dict(config.experiment):
            config.experiment.num_updates = (
                config.experiment.total_timesteps // config.experiment.num_steps
                // config.experiment.num_envs)
            config.experiment.minibatch_size = (
                config.experiment.num_envs * config.experiment.num_steps
                // config.experiment.num_minibatches)
            config.experiment.validation_interval = (
                config.experiment.num_updates // config.experiment.validation.num)
            config.experiment.validation.num = int(
                config.experiment.num_updates // config.experiment.validation_interval)
            config.experiment.obs_dim = int(env.info.observation_space.shape[0])
            config.experiment.action_dim = int(env.info.action_space.shape[0])

        exp = config.experiment
        hidden_layers = (exp.hidden_layers
                         if isinstance(exp.hidden_layers, (list, ListConfig))
                         else ast.literal_eval(exp.hidden_layers))

        network = ActorCriticHistory(
            action_dim=exp.action_dim,
            obs_dim=exp.obs_dim,
            history_len=int(exp.history_len),
            include_actions=bool(getattr(exp, 'include_actions_in_history', False)),
            encoder_type=str(getattr(exp, 'encoder_type', 'mlp')),
            hidden_layer_dims=hidden_layers,
            n_features=int(exp.n_features),
            d_model=int(getattr(exp, 'd_model', exp.n_features)),
            n_heads=int(getattr(exp, 'n_heads', 4)),
            n_layers=int(getattr(exp, 'n_layers', 2)),
            activation=str(exp.activation),
            init_std=float(exp.init_std),
            learnable_std=bool(exp.learnable_std),
        )
        return cls._agent_conf(config=config, network=network, tx=cls._get_optimizer(config))

    @classmethod
    def init_agent_state(cls, env, agent_conf, rng):
        """
        Initialize the agent state (network parameters, running statistics and optimizer state).

        Args:
            env: The (unwrapped) environment; not used.
            agent_conf (HistoryAgentConf): Agent configuration created by :meth:`init_agent_conf`.
            rng (jax.Array): PRNG key used to initialize the network parameters.

        Returns:
            PPOAgentState: The initial agent state.

        """
        config = agent_conf.config.experiment
        N = int(config.history_len)
        obs_dim = int(config.obs_dim)
        action_dim = int(config.action_dim)
        include_actions = bool(getattr(config, 'include_actions_in_history', False))
        entry_dim = obs_dim + (action_dim if include_actions else 0)

        rng, _rng = jax.random.split(rng)
        params = agent_conf.network.init(_rng, jnp.zeros((1, N * entry_dim)))
        train_state = TrainState.create(
            apply_fn=agent_conf.network.apply,
            params=params["params"],
            run_stats=params["run_stats"],
            tx=agent_conf.tx,
        )
        return PPOAgentState(train_state=train_state)

    # No NStepWrapper: history carry is threaded through the runner state
    @staticmethod
    def _wrap_env(env, config):
        """
        Wrap the environment for training.

        Applies a :class:`~loco_mujoco.core.wrappers.mjx.LogWrapper` and a
        :class:`~loco_mujoco.core.wrappers.mjx.VecEnv`. No ``NStepWrapper`` is used; the history is threaded through
        the runner state of the training loop.

        Args:
            env: The (unwrapped) environment.
            config (DictConfig): The ``experiment`` node of the configuration; not used.

        Returns:
            The wrapped, vectorized environment.

        """
        env = LogWrapper(env)
        env = VecEnv(env)
        # reward normalization moved into _train_fn; stats live on the agent state
        return env

    # ------------------------------------------------------------------
    # Play (override to maintain history carry)
    # ------------------------------------------------------------------

    @classmethod
    def play_policy(cls, env, agent_conf, agent_state,
                    n_envs: int, n_steps=None, render=True,
                    record=False, rng=None, deterministic=False,
                    use_mujoco=False, wrap_env=True,
                    train_state_seed=None, traj=None):
        """
        Roll out a trained history policy in a Python loop and render it.

        The history window starts with zeros and is filled with the observations (and actions) of the rollout. By
        default the policy runs in ``n_envs`` parallel MJX environments; with ``use_mujoco`` the CPU MuJoCo
        environment is stepped instead, and the history is cleared when an episode ends. The environment is stopped
        at the end.

        Args:
            env: The (unwrapped) environment.
            agent_conf (HistoryAgentConf): Agent configuration.
            agent_state (PPOAgentState): Agent state holding the trained policy.
            n_envs (int): Number of parallel environments.
            n_steps (int): Number of steps to run. If ``None``, the loop runs until it is interrupted.
            render (bool): Not used; the environment is rendered at every step.
            record (bool): Whether to record a video of the MJX rollout. The MuJoCo rollout is always recorded.
            rng (jax.Array): PRNG key for environment resets and action sampling. Defaults to ``jax.random.key(0)``.
            deterministic (bool): If True, the standard deviation of the policy is set to zero so that the mean
                action is applied. This overwrites ``log_std`` in the parameters of ``agent_state`` in place.
            use_mujoco (bool): If True, use the CPU MuJoCo backend instead of MJX.
            wrap_env (bool): If True (and not ``use_mujoco``), the environment is wrapped as during training.
                Pass False if ``env`` is already wrapped.
            train_state_seed (int): Index of the seed to play if the agent was trained with ``n_seeds > 1``.
            traj: Trajectory data passed on to ``env.reset`` and ``env.step`` of the MJX environment.

        """

        import warnings, numpy as np
        from loco_mujoco.algorithms.common.policies import PPOPolicy

        config = agent_conf.config.experiment
        N          = int(config.history_len)
        obs_dim    = int(config.obs_dim)
        action_dim = int(config.action_dim)
        include_actions = bool(getattr(config, 'include_actions_in_history', False))
        act_carry_dim   = action_dim if include_actions else 0
        entry_dim       = obs_dim + act_carry_dim

        train_state = agent_state.train_state
        if deterministic:
            train_state.params["log_std"] = np.ones_like(train_state.params["log_std"]) * -np.inf
        if config.n_seeds > 1:
            assert train_state_seed is not None
            train_state = jax.tree.map(lambda x: x[train_state_seed], train_state)

        if wrap_env and not use_mujoco:
            env = cls._wrap_env(env, config)

        if rng is None:
            rng = jax.random.key(0)

        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]

        _policy = PPOPolicy(agent_conf.network)

        @jax.jit
        def _step(ts, hist_obs, _rng):
            a, _, _, ts = _policy.get_action_and_value(hist_obs, ts, _rng)
            return a, ts

        def _assemble(raw_obs, obs_c, act_c):
            obs_win = jnp.concatenate([obs_c, raw_obs[None]], axis=0).transpose(1, 0, 2)
            if include_actions:
                win = jnp.concatenate([obs_win, act_c.transpose(1, 0, 2)], axis=-1)
            else:
                win = obs_win
            return win.reshape(n_envs, N * entry_dim)

        obs_carry = jnp.zeros((N - 1, n_envs, obs_dim))
        act_carry = jnp.zeros((N,     n_envs, act_carry_dim))

        if use_mujoco:
            raw_obs = env.reset()
            env_state = None
        else:
            raw_obs, env_state = env.reset(env_keys, traj)

        if n_steps is None:
            n_steps = np.iinfo(np.int32).max

        for _ in range(n_steps):
            rng, _rng = jax.random.split(rng)
            hist_obs = _assemble(raw_obs, obs_carry, act_carry)
            action, train_state = _step(train_state, hist_obs, _rng)
            action = jnp.atleast_2d(action)

            if use_mujoco:
                raw_obs, _, _, done, _ = env.step(action)
            else:
                raw_obs, _, _, done, _, env_state = env.step(env_state, action, traj)

            obs_carry = jnp.concatenate([obs_carry[1:], raw_obs[None]], axis=0)
            if include_actions:
                act_carry = jnp.concatenate([act_carry[1:], action[None]], axis=0)

            if use_mujoco:
                env.render(record=True)
                if done:
                    raw_obs = env.reset()
                    obs_carry = jnp.zeros((N - 1, n_envs, obs_dim))
                    act_carry = jnp.zeros((N,     n_envs, act_carry_dim))
            else:
                env.mjx_render(env_state, record=record)

        env.stop()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def _train_fn(cls, rng, env,
                  agent_conf: PPOAgentConf,
                  agent_state: PPOAgentState = None,
                  traj=None,
                  mh: MetricsHandler = None):
        """
        Run History-PPO for ``num_updates`` updates.

        Works like :meth:`PPOJax._train_fn <loco_mujoco.algorithms.ppo_jax.PPOJax._train_fn>` (rollout, optional
        reward normalization, GAE, clipped PPO update with optional adaptive learning rate, periodic validation),
        except that the network input is the history window assembled from the carried observations and actions, and
        that the windows of a minibatch are gathered on the fly during the update. The history starts with zeros at
        every call, also when the environment state is resumed from ``agent_state``.

        Args:
            rng (jax.Array): PRNG key.
            env: The (unwrapped) environment.
            agent_conf (HistoryAgentConf): Static agent configuration.
            agent_state (PPOAgentState): Agent state to start or resume from.
            traj: Trajectory data passed on to ``env.reset`` and ``env.step``; ``None`` if the task has no trajectory.
            mh (MetricsHandler): Optional metrics handler computing the validation metrics.

        Returns:
            dict: Dictionary with the keys ``"agent_state"``
            (:class:`~loco_mujoco.algorithms.ppo_jax.PPOAgentState` after training), ``"training_metrics"``
            (:class:`~loco_mujoco.algorithms.ppo_jax.PPOSummaryMetrics` with a leading axis of length
            ``num_updates``) and ``"validation_metrics"`` (:class:`~loco_mujoco.utils.metrics.ValidationSummary`
            with a leading axis of length ``num_updates``; entries of updates without validation are zero).

        """

        config, network, tx = (
            agent_conf.config.experiment, agent_conf.network, agent_conf.tx
        )

        N = int(config.history_len)
        obs_dim = int(config.obs_dim)
        action_dim = int(config.action_dim)
        include_actions = bool(getattr(config, 'include_actions_in_history', False))
        entry_dim = obs_dim + (action_dim if include_actions else 0)
        act_carry_dim = action_dim if include_actions else 0

        env = cls._wrap_env(env, config)
        policy = PPOPolicy(network)

        # ---- train state ----
        if agent_state is not None:
            train_state = agent_state.train_state.replace(apply_fn=network.apply)
        else:
            rng, _rng = jax.random.split(rng)
            params = network.init(_rng, jnp.zeros((1, N * entry_dim)))
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=params["params"],
                run_stats=params["run_stats"],
                tx=tx,
            )

        # ---- env ----
        if agent_state is not None and agent_state.env_state is not None:
            env_state = agent_state.env_state
            raw_obsv = agent_state.last_obs
        else:
            rng, _rng = jax.random.split(rng)
            raw_obsv, env_state = env.reset(jax.random.split(_rng, config.num_envs), traj)

        # ---- reward normalization stats (survive across chunks) ----
        if config.normalize_env:
            if agent_state is not None and agent_state.reward_norm_stats is not None:
                rew_stats = agent_state.reward_norm_stats
            else:
                rew_stats = RewardNormStats.create(config.num_envs)
        else:
            rew_stats = None

        ts_buf = TrainStateBuffer.create(train_state, config.validation.num)

        # ---- initial history carries (zeros = no prior context) ----
        # obs_carry:  last (N-1) raw observations → (N-1, num_envs, obs_dim)
        # act_carry:  last N prev-actions paired with each window entry → (N, num_envs, act_carry_dim)
        obs_carry_0 = jnp.zeros((N - 1, config.num_envs, obs_dim))
        act_carry_0 = jnp.zeros((N,     config.num_envs, act_carry_dim))

        # ---- single-step history assembly (used during rollout and eval) ----
        def _assemble_hist(raw_obs, obs_c, act_c):
            """
            raw_obs: (n_envs, obs_dim)
            obs_c:   (N-1, n_envs, obs_dim)
            act_c:   (N,   n_envs, act_carry_dim)
            Returns: (n_envs, N * entry_dim)
            """
            n_envs = raw_obs.shape[0]
            obs_win = jnp.concatenate([obs_c, raw_obs[None]], axis=0).transpose(1, 0, 2)
            if include_actions:
                win = jnp.concatenate([obs_win, act_c.transpose(1, 0, 2)], axis=-1)
            else:
                win = obs_win
            return win.reshape(n_envs, N * entry_dim)

        # ---- per-minibatch history gathering (used at update time) ----
        # full_obs_c / full_acts_c are built once per rollout chunk and closed over.
        # They are small: (N-1+T, envs, obs_dim) vs the avoided (T, envs, N*obs_dim).
        def _make_gather(full_obs_c, full_acts_c):
            def _gather(flat_idx):
                """
                flat_idx: (mb_size,) — indices into (T * num_envs)
                Returns:  (mb_size, N * entry_dim)
                """
                t_idx = flat_idx // config.num_envs                     # (mb,)
                e_idx = flat_idx % config.num_envs                      # (mb,)
                win_t = t_idx[:, None] + jnp.arange(N)[None, :]        # (mb, N)
                obs_win = full_obs_c[win_t, e_idx[:, None], :]          # (mb, N, obs_dim)
                if include_actions:
                    act_win = full_acts_c[win_t, e_idx[:, None], :]
                    hist = jnp.concatenate([obs_win, act_win], axis=-1)  # (mb, N, entry_dim)
                else:
                    hist = obs_win
                return hist.reshape(flat_idx.shape[0], N * entry_dim)
            return _gather

        # ---- main training loop ----
        def _update_step(runner_state, unused):
            (train_state, env_state, last_raw_obs, obs_carry, act_carry,
             rew_stats, ts_buf, rng) = runner_state

            obs_carry_start = obs_carry   # snapshot before rollout for history reconstruction
            act_carry_start = act_carry

            # -- collect rollout; store raw obs_dim per step (not N*obs_dim) --
            def _env_step(carry, unused):
                ts, es, raw_obs, obs_c, act_c, rns, ts_b, rng = carry

                hist_obs = _assemble_hist(raw_obs, obs_c, act_c)

                rng, _rng = jax.random.split(rng)
                action, log_prob, value, ts = policy.get_action_and_value(hist_obs, ts, _rng)

                new_raw_obs, reward, absorbing, done, info, es = env.step(es, action, traj)

                # NORMALIZE REWARD (stats threaded through runner_state)
                if rns is not None:
                    reward, rns = update_reward_norm(rns, reward, done, config.gamma)

                new_obs_c = jnp.concatenate([obs_c[1:], raw_obs[None]], axis=0)
                new_act_c = jnp.concatenate(
                    [act_c[1:],
                     action[None] if include_actions else jnp.zeros((1, act_c.shape[1], 0))],
                    axis=0
                )

                log_es = es.find(LogEnvState)
                transition = Transition(
                    done, absorbing, action, value, reward, log_prob,
                    raw_obs,   # ← raw obs (obs_dim), not history-augmented
                    info, es.additional_carry.traj_state, log_es.metrics
                )
                return (ts, es, new_raw_obs, new_obs_c, new_act_c, rns, ts_b, rng), transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )
            (train_state, env_state, last_raw_obs, obs_carry, act_carry,
             rew_stats, ts_buf, rng) = runner_state

            # -- value estimate for last step --
            last_hist_obs = _assemble_hist(last_raw_obs, obs_carry, act_carry)
            _, last_val, _ = policy.get_dist_and_value(last_hist_obs, train_state)

            # -- build full obs/act buffers with carry prepended (small, closed over below) --
            # Shape: (N-1+T, envs, obs_dim) and (N+T, envs, act_dim)
            # These replace the avoided (T, envs, N*obs_dim) materialisation.
            full_obs_c = jnp.concatenate([obs_carry_start, traj_batch.obs], axis=0)
            full_acts_c = (
                jnp.concatenate([act_carry_start, traj_batch.action], axis=0)
                if include_actions else None
            )
            _gather_hist = _make_gather(full_obs_c, full_acts_c)

            # Replace traj obs with a zero-size placeholder so it occupies no memory in the
            # epoch-scan carry (history is fetched per-minibatch from full_obs_c instead).
            traj_batch = traj_batch._replace(
                obs=jnp.zeros((config.num_steps, config.num_envs, 0))
            )

            # -- GAE --
            def _get_advantages(carry, transition):
                gae, next_val = carry
                delta = (transition.reward
                         + config.gamma * next_val * (1 - transition.absorbing)
                         - transition.value)
                gae = delta + config.gamma * config.gae_lambda * (1 - transition.done) * gae
                return (gae, transition.value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch, reverse=True, unroll=16,
            )
            targets = advantages + traj_batch.value

            # -- update epochs --
            def _update_epoch(update_state, unused):
                def _update_minibatch(ts, mb_info):
                    mb_traj, mb_adv, mb_tgt, mb_idx = mb_info
                    # Gather history windows for this minibatch on-the-fly.
                    # Peak memory here: (mb_size, N*entry_dim) — independent of T.
                    mb_hist = _gather_hist(mb_idx)
                    mb_traj = mb_traj._replace(obs=mb_hist)

                    def _loss_fn(params, mb_traj, gae, targets):
                        pi, value, _ = policy.get_dist_and_value(
                            mb_traj.obs, ts.replace(params=params)
                        )
                        log_prob = pi.log_prob(mb_traj.action)
                        vclip = mb_traj.value + (value - mb_traj.value).clip(
                            -config.clip_eps, config.clip_eps
                        )
                        value_loss = 0.5 * jnp.maximum(
                            jnp.square(value - targets),
                            jnp.square(vclip - targets),
                        ).mean()
                        ratio = jnp.exp(log_prob - mb_traj.log_prob)
                        gae_n = (gae - gae.mean()) / (gae.std() + 1e-8)
                        actor_loss = -jnp.minimum(
                            ratio * gae_n,
                            jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * gae_n,
                        ).mean()
                        entropy = pi.entropy().mean()
                        total = actor_loss + config.vf_coef * value_loss - config.ent_coef * entropy
                        kl = (mb_traj.log_prob - log_prob).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1.0) > config.clip_eps)
                        return total, (value_loss, actor_loss, entropy, kl, clip_frac)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(ts.params, mb_traj, mb_adv, mb_tgt)
                    return ts.apply_gradients(grads=grads), total_loss

                ts, traj, adv, tgt, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config.minibatch_size * config.num_minibatches
                assert batch_size == config.num_steps * config.num_envs

                perm = jax.random.permutation(_rng, batch_size)
                perm_mb = perm.reshape(config.num_minibatches, -1)    # (num_mb, mb_size)

                # Shuffle all traj fields (obs is zero-size, so no cost there)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), (traj, adv, tgt)
                )
                shuffled = jax.tree.map(lambda x: jnp.take(x, perm, axis=0), batch)
                mb_size = batch_size // config.num_minibatches
                minibatches = jax.tree.map(
                    lambda x: x.reshape([config.num_minibatches, mb_size] + list(x.shape[1:])),
                    shuffled
                )
                mb_traj, mb_adv, mb_tgt = minibatches

                # Add flat permutation indices so each minibatch can gather its own history
                ts, loss = jax.lax.scan(
                    _update_minibatch, ts, (mb_traj, mb_adv, mb_tgt, perm_mb)
                )

                # Adaptive KL learning rate
                desired_kl = getattr(config, 'desired_kl', None)
                if desired_kl is not None:
                    mean_kl = jnp.mean(loss[1][3])
                    lr = ts.opt_state.inner_state.hyperparams['learning_rate']
                    new_lr = jax.lax.cond(
                        mean_kl > desired_kl * 2.0,
                        lambda l: jnp.maximum(1e-5, l / 1.5),
                        lambda l: jax.lax.cond(
                            (mean_kl < desired_kl / 2.0) & (mean_kl > 0.0),
                            lambda l: jnp.minimum(1e-2, l * 1.5),
                            lambda l: l, l,
                        ),
                        lr,
                    )
                    ts = ts.replace(opt_state=ts.opt_state._replace(
                        inner_state=ts.opt_state.inner_state._replace(
                            hyperparams={'learning_rate': new_lr}
                        )
                    ))

                return (ts, traj, adv, tgt, rng), loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state, _, _, _, rng = update_state

            counter = ((train_state.step + 1) // config.num_minibatches) // config.update_epochs
            logged_metrics = traj_batch.metrics

            mean_value_loss    = jnp.mean(loss_info[1][0])
            mean_actor_loss    = jnp.mean(loss_info[1][1])
            mean_entropy       = jnp.mean(loss_info[1][2])
            mean_approx_kl     = jnp.mean(loss_info[1][3])
            mean_clip_fraction = jnp.mean(loss_info[1][4])

            desired_kl = getattr(config, 'desired_kl', None)
            if config.anneal_lr:
                current_lr = cls._linear_lr_schedule(
                    train_state.step, config.num_minibatches,
                    config.update_epochs, config.lr, config.num_updates
                )
            elif desired_kl is not None:
                current_lr = train_state.opt_state.inner_state.hyperparams['learning_rate']
            else:
                current_lr = jnp.array(config.lr)

            metric = PPOSummaryMetrics(
                mean_episode_return=jnp.sum(
                    jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)
                ) / jnp.sum(logged_metrics.done),
                mean_episode_length=jnp.sum(
                    jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)
                ) / jnp.sum(logged_metrics.done),
                max_timestep=jnp.max(logged_metrics.timestep * config.num_envs),
                mean_value_loss=mean_value_loss,
                mean_actor_loss=mean_actor_loss,
                mean_entropy=mean_entropy,
                mean_approx_kl=mean_approx_kl,
                mean_clip_fraction=mean_clip_fraction,
                learning_rate=current_lr,
            )

            # -- optional evaluation --
            def _evaluation_step():
                n_eval = config.validation.num_envs
                eval_obsv, eval_es = env.reset(jax.random.split(rng, n_eval), traj)
                eval_obs_c = jnp.zeros((N - 1, n_eval, obs_dim))
                eval_act_c = jnp.zeros((N, n_eval, act_carry_dim))

                def _eval_step(carry, unused):
                    ts, es, raw_obs, obs_c, act_c, rng_e = carry
                    rng_e, _rng_e = jax.random.split(rng_e)
                    hist_obs = _assemble_hist(raw_obs, obs_c, act_c)
                    y, updates = ts.apply_fn(
                        {'params': ts.params, 'run_stats': ts.run_stats},
                        hist_obs, mutable=["run_stats"]
                    )
                    pi, _ = y
                    ts = ts.replace(run_stats=updates['run_stats'])
                    action = pi.sample(seed=_rng_e)
                    new_raw_obs, _, _, _, _, es = env.step(es, action, traj)
                    new_obs_c = jnp.concatenate([obs_c[1:], raw_obs[None]], axis=0)
                    new_act_c = jnp.concatenate(
                        [act_c[1:],
                         action[None] if include_actions else jnp.zeros((1, n_eval, 0))],
                        axis=0
                    )
                    log_es = es.find(LogEnvState)
                    return (ts, es, new_raw_obs, new_obs_c, new_act_c, rng_e), \
                           MetricHandlerTransition(es, log_es.metrics)

                _, eval_batch = jax.lax.scan(
                    _eval_step,
                    (train_state, eval_es, eval_obsv, eval_obs_c, eval_act_c, rng),
                    None, config.validation.num_steps
                )
                return mh(eval_batch.env_state)

            if mh is None:
                validation_metrics = ValidationSummary()
            else:
                validation_metrics = jax.lax.cond(
                    counter % config.validation_interval == 0,
                    _evaluation_step, mh.get_zero_container
                )

            if config.debug:
                def callback(metrics):
                    rv = metrics.returned_episode_returns[metrics.done]
                    ts_ = metrics.timestep[metrics.done] * config.num_envs
                    for t in range(len(ts_)):
                        print(f"global step={ts_[t]}, episodic return={rv[t]}")
                jax.debug.callback(callback, env_state.metrics)

            ts_buf = jax.lax.cond(
                counter % config.validation_interval == 0,
                lambda x, y: TrainStateBuffer.add(x, y),
                lambda x, y: x, ts_buf, train_state
            )

            runner_state = (train_state, env_state, last_raw_obs,
                            obs_carry, act_carry, rew_stats, ts_buf, rng)
            return runner_state, (metric, validation_metrics)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, raw_obsv,
                        obs_carry_0, act_carry_0, rew_stats, ts_buf, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config.num_updates
        )

        agent_state = PPOAgentState(
            train_state=runner_state[0],
            env_state=runner_state[1],
            last_obs=runner_state[2],
            reward_norm_stats=runner_state[5],
        )
        return {
            "agent_state": agent_state,
            "training_metrics": metrics[0],
            "validation_metrics": metrics[1],
        }
