"""
Soft Actor-Critic (SAC) in JAX, built on :class:`~loco_mujoco.algorithms.experimental.offpolicy_base.OffPolicyBase`.

Implements the off-policy maximum-entropy actor-critic algorithm of Haarnoja et al. (2018). The shared
off-policy machinery (replay buffer, twin Q-networks, target updates, collection loop, scan over environment
steps) lives in :mod:`loco_mujoco.algorithms.experimental.offpolicy_base`. This module only owns the
SAC-specific pieces:

- the squashed-Gaussian actor (:class:`SACActorNet`) and the log-probability of its sampled actions,
- the entropy bonus in the Q-target, ``-alpha * log_pi(next_action)``,
- the actor loss ``E[alpha * log_pi - min(Q1, Q2)]``,
- the learnable temperature ``alpha`` with a target-entropy constraint,
- the metrics ``mean_alpha``, ``mean_entropy`` and ``mean_std``.

Example: ``examples/training_examples/jax_rl_mimic_sac``.

"""

import ast
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import flax
import optax
from flax import struct
from omegaconf import DictConfig, OmegaConf, ListConfig, open_dict

from loco_mujoco.algorithms import (AgentConfBase, AgentStateBase, TrainState)
from loco_mujoco.algorithms.common.networks import (RunningMeanStd,
                                                    get_activation_fn)
from loco_mujoco.core.wrappers import SummaryMetrics
from loco_mujoco.algorithms.experimental.offpolicy_base import (
    OffPolicyBase, OffPolicyCriticNet, ReplayBuffer, OffPolicyAgentState,
)


# ---------------------------------------------------------------------------
# Networks (SAC-specific actor)
# ---------------------------------------------------------------------------

class SACActorNet(nn.Module):
    """
    Diagonal Gaussian actor with tanh squashing.

    The network outputs the mean and the log standard deviation of the Gaussian *before* squashing; sampling and
    the tanh are applied by the algorithm.

    Attributes:
        action_dim (int): Dimension of the action space.
        hidden_layer_dims (tuple): Number of units of each hidden layer.
        activation (str): Name of the activation function in ``flax.linen``.
        init_std (float): Initial standard deviation.
        learnable_std (bool): If False, the gradient with respect to the log standard deviation is stopped.
        state_dependent_std (bool): If True, the log standard deviation is a network output, otherwise a
            state-independent parameter ``log_std``.
        log_std_min (float): Lower clipping bound of the log standard deviation.
        log_std_max (float): Upper clipping bound of the log standard deviation.
        use_obs_norm (bool): If True, the observation is normalized by a
            :class:`~loco_mujoco.algorithms.common.networks.RunningMeanStd` layer.

    """

    action_dim: int
    hidden_layer_dims: tuple = (256, 256)
    activation: str = "tanh"
    init_std: float = 1.0
    learnable_std: bool = True
    state_dependent_std: bool = True
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    use_obs_norm: bool = False

    @nn.compact
    def __call__(self, x):
        """
        Compute the parameters of the pre-squashing Gaussian.

        Args:
            x (jax.Array): Observations, shape ``(batch, obs_dim)``.

        Returns:
            Tuple[jax.Array, jax.Array]: Mean and clipped log standard deviation, both of shape
            ``(batch, action_dim)``.

        """
        activation_fn = get_activation_fn(self.activation)
        if self.use_obs_norm:
            x = RunningMeanStd()(x)
        for dim in self.hidden_layer_dims:
            x = nn.Dense(dim, kernel_init=orthogonal(jnp.sqrt(2)),
                         bias_init=constant(0.0))(x)
            x = activation_fn(x)
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                        bias_init=constant(0.0))(x)
        if self.state_dependent_std:
            log_std = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                               bias_init=constant(jnp.log(self.init_std)))(x)
        else:
            log_std = self.param("log_std", constant(jnp.log(self.init_std)),
                                 (self.action_dim,))
            log_std = jnp.broadcast_to(log_std, mean.shape)
        if not self.learnable_std:
            log_std = jax.lax.stop_gradient(log_std)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


# Re-export the critic so callers can keep importing from sac_jax.
SACCriticNet = OffPolicyCriticNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _squashed_gaussian_sample_and_log_prob(mean, log_std, rng):
    """
    Sample from a tanh-squashed Gaussian and compute the log-probability of the sample.

    Args:
        mean (jax.Array): Mean of the Gaussian, shape ``(batch, action_dim)``.
        log_std (jax.Array): Log standard deviation, shape ``(batch, action_dim)``.
        rng (jax.Array): PRNG key.

    Returns:
        Tuple[jax.Array, jax.Array]: The squashed actions in ``(-1, 1)``, shape ``(batch, action_dim)``, and
        their log-probabilities, shape ``(batch,)``.

    """
    std = jnp.exp(log_std)
    noise = jax.random.normal(rng, shape=mean.shape)
    x = mean + std * noise
    action = jnp.tanh(x)
    log_prob = (
        jnp.sum(-0.5 * ((noise ** 2) + 2 * log_std + jnp.log(2 * jnp.pi)), axis=-1)
        - jnp.sum(jnp.log(1 - action ** 2 + 1e-6), axis=-1)
    )
    return action, log_prob


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

@struct.dataclass
class SACSummaryMetrics(SummaryMetrics):
    """
    Training metrics logged by SAC.

    Extends :class:`~loco_mujoco.core.wrappers.mjx.SummaryMetrics`. Every entry is an average over ``log_every``
    training iterations (``buffer_size`` and ``max_timestep`` are taken at the end of that window).

    Attributes:
        mean_critic_loss (float): Mean critic loss (sum of the mean squared errors of all critics).
        mean_actor_loss (float): Mean actor loss.
        mean_alpha_loss (float): Temperature loss; currently always reported as zero.
        mean_alpha (float): Temperature ``alpha``.
        mean_std (float): Mean standard deviation of the pre-squashing Gaussian policy.
        mean_entropy (float): Single-sample estimate of the policy entropy, ``-mean(log_pi)``.
        buffer_size (int): Number of transitions in the replay buffer.

    """
    mean_critic_loss: float = 0.0
    mean_actor_loss: float = 0.0
    mean_alpha_loss: float = 0.0
    mean_alpha: float = 0.0
    mean_std: float = 0.0
    mean_entropy: float = 0.0
    buffer_size: int = 0


# ---------------------------------------------------------------------------
# Conf / state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SACAgentConf(AgentConfBase):
    """
    Static configuration of a SAC agent.

    Attributes:
        config (DictConfig): The full configuration; the hyperparameters live under ``config.experiment``.
        actor_net (SACActorNet): The actor network definition.
        critic_net (OffPolicyCriticNet): The critic network definition.
        actor_tx: The Optax optimizer of the actor.
        critic_tx: The Optax optimizer of the critic.
        alpha_tx: The Optax optimizer of the temperature.

    """
    config: DictConfig
    actor_net: SACActorNet
    critic_net: OffPolicyCriticNet
    actor_tx: Any
    critic_tx: Any
    alpha_tx: Any

    def serialize(self):
        """
        Serialize the agent configuration and the network definitions.

        The optimizers are not serialized; they are rebuilt from the configuration in :meth:`from_dict`.

        Returns:
            dict: Dictionary with the keys ``"config"``, ``"actor_net"`` and ``"critic_net"``.

        """
        conf_dict = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
        return {
            "config": conf_dict,
            "actor_net": flax.serialization.to_state_dict(self.actor_net),
            "critic_net": flax.serialization.to_state_dict(self.critic_net),
        }

    @classmethod
    def from_dict(cls, d):
        """
        Restore the agent configuration from a dictionary.

        Args:
            d (dict): Dictionary created by :meth:`serialize`.

        Returns:
            SACAgentConf: The restored configuration with newly built optimizers.

        """
        config = OmegaConf.create(d["config"])
        exp = config.experiment

        actor_net = flax.serialization.from_state_dict(SACActorNet, d["actor_net"])
        critic_net = flax.serialization.from_state_dict(OffPolicyCriticNet, d["critic_net"])

        max_grad_norm = float(getattr(exp, 'max_grad_norm', 0.5))
        actor_tx = optax.chain(
            optax.clip_by_global_norm(float(getattr(exp, 'max_actor_grad_norm', max_grad_norm))),
            optax.adam(float(exp.lr_actor)),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(float(getattr(exp, 'max_critic_grad_norm', max_grad_norm))),
            optax.adam(float(exp.lr_critic)),
        )
        alpha_tx = optax.adam(float(exp.lr_alpha))
        return cls(config=config, actor_net=actor_net, critic_net=critic_net,
                   actor_tx=actor_tx, critic_tx=critic_tx, alpha_tx=alpha_tx)


@struct.dataclass
class _LogAlphaState:
    """
    Wrapper around the learnable ``log_alpha`` parameter and its optimizer state.

    Attributes:
        train_state (TrainState): Train state whose parameters are ``{"log_alpha": ...}``.
        target_entropy (jax.Array): Target entropy of the policy, scalar.
        learnable (jax.Array): Scalar boolean; if False, the temperature is kept fixed.

    """
    train_state: TrainState
    target_entropy: float
    learnable: bool


@struct.dataclass
class SACAgentState(OffPolicyAgentState):
    """
    Dynamic state of a SAC agent.

    Same fields as :class:`~loco_mujoco.algorithms.experimental.offpolicy_base.OffPolicyAgentState`;
    ``extra_state`` carries the temperature state (``log_alpha`` train state, target entropy and learnable flag).

    """

    @classmethod
    def from_dict(cls, d, agent_conf):
        """
        Restore the agent state from a dictionary.

        Actor, critic, target critic and temperature are restored; the replay buffer is created empty and
        ``env_state`` and ``last_obs`` are ``None``.

        Args:
            d (dict): Dictionary created by ``serialize``.
            agent_conf (SACAgentConf): Agent configuration providing the networks and the optimizers.

        Returns:
            SACAgentState: The restored agent state.

        """
        exp = agent_conf.config.experiment
        obs_dim = int(exp.obs_dim)
        action_dim = int(exp.action_dim)
        actor_net = agent_conf.actor_net
        critic_net = agent_conf.critic_net

        rng = jax.random.PRNGKey(0)

        actor_init = actor_net.init(rng, jnp.zeros((1, obs_dim)))
        actor_state = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_init["params"],
            run_stats=actor_init.get("run_stats", {}),
            tx=agent_conf.actor_tx,
        )
        actor_state = flax.serialization.from_state_dict(actor_state, d["actor_state"])
        actor_state = actor_state.replace(apply_fn=actor_net.apply)

        critic_init = critic_net.init(rng, jnp.zeros((1, obs_dim)),
                                      jnp.zeros((1, action_dim)))
        critic_state = TrainState.create(
            apply_fn=critic_net.apply,
            params=critic_init["params"],
            run_stats=critic_init.get("run_stats", {}),
            tx=agent_conf.critic_tx,
        )
        critic_state = flax.serialization.from_state_dict(critic_state, d["critic_state"])
        critic_state = critic_state.replace(apply_fn=critic_net.apply)

        target_dict = flax.serialization.from_state_dict(
            {"params": critic_init["params"],
             "run_stats": critic_init.get("run_stats", {})},
            d["target_critic"],
        )

        # reconstruct extra_state (log_alpha)
        template_extra = SACJax._init_extra_state(rng, exp, alpha_tx=agent_conf.alpha_tx)
        extra_state = flax.serialization.from_state_dict(template_extra, d["extra_state"])
        extra_state = extra_state.replace(
            train_state=extra_state.train_state.replace(
                apply_fn=lambda p, x: p["log_alpha"])
        )

        replay_buffer = ReplayBuffer.create(
            obs_dim, action_dim, int(exp.buffer_size),
            add_subsample=bool(getattr(exp, 'buffer_add_subsample', False)),
            add_prob=float(getattr(exp, 'buffer_add_prob', 1.0)),
        )

        return cls(
            actor_state=actor_state,
            critic_state=critic_state,
            target_critic_params=target_dict["params"],
            target_critic_run_stats=target_dict["run_stats"],
            extra_state=extra_state,
            replay_buffer=replay_buffer,
        )


# ---------------------------------------------------------------------------
# SAC implementation
# ---------------------------------------------------------------------------

class SACJax(OffPolicyBase):
    """
    Soft Actor-Critic (Haarnoja et al., 2018).

    Off-policy maximum-entropy actor-critic with a tanh-squashed Gaussian policy, an ensemble of Q-networks (two
    by default) with a Polyak-averaged target, and an automatically tuned temperature. The training loop is
    implemented in :class:`~loco_mujoco.algorithms.experimental.offpolicy_base.OffPolicyBase`; the usage follows
    :class:`~loco_mujoco.algorithms.common.base_algorithm.JaxRLAlgorithmBase` (``init_agent_conf``,
    ``init_agent_state``, ``build_train_fn``, ``save_agent``, ``play_policy``). A complete script is in
    ``examples/training_examples/jax_rl_mimic_sac``.

    """
    _agent_conf = SACAgentConf
    _agent_state = SACAgentState

    # ---------- Conf init -----------------------------------------------
    @classmethod
    def init_agent_conf(cls, env, config):
        """
        Create the static agent configuration: derived hyperparameters, the networks and the optimizers.

        The hyperparameters are read from ``config.experiment``:

        - ``num_envs``, ``total_timesteps``: number of parallel environments and total number of environment steps.
        - ``buffer_size``, ``batch_size``, ``learning_starts``, ``gradient_steps``: replay buffer capacity, minibatch
          size, number of stored transitions before learning starts (default ``batch_size``) and gradient steps per
          environment step (default 1).
        - ``gamma``, ``tau``: discount factor and Polyak rate of the target critic (default 0.005).
        - ``lr_actor``, ``lr_critic``, ``lr_alpha``, ``max_grad_norm``: optimizer settings.
        - ``init_alpha``, ``learnable_alpha``, ``target_entropy``: temperature settings; the target entropy defaults
          to ``-action_dim``.
        - ``hidden_layers``, ``activation``, ``init_std``, ``learnable_std``, ``state_dependent_std``,
          ``log_std_min``, ``log_std_max``, ``use_obs_norm``: network settings.
        - ``log_every``: number of training iterations aggregated into one row of the logged metrics.

        The keys ``obs_dim``, ``action_dim`` and ``num_updates`` (``total_timesteps // num_envs``) are written to
        ``config.experiment`` (the configuration is modified in place).

        Args:
            env: The (unwrapped) environment the agent is trained on.
            config (DictConfig): Configuration with the hyperparameters under ``config.experiment``.

        Returns:
            SACAgentConf: The agent configuration.

        """
        obs_dim = int(env.info.observation_space.shape[0])
        action_dim = int(env.info.action_space.shape[0])

        with open_dict(config.experiment):
            config.experiment.obs_dim = obs_dim
            config.experiment.action_dim = action_dim
            config.experiment.num_updates = (
                config.experiment.total_timesteps // config.experiment.num_envs
            )

        exp = config.experiment
        hidden_layers = (exp.hidden_layers
                         if isinstance(exp.hidden_layers, (list, ListConfig))
                         else ast.literal_eval(exp.hidden_layers))

        actor_net = SACActorNet(
            action_dim=action_dim,
            hidden_layer_dims=tuple(hidden_layers),
            activation=str(exp.activation),
            init_std=float(getattr(exp, 'init_std', 1.0)),
            learnable_std=bool(getattr(exp, 'learnable_std', True)),
            state_dependent_std=bool(getattr(exp, 'state_dependent_std', True)),
            log_std_min=float(getattr(exp, 'log_std_min', -20.0)),
            log_std_max=float(getattr(exp, 'log_std_max', 2.0)),
            use_obs_norm=bool(getattr(exp, 'use_obs_norm', False)),
        )
        critic_net = cls._build_critic_net(exp)
        actor_tx, critic_tx = cls._build_optimisers(exp)
        alpha_tx = optax.adam(float(exp.lr_alpha))

        return cls._agent_conf(
            config=config,
            actor_net=actor_net,
            critic_net=critic_net,
            actor_tx=actor_tx,
            critic_tx=critic_tx,
            alpha_tx=alpha_tx,
        )

    @classmethod
    def init_agent_state(cls, env, agent_conf, rng):
        """
        Initialize the SAC agent state (actor, critic, target critic, temperature and an empty replay buffer).

        Args:
            env: The environment (unused; the dimensions are taken from the configuration).
            agent_conf (SACAgentConf): Agent configuration created by :meth:`init_agent_conf`.
            rng (jax.Array): PRNG key used to initialize the network parameters.

        Returns:
            SACAgentState: The initial agent state; ``env_state`` and ``last_obs`` are ``None`` and get initialized
            by the training function.

        """
        exp = agent_conf.config.experiment
        obs_dim = int(exp.obs_dim)
        action_dim = int(exp.action_dim)

        rng, rng_a, rng_c, rng_x = jax.random.split(rng, 4)

        actor_params = agent_conf.actor_net.init(rng_a, jnp.zeros((1, obs_dim)))
        actor_state = TrainState.create(
            apply_fn=agent_conf.actor_net.apply,
            params=actor_params["params"],
            run_stats=actor_params.get("run_stats", {}),
            tx=agent_conf.actor_tx,
        )

        critic_params = agent_conf.critic_net.init(
            rng_c, jnp.zeros((1, obs_dim)), jnp.zeros((1, action_dim))
        )
        critic_state = TrainState.create(
            apply_fn=agent_conf.critic_net.apply,
            params=critic_params["params"],
            run_stats=critic_params.get("run_stats", {}),
            tx=agent_conf.critic_tx,
        )
        target_critic_params = critic_params["params"]
        target_critic_run_stats = critic_params.get("run_stats", {})

        extra_state = cls._init_extra_state(rng_x, exp, alpha_tx=agent_conf.alpha_tx)
        replay_buffer = ReplayBuffer.create(
            obs_dim, action_dim, int(exp.buffer_size),
            add_subsample=bool(getattr(exp, 'buffer_add_subsample', False)),
            add_prob=float(getattr(exp, 'buffer_add_prob', 1.0)),
        )

        return SACAgentState(
            actor_state=actor_state,
            critic_state=critic_state,
            target_critic_params=target_critic_params,
            target_critic_run_stats=target_critic_run_stats,
            extra_state=extra_state,
            replay_buffer=replay_buffer,
        )

    # ---------- OffPolicyBase hooks -------------------------------------
    @classmethod
    def _init_extra_state(cls, rng, exp, alpha_tx=None):
        """
        Create the temperature state.

        Args:
            rng (jax.Array): PRNG key (unused).
            exp (DictConfig): The ``experiment`` node of the configuration.
            alpha_tx: Optimizer of the temperature; if ``None``, Adam with learning rate ``lr_alpha`` is used.

        Returns:
            _LogAlphaState: The temperature state initialized to ``init_alpha`` (default 0.1).

        """
        if alpha_tx is None:
            alpha_tx = optax.adam(float(exp.lr_alpha))
        init_log_alpha = jnp.log(float(getattr(exp, 'init_alpha', 0.1)))
        log_alpha_state = TrainState.create(
            apply_fn=lambda p, x: p["log_alpha"],
            params={"log_alpha": init_log_alpha},
            run_stats={},
            tx=alpha_tx,
        )
        target_entropy = float(getattr(exp, 'target_entropy', -float(exp.action_dim)))
        learnable = bool(getattr(exp, 'learnable_alpha', True))
        return _LogAlphaState(
            train_state=log_alpha_state,
            target_entropy=jnp.array(target_entropy),
            learnable=jnp.array(learnable),
        )

    @classmethod
    def _select_action(cls, actor_state, obs, rng, exp,
                       extra_state, deterministic=False):
        """Sample a squashed-Gaussian action, or return ``tanh(mean)`` if ``deterministic``; see the base class."""
        (mean, log_std), updates = actor_state.apply_fn(
            {"params": actor_state.params, "run_stats": actor_state.run_stats},
            obs, mutable=["run_stats"],
        )
        actor_state = actor_state.replace(run_stats=updates["run_stats"])
        if deterministic:
            action = jnp.tanh(mean)
        else:
            action, _ = _squashed_gaussian_sample_and_log_prob(mean, log_std, rng)
        return action, actor_state

    @classmethod
    def _next_action_and_q_bonus(cls, actor_state, next_obs, rng, exp, extra_state):
        """
        Sample the next action from the current policy and return the entropy bonus ``-alpha * log_pi``; see the base
        class.

        """
        (next_mean, next_log_std), _ = actor_state.apply_fn(
            {"params": actor_state.params, "run_stats": actor_state.run_stats},
            next_obs, mutable=["run_stats"],
        )
        next_action, next_log_pi = _squashed_gaussian_sample_and_log_prob(
            next_mean, next_log_std, rng
        )
        alpha = jnp.exp(extra_state.train_state.params["log_alpha"])
        next_q_bonus = -alpha * next_log_pi
        return next_action, next_q_bonus, {"log_pi": next_log_pi}

    @classmethod
    def _actor_loss(cls, actor_params, actor_apply_fn, actor_run_stats,
                    critic_st, critic_apply_fn,
                    obs_b, rng, exp, extra_state):
        """
        Compute the SAC actor loss ``mean(alpha * log_pi - min_k Q_k(s, a))`` for reparameterized actions; see the
        base class.

        """
        (mean_a, log_std_a), updates = actor_apply_fn(
            {"params": actor_params, "run_stats": actor_run_stats},
            obs_b, mutable=["run_stats"],
        )
        sampled_action, log_pi = _squashed_gaussian_sample_and_log_prob(
            mean_a, log_std_a, rng
        )
        actor_bundle = critic_st.run_stats
        actor_mutables = list(actor_bundle.keys())
        qs_a, _ = critic_apply_fn(
            {"params": critic_st.params, **actor_bundle},
            obs_b, sampled_action, mutable=actor_mutables, training=False,
        )
        # min over the critic ensemble; identical to jnp.minimum(q1, q2) at the
        # default num_critics=2.
        q_a = jnp.min(jnp.stack(qs_a, axis=0), axis=0)
        alpha = jnp.exp(extra_state.train_state.params["log_alpha"])
        loss = jnp.mean(alpha * log_pi - q_a)
        return loss, {
            "log_pi": log_pi,
            "run_stats": updates["run_stats"],
        }

    @classmethod
    def _update_extra(cls, extra_state, actor_aux, rng, exp):
        """
        Take a gradient step on the temperature loss ``-mean(log_alpha * (log_pi + target_entropy))`` if the
        temperature is learnable; see the base class.

        """
        log_pi = actor_aux["log_pi"]
        target_entropy = extra_state.target_entropy
        la_st = extra_state.train_state

        def _alpha_loss_fn(params):
            log_a = params["log_alpha"]
            return -jnp.mean(log_a * (log_pi + target_entropy)), log_a

        (alpha_loss, _), alpha_grads = jax.value_and_grad(
            _alpha_loss_fn, has_aux=True
        )(la_st.params)

        new_la_st = jax.lax.cond(
            extra_state.learnable,
            lambda s: s.apply_gradients(grads=alpha_grads),
            lambda s: s,
            la_st,
        )
        new_extra = extra_state.replace(train_state=new_la_st)
        return new_extra, {"alpha_loss": alpha_loss}

    @classmethod
    def _extra_metrics(cls, actor_state, obs, rng, extra_state, exp):
        """Compute the temperature, the mean policy standard deviation and an entropy estimate for logging."""
        (mean_pi, log_std), _ = actor_state.apply_fn(
            {"params": actor_state.params, "run_stats": actor_state.run_stats},
            obs, mutable=["run_stats"],
        )
        mean_std = jnp.mean(jnp.exp(log_std))
        _, log_pi = _squashed_gaussian_sample_and_log_prob(mean_pi, log_std, rng)
        mean_entropy = -jnp.mean(log_pi)
        alpha_val = jnp.exp(extra_state.train_state.params["log_alpha"])
        return {
            "mean_alpha": alpha_val,
            "mean_std": mean_std,
            "mean_entropy": mean_entropy,
        }

    @classmethod
    def _build_summary_metric(cls, base_kwargs, extra):
        """Assemble a :class:`SACSummaryMetrics` from the base and the extra metrics."""
        return SACSummaryMetrics(
            mean_alpha=extra.get("mean_alpha", 0.0),
            mean_std=extra.get("mean_std", 0.0),
            mean_entropy=extra.get("mean_entropy", 0.0),
            mean_alpha_loss=jnp.array(0.0),
            **base_kwargs,
        )

    # ---------- play_policy --------------------------------------------
    @classmethod
    def play_policy(cls, env, agent_conf, agent_state,
                    n_envs: int, n_steps=None, render=True,
                    record=False, rng=None, deterministic=False,
                    use_mujoco=False, wrap_env=True,
                    train_state_seed=None, traj=None):
        """
        Roll out a trained policy in a Python loop and render it.

        By default the policy runs in ``n_envs`` parallel MJX environments; with ``use_mujoco`` it runs in the CPU
        MuJoCo environment instead. The environment is stopped at the end.

        Args:
            env: The (unwrapped) environment.
            agent_conf (SACAgentConf): Agent configuration.
            agent_state (SACAgentState): Agent state holding the trained actor.
            n_envs (int): Number of parallel environments; use 1 with ``use_mujoco``.
            n_steps (int): Number of steps to run. If ``None``, the loop runs until it is interrupted.
            render (bool): Unused; the environment is rendered at every step.
            record (bool): Whether to record a video of the MJX rollout. The MuJoCo rollout is always recorded.
            rng (jax.Array): PRNG key for environment resets and action noise. Defaults to ``jax.random.key(0)``.
            deterministic (bool): If True, the action ``tanh(mean)`` is applied instead of a sample.
            use_mujoco (bool): If True, use the CPU MuJoCo backend instead of MJX.
            wrap_env (bool): If True (and not ``use_mujoco``), the environment is wrapped as during training.
                Pass False if ``env`` is already wrapped.
            train_state_seed (int): Unused; agents trained with several seeds are not supported here.
            traj: Trajectory data passed on to ``env.reset`` and ``env.step`` of the MJX environment.

        """
        import numpy as np

        exp = agent_conf.config.experiment
        actor_net = agent_conf.actor_net
        actor_state = agent_state.actor_state.replace(apply_fn=actor_net.apply)

        if wrap_env and not use_mujoco:
            env = cls._wrap_env(env, exp)

        if rng is None:
            rng = jax.random.key(0)

        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]

        @jax.jit
        def _act(actor_st, obs, _rng):
            (mean, log_std), updates = actor_net.apply(
                {"params": actor_st.params, "run_stats": actor_st.run_stats},
                obs, mutable=["run_stats"],
            )
            actor_st = actor_st.replace(run_stats=updates["run_stats"])
            if deterministic:
                action = jnp.tanh(mean)
            else:
                action, _ = _squashed_gaussian_sample_and_log_prob(mean, log_std, _rng)
            return action, actor_st

        if use_mujoco:
            obs = env.reset()
            env_state = None
        else:
            obs, env_state = env.reset(env_keys, traj)

        if n_steps is None:
            n_steps = np.iinfo(np.int32).max

        for _ in range(n_steps):
            rng, _rng = jax.random.split(rng)
            action, actor_state = _act(actor_state, obs, _rng)
            action = jnp.atleast_2d(action)

            if use_mujoco:
                obs, _, _, done, _ = env.step(action)
            else:
                obs, _, _, done, _, env_state = env.step(env_state, action, traj)

            if use_mujoco:
                env.render(record=True)
                if done:
                    obs = env.reset()
            else:
                env.mjx_render(env_state, record=record)

        env.stop()
