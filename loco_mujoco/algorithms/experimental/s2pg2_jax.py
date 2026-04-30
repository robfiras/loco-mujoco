"""S2PG2: Stochastic Stateful Policy Gradient with a separate z-policy.

Variant of S2PG-PPO with two key changes:

1. **Action and z' are independent given h.** The encoder maps
   (obs, z_in, prev_a) → h via a feed-forward MLP (no GRU). Four
   parallel heads sit on h:
     - π_a(a | h)         action distribution (Gaussian)
     - π_z(z' | h)        next-z distribution (Gaussian)
     - V(h)               return value (env reward)
     - V_z(h)             cost-return value (z-policy reward)
   The action does NOT see z', so the action gradient cannot reshape
   the z-selection. Recurrence is preserved purely through the carry:
   z' sampled at step t becomes z_in at step t+1.

2. **z' is trained via PPO on a separate reward signal:**
        r_z_t = exp(- |V(o_t, z_t, prev_a_t) − G_t| )
   This is bounded in (0, 1] and rewards z's that make V's prediction
   accurate. A separate z-value head V_z provides the baseline for
   GAE/PPO. The action policy is unaffected — standard PPO on the env
   reward, baselined with V.

Total loss:
    L = L_a_clip                   # PPO action surrogate (env return)
      + c_v   · MSE(V, G)          # value of env return
      + c_z   · L_z_clip           # PPO z surrogate (cost return)
      + c_v_z · MSE(V_z, G_z)      # value of cost return
      − c_ent_a · H[π_a]
      − c_ent_z · H[π_z]

This is the long-horizon credit-assignment recipe for the recurrent
latent — the principled S2PG-style alternative to per-step regression
on z. See `flow_dagger_variants.tex` (sec:s2pg-return) for the math.
"""
import ast
import warnings
from omegaconf import open_dict
from dataclasses import dataclass
from typing import Any, NamedTuple, Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import flax
import optax
import distrax
from omegaconf import DictConfig, OmegaConf, ListConfig

from loco_mujoco.algorithms import (JaxRLAlgorithmBase, AgentConfBase, AgentStateBase,
                                    TrainState, TrainStateBuffer, MetricHandlerTransition)
from loco_mujoco.algorithms.common.networks import FullyConnectedNet, RunningMeanStd, get_activation_fn
from loco_mujoco.algorithms.ppo_jax import PPOJax, PPOSummaryMetrics
from loco_mujoco.core.wrappers import LogWrapper, NStepWrapper, LogEnvState, VecEnv
from loco_mujoco.environments.base import TrajState
from loco_mujoco.core.wrappers.mjx import Metrics
from loco_mujoco.utils import MetricsHandler, ValidationSummary


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ActorCriticS2PG2(nn.Module):
    """Feed-forward encoder + four parallel heads.

    Encoder
    -------
    concat(obs_norm, z_in, prev_a) → MLP(hidden_layer_dims) → h

    Heads
    -----
    action     : h → MultivariateNormalDiag(μ_a(h), σ_a)
    z' (latent): h → MultivariateNormalDiag(μ_z(h), σ_z)
    V          : h → R              (return value, baselines env reward)
    V_z        : h → R              (cost-return value, baselines z reward)

    Critically the action head reads `h` only, never the z' sample, so
    the action gradient does NOT propagate to z'. z' is shaped only by
    the z-policy's PPO loss against r_z_t = exp(-|V − G_t|).
    """
    action_dim: int
    z_dim: int                                   # carry dim (= continuous latent size)
    activation: str = "tanh"
    init_std_a: float = 1.0
    init_std_z: float = 0.1
    learnable_std: bool = True
    hidden_layer_dims: Sequence[int] = (512, 256)
    use_running_mean_stand: bool = True

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)

    @nn.compact
    def __call__(self, obs, hidden_state, prev_action):
        if self.use_running_mean_stand:
            obs_norm = RunningMeanStd()(obs)
        else:
            obs_norm = jnp.atleast_2d(obs)

        # Shared encoder over (obs, z_in, prev_a).
        x = jnp.concatenate([obs_norm, hidden_state, prev_action], axis=-1)
        for dim in self.hidden_layer_dims:
            x = nn.Dense(dim,
                         kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                         bias_init=nn.initializers.constant(0.0))(x)
            x = self.activation_fn(x)
        h = x

        # Action head: depends on h only — no z' coupling.
        action_mean = nn.Dense(self.action_dim,
                               kernel_init=nn.initializers.orthogonal(0.01),
                               bias_init=nn.initializers.constant(0.0))(h)
        log_std_a = self.param("log_std_a",
                               nn.initializers.constant(jnp.log(self.init_std_a)),
                               (self.action_dim,))
        if not self.learnable_std:
            log_std_a = jax.lax.stop_gradient(log_std_a)
        dist_a = distrax.MultivariateNormalDiag(action_mean, jnp.exp(log_std_a))

        # z' head: depends on h.
        next_z_mean = nn.Dense(self.z_dim,
                               kernel_init=nn.initializers.orthogonal(0.01),
                               bias_init=nn.initializers.constant(0.0))(h)
        log_std_z = self.param("log_std_z",
                               nn.initializers.constant(jnp.log(self.init_std_z)),
                               (self.z_dim,))
        if not self.learnable_std:
            log_std_z = jax.lax.stop_gradient(log_std_z)
        dist_z = distrax.MultivariateNormalDiag(next_z_mean, jnp.exp(log_std_z))

        # V (env-return baseline).
        v = jnp.squeeze(
            nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0),
                     bias_init=nn.initializers.constant(0.0))(h),
            axis=-1,
        )
        # V_z (cost-return baseline for the z-policy).
        v_z = jnp.squeeze(
            nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0),
                     bias_init=nn.initializers.constant(0.0))(h),
            axis=-1,
        )

        return dist_a, dist_z, v, v_z


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------

class S2PG2Policy:
    """Forward-pass helper for S2PG2."""

    def __init__(self, network: nn.Module, action_dim: int, z_dim: int):
        self.network = network
        self.action_dim = action_dim
        self.z_dim = z_dim

    def get_action_z_and_values(self, obs, hidden_state, prev_action, train_state, rng):
        """Sample action and next-z independently. Returns:
           (a, z_next, log_prob_a, log_prob_z, V, V_z, train_state)"""
        dist_a, dist_z, v, v_z, train_state = self._forward(obs, hidden_state, prev_action, train_state)
        rng_a, rng_z = jax.random.split(rng)
        a      = dist_a.sample(seed=rng_a)
        z_next = dist_z.sample(seed=rng_z)
        log_prob_a = dist_a.log_prob(a)
        log_prob_z = dist_z.log_prob(z_next)
        return a, z_next, log_prob_a, log_prob_z, v, v_z, train_state

    def get_dists_and_values(self, obs, hidden_state, prev_action, train_state):
        return self._forward(obs, hidden_state, prev_action, train_state)

    def _forward(self, obs, hidden_state, prev_action, train_state):
        y, updates = self.network.apply(
            {"params": train_state.params, "run_stats": train_state.run_stats},
            obs, hidden_state, prev_action,
            mutable=["run_stats"],
        )
        dist_a, dist_z, v, v_z = y
        return dist_a, dist_z, v, v_z, train_state.replace(run_stats=updates["run_stats"])


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------

class S2PG2Transition(NamedTuple):
    done: jnp.ndarray
    absorbing: jnp.ndarray
    action: jnp.ndarray             # env action a (action_dim)
    next_hidden: jnp.ndarray        # z' sampled at this step (z_dim)
    hidden_state: jnp.ndarray       # z_t (input to encoder this step)
    prev_action: jnp.ndarray        # a_{t-1}
    value: jnp.ndarray              # V_t
    value_z: jnp.ndarray            # V_z_t (cost-return baseline)
    reward: jnp.ndarray             # env reward
    log_prob_a: jnp.ndarray
    log_prob_z: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    traj_state: TrajState
    metrics: Metrics


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------

@struct.dataclass
class S2PG2SummaryMetrics(PPOSummaryMetrics):
    mean_z_actor_loss: float = 0.0       # PPO surrogate on z' (negated)
    mean_z_value_loss: float = 0.0       # MSE(V_z, G_z)
    mean_z_entropy: float = 0.0
    mean_z_approx_kl: float = 0.0
    mean_z_clip_fraction: float = 0.0
    mean_r_z: float = 0.0                # per-step z reward: exp(-|V−G|)
    mean_g_z: float = 0.0                # discounted cost return: G_z_t = Σ γ^k r_z_{t+k}
    mean_v_z: float = 0.0                # learned baseline V_z's prediction (stored)
    mean_advantage_z: float = 0.0        # A_z_t = G_z_t − V_z_t (pre-normalisation)
    mean_v_fit_error: float = 0.0        # |V−G| diagnostic (the V's fit error)


# ---------------------------------------------------------------------------
# Agent conf / state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class S2PG2AgentConf(AgentConfBase):
    config: DictConfig
    network: ActorCriticS2PG2
    tx: Any
    z_dim: int

    def serialize(self):
        conf_dict = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
        return {"config": conf_dict,
                "network": flax.serialization.to_state_dict(self.network)}

    @classmethod
    def from_dict(cls, d):
        config = OmegaConf.create(d["config"])
        tx = S2PG2Jax._get_optimizer(config)
        network = flax.serialization.from_state_dict(
            S2PG2Jax._build_network(config), d["network"]
        )
        z_dim = int(network.z_dim)
        return cls(config=config, network=network, tx=tx, z_dim=z_dim)


@struct.dataclass
class S2PG2AgentState(AgentStateBase):
    train_state: TrainState
    env_state: Any = None
    last_obs: Any = None
    hidden_state: Any = None        # z carried across chunks
    prev_action: Any = None         # a_{t-1} carried across chunks

    def serialize(self):
        return {"train_state": flax.serialization.to_state_dict(self.train_state)}

    @classmethod
    def from_dict(cls, d, agent_conf):
        train_state = TrainState.create(
            apply_fn=agent_conf.network.apply,
            tx=agent_conf.tx,
            params=d["train_state"]["params"],
            run_stats=d["train_state"]["run_stats"],
        )
        opt_state = flax.serialization.from_state_dict(
            train_state.opt_state, d["train_state"]["opt_state"]
        )
        return cls(train_state=train_state.replace(opt_state=opt_state))


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class S2PG2Jax(PPOJax):
    """S2PG2: PPO with decoupled action and z' policies + V_z head.

    See module docstring for the formulation. Inherits PPOJax for env
    wrapping, optimizer construction, learning-rate schedule, and the
    overall train/eval skeleton.
    """

    _agent_conf = S2PG2AgentConf
    _agent_state = S2PG2AgentState

    # ------------------------------------------------------------------
    # Net construction
    # ------------------------------------------------------------------

    @classmethod
    def _build_network(cls, config: DictConfig) -> ActorCriticS2PG2:
        exp = config.experiment
        hidden_layers = (exp.hidden_layers
                         if isinstance(exp.hidden_layers, (list, ListConfig))
                         else ast.literal_eval(exp.hidden_layers))
        return ActorCriticS2PG2(
            action_dim=int(exp.action_dim),
            z_dim=int(exp.z_dim),
            activation=str(exp.activation),
            init_std_a=float(exp.init_std),
            init_std_z=float(getattr(exp, "init_std_z", 0.1)),
            learnable_std=bool(exp.learnable_std),
            hidden_layer_dims=tuple(hidden_layers),
            use_running_mean_stand=bool(getattr(exp, "use_running_mean_stand", True)),
        )

    # ------------------------------------------------------------------
    # Conf / state init
    # ------------------------------------------------------------------

    @classmethod
    def init_agent_conf(cls, env, config):
        with open_dict(config.experiment):
            config.experiment.num_updates = (
                config.experiment.total_timesteps // config.experiment.num_steps //
                config.experiment.num_envs)
            config.experiment.minibatch_size = (
                config.experiment.num_envs * config.experiment.num_steps //
                config.experiment.num_minibatches)
            config.experiment.validation_interval = (
                config.experiment.num_updates // config.experiment.validation.num)
            config.experiment.validation.num = int(
                config.experiment.num_updates // config.experiment.validation_interval)
            config.experiment.action_dim = env.info.action_space.shape[0]
            # z_dim is the recurrent-latent dimension; default if unset.
            if "z_dim" not in config.experiment:
                config.experiment.z_dim = int(getattr(config.experiment, "hidden_state_dim", 64))

        network = cls._build_network(config)
        tx = cls._get_optimizer(config)
        return cls._agent_conf(config=config, network=network, tx=tx,
                               z_dim=int(config.experiment.z_dim))

    @classmethod
    def init_agent_state(cls, env, agent_conf: S2PG2AgentConf, rng) -> S2PG2AgentState:
        config, network, tx = agent_conf.config.experiment, agent_conf.network, agent_conf.tx
        wrapped = cls._wrap_env(env, config)
        obs_dim = wrapped.info.observation_space.shape[0]
        action_dim = config.action_dim
        z_dim = agent_conf.z_dim
        rng, _rng = jax.random.split(rng)
        params = network.init(
            _rng,
            jnp.zeros((1, obs_dim)),
            jnp.zeros((1, z_dim)),
            jnp.zeros((1, action_dim)),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params["params"],
            run_stats=params["run_stats"],
            tx=tx,
        )
        return cls._agent_state(train_state=train_state)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def _train_fn(cls, rng, env,
                  agent_conf: S2PG2AgentConf,
                  agent_state: S2PG2AgentState = None,
                  traj=None,
                  mh: MetricsHandler = None):

        config, network, tx = agent_conf.config.experiment, agent_conf.network, agent_conf.tx
        action_dim = config.action_dim
        z_dim = agent_conf.z_dim

        env = cls._wrap_env(env, config)
        policy = S2PG2Policy(network, action_dim, z_dim)

        # ---- z-policy specific knobs ----
        # Coefficient on the z-policy clipped surrogate.
        z_coef = float(getattr(config, "z_coef", 1.0))
        # Coefficient on the V_z value loss.
        vf_z_coef = float(getattr(config, "vf_z_coef", 0.5))
        # Entropy bonus on the z-policy.
        ent_z_coef = float(getattr(config, "ent_z_coef", 0.0))
        # Discount + GAE-λ for the cost return. Default: same as env.
        gamma_z = float(getattr(config, "gamma_z", config.gamma))
        gae_lambda_z = float(getattr(config, "gae_lambda_z", config.gae_lambda))

        # ---- init train state ----
        if agent_state is not None:
            train_state = agent_state.train_state.replace(apply_fn=network.apply)
        else:
            rng, _rng1 = jax.random.split(rng)
            obs_dim = env.info.observation_space.shape[0]
            params = network.init(
                _rng1,
                jnp.zeros(obs_dim),
                jnp.zeros(z_dim),
                jnp.zeros(action_dim),
            )
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=params["params"],
                run_stats=params["run_stats"],
                tx=tx,
            )

        # ---- init env state ----
        if agent_state is not None and agent_state.env_state is not None:
            env_state = agent_state.env_state
            obsv = agent_state.last_obs
        else:
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config.num_envs)
            obsv, env_state = env.reset(reset_rng, traj)

        # ---- init carries ----
        if agent_state is not None and agent_state.hidden_state is not None:
            hidden_state = agent_state.hidden_state
        else:
            hidden_state = jnp.zeros((config.num_envs, z_dim))
        if agent_state is not None and agent_state.prev_action is not None:
            prev_action = agent_state.prev_action
        else:
            prev_action = jnp.zeros((config.num_envs, action_dim))

        train_state_buffer = TrainStateBuffer.create(train_state, config.validation.num)

        # ---- update step ----
        def _update_step(runner_state, unused):

            # -- collect rollout --
            def _env_step(rs, unused):
                train_state, env_state, last_obs, hidden_state, prev_action, tsb, rng = rs
                rng, _rng = jax.random.split(rng)
                a, z_next, lpa, lpz, v, v_z, train_state = \
                    policy.get_action_z_and_values(
                        last_obs, hidden_state, prev_action, train_state, _rng,
                    )

                obsv, reward, absorbing, done, info, env_state = env.step(env_state, a, traj)

                # Reset z carry and prev_action on episode termination.
                next_hidden = z_next * (1 - done)[..., None]
                next_prev_action = a * (1 - done)[..., None]

                logged_metrics = env_state.find(LogEnvState).metrics
                transition = S2PG2Transition(
                    done, absorbing, a, z_next, hidden_state, prev_action,
                    v, v_z, reward, lpa, lpz, last_obs, info,
                    env_state.additional_carry.traj_state, logged_metrics,
                )
                rs = (train_state, env_state, obsv, next_hidden, next_prev_action, tsb, rng)
                return rs, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config.num_steps)

            # -- bootstrap V at the end of the rollout for the env GAE --
            train_state, env_state, last_obs, hidden_state, prev_action, tsb, rng = runner_state
            _, _, last_val, last_val_z, _ = policy.get_dists_and_values(
                last_obs, hidden_state, prev_action, train_state,
            )

            # -- env-return GAE (standard) --
            def _gae(traj_batch, last_val, gamma, lam, value_field):
                def _step(carry, transition):
                    gae, next_value = carry
                    v_t = getattr(transition, value_field)
                    delta = transition.reward_or_cost + gamma * next_value * (1 - transition.absorbing) - v_t
                    gae = delta + gamma * lam * (1 - transition.done) * gae
                    return (gae, v_t), gae
                _, advs = jax.lax.scan(
                    _step, (jnp.zeros_like(last_val), last_val),
                    traj_batch, reverse=True, unroll=16,
                )
                return advs

            # The GAE step expects an attribute `reward_or_cost` for the per-step
            # "reward" — we splice the appropriate signal into a wrapper struct.
            class _GAEItem(NamedTuple):
                reward_or_cost: jnp.ndarray
                absorbing:      jnp.ndarray
                done:           jnp.ndarray
                value:          jnp.ndarray
                value_z:        jnp.ndarray

            # 1. Standard env-reward GAE → advantages, targets for V.
            env_items = _GAEItem(
                reward_or_cost=traj_batch.reward,
                absorbing=traj_batch.absorbing,
                done=traj_batch.done,
                value=traj_batch.value,
                value_z=traj_batch.value_z,
            )
            advantages = _gae(env_items, last_val, config.gamma, config.gae_lambda,
                              value_field="value")
            targets = advantages + traj_batch.value
            # `targets` are G_t for the V head. Now compute the per-step
            # z-reward r_z_t = exp(- |V_stored − G_t| ) and run GAE on
            # those with V_z as the baseline.
            r_z = jnp.exp(-jnp.abs(traj_batch.value - targets))
            cost_items = _GAEItem(
                reward_or_cost=r_z,
                absorbing=traj_batch.absorbing,
                done=traj_batch.done,
                value=traj_batch.value,
                value_z=traj_batch.value_z,
            )
            advantages_z = _gae(cost_items, last_val_z, gamma_z, gae_lambda_z,
                                value_field="value_z")
            targets_z = advantages_z + traj_batch.value_z

            # -- PPO update --
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets, advantages_z, targets_z = batch_info

                    def _loss_fn(params, traj_batch, A, R, A_z, R_z):
                        dist_a, dist_z, v, v_z, _ = policy.get_dists_and_values(
                            traj_batch.obs, traj_batch.hidden_state, traj_batch.prev_action,
                            train_state.replace(params=params),
                        )
                        lpa_new = dist_a.log_prob(traj_batch.action)
                        lpz_new = dist_z.log_prob(traj_batch.next_hidden)

                        # ---- V loss (env return, clipped) ----
                        v_clipped = traj_batch.value + (v - traj_batch.value).clip(
                            -config.clip_eps, config.clip_eps)
                        v_loss = 0.5 * jnp.maximum(
                            jnp.square(v - R),
                            jnp.square(v_clipped - R),
                        ).mean()

                        # ---- V_z loss (cost return, clipped) ----
                        v_z_clipped = traj_batch.value_z + (v_z - traj_batch.value_z).clip(
                            -config.clip_eps, config.clip_eps)
                        v_z_loss = 0.5 * jnp.maximum(
                            jnp.square(v_z - R_z),
                            jnp.square(v_z_clipped - R_z),
                        ).mean()

                        # ---- PPO action loss ----
                        ratio_a = jnp.exp(lpa_new - traj_batch.log_prob_a)
                        A_norm = (A - A.mean()) / (A.std() + 1e-8)
                        loss_a = -jnp.minimum(
                            ratio_a * A_norm,
                            jnp.clip(ratio_a, 1 - config.clip_eps, 1 + config.clip_eps) * A_norm,
                        ).mean()
                        ent_a = dist_a.entropy().mean()
                        approx_kl_a = (traj_batch.log_prob_a - lpa_new).mean()
                        clip_frac_a = jnp.mean(jnp.abs(ratio_a - 1.0) > config.clip_eps)

                        # ---- PPO z loss ----
                        ratio_z = jnp.exp(lpz_new - traj_batch.log_prob_z)
                        A_z_norm = (A_z - A_z.mean()) / (A_z.std() + 1e-8)
                        loss_z = -jnp.minimum(
                            ratio_z * A_z_norm,
                            jnp.clip(ratio_z, 1 - config.clip_eps, 1 + config.clip_eps) * A_z_norm,
                        ).mean()
                        ent_z = dist_z.entropy().mean()
                        approx_kl_z = (traj_batch.log_prob_z - lpz_new).mean()
                        clip_frac_z = jnp.mean(jnp.abs(ratio_z - 1.0) > config.clip_eps)

                        total = (loss_a
                                 + config.vf_coef * v_loss
                                 + z_coef * loss_z
                                 + vf_z_coef * v_z_loss
                                 - config.ent_coef * ent_a
                                 - ent_z_coef * ent_z)
                        aux = (v_loss, loss_a, ent_a, approx_kl_a, clip_frac_a,
                               loss_z, v_z_loss, ent_z, approx_kl_z, clip_frac_z)
                        return total, aux

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets,
                        advantages_z, targets_z,
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, advantages_z, targets_z, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config.minibatch_size * config.num_minibatches
                assert batch_size == config.num_steps * config.num_envs
                perm = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets, advantages_z, targets_z)
                batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled = jax.tree.map(lambda x: jnp.take(x, perm, axis=0), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:])),
                    shuffled,
                )
                train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets,
                                advantages_z, targets_z, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets,
                            advantages_z, targets_z, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs,
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # -- metrics --
            counter = ((train_state.step + 1) // config.num_minibatches) // config.update_epochs
            logged = traj_batch.metrics

            (v_loss_arr, a_loss_arr, ent_a_arr, kl_a_arr, clip_a_arr,
             z_loss_arr, vz_loss_arr, ent_z_arr, kl_z_arr, clip_z_arr) = loss_info[1]
            mean_v_fit_error = jnp.mean(jnp.abs(traj_batch.value - targets))

            if config.anneal_lr:
                current_lr = cls._linear_lr_schedule(
                    train_state.step, config.num_minibatches,
                    config.update_epochs, config.lr, config.num_updates,
                )
            else:
                current_lr = jnp.array(config.lr)

            metric = S2PG2SummaryMetrics(
                mean_episode_return=jnp.mean(logged.returned_episode_returns),
                mean_episode_length=jnp.mean(logged.returned_episode_lengths),
                max_timestep=jnp.max(logged.timestep * config.num_envs),
                mean_value_loss=jnp.mean(v_loss_arr),
                mean_actor_loss=jnp.mean(a_loss_arr),
                mean_entropy=jnp.mean(ent_a_arr),
                mean_approx_kl=jnp.mean(kl_a_arr),
                mean_clip_fraction=jnp.mean(clip_a_arr),
                learning_rate=current_lr,
                mean_z_actor_loss=jnp.mean(z_loss_arr),
                mean_z_value_loss=jnp.mean(vz_loss_arr),
                mean_z_entropy=jnp.mean(ent_z_arr),
                mean_z_approx_kl=jnp.mean(kl_z_arr),
                mean_z_clip_fraction=jnp.mean(clip_z_arr),
                mean_r_z=jnp.mean(jnp.exp(-jnp.abs(traj_batch.value - targets))),
                mean_g_z=jnp.mean(targets_z),
                mean_v_z=jnp.mean(traj_batch.value_z),
                mean_advantage_z=jnp.mean(advantages_z),
                mean_v_fit_error=mean_v_fit_error,
            )

            # -- validation --
            def _evaluation_step():
                eval_hidden = jnp.zeros((config.validation.num_envs, z_dim))
                eval_prev_action = jnp.zeros((config.validation.num_envs, action_dim))

                def _eval_env(eval_rs, unused):
                    ts, env_state, last_obs, eh, ep, rng = eval_rs
                    rng, _rng = jax.random.split(rng)
                    a, z_next, _, _, _, _, ts = policy.get_action_z_and_values(
                        last_obs, eh, ep, ts, _rng,
                    )
                    obsv, reward, absorbing, done, info, env_state = env.step(env_state, a, traj)
                    eh = z_next * (1 - done)[..., None]
                    ep = a * (1 - done)[..., None]
                    transition = MetricHandlerTransition(env_state, env_state.find(LogEnvState).metrics)
                    eval_rs = (ts, env_state, obsv, eh, ep, rng)
                    return eval_rs, transition

                rng_eval = runner_state[-1]
                reset_rng = jax.random.split(rng_eval, config.validation.num_envs)
                obsv_eval, eval_env_state = env.reset(reset_rng, traj)
                eval_rs = (train_state, eval_env_state, obsv_eval, eval_hidden, eval_prev_action, rng_eval)
                _, eval_traj_batch = jax.lax.scan(_eval_env, eval_rs, None, config.validation.num_steps)
                return mh(eval_traj_batch.env_state)

            if mh is None:
                validation_metrics = ValidationSummary()
            else:
                validation_metrics = jax.lax.cond(
                    counter % config.validation_interval == 0,
                    _evaluation_step, mh.get_zero_container,
                )

            tsb = jax.lax.cond(
                counter % config.validation_interval == 0,
                lambda x, y: TrainStateBuffer.add(x, y),
                lambda x, y: x,
                tsb, train_state,
            )

            runner_state = (train_state, env_state, last_obs, hidden_state, prev_action, tsb, rng)
            return runner_state, (metric, validation_metrics)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, hidden_state, prev_action,
                        train_state_buffer, _rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config.num_updates)

        agent_state_out = cls._agent_state(
            train_state=runner_state[0],
            env_state=runner_state[1],
            last_obs=runner_state[2],
            hidden_state=runner_state[3],
            prev_action=runner_state[4],
        )
        return {
            "agent_state": agent_state_out,
            "training_metrics": metrics[0],
            "validation_metrics": metrics[1],
        }

    # ------------------------------------------------------------------
    # Play
    # ------------------------------------------------------------------

    @classmethod
    def play_policy(cls, env,
                    agent_conf: S2PG2AgentConf,
                    agent_state: S2PG2AgentState,
                    n_envs: int, n_steps=None, render=True,
                    record=False, rng=None, deterministic=False,
                    use_mujoco=False, wrap_env=True,
                    train_state_seed=None, traj=None):

        if use_mujoco:
            assert n_envs == 1, "Only one mujoco env can be run at a time."

        config = agent_conf.config.experiment
        action_dim = config.action_dim
        z_dim = agent_conf.z_dim
        _policy = S2PG2Policy(agent_conf.network, action_dim, z_dim)

        train_state = agent_state.train_state
        if deterministic:
            train_state.params["log_std_a"] = np.ones_like(train_state.params["log_std_a"]) * -np.inf

        if config.n_seeds > 1:
            assert train_state_seed is not None, (
                "Loaded train state has multiple seeds. Please specify train_state_seed for replay."
            )
            train_state = jax.tree.map(lambda x: x[train_state_seed], train_state)

        if not render and n_steps is None and not record:
            warnings.warn("No rendering, no record, no n_steps specified — runs forever with no effect.")

        if wrap_env and not use_mujoco:
            env = cls._wrap_env(env, config)

        if rng is None:
            rng = jax.random.key(0)

        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]

        hidden_state = jnp.zeros((n_envs, z_dim))
        prev_action = jnp.zeros((n_envs, action_dim))

        def sample(ts, obs, h, p, _rng):
            a, z_next, _, _, _, _, ts = _policy.get_action_z_and_values(obs, h, p, ts, _rng)
            return a, z_next, ts

        plcy_call = jax.jit(sample)

        if use_mujoco:
            obs = env.reset()
            env_state = None
        else:
            obs, env_state = env.reset(env_keys, traj)

        if n_steps is None:
            n_steps = np.iinfo(np.int32).max

        for _ in range(n_steps):
            rng, _rng = jax.random.split(rng)
            action, next_hidden, train_state = plcy_call(train_state, obs, hidden_state, prev_action, _rng)
            action = jnp.atleast_2d(action)

            if use_mujoco:
                obs, reward, absorbing, done, info = env.step(action)
                done_f = jnp.array(done, dtype=jnp.float32)
                hidden_state = next_hidden * (1 - done_f)[..., None]
                prev_action = action * (1 - done_f)[..., None]
                env.render(record=True)
                if done:
                    obs = env.reset()
                    hidden_state = jnp.zeros((n_envs, z_dim))
                    prev_action = jnp.zeros((n_envs, action_dim))
            else:
                obs, reward, absorbing, done, info, env_state = env.step(env_state, action, traj)
                hidden_state = next_hidden * (1 - done)[..., None]
                prev_action = action * (1 - done)[..., None]
                env.mjx_render(env_state, record=record)

        # Finalise the video (closes the recorder, sets env.video_file_path).
        env.stop()
