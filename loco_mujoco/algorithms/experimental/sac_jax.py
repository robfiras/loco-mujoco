"""SAC (Soft Actor-Critic) on top of OffPolicyBase.

The shared off-policy machinery (replay buffer, twin Q, target updates,
collection loop, scan over env steps) lives in `offpolicy_base.py`. This
module only owns the SAC-specific pieces:

  - Squashed-Gaussian actor (`SACActorNet`)
  - Sampled-action log-prob (squashed Gaussian)
  - Entropy bonus in Q-target ( -alpha · log_pi_next )
  - Actor loss ( E[ alpha · log_pi - min(Q1,Q2) ] )
  - Learnable temperature (alpha) with target-entropy constraint
  - Metrics: mean_alpha, mean_entropy, mean_std
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
    """Diagonal-Gaussian actor with tanh squashing."""

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
    """Sample from a tanh-squashed Gaussian and compute log_prob."""
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
    config: DictConfig
    actor_net: SACActorNet
    critic_net: OffPolicyCriticNet
    actor_tx: Any
    critic_tx: Any
    alpha_tx: Any

    def serialize(self):
        conf_dict = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
        return {
            "config": conf_dict,
            "actor_net": flax.serialization.to_state_dict(self.actor_net),
            "critic_net": flax.serialization.to_state_dict(self.critic_net),
        }

    @classmethod
    def from_dict(cls, d):
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
    """Wrapper around a learnable log_alpha parameter + its optimiser state."""
    train_state: TrainState
    target_entropy: float
    learnable: bool


@struct.dataclass
class SACAgentState(OffPolicyAgentState):
    """SAC agent state — extra_state carries `_LogAlphaState`."""

    @classmethod
    def from_dict(cls, d, agent_conf):
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

        replay_buffer = ReplayBuffer.create(obs_dim, action_dim, int(exp.buffer_size))

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
    _agent_conf = SACAgentConf
    _agent_state = SACAgentState

    # ---------- Conf init -----------------------------------------------
    @classmethod
    def init_agent_conf(cls, env, config):
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
        replay_buffer = ReplayBuffer.create(obs_dim, action_dim, int(exp.buffer_size))

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
        (mean_a, log_std_a), updates = actor_apply_fn(
            {"params": actor_params, "run_stats": actor_run_stats},
            obs_b, mutable=["run_stats"],
        )
        sampled_action, log_pi = _squashed_gaussian_sample_and_log_prob(
            mean_a, log_std_a, rng
        )
        (q1_a, q2_a), _ = critic_apply_fn(
            {"params": critic_st.params, "run_stats": critic_st.run_stats},
            obs_b, sampled_action, mutable=["run_stats"],
        )
        q_a = jnp.minimum(q1_a, q2_a)
        alpha = jnp.exp(extra_state.train_state.params["log_alpha"])
        loss = jnp.mean(alpha * log_pi - q_a)
        return loss, {
            "log_pi": log_pi,
            "run_stats": updates["run_stats"],
        }

    @classmethod
    def _update_extra(cls, extra_state, actor_aux, rng, exp):
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
