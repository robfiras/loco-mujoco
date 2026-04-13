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

from loco_mujoco.algorithms import (JaxRLAlgorithmBase, AgentConfBase, AgentStateBase, TrainState)
from loco_mujoco.algorithms.common.networks import RunningMeanStd, get_activation_fn
from loco_mujoco.core.wrappers import LogWrapper, LogEnvState, VecEnv, NormalizeVecReward, SummaryMetrics
from loco_mujoco.utils import MetricsHandler, ValidationSummary


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class SACActorNet(nn.Module):
    """Diagonal-Gaussian actor with tanh squashing."""

    action_dim: int
    hidden_layer_dims: tuple = (256, 256)
    activation: str = "tanh"
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x):
        activation_fn = get_activation_fn(self.activation)
        x = RunningMeanStd()(x)
        for dim in self.hidden_layer_dims:
            x = nn.Dense(dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
            x = activation_fn(x)
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        log_std = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class _QNet(nn.Module):
    """Single Q-network sub-module."""

    hidden_layer_dims: tuple = (256, 256)
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation_fn = get_activation_fn(self.activation)
        for dim in self.hidden_layer_dims:
            x = nn.Dense(dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
            x = activation_fn(x)
        return jnp.squeeze(
            nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x), axis=-1
        )


class SACCriticNet(nn.Module):
    """Twin Q-networks.  Returns (q1, q2) given (obs, action)."""

    hidden_layer_dims: tuple = (256, 256)
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs, action):
        # Normalise only the observation part; actions are already in [-1, 1]
        obs = RunningMeanStd()(obs)
        x = jnp.concatenate([obs, action], axis=-1)
        q1 = _QNet(self.hidden_layer_dims, self.activation, name="q1")(x)
        q2 = _QNet(self.hidden_layer_dims, self.activation, name="q2")(x)
        return q1, q2


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

@struct.dataclass
class ReplayBuffer:
    """Fixed-size circular replay buffer stored as a JAX pytree."""

    obs: jnp.ndarray        # (capacity, obs_dim)
    next_obs: jnp.ndarray   # (capacity, obs_dim)
    action: jnp.ndarray     # (capacity, action_dim)
    reward: jnp.ndarray     # (capacity,)
    done: jnp.ndarray       # (capacity,)
    ptr: int                # write pointer
    size: int               # current number of valid entries

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
        """Insert a batch of transitions (vectorised over envs).

        obs/next_obs: (num_envs, obs_dim)
        action:       (num_envs, action_dim)
        reward/done:  (num_envs,)
        """
        capacity = self.obs.shape[0]
        batch_size = obs.shape[0]

        # circular indices
        indices = (self.ptr + jnp.arange(batch_size)) % capacity

        new_obs = self.obs.at[indices].set(obs)
        new_next_obs = self.next_obs.at[indices].set(next_obs)
        new_action = self.action.at[indices].set(action)
        new_reward = self.reward.at[indices].set(reward)
        new_done = self.done.at[indices].set(done.astype(jnp.float32))
        new_ptr = (self.ptr + batch_size) % capacity
        new_size = jnp.minimum(self.size + batch_size, capacity)

        return self.replace(
            obs=new_obs, next_obs=new_next_obs, action=new_action,
            reward=new_reward, done=new_done, ptr=new_ptr, size=new_size,
        )

    def sample(self, rng, batch_size: int):
        """Sample a random minibatch from the valid portion of the buffer."""
        indices = jax.random.randint(rng, (batch_size,), 0, self.size)
        return (
            self.obs[indices],
            self.next_obs[indices],
            self.action[indices],
            self.reward[indices],
            self.done[indices],
        )


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

@struct.dataclass
class SACSummaryMetrics(SummaryMetrics):
    mean_critic_loss: float = 0.0
    mean_actor_loss: float = 0.0
    mean_alpha_loss: float = 0.0
    mean_alpha: float = 0.0
    buffer_size: int = 0


# ---------------------------------------------------------------------------
# Agent config / state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SACAgentConf(AgentConfBase):
    config: DictConfig
    actor_net: SACActorNet
    critic_net: SACCriticNet
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
        critic_net = flax.serialization.from_state_dict(SACCriticNet, d["critic_net"])

        actor_tx = optax.chain(
            optax.clip_by_global_norm(float(getattr(exp, 'max_grad_norm', 0.5))),
            optax.adam(float(exp.lr_actor)),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(float(getattr(exp, 'max_grad_norm', 0.5))),
            optax.adam(float(exp.lr_critic)),
        )
        alpha_tx = optax.adam(float(exp.lr_alpha))
        return cls(config=config, actor_net=actor_net, critic_net=critic_net,
                   actor_tx=actor_tx, critic_tx=critic_tx, alpha_tx=alpha_tx)


@struct.dataclass
class SACAgentState(AgentStateBase):
    actor_state: TrainState
    critic_state: TrainState
    target_critic_params: Any
    target_critic_run_stats: Any
    log_alpha_state: TrainState
    replay_buffer: ReplayBuffer
    env_state: Any = None
    last_obs: Any = None

    def serialize(self):
        return {
            "actor_state": flax.serialization.to_state_dict(self.actor_state),
            "critic_state": flax.serialization.to_state_dict(self.critic_state),
            "target_critic_params": flax.serialization.to_state_dict(
                {"params": self.target_critic_params,
                 "run_stats": self.target_critic_run_stats}
            ),
            "log_alpha": flax.serialization.to_state_dict(self.log_alpha_state),
        }

    @classmethod
    def from_dict(cls, d, agent_conf):
        exp = agent_conf.config.experiment
        actor_net = agent_conf.actor_net
        critic_net = agent_conf.critic_net

        obs_dim = int(exp.obs_dim)
        action_dim = int(exp.action_dim)

        rng = jax.random.PRNGKey(0)

        # Reconstruct actor train state
        actor_params_init = actor_net.init(rng, jnp.zeros((1, obs_dim)))
        actor_state = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_params_init["params"],
            run_stats=actor_params_init["run_stats"],
            tx=agent_conf.actor_tx,
        )
        actor_state = flax.serialization.from_state_dict(actor_state, d["actor_state"])
        actor_state = actor_state.replace(apply_fn=actor_net.apply)

        # Reconstruct critic train state
        critic_params_init = critic_net.init(rng, jnp.zeros((1, obs_dim)), jnp.zeros((1, action_dim)))
        critic_state = TrainState.create(
            apply_fn=critic_net.apply,
            params=critic_params_init["params"],
            run_stats=critic_params_init["run_stats"],
            tx=agent_conf.critic_tx,
        )
        critic_state = flax.serialization.from_state_dict(critic_state, d["critic_state"])
        critic_state = critic_state.replace(apply_fn=critic_net.apply)

        # Reconstruct target critic
        target_dict = flax.serialization.from_state_dict(
            {"params": critic_params_init["params"],
             "run_stats": critic_params_init["run_stats"]},
            d["target_critic_params"]
        )

        # Reconstruct log_alpha state
        log_alpha_state = TrainState.create(
            apply_fn=lambda p, x: p["log_alpha"],
            params={"log_alpha": jnp.array(0.0)},
            run_stats={},
            tx=agent_conf.alpha_tx,
        )
        log_alpha_state = flax.serialization.from_state_dict(log_alpha_state, d["log_alpha"])
        log_alpha_state = log_alpha_state.replace(apply_fn=lambda p, x: p["log_alpha"])

        # Replay buffer not saved — create empty one
        capacity = int(exp.buffer_size)
        replay_buffer = ReplayBuffer.create(obs_dim, action_dim, capacity)

        return cls(
            actor_state=actor_state,
            critic_state=critic_state,
            target_critic_params=target_dict["params"],
            target_critic_run_stats=target_dict["run_stats"],
            log_alpha_state=log_alpha_state,
            replay_buffer=replay_buffer,
        )


# ---------------------------------------------------------------------------
# Helper: squashed Gaussian sample + log-prob
# ---------------------------------------------------------------------------

def _squashed_gaussian_sample_and_log_prob(mean, log_std, rng):
    """
    Sample from a tanh-squashed Gaussian and compute log_prob.

    Returns:
        action:   (*, action_dim)   tanh-squashed, in (-1, 1)
        log_prob: (*,)              log probability under the squashed distribution
    """
    std = jnp.exp(log_std)
    noise = jax.random.normal(rng, shape=mean.shape)
    x = mean + std * noise                          # pre-squash sample
    action = jnp.tanh(x)
    # log det Jacobian of tanh:  sum log(1 - tanh^2(x)) = sum log(1 - a^2)
    log_prob = (
        jnp.sum(-0.5 * ((noise ** 2) + 2 * log_std + jnp.log(2 * jnp.pi)), axis=-1)
        - jnp.sum(jnp.log(1 - action ** 2 + 1e-6), axis=-1)
    )
    return action, log_prob


def _squashed_gaussian_log_prob(mean, log_std, action_tanh):
    """
    Compute log_prob for an *already-squashed* action under the distribution
    defined by (mean, log_std) — used in actor loss.
    """
    # Same as sample path: we need the pre-squash value for the Gaussian pdf,
    # but since we sampled above, we re-use the action values as passed.
    # Instead, we recompute by atanh (safe clamp first).
    x = jnp.arctanh(jnp.clip(action_tanh, -1 + 1e-6, 1 - 1e-6))
    std = jnp.exp(log_std)
    log_prob_gauss = jnp.sum(
        -0.5 * (((x - mean) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)),
        axis=-1,
    )
    log_det_jac = jnp.sum(jnp.log(1 - action_tanh ** 2 + 1e-6), axis=-1)
    return log_prob_gauss - log_det_jac


# ---------------------------------------------------------------------------
# SAC Algorithm
# ---------------------------------------------------------------------------

class SACJax(JaxRLAlgorithmBase):

    _agent_conf = SACAgentConf
    _agent_state = SACAgentState

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @classmethod
    def init_agent_conf(cls, env, config):
        obs_dim = int(env.info.observation_space.shape[0])
        action_dim = int(env.info.action_space.shape[0])

        with open_dict(config.experiment):
            config.experiment.obs_dim = obs_dim
            config.experiment.action_dim = action_dim
            # total environment steps (each step produces num_envs transitions)
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
            log_std_min=float(getattr(exp, 'log_std_min', -20.0)),
            log_std_max=float(getattr(exp, 'log_std_max', 2.0)),
        )
        critic_net = SACCriticNet(
            hidden_layer_dims=tuple(hidden_layers),
            activation=str(exp.activation),
        )

        actor_tx = optax.chain(
            optax.clip_by_global_norm(float(getattr(exp, 'max_grad_norm', 0.5))),
            optax.adam(float(exp.lr_actor)),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(float(getattr(exp, 'max_grad_norm', 0.5))),
            optax.adam(float(exp.lr_critic)),
        )
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
    def init_agent_state(cls, env, agent_conf: SACAgentConf, rng) -> SACAgentState:
        exp = agent_conf.config.experiment
        obs_dim = int(exp.obs_dim)
        action_dim = int(exp.action_dim)

        rng, rng_a, rng_c, rng_env = jax.random.split(rng, 4)

        # Actor
        actor_params = agent_conf.actor_net.init(rng_a, jnp.zeros((1, obs_dim)))
        actor_state = TrainState.create(
            apply_fn=agent_conf.actor_net.apply,
            params=actor_params["params"],
            run_stats=actor_params["run_stats"],
            tx=agent_conf.actor_tx,
        )

        # Critic (twin)
        critic_params = agent_conf.critic_net.init(
            rng_c, jnp.zeros((1, obs_dim)), jnp.zeros((1, action_dim))
        )
        critic_state = TrainState.create(
            apply_fn=agent_conf.critic_net.apply,
            params=critic_params["params"],
            run_stats=critic_params["run_stats"],
            tx=agent_conf.critic_tx,
        )

        # Target critic (copy of critic)
        target_critic_params = critic_params["params"]
        target_critic_run_stats = critic_params["run_stats"]

        # log alpha
        init_log_alpha = jnp.log(float(getattr(exp, 'init_alpha', 0.1)))
        log_alpha_params = {"log_alpha": init_log_alpha}
        log_alpha_state = TrainState.create(
            apply_fn=lambda p, x: p["log_alpha"],
            params=log_alpha_params,
            run_stats={},
            tx=agent_conf.alpha_tx,
        )

        # Replay buffer
        capacity = int(exp.buffer_size)
        replay_buffer = ReplayBuffer.create(obs_dim, action_dim, capacity)

        return cls._agent_state(
            actor_state=actor_state,
            critic_state=critic_state,
            target_critic_params=target_critic_params,
            target_critic_run_stats=target_critic_run_stats,
            log_alpha_state=log_alpha_state,
            replay_buffer=replay_buffer,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def _train_fn(cls, rng, env,
                  agent_conf: SACAgentConf,
                  agent_state: SACAgentState = None,
                  traj=None,
                  mh: MetricsHandler = None):

        exp = agent_conf.config.experiment
        actor_net = agent_conf.actor_net
        critic_net = agent_conf.critic_net

        env = cls._wrap_env(env, exp)

        target_entropy = -float(exp.action_dim)

        # ------------------------------------------------------------------
        # Restore or initialise agent components
        # ------------------------------------------------------------------
        if agent_state is not None:
            actor_state = agent_state.actor_state.replace(apply_fn=actor_net.apply)
            critic_state = agent_state.critic_state.replace(apply_fn=critic_net.apply)
            target_critic_params = agent_state.target_critic_params
            target_critic_run_stats = agent_state.target_critic_run_stats
            log_alpha_state = agent_state.log_alpha_state.replace(
                apply_fn=lambda p, x: p["log_alpha"]
            )
            replay_buffer = agent_state.replay_buffer
        else:
            rng, rng_a, rng_c = jax.random.split(rng, 3)
            obs_dim = int(exp.obs_dim)
            action_dim = int(exp.action_dim)

            actor_params = actor_net.init(rng_a, jnp.zeros((1, obs_dim)))
            actor_state = TrainState.create(
                apply_fn=actor_net.apply,
                params=actor_params["params"],
                run_stats=actor_params["run_stats"],
                tx=agent_conf.actor_tx,
            )

            critic_params = critic_net.init(
                rng_c, jnp.zeros((1, obs_dim)), jnp.zeros((1, action_dim))
            )
            critic_state = TrainState.create(
                apply_fn=critic_net.apply,
                params=critic_params["params"],
                run_stats=critic_params["run_stats"],
                tx=agent_conf.critic_tx,
            )
            target_critic_params = critic_params["params"]
            target_critic_run_stats = critic_params["run_stats"]

            init_log_alpha = jnp.log(float(getattr(exp, 'init_alpha', 0.1)))
            log_alpha_state = TrainState.create(
                apply_fn=lambda p, x: p["log_alpha"],
                params={"log_alpha": init_log_alpha},
                run_stats={},
                tx=agent_conf.alpha_tx,
            )

            capacity = int(exp.buffer_size)
            replay_buffer = ReplayBuffer.create(int(exp.obs_dim), int(exp.action_dim), capacity)

        # ------------------------------------------------------------------
        # Init env
        # ------------------------------------------------------------------
        if agent_state is not None and agent_state.env_state is not None:
            env_state = agent_state.env_state
            last_obs = agent_state.last_obs
        else:
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, exp.num_envs)
            last_obs, env_state = env.reset(reset_rng, traj)

        # ------------------------------------------------------------------
        # Helper: actor forward pass (returns mean, log_std + updated stats)
        # ------------------------------------------------------------------
        def _actor_forward(obs, actor_st):
            (mean, log_std), updates = actor_st.apply_fn(
                {"params": actor_st.params, "run_stats": actor_st.run_stats},
                obs, mutable=["run_stats"],
            )
            actor_st = actor_st.replace(run_stats=updates["run_stats"])
            return mean, log_std, actor_st

        # ------------------------------------------------------------------
        # Helper: critic forward pass
        # ------------------------------------------------------------------
        def _critic_forward(obs, action, critic_st):
            (q1, q2), updates = critic_st.apply_fn(
                {"params": critic_st.params, "run_stats": critic_st.run_stats},
                obs, action, mutable=["run_stats"],
            )
            critic_st = critic_st.replace(run_stats=updates["run_stats"])
            return q1, q2, critic_st

        def _target_critic_forward(obs, action, params, run_stats):
            (q1, q2), updates = critic_net.apply(
                {"params": params, "run_stats": run_stats},
                obs, action, mutable=["run_stats"],
            )
            return q1, q2, updates["run_stats"]

        # ------------------------------------------------------------------
        # Config values used in update logic
        # ------------------------------------------------------------------
        learning_starts = int(getattr(exp, 'learning_starts', exp.batch_size))
        batch_size = int(exp.batch_size)
        tau = float(getattr(exp, 'tau', 0.005))
        gamma = float(exp.gamma)
        learnable_alpha = bool(getattr(exp, 'learnable_alpha', True))
        gradient_steps = int(getattr(exp, 'gradient_steps', 1))

        # ------------------------------------------------------------------
        # Sub-function: collect one transition from each parallel env
        # ------------------------------------------------------------------
        def _collect_transition(actor_state, replay_buffer, env_state, last_obs, rng):
            rng, rng_act = jax.random.split(rng)
            mean, log_std, actor_state = _actor_forward(last_obs, actor_state)
            action, _ = _squashed_gaussian_sample_and_log_prob(mean, log_std, rng_act)

            next_obs, reward, absorbing, done, info, env_state = env.step(env_state, action, traj)

            replay_buffer = replay_buffer.add_batch(
                last_obs, next_obs, action, reward, done.astype(jnp.float32)
            )
            return actor_state, replay_buffer, env_state, next_obs, rng

        # ------------------------------------------------------------------
        # Sub-function: single gradient update step
        # ------------------------------------------------------------------
        def _single_gradient_update(carry, unused):
            actor_st, critic_st, tgt_p, tgt_rs, la_st, buf, rng_up = carry

            rng_up, rng_sample = jax.random.split(rng_up)
            obs_b, nobs_b, act_b, rew_b, done_b = buf.sample(rng_sample, batch_size)

            alpha = jnp.exp(la_st.params["log_alpha"])

            # ---- critic loss ----
            rng_up, rng_next = jax.random.split(rng_up)
            next_mean, next_log_std, _ = _actor_forward(nobs_b, actor_st)
            next_action, next_log_pi = _squashed_gaussian_sample_and_log_prob(
                next_mean, next_log_std, rng_next
            )

            q1_next, q2_next, new_tgt_rs = _target_critic_forward(nobs_b, next_action, tgt_p, tgt_rs)
            q_next = jnp.minimum(q1_next, q2_next) - alpha * next_log_pi
            q_target = rew_b + gamma * (1.0 - done_b) * q_next
            q_target = jax.lax.stop_gradient(q_target)

            def _critic_loss_fn(params):
                (q1, q2), updates = critic_net.apply(
                    {"params": params, "run_stats": critic_st.run_stats},
                    obs_b, act_b, mutable=["run_stats"],
                )
                loss = jnp.mean((q1 - q_target) ** 2) + jnp.mean((q2 - q_target) ** 2)
                return loss, updates["run_stats"]

            (critic_loss, new_critic_rs), critic_grads = jax.value_and_grad(
                _critic_loss_fn, has_aux=True
            )(critic_st.params)
            critic_st = critic_st.apply_gradients(grads=critic_grads)
            critic_st = critic_st.replace(run_stats=new_critic_rs)

            # ---- actor loss ----
            rng_up, rng_actor = jax.random.split(rng_up)

            def _actor_loss_fn(params):
                (mean_a, log_std_a), updates = actor_net.apply(
                    {"params": params, "run_stats": actor_st.run_stats},
                    obs_b, mutable=["run_stats"],
                )
                sampled_action, log_pi = _squashed_gaussian_sample_and_log_prob(
                    mean_a, log_std_a, rng_actor
                )
                (q1_a, q2_a), _ = critic_net.apply(
                    {"params": critic_st.params, "run_stats": critic_st.run_stats},
                    obs_b, sampled_action, mutable=["run_stats"],
                )
                q_a = jnp.minimum(q1_a, q2_a)
                loss = jnp.mean(alpha * log_pi - q_a)
                return loss, (log_pi, updates["run_stats"])

            (actor_loss, (log_pi_a, new_actor_rs)), actor_grads = jax.value_and_grad(
                _actor_loss_fn, has_aux=True
            )(actor_st.params)
            actor_st = actor_st.apply_gradients(grads=actor_grads)
            actor_st = actor_st.replace(run_stats=new_actor_rs)

            # ---- alpha loss ----
            def _alpha_loss_fn(params):
                log_a = params["log_alpha"]
                return -jnp.mean(log_a * (log_pi_a + target_entropy)), log_a

            (alpha_loss, log_a_val), alpha_grads = jax.value_and_grad(
                _alpha_loss_fn, has_aux=True
            )(la_st.params)
            la_st = jax.lax.cond(
                learnable_alpha,
                lambda s: s.apply_gradients(grads=alpha_grads),
                lambda s: s,
                la_st,
            )

            # ---- soft update target critic ----
            new_tgt_p = jax.tree.map(
                lambda tp, cp: tau * cp + (1.0 - tau) * tp,
                tgt_p, critic_st.params,
            )

            new_carry = (actor_st, critic_st, new_tgt_p, new_tgt_rs,
                         la_st, buf, rng_up)
            losses = (critic_loss, actor_loss, alpha_loss)
            return new_carry, losses

        # ------------------------------------------------------------------
        # Sub-function: N gradient updates via scan
        # ------------------------------------------------------------------
        def _do_updates(actor_st, critic_st, tgt_p, tgt_rs, la_st, buf, rng_up):
            carry = (actor_st, critic_st, tgt_p, tgt_rs, la_st, buf, rng_up)
            carry, losses = jax.lax.scan(
                _single_gradient_update, carry, None, gradient_steps
            )
            actor_st, critic_st, tgt_p, tgt_rs, la_st, _, _ = carry
            # Average losses across gradient steps
            critic_loss = jnp.mean(losses[0])
            actor_loss = jnp.mean(losses[1])
            alpha_loss = jnp.mean(losses[2])
            return (actor_st, critic_st, tgt_p, tgt_rs, la_st,
                    critic_loss, actor_loss, alpha_loss)

        def _skip_updates(actor_st, critic_st, tgt_p, tgt_rs, la_st, buf, rng_up):
            return (actor_st, critic_st, tgt_p, tgt_rs, la_st,
                    jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))

        # ------------------------------------------------------------------
        # Update step (one env step + conditional gradient updates)
        # ------------------------------------------------------------------
        def _update_step(runner_state, unused):
            actor_state, critic_state, tgt_params, tgt_run_stats, \
                log_alpha_state, replay_buffer, env_state, last_obs, rng = runner_state

            # ---- collect one transition from each parallel env ----
            actor_state, replay_buffer, env_state, next_obs, rng = \
                _collect_transition(actor_state, replay_buffer, env_state, last_obs, rng)

            # ---- gradient updates (only when buffer has enough samples) ----
            rng, rng_update = jax.random.split(rng)
            result = jax.lax.cond(
                replay_buffer.size >= learning_starts,
                lambda args: _do_updates(*args),
                lambda args: _skip_updates(*args),
                (actor_state, critic_state, tgt_params, tgt_run_stats,
                 log_alpha_state, replay_buffer, rng_update),
            )
            (actor_state, critic_state, tgt_params, tgt_run_stats,
             log_alpha_state, critic_loss, actor_loss, alpha_loss) = result

            # ---- metrics ----
            log_env_state = env_state.find(LogEnvState)
            logged_metrics = log_env_state.metrics

            alpha_val = jnp.exp(log_alpha_state.params["log_alpha"])
            metric = SACSummaryMetrics(
                mean_episode_return=jnp.sum(
                    jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)
                ) / jnp.maximum(jnp.sum(logged_metrics.done), 1),
                mean_episode_length=jnp.sum(
                    jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)
                ) / jnp.maximum(jnp.sum(logged_metrics.done), 1),
                max_timestep=jnp.max(logged_metrics.timestep * exp.num_envs),
                mean_critic_loss=critic_loss,
                mean_actor_loss=actor_loss,
                mean_alpha_loss=alpha_loss,
                mean_alpha=alpha_val,
                buffer_size=replay_buffer.size,
            )

            runner_state = (actor_state, critic_state, tgt_params, tgt_run_stats,
                            log_alpha_state, replay_buffer, env_state, next_obs, rng)
            return runner_state, metric

        # ------------------------------------------------------------------
        # Main scan over environment steps
        # ------------------------------------------------------------------
        rng, _rng = jax.random.split(rng)
        runner_state = (
            actor_state, critic_state, target_critic_params, target_critic_run_stats,
            log_alpha_state, replay_buffer, env_state, last_obs, _rng,
        )
        runner_state, training_metrics = jax.lax.scan(
            _update_step, runner_state, None, exp.num_updates
        )

        (actor_state, critic_state, tgt_params, tgt_run_stats,
         log_alpha_state, replay_buffer, env_state, last_obs, _) = runner_state

        agent_state_out = cls._agent_state(
            actor_state=actor_state,
            critic_state=critic_state,
            target_critic_params=tgt_params,
            target_critic_run_stats=tgt_run_stats,
            log_alpha_state=log_alpha_state,
            replay_buffer=replay_buffer,
            env_state=env_state,
            last_obs=last_obs,
        )

        return {
            "agent_state": agent_state_out,
            "training_metrics": training_metrics,
            "validation_metrics": ValidationSummary(),
        }

    # ------------------------------------------------------------------
    # Play policy
    # ------------------------------------------------------------------

    @classmethod
    def play_policy(cls, env,
                    agent_conf: SACAgentConf,
                    agent_state: SACAgentState,
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

    # ------------------------------------------------------------------
    # Env wrapping
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_env(env, config):
        env = LogWrapper(env)
        env = VecEnv(env)
        if getattr(config, 'normalize_env', True):
            env = NormalizeVecReward(env, float(config.gamma))
        return env
