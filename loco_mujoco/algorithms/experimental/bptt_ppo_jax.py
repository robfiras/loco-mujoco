import ast
import warnings
from omegaconf import open_dict
from dataclasses import dataclass
from typing import Any, NamedTuple, Tuple, Sequence
from omegaconf import DictConfig, OmegaConf, ListConfig

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import flax
import optax
import distrax

from loco_mujoco.algorithms import (JaxRLAlgorithmBase, AgentConfBase, AgentStateBase,
                                    TrainState, TrainStateBuffer, MetricHandlerTransition)
from loco_mujoco.algorithms.common.networks import FullyConnectedNet, RunningMeanStd, get_activation_fn
from loco_mujoco.algorithms.ppo_jax import PPOJax, PPOSummaryMetrics
from loco_mujoco.core.wrappers import LogWrapper, NStepWrapper, LogEnvState, VecEnv
from loco_mujoco.environments.base import TrajState
from loco_mujoco.core.wrappers.mjx import Metrics
from loco_mujoco.utils import MetricsHandler, ValidationSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stored_hidden_dim(hidden_state_dim: int, rnn_type: str) -> int:
    """Total stored hidden-state size: 2x for LSTM (h and c), 1x otherwise."""
    return 2 * hidden_state_dim if rnn_type == "lstm" else hidden_state_dim


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ActorCriticBPTT(nn.Module):
    """
    Network for BPTT-PPO.

    Actor
    -----
    *rnn_type = "gru" or "lstm"* (default):
      1. Embed obs and prev_action separately.
      2. Feed concat([obs_embed, prev_action_embed]) into a GRU / LSTM cell
         whose initial carry is the current hidden state ``h``.
      3. The RNN carry output is the deterministic new hidden state h'.
      4. A post-RNN obs re-embedding + RNN output → action mean.

    *rnn_type = "vanilla"*:
      Embed obs and prev_action separately via Dense(n_features), concat with
      hidden_state → MLP → (action_mean, new_hidden).

    The actor returns a distrax.MultivariateNormalDiag over actions only.
    The new hidden state is returned deterministically (no distribution).

    Critic
    ------
    MLP over concat([obs, h, prev_action]) → scalar V(s, h, a_prev).

    Inputs
    ------
    obs:          (..., obs_dim)
    hidden_state: (..., stored_hidden_dim)   # for LSTM: concat(h, c)
    prev_action:  (..., action_dim)

    Returns
    -------
    (pi, value, new_hidden)
    """

    action_dim: int
    hidden_state_dim: int           # actual RNN hidden size (per vector; LSTM stores 2x)
    rnn_type: str = "gru"           # "gru" | "lstm" | "vanilla"
    n_features: int = 256           # pre / post-RNN embedding width
    activation: str = "tanh"
    init_std_a: float = 1.0
    learnable_std: bool = True
    hidden_layer_dims: Sequence[int] = (512, 256)   # critic MLP + "vanilla" actor MLP
    ema_coef: float = 1.0           # exponential smoothing for hidden state transition

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)

    @nn.compact
    def __call__(self, obs, hidden_state, prev_action):
        obs_norm = RunningMeanStd()(obs)

        stored_z_dim = _stored_hidden_dim(self.hidden_state_dim, self.rnn_type)

        if self.rnn_type == "vanilla":
            # ---- MLP actor ----
            # Embed obs and prev_action separately, then combine with hidden state
            obs_embed = self.activation_fn(nn.Dense(self.n_features)(obs_norm))
            prev_a_embed = self.activation_fn(nn.Dense(self.n_features)(prev_action))
            actor_in = jnp.concatenate([obs_embed, prev_a_embed, hidden_state], axis=-1)
            actor_out = FullyConnectedNet(
                self.hidden_layer_dims,
                self.action_dim + stored_z_dim,
                self.activation, None, False, False
            )(actor_in)
            action_mean = actor_out[..., :self.action_dim]
            new_hidden = actor_out[..., self.action_dim:]

        else:
            # ---- RNN actor ----
            # Pre-RNN embeddings
            obs_embed = self.activation_fn(nn.Dense(self.n_features)(obs_norm))
            prev_a_embed = self.activation_fn(nn.Dense(self.n_features)(prev_action))
            rnn_input = jnp.concatenate([obs_embed, prev_a_embed], axis=-1)

            if self.rnn_type == "gru":
                new_carry, rnn_out = nn.GRUCell(self.hidden_state_dim)(hidden_state, rnn_input)
                new_hidden = new_carry   # shape (..., hidden_state_dim)

            elif self.rnn_type == "lstm":
                h = hidden_state[..., :self.hidden_state_dim]
                c = hidden_state[..., self.hidden_state_dim:]
                (new_h, new_c), rnn_out = nn.LSTMCell(self.hidden_state_dim)((h, c), rnn_input)
                new_hidden = jnp.concatenate([new_h, new_c], axis=-1)
                rnn_out = new_h   # use h for downstream heads

            else:
                raise ValueError(f"Unknown rnn_type: {self.rnn_type!r}")

            # Post-RNN obs re-embedding → action mean
            obs_post = self.activation_fn(nn.Dense(self.n_features)(obs_norm))
            action_mean = nn.Dense(
                self.action_dim,
                kernel_init=nn.initializers.orthogonal(0.01),
                bias_init=nn.initializers.constant(0.0),
            )(jnp.concatenate([obs_post, rnn_out], axis=-1))

        # Exponential smoothing on the hidden state transition:
        #   new_h = (1 - ema_coef) * h_t + ema_coef * f_theta(...)
        # ema_coef=1.0 recovers the unsmoothed transition (default).
        if self.ema_coef < 1.0:
            new_hidden = (1.0 - self.ema_coef) * hidden_state + self.ema_coef * new_hidden

        # Action distribution (independent Gaussian)
        log_std_a = self.param("log_std_a",
                               nn.initializers.constant(jnp.log(self.init_std_a)),
                               (self.action_dim,))
        if not self.learnable_std:
            log_std_a = jax.lax.stop_gradient(log_std_a)

        pi = distrax.MultivariateNormalDiag(action_mean, jnp.exp(log_std_a))

        # Critic — MLP over (obs, h, prev_action)
        critic_in = jnp.concatenate([obs_norm, hidden_state, prev_action], axis=-1)
        critic = FullyConnectedNet(
            self.hidden_layer_dims, 1, self.activation, None, False, False
        )(critic_in)

        return pi, jnp.squeeze(critic, axis=-1), new_hidden


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class BPTTPolicy:
    """
    Policy wrapper for BPTT-PPO.

    All methods take obs, hidden_state, and prev_action as separate arguments.
    The hidden state is propagated deterministically (no stochastic transition).
    """

    def __init__(self, network: nn.Module, action_dim: int):
        self.network = network
        self.action_dim = action_dim

    def get_action(
        self, obs, hidden_state, prev_action, train_state, rng
    ):
        """Sample action and return new hidden state.

        Returns:
            (action, new_hidden, log_prob, value, updated_train_state)
        """
        pi, value, new_hidden, train_state = self._forward_pass(obs, hidden_state, prev_action, train_state)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        return action, new_hidden, log_prob, value, train_state

    def get_dist_and_value(self, obs, hidden_state, prev_action, train_state):
        return self._forward_pass(obs, hidden_state, prev_action, train_state)

    def _forward_pass(self, obs, hidden_state, prev_action, train_state):
        (pi, value, new_hidden), updates = self.network.apply(
            {"params": train_state.params, "run_stats": train_state.run_stats},
            obs, hidden_state, prev_action,
            mutable=["run_stats"],
        )
        return pi, value, new_hidden, train_state.replace(run_stats=updates["run_stats"])


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------

class BPTTTransition(NamedTuple):
    """Transition for BPTT-PPO — no per-step hidden_state needed in the batch."""
    done: jnp.ndarray
    absorbing: jnp.ndarray
    action: jnp.ndarray         # env action a_t
    prev_action: jnp.ndarray    # a_{t-1} (zeros at episode start)
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray            # raw env obs (not augmented)
    info: jnp.ndarray
    traj_state: TrajState
    metrics: Metrics


# ---------------------------------------------------------------------------
# Agent conf / state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BPTTAgentConf(AgentConfBase):
    config: DictConfig
    network: ActorCriticBPTT
    tx: Any

    def serialize(self):
        conf_dict = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
        serialized_network = flax.serialization.to_state_dict(self.network)
        return {"config": conf_dict, "network": serialized_network}

    @classmethod
    def from_dict(cls, d):
        config = OmegaConf.create(d["config"])
        tx = BPTTPPOJax._get_optimizer(config)
        return cls(config=config,
                   network=flax.serialization.from_state_dict(ActorCriticBPTT, d["network"]),
                   tx=tx)


@struct.dataclass
class BPTTAgentState(AgentStateBase):
    train_state: TrainState
    env_state: Any = None
    last_obs: Any = None
    hidden_state: Any = None    # h carried across chunks
    prev_action: Any = None     # a_{t-1} carried across chunks

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

class BPTTPPOJax(PPOJax):
    """
    BPTT-PPO: Proximal Policy Optimization with Backpropagation Through Time.

    Key properties
    --------------
    * The policy uses a GRU/LSTM/vanilla RNN to maintain a deterministic hidden
      state; the IS ratio is only over actions (standard PPO).
    * Gradients flow through time via jax.lax.scan over the sequence during
      the update step.
    * Minibatches are over environments (not individual timesteps) so sequences
      stay intact during the gradient computation.
    * h_init (hidden state at start of each rollout) is saved and used to seed
      the BPTT replay scan.

    Supported rnn_type values: "gru" (default), "lstm", "vanilla" (MLP fallback).
    """

    _agent_conf = BPTTAgentConf
    _agent_state = BPTTAgentState

    @classmethod
    def init_agent_conf(cls, env, config):
        with open_dict(config.experiment):
            config.experiment.num_updates = (
                config.experiment.total_timesteps // config.experiment.num_steps // config.experiment.num_envs)
            # For BPTT, minibatch_size is over environments not flattened steps
            config.experiment.minibatch_size = (
                config.experiment.num_envs // config.experiment.num_minibatches)
            config.experiment.validation_interval = config.experiment.num_updates // config.experiment.validation.num
            config.experiment.validation.num = int(
                config.experiment.num_updates // config.experiment.validation_interval)
            config.experiment.action_dim = env.info.action_space.shape[0]

        hidden_layers = config.experiment.hidden_layers \
            if isinstance(config.experiment.hidden_layers, (list, ListConfig)) \
            else ast.literal_eval(config.experiment.hidden_layers)

        hidden_state_dim = config.experiment.hidden_state_dim
        rnn_type = getattr(config.experiment, 'rnn_type', 'gru')
        n_features = getattr(config.experiment, 'n_features', 256)
        ema_coef = getattr(config.experiment, 'ema_coef', 1.0)

        network = ActorCriticBPTT(
            action_dim=env.info.action_space.shape[0],
            hidden_state_dim=hidden_state_dim,
            rnn_type=rnn_type,
            n_features=n_features,
            activation=config.experiment.activation,
            init_std_a=config.experiment.init_std,
            learnable_std=config.experiment.learnable_std,
            hidden_layer_dims=hidden_layers,
            ema_coef=ema_coef,
        )
        tx = cls._get_optimizer(config)
        return cls._agent_conf(config, network, tx)

    @classmethod
    def init_agent_state(cls, env, agent_conf: BPTTAgentConf, rng) -> BPTTAgentState:
        config, network, tx = agent_conf.config.experiment, agent_conf.network, agent_conf.tx
        wrapped_env = cls._wrap_env(env, config)
        obs_dim = wrapped_env.info.observation_space.shape[0]
        action_dim = config.action_dim
        stored_z_dim = _stored_hidden_dim(config.hidden_state_dim, network.rnn_type)
        rng, _rng = jax.random.split(rng)
        network_params = network.init(
            _rng,
            jnp.zeros((1, obs_dim)),
            jnp.zeros((1, stored_z_dim)),
            jnp.zeros((1, action_dim)),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params["params"],
            run_stats=network_params["run_stats"],
            tx=tx,
        )
        return cls._agent_state(train_state=train_state)

    @classmethod
    def _train_fn(cls, rng, env,
                  agent_conf: BPTTAgentConf,
                  agent_state: BPTTAgentState = None,
                  traj=None,
                  mh: MetricsHandler = None):

        config, network, tx = agent_conf.config.experiment, agent_conf.network, agent_conf.tx
        action_dim = config.action_dim
        hidden_state_dim = config.hidden_state_dim
        rnn_type = network.rnn_type
        stored_z_dim = _stored_hidden_dim(hidden_state_dim, rnn_type)

        assert config.num_envs % config.num_minibatches == 0, (
            f"num_envs ({config.num_envs}) must be divisible by num_minibatches ({config.num_minibatches})"
        )
        mb_size = config.num_envs // config.num_minibatches

        env = cls._wrap_env(env, config)
        policy = BPTTPolicy(network, action_dim)

        # ---- init train state ----
        if agent_state is not None:
            train_state = agent_state.train_state.replace(apply_fn=network.apply)
        else:
            rng, _rng1 = jax.random.split(rng)
            obs_dim = env.info.observation_space.shape[0]
            network_params = network.init(
                _rng1,
                jnp.zeros(obs_dim),
                jnp.zeros(stored_z_dim),
                jnp.zeros(action_dim),
            )
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params["params"],
                run_stats=network_params["run_stats"],
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

        # ---- init hidden state and prev_action ----
        if agent_state is not None and agent_state.hidden_state is not None:
            hidden_state = agent_state.hidden_state
        else:
            hidden_state = jnp.zeros((config.num_envs, stored_z_dim))

        if agent_state is not None and agent_state.prev_action is not None:
            prev_action = agent_state.prev_action
        else:
            prev_action = jnp.zeros((config.num_envs, action_dim))

        train_state_buffer = TrainStateBuffer.create(train_state, config.validation.num)

        # ---- training loop ----
        def _update_step(runner_state, unused):

            # -- save h_init before rollout for BPTT replay --
            train_state, env_state, last_obs, hidden_state, prev_action, train_state_buffer, rng = runner_state
            h_init = hidden_state  # (num_envs, stored_z_dim)

            # -- trajectory collection --
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, hidden_state, prev_action, train_state_buffer, rng = runner_state

                rng, _rng = jax.random.split(rng)
                action, next_hidden, log_prob, value, train_state = \
                    policy.get_action(last_obs, hidden_state, prev_action, train_state, _rng)

                obsv, reward, absorbing, done, info, env_state = env.step(env_state, action, traj)

                # reset hidden state and prev_action on episode termination
                next_hidden = next_hidden * (1 - done)[..., None]
                next_prev_action = action * (1 - done)[..., None]

                log_env_state = env_state.find(LogEnvState)
                logged_metrics = log_env_state.metrics

                transition = BPTTTransition(
                    done, absorbing, action, prev_action,
                    value, reward, log_prob, last_obs, info,
                    env_state.additional_carry.traj_state, logged_metrics
                )
                runner_state = (
                    train_state, env_state, obsv, next_hidden, next_prev_action,
                    train_state_buffer, rng
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config.num_steps)

            # -- advantage estimation --
            train_state, env_state, last_obs, hidden_state, prev_action, train_state_buffer, rng = runner_state
            _, last_val, _, _ = policy.get_dist_and_value(last_obs, hidden_state, prev_action, train_state)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    delta = transition.reward + config.gamma * next_value * (1 - transition.absorbing) - transition.value
                    gae = delta + config.gamma * config.gae_lambda * (1 - transition.done) * gae
                    return (gae, transition.value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # -- policy / value update --
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_mb, adv_mb, tgt_mb, h_init_mb = batch_info
                    # traj_mb.*: (mb_size, num_steps, ...)
                    # adv_mb, tgt_mb: (mb_size, num_steps)
                    # h_init_mb: (mb_size, stored_z_dim)

                    # Capture run_stats outside the grad (fixed during grad computation)
                    run_stats = train_state.run_stats

                    def _loss_fn(params):
                        # Transpose sequences to (num_steps, mb_size, ...) for scan
                        obs_seq = jnp.transpose(traj_mb.obs, (1, 0) + tuple(range(2, traj_mb.obs.ndim)))
                        done_seq = jnp.transpose(traj_mb.done, (1, 0) + tuple(range(2, traj_mb.done.ndim)))
                        action_seq = jnp.transpose(traj_mb.action, (1, 0) + tuple(range(2, traj_mb.action.ndim)))
                        prev_action_seq = jnp.transpose(traj_mb.prev_action, (1, 0) + tuple(range(2, traj_mb.prev_action.ndim)))

                        def _rnn_step(h, data):
                            obs_t, done_t, action_t, prev_action_t = data
                            (pi, value_t, new_h), _ = network.apply(
                                {"params": params, "run_stats": run_stats},
                                obs_t, h, prev_action_t,
                                mutable=["run_stats"],
                            )
                            log_prob_t = pi.log_prob(action_t)
                            entropy_t = pi.entropy()
                            # Reset hidden state on episode termination (after step)
                            new_h = new_h * (1 - done_t)[..., None]
                            return new_h, (log_prob_t, value_t, entropy_t)

                        _, (log_probs, values, entropies) = jax.lax.scan(
                            _rnn_step, h_init_mb, (obs_seq, done_seq, action_seq, prev_action_seq)
                        )
                        # log_probs, values, entropies: (num_steps, mb_size)

                        # Transpose back to (mb_size, num_steps) then flatten for losses
                        log_probs = jnp.transpose(log_probs, (1, 0))       # (mb_size, num_steps)
                        values = jnp.transpose(values, (1, 0))             # (mb_size, num_steps)
                        entropies = jnp.transpose(entropies, (1, 0))       # (mb_size, num_steps)

                        # old stored quantities are already (mb_size, num_steps)
                        old_log_probs = traj_mb.log_prob                    # (mb_size, num_steps)
                        old_values = traj_mb.value                          # (mb_size, num_steps)

                        # Flatten to (mb_size * num_steps,) for loss computation
                        log_probs_flat = log_probs.reshape(-1)
                        values_flat = values.reshape(-1)
                        entropies_flat = entropies.reshape(-1)
                        old_log_probs_flat = old_log_probs.reshape(-1)
                        old_values_flat = old_values.reshape(-1)
                        adv_flat = adv_mb.reshape(-1)
                        tgt_flat = tgt_mb.reshape(-1)

                        # Value loss (clipped)
                        value_pred_clipped = old_values_flat + (
                            values_flat - old_values_flat
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_loss = 0.5 * jnp.maximum(
                            jnp.square(values_flat - tgt_flat),
                            jnp.square(value_pred_clipped - tgt_flat),
                        ).mean()

                        # PPO actor loss (IS ratio over actions only)
                        ratio = jnp.exp(log_probs_flat - old_log_probs_flat)
                        gae_norm = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
                        loss_actor = -jnp.minimum(
                            ratio * gae_norm,
                            jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * gae_norm,
                        ).mean()

                        entropy = entropies_flat.mean()
                        total_loss = loss_actor + config.vf_coef * value_loss - config.ent_coef * entropy
                        old_approx_kl = (old_log_probs_flat - log_probs_flat).mean()
                        clip_fraction = jnp.mean(jnp.abs(ratio - 1.0) > config.clip_eps)
                        return total_loss, (value_loss, loss_actor, entropy, old_approx_kl, clip_fraction)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, h_init, rng = update_state
                rng, _rng = jax.random.split(rng)

                # Permute over environments (not individual timesteps)
                permutation = jax.random.permutation(_rng, config.num_envs)

                # traj_batch.*: (num_steps, num_envs, ...)
                # Transpose to (num_envs, num_steps, ...), shuffle, reshape to (num_minibatches, mb_size, num_steps, ...)
                def _prepare_traj(x):
                    # x: (num_steps, num_envs, ...)
                    # → (num_envs, num_steps, ...) → shuffle → (num_minibatches, mb_size, num_steps, ...)
                    x_T = jnp.swapaxes(x, 0, 1)  # (num_envs, num_steps, ...)
                    x_shuffled = jnp.take(x_T, permutation, axis=0)
                    return x_shuffled.reshape((config.num_minibatches, mb_size) + x_shuffled.shape[1:])

                def _prepare_flat(x):
                    # x: (num_steps, num_envs) or (num_envs, num_steps) — advantages/targets are (num_steps, num_envs)
                    x_T = jnp.swapaxes(x, 0, 1)  # (num_envs, num_steps)
                    x_shuffled = jnp.take(x_T, permutation, axis=0)
                    return x_shuffled.reshape((config.num_minibatches, mb_size) + x_shuffled.shape[1:])

                def _prepare_h_init(x):
                    # x: (num_envs, stored_z_dim)
                    x_shuffled = jnp.take(x, permutation, axis=0)
                    return x_shuffled.reshape((config.num_minibatches, mb_size) + x_shuffled.shape[1:])

                minibatch_traj = jax.tree.map(_prepare_traj, traj_batch)
                minibatch_adv = _prepare_flat(advantages)   # (num_minibatches, mb_size, num_steps)
                minibatch_tgt = _prepare_flat(targets)      # (num_minibatches, mb_size, num_steps)
                minibatch_h_init = _prepare_h_init(h_init)  # (num_minibatches, mb_size, stored_z_dim)

                minibatches = (minibatch_traj, minibatch_adv, minibatch_tgt, minibatch_h_init)
                train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)

                # adaptive KL learning rate
                desired_kl = getattr(config, 'desired_kl', None)
                if desired_kl is not None:
                    mean_kl = jnp.mean(total_loss[1][3])
                    current_lr = train_state.opt_state.inner_state.hyperparams['learning_rate']
                    new_lr = jax.lax.cond(
                        mean_kl > desired_kl * 2.0,
                        lambda lr: jnp.maximum(1e-5, lr / 1.5),
                        lambda lr: jax.lax.cond(
                            (mean_kl < desired_kl / 2.0) & (mean_kl > 0.0),
                            lambda lr: jnp.minimum(1e-2, lr * 1.5),
                            lambda lr: lr, lr,
                        ),
                        current_lr,
                    )
                    new_opt_state = train_state.opt_state._replace(
                        inner_state=train_state.opt_state.inner_state._replace(
                            hyperparams={'learning_rate': new_lr}
                        )
                    )
                    train_state = train_state.replace(opt_state=new_opt_state)

                update_state = (train_state, traj_batch, advantages, targets, h_init, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, h_init, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # -- metrics --
            counter = ((train_state.step + 1) // config.num_minibatches) // config.update_epochs
            logged_metrics = traj_batch.metrics

            mean_value_loss = jnp.mean(loss_info[1][0])
            mean_actor_loss = jnp.mean(loss_info[1][1])
            mean_entropy = jnp.mean(loss_info[1][2])
            mean_approx_kl = jnp.mean(loss_info[1][3])
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

            # -- validation --
            def _evaluation_step():
                eval_hidden = jnp.zeros((config.validation.num_envs, stored_z_dim))
                eval_prev_action = jnp.zeros((config.validation.num_envs, action_dim))

                def _eval_env(eval_runner_state, unused):
                    train_state, env_state, last_obs, eval_hidden, eval_prev_action, rng = eval_runner_state

                    rng, _rng = jax.random.split(rng)
                    action, next_hidden, _, _, train_state = \
                        policy.get_action(last_obs, eval_hidden, eval_prev_action, train_state, _rng)

                    obsv, reward, absorbing, done, info, env_state = env.step(env_state, action, traj)
                    next_hidden = next_hidden * (1 - done)[..., None]
                    next_prev_action = action * (1 - done)[..., None]

                    log_env_state = env_state.find(LogEnvState)
                    transition = MetricHandlerTransition(env_state, log_env_state.metrics)
                    eval_runner_state = (train_state, env_state, obsv, next_hidden, next_prev_action, rng)
                    return eval_runner_state, transition

                rng = runner_state[-1]
                reset_rng = jax.random.split(rng, config.validation.num_envs)
                obsv, eval_env_state = env.reset(reset_rng, traj)
                eval_runner_state = (train_state, eval_env_state, obsv, eval_hidden, eval_prev_action, rng)

                _, traj_batch_eval = jax.lax.scan(
                    _eval_env, eval_runner_state, None, config.validation.num_steps
                )
                return mh(traj_batch_eval.env_state)

            if mh is None:
                validation_metrics = ValidationSummary()
            else:
                validation_metrics = jax.lax.cond(
                    counter % config.validation_interval == 0,
                    _evaluation_step, mh.get_zero_container
                )

            if config.debug:
                def callback(metrics):
                    return_values = metrics.returned_episode_returns[metrics.done]
                    timesteps = metrics.timestep[metrics.done] * config.num_envs
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                jax.debug.callback(callback, env_state.metrics)

            train_state_buffer = jax.lax.cond(
                counter % config.validation_interval == 0,
                lambda x, y: TrainStateBuffer.add(x, y),
                lambda x, y: x,
                train_state_buffer, train_state,
            )

            runner_state = (
                train_state, env_state, last_obs, hidden_state, prev_action,
                train_state_buffer, rng
            )
            return runner_state, (metric, validation_metrics)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, hidden_state, prev_action, train_state_buffer, _rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config.num_updates)

        agent_state = cls._agent_state(
            train_state=runner_state[0],
            env_state=runner_state[1],
            last_obs=runner_state[2],
            hidden_state=runner_state[3],
            prev_action=runner_state[4],
        )
        return {
            "agent_state": agent_state,
            "training_metrics": metrics[0],
            "validation_metrics": metrics[1],
        }

    @classmethod
    def play_policy(cls, env,
                    agent_conf: BPTTAgentConf,
                    agent_state: BPTTAgentState,
                    n_envs: int, n_steps=None, render=True,
                    record=False, rng=None, deterministic=False,
                    use_mujoco=False, wrap_env=True,
                    train_state_seed=None, traj=None):

        if use_mujoco:
            assert n_envs == 1, "Only one mujoco env can be run at a time."

        config = agent_conf.config.experiment
        action_dim = config.action_dim
        stored_z_dim = _stored_hidden_dim(config.hidden_state_dim, agent_conf.network.rnn_type)
        _policy = BPTTPolicy(agent_conf.network, action_dim)

        train_state = agent_state.train_state
        if deterministic:
            train_state.params["log_std_a"] = np.ones_like(train_state.params["log_std_a"]) * -np.inf

        if config.n_seeds > 1:
            assert train_state_seed is not None, (
                "Loaded train state has multiple seeds. Please specify train_state_seed for replay."
            )
            train_state = jax.tree.map(lambda x: x[train_state_seed], train_state)

        if not render and n_steps is None and not record:
            warnings.warn("No rendering, no record, no n_steps specified. This will run forever with no effect.")

        if wrap_env and not use_mujoco:
            env = cls._wrap_env(env, config)

        if rng is None:
            rng = jax.random.key(0)

        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]

        hidden_state = jnp.zeros((n_envs, stored_z_dim))
        prev_action = jnp.zeros((n_envs, action_dim))

        def sample_actions(ts, obs, hidden, prev_a, _rng):
            action, next_h, _, _, ts = _policy.get_action(obs, hidden, prev_a, ts, _rng)
            return action, next_h, ts

        plcy_call = jax.jit(sample_actions)

        if use_mujoco:
            obs = env.reset()
            env_state = None
        else:
            obs, env_state = env.reset(env_keys, traj)

        if n_steps is None:
            n_steps = np.iinfo(np.int32).max

        for i in range(n_steps):
            rng, _rng = jax.random.split(rng)
            action, next_hidden, train_state = plcy_call(train_state, obs, hidden_state, prev_action, _rng)
            action = jnp.atleast_2d(action)

            if use_mujoco:
                obs, reward, absorbing, done, info = env.step(action)
                done_f = jnp.array(done, dtype=jnp.float32)
                hidden_state = next_hidden * (1 - done_f)[..., None]
                prev_action = action * (1 - done_f)[..., None]
            else:
                obs, reward, absorbing, done, info, env_state = env.step(env_state, action, traj)
                hidden_state = next_hidden * (1 - done)[..., None]
                prev_action = action * (1 - done)[..., None]

            if use_mujoco:
                env.render(record=True)
            else:
                env.mjx_render(env_state, record=record)

            if use_mujoco and done:
                obs = env.reset()
                hidden_state = jnp.zeros((n_envs, stored_z_dim))
                prev_action = jnp.zeros((n_envs, action_dim))

        env.stop()
