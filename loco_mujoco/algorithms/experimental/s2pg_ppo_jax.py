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
from loco_mujoco.algorithms.common.dataclasses import Transition
from loco_mujoco.algorithms.common.networks import FullyConnectedNet, RunningMeanStd, get_activation_fn
from loco_mujoco.algorithms.ppo_jax import PPOJax, PPOSummaryMetrics
from loco_mujoco.core.wrappers import LogWrapper, NStepWrapper, LogEnvState, VecEnv, NormalizeVecReward
from loco_mujoco.environments.base import TrajState
from loco_mujoco.core.wrappers.mjx import Metrics
from loco_mujoco.utils import MetricsHandler, ValidationSummary


# ---------------------------------------------------------------------------
# Network and policy — kept local to the experimental module
# ---------------------------------------------------------------------------

class IndependentJointDist:
    """
    Wraps two independent distributions (action and next hidden state) and exposes
    the standard log_prob / sample / entropy interface used by the training loop.

    Because the two distributions are independent:
        log p(a, z') = log p_a(a) + log p_z(z')
        entropy(a, z') = entropy(a) + entropy(z')
    """

    def __init__(self, dist_a: distrax.Distribution, dist_z: distrax.Distribution, action_dim: int):
        self.dist_a = dist_a
        self.dist_z = dist_z
        self.action_dim = action_dim

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        a = x[..., :self.action_dim]
        z = x[..., self.action_dim:]
        return self.dist_a.log_prob(a) + self.dist_z.log_prob(z)

    def sample(self, seed: jnp.ndarray) -> jnp.ndarray:
        seed_a, seed_z = jax.random.split(seed)
        a = self.dist_a.sample(seed=seed_a)
        z = self.dist_z.sample(seed=seed_z)
        return jnp.concatenate([a, z], axis=-1)

    def entropy(self) -> jnp.ndarray:
        return self.dist_a.entropy() + self.dist_z.entropy()


class ActorCriticS2PG(nn.Module):
    """
    Network for S2PG-PPO (Stochastic Stateful Policy Gradient).

    Accepts ``(obs, z)`` as separate inputs and outputs an IndependentJointDist
    over ``(action, next_hidden_state)`` plus a scalar value V(s, z).
    """
    action_dim: int
    hidden_state_dim: int
    activation: str = "tanh"
    init_std_a: float = 1.0
    init_std_z: float = 0.1
    learnable_std: bool = True
    hidden_layer_dims: Sequence[int] = (512, 256)

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)

    @nn.compact
    def __call__(self, obs, z):
        x = RunningMeanStd()(jnp.concatenate([obs, z], axis=-1))

        actor_out = FullyConnectedNet(
            self.hidden_layer_dims,
            self.action_dim + self.hidden_state_dim,
            self.activation, None, False, False
        )(x)
        mean_a = actor_out[..., :self.action_dim]
        mean_z = actor_out[..., self.action_dim:]

        log_std_a = self.param("log_std_a", nn.initializers.constant(jnp.log(self.init_std_a)),
                               (self.action_dim,))
        log_std_z = self.param("log_std_z", nn.initializers.constant(jnp.log(self.init_std_z)),
                               (self.hidden_state_dim,))
        if not self.learnable_std:
            log_std_a = jax.lax.stop_gradient(log_std_a)
            log_std_z = jax.lax.stop_gradient(log_std_z)

        pi = IndependentJointDist(
            distrax.MultivariateNormalDiag(mean_a, jnp.exp(log_std_a)),
            distrax.MultivariateNormalDiag(mean_z, jnp.exp(log_std_z)),
            self.action_dim,
        )
        critic = FullyConnectedNet(self.hidden_layer_dims, 1, self.activation, None, False, False)(x)
        return pi, jnp.squeeze(critic, axis=-1)


class S2PGPolicy:
    """
    Policy for S2PG-PPO.  Accepts obs and hidden_state as separate arguments —
    no manual concatenation needed at call sites.
    """

    def __init__(self, network: nn.Module, action_dim: int):
        self.network = network
        self.action_dim = action_dim

    def get_env_action_and_next_hidden(self, obs, hidden_state, train_state, rng):
        pi, value, train_state = self._forward_pass(obs, hidden_state, train_state)
        joint_action = pi.sample(seed=rng)
        log_prob = pi.log_prob(joint_action)
        env_action = joint_action[..., :self.action_dim]
        next_hidden = joint_action[..., self.action_dim:]
        return env_action, next_hidden, log_prob, value, joint_action, train_state

    def get_dist_and_value(self, obs, hidden_state, train_state):
        return self._forward_pass(obs, hidden_state, train_state)

    def _forward_pass(self, obs, hidden_state, train_state):
        y, updates = self.network.apply(
            {"params": train_state.params, "run_stats": train_state.run_stats},
            obs, hidden_state,
            mutable=["run_stats"],
        )
        pi, value = y
        return pi, value, train_state.replace(run_stats=updates["run_stats"])


class S2PGTransition(NamedTuple):
    """Transition tuple for S2PG — stores raw obs and hidden_state separately."""
    done: jnp.ndarray
    absorbing: jnp.ndarray
    action: jnp.ndarray         # joint action concat([a, z'])
    hidden_state: jnp.ndarray   # current z at this step (for policy re-run during update)
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray            # raw env obs (not augmented)
    info: jnp.ndarray
    traj_state: TrajState
    metrics: Metrics


@dataclass(frozen=True)
class S2PGAgentConf(AgentConfBase):
    config: DictConfig
    network: ActorCriticS2PG
    tx: Any

    def serialize(self):
        conf_dict = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
        serialized_network = flax.serialization.to_state_dict(self.network)
        return {"config": conf_dict, "network": serialized_network}

    @classmethod
    def from_dict(cls, d):
        config = OmegaConf.create(d["config"])
        tx = S2PGPPOJax._get_optimizer(config)
        return cls(config=config,
                   network=flax.serialization.from_state_dict(ActorCriticS2PG, d["network"]),
                   tx=tx)


@struct.dataclass
class S2PGAgentState(AgentStateBase):
    train_state: TrainState
    env_state: Any = None          # carried across chunks to preserve normalization stats
    last_obs: Any = None           # last env observation from previous chunk
    hidden_state: Any = None       # last hidden state z from previous chunk

    def serialize(self):
        serialized_train_state = flax.serialization.to_state_dict(self.train_state)
        return {"train_state": serialized_train_state}  # env_state/last_obs/hidden_state not saved

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
        train_state = train_state.replace(opt_state=opt_state)
        return cls(train_state=train_state)


class S2PGPPOJax(PPOJax):
    """
    S2PG-PPO: Proximal Policy Optimization with the Stochastic Stateful Policy Gradient.

    The policy jointly models the environment action *and* the next internal hidden
    state as a single distribution:

        (a, z') ~ π_θ(· | s, z)

    The importance-sampling ratio in the PPO surrogate therefore covers the full
    joint:

        ρ = π_θ(a, z' | s, z) / q(a, z' | s, z)

    and the value function conditions on both the environment observation and the
    current hidden state:  V(s, z).

    This avoids backpropagation through time (BPTT) while still allowing the policy
    to maintain a stateful internal representation.

    Reference:
        "Time-Efficient Reinforcement Learning with Stochastic Stateful Policies"
        Firas Al-Hafez et al., arXiv 2311.04082
    """

    _agent_conf = S2PGAgentConf
    _agent_state = S2PGAgentState

    @classmethod
    def init_agent_conf(cls, env, config):
        with open_dict(config.experiment):
            config.experiment.num_updates = (
                config.experiment.total_timesteps // config.experiment.num_steps // config.experiment.num_envs)
            config.experiment.minibatch_size = (
                config.experiment.num_envs * config.experiment.num_steps // config.experiment.num_minibatches)
            config.experiment.validation_interval = config.experiment.num_updates // config.experiment.validation.num
            config.experiment.validation.num = int(
                config.experiment.num_updates // config.experiment.validation_interval)
            config.experiment.action_dim = env.info.action_space.shape[0]

        hidden_layers = config.experiment.hidden_layers \
            if isinstance(config.experiment.hidden_layers, (list, ListConfig)) \
            else ast.literal_eval(config.experiment.hidden_layers)

        hidden_state_dim = config.experiment.hidden_state_dim
        init_std_z = getattr(config.experiment, 'init_std_z', 0.1)

        network = ActorCriticS2PG(
            action_dim=env.info.action_space.shape[0],
            hidden_state_dim=hidden_state_dim,
            activation=config.experiment.activation,
            init_std_a=config.experiment.init_std,
            init_std_z=init_std_z,
            learnable_std=config.experiment.learnable_std,
            hidden_layer_dims=hidden_layers,
        )
        tx = cls._get_optimizer(config)
        return cls._agent_conf(config, network, tx)

    @classmethod
    def init_agent_state(cls, env, agent_conf: S2PGAgentConf, rng) -> S2PGAgentState:
        config, network, tx = agent_conf.config.experiment, agent_conf.network, agent_conf.tx
        wrapped_env = cls._wrap_env(env, config)
        obs_dim = wrapped_env.info.observation_space.shape[0]
        hidden_state_dim = config.hidden_state_dim
        rng, _rng = jax.random.split(rng)
        network_params = network.init(_rng, jnp.zeros(obs_dim), jnp.zeros(hidden_state_dim))
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params["params"],
            run_stats=network_params["run_stats"],
            tx=tx,
        )
        return cls._agent_state(train_state=train_state)

    @classmethod
    def _train_fn(cls, rng, env,
                  agent_conf: S2PGAgentConf,
                  agent_state: S2PGAgentState = None,
                  traj=None,
                  mh: MetricsHandler = None):

        config, network, tx = agent_conf.config.experiment, agent_conf.network, agent_conf.tx
        action_dim = config.action_dim
        hidden_state_dim = config.hidden_state_dim

        env = cls._wrap_env(env, config)
        policy = S2PGPolicy(network, action_dim)

        # ---- init train state ----
        if agent_state is not None:
            train_state = agent_state.train_state.replace(apply_fn=network.apply)
        else:
            rng, _rng1 = jax.random.split(rng)
            obs_dim = env.info.observation_space.shape[0]
            init_obs = jnp.zeros(obs_dim)
            init_z = jnp.zeros(hidden_state_dim)
            network_params = network.init(_rng1, init_obs, init_z)
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

        # ---- init hidden state ----
        if agent_state is not None and agent_state.hidden_state is not None:
            hidden_state = agent_state.hidden_state
        else:
            hidden_state = jnp.zeros((config.num_envs, hidden_state_dim))

        train_state_buffer = TrainStateBuffer.create(train_state, config.validation.num)

        # ---- training loop ----
        def _update_step(runner_state, unused):

            # -- trajectory collection --
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, hidden_state, train_state_buffer, rng = runner_state

                # sample joint (env_action, z') from π(·|s, z) — obs and z passed separately
                rng, _rng = jax.random.split(rng)
                env_action, next_hidden, log_prob, value, joint_action, train_state = \
                    policy.get_env_action_and_next_hidden(last_obs, hidden_state, train_state, _rng)

                # step env
                obsv, reward, absorbing, done, info, env_state = env.step(env_state, env_action, traj)

                # reset hidden state on episode termination
                next_hidden = next_hidden * (1 - done)[..., None]

                log_env_state = env_state.find(LogEnvState)
                logged_metrics = log_env_state.metrics

                transition = S2PGTransition(
                    done, absorbing, joint_action, hidden_state, value, reward, log_prob,
                    last_obs, info, env_state.additional_carry.traj_state, logged_metrics
                )
                runner_state = (train_state, env_state, obsv, next_hidden, train_state_buffer, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config.num_steps)

            # -- advantage estimation --
            train_state, env_state, last_obs, hidden_state, train_state_buffer, rng = runner_state
            _, last_val, _ = policy.get_dist_and_value(last_obs, hidden_state, train_state)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, absorbing, value, reward = (
                        transition.done, transition.absorbing,
                        transition.value, transition.reward,
                    )
                    # V(s', z') is naturally next_value since value[t+1] = V(s_{t+1}, z_{t+1})
                    delta = reward + config.gamma * next_value * (1 - absorbing) - value
                    gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae
                    return (gae, value), gae

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
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # obs and hidden_state passed separately; action is joint (a, z')
                        pi, value, _ = policy.get_dist_and_value(
                            traj_batch.obs, traj_batch.hidden_state, train_state.replace(params=params)
                        )
                        # joint log prob: log π_θ(a, z' | s, z)
                        log_prob = pi.log_prob(traj_batch.action)

                        # value loss
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_loss = 0.5 * jnp.maximum(
                            jnp.square(value - targets),
                            jnp.square(value_pred_clipped - targets),
                        ).mean()

                        # PPO actor loss with joint importance ratio
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * gae,
                        ).mean()

                        entropy = pi.entropy().mean()
                        total_loss = (
                            loss_actor + config.vf_coef * value_loss - config.ent_coef * entropy
                        )
                        old_approx_kl = (traj_batch.log_prob - log_prob).mean()
                        clip_fraction = jnp.mean(jnp.abs(ratio - 1.0) > config.clip_eps)
                        return total_loss, (value_loss, loss_actor, entropy, old_approx_kl, clip_fraction)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config.minibatch_size * config.num_minibatches
                assert (
                    batch_size == config.num_steps * config.num_envs
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
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
                            lambda lr: lr,
                            lr,
                        ),
                        current_lr,
                    )
                    new_opt_state = train_state.opt_state._replace(
                        inner_state=train_state.opt_state.inner_state._replace(
                            hyperparams={'learning_rate': new_lr}
                        )
                    )
                    train_state = train_state.replace(opt_state=new_opt_state)

                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
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
                eval_hidden = jnp.zeros((config.validation.num_envs, hidden_state_dim))

                def _eval_env(eval_runner_state, unused):
                    train_state, env_state, last_obs, eval_hidden, rng = eval_runner_state

                    rng, _rng = jax.random.split(rng)
                    env_action, next_hidden, _, _, _, train_state = \
                        policy.get_env_action_and_next_hidden(last_obs, eval_hidden, train_state, _rng)

                    obsv, reward, absorbing, done, info, env_state = env.step(env_state, env_action, traj)
                    next_hidden = next_hidden * (1 - done)[..., None]

                    log_env_state = env_state.find(LogEnvState)
                    logged_metrics = log_env_state.metrics

                    transition = MetricHandlerTransition(env_state, logged_metrics)
                    eval_runner_state = (train_state, env_state, obsv, next_hidden, rng)
                    return eval_runner_state, transition

                rng = runner_state[-1]
                reset_rng = jax.random.split(rng, config.validation.num_envs)
                obsv, eval_env_state = env.reset(reset_rng, traj)
                eval_runner_state = (train_state, eval_env_state, obsv, eval_hidden, rng)

                _, traj_batch_eval = jax.lax.scan(
                    _eval_env, eval_runner_state, None, config.validation.num_steps
                )
                validation_metrics = mh(traj_batch_eval.env_state)
                return validation_metrics

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

            runner_state = (train_state, env_state, last_obs, hidden_state, train_state_buffer, rng)
            return runner_state, (metric, validation_metrics)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, hidden_state, train_state_buffer, _rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config.num_updates)

        agent_state = cls._agent_state(
            train_state=runner_state[0],
            env_state=runner_state[1],
            last_obs=runner_state[2],
            hidden_state=runner_state[3],
        )
        return {
            "agent_state": agent_state,
            "training_metrics": metrics[0],
            "validation_metrics": metrics[1],
        }

    @classmethod
    def play_policy(cls, env,
                    agent_conf: S2PGAgentConf,
                    agent_state: S2PGAgentState,
                    n_envs: int, n_steps=None, render=True,
                    record=False, rng=None, deterministic=False,
                    use_mujoco=False, wrap_env=True,
                    train_state_seed=None, traj=None):

        if use_mujoco:
            assert n_envs == 1, "Only one mujoco env can be run at a time."

        config = agent_conf.config.experiment
        action_dim = config.action_dim
        hidden_state_dim = config.hidden_state_dim
        _policy = S2PGPolicy(agent_conf.network, action_dim)

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

        hidden_state = jnp.zeros((n_envs, hidden_state_dim))

        def sample_actions(ts, obs, hidden, _rng):
            env_a, next_h, _, _, _, ts = _policy.get_env_action_and_next_hidden(obs, hidden, ts, _rng)
            return env_a, next_h, ts

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
            action, next_hidden, train_state = plcy_call(train_state, obs, hidden_state, _rng)
            action = jnp.atleast_2d(action)

            if use_mujoco:
                obs, reward, absorbing, done, info = env.step(action)
                hidden_state = next_hidden * (1 - jnp.array(done, dtype=jnp.float32))[..., None]
            else:
                obs, reward, absorbing, done, info, env_state = env.step(env_state, action, traj)
                hidden_state = next_hidden * (1 - done)[..., None]

            if use_mujoco:
                env.render(record=True)
            else:
                env.mjx_render(env_state, record=record)

            if use_mujoco and done:
                obs = env.reset()
                hidden_state = jnp.zeros((n_envs, hidden_state_dim))

        env.stop()
