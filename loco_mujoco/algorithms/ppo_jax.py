import ast
from omegaconf import open_dict
import warnings
from dataclasses import dataclass
from typing import Any
from omegaconf import DictConfig, OmegaConf, ListConfig

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import flax
import optax

from loco_mujoco.algorithms import (JaxRLAlgorithmBase, AgentConfBase, AgentStateBase, ActorCritic,
                                    Transition, TrainState, TrainStateBuffer, MetricHandlerTransition,
                                    RewardNormStats, update_reward_norm)
from loco_mujoco.algorithms.common.policies import PPOPolicy
from loco_mujoco.core.wrappers import LogWrapper, NStepWrapper, LogEnvState, VecEnv, SummaryMetrics

@struct.dataclass
class PPOSummaryMetrics(SummaryMetrics):
    mean_value_loss: float = 0.0
    mean_actor_loss: float = 0.0
    mean_entropy: float = 0.0
    mean_approx_kl: float = 0.0
    mean_clip_fraction: float = 0.0
    learning_rate: float = 0.0
from loco_mujoco.utils import MetricsHandler, ValidationSummary


@dataclass(frozen=True)
class PPOAgentConf(AgentConfBase):
    config: DictConfig
    network: ActorCritic
    tx: Any

    def serialize(self):
        """
        Serialize the agent configuration and network configuration.

        Returns:
            Serialized agent configuration as a dictionary.

        """
        conf_dict = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
        serialized_network = flax.serialization.to_state_dict(self.network)
        return {"config": conf_dict, "network": serialized_network}

    @classmethod
    def from_dict(cls, d):
        config = OmegaConf.create(d["config"])
        tx = PPOJax._get_optimizer(config)
        return cls(config=config,
                   network=flax.serialization.from_state_dict(ActorCritic, d["network"]),
                   tx=tx)


@struct.dataclass
class PPOAgentState(AgentStateBase):
    train_state: TrainState
    env_state: Any = None              # carried across chunks so chunk boundary is seamless
    last_obs: Any = None               # last observation from previous chunk
    reward_norm_stats: Any = None      # RewardNormStats; survives env_state reset and is serialized

    def serialize(self):
        serialized_train_state = flax.serialization.to_state_dict(self.train_state)
        out = {"train_state": serialized_train_state}  # env_state/last_obs not saved to disk
        if self.reward_norm_stats is not None:
            out["reward_norm_stats"] = flax.serialization.to_state_dict(self.reward_norm_stats)
        return out

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
        reward_norm_stats = None
        if "reward_norm_stats" in d and d["reward_norm_stats"] is not None:
            reward_norm_stats = flax.serialization.from_state_dict(
                RewardNormStats.create(1), d["reward_norm_stats"]
            )
        return cls(train_state=train_state, reward_norm_stats=reward_norm_stats)


class PPOJax(JaxRLAlgorithmBase):

    _agent_conf = PPOAgentConf
    _agent_state = PPOAgentState

    @classmethod
    def init_agent_conf(cls, env, config):

        with (open_dict(config.experiment)):
            # Data parallelism. `num_envs` in the config is the GLOBAL env count;
            # each of the n_devices replicas runs num_envs // n_devices of them.
            # We rewrite config.num_envs to the PER-DEVICE value here, because
            # _train_fn runs inside pmap and every use of it there (reset splits,
            # reward-norm stats, the minibatch assert) is per-replica. The global
            # figure is kept as num_envs_total for step accounting and logging.
            n_devices = int(getattr(config.experiment, "n_devices", 1) or 1)
            num_envs_total = int(config.experiment.num_envs)
            if n_devices > 1:
                if num_envs_total % n_devices != 0:
                    raise ValueError(
                        f"num_envs ({num_envs_total}) must be divisible by "
                        f"n_devices ({n_devices}); pmap requires an equal split."
                    )
                config.experiment.num_envs = num_envs_total // n_devices
            config.experiment.num_envs_total = num_envs_total

            # num_updates is driven by the GLOBAL env count: n_devices replicas
            # each stepping num_envs_per_device envs collect num_envs_total
            # transitions per step, so the same total_timesteps is reached in
            # 1/n_devices of the wall-clock updates a single device would need.
            config.experiment.num_updates = (
                    config.experiment.total_timesteps // config.experiment.num_steps // num_envs_total)
            config.experiment.minibatch_size = (
                    config.experiment.num_envs * config.experiment.num_steps // config.experiment.num_minibatches)
            config.experiment.validation_interval = config.experiment.num_updates // config.experiment.validation.num
            config.experiment.validation.num = int(
                config.experiment.num_updates // config.experiment.validation_interval)

        # INIT NETWORK
        hidden_layers = config.experiment.hidden_layers \
            if isinstance(config.experiment.hidden_layers, (list, ListConfig)) \
            else ast.literal_eval(config.experiment.hidden_layers)
        if hasattr(config.experiment, "actor_obs_group") and config.experiment.actor_obs_group is not None:
            actor_obs_ind = env.obs_container.get_obs_ind_by_group(config.experiment.actor_obs_group)
        else:
            actor_obs_ind = jnp.arange(env.mdp_info.observation_space.shape[0])
        if hasattr(config.experiment, "critic_obs_group") and config.experiment.critic_obs_group is not None:
            critic_obs_ind = env.obs_container.get_obs_ind_by_group(config.experiment.critic_obs_group)
        else:
            critic_obs_ind = jnp.arange(env.mdp_info.observation_space.shape[0])
        if hasattr(config.experiment, "len_obs_history") and config.experiment.len_obs_history > 1:
            obs_len = env.info.observation_space.shape[0]
            actor_obs_ind = jnp.concatenate([actor_obs_ind + i*obs_len
                                             for i in range(config.experiment.len_obs_history)])
            critic_obs_ind = jnp.concatenate([critic_obs_ind + i*obs_len
                                              for i in range(config.experiment.len_obs_history)])
        network = ActorCritic(
            env.info.action_space.shape[0],
            activation=config.experiment.activation,
            init_std=config.experiment.init_std,
            learnable_std=config.experiment.learnable_std,
            hidden_layer_dims=hidden_layers,
            actor_obs_ind=actor_obs_ind,
            critic_obs_ind=critic_obs_ind
        )

        # set up optimizers
        tx = cls._get_optimizer(config)

        return cls._agent_conf(config, network, tx)

    @classmethod
    def init_agent_state(cls, env, agent_conf: PPOAgentConf, rng) -> PPOAgentState:
        """ Initializes and returns the PPO agent state (network params + optimizer). """
        config, network, tx = agent_conf.config.experiment, agent_conf.network, agent_conf.tx
        wrapped_env = cls._wrap_env(env, config)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(wrapped_env.info.observation_space.shape)
        network_params = network.init(_rng, init_x)
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params["params"],
            run_stats=network_params["run_stats"],
            tx=tx,
        )
        return cls._agent_state(train_state=train_state)

    @classmethod
    def _get_optimizer(cls, config):
        desired_kl = getattr(config.experiment, 'desired_kl', None)
        if config.experiment.anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(config.experiment.max_grad_norm),
                optax.adamw(weight_decay=config.experiment.weight_decay, eps=1e-5,
                            learning_rate=lambda count: cls._linear_lr_schedule(count, config.experiment.num_minibatches,
                                                                                config.experiment.update_epochs, config.lr,
                                                                                config.experiment.num_updates))
            )
        elif desired_kl is not None:
            tx = optax.inject_hyperparams(
                lambda learning_rate: optax.chain(
                    optax.clip_by_global_norm(config.experiment.max_grad_norm),
                    optax.adamw(learning_rate, weight_decay=config.experiment.weight_decay, eps=1e-5),
                )
            )(learning_rate=config.experiment.lr)
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.experiment.max_grad_norm),
                optax.adamw(config.experiment.lr, weight_decay=config.experiment.weight_decay, eps=1e-5),
            )

        tx = optax.apply_if_finite(tx, max_consecutive_errors=10000000)

        return tx

    @classmethod
    def _train_fn(cls, rng, env,
                  agent_conf: PPOAgentConf,
                  agent_state: PPOAgentState = None,
                  traj=None,
                  mh: MetricsHandler = None):

        # extract static agent info
        config, network, tx =\
            (agent_conf.config.experiment, agent_conf.network, agent_conf.tx)

        # Data-parallel axis name, set only when this function is wrapped in
        # jax.pmap by a distributed driver. None => single-device, and every
        # collective below is skipped, so the single-device path is unchanged.
        #
        # pmap rather than automatic NamedSharding sharding is deliberate: the
        # env step enters through a mujoco_warp FFI custom call, which does not
        # participate in XLA's sharding propagation. Under pmap each device runs
        # its own instance over its own env batch, which the custom call handles
        # correctly.
        pmap_axis = getattr(config, "pmap_axis_name", None)

        env = cls._wrap_env(env, config)
        policy = PPOPolicy(network)

        if agent_state is not None:
            # resume: preserve params, opt_state, and step — only re-attach apply_fn (not serializable)
            train_state = agent_state.train_state.replace(apply_fn=network.apply)
        else:
            rng, _rng1 = jax.random.split(rng)
            init_x = jnp.zeros(env.info.observation_space.shape)
            network_params = network.init(_rng1, init_x)
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params["params"],
                run_stats=network_params["run_stats"],
                tx=tx,
            )

        # INIT ENV — reuse carried env state if available for a seamless chunk boundary
        if agent_state is not None and agent_state.env_state is not None:
            env_state = agent_state.env_state
            obsv = agent_state.last_obs
        else:
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config.num_envs)
            obsv, env_state = env.reset(reset_rng, traj)

        # INIT REWARD NORM STATS — reuse across chunks so they survive env_state resets
        if config.normalize_env:
            if agent_state is not None and agent_state.reward_norm_stats is not None:
                rew_stats = agent_state.reward_norm_stats
            else:
                rew_stats = RewardNormStats.create(config.num_envs)
        else:
            rew_stats = None

        train_state_buffer = TrainStateBuffer.create(train_state, config.validation.num)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rew_stats, train_state_buffer, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                action, log_prob, value, train_state = policy.get_action_and_value(last_obs, train_state, _rng)

                # STEP ENV
                obsv, reward, absorbing, done, info, env_state = env.step(env_state, action, traj)

                # NORMALIZE REWARD (stats live on agent state, threaded through runner_state)
                if rew_stats is not None:
                    reward, rew_stats = update_reward_norm(rew_stats, reward, done, config.gamma)

                # GET METRICS
                log_env_state = env_state.find(LogEnvState)
                logged_metrics = log_env_state.metrics

                transition = Transition(
                    done, absorbing, action, value, reward, log_prob, last_obs, info, env_state.additional_carry.traj_state,
                    logged_metrics
                )
                runner_state = (train_state, env_state, obsv, rew_stats, train_state_buffer, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rew_stats, train_state_buffer, rng = runner_state
            _, last_val, _ = policy.get_dist_and_value(last_obs, train_state)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, absorbing, value, reward, obs = (
                        transition.done,
                        transition.absorbing,
                        transition.value,
                        transition.reward,
                        transition.obs
                    )

                    delta = reward + config.gamma * next_value * (1 - absorbing) - value
                    gae = (
                        delta
                        + config.gamma * config.gae_lambda * (1 - done) * gae
                    )
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

            # UPDATE ACTOR & CRITIC NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value, _ = policy.get_dist_and_value(traj_batch.obs,
                                                                  train_state.replace(params=params))
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE PPO ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config.clip_eps,
                                    1.0 + config.clip_eps,
                                )
                                * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config.vf_coef * value_loss
                            - config.ent_coef * entropy
                        )
                        old_approx_kl = (traj_batch.log_prob - log_prob).mean()
                        clip_fraction = jnp.mean(jnp.abs(ratio - 1.0) > config.clip_eps)
                        return total_loss, (value_loss, loss_actor, entropy, old_approx_kl, clip_fraction)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    if pmap_axis is not None:
                        # Average gradients across devices before applying, so
                        # every replica steps identically and the params stay in
                        # sync. This is what makes the run equivalent to one
                        # process with n_devices x num_envs environments.
                        grads = jax.lax.pmean(grads, axis_name=pmap_axis)
                        # Losses are logged from device 0, so reduce them too or
                        # the reported numbers describe 1/n of the batch.
                        total_loss = jax.lax.pmean(total_loss, axis_name=pmap_axis)
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
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config.num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )

                # Adaptive KL learning rate (RSL-style)
                desired_kl = getattr(config, 'desired_kl', None)
                if desired_kl is not None:
                    # Under pmap this is already cross-device averaged, because
                    # _update_minbatch pmeans total_loss. That matters: an
                    # unsynced mean_kl would let each device pick a different
                    # learning rate and the replicas would silently diverge.
                    mean_kl = jnp.mean(total_loss[1][3])  # avg old_approx_kl across minibatches
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

            counter = ((train_state.step + 1) // config.num_minibatches) // config.update_epochs

            logged_metrics = traj_batch.metrics

            # aggregate loss metrics across epochs and minibatches
            mean_value_loss = jnp.mean(loss_info[1][0])
            mean_actor_loss = jnp.mean(loss_info[1][1])
            mean_entropy = jnp.mean(loss_info[1][2])
            mean_approx_kl = jnp.mean(loss_info[1][3])
            mean_clip_fraction = jnp.mean(loss_info[1][4])

            # extract current learning rate
            desired_kl = getattr(config, 'desired_kl', None)
            if config.anneal_lr:
                current_lr = cls._linear_lr_schedule(train_state.step, config.num_minibatches,
                                                     config.update_epochs, config.lr, config.num_updates)
            elif desired_kl is not None:
                current_lr = train_state.opt_state.inner_state.hyperparams['learning_rate']
            else:
                current_lr = jnp.array(config.lr)

            metric = PPOSummaryMetrics(
                mean_episode_return=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)) / jnp.sum(logged_metrics.done),
                mean_episode_length=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)) / jnp.sum(logged_metrics.done),
                max_timestep=jnp.max(logged_metrics.timestep * config.num_envs),
                mean_value_loss=mean_value_loss,
                mean_actor_loss=mean_actor_loss,
                mean_entropy=mean_entropy,
                mean_approx_kl=mean_approx_kl,
                mean_clip_fraction=mean_clip_fraction,
                learning_rate=current_lr,
            )

            def _evaluation_step():

                def _eval_env(runner_state, unused):
                    train_state, env_state, last_obs, train_state_buffer, rng = runner_state

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    y, updates = train_state.apply_fn({'params': train_state.params,
                                                       'run_stats': train_state.run_stats},
                                                      last_obs, mutable=["run_stats"])
                    pi, value = y
                    train_state = train_state.replace(run_stats=updates['run_stats'])  # update stats
                    action = pi.sample(seed=_rng)

                    # STEP ENV (eval uses raw reward — normalization is training-only)
                    obsv, reward, absorbing, done, info, env_state = env.step(env_state, action, traj)

                    # GET METRICS
                    log_env_state = env_state.find(LogEnvState)
                    logged_metrics = log_env_state.metrics

                    transition = MetricHandlerTransition(env_state, logged_metrics)

                    runner_state = (train_state, env_state, obsv, train_state_buffer, rng)
                    return runner_state, transition

                rng = runner_state[-1]
                reset_rng = jax.random.split(rng, config.validation.num_envs)
                obsv, env_state = env.reset(reset_rng, traj)
                runner_state_eval = (train_state, env_state, obsv, train_state_buffer, rng)

                # do evaluation runs
                _, traj_batch_eval = jax.lax.scan(
                    _eval_env, runner_state_eval, None, config.validation.num_steps
                )

                env_states = traj_batch_eval.env_state

                validation_metrics = mh(env_states)

                return validation_metrics

            if mh is None:
                validation_metrics = ValidationSummary()
            else:
                validation_metrics = jax.lax.cond(counter % config.validation_interval == 0, _evaluation_step,
                                                   mh.get_zero_container)

            if config.debug:
                def callback(metrics):
                    return_values = metrics.returned_episode_returns[metrics.done]
                    timesteps = metrics.timestep[metrics.done] * config.num_envs

                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")

                jax.debug.callback(callback, env_state.metrics)

            # add train state to buffer if needed
            train_state_buffer = jax.lax.cond(counter % config.validation_interval == 0,
                                              lambda x, y: TrainStateBuffer.add(x, y),
                                              lambda x, y: x, train_state_buffer, train_state)

            runner_state = (train_state, env_state, last_obs, rew_stats, train_state_buffer, rng)
            return runner_state, (metric, validation_metrics)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, rew_stats, train_state_buffer, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config.num_updates
        )

        agent_state = cls._agent_state(train_state=runner_state[0],
                                       env_state=runner_state[1],
                                       last_obs=runner_state[2],
                                       reward_norm_stats=runner_state[3])

        return {"agent_state": agent_state,
                "training_metrics": metrics[0],
                "validation_metrics": metrics[1]}

    @classmethod
    def _eval_fn(cls, rng, env,
                 agent_conf: PPOAgentConf,
                 agent_state: PPOAgentState,
                 traj=None,
                 mh: MetricsHandler = None):
        """Standalone eval rollout, same signature as _train_fn.

        Resets a fresh set of eval envs (`config.validation.num_envs`) and
        scans the policy for `config.validation.num_steps` steps. Returns
        episode-return/length summary plus the `mh` validation summary if a
        MetricsHandler is provided. Does not touch the agent's training
        state — run_stats updates during eval live only in the scan carry.
        """
        config = agent_conf.config.experiment
        network = agent_conf.network
        env = cls._wrap_env(env, config)
        policy = PPOPolicy(network)

        train_state = agent_state.train_state.replace(apply_fn=network.apply)

        def _eval_step(runner_state, unused):
            train_state, env_state, last_obs, rng = runner_state
            rng, _rng = jax.random.split(rng)
            y, updates = train_state.apply_fn(
                {"params": train_state.params, "run_stats": train_state.run_stats},
                last_obs, mutable=["run_stats"],
            )
            pi, _ = y
            train_state = train_state.replace(run_stats=updates["run_stats"])
            action = pi.sample(seed=_rng)

            obsv, reward, absorbing, done, info, env_state = env.step(
                env_state, action, traj
            )
            logged = env_state.find(LogEnvState).metrics
            transition = MetricHandlerTransition(env_state, logged)
            return (train_state, env_state, obsv, rng), transition

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.validation.num_envs)
        obsv, env_state = env.reset(reset_rng, traj)

        runner_state = (train_state, env_state, obsv, rng)
        _, traj_batch = jax.lax.scan(
            _eval_step, runner_state, None, config.validation.num_steps
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
            max_timestep=jnp.max(logged.timestep * config.validation.num_envs),
        )

        if mh is None:
            validation_metrics = ValidationSummary()
        else:
            validation_metrics = mh(traj_batch.env_state)

        return {"eval_summary": summary, "validation_metrics": validation_metrics}

    @classmethod
    def play_policy(cls, env,
                    agent_conf: PPOAgentConf,
                    agent_state: PPOAgentState,
                    n_envs: int, n_steps=None, render=True,
                    record=False, rng=None, deterministic=False,
                    use_mujoco=False, wrap_env=True,
                    train_state_seed=None, traj=None):

        if use_mujoco and wrap_env:
            if hasattr(agent_conf.experiment, "len_obs_history"):
                assert agent_conf.experiment.len_obs_history == 1, "len_obs_history must be 1 for mujoco envs."
        if use_mujoco:
            assert n_envs == 1, "Only one mujoco env can be run at a time."

        _policy = PPOPolicy(agent_conf.network)

        def sample_actions(ts, obs, _rng):
            a, _, _, ts = _policy.get_action_and_value(obs, ts, _rng)
            return a, ts

        config = agent_conf.config.experiment
        train_state = agent_state.train_state

        if deterministic:
            train_state.params["log_std"] = np.ones_like(train_state.params["log_std"]) * -np.inf

        if config.n_seeds > 1:
            assert train_state_seed is not None, ("Loaded train state has multiple seeds. Please specify "
                                                  "train_state_seed for replay.")

            # take the seed queried for evaluation
            train_state = jax.tree.map(lambda x: x[train_state_seed], train_state)

        if not render and n_steps is None and not record:
            warnings.warn("No rendering, no record, no n_steps specified. This will run forever with no effect.")

        # create env
        if wrap_env and not use_mujoco:
            env = cls._wrap_env(env, config)

        if rng is None:
            rng = jax.random.key(0)

        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]

        plcy_call = jax.jit(sample_actions)

        # reset env
        if use_mujoco:
            obs = env.reset()
            env_state = None
        else:
            obs, env_state = env.reset(env_keys, traj)

        if n_steps is None:
            n_steps = np.iinfo(np.int32).max

        for i in range(n_steps):

            # SAMPLE ACTION
            rng, _rng = jax.random.split(rng)
            action, train_state = plcy_call(train_state, obs, _rng)
            action = jnp.atleast_2d(action)

            # STEP ENV
            if use_mujoco:
                obs, reward, absorbing, done, info = env.step(action)
            else:
                obs, reward, absorbing, done, info, env_state = env.step(env_state, action, traj)

            # RENDER
            if use_mujoco:
                env.render(record=True)
            else:
                env.mjx_render(env_state, record=record)

            # RESET MUJOCO ENV (MJX resets by itself)
            if use_mujoco:
                if done:
                    obs = env.reset()

        env.stop()

    @classmethod
    def play_policy_mujoco(cls, env,
                           agent_conf: PPOAgentConf,
                           agent_state: PPOAgentState,
                           n_steps=None, render=True,
                           record=False, rng=None, deterministic=False,
                           train_state_seed=None):

        cls.play_policy(env, agent_conf, agent_state, 1, n_steps, render, record, rng, deterministic,
                        True, False, train_state_seed)

    @staticmethod
    def _wrap_env(env, config):

        if "len_obs_history" in config and config.len_obs_history > 1:
            env = NStepWrapper(env, config.len_obs_history)
        env = LogWrapper(env)
        env = VecEnv(env)
        # reward normalization moved into _train_fn; stats live on the agent state
        return env
