import ast
import jax.numpy as jnp

from omegaconf import open_dict, ListConfig
from loco_mujoco.algorithms.ppo_jax import PPOJax
from loco_mujoco.algorithms.common.networks import ActorCriticSkeletonMuscle

class CustomPPOJax(PPOJax):
    @classmethod
    def init_agent_conf(cls, env, config, output_activation=None):

        with (open_dict(config.experiment)):
            config.experiment.num_updates = (
                    config.experiment.total_timesteps // config.experiment.num_steps // config.experiment.num_envs)
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
            
        network = ActorCriticSkeletonMuscle(
            env.info.action_space.shape[0],
            number_upper_body_activation=config.experiment.number_upper_body_activation,
            custom_output_activation=config.experiment.custom_output_activation,
            activation=config.experiment.activation,
            init_std=config.experiment.init_std,
            learnable_std=config.experiment.learnable_std,
            hidden_layer_dims=hidden_layers,
            actor_obs_ind=actor_obs_ind,
            critic_obs_ind=critic_obs_ind,
        )

        # Set up optimizers
        tx = cls._get_optimizer(config)

        return cls._agent_conf(config, network, tx)