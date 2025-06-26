import os
import sys
import jax
import wandb
import jax.numpy as jnp
import traceback

import numpy as np

# Hydra:  key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line 
import hydra 
# from hydra.core.hydra_config import HydraConfig

from dataclasses import fields
from loco_mujoco.utils.metrics import QuantityContainer


from omegaconf import DictConfig, OmegaConf #OmegaConf is a YAML based hierarchical configuration system, with support for merging configurations from multiple sources

from loco_mujoco import TaskFactory
from loco_mujoco.core.wrappers import VecEnv

import time


# Set MUJOCO_GL to egl
os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering, which is more compatible with headless environments


@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):
    try: 

        # get task factory
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

        # randomization_type = config.randomization_config["randomization_type"]
        # # Convert to plain dict to allow adding new keys
        # randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)
        
        # # add prosthesis side to randomization params if it exists in config.experiment.env_params
        # if "prosthesis_side" in config.experiment.env_params:
        #     randomization_params["prosthesis_side"] = config.experiment.env_params["prosthesis_side"]

        # create env
        env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
        # domain_randomization_type=randomization_type, 
                        #    domain_randomization_params=randomization_params,
                        #    **config.experiment.env_params, **config.experiment.task_factory.params)


        # create keys
        key = jax.random.key(0)
        n_envs = config.experiment.num_envs 
        keys = jax.random.split(key, n_envs + 1)
        key, env_keys = keys[0], keys[1:]

        # env.th.to_jax()
        # env = VecEnv(env)
        jit_step  = jax.jit(jax.vmap(env.mjx_step))  #env.step)
        jit_reset  = jax.jit(jax.vmap(env.mjx_reset)) #env.reset)
        rng_sample_uni_action = jax.jit(jax.vmap(env.sample_action_space))

        state = env.reset(env_keys)
        i = 0 
        while i < 100000:

            # if i % 500==0  or i==0:
            keys = jax.random.split(key, n_envs + 1)
            key, action_keys = keys[0], keys[1:]
            action = rng_sample_uni_action(action_keys)
            state = jit_step(state, action) 
            env.mjx_render(state)
            i+=1
    
        print("Environment step successful in mjx")

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise

if __name__ == "__main__":
    experiment()
