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

# from loco_mujoco.algorithms import PPOJax
# from loco_mujoco.utils import MetricsHandler
# from loco_mujoco import ImitationFactory


# config_path = os.path.join(os.path.dirname(__file__), "conf.yaml")
# with open(config_path, "r") as f:
#     config = yaml.safe_load(f)


# print(config["env_randomization"])

# env_randomization = config["env_randomization"]


# env = MjxSkeletonMuscleProsthesis(prosthesis_side = config["experiment"]["env_params"]["prosthesis_side"], 
#                                   prosthesis_type = config["experiment"]["env_params"]["prosthesis_type"])
# spec = mujoco.MjSpec.from_file(env.get_default_xml_file_path())
# # spec = env.
# # xml_path = '/home/nadinebadie/loco-mujoco/loco_mujoco/models/skeleton/skeleton_muscle.xml'
# model = spec.compile() #mujoco.MjModel.from_xml_path(xml_path)
# data = mujoco.MjData(model)

# print("Model loaded successfully.")
# prosthesis_randomizer = TranstibialProsthesisRandomizer(env_randomization)
# print("Prosthesis randomizer created successfully.")
# prosthesis_randomizer._sample_joint_stiffness(
#     config["experiment"]["env_params"]["prosthesis_side"],
#     env_randomization["randomization_joints"],
#     model,
#     data,
#     jnp,  # Pass the mujoco module as the backend argument
# )
# print("Joint damping sampled successfully.")




# Set MUJOCO_GL to egl
os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering, which is more compatible with headless environments


@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):
    try: 
        # can increase the speed by ~30% on some GPUs
        # os.environ['XLA_FLAGS'] = (
        #     '--xla_gpu_triton_gemm_any=True ')
        
        # Accessing the current sweep number
        result_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir


        # Extract date and time from the result directory path for wandb run name
        result_dir_parts = result_dir.split("/")
        result_dir_date, result_dir_time = result_dir_parts[-2], result_dir_parts[-1]
        formatted_result_dir = f"{result_dir_date}_{result_dir_time}_"


        # # setup wandb 
        # wandb.login()
        # config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        # run = wandb.init(project=config.wandb.project, name= formatted_result_dir + config.wandb.run_name, config=config_dict)


        # get task factory
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

        randomization_type = config.randomization_config["randomization_type"]
        # Convert to plain dict to allow adding new keys
        randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)
        
        # add prosthesis side to randomization params if it exists in config.experiment.env_params
        if "prosthesis_side" in config.experiment.env_params:
            randomization_params["prosthesis_side"] = config.experiment.env_params["prosthesis_side"]

        # create env
        env = factory.make(domain_randomization_type=randomization_type, 
                           domain_randomization_params=randomization_params,
                           **config.experiment.env_params, **config.experiment.task_factory.params)


        key = jax.random.key(0)
        train_state_seed = 0  # Take first seed 

        keys = jax.random.split(key, 2)
        key, env_keys = keys[0], keys[1:]

        # # Test in mujoco with backend np 
        # env.reset()
        # print("env.model.dof_damping", env.model.dof_damping)
        # print("env.model.dof_stiffness", env.model.jnt_stiffness)
        # print("env.model.body_pos", env.model.body_pos)
        # print("env.model.body_quat", env.model.body_quat)

        # action_dim = env.info.action_space.shape[0]
        # action = np.random.randn(action_dim)
        # nstate, reward, absorbing, done, info = env.step(action) # Environment only in mjx 
        # print("Environment step successful in mujoco")
        # print("env.model.dof_damping", env.model.dof_damping)
        # print("env.model.dof_stiffness", env.model.jnt_stiffness)
        # print("env.model.body_pos", env.model.body_pos)
        # print("env.model.body_quat", env.model.body_quat)

        # # Test mjx with backend jnp
        env.th.to_jax()
        env = VecEnv(env)
        # env.mjx_reset(env_keys)
        # jit_reset  = jax.jit(jax.vmap(env.mjx_reset))
        # jit_step = jax.jit(jax.vmap(env.mjx_step))
        jit_reset  = jax.jit(env.mjx_reset)
        jit_step = jax.jit(env.mjx_step)
        env_state = jit_reset(env_keys)
        # print("Environment reset successful in mjx")

        rng_sample_uni_action = jax.jit(jax.vmap(env.sample_action_space))
        # # actions all to 1 


        keys = jax.random.split(key, 1)
        key, action_keys = keys[0], keys[1:]
        action = rng_sample_uni_action(action_keys)

        # # dof_damping_init = env.model.dof_damping
        # # jnt_stiffness_init = env.model.jnt_stiffness
        # # body_pos_init = env.model.body_pos
        # # body_quat_init = env.model.body_quat

        # # jax.debug.print("env.model.dof_damping: {dof_damping}", dof_damping = env.model.dof_damping)
        # # jax.debug.print("env.model.jnt_stiffness: {jnt_stiffness}", jnt_stiffness = env.model.jnt_stiffness) 
        # # jax.debug.print("env.model.body_pos: {body_pos}", body_pos = env.model.body_pos)
        # # jax.debug.print("env.model.body_quat: {body_quat}", body_quat = env.model.body_quat)   

        env_state = jit_step(env_state, action)
        i = 0 
        while i < 100000:

            if i % 500==0  or i==0:
                keys = jax.random.split(key, 2)
                key, action_keys = keys[0], keys[1:]
                env_state = jit_reset(env_keys)    


            # parallel render
            # keys = jax.random.split(key, 1)
            # key, action_keys = keys[0], keys[1:]
            # env_state = jit_reset(env_keys)
            action = rng_sample_uni_action(action_keys)
            env_state = jit_step(env_state, action)


            env.mjx_render(env_state)
            i+=1

        # jax.debug.print("env.model.dof_damping: {dof_damping}", dof_damping = env.model.dof_damping)
        # jax.debug.print("init damping: {dof_damping_init}", dof_damping_init = dof_damping_init)
        # jax.debug.print("env.model.jnt_stiffness: {jnt_stiffness}", jnt_stiffness = env.model.jnt_stiffness) 
        # jax.debug.print("init stiffness: {jnt_stiffness_init}", jnt_stiffness_init = jnt_stiffness_init)
        # jax.debug.print("env.model.body_pos: {body_pos}", body_pos = env.model.body_pos)
        # jax.debug.print("init body pos: {body_pos_init}", body_pos_init = body_pos_init)
        # jax.debug.print("env.model.body_quat: {body_quat}", body_quat = env.model.body_quat) 
        # jax.debug.print("init body quat: {body_quat_init}", body_quat_init = body_quat_init)  
        
        print("Environment step successful in mjx")

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise

if __name__ == "__main__":
    experiment()
