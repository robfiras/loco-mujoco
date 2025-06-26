import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["JAX_PLATFORMS"] = "cpu"

import jax 
jax.config.update('jax_platform_name', 'cpu')

import argparse


from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.environments.humanoids.skeleton_prosthesis import MjxSkeletonMuscleProsthesis

from omegaconf import OmegaConf

os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering, which is more compatible with headless environments
# os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_triton_gemm_any=True ')

# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

# Use the path from command line arguments
path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config

# get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# create env
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = False
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic
env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

# energy_params = PPOJax.evaluate_policy(env, agent_conf, agent_state, deterministic=False, n_steps=1000, n_envs=1, record=True,
                    #    train_state_seed=0)

# print("Energy parameters:", energy_params)
# Determine which evaluation environment to run
if args.use_mujoco:
    # run eval mujoco
    PPOJax.play_policy_mujoco(env, agent_conf, agent_state, deterministic=False, n_steps=1000, record=True,
                              train_state_seed=0)
else:
    # run eval mjx
    PPOJax.play_policy(env, agent_conf, agent_state, deterministic=False, n_steps=1000, n_envs=1, record=True,
                       train_state_seed=0)


# Calculate energy efficiency of the loaded agent
# Placeholder for energy efficiency calculation or remove if not required
# energy_efficiency = env.get_energy_efficiency(agent_state)
# Calculate energy efficiency of the loaded agent



# env = MjxSkeletonMuscleProsthesis(prosthesis_side="left_side", prosthesis_type="transtibial")

