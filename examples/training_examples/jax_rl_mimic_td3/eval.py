import os
import argparse

from loco_mujoco import TaskFactory
from loco_mujoco.algorithms.experimental import SACJax

from omegaconf import OmegaConf

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')

# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with SACJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

# Use the path from command line arguments
path = args.path
agent_conf, agent_state = SACJax.load_agent(path)
config = agent_conf.config

# get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# create env
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = False
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic
env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

# Determine which evaluation environment to run
if args.use_mujoco:
    # run eval mujoco
    SACJax.play_policy(env, agent_conf, agent_state, deterministic=False, n_steps=10000, n_envs=1,
                       record=True, use_mujoco=True, traj=traj)
else:
    # run eval mjx
    SACJax.play_policy(env, agent_conf, agent_state, deterministic=False, n_steps=10000, n_envs=1, record=True,
                       traj=traj)
