import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 
# os.environ["JAX_PLATFORMS"] = "cpu"

import jax 
# jax.config.update('jax_platform_name', 'cpu')

import argparse


from loco_mujoco import TaskFactory
from custom_ppo_jax import CustomPPOJax

from omegaconf import OmegaConf

os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering, which is more compatible with headless environments
# os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_triton_gemm_any=True ')

# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with CustomPPOJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

# Use the path from command line arguments
path = args.path
agent_conf, agent_state = CustomPPOJax.load_agent(path)
config = agent_conf.config

# get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# create env
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = False
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic
env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)


# Determine which evaluation environment to run
if args.use_mujoco:
    # run eval mujoco
    CustomPPOJax.play_policy_mujoco(env, agent_conf, agent_state, deterministic=False, n_steps=10000, record=True,
                              train_state_seed=0)
else:
    # run eval mjx
    CustomPPOJax.play_policy(env, agent_conf, agent_state, deterministic=False, n_steps=10000, n_envs=1, record=True,
                       train_state_seed=0)

# save the video
if env.video_recorder is not None:
    video_path = os.path.join(os.path.dirname(path), "eval_video.mp4")
    env.video_recorder.save(video_path)
    print(f"Video saved to {video_path}")