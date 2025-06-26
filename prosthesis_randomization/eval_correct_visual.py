import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["JAX_PLATFORMS"] = "cpu"

import jax 
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp

import pickle


import argparse

from loco_mujoco.core.wrappers import VecEnv


from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.environments.humanoids.skeleton_prosthesis import MjxSkeletonMuscleProsthesis


from loco_mujoco.algorithms.ppo_jax import PPOAgentConf, PPOAgentState
from omegaconf import OmegaConf

from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction

import mujoco
from datetime import datetime

import timeit 

import numpy as np

from loco_mujoco.core.visuals import MujocoViewer

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

randomization_type = config.randomization_config["randomization_type"]
# Convert to plain dict to allow adding new keys
randomization_params = OmegaConf.to_container(config.randomization_config["randomization_params"], resolve=True)

# add prosthesis side to randomization params if it exists in config.experiment.env_params
if "prosthesis_side" in config.experiment.env_params:
    randomization_params["prosthesis_side"] = config.experiment.env_params["prosthesis_side"]


# create env
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = False
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic
config.experiment.env_params["add_sensors"] = True
env = factory.make(domain_randomization_type=randomization_type, domain_randomization_params=randomization_params,
                   **config.experiment.env_params, **config.experiment.task_factory.params)
# env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

env.th.to_jax()
env = VecEnv(env)
jit_step  = jax.jit(jax.vmap(env.mjx_step_test))  #env.step)
# jit_step  = jax.jit(jax.vmap(env.mjx_step))  #env.step)
jit_reset  = jax.jit(jax.vmap(env.mjx_reset)) #env.reset)
# jit_step  = jax.jit(env.step)
# jit_reset  = jax.jit(env.reset)
model = env.get_model()


n_steps = 1000 #1000 #1000 #1000
n_envs = 1  # <--- Make sure this matches your training batch size
rng = jax.random.key(0)
train_state_seed = 0  # Take first seed 

keys = jax.random.split(rng, n_envs + 1)
rng, env_keys = keys[0], keys[1:]

def sample_actions_uncompiled(ts, obs, _rng): # Renamed for clarity
    y, updates = agent_conf.network.apply({'params': ts.params,
                                           'run_stats': ts.run_stats},
                                           obs, mutable=["run_stats"])
    ts = ts.replace(run_stats=updates['run_stats'])  # update stats
    pi, _ = y
    a = pi.sample(seed=_rng)
    return a, ts


# JIT compile the function
# sample_actions = jax.jit(sample_actions_uncompiled) # <--- ADD THIS LINE
sample_actions = jax.jit(sample_actions_uncompiled)#jax.jit(sample_actions_uncompiled)
env_state = jit_reset(env_keys) #env.reset(env_keys)
obs = env_state.observation

# obs, env_state = jit_reset(env_keys) #env.reset(env_keys)
# train_state = agent_state.train_state

step_total = 0


if config.experiment.n_seeds > 1:
    assert train_state_seed is not None, ("Loaded train state has multiple seeds. Please specify "
                                            "train_state_seed for replay.")
    
    intial_train_state = jax.tree.map(lambda x: x[train_state_seed], agent_state.train_state)
else: 
    # obs, env_state = jit_reset(env_keys) #env.reset(env_keys)
    intial_train_state = agent_state.train_state

train_state = intial_train_state


for i in range(n_steps):

    rng, _rng = jax.random.split(rng)
    # action, train_state = sample_actions(train_state, obs, _rng) # Now calls the JITted version
    action, train_state = sample_actions(train_state, obs, _rng) # Now calls the JITted version
    action = jnp.atleast_2d(action)


    env_state, sys = jit_step(env_state, action)  #env.step(env_state, action)
    print(f"sys.jnt_stiffness: {sys.jnt_stiffness}")
    print(f"sys.dof_damping: {sys.dof_damping}")
    print(f"sys.body_pos: {sys.body_pos}")
    print(f"sys.body_quat: {sys.body_quat}")

    obs = env_state.observation
    # obs, reward, absorbing, done, info, env_state = jit_step(env_state, action)  #env.step(env_state, action)

    step_total += n_envs 
    # env.mjx_render(env_state)
    env.mjx_render_domain_randomization(env_state, record=True)


env.stop()


