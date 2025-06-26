import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["JAX_PLATFORMS"] = "cpu"
import time 


import jax 
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')

import argparse
import mujoco

import numpy as np


from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.environments.humanoids.skeleton_prosthesis import MjxSkeletonMuscleProsthesis
from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction
from loco_mujoco.evaluation.evaluation_prosthesis import ProsthesisMetricsHandler

from omegaconf import OmegaConf

os.environ["MUJOCO_GL"] = "egl"

from loco_mujoco.utils import MetricsHandler


parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

# # Use the path from command line arguments
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


# prosthesis_side= "left_side"  # or "right_side"
# prosthesis_type = "transtibial"  # or "transfemoral", or "None"

# env = MjxSkeletonMuscleProsthesis(prosthesis_side=prosthesis_side, prosthesis_type=prosthesis_type)
# # setup metric handler (optional)
# mh = MetricsHandler(config, env) if config.experiment.validation.active else None


rng_reset = jax.jit(env.mjx_reset)
rng_step = jax.jit(env.mjx_step) #model, data)
# state = jit_reset(jax.random.PRNGKey(0))
rng_sample_uni_action = jax.jit(env.sample_action_space)


# # crmeate keys
# key = jax.random.key(0)
n_envs = 1 #100
# keys = jax.random.split(key, n_envs + 1)
# key, env_keys = keys[0], keys[1:]

# # jit and vmap all functions needed
# rng_reset = jax.jit(jax.vmap(env.mjx_reset))
# rng_step = jax.jit(jax.vmap(env.mjx_step))
# rng_sample_uni_action = jax.jit(jax.vmap(env.saple_action_space))

# reset env
# state = rng_reset(env_keys)
state = rng_reset(jax.random.PRNGKey(0)) #env_keys) #jax.random.PRNGKey(0))

# step = 0
previous_time = time.time()
LOGGING_FREQUENCY = 100000


# Initialize the metrics handler
prosthesis_metrics_handler = ProsthesisMetricsHandler(env) #(config, env)

data = state.data
act = data.ctrl

# test cont 
print('env.data.ncon after init env:', data.ncon)

all_left_indices = []
all_right_indices = []
all_data = []



which_muscles = "torso_muscles"  # or "back_muscles", "torso_muscles", "all_muscles", or specific muscle names
muscle_side = "left_side"  # or "right_side"
muscle_name = prosthesis_metrics_handler.get_muscle_group(which_muscles)
print(f"Muscle Name for {which_muscles} on {muscle_side}:", muscle_name)
all_extracted_actions = []

model = env.get_model()


# Run the environment for 1000 steps with actions sampled from the policy
# while i < 5:
for step in range(10000): #200):

    # step
    # keys = jax.random.split(key, n_envs + 1)
    # key, action_keys = keys[0], keys[1:]
    # action = rng_sample_uni_action(action_keys)
    # state = rng_step(state, action)


    action = rng_sample_uni_action(jax.random.PRNGKey(0))
    state = rng_step(state, action)

    # Update the environment data
    data = state.data
    # model = state.model

    # set options for renderer visualizing contact points and forces 

    # # Set visualization options for contact points and forces
    # env.model.vis.contactpoint = True
    # env.model.vis.contactforce = True
    # env.model.vis.transparent = True
    
    # # parallel render
    # env.mjx_render(state)

    step += n_envs

    # env.mjx_render(state)
    
    # Optionally, you can log or process the state here
    if step % 100 == 0:
        print(f"Step {step}") #: State time = {state.data.time}")


    # sigmoid 
    action = SkeletonMuscleControlFunction.adapted_sigmoid(SkeletonMuscleControlFunction, action)

    # get actions for specific muscle groups
    extracted_action = prosthesis_metrics_handler.extract_relevant_ctrl(model, muscle_name, muscle_side, action)
    all_extracted_actions.append(extracted_action)

    # append all env.mujoco.data
    all_data.append(state.data)

print('env.data.ncon in loop:', data.ncon)
# for i in range(data.ncon):
#     print('i', i)
#     contact = data.contact[i]
#     force = np.zeros(6)
#     mujoco.mj_contactForce(env.model, data, i, force)
#     jax.debug.print(f"Contact {i} between {contact.geom1} and {contact.geom2} with force: {force[:3]}")
#     # print(f"Contact {i} force: {force[:3]}")


# print(f"Extracted Action for {which_muscles} on {muscle_side}:", all_extracted_actions)
print(f"Sum of Extracted Action for {which_muscles} on {muscle_side}:", jnp.sum(extracted_action))


# Extract steps from the data
left_indices, right_indices = prosthesis_metrics_handler.extract_steps(model, data, step, all_data)
all_left_indices.append(left_indices)
all_right_indices.append(right_indices)
print("Left Steps Indices:", left_indices)
print("Right Steps Indices:", right_indices)


muscle_name = prosthesis_metrics_handler.get_muscle_group("vasti_muscles")  # or "back_muscles", "torso_muscles", "all_muscles", or specific muscle names

prosthesis_metrics_handler.evaluate_action_symmetry(model, all_data, muscle_name, left_indices, right_indices)


# Evaluate the policy and get energy parameters
# energy_params = PPOJax.evaluate_policy(env, agent_conf, agent_state, deterministic=False, n_steps=1000, n_envs=1, record=True,
#                                        train_state_seed=0)

# print("Energy parameters:", energy_params)


    
