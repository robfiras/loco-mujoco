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
from loco_mujoco.evaluation.evaluation_prosthesis import ProsthesisMetricsHandler
from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction
import mujoco
from datetime import datetime
import timeit 
import numpy as np
from loco_mujoco.core.visuals import MujocoViewer

os.environ["MUJOCO_GL"] = "egl"

add_perturb = True
perturb_force_min = 0
perturb_force_increment = 10
perturb_force_start = 0
perturb_force_max = 300
perturb_force_direction = ["x", "y", "z"]

# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

# Use the path from command line arguments
path = args.path
# Load the agent state, which might contain multiple seeds
agent_conf, agent_state_all_seeds = PPOJax.load_agent(path) 
config = agent_conf.config

# get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# create env
OmegaConf.set_struct(config, False)
config.experiment.env_params["headless"] = True
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"
config.experiment.env_params["add_sensors"] = True
env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
env.th.to_jax()
env = VecEnv(env)
jit_step = jax.jit(jax.vmap(env.mjx_step))
jit_reset = jax.jit(jax.vmap(env.mjx_reset))
model = env.get_model()

prosthesis_metrics_handler = ProsthesisMetricsHandler(config, env)

n_steps = 1000
n_envs = 1 # Keep this as 1 if you want to run one env per evaluation.
rng = jax.random.key(0)

# --- Select only the first seed (index 0) from the loaded train_states ---
train_state_seed_idx = 0 # Explicitly target the first seed
print(f"\n--- Evaluating only Seed {train_state_seed_idx} ---")

# Extract the train_state for the chosen seed
initial_train_state = jax.tree.map(lambda x: x[train_state_seed_idx], agent_state_all_seeds.train_state)

def sample_actions_uncompiled(ts, obs, _rng):
    y, updates = agent_conf.network.apply({'params': ts.params,
                                            'run_stats': ts.run_stats},
                                            obs, mutable=["run_stats"])
    ts = ts.replace(run_stats=updates['run_stats'])
    pi, _ = y
    a = pi.sample(seed=_rng)
    return a, ts

sample_actions = jax.jit(sample_actions_uncompiled)

# Initialize environment for the evaluation run
keys = jax.random.split(rng, n_envs + 1)
rng, env_keys = keys[0], keys[1:]

env_state = jit_reset(env_keys)
obs = env_state.observation

step_total = 0

if add_perturb:
    body_name = "torso"
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    perturbation_force_mjx = jnp.zeros((n_envs, model.nbody, 6))
    perturbation_force_mjx0 = perturbation_force_mjx.at[:, body_id, :].set(jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

force_time_dict = {}
for n in range(len(perturb_force_direction)):
    direction = perturb_force_direction[n]
    for f in range(perturb_force_min, perturb_force_max + 1, perturb_force_increment):
        if f not in force_time_dict:
            force_time_dict[f] = {}
        force_time_dict[f][direction] = []

for d in perturb_force_direction:
    for f in range(perturb_force_min, perturb_force_max + 1, perturb_force_increment):
        if add_perturb:
            if d == "x":
                perturbation_force_mjx_current = perturbation_force_mjx.at[:, body_id, :].set(jnp.array([f, 0.0, 0.0, 0.0, 0.0, 0.0]))
            elif d == "y":
                perturbation_force_mjx_current = perturbation_force_mjx.at[:, body_id, :].set(jnp.array([0.0, f, 0.0, 0.0, 0.0, 0.0]))
            elif d == "z":
                perturbation_force_mjx_current = perturbation_force_mjx.at[:, body_id, :].set(jnp.array([0.0, 0.0, f, 0.0, 0.0, 0.0]))
            else:
                perturbation_force_mjx_current = perturbation_force_mjx0 # No perturbation


        # Re-initialize environment and agent state for each force/direction combination
        env_state = jit_reset(env_keys)
        obs = env_state.observation
        # Re-assign the initial_train_state to reset the agent's internal state
        train_state = initial_train_state 

        step_total = 0 # Reset step_total for each new test

        for i in range(n_steps):
            if add_perturb:
                if i >= perturb_force_start: # Apply perturbation from perturb_force_start onwards
                    env_state = env_state.replace(data=env_state.data.replace(xfrc_applied=perturbation_force_mjx_current))
                else: # Apply zero perturbation before start time
                    env_state = env_state.replace(data=env_state.data.replace(xfrc_applied=perturbation_force_mjx0))

            rng, _rng = jax.random.split(rng)
            action, train_state = sample_actions(train_state, obs, _rng)
            action = jnp.atleast_2d(action)

            env_state = jit_step(env_state, action)
            obs = env_state.observation

            if env_state.done.any(): # Check if *any* environment is done
                # Record the step when the first environment became done
                force_time_dict[f][d] = step_total + env_state.done.argmax().item() + 1 
                print(f"Done at step {i+1} for force {f} in direction {d} (Seed {train_state_seed_idx})")
                break # Exit inner loop for current force/direction

            step_total += n_envs

        # If the loop completes without `done.any()`, it means it finished `n_steps`
        if not env_state.done.any():
            force_time_dict[f][d] = n_steps # If all environments completed `n_steps`
            print(f"Completed {n_steps} steps successfully for force {f} in direction {d} (Seed {train_state_seed_idx})")
        
# You can process or save force_time_dict for this single seed here
print(f"Final Results for Seed {train_state_seed_idx}: {force_time_dict}")

env.stop()



# Save the force_time_dict to a file
dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(os.path.dirname(path), f"{dt_str}_perturbation_force_time_seed0.pkl")
with open(output_path, 'wb') as f:
    pickle.dump(force_time_dict, f)
print(f"Perturbation force time data saved to {output_path}")
