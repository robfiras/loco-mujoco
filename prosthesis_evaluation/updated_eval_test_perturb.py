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

os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering, which is more compatible with headless environments
# os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_triton_gemm_any=True ')


add_perturb = True
perturb_force_min = 10
perturb_force_increment = 50 #10
# perturb_force_steps = 100
perturb_force_start = 10 #100 #100
# perturb_force_repeat = 10 
perturb_force_max = 80# 300 #200 #20 #100 #100
perturb_force_orientation = ["x", "y", "z"] # or "x"
perturb_force_direction_switch = True # Switch between + & - perturbation forces 
perturb_force_step_duration = 20 
perturb_force_step_distance = 10
cycle_length = perturb_force_step_duration + perturb_force_step_distance # Apply force for force_step_duration steps, then stop for force_step_distance steps, repeat



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
config.experiment.env_params["headless"] = True #False
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic
config.experiment.env_params["add_sensors"] = True
env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
env.th.to_jax()
env = VecEnv(env)
jit_step  = jax.jit(jax.vmap(env.mjx_step))  #env.step)
jit_reset  = jax.jit(jax.vmap(env.mjx_reset)) #env.reset)
# jit_step  = jax.jit(env.step)
# jit_reset  = jax.jit(env.reset)
model = env.get_model()

prosthesis_metrics_handler = ProsthesisMetricsHandler(config, env)

n_steps = 100 #1000 #1000 #1000
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

if add_perturb:
    body_name = "torso"
    # FIX: xfrc_applied acts on BODIES, so use mjOBJ_BODY, not mjOBJ_GEOM.
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) 

    # Initialize a new xfrc_applied array with zeros for all environments and bodies.
    # This implicitly "clears" any forces from the previous step.
    perturbation_force_mjx = jnp.zeros((n_envs, model.nbody, 6))

train_state = intial_train_state

force_time_dict = {}
for n in range(len(perturb_force_orientation)):
    direction = perturb_force_orientation[n]
    for f in range(perturb_force_min, perturb_force_max + 1, perturb_force_increment):
        if f not in force_time_dict:
            force_time_dict[f] = {}
        force_time_dict[f][direction] = []
# current_force = perturb_force_start

if add_perturb:
    perturbation_force_mjx0 = perturbation_force_mjx.at[:, body_id, :].set(jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


for d in perturb_force_orientation:
    for f in range(perturb_force_min, perturb_force_max + 1, perturb_force_increment):
        # if add_perturb:
        #     if d == "x":
        #         perturbation_force_mjx = perturbation_force_mjx.at[:, body_id, :].set(
        #             jnp.array([f, 0.0, 0.0, 0.0, 0.0, 0.0])
        #         )
        #     elif d == "y":
        #         perturbation_force_mjx = perturbation_force_mjx.at[:, body_id, :].set(
        #             jnp.array([0.0, f, 0.0, 0.0, 0.0, 0.0])
        #         )
        #     elif d == "z":
        #         perturbation_force_mjx = perturbation_force_mjx.at[:, body_id, :].set(
        #             jnp.array([0.0, 0.0, f, 0.0, 0.0, 0.0])
        #         )

        

        # Update the xfrc_applied field in the env_state's data.
        env_state = env_state.replace(data=env_state.data.replace(xfrc_applied=perturbation_force_mjx))

        for i in range(n_steps):
            
            if add_perturb:
                # Determine position within the cycle
                cycle_idx = (i - perturb_force_start) // cycle_length
                cycle_pos = (i - perturb_force_start) % cycle_length

                # Determine force direction: alternate sign every cycle
                sign = 1 if cycle_idx % 2 == 0 else -1

                if i >= perturb_force_start and cycle_pos < perturb_force_step_duration:
                    f_direction = sign * f  if perturb_force_direction_switch else f  # Apply force with the correct sign 
                    if d == "x":
                        perturbation_force_mjx = perturbation_force_mjx.at[:, body_id, :].set(
                            jnp.array([f_direction, 0.0, 0.0, 0.0, 0.0, 0.0])
                        )
                    elif d == "y":
                        perturbation_force_mjx = perturbation_force_mjx.at[:, body_id, :].set(
                            jnp.array([0.0, f_direction, 0.0, 0.0, 0.0, 0.0])
                        )
                    elif d == "z":
                        perturbation_force_mjx = perturbation_force_mjx.at[:, body_id, :].set(
                            jnp.array([0.0, 0.0, f_direction, 0.0, 0.0, 0.0])
                        )

                    # print(f"Applying perturbation force {perturbation_force_mjx} at step {i} for force {f} in direction {d}")
                    env_state = env_state.replace(data=env_state.data.replace(xfrc_applied=perturbation_force_mjx))
                else:
                    # No force applied
                    # print(f"Stopping perturbation force at step {i} for force {f} in direction {d}")
                    env_state = env_state.replace(data=env_state.data.replace(xfrc_applied=perturbation_force_mjx0))
                

            rng, _rng = jax.random.split(rng)
            # action, train_state = sample_actions(train_state, obs, _rng) # Now calls the JITted version
            action, train_state = sample_actions(train_state, obs, _rng) # Now calls the JITted version
            action = jnp.atleast_2d(action)


            env_state = jit_step(env_state, action)  #env.step(env_state, action)
            obs = env_state.observation
            # obs, reward, absorbing, done, info, env_state = jit_step(env_state, action)  #env.step(env_state, action)

            # if env_state.done: #.any():
            #     # Reset the environment if any environment is done
            #     env_state = jit_reset(env_keys)
            #     force_time_dict[f][d]= step_total
            #     step_total = 0 
            #     print(f"Done at step {i} for force {f} in direction {d}")
            #     env_state = jit_reset(env_keys) #env.reset(env_keys)
            #     obs = env_state.observation
            #     train_state = intial_train_state
        

            step_total += n_envs 
            # if step_total == n_steps * n_envs:
            #     force_time_dict[f][d] = step_total
            #     print(f"Completed {n_steps} steps successfully for force {f} in direction {d}")
            #     env_state = jit_reset(env_keys)
            #     obs = env_state.observation
            #     train_state = intial_train_state
            #     step_total = 0 
            # print(f"DONE: {env_state.done}")
            if env_state.done.any(): # Check if *any* environment is done
                # Record the step when the first environment became done
                force_time_dict[f][d] = step_total + env_state.done.argmax().item() + 1 
                print(f"Done at step {i+1} for force {f} in direction {d} (Seed {train_state_seed})")
                break # Exit inner loop for current force/direction


            # env.mjx_render(env_state, record=True)

        # If the loop completes without `done.any()`, it means it finished `n_steps`
        if not env_state.done.any():
            force_time_dict[f][d] = n_steps # If all environments completed `n_steps`
            print(f"Completed {n_steps} steps successfully for force {f} in direction {d} (Seed {train_state_seed})")
        


env.stop()


# Save the force_time_dict to a file
dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
if config.experiment.n_seeds > 1:
    output_path = os.path.join(os.path.dirname(path), f"{dt_str}_perturbation_force_time_seed0.pkl")
else:
    output_path = os.path.join(os.path.dirname(path), f"{dt_str}_perturbation_force_time.pkl")
with open(output_path, 'wb') as f:
    pickle.dump(force_time_dict, f)
print(f"Perturbation force time data saved to {output_path}")
