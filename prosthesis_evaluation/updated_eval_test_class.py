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
config.experiment.env_params["headless"] = True #False
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic
config.experiment.env_params["add_sensors"] = True
env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
env.th.to_jax()
env = VecEnv(env)
jit_step  = jax.jit(jax.vmap(env.mjx_step))  #env.step)
jit_reset  = jax.jit(jax.vmap(env.mjx_reset)) #env.reset)
model = env.get_model()

prosthesis_metrics_handler = ProsthesisMetricsHandler(env) #(config, env)

n_steps = 1000 #1000
n_envs = 1 #1  # <--- Make sure this matches your training batch size
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
sample_actions = jax.jit(sample_actions_uncompiled)
env_state = jit_reset(env_keys) #env.reset(env_keys)
obs = env_state.observation
# obs, env_state = jit_reset(env_keys) #env.reset(env_keys)
# train_state = agent_state.train_state

step_total = 0


if config.experiment.n_seeds > 1:
    assert train_state_seed is not None, ("Loaded train state has multiple seeds. Please specify "
                                            "train_state_seed for replay.")
    
    train_state = jax.tree.map(lambda x: x[train_state_seed], agent_state.train_state)
else: 
    # obs, env_state = jit_reset(env_keys) #env.reset(env_keys)
    train_state = agent_state.train_state


###### Some params for evaluation 
all_foot_ground_contact_left =[]
all_foot_ground_contact_right =[]            


joint_data = {}
for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    joint_data[joint_name] = {
        "angle": [],
        "velocity": [],
        "forces_constraint": [],
        "forces_smooth": [],
        "torques": [],
        "energy_exp": [],
    }
    # Per-step data
    if "_l" in joint_name or "_r" in joint_name:
        joint_data[joint_name].update({
            "angle_per_step": [],
            "velocity_per_step": [],
            "forces_constraint_per_step": [],
            "forces_smooth_per_step": [],
            "torques_per_step": [],
            "energy_exp_per_step": [],
        })
    else:
        joint_data[joint_name].update({
            "angle_per_step_left": [],
            "angle_per_step_right": [],
            "velocity_per_step_left": [],
            "velocity_per_step_right": [],
            "forces_constraint_per_step_left": [],
            "forces_constraint_per_step_right": [],
            "forces_smooth_per_step_left": [],
            "forces_smooth_per_step_right": [],
            "torques_per_step_left": [],
            "torques_per_step_right": [],
            "energy_exp_per_step_left": [],
            "energy_exp_per_step_right": [],
        })


# all_tibia_sensor_data = []
all_sensor_force = {}


evaluation_muscle_groups = ["back_muscles", "torso_muscles", "vasti_muscles", "rectus_muscles", "medial_muscles", "leg_muscles", "all_muscles"]


evaluation_muscle_names = {}
for n in evaluation_muscle_groups:
    evaluation_muscle_names[n] = prosthesis_metrics_handler.get_muscle_group(n)


for muscle_group, muscle_names in evaluation_muscle_names.items():
    locals()[f"run_{muscle_group}_activation_left"] = []
    locals()[f"run_{muscle_group}_activation_right"] = []


# Body name with possible contact to ground 
foot_name = "toes"  # name of the foot box in the model
all_grf_l = []
all_grf_r = []

left_side = "left_side"
right_side = "right_side"

all_actions = []
all_actuator_names = []


muscle_skeleton_control_activation = SkeletonMuscleControlFunction(env)


###########
time_all = []
time_all.append(timeit.default_timer())  # Start timer
for i in range(n_steps):
    rng, _rng = jax.random.split(rng)
    # action, train_state = sample_actions(train_state, obs, _rng) # Now calls the JITted version
    action, train_state = sample_actions(train_state, obs, _rng) # Now calls the JITted version
    action = jnp.atleast_2d(action)


    env_state = jit_step(env_state, action)  #env.step(env_state, action)
    obs = env_state.observation
    # obs, reward, absorbing, done, info, env_state = jit_step(env_state, action)  #env.step(env_state, action)

    if step_total % 100 == 0:
        print(f"Step {step_total}")


    # # # # # # # # Collect metrics 
    
    # Foot ground contact indices 
    contact_left, contact_right = prosthesis_metrics_handler.get_contact_steps(env_state.data, i)
    #print("contact_left: ", contact_left)
    #print("contact_right: ", contact_right)
    # if contact_left not equal [] then append to list

    all_foot_ground_contact_left.append(contact_left)
    all_foot_ground_contact_right.append(contact_right)

    # GRF 
    grf_l = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_l")
    grf_r = prosthesis_metrics_handler.get_grf(env_state.data, f"{foot_name}_r")
    all_grf_l.append(grf_l)
    all_grf_r.append(grf_r)


    # Joint data
    joint_angles = prosthesis_metrics_handler.get_joint_angles(env_state.data)
    joint_velocities = prosthesis_metrics_handler.get_joint_vels(env_state.data)
    joint_forces_constraint, joint_forces_smooth = prosthesis_metrics_handler.get_joint_frces(env_state.data)
    joint_torques = prosthesis_metrics_handler.get_joint_trques(env_state.data)
    joint_energy_exp = prosthesis_metrics_handler.calc_joint_energy_exp(joint_torques, joint_velocities)
    # Append all joint data to the joint_data dictionary
    for joint_name, angle in joint_angles.items():
        joint_data[joint_name]["angle"].append(angle)
    for joint_name, velocity in joint_velocities.items():
        joint_data[joint_name]["velocity"].append(velocity)
    for joint_name, force in joint_forces_constraint.items():
        joint_data[joint_name]["forces_constraint"].append(force)
    for joint_name, force in joint_forces_smooth.items():
        joint_data[joint_name]["forces_smooth"].append(force)
    for joint_name, torque in joint_torques.items():
        joint_data[joint_name]["torques"].append(torque)
    for joint_name, energy_exp in joint_energy_exp.items():
        joint_data[joint_name]["energy_exp"].append(energy_exp)



    #  Tibia sensor data 
    sensor_force, sensor_force_names = prosthesis_metrics_handler.get_sensor_data(env_state.data)
    for sensor_name, force in sensor_force.items():
        # Initialize the list if it doesn't exist
        if sensor_name not in all_sensor_force:
            all_sensor_force[sensor_name] = []
        # Append the sensor force data to the corresponding list
        all_sensor_force[sensor_name].append(force)



    # # Action data 
    # print(f"Step: {i}, Action: {action}")

    for i in range(model.nu):
        if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE: #Muscle
            # jax.debug.print("Actuator {i} is a Muscle", i=i)
            # apply sigmoid activation function for muscle control
            # action = action.at[i].set(jax.nn.sigmoid(action[i]))
            action = action.at[...,i].set(muscle_skeleton_control_activation.adapted_sigmoid(action[...,i]))

    # print(f"Step: {i}, Action after sigmoid: {action}")

    all_actions.append(action)

    all_actuator_names = []
    for a in range(model.nu):
        # get all actuator_names
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        all_actuator_names.append(actuator_name)


    for muscle_group, muscle_names in evaluation_muscle_names.items():
        locals()[f"{muscle_group}_activation_left"] = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, left_side, action)
        locals()[f"{muscle_group}_activation_right"] = prosthesis_metrics_handler.get_relevant_ctrl(muscle_names, right_side, action)
        locals()[f"run_{muscle_group}_activation_left"].append(locals()[f"{muscle_group}_activation_left"])
        locals()[f"run_{muscle_group}_activation_right"].append(locals()[f"{muscle_group}_activation_right"])



    step_total += n_envs 

    # env.mjx_render(env_state, record=True)

time_all.append(timeit.default_timer())  # End timer

env.stop()



# ACTIONS 
sum_step_activations = {}
step_norm_activations = {}
step_num_norm_activations = {}
sum_muscles_activations = {}
step_muscles_norm_activations = {}
step_num_norm_activations = {}

for muscle_group in evaluation_muscle_groups:
    sum_step_activations[f"{muscle_group}_left"] = jnp.sum(jnp.array(locals()[f"run_{muscle_group}_activation_left"]), axis=0)
    sum_step_activations[f"{muscle_group}_right"] = jnp.sum(jnp.array(locals()[f"run_{muscle_group}_activation_right"]), axis=0)

    step_norm_activations[f"{muscle_group}_left"] = sum_step_activations[f"{muscle_group}_left"] / n_steps
    step_norm_activations[f"{muscle_group}_right"] = sum_step_activations[f"{muscle_group}_right"] / n_steps

    # print(f"Sum of {muscle_group.replace('_', ' ').title()} Activation Left Per Muscle: {sum_step_activations[f'{muscle_group}_left']}")
    # print(f"Sum of {muscle_group.replace('_', ' ').title()} Activation Right Per Muscle: {sum_step_activations[f'{muscle_group}_right']}")

    # print(f"Step normalized {muscle_group.replace('_', ' ').title()} Activation Left Per Muscle: {step_norm_activations[f'{muscle_group}_left']}")
    # print(f"Step normalized {muscle_group.replace('_', ' ').title()} Activation Right Per Muscle: {step_norm_activations[f'{muscle_group}_right']}")

    # Get amount of muscles in the group
    n_musc = len(evaluation_muscle_names[muscle_group])
    # get sum along axis=0 and 1    
    sum_muscles_activations[f"{muscle_group}_left"] = jnp.sum(sum_step_activations[f"{muscle_group}_left"], axis=0)
    sum_muscles_activations[f"{muscle_group}_right"] = jnp.sum(sum_step_activations[f"{muscle_group}_right"], axis=0)
    step_muscles_norm_activations[f"{muscle_group}_left"] = sum_muscles_activations[f"{muscle_group}_left"] / n_steps
    step_muscles_norm_activations[f"{muscle_group}_right"] = sum_muscles_activations[f"{muscle_group}_right"] / n_steps
    step_num_norm_activations[f"{muscle_group}_left"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_left"]/ n_musc), #axis=0)
    step_num_norm_activations[f"{muscle_group}_right"] = jnp.sum(step_muscles_norm_activations[f"{muscle_group}_right"]/ n_musc) #, axis=0)

    # print(f"Sum of {muscle_group.replace('_', ' ').title()} Muscles Activation Left: {sum_muscles_activations[f'{muscle_group}_left']}")
    # print(f"Sum of {muscle_group.replace('_', ' ').title()} Muscles Activation Right: {sum_muscles_activations[f'{muscle_group}_right']}")
    # print(f"Step normalized {muscle_group.replace('_', ' ').title()} Muscles Activation Left: {step_muscles_norm_activations[f'{muscle_group}_left']}")
    # print(f"Step normalized {muscle_group.replace('_', ' ').title()} Muscles Activation Right: {step_muscles_norm_activations[f'{muscle_group}_right']}")





# GRF & STEP EXTRACTION 
# print(f"All Foot Ground Contact Left: {all_foot_ground_contact_left}")
# print(f"All Foot Ground Contact Right: {all_foot_ground_contact_right}")   
# filtered_all_foot_ground_contact_left = [contact for contact in all_foot_ground_contact_left if contact != []]
# filtered_all_foot_ground_contact_right = [contact for contact in all_foot_ground_contact_right if contact != []]
# print(f"Filtered All Foot Ground Contact Left: {filtered_all_foot_ground_contact_left}")
# print(f"Filtered All Foot Ground Contact Right: {filtered_all_foot_ground_contact_right}")
# all_step_start_left= prosthesis_metrics_handler.get_start_steps(filtered_all_foot_ground_contact_left)
# all_step_start_right = prosthesis_metrics_handler.get_start_steps(filtered_all_foot_ground_contact_right)
# print(f"All Step Start Left: {all_step_start_left}")
# print(f"All Step Start Right: {all_step_start_right}")



# Test getting joint angle per step 
# ??????????????? FILTER OR KEEP ALL?????????????????? 
# if filtered_step_start_left and filtered_step_start_right:
#     for joint_name in joint_data.keys():
#         joint_angle = jnp.array(joint_data[joint_name]["angle"])
#         if "_l" in joint_name:
#             joint_data[joint_name]["angle_per_step"] = prosthesis_metrics_handler.get_parameter_per_step(joint_angle, filtered_step_start_left)
#             print(f"Joint Angles Per Step {joint_name}: {joint_data[joint_name]['angle_per_step']}")
#         elif "_r" in joint_name:
#             joint_data[joint_name]["angle_per_step"] = prosthesis_metrics_handler.get_parameter_per_step(joint_angle, filtered_step_start_right)
#             print(f"Joint Angles Per Step {joint_name}: {joint_data[joint_name]['angle_per_step']}")
#         else:
#             joint_data[joint_name]["angle_per_step_left"] = prosthesis_metrics_handler.get_parameter_per_step(joint_angle, filtered_step_start_left)
#             joint_data[joint_name]["angle_per_step_right"] = prosthesis_metrics_handler.get_parameter_per_step(joint_angle, filtered_step_start_right)
#             print(f"Joint Angles Per Step Left ({joint_name}): {joint_data[joint_name]['angle_per_step_left']}")
#             print(f"Joint Angles Per Step Right ({joint_name}): {joint_data[joint_name]['angle_per_step_right']}")


joint_names = []
for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    joint_names.append(joint_name)

# Collect all relevant data into a dictionary
all_relevant_data = {
    "total_steps": step_total,
    "n_envs": n_envs,
    "all_grf_l": all_grf_l,
    "all_grf_r": all_grf_r,
    "all_foot_ground_contact_left": all_foot_ground_contact_left,
    "all_foot_ground_contact_right": all_foot_ground_contact_right,
    "all_sensor_force": all_sensor_force,
    "sensor_force_names": sensor_force_names,
    # "all_tibia_sensor_data": all_tibia_sensor_data,
    # "filtered_step_start_left": filtered_step_start_left,
    # "filtered_step_start_right": filtered_step_start_right,
    "sum_step_activations": sum_step_activations,
    "step_norm_activations": step_norm_activations,
    "sum_muscles_activations": sum_muscles_activations,
    "step_muscles_norm_activations": step_muscles_norm_activations,
    "step_num_norm_activations": step_num_norm_activations,
    "evaluation_muscle_groups": evaluation_muscle_groups,
    "evaluation_joint_names": joint_names,
    "evaluation_muscle_names": evaluation_muscle_names,
    "all_actions": all_actions,
    "all_actuator_names": all_actuator_names,
}


# Add joint data from joint_data dictionary
for joint_name, joint_dict in joint_data.items():
    for key, value in joint_dict.items():
        all_relevant_data[f"{joint_name}_{key}"] = value



# Add muscle activations
for muscle_group in evaluation_muscle_groups:
    all_relevant_data[f"run_{muscle_group}_activation_left"] = locals().get(f"run_{muscle_group}_activation_left", [])
    all_relevant_data[f"run_{muscle_group}_activation_right"] = locals().get(f"run_{muscle_group}_activation_right", [])

# Save to file
dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(os.path.dirname(path), f"{dt_str}_evaluation_results_{n_steps}steps.pkl")
with open(output_path, "wb") as f:
    pickle.dump(all_relevant_data, f)
print(f"Saved evaluation data to {output_path}")


#### Testing reading the data back from the file
# # read the data back from the file
# with open(output_path, "rb") as f:
#     loaded_data = pickle.load(f)
#     # print(f"Loaded data: {loaded_data.keys()}")  # Print the keys of the loaded data to verify


# # # Get step from the loaded data
# # test 
# total_steps = loaded_data.get("total_steps")
# loaded_data.get("lumbar_bending_angle")[0] # get first angle of lumbar_bending_angle
# print(f"Loaded Step: {total_steps}")
# print(f"Loaded Lumbar bending angle: {loaded_data.get('lumbar_bending_angle')[0]}")  # Example of accessing a specific joint angle


time_all.append(timeit.default_timer())  # End timer

print(f"Total time taken for {n_steps} steps: {time_all[-1] - time_all[0]} seconds")
print(f"Compilation time: {time_all[1] - time_all[0]} seconds")
print(f"Execution time taken: {time_all[-1] - time_all[1]} seconds")
print(f"Average time per step: {(time_all[-1] - time_all[1]) / (n_steps-1)} seconds")