import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["JAX_PLATFORMS"] = "cuda"
# os.environ["CUDA_VISIBLE_DEVICES"] = "" 
# os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
# import time
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.environments.humanoids.skeleton_prosthesis import MjxSkeletonMuscleProsthesis
from loco_mujoco.core.control_functions.skeleton_muscle import SkeletonMuscleControlFunction
from loco_mujoco.utils import MetricsHandler
# from loco_mujoco.evaluation.evaluation_prosthesis import ProsthesisMetricsHandler
from omegaconf import OmegaConf

import mujoco
import numpy as np
import jax.numpy as jnp


import jax 
# jax.config.update('jax_platform_name', 'cpu')
os.environ["MUJOCO_GL"] = "egl"
from loco_mujoco.utils import MetricsHandler

import argparse

parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
args = parser.parse_args()

# # # Use the path from command line arguments
path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config


# get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# create env
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = False
config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic

# add validation flag to config.experiment.env_params
config.experiment.env_params["eval_force"] = True

env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
# env = MjxSkeletonMuscleProsthesis(**config.experiment.env_params, **config.experiment.task_factory.params)
data = env.data

# model = env.model 
model = env.get_model()

rng_reset = jax.jit(env.mjx_reset)
rng_step = jax.jit(env.mjx_step) #model, data)
# state = jit_reset(jax.random.PRNGKey(0))
rng_sample_uni_action = jax.jit(env.sample_action_space)

n_envs = 1 

state = rng_reset(jax.random.PRNGKey(0))

# Import as mujoco environemnt not mjx 
# prosthesis_metrics_handler = ProsthesisMetricsHandler(config, env)

step = 0
action_dim = env.info.action_space.shape[0]
all_step_contact_left = []  # to store the step when contact is detected
all_step_contact_right = []  # to store the step when contact is detected
all_tibia_sensor_data = []  # to store tibia sensor data at amputated side
all_contact_force_l = []  # to store contact force for left foot
all_contact_force_r = []  # to store contact force for right foot
# run it in 
for i in range(10):
    action = rng_sample_uni_action(jax.random.PRNGKey(0))
    state = rng_step(state, action)

    mjx_data = state.data

    # action = np.random.randn(action_dim)
    # nstate, reward, absorbing, done, info = env.step(action)

    # data = env.data

    # mj_data = mujoco.mjx.get_data(model, mjx_data)

    step += n_envs

    # testing with sensor data can use mjx_data.sensordata 
    # testing to see if there is contact mjx_data.contact.dist: if dist <= 0, then there is contact, else no contact (pos gives position of contact point, even if there is nto contact Even if there's no active contact, MuJoCo might still track a "closest point" or a reference point for potential pairs. )
    # Get contact force: mjx_data.cfrc_ext (correct increment: Only toe_l & toe_r are collidable (see add_box_feet_to_spec)  
    #### body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "toes_r") body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "toes_l")
    #### mjx_data.cfrc_ext[body_id]  [Fx, Fy, Fz, Tx, Ty, Tz] represents the sum of all external contact forces and torques acting on that specific body


    if step % 100 == 0:
        print(f"Step {step}") #: State time = {state.data.time}")

    # print(f"data.ncon: {data.ncon}")

    # detect conact when there has not been contact in the previous step
    # mjx_data first 4 entries are right, last 4 entries are left foot
    for n in range(mjx_data.ncon):
        geom_distance = mjx_data.contact.dist[n]
        if geom_distance <= 0:
            # get geom_name2 if foot_box_l then add step to all_step_start_left, if foot_box_r then add step to all_step_start_right
            geom2 = mjx_data.contact.geom2[n]
            geom_name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2.item())
            if "_l" in geom_name2:
                step_start = step
                all_step_contact_left.append(step_start)
            elif "_r" in geom_name2:
                step_start = step
                all_step_contact_right.append(step_start)
            else:
                continue


    # Get contact force for toe_l and toe_r 
    body_id_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "toes_l")
    body_id_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "toes_r")
    contact_force_l = mjx_data.cfrc_ext[body_id_l]
    contact_force_r = mjx_data.cfrc_ext[body_id_r]
    all_contact_force_l.append(contact_force_l)
    all_contact_force_r.append(contact_force_r)
    print(f"Contact force left: {all_contact_force_l}, Contact force right: {all_contact_force_r}")



    # get sensor data for tibia 
    # Retrieve sensor names from the model
    sensor_names = []
    for sensor_id in range(model.nsensor):
        sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
        sensor_names.append(sensor_name)

    print(f"Number of sensors: {model.nsensor}")
    print(f"Sensor names: {sensor_names}")
    if model.nsensor == 1: 
       # tibia sensor is the only sensor available
        sensor_data = mjx_data.sensordata
        all_tibia_sensor_data.append(sensor_data)
        print(f"Human-prosthesis interface sensor data: {sensor_data}") 
    else: 
        # get sensor_names and iterate through all sensors
        for sensor_id in range(model.nsensor):
            sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
            sensor_data = mjx_data.sensordata[sensor_id]
            print(f"Sensor {sensor_name} data: {sensor_data}")
            # No specific sensor data is saved!!!! FIX 


# take out reps in all_step_contact_left and all_step_contact_right 
all_step_contact_left = list(set(all_step_contact_left))  # remove duplicates
all_step_contact_right = list(set(all_step_contact_right))  # remove duplicates
print(f"All steps with contact left: {all_step_contact_left}")
print(f"All steps with contact right: {all_step_contact_right}")
all_step_start_left =  [all_step_contact_left[0]] # to store the step when contact is detected
all_step_start_right = [all_step_contact_right[0]]  # to store the step when contact is detected
# Look through all_step_contact_left and all_step_contact_right extract numbers that are not consecutive
for m in range(len(all_step_contact_left) - 1):
    if all_step_contact_left[m + 1] - all_step_contact_left[m] > 1:
        all_step_start_left.append(all_step_contact_left[m])
for m in range(len(all_step_contact_right) - 1):
    if all_step_contact_right[m + 1] - all_step_contact_right[m] > 1:
        all_step_start_right.append(all_step_contact_right[m])
print(f"All start steps left: {all_step_start_left}")
print(f"All start steps right: {all_step_start_right}")

        

    



