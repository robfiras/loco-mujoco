import time 
import jax 
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.viewer
from loco_mujoco import ImitationFactory
from loco_mujoco.environments.humanoids.skeletons import MjxSkeletonMuscle

from loco_mujoco.evaluation.evaluation_prosthesis import ProsthesisMetricsHandler

import os 
import argparse
from omegaconf import OmegaConf
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.core.wrappers import VecEnv
from loco_mujoco import TaskFactory


# os.environ["MUJOCO_GL"] = "egl"  # Use EGL for rendering, which is more compatible with headless environments
# # os.environ["JAX_PLATFORMS"] = "cpu"
# # os.environ['XLA_FLAGS'] = (
# #     '--xla_gpu_triton_gemm_any=True ')

# # Set up argument parser
# parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
# parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
# parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
# args = parser.parse_args()

# # Use the path from command line arguments
# path = args.path
# agent_conf, agent_state = PPOJax.load_agent(path)
# config = agent_conf.config

# # get task factory
# factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# # create env
# OmegaConf.set_struct(config, False)  # Allow modifications
# config.experiment.env_params["headless"] = True #False
# config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic
# config.experiment.env_params["add_sensors"] = True
# env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)
# env.th.to_jax()
# env = VecEnv(env)
# jit_step  = jax.jit(jax.vmap(env.mjx_step))  #env.step)
# jit_reset  = jax.jit(jax.vmap(env.mjx_reset)) #env.reset)
# model = env.get_model()

# prosthesis_metrics_handler = ProsthesisMetricsHandler(env) #(config, env)

# n_envs = 1 #1  # <--- Make sure this matches your training batch size
# rng = jax.random.key(0)
# train_state_seed = 0  # Take first seed 

# keys = jax.random.split(rng, n_envs + 1)
# rng, env_keys = keys[0], keys[1:]

# def sample_actions_uncompiled(ts, obs, _rng): # Renamed for clarity
#     y, updates = agent_conf.network.apply({'params': ts.params,
#                                            'run_stats': ts.run_stats},
#                                            obs, mutable=["run_stats"])
#     ts = ts.replace(run_stats=updates['run_stats'])  # update stats
#     pi, _ = y
#     a = pi.sample(seed=_rng)
#     return a, ts

# # JIT compile the function
# # sample_actions = jax.jit(sample_actions_uncompiled) # <--- ADD THIS LINE
# sample_actions = jax.jit(sample_actions_uncompiled)
# state = jit_reset(env_keys) #env.reset(env_keys)
# obs = state.observation
# # obs, env_state = jit_reset(env_keys) #env.reset(env_keys)
# # train_state = agent_state.train_state

# step_total = 0


# if config.experiment.n_seeds > 1:
#     assert train_state_seed is not None, ("Loaded train state has multiple seeds. Please specify "
#                                             "train_state_seed for replay.")
    
#     train_state = jax.tree.map(lambda x: x[train_state_seed], agent_state.train_state)
# else: 
#     # obs, env_state = jit_reset(env_keys) #env.reset(env_keys)
#     train_state = agent_state.train_state


# # env = MjxSkeletonMuscle()
# # # spec = mujoco.MjSpec.from_file(env.get_default_xml_file_path())
# # # model = spec.compile() 
# # # data = mujoco.MjData(model)

# # prosthesis_metrics_handler = ProsthesisMetricsHandler(env)

# # action_dim = env.info.action_space.shape[0]



# # jit_reset = jax.jit(env.mjx_reset)
# # jit_step = jax.jit(env.mjx_step) #model, data)
# # state = jit_reset(jax.random.PRNGKey(0))

# # i=0

# # Body name with possible contact to ground 
# foot_name = "toes"  # name of the foot box in the model
# all_grf_l = []
# all_grf_r = []

# all_mean_grf_l = []
# all_mean_grf_r = []

# all_contact_l = []
# all_contact_r = []

# for i in range(1000):  # Run for a fixed number of steps
#     # if i == 1000:
#     #     jit_reset(jax.random.PRNGKey(0))
#     #     i = 0
    
#     rng, _rng = jax.random.split(rng)
#     # action, train_state = sample_actions(train_state, obs, _rng) # Now calls the JITted version
#     action, train_state = sample_actions(train_state, obs, _rng) # Now calls the JITted version
#     action = jnp.atleast_2d(action)


#     state = jit_step(state, action)  #env.step(env_state, action)
#     obs = state.observation
    
#     # action = np.random.randn(action_dim)
#     # state = jit_step(state, action)

#     mean_grf_l, mean_grf_r = prosthesis_metrics_handler.calc_mean_grf(state.data)
#     all_mean_grf_l.append(mean_grf_l)
#     all_mean_grf_r.append(mean_grf_r)
    
#     # GRF 
#     grf_l = prosthesis_metrics_handler.get_grf(state.data, f"{foot_name}_l")
#     grf_r = prosthesis_metrics_handler.get_grf(state.data, f"{foot_name}_r")
#     all_grf_l.append(grf_l)
#     all_grf_r.append(grf_r)

#     # Get contact step 
#     contact_l, contact_r = prosthesis_metrics_handler.get_contact_steps(state.data, i)
#     all_contact_l.append(contact_l)
#     all_contact_r.append(contact_r)


#     #env.mjx_render(state)
#     i += 1
#     if i % 100 == 0:
#         print(f"Step {i}")
#         # print(f"Contact Left: {contact_l}, Contact Right: {contact_r}")
#         # print(f"Step {i}: Left GRF: {grf_l}, Right GRF: {grf_r}")
#         # print(f"Mean Left GRF: {mean_grf_l}, Mean Right GRF: {mean_grf_r}")
#     # print(i)

# # take out [] of all_contact_l 
# filtered_all_contact_l= [contact for contact in all_contact_l if contact != []]
# filtered_all_contact_r= [contact for contact in all_contact_r if contact != []]
# # # save both force data to file 
# np.savez("grf_data_1000steps.npz", 
#          all_grf_l=np.array(all_grf_l), 
#          all_grf_r=np.array(all_grf_r),
#          all_mean_grf_l=np.array(all_mean_grf_l),
#          all_mean_grf_r=np.array(all_mean_grf_r), 
#          all_contact_l=np.array(filtered_all_contact_l),
#          all_contact_r=np.array(filtered_all_contact_r))


import matplotlib.pyplot as plt
# load data from file
loaded_data = np.load("/home/nadinebadie/loco-mujoco/prosthesis_test/grf_data_1000steps.npz")

contact_list_l = [0]*1000
for i in range(len(loaded_data['all_contact_l'])):
    ind = loaded_data['all_contact_l'][i]
    contact_list_l[ind] = 100
contact_list_r = [0]*1000
for i in range(len(loaded_data['all_contact_r'])):
    ind = loaded_data['all_contact_r'][i]
    contact_list_r[ind] = 100
     

plt.figure(figsize=(12, 6))
# plt.plot(loaded_data['all_grf_l'][:,4], label='Left GRF Y', alpha=0.5)
# plt.plot(loaded_data['all_grf_r'][:,4], label='Right GRF Y', alpha=0.5)
# plt.plot(loaded_data['all_mean_grf_l'][:,1], label='Mean Left GRF Y', linestyle     ='--', color='blue')
# plt.plot(loaded_data['all_mean_grf_r'][:,1], label='Mean Right GRF Y', linestyle='--', color='orange')
# plt.plot(loaded_data['all_grf_l'][:,5], label='Left GRF Z', alpha=0.5)
# plt.plot(loaded_data['all_grf_r'][:,5], label='Right GRF Z', alpha=0.5)
# plt.plot(loaded_data['all_mean_grf_l'][:,2], label='Mean Left GRF Z', linestyle     ='--', color='green')
# plt.plot(loaded_data['all_mean_grf_r'][:,2], label='Mean Right GRF Z', linestyle='--', color='red')

plt.plot(loaded_data['all_grf_l'][:,3], label='Left GRF X', alpha=0.5)
plt.plot(loaded_data['all_grf_r'][:,3], label='Right GRF X', alpha=0.5)
plt.plot(loaded_data['all_mean_grf_l'][:,0], label='Mean Left GRF X', linestyle     ='--', color='pink')
plt.plot(loaded_data['all_mean_grf_r'][:,0], label='Mean Right GRF X', linestyle='--', color='brown')

plt.plot(contact_list_l, label='Left Contact Step', linestyle=':', color='purple')
plt.plot(contact_list_r, label='Right Contact Step', linestyle=':', color='black')
plt.xlabel('Time Step')
plt.ylabel('Ground Reaction Force (GRF)')
plt.title('Ground Reaction Forces (GRF) Over Time')
plt.legend()
plt.grid()
plt.savefig('grf_plot.png')


