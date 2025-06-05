import os
import jax
import time

from loco_mujoco import ImitationFactory
from loco_mujoco.task_factories import DefaultDatasetConf, LAFAN1DatasetConf, CustomDatasetConf




# can increase the speed by ~30% on some GPUs
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_triton_gemm_any=True ')


# create env
env = ImitationFactory.make("MjxSkeletonMuscle", lafan1_dataset_conf=LAFAN1DatasetConf(["walk1_subject1"]))

# get action space of environment high and low values
# action_space = env.action_space
# action_space_high = action_space.high
# action_space_low = action_space.low
# print(action_space_high, action_space_low)


# # create keys
# key = jax.random.key(0)
# n_envs = 5 #100
# keys = jax.random.split(key, n_envs + 1)
# key, env_keys = keys[0], keys[1:]

# # jit and vmap all functions needed
# rng_reset = jax.jit(jax.vmap(env.mjx_reset))
# rng_step = jax.jit(jax.vmap(env.mjx_step))
# rng_sample_uni_action = jax.jit(jax.vmap(env.sample_action_space))

# # reset env
# state = rng_reset(env_keys)

# step = 0
# previous_time = time.time()
# LOGGING_FREQUENCY = 100000
# i = 0
# while i < 100000:

#     # step
#     keys = jax.random.split(key, n_envs + 1)
#     key, action_keys = keys[0], keys[1:]
#     action = rng_sample_uni_action(action_keys)
#     state = rng_step(state, action)

#     # parallel render
#     env.mjx_render(state)

#     step += n_envs

#     # log speed (disable rendering for accurate speed measurement)
#     if step % LOGGING_FREQUENCY == 0:
#         current_time = time.time()
#         print(f"{int(LOGGING_FREQUENCY / (current_time - previous_time))} steps per second.")
#         previous_time = current_time

#     i+=1

