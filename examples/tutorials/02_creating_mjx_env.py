import os
import jax
import time

from loco_mujoco import ImitationFactory


# can increase the speed by ~30% on some GPUs
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')

# create env and trajectory
env, traj = ImitationFactory.make("MjxUnitreeG1", disable_arms=True, default_dataset_conf=dict(task="squat"), use_mjwarp=False)

# create keys
key = jax.random.key(0)
n_envs = 100
keys = jax.random.split(key, n_envs + 1)
key, env_keys = keys[0], keys[1:]

# jit and vmap all functions needed
rng_reset = jax.jit(jax.vmap(env.mjx_reset, in_axes=(0, None)))
rng_step = jax.jit(jax.vmap(env.mjx_step, in_axes=(0, 0, None)))
rng_sample_uni_action = jax.jit(jax.vmap(env.sample_action_space))

# reset env
state = rng_reset(env_keys, traj)

step = 0
previous_time = time.time()
LOGGING_FREQUENCY = 100000
i = 0
while i < 100000:

    # step
    keys = jax.random.split(key, n_envs + 1)
    key, action_keys = keys[0], keys[1:]
    action = rng_sample_uni_action(action_keys)
    state = rng_step(state, action, traj)

    # parallel render
    env.mjx_render(state)

    step += n_envs

    # log speed (disable rendering for accurate speed measurement)
    if step % LOGGING_FREQUENCY == 0:
        current_time = time.time()
        print(f"{int(LOGGING_FREQUENCY / (current_time - previous_time))} steps per second.")
        previous_time = current_time

    i+=1

