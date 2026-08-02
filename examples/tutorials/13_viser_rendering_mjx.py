"""
Rendering many parallel Mjx environments in the browser with viser.

`mjx_render_viser()` is the viser counterpart of `mjx_render()`. It lays the environments out on
a square grid, exactly like the OpenGL viewer, and streams the whole batch to the browser.

Requires the optional dependencies:

    pip install loco-mujoco[viser]

Open the URL printed on startup (http://localhost:8080 by default). The "Scene" tab gets an
"Environment > Select" slider for picking which environment the overlays and the contact
visualization follow.
"""
import jax
import jax.numpy as jnp

from loco_mujoco import ImitationFactory


N_ENVS = 9

env, traj = ImitationFactory.make("MjxUnitreeH1",
                                  default_dataset_conf=dict(task="walk"),
                                  goal_type="GoalTrajMimicv2",
                                  goal_params=dict(visualize_goal=True),
                                  use_mjwarp=False,
                                  port=8080)

key = jax.random.key(0)
keys = jax.random.split(key, N_ENVS + 1)
key, env_keys = keys[0], keys[1:]

rng_reset = jax.jit(jax.vmap(env.mjx_reset))
rng_step = jax.jit(jax.vmap(env.mjx_step))
rng_sample_action = jax.jit(jax.vmap(env.sample_action_space))

state = rng_reset(env_keys)

while True:
    keys = jax.random.split(key, N_ENVS + 1)
    key, action_keys = keys[0], keys[1:]
    action = rng_sample_action(action_keys)
    state = rng_step(state, action)

    env.mjx_render_viser(state)
