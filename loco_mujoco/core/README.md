# Simple Mujoco Interface for RL Environments

A simple **unifying** interface that supports both **Mujoco-CPU**
and **Mujoco-Mjx** environments.

## Example
Here is a simple example to run a Mujoco-CPU environment:

```python
import jax
import numpy as np
from loco_mujoco.core import Mujoco, ObservationType
from loco_mujoco import PATH_TO_MODELS

# specify what observation you would like to retrieve from the xml
# --> checkout ObservationType to see what observations are supported by default
observation_spec = [ObservationType.JointPos("name_obs_1", "knee_angle_l"),
                    ObservationType.BodyVel("name_obs_2", "right_hip_yaw_link"),
                    ObservationType.SiteRot("name_obs_3", "left_knee_mimic")]   # --> concatenate more if needed

# specify the name of the actuators of the xml
action_spec = ["l_arm_shy_actuator", "hip_adduction_l_actuator"]    # --> use more motors if needed

# unitree_h1 xml for example
h1_model_path = PATH_TO_MODELS / "unitree_h1/h1.xml"

# define a simple Mujoco environment (CPU)
env = Mujoco(spec=str(h1_model_path),
             actuation_spec=action_spec,
             observation_spec=observation_spec,
             horizon=1000,
             gamma=0.99)

# get action dimensionality
action_dim = env.info.action_space.shape[0]

k = jax.random.key(0)
k, _k = jax.random.split(k)
env.reset(_k)
env.render()

while True:
    for i in range(500):
        env.step(np.random.randn(action_dim))
        env.render()
    k, _k = jax.random.split(k)
    env.reset(_k)
```

Similarily, we can run a Mujoco-Mjx environment:
```python
import time
import jax
import mujoco
from mujoco import MjSpec
from loco_mujoco.core import Mjx, ObservationType
from loco_mujoco import PATH_TO_MODELS


def _modify_spec_for_mjx(spec: MjSpec):
    """
    Mjx is bad in handling many complex contacts. To speed-up simulation significantly we apply
    some changes to the XML:
        1. Replace the complex foot meshes with primitive shapes. Here, one foot mesh is replaced with
           two capsules.
        2. Disable all contacts except the ones between feet and the floor.
        3. Reduce number of iterations and disable Euler damping.

    Args:
        spec (MjSpec): Mujoco specification.

    Returns:
        Modified Mujoco specification.

    """

    # --- 1. remove old feet and add new ones ---
    # remove original foot meshes
    for g in spec.geoms:
        if g.name in ["right_foot", "left_foot"]:
            spec.delete(g)

    # --- 2. Make all geoms have contype and conaffinity of 0 ---
    for g in spec.geoms:
        g.contype = 0
        g.conaffinity = 0

    # --- 3. add primitive foot shapes ---
    back_foot_attr = dict(type=mujoco.mjtGeom.mjGEOM_CAPSULE, quat=[1.0, 0.0, 1.0, 0.0], pos=[-0.03, 0.0, -0.05],
                          size=[0.015, 0.025, 0.0], rgba=[1.0, 1.0, 1.0, 0.2], contype=0, conaffinity=0)
    front_foot_attr = dict(type=mujoco.mjtGeom.mjGEOM_CAPSULE, quat=[1.0, 1.0, 0.0, 0.0], pos=[0.15, 0.0, -0.054],
                           size=[0.02, 0.025, 0.0], rgba=[1.0, 1.0, 1.0, 0.2], contype=0, conaffinity=0)

    r_foot_b = spec.find_body("right_ankle_link")
    r_foot_b.add_geom(name="right_foot1", **back_foot_attr)
    r_foot_b.add_geom(name="right_foot2", **front_foot_attr)

    l_foot_b = spec.find_body("left_ankle_link")
    l_foot_b.add_geom(name="left_foot1", **back_foot_attr)
    l_foot_b.add_geom(name="left_foot2", **front_foot_attr)

    # --- 4. Define specific contact pairs ---
    spec.add_pair(geomname1="floor", geomname2="right_foot1")
    spec.add_pair(geomname1="floor", geomname2="right_foot2")
    spec.add_pair(geomname1="floor", geomname2="left_foot1")
    spec.add_pair(geomname1="floor", geomname2="left_foot2")

    # --- 5. adapt options ---
    spec.option.iterations = 2
    spec.option.ls_iterations = 4
    spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_EULERDAMP

    return spec


# specify what observation you would like to retrieve from the xml
# --> checkout ObservationType to see what observations are supported by default
observation_spec = [ObservationType.JointPos("name_obs_1", "knee_angle_l"),
                    ObservationType.BodyVel("name_obs_2", "right_hip_yaw_link"),
                    ObservationType.SiteRot("name_obs_3", "left_knee_mimic")]   # --> concatenate more if needed

# specify the name of the actuators of the xml
action_spec = ["l_arm_shy_actuator", "hip_adduction_l_actuator"]    # --> use more motors if needed


# H1 model path
h1_model_path = PATH_TO_MODELS / "unitree_h1/h1.xml"

# load MjSpec
spec = MjSpec.from_file(str(h1_model_path))

# --- the sharp bit ---> modify the spec for mjx
spec = _modify_spec_for_mjx(spec)

# define a simple Mjx environment
mjx_env = Mjx(spec=spec,
              actuation_spec=action_spec,
              observation_spec=observation_spec,
              horizon=1000,
              gamma=0.99,
              n_envs=100)

action_dim = mjx_env.info.action_space.shape[0]

key = jax.random.key(0)
keys = jax.random.split(key, mjx_env.info.n_envs + 1)
key, env_keys = keys[0], keys[1:]

# jit and vmap all functions needed
rng_reset = jax.jit(jax.vmap(mjx_env.mjx_reset))
rng_step = jax.jit(jax.vmap(mjx_env.mjx_step))
rng_sample_uni_action = jax.jit(jax.vmap(mjx_env.sample_action_space))

# reset env
state = rng_reset(env_keys)

step = 0
previous_time = time.time()
LOGGING_FREQUENCY = 100000
i = 0
while i < 100000:

    keys = jax.random.split(key, mjx_env.info.n_envs + 1)
    key, action_keys = keys[0], keys[1:]
    action = rng_sample_uni_action(action_keys)
    state = rng_step(state, action)

    mjx_env.mjx_render(state)   # INFO: For speed check, comment this out and set n_envs to ~4000

    step += mjx_env.info.n_envs
    if step % LOGGING_FREQUENCY == 0:
        current_time = time.time()
        print(f"{int(LOGGING_FREQUENCY / (current_time - previous_time))} steps per second.")
        previous_time = current_time

    i += 1
```
Note that the Mjx environment can do everything the Mujoco environment can do, even running the simulation on CPU. Hence, 
we can still do a CPU rollout!

When using a Mjx environment, the `mjx_step` and `mjx_reset` functions are a jitted functions that run the simulation
on the GPU, while the `step` and `reset` functions are running the simulation on the CPU. It is important
to note that `mjx_step` does asyhronous resetting of each environment simiar to vector environments of Gymansium and 
stable-baselines3. `mjx_reset` resets all the environments at once.

## Rendering in the browser with viser

Next to the OpenGL viewer used by `render()` and `mjx_render()`, every environment has a
[viser](https://viser.studio)-based counterpart that serves the scene to a browser instead of
opening a local window. This is useful over SSH and on machines without a display. It requires
the optional dependencies:

```bash
pip install loco-mujoco[viser]
```

The methods are drop-in replacements and accept the same viewer parameters:

```python
env.render_viser()                          # instead of env.render()
mjx_env.mjx_render_viser(state)             # instead of mjx_env.mjx_render(state)
env.play_trajectory(n_episodes=3, viser=True)
```

The server URL (`http://localhost:8080` by default) is printed on startup; configure it with the
`host`, `port` and `verbose` environment kwargs. If the port is already taken, viser falls back to
the next free one and the viewer warns about it.

The browser panel exposes the same controls as the OpenGL viewer — geom and site group toggles,
the three camera modes, pause and run speed — plus mjviser's contact, tendon, inertia and
reference-frame overlays. The goal markers carried in `carry.user_scene` (velocity arrows, mimic
targets, ghost robots) are drawn as well. `mjx_render_viser` lays the parallel environments out on
the same grid as `mjx_render` and adds an environment selector to the panel.

### Appearance

The look is configurable through environment kwargs, and the ground plane and sky can also be
recolored live from the *Visualization* tab in the browser:

| kwarg | default | meaning |
|---|---|---|
| `floor_color` | `(28, 30, 34)` | RGB of the ground plane, or `"model"` to take the plane's material and texture colors, or `None` for mjviser's default |
| `sky` | `"model"` | Background gradient read from the model's skybox, a `(top_rgb, bottom_rgb)` pair, or `None` |
| `logo` / `logo_width` | LocoMuJoCo banner, `170` | Logo pinned to the top left. A URL is fetched by the browser, a local path is embedded, `None` disables it |

### Caveats

* There is no local framebuffer, so frames are read back from a connected browser via
  `client.get_render()`. `record=True` therefore only produces a non-empty video while a browser
  is open, and `render_viser(record=False)` returns a zero-filled array rather than the displayed
  image.
* The sky is a flat backdrop rather than a real cube map, so it does not rotate with the camera.
* The default logo is loaded from GitHub by the browser; pass a local path when working offline.
