# Simulation backends

Every LocoMuJoCo environment can be simulated by up to three physics backends. They share one environment
definition: observations, rewards, control functions, terminal conditions and domain randomization are written
once against an array-library argument (NumPy or `jax.numpy`) and run unchanged on all of them.

| | MuJoCo | MJX | MuJoCo Warp |
|---|---|---|---|
| Physics engine | [MuJoCo](https://mujoco.readthedocs.io) (C) | [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html), MuJoCo written in JAX | [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), called through the MJX API |
| Runs on | CPU | GPU, or CPU through JAX | NVIDIA GPU |
| Environments | One per process | Thousands in parallel with `jax.vmap` | Thousands in parallel with `jax.vmap` |
| Interface | Stateful: `env.reset()`, `env.step()` | Functional: `env.mjx_reset()`, `env.mjx_step()` | Functional: `env.mjx_reset()`, `env.mjx_step()` |
| Environment name | `UnitreeH1` | `MjxUnitreeH1` | `MjxUnitreeH1` with `use_mjwarp=True` |
| Collision model | Full model | Reduced to foot–floor contacts by default | Reduced by default; the full model is affordable |
| Differentiable physics | No | Yes | No |

## Which backend to use

MuJoCo (CPU)
: Use it for prototyping, debugging, replaying datasets, interactive rendering and evaluating a trained policy
  on the unmodified robot model. It is also the backend of the [Gymnasium interface](gymnasium.md).

MJX
: Use it for training with the bundled JAX algorithms on models with few contacts, and when you need gradients
  through the simulation. It also runs on the CPU, which is slow but useful for testing a training script on
  a machine without a GPU.

MuJoCo Warp
: Use it for large-scale training on an NVIDIA GPU. It handles many contacts and constraints much better than
  MJX, so it is the backend of choice for full collision models and for the highest throughput. See
  [throughput](#throughput) for measured numbers.

## Selecting a backend

The backend is chosen when the environment is created, by the **name** of the environment and one keyword
argument.

`````{tab-set}
````{tab-item} MuJoCo (CPU)
```python
from loco_mujoco import ImitationFactory

env, traj = ImitationFactory.make("UnitreeH1", default_dataset_conf=dict(task="squat"))
```
````

````{tab-item} MJX
```python
from loco_mujoco import ImitationFactory

env, traj = ImitationFactory.make("MjxUnitreeH1", default_dataset_conf=dict(task="squat"))
```
````

````{tab-item} MuJoCo Warp
```python
from loco_mujoco import ImitationFactory

n_envs = 4096
env, traj = ImitationFactory.make("MjxUnitreeH1", default_dataset_conf=dict(task="squat"),
                                  use_mjwarp=True,
                                  naconmax=64 * n_envs,   # contact buffer of all environments together
                                  njmax=256)              # constraint buffer of one environment
```
````
`````

Every embodiment is registered under both names, see [task factories](task_factories.md). An `Mjx...`
environment is a subclass of the CPU environment with the same observation and action space. It differs in
three ways:

- It builds the MJX (or Warp) model next to the MuJoCo model and provides `mjx_reset` and `mjx_step`.
- Its collision model is simplified. MJX is slow with many contacts, so by default only a few primitive foot
  geometries collide with the floor. The humanoid environments accept `mjx_reduced_contacts=False` to keep
  the collision geometry of the original model, which is practical with MuJoCo Warp.
- Its solver settings are cheaper. Most humanoids use two solver iterations, four line-search iterations and
  no Euler damping, at a simulation step of 2 ms with 5 substeps instead of 1 ms with 10, so that both variants
  run at a control frequency of 100 Hz. The quadrupeds use 8 and 12 iterations. `env.simulation_dt` and
  `env.dt` give the simulation and the control step of an environment. Override the settings with `timestep`,
  `n_substeps` and `model_option_conf`.

Plain environments such as `UnitreeH1` have no MJX model, and calling `mjx_reset` on them fails.

## The same loop on each backend

The loops below reset the environment and apply random actions. They run headless and can be pasted into a
Python session.

``````{tab-set}
`````{tab-item} MuJoCo (CPU)
```python
import numpy as np
import jax
from loco_mujoco import ImitationFactory

env, traj = ImitationFactory.make("UnitreeH1", default_dataset_conf=dict(task="squat"))
action_dim = env.info.action_space.shape[0]

obs = env.reset(jax.random.key(0))
for i in range(1000):
    action = np.random.randn(action_dim)
    obs, reward, absorbing, done, info = env.step(action)
    if absorbing or done:
        obs = env.reset()
```

The environment owns the simulation state. `step` returns the observation, the reward, a flag for an
absorbing (terminal) state, a flag for the end of the episode, and an info dictionary. Call `env.render()`
after a step to open the viewer.
`````

`````{tab-item} MJX
```python
import jax
from loco_mujoco import ImitationFactory

env, traj = ImitationFactory.make("MjxUnitreeH1", default_dataset_conf=dict(task="squat"))

n_envs = 1024
key = jax.random.key(0)
keys = jax.random.split(key, n_envs + 1)
key, env_keys = keys[0], keys[1:]

# vectorize over environments and compile; the trajectory is shared by all environments
mjx_reset = jax.jit(jax.vmap(env.mjx_reset, in_axes=(0, None)))
mjx_step = jax.jit(jax.vmap(env.mjx_step, in_axes=(0, 0, None)))
sample_action = jax.jit(jax.vmap(env.sample_action_space))

state = mjx_reset(env_keys, traj)
for i in range(1000):
    keys = jax.random.split(key, n_envs + 1)
    key, action_keys = keys[0], keys[1:]
    action = sample_action(action_keys)
    state = mjx_step(state, action, traj)    # environments that are done reset themselves
```

The state is an explicit value that goes into `mjx_step` and comes out again. The number of parallel
environments is the number of random keys passed to `mjx_reset`.
`````

`````{tab-item} MuJoCo Warp
```python
import jax
from loco_mujoco import ImitationFactory

n_envs = 1024
env, traj = ImitationFactory.make("MjxUnitreeH1", default_dataset_conf=dict(task="squat"),
                                  use_mjwarp=True, naconmax=64 * n_envs, njmax=256)

key = jax.random.key(0)
keys = jax.random.split(key, n_envs + 1)
key, env_keys = keys[0], keys[1:]

mjx_reset = jax.jit(jax.vmap(env.mjx_reset, in_axes=(0, None)))
mjx_step = jax.jit(jax.vmap(env.mjx_step, in_axes=(0, 0, None)))
sample_action = jax.jit(jax.vmap(env.sample_action_space))

state = mjx_reset(env_keys, traj)
for i in range(1000):
    keys = jax.random.split(key, n_envs + 1)
    key, action_keys = keys[0], keys[1:]
    action = sample_action(action_keys)
    state = mjx_step(state, action, traj)
```

Only the construction of the environment differs from MJX. The buffer sizes must be large enough for the
number of environments that is run, see [below](#mujoco-warp).
`````
``````

`examples/tutorials/02_creating_mjx_env.py` is the MJX loop with parallel rendering and a throughput printout:

```{literalinclude} ../../examples/tutorials/02_creating_mjx_env.py
:language: python
```

## The functional interface

{meth}`~loco_mujoco.core.mujoco_mjx.Mjx.mjx_reset` and {meth}`~loco_mujoco.core.mujoco_mjx.Mjx.mjx_step` are
pure functions of a single environment. This makes them compatible with the JAX transformations: `jax.vmap`
turns them into functions of a batch of environments, `jax.jit` compiles them, and `jax.lax.scan` rolls them
out without returning to Python.

| Function | Arguments | Returns |
|---|---|---|
| `env.mjx_reset(key, traj=None)` | A JAX random key and optionally the trajectory | An {class}`~loco_mujoco.core.mujoco_mjx.MjxState` |
| `env.mjx_step(state, action, traj=None)` | The current state, an action and optionally the trajectory | The next `MjxState` |
| `env.sample_action_space(key)` | A JAX random key | A uniformly sampled action |

Everything an environment knows about itself is in the {class}`~loco_mujoco.core.mujoco_mjx.MjxState`:

| Field | Content |
|---|---|
| `data` | The MJX `Data` structure with the simulation state (`qpos`, `qvel`, body poses, ...) |
| `observation` | The observation of the current step |
| `reward` | The reward of the last transition |
| `absorbing` | True if the state is terminal, for example because the robot fell |
| `done` | True if the episode ended, by termination, by reaching the horizon or by reaching the end of the reference motion |
| `info` | The info dictionary |
| `additional_carry` | The state of all stateful components: the random key, the step counters, the last action, the [trajectory state](trajectories.md) and one state entry each for observations, reward, domain randomization, terrain, control function and the initial and terminal state handlers |

After `jax.vmap`, every leaf of the state has a leading dimension of size `n_envs`, for example
`state.observation.shape == (n_envs, obs_dim)`.

### Automatic reset

`mjx_step` resets an environment as soon as it is done, as vectorized environments usually do. The returned
state then already holds the first observation of the new episode, while `reward`, `absorbing` and `done`
still describe the transition that ended the old one. The last observation of the old episode is kept in
`state.additional_carry.final_observation`. Pass `auto_reset=False` to `mjx_step` to receive the terminal state
itself and reset it yourself.

A step whose observation contains a NaN, for example after the simulation diverged, is marked as done and the
NaN entries are replaced by zeros, so one unstable environment does not corrupt the batch.

### Passing the trajectory

The trajectory argument is optional. If it is omitted, the trajectory stored in the environment is used and
becomes a constant of the compiled function. Passing it explicitly, with `None` in `in_axes` so that it is
shared by all environments, keeps the motion data out of the compiled code and allows swapping it for another
trajectory of the same shape without recompiling. See [trajectories](trajectories.md).

### Using the CPU interface of an Mjx environment

An `Mjx...` environment keeps the stateful `reset`, `step`, `render` and `play_trajectory` methods of the CPU
environment. They run MuJoCo on the CPU with the simplified `Mjx` model, which is convenient for a quick look
at a single environment without compiling anything. To render the states of a batch of environments, use
`env.mjx_render(state)`, see [rendering](rendering.md).

### Wrappers

The bundled algorithms do not call `mjx_reset` and `mjx_step` directly but go through the wrappers in
{mod}`loco_mujoco.core.wrappers.mjx`, which add an `(observation, state)` return convention, vectorization,
episode statistics and observation stacking. Use them as a starting point if you write your own training loop.

## MuJoCo Warp

`use_mjwarp=True` replaces the MJX physics by MuJoCo Warp while keeping the MJX interface, so the loop and the
state stay the same. Warp allocates fixed-size buffers that are shared by all parallel environments:

| Argument | Meaning |
|---|---|
| `naconmax` | Maximum number of contacts of **all** environments together. Scale it with the number of environments. `nconmax` is accepted as an older name for the same argument. |
| `njmax` | Maximum number of constraints of **one** environment. |
| `naccdmax` | Maximum number of contacts found by the convex collision routine, for example between a mesh and a heightfield, of all environments together. Defaults to `naconmax`. |

Contacts beyond these limits are dropped, so bodies can pass through each other or through the floor. Increase
the limits if that happens. A model with reduced contacts needs few contacts per environment; a full
collision model on uneven ground needs many more. `examples/speed_test.py` uses 64 contacts and 256
constraints per environment as a starting point.

Limitations of the Warp backend:

- It requires an NVIDIA GPU with CUDA.
- Terrains that generate a heightfield per environment at run time are rejected, because Warp shares the
  heightfield data between all environments. This applies to `RoughTerrain`; the constructor raises a
  `ValueError`. Use `HeightmapTerrain` instead, see [terrain](terrain.md).
- JAX cannot differentiate through the Warp physics step. Algorithms that need gradients of the dynamics have to
  use MJX. [SuperMimic](../training/supermimic.md) only differentiates through the kinematics and runs them on
  a JAX copy of the model, so it works with both backends.
- The contact and constraint buffers of one environment cannot be reset on their own. LocoMuJoCo handles this in
  the automatic reset; the buffers are rebuilt by the next step.

```{note}
After a step, quantities that are derived from the joint state, such as body positions, belong to the state
before the last simulation substep. This is how MuJoCo's integrator works and applies to all backends. With
`refresh_kinematics=True`, an `Mjx...` environment recomputes them after every control step at a small cost.
```

## Throughput

The table gives environment steps per second of the three backends on one robot, the 29-joint Unitree G1
of [LocoMuJoCo Real](../sim2real/index.md), with [mjlab](https://arxiv.org/abs/2601.22074) on the same robot
as a reference. An environment step is the full step: physics, control, observation, reward and termination.

| Collision model | Framework | Backend | 1 env | 8 envs | 1,024 envs | 4,096 envs | 8,192 envs |
|---|---|---|---:|---:|---:|---:|---:|
| Full self-collision | mjlab | MuJoCo Warp | 211 | | 135,086 | 315,642 | 391,556 |
| | LocoMuJoCo | MuJoCo (CPU) | 2,147 | 15,805 | | | |
| | LocoMuJoCo | MJX | 169 | | 12,667 | 9,388 | 8,974 |
| | LocoMuJoCo | MuJoCo Warp | 530 | | **251,472** | **442,715** | **487,553** |
| Feet–floor only | mjlab | MuJoCo Warp | 212 | | 136,799 | 319,886 | 401,512 |
| | LocoMuJoCo | MuJoCo (CPU) | 2,209 | 16,169 | | | |
| | LocoMuJoCo | MJX | 183 | | 61,097 | 79,935 | 83,899 |
| | LocoMuJoCo | MuJoCo Warp | 533 | | **252,548** | **447,007** | **493,827** |
| MJX-tuned model | LocoMuJoCo | MJX | 386 | | 232,833 | 343,052 | 324,378 |
| | LocoMuJoCo | MuJoCo Warp | 705 | | 358,874 | 694,743 | 770,061 |

An empty cell is a configuration that was not measured. The CPU backend runs one environment per process, so
its 8-environment column uses eight processes.

What the numbers show:

- **MuJoCo Warp is the fastest GPU backend.** With full self-collision it reaches about 490,000 steps per
  second, 20 to 54 times more than MJX. MJX evaluates all 502 candidate geometry pairs of this robot in every
  step, whereas the broad phase of MuJoCo Warp discards distant pairs.
- **MJX depends on the collision model.** Keeping only the contacts between the 14 foot spheres and the
  floor makes MJX 4.8 to 9.3 times faster, but changes MuJoCo Warp, mjlab and the CPU backend by less than 3 %.
  This is why the `Mjx...` environments reduce their contacts by default.
- **MJX can be fast with a model made for it.** The last two rows use the Unitree G1 that ships with
  LocoMuJoCo (`MjxUnitreeG1`), with simple collision geometry and the MJX solver settings described
  [above](#selecting-a-backend). MJX then reaches about 340,000 steps per second.
- **One environment is fastest on the CPU.** The GPU backends are limited by latency when they run a single
  environment. The CPU backend scales by a factor of 7.4 from one to eight processes.
- **Compared with mjlab**, the MuJoCo Warp backend of LocoMuJoCo is 1.9 times faster at 1,024 environments,
  1.4 times at 4,096 and 1.25 times at 8,192. These are environment-level numbers, so they include the logic of
  each task, which is heavier in mjlab's velocity task. As shipped, with its domain randomization and the
  `implicitfast` integrator, mjlab reaches 98,000, 244,000 and 313,000 steps per second at the same sizes.

```{dropdown} How the numbers were measured
:icon: info

- **Hardware.** One NVIDIA RTX 4090 (24 GB) and an Intel Core i9-13900KF.
- **Software.** LocoMuJoCo 1.2.0 with MJX 3.11.0, which bundles its own copy of MuJoCo Warp, Warp 1.14.0,
  JAX 0.10.1 and LocoMuJoCo Real 0.1.0. mjlab 1.6.1 with MuJoCo Warp 3.11.0, Warp 1.19.0 and PyTorch 2.14.1.
  All GPU libraries are their CUDA 12 builds.
- **Tasks.** mjlab runs its flat-ground G1 velocity task with domain randomization turned off, LocoMuJoCo its
  velocity task with the same robot. The two G1 models have identical bodies, masses and actuators, and 34
  collision geometries with self-collision.
- **Simulation settings.** MuJoCo 3.11, a 50 Hz control rate with four physics substeps of 5 ms, the Euler
  integrator (mjlab's default is `implicitfast`) and the same solver settings in all configurations except the
  MJX-tuned model. The feet–floor variant keeps only the contacts between the 14 foot spheres and the floor,
  set identically in both frameworks (mjlab's `FEET_ONLY_COLLISION`).
- **Protocol.** mjlab's benchmark protocol: zero actions, 50 warm-up steps and 200 timed environment steps
  (2,000 on the CPU). Each entry is the mean of five measurements in each of two passes run in opposite order;
  the passes agree within 2 % from 1,024 environments on and within 5 % otherwise.
- **Scope.** The measurements cover one embodiment on one GPU.
```

To measure the backends on your own hardware, run `examples/speed_test.py`. It sweeps backends, collision
models and numbers of environments, and writes the results to a CSV file:

```bash
python examples/speed_test.py --env UnitreeH1 --n-envs 1024 4096
```

It uses random actions and the robot models of LocoMuJoCo, so its numbers are not directly comparable with the
table above.

## The low-level classes

The embodiment classes derive from two general classes that can be used on their own to wrap any MuJoCo model
in the same interface: {class}`~loco_mujoco.core.mujoco_base.Mujoco` for the CPU and
{class}`~loco_mujoco.core.mujoco_mjx.Mjx` for MJX and Warp. They take a model, a list of actuator names and a
list of [observations](observations.md), and accept the same component arguments (`reward_type`,
`terminal_state_type`, ...) as the embodiment classes.

```python
import jax
import numpy as np
from loco_mujoco.core import Mujoco, ObservationType

# what to observe: each entry names the observation and the element of the XML it reads
observation_spec = [ObservationType.JointPos("q_knee_left", "knee_angle_l"),
                    ObservationType.BodyVel("v_right_hip", "right_hip_yaw_link"),
                    ObservationType.SitePos("p_left_foot", "left_foot_mimic")]

# what to control: names of actuators in the XML
actuation_spec = ["knee_angle_l_actuator", "knee_angle_r_actuator"]

xml_path = Mujoco.get_model_path("unitree_h1", "h1.xml")

env = Mujoco(spec=xml_path, actuation_spec=actuation_spec, observation_spec=observation_spec,
             horizon=1000, gamma=0.99)

obs = env.reset(jax.random.key(0))
obs, reward, absorbing, done, info = env.step(np.zeros(env.info.action_space.shape[0]))
```

{class}`~loco_mujoco.core.mujoco_mjx.Mjx` takes the same arguments plus the Warp arguments described above
and provides the functional interface. `spec` can also be an `MjSpec` object, which is how a model is modified
before it is compiled, for example to reduce its contacts for MJX:

```python
import mujoco
from loco_mujoco.core import Mjx

spec = mujoco.MjSpec.from_file(xml_path)
for geom in spec.geoms:              # disable all collisions as an example
    geom.contype = 0
    geom.conaffinity = 0

mjx_env = Mjx(spec=spec, actuation_spec=actuation_spec, observation_spec=observation_spec,
              horizon=1000, gamma=0.99)

keys = jax.random.split(jax.random.key(0), 8)
state = jax.jit(jax.vmap(mjx_env.mjx_reset))(keys)     # 8 parallel environments
```

Trajectory handling, the dataset interface and the embodiment properties are added by
{class}`~loco_mujoco.environments.base.LocoEnv`, which derives from `Mjx`.

## Practical tips

- **Compilation.** The first call of a jitted function compiles it, which takes from several seconds to a few
  minutes for a full training loop. Measure throughput only after the first call.
- **XLA flag.** Setting `XLA_FLAGS="--xla_gpu_triton_gemm_any=True"` before the first JAX computation speeds
  up MJX on some GPUs by about 30 %, as in the tutorial above.
- **GPU memory.** JAX reserves 75 % of the GPU memory on start-up. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` if
  another process or library, such as PyTorch during [retargeting](retargeting.md), needs the same GPU.
- **Rendering.** `mjx_render` copies states to the CPU and draws them, which dominates the run time. Render
  only for inspection, and create training environments with `headless=True`.
- **Console output.** Every environment prints a table of its observation space on creation. Disable it with
  `print_obs_summary=False`.
- **Testing without a GPU.** Set `JAX_PLATFORMS=cpu` to run MJX code on the CPU with a handful of
  environments.

## See also

- [Task factories](task_factories.md) for creating environments and passing arguments to them.
- [Trajectories](trajectories.md) for how reference motions are passed to `mjx_reset` and `mjx_step`.
- [Modular architecture](modular_architecture.md) for how components support both array libraries.
- [Rendering](rendering.md) for viewing single and parallel environments.
- [Training](../training/index.md) for the algorithms built on the functional interface.
- API reference: {mod}`loco_mujoco.core.mujoco_base`, {mod}`loco_mujoco.core.mujoco_mjx`,
  {mod}`loco_mujoco.core.wrappers.mjx`.
