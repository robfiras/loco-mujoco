# Architecture for developers

This page describes how the library is put together: where things live, which class calls which component
in which order, how state is carried through jitted code, and where models and datasets come from.
It is written for people who change the library. For how to *use* the components, see the
[user guide](../user_guide/modular_architecture.md).

## Package layout

| Package | Contents |
|---|---|
| `loco_mujoco/core/` | The simulation core. `mujoco_base.py` (CPU) and `mujoco_mjx.py` (MJX and MuJoCo Warp), and one sub-package per pluggable component: `observations/`, `reward/`, `control_functions/`, `domain_randomizer/`, `terrain/`, `initial_state_handler/`, `terminal_state_handler/`. Also `trajectory/` (trajectory dataclasses and handlers), `visuals/` (viewer, viser, USD export, video), `wrappers/` (Gymnasium, vectorisation, logging, observation history) and `utils/` (math, MuJoCo helpers, model download). |
| `loco_mujoco/environments/` | `base.py` with {class}`~loco_mujoco.environments.base.LocoEnv`, and one module per robot under `humanoids/` and `quadrupeds/`. Each robot has a CPU class and an `Mjx` class. |
| `loco_mujoco/task_factories/` | {class}`~loco_mujoco.task_factories.rl_factory.RLFactory` and {class}`~loco_mujoco.task_factories.imitation_factory.ImitationFactory`, and the dataset configuration classes. The factories are the entry point for users. |
| `loco_mujoco/algorithms/` | JAX implementations of PPO, GAIL, AMP and SuperMimic; `common/` holds networks, policies, dataclasses and the algorithm base class; `experimental/` holds BPTT-PPO, history-aware PPO, S2PG, SAC, TD3 and DAgger distillation. |
| `loco_mujoco/smpl/` | The retargeting pipeline: the SMPL parser, shape fitting, motion retargeting, and one configuration file per robot in `robot_confs/`. |
| `loco_mujoco/datasets/` | Dataset loaders (`humanoids/LAFAN1/`) and helpers for generating data (`data_generation/`). |
| `loco_mujoco/utils/` | Trajectory metrics (`metrics.py`), management of the path variables (`dataset.py`), the MyoSkeleton download, logging, running statistics and video conversion. Most command line tools are defined here. |

Outside the package: `examples/` (tutorials and training examples, the latter configured with Hydra),
`tests/`, `docs/` and `scripts/` (Blender rendering and figure scripts).

## Class hierarchy

```text
Mujoco                      loco_mujoco/core/mujoco_base.py
│   CPU simulation. Builds the model from an MjSpec, sets up all components,
│   implements reset() and step() on NumPy.
│
└── Mjx                     loco_mujoco/core/mujoco_mjx.py
    │   Adds mjx_reset() and mjx_step(): pure functions of an MjxState that run
    │   under jax.jit and jax.vmap, on MJX or on MuJoCo Warp (use_mjwarp=True).
    │
    └── LocoEnv             loco_mujoco/environments/base.py
        │   Adds the trajectory: processing, the trajectory handler and its state,
        │   end-of-clip termination, replay and dataset creation.
        │
        ├── BaseRobotHumanoid / BaseSkeleton / BaseRobotQuadruped
        │   └── UnitreeH1, UnitreeG1, Atlas, ...       mjx_enabled = False
        │       └── MjxUnitreeH1, MjxUnitreeG1, ...    mjx_enabled = True
        └── ...
```

Every environment inherits the whole chain, so the CPU class of a robot also has the MJX methods defined.
What decides the backend is the class attribute `mjx_enabled`: the constructor of
{class}`~loco_mujoco.environments.base.LocoEnv` calls the constructor of
{class}`~loco_mujoco.core.mujoco_mjx.Mjx` if it is `True`, and skips it and calls the constructor of
{class}`~loco_mujoco.core.mujoco_base.Mujoco` directly if it is `False`. Only the `Mjx` classes therefore
hold an MJX model (`env.sys`) and can run `mjx_reset` and `mjx_step`.

An `Mjx` robot class is a thin subclass of its CPU class. It sets `mjx_enabled = True`, chooses solver
options and a time step that suit MJX, and implements `_modify_spec_for_mjx`, which typically replaces
foot meshes by primitives and reduces the contact pairs.

A robot class provides what is specific to the embodiment: the path of the XML file
(`get_default_xml_file_path`), the default observations (`_get_observation_specification`) and actuators
(`_get_action_specification`), and a set of properties decorated with `@info_property`, such as
`root_body_name`, `upper_body_xml_name` and `root_height_healthy_range`. Components read these properties
through `env._get_all_info_properties()` instead of hard-coding robot names.

## Components and the registry

Each component type has a base class with a class-level dictionary `registered`, a `register()` class
method that enters the class under its class name, and `list_registered()`:

| Component | Base class | Constructor argument of the environment |
|---|---|---|
| Observation | {class}`~loco_mujoco.core.observations.base.Observation` | `observation_spec` |
| Goal | {class}`~loco_mujoco.core.observations.goals.Goal` | `goal_type`, `goal_params` |
| Reward | {class}`~loco_mujoco.core.reward.base.Reward` | `reward_type`, `reward_params` |
| Control function | {class}`~loco_mujoco.core.control_functions.base.ControlFunction` | `control_type`, `control_params` |
| Domain randomizer | {class}`~loco_mujoco.core.domain_randomizer.base.DomainRandomizer` | `domain_randomization_type`, `domain_randomization_params` |
| Terrain | {class}`~loco_mujoco.core.terrain.base.Terrain` | `terrain_type`, `terrain_params` |
| Initial-state handler | {class}`~loco_mujoco.core.initial_state_handler.base.InitialStateHandler` | `init_state_type`, `init_state_params` |
| Terminal-state handler | {class}`~loco_mujoco.core.terminal_state_handler.base.TerminalStateHandler` | `terminal_state_type`, `terminal_state_params` |
| Trajectory handler | {class}`~loco_mujoco.core.trajectory.handler.TrajectoryHandler` | `th_params` (with a `name` key) |

The built-in classes are registered in the `__init__.py` of their sub-package. The constructor of
{class}`~loco_mujoco.core.mujoco_base.Mujoco` looks each component up by name, for example
`Reward.registered[reward_type](self, **reward_params)`. Selecting components by string is what allows a
whole task to be described in a YAML file.

Environments use the same pattern: `Mujoco.registered_envs` maps class names to classes, each robot class is
registered in `environments/humanoids/__init__.py` or `environments/quadrupeds/__init__.py`, and the task
factories look the class up by name. The task factories themselves are registered in
`TaskFactory.registered`.

A component defined outside the library becomes available as soon as its module is imported and
`register()` has been called. See [Writing custom modules](../user_guide/custom_modules.md).

## How state is carried

Jitted code cannot mutate Python objects, so nothing that changes during an episode is stored on a
component or on the environment. There are three layers.

**{class}`~loco_mujoco.core.stateful_object.StatefulObject`** is the base class of all components that can
have state. It defines two methods, both of which do nothing by default:

- `init_state(env, key, model, data, backend, traj=None)` returns the initial state of the component, a
  `flax.struct.dataclass`. Components without state return `EmptyState()`.
- `reset_state(env, model, data, carry, backend, traj=None)` returns `(data, carry)`.

**The additional carry** collects these states. `AdditionalCarry` in `mujoco_base.py` is a
`flax.struct.dataclass` with the random key, the step counters, the last action, and one field per
component: `observation_states`, `reward_state`, `domain_randomizer_state`, `terrain_state`,
`init_state_handler_state`, `terminal_state_handler_state` and `control_func_state`. `MjxAdditionalCarry`
adds `final_observation` and `final_info`, which hold the last observation and info of an episode that was
reset automatically. `LocoCarry` in `environments/base.py` adds `traj_state`, the position in the reference
trajectory (clip number and step within the clip).

**{class}`~loco_mujoco.core.mujoco_mjx.MjxState`** is what `mjx_reset` returns and `mjx_step` consumes and
returns: the MJX `data`, the `observation`, `reward`, `absorbing`, `done`, the `additional_carry` and the
`info` dictionary. On the CPU the same carry exists but is stored on the environment
(`env._additional_carry`), because the CPU interface is stateful like Gymnasium's.

Every component method that may change state receives the carry and returns it. A component updates its
own field with `carry.replace(reward_state=new_state)`.

The **trajectory** is handled the same way since version 1.2.0: it is an argument (`traj`) of `reset`,
`step`, `mjx_reset` and `mjx_step`, not state of the environment. If it is omitted, the trajectory the
environment was created with (`env._traj`) is used. Trajectories are padded to fixed shapes, so a compiled
function can be called with a different set of clips without being recompiled.

## The life of a reset

The table lists what {meth}`~loco_mujoco.core.mujoco_base.Mujoco.reset` and
{meth}`~loco_mujoco.core.mujoco_mjx.Mjx.mjx_reset` do, in order. Calls marked *LocoEnv* are added by
{class}`~loco_mujoco.environments.base.LocoEnv` and only happen when a trajectory handler exists.

| Step | CPU: `reset(key, traj)` | MJX: `mjx_reset(key, traj)` |
|---|---|---|
| 1 | Resolve `traj` (fall back to `env._traj`) and convert it to NumPy. | Resolve `traj`; a NumPy trajectory raises an error. |
| 2 | `mj_resetData` and `mj_forward` on the stored `MjData`. | Start from `env._first_data`, the MJX data created at construction. |
| 3 | `_init_additional_carry`: call `init_state` of the observation container, the reward, the domain randomizer, the terrain, the initial-state handler, the control function and the terminal-state handler. *LocoEnv:* `th.init_state` for the trajectory state. | The same, with `backend=jnp`. |
| 4 | `_reset_carry`, see below. | `_mjx_reset_carry`, the same sequence with `backend=jnp`. |
| 5 | `mj_forward`, so that body and site poses match the `qpos` written in step 4. | (no separate forward call) |
| 6 | `obs_container.reset_state` resets stateful observations and goals. | The same. |
| 7 | `_create_observation` builds the observation vector. | `_mjx_create_observation`. |
| 8 | `_reset_info_dictionary`; the observation is returned. | `_mjx_reset_info_dictionary`; an `MjxState` with zero reward and `absorbing = done = False` is returned. |

`_reset_carry` and `_mjx_reset_carry` call the components in this order:

1. *LocoEnv:* `th.reset_state` chooses the clip and the start step.
2. `terminal_state_handler.reset`
3. `terrain.reset`
4. `init_state_handler.reset` writes `qpos` and `qvel`, for example from the trajectory.
5. `terrain.shift_to_env_origin` if the terrain places environments (`has_env_origins`), then
   `terrain.ground_robot` if the terrain asks for it (`grounds_robot`).
6. `domain_randomizer.reset` samples the randomization for the episode.
7. `reward_function.reset`
8. `control_func.reset`

The order matters: the trajectory state must exist before the initial-state handler reads the first frame
from it, and the terrain must be reset before the robot is placed on it.

## The life of a step

{meth}`~loco_mujoco.core.mujoco_base.Mujoco.step` and {meth}`~loco_mujoco.core.mujoco_mjx.Mjx.mjx_step`
run the same sequence. Each line is a method of the environment that a subclass can override; the MJX
variant has the prefix `_mjx_`.

| Step | Method (CPU name) | What it calls by default |
|---|---|---|
| 1 | | Store the action in `carry.last_action`. |
| 2 | `_preprocess_action` | `domain_randomizer.update_action` |
| 3 | `_simulation_pre_step` | `terrain.update`, then `domain_randomizer.update`. Both may change the model. |
| 4 | `_compute_action`, then the physics | `control_func.generate_action`, write `data.ctrl`, then `n_substeps` simulation steps. This pair is repeated `n_intermediate_steps` times. |
| 5 | `_simulation_post_step` | Nothing in the base class. *LocoEnv:* `th.update_state` advances the trajectory state. |
| 6 | `_create_observation` | Non-stateful observations are read in one indexing operation per type; stateful observations and goals are computed with `get_obs_and_update_state`. |
| 7 | `_step_finalize` | `domain_randomizer.update_observation` (observation noise) |
| 8 | `_update_info_dictionary` | Nothing by default. |
| 9 | `_is_absorbing` | `terminal_state_handler.is_absorbing` (MJX: `mjx_is_absorbing`) |
| 10 | `_reward` | `reward_function(obs, action, next_obs, absorbing, info, env, model, data, carry, backend, traj)` |
| 11 | `_is_done` | `absorbing` or the horizon is reached. *LocoEnv:* also when the end of the clip is reached or the goal reports done, if the goal or the reward needs the trajectory. |

Step 4 depends on the control function. By default the control function runs once per environment step and
MuJoCo is stepped `n_substeps` times with the same control signal. A control function with
`run_with_simulation_frequency` set, as `PDControl` has, is called before every
simulation step instead: the constructor then sets `n_intermediate_steps = n_substeps` and
`n_substeps = 1`.

The two backends differ after step 11:

- **CPU.** `step` returns `(observation, reward, absorbing, done, info)`. The caller resets the environment.
- **MJX.** `mjx_step` additionally marks an episode as done if the observation contains a NaN, and replaces
  NaNs by zero. It increments `cur_step_in_episode` and `global_step` in the carry. Then, if `done` is set,
  it resets the environment inside the step with `jax.lax.cond`: `_mjx_reset_in_step` restores the initial
  data, runs `_mjx_reset_carry` and `obs_container.reset_state`, stores the last observation and info of
  the ended episode in `carry.final_observation` and `carry.final_info`, and creates the first observation
  of the new episode. The returned state therefore has `done = True` together with the observation of the
  *next* episode. `init_state` is not called again; a component that needs fresh state per episode
  resets it in `reset`.

`mjx_step` has four optional arguments for algorithms that need more control: `auto_reset=False` returns
the terminal state itself, `physics_fn` replaces the simulation in step 4 by a user function (SuperMimic
uses it to run a learned dynamics model through the environment's own observation and reward functions),
`obs_ind` computes only a subset of the observation, and `skip_reward` skips steps 9 to 11.

Under MuJoCo Warp the contact and constraint buffers are shared by all worlds and cannot be reset for one
world, so `_mjx_reset_in_step` keeps them and resets everything else. With `refresh_kinematics=True` the
body and site poses are recomputed after the physics in step 4; otherwise they lag `qpos` by one simulation
step.

## Task factories

{meth}`RLFactory.make <loco_mujoco.task_factories.rl_factory.RLFactory.make>` looks up the environment class
and constructs it. {meth}`ImitationFactory.make <loco_mujoco.task_factories.imitation_factory.ImitationFactory.make>`
additionally loads the trajectories named by the dataset configurations, concatenates them, calls
{meth}`~loco_mujoco.environments.base.LocoEnv.process_trajectory`, which fits the trajectory to the model
and the control frequency and creates the trajectory handler, and returns `(env, traj)`. It sets
`TrajInitialStateHandler` and `RootPoseTrajTerminalStateHandler` as defaults. `RLFactory.make` returns
`(env, None)` so that both factories have the same signature.

## Where models and datasets come from

Neither robot models nor motion data are part of the Python package.

### Robot models

The XML files, meshes and textures live in the repository
[loco-mujoco-models](https://github.com/robfiras/loco-mujoco-models).
`loco_mujoco/core/utils/models.py` locates them in this order:

1. the directory given by `LOCOMUJOCO_ROBOT_MODELS_PATH`, read from the environment variable of that name
   or, if it is not set, from the variables file described below;
2. an installed `loco_mujoco_models` Python package;
3. the cache `~/.loco-mujoco-caches/robot_models/<revision>`, which is filled on first use by downloading
   the archive of one pinned commit of the models repository.

The commit is the constant `MODELS_REVISION` in `models.py`. A version of LocoMuJoCo therefore always
uses the models it was tested with. The download is protected by a file lock, so several processes can
start at once, and the cache directory appears under its final name only when it is complete.
`loco-mujoco-download-models` runs the download ahead of time. Environments obtain file paths through
`Mujoco.get_model_path(...)`.

The MyoSkeleton model has its own license and is not in the models repository.
`loco-mujoco-myomodel-init` asks the user to accept the license and clones a pinned commit of
[myolab/myo_model](https://github.com/myolab/myo_model) into the directory `LOCOMUJOCO_MODELS_PATH`
(default `~/.loco-mujoco-caches/models`).

### Datasets

| Source | Where it comes from | Where it is cached |
|---|---|---|
| Default (native) dataset | Downloaded per clip from the Hugging Face dataset repository [robfiras/loco-mujoco-datasets](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets), folder `DefaultDatasets/`. | The Hugging Face cache, and `LOCOMUJOCO_CONVERTED_DEFAULT_PATH` if set. |
| LAFAN1 | The same repository, folder `Lafan1/`. | The Hugging Face cache, and `LOCOMUJOCO_CONVERTED_LAFAN1_PATH` if set. |
| AMASS | Not redistributed. The user downloads AMASS and the SMPL model, and the clips are retargeted locally. | `LOCOMUJOCO_CONVERTED_AMASS_PATH` |
| Custom | A `Trajectory` object or file passed by the user. | not cached |

If a default or LAFAN1 clip is not hosted for an environment, the factory retargets it from a source
environment with the robot-to-robot pipeline and stores the result in the cache path. See
[datasets](../user_guide/datasets.md) and [retargeting](../user_guide/retargeting.md).

### The variables file

Paths are stored in `loco_mujoco/LOCOMUJOCO_VARIABLES.yaml`, next to the package's `__init__.py`. The file
is created on first import, is specific to the installation and is listed in `.gitignore`. It is written
with the command line tools (`loco-mujoco-set-amass-path`, `loco-mujoco-set-all-caches`,
`loco-mujoco-add-variable`, ...) and read with `loco_mujoco.get_variable(name)`; `loco-mujoco-show-variable
--all` prints it.

| Variable | Meaning |
|---|---|
| `LOCOMUJOCO_ROBOT_MODELS_PATH` | Local copy of the robot models. Also read from the process environment, which takes precedence. |
| `LOCOMUJOCO_MODELS_PATH` | Directory for additional models such as MyoSkeleton. |
| `LOCOMUJOCO_AMASS_PATH` | The downloaded AMASS dataset. |
| `LOCOMUJOCO_SMPL_MODEL_PATH` | The SMPL model files. |
| `LOCOMUJOCO_CONVERTED_AMASS_PATH`, `LOCOMUJOCO_CONVERTED_LAFAN1_PATH`, `LOCOMUJOCO_CONVERTED_DEFAULT_PATH` | Caches for retargeted trajectories. |
| `LOCOMUJOCO_CUSTOM_ROBOT_CONF_PATH` | List of directories with retargeting configurations for robots defined outside the library. |

`LOCOMUJOCO_ROBOT_MODELS_PATH` is the only variable that is also read from the process environment.

## Design rules

These rules keep the library consistent. A pull request that breaks one of them needs a reason.

**Support both backends with one implementation.** Component methods receive `backend`, which is `numpy`
or `jax.numpy`, and use it for all array operations. Where the two differ (for example in-place assignment
versus `.at[...].set(...)`), branch on `backend`. A component that cannot support a backend should fail at
construction with a clear message, as `Mjx` does for `RoughTerrain` under MuJoCo Warp.

**Keep jitted paths pure.** Everything reachable from `mjx_reset` and `mjx_step` is traced by JAX:

- no mutation of `self`, of globals or of arguments; return new values instead,
- no Python `if` or `for` on array values; use `jax.lax.cond`, `jax.lax.fori_loop`, `jnp.where`,
- array shapes must not depend on data; pad to a fixed size instead, as the trajectories are,
- randomness comes from the key in the carry, which is split and written back,
- no printing, file access or NumPy calls on traced values.

**Keep per-episode state in the carry.** Constructor arguments and quantities derived from the model at
construction (indices, limits, default poses) may be stored on `self`. Anything that changes after
construction belongs in the component's state dataclass.

**Do not hard-code robots in components.** Use names from the observation container and the
`@info_property` values of the environment, so that a component works for every embodiment.

**Changes to the MuJoCo model go through `MjSpec`.** Components that add geoms, sites or sensors set
`requires_spec_modification` and implement the corresponding hook (`modify_spec` for terrains,
`apply_spec_modifications` for goals); the environment recompiles the model at construction.

**New public names are registered and exported.** Register new components in the `__init__.py` of their
sub-package, and add a test (see [Testing](testing.md)) and a documentation entry.

**Heavy or licensed dependencies stay optional.** `torch`, `smplx`, `viser` and `usd-core` are imported
only where they are used and are installed through extras.
