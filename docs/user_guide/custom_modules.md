# Writing custom modules

Every module of an environment can be replaced by your own: observations, goals, rewards, control
functions, initial and terminal state handlers, domain randomizers and terrains. The procedure is the same
for all of them:

1. Derive a class from the base class of the module kind and implement its methods.
2. Call `register()` on the class.
3. Select it by its class name when you create the environment.

```python
from loco_mujoco import RLFactory
from loco_mujoco.core.reward import Reward

class NominalPoseReward(Reward):

    def __init__(self, env, scale=1.0, **kwargs):
        super().__init__(env, **kwargs)
        self._scale = scale

    def __call__(self, state, action, next_state, absorbing, info, env, model, data, carry, backend, traj=None):
        reward = backend.exp(-self._scale * backend.linalg.norm(data.qpos[7:] - model.qpos0[7:]))
        return reward, carry

NominalPoseReward.register()

env, _ = RLFactory.make("UnitreeG1", reward_type="NominalPoseReward", reward_params=dict(scale=0.5))
```

A registered module can be used with every environment and from YAML configurations, exactly like the
built-in ones. [Modular architecture](modular_architecture.md) explains the mechanism. This page lists what
each kind of module has to implement.

## Rules for all modules

A module runs unchanged on the CPU backend and on MJX, where its methods are compiled with `jax.jit` and
mapped over all parallel environments with `jax.vmap`. This works if the module follows a few rules:

- **Accept the standard arguments.** Every method receives `env`, `model`, `data`, `carry` and `backend`,
  and a last argument `traj=None` with the reference trajectory. The environment passes all of them, so a
  method has to accept `traj` even if it does not use it.
- **Compute with `backend`.** It is `numpy` on the CPU and `jax.numpy` on MJX. Write
  `backend.sum(backend.square(x))` instead of using one of the two libraries directly.
- **Do not store anything in `self` after construction.** The constructor may precompute constants such as
  joint indices (as NumPy arrays) and store parameters. Whatever changes during an episode belongs in the
  carry: define a `flax.struct.dataclass`, return an instance from `init_state`, and update it with
  `carry.replace(...)`. The state must keep its structure, shapes and dtypes. `init_state` runs when the
  carry is created. To start every episode with a fresh state, also assign it in `reset`.
- **Return the carry.** Almost every method returns its result together with the carry, changed or not.
- **Branch on the backend for in-place writes and random numbers.** `MjData` and NumPy arrays are written
  in place, and NumPy's global random state is used. MJX data is immutable and is updated with
  `data.replace(...)` and `array.at[...].set(...)`, and random numbers come from `carry.key`, which has to
  be split and written back.
- **No Python control flow on simulation values in JAX code.** Use `backend.where`, or `jax.lax.cond` and
  `jax.lax.select` in a JAX branch.

A module that you only use on the CPU can ignore the JAX rules and work with NumPy alone.

Modules are registered under their class name. Registering a control function, initial state handler,
domain randomizer or terrain twice raises an error, which matters when you run a notebook cell again. The
other kinds ignore a second registration.

## Observations

There are two base classes. Use {class}`~loco_mujoco.core.observations.base.Observation` for a quantity
that can be read from one array of `MjData`, and
{class}`~loco_mujoco.core.observations.base.StatefulObservation` for everything that has to be computed.

| Member | `Observation` | `StatefulObservation` |
|---|---|---|
| `dim` | Class or instance attribute with the number of entries | The same |
| `_init_from_mj(env, model, data, current_obs_size)` | Sets `min`, `max`, `obs_ind` (indices in the observation vector), `data_type_ind` (indices in the data array) and `_initialized_from_mj = True` | Sets `min`, `max`, `obs_ind` and `_initialized_from_mj = True` |
| `data_type()` | Class method that returns the name of the array of `MjData`, for example `"xipos"` | not used |
| `get_obs_and_update_state(env, model, data, carry, backend, traj=None)` | not used | Returns the observation and the carry |
| `init_state(env, key, model, data, backend, traj=None)` | not used | Optional. Returns the initial state, which is stored in `carry.observation_states` under the name of the observation. |
| `reset_state(env, model, data, carry, backend, traj=None)` | not used | Optional. Called at every reset, also at the automatic resets inside `mjx_step`. Returns the data and the carry. |

A simple observation of the center of mass of one body:

```python
import numpy as np
from loco_mujoco.core.observations import Observation

class BodyCOMPos(Observation):
    dim = 3

    def __init__(self, obs_name, xml_name, **kwargs):
        self.xml_name = xml_name
        super().__init__(obs_name, **kwargs)

    def _init_from_mj(self, env, model, data, current_obs_size):
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.data_type_ind = np.array(self.to_list(data.body(self.xml_name).id))
        self.obs_ind = np.arange(current_obs_size, current_obs_size + self.dim)
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls):
        return "xipos"
```

The environment reads all observations of this type in one indexing operation, `data.xipos[indices]`.
A stateful observation that returns a moving average of the same quantity keeps the average in the carry:

```python
import jax
from flax import struct
from loco_mujoco.core.observations import StatefulObservation

@struct.dataclass
class MovingAverageState:
    moving_average: jax.Array

class BodyCOMPosMovingAverage(StatefulObservation):
    dim = 3

    def __init__(self, obs_name, xml_name, **kwargs):
        self.xml_name = xml_name
        super().__init__(obs_name, **kwargs)

    def _init_from_mj(self, env, model, data, current_obs_size):
        self.min, self.max = [-np.inf] * self.dim, [np.inf] * self.dim
        self.body_id = data.body(self.xml_name).id
        self.obs_ind = np.arange(current_obs_size, current_obs_size + self.dim)
        self._initialized_from_mj = True

    def init_state(self, env, key, model, data, backend, traj=None):
        return MovingAverageState(moving_average=backend.zeros(self.dim))

    def get_obs_and_update_state(self, env, model, data, carry, backend, traj=None):
        obs_states = carry.observation_states
        state = getattr(obs_states, self.name)
        average = 0.9 * state.moving_average + 0.1 * data.xipos[self.body_id]
        obs_states = obs_states.replace(**{self.name: state.replace(moving_average=average)})
        return backend.ravel(average), carry.replace(observation_states=obs_states)
```

`register()` also adds the class to {class}`~loco_mujoco.core.observations.base.ObservationType`, so that it
is used like a built-in type:

```python
from loco_mujoco.core import ObservationType

BodyCOMPos.register()
BodyCOMPosMovingAverage.register()

observation_spec = [
    ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
    ObservationType.BodyCOMPos("pelvis_com", xml_name="pelvis"),
    ObservationType.BodyCOMPosMovingAverage("pelvis_com_avg", xml_name="pelvis"),
]
env, _ = RLFactory.make("UnitreeG1", observation_spec=observation_spec)
```

A **goal** is a stateful observation that derives from {class}`~loco_mujoco.core.observations.goals.Goal`.
Its constructor receives the info properties of the environment as `info_props`, and in addition to the
methods above it implements the properties `dim` and `has_visual`. It can end an episode through `is_done`
(CPU) and `mjx_is_done` (MJX), and it declares with the property `requires_trajectory` whether it reads the
reference trajectory. It is selected with `goal_type`.

## Rewards

Derive from {class}`~loco_mujoco.core.reward.base.Reward`.

| Method | Required | Purpose |
|---|---|---|
| `__call__(state, action, next_state, absorbing, info, env, model, data, carry, backend, traj=None)` | yes | Returns the reward and the carry. `state` and `next_state` are the observations before and after the step, `data` is the simulation state after the step. |
| `init_state(env, key, model, data, backend, traj=None)` | no | Initial value of `carry.reward_state` |
| `reset(env, model, data, carry, backend, traj=None)` | no | Called at every reset. Returns the data and the carry. |
| `requires_trajectory` | no | Property. Return `True` if the reward reads the reference trajectory. The episode then ends with the clip. |

The example at the top of the page is a complete reward. The base class constructor stores the info
properties of the environment in `self._info_props`, for example the list of mimic sites. A reward with
state is shown in [Modular architecture](modular_architecture.md#stateless-modules-and-the-carry).

## Control functions

Derive from {class}`~loco_mujoco.core.control_functions.base.ControlFunction`.

| Method | Required | Purpose |
|---|---|---|
| `__init__(env, ...)` | yes | Has to call `super().__init__(env, low, high)` with the bounds of the action space |
| `generate_action(env, action, model, data, carry, backend, traj=None)` | yes | Returns the control for the actuators and the carry |
| `run_with_simulation_frequency` | no | Property. Return `True` to be called before every physics step instead of once per environment step. |
| `init_state`, `reset` | no | State in `carry.control_func_state` |

```python
from loco_mujoco.core.control_functions import ControlFunction

class ScaledTorqueControl(ControlFunction):

    def __init__(self, env, scale=0.5, **kwargs):
        self._scale = scale
        low, high = self._get_actuator_limits(env._action_indices, env._model)
        super().__init__(env, low, high, **kwargs)

    def generate_action(self, env, action, model, data, carry, backend, traj=None):
        return self._scale * action, carry

ScaledTorqueControl.register()
env, _ = RLFactory.make("UnitreeG1", control_type="ScaledTorqueControl", control_params=dict(scale=0.5))
```

To change the built-in PD controller, derive from
{class}`~loco_mujoco.core.control_functions.pd.PDControl` and extend its state
`PDControlState` instead of starting from the base class.

## Initial and terminal state handlers

An initial state handler derives from
{class}`~loco_mujoco.core.initial_state_handler.base.InitialStateHandler` and implements one method,
`reset(env, model, data, carry, backend, traj=None)`, which writes the first state of an episode and returns
the data and the carry. The example shows the two ways of writing to the simulation state and of drawing
random numbers:

```python
import jax
import numpy as np
from loco_mujoco.core.initial_state_handler import InitialStateHandler

class RandomHeightInitialStateHandler(InitialStateHandler):

    def reset(self, env, model, data, carry, backend, traj=None):
        if backend == np:
            data.qpos[2] = np.random.uniform(1.0, 1.2)
        else:
            key, subkey = jax.random.split(carry.key)
            z = jax.random.uniform(subkey, minval=1.0, maxval=1.2)
            data = data.replace(qpos=data.qpos.at[2].set(z))
            carry = carry.replace(key=key)
        return data, carry
```

A terminal state handler derives from
{class}`~loco_mujoco.core.terminal_state_handler.base.TerminalStateHandler`.

| Method | Required | Purpose |
|---|---|---|
| `reset(env, model, data, carry, backend, traj=None)` | yes | Called at every reset. Returns the data and the carry. |
| `is_absorbing(env, obs, info, data, carry, traj=None)` | yes | CPU version. Returns the flag and the carry. |
| `mjx_is_absorbing(env, obs, info, data, carry, traj=None)` | yes | MJX version. Returns the flag and the carry. |
| `mjx_absorbing_prob(env, obs, info, data, carry, traj=None)` | no | Smooth termination probability for algorithms that differentiate through the termination |

The usual pattern is one shared implementation with a `backend` argument:

```python
import jax.numpy as jnp
from loco_mujoco.core.terminal_state_handler import TerminalStateHandler

class TiltTerminalStateHandler(TerminalStateHandler):

    def __init__(self, env, min_up_z=0.5, **kwargs):
        super().__init__(env, **kwargs)
        self._min_up_z = min_up_z

    def reset(self, env, model, data, carry, backend, traj=None):
        return data, carry

    def is_absorbing(self, env, obs, info, data, carry, traj=None):
        return self._is_absorbing_compat(env, obs, info, data, carry, backend=np, traj=traj)

    def mjx_is_absorbing(self, env, obs, info, data, carry, traj=None):
        return self._is_absorbing_compat(env, obs, info, data, carry, backend=jnp, traj=traj)

    def _is_absorbing_compat(self, env, obs, info, data, carry, backend, traj=None):
        w, x, y, z = data.qpos[3:7]
        up_z = 1.0 - 2.0 * (x ** 2 + y ** 2)     # vertical component of the root's up axis
        return up_z < self._min_up_z, carry
```

## Domain randomizers

Derive from {class}`~loco_mujoco.core.domain_randomizer.base.DomainRandomizer` and implement all four hooks.
A hook that you do not need returns its inputs.

| Method | Returns |
|---|---|
| `reset(env, model, data, carry, backend, traj=None)` | `data, carry`. Sample the parameters of the episode here. |
| `update(env, model, data, carry, backend, traj=None)` | `model, data, carry`. Apply the parameters to the model before every step. |
| `update_observation(env, obs, model, data, carry, backend, traj=None)` | `obs, carry` |
| `update_action(env, action, model, data, carry, backend, traj=None)` | `action, carry` |

The model is not stored per environment. On MJX `update` receives the shared base model at every step and
returns a modified copy, so the parameters have to be sampled in `reset`, kept in the carry and applied in
`update`. On the CPU the `MjModel` is changed in place and keeps the change.

```python
from loco_mujoco.core.domain_randomizer import DomainRandomizer

@struct.dataclass
class GravityRandomizerState:
    gravity_z: float

class GravityRandomizer(DomainRandomizer):

    def __init__(self, env, low=-10.1, high=-9.5, **kwargs):
        super().__init__(env, **kwargs)
        self._low, self._high = low, high

    def init_state(self, env, key, model, data, backend, traj=None):
        return GravityRandomizerState(gravity_z=-9.81)

    def reset(self, env, model, data, carry, backend, traj=None):
        if backend == np:
            gravity_z = np.random.uniform(self._low, self._high)
        else:
            key, subkey = jax.random.split(carry.key)
            gravity_z = jax.random.uniform(subkey, minval=self._low, maxval=self._high)
            carry = carry.replace(key=key)
        state = carry.domain_randomizer_state.replace(gravity_z=gravity_z)
        return data, carry.replace(domain_randomizer_state=state)

    def update(self, env, model, data, carry, backend, traj=None):
        gravity = backend.array([0.0, 0.0, carry.domain_randomizer_state.gravity_z])
        if backend == np:
            model.opt.gravity[:] = gravity                           # MjModel is changed in place
        else:
            model = model.tree_replace({"opt.gravity": gravity})     # an MJX model is immutable
        return model, data, carry

    def update_observation(self, env, obs, model, data, carry, backend, traj=None):
        return obs, carry

    def update_action(self, env, action, model, data, carry, backend, traj=None):
        return action, carry
```

## Terrains

Derive from {class}`~loco_mujoco.core.terrain.static.StaticTerrain` for a terrain that does not change after
construction, which only needs the first two methods of the table, and from
{class}`~loco_mujoco.core.terrain.dynamic.DynamicTerrain` or
{class}`~loco_mujoco.core.terrain.base.Terrain` otherwise.

| Method | Called | Purpose |
|---|---|---|
| `modify_spec(spec)` | Once, while the model is built | Adds the geometry of the terrain to the `MjSpec` and returns it |
| `get_height_at(xy, backend, env=None, carry=None)` | By observations and reset logic | Height of the surface at points of shape (..., 2). Needed for the `HeightScan` observation. |
| `init_state(...)` | When the carry is created | State per environment (`carry.terrain_state`) |
| `reset(env, model, data, carry, backend, traj=None)` | At every reset, before the initial state is set | Returns `data, carry`, for example with a newly generated surface in the state |
| `update(env, model, data, carry, backend, traj=None)` | Before every step | Returns `model, data, carry`. A terrain that differs per environment writes its state into the model here. |
| `is_dynamic` | | Property. `True` if the terrain writes heightfield data at run time. The viewer then uploads the heightfield with the id `hfield_id` again for every frame, from the field `height_field_raw` of the terrain state. |

```python
import mujoco
from loco_mujoco.core.terrain import StaticTerrain

class BoxTerrain(StaticTerrain):

    def __init__(self, env, height=0.1, **kwargs):
        super().__init__(env, **kwargs)
        self._height = height

    def modify_spec(self, spec):
        spec.worldbody.add_geom(name="step", type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[0.5, 1.0, self._height / 2], pos=[2.0, 0.0, self._height / 2])
        return spec

    def get_height_at(self, xy, backend, env=None, carry=None):
        on_box = (backend.abs(xy[..., 0] - 2.0) <= 0.5) & (backend.abs(xy[..., 1]) <= 1.0)
        return backend.where(on_box, self._height, 0.0)
```

Further properties control where the robot is placed (`has_env_origins`, `trajectories_in_world_frame`,
`grounds_robot`). They are described in the API of {class}`~loco_mujoco.core.terrain.base.Terrain`.

## Environments

An environment class describes a robot. A variant of an existing robot is a subclass that overrides what
differs, typically the default observations, the default actuators, or an info property that modules read.

```python
from loco_mujoco.core import ObservationType
from loco_mujoco.core.utils import info_property
from loco_mujoco.environments import UnitreeG1

class MyUnitreeG1(UnitreeG1):

    @info_property
    def root_height_healthy_range(self):
        return (0.5, 1.0)

    @staticmethod
    def _get_observation_specification(spec):
        observation_spec = UnitreeG1._get_observation_specification(spec)
        observation_spec.append(ObservationType.ProjectedGravityVector("proj_gravity", xml_name="root"))
        observation_spec.append(ObservationType.LastAction("last_action"))
        return observation_spec

MyUnitreeG1.register()
env, _ = RLFactory.make("MyUnitreeG1")
```

| Member | Purpose |
|---|---|
| `_get_observation_specification(spec)` | Default `observation_spec` of the robot |
| `_get_action_specification(spec)` | Default list of actuator names |
| Info properties (`@info_property`) | Static facts that modules read, for example `root_free_joint_xml_name`, `root_body_name`, `upper_body_xml_name`, `root_height_healthy_range`, `sites_for_mimic` and `foot_geom_names` |
| `mjx_enabled` | Class attribute. `True` makes the class an MJX environment. Derive from the MJX class of the robot, for example `MjxUnitreeG1`, to get an MJX variant. |

{class}`~loco_mujoco.task_factories.imitation_factory.ImitationFactory` looks datasets up by the class name of
the environment. For a subclass with a new name, load the trajectory with the original robot and pass it on:

```python
from loco_mujoco import ImitationFactory
from loco_mujoco.task_factories import CustomDatasetConf

_, traj = ImitationFactory.make("UnitreeG1", default_dataset_conf=dict(task="walk"))
env, traj = ImitationFactory.make("MyUnitreeG1", custom_dataset_conf=CustomDatasetConf(traj))
```

Adding a robot with a new model is described in [Adding a new robot](new_robot.md).

## Checking a module

Run a new module on both backends before training with it. A few CPU steps catch errors in shapes and
names quickly. One jitted and vmapped MJX step catches code that is not compatible with JAX:

```python
import jax
import jax.numpy as jnp

env, _ = RLFactory.make("MjxUnitreeG1", reward_type="NominalPoseReward")
reset, step = jax.jit(jax.vmap(env.mjx_reset)), jax.jit(jax.vmap(env.mjx_step))

state = reset(jax.random.split(jax.random.key(0), 4))
state = step(state, jnp.zeros((4, env.info.action_space.shape[0])))
print(state.reward)
```

The script `examples/tutorials/11_creating_custom_modules.py` combines a custom environment with custom
modules of several kinds in one file.

## See also

- [Modular architecture](modular_architecture.md) for registration, the carry and the two backends.
- The pages of the built-in modules: [observations](observations.md), [rewards](rewards.md),
  [control functions](control_functions.md), [domain randomization](domain_randomization.md),
  [terrain](terrain.md) and [initial and terminal states](initial_terminal_states.md).
- [Adding a new robot](new_robot.md) and [creating custom trajectories](custom_trajectories.md).
- API: {mod}`loco_mujoco.core`, {class}`~loco_mujoco.core.stateful_object.StatefulObject`.
