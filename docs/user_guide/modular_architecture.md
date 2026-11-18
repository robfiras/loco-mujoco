# Modular architecture

A LocoMuJoCo environment is not one monolithic class per robot and task. The robot class only knows its
MuJoCo model, its default observations and its actuators. Everything else that defines the task (what the
agent observes, what it is rewarded for, how its actions are turned into motor commands, where an episode
starts and when it ends, what is randomized, and what the ground looks like) is a **module**: a small class
that is chosen by name when the environment is created.

This is what lets one training script run on every embodiment, and what makes the step from velocity
tracking to motion imitation a change of two arguments instead of a new environment class.

```python
from loco_mujoco import ImitationFactory

env, traj = ImitationFactory.make(
    "UnitreeH1",
    default_dataset_conf=dict(task="walk"),
    goal_type="GoalTrajMimic",                       # what the policy is told to do
    reward_type="MimicReward",                       # what it is rewarded for
    control_type="PDControl",                        # how actions become motor commands
    control_params=dict(p_gain=100.0, d_gain=1.0),
    terrain_type="RoughTerrain",                     # what it walks on
)
```

Every `*_type` argument is the name of a registered class, and the matching `*_params` argument is a
dictionary of keyword arguments for the constructor of that class. Arguments that are left out fall back to
the defaults of the [task factory](task_factories.md) and of the environment.

## The module kinds

| Module | Decides | Base class | Constructor arguments | Built-in options | Guide |
|---|---|---|---|---|---|
| Observations | What the agent sees, in which order, and which entries belong to which group | {class}`~loco_mujoco.core.observations.base.Observation` | `observation_spec` | The types in {class}`~loco_mujoco.core.observations.base.ObservationType`, for example `JointPos`, `FreeJointVel`, `BodyPos`, `ProjectedGravityVector`, `LastAction`, `HeightScan` | [Observations](observations.md) |
| Goal | The task specification that is appended to the observation, for example a target velocity or the reference motion | {class}`~loco_mujoco.core.observations.goals.Goal` | `goal_type`, `goal_params` | `NoGoal`, `GoalRandomRootVelocity`, `GoalTrajRootVelocity`, `GoalTrajMimic`, `GoalTrajMimicv2` | [Observations](observations.md#goals) |
| Reward | The scalar reward of a transition | {class}`~loco_mujoco.core.reward.base.Reward` | `reward_type`, `reward_params` | `NoReward`, `TargetXVelocityReward`, `TargetVelocityGoalReward`, `LocomotionReward`, `TargetVelocityTrajReward`, `MimicReward`, `SiteTrackingReward` | [Rewards](rewards.md) |
| Control function | How an action is mapped to actuator controls, and the bounds of the action space | {class}`~loco_mujoco.core.control_functions.base.ControlFunction` | `control_type`, `control_params` | `DefaultControl`, `PDControl` | [Control functions](control_functions.md) |
| Domain randomizer | Which model parameters change between episodes and which noise is added to observations and actions | {class}`~loco_mujoco.core.domain_randomizer.base.DomainRandomizer` | `domain_randomization_type`, `domain_randomization_params` | `NoDomainRandomization`, `DefaultRandomizer` | [Domain randomization](domain_randomization.md) |
| Terrain | The ground geometry and where on it an episode starts | {class}`~loco_mujoco.core.terrain.base.Terrain` | `terrain_type`, `terrain_params` | `FlatTerrain` (also selectable as `StaticTerrain`), `RoughTerrain`, `HeightmapTerrain` | [Terrain](terrain.md) |
| Initial state handler | The simulation state at the start of an episode | {class}`~loco_mujoco.core.initial_state_handler.base.InitialStateHandler` | `init_state_type`, `init_state_params` | `DefaultInitialStateHandler`, `TrajInitialStateHandler` | [Initial and terminal states](initial_terminal_states.md) |
| Terminal state handler | Which states are absorbing, that is, end the episode as a failure | {class}`~loco_mujoco.core.terminal_state_handler.base.TerminalStateHandler` | `terminal_state_type`, `terminal_state_params` | `NoTerminalStateHandler`, `HeightBasedTerminalStateHandler`, `RootPoseTrajTerminalStateHandler`, `RootTrackingTerminalStateHandler` | [Initial and terminal states](initial_terminal_states.md) |
| Trajectory handler | Which reference clip and which frame of it an episode starts from | {class}`~loco_mujoco.core.trajectory.handler.TrajectoryHandler` | `th_params` (with the class name under the key `name`) | `RandomStartTrajectoryHandler`, `RandomTrajFixedStepTrajectoryHandler`, `FixedStartTrajectoryHandler` | [Trajectories](trajectories.md) |

The arguments are defined by the constructor of {class}`~loco_mujoco.core.mujoco_base.Mujoco`, the base class
of all environments, except `observation_spec`, which is handled by the robot classes, and `th_params`,
which belongs to {class}`~loco_mujoco.environments.base.LocoEnv`. The factories pass all keyword arguments
through, so the same names work with {class}`~loco_mujoco.task_factories.imitation_factory.ImitationFactory`,
{class}`~loco_mujoco.task_factories.rl_factory.RLFactory`, `gymnasium.make` and in the Hydra configurations of
the [training examples](../training/configuration.md).

The defaults of the base class are the neutral choice for every slot: `NoReward`, `NoGoal`,
`NoTerminalStateHandler`, `NoDomainRandomization`, `FlatTerrain`, `DefaultInitialStateHandler` and
`DefaultControl`. The task factories replace some of them, see
[Initial and terminal states](initial_terminal_states.md#defaults-of-the-task-factories) and
[Task factories](task_factories.md).

## How a module is selected

Each base class keeps a registry, a dictionary from class name to class. The environment looks the name up
and constructs the class with itself as the first argument:

```python
reward_cls = Reward.registered[reward_type]
self._reward_function = reward_cls(self, **reward_params)
```

A class enters the registry by calling `register()` on it. The built-in modules are registered when their
package is imported, and your own modules are registered in the same way:

```python
from loco_mujoco.core.reward import Reward

class MyReward(Reward):
    ...

MyReward.register()

env, _ = RLFactory.make("UnitreeG1", reward_type="MyReward", reward_params=dict(scale=0.5))
```

Most registries can be inspected with `list_registered()`:

```python
>>> from loco_mujoco.core.reward import Reward
>>> Reward.list_registered()
['NoReward', 'TargetVelocityGoalReward', 'TargetXVelocityReward', 'TargetVelocityTrajReward',
 'MimicReward', 'SiteTrackingReward', 'LocomotionReward']
```

Because modules are addressed by a string and configured by a plain dictionary, a complete environment can
be described in a YAML file. This is how the training examples are configured. Environments themselves are
registered in the same way, see [Writing custom modules](custom_modules.md).

```{note}
Observations are the exception to the `*_type` and `*_params` pattern, because an environment has many of
them. They are passed as a list, `observation_spec`, of observation objects or of dictionaries with the key
`type`. Goals are observations as well: they share the registry of the observations, and the goal is always
the last entry of the observation vector.
```

## When each module runs

The environment calls its modules at fixed points of `reset` and `step`. The order is the same on both
backends.

On **reset**, the terminal state handler, the terrain and the initial state handler are reset in this order,
then the terrain may move the robot to its place on the map, and then the domain randomizer, the reward and
the control function are reset. The stateful observations, including the goal, are reset last, before the
first observation is created.

On **step**:

1. The domain randomizer may change the action (`update_action`).
2. The terrain and the domain randomizer may change the model and the data (`update`). This is where a
   sampled friction coefficient or a per-environment heightfield is written into the model.
3. The control function turns the action into actuator controls (`generate_action`), and the simulation is
   advanced by `n_substeps` physics steps. A control function that runs at the simulation frequency, such as
   `PDControl`, is called before every physics step instead of once per environment step.
4. The observation is created from the new simulation state, and the domain randomizer may add noise to it
   (`update_observation`).
5. The terminal state handler decides whether the state is absorbing.
6. The reward is computed.
7. The episode is done if the state is absorbing, the horizon is reached, the reference trajectory has
   ended (for goals and rewards that need one), or the goal says so.

## Stateless modules and the carry

A module object is created once and shared by all parallel environments. On MJX the step function is
compiled with `jax.jit` and mapped over thousands of environments with `jax.vmap`, so a module must not keep
anything in its attributes that changes from step to step or differs between environments. A module is a
set of pure functions plus constants that are computed at construction time, such as joint indices or
weights.

Whatever does change is kept in the **carry**, a [Flax](https://flax.readthedocs.io) dataclass
({class}`~loco_mujoco.core.mujoco_base.AdditionalCarry`) that travels with the simulation state. On MJX it is
the field `additional_carry` of {class}`~loco_mujoco.core.mujoco_mjx.MjxState`. On the CPU the environment
stores it between calls. Besides bookkeeping such as the random key (`key`), the step counter
(`cur_step_in_episode`), the previous action (`last_action`) and, in a
{class}`~loco_mujoco.environments.base.LocoEnv`, the position in the reference trajectory (`traj_state`), it
has one field per module:

| Field of the carry | Owner |
|---|---|
| `observation_states` | One entry per stateful observation and for the goal, accessed by the name of the observation |
| `reward_state` | Reward |
| `control_func_state` | Control function |
| `domain_randomizer_state` | Domain randomizer |
| `terrain_state` | Terrain |
| `init_state_handler_state` | Initial state handler |
| `terminal_state_handler_state` | Terminal state handler |
| `user_scene` | Geometries that modules draw in the viewer, such as goal arrows |

All of these module classes derive from {class}`~loco_mujoco.core.stateful_object.StatefulObject`. Its method
`init_state` returns the initial state of the module, and the default is an empty state
({class}`~loco_mujoco.core.stateful_object.EmptyState`). A module that needs state defines a dataclass for it,
returns it from `init_state`, and from then on reads it from the carry and returns an updated carry:

```python
import jax
from flax import struct
from loco_mujoco.core.reward import Reward

@struct.dataclass
class MyRewardState:
    last_action: jax.Array

class MyReward(Reward):

    def init_state(self, env, key, model, data, backend, traj=None):
        return MyRewardState(last_action=backend.zeros(env.info.action_space.shape[0]))

    def __call__(self, state, action, next_state, absorbing, info, env, model, data, carry, backend, traj=None):
        penalty = backend.sum(backend.square(action - carry.reward_state.last_action))
        carry = carry.replace(reward_state=carry.reward_state.replace(last_action=action))
        return -penalty, carry
```

The carry is immutable. `replace` returns a new object, and the module hands it back to the environment
together with its result. This is why almost every module method returns a tuple that ends with `carry`.

`init_state` runs when the carry is created, which happens in `reset` and `mjx_reset`. On MJX an environment
that finishes an episode is reset inside `mjx_step` and keeps its carry. A module whose state has to start
fresh in every episode therefore also assigns it in its `reset` method, as `MimicReward` does.

## One module, two backends

The same module object serves the CPU environment and the MJX environment. Every module method receives the
arguments it needs to stay agnostic of the backend:

| Argument | MuJoCo (CPU) | MJX and MuJoCo Warp |
|---|---|---|
| `backend` | `numpy` | `jax.numpy` |
| `model` | `mujoco.MjModel` | `mujoco.mjx.Model` |
| `data` | `mujoco.MjData` | `mujoco.mjx.Data` |
| `carry` | Flax dataclass with NumPy arrays | Flax dataclass with JAX arrays |

Code that only does arithmetic is written once against `backend`, for example
`backend.exp(-backend.square(x))`. Code that cannot be shared is put behind a check of the backend. The usual
cases are in-place writes, random numbers and control flow:

```python
if backend == np:
    data.qpos[2] = np.random.uniform(0.9, 1.1)                  # in place, global NumPy random state
else:
    key, subkey = jax.random.split(carry.key)                    # explicit random key from the carry
    data = data.replace(qpos=data.qpos.at[2].set(jax.random.uniform(subkey, minval=0.9, maxval=1.1)))
    carry = carry.replace(key=key)
```

Two interfaces have separate methods for the two backends instead of a `backend` argument: a terminal state
handler implements `is_absorbing` and `mjx_is_absorbing`, and a goal implements `is_done` and `mjx_is_done`.
The built-in ones forward both to one shared implementation.

A module can also read static properties of the robot from the environment, such as the name of the root
joint or the list of mimic sites. The environment classes expose them as *info properties* (properties
decorated with `info_property`), and `env._get_all_info_properties()` returns them as a dictionary. This is
how `MimicReward` finds the sites to track without a robot-specific configuration.

## See also

- [Task factories](task_factories.md) for how the defaults of the modules are chosen.
- [Simulation backends](backends.md) for the two step APIs that drive the modules.
- [Writing custom modules](custom_modules.md) for the methods each base class expects.
- [Architecture for contributors](../development/architecture.md).
- API: {mod}`loco_mujoco.core`, {class}`~loco_mujoco.core.mujoco_base.Mujoco`,
  {class}`~loco_mujoco.core.stateful_object.StatefulObject`.
