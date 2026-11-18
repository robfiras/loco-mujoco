# Task factories

A task factory creates an environment that is ready for a particular kind of task. It picks the environment
class by name, sets the components that the task needs, and, for imitation learning, downloads the motion data
and loads it into the environment. There are two factories:

- {class}`~loco_mujoco.task_factories.imitation_factory.ImitationFactory` for tasks that follow reference
  motions,
- {class}`~loco_mujoco.task_factories.rl_factory.RLFactory` for tasks without reference motions.

Both are called the same way and both return a pair `(env, traj)`:

```python
from loco_mujoco import ImitationFactory, RLFactory

# an imitation task: the Unitree H1 with the walking motion of the default dataset
env, traj = ImitationFactory.make("UnitreeH1", default_dataset_conf=dict(task="walk"))

# a plain reinforcement learning task: the same robot tracking a commanded velocity
env, _ = RLFactory.make("UnitreeH1")
```

`env` is a {class}`~loco_mujoco.environments.base.LocoEnv`. `traj` is the
{class}`~loco_mujoco.core.trajectory.dataclasses.Trajectory` that was loaded into the environment, or `None`
for the `RLFactory`.

## Environment names

The first argument is the name of a registered environment. Every embodiment is registered twice: under its
plain name for the CPU backend and with the prefix `Mjx` for the GPU backends, for example `UnitreeG1` and
`MjxUnitreeG1`. The [backends page](backends.md) explains the difference.

List all names with:

```python
import loco_mujoco

print(list(loco_mujoco.get_registered_envs()))
# ['Atlas', 'MjxAtlas', 'Talos', 'MjxTalos', 'UnitreeH1', 'MjxUnitreeH1', ...]
```

{func}`loco_mujoco.get_registered_envs` returns a dictionary from name to environment class. The
[environment gallery](../environments/index.md) describes each embodiment. Environments defined in another
package, such as [LocoMuJoCo Real](../sim2real/index.md), appear in the registry once that package has been
imported.

## ImitationFactory

```python
ImitationFactory.make(env_name,
                      default_dataset_conf=None, amass_dataset_conf=None,
                      lafan1_dataset_conf=None, custom_dataset_conf=None,
                      terminal_state_type="RootPoseTrajTerminalStateHandler",
                      init_state_type="TrajInitialStateHandler",
                      **kwargs)
```

The factory creates the environment, loads the trajectories of every dataset configuration that is given,
concatenates them into one {class}`~loco_mujoco.core.trajectory.dataclasses.Trajectory` and hands it to the
environment with {meth}`~loco_mujoco.environments.base.LocoEnv.process_trajectory`. At least one dataset
configuration is required.

### Dataset configurations

Each motion source has its own configuration class in {mod}`loco_mujoco.task_factories.dataset_confs`.
Several can be passed in the same call; the clips of all sources end up in one trajectory.

| Argument | Class | Fields | Source |
|---|---|---|---|
| `default_dataset_conf` | {class}`~loco_mujoco.task_factories.dataset_confs.DefaultDatasetConf` | `task`: one name or a list, default `"walk"`. `dataset_type`: only `"mocap"`. `debug`: currently unused. | The native LocoMuJoCo dataset, downloaded from Hugging Face |
| `lafan1_dataset_conf` | {class}`~loco_mujoco.task_factories.dataset_confs.LAFAN1DatasetConf` | `dataset_name`: one name or a list. `dataset_group`: a predefined group. One of the two is required. | LAFAN1, downloaded from Hugging Face |
| `amass_dataset_conf` | {class}`~loco_mujoco.task_factories.dataset_confs.AMASSDatasetConf` | `rel_dataset_path`: one path or a list. `dataset_group`: a predefined group. One of the two is required. | AMASS, retargeted locally; needs the [AMASS setup](../getting_started/amass.md) |
| `custom_dataset_conf` | {class}`~loco_mujoco.task_factories.dataset_confs.CustomDatasetConf` | `traj`: a `Trajectory` object. | [Your own trajectory](custom_trajectories.md) |

A configuration can be given as an instance of its class or as a dictionary with the same fields. The
dictionary form is convenient in YAML configuration files:

```python
from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf, LAFAN1DatasetConf

# these two calls are equivalent
env, traj = ImitationFactory.make("UnitreeH1",
                                  default_dataset_conf=DefaultDatasetConf(["squat"]),
                                  lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4", "walk1_subject1"]))

env, traj = ImitationFactory.make("UnitreeH1",
                                  default_dataset_conf=dict(task=["squat"]),
                                  lafan1_dataset_conf=dict(dataset_name=["dance2_subject4", "walk1_subject1"]))
```

The available motion names, the predefined groups and the caching of downloaded data are described on the
[datasets page](datasets.md).

### Defaults

The `ImitationFactory` changes two components from the defaults of the environment, so that an episode starts
on the reference motion and ends when the robot leaves it:

| Component | Set by `ImitationFactory` | Behavior |
|---|---|---|
| Initial state | {class}`~loco_mujoco.core.initial_state_handler.traj_init_state.TrajInitialStateHandler` | An episode starts from a state of the trajectory. |
| Terminal state | {class}`~loco_mujoco.core.terminal_state_handler.traj.RootPoseTrajTerminalStateHandler` | An episode ends when the height or the orientation of the root leaves the range covered by the trajectory, plus a margin. |

Everything else keeps the default of the environment. In particular, the reward is `NoReward` and the goal is
`NoGoal`, so the factory alone gives you an environment for replaying motions and for imitation learning from
observations (for example [AMP or GAIL](../training/amp_gail.md)). For motion tracking, select a goal and a
reward:

```python
env, traj = ImitationFactory.make("UnitreeH1",
                                  default_dataset_conf=dict(task="walk"),
                                  goal_type="GoalTrajMimic",
                                  reward_type="MimicReward")
```

## RLFactory

```python
RLFactory.make(env_name,
               init_state_type="DefaultInitialStateHandler",
               terminal_state_type="HeightBasedTerminalStateHandler",
               goal_type="GoalRandomRootVelocity",
               reward_type="TargetVelocityGoalReward",
               **kwargs)
```

The `RLFactory` loads no trajectory and sets four components that together define a velocity-tracking
locomotion task:

| Component | Set by `RLFactory` | Behavior |
|---|---|---|
| Initial state | {class}`~loco_mujoco.core.initial_state_handler.default.DefaultInitialStateHandler` | An episode starts from the default pose of the model. |
| Terminal state | {class}`~loco_mujoco.core.terminal_state_handler.height.HeightBasedTerminalStateHandler` | An episode ends when the root height leaves the healthy range of the robot. |
| Goal | {class}`~loco_mujoco.core.observations.goals.GoalRandomRootVelocity` | A random target velocity of the root is drawn and appended to the observation. |
| Reward | {class}`~loco_mujoco.core.reward.default.TargetVelocityGoalReward` | The reward is high when the root velocity matches the target. |

## Overriding defaults and passing environment arguments

All keyword arguments that the factory does not consume are passed to the constructor of the environment.
This is how every component is selected and configured: each has a `..._type` argument with the registered
name of the class and a `..._params` argument with a dictionary of its parameters.

```python
env, traj = ImitationFactory.make(
    "MjxUnitreeG1",
    lafan1_dataset_conf=dict(dataset_name=["walk1_subject5"]),
    # override a default of the factory
    terminal_state_type="RootPoseTrajTerminalStateHandler",
    terminal_state_params=dict(root_height_margin=0.2),
    # select other components
    goal_type="GoalTrajMimic",
    reward_type="MimicReward",
    terrain_type="RoughTerrain",
    # general environment arguments
    horizon=1000,
    headless=True,
)
```

The most common arguments are listed below. The pages of the individual components list their options.

| Argument | Default | Meaning |
|---|---|---|
| `observation_spec` | The default observation of the embodiment | [Observations](observations.md) the policy receives |
| `goal_type`, `goal_params` | `"NoGoal"` | Goal appended to the [observation](observations.md) |
| `reward_type`, `reward_params` | `"NoReward"` | [Reward function](rewards.md) |
| `control_type`, `control_params` | `"DefaultControl"` | [Control function](control_functions.md) that maps actions to actuator commands |
| `init_state_type`, `init_state_params` | `"DefaultInitialStateHandler"` | [Initial state](initial_terminal_states.md) |
| `terminal_state_type`, `terminal_state_params` | `"NoTerminalStateHandler"` | [Terminal state](initial_terminal_states.md) |
| `domain_randomization_type`, `domain_randomization_params` | `"NoDomainRandomization"` | [Domain randomization](domain_randomization.md) |
| `terrain_type`, `terrain_params` | `"FlatTerrain"` | [Terrain](terrain.md) |
| `th_params` | `None` | Parameters of the [trajectory handler](trajectories.md), for example how episode starts are sampled |
| `horizon` | `1000` | Maximum number of steps of an episode |
| `gamma` | `0.99` | Discount factor stored in the environment information |
| `timestep`, `n_substeps` | Depends on the environment | Simulation step and number of simulation steps per environment step |
| `headless` | `False` | Render without opening a window |
| `use_mjwarp`, `naconmax`, `njmax` | `False`, `None`, `None` | Select [MuJoCo Warp](backends.md) for an `Mjx` environment |

Embodiments add their own arguments. The Unitree H1, for example, accepts `disable_arms` and
`disable_back_joint`. They are documented with each environment class, for example
{class}`~loco_mujoco.environments.humanoids.unitreeH1.UnitreeH1`, and in the
[environment gallery](../environments/index.md).

```{note}
The `Default` column gives the defaults of the environment. The factories replace some of them, as shown in
the two tables above.
```

## Selecting a factory by name

Factories are registered by class name, so a configuration file can name the factory together with its
arguments. The training examples use this to switch between imitation and plain reinforcement learning without
changing code:

```python
from loco_mujoco import TaskFactory

factory = TaskFactory.get_factory_cls("ImitationFactory")
env, traj = factory.make("MjxUnitreeH1", default_dataset_conf=dict(task="walk"))
```

See [configuration](../training/configuration.md) for how the training examples lay this out in YAML.

## Creating an environment without a factory

A factory is a convenience, not a requirement. Environment classes can be instantiated directly, and a
trajectory can be attached afterwards:

```python
from loco_mujoco.environments import UnitreeH1

env = UnitreeH1(reward_type="TargetVelocityGoalReward", goal_type="GoalRandomRootVelocity")
```

`LocoEnv.make` and `LocoEnv.generate` from earlier versions still work but are deprecated in favor of the
factories.

## See also

- [Simulation backends](backends.md) for the difference between `UnitreeH1` and `MjxUnitreeH1`.
- [Datasets](datasets.md) for the motions each dataset configuration can load.
- [Modular architecture](modular_architecture.md) for how the `..._type` arguments map to registered classes.
- [Gymnasium interface](gymnasium.md), which uses the factories internally.
- API reference: {mod}`loco_mujoco.task_factories.imitation_factory`,
  {mod}`loco_mujoco.task_factories.rl_factory`, {mod}`loco_mujoco.task_factories.dataset_confs`.
