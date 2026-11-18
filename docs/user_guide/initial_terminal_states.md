# Initial and terminal states

Two modules frame every episode. The **initial state handler** sets the simulation state at reset, and the
**terminal state handler** decides at every step whether the current state ends the episode as a failure.
Both have a large effect on learning: starting from states along a reference motion and ending episodes
early when the robot falls are two of the main ingredients of motion imitation.

```python
from loco_mujoco import ImitationFactory

env, traj = ImitationFactory.make(
    "UnitreeH1",
    default_dataset_conf=dict(task="walk"),
    init_state_type="TrajInitialStateHandler",                    # start on a frame of the reference
    terminal_state_type="RootPoseTrajTerminalStateHandler",       # end when the root leaves the dataset's range
    terminal_state_params=dict(root_height_margin=0.2, root_rot_margin_degrees=20.0),
)
```

## Initial state handlers

The initial state handler is selected with `init_state_type` and configured with `init_state_params`. It is
called at every reset and writes the joint positions and velocities of the first state.

| `init_state_type` | Initial state |
|---|---|
| `DefaultInitialStateHandler` | The default pose of the model with zero velocities, or a fixed state that you provide |
| `TrajInitialStateHandler` | A frame of the reference trajectory, with optional noise |

### DefaultInitialStateHandler

{class}`~loco_mujoco.core.initial_state_handler.default.DefaultInitialStateHandler` leaves the state as it
is after MuJoCo's reset, which is the pose `qpos0` of the model at rest. Two optional parameters replace it
by a fixed state:

| Parameter | Default | Meaning |
|---|---|---|
| `qpos_init` | `None` | Full vector of generalized positions, of the size of `qpos` of the model |
| `qvel_init` | `None` | Full vector of generalized velocities, of the size of `qvel` of the model |

Every episode then starts from exactly the same state.

### TrajInitialStateHandler

{class}`~loco_mujoco.core.initial_state_handler.traj_init_state.TrajInitialStateHandler` copies positions and
velocities from a frame of the reference trajectory, so the robot starts in motion and on the reference.
This is the reference state initialization of DeepMimic. It needs an environment with a trajectory.

Which frame is used is not decided by this handler but by the trajectory handler of the environment, which
picks a clip and a frame at every reset (see [Trajectories](trajectories.md)):

| `th_params` | Start of an episode |
|---|---|
| `dict(name="RandomStartTrajectoryHandler")` (default) | A random frame of a random clip |
| `dict(name="RandomTrajFixedStepTrajectoryHandler", ...)` | A fixed frame of a random clip |
| `dict(name="FixedStartTrajectoryHandler", start_conf=(clip, frame))` | Always the same frame of the same clip, useful for evaluation |

The handler can perturb the frame with uniform noise, so that the policy also sees states slightly off the
reference. Each parameter is the half-width of the interval the noise is drawn from.

| Parameter | Default | Noise on |
|---|---|---|
| `root_pos_noise` | `(0, 0, 0)` | Root position per axis, in meters |
| `root_rot_noise` | `(0, 0, 0)` | Root orientation, as a rotation vector in radians |
| `root_lin_vel_noise` | `(0, 0, 0)` | Root linear velocity per axis, in m/s |
| `root_ang_vel_noise` | `(0, 0, 0)` | Root angular velocity per axis, in rad/s |
| `joint_pos_noise` | `0.0` | Positions of all hinge and slide joints, in radians. The result is clipped to the joint range. |

```python
env, traj = ImitationFactory.make(
    "MjxUnitreeH1", default_dataset_conf=dict(task="walk"),
    init_state_params=dict(root_pos_noise=(0.02, 0.02, 0.0), joint_pos_noise=0.05),
)
```

```{note}
The reset noise is implemented for the MJX step functions only. Calling the CPU `reset` of an environment
with non-zero noise raises an assertion error.
```

## Terminal state handlers

The terminal state handler is selected with `terminal_state_type` and configured with
`terminal_state_params`. At every step it returns the `absorbing` flag of the transition.

| `terminal_state_type` | A state is absorbing when | Parameters (defaults) |
|---|---|---|
| `NoTerminalStateHandler` | Never | none |
| `HeightBasedTerminalStateHandler` | The height of the root is outside the healthy range of the robot | none |
| `RootPoseTrajTerminalStateHandler` | The root leaves the range of heights and orientations that occurs in the dataset, or strays too far from the reference horizontally | `root_height_margin` (0.3), `root_rot_margin_degrees` (30.0), `max_root_pos_deviation` (1e6) |
| `RootTrackingTerminalStateHandler` | The height or the tilt of the root deviates too much from the current reference frame | `max_height_error` (0.25), `max_tilt_error` (0.8) |

### HeightBasedTerminalStateHandler

{class}`~loco_mujoco.core.terminal_state_handler.height.HeightBasedTerminalStateHandler` is the usual fall
detection of locomotion tasks without a reference. The range is a property of the robot, its info property
`root_height_healthy_range`. To change it, override the property in a subclass of the environment, see
[Writing custom modules](custom_modules.md#environments).

### RootPoseTrajTerminalStateHandler

{class}`~loco_mujoco.core.terminal_state_handler.traj.RootPoseTrajTerminalStateHandler` derives its limits
from the loaded dataset, so it needs no tuning per robot or motion. A state is absorbing if one of three
conditions holds:

1. The height of the root is below the lowest root height in the dataset minus `root_height_margin`, or
   above the highest one plus the margin (in meters).
2. The orientation of the root is further from the mean root orientation of the dataset than any frame of
   the dataset, plus `root_rot_margin_degrees`.
3. The horizontal distance between the root and the root of the current reference frame exceeds
   `max_root_pos_deviation` (in meters). The default of $10^6$ turns this condition off.

For a dataset of upright walking this terminates the episode when the robot falls or leans far over. It
becomes loose when the dataset itself covers a wide range of poses: with a clip of lying down and getting up
in the dataset, lying on the floor is within the range of the dataset and is not a failure.

### RootTrackingTerminalStateHandler

{class}`~loco_mujoco.core.terminal_state_handler.traj.RootTrackingTerminalStateHandler` compares the root to
the *current* frame of the reference instead of to the range of the whole dataset, so it stays meaningful
for datasets with motions on the ground. A state is absorbing if

- the root height differs from that of the reference frame by more than `max_height_error` (in meters), or
- the tilt differs by more than `max_tilt_error`. The tilt is measured as the vertical component of the
  root's up axis, which is 1 when the root is upright and 0 when it is horizontal, so the default of 0.8
  corresponds to a large difference in tilt.

Both trajectory-based handlers never terminate an environment that has no trajectory.

### Smooth termination for gradient-based methods

Algorithms that differentiate through the simulation or through a learned model cannot use a binary
termination flag. For them a terminal state handler can implement `mjx_absorbing_prob`, a smooth
probability of termination in which every hard threshold is replaced by a sigmoid. The three handlers above
provide it. [SuperMimic](../training/supermimic.md) uses it for its imagined rollouts.

## Absorbing and done

`step` returns two flags, and the difference matters for value estimation:

| Flag | True when | Meaning for the algorithm |
|---|---|---|
| `absorbing` | The terminal state handler says so | The episode failed. The value of the state is zero, and there is no bootstrapping. |
| `done` | `absorbing` is true, or the horizon is reached, or the reference clip has ended, or the goal reports that it is finished | The episode is over and the environment has to be reset. If `absorbing` is false, this is a time limit, and the value of the next state should still be bootstrapped. |

The horizon is the constructor argument `horizon` with a default of 1000 steps. The end of the reference
clip only ends the episode if the goal or the reward needs the trajectory.

On the CPU you call `reset` yourself when `done` is true. The MJX step function resets a finished
environment automatically and returns the first observation of the new episode. The last observation of the
finished episode is kept in `state.additional_carry.final_observation`. An MJX environment is also marked as
done when its observation contains a NaN, which catches a diverged simulation.

## Defaults of the task factories

The base class has neutral defaults, and the task factories replace them with what the task needs. Any of
them can be overridden with the keyword arguments above.

| Module | {class}`~loco_mujoco.task_factories.rl_factory.RLFactory` | {class}`~loco_mujoco.task_factories.imitation_factory.ImitationFactory` | Base class |
|---|---|---|---|
| Initial state | `DefaultInitialStateHandler` | `TrajInitialStateHandler` | `DefaultInitialStateHandler` |
| Terminal state | `HeightBasedTerminalStateHandler` | `RootPoseTrajTerminalStateHandler` | `NoTerminalStateHandler` |
| Goal | `GoalRandomRootVelocity` | `NoGoal` | `NoGoal` |
| Reward | `TargetVelocityGoalReward` | `NoReward` | `NoReward` |

## See also

- [Trajectories](trajectories.md) for the trajectory handlers that choose the start frame.
- [Rewards](rewards.md) and [observations](observations.md#goals) for the other modules that define a task.
- [Writing custom modules](custom_modules.md#initial-and-terminal-state-handlers) for your own handlers.
- API: {mod}`loco_mujoco.core.initial_state_handler`, {mod}`loco_mujoco.core.terminal_state_handler`.
