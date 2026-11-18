# Trajectories

A trajectory is the reference motion of an imitation task: a time series of joint positions and velocities of
one embodiment, optionally with the body and site quantities derived from them. All motion data in
LocoMuJoCo, whether it comes from the bundled datasets, from AMASS or from your own files, is brought into one
format, the {class}`~loco_mujoco.core.trajectory.dataclasses.Trajectory`. The environment uses it to
initialize episodes, to compute tracking rewards and goals, to decide when the robot has left the motion, and
to produce demonstration data for adversarial imitation learning.

```python
from loco_mujoco import ImitationFactory

env, traj = ImitationFactory.make("UnitreeH1", default_dataset_conf=dict(task=["squat", "stepinplace1"]))

print(traj.info.frequency)           # 100.0, the control frequency of the environment
print(traj.data.n_trajectories)      # 2 clips
print(traj.data.n_samples)           # 11990 samples in total
print(traj.info.joint_names[:3])     # ['root', 'hip_rotation_l', 'hip_adduction_l']

env.play_trajectory(n_episodes=2, n_steps_per_episode=500)   # replay it in the viewer
```

## The trajectory format

A {class}`~loco_mujoco.core.trajectory.dataclasses.Trajectory` is a dataclass with four parts:

| Field | Type | Content |
|---|---|---|
| `info` | {class}`~loco_mujoco.core.trajectory.dataclasses.TrajectoryInfo` | Static description: names, frequency and a reduced copy of the model |
| `data` | {class}`~loco_mujoco.core.trajectory.dataclasses.TrajectoryData` | The time series, as arrays |
| `transitions` | {class}`~loco_mujoco.core.trajectory.dataclasses.TrajectoryTransitions`, optional | Observations computed from the data, see [below](#transition-datasets) |
| `obs_container` | `ObservationContainer`, optional | The observation specification the transitions were created with |

Only `data` consists of arrays that JAX traces. `info` is static, so a trajectory can be passed through
`jax.jit` and `jax.vmap` as an argument.

### TrajectoryData

{class}`~loco_mujoco.core.trajectory.dataclasses.TrajectoryData` holds the samples of all clips stacked along
the first axis. With `N` samples in total:

| Field | Shape | Content |
|---|---|---|
| `qpos` | `(N, nq)` | Joint positions in MuJoCo's convention. A free joint takes seven entries: position and a quaternion with the scalar first. |
| `qvel` | `(N, nv)` | Joint velocities. A free joint takes six entries. |
| `xpos` | `(N, nbody, 3)` | Body positions in the world frame |
| `xquat` | `(N, nbody, 4)` | Body orientations in the world frame, scalar first |
| `cvel` | `(N, nbody, 6)` | Body velocities, angular then linear, as in MuJoCo's `cvel` |
| `subtree_com` | `(N, nbody, 3)` | Center of mass of the subtree below each body |
| `site_xpos` | `(N, nsite, 3)` | Site positions in the world frame |
| `site_xmat` | `(N, nsite, 9)` | Site orientations as flattened rotation matrices |
| `split_points` | `(n_trajectories + 1,)` | Index of the first sample of each clip, followed by the total number of samples |

Only `qpos`, `qvel` and `split_points` are required. The other fields may be empty; a trajectory with all
fields filled is called *complete* (`traj.data.is_complete`). Clip `i` occupies the samples
`split_points[i]` to `split_points[i + 1] - 1`.

The same fields without the leading sample axis and without `split_points` form a
{class}`~loco_mujoco.core.trajectory.dataclasses.SingleData`, which is one sample of a trajectory.
`traj.data.get(traj_index, step_index)` returns one, and accessors such as `get_qpos(traj_index, step_index)`
and `get_qpos_slice(traj_index, step_index, length)` return a single field.

### TrajectoryInfo and TrajectoryModel

{class}`~loco_mujoco.core.trajectory.dataclasses.TrajectoryInfo` describes what the columns of the data mean:

| Field | Content |
|---|---|
| `joint_names` | Names of the joints, in the order of `qpos` and `qvel` |
| `frequency` | Sampling frequency in Hz |
| `body_names` | Names of the bodies in the order of `xpos`, or `None` if the trajectory holds no body data |
| `site_names` | Names of the sites in the order of `site_xpos`, or `None` if the trajectory holds no site data |
| `model` | A {class}`~loco_mujoco.core.trajectory.dataclasses.TrajectoryModel` |
| `metadata` | Optional free-form dictionary |

It also builds lookup tables from a name to its indices: `joint_name2ind_qpos`, `joint_name2ind_qvel`,
`body_name2ind` and `site_name2ind`. For example, the knee angle over a whole clip is:

```python
knee = traj.data.qpos[:, traj.info.joint_name2ind_qpos["knee_angle_l"]]
```

{class}`~loco_mujoco.core.trajectory.dataclasses.TrajectoryModel` is a reduced copy of the MuJoCo model the
trajectory belongs to. It stores the number and types of the joints (`njnt`, `jnt_type`) and, for complete
trajectories, the kinematic tree and local frames of the bodies and sites (`nbody`, `body_rootid`,
`body_weldid`, `body_mocapid`, `body_pos`, `body_quat`, `body_ipos`, `body_iquat`, `nsite`, `site_bodyid`,
`site_pos`, `site_quat`). With it, a trajectory file describes its own structure and can be checked against
the model of an environment. Only free, hinge and slide joints are supported.

## What happens when a trajectory is loaded

{meth}`~loco_mujoco.environments.base.LocoEnv.process_trajectory` attaches a trajectory to an environment. The
[task factory](task_factories.md) calls it for you. It runs four steps:

1. **Match the model.** Joints, bodies and sites that the trajectory contains but the model of the
   environment does not are removed. Elements that the model has but the trajectory lacks are added and
   filled with zeros. Everything is reordered to the order of the model. A trajectory recorded with the full
   robot can therefore be loaded into an environment with, for example, disabled arms.
2. **Resample.** If the frequency of the trajectory differs from the control frequency of the environment
   (`1 / env.dt`), every clip is interpolated to the control frequency: cubic interpolation for positions and
   velocities, spherical linear interpolation for quaternions and rotation matrices. The bundled datasets are
   stored at 40 Hz and are resampled to 100 Hz for the default environments.
3. **Create the trajectory handler**, described [below](#the-trajectory-handler).
4. **Pad.** The arrays are padded with zeros to a fixed number of samples and clips, and unused entries of
   `split_points` are set to `-1`. The compiled step function then has the same input shapes for every
   trajectory, so one trajectory can be swapped for another without recompiling.

The method returns the processed trajectory, which is also what `ImitationFactory.make` returns as `traj`.
Because of the padding, `traj.data.qpos.shape[0]` is the padded size, not the number of samples. Use
`traj.data.n_samples` and `traj.data.n_trajectories`, or remove the padding with `traj.data.trim()`, which
returns NumPy arrays:

```python
data = traj.data.trim()
print(data.qpos.shape, data.split_points)    # (11990, 26) [    0  5995 11990]
```

```{important}
The padded size defaults to 100,000 samples and 100 clips, which is 1,000 seconds of motion at 100 Hz. Larger
datasets raise an `AssertionError` that asks you to increase the limits. Pass them through the trajectory
handler parameters: `th_params=dict(max_n_samples=500_000, max_n_trajs=300)`.
```

## The trajectory handler

The {class}`~loco_mujoco.core.trajectory.handler.TrajectoryHandler`, available as `env.th`, decides where in
the trajectory an episode starts and advances the position in the trajectory with every environment step. It
stores only the static `TrajectoryInfo`; the data stays with the environment or is passed in as an argument.

### The trajectory state

The position in the trajectory is part of the environment state, like every other piece of per-environment
state. It is a {class}`~loco_mujoco.core.trajectory.handler.TrajState`:

| Field | Meaning |
|---|---|
| `traj_no` | Index of the current clip |
| `subtraj_step_no` | Index of the current sample within the clip |
| `subtraj_step_no_init` | Sample at which the episode started |
| `next_traj_no`, `next_subtraj_step_no` | Start of the next episode if an algorithm chose it, otherwise `-1` |

On the GPU backends it is found at `state.additional_carry.traj_state`, with one entry per parallel
environment, so every environment follows its own clip at its own phase. Each environment step increases
`subtraj_step_no` by one. If a goal or a reward uses the trajectory, the episode is done when the last sample
of the clip is reached. Otherwise the handler continues with the first sample of the next clip.

Components read the reference sample of the current step with
`env.th.get_current_traj_data(traj.data, carry, backend)`, which returns a `SingleData`.

### Choosing where episodes start

Three handlers are registered. Select one with the `name` entry of `th_params`; the remaining entries are
passed to its constructor.

| `name` | Episode start | Additional parameters |
|---|---|---|
| `RandomStartTrajectoryHandler` (default) | A random clip, then a random sample of that clip | |
| `RandomTrajFixedStepTrajectoryHandler` | A random clip at a fixed sample | `fixed_step` (default `0`) |
| `FixedStartTrajectoryHandler` | Always the same clip and sample | `start_conf=(traj_no, step_no)` (default `(0, 0)`) |

```python
# always start at sample 100 of the second clip, for example for a reproducible evaluation
env, traj = ImitationFactory.make("UnitreeH1",
                                  default_dataset_conf=dict(task=["squat", "stepinplace1"]),
                                  th_params=dict(name="FixedStartTrajectoryHandler", start_conf=(1, 100)))
```

With the default handler every clip is equally likely regardless of its length, so samples of short clips are
visited more often than samples of long ones. All handlers accept `max_n_samples` and `max_n_trajs`.

The handler only chooses the reference sample. Whether the robot is actually placed in that state is decided
by the [initial state handler](initial_terminal_states.md); the `ImitationFactory` selects
`TrajInitialStateHandler`, which does so. By default the motion is shifted so that an episode starts at the
horizontal origin of its environment; the environment argument `recenter_root_xy` controls this.

## NumPy and JAX

The arrays of a trajectory are either NumPy or JAX arrays. Loading and processing produce JAX arrays, which is
what `mjx_reset` and `mjx_step` require; they raise an error for a NumPy trajectory. The CPU methods `reset`,
`step` and `play_trajectory` accept both and convert internally where needed. Convert explicitly with:

```python
traj_np = traj.replace(data=traj.data.to_numpy())
traj_jax = traj_np.replace(data=traj_np.data.to_jax())
```

Methods that accept either take a `backend` argument (`numpy` or `jax.numpy`), for example
`traj.data.get(0, 10, backend=np)`.

## Passing trajectories to the functional interface

`mjx_reset` and `mjx_step` take the trajectory as an optional last argument:

```python
import jax

mjx_reset = jax.jit(jax.vmap(env.mjx_reset, in_axes=(0, None)))
mjx_step = jax.jit(jax.vmap(env.mjx_step, in_axes=(0, 0, None)))

state = mjx_reset(keys, traj)
state = mjx_step(state, action, traj)
```

The `None` in `in_axes` shares one trajectory between all environments. If the argument is omitted, the
trajectory stored in the environment is used and is compiled into the function as a constant. Passing it keeps
the data an input, so it can be exchanged between calls, for example to train on a different subset of a large
dataset in each phase of a curriculum. A replacement must have the same joints, bodies and sites and must be
processed by the same environment, which gives it the same padded shape.

## Replaying a trajectory

{meth}`~loco_mujoco.environments.base.LocoEnv.play_trajectory` sets the simulation to the samples of the
trajectory one after another, without running the physics. It is the quickest way to check a dataset or a
retargeting result.

```python
env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)
```

| Argument | Meaning |
|---|---|
| `n_episodes`, `n_steps_per_episode` | Number of episodes and steps to replay. Each episode starts where the trajectory handler puts it. |
| `from_velocity` | Integrate the joint velocities instead of setting the joint positions, which reveals inconsistent velocities |
| `render`, `record`, `recorder_params` | Show the viewer and record a video, see [rendering](rendering.md) |
| `viser`, `usd`, `usd_params` | Use the browser viewer, or export an animated USD stage instead |
| `callback_class` | A callable invoked at every step, used internally to record derived quantities |

## Transition datasets

Adversarial imitation algorithms such as [AMP and GAIL](../training/amp_gail.md) compare the observations of
the policy with observations of the reference motion.
{meth}`~loco_mujoco.environments.base.LocoEnv.create_dataset` produces the latter: it sets the simulation to
every sample of the trajectory, evaluates the observation of the environment, and returns a
{class}`~loco_mujoco.core.trajectory.dataclasses.TrajectoryTransitions`:

| Field | Shape | Content |
|---|---|---|
| `observations` | `(M, obs_dim)` | Observation at each sample |
| `next_observations` | `(M, obs_dim)` | Observation at the following sample |
| `absorbings` | `(M,)` | Always zero: a reference motion is assumed to contain no terminal states, and a warning is issued if one is found |
| `dones` | `(M,)` | One for the last transition of each clip, zero otherwise |
| `actions`, `rewards` | Empty | Motion capture data contains no actions and no rewards |

`M` is the number of samples minus the number of clips, because the last sample of a clip has no successor.

```python
transitions = env.create_dataset()
print(transitions.observations.shape)
```

The observations follow the observation specification of the environment, so the dataset has to be created
with the environment that is trained. The method steps through the samples one by one on the CPU, which takes
a few minutes for tens of thousands of samples. The result is stored in the trajectory of the environment and
returned directly on later calls.

## Saving and loading

A trajectory is saved to a single NumPy archive and loaded back with:

```python
from loco_mujoco.core.trajectory import Trajectory

traj.save("my_motion.npz")
traj = Trajectory.load("my_motion.npz")              # JAX arrays
traj = Trajectory.load("my_motion.npz", backend=np)  # NumPy arrays
```

The archive contains the fields of `data`, `info` and the `TrajectoryModel` under their own names, and the
transitions if they were created. Several trajectories with identical `info` are joined with
`Trajectory.concatenate([traj_a, traj_b])`.

A saved trajectory can be loaded into an environment through
{class}`~loco_mujoco.task_factories.dataset_confs.CustomDatasetConf`, with `env.process_trajectory(traj)`, or
with the environment argument `traj_params=dict(traj_path="my_motion.npz")`.
[Creating custom trajectories](custom_trajectories.md) shows how to build one from your own data.

## See also

- [Datasets](datasets.md) for the motion data that ships with LocoMuJoCo.
- [Creating custom trajectories](custom_trajectories.md) for building a trajectory from joint data.
- [Initial and terminal states](initial_terminal_states.md), [rewards](rewards.md) and
  [observations](observations.md) for the components that consume the trajectory.
- [Evaluation](../training/evaluation.md) for the metrics that compare a rollout with the trajectory.
- API reference: {mod}`loco_mujoco.core.trajectory.dataclasses`, {mod}`loco_mujoco.core.trajectory.handler`.
