# Creating custom trajectories

You can use your own motion as a reference: a motion from another dataset, the output of a trajectory
optimizer or a hand-crafted movement. All you need are the joint positions of the embodiment over time. This
page shows how to turn them into a {class}`~loco_mujoco.core.trajectory.dataclasses.Trajectory`, how to save
and load it, and how to create an imitation environment with it.

The format itself is described under [trajectories](trajectories.md). If your motion is for a different body
than the one you want to train, see [retargeting](retargeting.md) first.

## A complete example

The tutorial `examples/tutorials/10_creating_custom_traj.py` creates a trajectory in which the Unitree H1
stands still and swings its left elbow, and replays it:

```{literalinclude} ../../examples/tutorials/10_creating_custom_traj.py
:language: python
```

`examples/trajectory_generation/standing_humanoid.py` is a shorter variant without the elbow motion that also
saves and reloads the file. The sections below go through the steps.

## Step by step

### 1. Arrange the joint data

A trajectory needs two arrays with one row per time step:

| Array | Shape | Content |
|---|---|---|
| `qpos` | `(N, nq)` | Joint positions in the order of the MuJoCo model. The floating base comes first with seven entries: position `x, y, z` and orientation as a quaternion `w, x, y, z`. Each hinge or slide joint adds one entry. |
| `qvel` | `(N, nv)` | Joint velocities in the same order. The floating base has six entries: linear and angular velocity. |

Create the environment once to look up the model, the joint names and their order:

```python
import mujoco
from loco_mujoco.environments import UnitreeH1
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid

env = UnitreeH1()
model = env.get_model()

joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
elbow_index = mj_jntname2qposid("left_elbow", model)      # indices of a joint in qpos
```

The joints of your data do not have to cover the whole model, and they may be in a different order. When the
trajectory is loaded, joints are matched **by name**: joints that the environment does not have are dropped,
and joints that are missing in the trajectory are filled with zeros. What matters is that `joint_names` and
the joint types describe the columns of your arrays correctly.

### 2. Compute the velocities

If your source provides only positions, compute the velocities by finite differences. For a model whose first
joint is the floating base, a helper does this, including the conversion of the base orientation into an
angular velocity. It uses central differences and therefore returns two samples fewer:

```python
from loco_mujoco.datasets.data_generation import calculate_qvel_with_finite_difference

qpos, qvel = calculate_qvel_with_finite_difference(qpos, frequency)    # qpos loses its first and last row
```

### 3. Build the trajectory

```python
import jax.numpy as jnp
from loco_mujoco.core.trajectory import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData

traj_info = TrajectoryInfo(joint_names,
                           model=TrajectoryModel(model.njnt, jnp.array(model.jnt_type)),
                           frequency=frequency)

traj_data = TrajectoryData(jnp.array(qpos), jnp.array(qvel),
                           split_points=jnp.array([0, len(qpos)]))

traj = Trajectory(traj_info, traj_data)
```

`frequency` is the sampling frequency of your data in Hz. It does not have to match the environment: the
trajectory is resampled to the control frequency (`1 / env.dt`) when it is loaded.

`split_points` marks where clips begin. For a single clip it is `[0, N]`. To store several clips in one
trajectory, stack their samples and list the start of each clip followed by the total number of samples, for
example `[0, 500, 1200]` for two clips of 500 and 700 samples. Trajectories with the same joints can also be
joined afterwards with `Trajectory.concatenate([traj_a, traj_b])`.

### 4. Add body and site quantities

A trajectory with only `qpos` and `qvel` is sufficient for replaying, for initializing episodes from the
motion and for creating [transition datasets](trajectories.md#transition-datasets). Goals and rewards that
track bodies or sites, such as `GoalTrajMimic` and `MimicReward`, additionally need the positions,
orientations and velocities of the bodies and sites. Compute them once by replaying the trajectory in the
environment:

```python
from loco_mujoco.smpl.retargeting import extend_motion

traj = extend_motion("UnitreeH1", {}, traj)
print(traj.data.is_complete)       # True
```

The second argument is a dictionary of arguments for the environment that is created for the replay. Use the
arguments of the environment you will train, for example `dict(disable_arms=True)`, if they change the model.
The returned trajectory is resampled to the control frequency of that environment. This function does not
require the optional SMPL dependencies.

### 5. Save and load

```python
traj.save("my_motion.npz")
traj = Trajectory.load("my_motion.npz")
```

Saving the completed trajectory avoids repeating step 4 on every run.

### 6. Create an environment with it

Pass the trajectory to the [`ImitationFactory`](task_factories.md) with
{class}`~loco_mujoco.task_factories.dataset_confs.CustomDatasetConf`:

```python
from loco_mujoco.task_factories import ImitationFactory, CustomDatasetConf

env, traj = ImitationFactory.make("UnitreeH1",
                                  custom_dataset_conf=CustomDatasetConf(traj),
                                  goal_type="GoalTrajMimic",
                                  reward_type="MimicReward")

env.play_trajectory(n_episodes=1, n_steps_per_episode=500)
```

The factory sets the initial and terminal state handlers for imitation, as for the bundled datasets. A custom
trajectory can be combined with clips from the other [datasets](datasets.md) in the same call, provided that it
is complete, since all clips of an environment must have the same structure.

There are two other ways to attach a trajectory. `env.process_trajectory(traj)` attaches it to an existing
environment, as the tutorial does, and the environment argument `traj_params=dict(traj_path="my_motion.npz")`
loads a file when the environment is constructed. Neither changes the initial or terminal state handler.

`CustomDatasetConf` holds a `Trajectory` object. In a YAML configuration, which cannot contain one, either
build the trajectory in your training script and insert the configuration there, or use `traj_params` with the
path of the file.

## Checking the result

Replay the trajectory before training on it:

```python
env.play_trajectory(n_episodes=1, n_steps_per_episode=500)
env.play_trajectory(n_episodes=1, n_steps_per_episode=500, from_velocity=True)
```

The first call shows the positions. The second integrates the velocities instead; if the motion drifts away
or looks different, `qvel` is inconsistent with `qpos`, typically because of a wrong frequency or a wrong
quaternion convention.

Common problems:

| Symptom | Likely cause |
|---|---|
| The robot is rotated or tumbles | The base quaternion is not in the order `w, x, y, z` |
| Joints move in the wrong direction or the wrong joints move | The columns of `qpos` do not match `joint_names` |
| `Unsupported joint type` when creating `TrajectoryInfo` | The joint types passed to `TrajectoryModel` do not match the joint names; only free, hinge and slide joints are supported |
| An `IndexError` in a reward or goal on the first step | The trajectory holds no body or site data; run step 4 |
| `AssertionError` mentioning `max_n_samples` | The trajectory is longer than the default limit, see [trajectories](trajectories.md#what-happens-when-a-trajectory-is-loaded) |
| The motion plays too fast or too slowly | `frequency` does not match the data |

## See also

- [Trajectories](trajectories.md) for the data format and the trajectory handler.
- [Datasets](datasets.md) for the bundled motion data and for LocoMuJoCo Real's conversion of BONES-SEED.
- [Retargeting](retargeting.md) for transferring a trajectory to another embodiment.
- [Task factories](task_factories.md) for `CustomDatasetConf` and the other dataset configurations.
- API reference: {mod}`loco_mujoco.core.trajectory.dataclasses`.
