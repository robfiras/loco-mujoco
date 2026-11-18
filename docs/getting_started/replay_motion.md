# Replay a motion

<ul class="lmj-steps">
  <li><a href="installation.html">1 · Installation</a></li>
  <li><a href="first_environment.html">2 · First environment</a></li>
  <li class="current">3 · Replay a motion</li>
  <li><a href="train_policy.html">4 · Train a policy</a></li>
  <li><a href="next_steps.html">5 · Evaluate and customize</a></li>
</ul>

LocoMuJoCo ships motion capture data that is already retargeted to every humanoid. On this page you load a
few clips and play them back on the robot. This is the data a policy will later learn to imitate.

```{image} ../../imgs/unitree_g1_2.gif
:alt: The Unitree G1 replaying a motion capture clip
:width: 45%
:align: center
```

## Load a dataset and play it

{class}`~loco_mujoco.task_factories.imitation_factory.ImitationFactory` creates an environment together with
its motion data. The data to load is described by a dataset configuration:

```python
from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf

env, traj = ImitationFactory.make("UnitreeG1", default_dataset_conf=DefaultDatasetConf(["walk"]))

env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)
```

The first call downloads the clip from the
[LocoMuJoCo dataset repository](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets) and caches it.
A viewer opens and the robot walks. Each episode starts at a random point of the clip.

```{note}
`play_trajectory` does not simulate. It sets the robot to the recorded joint positions frame by frame, so
you see the reference motion itself and not a controller following it.
```

## Combine several sources

There are three sources of motion data, each with its own configuration class. They can be mixed freely in
one environment:

| Source | Configuration | Contents | Setup |
|---|---|---|---|
| Default | {class}`~loco_mujoco.task_factories.dataset_confs.DefaultDatasetConf` | Locomotion clips collected by the LocoMuJoCo maintainers, such as `walk`, `run`, `squat` and `stepinplace1`. | None, downloaded on demand. |
| LAFAN1 | {class}`~loco_mujoco.task_factories.dataset_confs.LAFAN1DatasetConf` | The LAFAN1 motion capture dataset: walking, running, dancing, fighting, getting up and more. | None, downloaded on demand. |
| AMASS | {class}`~loco_mujoco.task_factories.dataset_confs.AMASSDatasetConf` | The AMASS collection with more than 10,000 motions. | [Manual setup](amass.md), license-restricted. |

```python
from loco_mujoco.task_factories import ImitationFactory, DefaultDatasetConf, LAFAN1DatasetConf

env, traj = ImitationFactory.make(
    "UnitreeG1",
    default_dataset_conf=DefaultDatasetConf(["squat"]),
    lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4"]),
)

env.play_trajectory(n_episodes=5, n_steps_per_episode=500, render=True)
```

Each episode now starts in a randomly chosen clip. The same code works for every humanoid: replace
`"UnitreeG1"` with `"UnitreeH1"`, `"Atlas"` or `"SkeletonTorque"` and the clips are loaded for that embodiment.
The names of all available clips are listed in [Datasets](../user_guide/datasets.md).

```{tip}
An environment holds at most 100,000 trajectory samples by default, which is about 16 minutes of motion at
100 Hz. To load more, raise the limit with `th_params=dict(max_n_samples=200_000)` in `ImitationFactory.make`.
```

## What a trajectory contains

`traj` is a {class}`~loco_mujoco.core.trajectory.dataclasses.Trajectory`. Its two main parts are the static
description of the model and the motion data:

```python
print(traj.info.frequency)        # 100.0, in Hz, resampled to the control frequency of the environment
print(traj.info.joint_names[:3])  # ['root', 'left_hip_pitch_joint', 'left_hip_roll_joint']
print(traj.data.qpos.shape)       # (max_n_samples, 30): joint positions, padded to a fixed length
print(traj.data.split_points)     # where one clip ends and the next begins
```

Besides joint positions and velocities, `traj.data` holds the position, orientation and velocity of every
body and the position and orientation of every site. These are computed with forward kinematics when the
clip is loaded. [Trajectories](../user_guide/trajectories.md) describes the format in full.

## What the environment does with it

With a trajectory loaded, the environment uses it in three places by default:

Initial state
: Each episode starts from a randomly sampled frame of the data instead of a fixed pose.

Termination
: An episode ends when the root pose leaves the range seen in the data, for example when the robot falls.

Goals and rewards
: Observations and rewards can read the reference motion, for example to reward tracking it. You will use
  this on the next page.

## Turn the data into transitions

Adversarial imitation methods such as GAIL and AMP learn from observation transitions instead of tracking a
reference. The environment can build them from the loaded trajectory:

```python
transitions = env.create_dataset()
print(transitions.observations.shape, transitions.next_observations.shape)
```

## What you learned

- `ImitationFactory.make` returns an environment and the trajectory it will imitate.
- Default, LAFAN1 and AMASS data can be combined, and the same clip names work across humanoids.
- `env.play_trajectory` replays the reference motion kinematically.

Next: [train a tracking policy](train_policy.md).
