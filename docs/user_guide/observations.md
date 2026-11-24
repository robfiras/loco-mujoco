# Observations

The observation of an environment is a flat vector that is assembled from a list of **observation
objects**, the `observation_spec`. Each object contributes a few entries, for example the angle of one
joint or the position of one body. Every robot comes with a default list, and you can replace it with your
own to choose what the agent sees and in which order.

```{literalinclude} ../../examples/tutorials/05_changing_the_observation_space.py
:language: python
```

The names used for `xml_name` are the names of the joints, bodies, sites and geoms in the MuJoCo XML of the
robot. The [environment pages](../environments/index.md) list the default observation of every robot.

## Defining an observation

All observation classes are collected in the namespace
{class}`~loco_mujoco.core.observations.base.ObservationType`. Their constructors share three arguments, which
come from the base class {class}`~loco_mujoco.core.observations.base.Observation`:

| Argument | Default | Meaning |
|---|---|---|
| `obs_name` | required | Unique name of the entry. It is the key in `env.obs_container`. |
| `group` | `None` | Name of a group, or a list of names, that the entry belongs to. See [Observation groups](#observation-groups). |
| `allow_randomization` | `True` | Whether the [domain randomizer](domain_randomization.md) may add noise to the entry. |

Instead of an object, an entry of `observation_spec` can be a dictionary with the key `type` and the
constructor arguments. This form is used in YAML configurations:

```python
observation_spec = [
    dict(type="FreeJointPosNoXY", obs_name="q_root", xml_name="root"),
    dict(type="JointPos", obs_name="q_knee_l", xml_name="knee_angle_l"),
    dict(type="LastAction", obs_name="last_action", group="policy"),
]
env, _ = RLFactory.make("UnitreeH1", observation_spec=observation_spec)
```

The entries appear in the observation vector in the order of the list. The [goal](#goals) is appended as the
last entry.

## Built-in observation types

Most types read one named element directly from the simulation state. The column *source* gives the array
of `mujoco.MjData` (or `mjx.Data`) the values are taken from.

| Type | Dim | Content | Source | Specific arguments |
|---|---|---|---|---|
| **Joints** | | | | |
| `JointPos` | 1 | Position of a hinge or slide joint | `qpos` | `xml_name` |
| `JointVel` | 1 | Velocity of a hinge or slide joint | `qvel` | `xml_name` |
| `JointPosArray` | n | Positions of several joints in one entry | `qpos` | `xml_names` |
| `JointVelArray` | n | Velocities of several joints in one entry | `qvel` | `xml_names` |
| **Free joint (floating base)** | | | | |
| `FreeJointPos` | 7 | Position (3) and quaternion (4, scalar first) | `qpos` | `xml_name` |
| `FreeJointPosNoXY` | 5 | Height and quaternion, without the horizontal position | `qpos` | `xml_name` |
| `EntryFromFreeJointPos` | 1 | One entry of the 7D pose | `qpos` | `xml_name`, `entry_index` |
| `FreeJointVel` | 6 | Linear (3) and angular (3) velocity | `qvel` | `xml_name` |
| `EntryFromFreeJointVel` | 1 | One entry of the 6D velocity | `qvel` | `xml_name`, `entry_index` |
| `ProjectedGravityVector` | 3 | Unit gravity direction expressed in the frame of the free joint | computed from `qpos` | `xml_name` |
| **Bodies** | | | | |
| `BodyPos` | 3 | Position of the body frame in the world | `xpos` | `xml_name` |
| `BodyRot` | 4 | Orientation of the body frame as a quaternion | `xquat` | `xml_name` |
| `BodyVel` | 6 | Angular (3) and linear (3) velocity, MuJoCo's center-of-mass based `cvel` | `cvel` | `xml_name` |
| **Sites** | | | | |
| `SitePos` | 3 | Position of the site in the world | `site_xpos` | `xml_name` |
| `SiteRot` | 9 | Flattened rotation matrix of the site | `site_xmat` | `xml_name` |
| `RelativeSiteQuantaties` | 12 per site | Position (3), rotation vector (3) and velocity (6) of the mimic sites relative to the main site | computed | `site_names` |
| **Contacts** | | | | |
| `Force` | 6 | Contact force (3) and torque (3) between two geoms | computed | `xml_name_geom1`, `xml_name_geom2` |
| **Actions, model and terrain** | | | | |
| `LastAction` | action dim | The most recent action that was passed to the environment | carry | none |
| `ModelInfo` | size of the attributes | Flattened attributes of the model, for example `body_mass` | model | `model_attributes` |
| `HeightScan` | grid points | Terrain heights on a grid around the root | terrain | see below |

Notes on individual types:

- **Joint limits become bounds.** `JointPos` and `JointPosArray` take the limits of the observation space
  from the joint range in the model. All other types are unbounded. The bounds are available as
  `env.info.observation_space.low` and `.high`.
- **`FreeJointVel` and `BodyVel` are different quantities.** The first one is the entry of `qvel` of the free
  joint, with the linear velocity first. The second one is `cvel` of a body, with the angular velocity first.
- **`RelativeSiteQuantaties`** (the spelling is that of the class) uses the mimic sites of the robot if
  `site_names` is not given. The first site of the list is the main site, and the entry has
  $12 \cdot (n_\text{sites} - 1)$ values. The positions, rotations and velocities are expressed in the frame
  of the main site, so they do not change when the robot turns or moves in the world. This is the usual
  observation for AMP and GAIL discriminators.
- **`Force`** is only computed on the CPU backend. On MJX the entry is zero.
- **`ModelInfo`** makes parameters of the model visible, for example to give a critic the masses that the
  domain randomizer sampled: `ObservationType.ModelInfo("masses", model_attributes="body_mass")`.

### Height scan

{class}`~loco_mujoco.core.observations.base.HeightScan` samples the [terrain](terrain.md) on a regular grid
that is centered on the root and aligned with its heading. Each entry is
`root_height - terrain_height - offset`, clipped to `clip`.

| Argument | Default | Meaning |
|---|---|---|
| `size` | `(1.6, 1.0)` | Extent of the grid forward and sideways, in meters |
| `resolution` | `0.1` | Grid spacing in meters. The default grid has 17 x 11 = 187 points. |
| `forward_offset` | `0.0` | Shift of the grid center along the heading, in meters |
| `offset` | `0.0` | Subtracted from every entry, for example the nominal root height |
| `clip` | `(-1.0, 1.0)` | Lower and upper clipping value |
| `xml_name` | `None` | Free joint that gives the root pose. Defaults to the root joint of the environment. |
| `differentiable` | `False` | Whether gradients flow through the scan into the root pose |

```python
observation_spec = [..., ObservationType.HeightScan("height_scan", size=(1.6, 1.0), resolution=0.1)]
```

The scan works with all built-in terrains and on all backends.

## Goals

A goal is an observation that tells the policy what to do. There is exactly one goal per environment, it
is selected with `goal_type` and `goal_params` instead of being part of `observation_spec`, and it is always
the last entry of the observation, under the name of its class.

| `goal_type` | Dim | Content | Parameters (defaults) |
|---|---|---|---|
| `NoGoal` | 0 | Nothing. The default of the environment. | none |
| `GoalRandomRootVelocity` | 3 | A target velocity of the root (forward, lateral, yaw rate), sampled uniformly at every reset | `max_x_vel` (1.0), `max_y_vel` (1.0), `max_yaw_vel` (1.0) |
| `GoalTrajRootVelocity` | 6 | Linear and angular velocity of the root in the reference trajectory, averaged over the next frames and expressed in the root frame of the reference | `n_steps_average` (3) |
| `GoalTrajMimic` | robot dependent | The current frame of the reference trajectory, see below | `sites_for_mimic`, `main_site_name`, `rel_body_names` |
| `GoalTrajMimicv2` | as `GoalTrajMimic` | The same observation, with the reference drawn as a translucent copy of the robot | `target_geom_rgba`, `rel_body_names` |

All goals accept `visualize_goal=True` in `goal_params`, except `NoGoal`. The velocity goals then draw arrows
at the root, `GoalTrajMimic` draws the target positions of the mimic sites, and `GoalTrajMimicv2` draws the
whole reference robot. `GoalTrajMimicv2` adds one visual geometry per geometry of the robot and can slow
the simulation down, so it is meant for visualization and evaluation.

{class}`~loco_mujoco.core.observations.goals.GoalRandomRootVelocity` is the default goal of
{class}`~loco_mujoco.task_factories.rl_factory.RLFactory` and pairs with `TargetVelocityGoalReward` or
`LocomotionReward`. The goals that read the trajectory need an environment with a loaded trajectory, that
is, one created with {class}`~loco_mujoco.task_factories.imitation_factory.ImitationFactory`. With
`GoalTrajRootVelocity`, an episode ends when fewer than `n_steps_average` frames of the clip are left.

### The mimic goal

{class}`~loco_mujoco.core.observations.goals.GoalTrajMimic` is the goal for motion tracking
([DeepMimic-style training](../training/deepmimic.md)). It describes the current frame of the reference
motion in two ways, by joints and by *mimic sites*, the keypoints that every robot defines on its pelvis,
head, hands, feet and other links. Its content is, in this order:

1. the position, rotation vector and velocity of the robot's own mimic sites relative to its main site,
2. the joint positions of the reference (the free joint without its horizontal position),
3. the joint velocities of the reference,
4. the position, rotation vector and velocity of the reference's mimic sites relative to its main site.

For a robot with $n_j$ hinge joints and $n_s$ mimic sites the goal has
$(5 + n_j) + (6 + n_j) + 2 \cdot 12\,(n_s - 1)$ entries. `sites_for_mimic` restricts the sites (an empty list
removes all site quantities), and `main_site_name` chooses the reference site, which is the first one by
default.

### Trajectory observations for a free root

Three further classes in {mod}`loco_mujoco.core.observations.goals` read the reference trajectory of a robot
whose first joint is the free root joint. They were added for [SuperMimic](../training/supermimic.md).
Although they derive from the goal class, they are used as entries of `observation_spec`, not as
`goal_type`, and they take the name of their class instead of an `obs_name`.

| Type | Dim | Content | Parameters (defaults) |
|---|---|---|---|
| `TrajFutureFrames` | $n_\text{future}\,(15 + 2 n_j)$ | The next reference frames: root position and orientation in the heading frame of the current reference frame, root velocities, joint positions and scaled joint velocities | `n_future` (10), `joint_vel_scale` (0.1) |
| `RootOriRelTraj` | 6 | Orientation of the reference root in the frame of the robot's root, as the first two columns of the rotation matrix | none |
| `TrackingError` | $n_j + 3 n_b + 3$ | Errors between robot and reference for a critic: joint positions, positions of $n_b$ bodies and the root position | `body_names` |

```python
observation_spec = [
    ObservationType.FreeJointPosNoXY("q_root", xml_name="root", group=["actor", "critic"]),
    ObservationType.TrajFutureFrames(n_future=5, group="actor"),
    ObservationType.RootOriRelTraj(group="actor"),
    ObservationType.TrackingError(body_names=["pelvis"], group="critic"),
]
env, traj = ImitationFactory.make("MjxUnitreeH1", default_dataset_conf=dict(task="walk"),
                                  observation_spec=observation_spec)
```

```{note}
These three observations are computed in the MJX step functions (`mjx_reset`, `mjx_step`). The CPU step
functions `reset` and `step` do not pass the trajectory to observations and cannot compute them.
```

## Observation groups

Different parts of an algorithm often need different observations. The standard case is an asymmetric
actor-critic setup: the policy sees only what is measurable on the real robot, with noise, while the critic
sees the full and clean simulation state. Groups express this without a second environment. Every entry can
be assigned to one or more groups, and the container returns the indices of a group as a mask for the
observation vector.

```{literalinclude} ../../examples/tutorials/06_changing_the_observation_space_grouping.py
:language: python
```

The methods of `env.obs_container`
({class}`~loco_mujoco.core.observations.base.ObservationContainer`) that work with groups are:

| Method | Returns |
|---|---|
| `get_obs_ind_by_group(group)` | Indices of all entries of a group. A list of groups concatenates them in the given order. |
| `filter_by_group(obs, group)` | The entries of a group from an observation, also for batched observations |
| `get_all_group_names()` | All group names in use. Entries without a group are in the group `None`. |
| `get_randomizable_obs_indices()` | Indices of all entries with `allow_randomization=True` |

The index arrays are plain NumPy arrays and can be used inside jitted functions. The order of a mask is the
input layout of the network that consumes it, so changing the groups or the order of `observation_spec`
invalidates trained checkpoints.

The training algorithms take the groups from the configuration: `actor_obs_group` and `critic_obs_group`
select the inputs of the two networks in PPO, GAIL and AMP, see
[training configuration](../training/configuration.md). To give the actor noisy copies of entries that the
critic sees clean, add the entry twice under different names, as the
[domain randomization example](domain_randomization.md) does.

## Inspecting the observation space

When an environment is created, it prints a table of its observation to the terminal. Pass
`print_obs_summary=False` to turn this off.

| Method | Effect |
|---|---|
| `env.print_observation_summary()` | Prints one table with all entries (indices, name, type, bounds, groups, whether noise is allowed) and one table per group with the indices inside the filtered observation |
| `env.create_observation_summary(filename=None, open_in_browser=True)` | Writes the same tables to an HTML file and opens it in the browser. Without a browser the tables are printed instead. Returns the path of the file. |

A single entry can be located through the container, which is a dictionary from name to observation object:

```python
obs = env.reset()
knee = env.obs_container["q_knee_l"]
knee.obs_ind             # indices in the observation vector
knee.min, knee.max       # bounds
obs[knee.obs_ind]        # value in an observation
```

## Stateful observations and observation history

Observations that cannot be read from one array of the simulation state derive from
{class}`~loco_mujoco.core.observations.base.StatefulObservation`. They compute their value in
`get_obs_and_update_state` and may keep state between steps in `carry.observation_states`, for example a
moving average or a sampled goal. `LastAction`, `ModelInfo`, `RelativeSiteQuantaties`, `HeightScan` and all
goals are of this kind. See [Writing custom modules](custom_modules.md#observations) for an example.

A history of past observations is not an observation type. It is added by a wrapper around the MJX
environment, {class}`~loco_mujoco.core.wrappers.mjx.NStepWrapper`, which is described on the
[Gymnasium and wrappers page](gymnasium.md#wrappers-for-mjx-environments).

## See also

- [Modular architecture](modular_architecture.md) for how modules are selected and where their state lives.
- [Rewards](rewards.md) for the rewards that pair with each goal.
- [Domain randomization](domain_randomization.md) for observation noise.
- [Writing custom modules](custom_modules.md) for your own observation types.
- API: {mod}`loco_mujoco.core.observations.base`, {mod}`loco_mujoco.core.observations.goals`.
