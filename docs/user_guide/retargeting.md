# Retargeting

Retargeting maps a motion that was recorded on one body onto another body with different proportions, joints
and joint limits. LocoMuJoCo uses it in two places:

- to bring human motion capture data, given in the format of the [SMPL](https://smpl.is.tue.mpg.de/) body
  model, onto a humanoid robot. This is how [AMASS](datasets.md) motions are loaded.
- to transfer a trajectory from one robot to another. This is how the default and LAFAN1 datasets reach an
  embodiment for which no pre-retargeted data exists, such as a robot you added yourself.

For the twelve built-in humanoids the default and LAFAN1 datasets are already retargeted, and nothing on this
page has to be run to use them. Retargeting happens automatically, on first use, in the other cases. This page
explains what it does, how to configure it and what to expect from the result.

```{note}
Both directions pass through the SMPL model and therefore need the SMPL part of the
[AMASS and SMPL setup](../getting_started/amass.md), including the `smpl` extra with PyTorch.
```

## The idea

Every humanoid model in LocoMuJoCo carries 15 *mimic sites*: marker frames attached to the pelvis, the upper
body, the head and to both hips, knees, feet, shoulders, elbows and hands. Each site corresponds to one joint
of the SMPL skeleton:

| Mimic site | SMPL joint | Mimic site | SMPL joint |
|---|---|---|---|
| `pelvis_mimic` | `Pelvis` | `left_shoulder_mimic` | `L_Shoulder` |
| `upper_body_mimic` | `Chest` | `left_elbow_mimic` | `L_Elbow` |
| `head_mimic` | `Head` | `left_hand_mimic` | `L_Wrist` |
| `left_hip_mimic` | `L_Hip` | `right_shoulder_mimic` | `R_Shoulder` |
| `left_knee_mimic` | `L_Knee` | `right_elbow_mimic` | `R_Elbow` |
| `left_foot_mimic` | `L_Ankle` | `right_hand_mimic` | `R_Wrist` |
| `right_hip_mimic` | `R_Hip` | `right_foot_mimic` | `R_Ankle` |
| `right_knee_mimic` | `R_Knee` | | |

A motion is retargeted by making the mimic sites of the robot follow the corresponding joints of an SMPL body
that has first been reshaped to the proportions of the robot. The same site names are also what the
site-based [mimic rewards](rewards.md) and goals track during training, so retargeting and tracking agree on
what "the same pose" means.

The pipeline has three stages. The first runs once per robot, the second once per motion, and the third only
when the source of a motion is another robot.

## Stage 1: fitting an SMPL shape to the robot

A human and a robot differ in limb lengths, so driving the robot directly with human joint positions would
ask for poses it cannot reach. The first stage therefore finds the SMPL body that looks most like the robot.

The robot is put into a T-pose, using the joint offsets given in its configuration file, and the positions of
its mimic sites are read. The SMPL model, also in a T-pose, is then optimized with Adam: its 16 shape
parameters and one global scale factor are adjusted until the mean distance between the SMPL joints and the
corresponding mimic sites is minimal (1,000 iterations at a learning rate of 0.05 by default).

Along with the shape and the scale, the stage stores for every site the remaining position offset and the
relative rotation between the SMPL joint frame and the site frame, a height offset of the feet and the ratio
of the body heights before and after the fit. The offsets account for the fact that SMPL joints lie inside the
body while mimic sites may be placed elsewhere on a link. Everything is written to
`<retargeted AMASS folder>/<Environment>/shape_optimized.pkl` and reused for every motion.

## Stage 2: from an SMPL motion to a robot trajectory

An SMPL motion is a sequence of joint rotations and root translations. For each frame, the forward kinematics
of the SMPL model are evaluated with the shape fitted in stage 1, the horizontal root motion is scaled by the
height ratio, and the stored offsets are applied. The result is a target position and orientation for each
mimic site.

The targets are not solved with a conventional inverse kinematics routine but **with the simulator**. A
mocap body is added to the model for each site and connected to the site by a soft equality constraint, by
default a weld that constrains position and orientation. For every frame the mocap bodies are moved to their
targets and MuJoCo is stepped a few times (25 by default), which pulls the robot into the pose. The joint
positions after these steps are the retargeted frame, and the next frame starts from them.

Because the pose is found by a physics step, joint limits and collisions of the model are active while the
constraints pull. If a human pose is unreachable, the robot settles in a compromise that respects its limits
instead of in an invalid configuration. For the first frame, joint limits and collisions are switched off and
the simulation is run longer, so that the robot can reach the initial pose from its default configuration
without getting stuck.

Afterwards:

- Joint velocities are computed from the joint positions by central finite differences. The first and the last
  frame are dropped in the process.
- If foot geometries are listed in the configuration, the whole trajectory is shifted vertically so that the
  lowest point of the motion just touches the floor.
- The trajectory is replayed in the environment to compute the body and site quantities, which makes it
  [complete](trajectories.md), and it is saved.

To save time, AMASS motions are subsampled before retargeting (every fourth frame by default).

## Stage 3: from robot to robot

To transfer a trajectory from a source robot to a target robot, the source motion is first converted back
into an SMPL motion and then sent through stage 2 with the shape of the target robot.

The mimic-site positions and orientations of the source trajectory are computed and expressed in SMPL joint
frames with the offsets that stage 1 stored for the source robot. An SMPL pose and root translation for every
frame are then optimized with Adam, with the SMPL shape fixed to that of the *source* robot, to minimize a
weighted sum of the position error and the geodesic rotation error at the 15 correspondences (400 iterations,
weights 1.0 and 0.1 by default). After each iteration the body pose is smoothed over time with a small
Gaussian kernel to suppress jitter.

The result is a synthetic SMPL motion that reproduces what the source robot did. Stage 2 turns it into a
trajectory of the target robot. Stage 1 is run for either robot if its shape file does not exist yet.

## Robot configuration files

Each humanoid has a YAML file in `loco_mujoco/smpl/robot_confs`, named after its environment. It is merged
over `defaults.yaml` in the same folder, so a robot file contains only what differs. The file of the Unitree
H1:

```{literalinclude} ../../loco_mujoco/smpl/robot_confs/UnitreeH1.yaml
:language: yaml
```

| Section | Content |
|---|---|
| `robot_pose_modifier` | Joint offsets, as Python expressions, that take the robot from its default pose to the T-pose used in stage 1. The special entry `root` rotates the base by Euler angles. |
| `site_joint_matches` | Per mimic site: the SMPL joint (`smpl_joint`), the constraint type (`mjEQ_WELD` for position and orientation, `mjEQ_CONNECT` for position only), `torque_scale`, the weight of the orientation relative to the position for a weld, and optionally `solref` and `solimp`, which set the stiffness of the constraint. The defaults map all 15 sites with welds. |
| `optimization_params` | Settings of the three stages, see below. |
| `env_params` | Arguments for the environment that is created for retargeting, for example `disable_arms: false` to make sure the full model is used. |

The most relevant entries of `optimization_params`:

| Entry | Default | Meaning |
|---|---|---|
| `shape_iterations`, `shape_lr` | `1000`, `0.05` | Stage 1 optimization |
| `motion_iterations` | `25` | Simulation steps per frame in stage 2 |
| `init_motion_iterations` | `1000` | Simulation steps for the first frame |
| `skip_frames` | `4` | Subsampling of AMASS motions |
| `z_offset_feet` | `0.0` | Constant vertical offset added to the motion, to correct floating or penetrating feet |
| `height_adjustment_geom_names`, `max_height_adjustment` | `[]`, `1.0` | Foot geometries for the automatic vertical shift and its limit |
| `disable_joint_limits`, `disable_collisions` | `False`, `False` | Switch off limits or collisions during stage 2 |
| `joint_limit_solref`, `joint_limit_solimp` | `null` | Stiffness of the joint limits during retargeting; `null` keeps the model's values |
| `pose_iterations`, `pose_lr` | `400`, `0.005` | Stage 3 optimization |
| `pos_loss_weight`, `rot_loss_weight` | `1.0`, `0.1` | Weights of the stage 3 loss |
| `smoothing_kernel_size`, `smoothing_sigma` | `3`, `0.5` | Temporal smoothing in stage 3 |
| `torch_device` | `"cpu"` | Device of the PyTorch optimizations |

Configuration files for robots outside the repository are found through the variable
`LOCOMUJOCO_CUSTOM_ROBOT_CONF_PATH`, a list of folders:

```bash
loco-mujoco-add-variable --name LOCOMUJOCO_CUSTOM_ROBOT_CONF_PATH --value /path/to/my/robot_confs
```

Writing such a file for a new embodiment, including the placement of the mimic sites, is covered step by step
in [adding a new robot](new_robot.md).

## Running the pipeline

### Automatically, through the task factory

Nothing special is needed. Requesting an AMASS motion, or a default or LAFAN1 clip for an environment without
pre-retargeted data, triggers the required stages and caches the result, see
[datasets](datasets.md#download-processing-and-caching).

```python
from loco_mujoco.task_factories import ImitationFactory, AMASSDatasetConf

env, traj = ImitationFactory.make(
    "UnitreeH1", amass_dataset_conf=AMASSDatasetConf(["KIT/12/WalkInClockwiseCircle11_poses"]))
```

### Retargeting all datasets to an environment

On-the-fly retargeting inside a training run is inconvenient: it delays the start and lets PyTorch and JAX
compete for the GPU. The command `loco-mujoco-retarget-for-new-env`, implemented in
`loco_mujoco/utils/retarget_for_env.py`, retargets every default and LAFAN1 clip of the dataset repository to
one environment ahead of time and writes the results to the cache folders:

```bash
loco-mujoco-set-all-caches --path "$HOME/.loco-mujoco-caches"     # once, if not set yet
loco-mujoco-retarget-for-new-env --env MyRobot --import-module my_package
```

| Option | Meaning |
|---|---|
| `--env` | Name of the registered target environment (required) |
| `--import-module` | Modules to import first, so that environments defined outside LocoMuJoCo are registered |
| `--dataset-types` | `default`, `lafan1` or both (the default) |
| `--overwrite` | Retarget clips again even if a cached file exists |

The source embodiment of each dataset is read from the properties `default_dataset_source_env` and
`lafan1_dataset_source_env` of the target environment. They default to `SkeletonTorque` and `UnitreeH1v2`;
override them in your environment class to choose a source that is closer to your robot.

### From Python

The stages are also available as functions in {mod}`loco_mujoco.smpl.retargeting`:

```python
from loco_mujoco.core.trajectory import Trajectory
from loco_mujoco.smpl.retargeting import (retarget_traj_from_robot_to_robot, extend_motion,
                                          load_robot_conf_file)

traj_h1 = Trajectory.load("my_h1_motion.npz")
traj_g1 = retarget_traj_from_robot_to_robot("UnitreeH1", traj_h1, "UnitreeG1")

# compute the body and site quantities and save the result
traj_g1 = extend_motion("UnitreeG1", load_robot_conf_file("UnitreeG1").env_params, traj_g1)
traj_g1.save("my_g1_motion.npz")
```

{func}`~loco_mujoco.smpl.retargeting.load_retargeted_amass_trajectory` does the same for AMASS motions, and
{func}`~loco_mujoco.smpl.retargeting.fit_smpl_shape` and {func}`~loco_mujoco.smpl.retargeting.fit_smpl_motion`
expose stages 1 and 2. Several of them take `visualize=True` to watch the optimization in the viewer, which
helps when tuning a configuration file.

## Caching

| File | Location | Written by |
|---|---|---|
| Fitted shape of a robot | `<retargeted AMASS folder>/<Environment>/shape_optimized.pkl` | Stage 1 |
| Retargeted AMASS motion | `<retargeted AMASS folder>/<Environment>/<motion path>.npz` | Stage 2 |
| Retargeted default clip | `<default cache folder>/mocap/<Environment>/<task>.npz` | Stage 3, if the cache folder is set |
| Retargeted LAFAN1 clip | `<LAFAN1 cache folder>/<Environment>/<clip>.npz` | Stage 3, if the cache folder is set |

The shape file is stored in the folder for retargeted AMASS motions even when only robot-to-robot retargeting
is used, so that folder must be set in every case. After changing a robot model or its configuration file,
delete the shape file and the retargeted clips of that environment; they are not invalidated automatically.

## Limitations

- **Humanoids only.** The method relies on the 15 correspondences to the human skeleton. There is no
  retargeting for quadrupeds.
- **Kinematic result.** The trajectory respects joint limits and, where enabled, collisions, but it is not
  guaranteed to be dynamically feasible. Whether a robot can actually perform the motion is decided by the
  policy that is trained to track it.
- **Morphology gap.** The further a robot is from human proportions and joint ranges, the more the motion
  changes. Such robots may need a tuned configuration, for example softer or position-only constraints for
  some sites.
- **Ground contact.** Foot contact is not enforced explicitly. Feet may slide slightly or float; `z_offset_feet`
  and the height adjustment correct a constant offset only.
- **Hands and fingers.** Only the wrist is tracked. Finger motion of SMPL-H is ignored.
- **Accumulated error.** Robot-to-robot retargeting fits an SMPL motion to an already retargeted motion and
  retargets again, so some detail of the original recording is lost in each transfer.
- **Velocities** come from finite differences of the positions and can be noisy for fast motions.

Always look at the result with `env.play_trajectory()` before training on it.

## See also

- [Adding a new robot](new_robot.md) for the complete procedure, including mimic sites and the configuration
  file.
- [Datasets](datasets.md) for the data that is retargeted and how it is cached.
- [AMASS and SMPL setup](../getting_started/amass.md) for the prerequisites.
- [Trajectories](trajectories.md) for the format of the result.
- API reference: {mod}`loco_mujoco.smpl.retargeting`.
