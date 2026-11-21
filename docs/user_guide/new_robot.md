# Adding a new robot

This guide shows how to add your own humanoid to LocoMuJoCo without changing the LocoMuJoCo source, and how to
connect it to the [retargeting pipeline](retargeting.md) so that every published motion clip (the default
dataset, LAFAN1 and AMASS) becomes available for it.

The robot lives in a small package of its own that registers its environment with LocoMuJoCo when it is
imported. The robots of [LocoMuJoCo Real](https://github.com/robfiras/loco-mujoco-real) (`G1Real` and
`X2UltraReal`) are added this way, and the code on this page follows that package.

## What you will create

| | What | Where | Needed for |
|---|---|---|---|
| 1 | A MuJoCo model (MJCF) with a free joint named `root`, a floor and a set of *mimic sites* | `models/my_robot/my_robot.xml` | everything |
| 2 | An environment class deriving from {class}`~loco_mujoco.environments.humanoids.base_robot_humanoid.BaseRobotHumanoid`, and its MJX variant | `environments/my_robot.py`, `environments/my_robot_mjx.py` | simulation on the CPU, and on the GPU |
| 3 | A call to `register()` for both classes | `__init__.py` | creating the robot by name |
| 4 | A *robot conf*: a YAML file that tells the retargeting pipeline how the robot maps to the SMPL human body | `smpl_robot_confs/MyRobot.yaml` | motion capture data |
| 5 | Retargeted datasets, produced on first use or ahead of time | the dataset caches | imitation learning |
| 6 | A check of the result | | trusting the above |

Steps 1 to 3 are enough for reinforcement learning tasks. Steps 4 to 6 add the motion capture data.

The package layout used on this page:

```text
my_robots/
├── pyproject.toml
└── my_robots/
    ├── __init__.py               # registers the environments on import
    ├── environments/
    │   ├── my_robot.py           # MyRobot      (MuJoCo, CPU)
    │   └── my_robot_mjx.py       # MjxMyRobot   (MJX / MuJoCo Warp, GPU)
    ├── models/
    │   ├── common/scene.xml      # floor, lights
    │   └── my_robot/
    │       ├── my_robot.xml
    │       └── assets/*.stl
    └── smpl_robot_confs/
        └── MyRobot.yaml          # the robot conf; file name = class name
```

Ship the model and the robot conf with the package:

```toml
[tool.setuptools.package-data]
"my_robots" = ["models/**/*", "smpl_robot_confs/*.yaml"]
```

```{note}
The models of the built-in robots are not part of the Python package. They live in the repository
[loco-mujoco-models](https://github.com/robfiras/loco-mujoco-models) and are downloaded on first use. A robot
in a package of your own does not use that mechanism: its model is shipped with your package and found
relative to the environment file, as shown in step 2. See
[Contributing the robot to LocoMuJoCo](#contributing-the-robot-to-locomujoco) if you want it to become a
built-in robot instead.
```

## Step 1: Prepare the MuJoCo model

Start from the vendor's MJCF, or convert the URDF with MuJoCo's `compile` tool. LocoMuJoCo needs the following
from it.

### A free joint named `root`

Trajectories are matched to the model by joint name, so a free joint with another name silently loses its seven
`qpos` entries.

```xml
<body name="pelvis" pos="0 0 0.79">
  <joint name="root" type="free"/>
  ...
```

### A floor

The model must contain the ground: a plane geom named `floor`. The MJX variant below builds its contact pairs
with that name, and so do the terrains and the rewards that look at contacts with the ground. The built-in
robots include a shared scene file with the floor, the lights and the sky box; LocoMuJoCo Real ships a copy of
it in `lmj_real/models/common/scene.xml`. Do the same and include it from your model:

```xml
<mujoco model="my_robot">
  <include file="../common/scene.xml"/>
  ...
```

### Mimic sites

A mimic site marks the point of the robot that corresponds to a joint of the human body. The sites are the
only link between your robot and the motion data: retargeting pulls each site towards its human joint, and
tracking rewards and observations compare site poses between the robot and the reference. A humanoid has 15 of
them:

| Site | SMPL joint | Put it on |
|---|---|---|
| `pelvis_mimic` | `Pelvis` | the root body |
| `upper_body_mimic` | `Chest` | the torso, at chest height |
| `head_mimic` | `Head` | the head, or the torso at head height if there is no head link |
| `{left,right}_hip_mimic` | `L_Hip`, `R_Hip` | the first thigh link, at the hip joint |
| `{left,right}_knee_mimic` | `L_Knee`, `R_Knee` | the shank, at the knee joint |
| `{left,right}_foot_mimic` | `L_Ankle`, `R_Ankle` | the last foot link, near the ankle |
| `{left,right}_shoulder_mimic` | `L_Shoulder`, `R_Shoulder` | the upper arm, at the shoulder joint |
| `{left,right}_elbow_mimic` | `L_Elbow`, `R_Elbow` | the forearm, at the elbow joint |
| `{left,right}_hand_mimic` | `L_Wrist`, `R_Wrist` | the last arm link, at the wrist or palm |

A site is one line inside the body it belongs to. Give the mimic sites a group of their own so that they can be
hidden in the viewer:

```xml
<body name="left_knee_link" pos="0 0 -0.3">
  <joint name="left_knee_joint" axis="0 1 0"/>
  <site name="left_knee_mimic" pos="0 0 0" size="0.075 0.05 0.025"
        type="box" group="4" rgba="1 0 0 0.5"/>
```

The site names above are the defaults of
{meth}`~loco_mujoco.environments.humanoids.base_robot_humanoid.BaseRobotHumanoid.sites_for_mimic` and of the
robot conf, so keeping them saves configuration. If your robot lacks a body part, override `sites_for_mimic`
to return fewer sites.

### Actuators

Every actuator in the model becomes one action dimension by default. For a policy that is meant for a real
robot, use position actuators with the gains of the real joint drives, see
[sim-to-real deployment](../sim2real/training.md#actuation).

```{note}
If the model defines a default standing pose, store it as a `<keyframe>` and not as joint `ref` values. `ref`
shifts the meaning of `qpos` and with it the forward kinematics of every trajectory.
```

## Step 2: Write the environment class

The class loads the model, declares the default observation and action space, and answers a few questions
about the robot through *info properties*.

```python
# my_robots/environments/my_robot.py
import os
from typing import List, Tuple, Union

import mujoco
from mujoco import MjSpec

from loco_mujoco.core import Observation, ObservationType
from loco_mujoco.core.utils import info_property
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid


class MyRobot(BaseRobotHumanoid):

    mjx_enabled = False

    def __init__(self, spec: Union[str, MjSpec] = None,
                 observation_spec: List[Observation] = None,
                 actuation_spec: List[str] = None, **kwargs) -> None:

        if spec is None:
            spec = self.get_default_xml_file_path()
        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        if observation_spec is None:
            observation_spec = self._get_observation_specification(spec)
        else:
            observation_spec = self.parse_observation_spec(observation_spec)
        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)

        super().__init__(spec=spec, actuation_spec=actuation_spec,
                         observation_spec=observation_spec, **kwargs)

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[Observation]:
        """Default observation: root pose and velocity plus every joint."""
        joints = [j.name for j in spec.joints if j.name != "root"]
        return [
            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
            *(ObservationType.JointPos(f"q_{j}", xml_name=j) for j in joints),
            ObservationType.FreeJointVel("dq_root", xml_name="root"),
            *(ObservationType.JointVel(f"dq_{j}", xml_name=j) for j in joints),
        ]

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        """Default action space: every actuator in the model."""
        return [actuator.name for actuator in spec.actuators]

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        # resolve relative to this file, so the package works wherever it is installed
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(package_root, "models", "my_robot", "my_robot.xml")

    @info_property
    def root_body_name(self) -> str:
        return "pelvis"

    @info_property
    def upper_body_xml_name(self) -> str:
        return "torso_link"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        return (0.5, 1.0)

    @info_property
    def foot_geom_names(self) -> List[str]:
        return ["left_foot_collision", "right_foot_collision"]
```

The info properties LocoMuJoCo reads:

| Property | Meaning |
|---|---|
| `root_body_name` | Name of the floating-base body. `BaseRobotHumanoid` returns `"pelvis"`. |
| `root_free_joint_xml_name` | Name of the free joint. Defaults to `"root"`; leave it. |
| `upper_body_xml_name` | Name of the torso body. Required. |
| `root_height_healthy_range` | Root heights between which the robot counts as standing. Used by {class}`~loco_mujoco.core.terminal_state_handler.height.HeightBasedTerminalStateHandler`. Required. |
| `foot_geom_names` | The geoms that touch the ground. Read by rewards that look at foot contact, and a natural place for your own components to find the feet. |
| `sites_for_mimic` | The mimic sites. Defaults to the 15 names in the table above. |
| `default_dataset_source_env`, `lafan1_dataset_source_env` | The robot to retarget from, see [Choose the source robot](#choose-the-source-robot). |

Components of your own (rewards, observations, control functions, terminal state handlers, domain randomizers)
are added the same way, see [custom modules](custom_modules.md). They can declare further info properties on
the environment. `G1Real`, for example, adds `tracked_body_names` and `anchor_body_name` for its tracking
reward, which lets one reward class serve several robots.

### The MJX variant

Training on the GPU needs a second class with `mjx_enabled = True`. By convention its name is the CPU class
name with the prefix `Mjx`. It sets the solver options and simplifies the collision model, because MJX is slow
with many contacts.

```python
# my_robots/environments/my_robot_mjx.py
import mujoco
from mujoco import MjSpec

from my_robots.environments.my_robot import MyRobot


class MjxMyRobot(MyRobot):

    mjx_enabled = True

    def __init__(self, timestep=0.002, n_substeps=5, **kwargs):
        model_option_conf = kwargs.pop("model_option_conf", None) or dict(
            iterations=2, ls_iterations=4,
            disableflags=mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
        super().__init__(timestep=timestep, n_substeps=n_substeps,
                         model_option_conf=model_option_conf, **kwargs)

    def _modify_spec_for_mjx(self, spec: MjSpec) -> MjSpec:
        """Keep only the contacts between the feet and the floor."""
        for g in spec.geoms:
            g.contype = 0
            g.conaffinity = 0
        for name in self.foot_geom_names:
            spec.add_pair(geomname1="floor", geomname2=name)
        return spec
```

With the MuJoCo Warp backend (`use_mjwarp=True`, see [backends](backends.md)) full self-collision is
affordable. In that case `_modify_spec_for_mjx` can return the spec unchanged, as `MjxG1Real` does.

## Step 3: Register the environment

Environments are looked up by class name. Register both classes when your package is imported:

```python
# my_robots/__init__.py
def register_environments():
    from my_robots.environments.my_robot import MyRobot
    from my_robots.environments.my_robot_mjx import MjxMyRobot

    MyRobot.register()
    MjxMyRobot.register()


register_environments()
```

After `import my_robots` the robot behaves like a built-in one:

```python
import my_robots  # noqa: F401  (registers MyRobot and MjxMyRobot)
from loco_mujoco import RLFactory

env, _ = RLFactory.make("MyRobot")
env.reset()
env.render()
```

Check this before going on. A model that loads and renders here rules out most XML problems before
retargeting starts.

## Step 4: Write the robot conf

The robot conf is a YAML file that configures retargeting for one robot. The built-in ones are in
`loco_mujoco/smpl/robot_confs/`.

### Naming and lookup

The file is found by class name: for `MyRobot` and `MjxMyRobot` LocoMuJoCo looks for `MyRobot.yaml`, first
among the built-in confs and then in the directories listed in the variable
`LOCOMUJOCO_CUSTOM_ROBOT_CONF_PATH`. Register your directory once:

```bash
loco-mujoco-add-variable --name LOCOMUJOCO_CUSTOM_ROBOT_CONF_PATH --value /path/to/my_robots/my_robots/smpl_robot_confs
```

The variable is a list, so several packages can each add their directory. `loco-mujoco-show-variable --all`
prints what is set.

### Content

Your file is merged over `defaults.yaml`, so it only holds what differs. A typical conf:

```yaml
# my_robots/smpl_robot_confs/MyRobot.yaml

# joint offsets that bring the robot from qpos = 0 into a T-pose
robot_pose_modifier:
  - left_shoulder_roll_joint: "np.pi/2"
  - left_elbow_joint: "np.pi/2"
  - right_shoulder_roll_joint: "-np.pi/2"
  - right_elbow_joint: "np.pi/2"

# per-site overrides; sites not listed here keep the defaults
site_joint_matches:
  left_hand_mimic:
    smpl_joint: "L_Wrist"
    equality_constraint_type: mjEQ_WELD
    torque_scale: 10.0
  right_hand_mimic:
    smpl_joint: "R_Wrist"
    equality_constraint_type: mjEQ_WELD
    torque_scale: 10.0

optimization_params:
  z_offset_feet: 0.0

# must match the environment you train in
env_params:
  timestep: 0.005
  n_substeps: 4
```

`robot_pose_modifier`
: The SMPL body is fitted to your robot in a T-pose: standing upright, arms stretched out horizontally, palms
  down. List here the joint values that take the robot from its zero configuration into that pose. Values are
  Python expressions and may use `np`. This is the setting that decides the retargeting quality, so look at the
  pose in the viewer before retargeting anything (see [Step 6](#step-6-check-the-result)).

`site_joint_matches`
: For each mimic site, the SMPL joint it follows (`smpl_joint`), the constraint that pulls it there (`mjEQ_WELD`
  for position and orientation, `mjEQ_CONNECT` for position only) and `torque_scale`, the weight of orientation
  against position in a weld. `defaults.yaml` covers the 15 default sites with welds at `torque_scale: 1.0`, so
  an entry is needed only for a site you want to treat differently, or for a site with a name of your own.
  Raising `torque_scale` on hands, knees and feet, as above, makes their orientation follow the human more
  closely. `solref` and `solimp` can be set per site as well.

`optimization_params`
: Settings of the optimization. The ones that are changed most often:

  | Key | Meaning |
  |---|---|
  | `z_offset_feet` | Raises (or lowers) the human targets, in metres. Use it when the retargeted feet sink into the floor or float above it. |
  | `height_adjustment_geom_names` | Foot geoms whose distance to the floor is measured, to shift the whole clip so the lowest foot point touches the ground. |
  | `disable_joint_limits`, `disable_collisions` | Switch off joint limits or collisions in the retargeting model. |
  | `shape_iterations`, `shape_lr` | Iterations and learning rate of the shape fit. |
  | `motion_iterations` | Physics steps per retargeted frame. |

`env_params`
: Keyword arguments of the environment that is built for retargeting. `defaults.yaml` sets `n_substeps: 25`.
  Set `timestep` and `n_substeps` to the values of the environment you train in, so that the cached clips have
  your control frequency.

## Step 5: Retarget the datasets

Retargeting needs the SMPL setup described in [AMASS and SMPL](../getting_started/amass.md): an editable
install with the `smpl` extra, the SMPL-H model, and the cache paths.

```bash
loco-mujoco-set-smpl-model-path --path /path/to/smpl
```

```bash
loco-mujoco-set-all-caches --path "$HOME/.loco-mujoco-caches"
```

The AMASS and LAFAN1 raw data are not needed for the default and LAFAN1 clips. Those are downloaded for an
existing robot from the
[LocoMuJoCo dataset repository on Hugging Face](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main)
and transferred to yours.

### How it works

1. **Shape fit, once per robot.** The robot is put into the T-pose, and the shape and scale of the SMPL body
   are optimized until its joints coincide with the mimic sites. The result is cached as
   `<Env>/shape_optimized.pkl` in the converted-AMASS cache.
2. **Motion fit, per clip.** One mocap body per mimic site is added to the model and constrained to the site.
   The mocap bodies replay the human motion while the physics pulls the robot along, with joint limits and
   collisions active.
3. **Robot-to-robot retargeting.** For clips that only exist for another robot, that robot's motion is first
   fitted back to its own SMPL body and then goes through step 2.

[Retargeting](retargeting.md) describes the pipeline in more detail.

### On the fly

Nothing has to be called by hand. When
{class}`~loco_mujoco.task_factories.imitation_factory.ImitationFactory` does not find a clip for your robot in
the cache or on the hub, it retargets the clip and caches it:

```python
import my_robots  # noqa: F401
from loco_mujoco.task_factories import (ImitationFactory, DefaultDatasetConf,
                                        LAFAN1DatasetConf)

# first call: retargets and caches. Later calls load from the cache.
env, traj = ImitationFactory.make(
    "MyRobot",
    default_dataset_conf=DefaultDatasetConf(["walk", "squat"]),
    lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4"]),
)

env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)
```

AMASS clips ({class}`~loco_mujoco.task_factories.dataset_confs.AMASSDatasetConf`) work the same way, but are
retargeted directly from the human data and therefore need the AMASS dataset on disk.

### Choose the source robot

The robot that default and LAFAN1 clips are transferred from is read from two info properties of your
environment:

- {meth}`~loco_mujoco.environments.humanoids.base_robot_humanoid.BaseRobotHumanoid.default_dataset_source_env`,
  default `"SkeletonTorque"`
- {meth}`~loco_mujoco.environments.humanoids.base_robot_humanoid.BaseRobotHumanoid.lafan1_dataset_source_env`,
  default `"UnitreeH1v2"`

A source with similar proportions and joint ranges gives better motions. `G1Real` and `X2UltraReal` use the
built-in G1 for both:

```python
class MyRobot(BaseRobotHumanoid):

    @info_property
    def default_dataset_source_env(self) -> str:
        return "UnitreeG1"

    @info_property
    def lafan1_dataset_source_env(self) -> str:
        return "UnitreeG1"
```

### Everything at once

Retargeting a clip takes a few minutes and competes with JAX for GPU memory. Before a training run on many
clips, retarget them all ahead of time:

```bash
loco-mujoco-retarget-for-new-env --env MyRobot --import-module my_robots
```

`--import-module` names the package or packages that register your environment. Clips that are already cached
are skipped. Useful flags:

- `--dataset-types default lafan1` selects the sources (default: both).
- `--overwrite` retargets clips again that are already in the cache.

The command sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` so that JAX does not take 75% of the GPU memory from the
PyTorch-based SMPL fitting.

Where the results are cached:

| Data | Location |
|---|---|
| Default clips | `LOCOMUJOCO_CONVERTED_DEFAULT_PATH/mocap/MyRobot/<task>.npz` |
| LAFAN1 clips | `LOCOMUJOCO_CONVERTED_LAFAN1_PATH/MyRobot/<clip>.npz` |
| AMASS clips and the shape fit | `LOCOMUJOCO_CONVERTED_AMASS_PATH/MyRobot/` |

```{warning}
The cache does not notice changes. After editing the mimic sites, the robot conf or the model, delete
`MyRobot/shape_optimized.pkl` and the cached clips of your robot (or pass `--overwrite`), otherwise the old
results are loaded again.
```

## Step 6: Check the result

Replay a clip (`env.play_trajectory`, as above) and look at it before training on it. What usually goes wrong:

| Symptom | Likely cause |
|---|---|
| Arms or legs twisted in every clip | The T-pose in `robot_pose_modifier` is wrong. Set the offsets on a fresh environment and render it: arms horizontal, legs straight, facing `+x`. |
| The root does not follow the clip | The free joint is not named `root`. |
| Feet below or above the floor | Adjust `z_offset_feet` or set `height_adjustment_geom_names`. |
| Wrists or ankles far beyond their joint range | The weld on that site is too strong for what the joint can do. Lower its `torque_scale` or use `mjEQ_CONNECT`. |
| Hands or feet point the wrong way | Raise `torque_scale` for that site, or move the site to the last link of the limb. |
| `YAML file 'MyRobot.yaml' not found` | `LOCOMUJOCO_CUSTOM_ROBOT_CONF_PATH` is not set, or the file name is not the class name. |
| `KeyError` for your environment name | The package that registers it was not imported (`--import-module` on the command line). |
| Changes have no effect | Stale cache, see the warning above. |

Retargeted references are not guaranteed to be feasible: single frames can exceed a joint limit or
self-collide. Rewards and terminations that punish this should leave some margin.

## Contributing the robot to LocoMuJoCo

A robot that should become part of LocoMuJoCo follows the same steps, with the files in different places:

- **The model** goes into the repository
  [loco-mujoco-models](https://github.com/robfiras/loco-mujoco-models), in a folder of its own under
  `loco_mujoco_models/`, with the license of its source next to it. The built-in models include the shared
  files `../common/scene.xml` (floor and lights) and `../common/mimic_sites.xml` (a default class `mimic` for
  the sites).
- **The environment classes** go into `loco_mujoco/environments/humanoids/` and are imported and registered in
  the `__init__.py` of that package. `get_default_xml_file_path` returns
  `cls.get_model_path("<folder>", "<file>.xml")`, which resolves the path in the downloaded models.
- **The robot conf** goes into `loco_mujoco/smpl/robot_confs/<ClassName>.yaml`.
- **The pinned revision** of the models, `MODELS_REVISION` in `loco_mujoco/core/utils/models.py`, has to point
  to a commit of loco-mujoco-models that contains the new model.
- **The environment gallery** gets a page for the robot once it has an entry in
  `docs/_scripts/generate_environments.py`, see [writing documentation](../development/documentation.md).

To work on a model before it is published, point LocoMuJoCo at a local checkout of the models. The variable
takes precedence over an installed `loco_mujoco_models` package and over the download cache:

```bash
export LOCOMUJOCO_ROBOT_MODELS_PATH=/path/to/loco-mujoco-models/loco_mujoco_models
```

See [contributing](../development/contributing.md) for the pull request process.

## Checklist

- The model has a free joint named `root` and a plane geom named `floor`.
- The 15 mimic sites exist with their default names, or `sites_for_mimic` is overridden.
- A default pose, if any, is stored as a keyframe and not in joint `ref` values.
- `MyRobot` implements `get_default_xml_file_path`, `_get_observation_specification`,
      `_get_action_specification`, `upper_body_xml_name` and `root_height_healthy_range`.
- `MjxMyRobot` sets `mjx_enabled = True` and reduces the contacts in `_modify_spec_for_mjx`.
- Both classes are registered when the package is imported, and `RLFactory.make("MyRobot")` renders.
- `MyRobot.yaml` is named after the class, and its directory is in `LOCOMUJOCO_CUSTOM_ROBOT_CONF_PATH`.
- `robot_pose_modifier` puts the robot into a T-pose, checked in the viewer.
- `env_params` in the robot conf match the time step and substeps of the training environment.
- The source robots for the default and LAFAN1 clips are chosen.
- A replayed clip looks right, and the caches were cleared after the last change to the model or the conf.
