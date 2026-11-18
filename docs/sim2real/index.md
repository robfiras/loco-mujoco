# Sim-to-real deployment

```{image} https://raw.githubusercontent.com/robfiras/loco-mujoco-real/main/assets/x2_sim2real.gif
:alt: The AgiBot X2 Ultra running a tracking policy, on the robot and in simulation
:width: 85%
:align: center
```

<p class="lmj-caption">The AgiBot X2 Ultra tracking a motion with a policy trained in LocoMuJoCo, on the robot
and in simulation.</p>

This page describes the path from a policy trained in LocoMuJoCo to the same policy running on a real
humanoid. The reference implementation is
[LocoMuJoCo Real](https://github.com/robfiras/loco-mujoco-real), a companion repository. Both of its
pipelines, for the Unitree G1 and for the AgiBot X2 Ultra, have been run on the robot.

## What you get

LocoMuJoCo Real is a separate package that builds on LocoMuJoCo and registers its components when it is
imported. It contains:

- **Deployable environments** for the Unitree G1 (`G1Real`, `MjxG1Real`) and the AgiBot X2 Ultra
  (`X2UltraReal`, `MjxX2UltraReal`): 29-joint models with position servos, per-motor armature and full
  self-collision, together with the observations, rewards, terminations and randomization they need.
- **Training configurations** for motion tracking on these environments with [PPO](../training/ppo.md) and
  with [SuperMimic](../training/supermimic.md), in its `training` folder.
- **Deployment tools** in its `deployment` folder: the export of a checkpoint to a self-contained bundle, the
  validation of the bundle against the training environment, rollouts in plain MuJoCo, and per robot a
  simulated stand-in that speaks the robot's middleware and the controller that drives the robot.

The same repository is the worked example for [adding a new robot](../user_guide/new_robot.md) from a package
of your own.

```{warning}
Running a learned policy on a real robot is hazardous. A whole-body policy can move a humanoid fast and with
full torque, and it can fail without notice. Keep the robot on a gantry or crane until every stage below has
passed, keep the emergency stop in hand, and keep people out of the robot's reach. Everything described here
is research code that is provided without warranty; you deploy at your own risk.
```

```{note}
The deployment tools are under active development. The stages and the reasons for them are stable; script
names and options may still change. The README files in the `deployment` folder of LocoMuJoCo Real are the
authority on the details.
```

## The stages

The path has six stages. Each one removes a single source of error, so that a failure points at one cause.

| # | Stage | Tool in LocoMuJoCo Real | What it rules out |
|---|---|---|---|
| 1 | [Train](#stage-1-train-in-a-deployable-environment) in a deployable environment | `training/<robot>/train.py`, `train_supermimic.py` | a policy that depends on things the robot does not have |
| 2 | [Export](#stage-2-export-the-policy-to-a-bundle) the policy to a bundle | `deployment/export_bundle.py` | a dependency on the training stack at run time |
| 3 | [Validate](#stage-3-validate-the-bundle) the bundle | `deployment/validate_bundle.py` | a difference between the exported and the trained policy |
| 4 | [Synchronous sim-to-sim](#stage-4-synchronous-sim-to-sim) | `deployment/rollout_mujoco.py`, `rollout_viser.py` | errors in the control loop and the actuation model |
| 5 | [Asynchronous sim-to-sim](#stage-5-asynchronous-sim-to-sim-over-the-robots-middleware) over the robot's middleware | `deployment/<robot>/sim2sim_bridge.py` and the controller | timing, latency, and errors in the communication layer |
| 6 | [The robot](#stage-6-on-the-robot), hanging first | `deployment/<robot>/deploy_<robot>.py` | the remaining gap between the model and the hardware |

Stages 1 to 4 are the same for every robot. Stages 5 and 6 depend on the robot's middleware and have a guide
per robot in LocoMuJoCo Real.

## Setup

Training needs a GPU. Install LocoMuJoCo 1.2.0 or newer with the extras for retargeting and for the browser
viewers, then LocoMuJoCo Real from a checkout of its repository:

```bash
pip install "loco-mujoco[smpl,viser]>=1.2.0"
```

```bash
git clone https://github.com/robfiras/loco-mujoco-real
cd loco-mujoco-real
pip install -e .
```

Tell LocoMuJoCo where the retargeting configurations of the two robots are:

```bash
loco-mujoco-add-variable --name LOCOMUJOCO_CUSTOM_ROBOT_CONF_PATH --value /path/to/loco-mujoco-real/lmj_real/smpl_robot_confs
```

The reference clips are retargeted to the robot the first time they are used, which needs the SMPL-H model
(see [AMASS and SMPL](../getting_started/amass.md)). The AMASS dataset itself is not needed: the source
clips are downloaded from the
[LocoMuJoCo dataset repository](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets).

The commands on this page are run from the root of the LocoMuJoCo Real repository. `<robot>` is `g1` or `x2`.

## Stage 1: Train in a deployable environment

Whether a policy transfers is mostly decided here. Add the robot as described in
[adding a new robot](../user_guide/new_robot.md), then make the environment resemble the robot in the
following points.

### Actuation

Real joint drives close a PD loop on a position target. Model the actuators as MuJoCo *position* actuators
with the stiffness and damping the drives will run, so that the PD loop is part of the physics and runs at the
simulation rate, as it does on the robot. The policy then outputs position targets around a default standing
pose:

```text
target = default_pose + action_scale * action
```

Set the joint `armature` to the reflected rotor inertia of each motor type. Keep the default pose as a
keyframe of the model (`init_state` in LocoMuJoCo Real), since the controller on the robot needs the same
pose.

### Observations

The policy may only read what the robot measures:

| Available on the robot | Not available |
|---|---|
| joint positions and velocities | base position in the world |
| base angular velocity (IMU gyroscope) | base linear velocity |
| projected gravity (from the IMU orientation) | absolute heading (IMU yaw is arbitrary and drifts) |
| the previous action | contact forces, body positions in the world |
| the reference motion | |

Anything in the right column can still go to the critic. Use observation groups (see
[observations](../user_guide/observations.md)) to give the critic privileged observations and the actor only
the left column. For motion tracking this means that the goal observation must be expressed relative to the
robot (LocoMuJoCo Real anchors it to the pelvis) and must not contain a world position.

### Control rate

Choose `timestep` and `n_substeps` so that the policy runs at a rate the robot's computer and network can
hold. LocoMuJoCo Real uses 0.005 s × 4, a 50 Hz policy on a 200 Hz simulation. Use the same two values in the
robot conf, so that the retargeted clips have that frequency.

### Randomization

Randomize what differs between the model and the robot rather than everything: pushes, encoder offsets, foot
friction, the centre of mass of the torso, observation noise and, if the latency of the robot is known, an
observation delay. See [domain randomization](../user_guide/domain_randomization.md).

### Reference motions

A clip that is played on the robot should start and end in a standing pose. The controller then never has to
hold, or switch clips in, a pose in mid-stride.

### Run the training

Training itself is unchanged. LocoMuJoCo Real has a configuration per robot for PPO
(`training/<robot>/conf.yaml`) and for SuperMimic (`training/<robot>/supermimic.yaml`), both on the same
deployable environment:

`````{tab-set}
````{tab-item} PPO
```bash
python training/<robot>/train.py
```
````
````{tab-item} SuperMimic
```bash
python training/<robot>/train_supermimic.py
```
````
`````

Any setting of the configuration can be overridden on the command line, for example to train on fewer clips:

```bash
python training/<robot>/train.py 'experiment.task_factory.params.default_dataset_conf.task=[walk,run]'
```

Results are written to `outputs/`. Every later stage takes the checkpoint of either algorithm,
`PPOJax_saved.pkl` or `SuperMimicJax_saved.pkl`. A PPO policy can be watched in the browser before it is
exported:

```bash
python training/eval_viser.py --path outputs/<run>/PPOJax_saved.pkl --visualize-goal
```

## Stage 2: Export the policy to a bundle

The robot cannot call back into the training environment, and its computer should not need JAX. The
checkpoint is therefore exported to a *bundle*, a folder that is run with `numpy` and `mujoco` only (MuJoCo
is used for forward kinematics):

```text
deploy_bundles/<name>/
├── policy.json       joint and actuator order, action scale and offset, kp/kd,
│                     observation layout, tracked bodies, network shape, solver options
├── policy.npz        network weights and the observation normalizer
└── motions.lmj.npz   the reference clips, as a LocoMuJoCo Trajectory
```

```bash
python deployment/export_bundle.py --checkpoint outputs/<run>/PPOJax_saved.pkl --output deploy_bundles/<name>
```

For a dataset with many clips, limit the number of clips in the bundle with `--max-motions`.

The runtime library (`deployment/lmj_deploy.py`) rebuilds the actor's observation by hand from a `MjData`,
applies the frozen normalizer and the network, and maps the action to joint targets. The exporter refuses a
checkpoint whose control function or observation terms it does not implement, rather than exporting something
slightly wrong.

```{important}
Two orderings exist side by side, and mixing them fails without an error. Observations and the reference are
in MuJoCo *joint* order; actions, gains and `ctrl` are in *actuator* order. Both lists are stored in
`policy.json`, and the runtime resolves every joint and body by name. Do the same in a controller of your
own.
```

The reference clips are stored in LocoMuJoCo's own [trajectory format](../user_guide/trajectories.md), so a
bundle's motions can be loaded and replayed with the usual tools.

## Stage 3: Validate the bundle

A hand-written observation is only trustworthy if it is checked. `validate_bundle.py` rebuilds the training
environment from the checkpoint, feeds both sides the same states, and compares every observation term, the
action and the joint targets:

```bash
python deployment/validate_bundle.py --bundle deploy_bundles/<name> --checkpoint outputs/<run>/PPOJax_saved.pkl
```

It must print `PASS`. The differences are at the level of floating-point error (`1e-7` and below). Run it
again whenever the observation, the control function or the exporter changes. Do not look at any rollout of a
bundle that has not passed.

Stages 2 and 3 need the training setup, since they rebuild the environment from the checkpoint. Everything
after them does not.

## Stage 4: Synchronous sim-to-sim

The bundle now runs in plain MuJoCo, without LocoMuJoCo, with policy and physics stepped in lockstep:

```bash
python deployment/rollout_mujoco.py --bundle deploy_bundles/<name> --viewer
```

`rollout_viser.py` does the same in the browser, with a clip picker:

```bash
python deployment/rollout_viser.py --bundle deploy_bundles/<name>
```

The `--plant` option of both scripts selects the simulated robot:

| `--plant` | Purpose |
|---|---|
| `servo` (default) | The training model. A failure here is a bug in the pipeline, not a sim-to-real gap. |
| `torque` | Torque motors with the PD loop computed in Python, which is the actuation path of the deployment. It should reproduce `servo`. |
| a path to an MJCF | Another model of the same robot, for example the vendor's, with its own joint friction, damping and collision geometry. A first robustness test. |

## Stage 5: Asynchronous sim-to-sim over the robot's middleware

On the robot, the controller and the plant are separate processes: the state the policy sees is already old,
and a late command leaves the previous targets in force. This stage reproduces that. A *bridge* runs MuJoCo in
a process of its own and speaks the robot's protocol; the *controller* is the program that will later drive
the robot, unchanged.

| Robot | Middleware | Bridge | Controller |
|---|---|---|---|
| Unitree G1 | Unitree SDK 2 (DDS), `LowState` / `LowCmd` | `deployment/g1/sim2sim_bridge.py` | `deployment/g1/deploy_g1.py` |
| AgiBot X2 Ultra | ROS 2 Humble, AimDK joint and IMU topics | `deployment/x2/sim2sim_bridge.py` | `deployment/x2/deploy_x2.py` |

In two terminals:

`````{tab-set}
````{tab-item} Unitree G1
```bash
python deployment/g1/sim2sim_bridge.py --bundle deploy_bundles/<name> --viewer
```

```bash
python deployment/g1/deploy_g1.py --bundle deploy_bundles/<name> --motion <clip>
```

Both scripts default to the network interface `lo0`; pass `--interface lo` on Linux. They need the Unitree
SDK for Python, see `deployment/g1/README.md`.
````
````{tab-item} AgiBot X2 Ultra
```bash
python deployment/x2/sim2sim_bridge.py --bundle deploy_bundles/<name> --viewer
```

```bash
python deployment/x2/deploy_x2.py --bundle deploy_bundles/<name> --motion <clip> --viser
```

Both scripts need ROS 2 Humble and the `aimdk_msgs` package, see `deployment/x2/README.md`.
````
`````

The simulated robot hangs from an elastic band, like the real one from its crane, and the start goes as it
will on the robot: the controller ramps to the first pose of the clip and holds it, the robot is lowered onto
its feet, the policy is started, and the band is released. The viewer draws the reference as a ghost.

Both sides use a DDS or ROS domain other than the robot's, so that the rehearsal cannot reach a robot on the
same network.

### How the controller is built

How the controller is built matters as much as the policy:

- **One observation path.** The incoming robot state is written into a `MjData`, and the same observation
  builder as in stages 3 and 4 runs on it. There is no second implementation for the robot.
- **Poll the newest state** instead of handling every message in a callback. At 500 Hz, a Python callback
  takes the interpreter away from the control loop.
- **Heading is relative.** When the policy starts, rotate the reference onto the robot's current heading. IMU
  yaw then only has to hold for the length of a clip.
- **Limit the BLAS threads** before `numpy` is imported (for example `OPENBLAS_NUM_THREADS=4`). An unlimited
  thread pool can delay single policy steps by tens of milliseconds as soon as anything else needs a core.
- **Watch the timing.** Print the control rate, the compute time per step and the age of the state while the
  policy runs.

## Stage 6: On the robot

### Safety checks in the controller

The controller commands nothing if the first state it receives looks wrong, and afterwards damps the joints
(zero stiffness, some damping) on any of the following:

| Trigger | Limit used in LocoMuJoCo Real |
|---|---|
| stop button on the remote or keyboard | |
| no new state from the robot | 100 ms |
| a joint beyond its range | 0.1 rad past the model's range |
| joint speed | 25 rad/s |
| tilt of the pelvis | about 60 degrees |
| invalid IMU reading, non-finite observation or action | |
| `Ctrl-C` or an exception | |

Damping lets the robot fall, which is why it hangs. A softer second level is useful as well: the policy keeps
running but tracks the standing pose instead of the clip.

### The procedure

In this order, the first time with every new policy or motion:

1. The robot **hangs**, feet clear of the floor or just touching. Emergency stop in hand.
2. **Read-only checks**: the status fields of the robot are clean, projected gravity points down, and a live
   view of the measured state moves like the robot when a joint is moved by hand. This catches a wrong joint
   mapping or sign before anything is commanded.
3. **Prove the link**: swing one or two joints by a small amplitude with soft gains.
4. **Ramp and hold, without the policy**: ramp slowly (0.5 rad/s at most) to the first pose of the clip and
   hold it with the policy's gains.
5. **The policy, hanging.**
6. **The policy on the floor**: start it balancing on a standing pose, slacken the crane, then play the
   motion.

### Robot-specific guides

Before the controller can command the joints, the vendor's own controller has to release them. How that is
done, the network setup and the required software differ per robot:

- **Unitree G1**:
  [`deployment/g1/README.md`](https://github.com/robfiras/loco-mujoco-real/blob/main/deployment/g1/README.md)
  covers the Unitree SDK setup and the controller. The controller drives the robot from a laptop on its
  Ethernet port.
- **AgiBot X2 Ultra**:
  [`deployment/x2/X2_ROBOT_GUIDE.md`](https://github.com/robfiras/loco-mujoco-real/blob/main/deployment/x2/X2_ROBOT_GUIDE.md)
  covers the network, ROS 2 and `aimdk_msgs`, the switch to the developer mode, the topics, the gains and
  measurements taken on the robot.
  [`deployment/x2/README.md`](https://github.com/robfiras/loco-mujoco-real/blob/main/deployment/x2/README.md)
  describes the tools, including how to run the controller on the robot's own computer.

## Deploying on another robot

Stages 1 to 4 do not depend on the robot's middleware. A new robot needs:

1. an environment and a robot conf ([adding a new robot](../user_guide/new_robot.md)), with position
   actuators, armature and a default-pose keyframe,
2. a small transport module that reads the robot's state into a `MjData` and writes joint targets with gains
   (compare `deployment/g1/lmj_dds.py` and `deployment/x2/lmj_aimdk.py`),
3. a bridge that makes MuJoCo speak the same protocol, and
4. a controller with the start-up sequence, the safety checks and the operator input.

The observation builder, the actor and the bundle format are shared.

## When it works in simulation but not on the robot

| Symptom | Where to look |
|---|---|
| The robot shakes or hums while standing | Gains or armature differ from the drives; control latency. Compare a small sine on a single joint between the robot and the model. |
| The policy fails in the first step | Joint order or sign, a normalizer input that was constant in training, a stale first state. |
| The robot drifts or turns away from the reference | Heading alignment at the start; the policy reads a world-frame quantity. |
| Joints lag behind the model | Friction in the real joints that the model lacks. Test with a plant that has `frictionloss` in stage 4. |
| Single late control steps | BLAS threads, rendering in the control process, message callbacks. |
| A fall when a clip ends or is switched | The clip does not start or end standing. |

## Current limitations

These are limitations of the deployment tools at the time of writing; `deployment/REFERENCE.md` in
LocoMuJoCo Real has the current list.

- The exporter supports the position control function of LocoMuJoCo Real and the observation terms that
  `lmj_deploy.py` implements. Of SuperMimic's actors, it supports the encoder-decoder one.
- Observation delay and observation history exist in the environments but are off by default, and a bundle
  does not carry them. A policy trained with them cannot be exported faithfully yet.
- Only the mean action is exported, so the deployed policy is deterministic.
- Reference clips can ask for joint positions beyond the joint range. Look at a clip before playing it on the
  robot.
