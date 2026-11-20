# Sim-to-real deployment

<div class="lmj-media lmj-media-narrow">
  <video class="lmj-video" autoplay loop muted playsinline poster="../_static/media/x2_sim2real_poster.jpg" aria-label="The AgiBot X2 Ultra running a tracking policy, on the robot and in simulation">
    <source src="../_static/media/x2_sim2real.mp4" type="video/mp4">
  </video>
</div>

<p class="lmj-caption">The AgiBot X2 Ultra tracking a motion with a policy trained in LocoMuJoCo, on the robot
and in simulation.</p>

This guide describes the path from a policy trained in LocoMuJoCo to the same policy running on a real
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
full torque, and it can fail without notice. Keep the robot on a gantry or crane until every stage of this
guide has passed, keep the emergency stop in hand, and keep people out of the robot's reach. Everything
described here is research code that is provided without warranty; you deploy at your own risk.
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
| 1 | [Train](training.md) in a deployable environment | `training/<robot>/train.py`, `train_supermimic.py` | a policy that depends on things the robot does not have |
| 2 | [Export](export.md#stage-2-export-the-policy-to-a-bundle) the policy to a bundle | `deployment/export_bundle.py` | a dependency on the training stack at run time |
| 3 | [Validate](export.md#stage-3-validate-the-bundle) the bundle | `deployment/validate_bundle.py` | a difference between the exported and the trained policy |
| 4 | [Synchronous sim-to-sim](sim2sim.md#stage-4-synchronous-sim-to-sim) | `deployment/rollout_mujoco.py`, `rollout_viser.py` | errors in the control loop and the actuation model |
| 5 | [Asynchronous sim-to-sim](sim2sim.md#stage-5-asynchronous-sim-to-sim-over-the-robots-middleware) over the robot's middleware | `deployment/<robot>/sim2sim_bridge.py` and the controller | timing, latency, and errors in the communication layer |
| 6 | [The robot](robot.md), hanging first | `deployment/<robot>/deploy_<robot>.py` | the remaining gap between the model and the hardware |

Stages 1 to 4 are the same for every robot. Stages 5 and 6 depend on the robot's middleware and have a guide
per robot in LocoMuJoCo Real.

Start with the [setup](setup.md). If a policy works in simulation but not on the robot, see
[troubleshooting and limitations](troubleshooting.md).

```{toctree}
:hidden:
:maxdepth: 2

setup
training
export
sim2sim
robot
troubleshooting
```
