# On the robot

<ul class="lmj-steps">
  <li><a href="setup.html">Setup</a></li>
  <li><a href="training.html">1 · Train</a></li>
  <li><a href="export.html">2–3 · Export and validate</a></li>
  <li><a href="sim2sim.html">4–5 · Sim-to-sim</a></li>
  <li class="current">6 · On the robot</li>
</ul>

The last stage runs the controller of [stage 5](sim2sim.md#stage-5-asynchronous-sim-to-sim-over-the-robots-middleware)
on the real robot, hanging first.

## Safety checks in the controller

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

## The procedure

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

## A run on the X2 Ultra

```{figure} ../_static/figures/x2_sim_vs_real.jpg
:alt: Six moments of a dance routine, each shown in simulation and on the real AgiBot X2 Ultra
:width: 100%
:align: center

The same policy in simulation (left of each pair) and on the real AgiBot X2 Ultra (right), at the same moments
of a 43 s routine of chained dance clips from [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed).
```

The policy was trained with SuperMimic on all of BONES-SEED in the deployable environment of stage 1. It runs
at 50 Hz and observes only what the robot measures: joint positions and velocities from the encoders,
projected gravity and angular velocity from the IMU in the pelvis, the previous action and the reference frame
to track. There is no estimate of the base state and no external tracking. The actions are joint position
targets that the drives track with the stiffness and damping of the simulated position servos.

On the X2 Ultra, a policy step takes about 2 ms, on a robot state that is about 2 ms old. The controller runs
on a laptop connected over Ethernet or on the robot's own computer.

## Robot-specific guides

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

If the policy works in simulation but not on the robot, see
[troubleshooting and limitations](troubleshooting.md).
