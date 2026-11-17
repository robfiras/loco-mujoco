.. _real_robot_deployment_tutorial:

Real Robot Deployment
=====================

This tutorial describes the path from a policy trained in LocoMuJoCo to the same policy
running on a real humanoid. The reference implementation is
`LocoMuJoCo Real <https://github.com/robfiras/loco-mujoco-real>`__, an extension package
with environments for the Unitree G1 (``G1Real``) and the AgiBot X2 Ultra
(``X2UltraReal``), a motion-tracking task, and the deployment tools in its
``examples/deployment`` folder. The commands below are run from that repository.

.. note::

    The deployment tools are under active development. The stages and the reasons for them
    are stable; script names and options may still change. The README files in
    ``examples/deployment`` of LocoMuJoCo Real are the authority on the details.

    Status: both pipelines, the Unitree G1 and the AgiBot X2 Ultra, have been run on the
    robot.

.. warning::

    A learned whole-body policy can move a humanoid fast and with full torque. Keep the
    robot on a gantry or crane until every stage below has passed, keep the emergency stop
    in hand, and keep people away from the robot.

The path has six stages. Each one removes a single source of error, so that a failure
points at one cause:

.. list-table::
   :header-rows: 1
   :widths: 6 34 60

   * - #
     - stage
     - what it rules out
   * - 1
     - train in a deployable environment
     - a policy that depends on things the robot does not have
   * - 2
     - export the policy to a bundle
     - a dependency on the training stack at run time
   * - 3
     - validate the bundle
     - a difference between the exported and the trained policy
   * - 4
     - synchronous sim-to-sim
     - errors in the control loop and the actuation model
   * - 5
     - asynchronous sim-to-sim over the robot's middleware
     - timing, latency, and errors in the communication layer
   * - 6
     - the robot, hanging first
     - the remaining gap between the model and the hardware


Stage 1: Train in a Deployable Environment
------------------------------------------

Whether a policy transfers is mostly decided here. Add the robot as described in
:ref:`adding_new_robot_tutorial`, then make the environment resemble the robot in the
following points.

**Actuation.** Real joint drives close a PD loop on a position target. Model the actuators
as MuJoCo *position* actuators with the stiffness and damping the drives will run, so that
the PD loop is part of the physics and runs at the simulation rate, as it does on the
robot. The policy then outputs position targets around a default standing pose:

.. code-block:: text

    target = default_pose + action_scale * action

Set the joint ``armature`` to the reflected rotor inertia of each motor type. Keep the
default pose as a keyframe of the model (``init_state`` in LocoMuJoCo Real), since the
controller on the robot needs the same pose.

**Observations.** The policy may only read what the robot measures:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - available on the robot
     - not available
   * - joint positions and velocities
     - base position in the world
   * - base angular velocity (IMU gyroscope)
     - base linear velocity
   * - projected gravity (from the IMU orientation)
     - absolute heading (IMU yaw is arbitrary and drifts)
   * - the previous action
     - contact forces, body positions in the world
   * - the reference motion
     -

Anything in the right column can still go to the critic. Use
:doc:`observation groups <customizing_environments>` to give the critic privileged
observations and the actor only the left column. For motion tracking this means the goal
observation must be expressed relative to the robot (LocoMuJoCo Real anchors it to the
pelvis) and must not contain a world position.

**Control rate.** Choose ``timestep`` and ``n_substeps`` so that the policy runs at a rate
the robot's computer and network can hold. LocoMuJoCo Real uses 0.005 s × 4, a 50 Hz
policy on a 200 Hz simulation. Use the same two values in the robot conf, so that the
retargeted clips have that frequency.

**Randomization.** Randomize what differs between the model and the robot rather than
everything: pushes, encoder offsets, foot friction, the centre of mass of the torso,
observation noise and, if the latency of the robot is known, an observation delay. See the
domain randomization example in :doc:`customizing_environments`.

**Reference motions.** A clip that is played on the robot should start and end in a
standing pose. The controller then never has to hold, or switch clips in, a pose in
mid-stride.

Training itself is unchanged:

.. code-block:: bash

    python examples/training/train.py

.. code-block:: bash

    python examples/training/train.py env=mjx_x2_ultra_real


Stage 2: Export the Policy to a Bundle
--------------------------------------

The robot cannot call back into the training environment, and its computer should not need
JAX. The checkpoint is therefore exported to a **bundle**, a folder that is run with
``numpy`` and ``mujoco`` only (MuJoCo is used for forward kinematics):

.. code-block:: text

    deploy_bundles/<name>/
    ├── policy.json       joint and actuator order, action scale and offset, kp/kd,
    │                     observation layout, tracked bodies, network shape, solver options
    ├── policy.npz        network weights and the observation normalizer
    └── motions.lmj.npz   the reference clips, as a LocoMuJoCo Trajectory

.. code-block:: bash

    python examples/deployment/export_ppo_bundle.py --checkpoint outputs/<run>/PPOJax_saved.pkl --output deploy_bundles/<name>

The runtime library (``lmj_deploy.py``) rebuilds the actor's observation by hand from a
``MjData``, applies the frozen normalizer and the network, and maps the action to joint
targets. The exporter refuses a checkpoint whose control function or observation terms it
does not implement, rather than exporting something slightly wrong.

.. note::

    Two orderings exist side by side, and mixing them fails without an error. Observations
    and the reference are in MuJoCo **joint** order; actions, gains and ``ctrl`` are in
    **actuator** order. Both lists are stored in ``policy.json``, and the runtime resolves
    every joint and body **by name**. Do the same in a controller of your own.


Stage 3: Validate the Bundle
----------------------------

A hand-written observation is only trustworthy if it is checked. ``validate_bundle.py``
rebuilds the training environment from the checkpoint, feeds both sides the same states,
and compares every observation term, the action and the joint targets:

.. code-block:: bash

    python examples/deployment/validate_bundle.py --bundle deploy_bundles/<name> --checkpoint outputs/<run>/PPOJax_saved.pkl

It must print ``PASS``. The differences are at the level of floating-point error
(``1e-7`` and below). Run it again whenever the observation, the control function or the
exporter changes. Do not look at any rollout of a bundle that has not passed.


Stage 4: Synchronous Sim-to-Sim
-------------------------------

The bundle now runs in plain MuJoCo, without LocoMuJoCo, with policy and physics stepped in
lockstep:

.. code-block:: bash

    python examples/deployment/sim2sim_mujoco.py --bundle deploy_bundles/<name> --viewer

The ``--plant`` option selects the simulated robot:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``--plant``
     - purpose
   * - ``servo``
     - the training model. A failure here is a bug in the pipeline, not a sim-to-real gap.
   * - ``torque``
     - torque motors with the PD loop computed in Python, which is the actuation path of
       the deployment. It should reproduce ``servo``.
   * - a path to an MJCF
     - another model of the same robot, for example the vendor's, with its own joint
       friction, damping and collision geometry. A first robustness test.


Stage 5: Asynchronous Sim-to-Sim over the Robot's Middleware
------------------------------------------------------------

On the robot, the controller and the plant are separate processes: the state the policy
sees is already old, and a late command leaves the previous targets in force. This stage
reproduces that. A **bridge** runs MuJoCo in a process of its own and speaks the robot's
protocol; the **controller** is the program that will later drive the robot, unchanged.

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - robot
     - middleware
     - bridge
     - controller
   * - Unitree G1
     - Unitree SDK 2 (DDS), ``LowState`` / ``LowCmd``
     - ``sim2sim_dds_bridge.py``
     - ``deploy_g1.py``
   * - AgiBot X2 Ultra
     - ROS 2 Humble, AimDK joint and IMU topics
     - ``sim2sim_aimdk_bridge.py``
     - ``deploy_x2.py``

In two terminals:

.. code-block:: bash

    python examples/deployment/sim2sim_dds_bridge.py --bundle deploy_bundles/<name> --viewer

.. code-block:: bash

    python examples/deployment/deploy_g1.py --bundle deploy_bundles/<name> --motion <clip>

The simulated robot hangs from an elastic band, like the real one from its crane, and the
start goes as it will on the robot: the controller ramps to the first pose of the clip and
holds it, the robot is lowered onto its feet, the policy is started, and the band is
released. The viewer draws the reference as a ghost.

Both sides use a DDS or ROS domain other than the robot's, so that the rehearsal cannot
reach a robot on the same network.

How the controller is built matters as much as the policy:

- **One observation path.** The incoming robot state is written into a ``MjData``, and the
  same observation builder as in stages 3 and 4 runs on it. There is no second
  implementation for the robot.
- **Poll the newest state** instead of handling every message in a callback. At 500 Hz, a
  Python callback takes the interpreter away from the control loop.
- **Heading is relative.** When the policy starts, rotate the reference onto the robot's
  current heading. IMU yaw then only has to hold for the length of a clip.
- **Limit the BLAS threads** before ``numpy`` is imported (for example
  ``OPENBLAS_NUM_THREADS=4``). An unlimited thread pool can delay single policy steps by
  tens of milliseconds as soon as anything else needs a core.
- **Watch the timing.** Print the control rate, the compute time per step and the age of
  the state while the policy runs.


Stage 6: On the Robot
---------------------

**Safety checks in the controller.** The controller commands nothing if the first state it
receives looks wrong, and afterwards damps the joints (zero stiffness, some damping) on
any of the following:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - trigger
     - limit used in LocoMuJoCo Real
   * - stop button on the remote or keyboard
     -
   * - no new state from the robot
     - 100 ms
   * - a joint beyond its range
     - 0.1 rad past the model's range
   * - joint speed
     - 25 rad/s
   * - tilt of the pelvis
     - about 60 degrees
   * - invalid IMU reading, non-finite observation or action
     -
   * - ``Ctrl-C`` or an exception
     -

Damping lets the robot fall, which is why it hangs. A softer second level is useful as
well: the policy keeps running but tracks the standing pose instead of the clip.

**The procedure.** In this order, the first time with every new policy or motion:

1. The robot **hangs**, feet clear of the floor or just touching. Emergency stop in hand.
2. **Read-only checks**: the status fields of the robot are clean, projected gravity points
   down, and a live view of the measured state moves like the robot when a joint is moved
   by hand. This catches a wrong joint mapping or sign before anything is commanded.
3. **Prove the link**: swing one or two joints by a small amplitude with soft gains.
4. **Ramp and hold, without the policy**: ramp slowly (0.5 rad/s at most) to the first pose
   of the clip and hold it with the policy's gains.
5. **The policy, hanging.**
6. **The policy on the floor**: start it balancing on a standing pose, slacken the crane,
   then play the motion.

Before the controller can command the joints, the vendor's own controller has to release
them. How that is done, the network setup and the required software differ per robot:

- **AgiBot X2 Ultra**: ``examples/deployment/X2_ROBOT_GUIDE.md`` in LocoMuJoCo Real covers
  the network, ROS 2 and ``aimdk_msgs``, the switch to the developer mode, the topics, the
  gains and measurements taken on the robot.
- **Unitree G1**: ``examples/deployment/README.md`` covers the Unitree SDK setup and the
  controller.


Deploying on Another Robot
--------------------------

Stages 1 to 4 do not depend on the robot's middleware. A new robot needs:

1. an environment and a robot conf (:ref:`adding_new_robot_tutorial`), with position
   actuators, armature and a default-pose keyframe,
2. a small transport module that reads the robot's state into a ``MjData`` and writes
   joint targets with gains (compare ``lmj_dds.py`` and ``lmj_aimdk.py``),
3. a bridge that makes MuJoCo speak the same protocol, and
4. a controller with the start-up sequence, the safety checks and the operator input.

The observation builder, the actor and the bundle format are shared.


When It Works in Simulation but Not on the Robot
------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - symptom
     - where to look
   * - the robot shakes or hums while standing
     - gains or armature differ from the drives; control latency. Compare a small sine on
       a single joint between the robot and the model.
   * - the policy fails in the first step
     - joint order or sign, a normalizer input that was constant in training, a stale
       first state.
   * - the robot drifts or turns away from the reference
     - heading alignment at the start; the policy reads a world-frame quantity.
   * - joints lag behind the model
     - friction in the real joints that the model lacks. Test with a plant that has
       ``frictionloss`` in stage 4.
   * - single late control steps
     - BLAS threads, rendering in the control process, message callbacks.
   * - a fall when a clip ends or is switched
     - the clip does not start or end standing.
