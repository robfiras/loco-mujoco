# Train for deployment

<ul class="lmj-steps">
  <li><a href="setup.html">Setup</a></li>
  <li class="current">1 · Train</li>
  <li><a href="export.html">2–3 · Export and validate</a></li>
  <li><a href="sim2sim.html">4–5 · Sim-to-sim</a></li>
  <li><a href="robot.html">6 · On the robot</a></li>
</ul>

Whether a policy transfers is mostly decided in this first stage. Add the robot as described in
[adding a new robot](../user_guide/new_robot.md), then make the environment resemble the robot in the
following points.

## Actuation

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

## Observations

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

## Control rate

Choose `timestep` and `n_substeps` so that the policy runs at a rate the robot's computer and network can
hold. LocoMuJoCo Real uses 0.005 s × 4, a 50 Hz policy on a 200 Hz simulation. Use the same two values in the
robot conf, so that the retargeted clips have that frequency.

## Randomization

Randomize what differs between the model and the robot rather than everything: pushes, encoder offsets, foot
friction, the centre of mass of the torso, observation noise and, if the latency of the robot is known, an
observation delay. See [domain randomization](../user_guide/domain_randomization.md).

LocoMuJoCo Real randomizes sensing and disturbances and leaves the simulated plant (masses, inertias, joint
damping, gains) unchanged. Its randomizer, `G1RealRandomizer`, is the same for both robots
(`training/<robot>/env/randomization/sim2real.yaml`):

| What | How | Range |
|---|---|---|
| Pushes | A velocity impulse on the root, every 1 to 3 s | ±0.5 m/s in x and y, ±0.2 m/s in z; ±0.52 rad/s in roll and pitch, ±0.78 rad/s in yaw |
| Encoder bias | A constant offset on the measured joint positions, drawn per episode | ±0.02 rad |
| Observation noise | Uniform noise on the measured joint positions, joint velocities and base angular velocity | ±0.01 rad, ±0.5 rad/s, ±0.2 rad/s |
| Friction | One friction coefficient for all colliding geometries of the robot, drawn per episode | 0.3 to 1.2 |
| Centre of mass | An offset of the pelvis centre of mass, per axis | ±2.5 cm in x, ±5 cm in y and z |

Noise and bias are applied to the actor's observations only. The critic receives the clean values, and
projected gravity is not perturbed. With these settings, SuperMimic tracks all 40 LAFAN1 clips with one policy
per robot, see the [benchmark results](../training/benchmark.md#supermimic-on-the-real-robot-environments).

## Reference motions

A clip that is played on the robot should start and end in a standing pose. The controller then never has to
hold, or switch clips in, a pose in mid-stride.

## Run the training

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

Next: [export and validate the policy](export.md).
