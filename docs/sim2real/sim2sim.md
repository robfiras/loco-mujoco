# Sim-to-sim

<ul class="lmj-steps">
  <li><a href="setup.html">Setup</a></li>
  <li><a href="training.html">1 · Train</a></li>
  <li><a href="export.html">2–3 · Export and validate</a></li>
  <li class="current">4–5 · Sim-to-sim</li>
  <li><a href="robot.html">6 · On the robot</a></li>
</ul>

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

Next: [on the robot](robot.md).
