# Export and validate the policy

<ul class="lmj-steps">
  <li><a href="setup.html">Setup</a></li>
  <li><a href="training.html">1 · Train</a></li>
  <li class="current">2–3 · Export and validate</li>
  <li><a href="sim2sim.html">4–5 · Sim-to-sim</a></li>
  <li><a href="robot.html">6 · On the robot</a></li>
</ul>

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

Next: [sim-to-sim](sim2sim.md).
