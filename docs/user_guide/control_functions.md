# Control functions

The control function sits between the policy and the actuators of the model. It defines the action space
that the agent sees and converts every action into the control signal that is written to MuJoCo. Changing
it switches a robot from torque control to position control without touching the model, the policy or the
training loop.

```{literalinclude} ../../examples/tutorials/07_changing_control_type.py
:language: python
```

The control function is selected with `control_type` and configured with `control_params`.

## Built-in control functions

| `control_type` | Action means | Called | Parameters |
|---|---|---|---|
| `DefaultControl` | Normalized actuator control. For the motors of the robots this is a torque, for the muscle models a muscle excitation. | Once per environment step | none |
| `PDControl` | Target joint position for a PD controller that computes the torque | Before every physics step | `p_gain`, `d_gain`, `nominal_joint_positions`, `scale_action_to_jnt_limits` |

Both control functions expose an action space of $[-1, 1]$ per actuator, so the bounds reported in
`env.info.action_space` do not depend on the robot.

### DefaultControl

{class}`~loco_mujoco.core.control_functions.default.DefaultControl` passes the action to the actuators that
are defined in the XML of the robot, after rescaling it from $[-1, 1]$ to the control range of each
actuator:

$$
u = a \cdot \frac{u_\text{max} - u_\text{min}}{2} + \frac{u_\text{max} + u_\text{min}}{2}
$$

Here $u_\text{min}$ and $u_\text{max}$ are the `ctrlrange` of the actuator, so the rescaling needs actuators
with a control range. All robots in LocoMuJoCo define one. Actions outside $[-1, 1]$ are not clipped by the
control function. MuJoCo clamps the control to the control range of the actuator.

### PDControl

{class}`~loco_mujoco.core.control_functions.pd.PDControl` interprets the action as a target joint position
$q^*$ and computes the control from the position error and the joint velocity:

$$
u = \text{clip}\big(k_p\,(q^* - q) - k_d\,\dot q,\; u_\text{min},\; u_\text{max}\big)
$$

| Parameter | Default | Meaning |
|---|---|---|
| `p_gain` | required | Proportional gain $k_p$. A scalar, or an array with one entry per action. |
| `d_gain` | required | Derivative gain $k_d$. A scalar, or an array with one entry per action. |
| `nominal_joint_positions` | `None` | Nominal positions $q_\text{nom}$ of the actuated joints. `None` uses the default pose of the model (`qpos0`). |
| `scale_action_to_jnt_limits` | `True` | How the action is mapped to the target position, see below |

The target position depends on `scale_action_to_jnt_limits`:

| Value | Target position $q^*$ | Effect |
|---|---|---|
| `True` | $q_\text{low} + \tfrac{1}{2}(a + 1)\,(q_\text{high} - q_\text{low})$ | The action range $[-1, 1]$ covers the whole joint range. An action of 0 is the middle of the range, independent of the nominal positions. |
| `False` | $q_\text{nom} + a$ | The action is an offset from the nominal position in radians (or meters for slide joints). An action of 0 holds the nominal pose. |

In both cases the target is clipped to the joint range before the torque is computed.

The PD controller works on the torque motors of the model: it reads the joint that each actuator drives
and writes a torque. It supports hinge and slide joints only. It runs at the simulation frequency, so the
torque is recomputed before every physics step from the current joint state while the target stays fixed
for the whole environment step. With the default time step of 1 ms and 10 substeps this is a 1 kHz PD loop
under a 100 Hz policy.

```python
env, _ = RLFactory.make("UnitreeH1", control_type="PDControl",
                        control_params=dict(p_gain=100.0, d_gain=1.0, scale_action_to_jnt_limits=False))
```

The state of the controller (`carry.control_func_state`) holds a noise term for each gain, a position
offset and a control multiplier. The [default domain randomizer](domain_randomization.md) uses it to perturb
the gains of every episode.

## Defaults of the robots

Each robot chooses its default control function. Robots that default to `PDControl` come with their own
gains. All of them except AnymalC use `scale_action_to_jnt_limits=False`, so their actions are offsets from
a nominal pose.

| Default | Robots |
|---|---|
| `DefaultControl` | UnitreeH1, UnitreeG1, Talos, ToddlerBot, FourierGR1T2, SkeletonTorque, SkeletonMuscle, MyoSkeleton, UnitreeA1, UnitreeGo2 |
| `PDControl` with the gains of the robot | UnitreeH1v2, Apollo, Atlas, BoosterT1, BDSpot, AnymalC |

Passing `control_type` replaces the default of the robot. If you pass `control_type="PDControl"`, you also
have to pass the gains in `control_params`. To get torque control on a robot that defaults to PD control,
pass `control_type="DefaultControl"`.

## Action space and timing

The dimension of the action is the number of actuators in the `actuation_spec` of the environment, a list
of actuator names from the XML. Every robot has a default list with all of its actuators, and the
constructor argument `actuation_spec` selects a subset.

```python
env.info.action_space.low, env.info.action_space.high    # bounds, -1 and 1 for the built-in control functions
action = env.sample_action_space(key)                    # uniform sample, needs a JAX random key
```

One call of `step` advances the simulation by `n_substeps` physics steps of length `timestep`. The defaults
of {class}`~loco_mujoco.environments.base.LocoEnv` are `n_substeps=10` and `timestep=0.001`, which gives a
control time step `env.dt` of 0.01 s. A control function decides through its property
`run_with_simulation_frequency` whether it is called once per environment step (`DefaultControl`) or before
every physics step (`PDControl`). The control time step is the same in both cases.

The action that the control function receives has already passed through the
[domain randomizer](domain_randomization.md), which may modify it.

## See also

- [Modular architecture](modular_architecture.md) for the place of the control function in the step.
- [Domain randomization](domain_randomization.md) for randomized PD gains.
- [Writing custom modules](custom_modules.md#control-functions) for your own control function.
- [Sim-to-real](../sim2real/index.md) for the controllers used on the real robots.
- API: {mod}`loco_mujoco.core.control_functions.base`, {mod}`loco_mujoco.core.control_functions.default`,
  {mod}`loco_mujoco.core.control_functions.pd`.
