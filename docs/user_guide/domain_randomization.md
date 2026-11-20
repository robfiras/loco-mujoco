# Domain randomization

Domain randomization varies the simulation between episodes so that a policy does not overfit to one set of
physical parameters. It is the standard tool for transferring a policy from simulation to a real robot. In
LocoMuJoCo the randomization is a module of the environment: it is selected with
`domain_randomization_type` and configured with `domain_randomization_params`.

```python
env, traj = ImitationFactory.make(
    "ToddlerBot",
    default_dataset_conf=dict(task="walk"),
    domain_randomization_type="DefaultRandomizer",
    domain_randomization_params=randomization_config,   # a dictionary, see below
)
```

| `domain_randomization_type` | Effect |
|---|---|
| `NoDomainRandomization` | Nothing is randomized. The default. |
| `DefaultRandomizer` | Randomizes gravity, contact and joint properties, masses and PD gains at every reset, and adds noise to observations at every step |

## What a randomizer can change

A domain randomizer derives from {class}`~loco_mujoco.core.domain_randomizer.base.DomainRandomizer`. The
environment calls it at four points, which together give access to everything that enters or leaves the
simulation:

| Method | Called | Can change |
|---|---|---|
| `reset` | At every reset, after the initial state has been set | The simulation data and the carry. This is where new parameters are sampled and stored in `carry.domain_randomizer_state`. |
| `update` | Before the physics of every step | The **model** and the **data**, for example masses, friction, gravity, or an external push |
| `update_observation` | After the observation of every step has been created | The **observation**, for example sensor noise |
| `update_action` | At the beginning of every step, before the control function | The **action**, for example noise or a delay |

The sampled parameters are kept in the carry and not in the randomizer object, so every parallel
environment has its own set, see [Modular architecture](modular_architecture.md#stateless-modules-and-the-carry).

## The default randomizer

{class}`~loco_mujoco.core.domain_randomizer.default.DefaultRandomizer` covers the parameters that are
usually randomized for legged robots. Every group of parameters has a switch, and a switch that is off
leaves the model untouched.

```{literalinclude} ../../examples/tutorials/08_domain_randomization.py
:language: python
```

```{important}
The randomizer reads its configuration without defaults. Pass all keys of the tables below, with the
switches you do not need set to `False`. A missing key raises a `KeyError` at the first reset. The four keys
for the PD gains are only read when the environment uses `PDControl`. Keys that the randomizer does not know
are ignored, which in the example above applies to the two `joint_stiffness` entries.
```

### Model parameters

These parameters are sampled uniformly from their range at every reset and stay fixed for the episode.

| Switch | Range key | Effect |
|---|---|---|
| `randomize_gravity` | `gravity_range` | Magnitude of gravity in m/s². Gravity keeps pointing along the negative z-axis. |
| `randomize_geom_friction_tangential` | `geom_friction_tangential_range` | Sliding friction coefficient of every geom (first entry of `geom_friction`) |
| `randomize_geom_friction_torsional` | `geom_friction_torsional_range` | Torsional friction coefficient of every geom |
| `randomize_geom_friction_rolling` | `geom_friction_rolling_range` | Rolling friction coefficient of every geom |
| `randomize_geom_stiffness` | `geom_stiffness_range` | Contact stiffness of every geom |
| `randomize_geom_damping` | `geom_damping_range` | Contact damping of every geom |
| `randomize_joint_friction_loss` | `joint_friction_loss_range` | Dry friction of every joint (`dof_frictionloss`) |
| `randomize_joint_damping` | `joint_damping_range` | Damping of every joint (`dof_damping`) |
| `randomize_joint_armature` | `joint_armature_range` | Armature (reflected rotor inertia) of every joint (`dof_armature`) |
| `randomize_base_mass` | `base_mass_to_add_range` | Mass in kg that is added to the root body |
| `randomize_link_mass` | `link_mass_multiplier_range` | Factor on the mass of every body. A dictionary with one range for `root_body` and one for `other_bodies`. |
| `randomize_com_displacement` | `com_displacement_range` | Shift of the center of mass of the root body in meters, sampled per axis |

All ranges are a pair `[low, high]`. Notes on the units and the sampling:

- The friction, joint and gravity ranges are **absolute values** that replace the value in the model. Only
  the link masses are multiplied, and the base mass and the center of mass are shifted.
- The values are sampled independently per geom, per joint and per body. The three friction coefficients of
  one geom share one random number, so they are high or low together.
- Contact stiffness and damping are written in MuJoCo's direct format of `solref`, that is, as
  `(-stiffness, -damping)`. They belong together: enable both switches or neither.
- The joint parameters are applied to all degrees of freedom after the first six, which are the free joint
  of the root.
- The root body is the body named by the info property `root_body_name` of the robot.

### PD gains

These parameters only apply to environments with the [control function](control_functions.md) `PDControl`.

| Switch | Scale key | Effect |
|---|---|---|
| `add_p_gains_noise` | `p_gains_noise_scale` | Adds `u * scale * p_gain` to the proportional gain, with `u` uniform in $[-1, 1]$ |
| `add_d_gains_noise` | `d_gains_noise_scale` | The same for the derivative gain |

A scale of 0.1 therefore varies a gain by up to 10 percent. The noise is sampled at every reset, per
actuator if the gains were given as an array, and stored in the state of the control function.

### Observation noise

Observation noise is zero-mean Gaussian noise with the standard deviation given by the scale key. It is
sampled anew at every step.

| Switch | Scale key | Applies to observations of type |
|---|---|---|
| `add_joint_pos_noise` | `joint_pos_noise_scale` | `JointPos` |
| `add_joint_vel_noise` | `joint_vel_noise_scale` | `JointVel` |
| `add_gravity_noise` | `gravity_noise_scale` | `ProjectedGravityVector` |
| `add_free_joint_lin_vel_noise` | `lin_vel_noise_scale` | `FreeJointVel`, linear part |
| `add_free_joint_ang_vel_noise` | `ang_vel_noise_scale` | `FreeJointVel`, angular part |

Noise is only added to entries that were created with `allow_randomization=True`, which is the default.
This is how the example above gives a critic clean observations and the policy noisy ones: every quantity
appears twice in the `observation_spec`, once per [group](observations.md#observation-groups), and only the
copy in the `policy` group allows noise. Observation types that are not in the table are never changed.

```{note}
The split of `FreeJointVel` into a linear and an angular part is made over all `FreeJointVel` entries of the
observation at once: the first three values of the first entry count as linear velocity, and all further
values receive the angular velocity noise. With more than one `FreeJointVel` entry in the observation, put
the one that should get linear velocity noise first.
```

The default randomizer does not change the action. `update_action` returns it as it is.

## Differences between the backends

The configuration and the sampled distributions are the same on both backends. The mechanics differ:

| | MuJoCo (CPU) | MJX |
|---|---|---|
| Random numbers | Global NumPy random state. Seed it with `np.random.seed`. | The random key in the carry of each environment, which is split at every draw |
| Sampled parameters | One set, for the single environment | One set per parallel environment |
| Applying them to the model | The `MjModel` of the environment is modified in place before every step. The original values are saved at the first reset, so the randomization does not accumulate. | The model is not stored per environment. Before every step a modified copy of the shared model is built from the state in the carry, inside `jax.vmap`. |

On MJX this means that thousands of environments with different masses and friction coefficients run in
one batched simulation without any change to the training code.

## Giving the critic the sampled parameters

The sampled parameters are available as `state.additional_carry.domain_randomizer_state` (MJX), with the
fields `gravity`, `geom_friction`, `geom_stiffness`, `geom_damping`, `base_mass_to_add`, `com_displacement`,
`link_mass_multipliers`, `joint_friction_loss`, `joint_damping` and `joint_armature`. To feed the randomized
model to a privileged critic through the observation instead, add a
[`ModelInfo`](observations.md#built-in-observation-types) entry for the attributes of interest:

```python
ObservationType.ModelInfo("masses", model_attributes="body_mass", group="critic", allow_randomization=False)
```

## See also

- [Observations](observations.md) for groups and `allow_randomization`.
- [Control functions](control_functions.md) for the PD controller whose gains are randomized.
- [Writing custom modules](custom_modules.md#domain-randomizers) for your own randomizer.
- [Sim-to-real](../sim2real/training.md#randomization) for the randomization used for the real-robot policies.
- API: {mod}`loco_mujoco.core.domain_randomizer.base`, {mod}`loco_mujoco.core.domain_randomizer.default`.
