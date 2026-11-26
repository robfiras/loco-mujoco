# Evaluate and customize

<ul class="lmj-steps">
  <li><a href="installation.html">1 · Installation</a></li>
  <li><a href="first_environment.html">2 · First environment</a></li>
  <li><a href="replay_motion.html">3 · Replay a motion</a></li>
  <li><a href="train_policy.html">4 · Train a policy</a></li>
  <li class="current">5 · Evaluate and customize</li>
</ul>

You have a trained agent. On this page you run it, change parts of the environment it was trained in, and
see where to go from here.

## Run the trained policy

The training run saved the agent as `PPOJax_saved.pkl`. The evaluation script next to the training script
loads it, rebuilds the environment from the stored configuration and renders the policy:

```bash
cd examples/training_examples/jax_rl_mimic
python eval.py --path outputs/<date>/<time>/PPOJax_saved.pkl
```

The policy was trained in MJX. To check how it behaves in standard MuJoCo, add `--use_mujoco`:

```bash
python eval.py --path outputs/<date>/<time>/PPOJax_saved.pkl --use_mujoco
```

```{note}
MJX and MuJoCo are not numerically identical, so a policy can be less robust after the switch.
[Domain randomization](../user_guide/domain_randomization.md) during training closes most of that gap, and it
is essential before going to a real robot.
```

The same can be done from Python:

```python
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax

agent_conf, agent_state = PPOJax.load_agent("outputs/<date>/<time>/PPOJax_saved.pkl")
config = agent_conf.config

factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
env, traj = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

PPOJax.play_policy(env, agent_conf, agent_state, n_envs=1, n_steps=1000,
                   deterministic=True, record=True, traj=traj)
```

With `record=True` a video is written to `LocoMuJoCo_recordings/`.

## Customize the environment

An environment is assembled from interchangeable modules. Each one is selected by name with a `*_type`
argument and configured with a `*_params` dictionary, either in `ImitationFactory.make` or under
`env_params` in a `conf.yaml`. Three common changes:

::::{tab-set}

:::{tab-item} Position control
Let the policy output joint position targets for a PD controller instead of torques:

```python
env, traj = ImitationFactory.make(
    "UnitreeG1",
    default_dataset_conf=DefaultDatasetConf(["walk"]),
    control_type="PDControl",
    control_params=dict(p_gain=100.0, d_gain=2.0),
)
```

See [control functions](../user_guide/control_functions.md).
:::

:::{tab-item} Rough terrain
Train on a random uneven surface that is regenerated on every reset:

```python
env, traj = ImitationFactory.make(
    "UnitreeG1",
    default_dataset_conf=DefaultDatasetConf(["walk"]),
    terrain_type="RoughTerrain",
    terrain_params=dict(random_min_height=-0.05, random_max_height=0.05),
)
```

See [terrain](../user_guide/terrain.md).
:::

:::{tab-item} Domain randomization
Randomize physical parameters such as gravity, friction, masses and joint properties in every episode. The
default randomizer takes one dictionary that switches each randomization on or off and sets its range:

```python
randomization_config = {
    "randomize_gravity": True,
    "gravity_range": [9.51, 10.11],
    "randomize_geom_friction_tangential": True,
    "geom_friction_tangential_range": [0.8, 1.2],
    # ... one switch and one range per quantity
}

env, traj = ImitationFactory.make(
    "UnitreeG1",
    default_dataset_conf=DefaultDatasetConf(["walk"]),
    domain_randomization_type="DefaultRandomizer",
    domain_randomization_params=randomization_config,
)
```

The dictionary must contain every entry. Copy the complete one from
`examples/tutorials/08_domain_randomization.py`; [domain randomization](../user_guide/domain_randomization.md)
explains each entry.
:::
::::

The modules that can be swapped are:

| Module | Decides | Guide |
|---|---|---|
| Observations and goals | What the policy sees | [Observations](../user_guide/observations.md) |
| Reward | What is rewarded | [Rewards](../user_guide/rewards.md) |
| Control function | How actions become actuator commands | [Control functions](../user_guide/control_functions.md) |
| Initial state handler | Where an episode starts | [Initial and terminal states](../user_guide/initial_terminal_states.md) |
| Terminal state handler | When an episode ends | [Initial and terminal states](../user_guide/initial_terminal_states.md) |
| Domain randomizer | What varies between episodes | [Domain randomization](../user_guide/domain_randomization.md) |
| Terrain | What the robot stands on | [Terrain](../user_guide/terrain.md) |

If none of the built-in options fits, [write your own module](../user_guide/custom_modules.md) and register
it. It then becomes available by name like the built-in ones.

## Where to go next

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Understand the pieces
:link: ../user_guide/index
:link-type: doc

The user guide explains backends, task factories, trajectories and every module.
:::

:::{grid-item-card} Try another algorithm
:link: ../training/index
:link-type: doc

AMP and GAIL learn a style from unlabeled motion. SuperMimic scales tracking to many clips.
:::

:::{grid-item-card} Use more motion data
:link: ../user_guide/datasets
:link-type: doc

Browse the available clips, set up AMASS or bring your own trajectories.
:::

:::{grid-item-card} Pick another embodiment
:link: ../environments/index
:link-type: doc

Sixteen humanoids, human models and quadrupeds with their observation and action spaces.
:::

:::{grid-item-card} Go to the real robot
:link: ../sim2real/index
:link-type: doc

Deployable training setups and tools for the Unitree G1 and the AgiBot X2 Ultra.
:::

:::{grid-item-card} Add your own robot
:link: ../user_guide/new_robot
:link-type: doc

Add an embodiment and retarget the existing datasets to it.
:::
::::
