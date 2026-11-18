# Your first environment

<ul class="lmj-steps">
  <li><a href="installation.html">1 · Installation</a></li>
  <li class="current">2 · First environment</li>
  <li><a href="replay_motion.html">3 · Replay a motion</a></li>
  <li><a href="train_policy.html">4 · Train a policy</a></li>
  <li><a href="next_steps.html">5 · Evaluate and customize</a></li>
</ul>

On this page you create a Unitree G1 humanoid, look at its observation and action spaces, and step it with
random actions. Everything runs on the CPU.

```{image} ../../imgs/unitree_g1_1.gif
:alt: The Unitree G1 environment
:width: 45%
:align: center
```

## Create an environment

Environments are created through a *task factory*. {class}`~loco_mujoco.task_factories.rl_factory.RLFactory`
creates a plain reinforcement learning task; {class}`~loco_mujoco.task_factories.imitation_factory.ImitationFactory`,
which you will use on the next page, also loads motion capture data.

```python
from loco_mujoco.task_factories import RLFactory

env, traj = RLFactory.make("UnitreeG1")
```

Both factories return the environment and its trajectory. A reinforcement learning task has no trajectory,
so `traj` is `None` here. The first call downloads the robot models, which takes a moment.

When the environment is created, a summary of its observation space is printed to the terminal. Pass
`print_obs_summary=False` to turn that off.

## Look at the spaces

```python
print(env.info.observation_space.shape)  # (60,)
print(env.info.action_space.shape)       # (23,)
print(env.info.action_space.low[0], env.info.action_space.high[0])  # -1.0 1.0
print(env.dt)                            # 0.01, the control time step in seconds
```

The G1 has 23 actuators. Actions are normalized to `[-1, 1]` and mapped to the actuator ranges of the
model by the environment's [control function](../user_guide/control_functions.md), which is direct torque
control by default.

The 60-dimensional observation contains the pose and velocity of the floating base, without its horizontal
position, followed by the joint positions and the joint velocities (57 entries). The last three entries are
the goal that `RLFactory` adds: a target velocity for the root. To see exactly which entry is what:

```python
env.print_observation_summary()
```

## Step the environment

The CPU interface follows the familiar reset and step loop:

```python
import numpy as np
from loco_mujoco.task_factories import RLFactory

env, traj = RLFactory.make("UnitreeG1")
action_dim = env.info.action_space.shape[0]

obs = env.reset()
env.render()

for i in range(1000):
    action = np.random.randn(action_dim)
    obs, reward, absorbing, done, info = env.step(action)
    env.render()
    if absorbing or done:
        obs = env.reset()
```

A viewer window opens and the robot flails and falls over, as expected with random actions.

`step` returns five values:

| Value | Meaning |
|---|---|
| `obs` | The next observation, a NumPy array. |
| `reward` | The reward of the transition. `RLFactory` uses a reward for tracking a target root velocity by default. |
| `absorbing` | `True` if a terminal state was reached, for example because the robot fell. |
| `done` | `True` if the episode is over, either because of a terminal state or because the horizon of 1000 steps was reached. |
| `info` | A dictionary with additional information. |

```{tip}
On a machine without a display, replace `env.render()` with `env.render_viser()` to watch the robot in a
browser. It needs the `viser` extra, see [rendering](../user_guide/rendering.md).
```

## Pick another embodiment

Every embodiment is created the same way. Replace the name to try another one:

```python
env, traj = RLFactory.make("UnitreeH1")       # another humanoid robot
env, traj = RLFactory.make("SkeletonTorque")  # a torque-actuated human skeleton
env, traj = RLFactory.make("UnitreeGo2")      # a quadruped
```

The [environment gallery](../environments/index.md) lists all of them. You can also get the names from Python:

```python
import loco_mujoco
print(sorted(loco_mujoco.get_registered_envs()))
```

## Run thousands of environments on the GPU

Every environment has an MJX counterpart whose name starts with `Mjx`. It exposes a functional interface,
`mjx_reset` and `mjx_step`, that can be vectorized with `jax.vmap` and compiled with `jax.jit`:

```python
import jax
from loco_mujoco.task_factories import RLFactory

env, traj = RLFactory.make("MjxUnitreeG1")

n_envs = 100
keys = jax.random.split(jax.random.key(0), n_envs + 1)
key, env_keys = keys[0], keys[1:]

reset = jax.jit(jax.vmap(env.mjx_reset, in_axes=(0, None)))
step = jax.jit(jax.vmap(env.mjx_step, in_axes=(0, 0, None)))
sample_action = jax.jit(jax.vmap(env.sample_action_space))

state = reset(env_keys, traj)
for i in range(1000):
    keys = jax.random.split(key, n_envs + 1)
    key, action_keys = keys[0], keys[1:]
    action = sample_action(action_keys)
    state = step(state, action, traj)
    env.mjx_render(state)
```

The environment state, including the observation, reward and termination flags, is carried in `state`.
This is the interface the training algorithms are built on. [Simulation backends](../user_guide/backends.md)
explains it in detail, including how to switch to MuJoCo Warp.

## What you learned

- Environments are created with a task factory, which returns `(env, traj)`.
- `env.info` describes the observation and action spaces.
- CPU environments use `reset` and `step`; MJX environments use the functional `mjx_reset` and `mjx_step`.

Next: [replay a motion](replay_motion.md).
